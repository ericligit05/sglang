"""Grouped NVFP4 W4A16 routed-expert GEMMs for Hopper (sm_90a), with the top-k weighted sum.

Weights come from ``sglang.kernels.ops.quantization.nvfp4_sm90_repack``; the kernels are in
``jit/csrc/moe/nvfp4_w4a16_sm90.cuh``. The route plan is computed on the device, so a call captures in a CUDA graph
and replays with any routing. The decode multiplies one E2M1 code pair with a single IMAD, a trick from onnxruntime's
FP4 MoE GEMV (moe_gemv_fp4.cu, Copyright (c) Microsoft Corporation, MIT License).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.wgmma_rs_hazard import (
    CHECK_VERSION,
    require_no_rs_wgmma_hazard,
)
from sglang.kernels.ops.quantization.nvfp4_sm90_repack import Nvfp4Sm90Experts

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_VARIANTS = {"auto": 0, "single": 1, "pair": 2}
# "auto" runs a decode kernel up to this many tokens; at 3 and 4 tokens the grouped kernels stay.
DECODE_MAX_TOKENS = 2
# Per device and stream: the one-launch decode kernel's workspace (its call epoch, then its tagged intermediate).
_fused_ws: dict = {}


@cache_once
def _module(debug_delay_ns: int = 0) -> Module:
    # debug_delay_ns > 0 only in tests: it stretches the kernels' global-memory hand-offs so that a missing wait or a
    # reused buffer fails every time instead of rarely.
    return load_jit(
        "nvfp4_w4a16_sm90",
        str(debug_delay_ns),
        f"rs-check-{CHECK_VERSION}",
        cuda_files=["moe/nvfp4_w4a16_sm90.cuh"],
        cuda_wrappers=[
            ("run", "nvfp4_w4a16_sm90::run"),
            ("run_decode_fused", "nvfp4_w4a16_sm90::run_decode_fused"),
            ("run_decode_pdl", "nvfp4_w4a16_sm90::run_decode_pdl"),
        ],
        extra_cuda_cflags=[f"-DNVFP4_MOE_DEBUG_DELAY_NS={debug_delay_ns}"]
        if debug_delay_ns
        else [],
        post_build=require_no_rs_wgmma_hazard,
    )


def _fused_workspace(dev: torch.device, words: int) -> torch.Tensor:
    """The one-launch decode kernel's epoch and tagged intermediate for this device and stream. A word is current only
    when it carries the call's epoch, so epoch and words start zeroed together and persist across calls. The buffer
    only grows (a new one is all zero, epoch included); old buffers stay alive for CUDA graphs captured with them."""
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    ws = _fused_ws.get(key)
    if ws is None or ws[0].numel() < words:
        ws = (torch.zeros(words, dtype=torch.int64, device=dev), ws)
        _fused_ws[key] = ws
    return ws[0]


def _decode_variant(T: int, R: int, E: int, H: int, I: int) -> str:
    """The kernel "auto" takes, by measured speed on H100 (1 and 2 tokens, five model shapes): the one-launch decode
    kernel for large experts (I >= 1024), whose tagged exchange hides behind a long weight stream, the two-launch one
    for small experts, and the grouped kernels past DECODE_MAX_TOKENS."""
    if not (0 < T <= DECODE_MAX_TOKENS and R <= 256 and E <= 1024 and H % 256 == 0 and I % 128 == 0):
        return "auto"
    return "decode_fused" if I >= 1024 else "decode_pdl"


def nvfp4_w4a16_sm90_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: Nvfp4Sm90Experts,
    topk_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    variant: str = "auto",
    ctas: int = 0,
    debug_delay_ns: int = 0,
) -> torch.Tensor:
    """With topk_weights: ``out[t] = scale * sum_k topk_weights[t, k] * expert(topk_ids[t, k], x[t])`` over the
    local routes (ids in [0, E); other ranks' routes are -1 or any id outside it), into ``out`` [T, H] when given.
    Without: the per-route outputs [T * top_k, H] (row t * top_k + k, zeros for non-local routes), into ``out`` when
    given.
    x is BF16 [T, H] in natural column order, topk_ids int32 [T, top_k]. variant ("auto", "single", "pair",
    "decode_fused" or "decode_pdl") and ctas pin the kernel for tests; "auto" takes a decode kernel up to
    DECODE_MAX_TOKENS tokens."""
    T, H = x.shape
    R = T * topk_ids.shape[1]
    E, I = experts.gs13.shape[0], experts.intermediate
    dev = x.device
    if variant == "auto":
        variant = _decode_variant(T, R, E, H, I)
    if variant in ("decode_fused", "decode_pdl"):
        if topk_weights is None:
            y = out if out is not None else torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            combined = None
        else:
            y = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            combined = out if out is not None else torch.empty((T, H), dtype=torch.bfloat16, device=dev)
        args = (x, topk_ids, experts.w13, experts.w2, experts.gs13, experts.gs2)
        if variant == "decode_fused":
            ws = _fused_workspace(dev, 2 + R * I // 2)
            _module(debug_delay_ns).run_decode_fused(*args, ws, y, topk_weights, combined, float(scale), ctas)
        else:
            inter = torch.empty((R, I), dtype=torch.bfloat16, device=dev)
            _module(debug_delay_ns).run_decode_pdl(*args, inter, y, topk_weights, combined, float(scale), ctas)
        return y if combined is None else combined
    meta = torch.empty(9 * E + 68 + R // 128, dtype=torch.int32, device=dev)
    order = torch.empty(R, dtype=torch.int64, device=dev)
    gathered = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
    inter = torch.empty((R, I), dtype=torch.bfloat16, device=dev)
    combined = None
    if topk_weights is None:
        y = (
            out
            if out is not None
            else torch.empty((R, H), dtype=torch.bfloat16, device=dev)
        )
    else:
        y = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
        combined = (
            out
            if out is not None
            else torch.empty((T, H), dtype=torch.bfloat16, device=dev)
        )
    _module(debug_delay_ns).run(
        x,
        topk_ids,
        experts.w13,
        experts.w2,
        experts.gs13,
        experts.gs2,
        meta,
        order,
        gathered,
        inter,
        y,
        topk_weights,
        combined,
        float(scale),
        _VARIANTS[variant],
        ctas,
    )
    return y if combined is None else combined
