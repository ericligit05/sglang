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
# Token counts up to which "auto" runs the one-launch decode kernel. 0: it measured slower than Marlin at every
# shape on H100 (2026-11-17), so only variant="decode" reaches it.
DECODE_MAX_TOKENS = 0
# GEMM1 chunks (256 rows x 64 k) from which the decode kernel closes a batch of experts: about 16 per SM.
_DECODE_BATCH_CHUNKS = 16 * 132
# Per device and stream: the decode kernel's FP32 partial sums and its counters (zero between calls).
_decode_ws: dict = {}
_DECODE_DEBUG: dict = {}  # DEBUG: trace pointer, dbg bits, thr


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
            ("run_decode", "nvfp4_w4a16_sm90::run_decode"),
            ("run_decode2", "nvfp4_w4a16_sm90::run_decode2"),
        ],
        extra_cuda_cflags=[f"-DNVFP4_MOE_DEBUG_DELAY_NS={debug_delay_ns}"]
        if debug_delay_ns
        else [],
        post_build=require_no_rs_wgmma_hazard,
    )


def _decode_workspace(dev: torch.device, floats: int, words: int):
    """The decode kernel's partial sums and counters for this device and stream. The counters must be zero before a
    call and every call leaves them zero, so they are allocated once (zeroed) and reused. A larger request allocates
    larger buffers and keeps the old ones alive for CUDA graphs captured with them."""
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    ws = _decode_ws.get(key)
    if ws is None or ws[0].numel() < floats or ws[1].numel() < words:
        floats = max(floats, 0 if ws is None else ws[0].numel())
        words = max(words, 4096 if ws is None else ws[1].numel())
        ws = (
            torch.empty(floats, dtype=torch.float32, device=dev),
            torch.zeros(words, dtype=torch.int32, device=dev),
            ws,
        )
        _decode_ws[key] = ws
    return ws[0], ws[1]


def _decode_sizes(T: int, R: int, E: int, H: int, I: int, sms: int):
    nmax = min(E, R)
    rb1, rb2 = 2 * I // 256, H // 256
    floats = 256 * (2 * sms * R + (rb1 + rb2) * nmax * T)
    words = 1 + nmax * (rb1 + rb2) + nmax * rb1 + T * rb2
    return floats, words


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
    x is BF16 [T, H] in natural column order, topk_ids int32 [T, top_k]. variant ("auto", "single", "pair" or
    "decode") and ctas pin the kernel for tests; "auto" takes the decode kernel up to DECODE_MAX_TOKENS tokens."""
    T, H = x.shape
    R = T * topk_ids.shape[1]
    E, I = experts.gs13.shape[0], experts.intermediate
    dev = x.device
    if variant == "decode2":
        _, state = _decode_workspace(dev, 0, 32 * (1 + min(E, R)))
        inter = torch.empty((R, I), dtype=torch.bfloat16, device=dev)
        if topk_weights is None:
            y = out if out is not None else torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            combined = None
        else:
            y = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            combined = out if out is not None else torch.empty((T, H), dtype=torch.bfloat16, device=dev)
        _module(debug_delay_ns).run_decode2(
            x, topk_ids, experts.w13, experts.w2, experts.gs13, experts.gs2, inter, y, state, topk_weights,
            combined, float(scale), ctas, _DECODE_DEBUG.get("dbg2", 0),
        )
        return y if combined is None else combined
    if variant == "decode" or (
        variant == "auto" and 0 < T <= DECODE_MAX_TOKENS and R <= 256 and E <= 1024
    ):
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        part, state = _decode_workspace(dev, *_decode_sizes(T, R, E, H, I, sms))
        inter = torch.empty((R, I), dtype=torch.bfloat16, device=dev)
        if topk_weights is None:
            y = (
                out
                if out is not None
                else torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            )
            combined = None
        else:
            y = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            combined = (
                out
                if out is not None
                else torch.empty((T, H), dtype=torch.bfloat16, device=dev)
            )
        _module(debug_delay_ns).run_decode(
            x,
            topk_ids,
            experts.w13,
            experts.w2,
            experts.gs13,
            experts.gs2,
            inter,
            y,
            part,
            state,
            topk_weights,
            combined,
            float(scale),
            _DECODE_DEBUG.get("thr", _DECODE_BATCH_CHUNKS),
            _DECODE_DEBUG.get("trace", 0),
            _DECODE_DEBUG.get("dbg", 0),
        )
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
