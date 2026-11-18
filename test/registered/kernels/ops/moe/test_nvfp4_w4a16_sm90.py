"""The Hopper NVFP4 W4A16 MoE op against an FP32 reference with exact NVFP4 dequantization.

Bounds: relative L2 over all local routes < 1% and every single route < 2%; the op lands at about 0.4% and 0.5% (BF16
rounding of the activations, the intermediate and the output). A route that is skipped, written to another row, or
computed with a dropped K chunk, the wrong expert or the wrong scale is off by 9% (one of 112 chunks) to 100%, so the
per-route bound catches a single bad route. Per-route calls fill the output with a sentinel first and check that rows
of non-local routes hold zeros.
"""

import math
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.nvfp4_w4a16_sm90 import nvfp4_w4a16_sm90_moe
from sglang.kernels.ops.quantization.nvfp4_sm90_repack import (
    Nvfp4Sm90Experts,
    repack_modelopt_nvfp4_moe_for_sm90,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="needs an sm_90 GPU",
)

E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
# name: (hidden, intermediate, local experts, global experts, top_k)
SHAPES = {
    "deepseek_ep8": (7168, 2048, 32, 256, 8),
    "qwen3_30b": (
        2048,
        768,
        128,
        128,
        8,
    ),  # I / 64 = 12 chunks, not a multiple of the 4-stage ring
    "qwen3_next_ep8": (2048, 512, 64, 512, 10),
    "eplb36": (4096, 1536, 36, 288, 8),
}
SENTINEL = 7.0
_LAYERS = {}


def _dequant(w, sf, gs=None):
    tab = torch.tensor(E2M1, dtype=torch.float32, device=w.device)
    v = torch.stack((tab[(w & 15).long()], tab[(w >> 4).long()]), dim=-1).flatten(-2)
    v = v * sf.float().repeat_interleave(16, dim=-1)
    return v if gs is None else v * gs.float()


def _checkpoint(E, H, I, case, seed):
    """ModelOpt-layout experts. case: normal; subnormal (half the block scales are E4M3 subnormals); small (global
    scale 1, all block scales subnormal); gateup (each expert's up global scale is its gate scale times 0.1 .. 10)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ck = {}
    for name, (n, k) in (("w13", (2 * I, H)), ("w2", (H, I))):
        ck[name + "_weight"] = torch.randint(
            0, 256, (E, n, k // 2), dtype=torch.uint8, generator=g, device="cuda"
        )
        u = torch.rand((E, n, k // 16), generator=g, device="cuda")
        if case == "subnormal":
            sf = torch.exp2(-9.0 + 6.0 * u)
        elif case == "small":
            sf = torch.exp2(-9.0 + 3.0 * u)
        else:
            sf = torch.exp2(6.5 + u).clamp(8.0, 448.0)
        sf = sf.to(torch.float8_e4m3fn)
        gs = 1.0 / math.sqrt(8.6 * float(sf.float().pow(2).mean()) * k)
        gs = torch.full((E,), 1.0 if case == "small" else gs, device="cuda")
        if case in ("normal", "gateup"):
            gs = gs * torch.exp(0.5 * torch.randn((E,), generator=g, device="cuda"))
        ck[name + "_weight_scale"] = sf
        ck[name + "_weight_scale_2"] = gs
    gate = ck["w13_weight_scale_2"]
    up = (
        gate * torch.exp2(torch.empty_like(gate).uniform_(-3.3, 3.3, generator=g))
        if case == "gateup"
        else gate
    )
    ck["w13_weight_scale_2"] = torch.stack((gate, up), dim=1).contiguous()
    return ck


def _layer(shape, case="normal"):
    if (shape, case) not in _LAYERS:
        if len(_LAYERS) > 3:
            _LAYERS.clear()
            torch.cuda.empty_cache()
        H, I, E, _, _ = SHAPES[shape]
        ck = _checkpoint(E, H, I, case, seed=len(shape) * 7 + len(case))
        _LAYERS[(shape, case)] = (ck, repack_modelopt_nvfp4_moe_for_sm90(**ck))
    return _LAYERS[(shape, case)]


def _routing(T, topk, E, Eg, kind="skew", seed=0, remote="minus_one"):
    """Distinct expert ids per token, local experts 0 .. E - 1 the most popular (the busiest EP rank). kind: skew, hot
    (half of the tokens route to local expert 3), empty (every other local expert gets no route), remote (no local
    route). remote: other ranks' routes as -1 (SGLang's local id map) or as their global id."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pop = (
        torch.sort(torch.randn(Eg, generator=g, device="cuda"), descending=True).values
        * 1.2
    )
    s = pop - torch.log(
        -torch.log(torch.rand((T, Eg), generator=g, device="cuda").clamp_min(1e-20))
    )
    if kind == "hot":
        s[: T // 2 + 1, 3] = float("inf")
    elif kind == "empty":
        s[:, 1:E:2] = float("-inf")
    elif kind == "remote":
        s[:, :E] = float("-inf")
    ids = torch.topk(s, topk, dim=1).indices.to(torch.int32)
    if remote == "minus_one":
        ids = torch.where(ids < E, ids, -1)
    return ids.contiguous()


def _reference(ck, x, ids, E):
    """FP32 per-route outputs [R, H] and the local route mask; gate and up rows each with their own global scale."""
    flat = ids.reshape(-1).long()
    local = (flat >= 0) & (flat < E)
    out = torch.zeros((flat.numel(), x.shape[1]), dtype=torch.float32, device=x.device)
    I = ck["w2_weight"].shape[2] * 2
    gs13 = ck["w13_weight_scale_2"]
    for e in range(E):
        sel = (local & (flat == e)).nonzero().squeeze(1)
        if sel.numel() == 0:
            continue
        w13 = _dequant(ck["w13_weight"][e], ck["w13_weight_scale"][e])
        h = x.float()[sel // ids.shape[1]] @ w13.T
        inter = torch.nn.functional.silu(h[:, :I] * gs13[e, 0]) * (
            h[:, I:] * gs13[e, 1]
        )
        out[sel] = (
            inter
            @ _dequant(
                ck["w2_weight"][e], ck["w2_weight_scale"][e], ck["w2_weight_scale_2"][e]
            ).T
        )
    return out, local


def _assert_close(y, ref, rows):
    d, r = y[rows].float() - ref[rows], ref[rows]
    if r.numel() == 0:
        return
    assert bool(torch.isfinite(d).all())
    rel = float(d.norm() / r.norm())
    worst = float((d.norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)).max())
    assert rel < 0.01 and worst < 0.02, (rel, worst)


def _check(
    shape, T, kind="skew", case="normal", variant="auto", ctas=0, remote="minus_one"
):
    H, I, E, Eg, topk = SHAPES[shape]
    ck, experts = _layer(shape, case)
    x = torch.randn(
        (T, H), generator=torch.Generator(device="cuda").manual_seed(T), device="cuda"
    ).bfloat16()
    ids = _routing(T, topk, E, Eg, kind, seed=T, remote=remote)
    y = torch.full((T * topk, H), SENTINEL, dtype=torch.bfloat16, device="cuda")
    nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant=variant, ctas=ctas)
    ref, local = _reference(ck, x, ids, E)
    _assert_close(y, ref, local)
    assert bool((y[~local] == 0).all()), "a row of a non-local route is not zero"


@pytest.mark.parametrize("T", [1, 7, 8, 9, 33, 255, 753, 2047, 2048, 4096, 8191, 16384])
@pytest.mark.parametrize("shape", ["deepseek_ep8", "qwen3_30b"])
def test_token_counts(shape, T):
    """Every token count from 1 to 16384 on the auto kernel choice. Fails on: a stride that uses H for I (DeepSeek
    has H 7168 != I 2048), an off-by-one in an expert's last token tile, an expert the planner drops, the one-CTA
    kernel's unrolled producer loop taken when GEMM2's 12 chunks do not fill whole ring passes (Qwen3-30B)."""
    _check(shape, T)


@pytest.mark.parametrize(
    "variant,T,ctas",
    [
        ("pair", 9, 0),
        ("pair", 753, 0),
        ("single", 16384, 0),
        ("single", 753, 2),
        ("pair", 2048, 3),
    ],
)
def test_variants_and_ctas(variant, T, ctas):
    """Each kernel outside the range auto picks it for, and with 2 or 3 CTAs so every CTA wraps its 8-slot tile
    queue. Fails on: both CTAs of a pair taking the same row block, a queue parity error on wrap, a claim counter that
    skips or repeats a tile."""
    _check("deepseek_ep8", T, variant=variant, ctas=ctas)


@pytest.mark.parametrize(
    "kind,remote",
    [
        ("hot", "minus_one"),
        ("empty", "minus_one"),
        ("remote", "minus_one"),
        ("skew", "global"),
    ],
)
def test_routing_kinds(kind, remote):
    """One expert taking half of the tokens, empty experts, no local route, other ranks' routes as global ids.
    Fails on: a planner that shifts tile offsets past zero-route experts, -1 or a global id read as local, a hang
    or a write when no tile exists."""
    for T in (33, 4096):
        _check("deepseek_ep8", T, kind=kind, remote=remote)


@pytest.mark.parametrize("case", ["subnormal", "small", "gateup"])
def test_scale_cases(case):
    """E4M3 subnormal block scales, global scale 1 with all-subnormal block scales, and gate and up global scales
    that differ. Fails on: a scale decode that flushes subnormals, a repack that keeps only the gate's global scale,
    a fold applied as gate / up."""
    for T in (9, 4096):
        _check("deepseek_ep8", T, case=case)


@pytest.mark.parametrize("shape", ["qwen3_next_ep8", "eplb36"])
def test_other_shapes(shape):
    """Top 10 and a non-power-of-two local expert count. Fails on: top 8 hard-coded in the gather (route / 8), an
    expert table sized for a power of two."""
    for T in (1, 255, 4096):
        _check(shape, T)


def test_weighted_sum():
    """The combined output against the reference weighted sum, with other ranks' routes as -1 and a routed scaling
    factor. Fails on: a combine that reads rows of non-local routes (they hold NaN here), a weight applied to the
    wrong slot, the scale dropped."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    for T in (1, 300, 4096):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=5 + T)
        w = torch.rand((T, topk), device="cuda")
        ref, local = _reference(ck, x, ids, E)
        ref_sum = (ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)).sum(
            1
        ) * 2.5
        # The op's [R, H] workspaces reuse these freed blocks, so a read of a non-local route's row gives NaN.
        poison = [
            torch.full((T * topk, H), float("nan"), dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        del poison
        out = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, scale=2.5)
        _assert_close(out, ref_sum, slice(None))


def test_cuda_graph_capture_and_replay():
    """Capture at three token counts (the one-block plan, and each GEMM kernel), replay with fresh routings and
    inputs, compare with eager calls and the reference. Fails on: a host-side plan or sync (capture fails, or replay
    reuses the captured routing), a buffer outside the graph's pool, meta not zeroed per call (the second replay
    claims no tiles)."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    for T in (3, 753, 4096):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=1)
        w = torch.rand((T, topk), device="cuda")
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
        nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out)
        for rep, kind in enumerate(("skew", "hot", "empty")):
            x.copy_(torch.randn((T, H), device="cuda").bfloat16())
            ids.copy_(_routing(T, topk, E, Eg, kind=kind, seed=77 + rep))
            graph.replay()
            eager = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w)
            torch.cuda.synchronize()
            ref, local = _reference(ck, x, ids, E)
            ref_sum = (
                ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)
            ).sum(1)
            _assert_close(out, ref_sum, slice(None))
            assert float((out.float() - eager.float()).abs().max()) <= 1e-2 * float(
                eager.float().abs().max()
            )


def test_reload_in_place_keeps_graphs_valid():
    """Repack new weights into the same buffers and replay a graph captured before; the output must follow the new
    weights. Fails on: a repack that allocates new buffers on reload (the captured tensor maps keep reading the old
    weights) or leaves stale bytes."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck1, ck2 = _checkpoint(E, H, I, "normal", 11), _checkpoint(E, H, I, "gateup", 12)
    experts = repack_modelopt_nvfp4_moe_for_sm90(**ck1)
    T = 255
    x = torch.randn((T, H), device="cuda").bfloat16()
    ids = _routing(T, topk, E, Eg, seed=3)
    y = torch.empty((T * topk, H), dtype=torch.bfloat16, device="cuda")
    nvfp4_w4a16_sm90_moe(x, ids, experts, out=y)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nvfp4_w4a16_sm90_moe(x, ids, experts, out=y)
    assert repack_modelopt_nvfp4_moe_for_sm90(**ck2, out=experts) is experts
    graph.replay()
    torch.cuda.synchronize()
    ref, local = _reference(ck2, x, ids, E)
    _assert_close(y, ref, local)


def test_argument_checks():
    """Bad shapes and dtypes raise before any kernel runs. Fails on: a missing check (the call reads out of
    bounds)."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    _, ex = _layer("deepseek_ep8")
    x = torch.randn((16, H), device="cuda").bfloat16()
    ids = _routing(16, topk, E, Eg)
    bad = [
        (x.half(), ids, ex),
        (x[:, :4096].contiguous(), ids, ex),
        (x, ids.long(), ex),
        (x, ids[:8], ex),
        (x, ids, Nvfp4Sm90Experts(ex.w13[:16], ex.w2, ex.gs13, ex.gs2, H, I)),
        (x, ids, Nvfp4Sm90Experts(ex.w13, ex.w2, ex.gs13, ex.gs2, H, I + 128)),
    ]
    for args in bad:
        with pytest.raises(Exception):
            nvfp4_w4a16_sm90_moe(*args)


_SPIN = r"""
#include <ATen/cuda/CUDAContext.h>
__global__ void __launch_bounds__(1024, 1) spin(volatile int* flag, int* started) {
  extern __shared__ char s[];
  if (threadIdx.x == 0) {
    s[0] = 1;
    atomicAdd_system(started, 1);
    while (*flag == 0) __nanosleep(2000);
  }
  __syncthreads();
}
void launch(torch::Tensor flag, torch::Tensor started, int64_t blocks) {
  cudaFuncSetAttribute(spin, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024);
  spin<<<(int)blocks, 1024, 160 * 1024, at::cuda::getCurrentCUDAStream()>>>(
      (volatile int*)flag.data_ptr<int>(), started.data_ptr<int>());
}
"""


@pytest.mark.parametrize("variant", ["single", "pair"])
def test_progress_while_another_kernel_holds_sms(variant):
    """Another stream's kernel holds all but 16 SMs (an overlapped side stream, MPS, a green context); the op must
    finish on the SMs it has. Fails on: a persistent kernel that gives CTA b tile b statically, so the resident CTAs
    spin on GEMM2 tiles whose GEMM1 tiles belong to CTAs that are still queued."""
    from torch.utils.cpp_extension import load_inline

    spin = load_inline(
        "nvfp4_sm90_test_spin",
        cpp_sources="void launch(torch::Tensor, torch::Tensor, int64_t);",
        cuda_sources=_SPIN,
        functions=["launch"],
    )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    T = 753
    x = torch.randn((T, H), device="cuda").bfloat16()
    ids = _routing(T, topk, E, Eg, seed=3)
    y = torch.empty((T * topk, H), dtype=torch.bfloat16, device="cuda")
    nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant=variant)
    torch.cuda.synchronize()
    flag = torch.zeros(1, dtype=torch.int32).pin_memory()
    started = torch.zeros(1, dtype=torch.int32).pin_memory()
    safety = threading.Timer(
        60, lambda: flag.fill_(1)
    )  # never leave the blocker spinning
    safety.start()
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            spin.launch(flag, started, sms - 16)
        t0 = time.time()
        while int(started[0]) < sms - 16 and time.time() - t0 < 10:
            time.sleep(0.01)
        assert int(started[0]) == sms - 16, "the blocker did not become resident"
        done = torch.cuda.Event()
        with torch.cuda.stream(torch.cuda.Stream()):
            nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant=variant)
            done.record()
        t0 = time.time()
        while not done.query() and time.time() - t0 < 10:
            time.sleep(0.01)
        finished = done.query()
    finally:
        flag.fill_(1)
        safety.cancel()
    torch.cuda.synchronize()
    assert finished, "no progress on the 16 free SMs within 10s"
    ref, local = _reference(ck, x, ids, E)
    _assert_close(y, ref, local)


# ---- the one-launch decode kernel (variant "decode") ----


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 8, 9, 15, 16])
@pytest.mark.parametrize("shape", ["deepseek_ep8", "qwen3_30b"])
def test_decode_token_counts(shape, T):
    """Every token count the decode kernel accepts (1 to 16), across both n8-block widths. Fails on: a K split
    whose pieces miss or repeat a chunk (one chunk of 112 is 9% of a route), a piece's partial sums read from another
    CTA's slot, a token beyond the first n8 block dropped, the H and I strides swapped (DeepSeek), a GEMM2 chunk that
    reads an intermediate row block before its flag (Qwen3-30B's 12 chunks per row block span six flags)."""
    _check(shape, T, variant="decode")


@pytest.mark.parametrize(
    "kind,remote",
    [
        ("hot", "minus_one"),
        ("empty", "minus_one"),
        ("remote", "minus_one"),
        ("skew", "global"),
    ],
)
def test_decode_routing_kinds(kind, remote):
    """One expert taking half of the tokens (9 routes on one expert at 16 tokens), empty experts, no local route at
    all, other ranks' routes as global ids. Fails on: an expert slot table that miscounts routes, a hang when no
    expert is local (no piece anywhere), a global id read as local."""
    for T in (1, 4, 16):
        _check("deepseek_ep8", T, kind=kind, remote=remote, variant="decode")


@pytest.mark.parametrize("case", ["subnormal", "small", "gateup"])
def test_decode_scale_cases(case):
    """The decode kernel with subnormal block scales, global scale 1 and differing gate and up scales. Fails on: a
    finisher that applies GEMM1's global scale to GEMM2 or drops the folded up / gate factor."""
    for T in (1, 9):
        _check("deepseek_ep8", T, case=case, variant="decode")


@pytest.mark.parametrize("shape", ["qwen3_next_ep8", "eplb36"])
def test_decode_other_shapes(shape):
    """Top 10 and 36 local experts on the decode kernel. Fails on: top 8 hard-coded in the routing or the weighted
    sum, a batch split that drops the last expert when the experts do not divide evenly."""
    for T in (1, 7, 16):
        _check(shape, T, variant="decode")


def test_decode_weighted_sum_and_repeat():
    """The fused weighted sum against the reference, a token without a local route (its row must be zeros), then the
    same call repeated: the output must be bitwise identical every time. Fails on: a combine that misses a route or
    runs before a route's rows exist, a counter not returned to zero (the next call hangs or finishes early), a
    reduction whose order depends on timing."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    for T in (1, 3, 16):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=5 + T)
        if T > 1:  # a token with no local route (at one token the sum would be all zeros)
            ids[T // 2] = -1
        w = torch.rand((T, topk), device="cuda")
        ref, local = _reference(ck, x, ids, E)
        ref_sum = (
            ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)
        ).sum(1) * 2.5
        out = torch.full((T, H), SENTINEL, dtype=torch.bfloat16, device="cuda")
        nvfp4_w4a16_sm90_moe(
            x, ids, experts, topk_weights=w, out=out, scale=2.5, variant="decode"
        )
        _assert_close(out, ref_sum, slice(None))
        if T > 1:
            assert bool((out[T // 2] == 0).all()), "a token without a local route is not zero"
        for _ in range(5):
            again = nvfp4_w4a16_sm90_moe(
                x, ids, experts, topk_weights=w, scale=2.5, variant="decode"
            )
            assert torch.equal(again, out)


def test_decode_cuda_graph_capture_and_reload():
    """Capture the decode kernel at 1 and 4 tokens, replay with fresh routings, then repack new weights into the
    same buffers and replay again. Fails on: a host-side routing decision (replay reuses the captured routing),
    workspace or counters outside what the graph captured, a reload the captured kernel does not see."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck1, ck2 = _checkpoint(E, H, I, "normal", 21), _checkpoint(E, H, I, "gateup", 22)
    experts = repack_modelopt_nvfp4_moe_for_sm90(**ck1)
    for T in (1, 4):
        repack_modelopt_nvfp4_moe_for_sm90(**ck1, out=experts)  # undo the previous count's reload
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=1)
        w = torch.rand((T, topk), device="cuda")
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
        nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out, variant="decode")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out, variant="decode")
        for rep, (kind, ck) in enumerate((("skew", ck1), ("empty", ck1), ("skew", ck2))):
            if rep == 2:
                repack_modelopt_nvfp4_moe_for_sm90(**ck2, out=experts)
            x.copy_(torch.randn((T, H), device="cuda").bfloat16())
            ids.copy_(_routing(T, topk, E, Eg, kind=kind, seed=77 + rep))
            graph.replay()
            eager = nvfp4_w4a16_sm90_moe(
                x, ids, experts, topk_weights=w, variant="decode"
            )
            torch.cuda.synchronize()
            ref, local = _reference(ck, x, ids, E)
            ref_sum = (
                ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)
            ).sum(1)
            _assert_close(out, ref_sum, slice(None))
            assert torch.equal(out, eager)


def test_decode_matches_grouped_kernel_and_refuses_too_many_tokens():
    """The decode kernel agrees with the grouped kernel per route, and refuses more than 16 tokens. Fails on: a
    decode result that drifts from the grouped kernel beyond BF16 rounding, a call past the kernel's two n8 blocks
    of routes per expert."""
    H, I, E, Eg, topk = SHAPES["qwen3_30b"]
    ck, experts = _layer("qwen3_30b")
    for T in (1, 4, 16):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=9)
        a = nvfp4_w4a16_sm90_moe(x, ids, experts, variant="decode").float()
        b = nvfp4_w4a16_sm90_moe(x, ids, experts, variant="single").float()
        assert float((a - b).norm() / b.norm()) < 0.01
    x = torch.randn((17, H), device="cuda").bfloat16()
    with pytest.raises(Exception):
        nvfp4_w4a16_sm90_moe(x, _routing(17, topk, E, Eg), experts, variant="decode")


# ---- decode path 3 (variant "decode3", not dispatched) ----


def _dec3_cfg(g1h=0, ns=0):
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    op._DECODE_DEBUG["cfg3"] = (0xFFFF << 16) | (g1h << 12) | (ns << 8)


@pytest.mark.parametrize("g1h", [1, 2])
@pytest.mark.parametrize("T", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", ["deepseek_ep8", "qwen3_30b", "qwen3_next_ep8", "eplb36"])
def test_decode3_token_counts(shape, T, g1h):
    """Every token count decode3 accepts, with 32 + 32 and with 16 + 16 row GEMM1 items. Fails on: a tagged word
    written to the wrong intermediate column (the half items' column map), a fetch that takes a stale word, a slot's
    rows read from another slot's shared-memory rows."""
    _dec3_cfg(g1h)
    try:
        _check(shape, T, variant="decode3")
    finally:
        _dec3_cfg()


@pytest.mark.parametrize("ns", [2, 3])
def test_decode3_small_ring_and_rotating_buffers(ns):
    """Two and three ring stages; at four tokens DeepSeek's 32 routes do not all fit, so slots rotate through
    buffers. Fails on: a buffer refilled before every consumer warp has read it, a ring stage reused early."""
    _dec3_cfg(0, ns)
    try:
        for T in (1, 4):
            _check("deepseek_ep8", T, variant="decode3")
    finally:
        _dec3_cfg()


@pytest.mark.parametrize("kind,remote", [("hot", "minus_one"), ("empty", "minus_one"), ("remote", "minus_one")])
def test_decode3_routing_kinds(kind, remote):
    """One expert taking most routes, empty experts, no local route at all. Fails on: a hang when no slot exists,
    a slot with more than one route read with the wrong row stride."""
    for T in (1, 4):
        _check("deepseek_ep8", T, kind=kind, remote=remote, variant="decode3")


def test_decode3_weighted_sum_repeat_and_graph():
    """The weighted sum against the reference, then the call repeated eagerly and replayed from a CUDA graph with
    new routings: every result must match the reference and the eager call bitwise. Fails on: an epoch that does not
    advance (a later call takes the earlier call's words), per-route weights staged from the wrong route."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    for T in (1, 3):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=5 + T)
        w = torch.rand((T, topk), device="cuda")
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
        nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out, scale=2.5, variant="decode3")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out, scale=2.5, variant="decode3")
        for rep in range(4):
            x.copy_(torch.randn((T, H), device="cuda").bfloat16())
            ids.copy_(_routing(T, topk, E, Eg, kind=("skew", "hot")[rep % 2], seed=40 + rep))
            graph.replay()
            eager = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, scale=2.5, variant="decode3")
            torch.cuda.synchronize()
            ref, local = _reference(ck, x, ids, E)
            ref_sum = (ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)).sum(1) * 2.5
            _assert_close(out, ref_sum, slice(None))
            assert torch.equal(out, eager)
