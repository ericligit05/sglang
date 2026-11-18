/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* decode4's pair-3 multiply (below) uses the one-IMAD decode of onnxruntime's moe_gemv_fp4.cu, which carries this
license:

MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Grouped NVFP4 x BF16 routed-expert GEMMs for Hopper (sm_90a): y[route] = W2 (silu(W1 x) * (W3 x)) for every route
// whose expert is local, with runtime hidden size H, expert intermediate size I, local expert count E and top-k, and
// optionally the weighted sum over each token's routes. python/sglang/kernels/ops/moe/nvfp4_w4a16_sm90.py owns the
// workspace and python/sglang/kernels/ops/quantization/nvfp4_sm90_repack.py packs the weights. Per call:
//   count_k   per-expert route counts of the local routes (ids outside [0, E) are other ranks' routes)
//   plan_k    per expert its token tiling, the experts by route count (most first), both GEMMs' tile offsets
//   scatter_k route ids into expert-sorted order, other ranks' routes after the local ones
//             (front_small_k does these three in one block up to kSmallRoutes routes)
//   gather_k  the routed token rows into sorted order, applying the K permutation the packed weights expect, so the
//             caller passes hidden states in their natural column order. Without the weighted sum it also writes
//             zeros to the output rows of other ranks' routes.
//   grouped_k (or pair_k, CTA pairs sharing token rows by multicast, at 16384+ routes) both GEMMs in one persistent
//             launch: GEMM1 tiles first, each GEMM2 tile waits for its token tile's GEMM1 row blocks.
//   combine_k out[t] = scale * sum over t's local routes k of w[t, k] y[t top_k + k], in increasing k
// Weights are 16-row x 64-k units of 576B (512B of E2M1 codes in decode4's bit layout, then the 64 E4M3 block scales
// as sc_bf16 codes), 256 rows per GEMM1 row block (for each half of its 128 intermediate columns, 64 gate rows then
// the same 64 up rows). One FP32 global scale per expert per GEMM is applied in the epilogue. Differing gate and up
// scales are folded exactly at pack time (up / gate into GEMM2's scale).
// Every wgmma here takes A from registers, and ptxas through CUDA 13.2 can overwrite those registers while the wgmma
// is in flight, so the loader refuses a build whose SASS shows that (sglang.kernels.jit.utils.wgmma_rs_hazard).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/async.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

namespace sglang::nvfp4_w4a16_sm90 {

#define GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")

constexpr int NW = 8, UNIT2 = 576;       // consumer warps; bytes per 16-row x 64-k weight unit
constexpr int WB2 = NW * 2 * UNIT2;      // weight bytes per 64-k chunk of a 256-row tile (9216)
constexpr int WROWS = WB2 / 512;         // the same in 512B rows of the weight tensor maps (18)
constexpr int MX_SB = 37888, MX_NS = 4;  // ring stage bytes (1KB aligned), stages
constexpr int MX_PAIR = 18432;           // narrow tiles: x0 at 0, w0 at 8192, x1 at MX_PAIR, w1 at MX_PAIR + 8192
constexpr uint32_t PAIR_W1 = 8192 + WB2, PAIR_X1 = 28672;  // pair_k narrow stage: x0, w0 w1 (one box), x1
constexpr int XOFF = 16384;           // full-width stage: token rows at 0 (up to 128 x 128B), weights here
constexpr int ST = 264;               // epilogue staging stride (BF16) per token row
constexpr int EBYTES = 128 * ST * 2;  // epilogue staging: [128 tokens][264]
constexpr int SMEM = 1024 + MX_NS * MX_SB + EBYTES;
constexpr int TOKMAX = 128;        // tokens per tile
constexpr int MX_MS = 16;          // tiles of at most this many tokens use mma.sync m16n8k16
constexpr int INFO_NTT_BITS = 24;  // plan info word: token tile count below, tsz / 32 above

// Test builds only (tests/test_races.py builds with -DNVFP4_MOE_DEBUG_DELAY_NS=<ns>): stretch the windows of the
// global-memory hand-offs so a missing wait or a reused buffer fails deterministically instead of rarely. gather_k
// lets the GEMM kernel launch at its start and then sleeps before writing its rows (a missing PDL wait reads them
// unwritten); the copiers sleep before writing the last GEMM1 row block of a token tile (a GEMM2 tile that starts
// early reads it unwritten), and one copier warp sleeps before its GEMM2 stores (a slot buffer reused too early is
// overwritten under it). The host fills the gathered rows and the intermediate with NaN first.
#ifndef NVFP4_MOE_DEBUG_DELAY_NS
#define NVFP4_MOE_DEBUG_DELAY_NS 0
#endif
__device__ __forceinline__ void debug_delay() {
  if constexpr (NVFP4_MOE_DEBUG_DELAY_NS > 0) {
    uint64_t t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
      __nanosleep(1000);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < (uint64_t)NVFP4_MOE_DEBUG_DELAY_NS);
  }
}

// ---- per-call plan (int32 meta buffer, zeroed by the caller), offsets by local expert count E ----
// cnt[E] | start[E] | toff1[E + 1] | toff2[E + 1] | counter | nonlocal cursor | cursor[E] | perm[E] | info[E] | jt0[E]
// | cnt1[...]. perm, info, jt0 and the tile offsets are by plan position (experts by route count, most first); info =
// ntt | tsz / 32 << 24 (ntt token tiles of tsz tokens, the last one shorter; R < 2^30 keeps ntt below 2^23); cnt1
// counts the GEMM1 row blocks written per token tile. The non-local cursor places other ranks' routes after the local
// ones in the sorted order, so gather_k can zero their output rows.
__host__ __device__ __forceinline__ int m_start(int E) {
  return E;
}
__host__ __device__ __forceinline__ int m_toff1(int E) {
  return 2 * E;
}
__host__ __device__ __forceinline__ int m_toff2(int E) {
  return 3 * E + 1;
}
__host__ __device__ __forceinline__ int m_ctr(int E) {
  return 4 * E + 2;
}
__host__ __device__ __forceinline__ int m_nlcur(int E) {
  return 4 * E + 3;
}
__host__ __device__ __forceinline__ int m_cur(int E) {
  return 4 * E + 4;
}
__host__ __device__ __forceinline__ int m_perm(int E) {
  return 5 * E + 4;
}
__host__ __device__ __forceinline__ int m_info(int E) {
  return 6 * E + 4;
}
__host__ __device__ __forceinline__ int m_jt0(int E) {
  return 7 * E + 4;
}
__host__ __device__ __forceinline__ int m_cnt1(int E) {
  return 8 * E + 4;
}

struct Shape {
  int E;         // local experts
  int RB1, RB2;  // 256-row blocks per expert: GEMM1 (2 I / 256), GEMM2 (H / 256)
  int C1, C2;    // 64-wide K chunks: GEMM1 (H / 64), GEMM2 (I / 64)
  int I, H;      // row strides of the intermediate and of the output
};

// Token row maps of a (gathered input, [R][H]) and inter ([R][I]): boxes of 64 k x {64, 128, 8, 16, 32} rows; the
// weights as rows of 512B, boxes of one chunk (18 rows) and, for pair_k's narrow tiles, two chunks (36 rows).
struct Maps {
  CUtensorMap x64[2], x128[2], xn[2][3], w[2], w2c[2];
};

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ bool elect_one() {  // elect.sync over the full (converged) warp: true in one lane
  uint32_t p;
  asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0, 1, 0, P;\n}" : "=r"(p));
  return p != 0;
}
__device__ __forceinline__ void stsm4t(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr),
               "r"(a),
               "r"(b),
               "r"(c),
               "r"(d)
               : "memory");
}
__device__ __forceinline__ uint32_t pkbf(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(bar)
      : "memory");
}
__device__ __forceinline__ void
tma_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(bar),
      "h"(mask)
      : "memory");
}
__device__ __forceinline__ void
tma_2d_h(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(bar),
      "l"(pol)
      : "memory");
}
__device__ __forceinline__ uint64_t pol_evict_first() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// CTA-pair helpers for pair_k: a shared address in CTA r of the cluster, remote arrives, release / acquire arrives for
// the claim hand-off, a remote shared store and a cluster-wide barrier.
__device__ __forceinline__ uint32_t cluster_map(uint32_t a, uint32_t r) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r));
  return o;
}
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_arrive_release_cluster(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait_acquire_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::
          "r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void st_shared_cluster(uint32_t a, int v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void cluster_sync() {  // every thread of both CTAs (divergent call sites: no .aligned)
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr) {  // K-major, 128B swizzle, 8-row groups 1KB apart
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int K_>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K_) : "memory");
}

// clang-format off
// m64nNk16 BF16 wgmma with A from registers and FP32 accumulators: wgmma_n<N> on float[N / 2], and
// wgmma_part<N, FULL> on the first N / 2 of a float[FULL / 2] (a tile with fewer tokens than its accumulators).
template <int N>
__device__ __forceinline__ void wgmma_n(float (&d)[N / 2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);
template <int N, int FULL>
__device__ __forceinline__ void wgmma_part(float (&d)[FULL / 2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint64_t desc);
template <>
__device__ __forceinline__ void wgmma_n<32>(float (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15"
      "}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_n<64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31"
      "}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_n<128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
      "}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<72, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35"
      "}, {%36,%37,%38,%39}, %40, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<80, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39"
      "}, {%40,%41,%42,%43}, %44, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<88, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %49, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43"
      "}, {%44,%45,%46,%47}, %48, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<96, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47"
      "}, {%48,%49,%50,%51}, %52, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<104, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %57, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51"
      "}, {%52,%53,%54,%55}, %56, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<112, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55"
      "}, {%56,%57,%58,%59}, %60, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<120, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %65, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59"
      "}, {%60,%61,%62,%63}, %64, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<24, 32>(float (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11"
      "}, {%12,%13,%14,%15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<40, 64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %25, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19"
      "}, {%20,%21,%22,%23}, %24, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<48, 64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23"
      "}, {%24,%25,%26,%27}, %28, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<56, 64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %33, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27"
      "}, {%28,%29,%30,%31}, %32, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

// clang-format on

__device__ __forceinline__ void
mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// Exact decode with no table: an E2M1 code c (sign bit 3) placed as BF16 bits (c & 7) << 6 | (c & 8) << 12 is
// exactly e2m1(c) * 2^-126. Multiplying by the block scale times 2^63 (exact) gives e2m1 * scale * 2^-63, a normal
// BF16; the 2^63 goes on the output with the global scale. decode4 does all four A pairs of a word in 6 ALU ops and 3
// IMADs; the pack places each code's bits so every pair lands under the mask M. Pair 3's single multiply by 0x1084 is
// the one-IMAD trick of onnxruntime's FP4 MoE GEMV decode (moe_gemv_fp4.cu; its MIT notice is at the top).
__device__ __forceinline__ void decode4(uint32_t q, uint32_t* o) {
  const uint32_t M = 0x81C081C0u;
  o[0] = q & M;
  o[1] = (q << 3) & M;
  o[2] = (q << 6) & M;
  o[3] = ((__funnelshift_l(q, q, 6) & 0x001B001Bu) * 0x1084u) & M;
}
// Two block scales (bytes 0, 1 of v) -> BF16 bits of scale * 2^63, each in both halves. The pack stores each E4M3
// scale losslessly as the code 8 E + m with scale = 1.m * 2^(E - 9) (subnormals normalized), so its BF16 bits are
// code * 16 + (181 << 7).
__device__ __forceinline__ uint32_t sc_bf16(uint32_t v, uint32_t sel) {
  return prmt(v, 0, sel) * 16u + 0x5A805A80u;
}
__device__ __forceinline__ uint32_t bmulh(uint32_t a, uint32_t s, int h) {
  const __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), h ? __high2bfloat162(sv) : __low2bfloat162(sv));
  return *reinterpret_cast<uint32_t*>(&r);
}

// One 64-k chunk: this warp's two units (rows ubase and ubase + 4 of the stage's 16) decoded and issued as W-wide
// wgmmas on accumulators of width TN: the whole chunk is decoded first, then one fence and the eight wgmmas.
template <int W, int TN>
__device__ __forceinline__ void
chunk_mma(float (&acc)[2][TN / 2], uint32_t (&r)[2][16], const uint8_t* wb, int ubase, int lane, uint32_t xa) {
  constexpr int MB = 2;
  const uint64_t d0 = wg_desc(xa);  // k16 step kk is 32B further: + 2 in the 16B address field
  auto mma = [&](int b, int kk) {
    if constexpr (W == TN)
      wgmma_n<TN>(acc[b], r[b][4 * kk], r[b][4 * kk + 1], r[b][4 * kk + 2], r[b][4 * kk + 3], d0 + 2 * kk);
    else
      wgmma_part<W, TN>(acc[b], r[b][4 * kk], r[b][4 * kk + 1], r[b][4 * kk + 2], r[b][4 * kk + 3], d0 + 2 * kk);
  };
  uint4 nb[MB];
  uint32_t S[MB];  // one scale pair (rows g | g + 8, this lane's block) per unit per chunk
#pragma unroll
  for (int b = 0; b < MB; b++) {
    const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
    nb[b] = *reinterpret_cast<const uint4*>(up + lane * 16);
    S[b] = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
  }
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
#pragma unroll
    for (int b = 0; b < MB; b++) {
      const uint32_t w = kk == 0 ? nb[b].x : kk == 1 ? nb[b].y : kk == 2 ? nb[b].z : nb[b].w;
      uint32_t o[4];
      decode4(w, o);
      r[b][4 * kk + 0] = bmulh(o[0], S[b], 0);
      r[b][4 * kk + 1] = bmulh(o[1], S[b], 1);
      r[b][4 * kk + 2] = bmulh(o[2], S[b], 0);
      r[b][4 * kk + 3] = bmulh(o[3], S[b], 1);
    }
  }
  wg_fence();
#pragma unroll
  for (int kk = 0; kk < 4; kk++)
#pragma unroll
    for (int b = 0; b < MB; b++)
      mma(b, kk);
}
template <int V>
using IC = std::integral_constant<int, V>;

// MMA width of a tile with nv tokens: 8 / 16 (mma.sync), 32, 64, 128.
__host__ __device__ __forceinline__ int mx_w(int nv) {
  return nv <= MX_MS ? (nv + 7) / 8 * 8 : nv <= 32 ? 32 : nv <= 64 ? 64 : 128;
}

struct TileInfo {
  int t, mode, e, rb, row0, nv, jt;
};

// Locate tile t (warp-uniform, 0 <= t < total) from the plan: GEMM1 tiles first, then GEMM2's, each numbered over the
// experts in plan order; within an expert, row block outer and token tile inner. crank >= 0 (pair_k): t is even and
// the pair's two CTAs take row blocks 2 q and 2 q + 1. The binary search over the tile offsets is one ballot per 32
// experts (the offsets are monotone, so the expert is the last one whose offset is <= t).
__device__ __forceinline__ TileInfo locate_warp(const int* __restrict__ meta, const Shape& S, int t, int crank) {
  const int E = S.E, lane = threadIdx.x & 31;
  const int total1 = __ldg(meta + m_toff1(E) + E);
  TileInfo o;
  o.t = t;
  o.mode = t < total1 ? 1 : 2;
  const int tq = o.mode == 1 ? t : t - total1;
  const int* to = meta + (o.mode == 1 ? m_toff1(E) : m_toff2(E));
  int cnt = 0;
  for (int b = 0; b < E; b += 32) {
    const int i = b + lane;
    cnt += __popc(__ballot_sync(0xffffffffu, i < E && __ldg(to + i) <= tq));
  }
  const int pos = cnt - 1, lt = tq - __ldg(to + pos);
  o.e = __ldg(meta + m_perm(E) + pos);
  const int inf = __ldg(meta + m_info(E) + pos), ntt = inf & ((1 << INFO_NTT_BITS) - 1);
  const int tsz = (inf >> INFO_NTT_BITS) * 32;
  int tt;
  if (crank < 0) {
    o.rb = lt / ntt;
    tt = lt - o.rb * ntt;
  } else {
    const int qr = (lt >> 1) / ntt;
    o.rb = 2 * qr + crank;
    tt = (lt >> 1) - qr * ntt;
  }
  o.row0 = __ldg(meta + m_start(E) + o.e) + tt * tsz;
  o.nv = min(tsz, __ldg(meta + o.e) - tt * tsz);
  o.jt = __ldg(meta + m_jt0(E) + pos) + tt;
  return o;
}
// Queue entry: x = expert, y = row block | GEMM << 16 | MMA width << 20, z = first row, w = tokens.
__device__ __forceinline__ int4 tl_pack(const TileInfo& o, int W) {
  return make_int4(o.e, o.rb | o.mode << 16 | W << 20, o.row0, o.nv);
}

// Copy-out of tile k by the 96 copier threads (ct 0 .. 95) of grouped_k and pair_k, between the consumers' staging
// and the next tile. Named barrier 2: staging written (consumers arrive, copiers sync); barrier 3: staging free again
// (copiers arrive, consumers sync); barrier 4: the copiers alone. GEMM1 applies SiLU(gate) * up to the staged BF16
// gate and up rows, writes its 128 intermediate columns and then releases the row block to GEMM2's TMA reads (cnt1);
// GEMM2 writes each token's 256 output columns to its route's row. zs is the tile's slot buffer, double-buffered by
// tile parity: a copier loading tile k + 1's rows has passed tile k + 1's barrier, so every copier is done with tile
// k - 1's buffer.
__device__ __forceinline__ void copy_out(
    const int4 tl,
    int jt,
    int ct,
    int* zs,
    const __nv_bfloat16* obuf,
    __nv_bfloat16* out1,
    __nv_bfloat16* out2,
    const int64_t* slot_of,
    int* cnt1,
    const Shape& S) {
  constexpr int NC = NW * 32;
  const int rb = tl.y & 0xFFFF, mode = (tl.y >> 16) & 15, row0 = tl.z, nv = tl.w;
  if (mode == 2) {
    for (int i = ct; i < nv; i += 96)
      zs[i] = (int)slot_of[row0 + i];
    asm volatile("bar.sync 4, 96;" ::: "memory");
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
  if (mode == 1) {
    if (rb == S.RB1 - 1) debug_delay();
    for (int i = ct; i < nv * 16; i += 96) {
      const int tok = i >> 4, c8 = (i & 15) * 8;
      const uint4 gq = *reinterpret_cast<const uint4*>(obuf + tok * ST + c8);
      const uint4 uq = *reinterpret_cast<const uint4*>(obuf + tok * ST + 128 + c8);
      const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gq);
      const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uq);
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int x = 0; x < 8; x++) {
        const float gv = __bfloat162float(gp[x]);
        const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));
        o[x] = __float2bfloat16(sl * __bfloat162float(up[x]));
      }
      *reinterpret_cast<uint4*>(out1 + (size_t)(row0 + tok) * S.I + rb * 128 + c8) = *reinterpret_cast<const uint4*>(o);
    }
    // Release: the stores reach the async proxy and the GPU scope before any copier's count does.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __threadfence();
    asm volatile("bar.sync 4, 96;" ::: "memory");
    if (ct == 0) atomicAdd(&cnt1[jt], 1);
  } else {
    if (ct >= 64) debug_delay();
    for (int i = ct; i < nv * 32; i += 96) {
      const int tok = i >> 5, c8 = (i & 31) * 8;
      *reinterpret_cast<uint4*>(out2 + (size_t)zs[tok] * S.H + rb * 256 + c8) =
          *reinterpret_cast<const uint4*>(obuf + tok * ST + c8);
    }
  }
  asm volatile("bar.arrive 3, %0;" ::"n"(NC + 96) : "memory");
}

// Epilogue of tile k in grouped_k and pair_k, by each consumer warp: once the copiers are done with the staging
// (named barrier 3, from tile 1 on), scale the accumulators by the global scale times 2^63 (decode4's exponent offset)
// and stage them [token][ST] as BF16 by stmatrix.x4.trans, then tell the copiers (barrier 2). Columns: GEMM1 gate
// (warp / 4) * 64 + (warp % 4) * 16 (+ 8), up the same + 128 (block 1 holds the up rows of block 0's gate rows); GEMM2
// the tile's weight row. Per stmatrix: blocks (tokens 8j.., rows g), (8j.., g + 8), (8j + 8.., g), (8j + 8.., g + 8);
// lane 8i + t gives row t of block i.
template <int AW>
__device__ __forceinline__ void
stage_tile(const float (&acc)[2][AW / 2], float gscale, int k, int mode, int warp, int lane, __nv_bfloat16* obuf) {
  constexpr int NC = NW * 32;
  const float sc = gscale * 9.223372036854775808e18f;
  if (k > 0) asm volatile("bar.sync 3, %0;" ::"n"(NC + 96) : "memory");
  const int i4 = lane >> 3, t8 = lane & 7;
  const uint32_t ob = (uint32_t)__cvta_generic_to_shared(obuf);
#pragma unroll
  for (int b = 0; b < 2; b++) {
    const int cb = (mode == 1 ? 128 * b + (warp / 4) * 64 : 64 * b + (warp / 4) * 128) + (warp % 4) * 16 + 8 * (i4 & 1);
    const uint32_t base = ob + ((8 * (i4 >> 1) + t8) * ST + cb) * 2;
#pragma unroll
    for (int j = 0; j < AW / 8; j += 2)
      stsm4t(
          base + j * 8 * ST * 2,
          pkbf(acc[b][4 * j] * sc, acc[b][4 * j + 1] * sc),
          pkbf(acc[b][4 * j + 2] * sc, acc[b][4 * j + 3] * sc),
          pkbf(acc[b][4 * j + 4] * sc, acc[b][4 * j + 5] * sc),
          pkbf(acc[b][4 * j + 6] * sc, acc[b][4 * j + 7] * sc));
  }
  asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
}

// ---- the persistent GEMM kernel. Warp roles: 8 consumer warps (two warpgroups) decode and multiply; warp NW is the
// producer (TMA loads into a ring of MX_NS stages); warps NW + 1 .. NW + 3 copy finished tiles out, and the first of
// them claims and locates the next tile (scheduler). Every tile, a CTA's first included, is claimed from a counter, so
// only resident CTAs take tiles: a CTA that is still queued behind another kernel holds none, and the GEMM1 tiles a
// GEMM2 tile waits for were all claimed (by running CTAs) before it.
__global__ void __launch_bounds__((NW + 4) * 32, 1) grouped_k(
    const __grid_constant__ Maps mp,
    int* __restrict__ meta,
    const float* __restrict__ gs1,
    const float* __restrict__ gs2,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    const int64_t* __restrict__ slot_of,
    const Shape S) {
  constexpr int NC = NW * 32, SB = MX_SB, NS = MX_NS;
  const int E = S.E;
  int* const counter = meta + m_ctr(E);
  int* const cnt1 = meta + m_cnt1(E);
  extern __shared__ uint8_t mraw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ int s_tq[8];
  __shared__ int4 s_tl[8];
  __shared__ int s_tj[8];
  __shared__ __align__(8) uint64_t tbars[8];  // queue slot published (producer / scheduler -> all roles)
  __shared__ __align__(8) uint64_t sbars[8];  // queue slot k filled (scheduler -> producer)
  __shared__ __align__(8) uint64_t gbars[8];  // claim the next tile now (producer -> scheduler)
  __shared__ int z0_slot[2][TOKMAX];          // a GEMM2 tile's destination rows (route slots), double-buffered
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
  const uint32_t sb = (raw + 1023) & ~1023u;
  uint8_t* const sp0 = mraw + (sb - raw);
  __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
  const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, NW);
    }
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&sbars[s]), 1);
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&gbars[s]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid >= NC && tid < NC + 12) {  // tensor maps (kernel parameters, not predecessor output): before the PDL wait
    const CUtensorMap* pm = reinterpret_cast<const CUtensorMap*>(&mp) + (tid - NC);
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(pm)) : "memory");
  }
  __syncthreads();
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  const uint32_t bsl = (uint32_t)__cvta_generic_to_shared(sbars);
  const uint32_t bgo = (uint32_t)__cvta_generic_to_shared(gbars);
  if (warp >= NW) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    GDC_WAIT();
    const int total = __ldg(meta + m_toff1(E) + E) + __ldg(meta + m_toff2(E) + E);
    if (warp == NW) {    // producer warp: all 32 lanes stay converged; one elected lane issues each copy
      uint32_t gch = 0;  // chunks (stages) issued; unsigned so % NS and / NS are a mask and a shift
      const uint64_t pol = pol_evict_first();
      for (int k = 0;; k++) {
        int t, mode, e, rb, row0, nv, jt, W;
        if (k == 0) {
          TileInfo o{-1, 0, 0, 0, 0, 0, 0};
          int t0 = 0;
          if (lane == 0) t0 = atomicAdd(counter, 1);
          t0 = __shfl_sync(0xffffffffu, t0, 0);
          if (t0 < total) o = locate_warp(meta, S, t0, -1);
          t = o.t;
          mode = o.mode;
          e = o.e;
          rb = o.rb;
          row0 = o.row0;
          nv = o.nv;
          jt = o.jt;
          W = t >= 0 ? mx_w(nv) : 0;
          if (lane == 0) {
            s_tq[0] = t;
            s_tl[0] = tl_pack(o, W);
            s_tj[0] = jt;
          }
        } else {
          mbar_wait(bsl + 8 * (k & 7), ((k >> 3) - ((k & 7) == 0 ? 1 : 0)) & 1);
          t = s_tq[k & 7];
          const int4 tl0 = s_tl[k & 7];
          e = tl0.x;
          rb = tl0.y & 0xFFFF;
          mode = (tl0.y >> 16) & 15;
          W = tl0.y >> 20;
          row0 = tl0.z;
          nv = tl0.w;
          jt = s_tj[k & 7];
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(btq + 8 * (k & 7));
        if (t < 0) break;
        const int C = mode == 1 ? S.C1 : S.C2, RBm = mode == 1 ? S.RB1 : S.RB2;
        const CUtensorMap* wm = &mp.w[mode - 1];
        const int wr0 = ((e * RBm + rb) * C) * WROWS;
        if (mode == 2) {  // this token tile's intermediate rows: wait for its RB1 GEMM1 row blocks
          while (ld_acquire(&cnt1[jt]) < S.RB1)
            __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        if (W <= 64) {
          // Narrow tile: two chunks per stage, so the ring holds twice as many of these short, load-heavy chunks.
          const CUtensorMap* bm = W <= 32 ? &mp.xn[mode - 1][W == 8 ? 0 : W == 16 ? 1 : 2] : &mp.x64[mode - 1];
          const int ex = 2 * (W * 128 + WB2);
          const int cgo =
              C >= 8 ? C - 8 : 0;  // 4 stages from the end (or the first): the scheduler may claim tile k + 1
          for (int c = 0; c < C; c += 2, gch++) {
            const int s = gch % NS;
            if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (elect_one()) {
              mbar_expect(bfull + 8 * s, ex);
#pragma unroll
              for (int j = 0; j < 2; j++) {
                tma_2d_h(sb + s * SB + j * MX_PAIR + 8192, wm, 0, wr0 + (c + j) * WROWS, bfull + 8 * s, pol);
                tma_2d(sb + s * SB + j * MX_PAIR, bm, (c + j) * 64, row0, bfull + 8 * s);
              }
              if (c == cgo) mbar_arrive(bgo + 8 * (k & 7));
            }
          }
        } else {
          // W 128: one 128-row box of token rows and one chunk of weights per stage.
          const CUtensorMap* bmw = &mp.x128[mode - 1];
          const int ex = 128 * 128 + WB2;
          if ((gch % NS) == 0 && (C % NS) == 0) {  // stage slots known at compile time: unroll by NS
            for (int c = 0; c < C; c += NS, gch += NS) {
              const uint32_t par = ((gch / NS) - 1) & 1;
#pragma unroll
              for (int s = 0; s < NS; s++) {
                if (gch >= NS) mbar_wait(bempty + 8 * s, par);
                if (elect_one()) {
                  mbar_expect(bfull + 8 * s, ex);
                  tma_2d_h(sb + s * SB + XOFF, wm, 0, wr0 + (c + s) * WROWS, bfull + 8 * s, pol);
                  tma_2d(sb + s * SB, bmw, (c + s) * 64, row0, bfull + 8 * s);
                  if (s == 0 && c == C - NS) mbar_arrive(bgo + 8 * (k & 7));
                }
              }
            }
          } else {
            const int cgo = C >= 4 ? C - 4 : 0;
            for (int c = 0; c < C; c++, gch++) {
              const int s = gch % NS;
              if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
              if (elect_one()) {
                mbar_expect(bfull + 8 * s, ex);
                tma_2d_h(sb + s * SB + XOFF, wm, 0, wr0 + c * WROWS, bfull + 8 * s, pol);
                tma_2d(sb + s * SB, bmw, c * 64, row0, bfull + 8 * s);
                if (c == cgo) mbar_arrive(bgo + 8 * (k & 7));
              }
            }
          }
        }
      }
    } else {  // copy-out warps (copy_out); the first of them is also the scheduler
      const int ct = tid - (NC + 32);
      for (int k = 0;; k++) {
        mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
        const int t = s_tq[k & 7];
        if (t < 0) break;
        if (warp == NW + 1) {  // the scheduler warp claims, locates and publishes tile k + 1
          const int kk = k + 1;
          mbar_wait(bgo + 8 * (k & 7), (k >> 3) & 1);  // claim late: once the producer is 4 stages from tile k's end
          int t_ = 0;
          if (lane == 0) t_ = atomicAdd(counter, 1);
          t_ = __shfl_sync(0xffffffffu, t_, 0);
          TileInfo o{-1, 0, 0, 0, 0, 0, 0};
          if (t_ < total) o = locate_warp(meta, S, t_, -1);
          if (lane == 0) {
            s_tq[kk & 7] = o.t;
            s_tl[kk & 7] = tl_pack(o, o.t >= 0 ? mx_w(o.nv) : 0);
            s_tj[kk & 7] = o.jt;
            mbar_arrive(bsl + 8 * (kk & 7));
          }
          __syncwarp();
        }
        copy_out(s_tl[k & 7], s_tj[k & 7], ct, z0_slot[k & 1], obuf, out1, out2, slot_of, cnt1, S);
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  GDC_WAIT();
  uint32_t gch = 0;
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int e = tl.x, mode = (tl.y >> 16) & 15, wv = tl.y >> 20, ntok = tl.w;
    const int C = mode == 1 ? S.C1 : S.C2;
    // One tile at MMA width W on its own accumulator array (wgmmas of different shapes writing one array are
    // serialized by ptxas, warning C7512).
    auto tile = [&](auto Wc) {
      constexpr int W = decltype(Wc)::value, MB = 2;
      const int ubase = (warp < 4 ? 0 : 8) + (warp % 4);
      constexpr int AW = (W + 15) / 16 * 16;  // W = 8 stages a pair of n8 blocks, the extra one stays 0
      float acc[MB][AW / 2];
      uint32_t ra[MB][16], rb_[MB][16];
#pragma unroll
      for (int b = 0; b < MB; b++)
#pragma unroll
        for (int i = 0; i < AW / 2; i++)
          acc[b][i] = 0.f;
      // Full-width (W 128) chunk at MMA width NWc on the W 128 accumulators. At full width the stage is freed one
      // chunk late (wait 1); a narrower MMA waits for its own chunk and frees its stage at once.
      auto step = [&](auto NWc, int c, uint32_t (&r)[MB][16]) {
        constexpr int NWv = decltype(NWc)::value;
        const int s = gch % NS;
        mbar_wait(bfull + 8 * s, (gch / NS) & 1);
        if constexpr (W == 128) chunk_mma<NWv, 128>(acc, r, sp0 + s * SB + XOFF, ubase, lane, sb + s * SB);
        wg_commit();
        if constexpr (NWv == W) {
          wg_wait<1>();
          if (c > 0) {
            __syncwarp();
            if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
          }
        } else {
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * s);
        }
        gch++;
      };
      // Narrow pair loop (W 32 / 64) at MMA width NWc: both chunks of a stage, then wait 0 and release the stage.
      auto pairs = [&](auto NWc) {
        constexpr int NWv = decltype(NWc)::value;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
          chunk_mma<NWv, W>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
          wg_commit();
          chunk_mma<NWv, W>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
          wg_commit();
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * s);
        }
      };
      if constexpr (W < 32) {
        // At most MX_MS tokens: mma.sync m16n8k16 per n8 block on the same decoded A fragments, B fragments by
        // ldmatrix from the TMA-swizzled token rows.
        constexpr int NT = W / 8;
        const int lrow = lane & 7, lblk = lane >> 3;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const uint32_t xr = sb + s * SB + j * MX_PAIR + lrow * 128;
            uint32_t bf[NT][8];
#pragma unroll
            for (int n = 0; n < NT; n++) {
              const uint32_t xn = xr + n * 1024;  // token rows 8 n + lrow: the 128B swizzle repeats every 8 rows
              ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
              ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
            }
            const uint8_t* wb = sp0 + s * SB + j * MX_PAIR + 8192;
#pragma unroll
            for (int b = 0; b < 2; b++) {
              const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
              const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
              const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
              for (int kk = 0; kk < 4; kk++) {
                const uint32_t w = kk == 0 ? nb.x : kk == 1 ? nb.y : kk == 2 ? nb.z : nb.w;
                uint32_t o[4];
                decode4(w, o);
                const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                               a3 = bmulh(o[3], Sc, 1);
#pragma unroll
                for (int n = 0; n < NT; n++)
                  mma16816(&acc[b][4 * n], a0, a1, a2, a3, bf[n][2 * kk], bf[n][2 * kk + 1]);
              }
            }
          }
          __syncwarp();  // mma.sync read this stage synchronously, so it is free now
          // The last ldmatrix can still be outstanding here (ptxas may issue the arrive before the HMMA that
          // consumes it); the proxy fence makes the release wait for the reads.
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (lane == 0) mbar_arrive(bempty + 8 * s);
        }
      } else if constexpr (W == 64) {
        if (ntok <= 40)
          pairs(IC<40>{});
        else if (ntok <= 48)
          pairs(IC<48>{});
        else if (ntok <= 56)
          pairs(IC<56>{});
        else
          pairs(IC<64>{});
      } else if constexpr (W == 32) {
        if (ntok <= 24)
          pairs(IC<24>{});
        else
          pairs(IC<32>{});
      } else {
        const int nw = ntok <= 72    ? 72
                       : ntok <= 80  ? 80
                       : ntok <= 88  ? 88
                       : ntok <= 96  ? 96
                       : ntok <= 104 ? 104
                       : ntok <= 112 ? 112
                       : ntok <= 120 ? 120
                                     : 128;
        if (nw == 80) {
          for (int c = 0; c < C; c += 2) {
            step(IC<80>{}, c, ra);
            step(IC<80>{}, c + 1, rb_);
          }
        } else if (nw == 96) {
          for (int c = 0; c < C; c += 2) {
            step(IC<96>{}, c, ra);
            step(IC<96>{}, c + 1, rb_);
          }
        } else if (nw == 112) {
          for (int c = 0; c < C; c += 2) {
            step(IC<112>{}, c, ra);
            step(IC<112>{}, c + 1, rb_);
          }
        } else if (nw == 72) {
          for (int c = 0; c < C; c += 2) {
            step(IC<72>{}, c, ra);
            step(IC<72>{}, c + 1, rb_);
          }
        } else if (nw == 88) {
          for (int c = 0; c < C; c += 2) {
            step(IC<88>{}, c, ra);
            step(IC<88>{}, c + 1, rb_);
          }
        } else if (nw == 104) {
          for (int c = 0; c < C; c += 2) {
            step(IC<104>{}, c, ra);
            step(IC<104>{}, c + 1, rb_);
          }
        } else if (nw == 120) {
          for (int c = 0; c < C; c += 2) {
            step(IC<120>{}, c, ra);
            step(IC<120>{}, c + 1, rb_);
          }
        } else {
          for (int c = 0; c < C; c += 2) {
            step(IC<128>{}, c, ra);
            step(IC<128>{}, c + 1, rb_);
          }
          // only the full-width step's lagged release leaves the tile's last stage to free here
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
        }
      }
      wg_wait<0>();
      __syncwarp();
      stage_tile<AW>(acc, __ldg((mode == 1 ? gs1 : gs2) + e), k, mode, warp, lane, obuf);
    };
    if (wv == 8)
      tile(IC<8>{});
    else if (wv == 16)
      tile(IC<16>{});
    else if (wv == 32)
      tile(IC<32>{});
    else if (wv == 64)
      tile(IC<64>{});
    else
      tile(IC<128>{});
  }
}

// grouped_k on clusters of two CTAs (launched for 16384+ routes). The pair works on the same token tile: CTA rank r
// takes weight row blocks 2 q + r, so every claim covers two tiles (both GEMMs' row block counts are even). Rank 0
// claims pairs from the counter and forwards each claim to rank 1. The token rows are multicast: each CTA issues half
// of a stage's token boxes (or rank 0 the single box) to both CTAs, and loads its own weights. A stage is free again
// only when the consumers of both CTAs are done with it (empty barriers count 2 NW arrives).
__global__ void __launch_bounds__((NW + 4) * 32, 1) pair_k(
    const __grid_constant__ Maps mp,
    int* __restrict__ meta,
    const float* __restrict__ gs1,
    const float* __restrict__ gs2,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    const int64_t* __restrict__ slot_of,
    const Shape S) {
  constexpr int NC = NW * 32, SB = MX_SB, NS = MX_NS;
  const int E = S.E;
  int* const counter = meta + m_ctr(E);
  int* const cnt1 = meta + m_cnt1(E);
  extern __shared__ uint8_t mraw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ int s_tq[8];
  __shared__ int4 s_tl[8];
  __shared__ int s_tj[8];
  __shared__ __align__(8) uint64_t tbars[8];
  __shared__ int z0_slot[2][TOKMAX];
  __shared__ int s_claim[8];                  // claims forwarded by rank 0 (rank 1 reads them)
  __shared__ __align__(8) uint64_t cbars[8];  // claim slot filled
  uint32_t crank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
  const uint32_t sb = (raw + 1023) & ~1023u;
  uint8_t* const sp0 = mraw + (sb - raw);
  __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
  const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, 2 * NW);
    }
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
    for (int i = 0; i < 8; i++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&cbars[i]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid >= NC && tid < NC + 14) {
    const CUtensorMap* pm = reinterpret_cast<const CUtensorMap*>(&mp) + (tid - NC);
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(pm)) : "memory");
  }
  __syncthreads();
  cluster_sync();
  GDC_WAIT();
  const uint32_t peer_empty = cluster_map(bempty, crank ^ 1u);
  const int total = __ldg(meta + m_toff1(E) + E) + __ldg(meta + m_toff2(E) + E);
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  if (warp >= NW) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    if (warp == NW) {  // producer warp, converged; it also claims (rank 0) or receives (rank 1) the tile pairs
      uint32_t gch = 0;
      const uint32_t pclaim = cluster_map((uint32_t)__cvta_generic_to_shared(s_claim), crank ^ 1u);
      const uint32_t pcbar = cluster_map((uint32_t)__cvta_generic_to_shared(cbars), crank ^ 1u);
      const uint32_t lcbar = (uint32_t)__cvta_generic_to_shared(cbars);
      int next_pair = -1;
      for (int k = 0;; k++) {
        int pair;
        if (crank == 0) {
          auto claim = [&](int kk) {
            int p = 0;
            if (lane == 0) {
              p = atomicAdd(counter, 1);
              if (2 * p >= total) p = -1;
              st_shared_cluster(pclaim + 4 * (kk & 7), p);
              mbar_arrive_release_cluster(pcbar + 8 * (kk & 7));
            }
            return __shfl_sync(0xffffffffu, p, 0);
          };
          if (k == 0) next_pair = claim(0);  // one claim ahead, so rank 1 never waits on rank 0's issue loop
          pair = next_pair;
          if (pair >= 0) next_pair = claim(k + 1);
        } else {
          mbar_wait_acquire_cluster(lcbar + 8 * (k & 7), (k >> 3) & 1);
          pair = *reinterpret_cast<volatile int*>(&s_claim[k & 7]);
        }
        TileInfo o{-1, 0, 0, 0, 0, 0, 0};
        if (pair >= 0) o = locate_warp(meta, S, 2 * pair, (int)crank);
        const int W = o.t >= 0 ? mx_w(o.nv) : 0;
        if (lane == 0) {
          s_tq[k & 7] = o.t;
          s_tl[k & 7] = tl_pack(o, W);
          s_tj[k & 7] = o.jt;
          mbar_arrive(btq + 8 * (k & 7));
        }
        __syncwarp();
        if (o.t < 0) break;
        const int mode = o.mode, C = mode == 1 ? S.C1 : S.C2, RBm = mode == 1 ? S.RB1 : S.RB2;
        const int wr0 = ((o.e * RBm + o.rb) * C) * WROWS;
        if (mode == 2) {
          while (ld_acquire(&cnt1[o.jt]) < S.RB1)
            __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        if (W <= 64) {
          // Narrow tile: two chunks per stage (x0 at 0, w0 w1 at 8192 as one box, x1 at PAIR_X1). Rank r multicasts
          // chunk c + r's token rows.
          const CUtensorMap* bm = W <= 32 ? &mp.xn[mode - 1][W == 8 ? 0 : W == 16 ? 1 : 2] : &mp.x64[mode - 1];
          const int ex = 2 * (W * 128 + WB2);
          for (int c = 0; c < C; c += 2, gch++) {
            const int s = gch % NS;
            if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (elect_one()) {
              mbar_expect(bfull + 8 * s, ex);
              tma_2d_mc(
                  sb + s * SB + (crank ? PAIR_X1 : 0u), bm, (c + (int)crank) * 64, o.row0, bfull + 8 * s, (uint16_t)3);
              tma_2d(sb + s * SB + 8192, &mp.w2c[mode - 1], 0, wr0 + c * WROWS, bfull + 8 * s);  // both chunks' weights
            }
          }
        } else {
          for (int c = 0; c < C; c++, gch++) {
            const int s = gch % NS;
            if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (elect_one()) {
              mbar_expect(bfull + 8 * s, 128 * 128 + WB2);
              tma_2d(sb + s * SB + XOFF, &mp.w[mode - 1], 0, wr0 + c * WROWS, bfull + 8 * s);  // own weights first
              if (crank == 0) tma_2d_mc(sb + s * SB, &mp.x128[mode - 1], c * 64, o.row0, bfull + 8 * s, (uint16_t)3);
            }
          }
        }
      }
    } else {
      const int ct = tid - (NC + 32);
      for (int k = 0;; k++) {
        mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
        const int t = s_tq[k & 7];
        if (t < 0) break;
        copy_out(s_tl[k & 7], s_tj[k & 7], ct, z0_slot[k & 1], obuf, out1, out2, slot_of, cnt1, S);
      }
    }
    cluster_sync();
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
  uint32_t gch = 0;
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int e = tl.x, mode = (tl.y >> 16) & 15, wv = tl.y >> 20, ntok = tl.w;
    const int C = mode == 1 ? S.C1 : S.C2;
    auto tile = [&](auto Wc) {
      constexpr int W = decltype(Wc)::value, MB = 2;
      const int ubase = (warp < 4 ? 0 : 8) + (warp % 4);
      constexpr int AW = (W + 15) / 16 * 16;
      float acc[MB][AW / 2];
      uint32_t ra[MB][16], rb_[MB][16];
#pragma unroll
      for (int b = 0; b < MB; b++)
#pragma unroll
        for (int i = 0; i < AW / 2; i++)
          acc[b][i] = 0.f;
      auto release = [&](int s) {
        if (lane == 0) {
          mbar_arrive(bempty + 8 * s);
          mbar_arrive_cluster(peer_empty + 8 * s);
        }
      };
      auto step = [&](auto NWc, int c, uint32_t (&r)[MB][16]) {
        constexpr int NWv = decltype(NWc)::value;
        const int s = gch % NS;
        mbar_wait(bfull + 8 * s, (gch / NS) & 1);
        if constexpr (W == 128) chunk_mma<NWv, 128>(acc, r, sp0 + s * SB + XOFF, ubase, lane, sb + s * SB);
        wg_commit();
        if constexpr (NWv == W) {
          wg_wait<1>();
          if (c > 0) {
            __syncwarp();
            release((gch - 1) % NS);
          }
        } else {
          wg_wait<0>();
          __syncwarp();
          release(s);
        }
        gch++;
      };
      auto pairs = [&](auto NWc) {
        constexpr int NWv = decltype(NWc)::value;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
          chunk_mma<NWv, W>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
          wg_commit();
          chunk_mma<NWv, W>(acc, rb_, sp0 + s * SB + PAIR_W1, ubase, lane, sb + s * SB + PAIR_X1);
          wg_commit();
          wg_wait<0>();
          __syncwarp();
          release(s);
        }
      };
      if constexpr (W < 32) {
        constexpr int NT = W / 8;
        const int lrow = lane & 7, lblk = lane >> 3;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const uint32_t xr = sb + s * SB + j * PAIR_X1 + lrow * 128;  // chunk j's token rows
            uint32_t bf[NT][8];
#pragma unroll
            for (int n = 0; n < NT; n++) {
              const uint32_t xn = xr + n * 1024;
              ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
              ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
            }
            const uint8_t* wb = sp0 + s * SB + 8192 + j * (PAIR_W1 - 8192);  // chunk j's weights
#pragma unroll
            for (int b = 0; b < 2; b++) {
              const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
              const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
              const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
              for (int kk = 0; kk < 4; kk++) {
                const uint32_t w = kk == 0 ? nb.x : kk == 1 ? nb.y : kk == 2 ? nb.z : nb.w;
                uint32_t o[4];
                decode4(w, o);
                const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                               a3 = bmulh(o[3], Sc, 1);
#pragma unroll
                for (int n = 0; n < NT; n++)
                  mma16816(&acc[b][4 * n], a0, a1, a2, a3, bf[n][2 * kk], bf[n][2 * kk + 1]);
              }
            }
          }
          __syncwarp();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          release(s);
        }
      } else if constexpr (W == 64) {
        if (ntok <= 40)
          pairs(IC<40>{});
        else if (ntok <= 48)
          pairs(IC<48>{});
        else if (ntok <= 56)
          pairs(IC<56>{});
        else
          pairs(IC<64>{});
      } else if constexpr (W == 32) {
        if (ntok <= 24)
          pairs(IC<24>{});
        else
          pairs(IC<32>{});
      } else {
        const int nw = ntok <= 80 ? 80 : ntok <= 96 ? 96 : ntok <= 112 ? 112 : 128;
        if (nw == 80) {
          for (int c = 0; c < C; c += 2) {
            step(IC<80>{}, c, ra);
            step(IC<80>{}, c + 1, rb_);
          }
        } else if (nw == 96) {
          for (int c = 0; c < C; c += 2) {
            step(IC<96>{}, c, ra);
            step(IC<96>{}, c + 1, rb_);
          }
        } else if (nw == 112) {
          for (int c = 0; c < C; c += 2) {
            step(IC<112>{}, c, ra);
            step(IC<112>{}, c + 1, rb_);
          }
        } else {
          for (int c = 0; c < C; c += 2) {
            step(IC<128>{}, c, ra);
            step(IC<128>{}, c + 1, rb_);
          }
          wg_wait<0>();
          __syncwarp();
          release((gch - 1) % NS);
        }
      }
      wg_wait<0>();
      __syncwarp();
      stage_tile<AW>(acc, __ldg((mode == 1 ? gs1 : gs2) + e), k, mode, warp, lane, obuf);
    };
    if (wv == 8)
      tile(IC<8>{});
    else if (wv == 16)
      tile(IC<16>{});
    else if (wv == 32)
      tile(IC<32>{});
    else if (wv == 64)
      tile(IC<64>{});
    else
      tile(IC<128>{});
  }
  cluster_sync();
}

// ---- route list kernels ----
// Per-CTA histograms of the local routes' experts (ids in [0, E)), added into meta's counts (meta comes zeroed).
__global__ void __launch_bounds__(1024) count_k(const int* __restrict__ ids, int R, int E, int* __restrict__ meta) {
  extern __shared__ int c[];
  const int tid = threadIdx.x, lane = tid % 32;
  for (int i = tid; i < E; i += blockDim.x)
    c[i] = 0;
  __syncthreads();
  for (int base = blockIdx.x * blockDim.x; base < R; base += gridDim.x * blockDim.x) {
    const int i = base + tid;
    const int e = i < R ? ids[i] : -1;
    const bool loc = e >= 0 && e < E;
    const unsigned m = __match_any_sync(0xffffffffu, loc ? e : -1);
    if (loc && lane == __ffs(m) - 1) atomicAdd(&c[e], __popc(m));
  }
  __syncthreads();
  for (int i = tid; i < E; i += blockDim.x)
    if (c[i]) atomicAdd(&meta[i], c[i]);
}
// Exclusive prefix sum over the block (every thread calls it); total gets the block's sum.
__device__ __forceinline__ int block_excl_scan(int v, int* sw, int& total) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) sw[wid] = x;
  __syncthreads();
  if (wid == 0) {
    int w = lane < nw ? sw[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += y;
    }
    if (lane < nw) sw[lane] = w;
  }
  __syncthreads();
  const int pre = wid ? sw[wid - 1] : 0;
  total = sw[nw - 1];
  __syncthreads();
  return pre + x - v;
}
// The plan: per expert its token tiling (ntt tiles of tsz tokens, tsz a multiple of 32 and at most 128), the experts
// by route count (most first, ties by id: the longest tiles are claimed first, the last claims are short), starts and
// scatter cursors by id, and by plan position the info word, both GEMMs' tile offsets and the token tile base.
// One block of at least E threads (a multiple of 32), with sh = [2 E + 32] ints whose first E hold the route counts.
__device__ __forceinline__ void plan_block(int* __restrict__ meta, int* sh, int E, int rb1, int rb2) {
  int* const scnt = sh;        // [E] route counts by id
  int* const sperm = sh + E;   // [E] experts by plan position
  int* const sw = sh + 2 * E;  // [32] scan scratch
  const int e = threadIdx.x;
  const bool act = e < E;
  const int n = act ? scnt[e] : 0;
  if (act) {
    int pos = 0;
    for (int j = 0; j < E; j++) {
      const int cj = scnt[j];
      pos += cj > n || (cj == n && j < e);
    }
    sperm[pos] = e;
    meta[m_perm(E) + pos] = e;
  }
  int tot;
  const int st = block_excl_scan(n, sw, tot);
  if (act) {
    meta[m_start(E) + e] = st;
    meta[m_cur(E) + e] = st;
  }
  // by plan position e: the expert pe
  int pn = 0, info = 0;
  if (act) {
    const int pc = scnt[sperm[e]];
    if (pc > 0) {
      pn = (pc + TOKMAX - 1) / TOKMAX;
      const int tsz = ((pc + pn - 1) / pn + 31) / 32 * 32;
      info = pn | tsz / 32 << INFO_NTT_BITS;
    }
  }
  int tot1, tot2, totj;
  const int t1 = block_excl_scan(pn * rb1, sw, tot1);
  const int t2 = block_excl_scan(pn * rb2, sw, tot2);
  const int j0 = block_excl_scan(pn, sw, totj);
  if (act) {
    meta[m_toff1(E) + e] = t1;
    meta[m_toff2(E) + e] = t2;
    meta[m_info(E) + e] = info;
    meta[m_jt0(E) + e] = j0;
  }
  if (e == 0) {
    meta[m_toff1(E) + E] = tot1;
    meta[m_toff2(E) + E] = tot2;
    meta[m_ctr(E)] = 0;
    meta[m_nlcur(E)] = tot;  // other ranks' routes go after the tot local ones
  }
}
__global__ void plan_k(int* __restrict__ meta, int E, int rb1, int rb2) {
  extern __shared__ int sh[];
  if (threadIdx.x < E) sh[threadIdx.x] = meta[threadIdx.x];
  __syncthreads();
  plan_block(meta, sh, E, rb1, rb2);
}
// Route ids to their expert-sorted positions (warp-aggregated atomics; the order within an expert is arbitrary, each
// route is computed alone), and other ranks' routes after all local ones.
__device__ __forceinline__ void
scatter_routes(const int* __restrict__ ids, int R, int E, int64_t* __restrict__ order, int* __restrict__ meta) {
  const int lane = threadIdx.x % 32;
  for (int base = blockIdx.x * blockDim.x; base < R; base += gridDim.x * blockDim.x) {
    const int i = base + threadIdx.x;
    const int e = i < R ? ids[i] : -1;
    const bool loc = e >= 0 && e < E;
    const int key = loc ? e : i < R ? -1 : -2;  // an expert, another rank, or past the end
    const unsigned m = __match_any_sync(0xffffffffu, key);
    const int leader = __ffs(m) - 1;
    int p = 0;
    if (key != -2 && lane == leader) p = atomicAdd(&meta[loc ? m_cur(E) + e : m_nlcur(E)], __popc(m));
    p = __shfl_sync(0xffffffffu, p, leader);
    if (key != -2) order[p + __popc(m & ((1u << lane) - 1))] = i;
  }
}
__global__ void __launch_bounds__(1024)
    scatter_k(const int* __restrict__ ids, int R, int E, int64_t* __restrict__ order, int* __restrict__ meta) {
  scatter_routes(ids, R, E, order, meta);
}
// count_k, plan_k and scatter_k in one block for small calls (decode), zeroing meta itself: one launch instead of a
// memset and three.
__global__ void __launch_bounds__(1024) front_small_k(
    const int* __restrict__ ids,
    int R,
    int E,
    int rb1,
    int rb2,
    int64_t* __restrict__ order,
    int* __restrict__ meta,
    int meta_words) {
  extern __shared__ int sh[];
  for (int i = threadIdx.x; i < meta_words; i += blockDim.x)
    meta[i] = 0;
  for (int i = threadIdx.x; i < E; i += blockDim.x)
    sh[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < R; i += blockDim.x) {
    const int e = ids[i];
    if (e >= 0 && e < E) atomicAdd(&sh[e], 1);
  }
  __syncthreads();
  if (threadIdx.x < E) meta[threadIdx.x] = sh[threadIdx.x];
  plan_block(meta, sh, E, rb1, rb2);
  __syncthreads();
  scatter_routes(ids, R, E, order, meta);
}
// The routed rows of x in sorted order (token = route / topk), each 64-column chunk in the K order of the packed
// weights: position p = 16 kk + 8 half + 2 q + j holds column 16 q + 4 kk + 2 half + j, so (in 32-bit words of two
// columns) output word 8 kk + 4 half + q is input word 8 q + 2 kk + half, and output uint4 i of a chunk is input
// words i, 8 + i, 16 + i, 24 + i. One CTA per row, one thread per output uint4: the stores are contiguous and each
// load instruction reads whole 32B sectors (8 lanes x 4B per chunk). Sorted positions from n on are other ranks'
// routes: their output rows get zeros, as SGLang's Triton MoE writes for routes whose expert is not local, so the
// caller's weighted sum over a token's routes never reads stale bytes. combine_k needs no zeros (it skips those
// routes). (Zeroing them in the GEMM kernel's idle copier warps instead hides part of these stores at 2,048+ tokens but
// slows the GEMM kernel itself by 2% to 12%.)
__global__ void __launch_bounds__(256) gather_k(
    const uint32_t* __restrict__ x,
    const int64_t* __restrict__ order,
    const int* __restrict__ meta,
    int E,
    int R,
    uint4* __restrict__ a,
    uint4* __restrict__ y,
    int H,
    int topk) {
  GDC_WAIT();                                    // launched as a programmatic dependent of the plan
  if constexpr (NVFP4_MOE_DEBUG_DELAY_NS > 0) {  // test builds: let the GEMM kernel start before the rows exist
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    debug_delay();
  }
  const int n = meta[m_start(E) + E - 1] + meta[E - 1];
  const int H8 = H / 8, H2 = H / 2;
  // Without y (the caller takes the weighted sum, which skips other ranks' routes) only the local rows are visited.
  const int rows = y != nullptr ? R : n;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    if (r < n) {
      const uint32_t* in = x + order[r] / topk * H2;
      for (int c = threadIdx.x; c < H8; c += blockDim.x) {
        const uint32_t* p = in + (c >> 3) * 32 + (c & 7);
        a[(int64_t)r * H8 + c] = make_uint4(__ldg(p), __ldg(p + 8), __ldg(p + 16), __ldg(p + 24));
      }
    } else {
      uint4* const yr = y + order[r] * H8;
      for (int c = threadIdx.x; c < H8; c += blockDim.x)
        yr[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

// out[t] = scale * sum over the local routes k of t (ids in [0, E)) of w[t, k] y[t top_k + k], in increasing k. Rows
// of y for other routes are never read. One CTA per token, one thread per 8 columns.
__global__ void __launch_bounds__(256) combine_k(
    const uint4* __restrict__ y,
    const int* __restrict__ ids,
    const float* __restrict__ w,
    int T,
    int topk,
    int E,
    int H8,
    float scale,
    uint4* __restrict__ out) {
  GDC_WAIT();  // launched as a programmatic dependent of the GEMMs
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    for (int c = threadIdx.x; c < H8; c += blockDim.x) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < topk; k0 += 8) {  // eight routes' loads in flight
        uint4 v[8];
        float wk[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const int k = k0 + j, e = k < topk ? ids[t * topk + k] : -1;
          wk[j] = e >= 0 && e < E ? w[t * topk + k] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
          v[j] = wk[j] != 0.f ? y[(int64_t)(t * topk + k0 + j) * H8 + c] : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t u[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[i]));
            acc[2 * i] = fmaf(wk[j], f.x, acc[2 * i]);
            acc[2 * i + 1] = fmaf(wk[j], f.y, acc[2 * i + 1]);
          }
        }
      }
      out[(int64_t)t * H8 + c] = make_uint4(
          pkbf(acc[0] * scale, acc[1] * scale),
          pkbf(acc[2] * scale, acc[3] * scale),
          pkbf(acc[4] * scale, acc[5] * scale),
          pkbf(acc[6] * scale, acc[7] * scale));
    }
  }
}

// ---- decode path (dec_k): the whole call in one launch for few tokens ----
// At decode sizes the time is the selected experts' weight bytes, and grouped_k leaves most SMs idle behind its
// routing kernels (3 experts x 48 row blocks at one token on the Muse shape). dec_k computes the routing in every CTA
// (the ids are at most a few KB), then streams the weights from every SM at once:
//   * Slots are the experts with local routes, in increasing id. Batches are runs of consecutive slots holding at
//     least `thr` GEMM1 chunks in all (one expert per batch on large experts). Every CTA executes, in this order,
//     GEMM1 of batch 0, GEMM1 of batch 1, GEMM2 of batch 0, GEMM1 of batch 2, GEMM2 of batch 1, ..., GEMM2 of the last
//     batch, so a batch's GEMM2 starts one batch after its GEMM1 and never waits in steady state.
//   * Each of those steps is one GEMM of one batch, its 9216B chunks (one 256-row block x 64 k, the pack's layout)
//     flattened slot-major, row-block-major, chunk-minor and cut into one contiguous range per CTA. A range covers
//     the tail of one row block and the head of the next ("pieces"), so the K sum of a row block is split over the
//     CTAs whose ranges touch it.
//   * Warp NW (producer) copies the CTA's chunks in that order into a ring of DEC_NS stages with 1D bulk copies and
//     never waits for anything but a free stage: GEMM2's weights do not depend on GEMM1, so they land while GEMM1
//     finishes. Warps 0 .. NW - 1 (consumers) decode units w and w + 8 of each chunk (decode4 and the narrow tiles'
//     scale codes) and run mma.sync m16n8k16 with the slot's routes as n8 blocks; the B fragments are read straight
//     from x (GEMM1) or the intermediate (GEMM2), whose natural column order is what the pack's K order expects.
//     At a piece's end they write its FP32 partial sums and hand the piece to warp NW + 1 (finisher).
//   * The finisher counts the piece on its row block. The CTA that counts a row block's last piece sums its pieces
//     in CTA order (so the result never depends on timing), then: GEMM1 applies SiLU(gate) * up and writes 128
//     columns of the intermediate, and raises the row block's ready flag; GEMM2 writes the BF16 output rows and, with
//     the weighted sum, counts each route on its token, and the CTA that counts a token's last route writes
//     scale * sum_k w[t, k] y[t top_k + k] (increasing k) for those 256 columns.
//   * A GEMM2 chunk k reads intermediate columns [64 k, 64 k + 64), which one GEMM1 row block (k / 2) writes: the
//     consumers wait for that flag only, just before loading the chunk's B fragments.
// Every counter returns to zero by the end of the call (the last arriver resets a row block's or token's counter, and
// the last CTA to finish resets the ready flags), so `state` must be zero before the first call and is then reused.
// The launch is cooperative: all CTAs are resident, so a wait on another CTA's row block always ends.
constexpr int DEC_SC = 4;     // chunks per ring stage: one bulk copy of up to 36KB (9KB copies stream at a third of the rate)
constexpr int DEC_NS = 5;     // ring stages
constexpr int DEC_PR = 32;    // piece hand-offs in flight from the consumers to the finisher
constexpr int DEC_RMAX = 256; // routes (tokens x top_k)
constexpr int DEC_TMAX = 16;  // tokens (at most two n8 blocks of one expert's routes)
constexpr int DEC_EMAX = 1024;
constexpr int DEC_PMAX = 384;  // pieces per CTA
constexpr int DEC_BBUF = 16384;  // staged B rows
constexpr int DEC_CW = 8;                     // consumer warps
constexpr int DEC_FW = 4;                     // finisher warps (piece i goes to finisher i % DEC_FW)
constexpr int DEC_UPW = 16 / DEC_CW;          // 16-row units per consumer warp per chunk
constexpr int DEC_THREADS = (DEC_CW + 1 + DEC_FW) * 32;

struct DecPiece {
  int kind, slot, rb, c0, c1;  // GEMM (0, 1), slot, row block, chunks [c0, c1)
  int poff;                    // this CTA's partial (floats into part)
  int pfirst, pstride, np;     // the row block's first partial, the stride between its pieces' partials, its pieces
  int key;                     // the row block's arrival counter in state
};

struct DecArgs {
  const uint8_t* w13;
  const uint8_t* w2;
  const float* gs13;
  const float* gs2;
  const __nv_bfloat16* x;
  const int* ids;
  const float* tw;  // null: write y per route (zeros for other ranks' routes); else the weighted sum into out
  __nv_bfloat16* inter;
  __nv_bfloat16* y;
  __nv_bfloat16* out;
  float* part;
  int* state;
  float scale;
  int T, topk, E, H, I, RB1, RB2, C1, C2, thr, nmax;
  unsigned long long* trace;  // DEBUG
  int dbg;                    // DEBUG
};

constexpr int DEC_SB = DEC_SC * WB2;
constexpr int DEC_RING = DEC_NS * DEC_SB;
constexpr int DEC_SMEM = DEC_RING + (2 * DEC_NS + 2 * DEC_PR) * 8 + (8 * DEC_RMAX + DEC_EMAX + DEC_TMAX + 8) * 4 +
                         DEC_PMAX * (int)sizeof(DecPiece) + DEC_BBUF;

// state: [0] CTAs finished | arrival counters (GEMM1 row blocks, then GEMM2's) | GEMM1 ready flags | token counters
__host__ __device__ __forceinline__ int dec_rdy(int nmax, int rb1, int rb2) {
  return 1 + nmax * (rb1 + rb2);
}
__host__ __device__ __forceinline__ int dec_tcnt(int nmax, int rb1, int rb2) {
  return dec_rdy(nmax, rb1, rb2) + nmax * rb1;
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(dst),
      "l"(src),
      "r"(bytes),
      "r"(bar),
      "l"(pol)
      : "memory");
}
__device__ __forceinline__ uint4 ld_nc4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ float4 ld_cg4f(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
  int o;
  asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory");
  return o;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t u4w(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
// The CTA (of G) whose range [b W / G, (b + 1) W / G) holds unit u.
__device__ __forceinline__ int dec_owner(int64_t u, int64_t W, int G) {
  return (int)(((u + 1) * G + W - 1) / W) - 1;
}
// Natural intermediate column of gate row i (0 .. 63) of a 128-row half (pack.kperm_index).
__device__ __forceinline__ int dec_kp(int i) {
  return 16 * ((i >> 1) & 3) + 4 * (i >> 4) + 2 * ((i >> 3) & 1) + (i & 1);
}

template <int NT>
__global__ void __launch_bounds__(DEC_THREADS, 1) dec_k(const DecArgs A) {
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(dsm + DEC_RING);
  int* sids = reinterpret_cast<int*>(bars + 2 * DEC_NS + 2 * DEC_PR);  // the call's ids
  int* slot_e = sids + DEC_RMAX;                                          // per slot: expert, routes, first route
  int* slot_m = slot_e + DEC_RMAX;
  int* slot_r0 = slot_m + DEC_RMAX;
  int* routes = slot_r0 + DEC_RMAX;   // route indices by slot, increasing within a slot
  int* scnt = routes + DEC_RMAX;      // per expert: route count, then slot
  int* nloc = scnt + DEC_EMAX;        // per token: local routes
  int* misc = nloc + DEC_TMAX;        // [0] slots, [1] pieces of this CTA, [2] batches
  int* g2base = misc + 8;             // per batch: its GEMM2 step's first partial and partial stride
  int* g2stride = g2base + DEC_RMAX;
  int* rrank = g2stride + DEC_RMAX;   // per route: its position among its slot's routes
  DecPiece* pieces = reinterpret_cast<DecPiece*>(rrank + DEC_RMAX);
  uint8_t* bbuf = reinterpret_cast<uint8_t*>(pieces + DEC_PMAX);  // 16B aligned: 40B pieces x 384
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(dsm);
  const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * DEC_NS;
  const uint32_t bpiece = bempty + 8 * DEC_NS, bpfree = bpiece + 8 * DEC_PR;
  const int R = A.T * A.topk, E = A.E;
  if (A.dbg & 64) return;  // DEBUG
  if (A.trace && tid == 0) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    A.trace[b * 16] = t;
  }

  if (tid == 0) {
    for (int s = 0; s < DEC_NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, DEC_CW);
    }
    for (int s = 0; s < DEC_PR; s++) {
      mbar_init(bpiece + 8 * s, DEC_CW);
      mbar_init(bpfree + 8 * s, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < R; i += DEC_THREADS)
    sids[i] = A.ids[i];
  for (int e = tid; e < E; e += DEC_THREADS)
    scnt[e] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += DEC_THREADS)
    if ((unsigned)sids[i] < (unsigned)E) atomicAdd(&scnt[sids[i]], 1);
  __syncthreads();
  if (warp == 0) {  // slots in increasing expert id; scnt[e] becomes e's slot
    int n = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane, c = e < E ? scnt[e] : 0;
      const uint32_t bal = __ballot_sync(0xffffffffu, c > 0);
      if (c > 0) {
        const int pos = n + __popc(bal & ((1u << lane) - 1));
        slot_e[pos] = e;
        slot_m[pos] = c;
        scnt[e] = pos;
      }
      n += __popc(bal);
    }
    if (lane == 0) {
      int r0 = 0;
      for (int s = 0; s < n; s++) {
        slot_r0[s] = r0;
        r0 += slot_m[s];
      }
      misc[0] = n;
    }
  }
  __syncthreads();
  const int n = misc[0];
  for (int i = tid; i < R; i += DEC_THREADS) {
    const int e = sids[i];
    if ((unsigned)e < (unsigned)E) {
      int rank = 0;
      for (int j = 0; j < i; j++)
        rank += sids[j] == e;
      routes[slot_r0[scnt[e]] + rank] = i;
      rrank[i] = rank;
    }
  }
  for (int t = tid; t < A.T; t += DEC_THREADS) {
    int c = 0;
    for (int k = 0; k < A.topk; k++)
      c += (unsigned)sids[t * A.topk + k] < (unsigned)E;
    nloc[t] = c;
  }
  if (warp == 0) {  // this CTA's pieces, in execution order: one step per lane, offsets by warp scans
    const int nb = n == 0 ? 0 : min(n, max(1, n * A.RB1 * A.C1 / A.thr));
    int base = 0, np = 0;
    for (int q0 = 0; q0 < 2 * nb; q0 += 32) {
      const int q = q0 + lane;
      int kind = 0, j = 0, s0 = 0, RB = 1, C = 1, mm = 0, ng = 0, cnt = 0, size = 0;
      unsigned W = 1, lo = 0, hi = 0;
      if (q < 2 * nb) {
        if (q == 0) {
          kind = 0, j = 0;
        } else if (q == 2 * nb - 1) {
          kind = 1, j = nb - 1;
        } else {
          kind = q & 1 ? 0 : 1;
          j = q & 1 ? (q + 1) / 2 : (q + 1) / 2 - 1;
        }
        s0 = j * n / nb;
        const int s1 = (j + 1) * n / nb;
        RB = kind ? A.RB2 : A.RB1, C = kind ? A.C2 : A.C1;
        for (int s = s0; s < s1; s++)
          mm = max(mm, slot_m[s]);
        W = (unsigned)((s1 - s0) * RB * C);
        ng = (s1 - s0) * RB;
        lo = (unsigned)b * W / (unsigned)G, hi = (unsigned)(b + 1) * W / (unsigned)G;
        cnt = hi > lo ? (int)((hi - 1) / C - lo / C + 1) : 0;
        size = (G + ng) * 256 * mm;
      }
      int sb = size, cb = cnt;  // inclusive scans
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int a1 = __shfl_up_sync(0xffffffffu, sb, d), a2 = __shfl_up_sync(0xffffffffu, cb, d);
        if (lane >= d) sb += a1, cb += a2;
      }
      int myb = base + sb - size, myp = np + cb - cnt;
      if (kind == 1 && q < 2 * nb) g2base[j] = myb, g2stride[j] = 256 * mm;
      const int stride = 256 * mm;
      for (unsigned u = lo; u < hi;) {
        const int gr = (int)(u / C);
        const unsigned cend = min(hi, (unsigned)(gr + 1) * C);
        const unsigned g0 = (unsigned)gr * C, g1 = (unsigned)(gr + 1) * C - 1;
        const int bf = (int)(((g0 + 1) * G + W - 1) / W) - 1, bl = (int)(((g1 + 1) * G + W - 1) / W) - 1;
        if (myp >= DEC_PMAX) __trap();
        DecPiece& P = pieces[myp++];
        P.kind = kind;
        P.slot = s0 + gr / RB;
        P.rb = gr % RB;
        P.c0 = (int)(u - g0);
        P.c1 = (int)(cend - g0);
        P.poff = myb + (b + gr) * stride;
        P.pfirst = myb + (bf + gr) * stride;
        P.pstride = stride;
        P.np = bl - bf + 1;
        P.key = 1 + (kind ? A.nmax * A.RB1 + P.slot * A.RB2 : P.slot * A.RB1) + P.rb;
        u = cend;
      }
      base += __shfl_sync(0xffffffffu, sb, 31);
      np += __shfl_sync(0xffffffffu, cb, 31);
    }
    if (lane == 0) misc[1] = np, misc[2] = nb;
  }
  __syncthreads();
  const int npc = misc[1];
  if (A.dbg & 512) return;  // DEBUG
  auto gtime = []() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  };
  if (A.trace && tid == 0) A.trace[b * 16 + 1] = gtime(), A.trace[b * 16 + 7] = npc;
  int* const rdy = A.state + dec_rdy(A.nmax, A.RB1, A.RB2);
  int* const tcnt = A.state + dec_tcnt(A.nmax, A.RB1, A.RB2);

  if (warp == DEC_CW) {  // producer
    if (lane == 0) {
      const uint64_t pol = pol_evict_first();
      int k = 0;
      for (int pi = 0; pi < npc; pi++) {
        const DecPiece P = pieces[pi];
        const int e = slot_e[P.slot];
        const uint8_t* src = P.kind ? A.w2 + ((int64_t)e * A.RB2 + P.rb) * A.C2 * WB2
                                    : A.w13 + ((int64_t)e * A.RB1 + P.rb) * A.C1 * WB2;
        for (int c = P.c0; c < P.c1; c += DEC_SC, k++) {
          const int st = k % DEC_NS, bytes = min(DEC_SC, P.c1 - c) * WB2;
          if (k >= DEC_NS) mbar_wait(bempty + 8 * st, ((k / DEC_NS) + 1) & 1);
          mbar_expect(bfull + 8 * st, bytes);
          bulk_g2s(ring + st * DEC_SB, src + (int64_t)c * WB2, bytes, bfull + 8 * st, pol);
        }
      }
      if (A.trace) A.trace[b * 16 + 6] = gtime();
    }
  } else if (warp < DEC_CW) {  // consumers
    const int g = lane >> 2, q = lane & 3;
    int k = 0;
    uint64_t twait = 0, tflag = 0;
    bool g2seen = false;
    for (int pi = 0; pi < npc; pi++) {
      if (A.trace && tid == 0 && pi < 6) A.trace[b * 16 + 10 + pi] = gtime();
      const DecPiece P = pieces[pi];
      const int m = slot_m[P.slot], r0 = slot_r0[P.slot];
      const int* flags = rdy + P.slot * A.RB1;
      float acc[DEC_UPW][NT][4];
#pragma unroll
      for (int u = 0; u < DEC_UPW; u++)
#pragma unroll
        for (int nn = 0; nn < NT; nn++)
#pragma unroll
          for (int i = 0; i < 4; i++)
            acc[u][nn][i] = 0.f;
      // The piece's B rows (x for GEMM1, the intermediate for GEMM2), m tokens x up to nsub chunks of 128B at a time,
      // are staged in shared memory by all consumer threads in one round trip, so no chunk waits on an L2 load.
      const int nsub = (DEC_BBUF / m - 16) / 128 / DEC_SC * DEC_SC, rowb = nsub * 128 + 16;
      int sc0 = P.c0, sb0 = P.c0;  // next chunk to stage, first chunk of the staged rows
      for (int cs = P.c0; cs < P.c1; cs += DEC_SC) {
        if (cs == sc0) {
          const int sc1 = min(P.c1, sc0 + nsub);
          if (P.kind == 1 && !(A.dbg & 8)) {  // the GEMM1 row blocks these chunks read are written
            const uint64_t tf0 = A.trace ? gtime() : 0;
            if (A.trace && tid == 0 && !g2seen) g2seen = true, A.trace[b * 16 + 8] = tf0;
            for (int f = (sc0 >> 1) + lane; f <= (sc1 - 1) >> 1; f += 32)
              while (ld_acquire(flags + f) == 0)
                __nanosleep(32);
            if (A.trace) tflag += gtime() - tf0;
          }
          asm volatile("bar.sync 1, %0;" ::"n"(DEC_CW * 32) : "memory");  // every warp is done with the last rows
          const int per = (sc1 - sc0) * 8;
          for (int i = tid; i < m * per; i += DEC_CW * 32) {
            const int tok = i / per, cj = (i % per) >> 3, w = i & 7, route = routes[r0 + tok];
            const int64_t off = (int64_t)(sc0 + cj) * 64 + w * 8;
            const uint4 v = P.kind == 0 ? ld_nc4(A.x + (int64_t)(route / A.topk) * A.H + off)
                                        : __ldcg(reinterpret_cast<const uint4*>(A.inter + (int64_t)route * A.I + off));
            *reinterpret_cast<uint4*>(bbuf + tok * rowb + cj * 128 + w * 16) = v;
          }
          asm volatile("bar.sync 1, %0;" ::"n"(DEC_CW * 32) : "memory");
          sb0 = sc0;
          sc0 = sc1;
        }
        const int st = k % DEC_NS;
        {
          const uint64_t tw0 = A.trace ? gtime() : 0;
          mbar_wait(bfull + 8 * st, (k / DEC_NS) & 1);
          if (A.trace) { const uint64_t t1 = gtime(); twait += t1 - tw0; if (tid == 0 && k == 0) A.trace[b * 16 + 9] = t1; }
        }
#pragma unroll
        for (int j = 0; j < DEC_SC; j++) {
          if (cs + j >= P.c1 || (A.dbg & 9)) continue;
          uint4 bv[NT][2];
          const int cj = cs + j - sb0;
#pragma unroll
          for (int nn = 0; nn < NT; nn++) {
            const int tok = 8 * nn + g;
            if (tok < m) {
              const uint8_t* bp = bbuf + tok * rowb + cj * 128 + q * 32;
              bv[nn][0] = *reinterpret_cast<const uint4*>(bp);
              bv[nn][1] = *reinterpret_cast<const uint4*>(bp + 16);
            } else {
              bv[nn][0] = bv[nn][1] = make_uint4(0, 0, 0, 0);
            }
          }
          const uint8_t* wb = dsm + st * DEC_SB + j * WB2;
#pragma unroll
          for (int u = 0; u < DEC_UPW; u++) {
            const uint8_t* up = wb + (warp + DEC_CW * u) * UNIT2;
            const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
            const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
              uint32_t o[4];
              decode4(u4w(nb, kk), o);
              const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                             a3 = bmulh(o[3], Sc, 1);
#pragma unroll
              for (int nn = 0; nn < NT; nn++)
                mma16816(acc[u][nn], a0, a1, a2, a3, u4w(bv[nn][kk >> 1], 2 * (kk & 1)),
                         u4w(bv[nn][kk >> 1], 2 * (kk & 1) + 1));
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bempty + 8 * st);
        k++;
      }
      const int ps = pi % DEC_PR;
      if (pi >= DEC_PR) mbar_wait(bpfree + 8 * ps, ((pi / DEC_PR) + 1) & 1);
      float* pp = A.part + P.poff;
#pragma unroll
      for (int u = 0; u < DEC_UPW; u++) {
        const int row = 16 * (warp + DEC_CW * u) + g;
#pragma unroll
        for (int nn = 0; nn < NT; nn++) {
          const int t0 = 8 * nn + 2 * q;
          if (t0 < m) {
            pp[t0 * 256 + row] = acc[u][nn][0];
            pp[t0 * 256 + row + 8] = acc[u][nn][2];
          }
          if (t0 + 1 < m) {
            pp[(t0 + 1) * 256 + row] = acc[u][nn][1];
            pp[(t0 + 1) * 256 + row + 8] = acc[u][nn][3];
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bpiece + 8 * ps);
    }
    if (A.trace && tid == 0) A.trace[b * 16 + 2] = gtime(), A.trace[b * 16 + 3] = twait, A.trace[b * 16 + 4] = tflag;
  } else {  // finishers
    const int fw = warp - DEC_CW - 1;
    if (b == 0 && fw == 0) {  // rows nothing else writes: other ranks' routes of y, or tokens without local routes of out
      if (A.tw == nullptr) {
        for (int i = 0; i < R; i++)
          if ((unsigned)sids[i] >= (unsigned)E)
            for (int c = lane * 8; c < A.H; c += 256)
              *reinterpret_cast<uint4*>(A.y + (int64_t)i * A.H + c) = make_uint4(0, 0, 0, 0);
      } else {
        for (int t = 0; t < A.T; t++)
          if (nloc[t] == 0)
            for (int c = lane * 8; c < A.H; c += 256)
              *reinterpret_cast<uint4*>(A.out + (int64_t)t * A.H + c) = make_uint4(0, 0, 0, 0);
      }
    }
    for (int pi = fw; pi < npc; pi += DEC_FW) {
      const int ps = pi % DEC_PR;
      mbar_wait(bpiece + 8 * ps, (pi / DEC_PR) & 1);
      if (lane == 0) mbar_arrive(bpfree + 8 * ps);
      if (A.dbg & 128) continue;  // DEBUG
      const DecPiece P = pieces[pi];
      if (P.kind == 1 && A.tw != nullptr) {
        // GEMM2 with the weighted sum: count this piece's chunks on each of its routes' tokens for these 256 output
        // columns; the CTA that completes all of a token's nloc[t] * C2 chunks sums that token's routes' pieces and
        // writes its 256 columns of out. Tokens finish in different CTAs, so the sums run in parallel.
        bool done = false;
        if (lane < slot_m[P.slot]) {
          const int t = routes[slot_r0[P.slot] + lane] / A.topk, add = P.c1 - P.c0;
          int* tc = tcnt + t * A.RB2 + P.rb;
          done = atom_add_acq_rel(tc, add) + add == nloc[t] * A.C2;
          if (done) *tc = 0;
        }
        uint32_t mask = __ballot_sync(0xffffffffu, done);
        const int nbt = misc[2];
        while (mask) {
          const int jt = __ffs(mask) - 1;
          mask &= mask - 1;
          const int t = routes[slot_r0[P.slot] + jt] / A.topk;
          float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          // Four routes at a time, their pieces' loads issued together (one L2 round trip per piece index, not
          // per route), then added in route order.
          for (int k0 = 0; k0 < A.topk; k0 += 4) {
            const float* pp[4];
            int np[4], strd[4];
            float scl[4], wk[4], v[4][8];
            int maxnp = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
              np[i] = 0, pp[i] = A.part, strd[i] = 0, scl[i] = 0.f, wk[i] = 0.f;
#pragma unroll
              for (int c = 0; c < 8; c++)
                v[i][c] = 0.f;
              const int r = t * A.topk + k0 + i, e = k0 + i < A.topk ? sids[r] : -1;
              if ((unsigned)e < (unsigned)E) {
                const int s = scnt[e], j = ((s + 1) * nbt + n - 1) / n - 1;
                const int s0 = j * n / nbt, s1 = (j + 1) * n / nbt;
                const unsigned W = (unsigned)((s1 - s0) * A.RB2 * A.C2);
                const int gr = (s - s0) * A.RB2 + P.rb;
                const unsigned g0 = (unsigned)gr * A.C2, g1 = g0 + A.C2 - 1;
                const int bf = (int)(((g0 + 1) * G + W - 1) / W) - 1, bl = (int)(((g1 + 1) * G + W - 1) / W) - 1;
                pp[i] = A.part + g2base[j] + (bf + gr) * g2stride[j] + rrank[r] * 256 + 8 * lane;
                np[i] = bl - bf + 1, strd[i] = g2stride[j];
                scl[i] = __ldg(A.gs2 + e) * 9.223372036854775808e18f, wk[i] = __ldg(A.tw + r);
                maxnp = max(maxnp, np[i]);
              }
            }
#pragma unroll 2
            for (int q2 = 0; q2 < maxnp; q2++) {
#pragma unroll
              for (int i = 0; i < 4; i++) {
                if (q2 < np[i]) {
                  const float* f = pp[i] + q2 * strd[i];
                  const float4 v0 = __ldcg(reinterpret_cast<const float4*>(f));
                  const float4 v1 = __ldcg(reinterpret_cast<const float4*>(f + 4));
                  v[i][0] += v0.x, v[i][1] += v0.y, v[i][2] += v0.z, v[i][3] += v0.w;
                  v[i][4] += v1.x, v[i][5] += v1.y, v[i][6] += v1.z, v[i][7] += v1.w;
                }
              }
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
              if (np[i] > 0)
#pragma unroll
                for (int c = 0; c < 8; c++)
                  a[c] = fmaf(wk[i], __bfloat162float(__float2bfloat16(v[i][c] * scl[i])), a[c]);
          }
          const float sl = A.scale;
          *reinterpret_cast<uint4*>(A.out + (int64_t)t * A.H + 256 * P.rb + 8 * lane) =
              make_uint4(pkbf(a[0] * sl, a[1] * sl), pkbf(a[2] * sl, a[3] * sl), pkbf(a[4] * sl, a[5] * sl),
                         pkbf(a[6] * sl, a[7] * sl));
        }
        continue;
      }
      int old = 0;
      if (lane == 0) {
        old = atom_add_acq_rel(A.state + P.key, 1);
        if (old == P.np - 1) A.state[P.key] = 0;
      }
      __syncwarp();
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old != P.np - 1) continue;
      const int e = slot_e[P.slot], m = slot_m[P.slot], r0 = slot_r0[P.slot];
      if (P.kind == 0) {
        const float sc = __ldg(A.gs13 + e) * 9.223372036854775808e18f;
        const int h = lane >> 4, i0 = (lane & 15) * 4;
        const int col = 128 * P.rb + 64 * h + dec_kp(i0);
        for (int tok = 0; tok < m; tok++) {
          float4 gt = make_float4(0.f, 0.f, 0.f, 0.f), ut = gt;
          const float* pp = A.part + P.pfirst + tok * 256 + 128 * h + i0;
#pragma unroll 4
          for (int j = 0; j < P.np; j++, pp += P.pstride) {
            const float4 gv = __ldcg(reinterpret_cast<const float4*>(pp)), uv = __ldcg(reinterpret_cast<const float4*>(pp + 64));
            gt.x += gv.x, gt.y += gv.y, gt.z += gv.z, gt.w += gv.w;
            ut.x += uv.x, ut.y += uv.y, ut.z += uv.z, ut.w += uv.w;
          }
          auto act = [&](float gg, float uu) {
            gg *= sc;
            return gg / (1.f + __expf(-gg)) * (uu * sc);
          };
          __nv_bfloat16* ir = A.inter + (int64_t)routes[r0 + tok] * A.I + col;
          *reinterpret_cast<uint32_t*>(ir) = pkbf(act(gt.x, ut.x), act(gt.y, ut.y));
          *reinterpret_cast<uint32_t*>(ir + 16) = pkbf(act(gt.z, ut.z), act(gt.w, ut.w));
        }
        __syncwarp();
        if (lane == 0) st_release(rdy + P.slot * A.RB1 + P.rb, 1);
      } else {
        const float sc = __ldg(A.gs2 + e) * 9.223372036854775808e18f;
        for (int tok = 0; tok < m; tok++) {
          float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          const float* pp = A.part + P.pfirst + tok * 256 + 8 * lane;
#pragma unroll 4
          for (int j = 0; j < P.np; j++, pp += P.pstride) {
            const float4 v0 = __ldcg(reinterpret_cast<const float4*>(pp)), v1 = __ldcg(reinterpret_cast<const float4*>(pp + 4));
            a[0] += v0.x, a[1] += v0.y, a[2] += v0.z, a[3] += v0.w;
            a[4] += v1.x, a[5] += v1.y, a[6] += v1.z, a[7] += v1.w;
          }
          *reinterpret_cast<uint4*>(A.y + (int64_t)routes[r0 + tok] * A.H + 256 * P.rb + 8 * lane) =
              make_uint4(pkbf(a[0] * sc, a[1] * sc), pkbf(a[2] * sc, a[3] * sc), pkbf(a[4] * sc, a[5] * sc),
                         pkbf(a[6] * sc, a[7] * sc));
        }
      }
    }
    if (A.trace && lane == 0 && fw == 0) A.trace[b * 16 + 5] = gtime();
    // The last CTA to get here resets the ready flags: every CTA's consumers are past their last wait on them.
    int last = 0;
    if (lane == 0) {
      last = atom_add_acq_rel(A.state, 1) == G * DEC_FW - 1;
    }
    if (__shfl_sync(0xffffffffu, last, 0)) {
      for (int i = lane; i < n * A.RB1; i += 32)
        rdy[i] = 0;
      if (lane == 0) A.state[0] = 0;
    }
  }
}

// Route count from which the CTA-pair kernel is used (at fewer routes one CTA per tile is faster on the H200).
inline constexpr int64_t kPairMinRoutes = 16384;
// Route count up to which one block plans the call (the multi-block count and scatter win above it).
inline constexpr int64_t kSmallRoutes = 1024;

// Token rows: BF16 [rows][cols], boxes of 64 columns (128B, swizzled) x box_rows rows.
inline CUtensorMap row_map(void* base, int64_t cols, int64_t rows, uint32_t box_rows) {
  return host::tma::make_map(
      base,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      {(cuuint64_t)cols, (cuuint64_t)rows},
      {(cuuint64_t)cols * 2},
      {64, box_rows},
      {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B, .l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B});
}
// Packed weights as rows of 512B (256 x uint16), boxes of box_rows rows.
inline CUtensorMap weight_map(void* base, int64_t bytes, uint32_t box_rows, CUtensorMapL2promotion l2) {
  return host::tma::make_map(
      base,
      CU_TENSOR_MAP_DATA_TYPE_UINT16,
      {256, (cuuint64_t)(bytes / 512)},
      {512},
      {256, box_rows},
      {.l2_promotion = l2});
}

// y[r] = W2 (silu(W1 x[r / top_k]) * (W3 x[r / top_k])) * scales for every route r = token * top_k + slot whose id
// is in [0, E). With out, out = scale * the weighted sum by topk_weights, and y is workspace. Without it, the rows of
// y for other routes are zeros. meta (int32), order, a and inter are the caller's workspace. variant is 0 for auto, 1
// for one CTA per tile and 2 for CTA pairs. ctas is 0 for one CTA per SM, else that many (tests reach each path).
inline void
run(tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView meta,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView inter,
    tvm::ffi::TensorView y,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    int64_t variant,
    int64_t ctas) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto I = SymbolicSize{"intermediate"};
  auto E = SymbolicSize{"local_experts"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(a);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y);
  TensorMatcher({R, I}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(inter);
  TensorMatcher({R}).with_dtype<int64_t>().with_device(device).verify(order);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(meta);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), r = R.unwrap(), i = I.unwrap(), e = E.unwrap();
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(topk >= 1 && r < (int64_t{1} << 30), "nvfp4_w4a16_sm90: need 1 <= top_k and tokens * top_k < 2^30");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(i >= 128 && i % 128 == 0, "nvfp4_w4a16_sm90: intermediate size must be a multiple of 128, got ", i);
  RuntimeCheck(e >= 1 && e <= 1024, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  const int RB1 = (int)(2 * i / 256), RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, RB1, C1, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(meta.size(0) >= 8 * e + 4 + r / TOKMAX + e + 64, "nvfp4_w4a16_sm90: meta workspace too small");
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  RuntimeCheck(variant >= 0 && variant <= 2, "nvfp4_w4a16_sm90: variant must be 0 (auto), 1 or 2");
  if (r == 0) return;
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  const bool pair_ok = RB1 % 2 == 0 && RB2 % 2 == 0;
  const bool pair = variant == 2 || (variant == 0 && pair_ok && r >= kPairMinRoutes);
  RuntimeCheck(!pair || pair_ok, "nvfp4_w4a16_sm90: CTA pairs need an even number of 256-row blocks in both GEMMs");
  int grid = ctas > 0 ? (int)std::min<int64_t>(ctas, sms) : sms;
  if (pair) grid &= ~1;
  RuntimeCheck(grid >= (pair ? 2 : 1), "nvfp4_w4a16_sm90: too few CTAs");

  cudaStream_t stream = LaunchKernel::resolve_device(dev);
  int* m = static_cast<int*>(meta.data_ptr());
  const int* ids = static_cast<const int*>(topk_ids.data_ptr());
  auto* ord = static_cast<int64_t*>(order.data_ptr());
  const int Ei = (int)e, Ri = (int)r;
  const int pb = (Ei + 31) / 32 * 32;
  const size_t plan_smem = (2 * Ei + 32) * sizeof(int);
  if (r <= kSmallRoutes) {
    LaunchKernel(1, std::max(pb, 256), dev, plan_smem)(front_small_k, ids, Ri, Ei, RB1, RB2, ord, m, (int)meta.size(0));
  } else {
    const int g = (int)std::min<int64_t>(2L * sms, (r + 1023) / 1024);
    CHECK_CUDA(cudaMemsetAsync(m, 0, meta.size(0) * sizeof(int), stream));
    LaunchKernel(g, 1024, dev, Ei * sizeof(int))(count_k, ids, Ri, Ei, m);
    LaunchKernel(1, pb, dev, plan_smem)(plan_k, m, Ei, RB1, RB2);
    LaunchKernel(g, 1024, dev)(scatter_k, ids, Ri, Ei, ord, m);
  }
  if constexpr (NVFP4_MOE_DEBUG_DELAY_NS > 0) {  // test builds: rows read before they are written come out NaN
    CHECK_CUDA(cudaMemsetAsync(a.data_ptr(), 0xFF, r * h * 2, stream));
    CHECK_CUDA(cudaMemsetAsync(inter.data_ptr(), 0xFF, r * i * 2, stream));
  }
  LaunchKernel((int)std::min<int64_t>(2048, r), 256, dev)
      .enable_pdl(true)(
          gather_k,
          static_cast<const uint32_t*>(x.data_ptr()),
          (const int64_t*)ord,
          (const int*)m,
          Ei,
          Ri,
          static_cast<uint4*>(a.data_ptr()),
          out.has_value() ? nullptr : static_cast<uint4*>(y.data_ptr()),
          (int)h,
          (int)topk);

  Maps mp;
  for (int k = 0; k < 2; k++) {
    void* b = k == 0 ? a.data_ptr() : inter.data_ptr();
    const int64_t cols = k == 0 ? h : i;
    mp.x64[k] = row_map(b, cols, r, 64);
    mp.x128[k] = row_map(b, cols, r, 128);
    for (int j = 0; j < 3; j++)
      mp.xn[k][j] = row_map(b, cols, r, 8u << j);
    const tvm::ffi::TensorView& wt = k == 0 ? w13 : w2;
    const int64_t bytes = wt.size(0) * wt.size(1) * wt.size(2) * 16 * 576;
    mp.w[k] = weight_map(wt.data_ptr(), bytes, WROWS, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    mp.w2c[k] = weight_map(wt.data_ptr(), bytes, 2 * WROWS, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  }
  const Shape S{Ei, RB1, RB2, C1, C2, (int)i, (int)h};
  const auto* g1 = static_cast<const float*>(gs13.data_ptr());
  const auto* g2 = static_cast<const float*>(gs2.data_ptr());
  auto* o1 = static_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* o2 = static_cast<__nv_bfloat16*>(y.data_ptr());
  if (pair) {
    runtime::set_max_dynamic_smem_per_device<pair_k>(dev, SMEM);
    LaunchKernel(grid, (NW + 4) * 32, dev, SMEM)
        .enable_cluster(dim3{2, 1, 1})(pair_k, mp, m, g1, g2, o1, o2, (const int64_t*)ord, S);
  } else {  // programmatic dependent launch: the prologue overlaps the gather's tail
    runtime::set_max_dynamic_smem_per_device<grouped_k>(dev, SMEM);
    LaunchKernel(grid, (NW + 4) * 32, dev, SMEM)
        .enable_pdl(true)(grouped_k, mp, m, g1, g2, o1, o2, (const int64_t*)ord, S);
  }
  if (out.has_value()) {
    LaunchKernel((int)std::min<int64_t>(t, 4 * sms), 256, dev)
        .enable_pdl(true)(
            combine_k,
            static_cast<const uint4*>(y.data_ptr()),
            ids,
            static_cast<const float*>(topk_weights.value().data_ptr()),
            (int)t,
            (int)topk,
            Ei,
            (int)(h / 8),
            (float)scale,
            static_cast<uint4*>(out.value().data_ptr()));
  }
}

// Floats of FP32 partial sums dec_k may write for t tokens, r routes, nmax slots on g CTAs (the per-batch bound: every
// GEMM step stores (g + its row blocks) partials of 256 rows x its largest slot's routes).
__host__ __device__ __forceinline__ int64_t dec_part_floats(int64_t t, int64_t r, int64_t nmax, int64_t g, int rb1, int rb2) {
  return 256 * (2 * g * r + (int64_t)(rb1 + rb2) * nmax * t);
}

// The decode path: the whole call in one dec_k launch, for at most DEC_TMAX tokens and DEC_RMAX routes. Same contract as
// run() (with out, the weighted sum; without, y per route with zeros for other ranks' routes). part (FP32, at least
// dec_part_floats) and state (int32, zero before the first call; every call leaves it zero) are the caller's, and two
// calls must not share them at the same time. thr is the GEMM1 chunk count from which a slot batch closes.
inline void run_decode(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView inter,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView part,
    tvm::ffi::TensorView state,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    int64_t thr,
    int64_t dbg_trace,
    int64_t dbg) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto I = SymbolicSize{"intermediate"};
  auto E = SymbolicSize{"local_experts"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y);
  TensorMatcher({R, I}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(inter);
  TensorMatcher({-1}).with_dtype<float>().with_device(device).ensure_alignment(16).verify(part);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(state);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), r = R.unwrap(), i = I.unwrap(), e = E.unwrap();
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(t >= 1 && t <= DEC_TMAX && r <= DEC_RMAX, "nvfp4_w4a16_sm90 decode: at most ", DEC_TMAX, " tokens and ",
               DEC_RMAX, " routes");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(i >= 128 && i % 128 == 0, "nvfp4_w4a16_sm90: intermediate size must be a multiple of 128, got ", i);
  RuntimeCheck(e >= 1 && e <= DEC_EMAX, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  const int RB1 = (int)(2 * i / 256), RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, RB1, C1, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  RuntimeCheck(thr >= 1, "nvfp4_w4a16_sm90 decode: thr must be positive");
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  const int nmax = (int)std::min(e, r);
  RuntimeCheck(state.size(0) >= dec_tcnt(nmax, RB1, RB2) + t * RB2, "nvfp4_w4a16_sm90 decode: state too small");
  RuntimeCheck(part.size(0) >= dec_part_floats(t, r, nmax, sms, RB1, RB2), "nvfp4_w4a16_sm90 decode: part too small");
  DecArgs A;
  A.w13 = static_cast<const uint8_t*>(w13.data_ptr());
  A.w2 = static_cast<const uint8_t*>(w2.data_ptr());
  A.gs13 = static_cast<const float*>(gs13.data_ptr());
  A.gs2 = static_cast<const float*>(gs2.data_ptr());
  A.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  A.ids = static_cast<const int*>(topk_ids.data_ptr());
  A.tw = topk_weights.has_value() ? static_cast<const float*>(topk_weights.value().data_ptr()) : nullptr;
  A.inter = static_cast<__nv_bfloat16*>(inter.data_ptr());
  A.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  A.out = out.has_value() ? static_cast<__nv_bfloat16*>(out.value().data_ptr()) : nullptr;
  A.part = static_cast<float*>(part.data_ptr());
  A.state = static_cast<int*>(state.data_ptr());
  A.trace = reinterpret_cast<unsigned long long*>(dbg_trace);  // DEBUG
  A.dbg = (int)dbg;                                             // DEBUG
  A.scale = (float)scale;
  A.T = (int)t, A.topk = (int)topk, A.E = (int)e, A.H = (int)h, A.I = (int)i;
  A.RB1 = RB1, A.RB2 = RB2, A.C1 = C1, A.C2 = C2, A.thr = (int)thr, A.nmax = nmax;
  auto go = [&](auto kern) {
    runtime::set_max_dynamic_smem_per_device<decltype(kern)::value>(dev, DEC_SMEM);
    LaunchKernel(sms, DEC_THREADS, dev, DEC_SMEM).enable_cooperative(!(dbg & 32))(decltype(kern)::value, A);
  };
  if (t <= 8)
    go(std::integral_constant<decltype(&dec_k<1>), &dec_k<1>>{});
  else
    go(std::integral_constant<decltype(&dec_k<2>), &dec_k<2>>{});
}


// ---- decode path 2 (dec2_k): row ownership over the full K, for one to a few tokens ----
// At decode sizes the call costs the touched experts' weight bytes. dec2_k reads each byte once from every SM at once
// and never splits a K sum across CTAs (the structure of gauravjain14's one-token muse_moe_w4a16 kernel, generalized
// to runtime shapes and to this op's weight layout):
//   * Every CTA computes the routing (slots: experts with local routes in increasing id, each slot's routes in route
//     order). GEMM1 items are (slot, row block, half, pair): 32 gate rows and their 32 up rows over all of K, item i
//     on CTA i % G, slot-major, so the first slots' intermediates are complete early. GEMM2 pieces are 32 output rows
//     (row block, pair); piece j is on CTA j % G, which runs it for every slot in increasing expert id and sums the
//     weighted routes itself, so the weighted sum needs no other CTA and no atomics.
//   * Within a 9216B chunk (256 rows x 64 k) 32 rows are 1152 contiguous bytes, so a stage is one tensor copy of
//     1152B x up to 32 chunks (stride 9216B, 36KB, GEMM2) or two of 1152B x 16 chunks (gate and up rows, GEMM1).
//     Measured on H100 at 3 Muse experts, this streams within 4.5% of contiguous 36KB bulk copies.
//   * Roles: warp D2_CW (producer) issues the copies into the ring and waits only for free stages. Warp D2_CW + 1
//     stages every token's x row once, then each GEMM2 item's intermediate rows into one of two buffers once the
//     slot's GEMM1 items are all counted. The D2_CW consumer warps split each stage along K (chunk c to warp c % 8),
//     decode in registers and run mma.sync m16n8k16 with the slot's routes as the n8 columns, and at an item's end
//     hand their sums to warp D2_CW + 2 (epilogue) through one of two buffers, so they never wait on an epilogue.
//     The epilogue adds the eight K shares in warp order, then writes SwiGLU into the intermediate and counts the
//     slot's item (release), or keeps each token's weighted sum of the piece in registers and writes it after the
//     piece's last slot (or writes y per route).
//   * Counters (slot readiness, CTAs done) return to zero at the end of every call; the launch is cooperative.
constexpr int D2_CW = 8;                       // consumer warps
constexpr int D2_THREADS = (D2_CW + 3) * 32;   // + producer, B loader, epilogue
constexpr int D2_STB = 36864;                  // bytes per stage
constexpr int D2_TMAX = 4;                     // tokens
constexpr int D2_RMAX = 256;                   // routes
constexpr int D2_EMAX = 1024;
constexpr int D2_RED = D2_CW * 64 * D2_TMAX * 4;  // bytes per hand-off buffer: [warp][64 rows][D2_TMAX]
constexpr int D2_NBAR = 32;
constexpr int D2_FIXED = 2 * D2_RED + D2_NBAR * 8 + (5 * D2_RMAX + D2_EMAX + 8) * 4;  // past the ring and B rows
constexpr int D2_SMEM_MAX = 232448;

struct D2Maps {
  CUtensorMap w13, w2;  // {144 x 8B, 8 pairs, C chunks, E x RB row blocks}, boxes of 1152B x nc chunks
};
struct D2Args {
  const __nv_bfloat16* x;
  const int* ids;
  const float* tw;  // null: y per route (zeros for other ranks' routes); else the weighted sum into out
  const float* gs13;
  const float* gs2;
  __nv_bfloat16* inter;
  __nv_bfloat16* y;
  __nv_bfloat16* out;
  int* state;  // [0] CTAs done, [32 (1 + slot)] GEMM1 items done
  float scale;
  int T, topk, E, H, I, RB1, RB2, C1, C2, nc1, nc2;
  int ns, off_x, off_i, ibuf, off_fixed;  // ring stages; byte offsets of x rows, the two intermediate buffers, the rest
  int dbg;                    // DEBUG: 1 no consumer math, 2 no readiness wait, 16 plain launch
  unsigned long long* trace;  // DEBUG: per CTA [entry, setup done, first issue, last stage full, end, waits]
};

__device__ __forceinline__ void tma_4d_h(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, int c3,
                                         uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, "
      "%4, %5}], [%6], %7;" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// One stage of one item for this consumer warp: chunks w, w + 8, ... of the stage (< nv valid), U units of 16 rows
// per chunk at byte offsets uoff[u]; brow is this lane's B row (column g), null for an empty column.
template <int U>
__device__ __forceinline__ void d2_stage(float (&acc)[4][4], const uint8_t* sb, const int (&uoff)[4], int nv,
                                         const uint8_t* brow, int k0, int warp, int lane) {
  const int q = lane & 3;
  for (int c = warp; c < nv; c += D2_CW) {
    uint4 b0 = make_uint4(0, 0, 0, 0), b1 = b0;
    if (brow != nullptr) {
      const uint8_t* bp = brow + (k0 + c) * 128 + q * 32;
      b0 = *reinterpret_cast<const uint4*>(bp);
      b1 = *reinterpret_cast<const uint4*>(bp + 16);
    }
    const uint8_t* cb = sb + c * 1152;
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint8_t* up = cb + uoff[u];
      const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
      const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        uint32_t o[4];
        decode4(u4w(nb, kk), o);
        const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                       a3 = bmulh(o[3], Sc, 1);
        const uint4& bv = kk < 2 ? b0 : b1;
        mma16816(acc[u], a0, a1, a2, a3, u4w(bv, 2 * (kk & 1)), u4w(bv, 2 * (kk & 1) + 1));
      }
    }
  }
}

__global__ void __launch_bounds__(D2_THREADS, 1) dec2_k(const __grid_constant__ D2Maps mp, const D2Args A) {
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  uint8_t* fx = dsm + A.off_fixed;
  float* redb = reinterpret_cast<float*>(fx);  // two hand-off buffers
  uint64_t* bars = reinterpret_cast<uint64_t*>(fx + 2 * D2_RED);
  int* sids = reinterpret_cast<int*>(bars + D2_NBAR);
  int* slot_e = sids + D2_RMAX;
  int* slot_m = slot_e + D2_RMAX;
  int* slot_r0 = slot_m + D2_RMAX;
  int* routes = slot_r0 + D2_RMAX;  // route indices by slot, increasing within a slot
  int* scnt = routes + D2_RMAX;     // per expert: route count, then slot
  int* misc = scnt + D2_EMAX;       // [0] slots
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(dsm);
  const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(bars);
  const int NS = A.ns;
  // barriers: stage full [0, 8), stage free [8, 16), x rows [16], intermediate rows full [17, 19) / free [19, 21),
  // hand-off full [21, 23) / free [23, 25)
  const uint32_t bfull = bar0, bempty = bar0 + 64, bx = bar0 + 128, bif = bar0 + 136, bie = bar0 + 152,
                 brf = bar0 + 168, bre = bar0 + 184;
  const int R = A.T * A.topk, E = A.E;
  auto gtime = []() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  };
  if (A.trace && tid == 0) A.trace[b * 16] = gtime();

  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, D2_CW);
    }
    mbar_init(bx, 1);
    for (int s = 0; s < 2; s++) {
      mbar_init(bif + 8 * s, 1);
      mbar_init(bie + 8 * s, D2_CW);
      mbar_init(brf + 8 * s, D2_CW);
      mbar_init(bre + 8 * s, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < R; i += D2_THREADS)
    sids[i] = A.ids[i];
  for (int e = tid; e < E; e += D2_THREADS)
    scnt[e] = 0;
  __syncthreads();
  if (tid == (D2_CW + 1) * 32) {  // every token's x row (padded by 16B) by bulk copies; it needs no routing
    mbar_expect(bx, A.T * 2 * A.H);
    for (int t = 0; t < A.T; t++)
      bulk_g2s(ring + A.off_x + t * (2 * A.H + 16), A.x + (int64_t)t * A.H, 2 * A.H, bx, pol_evict_first());
  }
  for (int i = tid; i < R; i += D2_THREADS)
    if ((unsigned)sids[i] < (unsigned)E) atomicAdd(&scnt[sids[i]], 1);
  __syncthreads();
  if (warp == 0) {  // slots in increasing expert id; scnt[e] becomes e's slot
    int n = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane, c = e < E ? scnt[e] : 0;
      const uint32_t bal = __ballot_sync(0xffffffffu, c > 0);
      if (c > 0) {
        const int pos = n + __popc(bal & ((1u << lane) - 1));
        slot_e[pos] = e;
        slot_m[pos] = c;
        scnt[e] = pos;
      }
      n += __popc(bal);
    }
    if (lane == 0) {
      int r0 = 0;
      for (int s = 0; s < n; s++) {
        slot_r0[s] = r0;
        r0 += slot_m[s];
      }
      misc[0] = n;
    }
  }
  __syncthreads();
  for (int i = tid; i < R; i += D2_THREADS) {
    const int e = sids[i];
    if ((unsigned)e < (unsigned)E) {
      int rank = 0;
      for (int j = 0; j < i; j++)
        rank += sids[j] == e;
      routes[slot_r0[scnt[e]] + rank] = i;
    }
  }
  __syncthreads();
  const int n = misc[0];
  const int N1 = n * A.RB1 * 4, P2 = A.RB2 * 8;
  const int n1c = N1 > b ? (N1 - 1 - b) / G + 1 : 0, n2c = P2 > b ? (P2 - 1 - b) / G + 1 : 0;
  const int k1 = (A.C1 + A.nc1 - 1) / A.nc1, k2 = (A.C2 + A.nc2 - 1) / A.nc2;
  const int nitems = n1c + n2c * n;
  // item it of this CTA: GEMM1 (it < n1c) or GEMM2 piece (it - n1c) / n, slot (it - n1c) % n
  auto item = [&](int it, int& kind, int& s, int& rb, int& hp) {
    if (it < n1c) {
      const int i = b + it * G, per = A.RB1 * 4;
      kind = 0, s = i / per, rb = (i % per) >> 2, hp = i & 3;
    } else {
      const int j = b + ((it - n1c) / n) * G;
      kind = 1, s = (it - n1c) % n, rb = j >> 3, hp = j & 7;
    }
  };
  int* const ready = A.state + 32;  // one 128B line per slot: 132 CTAs poll them
  if (A.trace && tid == 0) A.trace[b * 16 + 1] = gtime();

  if (warp == D2_CW) {  // producer
    if (lane == 0) {
      const uint64_t pol = pol_evict_first();
      int k = 0;
      for (int it = 0; it < nitems; it++) {
        int kind, s, rb, hp;
        item(it, kind, s, rb, hp);
        const int e = slot_e[s];
        const int ks = kind ? k2 : k1;
        for (int j = 0; j < ks; j++, k++) {
          const int st = k % NS;
          if (k >= NS) mbar_wait(bempty + 8 * st, ((k / NS) + 1) & 1);
          const uint32_t dst = ring + st * D2_STB, bar = bfull + 8 * st;
          if (A.trace && k == 0) A.trace[b * 16 + 2] = gtime();
          if (kind == 0) {
            const int h = hp >> 1, p = hp & 1, row = e * A.RB1 + rb;
            mbar_expect(bar, 2 * 1152 * A.nc1);
            tma_4d_h(dst, &mp.w13, 0, 4 * h + p, j * A.nc1, row, bar, pol);
            tma_4d_h(dst + 1152 * A.nc1, &mp.w13, 0, 4 * h + 2 + p, j * A.nc1, row, bar, pol);
          } else {
            mbar_expect(bar, 1152 * A.nc2);
            tma_4d_h(dst, &mp.w2, 0, hp, j * A.nc2, e * A.RB2 + rb, bar, pol);
          }
        }
      }
    }
  } else if (warp == D2_CW + 1) {  // the intermediate rows of each GEMM2 item, double-buffered
    const int rowb = 2 * A.I + 16, per = A.I / 8;
    for (int j2 = 0; j2 < n2c * n; j2++) {
      const int s = j2 % n, buf = j2 & 1, m = slot_m[s], r0 = slot_r0[s];
      if (j2 >= 2) mbar_wait(bie + 8 * buf, ((j2 >> 1) + 1) & 1);
      if (A.trace && lane == 0 && j2 < 3) A.trace[b * 16 + 8 + 2 * j2] = gtime();
      if (!(A.dbg & 2)) {
        if (lane == 0)
          while (ld_acquire(ready + 32 * s) < A.RB1 * 4)
            __nanosleep(256);
        __syncwarp();
      }
      if (A.trace && lane == 0 && j2 < 3) A.trace[b * 16 + 9 + 2 * j2] = gtime();
      uint8_t* bb = dsm + A.off_i + buf * A.ibuf;
      for (int i0 = 0; i0 < m * per; i0 += 32 * 8) {
        uint4 v[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
          const int i = i0 + u * 32 + lane;
          if (i < m * per) v[u] = ld_cg4(A.inter + (int64_t)routes[r0 + i / per] * A.I + (i % per) * 8);
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
          const int i = i0 + u * 32 + lane;
          if (i < m * per) *reinterpret_cast<uint4*>(bb + (i / per) * rowb + (i % per) * 16) = v[u];
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bif + 8 * buf);
    }
  } else if (warp == D2_CW + 2) {  // epilogue
    float wsum[D2_TMAX];
#pragma unroll
    for (int t = 0; t < D2_TMAX; t++)
      wsum[t] = 0.f;
    for (int it = 0; it < nitems; it++) {
      int kind, s, rb, hp;
      item(it, kind, s, rb, hp);
      const int hb = it & 1, e = slot_e[s], m = slot_m[s], r0 = slot_r0[s];
      mbar_wait(brf + 8 * hb, (it >> 1) & 1);
      const float* red = redb + hb * (D2_RED / 4);
      const int jr = lane;
      if (kind == 0) {
        const float sc = __ldg(A.gs13 + e) * 9.223372036854775808e18f;
        const int h = hp >> 1, p = hp & 1;
        const int col = 128 * rb + 64 * h + dec_kp(32 * p + jr);
        for (int r = 0; r < m; r++) {
          float gt = 0.f, ut = 0.f;
#pragma unroll
          for (int w = 0; w < D2_CW; w++)
            gt += red[(w * 64 + jr) * D2_TMAX + r], ut += red[(w * 64 + 32 + jr) * D2_TMAX + r];
          gt *= sc;
          A.inter[(int64_t)routes[r0 + r] * A.I + col] = __float2bfloat16(gt / (1.f + __expf(-gt)) * (ut * sc));
        }
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(bre + 8 * hb);
          asm volatile("fence.acq_rel.gpu;" ::: "memory");
          red_release_add(ready + 32 * s, 1);
          if (A.trace) A.trace[b * 16 + 14] = gtime();
        }
      } else {
        const float sc = __ldg(A.gs2 + e) * 9.223372036854775808e18f;
        const int row0 = 256 * rb + 32 * hp;
        for (int r = 0; r < m; r++) {
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < D2_CW; w++)
            v += red[(w * 64 + jr) * D2_TMAX + r];
          const int route = routes[r0 + r];
          const __nv_bfloat16 yv = __float2bfloat16(v * sc);
          if (A.tw == nullptr) {
            A.y[(int64_t)route * A.H + row0 + jr] = yv;
          } else {
            const float c = __ldg(A.tw + route) * __bfloat162float(yv);
            const int t = route / A.topk;
#pragma unroll
            for (int tt = 0; tt < D2_TMAX; tt++)
              if (tt == t) wsum[tt] += c;
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bre + 8 * hb);
        if (s == n - 1) {  // the piece's last slot: its rows of out (or the zeros of other ranks' routes in y)
          if (A.tw != nullptr) {
#pragma unroll
            for (int t = 0; t < D2_TMAX; t++) {
              if (t < A.T) A.out[(int64_t)t * A.H + row0 + jr] = __float2bfloat16(wsum[t] * A.scale);
              wsum[t] = 0.f;
            }
          } else {
            for (int r = 0; r < R; r++)
              if ((unsigned)sids[r] >= (unsigned)E) A.y[(int64_t)r * A.H + row0 + jr] = __float2bfloat16(0.f);
          }
        }
      }
    }
    if (n == 0) {  // no local routes: this CTA's pieces are zeros
      for (int it = 0; it < n2c; it++) {
        const int j = b + it * G, row0 = 256 * (j >> 3) + 32 * (j & 7);
        const int rows = A.tw != nullptr ? A.T : R;
        __nv_bfloat16* dst = A.tw != nullptr ? A.out : A.y;
        for (int r = 0; r < rows; r++)
          dst[(int64_t)r * A.H + row0 + lane] = __float2bfloat16(0.f);
      }
    }
  } else {  // consumers
    const int g = lane >> 2, q = lane & 3;
    int k = 0;
    uint64_t wb_ = 0, wf_ = 0;  // DEBUG: waits on B rows, on stages
    const int rowx = 2 * A.H + 16, rowi = 2 * A.I + 16;
    if (n1c > 0) mbar_wait(bx, 0);
    for (int it = 0; it < nitems; it++) {
      int kind, s, rb, hp;
      item(it, kind, s, rb, hp);
      const int m = slot_m[s], r0 = slot_r0[s], j2 = it - n1c;
      const int C = kind ? A.C2 : A.C1, nc = kind ? A.nc2 : A.nc1;
      const uint8_t* brow = nullptr;
      if (kind == 0) {
        if (g < m) brow = dsm + A.off_x + (routes[r0 + g] / A.topk) * rowx;
      } else {
        const uint64_t t0 = A.trace ? gtime() : 0;
        mbar_wait(bif + 8 * (j2 & 1), (j2 >> 1) & 1);
        if (A.trace) wb_ += gtime() - t0;
        if (g < m) brow = dsm + A.off_i + (j2 & 1) * A.ibuf + g * rowi;
      }
      float acc[4][4];
#pragma unroll
      for (int u = 0; u < 4; u++)
#pragma unroll
        for (int i = 0; i < 4; i++)
          acc[u][i] = 0.f;
      const int ks = kind ? k2 : k1;
      for (int j = 0; j < ks; j++, k++) {
        const int st = k % NS;
        {
          const uint64_t t0 = A.trace ? gtime() : 0;
          mbar_wait(bfull + 8 * st, (k / NS) & 1);
          if (A.trace) wf_ += gtime() - t0;
        }
        if (A.trace && tid == 0) A.trace[b * 16 + 3] = gtime();
        const uint8_t* sb = dsm + st * D2_STB;
        const int nv = min(nc, C - j * nc);
        if (A.dbg & 1) {
        } else if (kind == 0) {
          const int uo[4] = {0, 576, 1152 * nc, 1152 * nc + 576};
          d2_stage<4>(acc, sb, uo, nv, brow, j * nc, warp, lane);
        } else {
          const int uo[4] = {0, 576, 0, 0};
          d2_stage<2>(acc, sb, uo, nv, brow, j * nc, warp, lane);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bempty + 8 * st);
      }
      if (kind == 1) {
        __syncwarp();
        if (lane == 0) mbar_arrive(bie + 8 * (j2 & 1));
      }
      // this warp's K share to the epilogue
      const int hb = it & 1;
      if (it >= 2) mbar_wait(bre + 8 * hb, ((it >> 1) + 1) & 1);
      float* red = redb + hb * (D2_RED / 4);
      const int U = kind ? 2 : 4;
#pragma unroll
      for (int u = 0; u < 4; u++) {
        if (u < U) {
          float* rp = red + (warp * 64 + 16 * u + g) * D2_TMAX;
          if (2 * q < m) rp[2 * q] = acc[u][0], rp[8 * D2_TMAX + 2 * q] = acc[u][2];
          if (2 * q + 1 < m) rp[2 * q + 1] = acc[u][1], rp[8 * D2_TMAX + 2 * q + 1] = acc[u][3];
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(brf + 8 * hb);
    }
    if (A.trace && tid == 0) A.trace[b * 16 + 5] = wb_, A.trace[b * 16 + 6] = wf_;
  }
  __syncthreads();
  if (A.trace && tid == 0) A.trace[b * 16 + 4] = gtime();
  // The last CTA to finish resets the readiness counters: every CTA is past its last wait on them.
  if (tid == 0) {
    __threadfence();
    if (atom_add_acq_rel(A.state, 1) == G - 1) {
      for (int s = 0; s < n; s++)
        ready[32 * s] = 0;
      A.state[0] = 0;
    }
  }
}

// The decode path 2: one dec2_k launch, at most D2_TMAX tokens and as many as fit in shared memory (the ring keeps at
// least two stages). Same contract as run_decode. state (int32, at least 32 (1 + min(E, T top_k)) words, zero before the
// first call) is left zero.
inline int64_t dec2_ring_stages(int64_t t, int64_t h, int64_t i) {
  const int64_t rows = ((t * (2 * h + 16) + 127) / 128 + 2 * ((t * (2 * i + 16) + 127) / 128)) * 128;
  return std::min<int64_t>(5, (D2_SMEM_MAX - D2_FIXED - rows) / D2_STB);
}
inline void run_decode2(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView inter,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView state,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    int64_t ctas,
    int64_t dbg) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto I = SymbolicSize{"intermediate"};
  auto E = SymbolicSize{"local_experts"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y);
  TensorMatcher({R, I}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(inter);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(state);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), r = R.unwrap(), i = I.unwrap(), e = E.unwrap();
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(t >= 1 && t <= D2_TMAX && r <= D2_RMAX, "nvfp4_w4a16_sm90 decode2: at most ", D2_TMAX, " tokens and ",
               D2_RMAX, " routes");
  const int64_t ns = dec2_ring_stages(t, h, i);
  RuntimeCheck(ns >= 2, "nvfp4_w4a16_sm90 decode2: tokens x width too large for shared memory");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(i >= 128 && i % 128 == 0, "nvfp4_w4a16_sm90: intermediate size must be a multiple of 128, got ", i);
  RuntimeCheck(e >= 1 && e <= D2_EMAX, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  const int RB1 = (int)(2 * i / 256), RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, RB1, C1, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  RuntimeCheck(state.size(0) >= 32 * (1 + std::min(e, r)), "nvfp4_w4a16_sm90 decode2: state too small");
  const int nc1 = std::min(16, C1), nc2 = std::min(32, C2);
  auto wmap = [&](void* base, int C, int64_t blocks, int nc) {
    return host::tma::make_map(
        base,
        CU_TENSOR_MAP_DATA_TYPE_UINT64,
        {144, 8, (cuuint64_t)C, (cuuint64_t)blocks},
        {1152, 9216, (cuuint64_t)C * 9216},
        {144, 1, (cuuint32_t)nc, 1},
        {.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B});
  };
  D2Maps mp;
  mp.w13 = wmap(w13.data_ptr(), C1, e * RB1, nc1);
  mp.w2 = wmap(w2.data_ptr(), C2, e * RB2, nc2);
  D2Args A;
  A.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  A.ids = static_cast<const int*>(topk_ids.data_ptr());
  A.tw = topk_weights.has_value() ? static_cast<const float*>(topk_weights.value().data_ptr()) : nullptr;
  A.gs13 = static_cast<const float*>(gs13.data_ptr());
  A.gs2 = static_cast<const float*>(gs2.data_ptr());
  A.inter = static_cast<__nv_bfloat16*>(inter.data_ptr());
  A.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  A.out = out.has_value() ? static_cast<__nv_bfloat16*>(out.value().data_ptr()) : nullptr;
  A.state = static_cast<int*>(state.data_ptr());
  A.scale = (float)scale;
  A.T = (int)t, A.topk = (int)topk, A.E = (int)e, A.H = (int)h, A.I = (int)i;
  A.RB1 = RB1, A.RB2 = RB2, A.C1 = C1, A.C2 = C2, A.nc1 = nc1, A.nc2 = nc2;
  A.ns = (int)ns;
  A.off_x = (int)ns * D2_STB;
  A.ibuf = (int)(((t * (2 * i + 16) + 127) / 128) * 128);
  A.off_i = A.off_x + (int)(((t * (2 * h + 16) + 127) / 128) * 128);
  A.off_fixed = A.off_i + 2 * A.ibuf;
  A.dbg = (int)(dbg & 0xffff);
  A.trace = reinterpret_cast<unsigned long long*>(dbg >> 16);
  const int smem = A.off_fixed + D2_FIXED;
  const int grid = ctas > 0 ? (int)std::min<int64_t>(ctas, sms) : sms;
  runtime::set_max_dynamic_smem_per_device<dec2_k>(dev, D2_SMEM_MAX);
  LaunchKernel(grid, D2_THREADS, dev, smem).enable_cooperative(!(dbg & 16))(dec2_k, mp, A);
}


// ---- decode path 3 (dec3_k): dec2_k's row ownership with an early, tagged intermediate exchange ----
// On H100 a global load issued while the weight stream runs returns only after the SM's queued copies (several us),
// so dec3_k keeps every such round trip off the path from a slot's last GEMM1 write to its first GEMM2 stage:
//   * Per slot scales and per route weights are staged in shared memory with the routing, before the stream starts.
//   * GEMM1 writes the intermediate as 8-byte words (two BF16 values, this call's epoch above them), so a reader needs
//     no counter, no fence and no second round trip: a word with this call's epoch holds this call's values.
//   * Every local route's intermediate row has its own shared-memory row when they fit (each slot is fetched once for
//     all of this CTA's GEMM2 pieces); otherwise slots rotate through as many buffers as fit, as in dec2_k.
//   * Four fetcher warps in two pairs take the slots alternately, so two slots' fetches are in flight at once. A
//     pair starts a slot's first round once this CTA's own GEMM1 items of the slot and every earlier slot are written
//     plus fdly ns (every CTA runs the same schedule, so the other writers are about as far along), then reloads only
//     the words whose epoch is still old.
//   * When a CTA would otherwise get few GEMM1 items, they are 16 gate and 16 up rows instead of 32 and 32 (stages of
//     576B x 32 chunks), so all CTAs stream GEMM1 and its slots finish earlier.
constexpr int D3_CW = 8;                          // consumer warps
constexpr int D3_NP = 3;                          // fetcher pairs
constexpr int D3_NF = 2 * D3_NP;                  // fetcher warps
constexpr int D3_TRW = 48;                        // DEBUG: trace words per CTA
constexpr int D3_THREADS = (D3_CW + 2 + D3_NF) * 32;  // + producer, epilogue
constexpr int D3_STB = 36864;
constexpr int D3_TMAX = 4;
constexpr int D3_RMAX = 256;
constexpr int D3_EMAX = 1024;
constexpr int D3_RED = D3_CW * 64 * D3_TMAX * 4;
constexpr int D3_NSMAX = 6;
constexpr int D3_NBMAX = 16;
constexpr int D3_NBAR = 2 * D3_NSMAX + 1 + 2 * D3_NBMAX + 4;
constexpr int D3_RL = 8;  // 16B loads per fetcher lane per round
constexpr int D3_CH = 64 * D3_RL;  // 16B loads per fetch task (a pair of warps)
constexpr int D3_FIXED = ((2 * D3_RED + D3_NBAR * 8 + (9 * D3_RMAX + 3 * D3_EMAX + 16) * 4) + 1023) / 1024 * 1024;
constexpr int D3_SMEM_MAX = 232448;

struct D3Maps {
  CUtensorMap w13, w13h, w2;  // w13h: boxes of one 576B unit x nc chunks (16 rows)
};
struct D3Args {
  const __nv_bfloat16* x;
  const int* ids;
  const float* tw;
  const float* gs13;
  const float* gs2;
  unsigned long long* inter;  // [routes][I / 2] tagged words, persistent (zero before the first call)
  __nv_bfloat16* y;
  __nv_bfloat16* out;
  int* state;  // [0] CTAs done, [1] the last call's epoch
  float scale;
  int T, topk, E, H, I, RB1, RB2, C1, C2, nc1, nc1h, nc2;
  int off_x, smem, nsmax, g1h, fdly, whint;
  int dbg;                    // DEBUG: 1 no consumer math, 2 take every word fresh (one round), 16 plain launch
  unsigned long long* trace;  // DEBUG
};

__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// two 8-byte words, each single-copy atomic: its epoch and its two values are seen together
__device__ __forceinline__ uint4 ld_relaxed_2u64(const unsigned long long* p) {
  unsigned long long a, c;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(c) : "l"(p) : "memory");
  return make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)c, (uint32_t)(c >> 32));
}

__device__ __forceinline__ void mbar_wait_h(uint32_t bar, uint32_t parity, uint32_t hint) {
  if (hint == 0) {
    mbar_wait(bar, parity);
    return;
  }
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra W;\n}" ::"r"(bar),
      "r"(parity), "r"(hint)
      : "memory");
}

// One stage for this consumer warp: chunks w, w + 8, ... (< nv), chunk stride cs, U units of 16 rows at uoff[u].
template <int U>
__device__ __forceinline__ void d3_stage(float (&acc)[4][4], const uint8_t* sb, const int (&uoff)[4], int cs, int nv,
                                         const uint8_t* brow, int k0, int warp, int lane) {
  const int q = lane & 3;
  for (int c = warp; c < nv; c += D3_CW) {
    uint4 b0 = make_uint4(0, 0, 0, 0), b1 = b0;
    if (brow != nullptr) {
      const uint8_t* bp = brow + (k0 + c) * 128 + q * 32;
      b0 = *reinterpret_cast<const uint4*>(bp);
      b1 = *reinterpret_cast<const uint4*>(bp + 16);
    }
    const uint8_t* cb = sb + c * cs;
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint8_t* up = cb + uoff[u];
      const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
      const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        uint32_t o[4];
        decode4(u4w(nb, kk), o);
        const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                       a3 = bmulh(o[3], Sc, 1);
        const uint4& bv = kk < 2 ? b0 : b1;
        mma16816(acc[u], a0, a1, a2, a3, u4w(bv, 2 * (kk & 1)), u4w(bv, 2 * (kk & 1) + 1));
      }
    }
  }
}

// TR: the debug trace build (every stamp compiled out otherwise, so the kernel's code stays small: after a read
// flush its first pass through each role's code fetches instructions from HBM)
template <bool TR>
__global__ void __launch_bounds__(D3_THREADS, 1) dec3_k(const __grid_constant__ D3Maps mp, const D3Args A) {
  unsigned long long* const TRP = TR ? A.trace : nullptr;
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  float* redb = reinterpret_cast<float*>(dsm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(dsm + 2 * D3_RED);
  int* sids = reinterpret_cast<int*>(bars + D3_NBAR);
  int* slot_e = sids + D3_RMAX;
  int* slot_m = slot_e + D3_RMAX;
  int* slot_r0 = slot_m + D3_RMAX;
  int* routes = slot_r0 + D3_RMAX;
  float* sgs13 = reinterpret_cast<float*>(routes + D3_RMAX);  // per slot
  float* sgs2 = sgs13 + D3_RMAX;
  float* stw = sgs2 + D3_RMAX;  // per route (in route order)
  float* egs = stw + D3_RMAX;  // per expert: gs13, then gs2 at + D3_EMAX
  int* scnt = reinterpret_cast<int*>(egs + 2 * D3_EMAX);
  int* misc = scnt + D3_EMAX;  // [0] slots, [1] own GEMM1 items written, [2] ring stages, [3] row buffers (0: per slot)
                               // [4] ring offset, [5] rows offset, [6] rows per buffer, [7] half GEMM1 items
  const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(bars);
  const uint32_t bfull = bar0, bempty = bar0 + 8 * D3_NSMAX, bx = bar0 + 16 * D3_NSMAX, bif = bx + 8,
                 bie = bif + 8 * D3_NBMAX, brf = bie + 8 * D3_NBMAX, bre = brf + 16;
  const int R = A.T * A.topk, E = A.E;
  auto gtime = []() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  };
  if (TRP && tid == 0) TRP[b * D3_TRW] = gtime();
  if (A.dbg & 64) return;  // DEBUG: the launch alone
  uint32_t ep = (uint32_t)A.state[1] + 1u;
  if (ep == 0) ep = 1;

  if (tid == 0) {
    for (int s = 0; s < D3_NSMAX; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, D3_CW);
    }
    mbar_init(bx, 1);
    for (int s = 0; s < D3_NBMAX; s++) {
      mbar_init(bif + 8 * s, 2);
      mbar_init(bie + 8 * s, D3_CW);
    }
    for (int s = 0; s < 2; s++) {
      mbar_init(brf + 8 * s, D3_CW);
      mbar_init(bre + 8 * s, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < R; i += D3_THREADS) {
    sids[i] = A.ids[i];
    stw[i] = A.tw != nullptr ? A.tw[i] : 0.f;
  }
  for (int e = tid; e < E; e += D3_THREADS) {
    scnt[e] = 0;
    egs[e] = A.gs13[e] * 9.223372036854775808e18f;
    egs[D3_EMAX + e] = A.gs2[e] * 9.223372036854775808e18f;
  }
  __syncthreads();
  if (tid == D3_CW * 32) {  // every token's x row (padded by 16B)
    mbar_expect(bx, A.T * 2 * A.H);
    for (int t = 0; t < A.T; t++)
      bulk_g2s((uint32_t)__cvta_generic_to_shared(dsm + A.off_x + t * (2 * A.H + 16)), A.x + (int64_t)t * A.H,
               2 * A.H, bx, pol_evict_first());
  }
  for (int i = tid; i < R; i += D3_THREADS)
    if ((unsigned)sids[i] < (unsigned)E) atomicAdd(&scnt[sids[i]], 1);
  __syncthreads();
  if (warp == 0) {
    int n = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane, c = e < E ? scnt[e] : 0;
      const uint32_t bal = __ballot_sync(0xffffffffu, c > 0);
      if (c > 0) {
        const int pos = n + __popc(bal & ((1u << lane) - 1));
        slot_e[pos] = e;
        slot_m[pos] = c;
        scnt[e] = pos;
      }
      n += __popc(bal);
    }
    __syncwarp();
    for (int s = lane; s < n; s += 32) {
      sgs13[s] = egs[slot_e[s]];
      sgs2[s] = egs[D3_EMAX + slot_e[s]];
    }
    if (lane == 0) {
      int r0 = 0, mmax = 1;
      for (int s = 0; s < n; s++) {
        slot_r0[s] = r0;
        r0 += slot_m[s];
        mmax = max(mmax, slot_m[s]);
      }
      misc[0] = n;
      misc[1] = 0;
      // shared memory past the x rows: the ring, then the intermediate rows
      const int rowi = 2 * A.I + 16;
      const int oring = (A.off_x + A.T * (2 * A.H + 16) + 1023) / 1024 * 1024;
      const int avail = A.smem - oring;
      int ns = min(A.nsmax, (avail - r0 * rowi) / D3_STB), nb = 0;
      if (ns < 2 || (r0 * (A.I / 4) + D3_CH - 1) / D3_CH > D3_NBMAX) {  // rotate: as many buffers of mmax rows as leave two stages
        nb = min(D3_NBMAX, (avail - 2 * D3_STB) / (mmax * rowi));
        ns = min(A.nsmax, (avail - nb * mmax * rowi) / D3_STB);
      }
      misc[2] = ns;
      misc[3] = nb;
      misc[4] = oring;
      misc[5] = oring + ns * D3_STB;
      misc[6] = mmax;
      // GEMM1 item size: halves when that lowers this grid's most GEMM1 stages per CTA
      const int k1 = (A.C1 + A.nc1 - 1) / A.nc1, k1h = (A.C1 + A.nc1h - 1) / A.nc1h;
      const int N1 = n * A.RB1 * 4;
      const int full = (N1 + G - 1) / G * k1, half = (2 * N1 + G - 1) / G * k1h;
      misc[7] = A.g1h == 1 ? 0 : A.g1h == 2 ? 1 : (half < full ? 1 : 0);
    }
  }
  __syncthreads();
  for (int i = tid; i < R; i += D3_THREADS) {
    const int e = sids[i];
    if ((unsigned)e < (unsigned)E) {
      int rank = 0;
      for (int j = 0; j < i; j++)
        rank += sids[j] == e;
      routes[slot_r0[scnt[e]] + rank] = i;
    }
  }
  __syncthreads();
  const int n = misc[0], NS = misc[2], NB = misc[3], mmax = misc[6], g1h = misc[7];
  uint8_t* const ring = dsm + misc[4];
  uint8_t* const rows = dsm + misc[5];
  const int per1 = A.RB1 * (g1h ? 8 : 4);
  const int N1 = n * per1, P2 = A.RB2 * 8;
  const int n1c = N1 > b ? (N1 - 1 - b) / G + 1 : 0, n2c = P2 > b ? (P2 - 1 - b) / G + 1 : 0;
  const int nc1 = g1h ? A.nc1h : A.nc1;
  const int k1 = (A.C1 + nc1 - 1) / nc1, k2 = (A.C2 + A.nc2 - 1) / A.nc2;
  const int nitems = n1c + n2c * n;
  const int rowi = 2 * A.I + 16, rowx = 2 * A.H + 16;
  // GEMM2 items whose weights are one small block (k2 == 1) share ring stages: G2 consecutive slots of a piece
  const int G2 = k2 == 1 && !(A.dbg & 8) ? max(1, min(8, D3_STB / (1152 * A.nc2))) : 1;  // (dbg 8: one block per stage)
  // item it: GEMM1 (it < n1c; slot s, row block rb, part hp) or GEMM2 (piece (it - n1c) / n, slot (it - n1c) % n)
  auto item = [&](int it, int& kind, int& s, int& rb, int& hp) {
    if (it < n1c) {
      const int i = b + it * G;
      kind = 0, s = i / per1, rb = (i % per1) / (per1 / A.RB1), hp = i % (per1 / A.RB1);
    } else {
      const int j = b + ((it - n1c) / n) * G;
      kind = 1, s = (it - n1c) % n, rb = j >> 3, hp = j & 7;
    }
  };
  // GEMM2 item j2's row buffer: its slot's own rows (NB == 0) or buffer j2 % NB
  auto rowbuf = [&](int j2, int s) { return NB == 0 ? rows + slot_r0[s] * rowi : rows + (j2 % NB) * mmax * rowi; };
  if (TRP && tid == 0) TRP[b * D3_TRW + 1] = gtime();

  if (warp == D3_CW) {  // producer
    if (lane == 0) {
      const uint64_t pol = pol_evict_first();
      const uint32_t rs = (uint32_t)__cvta_generic_to_shared(ring);
      int k = 0;
      for (int it = 0; it < nitems; it++) {
        int kind, s, rb, hp;
        item(it, kind, s, rb, hp);
        const int e = slot_e[s];
        const int ks = kind ? k2 : k1;
        if (kind == 1 && s % G2 != 0) continue;  // issued with its group's first slot
        for (int j = 0; j < ks; j++, k++) {
          const int st = k % NS;
          if (k >= NS) mbar_wait_h(bempty + 8 * st, ((k / NS) + 1) & 1, A.whint);
          const uint32_t dst = rs + st * D3_STB, bar = bfull + 8 * st;
          if (TRP && k == 0) TRP[b * D3_TRW + 2] = gtime();
          if (kind == 0 && g1h) {  // hp: unit hp & 1 of gate group 4 (hp >> 2) + ((hp >> 1) & 1) and of its up group
            const int h = hp >> 2, p = (hp >> 1) & 1, u = hp & 1, row = e * A.RB1 + rb;
            mbar_expect(bar, 2 * 576 * nc1);
            tma_4d_h(dst, &mp.w13h, 72 * u, 4 * h + p, j * nc1, row, bar, pol);
            tma_4d_h(dst + 576 * nc1, &mp.w13h, 72 * u, 4 * h + 2 + p, j * nc1, row, bar, pol);
          } else if (kind == 0) {
            const int h = hp >> 1, p = hp & 1, row = e * A.RB1 + rb;
            mbar_expect(bar, 2 * 1152 * nc1);
            tma_4d_h(dst, &mp.w13, 0, 4 * h + p, j * nc1, row, bar, pol);
            tma_4d_h(dst + 1152 * nc1, &mp.w13, 0, 4 * h + 2 + p, j * nc1, row, bar, pol);
          } else {
            const int gsz = min(G2, n - s);
            mbar_expect(bar, gsz * 1152 * A.nc2);
            for (int u = 0; u < gsz; u++)
              tma_4d_h(dst + u * 1152 * A.nc2, &mp.w2, 0, hp, j * A.nc2, slot_e[s + u] * A.RB2 + rb, bar, pol);
          }
        }
      }
    }
  } else if (warp >= D3_CW + 2) {  // fetchers: pair fp takes tasks fp, fp + D3_NP, ...
    // Every parameter this warp needs, read once before the stream and pinned in registers: a constant-cache miss
    // during the stream (ptxas rematerializes parameters instead of holding them) can wait several us.
    unsigned long long* F_inter = A.inter;
    unsigned long long* F_trace = TRP;
    int F_dbg = A.dbg, F_whint = A.whint, F_fdly = A.fdly, F_I = A.I;
    asm volatile("" : "+l"(F_inter), "+l"(F_trace), "+r"(F_dbg), "+r"(F_whint), "+r"(F_fdly), "+r"(F_I));
    // A task is one chunk of D3_CH 16B loads: in per-slot mode a chunk of every local route's row in slot order (a
    // chunk can span slots, so GLM's eight 512B rows are one task), when rotating one GEMM2 item's rows (chunks in
    // turn). A task starts once this CTA has written its GEMM1 items up to the round in which its last slot's
    // items finish on every CTA (the CTAs run the same schedule), plus fdly ns.
    const int fw = warp - D3_CW - 2, fp = fw >> 1, wi = fw & 1;
    const int per = F_I / 4;
    const int nr = n > 0 ? slot_r0[n - 1] + slot_m[n - 1] : 0;
    const int ntask = n2c > 0 ? (NB == 0 ? (nr * per + D3_CH - 1) / D3_CH : n2c * n) : 0;
    int rounds = 0;
    uint64_t ts_go = 0, ts_addr = 0, ts_iss = 0, ts_chk = 0;
    if ((F_dbg & 32) && fw == 2 && F_trace && n > 0) {
      // DEBUG: round trips during the stream, 16 samples 2us apart, kind i % 3: 0 one 16B load (lane 0), 1 one 16B
      // load per lane (512B), 2 sixteen per lane (8KB, the fetch's pattern); slots 16 + i issue time, 32 + i latency
      const unsigned long long* pa = F_inter + (int64_t)routes[0] * (F_I / 2);
      uint64_t tn = gtime();
      for (int i = 0; i < 16; i++) {
        while (gtime() < tn)
          __nanosleep(64);
        __syncwarp();
        const uint64_t t0 = gtime();
        uint32_t acc = 0;
        if (i % 3 == 0) {
          if (lane == 0) acc = ld_relaxed_2u64(pa).x;
        } else if (i % 3 == 1) {
          acc = ld_relaxed_2u64(pa + 2 * lane).x;
        } else {
          uint4 v[16];
#pragma unroll
          for (int u = 0; u < 16; u++)
            v[u] = ld_relaxed_2u64(pa + 2 * ((u * 32 + lane) % (F_I / 4)));
#pragma unroll
          for (int u = 0; u < 16; u++)
            acc ^= v[u].x;
        }
        if (acc == 0x7fffffffu) misc[8] = 1;  // every value is used before the time is read
        asm volatile("" ::: "memory");
        __syncwarp();
        const uint64_t t1 = gtime();
        if (lane == 0) {
          F_trace[b * D3_TRW + 16 + i] = t0;
          F_trace[b * D3_TRW + 32 + i] = t1 - t0;
        }
        tn = t0 + 2000;
      }
    }
    for (int t = fp; t < ntask; t += D3_NP) {
      int pos0, f0, f1, s;  // first route position, flat range [f0, f1) of 16B words from that position
      uint8_t* bb;
      if (NB == 0) {
        f0 = t * D3_CH, f1 = min(nr * per, f0 + D3_CH), pos0 = 0, bb = rows;
        const int lastpos = (f1 - 1) / per;
        s = 0;
        while (s + 1 < n && slot_r0[s + 1] <= lastpos)
          s++;
      } else {
        s = t % n, pos0 = slot_r0[s], f0 = 0, f1 = slot_m[s] * per, bb = rowbuf(t, s);
        if (t >= NB) mbar_wait_h(bie + 8 * (t % NB), ((t / NB) + 1) & 1, F_whint);
      }
      const int need = min(n1c, ((s + 1) * per1 + G - 1) / G);
      if (lane == 0) {
        while (*reinterpret_cast<volatile int*>(misc + 1) < need)
          __nanosleep(64);
        if (F_fdly > 0) {
          const uint64_t t0 = gtime();
          while (gtime() - t0 < (uint64_t)F_fdly)
            __nanosleep(64);
        }
        ts_go = gtime();
      }
      __syncwarp();
      for (int c0 = f0; c0 < f1; c0 += D3_CH) {
        // One division per chunk: a lane's 16B words are 64 apart, so its (row, column) advance by 64 with a carry
        // (per >= 32). The hot path stays a few hundred instructions: after a read flush every instruction line this
        // warp runs for the first time comes from HBM.
        uint4 v[D3_RL];
        const unsigned long long* sp[D3_RL];
        int doff[D3_RL];
        uint32_t pend = 0;
        {
          const int i0 = c0 + wi * 32 + lane;
          int row = i0 / per, col = i0 - row * per;
#pragma unroll
          for (int u = 0; u < D3_RL; u++) {
            const bool ok = i0 + u * 64 < f1;
            sp[u] = ok ? F_inter + (int64_t)routes[pos0 + row] * (F_I / 2) + col * 2 : F_inter;
            doff[u] = row * rowi + col * 8;
            pend |= (ok ? 1u : 0u) << u;
            col += 64;
            while (col >= per) col -= per, row++;
          }
        }
        // DEBUG (trace builds, not dbg 32): task 0's first chunk, addresses ready, loads issued, first check done
        const bool tr0 = F_trace && !(F_dbg & 32) && t == fp && fp == 0 && wi == 0 && lane == 0 && c0 == f0;
        if (tr0) ts_addr = gtime();
#pragma unroll
        for (int u = 0; u < D3_RL; u++)
          v[u] = ld_relaxed_2u64(sp[u]);
        if (tr0) ts_iss = gtime();
        int rr = 0;
        while (true) {
          rounds++;
#pragma unroll
          for (int u = D3_RL - 1; u >= 0; u--) {
            if (((pend >> u) & 1u) && ((v[u].y == ep && v[u].w == ep) || (F_dbg & 2))) {
              *reinterpret_cast<uint2*>(bb + doff[u]) = make_uint2(v[u].x, v[u].z);
              pend &= ~(1u << u);
            }
          }
          if (tr0 && rr == 0) ts_chk = gtime();
          rr++;
          if (!__any_sync(0xffffffffu, pend != 0)) break;
#pragma unroll
          for (int u = 0; u < D3_RL; u++)
            if ((pend >> u) & 1u) v[u] = ld_relaxed_2u64(sp[u]);
        }
      }
      __syncwarp();
      if (lane == 0) {
        if (F_trace && wi == 0 && t < 3) {
          F_trace[b * D3_TRW + 9 + 2 * t] = gtime();
          F_trace[b * D3_TRW + 8 + 2 * t] = ts_go;
          if (t == 0 && ts_addr) {
            F_trace[b * D3_TRW + 47] = ts_addr;
            F_trace[b * D3_TRW + 40] = ts_iss;
            F_trace[b * D3_TRW + 41] = ts_chk;
          }
        }
        if (F_trace && wi == 0 && t == 0) F_trace[b * D3_TRW + 7] = rounds;
        mbar_arrive(bif + 8 * (NB == 0 ? t : t % NB));
      }
    }
  } else if (warp == D3_CW + 1) {  // epilogue
    float wsum[D3_TMAX];
#pragma unroll
    for (int t = 0; t < D3_TMAX; t++)
      wsum[t] = 0.f;
    for (int it = 0; it < nitems; it++) {
      int kind, s, rb, hp;
      item(it, kind, s, rb, hp);
      const int hb = it & 1, m = slot_m[s], r0 = slot_r0[s];
      mbar_wait_h(brf + 8 * hb, (it >> 1) & 1, A.whint);
      const float* red = redb + hb * (D3_RED / 4);
      const int jr = lane;
      if (kind == 0) {
        const float sc = sgs13[s];
        // full items: gate rows jr (32) at red rows jr, up at 32 + jr; half items: 16 gate rows at red rows jr, 16 up
        // at 16 + jr (lanes 16 .. 31 idle)
        const int h = g1h ? hp >> 2 : hp >> 1, p = g1h ? (hp >> 1) & 1 : hp & 1;
        const int gi = g1h ? 16 * (hp & 1) + (jr & 15) : jr, uo = g1h ? 16 : 32;
        const int col = 128 * rb + 64 * h + dec_kp(32 * p + gi);
        for (int r = 0; r < m; r++) {
          float gt = 0.f, ut = 0.f;
#pragma unroll
          for (int w = 0; w < D3_CW; w++)
            gt += red[(w * 64 + (jr & (uo - 1))) * D3_TMAX + r], ut += red[(w * 64 + uo + (jr & (uo - 1))) * D3_TMAX + r];
          gt *= sc;
          // lanes 2c and 2c + 1 hold adjacent intermediate columns: one tagged word per pair
          const uint32_t hv = __bfloat16_as_ushort(__float2bfloat16(gt / (1.f + __expf(-gt)) * (ut * sc)));
          const uint32_t hn = __shfl_down_sync(0xffffffffu, hv, 1);
          if (!(lane & 1) && jr < uo)
            st_relaxed_u64(A.inter + (int64_t)routes[r0 + r] * (A.I / 2) + col / 2,
                           ((unsigned long long)ep << 32) | (hv | (hn << 16)));
        }
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(bre + 8 * hb);
          *reinterpret_cast<volatile int*>(misc + 1) = it + 1;
          if (TRP) {
            TRP[b * D3_TRW + 14] = gtime();
            if (it == 0) TRP[b * D3_TRW + 15] = gtime();
          }
        }
      } else {
        const float sc = sgs2[s];
        const int row0 = 256 * rb + 32 * hp;
        for (int r = 0; r < m; r++) {
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < D3_CW; w++)
            v += red[(w * 64 + jr) * D3_TMAX + r];
          const int route = routes[r0 + r];
          const __nv_bfloat16 yv = __float2bfloat16(v * sc);
          if (A.tw == nullptr) {
            A.y[(int64_t)route * A.H + row0 + jr] = yv;
          } else {
            const float c = stw[route] * __bfloat162float(yv);
            const int t = route / A.topk;
#pragma unroll
            for (int tt = 0; tt < D3_TMAX; tt++)
              if (tt == t) wsum[tt] += c;
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bre + 8 * hb);
        if (TRP && (A.dbg & 128) && lane == 0 && it - n1c < 8) TRP[b * D3_TRW + 24 + it - n1c] = gtime();
        if (s == n - 1) {
          if (A.tw != nullptr) {
#pragma unroll
            for (int t = 0; t < D3_TMAX; t++) {
              if (t < A.T) A.out[(int64_t)t * A.H + row0 + jr] = __float2bfloat16(wsum[t] * A.scale);
              wsum[t] = 0.f;
            }
          } else {
            for (int r = 0; r < R; r++)
              if ((unsigned)sids[r] >= (unsigned)E) A.y[(int64_t)r * A.H + row0 + jr] = __float2bfloat16(0.f);
          }
        }
      }
    }
    if (n == 0) {
      for (int it = 0; it < n2c; it++) {
        const int j = b + it * G, row0 = 256 * (j >> 3) + 32 * (j & 7);
        const int nr = A.tw != nullptr ? A.T : R;
        __nv_bfloat16* dst = A.tw != nullptr ? A.out : A.y;
        for (int r = 0; r < nr; r++)
          dst[(int64_t)r * A.H + row0 + lane] = __float2bfloat16(0.f);
      }
    }
  } else {  // consumers
    const int g = lane >> 2, q = lane & 3;
    int k = 0;
    uint64_t wb_ = 0, wf_ = 0;
    if (n1c > 0) mbar_wait_h(bx, 0, A.whint);
    for (int it = 0; it < nitems; it++) {
      int kind, s, rb, hp;
      item(it, kind, s, rb, hp);
      const int m = slot_m[s], r0 = slot_r0[s], j2 = it - n1c;
      const int C = kind ? A.C2 : A.C1, nc = kind ? A.nc2 : nc1;
      const uint8_t* brow = nullptr;
      if (kind == 0) {
        if (g < m) brow = dsm + A.off_x + (routes[r0 + g] / A.topk) * rowx;
      } else {
        const uint64_t t0 = TRP ? gtime() : 0;
        if (NB == 0) {
          const int per = A.I / 4;
          for (int c = r0 * per / D3_CH; c <= ((r0 + m) * per - 1) / D3_CH; c++)
            mbar_wait_h(bif + 8 * c, 0, A.whint);
        } else
          mbar_wait_h(bif + 8 * (j2 % NB), (j2 / NB) & 1, A.whint);
        if (TRP) wb_ += gtime() - t0;
        if (TRP && (A.dbg & 128) && tid == 0 && j2 < 8) TRP[b * D3_TRW + 32 + j2] = gtime();
        if (g < m) brow = rowbuf(j2, s) + g * rowi;
      }
      float acc[4][4];
#pragma unroll
      for (int u = 0; u < 4; u++)
#pragma unroll
        for (int i = 0; i < 4; i++)
          acc[u][i] = 0.f;
      const int ks = kind ? k2 : k1;
      const int sub = kind == 1 ? s % G2 : 0;
      const bool glast = kind == 0 || sub == G2 - 1 || s == n - 1;
      for (int j = 0; j < ks; j++) {
        const int st = k % NS;
        if (sub == 0) {
          const uint64_t t0 = TRP ? gtime() : 0;
          mbar_wait_h(bfull + 8 * st, (k / NS) & 1, A.whint);
          if (TRP) wf_ += gtime() - t0;
        }
        if (TRP && tid == 0) TRP[b * D3_TRW + 3] = gtime();
        if (TRP && (A.dbg & 128) && tid == 0 && kind == 1 && j == 0 && j2 < 8) TRP[b * D3_TRW + 16 + j2] = gtime();
        const uint8_t* sb = ring + st * D3_STB + sub * 1152 * A.nc2;
        const int nv = min(nc, C - j * nc);
        if (A.dbg & 1) {
        } else if (kind == 0 && g1h) {
          const int uo[4] = {0, 576 * nc, 0, 0};
          d3_stage<2>(acc, sb, uo, 576, nv, brow, j * nc, warp, lane);
        } else if (kind == 0) {
          const int uo[4] = {0, 576, 1152 * nc, 1152 * nc + 576};
          d3_stage<4>(acc, sb, uo, 1152, nv, brow, j * nc, warp, lane);
        } else {
          const int uo[4] = {0, 576, 0, 0};
          d3_stage<2>(acc, sb, uo, 1152, nv, brow, j * nc, warp, lane);
        }
        if (glast) {
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * st);
          k++;
        }
      }
      if (kind == 1 && NB != 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(bie + 8 * (j2 % NB));
      }
      const int hb = it & 1;
      if (it >= 2) mbar_wait_h(bre + 8 * hb, ((it >> 1) + 1) & 1, A.whint);
      float* red = redb + hb * (D3_RED / 4);
      const int U = kind ? 2 : (g1h ? 2 : 4);
#pragma unroll
      for (int u = 0; u < 4; u++) {
        if (u < U) {
          float* rp = red + (warp * 64 + 16 * u + g) * D3_TMAX;
          if (2 * q < m) rp[2 * q] = acc[u][0], rp[8 * D3_TMAX + 2 * q] = acc[u][2];
          if (2 * q + 1 < m) rp[2 * q + 1] = acc[u][1], rp[8 * D3_TMAX + 2 * q + 1] = acc[u][3];
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(brf + 8 * hb);
    }
    if (TRP && tid == 0) TRP[b * D3_TRW + 5] = wb_, TRP[b * D3_TRW + 6] = wf_;
  }
  __syncthreads();
  if (TRP && tid == 0) TRP[b * D3_TRW + 4] = gtime();
  // The last CTA to finish advances the epoch (the next call's reads of it are ordered by the kernel boundary).
  if (tid == 0) {
    __threadfence();
    if (atom_add_acq_rel(A.state, 1) == G - 1) {
      A.state[1] = (int)ep;
      A.state[0] = 0;
    }
  }
}

// cfg: bits 0-7 dbg (64: return at entry), 8-11 ring stages cap (0: 2), 12-13 GEMM1 half items (0 auto, 1 off, 2 on),
// 16-23 fdly in 10ns (0xff: the default), 24-27 wait suspend hint 32ns << k (0 spin, 15 the default 2us), 28-31 shared
// memory cap in 16KB (0 or past the max: the max).
inline void run_decode3(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView inter,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView state,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    int64_t ctas,
    int64_t cfg,
    int64_t trace) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto E = SymbolicSize{"local_experts"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y);
  TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).ensure_alignment(16).verify(inter);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(state);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), r = R.unwrap(), e = E.unwrap();
  RuntimeCheck(w13.dim() == 5, "nvfp4_w4a16_sm90: packed w13 must be 5-D");
  const int64_t i = w13.size(1) * 128;
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(t >= 1 && t <= D3_TMAX && r <= D3_RMAX, "nvfp4_w4a16_sm90 decode3: at most ", D3_TMAX, " tokens and ",
               D3_RMAX, " routes");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(i >= 128 && i % 128 == 0, "nvfp4_w4a16_sm90: intermediate size must be a multiple of 128, got ", i);
  RuntimeCheck(e >= 1 && e <= D3_EMAX, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  const int RB1 = (int)(2 * i / 256), RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, RB1, C1, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  RuntimeCheck(state.size(0) >= 2, "nvfp4_w4a16_sm90 decode3: state too small");
  RuntimeCheck(inter.size(0) >= r * i / 2, "nvfp4_w4a16_sm90 decode3: tagged intermediate too small");
  const int off_x = D3_FIXED;
  const int64_t oring = (off_x + t * (2 * h + 16) + 1023) / 1024 * 1024;
  // the rotating fallback needs two stages and two buffers of t rows
  RuntimeCheck(oring + 2 * D3_STB + 2 * t * (2 * i + 16) <= D3_SMEM_MAX,
               "nvfp4_w4a16_sm90 decode3: tokens x width too large for shared memory");
  const int nc1 = std::min(16, C1), nc1h = std::min(32, C1), nc2 = std::min(32, C2);
  auto wmap = [&](void* base, int C, int64_t blocks, int nc, int box0) {
    return host::tma::make_map(
        base,
        CU_TENSOR_MAP_DATA_TYPE_UINT64,
        {144, 8, (cuuint64_t)C, (cuuint64_t)blocks},
        {1152, 9216, (cuuint64_t)C * 9216},
        {(cuuint32_t)box0, 1, (cuuint32_t)nc, 1},
        {.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B});
  };
  D3Maps mp;
  mp.w13 = wmap(w13.data_ptr(), C1, e * RB1, nc1, 144);
  mp.w13h = wmap(w13.data_ptr(), C1, e * RB1, nc1h, 72);
  mp.w2 = wmap(w2.data_ptr(), C2, e * RB2, nc2, 144);
  D3Args A;
  A.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  A.ids = static_cast<const int*>(topk_ids.data_ptr());
  A.tw = topk_weights.has_value() ? static_cast<const float*>(topk_weights.value().data_ptr()) : nullptr;
  A.gs13 = static_cast<const float*>(gs13.data_ptr());
  A.gs2 = static_cast<const float*>(gs2.data_ptr());
  A.inter = static_cast<unsigned long long*>(inter.data_ptr());
  A.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  A.out = out.has_value() ? static_cast<__nv_bfloat16*>(out.value().data_ptr()) : nullptr;
  A.state = static_cast<int*>(state.data_ptr());
  A.scale = (float)scale;
  A.T = (int)t, A.topk = (int)topk, A.E = (int)e, A.H = (int)h, A.I = (int)i;
  A.RB1 = RB1, A.RB2 = RB2, A.C1 = C1, A.C2 = C2, A.nc1 = nc1, A.nc1h = nc1h, A.nc2 = nc2;
  A.off_x = off_x;
  A.smem = D3_SMEM_MAX;
  const int nscap = (int)((cfg >> 8) & 15);
  A.nsmax = std::min(D3_NSMAX, nscap ? nscap : 2);  // two stages measured fastest at every shape (runs/a3.log)
  A.g1h = (int)((cfg >> 12) & 3);
  const int fd = (int)((cfg >> 16) & 0xff);
  A.fdly = fd == 0xff ? 1000 : 10 * fd;
  const int hk = (int)((cfg >> 24) & 15);
  A.whint = hk == 15 ? 2048 : hk == 0 ? 0 : 32 << hk;
  const int smcap = (int)((cfg >> 28) & 15);
  if (smcap > 0) A.smem = std::min(A.smem, smcap * 16384);
  RuntimeCheck(oring + 2 * D3_STB + 2 * t * (2 * i + 16) <= A.smem, "nvfp4_w4a16_sm90 decode3: shared memory cap too small");
  A.dbg = (int)(cfg & 0xff);
  A.trace = reinterpret_cast<unsigned long long*>(trace);
  const int grid = ctas > 0 ? (int)std::min<int64_t>(ctas, sms) : sms;
  if (A.trace != nullptr) {
    runtime::set_max_dynamic_smem_per_device<dec3_k<true>>(dev, D3_SMEM_MAX);
    LaunchKernel(grid, D3_THREADS, dev, A.smem).enable_cooperative(!(A.dbg & 16))(dec3_k<true>, mp, A);
  } else {
    runtime::set_max_dynamic_smem_per_device<dec3_k<false>>(dev, D3_SMEM_MAX);
    LaunchKernel(grid, D3_THREADS, dev, A.smem).enable_cooperative(!(A.dbg & 16))(dec3_k<false>, mp, A);
  }
}

}  // namespace sglang::nvfp4_w4a16_sm90
