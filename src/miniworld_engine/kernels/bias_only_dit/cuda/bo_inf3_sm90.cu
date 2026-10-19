// bo_inf3_sm90.cu -- the bias-only token DiT's bf16 inference step on H100 (sm_90a) as three kernels per block, the Hopper twin of
// B200's bo_front_bf16.cu / pv_gate_inf -DPDL_INF / bo_tail_bf16.cu (runner._step3h). OPT-IN (inf3_h100._ON): measured 1.4-1.6x slower than
// the 12-launch step's same launches on H100 -- the column-split per-CTA tiles (128 x 96 / 128 x 192) are bound by the ~70 GB/s TMA
// intake per SM (docs/records/h100_bias_only_dit_inf3_20261019.md):
//
//   front   xa  = LN(x) sigmoid(s1) + sh1,  v | g = xa [Wv; Wg]^T                                   (bo90_front)
//   core    a   = sigmoid(g) (P v)                                                                  (pv_gate_inf.cu pvx, PDL launch)
//   tail    y   = a Wo^T,  x1 = x + sigmoid(gate1) y,  xt = LN(x1) sigmoid(s2) + sh2,
//           h   = silu(xt Wa^T) (xt Wb^T),  z = h Wsq^T,  out = x1 + sigmoid(gate2) z                (bo90_tail)
//
// wgmma m64nNk16 (bf16 operands, fp32 accumulators), every operand a K-major 128-B-swizzled [rows][64] bf16 tile filled by TMA. x is
// the block input (bf16: the step's single, -DXBF / XBF = true; else the fp32 residual of the previous block); out is fp32 (the next
// block's x) or bf16 (the step's output, OBF). The conditioning tables are the 12-launch step's (raw, the sigmoids applied here):
// G1 [T, 4, 768] = (s1, sh1, s2, sh2), G2 [T, 2, 768] = (gate1, gate2), row tok = row % T (T a multiple of 128), any row stride.
//
// Geometry (both kernels): a cluster of CL CTAs (8, or 16 non-portable) per 128-row tile. CTA c of the cluster owns
//   front: x / xa columns [NXC c, + NXC) (NXC = 768 / CL) and v | g columns [NV c, + NV) (NV = 2 DA / CL);
//   tail:  output columns [NC c, + NC) (NC = 768 / CL) and hidden units [NH c, + NH) (NH = 1536 / CL), the a | b GEMM in NH / 96
//          passes of 96 hidden units (n192 = 96 a rows + 96 b rows of [Wa; Wb]).
// Exchanges inside the cluster: the LayerNorm statistics as per-CTA (mean, M2) over its own columns, pushed to every peer through DSMEM
// (cp.async.bulk shared::cluster, complete_tx on the peer's xch barrier) and merged in rank order (Chan; bit-identical in every CTA);
// the full-K GEMM operands xa / xt / h through L2 scratch: every CTA stores its own columns (st.global), then ONE release
// (fence.proxy.async.global per thread, a named barrier, fence.acq_rel.cluster, CL relaxed remote arrivals on ready), and the A
// producers stream the k-blocks back by TMA, MULTICAST: k-block position i is issued by rank i % CL into every CTA of the cluster.
//
// Warp roles (384 threads): warp 0 lane 0 the A producer (activations: a / xa / xt / h k-blocks, [128][64] = 16 KB boxes, a ring of NAR
// slots); warp 1 lane 0 the W producer (the CTA's weight rows, a ring of NWS slots, issued BEFORE the PDL wait: the weights are packed
// once and never written by a step); warpgroups 1 and 2 the consumers, rows 64 (wg - 1) .. + 63 of the tile (each one m64 slab of every
// MMA), also every row pass. Ring protocol: afull (1 arrival + tx; the multicast's bytes land on every CTA's barrier at the slot's
// offset), aempty (2 CL arrivals: every consumer warpgroup of the cluster releases a slot remotely in every CTA, so the issuer of the
// next box knows the whole cluster has read it), wfull / wempty (local: 1 + tx / 2). Operand slots are only read by wgmma (async proxy),
// so no proxy fence sits before a release. The last cluster barrier keeps every CTA alive until no remote arrival / multicast / DSMEM
// push can target it.
// PDL: griddepcontrol.launch_dependents after setup; the consumers and the A producer griddepcontrol.wait before their first read of
// a kernel-made tensor and before their first store; the W producer never waits.
// Bit-identical reruns: fixed-order reductions, no atomics.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_ptx.cuh"
using namespace tmn;
#include "wgmma_n.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace bo3 {
#define DEVI __device__ __forceinline__
using bf = __nv_bfloat16;

// ------------------------------------------------------------------------------------------------ PTX
TMN_DEVI void mma_n48(float (&d)[24], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_n96(float (&d)[48], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]) : "l"(a), "l"(b));
}
template <int N> DEVI void mma_n(float (&d)[N / 2], uint64_t a, uint64_t b) {
  if constexpr (N == 48) mma_n48(d, a, b);
  else if constexpr (N == 96) mma_n96(d, a, b);
  else if constexpr (N == 128) mma_n128(d, a, b);
  else if constexpr (N == 192) mma_n192(d, a, b);
  else { static_assert(N == 256, "wgmma n"); mma_n256(d, a, b); }
}
DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }       // K-major, 128-B swizzle, 8-row groups 1 KB apart
DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEVI uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
DEVI void arrive_remote(uint64_t* bar, uint32_t rank) {                         // a slot release (the default .release.cta, as
  // CUTLASS's ClusterBarrier::arrive(cta): the reads it orders are wgmma's, complete at wgmma.wait_group; .release.cluster costs ~0.5 us each)
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(bar), rank)) : "memory");
}
DEVI void arrive_remote_relaxed(uint64_t* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(bar), rank)) : "memory");
}
DEVI void wait_cl(uint64_t* b, uint32_t parity) {                               // acquire at cluster scope (remote arrivals)
  uint32_t ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
}
DEVI void tma_ld(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_mc(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
DEVI void push_bulk(uint32_t dst_local, uint32_t src, uint32_t bytes, uint32_t bar_local, uint32_t rank) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(mapa(dst_local, rank)), "r"(src), "r"(bytes), "r"(mapa(bar_local, rank)) : "memory");
}
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI void fence_acq_rel_cluster() { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
DEVI void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVI float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
DEVI float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
DEVI uint32_t pk(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint32_t ldg_u32(const void* p) { uint32_t v; asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
DEVI float2 ldg_f2(const void* p) { float2 v; asm volatile("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p)); return v; }
DEVI float2 ldx2(const void* p, bool xbf) { return xbf ? bf2(ldg_u32(p)) : ldg_f2(p); }

// %globaltimer trace (probes only; g_tron set from the host by trace_on): [16 CTAs][128 events]
__device__ unsigned long long g_tr[16 * 128];
__device__ int g_tron = 0;
// 1: activation k-blocks multicast across the cluster (issuer = position % CL, cluster-wide slot release); 0: every CTA loads its own (A/B)
__device__ int g_mc = 1;
DEVI void TR(int e) {
  if (g_tron && blockIdx.x < 16) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_tr[blockIdx.x * 128 + e] = t; }
}
constexpr int D = 768, QM = 128, KB = 64, SLOT = QM * 128, NAR = 6, SMEM_MAX = 232448;
constexpr int WSL = 24576;                                                       // tail W slot: 2 x 96 rows of a k-block
constexpr uint16_t mask_of(int cl) { return (uint16_t)((1u << cl) - 1u); }

// One GEMM phase of a consumer warpgroup: NK k-blocks, N columns; A positions ia.. of the A ring (this warpgroup's 64-row slab at
// +cw 8 KB), W positions jw.. of the W ring holding KPS k-blocks of N rows each (B at +(k % KPS) N 128 B). Every A slot is released to
// every CTA of the cluster, each W slot locally once its last k-block is read.
template <int N, int KPS, int CL, int NWS, int WSZ = WSL>
DEVI void gemm_phase(float (&acc)[N / 2], int NK, int& ia, int& jw, uint32_t sa, uint32_t sw, uint64_t* afull, uint64_t* aempty,
                     uint64_t* wfull, uint64_t* wempty, int cw, int wtid, int tev = -1) {
#pragma unroll
  for (int e = 0; e < N / 2; ++e) acc[e] = 0.f;
  int pa = -1, pw = -1;
  for (int k = 0; k < NK; ++k) {
    const int s = ia % NAR, ws = jw % NWS;
    mbar_wait(afull + s, (uint32_t)(ia / NAR) & 1u);
    if (tev >= 0 && cw == 0 && wtid == 0) TR(tev + k);
    if (k % KPS == 0) mbar_wait(wfull + ws, (uint32_t)(jw / NWS) & 1u);
    const uint32_t a0 = sa + s * SLOT + cw * 8192, b0 = sw + ws * WSZ + (k % KPS) * N * 128;
    fence_regs(acc);
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_n<N>(acc, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
    wgmma_commit();
    wgmma_wait<1>();
    fence_regs(acc);
    if (wtid == 0) {
      if (pa >= 0) {
        if (g_mc) for (int r = 0; r < CL; ++r) arrive_remote(aempty + pa, (uint32_t)r);
        else mbar_arrive(aempty + pa);
      }
      if (pw >= 0) mbar_arrive(wempty + pw);
    }
    pa = s;
    pw = -1;
    ++ia;
    if (k % KPS == KPS - 1) { pw = ws; ++jw; }
  }
  wgmma_wait<0>();
  fence_regs(acc);
  if (wtid == 0) {
    if (g_mc) for (int r = 0; r < CL; ++r) arrive_remote(aempty + pa, (uint32_t)r);
    else mbar_arrive(aempty + pa);
    if (pw >= 0) mbar_arrive(wempty + pw);
  }
}

// The A producer's position i: wait until the cluster released the slot, arm this CTA's barrier for the box, issue it if it is ours.
template <int CL>
DEVI void a_issue(int i, uint32_t sa, uint64_t* afull, uint64_t* aempty, const CUtensorMap* m, int c0, int c1, uint32_t rank) {
  const int s = i % NAR;
  if (i >= NAR) mbar_wait(aempty + s, (uint32_t)((i / NAR) - 1) & 1u);
  TR(64 + i);
  mbar_arrive_expect_tx(afull + s, SLOT);
  if (!g_mc) tma_ld(sa + s * SLOT, m, afull + s, c0, c1);
  else if (i % CL == (int)rank) tma_mc(sa + s * SLOT, m, afull + s, c0, c1, mask_of(CL));
}

// Row statistics over the CTA's own columns -> red[rank] -> every peer (DSMEM push) -> merged (mean, rstd) of the full 768-wide row.
// Each consumer thread passes its (mean, M2) partial of row rl (only quad leader lanes of the tail / even threads of the front write).
template <int CL, int NCOL>
DEVI void stats_exchange(float* red, uint32_t rank, uint64_t* xch, uint32_t sred) {
  fence_proxy_async();                                                           // red[rank] (st.shared) -> the bulk copies
  named_bar_sync(1, 256);
  if (threadIdx.x == 128)
    for (int p = 1; p < CL; ++p)
      push_bulk(sred + rank * 1024, sred + rank * 1024, 1024, smem_u32(xch), (rank + (uint32_t)p) % CL);
  wait_cl(xch, 0);
}
template <int CL, int NCOL>
DEVI void stats_merge(const float* red, int rl, float eps, float& mean, float& rstd) {
  const float2* r2 = reinterpret_cast<const float2*>(red);
  float mu = 0.f;
#pragma unroll
  for (int p = 0; p < CL; ++p) mu += r2[p * 128 + rl].x;
  mu *= 1.f / CL;
  float m2 = 0.f;
#pragma unroll
  for (int p = 0; p < CL; ++p) {
    const float2 v = r2[p * 128 + rl];
    const float d = v.x - mu;
    m2 += v.y + (float)NCOL * d * d;
  }
  mean = mu;
  rstd = rsqrtf(m2 * (1.f / D) + eps);
}
// the one release of a cluster exchange through L2: this thread's st.global -> the peers' TMA loads
DEVI void release_ready(uint64_t* ready, int cl) {
  fence_proxy_async_global();
  named_bar_sync(1, 256);
  if (threadIdx.x == 128) {
    fence_acq_rel_cluster();
    for (int p = 0; p < cl; ++p) arrive_remote_relaxed(ready, (uint32_t)p);
  }
}

// ------------------------------------------------------------------------------------------------ front
template <int CL, int DA> struct FrontCfg {
  static constexpr int NXC = D / CL, NV = 2 * DA / CL, WSLOT = NV * 128;
  static constexpr int O_W = NAR * SLOT, O_RED_ = 0;
  static constexpr int NWS = (SMEM_MAX - O_W - CL * 1024 - 1024) / WSLOT > 6 ? 6 : (SMEM_MAX - O_W - CL * 1024 - 1024) / WSLOT;
  static constexpr int O_RED = O_W + NWS * WSLOT, O_BAR = O_RED + CL * 1024, SMEM = O_BAR + 512;
  static_assert(NWS >= 3 && SMEM <= SMEM_MAX, "front shared memory");
  static_assert(NV <= 256 && NV % 16 == 0 && (NXC / 2) % 8 == 0, "front tiling");
};
struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[8], wempty[8], xch, ready, ready2;
};

template <int CL, int DA, bool XBF>
__global__ void __launch_bounds__(384, 1)
bo90_front(const __grid_constant__ CUtensorMap mxa, const __grid_constant__ CUtensorMap mw, const void* __restrict__ X,
           const bf* __restrict__ G1, int g1s, bf* __restrict__ XA, bf* __restrict__ VG, int T, float eps) {
  using C = FrontCfg<CL, DA>;
  constexpr int NXC = C::NXC, NV = C::NV, NWS = C::NWS, HX = NXC / 2;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + C::O_BAR);
  float* red = reinterpret_cast<float*>(sm + C::O_RED);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const uint32_t c = cl_rank();
  const int m0 = (int)(blockIdx.x / CL) * QM, tr0 = m0 % T;
  if (tid == 0) {
    for (int s = 0; s < NAR; ++s) { mbar_init(B.afull + s, 1); mbar_init(B.aempty + s, g_mc ? 2 * CL : 2); }
    for (int s = 0; s < NWS; ++s) { mbar_init(B.wfull + s, 1); mbar_init(B.wempty + s, 2); }
    mbar_init(&B.xch, 1);
    mbar_init(&B.ready, CL);
    mbar_arrive_expect_tx(&B.xch, (CL - 1) * 1024);                             // the peers' statistics
    fence_barrier_init();
  }
  __syncthreads();
  cluster_sync();
  pdl_launch();
  if (tid == 128) TR(0);
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (warp == 1 && lane == 0) {                                                // W producer: [Wv; Wg] rows [NV c, + NV), all 12 k-blocks
      for (int k = 0; k < D / KB; ++k) {
        const int ws = k % NWS;
        if (k >= NWS) mbar_wait(B.wempty + ws, (uint32_t)((k / NWS) - 1) & 1u);
        mbar_arrive_expect_tx(B.wfull + ws, C::WSLOT);
        tma_ld(su + C::O_W + ws * C::WSLOT, &mw, B.wfull + ws, KB * k, NV * (int)c);
      }
    } else if (warp == 0 && lane == 0) {                                         // A producer: xa k-blocks, multicast
      pdl_wait();
      wait_cl(&B.ready, 0);
      fence_proxy_async_global();
      for (int k = 0; k < D / KB; ++k) a_issue<CL>(k, su, B.afull, B.aempty, &mxa, KB * k, m0, c);
    }
    __syncwarp();
  } else {
    setmaxnreg_inc<232>();
    const int cw = wg - 1;
    pdl_wait();
    // ---- statistics of the own columns: thread = (row rl, half hf), HX consecutive columns
    const int rl = 64 * cw + (wtid >> 1), hf = wtid & 1, row = m0 + rl, tok = tr0 + rl;
    const int col0 = NXC * (int)c + HX * hf;
    float xv[HX];
    if constexpr (XBF) {
      const uint4* p = reinterpret_cast<const uint4*>(reinterpret_cast<const bf*>(X) + (size_t)row * D + col0);
#pragma unroll
      for (int u = 0; u < HX / 8; ++u) {
        const uint4 q = __ldg(p + u);
        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 f = bf2(w[e]); xv[8 * u + 2 * e] = f.x; xv[8 * u + 2 * e + 1] = f.y; }
      }
    } else {
      const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(X) + (size_t)row * D + col0);
#pragma unroll
      for (int u = 0; u < HX / 4; ++u) {
        const float4 q = __ldg(p + u);
        xv[4 * u] = q.x; xv[4 * u + 1] = q.y; xv[4 * u + 2] = q.z; xv[4 * u + 3] = q.w;
      }
    }
    uint4 s1v[HX / 8], h1v[HX / 8];                                              // the tables, issued before the exchange
    {
      const uint4* ps = reinterpret_cast<const uint4*>(G1 + (size_t)tok * g1s + col0);
      const uint4* ph = reinterpret_cast<const uint4*>(G1 + (size_t)tok * g1s + D + col0);
#pragma unroll
      for (int u = 0; u < HX / 8; ++u) { s1v[u] = __ldg(ps + u); h1v[u] = __ldg(ph + u); }
    }
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < HX; ++e) s += xv[e];
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    const float mc = s * (1.f / NXC);
    float q = 0.f;
#pragma unroll
    for (int e = 0; e < HX; ++e) { const float d = xv[e] - mc; q += d * d; }
    q += __shfl_xor_sync(0xffffffffu, q, 1);
    if (hf == 0) reinterpret_cast<float2*>(red)[c * 128 + rl] = make_float2(mc, q);
    if (tid == 128) TR(1);
    stats_exchange<CL, NXC>(red, c, &B.xch, su + C::O_RED);
    if (tid == 128) TR(2);
    float mean, rstd;
    stats_merge<CL, NXC>(red, rl, eps, mean, rstd);
    // ---- xa = LN(x) sigmoid(s1) + sh1 of the own columns -> XA (L2), one release
    uint4* xo = reinterpret_cast<uint4*>(XA + (size_t)row * D + col0);
#pragma unroll
    for (int u = 0; u < HX / 8; ++u) {
      const uint32_t sw_[4] = {s1v[u].x, s1v[u].y, s1v[u].z, s1v[u].w}, hw[4] = {h1v[u].x, h1v[u].y, h1v[u].z, h1v[u].w};
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 sv = bf2(sw_[e]), hv = bf2(hw[e]);
        o[e] = pk((xv[8 * u + 2 * e] - mean) * rstd * sg(sv.x) + hv.x, (xv[8 * u + 2 * e + 1] - mean) * rstd * sg(sv.y) + hv.y);
      }
      xo[u] = make_uint4(o[0], o[1], o[2], o[3]);
    }
    release_ready(&B.ready, CL);
    if (tid == 128) TR(3);
    // ---- v | g = xa W^T: this warpgroup's 64 rows x NV columns
    float acc[NV / 2];
    int ia = 0, jw = 0;
    gemm_phase<NV, 1, CL, NWS, C::WSLOT>(acc, D / KB, ia, jw, su, su + C::O_W, B.afull, B.aempty, B.wfull, B.wempty, cw, wtid, 4);
    if (tid == 128) TR(127);
    const int r0 = m0 + 64 * cw + 16 * warp + (lane >> 2), cq = 2 * (lane & 3);
#pragma unroll
    for (int j = 0; j < NV / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int ix = 4 * j + 2 * rb;
        *reinterpret_cast<uint32_t*>(VG + (size_t)(r0 + 8 * rb) * (2 * DA) + NV * (int)c + 8 * j + cq) = pk(acc[ix], acc[ix + 1]);
      }
  }
  __syncwarp();
  cluster_sync();                                                                // no remote arrival / multicast lands on an exited CTA
}

// ------------------------------------------------------------------------------------------------ tail
template <int CL, int DA> struct TailCfg {
  static constexpr int NC = D / CL, NH = 1536 / CL, HP = 96, NPASS = NH / HP, NKA = DA / KB, NHK = 1536 / KB;
  static constexpr int KPS = WSL / (NC * 128);                                   // Wo / Wsq k-blocks per W slot
  static constexpr int O_W = NAR * SLOT;
  static constexpr int NWS = (SMEM_MAX - O_W - CL * 1024 - 1024) / WSL > 6 ? 6 : (SMEM_MAX - O_W - CL * 1024 - 1024) / WSL;
  static constexpr int O_RED = O_W + NWS * WSL, O_BAR = O_RED + CL * 1024, SMEM = O_BAR + 512;
  static_assert(NWS >= 3 && SMEM <= SMEM_MAX, "tail shared memory");
  static_assert(NH % HP == 0 && NKA % KPS == 0 && NHK % KPS == 0 && KPS * NC * 128 == WSL, "tail tiling");
};

template <int CL, int DA, bool XBF, bool OBF>
__global__ void __launch_bounds__(384, 1)
bo90_tail(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mxt, const __grid_constant__ CUtensorMap mh,
          const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwsq,
          const void* __restrict__ X, const bf* __restrict__ G1, int g1s, const bf* __restrict__ G2, int g2s, bf* __restrict__ XT,
          bf* __restrict__ HB, void* __restrict__ OUT, int T, float eps) {
  using C = TailCfg<CL, DA>;
  constexpr int NC = C::NC, NH = C::NH, HP = C::HP, NPASS = C::NPASS, NKA = C::NKA, NHK = C::NHK, KPS = C::KPS, NWS = C::NWS;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + C::O_BAR);
  float* red = reinterpret_cast<float*>(sm + C::O_RED);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const uint32_t c = cl_rank();
  const int m0 = (int)(blockIdx.x / CL) * QM, tr0 = m0 % T;
  if (tid == 0) {
    for (int s = 0; s < NAR; ++s) { mbar_init(B.afull + s, 1); mbar_init(B.aempty + s, g_mc ? 2 * CL : 2); }
    for (int s = 0; s < NWS; ++s) { mbar_init(B.wfull + s, 1); mbar_init(B.wempty + s, 2); }
    mbar_init(&B.xch, 1);
    mbar_init(&B.ready, CL);                                                     // xt in L2
    mbar_init(&B.ready2, CL);                                                    // h in L2
    mbar_arrive_expect_tx(&B.xch, (CL - 1) * 1024);
    fence_barrier_init();
  }
  __syncthreads();
  cluster_sync();
  pdl_launch();
  if (tid == 128) TR(0);
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (warp == 1 && lane == 0) {                                                // W producer: Wo, [Wa; Wb] per pass, Wsq
      int j = 0;
      auto wslot = [&](uint32_t bytes) -> uint32_t {
        const int ws = j % NWS;
        if (j >= NWS) mbar_wait(B.wempty + ws, (uint32_t)((j / NWS) - 1) & 1u);
        mbar_arrive_expect_tx(B.wfull + ws, bytes);
        return (uint32_t)ws;
      };
      for (int k = 0; k < NKA; k += KPS, ++j) {
        const uint32_t ws = wslot(WSL), dst = su + C::O_W + ws * WSL;
        for (int u = 0; u < KPS; ++u) tma_ld(dst + u * NC * 128, &mwo, B.wfull + ws, KB * (k + u), NC * (int)c);
      }
      for (int p = 0; p < NPASS; ++p)
        for (int k = 0; k < D / KB; ++k, ++j) {
          const uint32_t ws = wslot(WSL), dst = su + C::O_W + ws * WSL;
          tma_ld(dst, &mwab, B.wfull + ws, KB * k, NH * (int)c + HP * p);
          tma_ld(dst + HP * 128, &mwab, B.wfull + ws, KB * k, 1536 + NH * (int)c + HP * p);
        }
      for (int k = 0; k < NHK; k += KPS, ++j) {
        const uint32_t ws = wslot(WSL), dst = su + C::O_W + ws * WSL;
        for (int u = 0; u < KPS; ++u) tma_ld(dst + u * NC * 128, &mwsq, B.wfull + ws, KB * (k + u), NC * (int)c);
      }
    } else if (warp == 0 && lane == 0) {                                         // A producer: a, xt (per pass), h; multicast
      pdl_wait();
      int i = 0;
      for (int k = 0; k < NKA; ++k, ++i) a_issue<CL>(i, su, B.afull, B.aempty, &ma, KB * k, m0, c);
      wait_cl(&B.ready, 0);
      fence_proxy_async_global();
      for (int p = 0; p < NPASS; ++p)
        for (int k = 0; k < D / KB; ++k, ++i) a_issue<CL>(i, su, B.afull, B.aempty, &mxt, KB * k, m0, c);
      wait_cl(&B.ready2, 0);
      fence_proxy_async_global();
      for (int k = 0; k < NHK; ++k, ++i) a_issue<CL>(i, su, B.afull, B.aempty, &mh, KB * k, m0, c);
    }
    __syncwarp();
  } else {
    setmaxnreg_inc<232>();
    const int cw = wg - 1;
    pdl_wait();
    const int rq = 16 * warp + (lane >> 2), cq = 2 * (lane & 3);                 // fragment row (and + 8), column pair
    const int rl0 = 64 * cw + rq, row0 = m0 + rl0, tok0 = tr0 + rl0;
    const int ccol = NC * (int)c + cq;                                           // + 8 j
    // ---- prefetch x, gate1, s2, sh2 of this thread's fragment positions (their latency hides under P1)
    float2 xv[NC / 4];
    uint32_t g1v[NC / 4], s2v[NC / 4], h2v[NC / 4];
#pragma unroll
    for (int j = 0; j < NC / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int u = 2 * j + rb, col = ccol + 8 * j;
        const size_t row = (size_t)(row0 + 8 * rb), tok = (size_t)(tok0 + 8 * rb);
        xv[u] = XBF ? bf2(ldg_u32(reinterpret_cast<const bf*>(X) + row * D + col)) : ldg_f2(reinterpret_cast<const float*>(X) + row * D + col);
        g1v[u] = ldg_u32(G2 + tok * g2s + col);
        s2v[u] = ldg_u32(G1 + tok * g1s + 2 * D + col);
        h2v[u] = ldg_u32(G1 + tok * g1s + 3 * D + col);
      }
    int ia = 0, jw = 0;
    // ---- P1: y = a Wo^T
    float x1[NC / 2];
    {
      float acc[NC / 2];
      gemm_phase<NC, KPS, CL, NWS>(acc, NKA, ia, jw, su, su + C::O_W, B.afull, B.aempty, B.wfull, B.wempty, cw, wtid, 4);
      // ---- P2: x1 = x + sigmoid(gate1) y; (mean, M2) of the own NC columns per row
#pragma unroll
      for (int j = 0; j < NC / 8; ++j)
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int u = 2 * j + rb, ix = 4 * j + 2 * rb;
          const float2 g = bf2(g1v[u]);
          x1[ix] = xv[u].x + sg(g.x) * acc[ix];
          x1[ix + 1] = xv[u].y + sg(g.y) * acc[ix + 1];
        }
    }
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < NC / 8; ++j) s += x1[4 * j + 2 * rb] + x1[4 * j + 2 * rb + 1];
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      const float mc = s * (1.f / NC);
      float q = 0.f;
#pragma unroll
      for (int j = 0; j < NC / 8; ++j) {
        const float d0 = x1[4 * j + 2 * rb] - mc, d1 = x1[4 * j + 2 * rb + 1] - mc;
        q += d0 * d0 + d1 * d1;
      }
      q += __shfl_xor_sync(0xffffffffu, q, 1);
      q += __shfl_xor_sync(0xffffffffu, q, 2);
      if ((lane & 3) == 0) reinterpret_cast<float2*>(red)[c * 128 + rl0 + 8 * rb] = make_float2(mc, q);
    }
    // ---- P3: the cluster's statistics
    if (tid == 128) TR(1);
    stats_exchange<CL, NC>(red, c, &B.xch, su + C::O_RED);
    if (tid == 128) TR(2);
    float mean[2], rstd[2];
    stats_merge<CL, NC>(red, rl0, eps, mean[0], rstd[0]);
    stats_merge<CL, NC>(red, rl0 + 8, eps, mean[1], rstd[1]);
    // ---- P4: xt = LN(x1) sigmoid(s2) + sh2 -> XT (L2), one release
#pragma unroll
    for (int j = 0; j < NC / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int u = 2 * j + rb, ix = 4 * j + 2 * rb;
        const float2 sv = bf2(s2v[u]), hv = bf2(h2v[u]);
        const float v0 = (x1[ix] - mean[rb]) * rstd[rb] * sg(sv.x) + hv.x, v1 = (x1[ix + 1] - mean[rb]) * rstd[rb] * sg(sv.y) + hv.y;
        *reinterpret_cast<uint32_t*>(XT + (size_t)(row0 + 8 * rb) * D + ccol + 8 * j) = pk(v0, v1);
      }
    release_ready(&B.ready, CL);
    if (tid == 128) TR(3);
    // ---- P5 / P6: per pass, a | b = xt [Wa; Wb]^T (96 hidden units), h = silu(a) b -> H (L2)
#pragma unroll 1
    for (int p = 0; p < NPASS; ++p) {
      float acc[2 * HP / 2];
      gemm_phase<2 * HP, 1, CL, NWS>(acc, D / KB, ia, jw, su, su + C::O_W, B.afull, B.aempty, B.wfull, B.wempty, cw, wtid, 16 + 12 * p);
#pragma unroll
      for (int j = 0; j < HP / 8; ++j)
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int ix = 4 * j + 2 * rb, ib = ix + HP / 2;
          const float a0 = acc[ix], a1 = acc[ix + 1];
          const float h0 = a0 * sg(a0) * acc[ib], h1 = a1 * sg(a1) * acc[ib + 1];
          *reinterpret_cast<uint32_t*>(HB + (size_t)(row0 + 8 * rb) * 1536 + NH * (int)c + HP * p + 8 * j + cq) = pk(h0, h1);
        }
    }
    release_ready(&B.ready2, CL);
    if (tid == 128) TR(126);
    // ---- P7: z = h Wsq^T; P8: out = x1 + sigmoid(gate2) z
    uint32_t g2v[NC / 4];
#pragma unroll
    for (int j = 0; j < NC / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) g2v[2 * j + rb] = ldg_u32(G2 + (size_t)(tok0 + 8 * rb) * g2s + D + ccol + 8 * j);
    float acc[NC / 2];
    gemm_phase<NC, KPS, CL, NWS>(acc, NHK, ia, jw, su, su + C::O_W, B.afull, B.aempty, B.wfull, B.wempty, cw, wtid, 40);
    if (tid == 128) TR(127);
#pragma unroll
    for (int j = 0; j < NC / 8; ++j)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const int u = 2 * j + rb, ix = 4 * j + 2 * rb;
        const float2 g = bf2(g2v[u]);
        const float o0 = x1[ix] + sg(g.x) * acc[ix], o1 = x1[ix + 1] + sg(g.y) * acc[ix + 1];
        const size_t off = (size_t)(row0 + 8 * rb) * D + ccol + 8 * j;
        if constexpr (OBF) *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(OUT) + off) = pk(o0, o1);
        else *reinterpret_cast<float2*>(reinterpret_cast<float*>(OUT) + off) = make_float2(o0, o1);
      }
  }
  __syncwarp();
  cluster_sync();
}

// ------------------------------------------------------------------------------------------------ host
using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static CUtensorMap tmap(const at::Tensor& t, int box_rows) {                     // bf16 [rows, cols] (unit column stride), box [box_rows][64]
  static Encode enc = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "TMA encoder");
    return reinterpret_cast<Encode>(p);
  }();
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) % 8 == 0, "bo90: bf16 [rows, cols] operand");
  CUtensorMap m{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)}, str[1] = {(cuuint64_t)t.stride(0) * 2};
  const cuuint32_t box[2] = {64u, (cuuint32_t)box_rows}, e[2] = {1, 1};
  TORCH_CHECK(enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, str, box, e, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
              "bo90: TMA encode");
  return m;
}
template <class K, class... Args>
static void launch(K kern, int grid, int smem, int cl, bool pdl, Args... args) {
  if (cl > 8) TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess, "bo90: cluster 16");
  TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "bo90: shared memory");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = cl; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 2 : 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kern, args...);
  TORCH_CHECK(err == cudaSuccess, "bo90 launch: ", cudaGetErrorString(err));
}
static void check_tab(const at::Tensor& g, int width, int64_t T) {
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && g.dim() == 2 && g.stride(1) == 1 && g.size(0) == T && g.size(1) >= width &&
              g.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(g.data_ptr()) % 16) == 0, "bo90: table [T, >= ", width, "] bf16");
}

// x [M, 768] bf16 or fp32 contiguous; g1 [T, >= 4 768] bf16 view (s1 | sh1 | s2 | sh2); w [2 DA, 768]; vg [M, 2 DA]; xa [M, 768] scratch
void front(at::Tensor x, at::Tensor g1, at::Tensor w, at::Tensor vg, at::Tensor xa, int64_t cl, bool pdl, double eps) {
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), T = g1.size(0), DA_ = w.size(0) / 2;
  TORCH_CHECK(x.is_contiguous() && x.size(1) == D && M % QM == 0 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat), "front: x");
  TORCH_CHECK((DA_ == 768 || DA_ == 1024) && w.is_contiguous() && w.size(1) == D && vg.is_contiguous() && vg.size(0) == M && vg.size(1) == 2 * DA_ &&
              xa.is_contiguous() && xa.size(0) == M && xa.size(1) == D && T % QM == 0 && M % T == 0 && (cl == 8 || cl == 16), "front: shapes");
  check_tab(g1, 2 * D, T);
  const bool xbf = x.scalar_type() == at::kBFloat16;
  const int grid = (int)(M / QM) * (int)cl;
  auto go = [&](auto kern, int smem, int nv) {
    const auto mxa = tmap(xa, QM), mw = tmap(w, nv);
    launch(kern, grid, smem, (int)cl, pdl, mxa, mw, (const void*)x.data_ptr(), (const bf*)g1.data_ptr(), (int)g1.stride(0),
           (bf*)xa.data_ptr(), (bf*)vg.data_ptr(), (int)T, (float)eps);
  };
#define FRONT(CL_, DA__)                                                                                                        \
  if (cl == CL_ && DA_ == DA__) {                                                                                               \
    if (xbf) go(bo90_front<CL_, DA__, true>, FrontCfg<CL_, DA__>::SMEM, FrontCfg<CL_, DA__>::NV);                              \
    else go(bo90_front<CL_, DA__, false>, FrontCfg<CL_, DA__>::SMEM, FrontCfg<CL_, DA__>::NV);                                 \
    return;                                                                                                                     \
  }
  FRONT(8, 768) FRONT(16, 768) FRONT(8, 1024) FRONT(16, 1024)
#undef FRONT
}

// a [M, DA] bf16 (the core's output); x [M, 768] bf16 / fp32 (the block input); g1 [T, >= 4 768], g2 [T, >= 2 768] bf16 views;
// wo [768, DA], wab [3072, 768] = [Wa; Wb], wsq [768, 1536]; xt [M, 768], h [M, 1536] bf16 scratch; out [M, 768] fp32 or bf16
void tail(at::Tensor a, at::Tensor x, at::Tensor g1, at::Tensor g2, at::Tensor wo, at::Tensor wab, at::Tensor wsq, at::Tensor xt, at::Tensor h,
          at::Tensor out, int64_t cl, bool pdl, double eps) {
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), T = g1.size(0), DA_ = a.size(1);
  TORCH_CHECK(x.is_contiguous() && x.size(1) == D && M % QM == 0 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat), "tail: x");
  TORCH_CHECK((DA_ == 768 || DA_ == 1024) && a.size(0) == M && wo.is_contiguous() && wo.size(0) == D && wo.size(1) == DA_ && wab.is_contiguous() &&
              wab.size(0) == 3072 && wab.size(1) == D && wsq.is_contiguous() && wsq.size(0) == D && wsq.size(1) == 1536 && xt.is_contiguous() &&
              xt.size(0) == M && xt.size(1) == D && h.is_contiguous() && h.size(0) == M && h.size(1) == 1536 && out.is_contiguous() &&
              out.size(0) == M && out.size(1) == D && T % QM == 0 && M % T == 0 && (cl == 8 || cl == 16), "tail: shapes");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kFloat, "tail: out dtype");
  check_tab(g1, 4 * D, T);
  check_tab(g2, 2 * D, T);
  const bool xbf = x.scalar_type() == at::kBFloat16, obf = out.scalar_type() == at::kBFloat16;
  const int grid = (int)(M / QM) * (int)cl;
  auto go = [&](auto kern, int smem, int nc) {
    const auto ma = tmap(a, QM), mxt = tmap(xt, QM), mh = tmap(h, QM), mwo = tmap(wo, nc), mwab = tmap(wab, 96), mwsq = tmap(wsq, nc);
    launch(kern, grid, smem, (int)cl, pdl, ma, mxt, mh, mwo, mwab, mwsq, (const void*)x.data_ptr(), (const bf*)g1.data_ptr(), (int)g1.stride(0),
           (const bf*)g2.data_ptr(), (int)g2.stride(0), (bf*)xt.data_ptr(), (bf*)h.data_ptr(), out.data_ptr(), (int)T, (float)eps);
  };
#define TAIL(CL_, DA__)                                                                                                         \
  if (cl == CL_ && DA_ == DA__) {                                                                                               \
    constexpr int S_ = TailCfg<CL_, DA__>::SMEM, N_ = TailCfg<CL_, DA__>::NC;                                                   \
    if (xbf && obf) go(bo90_tail<CL_, DA__, true, true>, S_, N_);                                                               \
    else if (xbf) go(bo90_tail<CL_, DA__, true, false>, S_, N_);                                                                \
    else if (obf) go(bo90_tail<CL_, DA__, false, true>, S_, N_);                                                                \
    else go(bo90_tail<CL_, DA__, false, false>, S_, N_);                                                                        \
    return;                                                                                                                     \
  }
  TAIL(8, 768) TAIL(16, 768) TAIL(8, 1024) TAIL(16, 1024)
#undef TAIL
}

// probes: switch the %globaltimer trace on / off; read it back ([16, 128] int64, ns)
void set_mc(bool on) { int v = on ? 1 : 0; cudaMemcpyToSymbol(g_mc, &v, sizeof(int)); }
at::Tensor trace(bool on) {
  int v = on ? 1 : 0;
  cudaMemcpyToSymbol(g_tron, &v, sizeof(int));
  auto t = at::empty({16, 128}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  cudaMemcpyFromSymbol(t.data_ptr(), g_tr, sizeof(unsigned long long) * 16 * 128, 0, cudaMemcpyDeviceToDevice);
  return t;
}
// clusters of `cl` CTAs of the kernel that can be resident at once (cudaOccupancyMaxActiveClusters)
int64_t max_clusters(int64_t which, int64_t cl) {
  auto q = [&](auto kern, int smem) {
    if (cl > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)cl * 64);
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = (unsigned)cl; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    int n = 0;
    return cudaOccupancyMaxActiveClusters(&n, kern, &cfg) == cudaSuccess ? (int64_t)n : (int64_t)-1;
  };
  if (which == 0) return cl == 8 ? q(bo90_front<8, 768, true>, FrontCfg<8, 768>::SMEM) : q(bo90_front<16, 768, true>, FrontCfg<16, 768>::SMEM);
  return cl == 8 ? q(bo90_tail<8, 768, true, true>, TailCfg<8, 768>::SMEM) : q(bo90_tail<16, 768, true, true>, TailCfg<16, 768>::SMEM);
}
}  // namespace bo3

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("front", &bo3::front);
  m.def("tail", &bo3::tail);
  m.def("max_clusters", &bo3::max_clusters);
  m.def("trace", &bo3::trace);
  m.def("set_mc", &bo3::set_mc);
}
