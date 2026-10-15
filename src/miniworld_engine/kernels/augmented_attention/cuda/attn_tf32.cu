// attn_tf32.cu -- the token DiT attention core in fp32: TF32 wgmma (sm_90a), fp32 accumulation, TMA + mbarrier ring.
//
//   S[a] = q[a] k[a]^T + bias [+ key mask]      q pre-scaled into exp2 units (sm_scale log2 e), bias in log2 units
//   O[a] = softmax(S[a]) v[a]
//
// FORWARD: training writes O fp32 and the row LSE (log2 units); GATED (the fused token DiT step's inference core) writes
// sigmoid(g) o over q, the contract of the bf16 cores.
//
// TF32 wgmma takes both operands K-major only, which shapes three layouts:
//   * P V: B = V must be K-major, i.e. V^T [768, tokens] (keys contiguous). The inference step computes it with its own
//     cuBLAS GEMM (Wv xa^T); training passes one too.
//   * P as the A operand from registers: the TF32 A fragment holds k-indices (t, t + 4) of an 8-wide k-step, the S
//     accumulator holds columns (2t, 2t + 1). No shuffle is needed if S column 2t + i of every 8-key group IS key t + 4i:
//     the K tile is loaded with its rows permuted within each 8-row group -- smem row 2j + i holds key 4i + j -- by a
//     4-D TMA box (dims: head columns, i with a 4-row stride, j with a 1-row stride, 8-row groups). S columns are then keys
//     in that order, so the bias and the key mask must be too: the caller hands them PERMUTED the same way (the pair-bias
//     hoist writes them so), and every thread reads its (2t, 2t + 1) pair as one float2.
//   * the head dim 48 = 32 + 16: q and k tiles are two 128-B-swizzled chunks (columns 0..31, 32..63 of the head; the
//     chunk past the head is loaded but never read: k-steps 4, 5 use its first 64 bytes).
// One CTA = NWG consumer warpgroups x 64 query rows of one (sample, head) + one producer warp; key blocks of 64 through
// a STAGES ring; q stays in shared memory. Lazy running max (LAZY log2 units), floored so a block of -inf keys is finite.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef TOKEN_EMPTY_UNIFORM
#define TOKEN_EMPTY_UNIFORM 0
#endif

#ifndef DB_PER_SAMPLE
#define DB_PER_SAMPLE 0
#endif

#ifndef STAGES
#define STAGES 2
#endif
#ifndef NWG
#define NWG 2
#endif
#ifndef FWD_PIPE
#define FWD_PIPE 0   // 1: the forward as a software pipeline (attn_tf32_fwd_pipe_kernel). Measured 0.66x of the serial loop
                     // (167 registers -> one CTA per SM, half the warps): kept for reference, off
#endif
#ifndef PSTAGES
#define PSTAGES 4
#endif
#ifndef TRUNC_P
#define TRUNC_P 0    // 1: p goes to the TF32 MMA by clearing its low 13 bits (a LOP) instead of cvt.rna (XU pipe); l sums the same values
#endif
#ifndef QK_RS
#define QK_RS 0      // 1: q as register A operands of the QK wgmma (half the shared-memory reads of S = q k^T)
#endif
#ifndef BIAS_NOMEM
#define BIAS_NOMEM 0 // 1 (timing probe only, wrong results): the forward's bias seed is computed, not loaded
#endif
#ifndef NHP
#define NHP 16       // heads
#endif
#ifndef DHP
#define DHP 48       // head width: 32, 48 or 64
#endif
#ifndef INPLACE_P
#define INPLACE_P 0  // 1: p is written back into the score registers (no separate pa array): fewer live registers, room for BLKSM 3
#endif
#ifndef MC_BIAS
#define MC_BIAS 0    // 1: the cluster-multicast bias tile in shared memory (width 32 only fits two CTAs / SM); measured 0.99x at 24 x 32, off
#endif
#ifndef PREFETCH_BIAS
#define PREFETCH_BIAS 0  // 1: block n + 1's bias is loaded into registers while block n's QK wgmma / softmax run
#endif
#ifndef BLKSM
#define BLKSM 2
#endif
#ifndef LAZY
#define LAZY 8.0f
#endif

namespace {
constexpr int DH = DHP, NH = NHP, DM = NHP * DHP, BN = 64, QM = 64 * NWG;
static_assert(DH == 32 || DH == 48 || DH == 64, "head width 32 / 48 / 64");
constexpr int NCH = DH > 32 ? 2 : 1;                                // 128-B-wide (32-column) chunks of a head row
constexpr int JJ = DH / 8, ACC = DH / 2;                            // 8-column fragments of a head row; accumulator registers
constexpr int KCH = BN * 128, VCH = DH * 128;                       // bytes of one 128-B-wide chunk of the k / v^T tile
constexpr int ST_BYTES = NCH * KCH + 2 * VCH;                         // 28 KB a stage
constexpr int QCH = QM * 128, Q_BYTES = NCH * QCH;
static_assert(ST_BYTES % 1024 == 0 && QCH % 1024 == 0, "1-KB aligned regions (128-B swizzle atoms)");

TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }          // K-major, 128-B swizzle
TMN_DEVI float ex2(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
TMN_DEVI float trunc_tf32(float x) { return __uint_as_float(__float_as_uint(x) & 0xffffe000u); }
TMN_DEVI uint32_t tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
TMN_DEVI float sigmoid_t(float a) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a));
  return fmaf(0.5f, t, 0.5f);
}
TMN_DEVI float qmax(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
TMN_DEVI float qsum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
TMN_DEVI void tma_load_4d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

// the bias tile of a stage (the multicast form): two 128-B-swizzled boxes of 32 keys, QM rows each
constexpr int ST_MC = NCH * KCH + 2 * VCH + 2 * QM * 128;
TMN_DEVI void tma_load_mc2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
TMN_DEVI uint32_t mapa_u32(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
TMN_DEVI void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapa_u32(smem_u32(bar), rank)) : "memory");
}
TMN_DEVI void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }

// S (+)= q k^T, m64 n64 k8, both from shared memory
TMN_DEVI void mma_qk(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
// S (+)= q k^T, m64 n64 k8: A (q, TF32 bits) from registers, B (k) from shared memory
TMN_DEVI void mma_qk_rs(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
// O += p v, m64 n48 k8: A (p, TF32) from registers, B (v^T) from shared memory
TMN_DEVI void mma_pv(float (&d)[24], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n48k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
TMN_DEVI void mma_pv_n32(float (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16, %17, %18, %19}, %20, p, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
TMN_DEVI void mma_pv_n64(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32, %33, %34, %35}, %36, p, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
template <int N> TMN_DEVI void mma_pvd(float (&d)[N / 2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  if constexpr (N == 32) mma_pv_n32(d, a0, a1, a2, a3, b);
  else if constexpr (N == 48) mma_pv(d, a0, a1, a2, a3, b);
  else mma_pv_n64(d, a0, a1, a2, a3, b);
}

// grid: A * 16 * (L / QM) CTAs, the sample fastest (the A CTAs sharing a bias tile run together and read it from L2).
// mq / mk: q / k as [rows, >= 768 columns] (row stride per map); mk 4-D (the permuted rows); mvt: v^T [768, A L].
// Bias [16, L, L] fp32, key-permuted, log2 units; KM [A, L] additive, key-permuted (HASM).
// GATED: OUT = q's first element, row stride so, g at OUT + goff; else O [A L, 768] fp32 and LSE [A, 16, L].
template <bool GATED, bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, BLKSM)
attn_tf32_fwd_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                     const __grid_constant__ CUtensorMap mvt, const float* __restrict__ Bias, const float* __restrict__ KM,
                     float* __restrict__ O, float* __restrict__ LSE, float* __restrict__ OUT, long so, long goff, int L, int A) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sq = sm;
  uint8_t* sst = sm + Q_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(sst + STAGES * ST_BYTES);
  uint64_t* empty = full + STAGES;
  uint64_t* qbar = empty + STAGES;

  const int tid = threadIdx.x;
  const int mt = L / QM;
  const int samp = blockIdx.x % A, cid = blockIdx.x / A;
  const int m_tile = cid % mt, head = cid / mt;
  const int m0 = m_tile * QM, qcol = head * DH, srow = samp * L;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 2); mbar_init(&empty[s], 4 * NWG); }
    mbar_init(qbar, 1);
    fence_barrier_init();
  }
  __syncthreads();

  if (tid >= 128 * NWG) {                                           // producer warp: lane 0 k, lane 1 v^T
    const int iss = tid - 128 * NWG;
    if (iss == 0) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mvt);
      mbar_arrive_expect_tx(qbar, Q_BYTES);
      tma_load_2d(sq, &mq, qbar, qcol, srow + m0);
      if (NCH == 2) tma_load_2d(sq + QCH, &mq, qbar, qcol + 32, srow + m0);
    }
    if (iss < 2) {
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sst + s * ST_BYTES;
        const int k0 = srow + n * BN;
        if (iss == 0) {
          mbar_arrive_expect_tx(&full[s], NCH * KCH);
          tma_load_4d(slot, &mk, &full[s], qcol, 0, 0, k0 / 8);
          if (NCH == 2) tma_load_4d(slot + KCH, &mk, &full[s], qcol + 32, 0, 0, k0 / 8);
        } else {
          mbar_arrive_expect_tx(&full[s], 2 * VCH);
          tma_load_2d(slot + NCH * KCH, &mvt, &full[s], k0, qcol);
          tma_load_2d(slot + NCH * KCH + VCH, &mvt, &full[s], k0 + 32, qcol);
        }
      }
    }
    return;
  }

  // ---- consumer warpgroups: 64 query rows each
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int g = lane >> 2, t = lane & 3;
  const int row = m0 + wg * 64 + warp * 16 + g;                     // this lane's rows: row, row + 8 (within the sample)
  const float* brow0 = Bias + ((size_t)head * L + row) * L + 2 * t;
  const float* brow1 = brow0 + 8 * (size_t)L;
  const float* kmr = HASM ? KM + (size_t)samp * L + 2 * t : nullptr;
  float acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = 0.f;
  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};
  if (OUT != nullptr && t == 0) {                                   // the epilogue's g rows (192 B each): into L2 now
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float* gp = OUT + ((size_t)srow + row + 8 * h) * so + goff + qcol;
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + 24));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + DH - 1));
    }
  }
  const uint32_t qa = smem_u32(sq) + wg * 64 * 128;
  const uint64_t dQ0 = dsw(qa), dQ1 = dsw(qa + QCH);
  const uint32_t sbase = smem_u32(sst);
  mbar_wait(qbar, 0);
#if QK_RS
  uint32_t qf[JJ][4];                                                // q fragments of this lane, loaded once
  {
    const uint32_t r0 = wg * 64 + warp * 16 + g;
#pragma unroll
    for (int ks = 0; ks < JJ; ++ks) {
      const int c0 = ks * 8 + t, c1 = c0 + 4;
      uint32_t v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t r = r0 + (e & 1) * 8;
        const int c = (e >> 1) ? c1 : c0;
        const uint32_t addr = smem_u32(sq) + (c >> 5) * QCH + r * 128 + ((((c & 31) >> 2) ^ (r & 7)) << 4) + ((c & 3) << 2);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v[e]) : "r"(addr));
      }
      qf[ks][0] = v[0]; qf[ks][1] = v[1]; qf[ks][2] = v[2]; qf[ks][3] = v[3];
    }
  }
#endif

  float sc[32];
#if PREFETCH_BIAS
  float2 bpf[16];                                                    // block n + 1's bias, loaded under block n's MMAs / softmax
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    bpf[2 * j] = *reinterpret_cast<const float2*>(brow0 + 8 * j);
    bpf[2 * j + 1] = *reinterpret_cast<const float2*>(brow1 + 8 * j);
  }
#endif
  for (int n = 0; n < nblocks; ++n) {
    const int sn = n % STAGES;
    // seed with bias (+ key mask), loaded before the wait so the L2 latency hides under it
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#if PREFETCH_BIAS
      const float2 b0 = bpf[2 * j], b1 = bpf[2 * j + 1];
#elif BIAS_NOMEM
      const float2 b0 = make_float2(1e-3f * (n + j), 2e-3f * (j + t + wg)), b1 = make_float2(3e-3f * (n + t), 1e-3f * (j + g));
#else
      const float2 b0 = *reinterpret_cast<const float2*>(brow0 + n * BN + 8 * j);
      const float2 b1 = *reinterpret_cast<const float2*>(brow1 + n * BN + 8 * j);
#endif
      float2 km = make_float2(0.f, 0.f);
      if (HASM) km = *reinterpret_cast<const float2*>(kmr + n * BN + 8 * j);
      sc[4 * j] = b0.x + km.x; sc[4 * j + 1] = b0.y + km.y; sc[4 * j + 2] = b1.x + km.x; sc[4 * j + 3] = b1.y + km.y;
    }
    mbar_wait(&full[sn], (n / STAGES) & 1);
    const uint32_t slot = sbase + sn * ST_BYTES;
    const uint64_t dK0 = dsw(slot), dK1 = dsw(slot + KCH);
    wgmma_fence();
#if QK_RS
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_qk_rs(sc, qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], dK0 + ks * 2);
#pragma unroll
    for (int ks = 0; ks < JJ - 4; ++ks) mma_qk_rs(sc, qf[4 + ks][0], qf[4 + ks][1], qf[4 + ks][2], qf[4 + ks][3], dK1 + ks * 2);
#else
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_qk(sc, dQ0 + ks * 2, dK0 + ks * 2);            // head columns 0..31
#pragma unroll
    for (int ks = 0; ks < JJ - 4; ++ks) mma_qk(sc, dQ1 + ks * 2, dK1 + ks * 2);       // 32..DH - 1
#endif
    wgmma_commit();
#if PREFETCH_BIAS
    if (n + 1 < nblocks) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        bpf[2 * j] = *reinterpret_cast<const float2*>(brow0 + (n + 1) * BN + 8 * j);
        bpf[2 * j + 1] = *reinterpret_cast<const float2*>(brow1 + (n + 1) * BN + 8 * j);
      }
    }
#endif
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sc[i]);
    float m_new[2];
    bool moved = false;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
      mx = fmaxf(qmax(mx), -1e30f);
      m_new[h] = mx > m_i[h] + LAZY ? mx : m_i[h];
      moved |= m_new[h] != m_i[h];
    }
    if (__any_sync(0xffffffffu, moved)) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float alpha = ex2(m_i[h] - m_new[h]);
        l_i[h] *= alpha;
#pragma unroll
        for (int j = 0; j < JJ; ++j) { acc[4 * j + 2 * h] *= alpha; acc[4 * j + 2 * h + 1] *= alpha; }
        m_i[h] = m_new[h];
      }
    }
#if INPLACE_P
#define PA(i) (__float_as_uint(sc[i]))
#else
    uint32_t pa[32];
#define PA(i) (pa[i])
#endif
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#if TRUNC_P
        const float p0 = trunc_tf32(ex2(sc[4 * j + 2 * h] - m_i[h])), p1 = trunc_tf32(ex2(sc[4 * j + 2 * h + 1] - m_i[h]));
        l_i[h] += p0 + p1;
#if INPLACE_P
        sc[4 * j + 2 * h] = p0; sc[4 * j + 2 * h + 1] = p1;
#else
        pa[4 * j + 2 * h] = __float_as_uint(p0); pa[4 * j + 2 * h + 1] = __float_as_uint(p1);
#endif
#else
        const float p0 = ex2(sc[4 * j + 2 * h] - m_i[h]), p1 = ex2(sc[4 * j + 2 * h + 1] - m_i[h]);
        l_i[h] += p0 + p1;
#if INPLACE_P
        sc[4 * j + 2 * h] = __uint_as_float(tf32(p0)); sc[4 * j + 2 * h + 1] = __uint_as_float(tf32(p1));
#else
        pa[4 * j + 2 * h] = tf32(p0); pa[4 * j + 2 * h + 1] = tf32(p1);
#endif
#endif
      }
    }
    const uint64_t dV0 = dsw(slot + NCH * KCH), dV1 = dsw(slot + NCH * KCH + VCH);
    wgmma_fence();
#pragma unroll
    for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);
#pragma unroll
    for (int j = 0; j < 8; ++j)                                     // A (row g, k t) = S (g, 2t), (g, k t + 4) = S (g, 2t + 1)
      mma_pvd<DH>(acc, PA(4 * j), PA(4 * j + 2), PA(4 * j + 1), PA(4 * j + 3), (j < 4 ? dV0 : dV1) + (j % 4) * 2);
    wgmma_commit();
    wgmma_wait<0>();
    if ((tid & 31) == 0 && n + STAGES < nblocks) mbar_arrive(&empty[sn]);
  }
#pragma unroll
  for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);

  // ---- epilogue: rows row (h = 0), row + 8 (h = 1); columns 8j + 2t, + 1
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float l = qsum(l_i[h]);
    const float inv = 1.f / l;
    const size_t r = (size_t)srow + row + 8 * h;
    if constexpr (GATED) {
      float* orow = OUT + r * so + qcol;
#pragma unroll
      for (int j = 0; j < JJ; ++j) {
        const float2 gg = *reinterpret_cast<const float2*>(orow + goff + 8 * j + 2 * t);
        *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) =
            make_float2(acc[4 * j + 2 * h] * inv * sigmoid_t(gg.x), acc[4 * j + 2 * h + 1] * inv * sigmoid_t(gg.y));
      }
    } else {                                                        // O, or with OUT (the q|k|v|g rows) og = sigmoid(g) O
      float* orow = O + r * DM + qcol;
      const float* grow = OUT != nullptr ? OUT + r * so + goff + qcol : nullptr;
#pragma unroll
      for (int j = 0; j < JJ; ++j) {
        float2 o = make_float2(acc[4 * j + 2 * h] * inv, acc[4 * j + 2 * h + 1] * inv);
        if (grow != nullptr) {
          const float2 gg = *reinterpret_cast<const float2*>(grow + 8 * j + 2 * t);
          o.x *= 1.f / (1.f + __expf(-gg.x)); o.y *= 1.f / (1.f + __expf(-gg.y));
        }
        *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) = o;
      }
      if (t == 0) LSE[((size_t)samp * NH + head) * L + row + 8 * h] = m_i[h] + __log2f(l);
    }
  }
}

template <bool GATED, bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, BLKSM)
attn_tf32_fwd_mc_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                     const __grid_constant__ CUtensorMap mvt, const __grid_constant__ CUtensorMap mb, int cs, const float* __restrict__ KM,
                     float* __restrict__ O, float* __restrict__ LSE, float* __restrict__ OUT, long so, long goff, int L, int A) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sq = sm;
  uint8_t* sst = sm + Q_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(sst + STAGES * ST_MC);
  uint64_t* empty = full + STAGES;
  uint64_t* qbar = empty + STAGES;

  const int tid = threadIdx.x;
  const int mt = L / QM;
  const int samp = blockIdx.x % A, cid = blockIdx.x / A;
  const int m_tile = cid % mt, head = cid / mt;
  const int m0 = m_tile * QM, qcol = head * DH, srow = samp * L;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 3); mbar_init(&empty[s], cs * 4 * NWG); }
    mbar_init(qbar, 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (cs > 1) cluster_sync_all();                                   // every barrier of the cluster initialised
  const uint32_t rank = cs > 1 ? cluster_rank() : 0;

  if (tid >= 128 * NWG) {                                           // producer warp: lane 0 k, lane 1 v^T, lane 2 the bias tile
    const int iss = tid - 128 * NWG;
    if (iss == 0) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mvt);
      mbar_arrive_expect_tx(qbar, Q_BYTES);
      tma_load_2d(sq, &mq, qbar, qcol, srow + m0);
      if (NCH == 2) tma_load_2d(sq + QCH, &mq, qbar, qcol + 32, srow + m0);
    }
    if (iss < 3) {
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sst + s * ST_MC;
        const int k0 = srow + n * BN;
        if (iss == 0) {
          mbar_arrive_expect_tx(&full[s], NCH * KCH);
          tma_load_4d(slot, &mk, &full[s], qcol, 0, 0, k0 / 8);
          if (NCH == 2) tma_load_4d(slot + KCH, &mk, &full[s], qcol + 32, 0, 0, k0 / 8);
        } else if (iss == 1) {
          mbar_arrive_expect_tx(&full[s], 2 * VCH);
          tma_load_2d(slot + NCH * KCH, &mvt, &full[s], k0, qcol);
          tma_load_2d(slot + NCH * KCH + VCH, &mvt, &full[s], k0 + 32, qcol);
        } else {                                                    // the bias tile [QM rows][64 keys]: one rank issues, rotating
          mbar_arrive_expect_tx(&full[s], 2 * QM * 128);
          if (cs == 1 || rank == (uint32_t)(n % cs)) {
            uint8_t* bt = slot + NCH * KCH + 2 * VCH;
            const uint16_t msk = (uint16_t)((1u << cs) - 1);
            tma_load_mc2d(bt, &mb, &full[s], n * BN, head * L + m0, msk);
            tma_load_mc2d(bt + QM * 128, &mb, &full[s], n * BN + 32, head * L + m0, msk);
          }
        }
      }
    }
    __syncwarp();
    if (cs > 1) cluster_sync_all();                                 // no remote arrive may land on an exited CTA
    return;
  }

  // ---- consumer warpgroups: 64 query rows each
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int g = lane >> 2, t = lane & 3;
  const int row = m0 + wg * 64 + warp * 16 + g;                     // this lane's rows: row, row + 8 (within the sample)
  const int rl = wg * 64 + warp * 16 + g;                           // this lane's row inside the CTA's bias tile (+ 8)
  const float* kmr = HASM ? KM + (size_t)samp * L + 2 * t : nullptr;
  float acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = 0.f;
  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};
  if (OUT != nullptr && t == 0) {                                   // the epilogue's g rows (192 B each): into L2 now
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float* gp = OUT + ((size_t)srow + row + 8 * h) * so + goff + qcol;
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + 24));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + DH - 1));
    }
  }
  const uint32_t qa = smem_u32(sq) + wg * 64 * 128;
  const uint64_t dQ0 = dsw(qa), dQ1 = dsw(qa + QCH);
  const uint32_t sbase = smem_u32(sst);
  mbar_wait(qbar, 0);
#if QK_RS
  uint32_t qf[JJ][4];                                                // q fragments of this lane, loaded once
  {
    const uint32_t r0 = wg * 64 + warp * 16 + g;
#pragma unroll
    for (int ks = 0; ks < JJ; ++ks) {
      const int c0 = ks * 8 + t, c1 = c0 + 4;
      uint32_t v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t r = r0 + (e & 1) * 8;
        const int c = (e >> 1) ? c1 : c0;
        const uint32_t addr = smem_u32(sq) + (c >> 5) * QCH + r * 128 + ((((c & 31) >> 2) ^ (r & 7)) << 4) + ((c & 3) << 2);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v[e]) : "r"(addr));
      }
      qf[ks][0] = v[0]; qf[ks][1] = v[1]; qf[ks][2] = v[2]; qf[ks][3] = v[3];
    }
  }
#endif

  float sc[32];
  for (int n = 0; n < nblocks; ++n) {
    const int sn = n % STAGES;
    mbar_wait(&full[sn], (n / STAGES) & 1);
    const uint32_t slot = sbase + sn * ST_MC;
    {                                                               // seed with the bias tile from shared memory (swizzled 128 B rows)
      const uint32_t bt = slot + NCH * KCH + 2 * VCH;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t cb = bt + (j >> 2) * (QM * 128) + (((2 * (j & 3) + (t >> 1)) ^ (rl & 7)) << 4) + (t & 1) * 8;
        float2 b0, b1;
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(b0.x), "=f"(b0.y) : "r"(cb + rl * 128));
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(b1.x), "=f"(b1.y) : "r"(cb + (rl + 8) * 128));
        float2 km = make_float2(0.f, 0.f);
        if (HASM) km = *reinterpret_cast<const float2*>(kmr + n * BN + 8 * j);
        sc[4 * j] = b0.x + km.x; sc[4 * j + 1] = b0.y + km.y; sc[4 * j + 2] = b1.x + km.x; sc[4 * j + 3] = b1.y + km.y;
      }
    }
    const uint64_t dK0 = dsw(slot), dK1 = dsw(slot + KCH);
    wgmma_fence();
#if QK_RS
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_qk_rs(sc, qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], dK0 + ks * 2);
#pragma unroll
    for (int ks = 0; ks < JJ - 4; ++ks) mma_qk_rs(sc, qf[4 + ks][0], qf[4 + ks][1], qf[4 + ks][2], qf[4 + ks][3], dK1 + ks * 2);
#else
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_qk(sc, dQ0 + ks * 2, dK0 + ks * 2);            // head columns 0..31
#pragma unroll
    for (int ks = 0; ks < JJ - 4; ++ks) mma_qk(sc, dQ1 + ks * 2, dK1 + ks * 2);       // 32..DH - 1
#endif
    wgmma_commit();
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sc[i]);
    float m_new[2];
    bool moved = false;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
      mx = fmaxf(qmax(mx), -1e30f);
      m_new[h] = mx > m_i[h] + LAZY ? mx : m_i[h];
      moved |= m_new[h] != m_i[h];
    }
    if (__any_sync(0xffffffffu, moved)) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float alpha = ex2(m_i[h] - m_new[h]);
        l_i[h] *= alpha;
#pragma unroll
        for (int j = 0; j < JJ; ++j) { acc[4 * j + 2 * h] *= alpha; acc[4 * j + 2 * h + 1] *= alpha; }
        m_i[h] = m_new[h];
      }
    }
#if INPLACE_P
#define PA(i) (__float_as_uint(sc[i]))
#else
    uint32_t pa[32];
#define PA(i) (pa[i])
#endif
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#if TRUNC_P
        const float p0 = trunc_tf32(ex2(sc[4 * j + 2 * h] - m_i[h])), p1 = trunc_tf32(ex2(sc[4 * j + 2 * h + 1] - m_i[h]));
        l_i[h] += p0 + p1;
#if INPLACE_P
        sc[4 * j + 2 * h] = p0; sc[4 * j + 2 * h + 1] = p1;
#else
        pa[4 * j + 2 * h] = __float_as_uint(p0); pa[4 * j + 2 * h + 1] = __float_as_uint(p1);
#endif
#else
        const float p0 = ex2(sc[4 * j + 2 * h] - m_i[h]), p1 = ex2(sc[4 * j + 2 * h + 1] - m_i[h]);
        l_i[h] += p0 + p1;
#if INPLACE_P
        sc[4 * j + 2 * h] = __uint_as_float(tf32(p0)); sc[4 * j + 2 * h + 1] = __uint_as_float(tf32(p1));
#else
        pa[4 * j + 2 * h] = tf32(p0); pa[4 * j + 2 * h + 1] = tf32(p1);
#endif
#endif
      }
    }
    const uint64_t dV0 = dsw(slot + NCH * KCH), dV1 = dsw(slot + NCH * KCH + VCH);
    wgmma_fence();
#pragma unroll
    for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);
#pragma unroll
    for (int j = 0; j < 8; ++j)                                     // A (row g, k t) = S (g, 2t), (g, k t + 4) = S (g, 2t + 1)
      mma_pvd<DH>(acc, PA(4 * j), PA(4 * j + 2), PA(4 * j + 1), PA(4 * j + 3), (j < 4 ? dV0 : dV1) + (j % 4) * 2);
    wgmma_commit();
    wgmma_wait<0>();
    if ((tid & 31) == 0 && n + STAGES < nblocks) {
      if (cs == 1) mbar_arrive(&empty[sn]);
      else for (int r = 0; r < cs; ++r) mbar_arrive_remote(&empty[sn], r);   // free in every rank's view
    }
  }
#pragma unroll
  for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);

  // ---- epilogue: rows row (h = 0), row + 8 (h = 1); columns 8j + 2t, + 1
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float l = qsum(l_i[h]);
    const float inv = 1.f / l;
    const size_t r = (size_t)srow + row + 8 * h;
    if constexpr (GATED) {
      float* orow = OUT + r * so + qcol;
#pragma unroll
      for (int j = 0; j < JJ; ++j) {
        const float2 gg = *reinterpret_cast<const float2*>(orow + goff + 8 * j + 2 * t);
        *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) =
            make_float2(acc[4 * j + 2 * h] * inv * sigmoid_t(gg.x), acc[4 * j + 2 * h + 1] * inv * sigmoid_t(gg.y));
      }
    } else {                                                        // O, or with OUT (the q|k|v|g rows) og = sigmoid(g) O
      float* orow = O + r * DM + qcol;
      const float* grow = OUT != nullptr ? OUT + r * so + goff + qcol : nullptr;
#pragma unroll
      for (int j = 0; j < JJ; ++j) {
        float2 o = make_float2(acc[4 * j + 2 * h] * inv, acc[4 * j + 2 * h + 1] * inv);
        if (grow != nullptr) {
          const float2 gg = *reinterpret_cast<const float2*>(grow + 8 * j + 2 * t);
          o.x *= 1.f / (1.f + __expf(-gg.x)); o.y *= 1.f / (1.f + __expf(-gg.y));
        }
        *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) = o;
      }
      if (t == 0) LSE[((size_t)samp * NH + head) * L + row + 8 * h] = m_i[h] + __log2f(l);
    }
  }
}

template <bool GATED, bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, 1)
attn_tf32_fwd_pipe_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                     const __grid_constant__ CUtensorMap mvt, const float* __restrict__ Bias, const float* __restrict__ KM,
                     float* __restrict__ O, float* __restrict__ LSE, float* __restrict__ OUT, long so, long goff, int L, int A) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sq = sm;
  uint8_t* sst = sm + Q_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(sst + PSTAGES * ST_BYTES);
  uint64_t* empty = full + PSTAGES;
  uint64_t* qbar = empty + PSTAGES;

  const int tid = threadIdx.x;
  const int mt = L / QM;
  const int samp = blockIdx.x % A, cid = blockIdx.x / A;
  const int m_tile = cid % mt, head = cid / mt;
  const int m0 = m_tile * QM, qcol = head * DH, srow = samp * L;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < PSTAGES; ++s) { mbar_init(&full[s], 2); mbar_init(&empty[s], 4 * NWG); }
    mbar_init(qbar, 1);
    fence_barrier_init();
  }
  __syncthreads();

  if (tid >= 128 * NWG) {                                           // producer warp: lane 0 k, lane 1 v^T
    const int iss = tid - 128 * NWG;
    if (iss == 0) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mvt);
      mbar_arrive_expect_tx(qbar, Q_BYTES);
      tma_load_2d(sq, &mq, qbar, qcol, srow + m0);
      if (NCH == 2) tma_load_2d(sq + QCH, &mq, qbar, qcol + 32, srow + m0);
    }
    if (iss < 2) {
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % PSTAGES;
        mbar_wait(&empty[s], ((n / PSTAGES) & 1) ^ 1);
        uint8_t* slot = sst + s * ST_BYTES;
        const int k0 = srow + n * BN;
        if (iss == 0) {
          mbar_arrive_expect_tx(&full[s], NCH * KCH);
          tma_load_4d(slot, &mk, &full[s], qcol, 0, 0, k0 / 8);
          if (NCH == 2) tma_load_4d(slot + KCH, &mk, &full[s], qcol + 32, 0, 0, k0 / 8);
        } else {
          mbar_arrive_expect_tx(&full[s], 2 * VCH);
          tma_load_2d(slot + NCH * KCH, &mvt, &full[s], k0, qcol);
          tma_load_2d(slot + NCH * KCH + VCH, &mvt, &full[s], k0 + 32, qcol);
        }
      }
    }
    return;
  }

  // ---- consumer warpgroups: 64 query rows each
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int g = lane >> 2, t = lane & 3;
  const int row = m0 + wg * 64 + warp * 16 + g;                     // this lane's rows: row, row + 8 (within the sample)
  const float* brow0 = Bias + ((size_t)head * L + row) * L + 2 * t;
  const float* brow1 = brow0 + 8 * (size_t)L;
  const float* kmr = HASM ? KM + (size_t)samp * L + 2 * t : nullptr;
  float acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = 0.f;
  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};
  if (OUT != nullptr && t == 0) {                                   // the epilogue's g rows (192 B each): into L2 now
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float* gp = OUT + ((size_t)srow + row + 8 * h) * so + goff + qcol;
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + 24));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + DH - 1));
    }
  }
  const uint32_t qa = smem_u32(sq) + wg * 64 * 128;
  const uint64_t dQ0 = dsw(qa), dQ1 = dsw(qa + QCH);
  const uint32_t sbase = smem_u32(sst);
  mbar_wait(qbar, 0);

  // ---- software pipeline (FA3's intra-warpgroup overlap): QK(n + 1) runs on the tensor cores while softmax(n) runs on the SM.
  // Group commit order: QK(n), PV(n - 1), QK(n + 1); P(n) lives in its own registers, S(n) / S(n + 1) alternate (scA / scB).
  float scA[32], scB[32];
  uint32_t pa[32];
  auto seed = [&](float (&sc)[32], int n) {           // the bias (+ key mask) loaded before the wait: the L2 latency hides under it
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 b0 = *reinterpret_cast<const float2*>(brow0 + n * BN + 8 * j);
      const float2 b1 = *reinterpret_cast<const float2*>(brow1 + n * BN + 8 * j);
      float2 km = make_float2(0.f, 0.f);
      if (HASM) km = *reinterpret_cast<const float2*>(kmr + n * BN + 8 * j);
      sc[4 * j] = b0.x + km.x; sc[4 * j + 1] = b0.y + km.y; sc[4 * j + 2] = b1.x + km.x; sc[4 * j + 3] = b1.y + km.y;
    }
  };
  auto qk = [&](float (&sc)[32], int n) {
    const int sn = n % PSTAGES;
    mbar_wait(&full[sn], (n / PSTAGES) & 1);
    const uint32_t slot = sbase + sn * ST_BYTES;
    const uint64_t dK0 = dsw(slot), dK1 = dsw(slot + KCH);
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_qk(sc, dQ0 + ks * 2, dK0 + ks * 2);            // head columns 0..31
#pragma unroll
    for (int ks = 0; ks < JJ - 4; ++ks) mma_qk(sc, dQ1 + ks * 2, dK1 + ks * 2);       // 32..DH - 1
    wgmma_commit();
  };
  // S(n) in sc is complete. has_next: QK(n + 1) is the one group allowed to stay pending past PV(n - 1)'s wait.
  auto softmax_pv = [&](float (&sc)[32], int n, bool has_next) {
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sc[i]);
    float m_new[2];
    bool moved = false;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
      mx = fmaxf(qmax(mx), -1e30f);
      m_new[h] = mx > m_i[h] + LAZY ? mx : m_i[h];
      moved |= m_new[h] != m_i[h];
    }
    if (has_next) wgmma_wait<1>(); else wgmma_wait<0>();               // PV(n - 1) done: acc and pa are free again
#pragma unroll
    for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);
    if (__any_sync(0xffffffffu, moved)) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float alpha = ex2(m_i[h] - m_new[h]);
        l_i[h] *= alpha;
#pragma unroll
        for (int j = 0; j < JJ; ++j) { acc[4 * j + 2 * h] *= alpha; acc[4 * j + 2 * h + 1] *= alpha; }
        m_i[h] = m_new[h];
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float p0 = ex2(sc[4 * j + 2 * h] - m_i[h]), p1 = ex2(sc[4 * j + 2 * h + 1] - m_i[h]);
        l_i[h] += p0 + p1;
        pa[4 * j + 2 * h] = tf32(p0); pa[4 * j + 2 * h + 1] = tf32(p1);
      }
    }
    if ((tid & 31) == 0 && n > 0 && (n - 1) + PSTAGES < nblocks) mbar_arrive(&empty[(n - 1) % PSTAGES]);  // block n - 1 fully read
    const int sn = n % PSTAGES;
    const uint32_t slot = sbase + sn * ST_BYTES;
    const uint64_t dV0 = dsw(slot + NCH * KCH), dV1 = dsw(slot + NCH * KCH + VCH);
    wgmma_fence();
#pragma unroll
    for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);
#pragma unroll
    for (int j = 0; j < 8; ++j)                                     // A (row g, k t) = S (g, 2t), (g, k t + 4) = S (g, 2t + 1)
      mma_pvd<DH>(acc, pa[4 * j], pa[4 * j + 2], pa[4 * j + 1], pa[4 * j + 3], (j < 4 ? dV0 : dV1) + (j % 4) * 2);
    wgmma_commit();
  };
  seed(scA, 0);
  qk(scA, 0);
  for (int n = 0; n < nblocks; n += 2) {                            // nblocks is even (L is a multiple of 128): n in scA, n + 1 in scB
    seed(scB, n + 1);
    qk(scB, n + 1);
    if (n == 0) wgmma_wait<1>(); else wgmma_wait<2>();             // QK(n) complete
    softmax_pv(scA, n, true);
    const bool more = n + 2 < nblocks;
    if (more) { seed(scA, n + 2); qk(scA, n + 2); }
    if (more) wgmma_wait<2>(); else wgmma_wait<1>();               // QK(n + 1) complete
    softmax_pv(scB, n + 1, more);
  }
  wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < ACC; ++i) fence_reg(acc[i]);

  // ---- epilogue: rows row (h = 0), row + 8 (h = 1); columns 8j + 2t, + 1
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float l = qsum(l_i[h]);
    const float inv = 1.f / l;
    const size_t r = (size_t)srow + row + 8 * h;
    if constexpr (GATED) {
      float* orow = OUT + r * so + qcol;
#pragma unroll
      for (int j = 0; j < JJ; ++j) {
        const float2 gg = *reinterpret_cast<const float2*>(orow + goff + 8 * j + 2 * t);
        *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) =
            make_float2(acc[4 * j + 2 * h] * inv * sigmoid_t(gg.x), acc[4 * j + 2 * h + 1] * inv * sigmoid_t(gg.y));
      }
    } else {                                                        // O, or with OUT (the q|k|v|g rows) og = sigmoid(g) O
      float* orow = O + r * DM + qcol;
      const float* grow = OUT != nullptr ? OUT + r * so + goff + qcol : nullptr;
#pragma unroll
      for (int j = 0; j < JJ; ++j) {
        float2 o = make_float2(acc[4 * j + 2 * h] * inv, acc[4 * j + 2 * h + 1] * inv);
        if (grow != nullptr) {
          const float2 gg = *reinterpret_cast<const float2*>(grow + 8 * j + 2 * t);
          o.x *= 1.f / (1.f + __expf(-gg.x)); o.y *= 1.f / (1.f + __expf(-gg.y));
        }
        *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) = o;
      }
      if (t == 0) LSE[((size_t)samp * NH + head) * L + row + 8 * h] = m_i[h] + __log2f(l);
    }
  }
}
// ======================================================================================================== backward
// Natural logits s = S ln 2 (S: the forward's log2-unit logits), P = 2^(S - LSE), dP = dO v^T, dS = P (dP - D) = dL/ds.
//   dq = dS k / sqrt 48 (the unscaled q)     dk = dS^T qs / log2 e (qs = q log2 e / sqrt 48)     dv = P^T dO     dbias = sum_a dS
// Both kernels recompute S and dP; every product keeps the TF32 layout rules of the forward (the header): the B operand of a
// product over queries or keys is a TRANSPOSED copy (q^T, dO^T, k^T: [768, A L], contiguous in the tokens), and the operand
// whose rows index the accumulator columns that become an A fragment is loaded key- / query-permuted.

constexpr float LOG2E = 1.4426950408889634f, RSQD = DH == 32 ? 0.17677669529663687f : DH == 48 ? 0.14433756729740643f : 0.125f;   // 1 / sqrt DH
constexpr int TCH = 64 * 128;                                       // a 64-row, 128-B-wide chunk
constexpr int BK_ST = 2 * NCH * TCH + 4 * VCH;        // dkv stage: q_p, dO_p, q^T, dO^T -- 56 KB
constexpr int BQ_ST = 2 * NCH * TCH + 2 * VCH;                  // dqb stage: k_p, v_p, k^T -- 44 KB

TMN_DEVI void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
TMN_DEVI void bulk_reduce_add_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void token_store_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
TMN_DEVI void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
TMN_DEVI void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
// d (+)= A B, m64 n64 k8 from shared memory (S, dP and their transposes); first: overwrite
TMN_DEVI void mma_64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate));
}
// the 6 k-steps over the head dim: A, B two chunks each (columns 0..31, 32..47)
TMN_DEVI void mma_head(float (&d)[32], uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1, bool acc) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) mma_64(d, a0 + ks * 2, b0 + ks * 2, acc || ks);
#pragma unroll
  for (int ks = 0; ks < JJ - 4; ++ks) mma_64(d, a1 + ks * 2, b1 + ks * 2, 1);
}
// acc += A B over 64 tokens (8 k-steps): A the accumulator-shaped fragment in pa (TF32), B a 48-row transposed tile in two
// 32-token chunks
TMN_DEVI void mma_tok(float (&acc)[ACC], const uint32_t (&pa)[32], uint64_t b0, uint64_t b1) {
#pragma unroll
  for (int j = 0; j < 8; ++j) mma_pvd<DH>(acc, pa[4 * j], pa[4 * j + 2], pa[4 * j + 1], pa[4 * j + 3], (j < 4 ? b0 : b1) + (j % 4) * 2);
}

// dK, dV. grid: A * 16 * (L / QM) CTAs, the sample fastest; consumer warpgroup w owns 64 keys, a stage is 64 queries.
// mk, mv: k, v [A L, 768] (natural rows); mqp, mdop: q (exp2 units), dO, query-permuted 4-D maps; mqt, mdot: q^T, dO^T.
// BT [16, L(key), L(query position)]: the bias transposed, its query columns permuted; LSEP, DDP [A, 16, L] likewise permuted;
// KM [A, L] additive key mask (natural). DK, DV [A L, 768] fp32.
template <bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, 1)
attn_tf32_dkv_kernel(const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                     const __grid_constant__ CUtensorMap mqp, const __grid_constant__ CUtensorMap mdop,
                     const __grid_constant__ CUtensorMap mqt, const __grid_constant__ CUtensorMap mdot,
                     const float* __restrict__ BT, const float* __restrict__ LSEP, const float* __restrict__ DDP,
                     const float* __restrict__ KM, float* __restrict__ DK, float* __restrict__ DV, int L, int A) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sk = sm;                                                 // k: 2 chunks x QM rows
  uint8_t* sv = sm + NCH * QCH;
  uint8_t* sst = sm + 2 * NCH * QCH;
  uint64_t* full = reinterpret_cast<uint64_t*>(sst + STAGES * BK_ST);
  uint64_t* empty = full + STAGES;
  uint64_t* kvbar = empty + STAGES;

  const int tid = threadIdx.x;
  const int mt = L / QM;
  const int samp = blockIdx.x % A, cid = blockIdx.x / A;
  const int k_tile = cid % mt, head = cid / mt;
  const int k0 = k_tile * QM, hcol = head * DH, srow = samp * L;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 2); mbar_init(&empty[s], 4 * NWG); }
    mbar_init(kvbar, 1);
    fence_barrier_init();
  }
  __syncthreads();

  if (tid >= 128 * NWG) {                                           // producer: lane 0 q_p, dO_p; lane 1 q^T, dO^T
    const int iss = tid - 128 * NWG;
    if (iss == 0) {
      mbar_arrive_expect_tx(kvbar, 2 * NCH * QCH);
      tma_load_2d(sk, &mk, kvbar, hcol, srow + k0);
      if (NCH == 2) tma_load_2d(sk + QCH, &mk, kvbar, hcol + 32, srow + k0);
      tma_load_2d(sv, &mv, kvbar, hcol, srow + k0);
      if (NCH == 2) tma_load_2d(sv + QCH, &mv, kvbar, hcol + 32, srow + k0);
    }
    if (iss < 2) {
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sst + s * BK_ST;
        const int q0 = srow + n * BN;
        if (iss == 0) {
          mbar_arrive_expect_tx(&full[s], 2 * NCH * TCH);
          tma_load_4d(slot, &mqp, &full[s], hcol, 0, 0, q0 / 8);
          if (NCH == 2) tma_load_4d(slot + TCH, &mqp, &full[s], hcol + 32, 0, 0, q0 / 8);
          tma_load_4d(slot + NCH * TCH, &mdop, &full[s], hcol, 0, 0, q0 / 8);
          if (NCH == 2) tma_load_4d(slot + NCH * TCH + TCH, &mdop, &full[s], hcol + 32, 0, 0, q0 / 8);
        } else {
          mbar_arrive_expect_tx(&full[s], 4 * VCH);
          tma_load_2d(slot + 2 * NCH * TCH, &mqt, &full[s], q0, hcol);
          tma_load_2d(slot + 2 * NCH * TCH + VCH, &mqt, &full[s], q0 + 32, hcol);
          tma_load_2d(slot + 2 * NCH * TCH + 2 * VCH, &mdot, &full[s], q0, hcol);
          tma_load_2d(slot + 2 * NCH * TCH + 3 * VCH, &mdot, &full[s], q0 + 32, hcol);
        }
      }
    }
    return;
  }

  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;
  const int key = k0 + wg * 64 + warp * 16 + g;                     // this lane's key rows: key, key + 8
  const float* btr0 = BT + ((size_t)head * L + key) * L + 2 * t;
  const float* btr1 = btr0 + 8 * (size_t)L;
  const float* lsep = LSEP + ((size_t)samp * NH + head) * L + 2 * t;
  const float* ddp = DDP + ((size_t)samp * NH + head) * L + 2 * t;
  float kmr[2] = {0.f, 0.f};
  if (HASM) { kmr[0] = KM[(size_t)samp * L + key]; kmr[1] = KM[(size_t)samp * L + key + 8]; }
  const bool empty_uniform = TOKEN_EMPTY_UNIFORM && HASM && kmr[0] == -5e29f;
  float dk[ACC], dv[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) dk[i] = dv[i] = 0.f;
  const uint32_t ka = smem_u32(sk) + wg * 64 * 128, va = smem_u32(sv) + wg * 64 * 128;
  const uint64_t dK0 = dsw(ka), dK1 = dsw(ka + QCH), dV0 = dsw(va), dV1 = dsw(va + QCH);
  const uint32_t sbase = smem_u32(sst);
  mbar_wait(kvbar, 0);

  float st[32], dpt[32];
  uint32_t pa[32], da[32];
  for (int n = 0; n < nblocks; ++n) {
    const int sn = n % STAGES;
#pragma unroll
    for (int j = 0; j < 8; ++j) {                                   // seed: bias^T - LSE (+ key mask)
      const int c = n * BN + 8 * j;
      const float2 b0 = *reinterpret_cast<const float2*>(btr0 + c), b1 = *reinterpret_cast<const float2*>(btr1 + c);
      const float2 ls = *reinterpret_cast<const float2*>(lsep + c);
      st[4 * j] = b0.x - ls.x + kmr[0]; st[4 * j + 1] = b0.y - ls.y + kmr[0];
      st[4 * j + 2] = b1.x - ls.x + kmr[1]; st[4 * j + 3] = b1.y - ls.y + kmr[1];
    }
    mbar_wait(&full[sn], (n / STAGES) & 1);
    const uint32_t slot = sbase + sn * BK_ST;
    wgmma_fence();
    mma_head(st, dK0, dK1, dsw(slot), dsw(slot + TCH), true);                      // S^T = k q^T + bias^T - LSE
    mma_head(dpt, dV0, dV1, dsw(slot + NCH * TCH), dsw(slot + NCH * TCH + TCH), false);      // dP^T = v dO^T
    wgmma_commit();
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) { fence_reg(st[i]); fence_reg(dpt[i]); }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 dd = *reinterpret_cast<const float2*>(ddp + n * BN + 8 * j);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float p = empty_uniform ? 1.f / L : ex2(st[4 * j + e]);
        pa[4 * j + e] = tf32(p);
        da[4 * j + e] = tf32(empty_uniform ? 0.f : p * (dpt[4 * j + e] - ((e & 1) ? dd.y : dd.x)));
      }
    }
    const uint32_t tq = slot + 2 * NCH * TCH, tdo = tq + 2 * VCH;
    wgmma_fence();
#pragma unroll
    for (int i = 0; i < ACC; ++i) { fence_reg(dv[i]); fence_reg(dk[i]); }
    mma_tok(dv, pa, dsw(tdo), dsw(tdo + VCH));                                     // dV += P^T dO
    mma_tok(dk, da, dsw(tq), dsw(tq + VCH));                                       // dK += dS^T qs
    wgmma_commit();
    wgmma_wait<0>();
    if ((tid & 31) == 0 && n + STAGES < nblocks) mbar_arrive(&empty[sn]);
  }
#pragma unroll
  for (int i = 0; i < ACC; ++i) { fence_reg(dv[i]); fence_reg(dk[i]); }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const size_t r = (size_t)srow + key + 8 * h;
#pragma unroll
    for (int j = 0; j < JJ; ++j) {
      *reinterpret_cast<float2*>(DK + r * DM + hcol + 8 * j + 2 * t) =
          make_float2(dk[4 * j + 2 * h] * (1.f / LOG2E), dk[4 * j + 2 * h + 1] * (1.f / LOG2E));
      *reinterpret_cast<float2*>(DV + r * DM + hcol + 8 * j + 2 * t) = make_float2(dv[4 * j + 2 * h], dv[4 * j + 2 * h + 1]);
    }
  }
}

// dQ and dbias. grid: A * 16 * (L / QM) CTAs, the sample fastest; warpgroup w owns 64 queries, a stage is 64 keys.
// mq, mdo: q (exp2 units), dO natural rows; mkp, mvp: k, v key-permuted 4-D maps; mkt: k^T. BP [16, L, L] the forward's
// key-permuted bias (log2 units); LSE, DD [A, 16, L]; KMP [A, L] the permuted additive key mask. DQ [A L, 768] fp32;
// dbias: every (sample, 64 x 64 tile) of dS is TMA-reduce-added into mdb, [16 L, L] fp32 in the permuted key order.
template <bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, 1)
attn_tf32_dqb_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mdo,
                     const __grid_constant__ CUtensorMap mkp, const __grid_constant__ CUtensorMap mvp,
                     const __grid_constant__ CUtensorMap mkt, const __grid_constant__ CUtensorMap mdb,
                     const float* __restrict__ BP, const float* __restrict__ LSE, const float* __restrict__ DD,
                     const float* __restrict__ KMP, float* __restrict__ DQ, int L, int A, int ngi) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sq = sm;                                                 // q: 2 chunks x QM rows, then dO
  uint8_t* sdo = sm + NCH * QCH;
  uint8_t* sst = sm + 2 * NCH * QCH;
  float* sx = reinterpret_cast<float*>(sst + STAGES * BQ_ST);       // dS staging: one 64 x 64 fp32 tile per warpgroup
  uint64_t* full = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sx) + NWG * 64 * 64 * 4);
  uint64_t* empty = full + STAGES;
  uint64_t* qbar = empty + STAGES;
  uint64_t* sdone = qbar + 1;

  const int tid = threadIdx.x;
  const int mt = L / QM;
  // blockIdx: (chunk, head, m tile). A CTA walks its ngi samples chunk * ngi + gi IN ORDER and adds their dS tiles into the
  // chunk's own dbias buffer: the adds to one address come from one CTA, in sample order (bitwise repeatable)
  const int m_tile = blockIdx.x % mt, head = (blockIdx.x / mt) % NH, chunk = blockIdx.x / (mt * NH);
  const int m0 = m_tile * QM, hcol = head * DH;
  const int nblocks = L / BN;
  const int dbrow0 = (chunk * NH + head) * L + m0;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 2); mbar_init(&empty[s], 4 * NWG); }
    mbar_init(qbar, 1);
    mbar_init(sdone, 4 * NWG);                        // every consumer warp has finished the sample's q / dO / stages
    fence_barrier_init();
  }
  __syncthreads();

  if (tid >= 128 * NWG) {                                           // producer: lane 0 k_p, v_p; lane 1 k^T
    const int iss = tid - 128 * NWG;
    if (iss == 0) tma_prefetch_desc(&mdb);
    if (iss < 2) {
      for (int gi = 0; gi < ngi; ++gi) {
        const int srow = (chunk * ngi + gi) * L;
        if (iss == 0) {
          if (gi > 0) mbar_wait(sdone, (gi - 1) & 1);               // the previous sample has left q / dO
          mbar_arrive_expect_tx(qbar, 2 * NCH * QCH);
          tma_load_2d(sq, &mq, qbar, hcol, srow + m0);
          if (NCH == 2) tma_load_2d(sq + QCH, &mq, qbar, hcol + 32, srow + m0);
          tma_load_2d(sdo, &mdo, qbar, hcol, srow + m0);
          if (NCH == 2) tma_load_2d(sdo + QCH, &mdo, qbar, hcol + 32, srow + m0);
        }
        for (int n = 0; n < nblocks; ++n) {
          const int it = gi * nblocks + n;                          // the ring's running block counter
          const int s = it % STAGES;
          mbar_wait(&empty[s], ((it / STAGES) & 1) ^ 1);
          uint8_t* slot = sst + s * BQ_ST;
          const int kk = srow + n * BN;
          if (iss == 0) {
            mbar_arrive_expect_tx(&full[s], 2 * NCH * TCH);
            tma_load_4d(slot, &mkp, &full[s], hcol, 0, 0, kk / 8);
            if (NCH == 2) tma_load_4d(slot + TCH, &mkp, &full[s], hcol + 32, 0, 0, kk / 8);
            tma_load_4d(slot + NCH * TCH, &mvp, &full[s], hcol, 0, 0, kk / 8);
            if (NCH == 2) tma_load_4d(slot + NCH * TCH + TCH, &mvp, &full[s], hcol + 32, 0, 0, kk / 8);
          } else {
            mbar_arrive_expect_tx(&full[s], 2 * VCH);
            tma_load_2d(slot + 2 * NCH * TCH, &mkt, &full[s], kk, hcol);
            tma_load_2d(slot + 2 * NCH * TCH + VCH, &mkt, &full[s], kk + 32, hcol);
          }
        }
      }
    }
    return;
  }

  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;
  const int wt = tid & 127;
  const int row = m0 + wg * 64 + warp * 16 + g;                     // this lane's query rows: row, row + 8
  const float* br0 = BP + ((size_t)head * L + row) * L + 2 * t;
  const float* br1 = br0 + 8 * (size_t)L;
  for (int gi = 0; gi < ngi; ++gi) {
  const int samp = chunk * ngi + gi, srow = samp * L;
  const float* kmr = HASM ? KMP + (size_t)samp * L + 2 * t : nullptr;
  const bool empty_uniform = TOKEN_EMPTY_UNIFORM && HASM && kmr[0] == -5e29f;
  float lse[2], dd[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const size_t ri = ((size_t)samp * NH + head) * L + row + 8 * h;
    lse[h] = LSE[ri]; dd[h] = DD[ri];
  }
  float dq[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) dq[i] = 0.f;
  const uint32_t qa = smem_u32(sq) + wg * 64 * 128, da_ = smem_u32(sdo) + wg * 64 * 128;
  const uint64_t dQ0 = dsw(qa), dQ1 = dsw(qa + QCH), dO0 = dsw(da_), dO1 = dsw(da_ + QCH);
  const uint32_t sbase = smem_u32(sst);
  float* xs = sx + wg * 64 * 64;                                    // this warpgroup's dS tile, row-major [64][64]
  const int xr = warp * 16 + g;
  mbar_wait(qbar, gi & 1);

  float sc[32], dp[32];
  uint32_t da[32];
  for (int n = 0; n < nblocks; ++n) {
    const int it = gi * nblocks + n;                                // the ring's running block counter (across samples)
    const int sn = it % STAGES;
#pragma unroll
    for (int j = 0; j < 8; ++j) {                                   // seed: bias - LSE (+ key mask), keys permuted
      const int c = n * BN + 8 * j;
      const float2 b0 = *reinterpret_cast<const float2*>(br0 + c), b1 = *reinterpret_cast<const float2*>(br1 + c);
      float2 km = make_float2(0.f, 0.f);
      if (HASM) km = *reinterpret_cast<const float2*>(kmr + c);
      sc[4 * j] = b0.x + km.x - lse[0]; sc[4 * j + 1] = b0.y + km.y - lse[0];
      sc[4 * j + 2] = b1.x + km.x - lse[1]; sc[4 * j + 3] = b1.y + km.y - lse[1];
    }
    mbar_wait(&full[sn], (it / STAGES) & 1);
    const uint32_t slot = sbase + sn * BQ_ST;
    wgmma_fence();
    mma_head(sc, dQ0, dQ1, dsw(slot), dsw(slot + TCH), true);                      // S = qs k^T + bias - LSE
    mma_head(dp, dO0, dO1, dsw(slot + NCH * TCH), dsw(slot + NCH * TCH + TCH), false);       // dP = dO v^T
    wgmma_commit();
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) { fence_reg(sc[i]); fence_reg(dp[i]); }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float s = empty_uniform ? 0.f : ex2(sc[i]) * (dp[i] - dd[(i >> 1) & 1]); // masked_fill has zero dS
      sc[i] = s;
      da[i] = tf32(s);
    }
    const uint32_t tk = slot + 2 * NCH * TCH;
    wgmma_fence();
#pragma unroll
    for (int i = 0; i < ACC; ++i) fence_reg(dq[i]);
    mma_tok(dq, da, dsw(tk), dsw(tk + VCH));                                       // dQ += dS k
    wgmma_commit();
    // dbias: this block's dS tile -> shared memory -> one TMA reduce-add into the permuted dbias
    if (n > 0) {
      if (wt == 0) bulk_wait_read0();                               // the previous tile has left the staging buffer
      named_sync(1 + wg, 128);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      *reinterpret_cast<float2*>(xs + xr * 64 + 8 * j + 2 * t) = make_float2(sc[4 * j], sc[4 * j + 1]);
      *reinterpret_cast<float2*>(xs + (xr + 8) * 64 + 8 * j + 2 * t) = make_float2(sc[4 * j + 2], sc[4 * j + 3]);
    }
    fence_proxy_async();
    named_sync(1 + wg, 128);
    if (wt == 0) {
      if constexpr (DB_PER_SAMPLE)
        token_store_2d(&mdb, smem_u32(xs), n * BN, (samp * NH + head) * L + m0 + wg * 64);
      else
        bulk_reduce_add_2d(&mdb, smem_u32(xs), n * BN, dbrow0 + wg * 64);
      bulk_commit();
    }
    wgmma_wait<0>();
    if ((tid & 31) == 0 && it + STAGES < ngi * nblocks) mbar_arrive(&empty[sn]);
  }
  if (wt == 0) bulk_wait0();       // this sample's dbias adds LANDED before the next sample adds to the same addresses
  named_sync(5, 128 * NWG);
  if ((tid & 31) == 0) mbar_arrive(sdone);
#pragma unroll
  for (int i = 0; i < ACC; ++i) fence_reg(dq[i]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float* orow = DQ + ((size_t)srow + row + 8 * h) * DM + hcol;
#pragma unroll
    for (int j = 0; j < JJ; ++j)
      *reinterpret_cast<float2*>(orow + 8 * j + 2 * t) = make_float2(dq[4 * j + 2 * h] * RSQD, dq[4 * j + 2 * h + 1] * RSQD);
  }
  }  // samples
}
}  // namespace

// ------------------------------------------------------------------------------------------------ host
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "no TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
CUtensorMap encode(const float* ptr, int rank, const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                   CUtensorMapSwizzle swz = CU_TENSOR_MAP_SWIZZLE_128B) {
  CUtensorMap map{};
  const cuuint32_t elem[4] = {1, 1, 1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, rank, const_cast<float*>(ptr), dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "tensor map encode failed");
  return map;
}
// rows x cols fp32, row stride rs (elements); a box of 32 columns x brows rows
CUtensorMap map2d(const float* p, uint64_t rows, uint64_t cols, uint64_t rs, uint32_t brows) {
  const cuuint64_t dims[2] = {cols, rows}, strides[1] = {rs * 4};
  const cuuint32_t box[2] = {32, brows};
  return encode(p, 2, dims, strides, box);
}
// the k rows, permuted within each group of 8 (smem row 2j + i = key 4i + j): dims (columns, i, j, row group)
CUtensorMap map_kperm(const float* p, uint64_t rows, uint64_t cols, uint64_t rs) {
  const cuuint64_t dims[4] = {cols, 2, 4, rows / 8}, strides[3] = {4 * rs * 4, rs * 4, 8 * rs * 4};
  const cuuint32_t box[4] = {32, 2, 4, 8};
  return encode(p, 4, dims, strides, box);
}

template <bool GATED, bool HASM>
void launch(int A, int L, const CUtensorMap& mq, const CUtensorMap& mk, const CUtensorMap& mvt, const float* bias,
            const float* km, float* o, float* lse, float* out, long so, long goff) {
#if FWD_PIPE
  const size_t smem = 1024 + Q_BYTES + PSTAGES * ST_BYTES + 256;
  auto kern = attn_tf32_fwd_pipe_kernel<GATED, HASM>;
#else
  const size_t smem = 1024 + Q_BYTES + STAGES * ST_BYTES + 256;
  auto kern = attn_tf32_fwd_kernel<GATED, HASM>;
#endif
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<A * NH * (L / QM), 128 * NWG + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mq, mk, mvt, bias, km, o, lse, out, so,
                                                                                    goff, L, A);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool HASM>
void launch_mc(int A, int L, int cs, const CUtensorMap& mq, const CUtensorMap& mk, const CUtensorMap& mvt, const CUtensorMap& mb,
               const float* km, float* out, long so, long goff) {
  const size_t smem = Q_BYTES + STAGES * ST_MC + 256;                 // the 1024-B slack is dropped: 2 CTAs / SM at width 32
  auto kern = attn_tf32_fwd_mc_kernel<true, HASM>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(A * NH * (L / QM));
  cfg.blockDim = dim3(128 * NWG + 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cs; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, mq, mk, mvt, mb, cs, km, (float*)nullptr, (float*)nullptr, out, so, goff, L, A) == cudaSuccess,
              "mc launch");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void check_f32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, name, ": contiguous fp32");
}
}  // namespace

// Training forward: q (in exp2 units), k [A L, 768] fp32; vt = v^T [768, A L]; bias [16, L, L] fp32 (log2 units) and kmask
// [A, L] additive (or None), both key-permuted. O [A L, 768] fp32, LSE [A, 16, L] fp32 (log2).
void attn_tf32_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor vt, torch::Tensor bias, c10::optional<torch::Tensor> kmask,
                   torch::Tensor O, torch::Tensor LSE) {
  const int L = bias.size(1), A = q.size(0) / L;
  for (auto* t : {&q, &k, &vt, &bias, &O, &LSE}) check_f32(*t, "operand");
  TORCH_CHECK(q.size(1) == DM && k.sizes() == q.sizes() && O.sizes() == q.sizes() && vt.size(0) == DM && vt.size(1) == q.size(0)
              && q.size(0) == (int64_t)A * L && bias.size(0) == NH && bias.size(2) == L && LSE.numel() == (int64_t)A * NH * L
              && L % QM == 0, "shapes");
  const bool hasm = kmask.has_value() && kmask->defined();
  if (hasm) { check_f32(*kmask, "kmask"); TORCH_CHECK(kmask->numel() == (int64_t)A * L, "kmask [A, L]"); }
  const at::cuda::CUDAGuard guard(q.device());
  const uint64_t rows = (uint64_t)A * L;
  const auto mq = map2d(q.data_ptr<float>(), rows, DM, DM, QM), mk = map_kperm(k.data_ptr<float>(), rows, DM, DM),
             mvt = map2d(vt.data_ptr<float>(), DM, rows, rows, DH);
  auto f = [](torch::Tensor& t) { return t.data_ptr<float>(); };
  if (hasm) launch<false, true>(A, L, mq, mk, mvt, f(bias), kmask->data_ptr<float>(), f(O), f(LSE), nullptr, 0, 0);
  else launch<false, false>(A, L, mq, mk, mvt, f(bias), nullptr, f(O), f(LSE), nullptr, 0, 0);
}

// The token DiT training forward: as attn_tf32_fwd, but og = sigmoid(g) O into OG (g from qkvg [A L, 3072] fp32), no O.
void attn_tf32_fwd_og(torch::Tensor q, torch::Tensor k, torch::Tensor vt, torch::Tensor bias, c10::optional<torch::Tensor> kmask,
                      torch::Tensor qkvg, torch::Tensor OG, torch::Tensor LSE) {
  const int L = bias.size(1), A = q.size(0) / L;
  for (auto* t : {&q, &k, &vt, &bias, &qkvg, &OG, &LSE}) check_f32(*t, "operand");
  TORCH_CHECK(q.size(1) == DM && k.sizes() == q.sizes() && OG.sizes() == q.sizes() && vt.size(0) == DM && vt.size(1) == q.size(0)
              && qkvg.size(0) == q.size(0) && qkvg.size(1) == 4 * DM && q.size(0) == (int64_t)A * L && bias.size(0) == NH
              && bias.size(2) == L && LSE.numel() == (int64_t)A * NH * L && L % QM == 0, "shapes");
  const bool hasm = kmask.has_value() && kmask->defined();
  if (hasm) { check_f32(*kmask, "kmask"); TORCH_CHECK(kmask->numel() == (int64_t)A * L, "kmask [A, L]"); }
  const at::cuda::CUDAGuard guard(q.device());
  const uint64_t rows = (uint64_t)A * L;
  const auto mq = map2d(q.data_ptr<float>(), rows, DM, DM, QM), mk = map_kperm(k.data_ptr<float>(), rows, DM, DM),
             mvt = map2d(vt.data_ptr<float>(), DM, rows, rows, DH);
  auto f = [](torch::Tensor& t) { return t.data_ptr<float>(); };
  if (hasm) launch<false, true>(A, L, mq, mk, mvt, f(bias), kmask->data_ptr<float>(), f(OG), f(LSE), f(qkvg), 4 * DM, 3 * DM);
  else launch<false, false>(A, L, mq, mk, mvt, f(bias), nullptr, f(OG), f(LSE), f(qkvg), 4 * DM, 3 * DM);
}

// Inference: qkg [S L, >= 3 * 768] fp32 (q | k at columns 0 | 768, q in exp2 units; g at column goff), vt = v^T [768, S L];
// bias_all [nb 16, L, L] fp32, key-permuted, log2 units (masked keys -inf). Writes sigmoid(g) softmax(q k^T + bias) v over q
// for ``block``'s slice of the bias.
void attn_tf32_inf(torch::Tensor qkg, torch::Tensor vt, torch::Tensor bias_all, int64_t block, int64_t S, int64_t goff) {
  const int L = bias_all.size(1);
  for (auto* t : {&qkg, &vt, &bias_all}) check_f32(*t, "operand");
  const uint64_t rows = (uint64_t)S * L, rs = qkg.size(1);
  TORCH_CHECK(qkg.size(0) == (int64_t)rows && rs >= 2 * DM && goff + DM <= (int64_t)rs && vt.size(0) == DM
              && vt.size(1) == (int64_t)rows && bias_all.size(2) == L && bias_all.size(0) >= (block + 1) * NH && L % QM == 0,
              "shapes");
  const at::cuda::CUDAGuard guard(qkg.device());
  float* base = qkg.data_ptr<float>();
  const auto mq = map2d(base, rows, DM, rs, QM), mk = map_kperm(base + DM, rows, DM, rs),
             mvt = map2d(vt.data_ptr<float>(), DM, rows, rows, DH);
  float* bias = bias_all.data_ptr<float>() + (size_t)block * NH * L * L;
  int cs = 1;
  for (int c = 8; c >= 2; --c) if (S % c == 0) { cs = c; break; }   // samples of one (head, row tile) share the bias tile
  if (MC_BIAS && cs > 1) {
    const auto mb = map2d(bias, (uint64_t)NH * L, L, L, QM);          // [head * L + row, key]: boxes of 32 keys x QM rows
    launch_mc<false>((int)S, L, cs, mq, mk, mvt, mb, nullptr, base, (long)rs, (long)goff);
    return;
  }
  launch<true, false>((int)S, L, mq, mk, mvt, bias, nullptr, nullptr, nullptr, base, (long)rs, (long)goff);
}

// Backward, dK and dV: q (exp2 units), dob = dO, k, v [A L, 768] fp32; qt, dot = q^T, dO^T [768, A L]; bt [16, L(key), L]
// the bias transposed with its query columns key_perm-permuted; lsep, ddp [A, 16, L] the LSE and D = rowsum(dO O) permuted
// the same way; kmask [A, L] additive (natural) or None. Writes DK, DV [A L, 768] fp32.
void attn_tf32_dkv(torch::Tensor q, torch::Tensor dob, torch::Tensor qt, torch::Tensor dot, torch::Tensor k, torch::Tensor v,
                   torch::Tensor bt, torch::Tensor lsep, torch::Tensor ddp, c10::optional<torch::Tensor> kmask,
                   torch::Tensor DK, torch::Tensor DV) {
  const int L = bt.size(1), A = q.size(0) / L;
  for (auto* t : {&q, &dob, &qt, &dot, &k, &v, &bt, &lsep, &ddp, &DK, &DV}) check_f32(*t, "operand");
  const uint64_t rows = (uint64_t)A * L;
  TORCH_CHECK(q.size(1) == DM && q.size(0) == (int64_t)rows && qt.size(0) == DM && qt.size(1) == (int64_t)rows && L % QM == 0,
              "shapes");
  const bool hasm = kmask.has_value() && kmask->defined();
  const at::cuda::CUDAGuard guard(q.device());
  const auto mk = map2d(k.data_ptr<float>(), rows, DM, DM, QM), mv = map2d(v.data_ptr<float>(), rows, DM, DM, QM),
             mqp = map_kperm(q.data_ptr<float>(), rows, DM, DM), mdop = map_kperm(dob.data_ptr<float>(), rows, DM, DM),
             mqt = map2d(qt.data_ptr<float>(), DM, rows, rows, DH), mdot = map2d(dot.data_ptr<float>(), DM, rows, rows, DH);
  const size_t smem = 1024 + 2 * NCH * QCH + STAGES * BK_ST + 256;
  auto kern = hasm ? attn_tf32_dkv_kernel<true> : attn_tf32_dkv_kernel<false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<A * NH * (L / QM), 128 * NWG + 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      mk, mv, mqp, mdop, mqt, mdot, bt.data_ptr<float>(), lsep.data_ptr<float>(), ddp.data_ptr<float>(),
      hasm ? kmask->data_ptr<float>() : nullptr, DK.data_ptr<float>(), DV.data_ptr<float>(), L, A);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Backward, dQ and dbias: q (exp2 units), dob, k, v [A L, 768] fp32; kt = k^T [768, A L]; bp [16, L, L] the forward's
// key-permuted bias; lse, dd [A, 16, L]; kmp [A, L] the key-permuted additive mask or None. Writes DQ [A L, 768] fp32 and
// ADDS every sample's dS into dbp [16, L, L] (zero it first), key-permuted.
void attn_tf32_dqb(torch::Tensor q, torch::Tensor dob, torch::Tensor k, torch::Tensor v, torch::Tensor kt, torch::Tensor bp,
                   torch::Tensor lse, torch::Tensor dd, c10::optional<torch::Tensor> kmp, torch::Tensor DQ, torch::Tensor dbp) {
  const int L = bp.size(1), A = q.size(0) / L;
  for (auto* t : {&q, &dob, &k, &v, &kt, &bp, &lse, &dd, &DQ, &dbp}) check_f32(*t, "operand");
  const uint64_t rows = (uint64_t)A * L;
  TORCH_CHECK(q.size(1) == DM && q.size(0) == (int64_t)rows && kt.size(0) == DM && kt.size(1) == (int64_t)rows
              && L % QM == 0, "shapes");
  int C = A;   // chunk buffers: DB_PER_SAMPLE: A (one per sample, one sample a CTA); else C | A, each CTA walks A / C samples in order
  if constexpr (DB_PER_SAMPLE) {
    TORCH_CHECK(dbp.sizes() == at::IntArrayRef({A, NH, L, L}), "per-sample dbias [A,16,L,L]");
  } else {
    TORCH_CHECK(dbp.dim() == 4 && dbp.size(1) == NH && dbp.size(2) == L && dbp.size(3) == L && A % dbp.size(0) == 0,
                "chunked dbias [C,16,L,L], C | A");
    C = (int)dbp.size(0);
  }
  const int ngi = A / C;
  const bool hasm = kmp.has_value() && kmp->defined();
  const at::cuda::CUDAGuard guard(q.device());
  const auto mq = map2d(q.data_ptr<float>(), rows, DM, DM, QM), mdo = map2d(dob.data_ptr<float>(), rows, DM, DM, QM),
             mkp = map_kperm(k.data_ptr<float>(), rows, DM, DM), mvp = map_kperm(v.data_ptr<float>(), rows, DM, DM),
             mkt = map2d(kt.data_ptr<float>(), DM, rows, rows, DH);
  const cuuint64_t ddims[2] = {(cuuint64_t)L, (cuuint64_t)C * NH * L}, dstr[1] = {(cuuint64_t)L * 4};
  const cuuint32_t dbox[2] = {64, 64};
  const auto mdb = encode(dbp.data_ptr<float>(), 2, ddims, dstr, dbox, CU_TENSOR_MAP_SWIZZLE_NONE);
  const size_t smem = 1024 + 2 * NCH * QCH + STAGES * BQ_ST + NWG * 64 * 64 * 4 + 256;
  auto kern = hasm ? attn_tf32_dqb_kernel<true> : attn_tf32_dqb_kernel<false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<C * NH * (L / QM), 128 * NWG + 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      mq, mdo, mkp, mvp, mkt, mdb, bp.data_ptr<float>(), lse.data_ptr<float>(), dd.data_ptr<float>(),
      hasm ? kmp->data_ptr<float>() : nullptr, DQ.data_ptr<float>(), L, A, ngi);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_tf32_fwd", &attn_tf32_fwd);
  m.def("attn_tf32_fwd_og", &attn_tf32_fwd_og);
  m.def("attn_tf32_inf", &attn_tf32_inf);
  m.def("attn_tf32_dkv", &attn_tf32_dkv);
  m.def("attn_tf32_dqb", &attn_tf32_dqb);
}
