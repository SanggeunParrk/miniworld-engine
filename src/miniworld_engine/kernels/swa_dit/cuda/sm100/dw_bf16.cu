// dw_bf16.cu — the five weight-gradient GEMMs of the bf16 SWA atom block's backward in one launch on sm_100a (kind::f16 bf16 tcgen05.mma, fp32
// accumulation), the bf16 counterpart of dw_tf32.cu (same split-K structure, fp32 partial sums, deterministic reduction):
//   dWqkvg [4C, C] = dP^T x,  dWo = datt^T gated,  dWu [2 NHID, C] = dab^T y,  dWd [C, NHID] = dffn^T hh
// every operand a row-major [R, cols] bf16 tensor; D[m][n] = sum_r A[r][m] B[r][n] with both operands MN-major in the 128-B swizzle
// (TMA boxes [32 rows][64 bf16], UMMA layout type 2, SBO 1024, LBO = the distance between 64-column atoms).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;



#ifndef NS
#define NS 5                                                               // ring stages
#endif
constexpr int BKR = 32;                                                    // reduction rows per stage
constexpr int ABOX = BKR * 128;                                            // one [32 rows][64 bf16] box: 4 KB
constexpr int NBOX = 10;                                                   // boxes of a stage (<= 8 A atoms + 2 B atoms)
constexpr int STG = NBOX * ABOX;                                           // 40 KB
constexpr int O_BAR = NS * STG, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
// problem p: m_total = 128 MBf(p), n = NNf(p), A columns = 32 AAf(p), B columns = 32 BBf(p)
DEVI int MBf(int p) { return (p & 1) ? 1 : 4; }
DEVI int NNf(int p) { return p == 3 ? 256 : 128; }
DEVI int AAf(int p) { return (p & 1) ? 2 : 8; }                            // A atoms of 64 columns
DEVI int BBf(int p) { return p == 3 ? 4 : 2; }

struct Bars {
  uint64_t full[NS], empty[NS], done;
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(256, 1)
swa_dw_bf16_sm100(const __grid_constant__ CUtensorMap ma0, const __grid_constant__ CUtensorMap mb0, const __grid_constant__ CUtensorMap ma1,
                  const __grid_constant__ CUtensorMap mb1, const __grid_constant__ CUtensorMap ma2, const __grid_constant__ CUtensorMap mb2,
                  const __grid_constant__ CUtensorMap ma3, const __grid_constant__ CUtensorMap mb3, float* __restrict__ WS, int R,
                  int ns0, int ns1, int ns2, int ns3) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // problem and slice of this CTA (scalars: an indexed array would live in local memory)
  int p = 0, slice = (int)blockIdx.x, nsp = ns0;
  if (slice >= ns0) {
    slice -= ns0; p = 1; nsp = ns1;
    if (slice >= ns1) {
      slice -= ns1; p = 2; nsp = ns2;
      if (slice >= ns2) { slice -= ns2; p = 3; nsp = ns3; }
    }
  }
  const int rows_per = (((R + nsp - 1) / nsp) + BKR - 1) / BKR * BKR;
  const int r0 = min(slice * rows_per, R), r1 = min(R, r0 + rows_per);
  const int nblk = (r1 - r0) / BKR;
  const int mbn = MBf(p), nn = NNf(p), aa = AAf(p), bb = BBf(p);
  const CUtensorMap* mA = p == 0 ? &ma0 : p == 1 ? &ma1 : p == 2 ? &ma2 : &ma3;
  const CUtensorMap* mB = p == 0 ? &mb0 : p == 1 ? &mb1 : p == 2 ? &mb2 : &mb3;
  // workspace of this slice: [ns][m_total][n] fp32 after the earlier problems'
  size_t wso = 0;
  if (p > 0) wso += (size_t)ns0 * (4 * 128 * 128);
  if (p > 1) wso += (size_t)ns1 * (1 * 128 * 128);
  if (p > 2) wso += (size_t)ns2 * (4 * 128 * 128);
  float* ws = WS + wso + (size_t)slice * mbn * 128 * nn;

  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.done, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    if (lane == 0) {
      pdl_wait();                                                          // the operands come from the previous kernels
      for (int blk = 0; blk < nblk; ++blk) {
        const int s = blk % NS;
        if (blk >= NS) mbar_wait(&B.empty[s], ((blk / NS) - 1) & 1);
        const uint32_t st = su + s * STG;
        mbar_expect_tx(&B.full[s], (uint32_t)(aa + bb) * ABOX);
        const int row = r0 + blk * BKR;
        for (int i = 0; i < aa; ++i) tma_load_2d(st + i * ABOX, mA, &B.full[s], 64 * i, row);
        for (int j = 0; j < bb; ++j) tma_load_2d(st + (aa + j) * ABOX, mB, &B.full[s], 64 * j, row);
      }
    }
  } else if (warp == 1) {
    const uint32_t id = idesc_bf16(128, nn, 1, 1);
    for (int blk = 0; blk < nblk; ++blk) {
      const int s = blk % NS;
      mbar_wait(&B.full[s], (blk / NS) & 1);
      tc_fence_after();
      const uint32_t st = su + s * STG;
      if (elect_one()) {
        for (int mb = 0; mb < mbn; ++mb) {
          const uint64_t da = desc_mn128(st + 2 * mb * ABOX, ABOX), db = desc_mn128(st + aa * ABOX, ABOX);
#pragma unroll
          for (int ks = 0; ks < BKR / 16; ++ks)                             // K = 16 reduction rows: 2 KB of every atom
            umma_ss(tmem + mb * nn, da + (uint64_t)((ks * 2048) >> 4), db + (uint64_t)((ks * 2048) >> 4), id, (blk > 0 || ks > 0) ? 1u : 0u);
        }
        tc_commit(&B.empty[s]);
        if (blk == nblk - 1) tc_commit(&B.done);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    const int lq = warp & 3, r = lq * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    if (nblk > 0) { mbar_wait(&B.done, 0); tc_fence_after(); }
    for (int mb = 0; mb < mbn; ++mb) {
      float* dst = ws + ((size_t)mb * 128 + r) * nn;
      for (int c = 0; c < nn; c += 32) {
        uint32_t v[32];
        if (nblk > 0) { tmem_ld32(trow + mb * nn + c, v); tmem_wait_ld(); }
        else {
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = 0u;
        }
#pragma unroll
        for (int e = 0; e < 32; e += 4) *reinterpret_cast<uint4*>(dst + c + e) = make_uint4(v[e], v[e + 1], v[e + 2], v[e + 3]);
      }
    }
    tc_fence_before();
  }
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

// out_p[e] = sum over slices s (in order) of WS[offset_p + s m_total n + e]; one thread per four consecutive elements
DEVI void dw_sum(const float* __restrict__ w, float* __restrict__ out, int ns, size_t n4, size_t i) {
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  for (int s = 0; s < ns; ++s) {
    const float4 v = w4[(size_t)s * n4 + i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  reinterpret_cast<float4*>(out)[i] = acc;
}
extern "C" __global__ void __launch_bounds__(256)
swa_dw_reduce_bf16_sm100(const float* __restrict__ WS, float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2,
                         float* __restrict__ o3, int ns0, int ns1, int ns2, int ns3) {
  pdl_wait();                                                              // the partial sums come from the previous kernel
  constexpr size_t E0 = 4 * 128 * 128, E1 = 1 * 128 * 128, E2 = 4 * 128 * 128, E3 = 1 * 128 * 256;      // elements of the four outputs
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                // float4 index over the four outputs, back to back
  size_t off = 0;
  if (i < E0 / 4) { dw_sum(WS + off, o0, ns0, E0 / 4, i); return; }
  i -= E0 / 4; off += (size_t)ns0 * E0;
  if (i < E1 / 4) { dw_sum(WS + off, o1, ns1, E1 / 4, i); return; }
  i -= E1 / 4; off += (size_t)ns1 * E1;
  if (i < E2 / 4) { dw_sum(WS + off, o2, ns2, E2 / 4, i); return; }
  i -= E2 / 4; off += (size_t)ns2 * E2;
  if (i < E3 / 4) dw_sum(WS + off, o3, ns3, E3 / 4, i);
}
