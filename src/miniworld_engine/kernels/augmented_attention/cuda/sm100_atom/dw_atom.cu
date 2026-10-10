// dw_atom.cu — the weight-gradient products of the AF3 atom block's backward in ONE launch (+ one reduction launch), sm_100a, tcgen05, fp32 sums
// (the split-K structure of the SWA block's dw_bf16.cu / dw_tf32.cu, with a runtime problem table and the gradients written straight to the weights'
// own buffers, in their dtypes):
//
//   D_p[m][n] = sum_r A_p[r][m] B_p[r][n]          problem p: A_p [R, 128 mb_p] (a column view of a [R, *] tensor), B_p [R, nn_p]
//
// Every operand is a row-major [R, cols] tensor; D = A^T B with both operands MN-major in the 128-B swizzle (bf16: TMA boxes [32 rows][64 bf16], UMMA
// layout type 2, SBO 1024, LBO = the distance between 64-column atoms; fp32 (-DDW_F32, TF32 MMAs): boxes [16 rows][32 fp32] with 32-B atoms, "128a32").
// A CTA takes a row slice of ONE problem (CTAs per problem proportional to its bytes), streams it by TMA through a ring of stages, accumulates the whole
// [128 mb, nn] output in TMEM (mb accumulators of nn columns, <= 512) and writes the partial sum to a workspace; the reduction kernel adds the slices of
// every element in slice order (deterministic) and converts it to the destination dtype / address of its 128-row block (Out table).
// These products read ~1 GB at 196608 rows: the kernel is bound by memory, not by the tensor pipe.
// Warps: 0 TMA producer, 1 MMA issuer, 2 TMEM allocator, 4-7 epilogue (thread = row m = TMEM lane).
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int MAXP = 12, MAXO = 24;
#ifdef DW_F32
#ifndef DW_BKR
#define DW_BKR 16
#endif
constexpr int ATOM = 32, BKR = DW_BKR, ELEM = 4;                          // columns per atom / box, rows per stage, bytes per element
#else
constexpr int ATOM = 64, BKR = 32, ELEM = 2;
#endif
constexpr int ABOX = BKR * 128;                                            // one box: [BKR rows][128 B]
#ifndef NS
#define NS 5
#endif
constexpr int NBOX = 10 * (64 / ATOM);                                     // boxes of a stage: (8 + 2) 64-column atoms
constexpr int STG = NBOX * ABOX;
constexpr int O_BAR = NS * STG, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");

struct Maps { CUtensorMap a[MAXP]; CUtensorMap b[MAXP]; };
struct Tab {
  int np;                       // problems
  int mb[MAXP];                 // 128-column blocks of A
  int nn[MAXP];                 // columns of B (= the accumulator width)
  int ns[MAXP];                 // slices (CTAs)
  int outbase[MAXP];            // first Out entry of the problem (one per 128-row block)
  long long wsoff[MAXP];        // workspace offset (floats)
  long long e4off[MAXP];        // float4 offset of the problem in the reduction's linear index
  long long e4total;
};
struct Out { void* ptr[MAXO]; int bf16[MAXO]; };

struct Bars {
  uint64_t full[NS], empty[NS], done;
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(256, 1)
atom_dw(const __grid_constant__ Maps maps, const __grid_constant__ Tab tab, float* __restrict__ WS, int R) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int p = 0, slice = (int)blockIdx.x;
  while (p + 1 < tab.np && slice >= tab.ns[p]) { slice -= tab.ns[p]; ++p; }
  const int nsp = tab.ns[p];
  const int rows_per = (((R + nsp - 1) / nsp) + BKR - 1) / BKR * BKR;
  const int r0 = min(slice * rows_per, R), r1 = min(R, r0 + rows_per);
  const int nblk = (r1 - r0) / BKR;
  const int mbn = tab.mb[p], nn = tab.nn[p];
  const int aa = mbn * 128 / ATOM, bb = nn / ATOM;                         // atoms (boxes) of A and of B per stage
  const CUtensorMap* mA = &maps.a[p];
  const CUtensorMap* mB = &maps.b[p];
  float* ws = WS + tab.wsoff[p] + (size_t)slice * mbn * 128 * nn;

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
        for (int i = 0; i < aa; ++i) tma_load_2d(st + i * ABOX, mA, &B.full[s], ATOM * i, row);
        for (int j = 0; j < bb; ++j) tma_load_2d(st + (aa + j) * ABOX, mB, &B.full[s], ATOM * j, row);
      }
    }
  } else if (warp == 1) {
#ifdef DW_F32
    const uint32_t id = idesc_tf32(128, nn, 1, 1);
#else
    const uint32_t id = idesc_bf16(128, nn, 1, 1);
#endif
    for (int blk = 0; blk < nblk; ++blk) {
      const int s = blk % NS;
      mbar_wait(&B.full[s], (blk / NS) & 1);
      tc_fence_after();
      const uint32_t st = su + s * STG;
      if (elect_one()) {
        for (int mb = 0; mb < mbn; ++mb) {
#ifdef DW_F32
          const uint64_t da = desc_mn32b(st + 4 * mb * ABOX, ABOX), db = desc_mn32b(st + aa * ABOX, ABOX);
#pragma unroll
          for (int ks = 0; ks < BKR / 8; ++ks)                            // K = 8 reduction rows: 1 KB of every atom
            umma_ss_tf32(tmem + mb * nn, da + (uint64_t)((ks * 1024) >> 4), db + (uint64_t)((ks * 1024) >> 4), id, (blk > 0 || ks > 0) ? 1u : 0u);
#else
          const uint64_t da = desc_mn128(st + 2 * mb * ABOX, ABOX), db = desc_mn128(st + aa * ABOX, ABOX);
#pragma unroll
          for (int ks = 0; ks < BKR / 16; ++ks)                           // K = 16 reduction rows: 2 KB of every atom
            umma_ss(tmem + mb * nn, da + (uint64_t)((ks * 2048) >> 4), db + (uint64_t)((ks * 2048) >> 4), id, (blk > 0 || ks > 0) ? 1u : 0u);
#endif
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

// out of problem p, block mb, row, column group of four = sum over its slices (in order) of the workspace; fp32 or bf16 (RN) at the 128-row block's address
extern "C" __global__ void __launch_bounds__(256)
atom_dw_reduce(const float* __restrict__ WS, const __grid_constant__ Tab tab, const __grid_constant__ Out out) {
  pdl_wait();                                                              // the partial sums come from the previous kernel
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tab.e4total) return;
  int p = 0;
  while (p + 1 < tab.np && i >= tab.e4off[p + 1]) ++p;
  const long long j = i - tab.e4off[p];
  const int nn = tab.nn[p], n4 = nn >> 2;
  const long long tile4 = 128ll * n4;                                      // float4 of one 128-row block
  const int mb = (int)(j / tile4);
  const long long rem = j % tile4;
  const int row = (int)(rem / n4), c4 = (int)(rem % n4);
  const long long stride = (long long)tab.mb[p] * 128 * nn;                // floats per slice
  const float4* w = reinterpret_cast<const float4*>(WS + tab.wsoff[p]) + j + (long long)mb * 0;   // j already indexes block-major
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = 0; s < tab.ns[p]; ++s) {
    const float4 v = w[(size_t)s * (stride >> 2)];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  const int e = tab.outbase[p] + mb;
  if (out.bf16[e]) {
    __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(out.ptr[e]) + (size_t)row * nn + 4 * c4;
    *reinterpret_cast<uint2*>(d) = make_uint2(pack_bf16(acc.x, acc.y), pack_bf16(acc.z, acc.w));
  } else {
    *reinterpret_cast<float4*>(reinterpret_cast<float*>(out.ptr[e]) + (size_t)row * nn + 4 * c4) = acc;
  }
}
