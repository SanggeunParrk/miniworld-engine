// gemm_swiglu_sm100.cu -- h = bf16(silu(a) * b) with [a | b] = X W^T, on sm_100a (tcgen05, TMEM, TMA).
//
// The token DiT transition's expand GEMM with the SwiGLU in its epilogue, so the [M, 2H] pre-activation never reaches
// HBM. W is packed with gate / up rows interleaved (a0, b0, a1, b1, ...), so a 256-column accumulator tile holds 128
// (a, b) pairs and yields 128 columns of h.
//
//   X  [M, K]   bf16, K contiguous (the AdaLN output)            TMA map, box [64 (K), 128 (M)], 128-B swizzle
//   W  [2H, K]  bf16, interleaved rows                           TMA map, box [64 (K), 256 (N)], 128-B swizzle
//   H  [M, H]   bf16                                             TMA map, box [64, 128], 128-B swizzle (store)
//
// One CTA per 128 x 256 accumulator tile. Warp 0 streams X / W k-blocks through a STAGES-deep ring, warp 1 allocates
// TMEM and issues M128 N256 K16 products (fp32 accumulate, 256 TMEM columns), warps 2-5 run the epilogue: each reads its
// 32-lane quarter of the accumulator 32 columns at a time, forms 16 h values per row, writes them into a 128-B-swizzled
// staging tile, and one thread stores the two 64-column halves with TMA (a per-row st.global epilogue would be 32
// cache lines per warp store).
#include "sm100.cuh"

using namespace s100;

#ifndef STAGES
#define STAGES 4
#endif

constexpr int BM = 128, BN = 256, BK = 64;
constexpr int A_BYTES = BM * BK * 2, B_BYTES = BN * BK * 2, STAGE_BYTES = A_BYTES + B_BYTES;
constexpr int H_BYTES = BM * 64 * 2;                       // one 64-column half of the h tile
constexpr int SMEM_BYTES = 1024 + STAGES * STAGE_BYTES + 2 * H_BYTES + 256;
constexpr int NTHREADS = 192;

extern "C" __global__ void __launch_bounds__(NTHREADS, 1) gemm_swiglu_sm100(const __grid_constant__ CUtensorMap mx,
    const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mh, int K) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* base = (unsigned char*)(((uintptr_t)raw + 1023) & ~(uintptr_t)1023);
  unsigned char* stage = base;                                     // STAGES x [A | B]
  unsigned char* hst = base + STAGES * STAGE_BYTES;                // 2 x [128][64] bf16, swizzled
  uint64_t* full = (uint64_t*)(hst + 2 * H_BYTES);
  uint64_t* empty = full + STAGES;
  uint64_t* done = empty + STAGES;
  uint32_t* tslot = (uint32_t*)(done + 1);

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int nk = K / BK;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
    mbar_init(done, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mw); prefetch_map(&mh);
  }
  if (warp == 1) { tmem_alloc(smem_u32(tslot), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {                                                  // TMA producer
    if (elect_one()) {
      for (int kb = 0; kb < nk; ++kb) {
        const int s = kb % STAGES;
        if (kb >= STAGES) mbar_wait(&empty[s], ((kb / STAGES) - 1) & 1);
        const uint32_t a = smem_u32(stage + s * STAGE_BYTES), b = a + A_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES);
        tma_load_2d(a, &mx, &full[s], kb * BK, m0);
        tma_load_2d(b, &mw, &full[s], kb * BK, n0);
      }
    }
  } else if (warp == 1) {                                           // MMA issuer
    constexpr uint32_t idesc = idesc_bf16(BM, BN);
    for (int kb = 0; kb < nk; ++kb) {
      const int s = kb % STAGES;
      mbar_wait(&full[s], (kb / STAGES) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t a = smem_u32(stage + s * STAGE_BYTES), b = a + A_BYTES;
#pragma unroll
        for (int k = 0; k < BK / 16; ++k)                           // K16 steps: +32 B inside the 128-B swizzle atom
          umma_ss(tmem, desc_k128(a + k * 32), desc_k128(b + k * 32), idesc, (kb | k) != 0);
        tc_commit(&empty[s]);                                       // the stage is free once these products are done
        if (kb == nk - 1) tc_commit(done);
      }
      __syncwarp();
    }
  } else {                                                          // epilogue: warps 2..5
    const int q = warp % 4;                                         // TMEM lane quarter this warp may read
    const int row = q * 32 + lane;
    mbar_wait(done, 0);
    tc_fence_after();
#pragma unroll 1
    for (int c = 0; c < BN; c += 32) {
      uint32_t v[32];
      tmem_ld32(tmem + ((uint32_t)(q * 32) << 16) + c, v);
      tmem_wait_ld();
      uint32_t hp[8];                                               // 16 h values, bf16 pairs
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float a0 = __uint_as_float(v[4 * j]), b0 = __uint_as_float(v[4 * j + 1]);
        const float a1 = __uint_as_float(v[4 * j + 2]), b1 = __uint_as_float(v[4 * j + 3]);
        const float h0 = a0 * rcpf(1.f + ex2f(-1.4426950408889634f * a0)) * b0;
        const float h1 = a1 * rcpf(1.f + ex2f(-1.4426950408889634f * a1)) * b1;
        hp[j] = pack_bf16(h0, h1);
      }
      const int hc = c / 2;                                         // first h column of this chunk (16 wide)
      unsigned char* half = hst + (hc / 64) * H_BYTES;
      const uint32_t q16 = (hc % 64) / 8;                           // 16-B chunk index inside the 128-B row
      const uint32_t sa = smem_u32(half);
      sts128(sa + sw128(row, q16), make_uint4(hp[0], hp[1], hp[2], hp[3]));
      sts128(sa + sw128(row, q16 + 1), make_uint4(hp[4], hp[5], hp[6], hp[7]));
    }
    fence_proxy_async();
    named_bar_sync(1, 128);
    if (warp == 2 && elect_one()) {
      tma_store_2d(&mh, smem_u32(hst), n0 / 2, m0);
      tma_store_2d(&mh, smem_u32(hst + H_BYTES), n0 / 2 + 64, m0);
      tma_store_commit();
      tma_store_wait0();
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}
