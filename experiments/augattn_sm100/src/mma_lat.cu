// tcgen05 MMA round-trip latency probe (one CTA, one warp issues, garbage smem operands): issue a pattern, commit, wait, time it.
#include "sm100.cuh"
using namespace s100;
extern "C" __global__ void __launch_bounds__(128, 1) mma_lat(unsigned long long* out, int reps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  for (int i = threadIdx.x; i < 65536 / 4; i += 128) reinterpret_cast<uint32_t*>(sm)[i] = 0x3c003c00u;
  if (warp == 0) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t t = tm;
  constexpr uint32_t I64 = idesc_bf16(128, 64), I48 = idesc_bf16(128, 48, 0, 1);
  const uint64_t da = desc_k128(su), db = desc_k128(su + 16384), dm = desc_mn128(su + 32768, 8192);
  if (warp == 0) {
    uint32_t ph = 0;
    for (int pat = 0; pat < 6; ++pat) {
      unsigned long long best = ~0ull;
      for (int r = 0; r < reps; ++r) {
        const unsigned long long t0 = clock64();
        if (elect_one()) {
          if (pat == 0) umma_ss(t, da, db, I64, 0);                                             // 1 MMA
          if (pat == 1) for (int k = 0; k < 3; ++k) umma_ss(t, da + 2 * k, db + 2 * k, I64, k > 0);   // 3 chained (one S)
          if (pat == 2) { for (int k = 0; k < 3; ++k) umma_ss(t, da + 2 * k, db + 2 * k, I64, k > 0);
                          for (int k = 0; k < 3; ++k) umma_ss(t + 64, da + 2 * k, db + 2 * k, I64, k > 0); }   // S then dP
          if (pat == 3) for (int k = 0; k < 3; ++k) { umma_ss(t, da + 2 * k, db + 2 * k, I64, k > 0);
                                                      umma_ss(t + 64, da + 2 * k, db + 2 * k, I64, k > 0); }   // interleaved
          if (pat == 4) for (int k = 0; k < 4; ++k) umma_ts(t + 320, t + 256 + 8 * k, dm + (2048 >> 4) * k, I48, k > 0);  // one dQ (TS, 4 chained)
          if (pat == 5) for (int k = 0; k < 6; ++k) umma_ss(t + 64 * k, da, db, I64, 0);        // 6 independent
          tc_commit(&bar);
        }
        __syncwarp();
        mbar_wait(&bar, ph); ph ^= 1;
        const unsigned long long d = clock64() - t0;
        if (d < best) best = d;
      }
      if (threadIdx.x == 0) out[pat] = best;
    }
  }
  tc_fence_before(); __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(t, 512); }
}
