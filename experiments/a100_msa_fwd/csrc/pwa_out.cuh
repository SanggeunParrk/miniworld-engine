// pwa_out.cuh -- PWA split path, K_O: out[t, :] = msa[t, :] + u[t, :] . Wo^T   (u [T, 256] bf16 from K_C, Wo [64, 256]), one rounding.
// Memory-bound (512 B of u + 128 B in, 128 B out per token). Persistent CTA (8 warps), Wo resident in smem (64 rows x 512 B), 128-token
// u tiles double-buffered through cp.async (the next tile streams while this one multiplies). A warp owns 16 tokens x 64 outputs.
// Replaces torch.addmm, which first copies msa into the output (a 100 MB DtoD pass) before its beta = 1 GEMM.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaOutParams {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* u;     // [T, 256]
  const __nv_bfloat16* wo;    // [64, 256]
  __nv_bfloat16* out;         // [T, 64]
  int T;
};

constexpr int PWA_O_NTHR = 256, PWA_O_TT = 128;
constexpr int PWA_O_WO = 0, PWA_O_U = 64 * 512, PWA_O_UT = PWA_O_TT * 512;
constexpr int PWA_O_SMEM = PWA_O_U + 2 * PWA_O_UT;       // 32 KB + 2 x 64 KB

DEVI uint32_t swz512o(int row, int gr) { return row * 512 + ((gr ^ (row & 7)) << 4); }

__global__ void __launch_bounds__(PWA_O_NTHR, 1) pwa_out_kernel(PwaOutParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4;
  const int ntiles = p.T / PWA_O_TT;

  for (int idx = tid; idx < 64 * 32; idx += PWA_O_NTHR) {
    const int row = idx >> 5, gr = idx & 31;
    cp_async16(sb + PWA_O_WO + swz512o(row, gr), p.wo + row * 256 + gr * 8);
  }
  // u tile copy slots: granule (tid & 31) of rows (tid >> 5) + 8 k, k = 0..15
  const uint32_t u_dst = swz512o(tid >> 5, tid & 31);
  auto issue = [&](int tile, int buf) {
    const __nv_bfloat16* src = p.u + ((size_t)tile * PWA_O_TT + (tid >> 5)) * 256 + (tid & 31) * 8;
    const uint32_t dst = sb + PWA_O_U + buf * PWA_O_UT + u_dst;
#pragma unroll
    for (int k = 0; k < 16; ++k) cp_async16(dst + k * 8 * 512, src + (size_t)k * 8 * 256);
  };
  int tile = blockIdx.x, buf = 0;
  if (tile < ntiles) issue(tile, 0);
  cp_async_commit();

  const uint32_t a_off = swz512o(16 * warp + (lane & 15), l4);                               // ^ (kc << 5) : granule 2 kc + l4
  const uint32_t b_off = PWA_O_WO + swz512o((lane & 7) + (l4 << 3), (lane >> 3) & 1);        // + 16 np rows, ^ (kc << 5)
  for (; tile < ntiles; tile += gridDim.x, buf ^= 1) {
    if (tile + gridDim.x < ntiles) issue(tile + gridDim.x, buf ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint32_t ut = sb + PWA_O_U + buf * PWA_O_UT;
    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
    for (int kc = 0; kc < 16; ++kc) {
      uint32_t af[4];
      ldsm_x4(af, ut + (a_off ^ (kc << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4(bf, sb + ((b_off + 16 * np * 512) ^ (kc << 5)));
        mma16816(acc[2 * np], af, bf[0], bf[1]);
        mma16816(acc[2 * np + 1], af, bf[2], bf[3]);
      }
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t base = ((size_t)tile * PWA_O_TT + 16 * warp + g + 8 * hh) * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 8 * nt + 2 * q;
        const uint32_t r = __ldg(reinterpret_cast<const unsigned int*>(p.msa + base + c));
        stg32(p.out + base + c, pack_bf16(bf16lo(r) + acc[nt][2 * hh], bf16hi(r) + acc[nt][2 * hh + 1]));
      }
    }
    __syncthreads();      // every warp is done with this buffer before the next iteration's issue() refills it
  }
  cp_async_wait<0>();
}

}  // namespace a100
