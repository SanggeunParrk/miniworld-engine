// tsx_dw.cu -- the Transition backward at SMALL M, third kernel: the weight gradients and dgamma / dbeta (-DDIM=<D> -DHID=<H>
// -DNT=<64 | 128 | 192>), sm_100a, bf16 operands, fp32 results, a programmatic dependent launch after tsx_dxn.cu (or tsx_gdx.cu).
//   dWs  [D][H]   = dy^T h      over the M rows: (D / 128) x (H / NT) output tiles
//   dWab [2H][D]  = dab^T xn    (dWa rows, then dWb rows): (2 H / 128) x (D / NT) output tiles
// one CTA per output tile, K = M through a TMA ring; both operands are the row-major activations read MN-major (no transposed copies).
// The host picks the widest NT whose CTAs fit beside the d_xn clusters in one wave (54 at [1, 768, 384] n = 2 with NT = 128, 72 at
// n = 4 with NT = 192: a CTA that waits for a d_xn SM to free starts its whole K loop after d_xn ends). The first 2 D / 128 CTAs also
// add the d_xn kernel's per-tile dgamma | dbeta partials in tile order (deterministic).
// Every operand is a forward save, the backward input or the gate's output -- complete before the d_xn kernel this launch follows
// passed its own griddepcontrol.wait -- so the products and the tile stores run while d_xn finishes; only the partial sums wait for it.
// After tsx_gdx.cu (use_flags: the gate and d_xn in one launch) [dA | dB] is written by the launch this one runs beside: the dWab
// CTAs read a tile's rows only after its flag turns 1 (gdx clears the flags before it lets this launch start and sets a tile's flag
// with a gpu-scope release once the tile's cluster has stored its rows).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef NT
#define NT 64
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128;
constexpr int T1 = (D_ / 128) * (H_ / NT);                     // dWs tiles; the dWab tiles follow
static_assert(D_ % 128 == 0 && D_ % NT == 0 && H_ % NT == 0 && (NT == 64 || NT == 128 || NT == 192), "whole output tiles");
constexpr int ABT = 128 * 128, BBT = NT * 128;                 // A: [64 k][128 m'] (two 64-wide MN blocks), B: [64 k][NT n']
constexpr int STG = NT == 64 ? 6 : 5;
constexpr int TCOLS = NT == 192 ? 256 : NT;                    // tensor-memory allocation (a power of two)
constexpr int O_A = 0, O_B = STG * ABT, O_BAR = O_B + STG * BBT, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(128, NT, 1, 1);

struct Bars {
  uint64_t full[STG], empty[STG], acc_full;
  uint32_t tmem;
};

// maps (box 64 inner x 64 rows, MN-major operands): dy [M][D], h [M][H], dab [M][2 H], xn [M][D].
// dws [D][H], dwab [2 H][D] fp32 (obf16: bf16, for bf16 parameters), dgb [2 D] fp32; part [tiles][2 D] fp32 from tsx_dxn.cu.
extern "C" __global__ void __launch_bounds__(256, 1)
transition_sx_dw(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mh,
                 const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mxn, float* __restrict__ dws,
                 float* __restrict__ dwab, const float* __restrict__ part, float* __restrict__ dgb, int tiles, int obf16,
                 const int* __restrict__ flags, int use_flags) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool p1 = (int)blockIdx.x < T1;
  const int t = p1 ? (int)blockIdx.x : (int)blockIdx.x - T1;
  const int nN = p1 ? H_ / NT : D_ / NT;
  const int m0 = (t / nN) * 128, n0 = (t % nN) * NT;            // output rows (d or hidden k2) and columns
  const CUtensorMap* ma = p1 ? &mdy : &mdab;
  const CUtensorMap* mb = p1 ? &mh : &mxn;
  float* outp = p1 ? dws : dwab;
  const int ldo = p1 ? H_ : D_;
  const int nkb = tiles * 2;                                   // 64-row K-blocks over the M rows

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.acc_full, 1);
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(ma); prefetch_map(mb);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      const bool wait_flags = !p1 && use_flags;
      for (int kb = 0; kb < nkb; ++kb) {
        const int s = kb % STG;
        if (kb >= STG) mbar_wait(&B.empty[s], ((kb / STG) - 1) & 1);
        mbar_expect_tx(&B.full[s], ABT + BBT);
#pragma unroll
        for (int jb = 0; jb < NT / 64; ++jb) tma_load_2d(su + O_B + s * BBT + jb * 8192, mb, &B.full[s], n0 + jb * 64, kb * 64);
        if (wait_flags && !(kb & 1)) {                         // the tile's [dA | dB] rows (gdx's release) -> these TMA reads
          int f;
          for (;;) {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(f) : "l"(flags + (kb >> 1)) : "memory");
            if (f) break;
            __nanosleep(128);                                  // a poll every ~0.1 us: the gdx CTAs' loads share the L2
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        tma_load_2d(su + O_A + s * ABT, ma, &B.full[s], m0, kb * 64);
        tma_load_2d(su + O_A + s * ABT + 8192, ma, &B.full[s], m0 + 64, kb * 64);
      }
    }
  } else if (warp == 1) {
    for (int kb = 0; kb < nkb; ++kb) {
      const int s = kb % STG;
      mbar_wait(&B.full[s], (kb / STG) & 1);
      tc_fence_after();
      const uint64_t da = desc_mn128(su + O_A + s * ABT, 8192), db = desc_mn128(su + O_B + s * BBT, 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ss(tmem, da + (uint64_t)((ks * 2048) >> 4), db + (uint64_t)((ks * 2048) >> 4), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.empty[s]);
        if (kb == nkb - 1) tc_commit(&B.acc_full);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ---------------------------------------------------------------------------------------- the fp32 tile: one row per thread
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    const size_t ro0 = (size_t)m0 * ldo + n0;                  // the tile's first element
    mbar_wait(&B.acc_full, 0);
    tc_fence_after();
#pragma unroll 1
    for (int c0 = 0; c0 < NT; c0 += 32) {
      uint32_t v[32];
      tmem_ld32(trow + c0, v);
      tmem_wait_ld();
      if (obf16) {
        uint32_t p[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) p[i] = pack_bf16(__uint_as_float(v[2 * i]), __uint_as_float(v[2 * i + 1]));
        store_t(reinterpret_cast<__nv_bfloat16*>(outp) + ro0 + c0 + store_t_off<16>(r, lane, ldo, 2), ldo, p, lane);
      } else {
        store_t(outp + ro0 + c0 + store_t_off<32>(r, lane, ldo, 4), ldo, v, lane);
      }
    }
    tc_fence_before();
    pdl_wait();                                                // the d_xn kernel's dgamma | dbeta partials
    pdl_launch();
    if ((int)blockIdx.x < 2 * D_ / 128) {                      // dgamma | dbeta = sum over the tiles' partials, in order
      const int c = blockIdx.x * 128 + (tid - 128);
      float s = 0.f;
#pragma unroll 4
      for (int i = 0; i < tiles; ++i) s += part[(size_t)i * 2 * D_ + c];
      dgb[c] = s;
    }
  }
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
