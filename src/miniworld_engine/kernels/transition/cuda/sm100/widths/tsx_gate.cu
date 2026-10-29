// tsx_gate.cu -- the Transition backward at SMALL M, first kernel: the SwiGLU backward (-DDIM=<D> -DHID=<H>), sm_100a, bf16.
// One CTA per (128-row tile, 64-unit hidden chunk j): tiles x H / 64 CTAs (72 / 144 at [1, 768, 384], n = 2 / 4).
//   dh_j = bf16(dy Ws[:, j])   M 128, N 64, K D: A = the dy tile (K-major), B = Ws in its own [D][H] layout (MN-major: no Ws^T copy)
//   [dA | dB]_j from dh_j and the forward's saved a, b (tbwd_d384.cu's gate_pair, the kit sigmoid) -> dab [M][2 H] (dA | dB)
// The d_xn product + LayerNorm backward (tsx_dxn.cu) follows as a programmatic dependent launch; the weight gradients are cuBLAS
// GEMMs over dab, xn, dy and h. Ws K-blocks are loaded before griddepcontrol.wait (the forward's cast, long complete).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, CH = 64, NKB = D_ / 64, NCHK = H_ / CH;
constexpr int KBT = ROWS * 128, WKB = 64 * 128;                // dy K-block [128][64] 16 KB, Ws block [64 k rows][64 units] 8 KB
constexpr int O_DY = 0, O_W = NKB * KBT, O_BAR = O_W + NKB * WKB, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(128, CH, 0, 1);

struct Bars {
  uint64_t w_full, dy_full, mma_done;
  uint32_t tmem;
};

DEVI void gate_pair(uint32_t dh0, uint32_t dh1, uint32_t ap, uint32_t bp, uint32_t& dap, uint32_t& dbp) {
  const uint32_t gp = pack_bf16(__uint_as_float(dh0), __uint_as_float(dh1));
  const f2 G = mk2(bf16lo(gp), bf16hi(gp)), A = mk2(bf16lo(ap), bf16hi(ap)), Bv = mk2(bf16lo(bp), bf16hi(bp));
  const f2 S = mk2(sigmoid_kit(lo2(A)), sigmoid_kit(hi2(A)));
  const f2 L = mul2(A, S);
  const f2 DB = mul2(G, L);
  const f2 U = fma2(L, fma2(S, mk2(-1.f, -1.f), mk2(1.f, 1.f)), S);
  const f2 DA = mul2(mul2(G, Bv), U);
  dap = pack_bf16(lo2(DA), hi2(DA)); dbp = pack_bf16(lo2(DB), hi2(DB));
}

// maps: dy [M][D] (box 64 x 128), ws [D][H] (box 64 units x 64 rows, MN-major operand). a / b [M][H], dab [M][2 H] bf16.
extern "C" __global__ void __launch_bounds__(384, 1)
transition_sx_gate(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mws,
                   const __nv_bfloat16* __restrict__ as_, const __nv_bfloat16* __restrict__ bs, __nv_bfloat16* __restrict__ dab) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x / NCHK, j = blockIdx.x % NCHK, row0 = tile * ROWS, h0 = j * CH;

  if (tid == 0) {
    mbar_init(&B.w_full, 1); mbar_init(&B.dy_full, 1); mbar_init(&B.mma_done, 1);
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(&mdy); prefetch_map(&mws);
    mbar_expect_tx(&B.w_full, NKB * WKB);
    for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_W + kb * WKB, &mws, &B.w_full, h0, kb * 64);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), CH); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_wait();
  pdl_launch();

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.dy_full, NKB * KBT);
      for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_DY + kb * KBT, &mdy, &B.dy_full, kb * 64, row0);
    }
  } else if (warp == 1) {
    mbar_wait(&B.w_full, 0);
    mbar_wait(&B.dy_full, 0);
    tc_fence_after();
    if (elect_one()) {
#pragma unroll
      for (int kb = 0; kb < NKB; ++kb) {
        const uint64_t da = desc_k128(su + O_DY + kb * KBT), db = desc_mn128(su + O_W + kb * WKB, WKB);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ss(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)((ks * 2048) >> 4), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
      }
      tc_commit(&B.mma_done);
    }
    __syncwarp();
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ gate: one row per thread, warps 4-7
    // units 0..31 of the chunk, 8-11 units 32..63
    const int r = (warp & 3) * 32 + lane, g = (warp - 4) >> 2;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    const size_t off = (size_t)(row0 + r) * H_ + h0 + g * 32;
    uint4 av[4], bv[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) { av[q] = ldg128(as_ + off + q * 8); bv[q] = ldg128(bs + off + q * 8); }
    mbar_wait(&B.mma_done, 0);
    tc_fence_after();
    uint32_t v[32];
    tmem_ld32(trow + g * 32, v);
    tmem_wait_ld();
    tc_fence_before();
    uint32_t da[16], db[16];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t a4[4] = {av[q].x, av[q].y, av[q].z, av[q].w}, b4[4] = {bv[q].x, bv[q].y, bv[q].z, bv[q].w};
#pragma unroll
      for (int k = 0; k < 4; ++k) gate_pair(v[8 * q + 2 * k], v[8 * q + 2 * k + 1], a4[k], b4[k], da[4 * q + k], db[4 * q + k]);
    }
    __nv_bfloat16* pa = dab + (size_t)(row0 + r) * (2 * H_) + h0 + g * 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      stg128(pa + q * 8, make_uint4(da[4 * q], da[4 * q + 1], da[4 * q + 2], da[4 * q + 3]));
      stg128(pa + H_ + q * 8, make_uint4(db[4 * q], db[4 * q + 1], db[4 * q + 2], db[4 * q + 3]));
    }
  }
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, CH); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
