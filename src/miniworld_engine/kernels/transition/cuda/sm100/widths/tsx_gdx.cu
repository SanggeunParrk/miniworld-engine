// tsx_gdx.cu -- the Transition backward at SMALL M, first kernel: gate + d_xn + LayerNorm backward + residual + dgamma / dbeta
// partials in one launch (-DDIM=<D> -DHID=<H>), sm_100a, bf16; round sx3: tsx_gate.cu and tsx_dxn.cu (KS = 2) on one cluster per
// 128-row tile, so the tile's [dA | dB] is consumed without a kernel boundary (griddepcontrol.wait of a second launch, its clusters'
// residency behind the gate CTAs and its prologue).
// One cluster of 2 D / 64 CTAs per tile (12 at D = 384); CTA rank c:
//   gate:  dh = bf16(dy Ws[:, U_c])   M 128, N 64 GPC, K D, U_c = the CTA's GPC 64-unit hidden chunks (GPC = H / 64 / 12: 1 or 2)
//          A = the dy tile (K-major), B = Ws in its own [D][H] layout (MN-major); [dA | dB]_U from dh and the saved a, b (the
//          tsx_gate.cu arithmetic) -> dab [M][2 H] in global memory (the weight-gradient kernel's operand and the d_xn A operand).
//   barrier.cluster (release / acquire, after fence.proxy.async): the tile's [dA | dB] is complete; rank 0 then publishes it for the
//          weight-gradient kernel (a per-tile flag, gpu-scope release), which runs beside this one.
//   d_xn:  CTA (column block nb = c % 6, K half ks = c / 6) exactly as tsx_dxn.cu with KS = 2 (dA Wa / dB Wb halves summed in
//          distributed shared memory, the LayerNorm backward's row sums across the cluster, dgamma / dbeta partials per tile).
// The d_xn ring (A = [dA | dB] K-block 16 KB + W K-block 8 KB per stage) has NDED stages of its own, whose weight blocks load before
// griddepcontrol.wait, and NGR stages over the gate's shared memory (dy and the Ws chunk), whose weight blocks load once the gate
// products are complete; the A blocks load after the cluster barrier.
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
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NB = 64, NBLK = D_ / NB, KS = 2, CLS = NBLK * KS;
constexpr int NCHK = H_ / 64, GPC = NCHK / CLS, GN = 64 * GPC;  // gate: 64-unit chunks, chunks per CTA, units per CTA
static_assert(NCHK % CLS == 0 && (GPC == 1 || GPC == 2) && CLS <= 16, "whole gate chunks per CTA, one cluster per tile");
constexpr int NKD = D_ / 64, HKB = H_ / 64, NKX = 2 * HKB / KS;  // dy K-blocks, hidden K-blocks, d_xn K-blocks per CTA
constexpr int OW = NB / KS;                                     // owned output columns (32)
constexpr int DYB = ROWS * 128, WSB = GN * 128;                // dy K-block 16 KB, Ws K-block [64 k][GN units] (GN / 64 boxes of 8 KB)
constexpr int KBT = ROWS * 128, WBT = 64 * 128, STB = KBT + WBT;  // d_xn stage: [dA | dB] K-block 16 KB + weight block 8 KB
constexpr int O_WS = 0, O_DY = NKD * WSB, O_GEND = O_DY + NKD * DYB;
constexpr int O_RED = O_GEND;                                  // partner's partial of the owned columns: [128][32] fp32, 128-B rows
                                                               // with the 16-B chunks swizzled (sw128): conflict-free row accesses
constexpr int O_RS = O_RED + ROWS * 32 * 4;                    // row sums of every CTA: [CLS source ranks][128 rows] float2
constexpr int O_CP = O_RS + CLS * ROWS * 8;                    // column partials [4 quadrants][2][OW] fp32
constexpr int O_GM = O_CP + 4 * 2 * OW * 4;                    // gamma of the owned columns [OW] fp32
constexpr int O_RH = O_GM + OW * 4;                            // the two column halves' row sums [2][128] float2
constexpr int O_DED = (O_RH + 2 * ROWS * 8 + 1023) / 1024 * 1024;
constexpr int SMAX = 232448 - 512;
constexpr int NDED_MAX = (SMAX - O_DED) / STB, NGR_MAX = O_GEND / STB;
constexpr int NDED = NDED_MAX < NKX ? NDED_MAX : NKX;
constexpr int NGR = NGR_MAX < NKX - NDED ? NGR_MAX : NKX - NDED;
constexpr int STG = NDED + NGR;
constexpr int O_BAR = O_DED + NDED * STB, SMEM_BYTES = O_BAR + 512;
static_assert(STG >= 2 && SMEM_BYTES <= 232448 && STB % 1024 == 0, "shared memory budget");
constexpr int TC = GN + NB <= 128 ? 128 : 256;                 // tensor memory: dh [0, GN), the d_xn accumulator [GN, GN + 64)
constexpr uint32_t IDESC_G = idesc_bf16(128, GN, 0, 1), IDESC_X = idesc_bf16(128, NB, 0, 1);

struct Bars {
  uint64_t w_full, dy_full[NKD], mma_done, full[STG], empty[STG], acc_full, red_full, rs_full;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barrier block");

DEVI uint32_t stage_addr(uint32_t su, int s) { return su + (s < NDED ? O_DED + s * STB : (s - NDED) * STB); }

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

// warp reduce-scatter: lane l returns the sum over the warp of v[l]
DEVI float reduce_scatter16(float (&v)[16], int lane) {       // lane l returns the warp's sum of v[l & 15]
#pragma unroll
  for (int k = 0; k < 16; ++k) v[k] += __shfl_xor_sync(0xffffffffu, v[k], 16);
#pragma unroll
  for (int off = 8; off; off >>= 1) {
    const bool up = (lane & off) != 0;
#pragma unroll
    for (int k = 0; k < off; ++k) {
      const float lo = v[k], hi = v[k + off];
      v[k] = up ? hi : lo; v[k + off] = up ? lo : hi;
    }
#pragma unroll
    for (int k = 0; k < off; ++k) v[k] += __shfl_xor_sync(0xffffffffu, v[k + off], off);
  }
  return v[0];
}
DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.aligned;" ::: "memory"); }   // release
DEVI void cluster_arrive_rx() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }       // acquire
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI uint32_t cl_addr(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
// maps: dy [M][D] (box 64 x 128), ws [D][H] (box 64 units x 64 rows, MN-major), dab [M][2 H] (box 64 x 128), wa / wb [H][D] (box
// 64 cols x 64 rows, MN-major). a / b [M][H] bf16, dab (the same tensor as mdab) written here. part [tiles][2 D] fp32 dgamma | dbeta
// partials; flags [tiles] (uninitialised): cleared before griddepcontrol.launch_dependents, set to 1 (gpu-scope release) once the
// tile's [dA | dB] is in global memory.
extern "C" __global__ void __launch_bounds__(384, 1)
transition_sx_gdx(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mws,
                  const __nv_bfloat16* __restrict__ as_, const __nv_bfloat16* __restrict__ bs, __nv_bfloat16* __restrict__ dab,
                  const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mwa,
                  const __grid_constant__ CUtensorMap mwb, const __nv_bfloat16* __restrict__ xg,
                  const __nv_bfloat16* __restrict__ dyg, const float* __restrict__ rstd, const float* __restrict__ c1g,
                  const float* __restrict__ gamma, __nv_bfloat16* __restrict__ dx, float* __restrict__ part, int* __restrict__ flags) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cluster_rank(), nb = rank % NBLK, ks = rank / NBLK, tile = blockIdx.x / CLS, row0 = tile * ROWS;
  const int h0 = rank * GN;                                    // the gate's hidden units [h0, h0 + GN)
  const int own0 = nb * NB + ks * OW;                          // owned output columns [own0, own0 + OW)
  const int kb0 = ks * NKX;                                    // first global d_xn K-block (of 2 H / 64: Wa's, then Wb's)

  auto load_w = [&](int kb) {                                  // d_xn weight block of K-block kb into its stage
    const int g = kb0 + kb, s = kb % STG;
    tma_load_2d(stage_addr(su, s) + KBT, g < HKB ? &mwa : &mwb, &B.full[s], nb * NB, (g % HKB) * 64);
  };
  if (tid == 0) {
    mbar_init(&B.w_full, 1); mbar_init(&B.mma_done, 1);
    for (int kb = 0; kb < NKD; ++kb) mbar_init(&B.dy_full[kb], 1);
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.acc_full, 1); mbar_init(&B.red_full, 1); mbar_init(&B.rs_full, 1);
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(&mdy); prefetch_map(&mws); prefetch_map(&mdab); prefetch_map(&mwa); prefetch_map(&mwb);
    // parameters only (no dependence on the previous launch): the gate's Ws chunk and the dedicated stages' weight blocks
    mbar_expect_tx(&B.w_full, NKD * WSB);
    for (int kb = 0; kb < NKD; ++kb)
#pragma unroll
      for (int gb = 0; gb < GPC; ++gb) tma_load_2d(su + O_WS + kb * WSB + gb * 8192, &mws, &B.w_full, h0 + gb * 64, kb * 64);
    for (int kb = 0; kb < NDED; ++kb) { mbar_expect_tx(&B.full[kb], STB); load_w(kb); }
    mbar_expect_tx(&B.red_full, ROWS * OW * 4);                // the partner's partial (st.async, any time after the setup sync)
    mbar_expect_tx(&B.rs_full, CLS * ROWS * 8);                // every CTA's row sums (this one's included)
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TC); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();                                      // every CTA of the cluster runs before any distributed-shared access
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_wait();                                                  // dy (and the forward's a, b) complete and visible
  if (tid == 0 && rank == 0) {                                 // clear this tile's flag before the weight-gradient kernel may start
    flags[tile] = 0;
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  if (tid >= 32 && tid < 32 + OW) reinterpret_cast<float*>(sm + O_GM)[tid - 32] = gamma[own0 + tid - 32];
  __syncthreads();                                             // (the flag is cleared before this CTA's launch_dependents below)

  if (warp == 0) {
    if (lane == 0) {
      for (int kb = 0; kb < NKD; ++kb) {                        // a barrier per K-block: the products start as dy lands
        mbar_expect_tx(&B.dy_full[kb], DYB);
        tma_load_2d(su + O_DY + kb * DYB, &mdy, &B.dy_full[kb], kb * 64, row0);
      }
      mbar_wait(&B.mma_done, 0);                               // the gate products have read dy and Ws: their stages are free
      pdl_launch();                                            // the weight-gradient kernel may start: not earlier, its dWs loads
                                                               // would compete with this tile's dy (its dWab waits for the flags)
      for (int kb = NDED; kb < STG; ++kb) { mbar_expect_tx(&B.full[kb], STB); load_w(kb); }
    }
    __syncwarp();
    cluster_arrive_rx();
    cluster_wait();                                            // the tile's [dA | dB] is complete (every CTA's gate)
    if (lane == 0) {
      fence_proxy_async_global();                              // generic-proxy stores of the peers -> this CTA's TMA reads
      for (int kb = 0; kb < NKX; ++kb) {
        const int s = kb % STG;
        if (kb >= STG) {
          mbar_wait(&B.empty[s], ((kb / STG) - 1) & 1);
          mbar_expect_tx(&B.full[s], STB);
          load_w(kb);
        }
        tma_load_2d(stage_addr(su, s), &mdab, &B.full[s], (kb0 + kb) * 64, row0);
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------ MMA issue (converged warp)
    mbar_wait(&B.w_full, 0);
#pragma unroll 1
    for (int kb = 0; kb < NKD; ++kb) {
      mbar_wait(&B.dy_full[kb], 0);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_DY + kb * DYB), db = desc_mn128(su + O_WS + kb * WSB, 8192);
      if (elect_one()) {
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4)
          umma_ss(tmem, da + (uint64_t)(k4 * 2), db + (uint64_t)((k4 * 2048) >> 4), IDESC_G, (kb > 0 || k4 > 0) ? 1u : 0u);
        if (kb == NKD - 1) tc_commit(&B.mma_done);
      }
      __syncwarp();
    }
    cluster_arrive_rx();                                       // (no data of its own; must not hold the barrier behind d_xn)
#pragma unroll 1
    for (int kb = 0; kb < NKX; ++kb) {
      const int s = kb % STG;
      mbar_wait(&B.full[s], (kb / STG) & 1);
      tc_fence_after();
      const uint32_t sa = stage_addr(su, s);
      const uint64_t da = desc_k128(sa), db = desc_mn128(sa + KBT, WBT);
      if (elect_one()) {
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4)
          umma_ss(tmem + GN, da + (uint64_t)(k4 * 2), db + (uint64_t)((k4 * 2048) >> 4), IDESC_X, (kb > 0 || k4 > 0) ? 1u : 0u);
        tc_commit(&B.empty[s]);
        if (kb == NKX - 1) tc_commit(&B.acc_full);
      }
      __syncwarp();
    }
    cluster_wait();
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ gate: one row per thread, warps 4-7
    // units 0..31 of each 64-unit chunk, warps 8-11 units 32..63
    const int r = (warp & 3) * 32 + lane, g = (warp - 4) >> 2;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    uint4 av[GPC][4], bv[GPC][4];
#pragma unroll
    for (int c = 0; c < GPC; ++c) {
      const size_t off = (size_t)(row0 + r) * H_ + h0 + c * 64 + g * 32;
#pragma unroll
      for (int q = 0; q < 4; ++q) { av[c][q] = ldg128(as_ + off + q * 8); bv[c][q] = ldg128(bs + off + q * 8); }
    }
    mbar_wait(&B.mma_done, 0);
    tc_fence_after();
#pragma unroll
    for (int c = 0; c < GPC; ++c) {
      uint32_t v[32];
      tmem_ld32(trow + c * 64 + g * 32, v);
      tmem_wait_ld();
      uint32_t da[16], db[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t a4[4] = {av[c][q].x, av[c][q].y, av[c][q].z, av[c][q].w}, b4[4] = {bv[c][q].x, bv[c][q].y, bv[c][q].z, bv[c][q].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) gate_pair(v[8 * q + 2 * k], v[8 * q + 2 * k + 1], a4[k], b4[k], da[4 * q + k], db[4 * q + k]);
      }
      __nv_bfloat16* pa = dab + (size_t)row0 * (2 * H_) + h0 + c * 64 + g * 32 + store_t_off<16>(r, lane, 2 * H_, 2);
      store_t(pa, 2 * H_, da, lane);
      store_t(pa + H_, 2 * H_, db, lane);
    }
    tc_fence_before();
    fence_proxy_async_global();                                // these stores -> the cluster's (and tsx_dw.cu's) TMA reads
    cluster_arrive();                                          // release: this CTA's [dA | dB] rows are written
  } else {
    cluster_arrive_rx();
    if (warp == 3) {
      cluster_wait();
      if (rank == 0 && lane == 0) {                            // publish the tile for the weight-gradient kernel
        asm volatile("fence.acq_rel.gpu;" ::: "memory");     // cumulative over the peers' stores acquired by the cluster barrier
        asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" :: "l"(flags + tile), "r"(1) : "memory");
      }
      __syncwarp();
    }
  }
  if (warp >= 4 || warp == 2) cluster_wait();                  // (warps 0, 1, 3 waited above)

  // ------------------------------------------------------------------------------------------ d_xn epilogue: one row per thread,
  // warps 4-7 owned columns 0..15, warps 8-11 columns 16..31 (two warps per scheduler: this tail is latency bound)
  const bool epi = warp >= 4;
  const int r = (warp & 3) * 32 + lane, g = (warp - 4) >> 2, c0 = own0 + g * 16;
  const size_t grow = (size_t)(row0 + r);
  const uint32_t trow = tmem + GN + ((uint32_t)((warp & 3) * 32) << 16);
  float rs = 0.f, mean = 0.f;
  uint32_t np[8];                                              // d_xn in bf16, packed pairs
  uint4 xv[2], dyv[2];
  uint32_t v[16];
  if (epi) {
#pragma unroll
    for (int q = 0; q < 2; ++q) { xv[q] = ldg128(xg + grow * D_ + c0 + q * 8); dyv[q] = ldg128(dyg + grow * D_ + c0 + q * 8); }
    rs = rstd[grow];
    mean = c1g[grow] / rs;
    mbar_wait(&B.acc_full, 0);
    tc_fence_after();
    tmem_ld16(trow + (1 - ks) * OW + g * 16, v);               // the partner's half of this block -> its shared memory
    tmem_wait_ld();
    const uint32_t partner = (uint32_t)(ks ? rank - NBLK : rank + NBLK);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      st_async_v4(su + O_RED + sw128(r, g * 4 + i), smem_u32(&B.red_full), partner, v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
    tmem_ld16(trow + ks * OW + g * 16, v);
    tmem_wait_ld();
    tc_fence_before();
  }
  float cg[16], cb[16];
  if (epi) {
    mbar_wait(&B.red_full, 0);                                 // the partner's partial is here
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 t = lds128(su + O_RED + sw128(r, g * 4 + i));
      const uint32_t p4[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {                            // dA Wa + dB Wb, in that order
        const float mine = __uint_as_float(v[4 * i + e]), other = __uint_as_float(p4[e]);
        v[4 * i + e] = __float_as_uint(ks == 0 ? mine + other : other + mine);
      }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) np[k] = pack_bf16(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
    float pa = 0.f, pb = 0.f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const uint32_t x4[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w};
      const float4 ga = *reinterpret_cast<const float4*>(sm + O_GM + g * 64 + q * 32);
      const float4 gc = *reinterpret_cast<const float4*>(sm + O_GM + g * 64 + q * 32 + 16);
      const float g8[8] = {ga.x, ga.y, ga.z, ga.w, gc.x, gc.y, gc.z, gc.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 8 * q + 2 * e;
        const float n0 = bf16lo(np[i / 2]), n1 = bf16hi(np[i / 2]);
        const float x0 = (bf16lo(x4[e]) - mean) * rs, x1 = (bf16hi(x4[e]) - mean) * rs;
        const float w0 = g8[2 * e] * n0, w1 = g8[2 * e + 1] * n1;
        pa += x0 * w0 + x1 * w1; pb += w0 + w1;
        cg[i] = n0 * x0; cg[i + 1] = n1 * x1; cb[i] = n0; cb[i + 1] = n1;
      }
    }
    reinterpret_cast<float2*>(sm + O_RH)[g * ROWS + r] = make_float2(pa, pb);
    named_bar_sync(1, 256);
    const float2 h0v = reinterpret_cast<const float2*>(sm + O_RH)[r], h1v = reinterpret_cast<const float2*>(sm + O_RH)[ROWS + r];
    pa = h0v.x + h1v.x; pb = h0v.y + h1v.y;                    // this CTA's 32 columns of the row, halves in order
#pragma unroll 1
    for (int q = 0; q < CLS / 2; ++q)                          // to every CTA of the cluster (this one too), slot [rank][r]
      st_async_v2f(su + O_RS + (uint32_t)((rank * ROWS + r) * 8), smem_u32(&B.rs_full), (uint32_t)((rank + g * (CLS / 2) + q) % CLS), pa, pb);
    // dgamma | dbeta column partials: sum over the warp's 32 rows (lane l -> column g 16 + (l & 15)), then the quadrants
    const float sg = reduce_scatter16(cg, lane), sb = reduce_scatter16(cb, lane);
    float* cp = reinterpret_cast<float*>(sm + O_CP);
    if (lane < 16) {
      cp[((warp & 3) * 2 + 0) * OW + g * 16 + lane] = sg;
      cp[((warp & 3) * 2 + 1) * OW + g * 16 + lane] = sb;
    }
    named_bar_sync(1, 256);
    if (tid < 128 + 2 * OW) {                                  // this tile's partial of the owned columns
      const int k = tid - 128, which = k / OW, c = k % OW;
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) s += cp[(q * 2 + which) * OW + c];
      part[(size_t)tile * 2 * D_ + which * D_ + own0 + c] = s;
    }
  }
  float ca = 0.f, cbv = 0.f;
  if (epi) {
    mbar_wait(&B.rs_full, 0);                                  // every CTA's row sums are in this CTA's shared memory
    float pa = 0.f, pb = 0.f;
#pragma unroll
    for (int q = 0; q < CLS; ++q) {                            // rank order (deterministic)
      const float2 t = lds64f(su + O_RS + (uint32_t)((q * ROWS + r) * 8));
      pa += t.x; pb += t.y;
    }
    ca = pa * (1.f / D_); cbv = pb * (1.f / D_);
    uint32_t o[8];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const uint32_t x4[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w}, d4[4] = {dyv[q].x, dyv[q].y, dyv[q].z, dyv[q].w};
      const float4 ga = *reinterpret_cast<const float4*>(sm + O_GM + g * 64 + q * 32);
      const float4 gc = *reinterpret_cast<const float4*>(sm + O_GM + g * 64 + q * 32 + 16);
      const float g8[8] = {ga.x, ga.y, ga.z, ga.w, gc.x, gc.y, gc.z, gc.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 8 * q + 2 * e;
        const float x0 = (bf16lo(x4[e]) - mean) * rs, x1 = (bf16hi(x4[e]) - mean) * rs;
        const float w0 = g8[2 * e] * bf16lo(np[i / 2]), w1 = g8[2 * e + 1] * bf16hi(np[i / 2]);
        const uint32_t t = pack_bf16((w0 - (x0 * ca + cbv)) * rs, (w1 - (x1 * ca + cbv)) * rs);
        o[4 * q + e] = pack_bf16(bf16lo(t) + bf16lo(d4[e]), bf16hi(t) + bf16hi(d4[e]));
      }
    }
    store_t(dx + (size_t)row0 * D_ + c0 + store_t_off<8>(r, lane, D_, 2), D_, o, lane);
  }
  // no cluster barrier at the end: every distributed-shared write into this CTA (partner partial, row sums) completed on its own
  // barriers above, and this CTA reads no peer's shared memory
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TC); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
