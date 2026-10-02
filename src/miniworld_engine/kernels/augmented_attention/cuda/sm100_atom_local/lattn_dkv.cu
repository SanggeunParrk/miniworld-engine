// lattn_dkv.cu — dK, dV and dbias of the AF3 windowed atom attention on tcgen05 (sm_100a), key-centric.
//
// A CTA owns 128 keys [kb, kb + 128) with kb = 128 j - 112 (so every TMEM lane quadrant's 32 keys see the same 4 query windows), one head and
// a range of samples (SP CTAs split the samples; SP = 1 writes dbias directly). Its keys see the 7 windows W0 .. W0 + 6 (W0 = 4 j - 5,
// 224 queries from q0 = 32 W0). TMEM lane = key:
//   S^T = K Q^T [128 x 224] cols 0..223,  dP^T = V dO^T cols 224..447,  dK += dS^T Q cols 448..479,  dV += P^T dO cols 480..511.
// Warp w (lane quadrant w & 3, band quarter w >> 2 = one query window) reads its 32 x 32 band cells of S^T / dP^T, computes
// P = exp2(S c + bias - LSE) and dS = P (dP - D), adds dS into the dbias cells it holds in registers (summed over the samples), and -- after
// the quadrant's four warps have read -- writes P^T and dS^T as packed bf16 over S^T (cols 0..111 / 112..223, out-of-band cells zero): the
// TMEM A operands of dV += P^T dO and dK += dS^T Q. Each warp drains its 32 x 16 piece of dK or dV through shared memory with a TMA store
// (the first chunk, whose start row is negative, stores with plain 16-byte rows). Warp 15 also issues the TMA loads (3 stages) and the MMAs.
// The per-window bias (log2 domain, key mask folded in as -inf) sits in shared memory for all the samples.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QR = 224, KR = 128, NST = 3;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QR * 64, T_K = 2 * QR * 64, T_V = T_K + KR * 64, T_L = T_V + KR * 64, T_D = T_L + QR * 4;
constexpr int STAGE = 47104;
static_assert(T_D + QR * 4 <= STAGE, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 512 * 32 * 4, O_BAR = O_O + 2 * KR * 64;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int C_S = 0, C_DP = 224, C_DK = 448, C_DV = 480, C_PP = 0, C_PS = 112;
constexpr uint32_t TX = 2 * QR * 64 + 2 * KR * 64 + 2 * QR * 4;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_dkv(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask, __nv_bfloat16* __restrict__ dkv,
                float* __restrict__ dbias,
                int ldd, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 1, *dready = bars + NST + 2;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int kb = 128 * jc - 112, W0 = 4 * jc - 5, q0 = 32 * W0;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) mbar_init(&full[i], 1);
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int r = 32 * qd + lane, key = kb + r;
  const int wq = W0 + qd + qt;                                       // this warp's band quarter = one window, queries 32 wq ..
  const bool kok = key >= 0 && key < N && (kmask == nullptr || kmask[key]);
  {
    const int rr = key - (32 * wq - KOFF);
    for (int jj = 0; jj < 32; ++jj) {
      float v = 0.f;
      if (wq >= 0 && wq < nwin) v = kok ? gbias[(((size_t)h * nwin + wq) * WQ + jj) * WK + rr] * LOG2E : -INFINITY;
      sbias[(warp * 32 + jj) * 32 + lane] = v;
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_s = idesc_bf16(128, QR, 0, 0), id_d = idesc_bf16(128, 32, 0, 1);

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_DO, &tm_do, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_L, &tm_l, &full[s], q0, h, a);
    tma_load_3d(st + T_D, &tm_d, &full[s], q0, h, a);
  };
  auto mma_s = [&](int it) {                                         // S^T, dP^T of sample it
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dk = desc_sw64(st + T_K), dq = desc_sw64(st + T_Q), dv = desc_sw64(st + T_V), ddo = desc_sw64(st + T_DO);
    if (elect_one()) {
      umma_ss(tm + C_S, dk, dq, id_s, 0);
      umma_ss(tm + C_DP, dv, ddo, id_s, 0);
      umma_ss(tm + C_S, dk + 2, dq + 2, id_s, 1);
      umma_ss(tm + C_DP, dv + 2, ddo + 2, id_s, 1);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dV += P^T dO, dK += dS^T Q (A = packed bf16 in TMEM)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t ddo = desc_sw64(st + T_DO), dq = desc_sw64(st + T_Q);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < QR / 16; ++ks) {
        const uint64_t boff = (uint64_t)(ks * 1024 >> 4);
        umma_ts(tm + C_DV, tm + C_PP + 8 * ks, ddo + boff, id_d, ks);
        umma_ts(tm + C_DK, tm + C_PS + 8 * ks, dq + boff, id_d, ks);
      }
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST - 1 && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) { mma_s(0); if (elect_one()) tc_commit(sready); __syncwarp(); }
  }

  auto drain = [&](int itd) {                                        // per warp: 32 keys x 16 columns of dK (quarters 0, 1) / dV (2, 3)
    uint32_t v[16];
    const int col = (qt < 2 ? C_DK : C_DV) + 16 * (qt & 1);
    tmem_ld16(tm + ((uint32_t)(32 * qd) << 16) + col, v);
    tmem_wait_ld();
    const float sc = qt < 2 ? QSCALE : 1.f;
    uint8_t* ob = sm + O_O + warp * 1024;                             // [32 rows][32 B], no swizzle
    if (lane == 0) tma_store_wait_read0();                            // this warp's previous store has read the tile
    __syncwarp();
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint4 o;
      o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * sc, __uint_as_float(v[8 * g + 1]) * sc);
      o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * sc, __uint_as_float(v[8 * g + 3]) * sc);
      o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * sc, __uint_as_float(v[8 * g + 5]) * sc);
      o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * sc, __uint_as_float(v[8 * g + 7]) * sc);
      *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
    }
    fence_proxy_async();
    __syncwarp();
    const int kr = kb + 32 * qd;
    if (lane == 0) {
      if (kr >= 0) {
        tma_store_3d(&tm_o, su + O_O + warp * 1024, (qt < 2 ? 128 : 256) + h * DH + 16 * (qt & 1), kr, a_lo + itd);
        tma_store_commit();
      }
    }
    if (kr < 0) {                                                    // the first chunk's first quadrants (negative start row)
      const int kk = kb + r;
      if (kk >= 0 && kk < N) {
        __nv_bfloat16* d = dkv + ((size_t)(a_lo + itd) * N + kk) * ldd + (qt < 2 ? 128 : 256) + h * DH + 16 * (qt & 1);
        *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(ob + lane * 32);
        *reinterpret_cast<uint4*>(d + 8) = *reinterpret_cast<const uint4*>(ob + lane * 32 + 16);
      }
    }
  };

  float dbacc[32];
#pragma unroll
  for (int jj = 0; jj < 32; ++jj) dbacc[jj] = 0.f;
  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int x0 = 32 * qd + 32 * qt;                                  // chunk-relative query column of the band quarter
  const float* sb = sbias + warp * 32 * 32 + lane;
  for (int it = 0; it < nit; ++it) {
    const int s = it % NST;
    const float* sl = reinterpret_cast<const float*>(sm + s * STAGE + T_L);
    const float* sd = reinterpret_cast<const float*>(sm + s * STAGE + T_D);
    mbar_wait(sready, it & 1);                                       // S / dP of this sample (and the previous sample's dV / dK) done
    tc_fence_after();
    if (ctl && lane == 0 && it + NST - 1 < nit) load(it + NST - 1);  // stage of sample it - 1 is free
    uint32_t pw[16], dw[16];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint32_t sv[16], dv[16];
      tmem_ld16(tl + C_S + x0 + 16 * c, sv);
      tmem_ld16(tl + C_DP + x0 + 16 * c, dv);
      tmem_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 16; jj += 2) {
        const int x = x0 + 16 * c + jj, b = 16 * c + jj;
        const float2 l2 = *reinterpret_cast<const float2*>(sl + x), d2 = *reinterpret_cast<const float2*>(sd + x);
        const float p0 = ex2f(fmaf(__uint_as_float(sv[jj]), QSCALE * LOG2E, sb[b * 32]) - l2.x);
        const float p1 = ex2f(fmaf(__uint_as_float(sv[jj + 1]), QSCALE * LOG2E, sb[(b + 1) * 32]) - l2.y);
        const float t0 = p0 * (__uint_as_float(dv[jj]) - d2.x), t1 = p1 * (__uint_as_float(dv[jj + 1]) - d2.y);
        dbacc[b] += t0; dbacc[b + 1] += t1;
        pw[b >> 1] = pack_bf16(p0, p1); dw[b >> 1] = pack_bf16(t0, t1);
      }
    }
    tc_fence_before();
    named_bar_sync(1 + qd, 128);                                     // the quadrant's four warps have read S^T / dP^T
    tc_fence_after();
    tmem_st16(tl + C_PP + x0 / 2, pw);
    tmem_st16(tl + C_PS + x0 / 2, dw);
    if (qt < 3) {                                                    // out-of-band packed columns of this quadrant: [0, 16 qd) U [16 qd + 64, 112)
      const int z = qt < qd ? 16 * qt : 16 * qt + 64;
      tmem_st16z(tl + C_PP + z);
      tmem_st16z(tl + C_PS + z);
    }
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    if (ctl) {
      mbar_wait(tready, it & 1);
      tc_fence_after();
      mma_d(it);
      if (elect_one()) tc_commit(dready);
      __syncwarp();
      if (it + 1 < nit) {
        mma_s(it + 1);
        if (elect_one()) tc_commit(sready);
        __syncwarp();
      }
    }
    mbar_wait(dready, it & 1);                                       // this sample's dK / dV, while the next S / dP run
    tc_fence_after();
    drain(it);
  }
  if (lane == 0) tma_store_wait0();
  if (key >= 0 && key < N && wq >= 0 && wq < nwin) {
    const int rr = key - (32 * wq - KOFF);
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) {
      float* dst = dbias + (((size_t)h * nwin + wq) * WQ + jj) * WK + rr;
      if (SP == 1) *dst = dbacc[jj]; else atomicAdd(dst, dbacc[jj]);
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
