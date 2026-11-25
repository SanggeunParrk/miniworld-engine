// lattn_dkv_tf32.cu — dK, dV and dbias of the AF3 windowed atom attention for the fp32 path, tcgen05 kind::tf32 (sm_100a), key-centric:
// the fp32 twin of lattn_dkv.cu.
//
// A CTA owns 128 keys [kb, kb + 128) with kb = 128 j - 112 (every TMEM lane quadrant's 32 keys then see the same 4 query windows), one
// head and a range of samples (SP CTAs split the samples; SP = 1 writes dbias directly, SP = 2 adds into the zeroed dbias). Its keys see
// the 7 windows W0 .. W0 + 6 (W0 = 4 j - 5, the 224 queries from q0 = 32 W0). TMEM lane = key:
//   S^T = K Q^T [128 x 224] cols 0..223,  dP^T = V dO^T cols 224..447,  dK += dS^T Q cols 448..479,  dV += P^T dO cols 480..511.
// Warp w < 16 (lane quadrant w & 3, band quarter w >> 2 = one query window) reads its 32 x 32 band cells of S^T / dP^T, computes
// P = exp2(S c + bias - LSE) and dS = P (dP - D), adds dS into the dbias cells it holds in registers (summed over the samples), and writes
// P^T over S^T and dS^T over dP^T IN PLACE (fp32, each warp exactly the cells it read; band quarters < 3 also zero one out-of-band
// 32-column block of each): the TMEM A operands of dV += P^T dO and dK += dS^T Q.
// q and dO are read twice per sample: K-major for S^T / dP^T (the 128-B swizzle) and MN-major for dK / dV (the 128-B swizzle with 32-B
// atoms, the only MN-major layout a tf32 MMA takes; sm100.cuh).
// Budgets. smem: X = k 16 KB | v 16 KB | q 28 KB | dO 28 KB (one slot: S^T(it + 1) cannot start before dK / dV(it) anyway -- TMEM holds
// one S^T / dP^T -- so X(it + 1) loads during the softmax of it) and two Y slots of MN-major q 28 KB | MN-major dO 28 KB | LSE | D (58 KB;
// a Y slot doubles as its sample's dK / dV staging once the sample's MMAs are done: 16 warps x 2 KB), barriers: 209152 B. TMEM: 512 of
// 512 columns. Registers (<= 120, 544 threads): 32 dbias per thread, the band cells in quarters of 8. The bias is read per sample from
// global memory (__ldg, the 32 lanes on 32 consecutive keys of one row: 128-B coalesced, L1 / L2 resident; holding it in 32 registers spilled):
// log2 domain, -inf on invalid keys, 0 outside the windows.
// Threads: 17 warps: 16 P / dS warps (one key row per thread), warp 16 issues the TMA loads (lane 0) and the MMAs (whole warp,
// elect_one). Each warp drains its 32 x 16 piece of dK or dV through its staging with a TMA store (the first chunk, whose start row is
// negative, with plain 16-byte stores).
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QR = 224, KR = 128;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int X_K = 0, X_V = KR * 128, X_Q = 2 * KR * 128, X_DO = X_Q + QR * 128, XB = X_DO + QR * 128;        // 88 KB
constexpr int Y_QM = 0, Y_DOM = QR * 128, Y_L = 2 * QR * 128, Y_D = Y_L + QR * 4, YB = 59392;                    // 58 KB
constexpr int O_Y = XB, O_BAR = O_Y + 2 * YB, SMEM_BYTES = O_BAR + 256;
static_assert(Y_D + QR * 4 <= YB && 16 * 2048 <= Y_L, "Y slot");
static_assert(XB % 1024 == 0 && X_Q % 1024 == 0 && X_DO % 1024 == 0 && YB % 1024 == 0 && Y_DOM % 1024 == 0, "alignment");
static_assert(SMEM_BYTES == 209152, "keep sm100_atom_local.KERNELS_TF32 in step");
constexpr int C_S = 0, C_DP = QR, C_DK = 2 * QR, C_DV = 2 * QR + 32;
constexpr uint32_t TXX = XB, TXY = 2 * QR * 128 + 2 * QR * 4;
constexpr uint32_t I_S = idesc_tf32(128, QR), I_D = idesc_tf32(128, DH, 0, 1);

DEVI uint32_t rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }   // kind::tf32 truncates
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}
DEVI void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
}
DEVI float lds32f(uint32_t a) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory"); return v; }

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_dkv_tf32(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_qm, const __grid_constant__ CUtensorMap tm_k,
                    const __grid_constant__ CUtensorMap tm_v, const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_dom,
                    const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d, const __grid_constant__ CUtensorMap tm_o,
                    const float* __restrict__ gbias, const uint8_t* __restrict__ kmask, float* __restrict__ dkv, float* __restrict__ dbias,
                    int ldd, int kcol, int vcol, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *fullX = bars, *emptyX = bars + 1, *fullY = bars + 2, *yempty = bars + 4, *sready = bars + 6, *tready = bars + 7, *dready = bars + 8;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int kb = 128 * jc - 112, W0 = 4 * jc - 5, q0 = 32 * W0;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    mbar_init(fullX, 1); mbar_init(emptyX, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(&fullY[i], 1); mbar_init(&yempty[i], 16); }
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int r = 32 * qd + lane, key = kb + r;
  const int wq = W0 + qd + qt;                                       // this warp's band quarter = one window, queries 32 wq ..
  const bool wok = wq >= 0 && wq < nwin;
  const int rr = key - (32 * wq - KOFF);                             // the key's column in window wq's 128 keys (0..127)
  const bool kok = key >= 0 && key < N && (kmask == nullptr || kmask[key]);
  const bool bok = wok && kok;                                        // read the bias; else -inf (invalid key) or 0 (no such window)
  const float bfill = wok ? -INFINITY : 0.f;
  const float* bsrc = gbias + (((size_t)h * nwin + (wok ? wq : 0)) * WQ) * WK + (bok ? rr : 0);
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;

  auto loadX = [&](int it) {                                         // k, v, K-major q, K-major dO of sample it
    const int a = a_lo + it;
    mbar_expect_tx(fullX, TXX);
    tma_load_3d(su + X_K, &tm_k, fullX, h * DH, kb, a);
    tma_load_3d(su + X_V, &tm_v, fullX, h * DH, kb, a);
    tma_load_3d(su + X_Q, &tm_q, fullX, h * DH, q0, a);
    tma_load_3d(su + X_DO, &tm_do, fullX, h * DH, q0, a);
  };
  auto loadY = [&](int it) {                                         // MN-major q, MN-major dO, LSE, D of sample it into Y slot it & 1
    const int y = it & 1, a = a_lo + it;
    const uint32_t yb = su + O_Y + y * YB;
    mbar_expect_tx(&fullY[y], TXY);
    tma_load_3d(yb + Y_QM, &tm_qm, &fullY[y], h * DH, q0, a);
    tma_load_3d(yb + Y_DOM, &tm_dom, &fullY[y], h * DH, q0, a);
    tma_load_3d(yb + Y_L, &tm_l, &fullY[y], q0, h, a);
    tma_load_3d(yb + Y_D, &tm_d, &fullY[y], q0, h, a);
  };
  auto mma_s = [&](int it) {                                         // S^T, dP^T of sample it
    mbar_wait(fullX, it & 1);
    tc_fence_after();
    const uint64_t dk = desc_k128(su + X_K), dq = desc_k128(su + X_Q), dv = desc_k128(su + X_V), ddo = desc_k128(su + X_DO);
    if (elect_one()) {
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_S, dk + (uint64_t)(2 * k), dq + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_DP, dv + (uint64_t)(2 * k), ddo + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
      tc_commit(sready);
      tc_commit(emptyX);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dV = P^T dO, dK = dS^T Q (A = fp32 in TMEM, B = MN-major tiles)
    const int y = it & 1;
    mbar_wait(&fullY[y], (it >> 1) & 1);
    tc_fence_after();
    const uint32_t yb = su + O_Y + y * YB;
    const uint64_t dqm = desc_mn32b(yb + Y_QM, QR * 128), ddm = desc_mn32b(yb + Y_DOM, QR * 128);
    if (elect_one()) {
#pragma unroll 4
      for (int k = 0; k < QR / 8; ++k) {
        const uint64_t boff = (uint64_t)((k * 1024) >> 4);
        umma_ts_tf32(tm + C_DV, tm + C_S + 8 * k, ddm + boff, I_D, k > 0 ? 1u : 0u);
        umma_ts_tf32(tm + C_DK, tm + C_DP + 8 * k, dqm + boff, I_D, k > 0 ? 1u : 0u);
      }
      tc_commit(dready);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0 && nit > 0) {
      loadX(0);
      loadY(0);
      if (nit > 1) loadY(1);
    }
    __syncwarp();
    if (nit > 0) mma_s(0);
    for (int it = 0; it < nit; ++it) {
      if (it + 1 < nit) {                                            // X(it + 1) loads during the softmax of it
        mbar_wait(emptyX, it & 1);
        if (lane == 0) loadX(it + 1);
        __syncwarp();
      }
      mbar_wait(tready, it & 1);                                     // P^T / dS^T(it) in TMEM, dK / dV(it - 1) drained
      tc_fence_after();
      mma_d(it);                                                     // commits dready
      if (it + 1 < nit) mma_s(it + 1);                               // after dK / dV(it) in the MMA order
      if (it + 2 < nit) {
        mbar_wait(&yempty[it & 1], (it >> 1) & 1);                   // the drain of it has left Y slot it & 1
        if (lane == 0) loadY(it + 2);
        __syncwarp();
      }
    }
  } else {
    float dbacc[32];
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) dbacc[jj] = 0.f;
    const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
    const int x0 = 32 * qd + 32 * qt;                                // chunk-relative query column of the band quarter
    const int z0 = 32 * (qt < qd ? qt : qt + 4);                     // the out-of-band block this warp zeroes (qt < 3)
    for (int it = 0; it < nit; ++it) {
      const int y = it & 1;
      const uint32_t yb = su + O_Y + y * YB;
      mbar_wait(sready, it & 1);
      mbar_wait(&fullY[y], (it >> 1) & 1);                           // LSE / D of this sample
      tc_fence_after();
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {                               // quarters of 8 queries: S^T / dP^T in, P^T / dS^T out
        uint32_t sv[8], dv[8];
        tmem_ld8(tl + C_S + x0 + 8 * qq, sv);
        tmem_ld8(tl + C_DP + x0 + 8 * qq, dv);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int x = x0 + 8 * qq + k, b = 8 * qq + k;
          const float lse = lds32f(yb + Y_L + 4 * x), dd = lds32f(yb + Y_D + 4 * x);
          const float bb = bok ? __ldg(bsrc + b * WK) * LOG2E : bfill;
          const float p = ex2f(fmaf(__uint_as_float(sv[k]), QSCALE * LOG2E, bb) - lse);
          const float ds = p * (__uint_as_float(dv[k]) - dd);
          dbacc[b] += ds;
          sv[k] = rna_tf32(p);                                       // the dV / dK operands, rounded (dbias takes the exact dS)
          dv[k] = rna_tf32(ds);
        }
        tmem_st8(tl + C_S + x0 + 8 * qq, sv);
        tmem_st8(tl + C_DP + x0 + 8 * qq, dv);
      }
      if (qt < 3) {                                                  // out-of-band columns of this quadrant: [0, 32 qd) U [32 qd + 128, 224)
        tmem_st16z(tl + C_S + z0); tmem_st16z(tl + C_S + z0 + 16);
        tmem_st16z(tl + C_DP + z0); tmem_st16z(tl + C_DP + z0 + 16);
      }
      tmem_wait_st();
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(tready);
      mbar_wait(dready, it & 1);                                     // this sample's dK / dV (the MMAs that read Y slot y are done)
      tc_fence_after();
      {                                                              // drain: 32 keys x 16 columns of dK (quarters 0, 1) / dV (2, 3)
        uint32_t v[16];
        tmem_ld16(tl + (qt < 2 ? C_DK + 16 * qt : C_DV + 16 * (qt - 2)), v);
        tmem_wait_ld();
        const float sc = qt < 2 ? QSCALE : 1.f;
        const uint32_t ob = yb + warp * 2048;                        // [32 rows][64 B], 64-B swizzle, over the sample's MN-major q / dO
#pragma unroll
        for (int g = 0; g < 4; ++g)
          sts128(ob + sw64(lane, g), make_uint4(__float_as_uint(__uint_as_float(v[4 * g]) * sc), __float_as_uint(__uint_as_float(v[4 * g + 1]) * sc),
                                                __float_as_uint(__uint_as_float(v[4 * g + 2]) * sc), __float_as_uint(__uint_as_float(v[4 * g + 3]) * sc)));
        fence_proxy_async();
        __syncwarp();
        const int kr = kb + 32 * qd, a = a_lo + it;
        const int ocol = (qt < 2 ? kcol : vcol) + h * DH + 16 * (qt & 1);
        if (kr >= 0) {
          if (lane == 0) {
            tma_store_3d(&tm_o, ob, ocol, kr, a);
            tma_store_commit();
            tma_store_wait_read0();                                  // the slot may take sample it + 2
          }
        } else if (key >= 0 && key < N) {                            // the first chunk's quadrants (negative start row): plain stores
          float* d = dkv + ((size_t)a * N + key) * ldd + ocol;
#pragma unroll
          for (int g = 0; g < 4; ++g) *reinterpret_cast<uint4*>(d + 4 * g) = lds128(ob + sw64(lane, g));
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&yempty[y]);
      }
    }
    if (wok && key >= 0 && key < N) {
#pragma unroll
      for (int jj = 0; jj < 32; ++jj) {
        float* dst = dbias + (((size_t)h * nwin + wq) * WQ + jj) * WK + rr;
        if (SP == 1) *dst = dbacc[jj]; else atomicAdd(dst, dbacc[jj]);
      }
    }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
