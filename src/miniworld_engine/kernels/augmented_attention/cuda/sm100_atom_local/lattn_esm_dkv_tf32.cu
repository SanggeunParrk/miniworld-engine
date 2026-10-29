// lattn_esm_dkv_tf32.cu — dK and dV of ESMFold2's atom attention for the fp32 path, tcgen05 kind::tf32 (sm_100a): the fp32 twin of
// lattn_esm_dkv.cu.
//
//   P^T = exp2(S^T c + bias2 - LSE)   dS^T = P^T (dP^T - D)   dV = P^T dO   dK = dS^T Q / sqrt(32)        (fp32 into dkv[row, 128 / 256 + ..])
//
// A CTA owns 128 keys [k0, k0 + 128) of one head and a range of samples, seen by the 256 queries [k0 - 64, k0 + 192) = eight 32-query cells;
// TMEM lane = key, and the 32 keys of lane quadrant kq see the five cells kq .. kq + 4. In fp32 P^T / dS^T are as wide as S^T / dP^T and dV /
// dK need their own columns, so the queries go in two phases: cells 0..6 (S^T cols 0..223, dP^T cols 224..447, dV 448..479, dK 480..511),
// then cell 7 (seen by quadrant 3 only) over cell 0's columns once dV / dK of the first phase have read them. P^T / dS^T are written in place
// (every warp writes exactly the cells it read; out-of-band cells zeroed). Each lane quadrant (= SM sub-partition) then has five cells in
// phase 1 and at most one in phase 2 (a split into halves of 128 queries gave quadrants 0 and 3 four cells in one half: 8 on the critical
// path per sample instead of 6).
// q and dO are read K-major (S^T, dP^T) and MN-major (dK, dV: the 128-B swizzle with 32-B atoms), in boxes of 224 and 32 queries; every
// operand has its own slot so the next sample's loads go out as soon as their slot is read (the control warp learns this from the barrier it
// waits on anyway: sready after the phase's S^T, which also follows the previous dV / dK; dready after the last). K | V | LSE | D in two slots
// by sample parity. dK / dV leave through shared memory (the phase-1 MN-major slot, read by then) with per-warp TMA stores.
// Warp w (kq = w & 3, qt = w >> 2: cell kq + qt + 1, or the triangular cells kq and kq + 4 for qt = 3) holds the bias of its cells in registers
// (the triangle pair merged in 32 + 1). Warp 15 (quadrant 3's triangle warp: cell 7) also issues the TMA loads and the MMAs.
// Budgets: smem 196 KB; TMEM 512 columns; 512 threads (<= 128 registers).
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, KR = 128, QR = 256, Q1 = 224, Q2 = 32;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int R1 = Q1 * 128, R2 = Q2 * 128;                         // bytes of one operand (q or dO) in phase 1 / 2
constexpr int O_A1 = 0, O_A2 = O_A1 + 2 * R1, O_B1 = O_A2 + 2 * R2, O_B2 = O_B1 + 2 * R1, O_KV = O_B2 + 2 * R2;   // A: K-major, B: MN-major
constexpr int K_K = 0, K_V = KR * 128, K_L = 2 * KR * 128, K_D = K_L + QR * 4, KVSLOT = K_D + QR * 4;
static_assert(R1 % 1024 == 0 && R2 % 1024 == 0 && KVSLOT % 1024 == 0 && O_KV % 1024 == 0, "slots");
static_assert(2 * R1 >= 16 * 2048, "the dK / dV staging fits in the phase-1 MN-major slot");
constexpr int O_BAR = O_KV + 2 * KVSLOT, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES == 200960, "keep sm100_atom_local.KERNELS_ESM32 in step");
constexpr int C_S = 0, C_DP = Q1, C_DV = 2 * Q1, C_DK = 2 * Q1 + DH;
constexpr uint32_t I_1 = idesc_tf32(128, Q1), I_2 = idesc_tf32(128, Q2), I_D = idesc_tf32(128, DH, 0, 1);

// tf32 rounding, to nearest with ties away from zero (= cvt.rna.tf32.f32 for finite values, in two integer operations)
DEVI uint32_t rna_tf32(float x) { return (__float_as_uint(x) + 0x1000u) & 0xffffe000u; }
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}
DEVI void tmem_zero_cell(uint32_t tl, int cs, int cdp) {           // a 32 x 32 cell of S^T and of dP^T
  tmem_st16z(tl + cs); tmem_st16z(tl + cs + 16); tmem_st16z(tl + cdp); tmem_st16z(tl + cdp + 16);
}

DEVI void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_esm_dkv_tf32(const __grid_constant__ CUtensorMap tm_q1, const __grid_constant__ CUtensorMap tm_q2,
                        const __grid_constant__ CUtensorMap tm_qm1, const __grid_constant__ CUtensorMap tm_qm2,
                        const __grid_constant__ CUtensorMap tm_do1, const __grid_constant__ CUtensorMap tm_do2,
                        const __grid_constant__ CUtensorMap tm_dom1, const __grid_constant__ CUtensorMap tm_dom2,
                        const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                        const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                        const __grid_constant__ CUtensorMap tm_out, const float* __restrict__ gbias, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  // full: A1, A2, B1, B2, KV[0], KV[1]; then sready (one commit per phase), tready (phase 1, 16 warps), tready2 (phase 2, warps 12..15),
  // dready, dfree (16), sfree (16: the dK / dV staging read by the TMA stores)
#define BARS reinterpret_cast<uint64_t*>(sm + O_BAR)
#define fullA1 (BARS + 0)
#define fullA2 (BARS + 1)
#define fullB1 (BARS + 2)
#define fullB2 (BARS + 3)
#define fullKV (BARS + 4)
#define sready (BARS + 6)
#define tready (BARS + 7)
#define tready2 (BARS + 8)
#define dready (BARS + 9)
#define dfree (BARS + 10)
#define sfree (BARS + 11)
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int k0 = 128 * jc, qb = k0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;

  if (tid == 0) {
    for (int i = 0; i < 6; ++i) mbar_init(BARS + i, 1);
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(tready2, 4); mbar_init(dready, 1); mbar_init(dfree, 16); mbar_init(sfree, 16);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int kq = warp & 3, qt = warp >> 2;
  const bool tri = qt == 3;
  const int cell_a = tri ? 0 : qt + 1;
  float bl[32], bx = -INFINITY;                                      // bias(query jj of the cell, this key): the triangle pair merged
  {
    auto src = [&](int c) {                                          // cell c: window 4 jc - 2 + kq + c, key column lane + 128 - 32 c
      const int w = 4 * jc - 2 + kq + c;
      return (w >= 0 && w < nwin) ? gbias + (((size_t)h * nwin + w) * WQ) * SPAN + lane + 128 - 32 * c : nullptr;
    };
    const float* sa = src(cell_a);
    const float* s4 = tri ? src(4) : nullptr;
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) {
      const float va = sa ? __ldg(sa + jj * SPAN) : -INFINITY;
      if (!tri) bl[jj] = va;
      else {
        const float v4 = s4 ? __ldg(s4 + jj * SPAN) : -INFINITY;
        bl[jj] = jj >= lane ? va : v4;
        if (jj == lane) bx = v4;
      }
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;

  // loads of sample it (issued by lane 0 of the control warp)
  auto load_a = [&](int it, int ph) {                                // K-major q | dO of phase ph
    const int a = a_lo + it;
    uint64_t* bar = ph == 1 ? fullA1 : fullA2;
    const uint32_t d = su + (ph == 1 ? O_A1 : O_A2);
    const int r = ph == 1 ? R1 : R2;
    mbar_expect_tx(bar, 2 * r);
    tma_load_3d(d, ph == 1 ? &tm_q1 : &tm_q2, bar, h * DH, qb + (ph == 1 ? 0 : Q1), a);
    tma_load_3d(d + r, ph == 1 ? &tm_do1 : &tm_do2, bar, h * DH, qb + (ph == 1 ? 0 : Q1), a);
  };
  auto load_b = [&](int it, int ph) {                                // MN-major q | dO of phase ph
    const int a = a_lo + it;
    uint64_t* bar = ph == 1 ? fullB1 : fullB2;
    const uint32_t d = su + (ph == 1 ? O_B1 : O_B2);
    const int r = ph == 1 ? R1 : R2;
    mbar_expect_tx(bar, 2 * r);
    tma_load_3d(d, ph == 1 ? &tm_qm1 : &tm_qm2, bar, h * DH, qb + (ph == 1 ? 0 : Q1), a);
    tma_load_3d(d + r, ph == 1 ? &tm_dom1 : &tm_dom2, bar, h * DH, qb + (ph == 1 ? 0 : Q1), a);
  };
  auto load_kv = [&](int it) {
    const int a = a_lo + it, sl = it & 1;
    const uint32_t d = su + O_KV + sl * KVSLOT;
    mbar_expect_tx(fullKV + sl, KVSLOT);
    tma_load_3d(d + K_K, &tm_k, fullKV + sl, h * DH, k0, a);
    tma_load_3d(d + K_V, &tm_v, fullKV + sl, h * DH, k0, a);
    tma_load_3d(d + K_L, &tm_l, fullKV + sl, qb, h, a);
    tma_load_3d(d + K_D, &tm_d, fullKV + sl, qb, h, a);
  };
  auto mma_st = [&](int it, int ph) {                                // S^T = K Q^T, dP^T = V dO^T over the phase's queries
    const uint32_t kv = su + O_KV + (it & 1) * KVSLOT, qa = su + (ph == 1 ? O_A1 : O_A2);
    const uint64_t dk = desc_k128(kv + K_K), dv = desc_k128(kv + K_V);
    const uint64_t dq = desc_k128(qa), ddo = desc_k128(qa + (ph == 1 ? R1 : R2));
    const uint32_t id = ph == 1 ? I_1 : I_2;
    if (elect_one()) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        umma_ss_tf32(tm + C_S, dk + (uint64_t)(2 * k), dq + (uint64_t)(2 * k), id, k > 0 ? 1u : 0u);
        umma_ss_tf32(tm + C_DP, dv + (uint64_t)(2 * k), ddo + (uint64_t)(2 * k), id, k > 0 ? 1u : 0u);
      }
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_d = [&](int ph) {                                         // dV (+)= P^T dO, dK (+)= dS^T Q (B MN-major)
    const uint32_t qm = su + (ph == 1 ? O_B1 : O_B2);
    const uint32_t r = ph == 1 ? R1 : R2;
    const uint64_t dq = desc_mn32b(qm, r), ddo = desc_mn32b(qm + r, r);
    const int nk = (ph == 1 ? Q1 : Q2) / 8;
    if (elect_one()) {
#pragma unroll 4
      for (int k = 0; k < nk; ++k) {
        const uint64_t boff = (uint64_t)((k * 1024) >> 4);
        const uint32_t acc = (ph == 2 || k > 0) ? 1u : 0u;
        umma_ts_tf32(tm + C_DV, tm + C_S + 8 * k, ddo + boff, I_D, acc);
        umma_ts_tf32(tm + C_DK, tm + C_DP + 8 * k, dq + boff, I_D, acc);
      }
    }
    __syncwarp();
  };
  if (ctl && nit > 0) {
    if (lane == 0) { load_kv(0); load_a(0, 1); load_a(0, 2); load_b(0, 1); load_b(0, 2); }
    __syncwarp();
    mbar_wait(fullKV, 0);
    mbar_wait(fullA1, 0);
    tc_fence_after();
    mma_st(0, 1);
  }

  const uint32_t tl = tm + ((uint32_t)(32 * kq) << 16);
  // the out-of-band cell of phase 1 this warp zeroes (qt < 3): the qt-th of {0 .. kq - 1} + {kq + 5 .. 6}, or none (7)
  const int za = qt < kq ? qt : kq + 5 + (qt - kq);
  for (int it = 0; it < nit; ++it) {
    const float* sl = reinterpret_cast<const float*>(sm + O_KV + (it & 1) * KVSLOT + K_L);
    const float* sd = reinterpret_cast<const float*>(sm + O_KV + (it & 1) * KVSLOT + K_D);
    // P^T, dS^T of query cell ca (columns cs / cdp of S^T / dP^T), in place; second: the triangle's cell kq + 4
    auto cell_pd = [&](int ca, int cs, int cdp, bool second) {
#pragma unroll
      for (int hf = 0; hf < 4; ++hf) {                               // 8 columns at a time (128 registers)
        uint32_t sv[8], dv[8];
        tmem_ld8(tl + cs + 8 * hf, sv);
        tmem_ld8(tl + cdp + 8 * hf, dv);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const int b = 8 * hf + j, x = 32 * ca + b;
          const float2 l2 = *reinterpret_cast<const float2*>(sl + x), d2 = *reinterpret_cast<const float2*>(sd + x);
          float bb[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int q = b + e;
            bb[e] = !tri ? bl[q] : (!second ? (q >= lane ? bl[q] : -INFINITY) : (q < lane ? bl[q] : (q == lane ? bx : -INFINITY)));
          }
          const float p0 = ex2f(fmaf(__uint_as_float(sv[j]), QSCALE * LOG2E, bb[0]) - l2.x);
          const float p1 = ex2f(fmaf(__uint_as_float(sv[j + 1]), QSCALE * LOG2E, bb[1]) - l2.y);
          const float t0 = p0 * (__uint_as_float(dv[j]) - d2.x), t1 = p1 * (__uint_as_float(dv[j + 1]) - d2.y);
          sv[j] = rna_tf32(p0); sv[j + 1] = rna_tf32(p1);
          dv[j] = rna_tf32(t0); dv[j + 1] = rna_tf32(t1);
        }
        tmem_st8(tl + cs + 8 * hf, sv);
        tmem_st8(tl + cdp + 8 * hf, dv);
      }
    };
    // ---- phase 1: cells 0..6
    mbar_wait(sready, 0);                                            // (two commits per sample: phase 1 is always parity 0)
    tc_fence_after();
    {
      const int ca = kq + cell_a;
      cell_pd(ca, C_S + 32 * ca, C_DP + 32 * ca, false);
      if (tri) {
        if (kq < 3) cell_pd(kq + 4, C_S + 32 * (kq + 4), C_DP + 32 * (kq + 4), true);
      } else if (za < 7) {
        tmem_zero_cell(tl, C_S + 32 * za, C_DP + 32 * za);
      }
    }
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    const bool more = it + 1 < nit;
    if (ctl) {
      // S^T(it) of phase 1 is done: slot A1 is free, and the other K / V slot (its sample's last MMAs preceded this one)
      if (more && lane == 0) { load_a(it + 1, 1); load_kv(it + 1); }
      __syncwarp();
      mbar_wait(tready, it & 1);                                     // P^T / dS^T of phase 1 written by all
      if (it > 0) mbar_wait(dfree, (it - 1) & 1);                    // dV / dK(it - 1) drained
      mbar_wait(fullB1, it & 1);
      tc_fence_after();
      mma_d(1);
      mbar_wait(fullA2, it & 1);                                     // cell 7 over cell 0's columns: after phase 1's dV / dK
      tc_fence_after();
      mma_st(it, 2);
    }
    // ---- phase 2: cell 7 (quadrant 3's triangle warp); the triangle warps of quadrants 0..2 zero their rows of it
    if (tri) {
      mbar_wait(sready, 1);
      tc_fence_after();
      if (kq == 3) cell_pd(7, C_S, C_DP, true);
      else tmem_zero_cell(tl, C_S, C_DP);
      tmem_wait_st();
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(tready2);
    }
    if (ctl) {
      // S^T of phase 2 done (and phase 1's dV / dK before it): slot A2 is free
      if (more && lane == 0) load_a(it + 1, 2);
      __syncwarp();
      mbar_wait(tready2, it & 1);
      mbar_wait(fullB2, it & 1);
      tc_fence_after();
      mma_d(2);
      if (elect_one()) tc_commit(dready);
      __syncwarp();
      if (more) {
        mbar_wait(fullKV + ((it + 1) & 1), ((it + 1) >> 1) & 1);
        mbar_wait(fullA1, (it + 1) & 1);
        tc_fence_after();
        mma_st(it + 1, 1);
      }
    }
    // ---- dK / dV: 32 keys x 16 columns per warp, staged in slot B1 (read by phase 1's dV / dK, done), per-warp TMA stores
    if (!tri) mbar_wait(sready, 1);                                  // (keeps every warp in step with sready's phases)
    mbar_wait(dready, it & 1);
    tc_fence_after();
    {
      uint32_t v[16];
      tmem_ld16(tl + (qt < 2 ? C_DK : C_DV) + 16 * (qt & 1), v);
      tmem_wait_ld();
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(dfree);
      const float sc = qt < 2 ? QSCALE : 1.f;
      const uint32_t ob = su + O_B1 + warp * 2048;                   // [32 rows][64 B], 64-B swizzle
#pragma unroll
      for (int g = 0; g < 4; ++g)
        sts128(ob + sw64(lane, g), make_uint4(__float_as_uint(__uint_as_float(v[4 * g]) * sc), __float_as_uint(__uint_as_float(v[4 * g + 1]) * sc),
                                              __float_as_uint(__uint_as_float(v[4 * g + 2]) * sc), __float_as_uint(__uint_as_float(v[4 * g + 3]) * sc)));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_3d(&tm_out, ob, (qt < 2 ? 0 : 128) + h * DH + 16 * (qt & 1), k0 + 32 * kq, a_lo + it);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(sfree);
      }
      __syncwarp();
    }
    if (ctl && more) {                                               // dV / dK of phase 2 done (dready): slot B2 is free; B1 once the
      if (lane == 0) load_b(it + 1, 2);                              // stores have read the staging
      __syncwarp();
      mbar_wait(sfree, it & 1);
      if (lane == 0) load_b(it + 1, 1);
      __syncwarp();
    }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
#undef fullA1
#undef fullA2
#undef fullB1
#undef fullB2
#undef fullKV
#undef sready
#undef tready
#undef tready2
#undef dready
#undef dfree
#undef sfree
#undef BARS
