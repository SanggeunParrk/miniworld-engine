// ffn_bwd.cu — the SWA atom block's FFN-half backward on sm_100a (the B200 port of team-gm swa_cuda/swa_ffn_bwd.cu; the math and
// rounding points of the Triton _ffn_bwd):
//   dffn = rn(dq2 rn(gate_f));  a|b = y Wu^T (recomputed);  dh = dffn Wd;  sa = sigmoid(a);  h = rn(a sa b);
//   da = rn(dh b sa (1 + a (1 - sa)));  db = rn(dh a sa);  dy = da Wa + db Wb;
//   dq1 = dq2 + rstd (dxh - xh mean(dxh xh)),  dxh = dy (1 + scale_f),  xh = q1 rstd;
//   d shift_f += sum_aug dy,  d scale_f += sum_aug dy xh,  d gate_f += sum_aug dq2 ffn   (pre-summed over the tile's augments, then red.add)
// and writes the weight-gradient operands dffn [M, C], h [M, 256], [da | db] [M, 512] (dWu = [da|db]^T y, dWd = dffn^T h on cuBLAS).
// Structure: persistent CTAs, tiles of SP augments x AT atoms; per 32-unit hidden chunk j three MMAs -- a|b_j = y Wab_j^T (SS, y from
// smem), dh_j = dffn Wd_j (TS, dffn in TMEM, B = Wd^T rows), dy += [da_j | db_j] Wab_j (TS, [da|db] in TMEM, B = Wab^T rows) -- with the
// a|b / dh and [da|db] buffers doubled so chunk j + 1's GEMMs run while the row threads work on chunk j; weights stream through a ring of
// 16-KB slots (per chunk Wab_j | Wd^T_j | Wab^T_j). Two warpgroups split the channels and each chunk's hidden units. The per-atom sums
// go through the (then dead) y tile as an fp32 staging area.
// TMEM: dffn (bf16) at 0, a|b[2] at 64 / 128, dh[2] at 192 / 224, [da|db][2] (bf16) at 256 / 288, dy at 320.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int C = 128, NHID = 256, HC = 32, NCH = NHID / HC, RSMAX = 6;
constexpr int KB = 128 * 128;                                              // [128 rows][64 bf16] k-block (16 KB) = one ring slot
constexpr int O_Y = 0, O_D = O_Y + 2 * KB, O_Q1 = O_D + 2 * KB, O_R = O_Q1 + 2 * KB;   // then ring (RS slots), mod, ss, bars (runtime)
constexpr uint32_t T_DF = 0, T_AB = 64, T_DH = 192, T_DAB = 256, T_DY = 320;
constexpr uint32_t I_AB = idesc_bf16(128, 2 * HC), I_DH = idesc_bf16(128, HC), I_DY = idesc_bf16(128, 128);
constexpr int SPT = 3 * NCH;                                               // ring slots per tile

struct Bars {
  uint64_t infull, infree, wfull[RSMAX], wempty[RSMAX], dffull, abfull[2], abfree[2], dabfull[2], dabfree[2], dyfull, dyfree;
  uint32_t tmem;
};

DEVI float sigm(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }
DEVI void red4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
swa_ffn_bwd_sm100(const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mdq2, const __grid_constant__ CUtensorMap mq1,
                  const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwdt,
                  const __grid_constant__ CUtensorMap mwabt, int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps, int RS,
                  const __nv_bfloat16* __restrict__ FFN, __nv_bfloat16* __restrict__ DQ1, __nv_bfloat16* __restrict__ DFFN,
                  __nv_bfloat16* __restrict__ HH, __nv_bfloat16* __restrict__ DAB, float* __restrict__ DMOD) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int O_MOD = O_R + RS * KB, O_SS = O_MOD + 8 * AT * 128, O_BAR = (O_SS + 2 * 128 * 4 + 7) / 8 * 8;
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    mbar_init(&B.infull, 1); mbar_init(&B.infree, 8);
    for (int i = 0; i < RS; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    mbar_init(&B.dffull, 8);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.abfull[i], 1); mbar_init(&B.abfree[i], 8); mbar_init(&B.dabfull[i], 8); mbar_init(&B.dabfree[i], 1); }
    mbar_init(&B.dyfull, 1); mbar_init(&B.dyfree, 8);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: tile inputs
    if (lane == 0) {
      const uint32_t tb = (uint32_t)(SP * AT * 128);
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        if (T >= 1) mbar_wait(&B.infree, (T - 1) & 1);
        mbar_expect_tx(&B.infull, 6 * tb + (uint32_t)(8 * AT * 128));
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_4d(su + O_Y + kb * KB, &my, &B.infull, kb * 64, s0, b, a0);
          tma_load_4d(su + O_D + kb * KB, &mdq2, &B.infull, kb * 64, s0, b, a0);
          tma_load_4d(su + O_Q1 + kb * KB, &mq1, &B.infull, kb * 64, s0, b, a0);
        }
        tma_load_3d(su + O_MOD, &mmod, &B.infull, 0, b * S + s0, 16);      // blocks 16..23: scale_f | gate_f
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: weight ring
    // order = MMA issue order: Wab_0, Wdt_0, then per chunk j: Wab_{j+1}, Wdt_{j+1} (j < NCH - 1), Wabt_j
    if (lane == 0) {
      for (int T = 0, g = 0; T < ntT; ++T)
        for (int kk = 0; kk < SPT; ++kk, ++g) {
          const int sl = g % RS;
          if (g >= RS) mbar_wait(&B.wempty[sl], ((g / RS) - 1) & 1);
          const uint32_t d = su + O_R + sl * KB;
          int kind, j;                                                     // 0 Wab, 1 Wdt, 2 Wabt
          if (kk < 2) { kind = kk; j = 0; }
          else {
            const int jj = (kk - 2) / 3, e = (kk - 2) % 3;
            if (jj < NCH - 1) { kind = e < 2 ? e : 2; j = e < 2 ? jj + 1 : jj; } else { kind = 2; j = jj; }
          }
          if (kind == 0) {                                                 // [Wa_j ; Wb_j] [64 rows][128]: 2 k-blocks of 8 KB
            mbar_expect_tx(&B.wfull[sl], 2 * 64 * 128);
            for (int kb = 0; kb < 2; ++kb) tma_load_2d(d + kb * 8192, &mwab, &B.wfull[sl], kb * 64, j * 2 * HC);
          } else if (kind == 1) {                                          // Wd^T rows 32 j.. [32 rows][128]: 2 k-blocks of 4 KB
            mbar_expect_tx(&B.wfull[sl], 2 * 32 * 128);
            for (int kb = 0; kb < 2; ++kb) tma_load_2d(d + kb * 4096, &mwdt, &B.wfull[sl], kb * 64, j * HC);
          } else {                                                         // Wab^T columns 64 j.. [128 rows][64]
            mbar_expect_tx(&B.wfull[sl], KB);
            tma_load_2d(d, &mwabt, &B.wfull[sl], j * 2 * HC, 0);
          }
        }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    auto wslot = [&](int g) { const int sl = g % RS; mbar_wait(&B.wfull[sl], (g / RS) & 1); return sl; };
    for (int T = 0, g = 0; T < ntT; ++T) {
      mbar_wait(&B.infull, T & 1);
      mbar_wait(&B.dffull, T & 1);
      auto abdh = [&](int j) {                                             // a|b_j (SS) and dh_j (TS) into buffer (T NCH + j) & 1
        const int ci = T * NCH + j, bb = ci & 1;
        if (ci >= 2) mbar_wait(&B.abfree[bb], ((ci >> 1) - 1) & 1);
        int sl = wslot(g++);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ss(tmem + T_AB + bb * 64, desc_k128(su + O_Y + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2),
                    desc_k128(su + O_R + sl * KB + (ks >> 2) * 8192) + (uint64_t)((ks & 3) * 2), I_AB, ks > 0 ? 1u : 0u);
          tc_commit(&B.wempty[sl]);
        }
        __syncwarp();
        sl = wslot(g++);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_DH + bb * 32, tmem + T_DF + ks * 8, desc_k128(su + O_R + sl * KB + (ks >> 2) * 4096) + (uint64_t)((ks & 3) * 2), I_DH,
                    ks > 0 ? 1u : 0u);
          tc_commit(&B.wempty[sl]);
          tc_commit(&B.abfull[bb]);
        }
        __syncwarp();
      };
      abdh(0);
      if (T >= 1) mbar_wait(&B.dyfree, (T - 1) & 1);                       // the previous tile's dy has been read out
      for (int j = 0; j < NCH; ++j) {
        if (j + 1 < NCH) abdh(j + 1);
        const int ci = T * NCH + j, bb = ci & 1;
        mbar_wait(&B.dabfull[bb], (ci >> 1) & 1);
        const int sl = wslot(g++);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_DY, tmem + T_DAB + bb * 32 + ks * 8, desc_k128(su + O_R + sl * KB) + (uint64_t)(ks * 2), I_DY, (j > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.wempty[sl]);
          tc_commit(&B.dabfree[bb]);
          if (j == NCH - 1) tc_commit(&B.dyfull);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row threads: warpgroup w = channel half
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    const int al = rowok ? at : 0;
    const size_t NS = (size_t)A * Bn;
    float* ssx = reinterpret_cast<float*>(sm + O_SS);
    auto modv = [&](int kind, int j) {                                     // float4 of channels 4 j .. of kind (0 scale_f, 1 gate_f)
      const int row = (kind * 4 + (j >> 3)) * AT + al;
      return *reinterpret_cast<const float4*>(sm + O_MOD + row * 128 + (((j & 7) ^ (row & 7)) << 4));
    };
    // per-atom sums over the tile's SP augments of a [128 rows][64 channels] fp32 quantity held one row per thread (vals = this warpgroup's
    // 64 channels): staged in the (dead) y tile as [2 warpgroups][128 rows][32] x 2 passes, then one red.add.v4 per (atom, 4 channels)
    auto atom_sum = [&](const float* vals, int col0, int b, int s0) {
      float* P = reinterpret_cast<float*>(sm + O_Y);
      for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int rr = w * 128 + (int)r;
          *reinterpret_cast<float4*>(P + rr * 32 + ((q ^ (rr & 7)) << 2)) =
              rowok ? make_float4(vals[32 * hh + 4 * q], vals[32 * hh + 4 * q + 1], vals[32 * hh + 4 * q + 2], vals[32 * hh + 4 * q + 3])
                    : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        named_bar_sync(1, 256);
        for (int u = w * 128 + (int)r; u < 2 * AT * 8; u += 256) {        // work unit = (warpgroup half ww, atom, quad)
          const int ww = u / (AT * 8), rem = u % (AT * 8), a_ = rem / 8, q = rem % 8;
          float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
          for (int s_ = 0; s_ < SP; ++s_) {
            const int rr = ww * 128 + s_ * AT + a_;
            const float4 v = *reinterpret_cast<const float4*>(P + rr * 32 + ((q ^ (rr & 7)) << 2));
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
          }
          if (s0 + a_ < S) red4(DMOD + ((size_t)b * S + s0 + a_) * (6 * C) + col0 + 64 * ww + 32 * hh + 4 * q, acc.x, acc.y, acc.z, acc.w);
        }
        named_bar_sync(1, 256);
      }
    };
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && (size_t)n < NS && s < S;
      const size_t row = ((size_t)n * S + s);
      mbar_wait(&B.infull, T & 1);
      // ---- pass A: dffn = rn(dq2 rn(gate_f)) for this warpgroup's 64 channels -> TMEM (A of dh) and DFFN
      {
        uint32_t dp[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int ch = 8 * w + k;
          const uint4 du = lds128(su + O_D + (ch >> 3) * KB + sw128(r, ch & 7));
          const uint32_t d4[4] = {du.x, du.y, du.z, du.w};
          const float4 g0 = modv(1, 2 * ch), g1 = modv(1, 2 * ch + 1);
          const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
            dp[4 * k + e] = pack_bf16(bf16lo(d4[e]) * __bfloat162float(__float2bfloat16_rn(gv[2 * e])),
                                      bf16hi(d4[e]) * __bfloat162float(__float2bfloat16_rn(gv[2 * e + 1])));
        }
        tc_fence_after();                                                  // (the previous tile's dh MMAs are done: its dyfull was waited)
#pragma unroll
        for (int k = 0; k < 2; ++k) tmem_st16(trow + T_DF + 32 * w + 16 * k, *reinterpret_cast<uint32_t(*)[16]>(dp + 16 * k));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.dffull);
        if (ok) {
          uint4* dst = reinterpret_cast<uint4*>(DFFN + row * C + 64 * w);
#pragma unroll
          for (int k = 0; k < 8; ++k) dst[k] = make_uint4(dp[4 * k], dp[4 * k + 1], dp[4 * k + 2], dp[4 * k + 3]);
        }
      }
      // ---- chunks: h, da, db for this warpgroup's 16 hidden units of the chunk -> TMEM [da|db] and HH / DAB
      for (int j = 0; j < NCH; ++j) {
        const int ci = T * NCH + j, bb = ci & 1;
        mbar_wait(&B.abfull[bb], (ci >> 1) & 1);
        tc_fence_after();
        uint32_t av[16], bv[16], dv[16];
        tmem_ld16(trow + T_AB + bb * 64 + 16 * w, av);
        tmem_ld16(trow + T_AB + bb * 64 + 32 + 16 * w, bv);
        tmem_ld16(trow + T_DH + bb * 32 + 16 * w, dv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abfree[bb]);
        uint32_t hp[8], dap[8], dbp[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          float hv[2], dav[2], dbv[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float a = __uint_as_float(av[2 * k + e]), bq = __uint_as_float(bv[2 * k + e]), dh = __uint_as_float(dv[2 * k + e]);
            const float sa = sigm(a);
            hv[e] = a * sa * bq;
            dav[e] = dh * bq * sa * (1.f + a * (1.f - sa));
            dbv[e] = dh * a * sa;
          }
          hp[k] = pack_bf16(hv[0], hv[1]); dap[k] = pack_bf16(dav[0], dav[1]); dbp[k] = pack_bf16(dbv[0], dbv[1]);
        }
        if (ci >= 2) mbar_wait(&B.dabfree[bb], ((ci >> 1) - 1) & 1);
        tc_fence_after();
        tmem_st8(trow + T_DAB + bb * 32 + 8 * w, dap);
        tmem_st8(trow + T_DAB + bb * 32 + 16 + 8 * w, dbp);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.dabfull[bb]);
        if (ok) {
          uint4* dh_ = reinterpret_cast<uint4*>(HH + row * NHID + HC * j + 16 * w);
          dh_[0] = make_uint4(hp[0], hp[1], hp[2], hp[3]); dh_[1] = make_uint4(hp[4], hp[5], hp[6], hp[7]);
          uint4* da_ = reinterpret_cast<uint4*>(DAB + row * (2 * NHID) + HC * j + 16 * w);
          da_[0] = make_uint4(dap[0], dap[1], dap[2], dap[3]); da_[1] = make_uint4(dap[4], dap[5], dap[6], dap[7]);
          uint4* db_ = reinterpret_cast<uint4*>(DAB + row * (2 * NHID) + NHID + HC * j + 16 * w);
          db_[0] = make_uint4(dbp[0], dbp[1], dbp[2], dbp[3]); db_[1] = make_uint4(dbp[4], dbp[5], dbp[6], dbp[7]);
        }
      }
      // ---- final: dq1, and the per-atom sums of dy (shift_f), dy xh (scale_f), dq2 ffn (gate_f)
      mbar_wait(&B.dyfull, T & 1);
      tc_fence_after();
      float dy[64];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint32_t v[32];
        tmem_ld32(trow + T_DY + 64 * w + 32 * hh, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 32; ++k) dy[32 * hh + k] = __uint_as_float(v[k]);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dyfree);
      float ssq = 0.f;                                                     // full-row sum of q1^2 (both warpgroups compute it)
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const uint4 u = lds128(su + O_Q1 + (k >> 3) * KB + sw128(r, k & 7));
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float x0 = bf16lo(w4[e]), x1 = bf16hi(w4[e]); ssq = fmaf(x0, x0, fmaf(x1, x1, ssq)); }
      }
      const float rstd = rsqrtf(ssq * (1.f / C) + eps);
      float xh[64], dxh[64], s2p = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int ch = 8 * w + k;
        const uint4 u = lds128(su + O_Q1 + (ch >> 3) * KB + sw128(r, ch & 7));
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        const float4 c0 = modv(0, 2 * ch), c1 = modv(0, 2 * ch + 1);
        const float scv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = 8 * k + 2 * e;
          xh[i] = bf16lo(w4[e]) * rstd; xh[i + 1] = bf16hi(w4[e]) * rstd;
          dxh[i] = dy[i] * (1.f + scv[2 * e]); dxh[i + 1] = dy[i + 1] * (1.f + scv[2 * e + 1]);
          s2p = fmaf(dxh[i], xh[i], fmaf(dxh[i + 1], xh[i + 1], s2p));
        }
      }
      ssx[w * 128 + r] = s2p;
      named_bar_sync(1, 256);                                              // (also: every MMA of the tile is done -> y is dead)
      const float s2 = (ssx[r] + ssx[128 + r]) * (1.f / C);
      if (ok) {
        uint4* dst = reinterpret_cast<uint4*>(DQ1 + row * C + 64 * w);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int ch = 8 * w + k;
          const uint4 du = lds128(su + O_D + (ch >> 3) * KB + sw128(r, ch & 7));
          const uint32_t d4[4] = {du.x, du.y, du.z, du.w};
          uint32_t o4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = 8 * k + 2 * e;
            o4[e] = pack_bf16(bf16lo(d4[e]) + rstd * (dxh[i] - xh[i] * s2), bf16hi(d4[e]) + rstd * (dxh[i + 1] - xh[i + 1] * s2));
          }
          dst[k] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
      }
      atom_sum(dy, 3 * C, b, s0);                                          // d shift_f
#pragma unroll
      for (int i = 0; i < 64; ++i) dxh[i] = dy[i] * xh[i];
      atom_sum(dxh, 4 * C, b, s0);                                         // d scale_f
      {
        float gfp[64];                                                     // d gate_f partial = dq2 ffn
        const uint4* fp = reinterpret_cast<const uint4*>(FFN + row * C + 64 * w);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int ch = 8 * w + k;
          const uint4 du = lds128(su + O_D + (ch >> 3) * KB + sw128(r, ch & 7));
          const uint4 fu = ok ? __ldg(fp + k) : make_uint4(0u, 0u, 0u, 0u);
          const uint32_t d4[4] = {du.x, du.y, du.z, du.w}, f4[4] = {fu.x, fu.y, fu.z, fu.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { gfp[8 * k + 2 * e] = bf16lo(d4[e]) * bf16lo(f4[e]); gfp[8 * k + 2 * e + 1] = bf16hi(d4[e]) * bf16hi(f4[e]); }
        }
        atom_sum(gfp, 5 * C, b, s0);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.infree);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
