// tbwd.cu — the Transition backward (LayerNorm + SwiGLU expand + squeeze + residual) at D = 128, H = 512, bf16, as ONE fused
// sm_100a kernel plus a partial-sum reduction.  SPDX-License-Identifier: Apache-2.0
//
// Contract and schedule are the sm_90a kernel's (experiments/transition_fused/src/transition_bwd.cu):
//   dh = bf16(dy Ws)      a = xn Wa^T, b = xn Wb^T (fp32)      sig = rcp(1 + ex2(-a log2 e))      silu = a sig
//   h  = bf16(silu b)     dA = bf16((dh b)(sig + silu(1 - sig)))      dB = bf16(dh silu)
//   dWs = dy^T h, dWa = dA^T xn, dWb = dB^T xn (fp32 sums, rounded once)      d_xn = bf16(dA Wa + dB Wb)
//   LayerNorm backward from the saved statistics: xhat = (x - mean) rstd, w = gamma d_xn,
//     dx = bf16(bf16((w - xhat ca - cb) rstd) + dy),  ca = mean(xhat w), cb = mean(w);  dgamma += d_xn xhat, dbeta += d_xn
// Two CTA roles in one launch, no cluster, no atomics:
//   DW CTAs (8 hidden slices x R replicas) keep their slice's Ws / Wa / Wb resident, recompute dh / a / b for every tile they own,
//     write h, dA, dB to shared memory and accumulate [dWa_s; dWb_s] (M128 N128) and dWs_s (M128 N64) in tensor memory.
//   DX CTAs stream the eight weight chunks per tile: dh_j and [a|b]_j into TMEM, the gate writes [dA|dB] back into TMEM as bf16, and
//     d_xn accumulates with an A-from-TMEM product over the same packed [Wa_j; Wb_j] tile.  The LayerNorm backward reads d_xn straight
//     from TMEM, one row per thread.
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int KB = 16384;                                      // one K-block: [128 rows][64 bf16], 128-B swizzled
// ---- DW role shared memory
constexpr int W_WS = 0, W_WAB = 16384, W_IN = 49152, INB = 65536, IN_XN = 32768;
constexpr int W_H = W_IN + 2 * INB, W_DAB = W_H + KB, W_BAR = W_DAB + 2 * KB;
// ---- DX role shared memory
constexpr int X_SLOT = 49152, X_WAB = 16384, X_IN = 2 * X_SLOT;
constexpr int X_GAM = X_IN + 2 * INB, X_BAR = X_GAM + 512;
constexpr int SMEM_BYTES = (W_BAR > X_BAR ? W_BAR : X_BAR) + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");

constexpr uint32_t I_DH = idesc_bf16(128, 64, 0, 1), I_AB = idesc_bf16(128, 128), I_DXN = idesc_bf16(128, 128, 0, 1);
constexpr uint32_t I_DWAB = idesc_bf16(128, 128, 1, 1), I_DWS = idesc_bf16(128, 64, 1, 1);

struct Par {
  const CUtensorMap *dy, *xn, *x, *ws, *wa, *wb, *dx;
  const float *rstd, *c1, *gamma;
  float *partab, *parts, *dgbw;
  int tiles, ndw;
};

struct BarsW { uint64_t w_full, in_full[2], in_empty[2], dhab_full, gate_read, g_full, g_empty, wg_done; uint32_t tmem; };
struct BarsX {
  uint64_t w_full[2], w_empty[2], in_full[2], in_empty[2], x_full[2], xn_dead[2], abdh_full[2], g_full[2], ab_free[2];
  uint64_t dxn_full, dxn_empty; uint32_t tmem;
};

DEVI float gate_da(float g, float b, float s, float l) { return (g * b) * (s + l * (1.f - s)); }

// ================================================================================================ DW role
DEVI void weight_role(const Par& p, uint8_t* sm, int cta, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsW& B = *reinterpret_cast<BarsW*>(sm + W_BAR);
  const int slice = cta & 7, repl = cta >> 3, R = p.ndw >> 3;
  const int n_local = (p.tiles > repl) ? (p.tiles - repl + R - 1) / R : 0;
  constexpr uint32_t T_DH = 0, T_AB = 64, T_DWAB = 256, T_DWS = 384;
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int s = 0; s < 2; ++s) { mbar_init(&B.in_full[s], 1); mbar_init(&B.in_empty[s], 1); }
    mbar_init(&B.dhab_full, 1); mbar_init(&B.gate_read, 4); mbar_init(&B.g_full, 4); mbar_init(&B.g_empty, 1); mbar_init(&B.wg_done, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.w_full, 49152);
      tma_load_2d(su + W_WS, p.ws, &B.w_full, slice * HS, 0);
      tma_load_2d(su + W_WS + 8192, p.ws, &B.w_full, slice * HS, 64);
#pragma unroll
      for (int cb = 0; cb < 2; ++cb) {
        tma_load_2d(su + W_WAB + cb * KB, p.wa, &B.w_full, cb * 64, slice * HS);
        tma_load_2d(su + W_WAB + cb * KB + 8192, p.wb, &B.w_full, cb * 64, slice * HS);
      }
      for (int i = 0; i < n_local; ++i) {
        const int b = i & 1, row = (repl + i * R) * ROWS;
        if (i >= 2) mbar_wait(&B.in_empty[b], ((i >> 1) - 1) & 1);
        mbar_expect_tx(&B.in_full[b], INB);
        const uint32_t dst = su + W_IN + b * INB;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            tma_load_2d(dst + cb * KB + h * 8192, p.dy, &B.in_full[b], cb * 64, row + h * 64);
            tma_load_2d(dst + IN_XN + cb * KB + h * 8192, p.xn, &B.in_full[b], cb * 64, row + h * 64);
          }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      auto wgrad = [&](int k) {
        const int b = k & 1;
        mbar_wait(&B.g_full, k & 1);
        tc_fence_after();
        const uint32_t inb = su + W_IN + b * INB;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)       // [dWa_s; dWb_s] += [dA | dB]^T xn    (K = the tile's 128 rows)
          umma_ss(tmem + T_DWAB, desc_mn128(su + W_DAB + ks * 2048, KB), desc_mn128(inb + IN_XN + ks * 2048, KB), I_DWAB, (k > 0 || ks > 0) ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)       // dWs_s += dy^T h
          umma_ss(tmem + T_DWS, desc_mn128(inb + ks * 2048, KB), desc_mn128(su + W_H + ks * 2048, KB), I_DWS, (k > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.g_empty);
        tc_commit(&B.in_empty[b]);
      };
      mbar_wait(&B.w_full, 0);
      for (int i = 0; i < n_local; ++i) {
        const int b = i & 1;
        mbar_wait(&B.in_full[b], (i >> 1) & 1);
        if (i >= 1) mbar_wait(&B.gate_read, (i - 1) & 1);
        tc_fence_after();
        const uint32_t inb = su + W_IN + b * INB;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * KB + (ks & 3) * 32;
          umma_ss(tmem + T_DH, desc_k128(inb + off), desc_mn128(su + W_WS + ks * 2048, KB), I_DH, ks > 0 ? 1u : 0u);
        }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * KB + (ks & 3) * 32;
          umma_ss(tmem + T_AB, desc_k128(inb + IN_XN + off), desc_k128(su + W_WAB + off), I_AB, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.dhab_full);
        if (i >= 1) wgrad(i - 1);
      }
      if (n_local > 0) wgrad(n_local - 1);
      tc_commit(&B.wg_done);
    }
  } else if (warp >= 4 && warp < 8) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    for (int i = 0; i < n_local; ++i) {
      mbar_wait(&B.dhab_full, i & 1);
      tc_fence_after();
      if (i >= 1) mbar_wait(&B.g_empty, (i - 1) & 1);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t dh[32], av[32], bv[32];
        tmem_ld32(trow + T_DH + half * 32, dh);
        tmem_ld32(trow + T_AB + half * 32, av);
        tmem_ld32(trow + T_AB + 64 + half * 32, bv);
        tmem_wait_ld();
        if (half == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.gate_read); }
        uint32_t hp[16], dap[16], dbp[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
          const float g0 = bf16lo(gp), g1 = bf16hi(gp);
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          hp[k] = pack_bf16(l0 * b0, l1 * b1);
          dap[k] = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1));
          dbp[k] = pack_bf16(g0 * l0, g1 * l1);
        }
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint32_t off = sw128(r, half * 4 + qq);
          sts128(su + W_H + off, make_uint4(hp[4 * qq], hp[4 * qq + 1], hp[4 * qq + 2], hp[4 * qq + 3]));
          sts128(su + W_DAB + off, make_uint4(dap[4 * qq], dap[4 * qq + 1], dap[4 * qq + 2], dap[4 * qq + 3]));
          sts128(su + W_DAB + KB + off, make_uint4(dbp[4 * qq], dbp[4 * qq + 1], dbp[4 * qq + 2], dbp[4 * qq + 3]));
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full);
    }
  } else if (warp >= 8) {
    // the CTA's fp32 partials: [dWa_s; dWb_s] rows = TMEM lanes, dWs_s rows (d) = TMEM lanes
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    mbar_wait(&B.wg_done, 0);
    tc_fence_after();
    float* pab = p.partab + ((size_t)cta * 128 + r) * 128;
    float* ps = p.parts + ((size_t)cta * 128 + r) * 64;
#pragma unroll
    for (int cc = 0; cc < 6; ++cc) {
      uint32_t v[32];
      tmem_ld32(trow + (cc < 4 ? T_DWAB + cc * 32 : T_DWS + (cc - 4) * 32), v);
      tmem_wait_ld();
      float* dst = cc < 4 ? pab + cc * 32 : ps + (cc - 4) * 32;
#pragma unroll
      for (int k = 0; k < 8; ++k) *reinterpret_cast<uint4*>(dst + 4 * k) = make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

// warp reduce-scatter: lane l returns the sum over the warp of v[l]
DEVI float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    const bool up = (lane & off) != 0;
#pragma unroll
    for (int k = 0; k < off; ++k) {
      const float send = up ? v[k] : v[k + off], keep = up ? v[k + off] : v[k];
      v[k] = keep + __shfl_xor_sync(0xffffffffu, send, off);
    }
  }
  return v[0];
}

// ================================================================================================ DX role
DEVI void input_role(const Par& p, uint8_t* sm, int cta, int ndx, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsX& B = *reinterpret_cast<BarsX*>(sm + X_BAR);
  const int n_local = (p.tiles > cta) ? (p.tiles - cta + ndx - 1) / ndx : 0;
  const int nch = n_local * NCH;
  constexpr uint32_t T_AB = 0, T_DH = 256, T_DXN = 384;          // AB x2 (128 each; [dA|dB] is written back over its first 64), DH x2 (64 each)
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], 1); mbar_init(&B.in_full[s], 1); mbar_init(&B.in_empty[s], 1);
      mbar_init(&B.x_full[s], 1); mbar_init(&B.xn_dead[s], 1); mbar_init(&B.abdh_full[s], 1); mbar_init(&B.g_full[s], 4);
      mbar_init(&B.ab_free[s], 1);
    }
    mbar_init(&B.dxn_full, 1); mbar_init(&B.dxn_empty, 4);
    fence_barrier_init();
  }
  if (tid < 128) reinterpret_cast<float*>(sm + X_GAM)[tid] = p.gamma[tid];
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      auto issue_in = [&](int i) {
        const int b = i & 1, row = (cta + i * ndx) * ROWS;
        if (i >= 2) mbar_wait(&B.in_empty[b], ((i >> 1) - 1) & 1);
        mbar_expect_tx(&B.in_full[b], INB);
        const uint32_t dst = su + X_IN + b * INB;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            tma_load_2d(dst + cb * KB + h * 8192, p.dy, &B.in_full[b], cb * 64, row + h * 64);
            tma_load_2d(dst + IN_XN + cb * KB + h * 8192, p.xn, &B.in_full[b], cb * 64, row + h * 64);
          }
      };
      auto issue_x = [&](int i) {                         // the tile's xn is dead: reload that half with x for the epilogue
        const int b = i & 1, row = (cta + i * ndx) * ROWS;
        mbar_wait(&B.xn_dead[b], (i >> 1) & 1);
        mbar_expect_tx(&B.x_full[b], 32768);
        const uint32_t dst = su + X_IN + b * INB + IN_XN;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d(dst + cb * KB + h * 8192, p.x, &B.x_full[b], cb * 64, row + h * 64);
      };
      for (int i = 0; i <= n_local; ++i) {
        if (i < n_local) issue_in(i);
        if (i >= 1) issue_x(i - 1);
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      for (int c = 0; c < nch; ++c) {
        const int s = c & 1, u = c >> 1, j = c & (NCH - 1);
        if (c >= 2) mbar_wait(&B.w_empty[s], (u - 1) & 1);
        mbar_expect_tx(&B.w_full[s], X_SLOT);
        const uint32_t slot = su + s * X_SLOT;
        tma_load_2d(slot, p.ws, &B.w_full[s], j * HS, 0);
        tma_load_2d(slot + 8192, p.ws, &B.w_full[s], j * HS, 64);
#pragma unroll
        for (int cb = 0; cb < 2; ++cb) {
          tma_load_2d(slot + X_WAB + cb * KB, p.wa, &B.w_full[s], cb * 64, j * HS);
          tma_load_2d(slot + X_WAB + cb * KB + 8192, p.wb, &B.w_full[s], cb * 64, j * HS);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      auto dxn = [&](int q) {
        const int i = q >> 3, j = q & (NCH - 1), s = q & 1, u = q >> 1;
        mbar_wait(&B.g_full[s], u & 1);
        if (j == 0 && i >= 1) mbar_wait(&B.dxn_empty, (i - 1) & 1);
        tc_fence_after();
        const uint32_t wab = su + s * X_SLOT + X_WAB;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          umma_ts(tmem + T_DXN, tmem + T_AB + s * 128 + ks * 8, desc_mn128(wab + ks * 2048, KB), I_DXN, (j > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.ab_free[s]);
        tc_commit(&B.w_empty[s]);
        if (j == NCH - 1) tc_commit(&B.dxn_full);
      };
      for (int c = 0; c < nch; ++c) {
        const int i = c >> 3, j = c & (NCH - 1), s = c & 1, u = c >> 1;
        if (j == 0) mbar_wait(&B.in_full[i & 1], (i >> 1) & 1);
        mbar_wait(&B.w_full[s], u & 1);
        if (c >= 2) mbar_wait(&B.ab_free[s], (u - 1) & 1);
        tc_fence_after();
        const uint32_t inb = su + X_IN + (i & 1) * INB, slot = su + s * X_SLOT;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * KB + (ks & 3) * 32;
          umma_ss(tmem + T_DH + s * 64, desc_k128(inb + off), desc_mn128(slot + ks * 2048, KB), I_DH, ks > 0 ? 1u : 0u);
        }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * KB + (ks & 3) * 32;
          umma_ss(tmem + T_AB + s * 128, desc_k128(inb + IN_XN + off), desc_k128(slot + X_WAB + off), I_AB, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.abdh_full[s]);
        if (j == NCH - 1) tc_commit(&B.xn_dead[i & 1]);
        if (c > 0) dxn(c - 1);
      }
      if (nch > 0) dxn(nch - 1);
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------ gate: [dA | dB] back into TMEM
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
    for (int c = 0; c < nch; ++c) {
      const int s = c & 1, u = c >> 1;
      mbar_wait(&B.abdh_full[s], u & 1);
      tc_fence_after();
      uint32_t da[16], db[16], da2[16], db2[16];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t dh[32], av[32], bv[32];
        tmem_ld32(trow + T_DH + s * 64 + half * 32, dh);
        tmem_ld32(trow + T_AB + s * 128 + half * 32, av);
        tmem_ld32(trow + T_AB + s * 128 + 64 + half * 32, bv);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
          const float g0 = bf16lo(gp), g1 = bf16hi(gp);
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          const uint32_t pa = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1)), pb = pack_bf16(g0 * l0, g1 * l1);
          if (half == 0) { da[k] = pa; db[k] = pb; } else { da2[k] = pa; db2[k] = pb; }
        }
      }
      tmem_st16(trow + T_AB + s * 128, da);                // [dA | dB] as the K = 128 A operand: dA in columns 0..31, dB in 32..63
      tmem_st16(trow + T_AB + s * 128 + 16, da2);
      tmem_st16(trow + T_AB + s * 128 + 32, db);
      tmem_st16(trow + T_AB + s * 128 + 48, db2);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full[s]);
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------ LayerNorm backward + residual
    const int t2 = (int)tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const float* gam = reinterpret_cast<const float*>(sm + X_GAM);
    float accg[4] = {0.f, 0.f, 0.f, 0.f}, accb[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1, grow = (cta + i * ndx) * ROWS + (int)r;
      mbar_wait(&B.dxn_full, i & 1);
      tc_fence_after();
      uint32_t dn[64];                                     // d_xn rounded once to bf16
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_DXN + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) dn[cc * 16 + k] = pack_bf16(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxn_empty);
      const float rs = p.rstd[grow], mean = p.c1[grow] / rs;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      const uint32_t dyb = su + X_IN + b * INB, xb = dyb + IN_XN;
      // pass 1: the two row sums (the sm_90a quad order: partial l over columns with (col >> 1) & 3 == l) and the dgamma / dbeta terms
      float pca[4] = {0.f, 0.f, 0.f, 0.f}, pcb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        float vg[32], vb[32];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const int q = (g & 1) * 4 + qq;
          const uint4 xv = lds128(xb + (g >> 1) * KB + sw128(r, q));
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int col = g * 32 + qq * 8 + 2 * k;
            const uint32_t n = dn[col >> 1];
            const float n0 = bf16lo(n), n1 = bf16hi(n);
            const float x0 = (bf16lo(xw[k]) - mean) * rs, x1 = (bf16hi(xw[k]) - mean) * rs;
            const float w0 = gam[col] * n0, w1 = gam[col + 1] * n1;
            pca[k] += x0 * w0 + x1 * w1; pcb[k] += w0 + w1;
            vg[qq * 8 + 2 * k] = n0 * x0; vg[qq * 8 + 2 * k + 1] = n1 * x1;
            vb[qq * 8 + 2 * k] = n0; vb[qq * 8 + 2 * k + 1] = n1;
          }
        }
        accg[g] += reduce_scatter32(vg, lane);
        accb[g] += reduce_scatter32(vb, lane);
      }
      const float ca = ((pca[0] + pca[1]) + (pca[2] + pca[3])) * (1.f / D_);
      const float cbv = ((pcb[0] + pcb[1]) + (pcb[2] + pcb[3])) * (1.f / D_);
      // pass 2: dx = bf16(bf16((w - xhat ca - cb) rstd) + dy), written in place over dy for the TMA store
#pragma unroll
      for (int cb = 0; cb < 2; ++cb)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t off = cb * KB + sw128(r, q);
          const uint4 xv = lds128(xb + off), dv = lds128(dyb + off);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, dw[4] = {dv.x, dv.y, dv.z, dv.w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int col = cb * 64 + q * 8 + 2 * k;
            const uint32_t n = dn[col >> 1];
            const float x0 = (bf16lo(xw[k]) - mean) * rs, x1 = (bf16hi(xw[k]) - mean) * rs;
            const float w0 = gam[col] * bf16lo(n), w1 = gam[col + 1] * bf16hi(n);
            const uint32_t t = pack_bf16((w0 - (x0 * ca + cbv)) * rs, (w1 - (x1 * ca + cbv)) * rs);
            o[k] = pack_bf16(bf16lo(t) + bf16lo(dw[k]), bf16hi(t) + bf16hi(dw[k]));
          }
          sts128(dyb + off, make_uint4(o[0], o[1], o[2], o[3]));
        }
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        const int row0 = (cta + i * ndx) * ROWS;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(p.dx, dyb + cb * KB + h * 8192, cb * 64, row0 + h * 64);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.in_empty[b]);
      }
    }
    float* row = p.dgbw + ((size_t)cta * 4 + (warp & 3)) * 256;
#pragma unroll
    for (int g = 0; g < 4; ++g) { row[g * 32 + lane] = accg[g]; row[128 + g * 32 + lane] = accb[g]; }
    if (t2 == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __global__ void __launch_bounds__(384, 1)
transition_bwd_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mx,
                     const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
                     const __grid_constant__ CUtensorMap mdx, const float* __restrict__ rstd, const float* __restrict__ c1,
                     const float* __restrict__ gamma, float* __restrict__ partab, float* __restrict__ parts, float* __restrict__ dgbw,
                     int tiles, int ndw) {
  const Par p{&mdy, &mxn, &mx, &mws, &mwa, &mwb, &mdx, rstd, c1, gamma, partab, parts, dgbw, tiles, ndw};
  extern __shared__ __align__(1024) uint8_t sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if ((int)blockIdx.x < ndw) weight_role(p, sm, blockIdx.x, warp, lane);
  else input_role(p, sm, blockIdx.x - ndw, gridDim.x - ndw, warp, lane);
}

// partab [NDW][128][128] (rows 0..63 dWa_s, 64..127 dWb_s), parts [NDW][128 d][64 hs] -> bf16 dWa, dWb [512][128], dWs [128][512];
// dgbw [NDX * 4][256] -> dgamma, dbeta (fp32)
extern "C" __global__ void transition_bwd_reduce(const float* __restrict__ partab, const float* __restrict__ parts, const float* __restrict__ dgbw,
                                                 __nv_bfloat16* __restrict__ dwa, __nv_bfloat16* __restrict__ dwb, __nv_bfloat16* __restrict__ dws,
                                                 float* __restrict__ dgam, float* __restrict__ dbeta, int ndw, int nrows_dg) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, R = ndw >> 3;
  if (idx < 2 * H_ * D_) {                                  // dWa / dWb element: (which, slice, hs, d)
    const int which = idx / (H_ * D_), rem = idx % (H_ * D_), slice = rem / (HS * D_), hs = (rem / D_) % HS, d = rem % D_;
    float v = 0.f;
    for (int rr = 0; rr < R; ++rr) v += partab[((size_t)(rr * 8 + slice) * 128 + which * 64 + hs) * 128 + d];
    (which == 0 ? dwa : dwb)[(slice * HS + hs) * D_ + d] = __float2bfloat16_rn(v);
  } else if (idx < 3 * H_ * D_) {                           // dWs element (d, slice, hs)
    const int rem = idx - 2 * H_ * D_, d = rem / H_, hh = rem % H_, slice = hh / HS, hs = hh % HS;
    float v = 0.f;
    for (int rr = 0; rr < R; ++rr) v += parts[((size_t)(rr * 8 + slice) * 128 + d) * 64 + hs];
    dws[d * H_ + hh] = __float2bfloat16_rn(v);
  } else if (idx < 3 * H_ * D_ + 256) {
    const int c = idx - 3 * H_ * D_;
    float v = 0.f;
    for (int rr = 0; rr < nrows_dg; ++rr) v += dgbw[(size_t)rr * 256 + c];
    (c < 128 ? dgam : dbeta)[c & 127] = v;
  }
}
