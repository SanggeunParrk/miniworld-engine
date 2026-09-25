// tbwd8.cu — the Transition backward of tbwd.cu (same fusion: DW slice CTAs + DX tile CTAs, recompute of dh / a / b in both roles)
// with RELAXED PRECISION: every tensor-core operand is e4m3 (kind::f8f6f4, fp32 accumulate), the SwiGLU backward runs in packed
// fp32x2 arithmetic with sigmoid(a) = 0.5 tanh(a / 2) + 0.5. SPDX-License-Identifier: Apache-2.0
//
// Quantization (value = q * s, per-tensor dequantization scales in sc[]):
//   sc[0] s_x   xn (the forward stores xn as e4m3; bound-based: |xn| <= max|gamma| sqrt(D - 1) + max|beta|)
//   sc[1] s_wab Wa and Wb (one scale: d_xn = [dA | dB] [Wa; Wb] reduces over both)      sc[2] s_ws  Ws
//   sc[3] s_dy  dy (converted to e4m3 in-kernel from the bf16 input)
//   sc[4] s_h   h (DW role, for dWs)      sc[5] s_dab dA and dB (both roles)
// Weights arrive pre-quantized (quant8.cu): wab_q [1024][128] = [Wa; Wb] (K-major for [a|b], MN-major for d_xn), wst_q [512][128] = Ws^T.
//
// DW CTAs (8 hidden slices x R replicas; pairs of slices multicast each tile's xn_q and bf16 dy):
//   warp 0 loads, warps 8-11 convert dy -> dy_q (then write the partials), warp 1 issues [a|b] = xn Wab_s^T and dh = dy WsT_s^T,
//   warps 4-7 / 12-15 run the gate (h, dA, dB -> e4m3 in shared memory, double-buffered), warp 2 issues dWab += dAB^T xn, dWs += dy^T h.
// DX CTAs (one 128-row tile at a time, eight 64-unit chunks, weights streamed and multicast over the pair):
//   warp 0 loads dy (bf16) + xn_q, later x; warp 3 streams WsT_j / Wab_j; warp 1 issues dh_j, [a|b]_j; the gate writes [dA | dB]_j
//   (e4m3) back into TMEM; warp 2 issues d_xn += [dA | dB]_j Wab_j (A from TMEM); warps 8-11 convert dy -> dy_q one tile ahead and run
//   the LayerNorm backward + residual from d_xn.
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int KB = 16384;                                      // bf16 [128][64] or e4m3 [128][128] tile, 128-B swizzled
// ---- DW role shared memory: weights, NIN input stages (xn_q | dy_q, both e4m3, dy_q published by the DX role), h / dA / dB x2
constexpr int NIN = 4;
constexpr int W_WS = 0, W_WAB = 8192, W_IN = 24576, INS = 32768, IN_DY = 16384;
constexpr int W_H = W_IN + NIN * INS, W_DAB = W_H + 2 * 8192, W_BAR = W_DAB + 2 * KB;
// ---- DX role shared memory
constexpr int NWAB = 3;
constexpr int X_WS = 0, X_WAB = 2 * 8192, X_IN = X_WAB + NWAB * KB, XIS = 65536;  // stage: dy bf16 | xn_q | dy_q  (x over xn_q | dy_q)
constexpr int XI_XQ = 32768, XI_DQ = 49152;
constexpr int X_GAM = X_IN + 2 * XIS, X_BAR = X_GAM + 512;
constexpr int SMEM_BYTES = (W_BAR > X_BAR ? W_BAR : X_BAR) + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");

constexpr int CL = 2;
constexpr uint16_t CL_MASK = (1u << CL) - 1;
constexpr uint32_t I_AB = idesc_e4m3(128, 128), I_DH = idesc_e4m3(128, 64), I_DXN = idesc_e4m3(128, 128, 0, 1);
constexpr uint32_t I_DWAB = idesc_e4m3(128, 128, 1, 1), I_DWS = idesc_e4m3(128, 64, 1, 1);

#ifdef SPAN
__device__ unsigned long long g_spanb[256][2];
#endif
#ifdef TRACE
// DW CTA 0: [event][tile]: 0 w1 xn seen, 1 w1 dy_full seen, 2 w1 issued, 3 gate dhab seen, 4 gate g_empty seen, 5 gate loads done,
// 6 gate done, 7 wgrad g_full seen, 8 wgrad issued, 9 conv stg seen, 10 conv dy_empty seen, 11 conv done
__device__ unsigned long long g_tr8[12][1024];
#define TR8(ev, i) do { if (cta == 0 && (i) < 1024) g_tr8[ev][i] = clock64(); } while (0)
#else
#define TR8(ev, i) do { } while (0)
#endif

struct Par {
  const CUtensorMap *dy, *xq, *x, *wst, *wst32, *wab, *dx, *dyq;
  const float *rstd, *c1, *gamma, *sc;
  float *partab, *parts, *dgbw;
  uint8_t* dyq_g;                  // [M][128] e4m3 dy, written by the DX role (one tile ahead), read by the DW role
  unsigned* flags;                 // [tiles]: == epoch once the tile's dy_q is in global memory
  unsigned epoch;
  int tiles, ndw;
};
struct BarsW {
  uint64_t w_full, xn_full[NIN], dy_full[NIN], in_empty[NIN];
  uint64_t dhab_full, gate_read, g_full[2], g_empty[2], wg_done; uint32_t tmem;
};
DEVI uint32_t ld_acquire_gpu(const unsigned* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEVI void st_release_gpu(unsigned* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
struct BarsX {
  uint64_t ws_full[2], ws_empty[2], wab_full[NWAB], wab_empty[NWAB], in_full[2], in_empty[2], q_full[2], x_full[2], xn_dead[2];
  uint64_t abdh_full[2], g_full[2], ab_free[2], dxn_full, dxn_empty; uint32_t tmem;
};

// bf16 [128][128] (two 64-column K-blocks) -> e4m3 [128][128] of row r, scaled by inv (both 128-B swizzled)
DEVI void convert_row(uint32_t src, uint32_t dst, uint32_t r, f2 inv, uint8_t* gdst = nullptr) {
#pragma unroll
  for (int cb = 0; cb < 2; ++cb)
#pragma unroll
    for (int qp = 0; qp < 4; ++qp) {
      const uint4 v0 = lds128(src + cb * KB + sw128(r, 2 * qp)), v1 = lds128(src + cb * KB + sw128(r, 2 * qp + 1));
      const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
      uint32_t o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        o[k] = e4m3x4(mul2(mk2(bf16lo(w[2 * k]), bf16hi(w[2 * k])), inv), mul2(mk2(bf16lo(w[2 * k + 1]), bf16hi(w[2 * k + 1])), inv));
      sts128(dst + sw128(r, cb * 4 + qp), make_uint4(o[0], o[1], o[2], o[3]));
      if (gdst) stg128(gdst + (cb * 4 + qp) * 16, make_uint4(o[0], o[1], o[2], o[3]));
    }
}

// SwiGLU backward of one pair of hidden units, all in packed fp32x2: returns h (h units), dA, dB (dA / dB units)
struct GateK { f2 ca, cg, ih; };
DEVI void gate_pair(const GateK& K, uint32_t g0, uint32_t g1, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, f2& h, f2& da, f2& db) {
  const f2 A = mul2(mk2u(a0, a1), K.ca), B = mul2(mk2u(b0, b1), K.ca), G = mul2(mk2u(g0, g1), K.cg);
  const f2 s = sigmoid2(A), l = mul2(A, s);
  h = mul2(mul2(l, B), K.ih);
  db = mul2(G, l);
  da = mul2(mul2(G, B), add2(fma2(neg2(l), s, l), s));       // (g b)(s + l (1 - s))
}

// ================================================================================================ DW role
DEVI void weight_role(const Par& p, uint8_t* sm, int cta, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsW& B = *reinterpret_cast<BarsW*>(sm + W_BAR);
  const int slice = cta & 7, repl = cta >> 3, R = p.ndw >> 3;
  const int crank = (int)cluster_rank();
  const int n_local = (p.tiles > repl) ? (p.tiles - repl + R - 1) / R : 0;
  constexpr uint32_t T_DH = 0, T_AB = 64, T_DWAB = 256, T_DWS = 384;
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int s = 0; s < NIN; ++s) { mbar_init(&B.xn_full[s], 1); mbar_init(&B.dy_full[s], 1); mbar_init(&B.in_empty[s], CL); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.g_full[s], 8); mbar_init(&B.g_empty[s], 1); }
    mbar_init(&B.dhab_full, 1); mbar_init(&B.gate_read, 8); mbar_init(&B.wg_done, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.w_full, 8192 + KB);
      tma_load_2d(su + W_WS, p.wst, &B.w_full, 0, slice * HS);
      tma_load_2d(su + W_WAB, p.wab, &B.w_full, 0, slice * HS);
      tma_load_2d(su + W_WAB + 8192, p.wab, &B.w_full, 0, H_ + slice * HS);
      for (int i = 0; i < n_local; ++i) {
        const int b = i % NIN, t = repl + i * R, row = t * ROWS;
        if (i >= NIN) mbar_wait(&B.in_empty[b], ((i / NIN) - 1) & 1);
        // xn_q and (once published) dy_q: this CTA requests one of the two 64-row boxes of each, multicast to both
        mbar_expect_tx(&B.xn_full[b], KB);
        tma_load_2d_mc(su + W_IN + b * INS + crank * 8192, p.xq, &B.xn_full[b], 0, row + crank * 64, CL_MASK);
        TR8(9, i);
        while (ld_acquire_gpu(p.flags + t) != p.epoch) { __nanosleep(32); }
        TR8(10, i);
        fence_proxy_async_global();
        mbar_expect_tx(&B.dy_full[b], KB);
        tma_load_2d_mc(su + W_IN + b * INS + IN_DY + crank * 8192, p.dyq, &B.dy_full[b], 0, row + crank * 64, CL_MASK);
      }
    }
  } else if (warp == 1) {
    mbar_wait(&B.w_full, 0);
    const uint64_t dws = desc_k128(su + W_WS), dwab = desc_k128(su + W_WAB);
    for (int i = 0; i < n_local; ++i) {
      const int b = i % NIN;
      mbar_wait(&B.xn_full[b], (i / NIN) & 1);
      if (i >= 1) mbar_wait(&B.gate_read, (i - 1) & 1);
      if (lane == 0) TR8(0, i);
      tc_fence_after();
      const uint64_t dxn = desc_k128(su + W_IN + b * INS), ddy = desc_k128(su + W_IN + b * INS + IN_DY);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma8_ss(tmem + T_AB, dxn + (uint64_t)(ks * 2), dwab + (uint64_t)(ks * 2), I_AB, ks > 0 ? 1u : 0u);
      }
      __syncwarp();
      mbar_wait(&B.dy_full[b], (i / NIN) & 1);
      if (lane == 0) TR8(1, i);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma8_ss(tmem + T_DH, ddy + (uint64_t)(ks * 2), dws + (uint64_t)(ks * 2), I_DH, ks > 0 ? 1u : 0u);
        tc_commit(&B.dhab_full);
      }
      __syncwarp();
      if (lane == 0) TR8(2, i);
    }
  } else if (warp == 2) {
    for (int k = 0; k < n_local; ++k) {
      const int b = k % NIN, g = k & 1;
      mbar_wait(&B.g_full[g], (k >> 1) & 1);
      if (lane == 0) TR8(7, k);
      tc_fence_after();
      const uint64_t dab = desc_mn128(su + W_DAB + g * KB, KB), dh_ = desc_sw64(su + W_H + g * 8192);
      const uint64_t dxn = desc_mn128(su + W_IN + b * INS, KB), ddy = desc_mn128(su + W_IN + b * INS + IN_DY, KB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)       // [dWa_s; dWb_s] += [dA | dB]^T xn    (K = the tile's rows, 32 per instruction)
          umma8_ss(tmem + T_DWAB, dab + (uint64_t)(ks * 256), dxn + (uint64_t)(ks * 256), I_DWAB, (k > 0 || ks > 0) ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)       // dWs_s += dy^T h
          umma8_ss(tmem + T_DWS, ddy + (uint64_t)(ks * 256), dh_ + (uint64_t)(ks * 128), I_DWS, (k > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.g_empty[g]);
        tc_commit_mc(&B.in_empty[b], CL_MASK);             // the stage is dead in this slice (counted in both CTAs of the pair)
        if (k == n_local - 1) tc_commit(&B.wg_done);
      }
      __syncwarp();
      if (lane == 0) TR8(8, k);
    }
    if (n_local == 0 && elect_one()) mbar_arrive(&B.wg_done);
    __syncwarp();
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int half = warp >= 12 ? 1 : 0;
    GateK K;
    {
      const float s_x = p.sc[0], s_wab = p.sc[1], s_ws = p.sc[2], s_dy = p.sc[3], s_h = p.sc[4], s_dab = p.sc[5];
      K.ca = mk2(s_x * s_wab, s_x * s_wab); K.cg = mk2(s_dy * s_ws / s_dab, s_dy * s_ws / s_dab); K.ih = mk2(1.f / s_h, 1.f / s_h);
    }
    for (int i = 0; i < n_local; ++i) {
      const int g = i & 1;
      mbar_wait(&B.dhab_full, i & 1);
      if (warp == 4 && lane == 0) TR8(3, i);
      tc_fence_after();
      if (i >= 2) mbar_wait(&B.g_empty[g], ((i >> 1) - 1) & 1);
      if (warp == 4 && lane == 0) TR8(4, i);
      uint32_t dh[32], av[32], bv[32];
      tmem_ld32(trow + T_DH + half * 32, dh);
      tmem_ld32(trow + T_AB + half * 32, av);
      tmem_ld32(trow + T_AB + 64 + half * 32, bv);
      tmem_wait_ld();
      tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.gate_read);
      if (warp == 4 && lane == 0) TR8(5, i);
#pragma unroll
      for (int c = 0; c < 2; ++c) {                        // 16 hidden units -> one 16-byte chunk of h, dA and dB each
        uint32_t hw[4], aw[4], bw[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          f2 h0, a0, b0, h1, a1, b1;
          const int k = c * 8 + w * 2;
          gate_pair(K, dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], h0, a0, b0);
          gate_pair(K, dh[2 * k + 2], dh[2 * k + 3], av[2 * k + 2], av[2 * k + 3], bv[2 * k + 2], bv[2 * k + 3], h1, a1, b1);
          hw[w] = e4m3x4(h0, h1); aw[w] = e4m3x4(a0, a1); bw[w] = e4m3x4(b0, b1);
        }
        const int q = half * 2 + c;
        sts128(su + W_H + g * 8192 + sw64(r, q), make_uint4(hw[0], hw[1], hw[2], hw[3]));
        sts128(su + W_DAB + g * KB + sw128(r, q), make_uint4(aw[0], aw[1], aw[2], aw[3]));
        sts128(su + W_DAB + g * KB + sw128(r, 4 + q), make_uint4(bw[0], bw[1], bw[2], bw[3]));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full[g]);
      if (warp == 4 && lane == 0) TR8(6, i);
    }
  } else if (warp >= 8) {
    setmaxnreg_inc<152>();
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
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

DEVI float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
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

// ================================================================================================ DX role
DEVI void input_role(const Par& p, uint8_t* sm, int cta, int ndx, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsX& B = *reinterpret_cast<BarsX*>(sm + X_BAR);
  auto count = [&](int k) { return (p.tiles > k) ? (p.tiles - k + ndx - 1) / ndx : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  const int nch = n_local * NCH;
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * ndx : (n_valid > 0 ? cta + (n_valid - 1) * ndx : 0); };
  const int crank = (int)cluster_rank();
  constexpr uint32_t T_AB = 0, T_DH = 256, T_DXN = 384;
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.ws_full[s], 1); mbar_init(&B.ws_empty[s], CL); mbar_init(&B.in_full[s], 1); mbar_init(&B.in_empty[s], 1);
      mbar_init(&B.q_full[s], 4); mbar_init(&B.x_full[s], 1); mbar_init(&B.xn_dead[s], 1);
      mbar_init(&B.abdh_full[s], 1); mbar_init(&B.g_full[s], 8); mbar_init(&B.ab_free[s], 1);
    }
    for (int s = 0; s < NWAB; ++s) { mbar_init(&B.wab_full[s], 1); mbar_init(&B.wab_empty[s], CL); }
    mbar_init(&B.dxn_full, 1); mbar_init(&B.dxn_empty, 4);
    fence_barrier_init();
  }
  if (tid < 128) reinterpret_cast<float*>(sm + X_GAM)[tid] = p.gamma[tid];
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      auto issue_in = [&](int i) {
        const int b = i & 1, row = tile_of(i) * ROWS;
        if (i >= 2) mbar_wait(&B.in_empty[b], ((i >> 1) - 1) & 1);
        mbar_expect_tx(&B.in_full[b], 32768 + KB);
        const uint32_t dst = su + X_IN + b * XIS;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int cb = 0; cb < 2; ++cb) tma_load_2d(dst + cb * KB + h * 8192, p.dy, &B.in_full[b], cb * 64, row + h * 64);
          tma_load_2d(dst + XI_XQ + h * 8192, p.xq, &B.in_full[b], 0, row + h * 64);
        }
      };
      auto issue_x = [&](int i) {                         // xn_q and dy_q of the tile are dead: x (bf16) over them for the epilogue
        const int b = i & 1, row = tile_of(i) * ROWS;
        mbar_wait(&B.xn_dead[b], (i >> 1) & 1);
        mbar_expect_tx(&B.x_full[b], 32768);
        const uint32_t dst = su + X_IN + b * XIS + XI_XQ;
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
      int cs = 0, ca = 0;
      while (cs < nch || ca < nch) {
        if (cs < nch && (cs < 2 || mbar_test(&B.ws_empty[cs & 1], ((cs >> 1) - 1) & 1))) {
          const int s = cs & 1, j = cs & (NCH - 1);
          mbar_expect_tx(&B.ws_full[s], 8192);              // WsT_j [64][128]: this CTA requests 32 rows, multicast to both
          tma_load_2d_mc(su + X_WS + s * 8192 + crank * 4096, p.wst32, &B.ws_full[s], 0, j * HS + crank * 32, CL_MASK);
          ++cs;
        }
        if (ca < nch && (ca < NWAB || mbar_test(&B.wab_empty[ca % NWAB], ((ca / NWAB) - 1) & 1))) {
          const int s = ca % NWAB, j = ca & (NCH - 1);
          mbar_expect_tx(&B.wab_full[s], KB);               // [Wa_j; Wb_j]: this CTA requests Wa_j (rank 0) or Wb_j (rank 1)
          tma_load_2d_mc(su + X_WAB + s * KB + crank * 8192, p.wab, &B.wab_full[s], 0, crank * H_ + j * HS, CL_MASK);
          ++ca;
        }
      }
    }
  } else if (warp == 1) {
    for (int c = 0; c < nch; ++c) {
      const int i = c >> 3, j = c & (NCH - 1), s = c & 1, u = c >> 1, sw = c % NWAB;
      if (j == 0) { mbar_wait(&B.in_full[i & 1], (i >> 1) & 1); mbar_wait(&B.q_full[i & 1], (i >> 1) & 1); }
      mbar_wait(&B.ws_full[s], u & 1);
      mbar_wait(&B.wab_full[sw], (c / NWAB) & 1);
      if (c >= 2) mbar_wait(&B.ab_free[s], (u - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + X_IN + (i & 1) * XIS;
      const uint64_t dxq = desc_k128(st + XI_XQ), ddq = desc_k128(st + XI_DQ);
      const uint64_t dws = desc_k128(su + X_WS + s * 8192), dwab = desc_k128(su + X_WAB + sw * KB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma8_ss(tmem + T_DH + s * 64, ddq + (uint64_t)(ks * 2), dws + (uint64_t)(ks * 2), I_DH, ks > 0 ? 1u : 0u);
        tc_commit_mc(&B.ws_empty[s], CL_MASK);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma8_ss(tmem + T_AB + s * 128, dxq + (uint64_t)(ks * 2), dwab + (uint64_t)(ks * 2), I_AB, ks > 0 ? 1u : 0u);
        tc_commit(&B.abdh_full[s]);
        if (j == NCH - 1) tc_commit(&B.xn_dead[i & 1]);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // [dA | dB]_j in TMEM: dA(units 0..31) cols 0..7, dB(0..31) 8..15, dA(32..63) 32..39, dB(32..63) 40..47; the matching K rows of the
    // [Wa_j; Wb_j] tile (MN-major view, 128 B per row) start at rows 0, 64, 32, 96
    constexpr uint32_t aoff[4] = {0, 8, 32, 40}, boff[4] = {0, 8192 >> 4, 4096 >> 4, 12288 >> 4};
    for (int q = 0; q < nch; ++q) {
      const int i = q >> 3, j = q & (NCH - 1), s = q & 1, u = q >> 1;
      mbar_wait(&B.g_full[s], u & 1);
      if (j == 0 && i >= 1) mbar_wait(&B.dxn_empty, (i - 1) & 1);
      tc_fence_after();
      const uint64_t dwab = desc_mn128(su + X_WAB + (q % NWAB) * KB, KB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma8_ts(tmem + T_DXN, tmem + T_AB + s * 128 + aoff[ks], dwab + (uint64_t)boff[ks], I_DXN, (j > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.ab_free[s]);
        tc_commit_mc(&B.wab_empty[q % NWAB], CL_MASK);
        if (j == NCH - 1) tc_commit(&B.dxn_full);
      }
      __syncwarp();
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
    setmaxnreg_inc<152>();
    const int half = warp >= 12 ? 1 : 0;
    GateK K;
    {
      const float s_x = p.sc[0], s_wab = p.sc[1], s_ws = p.sc[2], s_dy = p.sc[3], s_dab = p.sc[5];
      K.ca = mk2(s_x * s_wab, s_x * s_wab); K.cg = mk2(s_dy * s_ws / s_dab, s_dy * s_ws / s_dab); K.ih = mk2(1.f, 1.f);
    }
    for (int c = 0; c < nch; ++c) {
      const int s = c & 1, u = c >> 1;
      mbar_wait(&B.abdh_full[s], u & 1);
      tc_fence_after();
      uint32_t dh[32], av[32], bv[32];
      tmem_ld32(trow + T_DH + s * 64 + half * 32, dh);
      tmem_ld32(trow + T_AB + s * 128 + half * 32, av);
      tmem_ld32(trow + T_AB + s * 128 + 64 + half * 32, bv);
      tmem_wait_ld();
      uint32_t aw[8], bw[8];
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        f2 h0, a0, b0, h1, a1, b1;
        const int k = w * 2;
        gate_pair(K, dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], h0, a0, b0);
        gate_pair(K, dh[2 * k + 2], dh[2 * k + 3], av[2 * k + 2], av[2 * k + 3], bv[2 * k + 2], bv[2 * k + 3], h1, a1, b1);
        aw[w] = e4m3x4(a0, a1); bw[w] = e4m3x4(b0, b1);
      }
      tmem_st8(trow + T_AB + s * 128 + half * 32, aw);
      tmem_st8(trow + T_AB + s * 128 + half * 32 + 8, bw);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full[s]);
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------ dy -> dy_q one tile ahead; LayerNorm
    // backward + residual
    setmaxnreg_inc<152>();
    const int t2 = (int)tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gam_u = su + X_GAM;
    const float idy = 1.f / p.sc[3], cdx = p.sc[5] * p.sc[1];
    const f2 inv = mk2(idy, idy);
    auto convert = [&](int i) {                           // dy_q of the tile into shared memory (own MMA) and global memory (DW role)
      const int b = i & 1, t = tile_of(i);
      const bool real = i < n_valid;
      mbar_wait(&B.in_full[b], (i >> 1) & 1);
      const uint32_t st = su + X_IN + b * XIS;
      convert_row(st, st + XI_DQ, r, inv, real ? p.dyq_g + ((size_t)t * ROWS + r) * D_ : nullptr);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.q_full[b]);
      if (real) {
        fence_proxy_async_global();
        __threadfence();
        named_bar_sync(2, 128);
        if (r == 0) st_release_gpu(p.flags + t, p.epoch);
      }
    };
    float accg[4] = {0.f, 0.f, 0.f, 0.f}, accb[4] = {0.f, 0.f, 0.f, 0.f};
    if (n_local > 0) convert(0);
    for (int i = 0; i < n_local; ++i) {
      if (i + 1 < n_local) convert(i + 1);
      const int b = i & 1, grow = tile_of(i) * ROWS + (int)r;
      const bool real = i < n_valid;
      mbar_wait(&B.dxn_full, i & 1);
      tc_fence_after();
      uint32_t dn[64];
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_DXN + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) dn[cc * 16 + k] = pack_bf16(__uint_as_float(v[2 * k]) * cdx, __uint_as_float(v[2 * k + 1]) * cdx);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxn_empty);
      const float rs = p.rstd[grow], mean = p.c1[grow] / rs;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      const uint32_t dyb = su + X_IN + b * XIS, xb = dyb + XI_XQ;
      float pca[4] = {0.f, 0.f, 0.f, 0.f}, pcb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        float work[32];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint4 xv = lds128(xb + (g >> 1) * KB + sw128(r, (g & 1) * 4 + qq));
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int e = qq * 4 + kk, col = g * 32 + 2 * e;
            const uint32_t n = dn[col >> 1];
            const float n0 = bf16lo(n), n1 = bf16hi(n);
            const float x0 = (bf16lo(xw[kk]) - mean) * rs, x1 = (bf16hi(xw[kk]) - mean) * rs;
            const float2 gg = lds64f(gam_u + col * 4);
            const float w0 = gg.x * n0, w1 = gg.y * n1;
            pca[kk] += x0 * w0 + x1 * w1; pcb[kk] += w0 + w1;
            work[2 * e] = n0 * x0; work[2 * e + 1] = n1 * x1;
          }
        }
        const float sg = reduce_scatter32(work, lane);
        if (real) accg[g] += sg;
#pragma unroll
        for (int e = 0; e < 16; ++e) { const uint32_t n = dn[(g * 32 + 2 * e) >> 1]; work[2 * e] = bf16lo(n); work[2 * e + 1] = bf16hi(n); }
        const float sb = reduce_scatter32(work, lane);
        if (real) accb[g] += sb;
      }
      const float ca = ((pca[0] + pca[1]) + (pca[2] + pca[3])) * (1.f / D_);
      const float cbv = ((pcb[0] + pcb[1]) + (pcb[2] + pcb[3])) * (1.f / D_);
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
            const float2 gg = lds64f(gam_u + col * 4);
            const float w0 = gg.x * bf16lo(n), w1 = gg.y * bf16hi(n);
            o[k] = pack_bf16((w0 - (x0 * ca + cbv)) * rs + bf16lo(dw[k]), (w1 - (x1 * ca + cbv)) * rs + bf16hi(dw[k]));
          }
          sts128(dyb + off, make_uint4(o[0], o[1], o[2], o[3]));
        }
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        const int row0 = tile_of(i) * ROWS;
        if (real) {
#pragma unroll
          for (int cb = 0; cb < 2; ++cb)
#pragma unroll
            for (int h = 0; h < 2; ++h) tma_store_2d(p.dx, dyb + cb * KB + h * 8192, cb * 64, row0 + h * 64);
          tma_store_commit();
          tma_store_wait_read0();
        }
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
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __global__ void __launch_bounds__(512, 1)
transition_bwd8_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxq, const __grid_constant__ CUtensorMap mx,
                      const __grid_constant__ CUtensorMap mwst, const __grid_constant__ CUtensorMap mwst32, const __grid_constant__ CUtensorMap mwab,
                      const __grid_constant__ CUtensorMap mdx, const __grid_constant__ CUtensorMap mdyq, const float* __restrict__ rstd,
                      const float* __restrict__ c1, const float* __restrict__ gamma, const float* __restrict__ sc, float* __restrict__ partab,
                      float* __restrict__ parts, float* __restrict__ dgbw, uint8_t* __restrict__ dyq_g, unsigned* __restrict__ flags,
                      const unsigned* __restrict__ epoch, int tiles, int ndw) {
  const Par p{&mdy, &mxq, &mx, &mwst, &mwst32, &mwab, &mdx, &mdyq, rstd, c1, gamma, sc, partab, parts, dgbw, dyq_g, flags, *epoch, tiles, ndw};
  extern __shared__ __align__(1024) uint8_t sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#ifdef SPAN
  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_spanb[blockIdx.x][0] = t; }
#endif
  if (warp < 4) setmaxnreg_dec<56>();
  if ((int)blockIdx.x < ndw) weight_role(p, sm, blockIdx.x, warp, lane);
  else input_role(p, sm, blockIdx.x - ndw, gridDim.x - ndw, warp, lane);
#ifdef SPAN
  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_spanb[blockIdx.x][1] = t; }
#endif
}

// partab [NDW][128][128] (rows 0..63 dWa_s, 64..127 dWb_s), parts [NDW][128 d][64 hs] (raw e4m3-product sums) -> bf16 dWa, dWb [512][128],
// dWs [128][512] with the dequantization scales; dgbw [NDX * 4][256] -> dgamma, dbeta (fp32)
extern "C" __global__ void transition_bwd8_reduce(const float* __restrict__ partab, const float* __restrict__ parts, const float* __restrict__ dgbw,
                                                  const float* __restrict__ sc, __nv_bfloat16* __restrict__ dwa, __nv_bfloat16* __restrict__ dwb,
                                                  __nv_bfloat16* __restrict__ dws, float* __restrict__ dgam, float* __restrict__ dbeta, int ndw, int nrows_dg,
                                                  unsigned* __restrict__ epoch) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, R = ndw >> 3;
  if (idx == 0) *epoch += 1u;                               // the next backward's dy_q flags compare against a fresh value
  if (idx < 2 * H_ * D_) {
    const int which = idx / (H_ * D_), rem = idx % (H_ * D_), slice = rem / (HS * D_), hs = (rem / D_) % HS, d = rem % D_;
    float v = 0.f;
    for (int rr = 0; rr < R; ++rr) v += partab[((size_t)(rr * 8 + slice) * 128 + which * 64 + hs) * 128 + d];
    (which == 0 ? dwa : dwb)[(slice * HS + hs) * D_ + d] = __float2bfloat16_rn(v * (sc[5] * sc[0]));
  } else if (idx < 3 * H_ * D_) {
    const int rem = idx - 2 * H_ * D_, d = rem / H_, hh = rem % H_, slice = hh / HS, hs = hh % HS;
    float v = 0.f;
    for (int rr = 0; rr < R; ++rr) v += parts[((size_t)(rr * 8 + slice) * 128 + d) * 64 + hs];
    dws[d * H_ + hh] = __float2bfloat16_rn(v * (sc[3] * sc[4]));
  } else if (idx < 3 * H_ * D_ + 256) {
    const int c = idx - 3 * H_ * D_;
    float v = 0.f;
    for (int rr = 0; rr < nrows_dg; ++rr) v += dgbw[(size_t)rr * 256 + c];
    (c < 128 ? dgam : dbeta)[c & 127] = v;
  }
}

// weights -> e4m3 with the per-tensor scales in sc[1] (Wa, Wb) and sc[2] (Ws): wab_q [1024][128] = [Wa; Wb], wst_q [512][128] = Ws^T,
// ws_q [128][512] = Ws (the forward's squeeze operand). One element pair per thread.
extern "C" __global__ void quant8_weights(const __nv_bfloat16* __restrict__ wa, const __nv_bfloat16* __restrict__ wb,
                                          const __nv_bfloat16* __restrict__ ws, const float* __restrict__ sc,
                                          uint8_t* __restrict__ wab_q, uint8_t* __restrict__ wst_q, uint8_t* __restrict__ ws_q) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;         // 0 .. 3 * 65536 / 2
  constexpr int NP = H_ * D_ / 2;
  if (idx < 2 * NP) {
    const float inv = 1.f / sc[1];
    const __nv_bfloat16* src = idx < NP ? wa : wb;
    const int e = (idx % NP) * 2;
    const float v0 = __bfloat162float(src[e]) * inv, v1 = __bfloat162float(src[e + 1]) * inv;
    *reinterpret_cast<uint16_t*>(wab_q + (idx < NP ? 0 : H_ * D_) + e) = (uint16_t)e4m3x2(v0, v1);
  } else if (idx < 3 * NP) {
    const float inv = 1.f / sc[2];
    const int e = (idx - 2 * NP) * 2, d = e / H_, n = e % H_;       // ws [d][n], two consecutive n
    const float v0 = __bfloat162float(ws[e]) * inv, v1 = __bfloat162float(ws[e + 1]) * inv;
    const uint32_t q = e4m3x2(v0, v1);
    *reinterpret_cast<uint16_t*>(ws_q + e) = (uint16_t)q;
    wst_q[n * D_ + d] = (uint8_t)(q & 0xff); wst_q[(n + 1) * D_ + d] = (uint8_t)(q >> 8);
  }
}
