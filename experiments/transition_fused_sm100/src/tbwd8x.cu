// tbwd8x.cu — the relaxed-precision (e4m3) Transition backward WITHOUT the DX role's recompute (algorithm change): the DW slice CTAs
// are tbwd8's (recompute dh / a / b of their slice, gate -> h, dA, dB in e4m3, dWab / dWs) and additionally publish the e4m3 [dA | dB]
// block of every (tile, slice) to global memory (16 KB, TMA store + flag). The DX CTAs only form d_xn = [dA | dB] [Wa; Wb] from the
// published blocks (A from shared memory, K = 1024 in 64-unit chunks, weights streamed) and run the LayerNorm backward + residual.
// Tensor work 22 -> 16 M D H; one gate per hidden unit instead of two. dy stays bf16: the DW products with dy (dh = dy Ws_s,
// dWs_s += dy^T h, h in bf16) run as kind::f16, the others (a | b, dWab, d_xn) in e4m3. Scales as tbwd8.cu (s_dy, s_h unused).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int KB = 16384;                                      // bf16 [128][64] or e4m3 [128][128] tile, 128-B swizzled
// ---- DW role shared memory: Ws_s (bf16, MN-major), [Wa_s; Wb_s] (e4m3), NIN input stages (xn_q e4m3 | dy bf16), h (bf16) / dA | dB x2
constexpr int NIN = 2;
constexpr int W_WS = 0, W_WAB = 16384, W_IN = 32768, INS = 49152, IN_DY = 16384;
constexpr int W_H = W_IN + NIN * INS, W_DAB = W_H + 2 * KB, W_BAR = W_DAB + 2 * KB;
// ---- DX role shared memory: [Wa_j; Wb_j] ring (multicast over the pair), [dA | dB] block ring, 2 stages of dy | x (bf16)
constexpr int NWAB = 2, NDAB = 3;
constexpr int X_WAB = 0, X_DAB = NWAB * KB, X_IN = X_DAB + NDAB * KB, XIS = 65536, XI_X = 32768;
constexpr int X_GAM = X_IN + 2 * XIS, X_RED = X_GAM + 512, X_BAR = X_RED + 2048;   // X_RED: per-row partial sums of the 2 groups
constexpr int SMEM_BYTES = (W_BAR > X_BAR ? W_BAR : X_BAR) + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");

constexpr int CL = 2;
constexpr uint16_t CL_MASK = (1u << CL) - 1;
constexpr uint32_t I_AB = idesc_e4m3(128, 128), I_DH = idesc_e4m3(128, 64), I_DXN = idesc_e4m3(128, 128, 0, 1);
constexpr uint32_t I_DWAB = idesc_e4m3(128, 128, 1, 1), I_DWS = idesc_e4m3(128, 64, 1, 1);
constexpr uint32_t I_DH16 = idesc_bf16(128, 64, 0, 1), I_DWS16 = idesc_bf16(128, 64, 1, 1);

#ifdef SPAN
__device__ unsigned long long g_spanb[256][2];
#endif
#ifdef DBG
__device__ unsigned* g_dbg;                                    // host-mapped progress counters [148][8]
#define PROG(k, v) do { if (g_dbg) *(volatile unsigned*)(g_dbg + blockIdx.x * 16 + (k)) = (unsigned)(v) + 1u; } while (0)
#else
#define PROG(k, v) do { } while (0)
#endif
#ifdef TRACE
// DW CTA 0: [event][tile]: 0 w1 xn seen, 1 w1 dy_full seen, 2 w1 issued, 3 gate dhab seen, 4 gate g_empty seen, 5 gate loads done,
// 6 gate done, 7 wgrad g_full seen, 8 wgrad issued, 9 conv stg seen, 10 conv dy_empty seen, 11 conv done
__device__ unsigned long long g_tr8[12][1024];
#define TR8(ev, i) do { if (cta == 0 && (i) < 1024) g_tr8[ev][i] = clock64(); } while (0)
// first DX CTA: [event][chunk or tile]: 0 dab flag wait start, 1 flag seen, 2 mma chunk wait start, 3 mma chunk issued,
// 4 epi dxn_full seen (tile), 5 epi pre-store (tile), 6 epi in_empty arrived (tile), 7 converter tile done, 8 wait dab_full done (mma)
__device__ unsigned long long g_trx[9][1024];
#define TRX(ev, i) do { if (cta == 0 && (i) < 1024) g_trx[ev][i] = clock64(); } while (0)
#else
#define TR8(ev, i) do { } while (0)
#define TRX(ev, i) do { } while (0)
#endif

struct Par {
  const CUtensorMap *dy, *xq, *x, *wst, *wst32, *wab, *dx, *dyq;
  const float *rstd, *c1, *gamma, *sc;
  float *partab, *parts, *dgbw;
  uint8_t* dyq_g;                  // [M][128] e4m3 dy, written by the DX role (one tile ahead), read by the DW role
  unsigned* flags;                 // [tiles]: == epoch once the tile's dy_q is in global memory
  const CUtensorMap* dab;          // [tiles * 8 * 128][128] e4m3: the (tile, slice) [dA | dB] blocks
  const __nv_bfloat16* dyg;        // dy (bf16) for the converter warps
  const uint8_t* dab_g;
  unsigned* dflags;                // [tiles * 8]: == epoch once the (tile, slice) block is in global memory
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
  uint64_t wab_full[NWAB], wab_empty[NWAB], dab_full[NDAB], dab_empty[NDAB], in_full[2], in_empty[2], dxn_full[2], dxn_empty[2];
  uint32_t tmem;
};
DEVI void discard_l2(const void* p) { asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory"); }

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
    for (int s = 0; s < 2; ++s) { mbar_init(&B.g_full[s], 8); mbar_init(&B.g_empty[s], 2); }   // wgrad commit + store read
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
#ifdef PDL
      pdl_wait();                                          // xn_q comes from the forward
#endif
      mbar_expect_tx(&B.w_full, KB + KB);
      tma_load_2d(su + W_WS, p.wst, &B.w_full, slice * HS, 0);          // p.wst: Ws bf16 [128 d][512], box 64 x 64
      tma_load_2d(su + W_WS + 8192, p.wst, &B.w_full, slice * HS, 64);
      tma_load_2d(su + W_WAB, p.wab, &B.w_full, 0, slice * HS);
      tma_load_2d(su + W_WAB + 8192, p.wab, &B.w_full, 0, H_ + slice * HS);
      for (int i = 0; i < n_local; ++i) {
        const int b = i % NIN, t = repl + i * R, row = t * ROWS;
        if (i >= NIN) mbar_wait(&B.in_empty[b], ((i / NIN) - 1) & 1);
        // xn_q and (once published) dy_q: this CTA requests one of the two 64-row boxes of each, multicast to both
        mbar_expect_tx(&B.xn_full[b], KB);
        tma_load_2d_mc(su + W_IN + b * INS + crank * 8192, p.xq, &B.xn_full[b], 0, row + crank * 64, CL_MASK);
        TR8(9, i);
        PROG(0, i);
        mbar_expect_tx(&B.dy_full[b], 32768);               // dy bf16: two of the four 8 KB boxes each, multicast
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int bx = crank * 2 + k, cb = bx >> 1, h = bx & 1;
          tma_load_2d_mc(su + W_IN + b * INS + IN_DY + cb * KB + h * 8192, p.dy, &B.dy_full[b], cb * 64, row + h * 64, CL_MASK);
        }
      }
    }
  } else if (warp == 1) {
    mbar_wait(&B.w_full, 0);
    const uint64_t dws = desc_mn128(su + W_WS, KB), dwab = desc_k128(su + W_WAB);
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
        for (int ks = 0; ks < 8; ++ks)
          umma_ss(tmem + T_DH, ddy + (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4), dws + (uint64_t)(ks * 2048 >> 4), I_DH16, ks > 0 ? 1u : 0u);
        tc_commit(&B.dhab_full);
      }
      __syncwarp();
      if (lane == 0) TR8(2, i);
      if (lane == 0) PROG(1, i);
    }
  } else if (warp == 2) {
    for (int k = 0; k < n_local; ++k) {
      const int b = k % NIN, g = k & 1;
      mbar_wait(&B.g_full[g], (k >> 1) & 1);
      if (lane == 0) TR8(7, k);
      tc_fence_after();
      const uint64_t dab = desc_mn128(su + W_DAB + g * KB, KB), dh_ = desc_mn128(su + W_H + g * KB, KB);
      const uint64_t dxn = desc_mn128(su + W_IN + b * INS, KB), ddy = desc_mn128(su + W_IN + b * INS + IN_DY, KB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)       // [dWa_s; dWb_s] += [dA | dB]^T xn    (K = the tile's rows, 32 per instruction)
          umma8_ss(tmem + T_DWAB, dab + (uint64_t)(ks * 256), dxn + (uint64_t)(ks * 256), I_DWAB, (k > 0 || ks > 0) ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)       // dWs_s += dy^T h   (bf16, K = 16 rows per instruction)
          umma_ss(tmem + T_DWS, ddy + (uint64_t)(ks * 2048 >> 4), dh_ + (uint64_t)(ks * 2048 >> 4), I_DWS16, (k > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.g_empty[g]);
        tc_commit_mc(&B.in_empty[b], CL_MASK);             // the stage is dead in this slice (counted in both CTAs of the pair)
        if (k == n_local - 1) tc_commit(&B.wg_done);
      }
      __syncwarp();
      if (lane == 0) TR8(8, k);
      if (lane == 0) PROG(3, k);
    }
    if (n_local == 0 && elect_one()) mbar_arrive(&B.wg_done);
    __syncwarp();
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int half = warp >= 12 ? 1 : 0;
    GateK K;
    {
      const float s_x = p.sc[0], s_wab = p.sc[1], s_dab = p.sc[5];
      K.ca = mk2(s_x * s_wab, s_x * s_wab); K.cg = mk2(1.f / s_dab, 1.f / s_dab); K.ih = mk2(1.f, 1.f);
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
      for (int c = 0; c < 2; ++c) {                        // 16 hidden units -> two 16-byte chunks of bf16 h, one of e4m3 dA and dB each
        uint32_t hw[8], aw[4], bw[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          f2 h0, a0, b0, h1, a1, b1;
          const int k = c * 8 + w * 2;
          gate_pair(K, dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], h0, a0, b0);
          gate_pair(K, dh[2 * k + 2], dh[2 * k + 3], av[2 * k + 2], av[2 * k + 3], bv[2 * k + 2], bv[2 * k + 3], h1, a1, b1);
          hw[2 * w] = pack_bf16(lo2(h0), hi2(h0)); hw[2 * w + 1] = pack_bf16(lo2(h1), hi2(h1));
          aw[w] = e4m3x4(a0, a1); bw[w] = e4m3x4(b0, b1);
        }
        const int q = half * 2 + c;
        sts128(su + W_H + g * KB + sw128(r, 2 * q), make_uint4(hw[0], hw[1], hw[2], hw[3]));
        sts128(su + W_H + g * KB + sw128(r, 2 * q + 1), make_uint4(hw[4], hw[5], hw[6], hw[7]));
        sts128(su + W_DAB + g * KB + sw128(r, q), make_uint4(aw[0], aw[1], aw[2], aw[3]));
        sts128(su + W_DAB + g * KB + sw128(r, 4 + q), make_uint4(bw[0], bw[1], bw[2], bw[3]));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full[g]);
      if (warp == 4 && lane == 0) TR8(6, i);
      if (warp == 4 && lane == 0) PROG(2, i);
    }
  } else if (warp >= 8) {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    if (warp == 8 && lane == 0) {
      // publish each tile's [dA | dB] block: store, release the buffer once read, raise the flag once written (one tile behind)
      int prev = -1;
      for (int i = 0; i < n_local; ++i) {
        const int g = i & 1, t = repl + i * R;
        mbar_wait(&B.g_full[g], (i >> 1) & 1);
        const int row = (t * 8 + slice) * ROWS;
        tma_store_2d(p.dab, su + W_DAB + g * KB, 0, row);
        tma_store_2d(p.dab, su + W_DAB + g * KB + 8192, 0, row + 64);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.g_empty[g]);
        PROG(4, i);
        if (prev >= 0) {
          asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
          fence_proxy_async_global(); __threadfence();
          st_release_gpu(p.dflags + prev * 8 + slice, p.epoch);
        }
        prev = t;
      }
      if (prev >= 0) {
        tma_store_wait0();
        fence_proxy_async_global(); __threadfence();
        st_release_gpu(p.dflags + prev * 8 + slice, p.epoch);
      }
    }
    __syncwarp();
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
// dy (bf16) -> e4m3 of row r, straight to global memory (the DW role's dy_q)
DEVI void convert_row_g(uint32_t src, uint32_t r, f2 inv, uint8_t* gdst) {
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
      stg128(gdst + (cb * 4 + qp) * 16, make_uint4(o[0], o[1], o[2], o[3]));
    }
}

DEVI void input_role(const Par& p, uint8_t* sm, int cta, int ndx, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsX& B = *reinterpret_cast<BarsX*>(sm + X_BAR);
  auto count = [&](int k) { return (p.tiles > k) ? (p.tiles - k + ndx - 1) / ndx : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  const int nch = n_local * NCH;
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * ndx : (n_valid > 0 ? cta + (n_valid - 1) * ndx : 0); };
  const int crank = (int)cluster_rank();
  constexpr uint32_t T_DXN = 0;                                // d_xn x2 (128 columns each)
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.in_full[s], 1); mbar_init(&B.in_empty[s], 1); mbar_init(&B.dxn_full[s], 1); mbar_init(&B.dxn_empty[s], 8);
    }
    for (int s = 0; s < NWAB; ++s) { mbar_init(&B.wab_full[s], 1); mbar_init(&B.wab_empty[s], CL); }
    for (int s = 0; s < NDAB; ++s) { mbar_init(&B.dab_full[s], 1); mbar_init(&B.dab_empty[s], 1); }
    fence_barrier_init();
  }
  if (tid < 128) reinterpret_cast<float*>(sm + X_GAM)[tid] = p.gamma[tid];
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {                    // dy and x (bf16) of each tile
        const int b = i & 1, row = tile_of(i) * ROWS;
        if (i >= 2) mbar_wait(&B.in_empty[b], ((i >> 1) - 1) & 1);
        PROG(0, i);
        mbar_expect_tx(&B.in_full[b], 65536);
        const uint32_t dst = su + X_IN + b * XIS;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            tma_load_2d(dst + cb * KB + h * 8192, p.dy, &B.in_full[b], cb * 64, row + h * 64);
            tma_load_2d(dst + XI_X + cb * KB + h * 8192, p.x, &B.in_full[b], cb * 64, row + h * 64);
          }
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {                                         // [Wa_j; Wb_j]: this CTA requests Wa_j (rank 0) or Wb_j (rank 1), multicast
      for (int ca = 0; ca < nch; ++ca) {
        const int s = ca % NWAB, j = ca & (NCH - 1);
        if (ca >= NWAB) mbar_wait(&B.wab_empty[s], ((ca / NWAB) - 1) & 1);
        PROG(1, ca);
        mbar_expect_tx(&B.wab_full[s], KB);
        tma_load_2d_mc(su + X_WAB + s * KB + crank * 8192, p.wab, &B.wab_full[s], 0, crank * H_ + j * HS, CL_MASK);
      }
    }
  } else if (warp == 4) {
    if (lane == 0) {                                         // the (tile, chunk) [dA | dB] blocks, once published
      for (int c = 0; c < nch; ++c) {
        const int s = c % NDAB, i = c >> 3, j = c & (NCH - 1), t = tile_of(i);
        if (c >= NDAB) mbar_wait(&B.dab_empty[s], ((c / NDAB) - 1) & 1);
        PROG(2, c);
        TRX(0, c);
        while (ld_acquire_gpu(p.dflags + t * 8 + j) != p.epoch) { __nanosleep(32); }
        TRX(1, c);
        PROG(6, c);
        fence_proxy_async_global();
        mbar_expect_tx(&B.dab_full[s], KB);
        const int row = (t * 8 + j) * ROWS;
        tma_load_2d(su + X_DAB + s * KB, p.dab, &B.dab_full[s], 0, row);
        tma_load_2d(su + X_DAB + s * KB + 8192, p.dab, &B.dab_full[s], 0, row + 64);
      }
    }
  } else if (warp == 5) {
    // once a block has landed in shared memory its lines are dead in L2: drop them without write-back (last tile only once, dummies skip)
    for (int c = 0; c < nch; ++c) {
      const int s = c % NDAB, i = c >> 3, j = c & (NCH - 1);
      mbar_wait(&B.dab_full[s], (c / NDAB) & 1);
#ifdef NO_DISCARD
      if (false) {
#else
      if (i < n_valid) {
#endif
        const uint8_t* g = p.dab_g + (size_t)((tile_of(i) * 8 + j) * ROWS) * D_;
#pragma unroll
        for (int l = 0; l < 4; ++l) discard_l2(g + (size_t)(lane * 4 + l) * 128);
      }
    }
  } else if (warp == 1) {
    // d_xn += [dA | dB]_j [Wa_j; Wb_j]: A K-major (128 B = dA 64 | dB 64 per row), B the MN-major view of the weight tile
    for (int c = 0; c < nch; ++c) {
      const int i = c >> 3, j = c & (NCH - 1), sd = c % NDAB, sw = c % NWAB, e = i & 1;
      if (lane == 0) PROG(7, c * 10 + 1);
      if (lane == 0) TRX(2, c);
      mbar_wait(&B.dab_full[sd], (c / NDAB) & 1);
      if (lane == 0) TRX(8, c);
      if (lane == 0) PROG(7, c * 10 + 2);
      mbar_wait(&B.wab_full[sw], (c / NWAB) & 1);
      if (lane == 0) PROG(7, c * 10 + 3);
      if (j == 0 && i >= 2) mbar_wait(&B.dxn_empty[e], ((i >> 1) - 1) & 1);
      if (lane == 0) PROG(7, c * 10 + 4);
      tc_fence_after();
      const uint64_t da = desc_k128(su + X_DAB + sd * KB), dw = desc_mn128(su + X_WAB + sw * KB, KB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma8_ss(tmem + T_DXN + e * 128, da + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 256), I_DXN, (j > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.dab_empty[sd]);
        tc_commit_mc(&B.wab_empty[sw], CL_MASK);
        if (j == NCH - 1) tc_commit(&B.dxn_full[e]);
      }
      __syncwarp();
      if (lane == 0) PROG(3, c);
      if (lane == 0) TRX(3, c);
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------ LayerNorm backward + residual: warps 8-11
    // columns 0..63 (K-block 0 of the bf16 tiles), warps 12-15 columns 64..127; one row per thread, row sums exchanged through X_RED
    setmaxnreg_inc<152>();
#ifdef PDL
    pdl_wait();                                              // rstd / c1 come from the forward
#endif
    const int G = warp >= 12 ? 1 : 0, t2 = (int)tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gam_u = su + X_GAM + G * 256, red = su + X_RED;
    const float cdx = p.sc[5] * p.sc[1];
    float accg[2] = {0.f, 0.f}, accb[2] = {0.f, 0.f};
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1, e = i & 1, grow = tile_of(i) * ROWS + (int)r;
      const bool real = i < n_valid;
      mbar_wait(&B.dxn_full[e], (i >> 1) & 1);
      if (r == 0 && G == 0) TRX(4, i);
      tc_fence_after();
      uint32_t dn[32];                                       // this group's 64 columns of d_xn, bf16 pairs
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_DXN + e * 128 + G * 64 + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) dn[cc * 16 + k] = pack_bf16(__uint_as_float(v[2 * k]) * cdx, __uint_as_float(v[2 * k + 1]) * cdx);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxn_empty[e]);
      const float rs = p.rstd[grow], mean = p.c1[grow] / rs;
      mbar_wait(&B.in_full[b], (i >> 1) & 1);
      const uint32_t dyb = su + X_IN + b * XIS + G * KB, xb = su + X_IN + b * XIS + XI_X + G * KB;
      float pca[4] = {0.f, 0.f, 0.f, 0.f}, pcb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        float work[32];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint4 xv = lds128(xb + sw128(r, g * 4 + qq));
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int e2 = qq * 4 + kk, col = g * 32 + 2 * e2;   // local column (0..63)
            const uint32_t n = dn[col >> 1];
            const float n0 = bf16lo(n), n1 = bf16hi(n);
            const float x0 = (bf16lo(xw[kk]) - mean) * rs, x1 = (bf16hi(xw[kk]) - mean) * rs;
            const float2 gg = lds64f(gam_u + col * 4);
            const float w0 = gg.x * n0, w1 = gg.y * n1;
            pca[kk] += x0 * w0 + x1 * w1; pcb[kk] += w0 + w1;
            work[2 * e2] = n0 * x0; work[2 * e2 + 1] = n1 * x1;
          }
        }
        const float sg = reduce_scatter32(work, lane);
        if (real) accg[g] += sg;
#pragma unroll
        for (int e2 = 0; e2 < 16; ++e2) { const uint32_t n = dn[(g * 32 + 2 * e2) >> 1]; work[2 * e2] = bf16lo(n); work[2 * e2 + 1] = bf16hi(n); }
        const float sb = reduce_scatter32(work, lane);
        if (real) accb[g] += sb;
      }
      float sa = (pca[0] + pca[1]) + (pca[2] + pca[3]), sbv = (pcb[0] + pcb[1]) + (pcb[2] + pcb[3]);
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(red + (G * 128 + r) * 8), "f"(sa), "f"(sbv) : "memory");
      named_bar_sync(1, 256);
      const float2 o2 = lds64f(red + ((G ^ 1) * 128 + r) * 8);
      const float ca = (G == 0 ? sa + o2.x : o2.x + sa) * (1.f / D_), cbv = (G == 0 ? sbv + o2.y : o2.y + sbv) * (1.f / D_);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t off = sw128(r, q);
        const uint4 xv = lds128(xb + off), dv = lds128(dyb + off);
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, dw[4] = {dv.x, dv.y, dv.z, dv.w};
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int col = q * 8 + 2 * k;
          const uint32_t n = dn[col >> 1];
          const float x0 = (bf16lo(xw[k]) - mean) * rs, x1 = (bf16hi(xw[k]) - mean) * rs;
          const float2 gg = lds64f(gam_u + col * 4);
          const float w0 = gg.x * bf16lo(n), w1 = gg.y * bf16hi(n);
          o[k] = pack_bf16((w0 - (x0 * ca + cbv)) * rs + bf16lo(dw[k]), (w1 - (x1 * ca + cbv)) * rs + bf16hi(dw[k]));
        }
        sts128(dyb + off, make_uint4(o[0], o[1], o[2], o[3]));
      }
      if (r == 0 && G == 0) TRX(5, i);
      fence_proxy_async();
      named_bar_sync(1, 256);
      if (t2 == 0) {
        const int row0 = tile_of(i) * ROWS;
        const uint32_t base = su + X_IN + b * XIS;
        if (real) {
#pragma unroll
          for (int cb = 0; cb < 2; ++cb)
#pragma unroll
            for (int h = 0; h < 2; ++h) tma_store_2d(p.dx, base + cb * KB + h * 8192, cb * 64, row0 + h * 64);
          tma_store_commit();
          tma_store_wait_read0();
        }
        mbar_arrive(&B.in_empty[b]);
        TRX(6, i);
      }
    }
    float* row = p.dgbw + ((size_t)cta * 4 + (warp & 3)) * 256;
#pragma unroll
    for (int g = 0; g < 2; ++g) { row[G * 64 + g * 32 + lane] = accg[g]; row[128 + G * 64 + g * 32 + lane] = accb[g]; }
    if (t2 == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}

extern "C" __global__ void __launch_bounds__(512, 1)
transition_bwd8x_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxq, const __grid_constant__ CUtensorMap mx,
                      const __grid_constant__ CUtensorMap mwst, const __grid_constant__ CUtensorMap mwst32, const __grid_constant__ CUtensorMap mwab,
                      const __grid_constant__ CUtensorMap mdx, const __grid_constant__ CUtensorMap mdyq, const float* __restrict__ rstd,
                      const float* __restrict__ c1, const float* __restrict__ gamma, const float* __restrict__ sc, float* __restrict__ partab,
                      float* __restrict__ parts, float* __restrict__ dgbw, uint8_t* __restrict__ dyq_g, unsigned* __restrict__ flags,
                      const unsigned* __restrict__ epoch, int tiles, int ndw, const __grid_constant__ CUtensorMap mdab,
                      const uint8_t* __restrict__ dab_g, unsigned* __restrict__ dflags, const __nv_bfloat16* __restrict__ dyg) {
  const Par p{&mdy, &mxq, &mx, &mwst, &mwst32, &mwab, &mdx, &mdyq, rstd, c1, gamma, sc, partab, parts, dgbw, dyq_g, flags, &mdab, dyg, dab_g,
              dflags, *epoch, tiles, ndw};
  extern __shared__ __align__(1024) uint8_t sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#ifdef PDL
  pdl_launch();
#endif
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

// partab [NDW][128][128] (rows 0..63 dWa_s, 64..127 dWb_s), parts [NDW][128 d][64 hs] -> bf16 dWa, dWb [512][128] (x s_dab s_x),
// dWs [128][512] (true-valued); dgbw [NDX * 4][256] -> dgamma, dbeta (fp32). Blocks 0 .. 767: one weight-gradient element per thread;
// blocks 768 .. 799: eight dgamma / dbeta columns per block, one per warp, rows strided over the lanes and reduced with shuffles.
extern "C" __global__ void transition_bwd8_reduce(const float* __restrict__ partab, const float* __restrict__ parts, const float* __restrict__ dgbw,
                                                  const float* __restrict__ sc, __nv_bfloat16* __restrict__ dwa, __nv_bfloat16* __restrict__ dwb,
                                                  __nv_bfloat16* __restrict__ dws, float* __restrict__ dgam, float* __restrict__ dbeta, int ndw, int nrows_dg,
                                                  unsigned* __restrict__ epoch) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, R = ndw >> 3;
#ifdef PDL
  pdl_wait();
#endif
  if (idx == 0) *epoch += 1u;                               // the next backward's flags compare against a fresh value
  if (blockIdx.x < 768) {
    float v0 = 0.f, v1 = 0.f;
    if (idx < 2 * H_ * D_) {
      const int which = idx / (H_ * D_), rem = idx % (H_ * D_), slice = rem / (HS * D_), hs = (rem / D_) % HS, d = rem % D_;
      const float* src = partab + ((size_t)slice * 128 + which * 64 + hs) * 128 + d;
      int rr = 0;
      for (; rr + 1 < R; rr += 2) { v0 += src[(size_t)(rr * 8) * 16384]; v1 += src[(size_t)((rr + 1) * 8) * 16384]; }
      if (rr < R) v0 += src[(size_t)(rr * 8) * 16384];
      (which == 0 ? dwa : dwb)[(slice * HS + hs) * D_ + d] = __float2bfloat16_rn((v0 + v1) * (sc[5] * sc[0]));
    } else {
      const int rem = idx - 2 * H_ * D_, d = rem / H_, hh = rem % H_, slice = hh / HS, hs = hh % HS;
      const float* src = parts + ((size_t)slice * 128 + d) * 64 + hs;
      int rr = 0;
      for (; rr + 1 < R; rr += 2) { v0 += src[(size_t)(rr * 8) * 8192]; v1 += src[(size_t)((rr + 1) * 8) * 8192]; }
      if (rr < R) v0 += src[(size_t)(rr * 8) * 8192];
      dws[d * H_ + hh] = __float2bfloat16_rn(v0 + v1);       // bf16 dy x bf16 h: true-valued
    }
  } else {
    const int c = (blockIdx.x - 768) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float v = 0.f;
    for (int rr = lane; rr < nrows_dg; rr += 32) v += dgbw[(size_t)rr * 256 + c];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) (c < 128 ? dgam : dbeta)[c & 127] = v;
  }
}

// weights -> e4m3 with the per-tensor scales in sc[1] (Wa, Wb) and sc[2] (Ws): wab_q [1024][128] = [Wa; Wb], wst_q [512][128] = Ws^T,
// ws_q [128][512] = Ws (the forward's squeeze operand). Threads 0 .. 16383: 8 elements of [Wa; Wb]; 16384 .. 24575: 8 of Ws;
// 24576 .. 40959: 4 consecutive d of one n of Ws^T.
DEVI uint2 q8(uint4 v, float inv) {
  return make_uint2(e4m3x4(mul2(mk2(bf16lo(v.x), bf16hi(v.x)), mk2(inv, inv)), mul2(mk2(bf16lo(v.y), bf16hi(v.y)), mk2(inv, inv))),
                    e4m3x4(mul2(mk2(bf16lo(v.z), bf16hi(v.z)), mk2(inv, inv)), mul2(mk2(bf16lo(v.w), bf16hi(v.w)), mk2(inv, inv))));
}
extern "C" __global__ void quant8_weights(const __nv_bfloat16* __restrict__ wa, const __nv_bfloat16* __restrict__ wb,
                                          const __nv_bfloat16* __restrict__ ws, const float* __restrict__ sc,
                                          uint8_t* __restrict__ wab_q, uint8_t* __restrict__ wst_q, uint8_t* __restrict__ ws_q) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 16384) {
    const int e = idx * 8;                                   // element of [Wa; Wb] (row-major, 1024 x 128)
    const __nv_bfloat16* src = e < H_ * D_ ? wa + e : wb + (e - H_ * D_);
    *reinterpret_cast<uint2*>(wab_q + e) = q8(*reinterpret_cast<const uint4*>(src), 1.f / sc[1]);
  } else if (idx < 24576) {
    const int e = (idx - 16384) * 8;
    *reinterpret_cast<uint2*>(ws_q + e) = q8(*reinterpret_cast<const uint4*>(ws + e), 1.f / sc[2]);
  } else if (idx < 40960) {
    const int k = idx - 24576, n = k % H_, d0 = (k / H_) * 4;
    const float inv = 1.f / sc[2];
    const f2 a = mul2(mk2(__bfloat162float(ws[d0 * H_ + n]), __bfloat162float(ws[(d0 + 1) * H_ + n])), mk2(inv, inv));
    const f2 b = mul2(mk2(__bfloat162float(ws[(d0 + 2) * H_ + n]), __bfloat162float(ws[(d0 + 3) * H_ + n])), mk2(inv, inv));
    *reinterpret_cast<uint32_t*>(wst_q + n * D_ + d0) = e4m3x4(a, b);
  }
}
