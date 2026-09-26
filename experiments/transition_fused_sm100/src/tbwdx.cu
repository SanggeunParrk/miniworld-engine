// tbwdx.cu — the Transition backward in bf16 (the H100 contract: bf16 operands, fp32 accumulate, the kit sigmoid and the rounding
// points of tbwd.cu) WITHOUT the DX role's recompute (algorithm change; the exchange structure of the e4m3 record tbwd8x.cu):
//   DW CTAs (8 hidden slices x R replicas, pairs multicast each tile's xn and dy): recompute dh / a / b of their slice, run the only
//     gate (h, dA, dB -> bf16 in shared memory), accumulate dWab / dWs, and publish the bf16 [dA | dB] block of every (tile, slice)
//     (32 KB, TMA store + epoch flag).
//   DX CTAs: d_xn = [dA | dB] [Wa; Wb] from the published blocks in 16 KB half-chunks (dA_j with Wa_j, dB_j with Wb_j; A K-major from
//     shared memory, B the MN-major view of the weight half), then the LayerNorm backward + residual on two warpgroups (fp32x2).
// Tensor work 22 -> 16 M D H; one gate per hidden unit instead of two. SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef PUB_DIRECT
#define PUB_TMA                                                // publish [dA | dB] by TMA store (PUB_DIRECT: gate threads store, slower)
#endif

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int KB = 16384;                                      // bf16 [128][64], 128-B swizzled
// ---- DW role: Ws_s (MN-major) | [Wa_s; Wb_s] (2 K-blocks) | NIN stages of xn | dy | h | dA dB (single-buffered, as tbwd.cu)
constexpr int NIN = 2;
constexpr int W_WS = 0, W_WAB = KB, W_IN = 3 * KB, INS = 4 * KB, IN_DY = 2 * KB;
constexpr int W_H = W_IN + NIN * INS, W_DAB = W_H + KB, W_BAR = W_DAB + 2 * KB;
// ---- DX role: weight halves ring | [dA | dB] half-block ring | 2 stages of dy | x
#ifndef NWAB_
#define NWAB_ 3
#define NDAB_ 3
#endif
constexpr int NWAB = NWAB_, NDAB = NDAB_;
constexpr int X_WAB = 0, X_DAB = NWAB * KB, X_IN = X_DAB + NDAB * KB, XIS = 4 * KB, XI_X = 2 * KB;
constexpr int X_GAM = X_IN + 2 * XIS, X_RED = X_GAM + 512, X_BAR = X_RED + 2048;
constexpr int SMEM_BYTES = (W_BAR > X_BAR ? W_BAR : X_BAR) + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");

constexpr int CL = 2;
constexpr uint16_t CL_MASK = (1u << CL) - 1;
constexpr uint32_t I_AB = idesc_bf16(128, 128), I_DH = idesc_bf16(128, 64, 0, 1), I_DXN = idesc_bf16(128, 128, 0, 1);
constexpr uint32_t I_DWAB = idesc_bf16(128, 128, 1, 1), I_DWS = idesc_bf16(128, 64, 1, 1);

#ifdef SPAN
__device__ unsigned long long g_spanb[256][2];
#endif

struct Par {
  const CUtensorMap *dy, *xn, *x, *ws, *wa, *wb, *dx, *dab;
  const float *rstd, *c1, *gamma;
  float *partab, *parts, *dgbw;
  __nv_bfloat16* dab_g;
  unsigned* dflags;                // [tiles * 8]: == epoch once the (tile, slice) block is in global memory
  unsigned epoch;
  int tiles, ndw;
};
struct BarsW {
  uint64_t w_full, xn_full[NIN], dy_full[NIN], in_empty[NIN], dhab_full, gate_read, g_full, g_empty, wg_done; uint32_t tmem;
};
struct BarsX {
  uint64_t wab_full[NWAB], wab_empty[NWAB], dab_full[NDAB], dab_empty[NDAB], in_full[2], in_empty[2], dxn_full[2], dxn_empty[2];
  uint32_t tmem;
};
DEVI uint32_t ld_acquire_gpu(const unsigned* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEVI void red_release_gpu_add(unsigned* p, uint32_t v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
DEVI void st_release_gpu(unsigned* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI void discard_l2(const void* p) { asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory"); }
DEVI float gate_da(float g, float b, float s, float l) { return (g * b) * (s + l * (1.f - s)); }

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
    mbar_init(&B.dhab_full, 1); mbar_init(&B.gate_read, 8); mbar_init(&B.g_full, 8);
#ifdef PUB_TMA
    mbar_init(&B.g_empty, 2);
#else
    mbar_init(&B.g_empty, 1);
#endif
    mbar_init(&B.wg_done, 1);
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
      mbar_expect_tx(&B.w_full, 3 * KB);
      tma_load_2d(su + W_WS, p.ws, &B.w_full, slice * HS, 0);
      tma_load_2d(su + W_WS + 8192, p.ws, &B.w_full, slice * HS, 64);
#pragma unroll
      for (int cb = 0; cb < 2; ++cb) {
        tma_load_2d(su + W_WAB + cb * KB, p.wa, &B.w_full, cb * 64, slice * HS);
        tma_load_2d(su + W_WAB + cb * KB + 8192, p.wb, &B.w_full, cb * 64, slice * HS);
      }
      for (int i = 0; i < n_local; ++i) {
        const int b = i % NIN, row = (repl + i * R) * ROWS;
        if (i >= NIN) mbar_wait(&B.in_empty[b], ((i / NIN) - 1) & 1);
        // xn first (the [a|b] product needs it first), then dy; each CTA of the pair requests two of the four 8 KB boxes, multicast
#pragma unroll
        for (int src = 1; src >= 0; --src) {
          uint64_t* full = src ? &B.xn_full[b] : &B.dy_full[b];
          mbar_expect_tx(full, 2 * KB);
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int bx = crank * 2 + k, cb = bx >> 1, h = bx & 1;
            tma_load_2d_mc(su + W_IN + b * INS + (src ? 0 : IN_DY) + cb * KB + h * 8192, src ? p.xn : p.dy, full, cb * 64, row + h * 64, CL_MASK);
          }
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
      tc_fence_after();
      const uint64_t dxn = desc_k128(su + W_IN + b * INS), ddy = desc_k128(su + W_IN + b * INS + IN_DY);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint64_t off = (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4);
          umma_ss(tmem + T_AB, dxn + off, dwab + off, I_AB, ks > 0 ? 1u : 0u);
        }
      }
      __syncwarp();
      mbar_wait(&B.dy_full[b], (i / NIN) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint64_t off = (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4);
          umma_ss(tmem + T_DH, ddy + off, dws + (uint64_t)(ks * 2048 >> 4), I_DH, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.dhab_full);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    const uint64_t ddab = desc_mn128(su + W_DAB, KB), dh_ = desc_mn128(su + W_H, KB);
    for (int k = 0; k < n_local; ++k) {
      const int b = k % NIN;
      mbar_wait(&B.g_full, k & 1);
      tc_fence_after();
      const uint64_t dxn = desc_mn128(su + W_IN + b * INS, KB), ddy = desc_mn128(su + W_IN + b * INS + IN_DY, KB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)       // [dWa_s; dWb_s] += [dA | dB]^T xn    (K = the tile's 128 rows)
          umma_ss(tmem + T_DWAB, ddab + (uint64_t)(ks * 2048 >> 4), dxn + (uint64_t)(ks * 2048 >> 4), I_DWAB, (k > 0 || ks > 0) ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)       // dWs_s += dy^T h
          umma_ss(tmem + T_DWS, ddy + (uint64_t)(ks * 2048 >> 4), dh_ + (uint64_t)(ks * 2048 >> 4), I_DWS, (k > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.g_empty);
        tc_commit_mc(&B.in_empty[b], CL_MASK);
        if (k == n_local - 1) tc_commit(&B.wg_done);
      }
      __syncwarp();
    }
    if (n_local == 0 && elect_one()) mbar_arrive(&B.wg_done);
    __syncwarp();
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    // gate (tbwd.cu v9 arithmetic): warps 4-7 hidden units 0..31 of the slice, warps 12-15 units 32..63
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int half = warp >= 12 ? 1 : 0;
    for (int i = 0; i < n_local; ++i) {
      const int t = repl + i * R;
      __nv_bfloat16* grow_p = p.dab_g + ((size_t)(t * 8 + slice) * ROWS + r) * D_ + half * 32;   // this thread's row of the block
      mbar_wait(&B.dhab_full, i & 1);
      tc_fence_after();
      if (i >= 1) mbar_wait(&B.g_empty, (i - 1) & 1);
      uint32_t dh[32], av[32], bv[32];
      tmem_ld32(trow + T_DH + half * 32, dh);
      tmem_ld32(trow + T_AB + half * 32, av);
      tmem_ld32(trow + T_AB + 64 + half * 32, bv);
      tmem_wait_ld();
      tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.gate_read);
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {                     // 8 hidden units -> one 16-byte chunk of h, dA and dB each
        uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int k = qq * 4 + kk;
          const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
          const float g0 = bf16lo(gp), g1 = bf16hi(gp);
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          hp[kk] = pack_bf16(l0 * b0, l1 * b1);
          dap[kk] = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1));
          dbp[kk] = pack_bf16(g0 * l0, g1 * l1);
        }
        const uint32_t off = sw128(r, half * 4 + qq);
        sts128(su + W_H + off, make_uint4(hp[0], hp[1], hp[2], hp[3]));
        sts128(su + W_DAB + off, make_uint4(dap[0], dap[1], dap[2], dap[3]));
        sts128(su + W_DAB + KB + off, make_uint4(dbp[0], dbp[1], dbp[2], dbp[3]));
#ifndef PUB_TMA
        stg128(grow_p + qq * 8, make_uint4(dap[0], dap[1], dap[2], dap[3]));
        stg128(grow_p + 64 + qq * 8, make_uint4(dbp[0], dbp[1], dbp[2], dbp[3]));
#endif
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full);
#ifndef PUB_TMA
      __threadfence();                                      // publish: 8 gate warps per (tile, slice), flag reaches 8 * epoch
      __syncwarp();
      if (lane == 0) red_release_gpu_add(p.dflags + t * 8 + slice, 1u);
#endif
    }
  } else if (warp >= 8) {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
#ifdef PUB_TMA
    if (warp == 8 && lane == 0) {
#else
    if (false) {
#endif
      // publish each tile's [dA | dB] block: store, release the buffer once read, raise the flag once written (one tile behind)
      int prev = -1;
      for (int i = 0; i < n_local; ++i) {
        const int t = repl + i * R, row = (t * 8 + slice) * ROWS;
        mbar_wait(&B.g_full, i & 1);
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(p.dab, su + W_DAB + cb * KB + h * 8192, cb * 64, row + h * 64);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.g_empty);
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
DEVI void input_role(const Par& p, uint8_t* sm, int cta, int ndx, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsX& B = *reinterpret_cast<BarsX*>(sm + X_BAR);
  const int n_local = (p.tiles > cta) ? (p.tiles - cta + ndx - 1) / ndx : 0;
  const int nhc = n_local * NCH * 2;                           // half-chunks: (tile, chunk j, part 0 = dA / Wa, 1 = dB / Wb)
  auto tile_of = [&](int i) { return cta + i * ndx; };
  constexpr uint32_t T_DXN = 0;                                // d_xn x2 (128 columns each)
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.in_full[s], 1); mbar_init(&B.in_empty[s], 1); mbar_init(&B.dxn_full[s], 1); mbar_init(&B.dxn_empty[s], 8);
    }
    for (int s = 0; s < NWAB; ++s) { mbar_init(&B.wab_full[s], 1); mbar_init(&B.wab_empty[s], 1); }
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
        mbar_expect_tx(&B.in_full[b], 4 * KB);
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
    if (lane == 0) {                                         // weight halves: Wa_j or Wb_j, [64 n][128 d] as two 64-column boxes
      for (int hc = 0; hc < nhc; ++hc) {
        const int s = hc % NWAB, j = (hc >> 1) & (NCH - 1), part = hc & 1;
        if (hc >= NWAB) mbar_wait(&B.wab_empty[s], ((hc / NWAB) - 1) & 1);
        mbar_expect_tx(&B.wab_full[s], KB);
        tma_load_2d(su + X_WAB + s * KB, part ? p.wb : p.wa, &B.wab_full[s], 0, j * HS);
        tma_load_2d(su + X_WAB + s * KB + 8192, part ? p.wb : p.wa, &B.wab_full[s], 64, j * HS);
      }
    }
  } else if (warp == 4) {
    if (lane == 0) {                                         // the published blocks: dA_j (part 0) and dB_j (part 1) halves
      for (int hc = 0; hc < nhc; ++hc) {
        const int s = hc % NDAB, i = hc >> 4, j = (hc >> 1) & (NCH - 1), part = hc & 1, t = tile_of(i);
        if (hc >= NDAB) mbar_wait(&B.dab_empty[s], ((hc / NDAB) - 1) & 1);
        if (part == 0) {
#ifdef PUB_TMA
          while (ld_acquire_gpu(p.dflags + t * 8 + j) != p.epoch) { __nanosleep(32); }
#else
          while (ld_acquire_gpu(p.dflags + t * 8 + j) != 8u * p.epoch) { __nanosleep(32); }
#endif
          fence_proxy_async_global();
        }
        mbar_expect_tx(&B.dab_full[s], KB);
        const int row = (t * 8 + j) * ROWS;
        tma_load_2d(su + X_DAB + s * KB, p.dab, &B.dab_full[s], part * 64, row);
        tma_load_2d(su + X_DAB + s * KB + 8192, p.dab, &B.dab_full[s], part * 64, row + 64);
      }
    }
  } else if (warp == 5) {
    // once a half-block has landed its lines are dead in L2: drop them without write-back
    for (int hc = 0; hc < nhc; ++hc) {
      const int s = hc % NDAB, i = hc >> 4, j = (hc >> 1) & (NCH - 1), part = hc & 1;
      mbar_wait(&B.dab_full[s], (hc / NDAB) & 1);
#ifndef NO_DISCARD
      const __nv_bfloat16* g = p.dab_g + (size_t)((tile_of(i) * 8 + j) * ROWS) * D_ + part * 64;
#pragma unroll
      for (int l = 0; l < 4; ++l) discard_l2(g + (size_t)(lane * 4 + l) * D_);
#endif
    }
  } else if (warp == 1) {
    // d_xn += dA_j Wa_j + dB_j Wb_j: A K-major [128 rows][64 units], B = the weight half's MN-major view (K = 64 units, N = 128 d)
    for (int hc = 0; hc < nhc; ++hc) {
      const int i = hc >> 4, sd = hc % NDAB, sw = hc % NWAB, e = i & 1, first = (hc & 15) == 0, last = (hc & 15) == 15;
      mbar_wait(&B.dab_full[sd], (hc / NDAB) & 1);
      mbar_wait(&B.wab_full[sw], (hc / NWAB) & 1);
      if (first && i >= 2) mbar_wait(&B.dxn_empty[e], ((i >> 1) - 1) & 1);
      tc_fence_after();
      const uint64_t da = desc_k128(su + X_DAB + sd * KB), dw = desc_mn128(su + X_WAB + sw * KB, 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ss(tmem + T_DXN + e * 128, da + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 2048 >> 4), I_DXN, (!first || ks > 0) ? 1u : 0u);
        tc_commit(&B.dab_empty[sd]);
        tc_commit(&B.wab_empty[sw]);
        if (last) tc_commit(&B.dxn_full[e]);
      }
      __syncwarp();
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------ LayerNorm backward + residual: warps 8-11
    // columns 0..63 (K-block 0 of the bf16 tiles), warps 12-15 columns 64..127; one row per thread, row sums exchanged through X_RED
    setmaxnreg_inc<152>();
    const int G = warp >= 12 ? 1 : 0, t2 = (int)tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gam_u = su + X_GAM + G * 256, red = su + X_RED;
    float accg[2] = {0.f, 0.f}, accb[2] = {0.f, 0.f};
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1, e = i & 1, grow = tile_of(i) * ROWS + (int)r;
      mbar_wait(&B.dxn_full[e], (i >> 1) & 1);
      tc_fence_after();
      f2 dn[32];                                             // this group's 64 columns of d_xn, rounded once to bf16 (the contract)
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_DXN + e * 128 + G * 64 + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const uint32_t pk = pack_bf16(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
          dn[cc * 16 + k] = mk2(bf16lo(pk), bf16hi(pk));
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxn_empty[e]);
      const float rs = p.rstd[grow], mean = p.c1[grow] / rs;
      mbar_wait(&B.in_full[b], (i >> 1) & 1);
      const uint32_t dyb = su + X_IN + b * XIS + G * KB, xb = su + X_IN + b * XIS + XI_X + G * KB;
      const f2 RS = mk2(rs, rs), NM = mk2(-mean * rs, -mean * rs);
      f2 pa2 = mk2(0.f, 0.f), pb2 = mk2(0.f, 0.f);
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        float work[32];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint4 xv = lds128_nv(xb + sw128(r, g * 4 + qq));
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int e2 = qq * 4 + kk, pr = g * 16 + e2;
            const f2 xh = fma2(mk2(bf16lo(xw[kk]), bf16hi(xw[kk])), RS, NM);
            const float2 gg = lds64f_nv(gam_u + pr * 8);
            const f2 w = mul2(mk2(gg.x, gg.y), dn[pr]);
            pa2 = fma2(xh, w, pa2); pb2 = add2(pb2, w);
            const f2 nx = mul2(dn[pr], xh);
            work[2 * e2] = lo2(nx); work[2 * e2 + 1] = hi2(nx);
          }
        }
        accg[g] += reduce_scatter32(work, lane);
#pragma unroll
        for (int e2 = 0; e2 < 16; ++e2) { work[2 * e2] = lo2(dn[g * 16 + e2]); work[2 * e2 + 1] = hi2(dn[g * 16 + e2]); }
        accb[g] += reduce_scatter32(work, lane);
      }
      const float sa = lo2(pa2) + hi2(pa2), sbv = lo2(pb2) + hi2(pb2);
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(red + (G * 128 + r) * 8), "f"(sa), "f"(sbv) : "memory");
      named_bar_sync(1, 256);
      const float2 o2 = lds64f(red + ((G ^ 1) * 128 + r) * 8);
      const float ca = (G == 0 ? sa + o2.x : o2.x + sa) * (1.f / D_), cbv = (G == 0 ? sbv + o2.y : o2.y + sbv) * (1.f / D_);
      const f2 NCA = mk2(-ca, -ca), NCB = mk2(-cbv, -cbv);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t off = sw128(r, q);
        const uint4 xv = lds128(xb + off), dv = lds128(dyb + off);
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, dw[4] = {dv.x, dv.y, dv.z, dv.w};
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int pr = q * 4 + k;
          const f2 xh = fma2(mk2(bf16lo(xw[k]), bf16hi(xw[k])), RS, NM);
          const float2 gg = lds64f(gam_u + pr * 8);
          const f2 w = mul2(mk2(gg.x, gg.y), dn[pr]);
          const f2 tt = mul2(fma2(xh, NCA, add2(w, NCB)), RS);        // bf16((w - xhat ca - cb) rstd) + dy, as tbwd.cu
          const uint32_t tp = pack_bf16(lo2(tt), hi2(tt));
          o[k] = pack_bf16(bf16lo(tp) + bf16lo(dw[k]), bf16hi(tp) + bf16hi(dw[k]));
        }
        sts128(dyb + off, make_uint4(o[0], o[1], o[2], o[3]));
      }
      fence_proxy_async();
      named_bar_sync(1, 256);
      if (t2 == 0) {
        const int row0 = tile_of(i) * ROWS;
        const uint32_t base = su + X_IN + b * XIS;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(p.dx, base + cb * KB + h * 8192, cb * 64, row0 + h * 64);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.in_empty[b]);
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
transition_bwdx_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mx,
                      const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
                      const __grid_constant__ CUtensorMap mdx, const __grid_constant__ CUtensorMap mdab, const float* __restrict__ rstd,
                      const float* __restrict__ c1, const float* __restrict__ gamma, float* __restrict__ partab, float* __restrict__ parts,
                      float* __restrict__ dgbw, __nv_bfloat16* __restrict__ dab_g, unsigned* __restrict__ dflags,
                      const unsigned* __restrict__ epoch, int tiles, int ndw) {
  const Par p{&mdy, &mxn, &mx, &mws, &mwa, &mwb, &mdx, &mdab, rstd, c1, gamma, partab, parts, dgbw, dab_g, dflags, *epoch, tiles, ndw};
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

// partab [NDW][128][128], parts [NDW][128 d][64 hs] -> bf16 dWa, dWb [512][128], dWs [128][512]; dgbw [NDX * 4][256] -> dgamma, dbeta.
// Blocks 0 .. 767: one weight-gradient element per thread; blocks 768 .. 799: eight dgamma / dbeta columns per block, one per warp.
extern "C" __global__ void transition_bwdx_reduce(const float* __restrict__ partab, const float* __restrict__ parts, const float* __restrict__ dgbw,
                                                  __nv_bfloat16* __restrict__ dwa, __nv_bfloat16* __restrict__ dwb, __nv_bfloat16* __restrict__ dws,
                                                  float* __restrict__ dgam, float* __restrict__ dbeta, int ndw, int nrows_dg, unsigned* __restrict__ epoch) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, R = ndw >> 3;
  if (idx == 0) *epoch += 1u;
  if (blockIdx.x < 768) {
    float v0 = 0.f, v1 = 0.f;
    if (idx < 2 * H_ * D_) {
      const int which = idx / (H_ * D_), rem = idx % (H_ * D_), slice = rem / (HS * D_), hs = (rem / D_) % HS, d = rem % D_;
      const float* src = partab + ((size_t)slice * 128 + which * 64 + hs) * 128 + d;
      int rr = 0;
      for (; rr + 1 < R; rr += 2) { v0 += src[(size_t)(rr * 8) * 16384]; v1 += src[(size_t)((rr + 1) * 8) * 16384]; }
      if (rr < R) v0 += src[(size_t)(rr * 8) * 16384];
      (which == 0 ? dwa : dwb)[(slice * HS + hs) * D_ + d] = __float2bfloat16_rn(v0 + v1);
    } else {
      const int rem = idx - 2 * H_ * D_, d = rem / H_, hh = rem % H_, slice = hh / HS, hs = hh % HS;
      const float* src = parts + ((size_t)slice * 128 + d) * 64 + hs;
      int rr = 0;
      for (; rr + 1 < R; rr += 2) { v0 += src[(size_t)(rr * 8) * 8192]; v1 += src[(size_t)((rr + 1) * 8) * 8192]; }
      if (rr < R) v0 += src[(size_t)(rr * 8) * 8192];
      dws[d * H_ + hh] = __float2bfloat16_rn(v0 + v1);
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
