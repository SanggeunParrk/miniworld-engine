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
#if defined(REGS_C)
#define REG_LOW 40
#define REG_GATE 152
#define REG_EPI 160
#elif defined(REGS_B)                                           // register split: producers / gates / epilogue (sum 4 x lo + 8 x g + 4 x e = 2048)
#define REG_LOW 40
#define REG_GATE 144
#define REG_EPI 184
#else
#define REG_LOW 56
#define REG_GATE 152
#define REG_EPI 152
#endif
#ifndef EPI_SCALAR
#define EPIF2                                                  // fp32x2 LayerNorm-backward epilogue in the DX role (EPI_SCALAR: v9)
#endif
#ifndef GATE1
#define GATE2                                                  // fp32x2 gate arithmetic, bit-identical to the scalar form (GATE1)
#endif

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int KB = 16384;                                      // one K-block: [128 rows][64 bf16], 128-B swizzled
// ---- DW role shared memory
constexpr int W_WS = 0, W_WAB = 16384, W_IN = 49152, INB = 65536, IN_XN = 32768;
constexpr int W_H = W_IN + 2 * INB, W_DAB = W_H + KB, W_BAR = W_DAB + 2 * KB;
// ---- DX role shared memory
// Ws ring 2 x 16 KB (released by dh_j), [Wa;Wb] ring 2 x 32 KB (released by d_xn_j), inputs 2 x (dy 32 KB + xn 32 KB); once a tile's
// last [a|b] is done its xn half is reloaded with x for the LayerNorm backward
constexpr int X_WS = 0, X_WAB = 32768, NWAB = 2, X_IN = X_WAB + NWAB * 32768;
constexpr int X_GAM = X_IN + 2 * INB, X_BAR = X_GAM + 512;
constexpr int SMEM_BYTES = (W_BAR > X_BAR ? W_BAR : X_BAR) + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
#ifdef FP8SIM
constexpr int KS = 4;                                          // timing model of fp8 operands: half the MMA instructions per product (numerics wrong)
#else
constexpr int KS = 8;
#endif

constexpr int CL = 2;                                          // cluster size (launch attribute): DW input tiles are multicast across it
constexpr uint16_t CL_MASK = (1u << CL) - 1;
constexpr uint32_t I_DH = idesc_bf16(128, 64, 0, 1), I_AB = idesc_bf16(128, 128), I_DXN = idesc_bf16(128, 128, 0, 1);
constexpr uint32_t I_DWAB = idesc_bf16(128, 128, 1, 1), I_DWS = idesc_bf16(128, 64, 1, 1);

#ifdef TRACE
// clock64 stamps: slot 0 = DW CTA 0, slot 1 = the first DX CTA; roles 0 = MMA warp 1, 1 = MMA warp 2, 2 = gate warp 4, 3 = epilogue
constexpr int TR_N = 2048;
__device__ unsigned long long g_trace[2][4][TR_N];
#define TRB(slot, role, idx) do { if ((idx) < TR_N) g_trace[slot][role][(idx)] = clock64(); } while (0)
#else
#define TRB(slot, role, idx) do { } while (0)
#endif

#ifdef SPAN
__device__ unsigned long long g_spanb[256][2];                // globaltimer at CTA entry / exit
#endif
struct Par {
  const CUtensorMap *dy, *xn, *x, *ws, *wa, *wb, *dx, *xch;
  const float *rstd, *c1, *gamma;
  const __nv_bfloat16* xg;
  float *partab, *parts, *dgbw;
  int tiles, ndw;
};

struct BarsW { uint64_t w_full, xn_full[2], dy_full[2], xn_empty[2], dy_empty[2], dhab_full, gate_read, g_full[2], g_empty[2], wg_done; uint32_t tmem; };
#ifdef DW_DB2
constexpr int GDB = 2;                                         // h / dA / dB double-buffered (timing model: both slots alias one buffer)
#else
constexpr int GDB = 1;
#endif
struct BarsX {
  uint64_t ws_full[2], ws_empty[2], wab_full[NWAB], wab_empty[NWAB], in_full[2], in_empty[2], x_full[2], xn_dead[2], abdh_full[2], g_full[2], ab_free[2];
  uint64_t dxn_full, dxn_empty; uint32_t tmem;
};

// the 16-row block of the packed [Wa_j; Wb_j] tile matching A-operand block ks of the half-by-half [dA | dB] layout
__device__ constexpr int kRowBlk[8] = {0, 1, 4, 5, 2, 3, 6, 7};
DEVI float gate_da(float g, float b, float s, float l) { return (g * b) * (s + l * (1.f - s)); }
// the same gate arithmetic on pairs of units in fp32x2 (FMUL2 / FFMA2 round like FMUL / FFMA; s + l (1 - s) contracted as ptxas
// does in the scalar form): h = l b, dA = (g b) fma(l, 1 - s, s), dB = g l, with l = a s and the kit sigmoid per unit
DEVI void gate_pair16(uint32_t dh0, uint32_t dh1, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, uint32_t& hp, uint32_t& dap, uint32_t& dbp) {
  const uint32_t gp = pack_bf16(__uint_as_float(dh0), __uint_as_float(dh1));
  const f2 G = mk2(bf16lo(gp), bf16hi(gp)), A = mk2u(a0, a1), Bv = mk2u(b0, b1);
  const f2 S = mk2(sigmoid_kit(__uint_as_float(a0)), sigmoid_kit(__uint_as_float(a1)));
  const f2 L = mul2(A, S);
  const f2 H = mul2(L, Bv), DB = mul2(G, L);
  const f2 U = fma2(L, fma2(S, mk2(-1.f, -1.f), mk2(1.f, 1.f)), S);
  const f2 DA = mul2(mul2(G, Bv), U);
  hp = pack_bf16(lo2(H), hi2(H)); dap = pack_bf16(lo2(DA), hi2(DA)); dbp = pack_bf16(lo2(DB), hi2(DB));
}

// ================================================================================================ DW role
DEVI void weight_role(const Par& p, uint8_t* sm, int cta, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  BarsW& B = *reinterpret_cast<BarsW*>(sm + W_BAR);
  const int slice = cta & 7, repl = cta >> 3, R = p.ndw >> 3;
  const int crank = (int)cluster_rank();                   // == slice % CL: the CL consecutive CTAs of a cluster are slices of one replica
  const int n_local = (p.tiles > repl) ? (p.tiles - repl + R - 1) / R : 0;
  constexpr uint32_t T_DH = 0, T_AB = 64, T_DWAB = 256, T_DWS = 384;
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int s = 0; s < 2; ++s) {                        // xn and dy of a tile travel separately; each is freed by all CL slices
      mbar_init(&B.xn_full[s], 1); mbar_init(&B.dy_full[s], 1); mbar_init(&B.xn_empty[s], CL); mbar_init(&B.dy_empty[s], CL);
    }
    mbar_init(&B.dhab_full, 1); mbar_init(&B.gate_read, 8); mbar_init(&B.wg_done, 1);
    for (int s = 0; s < 2; ++s) { mbar_init(&B.g_full[s], 8); mbar_init(&B.g_empty[s], 1); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                          // every CTA's barriers exist before any multicast or remote arrive
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
      // xn first (it is freed first, by the dWab product), then dy (freed by dWs); the CL slices share each tile, each requesting
      // 4 / CL of the four 8 KB boxes of each half and multicasting them
      for (int i = 0; i < n_local; ++i) {
        const int b = i & 1, row = (repl + i * R) * ROWS;
        const uint32_t dst = su + W_IN + b * INB;
#pragma unroll
        for (int src = 1; src >= 0; --src) {
          uint64_t* full = src ? &B.xn_full[b] : &B.dy_full[b];
          if (i >= 2) mbar_wait(src ? &B.xn_empty[b] : &B.dy_empty[b], ((i >> 1) - 1) & 1);
          if (cta == 0 && src == 1) TRB(0, 3, i);
#ifdef ABL_BIN
          if (i >= 2) { mbar_arrive(full); continue; }       // ablation: keep the first two tiles' inputs, reload nothing
#endif
          mbar_expect_tx(full, INB / 2);
#pragma unroll
          for (int k = 0; k < 4 / CL; ++k) {
            const int bx = crank * (4 / CL) + k, cb = bx >> 1, h = bx & 1;
            tma_load_2d_mc(dst + src * IN_XN + cb * KB + h * 8192, src ? p.xn : p.dy, full, cb * 64, row + h * 64, CL_MASK);
          }
        }
      }
    }
  } else if (warp == 1) {
    // dh and [a|b] of each tile (converged warp, one elected lane issues: descriptors stay in uniform registers)
    mbar_wait(&B.w_full, 0);
    const uint64_t dws = desc_mn128(su + W_WS, KB), dwab = desc_k128(su + W_WAB);
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      if (cta == 0 && lane == 0) TRB(0, 0, 4 * i);
      mbar_wait(&B.xn_full[b], (i >> 1) & 1);
      if (cta == 0 && lane == 0) TRB(0, 0, 4 * i + 1);
      if (i >= 1) mbar_wait(&B.gate_read, (i - 1) & 1);
      if (cta == 0 && lane == 0) TRB(0, 0, 4 * i + 2);
      tc_fence_after();
      const uint64_t ddy = desc_k128(su + W_IN + b * INB), dxn = desc_k128(su + W_IN + b * INB + IN_XN);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          const uint64_t off = (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4);
          umma_ss(tmem + T_AB, dxn + off, dwab + off, I_AB, ks > 0 ? 1u : 0u);
        }
      }
      __syncwarp();
      mbar_wait(&B.dy_full[b], (i >> 1) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          const uint64_t off = (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4);
          umma_ss(tmem + T_DH, ddy + off, dws + (uint64_t)(ks * 2048 >> 4), I_DH, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.dhab_full);
      }
      __syncwarp();
      if (cta == 0 && lane == 0) TRB(0, 0, 4 * i + 3);
    }
  } else if (warp == 2) {
    // the weight gradients of each tile
    const uint64_t ddab = desc_mn128(su + W_DAB, KB), dh_ = desc_mn128(su + W_H, KB);
    for (int k = 0; k < n_local; ++k) {
      const int b = k & 1;
      if (cta == 0 && lane == 0) TRB(0, 1, 4 * k);
      mbar_wait(&B.g_full[k % GDB], (k / GDB) & 1);
      if (cta == 0 && lane == 0) TRB(0, 1, 4 * k + 1);
      tc_fence_after();
      const uint64_t ddy = desc_mn128(su + W_IN + b * INB, KB), dxn = desc_mn128(su + W_IN + b * INB + IN_XN, KB);
      if (elect_one()) {
#ifdef EXCH_SIM
        {                                    // timing model of the fp8 exchange: 16 KB of [dA | dB] per (tile, slice) to global
          const int blk = (repl + k * R) * 8 + slice;
          tma_store_2d(p.xch, su + W_DAB, 0, blk * 128);
          tma_store_2d(p.xch, su + W_DAB + 8192, 0, blk * 128 + 64);
          tma_store_commit();
        }
#endif
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)       // [dWa_s; dWb_s] += [dA | dB]^T xn    (K = the tile's 128 rows)
          umma_ss(tmem + T_DWAB, ddab + (uint64_t)(ks * 2048 >> 4), dxn + (uint64_t)(ks * 2048 >> 4), I_DWAB, (k > 0 || ks > 0) ? 1u : 0u);
        tc_commit_mc(&B.xn_empty[b], CL_MASK);             // xn of this tile is dead in this slice (counted in every slice)
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)       // dWs_s += dy^T h
          umma_ss(tmem + T_DWS, ddy + (uint64_t)(ks * 2048 >> 4), dh_ + (uint64_t)(ks * 2048 >> 4), I_DWS, (k > 0 || ks > 0) ? 1u : 0u);
#ifdef EXCH_SIM
        tma_store_wait_read0();
#endif
        tc_commit(&B.g_empty[k % GDB]);
        tc_commit_mc(&B.dy_empty[b], CL_MASK);             // so is dy
        if (k == n_local - 1) tc_commit(&B.wg_done);
      }
      __syncwarp();
      if (cta == 0 && lane == 0) TRB(0, 1, 4 * k + 2);
    }
    if (n_local == 0 && elect_one()) mbar_arrive(&B.wg_done);
    __syncwarp();
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    // gate, split by columns over two warpgroups: warps 4-7 hidden units 0..31 of the slice, warps 12-15 units 32..63
    setmaxnreg_inc<REG_GATE>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int half = warp >= 12 ? 1 : 0;
    for (int i = 0; i < n_local; ++i) {
      if (cta == 0 && warp == 4 && lane == 0) TRB(0, 2, 4 * i);
      mbar_wait(&B.dhab_full, i & 1);
      if (cta == 0 && warp == 4 && lane == 0) TRB(0, 2, 4 * i + 1);
      tc_fence_after();
#ifndef GATE_LATEWAIT
      if (i >= GDB) mbar_wait(&B.g_empty[i % GDB], ((i / GDB) - 1) & 1);
#endif
      if (cta == 0 && warp == 4 && lane == 0) TRB(0, 2, 4 * i + 2);
#if !defined(ABL_BGATE) && !defined(ABL_DWGATE)
      {
#ifdef HALF_ACC
        uint32_t dh2[16], av2[16], bv2[16];                // timing model of f16 accumulators: 16 columns of packed pairs
        tmem_ld16(trow + T_DH + half * 16, dh2);
        tmem_ld16(trow + T_AB + half * 16, av2);
        tmem_ld16(trow + T_AB + 64 + half * 16, bv2);
        tmem_wait_ld();
        uint32_t dh[32], av[32], bv[32];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float2 x0 = h2f2(dh2[k]), x1 = h2f2(av2[k]), x2 = h2f2(bv2[k]);
          dh[2 * k] = __float_as_uint(x0.x); dh[2 * k + 1] = __float_as_uint(x0.y);
          av[2 * k] = __float_as_uint(x1.x); av[2 * k + 1] = __float_as_uint(x1.y);
          bv[2 * k] = __float_as_uint(x2.x); bv[2 * k + 1] = __float_as_uint(x2.y);
        }
#else
        uint32_t dh[32], av[32], bv[32];
        tmem_ld32(trow + T_DH + half * 32, dh);
        tmem_ld32(trow + T_AB + half * 32, av);
        tmem_ld32(trow + T_AB + 64 + half * 32, bv);
        tmem_wait_ld();
#endif
        tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.gate_read);
        if (cta == 0 && warp == 4 && lane == 0) TRB(0, 3, 1024 + i);
#ifdef GATE_LATEWAIT
        {                                                  // compute all 32 units first (overlapping the previous tile's wgrad), then wait, then store
          uint32_t hq[16], aq[16], bq[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) gate_pair16(dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], hq[k], aq[k], bq[k]);
          if (i >= GDB) mbar_wait(&B.g_empty[i % GDB], ((i / GDB) - 1) & 1);
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            const uint32_t off = sw128(r, half * 4 + qq);
            sts128(su + W_H + off, make_uint4(hq[4 * qq], hq[4 * qq + 1], hq[4 * qq + 2], hq[4 * qq + 3]));
            sts128(su + W_DAB + off, make_uint4(aq[4 * qq], aq[4 * qq + 1], aq[4 * qq + 2], aq[4 * qq + 3]));
            sts128(su + W_DAB + KB + off, make_uint4(bq[4 * qq], bq[4 * qq + 1], bq[4 * qq + 2], bq[4 * qq + 3]));
          }
        }
        if (false)
#endif
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {                   // 8 hidden units -> one 16-byte chunk of h, dA and dB each, stored right away
          uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int k = qq * 4 + kk;
#ifdef GATE_H2
            gate_h2(f2h2(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1])), f2h2(__uint_as_float(av[2 * k]), __uint_as_float(av[2 * k + 1])),
                    f2h2(__uint_as_float(bv[2 * k]), __uint_as_float(bv[2 * k + 1])), hp[kk], dap[kk], dbp[kk]);
            continue;
#endif
#ifdef GATE2
            gate_pair16(dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], hp[kk], dap[kk], dbp[kk]);
            continue;
#endif
            const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
            const float g0 = bf16lo(gp), g1 = bf16hi(gp);
            const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
            const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
            float s0, s1; SIGMOID2(a0, a1, s0, s1); const float l0 = a0 * s0, l1 = a1 * s1;
            hp[kk] = pack_bf16(l0 * b0, l1 * b1);
            dap[kk] = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1));
            dbp[kk] = pack_bf16(g0 * l0, g1 * l1);
          }
#ifdef GABL_NOCOMP
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) { hp[kk] = dh[qq * 8 + kk]; dap[kk] = av[qq * 8 + kk]; dbp[kk] = bv[qq * 8 + kk]; }
#endif
          const uint32_t off = sw128(r, half * 4 + qq);
#ifdef GABL_NOSTORE
          if ((hp[0] ^ hp[1] ^ hp[2] ^ hp[3] ^ dap[0] ^ dap[1] ^ dap[2] ^ dap[3] ^ dbp[0] ^ dbp[1] ^ dbp[2] ^ dbp[3]) == 0x7f7f1234u)
#endif
          sts128(su + W_H + off, make_uint4(hp[0], hp[1], hp[2], hp[3]));
          sts128(su + W_DAB + off, make_uint4(dap[0], dap[1], dap[2], dap[3]));
          sts128(su + W_DAB + KB + off, make_uint4(dbp[0], dbp[1], dbp[2], dbp[3]));
        }
      }
#else
      tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.gate_read);
#endif
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full[i % GDB]);
      if (cta == 0 && warp == 4 && lane == 0) TRB(0, 2, 4 * i + 3);
    }
  } else if (warp >= 8) {
    // the CTA's fp32 partials: [dWa_s; dWb_s] rows = TMEM lanes, dWs_s rows (d) = TMEM lanes
    setmaxnreg_inc<REG_EPI>();
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
  cluster_sync();                                          // no CTA leaves while a peer may still multicast into it
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

// warp reduce-scatter: lane l returns the sum over the warp of v[l]
DEVI float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    const bool up = (lane & off) != 0;
    // swap the halves in place on the upper lanes (the kept half then sits low, the sent half high) so no temporaries are live
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
  // the two CTAs of a cluster share (multicast) the weight chunks, so they must walk the same chunk sequence: both run the pair's
  // larger tile count, and a CTA short of one tile repeats its last tile as a dummy whose results are not stored
  auto count = [&](int k) { return (p.tiles > k) ? (p.tiles - k + ndx - 1) / ndx : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  const int nch = n_local * NCH;
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * ndx : (n_valid > 0 ? cta + (n_valid - 1) * ndx : 0); };
  const int crank = (int)cluster_rank();
  constexpr uint32_t T_AB = 0, T_DH = 256, T_DXN = 384;          // AB x2 (128 each; [dA|dB] is written back over its first 64), DH x2 (64 each)
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.ws_full[s], 1); mbar_init(&B.ws_empty[s], CL); mbar_init(&B.in_full[s], 1); mbar_init(&B.in_empty[s], 1);
      mbar_init(&B.x_full[s], 1); mbar_init(&B.xn_dead[s], 1);
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
  cluster_sync();                                          // every CTA's barriers exist before any multicast or remote arrive
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      auto issue_in = [&](int i) {
        const int b = i & 1, row = tile_of(i) * ROWS;
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
        const int b = i & 1, row = tile_of(i) * ROWS;
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
      // two rings polled by one thread: Ws_j is released by dh_j (early), [Wa_j; Wb_j] only by d_xn_j (late)
      int cs = 0, ca = 0;
      while (cs < nch || ca < nch) {
        if (cs < nch && (cs < 2 || mbar_test(&B.ws_empty[cs & 1], ((cs >> 1) - 1) & 1))) {
          const int s = cs & 1, j = cs & (NCH - 1);
          const uint32_t slot = su + X_WS + s * 16384;
#ifdef ABL_BW
          if (cs >= 2) { mbar_arrive(&B.ws_full[s]); ++cs; continue; }   // ablation: keep the first two chunks' weights
#endif
#ifdef EXCH_SIM
          mbar_expect_tx(&B.ws_full[s], 16384);             // this CTA's own [dA | dB] block of (tile, chunk j)
          { const int blk = tile_of(cs >> 3) * 8 + j;
            tma_load_2d(slot, p.xch, &B.ws_full[s], 0, blk * 128); tma_load_2d(slot + 8192, p.xch, &B.ws_full[s], 0, blk * 128 + 64); }
#else
          mbar_expect_tx(&B.ws_full[s], 16384);             // this CTA requests one of the two halves, multicast to both
          tma_load_2d_mc(slot + crank * 8192, p.ws, &B.ws_full[s], j * HS, crank * 64, CL_MASK);
#endif
          ++cs;
        }
        if (ca < nch && (ca < NWAB || mbar_test(&B.wab_empty[ca % NWAB], ((ca / NWAB) - 1) & 1))) {
          const int s = ca % NWAB, j = ca & (NCH - 1);
          const uint32_t slot = su + X_WAB + s * 32768;
#ifdef ABL_BW
          if (ca >= NWAB) { mbar_arrive(&B.wab_full[s]); ++ca; continue; }
#endif
          mbar_expect_tx(&B.wab_full[s], 32768);             // K-block `crank` of [Wa_j; Wb_j] from this CTA, multicast to both
          tma_load_2d_mc(slot + crank * KB, p.wa, &B.wab_full[s], crank * 64, j * HS, CL_MASK);
          tma_load_2d_mc(slot + crank * KB + 8192, p.wb, &B.wab_full[s], crank * 64, j * HS, CL_MASK);
          ++ca;
        }
      }
    }
  } else if (warp == 1) {
    // dh_j and [a|b]_j (converged warp, elected issue)
    for (int c = 0; c < nch; ++c) {
      const int i = c >> 3, j = c & (NCH - 1), s = c & 1, u = c >> 1;
      if (cta == 0 && lane == 0) TRB(1, 0, 8 * c);
      const int sw = c % NWAB;
      if (j == 0) mbar_wait(&B.in_full[i & 1], (i >> 1) & 1);
      if (cta == 0 && lane == 0) TRB(1, 0, 8 * c + 1);
      mbar_wait(&B.ws_full[s], u & 1);
      mbar_wait(&B.wab_full[sw], (c / NWAB) & 1);
      if (cta == 0 && lane == 0) TRB(1, 0, 8 * c + 2);
      if (c >= 2) mbar_wait(&B.ab_free[s], (u - 1) & 1);
      if (cta == 0 && lane == 0) TRB(1, 0, 8 * c + 3);
      tc_fence_after();
      const uint64_t ddy = desc_k128(su + X_IN + (i & 1) * INB), dxn = desc_k128(su + X_IN + (i & 1) * INB + IN_XN);
      const uint64_t dws = desc_mn128(su + X_WS + s * 16384, KB), dwab = desc_k128(su + X_WAB + sw * 32768);
      if (elect_one()) {
#ifndef DX_NOREC
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          const uint64_t off = (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4);
          umma_ss(tmem + T_DH + s * 64, ddy + off, dws + (uint64_t)(ks * 2048 >> 4), I_DH, ks > 0 ? 1u : 0u);
        }
#endif
#ifndef EXCH_SIM
        tc_commit_mc(&B.ws_empty[s], CL_MASK);             // Ws_j is dead here once dh_j is done (counted in both CTAs)
#endif
#ifndef DX_NOREC
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          const uint64_t off = (uint64_t)(((ks >> 2) * KB + (ks & 3) * 32) >> 4);
          umma_ss(tmem + T_AB + s * 128, dxn + off, dwab + off, I_AB, ks > 0 ? 1u : 0u);
        }
#endif
        tc_commit(&B.abdh_full[s]);
        if (j == NCH - 1) tc_commit(&B.xn_dead[i & 1]);
      }
      __syncwarp();
      if (cta == 0 && lane == 0) TRB(1, 0, 8 * c + 4);
    }
  } else if (warp == 2) {
    // d_xn += [dA | dB] [Wa_j; Wb_j]  (A from TMEM)
    for (int q = 0; q < nch; ++q) {
      const int i = q >> 3, j = q & (NCH - 1), s = q & 1, u = q >> 1;
      if (cta == 0 && lane == 0) TRB(1, 1, 8 * q);
      mbar_wait(&B.g_full[s], u & 1);
      if (cta == 0 && lane == 0) TRB(1, 1, 8 * q + 1);
      if (j == 0 && i >= 1) mbar_wait(&B.dxn_empty, (i - 1) & 1);
      if (cta == 0 && lane == 0) TRB(1, 1, 8 * q + 2);
      tc_fence_after();
      const uint64_t dwab = desc_mn128(su + X_WAB + (q % NWAB) * 32768, KB);
#ifdef EXCH_SIM
      mbar_wait(&B.ws_full[s], u & 1);
      tc_fence_after();
      const uint64_t dxa = desc_k128(su + X_WS + s * 16384);
#endif
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#ifdef EXCH_SIM
          umma_ss(tmem + T_DXN, dxa + (uint64_t)(ks * 2), dwab + (uint64_t)(kRowBlk[ks] * 2048 >> 4), I_DXN, (j > 0 || ks > 0) ? 1u : 0u);
        tc_commit_mc(&B.ws_empty[s], CL_MASK);
#else
          umma_ts(tmem + T_DXN, tmem + T_AB + s * 128 + ks * 8, dwab + (uint64_t)(kRowBlk[ks] * 2048 >> 4), I_DXN, (j > 0 || ks > 0) ? 1u : 0u);
#endif
        tc_commit(&B.ab_free[s]);
        tc_commit_mc(&B.wab_empty[q % NWAB], CL_MASK);
        if (j == NCH - 1) tc_commit(&B.dxn_full);
      }
      __syncwarp();
      if (cta == 0 && lane == 0) TRB(1, 1, 8 * q + 3);
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    // ------------------------------------------------------------------------------------ gate: [dA | dB] back into TMEM; the two
    // warpgroups split every chunk by columns (warps 4-7 hidden units 0..31, warps 12-15 units 32..63), halving the gate's latency
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
    setmaxnreg_inc<REG_GATE>();
    const int half = warp >= 12 ? 1 : 0;
    for (int c = 0; c < nch; ++c) {
      const int s = c & 1, u = c >> 1;
      if (cta == 0 && warp == 4 && lane == 0) TRB(1, 2, 4 * c);
      mbar_wait(&B.abdh_full[s], u & 1);
      if (cta == 0 && warp == 4 && lane == 0) TRB(1, 2, 4 * c + 1);
      tc_fence_after();
      // [dA | dB] is written back half by half: columns 0..15 dA(hs 0..31), 16..31 dB(hs 0..31), 32..47 dA(hs 32..63), 48..63 dB(hs 32..63),
      // so each half lands only on columns it has already read; the d_xn product pairs each 8-column block with the matching B rows
#if !defined(ABL_BGATE) && !defined(ABL_DXGATE)
      {
#ifdef HALF_ACC
        uint32_t dh2[16], av2[16], bv2[16];
        tmem_ld16(trow + T_DH + s * 64 + half * 16, dh2);
        tmem_ld16(trow + T_AB + s * 128 + half * 16, av2);
        tmem_ld16(trow + T_AB + s * 128 + 64 + half * 16, bv2);
        tmem_wait_ld();
        uint32_t dh[32], av[32], bv[32];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float2 x0 = h2f2(dh2[k]), x1 = h2f2(av2[k]), x2 = h2f2(bv2[k]);
          dh[2 * k] = __float_as_uint(x0.x); dh[2 * k + 1] = __float_as_uint(x0.y);
          av[2 * k] = __float_as_uint(x1.x); av[2 * k + 1] = __float_as_uint(x1.y);
          bv[2 * k] = __float_as_uint(x2.x); bv[2 * k + 1] = __float_as_uint(x2.y);
        }
#else
        uint32_t dh[32], av[32], bv[32];
        tmem_ld32(trow + T_DH + s * 64 + half * 32, dh);
        tmem_ld32(trow + T_AB + s * 128 + half * 32, av);
        tmem_ld32(trow + T_AB + s * 128 + 64 + half * 32, bv);
        tmem_wait_ld();
#endif
        uint32_t da[16], db[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
#ifdef GATE_H2
          uint32_t hh;
          gate_h2(f2h2(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1])), f2h2(__uint_as_float(av[2 * k]), __uint_as_float(av[2 * k + 1])),
                  f2h2(__uint_as_float(bv[2 * k]), __uint_as_float(bv[2 * k + 1])), hh, da[k], db[k]);
          continue;
#endif
#ifdef GATE2
          { uint32_t hh; gate_pair16(dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], hh, da[k], db[k]); (void)hh; }
          continue;
#endif
          const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
          const float g0 = bf16lo(gp), g1 = bf16hi(gp);
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
          float s0, s1; SIGMOID2(a0, a1, s0, s1); const float l0 = a0 * s0, l1 = a1 * s1;
          da[k] = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1));
          db[k] = pack_bf16(g0 * l0, g1 * l1);
        }
        if (cta == 0 && warp == 4 && lane == 0) TRB(1, 2, 4 * c + 2);
        tmem_st16(trow + T_AB + s * 128 + half * 32, da);
        tmem_st16(trow + T_AB + s * 128 + half * 32 + 16, db);
      }
#endif
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.g_full[s]);
      if (cta == 0 && warp == 4 && lane == 0) TRB(1, 2, 4 * c + 3);
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------ LayerNorm backward + residual
    setmaxnreg_inc<REG_EPI>();
    const int t2 = (int)tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gam_u = su + X_GAM;
    float accg[4] = {0.f, 0.f, 0.f, 0.f}, accb[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1, grow = tile_of(i) * ROWS + (int)r;
      const bool real = i < n_valid;
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i);
      mbar_wait(&B.dxn_full, i & 1);
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i + 1);
      tc_fence_after();
#ifndef ABL_BEPI
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
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i + 2);
      const float rs = p.rstd[grow], mean = p.c1[grow] / rs;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i + 3);
      const uint32_t dyb = su + X_IN + b * INB, xb = dyb + IN_XN;
#ifdef EPIF2
      // fp32x2 form of the two passes (same formulas; x - mean times rstd as one fma, row sums in pair order)
      const f2 RS = mk2(rs, rs), NMR = mk2(-mean * rs, -mean * rs);
      f2 pa2[2] = {mk2(0.f, 0.f), mk2(0.f, 0.f)}, pb2[2] = {mk2(0.f, 0.f), mk2(0.f, 0.f)};
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        float work[32];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint4 xv = lds128_nv(xb + (g >> 1) * KB + sw128(r, (g & 1) * 4 + qq));
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int e = qq * 4 + kk, pr = g * 16 + e;           // column pair
            const uint32_t n = dn[pr];
            const f2 N = mk2(bf16lo(n), bf16hi(n));
            const f2 XH = fma2(mk2(bf16lo(xw[kk]), bf16hi(xw[kk])), RS, NMR);
            const float2 gg = lds64f_nv(gam_u + pr * 8);
            const f2 W = mul2(mk2(gg.x, gg.y), N);
            pa2[kk & 1] = fma2(XH, W, pa2[kk & 1]); pb2[kk & 1] = add2(pb2[kk & 1], W);
            const f2 NX = mul2(N, XH);
            work[2 * e] = lo2(NX); work[2 * e + 1] = hi2(NX);
          }
        }
        const float sg = reduce_scatter32(work, lane);
        if (real) accg[g] += sg;
#pragma unroll
        for (int e = 0; e < 16; ++e) { const uint32_t n = dn[g * 16 + e]; work[2 * e] = bf16lo(n); work[2 * e + 1] = bf16hi(n); }
        const float sb = reduce_scatter32(work, lane);
        if (real) accb[g] += sb;
      }
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i + 4);
      const f2 PA = add2(pa2[0], pa2[1]), PB = add2(pb2[0], pb2[1]);
      const float ca = (lo2(PA) + hi2(PA)) * (1.f / D_), cbv = (lo2(PB) + hi2(PB)) * (1.f / D_);
      const f2 NCA = mk2(-ca, -ca), NCB = mk2(-cbv, -cbv);
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
            const int pr = cb * 32 + q * 4 + k;
            const uint32_t n = dn[pr];
            const f2 XH = fma2(mk2(bf16lo(xw[k]), bf16hi(xw[k])), RS, NMR);
            const float2 gg = lds64f(gam_u + pr * 8);
            const f2 W = mul2(mk2(gg.x, gg.y), mk2(bf16lo(n), bf16hi(n)));
            const f2 T = mul2(fma2(XH, NCA, add2(W, NCB)), RS);        // bf16((w - xhat ca - cb) rstd) + dy
            const uint32_t t = pack_bf16(lo2(T), hi2(T));
            o[k] = pack_bf16(bf16lo(t) + bf16lo(dw[k]), bf16hi(t) + bf16hi(dw[k]));
          }
          sts128(dyb + off, make_uint4(o[0], o[1], o[2], o[3]));
        }
#else
      // pass 1, 32 columns at a time: the two row sums (the sm_90a quad order: partial l over columns with (col >> 1) & 3 == l) and the
      // dgamma / dbeta terms, each reduced over the warp's 32 rows with one reusable 32-register work array
      float pca[4] = {0.f, 0.f, 0.f, 0.f}, pcb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        float work[32];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint4 xv = lds128(xb + (g >> 1) * KB + sw128(r, (g & 1) * 4 + qq));   // 8 columns of x at a time
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
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i + 4);
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
            const float2 gg = lds64f(gam_u + col * 4);
            const float w0 = gg.x * bf16lo(n), w1 = gg.y * bf16hi(n);
            const uint32_t t = pack_bf16((w0 - (x0 * ca + cbv)) * rs, (w1 - (x1 * ca + cbv)) * rs);
            o[k] = pack_bf16(bf16lo(t) + bf16lo(dw[k]), bf16hi(t) + bf16hi(dw[k]));
          }
          sts128(dyb + off, make_uint4(o[0], o[1], o[2], o[3]));
        }
#endif
#else
      tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.dxn_empty);
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      const uint32_t dyb = su + X_IN + b * INB; (void)grow; (void)real;
#endif
      if (cta == 0 && t2 == 0) TRB(1, 3, 8 * i + 5);
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
        if (cta == 0) TRB(1, 3, 8 * i + 6);
      }
    }
    float* row = p.dgbw + ((size_t)cta * 4 + (warp & 3)) * 256;
#pragma unroll
    for (int g = 0; g < 4; ++g) { row[g * 32 + lane] = accg[g]; row[128 + g * 32 + lane] = accb[g]; }
    if (t2 == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                          // no CTA leaves while a peer may still multicast into it
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __global__ void __launch_bounds__(512, 1)
transition_bwd_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mx,
                     const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
                     const __grid_constant__ CUtensorMap mdx, const float* __restrict__ rstd, const float* __restrict__ c1,
                     const float* __restrict__ gamma, const __nv_bfloat16* __restrict__ xg, float* __restrict__ partab,
                     float* __restrict__ parts, float* __restrict__ dgbw, int tiles, int ndw
#ifdef EXCH_SIM
                     , const __grid_constant__ CUtensorMap mxch
#endif
                     ) {
#ifdef EXCH_SIM
  const Par p{&mdy, &mxn, &mx, &mws, &mwa, &mwb, &mdx, &mxch, rstd, c1, gamma, xg, partab, parts, dgbw, tiles, ndw};
#else
  const Par p{&mdy, &mxn, &mx, &mws, &mwa, &mwb, &mdx, nullptr, rstd, c1, gamma, xg, partab, parts, dgbw, tiles, ndw};
#endif
  extern __shared__ __align__(1024) uint8_t sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // the producer / MMA warpgroup gives registers to the math warpgroups. The increase is issued INSIDE each math role's branch: an
  // if / else around both would merge the two limits and ptxas would compile everything after it at the lower one (56).
#ifdef SPAN
  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_spanb[blockIdx.x][0] = t; }
#endif
  if (warp < 4) setmaxnreg_dec<REG_LOW>();
  if ((int)blockIdx.x < ndw) weight_role(p, sm, blockIdx.x, warp, lane);
  else input_role(p, sm, blockIdx.x - ndw, gridDim.x - ndw, warp, lane);
#ifdef SPAN
  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_spanb[blockIdx.x][1] = t; }
#endif
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
