// k3.cu -- TriMul K3 on B200 (tcgen05): output LayerNorm + output projection + output gate (+ dropout) + residual.
//
// Fusion = the H100 K3 (inference: Anthropic tmn k3_body unmodified; training: MiniWorld save_k3):
//   xo  = bf16(LN_out(tri))  over the 256 packed channels (tri is channel-major [256, L*L])
//   xn  = bf16(LN_in(x))     recomputed from x (the gate input)
//   p   = xo Wp^T  [128],  g = xn Wg^T [128]   (fp32 accumulators)
//   inference:  y = bf16(x + bf16(sigmoid(g) p))
//   training:   y = bf16(fma(bf16(p) sigmoid(bf16(g)), ds[j], x));  saves xn [M,128] bf16 and LN_out mean / rstd [M] fp32
//   SAVE = 2:   the training epilogue (dropout scale ds) without the saves -- a no-grad forward with dropout (MiniWorld's
//               no-grad recycles in train mode): y bitwise equal to the training forward's, no xn / statistics stores, no x reload
//
// Mapping (128-token tiles, persistent CTA, 512 threads):
//   warp 0 TMA (x tiles, tri tile, Wp K-block ring, Wg once)   warp 1 MMA   warp 2 TMEM alloc
//   warps 4-7   LN_in: row per thread, xn -> TMEM (32x32b) [+ in place into the x tile -> TMA store xn]
//   warps 8-11  LN_out: ldmatrix.trans of the [ch][tok] tri tile gives each thread channel pairs of tokens t/4, t/4+8 (the mma
//               fragment layout); statistics by quad shuffles; xo -> TMEM with tcgen05.st.16x256b. The resulting TMEM K order is
//               a fixed permutation of the channels; Wp's columns are permuted identically on the host (k1w_prep writes it as wpp).
//   warps 12-15 epilogue: thread = token row; y written in place into the x tile, TMA store.
// TMEM: xo [0,128)  xn [128,192)  acc_p [256,384)  acc_g [384,512)
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;
#ifdef K3G_TLG   // S90: per-tile %globaltimer stamps (ns) of tiles 10 and 11 per CTA, stored to global (never printf)
static __device__ unsigned long long g_tlg[480][16];
#define TLG(i_, k_) { if ((i_) == 10 || (i_) == 11) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_tlg[blockIdx.x][((i_) - 10) * 8 + (k_)] = t_; } }
#else
#define TLG(i_, k_) {}
#endif
#ifdef K3G_TT
// per-tile %globaltimer stamps of CTA 0 (S90): [tile][event], read by k3g_tt()
static __device__ unsigned long long g_tt[32][8];
#define TT(i, e) { if (blockIdx.x == 0 && (i) < 32) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_tt[i][e] = t_; } }
#else
#define TT(i, e) {}
#endif
#ifdef TM_TL
static __device__ unsigned long long g_tl[160][16];
#define TL(slot) { g_tl[blockIdx.x][slot] = clock64(); }
#else
#define TL(slot) {}
#endif

#ifndef K3_LNIN1
#define K3_LNIN1 1   // LN_in statistics in one pivot-shifted pass (0: the two-pass form)
#endif
#ifndef K3_ABL
#define K3_ABL 0   // timing ablations (wrong results): 1 no LN_out affine, 2 no LN_out stats, 4 no LN_in stats, 8 no epilogue math, 16 no LN_in affine
#endif
#ifndef K3_SIG
#define K3_SIG 2   // epilogue sigmoid: 0 ex2 + rcp (two MUFU ops), 1 ex2 + Newton reciprocal, 2 tanh.approx (one MUFU op)
#endif
namespace k3g {

constexpr int TOK = 128, KBB = TOK * 128;       // one [128 rows][64 k] bf16 K-block (SW128) = 16 KB
#ifndef K3_NWP
#define K3_NWP 3
#endif
constexpr int NWP = K3_NWP;
// tri: a ring of 3 token halves ([256 ch][64 tok] = 32 KB each); tile i's half h goes to slot (2i + h) % 3, so the next tile's first
// half loads while this tile's second half is still being normalised
#ifndef K3_NTH
#define K3_NTH 2
#endif
#ifndef K3_LN1
#define K3_LN1 1   // LN_out statistics in one pivot-shifted pass (0: two passes, mean then centred squares)
#endif
#define K3_STR(x) #x
#define K3_UNROLL(n) _Pragma(K3_STR(unroll n))
#ifndef K3_LNU
#define K3_LNU 4   // LN_in row passes: a full unroll spilled ~100 B at C = 128
#endif
#ifndef K3_LNU4
#define K3_LNU4 8
#endif
#ifndef K3_EPW
#define K3_EPW 16
#endif
#ifndef K3_LNOM
#define K3_LNOM 1   // LN_out warp assignment at H == C (measured: -5 % there, +5 % at C64 H128): warps 8-11 take both token groups; 0 / other shapes: warps 4-7 (after LN_in) take group 1, warps 8-11 group 0
#endif
#ifndef K3_NXB
#define K3_NXB 3   // x tile ring depth (S90: the x tile's life, load -> LN_in -> products -> epilogue -> store, is ~2 tile periods at C = 64 / 128)
#endif
constexpr int NXB = K3_NXB;
#ifndef K3_NAB
#define K3_NAB 1
#endif
#ifndef K3_XNS
#define K3_XNS 1
#endif
#ifndef K3_WRES
#define K3_WRES 1   // the whole Wp fits the ring (H / 64 <= NWP): loaded once per CTA and kept, not re-streamed per tile
#endif
#ifndef K3_EARLY
#define K3_EARLY 0   // 1: every first load issued by thread 0 at kernel entry: with the PDL launch no faster at L128, slower at L256 (D64 eval 32.7 -> 34.7 us)
#endif
#ifndef K3_PDL
#define K3_PDL 1   // launched as a programmatic dependent of the contraction: set-up, x / weight loads and LN_in overlap its tail; t after griddepcontrol.wait
#endif
#ifndef K3_TRIG
#define K3_TRIG 1   // launch_dependents at entry: the PDL b1s schedules its CTAs as this grid's retire
#endif
#ifndef K3_PDL_TILES
#define K3_PDL_TILES 1   // PDL only while every CTA has at most this many tiles (multiples of the grid)
#endif
#ifndef K3_LAT
#define K3_LAT 1   // one tile per CTA: warps 4-7 (free after the LayerNorms) run the epilogue of the upper half of the channels
#endif
#ifndef K3_GSMEM
#define K3_GSMEM 1
#endif
#ifndef K3_PFX
#define K3_PFX 1   // L2 prefetch of the next tile's x boxes
#endif
#ifndef K3_INF_NXB
#define K3_INF_NXB 2   // x tile ring depth of inference at C = H = 128 (with K3_NTH_BIG tri half slots; 3 x tiles + 4 tri slots do not fit)
#endif
#ifndef K3_NTH_BIG
#define K3_NTH_BIG 2   // tri-tile half slots at C = 128, H = 128 (4: the next tile's t is already landing during this tile's LayerNorm)
#endif
constexpr int NTH = K3_NTH;
constexpr int NTHMAX = K3_NTH_BIG > K3_NTH ? K3_NTH_BIG : K3_NTH;   // barrier array size
constexpr int BARPAD = NTHMAX > 2 ? 320 : 256;                       // bytes after the barrier struct
// C = d_pair (64 / 128), H = packed hidden (C: one direction, 2C: bidirectional or one direction at hidden 2C, 4C: bidirectional
// at hidden 2C, D64 only), H <= 256
template <int C, int H, int SV = 0>   // SV: SAVE (0 inference, 1 / 2 training)
struct Geo {
  static constexpr int XT = TOK * C * 2;        // x tile [128 tok][C ch], C / 64 K-blocks of 16 KB (SW128)
  static constexpr int TT = H * TOK * 2;        // tri tile [H ch][128 tok] as 2 token halves of [H][64] SW128
  static constexpr int WPB = C * 64 * 2;        // Wp K-block [C rows][64 k]
  static constexpr int WG = C * C * 2;          // Wg [C][C] (C / 64 K-blocks of [C rows][64 k])
  // x tile ring depth / tri slots of this shape. Inference at C = H = 128 (SV 0): K3_INF_NXB x tiles and K3_NTH_BIG tri half slots (S90 part 6: the next tile's t lands
  // during this tile's LayerNorm); every other shape / training: K3_NXB x tiles (S90 training: the x tile's life is ~2 tile periods) where the shared memory allows, else 2
  static constexpr int NTHG = (SV == 0 && C == 128 && H == 128) ? K3_NTH_BIG : K3_NTH;
  static constexpr int NXB0 = (NTHG > K3_NTH) ? K3_INF_NXB : K3_NXB;
  static constexpr int NXG0 = WG + NWP * WPB + NXB0 * XT + NTHG * (TT / 2) + 8 * H + (K3_GSMEM ? 8 * C : 0) + BARPAD + 1024 + (C == 64 ? NXB0 * XT : 0);
  static constexpr int NXG = NXG0 <= 232448 ? NXB0 : 2;
  static constexpr int O_WG = 0, O_WP = O_WG + WG, O_X = O_WP + NWP * WPB, O_T = O_X + NXG * XT, O_GB = O_T + NTHG * (TT / 2),
                       O_GI = O_GB + 8 * H, O_BAR = O_GI + (K3_GSMEM ? 8 * C : 0);
  static constexpr int O_XS = (O_BAR + BARPAD + 1023) & ~1023;   // C = 64 training: two xn staging tiles (the x tile stays intact)
  static constexpr int SMEM = C == 64 ? O_XS + NXG * XT : O_BAR + BARPAD;
  static_assert(SMEM <= 232448, "smem");
  static_assert(O_WP % 1024 == 0 && O_X % 1024 == 0 && O_T % 1024 == 0 && (TT / 2) % 1024 == 0, "operand alignment");
  static_assert(H <= 256 && C <= 128 && H % 64 == 0 && C % 64 == 0, "shape");
};
constexpr uint32_t T_XO = 0, T_XN = 128, T_DS = 192, T_P = 256, T_G = 384;   // T_DS: training, the CTA's ds tile (bf16 pairs)

struct Bars {
  uint64_t x_full[NXB], x_empty[NXB], th_full[NTHMAX], th_empty[NTHMAX], wg_full, wp_full[NWP], wp_empty[NWP];
  uint64_t ds_full, xn_ready, xo_ready, a_free, acc_full[2], acc_empty[2], xn_stored[NXB], xr_full[NXB];
  uint32_t tmem;
};

DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float sigmoid_kit(float g) { return rcp_ftz(__fadd_rn(1.f, ex2_ftz(__fmul_rn(-1.4426950408889634f, g)))); }
DEV float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
DEV float2 rbf2(float2 v) {   // round both lanes to bf16 and back
  const uint32_t w = pack_bf16(v.x, v.y);
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float2 lds64f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory"); return v; }
DEV float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
DEV void ldsm_x4_trans(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
DEV void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
// 16 lanes x (4 x 8) 32-bit columns in the mma-fragment layout (see k1.cu tmem_ld16x256_x4): r[4j], r[4j+1] -> lane t/4,
// cols 8j + 2(t%4) + {0,1};  r[4j+2], r[4j+3] -> lane t/4 + 8, same cols.
DEV void tmem_st16x256_x4(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
                  "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
}

template <int C, int H, int SAVE, int SMJ>
__global__ void __launch_bounds__(512, 1)
    k3g_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
              const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mxn,
              const float* __restrict__ g_in, const float* __restrict__ b_in, const float* __restrict__ g_out,
              const float* __restrict__ b_out, const __nv_bfloat16* __restrict__ ds, const __nv_bfloat16* __restrict__ xglob,
              float* __restrict__ mean_out,
              float* __restrict__ rs_out, int L, int tiles, float eps, float4* __restrict__ zero_buf, int zero_n4, int n2) {
  extern __shared__ __align__(1024) uint8_t sm[];      // no static shared memory: the dynamic window starts 1024-aligned
  using GG = Geo<C, H, SAVE>;
  constexpr int NTH = GG::NTHG;
  constexpr int XT = GG::XT, TT = GG::TT, WPB = GG::WPB, WG = GG::WG;
  constexpr int O_WG = GG::O_WG, O_WP = GG::O_WP, O_X = GG::O_X, O_T = GG::O_T, O_GB = GG::O_GB, O_GI = GG::O_GI, O_BAR = GG::O_BAR;
  constexpr uint32_t IDESC = idesc_bf16(128, C);
  // C = 64: the accumulators (64 columns each) are double buffered -- tile i + 1's products run during tile i's epilogue -- and the
  // training xn save goes through its own staging tiles, so the residual x is not reloaded
  constexpr int NAB = (C == 64 && K3_NAB) ? 2 : 1;
  constexpr bool XNS = SAVE == 1 && C == 64 && K3_XNS;
  auto tp = [](int b) { return T_P + (uint32_t)b * 128u; };
  auto tg = [](int b) { return NAB == 2 ? T_P + 64u + (uint32_t)b * 128u : T_G; };
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  // Training: every CTA takes the tiles of one j class (tile % (L / 128)), so the ds rows of all its tiles are the same 128 rows;
  // they are loaded once into TMEM. Inference keeps the plain round robin.
  const int LC = SAVE ? L / TOK : 1, jc = cta % LC, kc = cta / LC, nc = (G - jc + LC - 1) / LC;
  const int tiles_c = (tiles - jc + LC - 1) / LC;
  const int n_local = kc < tiles_c ? (tiles_c - kc + nc - 1) / nc : 0;
  auto tile_row0 = [&](int i) { return (jc + LC * (kc + nc * i)) * TOK; };
  const bool lat = K3_LAT && tiles <= G;   // every CTA has at most one tile
  constexpr bool WRES = K3_WRES && H / 64 <= NWP;
  const bool pdl = K3_PDL && tiles <= K3_PDL_TILES * G;   // the host launches with the PDL attribute exactly then
#if K3_EARLY
  const int nx0 = min(n_local, GG::NXG), nth0 = pdl ? 0 : min(2 * n_local, NTH), nwp0 = min(n_local * (H / 64), NWP);
#else
  constexpr int nx0 = 0, nth0 = 0, nwp0 = 0;
#endif
  if (K3_TRIG) pdl_trigger();   // the next kernel (b1s) may launch its CTAs as this grid's CTAs retire
  if (tid == 0) TL(0)

  if (tid == 0) {
    for (int b = 0; b < NXB; ++b) { mbar_init(&B.x_full[b], 1); mbar_init(&B.x_empty[b], 1); mbar_init(&B.xn_stored[b], 1); mbar_init(&B.xr_full[b], 1); }
    for (int s = 0; s < NTH; ++s) { mbar_init(&B.th_full[s], 1); mbar_init(&B.th_empty[s], 4); }
    mbar_init(&B.wg_full, 1);
    for (int s = 0; s < NWP; ++s) { mbar_init(&B.wp_full[s], 1); mbar_init(&B.wp_empty[s], 1); }
    mbar_init(&B.ds_full, 1);
    mbar_init(&B.xn_ready, 1); mbar_init(&B.xo_ready, 1); mbar_init(&B.a_free, 1);
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    fence_mbar_init();
    prefetch_tmap(&mx); prefetch_tmap(&mtri); prefetch_tmap(&mwp); prefetch_tmap(&mwg); prefetch_tmap(&my);
    if (SAVE == 1) prefetch_tmap(&mxn);
#if K3_EARLY
    // one-tile latency: every first load is issued here, before the TMEM allocation and the CTA barrier
    for (int i = 0; i < nx0; ++i) {
      mbar_expect_tx(&B.x_full[i], XT);
      for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + O_X + i * XT + kb * KBB, &mx, &B.x_full[i], kb * 64, tile_row0(i), EVICT_FIRST);
    }
    for (int k = 0; k < nth0; ++k) {
      const int row0 = tile_row0(k >> 1);
      mbar_expect_tx(&B.th_full[k], TT / 2);
      if constexpr (SMJ != 0) {
        int tq, ts;
        tok_split(row0 + 64 * (k & 1), n2, tq, ts);
        tma_load_4d(sm + O_T + k * (TT / 2), &mtri, &B.th_full[k], tq, 0, ts, 0, EVICT_FIRST);
      } else {
        tma_load_2d(sm + O_T + k * (TT / 2), &mtri, &B.th_full[k], row0 + 64 * (k & 1), 0, EVICT_FIRST);
      }
    }
    mbar_expect_tx(&B.wg_full, WG);
    for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + O_WG + kb * (C * 128), &mwg, &B.wg_full, kb * 64, 0, EVICT_LAST);
    for (int q = 0; q < nwp0; ++q) {
      mbar_expect_tx(&B.wp_full[q], WPB);
      tma_load_2d(sm + O_WP + q * WPB, &mwp, &B.wp_full[q], (q % (H / 64)) * 64, 0, EVICT_LAST);
    }
#endif
  }
  if (tid == 0 && (smem_u32(sm) & 1023u)) asm volatile("trap;");
  // training: clear the next backward's atomically accumulated gradient buffer (saves a separate memset launch)
  for (int k = cta * blockDim.x + tid; k < zero_n4; k += G * blockDim.x) zero_buf[k] = make_float4(0.f, 0.f, 0.f, 0.f);
  if (tid < H) {                                        // LN_out affine (LN_in reads its gamma / beta through L1)
    float* gb = reinterpret_cast<float*>(sm + O_GB);
    gb[tid] = g_out[tid]; gb[H + tid] = b_out[tid];
  }
  if (K3_GSMEM && tid < C) { reinterpret_cast<float*>(sm + O_GI)[tid] = g_in[tid]; reinterpret_cast<float*>(sm + O_GI)[C + tid] = b_in[tid]; }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  PROF_BEGIN
  if (tid == 0) TL(1)

  if (warp == 0) {
    if (lane == 0) {
      // ---------------------------------------------------------------- TMA producer
#if !K3_EARLY
      mbar_expect_tx(&B.wg_full, WG);
      for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + O_WG + kb * (C * 128), &mwg, &B.wg_full, kb * 64, 0, EVICT_LAST);
#endif
      int qw = 0;
      for (int i = 0; i < n_local; ++i) {
        const int b = i & 1, row0 = tile_row0(i);
        if (WRES && i > 0) break;   // resident Wp: tile 0's blocks stay
        for (int kb = 0; kb < H / 64; ++kb, ++qw) {
          const int s = qw % NWP, u = qw / NWP;
          if (qw < nwp0) continue;
          PW(2, mbar_wait(&B.wp_empty[s], (u & 1) ^ 1));
          mbar_expect_tx(&B.wp_full[s], WPB);
          tma_load_2d(sm + O_WP + s * WPB, &mwp, &B.wp_full[s], kb * 64, 0, EVICT_LAST);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      // ---------------------------------------------------------------- MMA: acc_p = xo Wp'^T (K 256), acc_g = xn Wg^T (K 128)
      mbar_wait(&B.wg_full, 0);
      TL(7)
      int qw = 0;
      for (int i = 0; i < n_local; ++i) {
        const int ab = i % NAB, ak = i / NAB;
        PW(0, mbar_wait(&B.acc_empty[ab], (ak & 1) ^ 1));
        PW(1, mbar_wait(&B.xn_ready, i & 1));
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < C / 16; ++k)
          umma_ts(tmem + tg(ab), tmem + T_XN + k * 8, desc_k_sw128(su + O_WG + (k >> 2) * (C * 128) + (k & 3) * 32), IDESC, k != 0);
        PW(2, mbar_wait(&B.xo_ready, i & 1));
        TLG(i, 4)
        tc_fence_after();
        for (int kb = 0; kb < H / 64; ++kb, ++qw) {
          const int s = WRES ? kb : qw % NWP, u = WRES ? 0 : qw / NWP;
          if (!WRES || i == 0) PW(3, mbar_wait(&B.wp_full[s], u & 1));
          tc_fence_after();
          if (i == 0 && kb == 0) TL(8)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            umma_ts(tmem + tp(ab), tmem + T_XO + kb * 32 + k * 8, desc_k_sw128(su + O_WP + s * WPB + k * 32), IDESC, (kb | k) != 0);
          if (!WRES) umma_commit(&B.wp_empty[s]);
        }
        if (i == 0) TL(9)
        umma_commit(&B.a_free);
        umma_commit(&B.acc_full[ab]);
      }
    }
  }
  // LN_out: warps 8-11 take token group 0 of their quarter, warps 4-7 (after LN_in) group 1
  constexpr bool LNOM = K3_LNOM != 0 && H == C;
  auto lnout_tile = [&](int i, int hsel, bool fin) {
    // ------------------------------------------------------------------ LN_out over 256 channels, fragment layout
    const int q = warp & 3;
    const int t4 = lane & 3, tr = lane >> 2;         // this thread: tokens tr, tr + 8 of a 16-token group; channel sub-pair t4
    const float* gbo = reinterpret_cast<const float*>(sm + O_GB);   // g_out [H] then b_out [H]
    {
      const int row0 = tile_row0(i);
      const int kslot = 2 * i + (q >> 1), slot = kslot % NTH;   // this warp's tokens lie in half q / 2 of the tile
      PW(0, mbar_wait(&B.th_full[slot], (kslot / NTH) & 1));
      if (warp == 8 && lane == 0) TLG(i, 3)
      if (lane == 0 && hsel == 0 && warp == 8) TT(i, 6)
      if (i == 0 && lane == 0 && hsel == 0 && warp == 8) TL(4)
      if (i == 0 && lane == 0 && hsel == 0 && warp == 10) TL(5)
      auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
#pragma unroll 1
      for (int hh = hsel; hh < hsel + 1; ++hh) {     // this warp's 16-token group (64 registers of channel pairs)
        // three passes over the swizzled tri tile (sum, centred squares, affine) instead of holding 64 channel pairs in registers
        const int tok0 = q * 32 + hh * 16;
        const int tc = (tok0 & 63) >> 3;
        const uint32_t base = su + O_T + slot * (TT / 2);
        auto ldj = [&](int j, uint32_t (&w)[4]) {
          const int mm = lane >> 3, ch = 16 * j + 8 * (mm & 1) + (lane & 7), qc = tc + (mm >> 1);
          ldsm_x4_trans(base + sw128(ch, qc), w[0], w[1], w[2], w[3]);
        };
        float mean[2], rsd[2];
#if K3_LN1
        {
          // one pass over the tri tile: sums of x - k and (x - k)^2 with a per-token pivot k (its channel 0, held by lane t4 == 0),
          // so the variance m2 / H - (m1 / H)^2 does not cancel (|m1 / H| is the distance of the mean from a sample, not the mean)
          uint32_t w0[4]; ldj(0, w0);
          const float kA = __shfl_sync(0xffffffffu, up(w0[0]).x, lane & ~3), kB = __shfl_sync(0xffffffffu, up(w0[2]).x, lane & ~3);
          const float2 nkA = make_float2(-kA, -kA), nkB = make_float2(-kB, -kB);
          float2 sA = make_float2(0.f, 0.f), sA2 = sA, sB = sA, sB2 = sA, qA = sA, qA2 = sA, qB = sA, qB2 = sA;
#pragma unroll 4
          for (int j = 0; j < ((K3_ABL & 2) ? 1 : H / 16); ++j) {
            uint32_t w[4]; ldj(j, w);
            float2 d;
            d = __fadd2_rn(up(w[0]), nkA); sA = __fadd2_rn(sA, d); qA = __ffma2_rn(d, d, qA);
            d = __fadd2_rn(up(w[1]), nkA); sA2 = __fadd2_rn(sA2, d); qA2 = __ffma2_rn(d, d, qA2);
            d = __fadd2_rn(up(w[2]), nkB); sB = __fadd2_rn(sB, d); qB = __ffma2_rn(d, d, qB);
            d = __fadd2_rn(up(w[3]), nkB); sB2 = __fadd2_rn(sB2, d); qB2 = __ffma2_rn(d, d, qB2);
          }
          float tA = __fadd_rn(__fadd_rn(sA.x, sA.y), __fadd_rn(sA2.x, sA2.y)), tB = __fadd_rn(__fadd_rn(sB.x, sB.y), __fadd_rn(sB2.x, sB2.y));
          float uA = __fadd_rn(__fadd_rn(qA.x, qA.y), __fadd_rn(qA2.x, qA2.y)), uB = __fadd_rn(__fadd_rn(qB.x, qB.y), __fadd_rn(qB2.x, qB2.y));
#pragma unroll
          for (int o = 1; o <= 2; o <<= 1) {
            tA = __fadd_rn(tA, __shfl_xor_sync(0xffffffffu, tA, o)); tB = __fadd_rn(tB, __shfl_xor_sync(0xffffffffu, tB, o));
            uA = __fadd_rn(uA, __shfl_xor_sync(0xffffffffu, uA, o)); uB = __fadd_rn(uB, __shfl_xor_sync(0xffffffffu, uB, o));
          }
          const float mA = __fmul_rn(tA, 1.f / H), mB = __fmul_rn(tB, 1.f / H);
          mean[0] = __fadd_rn(kA, mA); mean[1] = __fadd_rn(kB, mB);
          const float vA = fmaxf(__fmaf_rn(-mA, mA, __fmul_rn(uA, 1.f / H)), 0.f), vB = fmaxf(__fmaf_rn(-mB, mB, __fmul_rn(uB, 1.f / H)), 0.f);
          rsd[0] = rsqrt_ftz(__fadd_rn(vA, eps)); rsd[1] = rsqrt_ftz(__fadd_rn(vB, eps));
        }
#else
        float2 sA = make_float2(0.f, 0.f), sA2 = sA, sB = sA, sB2 = sA;
#pragma unroll 4
        for (int j = 0; j < H / 16; ++j) {
          uint32_t w[4]; ldj(j, w);
          sA = __fadd2_rn(sA, up(w[0])); sA2 = __fadd2_rn(sA2, up(w[1])); sB = __fadd2_rn(sB, up(w[2])); sB2 = __fadd2_rn(sB2, up(w[3]));
        }
        {
          float tA = __fadd_rn(__fadd_rn(sA.x, sA.y), __fadd_rn(sA2.x, sA2.y)), tB = __fadd_rn(__fadd_rn(sB.x, sB.y), __fadd_rn(sB2.x, sB2.y));
          tA = __fadd_rn(tA, __shfl_xor_sync(0xffffffffu, tA, 1)); tA = __fadd_rn(tA, __shfl_xor_sync(0xffffffffu, tA, 2));
          tB = __fadd_rn(tB, __shfl_xor_sync(0xffffffffu, tB, 1)); tB = __fadd_rn(tB, __shfl_xor_sync(0xffffffffu, tB, 2));
          mean[0] = __fmul_rn(tA, 1.f / H); mean[1] = __fmul_rn(tB, 1.f / H);
        }
        const float2 nA = make_float2(-mean[0], -mean[0]), nB = make_float2(-mean[1], -mean[1]);
        sA = sA2 = sB = sB2 = make_float2(0.f, 0.f);
#pragma unroll 4
        for (int j = 0; j < H / 16; ++j) {
          uint32_t w[4]; ldj(j, w);
          float2 d;
          d = __fadd2_rn(up(w[0]), nA); sA = __ffma2_rn(d, d, sA);
          d = __fadd2_rn(up(w[1]), nA); sA2 = __ffma2_rn(d, d, sA2);
          d = __fadd2_rn(up(w[2]), nB); sB = __ffma2_rn(d, d, sB);
          d = __fadd2_rn(up(w[3]), nB); sB2 = __ffma2_rn(d, d, sB2);
        }
        {
          float qA = __fadd_rn(__fadd_rn(sA.x, sA.y), __fadd_rn(sA2.x, sA2.y)), qB = __fadd_rn(__fadd_rn(sB.x, sB.y), __fadd_rn(sB2.x, sB2.y));
          qA = __fadd_rn(qA, __shfl_xor_sync(0xffffffffu, qA, 1)); qA = __fadd_rn(qA, __shfl_xor_sync(0xffffffffu, qA, 2));
          qB = __fadd_rn(qB, __shfl_xor_sync(0xffffffffu, qB, 1)); qB = __fadd_rn(qB, __shfl_xor_sync(0xffffffffu, qB, 2));
          rsd[0] = rsqrt_ftz(__fadd_rn(__fmul_rn(qA, 1.f / H), eps)); rsd[1] = rsqrt_ftz(__fadd_rn(__fmul_rn(qB, 1.f / H), eps));
        }
#endif
        if (SAVE == 1 && t4 == 0) {
#pragma unroll
          for (int tb = 0; tb < 2; ++tb) {
            const int tok = row0 + tok0 + tr + 8 * tb;
            mean_out[tok] = mean[tb];
            rs_out[tok] = rsd[tb];
          }
        }
        if (i >= 1) PW(1, mbar_wait(&B.a_free, (i - 1) & 1));
        tc_fence_after();
#pragma unroll 1
        for (int jq = 0; jq < H / 64; ++jq) {         // affine + TMEM store, 16 registers at a time
          uint32_t rr[16];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int j = 4 * jq + jj;
            uint32_t w[4]; ldj(j, w);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              if (K3_ABL & 1) { rr[4 * jj + e] = w[e]; continue; }
              const int ch = 16 * j + 8 * (e & 1) + 2 * t4, tb = e >> 1;
              const float2 gg = lds64f(smem_u32(gbo + ch)), bb = lds64f(smem_u32(gbo + H + ch));
              const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w[e]), make_float2(-mean[tb], -mean[tb])),
                                                     make_float2(rsd[tb], rsd[tb])), gg, bb);
              rr[4 * jj + e] = pack_bf16(y.x, y.y);
            }
          }
          tmem_st16x256_x4(tmem + ((uint32_t)tok0 << 16) + T_XO + jq * 32, rr);
        }
        __syncwarp(); if (lane == 0) mbar_arrive(&B.th_empty[slot]);   // this warp's rows of the tri half are read
      }
      if (fin) {
        tmem_wait_st();
        tc_fence_before();
        named_bar_sync(2, LNOM ? 128 : 256);
        if (i == 0 && lane == 0 && warp == 8) TL(6)
        if (warp == 8 && lane == 0) mbar_arrive(&B.xo_ready);
      }
    }
  };
  if (warp == 2 && lane == 0) {
    // tri halves through the 3-slot ring
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // t is the previous kernel's output
    for (int i = 0; i < n_local; ++i) {
      const int row0 = tile_row0(i);
#pragma unroll 1
      for (int h = 0; h < 2; ++h) {
        const int k = 2 * i + h, slot = k % NTH;
        if (k < nth0) continue;
        if (k >= NTH) PW(1, mbar_wait(&B.th_empty[slot], ((k / NTH) - 1) & 1));
        mbar_expect_tx(&B.th_full[slot], TT / 2);
        if (h == 0) TLG(i, 0)
        if constexpr (SMJ != 0) {          // sample-major t (n2 tokens per sample): the 4-D map of tmap::make_chmajor
          int tq, ts;
          tok_split(row0 + 64 * h, n2, tq, ts);
          tma_load_4d(sm + O_T + slot * (TT / 2), &mtri, &B.th_full[slot], tq, 0, ts, 0, EVICT_FIRST);
        } else {
          tma_load_2d(sm + O_T + slot * (TT / 2), &mtri, &B.th_full[slot], row0 + 64 * h, 0, EVICT_FIRST);
        }
      }
    }
  }
  if (warp == 3 && lane == 0) {
    // x tiles (double buffered), off the Wp-ring producer's path
    for (int i = 0; i < n_local; ++i) {
      const int b = i % GG::NXG, row0 = tile_row0(i);
      if (i < nx0) continue;
      TT(i, 0)
      if (i >= GG::NXG) PW(0, mbar_wait(&B.x_empty[b], ((i / GG::NXG) - 1) & 1));
      TT(i, 7)
      mbar_expect_tx(&B.x_full[b], XT);
      TLG(i, 1)
      for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + O_X + b * XT + kb * KBB, &mx, &B.x_full[b], kb * 64, row0, EVICT_FIRST);
#if K3_PFX
      if (SAVE == 0)   // inference only: training keeps its validated schedule
      if (i + 1 < n_local) {   // L2 prefetch of the next tile's x boxes (its slot frees ~one tile period later; the load then hits L2, not HBM)
        const int nrow = tile_row0(i + 1);
        for (int kb = 0; kb < C / 64; ++kb)
          asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(reinterpret_cast<uint64_t>(&mx)), "r"(kb * 64), "r"(nrow) : "memory");
      }
#endif
    }
  }
  if (SAVE == 1 && !XNS && warp == 3 && lane == 16) {
    // training: once the xn save has left the x tile, reload x (L2) into it for the epilogue's residual
    for (int i = 0; i < n_local; ++i) {
      const int b = i % GG::NXG, row0 = tile_row0(i);
      PW(0, mbar_wait(&B.xn_stored[b], (i / GG::NXG) & 1));
      mbar_expect_tx(&B.xr_full[b], XT);
      for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + O_X + b * XT + kb * KBB, &mx, &B.xr_full[b], kb * 64, row0, EVICT_FIRST);
    }
  }
  if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------ LN_in: row r = 32 (warp % 4) + lane
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    for (int i = 0; i < n_local; ++i) {
      const int b = i % GG::NXG, row0 = tile_row0(i);
      PW(0, mbar_wait(&B.x_full[b], (i / GG::NXG) & 1));
      if (r == 0) TT(i, 1)
      if (warp == 4 && lane == 0) TLG(i, 2)
      if (r == 0 && i == 0) TL(2)
      const uint32_t xr = su + O_X + b * XT;
      auto ldx = [&](int q) { return lds128(xr + (q >> 3) * KBB + sw128(r, q & 7)); };
#if K3_LNIN1
      // LN_in statistics in one pass: sums of x - k and (x - k)^2 about the row's first element k (the pivot keeps the variance from cancelling)
      float mean, rs;
      float2 nm;
      {
        const float kp = up(ldx(0).x).x;
        const float2 nk = make_float2(-kp, -kp);
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0;
        K3_UNROLL(K3_LNU)
        for (int q = 0; q < C / 8; ++q) {
          const uint4 t = ldx(q);
          float2 d;
          d = __fadd2_rn(up(t.x), nk); a0 = __fadd2_rn(a0, d); c0 = __ffma2_rn(d, d, c0);
          d = __fadd2_rn(up(t.y), nk); a1 = __fadd2_rn(a1, d); c1 = __ffma2_rn(d, d, c1);
          d = __fadd2_rn(up(t.z), nk); a2 = __fadd2_rn(a2, d); c2 = __ffma2_rn(d, d, c2);
          d = __fadd2_rn(up(t.w), nk); a3 = __fadd2_rn(a3, d); c3 = __ffma2_rn(d, d, c3);
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        c0 = __fadd2_rn(__fadd2_rn(c0, c1), __fadd2_rn(c2, c3));
        const float m1 = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C);
        mean = __fadd_rn(kp, m1);
        const float var = fmaxf(__fmaf_rn(-m1, m1, __fmul_rn(__fadd_rn(c0.x, c0.y), 1.f / C)), 0.f);
        rs = rsqrt_ftz(__fadd_rn(var, eps));
        nm = make_float2(-mean, -mean);
      }
#else
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
      K3_UNROLL(K3_LNU)
      for (int q = 0; q < ((K3_ABL & 4) ? 1 : C / 8); ++q) {
        const uint4 t = ldx(q);
        a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float mean = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C);
      const float2 nm = make_float2(-mean, -mean);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
      K3_UNROLL(K3_LNU)
      for (int q = 0; q < ((K3_ABL & 4) ? 1 : C / 8); ++q) {
        const uint4 t = ldx(q);
        float2 d;
        d = __fadd2_rn(up(t.x), nm); a0 = __ffma2_rn(d, d, a0);
        d = __fadd2_rn(up(t.y), nm); a1 = __ffma2_rn(d, d, a1);
        d = __fadd2_rn(up(t.z), nm); a2 = __ffma2_rn(d, d, a2);
        d = __fadd2_rn(up(t.w), nm); a3 = __ffma2_rn(d, d, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C), eps));
#endif
      const float2 rr = make_float2(rs, rs);
      if (i >= 1) PW(1, mbar_wait(&B.a_free, (i - 1) & 1));   // tile i-1's products have read the xn / xo TMEM buffers
      tc_fence_after();
      K3_UNROLL(K3_LNU4)
      for (int q4 = 0; q4 < ((K3_ABL & 16) ? 1 : C / 16); ++q4) {
        uint32_t o[8];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int q = q4 * 2 + qq;
          const uint4 t = ldx(q);
#if K3_GSMEM
          const uint32_t gb = su + O_GI + q * 32;
          const float4 ga = lds128f(gb), gb4 = lds128f(gb + 16), ba = lds128f(gb + 4 * C), bb = lds128f(gb + 4 * C + 16);
#else
          const float4 ga = __ldg(reinterpret_cast<const float4*>(g_in) + 2 * q), gb4 = __ldg(reinterpret_cast<const float4*>(g_in) + 2 * q + 1);
          const float4 ba = __ldg(reinterpret_cast<const float4*>(b_in) + 2 * q), bb = __ldg(reinterpret_cast<const float4*>(b_in) + 2 * q + 1);
#endif
          auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
            const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
            return pack_bf16(y.x, y.y);
          };
          o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
          o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
          o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
          o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
          if (SAVE == 1) sts128((XNS ? su + GG::O_XS + b * XT : xr) + (q >> 3) * KBB + sw128(r, q & 7), o[4 * qq], o[4 * qq + 1], o[4 * qq + 2], o[4 * qq + 3]);
        }
        tmem_st8(trow + T_XN + q4 * 8, o);
      }
      tmem_wait_st();
      tc_fence_before();
      if (SAVE == 1) fence_async_smem();
      named_bar_sync(1, 128);
      if (r == 0 && i == 0) TL(3)
      if (r == 0) {
        TT(i, 2)
        mbar_arrive(&B.xn_ready);
        if (SAVE == 1) {
          for (int kb = 0; kb < C / 64; ++kb) tma_store_2d(&mxn, sm + (XNS ? GG::O_XS : O_X) + b * XT + kb * KBB, kb * 64, row0);
          bulk_commit();
          bulk_wait_read<0>();
          mbar_arrive(&B.xn_stored[b]);
        }
      }
      if (!LNOM) lnout_tile(i, 1, true);              // then this warp's LN_out token group
    }
  }
  else if (warp >= 8 && warp < 12) {
    for (int i = 0; i < n_local; ++i) {
      lnout_tile(i, 0, !LNOM);
      if (LNOM) lnout_tile(i, 1, true);
    }
  }
  // ------------------------------------------------------------------ epilogue: thread = token row; part 0: warps 12-15 (all the
  // channels, or the lower half when warps 4-7 take the upper half: part 1, one-tile CTAs)
  auto epilogue = [&](const int part) {
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    // this CTA's ds rows (sample sb, rows jc*128 + r) -> TMEM (lane r, 64 columns of bf16 pairs). ds is [B, L, C]: the tiles of a j class
    // run over the samples in order (L tile rows each), so it is reloaded when a tile's sample changes (once per CTA for B = 1).
    int cur_sb = -1;
    auto load_ds = [&](int sb) {
      const uint4* dsr = reinterpret_cast<const uint4*>(ds + ((size_t)sb * L + jc * TOK + r) * C);
#pragma unroll
      for (int q8 = 0; q8 < C / 16; ++q8) {
        const uint4 a0 = __ldg(dsr + 2 * q8), a1 = __ldg(dsr + 2 * q8 + 1);
        const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        tmem_st8(trow + T_DS + q8 * 8, w);
      }
      tmem_wait_st();
    };
    for (int i = 0; i < n_local; ++i) {
      const int b = i % GG::NXG, row0 = tile_row0(i);
      if (SAVE) {
        const int sb = (row0 / TOK) / (L * LC);
        if (part == 0 && sb != cur_sb) {
          load_ds(sb); cur_sb = sb;
          if (lat) { tc_fence_before(); named_bar_sync(4, 128); if (r == 0) mbar_arrive(&B.ds_full); }
        }
        if (part == 1) { mbar_wait(&B.ds_full, 0); tc_fence_after(); }
      }
      // training: the residual x has been reloaded into the x tile after the xn save; ds comes from TMEM
      const int ab = i % NAB, ak = i / NAB;
      PW(0, mbar_wait(&B.acc_full[ab], ak & 1));
      if (r == 0 && part == 0) TLG(i, 5)
      tc_fence_after();
      if (r == 0 && part == 0) TT(i, 3)
      if (r == 0 && i == 0 && part == 0) TL(10)
      if (SAVE == 1 && !XNS) PW(1, mbar_wait(&B.xr_full[b], (i / GG::NXG) & 1));     // x reloaded into this x tile
      const uint32_t xr = su + O_X + b * XT;
#pragma unroll
      constexpr int EW = K3_EPW;                      // output channels per step (16: no spills at C = 128)
      const int cq0 = part ? C / EW / 2 : 0, cq1 = (lat && part == 0) ? C / EW / 2 : C / EW;
      for (int cq = cq0; cq < cq1; ++cq) {
        uint32_t pv[EW], gv[EW], dsw[EW / 2];
#if K3_EPW == 32
        tmem_ld32(trow + tp(ab) + cq * 32, pv);
        tmem_ld32(trow + tg(ab) + cq * 32, gv);
        if (SAVE) tmem_ld16(trow + T_DS + cq * 16, dsw);
#else
        tmem_ld16(trow + tp(ab) + cq * 16, pv);
        tmem_ld16(trow + tg(ab) + cq * 16, gv);
        if (SAVE) tmem_ld8(trow + T_DS + cq * 8, dsw);
#endif
        tmem_wait_ld();
        if (!lat && cq == C / EW - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[ab]); }
#pragma unroll
        for (int q2 = 0; q2 < EW / 8; ++q2) {         // 8 channels = one 16-byte chunk of the row
          const int q = cq * (EW / 8) + q2;
          const uint32_t a = xr + (q >> 3) * KBB + sw128(r, q & 7);
          const uint4 xv = lds128(a);
          uint32_t o[4];
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
          const uint32_t dw[4] = {SAVE ? dsw[4 * q2] : 0u, SAVE ? dsw[4 * q2 + 1] : 0u, SAVE ? dsw[4 * q2 + 2] : 0u, SAVE ? dsw[4 * q2 + 3] : 0u};
#pragma unroll
          for (int k = 0; k < 4; ++k) {                // two channels at a time, packed f32x2 (same per-lane IEEE sequence)
            if (K3_ABL & 8) { o[k] = xw[k]; continue; }
            const int c = q2 * 8 + 2 * k;
            float2 pp = make_float2(__uint_as_float(pv[c]), __uint_as_float(pv[c + 1]));
            float2 gg = make_float2(__uint_as_float(gv[c]), __uint_as_float(gv[c + 1]));
            const float2 xres = make_float2(bf16lo(xw[k]), bf16hi(xw[k]));
            if (SAVE) { pp = rbf2(pp); gg = rbf2(gg); }
#if K3_SIG == 0
            const float2 t = __fmul2_rn(gg, make_float2(-1.4426950408889634f, -1.4426950408889634f));
            const float2 dnm = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)), make_float2(1.f, 1.f));
            const float2 sg = make_float2(rcp_ftz(dnm.x), rcp_ftz(dnm.y));
#elif K3_SIG == 2   // sigmoid(g) = 0.5 tanh(g / 2) + 0.5: one MUFU op per element
            const float2 h = __fmul2_rn(gg, make_float2(0.5f, 0.5f));
            float2 th;
            asm("tanh.approx.f32 %0, %1;" : "=f"(th.x) : "f"(h.x));
            asm("tanh.approx.f32 %0, %1;" : "=f"(th.y) : "f"(h.y));
            const float2 sg = __ffma2_rn(th, make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
#else   // kit ex2, reciprocal by 3 Newton steps on the FMA pipe (one MUFU op per element)
            float2 t = __fmul2_rn(gg, make_float2(-1.4426950408889634f, -1.4426950408889634f));
            t.x = fminf(t.x, 126.f); t.y = fminf(t.y, 126.f);
            const float2 dnm = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)), make_float2(1.f, 1.f));
            float2 sg = make_float2(__int_as_float(0x7EF311C3 - __float_as_int(dnm.x)), __int_as_float(0x7EF311C3 - __float_as_int(dnm.y)));
            {
              const float2 nd = make_float2(-dnm.x, -dnm.y);
#pragma unroll
              for (int it = 0; it < 3; ++it) sg = __fmul2_rn(sg, __ffma2_rn(nd, sg, make_float2(2.f, 2.f)));
            }
#endif
            if (SAVE) {
              const float2 dsv = make_float2(bf16lo(dw[k]), bf16hi(dw[k]));
              const float2 y = __ffma2_rn(__fmul2_rn(pp, sg), dsv, xres);
              o[k] = pack_bf16(y.x, y.y);
            } else {
              const float2 pr = __fmul2_rn(sg, pp);   // y = bf16(x + bf16(sigmoid(g) p)): one packed bf16 add of the two bf16 words
              asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(o[k]) : "r"(xw[k]), "r"(pack_bf16(pr.x, pr.y)));
            }
          }
          sts128(a, o[0], o[1], o[2], o[3]);
        }
      }
      fence_async_smem();
      named_bar_sync(3, lat ? 256 : 128);
      if (r == 0 && part == 0) {
        TT(i, 4)
        for (int kb = 0; kb < C / 64; ++kb) tma_store_2d(&my, sm + O_X + b * XT + kb * KBB, kb * 64, row0);
        bulk_commit(); TLG(i, 6)
        bulk_wait_read<0>();
        TT(i, 5)
        mbar_arrive(&B.x_empty[b]); TLG(i, 7)
      }
    }
    if (r == 0 && part == 0) TL(11)
    if (r == 0 && part == 0) bulk_wait<0>();
    if (r == 0 && part == 0) TL(12)
  };
  if (warp >= 12) epilogue(0);
  else if (lat && warp >= 4 && warp < 8) epilogue(1);
  if (lane == 0 && warp == 0) PROF_END(0);
  if (lane == 0 && warp == 1) PROF_END(1);
  if (lane == 0 && warp == 4) PROF_END(2);
  if (lane == 0 && warp == 8) PROF_END(3);
  if (lane == 0 && warp == 12) PROF_END(4);
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TL(13)
  if (warp == 2) tmem_dealloc(tmem, 512);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k3g

// x [M, C] bf16, tri [H, M] bf16, wp_perm [C, H] (columns in the TMEM K order, written by k1w_prep as wpp), wg [C, C], y [M, C];
// training (save = 1): ds [L, C] bf16, xn_out [M, C], mean_out / rs_out [M] fp32 (then L % 128 == 0); save = 2: the training
// epilogue with ds and no saves (a no-grad forward with dropout).
void k3g_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wp_perm, torch::Tensor wg, torch::Tensor g_in, torch::Tensor b_in,
                 torch::Tensor g_out, torch::Tensor b_out, torch::Tensor y, int64_t L, double eps, int64_t save,
                 c10::optional<torch::Tensor> ds, c10::optional<torch::Tensor> xn_out, c10::optional<torch::Tensor> mean_out,
                 c10::optional<torch::Tensor> rs_out, int64_t grid, c10::optional<torch::Tensor> zero_buf) {
  using namespace k3g;
  const int64_t M = x.size(0), C = x.size(1), H = tmap::chmajor_channels(tri);   // tri [H, ..] or sample-major [H / 64, B, 64, L, L]
  TORCH_CHECK(tri.numel() == H * M && M % TOK == 0 && M % (L * L) == 0 && (!save || L % TOK == 0));   // M = B L L, tokens b-major
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  const int n2 = tri.dim() == 5 ? (int)(L * L) : 0;   // 0: the [H, M] map as before
  TORCH_CHECK(!n2 || (C == 64 && save == 1), "k3g_forward: sample-major t is the D64 training forward only");
  auto mt = n2 ? tmap::make_chmajor(tri, n2, 64, (uint32_t)H, CU_TENSOR_MAP_SWIZZLE_128B)
               : tmap::make(tri.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)H}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwp = tmap::make(wp_perm.data_ptr(), bf, {(uint64_t)H, (uint64_t)C}, {(uint64_t)H * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto my = tmap::make(y.data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap mxn = my;
  const __nv_bfloat16* dsp = nullptr;
  float *mo = nullptr, *ro = nullptr;
  TORCH_CHECK(save >= 0 && save <= 2);
  if (save) {
    TORCH_CHECK(ds.has_value());
    TORCH_CHECK(ds->numel() == (M / (L * L)) * L * C, "k3g_forward: ds must be [B, L, C]");
    dsp = reinterpret_cast<const __nv_bfloat16*>(ds->data_ptr());
  }
  if (save == 1) {
    TORCH_CHECK(xn_out.has_value() && mean_out.has_value() && rs_out.has_value());
    mxn = tmap::make(xn_out->data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
    mo = mean_out->data_ptr<float>();
    ro = rs_out->data_ptr<float>();
  }
  const int tiles = (int)(M / TOK);
  float4* zp = nullptr;
  int zn4 = 0;
  if (zero_buf.has_value()) {
    TORCH_CHECK(zero_buf->is_contiguous() && zero_buf->scalar_type() == torch::kFloat32 && zero_buf->numel() % 4 == 0);
    zp = reinterpret_cast<float4*>(zero_buf->data_ptr<float>());
    zn4 = (int)(zero_buf->numel() / 4);
  }
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, tiles);
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto launch = [&](auto kern, int smem) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(g); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = (K3_PDL && tiles <= K3_PDL_TILES * g) ? 1 : 0;   // the kernel computes the same predicate for its t-load gate
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, mx, mt, mwp, mwg, my, mxn, (const float*)g_in.data_ptr<float>(), (const float*)b_in.data_ptr<float>(),
                                      (const float*)g_out.data_ptr<float>(), (const float*)b_out.data_ptr<float>(), dsp, xp, mo, ro, (int)L, tiles,
                                      (float)eps, zp, zn4, n2));
  };
#define K3G_CASE(CC, HH) \
  if (C == CC && H == HH) { \
    if (n2) launch(k3g_kernel<CC, HH, 1, CC == 64>, Geo<CC, HH, 1>::SMEM); \
    else if (save == 1) launch(k3g_kernel<CC, HH, 1, 0>, Geo<CC, HH, 1>::SMEM); \
    else if (save == 2) launch(k3g_kernel<CC, HH, 2, 0>, Geo<CC, HH, 2>::SMEM); \
    else launch(k3g_kernel<CC, HH, 0, 0>, Geo<CC, HH, 0>::SMEM); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K3G_CASE(64, 64) K3G_CASE(64, 128) K3G_CASE(64, 256) K3G_CASE(128, 128) K3G_CASE(128, 256)
#undef K3G_CASE
  TORCH_CHECK(false, "k3g_forward: unsupported (C, H) = (", C, ", ", H, ")");
}

torch::Tensor k3g_tt() {
#ifdef K3G_TT
  torch::Tensor h = torch::empty({32, 8}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tt, sizeof(unsigned long long) * 32 * 8);
  return h;
#else
  return torch::empty({0});
#endif
}

torch::Tensor k3g_tl() {
#ifdef TM_TL
  torch::Tensor h = torch::empty({160, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tl, sizeof(unsigned long long) * 160 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}

torch::Tensor k3g_tlg() {
#ifdef K3G_TLG
  torch::Tensor h = torch::empty({480, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tlg, sizeof(unsigned long long) * 480 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}
