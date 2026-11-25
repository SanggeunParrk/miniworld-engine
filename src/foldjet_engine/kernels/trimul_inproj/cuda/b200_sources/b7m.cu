// b7m.cu -- TriMul B7 (input-side backward) on B200 for d_pair C = 64, one CTA per 128-token tile, end to end.
//
// At C = 64 the whole packed W1 (NCH chunks x [128 rows][64], <= 64 KB) and Wg stay in shared memory and dW1 (NCH x 64
// TMEM columns) stays in TMEM, so the b7r / b7s source -> L2 ring -> consumer hand-over is not needed: each persistent CTA
//   for every chunk c of its tile:  pre = xn W1_c^T (TMEM) -> dg / dp (8 warps, fragment layout, as b7r) -> dgp smem tile
//                                   dxn += dgp W1_c,  dW1_c += dgp^T xn          (tcgen05, fp32 accumulate)
//   last:                           dxn += dGout Wg -> 4 epilogue warps (one row per thread): LN_in backward + residual -> dx,
//                                   dgi / dbi; they overlap the next tile's chunks (dxn double buffered in TMEM).
// Same math and rounding points as b7r.cu: dxn is one bf16 rounding of the fp32 sum over K = 64 + 128 NCH.
#include "sm100.cuh"
#include "tmap.h"
#include "pt_tl.cuh"

using namespace sm100;

namespace b7m {
DEV void tma_load_5d(void* dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2, int c3, int c4, uint64_t hint) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "l"(hint)
      : "memory");
}
// dA [P / 64, B, 64, L, L] as (token mod 64, channel, token / 64, sample, chunk): box (64, 64, 2, 1, 1) = a chunk's 128-token tile as
// two SW128 [64 ch][64 tok] halves, one box
inline CUtensorMap da5_map(const torch::Tensor& t, int64_t n2) {
  const int64_t nb = t.size(1), nc = t.size(0);
  TORCH_CHECK(t.dim() == 5 && t.is_contiguous() && t.size(2) == 64 && n2 % 64 == 0, "da5_map: sample-major dA");
  return tmap::make(t.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {64, 64, (uint64_t)(n2 / 64), (uint64_t)nb, (uint64_t)nc},
                    {(uint64_t)n2 * 2, 128, (uint64_t)(64 * n2) * 2, (uint64_t)(nb * 64 * n2) * 2}, {64, 64, 2, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
}

constexpr int C = 64, TOK = 128;
constexpr int XT = TOK * C * 2;            // 16 KB: [128][64] SW128 tile (xn, dGout, x, dy, a W1 chunk half ... )
constexpr int DAT = 64 * TOK * 2;          // 16 KB: dA chunk [64 ch][128 tok], two 64-token halves
constexpr int GT = TOK * 128 * 2;          // 32 KB: dgp tile [128 tok][gate 64 | proj 64]
#ifndef B7M_NXI
#define B7M_NXI 2
#endif
template <int NCH>
struct Geo {
  static constexpr int S_W1 = 0, S_WG = S_W1 + NCH * XT, S_X = S_WG + 8192, S_DA = S_X + 2 * XT, S_DGP = S_DA + 2 * DAT,
                       S_GO = S_DGP + GT, S_XI = S_GO + XT;
  // the epilogue's x / dy tiles are double buffered where shared memory allows (NCH = 2): the next tile's HBM reads run during
  // this tile's epilogue instead of after its dx store
  static constexpr int NXI = (NCH == 2 && B7M_NXI == 2) ? 2 : 1;
  static constexpr int S_DY = S_XI + NXI * XT, O_BAR = S_DY + NXI * XT, O_GI = O_BAR + 512, SMEM = O_GI + 256;
  static_assert(SMEM <= 232448, "smem");
  static constexpr int NPB = NCH == 4 ? 1 : 2;               // pre-activation TMEM buffers
  static constexpr uint32_t T_DW = 0, T_PRE = NCH * 64, T_ACC = 384;
  static_assert(T_PRE + NPB * 128 <= T_ACC, "tmem");
};
constexpr uint32_t ID_PRE = idesc_bf16_mj(128, 128, 0, 0);
constexpr uint32_t ID_DW = idesc_bf16_mj(128, C, 1, 1);
constexpr uint32_t ID_DX = idesc_bf16_mj(128, C, 0, 1);

struct Bars {
  uint64_t w_full, x_full[2], x_free[2], da_full[2], da_free[2], pre_full[2], pre_free[2], dgp_ready, dgp_free, go_full, go_free,
      acc_full[2], acc_empty[2], xi_full[2], xi_free[2], ln_last;
  uint32_t tmem;
};

DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV float2 up(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); }
DEV void ldsm_x4n(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
DEV void ldsm_x4_trans(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float2 lds64g(uint32_t ad) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(ad) : "memory"); return v; }
DEV void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
DEV void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

// Pair mask from the [B L] token mask: pair (i, j) of sample s (token (s L + i) L + j, b-major) is kept when tokens s L + i and
// s L + j are. A 128-token tile lies in one row i (L % 128 == 0); (s, i, j0) of a CTA's tile is advanced incrementally (no integer
// division on the elementwise warps' path: three divisions per token cost b7g ~15 %).
struct PairIdx {
  int s, i, j, dq, dr;      // dq, dr: the CTA's tile step (step rows = dq L + dr), split once
  DEV void init(int row0, int step, int L) { s = row0 / (L * L); i = (row0 / L) % L; j = row0 % L; dq = step / L; dr = step % L; }
  DEV void advance(int L) {   // no loop over L-sized steps (b7m steps 148 x 128 tokens: ~25 per tile at L768)
    j += dr; i += dq;
    if (j >= L) { j -= L; ++i; }
    while (i >= L) { i -= L; ++s; }
  }
  // the two tokens' masks of this thread (tile offsets tok, tok + 8): both loads issued together, no short-circuit
  DEV void mask2(const uint8_t* tm, int tok, int L, float& m0, float& m1) const {
    if (tm == nullptr) { m0 = m1 = 1.f; return; }
    const uint8_t* row = tm + s * L;
    const uint32_t ri = __ldg(row + i), a0 = __ldg(row + j + tok), a1 = __ldg(row + j + tok + 8);
    m0 = (ri & a0) ? 1.f : 0.f;
    m1 = (ri & a1) ? 1.f : 0.f;
  }
};

#if defined(PT_TL) && PT_TL
// phase trace (-DPT_TL=1): 0 entry (globaltimer), 1 set-up done, 2 loader past griddepcontrol.wait, 3 MMA has the weights, 4 / 5 dg-dp
// warps have chunk 0 pre / dA, 6 chunk 0 dgp handed over, 7 last chunk pre, 8 last chunk dgp handed over, 9 LN warps have x / dy,
// 10 LN warps have dxn, 11 dx store issued, 12 dW1 flushed, 13 lnpart written, 14 before TMEM release, 15 exit (globaltimer)
__device__ unsigned long long g_pt7[160][24];   // 16..23: the split LN (see ln_tile)
#define P7(s) if constexpr (!FU) PTM(g_pt7, s)
#else
#define P7(s) ((void)0)
#endif
struct Args {
  const uint8_t* tmask;     // [B L] token mask or null
  const float* gi;
  float *dw1, *lnpart;      // dw1 [grid][128 NCH][64]: this CTA's dW1 partial; lnpart [grid][dgi 64 | dbi 64]: this CTA's row
                            // (both summed in a fixed order by grad_reduce: deterministic)
  int tiles, n2, L;         // n2 > 0: dpl sample-major, tokens per sample (4-D map, tmap::make_chmajor); 0: [P, M]
  float eps;
  // xg > 0 (2026-11-17 part 6): xg extra CTAs past the tile CTAs sum the output-side partials (b1u / b1v / b1s: xpf [xnf][D (D + R)],
  // xlo [xnl][2R]) into their regions of the gradient buffer xout (grad_reduce's layout and arithmetic, the same bits) on the SMs the
  // tile CTAs leave free; grad_reduce then sums only this kernel's partials
  const float *xpf = nullptr, *xlo = nullptr;
  float* xout = nullptr;
  int xg = 0, xnf = 0, xnl = 0, xR = 0;
};

#ifndef B7M_TRIG
#define B7M_TRIG 0   // 1: launch_dependents at entry (grad_reduce scheduled early, waiting): measured 2-4 us slower in training
                     // 2 (part 7 probe): one thread (tid 384) when the LN warps get the last tile's dxn accumulator
#endif
#ifndef B7M_FRAG
#define B7M_FRAG 1   // LN_in backward warps in the MMA fragment layout (2026-11-17 part 4); 0: one row per thread
#endif
#ifndef B7M_DA1
#define B7M_DA1 1   // (2026-11-18 part 11) sample-major dA: a chunk's two 64-token halves in ONE 3-D-in-token TMA box (5-D map): fewer boxes per tile
#endif
#ifndef B7M_ML
#define B7M_ML 0   // 1: the loader's TMA loads issued by several lanes (2026-11-17 part 5: D64 L256 0.0954 -> 0.0974, hidden 128 L256 0.1323 -> 0.1342 ms; L128 level; not taken)
#endif
#ifndef B7M_GOE
#define B7M_GOE 0   // (2026-11-18 part 10) 1: the last chunk's dGout Wg MMAs + acc_full commit before its dW1 MMAs (bitwise; dxn 1.1 us earlier in the
                    // phase trace, step level: not taken)
#endif
#ifndef B7M_BWL
#define B7M_BWL 0   // (part 10) 1: the last dx store's completion waited after the dgi / dbi sums (bitwise; level: not taken)
#endif
#ifndef B7M_XNE
#define B7M_XNE 0   // (part 10) 1: the first tile's xn loaded before griddepcontrol.wait (bitwise; level: not taken)
#endif
#ifndef B7M_FTS
#define B7M_FTS 1   // (part 10) the dW1 flush by TMA stores (same values)
#endif
#ifndef B7M_LNH
#define B7M_LNH 1   // (2026-11-18 part 10) the last tile's LN_in backward rows split over warps 12-15 and warps 0-3 (B7M_FRAG, NCH 4 only)
#endif
template <int V>
struct IC { static constexpr int value = V; };
#ifndef B7M_PDL
#define B7M_PDL 1   // a programmatic dependent of the contraction backward: set-up, weights, xn / x / dy / dGout loads overlap its tail
#endif
// Phase dependency of the fused small-L backward (b7f): spin (acquire) until *cnt == need, then order the async proxy (TMA reads of
// what other CTAs wrote) after it.
DEV void dep_wait(const uint32_t* cnt, uint32_t need) {
  uint32_t v;
  do {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
    if (v < need) __nanosleep(20);
  } while (v < need);
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// The kernel's body as a device function (2026-11-17 part 5): b7m_kernel calls it with (blockIdx.x, gridDim.x); the fused small-L
// backward (b7f) with FU = true: a virtual CTA id, its own TMEM, and the dA dependency as a counter wait instead of griddepcontrol.wait.
template <int NCH, int SMJ, bool FU>
DEV void b7m_body(const CUtensorMap& mxn, const CUtensorMap& mda, const CUtensorMap& mw1, const CUtensorMap& mgo, const CUtensorMap& mwg,
                  const CUtensorMap& mdx, const CUtensorMap& mx, const CUtensorMap& mdy, const CUtensorMap& mdw, const Args& a, uint8_t* sm,
                  const int cta, const int G, const uint32_t tmem_fu, const uint32_t* dep, const uint32_t dep_n) {
  using G_ = Geo<NCH>;
  constexpr int NPB = G_::NPB;
  constexpr uint32_t T_DW = G_::T_DW, T_PRE = G_::T_PRE, T_ACC = G_::T_ACC;
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_local = a.tiles > cta ? (a.tiles - cta + G - 1) / G : 0;   // tiles t = cta + G i
  const int n_chunks = n_local * NCH;
  PT_BEGIN
  if constexpr (!FU) PT_G(g_pt7, 0);
  if (!FU && B7M_TRIG == 1) pdl_trigger();   // grad_reduce may launch (it waits for this grid)
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.x_full[b], 1); mbar_init(&B.x_free[b], 1); mbar_init(&B.da_full[b], 1); mbar_init(&B.da_free[b], 8);
      mbar_init(&B.pre_full[b], 1); mbar_init(&B.pre_free[b], 8); mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4);
    }
    mbar_init(&B.dgp_ready, 1); mbar_init(&B.dgp_free, 1); mbar_init(&B.go_full, 1); mbar_init(&B.go_free, 1);
    for (int b = 0; b < 2; ++b) { mbar_init(&B.xi_full[b], 1); mbar_init(&B.xi_free[b], 1); }
    mbar_init(&B.ln_last, 1);
    fence_mbar_init();
  }
  if (tid < C) reinterpret_cast<float*>(sm + G_::O_GI)[tid] = a.gi[tid];
  if (!FU && warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = FU ? tmem_fu : B.tmem;
  if (tid == 0) P7(1);
  PROF_BEGIN

  // LN_in backward + residual of tile i's row blocks k = 2 H0 .. 2 (H0 + NH) - 1 of lane quarter q = warp & 3 in the MMA fragment layout
  // (B7M_FRAG; see the LN warps below). H0 = 0, NH = 2: the whole tile (warps 12-15). (2026-11-18 part 10) On the last tile of a CTA
  // the rows are split: warps 12-15 take H0 = 0, warps 0-3 (their roles done) H0 = 1, NH = 1 -- each lane quarter's 32 rows by two
  // warps (the LN warps ran one warp per scheduler: latency-bound). Every row's sums stay inside one warp; only the dgi / dbi column sums
  // gain a second set of warps (added after the first, a fixed order). bar / nbar: the named barrier before the dx store.
  auto ln_tile = [&](const int i, auto h0c, auto nhc, float2(&agi)[8], float2(&abi)[8], const int bar, const int nbar) {
    constexpr int H0 = decltype(h0c)::value, NH = decltype(nhc)::value;
    const int q = warp & 3, cq = lane & 3;
    const uint32_t gbase = su + G_::O_GI + 8 * cq;      // gi of columns 8 j + 2 cq, +1: gbase + 32 j
    // ldmatrix / stmatrix row address of row block k (rows 32 q + 8 k ..), chunks 4 h4 .. 4 h4 + 3: matrix m = lane >> 3, row lane & 7
    auto blk = [&](uint32_t base, int k, int h4) { return base + sw128(32 * q + 8 * k + (lane & 7), 4 * h4 + (lane >> 3)); };
    auto shsum = [&](float v) { v += __shfl_xor_sync(0xffffffffu, v, 1); return v + __shfl_xor_sync(0xffffffffu, v, 2); };
    const int row0 = (cta + G * i) * TOK, acc = i & 1;
    const int xb = i % G_::NXI;
    const uint32_t sx = su + G_::S_XI + xb * XT, sy = su + G_::S_DY + xb * XT;
    PW(0, mbar_wait(&B.xi_full[xb], (i / G_::NXI) & 1));
    if (tid == 384 && i == 0) P7(9);
    // the split: warps 0-3 start once the LN warps have the last tile's x / dy (their barrier waits are then of the current phase)
    if (H0 == 0 && NH == 1 && tid == 384) mbar_arrive(&B.ln_last);
    float mean[2 * NH], rstd[2 * NH];
#pragma unroll
    for (int kq = 0; kq < 2 * NH; ++kq) {
      const int k = 2 * H0 + kq;
      uint32_t xr[8];
      ldsm_x4n(blk(sx, k, 0), xr[0], xr[1], xr[2], xr[3]);
      ldsm_x4n(blk(sx, k, 1), xr[4], xr[5], xr[6], xr[7]);
      float2 s = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < 8; ++j) s = __fadd2_rn(s, up(xr[j]));
      mean[kq] = __fmul_rn(shsum(__fadd_rn(s.x, s.y)), 1.f / C);
      const float2 nm = make_float2(-mean[kq], -mean[kq]);
      s = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < 8; ++j) { const float2 d = __fadd2_rn(up(xr[j]), nm); s = __ffma2_rn(d, d, s); }
      rstd[kq] = rsqrt_ftz(__fadd_rn(__fmul_rn(shsum(__fadd_rn(s.x, s.y)), 1.f / C), a.eps));
    }
    if (H0 == 1 && tid == 0) P7(17);
    PW(1, mbar_wait(&B.acc_full[acc], (i >> 1) & 1));
    if (tid == 384 && i == n_local - 1) P7(10);
    if (H0 == 1 && tid == 0) P7(18);
    if (!FU && B7M_TRIG == 2 && tid == 384 && i == n_local - 1) pdl_trigger();
    tc_fence_after();
#pragma unroll
    for (int hq = 0; hq < NH; ++hq) {                    // TMEM lanes 32 q + 16 h .. + 15: row blocks k = 2 h, 2 h + 1
      const int h = H0 + hq;
      uint32_t v0[16], v1[16];
      const uint32_t ta = tmem + T_ACC + acc * 64 + ((uint32_t)(32 * q + 16 * h) << 16);
      tmem_ld16x256_x4(ta, v0);
      tmem_ld16x256_x4(ta + 32, v1);
      tmem_wait_ld();
      // acc_empty (4 arrivals): the warps holding row half 0 arrive after their last load of the buffer (the split's other half is
      // only ever the CTA's last tile, whose buffer is not refilled)
      if (H0 == 0 && hq == NH - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[acc]); }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k = 2 * h + kk, kq = 2 * hq + kk;
        uint32_t xr[8], dp[8];
        ldsm_x4n(blk(sx, k, 0), xr[0], xr[1], xr[2], xr[3]);
        ldsm_x4n(blk(sx, k, 1), xr[4], xr[5], xr[6], xr[7]);
        const float2 nm = make_float2(-mean[kq], -mean[kq]), rs2 = make_float2(rstd[kq], rstd[kq]);
        float2 c1a = make_float2(0.f, 0.f), c2a = make_float2(0.f, 0.f);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t* vv = j < 4 ? v0 : v1;
          const int rj = 4 * (j & 3) + 2 * kk;
          dp[j] = pack_bf16(__uint_as_float(vv[rj]), __uint_as_float(vv[rj + 1]));
          const float2 d = up(dp[j]);
          const float2 xh = __fmul2_rn(__fadd2_rn(up(xr[j]), nm), rs2);
          const float2 dg = __fmul2_rn(d, lds64g(gbase + 32 * j));
          c1a = __ffma2_rn(dg, xh, c1a);
          c2a = __fadd2_rn(c2a, dg);
          agi[j] = __ffma2_rn(d, xh, agi[j]);
          abi[j] = __fadd2_rn(abi[j], d);
        }
        const float c1 = __fmul_rn(shsum(__fadd_rn(c1a.x, c1a.y)), 1.f / C), c2 = __fmul_rn(shsum(__fadd_rn(c2a.x, c2a.y)), 1.f / C);
        const float2 c1b = make_float2(c1, c1), c2b = make_float2(c2, c2);
        uint32_t yr[8], o[8];
        ldsm_x4n(blk(sy, k, 0), yr[0], yr[1], yr[2], yr[3]);
        ldsm_x4n(blk(sy, k, 1), yr[4], yr[5], yr[6], yr[7]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 d = up(dp[j]);
          const float2 xh = __fmul2_rn(__fadd2_rn(up(xr[j]), nm), rs2);
          const float2 uu = __fadd2_rn(__fmul2_rn(xh, c1b), c2b);
          const float2 vd = __fadd2_rn(__fmul2_rn(d, lds64g(gbase + 32 * j)), make_float2(-uu.x, -uu.y));
          const uint32_t lw = pack_bf16(__fmul_rn(rstd[kq], vd.x), __fmul_rn(rstd[kq], vd.y));
          const float2 tt = __fadd2_rn(up(lw), up(yr[j]));
          o[j] = pack_bf16(tt.x, tt.y);
        }
        stsm_x4(blk(sx, k, 0), o[0], o[1], o[2], o[3]);   // dx over this row block's x (read above, not read again)
        stsm_x4(blk(sx, k, 1), o[4], o[5], o[6], o[7]);
      }
    }
    if (H0 == 1 && tid == 0) P7(19);
    if (tid == 384 && i == n_local - 1) P7(21);
    fence_async_smem();
    PW(2, named_bar_sync(bar, nbar));
    if (H0 == 1 && tid == 0) P7(20);
    if (tid == 384 && i == n_local - 1) P7(22);
    if (tid == 384) {
      tma_store_2d(&mdx, sm + G_::S_XI + xb * XT, 0, row0);
      bulk_commit();
      bulk_wait_read<0>();
      mbar_arrive(&B.xi_free[xb]);
      if (i == n_local - 1) P7(11);
    }
  };
  // the column sums of a warp's 8 row groups (lanes with equal t & 3) by xor shuffles, stored as row q of xs (+ the dgi / dbi halves)
  auto ln_cols = [&](float2(&agi)[8], float2(&abi)[8], float* xs) {
    const int q = warp & 3, g = lane >> 2, cq = lane & 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int msk = 4; msk <= 16; msk <<= 1) {
        agi[j].x += __shfl_xor_sync(0xffffffffu, agi[j].x, msk); agi[j].y += __shfl_xor_sync(0xffffffffu, agi[j].y, msk);
        abi[j].x += __shfl_xor_sync(0xffffffffu, abi[j].x, msk); abi[j].y += __shfl_xor_sync(0xffffffffu, abi[j].y, msk);
      }
    }
    return [=, &agi, &abi]() {
      if (g == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int col = 8 * j + 2 * cq;
          xs[q * 128 + col] = agi[j].x; xs[q * 128 + col + 1] = agi[j].y;
          xs[q * 128 + 64 + col] = abi[j].x; xs[q * 128 + 64 + col + 1] = abi[j].y;
        }
      }
    };
  };
  const bool ln_split = !FU && B7M_FRAG && B7M_LNH && NCH == 4 && n_local > 0;   // NCH 2: measured slower (part 10)

  if (warp == 0 && B7M_ML) {
    // (2026-11-17 part 5) the loads issued by several lanes at once: one thread issues about one TMA per ~600 clk whatever the box
    // size, and at one tile per CTA the loader's serial issue (weights 3, then 2 + 2 NCH + 1 per tile) sat on the critical path.
    // Lane 0 does every wait and expect_tx of a step, then the step's loads go out one per lane.
    if (lane == 0) mbar_expect_tx(&B.w_full, NCH * XT + 8192);
    __syncwarp();
    if (lane < NCH) tma_load_2d(sm + G_::S_W1 + lane * XT, &mw1, &B.w_full, 0, 128 * lane, EVICT_LAST);
    else if (lane == NCH) tma_load_2d(sm + G_::S_WG, &mwg, &B.w_full, 0, 0, EVICT_LAST);
    if constexpr (FU) dep_wait(dep, dep_n); else if (B7M_PDL) pdl_wait();   // every lane: dA is the previous kernel's output
    __syncwarp();
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, b = i & 1;
#pragma unroll 1
      for (int c0 = 0; c0 < NCH; c0 += 2) {   // chunk pairs: two distinct dA buffers
        if (lane == 0) {
          if (c0 == 0) {
            if (i >= 2) PW(0, mbar_wait(&B.x_free[b], ((i >> 1) - 1) & 1));
            mbar_expect_tx(&B.x_full[b], XT);
          }
          for (int c = c0; c < c0 + 2; ++c) {
            const int u = i * NCH + c, db = u & 1;
            if (u >= 2) PW(1, mbar_wait(&B.da_free[db], ((u >> 1) - 1) & 1));
            mbar_expect_tx(&B.da_full[db], DAT);
          }
        }
        __syncwarp();
        if (lane == 4 && c0 == 0) {
          tma_load_2d(sm + G_::S_X + b * XT, &mxn, &B.x_full[b], 0, row0, EVICT_NORMAL);
        } else if (lane < 4) {
          const int c = c0 + (lane >> 1), hf = lane & 1, u = i * NCH + c, db = u & 1;
          if constexpr (SMJ != 0) {
            int tq, ts;
            tok_split(row0, a.n2, tq, ts);
            if (B7M_DA1) { if (hf == 0) tma_load_5d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], 0, 0, tq / 64, ts, c, EVICT_FIRST); }
            else tma_load_4d(sm + G_::S_DA + db * DAT + hf * (DAT / 2), &mda, &B.da_full[db], tq + 64 * hf, 0, ts, c, EVICT_FIRST);
          } else {
            if (B7M_DA1) { if (hf == 0) tma_load_3d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], 0, 64 * c, row0 / 64, EVICT_FIRST); }
            else tma_load_2d(sm + G_::S_DA + db * DAT + hf * (DAT / 2), &mda, &B.da_full[db], row0 + 64 * hf, 64 * c, EVICT_FIRST);
          }
        }
      }
      if (lane == 0) {
        if (i >= 1) PW(2, mbar_wait(&B.go_free, (i - 1) & 1));
        mbar_expect_tx(&B.go_full, XT);
        tma_load_2d(sm + G_::S_GO, &mgo, &B.go_full, 0, row0, EVICT_FIRST);
      }
      __syncwarp();
    }
  } else if (warp == 0) {
    if (lane == 0) {
      // weights once; then per tile: xn, the NCH dA chunks, dGout
      mbar_expect_tx(&B.w_full, NCH * XT + 8192);
      for (int c = 0; c < NCH; ++c) tma_load_2d(sm + G_::S_W1 + c * XT, &mw1, &B.w_full, 0, 128 * c, EVICT_LAST);
      tma_load_2d(sm + G_::S_WG, &mwg, &B.w_full, 0, 0, EVICT_LAST);
      // (part 10, B7M_XNE) xn of the first tile before the wait: it is the forward's (k1w, complete before the cuBLAS contraction
      // launched), so the pre-activation MMAs of the first chunks run before dA arrives and dA's loads are issued first after the wait
      if (B7M_XNE && n_local > 0) {
        mbar_expect_tx(&B.x_full[0], XT);
        tma_load_2d(sm + G_::S_X, &mxn, &B.x_full[0], 0, cta * TOK, EVICT_NORMAL);
      }
      if constexpr (FU) dep_wait(dep, dep_n); else if (B7M_PDL) pdl_wait();   // dA is the previous kernel's output (everything else this kernel reads is older)
      P7(2);
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK, b = i & 1;
        if (i >= 2) PW(0, mbar_wait(&B.x_free[b], ((i >> 1) - 1) & 1));
        if (!B7M_XNE || i > 0) {
          mbar_expect_tx(&B.x_full[b], XT);
          tma_load_2d(sm + G_::S_X + b * XT, &mxn, &B.x_full[b], 0, row0, EVICT_NORMAL);
        }
        for (int c = 0; c < NCH; ++c) {
          const int u = i * NCH + c, db = u & 1;
          if (u >= 2) PW(1, mbar_wait(&B.da_free[db], ((u >> 1) - 1) & 1));
          mbar_expect_tx(&B.da_full[db], DAT);
          if constexpr (SMJ != 0) {
            int tq, ts;
            tok_split(row0, a.n2, tq, ts);
            if (B7M_DA1) {
              tma_load_5d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], 0, 0, tq / 64, ts, c, EVICT_FIRST);
            } else {
              tma_load_4d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], tq, 0, ts, c, EVICT_FIRST);
              tma_load_4d(sm + G_::S_DA + db * DAT + DAT / 2, &mda, &B.da_full[db], tq + 64, 0, ts, c, EVICT_FIRST);
            }
          } else {
            if (B7M_DA1) {
              tma_load_3d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], 0, 64 * c, row0 / 64, EVICT_FIRST);
            } else {
              tma_load_2d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], row0, 64 * c, EVICT_FIRST);
              tma_load_2d(sm + G_::S_DA + db * DAT + DAT / 2, &mda, &B.da_full[db], row0 + 64, 64 * c, EVICT_FIRST);
            }
          }
        }
        if (i >= 1) PW(2, mbar_wait(&B.go_free, (i - 1) & 1));
        mbar_expect_tx(&B.go_full, XT);
        tma_load_2d(sm + G_::S_GO, &mgo, &B.go_full, 0, row0, EVICT_FIRST);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      mbar_wait(&B.w_full, 0);
      P7(3);
      const uint32_t sw1 = su + G_::S_W1, sdgp = su + G_::S_DGP;
      auto issue_pre = [&](int u) {
        const int i = u / NCH, c = u % NCH, pb = u % NPB;
        PW(0, mbar_wait(&B.x_full[i & 1], (i >> 1) & 1));
        if (u >= NPB) PW(1, mbar_wait(&B.pre_free[pb], ((u / NPB) - 1) & 1));
        tc_fence_after();
        const uint32_t sx = su + G_::S_X + (i & 1) * XT, sw = sw1 + c * XT;
#pragma unroll
        for (int k = 0; k < 4; ++k) umma_ss(tmem + T_PRE + pb * 128, desc_k_sw128(sx + k * 32), desc_k_sw128(sw + k * 32), ID_PRE, k != 0);
        umma_commit(&B.pre_full[pb]);
      };
      if (n_chunks > 0) issue_pre(0);
      for (int i = 0; i < n_local; ++i) {
        const int acc = i & 1;
        const uint32_t sx = su + G_::S_X + acc * XT, dacc = tmem + T_ACC + acc * 64;
        if (i >= 2) PW(2, mbar_wait(&B.acc_empty[acc], ((i >> 1) - 1) & 1));
        for (int c = 0; c < NCH; ++c) {
          const int u = i * NCH + c;
          if (u + 1 < n_chunks) issue_pre(u + 1);            // runs while the dg / dp warps work on chunk u
          PW(3, mbar_wait(&B.dgp_ready, u & 1));
          tc_fence_after();
          const uint32_t sw = sw1 + c * XT;
          // dxn [128 tok][64] += dgp [tok][128 k] . W1_c [128 k][64]   (A K-major: two 64-column K-blocks; B MN-major)
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(dacc, desc_k_sw128(sdgp + (k >> 2) * (GT / 2) + (k & 3) * 32), desc_mn_sw128(sw + k * 2048, 8192), ID_DX, (c | k) != 0);
          // (2026-11-18 part 10, B7M_GOE) the last chunk: dxn += dGout Wg and the acc_full commit before its dW1 MMAs (the LN warps no
          // longer wait for them; dxn and dW1 sum the same products in the same order: bitwise)
          const bool goe = B7M_GOE && c == NCH - 1;
          if (goe) {
            PW(4, mbar_wait(&B.go_full, i & 1));
            tc_fence_after();
            const uint32_t sgo = su + G_::S_GO, swg = su + G_::S_WG;
#pragma unroll
            for (int k = 0; k < 4; ++k) umma_ss(dacc, desc_k_sw128(sgo + k * 32), desc_mn_sw128(swg + k * 2048, 8192), ID_DX, 1);
            umma_commit(&B.go_free);
            umma_commit(&B.acc_full[acc]);
          }
          // dW1_c [128 rows][64] += dgp^T [rows][128 tok] . xn [tok][64]
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_DW + c * 64, desc_mn_sw128(sdgp + k * 2048, GT / 2), desc_mn_sw128(sx + k * 2048, XT), ID_DW, (i | k) != 0);
          umma_commit(&B.dgp_free);
        }
        umma_commit(&B.x_free[acc]);
        if (!B7M_GOE) {
          PW(4, mbar_wait(&B.go_full, i & 1));
          tc_fence_after();
          const uint32_t sgo = su + G_::S_GO, swg = su + G_::S_WG;
#pragma unroll
          for (int k = 0; k < 4; ++k) umma_ss(dacc, desc_k_sw128(sgo + k * 32), desc_mn_sw128(swg + k * 2048, 8192), ID_DX, 1);
          umma_commit(&B.go_free);
          umma_commit(&B.acc_full[acc]);
        }
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      // x / dy of each tile for the epilogue (refilled once dx has left the x tile)
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK;
        const int xb = i % G_::NXI, xk = i / G_::NXI;
        if (i >= G_::NXI) PW(0, mbar_wait(&B.xi_free[xb], (xk - 1) & 1));
        mbar_expect_tx(&B.xi_full[xb], 2 * XT);
        tma_load_2d(sm + G_::S_XI + xb * XT, &mx, &B.xi_full[xb], 0, row0, EVICT_FIRST);
        tma_load_2d(sm + G_::S_DY + xb * XT, &mdy, &B.xi_full[xb], 0, row0, EVICT_FIRST);
      }
    }
  } else if (warp >= 4 && warp < 12) {
    // dg / dp of chunk u (as b7r.cu): warps 4-7 token group hh 0, warps 8-11 hh 1 of each warp quarter
    const int q = warp & 3, hh = (warp - 4) >> 2, tr = lane >> 2;
    const int tok0 = q * 32 + hh * 16;
    float mA = 0.f, mB = 0.f;
    PairIdx pix;
    pix.init(cta * TOK, G * TOK, a.L);
    for (int u = 0; u < n_chunks; ++u) {
      const int i = u / NCH, pb = u % NPB, db = u & 1;
      if (u % NCH == 0) {
        if (i > 0) pix.advance(a.L);
        pix.mask2(a.tmask, tok0 + tr, a.L, mA, mB);
      }
      PW(0, mbar_wait(&B.pre_full[pb], (u / NPB) & 1));
      if (tid == 128) { if (u == 0) P7(4); if (u == n_chunks - 1) P7(7); }
      PW(1, mbar_wait(&B.da_full[db], (u >> 1) & 1));
      if (tid == 128 && u == 0) P7(5);
      tc_fence_after();
      const uint32_t ta = tmem + T_PRE + pb * 128 + ((uint32_t)tok0 << 16);
      uint32_t da[16];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int mm = lane >> 3, ch = 16 * jj + 8 * (mm & 1) + (lane & 7), qc = ((tok0 & 63) >> 3) + (mm >> 1);
        ldsm_x4_trans(su + G_::S_DA + db * DAT + (tok0 >> 6) * (DAT / 2) + sw128(ch, qc), da[4 * jj], da[4 * jj + 1], da[4 * jj + 2],
                      da[4 * jj + 3]);
      }
      uint32_t gall[2][16], pall[2][16];
      tmem_ld16x256_x4(ta, gall[0]);
      tmem_ld16x256_x4(ta + 64, pall[0]);
      tmem_wait_ld();
      tmem_ld16x256_x4(ta + 32, gall[1]);
      tmem_ld16x256_x4(ta + 96, pall[1]);
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.da_free[db]);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        if (half == 1) {
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.pre_free[pb]);
        }
        uint32_t odg[8], odp[8];
#pragma unroll
        for (int jx = 0; jx < 4; ++jx) {
          const int J = 4 * half + jx;
#pragma unroll
          for (int tb = 0; tb < 2; ++tb) {
            const float2 dA = up(da[4 * (J >> 1) + (J & 1) + 2 * tb]);
            const float mkk = tb ? mB : mA;
            const uint32_t gw = pack_bf16(__uint_as_float(gall[half][4 * jx + 2 * tb]), __uint_as_float(gall[half][4 * jx + 2 * tb + 1]));
            const uint32_t pw = pack_bf16(__uint_as_float(pall[half][4 * jx + 2 * tb]), __uint_as_float(pall[half][4 * jx + 2 * tb + 1]));
            const float2 t = __fmul2_rn(up(gw), make_float2(-1.4426950408889634f, -1.4426950408889634f));
            const float2 dn = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)), make_float2(1.f, 1.f));
            const float2 gg = make_float2(rcp_ftz(dn.x), rcp_ftz(dn.y));
            const float2 m = __fmul2_rn(dA, make_float2(mkk, mkk));
            const float2 rg = __fmul2_rn(__fmul2_rn(__fmul2_rn(m, up(pw)), gg), __fadd2_rn(make_float2(1.f, 1.f), make_float2(-gg.x, -gg.y)));
            const float2 rp = __fmul2_rn(m, gg);
            odg[2 * jx + tb] = pack_bf16(rg.x, rg.y);
            odp[2 * jx + tb] = pack_bf16(rp.x, rp.y);
          }
        }
        if (half == 0 && u >= 1) PW(2, mbar_wait(&B.dgp_free, (u - 1) & 1));   // chunk u-1's MMAs have read the dgp tile
#pragma unroll
        for (int x2 = 0; x2 < 2; ++x2) {
          const int m = lane >> 3, rr = lane & 7;
          const int tok = tok0 + rr + 8 * (m & 1), J = 4 * half + 2 * x2 + (m >> 1);
          stsm_x4(su + G_::S_DGP + sw128(tok, J), odg[4 * x2], odg[4 * x2 + 1], odg[4 * x2 + 2], odg[4 * x2 + 3]);
          stsm_x4(su + G_::S_DGP + GT / 2 + sw128(tok, J), odp[4 * x2], odp[4 * x2 + 1], odp[4 * x2 + 2], odp[4 * x2 + 3]);
        }
      }
      fence_async_smem();
      PW(3, named_bar_sync(1, 256));
      if (tid == 128) { mbar_arrive(&B.dgp_ready); if (u == 0) P7(6); if (u == n_chunks - 1) P7(8); }
    }
    // flush dW1: chunk c rows 128c + row, this warp's 32-column half
    tc_fence_before();
    named_bar_sync(1, 256);
    if (n_local > 0) PW(4, mbar_wait(&B.x_free[(n_local - 1) & 1], ((n_local - 1) >> 1) & 1));   // the last dW1 MMAs are done
    tc_fence_after();
#ifndef B7M_OLD_FLUSH
    // (2026-11-17 part 3) each chunk's partial (128 rows x 64, one contiguous 32 KB block of dw1) staged in the dgp tile (free: the
    // last MMAs are done) with a float4 XOR swizzle, then copied out with consecutive lanes on consecutive addresses (was one row per
    // lane, eight 16-byte pieces each)
    if (!FU && B7M_FTS && n_local > 0) {
      // (2026-11-18 part 10) the partial through TMA stores: each chunk staged as two 32-column SW128 boxes (thread row, column half
      // part: box part, row r at 128 r, 16-byte piece j at j ^ (r & 7): conflict-free), chunks alternating between the dgp tile and the
      // xn tiles (both free: the last MMAs are done), one thread issues the stores -- the LSU work of the copy-out (which ran against
      // the LN warps) is gone. Same values, same places.
      const int part = hh, row = q * 32 + lane;
      for (int n = 0; n < NCH; ++n) {
        const int c = (n + cta) % NCH;
        uint8_t* slot = sm + ((n & 1) ? G_::S_X : G_::S_DGP);
        uint32_t v[32];
        tmem_ld32(tmem + T_DW + ((uint32_t)(q * 32) << 16) + c * 64 + part * 32, v);
        tmem_wait_ld();
        if (n >= 2) {   // the slot's previous stores have read it
          if (tid == 128) bulk_wait_read<0>();
          named_bar_sync(1, 256);
        }
        float4* bx = reinterpret_cast<float4*>(slot + part * 16384 + row * 128);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          bx[j ^ (row & 7)] =
              make_float4(__uint_as_float(v[4 * j]), __uint_as_float(v[4 * j + 1]), __uint_as_float(v[4 * j + 2]), __uint_as_float(v[4 * j + 3]));
        fence_async_smem();
        named_bar_sync(1, 256);
        if (tid == 128) {
          const int r0 = cta * (128 * NCH) + 128 * c;
          tma_store_2d(&mdw, slot, 0, r0);
          tma_store_2d(&mdw, slot + 16384, 32, r0);
          bulk_commit();
        }
      }
      if (tid == 128) BULK_END();
    } else if (n_local > 0) {
      const int part = hh, row = q * 32 + lane, t2 = tid - 128;
      float4* stg = reinterpret_cast<float4*>(sm + G_::S_DGP);
      for (int n = 0; n < NCH; ++n) {
        const int c = (n + cta) % NCH;
        uint32_t v[32];
        tmem_ld32(tmem + T_DW + ((uint32_t)(q * 32) << 16) + c * 64 + part * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 8; ++j)
          stg[row * 16 + ((part * 8 + j) ^ (row & 15))] =
              make_float4(__uint_as_float(v[4 * j]), __uint_as_float(v[4 * j + 1]), __uint_as_float(v[4 * j + 2]), __uint_as_float(v[4 * j + 3]));
        named_bar_sync(1, 256);
        float4* dst = reinterpret_cast<float4*>(a.dw1 + ((size_t)cta * (128 * NCH) + 128 * c) * C);   // this CTA's partial (no atomics)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int idx = t2 + 256 * k, r = idx >> 4, c4 = idx & 15;
          dst[idx] = stg[r * 16 + (c4 ^ (r & 15))];
        }
        named_bar_sync(1, 256);
      }
    }
    if (tid == 128) P7(12);
#else
    if (n_local > 0) {
      const int part = hh, row = q * 32 + lane;
      for (int n = 0; n < NCH; ++n) {
        const int c = (n + cta) % NCH;
        uint32_t v[32];
        tmem_ld32(tmem + T_DW + ((uint32_t)(q * 32) << 16) + c * 64 + part * 32, v);
        tmem_wait_ld();
        float* dst = a.dw1 + ((size_t)cta * (128 * NCH) + 128 * c + row) * C + part * 32;   // this CTA's partial (no atomics)
#pragma unroll
        for (int k = 0; k < 32; k += 4)
          *reinterpret_cast<float4*>(dst + k) = make_float4(__uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]),
                                                             __uint_as_float(v[k + 3]));
      }
    }
#endif
  } else if (warp >= 12 && B7M_FRAG) {
    // (2026-11-17, round 3 training part 4) LN_in backward + residual in the MMA fragment layout: lane t of warp quarter q holds rows
    // 32 q + (t >> 2) + 8 k (k = 0..3) and columns 8 j + 2 (t & 3) + {0, 1} (j = 0..7) -- x / dy by ldmatrix, dxn by tcgen05.ld
    // 16x256b, dx by stmatrix. A row's sums (mean, variance, c1, c2) are 16 columns per lane + two xor shuffles; the dgi / dbi column
    // sums stay in registers over every row and tile of the CTA (was a 32-lane shuffle tree per row and tile) and are reduced once at
    // the end. Same per-element arithmetic as the row-per-thread form below; the fp32 summation orders differ (deterministic).
    // (ln_tile above; part 10: the last tile's row half 1 by warps 0-3 when ln_split)
    float2 agi[8], abi[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { agi[j] = abi[j] = make_float2(0.f, 0.f); }
    for (int i = 0; i < n_local; ++i) {
      if (ln_split && i == n_local - 1) ln_tile(i, IC<0>{}, IC<1>{}, agi, abi, 4, 256);
      else ln_tile(i, IC<0>{}, IC<2>{}, agi, abi, 3, 128);
    }
    if (!B7M_BWL && tid == 384) BULK_END();
    // dgi / dbi: the 8 row groups of a warp by xor shuffles, then the 4 warps in a fixed order (+ warps 0-3's 4 in the same order)
    float* xs = reinterpret_cast<float*>(sm + G_::S_DY);
    auto put = ln_cols(agi, abi, xs);
    if (ln_split) named_bar_sync(4, 256); else named_bar_sync(3, 128);   // every warp's last dy reads are done
    put();
    if (ln_split) named_bar_sync(4, 256); else named_bar_sync(3, 128);
    {
      const int c = tid - 384;
      float v = ((xs[c] + xs[128 + c]) + xs[256 + c]) + xs[384 + c];
      if (ln_split) v += ((xs[512 + c] + xs[640 + c]) + xs[768 + c]) + xs[896 + c];
      a.lnpart[(size_t)cta * 128 + c] = v;
      if (tid == 384) P7(13);
      if (B7M_BWL && tid == 384) BULK_END();   // (part 10) the dx store completes behind the column sums
    }
  } else if (warp >= 12) {
    // LN_in backward + residual, one row per thread (r = 32 (warp % 4) + lane, all 64 columns): x / dy from smem, dxn from TMEM
    const int q = warp & 3, r = q * 32 + lane;
    const uint32_t gbase = su + G_::O_GI;
    float agi[4], abi[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) { agi[m] = abi[m] = 0.f; }
    int xb = 0;
    auto xa = [&](int ch) { return su + G_::S_XI + xb * XT + sw128(r, ch); };
    auto ya = [&](int ch) { return su + G_::S_DY + xb * XT + sw128(r, ch); };
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, acc = i & 1;
      xb = i % G_::NXI;
      PW(0, mbar_wait(&B.xi_full[xb], (i / G_::NXI) & 1));
      float2 sa = make_float2(0.f, 0.f), sb = make_float2(0.f, 0.f);
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) {
        const uint4 v = lds128(xa(ch));
        sa = __fadd2_rn(sa, __fadd2_rn(up(v.x), up(v.y))); sb = __fadd2_rn(sb, __fadd2_rn(up(v.z), up(v.w)));
      }
      float mean;
      {
        const float2 s2 = __fadd2_rn(sa, sb);
        mean = __fmul_rn(__fadd_rn(s2.x, s2.y), 1.f / C);
      }
      const float2 nmean = make_float2(-mean, -mean);
      sa = sb = make_float2(0.f, 0.f);
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) {
        const uint4 v = lds128(xa(ch));
        float2 d = __fadd2_rn(up(v.x), nmean); sa = __ffma2_rn(d, d, sa);
        d = __fadd2_rn(up(v.y), nmean); sb = __ffma2_rn(d, d, sb);
        d = __fadd2_rn(up(v.z), nmean); sa = __ffma2_rn(d, d, sa);
        d = __fadd2_rn(up(v.w), nmean); sb = __ffma2_rn(d, d, sb);
      }
      float rstd;
      {
        const float2 s2 = __fadd2_rn(sa, sb);
        rstd = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(s2.x, s2.y), 1.f / C), a.eps));
      }
      const float2 rs2 = make_float2(rstd, rstd);
      PW(1, mbar_wait(&B.acc_full[acc], (i >> 1) & 1));
      tc_fence_after();
      const uint32_t trow = tmem + T_ACC + acc * 64 + ((uint32_t)(q * 32) << 16);
      uint32_t dpk[32];
      float c1, c2;
      {
        float2 c1a[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)}, c2a[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
        for (int h32 = 0; h32 < 2; ++h32) {
          uint32_t dv[32];
          tmem_ld32(trow + 32 * h32, dv);
          tmem_wait_ld();
          if (h32 == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[acc]); }
#pragma unroll
          for (int c4 = 0; c4 < 4; ++c4) {
            const int ch = 4 * h32 + c4;
            const uint4 xv = lds128(xa(ch));
            const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int w = 4 * ch + e;
              dpk[w] = pack_bf16(__uint_as_float(dv[8 * c4 + 2 * e]), __uint_as_float(dv[8 * c4 + 2 * e + 1]));
              const float2 d = up(dpk[w]);
              const float2 xh = __fmul2_rn(__fadd2_rn(up(xs[e]), nmean), rs2);
              const float2 dg = __fmul2_rn(d, lds64g(gbase + 8 * w));
              c1a[e & 1] = __ffma2_rn(dg, xh, c1a[e & 1]);
              c2a[e & 1] = __fadd2_rn(c2a[e & 1], dg);
            }
          }
        }
        const float2 c1s = __fadd2_rn(c1a[0], c1a[1]), c2s = __fadd2_rn(c2a[0], c2a[1]);
        c1 = __fmul_rn(__fadd_rn(c1s.x, c1s.y), 1.f / C); c2 = __fmul_rn(__fadd_rn(c2s.x, c2s.y), 1.f / C);
      }
      const float2 c1b = make_float2(c1, c1), c2b = make_float2(c2, c2);
#pragma unroll
      for (int hq = 0; hq < 4; ++hq) {                     // 16 columns (two chunks) at a time
        float2 pg[8], pb[8];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
          const int ch = 2 * hq + h2;
          const uint4 xv = lds128(xa(ch)), yv = lds128(ya(ch));
          const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w}, ys[4] = {yv.x, yv.y, yv.z, yv.w};
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int w = 4 * ch + e;
            const float2 d = up(dpk[w]);
            const float2 xh = __fmul2_rn(__fadd2_rn(up(xs[e]), nmean), rs2);
            const float2 uu = __fadd2_rn(__fmul2_rn(xh, c1b), c2b);
            const float2 v = __fadd2_rn(__fmul2_rn(d, lds64g(gbase + 8 * w)), make_float2(-uu.x, -uu.y));
            const uint32_t lw = pack_bf16(__fmul_rn(rstd, v.x), __fmul_rn(rstd, v.y));
            const float2 tt = __fadd2_rn(up(lw), up(ys[e]));
            o[e] = pack_bf16(tt.x, tt.y);
            pg[4 * h2 + e] = __fmul2_rn(d, xh); pb[4 * h2 + e] = d;
          }
          sts128(xa(ch), o[0], o[1], o[2], o[3]);
        }
#pragma unroll
        for (int st = 0, msk = 16, nh = 4; st < 3; ++st, msk >>= 1, nh >>= 1) {
          const bool hi = (lane & msk) != 0;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            if (kk < nh) {
              const float2 sg_ = hi ? pg[kk] : pg[kk + nh], kg = hi ? pg[kk + nh] : pg[kk];
              const float2 sb_ = hi ? pb[kk] : pb[kk + nh], kb = hi ? pb[kk + nh] : pb[kk];
              pg[kk] = __fadd2_rn(kg, make_float2(__shfl_xor_sync(0xffffffffu, sg_.x, msk), __shfl_xor_sync(0xffffffffu, sg_.y, msk)));
              pb[kk] = __fadd2_rn(kb, make_float2(__shfl_xor_sync(0xffffffffu, sb_.x, msk), __shfl_xor_sync(0xffffffffu, sb_.y, msk)));
            }
          }
        }
        {
          const bool hi = (lane & 2) != 0;
          float g_ = hi ? pg[0].x : pg[0].y, kg = hi ? pg[0].y : pg[0].x;
          float b_ = hi ? pb[0].x : pb[0].y, kb = hi ? pb[0].y : pb[0].x;
          kg += __shfl_xor_sync(0xffffffffu, g_, 2); kb += __shfl_xor_sync(0xffffffffu, b_, 2);
          kg += __shfl_xor_sync(0xffffffffu, kg, 1); kb += __shfl_xor_sync(0xffffffffu, kb, 1);
          agi[hq] += kg; abi[hq] += kb;
        }
      }
      fence_async_smem();
      PW(2, named_bar_sync(3, 128));
      if (tid == 384) {
        tma_store_2d(&mdx, sm + G_::S_XI + xb * XT, 0, row0);
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.xi_free[xb]);
      }
    }
    if (tid == 384) BULK_END();
    // dgi / dbi: the 4 warps' column sums combined in a fixed order (the dy tile is no longer read), one row per CTA
    float* xs = reinterpret_cast<float*>(sm + G_::S_DY);
    if ((lane & 1) == 0) {
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const int col = 16 * h + 2 * (lane >> 2) + ((lane >> 1) & 1);
        xs[q * 128 + col] = agi[h]; xs[q * 128 + 64 + col] = abi[h];
      }
    }
    named_bar_sync(3, 128);
    {
      const int c = tid - 384;
      a.lnpart[(size_t)cta * 128 + c] = ((xs[c] + xs[128 + c]) + xs[256 + c]) + xs[384 + c];
    }
  }
  if (ln_split && warp < 4) {
    // (part 10) warps 0-3, their roles done: row half 1 of the CTA's last tile (see ln_tile), then their dgi / dbi column sums
    __syncwarp();
    mbar_wait(&B.ln_last, 0);
    if (tid == 0) P7(16);
    float2 agi[8], abi[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { agi[j] = abi[j] = make_float2(0.f, 0.f); }
    ln_tile(n_local - 1, IC<1>{}, IC<1>{}, agi, abi, 4, 256);
    float* xs = reinterpret_cast<float*>(sm + G_::S_DY) + 512;
    auto put = ln_cols(agi, abi, xs);
    named_bar_sync(4, 256);
    put();
    named_bar_sync(4, 256);
  }
  if (lane == 0 && (warp < 5 || warp == 12)) PROF_END(warp == 12 ? 5 : warp);
  tc_fence_before();
  __syncthreads();
  if (tid == 0) P7(14);
  if (!FU && warp == 2) tmem_dealloc(tmem, 512);
  if constexpr (!FU) { if (tid == 0) PT_G(g_pt7, 15); }
}

// The extra CTAs (Args::xg): grad_reduce's 128-output blocks of the output-side regions [D D (dWg) | D R (dWp)] and [2R (dg_o | db_o)]
// of the gradient buffer, block k by extra CTA k mod xg; per block grad_reduce_kernel's arithmetic with its GR_W = 16 warps (warp w
// sums partial rows w, w + 16, ... in order, the 16 warp sums added in warp order): the same bits as grad_reduce.
DEV void xreduce(const Args& a, uint8_t* sm) {
  pdl_wait();   // the B1 kernel's partials: it completed before the contraction backward this grid depends on
  float4(*part)[32] = reinterpret_cast<float4(*)[32]>(sm);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int R = a.xR, D = C, RD = R * D;
  const int npf = (D * D + D * R) / 128, nlo = 2 * R / 128;
  const long long st_f = (long long)D * (D + R);
  for (int k = (int)blockIdx.x - ((int)gridDim.x - a.xg); k < npf + nlo; k += a.xg) {
    const int e = k < npf ? 4 * RD + 128 * k + 4 * lane : 4 * RD + D * D + D * R + 2 * D + 128 * (k - npf) + 4 * lane;
    const float* p;
    long long st;
    int n;
    if (e < 4 * RD + D * D) {
      const int f = e - 4 * RD;
      p = a.xpf + (long long)(f / D) * (D + R) + f % D; st = st_f; n = a.xnf;
    } else if (e < 4 * RD + D * D + D * R) {
      const int f = e - 4 * RD - D * D;
      p = a.xpf + (long long)(f / R) * (D + R) + D + f % R; st = st_f; n = a.xnf;
    } else {
      p = a.xlo + (e - 4 * RD - D * D - D * R - 2 * D); st = 2 * R; n = a.xnl;
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int i = w; i < n; i += 16) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(p + i * st));
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    part[w][lane] = acc;
    __syncthreads();
    if (w == 0) {
      float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 16; ++q) { const float4 v = part[q][lane]; t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w; }
      *reinterpret_cast<float4*>(a.xout + e) = t;
    }
    __syncthreads();
  }
}

template <int NCH, int SMJ>
__global__ void __launch_bounds__(512, 1)
    b7m_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mda, const __grid_constant__ CUtensorMap mw1,
               const __grid_constant__ CUtensorMap mgo, const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdx,
               const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mdw,
               const Args a) {
  extern __shared__ __align__(1024) uint8_t sm[];
  if (a.xg && (int)blockIdx.x >= (int)gridDim.x - a.xg) { xreduce(a, sm); return; }
  b7m_body<NCH, SMJ, false>(mxn, mda, mw1, mgo, mwg, mdx, mx, mdy, mdw, a, sm, (int)blockIdx.x, (int)gridDim.x - a.xg, 0u, nullptr, 0u);
}

}  // namespace b7m

// x / xn / dy / dgout / dx [M, 64]; dpl [64 NCH, M] plane gradient; tmask [B L] bool token mask or None; w1 [128 NCH, 64] packed
// (chunk c = 64 gate rows then 64 projection rows); wg [64, 64]; gi [64] fp32; dw1 [min(tiles, SMs), 128 NCH, 64] fp32 per-CTA
// partials, written; lnpart [>= min(tiles, SMs), 128] fp32: per-CTA (dg_i | db_i) rows, written (grad_reduce sums both).
void b7m_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout,
                  c10::optional<torch::Tensor> tmask, torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx,
                  torch::Tensor dw1, torch::Tensor lnpart, double eps, int64_t L, c10::optional<torch::Tensor> xpf, int64_t xnf,
                  c10::optional<torch::Tensor> xlo, int64_t xnl, c10::optional<torch::Tensor> xout, int64_t xg) {
  using namespace b7m;
  const int64_t M = x.size(0);
  const int P = (int)tmap::chmajor_channels(dpl), NCH = P / 64;   // dpl [P, M] or sample-major [P / 64, B, 64, L, L]
  const int64_t n2 = dpl.dim() == 5 ? dpl.size(3) * dpl.size(4) : 0;
  TORCH_CHECK(M % TOK == 0 && x.size(1) == C && dpl.numel() == (int64_t)P * M && (NCH == 2 || NCH == 4) && w1.size(0) == 128 * NCH);
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int tiles = (int)(M / TOK), grid = tiles < nsm ? tiles : nsm;
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mgo = tok_map(dgout, M), mdx = tok_map(dx, M), mx = tok_map(x, M), mdy = tok_map(dy, M);
  auto mw1 = tok_map(w1, w1.size(0));
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mda = n2 ? (B7M_DA1 ? b7m::da5_map(dpl, n2) : tmap::make_chmajor(dpl, n2, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B))
                : (B7M_DA1 ? tmap::make(dpl.data_ptr(), bf, {64, (uint64_t)P, (uint64_t)M / 64}, {(uint64_t)M * 2, 128}, {64, 64, 2}, CU_TENSOR_MAP_SWIZZLE_128B)
                           : tmap::make(dpl.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B));
  // dw1 partials [grid 128 NCH rows][64] fp32 as 32-column SW128 boxes of 128 rows (B7M_FTS: the dW1 flush by TMA stores)
  auto mdw = tmap::make(dw1.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {(uint64_t)C, (uint64_t)(dw1.numel() / C)}, {(uint64_t)C * 4}, {32, 128},
                        CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  TORCH_CHECK(M % (L * L) == 0 && (!tmask.has_value() || (tmask->numel() == M / L && tmask->element_size() == 1 && tmask->is_contiguous())));
  a.tmask = tmask.has_value() ? reinterpret_cast<const uint8_t*>(tmask->data_ptr()) : nullptr;
  a.gi = gi.data_ptr<float>();
  TORCH_CHECK(lnpart.numel() >= (int64_t)grid * 2 * C && dw1.numel() >= (int64_t)grid * 128 * NCH * C);
  a.dw1 = dw1.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  a.tiles = tiles; a.n2 = (int)n2; a.L = (int)L; a.eps = (float)eps;
  // xg extra CTAs (Args::xg): the output-side partials xpf [xnf][C (C + R)], xlo [xnl][2R] summed into xout's regions (grad_reduce layout)
  if (xg > 0) {
    TORCH_CHECK(xpf.has_value() && xlo.has_value() && xout.has_value() && tiles + xg <= nsm, "b7m: extra CTAs need xpf / xlo / xout and free SMs");
    const int R = (int)(xlo->size(-1) / 2);
    for (const torch::Tensor* t : {&*xpf, &*xlo, &*xout}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32);
    TORCH_CHECK(R % 64 == 0 && xpf->numel() >= xnf * C * (C + R) && xlo->numel() >= xnl * 2 * R &&
                xout->numel() == 4 * R * C + C * C + C * R + 2 * C + 2 * R, "b7m: extra CTA operands");
    a.xpf = xpf->data_ptr<float>(); a.xlo = xlo->data_ptr<float>(); a.xout = xout->data_ptr<float>();
    a.xg = (int)xg; a.xnf = (int)xnf; a.xnl = (int)xnl; a.xR = R;
  }
  auto st = at::cuda::getCurrentCUDAStream();
  auto run = [&](auto kern, int smem, bool& attr) {
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    tmap::launch_pdl(kern, dim3(grid + a.xg), dim3(512), smem, st, B7M_PDL, mxn, mda, mw1, mgo, mwg, mdx, mx, mdy, mdw, a);
  };
  static bool attr[4] = {false, false, false, false};
  if (NCH == 2) {
    if (n2) run(b7m_kernel<2, 1>, Geo<2>::SMEM, attr[1]); else run(b7m_kernel<2, 0>, Geo<2>::SMEM, attr[0]);
  } else {
    if (n2) run(b7m_kernel<4, 1>, Geo<4>::SMEM, attr[3]); else run(b7m_kernel<4, 0>, Geo<4>::SMEM, attr[2]);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor b7m_prof() {
#if defined(PT_TL) && PT_TL   // the phase trace instead of the role probes: [160][16] (see g_pt7)
  auto h = torch::empty({160, 24}, torch::kInt64);
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(h.data_ptr(), b7m::g_pt7, sizeof(unsigned long long) * 160 * 24));
  return h;
#elif defined(PROF)
  auto h = torch::empty({160, 8, 8}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_prof, sizeof(unsigned long long) * 160 * 64);
  return h;
#else
  return torch::Tensor();
#endif
}

// ================================================================================================ cgb: contraction gradients
// (2026-11-17, round 3 training pass 2) The D64 / D128 backward's plane gradient in ONE launch: every product of the contraction
// backward (two one direction, four bidirectional; cuBLAS ran each as its own batched GEMM launch) as 128 x 128 output tiles of a
// persistent tcgen05 GEMM. Operands are stacks of row-major [L, L] matrices (planes, dt, the plane gradient: [P, B, L, L] or
// sample-major [P / 64, B, 64, L, L], both a 2-D [matrices L, L] view); a product group g reads A / B from dt or the planes, K-major
// (as stored) or MN-major (transposed, through the descriptor), so no transposed copy exists anywhere. Per job: NS-deep TMA ring of
// 64-wide K blocks (A 16 KB | B 16 KB), 4 tcgen05.mma (M = N = 128, K = 16) per block into one of two TMEM accumulators (the next
// job's MMAs run during this job's epilogue), 4 epilogue warps: fp32 -> bf16 (one rounding, as cuBLAS's bf16 output) -> SW128 staging
// -> TMA store. K summed in index order: deterministic.
namespace cgb {
using b7m::sw128;
constexpr int BM = 128, BK = 64, NS = 4;
constexpr int AT = BM * BK * 2, STG = 2 * AT, OT = BM * 128 * 2;
constexpr int S_O = NS * STG, O_BAR = S_O + 2 * OT, SMEM = O_BAR + 256;
static_assert(SMEM <= 232448, "smem");
struct Grp { int a_src, a_base, a_mn, b_src, b_base, b_mn, c_base; };
struct Args { Grp g[4]; int ng, K, L, tpm, jobs; };
struct Bars { uint64_t full[NS], empty[NS], acc_full[2], acc_empty[2]; uint32_t tmem; };
#ifndef CGB_PDL
#define CGB_PDL 1
#endif
#ifndef CGB_MN1
#define CGB_MN1 0   // (2026-11-18 part 11) an MN-major operand's two 64-wide M blocks in ONE TMA box (3-D map); bitwise, measured level at L128 / L256 (D64, hidden 128, bidirectional): off
#endif
#ifndef CGB_ML
#define CGB_ML 0   // 1: the loader's TMA loads issued by four lanes (2026-11-17 part 5: hidden 128 L256 0.1323 -> 0.1343 ms, the rest level; not taken)
#endif
#ifndef CGB_TRIG
#define CGB_TRIG 1   // griddepcontrol.launch_dependents at entry: B7 is scheduled at once and starts on the SMs cgb leaves free
#endif

// The kernel's body as a device function (2026-11-17 part 5; b7f calls it with FU = true: virtual CTA id, its own TMEM, the dt
// dependency as a counter wait). Warps >= 8 of a larger block take no part (they only join the block-wide barriers).
template <bool FU>
DEV void cgb_body(const CUtensorMap& mk0, const CUtensorMap& mk1, const CUtensorMap& mm0, const CUtensorMap& mm1, const CUtensorMap& mout,
                  const Args& a, uint8_t* sm, const int cta, const int G, const uint32_t tmem_fu, const uint32_t* dep, const uint32_t dep_n) {
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = a.tpm * a.tpm, NK = a.L / BK;
  if (!FU && CGB_TRIG) pdl_trigger();
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    fence_mbar_init();
  }
  if (!FU && warp == 2) { tmem_alloc(&B.tmem, 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = FU ? tmem_fu : B.tmem;
  auto decode = [&](int job, int& g, int& j, int& tm, int& tn) {
    g = job / (a.K * T);
    const int r = job - g * a.K * T;
    j = r / T;
    const int t = r - j * T;
    tm = t / a.tpm; tn = t - tm * a.tpm;
  };
  if (warp == 0 && CGB_ML) {
    // (2026-11-17 part 5) every K block's (up to) four loads issued by four lanes: one thread issues about one TMA per ~600 clk
    if constexpr (FU) b7m::dep_wait(dep, dep_n); else if (CGB_PDL) pdl_wait();   // dt is the previous kernel's (b1u / b1s / b1g) output
    __syncwarp();
    int it = 0;
    for (int job = cta; job < a.jobs; job += G) {
      int g, j, tm, tn;
      decode(job, g, j, tm, tn);
      const Grp& gr = a.g[g];
      const int ra = (gr.a_base + j) * a.L, rb = (gr.b_base + j) * a.L;
      const bool isb = (lane & 2) != 0, mn = isb ? gr.b_mn : gr.a_mn;
      const int src = isb ? gr.b_src : gr.a_src, rr = isb ? rb : ra, t0 = isb ? tn : tm, hf = lane & 1;
      const CUtensorMap* mk = src ? &mk1 : &mk0;
      const CUtensorMap* mm = src ? &mm1 : &mm0;
      for (int kb = 0; kb < NK; ++kb, ++it) {
        const int s = it % NS;
        if (lane == 0) {
          if (it >= NS) mbar_wait(&B.empty[s], ((it / NS) - 1) & 1);
          mbar_expect_tx(&B.full[s], STG);
        }
        __syncwarp();
        uint8_t* pd = sm + s * STG + (isb ? AT : 0);
        if (lane < 4) {
          if (mn && CGB_MN1) { if (hf == 0) tma_load_3d(pd, mm, &B.full[s], 0, rr + kb * BK, t0 * (BM / 64), EVICT_NORMAL); }
          else if (mn)   // X[k][m]: K rows kb 64.., two 64-wide M blocks (one per lane)
            tma_load_2d(pd + hf * (AT / 2), mm, &B.full[s], t0 * BM + 64 * hf, rr + kb * BK, EVICT_NORMAL);
          else if (hf == 0)
            tma_load_2d(pd, mk, &B.full[s], kb * BK, rr + t0 * BM, EVICT_NORMAL);
        }
      }
    }
  } else if (warp == 0) {
    if (lane == 0) {
      if constexpr (FU) b7m::dep_wait(dep, dep_n); else if (CGB_PDL) pdl_wait();   // dt is the previous kernel's (b1u / b1s / b1g) output
      int it = 0;
      for (int job = cta; job < a.jobs; job += G) {
        int g, j, tm, tn;
        decode(job, g, j, tm, tn);
        const Grp& gr = a.g[g];
        const int ra = (gr.a_base + j) * a.L, rb = (gr.b_base + j) * a.L;
        const CUtensorMap* ka = gr.a_src ? &mk1 : &mk0;
        const CUtensorMap* na = gr.a_src ? &mm1 : &mm0;
        const CUtensorMap* kb_ = gr.b_src ? &mk1 : &mk0;
        const CUtensorMap* nb = gr.b_src ? &mm1 : &mm0;
        for (int kb = 0; kb < NK; ++kb, ++it) {
          const int s = it % NS;
          if (it >= NS) mbar_wait(&B.empty[s], ((it / NS) - 1) & 1);
          mbar_expect_tx(&B.full[s], STG);
          uint8_t* pa = sm + s * STG;
          uint8_t* pb = pa + AT;
          if (gr.a_mn) {   // A[m][k] = X[k][m]: K rows kb 64.., two 64-wide M blocks
            if (CGB_MN1) {
              tma_load_3d(pa, na, &B.full[s], 0, ra + kb * BK, tm * (BM / 64), EVICT_NORMAL);
            } else {
              tma_load_2d(pa, na, &B.full[s], tm * BM, ra + kb * BK, EVICT_NORMAL);
              tma_load_2d(pa + AT / 2, na, &B.full[s], tm * BM + 64, ra + kb * BK, EVICT_NORMAL);
            }
          } else {
            tma_load_2d(pa, ka, &B.full[s], kb * BK, ra + tm * BM, EVICT_NORMAL);
          }
          if (gr.b_mn) {
            if (CGB_MN1) {
              tma_load_3d(pb, nb, &B.full[s], 0, rb + kb * BK, tn * (BM / 64), EVICT_NORMAL);
            } else {
              tma_load_2d(pb, nb, &B.full[s], tn * BM, rb + kb * BK, EVICT_NORMAL);
              tma_load_2d(pb + AT / 2, nb, &B.full[s], tn * BM + 64, rb + kb * BK, EVICT_NORMAL);
            }
          } else {
            tma_load_2d(pb, kb_, &B.full[s], kb * BK, rb + tn * BM, EVICT_NORMAL);
          }
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int it = 0, n = 0;
      for (int job = cta; job < a.jobs; job += G, ++n) {
        int g, j, tm, tn;
        decode(job, g, j, tm, tn);
        const Grp& gr = a.g[g];
        const uint32_t idesc = idesc_bf16_mj(128, 128, gr.a_mn, gr.b_mn);
        const int acc = n & 1;
        if (n >= 2) mbar_wait(&B.acc_empty[acc], ((n >> 1) - 1) & 1);
        tc_fence_after();
        for (int kb = 0; kb < NK; ++kb, ++it) {
          const int s = it % NS;
          mbar_wait(&B.full[s], (it / NS) & 1);
          tc_fence_after();
          const uint32_t sa = su + s * STG, sb = sa + AT;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const uint64_t da = gr.a_mn ? desc_mn_sw128(sa + kk * 2048, AT / 2) : desc_k_sw128(sa + kk * 32);
            const uint64_t db = gr.b_mn ? desc_mn_sw128(sb + kk * 2048, AT / 2) : desc_k_sw128(sb + kk * 32);
            umma_ss(tmem + acc * 128, da, db, idesc, (kb | kk) != 0);
          }
          umma_commit(&B.empty[s]);
        }
        umma_commit(&B.acc_full[acc]);
      }
    }
  } else if (warp >= 4 && warp < 8) {
    const int q = warp & 3, row = q * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
    int n = 0;
    for (int job = cta; job < a.jobs; job += G, ++n) {
      int g, j, tm, tn;
      decode(job, g, j, tm, tn);
      const int acc = n & 1;
      const uint32_t ob = su + S_O + acc * OT;
      if (tid == 128) bulk_wait_read<1>();   // job n - 2's store has read this staging buffer
      named_bar_sync(1, 128);
      mbar_wait(&B.acc_full[acc], (n >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint32_t v[32];
        tmem_ld32(trow + acc * 128 + c * 32, v);
        tmem_wait_ld();
        const uint32_t base = ob + (c >> 1) * (OT / 2);
#pragma unroll
        for (int u = 0; u < 4; ++u)
          sts128(base + sw128(row, (c & 1) * 4 + u),
                 pack_bf16(__uint_as_float(v[8 * u]), __uint_as_float(v[8 * u + 1])),
                 pack_bf16(__uint_as_float(v[8 * u + 2]), __uint_as_float(v[8 * u + 3])),
                 pack_bf16(__uint_as_float(v[8 * u + 4]), __uint_as_float(v[8 * u + 5])),
                 pack_bf16(__uint_as_float(v[8 * u + 6]), __uint_as_float(v[8 * u + 7])));
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acc_empty[acc]);
      fence_async_smem();
      named_bar_sync(1, 128);
      if (tid == 128) {
        const int rc = (a.g[g].c_base + j) * a.L + tm * BM;
        tma_store_2d(&mout, sm + S_O + acc * OT, tn * BM, rc);
        tma_store_2d(&mout, sm + S_O + acc * OT + OT / 2, tn * BM + 64, rc);
        bulk_commit();
      }
    }
    if (tid == 128) BULK_END();
  }
  tc_fence_before();
  __syncthreads();
  if (!FU && warp == 2) tmem_dealloc(tmem, 256);
}

__global__ void __launch_bounds__(256, 1)
    cgb_kernel(const __grid_constant__ CUtensorMap mk0, const __grid_constant__ CUtensorMap mk1, const __grid_constant__ CUtensorMap mm0,
               const __grid_constant__ CUtensorMap mm1, const __grid_constant__ CUtensorMap mout, const Args a) {
  extern __shared__ __align__(1024) uint8_t sm[];
  cgb_body<false>(mk0, mk1, mm0, mm1, mout, a, sm, (int)blockIdx.x, (int)gridDim.x, 0u, nullptr, 0u);
}
}  // namespace cgb

// planes / dpl: the training planes (any of the D64 / D128 layouts, a stack of [L, L] matrices), dt: the contraction-output gradient
// (same layout family); direction 0 bidirectional, 1 outgoing, 2 incoming. dpl = the plane gradient (``_contract_bwd`` / ``_contract_bwd_sm``).
// cgb_plan: the maps (mk0, mk1, mm0, mm1, mout), the arguments and the grid of one call (shared by cgb_backward and b7f_backward).
static void cgb_plan(const torch::Tensor& planes, const torch::Tensor& dt, const torch::Tensor& dpl, int64_t L, int64_t direction, int nsm,
                     CUtensorMap (&m)[5], cgb::Args& a, int& grid) {
  using namespace cgb;
  TORCH_CHECK(planes.is_contiguous() && dt.is_contiguous() && dpl.is_contiguous() && L % 128 == 0 && L >= 128);
  const int64_t nd = dt.numel() / (L * L), np = planes.numel() / (L * L);
  const int64_t K = direction == 0 ? nd / 2 : nd;
  TORCH_CHECK(dt.numel() == nd * L * L && planes.numel() == np * L * L && dpl.numel() == planes.numel() &&
              np == (direction == 0 ? 4 : 2) * K, "cgb_backward: planes [2 or 4 K matrices], dt [1 or 2 K]");
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto map = [&](const torch::Tensor& t, int64_t n, uint32_t bc, uint32_t br) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)L, (uint64_t)(n * L)}, {(uint64_t)L * 2}, {bc, br}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  // the MN-major maps (CGB_MN1): (m mod 64, row, m / 64), box (64, 64, 2) = both 64-wide M blocks of a K block
  auto map3 = [&](const torch::Tensor& t, int64_t n) {
    return tmap::make(t.data_ptr(), bf, {64, (uint64_t)(n * L), (uint64_t)(L / 64)}, {(uint64_t)L * 2, 128}, {64, 64, 2}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  m[0] = map(dt, nd, 64, 128); m[1] = map(planes, np, 64, 128);
  m[2] = CGB_MN1 ? map3(dt, nd) : map(dt, nd, 64, 64); m[3] = CGB_MN1 ? map3(planes, np) : map(planes, np, 64, 64);
  m[4] = map(dpl, np, 64, 128);
  const int k = (int)K;
  // {a_src, a_base, a_mn, b_src, b_base, b_mn, c_base}; src 0 = dt, 1 = planes (see _contract_bwd)
  if (direction == 1) {          // da = dt b, db = dt^T a
    a.g[0] = {0, 0, 0, 1, k, 1, 0};
    a.g[1] = {0, 0, 1, 1, 0, 1, k};
    a.ng = 2;
  } else if (direction == 2) {   // da = b dt^T, db = a dt
    a.g[0] = {1, k, 0, 0, 0, 0, 0};
    a.g[1] = {1, 0, 0, 0, 0, 1, k};
    a.ng = 2;
  } else {                       // dpl = [dt0 b_o, b_i dt1^T, dt0^T a_o, a_i dt1]
    a.g[0] = {0, 0, 0, 1, 2 * k, 1, 0};
    a.g[1] = {1, 3 * k, 0, 0, k, 0, k};
    a.g[2] = {0, 0, 1, 1, 0, 1, 2 * k};
    a.g[3] = {1, k, 0, 0, k, 1, 3 * k};
    a.ng = 4;
  }
  a.K = k; a.L = (int)L; a.tpm = (int)(L / BM);
  const int64_t jobs = (int64_t)a.ng * K * a.tpm * a.tpm;
  TORCH_CHECK(jobs < (1LL << 31));
  a.jobs = (int)jobs;
  grid = jobs < nsm ? (int)jobs : nsm;
}

void cgb_backward(torch::Tensor planes, torch::Tensor dt, torch::Tensor dpl, int64_t L, int64_t direction) {
  using namespace cgb;
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  CUtensorMap m[5];
  Args a;
  int grid = 0;
  cgb_plan(planes, dt, dpl, L, direction, nsm, m, a, grid);
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(cgb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  tmap::launch_pdl(cgb_kernel, dim3(grid), dim3(256), SMEM, at::cuda::getCurrentCUDAStream(), CGB_PDL, m[0], m[1], m[2], m[3], m[4], a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
