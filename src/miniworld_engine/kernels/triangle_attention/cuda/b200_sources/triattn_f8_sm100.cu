// Triangle-attention inference forward with e4m3 P and V on the B200 (sm_100a): the H100 FP8 PV route (docs/fp8.md "TriAttn forward: FP8 PV")
// on tcgen05 `kind::f8f6f4`.  D = 32 (d_pair 128 x 4 heads, and every other head-dim-32 shape: they share the core) or, built with -DTF_D=64,
// 64 channels; bf16 q / k, fp16 bias, fp32 accumulation, bf16 out.  OPT-IN: the default route (triattn_sm100.cu) is untouched.
//
//   out[b,n,h,q,:] = softmax_k( scale * q . k + bias ) @ v                    P and V on e4m3, the rest as the bf16 core
//
// H100 recipe, redone for tcgen05 (docs/b200/fp8.md "TriAttn FP8 on the B200"):
//   * P is formed as e4m3 BYTES straight from the logit (no exponential): p = 2^(x - m + 8) as the e4m3 byte 8 (x - m) + 120 (an e4m3
//     byte b is 2^((b >> 3) - 7) (1 + (b & 7) / 8): the byte of 8 x is 2^x to ~3 bits).  Per key pair: x = f16(S) * scl + bias (HFMA2, the bias fp16
//     pre-scaled by log2 e), the tile maximum (HMNMX2), then HFMA2.RELU (t = relu(x - m + 15)) and HFMA2 (8 t + 1024: the byte is the low 8 bits of
//     the fp16 mantissa), one PRMT a word of 4 keys; the words go to TMEM as the A operand of the PV MMA (4 e4m3 a 32-bit column, 8 columns a K = 32).
//     127 is the e4m3 NaN byte: after the vote t <= 15.5, so no upper clamp is needed.
//   * the row sum comes from the PV GEMM: [O | l] += P [V | 1] as one M128 N(D + 16) K32 e4m3 MMA (B = V^T rows 0..D-1 + 16 ones rows), so the sum is
//     the sum of the quantised P (self-normalising, as the H100's).
//   * V arrives as e4m3 V^T planes [B N H D][S] (f8_vtq: a per-head static power-of-two scale vinv, calibrated on the first eager call, saturation
//     flag in `guard`), K-major in the 128-byte swizzle: 4 MMAs of K = 32 a 128-key tile (the bf16 core issues 8 of K = 16).
//   * the row offset m: tile 0 and any row whose tile maximum leaves the window top (x - m > 0.5) put it at max + TF_RAISE (window [m - 15, m + 0.5]); the
//     shift enters the byte conversion's constant and O / l are rescaled in TMEM (warp-uniform, after the previous PV of the row retired).  A row whose
//     final l is below 2^(8 - TF_RAISE) / 2 (every key under the window) or not finite is recomputed exactly at the drain (exact_row).
// TMEM: S [3 x 128] | O, l [2 rows x (D + 16)] = 480 of 512 columns at D = 32 (S x 2 + 2 x 80 = 416 at D = 64); two pair rows a task (R = 2).
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include "sm100.cuh"
using namespace sm100;

#ifndef TF_D
#define TF_D 32         // head dim: 32 (the d_pair 128 route and every head-dim-32 shape) or 64 (the 64-channel-head core)
#endif
#ifndef TF_NKV
#define TF_NKV (TF_D == 32 ? 6 : 3)
#endif
#ifndef TF_RAISE_E
#define TF_RAISE_E TF_RAISE   // the raise of a row that leaves the window after tile 0 (a larger value = fewer rescale events, a lower byte window for those rows)
#endif
#ifndef TF_RAISE
#define TF_RAISE 3.0f    // a raised offset puts the tile maximum this far below the window top (its probability is 2^(8 - TF_RAISE))
#endif
#ifndef TF_ABL
#define TF_ABL 0         // (timing only, WRONG results) ablation bits: 1 no logits, 4 no tile maximum, 8 no byte conversion, 16 no S loads, 32 no P stores, 64 one QK^T MMA of two (the e4m3 q / k ceiling: K = 32 in one MMA)
#endif

namespace {
constexpr int D = TF_D, BM = 128, BN = 128, R = 2, NGR = 2, NS = (TF_D == 32 ? 3 : 2), NKV = TF_NKV, NB = 2, CK = 16, NCK = BN / CK;
constexpr int QT = BM * D, KVT = BN * D, BT = BM * BN;          // elements: q tile [128][D] bf16 (64B swizzle at D = 32, 128B at 64), k tile, bias tile 32 KiB
constexpr int KB = KVT * 2, VB = (D + 16) * 128, STG = KB + VB;   // bytes: K tile, V^T D rows + 16 ones rows of 128 B
constexpr int OC = D + 16;                                       // [O (D), l (16)] columns a row
constexpr int COL_S = 0, COL_O = NS * BN;
static_assert(COL_O + R * OC <= 512, "TMEM");
constexpr int THREADS = 480;                                      // 0 TMA, 1 QK^T, 2-5 / 6-9 softmax groups (rows 0 / 1), 10-13 epilogue, 14 PV
constexpr int SMEM = 1024 + 2 * R * QT * 2 + NKV * STG + NB * BT * 2 + R * BM * 4 + 512;
static_assert(SMEM <= 232448, "smem");
constexpr uint32_t ID_S = idesc_bf16(128, BN, 0, 0);
// kind::f8f6f4, A = e4m3 (TMEM), B = e4m3 (smem, K-major), D = f32, M = 128, N = 48
constexpr uint32_t ID_O = (1u << 4) | ((uint32_t)(OC >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
constexpr float L2E = 1.4426950408889634f;
constexpr float LMIN = 0.5f * (float)(1 << (8 - (int)(TF_RAISE > TF_RAISE_E ? TF_RAISE : TF_RAISE_E)));   // a row with a finite key has l >= 2^(8 - TF_RAISE) (the maximum key, within the byte rounding): below half of it the row is recomputed

__device__ __forceinline__ void mma_ts_f8(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n}\n"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void tmem_st1(uint32_t taddr, uint32_t v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n" :: "r"(taddr), "r"(v) : "memory");
}
__device__ __forceinline__ float tmem_ld1(uint32_t taddr) {
  uint32_t r; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(r) : "r"(taddr) : "memory"); return __uint_as_float(r);
}
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t f16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) { uint32_t r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t hfma2r(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.relu.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) { uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }

// a row whose final l is out of range is recomputed exactly (fp32, two passes over the keys from q / k / v / the pre-scaled bias in global memory)
template <int C>
__device__ __noinline__ void exact_row(const __nv_bfloat16* __restrict__ qp, const __nv_bfloat16* __restrict__ kp, const __nv_bfloat16* __restrict__ vp,
                                       const __half* __restrict__ bp, int LD, int S, float scl, uint32_t* ob) {
  float qv[C];
#pragma unroll
  for (int c = 0; c < C; ++c) qv[c] = __bfloat162float(qp[c]);
  auto logit = [&](int kk) {
    const __nv_bfloat16* kr = kp + (size_t)kk * LD;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) s = fmaf(qv[c], __bfloat162float(kr[c]), s);
    return s * scl + __half2float(bp[kk]);
  };
  float m = -INFINITY;
  for (int kk = 0; kk < S; ++kk) m = fmaxf(m, logit(kk));
  float l = 0.f, o[C];
#pragma unroll
  for (int c = 0; c < C; ++c) o[c] = 0.f;
  if (m > -INFINITY) {
    for (int kk = 0; kk < S; ++kk) {
      const float p = exp2f(logit(kk) - m);
      const __nv_bfloat16* vr = vp + (size_t)kk * LD;
      l += p;
#pragma unroll
      for (int c = 0; c < C; ++c) o[c] = fmaf(p, __bfloat162float(vr[c]), o[c]);
    }
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
  for (int k2 = 0; k2 < C / 2; ++k2) ob[k2] = pack2(o[2 * k2] * inv, o[2 * k2 + 1] * inv);
}

__global__ void __launch_bounds__(THREADS, 1) triattn_f8_fwd_sm100(
    int N, int H, int S, int ntasks, float scl,                 // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vtmap,                  // V^T e4m3 [B*N*H*32][S], box (128, 32), 128B swizzle
    const __grid_constant__ CUtensorMap bmap,
    __nv_bfloat16* __restrict__ OUT, const float* __restrict__ VSC,   // per-head V scale (the inverse of f8_vtq's vinv)
    int* __restrict__ STAT,                                      // [0] rows recomputed exactly, [1] rare tiles (warp events)
    const __nv_bfloat16* __restrict__ QG, const __nv_bfloat16* __restrict__ KG, int LD,
    const __nv_bfloat16* __restrict__ VG, const __half* __restrict__ BG) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);                       // [2 tasks][R][QT]
  unsigned char* sKV = smb + 2 * R * QT * 2;                                        // [NKV][K 8 KiB | V^T + ones 6 KiB]
  __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(sKV + NKV * STG);            // [NB][BT]
  float* sRM = reinterpret_cast<float*>(sB + NB * BT);                              // [R][128] row offsets
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRM + R * BM);
  uint64_t* qf = bars;               // [2]
  uint64_t* qe = qf + 2;             // [2]
  uint64_t* kvf = qe + 2;            // [NKV]
  uint64_t* kve = kvf + NKV;         // [NKV]
  uint64_t* bf = kve + NKV;          // [NB]
  uint64_t* be = bf + NB;            // [NB] count 8
  uint64_t* sf = be + NB;            // [NS]
  uint64_t* pf = sf + NS;            // [NS] count 4
  uint64_t* of = pf + NS;            // [1]
  uint64_t* oe = of + 1;             // [1] count 4
  uint32_t* tslot = reinterpret_cast<uint32_t*>(oe + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < 2; ++k) { bar_init(qf + k, 1); bar_init(qe + k, 1); }
    for (int k = 0; k < NS; ++k) { bar_init(sf + k, 1); bar_init(pf + k, 4); }
    bar_init(of, 1); bar_init(oe, 4);
    for (int k = 0; k < NKV; ++k) { bar_init(kvf + k, 1); bar_init(kve + k, 1); }
    for (int k = 0; k < NB; ++k) { bar_init(bf + k, 1); bar_init(be + k, 4 * NGR); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  {   // the ones rows (e4m3 1.0 = 0x38) behind each stage's 32 V^T rows: B = [V^T ; 1] (rows 32 .. 47; a constant is swizzle-invariant)
    for (int i = tid; i < NKV * (16 * 128 / 4); i += THREADS) {
      const int st = i / (16 * 128 / 4), w = i % (16 * 128 / 4);
      reinterpret_cast<uint32_t*>(sKV + st * STG + KB + D * 128)[w] = 0x38383838u;
    }
    fence_proxy_async();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  pdl_wait();
  pdl_trigger();
  auto decode = [&](int t, int& bh, int& qt, int& g) { qt = t % QTILES; const int r = t / QTILES; g = r % NG; bh = r / NG; };
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H; n = min(n, N - 1); return (b * N + n) * S; };
  auto hcol = [&](int bh) -> int { return (bh % H) * D; };
  auto vrow = [&](int bh, int n) -> int { const int b = bh / H, h = bh % H; return ((b * N + min(n, N - 1)) * H + h) * D; };

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        const int nr = R, qb = lt & 1;
        if (lt >= 2) wait(qe + qb, ((lt >> 1) - 1) & 1);
        expect_tx(qf + qb, nr * QT * 2);
        for (int r = 0; r < nr; ++r) load_2d(&qmap, sQ + (qb * R + r) * QT, qf + qb, hcol(bh), row0(bh, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1);
          expect_tx(bf + bs, BT * 2);
          load_3d(&bmap, sB + bs * BT, bf + bs, 0, bh * S + qt * BM, j * 2);
          for (int r = 0; r < nr; ++r, ++x) {
            const int st = x % NKV;
            if (x >= NKV) wait(kve + st, ((x / NKV) - 1) & 1);
            expect_tx(kvf + st, KB + D * 128);
            load_2d(&kmap, sKV + st * STG, kvf + st, hcol(bh), row0(bh, g * R + r) + j * BN);
            load_2d(&vtmap, sKV + st * STG + KB, kvf + st, j * BN, vrow(bh, g * R + r));
          }
        }
      }
    }
  } else if (warp == 1 || warp == 14) {
    const int SPT = NK * R;
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    struct Ctr {
      int k = 0, w = 0, j = 0, r = 0, st = 0, b = 0, stp = 0, bp = 0;
      __device__ __forceinline__ void next(int spt) {
        if (++r == R) { r = 0; ++j; }
        if (++w == spt) { w = 0; j = 0; ++k; }
        if (++st == NKV) { st = 0; stp ^= 1; }
        if (++b == NS) { b = 0; bp ^= 1; }
      }
    } cq, cp;
    auto qk = [&]() {
      const int k = cq.k, w = cq.w, st = cq.st, b = cq.b, qb = k & 1, stp = cq.stp, r = cq.r;
      cq.next(SPT);
      if (w == 0) wait(qf + qb, (k >> 1) & 1);
      wait(kvf + st, stp);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* q = sQ + (qb * R + r) * QT;
        const __nv_bfloat16* kk = reinterpret_cast<const __nv_bfloat16*>(sKV + st * STG);
#pragma unroll
        for (int ks = 0; ks < ((TF_ABL & 64) ? 1 : D / 16); ++ks) mma_ss(tmem + COL_S + b * BN, D == 32 ? desc_k64(q + ks * 16) : desc_k128(q + ks * 16), D == 32 ? desc_k64(kk + ks * 16) : desc_k128(kk + ks * 16), ID_S, ks ? 1u : 0u);
        mma_commit(sf + b);
        if (w == SPT - 1) mma_commit(qe + qb);
      }
      __syncwarp();
    };
    auto pv = [&]() {                            // [O | l]_r += P . [V | 1]: four e4m3 MMAs of K = 32 (the 128 keys), N = 48
      const int k = cp.k, w = cp.w, j = cp.j, r = cp.r, st = cp.st, b = cp.b, bp = cp.bp;
      cp.next(SPT);
      wait(pf + b, bp);
      if (w == 0 && k >= 1) wait(oe, (k - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const unsigned char* vt = sKV + st * STG + KB;
        const uint32_t o = tmem + COL_O + r * OC;
#pragma unroll
        for (int ks = 0; ks < BN / 32; ++ks)
          mma_ts_f8(o, tmem + COL_S + b * BN + ks * 8, sdesc(sa(vt + ks * 32), 16, 1024, 2), ID_O, (j | ks) ? 1u : 0u);
        mma_commit(kve + st);
        if (w == SPT - 1) mma_commit(of);
      }
      __syncwarp();
    };
    if (warp == 1) {
      Ctr cw;
      for (int y = 0; y < total; ++y) {
        if (y >= NS) { wait(kve + cw.st, cw.stp); cw.next(SPT); }
        qk();
      }
    } else {
      for (int p = 0; p < total; ++p) pv();
    }
  } else if (warp < 2 + 4 * NGR) {
    const int q = warp & 3, row = q * 32 + lane;
    const int gi = (warp - 2) >> 2, r = gi;
    int lt = 0, x = 0, jb = 0;
    const uint32_t sclh = f16x2(scl, scl);
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      float* const mslot = sRM + r * BM + row;
      *mslot = 0.f;
      for (int j = 0; j < NK; ++j, ++jb) {
        const int bs = jb % NB;
        wait(bf + bs, (jb / NB) & 1);
        const __nv_bfloat16* brow = sB + bs * BT + row * 64;
        const int xs = x + r, b = xs % NS;
        wait(sf + b, (xs / NS) & 1);
        tc_fence_after();
        const uint32_t sb = tmem_at(tmem + COL_S + b * BN, q * 32, 0);
        float mr = *mslot;
        {
          // pass 1 (packed fp16): x = f16(S) * scl + bias (bias fp16, pre-scaled by log2 e) for the whole tile, the tile maximum on the way
          uint32_t xb[BN / 2];
          uint32_t vmax = 0xFC00FC00u;                      // fp16 -inf pair
          float s16[2][CK];
          tmem_ld16(sb, s16[0]);
          tmem_wait_ld();
#pragma unroll
          for (int c = 0; c < NCK; ++c) {
            if (c + 1 < NCK) tmem_ld16(sb + (c + 1) * CK, s16[(c + 1) & 1]);
            const float* sc = s16[c & 1];
            const int k0 = c * CK;
            const __nv_bfloat16* bp = brow + (k0 >> 6) * BM * 64;
#pragma unroll
            for (int e = 0; e < CK / 8; ++e) {
              const uint4 u = *reinterpret_cast<const uint4*>(bp + (((((k0 & 63) >> 3) + e) ^ (row & 7)) << 3));
              const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
              for (int k2 = 0; k2 < 4; ++k2) {
                const int i2 = e * 8 + k2 * 2;
                const uint32_t xv = TF_ABL & 1 ? f16x2(sc[i2], sc[i2 + 1]) : hfma2(f16x2(sc[i2], sc[i2 + 1]), sclh, w[k2]);
                xb[c * (CK / 2) + e * 4 + k2] = xv;
                if (!(TF_ABL & 4)) vmax = hmax2(vmax, xv);
              }
            }
          }
          // the tile maximum against the window top (+0.5): tile 0 and any row above the window put the offset at (max + TF_RAISE); the shift
          // goes into the byte conversion's constant (c = 15 - offset), O / l are rescaled by 2^(old - new) (j > 0: tile 0 has nothing yet)
          const __half2 hm = *reinterpret_cast<const __half2*>(&vmax);
          const float v = fmaxf(__low2float(hm), __high2float(hm));
          const bool fin = v > -60000.f && v < 60000.f;      // a tile of masked keys (-inf) or garbage keeps the offset
          const bool up = fin && (j == 0 || v > mr + 0.5f);
          const float mn = up ? v + (j == 0 ? TF_RAISE : TF_RAISE_E) : mr;
          if (j > 0 ? __any_sync(0xffffffffu, up) : false) {   // rare, warp-uniform: O / l rescale (the PV of the previous tile of this row must have retired)
            if (lane == 0) atomicAdd(STAT + 1, 1);
            const int yp = xs - R;
            wait(kve + yp % NKV, (yp / NKV) & 1);
            tc_fence_after();
            const float f = up ? ex2f(mr - mn) : 1.f;
            const uint32_t oa = tmem_at(tmem + COL_O + r * OC, q * 32, 0);
#pragma unroll 1
            for (int hf = 0; hf < D / 32; ++hf) {               // O in 32-column halves and l (column D: the epilogue reads that one only): one load round trip each
              float ov[32];
              tmem_ld32(oa + hf * 32, ov);
              float lv = 0.f;
              if (hf == D / 32 - 1) lv = tmem_ld1(oa + D);
              tmem_wait_ld();
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                uint32_t oq[8];
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) oq[k2] = __float_as_uint(ov[i * 8 + k2] * f);
                tmem_st8(oa + hf * 32 + i * 8, oq);
              }
              if (hf == D / 32 - 1) tmem_st1(oa + D, __float_as_uint(lv * f));
            }
          }
          mr = mn;
          if (up) *mslot = mr;
          const uint32_t cv = f16x2(15.f - mr, 15.f - mr);
          // pass 2: bytes: 8 t + 1024 with t = relu(x - m + 15) in [0, 15.5] after the vote (no upper clamp), 4 keys a word
#pragma unroll
          for (int w8 = 0; w8 < BN / 32; ++w8) {
            uint32_t pk[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
              const uint32_t ba = hfma2(hfma2r(xb[16 * w8 + 2 * k], 0x3C003C00u, cv), 0x48004800u, 0x64006400u);
              const uint32_t bb = hfma2(hfma2r(xb[16 * w8 + 2 * k + 1], 0x3C003C00u, cv), 0x48004800u, 0x64006400u);
              pk[k] = prmt(ba, bb, 0x6420u);
            }
            tmem_st8(sb + w8 * 8, pk);
          }
        }
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(pf + b);
        x += R;
        __syncwarp();
        if (lane == 0) arrive(be + bs);
      }
    }
  } else if (warp < 14) {
    // ---- epilogue warps: O / l -> bf16 -> out, overlapping the next task's softmax ----
    const int q = warp & 3, row = q * 32 + lane;
    int lt = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      wait(of, lt & 1);
      tc_fence_after();
      const float vs = VSC[bh % H];
      uint32_t ob[R][D / 2];
      int fix = 0;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t oa = tmem_at(tmem + COL_O + r * OC, q * 32, 0);
        const float lr = tmem_ld1(oa + D);
        tmem_wait_ld();
        const float inv = vs / lr;
        float sa = 0.f;
#pragma unroll
        for (int hf = 0; hf < D / 32; ++hf) {                 // 32 channels at a time (64 at once would spill at the 128-register cap)
          float o[32];
          tmem_ld32(oa + hf * 32, o);
          tmem_wait_ld();
#pragma unroll
          for (int c = 0; c < 32; ++c) sa += fabsf(o[c]);
#pragma unroll
          for (int k2 = 0; k2 < 16; ++k2) ob[r][hf * 16 + k2] = pack2(o[2 * k2] * inv, o[2 * k2 + 1] * inv);
        }
        if (g * R + r < N && (!(lr >= LMIN) || !(lr < 3.0e38f) || !(sa < 3.0e38f))) fix |= 1 << r;
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(oe);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (g * R + r >= N) continue;
        if (fix & (1 << r)) {
          const size_t qr = (size_t)row0(bh, g * R + r);
          exact_row<D>(QG + (qr + qt * BM + row) * LD + hcol(bh), KG + qr * LD + hcol(bh), VG + qr * LD + hcol(bh),
                        BG + ((size_t)bh * S + qt * BM + row) * S, LD, S, scl, ob[r]);
          atomicAdd(STAT, 1);
        }
        uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (H * D) + hcol(bh));
#pragma unroll
        for (int c8 = 0; c8 < D / 8; ++c8) dst[c8] = make_uint4(ob[r][c8 * 4], ob[r][c8 * 4 + 1], ob[r][c8 * 4 + 2], ob[r][c8 * 4 + 3]);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// V^T e4m3 planes: out[(b n h) * 32 + c][s] = e4m3(v[b n s][h * 32 + c] * vinv[h]); guard = max over |v vinv| (float bits, atomicMax)
// one CTA a (b n) row and 128-key tile: every head (the row's H * 64 bytes are read contiguously), tiles [H * 32][132] in dynamic shared memory
__global__ void __launch_bounds__(128) f8_vtq_kernel(const __nv_bfloat16* __restrict__ v, int ld, unsigned char* __restrict__ out, int S, int H,
                                                      const float* __restrict__ vinv, unsigned* __restrict__ guard) {
  extern __shared__ __align__(16) unsigned char vq_smem[];
  unsigned char (*tile)[132] = reinterpret_cast<unsigned char (*)[132]>(vq_smem);   // [H * 32][132]
  const int bn = blockIdx.y, s0 = blockIdx.x * 128, t = threadIdx.x;
  constexpr int NG8 = D / 8, KGN = 128 / NG8, NPASS = 32 / KGN;   // channel groups a head, key groups a pass (4 keys each), passes over the 128 keys
  const int e = t % NG8, kg0 = t / NG8;                // channel group (8 channels) and key group (4 keys) of this thread
  const size_t ldq = (size_t)ld / 8;                   // row stride in uint4
  float mx = 0.f;
  for (int pass = 0; pass < NPASS; ++pass) {
    const int kg = kg0 + pass * KGN;
    const uint4* rp = reinterpret_cast<const uint4*>(v + ((size_t)bn * S + s0 + 4 * kg) * ld);
    for (int h = 0; h < H; ++h) {
      const float vi = vinv[h];
      uint32_t pr[4][4];                                  // [key][channel pair]: two e4m3 bytes
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 u = rp[i * ldq + h * NG8 + e];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          const float2 f = bf2f(w[k2]);
          const float a = f.x * vi, b = f.y * vi;
          mx = fmaxf(mx, fmaxf(fabsf(a), fabsf(b)));
          pr[i][k2] = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
        }
      }
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2) {                    // channels 2 k2, 2 k2 + 1 of the 4 keys: one 32-bit word each
        const uint32_t A = pr[0][k2] | (pr[1][k2] << 16), Bq = pr[2][k2] | (pr[3][k2] << 16);
        uint32_t w0, w1;
        asm("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(w0) : "r"(A), "r"(Bq));
        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(w1) : "r"(A), "r"(Bq));
        *reinterpret_cast<uint32_t*>(&tile[h * D + e * 8 + 2 * k2][4 * kg]) = w0;
        *reinterpret_cast<uint32_t*>(&tile[h * D + e * 8 + 2 * k2 + 1][4 * kg]) = w1;
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((t & 31) == 0 && mx > 0.f) { atomicMax(guard, __float_as_uint(mx)); if (mx > 448.f) atomicOr(guard + 1, 1u); }   // guard[0]: running max of |v vinv| (float bits), guard[1]: saturation flag
  __syncthreads();
  for (int rr = t; rr < H * D; rr += 128) {          // one 128-byte row segment a thread
    const uint32_t* src = reinterpret_cast<const uint32_t*>(&tile[rr][0]);
    uint4* dst = reinterpret_cast<uint4*>(out + ((size_t)bn * H * D + rr) * S + s0);
#pragma unroll
    for (int e = 0; e < 8; ++e) dst[e] = make_uint4(src[4 * e], src[4 * e + 1], src[4 * e + 2], src[4 * e + 3]);
  }
}
}  // namespace

// v8t [B*N*H*32, S] uint8 (fresh), guard uint32 [1] (running max of |v vinv| as float bits)
torch::Tensor f8_vtq(torch::Tensor v, torch::Tensor vinv, torch::Tensor guard) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.dim() == 5 && v.size(4) == D, "v: [B, N, S, H, D] bf16");
  const long B = v.size(0), N = v.size(1), S = v.size(2), H = v.size(3), ld = v.stride(2);
  TORCH_CHECK(v.stride(4) == 1 && v.stride(3) == D && v.stride(1) == S * ld && v.stride(0) == N * S * ld && ld % 8 == 0 && S % 128 == 0, "v layout");
  TORCH_CHECK(vinv.numel() == H && vinv.scalar_type() == torch::kFloat32 && guard.scalar_type() == torch::kInt32 && guard.numel() >= 2, "vinv / guard");
  auto out = torch::empty({B * N * H * D, S}, v.options().dtype(torch::kUInt8));
  dim3 grid(S / 128, B * N);
  const int vsm = (int)H * D * 132;
  static bool vattr = false;
  if (!vattr) { C10_CUDA_CHECK(cudaFuncSetAttribute(f8_vtq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * D * 132)); vattr = true; }
  TORCH_CHECK(H <= 16, "f8_vtq: at most 16 heads");
  f8_vtq_kernel<<<grid, 128, vsm, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()), (int)ld,
      out.data_ptr<uint8_t>(), (int)S, (int)H, vinv.data_ptr<float>(), reinterpret_cast<unsigned*>(guard.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// q, k, v [B, N, S, H, 32] bf16 (rows at a common stride), v8t from f8_vtq, bias bf16 [B, H, S, S] (pre-scaled by log2 e), vsc fp32 [H] (1 / vinv)
// -> (out [B, N, S, H, 32] bf16, stat int32 [2] (cumulative: rows recomputed exactly, rare tiles))
std::vector<torch::Tensor> triattn_f8_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor v8t, torch::Tensor bias, double scale,
                                          torch::Tensor vsc, torch::Tensor out_) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, D] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == D && t->stride(2) == ld &&
                t->stride(1) == S * ld && t->stride(0) == N * S * ld, "q, k, v: same shape, [.., H, 32] dense per row, rows at one common stride");
  TORCH_CHECK(ld >= H * D && ld % 8 == 0 && S % BM == 0, "q / k / v row stride / S");
  TORCH_CHECK(v8t.scalar_type() == torch::kUInt8 && v8t.numel() == B * N * H * D * S && vsc.numel() == H, "v8t / vsc");
  TORCH_CHECK(bias.numel() == B * H * S * S, "bias: [B, H, S, S]");
  TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias: fp16 (bf16(bias * log2 e) converted)");
  auto b16 = bias.contiguous();
  const bool og = out_.defined() && out_.numel() > 0;
  if (og) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == q.numel(), "out: contiguous bf16 like q");
  auto out = og ? out_.view(q.sizes()) : torch::empty(q.sizes(), q.options());
  static torch::Tensor stat_dev[64];
  torch::Tensor& stat = stat_dev[q.device().index()];
  if (!stat.defined()) stat = torch::zeros({2}, q.options().dtype(torch::kInt32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  CUtensorMap qm = make_map<2>(q.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BM}, D == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B, "q");
  CUtensorMap km = make_map<2>(k.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BN}, D == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B, "k");
  CUtensorMap vm = make_map<2>(v8t.data_ptr(), {(uint64_t)S, (uint64_t)(B * N * H * D)}, {(uint64_t)S}, {128, D}, CU_TENSOR_MAP_SWIZZLE_128B, "v8t",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  CUtensorMap bm = make_map<3>(b16.data_ptr(), {64, (uint64_t)(B * H * S), (uint64_t)(S / 64)}, {(uint64_t)S, 64}, {64, BM, 2},
                               CU_TENSOR_MAP_SWIZZLE_128B, "bias", CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  const int ntasks = (int)(B * H * (S / BM) * ((N + R - 1) / R));
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_f8_fwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  launch_pdl(triattn_f8_fwd_sm100, dim3(grid), dim3(THREADS), SMEM, at::cuda::getCurrentCUDAStream(), (int)N, (int)H, (int)S, ntasks, (float)scale * L2E,
             qm, km, vm, bm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), vsc.data_ptr<float>(), stat.data_ptr<int>(),
             reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()), (int)ld,
             reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()), reinterpret_cast<const __half*>(b16.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, stat};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_f8_fwd", &triattn_f8_fwd, "sm100 triangle-attention inference forward, e4m3 P and V (D = 32)");
  m.def("f8_vtq", &f8_vtq, "V^T e4m3 planes");
}
