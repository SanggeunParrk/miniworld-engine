// pe_adaln_sm100.cuh -- B200 (sm_100a) forward and row backward of the La-Proteina denoiser's pair builder (PairEmbedAdaLN, 2026-11-17),
// included by pair_embed_adaln_sm100.cu after the A100 source (it uses that source's constants C / MASKED / HAS_SC).
//
//   pair[b,i,j,:] = mask_ij * ( LN(h) * gamma[b] + beta[b] ),   h = LN_affine( T[sep] + T[nS + dist_xt] + T[nS + nX + dist_sc] )
//
// The H100 kernels (pe_fwd: a non-persistent warp per 4-32 pairs of one row, 58 global limit loads a lane before its first store; pe_bwd_rows:
// ONE warp per (b, i) row walking its n pairs, 8 rows a CTA = 48 CTAs for L384 on 148 SMs) ran at 1.2 TB/s forward and 500 us backward at
// L384 on this card. Here (the decoder pair's design, pair_embed_geom/cuda/sm100/pe_ca_sm100.cuh):
//  * persistent warps over chunks of 32 consecutive pairs (flat order); lane l bins pair 32 q + l from the module's own inputs (int32 / int64
//    indices, the limits in shared memory) while the previous chunk is computed, the codes go to the lanes by shuffles;
//  * a group of G lanes owns a pair (32 / G pairs per warp step); lane g holds the 8-channel vectors (k G + g) of the pair: these are the
//    H100 kernel's 32 lanes of 8 channels, so every sum is added in ITS order (8 channels in sequence, then the xor tree 16, 8, 4, 2, 1: the
//    upper levels in registers, the lower log2(G) by shuffles): the pair, dx and the bin codes are the H100 kernel's bits;
//  * gamma / beta are read as they are (fp32 or bf16: the widening is exact), the elementwise arithmetic is fp32x2 (FADD2 / FMUL2 / FFMA2);
//  * backward: every sample has its own CTAs (cps a sample), whose warps add their pairs' parameter gradients in registers in pair order;
//    the CTA partials [dgamma | dbeta | dln_w | dln_b] are added in CTA order by red_rows (fixed order, no atomics: run-to-run bitwise; the
//    order differs from the H100 kernel's row order, so these four differ from it in the last bits). The next step's d pair is in flight
//    while the current one is computed.
#pragma once

namespace pe100ad {

constexpr int WARPS = 4;   // warps per CTA (128 threads)
typedef float2 f2;

__device__ __forceinline__ f2 splat(float v) { return make_float2(v, v); }
__device__ __forceinline__ void ld8(const float* __restrict__ p, f2 (&o)[4]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p + 4));
  o[0] = make_float2(a.x, a.y); o[1] = make_float2(a.z, a.w); o[2] = make_float2(b.x, b.y); o[3] = make_float2(b.z, b.w);
}
__device__ __forceinline__ f2 unpack2(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u)); }
__device__ __forceinline__ void unpack8(const uint4& u, f2 (&o)[4]) { o[0] = unpack2(u.x); o[1] = unpack2(u.y); o[2] = unpack2(u.z); o[3] = unpack2(u.w); }
__device__ __forceinline__ void ld8(const __nv_bfloat16* __restrict__ p, f2 (&o)[4]) { unpack8(__ldg(reinterpret_cast<const uint4*>(p)), o); }
__device__ __forceinline__ uint32_t pack2(f2 v) {
  const __nv_bfloat162 h = __float22bfloat162_rn(v);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// the sum over the G lanes of a pair (fixed tree: bitwise reproducible)
template <int G>
__device__ __forceinline__ float red(float v) {
#pragma unroll
  for (int o = G / 2; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}
__device__ __forceinline__ float hsum(f2 a, f2 b) { return __fadd_rn(__fadd_rn(a.x, a.y), __fadd_rn(b.x, b.y)); }

template <int NVL>
struct Fw {
  f2 yh1[NVL][4], yh2[NVL][4];
  float r1, r2;
};

// the forward statistics of the pair of code w (pe_fwd's fw(), the same operations in the same order); a masked code reads rows 0 / nS
template <int G>
__device__ __forceinline__ void fw(uint32_t w, const float* __restrict__ tbl, int g, const f2 (&lw)[32 / G][4], const f2 (&lb)[32 / G][4], int nS, int nX,
                                   float e1, float e2, Fw<32 / G>& f) {
  constexpr int NVL = 32 / G;
  const float* t0 = tbl + (size_t)(w & 255) * C;
  const float* t1 = tbl + (size_t)(nS + ((w >> 8) & 255)) * C;
  const float* t2 = tbl + (size_t)(nS + nX + ((w >> 16) & 255)) * C;
  f2 x[NVL][4];
  f2 a0 = splat(0.f), a1 = splat(0.f);                         // two fp32x2 accumulators a lane
#pragma unroll
  for (int k = 0; k < NVL; k++) {
    const int c = (k * G + g) * 8;
    f2 t[4];
    ld8(t0 + c, x[k]);
    ld8(t1 + c, t);
#pragma unroll
    for (int v = 0; v < 4; v++) x[k][v] = __fadd2_rn(x[k][v], t[v]);
    if (w & HAS_SC) {
      ld8(t2 + c, t);
#pragma unroll
      for (int v = 0; v < 4; v++) x[k][v] = __fadd2_rn(x[k][v], t[v]);
    }
    a0 = __fadd2_rn(a0, x[k][0]); a1 = __fadd2_rn(a1, x[k][1]); a0 = __fadd2_rn(a0, x[k][2]); a1 = __fadd2_rn(a1, x[k][3]);
  }
  f2 nm = splat(-__fmul_rn(red<G>(hsum(a0, a1)), 1.f / C));
  a0 = a1 = splat(0.f);
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v += 2) {
      x[k][v] = __fadd2_rn(x[k][v], nm);
      x[k][v + 1] = __fadd2_rn(x[k][v + 1], nm);
      a0 = __ffma2_rn(x[k][v], x[k][v], a0);
      a1 = __ffma2_rn(x[k][v + 1], x[k][v + 1], a1);
    }
  f.r1 = rsqrtf(__fmaf_rn(red<G>(hsum(a0, a1)), 1.f / C, e1));
  const f2 r1 = splat(f.r1);
  a0 = a1 = splat(0.f);
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v += 2) {
      f.yh1[k][v] = __fmul2_rn(x[k][v], r1);
      f.yh1[k][v + 1] = __fmul2_rn(x[k][v + 1], r1);
      x[k][v] = __ffma2_rn(f.yh1[k][v], lw[k][v], lb[k][v]);
      x[k][v + 1] = __ffma2_rn(f.yh1[k][v + 1], lw[k][v + 1], lb[k][v + 1]);
      a0 = __fadd2_rn(a0, x[k][v]);
      a1 = __fadd2_rn(a1, x[k][v + 1]);
    }
  nm = splat(-__fmul_rn(red<G>(hsum(a0, a1)), 1.f / C));
  a0 = a1 = splat(0.f);
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v += 2) {
      x[k][v] = __fadd2_rn(x[k][v], nm);
      x[k][v + 1] = __fadd2_rn(x[k][v + 1], nm);
      a0 = __ffma2_rn(x[k][v], x[k][v], a0);
      a1 = __ffma2_rn(x[k][v + 1], x[k][v + 1], a1);
    }
  f.r2 = rsqrtf(__fmaf_rn(red<G>(hsum(a0, a1)), 1.f / C, e2));
  const f2 r2 = splat(f.r2);
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v++) f.yh2[k][v] = __fmul2_rn(x[k][v], r2);
}

template <typename IdxT>
struct Raw {
  float xi, yi, zi, xj, yj, zj, ui, vi, wi, uj, vj, wj;
  IdxT ii, ij;
  uint8_t mi, mj;
};

template <typename IdxT>
__device__ __forceinline__ Raw<IdxT> load_raw(const float* __restrict__ xt, const float* __restrict__ sc, int has_sc, const uint8_t* __restrict__ mask,
                                              const IdxT* __restrict__ idx, unsigned n, unsigned p, unsigned P) {
  Raw<IdxT> r;
  r.mi = r.mj = 0;
  r.ii = r.ij = 0;
  r.xi = r.yi = r.zi = r.xj = r.yj = r.zj = r.ui = r.vi = r.wi = r.uj = r.vj = r.wj = 0.f;
  if (p < P) {
    const unsigned row = p / n;                    // b * n + i
    const unsigned rj = (row / n) * n + (p - row * n);   // b * n + j
    r.mi = __ldg(mask + row);
    r.mj = __ldg(mask + rj);
    r.ii = __ldg(idx + row);
    r.ij = __ldg(idx + rj);
    r.xi = __ldg(xt + (size_t)row * 3); r.yi = __ldg(xt + (size_t)row * 3 + 1); r.zi = __ldg(xt + (size_t)row * 3 + 2);
    r.xj = __ldg(xt + (size_t)rj * 3); r.yj = __ldg(xt + (size_t)rj * 3 + 1); r.zj = __ldg(xt + (size_t)rj * 3 + 2);
    if (has_sc) {
      r.ui = __ldg(sc + (size_t)row * 3); r.vi = __ldg(sc + (size_t)row * 3 + 1); r.wi = __ldg(sc + (size_t)row * 3 + 2);
      r.uj = __ldg(sc + (size_t)rj * 3); r.vj = __ldg(sc + (size_t)rj * 3 + 1); r.wj = __ldg(sc + (size_t)rj * 3 + 2);
    }
  }
  return r;
}

// torch.bucketize: the number of limits strictly below d (the limits in shared memory)
__device__ __forceinline__ int bucket_s(double d, const float* lim, int nlim) {
  int c = 0;
  for (int k = 0; k < nlim; k++) c += ((double)lim[k] < d);
  return c;
}
// the distance of two fp32 points in fp64 (differences, squares, sum, root: the fp64 gold's bin at every limit). In fp32 (the H100
// kernel's dist3) the bin flips where the distance meets a limit within an fp32 rounding, and which side it lands on depends on how the
// compiler contracts the sum (S1 L384 master: an sc distance of 1.5500000416 read 1.5499999523 = the limit 1.55: a wrong table row,
// err 0.385 against the gold where the H100 build read 1.55000007)
__device__ __forceinline__ double dist(float xi, float yi, float zi, float xj, float yj, float zj) {
  const double a = (double)xi - (double)xj, b = (double)yi - (double)yj, c = (double)zi - (double)zj;
  return sqrt(a * a + b * b + c * c);
}

// bin_word(): sequence-separation bin | xt-distance bin << 8 | HAS_SC | sc-distance bin << 16, or MASKED
template <typename IdxT>
__device__ __forceinline__ uint32_t code_of(const Raw<IdxT>& r, int has_sc, const float* xl_s, int nxl, const float* cl_s, int ncl, int nS) {
  if (!(r.mi && r.mj)) return MASKED;
  IdxT s = r.ii - r.ij + (IdxT)((nS - 1) / 2);
  s = min(max(s, (IdxT)0), (IdxT)(nS - 1));
  uint32_t w = (uint32_t)s | ((uint32_t)bucket_s(dist(r.xi, r.yi, r.zi, r.xj, r.yj, r.zj), xl_s, nxl) << 8);
  if (has_sc) w |= HAS_SC | ((uint32_t)bucket_s(dist(r.ui, r.vi, r.wi, r.uj, r.vj, r.wj), cl_s, ncl) << 16);
  return w;
}

// forward: out [P][256] bf16 (P = B n n pairs), bins [P] int32 when TRAIN. G lanes a pair.
template <typename IdxT, typename GbT, int G, bool TRAIN>
__global__ void __launch_bounds__(WARPS * 32)
fwd(const float* __restrict__ tbl, const float* __restrict__ lnw, const float* __restrict__ lnb, const GbT* __restrict__ gamma, const GbT* __restrict__ beta,
    const float* __restrict__ xt, const float* __restrict__ sc, int has_sc, const IdxT* __restrict__ idx, const uint8_t* __restrict__ mask,
    const float* __restrict__ xl, int nxl, const float* __restrict__ cl, int ncl, __nv_bfloat16* __restrict__ out, int* __restrict__ bins, unsigned n,
    unsigned P, int nS, int nX, float e1, float e2, int ch) {
  constexpr int NVL = 32 / G, PPS = 32 / G;   // 8-channel vectors per lane; pairs per warp step
  __shared__ float lim_s[512];
  for (int k = threadIdx.x; k < 512; k += WARPS * 32) lim_s[k] = k < 256 ? (k < nxl ? xl[k] : 0.f) : (k - 256 < ncl ? cl[k - 256] : 0.f);
  __syncthreads();
  const int lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  f2 lw[NVL][4], lb[NVL][4], gm[NVL][4], bt[NVL][4];
#pragma unroll
  for (int k = 0; k < NVL; k++) {
    ld8(lnw + (k * G + g) * 8, lw[k]);
    ld8(lnb + (k * G + g) * 8, lb[k]);
#pragma unroll
    for (int v = 0; v < 4; v++) gm[k][v] = bt[k][v] = splat(0.f);
  }
  const unsigned nn = n * n, nchunk = (P + ch - 1) / ch, wstride = gridDim.x * WARPS;   // ch (a multiple of PPS, <= 32) pairs a chunk
  unsigned bend = 0;                                           // the end of the sample whose gamma / beta the lane holds
  unsigned q = blockIdx.x * WARPS + (threadIdx.x >> 5);
  Raw<IdxT> cur = load_raw(xt, sc, has_sc, mask, idx, n, q < nchunk && lane < ch ? q * ch + lane : P, P);
  while (q < nchunk) {
    const unsigned qn = q + wstride;
    const Raw<IdxT> nxt = load_raw(xt, sc, has_sc, mask, idx, n, qn < nchunk && lane < ch ? qn * ch + lane : P, P);   // in flight meanwhile
    const uint32_t code = code_of(cur, has_sc, lim_s, nxl, lim_s + 256, ncl, nS);
    const unsigned pb = q * ch;
    if constexpr (TRAIN) {
      if (lane < ch && pb + lane < P) bins[pb + lane] = (int)code;
    }
#pragma unroll 1
    for (int t = 0; t < ch / PPS; t++) {
      if (pb + t * PPS >= P) break;                            // warp-uniform
      const unsigned p = pb + t * PPS + grp;
      const uint32_t w = __shfl_sync(0xFFFFFFFFu, code, t * PPS + grp);
      if (p < P && p >= bend) {
        const unsigned b = p / nn;
        bend = (b + 1) * nn;
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          ld8(gamma + (size_t)b * C + (k * G + g) * 8, gm[k]);
          ld8(beta + (size_t)b * C + (k * G + g) * 8, bt[k]);
        }
      }
      Fw<NVL> f;
      fw<G>(w, tbl, g, lw, lb, nS, nX, e1, e2, f);
      if (p < P) {
        const uint32_t keep = (w & MASKED) ? 0u : 0xFFFFFFFFu;
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          uint4 u;
          u.x = pack2(__ffma2_rn(f.yh2[k][0], gm[k][0], bt[k][0])) & keep;
          u.y = pack2(__ffma2_rn(f.yh2[k][1], gm[k][1], bt[k][1])) & keep;
          u.z = pack2(__ffma2_rn(f.yh2[k][2], gm[k][2], bt[k][2])) & keep;
          u.w = pack2(__ffma2_rn(f.yh2[k][3], gm[k][3], bt[k][3])) & keep;
          __stcs(reinterpret_cast<uint4*>(out + (size_t)p * C + (k * G + g) * 8), u);
        }
      }
    }
    cur = nxt;
    q = qn;
  }
}

// backward rows: CTA (bl, r) = CTA r of the cps CTAs of sample bl of this launch's samples; bins / dout / dx / gamma / part start at the
// launch's first sample. dx [.][256] bf16 = grad of the table sum (zeros for masked pairs); part [CTA][dgamma | dbeta | dln_w | dln_b][256].
template <typename GbT, int G>
__global__ void __launch_bounds__(WARPS * 32)
bwd_rows(const float* __restrict__ tbl, const float* __restrict__ lnw, const float* __restrict__ lnb, const GbT* __restrict__ gamma,
         const int* __restrict__ bins, const __nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ dx, float* __restrict__ part, unsigned nn,
         int cps, int nS, int nX, float e1, float e2, int ch) {
  constexpr int NVL = 32 / G, PPS = 32 / G;
  __shared__ float sm[WARPS * 4 * C];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  const unsigned bl = blockIdx.x / cps, r = blockIdx.x % cps;
  const size_t base = (size_t)bl * nn;                         // the sample's first pair
  f2 lw[NVL][4], lb[NVL][4], gm[NVL][4], dg[NVL][4], dbt[NVL][4], dw[NVL][4], dl[NVL][4];
#pragma unroll
  for (int k = 0; k < NVL; k++) {
    ld8(lnw + (k * G + g) * 8, lw[k]);
    ld8(lnb + (k * G + g) * 8, lb[k]);
    ld8(gamma + (size_t)bl * C + (k * G + g) * 8, gm[k]);
#pragma unroll
    for (int v = 0; v < 4; v++) dg[k][v] = dbt[k][v] = dw[k][v] = dl[k][v] = splat(0.f);
  }
  const unsigned nchunk = (nn + ch - 1) / ch, wstride = (unsigned)cps * WARPS;
  unsigned q = r * WARPS + warp;
  auto ldc = [&](unsigned qq) { return qq < nchunk && lane < ch && qq * ch + lane < nn ? (uint32_t)__ldg(bins + base + qq * ch + lane) : MASKED; };
  auto ldg8 = [&](unsigned pl, uint4 (&u)[NVL]) {              // the d pair of local pair pl (zeros past the sample)
#pragma unroll
    for (int k = 0; k < NVL; k++)
      u[k] = pl < nn ? __ldg(reinterpret_cast<const uint4*>(dout + (base + pl) * C + (k * G + g) * 8)) : make_uint4(0u, 0u, 0u, 0u);
  };
  uint32_t code = ldc(q);
  uint4 gn[NVL];
  ldg8(q < nchunk ? q * ch + grp : nn, gn);
  while (q < nchunk) {
    const unsigned qn = q + wstride;
    const uint32_t ncode = ldc(qn);
    const unsigned pb = q * ch;
#pragma unroll 1
    for (int t = 0; t < ch / PPS; t++) {
      if (pb + t * PPS >= nn) break;                           // warp-uniform
      const unsigned pl = pb + t * PPS + grp;
      const uint32_t w = __shfl_sync(0xFFFFFFFFu, code, t * PPS + grp);
      uint4 gc[NVL];
#pragma unroll
      for (int k = 0; k < NVL; k++) gc[k] = gn[k];
      ldg8(t + 1 < ch / PPS ? pl + PPS : (qn < nchunk ? qn * ch + grp : nn), gn);   // the next step's d pair, in flight meanwhile
      const uint32_t keep = (w & MASKED) ? 0u : 0xFFFFFFFFu;   // a masked pair: zero d pair -> nothing added, dx = 0
      Fw<NVL> f;
      fw<G>(w, tbl, g, lw, lb, nS, nX, e1, e2, f);
      f2 d[NVL][4];
      f2 s0 = splat(0.f), s1 = splat(0.f), q0 = splat(0.f), q1 = splat(0.f);
#pragma unroll
      for (int k = 0; k < NVL; k++) {
        f2 gg[4];
        gg[0] = unpack2(gc[k].x & keep); gg[1] = unpack2(gc[k].y & keep); gg[2] = unpack2(gc[k].z & keep); gg[3] = unpack2(gc[k].w & keep);
#pragma unroll
        for (int v = 0; v < 4; v += 2) {
          dbt[k][v] = __fadd2_rn(dbt[k][v], gg[v]);
          dbt[k][v + 1] = __fadd2_rn(dbt[k][v + 1], gg[v + 1]);
          dg[k][v] = __ffma2_rn(gg[v], f.yh2[k][v], dg[k][v]);
          dg[k][v + 1] = __ffma2_rn(gg[v + 1], f.yh2[k][v + 1], dg[k][v + 1]);
          d[k][v] = __fmul2_rn(gg[v], gm[k][v]);
          d[k][v + 1] = __fmul2_rn(gg[v + 1], gm[k][v + 1]);
          s0 = __fadd2_rn(s0, d[k][v]);
          s1 = __fadd2_rn(s1, d[k][v + 1]);
          q0 = __ffma2_rn(d[k][v], f.yh2[k][v], q0);
          q1 = __ffma2_rn(d[k][v + 1], f.yh2[k][v + 1], q1);
        }
      }
      f2 np = splat(-__fmul_rn(red<G>(hsum(s0, s1)), 1.f / C)), na = splat(-__fmul_rn(red<G>(hsum(q0, q1)), 1.f / C));
      const f2 r2 = splat(f.r2), r1 = splat(f.r1);
      s0 = s1 = q0 = q1 = splat(0.f);
#pragma unroll
      for (int k = 0; k < NVL; k++)
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const f2 dy = __fmul2_rn(r2, __ffma2_rn(f.yh2[k][v], na, __fadd2_rn(d[k][v], np)));   // r2 (d2 - p2 - yh2 a2)
          dl[k][v] = __fadd2_rn(dl[k][v], dy);
          dw[k][v] = __ffma2_rn(dy, f.yh1[k][v], dw[k][v]);
          d[k][v] = __fmul2_rn(dy, lw[k][v]);
          if (v & 1) { s1 = __fadd2_rn(s1, d[k][v]); q1 = __ffma2_rn(d[k][v], f.yh1[k][v], q1); }
          else { s0 = __fadd2_rn(s0, d[k][v]); q0 = __ffma2_rn(d[k][v], f.yh1[k][v], q0); }
        }
      np = splat(-__fmul_rn(red<G>(hsum(s0, s1)), 1.f / C));
      na = splat(-__fmul_rn(red<G>(hsum(q0, q1)), 1.f / C));
      if (pl < nn) {
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          uint4 u;
          u.x = pack2(__fmul2_rn(r1, __ffma2_rn(f.yh1[k][0], na, __fadd2_rn(d[k][0], np)))) & keep;
          u.y = pack2(__fmul2_rn(r1, __ffma2_rn(f.yh1[k][1], na, __fadd2_rn(d[k][1], np)))) & keep;
          u.z = pack2(__fmul2_rn(r1, __ffma2_rn(f.yh1[k][2], na, __fadd2_rn(d[k][2], np)))) & keep;
          u.w = pack2(__fmul2_rn(r1, __ffma2_rn(f.yh1[k][3], na, __fadd2_rn(d[k][3], np)))) & keep;
          *reinterpret_cast<uint4*>(dx + (base + pl) * C + (k * G + g) * 8) = u;
        }
      }
    }
    code = ncode;
    q = qn;
  }
  // the CTA partial: the groups of a warp hold the same channels (lane g: (k G + g) 8 .. + 7); lane g of group 0 adds the groups in group
  // order, then the warps are added in warp order
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v++) {
      f2 tg = dg[k][v], tb = dbt[k][v], tw = dw[k][v], tl = dl[k][v];
#pragma unroll
      for (int o = 1; o < PPS; o++) {
        tg.x = __fadd_rn(tg.x, __shfl_sync(0xFFFFFFFFu, dg[k][v].x, (o * G + g) & 31)); tg.y = __fadd_rn(tg.y, __shfl_sync(0xFFFFFFFFu, dg[k][v].y, (o * G + g) & 31));
        tb.x = __fadd_rn(tb.x, __shfl_sync(0xFFFFFFFFu, dbt[k][v].x, (o * G + g) & 31)); tb.y = __fadd_rn(tb.y, __shfl_sync(0xFFFFFFFFu, dbt[k][v].y, (o * G + g) & 31));
        tw.x = __fadd_rn(tw.x, __shfl_sync(0xFFFFFFFFu, dw[k][v].x, (o * G + g) & 31)); tw.y = __fadd_rn(tw.y, __shfl_sync(0xFFFFFFFFu, dw[k][v].y, (o * G + g) & 31));
        tl.x = __fadd_rn(tl.x, __shfl_sync(0xFFFFFFFFu, dl[k][v].x, (o * G + g) & 31)); tl.y = __fadd_rn(tl.y, __shfl_sync(0xFFFFFFFFu, dl[k][v].y, (o * G + g) & 31));
      }
      if (grp == 0) {
        const int c = (k * G + g) * 8 + 2 * v;
        float* m = sm + (size_t)warp * 4 * C;
        m[c] = tg.x; m[c + 1] = tg.y;
        m[C + c] = tb.x; m[C + c + 1] = tb.y;
        m[2 * C + c] = tw.x; m[2 * C + c + 1] = tw.y;
        m[3 * C + c] = tl.x; m[3 * C + c + 1] = tl.y;
      }
    }
  __syncthreads();
  for (int k = threadIdx.x; k < 4 * C; k += WARPS * 32) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; w++) s = __fadd_rn(s, sm[w * 4 * C + k]);
    part[(size_t)blockIdx.x * 4 * C + k] = s;
  }
}

// dgamma / dbeta [B][256] (GbT) = the sum of the sample's cps CTA partials; dln_w / dln_b [256] fp32 = the sum of all B cps partials (block B).
// A block is 4 groups of 256 channels: group j adds the partials j, j + 4, ... in order (8 in flight), then the 4 group sums are added in order.
template <typename GbT>
__global__ void __launch_bounds__(1024)
red_rows(const float* __restrict__ part, GbT* __restrict__ dgamma, GbT* __restrict__ dbeta, float* __restrict__ dlw, float* __restrict__ dlb, int B, int cps) {
  __shared__ float sm[4][2][C];
  const int c = threadIdx.x & 255, j = threadIdx.x >> 8, b = blockIdx.x;
  const bool glob = b == B;
  const float* p0 = glob ? part + 2 * C + c : part + ((size_t)b * cps * 4) * C + c;
  const int np = glob ? B * cps : cps;
  float s0 = 0.f, s1 = 0.f;
  for (int r0 = j; r0 < np; r0 += 32) {
    float u[8], v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int r = r0 + 4 * e;
      u[e] = r < np ? __ldg(p0 + (size_t)r * 4 * C) : 0.f;
      v[e] = r < np ? __ldg(p0 + (size_t)r * 4 * C + C) : 0.f;
    }
#pragma unroll
    for (int e = 0; e < 8; e++) { s0 = __fadd_rn(s0, u[e]); s1 = __fadd_rn(s1, v[e]); }
  }
  sm[j][0][c] = s0;
  sm[j][1][c] = s1;
  __syncthreads();
  if (j == 0) {
    s0 = __fadd_rn(__fadd_rn(__fadd_rn(sm[0][0][c], sm[1][0][c]), sm[2][0][c]), sm[3][0][c]);
    s1 = __fadd_rn(__fadd_rn(__fadd_rn(sm[0][1][c], sm[1][1][c]), sm[2][1][c]), sm[3][1][c]);
    if (glob) { dlw[c] = s0; dlb[c] = s1; }
    else { dgamma[(size_t)b * C + c] = sm100::from_f<GbT>(s0); dbeta[(size_t)b * C + c] = sm100::from_f<GbT>(s1); }
  }
}


// ---- inference with the LayerNorm statistics from tables (2026-11-17) ---------------------------------------------------------------------
// Both LayerNorms act on a sum of (at most three) table rows, so their statistics are sums of per-row and per-row-pair numbers that depend on
// the weights only:
//   h~ = h - mean(h) = sum_k T~[r_k]                       T~[r] = T[r] - mean(T[r])
//   var1 = (1 / C) sum_{k,l} G1[r_k][r_l]                  G1 = T~ T~^T
//   y - mean(y) = r1 sum_k U[r_k] + b~                     U[r] = lw T~[r] - mean(lw T~[r]),  b~ = lb - mean(lb)
//   var2 = (1 / C) (r1^2 sum_{k,l} G2[r_k][r_l] + 2 r1 sum_k V[r_k] + |b~|^2)      G2 = U U^T,  V[r] = U[r] . b~
//   pair = (gm u) (r1 r2) + ((gm b~) r2 + bt),             u = sum_k U[r_k]
// (centred rows: no cancellation that the direct sums do not have). The lane that bins a pair also gets its r1 r2 and r2 from six Gram
// entries and three V; the pair's lanes then need no reduction at all: 5 fp32x2 instructions per two channels instead of ~20 and four
// shuffle trees. The tables are made by two small kernels per weight version (stats_rows, stats_gram; fp64 sums).

// the sum of v over the 256 threads of a block (fixed tree)
__device__ __forceinline__ double block_sum(double v, double* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  double s = 0.;
#pragma unroll
  for (int k = 0; k < 8; k++) s += red[k];
  __syncthreads();
  return s;
}

// block r < NT: TC[r] = T~[r], U[r], V[r] from the weight w [256][NT] (T[r][c] = w[c][r]); block NT: BT = b~ [256], BB = |b~|^2
__global__ void __launch_bounds__(256)
stats_rows(const float* __restrict__ w, const float* __restrict__ lnw, const float* __restrict__ lnb, float* __restrict__ U, float* __restrict__ TC,
           float* __restrict__ V, float* __restrict__ BT, float* __restrict__ BB, int NT) {
  __shared__ double red[8];
  const int c = threadIdx.x, r = blockIdx.x;
  const double lb = (double)__ldg(lnb + c);
  const double bc = lb - block_sum(lb, red) * (1. / C);
  if (r == NT) {
    BT[c] = (float)bc;
    const double bb = block_sum(bc * bc, red);
    if (c == 0) BB[0] = (float)bb;
    return;
  }
  const double t = (double)__ldg(w + (size_t)c * NT + r);
  const float tc = (float)(t - block_sum(t, red) * (1. / C));
  TC[(size_t)r * C + c] = tc;
  const double u0 = (double)__ldg(lnw + c) * (double)tc;
  const float u = (float)(u0 - block_sum(u0, red) * (1. / C));
  U[(size_t)r * C + c] = u;
  const double v = block_sum((double)u * (double)(float)bc, red);
  if (c == 0) V[r] = (float)v;
}

// block r, thread j < NT: GG[r][j] = (T~[r] . T~[j], U[r] . U[j])
__global__ void __launch_bounds__(256)
stats_gram(const float* __restrict__ TC, const float* __restrict__ U, float2* __restrict__ GG, int NT) {
  __shared__ float a[C], b[C];
  const int j = threadIdx.x, r = blockIdx.x;
  a[j] = __ldg(TC + (size_t)r * C + j);
  b[j] = __ldg(U + (size_t)r * C + j);
  __syncthreads();
  if (j >= NT) return;
  double s1 = 0., s2 = 0.;
  for (int c0 = 0; c0 < C; c0 += 32) {
    float4 p[8], q[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      p[k] = __ldg(reinterpret_cast<const float4*>(TC + (size_t)j * C + c0) + k);
      q[k] = __ldg(reinterpret_cast<const float4*>(U + (size_t)j * C + c0) + k);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int c = c0 + 4 * k;
      s1 += (double)p[k].x * a[c] + (double)p[k].y * a[c + 1] + (double)p[k].z * a[c + 2] + (double)p[k].w * a[c + 3];
      s2 += (double)q[k].x * b[c] + (double)q[k].y * b[c + 1] + (double)q[k].z * b[c + 2] + (double)q[k].w * b[c + 3];
    }
  }
  GG[(size_t)r * NT + j] = make_float2((float)s1, (float)s2);
}

struct Gram {
  float2 d0, d1, d2, o01, o02, o12;
  float v0, v1, v2;
};
// the Gram entries and V of the rows of code w, requested (their use comes a chunk later)
__device__ __forceinline__ Gram gram_load(uint32_t w, const float2* __restrict__ GG, const float* __restrict__ V, int nS, int nX, int NT) {
  Gram g;
  g.d0 = g.d1 = g.d2 = g.o01 = g.o02 = g.o12 = make_float2(0.f, 0.f);
  g.v0 = g.v1 = g.v2 = 0.f;
  if (w & MASKED) return g;
  const int r0 = (int)(w & 255), r1 = nS + (int)((w >> 8) & 255);
  g.d0 = __ldg(GG + (size_t)r0 * NT + r0);
  g.d1 = __ldg(GG + (size_t)r1 * NT + r1);
  g.o01 = __ldg(GG + (size_t)r0 * NT + r1);
  g.v0 = __ldg(V + r0);
  g.v1 = __ldg(V + r1);
  if (w & HAS_SC) {
    const int r2 = nS + nX + (int)((w >> 16) & 255);
    g.d2 = __ldg(GG + (size_t)r2 * NT + r2);
    g.o02 = __ldg(GG + (size_t)r0 * NT + r2);
    g.o12 = __ldg(GG + (size_t)r1 * NT + r2);
    g.v2 = __ldg(V + r2);
  }
  return g;
}
// (r1 r2, r2) of the pair
__device__ __forceinline__ float2 gram_stats(const Gram& g, float bb, float e1, float e2) {
  const float s1 = g.d0.x + g.d1.x + g.d2.x + 2.f * (g.o01.x + g.o02.x + g.o12.x);
  const float s2 = g.d0.y + g.d1.y + g.d2.y + 2.f * (g.o01.y + g.o02.y + g.o12.y);
  const float r1 = rsqrtf(fmaxf(s1, 0.f) * (1.f / C) + e1);
  const float v2 = (r1 * r1 * s2 + 2.f * r1 * (g.v0 + g.v1 + g.v2) + bb) * (1.f / C);
  const float r2 = rsqrtf(fmaxf(v2, 0.f) + e2);
  return make_float2(r1 * r2, r2);
}

// inference forward: out [P][256] bf16 from the tables of stats_rows / stats_gram. G lanes a pair, ch pairs a warp chunk.
template <typename IdxT, typename GbT, int G>
__global__ void __launch_bounds__(WARPS * 32)
fwd_s(const float* __restrict__ U, const float2* __restrict__ GG, const float* __restrict__ V, const float* __restrict__ BT, const float* __restrict__ BB,
      const GbT* __restrict__ gamma, const GbT* __restrict__ beta, const float* __restrict__ xt, const float* __restrict__ sc, int has_sc,
      const IdxT* __restrict__ idx, const uint8_t* __restrict__ mask, const float* __restrict__ xl, int nxl, const float* __restrict__ cl, int ncl,
      __nv_bfloat16* __restrict__ out, unsigned n, unsigned P, int nS, int nX, int NT, float e1, float e2, int ch) {
  constexpr int NVL = 32 / G, PPS = 32 / G;
  __shared__ float lim_s[512];
  for (int k = threadIdx.x; k < 512; k += WARPS * 32) lim_s[k] = k < 256 ? (k < nxl ? xl[k] : 0.f) : (k - 256 < ncl ? cl[k - 256] : 0.f);
  __syncthreads();
  const int lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  const float bb = __ldg(BB);
  f2 bc[NVL][4], gm[NVL][4], gb[NVL][4], bt[NVL][4];
#pragma unroll
  for (int k = 0; k < NVL; k++) {
    ld8(BT + (k * G + g) * 8, bc[k]);
#pragma unroll
    for (int v = 0; v < 4; v++) gm[k][v] = gb[k][v] = bt[k][v] = splat(0.f);
  }
  const unsigned nn = n * n, nchunk = (P + ch - 1) / ch, wstride = gridDim.x * WARPS;
  unsigned bend = 0;
  unsigned q = blockIdx.x * WARPS + (threadIdx.x >> 5);
  auto raw_of = [&](unsigned qq) { return load_raw(xt, sc, has_sc, mask, idx, n, qq < nchunk && lane < ch ? qq * ch + lane : P, P); };
  // two chunks deep: the inputs of chunk q + 2 strides and the Gram entries of chunk q + stride are in flight while chunk q is written
  uint32_t code = code_of(raw_of(q), has_sc, lim_s, nxl, lim_s + 256, ncl, nS);
  Gram gr = gram_load(code, GG, V, nS, nX, NT);
  Raw<IdxT> raw1 = raw_of(q + wstride);
  while (q < nchunk) {
    const unsigned qn = q + wstride;
    const Raw<IdxT> raw2 = raw_of(qn + wstride);
    const uint32_t code1 = code_of(raw1, has_sc, lim_s, nxl, lim_s + 256, ncl, nS);
    const Gram gr1 = gram_load(code1, GG, V, nS, nX, NT);
    const float2 st = gram_stats(gr, bb, e1, e2);
    const unsigned pb = q * ch;
#pragma unroll 1
    for (int t = 0; t < ch / PPS; t++) {
      if (pb + t * PPS >= P) break;                            // warp-uniform
      const unsigned p = pb + t * PPS + grp;
      const uint32_t w = __shfl_sync(0xFFFFFFFFu, code, t * PPS + grp);
      const f2 s = splat(__shfl_sync(0xFFFFFFFFu, st.x, t * PPS + grp)), r2 = splat(__shfl_sync(0xFFFFFFFFu, st.y, t * PPS + grp));
      if (p < P && p >= bend) {
        const unsigned b = p / nn;
        bend = (b + 1) * nn;
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          ld8(gamma + (size_t)b * C + (k * G + g) * 8, gm[k]);
          ld8(beta + (size_t)b * C + (k * G + g) * 8, bt[k]);
#pragma unroll
          for (int v = 0; v < 4; v++) gb[k][v] = __fmul2_rn(gm[k][v], bc[k][v]);
        }
      }
      if (p < P) {
        const uint32_t keep = (w & MASKED) ? 0u : 0xFFFFFFFFu;
        const float* u0 = U + (size_t)(w & 255) * C;
        const float* u1 = U + (size_t)(nS + ((w >> 8) & 255)) * C;
        const float* u2 = U + (size_t)(nS + nX + ((w >> 16) & 255)) * C;
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          const int c = (k * G + g) * 8;
          f2 x[4], y[4];
          ld8(u0 + c, x);
          ld8(u1 + c, y);
#pragma unroll
          for (int v = 0; v < 4; v++) x[v] = __fadd2_rn(x[v], y[v]);
          if (w & HAS_SC) {
            ld8(u2 + c, y);
#pragma unroll
            for (int v = 0; v < 4; v++) x[v] = __fadd2_rn(x[v], y[v]);
          }
          uint4 o;
          o.x = pack2(__ffma2_rn(__fmul2_rn(gm[k][0], x[0]), s, __ffma2_rn(gb[k][0], r2, bt[k][0]))) & keep;
          o.y = pack2(__ffma2_rn(__fmul2_rn(gm[k][1], x[1]), s, __ffma2_rn(gb[k][1], r2, bt[k][1]))) & keep;
          o.z = pack2(__ffma2_rn(__fmul2_rn(gm[k][2], x[2]), s, __ffma2_rn(gb[k][2], r2, bt[k][2]))) & keep;
          o.w = pack2(__ffma2_rn(__fmul2_rn(gm[k][3], x[3]), s, __ffma2_rn(gb[k][3], r2, bt[k][3]))) & keep;
          __stcs(reinterpret_cast<uint4*>(out + (size_t)p * C + c), o);
        }
      }
    }
    code = code1;
    gr = gr1;
    raw1 = raw2;
    q = qn;
  }
}

}  // namespace pe100ad
