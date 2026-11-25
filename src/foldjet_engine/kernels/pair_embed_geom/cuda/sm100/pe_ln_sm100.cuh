// pe_ln_sm100.cuh -- B200 (sm_100a) LayerNorm backward of the backbone pair builders (PairEmbedGeom MODE 0 / PairEmbedCA MODE 1, 2026-11-17),
// included by pair_embed_geom_sm100.cu after pe_ca_sm100.cuh (it uses the A100 source's constants and pe100ca's bf16 pack helpers).
//
//   hat = LN(bf16(pair)),  dx = rstd * (dh - mean(dh) - hat * mean(dh * hat))     per pair, bf16, zeros for padding
//
// The H100 kernel (pg_ln_bwd: one CTA of 16 warps per SM holding the table in shared memory, a warp per pair) took 14.7 us at L128 and 84.5 us
// at L384 on this card (S1, encoder). Here, as in pe_ca_sm100.cuh / pe_adaln_sm100.cuh: persistent warps over chunks of 32 consecutive pairs,
// the 16-byte codes of the next chunk and the d pair of the next step in flight while a step is computed, a group of G lanes a pair (32 / G
// pairs per warp step), the table through the L1, fp32x2 elementwise arithmetic.
// The rows of a pair are added in the H100 kernel's order (the pair itself has its bits); the LayerNorm sums use two fp32x2
// accumulators a lane and a shuffle tree over the G lanes (fixed order: bitwise reproducible; not the H100 kernel's order).
#pragma once

namespace pe100ln {

constexpr int WARPS = 4;
typedef float2 f2;
using pe100ca::pack2;
using pe100ca::unpack2;

__device__ __forceinline__ f2 splat(float v) { return make_float2(v, v); }

template <int G>
__device__ __forceinline__ float red(float v) {
#pragma unroll
  for (int o = G / 2; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}
__device__ __forceinline__ float hsum(f2 a, f2 b) { return __fadd_rn(__fadd_rn(a.x, a.y), __fadd_rn(b.x, b.y)); }
// channels c .. c + 7 of table row r as fp32 (S: tbl is a shared-memory copy of the table, ld.shared)
template <bool S = false>
__device__ __forceinline__ void ldrow(const __nv_bfloat16* __restrict__ tbl, int r, int c, f2 (&o)[4]) {
  uint4 u;
  if constexpr (S) {
    u = *reinterpret_cast<const uint4*>(tbl + r * C + c);   // the caller passes the __shared__ array itself: ld.shared after inlining
  } else {
    u = __ldg(reinterpret_cast<const uint4*>(tbl + (size_t)r * C + c));
  }
  o[0] = unpack2(u.x); o[1] = unpack2(u.y); o[2] = unpack2(u.z); o[3] = unpack2(u.w);
}

// channels c .. c + 7 of the pair of code (w0 with its LIVE bit, w1, chain product pc): the fp32 sum of its table rows in gather_rows' order
// (a dead pair: zeros), not yet rounded to bf16
template <int MODE, typename TT, bool S = false>
__device__ __forceinline__ void pair8(const TT* __restrict__ tbl, uint32_t w0, uint32_t w1, float pc, int c, f2 (&x)[4]) {
  x[0] = x[1] = x[2] = x[3] = splat(0.f);
  if (!(w0 & LIVE)) return;
  auto add = [&](int r) {
    f2 t[4];
    ldrow<S>(tbl, r, c, t);
    x[0] = __fadd2_rn(x[0], t[0]); x[1] = __fadd2_rn(x[1], t[1]); x[2] = __fadd2_rn(x[2], t[2]); x[3] = __fadd2_rn(x[3], t[3]);
  };
  add(w0 & 127);
  if constexpr (MODE == 0) {
    if (w0 & F_DIST) {
      add(R_DIST + ((w0 >> 8) & 31));
      add(R_DIST + 21 + ((w0 >> 13) & 31));
      add(R_DIST + 42 + ((w0 >> 18) & 31));
      add(R_DIST + 63 + ((w0 >> 23) & 31));
    }
    if (w0 & F_ORI) {
#pragma unroll
      for (int k = 0; k < 5; k++) add(R_ANG + 21 * k + ((w1 >> (5 * k)) & 31));
    }
    if (pc != 0.f) {
      f2 t[4];
      ldrow<S>(tbl, R_CHAIN, c, t);
      const f2 s = splat(pc);
      x[0] = __ffma2_rn(s, t[0], x[0]); x[1] = __ffma2_rn(s, t[1], x[1]); x[2] = __ffma2_rn(s, t[2], x[2]); x[3] = __ffma2_rn(s, t[3], x[3]);
    }
  } else {
    add(R_CAD + ((w0 >> 8) & 31));
  }
}

// the LayerNorm statistics of a pair: x (the fp32 row sums) -> hat = (bf16(x) - mean) * rstd; returns rstd
template <int G>
__device__ __forceinline__ float ln_hat(f2 (&x)[32 / G][4], float eps) {
  constexpr int NVL = 32 / G;
  f2 a0 = splat(0.f), a1 = splat(0.f);
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v += 2) {
      x[k][v] = unpack2(pack2(x[k][v]));                       // the pair as a bf16 Linear output holds it
      x[k][v + 1] = unpack2(pack2(x[k][v + 1]));
      a0 = __fadd2_rn(a0, x[k][v]);
      a1 = __fadd2_rn(a1, x[k][v + 1]);
    }
  const f2 nm = splat(-__fmul_rn(red<G>(hsum(a0, a1)), 1.f / C));
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
  const float rstd = rsqrtf(__fmaf_rn(red<G>(hsum(a0, a1)), 1.f / C, eps));
  const f2 r = splat(rstd);
#pragma unroll
  for (int k = 0; k < NVL; k++)
#pragma unroll
    for (int v = 0; v < 4; v++) x[k][v] = __fmul2_rn(x[k][v], r);
  return rstd;
}

template <int MODE, typename TT, int G>
__global__ void __launch_bounds__(WARPS * 32)
ln_bwd(const TT* __restrict__ tbl, const uint4* __restrict__ codes, const __nv_bfloat16* __restrict__ dh, __nv_bfloat16* __restrict__ dx, unsigned P,
       float eps, int ch) {
  constexpr int NVL = 32 / G, PPS = 32 / G;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the table gradient (PDL) may set up meanwhile
  const int lane = threadIdx.x & 31, grp = lane / G, g = lane % G;
  const unsigned nchunk = (P + ch - 1) / ch, wstride = gridDim.x * WARPS;     // ch (8 / 16 / 32) pairs a chunk
  unsigned q = blockIdx.x * WARPS + (threadIdx.x >> 5);
  auto ldc = [&](unsigned qq) { return qq < nchunk && lane < ch && qq * ch + lane < P ? __ldg(codes + (size_t)qq * ch + lane) : make_uint4(0u, 0u, 0u, 0u); };
  auto ldg8 = [&](unsigned p, uint4 (&u)[NVL]) {
#pragma unroll
    for (int k = 0; k < NVL; k++)
      u[k] = p < P ? __ldg(reinterpret_cast<const uint4*>(dh + (size_t)p * C + (k * G + g) * 8)) : make_uint4(0u, 0u, 0u, 0u);
  };
  uint4 code = ldc(q);
  uint4 gn[NVL];
  ldg8(q < nchunk ? q * ch + grp : P, gn);
  while (q < nchunk) {
    const unsigned qn = q + wstride;
    const uint4 ncode = ldc(qn);
    const unsigned pb = q * ch;
#pragma unroll 1
    for (int t = 0; t < ch / PPS; t++) {
      if (pb + t * PPS >= P) break;                            // warp-uniform
      const unsigned p = pb + t * PPS + grp;
      const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, code.x, t * PPS + grp);
      uint32_t w1 = 0u;
      float pc = 0.f;
      if constexpr (MODE == 0) {
        w1 = __shfl_sync(0xFFFFFFFFu, code.y, t * PPS + grp);
        pc = __uint_as_float(__shfl_sync(0xFFFFFFFFu, code.z, t * PPS + grp));
      }
      uint4 gc[NVL];
#pragma unroll
      for (int k = 0; k < NVL; k++) gc[k] = gn[k];
      ldg8(t + 1 < ch / PPS ? p + PPS : (qn < nchunk ? qn * ch + grp : P), gn);   // the next step's d pair, in flight meanwhile
      const uint32_t keep = (w0 & LIVE) ? 0xFFFFFFFFu : 0u;
      f2 x[NVL][4];
#pragma unroll
      for (int k = 0; k < NVL; k++) pair8<MODE, TT>(tbl, w0, w1, pc, (k * G + g) * 8, x[k]);
      const f2 r = splat(ln_hat<G>(x, eps));
      f2 gg[NVL][4];
      f2 s0 = splat(0.f), s1 = splat(0.f), q0 = splat(0.f), q1 = splat(0.f);
#pragma unroll
      for (int k = 0; k < NVL; k++) {
        gg[k][0] = unpack2(gc[k].x & keep); gg[k][1] = unpack2(gc[k].y & keep); gg[k][2] = unpack2(gc[k].z & keep); gg[k][3] = unpack2(gc[k].w & keep);
#pragma unroll
        for (int v = 0; v < 4; v += 2) {
          s0 = __fadd2_rn(s0, gg[k][v]);
          s1 = __fadd2_rn(s1, gg[k][v + 1]);
          q0 = __ffma2_rn(gg[k][v], x[k][v], q0);
          q1 = __ffma2_rn(gg[k][v + 1], x[k][v + 1], q1);
        }
      }
      const f2 nam = splat(-__fmul_rn(red<G>(hsum(s0, s1)), 1.f / C)), nbm = splat(-__fmul_rn(red<G>(hsum(q0, q1)), 1.f / C));
      if (p < P) {
#pragma unroll
        for (int k = 0; k < NVL; k++) {
          uint4 u;
          u.x = pack2(__fmul2_rn(r, __ffma2_rn(x[k][0], nbm, __fadd2_rn(gg[k][0], nam)))) & keep;
          u.y = pack2(__fmul2_rn(r, __ffma2_rn(x[k][1], nbm, __fadd2_rn(gg[k][1], nam)))) & keep;
          u.z = pack2(__fmul2_rn(r, __ffma2_rn(x[k][2], nbm, __fadd2_rn(gg[k][2], nam)))) & keep;
          u.w = pack2(__fmul2_rn(r, __ffma2_rn(x[k][3], nbm, __fadd2_rn(gg[k][3], nam)))) & keep;
          *reinterpret_cast<uint4*>(dx + (size_t)p * C + (k * G + g) * 8) = u;
        }
      }
    }
    code = ncode;
    q = qn;
  }
}

}  // namespace pe100ln

