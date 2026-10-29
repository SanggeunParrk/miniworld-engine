// rows_tf32.cu — the row-wise stages of the atom block's fp32 path (fp32 in, fp32 out, CUDA cores) between atom_gemm_tf32's TF32 products.
//
// Rows of 128 channels, one warp per row, a lane holds 4 consecutive channels (16-byte loads / stores); 8 rows per 256-thread block.
// LayerNorms are two-pass in fp32 (mean, then the variance of the centred values), eps as given. "scale" operands are the stored
// sigmoids s (the conditioning tables keep sigmoid(pre)), so the pre-activation gradients are d s * s (1 - s). Strided operands (column
// blocks of the [M, 768] conditioning table, of the [M, 512] q | k | v | g projection) come with their row stride.
// Every kernel executes griddepcontrol.wait before its first global access (programmatic dependent launch; a no-op when launched without it).
//   f32_ln_aff      cn_k = LN(c) * g_k for up to three weights (the conditioning LayerNorms: AdaLN 1, AdaLN 2, the cross mode's K / V AdaLN),
//                   rows ld apart; optionally a copy of c (oc): the [cn1 | cn2 | c | cnkv] operand of the grouped conditioning GEMM
//   f32_adaln       x = LN(a) * scale + shift (AdaLN 1, AdaLN 2, the K / V AdaLN)
//   f32_gate        gated = sigmoid(g) * o
//   f32_tail_bwd    dt = dy * st, d st_pre = dy * t * st (1 - st)                                    (the transition's output gate)
//   f32_swiglu_bwd  dh -> d[a | b] in the interleaved [M, 512] layout of the forward's u, and h = silu(a) b (the dWs operand)
//   f32_adaln_bwd   da = dres + LN backward of dx * scale; d scale_pre, d shift; optionally (y != 0) the attention's output gate after it:
//                   d so_pre = da * y * so (1 - so), dyo = da * so
//   f32_gate_bwd    dO = dgated * s, dg_pre = dgated * o * s (1 - s) (s = sigmoid(g)) and D = rowsum per head of dO * o  [A, 4, n];
//                   dO feeds only the attention backward's MMAs, so it is stored rounded to tf32 (RNA; kind::tf32 would truncate it)
//                   and D is taken from the rounded values (the dP the MMAs form)
//   f32_ln_bwd      dc += sum_k LN backward of dcn_k * g_k, dg_k += column sums of dcn_k * LN(c) (block-reduced, v4 reductions)
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int C = 128;

DEVI float wsum(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI void ld4(const float* p, float (&x)[4]) {
  const float4 v = __ldg(reinterpret_cast<const float4*>(p));
  x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
}
DEVI void st4(float* p, const float (&x)[4]) { *reinterpret_cast<float4*>(p) = make_float4(x[0], x[1], x[2], x[3]); }
DEVI float sigf(float x) { return rcpf(1.f + __expf(-x)); }       // rcp.approx: the IEEE division was a hot spot (fused_tf32)
DEVI float rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
// x -> (x - mean) * rstd over the warp's 128 channels; returns rstd
DEVI float norm128(float (&x)[4], float eps) {
  const float mean = wsum(x[0] + x[1] + x[2] + x[3]) * (1.f / C);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { x[i] -= mean; v += x[i] * x[i]; }
  const float rstd = 1.f / sqrtf(wsum(v) * (1.f / C) + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= rstd;
  return rstd;
}
// LN backward of the upstream gradient dl (w.r.t. the normalised x = xn): rstd (dl - mean(dl) - xn mean(dl xn))
DEVI void ln_back(const float (&xn)[4], const float (&dl)[4], float rstd, float (&o)[4]) {
  const float m1 = wsum(dl[0] + dl[1] + dl[2] + dl[3]) * (1.f / C);
  const float m2 = wsum(dl[0] * xn[0] + dl[1] * xn[1] + dl[2] * xn[2] + dl[3] * xn[3]) * (1.f / C);
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = rstd * (dl[i] - m1 - xn[i] * m2);
}
DEVI void red4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

#define ROW_OF_WARP                                                       \
  pdl_wait();                                                             \
  pdl_launch();                                                           \
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31; \
  if (row >= M) return;                                                   \
  const int ch = 4 * lane;

// The table-gradient kernels below (tail / AdaLN backward): a warp per table row i (mrows rows; mrows < M when the A = M / mrows samples share
// one conditioning, table row = r % mrows) runs through the rows a mrows + i in the samples' order: the per-row outputs as before, the
// tables' gradient summed over the samples in registers (fp32, fixed order) and written once.
#define TABLE_ROWS                                                        \
  pdl_wait();                                                             \
  pdl_launch();                                                           \
  const int irow = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31; \
  if (irow >= mrows) return;                                              \
  const int ch = 4 * lane, nsamp = M / mrows;

extern "C" __global__ void __launch_bounds__(256)
f32_ln_aff(const float* __restrict__ c, const float* __restrict__ g1, float* __restrict__ o1, const float* __restrict__ g2,
           float* __restrict__ o2, const float* __restrict__ g3, float* __restrict__ o3, float* __restrict__ oc, int ld, int M, float eps) {
  ROW_OF_WARP
  float x[4], g[4], y[4];
  ld4(c + (size_t)row * C + ch, x);
  if (oc != nullptr) st4(oc + (size_t)row * ld + ch, x);
  norm128(x, eps);
  const float* gs[3] = {g1, g2, g3};
  float* os[3] = {o1, o2, o3};
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    if (os[k] == nullptr) continue;
    ld4(gs[k] + ch, g);
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = x[i] * g[i];
    st4(os[k] + (size_t)row * ld + ch, y);
  }
}

extern "C" __global__ void __launch_bounds__(256)
f32_adaln(const float* __restrict__ a, const float* __restrict__ sc, int scld, const float* __restrict__ sh, int shld,
          float* __restrict__ out, int M, float eps, int mrows) {
  ROW_OF_WARP
  float x[4], s[4], b[4];
  ld4(a + (size_t)row * C + ch, x);
  ld4(sc + (size_t)(row % mrows) * scld + ch, s);
  ld4(sh + (size_t)(row % mrows) * shld + ch, b);
  norm128(x, eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = fmaf(x[i], s[i], b[i]);
  st4(out + (size_t)row * C + ch, x);
}

extern "C" __global__ void __launch_bounds__(256)
f32_gate(const float* __restrict__ g, int gld, const float* __restrict__ o, float* __restrict__ out, int M) {
  ROW_OF_WARP
  float gv[4], ov[4];
  ld4(g + (size_t)row * gld + ch, gv);
  ld4(o + (size_t)row * C + ch, ov);
#pragma unroll
  for (int i = 0; i < 4; ++i) ov[i] *= sigf(gv[i]);
  st4(out + (size_t)row * C + ch, ov);
}

extern "C" __global__ void __launch_bounds__(256)
f32_tail_bwd(const float* __restrict__ dy, const float* __restrict__ t, const float* __restrict__ st, int stld, float* __restrict__ dt,
             float* __restrict__ dst, int dstld, int M, int mrows) {
  TABLE_ROWS
  float s[4], p[4] = {0.f, 0.f, 0.f, 0.f};
  ld4(st + (size_t)irow * stld + ch, s);
  for (int a = 0; a < nsamp; ++a) {
    const size_t row = (size_t)a * mrows + irow;
    float d[4], tv[4], o[4];
    ld4(dy + row * C + ch, d);
    ld4(t + row * C + ch, tv);
#pragma unroll
    for (int i = 0; i < 4; ++i) { o[i] = d[i] * s[i]; p[i] += d[i] * tv[i] * s[i] * (1.f - s[i]); }
    st4(dt + row * C + ch, o);
  }
  st4(dst + (size_t)irow * dstld + ch, p);
}

// u / dab [M, 512]: hidden column j of half hb = j / 128 sits at 256 hb + j % 128 (a) and 256 hb + 128 + j % 128 (b)
extern "C" __global__ void __launch_bounds__(256)
f32_swiglu_bwd(const float* __restrict__ dh, const float* __restrict__ u, float* __restrict__ dab, float* __restrict__ hh, int M) {
  ROW_OF_WARP
#pragma unroll
  for (int hb = 0; hb < 2; ++hb) {
    float av[4], bv[4], g[4], da[4], db[4], h[4];
    ld4(u + (size_t)row * 4 * C + 2 * C * hb + ch, av);
    ld4(u + (size_t)row * 4 * C + 2 * C * hb + C + ch, bv);
    ld4(dh + (size_t)row * 2 * C + C * hb + ch, g);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float s = sigf(av[i]), silu = av[i] * s;
      h[i] = silu * bv[i];
      db[i] = g[i] * silu;
      da[i] = g[i] * bv[i] * s * (1.f + av[i] * (1.f - s));
    }
    st4(dab + (size_t)row * 4 * C + 2 * C * hb + ch, da);
    st4(dab + (size_t)row * 4 * C + 2 * C * hb + C + ch, db);
    st4(hh + (size_t)row * 2 * C + C * hb + ch, h);
  }
}

extern "C" __global__ void __launch_bounds__(256)
f32_adaln_bwd(const float* __restrict__ a_, const float* __restrict__ sc, int scld, const float* __restrict__ dx, const float* __restrict__ dres,
              float* __restrict__ da, float* __restrict__ dsc, int dscld, float* __restrict__ dsh, int dshld, const float* __restrict__ y,
              const float* __restrict__ so, int sold, float* __restrict__ dso, int dsold, float* __restrict__ dyo, int M, float eps, int mrows) {
  TABLE_ROWS
  float s[4], g[4], psc[4] = {0.f, 0.f, 0.f, 0.f}, psh[4] = {0.f, 0.f, 0.f, 0.f}, pso[4] = {0.f, 0.f, 0.f, 0.f};
  ld4(sc + (size_t)irow * scld + ch, s);
  if (y != nullptr) ld4(so + (size_t)irow * sold + ch, g);
  for (int a = 0; a < nsamp; ++a) {
    const size_t row = (size_t)a * mrows + irow;
    float x[4], d[4], dl[4], o[4];
    ld4(a_ + row * C + ch, x);
    ld4(dx + row * C + ch, d);
    const float rstd = norm128(x, eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) { dl[i] = d[i] * s[i]; psc[i] += d[i] * x[i] * s[i] * (1.f - s[i]); psh[i] += d[i]; }
    ln_back(x, dl, rstd, o);
    if (dres != nullptr) {
      float r[4];
      ld4(dres + row * C + ch, r);
#pragma unroll
      for (int i = 0; i < 4; ++i) o[i] += r[i];
    }
    st4(da + row * C + ch, o);
    if (y != nullptr) {                                                 // a1 = a + so * y upstream: the output gate's gradients
      float yv[4], q[4];
      ld4(y + row * C + ch, yv);
#pragma unroll
      for (int i = 0; i < 4; ++i) { pso[i] += o[i] * yv[i] * g[i] * (1.f - g[i]); q[i] = o[i] * g[i]; }
      st4(dyo + row * C + ch, q);
    }
  }
  st4(dsc + (size_t)irow * dscld + ch, psc);
  st4(dsh + (size_t)irow * dshld + ch, psh);
  if (y != nullptr) st4(dso + (size_t)irow * dsold + ch, pso);
}

// rows are (sample, atom) pairs of an [A, n] grid: D[a, h, i] for row a n + i; head h = channels 32 h .. 32 h + 31 = lanes 8 h .. 8 h + 7
extern "C" __global__ void __launch_bounds__(256)
f32_gate_bwd(const float* __restrict__ dgt, const float* __restrict__ g, int gld, const float* __restrict__ o, float* __restrict__ dout,
             float* __restrict__ dg, int dgld, float* __restrict__ D, int M, int n) {
  ROW_OF_WARP
  float d[4], gv[4], ov[4], dov[4], dgv[4];
  ld4(dgt + (size_t)row * C + ch, d);
  ld4(g + (size_t)row * gld + ch, gv);
  ld4(o + (size_t)row * C + ch, ov);
  float part = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float s = sigf(gv[i]);
    dov[i] = rna_tf32(d[i] * s);
    dgv[i] = d[i] * ov[i] * s * (1.f - s);
    part = fmaf(dov[i], ov[i], part);
  }
  st4(dout + (size_t)row * C + ch, dov);
  st4(dg + (size_t)row * dgld + ch, dgv);
  part += __shfl_xor_sync(0xffffffffu, part, 1);
  part += __shfl_xor_sync(0xffffffffu, part, 2);
  part += __shfl_xor_sync(0xffffffffu, part, 4);
  if ((lane & 7) == 0) {
    const int a = row / n, i = row - a * n;
    D[((size_t)a * 4 + (lane >> 3)) * n + i] = part;
  }
}

// grid-stride over the rows; dg [3 * 128] (k = 0, 1, 2) += block sums. dcn3 / g3 may be null (two LayerNorms).
extern "C" __global__ void __launch_bounds__(256)
f32_ln_bwd(const float* __restrict__ c, float* __restrict__ dc, const float* __restrict__ dcn1, const float* __restrict__ g1,
           const float* __restrict__ dcn2, const float* __restrict__ g2, const float* __restrict__ dcn3, const float* __restrict__ g3,
           float* __restrict__ dg, int M, float eps) {
  __shared__ float red[8][3 * C];
  pdl_wait();
  pdl_launch();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, ch = 4 * lane;
  const int nk = dcn3 != nullptr ? 3 : 2;
  const float* dcns[3] = {dcn1, dcn2, dcn3};
  const float* gs[3] = {g1, g2, g3};
  float acc[3][4];
#pragma unroll
  for (int k = 0; k < 3; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[k][i] = 0.f;
  for (int row = blockIdx.x * 8 + warp; row < M; row += gridDim.x * 8) {
    float x[4], tot[4];
    ld4(c + (size_t)row * C + ch, x);
    const float rstd = norm128(x, eps);
    {
      const float4 t = *reinterpret_cast<const float4*>(dc + (size_t)row * C + ch);
      tot[0] = t.x; tot[1] = t.y; tot[2] = t.z; tot[3] = t.w;
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (k >= nk) break;
      float d[4], g[4], dl[4], o[4];
      ld4(dcns[k] + (size_t)row * C + ch, d);
      ld4(gs[k] + ch, g);
#pragma unroll
      for (int i = 0; i < 4; ++i) { acc[k][i] = fmaf(d[i], x[i], acc[k][i]); dl[i] = d[i] * g[i]; }
      ln_back(x, dl, rstd, o);
#pragma unroll
      for (int i = 0; i < 4; ++i) tot[i] += o[i];
    }
    st4(dc + (size_t)row * C + ch, tot);
  }
#pragma unroll
  for (int k = 0; k < 3; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][k * C + ch + i] = acc[k][i];
  __syncthreads();
  if (threadIdx.x < 3 * C / 4) {
    const int j = 4 * threadIdx.x;
    if (j < nk * C) {
      float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] += red[w][j + i];
      red4(dg + j, s[0], s[1], s[2], s[3]);
    }
  }
}

// column sums of a [M, 128 g] fp32 view (row pitch ld): grid (G, g); block (blockIdx.x, blockIdx.y) sums rows blockIdx.x * 8 + warp, + G * 8, ...
// of its 128-column block and writes the block's 128 partial sums to part[blockIdx.x][128 blockIdx.y ..]; the G partial rows are summed by the host
// (the weight-gradient bias sums: torch's reduce_kernel took 130-170 us per [196608, 128..768] sum, ten times a bandwidth bound)
extern "C" __global__ void __launch_bounds__(256)
f32_colsum(const float* __restrict__ x, int ld, float* __restrict__ part, int ldp, int M) {
  pdl_wait();
  pdl_launch();
  __shared__ float4 red[8][32];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, cb = blockIdx.y;
  const float* xp = x + 128 * cb + 4 * lane;
  float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
  const int step = gridDim.x * 8;
  int r = blockIdx.x * 8 + w;
  for (; r + step < M; r += 2 * step) {                              // two independent 16-byte loads per lane in flight
    const float4 v0 = __ldg(reinterpret_cast<const float4*>(xp + (size_t)r * ld));
    const float4 v1 = __ldg(reinterpret_cast<const float4*>(xp + (size_t)(r + step) * ld));
    a0.x += v0.x; a0.y += v0.y; a0.z += v0.z; a0.w += v0.w;
    a1.x += v1.x; a1.y += v1.y; a1.z += v1.z; a1.w += v1.w;
  }
  if (r < M) {
    const float4 v0 = __ldg(reinterpret_cast<const float4*>(xp + (size_t)r * ld));
    a0.x += v0.x; a0.y += v0.y; a0.z += v0.z; a0.w += v0.w;
  }
  a0.x += a1.x; a0.y += a1.y; a0.z += a1.z; a0.w += a1.w;
  red[w][lane] = a0;
  __syncthreads();
  if (w == 0) {
    float4 s = red[0][lane];
#pragma unroll
    for (int i = 1; i < 8; ++i) { const float4 v = red[i][lane]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    *reinterpret_cast<float4*>(part + (size_t)blockIdx.x * ldp + 128 * cb + 4 * lane) = s;
  }
}

// ------------------------------------------------------------------------------------------------ ESMFold2 (``LocalDiTBlock(**ESMFOLD2)``)
// Its AdaLNs are RMSNorms of the atoms with raw (1 + s) scales, its output gates raw, and every conditioning table a product of silu(c):
//   f32_adaln_rms      x = RMS(a) * scale + shift                       (scale = the stored 1 + s)
//   f32_adaln_bwd_rms  f32_adaln_bwd with the RMSNorm backward and raw scales: d scale = dx RMS(a), d so = da y (no sigmoid backward)
//   f32_tail_bwd_rms   dt = dy * st, d st = dy * t                       (the raw transition gate)
//   f32_silu           sc = silu(c); f32_silu_bwd: dc = g silu'(c) (g: the tables' gradient through their weights)
DEVI float rms128(float (&x)[4], float eps) {
  const float rstd = 1.f / sqrtf(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) * (1.f / C) + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= rstd;
  return rstd;
}
DEVI void rms_back(const float (&xn)[4], const float (&dl)[4], float rstd, float (&o)[4]) {
  const float m2 = wsum(dl[0] * xn[0] + dl[1] * xn[1] + dl[2] * xn[2] + dl[3] * xn[3]) * (1.f / C);
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = rstd * (dl[i] - xn[i] * m2);
}

extern "C" __global__ void __launch_bounds__(256)
f32_adaln_rms(const float* __restrict__ a, const float* __restrict__ sc, int scld, const float* __restrict__ sh, int shld,
              float* __restrict__ out, int M, float eps, int mrows) {
  ROW_OF_WARP
  float x[4], s[4], b[4];
  ld4(a + (size_t)row * C + ch, x);
  ld4(sc + (size_t)(row % mrows) * scld + ch, s);
  ld4(sh + (size_t)(row % mrows) * shld + ch, b);
  rms128(x, eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = fmaf(x[i], s[i], b[i]);
  st4(out + (size_t)row * C + ch, x);
}

extern "C" __global__ void __launch_bounds__(256)
f32_adaln_bwd_rms(const float* __restrict__ a_, const float* __restrict__ sc, int scld, const float* __restrict__ dx, const float* __restrict__ dres,
                  float* __restrict__ da, float* __restrict__ dsc, int dscld, float* __restrict__ dsh, int dshld, const float* __restrict__ y,
                  const float* __restrict__ so, int sold, float* __restrict__ dso, int dsold, float* __restrict__ dyo, int M, float eps, int mrows) {
  TABLE_ROWS
  float s[4], g[4], psc[4] = {0.f, 0.f, 0.f, 0.f}, psh[4] = {0.f, 0.f, 0.f, 0.f}, pso[4] = {0.f, 0.f, 0.f, 0.f};
  ld4(sc + (size_t)irow * scld + ch, s);
  if (y != nullptr) ld4(so + (size_t)irow * sold + ch, g);
  for (int a = 0; a < nsamp; ++a) {
    const size_t row = (size_t)a * mrows + irow;
    float x[4], d[4], dl[4], o[4];
    ld4(a_ + row * C + ch, x);
    ld4(dx + row * C + ch, d);
    const float rstd = rms128(x, eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) { dl[i] = d[i] * s[i]; psc[i] += d[i] * x[i]; psh[i] += d[i]; }
    rms_back(x, dl, rstd, o);
    if (dres != nullptr) {
      float r[4];
      ld4(dres + row * C + ch, r);
#pragma unroll
      for (int i = 0; i < 4; ++i) o[i] += r[i];
    }
    st4(da + row * C + ch, o);
    if (y != nullptr) {                                                 // a1 = a + so * y upstream: the output gate's gradients
      float yv[4], q[4];
      ld4(y + row * C + ch, yv);
#pragma unroll
      for (int i = 0; i < 4; ++i) { pso[i] += o[i] * yv[i]; q[i] = o[i] * g[i]; }
      st4(dyo + row * C + ch, q);
    }
  }
  st4(dsc + (size_t)irow * dscld + ch, psc);
  st4(dsh + (size_t)irow * dshld + ch, psh);
  if (y != nullptr) st4(dso + (size_t)irow * dsold + ch, pso);
}

extern "C" __global__ void __launch_bounds__(256)
f32_tail_bwd_rms(const float* __restrict__ dy, const float* __restrict__ t, const float* __restrict__ st, int stld, float* __restrict__ dt,
                 float* __restrict__ dst, int dstld, int M, int mrows) {
  TABLE_ROWS
  float s[4], p[4] = {0.f, 0.f, 0.f, 0.f};
  ld4(st + (size_t)irow * stld + ch, s);
  for (int a = 0; a < nsamp; ++a) {
    const size_t row = (size_t)a * mrows + irow;
    float d[4], tv[4], o[4];
    ld4(dy + row * C + ch, d);
    ld4(t + row * C + ch, tv);
#pragma unroll
    for (int i = 0; i < 4; ++i) { o[i] = d[i] * s[i]; p[i] += d[i] * tv[i]; }
    st4(dt + row * C + ch, o);
  }
  st4(dst + (size_t)irow * dstld + ch, p);
}

extern "C" __global__ void __launch_bounds__(256)
f32_silu(const float* __restrict__ c, float* __restrict__ out, int M) {
  ROW_OF_WARP
  float x[4];
  ld4(c + (size_t)row * C + ch, x);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = x[i] / (1.f + __expf(-x[i]));
  st4(out + (size_t)row * C + ch, x);
}

extern "C" __global__ void __launch_bounds__(256)
f32_silu_bwd(const float* __restrict__ c, const float* __restrict__ g, float* __restrict__ dc, int M) {
  ROW_OF_WARP
  float x[4], gv[4];
  ld4(c + (size_t)row * C + ch, x);
  ld4(g + (size_t)row * C + ch, gv);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float s = 1.f / (1.f + __expf(-x[i]));
    x[i] = gv[i] * s * (1.f + x[i] * (1.f - s));
  }
  st4(dc + (size_t)row * C + ch, x);
}
