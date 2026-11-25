// var_rows_tf32.cu — the row-wise fp32 stages the B200 variant blocks' fp32 training adds to rows_tf32.cu's (2026-11-17; sm_100a, CUDA cores;
// rows of 128 channels, one warp per row, a lane holds 4 consecutive channels):
//   f32_adaln_bwd_id  f32_adaln_bwd for Chai-1's identity AdaLN: the scale is raw (1 + W c), so d scale = dx LN(a) (no sigmoid backward);
//                     da = dres + LN backward of dx * scale, d shift = dx. Tables with fewer rows than a (shared by the samples): summed.
//   f32_kqln_fwd      RoseTTAFold3's flat q / k LayerNorm on the projection P [M][512] = [q | k | v | g], in place: the raw q | k out to QKR
//                     [M][256], y_p = rna_tf32(LN(x_p) w_p + b_p) (q / k only feed the attention's kind::tf32 MMAs, which truncate)
//   f32_kqln_bwd      its backward on dP's dq | dk, in place, and the column sums [d wq | d bq | d wk | d bk] as one partial row per CTA
//                     (PARTS [grid][512]; the host sums them in order: deterministic)
// KQ [4][128] fp32 = [wq | bq | wk | bk].
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

constexpr int C = 128;
#define DEVI static __device__ __forceinline__
DEVI float wsum(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI void ld4(const float* p, float (&x)[4]) {
  const float4 v = __ldg(reinterpret_cast<const float4*>(p));
  x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
}
DEVI void ld4p(const float* p, float (&x)[4]) {                           // a plain load (of a buffer this kernel also writes)
  const float4 v = *reinterpret_cast<const float4*>(p);
  x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
}
DEVI void st4(float* p, const float (&x)[4]) { *reinterpret_cast<float4*>(p) = make_float4(x[0], x[1], x[2], x[3]); }
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
DEVI void ln_back(const float (&xn)[4], const float (&dl)[4], float rstd, float (&o)[4]) {
  const float m1 = wsum(dl[0] + dl[1] + dl[2] + dl[3]) * (1.f / C);
  const float m2 = wsum(dl[0] * xn[0] + dl[1] * xn[1] + dl[2] * xn[2] + dl[3] * xn[3]) * (1.f / C);
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = rstd * (dl[i] - m1 - xn[i] * m2);
}

extern "C" __global__ void __launch_bounds__(256)
f32_adaln_bwd_id(const float* __restrict__ a_, const float* __restrict__ sc, int scld, const float* __restrict__ dx, const float* __restrict__ dres,
                 float* __restrict__ da, float* __restrict__ dsc, int dscld, float* __restrict__ dsh, int dshld, int M, float eps, int mrows) {
  const int irow = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (irow >= mrows) return;
  const int ch = 4 * lane, nsamp = M / mrows;
  float s[4], psc[4] = {0.f, 0.f, 0.f, 0.f}, psh[4] = {0.f, 0.f, 0.f, 0.f};
  ld4(sc + (size_t)irow * scld + ch, s);
  for (int a = 0; a < nsamp; ++a) {
    const size_t row = (size_t)a * mrows + irow;
    float x[4], d[4], dl[4], o[4];
    ld4(a_ + row * C + ch, x);
    ld4(dx + row * C + ch, d);
    const float rstd = norm128(x, eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) { dl[i] = d[i] * s[i]; psc[i] += d[i] * x[i]; psh[i] += d[i]; }
    ln_back(x, dl, rstd, o);
    if (dres != nullptr) {
      float r[4];
      ld4(dres + row * C + ch, r);
#pragma unroll
      for (int i = 0; i < 4; ++i) o[i] += r[i];
    }
    st4(da + row * C + ch, o);
  }
  st4(dsc + (size_t)irow * dscld + ch, psc);
  st4(dsh + (size_t)irow * dshld + ch, psh);
}

extern "C" __global__ void __launch_bounds__(256)
f32_kqln_fwd(float* __restrict__ P, const float* __restrict__ KQ, float* __restrict__ QKR, int M, float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const int ch = 4 * lane;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    float x[4], w[4], b[4];
    ld4p(P + (size_t)row * 512 + 128 * p + ch, x);
    ld4(KQ + 256 * p + ch, w);
    ld4(KQ + 256 * p + 128 + ch, b);
    st4(QKR + (size_t)row * 256 + 128 * p + ch, x);
    norm128(x, eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = rna_tf32(fmaf(x[i], w[i], b[i]));
    st4(P + (size_t)row * 512 + 128 * p + ch, x);
  }
}

extern "C" __global__ void __launch_bounds__(256)
f32_kqln_bwd(float* __restrict__ dP, const float* __restrict__ QKR, const float* __restrict__ KQ, float* __restrict__ PARTS, int M, float eps) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, ch = 4 * lane;
  __shared__ float red[8][32][16];
  float w[2][4], acc[16];                                                  // acc: [p][dw 4 | db 4]
  ld4(KQ + ch, w[0]);
  ld4(KQ + 256 + ch, w[1]);
#pragma unroll
  for (int e = 0; e < 16; ++e) acc[e] = 0.f;
  for (int r = 8 * (int)blockIdx.x + warp; r < M; r += 8 * (int)gridDim.x) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      float x[4], d[4], g[4], o[4];
      ld4(QKR + (size_t)r * 256 + 128 * p + ch, x);
      ld4p(dP + (size_t)r * 512 + 128 * p + ch, d);
      const float rstd = norm128(x, eps);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[8 * p + j] += d[j] * x[j];
        acc[8 * p + 4 + j] += d[j];
        g[j] = d[j] * w[p][j];
      }
      ln_back(x, g, rstd, o);
      st4(dP + (size_t)r * 512 + 128 * p + ch, o);
    }
  }
#pragma unroll
  for (int e = 0; e < 16; ++e) red[warp][lane][e] = acc[e];
  __syncthreads();
  for (int c = tid; c < 512; c += 256) {
    const int p = c >> 8, kind = (c >> 7) & 1, cc = c & 127;
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) v += red[k][cc >> 2][8 * p + 4 * kind + (cc & 3)];
    PARTS[(size_t)blockIdx.x * 512 + c] = v;
  }
}

// f32_ln_bwd2: the two conditioning LayerNorms' backward from ONE grouped product D [M][384] = [dcn1 | dcn2 | dc0] (the variant blocks' fp32
// training; rows_tf32.cu's f32_ln_bwd reads three [M][128] tensors):
//   xn = LN(c);  dc = dc0 + sum_k LN backward of dcn_k g_k;  d g_k += sum_rows dcn_k xn   (PARTS [grid][256] = [d g1 | d g2], a partial row per
//   CTA, summed by the host in order: deterministic). DC [M][128] is a fresh output.
extern "C" __global__ void __launch_bounds__(256)
f32_ln_bwd2(const float* __restrict__ c, const float* __restrict__ D, const float* __restrict__ g1, const float* __restrict__ g2,
            float* __restrict__ DC, float* __restrict__ PARTS, int M, float eps) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, ch = 4 * lane;
  __shared__ float red[8][32][8];
  float g[2][4], acc[8];
  ld4(g1 + ch, g[0]);
  ld4(g2 + ch, g[1]);
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int r = 8 * (int)blockIdx.x + warp; r < M; r += 8 * (int)gridDim.x) {
    float x[4], o[4];
    ld4(c + (size_t)r * C + ch, x);
    ld4(D + (size_t)r * 384 + 256 + ch, o);
    const float rstd = norm128(x, eps);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      float d[4], dl[4], b[4];
      ld4(D + (size_t)r * 384 + 128 * k + ch, d);
#pragma unroll
      for (int j = 0; j < 4; ++j) { acc[4 * k + j] += d[j] * x[j]; dl[j] = d[j] * g[k][j]; }
      ln_back(x, dl, rstd, b);
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] += b[j];
    }
    st4(DC + (size_t)r * C + ch, o);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) red[warp][lane][e] = acc[e];
  __syncthreads();
  {
    const int k = tid >> 7, cc = tid & 127;                                // 256 threads: [d g1 | d g2]
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w][cc >> 2][4 * k + (cc & 3)];
    PARTS[(size_t)blockIdx.x * 256 + tid] = v;
  }
}

// f32_swiglu_up_bwd: Boltz-2's up-gated SwiGLU backward in one launch (h = silu(a) b up; rows_tf32.cu's f32_swiglu_bwd between three
// elementwise products otherwise): dh [M][256], u [M][512] = [a_0 | b_0 | a_1 | b_1], up [M][256] ->
//   DABU [M][768] = [d a_0 | d b_0 | d a_1 | d b_1 | d up] (ONE operand of the dx2 GEMM against [Wu^T | Wup^T], and the weight gradients'),
//   HH [M][256] = silu(a) b up (the dWs operand).
DEVI float rcpf_(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
extern "C" __global__ void __launch_bounds__(256)
f32_swiglu_up_bwd(const float* __restrict__ dh, const float* __restrict__ u, const float* __restrict__ up, float* __restrict__ dabu,
                  float* __restrict__ hh, int M) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const int ch = 4 * lane;
#pragma unroll
  for (int hb = 0; hb < 2; ++hb) {
    float av[4], bv[4], g[4], uv[4], da[4], db[4], du[4], h[4];
    ld4(u + (size_t)row * 4 * C + 2 * C * hb + ch, av);
    ld4(u + (size_t)row * 4 * C + 2 * C * hb + C + ch, bv);
    ld4(dh + (size_t)row * 2 * C + C * hb + ch, g);
    ld4(up + (size_t)row * 2 * C + C * hb + ch, uv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float s = rcpf_(1.f + __expf(-av[i])), silu = av[i] * s, h0 = silu * bv[i], g0 = g[i] * uv[i];
      h[i] = h0 * uv[i];
      du[i] = g[i] * h0;
      db[i] = g0 * silu;
      da[i] = g0 * bv[i] * s * (1.f + av[i] * (1.f - s));
    }
    st4(dabu + (size_t)row * 6 * C + 2 * C * hb + ch, da);
    st4(dabu + (size_t)row * 6 * C + 2 * C * hb + C + ch, db);
    st4(dabu + (size_t)row * 6 * C + 4 * C + C * hb + ch, du);
    st4(hh + (size_t)row * 2 * C + C * hb + ch, h);
  }
}
