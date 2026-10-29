// aa_rows.cu — the row passes of AugmentedAttentionPairBias's B200 module route (``module_route.py``): everything between the
// cuBLAS GEMMs and the tcgen05 attention cores that the module ran as PyTorch statements (sigmoid gates, products, residual adds,
// casts, the gradient branches' adds), one pass each. T: the compute dtype (bf16 or fp32 = TF32 path); S: the residual stream's.
//
//   aam_gate_o        og = sigmoid(g) * O                                           (g: the gate columns of the q|k|v|g GEMM output)
//   aam_gate_bwd      dO = dog sigmoid(g) (bf16, or fp32 on the TF32 grid), D = rowsum_head(dO O), dg = dog O s (1 - s) written into
//                     the gate columns of the q|k|v|g gradient
//   aam_finish        dq = the core's key-chunk dQ partials summed in chunk order, written into the q columns of the q|k|v|g gradient
//                     (bf16: dk / dv converted into their columns too), and per-block column sums of dq (the query bias gradient)
//   aam_res_gate      out = res + sigmoid(sp) * y      (the output gate and the residual; y may carry padded rows, see ``Lp``)
//   aam_res_gate_bwd  dy2 = dout sigmoid(sp) (zero on padded rows), dsp = dout y s (1 - s), per-block column sums of dsp
//   aam_bias_fix      the core's bias [S, H, Lp, Lp] from the pair pass's [H, L, L]: + extra [L, L], the per-sample key mask (S > 1),
//                     padded keys masked, padded query rows zero
//   aam_dbias_fix     its backward: dbias [H, L, L] = the planes summed over S in order, d extra [L, L] = the heads summed in order
//   aam_round_qkv     q | k | v columns of the q|k|v|g GEMM output rounded to the TF32 grid in place (to nearest, as cuBLAS takes fp32
//                     operands; the tensor core alone would truncate)
//
// Every column sum is a fixed-order per-block partial (grid a function of the shape alone) summed in block order by the last block
// (``colsum_tail``): the gradients are bitwise repeatable. Rows / columns: W (= D) a multiple of 8 (eight columns per thread), every row
// 16-byte aligned. Padded rows: the attention operands live as Ap samples of
// Lp rows (Lp a multiple of 128 for the training cores); the module's own rows are A samples of L rows.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float tf32r(float x) {
  uint32_t u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  return __uint_as_float(u);
}
template <typename T> __device__ __forceinline__ float4 ld4(const T* p);
template <> __device__ __forceinline__ float4 ld4<float>(const float* p) { return *reinterpret_cast<const float4*>(p); }
template <> __device__ __forceinline__ float4 ld4<bf16>(const bf16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
template <typename T> __device__ __forceinline__ void st4(T* p, float4 v);
template <> __device__ __forceinline__ void st4<float>(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
template <> __device__ __forceinline__ void st4<bf16>(bf16* p, float4 v) {
  __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&a);
  u.y = *reinterpret_cast<uint32_t*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}
template <typename T> __device__ __forceinline__ float ld1(const T* p);
template <> __device__ __forceinline__ float ld1<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld1<bf16>(const bf16* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ void st1(T* p, float v);
template <> __device__ __forceinline__ void st1<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void st1<bf16>(bf16* p, float v) { *p = __float2bfloat16_rn(v); }

// eight consecutive columns per thread (16 B of bf16, 32 B of fp32): every width here (W, D, a head's DH) is a multiple of 8
template <typename T> __device__ __forceinline__ void ld8(const T* p, float* v);
template <> __device__ __forceinline__ void ld8<float>(const float* p, float* v) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
template <> __device__ __forceinline__ void ld8<bf16>(const bf16* p, float* v) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[k]));
    v[2 * k] = f.x; v[2 * k + 1] = f.y;
  }
}
template <typename T> __device__ __forceinline__ void st8(T* p, const float* v);
template <> __device__ __forceinline__ void st8<float>(float* p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
template <> __device__ __forceinline__ void st8<bf16>(bf16* p, const float* v) {
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_bfloat162 b = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
    w[k] = *reinterpret_cast<uint32_t*>(&b);
  }
  *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

// Row-block kernels: block (C / 8, RB) -- thread (tx, ty) takes columns 8 tx .. 8 tx + 7 of rows ty + RB (blockIdx.x + k gridDim.x).
// A column sum leaves as one fixed-order partial row per block (part[blockIdx.x, C]); the last block to finish (a counter, reset by
// it) sums the partial rows in block order into ``sum`` [C]: one launch, bitwise repeatable.
constexpr int RB = 8;
__device__ __forceinline__ void colsum_tail(float* acc8, float* __restrict__ part, float* __restrict__ sum, unsigned* __restrict__ counter, int C) {
  __shared__ float red[RB][1024 + 8];
  __shared__ bool last;
  const int tx = threadIdx.x, ty = threadIdx.y, c = 8 * tx;
#pragma unroll
  for (int k = 0; k < 8; ++k) red[ty][c + k] = acc8[k];
  __syncthreads();
  if (ty == 0) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = red[0][c + k];
    for (int y = 1; y < RB; ++y)
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] += red[y][c + k];
    st8<float>(part + (long)blockIdx.x * C + c, v);
  }
  __threadfence();
  __syncthreads();
  if (tx == 0 && ty == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  // the last block: thread row ty sums the partial rows ty, ty + RB, ... (L2 loads: the other blocks' rows never passed through this
  // SM's L1), then thread row 0 adds the RB sums in order
  float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int b = ty; b < (int)gridDim.x; b += RB) {
    const float4 u0 = __ldcg(reinterpret_cast<const float4*>(part + (long)b * C + c)), u1 = __ldcg(reinterpret_cast<const float4*>(part + (long)b * C + c + 4));
    v[0] += u0.x; v[1] += u0.y; v[2] += u0.z; v[3] += u0.w; v[4] += u1.x; v[5] += u1.y; v[6] += u1.z; v[7] += u1.w;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 8; ++k) red[ty][c + k] = v[k];
  __syncthreads();
  if (ty == 0) {
    for (int y = 1; y < RB; ++y)
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] += red[y][c + k];
    st8<float>(sum + c, v);
  }
  if (tx == 0 && ty == 0) *counter = 0u;
}

// ------------------------------------------------------------------------------------------------ og = sigmoid(g) O
// n8 groups of 8 of the [M, W] rows; g has row stride ldg (a column view of the q|k|v|g output)
template <typename T>
__device__ __forceinline__ void gate_o(const float* __restrict__ O, const T* __restrict__ g, int ldg, T* __restrict__ og, int n8, int W) {
  const int w8 = W >> 3;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const long r = i / w8;
    const int c = (int)(i - r * w8) << 3;
    float o[8], gv[8];
    ld8<float>(O + r * W + c, o);
    ld8<T>(g + r * ldg + c, gv);
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] *= sigm(gv[k]);
    st8<T>(og + r * W + c, o);
  }
}
extern "C" __global__ void __launch_bounds__(256) aam_gate_o_bf16(const float* O, const bf16* g, int ldg, bf16* og, int n8, int W) {
  gate_o<bf16>(O, g, ldg, og, n8, W);
}
extern "C" __global__ void __launch_bounds__(256) aam_gate_o_f32(const float* O, const float* g, int ldg, float* og, int n8, int W) {
  gate_o<float>(O, g, ldg, og, n8, W);
}

// ------------------------------------------------------------------------------------------------ the gate's backward + the core's dO prep
// row blocks over M rows of W = H DH; rows are Lp per sample (dd [rows / Lp, H, Lp]). dO = dog s (bf16, or fp32 on the TF32 grid),
// D = rowsum_head(dO O), dg = dog O s (1 - s) into the gate columns of the q|k|v|g gradient (row stride lddg)
template <typename T, bool R32>
__device__ __forceinline__ void gate_bwd(const T* __restrict__ dog, const float* __restrict__ O, const T* __restrict__ g, int ldg,
                                         void* __restrict__ dob, float* __restrict__ dd, T* __restrict__ dg, int lddg, int M, int Lp, int H,
                                         int DH) {
  __shared__ float red[RB][128 + 1];
  const int tx = threadIdx.x, ty = threadIdx.y, c = 8 * tx, W = H * DH, per = DH / 8;
  for (int r0 = blockIdx.x * RB; r0 < M; r0 += gridDim.x * RB) {
    const int r = r0 + ty;
    float s = 0.f;
    if (r < M) {
      float x[8], o[8], gv[8], d[8], e[8];
      ld8<T>(dog + (long)r * W + c, x);
      ld8<float>(O + (long)r * W + c, o);
      ld8<T>(g + (long)r * ldg + c, gv);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float sg = sigm(gv[k]);
        d[k] = x[k] * sg;
        e[k] = x[k] * o[k] * sg * (1.f - sg);
        s += d[k] * o[k];
        if (R32) d[k] = tf32r(d[k]);
      }
      st8<T>(dg + (long)r * lddg + c, e);
      if (R32) st8<float>(reinterpret_cast<float*>(dob) + (long)r * W + c, d);
      else st8<bf16>(reinterpret_cast<bf16*>(dob) + (long)r * W + c, d);
    }
    red[ty][tx] = s;
    __syncthreads();
    if (tx < H && r < M) {
      float t = 0.f;
      for (int k = 0; k < per; ++k) t += red[ty][tx * per + k];
      dd[((long)(r / Lp) * H + tx) * Lp + r % Lp] = t;
    }
    __syncthreads();
  }
}
extern "C" __global__ void __launch_bounds__(1024) aam_gate_bwd_bf16(const bf16* dog, const float* O, const bf16* g, int ldg, bf16* dob, float* dd,
                                                                     bf16* dg, int lddg, int M, int Lp, int H, int DH) {
  gate_bwd<bf16, false>(dog, O, g, ldg, dob, dd, dg, lddg, M, Lp, H, DH);
}
extern "C" __global__ void __launch_bounds__(1024) aam_gate_bwd_f32(const float* dog, const float* O, const float* g, int ldg, float* dob, float* dd,
                                                                    float* dg, int lddg, int M, int Lp, int H, int DH) {
  gate_bwd<float, true>(dog, O, g, ldg, dob, dd, dg, lddg, M, Lp, H, DH);
}

// ------------------------------------------------------------------------------------------------ dq (+ dk, dv) into the q|k|v|g gradient
// row blocks over M rows of W: dq = the np partials summed in order (stride pstride), written at dst (row stride ld); bf16: dk / dv
// converted into the next two column blocks; dbq = the column sums of dq (fp32, before rounding)
template <typename T>
__device__ __forceinline__ void finish(const float* __restrict__ DQ, int np, int pstride, const float* __restrict__ DK, const float* __restrict__ DV,
                                       T* __restrict__ dst, int ld, int M, int W, float* __restrict__ part, float* __restrict__ sum,
                                       unsigned* __restrict__ counter) {
  const int c = 8 * threadIdx.x;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int r = blockIdx.x * RB + threadIdx.y; r < M; r += gridDim.x * RB) {
    const long o = (long)r * W + c;
    float a[8];
    ld8<float>(DQ + o, a);
#pragma unroll 4
    for (int p = 1; p < np; ++p) {
      float b[8];
      ld8<float>(DQ + (long)p * pstride + o, b);
#pragma unroll
      for (int k = 0; k < 8; ++k) a[k] += b[k];
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += a[k];
    T* d = dst + (long)r * ld + c;
    st8<T>(d, a);
    if (DK) {
      ld8<float>(DK + o, a);
      st8<T>(d + W, a);
      ld8<float>(DV + o, a);
      st8<T>(d + 2 * W, a);
    }
  }
  colsum_tail(acc, part, sum, counter, W);
}
extern "C" __global__ void __launch_bounds__(1024) aam_finish_bf16(const float* DQ, int np, int pstride, const float* DK, const float* DV, bf16* dst,
                                                                   int ld, int M, int W, float* part, float* sum, unsigned* counter) {
  finish<bf16>(DQ, np, pstride, DK, DV, dst, ld, M, W, part, sum, counter);
}
extern "C" __global__ void __launch_bounds__(1024) aam_finish_f32(const float* DQ, int np, int pstride, float* dst, int ld, int M, int W, float* part,
                                                                  float* sum, unsigned* counter) {
  finish<float>(DQ, np, pstride, nullptr, nullptr, dst, ld, M, W, part, sum, counter);
}

// ------------------------------------------------------------------------------------------------ out = res + sigmoid(sp) y
// out / res / sp: the module's M = A L rows of D; y: Ap Lp rows (row a L + i of the output reads y's row a Lp + i)
template <typename T, typename S>
__device__ __forceinline__ void res_gate(const S* __restrict__ res, const T* __restrict__ sp, const T* __restrict__ y, S* __restrict__ out, int n8,
                                         int D, int L, int Lp) {
  const int d8 = D >> 3;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const long r = i / d8;
    const int c = (int)(i - r * d8) << 3;
    const long ry = (L == Lp) ? r : (r / L) * Lp + r % L;
    float s[8], v[8];
    ld8<T>(sp + r * D + c, s);
    ld8<T>(y + ry * D + c, v);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] *= sigm(s[k]);
    if (res) {
      float x[8];
      ld8<S>(res + r * D + c, x);
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] += x[k];
    }
    st8<S>(out + r * D + c, v);
  }
}
#define RES_GATE(NAME, T, S)                                                                                                               \
  extern "C" __global__ void __launch_bounds__(256) NAME(const S* res, const T* sp, const T* y, S* out, int n8, int D, int L, int Lp) { \
    res_gate<T, S>(res, sp, y, out, n8, D, L, Lp);                                                                                         \
  }
RES_GATE(aam_res_gate_bb, bf16, bf16)
RES_GATE(aam_res_gate_bf, bf16, float)
RES_GATE(aam_res_gate_ff, float, float)

// row blocks over the Mp rows of y: dy2 [Mp, D] (zero on padded rows), dsp [M, D]; dbsc = the column sums of dsp (fp32, before rounding)
template <typename T, typename S>
__device__ __forceinline__ void res_gate_bwd(const S* __restrict__ dout, const T* __restrict__ sp, const T* __restrict__ y, T* __restrict__ dy2,
                                             T* __restrict__ dsp, float* __restrict__ part, float* __restrict__ sum, unsigned* __restrict__ counter,
                                             int Mp, int D, int L, int Lp, int A) {
  const int c = 8 * threadIdx.x;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int ry = blockIdx.x * RB + threadIdx.y; ry < Mp; ry += gridDim.x * RB) {
    const int a = ry / Lp, i = ry - a * Lp;
    float d[8];
    if (a >= A || i >= L) {
#pragma unroll
      for (int k = 0; k < 8; ++k) d[k] = 0.f;
      st8<T>(dy2 + (long)ry * D + c, d);
      continue;
    }
    const long r = (long)a * L + i;
    float g[8], s[8], v[8];
    ld8<S>(dout + r * D + c, g);
    ld8<T>(sp + r * D + c, s);
    ld8<T>(y + (long)ry * D + c, v);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float sg = sigm(s[k]);
      d[k] = g[k] * sg;
      v[k] = g[k] * v[k] * sg * (1.f - sg);
      acc[k] += v[k];
    }
    st8<T>(dy2 + (long)ry * D + c, d);
    st8<T>(dsp + r * D + c, v);
  }
  colsum_tail(acc, part, sum, counter, D);
}
#define RES_GATE_BWD(NAME, T, S)                                                                                                         \
  extern "C" __global__ void __launch_bounds__(1024) NAME(const S* dout, const T* sp, const T* y, T* dy2, T* dsp, float* part, float* sum, \
                                                          unsigned* counter, int Mp, int D, int L, int Lp, int A) {                      \
    res_gate_bwd<T, S>(dout, sp, y, dy2, dsp, part, sum, counter, Mp, D, L, Lp, A);                                                       \
  }
RES_GATE_BWD(aam_res_gate_bwd_bb, bf16, bf16)
RES_GATE_BWD(aam_res_gate_bwd_bf, bf16, float)
RES_GATE_BWD(aam_res_gate_bwd_ff, float, float)

// ------------------------------------------------------------------------------------------------ the core's bias from the pair pass's
// out[s, h, i, j] (S planes of H heads, Lp x Lp) from bias [H, Lb, Lb] (Lb >= L: the pair pass's length, the pair padded to it):
// i, j < L: bias[h, i, j] + xs extra[i, j] (extra may be null; bf16 or fp32: ef32; xs: its units), neg where the sample's key mask
// (pm [S, L], may be null) is False; j >= L: neg; i >= L > j: 0. One thread per (s, h, i, j); out may be bias (S = 1, Lp = Lb: in place).
// One block per output row (s, h, i) (grid-stride), threads along j.
template <typename T>
__device__ __forceinline__ void bias_fix(const T* bias, const void* __restrict__ extra, int ef32, float xs, const bool* __restrict__ pm, T* out, int S,
                                         int H, int L, int Lb, int Lp, float neg) {
  const int rows = S * H * Lp;
  for (int q = blockIdx.x; q < rows; q += gridDim.x) {
    const int i = q % Lp, sh = q / Lp, h = sh % H, s = sh / H;
    T* o = out + (long)q * Lp;
    const T* b = bias + ((long)h * Lb + i) * Lb;
    const bool* m = pm ? pm + (long)s * L : nullptr;
    for (int j = threadIdx.x; j < Lp; j += blockDim.x) {
      float v;
      if (j >= L) v = neg;
      else if (i >= L) v = 0.f;
      else {
        v = ld1<T>(b + j);
        if (extra)
          v += xs * (ef32 ? reinterpret_cast<const float*>(extra)[(long)i * L + j] : __bfloat162float(reinterpret_cast<const bf16*>(extra)[(long)i * L + j]));
        if (m && !m[j]) v = neg;
      }
      st1<T>(o + j, v);
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) aam_bias_fix_bf16(const bf16* bias, const void* extra, int ef32, float xs, const bool* pm, bf16* out,
                                                                    int S, int H, int L, int Lb, int Lp, float neg) {
  bias_fix<bf16>(bias, extra, ef32, xs, pm, out, S, H, L, Lb, Lp, neg);
}
extern "C" __global__ void __launch_bounds__(256) aam_bias_fix_f32(const float* bias, const void* extra, int ef32, float xs, const bool* pm, float* out,
                                                                   int S, int H, int L, int Lb, int Lp, float neg) {
  bias_fix<float>(bias, extra, ef32, xs, pm, out, S, H, L, Lb, Lp, neg);
}

// dpb[h, i, j] = sum_s DB[s, h, i, j] for i, j < L (s in order), 0 for the rest of its [H, Lb, Lb]; dx[i, j] = sum_h dpb[h, i, j] (h in
// order). DB: [S, H, Lp, Lp]. One block per row i of Lb (grid-stride), threads along j. dpb may be DB itself (S = 1, Lp = Lb: then it is
// not written); dx may be null.
extern "C" __global__ void __launch_bounds__(256) aam_dbias_fix(const float* DB, float* dpb, float* dx, int S, int H, int L, int Lb, int Lp) {
  const bool wr = dpb != DB;
  for (int i = blockIdx.x; i < Lb; i += gridDim.x) {
    for (int j = threadIdx.x; j < Lb; j += blockDim.x) {
      if (i >= L || j >= L) {
        if (wr)
          for (int h = 0; h < H; ++h) dpb[((long)h * Lb + i) * Lb + j] = 0.f;
        continue;
      }
      float x = 0.f;
      for (int h = 0; h < H; ++h) {
        float v = 0.f;
        for (int s = 0; s < S; ++s) v += DB[(((long)s * H + h) * Lp + i) * Lp + j];
        if (wr) dpb[((long)h * Lb + i) * Lb + j] = v;
        x += v;
      }
      if (dx) dx[(long)i * L + j] = x;
    }
  }
}

// ------------------------------------------------------------------------------------------------ q | k | v on the TF32 grid, in place
extern "C" __global__ void __launch_bounds__(256) aam_round_qkv(float* qkvg, int ld, int n4, int W3) {
  const int w4 = W3 >> 2;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    const long r = i / w4;
    float* p = qkvg + r * ld + ((int)(i - r * w4) << 2);
    float4 v = *reinterpret_cast<float4*>(p);
    *reinterpret_cast<float4*>(p) = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
  }
}

// ------------------------------------------------------------------------------------------------ the per-sample key mask as bits
// out [Ap, KW] words: bit j of word n of sample a = m[a, 32 n + j] for a < A and 32 n + j < L, 1 for a pad sample's keys < L, 0 past L
extern "C" __global__ void __launch_bounds__(256) aam_pack_kmask(const bool* __restrict__ m, uint32_t* __restrict__ out, int A, int L, int Ap, int KW) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= Ap * KW) return;
  const int a = t / KW, n = t - a * KW;
  uint32_t w = 0u;
  for (int j = 0; j < 32; ++j) {
    const int key = 32 * n + j;
    if (key < L && (a >= A || m[(long)a * L + key])) w |= 1u << j;
  }
  out[t] = w;
}

// ------------------------------------------------------------------------------------------------ QK-norm (per-head RMSNorm of q and k)
// Row blocks over M rows of the q|k|v|g output (row stride ld): qn = q rstd_q wq, kn = k rstd_k wk per head (rstd = 1 / sqrt(mean(x^2) +
// eps)), times ``qs`` for q. Out: qn / kn [M, W] (row stride ldo; may be the q / k columns themselves: in place), the rstd [M, 2 H] fp32
// (null: not kept), R32: rounded to the TF32 grid.
template <typename T, bool R32>
__device__ __forceinline__ void qknorm(const T* __restrict__ qkvg, int ld, const float* __restrict__ wq, const float* __restrict__ wk, T* qn, T* kn,
                                       int ldo, float* __restrict__ rstd, int M, int H, int DH, float eq, float ek, float qs) {
  __shared__ float red[RB][128 + 1];
  const int tx = threadIdx.x, ty = threadIdx.y, c = 8 * tx, W = H * DH, per = DH / 8, h = c / DH, ch = c - h * DH;
  for (int r0 = blockIdx.x * RB; r0 < M; r0 += gridDim.x * RB) {
    const int r = r0 + ty;
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      float x[8];
      float ss = 0.f;
      if (r < M) {
        ld8<T>(qkvg + (long)r * ld + which * W + c, x);
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += x[k] * x[k];
      }
      red[ty][tx] = ss;
      __syncthreads();
      float t = 0.f;
      for (int k = 0; k < per; ++k) t += red[ty][h * per + k];
      const float rs = rsqrtf(t / (float)DH + (which ? ek : eq));
      if (r < M) {
        const float* w = which ? wk : wq;
        const float sc = which ? 1.f : qs;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          x[k] = x[k] * rs * w[ch + k] * sc;
          if (R32) x[k] = tf32r(x[k]);
        }
        st8<T>((which ? kn : qn) + (long)r * ldo + c, x);
        if (rstd && ch == 0) rstd[(long)r * 2 * H + which * H + h] = rs;
      }
      __syncthreads();
    }
  }
}
extern "C" __global__ void __launch_bounds__(1024) aam_qknorm_bf16(const bf16* qkvg, int ld, const float* wq, const float* wk, bf16* qn, bf16* kn, int ldo,
                                                                   float* rstd, int M, int H, int DH, float eq, float ek, float qs) {
  qknorm<bf16, false>(qkvg, ld, wq, wk, qn, kn, ldo, rstd, M, H, DH, eq, ek, qs);
}
extern "C" __global__ void __launch_bounds__(1024) aam_qknorm_f32(const float* qkvg, int ld, const float* wq, const float* wk, float* qn, float* kn, int ldo,
                                                                  float* rstd, int M, int H, int DH, float eq, float ek, float qs) {
  qknorm<float, true>(qkvg, ld, wq, wk, qn, kn, ldo, rstd, M, H, DH, eq, ek, qs);
}

// The backward: dq / dk of the normalized q / k (``d``: the q | k columns of the q|k|v|g gradient, row stride ld) rewritten in place as the
// gradients of the raw q / k (``qkvg``, row stride ld): with xh = x rstd, g = w dy: dx = rstd (g - xh mean(g xh)). part [blocks, 3 W]: per
// block column sums of dq (the query bias), dy xh (q: the weight's per column partials), dy xh (k), in fixed order.
template <typename T>
__device__ __forceinline__ void qknorm_bwd(const T* __restrict__ qkvg, T* d, int ld, const float* __restrict__ wq, const float* __restrict__ wk,
                                           const float* __restrict__ rstd, float* __restrict__ part, int M, int H, int DH) {
  __shared__ float red[RB][128 + 1];
  const int tx = threadIdx.x, ty = threadIdx.y, c = 8 * tx, W = H * DH, per = DH / 8, h = c / DH, ch = c - h * DH;
  float abq[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, aw[2][8] = {{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};
  for (int r0 = blockIdx.x * RB; r0 < M; r0 += gridDim.x * RB) {
    const int r = r0 + ty;
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      float x[8], dy[8], g[8];
      float s = 0.f, rs = 0.f;
      const float* w = which ? wk : wq;
      if (r < M) {
        rs = rstd[(long)r * 2 * H + which * H + h];
        ld8<T>(qkvg + (long)r * ld + which * W + c, x);
        ld8<T>(d + (long)r * ld + which * W + c, dy);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          x[k] *= rs;                                                    // xh
          g[k] = w[ch + k] * dy[k];
          s += g[k] * x[k];
          aw[which][k] += dy[k] * x[k];
        }
      }
      red[ty][tx] = s;
      __syncthreads();
      float t = 0.f;
      for (int k = 0; k < per; ++k) t += red[ty][h * per + k];
      t /= (float)DH;
      if (r < M) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          g[k] = rs * (g[k] - x[k] * t);
          if (!which) abq[k] += g[k];
        }
        st8<T>(d + (long)r * ld + which * W + c, g);
      }
      __syncthreads();
    }
  }
  // the block's sums over its RB row slots, one vector at a time through shared memory: thread row 0 writes them in slot order
  __shared__ float acc[RB][1024 + 8];
#pragma unroll
  for (int v3 = 0; v3 < 3; ++v3) {
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[ty][c + k] = v3 == 0 ? abq[k] : v3 == 1 ? aw[0][k] : aw[1][k];
    __syncthreads();
    if (ty == 0) {
      float v[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] = acc[0][c + k];
      for (int y = 1; y < RB; ++y)
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] += acc[y][c + k];
      st8<float>(part + (long)blockIdx.x * 3 * W + v3 * W + c, v);
    }
    __syncthreads();
  }
}
extern "C" __global__ void __launch_bounds__(1024) aam_qknorm_bwd_bf16(const bf16* qkvg, bf16* d, int ld, const float* wq, const float* wk,
                                                                       const float* rstd, float* part, int M, int H, int DH) {
  qknorm_bwd<bf16>(qkvg, d, ld, wq, wk, rstd, part, M, H, DH);
}
extern "C" __global__ void __launch_bounds__(1024) aam_qknorm_bwd_f32(const float* qkvg, float* d, int ld, const float* wq, const float* wk,
                                                                      const float* rstd, float* part, int M, int H, int DH) {
  qknorm_bwd<float>(qkvg, d, ld, wq, wk, rstd, part, M, H, DH);
}
