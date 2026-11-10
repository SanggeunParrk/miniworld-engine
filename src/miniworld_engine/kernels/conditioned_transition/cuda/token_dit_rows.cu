// Token DiT row kernels (CUDA): the passes between the GEMMs of the fused token DiT step.
//
//   adaln_rows          out = LN(x) * sigmoid(ms[tok]) + mb[tok]                        (x fp32, no affine)
//   resgate_adaln_rows  x += sigmoid(gl[tok]) * y  (fp32, in place); then optionally out = LN(x) * sigmoid(ms) + mb
//   gate_rows           out = o * sigmoid(g)
//   swiglu_rows         out = silu(a) * b,  ab = [a | b]
//   layernorm128_rows   out = LN(z), C = 128 or 256, no affine (the pair rows of the hoisted pair bias; cuBLAS projects them)
//   qknorm_rows         q, k (columns 0-767, 768-1535 of a row) <- RMSNorm per 48-wide head * weight, in place (QK-norm)
//   adaln_in_rows       x = xin (the step's input, any dtype, into the fp32 residual) and out = AdaLN(x), one pass
//   resgate_out_rows    out = x + sigmoid(gl[tok]) * y in the output dtype (the block's last residual, no fp32 write-back)
//   layernorm_rows      out = LN(z), C = 128 / 256 / 384, no affine (the conditioning rows, the DiT variants' pair rows)
//
// tok = row % L: the AdaLN / gate tables are per token, shared by the samples of a step. Every operand is read and
// written once; sums in fp32. One block per row, a thread owns 4 consecutive columns (a 16-B fp32 / 8-B bf16 vector),
// so rows need 16-B aligned fp32 / 8-B aligned bf16 starts -- the host checks the strides.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>

#include "token_dit_common.cuh"

namespace {
using namespace tdr;

// One block per row, one float4 per thread (NT = cols / 4 threads): a row of 768 is 192 threads, so even the step's
// smallest M (1920 rows at L = 384, S = 5) puts ~80 warps on every SM. One warp per row left ~13 and was latency-bound.
// xhat = LN(x) of the row, then out = xhat * sigmoid(ms) + mb; x is this thread's float4.
// x rounded to TF32 as cvt.rna.tf32.f32 does (half away from zero on the magnitude: add half an ulp of the kept bits, clear the rest;
// inf and nan keep their patterns): the operand the next TF32 GEMM would round to anyway -- so a GEMM that reads its operands
// unrounded (the TMA-fed wgmma kernels truncate) sees the bits cuBLAS's rounding would give
using rowmath::rna_tf32;

// The AdaLN scale of a token DiT's model: SIG (AF3 / Protenix / Boltz-2 / RoseTTAFold3) sigmoid(ms); else (Chai-1's identity AdaLN)
// ms itself, its "+ 1" folded into the table's bias.
template <bool SIG>
__device__ __forceinline__ float scl(float v) {
  if constexpr (SIG) return sigm(v); else return v;
}

// out = xhat * f(ms) + mb for this thread's 4 columns (xhat = (x - mean) rstd)
template <int NT, typename GT, typename OutT, bool RND = false, bool SIG = true>
__device__ __forceinline__ void adaln_out(float a, float b, float c, float d, float rstd, const GT* ms, const GT* mb, OutT* out) {
  const int col = threadIdx.x * 4;
  float4 m = V4<GT>::load(ms + col), h = V4<GT>::load(mb + col), o;
  o.x = a * rstd * scl<SIG>(m.x) + h.x;
  o.y = b * rstd * scl<SIG>(m.y) + h.y;
  o.z = c * rstd * scl<SIG>(m.z) + h.z;
  o.w = d * rstd * scl<SIG>(m.w) + h.w;
  if constexpr (RND && std::is_same<OutT, float>::value) { o.x = rna_tf32(o.x); o.y = rna_tf32(o.y); o.z = rna_tf32(o.z); o.w = rna_tf32(o.w); }
  V4<OutT>::store(out + col, o);
}

// ms2 / mb2 / out2 (TWO): a second AdaLN of the same LN(x) -- the parallel block's transition input (RoseTTAFold3, Chai-1); RND2: that
// output rounded to TF32 (the fp32 fused expand GEMM reads it)
template <int NT, typename GT, typename OutT, bool RND = false, bool SIG = true, bool TWO = false, bool RND2 = false>
__device__ __forceinline__ void adaln_store(float4 x, float eps, const GT* ms, const GT* mb, OutT* out, float* red,
                                            const GT* ms2 = nullptr, const GT* mb2 = nullptr, OutT* out2 = nullptr) {
  constexpr int D = NT * 4;
  const float mean = block_sum<NT>(x.x + x.y + x.z + x.w, red) / D;
  const float a = x.x - mean, b = x.y - mean, c = x.z - mean, d = x.w - mean;
  const float rstd = rsqrtf(block_sum<NT>(a * a + b * b + c * c + d * d, red) / D + eps);
  adaln_out<NT, GT, OutT, RND, SIG>(a, b, c, d, rstd, ms, mb, out);
  if constexpr (TWO) adaln_out<NT, GT, OutT, RND2, SIG>(a, b, c, d, rstd, ms2, mb2, out2);
}

template <int NT, typename GT, typename OutT, bool SIG = true>
__global__ void __launch_bounds__(NT) adaln_rows_kernel(const float* __restrict__ X, const GT* __restrict__ MS,
    const GT* __restrict__ MB, OutT* __restrict__ OUT, int L, long sx, long sms, long smb, float eps) {
  __shared__ float red[NT / 32];
  const long row = blockIdx.x;
  const int tok = row % L;
  float4 x = V4<float>::load(X + row * sx + threadIdx.x * 4);
  adaln_store<NT, GT, OutT, false, SIG>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NT * 4), red);
}

template <int NT, typename XT, typename GT, typename OutT, bool SIG = true, bool TWO = false, bool RND2 = false>
__global__ void __launch_bounds__(NT) adaln_in_rows_kernel(const XT* __restrict__ XIN, float* __restrict__ X,
    const GT* __restrict__ MS, const GT* __restrict__ MB, OutT* __restrict__ OUT, int L, long sxin, long sms, long smb, float eps,
    const GT* __restrict__ MS2 = nullptr, const GT* __restrict__ MB2 = nullptr, OutT* __restrict__ OUT2 = nullptr) {
  __shared__ float red[NT / 32];
  const long row = blockIdx.x;
  const int tok = row % L;
  const float4 x = V4<XT>::load(XIN + row * sxin + threadIdx.x * 4);
  V4<float>::store(X + row * (NT * 4) + threadIdx.x * 4, x);
  adaln_store<NT, GT, OutT, false, SIG, TWO, RND2>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NT * 4), red,
                                                   TWO ? MS2 + tok * sms : nullptr, TWO ? MB2 + tok * smb : nullptr,
                                                   TWO ? OUT2 + row * (NT * 4) : nullptr);
}

// out = x + sigmoid(gl) y [+ sigmoid(gl2) y2 (TWO: the parallel block's two updates in one pass)]
template <int NT, typename YT, typename GT, typename FT, bool TWO = false>
__global__ void __launch_bounds__(NT) resgate_out_rows_kernel(const float* __restrict__ X, const YT* __restrict__ Y,
    const GT* __restrict__ GL, FT* __restrict__ OUT, int L, long sx, long sy, long sgl,
    const YT* __restrict__ Y2 = nullptr, const GT* __restrict__ GL2 = nullptr, long sy2 = 0, long sgl2 = 0) {
  const long row = blockIdx.x;
  const int tok = row % L, col = threadIdx.x * 4;
  float4 x = V4<float>::load(X + row * sx + col);
  const float4 y = V4<YT>::load(Y + row * sy + col), g = V4<GT>::load(GL + tok * sgl + col);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  if constexpr (TWO) {
    const float4 y2 = V4<YT>::load(Y2 + row * sy2 + col), g2 = V4<GT>::load(GL2 + tok * sgl2 + col);
    x.x += sigm(g2.x) * y2.x; x.y += sigm(g2.y) * y2.y; x.z += sigm(g2.z) * y2.z; x.w += sigm(g2.w) * y2.w;
  }
  V4<FT>::store(OUT + row * (NT * 4) + col, x);
}

// LayerNorm of C-wide rows (C = 128, 256 or 384), no affine: one warp per row (C / 128 float4 per lane), 8 rows per block.
// The one LN-rows kernel of the row extensions (the token DiT conditioning rows, the hoisted pair rows, the DiT variants'
// pair rows). PERM (C = 128): row r lands at the position of its key (r % 8 = 4 i + j -> 2 j + i within its group of 8),
// the key order of the H100 TF32 attention core.
template <int C, typename ZT, typename OutT>
__global__ void __launch_bounds__(256) layernorm_rows_kernel(const ZT* __restrict__ Z, OutT* __restrict__ OUT, long R,
    long sz, float eps, bool perm) {
  constexpr int K = C / 128;
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 z[K];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { z[k] = V4<ZT>::load(Z + row * sz + k * 128 + lane * 4); s += z[k].x + z[k].y + z[k].z + z[k].w; }
  const float mean = warp_sum(s) / (float)C;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    z[k].x -= mean; z[k].y -= mean; z[k].z -= mean; z[k].w -= mean;
    q += z[k].x * z[k].x + z[k].y * z[k].y + z[k].z * z[k].z + z[k].w * z[k].w;
  }
  const float rstd = rsqrtf(warp_sum(q) / (float)C + eps);
  const long orow = perm ? row - (row & 7) + 2 * (row & 3) + ((row >> 2) & 1) : row;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    z[k].x *= rstd; z[k].y *= rstd; z[k].z *= rstd; z[k].w *= rstd;
    V4<OutT>::store(OUT + orow * C + k * 128 + lane * 4, z[k]);
  }
}

template <int NT, typename YT, typename GT, typename OutT, bool RND = false, bool SIG = true>
__global__ void __launch_bounds__(NT) resgate_adaln_rows_kernel(float* __restrict__ X, const YT* __restrict__ Y,
    const GT* __restrict__ GL, const GT* __restrict__ MS, const GT* __restrict__ MB, OutT* __restrict__ OUT, int L,
    long sx, long sy, long sgl, long sms, long smb, float eps, bool has_adaln) {
  __shared__ float red[NT / 32];
  const long row = blockIdx.x;
  const int tok = row % L, col = threadIdx.x * 4;
  float4 x = V4<float>::load(X + row * sx + col), y = V4<YT>::load(Y + row * sy + col), g = V4<GT>::load(GL + tok * sgl + col);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  V4<float>::store(X + row * sx + col, x);
  if (has_adaln) adaln_store<NT, GT, OutT, RND, SIG>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NT * 4), red);
}

template <typename T>
__global__ void __launch_bounds__(256) gate_rows_kernel(const T* __restrict__ O, const T* __restrict__ G,
    T* __restrict__ OUT, int D, long so, long sg) {
  const long row = blockIdx.x;
  for (int c = threadIdx.x * 4; c < D; c += blockDim.x * 4) {
    float4 o = V4<T>::load(O + row * so + c), g = V4<T>::load(G + row * sg + c);
    o.x *= sigm(g.x); o.y *= sigm(g.y); o.z *= sigm(g.z); o.w *= sigm(g.w);
    V4<T>::store(OUT + row * D + c, o);
  }
}

// out = silu(a) b [* u (UP: Boltz-2's up-gate transition, ab = [a | b | u])]
template <typename T, bool UP = false>
__global__ void __launch_bounds__(512) swiglu_rows_kernel(const T* __restrict__ AB, T* __restrict__ OUT, int N, long sab) {
  const long row = blockIdx.x;
  for (int c = threadIdx.x * 4; c < N; c += blockDim.x * 4) {
    float4 a = V4<T>::load(AB + row * sab + c), b = V4<T>::load(AB + row * sab + N + c), o;
    o.x = a.x * sigm(a.x) * b.x; o.y = a.y * sigm(a.y) * b.y; o.z = a.z * sigm(a.z) * b.z; o.w = a.w * sigm(a.w) * b.w;
    if constexpr (UP) {
      const float4 u = V4<T>::load(AB + row * sab + 2 * N + c);
      o.x *= u.x; o.y *= u.y; o.z *= u.z; o.w *= u.w;
    }
    V4<T>::store(OUT + row * N + c, o);
  }
}

// The flat q / k LayerNorm with affine over all D columns, in place (RoseTTAFold3's kq-norm): q = row[0, D), k = row[D, 2D) of every row;
// a logit scale is folded into q's weight / bias. One block of NT = D / 4 threads per row, q then k.
// RND: the outputs rounded to TF32 (cvt.rna; fp32 only): the operands the TF32 attention core would otherwise truncate
template <int NT, typename T, bool RND = false>
__global__ void __launch_bounds__(NT) qk_ln_rows_kernel(T* __restrict__ QK, long sq, const float* __restrict__ WQ,
    const float* __restrict__ BQ, const float* __restrict__ WK, const float* __restrict__ BK, float eps) {
  constexpr int D = NT * 4;
  __shared__ float red[NT / 32];
  const int col = threadIdx.x * 4;
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    T* p = QK + (long)blockIdx.x * sq + which * D + col;
    const float4 x = V4<T>::load(p);
    const float mean = block_sum<NT>(x.x + x.y + x.z + x.w, red) / D;
    const float a = x.x - mean, b = x.y - mean, c = x.z - mean, d = x.w - mean;
    const float rstd = rsqrtf(block_sum<NT>(a * a + b * b + c * c + d * d, red) / D + eps);
    const float4 w = *reinterpret_cast<const float4*>((which ? WK : WQ) + col);
    const float4 o = *reinterpret_cast<const float4*>((which ? BK : BQ) + col);
    float4 r = make_float4(a * rstd * w.x + o.x, b * rstd * w.y + o.y, c * rstd * w.z + o.z, d * rstd * w.w + o.w);
    if constexpr (RND && std::is_same<T, float>::value) { r.x = rna_tf32(r.x); r.y = rna_tf32(r.y); r.z = rna_tf32(r.z); r.w = rna_tf32(r.w); }
    V4<T>::store(p, r);
  }
}

// fp32 rows rounded to TF32 in place (cvt.rna), the first cols4 x 4 columns of each: the TF32 attention core truncates its operands
__global__ void __launch_bounds__(256) rna_rows_kernel(float* __restrict__ QK, long sq, int cols4, long n4) {
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long row = e / cols4;
    float* p = QK + row * sq + (e - row * cols4) * 4;
    float4 v = *reinterpret_cast<float4*>(p);
    v.x = rna_tf32(v.x); v.y = rna_tf32(v.y); v.z = rna_tf32(v.z); v.w = rna_tf32(v.w);
    *reinterpret_cast<float4*>(p) = v;
  }
}

// QK-norm in place: one block of D / 2 threads per row (D = the model width), the first half on q (columns 0 .. D - 1), the
// second on k (D .. 2D - 1), 4 columns each; a head is HD / 4 consecutive threads, summed through shared memory. The weight
// carries any logit scale folded into it.
template <typename T, int D, int HD>
__global__ void __launch_bounds__(D / 2) qknorm_rows_kernel(T* __restrict__ QK, long sq, const float* __restrict__ WQ,
    const float* __restrict__ WK, float eq, float ek) {
  constexpr int NQ = D / 4, HT = HD / 4;
  __shared__ float part[2 * NQ];
  const int t = threadIdx.x, which = t / NQ, c = (t % NQ) * 4;
  T* p = QK + (long)blockIdx.x * sq + which * D + c;
  float4 x = V4<T>::load(p);
  part[t] = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  __syncthreads();
  const int h0 = (t / HT) * HT;
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < HT; ++j) ss += part[h0 + j];
  const float r = rsqrtf(ss / (float)HD + (which ? ek : eq));
  const float4 w = *reinterpret_cast<const float4*>((which ? WK : WQ) + c % HD);
  x.x *= r * w.x; x.y *= r * w.y; x.z *= r * w.z; x.w *= r * w.w;
  V4<T>::store(p, x);
}

// ------------------------------------------------------------------------------------------------ host
void check_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.stride(1) == 1, name, ": a CUDA [rows, cols] view with unit column stride");
  TORCH_CHECK(t.size(1) >= cols, name, ": needs ", cols, " columns");
  const int64_t align = t.scalar_type() == at::kFloat ? 4 : 4;   // 4 elements per lane step: 16 B fp32 / 8 B bf16
  TORCH_CHECK(t.stride(0) % align == 0 && (reinterpret_cast<uintptr_t>(t.data_ptr()) % (align * t.element_size())) == 0,
              name, ": rows must start on a ", align * t.element_size(), "-byte boundary");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, name, ": fp32 or bf16");
}


#define TDR_DISPATCH(T, NAME, ...)                                                                   \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    using NAME = __nv_bfloat16; return __VA_ARGS__();                                                \
  }()

template <typename C> auto ptr(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }
// the model widths the row kernels take: 768 (192 threads, a float4 each) and 1024 (256)
#define TDR_WIDTH(D, NT, ...)                                                                        \
  [&] {                                                                                              \
    TORCH_CHECK((D) == 768 || (D) == 1024, "token DiT rows: width 768 or 1024");                     \
    if ((D) == 768) { constexpr int NT = 192; return __VA_ARGS__(); }                                \
    constexpr int NT = 256; return __VA_ARGS__();                                                    \
  }()

// The conditioning rows of a step (L of them, d_cond = DC: 384, or 768 -- ESMFold2's token DiT conditioned at its own width):
// cn = LN(c) and cc = c in the GEMM dtype. One warp per row, a lane owning DC / 32 columns -- no block barriers: at L rows the kernel
// is latency-bound, not bandwidth-bound.
template <int DC, typename CT, typename OT>
__global__ void __launch_bounds__(256) cond_rows_kernel(const CT* __restrict__ C, long sc, OT* __restrict__ CN, OT* __restrict__ CC,
    int L, float eps) {
  constexpr int NV = DC / 128;
  const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= L) return;
  float4 v[NV];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    v[i] = V4<CT>::load(C + row * sc + (i * 32 + lane) * 4);
    V4<OT>::store(CC + (long)row * DC + (i * 32 + lane) * 4, v[i]);
    s += v[i].x + v[i].y + v[i].z + v[i].w;
  }
  const float mean = warp_sum(s) / (float)DC;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    v[i].x -= mean; v[i].y -= mean; v[i].z -= mean; v[i].w -= mean;
    q += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
  }
  const float rstd = rsqrtf(warp_sum(q) / (float)DC + eps);
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int col = (i * 32 + lane) * 4;
    V4<OT>::store(CN + (long)row * DC + col, make_float4(v[i].x * rstd, v[i].y * rstd, v[i].z * rstd, v[i].w * rstd));
  }
}


// SIG: the sigmoid AdaLN scale (else the identity AdaLN's, Chai-1)
#define TDR_SIG(S, NAME, ...)                                                                        \
  [&] {                                                                                              \
    if (S) { constexpr bool NAME = true; return __VA_ARGS__(); }                                     \
    constexpr bool NAME = false; return __VA_ARGS__();                                               \
  }()

void adaln_rows(at::Tensor x, at::Tensor ms, at::Tensor mb, at::Tensor out, int64_t L, double eps, bool sig) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kFloat, "adaln_rows: fp32 x");
  for (auto* t : {&ms, &mb}) check_rows(*t, "adaln_rows ms/mb", D);
  check_rows(x, "adaln_rows x", D); check_rows(out, "adaln_rows out", D);
  TORCH_CHECK(ms.scalar_type() == mb.scalar_type() && out.is_contiguous(), "adaln_rows: ms/mb dtypes, contiguous out");
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(ms.scalar_type(), GT, [&] {
    TDR_DISPATCH(out.scalar_type(), OT, [&] {
      TDR_WIDTH(D, NT, [&] {
        TDR_SIG(sig, SG, [&] {
          adaln_rows_kernel<NT, GT, OT, SG><<<(unsigned)M, NT, 0, st>>>(ptr<float>(x), ptr<const GT>(ms), ptr<const GT>(mb),
              ptr<OT>(out), (int)L, x.stride(0), ms.stride(0), mb.stride(0), (float)eps);
        });
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void resgate_adaln_rows_impl(at::Tensor x, at::Tensor y, at::Tensor gl, c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb,
                             at::Tensor out, int64_t L, double eps, bool tf32_round, bool sig) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kFloat, "resgate_adaln_rows: fp32 x");
  const bool has = ms.has_value();
  check_rows(x, "x", D); check_rows(y, "y", D); check_rows(gl, "gl", D);
  const at::Tensor& msv = has ? *ms : gl;
  const at::Tensor& mbv = has ? *mb : gl;
  if (has) { check_rows(msv, "ms", D); check_rows(mbv, "mb", D); check_rows(out, "out", D);
             TORCH_CHECK(out.is_contiguous() && msv.scalar_type() == gl.scalar_type(), "out contiguous, ms like gl"); }
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(y.scalar_type(), YT, [&] {
    TDR_DISPATCH(gl.scalar_type(), GT, [&] {
      TDR_DISPATCH(out.scalar_type(), OT, [&] {
        TDR_WIDTH(D, NT, [&] {
          TDR_SIG(sig, SG, [&] {
            if (tf32_round)
              resgate_adaln_rows_kernel<NT, YT, GT, OT, true, SG><<<(unsigned)M, NT, 0, st>>>(ptr<float>(x), ptr<const YT>(y),
                  ptr<const GT>(gl), ptr<const GT>(msv), ptr<const GT>(mbv), ptr<OT>(out), (int)L, x.stride(0), y.stride(0),
                  gl.stride(0), msv.stride(0), mbv.stride(0), (float)eps, has);
            else
              resgate_adaln_rows_kernel<NT, YT, GT, OT, false, SG><<<(unsigned)M, NT, 0, st>>>(ptr<float>(x), ptr<const YT>(y),
                  ptr<const GT>(gl), ptr<const GT>(msv), ptr<const GT>(mbv), ptr<OT>(out), (int)L, x.stride(0), y.stride(0),
                  gl.stride(0), msv.stride(0), mbv.stride(0), (float)eps, has);
          });
        });
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void resgate_adaln_rows(at::Tensor x, at::Tensor y, at::Tensor gl, c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb,
                        at::Tensor out, int64_t L, double eps, bool sig) {
  resgate_adaln_rows_impl(x, y, gl, ms, mb, out, L, eps, false, sig);
}
void resgate_adaln_rows_r(at::Tensor x, at::Tensor y, at::Tensor gl, c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb,
                          at::Tensor out, int64_t L, double eps, bool sig) {
  resgate_adaln_rows_impl(x, y, gl, ms, mb, out, L, eps, true, sig);   // out rounded to TF32 (cvt.rna) where it is fp32
}

#define TDR_IN_ARGS ptr<const XT>(xin), ptr<float>(x), ptr<const GT>(ms), ptr<const GT>(mb), ptr<OT>(out), (int)L, xin.stride(0), \
                    ms.stride(0), mb.stride(0), (float)eps, m2, b2, o2
// ms2 / mb2 / out2: the parallel block's second AdaLN of the same LN(x) (the transition input); rnd2: out2 rounded to TF32
void adaln_in_rows(at::Tensor xin, at::Tensor x, at::Tensor ms, at::Tensor mb, at::Tensor out, int64_t L, double eps, bool sig,
                   c10::optional<at::Tensor> ms2, c10::optional<at::Tensor> mb2, c10::optional<at::Tensor> out2, bool rnd2) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous() && xin.size(0) == M, "adaln_in_rows: fp32 x [M, D]");
  check_rows(xin, "xin", D); check_rows(ms, "ms", D); check_rows(mb, "mb", D); check_rows(out, "out", D);
  TORCH_CHECK(ms.scalar_type() == mb.scalar_type() && out.is_contiguous(), "adaln_in_rows: ms/mb dtypes, contiguous out");
  const bool two = out2.has_value();
  if (two) {
    TORCH_CHECK(ms2.has_value() && mb2.has_value() && ms2->scalar_type() == ms.scalar_type() && mb2->scalar_type() == ms.scalar_type()
                && ms2->stride(0) == ms.stride(0) && mb2->stride(0) == mb.stride(0) && out2->is_contiguous()
                && out2->scalar_type() == out.scalar_type(), "adaln_in_rows: the second AdaLN's tables like the first's, contiguous out2");
    check_rows(*ms2, "ms2", D); check_rows(*mb2, "mb2", D); check_rows(*out2, "out2", D);
  }
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(xin.scalar_type(), XT, [&] {
    TDR_DISPATCH(ms.scalar_type(), GT, [&] {
      TDR_DISPATCH(out.scalar_type(), OT, [&] {
        TDR_WIDTH(D, NT, [&] {
          TDR_SIG(sig, SG, [&] {
            const GT* m2 = two ? ptr<const GT>(*ms2) : nullptr;
            const GT* b2 = two ? ptr<const GT>(*mb2) : nullptr;
            OT* o2 = two ? ptr<OT>(*out2) : nullptr;
            if (!two) adaln_in_rows_kernel<NT, XT, GT, OT, SG, false, false><<<(unsigned)M, NT, 0, st>>>(TDR_IN_ARGS);
            else if (rnd2) adaln_in_rows_kernel<NT, XT, GT, OT, SG, true, true><<<(unsigned)M, NT, 0, st>>>(TDR_IN_ARGS);
            else adaln_in_rows_kernel<NT, XT, GT, OT, SG, true, false><<<(unsigned)M, NT, 0, st>>>(TDR_IN_ARGS);
          });
        });
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y2 / gl2: the parallel block's second update (out = x + sigmoid(gl) y + sigmoid(gl2) y2)
void resgate_out_rows(at::Tensor x, at::Tensor y, at::Tensor gl, at::Tensor out, int64_t L, c10::optional<at::Tensor> y2,
                      c10::optional<at::Tensor> gl2) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kFloat && out.is_contiguous() && out.size(0) == M, "resgate_out_rows");
  check_rows(x, "x", D); check_rows(y, "y", D); check_rows(gl, "gl", D); check_rows(out, "out", D);
  const bool two = y2.has_value();
  if (two) {
    TORCH_CHECK(gl2.has_value() && y2->scalar_type() == y.scalar_type() && gl2->scalar_type() == gl.scalar_type(),
                "resgate_out_rows: y2 like y, gl2 like gl");
    check_rows(*y2, "y2", D); check_rows(*gl2, "gl2", D);
  }
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(y.scalar_type(), YT, [&] {
    TDR_DISPATCH(gl.scalar_type(), GT, [&] {
      TDR_DISPATCH(out.scalar_type(), FT, [&] {
        TDR_WIDTH(D, NT, [&] {
          if (two)
            resgate_out_rows_kernel<NT, YT, GT, FT, true><<<(unsigned)M, NT, 0, st>>>(ptr<const float>(x), ptr<const YT>(y),
                ptr<const GT>(gl), ptr<FT>(out), (int)L, x.stride(0), y.stride(0), gl.stride(0), ptr<const YT>(*y2),
                ptr<const GT>(*gl2), y2->stride(0), gl2->stride(0));
          else
            resgate_out_rows_kernel<NT, YT, GT, FT, false><<<(unsigned)M, NT, 0, st>>>(ptr<const float>(x), ptr<const YT>(y),
                ptr<const GT>(gl), ptr<FT>(out), (int)L, x.stride(0), y.stride(0), gl.stride(0));
        });
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int C>
void launch_layernorm_rows(const at::Tensor& z, const at::Tensor& out, double eps, bool perm) {
  const int64_t R = z.size(0);
  TDR_DISPATCH(z.scalar_type(), ZT, [&] {
    TDR_DISPATCH(out.scalar_type(), OT, [&] {
      layernorm_rows_kernel<C, ZT, OT><<<(unsigned)((R + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          ptr<const ZT>(z), ptr<OT>(out), R, z.stride(0), (float)eps, perm);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out = LN(z), no affine, C = 128, 256, 384 or 512 columns (fp32 or bf16 in / out; 512: IntelliFold2's diffusion pair)
void layernorm_rows(at::Tensor z, at::Tensor out, double eps) {
  const int64_t R = z.size(0), C = z.size(1);
  TORCH_CHECK((C == 128 || C == 256 || C == 384 || C == 512) && out.is_contiguous() && out.size(1) == C && out.size(0) == R,
              "layernorm_rows: [R, 128 | 256 | 384 | 512]");
  check_rows(z, "z", C);
  const at::cuda::CUDAGuard g(z.device());
  if (C == 128) launch_layernorm_rows<128>(z, out, eps, false);
  else if (C == 256) launch_layernorm_rows<256>(z, out, eps, false);
  else if (C == 384) launch_layernorm_rows<384>(z, out, eps, false);
  else launch_layernorm_rows<512>(z, out, eps, false);
}

void gate_rows(at::Tensor o, at::Tensor gt, at::Tensor out) {
  const int64_t M = o.size(0), D = o.size(1);
  TORCH_CHECK(D % 128 == 0 && o.scalar_type() == gt.scalar_type() && o.scalar_type() == out.scalar_type(), "gate_rows");
  check_rows(o, "o", D); check_rows(gt, "g", D); check_rows(out, "out", D);
  TORCH_CHECK(out.is_contiguous(), "gate_rows: contiguous out");
  const at::cuda::CUDAGuard g(o.device());
  TDR_DISPATCH(o.scalar_type(), T, [&] {
    gate_rows_kernel<T><<<(unsigned)M, (unsigned)std::min<int64_t>(256, D / 4), 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<const T>(o), ptr<const T>(gt), ptr<T>(out), (int)D, o.stride(0), gt.stride(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// up: ab = [a | b | u] (Boltz-2's up-gate transition), out = silu(a) b u
void swiglu_rows(at::Tensor ab, at::Tensor out, bool up) {
  const int64_t M = ab.size(0), N = out.size(1);
  TORCH_CHECK(N % 128 == 0 && ab.scalar_type() == out.scalar_type() && out.is_contiguous() && ab.size(1) == (up ? 3 : 2) * N,
              "swiglu_rows: ab [M, 2 N] ([M, 3 N] up-gated), out [M, N]");
  check_rows(ab, "ab", (up ? 3 : 2) * N);
  const at::cuda::CUDAGuard g(ab.device());
  TDR_DISPATCH(ab.scalar_type(), T, [&] {
    cudaKernel_t kernel; C10_CUDA_CHECK(cudaGetKernel(&kernel, up ? swiglu_rows_kernel<T, true> : swiglu_rows_kernel<T, false>));
    const T* src = ptr<const T>(ab); T* dst = ptr<T>(out);
    int cols = (int)N; long stride = ab.stride(0);
    void* args[] = {&src, &dst, &cols, &stride};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3((unsigned)M),
        dim3((unsigned)std::min<int64_t>(512, N / 4)), args, 0, at::cuda::getCurrentCUDAStream()));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the pair rows of the hoisted bias, 128 (the name's), 256 (Protenix-v2's diffusion pair) or 512 columns (IntelliFold2's);
// ``perm``: rows written in the TF32 core's key order
void layernorm128_rows(at::Tensor z, at::Tensor out, double eps, bool perm) {
  const int64_t R = z.size(0), C = z.size(1);
  TORCH_CHECK((C == 128 || C == 256 || C == 512) && out.is_contiguous() && out.size(1) == C && out.size(0) == R,
              "layernorm128_rows: [R, 128 | 256 | 512]");
  check_rows(z, "z", C);
  const at::cuda::CUDAGuard g(z.device());
  if (C == 128) launch_layernorm_rows<128>(z, out, eps, perm);
  else if (C == 256) launch_layernorm_rows<256>(z, out, eps, perm);
  else launch_layernorm_rows<512>(z, out, eps, perm);
}

// the conditioning rows: c [L, DC], DC = 384 or 768
void cond_rows(at::Tensor c, at::Tensor cn, at::Tensor cc, double eps) {
  const int64_t L = c.size(0), DC = c.size(1);
  TORCH_CHECK(DC == 384 || DC == 768, "cond_rows: [L, 384 | 768]");
  check_rows(c, "cond_rows c", DC);
  TORCH_CHECK(cn.is_contiguous() && cc.is_contiguous() && cn.sizes() == c.sizes() && cc.sizes() == c.sizes()
              && cn.scalar_type() == cc.scalar_type(), "cond_rows: [L, 384 | 768]");
  const at::cuda::CUDAGuard g(c.device());
  TDR_DISPATCH(c.scalar_type(), CT, [&] {
    TDR_DISPATCH(cn.scalar_type(), OT, [&] {
      auto kern = DC == 384 ? cond_rows_kernel<384, CT, OT> : cond_rows_kernel<768, CT, OT>;
      kern<<<(unsigned)((L + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          ptr<const CT>(c), c.stride(0), ptr<OT>(cn), ptr<OT>(cc), (int)L, (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// d: the model width (q = columns 0 .. d - 1, k = d .. 2d - 1); the head dim is the weights' length (d / heads)
void qknorm_rows(at::Tensor qk, at::Tensor wq, at::Tensor wk, double eq, double ek, int64_t d) {
  check_rows(qk, "qk", 2 * d);
  const int64_t hd = wq.numel();
  TORCH_CHECK(wq.scalar_type() == at::kFloat && wk.scalar_type() == at::kFloat && wq.is_contiguous() && wk.is_contiguous() &&
              wk.numel() == hd, "qknorm_rows: fp32 [head dim] weights");
  TORCH_CHECK((d == 768 && (hd == 48 || hd == 32 || hd == 64)) || (d == 1024 && hd == 64), "qknorm_rows: 768 x 48 / 32 / 64, 1024 x 64");
  const at::cuda::CUDAGuard g(qk.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(qk.scalar_type(), T, [&] {
    auto go = [&](auto kern, int threads) {
      kern<<<(unsigned)qk.size(0), threads, 0, st>>>(ptr<T>(qk), qk.stride(0), ptr<const float>(wq), ptr<const float>(wk), (float)eq,
                                                     (float)ek);
    };
    if (d == 1024) go(qknorm_rows_kernel<T, 1024, 64>, 512);
    else if (hd == 48) go(qknorm_rows_kernel<T, 768, 48>, 384);
    else if (hd == 32) go(qknorm_rows_kernel<T, 768, 32>, 384);
    else go(qknorm_rows_kernel<T, 768, 64>, 384);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// qk [M, >= 2D] rows (q = columns 0 .. D - 1, k = D .. 2D - 1), fp32 [D] affine of each
void qk_ln_rows(at::Tensor qk, at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor bk, double eps, int64_t d, bool rnd) {
  check_rows(qk, "qk", 2 * d);
  for (auto* w : {&wq, &bq, &wk, &bk})
    TORCH_CHECK(w->scalar_type() == at::kFloat && w->is_contiguous() && w->numel() == d, "qk_ln_rows: fp32 [D] affine");
  const at::cuda::CUDAGuard g(qk.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(qk.scalar_type(), T, [&] {
    TDR_WIDTH(d, NT, [&] {
      auto kern = rnd ? qk_ln_rows_kernel<NT, T, true> : qk_ln_rows_kernel<NT, T, false>;
      kern<<<(unsigned)qk.size(0), NT, 0, st>>>(ptr<T>(qk), qk.stride(0), ptr<const float>(wq), ptr<const float>(bq),
                                                ptr<const float>(wk), ptr<const float>(bk), (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the first `cols` columns of every fp32 row of t rounded to TF32 in place (q | k of the qkvg rows; v^T whole)
void rna_rows(at::Tensor t, int64_t cols) {
  check_rows(t, "t", cols);
  TORCH_CHECK(t.scalar_type() == at::kFloat && cols % 4 == 0, "rna_rows: fp32 rows, a multiple of 4 columns");
  const at::cuda::CUDAGuard g(t.device());
  const long n4 = t.size(0) * (cols / 4);
  const unsigned grid = (unsigned)std::min<long>((n4 + 255) / 256, (long)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 16);
  rna_rows_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<float>(t), t.stride(0), (int)(cols / 4), n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

namespace {
// One live-weight pack for a token DiT block. Folding norm affines and logit
// scales here preserves the B200 compute boundaries without framework copies.
struct TokenPack {
  const void* src[25]; bool fp32[25];
  void* dst[13]; int size[13]; int blocks[13];
  int D, DC, H, HH; float qs; bool qk; bool out_fp32;
  int DP;   // pair width: the to_bias / ln_pair columns (128; Protenix-v2's diffusion pair, 256; IntelliFold2's, 512)
};
// the pair width of a token DiT leaf list: ln_pair.weight (leaf 12) has DP entries, to_bias.weight (leaf 13) is [heads, DP]
static int pair_width(const std::vector<at::Tensor>& weights) {
  const int DP=weights[12].numel();
  TORCH_CHECK((DP==128 || DP==256 || DP==512) && weights[13].numel()%DP==0,"token DiT pair width: 128, 256 or 512");
  return DP;
}
// the conditioning width of a token DiT leaf list (attention.ada_ln_in.ln_cond.weight, leaf 6) on the H100 (vector) pack:
// 384, or 768 (ESMFold2's token DiT, conditioned at its own width)
static int cond_width90(const std::vector<at::Tensor>& weights) {
  const int DC=weights[6].numel();
  TORCH_CHECK(DC==384 || DC==768,"Hopper token pack: cond width 384 or 768");
  return DC;
}
__device__ float pack_read(const TokenPack& p,int j,int i) {
  return p.fp32[j] ? static_cast<const float*>(p.src[j])[i]
                  : __bfloat162float(static_cast<const __nv_bfloat16*>(p.src[j])[i]);
}
template <int D>
__global__ void token_pack_kernel(TokenPack p) {
  int block=blockIdx.x, out=0;
  for(;out<13;++out) {if(block<p.blocks[out])break;block-=p.blocks[out];}
  if(out==13)return;
  const int first=(block*256+threadIdx.x)*4;
  #pragma unroll
  for(int lane=0;lane<4;++lane) {
    const int i=first+lane;if(i>=p.size[out])continue;
    // The host admits only these widths. Constant integer divisions keep
    // live master-weight staging bandwidth-bound instead of divide-bound.
    float v=0.f;constexpr int DC=384;
    if(out==0) {
      int segment=i/(D*DC),c=i%DC,j=i%(D*DC);
      int wi=segment==0?7:segment==1?9:segment==2?15:17;
      v=pack_read(p,wi,j)*pack_read(p,segment<2?6:14,c);
    } else if(out==1) {int s=i/D;if(s==0 || s==2)v=pack_read(p,s==0?8:16,i%D);}
    else if(out==2) {v=pack_read(p,i/(D*DC)==0?10:18,i%(D*DC));}
    else if(out==3) {v=pack_read(p,i/D==0?11:19,i%D);}
    else if(out==4) {v=pack_read(p,13,i)*pack_read(p,12,i%p.DP)*1.4426950408889634f;}
    else if(out==5) {
      const int s=i/(D*D);const int wi=s==0?0:s==1?2:s==2?3:4;
      v=pack_read(p,wi,i%(D*D));if(s==0 && !p.qk)v*=p.qs;
    } else if(out==6) {if(i<D)v=pack_read(p,1,i)*(p.qk?1.f:p.qs);}
    else if(out==7)v=pack_read(p,5,i);
    else if(out==8)v=pack_read(p,i/(p.HH*D)==0?20:21,i%(p.HH*D));
    else if(out==9)v=pack_read(p,22,i);
    else if(out==10)v=pack_read(p,23,i)*p.qs;
    else if(out==11)v=pack_read(p,24,i);
    else {
      const int row=i/D,col=i%D,g=row/256,r=row%256;
      v=pack_read(p,r<128?20:21,(g*128+r%128)*D+col);
    }
    if(out==10 || out==11)static_cast<float*>(p.dst[out])[i]=v;
    else static_cast<__nv_bfloat16*>(p.dst[out])[i]=__float2bfloat16_rn(v);
  }
}
std::vector<at::Tensor> token_pack(std::vector<at::Tensor> weights,int64_t heads,bool qk,double qs) {
  TORCH_CHECK(weights.size()==(qk?25:23),"token DiT leaf count");
  const at::cuda::CUDAGuard guard(weights[0].device());
  const int D=weights[0].size(1),DC=weights[6].numel(),HH=weights[20].size(0);
  const int DP=pair_width(weights);
  TORCH_CHECK(((D==768 && (heads==12 || heads==16 || heads==24)) || (D==1024 && heads==16)) && DC==384 && HH%128==0,
              "Hopper token pack requires 768 x (12, 16, 24 heads) or 1024 x 16 heads, cond 384, hidden multiple of 128");
  TokenPack p{};p.D=D;p.DC=DC;p.H=heads;p.HH=HH;p.qs=qs;p.qk=qk;p.DP=DP;
  for(int j=0;j<(int)weights.size();++j) {
    const auto& t=weights[j];
    TORCH_CHECK(t.device()==weights[0].device() && t.is_contiguous()
      && (t.scalar_type()==at::kFloat || t.scalar_type()==at::kBFloat16),"token pack leaf dtype/device/layout");
    p.src[j]=t.data_ptr();p.fp32[j]=t.scalar_type()==at::kFloat;
  }
  std::vector<std::vector<int64_t>> shapes={{4*D,DC},{4*D},{2*D,DC},{2*D},{heads,DP},{4*D,D},{4*D},{D,D},{0},{D,HH},{qk?D/heads:0},{qk?D/heads:0},{2*HH,D}};
  std::vector<at::Tensor> outputs;int blocks=0;
  for(int j=0;j<13;++j) {
    if((j==7 && !p.fp32[5]) || (j==9 && !p.fp32[22])) {
      outputs.push_back(weights[j==7?5:22]);
      continue;
    }
    outputs.push_back(at::empty(shapes[j],weights[0].options().dtype(j==10 || j==11?at::kFloat:at::kBFloat16)));
    TORCH_CHECK(outputs.back().numel()<=2147483647,"token pack requires 32-bit buffer offsets");
    p.dst[j]=outputs.back().data_ptr();p.size[j]=outputs.back().numel();p.blocks[j]=(p.size[j]+1023)/1024;blocks+=p.blocks[j];
  }
  cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,D==768?token_pack_kernel<768>:token_pack_kernel<1024>));
  void* args[]={&p};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel,dim3(blocks),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return outputs;
}

__device__ __forceinline__ float4 pack_read4(const TokenPack& p, int j, int i) {
  return p.fp32[j] ? tdr::V4<float>::load(static_cast<const float*>(p.src[j]) + i)
                  : tdr::V4<__nv_bfloat16>::load(static_cast<const __nv_bfloat16*>(p.src[j]) + i);
}
__device__ __forceinline__ float4 pack_mul4(float4 a, float4 b) {
  return make_float4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);
}
__device__ __forceinline__ float4 pack_scale4(float4 a,float b) {
  return make_float4(a.x*b,a.y*b,a.z*b,a.w*b);
}
// DC: the conditioning width (a template argument: constant divisions, as D)
template <int D, int DC>
__global__ void token_pack_vector_kernel(TokenPack p) {
  int block=blockIdx.x, out=0;
  for(;out<13;++out) {if(block<p.blocks[out])break;block-=p.blocks[out];}
  if(out==13)return;
  const int i=(block*256+threadIdx.x)*4;
  if(i>=p.size[out])return;
  float4 v=make_float4(0.f,0.f,0.f,0.f);
  if(out==0) {
    const int s=i/(D*DC),j=i%(D*DC),c=i%DC;
    const int w=s==0?7:s==1?9:s==2?15:17;
    v=pack_mul4(pack_read4(p,w,j),pack_read4(p,s<2?6:14,c));
  } else if(out==1) {
    const int s=i/D;
    if(s==0||s==2)v=pack_read4(p,s==0?8:16,i%D);
  } else if(out==2)v=pack_read4(p,i/(D*DC)==0?10:18,i%(D*DC));
  else if(out==3)v=pack_read4(p,i/D==0?11:19,i%D);
  else if(out==4)v=pack_scale4(pack_mul4(pack_read4(p,13,i),pack_read4(p,12,i%p.DP)),1.4426950408889634f);
  else if(out==5) {
    const int s=i/(D*D),w=s==0?0:s==1?2:s==2?(p.out_fp32?4:3):(p.out_fp32?3:4);
    v=pack_read4(p,w,i%(D*D));if(s==0&&!p.qk)v=pack_scale4(v,p.qs);
  } else if(out==6) {if(i<D)v=pack_scale4(pack_read4(p,1,i),p.qk?1.f:p.qs);}
  else if(out==7)v=pack_read4(p,5,i);
  else if(out==8)v=pack_read4(p,i/(p.HH*D)==0?20:21,i%(p.HH*D));
  else if(out==9)v=pack_read4(p,22,i);
  else if(out==10)v=pack_scale4(pack_read4(p,23,i),p.qs);
  else if(out==11)v=pack_read4(p,24,i);
  else {
    const int row=i/D,col=i%D,g=row/256,r=row%256;
    v=pack_read4(p,r<128?20:21,(g*128+r%128)*D+col);
  }
  if(p.out_fp32||out==10||out==11)tdr::V4<float>::store(static_cast<float*>(p.dst[out])+i,v);
  else tdr::V4<__nv_bfloat16>::store(static_cast<__nv_bfloat16*>(p.dst[out])+i,v);
}

// the vector pack kernel for (D, DC)
static cudaKernel_t token_pack_vector_fn(int D,int DC) {
  cudaKernel_t kernel;
  if(D==768)C10_CUDA_CHECK(cudaGetKernel(&kernel,DC==384?token_pack_vector_kernel<768,384>:token_pack_vector_kernel<768,768>));
  else C10_CUDA_CHECK(cudaGetKernel(&kernel,DC==384?token_pack_vector_kernel<1024,384>:token_pack_vector_kernel<1024,768>));
  return kernel;
}

std::vector<at::Tensor> token_pack_vector(std::vector<at::Tensor> weights,int64_t heads,bool qk,double qs,bool out_fp32) {
  TORCH_CHECK(weights.size()==(qk?25:23),"token DiT leaf count");
  const at::cuda::CUDAGuard guard(weights[0].device());
  const int D=weights[0].size(1),DC=cond_width90(weights),HH=weights[20].size(0);
  const int DP=pair_width(weights);
  TORCH_CHECK(((D==768 && (heads==12 || heads==16 || heads==24)) || (D==1024 && heads==16)) && HH%128==0,
              "Hopper token pack requires 768 x (12, 16, 24 heads) or 1024 x 16 heads, cond 384 / 768, hidden multiple of 128");
  TokenPack p{};p.D=D;p.DC=DC;p.H=heads;p.HH=HH;p.qs=qs;p.qk=qk;p.out_fp32=out_fp32;p.DP=DP;
  for(int j=0;j<(int)weights.size();++j) {
    const auto& t=weights[j];
    TORCH_CHECK(t.device()==weights[0].device() && t.is_contiguous()
      && (t.scalar_type()==at::kFloat || t.scalar_type()==at::kBFloat16),"token pack leaf dtype/device/layout");
    p.src[j]=t.data_ptr();p.fp32[j]=t.scalar_type()==at::kFloat;
  }
  std::vector<std::vector<int64_t>> shapes={{4*D,DC},{4*D},{2*D,DC},{2*D},{heads,DP},{4*D,D},{4*D},{D,D},{out_fp32?2*HH:0,D},{D,HH},{qk?D/heads:0},{qk?D/heads:0},{out_fp32?0:2*HH,D}};
  std::vector<at::Tensor> outputs;int blocks=0;
  for(int j=0;j<13;++j) {
    if((j==7 && p.fp32[5]==out_fp32) || (j==9 && p.fp32[22]==out_fp32)) {
      outputs.push_back(weights[j==7?5:22]);
      continue;
    }
    outputs.push_back(at::empty(shapes[j],weights[0].options().dtype(out_fp32 || j==10 || j==11?at::kFloat:at::kBFloat16)));
    TORCH_CHECK(outputs.back().numel()<=2147483647,"token pack requires 32-bit buffer offsets");
    p.dst[j]=outputs.back().data_ptr();p.size[j]=outputs.back().numel();p.blocks[j]=(p.size[j]+1023)/1024;blocks+=p.blocks[j];
  }
  cudaKernel_t kernel=token_pack_vector_fn(D,DC);
  void* args[]={&p};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel,dim3(blocks),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return outputs;
}



std::vector<at::Tensor> token_alloc90(std::vector<at::Tensor> weights,int64_t heads,bool qk,bool fp32) {
 TORCH_CHECK(weights.size()==(qk?25:23)&&(heads==12||heads==16||heads==24),"native inference pack layout");
 const at::cuda::CUDAGuard guard(weights[0].device());
 const int D=weights[0].size(1),DC=cond_width90(weights),HH=weights[20].size(0);
 TORCH_CHECK(D==768||(D==1024&&heads==16),"native inference pack width");
 const int DP=pair_width(weights);
 std::vector<std::vector<int64_t>> shapes={{4*D,DC},{4*D},{2*D,DC},{2*D},{heads,DP},{4*D,D},{4*D},{D,D},
  {fp32?2*HH:0,D},{D,HH},{qk?D/heads:0},{qk?D/heads:0},{fp32?0:2*HH,D}};
 std::vector<at::Tensor> out;
 for(int o=0;o<13;++o){
  const int a=o==7?5:o==9?22:-1;
  if(a>=0&&(weights[a].scalar_type()==at::kFloat)==fp32)out.push_back(weights[a]);
  else out.push_back(at::empty(shapes[o],weights[0].options().dtype(fp32||o==10||o==11?at::kFloat:at::kBFloat16)));
 }
 return out;
}
void token_stage90(std::vector<at::Tensor> weights,std::vector<at::Tensor> out,int64_t heads,bool qk,double qs,bool fp32,int begin,int end) {
 TORCH_CHECK(out.size()==13&&weights.size()==(qk?25:23)&&0<=begin&&begin<end&&end<=13,"native inference pack stage");
 const at::cuda::CUDAGuard guard(weights[0].device());
 TokenPack p{};p.D=weights[0].size(1);p.DC=cond_width90(weights);p.H=heads;p.HH=weights[20].size(0);p.qs=qs;p.qk=qk;p.out_fp32=fp32;p.DP=pair_width(weights);
 TORCH_CHECK(p.D==768||(p.D==1024&&heads==16),"native inference pack width");
 for(int j=0;j<(int)weights.size();++j){
  TORCH_CHECK(weights[j].device()==weights[0].device()&&weights[j].is_contiguous()
    &&(weights[j].scalar_type()==at::kFloat||weights[j].scalar_type()==at::kBFloat16),"native inference pack leaf");
  p.src[j]=weights[j].data_ptr();p.fp32[j]=weights[j].scalar_type()==at::kFloat;
 }
 int blocks=0;
 for(int o=begin;o<end;++o){
  const int a=o==7?5:o==9?22:-1;if(a>=0&&p.fp32[a]==fp32)continue;
  p.dst[o]=out[o].data_ptr();p.size[o]=out[o].numel();p.blocks[o]=(p.size[o]+1023)/1024;blocks+=p.blocks[o];
 }
 void* args[]={&p};
 C10_CUDA_CHECK(cudaLaunchKernel((void*)token_pack_vector_fn(p.D,p.DC),dim3(blocks),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- fp32 pair bias in ONE pass over the pair ------------------------------------------------------------------------------
// out[h, i L + pos(j)] = (Wf[h] . LN(z[i L + j])) for h < NH, -inf where MASK[j] is false; z [R, 128] fp32, Wf [NH, 128] fp32 (the
// LayerNorm weight and any log2 e scale folded in). One thread per pair row, a 128-row tile through shared memory (read once,
// coalesced), the 16 projections on the FMA pipe in exact fp32. ``perm``: pos(j) is the H100 TF32 attention core's key order
// (position 2 (j & 3) + (j & 7) / 4 of each group of 8). LN + a cuBLAS GEMM read the pair three times (302 MB each at fp32).
template <int NH>
__global__ void __launch_bounds__(128, 3) pair_bias_f32_k(const float* __restrict__ Z, const float* __restrict__ WF,
    float* __restrict__ OUT, const bool* __restrict__ MASK, long R, int L, float eps, bool perm) {
  constexpr int TR = 128, PITCH = 129;
  extern __shared__ float pbs[];
  float* zs = pbs;                        // [TR][PITCH]
  float* wt = pbs + TR * PITCH;           // [128][NH]
  const int tid = threadIdx.x;
  for (int i = tid; i < NH * 128; i += 128) { const int h = i / 128, c = i % 128; wt[c * NH + h] = WF[i]; }
  for (long r0 = (long)blockIdx.x * TR; r0 < R; r0 += (long)gridDim.x * TR) {
    __syncthreads();
    const int nr = (int)min((long)TR, R - r0);
#pragma unroll 4
    for (int k = 0; k < 32; ++k) {
      const int idx = k * 128 + tid, row = idx >> 5, c4 = idx & 31;
      if (row < nr) {
        const float4 v = *reinterpret_cast<const float4*>(Z + (r0 + row) * 128 + c4 * 4);
        float* d = zs + row * PITCH + c4 * 4;
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
      }
    }
    __syncthreads();
    if (tid < nr) {
      const float* zr = zs + tid * PITCH;
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int c = 0; c < 128; ++c) s4[c & 3] += zr[c];
      const float mean = ((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / 128);
      float q4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int c = 0; c < 128; ++c) { const float d = zr[c] - mean; q4[c & 3] += d * d; }
      const float rstd = rsqrtf(((q4[0] + q4[1]) + (q4[2] + q4[3])) * (1.f / 128) + eps);
      float acc[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) acc[h] = 0.f;
#pragma unroll 2
      for (int c = 0; c < 128; ++c) {
        const float xh = (zr[c] - mean) * rstd;
        const float4* w4 = reinterpret_cast<const float4*>(wt + c * NH);
#pragma unroll
        for (int q = 0; q < NH / 4; ++q) {
          const float4 w = w4[q];
          acc[4 * q + 0] = fmaf(xh, w.x, acc[4 * q + 0]); acc[4 * q + 1] = fmaf(xh, w.y, acc[4 * q + 1]);
          acc[4 * q + 2] = fmaf(xh, w.z, acc[4 * q + 2]); acc[4 * q + 3] = fmaf(xh, w.w, acc[4 * q + 3]);
        }
      }
      const long rr = r0 + tid;
      const int j = (int)(rr % L);
      const long i = rr / L;
      const int p = perm ? (j - (j & 7) + 2 * (j & 3) + ((j & 7) >> 2)) : j;
      const bool live = MASK == nullptr || MASK[j];
      const long col = i * L + p;
#pragma unroll
      for (int h = 0; h < NH; ++h) OUT[(long)h * R + col] = live ? acc[h] : -INFINITY;
    }
  }
}

void pair_bias_f32_fused(at::Tensor z, at::Tensor wf, at::Tensor out, c10::optional<at::Tensor> mask, int64_t L, double eps, bool perm) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kFloat && z.dim() == 2 && z.size(1) == 128 && z.is_contiguous(), "pair_bias_f32_fused: z [R, 128] fp32");
  TORCH_CHECK(wf.scalar_type() == at::kFloat && wf.is_contiguous() && wf.dim() == 2 && wf.size(0) == 16 && wf.size(1) == 128, "pair_bias_f32_fused: Wf [16, 128] fp32");
  const int64_t R = z.size(0);
  TORCH_CHECK(R == L * L && out.scalar_type() == at::kFloat && out.is_contiguous() && out.numel() == 16 * R, "pair_bias_f32_fused: out [16, L L] fp32");
  const bool* mp = nullptr;
  if (mask.has_value()) {
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "pair_bias_f32_fused: mask [L] bool");
    mp = mask->data_ptr<bool>();
  }
  const at::cuda::CUDAGuard g(z.device());
  const size_t smem = (size_t)(128 * 129 + 128 * 16) * 4;
  C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_f32_k<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, pair_bias_f32_k<16>, 128, smem);
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const unsigned grid = (unsigned)std::max<int64_t>(1, std::min<int64_t>((R + 127) / 128, (int64_t)std::max(per, 1) * nsm));
  pair_bias_f32_k<16><<<grid, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      z.data_ptr<float>(), wf.data_ptr<float>(), out.data_ptr<float>(), mp, R, (int)L, (float)eps, perm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("token_pack", &token_pack);
  m.def("token_pack_vector", &token_pack_vector);
  m.def("token_alloc90", &token_alloc90);
  m.def("token_stage90", &token_stage90);
  m.def("adaln_rows_cuda", &adaln_rows);
  m.def("resgate_adaln_rows_cuda", &resgate_adaln_rows);
  m.def("resgate_adaln_rows_r_cuda", &resgate_adaln_rows_r);
  m.def("gate_rows_cuda", &gate_rows);
  m.def("swiglu_rows_cuda", &swiglu_rows);
  m.def("layernorm128_rows", &layernorm128_rows);
  m.def("pair_bias_f32_fused", &pair_bias_f32_fused);
  m.def("cond_rows_cuda", &cond_rows);
  m.def("qknorm_rows_cuda", &qknorm_rows);
  m.def("qk_ln_rows_cuda", &qk_ln_rows);
  m.def("rna_rows_cuda", &rna_rows);
  m.def("adaln_in_rows_cuda", &adaln_in_rows);
  m.def("resgate_out_rows_cuda", &resgate_out_rows);
  m.def("layernorm_rows_cuda", &layernorm_rows);
}
