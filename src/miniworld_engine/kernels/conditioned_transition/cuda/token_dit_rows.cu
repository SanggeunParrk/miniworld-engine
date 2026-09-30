// Token DiT row kernels (CUDA): the passes between the GEMMs of the fused token DiT step.
//
//   adaln_rows          out = LN(x) * sigmoid(ms[tok]) + mb[tok]                        (x fp32, no affine)
//   resgate_adaln_rows  x += sigmoid(gl[tok]) * y  (fp32, in place); then optionally out = LN(x) * sigmoid(ms) + mb
//   gate_rows           out = o * sigmoid(g)
//   swiglu_rows         out = silu(a) * b,  ab = [a | b]
//   layernorm128_rows   out = LN(z), C = 128, no affine (the pair rows of the hoisted pair bias; cuBLAS projects them)
//   qknorm_rows         q, k (columns 0-767, 768-1535 of a row) <- RMSNorm per 48-wide head * weight, in place (QK-norm)
//   adaln_in_rows       x = xin (the step's input, any dtype, into the fp32 residual) and out = AdaLN(x), one pass
//   resgate_out_rows    out = x + sigmoid(gl[tok]) * y in the output dtype (the block's last residual, no fp32 write-back)
//   layernorm_rows      out = LN(z), C = 384, no affine (the conditioning rows), any in / out dtype
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
template <int NT, typename GT, typename OutT>
__device__ __forceinline__ void adaln_store(float4 x, float eps, const GT* ms, const GT* mb, OutT* out, float* red) {
  constexpr int D = NT * 4;
  const float mean = block_sum<NT>(x.x + x.y + x.z + x.w, red) / D;
  const float a = x.x - mean, b = x.y - mean, c = x.z - mean, d = x.w - mean;
  const float rstd = rsqrtf(block_sum<NT>(a * a + b * b + c * c + d * d, red) / D + eps);
  const int col = threadIdx.x * 4;
  float4 m = V4<GT>::load(ms + col), h = V4<GT>::load(mb + col), o;
  o.x = a * rstd * sigm(m.x) + h.x;
  o.y = b * rstd * sigm(m.y) + h.y;
  o.z = c * rstd * sigm(m.z) + h.z;
  o.w = d * rstd * sigm(m.w) + h.w;
  V4<OutT>::store(out + col, o);
}

template <int NT, typename GT, typename OutT>
__global__ void __launch_bounds__(NT) adaln_rows_kernel(const float* __restrict__ X, const GT* __restrict__ MS,
    const GT* __restrict__ MB, OutT* __restrict__ OUT, int L, long sx, long sms, long smb, float eps) {
  __shared__ float red[NT / 32];
  const long row = blockIdx.x;
  const int tok = row % L;
  float4 x = V4<float>::load(X + row * sx + threadIdx.x * 4);
  adaln_store<NT>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NT * 4), red);
}

template <int NT, typename XT, typename GT, typename OutT>
__global__ void __launch_bounds__(NT) adaln_in_rows_kernel(const XT* __restrict__ XIN, float* __restrict__ X,
    const GT* __restrict__ MS, const GT* __restrict__ MB, OutT* __restrict__ OUT, int L, long sxin, long sms, long smb, float eps) {
  __shared__ float red[NT / 32];
  const long row = blockIdx.x;
  const int tok = row % L;
  const float4 x = V4<XT>::load(XIN + row * sxin + threadIdx.x * 4);
  V4<float>::store(X + row * (NT * 4) + threadIdx.x * 4, x);
  adaln_store<NT>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NT * 4), red);
}

template <int NT, typename YT, typename GT, typename FT>
__global__ void __launch_bounds__(NT) resgate_out_rows_kernel(const float* __restrict__ X, const YT* __restrict__ Y,
    const GT* __restrict__ GL, FT* __restrict__ OUT, int L, long sx, long sy, long sgl) {
  const long row = blockIdx.x;
  const int tok = row % L, col = threadIdx.x * 4;
  float4 x = V4<float>::load(X + row * sx + col);
  const float4 y = V4<YT>::load(Y + row * sy + col), g = V4<GT>::load(GL + tok * sgl + col);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  V4<FT>::store(OUT + row * (NT * 4) + col, x);
}

// LayerNorm of C = 384 rows, no affine: one warp per row (3 float4 per lane), 8 rows per block
template <typename ZT, typename OutT>
__global__ void __launch_bounds__(256) layernorm384_rows_kernel(const ZT* __restrict__ Z, OutT* __restrict__ OUT, long R,
    long sz, float eps) {
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 z[3];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 3; ++k) { z[k] = V4<ZT>::load(Z + row * sz + k * 128 + lane * 4); s += z[k].x + z[k].y + z[k].z + z[k].w; }
  const float mean = warp_sum(s) / 384.f;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    z[k].x -= mean; z[k].y -= mean; z[k].z -= mean; z[k].w -= mean;
    q += z[k].x * z[k].x + z[k].y * z[k].y + z[k].z * z[k].z + z[k].w * z[k].w;
  }
  const float rstd = rsqrtf(warp_sum(q) / 384.f + eps);
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    z[k].x *= rstd; z[k].y *= rstd; z[k].z *= rstd; z[k].w *= rstd;
    V4<OutT>::store(OUT + row * 384 + k * 128 + lane * 4, z[k]);
  }
}

template <int NT, typename YT, typename GT, typename OutT>
__global__ void __launch_bounds__(NT) resgate_adaln_rows_kernel(float* __restrict__ X, const YT* __restrict__ Y,
    const GT* __restrict__ GL, const GT* __restrict__ MS, const GT* __restrict__ MB, OutT* __restrict__ OUT, int L,
    long sx, long sy, long sgl, long sms, long smb, float eps, bool has_adaln) {
  __shared__ float red[NT / 32];
  const long row = blockIdx.x;
  const int tok = row % L, col = threadIdx.x * 4;
  float4 x = V4<float>::load(X + row * sx + col), y = V4<YT>::load(Y + row * sy + col), g = V4<GT>::load(GL + tok * sgl + col);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  V4<float>::store(X + row * sx + col, x);
  if (has_adaln) adaln_store<NT>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NT * 4), red);
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

template <typename T>
__global__ void __launch_bounds__(512) swiglu_rows_kernel(const T* __restrict__ AB, T* __restrict__ OUT, int N, long sab) {
  const long row = blockIdx.x;
  for (int c = threadIdx.x * 4; c < N; c += blockDim.x * 4) {
    float4 a = V4<T>::load(AB + row * sab + c), b = V4<T>::load(AB + row * sab + N + c), o;
    o.x = a.x * sigm(a.x) * b.x; o.y = a.y * sigm(a.y) * b.y; o.z = a.z * sigm(a.z) * b.z; o.w = a.w * sigm(a.w) * b.w;
    V4<T>::store(OUT + row * N + c, o);
  }
}

// LayerNorm of C = 128 rows, no affine (the pair rows): one warp per row, 8 rows per block.
template <typename ZT, typename OutT>
__global__ void __launch_bounds__(256) layernorm128_rows_kernel(const ZT* __restrict__ Z, OutT* __restrict__ OUT, long R,
    long sz, float eps) {
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 z = V4<ZT>::load(Z + row * sz + lane * 4);
  const float mean = warp_sum(z.x + z.y + z.z + z.w) / 128.f;
  z.x -= mean; z.y -= mean; z.z -= mean; z.w -= mean;
  const float rstd = rsqrtf(warp_sum(z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w) / 128.f + eps);
  z.x *= rstd; z.y *= rstd; z.z *= rstd; z.w *= rstd;
  V4<OutT>::store(OUT + row * 128 + lane * 4, z);
}

// QK-norm in place: one block of 384 threads per row, threads 0-191 on q (columns 0-767), 192-383 on k (768-1535), 4 columns
// each; a head is 12 consecutive threads, summed through shared memory. The weight carries any logit scale folded into it.
template <typename T>
__global__ void __launch_bounds__(384) qknorm_rows_kernel(T* __restrict__ QK, long sq, const float* __restrict__ WQ,
    const float* __restrict__ WK, float eq, float ek) {
  __shared__ float part[384];
  const int t = threadIdx.x, which = t / 192, c = (t % 192) * 4;
  T* p = QK + (long)blockIdx.x * sq + which * 768 + c;
  float4 x = V4<T>::load(p);
  part[t] = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  __syncthreads();
  const int h0 = (t / 12) * 12;
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 12; ++j) ss += part[h0 + j];
  const float r = rsqrtf(ss / 48.f + (which ? ek : eq));
  const float4 w = *reinterpret_cast<const float4*>((which ? WK : WQ) + c % 48);
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

void adaln_rows(at::Tensor x, at::Tensor ms, at::Tensor mb, at::Tensor out, int64_t L, double eps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat, "adaln_rows: fp32 x of width 768");
  for (auto* t : {&ms, &mb}) check_rows(*t, "adaln_rows ms/mb", D);
  check_rows(x, "adaln_rows x", D); check_rows(out, "adaln_rows out", D);
  TORCH_CHECK(ms.scalar_type() == mb.scalar_type() && out.is_contiguous(), "adaln_rows: ms/mb dtypes, contiguous out");
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(ms.scalar_type(), GT, [&] {
    TDR_DISPATCH(out.scalar_type(), OT, [&] {
      adaln_rows_kernel<192, GT, OT><<<(unsigned)M, 192, 0, st>>>(ptr<float>(x), ptr<const GT>(ms), ptr<const GT>(mb),
          ptr<OT>(out), (int)L, x.stride(0), ms.stride(0), mb.stride(0), (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void resgate_adaln_rows(at::Tensor x, at::Tensor y, at::Tensor gl, c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb,
                        at::Tensor out, int64_t L, double eps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat, "resgate_adaln_rows: fp32 x of width 768");
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
        resgate_adaln_rows_kernel<192, YT, GT, OT><<<(unsigned)M, 192, 0, st>>>(ptr<float>(x), ptr<const YT>(y),
            ptr<const GT>(gl), ptr<const GT>(msv), ptr<const GT>(mbv), ptr<OT>(out), (int)L, x.stride(0), y.stride(0),
            gl.stride(0), msv.stride(0), mbv.stride(0), (float)eps, has);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adaln_in_rows(at::Tensor xin, at::Tensor x, at::Tensor ms, at::Tensor mb, at::Tensor out, int64_t L, double eps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat && x.is_contiguous() && xin.size(0) == M, "adaln_in_rows: fp32 x [M, 768]");
  check_rows(xin, "xin", D); check_rows(ms, "ms", D); check_rows(mb, "mb", D); check_rows(out, "out", D);
  TORCH_CHECK(ms.scalar_type() == mb.scalar_type() && out.is_contiguous(), "adaln_in_rows: ms/mb dtypes, contiguous out");
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(xin.scalar_type(), XT, [&] {
    TDR_DISPATCH(ms.scalar_type(), GT, [&] {
      TDR_DISPATCH(out.scalar_type(), OT, [&] {
        adaln_in_rows_kernel<192, XT, GT, OT><<<(unsigned)M, 192, 0, st>>>(ptr<const XT>(xin), ptr<float>(x), ptr<const GT>(ms),
            ptr<const GT>(mb), ptr<OT>(out), (int)L, xin.stride(0), ms.stride(0), mb.stride(0), (float)eps);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void resgate_out_rows(at::Tensor x, at::Tensor y, at::Tensor gl, at::Tensor out, int64_t L) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat && out.is_contiguous() && out.size(0) == M, "resgate_out_rows");
  check_rows(x, "x", D); check_rows(y, "y", D); check_rows(gl, "gl", D); check_rows(out, "out", D);
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  TDR_DISPATCH(y.scalar_type(), YT, [&] {
    TDR_DISPATCH(gl.scalar_type(), GT, [&] {
      TDR_DISPATCH(out.scalar_type(), FT, [&] {
        resgate_out_rows_kernel<192, YT, GT, FT><<<(unsigned)M, 192, 0, st>>>(ptr<const float>(x), ptr<const YT>(y),
            ptr<const GT>(gl), ptr<FT>(out), (int)L, x.stride(0), y.stride(0), gl.stride(0));
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layernorm_rows(at::Tensor z, at::Tensor out, double eps) {
  const int64_t R = z.size(0);
  TORCH_CHECK(z.size(1) == 384 && out.is_contiguous() && out.size(1) == 384 && out.size(0) == R, "layernorm_rows: [R, 384]");
  check_rows(z, "z", 384);
  const at::cuda::CUDAGuard g(z.device());
  TDR_DISPATCH(z.scalar_type(), ZT, [&] {
    TDR_DISPATCH(out.scalar_type(), OT, [&] {
      layernorm384_rows_kernel<ZT, OT><<<(unsigned)((R + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          ptr<const ZT>(z), ptr<OT>(out), R, z.stride(0), (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

void swiglu_rows(at::Tensor ab, at::Tensor out) {
  const int64_t M = ab.size(0), N = ab.size(1) / 2;
  TORCH_CHECK(N % 128 == 0 && ab.scalar_type() == out.scalar_type() && out.is_contiguous() && out.size(1) == N, "swiglu_rows");
  check_rows(ab, "ab", 2 * N);
  const at::cuda::CUDAGuard g(ab.device());
  TDR_DISPATCH(ab.scalar_type(), T, [&] {
    swiglu_rows_kernel<T><<<(unsigned)M, (unsigned)std::min<int64_t>(512, N / 4), 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<const T>(ab), ptr<T>(out), (int)N, ab.stride(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layernorm128_rows(at::Tensor z, at::Tensor out, double eps) {
  const int64_t R = z.size(0);
  TORCH_CHECK(z.size(1) == 128 && out.is_contiguous() && out.size(1) == 128 && out.size(0) == R, "layernorm128_rows");
  check_rows(z, "z", 128);
  const at::cuda::CUDAGuard g(z.device());
  TDR_DISPATCH(z.scalar_type(), ZT, [&] {
    TDR_DISPATCH(out.scalar_type(), OT, [&] {
      layernorm128_rows_kernel<ZT, OT><<<(unsigned)((R + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          ptr<const ZT>(z), ptr<OT>(out), R, z.stride(0), (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qknorm_rows(at::Tensor qk, at::Tensor wq, at::Tensor wk, double eq, double ek) {
  check_rows(qk, "qk", 1536);
  TORCH_CHECK(wq.scalar_type() == at::kFloat && wk.scalar_type() == at::kFloat && wq.is_contiguous() && wk.is_contiguous() &&
              wq.numel() == 48 && wk.numel() == 48, "qknorm_rows: fp32 [48] weights");
  const at::cuda::CUDAGuard g(qk.device());
  TDR_DISPATCH(qk.scalar_type(), T, [&] {
    qknorm_rows_kernel<T><<<(unsigned)qk.size(0), 384, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<T>(qk), qk.stride(0), ptr<const float>(wq), ptr<const float>(wk), (float)eq, (float)ek);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adaln_rows_cuda", &adaln_rows);
  m.def("resgate_adaln_rows_cuda", &resgate_adaln_rows);
  m.def("gate_rows_cuda", &gate_rows);
  m.def("swiglu_rows_cuda", &swiglu_rows);
  m.def("layernorm128_rows", &layernorm128_rows);
  m.def("qknorm_rows_cuda", &qknorm_rows);
  m.def("adaln_in_rows_cuda", &adaln_in_rows);
  m.def("resgate_out_rows_cuda", &resgate_out_rows);
  m.def("layernorm_rows_cuda", &layernorm_rows);
}
