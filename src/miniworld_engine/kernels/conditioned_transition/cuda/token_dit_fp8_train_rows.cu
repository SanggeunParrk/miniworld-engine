// token_dit_fp8_train_rows.cu -- EXPERIMENTAL (engine 3.1.0 line, opt-in, default off): the row passes of the H100 bf16 token DiT
// TRAINING block's FP8 placements (docs/fp8_training.md "Token DiT"). Every pass writes an 8-bit GEMM operand with a DYNAMIC per-row
// scale (row max / FMAX: e4m3 448, e5m2 57344), so nothing saturates whatever the noise level of a diffusion step: no calibration, no
// overflow guard. One warp per row (D a multiple of 256, at most 3072): no block barrier, the row held in registers.
//
//   q8_rows        x [M, D] bf16 (row stride sx)              -> x8 [M, D] e4m3 / e5m2, s [M]   (xa, xt, og, dz, dy, dqkvg)
//   swiglu_q8      ab [M, 2N] bf16                            -> h8 = q(silu(a) b) [M, N], s    (forward: the squeeze input; h not stored)
//   swiglu_bwd_q8  dh [M, N], ab [M, 2N] bf16                 -> dab [M, 2N] bf16, h [M, N] bf16 (as swiglu_bwd), dab8 [M, 2N], s
//
// The default route's kernels (token_dit_train_rows.cu) are untouched; this is a separate extension built only when a placement is on.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int WPB = 4;  // warps (rows) per block

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ void ld8(const __nv_bfloat16* p, float* v) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ void st8(__nv_bfloat16* p, const float* v) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

template <bool E5>
__device__ __forceinline__ void stq8(uint8_t* p, const float* v, float inv) {
  constexpr __nv_fp8_interpretation_t F = E5 ? __NV_E5M2 : __NV_E4M3;
  uint32_t w[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[4 * j] * inv, v[4 * j + 1] * inv), __NV_SATFINITE, F);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[4 * j + 2] * inv, v[4 * j + 3] * inv), __NV_SATFINITE, F);
    w[j] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
  *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
}

template <bool E5>
__device__ __forceinline__ float fmax_of() { return E5 ? 57344.f : 448.f; }

template <int IT, bool E5>
__global__ void __launch_bounds__(32 * WPB) q8_rows_k(const __nv_bfloat16* __restrict__ X, long sx, uint8_t* __restrict__ OUT,
                                                      float* __restrict__ S, int M) {
  const long r = (long)blockIdx.x * WPB + threadIdx.x / 32;
  if (r >= M) return;
  const int lane = threadIdx.x % 32;
  constexpr int D = IT * 256;
  float v[IT * 8];
  float am = 0.f;
#pragma unroll
  for (int i = 0; i < IT; ++i) {
    ld8(X + r * sx + i * 256 + lane * 8, v + 8 * i);
#pragma unroll
    for (int j = 0; j < 8; ++j) am = fmaxf(am, fabsf(v[8 * i + j]));
  }
  am = wmax(am);
  const float sc = fmaxf(am / fmax_of<E5>(), 1e-30f), inv = 1.f / sc;  // floor: a channel with amax ~ 1e-37 (real weights) made inv = inf -> NaN
#pragma unroll
  for (int i = 0; i < IT; ++i) stq8<E5>(OUT + r * D + i * 256 + lane * 8, v + 8 * i, inv);
  if (lane == 0) S[r] = sc;
}

// forward: h = silu(a) b in fp32 from the bf16 a | b the expand GEMM wrote (swiglu_rows' expression), quantized per row
template <int IT>
__global__ void __launch_bounds__(32 * WPB) swiglu_q8_k(const __nv_bfloat16* __restrict__ AB, uint8_t* __restrict__ H8,
                                                        float* __restrict__ S, int M) {
  const long r = (long)blockIdx.x * WPB + threadIdx.x / 32;
  if (r >= M) return;
  const int lane = threadIdx.x % 32;
  constexpr int N = IT * 256;
  float h[IT * 8];
  float am = 0.f;
#pragma unroll
  for (int i = 0; i < IT; ++i) {
    float a[8], b[8];
    ld8(AB + r * 2 * N + i * 256 + lane * 8, a);
    ld8(AB + r * 2 * N + N + i * 256 + lane * 8, b);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = 1.f / (1.f + __expf(-a[j]));
      h[8 * i + j] = a[j] * s * b[j];
      am = fmaxf(am, fabsf(h[8 * i + j]));
    }
  }
  am = wmax(am);
  const float sc = fmaxf(am / 448.f, 1e-30f), inv = 1.f / sc;
#pragma unroll
  for (int i = 0; i < IT; ++i) stq8<false>(H8 + r * N + i * 256 + lane * 8, h + 8 * i, inv);
  if (lane == 0) S[r] = sc;
}

// backward: swiglu_bwd's outputs (dab, h in bf16) plus dab quantized per row (the dgrad dxt = dab Wab operand)
template <int IT, bool E5>
__global__ void __launch_bounds__(32 * WPB) swiglu_bwd_q8_k(const __nv_bfloat16* __restrict__ DH, const __nv_bfloat16* __restrict__ AB,
                                                            __nv_bfloat16* __restrict__ DAB, __nv_bfloat16* __restrict__ Hh,
                                                            uint8_t* __restrict__ DAB8, float* __restrict__ S, int M) {
  const long r = (long)blockIdx.x * WPB + threadIdx.x / 32;
  if (r >= M) return;
  const int lane = threadIdx.x % 32;
  constexpr int N = IT * 256;
  float da[IT * 8], db[IT * 8];
  float am = 0.f;
#pragma unroll
  for (int i = 0; i < IT; ++i) {
    const int c = i * 256 + lane * 8;
    float a[8], b[8], dh[8], h[8];
    ld8(DH + r * N + c, dh);
    ld8(AB + r * 2 * N + c, a);
    ld8(AB + r * 2 * N + N + c, b);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = 1.f / (1.f + __expf(-a[j]));
      const float as = a[j] * s;
      h[j] = as * b[j];
      da[8 * i + j] = dh[j] * b[j] * s * (1.f + a[j] * (1.f - s));
      db[8 * i + j] = dh[j] * as;
    }
    st8(Hh + r * N + c, h);
    // the bf16 values the wgrad reads; the e4m3 / e5m2 copy quantizes the same rounded values
    st8(DAB + r * 2 * N + c, da + 8 * i);
    st8(DAB + r * 2 * N + N + c, db + 8 * i);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      da[8 * i + j] = __bfloat162float(__float2bfloat16_rn(da[8 * i + j]));
      db[8 * i + j] = __bfloat162float(__float2bfloat16_rn(db[8 * i + j]));
      am = fmaxf(am, fmaxf(fabsf(da[8 * i + j]), fabsf(db[8 * i + j])));
    }
  }
  am = wmax(am);
  const float sc = fmaxf(am / fmax_of<E5>(), 1e-30f), inv = 1.f / sc;  // floor: a channel with amax ~ 1e-37 (real weights) made inv = inf -> NaN
#pragma unroll
  for (int i = 0; i < IT; ++i) {
    const int c = i * 256 + lane * 8;
    stq8<E5>(DAB8 + r * 2 * N + c, da + 8 * i, inv);
    stq8<E5>(DAB8 + r * 2 * N + N + c, db + 8 * i, inv);
  }
  if (lane == 0) S[r] = sc;
}

template <typename T> T* P(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

bool is8(const at::Tensor& t) { return t.scalar_type() == at::kFloat8_e4m3fn || t.scalar_type() == at::kFloat8_e5m2; }

#define DISPATCH_IT(D, ...)                                              \
  switch ((D) / 256) {                                                    \
    case 3: { constexpr int IT = 3; __VA_ARGS__; break; }                 \
    case 6: { constexpr int IT = 6; __VA_ARGS__; break; }                 \
    case 12: { constexpr int IT = 12; __VA_ARGS__; break; }               \
    default: TORCH_CHECK(false, "fp8 train rows: width 768 / 1536 / 3072"); \
  }
}  // namespace

void q8_rows(at::Tensor x, at::Tensor out8, at::Tensor s) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0, "q8_rows: bf16 [M, D] rows");
  TORCH_CHECK(is8(out8) && out8.is_contiguous() && out8.size(0) == M && out8.size(1) == D, "q8_rows: 8-bit out [M, D]");
  TORCH_CHECK(s.scalar_type() == at::kFloat && s.is_contiguous() && s.numel() == M, "q8_rows: fp32 scales [M]");
  const at::cuda::CUDAGuard g(x.device());
  const bool e5 = out8.scalar_type() == at::kFloat8_e5m2;
  const unsigned grid = (unsigned)((M + WPB - 1) / WPB);
  auto st = at::cuda::getCurrentCUDAStream();
  DISPATCH_IT(D, if (e5) q8_rows_k<IT, true><<<grid, 32 * WPB, 0, st>>>(P<const __nv_bfloat16>(x), (long)x.stride(0), P<uint8_t>(out8), P<float>(s), (int)M);
                 else q8_rows_k<IT, false><<<grid, 32 * WPB, 0, st>>>(P<const __nv_bfloat16>(x), (long)x.stride(0), P<uint8_t>(out8), P<float>(s), (int)M));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_q8(at::Tensor ab, at::Tensor h8, at::Tensor s) {
  const int64_t M = ab.size(0), N = ab.size(1) / 2;
  TORCH_CHECK(ab.scalar_type() == at::kBFloat16 && ab.is_contiguous(), "swiglu_q8: bf16 [M, 2N]");
  TORCH_CHECK(h8.scalar_type() == at::kFloat8_e4m3fn && h8.is_contiguous() && h8.size(0) == M && h8.size(1) == N, "swiglu_q8: e4m3 [M, N]");
  TORCH_CHECK(s.scalar_type() == at::kFloat && s.numel() == M, "swiglu_q8: fp32 scales [M]");
  const at::cuda::CUDAGuard g(ab.device());
  const unsigned grid = (unsigned)((M + WPB - 1) / WPB);
  auto st = at::cuda::getCurrentCUDAStream();
  DISPATCH_IT(N, swiglu_q8_k<IT><<<grid, 32 * WPB, 0, st>>>(P<const __nv_bfloat16>(ab), P<uint8_t>(h8), P<float>(s), (int)M));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_bwd_q8(at::Tensor dh, at::Tensor ab, at::Tensor dab, at::Tensor h, at::Tensor dab8, at::Tensor s) {
  const int64_t M = dh.size(0), N = dh.size(1);
  TORCH_CHECK(dh.scalar_type() == at::kBFloat16 && dh.is_contiguous() && ab.scalar_type() == at::kBFloat16 && ab.is_contiguous() &&
                  ab.size(1) == 2 * N && dab.sizes() == ab.sizes() && dab.is_contiguous() && h.sizes() == dh.sizes() && h.is_contiguous(),
              "swiglu_bwd_q8: bf16 dh [M, N], ab / dab [M, 2N], h [M, N]");
  TORCH_CHECK(is8(dab8) && dab8.is_contiguous() && dab8.sizes() == ab.sizes(), "swiglu_bwd_q8: 8-bit dab [M, 2N]");
  TORCH_CHECK(s.scalar_type() == at::kFloat && s.numel() == M, "swiglu_bwd_q8: fp32 scales [M]");
  const at::cuda::CUDAGuard g(dh.device());
  const bool e5 = dab8.scalar_type() == at::kFloat8_e5m2;
  const unsigned grid = (unsigned)((M + WPB - 1) / WPB);
  auto st = at::cuda::getCurrentCUDAStream();
  DISPATCH_IT(N, if (e5) swiglu_bwd_q8_k<IT, true><<<grid, 32 * WPB, 0, st>>>(P<const __nv_bfloat16>(dh), P<const __nv_bfloat16>(ab),
                                                                               P<__nv_bfloat16>(dab), P<__nv_bfloat16>(h), P<uint8_t>(dab8), P<float>(s), (int)M);
                 else swiglu_bwd_q8_k<IT, false><<<grid, 32 * WPB, 0, st>>>(P<const __nv_bfloat16>(dh), P<const __nv_bfloat16>(ab),
                                                                             P<__nv_bfloat16>(dab), P<__nv_bfloat16>(h), P<uint8_t>(dab8), P<float>(s), (int)M));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("q8_rows", &q8_rows);
  m.def("swiglu_q8", &swiglu_q8);
  m.def("swiglu_bwd_q8", &swiglu_bwd_q8);
}
