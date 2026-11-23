// token_f8_sm120.cu -- OPT-IN e4m3 row passes of the RTX 5090 (sm_120a) token DiT FP8 route (token_dit_fp8_sm120.py, docs/fp8.md "RTX 5090").
//
// The GEMMs of the route are cuBLASLt e4m3 with row-wise scales (torch._scaled_mm: fp32 accumulation, bf16 out); these passes make their
// e4m3 operands with a DYNAMIC per-row scale (row max / 448, the H100 token DiT recipe: a row's own maximum maps to 448, nothing saturates):
//
//   adaln_q8     the transition's AdaLN epilogue  y = g LN(x) + B   (g = sigmoid(S + sb), or 1 + S for the identity AdaLN), y in fp32 -> e4m3 + scale
//                (the default route's adaln_epi_kernel with the e4m3 store: one pass, the bf16 xa is never written)
//   quant_rows   any bf16 rows [M, D] (row stride) -> e4m3 + scale (the attention's qkvg input, its out-projection input og)
//   swiglu_q8    h = silu(a) b (u) from the expand's bf16 [a | b (| u)] rows -> e4m3 + scale (the squeeze's input)
//
// One warp per row, every value of the row in registers (D / 32 per lane), fp32 arithmetic, cvt.rn.satfinite.e4m3x2.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace tf8 {

constexpr int RW = 8;  // rows (warps) per block

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// 4 bf16 at p (8-byte aligned) -> float4
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  uint2 u = *reinterpret_cast<const uint2*>(p);
  __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&u.x), b = *reinterpret_cast<__nv_bfloat162*>(&u.y);
  float2 fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
  return make_float4(fa.x, fa.y, fb.x, fb.y);
}

// four floats (already scaled) -> four e4m3 bytes, x in the lowest byte
__device__ __forceinline__ uint32_t q4(float4 v) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(v.y), "f"(v.x));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(v.w), "f"(v.z));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ float amax4(float4 v) { return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))); }
__device__ __forceinline__ float4 mul4(float4 v, float s) { return make_float4(v.x * s, v.y * s, v.z * s, v.w * s); }
__device__ __forceinline__ float sig(float v) { return 1.f / (1.f + __expf(-v)); }

// Store the row's NV float4 (column vcol = v * 128 + 4 lane) as e4m3 with the row scale amax / 448.
template <int NV>
__device__ __forceinline__ void store_q8(const float4 (&y)[NV], float am, uint8_t* __restrict__ out, float* __restrict__ srow, int lane) {
  // floor 1e-30 (the H100 rows' quant_weight_ref rule): a row of tiny values (am 1e-39) made 448 / am = inf and 0 x inf = NaN bytes
  am = warp_max(am);
  if (am > 0.f) am = fmaxf(am, 1e-30f);
  const float s = am > 0.f ? am / 448.f : 1.f;
  const float inv = am > 0.f ? 448.f / am : 1.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) *reinterpret_cast<uint32_t*>(out + v * 128 + 4 * lane) = q4(mul4(y[v], inv));
  if (lane == 0) *srow = s;
}

// y = g LN(x) + B -> e4m3 [M, D] + scale [M]; row r reads SBT row r % P ([S | B], 2 D wide); SB [D] (x's dtype) unless ID (g = 1 + S).
template <int D, bool ID>
__global__ void __launch_bounds__(RW * 32) adaln_q8_kernel(const __nv_bfloat16* __restrict__ X, long sx, const __nv_bfloat16* __restrict__ SBT, long ssb,
                                                           const __nv_bfloat16* __restrict__ SB, uint8_t* __restrict__ OUT, float* __restrict__ SROW, long M, long P,
                                                           float eps) {
  constexpr int NV = D / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= M) return;
  const __nv_bfloat16* t = SBT + (row % P) * ssb;
  float4 x[NV], sc[NV], bi[NV];
  float s = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int c = v * 128 + 4 * lane;
    x[v] = ld4(X + row * sx + c);
    sc[v] = ld4(t + c);
    bi[v] = ld4(t + D + c);
    s += x[v].x + x[v].y + x[v].z + x[v].w;
  }
  const float mean = warp_sum(s) / D;
  float q = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    x[v] = make_float4(x[v].x - mean, x[v].y - mean, x[v].z - mean, x[v].w - mean);
    q += x[v].x * x[v].x + x[v].y * x[v].y + x[v].z * x[v].z + x[v].w * x[v].w;
  }
  const float rstd = rsqrtf(warp_sum(q) / D + eps);
  float am = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    float4 g;
    if constexpr (ID) {
      g = make_float4(sc[v].x + 1.f, sc[v].y + 1.f, sc[v].z + 1.f, sc[v].w + 1.f);
    } else {
      const float4 b = ld4(SB + v * 128 + 4 * lane);
      g = make_float4(sig(sc[v].x + b.x), sig(sc[v].y + b.y), sig(sc[v].z + b.z), sig(sc[v].w + b.w));
    }
    x[v] = make_float4(g.x * x[v].x * rstd + bi[v].x, g.y * x[v].y * rstd + bi[v].y, g.z * x[v].z * rstd + bi[v].z, g.w * x[v].w * rstd + bi[v].w);
    am = fmaxf(am, amax4(x[v]));
  }
  store_q8<NV>(x, am, OUT + row * D, SROW + row, lane);
}

template <int D>
__global__ void __launch_bounds__(RW * 32) quant_rows_kernel(const __nv_bfloat16* __restrict__ X, long sx, uint8_t* __restrict__ OUT, float* __restrict__ SROW, long M) {
  constexpr int NV = D / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= M) return;
  float4 x[NV];
  float am = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    x[v] = ld4(X + row * sx + v * 128 + 4 * lane);
    am = fmaxf(am, amax4(x[v]));
  }
  store_q8<NV>(x, am, OUT + row * D, SROW + row, lane);
}

// h = silu(a) b (UP: silu(a) b u) of the expand's rows AB [M, (2|3) H] (row stride sab) -> e4m3 [M, H] + scale
template <int H, bool UP>
__global__ void __launch_bounds__(RW * 32) swiglu_q8_kernel(const __nv_bfloat16* __restrict__ AB, long sab, uint8_t* __restrict__ OUT, float* __restrict__ SROW, long M) {
  constexpr int NV = H / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= M) return;
  const __nv_bfloat16* r = AB + row * sab;
  float4 h[NV];
  float am = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int c = v * 128 + 4 * lane;
    const float4 a = ld4(r + c), b = ld4(r + H + c);
    h[v] = make_float4(a.x * sig(a.x) * b.x, a.y * sig(a.y) * b.y, a.z * sig(a.z) * b.z, a.w * sig(a.w) * b.w);
    if constexpr (UP) {
      const float4 u = ld4(r + 2 * H + c);
      h[v] = make_float4(h[v].x * u.x, h[v].y * u.y, h[v].z * u.z, h[v].w * u.w);
    }
    am = fmaxf(am, amax4(h[v]));
  }
  store_q8<NV>(h, am, OUT + row * H, SROW + row, lane);
}

}  // namespace tf8

using namespace tf8;

static void check_rows(const torch::Tensor& x, const torch::Tensor& out, const torch::Tensor& s, int64_t D) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "bf16 [M, D] rows, unit column stride");
  TORCH_CHECK(out.is_contiguous() && out.element_size() == 1 && out.size(0) == x.size(0) && out.size(1) == D, "out: e4m3 [M, D]");
  TORCH_CHECK(s.is_contiguous() && s.scalar_type() == at::kFloat && s.numel() == x.size(0), "scale: fp32 [M]");
}

#define DISPATCH_D(D, ...)                                     \
  switch (D) {                                                 \
    case 768: { constexpr int DD = 768; __VA_ARGS__; } break;   \
    case 1536: { constexpr int DD = 1536; __VA_ARGS__; } break; \
    case 384: { constexpr int DD = 384; __VA_ARGS__; } break;   \
    case 1024: { constexpr int DD = 1024; __VA_ARGS__; } break; \
    case 512: { constexpr int DD = 512; __VA_ARGS__; } break;   \
    case 2048: { constexpr int DD = 2048; __VA_ARGS__; } break; \
    default: TORCH_CHECK(false, "token_f8_sm120: width ", D, " not built"); \
  }

// x [M, D] bf16, sbt [P, 2 D] bf16, sb [D] bf16 or empty (identity AdaLN) -> out e4m3 [M, D], s [M]
void adaln_q8(torch::Tensor x, torch::Tensor sbt, torch::Tensor sb, torch::Tensor out, torch::Tensor s, double eps) {
  const int64_t M = x.size(0), D = x.size(1), P = sbt.size(0);
  check_rows(x, out, s, D);
  TORCH_CHECK(sbt.scalar_type() == at::kBFloat16 && sbt.stride(1) == 1 && sbt.size(1) == 2 * D && sbt.stride(0) % 4 == 0 && M % P == 0, "sbt: bf16 [P, 2 D], P | M");
  const bool id = sb.numel() == 0;
  TORCH_CHECK(id || (sb.is_contiguous() && sb.scalar_type() == at::kBFloat16 && sb.numel() == D), "sb: bf16 [D]");
  if (M == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid((M + RW - 1) / RW), block(RW * 32);
  auto X = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto T = reinterpret_cast<const __nv_bfloat16*>(sbt.data_ptr());
  auto B = id ? nullptr : reinterpret_cast<const __nv_bfloat16*>(sb.data_ptr());
  auto O = reinterpret_cast<uint8_t*>(out.data_ptr());
  auto S = s.data_ptr<float>();
  DISPATCH_D(D, if (id) adaln_q8_kernel<DD, true><<<grid, block, 0, st>>>(X, x.stride(0), T, sbt.stride(0), B, O, S, M, P, (float)eps);
             else adaln_q8_kernel<DD, false><<<grid, block, 0, st>>>(X, x.stride(0), T, sbt.stride(0), B, O, S, M, P, (float)eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quant_rows(torch::Tensor x, torch::Tensor out, torch::Tensor s) {
  const int64_t M = x.size(0), D = x.size(1);
  check_rows(x, out, s, D);
  if (M == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid((M + RW - 1) / RW), block(RW * 32);
  DISPATCH_D(D, quant_rows_kernel<DD><<<grid, block, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
                                                             reinterpret_cast<uint8_t*>(out.data_ptr()), s.data_ptr<float>(), M));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ab [M, 2 H] (up: [M, 3 H]) bf16 -> out e4m3 [M, H], s [M]
void swiglu_q8(torch::Tensor ab, torch::Tensor out, torch::Tensor s, bool up) {
  const int64_t M = ab.size(0), H = out.size(1);
  TORCH_CHECK(ab.is_cuda() && ab.scalar_type() == at::kBFloat16 && ab.dim() == 2 && ab.stride(1) == 1 && ab.stride(0) % 4 == 0 && ab.size(1) == (up ? 3 : 2) * H,
              "ab: bf16 [M, (2|3) H]");
  TORCH_CHECK(out.is_contiguous() && out.element_size() == 1 && out.size(0) == M, "out: e4m3 [M, H]");
  TORCH_CHECK(s.is_contiguous() && s.scalar_type() == at::kFloat && s.numel() == M, "scale: fp32 [M]");
  if (M == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid((M + RW - 1) / RW), block(RW * 32);
  auto A = reinterpret_cast<const __nv_bfloat16*>(ab.data_ptr());
  auto O = reinterpret_cast<uint8_t*>(out.data_ptr());
  DISPATCH_D(H, if (up) swiglu_q8_kernel<DD, true><<<grid, block, 0, st>>>(A, ab.stride(0), O, s.data_ptr<float>(), M);
             else swiglu_q8_kernel<DD, false><<<grid, block, 0, st>>>(A, ab.stride(0), O, s.data_ptr<float>(), M));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adaln_q8", &adaln_q8);
  m.def("quant_rows", &quant_rows);
  m.def("swiglu_q8", &swiglu_q8);
}
