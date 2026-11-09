// outer_product_mean gather.cu -- the composed OPM's pass between its batched GEMM and its out-projection (H100 inference):
//
//   out[(b, i, j), (g, k, l)] = prod[b, g, i K + k, j K + l]   [then LayerNorm over the G K K columns * w + b (Chai-1's product norm)]
//
// prod [B G, N K, N K] bf16 is the per-group outer product summed over the MSA rows (one cuBLAS batched GEMM); out [B N N, G K K] bf16 is
// what the out-projection reads. A CTA takes one (b, i) and 32 consecutive j: for every plane (g, k) the 32 j x K values are contiguous in
// prod (read as 16-byte pieces, a warp covering 512 contiguous bytes), staged in shared memory as the 32 output rows; then a warp per row
// (LayerNorm on the way) writes the rows, which are contiguous in out. G K K = 512 (Chai-1: 8 x 8 x 8) or 1024 (RoseTTAFold3: 1 x 32 x 32).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {
typedef __nv_bfloat16 bf16;
constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int R, int JT = (R == 512 ? 32 : 16)>   // R = G K K, the row width; JT consecutive j per CTA (a ~33 KB tile)
__global__ void __launch_bounds__(THREADS) gather_kernel(const bf16* __restrict__ P, bf16* __restrict__ OUT, const float* __restrict__ W,
    const float* __restrict__ B, int G, int K, int N, float eps) {
  constexpr int LDS = R + 8;                     // padded smem row (bf16)
  extern __shared__ __align__(16) bf16 tile[];   // [JT][LDS]
  const int jt = blockIdx.x, i = blockIdx.y, b = blockIdx.z;
  const int j0 = jt * JT, nj = min(JT, N - j0);
  const long nk = (long)N * K;
  const int per_plane = JT * K / 8;              // 16-byte pieces of a plane's 32 j x K values
  for (int q = threadIdx.x; q < G * K * per_plane; q += THREADS) {
    const int p = q / per_plane, r = q % per_plane;
    const int jl = r / (K / 8), l0 = (r % (K / 8)) * 8;
    if (jl < nj) {
      const int g = p / K, k = p % K;
      const uint4 u = *reinterpret_cast<const uint4*>(P + (((long)b * G + g) * nk + (long)i * K + k) * nk + (long)(j0 + jl) * K + l0);
      *reinterpret_cast<uint4*>(tile + jl * LDS + p * K + l0) = u;
    }
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr int PER = R / 32;                    // values per lane: 16 or 32 (2 or 4 pieces of 8)
  for (int jl = warp; jl < nj; jl += THREADS / 32) {
    float v[PER];
#pragma unroll
    for (int c = 0; c < PER / 8; ++c) {
      const uint4 u = *reinterpret_cast<const uint4*>(tile + jl * LDS + (c * 32 + lane) * 8);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int e = 0; e < 4; ++e) { v[c * 8 + 2 * e] = __low2float(h[e]); v[c * 8 + 2 * e + 1] = __high2float(h[e]); }
    }
    if (W != nullptr) {
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < PER; ++e) s += v[e];
      const float mean = warp_sum(s) / R;
      float q = 0.f;
#pragma unroll
      for (int e = 0; e < PER; ++e) { v[e] -= mean; q += v[e] * v[e]; }
      const float rs = rsqrtf(warp_sum(q) / R + eps);
#pragma unroll
      for (int c = 0; c < PER / 8; ++c)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int col = (c * 32 + lane) * 8 + e;
          v[c * 8 + e] = v[c * 8 + e] * rs * W[col] + B[col];
        }
    }
    bf16* dst = OUT + (((long)b * N + i) * N + j0 + jl) * R;
#pragma unroll
    for (int c = 0; c < PER / 8; ++c) {
      uint4 u;
      __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
      for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(v[c * 8 + 2 * e], v[c * 8 + 2 * e + 1]);
      *reinterpret_cast<uint4*>(dst + (c * 32 + lane) * 8) = u;
    }
  }
}

template <int R, int JT = (R == 512 ? 32 : 16)>
void launch(const at::Tensor& prod, at::Tensor& out, const float* wp, const float* bp, int G, int K, int B, int N, double eps) {
  const int smem = JT * (R + 8) * 2;
  auto f = gather_kernel<R, JT>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  f<<<dim3((N + JT - 1) / JT, N, B), THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(prod.data_ptr()), reinterpret_cast<bf16*>(out.data_ptr()), wp, bp, G, K, N, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// prod [B G, N K, N K] bf16 -> [B N N, G K K] bf16 [, LayerNorm with fp32 w / b]
at::Tensor opm_gather(at::Tensor prod, int64_t G, int64_t K, c10::optional<at::Tensor> w, c10::optional<at::Tensor> b, double eps) {
  TORCH_CHECK(prod.is_cuda() && prod.is_contiguous() && prod.scalar_type() == at::kBFloat16 && prod.dim() == 3 && prod.size(1) == prod.size(2)
              && prod.size(0) % G == 0 && prod.size(1) % K == 0 && K % 8 == 0, "prod: contiguous bf16 [B G, N K, N K], K % 8 == 0");
  const int64_t r = G * K * K;
  TORCH_CHECK(r == 512 || r == 1024, "G K K = 512 or 1024 (Chai-1 8 x 8 x 8, RoseTTAFold3 1 x 32 x 32)");
  TORCH_CHECK(!w || (w->scalar_type() == at::kFloat && w->numel() == r && b && b->numel() == r), "LayerNorm w / b: fp32 [G K K]");
  const int64_t B = prod.size(0) / G, N = prod.size(1) / K;
  const at::cuda::OptionalCUDAGuard guard(prod.device());
  auto out = at::empty({B * N * N, r}, prod.options());
  const float* wp = w ? w->data_ptr<float>() : nullptr;
  const float* bp = b ? b->data_ptr<float>() : nullptr;
  if (r == 512) launch<512>(prod, out, wp, bp, (int)G, (int)K, (int)B, (int)N, eps);
  else launch<1024>(prod, out, wp, bp, (int)G, (int)K, (int)B, (int)N, eps);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_gather", &opm_gather);
}
