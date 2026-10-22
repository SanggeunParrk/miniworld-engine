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

// [B R, G K] (row (b, r), column (g, k)) -> [B G, R K]: group g's K channels of every row, rows in order. A CTA moves 64 rows: they are
// read as one contiguous block into shared memory, and written as G contiguous blocks (64 rows x K) -- 16-byte pieces both ways.
constexpr int SR = 64;
__global__ void __launch_bounds__(256) split_kernel(const bf16* __restrict__ X, bf16* __restrict__ OUT, int G, int K, int rows) {
  extern __shared__ __align__(16) bf16 buf[];          // [SR][G K + 8]
  const int ch = G * K, ld = ch + 8, kp = K / 8, cp = ch / 8;
  const int r0 = blockIdx.x * SR, b = blockIdx.y, nr = min(SR, rows - r0);
  const bf16* src = X + ((long)b * rows + r0) * ch;
  for (int q = threadIdx.x; q < nr * cp; q += 256) {
    const int r = q / cp, c = q - r * cp;
    *reinterpret_cast<uint4*>(buf + r * ld + c * 8) = *reinterpret_cast<const uint4*>(src + (long)q * 8);
  }
  __syncthreads();
  for (int q = threadIdx.x; q < G * SR * kp; q += 256) {
    const int g = q / (SR * kp), rest = q - g * (SR * kp), r = rest / kp, c = rest - r * kp;
    if (r < nr)
      *reinterpret_cast<uint4*>(OUT + (((long)b * G + g) * rows + r0 + r) * K + c * 8) =
          *reinterpret_cast<const uint4*>(buf + r * ld + g * K + c * 8);
  }
}

// out[m, f] += (c1[m, f] + c2[m, f]) / max(cnt[m], 1) + (cnt[m] > 0 ? c[f] : 0)  (RoseTTAFold3's projection-bias terms, in place)
__global__ void __launch_bounds__(256) bias_finish_kernel(bf16* __restrict__ OUT, const float* __restrict__ C1, const float* __restrict__ C2,
                                                          const float* __restrict__ CNT, const float* __restrict__ C, int F, long n2) {
  for (long p = (long)blockIdx.x * 256 + threadIdx.x; p < n2; p += (long)gridDim.x * 256) {
    const long e = 2 * p, m = e / F;
    const int f = (int)(e % F);
    const float cnt = CNT[m], inv = 1.f / fmaxf(cnt, 1.f), on = cnt > 0.f ? 1.f : 0.f;
    const float2 a = *reinterpret_cast<const float2*>(C1 + e), b = *reinterpret_cast<const float2*>(C2 + e);
    __nv_bfloat162 o = *reinterpret_cast<__nv_bfloat162*>(OUT + e);
    const float x = __low2float(o) + (a.x + b.x) * inv + on * C[f], y = __high2float(o) + (a.y + b.y) * inv + on * C[f + 1];
    *reinterpret_cast<__nv_bfloat162*>(OUT + e) = __floats2bfloat162_rn(x, y);
  }
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


// x [B S N, G K] bf16 -> [B G, S, N K] bf16 (per group, the rows' K channels in row order)
at::Tensor opm_split_groups(at::Tensor x, int64_t B, int64_t S, int64_t N, int64_t G, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.numel() == B * S * N * G * K && K % 8 == 0,
              "x: contiguous bf16 [B S N, G K], K % 8 == 0");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto out = at::empty({B * G, S, N * K}, x.options());
  const int rows = (int)(S * N);
  TORCH_CHECK(G * K <= 512, "G K <= 512");
  const dim3 grid((rows + SR - 1) / SR, (unsigned)B);
  const int smem = SR * (int)(G * K + 8) * 2;
  if (smem > 48 * 1024) cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  split_kernel<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), (int)G, (int)K, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void opm_bias_finish(at::Tensor out, at::Tensor c1, at::Tensor c2, at::Tensor cnt, at::Tensor c) {
  const int64_t F = out.size(-1);
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && F % 2 == 0, "out: contiguous bf16 [..., F], F even");
  for (auto* t : {&c1, &c2})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == at::kFloat && t->numel() == out.numel(), "c1 / c2: contiguous fp32 like out");
  TORCH_CHECK(cnt.is_contiguous() && cnt.scalar_type() == at::kFloat && cnt.numel() * F == out.numel(), "cnt: fp32, one per row");
  TORCH_CHECK(c.is_contiguous() && c.scalar_type() == at::kFloat && c.numel() == F, "c: fp32 [F]");
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const long n2 = out.numel() / 2;
  const int grid = (int)std::min<long>((n2 + 255) / 256, 132L * 32);
  bias_finish_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<bf16*>(out.data_ptr()), c1.data_ptr<float>(),
      c2.data_ptr<float>(), cnt.data_ptr<float>(), c.data_ptr<float>(), (int)F, n2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_gather", &opm_gather);
  m.def("opm_split_groups", &opm_split_groups);
  m.def("opm_bias_finish", &opm_bias_finish);
}
