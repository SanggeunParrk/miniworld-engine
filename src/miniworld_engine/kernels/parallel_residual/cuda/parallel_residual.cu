// The fused residual add of parallel branches (modules/residual.py, FoldForge R12), hand-written CUDA: ONE source built for H100
// (sm_90a, `parallel_residual_sm90`) and A100 (sm_80, `parallel_residual_sm80`) by kernels/parallel_residual/cuda/__init__.py.
//
//   forward   out = x + sum_i (y_i - x)      1 to 4 branches; the differences and the sum in fp32, ONE rounding to the dtype at the end
//   backward  gx  = (1 - n) g                (d out / d y_i = g needs no kernel); fp32 product, one rounding
//
// The same arithmetic in the same order as the Triton kernel (kernels/parallel_residual/triton/main.py) and the pytorch reference: acc =
// (y0 - x) + (y1 - x) + ..., out = x + acc. A pure streaming pass (n + 2 tensor passes), so the kernel is shaped for the HBM bandwidth:
// 16-byte vector accesses, all of a thread's vectors loaded before any is used (bytes in flight; fwd_unroll / kUnroll), and
// streaming cache hints (ld.global.cs / st.global.cs: every byte is touched once). bf16 / fp16 / fp32; flat contiguous tensors (no
// shape dependence: no launch-config table). A pointer that is not 16-byte aligned takes the scalar kernel.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kUnroll = 4;  // 16-byte vectors per thread of the backward (one tensor in, one out)
// 16-byte vectors per thread and tensor of the forward: the loads of (1 + NB) tensors in flight fill the registers (measured H100: 4
// vectors a tensor at NB = 3 ran 25 % under the bandwidth; one vector a tensor, as the Triton kernel, keeps the occupancy)
__host__ __device__ constexpr int fwd_unroll(int nb) { return nb == 1 ? 2 : 1; }
constexpr int kMaxBranches = 4;

struct Branches {
  const void* p[kMaxBranches];
};

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }

template <typename T, int NB>
__global__ void __launch_bounds__(kThreads) parallel_residual_fwd_k(const T* __restrict__ x, Branches ys, T* __restrict__ out, int64_t n) {
  constexpr int V = 16 / sizeof(T), U = fwd_unroll(NB);
  const int64_t nvec = n / V;
  const int64_t first = (int64_t)blockIdx.x * kThreads * U + threadIdx.x;
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  uint4 a[U], b[NB][U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int64_t q = first + (int64_t)u * kThreads;
    if (q < nvec) {
      a[u] = __ldcs(xv + q);
#pragma unroll
      for (int i = 0; i < NB; ++i) b[i][u] = __ldcs(reinterpret_cast<const uint4*>(ys.p[i]) + q);
    }
  }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int64_t q = first + (int64_t)u * kThreads;
    if (q < nvec) {
      const T* xe = reinterpret_cast<const T*>(&a[u]);
      uint4 r;
      T* re = reinterpret_cast<T*>(&r);
#pragma unroll
      for (int e = 0; e < V; ++e) {
        const float xf = to_f(xe[e]);
        float acc = to_f(reinterpret_cast<const T*>(&b[0][u])[e]) - xf;
#pragma unroll
        for (int i = 1; i < NB; ++i) acc += to_f(reinterpret_cast<const T*>(&b[i][u])[e]) - xf;
        re[e] = from_f<T>(xf + acc);
      }
      __stcs(reinterpret_cast<uint4*>(out) + q, r);
    }
  }
  if (blockIdx.x == gridDim.x - 1) {  // the < V elements past the last whole vector
    for (int64_t k = nvec * V + threadIdx.x; k < n; k += kThreads) {
      const float xf = to_f(x[k]);
      float acc = to_f(static_cast<const T*>(ys.p[0])[k]) - xf;
#pragma unroll
      for (int i = 1; i < NB; ++i) acc += to_f(static_cast<const T*>(ys.p[i])[k]) - xf;
      out[k] = from_f<T>(xf + acc);
    }
  }
}

// any alignment: one element a thread, grid-stride
template <typename T, int NB>
__global__ void __launch_bounds__(kThreads) parallel_residual_fwd_scalar_k(const T* __restrict__ x, Branches ys, T* __restrict__ out, int64_t n) {
  for (int64_t k = (int64_t)blockIdx.x * kThreads + threadIdx.x; k < n; k += (int64_t)gridDim.x * kThreads) {
    const float xf = to_f(x[k]);
    float acc = to_f(static_cast<const T*>(ys.p[0])[k]) - xf;
#pragma unroll
    for (int i = 1; i < NB; ++i) acc += to_f(static_cast<const T*>(ys.p[i])[k]) - xf;
    out[k] = from_f<T>(xf + acc);
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads) parallel_residual_bwd_k(const T* __restrict__ g, T* __restrict__ gx, float scale, int64_t n, int vec) {
  constexpr int V = 16 / sizeof(T);
  if (vec) {
    const int64_t nvec = n / V;
    const int64_t first = (int64_t)blockIdx.x * kThreads * kUnroll + threadIdx.x;
    uint4 a[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int64_t q = first + (int64_t)u * kThreads;
      if (q < nvec) a[u] = __ldcs(reinterpret_cast<const uint4*>(g) + q);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int64_t q = first + (int64_t)u * kThreads;
      if (q < nvec) {
        const T* ge = reinterpret_cast<const T*>(&a[u]);
        uint4 r;
        T* re = reinterpret_cast<T*>(&r);
#pragma unroll
        for (int e = 0; e < V; ++e) re[e] = from_f<T>(to_f(ge[e]) * scale);
        __stcs(reinterpret_cast<uint4*>(gx) + q, r);
      }
    }
    if (blockIdx.x == gridDim.x - 1)
      for (int64_t k = nvec * V + threadIdx.x; k < n; k += kThreads) gx[k] = from_f<T>(to_f(g[k]) * scale);
  } else {
    for (int64_t k = (int64_t)blockIdx.x * kThreads + threadIdx.x; k < n; k += (int64_t)gridDim.x * kThreads)
      gx[k] = from_f<T>(to_f(g[k]) * scale);
  }
}

bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

int64_t vec_blocks(int64_t n, int elem, int unroll) {
  const int64_t nvec = n / (16 / elem);
  return std::max<int64_t>(1, (nvec + (int64_t)kThreads * unroll - 1) / ((int64_t)kThreads * unroll));
}

int64_t scalar_blocks(int64_t n) { return std::min<int64_t>((n + kThreads - 1) / kThreads, 65535); }

template <typename T, int NB>
void launch_fwd(const at::Tensor& x, const Branches& ys, bool vec, at::Tensor& out) {
  const int64_t n = x.numel();
  const auto stream = at::cuda::getCurrentCUDAStream();
  const T* xp = reinterpret_cast<const T*>(x.data_ptr());
  T* op = reinterpret_cast<T*>(out.data_ptr());
  if (vec)
    parallel_residual_fwd_k<T, NB><<<(unsigned)vec_blocks(n, sizeof(T), fwd_unroll(NB)), kThreads, 0, stream>>>(xp, ys, op, n);
  else
    parallel_residual_fwd_scalar_k<T, NB><<<(unsigned)scalar_blocks(n), kThreads, 0, stream>>>(xp, ys, op, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void dispatch_nb(const at::Tensor& x, const Branches& ys, int nb, bool vec, at::Tensor& out) {
  switch (nb) {
    case 1: launch_fwd<T, 1>(x, ys, vec, out); break;
    case 2: launch_fwd<T, 2>(x, ys, vec, out); break;
    case 3: launch_fwd<T, 3>(x, ys, vec, out); break;
    default: launch_fwd<T, 4>(x, ys, vec, out); break;
  }
}

void check_dtype(const at::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat,
              "parallel_residual: bfloat16 / float16 / float32 tensors");
}

}  // namespace

// out = x + sum_i (y_i - x) over 1 .. 4 contiguous CUDA tensors of x's shape, dtype and device
at::Tensor forward(at::Tensor x, std::vector<at::Tensor> ys) {
  const int nb = (int)ys.size();
  TORCH_CHECK(nb >= 1 && nb <= kMaxBranches, "parallel_residual: 1 to 4 branches");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "parallel_residual: a contiguous CUDA x");
  check_dtype(x);
  Branches b{};
  bool vec = aligned16(x.data_ptr());
  for (int i = 0; i < nb; ++i) {
    TORCH_CHECK(ys[i].is_contiguous() && ys[i].device() == x.device() && ys[i].scalar_type() == x.scalar_type() && ys[i].numel() == x.numel(),
                "parallel_residual: every branch a contiguous tensor of x's size, dtype and device");
    b.p[i] = ys[i].data_ptr();
    vec = vec && aligned16(b.p[i]);
  }
  at::Tensor out = at::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  if (x.numel() == 0) return out;
  vec = vec && aligned16(out.data_ptr());
  const at::cuda::CUDAGuard guard(x.device());
  switch (x.scalar_type()) {
    case at::kBFloat16: dispatch_nb<__nv_bfloat16>(x, b, nb, vec, out); break;
    case at::kHalf: dispatch_nb<__half>(x, b, nb, vec, out); break;
    default: dispatch_nb<float>(x, b, nb, vec, out); break;
  }
  return out;
}

// gx = (1 - n) g: the gradient of x (fp32 product, one rounding to g's dtype)
at::Tensor backward(at::Tensor g, int64_t nb) {
  TORCH_CHECK(g.is_cuda(), "parallel_residual backward: a CUDA gradient");
  check_dtype(g);
  g = g.contiguous();
  at::Tensor gx = at::empty_like(g, g.options().memory_format(at::MemoryFormat::Contiguous));
  const int64_t n = g.numel();
  if (n == 0) return gx;
  const at::cuda::CUDAGuard guard(g.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float scale = (float)(1 - nb);
  const bool vec = aligned16(g.data_ptr()) && aligned16(gx.data_ptr());
  auto go = [&](auto tag) {
    using T = decltype(tag);
    const unsigned blocks = (unsigned)(vec ? vec_blocks(n, sizeof(T), kUnroll) : scalar_blocks(n));
    parallel_residual_bwd_k<T><<<blocks, kThreads, 0, stream>>>(reinterpret_cast<const T*>(g.data_ptr()), reinterpret_cast<T*>(gx.data_ptr()),
                                                               scale, n, vec ? 1 : 0);
  };
  switch (g.scalar_type()) {
    case at::kBFloat16: go(__nv_bfloat16{}); break;
    case at::kHalf: go(__half{}); break;
    default: go(float{}); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "out = x + sum_i (y_i - x), 1 to 4 branches, fp32 accumulation, one rounding", py::arg("x"), py::arg("ys"));
  m.def("backward", &backward, "gx = (1 - n) g", py::arg("g"), py::arg("n"));
}
