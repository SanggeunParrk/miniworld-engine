// Bidirectional bias-only TriangleAttention glue for H100 (sm_90a), next to the width-generic kernels of triattn_generic_sm90.cu (whose
// ln_fwd / gate_u / ln_bwd the same integration uses: kernels/integrations/bidir_triattn_h100.py). The tensors are channel planes
// [R, N] (one row per channel, N = L L pair tokens), the layout the cuBLAS GEMMs of the path read and write.
//
//   backward  gate_bwd       du -> do = du sigmoid(g), dg = du o sigmoid(g) (1 - sigmoid(g))     (u = sigmoid(g) o, the forward's gate_u)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return v;
}

// du [R, N] (row stride ld_du), g [R, N] (ld_g), o [R, N] (ld_o) -> dout [R, N] (contiguous), dg [R, N] (ld_dg)
__global__ void __launch_bounds__(256) gate_bwd_k(const bf16* __restrict__ du, const bf16* __restrict__ g, const bf16* __restrict__ o,
                                                  bf16* __restrict__ dout, bf16* __restrict__ dg, long R, long N, long ld_du, long ld_g,
                                                  long ld_o, long ld_dg) {
  const long nv = N >> 3, total = R * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv, c = (i - row * nv) * 8;
    float a[8], gv[8], ov[8], r[8], q[8];
    unpack8(*reinterpret_cast<const uint4*>(du + row * ld_du + c), a);
    unpack8(*reinterpret_cast<const uint4*>(g + row * ld_g + c), gv);
    unpack8(*reinterpret_cast<const uint4*>(o + row * ld_o + c), ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float s = 1.f / (1.f + __expf(-gv[e]));
      r[e] = a[e] * s;
      q[e] = a[e] * ov[e] * s * (1.f - s);
    }
    *reinterpret_cast<uint4*>(dout + row * N + c) = pack8(r);
    *reinterpret_cast<uint4*>(dg + row * ld_dg + c) = pack8(q);
  }
}

void check_rows(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.scalar_type() == torch::kBFloat16 && t.stride(1) == 1 && t.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              name, ": bf16 [R, N] rows, unit column stride, 16-byte aligned rows");
}
}  // namespace

// du, g, o: [R, N] -> do [R, N]; dg (preallocated [R, N] view, e.g. rows of the path's gradient planes) is written
torch::Tensor gate_bwd(torch::Tensor du, torch::Tensor g, torch::Tensor o, torch::Tensor dg) {
  check_rows(du, "du");
  check_rows(g, "g");
  check_rows(o, "o");
  check_rows(dg, "dg");
  const long R = du.size(0), N = du.size(1);
  TORCH_CHECK(N % 8 == 0 && g.size(0) == R && g.size(1) == N && o.size(0) == R && o.size(1) == N && dg.size(0) == R && dg.size(1) == N,
              "gate_bwd: matching [R, N], N % 8 == 0");
  auto dout = torch::empty({R, N}, du.options());
  const long work = R * (N / 8);
  const int grid = (int)std::min<long>((work + 255) / 256, 132L * 16);
  gate_bwd_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(du.data_ptr()), reinterpret_cast<const bf16*>(g.data_ptr()), reinterpret_cast<const bf16*>(o.data_ptr()),
      reinterpret_cast<bf16*>(dout.data_ptr()), reinterpret_cast<bf16*>(dg.data_ptr()), R, N, du.stride(0), g.stride(0), o.stride(0),
      dg.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dout;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gate_bwd", &gate_bwd); }
