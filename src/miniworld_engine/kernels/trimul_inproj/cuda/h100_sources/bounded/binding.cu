// SPDX-License-Identifier: Apache-2.0
// The bounded TriMul contraction's binding (gemm.cuh; one TU per config: cfg<i>.cu).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
void mwe_bounded_cfg0(const void*, const void*, void*, int, int, int, const int*, int, int, int, int, cudaStream_t);
void mwe_bounded_cfg1(const void*, const void*, void*, int, int, int, const int*, int, int, int, int, cudaStream_t);
void mwe_bounded_cfg2(const void*, const void*, void*, int, int, int, const int*, int, int, int, int, cudaStream_t);
void mwe_bounded_cfg3(const void*, const void*, void*, int, int, int, const int*, int, int, int, int, cudaStream_t);
void mwe_bounded_cfg4(const void*, const void*, void*, int, int, int, const int*, int, int, int, int, cudaStream_t);
void mwe_bounded_cfg5(const void*, const void*, void*, int, int, int, const int*, int, int, int, int, cudaStream_t);

// a, b: contiguous bf16 planes [P, n, n] as stored (mode 0: t = a b^T per plane, mode 1: t = a^T b); out [P, n, n]; bounds: int32
// [nsamples, 3] (live rows, columns, k) on the device or None (the full product); plane l belongs to sample l % nsamples.
void contract(torch::Tensor a, torch::Tensor b, torch::Tensor out, int64_t mode, int64_t cfg, c10::optional<torch::Tensor> bounds,
              int64_t nsamples, int64_t swizzle, int64_t raster) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && a.dim() == 3 && a.size(1) == a.size(2),
              "bounded contraction: a contiguous bf16 [P, n, n]");
  TORCH_CHECK(b.sizes() == a.sizes() && b.is_contiguous() && b.scalar_type() == at::kBFloat16, "bounded contraction: b like a");
  TORCH_CHECK(out.sizes() == a.sizes() && out.is_contiguous() && out.scalar_type() == at::kBFloat16, "bounded contraction: out like a");
  TORCH_CHECK(a.size(1) % 8 == 0, "bounded contraction: n % 8 == 0");
  const int* bd = nullptr;
  if (bounds.has_value()) {
    TORCH_CHECK(bounds->is_cuda() && bounds->scalar_type() == at::kInt && bounds->is_contiguous() && nsamples >= 1 &&
                bounds->numel() == 3 * nsamples, "bounded contraction: bounds int32 [nsamples, 3]");
    TORCH_CHECK(a.size(0) % nsamples == 0, "bounded contraction: planes a multiple of nsamples");
    bd = bounds->data_ptr<int>();
  }
  const at::cuda::CUDAGuard guard(a.device());
  const int dev = a.device().index();
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const int n = (int)a.size(1), planes = (int)a.size(0);
  switch (cfg) {
    case 0: mwe_bounded_cfg0(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, planes, (int)mode, bd, (int)nsamples, (int)swizzle, (int)raster, dev, s); break;
    case 1: mwe_bounded_cfg1(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, planes, (int)mode, bd, (int)nsamples, (int)swizzle, (int)raster, dev, s); break;
    case 2: mwe_bounded_cfg2(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, planes, (int)mode, bd, (int)nsamples, (int)swizzle, (int)raster, dev, s); break;
    case 3: mwe_bounded_cfg3(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, planes, (int)mode, bd, (int)nsamples, (int)swizzle, (int)raster, dev, s); break;
    case 4: mwe_bounded_cfg4(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, planes, (int)mode, bd, (int)nsamples, (int)swizzle, (int)raster, dev, s); break;
    case 5: mwe_bounded_cfg5(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, planes, (int)mode, bd, (int)nsamples, (int)swizzle, (int)raster, dev, s); break;
    default: TORCH_CHECK(false, "bounded contraction: config ", cfg);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("contract", &contract); m.attr("configs") = 6; }
