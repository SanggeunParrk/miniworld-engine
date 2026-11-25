// epilogue_rows_sm120.cu -- the RTX 5090 (sm_120a) launcher of the OuterProductMean epilogue over a chunk of pair rows i (inference only).
//
// The grouped outer product O [(L 32), (L 32)] bf16 is 2 KiB a pair (14 GiB at L 2688, 18 GiB at L 3072): past L 2048 the 5090 forms it per chunk of rows,
// O_c = A[:, i0 32 : i1 32]^T B [(ci 32), (L 32)], and runs this epilogue on it: the same kernel and schedule as ``epilogue`` (variant 0), the grid's i axis covering
// the chunk's rows (the kernel's row index is local to the chunk: O, norm, residual and out point at the chunk's first row; the j extent is L).
// The sm_80 sources are untouched (this file is built for sm_120a only).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <set>
#include <utility>

#include "opm_epilogue_sm80.cuh"

namespace {

template <class K> void raise_smem(K kernel, int bytes) {
  static std::set<std::pair<const void*, int>> done;
  const std::pair<const void*, int> key{reinterpret_cast<const void*>(kernel), (int)at::cuda::current_device()};
  if (done.insert(key).second) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <int CZ, int NS, int NB> void launch_rows(const opm80::EpilogueParams& p, int rows, cudaStream_t st) {
  using C = opm80::EpCfg<CZ, NS, NB>;
  static_assert(C::SMEM <= 101376, "the RTX 5090's 99 KB of dynamic shared memory per block");
  raise_smem(opm80::opm_epilogue_kernel<CZ, NS, NB, false>, C::SMEM);
  dim3 grid(((p.L + C::BJ - 1) / C::BJ) * p.ntile, rows / C::BI);
  opm80::opm_epilogue_kernel<CZ, NS, NB, false><<<grid, C::NTHR, C::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// O [rows 32, >= L 32] bf16 (row stride O.stride(0)), norm fp32 [rows, L] or empty, wo bf16 [CZ, 1024], bias fp32 [CZ], residual bf16 [rows, L, CZ] or empty,
// out [rows, L, CZ] bf16; rows a multiple of 4.
void epilogue_rows(torch::Tensor O, torch::Tensor norm, double norm_const, torch::Tensor wo, torch::Tensor bias, torch::Tensor residual, torch::Tensor out,
                   int64_t norm_first, double norm_min, double norm_add) {
  TORCH_CHECK(O.is_cuda() && O.scalar_type() == at::kBFloat16 && O.dim() == 2 && O.stride(1) == 1 && O.stride(0) % 8 == 0, "O: bf16 [rows 32, L 32] rows");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 3, "out: contiguous bf16 [rows, L, CZ]");
  TORCH_CHECK(wo.scalar_type() == at::kBFloat16 && wo.is_contiguous() && bias.scalar_type() == at::kFloat && bias.is_contiguous(), "wo bf16 / bias fp32, contiguous");
  const int rows = (int)out.size(0), L = (int)out.size(1), CZ = (int)out.size(2);
  TORCH_CHECK(rows % 4 == 0 && O.size(0) == (int64_t)rows * 32 && O.size(1) >= (int64_t)L * 32, "O / out shapes (rows a multiple of 4)");
  TORCH_CHECK(wo.sizes() == torch::IntArrayRef({CZ, 1024}) && bias.numel() == CZ, "wo / bias");
  TORCH_CHECK(norm.numel() == 0 || (norm.is_cuda() && norm.scalar_type() == at::kFloat && norm.is_contiguous() && norm.numel() == (int64_t)rows * L), "norm: fp32 [rows, L]");
  TORCH_CHECK(residual.numel() == 0 || (residual.is_cuda() && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() && residual.numel() == out.numel()), "residual");
  const at::cuda::CUDAGuard guard(O.device());
  opm80::EpilogueParams p;
  p.O = reinterpret_cast<const __nv_bfloat16*>(O.data_ptr()); p.norm = norm.numel() ? norm.data_ptr<float>() : nullptr;
  p.wo = reinterpret_cast<const __nv_bfloat16*>(wo.data_ptr()); p.bias = bias.data_ptr<float>();
  p.residual = residual.numel() ? reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()) : nullptr; p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldo = O.stride(0); p.L = L; p.norm_const = (float)norm_const; p.norm_first = (int)norm_first; p.cz = CZ; p.ntile = 1;
  p.norm_min = (float)norm_min; p.norm_add = (float)norm_add;
  auto st = at::cuda::getCurrentCUDAStream();
  if (CZ == 128 || CZ == 256 || CZ == 384 || CZ == 512) {  // epilogue's variant 0: column tiles of 128 channels on the d_pair 128 CTA, two per SM
    p.ntile = CZ / 128;
    launch_rows<128, 2, 2>(p, rows, st);
  } else {
    TORCH_CHECK(false, "epilogue_rows: d_pair must be 128, 256, 384 or 512");
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) { mod.def("epilogue_rows", &epilogue_rows); }
