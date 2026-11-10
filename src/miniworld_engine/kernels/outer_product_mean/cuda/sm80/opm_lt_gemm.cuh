// opm_lt_gemm.cuh -- the three outer-product GEMMs of the OuterProductMean through cuBLASLt with a searched explicit algorithm.
//
//   mm(a, b, ta, tb) = op(a) @ op(b), row-major bf16 [M, N], fp32 accumulation, the same arithmetic as at::mm.
//
// cuBLAS's heuristic runs these GEMMs on kernels that are 1.3x slower than the best one at L128 (dA, dB) and 2-10 % slower at most other lengths, and at L256 it picks a split-K
// with a bf16 reduction (error 2.03e-3 against the fp32 product, 1.66e-3 for the algorithms below).  `opm_lt_table.inc` holds the explicit configuration of every (M, N, K,
// layout) of the registry shapes (S 1024 / 2048 / 4096, L 128 .. 768) that beat the default by >= 1.5 %, found by an exhaustive search (tile, stages, split-K with an fp32
// workspace reduction, swizzle, custom option) with an accuracy and repeatability gate; any other shape, or an algorithm this cuBLAS does not accept, runs at::mm.
#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/mm.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <tuple>

namespace opm_lt {

struct Entry { int64_t M, N, K; int ta, tb; int id, tile, stages, splitk, red, swz, custom; };
static const Entry kTable[] = {
#include "opm_lt_table.inc"
};

struct Plan {
  bool ok = false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t ws = 0;
};

inline const Entry* find(int64_t M, int64_t N, int64_t K, int ta, int tb) {
  for (const Entry& e : kTable)
    if (e.M == M && e.N == N && e.K == K && e.ta == ta && e.tb == tb) return &e;
  return nullptr;
}

inline Plan make_plan(const Entry& e, int64_t arows, int64_t acols, int64_t brows, int64_t bcols, int64_t M, int64_t N) {
  Plan p;
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto set = [&](cublasLtMatmulAlgoConfigAttributes_t at, int v) { uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&p.algo, at, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; };
  // the row-major product C = op(a) op(b) is, column-major, C^T (N x M) = op(b)^T op(a)^T: b is the first operand, a the second, both described by their column-major view
  cublasOperation_t o1 = e.tb ? CUBLAS_OP_T : CUBLAS_OP_N, o2 = e.ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return p;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &o1, sizeof(o1));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &o2, sizeof(o2));
  if (cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, bcols, brows, bcols) != CUBLAS_STATUS_SUCCESS) return p;
  if (cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, acols, arows, acols) != CUBLAS_STATUS_SUCCESS) return p;
  if (cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N) != CUBLAS_STATUS_SUCCESS) return p;
  if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, e.id, &p.algo) != CUBLAS_STATUS_SUCCESS) return p;
  if (!(set(CUBLASLT_ALGO_CONFIG_TILE_ID, e.tile) && set(CUBLASLT_ALGO_CONFIG_STAGES_ID, e.stages) && set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, e.splitk) &&
        set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, e.red) && set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, e.swz) && set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, e.custom)))
    return p;
  cublasLtMatmulHeuristicResult_t hr;
  if (cublasLtMatmulAlgoCheck(handle, p.op, p.la, p.lb, p.lc, p.lc, &p.algo, &hr) != CUBLAS_STATUS_SUCCESS) return p;
  p.ws = hr.workspaceSize;
  p.ok = true;
  return p;
}

// C = op(a) @ op(b); a, b contiguous bf16 CUDA matrices
inline at::Tensor mm(const at::Tensor& a, const at::Tensor& b, bool ta, bool tb) {
  const int64_t M = ta ? a.size(1) : a.size(0), K = ta ? a.size(0) : a.size(1), N = tb ? b.size(0) : b.size(1);
  const bool plain = a.is_cuda() && b.is_cuda() && a.dim() == 2 && b.dim() == 2 && a.is_contiguous() && b.is_contiguous() && a.scalar_type() == at::kBFloat16 &&
                     b.scalar_type() == at::kBFloat16 && (tb ? b.size(1) : b.size(0)) == K;
  const Entry* e = plain ? find(M, N, K, ta, tb) : nullptr;
  if (e == nullptr) return at::mm(ta ? a.t() : a, tb ? b.t() : b);
  static std::mutex mu;
  static std::map<std::tuple<int, int64_t, int64_t, int64_t, int, int>, Plan> plans;
  const at::cuda::CUDAGuard guard(a.device());
  const Plan* plan;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple((int)a.get_device(), M, N, K, e->ta, e->tb);
    auto it = plans.find(key);
    if (it == plans.end()) it = plans.emplace(key, make_plan(*e, a.size(0), a.size(1), b.size(0), b.size(1), M, N)).first;
    plan = &it->second;
  }
  if (!plan->ok) return at::mm(ta ? a.t() : a, tb ? b.t() : b);
  auto c = at::empty({M, N}, a.options());
  at::Tensor ws;
  if (plan->ws > 0) ws = at::empty({(int64_t)plan->ws}, a.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  const auto status = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->op, &alpha, b.data_ptr(), plan->la, a.data_ptr(), plan->lb, &beta, c.data_ptr(), plan->lc, c.data_ptr(),
                                     plan->lc, &plan->algo, plan->ws > 0 ? ws.data_ptr() : nullptr, plan->ws, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "OPM cuBLASLt GEMM failed: ", (int)status);
  return c;
}

}  // namespace opm_lt
