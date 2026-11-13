// lt_f32_sm120.cuh -- the fp32 inference block's GEMMs on the RTX 5090 (MWE_SM120 build only) through cuBLASLt with an explicit, measured algorithm.
//
//   lt_mm32(a, w, out, cfg)   out[b] = a[b] w[b]^T (row-major fp32, TF32 tensor cores: CUBLAS_COMPUTE_32F_FAST_TF32 -- the arithmetic of at::mm under allow_tf32),
//                             a [B, M, K] / [M, K], w [B, N, K] / [N, K], out [B, M, N] / [M, N] (a row-major view: any leading dimension); B > 1 a strided batch.
//   lt_algos32(...)           the heuristic's candidates for a shape: the configurations a search (integrations/bias_only_dit_sm80.py, _LT_F32) measures.
//
// cuBLAS's own pick for these shapes leaves most of a last wave of the 170 SMs idle (M 3840 x N 768: 180 tiles of 128 x 128) or takes a 16-deep K-block where a
// 32-deep one runs faster; the table names the (algorithm, tile, stages, split-K, reduction, swizzle, custom option) that measured fastest per shape. An algorithm
// this cuBLAS does not accept is an error (never a silent fallback).
#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <array>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace lt120 {

using Cfg = std::array<int64_t, 7>;  // algo id, tile, stages, split-K, reduction scheme, CTA swizzle, custom option

struct Plan {
  bool ok = false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t ws = 0;
};

// the row-major C [M, N] = A [M, K] W [N, K]^T is, column-major, C^T (N x M, ld ldc) = op(W^T) A^T with W^T (K x N, ld K) transposed and A^T (K x M, ld K) as is
inline bool layouts(Plan& p, int64_t B, int64_t M, int64_t N, int64_t K, int64_t ldc) {
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (cublasLtMatrixLayoutCreate(&p.la, CUDA_R_32F, K, N, K) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_32F, K, M, K) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, N, M, ldc) != CUBLAS_STATUS_SUCCESS) return false;
  if (B > 1) {
    const int32_t bc = (int32_t)B;
    const int64_t sa = N * K, sb = M * K, sc = M * ldc;
    cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa));
    cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb));
    cublasLtMatrixLayoutSetAttribute(p.lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    cublasLtMatrixLayoutSetAttribute(p.lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof(sc));
  }
  return true;
}

constexpr size_t WS_MAX = 32u << 20;

inline Cfg cfg_of(const cublasLtMatmulAlgo_t& a) {
  auto get = [&](cublasLtMatmulAlgoConfigAttributes_t at) {
    uint32_t v = 0;
    size_t n = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, sizeof(v), &n);
    return (int64_t)v;
  };
  return {get(CUBLASLT_ALGO_CONFIG_ID), get(CUBLASLT_ALGO_CONFIG_TILE_ID), get(CUBLASLT_ALGO_CONFIG_STAGES_ID), get(CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
          get(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME), get(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING), get(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)};
}

inline Plan make_plan(int64_t B, int64_t M, int64_t N, int64_t K, int64_t ldc, const Cfg& c) {
  Plan p;
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  if (!layouts(p, B, M, N, K, ldc)) return p;
  if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, (int)c[0], &p.algo) != CUBLAS_STATUS_SUCCESS) return p;
  auto set = [&](cublasLtMatmulAlgoConfigAttributes_t at, int64_t v) {
    uint32_t x = (uint32_t)v;
    return cublasLtMatmulAlgoConfigSetAttribute(&p.algo, at, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
  };
  if (!(set(CUBLASLT_ALGO_CONFIG_TILE_ID, c[1]) && set(CUBLASLT_ALGO_CONFIG_STAGES_ID, c[2]) && set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, c[3]) &&
        set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, c[4]) && set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, c[5]) && set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, c[6])))
    return p;
  cublasLtMatmulHeuristicResult_t hr;
  if (cublasLtMatmulAlgoCheck(handle, p.op, p.la, p.lb, p.lc, p.lc, &p.algo, &hr) != CUBLAS_STATUS_SUCCESS) return p;
  if (hr.workspaceSize > WS_MAX) return p;
  p.ws = hr.workspaceSize;
  p.ok = true;
  return p;
}

inline void dims(const at::Tensor& a, const at::Tensor& w, const at::Tensor& out, int64_t& B, int64_t& M, int64_t& N, int64_t& K, int64_t& ldc) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda() && a.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
              "lt_mm32: fp32 CUDA tensors");
  TORCH_CHECK(a.dim() == w.dim() && a.dim() == out.dim() && (a.dim() == 2 || a.dim() == 3), "lt_mm32: a, w, out all [M, K] / [N, K] / [M, N] or all batched");
  const bool bat = a.dim() == 3;
  B = bat ? a.size(0) : 1;
  M = a.size(-2);
  K = a.size(-1);
  N = w.size(-2);
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && w.size(-1) == K && (!bat || w.size(0) == B), "lt_mm32: contiguous a [B, M, K], w [B, N, K]");
  TORCH_CHECK(out.size(-2) == M && out.size(-1) == N && out.stride(-1) == 1 && (!bat || (out.size(0) == B && out.is_contiguous())), "lt_mm32: out [B, M, N] (2-D: a row-major view)");
  ldc = out.stride(-2);
}

// out = a w^T with the configuration cfg
inline void mm(at::Tensor a, at::Tensor w, at::Tensor out, std::vector<int64_t> cv) {
  TORCH_CHECK(cv.size() == 7, "lt_mm32: cfg = 7 ints");
  int64_t B, M, N, K, ldc;
  dims(a, w, out, B, M, N, K, ldc);
  const Cfg c{cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6]};
  static std::mutex mu;
  static std::map<std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, Cfg>, Plan> plans;
  const at::cuda::CUDAGuard guard(a.device());
  const Plan* plan;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple((int)a.get_device(), B, M, N, K, ldc, c);
    auto it = plans.find(key);
    if (it == plans.end()) it = plans.emplace(key, make_plan(B, M, N, K, ldc, c)).first;
    plan = &it->second;
  }
  TORCH_CHECK(plan->ok, "lt_mm32: cuBLASLt does not accept the configuration for B ", B, " M ", M, " N ", N, " K ", K);
  at::Tensor ws;
  if (plan->ws > 0) ws = at::empty({(int64_t)plan->ws}, a.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  const auto st = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->op, &alpha, w.data_ptr(), plan->la, a.data_ptr(), plan->lb, &beta, out.data_ptr(), plan->lc,
                                 out.data_ptr(), plan->lc, &plan->algo, plan->ws > 0 ? ws.data_ptr() : nullptr, plan->ws, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "lt_mm32: cuBLASLt GEMM failed: ", (int)st);
}

// up to `count` heuristic candidates [count][7] for the shape of (a, w, out)
inline std::vector<std::vector<int64_t>> algos(at::Tensor a, at::Tensor w, at::Tensor out, int64_t count) {
  int64_t B, M, N, K, ldc;
  dims(a, w, out, B, M, N, K, ldc);
  const at::cuda::CUDAGuard guard(a.device());
  Plan p;
  TORCH_CHECK(layouts(p, B, M, N, K, ldc), "lt_algos32: layouts");
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  const size_t wsm = WS_MAX;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsm, sizeof(wsm));
  std::vector<cublasLtMatmulHeuristicResult_t> r((size_t)count);
  int got = 0;
  cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.la, p.lb, p.lc, p.lc, pref, (int)count, r.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<std::vector<int64_t>> outv;
  for (int i = 0; i < got; ++i) {
    if (r[i].state != CUBLAS_STATUS_SUCCESS) continue;
    const Cfg c = cfg_of(r[i].algo);
    outv.emplace_back(c.begin(), c.end());
  }
  return outv;
}

// ---- the general form (the bf16 / master training's GEMMs): out [M, N] (row-major, any leading dimension) = op(a) op(b), a / b contiguous 2-D of one
// dtype (bf16: fp32 accumulation, CUBLAS_COMPUTE_32F; fp32: TF32), op = transpose when ta / tb; out bf16 or fp32 (bf16 operands) -- the arithmetic of
// at::mm (and of an fp32-output mm of bf16 operands: one rounding of the fp32 accumulator).
struct GPlan {
  bool ok = false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t ws = 0;
};
using GKey = std::tuple<int, int, int, int64_t, int64_t, int64_t, int64_t, int, int, Cfg>;

inline cudaDataType_t dt_of(const at::Tensor& t) { return t.scalar_type() == at::kBFloat16 ? CUDA_R_16BF : CUDA_R_32F; }

// the column-major view: C^T (N x M) = op(b)^T op(a)^T; b first (rows = its columns), a second
inline bool glayouts(GPlan& p, const at::Tensor& a, const at::Tensor& b, int64_t M, int64_t N, int64_t ldc, bool ta, bool tb, cudaDataType_t tc) {
  const cudaDataType_t ti = dt_of(a);
  const cublasComputeType_t ct = ti == CUDA_R_32F ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  const cublasOperation_t o1 = tb ? CUBLAS_OP_T : CUBLAS_OP_N, o2 = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (cublasLtMatmulDescCreate(&p.op, ct, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &o1, sizeof(o1));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &o2, sizeof(o2));
  if (cublasLtMatrixLayoutCreate(&p.la, ti, b.size(1), b.size(0), b.size(1)) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.lb, ti, a.size(1), a.size(0), a.size(1)) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.lc, tc, N, M, ldc) != CUBLAS_STATUS_SUCCESS) return false;
  return true;
}

inline void gdims(const at::Tensor& a, const at::Tensor& b, const at::Tensor& out, bool ta, bool tb, int64_t& M, int64_t& N, int64_t& K) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda() && a.dim() == 2 && b.dim() == 2 && out.dim() == 2 && a.is_contiguous() && b.is_contiguous() &&
                  a.scalar_type() == b.scalar_type() && (a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kFloat) &&
                  (out.scalar_type() == at::kFloat || out.scalar_type() == a.scalar_type()),
              "lt_mm: contiguous 2-D a, b of one dtype (bf16 / fp32), out fp32 or theirs");
  M = ta ? a.size(1) : a.size(0);
  K = ta ? a.size(0) : a.size(1);
  N = tb ? b.size(0) : b.size(1);
  TORCH_CHECK((tb ? b.size(1) : b.size(0)) == K && out.size(0) == M && out.size(1) == N && out.stride(1) == 1, "lt_mm: shapes");
}

inline void mmg(at::Tensor a, at::Tensor b, at::Tensor out, bool ta, bool tb, std::vector<int64_t> cv) {
  TORCH_CHECK(cv.size() == 7, "lt_mm: cfg = 7 ints");
  int64_t M, N, K;
  gdims(a, b, out, ta, tb, M, N, K);
  const Cfg c{cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6]};
  const int64_t ldc = out.stride(0);
  static std::mutex mu;
  static std::map<GKey, GPlan> plans;
  const at::cuda::CUDAGuard guard(a.device());
  const GPlan* plan;
  {
    std::lock_guard<std::mutex> lock(mu);
    const GKey key{(int)a.get_device(), (int)dt_of(a), (int)dt_of(out), M, N, K, ldc, (int)ta, (int)tb, c};
    auto it = plans.find(key);
    if (it == plans.end()) {
      GPlan p;
      auto handle = at::cuda::getCurrentCUDABlasLtHandle();
      const cudaDataType_t ti = dt_of(a), tc = dt_of(out);
      const cublasComputeType_t ct = ti == CUDA_R_32F ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
      if (glayouts(p, a, b, M, N, ldc, ta, tb, tc) &&
          cublasLtMatmulAlgoInit(handle, ct, CUDA_R_32F, ti, ti, tc, tc, (int)c[0], &p.algo) == CUBLAS_STATUS_SUCCESS) {
        auto set = [&](cublasLtMatmulAlgoConfigAttributes_t at, int64_t v) {
          uint32_t x = (uint32_t)v;
          return cublasLtMatmulAlgoConfigSetAttribute(&p.algo, at, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
        };
        cublasLtMatmulHeuristicResult_t hr;
        if (set(CUBLASLT_ALGO_CONFIG_TILE_ID, c[1]) && set(CUBLASLT_ALGO_CONFIG_STAGES_ID, c[2]) && set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, c[3]) &&
            set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, c[4]) && set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, c[5]) && set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, c[6]) &&
            cublasLtMatmulAlgoCheck(handle, p.op, p.la, p.lb, p.lc, p.lc, &p.algo, &hr) == CUBLAS_STATUS_SUCCESS && hr.workspaceSize <= WS_MAX) {
          p.ws = hr.workspaceSize;
          p.ok = true;
        }
      }
      it = plans.emplace(key, p).first;
    }
    plan = &it->second;
  }
  TORCH_CHECK(plan->ok, "lt_mm: cuBLASLt does not accept the configuration for M ", M, " N ", N, " K ", K);
  at::Tensor ws;
  if (plan->ws > 0) ws = at::empty({(int64_t)plan->ws}, a.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  const auto st = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->op, &alpha, b.data_ptr(), plan->la, a.data_ptr(), plan->lb, &beta, out.data_ptr(), plan->lc,
                                 out.data_ptr(), plan->lc, &plan->algo, plan->ws > 0 ? ws.data_ptr() : nullptr, plan->ws, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "lt_mm: cuBLASLt GEMM failed: ", (int)st);
}

inline std::vector<std::vector<int64_t>> algosg(at::Tensor a, at::Tensor b, at::Tensor out, bool ta, bool tb, int64_t count) {
  int64_t M, N, K;
  gdims(a, b, out, ta, tb, M, N, K);
  const at::cuda::CUDAGuard guard(a.device());
  GPlan p;
  TORCH_CHECK(glayouts(p, a, b, M, N, out.stride(0), ta, tb, dt_of(out)), "lt_algos: layouts");
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  const size_t wsm = WS_MAX;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsm, sizeof(wsm));
  std::vector<cublasLtMatmulHeuristicResult_t> r((size_t)count);
  int got = 0;
  cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.la, p.lb, p.lc, p.lc, pref, (int)count, r.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<std::vector<int64_t>> outv;
  for (int i = 0; i < got; ++i)
    if (r[i].state == CUBLAS_STATUS_SUCCESS) {
      const Cfg c = cfg_of(r[i].algo);
      outv.emplace_back(c.begin(), c.end());
    }
  return outv;
}

}  // namespace lt120
