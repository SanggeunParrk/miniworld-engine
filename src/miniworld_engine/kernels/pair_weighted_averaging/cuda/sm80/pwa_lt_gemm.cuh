// pwa_lt_gemm.cuh -- the batched products of the MSA pair-weighted averaging through cuBLASLt with a searched explicit algorithm.
//
//   bcast  o = w v  (op 0) and  dv = w^T dO  (op 1): the head-major tensors [8 ns, L, kp C] (the MSA rows in ns chunks, one batch each) times the [8, L, L] softmax weights of their
//          head, shared by the ns chunks of a head -- cuBLASLt's pointer-array batch mode takes the w pointer of every batch from a device array, so the `_expand` copy of w
//          (a 9 MB copy per product, 22-25 us at L384) never exists.
//   dw     dW = dO v^T (op 2), the weight-gradient product: [8 ns, L, kp C] x [8 ns, L, kp C]^T -> fp32 [8 ns, L, L] (strided batches; K = kp C is large).
//
// The cuBLAS heuristic runs these on kernels 3-35 % slower than the best one (dw: 1.2-1.4x at every length; o / dv: 1.0-1.1x at C32, up to 3x at small L and C).  `pwa_lt_table.inc`
// holds, for every registry shape, the explicit algorithm (tile, stages, swizzle; split-K never: an in-place bf16 reduction costs accuracy) found by an exhaustive search with
// an accuracy (<= 1.05x the default's error against the fp32 product) and repeatability gate, and confirmed in 11 interleaved rounds (median >= 1.015x).  A shape that is not in the
// table, or an algorithm this cuBLAS does not accept, returns false and the caller keeps its previous path.
#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <map>
#include <mutex>
#include <tuple>

namespace pwa_lt {

struct Entry { int op, B, L, N; int id, tile, stages, splitk, red, swz, custom; };
static const Entry kTable[] = {
#include "pwa_lt_table.inc"
};

struct Plan {
  bool ok = false;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t ws = 0;
};

inline const Entry* find(int op, int B, int L, int N) {
  for (const Entry& e : kTable)
    if (e.op == op && e.B == B && e.L == L && e.N == N) return &e;
  return nullptr;
}

__global__ void pointers_kernel(const __nv_bfloat16* w, const __nv_bfloat16* v, __nv_bfloat16* o, const void** wp, const void** vp, void** op, int B, int ns, long ww, long vv) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < B) { wp[b] = w + (long)(b / ns) * ww; vp[b] = v + (long)b * vv; op[b] = o + (long)b * vv; }
}

// column-major view of the row-major C = op(a) op(b): D (N x M) = op1(b_cm) op2(a_cm).  b_cm is (b.cols x b.rows), a_cm (a.cols x a.rows); batch: pointer array or strided
inline Plan make_plan(const Entry& e, int64_t arows, int64_t acols, int64_t brows, int64_t bcols, int64_t M, int64_t N, bool ta, bool tb, bool f32, bool ptrmode, long sa, long sb, long sc) {
  Plan p;
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto set = [&](cublasLtMatmulAlgoConfigAttributes_t at, int v) { uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&p.algo, at, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; };
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  cublasOperation_t o1 = tb ? CUBLAS_OP_T : CUBLAS_OP_N, o2 = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return p;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &o1, sizeof(o1));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &o2, sizeof(o2));
  if (cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, bcols, brows, bcols) != CUBLAS_STATUS_SUCCESS) return p;
  if (cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, acols, arows, acols) != CUBLAS_STATUS_SUCCESS) return p;
  if (cublasLtMatrixLayoutCreate(&p.lc, ct, N, M, N) != CUBLAS_STATUS_SUCCESS) return p;
  auto batch = [&](cublasLtMatrixLayout_t l, long stride) {
    int32_t bc = e.B; int64_t st = stride;
    if (ptrmode) { uint32_t mode = CUBLASLT_BATCH_MODE_POINTER_ARRAY; cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)); }
    cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    if (!ptrmode) cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st, sizeof(st));
  };
  batch(p.la, sb); batch(p.lb, sa); batch(p.lc, sc);
  if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, e.id, &p.algo) != CUBLAS_STATUS_SUCCESS) return p;
  if (!(set(CUBLASLT_ALGO_CONFIG_TILE_ID, e.tile) && set(CUBLASLT_ALGO_CONFIG_STAGES_ID, e.stages) && set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, e.splitk) &&
        set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, e.red) && set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, e.swz) && set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, e.custom)))
    return p;
  cublasLtMatmulHeuristicResult_t hr;
  if (cublasLtMatmulAlgoCheck(handle, p.desc, p.la, p.lb, p.lc, p.lc, &p.algo, &hr) != CUBLAS_STATUS_SUCCESS) return p;
  p.ws = hr.workspaceSize;
  p.ok = true;
  return p;
}

inline const Plan* cached(const Entry& e, int dev, int64_t arows, int64_t acols, int64_t brows, int64_t bcols, int64_t M, int64_t N, bool ta, bool tb, bool f32, bool ptrmode, long sa, long sb, long sc) {
  static std::mutex mu;
  static std::map<std::tuple<int, int, int, int, int>, Plan> plans;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_tuple(dev, e.op, e.B, e.L, e.N);
  auto it = plans.find(key);
  if (it == plans.end()) it = plans.emplace(key, make_plan(e, arows, acols, brows, bcols, M, N, ta, tb, f32, ptrmode, sa, sb, sc)).first;
  return &it->second;
}

// out = w[head] v (transpose: w^T v) for every batch b = (head, chunk); w [8, L, L], v and out [8 ns, L, N] bf16, contiguous.  false: not in the table (nothing was written)
inline bool bcast(const at::Tensor& w, const at::Tensor& v, at::Tensor& out, bool transpose) {
  const int B = (int)v.size(0), L = (int)v.size(1), N = (int)v.size(2), ns = B / 8;
  const Entry* e = find(transpose ? 1 : 0, B, L, N);
  if (e == nullptr || !w.is_cuda() || !v.is_cuda() || w.scalar_type() != at::kBFloat16 || v.scalar_type() != at::kBFloat16 || !w.is_contiguous() || !v.is_contiguous() || !out.is_contiguous() ||
      w.dim() != 3 || w.size(0) != 8 || w.size(1) != L || w.size(2) != L || B % 8 != 0 || out.sizes() != v.sizes())
    return false;
  const at::cuda::CUDAGuard guard(v.device());
  const Plan* plan = cached(*e, (int)v.get_device(), L, L, L, N, L, N, transpose, false, false, true, 0, 0, 0);
  if (!plan->ok) return false;
  auto ptrs = at::empty({3L * B}, v.options().dtype(at::kLong));
  const void** wp = reinterpret_cast<const void**>(ptrs.data_ptr());
  const void** vp = wp + B;
  void** op = reinterpret_cast<void**>(ptrs.data_ptr()) + 2L * B;
  pointers_kernel<<<(B + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
                                                                             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), wp, vp, op, B, ns, (long)L * L, (long)L * N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor ws;
  if (plan->ws > 0) ws = at::empty({(int64_t)plan->ws}, v.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  // first operand: the per-batch v (cols N, rows L); second: the shared w; the matrix arguments of a pointer-array batch are the device arrays of the pointers
  const auto status = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->desc, &alpha, vp, plan->la, wp, plan->lb, &beta, op, plan->lc, op, plan->lc, &plan->algo,
                                     plan->ws > 0 ? ws.data_ptr() : nullptr, plan->ws, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "PWA cuBLASLt batched GEMM failed: ", (int)status);
  return true;
}

// out[b] = d[b] v[b]^T in fp32; d, v [B, L, N] bf16, out [B, L, L] fp32, contiguous.  false: not in the table
inline bool dw(const at::Tensor& d, const at::Tensor& v, at::Tensor& out) {
  const int B = (int)d.size(0), L = (int)d.size(1), N = (int)d.size(2);
  const Entry* e = find(2, B, L, N);
  if (e == nullptr || !d.is_cuda() || d.scalar_type() != at::kBFloat16 || v.scalar_type() != at::kBFloat16 || out.scalar_type() != at::kFloat || !d.is_contiguous() || !v.is_contiguous() ||
      !out.is_contiguous() || v.sizes() != d.sizes() || out.dim() != 3 || out.size(0) != B || out.size(1) != L || out.size(2) != L)
    return false;
  const at::cuda::CUDAGuard guard(d.device());
  // C[B, L, L] = d (L x N) v^T (N x L): K = N; first operand v (transposed), second d
  const Plan* plan = cached(*e, (int)d.get_device(), L, N, L, N, L, L, false, true, true, false, (long)L * N, (long)L * N, (long)L * L);
  if (!plan->ok) return false;
  at::Tensor ws;
  if (plan->ws > 0) ws = at::empty({(int64_t)plan->ws}, d.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  const auto status = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->desc, &alpha, v.data_ptr(), plan->la, d.data_ptr(), plan->lb, &beta, out.data_ptr(), plan->lc, out.data_ptr(),
                                     plan->lc, &plan->algo, plan->ws > 0 ? ws.data_ptr() : nullptr, plan->ws, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "PWA cuBLASLt dw GEMM failed: ", (int)status);
  return true;
}

}  // namespace pwa_lt
