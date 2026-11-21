// B200 (sm_100a): tuned cuBLASLt bf16 -> fp32 GEMMs for the bias-only training backward's tall-K contractions (dP, dWo, dW).
// torch.mm / torch.bmm take the heuristic's first algorithm; for K = 16384 (dP, 4 heads of [L, K] x [K, L]) and K = L L (dWo, dW: [128, L L] x [L L, 128])
// that algorithm leaves SMs idle. Here the first call of a shape (outside a graph capture) times the heuristic's 32 candidates on the call's own
// operands, drops every one that is not deterministic (two runs bitwise equal) or not accurate (max error against the fp32 product), keeps the fastest.
// A shape not calibrated while a graph is captured falls back to torch (the module's warm-up always calibrates first).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#define LT_CHECK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "cublasLt error ", (int)s_, " at ", #x); } while (0)
namespace {
struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t lf, ls, lc; std::vector<cublasLtMatmulHeuristicResult_t> cands; int chosen = -2; };
using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int, int, int64_t, int64_t>;
// The calibration reference and the torch fallbacks are plain fp32 products: the measurer allows TF32 for the cuda backend, and a TF32 reference (or fallback)
// has an error of ~1e-3 of the largest value at K = L L, which rejected every candidate at L768 (chosen = -1: the bf16 operands cast to fp32 and a TF32 GEMM, +1.4 ms of the step).
struct Tf32Off { bool prev; Tf32Off() : prev(at::globalContext().allowTF32CuBLAS()) { at::globalContext().setAllowTF32CuBLAS(false); } ~Tf32Off() { at::globalContext().setAllowTF32CuBLAS(prev); } };
}
// a: logical [(B,) M, K], b: logical [(B,) K, N] bf16, each row-major or the transpose of a row-major; -> out fp32 [(B,) M, N] row-major
torch::Tensor lt_mm(torch::Tensor a, torch::Tensor b, torch::Tensor fallback_dummy) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16 && a.dim() == b.dim() && (a.dim() == 2 || a.dim() == 3), "lt_mm: bf16 2-D / 3-D");
  c10::cuda::CUDAGuard guard(a.device());
  Tf32Off tf32_off;
  const bool batched = a.dim() == 3;
  const int64_t B = batched ? a.size(0) : 1, M = a.size(-2), K = a.size(-1), N = b.size(-1);
  TORCH_CHECK(b.size(-2) == K && (!batched || b.size(0) == B), "lt_mm: shapes");
  const bool a_rm = a.stride(-1) == 1, b_rm = b.stride(-1) == 1;
  TORCH_CHECK(a_rm || a.stride(-2) == 1, "lt_mm: a layout"); TORCH_CHECK(b_rm || b.stride(-2) == 1, "lt_mm: b layout");
  const int64_t lda = a_rm ? a.stride(-2) : a.stride(-1), ldb = b_rm ? b.stride(-2) : b.stride(-1);
  const int64_t sa = batched ? a.stride(0) : 0, sb = batched ? b.stride(0) : 0;
  auto out = torch::empty(batched ? std::vector<int64_t>{B, M, N} : std::vector<int64_t>{M, N}, a.options().dtype(torch::kFloat32));
  static std::map<Key, Plan> plans; static std::mutex mu;
  auto lt = reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasLtHandle());
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone; C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
  auto ref_mm = [&]() { // the torch result: fp32 product of the bf16 operands
    if (batched) return torch::bmm(a.to(torch::kFloat32), b.to(torch::kFloat32));
    return torch::mm(a.to(torch::kFloat32), b.to(torch::kFloat32)); };
  std::lock_guard<std::mutex> lk(mu);
  const Key key{a.get_device(), B, M, N, K, a_rm, b_rm, lda * 1000003 + ldb, sa * 31 + sb};
  auto it = plans.find(key);
  if (it == plans.end()) {
    Plan q;
    LT_CHECK(cublasLtMatmulDescCreate(&q.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t nn = CUBLAS_OP_N, tt = CUBLAS_OP_T;
    const cublasOperation_t opf = b_rm ? nn : tt, ops = a_rm ? nn : tt;   // first = logical b, second = logical a: C^T = b^T a^T
    LT_CHECK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSA, &opf, sizeof(opf)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSB, &ops, sizeof(ops)));
    if (b_rm) LT_CHECK(cublasLtMatrixLayoutCreate(&q.lf, CUDA_R_16BF, N, K, ldb)); else LT_CHECK(cublasLtMatrixLayoutCreate(&q.lf, CUDA_R_16BF, K, N, ldb));
    if (a_rm) LT_CHECK(cublasLtMatrixLayoutCreate(&q.ls, CUDA_R_16BF, K, M, lda)); else LT_CHECK(cublasLtMatrixLayoutCreate(&q.ls, CUDA_R_16BF, M, K, lda));
    LT_CHECK(cublasLtMatrixLayoutCreate(&q.lc, CUDA_R_32F, N, M, N));
    if (batched) {
      int32_t bc = (int32_t)B; int64_t so_f = sb, so_s = sa, so_c = M * N;
      for (auto* l : {&q.lf, &q.ls, &q.lc}) LT_CHECK(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
      LT_CHECK(cublasLtMatrixLayoutSetAttribute(q.lf, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so_f, sizeof(so_f)));
      LT_CHECK(cublasLtMatrixLayoutSetAttribute(q.ls, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so_s, sizeof(so_s)));
      LT_CHECK(cublasLtMatrixLayoutSetAttribute(q.lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so_c, sizeof(so_c)));
    }
    cublasLtMatmulPreference_t pref; LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsmax = 64u << 20; LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
    q.cands.resize(32); int n = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, q.op, q.lf, q.ls, q.lc, q.lc, pref, 32, q.cands.data(), &n));
    cublasLtMatmulPreferenceDestroy(pref); q.cands.resize(n);
    it = plans.emplace(key, q).first;
  }
  Plan& p = it->second;
  const float alpha = 1.f, beta = 0.f;
  auto run = [&](const cublasLtMatmulHeuristicResult_t& c, torch::Tensor& dst, torch::Tensor& ws) {
    return cublasLtMatmul(lt, p.op, &alpha, b.data_ptr(), p.lf, a.data_ptr(), p.ls, &beta, dst.data_ptr(), p.lc, dst.data_ptr(), p.lc, &c.algo, ws.data_ptr(), c.workspaceSize, stream); };
  if (p.chosen == -2) {
    if (cap != cudaStreamCaptureStatusNone) return torch::empty({0}, a.options().dtype(torch::kFloat32));  // caller falls back to torch.mm(a, b, out_dtype=float32): bf16 operands
    auto ref = ref_mm(); const float rmax = ref.abs().max().item<float>();
    auto t1 = torch::empty_like(out), t2 = torch::empty_like(out);
    auto ws = torch::empty({(int64_t)(64u << 20)}, a.options().dtype(torch::kByte));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    p.chosen = -1; float best = 1e30f;
    for (int c = 0; c < (int)p.cands.size(); ++c) {
      if (p.cands[c].state != CUBLAS_STATUS_SUCCESS || run(p.cands[c], t1, ws) != CUBLAS_STATUS_SUCCESS) continue;
      if (run(p.cands[c], t2, ws) != CUBLAS_STATUS_SUCCESS || !at::equal(t1, t2)) continue;          // deterministic
      if ((t1 - ref).abs().max().item<float>() > 1e-4f * rmax) continue;                                // accurate
      cudaEventRecord(e0, stream); for (int r = 0; r < 8; ++r) run(p.cands[c], t1, ws); cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
      float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
      if (ms < best) { best = ms; p.chosen = c; }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  if (p.chosen < 0) return torch::empty({0}, a.options().dtype(torch::kFloat32));  // caller falls back to torch.mm(a, b, out_dtype=float32): bf16 operands
  auto ws = torch::empty({(int64_t)p.cands[p.chosen].workspaceSize + 16}, a.options().dtype(torch::kByte));
  LT_CHECK(run(p.cands[p.chosen], out, ws));
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("lt_mm", &lt_mm, "tuned cublasLt bf16 -> fp32 GEMM"); }
