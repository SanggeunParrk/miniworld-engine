// cuBLASLt algorithm sweep for the OPM product O[(i,c),(j,e)] = A2 . BT^T (bf16 in, bf16 or fp32 out, fp32 accumulate):
// lt_algos(a2, bt, fp32_out, n) -> the heuristic's first n algorithms (their config attributes); lt_run(a2, bt, out, idx) runs one.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>

static std::vector<cublasLtMatmulHeuristicResult_t> g_algos;
static cublasLtMatmulDesc_t g_op = nullptr;
static cublasLtMatrixLayout_t g_a = nullptr, g_b = nullptr, g_c = nullptr;
static size_t g_ws = 64ull << 20;
static torch::Tensor g_wsbuf;

static void ck(cublasStatus_t s, const char* w) { TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLt ", w, " status ", int(s)); }

std::vector<std::vector<int64_t>> lt_algos(torch::Tensor a2, torch::Tensor bt, bool fp32_out, int64_t n) {
  const int64_t M = a2.size(0), K = a2.size(1), N = bt.size(0);
  if (g_op) { cublasLtMatmulDescDestroy(g_op); cublasLtMatrixLayoutDestroy(g_a); cublasLtMatrixLayoutDestroy(g_b); cublasLtMatrixLayoutDestroy(g_c); }
  ck(cublasLtMatmulDescCreate(&g_op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  ck(cublasLtMatmulDescSetAttribute(g_op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "ta");
  ck(cublasLtMatmulDescSetAttribute(g_op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "tb");
  ck(cublasLtMatrixLayoutCreate(&g_a, CUDA_R_16BF, K, N, K), "la");   // bt: col-major K x N, op T -> N x K
  ck(cublasLtMatrixLayoutCreate(&g_b, CUDA_R_16BF, K, M, K), "lb");   // a2: col-major K x M
  ck(cublasLtMatrixLayoutCreate(&g_c, fp32_out ? CUDA_R_32F : CUDA_R_16BF, N, M, N), "lc");   // O^T col-major N x M == O row-major
  cublasLtMatmulPreference_t pref;
  ck(cublasLtMatmulPreferenceCreate(&pref), "pref");
  ck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws, sizeof(g_ws)), "ws");
  g_algos.assign(n, {});
  int found = 0;
  ck(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), g_op, g_a, g_b, g_c, g_c, pref, (int)n, g_algos.data(), &found), "heur");
  cublasLtMatmulPreferenceDestroy(pref);
  g_algos.resize(found);
  g_wsbuf = torch::empty({(int64_t)g_ws}, a2.options().dtype(torch::kByte));
  std::vector<std::vector<int64_t>> out;
  for (auto& h : g_algos) {
    std::vector<int64_t> row;
    const cublasLtMatmulAlgoConfigAttributes_t at[] = {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID};
    for (auto a : at) {
      uint64_t v = 0; size_t w = 0;
      if (cublasLtMatmulAlgoConfigGetAttribute(&h.algo, a, &v, sizeof(v), &w) != CUBLAS_STATUS_SUCCESS) v = (uint64_t)-1;
      uint64_t mask = w >= 8 ? ~0ull : ((1ull << (8 * w)) - 1);
      row.push_back((int64_t)(v & mask));
    }
    row.push_back((int64_t)h.workspaceSize);
    row.push_back((int64_t)(h.wavesCount * 1000));
    out.push_back(row);
  }
  return out;
}

void lt_run(torch::Tensor a2, torch::Tensor bt, torch::Tensor out, int64_t idx) {
  float alpha = 1.f, beta = 0.f;
  ck(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), g_op, &alpha, bt.data_ptr(), g_a, a2.data_ptr(), g_b, &beta, out.data_ptr(), g_c,
                    out.data_ptr(), g_c, &g_algos[idx].algo, g_wsbuf.data_ptr(), g_ws, at::cuda::getCurrentCUDAStream()), "matmul");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lt_algos", &lt_algos);
  m.def("lt_run", &lt_run);
}
