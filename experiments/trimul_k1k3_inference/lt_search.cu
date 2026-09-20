// Enumerate cuBLASLt heuristic algorithms for the TriMul contraction  X[c] = A[c] * B[c]^T  (bf16 in, fp32 accumulate, bf16 out), batch = ch.
// Row-major torch view: A [ch][N][K], B [ch][N][K], X [ch][N][N].  In cuBLAS column-major terms X^T = B * A^T  ->  we express the same product
// as torch does: op(A)=T on a col-major view.  Times every returned algo with CUDA events over `iters` back-to-back calls (graph-free) and prints JSON.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "ERR %s -> %d at %d\n", #x, (int)e, __LINE__); exit(1);} } while (0)
int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 768, K = N, ch = argc > 2 ? atoi(argv[2]) : 128, iters = argc > 3 ? atoi(argv[3]) : 50;
  size_t wsz = argc > 4 ? (size_t)atol(argv[4]) : (size_t)32 << 20;
  __nv_bfloat16 *A, *B, *X; void* ws;
  CK(cudaMalloc(&A, (size_t)ch * N * K * 2)); CK(cudaMalloc(&B, (size_t)ch * N * K * 2)); CK(cudaMalloc(&X, (size_t)ch * N * N * 2)); CK(cudaMalloc(&ws, wsz));
  { std::vector<__nv_bfloat16> h((size_t)ch * N * K); for (size_t i = 0; i < h.size(); ++i) h[i] = __float2bfloat16((float)((i * 2654435761u) % 1000) / 500.f - 1.f);
    CK(cudaMemcpy(A, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); }
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  // torch.bmm(a, b^T): row-major C[N][N] = A[N][K] * B[N][K]^T.  Column-major equivalent: C^T (N x N, ld N) = B^T?  Use the standard trick:
  // row-major C = A*B^T  <=>  col-major C' (= C^T) = B * A^T  with B col-major [K x N] via transpose... simplest: describe col-major
  // Cc[N][N] = op(Ac) * op(Bc) with Ac = B viewed col-major [K][N] (ld K) transposed -> (N x K), Bc = A viewed col-major [K][N] (ld K) not transposed -> (K x N).
  // That yields Cc(i,j) = sum_k B[i][k] A[j][k] = C^T -> stored col-major == row-major C.  So: transa = T (on B), transb = N (on A).
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));   // B: K x N col-major (ld K), transposed in the op
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, N, K));   // A: K x N col-major (ld K)
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, N, N));
  int32_t bc = ch; int64_t sa = (int64_t)N * K, sc = (int64_t)N * N;
  for (auto l : {la, lb}) { CK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc))); CK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa))); }
  CK(cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc))); CK(cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof(sc)));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  const int MAXR = 64; std::vector<cublasLtMatmulHeuristicResult_t> res(MAXR); int nres = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, MAXR, res.data(), &nres));
  printf("{\"N\": %d, \"ch\": %d, \"nres\": %d, \"algos\": [\n", N, ch, nres);
  float alpha = 1.f, beta = 0.f; cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int r = 0; r < nres; ++r) {
    if (res[r].state != CUBLAS_STATUS_SUCCESS) continue;
    int algo_id = -1, tile = -1, stages = -1, splitk = -1, cta_swz = -1, custom = -1, cluster_m = -1, cluster_n = -1, inner = -1; size_t sz;
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &cta_swz, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(int), &sz);
    cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster_m, sizeof(int), &sz);
    // warm + time
    bool ok = true;
    for (int i = 0; i < 5; ++i) if (cublasLtMatmul(lt, op, &alpha, B, la, A, lb, &beta, X, lc, X, lc, &res[r].algo, ws, res[r].workspaceSize, 0) != CUBLAS_STATUS_SUCCESS) { ok = false; break; }
    if (!ok) { printf("  {\"r\": %d, \"failed\": true},\n", r); continue; }
    CK(cudaDeviceSynchronize());
    float best = 1e9f;
    for (int rep = 0; rep < 3; ++rep) {
      CK(cudaEventRecord(e0, 0));
      for (int i = 0; i < iters; ++i) cublasLtMatmul(lt, op, &alpha, B, la, A, lb, &beta, X, lc, X, lc, &res[r].algo, ws, res[r].workspaceSize, 0);
      CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms * 1000.f / iters);
    }
    printf("  {\"r\": %d, \"algo\": %d, \"tile\": %d, \"stages\": %d, \"splitk\": %d, \"swz\": %d, \"custom\": %d, \"inner\": %d, \"cluster\": %d, \"ws\": %zu, \"waves\": %.3f, \"us\": %.2f},\n",
           r, algo_id, tile, stages, splitk, cta_swz, custom, inner, cluster_m, res[r].workspaceSize, res[r].wavesCount, best);
    fflush(stdout);
  }
  printf("  null]}\n");
  // reference: what the heuristic's first pick gives via the same path (already r=0)
  return 0;
}
