// Bidirectional bias-only TriangleAttention glue for H100 (sm_90a), next to the width-generic kernels of triattn_generic_sm90.cu (whose
// ln_fwd / gate_u / ln_bwd the same integration uses: kernels/integrations/bidir_triattn_h100.py). The tensors are channel planes
// [R, N] (one row per channel, N = L L pair tokens), the layout the cuBLAS GEMMs of the path read and write.
//
//   backward  gate_bwd       du -> do = du sigmoid(g), dg = du o sigmoid(g) (1 - sigmoid(g))     (u = sigmoid(g) o, the forward's gate_u)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

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

// The bias planes of one sample [2 H rows (row stride ld), L L] -> the softmax logits [2 H, L, L] fp32: rows 0..H-1 (starting) as they are,
// rows H..2H-1 (ending) transposed, [h, i, k] = bias_e[h, k, i]; masked keys k -> -inf. One pass in place of float() / cat / transpose copy /
// masked_fill (the values are the same ones: a bf16 -> fp32 widening and a copy).
__global__ void __launch_bounds__(256) logits_k(const bf16* __restrict__ b, const bool* __restrict__ keep, float* __restrict__ out, int nstart, int L,
                                                long ld) {
  __shared__ float tile[32][33];
  const int r = blockIdx.z;
  const int a0 = blockIdx.y * 32, c0 = blockIdx.x * 32;  // out rows a (j or i), cols c (k)
  const bf16* src = b + (long)r * ld;
  float* dst = out + (long)r * L * L;
  const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
  if (r < nstart) {
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int a = a0 + ty + 8 * t, c = c0 + tx;
      if (a < L && c < L) dst[(long)a * L + c] = (keep == nullptr || keep[c]) ? __bfloat162float(src[(long)a * L + c]) : -INFINITY;
    }
  } else {
#pragma unroll
    for (int t = 0; t < 4; ++t) {  // read [k][i] coalesced along i
      const int k = c0 + ty + 8 * t, i = a0 + tx;
      tile[ty + 8 * t][tx] = (k < L && i < L) ? __bfloat162float(src[(long)k * L + i]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 4; ++t) {  // write [i][k] coalesced along k
      const int i = a0 + ty + 8 * t, k = c0 + tx;
      if (i < L && k < L) dst[(long)i * L + k] = (keep == nullptr || keep[k]) ? tile[tx][ty + 8 * t] : -INFINITY;
    }
  }
}

#define LT_CHECK(x) do { auto st_ = (x); TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cublasLt status ", int(st_), " at " #x); } while (0)
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; std::vector<cublasLtMatmulHeuristicResult_t> cands; };

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

// bias: [R rows, L L] plane rows (bf16, unit column stride; the first nstart rows starting, the rest ending) -> logits [R, L, L] fp32; keep: optional bool [L] key mask
torch::Tensor logits(torch::Tensor bias, c10::optional<torch::Tensor> keep, int64_t nstart, int64_t L) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kBFloat16 && bias.dim() == 2 && bias.size(1) == L * L && nstart >= 0 && nstart <= bias.size(0) &&
                  bias.stride(1) == 1,
              "logits: bf16 [2 H, L L] planes");
  const bool* kp = nullptr;
  if (keep.has_value()) {
    TORCH_CHECK(keep->is_cuda() && keep->scalar_type() == torch::kBool && keep->is_contiguous() && keep->numel() == L, "logits: bool [L] mask");
    kp = keep->data_ptr<bool>();
  }
  c10::cuda::CUDAGuard guard(bias.device());
  auto out = torch::empty({bias.size(0), L, L}, bias.options().dtype(torch::kFloat32));
  const dim3 grid((L + 31) / 32, (L + 31) / 32, bias.size(0));
  logits_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(bias.data_ptr()), kp, out.data_ptr<float>(), (int)nstart,
                                                               (int)L, bias.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// OPT-IN (BIDIR_RESIDUAL_LT, default off): out = x + u^T wo^T through cublasLt with D != C, without torch.addmm's copy of x into its output (a
// full read + write of the pair). NOT the bits of torch.addmm: its kernel (cuBLAS' nvjet "badd") rounds the product to bf16 and then adds x
// (two roundings), no cublasLt algorithm of the heuristic's list (up to 64 were compared) does that, and the epilogue rounds once
// (x + product in fp32) -- the more accurate value, but a different output. u: [K, M] channel planes (row stride ld, unit column
// stride), wo [C, K] row-major, x / out [M, C] contiguous. The algorithm is the heuristic's first (a plan per shape, never timed).
torch::Tensor out_proj(torch::Tensor u, torch::Tensor wo, torch::Tensor x, torch::Tensor out) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == torch::kBFloat16 && wo.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "out_proj: bf16 cuda");
  TORCH_CHECK(u.dim() == 2 && u.stride(1) == 1 && wo.dim() == 2 && wo.is_contiguous() && wo.size(1) == u.size(0) && x.is_contiguous() &&
                  out.is_contiguous() && x.size(1) == wo.size(0) && x.size(0) == u.size(1) && out.sizes() == x.sizes(),
              "out_proj: shapes");
  c10::cuda::CUDAGuard guard(u.device());
  const int64_t K = u.size(0), M = u.size(1), C = wo.size(0), ld = u.stride(0);
  static std::map<std::tuple<int, int64_t, int64_t, int64_t, int64_t>, LtPlan> plans;
  static std::mutex mu;
  auto lt = reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasLtHandle());
  auto stream = at::cuda::getCurrentCUDAStream();
  LtPlan p;
  {
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_tuple((int)u.get_device(), M, K, C, ld);
    auto it = plans.find(key);
    if (it == plans.end()) {
      LtPlan q;
      LT_CHECK(cublasLtMatmulDescCreate(&q.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t t = CUBLAS_OP_T;
      LT_CHECK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
      LT_CHECK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSB, &t, sizeof(t)));
      LT_CHECK(cublasLtMatrixLayoutCreate(&q.la, CUDA_R_16BF, K, C, K));   // wo row-major [C, K] = column-major [K, C]
      LT_CHECK(cublasLtMatrixLayoutCreate(&q.lb, CUDA_R_16BF, M, K, ld));  // u row-major [K, M] = column-major [M, K]
      LT_CHECK(cublasLtMatrixLayoutCreate(&q.lc, CUDA_R_16BF, C, M, C));   // out row-major [M, C] = column-major [C, M]
      cublasLtMatmulPreference_t pref;
      LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
      size_t wsmax = 32u << 20;
      LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
      cublasLtMatmulHeuristicResult_t h;
      int n = 0;
      LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, q.op, q.la, q.lb, q.lc, q.lc, pref, 1, &h, &n));
      cublasLtMatmulPreferenceDestroy(pref);
      TORCH_CHECK(n > 0, "out_proj: no cublasLt algorithm");
      q.cands = {h};
      it = plans.emplace(key, q).first;
    }
    p = it->second;
  }
  auto ws = torch::empty({(int64_t)p.cands[0].workspaceSize + 16}, u.options().dtype(torch::kByte));
  const float alpha = 1.f, beta = 1.f;
  LT_CHECK(cublasLtMatmul(lt, p.op, &alpha, wo.data_ptr(), p.la, u.data_ptr(), p.lb, &beta, x.data_ptr(), p.lc, out.data_ptr(), p.lc, &p.cands[0].algo,
                          ws.data_ptr(), p.cands[0].workspaceSize, stream));
  return out;
}

// The planes GEMM C[N, M] = w[N, K] y[M, K]^T (the module's [Wv; Wg; Wb] y^T, channel planes out) through cublasLt. torch.mm's choice is
// not the fastest algorithm of the heuristic's list for this shape (its output is byte-bound: 4 x 2 H rows of L L tokens), and every
// algorithm of the list that does not split K has the same arithmetic (fp32 accumulation over the one K loop, one rounding): the bits are
// the same. The plan of a shape is calibrated once, outside a graph capture, on the call's own operands: every candidate (up to 32) is timed
// and compared bit for bit with torch.mm, the fastest equal one is kept; none equal, or a shape not calibrated while a graph is being
// captured: torch.mm. The choice of algorithm may vary with timing noise; the output bits cannot (they equal torch.mm's).
namespace {
struct CPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; std::vector<cublasLtMatmulHeuristicResult_t> cands; int chosen; };
}  // namespace

torch::Tensor cgemm(torch::Tensor w, torch::Tensor y, torch::Tensor out) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && y.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16 &&
                  w.dim() == 2 && y.dim() == 2 && out.dim() == 2 && w.is_contiguous() && y.is_contiguous() && out.is_contiguous() && w.size(1) == y.size(1) &&
                  out.size(0) == w.size(0) && out.size(1) == y.size(0),
              "cgemm: bf16 w [N, K], y [M, K], out [N, M], contiguous");
  c10::cuda::CUDAGuard guard(w.device());
  const int64_t N = w.size(0), K = w.size(1), M = y.size(0);
  static std::map<std::tuple<int, int64_t, int64_t, int64_t>, CPlan> plans;
  static std::mutex mu;
  auto lt = reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasLtHandle());
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_tuple((int)w.get_device(), N, K, M);
  auto it = plans.find(key);
  if (it == plans.end()) {
    CPlan q;
    LT_CHECK(cublasLtMatmulDescCreate(&q.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tt = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSA, &tt, sizeof(tt)));  // A = y^T: col-major [K, M] -> [M, K]
    LT_CHECK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn)));  // B = w: col-major [K, N]
    LT_CHECK(cublasLtMatrixLayoutCreate(&q.la, CUDA_R_16BF, K, M, K));
    LT_CHECK(cublasLtMatrixLayoutCreate(&q.lb, CUDA_R_16BF, K, N, K));
    LT_CHECK(cublasLtMatrixLayoutCreate(&q.lc, CUDA_R_16BF, M, N, M));  // out row-major [N, M] = column-major [M, N]
    cublasLtMatmulPreference_t pref;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsmax = 32u << 20;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
    q.cands.resize(32);
    int n = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, q.op, q.la, q.lb, q.lc, q.lc, pref, 32, q.cands.data(), &n));
    cublasLtMatmulPreferenceDestroy(pref);
    q.cands.resize(n);
    q.chosen = -2;  // not calibrated yet
    it = plans.emplace(key, q).first;
  }
  CPlan& p = it->second;
  const float alpha = 1.f, beta = 0.f;
  auto run = [&](const cublasLtMatmulHeuristicResult_t& c, torch::Tensor& dst, torch::Tensor& ws) {
    return cublasLtMatmul(lt, p.op, &alpha, y.data_ptr(), p.la, w.data_ptr(), p.lb, &beta, dst.data_ptr(), p.lc, dst.data_ptr(), p.lc, &c.algo, ws.data_ptr(),
                          c.workspaceSize, stream);
  };
  if (p.chosen == -2) {
    if (cap != cudaStreamCaptureStatusNone) {
      torch::mm_out(out, w, y.t());
      return out;
    }
    auto ref = torch::mm(w, y.t());
    auto tmp = torch::empty_like(out);
    auto ws = torch::empty({(int64_t)(32u << 20)}, w.options().dtype(torch::kByte));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    p.chosen = -1;
    float best = 1e30f;
    for (int c = 0; c < (int)p.cands.size(); ++c) {
      if (p.cands[c].state != CUBLAS_STATUS_SUCCESS || run(p.cands[c], tmp, ws) != CUBLAS_STATUS_SUCCESS) continue;
      if (!at::equal(tmp, ref)) continue;
      cudaEventRecord(e0, stream);
      for (int r = 0; r < 6; ++r) run(p.cands[c], tmp, ws);
      cudaEventRecord(e1, stream);
      cudaEventSynchronize(e1);
      float ms = 0.f;
      cudaEventElapsedTime(&ms, e0, e1);
      if (ms < best) { best = ms; p.chosen = c; }
    }
    // torch.mm's own time: keep it when no candidate is faster
    cudaEventRecord(e0, stream);
    for (int r = 0; r < 6; ++r) torch::mm_out(tmp, w, y.t());
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (ms <= best) p.chosen = -1;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
  if (p.chosen < 0) {
    torch::mm_out(out, w, y.t());
    return out;
  }
  auto ws = torch::empty({(int64_t)p.cands[p.chosen].workspaceSize + 16}, w.options().dtype(torch::kByte));
  LT_CHECK(run(p.cands[p.chosen], out, ws));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_bwd", &gate_bwd);
  m.def("logits", &logits);
  m.def("out_proj", &out_proj);
  m.def("cgemm", &cgemm);
}
