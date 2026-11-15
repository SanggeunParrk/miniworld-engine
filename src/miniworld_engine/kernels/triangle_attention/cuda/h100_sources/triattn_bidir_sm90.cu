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

// ===================================================================================== fused tail (OPT-IN, inference only)
// out = x + bf16(bf16(sigmoid(G) o O)^T Wo^T) in one pass over the planes: the gate (gate_u_k's arithmetic, u rounded to bf16) is formed in
// shared memory and feeds mma.sync m16n8k16 straight away, so u (K x L L bf16: written by gate_u, read by the out-projection) never exists.
// The product is rounded to bf16 and then x is added (the double rounding of torch.addmm's nvjet kernel, so the epilogue is the same); the
// fp32 accumulation runs over the K = 2 Dh channels of one thread's k16 steps in order, which is not guaranteed to be cuBLAS' order: the
// output is NOT bit-identical to the unfused tail (rare 1-ulp differences of the accumulation order), hence opt-in.
// One CTA = 128 tokens x all 128 pair channels, 8 warps x 16 tokens; Wo resident in smem, the planes streamed 32 channels at a time
// through a 2-stage cp.async ring. Needs d_pair == 128, K in {128, 256}, L L % 128 == 0.
namespace tl {
constexpr int TM = 128, KC = 32, SP = 136;
__device__ __forceinline__ void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
}  // namespace tl

template <int K>
__global__ void __launch_bounds__(256, 2) tail_k(const bf16* __restrict__ G, const bf16* __restrict__ O, const bf16* __restrict__ Wo, const bf16* __restrict__ X,
                                                 bf16* __restrict__ out, long LL, long ldg, long ldo) {
  using namespace tl;
  constexpr int WS = K + 8, NK = K / KC, STG = 2 * KC * SP;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sWo = reinterpret_cast<bf16*>(smraw);  // [128][WS]
  bf16* sT = sWo + 128 * WS;                   // 2 stages x { G [KC][SP], O [KC][SP] }; reused as the epilogue tile [TM][SP]
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  for (int i = tid; i < 128 * (K / 8); i += 256) {
    const int n = i / (K / 8), c = i % (K / 8);
    cp16(sWo + n * WS + c * 8, Wo + (long)n * K + c * 8);
  }
  cp_commit();
  const long ntiles = LL / TM;
  for (long tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const long tok0 = tile * TM;
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    auto issue = [&](int kc, int s) {
      bf16* g = sT + s * STG;
      bf16* o = g + KC * SP;
      for (int i = tid; i < KC * 16; i += 256) {
        const int r = i >> 4, ch = i & 15;
        cp16(g + r * SP + ch * 8, G + (long)(kc * KC + r) * ldg + tok0 + ch * 8);
        cp16(o + r * SP + ch * 8, O + (long)(kc * KC + r) * ldo + tok0 + ch * 8);
      }
      cp_commit();
    };
    issue(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
      if (kc + 1 < NK) issue(kc + 1, (kc + 1) & 1);
      else cp_commit();
      cp_wait<1>();
      __syncthreads();
      bf16* g = sT + (kc & 1) * STG;
      bf16* o = g + KC * SP;
      for (int i = tid; i < KC * 16; i += 256) {  // u = bf16(sigmoid(g) o), in place of o
        const int r = i >> 4, ch = i & 15;
        float gv[8], ov[8], rv[8];
        unpack8(*reinterpret_cast<const uint4*>(g + r * SP + ch * 8), gv);
        unpack8(*reinterpret_cast<const uint4*>(o + r * SP + ch * 8), ov);
#pragma unroll
        for (int e = 0; e < 8; ++e) rv[e] = (1.f / (1.f + __expf(-gv[e]))) * ov[e];
        *reinterpret_cast<uint4*>(o + r * SP + ch * 8) = pack8(rv);
      }
      __syncthreads();
#pragma unroll
      for (int ks = 0; ks < KC / 16; ++ks) {
        unsigned a[4];
        {
          const int mi = lane >> 3;
          ldsm4t(a, o + (ks * 16 + (mi >> 1) * 8 + (lane & 7)) * SP + w * 16 + (mi & 1) * 8);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          unsigned b[4];
          const int mi = lane >> 3;
          ldsm4(b, sWo + (j * 16 + (mi >> 1) * 8 + (lane & 7)) * WS + kc * KC + ks * 16 + (mi & 1) * 8);
          mma16816(acc[2 * j], a, b[0], b[1]);
          mma16816(acc[2 * j + 1], a, b[2], b[3]);
        }
      }
      __syncthreads();
    }
    // epilogue: bf16(acc) into smem, then x added in fp32 and rounded again (coalesced 16-byte rows)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int c = j * 8 + 2 * (lane & 3), r0 = w * 16 + (lane >> 2);
      *reinterpret_cast<__nv_bfloat162*>(sT + r0 * SP + c) = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
      *reinterpret_cast<__nv_bfloat162*>(sT + (r0 + 8) * SP + c) = __floats2bfloat162_rn(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    for (int i = tid; i < TM * 16; i += 256) {
      const int r = i >> 4, ch = i & 15;
      float p[8], xf[8], q[8];
      unpack8(*reinterpret_cast<const uint4*>(sT + r * SP + ch * 8), p);
      unpack8(*reinterpret_cast<const uint4*>(X + (tok0 + r) * 128 + ch * 8), xf);
#pragma unroll
      for (int e = 0; e < 8; ++e) q[e] = p[e] + xf[e];
      *reinterpret_cast<uint4*>(out + (tok0 + r) * 128 + ch * 8) = pack8(q);
    }
    __syncthreads();
  }
}

// g: [K, LL] gate planes (row stride ldg), o: [K, LL] (row stride ldo), wo [128, K], x / out [LL, 128]
torch::Tensor tail(torch::Tensor g, torch::Tensor o, torch::Tensor wo, torch::Tensor x, torch::Tensor out) {
  const long K = wo.size(1), LL = g.size(1);
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == torch::kBFloat16 && o.scalar_type() == torch::kBFloat16 && wo.scalar_type() == torch::kBFloat16 &&
                  x.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16 && (K == 128 || K == 256) && wo.size(0) == 128 &&
                  wo.is_contiguous() && g.size(0) == K && o.size(0) == K && o.size(1) == LL && g.stride(1) == 1 && o.stride(1) == 1 &&
                  g.stride(0) % 8 == 0 && o.stride(0) % 8 == 0 && LL % 128 == 0 && x.is_contiguous() && out.is_contiguous() && x.size(0) == LL &&
                  x.size(1) == 128 && out.sizes() == x.sizes() && reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0,
              "tail: bf16 planes [K, LL], wo [128, K], x / out [LL, 128], K in {128, 256}, LL % 128 == 0, 16-byte aligned");
  c10::cuda::CUDAGuard guard(g.device());
  static int sms = 0;
  if (sms == 0) sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long ntiles = LL / 128;
  const int grid = (int)std::min<long>(ntiles, 2L * sms);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, size_t smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kern<<<grid, 256, smem, stream>>>(reinterpret_cast<const bf16*>(g.data_ptr()), reinterpret_cast<const bf16*>(o.data_ptr()),
                                      reinterpret_cast<const bf16*>(wo.data_ptr()), reinterpret_cast<const bf16*>(x.data_ptr()),
                                      reinterpret_cast<bf16*>(out.data_ptr()), LL, g.stride(0), o.stride(0));
  };
  if (K == 256) go(tail_k<256>, (size_t)(128 * 264 + 2 * 2 * 32 * 136) * 2);
  else go(tail_k<128>, (size_t)(128 * 136 + 2 * 2 * 32 * 136) * 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The module's fp32 masters -> the path's bf16 weights in one launch: w = [Wv; Wg; Wb] (rows concatenated) and wo, each rounded to nearest
// even (the bits of ``.to(torch.bfloat16)`` and ``torch.cat``: five small launches there).
__global__ void __launch_bounds__(256) cast_w_k(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, const float* __restrict__ d,
                                                bf16* __restrict__ w, bf16* __restrict__ wo, long na, long nb, long nc, long nd) {
  const long total = na + nb + nc + nd;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    if (i < na) w[i] = __float2bfloat16_rn(a[i]);
    else if (i < na + nb) w[i] = __float2bfloat16_rn(b[i - na]);
    else if (i < na + nb + nc) w[i] = __float2bfloat16_rn(c[i - na - nb]);
    else wo[i - na - nb - nc] = __float2bfloat16_rn(d[i - na - nb - nc]);
  }
}

// wv, wg, wb, wo: fp32 contiguous; returns (w [rows(wv) + rows(wg) + rows(wb), C] bf16, wo bf16)
std::vector<torch::Tensor> cast_weights(torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo) {
  for (const auto* t : {&wv, &wg, &wb, &wo}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->is_contiguous() && t->dim() == 2, "cast_weights: fp32 contiguous 2-D");
  TORCH_CHECK(wv.size(1) == wg.size(1) && wv.size(1) == wb.size(1), "cast_weights: widths");
  c10::cuda::CUDAGuard guard(wv.device());
  auto w = torch::empty({wv.size(0) + wg.size(0) + wb.size(0), wv.size(1)}, wv.options().dtype(torch::kBFloat16));
  auto wo16 = torch::empty(wo.sizes(), wo.options().dtype(torch::kBFloat16));
  const long total = wv.numel() + wg.numel() + wb.numel() + wo.numel();
  const int grid = (int)std::min<long>((total + 255) / 256, 132L * 8);
  cast_w_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(wv.data_ptr<float>(), wg.data_ptr<float>(), wb.data_ptr<float>(), wo.data_ptr<float>(),
                                                               reinterpret_cast<bf16*>(w.data_ptr()), reinterpret_cast<bf16*>(wo16.data_ptr()), wv.numel(),
                                                               wg.numel(), wb.numel(), wo.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, wo16};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_bwd", &gate_bwd);
  m.def("logits", &logits);
  m.def("out_proj", &out_proj);
  m.def("cgemm", &cgemm);
  m.def("tail", &tail);
  m.def("cast_weights", &cast_weights);
}
