// wide_ops.cu -- the A100 (sm_80) MSA pair-weighted averaging at the BIG width (d_msa 256, 8 heads x 32, IntelliFold2; integrations/pwa_wide_sm80.py):
// the layout / gate kernels between the cuBLAS GEMMs. Every GEMM of the path is cuBLAS; these four kernels move the 64-byte head runs between
// the natural layout of the projections and the head-major layout of the key contractions, the gate fused into the moves:
//
//   natural      x[s, j, h * 32 + c]      rows (s, j) of the projections ([S L, ld]: vg = y [Wv; Wg]^T has ld 512, v at columns 0.., g at 256..)
//   head-major   x[h, j, s * 32 + c]      [8, L, S 32]: for one (h, j) the S x 32 values are one run, so o = w v, dv = w^T do and dw = do v^T are
//                                         plain batched GEMMs (M = L, N or K = S 32)
//
//   to_hm      v natural -> head-major                                       (forward: the contraction's B operand)
//   gate_fwd   u = sigmoid(g) * o, o head-major -> u natural                 (the out-projection's A operand)
//   gate_bwd   du natural -> do = du sigmoid(g) head-major, dg = du o sigmoid(g) (1 - sigmoid(g)) natural (into the [dv | dg] buffer)
//   from_hm    dv head-major -> natural (the [dv | dg] buffer's v columns: the A operand of dy = [dv | dg] [Wv; Wg] and of d[Wv; Wg])
//
//   head_bmm   inference only: o = w v per head on the natural layout (cuBLAS strided batches over the MSA rows), no head-major copy
//
// A thread moves 16 bytes (8 channels); a run of 32 channels is 4 threads, read and written as one 64-byte segment on both sides. The gate
// follows the reference's rounding (modules/functional.sigmoid_gate on bf16: sigmoid rounded to bf16, then the product).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>

namespace {
using bf16 = __nv_bfloat16;
constexpr int H = 8, C = 32, HC = H * C;

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 t = __bfloat1622float2(p[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 v;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return v;
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// natural row (s, j) chunk k (16 bytes: head k / 4, channels 8 (k % 4)..) <-> head-major element offset
__device__ __forceinline__ long hm_off(int s, int j, int k, int S, int L) { return (((long)(k >> 2) * L + j) * S + s) * C + (k & 3) * 8; }

// head-major chunk order (h, j, s, q): the writes of to_hm are contiguous
__global__ void __launch_bounds__(256) to_hm_k(const bf16* __restrict__ src, long ld, bf16* __restrict__ dst, int S, int L) {
  const long n = (long)H * L * S * 4;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const int q = (int)(i & 3);
    long t = i >> 2;
    const int s = (int)(t % S); t /= S;
    const int j = (int)(t % L), h = (int)(t / L);
    reinterpret_cast<uint4*>(dst)[i] = *reinterpret_cast<const uint4*>(src + ((long)s * L + j) * ld + h * C + q * 8);
  }
}

__global__ void __launch_bounds__(256) from_hm_k(const bf16* __restrict__ src, bf16* __restrict__ dst, long ld, int S, int L) {
  const long n = (long)H * L * S * 4;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const int q = (int)(i & 3);
    long t = i >> 2;
    const int s = (int)(t % S); t /= S;
    const int j = (int)(t % L), h = (int)(t / L);
    *reinterpret_cast<uint4*>(dst + ((long)s * L + j) * ld + h * C + q * 8) = reinterpret_cast<const uint4*>(src)[i];
  }
}

// natural chunk order (row, k): u / g / du / dg contiguous, the head-major side in 64-byte segments
template <bool HM>
__global__ void __launch_bounds__(256) gate_fwd_k(const bf16* __restrict__ o, const bf16* __restrict__ vg, long ld, int gcol, bf16* __restrict__ u, int S,
                                                  int L) {
  const long n = (long)S * L * (HC / 8);
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const int k = (int)(i & 31);
    const long row = i >> 5;
    const int s = (int)(row / L), j = (int)(row % L);
    float g[8], ov[8], r[8];
    unpack8(*reinterpret_cast<const uint4*>(vg + row * ld + gcol + k * 8), g);
    unpack8(*reinterpret_cast<const uint4*>(o + (HM ? hm_off(s, j, k, S, L) : row * HC + k * 8)), ov);   // HM: o head-major, else natural [S L, 256]
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = bfr(sigm(g[e])) * ov[e];
    reinterpret_cast<uint4*>(u)[i] = pack8(r);
  }
}

__global__ void __launch_bounds__(256) gate_bwd_k(const bf16* __restrict__ du, const bf16* __restrict__ vg, long ld, int gcol, const bf16* __restrict__ o,
                                                  bf16* __restrict__ dout, bf16* __restrict__ dvg, long ldd, int S, int L) {
  const long n = (long)S * L * (HC / 8);
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const int k = (int)(i & 31);
    const long row = i >> 5;
    const int s = (int)(row / L), j = (int)(row % L);
    const long ho = hm_off(s, j, k, S, L);
    float d[8], g[8], ov[8], a[8], b[8];
    unpack8(reinterpret_cast<const uint4*>(du)[i], d);
    unpack8(*reinterpret_cast<const uint4*>(vg + row * ld + gcol + k * 8), g);
    unpack8(*reinterpret_cast<const uint4*>(o + ho), ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float sg = sigm(g[e]);
      a[e] = d[e] * bfr(sg);
      b[e] = d[e] * ov[e] * sg * (1.f - sg);
    }
    *reinterpret_cast<uint4*>(dout + ho) = pack8(a);
    *reinterpret_cast<uint4*>(dvg + row * ldd + gcol + k * 8) = pack8(b);
  }
}

int grid(long n) {
  const long sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return (int)std::max<long>(1, std::min<long>((n + 255) / 256, sms * 16));
}
const bf16* cp(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); }
bf16* mp(const torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr<at::BFloat16>()); }
void check_nat(const torch::Tensor& t, long rows, int cols, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(0) == rows && t.size(1) >= cols && t.stride(1) == 1 &&
              t.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": bf16 [S L, >= cols] rows, 16-byte aligned");
}
void check_hm(const torch::Tensor& t, int S, int L, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.numel() == (long)H * L * S * C, name, ": bf16 [8, L, S 32]");
}
cudaStream_t st() { return at::cuda::getCurrentCUDAStream(); }
}  // namespace

// src [S L, ld] (v at columns 0..255) -> dst [8, L, S 32]
void to_hm(torch::Tensor src, torch::Tensor dst, int64_t S, int64_t L) {
  check_nat(src, S * L, HC, "to_hm src");
  check_hm(dst, (int)S, (int)L, "to_hm dst");
  const long n = (long)H * L * S * 4;
  to_hm_k<<<grid(n), 256, 0, st()>>>(cp(src), src.stride(0), mp(dst), (int)S, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// src [8, L, S 32] -> dst [S L, ld] columns 0..255
void from_hm(torch::Tensor src, torch::Tensor dst, int64_t S, int64_t L) {
  check_hm(src, (int)S, (int)L, "from_hm src");
  check_nat(dst, S * L, HC, "from_hm dst");
  const long n = (long)H * L * S * 4;
  from_hm_k<<<grid(n), 256, 0, st()>>>(cp(src), mp(dst), dst.stride(0), (int)S, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// o [8, L, S 32] (head-major) or [S L, 256] (natural), vg [S L, ld] (g at columns gcol..gcol+255) -> u [S L, 256]
void gate_fwd(torch::Tensor o, torch::Tensor vg, int64_t gcol, torch::Tensor u, int64_t S, int64_t L) {
  const bool hm = o.dim() == 3;
  if (hm) check_hm(o, (int)S, (int)L, "gate_fwd o");
  else { check_nat(o, S * L, HC, "gate_fwd o"); TORCH_CHECK(o.is_contiguous(), "gate_fwd: natural o contiguous"); }
  check_nat(vg, S * L, (int)gcol + HC, "gate_fwd vg");
  TORCH_CHECK(gcol % 8 == 0 && u.is_contiguous(), "gate_fwd: gcol % 8, u contiguous");
  check_nat(u, S * L, HC, "gate_fwd u");
  const long n = (long)S * L * (HC / 8);
  if (hm) gate_fwd_k<true><<<grid(n), 256, 0, st()>>>(cp(o), cp(vg), vg.stride(0), (int)gcol, mp(u), (int)S, (int)L);
  else gate_fwd_k<false><<<grid(n), 256, 0, st()>>>(cp(o), cp(vg), vg.stride(0), (int)gcol, mp(u), (int)S, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// du [S L, 256], vg (g at gcol), o [8, L, S 32] -> dout [8, L, S 32]; dg into dvg [S L, ldd] columns gcol..
void gate_bwd(torch::Tensor du, torch::Tensor vg, int64_t gcol, torch::Tensor o, torch::Tensor dout, torch::Tensor dvg, int64_t S, int64_t L) {
  TORCH_CHECK(du.is_contiguous(), "gate_bwd: du contiguous");
  check_nat(du, S * L, HC, "gate_bwd du");
  check_nat(vg, S * L, (int)gcol + HC, "gate_bwd vg");
  check_nat(dvg, S * L, (int)gcol + HC, "gate_bwd dvg");
  check_hm(o, (int)S, (int)L, "gate_bwd o");
  check_hm(dout, (int)S, (int)L, "gate_bwd dout");
  TORCH_CHECK(gcol % 8 == 0, "gate_bwd: gcol % 8");
  const long n = (long)S * L * (HC / 8);
  gate_bwd_k<<<grid(n), 256, 0, st()>>>(cp(du), cp(vg), vg.stride(0), (int)gcol, cp(o), mp(dout), mp(dvg), dvg.stride(0), (int)S, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The key contraction on the natural layout (inference: no head-major copy of v): per head h, batched over the S MSA rows,
//   o[s, i, h 32 + c] = sum_j w[h, i, j] v[s, j, h 32 + c]       (cuBLAS, column-major: O_s^T [32 x L] = V_s^T [32 x L] . W_h^T)
// v [S L, ldv] (columns 0..255), w [8, L, L] bf16 -> o [S L, 256] bf16, fp32 accumulation.
void head_bmm(torch::Tensor w, torch::Tensor v, torch::Tensor o, int64_t S, int64_t L) {
  check_nat(v, S * L, HC, "head_bmm v");
  check_nat(o, S * L, HC, "head_bmm o");
  TORCH_CHECK(o.is_contiguous() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.numel() == (long)H * L * L, "head_bmm: w [8, L, L] bf16, o contiguous");
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.f, beta = 0.f;
  const long ldv = v.stride(0);
  for (int h = 0; h < H; ++h) {
    const cublasStatus_t s = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, (int)L, (int)L, &alpha, cp(v) + h * C, CUDA_R_16BF, (int)ldv,
                                                        L * ldv, cp(w) + (long)h * L * L, CUDA_R_16BF, (int)L, 0, &beta, mp(o) + h * C, CUDA_R_16BF, HC,
                                                        L * HC, (int)S, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "head_bmm: cublasGemmStridedBatchedEx status ", (int)s);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_bmm", &head_bmm);
  m.def("to_hm", &to_hm);
  m.def("from_hm", &from_hm);
  m.def("gate_fwd", &gate_fwd);
  m.def("gate_bwd", &gate_bwd);
}
