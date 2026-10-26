// qknorm_local.cu -- QK-norm of the AF3 atom block's q and k (H100 mma family, integrations/local_dit_h100.py): per head (4 heads x 32) RMSNorm
// with one shared weight per tensor, eps in the denominator as torch's rms_norm.
//
//   forward   qo = q * rsqrt(mean(q^2) + eps) * wq,    ko likewise              (q, k [M, 128] bf16 or fp32, weights fp32 [32])
//   backward  with x_hat = x rstd, g = dn w:   dx = rstd (g - x_hat mean(g x_hat)),   dw += dn x_hat
//   rotary    (qknorm_rope_fwd, ESMFold2, inference) then qo = rot(qo), ko = rot(ko) with per-atom tables cos / sin [N, 16] (row r is atom r % N):
//             channel c < 16 of a head pairs with c + 16, (a, b) -> (a cos - b sin, b cos + a sin) -- the partner is lane ^ 4.
//
// A warp per row, 8 lanes per head, 4 elements a lane: the mean of squares and the backward's mean are quad-of-octet shuffles, nothing
// touches shared memory in the forward. The backward runs in place on the gradient slices of the block's [dg | dq | dk | dv] buffer
// (row stride ``ld``) and leaves the weight gradients as per-block partial rows [blocks, 2, 32] that the caller sums in order: the
// rows go to the blocks in a fixed pattern and each block adds its warps in a fixed order, so the sum is bitwise repeatable.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int DS = 128, HD = 32, WARPS = 8;

template <typename T> __device__ __forceinline__ float4 load4(const T* p);
template <> __device__ __forceinline__ float4 load4<float>(const float* p) { return *reinterpret_cast<const float4*>(p); }
template <> __device__ __forceinline__ float4 load4<__nv_bfloat16>(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x), b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
  return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}
__device__ __forceinline__ float rna(float x) {                   // round to nearest tf32: the MMA operands of the fp32 path
  unsigned r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r);
}
template <typename T> __device__ __forceinline__ void store4(T* p, float4 v, bool rnd);
template <> __device__ __forceinline__ void store4<float>(float* p, float4 v, bool rnd) {
  if (rnd) v = make_float4(rna(v.x), rna(v.y), rna(v.z), rna(v.w));
  *reinterpret_cast<float4*>(p) = v;
}
template <> __device__ __forceinline__ void store4<__nv_bfloat16>(__nv_bfloat16* p, float4 v, bool) {
  const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<const unsigned*>(&a);
  u.y = *reinterpret_cast<const unsigned*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float octet_sum(float v) {          // the 8 lanes of a head
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 4);
}

template <typename T>
__global__ void __launch_bounds__(WARPS * 32)
qknorm_fwd_k(const T* __restrict__ q, const T* __restrict__ k, const float* __restrict__ wq, const float* __restrict__ wk,
             T* __restrict__ qo, T* __restrict__ ko, long M, float eq, float ek, bool rnd, const float* __restrict__ cs = nullptr,
             const float* __restrict__ sn = nullptr, long natoms = 1) {
  const long row = (long)blockIdx.x * WARPS + threadIdx.x / 32;
  if (row >= M) return;
  const int lane = threadIdx.x % 32, col = lane * 4, d = (lane % 8) * 4;
  const T* src[2] = {q + row * DS + col, k + row * DS + col};
  T* dst[2] = {qo + row * DS + col, ko + row * DS + col};
  const float* w[2] = {wq + d, wk + d};
  const float eps[2] = {eq, ek};
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const float4 x = load4<T>(src[t]);
    const float rstd = rsqrtf(octet_sum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w) / HD + eps[t]);
    const float4 wv = *reinterpret_cast<const float4*>(w[t]);
    float4 y = make_float4(x.x * rstd * wv.x, x.y * rstd * wv.y, x.z * rstd * wv.z, x.w * rstd * wv.w);
    if (cs != nullptr) {                                         // the rotary of the normalised values (fp32)
      const int r = d % 16;
      const float4 c = *reinterpret_cast<const float4*>(cs + (row % natoms) * 16 + r), s = *reinterpret_cast<const float4*>(sn + (row % natoms) * 16 + r);
      const float4 p = make_float4(__shfl_xor_sync(0xffffffffu, y.x, 4), __shfl_xor_sync(0xffffffffu, y.y, 4),
                                   __shfl_xor_sync(0xffffffffu, y.z, 4), __shfl_xor_sync(0xffffffffu, y.w, 4));
      const float sg = d < 16 ? -1.f : 1.f;                     // first half: a cos - b sin; second: b cos + a sin
      y = make_float4(y.x * c.x + sg * p.x * s.x, y.y * c.y + sg * p.y * s.y, y.z * c.z + sg * p.z * s.z, y.w * c.w + sg * p.w * s.w);
    }
    store4<T>(dst[t], y, rnd);
  }
}

template <typename T>
__global__ void __launch_bounds__(WARPS * 32)
qknorm_bwd_k(T* __restrict__ dq, T* __restrict__ dk, long ld, const T* __restrict__ qr, const T* __restrict__ kr,
             const float* __restrict__ wq, const float* __restrict__ wk, float* __restrict__ part, long M, float eq, float ek, bool rnd,
             const float* __restrict__ cs = nullptr, const float* __restrict__ sn = nullptr, long natoms = 1) {
  __shared__ float red[WARPS][2][4][8][4];                      // [warp][q/k][head][lane in head][dim]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, col = lane * 4, d = (lane % 8) * 4, head = lane / 8;
  T* dg[2] = {dq, dk};
  const T* raw[2] = {qr, kr};
  const float* w[2] = {wq + d, wk + d};
  const float eps[2] = {eq, ek};
  float4 acc[2] = {make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)};
  for (long row = (long)blockIdx.x * WARPS + warp; row < M; row += (long)gridDim.x * WARPS) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const float4 x = load4<T>(raw[t] + row * DS + col);
      float4 dn = load4<T>(dg[t] + row * ld + col);
      if (cs != nullptr) {   // the gradient of the rotated values back through the rotary: the transpose of the forward's rotation
        const int r = d % 16;
        const float4 c = *reinterpret_cast<const float4*>(cs + (row % natoms) * 16 + r), s = *reinterpret_cast<const float4*>(sn + (row % natoms) * 16 + r);
        const float4 p = make_float4(__shfl_xor_sync(0xffffffffu, dn.x, 4), __shfl_xor_sync(0xffffffffu, dn.y, 4),
                                     __shfl_xor_sync(0xffffffffu, dn.z, 4), __shfl_xor_sync(0xffffffffu, dn.w, 4));
        const float sg = d < 16 ? 1.f : -1.f;
        dn = make_float4(dn.x * c.x + sg * p.x * s.x, dn.y * c.y + sg * p.y * s.y, dn.z * c.z + sg * p.z * s.z, dn.w * c.w + sg * p.w * s.w);
      }
      const float rstd = rsqrtf(octet_sum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w) / HD + eps[t]);
      const float4 xh = make_float4(x.x * rstd, x.y * rstd, x.z * rstd, x.w * rstd);
      const float4 wv = *reinterpret_cast<const float4*>(w[t]);
      const float4 g = make_float4(dn.x * wv.x, dn.y * wv.y, dn.z * wv.z, dn.w * wv.w);
      const float mean = octet_sum(g.x * xh.x + g.y * xh.y + g.z * xh.z + g.w * xh.w) / HD;
      store4<T>(dg[t] + row * ld + col, make_float4(rstd * (g.x - xh.x * mean), rstd * (g.y - xh.y * mean),
                                                    rstd * (g.z - xh.z * mean), rstd * (g.w - xh.w * mean)), rnd);
      acc[t].x += dn.x * xh.x; acc[t].y += dn.y * xh.y; acc[t].z += dn.z * xh.z; acc[t].w += dn.w * xh.w;
    }
  }
#pragma unroll
  for (int t = 0; t < 2; ++t) *reinterpret_cast<float4*>(&red[warp][t][head][lane % 8][0]) = acc[t];
  __syncthreads();
  if (threadIdx.x < 64) {                                       // (q/k, dim): heads and warps added in a fixed order
    const int t = threadIdx.x / 32, dim = threadIdx.x % 32, j = dim / 4, e = dim % 4;
    float s = 0.f;
    for (int h = 0; h < 4; ++h)
      for (int wp = 0; wp < WARPS; ++wp) s += red[wp][t][h][j][e];
    part[((long)blockIdx.x * 2 + t) * HD + dim] = s;
  }
}

void check(const at::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(1) == DS && t.stride(1) == 1, what, ": [M, 128] rows, last dim contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, what, ": fp32 or bf16");
}
void check_w(const at::Tensor& w, const char* what) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.numel() == HD, what, ": fp32 [32]");
}
}  // namespace

void qknorm_fwd(at::Tensor q, at::Tensor k, at::Tensor wq, at::Tensor wk, double eq, double ek, at::Tensor qo, at::Tensor ko) {
  check(q, "q"); check(k, "k"); check(qo, "qo"); check(ko, "ko"); check_w(wq, "wq"); check_w(wk, "wk");
  TORCH_CHECK(q.stride(0) == DS && k.stride(0) == DS && qo.stride(0) == DS && ko.stride(0) == DS, "contiguous rows");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == qo.scalar_type() && k.sizes() == q.sizes(), "operands match");
  const long M = q.size(0);
  const at::cuda::CUDAGuard guard(q.device());
  const dim3 grid((unsigned)((M + WARPS - 1) / WARPS));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kFloat)
    qknorm_fwd_k<float><<<grid, WARPS * 32, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(), wq.data_ptr<float>(), wk.data_ptr<float>(),
                                                          qo.data_ptr<float>(), ko.data_ptr<float>(), M, (float)eq, (float)ek, true);
  else
    qknorm_fwd_k<__nv_bfloat16><<<grid, WARPS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), wq.data_ptr<float>(),
        wk.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(qo.data_ptr()), reinterpret_cast<__nv_bfloat16*>(ko.data_ptr()), M, (float)eq, (float)ek, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ESMFold2 (inference): the norm, then the rotary with the per-atom tables cos, sin [natoms, 16] fp32 (rows r of q / k are atom r % natoms).
void qknorm_rope_fwd(at::Tensor q, at::Tensor k, at::Tensor wq, at::Tensor wk, double eq, double ek, at::Tensor cs, at::Tensor sn, at::Tensor qo,
                     at::Tensor ko) {
  check(q, "q"); check(k, "k"); check(qo, "qo"); check(ko, "ko"); check_w(wq, "wq"); check_w(wk, "wk");
  TORCH_CHECK(q.stride(0) == DS && k.stride(0) == DS && qo.stride(0) == DS && ko.stride(0) == DS, "contiguous rows");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == qo.scalar_type() && k.sizes() == q.sizes(), "operands match");
  TORCH_CHECK(cs.is_cuda() && sn.is_cuda() && cs.scalar_type() == at::kFloat && sn.scalar_type() == at::kFloat && cs.is_contiguous() &&
              sn.is_contiguous() && cs.dim() == 2 && cs.size(1) == HD / 2 && sn.sizes() == cs.sizes(), "rotary tables: fp32 [N, 16]");
  const long M = q.size(0), natoms = cs.size(0);
  TORCH_CHECK(natoms > 0 && M % natoms == 0, "rows are samples x atoms");
  const at::cuda::CUDAGuard guard(q.device());
  const dim3 grid((unsigned)((M + WARPS - 1) / WARPS));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kFloat)
    qknorm_fwd_k<float><<<grid, WARPS * 32, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(), wq.data_ptr<float>(), wk.data_ptr<float>(),
                                                          qo.data_ptr<float>(), ko.data_ptr<float>(), M, (float)eq, (float)ek, true,
                                                          cs.data_ptr<float>(), sn.data_ptr<float>(), natoms);
  else
    qknorm_fwd_k<__nv_bfloat16><<<grid, WARPS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), wq.data_ptr<float>(),
        wk.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(qo.data_ptr()), reinterpret_cast<__nv_bfloat16*>(ko.data_ptr()), M, (float)eq, (float)ek, false,
        cs.data_ptr<float>(), sn.data_ptr<float>(), natoms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dq, dk: [M, 128] views into the block's gradient buffer (row stride ld), rewritten in place from the gradient of the normalised
// tensor to that of the raw one; qr, kr: the raw q, k. Returns the weight-gradient partials [blocks, 2, 32] fp32.
at::Tensor qknorm_bwd(at::Tensor dq, at::Tensor dk, at::Tensor qr, at::Tensor kr, at::Tensor wq, at::Tensor wk, double eq, double ek) {
  check(dq, "dq"); check(dk, "dk"); check(qr, "q raw"); check(kr, "k raw"); check_w(wq, "wq"); check_w(wk, "wk");
  TORCH_CHECK(dq.stride(0) == dk.stride(0) && qr.stride(0) == DS && kr.stride(0) == DS, "row strides");
  TORCH_CHECK(dq.scalar_type() == qr.scalar_type() && dk.scalar_type() == qr.scalar_type() && kr.scalar_type() == qr.scalar_type(), "operands match");
  const long M = qr.size(0);
  const at::cuda::CUDAGuard guard(qr.device());
  const int blocks = (int)std::min<long>((M + WARPS - 1) / WARPS, 1056);          // a fixed function of M: the partials' order is too
  auto part = at::empty({blocks, 2, HD}, qr.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (qr.scalar_type() == at::kFloat)
    qknorm_bwd_k<float><<<blocks, WARPS * 32, 0, stream>>>(dq.data_ptr<float>(), dk.data_ptr<float>(), dq.stride(0), qr.data_ptr<float>(),
                                                            kr.data_ptr<float>(), wq.data_ptr<float>(), wk.data_ptr<float>(), part.data_ptr<float>(), M, (float)eq, (float)ek, true);
  else
    qknorm_bwd_k<__nv_bfloat16><<<blocks, WARPS * 32, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()), dq.stride(0),
        reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(kr.data_ptr()), wq.data_ptr<float>(),
        wk.data_ptr<float>(), part.data_ptr<float>(), M, (float)eq, (float)ek, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return part;
}

// ESMFold2 (training): :func:`qknorm_bwd` for q / k that were rotated after the norm; cs, sn the forward's tables [natoms, 16].
at::Tensor qknorm_rope_bwd(at::Tensor dq, at::Tensor dk, at::Tensor qr, at::Tensor kr, at::Tensor wq, at::Tensor wk, double eq, double ek, at::Tensor cs,
                           at::Tensor sn) {
  check(dq, "dq"); check(dk, "dk"); check(qr, "q raw"); check(kr, "k raw"); check_w(wq, "wq"); check_w(wk, "wk");
  TORCH_CHECK(dq.stride(0) == dk.stride(0) && qr.stride(0) == DS && kr.stride(0) == DS, "row strides");
  TORCH_CHECK(dq.scalar_type() == qr.scalar_type() && dk.scalar_type() == qr.scalar_type() && kr.scalar_type() == qr.scalar_type(), "operands match");
  TORCH_CHECK(cs.is_cuda() && sn.is_cuda() && cs.scalar_type() == at::kFloat && sn.scalar_type() == at::kFloat && cs.is_contiguous() &&
              sn.is_contiguous() && cs.dim() == 2 && cs.size(1) == HD / 2 && sn.sizes() == cs.sizes(), "rotary tables: fp32 [N, 16]");
  const long M = qr.size(0), natoms = cs.size(0);
  TORCH_CHECK(natoms > 0 && M % natoms == 0, "rows are samples x atoms");
  TORCH_CHECK(qr.scalar_type() == at::kBFloat16, "qknorm_rope_bwd: bf16");
  const at::cuda::CUDAGuard guard(qr.device());
  const int blocks = (int)std::min<long>((M + WARPS - 1) / WARPS, 1056);          // a fixed function of M: the partials' order is too
  auto part = at::empty({blocks, 2, HD}, qr.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  qknorm_bwd_k<__nv_bfloat16><<<blocks, WARPS * 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()), dq.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(kr.data_ptr()), wq.data_ptr<float>(),
      wk.data_ptr<float>(), part.data_ptr<float>(), M, (float)eq, (float)ek, false, cs.data_ptr<float>(), sn.data_ptr<float>(), natoms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return part;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qknorm_rope_bwd", &qknorm_rope_bwd);
  m.def("qknorm_fwd", &qknorm_fwd);
  m.def("qknorm_rope_fwd", &qknorm_rope_fwd);
  m.def("qknorm_bwd", &qknorm_bwd);
}
