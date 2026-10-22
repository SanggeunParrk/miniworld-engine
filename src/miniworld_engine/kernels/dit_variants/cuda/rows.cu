// dit_variants rows.cu -- the row passes of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1 token DiT; inference).
//
//   ln_mod_rows    xhat = LN(x) (no affine); out1 = xhat * f(ms1[tok]) + mb1[tok]  [, out2 = xhat * f(ms2[tok]) + mb2[tok]]
//                  f = sigmoid (AF3 / Boltz-2 / RF3 AdaLN) or identity (Chai-1: the table already holds W s + 1).
//                  xin given: x = xin first (the step's input rows into the fp32 residual).
//   res_rows       x += sigmoid(ga[tok]) * ya  [+ sigmoid(gt[tok]) * yt]  (fp32); then x back to fp32 or, for the last block,
//                  straight into the output dtype; [out_ln = LN(x) * f(ms) + mb for the sequential block's transition]
//   qk_ln_rows     q = row[0, D), k = row[D, 2D) <- LayerNorm over all D columns * w + b, in place (RF3 kq-norm; the logit scale
//                  folded into q's w / b)
//   glu_rows       out = silu(g) * v [* u],  row = [g | v | u]  (u: Boltz-2's up-gate)
//   ln_rows        out = LN(z) (no affine), C = 128 or 256 (the pair rows of the hoisted pair bias)
//
// tok = row % L: the AdaLN / gate tables are per token, shared by the samples. One block per row, a thread owns 4 consecutive
// columns; sums in fp32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;

template <bool SIG>
__device__ __forceinline__ float scl(float v) {
  if constexpr (SIG) return sigm(v); else return v;
}

template <int NT, bool SIG, typename GT, typename OutT>
__device__ __forceinline__ void mod_store(const float4& xh, const GT* ms, const GT* mb, OutT* out) {
  const int col = threadIdx.x * 4;
  const float4 m = V4<GT>::load(ms + col), h = V4<GT>::load(mb + col);
  float4 o;
  o.x = xh.x * scl<SIG>(m.x) + h.x; o.y = xh.y * scl<SIG>(m.y) + h.y;
  o.z = xh.z * scl<SIG>(m.z) + h.z; o.w = xh.w * scl<SIG>(m.w) + h.w;
  V4<OutT>::store(out + col, o);
}

template <int NT>
__device__ __forceinline__ float4 layernorm(float4 x, float eps, float* red) {
  constexpr int D = NT * 4;
  const float mean = block_sum<NT>(x.x + x.y + x.z + x.w, red) / D;
  x.x -= mean; x.y -= mean; x.z -= mean; x.w -= mean;
  const float r = rsqrtf(block_sum<NT>(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w, red) / D + eps);
  x.x *= r; x.y *= r; x.z *= r; x.w *= r;
  return x;
}

template <int NT, bool SIG, typename XT, typename GT, typename OutT>
__global__ void __launch_bounds__(NT) ln_mod_kernel(const XT* __restrict__ XIN, float* __restrict__ X, const GT* MS1,
    const GT* MB1, OutT* OUT1, const GT* MS2, const GT* MB2, OutT* OUT2, int L, long sms, float eps) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = blockIdx.x;
  const int tok = row % L, col = threadIdx.x * 4;
  float4 x;
  if (XIN != nullptr) {
    x = V4<XT>::load(XIN + row * D + col);
    V4<float>::store(X + row * D + col, x);
  } else {
    x = V4<float>::load(X + row * D + col);
  }
  const float4 xh = layernorm<NT>(x, eps, red);
  mod_store<NT, SIG>(xh, MS1 + tok * sms, MB1 + tok * sms, OUT1 + row * D);
  if (OUT2 != nullptr) mod_store<NT, SIG>(xh, MS2 + tok * sms, MB2 + tok * sms, OUT2 + row * D);
}

template <int NT, bool SIG, typename YT, typename GT, typename OutT, typename FT>
__global__ void __launch_bounds__(NT) res_kernel(float* __restrict__ X, const YT* YA, long sya, const GT* GA, const YT* YT2,
    long syt, const GT* GT2, const GT* MS, const GT* MB, OutT* OUT_LN, FT* FINAL, int L, long sg, long sms, float eps) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = blockIdx.x;
  const int tok = row % L, col = threadIdx.x * 4;
  float4 x = V4<float>::load(X + row * D + col);
  float4 y = V4<YT>::load(YA + row * sya + col), g = V4<GT>::load(GA + tok * sg + col);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  if (YT2 != nullptr) {
    y = V4<YT>::load(YT2 + row * syt + col); g = V4<GT>::load(GT2 + tok * sg + col);
    x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  }
  if (FINAL != nullptr) V4<FT>::store(FINAL + row * D + col, x);
  else V4<float>::store(X + row * D + col, x);
  if (OUT_LN != nullptr) mod_store<NT, SIG>(layernorm<NT>(x, eps, red), MS + tok * sms, MB + tok * sms, OUT_LN + row * D);
}

// q and k of a row, each LayerNorm'd over its D columns; one block of NT = D / 4 threads per row walks q then k.
template <int NT, typename T>
__global__ void __launch_bounds__(NT) qk_ln_kernel(T* __restrict__ QK, long sq, const float* __restrict__ WQ,
    const float* __restrict__ BQ, const float* __restrict__ WK, const float* __restrict__ BK, float eps) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const int col = threadIdx.x * 4;
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    T* p = QK + (long)blockIdx.x * sq + which * D + col;
    const float4 xh = layernorm<NT>(V4<T>::load(p), eps, red);
    const float4 w = *reinterpret_cast<const float4*>((which ? WK : WQ) + col);
    const float4 b = *reinterpret_cast<const float4*>((which ? BK : BQ) + col);
    V4<T>::store(p, make_float4(xh.x * w.x + b.x, xh.y * w.y + b.y, xh.z * w.z + b.z, xh.w * w.w + b.w));
  }
}

template <typename T, bool UP>
__global__ void __launch_bounds__(512) glu_kernel(const T* __restrict__ AB, T* __restrict__ OUT, int N, long sab) {
  const long row = blockIdx.x;
  for (int c = threadIdx.x * 4; c < N; c += blockDim.x * 4) {
    const float4 g = V4<T>::load(AB + row * sab + c), v = V4<T>::load(AB + row * sab + N + c);
    float4 o = make_float4(g.x * sigm(g.x) * v.x, g.y * sigm(g.y) * v.y, g.z * sigm(g.z) * v.z, g.w * sigm(g.w) * v.w);
    if constexpr (UP) {
      const float4 u = V4<T>::load(AB + row * sab + 2 * N + c);
      o.x *= u.x; o.y *= u.y; o.z *= u.z; o.w *= u.w;
    }
    V4<T>::store(OUT + row * N + c, o);
  }
}

// LayerNorm of C-wide rows (C = 128 or 256), no affine: one warp per row (C / 128 float4 per lane), 8 rows per block.
template <int C, typename ZT, typename OutT>
__global__ void __launch_bounds__(256) ln_rows_kernel(const ZT* __restrict__ Z, OutT* __restrict__ OUT, long R, float eps) {
  constexpr int K = C / 128;
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 z[K];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { z[k] = V4<ZT>::load(Z + row * C + k * 128 + lane * 4); s += z[k].x + z[k].y + z[k].z + z[k].w; }
  const float mean = warp_sum(s) / C;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    z[k].x -= mean; z[k].y -= mean; z[k].z -= mean; z[k].w -= mean;
    q += z[k].x * z[k].x + z[k].y * z[k].y + z[k].z * z[k].z + z[k].w * z[k].w;
  }
  const float r = rsqrtf(warp_sum(q) / C + eps);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    z[k].x *= r; z[k].y *= r; z[k].z *= r; z[k].w *= r;
    V4<OutT>::store(OUT + row * C + k * 128 + lane * 4, z[k]);
  }
}

// ------------------------------------------------------------------------------------------------ host
using BF = __nv_bfloat16;

void check(const at::Tensor& t, const char* name, at::ScalarType dt, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1, name,
              ": a CUDA ", c10::toString(dt), " [rows, ", cols, "] matrix with unit column stride");
}

template <typename T>
const T* ptr(const c10::optional<at::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr;
}

constexpr int D = 768, NT = D / 4;

// x [M, 768] fp32 (written when xin is given), ms / mb [L, *] bf16 views (row stride sms, the same for all four), out bf16
void ln_mod_rows(c10::optional<at::Tensor> xin, at::Tensor x, at::Tensor ms1, at::Tensor mb1, at::Tensor out1,
                 c10::optional<at::Tensor> ms2, c10::optional<at::Tensor> mb2, c10::optional<at::Tensor> out2, int64_t L,
                 double eps, bool sig) {
  check(x, "x", at::kFloat, D);
  check(out1, "out1", at::kBFloat16, D);
  if (xin) check(*xin, "xin", at::kBFloat16, D);
  TORCH_CHECK(x.is_contiguous() && out1.is_contiguous() && (!xin || xin->is_contiguous()), "contiguous rows");
  TORCH_CHECK(ms1.scalar_type() == at::kBFloat16 && mb1.scalar_type() == at::kBFloat16 && ms1.stride(1) == 1 && mb1.stride(1) == 1, "bf16 tables");
  TORCH_CHECK(ms1.stride(0) == mb1.stride(0) && (!ms2 || (ms2->stride(0) == ms1.stride(0) && mb2->stride(0) == ms1.stride(0))),
              "one row stride for every table");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto s = at::cuda::getCurrentCUDAStream();
  auto f = sig ? ln_mod_kernel<NT, true, BF, BF, BF> : ln_mod_kernel<NT, false, BF, BF, BF>;
  f<<<x.size(0), NT, 0, s>>>(ptr<BF>(xin), x.data_ptr<float>(), reinterpret_cast<const BF*>(ms1.data_ptr()), reinterpret_cast<const BF*>(mb1.data_ptr()),
                             reinterpret_cast<BF*>(out1.data_ptr()), ptr<BF>(ms2), ptr<BF>(mb2),
                             out2 ? reinterpret_cast<BF*>(out2->data_ptr()) : nullptr, (int)L, ms1.stride(0), (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x += sig(ga) ya [+ sig(gt) yt]; ya / yt bf16 [M, >= 768] (row strides), tables bf16 [L, *]; final (bf16 [M, 768]) or x
void res_rows(at::Tensor x, at::Tensor ya, at::Tensor ga, c10::optional<at::Tensor> yt, c10::optional<at::Tensor> gt,
              c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb, c10::optional<at::Tensor> out_ln,
              c10::optional<at::Tensor> final_out, int64_t L, double eps, bool sig) {
  check(x, "x", at::kFloat, D);
  TORCH_CHECK(x.is_contiguous(), "x contiguous");
  TORCH_CHECK(ya.stride(1) == 1 && ya.scalar_type() == at::kBFloat16 && (!yt || (yt->stride(1) == 1 && yt->scalar_type() == at::kBFloat16)),
              "ya / yt: bf16 rows with unit column stride");
  TORCH_CHECK(ga.scalar_type() == at::kBFloat16 && ga.stride(1) == 1 && (!gt || (gt->scalar_type() == at::kBFloat16 && gt->stride(0) == ga.stride(0))),
              "ga / gt: bf16 tables, one row stride");
  if (out_ln) TORCH_CHECK(out_ln->is_contiguous() && ms && mb && ms->stride(0) == mb->stride(0), "out_ln needs ms / mb");
  if (final_out) check(*final_out, "final", at::kBFloat16, D);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto s = at::cuda::getCurrentCUDAStream();
  auto f = sig ? res_kernel<NT, true, BF, BF, BF, BF> : res_kernel<NT, false, BF, BF, BF, BF>;
  f<<<x.size(0), NT, 0, s>>>(x.data_ptr<float>(), reinterpret_cast<const BF*>(ya.data_ptr()), ya.stride(0),
                             reinterpret_cast<const BF*>(ga.data_ptr()), yt ? reinterpret_cast<const BF*>(yt->data_ptr()) : nullptr,
                             yt ? yt->stride(0) : 0, ptr<BF>(gt), ptr<BF>(ms), ptr<BF>(mb),
                             out_ln ? reinterpret_cast<BF*>(out_ln->data_ptr()) : nullptr,
                             final_out ? reinterpret_cast<BF*>(final_out->data_ptr()) : nullptr, (int)L, ga.stride(0),
                             ms ? ms->stride(0) : 0, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_ln_rows(at::Tensor qk, at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor bk, double eps) {
  TORCH_CHECK(qk.scalar_type() == at::kBFloat16 && qk.stride(1) == 1 && qk.size(1) >= 2 * D, "qk: bf16 rows, q | k first");
  for (auto* w : {&wq, &bq, &wk, &bk}) TORCH_CHECK(w->scalar_type() == at::kFloat && w->is_contiguous() && w->numel() == D, "fp32 [768]");
  const at::cuda::OptionalCUDAGuard guard(qk.device());
  qk_ln_kernel<NT, BF><<<qk.size(0), NT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<BF*>(qk.data_ptr()), qk.stride(0), wq.data_ptr<float>(), bq.data_ptr<float>(), wk.data_ptr<float>(),
      bk.data_ptr<float>(), (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void glu_rows(at::Tensor ab, at::Tensor out, bool up) {
  const int64_t n = out.size(1);
  TORCH_CHECK(ab.scalar_type() == at::kBFloat16 && ab.stride(1) == 1 && ab.size(1) == (up ? 3 : 2) * n && n % 4 == 0,
              "ab: bf16 [rows, (2 | 3) n]");
  check(out, "out", at::kBFloat16, n);
  TORCH_CHECK(out.is_contiguous(), "out contiguous");
  const at::cuda::OptionalCUDAGuard guard(ab.device());
  auto f = up ? glu_kernel<BF, true> : glu_kernel<BF, false>;
  f<<<ab.size(0), 512, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const BF*>(ab.data_ptr()),
                                                              reinterpret_cast<BF*>(out.data_ptr()), (int)n, ab.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_rows(at::Tensor z, at::Tensor out, double eps) {
  const int64_t c = z.size(1);
  TORCH_CHECK((c == 128 || c == 256) && z.is_contiguous() && out.is_contiguous() && out.sizes() == z.sizes() &&
              z.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "z / out: contiguous bf16 [rows, 128 | 256]");
  const at::cuda::OptionalCUDAGuard guard(z.device());
  const long r = z.size(0);
  auto f = c == 128 ? ln_rows_kernel<128, BF, BF> : ln_rows_kernel<256, BF, BF>;
  f<<<(r + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const BF*>(z.data_ptr()),
                                                               reinterpret_cast<BF*>(out.data_ptr()), r, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod_rows", &ln_mod_rows);
  m.def("res_rows", &res_rows);
  m.def("qk_ln_rows", &qk_ln_rows);
  m.def("glu_rows", &glu_rows);
  m.def("ln_rows", &ln_rows);
}
