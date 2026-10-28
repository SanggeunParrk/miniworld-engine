// dit_variants rows.cu -- the row passes of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1 token and atom DiT; inference).
// Rows of D = 768 (token) or 128 (atom) columns.
//
//   ln_mod_rows    xhat = LN(x) (no affine); out_i = xhat * f(ms_i[tok]) + mb_i[tok] for up to three (ms, mb, out)
//                  f = sigmoid (AF3 / Boltz-2 / RF3 AdaLN) or identity (Chai-1: the table already holds W s + 1).
//                  xin given: x = xin first (the step's input rows into the fp32 residual).
//   res_rows       x += sigmoid(ga[tok]) * ya  [+ sigmoid(gt[tok]) * yt]  (fp32); then x back to fp32 or, for the last block,
//                  straight into the output dtype; [out_ln = LN(x) * f(ms) + mb for the sequential block's transition]
//   qk_ln_rows     q <- LayerNorm(q) * wq + bq, k <- LayerNorm(k) * wk + bk over all D columns, in place (RF3 kq-norm; a logit scale
//                  folded into q's w / b)
//   gate_rows_inplace      o <- o * sigmoid(g)
//   glu_rows       out = silu(g) * v [* u],  row = [g | v | u]  (u: Boltz-2's up-gate)
//   parallel_residual  out = x + sum_i (y_i - x): a parallel block (Chai-1, RoseTTAFold3) from its parts' residual outputs
//
// tok = row % L: the AdaLN / gate tables are per token (atom), shared by the samples. One block per row, a thread owns 4
// consecutive columns; sums in fp32.
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

// Row geometry: a 768-wide row is a block of NT = 192 threads; a 128-wide row (NT = 32) is a warp, 8 rows to a 256-thread block (a
// 32-thread block per row left the SMs half empty: 32 blocks per SM at most).
template <int NT>
struct Geo {
  static constexpr int RPB = NT == 32 ? 8 : 1, THREADS = NT * RPB;
  static __device__ __forceinline__ long row() { return NT == 32 ? (long)blockIdx.x * RPB + threadIdx.x / 32 : (long)blockIdx.x; }
  static __device__ __forceinline__ int col() { return (NT == 32 ? threadIdx.x % 32 : threadIdx.x) * 4; }
  static long blocks(long rows) { return (rows + RPB - 1) / RPB; }
};

template <int NT, bool SIG, typename GT, typename OutT>
__device__ __forceinline__ void mod_store(const float4& xh, const GT* ms, const GT* mb, OutT* out) {
  const int col = Geo<NT>::col();
  const float4 m = V4<GT>::load(ms + col), h = V4<GT>::load(mb + col);
  float4 o;
  o.x = xh.x * scl<SIG>(m.x) + h.x; o.y = xh.y * scl<SIG>(m.y) + h.y;
  o.z = xh.z * scl<SIG>(m.z) + h.z; o.w = xh.w * scl<SIG>(m.w) + h.w;
  V4<OutT>::store(out + col, o);
}

template <int NT>
__device__ __forceinline__ float row_sum(float v, float* red) {
  if constexpr (NT == 32) return warp_sum(v); else return block_sum<NT>(v, red);
}

template <int NT>
__device__ __forceinline__ float4 layernorm(float4 x, float eps, float* red) {
  constexpr int D = NT * 4;
  const float mean = row_sum<NT>(x.x + x.y + x.z + x.w, red) / D;
  x.x -= mean; x.y -= mean; x.z -= mean; x.w -= mean;
  const float r = rsqrtf(row_sum<NT>(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w, red) / D + eps);
  x.x *= r; x.y *= r; x.z *= r; x.w *= r;
  return x;
}

struct Mods {
  const __nv_bfloat16* ms[3];
  const __nv_bfloat16* mb[3];
  __nv_bfloat16* out[3];
  int n;
};

template <int NT, bool SIG>
__global__ void __launch_bounds__(Geo<NT>::THREADS) ln_mod_kernel(const __nv_bfloat16* __restrict__ XIN, float* __restrict__ X, Mods m,
                                                                  int L, long sms, float eps, long M) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;                            // only a warp-per-row launch has a ragged last block (no block barrier there)
  const int tok = row % L, col = Geo<NT>::col();
  float4 x;
  if (XIN != nullptr) {
    x = V4<__nv_bfloat16>::load(XIN + row * D + col);
    V4<float>::store(X + row * D + col, x);
  } else {
    x = V4<float>::load(X + row * D + col);
  }
  const float4 xh = layernorm<NT>(x, eps, red);
#pragma unroll
  for (int i = 0; i < 3; ++i)
    if (i < m.n) mod_store<NT, SIG>(xh, m.ms[i] + tok * sms, m.mb[i] + tok * sms, m.out[i] + row * D);
}

template <int NT, bool SIG, typename T>
__global__ void __launch_bounds__(Geo<NT>::THREADS) res_kernel(float* __restrict__ X, const T* YA, long sya, const T* GA, const T* YT2,
    long syt, const T* GT2, const T* MS, const T* MB, T* OUT_LN, T* FINAL, int L, long sg, long sms, float eps, long M) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int tok = row % L, col = Geo<NT>::col();
  float4 x = V4<float>::load(X + row * D + col);
  float4 y = V4<T>::load(YA + row * sya + col), g = V4<T>::load(GA + tok * sg + col);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  if (YT2 != nullptr) {
    y = V4<T>::load(YT2 + row * syt + col); g = V4<T>::load(GT2 + tok * sg + col);
    x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  }
  if (FINAL != nullptr) V4<T>::store(FINAL + row * D + col, x);
  else V4<float>::store(X + row * D + col, x);
  if (OUT_LN != nullptr) mod_store<NT, SIG>(layernorm<NT>(x, eps, red), MS + tok * sms, MB + tok * sms, OUT_LN + row * D);
}

// q and k of a row, each LayerNorm'd over its D columns, q then k.
template <int NT, typename T>
__global__ void __launch_bounds__(Geo<NT>::THREADS) qk_ln_kernel(T* __restrict__ Q, long sq, T* __restrict__ K, long sk,
    const float* __restrict__ WQ, const float* __restrict__ BQ, const float* __restrict__ WK, const float* __restrict__ BK, float eps,
    long M) {
  __shared__ float red[NT / 32];
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int col = Geo<NT>::col();
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    T* p = which ? K + row * sk + col : Q + row * sq + col;
    const float4 xh = layernorm<NT>(V4<T>::load(p), eps, red);
    const float4 w = *reinterpret_cast<const float4*>((which ? WK : WQ) + col);
    const float4 b = *reinterpret_cast<const float4*>((which ? BK : BQ) + col);
    V4<T>::store(p, make_float4(xh.x * w.x + b.x, xh.y * w.y + b.y, xh.z * w.z + b.z, xh.w * w.w + b.w));
  }
}

// gate / GLU: flat over the rows' 4-wide pieces (any width keeps the SMs full; a block per row starved them at 128 / 256 wide)
template <typename T>
__global__ void __launch_bounds__(256) gate_kernel(T* __restrict__ O, long so, const T* __restrict__ G, long sg, int D, long n4) {
  const int per = D / 4;
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long row = e / per;
    const int c = (int)(e - row * per) * 4;
    float4 o = V4<T>::load(O + row * so + c);
    const float4 g = V4<T>::load(G + row * sg + c);
    o.x *= sigm(g.x); o.y *= sigm(g.y); o.z *= sigm(g.z); o.w *= sigm(g.w);
    V4<T>::store(O + row * so + c, o);
  }
}

template <typename T, bool UP>
__global__ void __launch_bounds__(256) glu_kernel(const T* __restrict__ AB, T* __restrict__ OUT, int N, long sab, long n4) {
  const int per = N / 4;
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long row = e / per;
    const int c = (int)(e - row * per) * 4;
    const float4 g = V4<T>::load(AB + row * sab + c), v = V4<T>::load(AB + row * sab + N + c);
    float4 o = make_float4(g.x * sigm(g.x) * v.x, g.y * sigm(g.y) * v.y, g.z * sigm(g.z) * v.z, g.w * sigm(g.w) * v.w);
    if constexpr (UP) {
      const float4 u = V4<T>::load(AB + row * sab + 2 * N + c);
      o.x *= u.x; o.y *= u.y; o.z *= u.z; o.w *= u.w;
    }
    V4<T>::store(OUT + row * N + c, o);
  }
}

long flat_grid(long n4) { return std::min<long>((n4 + 255) / 256, 132L * 16); }


// out = x + sum_i (y_i - x): a parallel block's update from residual outputs y_i = x + f_i(x) (fp32 sums, one rounding at the end)
struct Outs {
  const void* y[8];
  int n;
};

template <typename T>
__global__ void __launch_bounds__(256) parallel_residual_kernel(const T* __restrict__ X, Outs ys, T* __restrict__ OUT, long n4) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    const float4 x = V4<T>::load(X + 4 * i);
    float4 acc = x;
    for (int k = 0; k < ys.n; ++k) {
      const float4 y = V4<T>::load(reinterpret_cast<const T*>(ys.y[k]) + 4 * i);
      acc.x += y.x - x.x; acc.y += y.y - x.y; acc.z += y.z - x.z; acc.w += y.w - x.w;
    }
    V4<T>::store(OUT + 4 * i, acc);
  }
}

// ------------------------------------------------------------------------------------------------ host
using BF = __nv_bfloat16;

void check(const at::Tensor& t, const char* name, at::ScalarType dt, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1, name,
              ": a CUDA ", c10::toString(dt), " [rows, ", cols, "] matrix with unit column stride");
}

void check_table(const at::Tensor& t, const char* name, int64_t cols, int64_t stride) {
  check(t, name, at::kBFloat16, cols);
  TORCH_CHECK(t.stride(0) == stride, name, ": every table of a call shares one row stride");
}

template <typename T>
const T* ptr(const c10::optional<at::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr;
}

const BF* bf(const at::Tensor& t) { return reinterpret_cast<const BF*>(t.data_ptr()); }
BF* bfw(const at::Tensor& t) { return reinterpret_cast<BF*>(t.data_ptr()); }

int64_t width(const at::Tensor& x) {
  const int64_t d = x.size(1);
  TORCH_CHECK(d == 768 || d == 128, "rows of 768 (token) or 128 (atom) columns, got ", d);
  return d;
}

// x [M, D] fp32 (written when xin is given), ms / mb [L, D] bf16 table views (one row stride), outs bf16 [M, D] contiguous
void ln_mod_rows(c10::optional<at::Tensor> xin, at::Tensor x, std::vector<at::Tensor> ms, std::vector<at::Tensor> mb,
                 std::vector<at::Tensor> out, int64_t L, double eps, bool sig) {
  const int64_t d = width(x);
  check(x, "x", at::kFloat, d);
  TORCH_CHECK(x.is_contiguous() && (!xin || (xin->is_contiguous() && xin->sizes() == x.sizes() && xin->scalar_type() == at::kBFloat16)),
              "x / xin: contiguous rows (xin bf16)");
  const int n = (int)out.size();
  TORCH_CHECK(n >= 1 && n <= 3 && (int)ms.size() == n && (int)mb.size() == n, "one to three (ms, mb, out)");
  Mods m{};
  m.n = n;
  for (int i = 0; i < n; ++i) {
    check_table(ms[i], "ms", d, ms[0].stride(0));
    check_table(mb[i], "mb", d, ms[0].stride(0));
    check(out[i], "out", at::kBFloat16, d);
    TORCH_CHECK(out[i].is_contiguous(), "out contiguous");
    m.ms[i] = bf(ms[i]); m.mb[i] = bf(mb[i]); m.out[i] = bfw(out[i]);
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto s = at::cuda::getCurrentCUDAStream();
  const BF* in = ptr<BF>(xin);
  const long sms = ms[0].stride(0);
  if (d == 768) (sig ? ln_mod_kernel<192, true> : ln_mod_kernel<192, false>)<<<x.size(0), 192, 0, s>>>(in, x.data_ptr<float>(), m, (int)L, sms, (float)eps, x.size(0));
  else (sig ? ln_mod_kernel<32, true> : ln_mod_kernel<32, false>)<<<Geo<32>::blocks(x.size(0)), Geo<32>::THREADS, 0, s>>>(in, x.data_ptr<float>(), m, (int)L, sms, (float)eps, x.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x += sig(ga) ya [+ sig(gt) yt]; ya / yt bf16 [M, >= D] (row strides), tables bf16 [L, *]; into final (bf16 [M, D]) or x
void res_rows(at::Tensor x, at::Tensor ya, at::Tensor ga, c10::optional<at::Tensor> yt, c10::optional<at::Tensor> gt,
              c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb, c10::optional<at::Tensor> out_ln,
              c10::optional<at::Tensor> final_out, int64_t L, double eps, bool sig) {
  const int64_t d = width(x);
  check(x, "x", at::kFloat, d);
  TORCH_CHECK(x.is_contiguous(), "x contiguous");
  check(ya, "ya", at::kBFloat16, d);
  if (yt) check(*yt, "yt", at::kBFloat16, d);
  check_table(ga, "ga", d, ga.stride(0));
  if (gt) check_table(*gt, "gt", d, ga.stride(0));
  if (out_ln) {
    TORCH_CHECK(ms && mb, "out_ln needs ms / mb");
    check_table(*ms, "ms", d, ms->stride(0));
    check_table(*mb, "mb", d, ms->stride(0));
    check(*out_ln, "out_ln", at::kBFloat16, d);
    TORCH_CHECK(out_ln->is_contiguous(), "out_ln contiguous");
  }
  if (final_out) { check(*final_out, "final", at::kBFloat16, d); TORCH_CHECK(final_out->is_contiguous(), "final contiguous"); }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto s = at::cuda::getCurrentCUDAStream();
  BF* oln = out_ln ? bfw(*out_ln) : nullptr;
  BF* fin = final_out ? bfw(*final_out) : nullptr;
  const long syt = yt ? yt->stride(0) : 0, sms = ms ? ms->stride(0) : 0;
#define RES_ARGS x.data_ptr<float>(), bf(ya), ya.stride(0), bf(ga), ptr<BF>(yt), syt, ptr<BF>(gt), ptr<BF>(ms), ptr<BF>(mb), oln, fin, \
                 (int)L, ga.stride(0), sms, (float)eps, x.size(0)
  if (d == 768) (sig ? res_kernel<192, true, BF> : res_kernel<192, false, BF>)<<<x.size(0), 192, 0, s>>>(RES_ARGS);
  else (sig ? res_kernel<32, true, BF> : res_kernel<32, false, BF>)<<<Geo<32>::blocks(x.size(0)), Geo<32>::THREADS, 0, s>>>(RES_ARGS);
#undef RES_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_ln_rows(at::Tensor q, at::Tensor k, at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor bk, double eps) {
  const int64_t d = width(q);
  check(q, "q", at::kBFloat16, d);
  check(k, "k", at::kBFloat16, d);
  TORCH_CHECK(q.size(0) == k.size(0), "q / k: the same rows");
  for (auto* w : {&wq, &bq, &wk, &bk})
    TORCH_CHECK(w->scalar_type() == at::kFloat && w->is_contiguous() && w->numel() == d, "LayerNorm affine: fp32 [D]");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto s = at::cuda::getCurrentCUDAStream();
#define QK_ARGS bfw(q), q.stride(0), bfw(k), k.stride(0), wq.data_ptr<float>(), bq.data_ptr<float>(), wk.data_ptr<float>(), \
                bk.data_ptr<float>(), (float)eps, q.size(0)
  if (d == 768) qk_ln_kernel<192, BF><<<q.size(0), 192, 0, s>>>(QK_ARGS);
  else qk_ln_kernel<32, BF><<<Geo<32>::blocks(q.size(0)), Geo<32>::THREADS, 0, s>>>(QK_ARGS);
#undef QK_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_rows_inplace(at::Tensor o, at::Tensor g) {
  const int64_t d = o.size(1);
  check(o, "o", at::kBFloat16, d);
  check(g, "g", at::kBFloat16, d);
  TORCH_CHECK(d % 4 == 0 && o.size(0) == g.size(0), "o / g: the same [rows, D], D % 4 == 0");
  const at::cuda::OptionalCUDAGuard guard(o.device());
  const long n4 = o.size(0) * d / 4;
  gate_kernel<BF><<<flat_grid(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      bfw(o), o.stride(0), bf(g), g.stride(0), (int)d, n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void glu_rows(at::Tensor ab, at::Tensor out, bool up) {
  const int64_t n = out.size(1);
  TORCH_CHECK(ab.scalar_type() == at::kBFloat16 && ab.stride(1) == 1 && ab.size(1) == (up ? 3 : 2) * n && n % 4 == 0,
              "ab: bf16 [rows, (2 | 3) n]");
  check(out, "out", at::kBFloat16, n);
  TORCH_CHECK(out.is_contiguous(), "out contiguous");
  const at::cuda::OptionalCUDAGuard guard(ab.device());
  const long n4 = ab.size(0) * n / 4;
  (up ? glu_kernel<BF, true> : glu_kernel<BF, false>)<<<flat_grid(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf(ab), bfw(out), (int)n, ab.stride(0), n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor parallel_residual(at::Tensor x, std::vector<at::Tensor> ys) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat) && x.numel() % 4 == 0,
              "x: contiguous bf16 / fp32, numel % 4 == 0");
  TORCH_CHECK(!ys.empty() && ys.size() <= 8, "one to eight outputs");
  Outs o{};
  o.n = (int)ys.size();
  for (size_t k = 0; k < ys.size(); ++k) {
    TORCH_CHECK(ys[k].is_contiguous() && ys[k].scalar_type() == x.scalar_type() && ys[k].sizes() == x.sizes(), "outputs: contiguous, like x");
    o.y[k] = ys[k].data_ptr();
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  const long n4 = x.numel() / 4;
  const int grid = (int)std::min<long>((n4 + 255) / 256, 132L * 16);
  auto s = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kBFloat16)
    parallel_residual_kernel<BF><<<grid, 256, 0, s>>>(bf(x), o, bfw(out), n4);
  else
    parallel_residual_kernel<float><<<grid, 256, 0, s>>>(x.data_ptr<float>(), o, out.data_ptr<float>(), n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod_rows", &ln_mod_rows);
  m.def("res_rows", &res_rows);
  m.def("qk_ln_rows", &qk_ln_rows);
  m.def("gate_rows_inplace", &gate_rows_inplace);
  m.def("glu_rows", &glu_rows);
  m.def("parallel_residual", &parallel_residual);
}
