// token_dit_fp8_rows.cu -- EXPERIMENTAL (engine 3.0.0 line, opt-in, default off): the two AdaLN row passes of the bf16 token DiT
// inference block (CondFold form, docs/h100/token_dit.md) with an e4m3 output and a DYNAMIC per-row scale, for the e4m3 GEMMs that
// read them (the q|k|v|g projection and the expand + SwiGLU GEMM, docs/fp8.md "Token DiT").
//
//   adaln_in_cf_q8       out8 = e4m3(AdaLN(xin) / s), s[row] = max|AdaLN(xin)| / 448   (stats of the conditioning row stored as the bf16 pass does)
//   resgate_adaln_cf_q8  out8 = e4m3(AdaLN(xin + sigmoid(gA) ya) / s), s[row] likewise
//
// The value quantized is the fp32 AdaLN output the bf16 pass rounds to bf16 (same expressions, same order: token_dit_rows.cu), so
// the e4m3 operand is one rounding of the exact row, not a rounding of the bf16 row. A per-row scale needs no calibration and no
// overflow guard: the row's own maximum maps to 448 (no saturation by construction), whatever the noise level of the step.
// The device code below repeats the few CondFold helpers of token_dit_rows.cu (that file's build is the default route's and stays
// byte-for-byte unchanged).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "token_dit_common.cuh"

namespace {
using namespace tdr;

struct CondFold {
  const float* r = nullptr;
  const float* b = nullptr;
  const float* gb = nullptr;
  float2* st = nullptr;
  const __nv_bfloat16* c = nullptr;
  long sc = 0;
  int dc = 0;
  float eps = 1e-5f;
  int js = 0;
};

template <int NT>
__device__ __forceinline__ void cond_fix(float4& m, float4& h, const CondFold& cf, int js, float2 s) {
  constexpr int D = NT * 4;
  const int col = threadIdx.x * 4;
  const float4 rs = V4<float>::load(cf.r + js * D + col), rb = V4<float>::load(cf.r + (js + 1) * D + col);
  const float4 bs = V4<float>::load(cf.b + js * D + col), bb = V4<float>::load(cf.b + (js + 1) * D + col);
  m.x = s.y * fmaf(-s.x, rs.x, m.x) + bs.x; m.y = s.y * fmaf(-s.x, rs.y, m.y) + bs.y;
  m.z = s.y * fmaf(-s.x, rs.z, m.z) + bs.z; m.w = s.y * fmaf(-s.x, rs.w, m.w) + bs.w;
  h.x = s.y * fmaf(-s.x, rb.x, h.x) + bb.x; h.y = s.y * fmaf(-s.x, rb.y, h.y) + bb.y;
  h.z = s.y * fmaf(-s.x, rb.z, h.z) + bb.z; h.w = s.y * fmaf(-s.x, rb.w, h.w) + bb.w;
}

template <int NT>
__device__ __forceinline__ float4 gate_cf(const float* G, const CondFold& cf, int j) {
  const float4 g = V4<float>::load(G + (4 + j) * NT * 4 + threadIdx.x * 4);
  const float4 b = V4<float>::load(cf.gb + j * NT * 4 + threadIdx.x * 4);
  return make_float4(g.x + b.x, g.y + b.y, g.z + b.z, g.w + b.w);
}

template <int NT>
__device__ __forceinline__ float2 block_sum2(float2 v, float2* red) {
  v.x = warp_sum(v.x); v.y = warp_sum(v.y);
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (l == 0) red[w] = v;
  __syncthreads();
  float2 t = make_float2(0.f, 0.f);
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) { t.x += red[i].x; t.y += red[i].y; }
  __syncthreads();
  return t;
}

template <int NT>
// the row maximum, the block's LAST reduction: red is used once (no barrier after the read)
__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) t = fmaxf(t, red[i]);
  return t;
}

// o = AdaLN of this thread's 4 columns (the bf16 pass's adaln_out<CF = true, SIG = true> before its store), then the row's e4m3 image
// The AdaLN scale of the model: sigmoid (SIG) or, Chai-1's identity AdaLN, the table itself (its "+ 1" is in the bias table)
template <bool SIG>
__device__ __forceinline__ float scl(float v) {
  if constexpr (SIG) return sigm(v); else return v;
}

template <int NT, bool SIG = true>
__device__ __forceinline__ void adaln_q8(float a, float b, float c, float d, float rstd, const float* ms, const float* mb,
                                         uint8_t* out8, float* srow, const CondFold& cf, int js, float2 s, float* red) {
  const int col = threadIdx.x * 4;
  float4 m = V4<float>::load(ms + col), h = V4<float>::load(mb + col), o;
  cond_fix<NT>(m, h, cf, js, s);
  o.x = a * rstd * scl<SIG>(m.x) + h.x;
  o.y = b * rstd * scl<SIG>(m.y) + h.y;
  o.z = c * rstd * scl<SIG>(m.z) + h.z;
  o.w = d * rstd * scl<SIG>(m.w) + h.w;
  const float amax = block_max<NT>(fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))), red);
  const float sc = amax > 0.f ? amax / 448.f : 1.f, inv = 1.f / sc;
  const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(o.x * inv, o.y * inv), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(o.z * inv, o.w * inv), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint32_t*>(out8 + col) = (uint32_t)lo | ((uint32_t)hi << 16);
  if (threadIdx.x == 0) *srow = sc;
}

template <int NT>
__global__ void __launch_bounds__(NT) adaln_in_cf_q8_kernel(const __nv_bfloat16* __restrict__ XIN, const float* __restrict__ G,
    uint8_t* __restrict__ OUT, float* __restrict__ SROW, int P, long sxin, long sg, float eps, CondFold cf) {
  constexpr int D = NT * 4;
  __shared__ float2 red[NT / 32];
  __shared__ float redm[NT / 32];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const long row = blockIdx.x;
  const int tok = row % P, col = threadIdx.x * 4;
  const float4 x = V4<__nv_bfloat16>::load(XIN + row * sxin + col);
  const bool on = col < cf.dc;
  const float4 v = on ? V4<__nv_bfloat16>::load(cf.c + (long)tok * cf.sc + col) : make_float4(0.f, 0.f, 0.f, 0.f);
  const float2 m = block_sum2<NT>(make_float2(x.x + x.y + x.z + x.w, v.x + v.y + v.z + v.w), red);
  const float mean = m.x / D, cmean = m.y / (float)cf.dc;
  const float a = x.x - mean, b = x.y - mean, c = x.z - mean, d = x.w - mean;
  const float ca = v.x - cmean, cb = v.y - cmean, cc = v.z - cmean, cd = v.w - cmean;
  const float2 q = block_sum2<NT>(make_float2(a * a + b * b + c * c + d * d, on ? ca * ca + cb * cb + cc * cc + cd * cd : 0.f), red);
  const float rstd = rsqrtf(q.x / D + eps);
  const float2 s = make_float2(cmean, rsqrtf(q.y / (float)cf.dc + cf.eps));
  if (row < P && threadIdx.x == 0) cf.st[tok] = s;
  const float* Gt = G + (long)tok * sg;
  adaln_q8<NT>(a, b, c, d, rstd, Gt, Gt + D, OUT + row * D, SROW + row, cf, 0, s, redm);
}

// The parallel block (Chai-1: identity AdaLN, SIG = false; the transition's AdaLN reads the same LN(xin)): out = AdaLN_{tables 0, 1}(xin) as
// bf16 (the attention's q|k|v|g GEMM stays bf16) and out2 = e4m3(AdaLN_{tables 2, 3}(xin) / s2), s2[row] = its row max / 448 (the expand +
// SwiGLU GEMM's operand). The values are the bf16 pass's (adaln_in_cf_kernel<NT, true, SIG>) before its store: out is bit-identical to it.
template <int NT, bool SIG>
__global__ void __launch_bounds__(NT) adaln_in_cf_par_q8_kernel(const __nv_bfloat16* __restrict__ XIN, const float* __restrict__ G,
    __nv_bfloat16* __restrict__ OUT, uint8_t* __restrict__ OUT8, float* __restrict__ SROW, int P, long sxin, long sg, float eps, CondFold cf) {
  constexpr int D = NT * 4;
  __shared__ float2 red[NT / 32];
  __shared__ float redm[NT / 32];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const long row = blockIdx.x;
  const int tok = row % P, col = threadIdx.x * 4;
  const float4 x = V4<__nv_bfloat16>::load(XIN + row * sxin + col);
  const bool on = SIG && col < cf.dc;
  const float4 v = on ? V4<__nv_bfloat16>::load(cf.c + (long)tok * cf.sc + col) : make_float4(0.f, 0.f, 0.f, 0.f);
  const float2 m = block_sum2<NT>(make_float2(x.x + x.y + x.z + x.w, v.x + v.y + v.z + v.w), red);
  const float mean = m.x / D, cmean = m.y / (float)cf.dc;
  const float a = x.x - mean, b = x.y - mean, c = x.z - mean, d = x.w - mean;
  const float ca = v.x - cmean, cb = v.y - cmean, cc = v.z - cmean, cd = v.w - cmean;
  const float2 q = block_sum2<NT>(make_float2(a * a + b * b + c * c + d * d, on ? ca * ca + cb * cb + cc * cc + cd * cd : 0.f), red);
  const float rstd = rsqrtf(q.x / D + eps);
  const float2 s = SIG ? make_float2(cmean, rsqrtf(q.y / (float)cf.dc + cf.eps)) : make_float2(0.f, 1.f);
  if (SIG && row < P && threadIdx.x == 0) cf.st[tok] = s;
  const float* Gt = G + (long)tok * sg;
  {  // tables 0, 1 -> bf16 (the bf16 route's adaln_out<NT, float, bf16, false, SIG, true>)
    float4 mm = V4<float>::load(Gt + col), hh = V4<float>::load(Gt + D + col), o;
    cond_fix<NT>(mm, hh, cf, 0, s);
    o.x = a * rstd * scl<SIG>(mm.x) + hh.x; o.y = b * rstd * scl<SIG>(mm.y) + hh.y;
    o.z = c * rstd * scl<SIG>(mm.z) + hh.z; o.w = d * rstd * scl<SIG>(mm.w) + hh.w;
    V4<__nv_bfloat16>::store(OUT + row * D + col, o);
  }
  adaln_q8<NT, SIG>(a, b, c, d, rstd, Gt + 2 * D, Gt + 3 * D, OUT8 + row * D, SROW + row, cf, 2, s, redm);
}

template <int NT>
__global__ void __launch_bounds__(NT) resgate_adaln_cf_q8_kernel(const __nv_bfloat16* __restrict__ XIN, const __nv_bfloat16* __restrict__ YA,
    const float* __restrict__ G, uint8_t* __restrict__ OUT, float* __restrict__ SROW, int P, long sxin, long sya, long sg, float eps, CondFold cf) {
  constexpr int D = NT * 4;
  __shared__ float red[NT / 32];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const long row = blockIdx.x;
  const int tok = row % P, col = threadIdx.x * 4;
  const float* Gt = G + (long)tok * sg;
  float4 x = V4<__nv_bfloat16>::load(XIN + row * sxin + col);
  const float4 y = V4<__nv_bfloat16>::load(YA + row * sya + col), g = gate_cf<NT>(Gt, cf, 0);
  x.x += sigm(g.x) * y.x; x.y += sigm(g.y) * y.y; x.z += sigm(g.z) * y.z; x.w += sigm(g.w) * y.w;
  const float mean = block_sum<NT>(x.x + x.y + x.z + x.w, red) / D;
  const float a = x.x - mean, b = x.y - mean, c = x.z - mean, d = x.w - mean;
  const float rstd = rsqrtf(block_sum<NT>(a * a + b * b + c * c + d * d, red) / D + eps);
  adaln_q8<NT>(a, b, c, d, rstd, Gt + cf.js * D, Gt + (cf.js + 1) * D, OUT + row * D, SROW + row, cf, cf.js, cf.st[tok], red);
}

// bf16 [M, D] rows -> e4m3 [M, D] with a dynamic per-row scale (the attention output og before the out projection, h before the squeeze)
template <int NT, int VPT>
__global__ void __launch_bounds__(NT) quant_rows_kernel(const __nv_bfloat16* __restrict__ X, long sx, uint8_t* __restrict__ OUT, float* __restrict__ SROW, int D) {
  __shared__ float red[NT / 32];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const long row = blockIdx.x;
  float v[VPT * 4];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int col = (i * NT + threadIdx.x) * 4;
    const float4 x = col < D ? V4<__nv_bfloat16>::load(X + row * sx + col) : make_float4(0.f, 0.f, 0.f, 0.f);
    v[4 * i] = x.x; v[4 * i + 1] = x.y; v[4 * i + 2] = x.z; v[4 * i + 3] = x.w;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
  }
  amax = block_max<NT>(amax, red);
  const float sc = amax > 0.f ? amax / 448.f : 1.f, inv = 1.f / sc;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int col = (i * NT + threadIdx.x) * 4;
    if (col >= D) continue;
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[4 * i] * inv, v[4 * i + 1] * inv), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[4 * i + 2] * inv, v[4 * i + 3] * inv), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint32_t*>(OUT + row * D + col) = (uint32_t)lo | ((uint32_t)hi << 16);
  }
  if (threadIdx.x == 0) SROW[row] = sc;
}

template <typename C> auto ptr(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }

template <typename... KArgs, typename... Args>
void launch_rows(void (*kern)(KArgs...), int64_t M, int nt, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((unsigned)M);
  cfg.blockDim = dim3((unsigned)nt);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
  TORCH_CHECK(err == cudaSuccess, "token DiT fp8 rows launch: ", cudaGetErrorString(err));
}

CondFold make_fold(const at::Tensor& r, const at::Tensor& b, const at::Tensor& b2, const at::Tensor& stats, const c10::optional<at::Tensor>& c,
                   double ceps, int js) {
  CondFold f;
  f.r = ptr<const float>(r); f.b = ptr<const float>(b); f.gb = ptr<const float>(b2);
  f.st = reinterpret_cast<float2*>(stats.data_ptr()); f.eps = (float)ceps; f.js = js;
  if (c.has_value()) { f.c = ptr<const __nv_bfloat16>(*c); f.sc = c->stride(0); f.dc = (int)c->size(1); }
  return f;
}

void check_q8(const at::Tensor& xin, const at::Tensor& g, const at::Tensor& out8, const at::Tensor& srow) {
  const int64_t M = xin.size(0), D = xin.size(1);
  TORCH_CHECK(D == 768, "fp8 rows: width 768");
  TORCH_CHECK(xin.scalar_type() == at::kBFloat16 && xin.stride(1) == 1 && g.scalar_type() == at::kFloat && g.size(1) == 6 * D && g.stride(1) == 1,
              "fp8 rows: bf16 xin, fp32 G [P, 6 D]");
  TORCH_CHECK(out8.scalar_type() == at::kFloat8_e4m3fn && out8.is_contiguous() && out8.size(0) == M && out8.size(1) == D, "fp8 rows: e4m3 out [M, D]");
  TORCH_CHECK(srow.scalar_type() == at::kFloat && srow.is_contiguous() && srow.numel() == M, "fp8 rows: fp32 row scales [M]");
}
}  // namespace

void adaln_in_rows_cf_q8(at::Tensor xin, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor c, at::Tensor stats,
                         at::Tensor out8, at::Tensor srow, double eps, double ceps, bool pdl) {
  check_q8(xin, g, out8, srow);
  const int64_t M = xin.size(0), P = g.size(0);
  TORCH_CHECK(M % P == 0 && c.scalar_type() == at::kBFloat16 && c.stride(1) == 1 && c.size(1) <= 768, "adaln_in_rows_cf_q8: shapes");
  const CondFold f = make_fold(r, b, b2, stats, c, ceps, 0);
  const at::cuda::CUDAGuard gd(xin.device());
  launch_rows(adaln_in_cf_q8_kernel<192>, M, 192, pdl, ptr<const __nv_bfloat16>(xin), ptr<const float>(g), ptr<uint8_t>(out8),
              ptr<float>(srow), (int)P, (long)xin.stride(0), (long)g.stride(0), (float)eps, f);
}

// The parallel block (Chai-1): out = bf16 AdaLN(xin) with tables 0, 1; out8 / srow = e4m3 AdaLN(xin) with tables 2, 3 and its row scale.
void adaln_in_rows_cf_par_q8(at::Tensor xin, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor c, at::Tensor stats,
                             at::Tensor out, at::Tensor out8, at::Tensor srow, double eps, double ceps, bool pdl, bool sig) {
  check_q8(xin, g, out8, srow);
  const int64_t M = xin.size(0), P = g.size(0);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M && out.size(1) == 768, "adaln_in_rows_cf_par_q8: bf16 out [M, 768]");
  TORCH_CHECK(M % P == 0 && c.scalar_type() == at::kBFloat16 && c.stride(1) == 1 && c.size(1) <= 768, "adaln_in_rows_cf_par_q8: shapes");
  const CondFold f = make_fold(r, b, b2, stats, c, ceps, 0);
  const at::cuda::CUDAGuard gd(xin.device());
  auto kern = sig ? adaln_in_cf_par_q8_kernel<192, true> : adaln_in_cf_par_q8_kernel<192, false>;
  launch_rows(kern, M, 192, pdl, ptr<const __nv_bfloat16>(xin), ptr<const float>(g), ptr<__nv_bfloat16>(out), ptr<uint8_t>(out8),
              ptr<float>(srow), (int)P, (long)xin.stride(0), (long)g.stride(0), (float)eps, f);
}

void resgate_adaln_cf_q8(at::Tensor xin, at::Tensor ya, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor stats,
                         at::Tensor out8, at::Tensor srow, double eps, bool pdl) {
  check_q8(xin, g, out8, srow);
  const int64_t M = xin.size(0), P = g.size(0);
  TORCH_CHECK(M % P == 0 && ya.scalar_type() == at::kBFloat16 && ya.stride(1) == 1, "resgate_adaln_cf_q8: shapes");
  const CondFold f = make_fold(r, b, b2, stats, c10::nullopt, 1e-5, 2);
  const at::cuda::CUDAGuard gd(xin.device());
  launch_rows(resgate_adaln_cf_q8_kernel<192>, M, 192, pdl, ptr<const __nv_bfloat16>(xin), ptr<const __nv_bfloat16>(ya), ptr<const float>(g),
              ptr<uint8_t>(out8), ptr<float>(srow), (int)P, (long)xin.stride(0), (long)ya.stride(0), (long)g.stride(0), (float)eps, f);
}

void quant_rows(at::Tensor x, at::Tensor out8, at::Tensor srow, bool pdl) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0 && D % 4 == 0 && D <= 2048, "quant_rows: bf16 [M, D <= 2048]");
  TORCH_CHECK(out8.scalar_type() == at::kFloat8_e4m3fn && out8.is_contiguous() && out8.size(0) == M && out8.size(1) == D, "quant_rows: e4m3 out");
  TORCH_CHECK(srow.scalar_type() == at::kFloat && srow.numel() == M, "quant_rows: fp32 scales [M]");
  const at::cuda::CUDAGuard gd(x.device());
  if (D <= 768) launch_rows(quant_rows_kernel<192, 1>, M, 192, pdl, ptr<const __nv_bfloat16>(x), (long)x.stride(0), ptr<uint8_t>(out8), ptr<float>(srow), (int)D);
  else launch_rows(quant_rows_kernel<256, 2>, M, 256, pdl, ptr<const __nv_bfloat16>(x), (long)x.stride(0), ptr<uint8_t>(out8), ptr<float>(srow), (int)D);
}
