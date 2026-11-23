// atom_f8_sm120.cu -- OPT-IN e4m3 row passes of the RTX 5090 (sm_120a) atom DiT FP8 route (dit_variants/cuda/atom_fp8_sm120.py, docs/fp8.md
// "RTX 5090"). The atom block of train.py's atom_fwd (D = 128, one warp a row, 4 values a lane) with the rows that feed a GEMM written as e4m3
// with a DYNAMIC per-row scale (row max / 448: nothing saturates) instead of bf16; the GEMMs are cuBLASLt e4m3 with row-wise scales
// (torch._scaled_mm). Inference only (no statistics saved); the conditioning table tab [T, 8 D] (row r reads tab row r % T) in bf16.
//
//   ln_mod_q8   xhat = LN(x);  out_i = xhat f(sc_i) + sh_i  (i < n; chain: out_1 = LN(out_0) f(sc_1) + sh_1, Protenix's K / V AdaLN)
//   res_ln_q8   xp = x + sigmoid(gA) ya (fp32, stored for the residual-out pass);  xt = LN(xp) f(scT) + shT
//   gate_q8     og = sigmoid(g) o
//   glu_q8      h = silu(a) b (up: silu(a) b u),  H = 256
// f(s) = sigmoid(s) (``sig``) or s.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace af8 {

constexpr int D = 128, RPB = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  uint2 u = *reinterpret_cast<const uint2*>(p);
  float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.x)), b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float sig(float v) { return 1.f / (1.f + __expf(-v)); }
__device__ __forceinline__ float4 sig4(float4 v) { return make_float4(sig(v.x), sig(v.y), sig(v.z), sig(v.w)); }
__device__ __forceinline__ float4 fma4(float4 a, float4 b, float4 c) {
  return make_float4(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y), fmaf(a.z, b.z, c.z), fmaf(a.w, b.w, c.w));
}
__device__ __forceinline__ float amax4(float4 v) { return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))); }
__device__ __forceinline__ uint32_t q4(float4 v, float s) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(v.y * s), "f"(v.x * s));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(v.w * s), "f"(v.z * s));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// one value vector of NV float4 a lane (column v * 128 + 4 lane) -> e4m3 row + its scale amax / 448
template <int NV>
__device__ __forceinline__ void store_q8(const float4 (&y)[NV], uint8_t* __restrict__ out, float* __restrict__ srow, int lane) {
  float am = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) am = fmaxf(am, amax4(y[v]));
  am = warp_max(am);
  const float inv = am > 0.f ? 448.f / am : 1.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) *reinterpret_cast<uint32_t*>(out + v * 128 + 4 * lane) = q4(y[v], inv);
  if (lane == 0) *srow = am > 0.f ? am / 448.f : 1.f;
}
__device__ __forceinline__ float4 ln4(float4 x, float eps) {
  const float mean = warp_sum(x.x + x.y + x.z + x.w) / D;
  const float4 d = make_float4(x.x - mean, x.y - mean, x.z - mean, x.w - mean);
  const float rstd = rsqrtf(warp_sum(d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w) / D + eps);
  return make_float4(d.x * rstd, d.y * rstd, d.z * rstd, d.w * rstd);
}

struct ModQ {
  const __nv_bfloat16* sc[3];
  const __nv_bfloat16* sh[3];
  uint8_t* out[3];
  float* s[3];
  int n;
};

__global__ void __launch_bounds__(RPB * 32) ln_mod_q8_k(const __nv_bfloat16* __restrict__ X, ModQ m, long sTab, int T, float eps, long M, bool sg,
                                                       bool chain) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RPB + (threadIdx.x >> 5);
  if (row >= M) return;
  const long tok = row % T;
  const int col = 4 * lane;
  const float4 xh = ln4(ld4(X + row * D + col), eps);
  float4 prev = xh;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    if (i < m.n) {
      const float4 s = ld4(m.sc[i] + tok * sTab + col), h = ld4(m.sh[i] + tok * sTab + col);
      const float4 base = (chain && i == 1) ? ln4(prev, eps) : xh;
      const float4 y[1] = {fma4(base, sg ? sig4(s) : s, h)};
      store_q8<1>(y, m.out[i] + row * D, m.s[i] + row, lane);
      prev = y[0];
    }
  }
}

__global__ void __launch_bounds__(RPB * 32) res_ln_q8_k(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ YA,
                                                       const __nv_bfloat16* __restrict__ GA, const __nv_bfloat16* __restrict__ SC,
                                                       const __nv_bfloat16* __restrict__ SH, long sTab, float* __restrict__ XP, uint8_t* __restrict__ XT8,
                                                       float* __restrict__ SXT, int T, float eps, long M, bool sg) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RPB + (threadIdx.x >> 5);
  if (row >= M) return;
  const long tok = row % T;
  const int col = 4 * lane;
  const float4 x = fma4(sig4(ld4(GA + tok * sTab + col)), ld4(YA + row * D + col), ld4(X + row * D + col));
  *reinterpret_cast<float4*>(XP + row * D + col) = x;
  const float4 s = ld4(SC + tok * sTab + col), h = ld4(SH + tok * sTab + col);
  const float4 y[1] = {fma4(ln4(x, eps), sg ? sig4(s) : s, h)};
  store_q8<1>(y, XT8 + row * D, SXT + row, lane);
}

__global__ void __launch_bounds__(RPB * 32) gate_q8_k(const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ G, long sg, uint8_t* __restrict__ OG8,
                                                     float* __restrict__ S, long M) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RPB + (threadIdx.x >> 5);
  if (row >= M) return;
  const int col = 4 * lane;
  const float4 o = ld4(O + row * D + col), g = sig4(ld4(G + row * sg + col));
  const float4 y[1] = {make_float4(o.x * g.x, o.y * g.y, o.z * g.z, o.w * g.w)};
  store_q8<1>(y, OG8 + row * D, S + row, lane);
}

// generic rows (any bf16 [M, W], W = 128 or 256, row stride sx) -> e4m3 + scale
template <int W>
__global__ void __launch_bounds__(RPB * 32) quant_q8_k(const __nv_bfloat16* __restrict__ X, long sx, uint8_t* __restrict__ OUT, float* __restrict__ S, long M) {
  constexpr int NV = W / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RPB + (threadIdx.x >> 5);
  if (row >= M) return;
  float4 y[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) y[v] = ld4(X + row * sx + v * 128 + 4 * lane);
  store_q8<NV>(y, OUT + row * W, S + row, lane);
}

template <bool UP>
__global__ void __launch_bounds__(RPB * 32) glu_q8_k(const __nv_bfloat16* __restrict__ AB, uint8_t* __restrict__ H8, float* __restrict__ S, long M) {
  constexpr int H = 256, NV = H / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RPB + (threadIdx.x >> 5);
  if (row >= M) return;
  const __nv_bfloat16* r = AB + row * (UP ? 3 : 2) * H;
  float4 y[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int c = v * 128 + 4 * lane;
    const float4 a = ld4(r + c), b = ld4(r + H + c);
    y[v] = make_float4(a.x * sig(a.x) * b.x, a.y * sig(a.y) * b.y, a.z * sig(a.z) * b.z, a.w * sig(a.w) * b.w);
    if constexpr (UP) {
      const float4 u = ld4(r + 2 * H + c);
      y[v] = make_float4(y[v].x * u.x, y[v].y * u.y, y[v].z * u.z, y[v].w * u.w);
    }
  }
  store_q8<NV>(y, H8 + row * H, S + row, lane);
}

}  // namespace af8

using namespace af8;

static inline const __nv_bfloat16* bp(const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
static inline uint8_t* u8(const at::Tensor& t) { return reinterpret_cast<uint8_t*>(t.data_ptr()); }
static inline dim3 grid(long M) { return dim3((M + RPB - 1) / RPB); }
static void bf_rows(const at::Tensor& t, int64_t w, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(1) == w && t.is_contiguous(), what, ": bf16 [M, ", w, "] contiguous");
}
static void q_rows(const at::Tensor& o, const at::Tensor& s, int64_t M, int64_t w) {
  TORCH_CHECK(o.is_contiguous() && o.element_size() == 1 && o.size(0) == M && o.size(1) == w, "e4m3 [M, ", w, "]");
  TORCH_CHECK(s.is_contiguous() && s.scalar_type() == at::kFloat && s.numel() == M, "scale fp32 [M]");
}

// x [M, 128] bf16; tab [T, 8 D] bf16; cols: the first column of each scale / shift block; outs / scales: n e4m3 [M, 128] / fp32 [M]
void ln_mod_q8(at::Tensor x, at::Tensor tab, std::vector<int64_t> sc_cols, std::vector<int64_t> sh_cols, std::vector<at::Tensor> outs,
               std::vector<at::Tensor> scales, int64_t T, double eps, bool sg, bool chain) {
  bf_rows(x, D, "x");
  const long M = x.size(0);
  TORCH_CHECK(tab.scalar_type() == at::kBFloat16 && tab.stride(1) == 1 && tab.size(0) == T, "tab: bf16 [T, 8 D]");
  const int n = (int)outs.size();
  TORCH_CHECK(n >= 1 && n <= 3 && (int)sc_cols.size() == n && (int)sh_cols.size() == n && (int)scales.size() == n, "ln_mod_q8: 1-3 outputs");
  ModQ m{};
  m.n = n;
  for (int i = 0; i < n; ++i) {
    q_rows(outs[i], scales[i], M, D);
    m.sc[i] = bp(tab) + sc_cols[i];
    m.sh[i] = bp(tab) + sh_cols[i];
    m.out[i] = u8(outs[i]);
    m.s[i] = scales[i].data_ptr<float>();
  }
  if (M == 0) return;
  ln_mod_q8_k<<<grid(M), RPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(bp(x), m, tab.stride(0), (int)T, (float)eps, M, sg, chain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_ln_q8(at::Tensor x, at::Tensor ya, at::Tensor tab, int64_t g_col, int64_t sc_col, int64_t sh_col, at::Tensor xp, at::Tensor xt8, at::Tensor sxt,
               int64_t T, double eps, bool sg) {
  bf_rows(x, D, "x");
  bf_rows(ya, D, "ya");
  const long M = x.size(0);
  TORCH_CHECK(xp.is_contiguous() && xp.scalar_type() == at::kFloat && xp.numel() == M * D, "xp: fp32 [M, 128]");
  q_rows(xt8, sxt, M, D);
  if (M == 0) return;
  const auto t = bp(tab);
  res_ln_q8_k<<<grid(M), RPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(bp(x), bp(ya), t + g_col, t + sc_col, t + sh_col, tab.stride(0), xp.data_ptr<float>(),
                                                                           u8(xt8), sxt.data_ptr<float>(), (int)T, (float)eps, M, sg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_q8(at::Tensor o, at::Tensor g, at::Tensor og8, at::Tensor s) {
  bf_rows(o, D, "o");
  const long M = o.size(0);
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && g.stride(1) == 1 && g.size(0) == M && g.size(1) == D, "g: bf16 [M, 128]");
  q_rows(og8, s, M, D);
  if (M == 0) return;
  gate_q8_k<<<grid(M), RPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(bp(o), bp(g), g.stride(0), u8(og8), s.data_ptr<float>(), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quant_q8(at::Tensor x, at::Tensor out, at::Tensor s) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x: bf16 rows");
  const long M = x.size(0), W = x.size(1);
  q_rows(out, s, M, W);
  if (M == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  if (W == 128) quant_q8_k<128><<<grid(M), RPB * 32, 0, st>>>(bp(x), x.stride(0), u8(out), s.data_ptr<float>(), M);
  else if (W == 256) quant_q8_k<256><<<grid(M), RPB * 32, 0, st>>>(bp(x), x.stride(0), u8(out), s.data_ptr<float>(), M);
  else TORCH_CHECK(false, "quant_q8: width 128 / 256");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void glu_q8(at::Tensor ab, at::Tensor h8, at::Tensor s, bool up) {
  bf_rows(ab, up ? 768 : 512, "ab");
  const long M = ab.size(0);
  q_rows(h8, s, M, 256);
  if (M == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  if (up) glu_q8_k<true><<<grid(M), RPB * 32, 0, st>>>(bp(ab), u8(h8), s.data_ptr<float>(), M);
  else glu_q8_k<false><<<grid(M), RPB * 32, 0, st>>>(bp(ab), u8(h8), s.data_ptr<float>(), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod_q8", &ln_mod_q8);
  m.def("res_ln_q8", &res_ln_q8);
  m.def("gate_q8", &gate_q8);
  m.def("quant_q8", &quant_q8);
  m.def("glu_q8", &glu_q8);
}
