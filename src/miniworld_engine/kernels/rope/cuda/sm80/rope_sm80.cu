// A100 (sm_80) 3D-RoPE kernels: the standalone rotation and the fused Q/K RMSNorm + rotation (forward, and the fused input backward), bf16 or fp32.
//
//   rotary prefix: the first 2 HALF channels of a head, lo = x[:HALF], hi = x[HALF:2 HALF]:   lo' = lo cos - hi sin,  hi' = hi cos + lo sin   (the tail passes through)
//   fused forward:  x -> n = round(x rsqrt(mean(x^2) + eps)) -> rotate(n)
//   fused backward: g (the gradient of the rotated n) -> gn = round(rotate^T(g)) -> dx = rq (gn - n mean(gn n)),  n = x rq (not rounded)
//
// A row is one (position, tensor, head) of D channels, owned by G = D / CE lanes (CE = 8 bf16 / 4 fp32 per 16-byte chunk), so a warp serves 32 / G rows: for D = 32, 4 heads, a warp
// takes one position's q AND k rows (the cos / sin of the position are read once).  Chunk c of the rotary prefix pairs with chunk c +- HALF / CE of the same row: one shuffle.
// The Q / K inputs are the strided views of the interleaved QKV projection (unit channel stride, 16-byte aligned rows); outputs are contiguous [N, S, H, D].
#include "norm_common.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <map>
#include <utility>

using namespace norms;

namespace {

constexpr int NT = 128;                      // threads of a CTA: a one-shot grid of small CTAs (the hardware balances the load; a persistent grid-stride loop was 6-16 % slower on this streaming kernel)
constexpr int NW = NT / 32;

struct Str { long n, s, h; };   // element strides of an [N, S, H, D] view (the channel stride is 1)

template <typename XT> __device__ __forceinline__ float rnd(float v) { return v; }
template <> __device__ __forceinline__ float rnd<bf>(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// MODE 0: standalone rotation of Q (K, GQ, GK unused);  1: fused norm + rotation (Q, K -> OQ, OK);  2: its backward (Q, K, GQ, GK -> OQ = dq, OK = dk)
template <typename XT, int D, int MODE>
__global__ void __launch_bounds__(NT) rope_kernel(const XT* __restrict__ Q, const XT* __restrict__ K, const XT* __restrict__ GQ, const XT* __restrict__ GK,
                                                  XT* __restrict__ OQ, XT* __restrict__ OK, const float* __restrict__ C, const float* __restrict__ S, Str sq, Str sk,
                                                  Str sgq, Str sgk, long cs_n, long cs_s, int SEQ, int H, int HALF, long NPOS, float eps, float ssign) {
  constexpr int CE = Chunk<XT>::CE, G = D / CE, RPW = 32 / G;
  constexpr int NTEN = MODE == 0 ? 1 : 2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = lane & (G - 1), sub = lane / G;
  const int P = HALF / CE;                               // chunks of the rotary half
  const bool rot = c < 2 * P, lo = c < P;
  const int pc = lo ? c + P : c - P;                      // the partner chunk (rotary lanes)
  const int src = (lane - c) + (rot ? pc : c);
  const long total = NPOS * NTEN * H;
  {
    const long it = (long)blockIdx.x * NW + warp;
    if (it * RPW >= total) return;
    const int rr = (int)(it * RPW + sub);                // total < 2^31 (checked by the launcher): 32-bit index arithmetic
    const bool valid = rr < (int)total;
    const int pos = rr / (NTEN * H);
    const int rem = rr - pos * (NTEN * H);
    const int t = rem / H, h = rem - t * H;
    const int n = pos / SEQ, s = pos - n * SEQ;
    const Str& si = t == 0 ? sq : sk;
    const XT* xp = (t == 0 ? Q : K) + n * si.n + s * si.s + h * si.h + c * CE;
    uint4 xr = make_uint4(0u, 0u, 0u, 0u), gr = make_uint4(0u, 0u, 0u, 0u);
    if (valid) {
      xr = Chunk<XT>::load_raw(xp);
      if constexpr (MODE == 2) {
        const Str& sg = t == 0 ? sgq : sgk;
        gr = Chunk<XT>::load_raw((t == 0 ? GQ : GK) + n * sg.n + s * sg.s + h * sg.h + c * CE);
      }
    }
    // the angles of this position (the rotary lanes): columns (c mod P) CE ..+CE of cos / sin
    float cs[CE], sn[CE];
#pragma unroll
    for (int e = 0; e < CE; ++e) { cs[e] = 1.f; sn[e] = 0.f; }
    if (valid && rot) {
      const long ao = n * cs_n + s * cs_s + (long)(lo ? c : c - P) * CE;
#pragma unroll
      for (int k = 0; k < CE / 4; ++k) {
        const float4 a = *(reinterpret_cast<const float4*>(C + ao) + k), b = *(reinterpret_cast<const float4*>(S + ao) + k);
        cs[4 * k] = a.x; cs[4 * k + 1] = a.y; cs[4 * k + 2] = a.z; cs[4 * k + 3] = a.w;
        sn[4 * k] = b.x * ssign; sn[4 * k + 1] = b.y * ssign; sn[4 * k + 2] = b.z * ssign; sn[4 * k + 3] = b.w * ssign;
      }
    }
    float x[CE];
    Chunk<XT>::unpack(xr, x);
    float v[CE];                                         // what is rotated (the normalised x, or the gradient)
    float rq = 1.f;
    if constexpr (MODE == 0) {
#pragma unroll
      for (int e = 0; e < CE; ++e) v[e] = x[e];
    } else {
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < CE; ++e) ss += x[e] * x[e];
      ss = group_sum<G>(ss);
      rq = rsqrtf(ss * (1.f / (float)D) + eps);
      if constexpr (MODE == 1) {
#pragma unroll
        for (int e = 0; e < CE; ++e) v[e] = rnd<XT>(x[e] * rq);
      } else {
        Chunk<XT>::unpack(gr, v);
      }
    }
    // the partner chunk's values (bf16 packed for the shuffle: v is exactly representable in bf16 in the fused modes; the standalone rotation reads bf16 inputs)
    float pv[CE];
    if constexpr (sizeof(XT) == 2) {
      uint4 pk;
      pk.x = pack_bf2(v[0], v[1]); pk.y = pack_bf2(v[2], v[3]); pk.z = pack_bf2(v[4], v[5]); pk.w = pack_bf2(v[6], v[7]);
      pk.x = __shfl_sync(0xffffffffu, pk.x, src); pk.y = __shfl_sync(0xffffffffu, pk.y, src);
      pk.z = __shfl_sync(0xffffffffu, pk.z, src); pk.w = __shfl_sync(0xffffffffu, pk.w, src);
      Chunk<XT>::unpack(pk, pv);
    } else {
#pragma unroll
      for (int e = 0; e < CE; ++e) pv[e] = __shfl_sync(0xffffffffu, v[e], src);
    }
    float o[CE];
    if constexpr (MODE == 2) {
      // rotate^T: lo' = g_lo cos + g_hi sin, hi' = g_hi cos - g_lo sin; non-rotary channels pass (cos 1, sin 0); rounded to the activation dtype
      float gn[CE], dot = 0.f;
#pragma unroll
      for (int e = 0; e < CE; ++e) {
        gn[e] = rot ? rnd<XT>(v[e] * cs[e] + (lo ? 1.f : -1.f) * (pv[e] * sn[e])) : v[e];
        dot += gn[e] * (x[e] * rq);
      }
      dot = group_sum<G>(dot) * (1.f / (float)D);
#pragma unroll
      for (int e = 0; e < CE; ++e) o[e] = rq * (gn[e] - (x[e] * rq) * dot);
    } else {
#pragma unroll
      for (int e = 0; e < CE; ++e) o[e] = rot ? v[e] * cs[e] + (lo ? -1.f : 1.f) * (pv[e] * sn[e]) : v[e];
    }
    if (valid) {
      XT* op = (t == 0 ? OQ : OK) + (long)(pos * H + h) * D + c * CE;
      Chunk<XT>::store(op, o);
    }
  }
}

int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

template <typename K> int blocks_per_sm(K kernel) {
  static std::map<const void*, int> cache;
  const void* key = reinterpret_cast<const void*>(kernel);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, NT, 0);
  n = std::max(n, 1);
  cache[key] = n;
  return n;
}

Str strides_of(const at::Tensor& t) { return Str{t.stride(0), t.stride(1), t.stride(2)}; }
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

template <typename XT> void check_view(const char* what, const at::Tensor& t, int D) {
  constexpr int CE = Chunk<XT>::CE;
  TORCH_CHECK(t.is_cuda() && t.dim() == 4 && t.size(3) == D && t.stride(3) == 1, what, ": a CUDA [N, S, H, D] view with unit channel stride");
  TORCH_CHECK(t.stride(0) % CE == 0 && t.stride(1) % CE == 0 && t.stride(2) % CE == 0 && aligned16(t.data_ptr()), what, ": 16-byte aligned rows");
  TORCH_CHECK(t.size(0) * t.size(1) * t.size(2) * 2 < (1L << 31), what, ": too many rows");
}

void check_angles(const at::Tensor& cos, const at::Tensor& sin, int64_t N, int64_t S) {
  TORCH_CHECK(cos.is_cuda() && cos.scalar_type() == at::kFloat && cos.sizes() == sin.sizes() && cos.dim() == 3 && cos.size(1) == S && (cos.size(0) == N || cos.size(0) == 1) &&
              cos.stride(2) == 1 && sin.stride(0) == cos.stride(0) && sin.stride(1) == cos.stride(1) && sin.stride(2) == 1, "cos / sin: fp32 [N or 1, S, HALF] with unit last stride");
  TORCH_CHECK(cos.size(2) % 4 == 0 && cos.stride(0) % 4 == 0 && cos.stride(1) % 4 == 0 && aligned16(cos.data_ptr()) && aligned16(sin.data_ptr()), "cos / sin: 16-byte aligned rows");
}

template <typename XT, int D, int MODE>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& gq, const at::Tensor& gk, at::Tensor& oq, at::Tensor& ok, const at::Tensor& cos, const at::Tensor& sin,
            double eps, double ssign) {
  constexpr int CE = Chunk<XT>::CE, G = D / CE, RPW = 32 / G;
  constexpr int NTEN = MODE == 0 ? 1 : 2;
  const long N = q.size(0), SEQ = q.size(1), H = q.size(2), HALF = cos.size(2);
  TORCH_CHECK(HALF % CE == 0 && 2 * HALF <= D, "rope: the rotary half must be a multiple of one 16-byte chunk and fit the head");
  const long NPOS = N * SEQ;
  const long total = NPOS * NTEN * H;
  auto kern = rope_kernel<XT, D, MODE>;
  const long iters = (total + RPW - 1) / RPW;
  const long grid = (iters + NW - 1) / NW;
  const Str none{0, 0, 0};
  const long csn = cos.size(0) == 1 ? 0 : cos.stride(0), css = cos.stride(1);
  const XT* Q = reinterpret_cast<const XT*>(q.data_ptr());
  const XT* K = MODE == 0 ? Q : reinterpret_cast<const XT*>(k.data_ptr());
  const XT* GQ = MODE == 2 ? reinterpret_cast<const XT*>(gq.data_ptr()) : Q;
  const XT* GK = MODE == 2 ? reinterpret_cast<const XT*>(gk.data_ptr()) : Q;
  XT* OQ = reinterpret_cast<XT*>(oq.data_ptr());
  XT* OK = MODE == 0 ? OQ : reinterpret_cast<XT*>(ok.data_ptr());
  kern<<<(unsigned)grid, NT, 0, at::cuda::getCurrentCUDAStream()>>>(Q, K, GQ, GK, OQ, OK, cos.data_ptr<float>(), sin.data_ptr<float>(), strides_of(q), MODE == 0 ? none : strides_of(k),
                                                          MODE == 2 ? strides_of(gq) : none, MODE == 2 ? strides_of(gk) : none, csn, css, (int)SEQ, (int)H, (int)HALF, NPOS,
                                                          (float)eps, (float)ssign);
}

template <typename XT, int MODE>
void by_width(int D, const at::Tensor& q, const at::Tensor& k, const at::Tensor& gq, const at::Tensor& gk, at::Tensor& oq, at::Tensor& ok, const at::Tensor& cos,
              const at::Tensor& sin, double eps, double ssign) {
  if (D == 32) launch<XT, 32, MODE>(q, k, gq, gk, oq, ok, cos, sin, eps, ssign);
  else if (D == 64) launch<XT, 64, MODE>(q, k, gq, gk, oq, ok, cos, sin, eps, ssign);
  else if (D == 128) launch<XT, 128, MODE>(q, k, gq, gk, oq, ok, cos, sin, eps, ssign);
  else TORCH_CHECK(false, "rope: head dim 32, 64 or 128");
}

template <int MODE>
void by_dtype(const at::Tensor& q, const at::Tensor& k, const at::Tensor& gq, const at::Tensor& gk, at::Tensor& oq, at::Tensor& ok, const at::Tensor& cos,
              const at::Tensor& sin, double eps, double ssign) {
  const int D = (int)q.size(3);
  auto run = [&](auto tag) {
    using XT = decltype(tag);
    check_view<XT>("q", q, D);
    if (MODE != 0) check_view<XT>("k", k, D);
    if (MODE == 2) { check_view<XT>("gq", gq, D); check_view<XT>("gk", gk, D); }
    by_width<XT, MODE>(D, q, k, gq, gk, oq, ok, cos, sin, eps, ssign);
  };
  if (q.scalar_type() == at::kBFloat16) {
    run(bf{});
  } else {
    TORCH_CHECK(q.scalar_type() == at::kFloat, "rope: bf16 or fp32");
    run(float{});
  }
}

}  // namespace

// out = rotate(x) over the leading 2 HALF channels (x [N, S, H, D] strided view, cos / sin fp32 [N or 1, S, HALF]); ssign = -1 rotates by the opposite angle (the backward)
void rope_apply(at::Tensor x, at::Tensor cos, at::Tensor sin, at::Tensor out, double ssign) {
  c10::cuda::CUDAGuard guard(x.device());
  check_angles(cos, sin, x.size(0), x.size(1));
  TORCH_CHECK(out.is_contiguous() && out.sizes() == x.sizes() && out.scalar_type() == x.scalar_type(), "rope_apply: out");
  if (x.numel() == 0) return;
  by_dtype<0>(x, x, x, x, out, out, cos, sin, 0.0, ssign);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "rope_apply: launch failed");
}

// (oq, ok) = rotate(rmsnorm(q)), rotate(rmsnorm(k)) over the head dim (no weight)
void qk_norm_rope_fwd(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin, at::Tensor oq, at::Tensor ok, double eps) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.sizes() == k.sizes() && q.scalar_type() == k.scalar_type(), "q / k: one shape and dtype");
  check_angles(cos, sin, q.size(0), q.size(1));
  TORCH_CHECK(oq.is_contiguous() && ok.is_contiguous() && oq.sizes() == q.sizes() && ok.sizes() == q.sizes() && oq.scalar_type() == q.scalar_type() && ok.scalar_type() == q.scalar_type(),
              "qk_norm_rope_fwd: outputs");
  if (q.numel() == 0) return;
  by_dtype<1>(q, k, q, q, oq, ok, cos, sin, eps, 1.0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "qk_norm_rope_fwd: launch failed");
}

// (dq, dk) from the inputs and the gradients of the rotated outputs
void qk_norm_rope_bwd(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin, at::Tensor gq, at::Tensor gk, at::Tensor dq, at::Tensor dk, double eps) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == gq.sizes() && q.sizes() == gk.sizes() && q.scalar_type() == k.scalar_type() && q.scalar_type() == gq.scalar_type() &&
              q.scalar_type() == gk.scalar_type(), "q / k / gq / gk: one shape and dtype");
  check_angles(cos, sin, q.size(0), q.size(1));
  TORCH_CHECK(dq.is_contiguous() && dk.is_contiguous() && dq.sizes() == q.sizes() && dk.sizes() == q.sizes() && dq.scalar_type() == q.scalar_type() && dk.scalar_type() == q.scalar_type(),
              "qk_norm_rope_bwd: outputs");
  if (q.numel() == 0) return;
  by_dtype<2>(q, k, gq, gk, dq, dk, cos, sin, eps, 1.0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "qk_norm_rope_bwd: launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_apply", &rope_apply);
  m.def("qk_norm_rope_fwd", &qk_norm_rope_fwd);
  m.def("qk_norm_rope_bwd", &qk_norm_rope_bwd);
}
