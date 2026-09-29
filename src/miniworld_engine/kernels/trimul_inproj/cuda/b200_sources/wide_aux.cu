// wide_aux.cu -- small pieces around the B200 wide TriMul kernels (k1w front, k3w output):
//   ln_stats   per-token mean / rstd over the H channels of the channel-major contraction output t [H, M] (k3w's LN_out fold)
//   fold_prep  k3w's folded weight operands bf16(Wp g_o), bf16(Wg g_i) and their row sums / bias projections
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace wide {

__device__ __forceinline__ float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// ---------------------------------------------------------------------------------------------------------------- ln_stats
// t is channel-major [H, M]: every read covers 512 contiguous bytes (256 tokens) of a channel row (128-byte rows ran HBM at
// ~3 TB/s). Block = 256 tokens (lane = 8 tokens), 8 warps split the channels; each thread keeps pivot-shifted sums (pivot = its
// first channel's value), partitions are merged with Chan's formula. (Welford per element costs an fp32 division each and was
// compute-bound.)
// TPL tokens per lane: 8 (one 16-byte load per channel row, 256 tokens per block) or, when that leaves fewer than two blocks
// per SM (L <= 256), 2 (one 4-byte load, 64 tokens per block): 2x faster there (D512 bidirectional L128 21 -> 11 us), equal
// from L384 on, where both run at HBM bandwidth. The 16-deep unroll keeps enough loads in flight for the 4-byte variant.
template <int H, int TPL>
__global__ void __launch_bounds__(256) ln_stats_kernel(const __nv_bfloat16* __restrict__ t, float* __restrict__ mean,
                                                        float* __restrict__ rstd, int M, float eps) {
  constexpr int CW = H / 8, BT = 32 * TPL, NW = TPL / 2;     // channels per warp, tokens per block, 32-bit words per lane
  __shared__ float pm[8][BT], pq[8][BT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tok0 = blockIdx.x * BT + lane * TPL;
  const __nv_bfloat16* base = t + (size_t)(warp * CW) * M + tok0;
  auto load = [&](size_t off, uint32_t (&w)[NW]) {
    if constexpr (TPL == 8) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + off);
      w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
    } else {
      w[0] = *reinterpret_cast<const uint32_t*>(base + off);
    }
  };
  float piv[TPL], s[TPL], q[TPL];
  {
    uint32_t w[NW];
    load(0, w);
#pragma unroll
    for (int j = 0; j < NW; ++j) { piv[2 * j] = bf_lo(w[j]); piv[2 * j + 1] = bf_hi(w[j]); }
#pragma unroll
    for (int j = 0; j < TPL; ++j) { s[j] = 0.f; q[j] = 0.f; }
  }
#pragma unroll 16
  for (int k = 1; k < CW; ++k) {
    uint32_t w[NW];
    load((size_t)k * M, w);
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const float a = bf_lo(w[j]) - piv[2 * j], c = bf_hi(w[j]) - piv[2 * j + 1];
      s[2 * j] += a; q[2 * j] = fmaf(a, a, q[2 * j]);
      s[2 * j + 1] += c; q[2 * j + 1] = fmaf(c, c, q[2 * j + 1]);
    }
  }
#pragma unroll
  for (int j = 0; j < TPL; ++j) {          // partition mean / M2 over CW values (the pivot contributes d = 0)
    pm[warp][lane * TPL + j] = piv[j] + s[j] * (1.f / CW);
    pq[warp][lane * TPL + j] = fmaxf(q[j] - s[j] * s[j] * (1.f / CW), 0.f);
  }
  __syncthreads();
  for (int tk = threadIdx.x; tk < BT; tk += 256) {
    float N = 0.f, Mu = 0.f, Q = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const float nb = (float)CW, mb = pm[w][tk], qb = pq[w][tk];
      const float nt = N + nb, d = mb - Mu;
      Mu += d * (nb / nt);
      Q += qb + d * d * (N * nb / nt);
      N = nt;
    }
    mean[blockIdx.x * BT + tk] = Mu;
    rstd[blockIdx.x * BT + tk] = rsqrt_ftz(Q * (1.f / H) + eps);
  }
}

// ---------------------------------------------------------------------------------------------------------------- fold_prep
// Row n of wq = bf16(W[n, :] * g); vec[0|2][n] = sum_c wq[n, c] (of the bf16 values), vec[1|3][n] = sum_c W[n, c] b[c].
// Block n < D does Wp (width H), block D + n does Wg (width D).
__global__ void __launch_bounds__(256) fold_prep_kernel(const __nv_bfloat16* __restrict__ wp, const float* __restrict__ go,
                                                         const float* __restrict__ bo, const __nv_bfloat16* __restrict__ wg,
                                                         const float* __restrict__ gi, const float* __restrict__ bi,
                                                         __nv_bfloat16* __restrict__ wpq, __nv_bfloat16* __restrict__ wgq,
                                                         float* __restrict__ vec, int D, int H) {
  __shared__ float red[2][8];
  const bool is_p = blockIdx.x < D;
  const int n = is_p ? blockIdx.x : blockIdx.x - D, K = is_p ? H : D;
  const __nv_bfloat16* w = (is_p ? wp : wg) + (size_t)n * K;
  const float *g = is_p ? go : gi, *b = is_p ? bo : bi;
  __nv_bfloat16* wq = (is_p ? wpq : wgq) + (size_t)n * K;
  float s = 0.f, e = 0.f;
  for (int c = threadIdx.x; c < K; c += 256) {
    const float v = __bfloat162float(w[c]);
    const __nv_bfloat16 q = __float2bfloat16_rn(v * g[c]);
    wq[c] = q;
    s += __bfloat162float(q);
    e = fmaf(v, b[c], e);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); e += __shfl_xor_sync(0xffffffffu, e, o); }
  if ((threadIdx.x & 31) == 0) { red[0][threadIdx.x >> 5] = s; red[1][threadIdx.x >> 5] = e; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f, c = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) { a += red[0][k]; c += red[1][k]; }
    vec[(is_p ? 0 : 2) * D + n] = a;
    vec[(is_p ? 1 : 3) * D + n] = c;
  }
}

}  // namespace wide

// mean / rstd [M] over the H channels of channel-major t [H, M]
void wide_ln_stats(torch::Tensor t, torch::Tensor mean, torch::Tensor rstd, double eps) {
  using namespace wide;
  const int H = (int)t.size(0), M = (int)(t.numel() / H);
  TORCH_CHECK(M % 64 == 0 && t.is_contiguous());
  auto st = at::cuda::getCurrentCUDAStream();
  auto T = reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
  const bool wide = M % 256 == 0 && M / 256 >= 2 * 148;
  float *mo = mean.data_ptr<float>(), *ro = rstd.data_ptr<float>();
#define LNS(HH) if (H == HH) { if (wide) ln_stats_kernel<HH, 8><<<M / 256, 256, 0, st>>>(T, mo, ro, M, (float)eps); \
                               else ln_stats_kernel<HH, 2><<<M / 64, 256, 0, st>>>(T, mo, ro, M, (float)eps); } else
  LNS(256) LNS(384) LNS(512) LNS(768) LNS(1024) TORCH_CHECK(false, "wide_ln_stats: unsupported H ", H);
#undef LNS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// wp [D, H], wg [D, D] bf16; go / bo [H], gi / bi [D] fp32 -> wpq [D, H], wgq [D, D] bf16, vec [4, D] fp32 = (sp, ep, sg, eg)
void wide_fold_prep(torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor wg, torch::Tensor gi, torch::Tensor bi,
                    torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec) {
  using namespace wide;
  const int D = (int)wp.size(0), H = (int)wp.size(1);
  TORCH_CHECK(wp.is_contiguous() && wg.is_contiguous() && wpq.is_contiguous() && wgq.is_contiguous() && vec.is_contiguous());
  fold_prep_kernel<<<2 * D, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(wp.data_ptr()), go.data_ptr<float>(), bo.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr()), gi.data_ptr<float>(), bi.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(wpq.data_ptr()), reinterpret_cast<__nv_bfloat16*>(wgq.data_ptr()), vec.data_ptr<float>(), D, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
