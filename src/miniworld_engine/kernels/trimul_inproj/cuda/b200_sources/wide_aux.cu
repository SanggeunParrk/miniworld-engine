// wide_aux.cu -- small pieces around the B200 wide TriMul kernels (k1w front, k3w output):
//   ln_stats   per-token mean / rstd over the H channels of the channel-major contraction output t [H, M] (k3w's LN_out fold)
//   fold_prep  k3w's folded weight operands bf16(Wp g_o), bf16(Wg g_i) and their row sums / bias projections
//   trimul_forms  the D64 / D128 kernels' bf16 weight forms in one launch (cached by the caller while the parameters are unchanged)
//   grad_reduce   the D64 / D128 backward's per-CTA parameter-gradient partials summed in a fixed order, in parameter order
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

// -------------------------------------------------------------------------------------------------------------- lnpart_sum
// Column sums of two row-partial matrices a [R, Wa], b [R, Wb] -> out [Wa + Wb]. Per column, warp g adds rows g, g + 8, ...
// and the 8 warp sums are added in warp order: a fixed order, so bit-identical across runs. One launch of 32-column blocks
// replaces two torch dim-0 sums (a strided reduce of ~8 us each at R = 148).
__global__ void __launch_bounds__(256) lnpart_sum_kernel(const float* __restrict__ a, int wa, const float* __restrict__ b, int wb,
                                                         int R, float* __restrict__ out) {
  __shared__ float part[8][32];
  const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
  const int c = blockIdx.x * 32 + lane;
  const bool in_a = c < wa, live = c < wa + wb;
  const float* src = in_a ? a : b;
  const int w = in_a ? wa : wb, col = in_a ? c : c - wa;
  float s = 0.f;
  if (live)
#pragma unroll 4
    for (int r = g; r < R; r += 8) s += src[(size_t)r * w + col];
  part[g][lane] = s;
  __syncthreads();
  if (g == 0 && live) {
    float t = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) t += part[k][lane];
    out[c] = t;
  }
}


// -------------------------------------------------------------------------------------------------------------- trimul_forms
// Every bf16 weight form the D64 / D128 kernels read, from the parameters as they are stored (fp32 master or bf16, any strides: the D128
// bidirectional module keeps the four front matrices column-major), one element per thread, into one buffer:
//   [wl | wlg | wr | wrg] row-major [R, D] each (k1w), w1 [4R, D] packed (b7m / b7g: chunk c = 64 gate rows, then 64 value rows, of
//   the P = 2R planes [left | right]), wpp [D, R] (k3g: columns in its TMEM K order), wg [D, D], wp [D, R] (b1s / b1g).
// R = P / 2 = the contraction width h. Rounding: round to nearest even, as torch's cast.
struct FormSrc { const void* p[6]; long long s0[6], s1[6]; int f32[6]; };   // wl, wlg, wr, wrg, wg, wp
__device__ __forceinline__ __nv_bfloat16 form_ld(const FormSrc& s, int k, int r, int c) {
  const long long o = (long long)r * s.s0[k] + (long long)c * s.s1[k];
  return s.f32[k] ? __float2bfloat16_rn(reinterpret_cast<const float*>(s.p[k])[o]) : reinterpret_cast<const __nv_bfloat16*>(s.p[k])[o];
}
__global__ void __launch_bounds__(256) forms_kernel(FormSrc s, int R, int D, __nv_bfloat16* __restrict__ out, int total) {
  const int e = blockIdx.x * 256 + threadIdx.x;
  if (e >= total) return;
  const int RD = R * D;
  __nv_bfloat16 v;
  if (e < 4 * RD) {
    v = form_ld(s, e / RD, (e % RD) / D, e % D);
  } else if (e < 8 * RD) {
    const int f = e - 4 * RD, pr = f / D, c = f % D, rr = pr & 127, idx = 64 * (pr >> 7) + (rr & 63);
    const int k = (rr < 64 ? 1 : 0) + (idx < R ? 0 : 2);      // gate rows: wlg / wrg; value rows: wl / wr
    v = form_ld(s, k, idx < R ? idx : idx - R, c);
  } else if (e < 8 * RD + D * R) {
    const int f = e - 8 * RD, n = f / R, col = f % R;
    const int b = col >> 4, q = (col >> 2) & 3, rr = (col >> 1) & 1, u = col & 1;
    v = form_ld(s, 5, n, 16 * b + 8 * rr + 2 * q + u);
  } else if (e < 8 * RD + D * R + D * D) {
    const int f = e - 8 * RD - D * R;
    v = form_ld(s, 4, f / D, f % D);
  } else {
    const int f = e - 8 * RD - D * R - D * D;
    v = form_ld(s, 5, f / R, f % R);
  }
  out[e] = v;
}

// -------------------------------------------------------------------------------------------------------------- grad_reduce
// The D64 / D128 backward's parameter gradients from its per-CTA partials, each output element summed over the partials in index
// order (bit-identical run to run), written in the parameters' order [dwl | dwlg | dwr | dwrg | dwg | dwp | dgi | dbi | dgo | dbo]:
//   pf [nf][D][D + R]  b1s / b1g front CTAs: row o = (dWg[o] | dWp[o])        lo [nl][2R]  b1s / b1g LN CTAs: (dgo | dbo)
//   pw [ns][4R][D]     b7m CTAs / b7g source groups: packed dW1               li [nc][2D]  b7m / b7g LN rows: (dgi | dbi)
__global__ void __launch_bounds__(256) grad_reduce_kernel(const float* __restrict__ pf, int nf, const float* __restrict__ lo, int nl,
                                                           const float* __restrict__ pw, int ns, const float* __restrict__ li, int nc,
                                                           int R, int D, float* __restrict__ out, int total) {
  // block = 128 consecutive outputs, 4 per lane (float4; every region size is a multiple of 128); warp w sums partials w, w + 8, ...,
  // the 8 warp sums are added in warp order: a fixed order, and at most ceil(n / 8) dependent loads per thread
  __shared__ float4 part[8][32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int e = blockIdx.x * 128 + 4 * lane;
  const int RD = R * D;
  const float* p;
  long long st;
  int n;
  if (e < 4 * RD) {
    const int k = e / RD, r = (e % RD) / D, c = e % D, idx = (k >> 1) * R + r;   // k: wl, wlg, wr, wrg
    p = pw + (long long)(128 * (idx >> 6) + ((k & 1) ? 0 : 64) + (idx & 63)) * D + c;
    st = 4LL * RD; n = ns;
  } else if (e < 4 * RD + D * D) {
    const int f = e - 4 * RD;
    p = pf + (long long)(f / D) * (D + R) + f % D; st = (long long)D * (D + R); n = nf;
  } else if (e < 4 * RD + D * D + D * R) {
    const int f = e - 4 * RD - D * D;
    p = pf + (long long)(f / R) * (D + R) + D + f % R; st = (long long)D * (D + R); n = nf;
  } else if (e < 4 * RD + D * D + D * R + 2 * D) {
    p = li + (e - 4 * RD - D * D - D * R); st = 2 * D; n = nc;
  } else {
    p = lo + (e - 4 * RD - D * D - D * R - 2 * D); st = 2 * R; n = nl;
  }
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  if (e < total) {
#pragma unroll 4
    for (int i = w; i < n; i += 8) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(p + i * st));
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  }
  part[w][lane] = acc;
  __syncthreads();
  if (w == 0 && e < total) {
    float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < 8; ++k) { const float4 v = part[k][lane]; t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w; }
    *reinterpret_cast<float4*>(out + e) = t;
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

// a [R, Wa], b [R, Wb] fp32 row partials -> out [Wa + Wb] fp32 column sums (fixed order)
void lnpart_sum(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  using namespace wide;
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == b.size(0) && a.is_contiguous() && b.is_contiguous() &&
              out.is_contiguous() && a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32 &&
              out.scalar_type() == torch::kFloat32 && out.numel() == a.size(1) + b.size(1));
  const int wa = (int)a.size(1), wb = (int)b.size(1);
  lnpart_sum_kernel<<<(wa + wb + 31) / 32, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(), wa, b.data_ptr<float>(), wb, (int)a.size(0), out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// wl, wlg, wr, wrg [R, D], wg [D, D], wp [D, R]: fp32 or bf16, any strides -> out bf16 [8 R D + 2 D R + D D] (see forms_kernel)
void trimul_forms(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wp,
                  torch::Tensor out) {
  using namespace wide;
  const int R = (int)wl.size(0), D = (int)wl.size(1);
  const torch::Tensor* ts[6] = {&wl, &wlg, &wr, &wrg, &wg, &wp};
  FormSrc s;
  for (int k = 0; k < 6; ++k) {
    const torch::Tensor& t = *ts[k];
    TORCH_CHECK(t.dim() == 2 && t.is_cuda() && (t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16));
    TORCH_CHECK(k < 4 ? (t.size(0) == R && t.size(1) == D) : (t.size(0) == D && t.size(1) == (k == 4 ? D : R)), "trimul_forms: shapes");
    s.p[k] = t.data_ptr(); s.s0[k] = t.stride(0); s.s1[k] = t.stride(1); s.f32[k] = t.scalar_type() == torch::kFloat32;
  }
  const int total = 8 * R * D + 2 * D * R + D * D;
  TORCH_CHECK(R % 64 == 0 && out.is_contiguous() && out.scalar_type() == torch::kBFloat16 && out.numel() == total);
  forms_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(s, R, D,
                                                                                   reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// pf [>= nf, D (D + R)], lo [>= nl, 2R], pw [>= ns, 4 R D], li [>= nc, 2D] fp32 partials -> out [4 R D + D D + D R + 2D + 2R] fp32
void grad_reduce(torch::Tensor pf, int64_t nf, torch::Tensor lo, int64_t nl, torch::Tensor pw, int64_t ns, torch::Tensor li, int64_t nc,
                 torch::Tensor out) {
  using namespace wide;
  const int D = (int)(li.size(-1) / 2), R = (int)(lo.size(-1) / 2);
  for (const torch::Tensor* t : {&pf, &lo, &pw, &li, &out}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32);
  TORCH_CHECK(pf.numel() >= nf * D * (D + R) && lo.numel() >= nl * 2 * R && pw.numel() >= ns * 4LL * R * D && li.numel() >= nc * 2 * D);
  const int total = 4 * R * D + D * D + D * R + 2 * D + 2 * R;
  TORCH_CHECK(out.numel() == total, "grad_reduce: out size");
  TORCH_CHECK(R % 64 == 0 && D % 64 == 0 && total % 128 == 0);
  grad_reduce_kernel<<<total / 128, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      pf.data_ptr<float>(), (int)nf, lo.data_ptr<float>(), (int)nl, pw.data_ptr<float>(), (int)ns, li.data_ptr<float>(), (int)nc, R, D,
      out.data_ptr<float>(), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
