// wide_aux.cu -- small pieces around the B200 wide TriMul kernels (k1w front, k3w output):
//   ln_stats   per-token mean / rstd over the H channels of the channel-major contraction output t [H, M] (k3w's LN_out fold)
//   fold_prep  k3w's folded weight operands bf16(Wp g_o), bf16(Wg g_i) and their row sums / bias projections
//   lnpart_sum the D64 / D128 backward's per-CTA LayerNorm-gradient rows summed in a fixed order
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

}  // namespace wide

#ifndef WLS_PDL
#define WLS_PDL 1   // B200: ln_stats as a programmatic dependent of the contraction, triggering k3w at entry (2026-11-17)
#endif
bool wide_ln_stats_pdl(const __nv_bfloat16* t, float* mo, float* ro, int M, int H, float eps, bool wide);   // below (B200 part)
// mean / rstd [M] over the H channels of channel-major t [H, M]
void wide_ln_stats(torch::Tensor t, torch::Tensor mean, torch::Tensor rstd, double eps) {
  using namespace wide;
  const int H = (int)t.size(0), M = (int)(t.numel() / H);
  TORCH_CHECK(M % 64 == 0 && t.is_contiguous());
  auto st = at::cuda::getCurrentCUDAStream();
  auto T = reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
  const bool wide = M % 256 == 0 && M / 256 >= 2 * 148;
  float *mo = mean.data_ptr<float>(), *ro = rstd.data_ptr<float>();
#if WLS_PDL
  if (wide_ln_stats_pdl(T, mo, ro, M, H, (float)eps, wide)) { C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
#endif
#define LNS(HH) if (H == HH) { if (wide) ln_stats_kernel<HH, 8><<<M / 256, 256, 0, st>>>(T, mo, ro, M, (float)eps); \
                               else ln_stats_kernel<HH, 2><<<M / 64, 256, 0, st>>>(T, mo, ro, M, (float)eps); } else
  LNS(256) LNS(384) LNS(512) LNS(768) LNS(1024) TORCH_CHECK(false, "wide_ln_stats: unsupported H ", H);
#undef LNS
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

// ============================================================================================ B200-only additions
// Reopened namespace AFTER the first "}  // namespace wide": h100_b200style_aux.py compiles this file's text up to that first close for
// the H100, so everything below is in the B200 extension only (the H100 source text stays byte-identical).
//   trimul_forms  the D64 / D128 kernels' bf16 weight forms in one launch (cached by the caller while the parameters are unchanged)
//   grad_reduce   the D64 / D128 backward's per-CTA parameter-gradient partials summed in a fixed order, in parameter order
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <mutex>
namespace wide {
#ifndef WA_PDL
#define WA_PDL 1   // forms / drop_scale / grad_reduce: programmatic dependents of the previous kernel (launch overlapped; they wait first)
#endif
// griddepcontrol.wait (the previous grid complete, its writes visible) as the first statement, then let the next grid launch
__device__ __forceinline__ void wa_pdl() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
template <int BD = 256, typename... KArgs, typename... Args>
inline void wa_launch(void (*kernel)(KArgs...), int grid, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(BD); cfg.dynamicSmemBytes = 0; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = WA_PDL ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}
// ------------------------------------------------------------------------------------------------------------ trimul_forms
// Every bf16 weight form the D64 / D128 kernels read, from the parameters as they are stored (fp32 master or bf16, any strides: the D128
// bidirectional module keeps the four front matrices column-major), one element per thread, into one buffer:
//   [wl | wlg | wr | wrg] row-major [R, D] each (k1w), w1 [4R, D] packed (b7m / b7g: chunk c = 64 gate rows, then 64 value rows, of
//   the P = 2R planes [left | right]), wpp [D, R] (k3g: columns in its TMEM K order), wg [D, D], wp [D, R] (b1s / b1g),
//   wpf [D, R] = bf16(Wp g_o) (b1s / b1g's LN_out-folded projection, 2026-11-17: ONE rounding of the stored parameter times g_o; the
//   kernels formed it as bf16(bf16(Wp) g_o) before, a second rounding of every weight that the backward's sums over L^2 tokens carry
//   coherently -- the D64 / D128 YELLOW weight gradients).
// R = P / 2 = the contraction width h. Rounding: round to nearest even, as torch's cast.
// RoseTTAFold3 (rf3 != 0): W_l is the reference's W_l * f, f = bf16(1 / max(n, 1)) the sample's length factor (n its kept tokens: the
// token mask tm [tl], or tl without one), formed here: v = bf16(float(W_l) * f) as the reference's product then cast; f is also written
// to fout (grad_reduce scales dW_l by it). (2026-11-17: the factor's ~7 launches -- mask sum, casts, reciprocal, the weight product --
// and the uncached forms per call become this one launch.)
// Row dropout (n8 > 0, 2026-11-17 part 3): blocks past the forms' also turn the reference's uniform draw dr (torch.rand, bf16) into the
// row-dropout scale dout = bf16(float(r > pb) * inv) -- drop_scale_kernel's arithmetic, 8 elements per thread -- so a training call
// runs one launch for its forms and its dropout scale (the forms' cache misses once per step) instead of two.
struct FormSrc { const void* p[6]; long long s0[6], s1[6]; int f32[6]; const float* go; const bool* tm; int tl; int rf3; float* fout;
                 const uint4* dr; uint4* dout; float pb, dinv; int n8;
                 at::PhiloxCudaState rng; long long pstride; int philox; };
// The reference's uniform draw element e of torch.rand(shape, dtype=bf16) (ATen distribution_nullary_kernel + uniform_kernel, bitwise):
// grid stride S = G * 256 (G = min(ceil(n / 256), SMs * maxThreadsPerSM / 256)), thread t = e % S draws Philox subsequence t, one
// curand_uniform4 per pass of 4 S elements; component (e / S) % 4 of pass e / (4 S); bf16(u) with 1 mapped to 0 (the (0, 1] -> [0, 1)
// reversal). The counter increment the caller reserved is torch's ((n - 1) / (4 S) + 1) * 4.
__device__ __forceinline__ float philox_rand_bf16(unsigned long long seed, unsigned long long off, long long e, long long S) {
  const long long t = e % S, q = e / S;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, (unsigned long long)t, off + 4ull * (unsigned long long)(q >> 2), &st);
  const float4 u = curand_uniform4(&st);
  const int j = (int)(q & 3);
  const float v = j == 0 ? u.x : j == 1 ? u.y : j == 2 ? u.z : u.w;
  const float b = __bfloat162float(__float2bfloat16_rn(v));
  return b == 1.f ? 0.f : b;
}
__device__ __forceinline__ float form_f(const FormSrc& s, int k, int r, int c) {
  const long long o = (long long)r * s.s0[k] + (long long)c * s.s1[k];
  return s.f32[k] ? reinterpret_cast<const float*>(s.p[k])[o] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(s.p[k])[o]);
}
__device__ __forceinline__ __nv_bfloat16 form_ld(const FormSrc& s, int k, int r, int c, float f) {
  if (k == 0 && s.rf3) return __float2bfloat16_rn(form_f(s, 0, r, c) * f);
  const long long o = (long long)r * s.s0[k] + (long long)c * s.s1[k];
  return s.f32[k] ? __float2bfloat16_rn(reinterpret_cast<const float*>(s.p[k])[o]) : reinterpret_cast<const __nv_bfloat16*>(s.p[k])[o];
}
__global__ void __launch_bounds__(256) forms_kernel(FormSrc s, int R, int D, __nv_bfloat16* __restrict__ out, int total, int skip) {
  wa_pdl();
  {
    const int fb = (total + 255) / 256;
    if ((int)blockIdx.x >= fb) {   // the row-dropout scale (drop_scale_kernel's body)
      const int i = ((int)blockIdx.x - fb) * 256 + threadIdx.x;
      if (i >= s.n8) return;
      const __nv_bfloat16 one = __float2bfloat16_rn(s.dinv), zero = __float2bfloat16_rn(0.f);
      if (s.philox) {   // the draw itself (philox_rand_bf16: torch.rand's bits), no r tensor (2026-11-17 part 6: one launch fewer);
        // one element per thread (n8 = n here): eight Philox blocks per thread made the four draw CTAs the forms launch's tail
        const auto seeds = at::cuda::philox::unpack(s.rng);
        reinterpret_cast<__nv_bfloat16*>(s.dout)[i] = philox_rand_bf16(std::get<0>(seeds), std::get<1>(seeds), i, s.pstride) > s.pb ? one : zero;
        return;
      }
      const uint4 v = __ldg(s.dr + i);
      uint4 o;
      uint32_t* ov = reinterpret_cast<uint32_t*>(&o);
      const uint32_t* iv = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&iv[k]);
        __nv_bfloat162 q;
        q.x = __bfloat162float(h.x) > s.pb ? one : zero;
        q.y = __bfloat162float(h.y) > s.pb ? one : zero;
        ov[k] = *reinterpret_cast<uint32_t*>(&q);
      }
      s.dout[i] = o;
      return;
    }
  }
  const int eo = blockIdx.x * 256 + threadIdx.x;
  // skip = D R: the wide layout (wide_forms) -- no wpp section, so the packed W1 and wg form one contiguous [4R + D, D] operand
  const int e = (skip && eo >= 8 * R * D) ? eo + skip : eo;
  float f = 1.f;
  if (s.rf3) {
    int n = s.tl;
    if (s.tm) {
      n = 0;
      for (int i0 = 0; i0 < s.tl; i0 += 256) n += __syncthreads_count(i0 + (int)threadIdx.x < s.tl && s.tm[i0 + threadIdx.x]);
      f = __bfloat162float(__float2bfloat16_rn(1.0f / (float)max(n, 1)));    // count.to(fp32).reciprocal().to(bf16)
    } else {
      f = __bfloat162float(__float2bfloat16_rn((float)(1.0 / (double)max(n, 1))));   // new_full(1.0 / L) in bf16
    }
    if (e == 0 && s.fout) *s.fout = f;
  }
  if (eo >= total) return;
  const int RD = R * D;
  __nv_bfloat16 v;
  if (e < 4 * RD) {
    v = form_ld(s, e / RD, (e % RD) / D, e % D, f);
  } else if (e < 8 * RD) {
    const int f8 = e - 4 * RD, pr = f8 / D, c = f8 % D, rr = pr & 127, idx = 64 * (pr >> 7) + (rr & 63);
    const int k = (rr < 64 ? 1 : 0) + (idx < R ? 0 : 2);      // gate rows: wlg / wrg; value rows: wl / wr
    v = form_ld(s, k, idx < R ? idx : idx - R, c, f);
  } else if (e < 8 * RD + D * R) {
    const int f8 = e - 8 * RD, n = f8 / R, col = f8 % R;
    const int b = col >> 4, q = (col >> 2) & 3, rr = (col >> 1) & 1, u = col & 1;
    v = form_ld(s, 5, n, 16 * b + 8 * rr + 2 * q + u, f);
  } else if (e < 8 * RD + D * R + D * D) {
    const int f8 = e - 8 * RD - D * R;
    v = form_ld(s, 4, f8 / D, f8 % D, f);
  } else if (e < 8 * RD + 2 * D * R + D * D) {
    const int f8 = e - 8 * RD - D * R - D * D;
    v = form_ld(s, 5, f8 / R, f8 % R, f);
  } else {
    const int f8 = e - 8 * RD - 2 * D * R - D * D, c = f8 % R;
    v = __float2bfloat16_rn(form_f(s, 5, f8 / R, c) * s.go[c]);
  }
  out[eo] = v;
}

// (2026-11-18, training part 8) forms_kernel's work with the draw made in it (philox), no RoseTTAFold3 factor, the D64 / D128 layout
// (skip 0), on a FEW 1024-thread blocks with a grid stride: the training front k1wt (launched just before it, running
// concurrently) needs whole SMs (512 threads, the register file), so this launch stays on the SMs k1wt leaves free and both run at
// once. It is launched right after k1wt (which waited for the work before it, then triggered): it does NOT wait at entry (it reads the
// parameters and the RNG scalars only, both older than k1wt's wait) and waits at its end, so its completion implies k1wt's (the
// contraction after it is an ordinary launch). Per element the same arithmetic as forms_kernel.
// mode (2026-11-18, training part 12): 0 the above (k1wt first). 1 wait-first: the forms launch BEFORE the one-launch training forward
// tft (from the fp32 masters), which reads only ds of it, at its epilogue: wait at entry, then trigger. 2 chain: the forms launch AFTER
// tft (tft = the first kernel, a programmatic dependent waiting at entry, triggering at entry so this launch runs beside it on the SMs
// it leaves free). cnt[0] / cnt[1] are monotonic counters shared with tft: this launch draws ds first and adds 1 a block to cnt[0]
// (tft's epilogue waits for all of them before it reads ds), makes the forms, then waits until cnt[1] (one add per tft CTA at its
// epilogue, after its ds wait) has grown by 128 since this block's entry -- every tft CTA at its epilogue, where tft itself triggered
// its dependents before -- and only then triggers its own dependent (the backward's first kernel) and waits for tft's completion. The
// entry read of cnt[1] is exact: no tft CTA adds to it before every block of this launch has added to cnt[0], after that read.
__device__ __forceinline__ unsigned ld_acq_u32(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__global__ void __launch_bounds__(1024) forms_gs_kernel(FormSrc s, int R, int D, __nv_bfloat16* __restrict__ out, int total, int mode,
                                                        unsigned* __restrict__ cnt) {
  if (mode == 1) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (mode != 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int RD = R * D;
  const long long n = (long long)total + s.n8;
  const __nv_bfloat16 one = __float2bfloat16_rn(s.dinv), zero = __float2bfloat16_rn(0.f);
  const auto seeds = at::cuda::philox::unpack(s.rng);
  __shared__ unsigned epb;
  if (mode == 2) {
    if (threadIdx.x == 0) epb = ld_acq_u32(cnt + 1);
    for (long long i = (long long)blockIdx.x * 1024 + threadIdx.x; i < s.n8; i += (long long)gridDim.x * 1024)
      reinterpret_cast<__nv_bfloat16*>(s.dout)[i] = philox_rand_bf16(std::get<0>(seeds), std::get<1>(seeds), i, s.pstride) > s.pb ? one : zero;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
  }
  for (long long idx = (long long)blockIdx.x * 1024 + threadIdx.x; idx < (mode == 2 ? (long long)total : n); idx += (long long)gridDim.x * 1024) {
    if (idx >= total) {
      const long long i = idx - total;
      reinterpret_cast<__nv_bfloat16*>(s.dout)[i] = philox_rand_bf16(std::get<0>(seeds), std::get<1>(seeds), i, s.pstride) > s.pb ? one : zero;
      continue;
    }
    const int e = (int)idx;
    __nv_bfloat16 v;
    if (e < 4 * RD) {
      v = form_ld(s, e / RD, (e % RD) / D, e % D, 1.f);
    } else if (e < 8 * RD) {
      const int f8 = e - 4 * RD, pr = f8 / D, c = f8 % D, rr = pr & 127, ix = 64 * (pr >> 7) + (rr & 63);
      const int k = (rr < 64 ? 1 : 0) + (ix < R ? 0 : 2);
      v = form_ld(s, k, ix < R ? ix : ix - R, c, 1.f);
    } else if (e < 8 * RD + D * R) {
      const int f8 = e - 8 * RD, nn = f8 / R, col = f8 % R;
      const int b = col >> 4, q = (col >> 2) & 3, rr = (col >> 1) & 1, u = col & 1;
      v = form_ld(s, 5, nn, 16 * b + 8 * rr + 2 * q + u, 1.f);
    } else if (e < 8 * RD + D * R + D * D) {
      const int f8 = e - 8 * RD - D * R;
      v = form_ld(s, 4, f8 / D, f8 % D, 1.f);
    } else if (e < 8 * RD + 2 * D * R + D * D) {
      const int f8 = e - 8 * RD - D * R - D * D;
      v = form_ld(s, 5, f8 / R, f8 % R, 1.f);
    } else {
      const int f8 = e - 8 * RD - 2 * D * R - D * D, c = f8 % R;
      v = __float2bfloat16_rn(form_f(s, 5, f8 / R, c) * s.go[c]);
    }
    out[e] = v;
  }
  if (mode == 2) {
    if (threadIdx.x == 0) {
      const unsigned tgt = epb + 128u;
      while ((int)(ld_acq_u32(cnt + 1) - tgt) < 0) {}
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      asm volatile("griddepcontrol.wait;" ::: "memory");   // complete after tft
    }
    return;
  }
  if (mode == 0) asm volatile("griddepcontrol.wait;" ::: "memory");   // complete after k1wt (the previous grid)
}

// -------------------------------------------------------------------------------------------------------------- fold_prep_t
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
// fold_prep_kernel (above, shared with the H100) for parameters stored fp32 (the master) or bf16: wq = bf16(W g) from the stored value
// (one rounding; the bf16 kernel rounded an already-rounded bf16(W) a second time), the row sums of the bf16 wq, the bias projections
// from the stored W. Block n < D: Wp (width H), block D + n: Wg (width D).
template <typename T>
__global__ void __launch_bounds__(256) fold_prep_t_kernel(const T* __restrict__ wp, const float* __restrict__ go, const float* __restrict__ bo,
                                                          const T* __restrict__ wg, const float* __restrict__ gi, const float* __restrict__ bi,
                                                          __nv_bfloat16* __restrict__ wpq, __nv_bfloat16* __restrict__ wgq,
                                                          float* __restrict__ vec, int D, int H) {
  __shared__ float red[2][8];
  const bool is_p = blockIdx.x < D;
  const int n = is_p ? blockIdx.x : blockIdx.x - D, K = is_p ? H : D;
  const T* w = (is_p ? wp : wg) + (size_t)n * K;
  const float *g = is_p ? go : gi, *b = is_p ? bo : bi;
  __nv_bfloat16* wq = (is_p ? wpq : wgq) + (size_t)n * K;
  float s = 0.f, e = 0.f;
  for (int c = threadIdx.x; c < K; c += 256) {
    const float v = to_f(w[c]);
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

// -------------------------------------------------------------------------------------------------------------- grad_reduce
// The D64 / D128 backward's parameter gradients from its per-CTA partials, each output element summed over the partials in index
// order (bit-identical run to run), written in the parameters' order [dwl | dwlg | dwr | dwrg | dwg | dwp | dgi | dbi | dgo | dbo]:
//   pf [nf][D][D + R]  b1s / b1g front CTAs: row o = (dWg[o] | dWp[o])        lo [nl][2R]  b1s / b1g LN CTAs: (dgo | dbo)
//   pw [ns][4R][D]     b7m CTAs / b7g source groups: packed dW1               li [nc][2D]  b7m / b7g LN rows: (dgi | dbi)
#ifndef GR_U
#define GR_U 4    // grad_reduce loads in flight per thread (loop unroll)
#endif
constexpr int kGrU = GR_U;   // (#pragma unroll takes a constant expression, not a macro)
#ifndef GR_W
#define GR_W 16   // (16, 2026-11-17: D64 training L128 69.8 -> 67.7 us, D128 bidirectional L128 149.7 -> 145.5 / 150.2 us; 32 slower) grad_reduce warps per block (each sums every GR_W-th partial of the block's 128 outputs)
#endif
__global__ void __launch_bounds__(32 * GR_W) grad_reduce_kernel(const float* __restrict__ pf, int nf, const float* __restrict__ lo, int nl,
                                                           const float* __restrict__ pw, int ns, const float* __restrict__ li, int nc,
                                                           int R, int D, float* __restrict__ out, int total, const float* __restrict__ wls, int skip_o) {
  wa_pdl();
  // block = 128 consecutive outputs, 4 per lane (float4; every region size is a multiple of 128); warp w sums partials w, w + 8, ...,
  // the 8 warp sums are added in warp order: a fixed order, and at most ceil(n / 8) dependent loads per thread
  __shared__ float4 part[GR_W][32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int RD = R * D;
  // skip_o: blocks past the pw region are the li blocks (the output-side regions between them: b7m's extra CTAs)
  const int blk = (skip_o && (int)blockIdx.x >= 4 * RD / 128) ? (int)blockIdx.x + (D * D + D * R) / 128 : (int)blockIdx.x;
  const int e = blk * 128 + 4 * lane;
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
#pragma unroll kGrU
    for (int i = w; i < n; i += GR_W) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(p + i * st));
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  }
  part[w][lane] = acc;
  __syncthreads();
  if (w == 0 && e < total) {
    float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < GR_W; ++k) { const float4 v = part[k][lane]; t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w; }
    if (wls && e < R * D) { const float f = *wls; t.x *= f; t.y *= f; t.z *= f; t.w *= f; }   // RF3: dW_l = f dW_l'
    *reinterpret_cast<float4*>(out + e) = t;
  }
}

// -------------------------------------------------------------------------------------------------------------- drop_scale
// The row-dropout scale from the reference's uniform draw r (torch.rand, bf16) in one pass: ds = bf16(float(r > pb) * inv), the
// bits of the reference's `(r > p).to(bf16) / (1 - p)` (torch compares against bf16(p) and divides by a scalar as a product with the
// fp32 reciprocal 1 / (1 - p)); 8 elements per thread.
__global__ void __launch_bounds__(256) drop_scale_kernel(const uint4* __restrict__ r, uint4* __restrict__ out, float pb, float inv, int n8) {
  wa_pdl();
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n8) return;
  const uint4 v = __ldg(r + i);
  const __nv_bfloat16 one = __float2bfloat16_rn(inv), zero = __float2bfloat16_rn(0.f);
  uint4 o;
  uint32_t* ov = reinterpret_cast<uint32_t*>(&o);
  const uint32_t* iv = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&iv[k]);
    __nv_bfloat162 q;
    q.x = __bfloat162float(h.x) > pb ? one : zero;
    q.y = __bfloat162float(h.y) > pb ? one : zero;
    ov[k] = *reinterpret_cast<uint32_t*>(&q);
  }
  out[i] = o;
}


// ------------------------------------------------------------------------------------------------------------ ln_stats (PDL)
// ln_stats_kernel (above, shared with the H100) as a programmatic dependent (2026-11-17): waits for the contraction, then lets k3w
// launch at once, so k3w's set-up and its t / x / weight loads run during this pass (k3w waits for it before reading mu / rstd).
// The same arithmetic in the same order as ln_stats_kernel (bitwise the same statistics).
template <int H, int TPL>
__global__ void __launch_bounds__(256) ln_stats_pdl_kernel(const __nv_bfloat16* __restrict__ t, float* __restrict__ mean,
                                                            float* __restrict__ rstd, int M, float eps) {
  wa_pdl();
  constexpr int CW = H / 8, BT = 32 * TPL, NW = TPL / 2;
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
  for (int j = 0; j < TPL; ++j) {
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

// ------------------------------------------------------------------------------------------------- graph RNG prologue writer
// (2026-11-17 part 6; shared since part 7) A CUDA-graph replay that holds a Philox draw first refreshes the generator's two graph-owned
// int64 scalars with two aten::fill_ launches (CUDAGeneratorState::replay_prologue: seed, then offset). The scalars of a graph holding
// this file's draw are registered at capture with the ONE B200 writer (kernels/_graph_rng_sm100.cu, shared with TriangleAttention's
// drop_scale): it serves both fills with one programmatic launch. Its function table comes in once per load (trimul_graph_rng_api).
struct RngApi {
  int64_t abi;
  void (*register_state)(void*, void*);
};
inline const RngApi*& rng_api() {
  static const RngApi* a = nullptr;
  return a;
}
// Reserve the default generator's Philox counter for a torch.rand of n elements on device dev (the same increment, under the
// generator's lock, so the draw has torch.rand's bits and the stream of draws after it is unchanged) -> state, grid stride. Inside a
// capture the graph-owned scalars are registered for the prologue writer.
inline void philox_reserve(int64_t n, c10::Device dev, at::PhiloxCudaState* state, long long* stride) {
  const auto* prop = at::cuda::getDeviceProperties(dev.index());
  const int64_t g = std::min<int64_t>((n + 255) / 256, (int64_t)prop->multiProcessorCount * (prop->maxThreadsPerMultiProcessor / 256));
  *stride = g * 256;
  const uint64_t inc = (uint64_t)((n - 1) / (g * 256 * 4) + 1) * 4;
  auto* gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator(dev.index()));
  {
    std::lock_guard<std::mutex> lk(gen->mutex_);
    *state = gen->philox_cuda_state(inc);
  }
  if (!state->captured_) return;
  const RngApi* api = rng_api();
  TORCH_CHECK(api && api->abi == 1, "trimul_forms: captured Philox draw without the graph RNG writer (trimul_graph_rng_api)");
  api->register_state(static_cast<void*>(state->seed_.ptr), static_cast<void*>(state->offset_.ptr));
}
}  // namespace wide (B200-only additions)

// the shared graph RNG writer's function table (kernels/_graph_rng_sm100.cu api(): its address), handed over once per load
void trimul_graph_rng_api(int64_t address) {
  const auto* a = reinterpret_cast<const wide::RngApi*>(address);
  TORCH_CHECK(a && a->abi == 1, "trimul_graph_rng_api: unknown writer table");
  wide::rng_api() = a;
}

// r bf16 (any shape, contiguous, numel % 8 == 0) -> out bf16 of r's shape: the row-dropout scale (drop_scale_kernel)
void drop_scale(torch::Tensor r, torch::Tensor out, double p) {
  using namespace wide;
  TORCH_CHECK(r.is_contiguous() && out.is_contiguous() && r.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16 &&
              r.numel() == out.numel() && r.numel() % 8 == 0, "drop_scale: bf16 contiguous, numel % 8 == 0");
  const float pb = __bfloat162float(__float2bfloat16_rn((float)p));
  const float inv = 1.0f / (float)(1.0 - p);
  const int n8 = (int)(r.numel() / 8);
  if (n8 == 0) return;
  wa_launch(drop_scale_kernel, (n8 + 255) / 256, reinterpret_cast<const uint4*>(r.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()), pb, inv, n8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// wl, wlg, wr, wrg [R, D], wg [D, D], wp [D, R]: fp32 or bf16, any strides; go [R] fp32 -> out bf16 [8 R D + 3 D R + D D]
// (see forms_kernel)
void trimul_forms(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wp,
                  torch::Tensor go, torch::Tensor out, c10::optional<torch::Tensor> rf3_mask, int64_t rf3_len,
                  c10::optional<torch::Tensor> rf3_out, c10::optional<torch::Tensor> drop_r, c10::optional<torch::Tensor> drop_out,
                  double drop_p, bool drop_philox, int64_t gs_blocks, int64_t gs_mode, c10::optional<torch::Tensor> gs_cnt) {
  using namespace wide;
  const int R = (int)wl.size(0), D = (int)wl.size(1);
  const torch::Tensor* ts[6] = {&wl, &wlg, &wr, &wrg, &wg, &wp};
  FormSrc s;
  TORCH_CHECK(go.is_cuda() && go.is_contiguous() && go.scalar_type() == torch::kFloat32 && go.numel() == R, "trimul_forms: go fp32 [R]");
  s.go = go.data_ptr<float>();
  // RoseTTAFold3 (rf3_len > 0): W_l scaled by the length factor of the [rf3_len] token mask (or of rf3_len without one); f -> rf3_out
  s.rf3 = rf3_len > 0; s.tl = (int)rf3_len; s.tm = nullptr; s.fout = nullptr;
  if (rf3_mask.has_value()) {
    TORCH_CHECK(rf3_mask->is_cuda() && rf3_mask->is_contiguous() && rf3_mask->scalar_type() == torch::kBool && rf3_mask->numel() == rf3_len,
                "trimul_forms: rf3_mask bool [L]");
    s.tm = rf3_mask->data_ptr<bool>();
  }
  if (rf3_out.has_value()) {
    TORCH_CHECK(rf3_out->is_cuda() && rf3_out->scalar_type() == torch::kFloat32 && rf3_out->numel() == 1, "trimul_forms: rf3_out fp32 [1]");
    s.fout = rf3_out->data_ptr<float>();
  }
  for (int k = 0; k < 6; ++k) {
    const torch::Tensor& t = *ts[k];
    TORCH_CHECK(t.dim() == 2 && t.is_cuda() && (t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16));
    TORCH_CHECK(k < 4 ? (t.size(0) == R && t.size(1) == D) : (t.size(0) == D && t.size(1) == (k == 4 ? D : R)), "trimul_forms: shapes");
    s.p[k] = t.data_ptr(); s.s0[k] = t.stride(0); s.s1[k] = t.stride(1); s.f32[k] = t.scalar_type() == torch::kFloat32;
  }
  const int total = 8 * R * D + 3 * D * R + D * D;
  TORCH_CHECK(R % 64 == 0 && out.is_contiguous() && out.scalar_type() == torch::kBFloat16 && out.numel() == total);
  // drop_r / drop_out (bf16, contiguous, numel % 8 == 0): the row-dropout scale of the draw drop_r in the same launch (drop_scale's operands)
  s.dr = nullptr; s.dout = nullptr; s.pb = 0.f; s.dinv = 0.f; s.n8 = 0;
  if (drop_r.has_value()) {
    TORCH_CHECK(drop_out.has_value() && drop_r->is_contiguous() && drop_out->is_contiguous() && drop_r->scalar_type() == torch::kBFloat16 &&
                drop_out->scalar_type() == torch::kBFloat16 && drop_r->numel() == drop_out->numel() && drop_r->numel() % 8 == 0,
                "trimul_forms: drop_r / drop_out bf16 contiguous, numel % 8 == 0");
    s.dr = reinterpret_cast<const uint4*>(drop_r->data_ptr()); s.dout = reinterpret_cast<uint4*>(drop_out->data_ptr());
    s.pb = __bfloat162float(__float2bfloat16_rn((float)drop_p)); s.dinv = 1.0f / (float)(1.0 - drop_p); s.n8 = (int)(drop_r->numel() / 8);
  }
  s.philox = 0; s.pstride = 0;
  if (drop_philox) {
    // the draw inside this launch (forms_kernel's philox branch): the default generator's counter reserved here as torch.rand(n) reserves it
    TORCH_CHECK(!drop_r.has_value() && drop_out.has_value() && drop_out->is_cuda() && drop_out->is_contiguous() &&
                drop_out->scalar_type() == torch::kBFloat16 && drop_out->numel() % 8 == 0 && drop_out->numel() > 0,
                "trimul_forms: drop_philox needs drop_out bf16 contiguous, numel % 8 == 0, and no drop_r");
    s.dout = reinterpret_cast<uint4*>(drop_out->data_ptr());
    s.pb = __bfloat162float(__float2bfloat16_rn((float)drop_p)); s.dinv = 1.0f / (float)(1.0 - drop_p); s.n8 = (int)drop_out->numel();   // philox: one element per thread
    s.philox = 1;
    philox_reserve(drop_out->numel(), drop_out->device(), &s.rng, &s.pstride);
  }
  if (gs_blocks > 0) {   // (part 8) on gs_blocks 1024-thread blocks, beside k1wt (forms_gs_kernel)
    TORCH_CHECK(drop_philox && !s.rf3, "trimul_forms: gs_blocks needs the philox draw and no RoseTTAFold3 factor");
    // gs_mode (part 12): 0 beside k1wt, 1 wait-first before tft, 2 after tft (chain; gs_cnt: tft's int32 [>= 2] counters)
    TORCH_CHECK(gs_mode >= 0 && gs_mode <= 2 && (gs_mode != 2 || (gs_cnt.has_value() && gs_cnt->is_cuda() && gs_cnt->numel() >= 2 &&
                                                                    gs_cnt->scalar_type() == torch::kInt32)),
                "trimul_forms: gs_mode 0 / 1 / 2 (2 with the int32 counters)");
    unsigned* cp = gs_mode == 2 ? reinterpret_cast<unsigned*>(gs_cnt->data_ptr()) : nullptr;
    wa_launch<1024>(forms_gs_kernel, (int)gs_blocks, s, R, D, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total, (int)gs_mode, cp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  wa_launch(forms_kernel, (total + 255) / 256 + (s.n8 + 255) / 256, s, R, D, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// (part 12) load forms_gs_kernel's module now (CUDA lazy loading): the chain launches it while tft runs and spins on it, and a
// first launch that loads the module waits for the running kernel -- measured: the first chain step's two spins never met
void trimul_forms_gs_preload() {
  static bool done = false;
  if (done) return;
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, wide::forms_gs_kernel));
  done = true;
}

// The D256-512 training's bf16 weight casts in one launch (2026-11-17; was six torch casts plus the packed-W1 / Wg concatenation of the
// input-gradient GEMM, four cat / two copy launches per backward): out bf16 [8 R D + D D + D R] = [wl | wlg | wr | wrg] [R, D] row-major,
// W1 packed [4R, D] (as trimul_forms) directly followed by wg [D, D] (the [4R + D, D] operand of dxn = dcat [W1; Wg]), wp [D, R].
// Values: round to nearest even of the stored parameter, exactly torch's .to(bf16).
// (S90 2026-11-18) drop_out (bf16 [rows, cols], numel % 8 == 0), drop_p: the row-dropout scale of torch.rand's draw made in the same
// launch (forms_kernel's philox branch, as trimul_forms) -- the D256-512 training call's ds without its own two RNG-state fills, the
// rand launch and the drop_scale launch.
void wide_forms(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wp,
                torch::Tensor out, c10::optional<torch::Tensor> drop_out, double drop_p) {
  using namespace wide;
  const int R = (int)wl.size(0), D = (int)wl.size(1);
  const torch::Tensor* ts[6] = {&wl, &wlg, &wr, &wrg, &wg, &wp};
  FormSrc s;
  s.go = nullptr; s.rf3 = 0; s.tl = 0; s.tm = nullptr; s.fout = nullptr;
  s.dr = nullptr; s.dout = nullptr; s.pb = 0.f; s.dinv = 0.f; s.n8 = 0; s.philox = 0; s.pstride = 0;
  for (int k = 0; k < 6; ++k) {
    const torch::Tensor& t = *ts[k];
    TORCH_CHECK(t.dim() == 2 && t.is_cuda() && (t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16));
    TORCH_CHECK(k < 4 ? (t.size(0) == R && t.size(1) == D) : (t.size(0) == D && t.size(1) == (k == 4 ? D : R)), "wide_forms: shapes");
    s.p[k] = t.data_ptr(); s.s0[k] = t.stride(0); s.s1[k] = t.stride(1); s.f32[k] = t.scalar_type() == torch::kFloat32;
  }
  const int total = 8 * R * D + D * D + D * R;
  TORCH_CHECK(R % 64 == 0 && out.is_contiguous() && out.scalar_type() == torch::kBFloat16 && out.numel() == total);
  if (drop_out.has_value()) {
    TORCH_CHECK(drop_out->is_cuda() && drop_out->is_contiguous() && drop_out->scalar_type() == torch::kBFloat16 &&
                drop_out->numel() % 8 == 0 && drop_out->numel() > 0 && drop_out->numel() < (1ll << 31),
                "wide_forms: drop_out bf16 contiguous, numel % 8 == 0");
    s.dout = reinterpret_cast<uint4*>(drop_out->data_ptr());
    s.pb = __bfloat162float(__float2bfloat16_rn((float)drop_p)); s.dinv = 1.0f / (float)(1.0 - drop_p); s.n8 = (int)drop_out->numel();
    s.philox = 1;
    philox_reserve(drop_out->numel(), drop_out->device(), &s.rng, &s.pstride);
  }
  wa_launch(forms_kernel, (total + 255) / 256 + (s.n8 + 255) / 256, s, R, D, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total, D * R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// pf [>= nf, D (D + R)], lo [>= nl, 2R], pw [>= ns, 4 R D], li [>= nc, 2D] fp32 partials -> out [4 R D + D D + D R + 2D + 2R] fp32
void grad_reduce(torch::Tensor pf, int64_t nf, torch::Tensor lo, int64_t nl, torch::Tensor pw, int64_t ns, torch::Tensor li, int64_t nc,
                 torch::Tensor out, c10::optional<torch::Tensor> wl_scale, bool skip_o) {
  using namespace wide;
  const int D = (int)(li.size(-1) / 2), R = (int)(lo.size(-1) / 2);
  for (const torch::Tensor* t : {&pf, &lo, &pw, &li, &out}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32);
  TORCH_CHECK(pf.numel() >= nf * D * (D + R) && lo.numel() >= nl * 2 * R && pw.numel() >= ns * 4LL * R * D && li.numel() >= nc * 2 * D);
  const int total = 4 * R * D + D * D + D * R + 2 * D + 2 * R;
  TORCH_CHECK(out.numel() == total, "grad_reduce: out size");
  TORCH_CHECK(R % 64 == 0 && D % 64 == 0 && total % 128 == 0);
  // skip_o (2026-11-17 part 6): the output-side regions were summed by b7m's extra CTAs; only the pw blocks and the li blocks here
  const int nb = skip_o ? (4 * R * D + 2 * D) / 128 : total / 128;
  wa_launch<32 * GR_W>(grad_reduce_kernel, nb,
      pf.data_ptr<float>(), (int)nf, lo.data_ptr<float>(), (int)nl, pw.data_ptr<float>(), (int)ns, li.data_ptr<float>(), (int)nc, R, D,
      out.data_ptr<float>(), total, wl_scale.has_value() ? wl_scale->data_ptr<float>() : nullptr, (int)skip_o);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// wp [D, H], wg [D, D] both bf16 or both fp32 (the stored parameters: an fp32 master is folded from its fp32 value, one rounding);
// go / bo [H], gi / bi [D] fp32 -> wpq [D, H], wgq [D, D] bf16, vec [4, D] fp32 = (sp, ep, sg, eg)
void wide_fold_prep(torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor wg, torch::Tensor gi, torch::Tensor bi,
                    torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec) {
  using namespace wide;
  const int D = (int)wp.size(0), H = (int)wp.size(1);
  TORCH_CHECK(wp.is_contiguous() && wg.is_contiguous() && wpq.is_contiguous() && wgq.is_contiguous() && vec.is_contiguous());
  TORCH_CHECK(wp.scalar_type() == wg.scalar_type() && wpq.scalar_type() == torch::kBFloat16 && wgq.scalar_type() == torch::kBFloat16,
              "wide_fold_prep: wp / wg the same dtype, wpq / wgq bf16");
  auto st = at::cuda::getCurrentCUDAStream();
  auto q = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
  if (wp.scalar_type() == torch::kFloat32)
    fold_prep_t_kernel<float><<<2 * D, 256, 0, st>>>(wp.data_ptr<float>(), go.data_ptr<float>(), bo.data_ptr<float>(), wg.data_ptr<float>(),
                                                     gi.data_ptr<float>(), bi.data_ptr<float>(), q(wpq), q(wgq), vec.data_ptr<float>(), D, H);
  else
    fold_prep_t_kernel<__nv_bfloat16><<<2 * D, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(wp.data_ptr()), go.data_ptr<float>(), bo.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr()), gi.data_ptr<float>(), bi.data_ptr<float>(), q(wpq), q(wgq),
        vec.data_ptr<float>(), D, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================================ wide training: gradient pack, column sums
// (B200 only, 2026-11-17)
namespace wide {
// The D256-512 backward's ten parameter gradients written once, in parameter order, into one fp32 buffer
// [4 R D (wl, wlg, wr, wrg) | D D (wg) | D R (wp) | D (g_i) | D (b_i) | R (g_o) | R (b_o)] (the D64 / D128 grad_reduce layout):
// the front gradients unpacked from the weight-gradient GEMM's packed rows dw [4R + D, D], dWp = g_o (mm - r1) + r0 b_o from the
// GEMM mm = dpr^T t^T [D, R] (the same fp32 operations as the torch expression it replaces, no contraction), the LayerNorm vectors
// from acc [r0 | r1 | dg_i | db_i | dg_o | db_o]. One launch for what took ten copies and four elementwise launches.
__global__ void __launch_bounds__(256) grad_pack_kernel(const float* __restrict__ dw, const float* __restrict__ mm,
                                                        const float* __restrict__ acc, const float* __restrict__ go,
                                                        const float* __restrict__ bo, float* __restrict__ out, int R, int D, int total) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int e = blockIdx.x * 256 + threadIdx.x;
  if (e >= total) return;
  const int RD = R * D;
  float v;
  if (e < 4 * RD) {
    const int k = e / RD, r = (e % RD) / D, c = e % D;      // k: wl, wlg, wr, wrg
    const int plane = (k >= 2 ? R : 0) + r, gate = k & 1;
    v = dw[(size_t)((plane >> 6) * 128 + (gate ? 0 : 64) + (plane & 63)) * D + c];
  } else if (e < 4 * RD + D * D) {
    v = dw[(size_t)4 * RD + (e - 4 * RD)];
  } else if (e < 4 * RD + D * D + D * R) {
    const int f = e - 4 * RD - D * D, i = f / R, j = f % R;
    v = __fadd_rn(__fmul_rn(go[j], __fsub_rn(mm[f], acc[D + i])), __fmul_rn(acc[i], bo[j]));
  } else {
    v = acc[2 * D + (e - 4 * RD - D * D - D * R)];
  }
  out[e] = v;
}

// out[c] = sum_r part[r][c] over 32 warps (1024 threads; colsum_kernel's 8 warps left a few dozen blocks walking up to 2220 rows each,
// 7-17 us at L128): fixed order (warp w adds rows w, w + 32, ...; the warps' sums are added in warp order) -> deterministic.
__global__ void __launch_bounds__(1024) colsum32_kernel(const float* __restrict__ part, float* __restrict__ out, int R, int C) {
  __shared__ float red[32][33];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c = blockIdx.x * 32 + lane;
  float a = 0.f;
  if (c < C) {
#pragma unroll 4
    for (int r = warp; r < R; r += 32) a += part[(size_t)r * C + c];
  }
  red[warp][lane] = a;
  __syncthreads();
  if (warp == 0 && c < C) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 32; ++w) s += red[w][lane];
    out[c] = s;
  }
}

// (S90 2026-11-18) the wide backward's three per-block partial tables (gate_bwd's r0 / r1, lnout_bwd's dg_o / db_o, lnin_bwd's dg_i / db_i)
// reduced in ONE launch at the backward's end, instead of one colsum32 launch after each producer (three launches of 2-7 us each: 16-64
// blocks walking up to 1000 rows). A (tile = 32 columns of one table) x (row split) grid; each block sums its rows in a fixed order
// (warp w: rows w, w + 32, ...; the 32 warp sums in warp order), the LAST block of a tile (an arrival counter, zeroed by the gate_bwd
// launch's memset) adds the S row-split sums in split order -> deterministic.
struct Cs3 { const float* part[3]; float* out[3]; int R[3]; int t0[4]; };
__global__ void __launch_bounds__(1024) colsum3_kernel(Cs3 a, float* __restrict__ scr, unsigned* __restrict__ cnt, int S) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  __shared__ float red[32][33];
  __shared__ int last;
  const int tile = blockIdx.x, k = tile >= a.t0[2] ? 2 : (tile >= a.t0[1] ? 1 : 0), tl = tile - a.t0[k];
  const int C = 32 * (a.t0[k + 1] - a.t0[k]);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c = tl * 32 + lane;
  const int R = a.R[k], rs = (R + S - 1) / S, lo = blockIdx.y * rs, hi = min(R, lo + rs);
  float s = 0.f;
#pragma unroll 4
  for (int r = lo + warp; r < hi; r += 32) s += a.part[k][(size_t)r * C + c];
  red[warp][lane] = s;
  __syncthreads();
  if (warp == 0) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 32; ++w) v += red[w][lane];
    scr[((size_t)tile * S + blockIdx.y) * 32 + lane] = v;
    __threadfence();
  }
  __syncthreads();
  if (threadIdx.x == 0) last = (int)atomicAdd(cnt + tile, 1u) == S - 1;
  __syncthreads();
  if (last && warp == 0) {
    __threadfence();
    float v = 0.f;
    for (int y = 0; y < S; ++y) v += __ldcg(scr + ((size_t)tile * S + y) * 32 + lane);
    a.out[k][c] = v;
  }
}
}  // namespace wide (B200-only: wide training)

// dw [4R + D, D], mm [D, R], acc [4D + 2R], go / bo [R] fp32 -> out [4RD + DD + DR + 2D + 2R] fp32
void wide_grad_pack(torch::Tensor dw, torch::Tensor mm, torch::Tensor acc, torch::Tensor go, torch::Tensor bo, torch::Tensor out) {
  using namespace wide;
  const int D = (int)dw.size(1), R = (int)mm.size(1);
  for (const torch::Tensor* t : {&dw, &mm, &acc, &go, &bo, &out}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32);
  TORCH_CHECK(dw.size(0) == 4 * R + D && mm.size(0) == D && acc.numel() == 4 * D + 2 * R && go.numel() == R && bo.numel() == R);
  const int total = 4 * R * D + D * D + D * R + 2 * D + 2 * R;
  TORCH_CHECK(out.numel() == total, "wide_grad_pack: out size");
  wa_launch(grad_pack_kernel, (total + 255) / 256, dw.data_ptr<float>(), mm.data_ptr<float>(), acc.data_ptr<float>(), go.data_ptr<float>(),
            bo.data_ptr<float>(), out.data_ptr<float>(), R, D, total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// part [R, C] -> out [C] (fixed order), for the wide backward's per-block partials (wide_bwd.cu host functions)
void wide_colsum(const float* part, float* out, int R, int C, cudaStream_t st) {
  wide::colsum32_kernel<<<(C + 31) / 32, 1024, 0, st>>>(part, out, R, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The three partial tables of the wide backward -> their column sums (colsum3_kernel); cnt: [256 zeroed words | scratch], rs the row splits.
void wide_colsum3(torch::Tensor pg, torch::Tensor po, torch::Tensor pi, torch::Tensor og, torch::Tensor oo, torch::Tensor oi, torch::Tensor cnt,
                  int64_t S) {
  using namespace wide;
  Cs3 a;
  const torch::Tensor* ps[3] = {&pg, &po, &pi};
  const torch::Tensor* os[3] = {&og, &oo, &oi};
  a.t0[0] = 0;
  for (int k = 0; k < 3; ++k) {
    const int C = (int)ps[k]->size(1);
    TORCH_CHECK(ps[k]->is_contiguous() && ps[k]->scalar_type() == torch::kFloat32 && C % 32 == 0 && os[k]->numel() == C, "wide_colsum3: shapes");
    a.part[k] = ps[k]->data_ptr<float>(); a.out[k] = os[k]->data_ptr<float>(); a.R[k] = (int)ps[k]->size(0);
    a.t0[k + 1] = a.t0[k] + C / 32;
  }
  TORCH_CHECK(cnt.scalar_type() == torch::kFloat32 && cnt.numel() >= 256 + (int64_t)a.t0[3] * S * 32 && a.t0[3] <= 256, "wide_colsum3: cnt");
  float* base = cnt.data_ptr<float>();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.t0[3], (unsigned)S); cfg.blockDim = dim3(1024); cfg.dynamicSmemBytes = 0; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = WA_PDL;
  cfg.attrs = attr; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, colsum3_kernel, a, base + 256, reinterpret_cast<unsigned*>(base), (int)S));
}

// wide_ln_stats' B200 launch (programmatic dependent; see ln_stats_pdl_kernel)
bool wide_ln_stats_pdl(const __nv_bfloat16* t, float* mo, float* ro, int M, int H, float eps, bool wide) {
  using namespace wide;
#define LNP(HH) if (H == HH) { if (wide) wa_launch(ln_stats_pdl_kernel<HH, 8>, M / 256, t, mo, ro, M, eps); \
                               else wa_launch(ln_stats_pdl_kernel<HH, 2>, M / 64, t, mo, ro, M, eps); return true; }
  LNP(256) LNP(384) LNP(512) LNP(768) LNP(1024)
#undef LNP
  return false;
}
