// B200 (sm_100a) LayerNorm backward of the bias-only TriangleAttention (C = 128 channels): the shared H100 glue's `ln_bwd_k<16>`
// (h100_sources/triattn_generic_sm90.cu, untouched) with the B200 memory pattern: the three row loads (x, dyln, dy) are issued
// together and one row step ahead of the arithmetic (the shared kernel loads dy only after both statistics reductions: one more
// dependent HBM round trip a row step), the weight vector is a register, a persistent grid of `gm` CTAs an SM (fewer d gamma / d beta
// partial rows for the column sum), streaming loads / stores.  The per-row arithmetic is the shared kernel's, in the same order: dx is
// bitwise equal; the d gamma / d beta partials group the rows differently (fp32 sums, deterministic for a fixed grid).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace {
using bf16 = __nv_bfloat16;
constexpr int C = 128, LPR = 16, RPW = 2;
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y); f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) { return make_uint4(pk(f[0], f[1]), pk(f[2], f[3]), pk(f[4], f[5]), pk(f[6], f[7])); }
__device__ __forceinline__ long phys_row(long row, int L, int tr) {
  if (!tr) return row;
  const long ll = (long)L * L, b = row / ll, rem = row - b * ll, i = rem / L, j = rem - i * L;
  return b * ll + j * L + i;
}
__device__ __forceinline__ float grp_sum(float v) {
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ uint4 ldcs(const void* p) { return __ldcs(reinterpret_cast<const uint4*>(p)); }

template <typename TD> struct Raw { uint4 x, y, d0, d1; float mean, rstd; };
__device__ unsigned g_ln_ticket;   // arrivals of the last-CTA column sum (self-resetting; zero at load)

template <typename TD, bool TR>
__device__ __forceinline__ void fetch(Raw<TD>& R, long row0, int grp, int sub, const TD* dyln, const bf16* x, const bf16* dy, const float* mean_i,
                                      const float* rstd_i, long M, int L, int tr) {
  const long pr = row0 + grp;                      // physical row (x, dy, dx)
  if (pr < M) {
    const long row = TR ? phys_row(pr, L, 1) : pr;  // the logical row (dyln, mean, rstd); the swap is its own inverse
    R.mean = mean_i[row]; R.rstd = rstd_i[row];
    R.x = ldcs(x + pr * C + sub * 8);
    R.y = ldcs(dy + pr * C + sub * 8);
    R.d0 = ldcs(dyln + row * C + sub * 8);
    if constexpr (std::is_same_v<TD, float>) R.d1 = ldcs(dyln + row * C + sub * 8 + 4);
  }
}

template <typename TD, bool TR, bool FOLD>
__global__ void __launch_bounds__(256) ln_bwd128_k(const TD* __restrict__ dyln, const bf16* __restrict__ x, const bf16* __restrict__ dy,
                                                   const float* __restrict__ w, const float* __restrict__ mean_i, const float* __restrict__ rstd_i,
                                                   bf16* __restrict__ dx, float* __restrict__ part, float* __restrict__ dgb, long M, int L, int tr) {
  __shared__ float red[8 * RPW * 2 * C];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, sub = lane % LPR, grp = lane / LPR;
  float ww[8];
  {
    const float4 w0 = *reinterpret_cast<const float4*>(w + sub * 8), w1 = *reinterpret_cast<const float4*>(w + sub * 8 + 4);
    ww[0] = w0.x; ww[1] = w0.y; ww[2] = w0.z; ww[3] = w0.w; ww[4] = w1.x; ww[5] = w1.y; ww[6] = w1.z; ww[7] = w1.w;
  }
  float pg[8], pb[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) { pg[e] = 0.f; pb[e] = 0.f; }
  const long stride = (long)gridDim.x * 8 * RPW;
  long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW;
  Raw<TD> cur, nxt;
  fetch<TD, TR>(cur, row0, grp, sub, dyln, x, dy, mean_i, rstd_i, M, L, tr);
  for (; row0 < M; row0 += stride) {
    fetch<TD, TR>(nxt, row0 + stride, grp, sub, dyln, x, dy, mean_i, rstd_i, M, L, tr);
    const long row = row0 + grp;
    const bool ok = row < M;
    float s1 = 0.f, s2 = 0.f, g[8], xh[8];
    if (ok) {
      float xv[8], dd[8];
      unpack8(cur.x, xv);
      if constexpr (std::is_same_v<TD, float>) {
        dd[0] = __uint_as_float(cur.d0.x); dd[1] = __uint_as_float(cur.d0.y); dd[2] = __uint_as_float(cur.d0.z); dd[3] = __uint_as_float(cur.d0.w);
        dd[4] = __uint_as_float(cur.d1.x); dd[5] = __uint_as_float(cur.d1.y); dd[6] = __uint_as_float(cur.d1.z); dd[7] = __uint_as_float(cur.d1.w);
      } else {
        unpack8(cur.d0, dd);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        xh[e] = (xv[e] - cur.mean) * cur.rstd;
        g[e] = dd[e] * ww[e];
        pg[e] = fmaf(dd[e], xh[e], pg[e]);
        pb[e] += dd[e];
        s1 += g[e];
        s2 = fmaf(g[e], xh[e], s2);
      }
    }
    const float m1 = grp_sum(s1) / C, m2 = grp_sum(s2) / C;
    if (ok) {
      float r[8];
      unpack8(cur.y, r);
#pragma unroll
      for (int e = 0; e < 8; ++e) r[e] += cur.rstd * (g[e] - m1 - xh[e] * m2);
      __stcs(reinterpret_cast<uint4*>(dx + row * C + sub * 8), pack8(r));
    }
    cur = nxt;
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    red[((wi * RPW + grp) * 2) * C + sub * 8 + e] = pg[e];
    red[((wi * RPW + grp) * 2 + 1) * C + sub * 8 + e] = pb[e];
  }
  __syncthreads();
  for (int c = threadIdx.x; c < 2 * C; c += 256) {
    const int k = c / C, cc = c - k * C;
    float s = 0.f;
    for (int sl = 0; sl < 8 * RPW; ++sl) s += red[(sl * 2 + k) * C + cc];
    part[((long)blockIdx.x * 2 + k) * C + cc] = s;
  }
  if constexpr (FOLD) {   // (off by default: the caller's bo_colsum is 2-4 us faster than this fold at L >= 256)
  __shared__ unsigned last;
  // fold: the last CTA to arrive sums the partial rows in a fixed order (row 0 .. grid - 1): d gamma / d beta without a second launch
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned t = atomicAdd(&g_ln_ticket, 1u);
    last = (t == gridDim.x - 1) ? 1u : 0u;
    if (last) g_ln_ticket = 0u;
  }
  __syncthreads();
  if (last) {
    __threadfence();
    for (int c = threadIdx.x; c < 2 * C; c += 256) {
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      int r = 0;
      for (; r + 4 <= (int)gridDim.x; r += 4) {
        s0 += __ldcg(part + ((long)(r) * 2) * C + c); s1 += __ldcg(part + ((long)(r + 1) * 2) * C + c);
        s2 += __ldcg(part + ((long)(r + 2) * 2) * C + c); s3 += __ldcg(part + ((long)(r + 3) * 2) * C + c);
      }
      for (; r < (int)gridDim.x; ++r) s0 += __ldcg(part + ((long)r * 2) * C + c);
      dgb[c] = (s0 + s1) + (s2 + s3);
    }
  }
  }
}

// y = LN(x) * w + b, C = 128, starting layout (the shared glue's ln_fwd_k<16>, same arithmetic order: y / mean / rstd bitwise equal): the next row
// step's load is issued before this one's reductions, w / b are registers, a persistent grid, streaming stores.
__global__ void __launch_bounds__(256) ln_fwd128_k(const bf16* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bs, bf16* __restrict__ y,
                                                   float* __restrict__ mean_o, float* __restrict__ rstd_o, long M, float eps) {
  const int lane = threadIdx.x & 31, sub = lane % LPR, grp = lane / LPR;
  float ww[8], bb[8];
  {
    const float4 w0 = *reinterpret_cast<const float4*>(w + sub * 8), w1 = *reinterpret_cast<const float4*>(w + sub * 8 + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(bs + sub * 8), b1 = *reinterpret_cast<const float4*>(bs + sub * 8 + 4);
    ww[0] = w0.x; ww[1] = w0.y; ww[2] = w0.z; ww[3] = w0.w; ww[4] = w1.x; ww[5] = w1.y; ww[6] = w1.z; ww[7] = w1.w;
    bb[0] = b0.x; bb[1] = b0.y; bb[2] = b0.z; bb[3] = b0.w; bb[4] = b1.x; bb[5] = b1.y; bb[6] = b1.z; bb[7] = b1.w;
  }
  const long stride = (long)gridDim.x * 8 * RPW;
  long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW;
  uint4 cur = make_uint4(0, 0, 0, 0), nxt;
  if (row0 + grp < M) cur = ldcs(x + (row0 + grp) * C + sub * 8);
  for (; row0 < M; row0 += stride) {
    nxt = make_uint4(0, 0, 0, 0);
    if (row0 + stride + grp < M) nxt = ldcs(x + (row0 + stride + grp) * C + sub * 8);
    const long row = row0 + grp;
    const bool ok = row < M;
    float v[8], s = 0.f;
    if (ok) {
      unpack8(cur, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) s += v[e];
    }
    const float mean = grp_sum(s) / C;
    float q = 0.f;
    if (ok) {
#pragma unroll
      for (int e = 0; e < 8; ++e) { const float d = v[e] - mean; q = fmaf(d, d, q); }
    }
    const float rstd = rsqrtf(grp_sum(q) / C + eps);
    if (ok) {
      float o[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = fmaf((v[e] - mean) * rstd, ww[e], bb[e]);
      __stcs(reinterpret_cast<uint4*>(y + row * C + sub * 8), pack8(o));
    }
    if (sub == 0 && ok) { mean_o[row] = mean; rstd_o[row] = rstd; }
    cur = nxt;
  }
}

// ---- gate backward with the du GEMM (K = 128): the shared `gate_bwd_fused_k<128>` (h100_sources/triattn_bidir_sm90.cu, untouched), bitwise the same
// arithmetic, with the gate / o planes' loads of a tile issued BEFORE its mma.sync GEMM (the shared kernel loads them after it: one exposed HBM round
// trip a tile; 2 CTAs an SM do not cover it).
namespace gb {
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
}  // namespace gb

__global__ void __launch_bounds__(256, 2) gate_bwd128_k(const bf16* __restrict__ DY, const bf16* __restrict__ Wo, const bf16* __restrict__ G, const bf16* __restrict__ O,
                                                        bf16* __restrict__ dout, bf16* __restrict__ dg, long LL, long ldg, long ldo, long lddg) {
  using namespace gb;
  constexpr int K = 128, TMB = 64, SD = 136, WS = K + 8, WR = K / 8, BUF = 9216;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sWo = reinterpret_cast<bf16*>(smraw);
  bf16* sDy = sWo + 128 * WS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, q4 = lane & 3, g8 = lane >> 2, mi = lane >> 3;
  for (int i = tid; i < 128 * (K / 8); i += 256) {
    const int n = i / (K / 8), c = i % (K / 8);
    cp16(sWo + n * WS + c * 8, Wo + (long)n * K + c * 8);
  }
  cp_commit();
  const long ntiles = LL / TMB;
  auto issue = [&](long tile, int s) {
    for (int i = tid; i < TMB * 16; i += 256) {
      const int r = i >> 4, ch = i & 15;
      cp16(sDy + s * BUF + r * SD + ch * 8, DY + (tile * TMB + r) * 128 + ch * 8);
    }
    cp_commit();
  };
  if ((long)blockIdx.x < ntiles) issue(blockIdx.x, 0);
  else cp_commit();
  int it = 0;
  for (long tile = blockIdx.x; tile < ntiles; tile += gridDim.x, ++it) {
    if (tile + gridDim.x < ntiles) issue(tile + gridDim.x, (it + 1) & 1);
    else cp_commit();
    const long tok0 = tile * TMB;
    uint4 gg[4], oo[4];                                  // this warp's 16 channels of the gate / o planes: in flight during the GEMM
    {
      const long tk = tok0 + (lane & 7) * 8;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const long row = w * WR + i * 4 + (lane >> 3);
        gg[i] = __ldcs(reinterpret_cast<const uint4*>(G + row * ldg + tk));
        oo[i] = __ldcs(reinterpret_cast<const uint4*>(O + row * ldo + tk));
      }
    }
    cp_wait<1>();
    __syncthreads();
    const bf16* dy = sDy + (it & 1) * BUF;
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      unsigned a[4];
      ldsm4t(a, sWo + (ks * 16 + (mi >> 1) * 8 + (lane & 7)) * WS + w * WR + (mi & 1) * 8);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned b[4];
        ldsm4(b, dy + (j * 16 + (mi >> 1) * 8 + (lane & 7)) * SD + ks * 16 + (mi & 1) * 8);
        mma16816(acc[2 * j], a, b[0], b[1]);
        mma16816(acc[2 * j + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();
    bf16* stg = sDy + (it & 1) * BUF + w * 16 * 72;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = j * 8 + 2 * q4;
      *reinterpret_cast<__nv_bfloat162*>(stg + g8 * 72 + c) = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
      *reinterpret_cast<__nv_bfloat162*>(stg + (g8 + 8) * 72 + c) = __floats2bfloat162_rn(acc[j][2], acc[j][3]);
    }
    __syncwarp();
    uint4 du[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) du[i] = *reinterpret_cast<const uint4*>(stg + (i * 4 + (lane >> 3)) * 72 + (lane & 7) * 8);
    __syncwarp();
    const long tk = tok0 + (lane & 7) * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float a[8], gv[8], ovv[8], r[8], q[8];
      unpack8(du[i], a);
      unpack8(gg[i], gv);
      unpack8(oo[i], ovv);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s = 1.f / (1.f + __expf(-gv[e]));
        r[e] = a[e] * s;
        q[e] = a[e] * ovv[e] * s * (1.f - s);
      }
      const long row = w * WR + i * 4 + (lane >> 3);
      *reinterpret_cast<uint4*>(dout + row * LL + tk) = pack8(r);
      *reinterpret_cast<uint4*>(dg + row * lddg + tk) = pack8(q);
    }
    __syncthreads();
  }
}

// d gamma | d beta = the partial rows [n, 2, 128] summed in row order, ONE launch (the shared `colsum` of bidir_sm100.cu is two): 8 CTAs of 32 columns,
// each thread one column over every 8th row, the 8 row groups added in a fixed order.
__global__ void __launch_bounds__(256) colsum256_k(const float* __restrict__ part, float* __restrict__ out, int n) {
  __shared__ float red[8][32];
  const int c = blockIdx.x * 32 + (threadIdx.x & 31), rg = threadIdx.x >> 5;
  float s = 0.f;
  for (int r = rg; r < n; r += 8) s += __ldcg(part + (long)r * 256 + c);
  red[rg][threadIdx.x & 31] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = 0.f;
#pragma unroll
    for (int g = 0; g < 8; ++g) t += red[g][threadIdx.x];
    out[blockIdx.x * 32 + threadIdx.x] = t;
  }
}

}  // namespace

// -> (dx bf16 [M, 128], partial sums fp32 [grid, 2, 128]) or, fold, (dx, d gamma | d beta fp32 [2, 128] summed by the last CTA); gm CTAs an SM
std::vector<torch::Tensor> ln_bwd128(torch::Tensor dyln, torch::Tensor x, torch::Tensor dy, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, int64_t L,
                                     bool trans, int64_t gm, bool fold) {
  const long M = x.size(0);
  TORCH_CHECK(x.size(1) == C && x.is_contiguous() && dy.is_contiguous() && dyln.is_contiguous() && (dyln.scalar_type() == torch::kFloat32 || dyln.scalar_type() == torch::kBFloat16),
              "ln_bwd128: C = 128, dyln fp32 / bf16");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = (int)std::min<long>((M + 8 * RPW - 1) / (8 * RPW), (long)sms * gm);
  auto dx = torch::empty_like(x);
  auto part = torch::empty({grid, 2, C}, x.options().dtype(torch::kFloat32));
  auto dgb = torch::empty({2, C}, x.options().dtype(torch::kFloat32));
  auto xb = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()); auto dyb = reinterpret_cast<const bf16*>(dy.data_ptr<at::BFloat16>());
  auto dxb = reinterpret_cast<bf16*>(dx.data_ptr<at::BFloat16>());
  auto go = [&](auto tdp, auto trc) {
    using TD = std::remove_cv_t<std::remove_pointer_t<decltype(tdp)>>;
    constexpr bool TR = decltype(trc)::value;
    constexpr bool FOLD = false;   // (the fold variant is built by changing this: measured slower)
    ln_bwd128_k<TD, TR, FOLD><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const TD*>(dyln.data_ptr()), xb, dyb, w.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), dxb, part.data_ptr<float>(), dgb.data_ptr<float>(), M, (int)L, (int)trans);
  };
  if (dyln.scalar_type() == torch::kFloat32) { if (trans) go((float*)nullptr, std::true_type{}); else go((float*)nullptr, std::false_type{}); }
  else { if (trans) go((bf16*)nullptr, std::true_type{}); else go((bf16*)nullptr, std::false_type{}); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(!fold, "ln_bwd128: the fold variant (FOLD in the source) is not built");
  return {dx, part};
}

// -> (y bf16 [M, 128], mean fp32 [M], rstd fp32 [M])
std::vector<torch::Tensor> ln_fwd128(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, int64_t gm) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == C && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "ln_fwd128: x [M, 128] bf16 contiguous");
  const long M = x.size(0);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = (int)std::min<long>((M + 8 * RPW - 1) / (8 * RPW), (long)sms * gm);
  auto y = torch::empty_like(x);
  auto mean = torch::empty({M}, x.options().dtype(torch::kFloat32)), rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));
  ln_fwd128_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()), w.data_ptr<float>(), b.data_ptr<float>(),
      reinterpret_cast<bf16*>(y.data_ptr<at::BFloat16>()), mean.data_ptr<float>(), rstd.data_ptr<float>(), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// dy [LL, 128], wo [128, 128], g / o [128, LL] gate planes -> do [128, LL]; dg (a [128, LL] view) is written
torch::Tensor gate_bwd128(torch::Tensor dy, torch::Tensor wo, torch::Tensor g, torch::Tensor o, torch::Tensor dg, int64_t gm) {
  const long LL = dy.size(0);
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == torch::kBFloat16 && dy.is_contiguous() && dy.dim() == 2 && dy.size(1) == 128 && wo.scalar_type() == torch::kBFloat16 &&
              wo.is_contiguous() && wo.dim() == 2 && wo.size(0) == 128 && wo.size(1) == 128 && LL % 64 == 0 && g.size(0) == 128 && g.size(1) == LL && o.size(0) == 128 &&
              o.size(1) == LL && dg.size(0) == 128 && dg.size(1) == LL && g.stride(1) == 1 && o.stride(1) == 1 && dg.stride(1) == 1 && g.stride(0) % 8 == 0 &&
              o.stride(0) % 8 == 0 && dg.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dg.data_ptr()) % 16 == 0, "gate_bwd128: dy [LL, 128], wo [128, 128], planes [128, LL]");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto dout = torch::empty({128, LL}, dy.options());
  const int grid = (int)std::min<long>(LL / 64, (long)sms * gm);
  const size_t smem = (size_t)(128 * 136 + 2 * 9216) * 2;
  C10_CUDA_CHECK(cudaFuncSetAttribute(gate_bwd128_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  gate_bwd128_k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(dy.data_ptr<at::BFloat16>()), reinterpret_cast<const bf16*>(wo.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(g.data_ptr<at::BFloat16>()), reinterpret_cast<const bf16*>(o.data_ptr<at::BFloat16>()), reinterpret_cast<bf16*>(dout.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(dg.data_ptr<at::BFloat16>()), LL, g.stride(0), o.stride(0), dg.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dout;
}

// part [n, 2, 128] fp32 -> [2, 128] fp32, the rows summed in order
torch::Tensor colsum256(torch::Tensor part) {
  TORCH_CHECK(part.is_cuda() && part.scalar_type() == torch::kFloat32 && part.is_contiguous() && part.dim() == 3 && part.size(1) == 2 && part.size(2) == C, "colsum256: fp32 [n, 2, 128]");
  auto out = torch::empty({2, C}, part.options());
  colsum256_k<<<8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), out.data_ptr<float>(), (int)part.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("colsum256", &colsum256, "B200 column sum of the LayerNorm backward's partial rows");
  m.def("gate_bwd128", &gate_bwd128, "B200 fused du GEMM + gate backward, K = 128");
  m.def("ln_bwd128", &ln_bwd128, "B200 LayerNorm backward, C = 128");
  m.def("ln_fwd128", &ln_fwd128, "B200 LayerNorm forward, C = 128");
}
