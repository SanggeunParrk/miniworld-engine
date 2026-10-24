// Width-generic TriangleAttention glue for H100 (sm_90a): the hand-written kernels around the attention core (triattn_sm90.cu /
// triattn_bwd_sm90.cu) for every (d_pair C, heads H x 32) the fused d_pair-128 kernels (triattn_mod_sm90.cu) do not cover.  The
// projections and the weight gradients are cuBLAS GEMMs (bf16 in, fp32 accumulate); everything between them is here:
//
//   forward   ln_fwd         x -> y = LN(x) (bf16) + mean / rstd (fp32)
//             front_bias     the bias GEMM's [M, Hp] output -> head-major sbias [B, H, L, L] = bf16(bias / scale), masked keys -inf
//             (core)         triattn_fwd with `gate`: u = sigmoid(g) o O in its epilogue (O itself only when the backward needs it)
//             residual       out = x + drop o t        (t = u Wo^T from cuBLAS; drop = [B, L, C] broadcast over the rows i)
//   backward  gate_bwd_in    ddrop = dy o drop
//             gate_bwd_out   du -> do = du sigmoid(g), dg = du o sigmoid'(g) (into dall), Delta = sum do o O per head
//             (core)         triattn_bwd_kv / triattn_bwd_q write dk, dv, dq straight into their columns of dall
//             assemble       dbias (fp32 [B, H, L, L]) -> the bf16 bias columns of dall
//             ln_bwd         dx = dy + LN backward of (dall Wcat), and the per-CTA partial sums of d gamma / d beta
//
// Row m = (b, i, j) of the [M, .] matrices; dall is [M, 4 HD + Hp] (HD = 32 H, Hp = H rounded up to 8): the columns dq | dk | dv | dg |
// dbias, the layout of the packed weights Wcat = [Wq; Wk; Wv; Wg; Wb] so that dyln = dall Wcat and dW = dall^T y are one GEMM each.
// A row of C channels is handled by C / 8 lanes (C = 64, 128: a warp takes several rows) or by one warp (C <= 512).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace {
using bf16 = __nv_bfloat16;
constexpr int MAXV = 2;            // 16-byte vectors per lane and row: C <= 512

__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) { return make_uint4(pk(f[0], f[1]), pk(f[2], f[3]), pk(f[4], f[5]), pk(f[6], f[7])); }
__device__ __forceinline__ uint32_t pkh(float a, float b) {     // fp16 pair, a -> low half
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
__device__ __forceinline__ uint4 pack8h(const float (&f)[8]) { return make_uint4(pkh(f[0], f[1]), pkh(f[2], f[3]), pkh(f[4], f[5]), pkh(f[6], f[7])); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
// ending node: the logical row (b, i, j) of the starting layout is the physical row (b, j, i) of the pair tensor
__device__ __forceinline__ long phys_row(long row, int L, int tr) {
  if (!tr) return row;
  const long ll = (long)L * L, b = row / ll, rem = row - b * ll, i = rem / L, j = rem - i * L;
  return b * ll + j * L + i;
}
auto bp(const torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr<at::BFloat16>()); }
auto cbp(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); }
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// y = LN(x) * w + b: LPR lanes per row (LPR = C / 8 for C = 64 / 128: a warp takes 32 / LPR rows; 32 lanes, up to two vectors each, above),
// two-pass statistics in registers.
template <int LPR>
__device__ __forceinline__ float grp_sum(float v) {
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <int LPR>
__global__ void __launch_bounds__(256) ln_fwd_k(const bf16* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bs,
                                                bf16* __restrict__ y, float* __restrict__ mean_o, float* __restrict__ rstd_o, long M, int C,
                                                float eps, int L, int tr, int h16) {     // h16: y as fp16 bits (the fp16 core's GEMM input)
  constexpr int RPW = 32 / LPR, NA = LPR == 32 ? MAXV : 1;
  const int lane = threadIdx.x & 31, sub = lane % LPR, grp = lane / LPR;
  const int nv = C >> 3;
  for (long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW; row0 < M; row0 += (long)gridDim.x * 8 * RPW) {
    const long row = row0 + grp;
    const bool ok = row < M;
    float v[NA][8];
    float s = 0.f;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        unpack8(*reinterpret_cast<const uint4*>(x + phys_row(row, L, tr) * C + idx * 8), v[a]);
#pragma unroll
        for (int e = 0; e < 8; ++e) s += v[a][e];
      }
    }
    const float mean = grp_sum<LPR>(s) / C;
    float q = 0.f;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
#pragma unroll
        for (int e = 0; e < 8; ++e) { const float d = v[a][e] - mean; q = fmaf(d, d, q); }
      }
    }
    const float rstd = rsqrtf(grp_sum<LPR>(q) / C + eps);
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        const float4 w0 = *reinterpret_cast<const float4*>(w + idx * 8), w1 = *reinterpret_cast<const float4*>(w + idx * 8 + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(bs + idx * 8), b1 = *reinterpret_cast<const float4*>(bs + idx * 8 + 4);
        const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}, bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = fmaf((v[a][e] - mean) * rstd, ww[e], bb[e]);
        *reinterpret_cast<uint4*>(y + row * C + idx * 8) = h16 ? pack8h(o) : pack8(o);
      }
    }
    if (sub == 0 && ok) { mean_o[row] = mean; rstd_o[row] = rstd; }
  }
}

// sbias[b, h, i, j] = T(qkvb[(b, i, j), bias_col + h] / scale), -inf where mask[b, j] is false (T = bf16, or fp16 for the fp16 inference core:
// finite values clamped to its range).  A block = 128 consecutive j of one (b, i)
template <typename T>
__global__ void __launch_bounds__(256) front_bias_k(const T* __restrict__ qkvb, const bool* __restrict__ mask, T* __restrict__ sbias,
                                                    long ld, int bias_col, int H, int L, float inv_scale) {
  const long row0 = (long)blockIdx.x * 128;                  // (b * L + i) * L + j0
  const long bi = row0 / L;
  const int j0 = (int)(row0 - bi * L), b = (int)(bi / L), i = (int)(bi - (long)b * L);
  for (int idx = threadIdx.x; idx < 128 * H; idx += 256) {
    const int h = idx >> 7, r = idx & 127;
    const bool keep = mask == nullptr || mask[(long)b * L + j0 + r];
    const T in = qkvb[(row0 + r) * ld + bias_col + h];
    float f;
    if constexpr (std::is_same_v<T, __half>) f = __half2float(in); else f = __bfloat162float(in);
    const float val = keep ? f * inv_scale : -INFINITY;
    if constexpr (std::is_same_v<T, __half>) sbias[(((long)b * H + h) * L + i) * L + j0 + r] = __float2half_rn(keep ? fmaxf(fminf(val, 6.0e4f), -6.0e4f) : val);
    else sbias[(((long)b * H + h) * L + i) * L + j0 + r] = __float2bfloat16(val);
  }
}

// u = sigmoid(g) o o (g: the qkvb column block, o contiguous [M, HD]); u [M, HD]
__global__ void __launch_bounds__(256) gate_u_k(const bf16* __restrict__ qkvb, const bf16* __restrict__ o, bf16* __restrict__ u, long M,
                                                int HD, long ld, int g_col) {
  const int nv = HD >> 3;
  const long total = M * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv;
    const int c = (int)(i - row * nv) * 8;
    float g[8], ov[8], r[8];
    unpack8(*reinterpret_cast<const uint4*>(qkvb + row * ld + g_col + c), g);
    unpack8(*reinterpret_cast<const uint4*>(o + row * HD + c), ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = sigmoidf_(g[e]) * ov[e];
    *reinterpret_cast<uint4*>(u + row * HD + c) = pack8(r);
  }
}

// out = x + drop o t; drop [B, L, C] (index (b, j)) or nullptr
__global__ void __launch_bounds__(256) residual_k(const bf16* __restrict__ x, const bf16* __restrict__ t, const bf16* __restrict__ drop,
                                                  bf16* __restrict__ out, long M, int C, int L, int tr) {
  const int nv = C >> 3;
  const long total = M * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv;
    const long prow = phys_row(row, L, tr);
    const int c = (int)(i - row * nv) * 8;
    float xv[8], tv[8], r[8];
    unpack8(*reinterpret_cast<const uint4*>(x + prow * C + c), xv);
    unpack8(*reinterpret_cast<const uint4*>(t + row * C + c), tv);
    if (drop != nullptr) {
      const long bj = (row / ((long)L * L)) * L + row % L;
      float dv[8];
      unpack8(*reinterpret_cast<const uint4*>(drop + bj * C + c), dv);
#pragma unroll
      for (int e = 0; e < 8; ++e) tv[e] *= dv[e];
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = xv[e] + tv[e];
    *reinterpret_cast<uint4*>(out + prow * C + c) = pack8(r);
  }
}

// ddrop = dy o drop
__global__ void __launch_bounds__(256) gate_bwd_in_k(const bf16* __restrict__ dy, const bf16* __restrict__ drop, bf16* __restrict__ dd,
                                                     long M, int C, int L, int tr) {
  const int nv = C >> 3;
  const long total = M * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv;
    const int c = (int)(i - row * nv) * 8;
    float v[8];
    unpack8(*reinterpret_cast<const uint4*>(dy + phys_row(row, L, tr) * C + c), v);
    if (drop != nullptr) {
      const long bj = (row / ((long)L * L)) * L + row % L;
      float dv[8];
      unpack8(*reinterpret_cast<const uint4*>(drop + bj * C + c), dv);
#pragma unroll
      for (int e = 0; e < 8; ++e) v[e] *= dv[e];
    }
    *reinterpret_cast<uint4*>(dd + row * C + c) = pack8(v);
  }
}

// do = du sigmoid(g) -> do [M, HD];  dg = du o sigmoid'(g) -> dall[:, g_col ...];  delta[b, i, h, j] = sum_c do o o (the core backward's Delta;
// the 4 lanes of a head are consecutive threads)
__global__ void __launch_bounds__(256) gate_bwd_out_k(const bf16* __restrict__ du, const bf16* __restrict__ qkvb, const bf16* __restrict__ o,
                                                      bf16* __restrict__ dout, bf16* __restrict__ dall, float* __restrict__ delta, long M, int HD,
                                                      long ld, int g_col, long ld_all, int H, int L) {
  const int nv = HD >> 3;
  const long total = M * nv;
  for (long i0 = (long)blockIdx.x * 256; i0 < total; i0 += (long)gridDim.x * 256) {
    const long i = i0 + threadIdx.x;
    const bool ok = i < total;
    const long row = ok ? i / nv : 0;
    const int cv = ok ? (int)(i - row * nv) : 0, c = cv * 8;
    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0}, g[8] = {0, 0, 0, 0, 0, 0, 0, 0}, ov[8] = {0, 0, 0, 0, 0, 0, 0, 0}, a[8], b[8];
    if (ok) {
      unpack8(*reinterpret_cast<const uint4*>(du + row * HD + c), d);
      unpack8(*reinterpret_cast<const uint4*>(qkvb + row * ld + g_col + c), g);
      unpack8(*reinterpret_cast<const uint4*>(o + row * HD + c), ov);
    }
    float part = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float s = sigmoidf_(g[e]);
      a[e] = d[e] * s;
      b[e] = d[e] * ov[e] * s * (1.f - s);
    }
    const uint4 ao = pack8(a);
    const uint32_t aw[4] = {ao.x, ao.y, ao.z, ao.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { part = fmaf(bf_lo(aw[e]), ov[2 * e], part); part = fmaf(bf_hi(aw[e]), ov[2 * e + 1], part); }
    part += __shfl_xor_sync(0xffffffffu, part, 1);
    part += __shfl_xor_sync(0xffffffffu, part, 2);
    if (ok) {
      *reinterpret_cast<uint4*>(dout + row * HD + c) = ao;
      *reinterpret_cast<uint4*>(dall + row * ld_all + g_col + c) = pack8(b);
      if ((cv & 3) == 0) delta[((row / L) * H + (cv >> 2)) * L + row % L] = part;
    }
  }
}

// dall[:, 4 HD + h] <- bf16(dbias[b, h, i, j]) (zero for h >= H); the first 4 HD columns are written by the core backward and the gate backward.
// One thread per 16 bytes of a row's bias block.
__global__ void __launch_bounds__(256) assemble_k(const float* __restrict__ dbias, bf16* __restrict__ dall, long M, int HD, long ld, int H, int Hp,
                                                  int L) {
  const int nb = Hp >> 3;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < M * nb; i += (long)gridDim.x * 256) {
    const long row = i / nb;
    const int h0 = (int)(i - row * nb) * 8;
    const long bi = row / L;
    const int j = (int)(row - bi * L), b = (int)(bi / L), ii = (int)(bi - (long)b * L);
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = h0 + e < H ? dbias[(((long)b * H + h0 + e) * L + ii) * L + j] : 0.f;
    *reinterpret_cast<uint4*>(dall + row * ld + 4 * HD + h0) = pack8(f);
  }
}

__device__ __forceinline__ void load8(const float* p, float (&f)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ void load8(const bf16* p, float (&f)[8]) { unpack8(*reinterpret_cast<const uint4*>(p), f); }
template <int LPR, typename TD>
__global__ void __launch_bounds__(256) ln_bwd_k(const TD* __restrict__ dyln, const bf16* __restrict__ x, const bf16* __restrict__ dy,
                                                const float* __restrict__ w, const float* __restrict__ mean_i, const float* __restrict__ rstd_i,
                                                bf16* __restrict__ dx, float* __restrict__ part, long M, int C, int L, int tr) {
  constexpr int RPW = 32 / LPR, NA = LPR == 32 ? MAXV : 1;
  __shared__ float red[8192];                              // [8 warps x RPW][2][C]
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, sub = lane % LPR, grp = lane / LPR;
  const int nv = C >> 3;
  float pg[NA][8], pb[NA][8];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int e = 0; e < 8; ++e) { pg[a][e] = 0.f; pb[a][e] = 0.f; }
  for (long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW; row0 < M; row0 += (long)gridDim.x * 8 * RPW) {
    const long row = row0 + grp;
    const bool ok = row < M;
    const float mean = ok ? mean_i[row] : 0.f, rstd = ok ? rstd_i[row] : 0.f;
    float g[NA][8], xh[NA][8];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        float xv[8];
        unpack8(*reinterpret_cast<const uint4*>(x + phys_row(row, L, tr) * C + idx * 8), xv);
        float dd[8];
        load8(dyln + row * C + idx * 8, dd);
        const float4 w0 = *reinterpret_cast<const float4*>(w + idx * 8), w1 = *reinterpret_cast<const float4*>(w + idx * 8 + 4);
        const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          xh[a][e] = (xv[e] - mean) * rstd;
          g[a][e] = dd[e] * ww[e];
          pg[a][e] = fmaf(dd[e], xh[a][e], pg[a][e]);
          pb[a][e] += dd[e];
          s1 += g[a][e];
          s2 = fmaf(g[a][e], xh[a][e], s2);
        }
      }
    }
    const float m1 = grp_sum<LPR>(s1) / C, m2 = grp_sum<LPR>(s2) / C;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        float r[8];
        unpack8(*reinterpret_cast<const uint4*>(dy + phys_row(row, L, tr) * C + idx * 8), r);
#pragma unroll
        for (int e = 0; e < 8; ++e) r[e] += rstd * (g[a][e] - m1 - xh[a][e] * m2);
        *reinterpret_cast<uint4*>(dx + phys_row(row, L, tr) * C + idx * 8) = pack8(r);
      }
    }
  }
#pragma unroll
  for (int a = 0; a < NA; ++a) {
    const int idx = sub + LPR * a;
    if (idx < nv)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        red[((wi * RPW + grp) * 2) * C + idx * 8 + e] = pg[a][e];
        red[((wi * RPW + grp) * 2 + 1) * C + idx * 8 + e] = pb[a][e];
      }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < 2 * C; c += 256) {
    const int k = c / C, cc = c - k * C;
    float s = 0.f;
    for (int sl = 0; sl < 8 * RPW; ++sl) s += red[(sl * 2 + k) * C + cc];
    part[((long)blockIdx.x * 2 + k) * C + cc] = s;
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
namespace {
int grid_for(long work, int per_block) { return (int)std::min<long>((work + per_block - 1) / per_block, 132L * 16); }
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
}  // namespace

// x [M, C] bf16 -> (y bf16 [M, C], mean fp32 [M], rstd fp32 [M])
std::vector<torch::Tensor> g_ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, int64_t L, bool trans, bool h16) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x [M, C] bf16 contiguous");
  const long M = x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK(C % 8 == 0 && C <= 256 * MAXV, "C % 8 == 0 and C <= 512");
  auto y = h16 ? torch::empty_like(x, x.options().dtype(torch::kHalf)) : torch::empty_like(x);
  auto mean = torch::empty({M}, x.options().dtype(torch::kFloat32)), rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));
  const int lpr = C == 64 ? 8 : C == 128 ? 16 : 32;
  const int grid = grid_for(M, 8 * (32 / lpr));
  auto args = std::make_tuple(cbp(x), w.data_ptr<float>(), b.data_ptr<float>(), reinterpret_cast<bf16*>(y.data_ptr()), mean.data_ptr<float>(), rstd.data_ptr<float>(), M, C, (float)eps, (int)L, (int)trans,
                              (int)h16);
  if (lpr == 8) std::apply([&](auto... a) { ln_fwd_k<8><<<grid, 256, 0, stream()>>>(a...); }, args);
  else if (lpr == 16) std::apply([&](auto... a) { ln_fwd_k<16><<<grid, 256, 0, stream()>>>(a...); }, args);
  else std::apply([&](auto... a) { ln_fwd_k<32><<<grid, 256, 0, stream()>>>(a...); }, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// qkvb [M, ld] bf16 -> sbias [B, H, L, L] bf16; mask [B, L] bool or empty; L % 128 == 0
torch::Tensor g_front_bias(torch::Tensor qkvb, torch::Tensor mask, int64_t B, int64_t L, int64_t H, int64_t bias_col, double scale) {
  TORCH_CHECK(L % 128 == 0, "L % 128 == 0");
  auto sbias = torch::empty({B, H, L, L}, qkvb.options());
  torch::Tensor mk;
  const bool* mp = nullptr;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  if (qkvb.scalar_type() == torch::kHalf)          // fp16 GEMM output -> fp16 sbias (the fp16 inference core)
    front_bias_k<__half><<<(unsigned)(B * L * L / 128), 256, 0, stream()>>>(reinterpret_cast<const __half*>(qkvb.data_ptr()), mp,
        reinterpret_cast<__half*>(sbias.data_ptr()), qkvb.stride(0), (int)bias_col, (int)H, (int)L, (float)(1.0 / scale));
  else
    front_bias_k<bf16><<<(unsigned)(B * L * L / 128), 256, 0, stream()>>>(cbp(qkvb), mp, bp(sbias), qkvb.stride(0), (int)bias_col, (int)H, (int)L,
                                                                        (float)(1.0 / scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sbias;
}

torch::Tensor g_gate_u(torch::Tensor qkvb, torch::Tensor o, int64_t g_col) {
  const long M = o.size(0);
  const int HD = (int)o.size(1);
  auto u = torch::empty_like(o);
  gate_u_k<<<grid_for(M * (HD / 8), 256), 256, 0, stream()>>>(cbp(qkvb), cbp(o), bp(u), M, HD, qkvb.stride(0), (int)g_col);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return u;
}

torch::Tensor g_residual(torch::Tensor x, torch::Tensor t, torch::Tensor drop, int64_t L, bool trans) {
  const long M = x.size(0);
  const int C = (int)x.size(1);
  auto out = torch::empty_like(x);
  residual_k<<<grid_for(M * (C / 8), 256), 256, 0, stream()>>>(cbp(x), cbp(t), drop.numel() > 0 ? cbp(drop) : nullptr, bp(out), M, C, (int)L, (int)trans);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor g_gate_bwd_in(torch::Tensor dy, torch::Tensor drop, int64_t L, bool trans) {
  const long M = dy.size(0);
  const int C = (int)dy.size(1);
  auto dd = torch::empty_like(dy);
  gate_bwd_in_k<<<grid_for(M * (C / 8), 256), 256, 0, stream()>>>(cbp(dy), drop.numel() > 0 ? cbp(drop) : nullptr, bp(dd), M, C, (int)L, (int)trans);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dd;
}

// du [M, HD], qkvg [M, ld] (g at columns g_col ..), o [M, HD] -> (do [M, HD], delta fp32 [B, L, H, L]); the dg columns of dall [M, ld_all]
// (g_col ..) are filled, the rest is written by g_assemble
std::vector<torch::Tensor> g_gate_bwd_out(torch::Tensor du, torch::Tensor qkvb, torch::Tensor o, torch::Tensor dall, int64_t g_col, int64_t H,
                                          int64_t L) {
  const long M = du.size(0);
  const int HD = (int)du.size(1);
  auto dout = torch::empty_like(du);
  auto delta = torch::empty({M / L / L, L, H, L}, du.options().dtype(torch::kFloat32));
  gate_bwd_out_k<<<grid_for(M * (HD / 8), 256), 256, 0, stream()>>>(cbp(du), cbp(qkvb), cbp(o), bp(dout), bp(dall), delta.data_ptr<float>(), M, HD,
                                                                    qkvb.stride(0), (int)g_col, dall.stride(0), (int)H, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, delta};
}

void g_assemble(torch::Tensor dbias, torch::Tensor dall, int64_t H, int64_t L, int64_t HD) {
  const long M = dall.size(0);
  const int Hp = (int)(dall.size(1) - 4 * HD);
  TORCH_CHECK(Hp % 8 == 0 && Hp >= H && dall.stride(0) == dall.size(1), "dall [M, 4 HD + Hp]");
  assemble_k<<<grid_for(M * (Hp / 8), 256), 256, 0, stream()>>>(dbias.data_ptr<float>(), bp(dall), M, (int)HD, dall.stride(0), (int)H, Hp, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -> (dx bf16 [M, C], partial sums fp32 [grid, 2, C])
std::vector<torch::Tensor> g_ln_bwd(torch::Tensor dyln, torch::Tensor x, torch::Tensor dy, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                    int64_t L, bool trans) {
  const long M = x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK((dyln.scalar_type() == torch::kFloat32 || dyln.scalar_type() == torch::kBFloat16) && dyln.is_contiguous() && C % 8 == 0 && C <= 256 * MAXV,
              "dyln fp32 or bf16 [M, C]");
  const int lpr = C == 64 ? 8 : C == 128 ? 16 : 32;
  const int grid = grid_for(M, 8 * (32 / lpr));
  auto dx = torch::empty_like(x);
  auto part = torch::empty({grid, 2, C}, x.options().dtype(torch::kFloat32));
  auto launch = [&](auto* dptr) {
    using TD = std::remove_cv_t<std::remove_pointer_t<decltype(dptr)>>;
    auto args = std::make_tuple(dptr, cbp(x), cbp(dy), w.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), bp(dx), part.data_ptr<float>(), M, C, (int)L, (int)trans);
    if (lpr == 8) std::apply([&](auto... a) { ln_bwd_k<8, TD><<<grid, 256, 0, stream()>>>(a...); }, args);
    else if (lpr == 16) std::apply([&](auto... a) { ln_bwd_k<16, TD><<<grid, 256, 0, stream()>>>(a...); }, args);
    else std::apply([&](auto... a) { ln_bwd_k<32, TD><<<grid, 256, 0, stream()>>>(a...); }, args);
  };
  if (dyln.scalar_type() == torch::kFloat32) launch(dyln.data_ptr<float>());
  else launch(cbp(dyln));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, part};
}

// ---- weight pack: [Wq; Wk; Wv; Wg; Wb] (heads padded to 32 channels, Wb rows zero-padded to 64) and Wo [C, HD] in one launch
template <typename T> __device__ __forceinline__ bf16 to_bf(T v);
template <> __device__ __forceinline__ bf16 to_bf<float>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ bf16 to_bf<bf16>(bf16 v) { return v; }
template <typename T>
__global__ void __launch_bounds__(256) pack_generic_k(const T* __restrict__ wq, const T* __restrict__ wk, const T* __restrict__ wv, const T* __restrict__ wg,
                                                      const T* __restrict__ wb, const T* __restrict__ wo, bf16* __restrict__ wall, bf16* __restrict__ wop,
                                                      int C, int H, int hd) {
  const long HD = 32L * H, n1 = (4 * HD + 64) * C, n = n1 + (long)C * HD;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    bf16 v = __float2bfloat16_rn(0.f);
    if (i < n1) {
      const long row = i / C; const int col = (int)(i - row * C);
      if (row < 4 * HD) {
        const int part = (int)(row / HD), r = (int)(row - part * HD), h = r >> 5, d = r & 31;
        if (d < hd) {
          const T* w = part == 0 ? wq : part == 1 ? wk : part == 2 ? wv : wg;
          v = to_bf<T>(w[(long)(h * hd + d) * C + col]);
        }
      } else if (row - 4 * HD < H) {
        v = to_bf<T>(wb[(row - 4 * HD) * C + col]);
      }
      wall[i] = v;
    } else {
      const long j = i - n1, c = j / HD; const int r = (int)(j - c * HD), h = r >> 5, d = r & 31;
      if (d < hd) v = to_bf<T>(wo[c * (long)H * hd + h * hd + d]);
      wop[j] = v;
    }
  }
}

// -> (wall [4 HD + 64, C] bf16, wo_p [C, HD] bf16); weights fp32 or bf16, contiguous: wq .. wg [H hd, C], wb [H, C], wo [C, H hd]
std::vector<torch::Tensor> g_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo) {
  const int H = (int)wb.size(0), C = (int)wq.size(1), hd = (int)(wq.size(0) / H);
  TORCH_CHECK(hd >= 1 && hd <= 32 && wq.size(0) == (long)H * hd, "heads of at most 32 channels");
  for (const auto* t : {&wq, &wk, &wv, &wg, &wb, &wo}) TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == wq.scalar_type(), "contiguous, one dtype");
  const long HD = 32L * H;
  auto wall = torch::empty({4 * HD + 64, C}, wq.options().dtype(torch::kBFloat16));
  auto wop = torch::empty({C, HD}, wq.options().dtype(torch::kBFloat16));
  const long n = (4 * HD + 64) * C + (long)C * HD;
  const int grid = (int)std::min<long>((n + 255) / 256, 1024);
  if (wq.scalar_type() == torch::kFloat32)
    pack_generic_k<float><<<grid, 256, 0, stream()>>>(wq.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), wg.data_ptr<float>(),
                                                      wb.data_ptr<float>(), wo.data_ptr<float>(), bp(wall), bp(wop), C, H, hd);
  else {
    TORCH_CHECK(wq.scalar_type() == torch::kBFloat16, "fp32 or bf16 weights");
    pack_generic_k<bf16><<<grid, 256, 0, stream()>>>(cbp(wq), cbp(wk), cbp(wv), cbp(wg), cbp(wb), cbp(wo), bp(wall), bp(wop), C, H, hd);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wall, wop};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_fwd", &g_ln_fwd, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("L"), py::arg("trans"), py::arg("h16") = false);
  m.def("front_bias", &g_front_bias);
  m.def("gate_u", &g_gate_u);
  m.def("residual", &g_residual);
  m.def("gate_bwd_in", &g_gate_bwd_in);
  m.def("gate_bwd_out", &g_gate_bwd_out);
  m.def("assemble", &g_assemble);
  m.def("ln_bwd", &g_ln_bwd);
  m.def("pack", &g_pack);
}
