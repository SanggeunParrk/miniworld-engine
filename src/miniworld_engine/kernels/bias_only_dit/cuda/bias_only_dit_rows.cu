// Bias-only token DiT row kernels (CUDA): the passes between the GEMMs of the fused inference step.
//
//   ln_rows             out = LN(x), no affine                                   (the conditioning rows, width 384)
//   adaln_in_rows       x = single (fp32 copy of the input); out = LN(x) * sigmoid(ms[tok]) + mb[tok]
//   resgate_adaln_rows  x += sigmoid(gl[tok]) * y (fp32, in place); then, when ms is given, out = LN(x) * sigmoid(ms) + mb
//   resgate_out_rows    out = x + sigmoid(gl[tok]) * y in the output dtype      (the last half-block: no fp32 round trip)
//   swiglu_rows         out = silu(a) * b, ab = [a | b]
//   gate_bwd_rows       training backward of a = sigmoid(g) o:  do = da sigmoid(g), dg = da a (1 - sigmoid(g)), and the per-sample
//                       D rows dd[a, h, i] = sum_d da a (= sum_d do o, the softmax backward's row term)
//   transpose_hll       [H][L][L] -> its per-head transpose (P^T for dV = P^T do)
//   softmax_rows        p = softmax(bias row) over the keys, masked keys at the largest negative float (a fully masked row is
//                       uniform, as the torch reference's finfo.min fill makes it)
//
// tok = row % T, T = the table's row count (L for one conditioning shared by the samples, S L for one per sample).
//
// ONE WARP PER ROW, no block barrier: a lane owns the float4 chunks lane, lane + 32, ... (a warp access is 512 contiguous bytes of
// fp32, 256 of bf16), issues every load of its row before any math, and the row statistics are two warp reductions over values
// held in registers (two-pass variance). The token DiT's block-per-row kernels (kernels/conditioned_transition/cuda, two
// __syncthreads reductions per row) took 5-6.5 us per pass at 1920 rows; these are the same arithmetic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;

constexpr int WPB = 4;                                   // rows (warps) per block

__device__ __forceinline__ float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
// a table entry that passes through a sigmoid: already applied by the table kernel when ps (cond_tables_sm100.cu)
__device__ __forceinline__ float gs(float v, bool ps) { return ps ? v : sg(v); }

// the row's mean and rstd from the NCH float4 chunks a lane holds
template <int NCH>
__device__ __forceinline__ void row_stats(const float4 (&x)[NCH], float eps, float& mean, float& rstd) {
  constexpr float inv = 1.f / (NCH * 128);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NCH; ++j) s += (x[j].x + x[j].y) + (x[j].z + x[j].w);
  mean = warp_sum(s) * inv;
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const float a = x[j].x - mean, b = x[j].y - mean, c = x[j].z - mean, d = x[j].w - mean;
    q += (a * a + b * b) + (c * c + d * d);
  }
  rstd = rsqrtf(warp_sum(q) * inv + eps);
}

// out = (x - mean) rstd sigmoid(ms) + mb for this lane's chunks
template <int NCH, typename GT, typename OT>
__device__ __forceinline__ void adaln_store(const float4 (&x)[NCH], float eps, const GT* ms, const GT* mb, OT* out, int lane, bool ps) {
  float mean, rstd;
  row_stats<NCH>(x, eps, mean, rstd);
  float4 m[NCH], h[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) { m[j] = V4<GT>::load(ms + (j * 32 + lane) * 4); h[j] = V4<GT>::load(mb + (j * 32 + lane) * 4); }
#pragma unroll
  for (int j = 0; j < NCH; ++j)
    V4<OT>::store(out + (j * 32 + lane) * 4, make_float4((x[j].x - mean) * rstd * gs(m[j].x, ps) + h[j].x,
                                                         (x[j].y - mean) * rstd * gs(m[j].y, ps) + h[j].y,
                                                         (x[j].z - mean) * rstd * gs(m[j].z, ps) + h[j].z,
                                                         (x[j].w - mean) * rstd * gs(m[j].w, ps) + h[j].w));
}

template <int NCH, typename IT, typename OT>
__global__ void __launch_bounds__(WPB * 32) ln_rows_kernel(const IT* __restrict__ X, OT* __restrict__ OUT, long R, long sx, float eps) {
  const long row = (long)blockIdx.x * WPB + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 x[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) x[j] = V4<IT>::load(X + row * sx + (j * 32 + lane) * 4);
  float mean, rstd;
  row_stats<NCH>(x, eps, mean, rstd);
#pragma unroll
  for (int j = 0; j < NCH; ++j)
    V4<OT>::store(OUT + row * (NCH * 128) + (j * 32 + lane) * 4, make_float4((x[j].x - mean) * rstd, (x[j].y - mean) * rstd,
                                                                             (x[j].z - mean) * rstd, (x[j].w - mean) * rstd));
}

template <int NCH, typename IT, typename GT, typename OT>
__global__ void __launch_bounds__(WPB * 32) adaln_in_rows_kernel(const IT* __restrict__ IN, float* __restrict__ X,
    const GT* __restrict__ MS, const GT* __restrict__ MB, OT* __restrict__ OUT, long R, int T, long sin, long sms, long smb, float eps,
    bool ps) {
  const long row = (long)blockIdx.x * WPB + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  const int tok = row % T;
  float4 x[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) x[j] = V4<IT>::load(IN + row * sin + (j * 32 + lane) * 4);
#pragma unroll
  for (int j = 0; j < NCH; ++j) V4<float>::store(X + row * (NCH * 128) + (j * 32 + lane) * 4, x[j]);
  adaln_store<NCH>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NCH * 128), lane, ps);
}

template <int NCH, typename YT, typename GT, typename OT>
__global__ void __launch_bounds__(WPB * 32) resgate_adaln_rows_kernel(float* __restrict__ X, const YT* __restrict__ Y,
    const GT* __restrict__ GL, const GT* __restrict__ MS, const GT* __restrict__ MB, OT* __restrict__ OUT, long R, int T, long sy,
    long sgl, long sms, long smb, float eps, bool has_adaln, bool ps) {
  const long row = (long)blockIdx.x * WPB + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  const int tok = row % T;
  float4 x[NCH], y[NCH], g[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const int c = (j * 32 + lane) * 4;
    x[j] = V4<float>::load(X + row * (NCH * 128) + c); y[j] = V4<YT>::load(Y + row * sy + c); g[j] = V4<GT>::load(GL + tok * sgl + c);
  }
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    x[j].x += gs(g[j].x, ps) * y[j].x; x[j].y += gs(g[j].y, ps) * y[j].y; x[j].z += gs(g[j].z, ps) * y[j].z; x[j].w += gs(g[j].w, ps) * y[j].w;
    V4<float>::store(X + row * (NCH * 128) + (j * 32 + lane) * 4, x[j]);
  }
  if (has_adaln) adaln_store<NCH>(x, eps, MS + tok * sms, MB + tok * smb, OUT + row * (NCH * 128), lane, ps);
}

template <int NCH, typename YT, typename GT, typename OT>
__global__ void __launch_bounds__(WPB * 32) resgate_out_rows_kernel(const float* __restrict__ X, const YT* __restrict__ Y,
    const GT* __restrict__ GL, OT* __restrict__ OUT, long R, int T, long sy, long sgl, bool ps) {
  const long row = (long)blockIdx.x * WPB + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  const int tok = row % T;
  float4 x[NCH], y[NCH], g[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const int c = (j * 32 + lane) * 4;
    x[j] = V4<float>::load(X + row * (NCH * 128) + c); y[j] = V4<YT>::load(Y + row * sy + c); g[j] = V4<GT>::load(GL + tok * sgl + c);
  }
#pragma unroll
  for (int j = 0; j < NCH; ++j)
    V4<OT>::store(OUT + row * (NCH * 128) + (j * 32 + lane) * 4, make_float4(x[j].x + gs(g[j].x, ps) * y[j].x, x[j].y + gs(g[j].y, ps) * y[j].y,
                                                                             x[j].z + gs(g[j].z, ps) * y[j].z, x[j].w + gs(g[j].w, ps) * y[j].w));
}

// out = silu(a) * b over [M, N] (ab = [a | b] rows of 2N), 8 elements (16 B of bf16) per thread
__global__ void __launch_bounds__(256) swiglu_rows_kernel(const __nv_bfloat16* __restrict__ AB, __nv_bfloat16* __restrict__ OUT,
    long total8, int n8, long sab) {
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= total8) return;
  const long row = i / n8;
  const int c = (int)(i % n8) * 8;
  const __nv_bfloat16* ab = AB + row * sab + c;
  const float4 a0 = V4<__nv_bfloat16>::load(ab), a1 = V4<__nv_bfloat16>::load(ab + 4);
  const float4 b0 = V4<__nv_bfloat16>::load(ab + n8 * 8), b1 = V4<__nv_bfloat16>::load(ab + n8 * 8 + 4);
  __nv_bfloat16* o = OUT + row * (n8 * 8) + c;
  V4<__nv_bfloat16>::store(o, make_float4(a0.x * sg(a0.x) * b0.x, a0.y * sg(a0.y) * b0.y, a0.z * sg(a0.z) * b0.z, a0.w * sg(a0.w) * b0.w));
  V4<__nv_bfloat16>::store(o + 4, make_float4(a1.x * sg(a1.x) * b1.x, a1.y * sg(a1.y) * b1.y, a1.z * sg(a1.z) * b1.z, a1.w * sg(a1.w) * b1.w));
}

// One warp per row of 768 = 16 heads x 48, 24 x 32 or 12 x 64 (or 1024 = 16 x 64); lane l owns the 4-wide chunks c = l, l + 32, ... (every access 256 contiguous bytes).
// Chunk c belongs to head c / 12: the warp's 192 chunk sums go through shared memory and lanes 0..15 add their head's twelve.
template <int NCH>                                   // 4-wide chunks per lane: 6 (768 channels) or 8 (1024: 16 x 64)
__global__ void __launch_bounds__(WPB * 32) gate_bwd_rows_kernel(const __nv_bfloat16* __restrict__ DA, const __nv_bfloat16* __restrict__ AO,
    const __nv_bfloat16* __restrict__ G, __nv_bfloat16* __restrict__ DO, __nv_bfloat16* __restrict__ DG, float* __restrict__ DD,
    long R, int L, long gstride, long dgstride, int nh) {
  constexpr int W = NCH * 128, NQ = NCH * 32;          // row width, quads per row
  __shared__ float hs[WPB][NQ];
  const long row = (long)blockIdx.x * WPB + threadIdx.x / 32;
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  if (row >= R) return;
  float4 da[NCH], ao[NCH], g[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const int c = (j * 32 + lane) * 4;
    da[j] = V4<__nv_bfloat16>::load(DA + row * W + c);
    ao[j] = V4<__nv_bfloat16>::load(AO + row * W + c);
    g[j] = V4<__nv_bfloat16>::load(G + row * gstride + c);
  }
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const int c = (j * 32 + lane) * 4;
    const float s0 = sg(g[j].x), s1 = sg(g[j].y), s2 = sg(g[j].z), s3 = sg(g[j].w);
    V4<__nv_bfloat16>::store(DO + row * W + c, make_float4(da[j].x * s0, da[j].y * s1, da[j].z * s2, da[j].w * s3));
    V4<__nv_bfloat16>::store(DG + row * dgstride + c, make_float4(da[j].x * ao[j].x * (1.f - s0), da[j].y * ao[j].y * (1.f - s1),
                                                                  da[j].z * ao[j].z * (1.f - s2), da[j].w * ao[j].w * (1.f - s3)));
    hs[w][j * 32 + lane] = (da[j].x * ao[j].x + da[j].y * ao[j].y) + (da[j].z * ao[j].z + da[j].w * ao[j].w);
  }
  __syncwarp();
  if (lane < nh) {                                     // head lane: its NQ / nh quads
    const int qph = NQ / nh;
    float t = 0.f;
    for (int k = 0; k < qph; ++k) t += hs[w][lane * qph + k];
    DD[((row / L) * nh + lane) * L + row % L] = t;
  }
}

// out[h][j][i] = in[h][i][j], 64 x 64 tiles through shared memory (bf16): 16-byte loads along the input rows, 16-byte stores
// along the output rows (each thread gathers 8 elements of a tile column; the row pitch of 72 keeps the gather conflict-free)
__global__ void __launch_bounds__(256) transpose_hll_kernel(const __nv_bfloat16* __restrict__ IN, __nv_bfloat16* __restrict__ OUT, int L) {
  __shared__ __align__(16) __nv_bfloat16 t[64][72];
  const long base = (long)blockIdx.z * L * L;
  const int i0 = blockIdx.y * 64, j0 = blockIdx.x * 64, c8 = threadIdx.x % 8, r = threadIdx.x / 8;
#pragma unroll
  for (int k = 0; k < 2; ++k)
    *reinterpret_cast<uint4*>(&t[r + 32 * k][c8 * 8]) = *reinterpret_cast<const uint4*>(IN + base + (long)(i0 + r + 32 * k) * L + j0 + c8 * 8);
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int j = r + 32 * k;                        // output row j0 + j takes tile column j, rows c8 * 8 .. + 7
    __align__(16) __nv_bfloat16 v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = t[c8 * 8 + e][j];
    *reinterpret_cast<uint4*>(OUT + base + (long)(j0 + j) * L + i0 + c8 * 8) = *reinterpret_cast<const uint4*>(v);
  }
}

// One warp per row of L keys, 8 keys (16 B of bf16) per lane step, CPL steps per lane; in place is allowed (a row is read whole before
// it is written).
template <int CPL>
__global__ void __launch_bounds__(256) softmax_rows_kernel(const __nv_bfloat16* __restrict__ BIAS, __nv_bfloat16* P,
    const bool* __restrict__ MASK, long R, int L) {
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32, nch = L / 8;
  if (row >= R) return;
  constexpr float NEG = -3.0e38f;
  float v[CPL][8];
  float mx = NEG;
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    const int ch = lane + 32 * c;
    if (ch < nch) {
      const float4 lo = V4<__nv_bfloat16>::load(BIAS + row * L + ch * 8), hi = V4<__nv_bfloat16>::load(BIAS + row * L + ch * 8 + 4);
      const float e[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      unsigned char mk[8];
      if (MASK) *reinterpret_cast<uint2*>(mk) = *reinterpret_cast<const uint2*>(MASK + ch * 8);
#pragma unroll
      for (int k = 0; k < 8; ++k) { v[c][k] = (MASK && !mk[k]) ? NEG : e[k]; mx = fmaxf(mx, v[c][k]); }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    if (lane + 32 * c < nch) {
#pragma unroll
      for (int k = 0; k < 8; ++k) { v[c][k] = __expf(v[c][k] - mx); sum += v[c][k]; }
    }
  }
  const float inv = 1.f / warp_sum(sum);
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    const int ch = lane + 32 * c;
    if (ch < nch) {
      V4<__nv_bfloat16>::store(P + row * L + ch * 8, make_float4(v[c][0] * inv, v[c][1] * inv, v[c][2] * inv, v[c][3] * inv));
      V4<__nv_bfloat16>::store(P + row * L + ch * 8 + 4, make_float4(v[c][4] * inv, v[c][5] * inv, v[c][6] * inv, v[c][7] * inv));
    }
  }
}

// ------------------------------------------------------------------------------------------------ host
void check_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.stride(1) == 1, name, ": a CUDA [rows, cols] view with unit column stride");
  TORCH_CHECK(t.size(1) >= cols, name, ": needs ", cols, " columns");
  TORCH_CHECK(t.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) == 0,
              name, ": rows must start on a 16-byte boundary");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, name, ": fp32 or bf16");
}

#define BOR_DISPATCH(T, NAME, ...)                                                                   \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    using NAME = __nv_bfloat16; return __VA_ARGS__();                                                \
  }()

template <typename C> auto ptr(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }
unsigned blocks(int64_t rows) { return (unsigned)((rows + WPB - 1) / WPB); }

void ln_rows(at::Tensor x, at::Tensor out, double eps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 384 && out.is_contiguous() && out.size(0) == M && out.size(1) == D, "ln_rows: width 384, contiguous out");
  check_rows(x, "ln_rows x", D);
  const at::cuda::CUDAGuard g(x.device());
  BOR_DISPATCH(x.scalar_type(), IT, [&] {
    BOR_DISPATCH(out.scalar_type(), OT, [&] {
      ln_rows_kernel<3, IT, OT><<<blocks(M), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<const IT>(x), ptr<OT>(out), M,
          x.stride(0), (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adaln_in_rows(at::Tensor in, at::Tensor x, at::Tensor ms, at::Tensor mb, at::Tensor out, int64_t T, double eps, bool ps) {
  const int64_t M = in.size(0), D = in.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat && x.is_contiguous() && out.is_contiguous(), "adaln_in_rows: width 768");
  check_rows(in, "in", D); check_rows(ms, "ms", D); check_rows(mb, "mb", D);
  TORCH_CHECK(ms.scalar_type() == mb.scalar_type(), "adaln_in_rows: ms / mb dtypes");
  const at::cuda::CUDAGuard g(in.device());
  BOR_DISPATCH(in.scalar_type(), IT, [&] {
    BOR_DISPATCH(ms.scalar_type(), GT, [&] {
      BOR_DISPATCH(out.scalar_type(), OT, [&] {
        adaln_in_rows_kernel<6, IT, GT, OT><<<blocks(M), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<const IT>(in),
            ptr<float>(x), ptr<const GT>(ms), ptr<const GT>(mb), ptr<OT>(out), M, (int)T, in.stride(0), ms.stride(0), mb.stride(0),
            (float)eps, ps);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void resgate_adaln_rows(at::Tensor x, at::Tensor y, at::Tensor gl, c10::optional<at::Tensor> ms, c10::optional<at::Tensor> mb,
                        at::Tensor out, int64_t T, double eps, bool ps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat && x.is_contiguous(), "resgate_adaln_rows: contiguous fp32 x of width 768");
  const bool has = ms.has_value();
  check_rows(y, "y", D); check_rows(gl, "gl", D);
  const at::Tensor& msv = has ? *ms : gl;
  const at::Tensor& mbv = has ? *mb : gl;
  if (has) { check_rows(msv, "ms", D); check_rows(mbv, "mb", D);
             TORCH_CHECK(out.is_contiguous() && msv.scalar_type() == gl.scalar_type() && mbv.scalar_type() == gl.scalar_type(),
                         "out contiguous, ms / mb like gl"); }
  const at::cuda::CUDAGuard g(x.device());
  BOR_DISPATCH(y.scalar_type(), YT, [&] {
    BOR_DISPATCH(gl.scalar_type(), GT, [&] {
      BOR_DISPATCH(out.scalar_type(), OT, [&] {
        resgate_adaln_rows_kernel<6, YT, GT, OT><<<blocks(M), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<float>(x),
            ptr<const YT>(y), ptr<const GT>(gl), ptr<const GT>(msv), ptr<const GT>(mbv), ptr<OT>(out), M, (int)T, y.stride(0),
            gl.stride(0), msv.stride(0), mbv.stride(0), (float)eps, has, ps);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void resgate_out_rows(at::Tensor x, at::Tensor y, at::Tensor gl, at::Tensor out, int64_t T, bool ps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D == 768 && x.scalar_type() == at::kFloat && x.is_contiguous() && out.is_contiguous(), "resgate_out_rows: width 768");
  check_rows(y, "y", D); check_rows(gl, "gl", D);
  const at::cuda::CUDAGuard g(x.device());
  BOR_DISPATCH(y.scalar_type(), YT, [&] {
    BOR_DISPATCH(gl.scalar_type(), GT, [&] {
      BOR_DISPATCH(out.scalar_type(), OT, [&] {
        resgate_out_rows_kernel<6, YT, GT, OT><<<blocks(M), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<const float>(x),
            ptr<const YT>(y), ptr<const GT>(gl), ptr<OT>(out), M, (int)T, y.stride(0), gl.stride(0), ps);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_rows(at::Tensor ab, at::Tensor out) {
  const int64_t M = ab.size(0), N = ab.size(1) / 2;
  TORCH_CHECK(ab.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && N % 8 == 0 && out.is_contiguous()
              && out.size(0) == M && out.size(1) == N, "swiglu_rows: bf16 [M, 2N] -> contiguous [M, N]");
  check_rows(ab, "ab", 2 * N);
  const at::cuda::CUDAGuard g(ab.device());
  const int64_t total8 = M * (N / 8);
  swiglu_rows_kernel<<<(unsigned)((total8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<const __nv_bfloat16>(ab),
      ptr<__nv_bfloat16>(out), total8, (int)(N / 8), ab.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_bwd_rows(at::Tensor da, at::Tensor ao, at::Tensor g, at::Tensor dout, at::Tensor dg, at::Tensor dd, int64_t L) {
  const int64_t M = da.size(0), W = da.size(1);
  TORCH_CHECK(W == 768 || W == 1024, "gate_bwd_rows: 768 or 1024 attention channels");
  for (auto* t : {&da, &ao, &dout})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->size(1) == W && t->size(0) == M, "gate_bwd_rows: bf16 [M, W]");
  check_rows(g, "g", W); check_rows(dg, "dg", W);
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && dg.scalar_type() == at::kBFloat16, "gate_bwd_rows: bf16 g / dg");
  const int64_t nh = dd.numel() / M;
  TORCH_CHECK(dd.scalar_type() == at::kFloat && dd.is_contiguous() && (nh == 12 || nh == 16 || nh == 24) && dd.numel() == M * nh
              && M % L == 0 && (W / 4) % nh == 0, "gate_bwd_rows: dd [A, 12, 16 or 24, L]");
  const at::cuda::CUDAGuard gd(da.device());
  auto k = W == 768 ? gate_bwd_rows_kernel<6> : gate_bwd_rows_kernel<8>;
  k<<<blocks(M), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<const __nv_bfloat16>(da),
      ptr<const __nv_bfloat16>(ao), ptr<const __nv_bfloat16>(g), ptr<__nv_bfloat16>(dout), ptr<__nv_bfloat16>(dg), ptr<float>(dd), M,
      (int)L, g.stride(0), dg.stride(0), (int)nh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transpose_hll(at::Tensor in, at::Tensor out) {
  TORCH_CHECK(in.scalar_type() == at::kBFloat16 && in.is_contiguous() && in.dim() == 3 && in.size(1) == in.size(2) && in.size(1) % 64 == 0,
              "transpose_hll: bf16 [H, L, L], L a multiple of 64");
  TORCH_CHECK(out.sizes() == in.sizes() && out.is_contiguous() && out.scalar_type() == at::kBFloat16, "transpose_hll: out like in");
  const int L = (int)in.size(1);
  const at::cuda::CUDAGuard g(in.device());
  transpose_hll_kernel<<<dim3(L / 64, L / 64, (unsigned)in.size(0)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      ptr<const __nv_bfloat16>(in), ptr<__nv_bfloat16>(out), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// softmax_rows that also writes P^T (training: the backward's dV = P^T dO wants the keys as rows). A block takes 32 query rows of
// one head (four per warp, the softmax as above), keeps them in shared memory and writes the [L keys x 32 queries] tile of P^T in
// 64-byte row pieces. The tile's row pitch is L + 2 elements: the column gathers of the transpose hit distinct banks.
template <int CPL>
__global__ void __launch_bounds__(256) softmax_t_kernel(const __nv_bfloat16* __restrict__ BIAS, __nv_bfloat16* P,
    __nv_bfloat16* __restrict__ PT, const bool* __restrict__ MASK, int L) {
  extern __shared__ __align__(16) uint8_t sm_raw[];
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(sm_raw);
  const int pitch = L + 2, lane = threadIdx.x % 32, warp = threadIdx.x / 32, nch = L / 8;
  const long r0 = (long)blockIdx.x * 32;                       // the block's first row of the [16 L, L] matrix
  const long h = r0 / L;
  const int i0 = (int)(r0 - h * L);
  constexpr float NEG = -3.0e38f;
  for (int rr = warp; rr < 32; rr += 8) {
    const long row = r0 + rr;
    float v[CPL][8];
    float mx = NEG;
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
      const int ch = lane + 32 * c;
      if (ch < nch) {
        const float4 lo = V4<__nv_bfloat16>::load(BIAS + row * L + ch * 8), hi = V4<__nv_bfloat16>::load(BIAS + row * L + ch * 8 + 4);
        const float e[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        unsigned char mk[8];
        if (MASK) *reinterpret_cast<uint2*>(mk) = *reinterpret_cast<const uint2*>(MASK + ch * 8);
#pragma unroll
        for (int k = 0; k < 8; ++k) { v[c][k] = (MASK && !mk[k]) ? NEG : e[k]; mx = fmaxf(mx, v[c][k]); }
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
      if (lane + 32 * c < nch) {
#pragma unroll
        for (int k = 0; k < 8; ++k) { v[c][k] = __expf(v[c][k] - mx); sum += v[c][k]; }
      }
    }
    const float inv = 1.f / warp_sum(sum);
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
      const int ch = lane + 32 * c;
      if (ch < nch) {
        __nv_bfloat162 b[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) b[k] = __floats2bfloat162_rn(v[c][2 * k] * inv, v[c][2 * k + 1] * inv);
        *reinterpret_cast<uint4*>(P + row * L + ch * 8) = *reinterpret_cast<const uint4*>(b);
        uint32_t* t = reinterpret_cast<uint32_t*>(tile + rr * pitch + ch * 8);     // 4-byte aligned (pitch even)
#pragma unroll
        for (int k = 0; k < 4; ++k) t[k] = *reinterpret_cast<const uint32_t*>(&b[k]);
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 4 * L; idx += blockDim.x) {
    const int j = idx / 4, c = idx % 4;
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = tile[(c * 8 + e) * pitch + j];
    *reinterpret_cast<uint4*>(PT + (h * L + j) * L + i0 + c * 8) = *reinterpret_cast<const uint4*>(o);
  }
}

void softmax_t(at::Tensor bias, at::Tensor p, at::Tensor pt, c10::optional<at::Tensor> mask) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.dim() == 2, "softmax_t: bf16 [16 L, L]");
  TORCH_CHECK(p.scalar_type() == at::kBFloat16 && p.is_contiguous() && p.sizes() == bias.sizes() && pt.sizes() == bias.sizes()
              && pt.scalar_type() == at::kBFloat16 && pt.is_contiguous() && pt.data_ptr() != bias.data_ptr(), "softmax_t: p, pt like bias");
  const int64_t R = bias.size(0), L = bias.size(1);
  TORCH_CHECK(L % 32 == 0 && L <= 1024 && R % L == 0, "softmax_t: L a multiple of 32, at most 1024; [H L, L]");
  const bool* mk = nullptr;
  if (mask.has_value()) {
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "softmax_t: bool mask [L]");
    mk = mask->data_ptr<bool>();
  }
  const at::cuda::CUDAGuard g(bias.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const size_t smem = 32 * (L + 2) * 2;
  static bool attr = [] {
    return cudaFuncSetAttribute(softmax_t_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1026 * 2) == cudaSuccess
        && cudaFuncSetAttribute(softmax_t_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1026 * 2) == cudaSuccess;
  }();
  TORCH_CHECK(attr, "softmax_t: shared memory attribute");
  const auto* b = ptr<const __nv_bfloat16>(bias);
  auto *o = ptr<__nv_bfloat16>(p), *ot = ptr<__nv_bfloat16>(pt);
  if (L <= 512) softmax_t_kernel<2><<<(unsigned)(R / 32), 256, smem, st>>>(b, o, ot, mk, (int)L);
  else softmax_t_kernel<4><<<(unsigned)(R / 32), 256, smem, st>>>(b, o, ot, mk, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_rows(at::Tensor bias, at::Tensor p, c10::optional<at::Tensor> mask) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.dim() == 2, "softmax_rows: bf16 [R, L]");
  TORCH_CHECK(p.scalar_type() == at::kBFloat16 && p.is_contiguous() && p.sizes() == bias.sizes(), "softmax_rows: p like bias");
  const int64_t R = bias.size(0), L = bias.size(1);
  TORCH_CHECK(L % 8 == 0 && L <= 2048, "softmax_rows: L a multiple of 8, at most 2048");
  const bool* mk = nullptr;
  if (mask.has_value()) {
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "softmax_rows: bool mask [L]");
    mk = mask->data_ptr<bool>();
  }
  const at::cuda::CUDAGuard g(bias.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const unsigned grid = (unsigned)((R + 7) / 8);
  const auto* b = ptr<const __nv_bfloat16>(bias);
  auto* o = ptr<__nv_bfloat16>(p);
  if (L <= 512) softmax_rows_kernel<2><<<grid, 256, 0, st>>>(b, o, mk, R, (int)L);
  else if (L <= 1024) softmax_rows_kernel<4><<<grid, 256, 0, st>>>(b, o, mk, R, (int)L);
  else softmax_rows_kernel<8><<<grid, 256, 0, st>>>(b, o, mk, R, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_t_cuda", &softmax_t);
  m.def("ln_rows_cuda", &ln_rows);
  m.def("adaln_in_rows_cuda", &adaln_in_rows);
  m.def("resgate_adaln_rows_cuda", &resgate_adaln_rows);
  m.def("resgate_out_rows_cuda", &resgate_out_rows);
  m.def("swiglu_rows_cuda", &swiglu_rows);
  m.def("softmax_rows_cuda", &softmax_rows);
  m.def("gate_bwd_rows_cuda", &gate_bwd_rows);
  m.def("transpose_hll_cuda", &transpose_hll);
}
