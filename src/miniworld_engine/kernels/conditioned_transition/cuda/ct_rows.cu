// ct_rows.cu -- the elementwise passes between the GEMMs of the ConditionedTransition tail's training step (bf16, or fp32 on TF32 GEMMs), sm_90.
//
//   swiglu_fwd   h = silu(a) * b                         ab [M, 2 ND] = [a | b]  ->  h [M, ND]
//   swiglu_bwd   da = dh b silu'(a), db = dh silu(a)     dh [M, ND], ab  ->  dab [M, 2 ND] = [da | db]
//   sigmul_fwd   y = res + sigmoid(sp) * out             (the gated update rounds to the stream dtype before the add, as the unfused ops do)
//   sigmul_bwd   dout = dy g, dscale = dy out g (1 - g)  g = sigmoid(sp)
//
// Every pass is a vectorised grid-stride loop over 16-byte chunks (8 bf16 / 4 fp32): they are bandwidth-bound and read each operand once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "tpose.cuh"

namespace {
using bf = __nv_bfloat16;
template <bool F32> struct Vec;
template <> struct Vec<false> {
  static constexpr int N = 8;
  using T = bf;
  __device__ static void ld(const bf* p, float (&v)[8]) {
    const uint4 q = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[2 * j] = __uint_as_float(w[j] << 16); v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u); }
  }
  __device__ static void st(bf* p, const float (&v)[8]) {
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __nv_bfloat162 t = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
      w[j] = *reinterpret_cast<uint32_t*>(&t);
    }
    *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  __device__ static float rnd(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
};
template <> struct Vec<true> {
  static constexpr int N = 4;
  using T = float;
  __device__ static void ld(const float* p, float (&v)[4]) {
    const float4 q = *reinterpret_cast<const float4*>(p);
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
  }
  __device__ static void st(float* p, const float (&v)[4]) { *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]); }
  __device__ static float rnd(float x) { return x; }
};
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

template <bool F32>
__global__ void swiglu_fwd_k(const typename Vec<F32>::T* __restrict__ ab, typename Vec<F32>::T* __restrict__ h, long chunks, int ndc) {
  constexpr int V = Vec<F32>::N;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    const long m = i / ndc, j = i % ndc;                     // row, chunk of the row
    float a[V], b[V], o[V];
    Vec<F32>::ld(ab + m * 2 * ndc * V + j * V, a);
    Vec<F32>::ld(ab + m * 2 * ndc * V + ndc * V + j * V, b);
#pragma unroll
    for (int k = 0; k < V; ++k) o[k] = a[k] * sigm(a[k]) * b[k];
    Vec<F32>::st(h + i * V, o);
  }
}
template <bool F32>
__global__ void swiglu_bwd_k(const typename Vec<F32>::T* __restrict__ dh, const typename Vec<F32>::T* __restrict__ ab, typename Vec<F32>::T* __restrict__ dab,
                             long chunks, int ndc) {
  constexpr int V = Vec<F32>::N;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    const long m = i / ndc, j = i % ndc;
    float a[V], b[V], g[V], da[V], db[V];
    Vec<F32>::ld(ab + m * 2 * ndc * V + j * V, a);
    Vec<F32>::ld(ab + m * 2 * ndc * V + ndc * V + j * V, b);
    Vec<F32>::ld(dh + i * V, g);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float s = sigm(a[k]);
      da[k] = g[k] * b[k] * s * (1.f + a[k] * (1.f - s));
      db[k] = g[k] * a[k] * s;
    }
    Vec<F32>::st(dab + m * 2 * ndc * V + j * V, da);
    Vec<F32>::st(dab + m * 2 * ndc * V + ndc * V + j * V, db);
  }
}
template <bool F32, bool RES>
__global__ void sigmul_fwd_k(const typename Vec<F32>::T* __restrict__ out, const typename Vec<F32>::T* __restrict__ sp,
                             const typename Vec<F32>::T* __restrict__ res, typename Vec<F32>::T* __restrict__ y, long chunks) {
  constexpr int V = Vec<F32>::N;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    float o[V], s[V], r[V], v[V];
    Vec<F32>::ld(out + i * V, o);
    Vec<F32>::ld(sp + i * V, s);
    if (RES) Vec<F32>::ld(res + i * V, r);
#pragma unroll
    for (int k = 0; k < V; ++k) v[k] = RES ? Vec<F32>::rnd(sigm(s[k]) * o[k]) + r[k] : sigm(s[k]) * o[k];
    Vec<F32>::st(y + i * V, v);
  }
}
// dout = dy g, dscale = dy out g (1 - g); also the block partials of the column sums of dscale (the gate-bias gradient): the block is a whole number
// of rows of chunks, so a thread's column chunk never changes and it accumulates in registers; the row slots are summed in order through shared memory.
template <bool F32>
__global__ void sigmul_bwd_k(const typename Vec<F32>::T* __restrict__ dy, const typename Vec<F32>::T* __restrict__ out, const typename Vec<F32>::T* __restrict__ sp,
                             typename Vec<F32>::T* __restrict__ dout, typename Vec<F32>::T* __restrict__ dsc, float* __restrict__ part, long chunks, int ndc) {
  constexpr int V = Vec<F32>::N;
  extern __shared__ float red[];
  float acc[V];
#pragma unroll
  for (int k = 0; k < V; ++k) acc[k] = 0.f;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    float d[V], o[V], s[V], a[V], b[V];
    Vec<F32>::ld(dy + i * V, d);
    Vec<F32>::ld(out + i * V, o);
    Vec<F32>::ld(sp + i * V, s);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float g = sigm(s[k]);
      a[k] = d[k] * g;
      b[k] = Vec<F32>::rnd(d[k] * o[k] * g * (1.f - g));
      acc[k] += b[k];
    }
    Vec<F32>::st(dout + i * V, a);
    Vec<F32>::st(dsc + i * V, b);
  }
  const int slot = threadIdx.x / ndc, col = threadIdx.x % ndc, slots = blockDim.x / ndc;
#pragma unroll
  for (int k = 0; k < V; ++k) red[slot * ndc * V + col * V + k] = acc[k];
  __syncthreads();
  for (int c = threadIdx.x; c < ndc * V; c += blockDim.x) {
    float t = 0.f;
    for (int q = 0; q < slots; ++q) t += red[q * ndc * V + c];
    part[(long)blockIdx.x * ndc * V + c] = t;
  }
}


// ---- fp32 feature-major gradient passes (tpose.cuh): the TF32 weight gradients contract over the rows, so they want the gradient with the rows contiguous.
// Tiles of 32 rows x 128 features: loaded row-major (a warp reads 512 B of a row), transposed through shared memory, written as 128 B rows of [features, ldd].
// swiglu_bwd_t:  dab^T [2 ND, ldd]  (rows f: da, rows ND + f: db)
// sigmul_bwd_t:  dout^T and dscale^T [D, ldd] and the block partials of sum(dscale) over the rows (gate-bias gradient): per-lane accumulators over the
//                tiles a block takes, the warps summed in order through shared memory.
constexpr int TW = 8;   // warps a block
__global__ void __launch_bounds__(32 * TW) swiglu_bwd_t_k(const float* __restrict__ dh, const float* __restrict__ ab, float* __restrict__ dabT, long ldd, long R, int nd) {
  __shared__ float Ta[128 * tpose::LD], Tb[128 * tpose::LD];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
    for (int f0 = 0; f0 < nd; f0 += 128) {
      for (int q = 0; q < tpose::TR / TW; ++q) {
        const int rl = q * TW + wp;
        if (rl < rows) {
          const long row = m0 + rl;
          float a[4], b[4], g[4];
          Vec<true>::ld(ab + row * 2 * nd + f0 + 4 * lane, a);
          Vec<true>::ld(ab + row * 2 * nd + nd + f0 + 4 * lane, b);
          Vec<true>::ld(dh + row * nd + f0 + 4 * lane, g);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float s = sigm(a[k]);
            tpose::put(Ta, 4 * lane + k, rl, g[k] * b[k] * s * (1.f + a[k] * (1.f - s)));
            tpose::put(Tb, 4 * lane + k, rl, g[k] * a[k] * s);
          }
        }
      }
      __syncthreads();
      tpose::flush(Ta, dabT, ldd, f0, 128, (int)m0, rows, wp, TW, lane);
      tpose::flush(Tb, dabT, ldd, nd + f0, 128, (int)m0, rows, wp, TW, lane);
      __syncthreads();
    }
  }
}
template <int NCH>
__global__ void __launch_bounds__(32 * TW) sigmul_bwd_t_k(const float* __restrict__ dy, const float* __restrict__ out, const float* __restrict__ sp, float* __restrict__ doutT,
                                                         float* __restrict__ dscT, long ldd, float* __restrict__ part, long R) {
  constexpr int D = NCH * 128;
  extern __shared__ float sm[];
  float* Ta = sm;
  float* Tb = sm + 128 * tpose::LD;
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float acc[NCH][4];
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[c][k] = 0.f;
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      for (int q = 0; q < tpose::TR / TW; ++q) {
        const int rl = q * TW + wp;
        if (rl < rows) {
          const long row = m0 + rl;
          float d[4], o[4], s[4];
          Vec<true>::ld(dy + row * D + 128 * c + 4 * lane, d);
          Vec<true>::ld(out + row * D + 128 * c + 4 * lane, o);
          Vec<true>::ld(sp + row * D + 128 * c + 4 * lane, s);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float g = sigm(s[k]);
            const float b = d[k] * o[k] * g * (1.f - g);
            acc[c][k] += b;
            tpose::put(Ta, 4 * lane + k, rl, d[k] * g);
            tpose::put(Tb, 4 * lane + k, rl, b);
          }
        }
      }
      __syncthreads();
      tpose::flush(Ta, doutT, ldd, 128 * c, 128, (int)m0, rows, wp, TW, lane);
      tpose::flush(Tb, dscT, ldd, 128 * c, 128, (int)m0, rows, wp, TW, lane);
      __syncthreads();
    }
  }
  float (*red)[D] = reinterpret_cast<float (*)[D]>(sm);
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[wp][128 * c + 4 * lane + k] = acc[c][k];
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 32 * TW) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < TW; ++w) t += red[w][c];
    part[(long)blockIdx.x * D + c] = t;
  }
}

// sigmul_bwd_x: sigmul_bwd (row-major dout, dscale, the block partials) plus the K-major copy of dout, [D, ldd]: the left operand of dWs = dout^T h that cuBLAS runs at
// ~290 TFLOP/s (with dout read MN-major it is ~185). One more write of dout (56 MB at 18432 x 768) instead of a separate transposing kernel (read + write).
template <int NCH>
__global__ void __launch_bounds__(32 * TW) sigmul_bwd_x_k(const float* __restrict__ dy, const float* __restrict__ out, const float* __restrict__ sp, float* __restrict__ dout,
                                                         float* __restrict__ dsc, float* __restrict__ doutT, long ldd, float* __restrict__ part, long R) {
  constexpr int D = NCH * 128;
  extern __shared__ float sm[];
  float* Ta = sm;
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float acc[NCH][4];
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[c][k] = 0.f;
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      for (int q = 0; q < tpose::TR / TW; ++q) {
        const int rl = q * TW + wp;
        if (rl < rows) {
          const long row = m0 + rl;
          float d[4], o[4], s[4], a[4], b[4];
          Vec<true>::ld(dy + row * D + 128 * c + 4 * lane, d);
          Vec<true>::ld(out + row * D + 128 * c + 4 * lane, o);
          Vec<true>::ld(sp + row * D + 128 * c + 4 * lane, s);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float g = sigm(s[k]);
            a[k] = d[k] * g;
            b[k] = d[k] * o[k] * g * (1.f - g);
            acc[c][k] += b[k];
            tpose::put(Ta, 4 * lane + k, rl, a[k]);
          }
          Vec<true>::st(dout + row * D + 128 * c + 4 * lane, a);
          Vec<true>::st(dsc + row * D + 128 * c + 4 * lane, b);
        }
      }
      __syncthreads();
      tpose::flush(Ta, doutT, ldd, 128 * c, 128, (int)m0, rows, wp, TW, lane);
      __syncthreads();
    }
  }
  float (*red)[D] = reinterpret_cast<float (*)[D]>(sm);
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[wp][128 * c + 4 * lane + k] = acc[c][k];
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 32 * TW) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < TW; ++w) t += red[w][c];
    part[(long)blockIdx.x * D + c] = t;
  }
}

// fp32 -> bf16 casts of up to eight tensors in ONE launch (the weights of a training step; the PyTorch casts are 4 - 6 launches of 2 - 10 us each).
// Job j copies n[j] elements of src[j] to dst[j]; a thread block covers a contiguous slice of the concatenated element range.
struct CastJobs {
  const float* src[8];
  __nv_bfloat16* dst[8];
  long end[8];  // prefix sums of the element counts
  int n;
};
__global__ void cast_pack_k(CastJobs jobs) {   // four elements a thread (all sizes are multiples of four)
  const long quads = jobs.end[jobs.n - 1] / 4;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < quads; i += (long)gridDim.x * blockDim.x) {
    int j = 0;
    while (4 * i >= jobs.end[j]) ++j;
    const long o = 4 * i - (j ? jobs.end[j - 1] : 0);
    const float4 v = *reinterpret_cast<const float4*>(jobs.src[j] + o);
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    uint2 u;
    u.x = *reinterpret_cast<const unsigned*>(&a);
    u.y = *reinterpret_cast<const unsigned*>(&b);
    *reinterpret_cast<uint2*>(jobs.dst[j] + o) = u;
  }
}

cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
unsigned grid_of(long chunks) { return (unsigned)std::min<long>((chunks + 255) / 256, 132L * 16); }
bool is_f32(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat), "contiguous bf16 / fp32 CUDA tensor");
  return t.scalar_type() == at::kFloat;
}
template <typename T> const T* P(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> T* PM(at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
}  // namespace

at::Tensor swiglu_fwd(at::Tensor ab) {
  const bool f = is_f32(ab);
  const int V = f ? 4 : 8;
  const long M = ab.size(0), nd = ab.size(1) / 2;
  TORCH_CHECK(nd % V == 0, "ND % 8");
  const at::cuda::CUDAGuard g(ab.device());
  auto h = at::empty({M, nd}, ab.options());
  const long chunks = M * nd / V;
  if (f) swiglu_fwd_k<true><<<grid_of(chunks), 256, 0, S()>>>(P<float>(ab), PM<float>(h), chunks, (int)(nd / V));
  else swiglu_fwd_k<false><<<grid_of(chunks), 256, 0, S()>>>(P<bf>(ab), PM<bf>(h), chunks, (int)(nd / V));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
at::Tensor swiglu_bwd(at::Tensor dh, at::Tensor ab) {
  const bool f = is_f32(ab);
  is_f32(dh);
  const int V = f ? 4 : 8;
  const long M = ab.size(0), nd = ab.size(1) / 2;
  TORCH_CHECK(dh.size(0) == M && dh.size(1) == nd && nd % V == 0 && dh.scalar_type() == ab.scalar_type(), "shapes");
  const at::cuda::CUDAGuard g(ab.device());
  auto dab = at::empty_like(ab);
  const long chunks = M * nd / V;
  if (f) swiglu_bwd_k<true><<<grid_of(chunks), 256, 0, S()>>>(P<float>(dh), P<float>(ab), PM<float>(dab), chunks, (int)(nd / V));
  else swiglu_bwd_k<false><<<grid_of(chunks), 256, 0, S()>>>(P<bf>(dh), P<bf>(ab), PM<bf>(dab), chunks, (int)(nd / V));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dab;
}
at::Tensor sigmul_fwd(at::Tensor out, at::Tensor sp, c10::optional<at::Tensor> res) {
  const bool f = is_f32(out);
  is_f32(sp);
  const int V = f ? 4 : 8;
  TORCH_CHECK(out.numel() % V == 0 && sp.sizes() == out.sizes() && sp.scalar_type() == out.scalar_type(), "shapes");
  if (res) TORCH_CHECK(res->sizes() == out.sizes() && res->scalar_type() == out.scalar_type() && res->is_contiguous(), "residual");
  const at::cuda::CUDAGuard g(out.device());
  auto y = at::empty_like(out);
  const long chunks = out.numel() / V;
  const bool r = res.has_value();
#define SF(F, T, R) sigmul_fwd_k<F, R><<<grid_of(chunks), 256, 0, S()>>>(P<T>(out), P<T>(sp), r ? P<T>(*res) : nullptr, PM<T>(y), chunks)
  if (f) { if (r) SF(true, float, true); else SF(true, float, false); }
  else { if (r) SF(false, bf, true); else SF(false, bf, false); }
#undef SF
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
std::vector<at::Tensor> sigmul_bwd(at::Tensor dy, at::Tensor out, at::Tensor sp, int64_t grid_blocks) {
  const bool f = is_f32(dy);
  is_f32(out);
  is_f32(sp);
  const int V = f ? 4 : 8;
  TORCH_CHECK(dy.numel() % V == 0 && out.sizes() == dy.sizes() && sp.sizes() == dy.sizes(), "shapes");
  const at::cuda::CUDAGuard g(dy.device());
  auto dout = at::empty_like(dy), dsc = at::empty_like(dy);
  const long chunks = dy.numel() / V;
  const int ndc = (int)(dy.size(-1) / V);
  const int threads = ndc * std::max(1, 128 / ndc);
  TORCH_CHECK(threads <= 1024, "width");
  const unsigned grid = (unsigned)grid_blocks;
  auto part = at::empty({(long)grid, dy.size(-1)}, dy.options().dtype(at::kFloat));
  const size_t shm = (size_t)threads / ndc * ndc * V * sizeof(float);
  if (f) sigmul_bwd_k<true><<<grid, threads, shm, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(dout), PM<float>(dsc), PM<float>(part), chunks, ndc);
  else sigmul_bwd_k<false><<<grid, threads, shm, S()>>>(P<bf>(dy), P<bf>(out), P<bf>(sp), PM<bf>(dout), PM<bf>(dsc), PM<float>(part), chunks, ndc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, dsc, part};
}

// fp32 feature-major backward passes: [features, ldd] outputs, ldd = rows rounded up to 32 (a GEMM operand is its [:, :rows] view)
at::Tensor swiglu_bwd_t(at::Tensor dh, at::Tensor ab) {
  TORCH_CHECK(is_f32(ab) && is_f32(dh), "fp32 only");
  const long M = ab.size(0), nd = ab.size(1) / 2;
  TORCH_CHECK(dh.size(0) == M && dh.size(1) == nd && nd % 128 == 0, "shapes: ND % 128");
  const at::cuda::CUDAGuard g(ab.device());
  const long ldd = (M + 31) / 32 * 32;
  auto dabT = at::empty({2 * nd, ldd}, ab.options());
  const unsigned grid = (unsigned)std::min<long>((M + 31) / 32, 132L * 6);
  swiglu_bwd_t_k<<<grid, 32 * TW, 0, S()>>>(P<float>(dh), P<float>(ab), PM<float>(dabT), ldd, M, (int)nd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dabT;
}
std::vector<at::Tensor> sigmul_bwd_t(at::Tensor dy, at::Tensor out, at::Tensor sp, int64_t grid_blocks) {
  TORCH_CHECK(is_f32(dy) && is_f32(out) && is_f32(sp), "fp32 only");
  const long D = dy.size(-1), M = dy.numel() / D;
  TORCH_CHECK(D % 128 == 0 && D <= 1024 && out.sizes() == dy.sizes() && sp.sizes() == dy.sizes(), "shapes: D % 128, <= 1024");
  const at::cuda::CUDAGuard g(dy.device());
  const long ldd = (M + 31) / 32 * 32;
  auto doutT = at::empty({D, ldd}, dy.options()), dscT = at::empty({D, ldd}, dy.options());
  const unsigned grid = (unsigned)std::min<long>(grid_blocks, (M + 31) / 32);
  auto part = at::empty({(long)grid, D}, dy.options());
  const size_t shm = std::max<size_t>(2 * 128 * tpose::LD * sizeof(float), (size_t)TW * D * sizeof(float));
#define SBT(NCH) sigmul_bwd_t_k<NCH><<<grid, 32 * TW, shm, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(doutT), PM<float>(dscT), ldd, PM<float>(part), M)
  switch (D / 128) {
    case 1: SBT(1); break; case 2: SBT(2); break; case 3: SBT(3); break; case 4: SBT(4); break;
    case 5: SBT(5); break; case 6: SBT(6); break; case 7: SBT(7); break; default: SBT(8); break;
  }
#undef SBT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {doutT, dscT, part};
}

// (dout, dsc, part, doutT [D, ldd]); `grid_blocks` as sigmul_bwd_t (the partials have one row per block)
std::vector<at::Tensor> sigmul_bwd_x(at::Tensor dy, at::Tensor out, at::Tensor sp, int64_t grid_blocks) {
  TORCH_CHECK(is_f32(dy) && is_f32(out) && is_f32(sp), "fp32 only");
  const long D = dy.size(-1), M = dy.numel() / D;
  TORCH_CHECK(D % 128 == 0 && D <= 1024 && out.sizes() == dy.sizes() && sp.sizes() == dy.sizes(), "shapes: D % 128, <= 1024");
  const at::cuda::CUDAGuard g(dy.device());
  const long ldd = (M + 31) / 32 * 32;
  auto dout = at::empty_like(dy), dsc = at::empty_like(dy), doutT = at::empty({D, ldd}, dy.options());
  const unsigned grid = (unsigned)std::min<long>(grid_blocks, (M + 31) / 32);
  auto part = at::empty({(long)grid, D}, dy.options());
  const size_t shm = std::max<size_t>(128 * tpose::LD * sizeof(float), (size_t)TW * D * sizeof(float));
#define SBX(NCH) sigmul_bwd_x_k<NCH><<<grid, 32 * TW, shm, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(dout), PM<float>(dsc), PM<float>(doutT), ldd, PM<float>(part), M)
  switch (D / 128) {
    case 1: SBX(1); break; case 2: SBX(2); break; case 3: SBX(3); break; case 4: SBX(4); break;
    case 5: SBX(5); break; case 6: SBX(6); break; case 7: SBX(7); break; default: SBX(8); break;
  }
#undef SBX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, dsc, part, doutT};
}

void cast_pack(std::vector<at::Tensor> srcs, std::vector<at::Tensor> dsts) {
  TORCH_CHECK(srcs.size() == dsts.size() && !srcs.empty() && srcs.size() <= 8, "1 .. 8 casts");
  CastJobs j{};
  long total = 0;
  for (size_t k = 0; k < srcs.size(); ++k) {
    TORCH_CHECK(srcs[k].is_cuda() && srcs[k].scalar_type() == at::kFloat && srcs[k].is_contiguous() && dsts[k].scalar_type() == at::kBFloat16 &&
                dsts[k].is_contiguous() && dsts[k].numel() == srcs[k].numel(), "cast_pack: contiguous fp32 -> bf16 of equal size"); TORCH_CHECK(srcs[k].numel() % 4 == 0, "cast_pack: sizes multiples of 4");
    j.src[k] = reinterpret_cast<const float*>(srcs[k].data_ptr());
    j.dst[k] = reinterpret_cast<__nv_bfloat16*>(dsts[k].data_ptr());
    total += srcs[k].numel();
    j.end[k] = total;
  }
  j.n = (int)srcs.size();
  const at::cuda::CUDAGuard g(srcs[0].device());
  cast_pack_k<<<(unsigned)((total / 4 + 255) / 256), 256, 0, S()>>>(j);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cast_pack", &cast_pack);
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("sigmul_fwd", &sigmul_fwd);
  m.def("sigmul_bwd", &sigmul_bwd);
  m.def("swiglu_bwd_t", &swiglu_bwd_t);
  m.def("sigmul_bwd_t", &sigmul_bwd_t);
  m.def("sigmul_bwd_x", &sigmul_bwd_x);
}
