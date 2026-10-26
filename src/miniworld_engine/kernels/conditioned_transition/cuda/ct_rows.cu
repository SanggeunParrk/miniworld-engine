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
std::vector<at::Tensor> sigmul_bwd(at::Tensor dy, at::Tensor out, at::Tensor sp) {
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
  const unsigned grid = (unsigned)std::min<long>((chunks + threads - 1) / threads, 132L * 8);
  auto part = at::empty({(long)grid, dy.size(-1)}, dy.options().dtype(at::kFloat));
  const size_t shm = (size_t)threads / ndc * ndc * V * sizeof(float);
  if (f) sigmul_bwd_k<true><<<grid, threads, shm, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(dout), PM<float>(dsc), PM<float>(part), chunks, ndc);
  else sigmul_bwd_k<false><<<grid, threads, shm, S()>>>(P<bf>(dy), P<bf>(out), P<bf>(sp), PM<bf>(dout), PM<bf>(dsc), PM<float>(part), chunks, ndc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, dsc, part};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("sigmul_fwd", &sigmul_fwd);
  m.def("sigmul_bwd", &sigmul_bwd);
}
