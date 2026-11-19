// sampler_ops_f32_sm80.cu -- the encoder pooling (PoolAtoms) on fp32 atom features: A100 (sm_80) and RTX 5090 (sm_120a). The same statement as the bf16 kernels of
// sampler_ops.cu:  out[s, t] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)  (fp32 sums in atom order), and its gradient wrt x
// (a gather over the slots sorted by row, as pool_atoms_bwd_k; ``pool_atoms_prep`` of the bf16 extension makes the slots' key and weight).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <algorithm>
#include <climits>
#include <vector>

namespace {

__device__ __forceinline__ float ldf(const void* p, int code, long i) {
  switch (code) {
    case 0: return ((const float*)p)[i];
    case 1: return ((const bool*)p)[i] ? 1.f : 0.f;
    case 2: return (float)((const int*)p)[i];
    case 3: return (float)((const long long*)p)[i];
    default: return __uint_as_float((unsigned)((const unsigned short*)p)[i] << 16);  // bf16
  }
}

int code_of(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return 0;
    case at::kBool: return 1;
    case at::kInt: return 2;
    case at::kLong: return 3;
    case at::kBFloat16: return 4;
    default: TORCH_CHECK(false, "pool_atoms_f32: unsupported index / mask dtype ", t.scalar_type());
  }
}

constexpr int NO_ROW = INT_MIN;

// grid (T, S), block ceil(C / 4 / 32) * 32 threads: a thread owns 4 consecutive channels. The atoms' (mask, row) are read one per lane and handed to the warp by shuffles.
__global__ void pool_f32_k(const float* __restrict__ x, const void* idx, int ic, const void* gm, int gc, const void* am, int mc, float* out, int atoms, int tokens,
                           int rows, int channels) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x * 4, lane = threadIdx.x & 31;
  const bool live = c < channels;
  const float* xs = x + (size_t)s * rows * channels + c;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, den = 0.f;
  for (int g0 = 0; g0 < atoms; g0 += 32) {
    float ml = 0.f;
    int rl = NO_ROW;
    if (g0 + lane < atoms) {
      const long k = (long)t * atoms + g0 + lane;
      ml = ldf(am, mc, k);
      const float gl = ldf(gm, gc, k);
      const int ri = (int)(long)ldf(idx, ic, k);
      if (ml != 0.f && gl != 0.f) rl = ri;
    }
    const int nvalid = min(32, atoms - g0);
    for (int u0 = 0; u0 < nvalid; u0 += 8) {
      float4 raw[8];
      float mu[8];
      bool use[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        mu[u] = __shfl_sync(0xffffffffu, ml, (u0 + u) & 31);
        const int r = __shfl_sync(0xffffffffu, rl, (u0 + u) & 31);
        use[u] = (u0 + u < nvalid) && r != NO_ROW;
        raw[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (live && use[u]) raw[u] = *reinterpret_cast<const float4*>(xs + (size_t)r * channels);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        if (u0 + u >= nvalid) continue;
        den += mu[u];
        if (use[u]) {
          acc[0] += mu[u] * fmaxf(raw[u].x, 0.f); acc[1] += mu[u] * fmaxf(raw[u].y, 0.f);
          acc[2] += mu[u] * fmaxf(raw[u].z, 0.f); acc[3] += mu[u] * fmaxf(raw[u].w, 0.f);
        }
      }
    }
  }
  if (!live) return;
  const float inv = fmaxf(den, 1e-10f);
  *reinterpret_cast<float4*>(out + ((size_t)s * tokens + t) * channels + c) = make_float4(acc[0] / inv, acc[1] / inv, acc[2] / inv, acc[3] / inv);
}

constexpr int BWD_ROWS = 8, BWD_SAMPLES = 8;

// the SM count by the attribute query (torch's cudaDeviceProp can be laid out by another CUDA version than the toolkit nvcc compiles against)
int sm_count() {
  int dev = 0, n = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  return n;
}

__device__ __forceinline__ int warp_lower_bound(const int* __restrict__ v, int n, int x, int lane) {
  int lo = 0, hi = n;
  while (hi - lo > 32) {
    const int step = (hi - lo + 31) / 32, pos = lo + lane * step;
    const int cnt = __popc(__ballot_sync(0xffffffffu, pos < hi && __ldg(v + pos) < x));
    if (cnt == 0) return lo;
    const int nlo = lo + (cnt - 1) * step + 1;
    hi = min(hi, lo + cnt * step);
    lo = nlo;
  }
  const int pos = lo + lane;
  return lo + __popc(__ballot_sync(0xffffffffu, pos < hi && __ldg(v + pos) < x));
}

__device__ __forceinline__ int warp_run_end(const int* __restrict__ v, int n, int beg, int x, int lane) {
  for (int k = beg;; k += 32) {
    const unsigned eq = __ballot_sync(0xffffffffu, k + lane < n && __ldg(v + k + lane) == x);
    if (eq != 0xffffffffu) return k + __popc(eq);
  }
}

__global__ void __launch_bounds__(32 * BWD_ROWS) pool_f32_bwd_k(const float* __restrict__ gout, const float* __restrict__ x, const int* __restrict__ skey,
                                                               const long long* __restrict__ perm, const float* __restrict__ w, float* __restrict__ gx,
                                                               int atoms, int tokens, int rows, int channels, int samples, int slots, int spb) {
  const int r = blockIdx.x * BWD_ROWS + threadIdx.y;
  if (r >= rows) return;
  const int beg = warp_lower_bound(skey, slots, r, threadIdx.x), end = warp_run_end(skey, slots, beg, r, threadIdx.x);
  const int s0 = blockIdx.y * spb, s1 = min(samples, s0 + spb);
  for (int s = s0; s < s1; ++s) {
    const float* gs = gout + (size_t)s * tokens * channels;
    for (int c = threadIdx.x * 4; c < channels; c += 128) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k = beg; k < end; ++k) {
        const long long slot = __ldg(perm + k);
        const float wk = __ldg(w + slot);
        const float4 g = *reinterpret_cast<const float4*>(gs + (size_t)(slot / atoms) * channels + c);
        acc[0] += wk * g.x; acc[1] += wk * g.y; acc[2] += wk * g.z; acc[3] += wk * g.w;
      }
      const size_t o = ((size_t)s * rows + r) * channels + c;
      const float4 xr = *reinterpret_cast<const float4*>(x + o);
      *reinterpret_cast<float4*>(gx + o) = make_float4(xr.x > 0.f ? acc[0] : 0.f, xr.y > 0.f ? acc[1] : 0.f, xr.z > 0.f ? acc[2] : 0.f, xr.w > 0.f ? acc[3] : 0.f);
    }
  }
}

}  // namespace

at::Tensor pool_atoms_f32(at::Tensor x, at::Tensor idx, at::Tensor gather_mask, at::Tensor atom_mask) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 3, "x: contiguous fp32 cuda [S, Q, C]");
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && gather_mask.is_contiguous() && atom_mask.is_contiguous() && idx.dim() == 2 &&
              gather_mask.sizes() == idx.sizes() && atom_mask.sizes() == idx.sizes(), "idx / masks: contiguous [T, A]");
  const int64_t S = x.size(0), R = x.size(1), C = x.size(2), T = idx.size(0), A = idx.size(1);
  TORCH_CHECK(C % 4 == 0 && C / 4 <= 1024, "channels % 4 == 0 and <= 4096");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto out = at::empty({S, T, C}, x.options());
  const dim3 grid((unsigned)T, (unsigned)S);
  const unsigned threads = (unsigned)(((C / 4) + 31) / 32 * 32);
  pool_f32_k<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), idx.data_ptr(), code_of(idx), gather_mask.data_ptr(), code_of(gather_mask),
                                                                     atom_mask.data_ptr(), code_of(atom_mask), out.data_ptr<float>(), (int)A, (int)T, (int)R, (int)C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor pool_atoms_f32_bwd(at::Tensor gout, at::Tensor x, at::Tensor skey, at::Tensor perm, at::Tensor w, int64_t atoms) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 3, "x: contiguous fp32 cuda [S, Q, C]");
  TORCH_CHECK(gout.is_contiguous() && gout.scalar_type() == at::kFloat && gout.dim() == 3 && gout.size(0) == x.size(0) && gout.size(2) == x.size(2),
              "gout: contiguous fp32 [S, T, C]");
  TORCH_CHECK(skey.is_contiguous() && skey.scalar_type() == at::kInt && perm.is_contiguous() && perm.scalar_type() == at::kLong && w.is_contiguous() &&
              w.scalar_type() == at::kFloat && perm.numel() == skey.numel() && w.numel() == skey.numel() && skey.numel() == gout.size(1) * atoms,
              "skey int32 / perm int64 / w fp32 [T A]");
  const int64_t S = x.size(0), R = x.size(1), C = x.size(2), T = gout.size(1);
  TORCH_CHECK(C % 4 == 0, "channels % 4 == 0");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto gx = at::empty_like(x);
  if (gx.numel() == 0) return gx;
  const int64_t row_blocks = (R + BWD_ROWS - 1) / BWD_ROWS, target = 32LL * sm_count();
  const int64_t want = std::max<int64_t>(1, (target + row_blocks - 1) / row_blocks);
  const int spb = (int)std::min<int64_t>(BWD_SAMPLES, std::max<int64_t>(1, (S + want - 1) / want));
  const dim3 grid((unsigned)row_blocks, (unsigned)((S + spb - 1) / spb)), block(32, BWD_ROWS);
  pool_f32_bwd_k<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(gout.data_ptr<float>(), x.data_ptr<float>(), skey.data_ptr<int>(),
                                                                       reinterpret_cast<const long long*>(perm.data_ptr<int64_t>()), w.data_ptr<float>(),
                                                                       gx.data_ptr<float>(), (int)atoms, (int)T, (int)R, (int)C, (int)S, (int)skey.numel(), spb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pool_atoms_f32", &pool_atoms_f32);
  m.def("pool_atoms_f32_bwd", &pool_atoms_f32_bwd);
}
