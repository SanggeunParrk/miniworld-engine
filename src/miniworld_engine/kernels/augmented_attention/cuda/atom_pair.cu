// Pair -> bias projection of the dense atom attention (d_pair 16, NH heads) with its LayerNorm folded in, forward and backward.
//
//   atom_pair_fwd   bias[h, r] = scale * sum_c Wf[h, c] * xhat[r, c]       r = i * L + j, xhat = LayerNorm(pair[r]) without affine
//                   (its weight is folded into Wf by the caller); a masked key j gets ``neg``.
//   atom_pair_bwd   dpair[r, c] = rstd (g - mean g - xhat mean(g xhat)),  g[c] = sum_h Wf[h, c] dbias[h, r]
//                   dWf[h, c]   = sum_r dbias[h, r] xhat[r, c]            (block partial rows; the caller adds them in order)
//
// One thread owns one pair row (32 bytes: two 16-byte loads), so the pair tensor is read once and the bias written once
// head-major (coalesced along j); the unfused route reads / writes it four times (LayerNorm, GEMM, permute, scale).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
using bf = __nv_bfloat16;
constexpr int C = 16;
constexpr int BT = 256;

__device__ __forceinline__ void load_row(const bf* __restrict__ z, float (&x)[C]) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(z)), b = __ldg(reinterpret_cast<const uint4*>(z) + 1);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x[2 * i] = __uint_as_float(w[i] << 16);
    x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void normalize(float (&x)[C], float eps, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) s += x[c];
  const float mean = s * (1.f / C);
  float v = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    x[c] -= mean;
    v += x[c] * x[c];
  }
  rstd = rsqrtf(v * (1.f / C) + eps);
#pragma unroll
  for (int c = 0; c < C; ++c) x[c] *= rstd;
}

template <int NH>
__global__ void __launch_bounds__(BT) fwd_k(const bf* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                            bf* __restrict__ OUT, long R, int L, float eps, float scale, float neg) {
  __shared__ float w[NH * C];
  if (threadIdx.x < NH * C) w[threadIdx.x] = W[threadIdx.x] * scale;
  __syncthreads();
  for (long r = (long)blockIdx.x * BT + threadIdx.x; r < R; r += (long)gridDim.x * BT) {
    float x[C], rstd;
    load_row(Z + r * C, x);
    normalize(x, eps, rstd);
    const bool drop = MASK != nullptr && !MASK[r % L];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      float d = 0.f;
#pragma unroll
      for (int c = 0; c < C; ++c) d += w[h * C + c] * x[c];
      OUT[(long)h * R + r] = __float2bfloat16_rn(drop ? neg : d);
    }
  }
}

template <int NH>
__global__ void __launch_bounds__(BT) bwd_k(const bf* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                            bf* __restrict__ DZ, float* __restrict__ ACC, long R, float eps) {
  __shared__ float w[NH * C];
  __shared__ float red[BT / 32][NH * C];
  if (threadIdx.x < NH * C) w[threadIdx.x] = W[threadIdx.x];
  __syncthreads();
  float part[NH * C];
#pragma unroll
  for (int i = 0; i < NH * C; ++i) part[i] = 0.f;
  for (long r = (long)blockIdx.x * BT + threadIdx.x; r < R; r += (long)gridDim.x * BT) {
    float x[C], rstd, db[NH], g[C];
    load_row(Z + r * C, x);
    normalize(x, eps, rstd);
#pragma unroll
    for (int h = 0; h < NH; ++h) db[h] = __ldg(DB + (long)h * R + r);
    float mg = 0.f, mgx = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float s = 0.f;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        s += w[h * C + c] * db[h];
        part[h * C + c] += db[h] * x[c];
      }
      g[c] = s;
      mg += s;
      mgx += s * x[c];
    }
    mg *= 1.f / C;
    mgx *= 1.f / C;
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a = rstd * (g[2 * i] - mg - x[2 * i] * mgx), b = rstd * (g[2 * i + 1] - mg - x[2 * i + 1] * mgx);
      const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
      o[i] = *reinterpret_cast<const uint32_t*>(&p);
    }
    uint4* dz = reinterpret_cast<uint4*>(DZ + r * C);
    dz[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dz[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
  // block partial of dWf: warp shuffle, then the warps in order
#pragma unroll
  for (int i = 0; i < NH * C; ++i) {
    float v = part[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][i] = v;
  }
  __syncthreads();
  if (threadIdx.x < NH * C) {
    float v = 0.f;
#pragma unroll
    for (int wp = 0; wp < BT / 32; ++wp) v += red[wp][threadIdx.x];
    ACC[(long)blockIdx.x * NH * C + threadIdx.x] = v;
  }
}

int nsm_of(const at::Tensor& t) {
  int n = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, t.get_device()));
  return n;
}
void chk(const at::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), n, ": contiguous CUDA tensor of the expected dtype");
}
}  // namespace

at::Tensor atom_pair_fwd(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double scale, double neg) {
  chk(z, at::kBFloat16, "pair");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), R = L * L;
  TORCH_CHECK(wf.size(1) == C && z.numel() == R * C, "atom_pair_fwd shapes");
  if (mask) {
    chk(*mask, at::kBool, "mask");
    TORCH_CHECK(mask->numel() == L, "mask [L]");
  }
  const at::cuda::CUDAGuard gd(z.device());
  auto out = at::empty({nh, L, L}, z.options());
  const unsigned grid = (unsigned)std::min<int64_t>((R + BT - 1) / BT, 8L * nsm_of(z));
  const bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* op = reinterpret_cast<bf*>(out.data_ptr());
  if (nh == 4) fwd_k<4><<<grid, BT, 0, s>>>(zp, wf.data_ptr<float>(), mp, op, R, (int)L, (float)eps, (float)scale, (float)neg);
  else if (nh == 8) fwd_k<8><<<grid, BT, 0, s>>>(zp, wf.data_ptr<float>(), mp, op, R, (int)L, (float)eps, (float)scale, (float)neg);
  else TORCH_CHECK(false, "atom_pair: 4 or 8 heads");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> atom_pair_bwd(at::Tensor z, at::Tensor db, at::Tensor wf, int64_t L, double eps) {
  chk(z, at::kBFloat16, "pair");
  chk(db, at::kFloat, "dbias");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), R = L * L;
  TORCH_CHECK(wf.size(1) == C && z.numel() == R * C && db.numel() == nh * R, "atom_pair_bwd shapes");
  const at::cuda::CUDAGuard gd(z.device());
  auto dz = at::empty_like(z);
  const unsigned grid = (unsigned)std::min<int64_t>((R + BT - 1) / BT, 4L * nsm_of(z));
  auto acc = at::empty({(int64_t)grid, nh * C}, z.options().dtype(at::kFloat));
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* dp = reinterpret_cast<bf*>(dz.data_ptr());
  if (nh == 4) bwd_k<4><<<grid, BT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), dp, acc.data_ptr<float>(), R, (float)eps);
  else if (nh == 8) bwd_k<8><<<grid, BT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), dp, acc.data_ptr<float>(), R, (float)eps);
  else TORCH_CHECK(false, "atom_pair: 4 or 8 heads");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("atom_pair_fwd", &atom_pair_fwd);
  m.def("atom_pair_bwd", &atom_pair_bwd);
}
