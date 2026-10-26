// Pair -> bias projection with its LayerNorm folded in, for any d_pair in {64, 128, 256} and NH heads, forward and backward.
//
//   pair_rows_fwd   bias[h, r] = scale * sum_c Wf[h, c] * xhat[r, c]       r = i * L + j, xhat = LayerNorm(pair[r]) without affine
//                   (its weight is folded into Wf by the caller); a masked key j gets ``neg``. bf16 head-major.
//   pair_rows_bwd   dpair[r, c] = rstd (g - mean g - xhat mean(g xhat)),  g[c] = sum_h Wf[h, c] dbias[h, r]
//                   dWf[h, c]   = sum_r dbias[h, r] xhat[r, c]            (block partial rows; the caller adds them in order)
//
// A warp owns a row at a time (lane l: channels DP / 32 l ..), the pair tensor is read once with 8 / 16-byte loads, the heads' dot
// products are warp butterfly sums; 32 rows' results are collected one per lane so the bias is written coalesced along j (64 B per
// head and 32 rows). CUDA cores only: the step is bound by the pair read (one pass of L * L * DP * 2 bytes), the arithmetic
// (NH * DP FMA a row) is a fifth of the memory time. The unfused route reads / writes the pair tensor four times (LayerNorm, GEMM,
// permute, scale).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
using bf = __nv_bfloat16;
constexpr int BT = 256;
constexpr int WPB = BT / 32;

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int DPL>
__device__ __forceinline__ void load_row(const bf* __restrict__ z, float (&x)[DPL]) {
  static_assert(DPL == 2 || DPL == 4 || DPL == 8, "channels per lane");
  if constexpr (DPL == 8) {
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(z));
    const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[2 * i] = __uint_as_float(w[i] << 16);
      x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  } else if constexpr (DPL == 4) {
    const uint2 a = __ldg(reinterpret_cast<const uint2*>(z));
    const uint32_t w[2] = {a.x, a.y};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      x[2 * i] = __uint_as_float(w[i] << 16);
      x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  } else {
    const uint32_t w = __ldg(reinterpret_cast<const unsigned*>(z));
    x[0] = __uint_as_float(w << 16);
    x[1] = __uint_as_float(w & 0xffff0000u);
  }
}
template <int DPL>
__device__ __forceinline__ void store_row(bf* __restrict__ z, const float (&x)[DPL]) {
  uint32_t o[DPL / 2];
#pragma unroll
  for (int i = 0; i < DPL / 2; ++i) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(x[2 * i], x[2 * i + 1]);
    o[i] = *reinterpret_cast<const uint32_t*>(&p);
  }
  if constexpr (DPL == 8) *reinterpret_cast<uint4*>(z) = make_uint4(o[0], o[1], o[2], o[3]);
  else if constexpr (DPL == 4) *reinterpret_cast<uint2*>(z) = make_uint2(o[0], o[1]);
  else *reinterpret_cast<unsigned*>(z) = o[0];
}
template <int DP, int DPL>
__device__ __forceinline__ void normalize(float (&x)[DPL], float eps, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) s += x[i];
  const float mean = wsum(s) * (1.f / DP);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) {
    x[i] -= mean;
    v += x[i] * x[i];
  }
  rstd = rsqrtf(wsum(v) * (1.f / DP) + eps);
#pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= rstd;
}

template <int DP, int NH>
__global__ void __launch_bounds__(BT) fwd_k(const bf* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                            bf* __restrict__ OUT, long R, int L, float eps, float scale, float neg) {
  constexpr int DPL = DP / 32;
  const int lane = threadIdx.x & 31;
  float w[NH][DPL];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < DPL; ++i) w[h][i] = W[h * DP + lane * DPL + i] * scale;
  const long warp = ((long)blockIdx.x * BT + threadIdx.x) >> 5, nwarps = (long)gridDim.x * WPB;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    float res[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) res[h] = 0.f;
    const int n = (int)min(32L, R - base);
    for (int r = 0; r < n; ++r) {
      float x[DPL], rstd;
      load_row<DPL>(Z + (base + r) * DP + lane * DPL, x);
      normalize<DP, DPL>(x, eps, rstd);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) s += w[h][i] * x[i];
        s = wsum(s);
        if (lane == r) res[h] = s;
      }
    }
    if (lane < n) {
      const long row = base + lane;
      const bool drop = MASK != nullptr && !MASK[row % L];
#pragma unroll
      for (int h = 0; h < NH; ++h) OUT[(long)h * R + row] = __float2bfloat16_rn(drop ? neg : res[h]);
    }
  }
}

template <int DP, int NH>
__global__ void __launch_bounds__(BT) bwd_k(const bf* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                            bf* __restrict__ DZ, float* __restrict__ ACC, long R, float eps) {
  constexpr int DPL = DP / 32;
  __shared__ float red[WPB][DP];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float w[NH][DPL], part[NH][DPL];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      w[h][i] = W[h * DP + lane * DPL + i];
      part[h][i] = 0.f;
    }
  const long warp = ((long)blockIdx.x * BT + threadIdx.x) >> 5, nwarps = (long)gridDim.x * WPB;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    float dbr[NH];
    const long mine = base + lane;
#pragma unroll
    for (int h = 0; h < NH; ++h) dbr[h] = mine < R ? __ldg(DB + (long)h * R + mine) : 0.f;
    const int n = (int)min(32L, R - base);
    for (int r = 0; r < n; ++r) {
      float x[DPL], rstd, d[NH], g[DPL];
      load_row<DPL>(Z + (base + r) * DP + lane * DPL, x);
      normalize<DP, DPL>(x, eps, rstd);
#pragma unroll
      for (int h = 0; h < NH; ++h) d[h] = __shfl_sync(0xffffffffu, dbr[h], r);
      float mg = 0.f, mgx = 0.f;
#pragma unroll
      for (int i = 0; i < DPL; ++i) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          s += w[h][i] * d[h];
          part[h][i] += d[h] * x[i];
        }
        g[i] = s;
        mg += s;
        mgx += s * x[i];
      }
      mg = wsum(mg) * (1.f / DP);
      mgx = wsum(mgx) * (1.f / DP);
#pragma unroll
      for (int i = 0; i < DPL; ++i) g[i] = rstd * (g[i] - mg - x[i] * mgx);
      store_row<DPL>(DZ + (base + r) * DP + lane * DPL, g);
    }
  }
  // block partial of dWf, head by head: the warps in order
  for (int h = 0; h < NH; ++h) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) red[wp][lane * DPL + i] = part[h][i];
    __syncthreads();
    for (int c = threadIdx.x; c < DP; c += BT) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < WPB; ++q) v += red[q][c];
      ACC[((long)blockIdx.x * NH + h) * DP + c] = v;
    }
    __syncthreads();
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

#define PAIR_SWITCH(DPV, NHV, ...)                                                      \
  do {                                                                                  \
    if ((DPV) == 128 && (NHV) == 16) { constexpr int DPK = 128, NHK = 16; __VA_ARGS__; } \
    else if ((DPV) == 128 && (NHV) == 8) { constexpr int DPK = 128, NHK = 8; __VA_ARGS__; } \
    else if ((DPV) == 256 && (NHV) == 8) { constexpr int DPK = 256, NHK = 8; __VA_ARGS__; } \
    else if ((DPV) == 256 && (NHV) == 16) { constexpr int DPK = 256, NHK = 16; __VA_ARGS__; } \
    else if ((DPV) == 64 && (NHV) == 8) { constexpr int DPK = 64, NHK = 8; __VA_ARGS__; } \
    else TORCH_CHECK(false, "pair_rows: d_pair x heads 128 x 8 / 16, 256 x 8 / 16 or 64 x 8, got ", (long)(DPV), " x ", (long)(NHV)); \
  } while (0)
}  // namespace

at::Tensor pair_rows_fwd(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double scale, double neg) {
  chk(z, at::kBFloat16, "pair");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), dp = wf.size(1), R = L * L;
  TORCH_CHECK(z.numel() == R * dp, "pair_rows_fwd shapes");
  if (mask) {
    chk(*mask, at::kBool, "mask");
    TORCH_CHECK(mask->numel() == L, "mask [L]");
  }
  const at::cuda::CUDAGuard gd(z.device());
  auto out = at::empty({nh, L, L}, z.options());
  const unsigned grid = (unsigned)std::min<int64_t>((R / 32 + WPB - 1) / WPB, 4L * nsm_of(z));
  const bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* op = reinterpret_cast<bf*>(out.data_ptr());
  PAIR_SWITCH(dp, nh, fwd_k<DPK, NHK><<<grid, BT, 0, s>>>(zp, wf.data_ptr<float>(), mp, op, R, (int)L, (float)eps, (float)scale, (float)neg));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> pair_rows_bwd(at::Tensor z, at::Tensor db, at::Tensor wf, int64_t L, double eps) {
  chk(z, at::kBFloat16, "pair");
  chk(db, at::kFloat, "dbias");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), dp = wf.size(1), R = L * L;
  TORCH_CHECK(z.numel() == R * dp && db.numel() == nh * R, "pair_rows_bwd shapes");
  const at::cuda::CUDAGuard gd(z.device());
  auto dz = at::empty_like(z);
  const unsigned grid = (unsigned)std::min<int64_t>((R / 32 + WPB - 1) / WPB, 2L * nsm_of(z));
  auto acc = at::empty({(int64_t)grid, nh * dp}, z.options().dtype(at::kFloat));
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* dp_ = reinterpret_cast<bf*>(dz.data_ptr());
  PAIR_SWITCH(dp, nh, bwd_k<DPK, NHK><<<grid, BT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), dp_, acc.data_ptr<float>(), R, (float)eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_rows_fwd", &pair_rows_fwd);
  m.def("pair_rows_bwd", &pair_rows_bwd);
}
