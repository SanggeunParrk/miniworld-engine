// fp32 pair -> bias pass of the token DiT's TF32 inference: bias[h, i, p] = sum_c Wf[h, c] LN(pair[i L + key])[c] for a 128- or 256-wide
// fp32 pair and 16 heads (``pair_bias_rows32``, the only entry point; ``conditioned_transition/cuda/__init__.py`` calls it for the 256-wide
// diffusion pair). Wf carries the LayerNorm weight, the head projection and log2 e; the bias is key-permuted for the TF32 core (position p of
// each group of 8 holds key kperm(p)); a masked key gets ``neg``. A warp owns a row at a time (lane l: channels DP / 32 l ..), the heads'
// dot products are warp reduce-scatter sums; CUDA cores only: the step is bound by the pair read.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"  // tdr::warp_sum, tdr::transpose_k: shared with the token DiT row kernels

namespace {
using bf = __nv_bfloat16;
constexpr int BT = 256;
constexpr int WPB = BT / 32;
constexpr int PF = 4;  // rows whose loads are in flight together

// ------------------------------------------------------------------------------------------------ fp32 TF32-core glue
// key_perm (conditioned_transition.cuda): position p of each group of 8 holds key perm(p); inv is its inverse
__device__ __forceinline__ int kperm(int p) { const int r = p & 7; return p - r + 4 * (r & 1) + (r >> 1); }
template <int DP, int DPL>
__device__ __forceinline__ void normalize(float (&x)[DPL], float eps, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) s += x[i];
  const float mean = tdr::warp_sum(s) * (1.f / DP);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) {
    x[i] -= mean;
    v += x[i] * x[i];
  }
  rstd = rsqrtf(tdr::warp_sum(v) * (1.f / DP) + eps);
#pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= rstd;
}

// 16 values per lane summed over the warp and scattered: value i ends (summed) in lanes 2i, 2i + 1 -- 16 shuffles for what 16
// butterfly sums do in 80
__device__ __forceinline__ float reduce_scatter16(const float (&v)[16], int lane) {
  float a[8], b[4], c[2];
  {
    const bool hi = lane & 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = (hi ? v[i + 8] : v[i]) + __shfl_xor_sync(0xffffffffu, hi ? v[i] : v[i + 8], 16);
  }
  {
    const bool hi = lane & 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = (hi ? a[i + 4] : a[i]) + __shfl_xor_sync(0xffffffffu, hi ? a[i] : a[i + 4], 8);
  }
  {
    const bool hi = lane & 4;
#pragma unroll
    for (int i = 0; i < 2; ++i) c[i] = (hi ? b[i + 2] : b[i]) + __shfl_xor_sync(0xffffffffu, hi ? b[i] : b[i + 2], 4);
  }
  const bool hi = lane & 2;
  float d = (hi ? c[1] : c[0]) + __shfl_xor_sync(0xffffffffu, hi ? c[0] : c[1], 2);
  return d + __shfl_xor_sync(0xffffffffu, d, 1);
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


// fp32 pair [L L, DP] (DP = 128 or 256), 16 heads: bias[h, i, p] (key-permuted for the TF32 core: position p of each group of 8 holds key kperm(p)) =
// sum_c Wf[h, c] LN(pair[i L + key])[c], a masked key = neg. Wf (carries the LayerNorm weight, the head projection and log2 e) lives in shared memory;
// a warp owns a row at a time (channels q * 128 + 4 lane ..), 32 rows' results are staged so the bias is written coalesced along the key.
template <int DP, int NW>
__global__ void __launch_bounds__(NW * 32) bias32_k(const float* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                                float* __restrict__ OUT, long R, int L, float eps, float neg) {
  constexpr int DPL = DP / 32, NQ = DP / 128;
  __shared__ __align__(16) float wsm[16 * DP];
  __shared__ float stage[NW][32][17];
  for (int c = threadIdx.x; c < 16 * DP / 4; c += NW * 32) reinterpret_cast<float4*>(wsm)[c] = reinterpret_cast<const float4*>(W)[c];
  __syncthreads();
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const long warp = ((long)blockIdx.x * NW * 32 + threadIdx.x) >> 5, nwarps = (long)gridDim.x * NW;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    for (int rb = 0; rb < 32; rb += PF) {
      float xb[PF][DPL];
#pragma unroll
      for (int q = 0; q < PF; ++q)
#pragma unroll
        for (int c = 0; c < NQ; ++c) {
          const float4 a = __ldg(reinterpret_cast<const float4*>(Z + (base + rb + q) * DP + c * 128 + lane * 4));
          xb[q][4 * c] = a.x; xb[q][4 * c + 1] = a.y; xb[q][4 * c + 2] = a.z; xb[q][4 * c + 3] = a.w;
        }
#pragma unroll
      for (int q = 0; q < PF; ++q) {
        float x[DPL], v[16], rstd;
#pragma unroll
        for (int i = 0; i < DPL; ++i) x[i] = xb[q][i];
        normalize<DP, DPL>(x, eps, rstd);
#pragma unroll
        for (int h = 0; h < 16; ++h) {
          float d = 0.f;
#pragma unroll
          for (int c = 0; c < NQ; ++c) {
            const float4 w = *reinterpret_cast<const float4*>(wsm + h * DP + c * 128 + lane * 4);
            d += w.x * x[4 * c] + w.y * x[4 * c + 1] + w.z * x[4 * c + 2] + w.w * x[4 * c + 3];
          }
          v[h] = d;
        }
        const float u = reduce_scatter16(v, lane);
        if ((lane & 1) == 0) stage[wp][rb + q][lane >> 1] = u;
      }
    }
    __syncwarp();
    const int j0 = (int)(base % L), kk = kperm(lane);
    const bool drop = MASK != nullptr && !MASK[j0 + kk];
#pragma unroll
    for (int h = 0; h < 16; ++h) OUT[(long)h * R + base + lane] = drop ? neg : stage[wp][kk][h];
    __syncwarp();
  }
}

void pair_bias_rows32(at::Tensor z, at::Tensor wf, at::Tensor out, c10::optional<at::Tensor> mask, int64_t L, double eps, double neg) {
  chk(z, at::kFloat, "pair");
  chk(wf, at::kFloat, "wf");
  chk(out, at::kFloat, "bias");
  const int64_t R = L * L, dp = wf.size(1);
  const int64_t nh = wf.size(0);
  TORCH_CHECK((nh == 16 && (dp == 128 || dp == 256)) && z.numel() == R * dp && out.numel() == nh * R && L % 32 == 0, "pair_bias_rows32 shapes");
  if (mask) {
    chk(*mask, at::kBool, "mask");
    TORCH_CHECK(mask->numel() == L, "mask [L]");
  }
  const at::cuda::CUDAGuard gd(z.device());
  const bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  {
    // few rows (L 128): small blocks so that every SM has work
    const bool small = (R / 32 + 7) / 8 < 2L * nsm_of(z);
    const unsigned g = (unsigned)std::min<int64_t>((R / 32 + (small ? 2 : 8) - 1) / (small ? 2 : 8), 4L * nsm_of(z));
    const float* zp = z.data_ptr<float>(); const float* wp = wf.data_ptr<float>(); float* op = out.data_ptr<float>();
    if (dp == 128) { if (small) bias32_k<128, 2><<<g, 64, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); else bias32_k<128, 8><<<g, 256, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); }
    else { if (small) bias32_k<256, 2><<<g, 64, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); else bias32_k<256, 8><<<g, 256, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_bias_rows32", &pair_bias_rows32);
}
