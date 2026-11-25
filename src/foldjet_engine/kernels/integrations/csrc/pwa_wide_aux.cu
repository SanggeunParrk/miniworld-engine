// MSAPairWeightedAveraging, wide widths (pwa_wide.py), backward: the two bandwidth passes that were PyTorch elementwise / permute-copy ops.
// One shape-independent build (every size is a launch argument).
//
//   drop_bwd_w  dout = bf16(bf16(dres * keep) * scale): the module's row dropout on the update, x * mask / (1 - p), what autograd of
//               `x * mask` then `* (1 / (1 - p))` computes in bf16 (the mask is exactly 0 / 1, the product is the identity or zero; the scale is
//               one fp32 multiply rounded to bf16, the same arithmetic as the two PyTorch kernels it replaces: bit-identical, one read + one write
//               of [S, N, D] instead of two of each plus a temporary).
//   perm_dv_w   dgv[s, n, HC + h C + c] = dvh[h, n, s C + c]: the head-major dv of the cuBLAS contraction into the second half of the natural
//               [S, N, 2 HC] buffer (the BIG build's dgv; PyTorch's strided copy ran at 0.3 of the bandwidth).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ float2 unpack(uint32_t u) {
  return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ uint32_t pack(float a, float b) {
  __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&r);
}

// one 16-byte vector (8 channels) per thread, grid-stride; `rows` = S * N, mask row = row % N
__global__ void __launch_bounds__(256) drop_bwd_kernel(const uint4* __restrict__ X, const uint4* __restrict__ M, uint4* __restrict__ Y, long nvec,
                                                       int vpr, int N, float scale) {
  for (long v = (long)blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += (long)gridDim.x * blockDim.x) {
    const long row = v / vpr;
    const int c = (int)(v - row * vpr);
    const int n = (int)(row % N);
    const uint4 x = X[v], m = __ldg(M + (long)n * vpr + c);
    uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ms[4] = {m.x, m.y, m.z, m.w}, o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 a = unpack(xs[i]), k = unpack(ms[i]);
      // round the masked product to bf16 first (exact for a 0 / 1 mask), then the scale
      const float2 t = unpack(pack(a.x * k.x, a.y * k.y));
      o[i] = pack(t.x * scale, t.y * scale);
    }
    Y[v] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// dvh [H][N][S*C] -> dgv[s][n][off + h*C + c], row length ld = 2 HC; one 16-byte vector per thread, C % 8 == 0
__global__ void __launch_bounds__(256) perm_dv_kernel(const uint4* __restrict__ X, uint4* __restrict__ Y, long nvec, int H, int N, int S, int C8, long ld8, long off8) {
  for (long v = (long)blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += (long)gridDim.x * blockDim.x) {
    const int c8 = (int)(v % C8);
    long r = v / C8;
    const int s = (int)(r % S);
    r /= S;
    const int n = (int)(r % N), h = (int)(r / N);
    Y[((long)s * N + n) * ld8 + off8 + (long)h * C8 + c8] = X[v];
  }
}
}  // namespace

torch::Tensor drop_bwd_w(torch::Tensor dres, torch::Tensor dmask, double scale) {
  TORCH_CHECK(dres.is_cuda() && dres.scalar_type() == torch::kBFloat16 && dres.is_contiguous() && dres.dim() == 3, "dres: contiguous bf16 [S, N, D]");
  const long S = dres.size(0), N = dres.size(1), D = dres.size(2);
  TORCH_CHECK(D % 8 == 0, "D % 8 == 0");
  TORCH_CHECK(dmask.is_cuda() && dmask.scalar_type() == torch::kBFloat16 && dmask.is_contiguous() && dmask.sizes() == torch::IntArrayRef({N, D}), "dmask: contiguous bf16 [N, D]");
  auto out = torch::empty_like(dres);
  const long nvec = dres.numel() / 8;
  if (nvec == 0) return out;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dres.get_device());
  const int grid = (int)std::min<long>((nvec + 255) / 256, (long)sms * 16);
  drop_bwd_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(dres.data_ptr()), reinterpret_cast<const uint4*>(dmask.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()), nvec,
      (int)(D / 8), (int)N, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// dvh [H, N, S*C] bf16 -> dgv[..., col0 : col0 + H*C] of dgv [S, N, ld]
void perm_dv_w(torch::Tensor dvh, torch::Tensor dgv, int64_t col0) {
  TORCH_CHECK(dvh.is_cuda() && dvh.scalar_type() == torch::kBFloat16 && dvh.is_contiguous() && dvh.dim() == 3, "dvh: contiguous bf16 [H, N, S*C]");
  TORCH_CHECK(dgv.is_cuda() && dgv.scalar_type() == torch::kBFloat16 && dgv.is_contiguous() && dgv.dim() == 3, "dgv: contiguous bf16 [S, N, ld]");
  const long H = dvh.size(0), N = dvh.size(1), S = dgv.size(0), ld = dgv.size(2);
  TORCH_CHECK(dgv.size(1) == N && dvh.size(2) % S == 0, "shapes");
  const long C = dvh.size(2) / S;
  TORCH_CHECK(C % 8 == 0 && ld % 8 == 0 && col0 % 8 == 0 && col0 + H * C <= ld, "C, ld, col0 % 8 == 0 and the columns inside the row");
  const long nvec = dvh.numel() / 8;
  if (nvec == 0) return;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dvh.get_device());
  const int grid = (int)std::min<long>((nvec + 255) / 256, (long)sms * 16);
  perm_dv_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(dvh.data_ptr()), reinterpret_cast<uint4*>(dgv.data_ptr()), nvec, (int)H,
                                                                   (int)N, (int)S, (int)(C / 8), ld / 8, col0 / 8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("drop_bwd_w", &drop_bwd_w, "dout = bf16(bf16(dres * keep) * scale) for dres [S, N, D], keep [N, D]");
  mod.def("perm_dv_w", &perm_dv_w, "head-major dv [H, N, S*C] into dgv[..., col0:col0 + H*C] ([S, N, ld])");
}
