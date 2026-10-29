// Pair builder of La-Proteina's denoiser (pair embedding + adaptive LayerNorm) on sm_100a (B200), forward and backward.
//
// The forward (pe_fwd) and the backward's rows kernel (pe_bwd_rows: the pair recomputed, dx = d(table sum) in bf16, the parameter partials)
// are the sm_90a CUDA-core kernels built for sm_100a. The table gradient dT = onehot^T dx is the B200 tcgen05 kernel of
// pair_embed_geom/cuda/sm100/pe_tbl_sm100.cuh (sm_100a has no wgmma); fp32 partials, fixed-order reduce, no atomics.
#define PE_SM100 1
#include "../sm90/pair_embed_adaln.cu"
#include "../../../pair_embed_geom/cuda/sm100/pe_tbl_sm100.cuh"

namespace {
// the (at most three) table rows of a pair: sequence separation | nS + CA-distance bin | nS + nX + self-conditioning bin; NT <= 224
struct AdalnRows {
  static constexpr int NPADMAX = 224, TCOLS = 256, MAXR = 3, CB = 4;
  using Code = uint32_t;
  static constexpr uint32_t DEAD = MASKED;
  int nS, nX;
  __device__ __forceinline__ int rows(const uint32_t& w, int (&r)[MAXR], uint32_t (&v)[MAXR]) const {
    v[0] = v[1] = v[2] = 0x3F80u;
    if (w & MASKED) return 0;
    r[0] = (int)(w & 255);
    r[1] = nS + (int)((w >> 8) & 255);
    if (!(w & HAS_SC)) return 2;
    r[2] = nS + nX + (int)((w >> 16) & 255);
    return 3;
  }
};
}  // namespace

constexpr int NBLK100 = pe100::NBLK;

// impl 1: tcgen05 (NBLK100 slices per chunk); impl 0: the sm_90 shared-memory kernel (NBLK = 33 slices per chunk; A/B only).
void backward(torch::Tensor tbl, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor gamma, torch::Tensor bins, torch::Tensor dout,
              torch::Tensor dx, torch::Tensor part, torch::Tensor tpart, torch::Tensor dtbl, torch::Tensor dgamma,
              torch::Tensor dbeta, torch::Tensor dlw, torch::Tensor dlb, int64_t impl, int64_t chunk_b, int64_t nS, int64_t nX, double e1, double e2) {
  const at::cuda::CUDAGuard guard(tbl.device());
  int B = bins.size(0), n = bins.size(1), NT = tbl.size(0);
  TORCH_CHECK(NT <= AdalnRows::NPADMAX, "pair_embed_adaln sm100: at most ", AdalnRows::NPADMAX, " table rows");
  int nrb = (n + NW - 1) / NW;
  int nchunk = (B + chunk_b - 1) / chunk_b;
  const int nblk = impl == 1 ? NBLK100 : NBLK;
  TORCH_CHECK(tpart.numel() >= (long long)nchunk * nblk * NT * C, "pair_embed_adaln backward: tpart too small");
  TORCH_CHECK((long long)chunk_b * n * n < (1ll << 31), "pair_embed_adaln backward: at most 2^31 pairs per chunk");
  auto st = at::cuda::getCurrentCUDAStream();
  size_t smem = (size_t)4 * NT * 64 * sizeof(float);
  if (impl != 1) cudaFuncSetAttribute(pe_bwd_tbl, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  auto* dxp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>());
  auto* dop = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>());
  const int ncw = (NT + 15) / 16 * 16;
  for (int ch = 0; ch < nchunk; ch++) {
    int b0 = ch * chunk_b, Bc = std::min<int>(chunk_b, B - b0);
    pe_bwd_rows<<<Bc * nrb, NW * 32, 0, st>>>(tbl.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(), gamma.data_ptr<float>(),
                                              bins.data_ptr<int>(), dop, dxp, part.data_ptr<float>() + (size_t)b0 * nrb * 4 * C, b0, n, nrb,
                                              (int)nS, (int)nX, (float)e1, (float)e2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int* bp = bins.data_ptr<int>() + (size_t)b0 * n * n;
    float* tp = tpart.data_ptr<float>() + (size_t)ch * nblk * NT * C;
    if (impl == 1) {
      // a zero bin word past the end of the bins would read as a live pair: the kernel kills every code past the chunk's last pair
      pe100::launch(dxp, (long long)Bc * n * n, bins.data_ptr<int>(), (long long)B * n * n, (long long)b0 * n * n, AdalnRows{(int)nS, (int)nX}, NT, 1,
                    ncw, tp, st);
    } else {
      pe_bwd_tbl<<<dim3(4, NBLK), 128, smem, st>>>(dxp, bp, (long long)Bc * n * n, (int)nS, (int)nX, NT, tp);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  pe_reduce_rows<<<B + 1, C, 0, st>>>(part.data_ptr<float>(), dgamma.data_ptr<float>(), dbeta.data_ptr<float>(), dlw.data_ptr<float>(),
                                      dlb.data_ptr<float>(), B, nrb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pe100::reduce_parts<float><<<(NT * C / 4 + 255) / 256, 256, 0, st>>>(tpart.data_ptr<float>(), dtbl.data_ptr<float>(), nchunk * nblk, (long)NT * C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
}
