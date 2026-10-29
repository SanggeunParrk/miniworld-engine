// Backbone pair embedding of La-Proteina's autoencoder (encoder pair + decoder CA pair) on sm_100a (B200).
//
// The forward kernels (pg_fwd / pg_fwd_ca: bins + a gather of bf16 table rows from shared memory, CUDA cores, bandwidth bound) and the
// LayerNorm backward (pg_ln_bwd) are the sm_90a ones, built for sm_100a: nothing in them is Hopper-specific. The table gradient is the
// B200 kernel of pe_tbl_sm100.cuh (tcgen05 / TMEM / TMA): sm_100a has no wgmma.
#define PE_SM100 1
#include "../sm90/pair_embed_geom.cu"
#include "pe_tbl_sm100.cuh"

namespace {
// the rows of the encoder (MODE 0: 317 = two N = 160 MMAs) / decoder (MODE 1: 157, one N = 160 MMA) table a pair adds to, from its 16-byte code
template <int MODE>
struct GeomRows {
  static constexpr int NPADMAX = MODE == 0 ? 320 : 160, TCOLS = MODE == 0 ? 512 : 256, MAXR = MODE == 0 ? 11 : 2, CB = 16;
  using Code = uint4;
  static constexpr uint4 DEAD = {0u, 0u, 0u, 0u};      // no live bit
  __device__ __forceinline__ int rows(const uint4& c, int (&r)[MAXR], uint32_t (&v)[MAXR]) const {
    int rr[11];
    uint32_t cvb;
    const int n = scatter_rows<MODE>(c, rr, cvb);
#pragma unroll
    for (int k = 0; k < MAXR; k++) {
      r[k] = rr[k];
      v[k] = (MODE == 0 && rr[k] == R_CHAIN) ? cvb : 0x3F80u;
    }
    return n;
  }
};
}  // namespace

// mode 0: encoder table [317][256]; mode 1: decoder table [157][256].  impl 1: tcgen05 (pe_tbl_sm100.cuh, pe100::NBLK slices),
// impl 2: the sm_90 mma.sync kernel, impl 0: the shared-memory read-modify-write kernel (both NBLK = 33 slices; A/B only).
// dtbl bf16 [NTAB][256], tpart fp32 scratch [slices][NTAB][256].
void backward(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl) {
  const at::cuda::CUDAGuard guard(g.device());
  int B = g.size(0), n = g.size(1);
  long long P = (long long)B * n * n;
  TORCH_CHECK(P < (1ll << 31), "pair_embed backward: at most 2^31 pairs");
  auto st = at::cuda::getCurrentCUDAStream();
  int ntab = mode == 0 ? NT : NT_CA;
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* tp = tpart.data_ptr<float>();
  auto* dp = reinterpret_cast<__nv_bfloat16*>(dtbl.data_ptr<at::BFloat16>());
  if (impl == 1) {
    TORCH_CHECK(tpart.numel() >= (long long)pe100::NBLK * ntab * C, "pair_embed backward: tpart too small");
    if (mode == 0) pe100::launch(gp, P, cp, P, 0, GeomRows<0>{}, ntab, 2, 160, tp, st);
    else pe100::launch(gp, P, cp, P, 0, GeomRows<1>{}, ntab, 1, 160, tp, st);
    pe100::reduce_parts<__nv_bfloat16><<<(ntab * C / 4 + 255) / 256, 256, 0, st>>>(tp, dp, pe100::NBLK, (long)ntab * C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  TORCH_CHECK(tpart.numel() >= (long long)NBLK * ntab * C, "pair_embed backward: tpart too small");
  if (impl == 2) {
    if (mode == 0) pg_bwd_mma<0><<<dim3(4, NBLK), 13 * 32, 0, st>>>(gp, cp, P, tp);
    else pg_bwd_mma<1><<<dim3(4, NBLK), 5 * 32, 0, st>>>(gp, cp, P, tp);
  } else {
    size_t smem = (size_t)4 * ntab * 32 * sizeof(float);
    if (mode == 0) {
      cudaFuncSetAttribute(pg_bwd_tbl<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      pg_bwd_tbl<0><<<dim3(8, NBLK), 128, smem, st>>>(gp, cp, P, tp);
    } else {
      cudaFuncSetAttribute(pg_bwd_tbl<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      pg_bwd_tbl<1><<<dim3(8, NBLK), 128, smem, st>>>(gp, cp, P, tp);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pg_reduce_tbl<<<(ntab * C + 255) / 256, 256, 0, st>>>(tp, dp, ntab);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("forward_ca", &forward_ca);
  m.def("backward", &backward);
  m.def("ln_backward", &ln_backward);
}
