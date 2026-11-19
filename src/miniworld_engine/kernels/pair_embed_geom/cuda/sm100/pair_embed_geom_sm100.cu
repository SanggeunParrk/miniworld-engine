// pair_embed_geom_sm100.cu -- the B200 (sm_100a) build of the backbone pair embedding (encoder pair, decoder CA pair).
//
// The A100 source (../sm80/pair_embed_geom_sm80.cu: sm90/pair_embed_geom.cu compiled unchanged with the mma.sync forms of its wgmma
// helpers) is included as it is: its forwards, LayerNorm backward and table-gradient impls 0-3 are legal sm_100a code. Its module definition
// is renamed into a function this file's module calls, which adds one entry, `backward_tc`: the table gradient on the B200 tensor cores
// (pe_tbl_sm100.cuh: tcgen05 / TMEM / TMA, fp32 partials per fixed pair slice, summed in slice order: bitwise deterministic, no atomics).
// sm_100a has no wgmma; the mma.sync table gradient built for it was measured 8x slower than the tcgen05 kernel (B16 x L512, 2026-10).
// MWE_SM120 is defined for this build too: the A100 source then also compiles the RTX 5090 additions (the one-launch inference forwards
// from the module's own inputs, gt 2; the L1-table LayerNorm backward, bound as `ln_backward` with the H100 one as `ln_backward_h100`);
// nothing in them is sized for the 5090 but the shared-memory budget, which the B200 exceeds.
#define MWE_SM120 1
#include <torch/extension.h>
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void pair_embed_geom_sm80_bindings(pybind11::module_& variable)
#include "../sm80/pair_embed_geom_sm80.cu"
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")
#include "pe_tbl_sm100.cuh"
#include "pe_ca_sm100.cuh"

namespace {
// the rows of the encoder (MODE 0: 317 = two N = 160 MMAs) / decoder (MODE 1: 157, one N = 160 MMA) table a pair adds to, from its 16-byte code
template <int MODE>
struct GeomRows {
  static constexpr int NPADMAX = MODE == 0 ? 320 : 160, TCOLS = MODE == 0 ? 512 : 256, MAXR = MODE == 0 ? 11 : 2, CB = 16;
  using Code = uint4;
  static constexpr uint4 DEAD = {0u, 0u, 0u, 0u};  // no live bit
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

#include "pe_ln_sm100.cuh"
#include "pe_geom_sm100.cuh"

// mode 0: encoder table [317][256]; mode 1: decoder table [157][256]. g [B, n, n, 256] bf16, codes [B, n, n, 4] int32,
// tpart fp32 scratch [pe100::NBLK][NTAB][256], dtbl bf16 [NTAB][256].
void backward_tc(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode) {
  const at::cuda::CUDAGuard guard(g.device());
  const long long P = (long long)g.size(0) * g.size(1) * g.size(2);
  TORCH_CHECK(P < (1ll << 31), "pair_embed backward: at most 2^31 pairs");
  const int ntab = mode == 0 ? NT : NT_CA;
  TORCH_CHECK(tpart.numel() >= (long long)pe100::NBLK * ntab * C, "pair_embed backward: tpart too small");
  auto st = at::cuda::getCurrentCUDAStream();
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* tp = tpart.data_ptr<float>();
  if (mode == 0) pe100::launch(gp, P, cp, P, 0, GeomRows<0>{}, ntab, 2, 160, tp, st);
  else pe100::launch(gp, P, cp, P, 0, GeomRows<1>{}, ntab, 1, 160, tp, st);
  pe100::reduce_parts<__nv_bfloat16><<<(ntab * C / 4 + 255) / 256, 256, 0, st>>>(
      tp, reinterpret_cast<__nv_bfloat16*>(dtbl.data_ptr<at::BFloat16>()), pe100::NBLK, (long)ntab * C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pair_embed_geom_sm80_bindings(m);
  m.def("ln_backward_h100", &ln_backward_h100);
  m.def("backward_tc", &backward_tc);
  m.def("forward_ca_b200", &forward_ca_b200);
  m.def("forward_b200", &forward_b200);
  m.def("forward_b200s", &forward_b200s);
  m.def("ln_backward_b200", &ln_backward_b200);
  m.def("backward_w_b200", &backward_w_b200);
}
