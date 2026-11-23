// triattn_f8_ops.cu -- torch bindings of the RTX 5090 (sm_120a) e4m3 P.V triangle-attention core (attn_rf8_sm120.cuh), OPT-IN.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <utility>

#include "attn_rf8_sm120.cuh"

namespace {
using G8 = a100::AttnRCfg<1, 2, 4, RF8_ST, true, 32, false>;

// v8 [z L rows, H, L / 32, 1024] <- the e4m3 key-permuted tiles of v (token-major [z L L, ldv], head h = columns [32 h, 32 h + 32)) x vinv[h]; peak int32 [2] (see the kernel)
void vt_quant(torch::Tensor v, torch::Tensor vinv, torch::Tensor v8, torch::Tensor peak, int64_t L, int64_t H, int64_t rows, int64_t ldv) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kBFloat16 && v8.scalar_type() == at::kByte && vinv.scalar_type() == at::kFloat && peak.scalar_type() == at::kInt,
              "vt_quant: dtypes");
  TORCH_CHECK(L % 32 == 0 && H >= 1 && H <= 16 && ldv >= H * 32 && ldv % 8 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "vt_quant: geometry / alignment");
  TORCH_CHECK(v8.numel() == rows * H * L * 32 && vinv.numel() == H && peak.numel() >= 2, "vt_quant: sizes");
  TORCH_CHECK((v.storage_offset() + (rows * L - 1) * ldv + H * 32) * v.element_size() <= (int64_t)v.storage().nbytes(), "vt_quant: v extents");
  const c10::cuda::CUDAGuard guard(v.device());
  const dim3 grid((unsigned)(L / 32), (unsigned)rows);
  const auto st = at::cuda::getCurrentCUDAStream();
#define VTQ(HH) case HH: a100::vt_quant_kernel<HH><<<grid, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), ldv, vinv.data_ptr<float>(), v8.data_ptr<uint8_t>(), (int)L, peak.data_ptr<int>()); break;
  switch ((int)H) { VTQ(1) VTQ(2) VTQ(3) VTQ(4) VTQ(6) VTQ(8) VTQ(12) VTQ(16) default: TORCH_CHECK(false, "vt_quant: heads 1 2 3 4 6 8 12 16"); }
#undef VTQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the gated inference core with e4m3 P V: aout [z L L, ldao] = bf16(sigmoid(gate) bf16(softmax(...) v)); v8 from vt_quant, vsc fp32 [H] = 1 / vinv; key_mask [Z, L] bool / uint8 or empty
void attn_f8_gate(torch::Tensor q, torch::Tensor k, torch::Tensor v8, torch::Tensor vsc, torch::Tensor bias, torch::Tensor gate, torch::Tensor aout, int64_t L, int64_t H, int64_t Z,
                  int64_t ldq, int64_t ldk, int64_t ldgate, int64_t ldao, double scl, torch::Tensor key_mask) {
  for (auto* t : {&q, &k, &bias, &gate, &aout}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_f8_gate: bf16 CUDA tensors");
  TORCH_CHECK(v8.scalar_type() == at::kByte && vsc.scalar_type() == at::kFloat && vsc.numel() == H, "attn_f8_gate: v8 / vsc");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == Z * H * L * L && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0, "attn_f8_gate: bias [z, H, L, L] contiguous");
  TORCH_CHECK(L % 128 == 0 && L >= 256 && v8.numel() == Z * L * H * L * 32, "attn_f8_gate: L a multiple of 128");
  TORCH_CHECK(Z * H <= 65535 && L / 2 <= 65535, "attn_f8_gate: grid");
  const int64_t T = Z * L * L;
  for (auto [t, ld] : std::initializer_list<std::pair<const torch::Tensor*, int64_t>>{{&q, ldq}, {&k, ldk}, {&gate, ldgate}, {&aout, ldao}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_f8_gate: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_f8_gate: tensor extents");
  }
  const c10::cuda::CUDAGuard guard(q.device());
  a100::Attn8Params a8;
  a100::AttnParams& p = a8.p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()); p.v = nullptr;
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()); p.out = nullptr; p.lse = nullptr;
  p.ldq = ldq; p.ldk = ldk; p.ldv = 0; p.ldo = 0; p.L = (int)L; p.H = (int)H; p.scl = (float)scl;
  p.gate = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()); p.aout = reinterpret_cast<__nv_bfloat16*>(aout.data_ptr()); p.ldgate = ldgate; p.ldao = ldao;
  a8.v8 = v8.data_ptr<uint8_t>(); a8.vsc = vsc.data_ptr<float>();
  at::Tensor kt;
  if (key_mask.defined() && key_mask.numel()) {
    TORCH_CHECK(key_mask.is_cuda() && (key_mask.scalar_type() == at::kBool || key_mask.scalar_type() == at::kByte) && key_mask.is_contiguous() && key_mask.numel() == Z * L, "attn_f8_gate: key_mask");
    kt = at::empty({Z}, q.options().dtype(at::kInt));
    C10_CUDA_CHECK(a100::rf_key_tiles(reinterpret_cast<const uint8_t*>(key_mask.data_ptr()), kt.data_ptr<int>(), (int)Z, (int)L, at::cuda::getCurrentCUDAStream()));
    p.key_tiles = kt.data_ptr<int>();
  }
  C10_CUDA_CHECK(a100::attn_rf8_launch<G8>(a8, (int)Z, scl, at::cuda::getCurrentCUDAStream()));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vt_quant", &vt_quant);
  m.def("attn_f8_gate", &attn_f8_gate);
}
