// triattn_f8_ops.cu -- torch bindings of the RTX 5090 (sm_120a) e4m3 triangle-attention cores (attn_rf8_sm120.cuh), OPT-IN.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <utility>

#include "attn_rf8_sm120.cuh"

namespace {
using G8 = a100::AttnRCfg<1, 2, 4, RF8_ST, true, 32, false>;

// v8 [z L rows, H, L / 32, 1024] <- the e4m3 key-permuted tiles of v (token-major [z L L, ldv], head h = columns [32 h, 32 h + 32)) x vinv[h]; with k8 (numel > 0) also the
// e4m3 [key][channel-permuted] tiles of k x ks[h]; peak int32 [4] (see the kernel)
void quant(torch::Tensor v, torch::Tensor vinv, torch::Tensor v8, torch::Tensor k, torch::Tensor ks, torch::Tensor k8, torch::Tensor peak, int64_t L, int64_t H, int64_t rows, int64_t ldv, int64_t ldk) {
  const bool qk = k8.numel() > 0;
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kBFloat16 && v8.scalar_type() == at::kByte && vinv.scalar_type() == at::kFloat && peak.scalar_type() == at::kInt, "quant: dtypes");
  TORCH_CHECK(L % 32 == 0 && H >= 1 && H <= 16 && ldv >= H * 32 && ldv % 8 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "quant: geometry / alignment");
  TORCH_CHECK(v8.numel() == rows * H * L * 32 && vinv.numel() == H && peak.numel() >= 4, "quant: sizes");
  TORCH_CHECK((v.storage_offset() + (rows * L - 1) * ldv + H * 32) * v.element_size() <= (int64_t)v.storage().nbytes(), "quant: v extents");
  if (qk) {
    TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kBFloat16 && k8.scalar_type() == at::kByte && ks.scalar_type() == at::kFloat && ks.numel() == H && k8.numel() == v8.numel(), "quant: k");
    TORCH_CHECK(ldk >= H * 32 && ldk % 8 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                (k.storage_offset() + (rows * L - 1) * ldk + H * 32) * k.element_size() <= (int64_t)k.storage().nbytes(), "quant: k extents");
  }
  const c10::cuda::CUDAGuard guard(v.device());
  const dim3 grid((unsigned)(L / 32), (unsigned)rows);
  const auto st = at::cuda::getCurrentCUDAStream();
  const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  const auto* kp = qk ? reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()) : nullptr;
  const float* ksp = qk ? ks.data_ptr<float>() : nullptr;
  uint8_t* k8p = qk ? k8.data_ptr<uint8_t>() : nullptr;
#define QNT(HH) case HH: if (qk) a100::quant_kernel<HH, true><<<grid, 256, 0, st>>>(vp, ldv, vinv.data_ptr<float>(), v8.data_ptr<uint8_t>(), kp, ldk, ksp, k8p, (int)L, peak.data_ptr<int>()); \
                         else a100::quant_kernel<HH, false><<<grid, 256, 0, st>>>(vp, ldv, vinv.data_ptr<float>(), v8.data_ptr<uint8_t>(), kp, ldk, ksp, k8p, (int)L, peak.data_ptr<int>()); break;
  switch ((int)H) { QNT(1) QNT(2) QNT(3) QNT(4) QNT(6) QNT(8) QNT(12) QNT(16) default: TORCH_CHECK(false, "quant: heads 1 2 3 4 6 8 12 16"); }
#undef QNT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the gated inference core with e4m3 P V (and, with k8 / qs given, e4m3 Q K^T): aout [z L L, ldao] = bf16(sigmoid(gate) bf16(softmax(...) v)); v8 / k8 from quant, vsc fp32 [H] = 1 / vinv,
// qs fp32 [H]; key_mask [Z, L] bool / uint8 or empty
void attn_f8_gate(torch::Tensor q, torch::Tensor k, torch::Tensor v8, torch::Tensor vsc, torch::Tensor k8, torch::Tensor qs, torch::Tensor peak, torch::Tensor bias, torch::Tensor gate,
                  torch::Tensor aout, int64_t L, int64_t H, int64_t Z, int64_t ldq, int64_t ldk, int64_t ldgate, int64_t ldao, double scl, torch::Tensor key_mask) {
  const bool qk = k8.numel() > 0;
  for (auto* t : {&q, &k, &bias, &gate, &aout}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_f8_gate: bf16 CUDA tensors");
  TORCH_CHECK(v8.scalar_type() == at::kByte && vsc.scalar_type() == at::kFloat && vsc.numel() == H, "attn_f8_gate: v8 / vsc");
  TORCH_CHECK(!qk || (k8.scalar_type() == at::kByte && k8.numel() == v8.numel() && qs.scalar_type() == at::kFloat && qs.numel() == H && peak.scalar_type() == at::kInt && peak.numel() >= 4), "attn_f8_gate: k8 / qs");
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
  a8.k8 = qk ? k8.data_ptr<uint8_t>() : nullptr; a8.qs = qk ? qs.data_ptr<float>() : nullptr; a8.peak = qk ? peak.data_ptr<int>() : nullptr;
  at::Tensor kt;
  if (key_mask.defined() && key_mask.numel()) {
    TORCH_CHECK(key_mask.is_cuda() && (key_mask.scalar_type() == at::kBool || key_mask.scalar_type() == at::kByte) && key_mask.is_contiguous() && key_mask.numel() == Z * L, "attn_f8_gate: key_mask");
    kt = at::empty({Z}, q.options().dtype(at::kInt));
    C10_CUDA_CHECK(a100::rf_key_tiles(reinterpret_cast<const uint8_t*>(key_mask.data_ptr()), kt.data_ptr<int>(), (int)Z, (int)L, at::cuda::getCurrentCUDAStream()));
    p.key_tiles = kt.data_ptr<int>();
  }
  if (qk) C10_CUDA_CHECK((a100::attn_rf8_launch<G8, true>(a8, (int)Z, scl, at::cuda::getCurrentCUDAStream())));
  else C10_CUDA_CHECK((a100::attn_rf8_launch<G8, false>(a8, (int)Z, scl, at::cuda::getCurrentCUDAStream())));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant", &quant);
  m.def("attn_f8_gate", &attn_f8_gate);
}
