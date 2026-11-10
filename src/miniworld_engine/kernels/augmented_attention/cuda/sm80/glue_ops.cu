// glue_ops.cu -- torch bindings of glue_sm80.cuh (the gates of the A100 AugmentedAttentionPairBias path); a second extension of the family, so a change of the attention
// kernels does not rebuild it and the other way round.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "glue_sm80.cuh"

namespace {

// a [M][d] operand with a row stride (bf16 or fp32, unit column stride, 16-byte aligned rows)
void check_rows(const torch::Tensor& t, const char* name, int64_t M, int64_t d) {
  TORCH_CHECK(t.is_cuda() && (t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat) && t.dim() == 2 && t.size(0) == M && t.size(1) == d && t.stride(1) == 1,
              name, ": a [M, d] CUDA tensor (bf16 / fp32, unit column stride)");
  const int64_t eb = t.element_size();
  TORCH_CHECK(t.stride(0) * eb % 16 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned rows");
}

inline unsigned grid_for(int64_t M, int64_t d) {
  const int64_t chunks = M * (d / 8);
  const int64_t blocks = (chunks + aa80::GLUE_NT - 1) / aa80::GLUE_NT;
  const int64_t cap = (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 16;
  return (unsigned)std::max<int64_t>(1, std::min(blocks, cap));
}

void same_type(std::initializer_list<const torch::Tensor*> ts, const char* what) {
  for (const auto* t : ts) TORCH_CHECK(t->scalar_type() == (*ts.begin())->scalar_type(), what, ": one dtype");
}

#define DISPATCH_T(tensor, ...)                                                            \
  if ((tensor).scalar_type() == at::kBFloat16) { using T = __nv_bfloat16; __VA_ARGS__ }     \
  else { using T = float; __VA_ARGS__ }

}  // namespace

// og = sigmoid(g) o
void gate_rows(torch::Tensor o, torch::Tensor g, torch::Tensor og) {
  const int64_t M = o.size(0), d = o.size(1);
  TORCH_CHECK(d % 8 == 0, "gate_rows: d a multiple of 8");
  check_rows(o, "gate_rows o", M, d); check_rows(g, "gate_rows g", M, d); check_rows(og, "gate_rows og", M, d);
  same_type({&o, &g, &og}, "gate_rows");
  const at::cuda::CUDAGuard guard(o.device());
  DISPATCH_T(o, aa80::gate_rows_kernel<T><<<grid_for(M, d), aa80::GLUE_NT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(o.data_ptr()), reinterpret_cast<const T*>(g.data_ptr()), reinterpret_cast<T*>(og.data_ptr()), M, (int)d, o.stride(0), g.stride(0), og.stride(0));)
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dob = dog sigmoid(g), dg = dog o sigmoid(g) (1 - sigmoid(g))
void gate_bwd(torch::Tensor dog, torch::Tensor o, torch::Tensor g, torch::Tensor dob, torch::Tensor dg) {
  const int64_t M = o.size(0), d = o.size(1);
  TORCH_CHECK(d % 8 == 0, "gate_bwd: d a multiple of 8");
  check_rows(dog, "gate_bwd dog", M, d); check_rows(o, "gate_bwd o", M, d); check_rows(g, "gate_bwd g", M, d); check_rows(dob, "gate_bwd dob", M, d);
  check_rows(dg, "gate_bwd dg", M, d);
  same_type({&dog, &o, &g, &dob, &dg}, "gate_bwd");
  const at::cuda::CUDAGuard guard(o.device());
  DISPATCH_T(o, aa80::gate_bwd_kernel<T><<<grid_for(M, d), aa80::GLUE_NT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(dog.data_ptr()), reinterpret_cast<const T*>(o.data_ptr()), reinterpret_cast<const T*>(g.data_ptr()), reinterpret_cast<T*>(dob.data_ptr()),
      reinterpret_cast<T*>(dg.data_ptr()), M, (int)d, dog.stride(0), o.stride(0), g.stride(0), dob.stride(0), dg.stride(0));)
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out = res + sigmoid(g2) y   (res optional: an empty tensor means out = sigmoid(g2) y)
void res_gate(torch::Tensor y, torch::Tensor g2, torch::Tensor res, torch::Tensor out) {
  const int64_t M = y.size(0), d = y.size(1);
  TORCH_CHECK(d % 8 == 0, "res_gate: d a multiple of 8");
  const bool has_res = res.numel() > 0;
  check_rows(y, "res_gate y", M, d); check_rows(g2, "res_gate g2", M, d); check_rows(out, "res_gate out", M, d);
  if (has_res) { check_rows(res, "res_gate res", M, d); same_type({&y, &g2, &res, &out}, "res_gate"); }
  else same_type({&y, &g2, &out}, "res_gate");
  const at::cuda::CUDAGuard guard(y.device());
  auto st = at::cuda::getCurrentCUDAStream();
  DISPATCH_T(y,
    if (has_res) aa80::res_gate_kernel<T, true><<<grid_for(M, d), aa80::GLUE_NT, 0, st>>>(
        reinterpret_cast<const T*>(y.data_ptr()), reinterpret_cast<const T*>(g2.data_ptr()), reinterpret_cast<const T*>(res.data_ptr()), reinterpret_cast<T*>(out.data_ptr()), M,
        (int)d, y.stride(0), g2.stride(0), res.stride(0), out.stride(0));
    else aa80::res_gate_kernel<T, false><<<grid_for(M, d), aa80::GLUE_NT, 0, st>>>(
        reinterpret_cast<const T*>(y.data_ptr()), reinterpret_cast<const T*>(g2.data_ptr()), nullptr, reinterpret_cast<T*>(out.data_ptr()), M, (int)d, y.stride(0), g2.stride(0), 0,
        out.stride(0));)
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dy = dout sigmoid(g2), dg2 = dout y sigmoid(g2) (1 - sigmoid(g2))
void res_gate_bwd(torch::Tensor dout, torch::Tensor y, torch::Tensor g2, torch::Tensor dy, torch::Tensor dg2) {
  const int64_t M = y.size(0), d = y.size(1);
  TORCH_CHECK(d % 8 == 0, "res_gate_bwd: d a multiple of 8");
  check_rows(dout, "res_gate_bwd dout", M, d); check_rows(y, "res_gate_bwd y", M, d); check_rows(g2, "res_gate_bwd g2", M, d); check_rows(dy, "res_gate_bwd dy", M, d);
  check_rows(dg2, "res_gate_bwd dg2", M, d);
  same_type({&dout, &y, &g2, &dy, &dg2}, "res_gate_bwd");
  const at::cuda::CUDAGuard guard(y.device());
  DISPATCH_T(y, aa80::res_gate_bwd_kernel<T><<<grid_for(M, d), aa80::GLUE_NT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(dout.data_ptr()), reinterpret_cast<const T*>(y.data_ptr()), reinterpret_cast<const T*>(g2.data_ptr()), reinterpret_cast<T*>(dy.data_ptr()),
      reinterpret_cast<T*>(dg2.data_ptr()), M, (int)d, dout.stride(0), y.stride(0), g2.stride(0), dy.stride(0), dg2.stride(0));)
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out [P][Lp][Lp] bf16 (the core's raw-unit bias) from in [P][L][L] bf16 / fp32 (natural units, P = B H planes): in x scale, ``fill`` on masked / padded keys, 0 on padded query rows
// (see glue_sm80.cuh); km: a bool [B][L] key mask or an empty tensor
void bias_pack(torch::Tensor in, torch::Tensor km, torch::Tensor out, int64_t L, int64_t Lp, int64_t H, double scale, double fill) {
  TORCH_CHECK(in.is_cuda() && (in.scalar_type() == at::kBFloat16 || in.scalar_type() == at::kFloat) && in.is_contiguous() && in.dim() == 3 && in.size(1) == L && in.size(2) == L,
              "bias_pack: in [P, L, L] contiguous bf16 / fp32");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 3 && out.size(0) == in.size(0) && out.size(1) == Lp && out.size(2) == Lp,
              "bias_pack: out [P, Lp, Lp] contiguous bf16");
  TORCH_CHECK(L % 8 == 0 && Lp % 8 == 0 && Lp >= L && H >= 1 && in.size(0) % H == 0, "bias_pack: L, Lp multiples of 8");
  const unsigned char* kmp = nullptr;
  if (km.numel() > 0) {
    TORCH_CHECK(km.is_cuda() && km.scalar_type() == at::kBool && km.is_contiguous() && km.numel() == (in.size(0) / H) * L, "bias_pack: km a bool [B, L] tensor");
    kmp = reinterpret_cast<const unsigned char*>(km.data_ptr());
  }
  const int64_t rows = in.size(0) * Lp;
  const at::cuda::CUDAGuard guard(in.device());
  auto st = at::cuda::getCurrentCUDAStream();
  DISPATCH_T(in, aa80::bias_pack_kernel<T><<<grid_for(rows, Lp), aa80::GLUE_NT, 0, st>>>(reinterpret_cast<const T*>(in.data_ptr()), kmp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                                                                         (int)L, (int)Lp, (int)H, rows, (float)scale, (float)fill);)
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_rows_cuda", &gate_rows);
  m.def("gate_bwd_cuda", &gate_bwd);
  m.def("res_gate_cuda", &res_gate);
  m.def("res_gate_bwd_cuda", &res_gate_bwd);
  m.def("bias_pack_cuda", &bias_pack);
}
