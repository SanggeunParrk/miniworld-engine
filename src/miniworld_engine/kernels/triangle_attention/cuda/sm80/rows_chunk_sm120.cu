// rows_chunk_sm120.cu -- the RTX 5090 (sm_120a) launcher of the gated attention core over a chunk of pair rows (inference only).
//
// The width-generic inference forward (sm80_wide.py) materialises the front's output y [T, 4 H 32 + 8] for every token: 1040 B a token at d_pair 128,
// 7 GiB at L 2688, which no longer fits the card's 32 GB next to an AF3 trunk.  The core needs no more than one pair row at a time (the bias [H, L, L] is
// shared by all rows), so the chunked path (``sm80_wide._chunked_inference``) runs the front, this core and the output projection on chunks of rows:
// q / k / v / gate / aout point at the chunk's first token, the grid covers its `rows` pair rows (the kernel's row index is blockIdx.y R), the bias is the
// whole [1, H, L, L] plane.  The schedule is the one ``attn_fwd_gate<3>`` (``attn_fwd_gate_fast``) picks for the same L / H without out / lse, so a chunk's
// rows are computed exactly as the monolithic call computes them.  The sm_80 sources are untouched (this file is built for sm_120a only).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "attn_fwd_sm80.cuh"

namespace {

template <class G>
void launch_rows(const a100::AttnParams& p, int rows) {
  static_assert(G::SMEM <= 101376, "the RTX 5090's 99 KB of dynamic shared memory per block");
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::attn_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / G::BM, rows / G::R, p.H);
  a100::attn_fwd_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// aout [rows L, ldao] = bf16(sigmoid(gate) attention(q, k, v, bias)) for `rows` pair rows: q / k / v / gate token-major views of the chunk (row strides ld*),
// bias [1, H, L, L] bf16 contiguous (masked keys = bf16 min).
void attn_fwd_gate_rows(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor gate, torch::Tensor aout, int64_t L, int64_t H,
                        int64_t rows, int64_t ldq, int64_t ldk, int64_t ldv, int64_t ldgate, int64_t ldao, double scl) {
  for (auto* t : {&q, &k, &v, &bias, &gate, &aout}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_fwd_gate_rows: bf16 CUDA tensors");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == H * L * L, "attn_fwd_gate_rows: bias [1, H, L, L] contiguous");
  TORCH_CHECK(L % a100::TA_BM == 0 && L >= a100::TA_BM && rows > 0 && rows % 2 == 0 && rows <= L, "attn_fwd_gate_rows: L a multiple of 128, rows even");
  const int64_t T = rows * L;
  for (auto [t, ld] : std::initializer_list<std::pair<const torch::Tensor*, int64_t>>{{&q, ldq}, {&k, ldk}, {&v, ldv}, {&gate, ldgate}, {&aout, ldao}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_fwd_gate_rows: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_fwd_gate_rows: tensor extents");
  }
  TORCH_CHECK(H <= 65535 && rows / 2 <= 65535, "attn_fwd_gate_rows: grid");
  const at::cuda::CUDAGuard guard(q.device());
  a100::AttnParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.out = nullptr; p.lse = nullptr;
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.ldo = ldao; p.L = (int)L; p.H = (int)H; p.scl = (float)scl;
  p.gate = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()); p.aout = reinterpret_cast<__nv_bfloat16*>(aout.data_ptr()); p.ldgate = ldgate; p.ldao = ldao;
  const int r = (int)rows;
  // attn_fwd_gate<3> (inference: no out, no lse)
  if (L >= 512 && (L < 768 || H <= 8)) launch_rows<a100::AttnCfg<2, 4, 32, 2, true, false, 128, true>>(p, r);
  else if (L >= 256) launch_rows<a100::AttnCfg<2, 2, 64, 3, true, false, 64, (bool)A100_ATTN_FIXALL>>(p, r);
  else if (L % 384 == 0) launch_rows<a100::AttnCfg<2, 2, 48, 2, true, false, 128, (bool)A100_ATTN_FIXALL>>(p, r);
  else launch_rows<a100::AttnCfg<2, 4, 32, 2, true>>(p, r);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_fwd_gate_rows", &attn_fwd_gate_rows); }
