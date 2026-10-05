// attn2_ops.cu -- torch bindings of the generalised triangle-attention core (attn2_fwd_sm80.cuh, attn2_bwd_sm80.cuh): head dim 16 | 32, element strides, optional per-pair-row key mask.
// A separate extension from ops.cu's (the d_pair-128 kernels stay as they are).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "attn_bwd_dkv_sm80.cuh"      // bias_transpose_kernel
#include "attn_bwd_dq_sm80.cuh"       // db_reduce_kernel
#include "attn2_bwd_sm80.cuh"

namespace {

using bf = __nv_bfloat16;

a100::Lay lay_of(const std::vector<int64_t>& v) {
  TORCH_CHECK(v.size() == 4, "a layout is (token, row, batch, head) element strides");
  return a100::Lay{v[0], v[1], v[2], v[3]};
}

// a strided bf16 tensor (a [z, a, j, h, d] view with the strides `l`): 16-byte aligned base and strides (the granule loads), extents inside the storage
void check_strided(const torch::Tensor& t, const a100::Lay& l, int64_t Z, int64_t L, int64_t H, int64_t hd, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, what, ": bf16 CUDA tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, what, ": 16-byte aligned base");
  TORCH_CHECK(l.tok % 8 == 0 && l.row % 8 == 0 && l.z % 8 == 0 && l.head % 8 == 0 && l.tok >= 1 && l.row >= 0 && l.z >= 0 && l.head >= 0, what, ": strides must be multiples of 8 elements");
  const int64_t last = (Z - 1) * l.z + (L - 1) * l.row + (L - 1) * l.tok + (H - 1) * l.head + hd;
  TORCH_CHECK((t.storage_offset() + last) * t.element_size() <= (int64_t)t.storage().nbytes(), what, ": tensor extents");
}

template <class G, class P>
void launch(void (*kernel)(const P), int smem, dim3 grid, const P& p) {
  static bool set[64] = {};                                   // the attribute is per device and per instantiation
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    set[dev] = true;
  }
  kernel<<<grid, G::NTHR, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_mask(const torch::Tensor& m, int64_t Z, int64_t L, const char* what) {
  TORCH_CHECK(m.numel() == 0 || (m.is_cuda() && m.scalar_type() == at::kByte && m.is_contiguous() && m.numel() == Z * L * L),
              what, ": rowmask [Z, L, L] uint8 (or empty)");
  TORCH_CHECK(m.numel() == 0 || L <= 32 * a100::MASK_WORDS, what, ": a masked call takes L <= ", 32 * a100::MASK_WORDS);
}

}  // namespace

// Forward: out (Lay lo) = softmax(sm_scale q . k + bias [+ rowmask]) v with q / k / v / out strided [z, a, j, h, d] views (a = the pair row, L of them); bias [z, H, L, L] bf16 contiguous
// (the shared key mask folded in: masked = bf16 min); rowmask [z, L, L] uint8 (the per-row key mask) or empty; lse [z, H, L, L] fp32 or empty.  The default schedule (2 rows, 32 keys, 2 CTAs / SM).
void attn2_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor rowmask, torch::Tensor out, torch::Tensor lse, std::vector<int64_t> lq,
               std::vector<int64_t> lk, std::vector<int64_t> lv, std::vector<int64_t> lo, int64_t L, int64_t H, int64_t Z, int64_t hd, double scl) {
  TORCH_CHECK(hd == 16 || hd == 32, "attn2_fwd: head dim 16 or 32");
  TORCH_CHECK(L % 128 == 0 && L >= 128 && Z * H <= 65535 && L / 2 <= 65535, "attn2_fwd: L a multiple of 128, grid");
  const a100::Lay Lq = lay_of(lq), Lk = lay_of(lk), Lv = lay_of(lv), Lo = lay_of(lo);
  check_strided(q, Lq, Z, L, H, hd, "attn2_fwd q");
  check_strided(k, Lk, Z, L, H, hd, "attn2_fwd k");
  check_strided(v, Lv, Z, L, H, hd, "attn2_fwd v");
  check_strided(out, Lo, Z, L, H, hd, "attn2_fwd out");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.numel() == Z * H * L * L && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "attn2_fwd: bias [z, H, L, L] bf16 contiguous");
  check_mask(rowmask, Z, L, "attn2_fwd");
  TORCH_CHECK(!lse.numel() || (lse.scalar_type() == at::kFloat && lse.numel() == Z * H * L * L && lse.is_contiguous()), "attn2_fwd: lse [z, H, L, L] fp32");
  const c10::cuda::CUDAGuard guard(q.device());
  a100::Attn2Params p;
  p.q = reinterpret_cast<const bf*>(q.data_ptr()); p.k = reinterpret_cast<const bf*>(k.data_ptr()); p.v = reinterpret_cast<const bf*>(v.data_ptr());
  p.bias = reinterpret_cast<const bf*>(bias.data_ptr()); p.rowmask = rowmask.numel() ? rowmask.data_ptr<uint8_t>() : nullptr;
  p.out = reinterpret_cast<bf*>(out.data_ptr()); p.lse = lse.numel() ? lse.data_ptr<float>() : nullptr;
  p.lq = Lq; p.lk = Lk; p.lv = Lv; p.lo = Lo; p.L = (int)L; p.H = (int)H; p.scl = (float)scl;
  const dim3 grid((unsigned)(L / a100::T2_BM), (unsigned)(L / 2), (unsigned)(Z * H));
  const bool mask = rowmask.numel() != 0;
#define FWD2(HD, MASK) { using G = a100::Attn2Cfg<HD, 2, 4, 32, 2, MASK>; launch<G>(a100::attn2_fwd_kernel<G>, G::SMEM, grid, p); }
  if (hd == 32) { if (mask) FWD2(32, true) else FWD2(32, false) }
  else { if (mask) FWD2(16, true) else FWD2(16, false) }
#undef FWD2
}

// Backward, query side: dq (Lay ldq) and the bias gradient partials dbp [L / rows, z H, L, L] bf16 (db_reduce sums them).  lse / delta [z, H, L, L] fp32.  Default schedule: 4 rows, 3 K | V stages, 32 keys, 1 CTA / SM.
void attn2_bwd_dq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor rowmask, torch::Tensor lse, torch::Tensor delta, torch::Tensor dq,
                  torch::Tensor dbp, std::vector<int64_t> lq, std::vector<int64_t> lk, std::vector<int64_t> lv, std::vector<int64_t> ld, std::vector<int64_t> ldq, int64_t L, int64_t H,
                  int64_t Z, int64_t hd, double scl, double sm_scale) {
  constexpr int ROWS = 4;
  TORCH_CHECK(hd == 16 || hd == 32, "attn2_bwd_dq: head dim 16 or 32");
  TORCH_CHECK(L % 128 == 0 && L >= 128 && Z * H <= 65535 && L / ROWS <= 65535, "attn2_bwd_dq: L a multiple of 128, grid");
  const a100::Lay Lq = lay_of(lq), Lk = lay_of(lk), Lv = lay_of(lv), Ld = lay_of(ld), Ldq = lay_of(ldq);
  check_strided(q, Lq, Z, L, H, hd, "attn2_bwd_dq q");
  check_strided(k, Lk, Z, L, H, hd, "attn2_bwd_dq k");
  check_strided(v, Lv, Z, L, H, hd, "attn2_bwd_dq v");
  check_strided(dov, Ld, Z, L, H, hd, "attn2_bwd_dq dov");
  check_strided(dq, Ldq, Z, L, H, hd, "attn2_bwd_dq dq");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.numel() == Z * H * L * L, "attn2_bwd_dq: bias [z, H, L, L] bf16 contiguous");
  check_mask(rowmask, Z, L, "attn2_bwd_dq");
  for (const torch::Tensor* t : {&lse, &delta})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == Z * H * L * L, "attn2_bwd_dq: lse / delta [z, H, L, L] fp32");
  TORCH_CHECK(dbp.is_cuda() && dbp.scalar_type() == at::kBFloat16 && dbp.is_contiguous() && dbp.numel() == (L / ROWS) * Z * H * L * L, "attn2_bwd_dq: dbp [L / 4, z H, L, L] bf16");
  const c10::cuda::CUDAGuard guard(q.device());
  a100::Dq2Params p;
  p.q = reinterpret_cast<const bf*>(q.data_ptr()); p.k = reinterpret_cast<const bf*>(k.data_ptr()); p.v = reinterpret_cast<const bf*>(v.data_ptr());
  p.dov = reinterpret_cast<const bf*>(dov.data_ptr()); p.bias = reinterpret_cast<const bf*>(bias.data_ptr()); p.rowmask = rowmask.numel() ? rowmask.data_ptr<uint8_t>() : nullptr;
  p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>(); p.dq = reinterpret_cast<bf*>(dq.data_ptr()); p.dbp = reinterpret_cast<bf*>(dbp.data_ptr());
  p.lq = Lq; p.lk = Lk; p.lv = Lv; p.ld = Ld; p.ldq = Ldq; p.L = (int)L; p.H = (int)H; p.ZH = (int)(Z * H); p.scl = (float)scl; p.sm_scale = (float)sm_scale;
  const dim3 grid((unsigned)(L / a100::T2_BM), (unsigned)(L / ROWS), (unsigned)(Z * H));
  const bool mask = rowmask.numel() != 0;
#define DQ2(HD, MASK) { using G = a100::Dq2Cfg<HD, ROWS, 3, 32, 1, MASK>; launch<G>(a100::attn2_bwd_dq_kernel<G>, G::SMEM, grid, p); }
  if (hd == 32) { if (mask) DQ2(32, true) else DQ2(32, false) }
  else { if (mask) DQ2(16, true) else DQ2(16, false) }
#undef DQ2
}

// db [z H, L, L] bf16 = the sum of the `groups` partials dbp [groups, z H, L, L] (fixed order).
void attn2_db_reduce(torch::Tensor dbp, torch::Tensor db, int64_t groups) {
  TORCH_CHECK(dbp.is_cuda() && dbp.scalar_type() == at::kBFloat16 && dbp.is_contiguous() && db.scalar_type() == at::kBFloat16 && db.is_contiguous() && dbp.numel() == groups * db.numel() &&
              db.numel() % 8 == 0, "attn2_db_reduce: dbp [groups, ...] / db [...] bf16 contiguous");
  const c10::cuda::CUDAGuard guard(dbp.device());
  const long long plane8 = db.numel() / 8;
  a100::db_reduce_kernel<<<(unsigned)((plane8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf*>(dbp.data_ptr()), reinterpret_cast<bf*>(db.data_ptr()), (int)groups, plane8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// bias_t [z H, L, L] = the transpose of each [L, L] plane of bias.
void attn2_bias_transpose(torch::Tensor bias, torch::Tensor bias_t, int64_t L) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias_t.scalar_type() == at::kBFloat16 && bias_t.is_contiguous() && bias.numel() == bias_t.numel() &&
              L % 32 == 0 && bias.numel() % (L * L) == 0, "attn2_bias_transpose: [zh, L, L] bf16 contiguous");
  const int64_t zh = bias.numel() / (L * L);
  TORCH_CHECK(zh <= 65535 && L / 32 <= 65535, "attn2_bias_transpose: grid");
  const c10::cuda::CUDAGuard guard(bias.device());
  a100::bias_transpose_kernel<<<dim3((unsigned)(L / 32), (unsigned)(L / 32), (unsigned)zh), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf*>(bias.data_ptr()), reinterpret_cast<bf*>(bias_t.data_ptr()), (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Backward, key side: dk (Lay ldk) and dv (Lay ldv); bias_t [z H, L, L] = the bias transposed.  Default schedule: 4 query-tile stages, 32 queries per tile, 2 CTAs / SM.
void attn2_bwd_dkv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias_t, torch::Tensor rowmask, torch::Tensor lse, torch::Tensor delta, torch::Tensor dk,
                   torch::Tensor dv, std::vector<int64_t> lq, std::vector<int64_t> lk, std::vector<int64_t> lv, std::vector<int64_t> ld, std::vector<int64_t> ldk, std::vector<int64_t> ldv,
                   int64_t L, int64_t H, int64_t Z, int64_t hd, double scl, double sm_scale) {
  TORCH_CHECK(hd == 16 || hd == 32, "attn2_bwd_dkv: head dim 16 or 32");
  TORCH_CHECK(L % 128 == 0 && L >= 128 && Z * H <= 65535 && L <= 65535, "attn2_bwd_dkv: L a multiple of 128, grid");
  const a100::Lay Lq = lay_of(lq), Lk = lay_of(lk), Lv = lay_of(lv), Ld = lay_of(ld), Ldk = lay_of(ldk), Ldv = lay_of(ldv);
  check_strided(q, Lq, Z, L, H, hd, "attn2_bwd_dkv q");
  check_strided(k, Lk, Z, L, H, hd, "attn2_bwd_dkv k");
  check_strided(v, Lv, Z, L, H, hd, "attn2_bwd_dkv v");
  check_strided(dov, Ld, Z, L, H, hd, "attn2_bwd_dkv dov");
  check_strided(dk, Ldk, Z, L, H, hd, "attn2_bwd_dkv dk");
  check_strided(dv, Ldv, Z, L, H, hd, "attn2_bwd_dkv dv");
  TORCH_CHECK(bias_t.is_cuda() && bias_t.scalar_type() == at::kBFloat16 && bias_t.is_contiguous() && bias_t.numel() == Z * H * L * L, "attn2_bwd_dkv: bias_t [z, H, L, L] bf16 contiguous");
  check_mask(rowmask, Z, L, "attn2_bwd_dkv");
  for (const torch::Tensor* t : {&lse, &delta})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == Z * H * L * L, "attn2_bwd_dkv: lse / delta [z, H, L, L] fp32");
  const c10::cuda::CUDAGuard guard(q.device());
  a100::Dkv2Params p;
  p.q = reinterpret_cast<const bf*>(q.data_ptr()); p.k = reinterpret_cast<const bf*>(k.data_ptr()); p.v = reinterpret_cast<const bf*>(v.data_ptr());
  p.dov = reinterpret_cast<const bf*>(dov.data_ptr()); p.bias_t = reinterpret_cast<const bf*>(bias_t.data_ptr()); p.rowmask = rowmask.numel() ? rowmask.data_ptr<uint8_t>() : nullptr;
  p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>(); p.dk = reinterpret_cast<bf*>(dk.data_ptr()); p.dv = reinterpret_cast<bf*>(dv.data_ptr());
  p.lq = Lq; p.lk = Lk; p.lv = Lv; p.ld = Ld; p.ldk = Ldk; p.ldv = Ldv; p.L = (int)L; p.H = (int)H; p.scl = (float)scl; p.sm_scale = (float)sm_scale;
  const dim3 grid((unsigned)(L / a100::T2_BM), (unsigned)L, (unsigned)(Z * H));
  const bool mask = rowmask.numel() != 0;
#define DKV2(HD, MASK) { using G = a100::Dkv2Cfg<HD, 4, 32, 2, MASK>; launch<G>(a100::attn2_bwd_dkv_kernel<G>, G::SMEM, grid, p); }
  if (hd == 32) { if (mask) DKV2(32, true) else DKV2(32, false) }
  else { if (mask) DKV2(16, true) else DKV2(16, false) }
#undef DKV2
}

// delta [z, H, L rows, L queries] fp32 = sum_d o . do for the strided [z, a, j, h, d] views o (Lay lo) and dov (Lay ld): the backward's row term.
void attn2_delta(torch::Tensor o, torch::Tensor dov, torch::Tensor delta, std::vector<int64_t> lo, std::vector<int64_t> ld, int64_t L, int64_t H, int64_t Z, int64_t hd) {
  TORCH_CHECK(hd == 16 || hd == 32, "attn2_delta: head dim 16 or 32");
  TORCH_CHECK(L % 128 == 0 && L >= 128 && Z * H <= 65535 && L <= 65535, "attn2_delta: L a multiple of 128, grid");
  const a100::Lay Lo = lay_of(lo), Ld = lay_of(ld);
  check_strided(o, Lo, Z, L, H, hd, "attn2_delta o");
  check_strided(dov, Ld, Z, L, H, hd, "attn2_delta dov");
  TORCH_CHECK(delta.is_cuda() && delta.scalar_type() == at::kFloat && delta.is_contiguous() && delta.numel() == Z * H * L * L, "attn2_delta: delta [z, H, L, L] fp32");
  const c10::cuda::CUDAGuard guard(o.device());
  a100::DeltaParams p;
  p.o = reinterpret_cast<const bf*>(o.data_ptr()); p.dov = reinterpret_cast<const bf*>(dov.data_ptr()); p.delta = delta.data_ptr<float>();
  p.lo = Lo; p.ld = Ld; p.L = (int)L; p.H = (int)H;
  const dim3 grid((unsigned)((L + 127) / 128), (unsigned)L, (unsigned)(Z * H));
  if (hd == 32) a100::attn2_delta_kernel<32><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  else a100::attn2_delta_kernel<16><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn2_delta", &attn2_delta);
  m.def("attn2_fwd", &attn2_fwd);
  m.def("attn2_bwd_dq", &attn2_bwd_dq);
  m.def("attn2_db_reduce", &attn2_db_reduce);
  m.def("attn2_bias_transpose", &attn2_bias_transpose);
  m.def("attn2_bwd_dkv", &attn2_bwd_dkv);
}
