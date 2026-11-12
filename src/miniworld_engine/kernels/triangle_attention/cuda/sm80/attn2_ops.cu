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
#ifdef TRIATTN_SM120
  static_assert(G::SMEM <= 101376, "RTX 5090: 99 KB of dynamic shared memory per block");
#endif
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

// schedule experiments of the 16-channel backward kernels (0 = the default schedule), set from Python before the launches
static int g_dq_sched = 0, g_dkv_sched = 0;
void attn2_set_sched(int64_t dq, int64_t dkv) { g_dq_sched = (int)dq; g_dkv_sched = (int)dkv; }

// Forward: out (Lay lo) = softmax(sm_scale q . k + bias [+ rowmask]) v with q / k / v / out strided [z, a, j, h, d] views (a = the pair row, L of them); bias [z, H, L, L] bf16 contiguous
// (the shared key mask folded in: masked = bf16 min); rowmask [z, L, L] uint8 (the per-row key mask) or empty; lse [z, H, L, L] fp32 or empty.  The default schedule (2 rows, 32 keys, 2 CTAs / SM).
void attn2_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor rowmask, torch::Tensor out, torch::Tensor lse, std::vector<int64_t> lq,
               std::vector<int64_t> lk, std::vector<int64_t> lv, std::vector<int64_t> lo, int64_t L, int64_t H, int64_t Z, int64_t hd, double scl, int64_t fixed, int64_t sched) {
  TORCH_CHECK(hd == 16 || hd == 32 || hd == 64, "attn2_fwd: head dim 16, 32 or 64");
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
#define FWD2(HD, MASK, FIX) { using G = a100::Attn2Cfg<HD, 2, 4, 32, 2, MASK, FIX>; launch<G>(a100::attn2_fwd_kernel<G>, G::SMEM, grid, p); }
  // `fixed` (the first key tile's maximum as the softmax shift, a CTA with a non-finite seed recomputed online) is for the unmasked variants
#define FWD2S(R_, NKV_, BN_, MINB_) { const dim3 g((unsigned)(L / a100::T2_BM), (unsigned)(L / R_), (unsigned)(Z * H)); \
    if (fixed == 2) { using G = a100::Attn2Cfg<16, R_, NKV_, BN_, MINB_, false, 2>; launch<G>(a100::attn2_fwd_kernel<G>, G::SMEM, g, p); } \
    else { using G = a100::Attn2Cfg<16, R_, NKV_, BN_, MINB_, false, 1>; launch<G>(a100::attn2_fwd_kernel<G>, G::SMEM, g, p); } }
  // schedules of the 16-channel fixed-shift core (R rows, NKV stages, BN keys, CTAs / SM); 0 = automatic: 4 rows x 4 stages from L = 256 (measured -5..-8 % at L384-768, +4 % at L128), else the base 2 x 4
  const int64_t eff = sched != 0 ? sched : (L >= 256 ? 4 : 0);
  if (hd == 16 && !mask && fixed && eff != 0) {
    switch (eff) {
      case 1: FWD2S(2, 3, 64, 2) break;
      case 2: FWD2S(2, 4, 64, 2) break;
      case 3: FWD2S(4, 3, 32, 2) break;
      case 4: FWD2S(4, 4, 32, 2) break;
      case 5: FWD2S(2, 3, 32, 3) break;
      case 6: FWD2S(4, 3, 64, 1) break;
      case 7: FWD2S(1, 4, 32, 3) break;
      case 8: FWD2S(8, 4, 32, 2) break;
      case 9: FWD2S(4, 6, 32, 2) break;
      case 10: FWD2S(8, 3, 32, 2) break;
      case 11: FWD2S(8, 6, 32, 2) break;
      case 12: FWD2S(4, 3, 64, 2) break;
      default: TORCH_CHECK(false, "attn2_fwd: schedule");
    }
    return;
  }
#undef FWD2S
  if (hd == 64) {      // 64-channel heads: 2 pair rows, 3 stages (the prefetch stays within the two bias buffers: R = 1 would need NKV <= 2), online softmax, 72 KiB
    if (mask) { using G = a100::Attn2Cfg<64, 2, 3, 32, 2, true, 0>; launch<G>(a100::attn2_fwd_kernel<G>, G::SMEM, grid, p); }
    else { using G = a100::Attn2Cfg<64, 2, 3, 32, 2, false, 0>; launch<G>(a100::attn2_fwd_kernel<G>, G::SMEM, grid, p); }
    return;
  }
  if (hd == 32) { if (mask) FWD2(32, true, 0) else if (fixed) FWD2(32, false, 1) else FWD2(32, false, 0) }
  else { if (mask) FWD2(16, true, 0) else if (fixed == 2) FWD2(16, false, 2) else if (fixed) FWD2(16, false, 1) else FWD2(16, false, 0) }
#undef FWD2
}

// Backward, query side: dq (Lay ldq) and the bias gradient partials dbp [L / rows, z H, L, L] bf16 (db_reduce sums them).  lse / delta [z, H, L, L] fp32.  Default schedule: 4 rows, 3 K | V stages, 32 keys, 1 CTA / SM.
void attn2_bwd_dq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor rowmask, torch::Tensor lse, torch::Tensor delta, torch::Tensor dq,
                  torch::Tensor dbp, std::vector<int64_t> lq, std::vector<int64_t> lk, std::vector<int64_t> lv, std::vector<int64_t> ld, std::vector<int64_t> ldq, int64_t L, int64_t H,
                  int64_t Z, int64_t hd, double scl, double sm_scale) {
  constexpr int ROWS = 4;
  constexpr int ROWS64 = 2;                                   // 64-channel heads: Q and dO of 2 rows (128 B rows) fit the shared memory
#ifdef TRIATTN_SM120
  constexpr int NKV64 = 2;                                    // RTX 5090 (99 KB per block): a two-stage K | V ring keeps the 64-channel dq kernel at 98 KB (3 stages: 106 KB)
#else
  constexpr int NKV64 = 3;
#endif
  TORCH_CHECK(hd == 16 || hd == 32 || hd == 64, "attn2_bwd_dq: head dim 16, 32 or 64");
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
  TORCH_CHECK(dbp.is_cuda() && dbp.scalar_type() == at::kBFloat16 && dbp.is_contiguous() && dbp.numel() == (L / (hd == 64 ? ROWS64 : ROWS)) * Z * H * L * L, "attn2_bwd_dq: dbp [L / rows, z H, L, L] bf16 (rows 4; 2 for 64-channel heads)");
  const c10::cuda::CUDAGuard guard(q.device());
  a100::Dq2Params p;
  p.q = reinterpret_cast<const bf*>(q.data_ptr()); p.k = reinterpret_cast<const bf*>(k.data_ptr()); p.v = reinterpret_cast<const bf*>(v.data_ptr());
  p.dov = reinterpret_cast<const bf*>(dov.data_ptr()); p.bias = reinterpret_cast<const bf*>(bias.data_ptr()); p.rowmask = rowmask.numel() ? rowmask.data_ptr<uint8_t>() : nullptr;
  p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>(); p.dq = reinterpret_cast<bf*>(dq.data_ptr()); p.dbp = reinterpret_cast<bf*>(dbp.data_ptr());
  p.lq = Lq; p.lk = Lk; p.lv = Lv; p.ld = Ld; p.ldq = Ldq; p.L = (int)L; p.H = (int)H; p.ZH = (int)(Z * H); p.scl = (float)scl; p.sm_scale = (float)sm_scale;
  const dim3 grid((unsigned)(L / a100::T2_BM), (unsigned)(L / ROWS), (unsigned)(Z * H));
  const bool mask = rowmask.numel() != 0;
#define DQ2(HD, MASK) { using G = a100::Dq2Cfg<HD, ROWS, 3, 32, 1, MASK>; launch<G>(a100::attn2_bwd_dq_kernel<G>, G::SMEM, grid, p); }
#define DQ2S(NKV_, BN_, MINB_) { using G = a100::Dq2Cfg<16, ROWS, NKV_, BN_, MINB_, false>; launch<G>(a100::attn2_bwd_dq_kernel<G>, G::SMEM, grid, p); }
  if (hd == 64) {
    const dim3 g((unsigned)(L / a100::T2_BM), (unsigned)(L / ROWS64), (unsigned)(Z * H));
    if (mask) { using G = a100::Dq2Cfg<64, ROWS64, NKV64, 32, 1, true>; launch<G>(a100::attn2_bwd_dq_kernel<G>, G::SMEM, g, p); }
    else { using G = a100::Dq2Cfg<64, ROWS64, NKV64, 32, 1, false>; launch<G>(a100::attn2_bwd_dq_kernel<G>, G::SMEM, g, p); }
    return;
  }
  if (hd == 16 && !mask && g_dq_sched != 0) {                                 // schedule experiments of the 16-channel query side (rows stay 4: the bias-gradient partials)
    switch (g_dq_sched) {
      case 1: DQ2S(2, 32, 1) break;
      case 2: DQ2S(4, 32, 1) break;
      case 3: DQ2S(3, 32, 2) break;
      case 4: DQ2S(4, 32, 2) break;
      case 5: DQ2S(2, 32, 2) break;
      case 6: DQ2S(3, 64, 1) break;
      case 7: DQ2S(4, 64, 1) break;
      default: TORCH_CHECK(false, "attn2_bwd_dq: schedule");
    }
    return;
  }
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
  TORCH_CHECK(hd == 16 || hd == 32 || hd == 64, "attn2_bwd_dkv: head dim 16, 32 or 64");
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
#define DKV2S(NST_, MINB_) { using G = a100::Dkv2Cfg<16, NST_, 32, MINB_, false>; launch<G>(a100::attn2_bwd_dkv_kernel<G>, G::SMEM, grid, p); }
  if (hd == 64) {                                              // K | V resident (2 x 16 KiB), 3 query-tile stages, 1 CTA / SM (the accumulators of 64 channels)
    if (mask) { using G = a100::Dkv2Cfg<64, 3, 32, 1, true>; launch<G>(a100::attn2_bwd_dkv_kernel<G>, G::SMEM, grid, p); }
    else { using G = a100::Dkv2Cfg<64, 3, 32, 1, false>; launch<G>(a100::attn2_bwd_dkv_kernel<G>, G::SMEM, grid, p); }
    return;
  }
  if (hd == 16 && !mask && g_dkv_sched != 0) {
    switch (g_dkv_sched) {
      case 1: DKV2S(3, 2) break;
      case 2: DKV2S(2, 2) break;
      case 3: DKV2S(4, 3) break;
      case 4: DKV2S(3, 3) break;
      case 5: DKV2S(2, 3) break;
      case 6: DKV2S(5, 2) break;
      case 7: DKV2S(6, 1) break;
      default: TORCH_CHECK(false, "attn2_bwd_dkv: schedule");
    }
    return;
  }
  if (hd == 32) { if (mask) DKV2(32, true) else DKV2(32, false) }
  else { if (mask) DKV2(16, true) else DKV2(16, false) }
#undef DKV2
}

// delta [z, H, L rows, L queries] fp32 = sum_d o . do for the strided [z, a, j, h, d] views o (Lay lo) and dov (Lay ld): the backward's row term.
void attn2_delta(torch::Tensor o, torch::Tensor dov, torch::Tensor delta, std::vector<int64_t> lo, std::vector<int64_t> ld, int64_t L, int64_t H, int64_t Z, int64_t hd) {
  TORCH_CHECK(hd == 16 || hd == 32 || hd == 64, "attn2_delta: head dim 16, 32 or 64");
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
  if (hd == 64) a100::attn2_delta_kernel<64><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  else if (hd == 32) a100::attn2_delta_kernel<32><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  else a100::attn2_delta_kernel<16><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn2_delta", &attn2_delta);
  m.def("attn2_fwd", &attn2_fwd);
  m.def("attn2_set_sched", &attn2_set_sched);
  m.def("attn2_bwd_dq", &attn2_bwd_dq);
  m.def("attn2_db_reduce", &attn2_db_reduce);
  m.def("attn2_bias_transpose", &attn2_bias_transpose);
  m.def("attn2_bwd_dkv", &attn2_bwd_dkv);
}
