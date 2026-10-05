// ops.cu -- torch bindings of the A100 (sm_80) attention-with-shared-bias kernels (the samples share the pair bias; head dim 32 or 48).
//
//   attn_fwd       attn_fwd_sm80.cuh      MODE_GATE (inference: sigmoid(g) o over q), MODE_TRAIN (fp32 o + lse), MODE_PLAIN (bf16 o, optional lse)
//   attn_bwd_dq    attn_bwd_dq_sm80.cuh   dq (fp32 or bf16) and the bf16 bias-gradient partials;  db_reduce: their fixed-order sum (fp32)
//   attn_bwd_dkv   attn_bwd_dkv_sm80.cuh  dk, dv (fp32 or bf16);  bias_transpose: the bias transposed for it
//   attn_delta     aux_sm80.cuh           the backward's row term sum_d dO o
//   pair_bias_fwd / pair_bias_bwd         aux_sm80.cuh   the atom width's pair bias (LayerNorm of 16 channels + a 16 -> 4 projection) and its backward
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_bwd_dkv_sm80.cuh"
#include "aux_sm80.cuh"

namespace {

void check_bf16(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, ": a bf16 CUDA tensor");
}

// a token-major operand [T][ld] viewed over `t`: row stride ld (>= H hd), 16-byte aligned rows, and the whole [T][H hd] window inside the storage
void check_tokens(const torch::Tensor& t, const char* name, int64_t T, int64_t ld, int64_t cols) {
  TORCH_CHECK(ld >= cols && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": row stride / alignment");
  TORCH_CHECK((t.storage_offset() + (T - 1) * ld + cols) * t.element_size() <= (int64_t)t.storage().nbytes(), name, ": tensor extents");
}

template <class G>
void launch_fwd(const aa80::FwdParams& p, int A) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / aa80::BM, A / G::R, p.H);
  aa80::attn_fwd_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_dq(const aa80::DqParams& p, int A) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dq_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / aa80::BM, A / G::R, p.H);
  aa80::attn_bwd_dq_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_dkv(const aa80::DkvParams& p, int A) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dkv_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / aa80::BM, A, p.H);
  aa80::attn_bwd_dkv_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_f32(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous() && t.numel() == n, name, ": a contiguous fp32 CUDA tensor of ", n, " elements");
}

// a gradient buffer [T][cols]: fp32 or bf16, contiguous
void check_grad(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && (t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16) && t.is_contiguous() && t.numel() == n, name,
              ": a contiguous fp32 or bf16 CUDA tensor of ", n, " elements");
}

// a gradient operand [T][ld] (a view of a wider buffer is fine: the gradient of q / k / v written straight into the columns of the projection's gradient):
// fp32 or bf16, unit column stride, row stride ld >= cols (a multiple of 2 elements), the whole [T][cols] window inside the storage
void check_grad_view(const torch::Tensor& t, const char* name, int64_t T, int64_t ld, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && (t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16) && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) == ld && ld >= cols &&
              ld % 2 == 0, name, ": a fp32 / bf16 CUDA view [T][ld]");
  TORCH_CHECK((t.storage_offset() + (T - 1) * ld + cols) * t.element_size() <= (int64_t)t.storage().nbytes() &&
              reinterpret_cast<uintptr_t>(t.data_ptr()) % (2 * t.element_size()) == 0, name, ": extents / alignment");
}

}  // namespace

// q / k / v / gate: token-major views [A L][ld] of bf16 (head h = columns [hd h, hd h + hd)), bias [H][L][L] bf16 (masked keys -inf or very negative).
// MODE_GATE: out = the bf16 tensor the gated result is written to (the caller passes q itself: in place); MODE_TRAIN: out = float [A L][ldo = H hd],
// lse = float [A][H][L]; MODE_PLAIN: out = bf16 [A L][ldo >= H hd], lse = float [A][H][L] or an empty tensor; MODE_PGATE: as MODE_GATE with the raw-unit bias of MODE_PLAIN.
// ss: tokens between consecutive samples (<= 0: L); kpen: per-sample key penalties fp32 [A][L] (sample stride kps floats) or an empty tensor (see attn_fwd_sm80.cuh).
void attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor gate, torch::Tensor out, torch::Tensor lse,
              int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t ldg, int64_t ldo,
              double scl, double bscl, int64_t mode, int64_t rows, int64_t nkv, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps) {
  for (auto* t : {&q, &k, &v, &bias}) check_bf16(*t, "attn_fwd");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == H * L * L, "attn_fwd: bias [H, L, L] contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A % rows == 0 && A / rows <= 65535 && H <= 65535, "attn_fwd: L a multiple of 128, A a multiple of rows");
  if (ss <= 0) ss = L;
  TORCH_CHECK(ss >= L, "attn_fwd: ss >= L");
  const int64_t T = (A - 1) * ss + L, cols = H * hd;       // tokens spanned by the operands
  check_tokens(q, "attn_fwd q", T, ldq, cols); check_tokens(k, "attn_fwd k", T, ldk, cols); check_tokens(v, "attn_fwd v", T, ldv, cols);
  aa80::FwdParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.gate = nullptr; p.lse = nullptr; p.out = nullptr;
  if (mode == aa80::MODE_GATE || mode == aa80::MODE_PLAIN || mode == aa80::MODE_PGATE) {
    check_bf16(out, "attn_fwd out");
    check_tokens(out, "attn_fwd out", T, ldo, cols);
    p.out = out.data_ptr();
    if (mode == aa80::MODE_GATE || mode == aa80::MODE_PGATE) {
      check_bf16(gate, "attn_fwd gate");
      check_tokens(gate, "attn_fwd gate", T, ldg, cols);
      p.gate = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
    } else if (lse.numel() > 0) {
      check_f32(lse, "attn_fwd lse", A * H * L);
      p.lse = lse.data_ptr<float>();
    }
  } else {
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.numel() == A * L * cols && ldo == cols && ss == L,
                "attn_fwd: out float [A L, H hd]");
    check_f32(lse, "attn_fwd lse", A * H * L);
    p.out = out.data_ptr(); p.lse = lse.data_ptr<float>();
  }
  const int64_t km = kpen.numel() > 0;
  p.kpen = nullptr; p.kps = 0;
  if (km) {
    TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && (A - 1) * kps + L <= kpen.numel() &&
                reinterpret_cast<uintptr_t>(kpen.data_ptr()) % 16 == 0 && kps % 4 == 0, "attn_fwd: kpen fp32 [A][L] (16-byte aligned rows)");
    p.kpen = kpen.data_ptr<float>(); p.kps = kps;
  }
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.ldg = ldg; p.ldo = ldo; p.ss = ss; p.L = (int)L; p.H = (int)H; p.scl = (float)scl; p.bscl = (float)bscl;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t cfg = ((((hd * 10 + rows) * 10 + nkv) * 10 + minb) * 10 + mode) * 10 + km;
#define FWD_CFG(HD, R, NKV, MINB, MODE, KM) \
  if (cfg == ((((int64_t(HD) * 10 + (R)) * 10 + (NKV)) * 10 + (MINB)) * 10 + (MODE)) * 10 + (KM)) { \
    launch_fwd<aa80::FwdCfg<HD, R, NKV, MINB, MODE, KM>>(p, (int)A); return; }
  FWD_CFG(48, 1, 3, 2, 0, 0) FWD_CFG(48, 2, 3, 2, 0, 0) FWD_CFG(48, 1, 3, 2, 1, 0) FWD_CFG(48, 2, 3, 2, 1, 0) FWD_CFG(48, 1, 3, 2, 2, 0) FWD_CFG(48, 2, 3, 2, 2, 0)
  FWD_CFG(48, 1, 4, 2, 0, 0) FWD_CFG(48, 1, 4, 1, 0, 0) FWD_CFG(48, 2, 4, 1, 0, 0) FWD_CFG(48, 1, 4, 2, 1, 0) FWD_CFG(48, 1, 4, 1, 1, 0) FWD_CFG(48, 2, 4, 1, 1, 0)
  FWD_CFG(48, 1, 4, 2, 2, 0)
  FWD_CFG(32, 1, 3, 2, 2, 0) FWD_CFG(32, 2, 3, 2, 2, 0) FWD_CFG(32, 1, 4, 2, 2, 0) FWD_CFG(32, 1, 3, 2, 1, 0) FWD_CFG(32, 2, 3, 2, 1, 0) FWD_CFG(32, 1, 4, 2, 1, 0)
  FWD_CFG(32, 1, 3, 2, 0, 0) FWD_CFG(32, 2, 3, 2, 0, 0) FWD_CFG(32, 1, 4, 2, 0, 0)
  // the module-level core: MODE_PGATE (inference) and, with a per-sample key mask, MODE_PLAIN / MODE_PGATE (the schedules of ``_fwd_schedule``)
  FWD_CFG(48, 1, 4, 2, 3, 0) FWD_CFG(48, 2, 3, 2, 3, 0) FWD_CFG(32, 1, 4, 2, 3, 0) FWD_CFG(32, 2, 3, 2, 3, 0)
  FWD_CFG(48, 1, 4, 2, 2, 1) FWD_CFG(48, 2, 3, 2, 2, 1) FWD_CFG(32, 1, 4, 2, 2, 1) FWD_CFG(32, 2, 3, 2, 2, 1)
  FWD_CFG(48, 1, 4, 2, 3, 1) FWD_CFG(48, 2, 3, 2, 3, 1) FWD_CFG(32, 1, 4, 2, 3, 1) FWD_CFG(32, 2, 3, 2, 3, 1)
#undef FWD_CFG
  TORCH_CHECK(false, "attn_fwd: unsupported schedule (hd, rows, nkv, minb, mode, kpen)");
}

// The backward, query side (attn_bwd_dq_sm80.cuh): q / k / v token-major bf16 views, dov [A L][lddo] bf16, bias [H][L][L] natural units (masked keys -inf),
// lse / delta [A][H][L] fp32 -> dq [A L][lddq = H hd] fp32 or bf16 (the dtype of the tensor passed) and the partial bias gradients dbp [A / rows][H][L][L]
// bf16 (``db_reduce`` sums them).
void attn_bwd_dq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor lse, torch::Tensor delta,
                 torch::Tensor dq, torch::Tensor dbp, int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo,
                 int64_t lddq, double scl, double bscl, double sm_scale, int64_t rows, int64_t nkv, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps) {
  for (auto* t : {&q, &k, &v, &dov, &bias, &dbp}) check_bf16(*t, "attn_bwd_dq");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == H * L * L, "attn_bwd_dq: bias [H, L, L] contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A % rows == 0 && A / rows <= 65535 && H <= 65535, "attn_bwd_dq: L a multiple of 128, A a multiple of rows");
  if (ss <= 0) ss = L;
  TORCH_CHECK(ss >= L, "attn_bwd_dq: ss >= L");
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_tokens(q, "attn_bwd_dq q", T, ldq, cols); check_tokens(k, "attn_bwd_dq k", T, ldk, cols); check_tokens(v, "attn_bwd_dq v", T, ldv, cols);
  check_tokens(dov, "attn_bwd_dq dov", T, lddo, cols);
  check_f32(lse, "attn_bwd_dq lse", A * H * L); check_f32(delta, "attn_bwd_dq delta", A * H * L);
  check_grad_view(dq, "attn_bwd_dq dq", T, lddq, cols);
  TORCH_CHECK(dbp.is_contiguous() && dbp.numel() == (A / rows) * H * L * L, "attn_bwd_dq: dbp [A / rows, H, L, L] contiguous");
  aa80::DqParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dq = dq.data_ptr(); p.dbp = reinterpret_cast<__nv_bfloat16*>(dbp.data_ptr());
  const int64_t km = kpen.numel() > 0;
  p.kpen = nullptr; p.kps = 0;
  if (km) {
    TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && (A - 1) * kps + L <= kpen.numel() &&
                reinterpret_cast<uintptr_t>(kpen.data_ptr()) % 16 == 0 && kps % 4 == 0, "attn_bwd_dq: kpen fp32 [A][L] (16-byte aligned rows)");
    p.kpen = kpen.data_ptr<float>(); p.kps = kps;
  }
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddq = lddq; p.ss = ss; p.L = (int)L; p.H = (int)H;
  p.scl = (float)scl; p.bscl = (float)bscl; p.sm_scale = (float)sm_scale;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t ob = dq.scalar_type() == at::kBFloat16;
  const int64_t cfg = ((((hd * 10 + rows) * 10 + nkv) * 10 + minb) * 10 + ob) * 10 + km;
#define DQ_CFG(HD, R, NKV, MINB, OB, KM) \
  if (cfg == ((((int64_t(HD) * 10 + (R)) * 10 + (NKV)) * 10 + (MINB)) * 10 + (OB)) * 10 + (KM)) { launch_dq<aa80::DqCfg<HD, R, NKV, MINB, OB, KM>>(p, (int)A); return; }
  DQ_CFG(48, 4, 3, 1, 0, 0) DQ_CFG(48, 2, 4, 1, 0, 0) DQ_CFG(48, 1, 3, 2, 0, 0) DQ_CFG(48, 4, 3, 1, 1, 0) DQ_CFG(48, 2, 4, 1, 1, 0) DQ_CFG(48, 1, 3, 2, 1, 0)
  DQ_CFG(48, 2, 3, 1, 0, 0) DQ_CFG(48, 2, 3, 2, 0, 0) DQ_CFG(48, 1, 4, 2, 0, 0) DQ_CFG(48, 1, 3, 1, 0, 0)
  DQ_CFG(32, 4, 3, 1, 0, 0) DQ_CFG(32, 2, 4, 1, 0, 0) DQ_CFG(32, 1, 3, 2, 0, 0) DQ_CFG(32, 4, 3, 1, 1, 0) DQ_CFG(32, 2, 4, 1, 1, 0) DQ_CFG(32, 1, 3, 2, 1, 0)
  DQ_CFG(32, 6, 3, 1, 1, 0) DQ_CFG(32, 6, 3, 1, 0, 0)
  // a per-sample key mask: the module-level core's schedules (bf16 gradients)
  DQ_CFG(48, 4, 3, 1, 1, 1) DQ_CFG(48, 2, 4, 1, 1, 1) DQ_CFG(48, 1, 3, 2, 1, 1) DQ_CFG(32, 4, 3, 1, 1, 1) DQ_CFG(32, 2, 4, 1, 1, 1) DQ_CFG(32, 1, 3, 2, 1, 1)
#undef DQ_CFG
  TORCH_CHECK(false, "attn_bwd_dq: unsupported schedule (hd, rows, nkv, minb, kpen)");
}

// db [H, L, L] fp32 = scale x the sum of the `groups` partials dbp [groups, H, L, L] bf16 (fixed order).
void db_reduce(torch::Tensor dbp, torch::Tensor db, int64_t groups, double scale) {
  check_bf16(dbp, "db_reduce dbp");
  TORCH_CHECK(dbp.is_contiguous() && db.is_cuda() && db.scalar_type() == at::kFloat && db.is_contiguous() && dbp.numel() == groups * db.numel() && db.numel() % 8 == 0,
              "db_reduce: dbp [groups, ...] bf16 / db [...] fp32 contiguous");
  const long long plane8 = db.numel() / 8;
  const at::cuda::CUDAGuard guard(db.device());
  aa80::db_reduce_kernel<<<(unsigned)((plane8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dbp.data_ptr()), db.data_ptr<float>(), (int)groups, plane8, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The backward, key side (attn_bwd_dkv_sm80.cuh): as attn_bwd_dq plus bias_t [H][L][L] = the bias transposed (bias_t[h][k][j]) -> dk / dv [A L][ld] fp32 or bf16.
void attn_bwd_dkv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias_t, torch::Tensor lse, torch::Tensor delta,
                  torch::Tensor dk, torch::Tensor dv, int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo,
                  int64_t lddk, int64_t lddv, double scl, double bscl, double sm_scale, int64_t nst, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps) {
  for (auto* t : {&q, &k, &v, &dov, &bias_t}) check_bf16(*t, "attn_bwd_dkv");
  TORCH_CHECK(bias_t.is_contiguous() && bias_t.numel() == H * L * L, "attn_bwd_dkv: bias_t [H, L, L] contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A <= 65535 && H <= 65535, "attn_bwd_dkv: L a multiple of 128");
  if (ss <= 0) ss = L;
  TORCH_CHECK(ss >= L, "attn_bwd_dkv: ss >= L");
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_tokens(q, "attn_bwd_dkv q", T, ldq, cols); check_tokens(k, "attn_bwd_dkv k", T, ldk, cols); check_tokens(v, "attn_bwd_dkv v", T, ldv, cols);
  check_tokens(dov, "attn_bwd_dkv dov", T, lddo, cols);
  check_f32(lse, "attn_bwd_dkv lse", A * H * L); check_f32(delta, "attn_bwd_dkv delta", A * H * L);
  check_grad_view(dk, "attn_bwd_dkv dk", T, lddk, cols); check_grad_view(dv, "attn_bwd_dkv dv", T, lddv, cols);
  TORCH_CHECK(dk.scalar_type() == dv.scalar_type(), "attn_bwd_dkv: dk and dv in one dtype");
  aa80::DkvParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias_t = reinterpret_cast<const __nv_bfloat16*>(bias_t.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dk = dk.data_ptr(); p.dv = dv.data_ptr();
  const int64_t km = kpen.numel() > 0;
  p.kpen = nullptr; p.kps = 0;
  if (km) {
    TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && (A - 1) * kps + L <= kpen.numel(), "attn_bwd_dkv: kpen fp32 [A][L]");
    p.kpen = kpen.data_ptr<float>(); p.kps = kps;
  }
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddk = lddk; p.lddv = lddv; p.ss = ss; p.L = (int)L; p.H = (int)H;
  p.scl = (float)scl; p.bscl = (float)bscl; p.sm_scale = (float)sm_scale;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t ob = dk.scalar_type() == at::kBFloat16;
  const int64_t cfg = (((hd * 10 + nst) * 10 + minb) * 10 + ob) * 10 + km;
#define DKV_CFG(HD, NST, MINB, OB, KM) \
  if (cfg == (((int64_t(HD) * 10 + (NST)) * 10 + (MINB)) * 10 + (OB)) * 10 + (KM)) { launch_dkv<aa80::DkvCfg<HD, NST, MINB, OB, KM>>(p, (int)A); return; }
  DKV_CFG(48, 3, 2, 0, 0) DKV_CFG(48, 3, 2, 1, 0) DKV_CFG(48, 4, 2, 0, 0) DKV_CFG(48, 4, 1, 0, 0) DKV_CFG(48, 3, 1, 0, 0) DKV_CFG(48, 2, 2, 0, 0)
  DKV_CFG(32, 3, 2, 0, 0) DKV_CFG(32, 3, 2, 1, 0) DKV_CFG(32, 4, 2, 0, 0) DKV_CFG(32, 4, 2, 1, 0) DKV_CFG(32, 4, 1, 1, 0)
  DKV_CFG(48, 3, 2, 1, 1) DKV_CFG(32, 4, 2, 1, 1)
#undef DKV_CFG
  TORCH_CHECK(false, "attn_bwd_dkv: unsupported schedule (hd, nst, minb, kpen)");
}

// bias_t [H, L, L] = the transpose of each [L, L] plane of bias [H, L, L] (bf16).
void bias_transpose(torch::Tensor bias, torch::Tensor bias_t, int64_t L) {
  check_bf16(bias, "bias_transpose"); check_bf16(bias_t, "bias_transpose");
  TORCH_CHECK(bias.is_contiguous() && bias_t.is_contiguous() && bias.numel() == bias_t.numel() && L % 32 == 0 && bias.numel() % (L * L) == 0,
              "bias_transpose: contiguous [H, L, L], L a multiple of 32");
  const at::cuda::CUDAGuard guard(bias.device());
  dim3 grid((unsigned)(L / 32), (unsigned)(L / 32), (unsigned)(bias.numel() / (L * L)));
  aa80::bias_transpose_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), reinterpret_cast<__nv_bfloat16*>(bias_t.data_ptr()), (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// delta [A, H, L] fp32 = sum_d dO o per (sample, head, query): dO bf16 [A L][lddo], o bf16 or fp32 [A L][ldo] (head h = columns [hd h, hd h + hd)).
void attn_delta(torch::Tensor dO, torch::Tensor O, torch::Tensor delta, int64_t A, int64_t L, int64_t H, int64_t hd) {
  check_bf16(dO, "attn_delta dO");
  TORCH_CHECK(O.is_cuda() && (O.scalar_type() == at::kBFloat16 || O.scalar_type() == at::kFloat) && O.dim() == 2 && dO.dim() == 2 && O.size(1) >= H * hd
              && dO.size(1) >= H * hd && O.stride(1) == 1 && dO.stride(1) == 1 && O.size(0) == A * L && dO.size(0) == A * L, "attn_delta: [A L, >= H hd] operands");
  check_f32(delta, "attn_delta delta", A * H * L);
  const int64_t lddo = dO.stride(0), ldo = O.stride(0);
  TORCH_CHECK(lddo % 8 == 0 && ldo % 8 == 0 && reinterpret_cast<uintptr_t>(dO.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(O.data_ptr()) % 16 == 0,
              "attn_delta: row strides / alignment");
  const at::cuda::CUDAGuard guard(dO.device());
  const int64_t tokens = A * L;
  dim3 grid((unsigned)((tokens + 127) / 128), (unsigned)H);
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* d = reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr());
  const bool bf = O.scalar_type() == at::kBFloat16;
  if (hd == 32 && bf) aa80::attn_delta_kernel<32, __nv_bfloat16><<<grid, 128, 0, st>>>(d, reinterpret_cast<const __nv_bfloat16*>(O.data_ptr()), delta.data_ptr<float>(), lddo, ldo, (int)L, (int)H, tokens);
  else if (hd == 32) aa80::attn_delta_kernel<32, float><<<grid, 128, 0, st>>>(d, O.data_ptr<float>(), delta.data_ptr<float>(), lddo, ldo, (int)L, (int)H, tokens);
  else if (hd == 48 && bf) aa80::attn_delta_kernel<48, __nv_bfloat16><<<grid, 128, 0, st>>>(d, reinterpret_cast<const __nv_bfloat16*>(O.data_ptr()), delta.data_ptr<float>(), lddo, ldo, (int)L, (int)H, tokens);
  else if (hd == 48) aa80::attn_delta_kernel<48, float><<<grid, 128, 0, st>>>(d, O.data_ptr<float>(), delta.data_ptr<float>(), lddo, ldo, (int)L, (int)H, tokens);
  else TORCH_CHECK(false, "attn_delta: head dim 32 or 48");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
const unsigned char* mask_bytes(const torch::Tensor& kv, int64_t nz, const char* name) {
  if (kv.numel() == 0) return nullptr;
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == at::kBool && kv.is_contiguous() && kv.numel() == nz, name, ": kv a bool CUDA tensor [NZ]");
  return reinterpret_cast<const unsigned char*>(kv.data_ptr());
}
}  // namespace

// bias bo [4, N, N] (head-major; bf16 for bf16 z, fp32 for fp32 z) from z [NZ, NZ, 16] (bf16 or fp32) and w = gamma * Wb [4, 16] fp32; N a multiple of 64 >= NZ
// (the attention's length), kv a bool [NZ] key mask or an empty tensor.  A masked or padded key's column holds oscale * PB_MASKED (finite).
void pair_bias_fwd(torch::Tensor z, torch::Tensor w, torch::Tensor bo, torch::Tensor kv, int64_t N, int64_t NZ, double eps, double oscale) {
  TORCH_CHECK(z.is_cuda() && (z.scalar_type() == at::kBFloat16 || z.scalar_type() == at::kFloat) && bo.scalar_type() == z.scalar_type() && bo.is_cuda(),
              "pair_bias_fwd: z and bo CUDA tensors of one dtype (bf16 or fp32)");
  TORCH_CHECK(z.is_contiguous() && z.numel() == NZ * NZ * aa80::PB_C && NZ <= N && N % 64 == 0, "pair_bias_fwd: z [NZ, NZ, 16] contiguous, N a multiple of 64 >= NZ");
  check_f32(w, "pair_bias_fwd w", aa80::PB_H * aa80::PB_C);
  TORCH_CHECK(bo.is_contiguous() && bo.numel() == aa80::PB_H * N * N, "pair_bias_fwd: bo [4, N, N] contiguous");
  const unsigned char* kvp = mask_bytes(kv, NZ, "pair_bias_fwd");
  const at::cuda::CUDAGuard guard(z.device());
  dim3 grid((unsigned)(N / aa80::PB_TJ), (unsigned)(N / aa80::PB_TI));
  auto st = at::cuda::getCurrentCUDAStream();
  const float fill = (float)(aa80::PB_MASKED * oscale);
#define PB_FWD(T, TP) \
  { \
    const auto* zp = reinterpret_cast<const TP*>(z.data_ptr()); auto* bp = reinterpret_cast<TP*>(bo.data_ptr()); \
    if (N == NZ) aa80::pair_bias_fwd_kernel<false, TP, TP><<<grid, aa80::PB_NT, 0, st>>>(zp, w.data_ptr<float>(), bp, (int)N, (int)NZ, kvp, (float)eps, (float)oscale, fill); \
    else aa80::pair_bias_fwd_kernel<true, TP, TP><<<grid, aa80::PB_NT, 0, st>>>(zp, w.data_ptr<float>(), bp, (int)N, (int)NZ, kvp, (float)eps, (float)oscale, fill); \
  }
  if (z.scalar_type() == at::kBFloat16) PB_FWD(bf16, __nv_bfloat16) else PB_FWD(f32, float)
#undef PB_FWD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
template <int H, int R, bool PAD, typename T>
void launch_gen(const torch::Tensor& z, const torch::Tensor& wcm, torch::Tensor& bo, const unsigned char* kvp, int64_t N, int64_t NZ, int64_t C, double eps, double oscale, float fill) {
  const size_t smem = ((size_t)C * H + H) * sizeof(float);
  TORCH_CHECK(smem <= 160 * 1024, "pair_bias_gen: C x H too large for shared memory");
  static bool set[64] = {};                                   // the attribute is per device (and per instantiation: a static of the template)
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::pair_bias_gen_kernel<H, R, PAD, T, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
    set[dev] = true;
  }
  dim3 grid((unsigned)(N / aa80::PB_TJ), (unsigned)(N / (4 * R)));
  aa80::pair_bias_gen_kernel<H, R, PAD, T, T><<<grid, aa80::PB_NT, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(z.data_ptr()), wcm.data_ptr<float>(), reinterpret_cast<T*>(bo.data_ptr()), (int)N, (int)NZ, (int)C, kvp, (float)eps, (float)oscale, fill);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int R, bool PAD, typename T>
void launch_gen_h(int64_t H, const torch::Tensor& z, const torch::Tensor& wcm, torch::Tensor& bo, const unsigned char* kvp, int64_t N, int64_t NZ, int64_t C, double eps, double oscale, float fill) {
#define GEN_H(HH) if (H == HH) { launch_gen<HH, R, PAD, T>(z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill); return; }
  GEN_H(4) GEN_H(8) GEN_H(12) GEN_H(16) GEN_H(24) GEN_H(32)
#undef GEN_H
  TORCH_CHECK(false, "pair_bias_gen: heads 4, 8, 12, 16, 24 or 32");
}

template <bool PAD, typename T>
void launch_gen_r(int64_t rows, int64_t H, const torch::Tensor& z, const torch::Tensor& wcm, torch::Tensor& bo, const unsigned char* kvp, int64_t N, int64_t NZ, int64_t C, double eps, double oscale, float fill) {
  if (rows == 4) launch_gen_h<4, PAD, T>(H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
  else if (rows == 1) launch_gen_h<1, PAD, T>(H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
  else launch_gen_h<2, PAD, T>(H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
}
}  // namespace

// The pair bias at any pair width: bo [H, N, N] (z's dtype: bf16 or fp32) = oscale rstd (z . W' - mean sum W') from z [NZ, NZ, C] (C a multiple of 8) and wcm = W' channel-major [C, H] fp32
// (H = 4, 8, 12, 16, 24, 32); the key mask kv, the padding N > NZ and the fill as ``pair_bias_fwd``.  ``rows`` (1, 2 or 4) is the rows a thread takes (the weights' shared-memory reads are shared by them).
void pair_bias_gen(torch::Tensor z, torch::Tensor wcm, torch::Tensor bo, torch::Tensor kv, int64_t N, int64_t NZ, double eps, double oscale, int64_t rows) {
  TORCH_CHECK(z.is_cuda() && (z.scalar_type() == at::kBFloat16 || z.scalar_type() == at::kFloat) && bo.is_cuda() && bo.scalar_type() == z.scalar_type() && z.is_contiguous() && bo.is_contiguous(),
              "pair_bias_gen: z and bo contiguous CUDA tensors of one dtype (bf16 or fp32)");
  const int64_t C = z.numel() / (NZ * NZ), H = wcm.size(1);
  TORCH_CHECK(C % 8 == 0 && z.numel() == NZ * NZ * C && NZ <= N && N % 64 == 0 && wcm.dim() == 2 && wcm.size(0) == C && wcm.scalar_type() == at::kFloat && wcm.is_contiguous() && wcm.is_cuda(),
              "pair_bias_gen: z [NZ, NZ, C] with C a multiple of 8, wcm fp32 [C, H], N a multiple of 64 >= NZ");
  TORCH_CHECK(bo.numel() == H * N * N, "pair_bias_gen: bo [H, N, N]");
  const unsigned char* kvp = mask_bytes(kv, NZ, "pair_bias_gen");
  const at::cuda::CUDAGuard guard(z.device());
  const float fill = (float)(aa80::PB_MASKED * oscale);
  if (z.scalar_type() == at::kBFloat16) {
    if (N == NZ) launch_gen_r<false, __nv_bfloat16>(rows, H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
    else launch_gen_r<true, __nv_bfloat16>(rows, H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
  } else {
    if (N == NZ) launch_gen_r<false, float>(rows, H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
    else launch_gen_r<true, float>(rows, H, z, wcm, bo, kvp, N, NZ, C, eps, oscale, fill);
  }
}

namespace {
template <int H, int C, typename T>
void launch_gen_bwd(const torch::Tensor& z, const torch::Tensor& wcm, const torch::Tensor& db, torch::Tensor& dz, torch::Tensor& pw, const unsigned char* kvp, int64_t N, int64_t NZ, int64_t rows,
                    double eps) {
  using G = aa80::PbBwdCfg<C>;
  const size_t smem = (size_t)G::smem(H);
  TORCH_CHECK(smem <= 160 * 1024, "pair_bias_gen_bwd: C x H too large for shared memory");
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::pair_bias_gen_bwd_kernel<H, C, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
    set[dev] = true;
  }
  dim3 grid((unsigned)((NZ + G::TE - 1) / G::TE), (unsigned)((NZ + rows - 1) / rows));
  TORCH_CHECK(pw.numel() >= (int64_t)grid.x * grid.y * H * C, "pair_bias_gen_bwd: pw needs ", (int64_t)grid.x * grid.y * H * C, " floats");
  aa80::pair_bias_gen_bwd_kernel<H, C, T><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(z.data_ptr()), wcm.data_ptr<float>(), db.data_ptr<float>(), reinterpret_cast<T*>(dz.data_ptr()), pw.data_ptr<float>(), (int)N, (int)NZ, (int)rows, kvp, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int C, typename T>
void launch_gen_bwd_h(int64_t H, const torch::Tensor& z, const torch::Tensor& wcm, const torch::Tensor& db, torch::Tensor& dz, torch::Tensor& pw, const unsigned char* kvp, int64_t N, int64_t NZ,
                      int64_t rows, double eps) {
#define GBWD_H(HH) if (H == HH) { launch_gen_bwd<HH, C, T>(z, wcm, db, dz, pw, kvp, N, NZ, rows, eps); return; }
  GBWD_H(8) GBWD_H(12) GBWD_H(16) GBWD_H(24)
#undef GBWD_H
  TORCH_CHECK(false, "pair_bias_gen_bwd: heads 8, 12, 16 or 24");
}

template <typename T>
void launch_gen_bwd_c(int64_t C, int64_t H, const torch::Tensor& z, const torch::Tensor& wcm, const torch::Tensor& db, torch::Tensor& dz, torch::Tensor& pw, const unsigned char* kvp, int64_t N,
                      int64_t NZ, int64_t rows, double eps) {
  if (C == 64) launch_gen_bwd_h<64, T>(H, z, wcm, db, dz, pw, kvp, N, NZ, rows, eps);
  else if (C == 128) launch_gen_bwd_h<128, T>(H, z, wcm, db, dz, pw, kvp, N, NZ, rows, eps);
  else if (C == 256) launch_gen_bwd_h<256, T>(H, z, wcm, db, dz, pw, kvp, N, NZ, rows, eps);
  else if (C == 512) launch_gen_bwd_h<512, T>(H, z, wcm, db, dz, pw, kvp, N, NZ, rows, eps);
  else TORCH_CHECK(false, "pair_bias_gen_bwd: C 64, 128, 256 or 512");
}
}  // namespace

// The backward of ``pair_bias_gen``: dz [NZ, NZ, C] (z's dtype) and the per-block partials pw [grid.x grid.y, H, C] fp32 of dW' (grid.x = ceil(NZ / TE), TE = 128 / 64 / 32 / 16 for C = 64 / 128 /
// 256 / 512; grid.y = ceil(NZ / rows)) from z, wcm = W' channel-major [C, H] fp32 and db [H, N, N] fp32 (the bias gradient in natural units, summed over the samples); the key mask kv drops
// dbias on masked keys.  ``rows`` rows per block (the weight gradient is accumulated across them).
void pair_bias_gen_bwd(torch::Tensor z, torch::Tensor wcm, torch::Tensor db, torch::Tensor dz, torch::Tensor pw, torch::Tensor kv, int64_t N, int64_t NZ, double eps, int64_t rows) {
  TORCH_CHECK(z.is_cuda() && (z.scalar_type() == at::kBFloat16 || z.scalar_type() == at::kFloat) && dz.is_cuda() && dz.scalar_type() == z.scalar_type() && z.is_contiguous() && dz.is_contiguous()
              && dz.numel() == z.numel(), "pair_bias_gen_bwd: z and dz contiguous CUDA tensors of one dtype (bf16 or fp32)");
  const int64_t C = z.numel() / (NZ * NZ), H = wcm.size(1);
  TORCH_CHECK(C % 64 == 0 && z.numel() == NZ * NZ * C && NZ <= N && N % 64 == 0 && rows >= 1 && wcm.dim() == 2 && wcm.size(0) == C && wcm.scalar_type() == at::kFloat && wcm.is_contiguous() && wcm.is_cuda(),
              "pair_bias_gen_bwd: z [NZ, NZ, C] with C a multiple of 64, wcm fp32 [C, H]");
  check_f32(db, "pair_bias_gen_bwd db", H * N * N);
  TORCH_CHECK(pw.is_cuda() && pw.scalar_type() == at::kFloat && pw.is_contiguous(), "pair_bias_gen_bwd: pw fp32");
  const unsigned char* kvp = mask_bytes(kv, NZ, "pair_bias_gen_bwd");
  const at::cuda::CUDAGuard guard(z.device());
  if (z.scalar_type() == at::kBFloat16) launch_gen_bwd_c<__nv_bfloat16>(C, H, z, wcm, db, dz, pw, kvp, N, NZ, rows, eps);
  else launch_gen_bwd_c<float>(C, H, z, wcm, db, dz, pw, kvp, N, NZ, rows, eps);
}

// dz [NZ, NZ, 16] (z's dtype) and the per-block partials pw [(N / 64) (N / 32), 4, 16] fp32 of d(W') from db [4, N, N] fp32 (summed over the samples).
void pair_bias_bwd(torch::Tensor z, torch::Tensor w, torch::Tensor db, torch::Tensor dz, torch::Tensor pw, torch::Tensor kv, int64_t N, int64_t NZ, double eps) {
  TORCH_CHECK(z.is_cuda() && (z.scalar_type() == at::kBFloat16 || z.scalar_type() == at::kFloat) && dz.scalar_type() == z.scalar_type() && dz.is_cuda(),
              "pair_bias_bwd: z and dz CUDA tensors of one dtype (bf16 or fp32)");
  TORCH_CHECK(z.is_contiguous() && dz.is_contiguous() && z.numel() == NZ * NZ * aa80::PB_C && dz.numel() == z.numel() && NZ <= N && N % 64 == 0,
              "pair_bias_bwd: z / dz [NZ, NZ, 16] contiguous, N a multiple of 64 >= NZ");
  check_f32(w, "pair_bias_bwd w", aa80::PB_H * aa80::PB_C);
  check_f32(db, "pair_bias_bwd db", aa80::PB_H * N * N);
  const int64_t nb = (N / aa80::PB_TJ) * (N / aa80::PB_TI);
  check_f32(pw, "pair_bias_bwd pw", nb * aa80::PB_H * aa80::PB_C);
  const unsigned char* kvp = mask_bytes(kv, NZ, "pair_bias_bwd");
  const at::cuda::CUDAGuard guard(z.device());
  dim3 grid((unsigned)(N / aa80::PB_TJ), (unsigned)(N / aa80::PB_TI));
  auto st = at::cuda::getCurrentCUDAStream();
  if (z.scalar_type() == at::kBFloat16)
    aa80::pair_bias_bwd_kernel<__nv_bfloat16><<<grid, aa80::PB_NT, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()), w.data_ptr<float>(), db.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
        pw.data_ptr<float>(), (int)N, (int)NZ, kvp, (float)eps);
  else
    aa80::pair_bias_bwd_kernel<float><<<grid, aa80::PB_NT, 0, st>>>(z.data_ptr<float>(), w.data_ptr<float>(), db.data_ptr<float>(), dz.data_ptr<float>(),
                                                                   pw.data_ptr<float>(), (int)N, (int)NZ, kvp, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_fwd", &attn_fwd);
  m.def("attn_bwd_dq", &attn_bwd_dq);
  m.def("db_reduce", &db_reduce);
  m.def("attn_bwd_dkv", &attn_bwd_dkv);
  m.def("bias_transpose", &bias_transpose);
  m.def("attn_delta", &attn_delta);
  m.def("pair_bias_fwd", &pair_bias_fwd);
  m.def("pair_bias_gen", &pair_bias_gen);
  m.def("pair_bias_gen_bwd", &pair_bias_gen_bwd);
  m.def("pair_bias_bwd", &pair_bias_bwd);
}
