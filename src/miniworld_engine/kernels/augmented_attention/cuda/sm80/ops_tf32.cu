// ops_tf32.cu -- torch bindings of the A100 (sm_80) TF32 attention-with-shared-bias kernels (fp32 operands, TF32 tensor cores; head dim 32 or 48): a third extension of the family
// (the bf16 kernels are ops.cu, the gates glue_ops.cu).
//
//   attn_fwd_tf32      attn_fwd_tf32_sm80.cuh       o (or sigmoid(g) o) fp32 and the log-sum-exp
//   attn_bwd_dq_tf32   attn_bwd_dq_tf32_sm80.cuh    dq and the fp32 bias-gradient partials of a chunk of query tiles;  db_reduce32: their fixed-order sum
//   attn_bwd_dkv_tf32  attn_bwd_dkv_tf32_sm80.cuh   dk, dv;  bias_transpose32: the bias transposed for it;  attn_delta32: the backward's row term sum_d dO o
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_bwd_dkv_tf32_sm80.cuh"

namespace {

void check_f32t(const torch::Tensor& t, const char* name, int64_t T, int64_t ld, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) == ld && ld >= cols && ld % 4 == 0, name, ": a fp32 CUDA view [T][ld], ld a multiple of 4");
  TORCH_CHECK((t.storage_offset() + (T - 1) * ld + cols) * 4 <= (int64_t)t.storage().nbytes() && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": extents / alignment");
}

void check_f32v(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous() && t.numel() == n && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              ": a contiguous fp32 CUDA tensor of ", n, " elements");
}

const float* penalties(const torch::Tensor& kpen, int64_t kps, int64_t A, int64_t L, const char* name) {
  if (kpen.numel() == 0) return nullptr;
  TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && kps % 4 == 0 && (A - 1) * kps + L <= kpen.numel() &&
              reinterpret_cast<uintptr_t>(kpen.data_ptr()) % 16 == 0, name, ": kpen fp32 [A][L] (16-byte aligned rows)");
  return kpen.data_ptr<float>();
}

template <class G>
void launch_fwd32(const aa80::Fwd32Params& p, int A) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_fwd_tf32_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / aa80::BM, A, p.H);
  aa80::attn_fwd_tf32_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_dq32(const aa80::Dq32Params& p, int A, int nqt) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dq_tf32_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(nqt, A / G::R, p.H);
  aa80::attn_bwd_dq_tf32_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_dkv32(const aa80::Dkv32Params& p, int A) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dkv_tf32_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / aa80::BM, A, p.H);
  aa80::attn_bwd_dkv_tf32_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// q / k / v / gate: token-major fp32 views [A L][ld] (head h = columns [hd h, hd h + hd)), bias [H][L][L] fp32 natural units (masked keys -inf).  gate: g (sigmoid(g) o is written to
// ``out``, which may be q; lse is not written) or an empty tensor (o is written and lse [A][H][L]).  ss: tokens between samples (<= 0: L); kpen: per-sample key penalties.
void attn_fwd_tf32(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor gate, torch::Tensor out, torch::Tensor lse, int64_t A, int64_t L, int64_t H,
                   int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t ldg, int64_t ldo, double scl, double bscl, int64_t nkv, int64_t minb, int64_t ss, torch::Tensor kpen,
                   int64_t kps) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == H * L * L && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "attn_fwd_tf32: bias [H, L, L] fp32 contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A <= 65535 && H <= 65535, "attn_fwd_tf32: L a multiple of 128");
  if (ss <= 0) ss = L;
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_f32t(q, "attn_fwd_tf32 q", T, ldq, cols); check_f32t(k, "attn_fwd_tf32 k", T, ldk, cols); check_f32t(v, "attn_fwd_tf32 v", T, ldv, cols);
  check_f32t(out, "attn_fwd_tf32 out", T, ldo, cols);
  const bool gated = gate.numel() > 0;
  aa80::Fwd32Params p;
  p.q = q.data_ptr<float>(); p.k = k.data_ptr<float>(); p.v = v.data_ptr<float>(); p.bias = bias.data_ptr<float>(); p.out = out.data_ptr<float>();
  p.gate = nullptr; p.lse = nullptr;
  if (gated) { check_f32t(gate, "attn_fwd_tf32 gate", T, ldg, cols); p.gate = gate.data_ptr<float>(); }
  else { check_f32v(lse, "attn_fwd_tf32 lse", A * H * L); p.lse = lse.data_ptr<float>(); }
  p.kpen = penalties(kpen, kps, A, L, "attn_fwd_tf32"); p.kps = p.kpen ? kps : 0;
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.ldg = ldg; p.ldo = ldo; p.ss = ss; p.L = (int)L; p.H = (int)H; p.scl = (float)scl; p.bscl = (float)bscl;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t km = p.kpen != nullptr, cfg = ((((hd * 10 + nkv) * 10 + minb) * 10 + (gated ? 1 : 0)) * 10 + km);
#define FWD32(HD, NKV, MINB, GATE, KM) \
  if (cfg == ((((int64_t(HD) * 10 + (NKV)) * 10 + (MINB)) * 10 + (GATE)) * 10 + (KM))) { launch_fwd32<aa80::Fwd32Cfg<HD, NKV, MINB, GATE, KM>>(p, (int)A); return; }
  FWD32(32, 2, 2, 0, 0) FWD32(32, 2, 2, 1, 0) FWD32(32, 2, 2, 0, 1) FWD32(32, 2, 2, 1, 1)
  FWD32(32, 3, 1, 0, 0) FWD32(32, 3, 1, 1, 0) FWD32(32, 3, 1, 0, 1) FWD32(32, 3, 1, 1, 1)
  FWD32(48, 2, 1, 0, 0) FWD32(48, 2, 1, 1, 0) FWD32(48, 2, 1, 0, 1) FWD32(48, 2, 1, 1, 1)
  FWD32(48, 3, 1, 0, 0) FWD32(48, 3, 1, 1, 0) FWD32(48, 3, 1, 0, 1) FWD32(48, 3, 1, 1, 1)
#undef FWD32
  TORCH_CHECK(false, "attn_fwd_tf32: unsupported schedule (hd, nkv, minb, gate, kpen)");
}

// The backward, query side: q / k / v / dov token-major fp32 views, bias [H][L][L] fp32 natural units, lse / delta [A][H][L] -> dq [A L][lddq] fp32 and the partial bias gradients
// dbp [A / rows][H][nqt 128][L] fp32 of the query tiles qt0 .. qt0 + nqt - 1 (``db_reduce32`` sums them into db's rows of those tiles).
void attn_bwd_dq_tf32(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor lse, torch::Tensor delta, torch::Tensor dq,
                      torch::Tensor dbp, int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo, int64_t lddq, double scl, double bscl,
                      double sm_scale, int64_t rows, int64_t nkv, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps, int64_t qt0, int64_t nqt) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == H * L * L && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "attn_bwd_dq_tf32: bias [H, L, L] fp32 contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A % rows == 0 && A / rows <= 65535 && H <= 65535 && qt0 >= 0 && nqt >= 1 && (qt0 + nqt) * aa80::BM <= L,
              "attn_bwd_dq_tf32: L a multiple of 128, A a multiple of rows, a valid query-tile range");
  if (ss <= 0) ss = L;
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_f32t(q, "attn_bwd_dq_tf32 q", T, ldq, cols); check_f32t(k, "attn_bwd_dq_tf32 k", T, ldk, cols); check_f32t(v, "attn_bwd_dq_tf32 v", T, ldv, cols);
  check_f32t(dov, "attn_bwd_dq_tf32 dov", T, lddo, cols); check_f32t(dq, "attn_bwd_dq_tf32 dq", T, lddq, cols);
  check_f32v(lse, "attn_bwd_dq_tf32 lse", A * H * L); check_f32v(delta, "attn_bwd_dq_tf32 delta", A * H * L);
  check_f32v(dbp, "attn_bwd_dq_tf32 dbp", (A / rows) * H * nqt * aa80::BM * L);
  aa80::Dq32Params p;
  p.q = q.data_ptr<float>(); p.k = k.data_ptr<float>(); p.v = v.data_ptr<float>(); p.dov = dov.data_ptr<float>(); p.bias = bias.data_ptr<float>();
  p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>(); p.dq = dq.data_ptr<float>(); p.dbp = dbp.data_ptr<float>();
  p.kpen = penalties(kpen, kps, A, L, "attn_bwd_dq_tf32"); p.kps = p.kpen ? kps : 0;
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddq = lddq; p.ss = ss; p.L = (int)L; p.H = (int)H; p.qt0 = (int)qt0; p.rows_chunk = (int)(nqt * aa80::BM);
  p.scl = (float)scl; p.bscl = (float)bscl; p.sm_scale = (float)sm_scale;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t km = p.kpen != nullptr, cfg = (((hd * 10 + rows) * 10 + nkv) * 10 + minb) * 10 + km;
#define DQ32(HD, R, NKV, MINB, KM) \
  if (cfg == (((int64_t(HD) * 10 + (R)) * 10 + (NKV)) * 10 + (MINB)) * 10 + (KM)) { launch_dq32<aa80::Dq32Cfg<HD, R, NKV, MINB, KM>>(p, (int)A, (int)nqt); return; }
  DQ32(32, 1, 2, 1, 0) DQ32(32, 2, 2, 1, 0) DQ32(48, 1, 2, 1, 0) DQ32(32, 1, 2, 1, 1) DQ32(32, 2, 2, 1, 1) DQ32(48, 1, 2, 1, 1)
#undef DQ32
  TORCH_CHECK(false, "attn_bwd_dq_tf32: unsupported schedule (hd, rows, nkv, minb, kpen)");
}

// db [H][L][L] rows r0 .. r0 + rows_chunk of every plane = scale x the sum of the `groups` partials dbp [groups][H][rows_chunk][L] (fixed order).
void db_reduce32(torch::Tensor dbp, torch::Tensor db, int64_t groups, int64_t H, int64_t rows_chunk, int64_t L, int64_t r0, double scale) {
  TORCH_CHECK(dbp.is_cuda() && dbp.scalar_type() == at::kFloat && dbp.is_contiguous() && dbp.numel() == groups * H * rows_chunk * L && L % 8 == 0, "db_reduce32: dbp [groups, H, rows, L] fp32");
  TORCH_CHECK(db.is_cuda() && db.scalar_type() == at::kFloat && db.is_contiguous() && db.numel() == H * L * L && r0 >= 0 && r0 + rows_chunk <= L, "db_reduce32: db [H, L, L] fp32");
  const long long plane8 = H * rows_chunk * L / 8;
  const at::cuda::CUDAGuard guard(db.device());
  aa80::db_reduce32_kernel<<<(unsigned)((plane8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(dbp.data_ptr<float>(), db.data_ptr<float>(), (int)groups, (int)H, (int)rows_chunk,
                                                                                                          (int)L, (int)r0, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The backward, key side: as attn_bwd_dq_tf32 plus bias_t [H][L][L] = the bias transposed -> dk / dv [A L][ld] fp32.
void attn_bwd_dkv_tf32(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias_t, torch::Tensor lse, torch::Tensor delta, torch::Tensor dk,
                       torch::Tensor dv, int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo, int64_t lddk, int64_t lddv, double scl,
                       double bscl, double sm_scale, int64_t nst, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps) {
  TORCH_CHECK(bias_t.is_cuda() && bias_t.scalar_type() == at::kFloat && bias_t.is_contiguous() && bias_t.numel() == H * L * L && reinterpret_cast<uintptr_t>(bias_t.data_ptr()) % 16 == 0,
              "attn_bwd_dkv_tf32: bias_t [H, L, L] fp32 contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A <= 65535 && H <= 65535, "attn_bwd_dkv_tf32: L a multiple of 128");
  if (ss <= 0) ss = L;
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_f32t(q, "attn_bwd_dkv_tf32 q", T, ldq, cols); check_f32t(k, "attn_bwd_dkv_tf32 k", T, ldk, cols); check_f32t(v, "attn_bwd_dkv_tf32 v", T, ldv, cols);
  check_f32t(dov, "attn_bwd_dkv_tf32 dov", T, lddo, cols); check_f32t(dk, "attn_bwd_dkv_tf32 dk", T, lddk, cols); check_f32t(dv, "attn_bwd_dkv_tf32 dv", T, lddv, cols);
  check_f32v(lse, "attn_bwd_dkv_tf32 lse", A * H * L); check_f32v(delta, "attn_bwd_dkv_tf32 delta", A * H * L);
  aa80::Dkv32Params p;
  p.q = q.data_ptr<float>(); p.k = k.data_ptr<float>(); p.v = v.data_ptr<float>(); p.dov = dov.data_ptr<float>(); p.bias_t = bias_t.data_ptr<float>();
  p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>(); p.dk = dk.data_ptr<float>(); p.dv = dv.data_ptr<float>();
  p.kpen = penalties(kpen, kps, A, L, "attn_bwd_dkv_tf32"); p.kps = p.kpen ? kps : 0;
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddk = lddk; p.lddv = lddv; p.ss = ss; p.L = (int)L; p.H = (int)H;
  p.scl = (float)scl; p.bscl = (float)bscl; p.sm_scale = (float)sm_scale;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t km = p.kpen != nullptr, cfg = ((hd * 10 + nst) * 10 + minb) * 10 + km;
#define DKV32(HD, NST, MINB, KM) \
  if (cfg == ((int64_t(HD) * 10 + (NST)) * 10 + (MINB)) * 10 + (KM)) { launch_dkv32<aa80::Dkv32Cfg<HD, NST, MINB, KM>>(p, (int)A); return; }
  DKV32(32, 2, 1, 0) DKV32(48, 2, 1, 0) DKV32(32, 2, 1, 1) DKV32(48, 2, 1, 1) DKV32(32, 3, 1, 0)
#undef DKV32
  TORCH_CHECK(false, "attn_bwd_dkv_tf32: unsupported schedule (hd, nst, minb, kpen)");
}

// bias_t [H, L, L] = the transpose of each [L, L] plane of bias [H, L, L] (fp32).
void bias_transpose32(torch::Tensor bias, torch::Tensor bias_t, int64_t L) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias_t.is_cuda() && bias_t.scalar_type() == at::kFloat && bias.is_contiguous() && bias_t.is_contiguous() &&
              bias.numel() == bias_t.numel() && L % 32 == 0 && bias.numel() % (L * L) == 0, "bias_transpose32: contiguous fp32 [H, L, L], L a multiple of 32");
  const at::cuda::CUDAGuard guard(bias.device());
  dim3 grid((unsigned)(L / 32), (unsigned)(L / 32), (unsigned)(bias.numel() / (L * L)));
  aa80::bias_transpose32_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(bias.data_ptr<float>(), bias_t.data_ptr<float>(), (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// delta [A, H, L] fp32 = sum_d dO o per (sample, head, query) for fp32 dO / o views [A L][ld].
void attn_delta32(torch::Tensor dO, torch::Tensor O, torch::Tensor delta, int64_t A, int64_t L, int64_t H, int64_t hd) {
  const int64_t T = A * L, cols = H * hd;
  check_f32t(dO, "attn_delta32 dO", T, dO.stride(0), cols); check_f32t(O, "attn_delta32 O", T, O.stride(0), cols);
  check_f32v(delta, "attn_delta32 delta", A * H * L);
  const at::cuda::CUDAGuard guard(dO.device());
  dim3 grid((unsigned)((T + 127) / 128), (unsigned)H);
  auto st = at::cuda::getCurrentCUDAStream();
  if (hd == 32) aa80::attn_delta32_kernel<32><<<grid, 128, 0, st>>>(dO.data_ptr<float>(), O.data_ptr<float>(), delta.data_ptr<float>(), dO.stride(0), O.stride(0), (int)L, (int)H, T);
  else if (hd == 48) aa80::attn_delta32_kernel<48><<<grid, 128, 0, st>>>(dO.data_ptr<float>(), O.data_ptr<float>(), delta.data_ptr<float>(), dO.stride(0), O.stride(0), (int)L, (int)H, T);
  else TORCH_CHECK(false, "attn_delta32: head dim 32 or 48");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_fwd_tf32", &attn_fwd_tf32);
  m.def("attn_bwd_dq_tf32", &attn_bwd_dq_tf32);
  m.def("db_reduce32", &db_reduce32);
  m.def("attn_bwd_dkv_tf32", &attn_bwd_dkv_tf32);
  m.def("bias_transpose32", &bias_transpose32);
  m.def("attn_delta32", &attn_delta32);
}
