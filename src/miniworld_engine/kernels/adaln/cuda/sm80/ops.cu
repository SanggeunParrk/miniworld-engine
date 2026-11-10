// ops.cu -- torch bindings of the A100 (sm_80) AdaLN kernels: the row passes (adaln_rows.cuh).  Built on first use by kernels/adaln/cuda/sm80.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <type_traits>

#include "adaln_atom_bwd.cuh"
#include "adaln_atom_fwd.cuh"
#include "adaln_atom_fwd_tf32.cuh"
#include "adaln_finish.cuh"
#include "adaln_launch.cuh"
#include "adaln_rows.cuh"

namespace {
using namespace adl;


template <typename T> T* P_(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* CP_(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> const T* OP_(const c10::optional<at::Tensor>& t) { return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr; }
cudaStream_t S_() { return at::cuda::getCurrentCUDAStream(); }

// row widths the row kernels take (a warp per row: 1, 3 or 6 vectors of 4 elements a lane)
template <typename F> void with_width(int64_t width, const char* what, F&& f) {
  switch (width) {
    case 128: f(std::integral_constant<int, 128>{}); break;
    case 384: f(std::integral_constant<int, 384>{}); break;
    case 768: f(std::integral_constant<int, 768>{}); break;
    default: TORCH_CHECK(false, what, ": row width 128, 384 or 768, got ", width);
  }
}
template <typename F> void with_dtype(const at::Tensor& t, const char* what, F&& f) {
  if (t.scalar_type() == at::kFloat) f(float{});
  else if (t.scalar_type() == at::kBFloat16) f(bf{});
  else TORCH_CHECK(false, what, ": fp32 or bf16 rows");
}

void check_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1, name, ": a CUDA [rows, ", cols, "] view with unit column stride");
  const int64_t es = t.element_size();
  TORCH_CHECK(t.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % (4 * es) == 0, name, ": rows must start on a ", 4 * es, "-byte boundary");
}
void check_vec(const at::Tensor& t, const char* name, int64_t n, at::ScalarType dt) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.numel() == n && t.scalar_type() == dt, name, ": a contiguous CUDA vector of ", n, " (", dt, ")");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

// aff [P, dc] = LN(cond) w (dtype of aff), st [P, 2] fp32 (optional)
void cond_ln(at::Tensor cond, at::Tensor w, at::Tensor aff, c10::optional<at::Tensor> st, double eps) {
  const int64_t P = cond.size(0), dc = cond.size(1);
  check_rows(cond, "cond_ln cond", dc);
  check_rows(aff, "cond_ln aff", dc);
  TORCH_CHECK(aff.is_contiguous() && aff.size(0) == P, "cond_ln: aff [P, dc] contiguous");
  check_vec(w, "cond_ln w", dc, at::kFloat);
  if (st) TORCH_CHECK(st->is_cuda() && st->is_contiguous() && st->scalar_type() == at::kFloat && st->numel() == 2 * P, "cond_ln: st [P, 2] fp32");
  const at::cuda::CUDAGuard g(cond.device());
  with_width(dc, "cond_ln", [&](auto wd) {
    constexpr int D = decltype(wd)::value;
    TORCH_CHECK(aff.scalar_type() == cond.scalar_type(), "cond_ln: aff in cond's dtype");
    with_dtype(cond, "cond_ln cond", [&](auto ct) {
      using T = decltype(ct);
      cond_ln_kernel<D, T><<<(unsigned)((P + RW - 1) / RW), RW * 32, 0, S_()>>>(CP_<T>(cond), cond.stride(0), CP_<float>(w), P_<T>(aff), st ? P_<float2>(*st) : nullptr, P, (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y [M, d] = sigmoid(S + sb) LN(x) + B; sbt [P, 2 d] = S | B (row r of x reads row r % P); sb [d] (x's dtype); xst [M, 2] fp32 (optional)
void adaln_epi(at::Tensor x, at::Tensor sbt, at::Tensor sb, at::Tensor y, c10::optional<at::Tensor> xst, int64_t period, double eps) {
  const int64_t M = x.size(0), d = x.size(1);
  check_rows(x, "adaln_epi x", d);
  check_rows(sbt, "adaln_epi sbt", 2 * d);
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.size(0) == M && y.size(1) == d && y.scalar_type() == x.scalar_type(), "adaln_epi: y [M, d] like x");
  TORCH_CHECK(sb.is_cuda() && sb.is_contiguous() && sb.numel() == d && sb.scalar_type() == x.scalar_type(), "adaln_epi: sb [d] like x");
  TORCH_CHECK(period > 0 && M % period == 0 && sbt.size(0) == period, "adaln_epi: the table has `period` rows and M is a multiple of it");
  if (xst) TORCH_CHECK(xst->is_cuda() && xst->is_contiguous() && xst->scalar_type() == at::kFloat && xst->numel() == 2 * M, "adaln_epi: xst [M, 2] fp32");
  const at::cuda::CUDAGuard g(x.device());
  with_width(d, "adaln_epi", [&](auto wd) {
    constexpr int D = decltype(wd)::value;
    TORCH_CHECK(sbt.scalar_type() == x.scalar_type(), "adaln_epi: sbt in x's dtype");
    with_dtype(x, "adaln_epi x", [&](auto xt) {
      using XT = decltype(xt);
      adaln_epi_kernel<D, XT><<<(unsigned)((M + RW - 1) / RW), RW * 32, 0, S_()>>>(
          CP_<XT>(x), x.stride(0), CP_<XT>(sbt), sbt.stride(0), CP_<XT>(sb), P_<XT>(y), xst ? P_<float2>(*xst) : nullptr, M, period, (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dm [M, 2 d] = dscale | dy (dtype of dm), dx [M, d] (+ dres), psb [blocks, d] fp32 = the column sums of dscale, one row per block
void adaln_bwd_x(at::Tensor dy, at::Tensor x, at::Tensor xst, at::Tensor sbt, at::Tensor sb, c10::optional<at::Tensor> dres, at::Tensor dm, at::Tensor dx,
                 at::Tensor psb, int64_t period) {
  const int64_t M = x.size(0), d = x.size(1);
  check_rows(dy, "adaln_bwd_x dy", d);
  check_rows(x, "adaln_bwd_x x", d);
  TORCH_CHECK(dy.is_contiguous() && x.is_contiguous() && dy.scalar_type() == x.scalar_type(), "adaln_bwd_x: contiguous dy, x of one dtype");
  check_rows(sbt, "adaln_bwd_x sbt", 2 * d);
  check_rows(dm, "adaln_bwd_x dm", 2 * d);
  TORCH_CHECK(dm.is_contiguous() && dm.size(0) == M, "adaln_bwd_x: dm [M, 2 d] contiguous");
  TORCH_CHECK(dx.is_cuda() && dx.is_contiguous() && dx.size(0) == M && dx.size(1) == d && dx.scalar_type() == x.scalar_type(), "adaln_bwd_x: dx like x");
  if (dres) TORCH_CHECK(dres->is_cuda() && dres->is_contiguous() && dres->scalar_type() == x.scalar_type() && dres->size(0) == M, "adaln_bwd_x: dres like x");
  TORCH_CHECK(xst.is_cuda() && xst.is_contiguous() && xst.scalar_type() == at::kFloat && xst.numel() == 2 * M, "adaln_bwd_x: xst [M, 2] fp32");
  TORCH_CHECK(sb.is_cuda() && sb.is_contiguous() && sb.numel() == d && sb.scalar_type() == x.scalar_type(), "adaln_bwd_x: sb [d] like x");
  TORCH_CHECK(psb.is_cuda() && psb.is_contiguous() && psb.scalar_type() == at::kFloat && psb.dim() == 2 && psb.size(1) == d, "adaln_bwd_x: psb [blocks, d] fp32");
  TORCH_CHECK(period > 0 && M % period == 0 && sbt.size(0) == period, "adaln_bwd_x: the table has `period` rows and M is a multiple of it");
  const at::cuda::CUDAGuard g(x.device());
  with_width(d, "adaln_bwd_x", [&](auto wd) {
    constexpr int D = decltype(wd)::value;
    TORCH_CHECK(sbt.scalar_type() == x.scalar_type() && dm.scalar_type() == x.scalar_type(), "adaln_bwd_x: sbt, dm in x's dtype");
    with_dtype(x, "adaln_bwd_x x", [&](auto xt) {
      using XT = decltype(xt);
      adaln_bwd_x_kernel<D, XT><<<(unsigned)psb.size(0), RW * 32, 0, S_()>>>(
          CP_<XT>(dy), CP_<XT>(x), CP_<float2>(xst), CP_<XT>(sbt), sbt.stride(0), CP_<XT>(sb), OP_<XT>(dres), P_<XT>(dm), P_<XT>(dx), P_<float>(psb), M, period);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dcond [M, dc] (+ dextra); pw [blocks, dc] fp32 = the column sums of dcond_aff cond_hat, one row per block
void cond_ln_bwd(at::Tensor dca, at::Tensor cond, at::Tensor cst, at::Tensor w, c10::optional<at::Tensor> dextra, at::Tensor dcond, at::Tensor pw) {
  const int64_t M = cond.size(0), dc = cond.size(1);
  check_rows(cond, "cond_ln_bwd cond", dc);
  TORCH_CHECK(cond.is_contiguous(), "cond_ln_bwd: contiguous cond");
  TORCH_CHECK(dca.is_cuda() && dca.is_contiguous() && dca.scalar_type() == at::kFloat && dca.size(0) == M && dca.size(1) == dc, "cond_ln_bwd: dca [M, dc] fp32");
  TORCH_CHECK(cst.is_cuda() && cst.is_contiguous() && cst.scalar_type() == at::kFloat && cst.numel() == 2 * M, "cond_ln_bwd: cst [M, 2] fp32");
  check_vec(w, "cond_ln_bwd w", dc, at::kFloat);
  TORCH_CHECK(dcond.is_cuda() && dcond.is_contiguous() && dcond.size(0) == M && dcond.size(1) == dc && dcond.scalar_type() == cond.scalar_type(),
              "cond_ln_bwd: dcond like cond");
  if (dextra) TORCH_CHECK(dextra->is_cuda() && dextra->is_contiguous() && dextra->scalar_type() == cond.scalar_type() && dextra->size(0) == M,
                          "cond_ln_bwd: dextra like cond");
  TORCH_CHECK(pw.is_cuda() && pw.is_contiguous() && pw.scalar_type() == at::kFloat && pw.dim() == 2 && pw.size(1) == dc, "cond_ln_bwd: pw [blocks, dc] fp32");
  const at::cuda::CUDAGuard g(cond.device());
  with_width(dc, "cond_ln_bwd", [&](auto wd) {
    constexpr int D = decltype(wd)::value;
    with_dtype(cond, "cond_ln_bwd cond", [&](auto ct) {
      using CT = decltype(ct);
      cond_ln_bwd_kernel<D, CT><<<(unsigned)pw.size(0), RW * 32, 0, S_()>>>(CP_<float>(dca), CP_<CT>(cond), CP_<float2>(cst), CP_<float>(w), OP_<CT>(dextra), P_<CT>(dcond),
                                                                            P_<float>(pw), M);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------- atom width (128), bf16
void check_bf16_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.size(1) == cols && t.scalar_type() == at::kBFloat16, name, ": a contiguous bf16 CUDA [rows, ", cols, "]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

// y [M, 128] = the fused AdaLN of x [M, 128] with cond [P, 128] (row r reads cond row r % P); xst [M, 2] / cst [P, 2] fp32 (P == M) when the backward follows; cfg 0: 8 warps x 2 CTAs / SM, 1: 16 warps x 1, 2: 8 warps x 1 with the next tile prefetched by cp.async
void adaln_atom_fwd(at::Tensor x, at::Tensor cond, at::Tensor lnw, at::Tensor ws, at::Tensor wb, at::Tensor sb, at::Tensor y, c10::optional<at::Tensor> xst,
                    c10::optional<at::Tensor> cst, double eps_x, double eps_c, int64_t cfg) {
  const int64_t M = x.size(0), P = cond.size(0);
  check_bf16_rows(x, "adaln_atom_fwd x", 128);
  check_bf16_rows(cond, "adaln_atom_fwd cond", 128);
  check_bf16_rows(y, "adaln_atom_fwd y", 128);
  check_bf16_rows(ws, "adaln_atom_fwd ws", 128);
  check_bf16_rows(wb, "adaln_atom_fwd wb", 128);
  TORCH_CHECK(y.size(0) == M && P > 0 && M % P == 0, "adaln_atom_fwd: y like x, P divides M");
  TORCH_CHECK(lnw.is_cuda() && lnw.is_contiguous() && lnw.numel() == 128 && lnw.scalar_type() == at::kFloat && reinterpret_cast<uintptr_t>(lnw.data_ptr()) % 16 == 0, "adaln_atom_fwd: lnw [128] fp32");
  TORCH_CHECK(sb.is_cuda() && sb.is_contiguous() && sb.numel() == 128 && sb.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(sb.data_ptr()) % 16 == 0, "adaln_atom_fwd: sb [128] bf16");
  if (xst) TORCH_CHECK(xst->is_cuda() && xst->is_contiguous() && xst->scalar_type() == at::kFloat && xst->numel() == 2 * M, "adaln_atom_fwd: xst [M, 2] fp32");
  if (cst) TORCH_CHECK(cst->is_cuda() && cst->is_contiguous() && cst->scalar_type() == at::kFloat && cst->numel() == 2 * P && P == M, "adaln_atom_fwd: cst [M, 2] fp32 (one cond row per row)");
  const at::cuda::CUDAGuard g(x.device());
  AdalnAtomFwdParams p{CP_<bf>(x), CP_<bf>(cond), CP_<float>(lnw), CP_<bf>(ws), CP_<bf>(wb), CP_<bf>(sb), P_<bf>(y), xst ? P_<float2>(*xst) : nullptr,
                       cst ? P_<float2>(*cst) : nullptr, M, P, (float)eps_x, (float)eps_c};
  const int64_t ntile = (M + 15) / 16;
  if (cfg == 2) launch_persistent<AdalnAtomFwdCfg<8, 1, true>>(adaln_atom_fwd_kernel<AdalnAtomFwdCfg<8, 1, true>>, p, ntile, S_());
  else if (cfg == 1) launch_persistent<AdalnAtomFwdCfg<16, 1>>(adaln_atom_fwd_kernel<AdalnAtomFwdCfg<16, 1>>, p, ntile, S_());
  else launch_persistent<AdalnAtomFwdCfg<8, 2>>(adaln_atom_fwd_kernel<AdalnAtomFwdCfg<8, 2>>, p, ntile, S_());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the backward in one kernel (adaln_atom_bwd.cuh): dx (+ dres), dcond (+ dextra), dscale and aff (bf16, the operands of the weight-gradient GEMMs), psb / plnw [grid, 128] fp32 = per-CTA
// partial column sums of dscale and of dcond_aff chat (grid = atom_bwd_grid(M)); cond is one row per row of x (P == M); wt1 = Ws^T, wt2 = Wb^T
int64_t atom_bwd_grid(int64_t M) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return std::max<int64_t>(1, std::min<int64_t>(((M + 15) / 16 + AdalnAtomBwdCfg<8>::NW - 1) / AdalnAtomBwdCfg<8>::NW, (int64_t)sms));
}
void adaln_atom_bwd(at::Tensor dy, at::Tensor x, at::Tensor cond, at::Tensor xst, at::Tensor cst, at::Tensor lnw, at::Tensor ws, at::Tensor wt1, at::Tensor wt2, at::Tensor sb,
                    c10::optional<at::Tensor> dres, c10::optional<at::Tensor> dextra, at::Tensor dx, at::Tensor dcond, at::Tensor dsc, at::Tensor aff, at::Tensor psb,
                    at::Tensor plnw) {
  const int64_t M = x.size(0);
  for (const auto* t : {&dy, &x, &cond, &ws, &wt1, &wt2, &dx, &dcond, &dsc, &aff}) check_bf16_rows(*t, "adaln_atom_bwd", 128);
  TORCH_CHECK(dy.size(0) == M && cond.size(0) == M && dx.size(0) == M && dcond.size(0) == M && dsc.size(0) == M && aff.size(0) == M, "adaln_atom_bwd: [M, 128] rows");
  TORCH_CHECK(xst.is_cuda() && xst.is_contiguous() && xst.scalar_type() == at::kFloat && xst.numel() == 2 * M && cst.is_cuda() && cst.is_contiguous() && cst.scalar_type() == at::kFloat && cst.numel() == 2 * M,
              "adaln_atom_bwd: xst, cst [M, 2] fp32");
  TORCH_CHECK(lnw.is_cuda() && lnw.is_contiguous() && lnw.numel() == 128 && lnw.scalar_type() == at::kFloat && reinterpret_cast<uintptr_t>(lnw.data_ptr()) % 16 == 0, "adaln_atom_bwd: lnw [128] fp32");
  TORCH_CHECK(sb.is_cuda() && sb.is_contiguous() && sb.numel() == 128 && sb.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(sb.data_ptr()) % 16 == 0, "adaln_atom_bwd: sb [128] bf16");
  if (dres) check_bf16_rows(*dres, "adaln_atom_bwd dres", 128);
  if (dextra) check_bf16_rows(*dextra, "adaln_atom_bwd dextra", 128);
  const int64_t grid = atom_bwd_grid(M);
  for (const auto* t : {&psb, &plnw}) TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat && t->dim() == 2 && t->size(1) == 128 && t->size(0) >= grid, "adaln_atom_bwd: partials [grid, 128] fp32");
  const at::cuda::CUDAGuard g(x.device());
  AdalnAtomBwdParams p{CP_<bf>(dy), CP_<bf>(x), CP_<bf>(cond), CP_<float2>(xst), CP_<float2>(cst), CP_<float>(lnw), CP_<bf>(ws), CP_<bf>(wt1), CP_<bf>(wt2), CP_<bf>(sb), OP_<bf>(dres),
                       OP_<bf>(dextra), P_<bf>(dx), P_<bf>(dcond), P_<bf>(dsc), P_<bf>(aff), P_<float>(psb), P_<float>(plnw), M};
  using G = AdalnAtomBwdCfg<8>;
  launch_persistent<G>(adaln_atom_bwd_kernel<G>, p, (M + 15) / 16, S_(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------------------------------------------------- atom width (128), fp32 (TF32)
void check_f32_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.size(1) == cols && t.scalar_type() == at::kFloat, name, ": a contiguous fp32 CUDA [rows, ", cols, "]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}
void check_f32_vec(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.numel() == 128 && t.scalar_type() == at::kFloat && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": [128] fp32, 16-byte aligned");
}

// y [M, 128] = the fused AdaLN of x [M, 128] with cond [P, 128] (row r reads cond row r % P), fp32 rows, TF32 products (adaln_atom_fwd_tf32.cuh)
void adaln_atom_fwd_tf32(at::Tensor x, at::Tensor cond, at::Tensor lnw, at::Tensor ws, at::Tensor wb, at::Tensor sb, at::Tensor y, double eps_x, double eps_c) {
  const int64_t M = x.size(0), P = cond.size(0);
  check_f32_rows(x, "adaln_atom_fwd_tf32 x", 128);
  check_f32_rows(cond, "adaln_atom_fwd_tf32 cond", 128);
  check_f32_rows(y, "adaln_atom_fwd_tf32 y", 128);
  check_f32_rows(ws, "adaln_atom_fwd_tf32 ws", 128);
  check_f32_rows(wb, "adaln_atom_fwd_tf32 wb", 128);
  check_f32_vec(lnw, "adaln_atom_fwd_tf32 lnw");
  check_f32_vec(sb, "adaln_atom_fwd_tf32 sb");
  TORCH_CHECK(y.size(0) == M && P > 0 && M % P == 0, "adaln_atom_fwd_tf32: y like x, P divides M");
  const at::cuda::CUDAGuard g(x.device());
  AdalnAtomFwdTf32Params p{CP_<float>(x), CP_<float>(cond), CP_<float>(lnw), CP_<float>(ws), CP_<float>(wb), CP_<float>(sb), P_<float>(y), M, P, (float)eps_x, (float)eps_c};
  using G = AdalnAtomFwdTf32Cfg<8>;
  launch_persistent<G>(adaln_atom_fwd_tf32_kernel<G>, p, (M + 15) / 16, S_());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ----------------------------------------------------------------------------------------------------------------------------------------------- the closing pass
// sums[i] [rows, n] fp32 partial rows -> a fresh [n] vector shaped / typed like sum_like[i]; casts[i] a [R, C] fp32 matrix (any strides) -> a fresh contiguous [R, C] shaped / typed like
// cast_like[i] (fp32 or bf16).  One launch (adaln_finish.cuh); the outputs are returned sums first.
std::vector<at::Tensor> finish(std::vector<at::Tensor> sums, std::vector<at::Tensor> sum_like, std::vector<at::Tensor> casts, std::vector<at::Tensor> cast_like) {
  TORCH_CHECK(sums.size() == sum_like.size() && casts.size() == cast_like.size(), "finish: one `like` tensor per source");
  const size_t njob = sums.size() + casts.size();
  TORCH_CHECK(njob > 0 && njob <= (size_t)FIN_JOBS, "finish: 1 .. ", FIN_JOBS, " jobs");
  const at::Tensor& first = sums.empty() ? casts[0] : sums[0];
  const at::cuda::CUDAGuard g(first.device());
  FinParams p{};
  std::vector<at::Tensor> outs;
  int64_t tiles = 0;
  auto dtype_ok = [](const at::Tensor& like) { return like.scalar_type() == at::kFloat || like.scalar_type() == at::kBFloat16; };
  for (size_t i = 0; i < sums.size(); ++i) {
    const at::Tensor &s = sums[i], &like = sum_like[i];
    TORCH_CHECK(s.is_cuda() && s.is_contiguous() && s.scalar_type() == at::kFloat && s.dim() == 2 && s.size(0) > 0, "finish: sums[", i, "] a contiguous fp32 [rows, n]");
    TORCH_CHECK(dtype_ok(like) && like.numel() == s.size(1), "finish: sum_like[", i, "] holds n fp32 / bf16 elements");
    outs.push_back(at::empty({s.size(1)}, like.options()));
    FinJob& jb = p.job[p.njob++];
    jb = FinJob{CP_<float>(s), outs.back().data_ptr(), s.size(1), s.size(0), 0, 0, tiles, 0, like.scalar_type() == at::kBFloat16 ? 1 : 0, 0};
    tiles += (s.size(1) + FIN_SUM_COLS - 1) / FIN_SUM_COLS;
  }
  for (size_t i = 0; i < casts.size(); ++i) {
    const at::Tensor &c = casts[i], &like = cast_like[i];
    TORCH_CHECK(c.is_cuda() && c.scalar_type() == at::kFloat && c.dim() == 2 && c.numel() > 0, "finish: casts[", i, "] a CUDA fp32 matrix");
    TORCH_CHECK(dtype_ok(like) && like.sizes() == c.sizes(), "finish: cast_like[", i, "] shaped like the source, fp32 / bf16");
    outs.push_back(at::empty(c.sizes(), like.options()));
    const bool vec = c.stride(1) == 1 && c.size(1) % 4 == 0 && c.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(c.data_ptr()) % 16 == 0;
    FinJob& jb = p.job[p.njob++];
    jb = FinJob{CP_<float>(c), outs.back().data_ptr(), c.numel(), c.size(0), c.stride(0), c.stride(1), tiles, 1, like.scalar_type() == at::kBFloat16 ? 1 : 0, vec ? 1 : 0};
    tiles += (c.numel() + FIN_CAST_ELEMS - 1) / FIN_CAST_ELEMS;
  }
  finish_kernel<<<(unsigned)tiles, FIN_THREADS, 0, S_()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outs;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("finish", &finish);
  m.def("adaln_atom_fwd_tf32", &adaln_atom_fwd_tf32);
  m.def("adaln_atom_fwd", &adaln_atom_fwd);
  m.def("adaln_atom_bwd", &adaln_atom_bwd);
  m.def("atom_bwd_grid", &atom_bwd_grid);
  m.def("cond_ln", &cond_ln);
  m.def("adaln_epi", &adaln_epi);
  m.def("adaln_bwd_x", &adaln_bwd_x);
  m.def("cond_ln_bwd", &cond_ln_bwd);
}
