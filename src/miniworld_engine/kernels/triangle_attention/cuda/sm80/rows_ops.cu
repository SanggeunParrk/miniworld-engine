// rows_ops.cu -- torch bindings of the generic-width row kernels of the A100 triangle attention (rows_sm80.cuh), on the current CUDA stream.
// The attention core's kernels are ops.cu's (a separate extension: this one builds in seconds).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>

#include "rows_sm80.cuh"
#include "fused_sm80.cuh"

namespace {

using bf = __nv_bfloat16;

const bf* bp(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
bf* bm(const torch::Tensor& t) { return reinterpret_cast<bf*>(t.data_ptr()); }
bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }
void check_bf16(const torch::Tensor& t, const char* what) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, what, ": bf16 CUDA tensor"); }
void check_rows(const torch::Tensor& t, int64_t rows, int64_t cols, const char* what) {
  check_bf16(t, what);
  TORCH_CHECK(t.is_contiguous() && t.numel() == rows * cols && aligned16(t), what, ": [", rows, ", ", cols, "] bf16 contiguous, 16-byte aligned");
}
// a [T, cols] view with rows of `ld` elements (the GEMM's output columns): 16-byte aligned rows, extents inside the storage
void check_strided_rows(const torch::Tensor& t, int64_t rows, int64_t cols, const char* what) {
  check_bf16(t, what);
  TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols && t.stride(1) == 1 && t.stride(0) >= cols && t.stride(0) % 8 == 0 && aligned16(t) &&
              (t.storage_offset() + (rows - 1) * t.stride(0) + cols) * t.element_size() <= (int64_t)t.storage().nbytes(), what, ": [T, cols] rows of 16-byte aligned bf16");
}
int nsm() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

}  // namespace

// ---- weights: wq / wk / wv / wg [H hd, D], wb [H, D] bf16 (contiguous) -> out [NR, D + 8] (folded: W diag(gamma) | b_hi | b_lo | 0 ...) or [4 DHp, D] (plain)
void w_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor gamma, torch::Tensor beta, torch::Tensor out,
            int64_t D, int64_t H, int64_t hd, int64_t folded) {
  const torch::Tensor* ws[5] = {&wq, &wk, &wv, &wg, &wb};
  a100::WPackParams p;
  const int64_t DHp = H * 32, Hpad = (H + 7) / 8 * 8;
  TORCH_CHECK(D % 64 == 0 && D >= 64 && D <= 512 && H >= 1 && H <= 16 && (hd == 16 || hd == 32), "w_pack: D a multiple of 64 up to 512, 1..16 heads of 16 / 32 channels");
  for (int i = 0; i < 5; ++i) {
    check_bf16(*ws[i], "w_pack");
    TORCH_CHECK(ws[i]->is_contiguous() && ws[i]->numel() == (i < 4 ? H * hd : H) * D && reinterpret_cast<uintptr_t>(ws[i]->data_ptr()) % 4 == 0, "w_pack: weights [H hd | H, D] bf16 contiguous");
    p.w[i] = bp(*ws[i]);
  }
  p.NR = (int)(folded ? 4 * DHp + Hpad : 4 * DHp);
  const int64_t ld = folded ? D + 8 : D;
  check_bf16(out, "w_pack");
  TORCH_CHECK(out.is_contiguous() && out.numel() == p.NR * ld, "w_pack: out [NR, D (+ 8)]");
  if (folded) {
    TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && gamma.is_contiguous() && beta.is_contiguous() && gamma.numel() == D && beta.numel() == D &&
                reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(beta.data_ptr()) % 8 == 0, "w_pack: gamma / beta [D] fp32");
    p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>();
  } else {
    p.gamma = p.beta = nullptr;
  }
  p.out = bm(out); p.D = (int)D; p.H = (int)H; p.hd = (int)hd; p.DHp = (int)DHp; p.folded = (int)folded;
  a100::w_pack_kernel<<<(p.NR + 7) / 8, 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// wo [D, H hd] -> out [D, H 32] (zero columns for the padded channels)
void wo_pad(torch::Tensor wo, torch::Tensor out, int64_t D, int64_t H, int64_t hd) {
  check_rows(wo, D, H * hd, "wo_pad");
  check_rows(out, D, H * 32, "wo_pad out");
  const int64_t n = D * H * 32;
  a100::wo_pad_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream()>>>(bp(wo), bm(out), (int)D, (int)H, (int)hd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the front's LayerNorm rows: x [T, D] (the module's layout) -> xh [T, D + 8] (the starting frame), stats [T, 2] or empty
void ln_rows(torch::Tensor x, torch::Tensor xh, torch::Tensor stats, int64_t D, int64_t L, int64_t transposed, double eps) {
  const int64_t T = x.numel() / D;
  TORCH_CHECK(D % 64 == 0 && D >= 64 && D <= 512 && T % (L * L) == 0 && T < (int64_t(1) << 31), "ln_rows: D a multiple of 64 up to 512, T a multiple of L^2 below 2^31");
  check_rows(x, T, D, "ln_rows x");
  check_rows(xh, T, D + 8, "ln_rows xh");
  TORCH_CHECK(!stats.numel() || (stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == 2 * T), "ln_rows: stats [T, 2] fp32");
  a100::LnRowsParams p;
  p.x = bp(x); p.xh = bm(xh); p.stats = stats.numel() ? stats.data_ptr<float>() : nullptr;
  p.T = (unsigned)T; p.L = (int)L; p.transposed = (int)transposed; p.eps = (float)eps;
#define LN_CASE(DD) case DD: { using S = a100::RowShape<DD>; \
    const unsigned grid = (unsigned)std::min<int64_t>((T + 8 * S::RPW - 1) / (8 * S::RPW), (int64_t)nsm() * 16); \
    a100::ln_rows_kernel<DD><<<grid, 256, 0, stream()>>>(p); break; }
  switch (D) { LN_CASE(64) LN_CASE(128) LN_CASE(192) LN_CASE(256) LN_CASE(320) LN_CASE(384) LN_CASE(448) LN_CASE(512) default: TORCH_CHECK(false, "ln_rows: width"); }
#undef LN_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- bias planes: y [T, ldy] (the GEMM's output; the bias heads at columns col0 ..), mask [Z, L] uint8 or empty -> bias [Z, H, L, L]
void bias_planes(torch::Tensor y, torch::Tensor mask, torch::Tensor bias, int64_t col0, int64_t H, int64_t L) {
  const int64_t T = y.size(0);
  check_strided_rows(y, T, y.size(1), "bias_planes y");
  TORCH_CHECK(col0 % 8 == 0 && col0 + (H + 7) / 8 * 8 <= y.size(1) && T % (L * L) == 0 && T < (int64_t(1) << 31), "bias_planes: columns");
  check_bf16(bias, "bias_planes");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == T * H, "bias_planes: bias [Z, H, L, L]");
  const int64_t Z = T / (L * L);
  TORCH_CHECK(!mask.numel() || (mask.scalar_type() == at::kByte && mask.is_contiguous() && mask.numel() == Z * L), "bias_planes: mask [Z, L] uint8");
  a100::BiasPlanesParams p;
  p.y = bp(y); p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr; p.bias = bm(bias); p.ldy = y.stride(0);
  p.T = (unsigned)T; p.col0 = (int)col0; p.H = (int)H; p.L = (int)L;
  a100::bias_planes_kernel<<<(unsigned)((T + 255) / 256), 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the back: a = bf16(sigmoid(g) o).  o [T, DHp] contiguous, g [T, DHp] a strided view (the GEMM's gate columns), a [T, DHp]
void gate_rows(torch::Tensor o, torch::Tensor g, torch::Tensor a) {
  const int64_t T = o.size(0), DHp = o.size(1);
  TORCH_CHECK(DHp % 32 == 0 && DHp <= 512 && T < (int64_t(1) << 31), "gate_rows: DHp");
  check_rows(o, T, DHp, "gate_rows o");
  check_rows(a, T, DHp, "gate_rows a");
  check_strided_rows(g, T, DHp, "gate_rows g");
  a100::GateRowsParams p;
  p.o = bp(o); p.g = bp(g); p.a = bm(a); p.ldg = g.stride(0); p.T = (unsigned)T; p.DHp = (int)DHp;
  a100::gate_rows_kernel<<<(unsigned)((T + 7) / 8), dim3((unsigned)(DHp / 8), 8), 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out = bf16(res + bf16(y ds)): y [T, D] (the starting frame), res / out [T, D] (the module's layout; distinct), ds [Z, L, D] bf16 or empty
void out_rows(torch::Tensor y, torch::Tensor res, torch::Tensor ds, torch::Tensor out, int64_t L, int64_t transposed) {
  const int64_t T = y.size(0), D = y.size(1);
  TORCH_CHECK(D % 64 == 0 && D <= 512 && T % (L * L) == 0 && T < (int64_t(1) << 31), "out_rows: D");
  check_rows(y, T, D, "out_rows y");
  check_rows(res, T, D, "out_rows res");
  check_rows(out, T, D, "out_rows out");
  TORCH_CHECK(res.data_ptr() != out.data_ptr(), "out_rows: res / out distinct");
  TORCH_CHECK(!ds.numel() || (ds.scalar_type() == at::kBFloat16 && ds.is_contiguous() && ds.numel() == (T / L) * D && aligned16(ds)), "out_rows: ds [Z, L, D] bf16");
  a100::OutRowsParams p;
  p.y = bp(y); p.res = bp(res); p.ds = ds.numel() ? bp(ds) : nullptr; p.out = bm(out);
  p.T = (unsigned)T; p.L = (int)L; p.D = (int)D; p.transposed = (int)transposed;
  a100::out_rows_kernel<<<(unsigned)((T + 7) / 8), dim3((unsigned)(D / 8), 8), 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dy = bf16(dout ds) in the starting frame (dout in the module's layout)
void dy_rows(torch::Tensor dout, torch::Tensor ds, torch::Tensor dy, int64_t L, int64_t transposed) {
  const int64_t T = dout.size(0), D = dout.size(1);
  TORCH_CHECK(D % 64 == 0 && D <= 512 && T % (L * L) == 0 && T < (int64_t(1) << 31), "dy_rows: D");
  check_rows(dout, T, D, "dy_rows dout");
  check_rows(dy, T, D, "dy_rows dy");
  TORCH_CHECK(!ds.numel() || (ds.scalar_type() == at::kBFloat16 && ds.is_contiguous() && ds.numel() == (T / L) * D && aligned16(ds)), "dy_rows: ds [Z, L, D] bf16");
  a100::DyRowsParams p;
  p.dout = bp(dout); p.ds = ds.numel() ? bp(ds) : nullptr; p.dy = bm(dy); p.T = (unsigned)T; p.L = (int)L; p.D = (int)D; p.transposed = (int)transposed;
  a100::dy_rows_kernel<<<(unsigned)((T + 7) / 8), dim3((unsigned)(D / 8), 8), 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the gate's gradients: da [T, DHp] (= bf16(dy Wo)), o [T, DHp], g (strided gate columns) -> dg (strided: the gate columns of the gradient buffer), dov, a [T, DHp], delta [Z, H, L, L] fp32 or empty
void gate_bwd(torch::Tensor da, torch::Tensor o, torch::Tensor g, torch::Tensor dg, torch::Tensor dov, torch::Tensor a, torch::Tensor delta, int64_t L, int64_t H) {
  const int64_t T = o.size(0), DHp = o.size(1);
  TORCH_CHECK(DHp == H * 32 && DHp <= 512 && T % (L * L) == 0 && T < (int64_t(1) << 31), "gate_bwd: DHp = 32 H");
  check_rows(da, T, DHp, "gate_bwd da");
  check_rows(o, T, DHp, "gate_bwd o");
  check_rows(dov, T, DHp, "gate_bwd dov");
  check_rows(a, T, DHp, "gate_bwd a");
  check_strided_rows(g, T, DHp, "gate_bwd g");
  check_strided_rows(dg, T, DHp, "gate_bwd dg");
  TORCH_CHECK(!delta.numel() || (delta.scalar_type() == at::kFloat && delta.is_contiguous() && delta.numel() == T * H), "gate_bwd: delta [Z, H, L, L] fp32");
  a100::GateBwdParams p;
  p.da = bp(da); p.o = bp(o); p.g = bp(g); p.dg = bm(dg); p.dov = bm(dov); p.a = bm(a); p.delta = delta.numel() ? delta.data_ptr<float>() : nullptr;
  p.ldg = g.stride(0); p.lddg = dg.stride(0); p.T = (unsigned)T; p.DHp = (int)DHp; p.L = (int)L; p.H = (int)H;
  a100::gate_bwd_rows_kernel<<<(unsigned)((T + 7) / 8), dim3((unsigned)(DHp / 8), 8), (unsigned)(H * 8 * sizeof(float)), stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the front's backward rows: dxn [T, D] (bf16(D W)), db [Z, H, L, L], wb [H, D], x / dout [T, D] (the module's layout), stats [T, 2], gamma [D] fp32 -> dpair [T, D] (the module's layout; not x / dout)
void ln_bwd(torch::Tensor dxn, torch::Tensor db, torch::Tensor wb, torch::Tensor x, torch::Tensor stats, torch::Tensor dout, torch::Tensor gamma, torch::Tensor dpair,
            int64_t L, int64_t H, int64_t transposed) {
  const int64_t T = x.size(0), D = x.size(1);
  TORCH_CHECK(D % 64 == 0 && D >= 64 && D <= 512 && H >= 1 && H <= 16 && T % (L * L) == 0 && T < (int64_t(1) << 31), "ln_bwd: shapes");
  check_rows(dxn, T, D, "ln_bwd dxn");
  check_rows(x, T, D, "ln_bwd x");
  check_rows(dout, T, D, "ln_bwd dout");
  check_rows(dpair, T, D, "ln_bwd dpair");
  check_rows(wb, H, D, "ln_bwd wb");
  check_bf16(db, "ln_bwd db");
  TORCH_CHECK(db.is_contiguous() && db.numel() == T * H, "ln_bwd: db [Z, H, L, L]");
  TORCH_CHECK(stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == 2 * T, "ln_bwd: stats [T, 2] fp32");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && gamma.is_contiguous() && gamma.numel() == D && aligned16(gamma), "ln_bwd: gamma [D] fp32");
  TORCH_CHECK(dpair.data_ptr() != x.data_ptr() && dpair.data_ptr() != dout.data_ptr(), "ln_bwd: dpair must not alias x or dout");
  a100::LnBwdParams p;
  p.dxn = bp(dxn); p.db = bp(db); p.wb = bp(wb); p.x = bp(x); p.stats = stats.data_ptr<float>(); p.dout = bp(dout); p.gamma = gamma.data_ptr<float>(); p.dpair = bm(dpair);
  p.T = (unsigned)T; p.L = (int)L; p.H = (int)H; p.transposed = (int)transposed;
  const unsigned smem = (unsigned)(H * D * 2);
#define LB_CASE(DD) case DD: { using S = a100::RowShape<DD>; \
    const unsigned grid = (unsigned)std::min<int64_t>((T + 8 * S::RPW - 1) / (8 * S::RPW), (int64_t)nsm() * 8); \
    a100::ln_bwd_rows_kernel<DD><<<grid, 256, smem, stream()>>>(p); break; }
  switch (D) { LB_CASE(64) LB_CASE(128) LB_CASE(192) LB_CASE(256) LB_CASE(320) LB_CASE(384) LB_CASE(448) LB_CASE(512) default: TORCH_CHECK(false, "ln_bwd: width"); }
#undef LB_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the parameters' gradients (rows_sm80.cuh, wgrad_w_kernel): g [4 DHp + H, D + 8] fp32, go [D, DHp] fp32 -> dwq .. dwg [H hd, D], dwb [H, D], dwo [D, H hd] (bf16), dgamma / dbeta [D] (the LayerNorm parameters' dtype)
void wgrad_w(torch::Tensor g, torch::Tensor go, torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor dwq, torch::Tensor dwk,
             torch::Tensor dwv, torch::Tensor dwg, torch::Tensor dwb, torch::Tensor dwo, torch::Tensor gamma, torch::Tensor beta, torch::Tensor dgamma, torch::Tensor dbeta,
             int64_t D, int64_t H, int64_t hd) {
  const int64_t DHp = go.numel() / D;                                  // the hidden width of the layout: H 32 (heads padded to 32) or H hd (native)
  const int64_t slot = DHp == H * hd ? hd : 32;
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.is_contiguous() && g.numel() == (4 * DHp + H) * (D + 8), "wgrad_w: g [4 DHp + H, D + 8] fp32");
  TORCH_CHECK(go.is_cuda() && go.scalar_type() == at::kFloat && go.is_contiguous() && go.numel() == D * DHp, "wgrad_w: go [D, DHp] fp32");
  a100::WgradWParams p;
  const torch::Tensor* ws[5] = {&wq, &wk, &wv, &wg, &wb};
  const torch::Tensor* ds[6] = {&dwq, &dwk, &dwv, &dwg, &dwb, &dwo};
  for (int i = 0; i < 5; ++i) {
    check_bf16(*ws[i], "wgrad_w w");
    TORCH_CHECK(ws[i]->is_contiguous() && ws[i]->numel() == (i < 4 ? H * hd : H) * D, "wgrad_w: weights");
    p.w[i] = bp(*ws[i]);
  }
  for (int i = 0; i < 6; ++i) {
    check_bf16(*ds[i], "wgrad_w dw");
    TORCH_CHECK(ds[i]->is_contiguous() && ds[i]->numel() == (i < 4 ? H * hd * D : (i == 4 ? H * D : D * H * hd)), "wgrad_w: gradients");
    p.dw[i] = bm(*ds[i]);
  }
  const auto dt = gamma.scalar_type();
  TORCH_CHECK((dt == at::kFloat || dt == at::kBFloat16) && beta.scalar_type() == dt && dgamma.scalar_type() == dt && dbeta.scalar_type() == dt && gamma.is_contiguous() &&
              beta.is_contiguous() && dgamma.is_contiguous() && dbeta.is_contiguous() && gamma.numel() == D && beta.numel() == D && dgamma.numel() == D && dbeta.numel() == D,
              "wgrad_w: LayerNorm parameters / gradients [D], one dtype (fp32 or bf16)");
  p.g = g.data_ptr<float>(); p.go = go.data_ptr<float>();
  p.gamma = gamma.data_ptr(); p.beta = beta.data_ptr(); p.dgamma = dgamma.data_ptr(); p.dbeta = dbeta.data_ptr(); p.ln_bf16 = dt == at::kBFloat16;
  p.D = (int)D; p.H = (int)H; p.hd = (int)hd; p.DHp = (int)DHp; p.slot = (int)slot;
  const int64_t NW = 4 * H * hd + H;
  a100::wgrad_w_kernel<<<(unsigned)(NW + 2 * D), 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ====================================================================================================================================================================
// The fused front / back kernels for d_pair 64 | 128 and hidden 64 | 128 (fused_sm80.cuh)
// ====================================================================================================================================================================
namespace {

// One persistent CTA of G::NW warps per SM (G::MINB per SM): each warp walks the tiles of the grid-stride with the weights resident in shared memory.
template <class G, int TILE, class Params>
void launch_fused(void (*kernel)(const Params), int smem, const Params& p, int nsm_) {
  static bool set[64] = {};                                   // the attribute is per device and per kernel instantiation (this function is instantiated per kernel)
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    set[dev] = true;
  }
  const unsigned ntile = p.ntile;
  const unsigned grid = std::min<unsigned>((unsigned)(nsm_ * G::MINB), (ntile + G::NW - 1) / G::NW);
  kernel<<<grid, G::NTHR, smem, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the instantiated geometries: d_pair 64 | 128 x hidden 64 | 128 with 32-channel heads, and the native 16-channel heads of d_pair 64 / hidden 64 (4 heads)
#define FUSED_WIDTHS(DIN, DH, HDV, ...)                                                                                                     \
  if ((DIN) == 64 && (DH) == 64 && (HDV) == 32) { constexpr int kDIN = 64, kDH = 64, kHD = 32; __VA_ARGS__ }                                \
  else if ((DIN) == 64 && (DH) == 128 && (HDV) == 32) { constexpr int kDIN = 64, kDH = 128, kHD = 32; __VA_ARGS__ }                         \
  else if ((DIN) == 128 && (DH) == 64 && (HDV) == 32) { constexpr int kDIN = 128, kDH = 64, kHD = 32; __VA_ARGS__ }                         \
  else if ((DIN) == 128 && (DH) == 128 && (HDV) == 32) { constexpr int kDIN = 128, kDH = 128, kHD = 32; __VA_ARGS__ }                       \
  else if ((DIN) == 64 && (DH) == 64 && (HDV) == 16) { constexpr int kDIN = 64, kDH = 64, kHD = 16; __VA_ARGS__ }                           \
  else { TORCH_CHECK(false, "fused kernels: d_pair 64 | 128 with hidden 64 | 128 (32-channel heads) or d_pair 64 with hidden 64 (16-channel heads)"); }

}  // namespace

// the front's weights: wq / wk / wv / wg [H hd, DIN], wb [H, DIN] bf16, gamma / beta [DIN] fp32 -> wp [4 DH + 8, DIN] bf16 (W' in f1_channel order), bvec [4 DH + 8] fp32
void fused_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wp,
                torch::Tensor bvec, int64_t DIN, int64_t DH, int64_t H, int64_t hd) {
  const torch::Tensor* ws[5] = {&wq, &wk, &wv, &wg, &wb};
  a100::FusedPackParams p;
  TORCH_CHECK((DIN == 64 || DIN == 128) && (DH == 32 * H || DH == H * hd) && (hd == 16 || hd == 32) && H * hd <= DH, "fused_pack: widths");
  for (int i = 0; i < 5; ++i) {
    check_bf16(*ws[i], "fused_pack");
    TORCH_CHECK(ws[i]->is_contiguous() && ws[i]->numel() == (i < 4 ? H * hd : H) * DIN && reinterpret_cast<uintptr_t>(ws[i]->data_ptr()) % 4 == 0, "fused_pack: weights");
    p.w[i] = bp(*ws[i]);
  }
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && gamma.is_contiguous() && beta.is_contiguous() && gamma.numel() == DIN && beta.numel() == DIN &&
              reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(beta.data_ptr()) % 8 == 0, "fused_pack: gamma / beta [DIN] fp32");
  const int64_t rows = 4 * DH + 8;
  check_bf16(wp, "fused_pack");
  TORCH_CHECK(wp.is_contiguous() && wp.numel() == rows * DIN && bvec.scalar_type() == at::kFloat && bvec.is_contiguous() && bvec.numel() == rows, "fused_pack: outputs");
  p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>(); p.wp = bm(wp); p.bvec = bvec.data_ptr<float>();
  p.DIN = (int)DIN; p.DH = (int)DH; p.H = (int)H; p.hd = (int)hd; p.slot = DH == H * hd ? (int)hd : 32;
  a100::fused_pack_kernel<<<(unsigned)((rows + 7) / 8), 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the front: x [T, DIN] bf16 (module layout), w [4 DH + 8, DIN], bvec, mask [Z, L] uint8 or empty -> out [T, 4 DH] (q | k | v | g), bias [Z, H, L, L], lnst [T, 2] fp32 / xh [T, DIN + 8] (training) or empty
void fused_front(torch::Tensor x, torch::Tensor w, torch::Tensor bvec, torch::Tensor mask, torch::Tensor out, torch::Tensor bias, torch::Tensor lnst, torch::Tensor xh,
                 int64_t L, int64_t transposed, double eps, int64_t DIN, int64_t DH, int64_t hd) {
  const int64_t T = x.numel() / DIN, H = DH / hd, Z = T / (L * L);
  TORCH_CHECK(L % 128 == 0 && L >= 128 && T % (L * L) == 0 && T < (int64_t(1) << 31), "fused_front: L a multiple of 128");
  check_rows(x, T, DIN, "fused_front x");
  check_rows(w, 4 * DH + 8, DIN, "fused_front w");
  TORCH_CHECK(bvec.scalar_type() == at::kFloat && bvec.is_contiguous() && bvec.numel() == 4 * DH + 8, "fused_front: bvec");
  check_rows(out, T, 4 * DH, "fused_front out");
  check_bf16(bias, "fused_front bias");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == T * H, "fused_front: bias [Z, H, L, L]");
  TORCH_CHECK(!mask.numel() || (mask.scalar_type() == at::kByte && mask.is_contiguous() && mask.numel() == Z * L), "fused_front: mask [Z, L] uint8");
  TORCH_CHECK(!lnst.numel() || (lnst.scalar_type() == at::kFloat && lnst.is_contiguous() && lnst.numel() == 2 * T), "fused_front: lnst [T, 2] fp32");
  TORCH_CHECK(!xh.numel() || (xh.scalar_type() == at::kBFloat16 && xh.is_contiguous() && xh.numel() == T * (DIN + 8) && aligned16(xh)), "fused_front: xh [T, DIN + 8] bf16");
  a100::F1Params p;
  p.x = bp(x); p.w = bp(w); p.bvec = bvec.data_ptr<float>(); p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.out = bm(out); p.bias = bm(bias); p.lnst = lnst.numel() ? lnst.data_ptr<float>() : nullptr; p.xh = xh.numel() ? bm(xh) : nullptr;
  p.L = (int)L; p.transposed = (int)transposed; p.eps = (float)eps; p.ntile = (unsigned)(T / 32);
  FUSED_WIDTHS(DIN, DH, hd, {
    using G = a100::FusedCfg<kDIN, kDH, 8, 1, kHD>;
    launch_fused<G, 32>(a100::fused_f1_kernel<G>, G::F1_SMEM, p, nsm());
  })
}

// the back: o [T, DH], g [T, DH] (row stride g.stride(0)), wo [DIN, DH] bf16, res / out [T, DIN] (module layout, distinct), ds [Z, L, DIN] or empty
void fused_back(torch::Tensor o, torch::Tensor g, torch::Tensor wo, torch::Tensor res, torch::Tensor out, torch::Tensor ds, int64_t L, int64_t transposed, int64_t DIN, int64_t DH,
                int64_t hd) {
  const int64_t T = o.size(0);
  TORCH_CHECK(L % 128 == 0 && T % (L * L) == 0 && T < (int64_t(1) << 31), "fused_back: L a multiple of 128");
  check_rows(o, T, DH, "fused_back o");
  check_strided_rows(g, T, DH, "fused_back g");
  check_rows(wo, DIN, DH, "fused_back wo");
  check_rows(res, T, DIN, "fused_back res");
  check_rows(out, T, DIN, "fused_back out");
  TORCH_CHECK(res.data_ptr() != out.data_ptr(), "fused_back: res / out distinct");
  TORCH_CHECK(!ds.numel() || (ds.scalar_type() == at::kBFloat16 && ds.is_contiguous() && ds.numel() == (T / L) * DIN && aligned16(ds)), "fused_back: ds [Z, L, DIN] bf16");
  a100::F3Params p;
  p.o = bp(o); p.g = bp(g); p.wo = bp(wo); p.res = bp(res); p.out = bm(out); p.ds = ds.numel() ? bp(ds) : nullptr;
  p.ldg = g.stride(0); p.ntile = (unsigned)(T / 16); p.L = (int)L; p.transposed = (int)transposed;
  FUSED_WIDTHS(DIN, DH, hd, {
    using G = a100::FusedCfg<kDIN, kDH, 8, 1, kHD>;
    launch_fused<G, 16>(a100::fused_f3_kernel<G>, G::F3_SMEM, p, nsm());
  })
}

// the back's backward: dout [T, DIN] (module layout), ds, o [T, DH], g (strided), wot [DH, DIN] -> dg (strided), dov [T, DH], dy [T, DIN], a [T, DH], delta [Z, H, L, L] fp32 or empty
void fused_back_bwd(torch::Tensor dout, torch::Tensor ds, torch::Tensor o, torch::Tensor g, torch::Tensor wot, torch::Tensor dg, torch::Tensor dov, torch::Tensor dy, torch::Tensor a,
                    torch::Tensor delta, int64_t L, int64_t transposed, int64_t DIN, int64_t DH, int64_t hd) {
  const int64_t T = o.size(0), H = DH / hd;
  TORCH_CHECK(L % 128 == 0 && T % (L * L) == 0 && T < (int64_t(1) << 31), "fused_back_bwd: L a multiple of 128");
  check_rows(dout, T, DIN, "fused_back_bwd dout");
  check_rows(o, T, DH, "fused_back_bwd o");
  check_strided_rows(g, T, DH, "fused_back_bwd g");
  check_strided_rows(dg, T, DH, "fused_back_bwd dg");
  check_rows(wot, DH, DIN, "fused_back_bwd wot");
  check_rows(dov, T, DH, "fused_back_bwd dov");
  check_rows(dy, T, DIN, "fused_back_bwd dy");
  check_rows(a, T, DH, "fused_back_bwd a");
  TORCH_CHECK(!ds.numel() || (ds.scalar_type() == at::kBFloat16 && ds.is_contiguous() && ds.numel() == (T / L) * DIN && aligned16(ds)), "fused_back_bwd: ds [Z, L, DIN] bf16");
  TORCH_CHECK(!delta.numel() || (delta.scalar_type() == at::kFloat && delta.is_contiguous() && delta.numel() == T * H), "fused_back_bwd: delta [Z, H, L, L] fp32");
  a100::B3Params p;
  p.dout = bp(dout); p.ds = ds.numel() ? bp(ds) : nullptr; p.o = bp(o); p.g = bp(g); p.wot = bp(wot); p.dg = bm(dg); p.dov = bm(dov); p.dy = bm(dy); p.a = bm(a);
  p.delta = delta.numel() ? delta.data_ptr<float>() : nullptr;
  p.ldg = g.stride(0); p.lddg = dg.stride(0); p.ntile = (unsigned)(T / 16); p.L = (int)L; p.transposed = (int)transposed;
  FUSED_WIDTHS(DIN, DH, hd, {
    using G = a100::FusedCfg<kDIN, kDH, 8, 1, kHD>;
    launch_fused<G, 16>(a100::fused_b3_kernel<G>, G::B3_SMEM, p, nsm());
  })
}

// the front's backward: d [T, 4 DH] (dq | dk | dv | dg, row stride d.stride(0)), db [Z, H, L, L], x / dout [T, DIN] (module layout), lnst [T, 2], wt [4, DIN, DH], wbt [DIN, H], gamma [DIN] fp32 -> dpair [T, DIN]
void fused_front_bwd(torch::Tensor d, torch::Tensor db, torch::Tensor x, torch::Tensor lnst, torch::Tensor dout, torch::Tensor dpair, torch::Tensor wt, torch::Tensor wbt,
                     torch::Tensor gamma, int64_t L, int64_t transposed, int64_t DIN, int64_t DH, int64_t hd) {
  const int64_t T = x.size(0), H = DH / hd;
  TORCH_CHECK(L % 128 == 0 && T % (L * L) == 0 && T < (int64_t(1) << 31), "fused_front_bwd: L a multiple of 128");
  check_rows(x, T, DIN, "fused_front_bwd x");
  check_rows(dout, T, DIN, "fused_front_bwd dout");
  check_rows(dpair, T, DIN, "fused_front_bwd dpair");
  TORCH_CHECK(dpair.data_ptr() != x.data_ptr() && dpair.data_ptr() != dout.data_ptr(), "fused_front_bwd: dpair must not alias x or dout");
  check_strided_rows(d, T, 4 * DH, "fused_front_bwd d");
  check_bf16(db, "fused_front_bwd db");
  TORCH_CHECK(db.is_contiguous() && db.numel() == T * H, "fused_front_bwd: db [Z, H, L, L]");
  TORCH_CHECK(lnst.scalar_type() == at::kFloat && lnst.is_contiguous() && lnst.numel() == 2 * T, "fused_front_bwd: lnst [T, 2] fp32");
  check_bf16(wt, "fused_front_bwd wt");
  check_bf16(wbt, "fused_front_bwd wbt");
  TORCH_CHECK(wt.is_contiguous() && wt.numel() == 4 * DIN * DH && wbt.is_contiguous() && wbt.numel() == DIN * H, "fused_front_bwd: wt [4, DIN, DH] / wbt [DIN, H]");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && gamma.is_contiguous() && gamma.numel() == DIN, "fused_front_bwd: gamma [DIN] fp32");
  a100::B1Params p;
  p.d = bp(d); p.db = bp(db); p.x = bp(x); p.lnst = lnst.data_ptr<float>(); p.dout = bp(dout); p.dpair = bm(dpair); p.wt = bp(wt); p.wbt = bp(wbt); p.gamma = gamma.data_ptr<float>();
  p.ldd = d.stride(0); p.ntile = (unsigned)(T / 16); p.L = (int)L; p.transposed = (int)transposed;
  FUSED_WIDTHS(DIN, DH, hd, {
    using G = a100::FusedCfg<kDIN, kDH, 8, 1, kHD>;
    launch_fused<G, 16>(a100::fused_b1_kernel<G>, G::B1_SMEM, p, nsm());
  })
}

// the backward's weights: wq .. wg [H hd, DIN], wb [H, DIN], wo [DIN, H hd] bf16, gamma [DIN] (fp32 / bf16) -> wt [4, DIN, DH], wbt [DIN, H], wot [DH, DIN] bf16, gamma32 [DIN] fp32
void fused_bwd_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo, torch::Tensor gamma, torch::Tensor wt,
                    torch::Tensor wbt, torch::Tensor wot, torch::Tensor gamma32, int64_t DIN, int64_t DH, int64_t H, int64_t hd) {
  const torch::Tensor* ws[6] = {&wq, &wk, &wv, &wg, &wb, &wo};
  a100::FusedBwdPackParams p;
  TORCH_CHECK((DIN == 64 || DIN == 128) && (DH == 32 * H || DH == H * hd) && (hd == 16 || hd == 32), "fused_bwd_pack: widths");
  for (int i = 0; i < 6; ++i) {
    check_bf16(*ws[i], "fused_bwd_pack");
    TORCH_CHECK(ws[i]->is_contiguous() && ws[i]->numel() == (i == 4 ? H : (i == 5 ? DIN : H * hd)) * (i == 5 ? H * hd : DIN), "fused_bwd_pack: weights");
    p.w[i] = bp(*ws[i]);
  }
  TORCH_CHECK(gamma.is_contiguous() && gamma.numel() == DIN && (gamma.scalar_type() == at::kFloat || gamma.scalar_type() == at::kBFloat16), "fused_bwd_pack: gamma [DIN] fp32 / bf16");
  check_bf16(wt, "fused_bwd_pack"); check_bf16(wbt, "fused_bwd_pack"); check_bf16(wot, "fused_bwd_pack");
  TORCH_CHECK(wt.is_contiguous() && wt.numel() == 4 * DIN * DH && wbt.is_contiguous() && wbt.numel() == DIN * H && wot.is_contiguous() && wot.numel() == DH * DIN &&
              gamma32.scalar_type() == at::kFloat && gamma32.is_contiguous() && gamma32.numel() == DIN, "fused_bwd_pack: outputs");
  p.gamma = gamma.data_ptr(); p.gamma_bf16 = gamma.scalar_type() == at::kBFloat16;
  p.wt = bm(wt); p.wbt = bm(wbt); p.wot = bm(wot); p.gamma32 = gamma32.data_ptr<float>();
  p.DIN = (int)DIN; p.DH = (int)DH; p.H = (int)H; p.hd = (int)hd; p.slot = DH == H * hd ? (int)hd : 32;
  const int64_t total = 4 * DIN * DH + DH * DIN + DIN * H + DIN;
  a100::fused_bwd_pack_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_pack", &fused_pack);
  m.def("fused_front", &fused_front);
  m.def("fused_back", &fused_back);
  m.def("fused_back_bwd", &fused_back_bwd);
  m.def("fused_front_bwd", &fused_front_bwd);
  m.def("fused_bwd_pack", &fused_bwd_pack);
  m.def("w_pack", &w_pack);
  m.def("wo_pad", &wo_pad);
  m.def("ln_rows", &ln_rows);
  m.def("bias_planes", &bias_planes);
  m.def("gate_rows", &gate_rows);
  m.def("out_rows", &out_rows);
  m.def("dy_rows", &dy_rows);
  m.def("gate_bwd", &gate_bwd);
  m.def("ln_bwd", &ln_bwd);
  m.def("wgrad_w", &wgrad_w);
}
