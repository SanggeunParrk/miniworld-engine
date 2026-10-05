// ops.cu -- torch bindings of the A100 (sm_80) MPNN edge kernels, on the current CUDA stream.  Built once by ``cuda/sm80.py`` (``load_extension``), never at import.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "pack.cuh"
#include "tail_bwd.cuh"
#include "dw.cuh"
#include "scatter.cuh"
#include "mlp.cuh"
#include "rows.cuh"

using namespace me80;

namespace {

int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
template <class K>
void set_smem(K kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
template <class T> const T* cptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <class T> T* mptr(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

// fork / join events of the side stream of the backward (per thread and device, created once, never destroyed: the CUDA context may be gone when a thread exits)
at::cuda::CUDAEvent& side_event(int64_t device, int which) {
  static thread_local std::vector<at::cuda::CUDAEvent>* events = new std::vector<at::cuda::CUDAEvent>(64);
  return (*events)[(size_t)(device & 31) * 2 + which];
}

void check_weight(const torch::Tensor& w, const char* name) {
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.size(0) == 128 && w.size(1) == 128 && w.stride(1) == 1, name, ": a [128, 128] CUDA matrix with unit column stride");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 || w.scalar_type() == at::kFloat, name, ": bf16 or fp32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && (w.stride(0) * w.element_size()) % 16 == 0, name, ": 16-byte aligned rows");
}
void check_vec(const torch::Tensor& v, const char* name) {
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.numel() == 128, name, ": a contiguous [128] CUDA vector");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16 || v.scalar_type() == at::kFloat, name, ": bf16 or fp32");
}
void check_act(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16 && t.size(-1) == 128, name, ": a contiguous bf16 CUDA tensor [..., 128]");
}

}  // namespace

// ---- weights: layer l of the image is the matrix wl itself (forward image) or its transpose (trl: the backward image) -> (img uint8 [96 KiB], tab fp32 [512])
std::vector<torch::Tensor> pack(bool tr0, bool tr1, bool tr2, torch::Tensor w0, torch::Tensor w1, torch::Tensor w2, torch::Tensor b2, torch::Tensor b3, torch::Tensor gamma, torch::Tensor beta) {
  const c10::cuda::CUDAGuard guard(w0.device());
  check_weight(w0, "w0"); check_weight(w1, "w1"); check_weight(w2, "w2");
  check_vec(b2, "b2"); check_vec(b3, "b3"); check_vec(gamma, "gamma"); check_vec(beta, "beta");
  auto img = torch::empty({IMG_BYTES}, w0.options().dtype(at::kByte));
  auto tab = torch::empty({TAB_FLOATS}, w0.options().dtype(at::kFloat));
  PackParams p{};
  const torch::Tensor* ws[3] = {&w0, &w1, &w2};
  const torch::Tensor* vs[4] = {&b2, &b3, &gamma, &beta};
  for (int l = 0; l < 3; ++l) { p.w[l] = ws[l]->data_ptr(); p.w_ld[l] = (int)ws[l]->stride(0); p.w_fp32[l] = ws[l]->scalar_type() == at::kFloat; }
  for (int v = 0; v < 4; ++v) { p.vec[v] = vs[v]->data_ptr(); p.vec_fp32[v] = vs[v]->scalar_type() == at::kFloat; }
  p.img = mptr<uint8_t>(img);
  p.tab = mptr<float>(tab);
  auto st = at::cuda::getCurrentCUDAStream();
  p.tr[0] = tr0; p.tr[1] = tr1; p.tr[2] = tr2;
  pack_kernel<<<24, 256, 0, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {img, tab};
}

// ---- the edge tail forward: (out, act1, d1, act2, d2, values, stats, keep); everything after ``out`` is empty without ``save`` (keep: also without dropout)
#ifndef MPNN_FWD_NW
#define MPNN_FWD_NW 12
#endif
constexpr int FWD_NW = MPNN_FWD_NW;
constexpr int FWD_SMEM = IMG_BYTES + TAB_FLOATS * 4;

template <bool SAVE, bool DROP>
static void launch_fwd(const TailFwdParams& p, int ntile, cudaStream_t st) {
  const int grid = std::max(1, std::min((ntile + FWD_NW - 1) / FWD_NW, num_sms()));
  set_smem(tail_fwd_kernel<FWD_NW, SAVE, DROP>, FWD_SMEM);
  tail_fwd_kernel<FWD_NW, SAVE, DROP><<<grid, FWD_NW * 32, FWD_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> tail_fwd(torch::Tensor edge, torch::Tensor query, torch::Tensor nbr, torch::Tensor idx, torch::Tensor img, torch::Tensor tab, torch::Tensor seed,
                                    int64_t K, double eps, double drop_p, bool save, int64_t row_base) {
  const c10::cuda::CUDAGuard guard(edge.device());
  check_act(edge, "edge"); check_act(query, "query"); check_act(nbr, "nbr");
  const int64_t rows = edge.numel() / D;
  TORCH_CHECK(K >= 1 && rows % K == 0 && query.numel() / D == rows / K, "query must be [rows / K, 128]");
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && idx.scalar_type() == at::kLong && idx.numel() == rows, "idx: contiguous int64 [rows]");
  TORCH_CHECK(img.is_cuda() && img.numel() == IMG_BYTES && tab.is_cuda() && tab.numel() == TAB_FLOATS, "packed weights");
  TORCH_CHECK(seed.is_cuda() && seed.scalar_type() == at::kLong && seed.numel() == 1, "seed: one int64");
  TORCH_CHECK(rows > 0 && rows * D <= 2147483647LL && drop_p >= 0.0 && drop_p < 1.0, "rows / dropout probability");
  const bool drop = drop_p > 0.0;
  auto out = torch::empty_like(edge);
  auto act1 = save ? torch::empty_like(edge) : torch::empty({0}, edge.options());
  auto act2 = save ? torch::empty_like(edge) : torch::empty({0}, edge.options());
  auto d1 = save ? torch::empty(edge.sizes(), edge.options().dtype(at::kHalf)) : torch::empty({0}, edge.options().dtype(at::kHalf));
  auto d2 = save ? torch::empty(edge.sizes(), edge.options().dtype(at::kHalf)) : torch::empty({0}, edge.options().dtype(at::kHalf));
  auto values = save ? torch::empty_like(edge) : torch::empty({0}, edge.options());
  auto stats = torch::empty({save ? rows : 0, 2}, edge.options().dtype(at::kFloat));
  auto keep = torch::empty({save && drop ? rows : 0, 4}, edge.options().dtype(at::kInt));
  TailFwdParams p{};
  p.edge = cptr<__nv_bfloat16>(edge); p.query = cptr<__nv_bfloat16>(query); p.nbr = cptr<__nv_bfloat16>(nbr); p.idx = cptr<int64_t>(idx);
  p.img = cptr<uint8_t>(img); p.tab = cptr<float>(tab); p.seed = cptr<int64_t>(seed);
  p.out = mptr<__nv_bfloat16>(out);
  p.act1 = save ? mptr<__nv_bfloat16>(act1) : nullptr; p.act2 = save ? mptr<__nv_bfloat16>(act2) : nullptr; p.values = save ? mptr<__nv_bfloat16>(values) : nullptr;
  p.d1 = save ? reinterpret_cast<__nv_bfloat16*>(d1.data_ptr()) : nullptr; p.d2 = save ? reinterpret_cast<__nv_bfloat16*>(d2.data_ptr()) : nullptr;
  p.stats = save ? mptr<float2>(stats) : nullptr; p.keep = (save && drop) ? mptr<uint32_t>(keep) : nullptr;
  TORCH_CHECK(row_base >= 0 && (row_base + rows) * D <= 2147483647LL, "row_base: the launch is a slice of a tensor of at most 2^31 - 1 elements");
  p.rows = (int)rows; p.K = (int)K; p.eps = (float)eps; p.row_base = (int)row_base;
  p.scale = drop ? (float)(1.0 / (1.0 - drop_p)) : 1.f;
  p.thr = keep_threshold((float)(1.0 - drop_p));
  const int ntile = (int)((rows + 15) / 16);
  auto st = at::cuda::getCurrentCUDAStream();
  if (save) { if (drop) launch_fwd<true, true>(p, ntile, st); else launch_fwd<true, false>(p, ntile, st); }
  else { if (drop) launch_fwd<false, true>(p, ntile, st); else launch_fwd<false, false>(p, ntile, st); }
  return {out, act1, d1, act2, d2, values, stats, keep};
}

// ---- the dropout decisions of the kernels as a bool [rows, 128] (tests: the same draw, element by element)
__global__ void dropout_mask_kernel(const int64_t* seed, uint8_t* mask, int rows, uint32_t thr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * D) return;
  const int row = i >> 7, c = i & 127, a = c >> 5, q = (c >> 3) & 3, s = (c >> 1) & 3, e = c & 1;
  const uint32_t w = drop_word((uint32_t)row * 64u + (4 * a + s) * 4 + q, drop_key((unsigned long long)seed[0]));
  mask[i] = (e ? (w >> 16) : (w & 0xffffu)) < thr ? 1 : 0;
}

torch::Tensor dropout_mask(torch::Tensor seed, int64_t rows, double drop_p) {
  const c10::cuda::CUDAGuard guard(seed.device());
  auto mask = torch::empty({rows, D}, seed.options().dtype(at::kBool));
  const int n = (int)(rows * D);
  dropout_mask_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cptr<int64_t>(seed), mptr<uint8_t>(mask), (int)rows, keep_threshold((float)(1.0 - drop_p)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mask;
}

// ---- the edge tail backward (saveact policy): the chain kernel (LayerNorm / dropout backward + dX x 3), the weight-gradient kernel, the two reductions of G1 and the fixed-order sums
//      -> (grad_edge, grad_query, grad_neighbor, dW1e, dW2, dW3, db2, db3, dgamma, dbeta); parameter gradients in the parameters' dtypes (bf16 or fp32)
constexpr int BWD_NW = 8;

template <bool DROP>
static void launch_bwd(const TailBwdParams& p, int grid, cudaStream_t st) {
  set_smem(tail_bwd_kernel<BWD_NW, DROP>, FWD_SMEM);
  tail_bwd_kernel<BWD_NW, DROP><<<grid, BWD_NW * 32, FWD_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static torch::Tensor like_param(const torch::Tensor& ref, at::IntArrayRef shape, bool fp32) {
  return torch::empty(shape, ref.options().dtype(fp32 ? at::kFloat : at::kBFloat16));
}

std::vector<torch::Tensor> tail_bwd(torch::Tensor go, torch::Tensor edge, torch::Tensor idx, torch::Tensor values, torch::Tensor stats, torch::Tensor keep, torch::Tensor act1,
                                    torch::Tensor d1, torch::Tensor act2, torch::Tensor d2, torch::Tensor img, torch::Tensor tab, int64_t K, int64_t nodes, double drop_p, bool w_fp32, bool b_fp32, bool n_fp32) {
  const c10::cuda::CUDAGuard guard(edge.device());
  check_act(go, "go"); check_act(edge, "edge"); check_act(values, "values"); check_act(act1, "act1"); check_act(act2, "act2");
  TORCH_CHECK(d1.is_cuda() && d1.is_contiguous() && d1.scalar_type() == at::kHalf && d2.is_cuda() && d2.is_contiguous() && d2.scalar_type() == at::kHalf, "d1 / d2: contiguous fp16");
  const int64_t rows = edge.numel() / D, groups = rows / K;
  TORCH_CHECK(go.numel() == edge.numel() && values.numel() == edge.numel() && act1.numel() == edge.numel() && act2.numel() == edge.numel() && d1.numel() == edge.numel() && d2.numel() == edge.numel(), "operand shapes");
  TORCH_CHECK(rows % K == 0 && nodes >= 1 && idx.is_cuda() && idx.scalar_type() == at::kLong && idx.numel() == rows, "idx / K");
  TORCH_CHECK(stats.is_cuda() && stats.scalar_type() == at::kFloat && stats.numel() == rows * 2, "stats");
  const bool drop = drop_p > 0.0;
  TORCH_CHECK(!drop || (keep.is_cuda() && keep.scalar_type() == at::kInt && keep.numel() == rows * 4), "keep words");
  auto st = at::cuda::getCurrentCUDAStream();
  const int nsm = num_sms();
  auto f32 = go.options().dtype(at::kFloat);
  auto gedge = torch::empty_like(go);

  // scratch, three allocations instead of nine (a host-side cost at small sizes): G3 | G2 | G1 (bf16), the fp32 partials (LayerNorm | weight gradients | bias column sums) and the integer
  // work of the reverse graph (counts | fill cursors | offsets | row lists)
  const int ntile = (int)((rows + 15) / 16);
  const int grid = std::max(1, std::min((ntile + BWD_NW - 1) / BWD_NW, nsm));
  const int stages_total = (int)((rows + 63) / 64);
  const int slabs0 = std::max(1, nsm / 3);
  const int stages_per_slab = (stages_total + slabs0 - 1) / slabs0;
  const int slab_rows = stages_per_slab * 64;
  const int slabs = (int)((rows + slab_rows - 1) / slab_rows);
  const int64_t n_ln = (int64_t)grid * 256, n_part = (int64_t)3 * slabs * D * D, n_csum = (int64_t)3 * slabs * D;
  auto gbuf = torch::empty({3 * rows, D}, go.options());
  auto fbuf = torch::empty({n_ln + n_part + n_csum}, f32);
  const int64_t n_cnt = 2 * nodes, n_offs = (nodes + 1 + 3) / 4 * 4;
  auto ibuf = torch::empty({n_cnt + n_offs + rows}, go.options().dtype(at::kInt));
  __nv_bfloat16* const g3p = mptr<__nv_bfloat16>(gbuf);
  __nv_bfloat16* const g2p = g3p + rows * D;
  __nv_bfloat16* const g1p = g2p + rows * D;
  float* const ln_part_p = mptr<float>(fbuf);
  float* const part_p = ln_part_p + n_ln;
  float* const csum_p = part_p + n_part;
  int* const cnt = mptr<int>(ibuf);
  int* const cursor = cnt + nodes;
  int* const offs_p = cnt + n_cnt;
  int* const perm_p = offs_p + n_offs;
  C10_CUDA_CHECK(cudaMemsetAsync(cnt, 0, (size_t)n_cnt * sizeof(int), st));       // counts and fill cursors start at zero

  // -- chain kernel
  TailBwdParams bp{};
  bp.go = cptr<__nv_bfloat16>(go); bp.values = cptr<__nv_bfloat16>(values); bp.stats = cptr<float2>(stats); bp.keep = drop ? cptr<uint32_t>(keep) : nullptr;
  bp.d2 = reinterpret_cast<const __nv_bfloat16*>(d2.data_ptr()); bp.d1 = reinterpret_cast<const __nv_bfloat16*>(d1.data_ptr()); bp.img = cptr<uint8_t>(img); bp.tab = cptr<float>(tab);
  bp.g3 = g3p; bp.g2 = g2p; bp.g1 = g1p; bp.gedge = mptr<__nv_bfloat16>(gedge);
  bp.ln_part = ln_part_p; bp.rows = (int)rows; bp.scale = drop ? (float)(1.0 / (1.0 - drop_p)) : 1.f;
  if (drop) launch_bwd<true>(bp, grid, st); else launch_bwd<false>(bp, grid, st);

  // -- weight gradients: 3 jobs x slabs of rows
  DwParams dp{};
  dp.g[0] = g3p; dp.g[1] = g2p; dp.g[2] = g1p;
  dp.a[0] = cptr<__nv_bfloat16>(act2); dp.a[1] = cptr<__nv_bfloat16>(act1); dp.a[2] = cptr<__nv_bfloat16>(edge);
  dp.njob = 3; dp.rows = (int)rows; dp.slabs = slabs; dp.slab_rows = slab_rows;
  dp.part = part_p; dp.csum = csum_p;
  set_smem(dw_kernel<false>, DW_SMEM_PLAIN);

  // -- everything that does not need the weight gradients runs on a pool stream next to the weight-gradient kernel: the reverse graph (CSR: counts, offsets, row lists; it depends on idx
  //    alone) and the two reductions of G1.  A weight-gradient CTA leaves 8 K of an SM's registers and half its shared memory free (a chain-kernel CTA leaves nothing), enough for these
  //    small kernels (4-warp CTAs); the main stream goes on with the fixed-order sums and joins the side stream at the end.  The events are the fork / join edges of a CUDA graph capture too.
  auto gnbr = torch::empty({nodes, D}, go.options()), gquery = torch::empty({groups, D}, go.options());
  auto dw1e = like_param(go, {D, D}, w_fp32), dw2 = like_param(go, {D, D}, w_fp32), dw3 = like_param(go, {D, D}, w_fp32);
  auto db2 = like_param(go, {D}, b_fp32), db3 = like_param(go, {D}, b_fp32);
  auto dgamma = like_param(go, {D}, n_fp32), dbeta = like_param(go, {D}, n_fp32);
  FinParams fp{};
  fp.part = part_p; fp.csum = csum_p; fp.ln_part = ln_part_p;
  fp.njob = 3; fp.slabs = slabs; fp.nwarps = grid;
  fp.dw[0] = dw3.data_ptr(); fp.dw[1] = dw2.data_ptr(); fp.dw[2] = dw1e.data_ptr();
  fp.dw_fp32[0] = fp.dw_fp32[1] = fp.dw_fp32[2] = w_fp32;
  fp.db[0] = db3.data_ptr(); fp.db[1] = db2.data_ptr(); fp.db_fp32[0] = fp.db_fp32[1] = b_fp32;
  fp.dn[0] = dgamma.data_ptr(); fp.dn[1] = dbeta.data_ptr(); fp.dn_fp32[0] = fp.dn_fp32[1] = n_fp32;

  const c10::cuda::CUDAStream side = c10::cuda::getStreamFromPool(false, (c10::DeviceIndex)edge.get_device());
  at::cuda::CUDAEvent& forked = side_event(edge.get_device(), 0);
  at::cuda::CUDAEvent& joined = side_event(edge.get_device(), 1);
  forked.record(st);                                                  // the chain kernel (and the zero fill) are behind this point
  dw_kernel<false><<<3 * slabs, 256, DW_SMEM_PLAIN, st>>>(dp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dw_finalize_kernel<<<(((3 * D * D + 2 * D) / 4 + 255) / 256) + 8, 256, 0, st>>>(fp);       // region A blocks + 8 blocks (64 warps) for the LayerNorm partials
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  forked.block(side);
  csr_count_kernel<<<(int)((rows + 255) / 256), 256, 0, side>>>(cptr<int64_t>(idx), cnt, (int)rows);
  csr_scan_kernel<<<1, 1024, 0, side>>>(cnt, offs_p, (int)nodes);
  csr_fill_kernel<<<(int)((rows + 255) / 256), 256, 0, side>>>(cptr<int64_t>(idx), offs_p, cursor, perm_p, (int)rows);
  nbr_reduce_kernel<<<(int)((nodes + 3) / 4), 128, 0, side>>>(g1p, offs_p, perm_p, mptr<__nv_bfloat16>(gnbr), (int)nodes);
  query_reduce_kernel<<<(int)((groups + 3) / 4), 128, 0, side>>>(g1p, mptr<__nv_bfloat16>(gquery), (int)groups, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  joined.record(side);
  joined.block(st);
  return {gedge, gquery, gnbr, dw1e, dw2, dw3, db2, db3, dgamma, dbeta};
}

// ---- the edge MLP: forward (out, hid) and backward (grad_x, dWh, dWo, dbh, dbo)
constexpr int MLP_FWD_NW = 12, MLP_BWD_NW = 8;

template <bool SAVE>
static void launch_mlp_fwd(const MlpFwdParams& p, int ntile, cudaStream_t st) {
  const int grid = std::max(1, std::min((ntile + MLP_FWD_NW - 1) / MLP_FWD_NW, num_sms()));
  set_smem(mlp_fwd_kernel<MLP_FWD_NW, SAVE>, MLP_FWD_SMEM);
  mlp_fwd_kernel<MLP_FWD_NW, SAVE><<<grid, MLP_FWD_NW * 32, MLP_FWD_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> mlp_fwd(torch::Tensor x, torch::Tensor img, torch::Tensor tab, bool save) {
  const c10::cuda::CUDAGuard guard(x.device());
  check_act(x, "x");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows > 0 && rows * D <= 2147483647LL, "rows");
  TORCH_CHECK(img.is_cuda() && img.numel() == IMG_BYTES && tab.is_cuda() && tab.numel() == TAB_FLOATS, "packed weights");
  auto out = torch::empty_like(x);
  auto hid = save ? torch::empty_like(x) : torch::empty({0}, x.options());
  MlpFwdParams p{};
  p.x = cptr<__nv_bfloat16>(x); p.img = cptr<uint8_t>(img); p.tab = cptr<float>(tab); p.out = mptr<__nv_bfloat16>(out); p.hid = save ? mptr<__nv_bfloat16>(hid) : nullptr; p.rows = (int)rows;
  auto st = at::cuda::getCurrentCUDAStream();
  const int ntile = (int)((rows + 15) / 16);
  if (save) launch_mlp_fwd<true>(p, ntile, st); else launch_mlp_fwd<false>(p, ntile, st);
  return {out, hid};
}

template <bool RECOMP>
static void launch_mlp_bwd(const MlpBwdParams& p, int ntile, cudaStream_t st) {
  constexpr int smem = (RECOMP ? 3 : 2) * LAYER_BYTES + TAB_FLOATS * 4;
  const int grid = std::max(1, std::min((ntile + MLP_BWD_NW - 1) / MLP_BWD_NW, num_sms()));
  set_smem(mlp_bwd_kernel<MLP_BWD_NW, RECOMP>, smem);
  mlp_bwd_kernel<MLP_BWD_NW, RECOMP><<<grid, MLP_BWD_NW * 32, smem, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> mlp_bwd(torch::Tensor go, torch::Tensor x, torch::Tensor hid, torch::Tensor img, torch::Tensor tab, bool recompute, bool w_fp32, bool b_fp32) {
  const c10::cuda::CUDAGuard guard(x.device());
  check_act(go, "go"); check_act(x, "x");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(go.numel() == x.numel() && rows > 0 && rows * D <= 2147483647LL, "operand shapes");
  TORCH_CHECK(recompute || (hid.is_cuda() && hid.is_contiguous() && hid.scalar_type() == at::kBFloat16 && hid.numel() == x.numel()), "hid");
  TORCH_CHECK(img.is_cuda() && img.numel() == IMG_BYTES && tab.is_cuda() && tab.numel() == TAB_FLOATS, "packed weights");
  auto st = at::cuda::getCurrentCUDAStream();
  const int nsm = num_sms();
  auto g2 = torch::empty_like(go), gx = torch::empty_like(go);
  auto hid_buf = recompute ? torch::empty_like(go) : hid;
  MlpBwdParams p{};
  p.go = cptr<__nv_bfloat16>(go); p.x = cptr<__nv_bfloat16>(x); p.hid = recompute ? nullptr : cptr<__nv_bfloat16>(hid); p.img = cptr<uint8_t>(img); p.tab = cptr<float>(tab);
  p.g2 = mptr<__nv_bfloat16>(g2); p.gx = mptr<__nv_bfloat16>(gx); p.hid_out = recompute ? mptr<__nv_bfloat16>(hid_buf) : nullptr; p.rows = (int)rows;
  const int ntile = (int)((rows + 15) / 16);
  if (recompute) launch_mlp_bwd<true>(p, ntile, st); else launch_mlp_bwd<false>(p, ntile, st);

  auto f32 = go.options().dtype(at::kFloat);
  const int stages_total = (int)((rows + 63) / 64);
  const int slabs0 = std::max(1, nsm / 2);
  const int stages_per_slab = (stages_total + slabs0 - 1) / slabs0;
  const int slab_rows = stages_per_slab * 64;
  const int slabs = (int)((rows + slab_rows - 1) / slab_rows);
  auto part = torch::empty({(int64_t)2 * slabs, (int64_t)D * D}, f32);
  auto csum = torch::empty({(int64_t)2 * slabs, (int64_t)D}, f32);
  DwParams dp{};
  dp.g[0] = cptr<__nv_bfloat16>(go); dp.g[1] = cptr<__nv_bfloat16>(g2); dp.g[2] = dp.g[1];
  dp.a[0] = cptr<__nv_bfloat16>(hid_buf); dp.a[1] = cptr<__nv_bfloat16>(x); dp.a[2] = dp.a[1];
  dp.njob = 2; dp.rows = (int)rows; dp.slabs = slabs; dp.slab_rows = slab_rows;
  dp.part = mptr<float>(part); dp.csum = mptr<float>(csum);
  set_smem(dw_kernel<true>, DW_SMEM_GELU);
  dw_kernel<true><<<2 * slabs, 256, DW_SMEM_GELU, st>>>(dp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto dwh = like_param(go, {D, D}, w_fp32), dwo = like_param(go, {D, D}, w_fp32);
  auto dbh = like_param(go, {D}, b_fp32), dbo = like_param(go, {D}, b_fp32);
  FinParams fp{};
  fp.part = mptr<float>(part); fp.csum = mptr<float>(csum); fp.ln_part = nullptr;
  fp.njob = 2; fp.slabs = slabs; fp.nwarps = 0;
  fp.dw[0] = dwo.data_ptr(); fp.dw[1] = dwh.data_ptr(); fp.dw_fp32[0] = fp.dw_fp32[1] = w_fp32;
  fp.db[0] = dbo.data_ptr(); fp.db[1] = dbh.data_ptr(); fp.db_fp32[0] = fp.db_fp32[1] = b_fp32;
  dw_finalize_kernel<<<((2 * D * D + 2 * D) / 4 + 255) / 256, 256, 0, st>>>(fp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gx, dwh, dwo, dbh, dbo};
}

// ---- the edge LayerNorm's backward from the compressed copy: (dx in dy's dtype, dw, db in w's dtype)
std::vector<torch::Tensor> ln_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, torch::Tensor w) {
  const c10::cuda::CUDAGuard guard(dy.device());
  check_act(x, "x");
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.numel() == x.numel() && (dy.scalar_type() == at::kBFloat16 || dy.scalar_type() == at::kFloat), "dy");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "dy / x: 16-byte aligned");
  TORCH_CHECK(mean.is_cuda() && mean.scalar_type() == at::kFloat && rstd.scalar_type() == at::kFloat && mean.numel() * D == x.numel() && rstd.numel() == mean.numel(), "statistics");
  check_vec(w, "w");
  const int64_t rows = x.numel() / D;
  auto st = at::cuda::getCurrentCUDAStream();
  const int grid = (int)std::max<int64_t>(1, std::min<int64_t>((rows + 63) / 64, (int64_t)num_sms() * 4));       // 4 CTAs of 8 warps per SM keep ~64 KB of loads in flight; fewer partials to add up
  auto dx = torch::empty_like(dy);
  auto part = torch::empty({grid, 256}, dy.options().dtype(at::kFloat));
  auto dw = torch::empty({D}, w.options()), db = torch::empty({D}, w.options());
  LnBwdParams p{};
  p.dy = dy.data_ptr(); p.x = cptr<__nv_bfloat16>(x); p.mean = cptr<float>(mean); p.rstd = cptr<float>(rstd); p.w = w.data_ptr(); p.dx = dx.data_ptr();
  p.part = mptr<float>(part); p.rows = (int)rows; p.dy_fp32 = dy.scalar_type() == at::kFloat; p.w_fp32 = w.scalar_type() == at::kFloat;
  ln_bwd_kernel<<<grid, 256, 0, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ln_finalize_kernel<<<8, 1024, 0, st>>>(cptr<float>(part), grid, dw.data_ptr(), db.data_ptr(), p.w_fp32);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dw, db};
}

// ---- edge dropout: the ATen mask (bool) to one bit per element, and the backward off the bits
torch::Tensor dropout_pack(torch::Tensor mask) {
  const c10::cuda::CUDAGuard guard(mask.device());
  TORCH_CHECK(mask.is_cuda() && mask.is_contiguous() && mask.scalar_type() == at::kBool && reinterpret_cast<uintptr_t>(mask.data_ptr()) % 8 == 0, "mask: contiguous 8-byte aligned bool");
  const int64_t n = mask.numel(), nbytes = (n + 7) / 8;
  auto packed = torch::empty({nbytes}, mask.options().dtype(at::kByte));
  pack_mask_kernel<<<(int)((nbytes + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(cptr<uint8_t>(mask), mptr<uint8_t>(packed), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

torch::Tensor dropout_bwd(torch::Tensor grad, torch::Tensor packed, double scale) {
  const c10::cuda::CUDAGuard guard(grad.device());
  TORCH_CHECK(grad.is_cuda() && grad.is_contiguous() && (grad.scalar_type() == at::kBFloat16 || grad.scalar_type() == at::kFloat) && reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0,
              "grad: contiguous 16-byte aligned bf16 / fp32");
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() && packed.scalar_type() == at::kByte && packed.numel() == (grad.numel() + 7) / 8, "packed mask");
  const int64_t n = grad.numel(), nbytes = (n + 7) / 8;
  auto out = torch::empty_like(grad);
  auto st = at::cuda::getCurrentCUDAStream();
  if (grad.scalar_type() == at::kFloat) dropout_bwd_kernel<true><<<(int)((nbytes + 255) / 256), 256, 0, st>>>(grad.data_ptr(), cptr<uint8_t>(packed), out.data_ptr(), n, (float)scale);
  else dropout_bwd_kernel<false><<<(int)((nbytes + 255) / 256), 256, 0, st>>>(grad.data_ptr(), cptr<uint8_t>(packed), out.data_ptr(), n, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "weight image + vectors for the chain kernels");
  m.def("tail_fwd", &tail_fwd, "edge tail forward");
  m.def("tail_bwd", &tail_bwd, "edge tail backward");
  m.def("mlp_fwd", &mlp_fwd, "edge MLP forward");
  m.def("mlp_bwd", &mlp_bwd, "edge MLP backward");
  m.def("ln_bwd", &ln_bwd, "edge LayerNorm backward from the compressed copy");
  m.def("dropout_pack", &dropout_pack, "dropout mask to one bit per element");
  m.def("dropout_bwd", &dropout_bwd, "dropout backward from the packed mask");
  m.def("dropout_mask", &dropout_mask, "the dropout decisions of the kernels");
}
