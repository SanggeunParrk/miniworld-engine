// ops.cu -- the torch extension of the A100 (sm_80) row stages of the AF3 local atom DiT block (kernels/augmented_attention/cuda/sm80_atom_rows), bf16, C = 128.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cond_fwd.cuh"
#include "pre_fwd.cuh"
#include "post_fwd.cuh"
#include "tr_bwd.cuh"
#include "post_bwd.cuh"
#include "pre_bwd.cuh"
#include "cond_bwd.cuh"
#include "wgrad.cuh"

#define CHECK_CUDA_BF16(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kBFloat16, #x " must be a CUDA bf16 tensor")
#define CHECK_CUDA_F32(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a CUDA fp32 tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline bf* bptr_w(torch::Tensor& t) { return reinterpret_cast<bf*>(t.data_ptr()); }
inline int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
}  // namespace

template <class G>
void launch_cond(const lr80::CondFwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::cond_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + G::ROWS - 1) / G::ROWS;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, ntile);
  lr80::cond_fwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// c [M, 128] bf16, w [1024, 128] bf16, bias [1024] fp32, gamma [3, 128] fp32 -> mod [M, 1024] bf16
void local_cond_fwd(torch::Tensor c, torch::Tensor w, torch::Tensor bias, torch::Tensor gamma, torch::Tensor mod, double eps, bool cross, int64_t cfg) {
  CHECK_CUDA_BF16(c); CHECK_CUDA_BF16(w); CHECK_CUDA_BF16(mod); CHECK_CUDA_F32(bias); CHECK_CUDA_F32(gamma);
  CHECK_CONTIG(c); CHECK_CONTIG(w); CHECK_CONTIG(bias); CHECK_CONTIG(gamma); CHECK_CONTIG(mod);
  const int64_t M = c.size(0);
  TORCH_CHECK(c.size(1) == 128 && w.size(0) == 1024 && w.size(1) == 128 && bias.numel() == 1024 && gamma.numel() == 384 && mod.numel() == M * 1024, "bad operand sizes");
  c10::cuda::CUDAGuard guard(c.device());
  lr80::CondFwdParams p{bptr(c), bptr(w), bias.data_ptr<float>(), gamma.data_ptr<float>(), bptr_w(mod), (int)M, (float)eps, cross ? 1 : 0};
  switch (cfg) {
    case 0: launch_cond<lr80::CondCfg<8, 1, 3>>(p); break;
    case 1: launch_cond<lr80::CondCfg<4, 2, 2>>(p); break;
    case 2: launch_cond<lr80::CondCfg<2, 3, 3>>(p); break;
    case 3: launch_cond<lr80::CondCfg<4, 1, 3>>(p); break;
    case 4: launch_cond<lr80::CondCfg<2, 2, 3>>(p); break;
    case 5: launch_cond<lr80::CondCfg<8, 2, 2, 1>>(p); break;
    case 6: launch_cond<lr80::CondCfg<4, 3, 2, 1>>(p); break;
    case 7: launch_cond<lr80::CondCfg<6, 2, 2, 1>>(p); break;
    default: TORCH_CHECK(false, "unknown cond_fwd configuration");
  }
}

template <class G>
void launch_pre(const lr80::PreFwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::pre_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + G::ROWS - 1) / G::ROWS;
  const unsigned ctas = (unsigned)std::min(num_sms(), ntile);
  lr80::pre_fwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x [M, 128], mod [M, 1024] bf16, w [512, 128] = Wq | Wg | Wk | Wv, bq [128] fp32 -> q, k, v, sg [M, 128]; with save also x1s, xkvs
void local_pre_fwd(torch::Tensor x, torch::Tensor mod, torch::Tensor w, torch::Tensor bq, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sg,
                   torch::Tensor x1s, torch::Tensor xkvs, double eps, bool cross, bool save, bool prefetch, int64_t cfg) {
  CHECK_CUDA_BF16(x); CHECK_CUDA_BF16(mod); CHECK_CUDA_BF16(w); CHECK_CUDA_F32(bq); CHECK_CUDA_BF16(q); CHECK_CUDA_BF16(k); CHECK_CUDA_BF16(v); CHECK_CUDA_BF16(sg);
  CHECK_CONTIG(x); CHECK_CONTIG(mod); CHECK_CONTIG(w); CHECK_CONTIG(bq); CHECK_CONTIG(q); CHECK_CONTIG(k); CHECK_CONTIG(v); CHECK_CONTIG(sg);
  const int64_t M = x.size(0);
  TORCH_CHECK(x.size(1) == 128 && mod.numel() == M * 1024 && w.size(0) == 512 && w.size(1) == 128 && bq.numel() == 128, "bad operand sizes");
  if (save) { CHECK_CUDA_BF16(x1s); CHECK_CONTIG(x1s); }
  if (save && cross) { CHECK_CUDA_BF16(xkvs); CHECK_CONTIG(xkvs); }
  c10::cuda::CUDAGuard guard(x.device());
  lr80::PreFwdParams p{bptr(x), bptr(mod), bptr(w), bq.data_ptr<float>(), bptr_w(q), bptr_w(k), bptr_w(v), bptr_w(sg),
                       save ? bptr_w(x1s) : nullptr, save && cross ? bptr_w(xkvs) : nullptr, (int)M, (float)eps, cross ? 1 : 0, prefetch ? 1 : 0};
  switch (cfg) {
    case 0: launch_pre<lr80::PreCfg<8, 2>>(p); break;
    case 1: launch_pre<lr80::PreCfg<16, 1>>(p); break;
    case 2: launch_pre<lr80::PreCfg<12, 1>>(p); break;
    default: TORCH_CHECK(false, "unknown pre_fwd configuration");
  }
}

template <class G>
void launch_post(const lr80::PostFwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::post_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + 15) / 16;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, (ntile + G::NW - 1) / G::NW);
  lr80::post_fwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x, o, sg [M, 128], mod [M, 1024], wo [128, 128], wu [512, 128] = Wa | Wb, wd [128, 256] -> out [M, 128]; with save also u, a2, x2, t
void local_post_fwd(torch::Tensor x, torch::Tensor o, torch::Tensor sg, torch::Tensor mod, torch::Tensor wo, torch::Tensor wu, torch::Tensor wd, torch::Tensor out,
                    torch::Tensor us, torch::Tensor a2s, torch::Tensor x2s, torch::Tensor ts, double eps, bool save, bool prefetch, int64_t cfg) {
  CHECK_CUDA_BF16(x); CHECK_CUDA_BF16(o); CHECK_CUDA_BF16(sg); CHECK_CUDA_BF16(mod); CHECK_CUDA_BF16(wo); CHECK_CUDA_BF16(wu); CHECK_CUDA_BF16(wd); CHECK_CUDA_BF16(out);
  CHECK_CONTIG(x); CHECK_CONTIG(o); CHECK_CONTIG(sg); CHECK_CONTIG(mod); CHECK_CONTIG(wo); CHECK_CONTIG(wu); CHECK_CONTIG(wd); CHECK_CONTIG(out);
  const int64_t M = x.size(0);
  TORCH_CHECK(x.size(1) == 128 && mod.numel() == M * 1024, "bad operand sizes");
  TORCH_CHECK(wo.size(0) == 128 && wo.size(1) == 128 && wu.size(0) == 512 && wu.size(1) == 128 && wd.size(0) == 128 && wd.size(1) == 256, "bad weight sizes");
  if (save) { CHECK_CUDA_BF16(us); CHECK_CUDA_BF16(a2s); CHECK_CUDA_BF16(x2s); CHECK_CUDA_BF16(ts); CHECK_CONTIG(us); CHECK_CONTIG(a2s); CHECK_CONTIG(x2s); CHECK_CONTIG(ts); }
  c10::cuda::CUDAGuard guard(x.device());
  lr80::PostFwdParams p{bptr(x), bptr(o), bptr(sg), bptr(mod), bptr(wo), bptr(wu), bptr(wd), bptr_w(out), save ? bptr_w(us) : nullptr, save ? bptr_w(a2s) : nullptr,
                        save ? bptr_w(x2s) : nullptr, save ? bptr_w(ts) : nullptr, (int)M, (float)eps, prefetch ? 1 : 0};
  switch (cfg) {
    case 0: launch_post<lr80::PostCfg<8>>(p); break;
    case 1: launch_post<lr80::PostCfg<4, 32, 2>>(p); break;
    case 2: launch_post<lr80::PostCfg<8, 32, 1>>(p); break;
    default: TORCH_CHECK(false, "unknown post_fwd configuration");
  }
}

template <class G>
void launch_tr_bwd(const lr80::TrBwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::tr_bwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ngrp = ((p.M + 15) / 16 + G::NW - 1) / G::NW;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, ngrp);
  lr80::tr_bwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dy, t, x2 [M, 128], mod [M, 1024], wu [512, 128], wst [256, 128] = Ws^T -> dt [M, 128], hh [M, 256], dab [M, 512], dts into dmod columns 896 .., d bts into dbts [128] (accumulated)
void local_tr_bwd(torch::Tensor dy, torch::Tensor t, torch::Tensor x2, torch::Tensor mod, torch::Tensor wu, torch::Tensor wst, torch::Tensor dt, torch::Tensor hh,
                  torch::Tensor dab, torch::Tensor dmod, torch::Tensor dbts, bool prefetch, int64_t cfg) {
  CHECK_CUDA_BF16(dy); CHECK_CUDA_BF16(t); CHECK_CUDA_BF16(x2); CHECK_CUDA_BF16(mod); CHECK_CUDA_BF16(wu); CHECK_CUDA_BF16(wst);
  CHECK_CUDA_BF16(dt); CHECK_CUDA_BF16(hh); CHECK_CUDA_BF16(dab); CHECK_CUDA_BF16(dmod); CHECK_CUDA_F32(dbts);
  CHECK_CONTIG(dy); CHECK_CONTIG(t); CHECK_CONTIG(x2); CHECK_CONTIG(mod); CHECK_CONTIG(wu); CHECK_CONTIG(wst); CHECK_CONTIG(dt); CHECK_CONTIG(hh); CHECK_CONTIG(dab); CHECK_CONTIG(dmod);
  const int64_t M = dy.size(0);
  TORCH_CHECK(dy.size(1) == 128 && mod.numel() == M * 1024 && dmod.numel() == M * 1024 && wu.size(0) == 512 && wst.size(0) == 256 && dbts.numel() == 128, "bad operand sizes");
  c10::cuda::CUDAGuard guard(dy.device());
  lr80::TrBwdParams p{bptr(dy), bptr(t), bptr(x2), bptr(mod), bptr(wu), bptr(wst), bptr_w(dt), bptr_w(hh), bptr_w(dab), bptr_w(dmod), dbts.data_ptr<float>(), (int)M, prefetch ? 1 : 0};
  switch (cfg) {
    case 0: launch_tr_bwd<lr80::TrBwdCfg<4, 2>>(p); break;
    case 1: launch_tr_bwd<lr80::TrBwdCfg<8, 1>>(p); break;
    default: TORCH_CHECK(false, "unknown tr_bwd configuration");
  }
}

template <class G>
void launch_post_bwd(const lr80::PostBwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::post_bwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + 15) / 16;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, (ntile + G::NW - 1) / G::NW);
  lr80::post_bwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void local_post_bwd(torch::Tensor dab, torch::Tensor dy, torch::Tensor a2, torch::Tensor mod, torch::Tensor u, torch::Tensor sg, torch::Tensor o, torch::Tensor wut, torch::Tensor wot,
                    torch::Tensor da, torch::Tensor du, torch::Tensor gated, torch::Tensor dO, torch::Tensor dp, torch::Tensor dmod, torch::Tensor dD, torch::Tensor dbias, int64_t N,
                    double eps, bool prefetch, int64_t cfg) {
  CHECK_CUDA_BF16(dab); CHECK_CUDA_BF16(dy); CHECK_CUDA_BF16(a2); CHECK_CUDA_BF16(mod); CHECK_CUDA_BF16(u); CHECK_CUDA_BF16(sg); CHECK_CUDA_BF16(o); CHECK_CUDA_BF16(wut); CHECK_CUDA_BF16(wot);
  CHECK_CUDA_BF16(da); CHECK_CUDA_BF16(du); CHECK_CUDA_BF16(gated); CHECK_CUDA_BF16(dO); CHECK_CUDA_BF16(dp); CHECK_CUDA_BF16(dmod); CHECK_CUDA_F32(dD); CHECK_CUDA_F32(dbias);
  CHECK_CONTIG(dab); CHECK_CONTIG(dy); CHECK_CONTIG(a2); CHECK_CONTIG(mod); CHECK_CONTIG(u); CHECK_CONTIG(sg); CHECK_CONTIG(o); CHECK_CONTIG(wut); CHECK_CONTIG(wot);
  CHECK_CONTIG(da); CHECK_CONTIG(du); CHECK_CONTIG(gated); CHECK_CONTIG(dO); CHECK_CONTIG(dp); CHECK_CONTIG(dmod); CHECK_CONTIG(dD); CHECK_CONTIG(dbias);
  const int64_t M = dy.size(0);
  TORCH_CHECK(dy.size(1) == 128 && dab.numel() == M * 512 && mod.numel() == M * 1024 && dp.numel() == M * 512 && dmod.numel() == M * 1024 && wut.size(0) == 128 && wut.size(1) == 512 &&
              wot.size(0) == 128 && wot.size(1) == 128 && dbias.numel() == 1024 && M % N == 0, "bad operand sizes");
  c10::cuda::CUDAGuard guard(dy.device());
  lr80::PostBwdParams p{bptr(dab), bptr(dy), bptr(a2), bptr(mod), bptr(u), bptr(sg), bptr(o), bptr(wut), bptr(wot), bptr_w(da), bptr_w(du), bptr_w(gated), bptr_w(dO), bptr_w(dp),
                        bptr_w(dmod), dD.data_ptr<float>(), dbias.data_ptr<float>(), (int)M, (int)N, (float)eps, prefetch ? 1 : 0};
  switch (cfg) {
    case 0: launch_post_bwd<lr80::PostBwdCfg<8>>(p); break;
    case 1: launch_post_bwd<lr80::PostBwdCfg<4, true, 2>>(p); break;
    default: TORCH_CHECK(false, "unknown post_bwd configuration");
  }
}

template <class G>
void launch_pre_bwd(const lr80::PreBwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::pre_bwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + 15) / 16;
  const unsigned ctas = (unsigned)std::min(num_sms(), (ntile + G::NW - 1) / G::NW);
  lr80::pre_bwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void local_pre_bwd(torch::Tensor dp, torch::Tensor x, torch::Tensor x1, torch::Tensor da, torch::Tensor mod, torch::Tensor wqgt, torch::Tensor wkvt, torch::Tensor ds, torch::Tensor dmod,
                   torch::Tensor dbias, torch::Tensor dbq, bool cross, double eps, bool prefetch, int64_t cfg) {
  CHECK_CUDA_BF16(dp); CHECK_CUDA_BF16(x); CHECK_CUDA_BF16(da); CHECK_CUDA_BF16(mod); CHECK_CUDA_BF16(wqgt); CHECK_CUDA_BF16(wkvt); CHECK_CUDA_BF16(ds); CHECK_CUDA_BF16(dmod);
  CHECK_CUDA_F32(dbias); CHECK_CUDA_F32(dbq);
  CHECK_CONTIG(dp); CHECK_CONTIG(x); CHECK_CONTIG(da); CHECK_CONTIG(mod); CHECK_CONTIG(wqgt); CHECK_CONTIG(wkvt); CHECK_CONTIG(ds); CHECK_CONTIG(dmod);
  if (cross) { CHECK_CUDA_BF16(x1); CHECK_CONTIG(x1); }
  const int64_t M = x.size(0);
  TORCH_CHECK(x.size(1) == 128 && dp.numel() == M * 512 && mod.numel() == M * 1024 && dmod.numel() == M * 1024 && wqgt.size(0) == 128 && wqgt.size(1) == 256 && wkvt.size(0) == 128 &&
              wkvt.size(1) == 256 && dbias.numel() == 1024 && dbq.numel() == 128, "bad operand sizes");
  c10::cuda::CUDAGuard guard(x.device());
  lr80::PreBwdParams p{bptr(dp), bptr(x), cross ? bptr(x1) : nullptr, bptr(da), bptr(mod), bptr(wqgt), bptr(wkvt), bptr_w(ds), bptr_w(dmod), dbias.data_ptr<float>(), dbq.data_ptr<float>(),
                       (int)M, cross ? 1 : 0, (float)eps, prefetch ? 1 : 0};
  switch (cfg) {
    case 0: launch_pre_bwd<lr80::PreBwdCfg<8>>(p); break;
    default: TORCH_CHECK(false, "unknown pre_bwd configuration");
  }
}

template <class G>
void launch_cond_bwd(const lr80::CondBwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::cond_bwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + 16 * G::NW - 1) / (16 * G::NW);
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, ntile);
  lr80::cond_bwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dmod [M, 1024], c [M, 128], wt [4, 128, 256] (the variants' transposed weights), gamma [3, 128] -> dc [M, 128], cn [3, M, 128], d gamma [3, 128] (accumulated)
void local_cond_bwd(torch::Tensor dmod, torch::Tensor c, torch::Tensor wt, torch::Tensor gamma, torch::Tensor dc, torch::Tensor cn, torch::Tensor dgamma, bool cross, double eps, bool prefetch, int64_t cfg) {
  CHECK_CUDA_BF16(dmod); CHECK_CUDA_BF16(c); CHECK_CUDA_BF16(wt); CHECK_CUDA_F32(gamma); CHECK_CUDA_BF16(dc); CHECK_CUDA_BF16(cn); CHECK_CUDA_F32(dgamma);
  CHECK_CONTIG(dmod); CHECK_CONTIG(c); CHECK_CONTIG(wt); CHECK_CONTIG(gamma); CHECK_CONTIG(dc); CHECK_CONTIG(cn); CHECK_CONTIG(dgamma);
  const int64_t M = c.size(0);
  TORCH_CHECK(c.size(1) == 128 && dmod.numel() == M * 1024 && wt.numel() == 4 * 128 * 256 && gamma.numel() == 384 && cn.numel() == 3 * M * 128 && dgamma.numel() == 384, "bad operand sizes");
  c10::cuda::CUDAGuard guard(c.device());
  lr80::CondBwdParams p{bptr(dmod), bptr(c), bptr(wt), gamma.data_ptr<float>(), bptr_w(dc), bptr_w(cn), dgamma.data_ptr<float>(), (int)M, cross ? 1 : 0, (float)eps, prefetch ? 1 : 0};
  switch (cfg) {
    case 0: launch_cond_bwd<lr80::CondBwdCfg<4, 2>>(p); break;
    case 1: launch_cond_bwd<lr80::CondBwdCfg<8, 1>>(p); break;
    case 3: launch_cond_bwd<lr80::CondBwdCfg<8, 1, 4>>(p); break;
    case 4: launch_cond_bwd<lr80::CondBwdCfg<8, 1, 5>>(p); break;
    case 5: launch_cond_bwd<lr80::CondBwdCfg<4, 2, 3>>(p); break;
    default: TORCH_CHECK(false, "unknown cond_bwd configuration");
  }
}

// the grouped weight gradients: problems D_i = A_i^T B_i (bf16 views, any row strides; D [Ka][Kb]) over their M rows in ``splits`` partial sums (part [splits, slab] fp32), then the segments
// (a range of a problem's partial sums added over the splits, or a plain fp32 copy) rounded into the weights' gradient tensors
void local_wgrad(std::vector<torch::Tensor> A, std::vector<torch::Tensor> B, std::vector<int64_t> KA, std::vector<int64_t> KB, int64_t M, int64_t splits, torch::Tensor part,
                 std::vector<torch::Tensor> src, std::vector<int64_t> src_off, std::vector<int64_t> src_n, std::vector<int64_t> src_nsplit, std::vector<torch::Tensor> dst) {
  const int np = (int)A.size(), ns = (int)src.size();
  TORCH_CHECK(np > 0 && np <= lr80::WG_MAXP && (int)B.size() == np && (int)KA.size() == np && (int)KB.size() == np, "bad problem list");
  TORCH_CHECK(ns <= lr80::WG_MAXS && (int)src_off.size() == ns && (int)src_n.size() == ns && (int)src_nsplit.size() == ns && (int)dst.size() == ns, "bad segment list");
  CHECK_CUDA_F32(part); CHECK_CONTIG(part);
  c10::cuda::CUDAGuard guard(part.device());
  lr80::WgParams p;
  memset(&p, 0, sizeof(p));
  long long slab = 0;
  int tiles = 0;
  for (int i = 0; i < np; ++i) {
    CHECK_CUDA_BF16(A[i]); CHECK_CUDA_BF16(B[i]);
    TORCH_CHECK(A[i].dim() == 2 && B[i].dim() == 2 && A[i].stride(1) == 1 && B[i].stride(1) == 1 && A[i].size(0) == M && B[i].size(0) == M, "operand ", i, " must be [M, K] with unit column stride");
    TORCH_CHECK(KA[i] % 128 == 0 && KB[i] % 128 == 0 && A[i].size(1) >= KA[i] && B[i].size(1) >= KB[i], "bad problem sizes");
    TORCH_CHECK(A[i].stride(0) % 8 == 0 && B[i].stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(A[i].data_ptr()) & 15) == 0 && (reinterpret_cast<uintptr_t>(B[i].data_ptr()) & 15) == 0, "operands must be 16-byte aligned");
    p.prob[i] = {bptr(A[i]), bptr(B[i]), A[i].stride(0), B[i].stride(0), slab, (int)KA[i], (int)KB[i]};
    p.tile_prefix[i] = tiles;
    tiles += (int)(KA[i] / 128 * (KB[i] / 128));
    slab += KA[i] * KB[i];
  }
  p.tile_prefix[np] = tiles;
  p.nprob = np; p.M = (int)M; p.splits = (int)splits; p.part = part.data_ptr<float>(); p.slab = slab;
  p.rps = (int)((((M + splits - 1) / splits) + 63) / 64 * 64);
  TORCH_CHECK(part.numel() >= splits * slab, "part is too small: need ", splits * slab);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lr80::wgrad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, lr80::WG_SMEM);
    attr = true;
  }
  lr80::wgrad_kernel<<<(unsigned)(tiles * splits), 256, lr80::WG_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  lr80::WgSegs sg;
  memset(&sg, 0, sizeof(sg));
  long long maxn = 0;
  for (int i = 0; i < ns; ++i) {
    CHECK_CUDA_F32(src[i]);
    TORCH_CHECK(dst[i].is_cuda() && dst[i].is_contiguous() && (dst[i].scalar_type() == torch::kBFloat16 || dst[i].scalar_type() == torch::kFloat32) && dst[i].numel() == src_n[i], "bad segment destination ", i);
    TORCH_CHECK(src_n[i] % 2 == 0 && src_off[i] % 2 == 0, "segment ", i, " must be an even range");
    sg.seg[i] = {src[i].data_ptr<float>() + src_off[i], dst[i].data_ptr(), src_n[i], slab, (int)src_nsplit[i], dst[i].scalar_type() == torch::kBFloat16 ? 1 : 0};
    maxn = std::max<long long>(maxn, src_n[i]);
  }
  const dim3 grid((unsigned)((maxn / 2 + 255) / 256), (unsigned)ns);
  lr80::wgrad_reduce_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(sg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("local_cond_fwd", &local_cond_fwd);
  m.def("local_pre_fwd", &local_pre_fwd);
  m.def("local_post_fwd", &local_post_fwd);
  m.def("local_tr_bwd", &local_tr_bwd);
  m.def("local_post_bwd", &local_post_bwd);
  m.def("local_pre_bwd", &local_pre_bwd);
  m.def("local_cond_bwd", &local_cond_bwd);
  m.def("local_wgrad", &local_wgrad);
}
