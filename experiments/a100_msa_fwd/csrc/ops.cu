// ops.cu -- torch bindings of the A100 MSA forward kernels (OPM prologue / epilogue, PWA pair / value / main), on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "opm_bwd.cuh"
#include "opm_epilogue.cuh"
#include "opm_prologue.cuh"
#include "pwa_compact.cuh"
#include "pwa_bwd.cuh"
#include "pwa_ctr.cuh"
#include "pwa_main.cuh"
#include "pwa_out.cuh"
#include "pwa_out2.cuh"
#include "pwa_pair.cuh"
#include "pwa_value.cuh"

namespace {

int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}

const __nv_bfloat16* bf(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
__nv_bfloat16* bfw(torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }

}  // namespace

#ifndef OPM_E_NS
#define OPM_E_NS 4
#endif
#ifndef OPM_E_DPS
#define OPM_E_DPS 1
#endif

// msa [T,64] bf16, mask [T] uint8 or empty, w [64,64] bf16, bias [64] fp32 -> a, b [T,32] bf16
void opm_prologue(torch::Tensor msa, torch::Tensor mask, torch::Tensor w, torch::Tensor bias, torch::Tensor a, torch::Tensor b,
                  double eps, int64_t grid) {
  a100::OpmPrologueParams p;
  p.msa = bf(msa);
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.w = bf(w);
  p.bias = bias.data_ptr<float>();
  p.a = bfw(a);
  p.b = bfw(b);
  p.T = (int)msa.size(0);
  p.eps = (float)eps;
  TORCH_CHECK(p.T % 16 == 0, "opm_prologue: T must be a multiple of 16");
  const int steps = p.T / 16;
  const int g = grid > 0 ? (int)grid : std::min((steps + a100::OPM_P_WARPS - 1) / a100::OPM_P_WARPS, num_sms() * 16);
  a100::opm_prologue_kernel<<<g, a100::OPM_P_WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// mask [S*L] uint8 or empty -> bits [L, S/32] int32
void opm_maskbits(torch::Tensor mask, torch::Tensor bits, int64_t S, int64_t L) {
  const int n = (int)(S / 32 * L);
  a100::opm_maskbits_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      mask.numel() ? mask.data_ptr<uint8_t>() : nullptr, reinterpret_cast<uint32_t*>(bits.data_ptr()), (int)S, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// o [32L, 32L] bf16, w [128, 1024] bf16, bias [128] fp32, bits [L, nw] int32, res [L*L,128] bf16 or empty -> out [L*L,128] bf16
// o may be the i-chunk [32 ni, 32 L] of rows i_base .. i_base + ni - 1
void opm_epilogue(torch::Tensor o, torch::Tensor w, torch::Tensor bias, torch::Tensor bits, torch::Tensor res, torch::Tensor out,
                  int64_t L, int64_t i_base, int64_t transposed) {
  using G = a100::OpmECfg<OPM_E_NS, OPM_E_DPS>;
  a100::OpmEpilogueParams p;
  p.o = bf(o);
  p.w = bf(w);
  p.bias = bias.data_ptr<float>();
  p.bits = reinterpret_cast<const uint32_t*>(bits.data_ptr());
  p.res = res.numel() ? bf(res) : nullptr;
  p.out = bfw(out);
  p.L = (int)L;
  p.nw = (int)bits.size(1);
  p.i_base = (int)i_base;
  p.transposed = (int)transposed;
  const int ni = (int)(o.size(0) / 32);
  TORCH_CHECK(L % 32 == 0 && ni % 4 == 0 && o.size(1) == 32 * L, "opm_epilogue: L must be a multiple of 32");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::opm_epilogue_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set = true;
  }
#ifndef OPM_E_BIG
#define OPM_E_BIG 0          // the 256-pair tile (1 CTA / SM) measured 1378 vs 1150 us at L768: rejected
#endif
  if (OPM_E_BIG && ni % 8 == 0) {
    static bool setb = false;
    if (!setb) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(a100::opm_epilogue_big_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::OPM_EB_SMEM));
      setb = true;
    }
    a100::opm_epilogue_big_kernel<<<(int)(ni / 8 * (L / 32)), 256, a100::OPM_EB_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int grid = (int)(ni / G::TI * (L / G::TJ));
  a100::opm_epilogue_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifndef PWA_M_NS
#define PWA_M_NS 2
#endif

// z [L*L,128] bf16, mask [L] uint8 or empty, wb [8,128] bf16, bb [8] fp32 -> w [8,L,L] bf16
// idx / cnt empty: dense w [8,L,L]; else compacted w_c [8,L,ldw] over the keys of idx (cnt = {n, n_pad} on the device)
void pwa_pair(torch::Tensor z, torch::Tensor mask, torch::Tensor wb, torch::Tensor bb, torch::Tensor w, torch::Tensor idx, torch::Tensor cnt,
              int64_t L, double eps) {
  a100::PwaPairParams p;
  p.z = bf(z);
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.wb = bf(wb);
  p.bb = bb.data_ptr<float>();
  p.w = bfw(w);
  p.idx = idx.numel() ? idx.data_ptr<int>() : nullptr;
  p.cnt = idx.numel() ? cnt.data_ptr<int>() : nullptr;
  p.ldw = (int)w.size(2);
  p.L = (int)L;
  p.eps = (float)eps;
  TORCH_CHECK(L % 64 == 0, "pwa_pair: L must be a multiple of 64");
  const int sm = a100::pwa_pair_smem((int)L);
  C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
  a100::pwa_pair_kernel<<<(int)L, a100::PWA_Z_WARPS * 32, sm, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// msa [S*L,64] bf16, wv [256,64] bf16, bv [256] fp32 -> v [8, L, S*32] bf16
void pwa_value(torch::Tensor msa, torch::Tensor wv, torch::Tensor bv, torch::Tensor v, torch::Tensor y, torch::Tensor idx, torch::Tensor cnt,
               int64_t S, int64_t L, double eps, int64_t grid) {
  a100::PwaValueParams p;
  p.msa = bf(msa);
  p.wv = bf(wv);
  p.bv = bv.data_ptr<float>();
  p.v = bfw(v);
  p.y = y.numel() ? bfw(y) : nullptr;
  p.idx = idx.numel() ? idx.data_ptr<int>() : nullptr;
  p.cnt = idx.numel() ? cnt.data_ptr<int>() : nullptr;
  p.ldw = (int)v.size(1);
  TORCH_CHECK(!(p.idx && p.y), "pwa_value: key compaction has no y (the gate-in-out path)");
  p.S = (int)S;
  p.L = (int)L;
  p.eps = (float)eps;
  TORCH_CHECK(S % 16 == 0, "pwa_value: S must be a multiple of 16");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_value_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::PWA_V_SMEM));
    set = true;
  }
#ifndef PWA_V_GRIDMUL
#define PWA_V_GRIDMUL 8
#endif
  const int g = grid > 0 ? (int)grid : num_sms() * PWA_V_GRIDMUL;
  a100::pwa_value_kernel<<<g, a100::PWA_V_WARPS * 32, a100::PWA_V_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// msa [S*L,64], w [8,L,L], v [8,L,S*32], wg [256,64], bg [256] fp32, wof [8,32,32] int32 (Wo B fragments) -> out [S*L,64]
void pwa_main(torch::Tensor msa, torch::Tensor w, torch::Tensor v, torch::Tensor wg, torch::Tensor bg, torch::Tensor wof, torch::Tensor out,
              int64_t S, int64_t L, double eps) {
  using G = a100::PwaMCfg<PWA_M_NS>;
  a100::PwaMainParams p;
  p.msa = bf(msa); p.w = bf(w); p.v = bf(v); p.wg = bf(wg); p.bg = bg.data_ptr<float>(); p.wof = reinterpret_cast<const uint4*>(wof.data_ptr()); p.out = bfw(out);
  p.S = (int)S;
  p.L = (int)L;
  p.eps = (float)eps;
  TORCH_CHECK(L % 128 == 0 && S % 2 == 0, "pwa_main: L must be a multiple of 128 and S even");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_main_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set = true;
  }
  a100::pwa_main_kernel<G><<<(int)(L / 128 * (S / 2)), G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



// split path: y [S*L,64] (LN(msa)), w [8,L,L], v [8,L,S*32], wgf [8,32,32] int32 (Wg' B fragments), bg [256] fp32 -> u [S*L,256]
void pwa_ctr(torch::Tensor y, torch::Tensor w, torch::Tensor v, torch::Tensor wgf, torch::Tensor bg, torch::Tensor u, torch::Tensor cnt,
             int64_t S, int64_t L, double eps) {
  using G = a100::PwaCCfg<2, 64>;
  a100::PwaCtrParams p;
  p.y = bf(y); p.w = bf(w); p.v = bf(v); p.wgf = reinterpret_cast<const uint4*>(wgf.data_ptr()); p.bg = bg.data_ptr<float>();
  p.u = bfw(u);
  p.cnt = cnt.numel() ? cnt.data_ptr<int>() : nullptr;
  p.ldw = (int)w.size(2);
  TORCH_CHECK(p.cnt || p.ldw == L, "pwa_ctr: dense w must be [8, L, L]");
  p.S = (int)S;
  p.L = (int)L;
  p.eps = (float)eps;
  TORCH_CHECK(L % 128 == 0 && S % 4 == 0, "pwa_ctr: L must be a multiple of 128 and S of 4");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_ctr_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set = true;
  }
#ifndef PWA_C_GRID
#define PWA_C_GRID 2
#endif
  const int ntiles_max = (int)(L / 128 * 8 * (S / 4));
  a100::pwa_ctr_kernel<G><<<std::min(ntiles_max, num_sms() * PWA_C_GRID), G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// split path: out [T,64] = msa [T,64] + u [T,256] . wo [64,256]^T
void pwa_out(torch::Tensor msa, torch::Tensor u, torch::Tensor wo, torch::Tensor out, int64_t grid) {
  a100::PwaOutParams p;
  p.msa = bf(msa); p.u = bf(u); p.wo = bf(wo); p.out = bfw(out);
  p.T = (int)msa.size(0);
  TORCH_CHECK(p.T % a100::PWA_O_TT == 0, "pwa_out: T must be a multiple of 128");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_out_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::PWA_O_SMEM));
    set = true;
  }
  const int g = grid > 0 ? (int)grid : std::min(p.T / a100::PWA_O_TT, num_sms());
  a100::pwa_out_kernel<<<g, a100::PWA_O_NTHR, a100::PWA_O_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// gate-in-out split path (default): out [T,64] = msa + sum_h (sigmoid(LN(msa) Wg_h'^T + bg_h) .* o_h) Wo_h^T
void pwa_out2(torch::Tensor msa, torch::Tensor o, torch::Tensor wg, torch::Tensor bg, torch::Tensor wo, torch::Tensor out, double eps,
              int64_t grid, torch::Tensor keep, double scale, int64_t L) {
  a100::PwaOut2Params p;
  p.msa = bf(msa); p.o = bf(o); p.eps = (float)eps; p.wg = bf(wg); p.bg = bg.data_ptr<float>(); p.wo = bf(wo); p.out = bfw(out);
  p.T = (int)msa.size(0);
  p.keep = keep.numel() ? bf(keep) : nullptr;
  p.scale = (float)scale;
  p.L = (int)L;
  TORCH_CHECK(p.T % 16 == 0, "pwa_out2: T must be a multiple of 16");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_out2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::PWA_O2_SMEM));
    set = true;
  }
  const int g = grid > 0 ? (int)grid : num_sms();
  a100::pwa_out2_kernel<<<g, a100::PWA_O2_WARPS * 32, a100::PWA_O2_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool ctr_nogate() { return true; }      // the gate-in-ctr path (PWA_C_GATE) is retired: the gate always runs in pwa_out2

// mask [L] uint8 or empty -> idx [L] int32 (valid keys first), cnt [2] int32 = {n, n_pad}
void pwa_compact(torch::Tensor mask, torch::Tensor idx, torch::Tensor cnt, int64_t L, torch::Tensor posinv) {
  TORCH_CHECK(L <= 1024, "pwa_compact: L <= 1024");
  a100::pwa_compact_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(mask.numel() ? mask.data_ptr<uint8_t>() : nullptr, (int)L,
                                                                             idx.data_ptr<int>(), cnt.data_ptr<int>(),
                                                                             posinv.numel() ? posinv.data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- OPM backward
// dz [L*L,128] bf16, woT [1024,128] bf16, bits [L,nw] -> dzn [L*L,128], dO [32L,32L] bf16, dbo [128] fp32 (+=)
void opm_dgrad(torch::Tensor dz, torch::Tensor woT, torch::Tensor bits, torch::Tensor dzn, torch::Tensor dO, torch::Tensor dbo, int64_t L) {
  a100::OpmDgradParams p;
  p.dz = bf(dz); p.woT = bf(woT); p.bits = reinterpret_cast<const uint32_t*>(bits.data_ptr()); p.dzn = bfw(dzn); p.dO = bfw(dO);
  p.dbo = dbo.data_ptr<float>(); p.L = (int)L; p.nw = (int)bits.size(1);
  TORCH_CHECK(L % 32 == 0, "opm_dgrad: L % 32");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::opm_dgrad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::OPM_DG_SMEM)); set = true; }
  a100::opm_dgrad_kernel<<<(int)(L / 4 * (L / 32)), 256, a100::OPM_DG_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dzn [L*L,128], o [32L,32L] -> dwo [128,1024] fp32 (+=); splits = split-K CTAs per n-block (0: auto)
void opm_dwo(torch::Tensor dzn, torch::Tensor o, torch::Tensor dwo, int64_t L, int64_t splits) {
  a100::OpmDwoParams p;
  p.dzn = bf(dzn); p.o = bf(o); p.dwo = dwo.data_ptr<float>(); p.L = (int)L;
  const int nchunks = (int)(L * (L / a100::OPM_DW_KP_));
  const int sp = splits > 0 ? (int)splits : std::max(1, num_sms() * 2 / 8 * 2);
  p.chunks_per = (nchunks + sp - 1) / sp;
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::opm_dwo_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::OPM_DW_SMEM)); set = true; }
  a100::opm_dwo_kernel<<<8 * sp, 256, a100::OPM_DW_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// msa [T,64], mask [T] uint8 or empty, da/db [T,32], w [64,64] bf16, gamma/beta [64] fp32 -> dmsa [T,64] bf16; dw [64,64], dgamma, dbeta fp32 (+=)
void opm_pbwd(torch::Tensor msa, torch::Tensor mask, torch::Tensor da, torch::Tensor db, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta,
              torch::Tensor dmsa, torch::Tensor dw, torch::Tensor dgamma, torch::Tensor dbeta, double eps, int64_t grid) {
  a100::OpmPbwdParams p;
  p.msa = bf(msa); p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr; p.da = bf(da); p.db = bf(db); p.w = bf(w);
  p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>(); p.dmsa = bfw(dmsa);
  p.dw = dw.data_ptr<float>(); p.dgamma = dgamma.data_ptr<float>(); p.dbeta = dbeta.data_ptr<float>();
  p.T = (int)msa.size(0); p.eps = (float)eps;
  TORCH_CHECK(p.T % 64 == 0, "opm_pbwd: T % 64");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::opm_pbwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::OPM_PB_SMEM)); set = true; }
  const int g = grid > 0 ? (int)grid : std::min(p.T / 64, num_sms() * 4);
  a100::opm_pbwd_kernel<<<g, 128, a100::OPM_PB_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- PWA backward
void pwa_bglue(torch::Tensor msa, torch::Tensor dres, torch::Tensor o, torch::Tensor keep, double scale, torch::Tensor wg, torch::Tensor bg,
               torch::Tensor wo, torch::Tensor dout_h, torch::Tensor dgp, torch::Tensor dwo, int64_t S, int64_t L, double eps, int64_t grid) {
  a100::PwaBglueParams p;
  p.msa = bf(msa); p.dres = bf(dres); p.o = bf(o); p.keep = keep.numel() ? bf(keep) : nullptr; p.scale = (float)scale;
  p.wg = bf(wg); p.bg = bg.data_ptr<float>(); p.wo = bf(wo); p.dout_h = bfw(dout_h); p.dgp = bfw(dgp); p.dwo = dwo.data_ptr<float>();
  p.S = (int)S; p.L = (int)L; p.eps = (float)eps;
  TORCH_CHECK(S % 32 == 0, "pwa_bglue: S % 32");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_bglue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::PB_SMEM)); set = true; }
  const int g = grid > 0 ? (int)grid : num_sms();
  a100::pwa_bglue_kernel<<<g, 256, a100::PB_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dv [T,256] natural = w^T . do  (w [8,L,L], do head-major [8,L,S*32]); idx / cnt: the keys of a compacted w (empty: dense)
void pwa_ctr_dv(torch::Tensor w, torch::Tensor dout_h, torch::Tensor dv, int64_t S, int64_t L, torch::Tensor idx, torch::Tensor cnt) {
  using G = a100::PwaCCfg<2, 64, true, true>;
  a100::PwaCtrParams p;
  p.y = nullptr; p.w = bf(w); p.v = bf(dout_h); p.wgf = nullptr; p.bg = nullptr; p.u = bfw(dv); p.cnt = nullptr; p.ldw = (int)w.size(2);
  p.midx = idx.numel() ? idx.data_ptr<int>() : nullptr; p.mcnt = idx.numel() ? cnt.data_ptr<int>() : nullptr;
  p.S = (int)S; p.L = (int)L; p.eps = 0.f;
  TORCH_CHECK(L % 128 == 0 && S % 4 == 0 && p.ldw == L, "pwa_ctr_dv: shapes");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_ctr_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set = true; }
  const int ntiles_max = (int)(L / 128 * 8 * (S / 4));
  a100::pwa_ctr_kernel<G><<<std::min(ntiles_max, num_sms() * PWA_C_GRID), G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pwa_bproj(torch::Tensor msa, torch::Tensor dres, torch::Tensor dgp, torch::Tensor dv, torch::Tensor w, torch::Tensor gamma,
               torch::Tensor beta, torch::Tensor dmsa, torch::Tensor dw, torch::Tensor dgamma, torch::Tensor dbeta, double eps, int64_t grid,
               torch::Tensor posinv, int64_t L) {
  a100::PwaBprojParams p;
  p.posinv = posinv.numel() ? posinv.data_ptr<int>() : nullptr;
  p.L = (int)L;
  p.msa = bf(msa); p.dres = bf(dres); p.dgp = bf(dgp); p.dv = bf(dv); p.w = bf(w); p.gamma = gamma.data_ptr<float>();
  p.beta = beta.data_ptr<float>(); p.dmsa = bfw(dmsa); p.dw = dw.data_ptr<float>(); p.dgamma = dgamma.data_ptr<float>();
  p.dbeta = dbeta.data_ptr<float>(); p.T = (int)msa.size(0); p.eps = (float)eps;
  TORCH_CHECK(p.T % 64 == 0, "pwa_bproj: T % 64");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_bproj_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::BP_SMEM)); set = true; }
  const int g = grid > 0 ? (int)grid : num_sms();
  a100::pwa_bproj_kernel<<<g, 256, a100::BP_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dw: fp32 split-K partials [ksplit, 8, L, L] of pwa_dw; posinv / cnt empty: dense keys
void pwa_bpair(torch::Tensor z, torch::Tensor w, torch::Tensor dw, torch::Tensor wb, torch::Tensor gamma, torch::Tensor beta, torch::Tensor dz,
               torch::Tensor dwb, torch::Tensor dgamma, torch::Tensor dbeta, int64_t L, double eps, torch::Tensor posinv, torch::Tensor cnt,
               torch::Tensor mask) {
  a100::PwaBpairParams p;
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.z = bf(z); p.w = bf(w); p.dw = dw.data_ptr<float>(); p.ksplit = (int)dw.size(0);
  p.posinv = posinv.numel() ? posinv.data_ptr<int>() : nullptr; p.cnt = posinv.numel() ? cnt.data_ptr<int>() : nullptr; p.wb = wb.data_ptr<float>(); p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>();
  p.dz = bfw(dz); p.dwb = dwb.data_ptr<float>(); p.dgamma = dgamma.data_ptr<float>(); p.dbeta = dbeta.data_ptr<float>();
  p.L = (int)L; p.eps = (float)eps;
  const int sm = (int)(8 * L * 4 + 8 * 10 * 128 * 4);
  C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_bpair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
  a100::pwa_bpair_kernel<<<(int)L, 256, sm, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dw partials [ksplit, 8, L, L] fp32 = do_h . v^T per head (K = S * 32 split ksplit ways); cnt: compacted keys (k-blocks past n_pad skip)
void pwa_dw(torch::Tensor dout_h, torch::Tensor v, torch::Tensor out, torch::Tensor cnt, int64_t L) {
  a100::PwaDwParams p;
  p.a = bf(dout_h); p.b = bf(v); p.out = out.data_ptr<float>(); p.cnt = cnt.numel() ? cnt.data_ptr<int>() : nullptr;
  p.L = (int)L; p.K = (int)dout_h.size(2); p.ksplit = (int)out.size(0);
  TORCH_CHECK(L % 128 == 0 && p.K % (64 * p.ksplit) == 0, "pwa_dw: shapes");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_dw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::DW_SMEM)); set = true; }
  const int nb = (int)(L / 128);
  a100::pwa_dw_kernel<<<nb * nb * p.ksplit * 8, 128, a100::DW_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pwa_bglue2(torch::Tensor msa, torch::Tensor dres, torch::Tensor o, torch::Tensor keep, double scale, torch::Tensor wg, torch::Tensor gamma,
                torch::Tensor beta, torch::Tensor wo, torch::Tensor dout_h, torch::Tensor dyg, torch::Tensor dwo, torch::Tensor dwg, int64_t S,
                int64_t L, double eps, int64_t grid) {
  a100::PwaBglue2Params p;
  p.msa = bf(msa); p.dres = bf(dres); p.o = bf(o); p.keep = keep.numel() ? bf(keep) : nullptr; p.scale = (float)scale; p.wg = bf(wg);
  p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>(); p.wo = bf(wo); p.dout_h = bfw(dout_h); p.dyg = bfw(dyg);
  p.dwo = dwo.data_ptr<float>(); p.dwg = dwg.data_ptr<float>(); p.S = (int)S; p.L = (int)L; p.eps = (float)eps;
  TORCH_CHECK(S % 32 == 0, "pwa_bglue2: S % 32");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_bglue2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::PB2_SMEM)); set = true; }
  const int g = grid > 0 ? (int)grid : num_sms();
  a100::pwa_bglue2_kernel<<<g, 256, a100::PB2_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pwa_bproj2(torch::Tensor msa, torch::Tensor dres, torch::Tensor dyg, torch::Tensor dv, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta,
                torch::Tensor posinv, torch::Tensor dmsa, torch::Tensor dw, torch::Tensor dgamma, torch::Tensor dbeta, int64_t L, double eps,
                int64_t grid) {
  a100::PwaBproj2Params p;
  p.msa = bf(msa); p.dres = bf(dres); p.dyg = bf(dyg); p.dv = bf(dv); p.w = bf(w); p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>();
  p.posinv = posinv.numel() ? posinv.data_ptr<int>() : nullptr; p.dmsa = bfw(dmsa); p.dw = dw.data_ptr<float>();
  p.dgamma = dgamma.data_ptr<float>(); p.dbeta = dbeta.data_ptr<float>(); p.T = (int)msa.size(0); p.L = (int)L; p.eps = (float)eps;
  TORCH_CHECK(p.T % 64 == 0 && L % 64 == 0, "pwa_bproj2: T, L % 64");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_bproj2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::BP2_SMEM)); set = true; }
  const int g = grid > 0 ? (int)grid : num_sms();
  a100::pwa_bproj2_kernel<<<g, 256, a100::BP2_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_prologue", &opm_prologue);
  m.def("opm_maskbits", &opm_maskbits);
  m.def("opm_epilogue", &opm_epilogue);
  m.def("opm_dgrad", &opm_dgrad);
  m.def("opm_dwo", &opm_dwo);
  m.def("opm_pbwd", &opm_pbwd);
  m.def("pwa_pair", &pwa_pair);
  m.def("pwa_value", &pwa_value);
  m.def("pwa_main", &pwa_main);
  m.def("pwa_ctr", &pwa_ctr);
  m.def("pwa_out", &pwa_out);
  m.def("pwa_out2", &pwa_out2);
  m.def("pwa_compact", &pwa_compact);
  m.def("pwa_bglue", &pwa_bglue);
  m.def("pwa_ctr_dv", &pwa_ctr_dv);
  m.def("pwa_bproj", &pwa_bproj);
  m.def("pwa_bpair", &pwa_bpair);
  m.def("pwa_dw", &pwa_dw);
  m.def("pwa_bglue2", &pwa_bglue2);
  m.def("pwa_bproj2", &pwa_bproj2);
  m.def("ctr_nogate", &ctr_nogate);
}
