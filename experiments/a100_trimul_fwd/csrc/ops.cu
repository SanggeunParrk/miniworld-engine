// ops.cu -- torch bindings of the A100 TriMul forward kernels (K1, K3), on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "k1_sm80.cuh"
#include "k3_sm80.cuh"
#include "contract_sm80.cuh"
#include "b1_sm80.cuh"
#include "b7_sm80.cuh"
#include "b8_sm80.cuh"
#include "b7j_sm80.cuh"
#include "b1g_sm80.cuh"

namespace {

int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}

template <class G>
void launch_k1(const a100::K1Params& p, int grid) {
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::k1_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set = true;
  }
  a100::k1_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_k3(const a100::K3Params& p, int grid) {
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::k3_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set = true;
  }
  a100::k3_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// z [T,128] bf16, mask [L] uint8 or empty, w packed [4CH/64*64, 128] bf16, gamma/beta [128] fp32, ab [2CH, T] bf16 (written)
void k1_impl(torch::Tensor z, torch::Tensor mask, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta, torch::Tensor ab, int64_t L,
             double eps, int64_t grid, torch::Tensor prof, float* zst) {
  const int CH = (int)(ab.size(0) / 2);
  a100::K1Params p;
  p.zst = zst;
  p.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr());
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.gamma = gamma.data_ptr<float>();
  p.beta = beta.data_ptr<float>();
  p.ab = reinterpret_cast<__nv_bfloat16*>(ab.data_ptr());
  p.prof = prof.numel() ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr;
  p.T = (int)z.size(0);
  p.L = (int)L;
#ifndef K1_MT
#define K1_MT 4
#endif
#ifndef K1_TAIL
#define K1_TAIL 0
#endif
  // main launch: 128-token tiles in whole waves of (SMs x 2) persistent CTAs; the tokens of a ragged last wave go to a tail launch of 64-token
  // tiles (one wave), so the last wave is half as long instead of leaving most SMs idle
#ifndef K1_MG
#define K1_MG 2
#endif
#ifndef K1_NST
#define K1_NST 2
#endif
  using M128 = a100::K1Cfg<128, K1_NST, K1_MT, K1_MG>;
  using M256 = a100::K1Cfg<256, K1_NST, K1_MT, K1_MG>;
  using T128 = a100::K1Cfg<128, 2, 2>;
  using T256 = a100::K1Cfg<256, 2, 2>;
  p.eps = (float)eps;
  const int slots = num_sms() * M128::MINB;
  const int tiles = (p.T + M128::BM - 1) / M128::BM;
  int main_tiles = tiles;
  if (K1_TAIL && grid <= 0 && tiles > slots && tiles % slots != 0) {
    const int full = (tiles / slots) * slots;
    const int tail_tiles = (p.T - full * M128::BM + T128::BM - 1) / T128::BM;
    if (tail_tiles <= slots) main_tiles = full;
  }
  p.tok0 = 0;
  p.num_tiles = main_tiles;
  const int g = grid > 0 ? (int)grid : std::min(main_tiles, slots);
  if (CH == 128) launch_k1<M128>(p, g); else if (CH == 256) launch_k1<M256>(p, g);
  if (main_tiles < tiles) {
    a100::K1Params t = p;
    t.tok0 = main_tiles * M128::BM;
    t.num_tiles = (p.T - t.tok0 + T128::BM - 1) / T128::BM;
    if (CH == 128) launch_k1<T128>(t, std::min(t.num_tiles, slots)); else if (CH == 256) launch_k1<T256>(t, std::min(t.num_tiles, slots));
  }
  TORCH_CHECK(CH == 128 || CH == 256, "K1: CH must be 128 or 256");
}
void k1(torch::Tensor z, torch::Tensor mask, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta, torch::Tensor ab, int64_t L,
        double eps, int64_t grid, torch::Tensor prof) {
  k1_impl(z, mask, w, gamma, beta, ab, L, eps, grid, prof, nullptr);
}
// + zst [T, 2]: the LN_in (mean, rstd) of every token, for K3
void k1z(torch::Tensor z, torch::Tensor mask, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta, torch::Tensor ab, int64_t L,
         double eps, torch::Tensor zst) {
  k1_impl(z, mask, w, gamma, beta, ab, L, eps, 0, torch::empty({0}, z.options()), zst.data_ptr<float>());
}

// x [CH, T] bf16, z [T,128] bf16, wo [128, CH] bf16, wg [128,128] bf16, so/bo/sg/bg [128] fp32, out [T,128] bf16 (written)
void k3_impl(torch::Tensor x, torch::Tensor z, torch::Tensor wo, torch::Tensor wg, torch::Tensor so, torch::Tensor bo, torch::Tensor sg,
             torch::Tensor bg, torch::Tensor out, double eps, int64_t grid, torch::Tensor prof, torch::Tensor ds, int64_t L,
             float* st, __nv_bfloat16* xn, const float* gin, const float* bin, const float* zst = nullptr) {
  const int CH = (int)x.size(0);
  a100::K3Params p;
  p.st = st; p.xn = xn; p.gin = gin; p.bin = bin; p.zst = zst;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr());
  p.wo = reinterpret_cast<const __nv_bfloat16*>(wo.data_ptr());
  p.wg = reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr());
  p.so = so.data_ptr<float>(); p.bo = bo.data_ptr<float>(); p.sg = sg.data_ptr<float>(); p.bg = bg.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.prof = prof.numel() ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr;
  p.ds = ds.numel() ? reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr()) : nullptr;
  p.L = (int)L;
  p.T = (int)z.size(0);
  p.num_tiles = (p.T + 31) / 32;
  p.eps = (float)eps;
  const int g = grid > 0 ? (int)grid : std::min(p.num_tiles, num_sms());
#ifndef K3_NG128
#define K3_NG128 3
#endif
  if (CH == 128) launch_k3<a100::K3Cfg<128, K3_NG128>>(p, g);
  else if (CH == 256) launch_k3<a100::K3Cfg<256>>(p, g);
  else TORCH_CHECK(false, "K3: CH must be 128 or 256");
}
void k3(torch::Tensor x, torch::Tensor z, torch::Tensor wo, torch::Tensor wg, torch::Tensor so, torch::Tensor bo, torch::Tensor sg,
        torch::Tensor bg, torch::Tensor out, double eps, int64_t grid, torch::Tensor prof, torch::Tensor ds, int64_t L) {
  k3_impl(x, z, wo, wg, so, bo, sg, bg, out, eps, grid, prof, ds, L, nullptr, nullptr, nullptr, nullptr);
}
// training forward: + st [T, 4] (mu_o, r_o, mu_i, r_i) and xn [T, 128] outputs, gin / bin the LN_in affine
void k3_train(torch::Tensor x, torch::Tensor z, torch::Tensor wo, torch::Tensor wg, torch::Tensor so, torch::Tensor bo, torch::Tensor sg,
              torch::Tensor bg, torch::Tensor out, double eps, torch::Tensor ds, int64_t L, torch::Tensor st, torch::Tensor xn,
              torch::Tensor gin, torch::Tensor bin, torch::Tensor zst) {
  k3_impl(x, z, wo, wg, so, bo, sg, bg, out, eps, 0, torch::empty({0}, x.options()), ds, L, st.data_ptr<float>(),
          reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()), gin.data_ptr<float>(), bin.data_ptr<float>(), zst.numel() ? zst.data_ptr<float>() : nullptr);
}
// inference with K1's LN_in statistics
void k3z(torch::Tensor x, torch::Tensor z, torch::Tensor wo, torch::Tensor wg, torch::Tensor so, torch::Tensor bo, torch::Tensor sg,
         torch::Tensor bg, torch::Tensor out, double eps, torch::Tensor zst) {
  k3_impl(x, z, wo, wg, so, bo, sg, bg, out, eps, 0, torch::empty({0}, x.options()), torch::empty({0}, x.options()), 1, nullptr, nullptr, nullptr,
          nullptr, zst.data_ptr<float>());
}

#ifndef CONTRACT_ST
#define CONTRACT_ST 4
#endif
// a, b [CH, L, L] bf16 planes, x [CH, L, L] bf16 (written); channels [0, h) outgoing (NT), [h, CH) incoming (TN)
void contract(torch::Tensor a, torch::Tensor b, torch::Tensor x, int64_t h, int64_t grid) {
  using G = a100::ContractCfg<CONTRACT_ST>;
  a100::ContractParams p;
  p.a = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
  p.b = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  p.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  p.CH = (int)a.size(0); p.L = (int)a.size(1); p.h = (int)h;
  TORCH_CHECK(p.L % G::BM == 0, "contract: L must be a multiple of 128");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::contract_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set = true; }
  const int total = (p.L / G::BM) * (p.L / G::BN) * p.CH;
  const int g = grid > 0 ? (int)grid : total;
  a100::contract_kernel<G><<<g, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


#ifndef B1_NG128
#define B1_NG128 2
#endif
#ifndef B1_NG256
#define B1_NG256 2
#endif
int b1_ng(int ch) { return ch == 128 ? B1_NG128 : B1_NG256; }
int b1_grid(int num_tiles, int ng) { return std::min((num_tiles + ng - 1) / ng, num_sms()); }
template <class G>
void launch_b1(const a100::B1Params& p) {
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b1_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set = true; }
  a100::b1_kernel<G><<<b1_grid(p.num_tiles, G::NG), G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// B1: x [CH, T], z/dy [T, 128], ds [L, 128] or empty, wo [128, CH], wg [128, 128], fold vectors; outputs dx [CH, T], ao [T, 128], dg (view with
// row stride ldg), stats [T, 4]
void b1(torch::Tensor x, torch::Tensor z, torch::Tensor dy, torch::Tensor ds, torch::Tensor wo, torch::Tensor wg, torch::Tensor so,
        torch::Tensor bo, torch::Tensor sg, torch::Tensor bg, torch::Tensor dx, torch::Tensor ao, torch::Tensor dg,
        int64_t ldg, torch::Tensor stats, torch::Tensor red, torch::Tensor xn, torch::Tensor gin, torch::Tensor bin, int64_t L, double eps) {
  a100::B1Params p;
  p.xn = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()); p.gin = gin.data_ptr<float>(); p.bin = bin.data_ptr<float>();
  const int CH = (int)x.size(0);
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr());
  p.dy = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  p.ds = ds.numel() ? reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr()) : nullptr;
  p.wo = reinterpret_cast<const __nv_bfloat16*>(wo.data_ptr()); p.wg = reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr());
  p.so = so.data_ptr<float>(); p.bo = bo.data_ptr<float>(); p.sg = sg.data_ptr<float>(); p.bg = bg.data_ptr<float>();
  p.dx = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()); p.ao = reinterpret_cast<__nv_bfloat16*>(ao.data_ptr());
  p.dg = reinterpret_cast<__nv_bfloat16*>(dg.data_ptr());
  p.stats = stats.data_ptr<float>(); p.red = red.data_ptr<float>();
  p.T = (int)z.size(0); p.L = (int)L; p.num_tiles = (p.T + 31) / 32;
  TORCH_CHECK(red.numel() == (int64_t)b1_grid(p.num_tiles, b1_ng(CH)) * b1_ng(CH) * 2 * 128, "B1: red must be [b1_red_rows, 4, 128]"); p.ldg = (int)ldg; p.eps = (float)eps;
  if (CH == 128) launch_b1<a100::B1Cfg<128, B1_NG128>>(p); else if (CH == 256) launch_b1<a100::B1Cfg<256, B1_NG256>>(p);
  else TORCH_CHECK(false, "B1: CH must be 128 or 256");
}
// B7src: xn [T,128], w packed [NSTEP*64, 128] (the K1 packing, 0.5-scaled, granule-major blocks), dab [2CH, T], mask [L] or empty,
// dgp (row stride ldd, columns 0 .. 4CH), dw [S, NSTEP*64, 128] fp32 partials
void b7src(torch::Tensor xn, torch::Tensor w, torch::Tensor dab, torch::Tensor mask, torch::Tensor dgp, int64_t ldd, torch::Tensor dw, int64_t L) {
  a100::B7Params p;
  p.xn = reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr());
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()); p.dab = reinterpret_cast<const __nv_bfloat16*>(dab.data_ptr());
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.dgp = reinterpret_cast<__nv_bfloat16*>(dgp.data_ptr()); p.dw = dw.data_ptr<float>();
  p.T = (int)xn.size(0); p.L = (int)L; p.num_tiles = (p.T + 127) / 128; p.nstep = (int)(w.numel() / (64 * 128)); p.splits = (int)dw.size(0); p.ldd = (int)ldd;
  static bool set = false;
  TORCH_CHECK(p.nstep % 2 == 0, "B7src: even block count (pairs of weight blocks per CTA)");
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b7pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::B7PCfg::SMEM)); set = true; }
  a100::b7pair_kernel<<<p.nstep / 2 * p.splits, 256, a100::B7PCfg::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// B8: dgp [T, ldd] (first K columns used), w [K, 128], z / dy [T, 128], stats [T, 4], gamma [128] fp32 -> dz [T, 128], part [grid, 2, 128]
void b8(torch::Tensor dgp, int64_t ldd, torch::Tensor w, torch::Tensor z, torch::Tensor dy, torch::Tensor stats, torch::Tensor gamma,
        torch::Tensor dz, torch::Tensor part) {
  a100::B8Params p;
  p.dgp = reinterpret_cast<const __nv_bfloat16*>(dgp.data_ptr()); p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()); p.dy = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  p.stats = stats.data_ptr<float>(); p.gamma = gamma.data_ptr<float>();
  p.dz = reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()); p.part = part.data_ptr<float>();
  p.T = (int)z.size(0); p.K = (int)w.size(0); p.ldd = (int)ldd; p.num_tiles = (p.T + 127) / 128;   // the tail tile is masked
  TORCH_CHECK(p.K % 32 == 0 && w.size(1) == 128, "B8: K % 32, 128 output channels");
  TORCH_CHECK(part.size(0) <= p.num_tiles, "B8: more CTAs than tiles");
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::B8Cfg::SMEM)); set = true; }
  a100::b8_kernel<<<(int)part.size(0), 128, a100::B8Cfg::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int b7j_per_sm() {
  static int per_sm = -1;
  if (per_sm < 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b7j_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, a100::B7JCfg::SMEM));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, a100::b7j_kernel, 128, a100::B7JCfg::SMEM));
  }
  return per_sm;
}
// B7 joint (cooperative): sources S = w1 blocks, C consumers per group, G groups; ring [G*RINGS*128*K4] bf16, prod [G*RINGS*S] / cons [G*RINGS]
// int32 zeroed; dwpart [G, S*64, 128], lnpart [G*C, 2, 128] fp32
void b7j(torch::Tensor xn, torch::Tensor w1, torch::Tensor dab, torch::Tensor mask, torch::Tensor wdx, torch::Tensor dg, torch::Tensor z,
         torch::Tensor dy, torch::Tensor stats, torch::Tensor gamma, torch::Tensor ring, torch::Tensor prod, torch::Tensor cons, torch::Tensor dz,
         torch::Tensor dwpart, torch::Tensor lnpart, int64_t L, int64_t C, int64_t G, int64_t RINGS) {
  a100::B7JParams p;
  p.xn = reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr()); p.w1 = reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr());
  p.dab = reinterpret_cast<const __nv_bfloat16*>(dab.data_ptr()); p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.wdx = reinterpret_cast<const __nv_bfloat16*>(wdx.data_ptr()); p.dg = reinterpret_cast<const __nv_bfloat16*>(dg.data_ptr());
  p.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()); p.dy = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  p.stats = stats.data_ptr<float>(); p.gamma = gamma.data_ptr<float>();
  p.ring = reinterpret_cast<__nv_bfloat16*>(ring.data_ptr());
  p.prod = reinterpret_cast<unsigned*>(prod.data_ptr<int>()); p.cons = reinterpret_cast<unsigned*>(cons.data_ptr<int>());
  p.dz = reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()); p.dwpart = dwpart.data_ptr<float>(); p.lnpart = lnpart.data_ptr<float>();
  p.T = (int)xn.size(0); p.L = (int)L; p.num_tiles = p.T / 128; p.S = (int)(w1.numel() / (64 * 128)); p.C = (int)C; p.G = (int)G;
  p.RINGS = (int)RINGS; p.K4 = 64 * p.S;
  TORCH_CHECK(p.T % 128 == 0 && p.S <= 32 && wdx.size(0) == p.K4 + 128, "B7J: T % 128, S <= 32, wdx rows");
  TORCH_CHECK(ring.numel() >= (int64_t)p.G * p.RINGS * 128 * p.K4 && prod.numel() >= p.G * p.RINGS * p.S && cons.numel() >= p.G * p.RINGS, "B7J buffers");
  TORCH_CHECK(dwpart.numel() == (int64_t)p.G * p.S * 64 * 128 && lnpart.numel() == (int64_t)p.G * p.C * 256, "B7J partials");
  const int per_sm = b7j_per_sm();
  TORCH_CHECK(p.G >= 1, "B7J: no group fits");
  const int grid = p.G * (p.S + p.C);
  TORCH_CHECK(grid <= per_sm * num_sms(), "B7J: grid ", grid, " exceeds co-resident capacity ", per_sm * num_sms());
  void* args[] = {&p};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)a100::b7j_kernel, dim3(grid), dim3(128), args, a100::B7JCfg::SMEM, at::cuda::getCurrentCUDAStream()));
}

int b1g_grid(int T) { return std::min(((T + 31) / 32 + 1) / 2, num_sms()); }
template <class G>
void launch_b1g(const a100::B1GParams& p, int grid) {
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b1g_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set = true; }
  a100::b1g_kernel<G><<<grid, 256, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// B1 with the forward's saved stats / x_n and the on-chip G = A_o^T X^T: red [grid, 2, 128], gpart [grid, 128, CH] (grid = b1g_grid(T))
void b1g(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor ds, torch::Tensor wo, torch::Tensor wg, torch::Tensor so,
         torch::Tensor bo, torch::Tensor stats, torch::Tensor dx, torch::Tensor dg, int64_t ldg, torch::Tensor red, torch::Tensor gpart, int64_t L) {
  a100::B1GParams p;
  const int CH = (int)x.size(0);
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.xn = reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr());
  p.dy = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  p.ds = ds.numel() ? reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr()) : nullptr;
  p.wo = reinterpret_cast<const __nv_bfloat16*>(wo.data_ptr()); p.wg = reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr());
  p.so = so.data_ptr<float>(); p.bo = bo.data_ptr<float>(); p.stats = stats.data_ptr<float>();
  p.dx = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()); p.dg = reinterpret_cast<__nv_bfloat16*>(dg.data_ptr());
  p.red = red.data_ptr<float>(); p.gpart = gpart.data_ptr<float>();
  p.T = (int)xn.size(0); p.L = (int)L; p.num_tiles = (p.T + 31) / 32; p.ldg = (int)ldg;
  const int grid = b1g_grid(p.T);
  TORCH_CHECK(red.numel() == (int64_t)grid * 256 && gpart.numel() == (int64_t)grid * 128 * CH, "B1G partial shapes");
  if (CH == 128) launch_b1g<a100::B1GCfg<128>>(p, grid); else if (CH == 256) launch_b1g<a100::B1GCfg<256>>(p, grid);
  else TORCH_CHECK(false, "B1G: CH must be 128 or 256");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b1g", &b1g);
  m.def("b1g_grid", [](int64_t T) { return b1g_grid((int)T); });
  m.def("b7j", &b7j);
  m.def("b7j_prof", []() {
    unsigned long long h[4];
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(h, a100::b7j_prof, sizeof(h)));
    const unsigned long long z[4] = {0, 0, 0, 0};
    C10_CUDA_CHECK(cudaMemcpyToSymbol(a100::b7j_prof, z, sizeof(z)));
    return std::vector<double>{(double)h[0], (double)h[1], (double)h[2], (double)h[3]};
  });
  m.def("b7j_capacity", []() { return b7j_per_sm() * num_sms(); });
  m.def("b8", &b8);
  m.def("b1", &b1);
  m.def("b1_red_rows", [](int64_t T, int64_t ch) { const int ng = b1_ng((int)ch); return b1_grid((int)((T + 31) / 32), ng) * ng; });
  m.def("b7src", &b7src);
  m.def("contract", &contract);
  m.def("k1", &k1);
  m.def("k3", &k3);
  m.def("k3_train", &k3_train);
  m.def("k3z", &k3z);
  m.def("k1z", &k1z);
}
