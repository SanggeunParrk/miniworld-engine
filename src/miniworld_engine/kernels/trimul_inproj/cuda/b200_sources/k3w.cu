// k3w.cu -- TriMul output side on B200 (tcgen05) for D = 256 / 384 / 512: both output GEMMs on the raw operands with the
// LayerNorms folded into the weights, and the whole output epilogue in the same kernel:
//   p = rs_o (t^T Wp'^T - mu_o sp) + ep        Wp' = bf16(Wp g_o) [D, H], sp = rowsum(Wp'), ep = Wp b_o
//   g = rs_i (x Wg'^T  - mu_i sg) + eg         Wg' = bf16(Wg g_i) [D, D], sg = rowsum(Wg'), eg = Wg b_i
//   y = bf16(x + bf16(sigmoid(g) p ds))
// t is the channel-major contraction output [H, M] (H = 2D bidirectional, D one direction): it is the MN-major A operand as
// stored, so nothing is transposed or normalised in memory. mu / rs are per-token statistics (mu_o / rs_o over the H channels
// of t, mu_i / rs_i over the D channels of x).
//
// Work item = 128 tokens x 128 output channels (p and g accumulators, 2 x 128 TMEM columns, double buffered). K streams through
// a ring of 32 KB stages (A 16 KB + B 16 KB): H / 64 stages of (t block, Wp' block), then D / 64 of (x block, Wg' block).
// 256 threads: warp 0 TMA (stages), warp 1 MMA, warp 2 TMEM alloc, warp 3 TMA (residual x tile); warps 4-7 epilogue (thread = token).
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace k3w {

constexpr int TOK = 128, NTILE = 128, KBB = 16384, STAGE = 2 * KBB;
// SAVE (training): the epilogue also stores p and g (bf16) for the backward, through two more [128 tok][128 ch] staging tiles;
// the stage ring shrinks from 5 to 3 to make room.
template <int SAVE>
struct Lay {
  static constexpr int NST = SAVE ? 3 : 5;
  static constexpr int O_ST = 0, O_XR = O_ST + NST * STAGE, O_PS = O_XR + 2 * KBB, O_GS = O_PS + (SAVE ? 2 * KBB : 0),
                       O_VEC = O_GS + (SAVE ? 2 * KBB : 0), O_BAR = O_VEC + 4 * NTILE * 4;
  static constexpr int SMEM = O_BAR + 256 + 1024;
  static_assert(SMEM <= 232448, "smem");
};
constexpr int NSTMAX = 5;
constexpr uint32_t ID_P = idesc_bf16_mj(128, NTILE, 1, 0), ID_G = idesc_bf16_mj(128, NTILE, 0, 0);

struct Bars {
  uint64_t st_full[NSTMAX], st_empty[NSTMAX], acc_full[2], acc_empty[2], xr_full, xr_empty;
  uint32_t tmem;
};

DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

template <int D, int H, int SAVE>
__global__ void __launch_bounds__(256, 1)
    k3w_kernel(const __grid_constant__ CUtensorMap mt, const __grid_constant__ CUtensorMap mwp, const __grid_constant__ CUtensorMap mx,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mps,
               const __grid_constant__ CUtensorMap mgs, const float* __restrict__ vec,
               const float* __restrict__ mu_o, const float* __restrict__ rs_o, const float* __restrict__ mu_i,
               const float* __restrict__ rs_i, const __nv_bfloat16* __restrict__ ds, int items, int L) {
  constexpr int KBT = H / 64, KBX = D / 64, NKB = KBT + KBX, NTL = D / NTILE;
  using LY = Lay<SAVE>;
  constexpr int NST = LY::NST, O_ST = LY::O_ST, O_XR = LY::O_XR, O_PS = LY::O_PS, O_GS = LY::O_GS, O_VEC = LY::O_VEC, O_BAR = LY::O_BAR;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.st_full[s], 1); mbar_init(&B.st_empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    mbar_init(&B.xr_full, 1); mbar_init(&B.xr_empty, 1);
    fence_mbar_init();
    prefetch_tmap(&mt); prefetch_tmap(&mwp); prefetch_tmap(&mx); prefetch_tmap(&mwg); prefetch_tmap(&my);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      // ---------------------------------------------------------------- stages: (t block, Wp' block) x KBT, (x block, Wg' block) x KBX
      int s = 0; uint32_t ph = 0;
      for (int w = cta; w < items; w += G) {
        const int m0 = (w / NTL) * TOK, n0 = (w % NTL) * NTILE;
        for (int kb = 0; kb < NKB; ++kb) {
          mbar_wait(&B.st_empty[s], ph ^ 1);
          mbar_expect_tx(&B.st_full[s], STAGE);
          uint8_t* a = sm + O_ST + s * STAGE;
          if (kb < KBT) {
            tma_load_2d(a, &mt, &B.st_full[s], m0, kb * 64, EVICT_NORMAL);             // [64 ch][64 tok], tokens 0-63
            tma_load_2d(a + KBB / 2, &mt, &B.st_full[s], m0 + 64, kb * 64, EVICT_NORMAL);   // tokens 64-127
            tma_load_2d(a + KBB, &mwp, &B.st_full[s], kb * 64, n0, EVICT_LAST);
          } else {
            const int kx = kb - KBT;
            tma_load_2d(a, &mx, &B.st_full[s], kx * 64, m0, EVICT_NORMAL);
            tma_load_2d(a + KBB, &mwg, &B.st_full[s], kx * 64, n0, EVICT_LAST);
          }
          if (++s == NST) { s = 0; ph ^= 1; }
        }
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      // ---------------------------------------------------------------- residual x tile [128 tok][128 ch] (2 K-blocks)
      int i = 0;
      for (int w = cta; w < items; w += G, ++i) {
        const int m0 = (w / NTL) * TOK, n0 = (w % NTL) * NTILE;
        if (i >= 1) mbar_wait(&B.xr_empty, (i - 1) & 1);
        mbar_expect_tx(&B.xr_full, 2 * KBB);
        tma_load_2d(sm + O_XR, &mx, &B.xr_full, n0, m0, EVICT_FIRST);
        tma_load_2d(sm + O_XR + KBB, &mx, &B.xr_full, n0 + 64, m0, EVICT_FIRST);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------ MMA
    const bool ldr = elect_one();
    int s = 0, i = 0; uint32_t ph = 0;
    for (int w = cta; w < items; w += G, ++i) {
      const int b = i & 1;
      mbar_wait(&B.acc_empty[b], ((i >> 1) & 1) ^ 1);
      tc_fence_after();
      const uint32_t dp = tmem + b * 256, dg = dp + 128;
#pragma unroll 1
      for (int kb = 0; kb < NKB; ++kb) {
        mbar_wait(&B.st_full[s], ph);
        tc_fence_after();
        if (ldr) {
          const uint32_t a = su + O_ST + s * STAGE, bw = a + KBB;
          if (kb < KBT) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
              umma_ss(dp, desc_mn_sw128(a + k * 2048, KBB / 2), desc_k_sw128(bw + k * 32), ID_P, (kb | k) != 0);
          } else {
            const int kx = kb - KBT;
#pragma unroll
            for (int k = 0; k < 4; ++k)
              umma_ss(dg, desc_k_sw128(a + k * 32), desc_k_sw128(bw + k * 32), ID_G, (kx | k) != 0);
          }
          umma_commit(&B.st_empty[s]);
        }
        __syncwarp();
        if (++s == NST) { s = 0; ph ^= 1; }
      }
      if (ldr) umma_commit(&B.acc_full[b]);
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------ epilogue: thread = token row r
    const int r = (warp & 3) * 32 + lane, et = tid - 128;
    const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
    float* vs = reinterpret_cast<float*>(sm + O_VEC);     // sp, ep, sg, eg of this item's 128 channels
    int i = 0;
    for (int w = cta; w < items; w += G, ++i) {
      const int m0 = (w / NTL) * TOK, n0 = (w % NTL) * NTILE, b = i & 1, tok = m0 + r;
      named_bar_sync(1, 128);                            // the previous item's epilogue has finished with vs
      for (int k = et; k < 4 * NTILE; k += 128) vs[k] = vec[(k / NTILE) * D + n0 + (k % NTILE)];
      named_bar_sync(1, 128);
      const float mo = mu_o[tok], ro = rs_o[tok], mi = mu_i[tok], ri = rs_i[tok];
      const __nv_bfloat16* dsr = ds ? ds + (size_t)(tok % L) * D + n0 : nullptr;
      mbar_wait(&B.acc_full[b], (i >> 1) & 1);
      tc_fence_after();
      mbar_wait(&B.xr_full, i & 1);
#pragma unroll 1
      for (int cq = 0; cq < 4; ++cq) {                  // 32 channels per step
        uint32_t pv[32], gv[32];
        tmem_ld32(tmem + lane_off + b * 256 + cq * 32, pv);
        tmem_ld32(tmem + lane_off + b * 256 + 128 + cq * 32, gv);
        tmem_wait_ld();
        if (cq == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[b]); }
#pragma unroll
        for (int q2 = 0; q2 < 4; ++q2) {                // 8 channels = one 16-byte chunk of the row
          const int cl = cq * 32 + q2 * 8;              // channel within the tile
          const uint32_t addr = su + O_XR + (cl >> 6) * KBB + sw128(r, (cl & 63) >> 3);
          const uint4 xv = lds128(addr);
          uint4 dv = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
          if (dsr) dv = *reinterpret_cast<const uint4*>(dsr + cl);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, dw[4] = {dv.x, dv.y, dv.z, dv.w};
          uint32_t o[4], po[4], go[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            float u[2], pk[2], gk[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int c = cl + 2 * k + h, j = q2 * 8 + 2 * k + h;
              const float p = fmaf(ro, __uint_as_float(pv[j]) - mo * vs[c], vs[NTILE + c]);
              const float g = fmaf(ri, __uint_as_float(gv[j]) - mi * vs[2 * NTILE + c], vs[3 * NTILE + c]);
              const float sg = rcp_ftz(1.f + ex2_ftz(-1.4426950408889634f * g));
              const float dd = h ? bf16hi(dw[k]) : bf16lo(dw[k]);
              u[h] = __bfloat162float(__float2bfloat16_rn(sg * p * dd));
              pk[h] = p; gk[h] = g;
            }
            o[k] = pack_bf16(bf16lo(xw[k]) + u[0], bf16hi(xw[k]) + u[1]);
            po[k] = pack_bf16(pk[0], pk[1]);
            go[k] = pack_bf16(gk[0], gk[1]);
          }
          sts128(addr, o[0], o[1], o[2], o[3]);
          if (SAVE) {
            const uint32_t off = addr - (su + O_XR);
            sts128(su + O_PS + off, po[0], po[1], po[2], po[3]);
            sts128(su + O_GS + off, go[0], go[1], go[2], go[3]);
          }
        }
      }
      fence_async_smem();
      named_bar_sync(2, 128);
      if (et == 0) {
        tma_store_2d(&my, sm + O_XR, n0, m0);
        tma_store_2d(&my, sm + O_XR + KBB, n0 + 64, m0);
        if (SAVE) {
          tma_store_2d(&mps, sm + O_PS, n0, m0);
          tma_store_2d(&mps, sm + O_PS + KBB, n0 + 64, m0);
          tma_store_2d(&mgs, sm + O_GS, n0, m0);
          tma_store_2d(&mgs, sm + O_GS + KBB, n0 + 64, m0);
        }
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.xr_empty);
      }
    }
    if (et == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k3w

// x [M, D], t [H, M], wpq = bf16(Wp g_o) [D, H], wgq = bf16(Wg g_i) [D, D] bf16; vec [4, D] fp32 = (sp, ep, sg, eg);
// mu_o / rs_o / mu_i / rs_i [M] fp32; ds [L, D] bf16 or none; y [M, D] bf16; p_out / g_out [M, D] bf16 (training, optional):
// the output projection p and the gate logit g, before the gate.
void k3w_forward(torch::Tensor x, torch::Tensor t, torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec, torch::Tensor mu_o,
                 torch::Tensor rs_o, torch::Tensor mu_i, torch::Tensor rs_i, c10::optional<torch::Tensor> ds, torch::Tensor y,
                 int64_t L, int64_t grid, c10::optional<torch::Tensor> p_out, c10::optional<torch::Tensor> g_out) {
  using namespace k3w;
  const int64_t M = x.size(0), D = x.size(1), H = t.size(0);
  TORCH_CHECK(M % TOK == 0 && t.numel() == H * M && wpq.size(0) == D && wpq.size(1) == H && wgq.size(0) == D && wgq.size(1) == D);
  TORCH_CHECK(x.is_contiguous() && t.is_contiguous() && wpq.is_contiguous() && wgq.is_contiguous() && vec.is_contiguous() &&
              y.is_contiguous());
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mt = tmap::make(t.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwp = tmap::make(wpq.data_ptr(), bf, {(uint64_t)H, (uint64_t)D}, {(uint64_t)H * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wgq.data_ptr(), bf, {(uint64_t)D, (uint64_t)D}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto my = tmap::make(y.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  const bool save = p_out.has_value();
  TORCH_CHECK(save == g_out.has_value());
  auto mps = save ? tmap::make(p_out->data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B) : my;
  auto mgs = save ? tmap::make(g_out->data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B) : my;
  const int items = (int)(M / TOK * (D / NTILE));
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, items);
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* dsp = ds.has_value() ? reinterpret_cast<const __nv_bfloat16*>(ds->data_ptr()) : nullptr;
  auto launch = [&](auto kern, int smem) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<g, 256, smem, st>>>(mt, mwp, mx, mwg, my, mps, mgs, vec.data_ptr<float>(), mu_o.data_ptr<float>(), rs_o.data_ptr<float>(),
                               mu_i.data_ptr<float>(), rs_i.data_ptr<float>(), dsp, items, (int)L);
  };
#define K3W_CASE(DD, HH) if (D == DD && H == HH) { \
    if (save) launch(k3w_kernel<DD, HH, 1>, Lay<1>::SMEM); else launch(k3w_kernel<DD, HH, 0>, Lay<0>::SMEM); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K3W_CASE(256, 512) K3W_CASE(256, 256) K3W_CASE(384, 768) K3W_CASE(384, 384) K3W_CASE(512, 1024) K3W_CASE(512, 512)
#undef K3W_CASE
  TORCH_CHECK(false, "k3w_forward: unsupported (D, H) = (", D, ", ", H, ")");
}
