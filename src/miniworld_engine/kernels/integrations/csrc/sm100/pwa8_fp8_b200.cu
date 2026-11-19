// pwa8_fp8_b200.cu -- EXPERIMENTAL (opt-in e4m3 route, docs/b200/fp8.md): the MSAPairWeightedAveraging key contraction of ALL heads in ONE launch on tcgen05 kind::f8f6f4:
//   o[h][i][c] = dq * sum_k w8[h][i][k] * v8[h][k][c]        w8 [H, M, K] e4m3 (K contiguous: a K-major A operand), v8 [H, K, NC] e4m3 (NC = S C contiguous: an MN-major B operand), o [H, M, NC] bf16
// The structure of ctr8.cu (the TriMul e4m3 contraction: TMA producer, one MMA thread, TMEM double buffer, 8 epilogue warps) with the two operands in different majors (A K-major
// like its outgoing form, B MN-major like its incoming form), the plane index = the head, and a rectangular output (M x NC) instead of a square L x L. dq = sw sv (a device scalar).
// Replaces the eight per-head `torch._scaled_mm` launches (a single equal-flop cuBLASLt launch measured 0.64-1.0 of the eight, docs/b200/fp8.md).
#include "../../../trimul_inproj/cuda/b200_sources/sm100.cuh"
#include "../../../trimul_inproj/cuda/b200_sources/tmap.h"
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

using namespace sm100;

namespace pwa8 {

constexpr int NTHR = 384, TOK = 128, BOX = TOK * 128;          // [128][128 B] e4m3 box = 16 KB
constexpr int OSTW = 32 * 64 * 2;                               // per epilogue warp: one [32 rows][64 cols] bf16 box = 4 KB

template <int BN, int KB>
struct Cfg {
  static constexpr int NBA = KB, NBB = KB * BN / 128, NB = NBA + NBB;   // boxes of a stage
  static constexpr int STG = NB * BOX;
  static constexpr int NST = (BN == 128) ? (KB == 2 ? 3 : 6) : (KB == 2 ? 2 : 4);
  static constexpr int O_ST = 0, O_OUT = NST * STG, O_BAR = O_OUT + 8 * OSTW, SMEM = O_BAR + 256 + 1024;
  static_assert(SMEM <= 232448, "smem");
  static_assert(NB <= 32, "lanes");
};

template <int NST>
struct Bars {
  uint64_t full[NST], empty[NST], acc_full[2], acc_empty[2];
  uint32_t tmem;
};

DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }

// kind::f8f6f4, e4m3 x e4m3 -> f32, M x N, a_mn / b_mn = 1: MN-major operand
__host__ __device__ constexpr uint32_t idesc_e4m3(int M, int N, int a_mn, int b_mn) {
  return (1u << 4) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
DEV void umma_f8(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(tmem_d),
      "l"(da), "l"(db), "r"(idesc), "r"(accumulate)
      : "memory");
}

struct Item {
  int h, i0, j0;
};
DEV Item item_of(int it, int M, int NC, int bn) {
  const int MT = (M + TOK - 1) / TOK, NT = (NC + bn - 1) / bn;
  // row tiles fastest: the 148 concurrent CTAs cover MT row tiles x a band of column tiles, so the v8 column tile (K x bn bytes) is read once from HBM and re-used by the MT row tiles from L2
  const int mt = it % MT, r1 = it / MT, nt = r1 % NT, h = r1 / NT;
  Item m;
  m.h = h; m.i0 = mt * TOK; m.j0 = nt * bn;
  return m;
}

template <int BN, int KB, int MINB = 1>
__global__ void __launch_bounds__(NTHR, MINB)
    pwa8_kernel(const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mo, const float* __restrict__ dq,
                int M, int K, int NC, int items) {
  using G_ = Cfg<BN, KB>;
  constexpr int bn = BN, NST = G_::NST, STG = G_::STG, NB = G_::NB, O_ST = G_::O_ST, O_OUT = G_::O_OUT, O_BAR = G_::O_BAR;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars<NST>& B = *reinterpret_cast<Bars<NST>*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int KS = (K + TOK * KB - 1) / (TOK * KB);          // K stages per item (the tail reads zeros)
  constexpr int EG = 2;
  constexpr int BB = bn * 128;            // one B K-box set: [128 k rows][bn cols] as BN / 128 boxes of 128 cols
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4 * EG); }
    fence_mbar_init();
    prefetch_tmap(&mw); prefetch_tmap(&mv); prefetch_tmap(&mo);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 2 * BN); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane < NB) {   // the producer: NB lanes each issue one box of a stage
      int q = 0;
      for (int it = cta; it < items; it += G) {
        const Item m = item_of(it, M, NC, bn);
        for (int ks = 0; ks < KS; ++ks, ++q) {
          const int slot = q % NST;
          if (q >= NST) mbar_wait(&B.empty[slot], ((q / NST) - 1) & 1);
          uint8_t* a = sm + O_ST + slot * STG;
          uint8_t* b = a + KB * BOX;
          const int k0 = ks * TOK * KB;
          if (lane == 0) mbar_expect_tx(&B.full[slot], NB * BOX);
          __syncwarp((1u << NB) - 1u);
          if (lane < KB) tma_load_3d(a + lane * BOX, &mw, &B.full[slot], k0 + 128 * lane, m.i0, m.h, EVICT_NORMAL);   // A [128 rows i][128 k] x KB
          else { const int hh = (lane - KB) / (BN / 128), r = (lane - KB) % (BN / 128);                               // B [128 k rows][128 cols] boxes
                 tma_load_3d(b + hh * BB + r * BOX, &mv, &B.full[slot], m.j0 + 128 * r, k0 + 128 * hh, m.h, EVICT_NORMAL); }
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      const uint32_t id = idesc_e4m3(128, bn, 0, 1);
      int q = 0, n = 0;
      for (int it = cta; it < items; it += G, ++n) {
        const int bb = n & 1;
        if (n >= 2) mbar_wait(&B.acc_empty[bb], ((n >> 1) - 1) & 1);
        tc_fence_after();
        const uint32_t acc = tmem + bb * bn;
        for (int ks = 0; ks < KS; ++ks, ++q) {
          const int slot = q % NST;
          mbar_wait(&B.full[slot], (q / NST) & 1);
          tc_fence_after();
          const uint32_t a = su + O_ST + slot * STG, b = a + KB * BOX;
#pragma unroll
          for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)   // A K-major (32 k = 32 B per MMA), B MN-major (32 k rows = 4096 B per MMA, the next 128 cols a box away)
              umma_f8(acc, desc_k_sw128(a + kb * BOX + kk * 32), desc_mn_sw128(b + kb * BB + kk * 4096, BOX), id, (ks | kb | kk) != 0);
          }
          umma_commit(&B.empty[slot]);
        }
        umma_commit(&B.acc_full[bb]);
      }
    }
  } else if (warp >= 4 && warp < 4 + 4 * EG) {
    const int qd = warp & 3, h = (warp - 4) >> 2;
    int n = 0;
    const float cs = __ldg(dq);
    for (int it = cta; it < items; it += G, ++n) {
      const Item m = item_of(it, M, NC, bn);
      const int bb = n & 1;
      mbar_wait(&B.acc_full[bb], (n >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int hh = 0; hh < BN / 128; ++hh) {
        const int hc = h + 2 * hh;
        const uint32_t ta = tmem + ((uint32_t)(qd * 32) << 16) + hc * 64;
        const uint32_t stg = su + O_OUT + (warp - 4) * OSTW;
        if (lane == 0) bulk_wait_read<0>();
        __syncwarp();
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          uint32_t v[32];
          tmem_ld32(ta + bb * bn + cc * 32, v);
          tmem_wait_ld();
          if (cc == 1 && hh == BN / 128 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[bb]); }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            auto pk = [&](int e) { return pack_bf16(__uint_as_float(v[8 * k + 2 * e]) * cs, __uint_as_float(v[8 * k + 2 * e + 1]) * cs); };
            sts128(stg + sw128(lane, cc * 4 + k), pk(0), pk(1), pk(2), pk(3));
          }
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          tma_store_3d(&mo, sm + O_OUT + (warp - 4) * OSTW, m.j0 + 64 * hc, m.i0 + qd * 32, m.h);
          bulk_commit();
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 2 * BN);
}

static int num_sms_() {
  static int n = 0;
  if (!n) { int dv; cudaGetDevice(&dv); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dv); }
  return n;
}

template <int BN, int KB>
static void launch(const CUtensorMap& mw, const CUtensorMap& mv, const CUtensorMap& mo, const float* dq, int M, int K, int NC, int items, int g) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(pwa8_kernel<BN, KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<BN, KB>::SMEM); attr = true; }
  pwa8_kernel<BN, KB><<<g, NTHR, Cfg<BN, KB>::SMEM, at::cuda::getCurrentCUDAStream()>>>(mw, mv, mo, dq, M, K, NC, items);
}

}  // namespace pwa8

// w8 [H, M, K] e4m3 contiguous, v8 [H, K, NC] e4m3 contiguous, o [H, M, NC] bf16 contiguous, dq [1] fp32 (device): o[h] = dq (w8[h] . v8[h]); M % 128 == 0, K % 16 == 0, NC % 128 == 0. cfg: 0 = BN 256 (NC % 256 == 0) else 128; 100 KB + BN / 128 forces.
void pwa8_forward(torch::Tensor w8, torch::Tensor v8, torch::Tensor o, torch::Tensor dq, int64_t grid, int64_t cfg) {
  using namespace pwa8;
  TORCH_CHECK(w8.is_cuda() && w8.is_contiguous() && w8.scalar_type() == at::kFloat8_e4m3fn && w8.dim() == 3, "w8: contiguous e4m3 [H, M, K]");
  TORCH_CHECK(v8.is_contiguous() && v8.scalar_type() == at::kFloat8_e4m3fn && v8.dim() == 3 && v8.size(0) == w8.size(0) && v8.size(1) == w8.size(2), "v8: contiguous e4m3 [H, K, NC]");
  const int64_t H = w8.size(0), M = w8.size(1), K = w8.size(2), NC = v8.size(2);
  TORCH_CHECK(o.is_contiguous() && o.scalar_type() == at::kBFloat16 && o.dim() == 3 && o.size(0) == H && o.size(1) == M && o.size(2) == NC, "o: contiguous bf16 [H, M, NC]");
  TORCH_CHECK(dq.scalar_type() == at::kFloat && dq.is_cuda() && dq.numel() >= 1, "dq: fp32 device scalar");
  TORCH_CHECK(M % 128 == 0 && K % 16 == 0 && NC % 128 == 0, "pwa8_forward: M % 128, K % 16, NC % 128");
  const c10::cuda::CUDAGuard gd(w8.device());
  auto mw = tmap::make(w8.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, {(uint64_t)K, (uint64_t)M, (uint64_t)H}, {(uint64_t)K, (uint64_t)M * K}, {128, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mv = tmap::make(v8.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, {(uint64_t)NC, (uint64_t)K, (uint64_t)H}, {(uint64_t)NC, (uint64_t)K * NC}, {128, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mo = tmap::make(o.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {(uint64_t)NC, (uint64_t)M, (uint64_t)H}, {(uint64_t)NC * 2, (uint64_t)M * NC * 2}, {64, 32, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  int bn = (NC % 256 == 0) ? 256 : 128, kb = 1;
  if (cfg > 0) { kb = (int)(cfg / 100); bn = 128 * (int)(cfg % 100); }
  const int items = (int)(H * ((M + TOK - 1) / TOK) * ((NC + bn - 1) / bn));
  int g = grid > 0 ? (int)grid : num_sms_();
  g = std::min(g, items);
  const float* dp = dq.data_ptr<float>();
  if (bn == 256 && kb == 2) launch<256, 2>(mw, mv, mo, dp, (int)M, (int)K, (int)NC, items, g);
  else if (bn == 256) launch<256, 1>(mw, mv, mo, dp, (int)M, (int)K, (int)NC, items, g);
  else if (kb == 2) launch<128, 2>(mw, mv, mo, dp, (int)M, (int)K, (int)NC, items, g);
  else launch<128, 1>(mw, mv, mo, dp, (int)M, (int)K, (int)NC, items, g);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pwa8_forward", &pwa8_forward, "EXPERIMENTAL sm100 PWA e4m3 contraction of all heads in one launch: o[h] = dq w8[h] . v8[h]", py::arg("w8"), py::arg("v8"), py::arg("o"), py::arg("dq"),
        py::arg("grid") = 0, py::arg("cfg") = 0);
}
