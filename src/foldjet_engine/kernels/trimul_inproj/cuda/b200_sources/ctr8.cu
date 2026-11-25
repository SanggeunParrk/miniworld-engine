// ctr8.cu -- the e4m3 twin of ctr.cu: the TriMul triangle contraction on B200 (tcgen05 kind::f8f6f4) on e4m3 planes, FP8 route (opt-in).
//   outgoing  t[c][i][j] = csc * sum_k a[c][i][k] b[c][j][k]     incoming  t[c][i][j] = csc * sum_k a[c][k][i] b[c][k][j]
// planes [P, nb, L, L] uint8 (e4m3 bytes; k1w's F8 epilogue output = the plane value times its power-of-two qinv), t [T, nb, L, L] bf16,
// csc [2] fp32 device scalars: csc[0] = sa sb of the outgoing (or the only) direction, csc[1] the incoming direction of the bidirectional
// block (powers of two: the product is exact, applied to the fp32 accumulator in the epilogue -- K3 sees t at its true scale).
// Same item / pipeline structure as ctr.cu; the e4m3 stage is KB boxes of 128 k per operand (a [128 rows][128 B] SW128 box is 16 KB, as
// ctr.cu's two bf16 boxes), 32 k per MMA, so a stage of KB = 2 carries 256 k.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace ctr8 {

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
  int ra, rb, i0, j0, trow;
  bool inc;
};
DEV Item item_of(int it, int L, int nb, int d, int dir, int bn) {
  const int MT = (L + TOK - 1) / TOK, NT = (L + bn - 1) / bn;
  const int nt = it % NT, r1 = it / NT, mt = r1 % MT, cs = r1 / MT, s = cs % nb, c = cs / nb;
  Item m;
  m.inc = dir == 2 || (dir == 0 && c >= d);
  const int pa = c, pb = c + (dir == 0 ? 2 * d : d);
  m.ra = pa * nb + s; m.rb = pb * nb + s;   // plane indices (the maps are 3-D: [plane][row][col], out-of-range rows / columns of a tile read as zero and are not stored)
  m.i0 = mt * TOK; m.j0 = nt * bn;
  m.trow = c * nb + s;
  return m;
}

template <int BN, int KB, int MINB = 1>
__global__ void __launch_bounds__(NTHR, MINB)
    ctr8_kernel(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mt, const float* __restrict__ csc, int L, int nb,
                int d, int dir, int items, int trig) {
  using G_ = Cfg<BN, KB>;
  constexpr int bn = BN, NST = G_::NST, STG = G_::STG, NB = G_::NB, O_ST = G_::O_ST, O_OUT = G_::O_OUT, O_BAR = G_::O_BAR;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars<NST>& B = *reinterpret_cast<Bars<NST>*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int KS = (L + TOK * KB - 1) / (TOK * KB);          // K stages per item (the tail reads zeros)
  constexpr int EG = 2;
  constexpr int BB = bn * 128;            // one B K-box set: [bn rows][128 B]
  if (trig == 1) pdl_trigger();
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4 * EG); }
    fence_mbar_init();
    prefetch_tmap(&mp); prefetch_tmap(&mt);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 2 * BN); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane < NB) {   // the producer: NB lanes each issue one box of a stage
      pdl_wait();
      if (trig == 2 && lane == 0) pdl_trigger();
      int q = 0;
      for (int it = cta; it < items; it += G) {
        const Item m = item_of(it, L, nb, d, dir, bn);
        for (int ks = 0; ks < KS; ++ks, ++q) {
          const int slot = q % NST;
          if (q >= NST) mbar_wait(&B.empty[slot], ((q / NST) - 1) & 1);
          uint8_t* a = sm + O_ST + slot * STG;
          uint8_t* b = a + KB * BOX;
          const int k0 = ks * TOK * KB;
          if (lane == 0) mbar_expect_tx(&B.full[slot], NB * BOX);
          __syncwarp((1u << NB) - 1u);
          if (!m.inc) {   // A [128 rows i][128 k] x KB; B [bn rows j][128 k] x KB, BN / 128 boxes of 128 rows each
            if (lane < KB) tma_load_3d(a + lane * BOX, &mp, &B.full[slot], k0 + 128 * lane, m.i0, m.ra, EVICT_NORMAL);
            else { const int h = (lane - KB) / (BN / 128), r = (lane - KB) % (BN / 128);
                   tma_load_3d(b + h * BB + r * BOX, &mp, &B.full[slot], k0 + 128 * h, m.j0 + 128 * r, m.rb, EVICT_NORMAL); }
          } else {        // A [128 rows k][128 i] x KB; B [128 rows k][128 j] x (KB bn / 128)
            if (lane < KB) tma_load_3d(a + lane * BOX, &mp, &B.full[slot], m.i0, k0 + 128 * lane, m.ra, EVICT_NORMAL);
            else { const int h = (lane - KB) / (BN / 128), r = (lane - KB) % (BN / 128);
                   tma_load_3d(b + h * BB + r * BOX, &mp, &B.full[slot], m.j0 + 128 * r, k0 + 128 * h, m.rb, EVICT_NORMAL); }
          }
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      const uint32_t id_o = idesc_e4m3(128, bn, 0, 0), id_i = idesc_e4m3(128, bn, 1, 1);
      int q = 0, n = 0;
      for (int it = cta; it < items; it += G, ++n) {
        const Item m = item_of(it, L, nb, d, dir, bn);
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
            for (int kk = 0; kk < 4; ++kk) {
              if (!m.inc)
                umma_f8(acc, desc_k_sw128(a + kb * BOX + kk * 32), desc_k_sw128(b + kb * BB + kk * 32), id_o, (ks | kb | kk) != 0);
              else   // MN-major: 32 k rows (4 groups of 8) per MMA, 128 MN elements per 128 B row, the next 128 of B a box (BOX) away
                umma_f8(acc, desc_mn_sw128(a + kb * BOX + kk * 4096, BOX), desc_mn_sw128(b + kb * BB + kk * 4096, BOX), id_i, (ks | kb | kk) != 0);
            }
          }
          umma_commit(&B.empty[slot]);
        }
        umma_commit(&B.acc_full[bb]);
      }
    }
  } else if (warp >= 4 && warp < 4 + 4 * EG) {
    const int qd = warp & 3, h = (warp - 4) >> 2;
    const uint32_t stg = su + O_OUT + (warp - 4) * OSTW;
    int n = 0;
    for (int it = cta; it < items; it += G, ++n) {
      const Item m = item_of(it, L, nb, d, dir, bn);
      const float cs = __ldg(csc + (dir == 0 && m.inc ? 1 : 0));
      const int bb = n & 1;
      mbar_wait(&B.acc_full[bb], (n >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int hh = 0; hh < BN / 128; ++hh) {
        const int hc = h + 2 * hh;
        const uint32_t ta = tmem + ((uint32_t)(qd * 32) << 16) + hc * 64;
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
            if (!m.inc) {
              auto pk = [&](int e) { return pack_bf16(__uint_as_float(v[8 * k + 2 * e]) * cs, __uint_as_float(v[8 * k + 2 * e + 1]) * cs); };
              sts128(stg + sw128(lane, cc * 4 + k), pk(0), pk(1), pk(2), pk(3));
            } else {   // incoming: the planes come in the front's shuffled token order (position p holds token (p >> 1) + 8 (p & 1) of its 16-group), so row and column p of t
                       // are token (p >> 1) + 8 (p & 1): undo it here (row: the staging row; column: natural column n = accumulator column 2 n (n < 8) / 2 (n - 8) + 1)
              const int g = k >> 1, od = k & 1, rn = (lane & ~15) | ((lane & 15) >> 1) | ((lane & 1) << 3);
              auto pk = [&](int e) { return pack_bf16(__uint_as_float(v[16 * g + 4 * e + od]) * cs, __uint_as_float(v[16 * g + 4 * e + 2 + od]) * cs); };
              sts128(stg + sw128(rn, cc * 4 + k), pk(0), pk(1), pk(2), pk(3));
            }
          }
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          tma_store_3d(&mt, sm + O_OUT + (warp - 4) * OSTW, m.j0 + 64 * hc, m.i0 + qd * 32, m.trow);
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

int num_sms() {
  static int n = 0;
  if (!n) { int dv; cudaGetDevice(&dv); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dv); }
  return n;
}

template <int BN, int KB>
void launch(const CUtensorMap& mp, const CUtensorMap& mt, const float* csc, int L, int nb, int d, int dir, int items, int g, bool pdl, int trig) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(ctr8_kernel<BN, KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<BN, KB>::SMEM); attr = true; }
  tmap::launch_pdl(ctr8_kernel<BN, KB>, dim3(g), dim3(NTHR), Cfg<BN, KB>::SMEM, at::cuda::getCurrentCUDAStream(), pdl, mp, mt, csc, L, nb, d, dir, items, trig);
}

}  // namespace ctr8

// planes [P, (nb,) L, L] uint8 contiguous -> t [T, (nb,) L, L] bf16 contiguous (L % 128 == 0); csc [2] fp32 (device); d, dir, pdl as ctr_forward.
// cfg: 0 = default (BN 256 when L % 256 == 0 else 128; KB 1), else 100 * KB + BN / 128 forces a variant (A/B).
void ctr8_forward(torch::Tensor planes, torch::Tensor t, torch::Tensor csc, int64_t d, int64_t dir, int64_t grid, int64_t pdl, int64_t cfg) {
  using namespace ctr8;
  const int64_t L = planes.size(-1);
  const int64_t P = planes.size(0), T = t.size(0);
  TORCH_CHECK(planes.is_contiguous() && t.is_contiguous() && planes.scalar_type() == torch::kUInt8 && t.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(csc.scalar_type() == torch::kFloat32 && csc.numel() >= 2 && csc.is_cuda());
  TORCH_CHECK(L % 16 == 0 && planes.size(-2) == L, "ctr8_forward: square planes, L % 16 == 0");
  TORCH_CHECK(dir >= 0 && dir <= 2 && P == (dir == 0 ? 4 : 2) * d && T == (dir == 0 ? 2 : 1) * d, "ctr8_forward: plane / channel counts");
  const int64_t nb = planes.numel() / (P * L * L);
  TORCH_CHECK(t.numel() == T * nb * L * L, "ctr8_forward: t shape");
  auto mp = tmap::make(planes.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, {(uint64_t)L, (uint64_t)L, (uint64_t)(P * nb)}, {(uint64_t)L, (uint64_t)L * L}, {128, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mt = tmap::make(t.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {(uint64_t)L, (uint64_t)L, (uint64_t)(T * nb)}, {(uint64_t)L * 2, (uint64_t)L * L * 2}, {64, 32, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  const int64_t up128 = (L + 127) / 128 * 128, up256 = (L + 255) / 256 * 256;
  int bn = (up256 == up128) ? 256 : 128, kb = 1;   // BN 256 unless it wastes a column tile; KB 2 (256 k a stage) measured slower than KB 1 at every L (B200 2026-11-19)
  if (cfg > 0) { kb = (int)(cfg / 100); bn = 128 * (int)(cfg % 100); }

  const int items = (int)(T * nb * ((L + TOK - 1) / TOK) * ((L + bn - 1) / bn));
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, items);
  const int trig = ((pdl >> 1) & 1) ? 1 : (int)((pdl >> 2) & 1) * 2;
  const bool use_pdl = (pdl & 1) != 0;
  const float* cp = csc.data_ptr<float>();
  if (bn == 256 && kb == 2) launch<256, 2>(mp, mt, cp, (int)L, (int)nb, (int)d, (int)dir, items, g, use_pdl, trig);
  else if (bn == 256) launch<256, 1>(mp, mt, cp, (int)L, (int)nb, (int)d, (int)dir, items, g, use_pdl, trig);
  else if (kb == 2) launch<128, 2>(mp, mt, cp, (int)L, (int)nb, (int)d, (int)dir, items, g, use_pdl, trig);
  else launch<128, 1>(mp, mt, cp, (int)L, (int)nb, (int)d, (int)dir, items, g, use_pdl, trig);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
