// ctr.cu -- the TriMul triangle contraction on B200 (tcgen05) for small L (inference), in ONE programmatic-dependent launch:
//   outgoing  t[c][i][j] = sum_k a[c][i][k] b[c][j][k]     incoming  t[c][i][j] = sum_k a[c][k][i] b[c][k][j]
// planes [P, nb, L, L] bf16 (k1w's channel-major output), t [T, nb, L, L] bf16; d channels per direction (the contracted width);
// dir 0 bidirectional (T = 2d: t channel c < d outgoing from planes c, 2d + c; c >= d incoming from planes c, 2d + c), 1 outgoing,
// 2 incoming (T = d: planes c, d + c).
//
// Why (2026-11-17, round 3 fifth pass): at L128 / L256 the cuBLAS products are 2.9-4.6 us launches that neither start before the
// front has ended nor let the output kernel start before they end, and the bidirectional module issues two of them. Here one launch
// serves both directions, triggers its dependents at entry (k3g / wide_ln_stats launch as its CTAs retire) and is itself launched as a
// programmatic dependent of k1w (barrier init, TMEM allocation and tensor-map prefetch overlap k1w's tail; griddepcontrol.wait
// before the first plane is read). Products are bitwise cuBLAS's (fp32 accumulation of the same bf16 operands over K in order,
// one bf16 rounding).
//
// Item = (output channel c, sample s, 128-row tile mt, 128-column tile nt): an M128 N128 K=L product in K stages of 128 (stage =
// A 32 KB + B 32 KB, a ring of NST stages). (N = 64 items at few products -- twice the CTAs -- measured no different and were
// dropped.) Operands from one 2-D map of the
// planes ([P nb L rows][L cols], SW128): outgoing both K-major (rows i / j, two 64-wide K boxes), incoming both MN-major (rows k,
// 64-wide MN boxes, LBO 16 KB). 384 threads: warp 0 TMA, warp 1 MMA, warp 2 TMEM (2 x 128 columns), warps 4-11 epilogue (thread =
// row, two warps per TMEM lane quarter, one 64-column half each: TMEM -> bf16 -> SW128 staging [32 rows][64 cols] -> TMA store).
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

// S90 part 5: the item is M128 x BN (128, or 256 when L % 256 == 0: A / B operand traffic per output element 25 % lower, the kernel is L2-bandwidth
// bound at L >= 384), a stage's boxes (2 + BN / 64, 16 KB each) are issued by that many lanes of warp 0 (one thread issues ~one TMA per 600 clk).
namespace ctr {

constexpr int NTHR = 384, TOK = 128, BOX = TOK * 64 * 2;     // [128][64] bf16 = 16 KB
constexpr int OSTW = 32 * 64 * 2;                             // per epilogue warp: one [32 rows][64 cols] box = 4 KB

// NSV: stages of the ring (0: the default, 3 at BN 128 / 2 at BN 256). (n2 2026-11-18) L128 (K = 128: one stage an item) with more items than SMs
// (D384 / D512, bidirectional D256: 384-512 items for 148 SMs; measured: D512 -2 %, D384 / bidirectional D256 -1 %; D256 with 256 items: +2 %, not
// used): NSV 1 -- 99 KB of shared memory, two CTAs an SM (256 TMEM columns each), two items resident at once instead of one.
template <int BN, int NSV = 0>
struct Cfg {
  static constexpr int NB = 2 + BN / 64;                      // boxes of a stage: A 2, B 2 K halves x BN / 128 row boxes (K-major) or BN / 64 MN boxes
  static constexpr int STG = NB * BOX;
  static constexpr int NST = NSV ? NSV : (BN == 128 ? 3 : 2);
  static constexpr int O_ST = 0, O_OUT = NST * STG, O_BAR = O_OUT + 8 * OSTW, SMEM = O_BAR + 256 + 1024;
  static_assert(SMEM <= 232448, "smem");
};

template <int NST>
struct Bars {
  uint64_t full[NST], empty[NST], acc_full[2], acc_empty[2];
  uint32_t tmem;
};

DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }

struct Item {
  int ra, rb, i0, j0, trow;
  bool inc;
};
// item -> (channel c, sample s, row tile mt of 128, column tile nt of bn)
DEV Item item_of(int it, int L, int nb, int d, int dir, int bn) {
  const int MT = L / TOK, NT = L / bn;
  const int nt = it % NT, r1 = it / NT, mt = r1 % MT, cs = r1 / MT, s = cs % nb, c = cs / nb;
  Item m;
  m.inc = dir == 2 || (dir == 0 && c >= d);
  const int pa = c, pb = c + (dir == 0 ? 2 * d : d);
  m.ra = (pa * nb + s) * L; m.rb = (pb * nb + s) * L;
  m.i0 = mt * TOK; m.j0 = nt * bn;
  m.trow = (c * nb + s) * L + m.i0;
  return m;
}

template <int BN, int NSV = 0, int MINB = 1>
__global__ void __launch_bounds__(NTHR, MINB)
    ctr_kernel(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mt, int L, int nb, int d,
               int dir, int items, int trig) {
  using G_ = Cfg<BN, NSV>;
  constexpr int bn = BN, NST = G_::NST, STG = G_::STG, NB = G_::NB, O_ST = G_::O_ST, O_OUT = G_::O_OUT, O_BAR = G_::O_BAR;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars<NST>& B = *reinterpret_cast<Bars<NST>*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int KS = L / TOK;                 // K stages per item
  constexpr int EG = 2;                   // epilogue warp groups (each handles BN / 128 column halves of 64)
  constexpr int BB = bn * 64 * 2;         // one B K-half box set: [bn rows][64 k]
  // trig: the next kernel (k3g) may launch its CTAs as these retire -- only for a dependent that waits for this grid before it reads t
  // trig 2 (pass 15): only after this CTA's griddepcontrol.wait (below)
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
      pdl_wait();   // the planes are the previous kernel's (k1w) output; nothing global is touched before this
      if (trig == 2 && lane == 0) pdl_trigger();
      int q = 0;
      for (int it = cta; it < items; it += G) {
        const Item m = item_of(it, L, nb, d, dir, bn);
        for (int ks = 0; ks < KS; ++ks, ++q) {
          const int slot = q % NST;
          if (q >= NST) mbar_wait(&B.empty[slot], ((q / NST) - 1) & 1);
          uint8_t* a = sm + O_ST + slot * STG;
          uint8_t* b = a + 2 * BOX;
          const int k0 = ks * TOK;
          if (lane == 0) mbar_expect_tx(&B.full[slot], NB * BOX);
          __syncwarp((1u << NB) - 1u);
          if (!m.inc) {   // A [128 rows i][64 k] x 2 K halves; B [bn rows j][64 k] x 2 K halves, as BN / 128 boxes of 128 rows each
            if (lane < 2) tma_load_2d(a + lane * BOX, &mp, &B.full[slot], k0 + 64 * lane, m.ra + m.i0, EVICT_NORMAL);
            else { const int h = (lane - 2) / (BN / 128), r = (lane - 2) % (BN / 128);
                   tma_load_2d(b + h * BB + r * BOX, &mp, &B.full[slot], k0 + 64 * h, m.rb + m.j0 + 128 * r, EVICT_NORMAL); }
          } else {        // A [128 rows k][64 i] x 2 MN halves; B [128 rows k][64 j] x bn / 64
            if (lane < 2) tma_load_2d(a + lane * BOX, &mp, &B.full[slot], m.i0 + 64 * lane, m.ra + k0, EVICT_NORMAL);
            else tma_load_2d(b + (lane - 2) * BOX, &mp, &B.full[slot], m.j0 + 64 * (lane - 2), m.rb + k0, EVICT_NORMAL);
          }
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      const uint32_t id_o = idesc_bf16_mj(128, bn, 0, 0), id_i = idesc_bf16_mj(128, bn, 1, 1);
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
          const uint32_t a = su + O_ST + slot * STG, b = a + 2 * BOX;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            if (!m.inc)
              umma_ss(acc, desc_k_sw128(a + (kk >> 2) * BOX + (kk & 3) * 32), desc_k_sw128(b + (kk >> 2) * BB + (kk & 3) * 32), id_o,
                      (ks | kk) != 0);
            else
              umma_ss(acc, desc_mn_sw128(a + kk * 2048, BOX), desc_mn_sw128(b + kk * 2048, BOX), id_i, (ks | kk) != 0);
          }
          umma_commit(&B.empty[slot]);
        }
        umma_commit(&B.acc_full[bb]);
      }
    }
  } else if (warp >= 4 && warp < 4 + 4 * EG) {
    const int qd = warp & 3, h = (warp - 4) >> 2;   // lane quarter, 64-column half (h, h + 2 at BN = 256)
    const uint32_t stg = su + O_OUT + (warp - 4) * OSTW;
    int n = 0;
    for (int it = cta; it < items; it += G, ++n) {
      const Item m = item_of(it, L, nb, d, dir, bn);
      const int bb = n & 1;
      mbar_wait(&B.acc_full[bb], (n >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int hh = 0; hh < BN / 128; ++hh) {
        const int hc = h + 2 * hh;
        const uint32_t ta = tmem + ((uint32_t)(qd * 32) << 16) + hc * 64;
        if (lane == 0) bulk_wait_read<0>();   // the previous store has left the staging box
        __syncwarp();
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {      // 32 columns per load: 16-byte chunks 4 cc .. 4 cc + 3 of the box
          uint32_t v[32];
          tmem_ld32(ta + bb * bn + cc * 32, v);
          tmem_wait_ld();
          if (cc == 1 && hh == BN / 128 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[bb]); }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            auto pk = [&](int e) { return pack_bf16(__uint_as_float(v[8 * k + 2 * e]), __uint_as_float(v[8 * k + 2 * e + 1])); };
            sts128(stg + sw128(lane, cc * 4 + k), pk(0), pk(1), pk(2), pk(3));
          }
        }

        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mt, sm + O_OUT + (warp - 4) * OSTW, m.j0 + 64 * hc, m.trow + qd * 32);
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

}  // namespace ctr

// planes [P, (nb,) L, L] bf16 contiguous -> t [T, (nb,) L, L] bf16 contiguous (L % 128 == 0); d = contracted channels per direction;
// dir 0 bidirectional (P = 4 d, T = 2 d), 1 outgoing, 2 incoming (P = 2 d, T = d); pdl bit 0: launched as a programmatic dependent,
// bit 1: triggers its own dependents at entry (only when the next kernel waits before reading t); bit 2: triggers them after its wait.
void ctr_forward(torch::Tensor planes, torch::Tensor t, int64_t d, int64_t dir, int64_t grid, int64_t pdl) {
  using namespace ctr;
  const int64_t L = planes.size(-1);
  const int64_t P = planes.size(0), T = t.size(0);
  TORCH_CHECK(planes.is_contiguous() && t.is_contiguous() && planes.scalar_type() == torch::kBFloat16 && t.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(L % TOK == 0 && planes.size(-2) == L, "ctr_forward: square planes, L % 128 == 0");
  TORCH_CHECK(dir >= 0 && dir <= 2 && P == (dir == 0 ? 4 : 2) * d && T == (dir == 0 ? 2 : 1) * d, "ctr_forward: plane / channel counts");
  const int64_t nb = planes.numel() / (P * L * L);
  TORCH_CHECK(t.numel() == T * nb * L * L, "ctr_forward: t shape");
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  const uint64_t rows = (uint64_t)(P * nb * L);
  auto mp = tmap::make(planes.data_ptr(), bf, {(uint64_t)L, rows}, {(uint64_t)L * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mt = tmap::make(t.data_ptr(), bf, {(uint64_t)L, (uint64_t)(T * nb * L)}, {(uint64_t)L * 2}, {64, 32}, CU_TENSOR_MAP_SWIZZLE_128B);
  const char* bn_env = std::getenv("TRIMUL_CTR_BN");   // 128: always BN = 128 (A/B)
  const int bn = (L % 256 == 0 && !(bn_env != nullptr && (bn_env[0] == '1' || bn_env[0] == '0'))) ? 256 : 128;
  const int items = (int)(T * nb * (L / TOK) * (L / bn));
  const char* two_env = std::getenv("TRIMUL_CTR_TWO");   // 0: never two CTAs an SM (A/B)
  const bool two = bn == 128 && L == TOK && items > 2 * num_sms() && grid <= 0 && !(two_env != nullptr && two_env[0] == '0');
  int g = grid > 0 ? (int)grid : (two ? 2 * num_sms() : num_sms());
  g = std::min(g, items);
  const int trig = ((pdl >> 1) & 1) ? 1 : (int)((pdl >> 2) & 1) * 2;
  if (two) {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(ctr_kernel<128, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<128, 1>::SMEM); attr = true; }
    tmap::launch_pdl(ctr_kernel<128, 1, 2>, dim3(g), dim3(NTHR), Cfg<128, 1>::SMEM, at::cuda::getCurrentCUDAStream(), (pdl & 1) != 0, mp, mt, (int)L, (int)nb,
                     (int)d, (int)dir, items, trig);
  } else if (bn == 256) {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(ctr_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::SMEM); attr = true; }
    tmap::launch_pdl(ctr_kernel<256>, dim3(g), dim3(NTHR), Cfg<256>::SMEM, at::cuda::getCurrentCUDAStream(), (pdl & 1) != 0, mp, mt, (int)L, (int)nb,
                     (int)d, (int)dir, items, trig);
  } else {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(ctr_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<128>::SMEM); attr = true; }
    tmap::launch_pdl(ctr_kernel<128>, dim3(g), dim3(NTHR), Cfg<128>::SMEM, at::cuda::getCurrentCUDAStream(), (pdl & 1) != 0, mp, mt, (int)L, (int)nb,
                     (int)d, (int)dir, items, trig);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
