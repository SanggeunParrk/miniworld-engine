// tg_dw.cu -- the generic Transition backward's weight gradients on the B200 (sm_100a) with a deterministic split over the rows
// (-DF32=<0 bf16 | 1 fp32 on the TF32 tensor cores> -DNT=<64 | 128 | 256>), round 3 part 5 (2026-11-17).
//   dWs  [D][H]  = dy^T h      (D / 128) x (H / NT) output tiles
//   dWab [2H][D] = dab^T xn    (2 H / 128) x (D / NT) output tiles (dWa rows, then dWb rows)
// One launch for both. The contraction runs over the M rows, read MN-major from the row-major activations (no transposed copies; tcgen05 reads MN-major tf32 and bf16 operands). CTA (tile t, split s) of the
// grid T x S takes the 64-row K-blocks [s kb / S, (s + 1) kb / S) through a TMA ring into one 128 x NT fp32 accumulator in tensor memory
// (TMA zero-fills the rows past M). S = 1: the tile goes straight to dWs / dWab; S > 1: to the slab s of the workspace WS [S][D H + 2 H D],
// which tg_wsum (tg_rows.cu, the next launch) adds in slab order -- deterministic. This replaces the cuBLAS dW GEMMs (fp32 output,
// K = M: at 16 384+ rows cuBLAS took 18 + 36 us at D 128 n 2 fp32 and 40 + 41 us at D 256 with a split-K reduce launch of its own).
// The operands (dy, h, dab, xn) are complete when the gate GEMM this launch follows ends (griddepcontrol.wait before any load).
// SPDX-License-Identifier: Apache-2.0
#include "../widths/tsx_store.cuh"
using namespace s100;

#ifndef F32
#define F32 0
#endif
#ifndef NT
#define NT 128
#endif
#ifndef SPL
#define SPL 0                                                   // fp32: dWs = (dy_hi + dy_lo)^T h, dy split into two TF32 values
#endif
static_assert(!SPL || F32, "the dy split is an fp32 build");
constexpr int ESZ = F32 ? 4 : 2, EPB = 128 / ESZ;             // element bytes, MN elements per 128-B block row
constexpr int KI = F32 ? 8 : 16;                              // K rows per MMA instruction
constexpr int NAB = 128 / EPB, NBB = NT / EPB;                // 8 KB MN blocks of one 64-row K-block: A (128 rows of out), B (NT)
constexpr int ABT = NAB * 8192, BBT = NBB * 8192, STB = ABT + BBT + (SPL ? ABT : 0);
constexpr int STG = (200 * 1024 / STB) > 8 ? 8 : (200 * 1024 / STB);
constexpr int O_A = 0, O_B = STG * ABT, O_L = O_B + STG * BBT, O_BAR = O_L + (SPL ? STG * ABT : 0), SMEM_BYTES = O_BAR + 256;
static_assert(STG >= 2 && SMEM_BYTES <= 232448, "shared memory budget");
static_assert(NT == 64 || NT == 128 || NT == 256, "tile width");
#if F32
constexpr uint32_t IDESC = (1u << 4) | (2u << 7) | (2u << 10) | (1u << 15) | (1u << 16) | ((uint32_t)(NT >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
}
#else
constexpr uint32_t IDESC = idesc_bf16(128, NT, 1, 1);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#endif

struct Bars {
  uint64_t full[STG], empty[STG], acc_full;
  uint32_t tmem;
};

// MN-major operand descriptor: bf16 128-B swizzle (layout type 2); tf32 128-B swizzle with 32-B atoms (layout type 1, the TMA map's
// CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B: MN-major kind::tf32 reads this layout). LBO = 8 KB between MN blocks, SBO = 1 KB (bf16) between K groups.
DEVI uint64_t desc_op(uint32_t saddr) {
  const uint64_t d = desc_mn128(saddr, 8192);
  // BASE32B (Swizzle<2,5,2>): the pattern repeats every 4 rows of 128 B, so the K groups are 4 rows (SBO 512 B)
  return F32 ? ((d & ~((uint64_t)7 << 61) & ~((uint64_t)0x3FFF << 32)) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)1 << 61)) : d;
}

// the barriers of one CTA (thread 0)
DEVI void tg_dw_init(Bars& B) {
  for (int i = 0; i < STG; ++i) { mbar_init(&B.full[i], 1); mbar_init(&B.empty[i], 1); }
  mbar_init(&B.acc_full, 1);
  fence_barrier_init();
  fence_proxy_async();
}

// (round 3, training) one CTA's roles for grid index `bid` (the kernel: blockIdx.x), all threads; barriers fresh (tg_dw_init) and NT
// tensor-memory columns at `tmem`. tg_bwd_small.cu (-DTG_DW_BODY, no kernel) runs it as a phase of one launch, several tiles per CTA on
// the same barriers: it0 ring iterations and tc0 tiles before this one (the kernel: 0, 0). The epilogue is warps 4-7.
DEVI void tg_dw_roles(const CUtensorMap* pdy, const CUtensorMap* ph, const CUtensorMap* pdab, const CUtensorMap* pxn, const CUtensorMap* pdylo,
                      float* __restrict__ ws, float* __restrict__ o1, float* __restrict__ o2, int m, int d, int hid, int nsplit,
                      const float* __restrict__ part, float* __restrict__ dgb, int nrows, Bars& B, uint32_t su, uint32_t tmem, int bid,
                      int it0 = 0, int tc0 = 0) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = bid / nsplit, s = bid % nsplit;
  const int t1 = (d / 128) * (hid / NT);
  const bool p1 = tile < t1;
  const int t = p1 ? tile : tile - t1;
  const int nN = p1 ? hid / NT : d / NT;
  const int m0 = (t / nN) * 128, n0 = (t % nN) * NT;
  const CUtensorMap* ma = p1 ? pdy : pdab;
  const CUtensorMap* mb = p1 ? ph : pxn;
  const int ldo = p1 ? hid : d;
  const size_t ntot = (size_t)3 * d * hid;
  float* outp = nsplit > 1 ? ws + (size_t)s * ntot + (p1 ? 0 : (size_t)d * hid) : (p1 ? o1 : o2);
  const int nkb = (m + 63) / 64;
  const int kb0 = (int)((long)s * nkb / nsplit), kb1 = (int)((long)(s + 1) * nkb / nsplit), n = kb1 - kb0;

  if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < n; ++i) {
        const int st = (it0 + i) % STG, kb = kb0 + i;
        if (it0 + i >= STG) mbar_wait(&B.empty[st], (((it0 + i) / STG) - 1) & 1);
        mbar_expect_tx(&B.full[st], (SPL && p1) ? STB : STB - (SPL ? ABT : 0));
#if SPL
        if (p1) {
#pragma unroll
          for (int j = 0; j < NAB; ++j) tma_load_2d(su + O_L + st * ABT + j * 8192, pdylo, &B.full[st], m0 + j * EPB, kb * 64);
        }
#endif
#pragma unroll
        for (int j = 0; j < NAB; ++j) tma_load_2d(su + O_A + st * ABT + j * 8192, ma, &B.full[st], m0 + j * EPB, kb * 64);
#pragma unroll
        for (int j = 0; j < NBB; ++j) tma_load_2d(su + O_B + st * BBT + j * 8192, mb, &B.full[st], n0 + j * EPB, kb * 64);
      }
    }
  } else if (warp == 1) {
    for (int i = 0; i < n; ++i) {
      const int st = (it0 + i) % STG;
      mbar_wait(&B.full[st], ((it0 + i) / STG) & 1);
      tc_fence_after();
      const uint64_t da = desc_op(su + O_A + st * ABT), db = desc_op(su + O_B + st * BBT);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 64 / KI; ++ks)
          mma(tmem, da + (uint64_t)((ks * KI * 128) >> 4), db + (uint64_t)((ks * KI * 128) >> 4), (i > 0 || ks > 0) ? 1u : 0u);
#if SPL
        if (p1) {
          const uint64_t dl = desc_op(su + O_L + st * ABT);
#pragma unroll
          for (int ks = 0; ks < 64 / KI; ++ks) mma(tmem, dl + (uint64_t)((ks * KI * 128) >> 4), db + (uint64_t)((ks * KI * 128) >> 4), 1u);
        }
#endif
        tc_commit(&B.empty[st]);
        if (i == n - 1) tc_commit(&B.acc_full);
      }
      __syncwarp();
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------------ the fp32 tile: one row per thread
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    float* ro = outp + (size_t)m0 * ldo + n0;
    if (n > 0) {
      mbar_wait(&B.acc_full, tc0 & 1);
      tc_fence_after();
    }
#pragma unroll 1
    for (int c0 = 0; c0 < NT; c0 += 32) {
      uint32_t v[32];
      if (n > 0) {
        tmem_ld32(trow + c0, v);
        tmem_wait_ld();
      } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = 0u;
      }
      store_t(ro + c0 + store_t_off<32>(r, lane, ldo, 4), ldo, v, lane);
    }
    tc_fence_before();
    if (nrows > 0 && bid < (2 * d + 127) / 128) {   // S = 1: dgamma | dbeta = the LayerNorm backward's partial rows, in order
      const int c = bid * 128 + r;
      if (c < 2 * d) {
        float a = 0.f;
        for (int i = 0; i < nrows; ++i) a += part[(size_t)i * 2 * d + c];
        dgb[c] = a;
      }
    }
  }
}

#ifndef TG_DW_BODY
// maps (box EPB inner x 64 rows, MN-major operands): dy [M][D], h [M][H], dab [M][2 H], xn [M][D] (activation dtype); dylo: SPL, the
// TF32 residual dy - dy_hi of the fp32 dy whose TF32 value is mdy (other builds: any map, unread).
// o1 = dWs [D][H], o2 = dWab [2H][D] fp32 (S = 1), else the workspace ws [S][D H + 2 H D] (tg_wsum writes o1 / o2).
// nrows > 0 (S = 1, launched after the LayerNorm backward): the first 2 D / 128 CTAs also sum its [nrows][2 D] dgamma | dbeta partials into dgb (the tg_colsum launch folded in).
extern "C" __global__ void __launch_bounds__(256, 1)
tg_dw_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mh, const __grid_constant__ CUtensorMap mdab,
            const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mdylo, float* __restrict__ ws, float* __restrict__ o1, float* __restrict__ o2, int m, int d,
            int hid, int nsplit, const float* __restrict__ part, float* __restrict__ dgb, int nrows) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    tg_dw_init(B);
    const bool p1 = (int)blockIdx.x / nsplit < (d / 128) * (hid / NT);
    prefetch_map(p1 ? &mdy : &mdab); prefetch_map(p1 ? &mh : &mxn);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), NT); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_wait();
  pdl_launch();
  tg_dw_roles(&mdy, &mh, &mdab, &mxn, &mdylo, ws, o1, o2, m, d, hid, nsplit, part, dgb, nrows, B, su, tmem, (int)blockIdx.x);
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, NT); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
#endif
