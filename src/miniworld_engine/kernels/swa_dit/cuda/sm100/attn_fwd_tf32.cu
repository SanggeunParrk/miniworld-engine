// attn_fwd_tf32.cu — the SWA atom block's window attention forward for the fp32 path on sm_100a, TF32 tensor cores (tcgen05.mma
// kind::tf32, fp32 accumulation), fp32 softmax. Same masks and conventions as attn_fwd3.cu / attn_fwd1h.cu: keys j with |i - j| <= 64
// and j < seqused[n]; O = softmax(q K^T scale) V (fp32, padding query rows 0); LSE = m scale + ln l (natural log of the sum of the
// scaled scores' exponentials, 0 on rows without a valid key; a padding query row scores 0 against its window's valid keys, as the
// bf16 kernels and the Triton kernel do).
// Q / K / V are the qkvg_fwd_tf32 outputs: head-major [N, H, S, 32] fp32 already rounded to TF32, so the S = Q K^T products are exact
// TF32 products; P is rounded to TF32 (cvt.rna) when it is written back, and the row sum l adds the rounded P the PV MMA reads.
//
// Items = (sample, 128-query tile i0, head), one head per item (the attn_fwd1h layout): fp32 operands are twice the bytes, so the
// stage of one head (q 16 KB | K 32 KB | V 32 KB over the window [i0 - 64, i0 + 192)) is what two heads were in bf16, and the item
// stage is double-buffered (the next item loads under this one's math). The two softmax warpgroups split the item's keys: warpgroup
// kb takes key block kb (S columns 128 kb .. + 127) for all 128 query rows (thread = row = TMEM lane), exchanges its row maximum and
// row sum with the other through shared memory, and every row's P is taken against the row maximum over both blocks (no rescaling).
//   * S = q K^T: four M128 N256 K8 SS MMAs (q, K K-major in the 128-B swizzle: one 128-B row = the 32 fp32 of a head row).
//   * P overwrites S in place (fp32 P: one TMEM column per element, the column it came from); dead 32-column chunks are zeroed.
//   * PV: per key block, sixteen M128 N32 K8 TS MMAs, P from TMEM (A), V the MN-major B operand -- keys as rows, d contiguous: the
//     128-B swizzle with 32-B atoms (TMA CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, UMMA layout type 1, SBO 512), the layout a kind::tf32
//     MN-major operand needs (augmented_attention/cuda/sm100/sm100.cuh; measured there: the plain 128-B swizzle multiplies to zeros).
//     A K step of 8 keys is 1 KB of V.
//   shared memory  2 item stages x 80 KB | row max / row sum exchange [2][2][128] fp32 (2 KB) | barriers  -> 166144 B
//   TMEM (512)     S / P at 0 (256 columns, single: an item's S overwrites the previous item's P only after its PV MMAs completed --
//                  the MMA warp waits on that commit unless MMA_INORDER=1 relies on in-order tcgen05.mma execution), O[b] at
//                  256 + 32 b (b = item parity: the epilogue of item li reads O while item li + 1's MMAs run)
//   threads        warp 0 TMA producer (lane 0), warp 2 TMEM allocator, warp 1 MMA issuer (whole warp waits, elect_one() issues),
//                  warps 4-11 softmax (warpgroup kb = key block kb); registers: one 32-column chunk (32 fp32) at a time
// The epilogue stores O (16 of the 32 columns per warpgroup, 64 contiguous bytes per thread) and LSE with plain global stores.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef MMA_INORDER
#define MMA_INORDER 0                      // 1: issue item li's S MMAs right after item li - 1's PV MMAs (tcgen05.mma executes in issue order)
#endif

// ------------------------------------------------------------------ kind::tf32 (local: sm100.cuh is the bf16 kernels' header)
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
// MN-major operand in the 128-B swizzle with 32-B atoms (layout type 1): SBO = 512 (4-row groups), LBO = the distance between
// 32-element MN atoms (one atom here: D = 32)
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void umma_ss_tf32(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void umma_ts_tf32(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
// 32 lanes x 32 bits, 32 consecutive columns from thread t's registers to lane (base lane + t)
DEVI void tmem_st32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
                  "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]),
                  "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]),
                  "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory");
}

constexpr int DH = 32, QM = 128, H = 4, HW = 64, KR = 256;
constexpr int TQ = QM * 128, TK = KR * 128;                                 // q tile 16 KB, K / V 256-row boxes 32 KB (fp32 rows: 128 B)
constexpr int IST = TQ + 2 * TK;                                            // q | K | V = 80 KB
constexpr int O_ST = 0, O_RX = 2 * IST, O_BAR = O_RX + 2 * 2 * 128 * 4;     // stages, then rmax[2][128] | rsum[2][128]
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((IST % 1024) == 0 && (TQ % 1024) == 0 && (TK % 1024) == 0, "1 KB alignment of the swizzled tiles");
constexpr uint32_t T_S = 0, T_O = 256;
constexpr uint32_t I_S = idesc_tf32(128, KR), I_PV = idesc_tf32(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;

struct Bars {
  uint64_t full[2], empty[2], s_full, p_full[2], o_full[2], t_free[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                        const int* __restrict__ SEQU, float* __restrict__ O, float* __restrict__ LSE, int S, int N, float scale) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  float* rmax = reinterpret_cast<float*>(sm + O_RX);                       // [2 blocks][128 rows]
  float* rsum = rmax + 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM;
  const int items = N * mt * H;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item_of = [&](int li, int& n, int& i0, int& h, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    h = wi % H; const int r = wi / H;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
  };

  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); mbar_init(&B.p_full[s], 4); mbar_init(&B.o_full[s], 1); mbar_init(&B.t_free[s], 8);
    }
    mbar_init(&B.s_full, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: q, K, V of the head
    if (lane == 0) {
      pdl_wait();                                                          // q / K / V come from the previous kernel
      for (int li = 0; li < my_items; ++li) {
        int n, i0, h, sq; item_of(li, n, i0, h, sq);
        const int s = li & 1;
        if (li >= 2) mbar_wait(&B.empty[s], ((li >> 1) - 1) & 1);
        const uint32_t st = su + O_ST + s * IST;
        mbar_expect_tx(&B.full[s], TQ + 2 * TK);
        const int row = (n * H + h) * S;
        tma_load_2d(st, &mq, &B.full[s], 0, row + i0);
        tma_load_2d(st + TQ, &mk, &B.full[s], 0, row + i0 - HW);           // keys [i0 - 64, i0 + 192): rows outside the head are masked
        tma_load_2d(st + TQ + TK, &mv, &B.full[s], 0, row + i0 - HW);      //  (or zero-filled)
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    for (int li = 0; li < my_items; ++li) {
      const int s = li & 1, ob = li & 1;
      const uint32_t st = su + O_ST + s * IST;
      mbar_wait(&B.full[s], (li >> 1) & 1);
      if (!MMA_INORDER && li >= 1) mbar_wait(&B.o_full[(li - 1) & 1], ((li - 1) >> 1) & 1);   // PV(li - 1) has read P out of S
      if (li >= 2) mbar_wait(&B.t_free[ob], ((li >> 1) - 1) & 1);          // O[ob] (item li - 2) has been read out
      tc_fence_after();
      if (elect_one()) {
        const uint64_t dq = desc_k128(st), dk = desc_k128(st + TQ);
#pragma unroll
        for (int ks = 0; ks < DH / 8; ++ks) umma_ss_tf32(tmem + T_S, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full);
      }
      __syncwarp();
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        mbar_wait(&B.p_full[kb], li & 1);
        tc_fence_after();
        if (elect_one()) {
          const uint64_t dv = desc_mn32b(st + TQ + TK + kb * 128 * 128, TK);   // keys 128 kb .. of the window, 8 keys (1 KB) per K step
#pragma unroll
          for (int ks = 0; ks < 16; ++ks)
            umma_ts_tf32(tmem + T_O + 32 * ob, tmem + T_S + 128 * kb + 8 * ks, dv + (uint64_t)((ks * 1024) >> 4), I_PV, (kb | ks) ? 1u : 0u);
          if (kb == 1) { tc_commit(&B.o_full[ob]); tc_commit(&B.empty[s]); }
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    pdl_wait();                                                            // LSE / O stores
    // ------------------------------------------------------------------------------------------------ softmax: warpgroup kb = key block kb
    const int kb = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const uint32_t trow = tmem + (lb << 16) + T_S + 128 * kb;              // this row's S / P columns of block kb
    const float SC = scale * LOG2E;
    uint32_t z[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) z[k] = 0u;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, sq; item_of(li, n, i0, h, sq);
      const int ob = li & 1;
      const int i = i0 + (int)r, j0 = i0 - HW;
      const bool qok = i < sq;
      // valid columns of this row over the 256 keys, then relative to block kb: [clo, chi]
      const int clo = max(i - HW - j0, -j0) - 128 * kb, chi = min(i + HW, sq - 1) - j0 - 128 * kb;
      mbar_wait(&B.s_full, li & 1);
      tc_fence_after();
      // the warp's rows need columns [wlo, whi] of this block: 32-column chunks outside are not read (their P is zeroed)
      const int wlo = __reduce_min_sync(~0u, chi >= clo ? clo : KR), whi = __reduce_max_sync(~0u, chi >= clo ? chi : -KR);
      auto live = [&](int c32) { return 32 * c32 + 31 >= wlo && 32 * c32 <= whi; };
      // chunks valid on all 32 columns for every row of the warp (no padding rows): no masks
      const int flo = __reduce_max_sync(~0u, clo), fhi = __reduce_min_sync(~0u, chi);
      const bool allq = __all_sync(~0u, qok);
      auto full = [&](int c32) { return allq && 32 * c32 >= flo && 32 * c32 + 31 <= fhi; };
      // pass 1: this block's row maximum, then the row's over both blocks
      float mx = -INFINITY;
#pragma unroll 1
      for (int c4 = 0; c4 < 4; ++c4) {
        if (!live(c4)) continue;
        uint32_t v[32];
        tmem_ld32(trow + 32 * c4, v);
        tmem_wait_ld();
        if (full(c4)) {
          float m4[4] = {mx, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
          for (int k = 0; k < 32; ++k) m4[k & 3] = fmaxf(m4[k & 3], __uint_as_float(v[k]));
          mx = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
        } else {
#pragma unroll
          for (int k = 0; k < 32; ++k) { const int c = 32 * c4 + k; if (c >= clo && c <= chi) mx = fmaxf(mx, qok ? __uint_as_float(v[k]) : 0.f); }
        }
      }
      rmax[kb * 128 + r] = mx;
      named_bar_sync(1, 256);
      mx = fmaxf(rmax[r], rmax[128 + r]);
      const float NM = mx == -INFINITY ? 0.f : -mx * SC;
      // pass 2: P = 2^(S scale log2 e - m), rounded to TF32, over the S columns it came from; l sums the rounded P
      float l0 = 0.f, l1 = 0.f;
#pragma unroll 1
      for (int c4 = 0; c4 < 4; ++c4) {
        if (!live(c4)) { tmem_st32(trow + 32 * c4, z); continue; }
        uint32_t v[32];
        tmem_ld32(trow + 32 * c4, v);
        tmem_wait_ld();
        if (full(c4)) {
#pragma unroll
          for (int k = 0; k < 32; k += 2) {
            v[k] = tf32r(ex2f(fmaf(__uint_as_float(v[k]), SC, NM)));
            v[k + 1] = tf32r(ex2f(fmaf(__uint_as_float(v[k + 1]), SC, NM)));
            l0 += __uint_as_float(v[k]);
            l1 += __uint_as_float(v[k + 1]);
          }
        } else {
#pragma unroll
          for (int k = 0; k < 32; k += 2) {
            const int c0 = 32 * c4 + k;
            const float p0 = ex2f(c0 >= clo && c0 <= chi ? fmaf(qok ? __uint_as_float(v[k]) : 0.f, SC, NM) : -INFINITY);
            const float p1 = ex2f(c0 + 1 >= clo && c0 + 1 <= chi ? fmaf(qok ? __uint_as_float(v[k + 1]) : 0.f, SC, NM) : -INFINITY);
            v[k] = tf32r(p0);
            v[k + 1] = tf32r(p1);
            l0 += __uint_as_float(v[k]);
            l1 += __uint_as_float(v[k + 1]);
          }
        }
        tmem_st32(trow + 32 * c4, v);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.p_full[kb]);
      rsum[kb * 128 + r] = l0 + l1;
      named_bar_sync(1, 256);
      const float l = rsum[r] + rsum[128 + r];
      // ---- epilogue: O = acc / l (fp32; 0 on padding / empty rows), this warpgroup's 16 of the 32 columns; LSE by warpgroup 0
      mbar_wait(&B.o_full[ob], (li >> 1) & 1);
      tc_fence_after();
      uint32_t ov[16];
      tmem_ld16(tmem + (lb << 16) + T_O + 32 * ob + 16 * kb, ov);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.t_free[ob]);
      const bool ok = qok && l > 0.f;
      const float inv = ok ? 1.f / l : 0.f;
      float* dst = O + ((size_t)n * S + i) * (H * DH) + h * DH + 16 * kb;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        stg128(dst + 4 * q, make_uint4(__float_as_uint(__uint_as_float(ov[4 * q]) * inv), __float_as_uint(__uint_as_float(ov[4 * q + 1]) * inv),
                                       __float_as_uint(__uint_as_float(ov[4 * q + 2]) * inv), __float_as_uint(__uint_as_float(ov[4 * q + 3]) * inv)));
      if (kb == 0) LSE[((size_t)n * H + h) * S + i] = l > 0.f ? (mx * SC + __log2f(l)) * LN2 : 0.f;
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
