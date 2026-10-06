// attn_dqb_tf32.cu — the augmented pair-bias attention core's backward dQ + dbias pass for the fp32 path, sm_100a, TF32 tensor cores
// (the fp32 twin of attn_dqb.cu).
//
// Same fusion and ownership as attn_dqb: a CTA owns (head, 128-query tile, key chunk) and walks all A samples, so dbias is summed on
// chip and written once; dQ is a partial per key chunk, added into a zeroed fp32 dQ with v4 reductions. Per sample (warpgroup w owns
// keys w * 32 .. w * 32 + 31 of the chunk, one query row per thread):
//   S = q K_w^T, dP = dO V_w^T  (TMEM, fp32)      P = 2^(S log2 e / sqrt 48 + bias log2 e - LSE)      dS = P (dP - D)
//   dbias_w += dS (registers)   dS -> TMEM (fp32)  dQ += dS K_w  (TS MMA, both warpgroups into one accumulator)
// Sample split (``NS`` > 1, the host's choice when the (head, query tile, key chunk) items alone leave SMs idle -- L = 128 at 16
// heads: 32 items for 148 SMs): an item is (head, query tile, key chunk, sample group of A / NS samples) and its dbias leaves as
// v4 reductions into a zeroed DB instead of plain stores. dQ goes to rows of stride ``ldq`` floats (a column block of a wider
// buffer, e.g. the q columns of the q|k|v|g gradient).
// What the fp32 operands change: 64-key chunks (the fp32 tiles of a 128-key chunk do not fit twice), K read twice per sample --
// K-major for S (32 + 16 columns) and MN-major for dQ, which a tf32 MMA takes only in the 128-B swizzle with 32-B atoms (two
// 32-column boxes) -- and the dQ partials leave through per-thread v4 reductions (no staging buffer left).
// TMEM: S[w] at w * 64, dP[w] at w * 64 + 32, dS[w] at 128 + w * 32 (fp32), dQ[b] at 192 + b * DH (NDQ buffers, round robin).
//
// Head layouts (-DNHEAD -DDHP -DRSQDV, as the bf16 cores; none = 16 x 48): a DH-wide fp32 row is NA 32-column boxes in the
// 128-B swizzle and NBX 16-column ones in the 64-B swizzle (48 = 32 + 16, 32 = 32, 64 = 32 + 32); the MN-major K NV 32-column
// atoms. smem: 16 x 48 208 KB (STA 2, STK 2), 24 x 32 144 KB (2, 2), 12 x 64 / 16 x 64 224 KB (STA 2, STK 1: a second K slot
// does not fit; the next step's K then loads behind this step's dQ MMAs).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef PP
#define PP 1                             // the two dS warpgroups take turns on the exponentials (named bars 3, 4)
#endif
#ifndef DHP
#define DHP 48                           // head width: 48 (16 x 48), 32 (24 x 32), 64 (12 x 64; 16 x 64 at d 1024)
#endif
#ifndef NHEAD
#define NHEAD 16
#endif
#ifndef RSQDV
#define RSQDV 0.14433756729740643f       // 1 / sqrt(head dim): 1 / sqrt 48; 1 / sqrt 32 = 0.17677669529663687f; 1 / sqrt 64 = 0.125f
#endif
static_assert(DHP == 64 || DHP == 48 || DHP == 32, "TF32 head width 64, 48 or 32");
#ifndef STA
#define STA 2                            // q | dO | V ring (60 KB slots at 48): released as soon as both S / dP MMAs of the step are done
#endif
#ifndef STK
#define STK (DHP == 64 ? 1 : 2)          // K ring (28 KB slots at 48): released after the step's dQ MMAs
#endif
#ifndef ROT
#define ROT 1                            // start each key chunk's sample walk at a different sample (spreads the dQ reductions)
#endif
constexpr int NDQ = 3;                   // dQ accumulator buffers (the epilogue of step g runs NDQ - 1 = 2 steps later)
constexpr int BN = 32, KC = 64, DH = DHP, QM = 128, DM = NHEAD * DHP;
constexpr int NA = DH / 32, NBX = (DH % 32) / 16, NV = (DH + 31) / 32;    // 128-B boxes, 64-B boxes, MN atoms of a head row
constexpr int QA = QM * 128, QB = QM * 64, VA = KC * 128, VB = KC * 64, KMN = NV * KC * 128;
constexpr int TQX = NA * QA + NBX * QB;                                                        // q (or dO) of one sample
// q boxes j at A_Q + j QA (the 16-column box after them), dO likewise at A_D, V boxes j at A_VA + j VA, its 16-column box at A_VB
constexpr int A_Q = 0, A_D = TQX, A_VA = 2 * TQX, A_VB = A_VA + NA * VA, SA = A_VB + NBX * VB;  // q | dO | V: 60 KB at 48
constexpr int K_A = 0, K_B = NA * VA, K_M = K_B + NBX * VB, SK = K_M + KMN;                   // K (K-major) | K (MN-major): 28 KB at 48
constexpr int TBI = QM * BN * 4;                                                               // bias: two [128 q][32 keys] boxes
constexpr int O_A = 0, O_K = STA * SA, O_B = O_K + STK * SK, O_BAR = O_B + 2 * TBI, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((SA % 1024) == 0 && (A_D % 1024) == 0 && (A_VA % 1024) == 0 && (SK % 1024) == 0 && (K_M % 1024) == 0 &&
              (O_K % 1024) == 0 && (O_B % 1024) == 0, "alignment");
constexpr uint32_t T_S = 0, T_DS = 128, T_DQ = 192;
static_assert(T_DQ + NDQ * DH <= 512, "TMEM");
constexpr uint32_t I_S = idesc_tf32(128, BN), I_DQ = idesc_tf32(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = RSQDV;
constexpr int EC0 = DH == 32 ? 16 : 32;  // the dQ read-out: warpgroup 0 takes columns 0 .. EC0 - 1, warpgroup 1 the rest

struct Bars {
  uint64_t fullA[STA], emptyA[STA], fullK[STK], emptyK[STK], bfull, bempty, s_full[2], s_free[2], ds_full[2], ds_free[2], dq_full[NDQ],
      dq_free[NDQ];
  uint32_t tmem;
};
DEVI void red4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_dqb_tf32_sm100(const __grid_constant__ CUtensorMap mqa, const __grid_constant__ CUtensorMap mqb,
                       const __grid_constant__ CUtensorMap mda, const __grid_constant__ CUtensorMap mdb,
                       const __grid_constant__ CUtensorMap mva, const __grid_constant__ CUtensorMap mvb,
                       const __grid_constant__ CUtensorMap mka, const __grid_constant__ CUtensorMap mkb,
                       const __grid_constant__ CUtensorMap mkm, const __grid_constant__ CUtensorMap mb,
                       const float* __restrict__ LSE, const float* __restrict__ DD, float* __restrict__ DQ, float* __restrict__ DB,
                       int L, int A, int NS, int ldq) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nch = L / KC;
  const int AS = A / NS;                                                   // samples per item (the host keeps A % NS == 0)
  const int items = NHEAD * mt * nch * NS;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int ng = my_items * AS;                                            // (item, sample) steps of this CTA
  // an item's sample group is its first sample s0 = (wi % NS) AS; the walk covers s0 .. s0 + AS - 1
  auto item_of = [&](int li, int& c, int& m0, int& head, int& s0) {
    int wi = (int)blockIdx.x + li * (int)gridDim.x;
    s0 = (wi % NS) * AS; wi /= NS;
    c = wi % nch; const int r = wi / nch;
    m0 = (r % mt) * QM; head = r / mt;
  };
  auto samp = [&](int c, int s0, int i) { return s0 + (ROT ? (i + c) % AS : i); };

  if (tid == 0) {
    for (int s = 0; s < STA; ++s) { mbar_init(&B.fullA[s], 1); mbar_init(&B.emptyA[s], 2); }
    for (int s = 0; s < STK; ++s) { mbar_init(&B.fullK[s], 1); mbar_init(&B.emptyK[s], 2); }
    mbar_init(&B.bfull, 1); mbar_init(&B.bempty, 8);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4);
      mbar_init(&B.ds_full[b], 4); mbar_init(&B.ds_free[b], 1);
    }
    for (int b = 0; b < NDQ; ++b) { mbar_init(&B.dq_full[b], 2); mbar_init(&B.dq_free[b], 8); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4 && warp < 8) {                                             // the dQ buffers start at zero; every dQ MMA accumulates
    const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t trow = tmem + ((uint32_t)(warp & 3) * 32 << 16);
    for (int c = 0; c < NDQ * DH / 16; ++c) tmem_st16(trow + T_DQ + c * 16, z);
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  // no setmaxnreg: the kernel compiles to 168 registers, so inc<224> gains the softmax nothing, and dec<56> spilled the control warps
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int c, m0, head, s0; item_of(li, c, m0, head, s0);
        const int qcol = head * DH;
        if (li >= 1) mbar_wait(&B.bempty, (li - 1) & 1);
        mbar_expect_tx(&B.bfull, 2 * TBI);
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_B + w * TBI, &mb, &B.bfull, c * KC + w * BN, head * L + m0);
        for (int i = 0; i < AS; ++i, ++g) {
          const int sa = g % STA, sk = g % STK, a = samp(c, s0, i);
          if (g >= STA) mbar_wait(&B.emptyA[sa], ((g / STA) - 1) & 1);
          const uint32_t pa = su + O_A + sa * SA;
          mbar_expect_tx(&B.fullA[sa], SA);
#pragma unroll
          for (int j = 0; j < NA; ++j) {
            tma_load_2d(pa + A_Q + j * QA, &mqa, &B.fullA[sa], qcol + 32 * j, a * L + m0);
            tma_load_2d(pa + A_D + j * QA, &mda, &B.fullA[sa], qcol + 32 * j, a * L + m0);
            tma_load_2d(pa + A_VA + j * VA, &mva, &B.fullA[sa], qcol + 32 * j, a * L + c * KC);
          }
          if (NBX) {
            tma_load_2d(pa + A_Q + NA * QA, &mqb, &B.fullA[sa], qcol + 32 * NA, a * L + m0);
            tma_load_2d(pa + A_D + NA * QA, &mdb, &B.fullA[sa], qcol + 32 * NA, a * L + m0);
            tma_load_2d(pa + A_VB, &mvb, &B.fullA[sa], qcol + 32 * NA, a * L + c * KC);
          }
          if (g >= STK) mbar_wait(&B.emptyK[sk], ((g / STK) - 1) & 1);
          const uint32_t pk = su + O_K + sk * SK;
          mbar_expect_tx(&B.fullK[sk], SK);
#pragma unroll
          for (int j = 0; j < NA; ++j) tma_load_2d(pk + K_A + j * VA, &mka, &B.fullK[sk], qcol + 32 * j, a * L + c * KC);
          if (NBX) tma_load_2d(pk + K_B, &mkb, &B.fullK[sk], qcol + 32 * NA, a * L + c * KC);
#pragma unroll
          for (int j = 0; j < NV; ++j) tma_load_2d(pk + K_M + j * KC * 128, &mkm, &B.fullK[sk], qcol + 32 * j, a * L + c * KC);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves
    // warpgroup w. dQ is shared: every MMA accumulates onto it and the epilogue zeroes it after the read-out.
    const int w = warp - 1;
    auto sdp_ready = [&](int g) { return mbar_test(&B.fullA[g % STA], (g / STA) & 1) && mbar_test(&B.fullK[g % STK], (g / STK) & 1) &&
                                         (g < 1 || mbar_test(&B.s_free[w], (g - 1) & 1)); };
    auto sdp = [&](int g) {                                                // S and dP of step g
      mbar_wait(&B.fullA[g % STA], (g / STA) & 1);
      mbar_wait(&B.fullK[g % STK], (g / STK) & 1);
      if (g >= 1) mbar_wait(&B.s_free[w], (g - 1) & 1);
      tc_fence_after();
      const uint32_t pa = su + O_A + (g % STA) * SA, pk = su + O_K + (g % STK) * SK;
      const uint32_t ts = tmem + T_S + w * 64;
      if (elect_one()) {
#pragma unroll
        for (int j = 0; j < NA; ++j) {
          const uint64_t dqa = desc_k128(pa + A_Q + j * QA), dka = desc_k128(pk + K_A + j * VA + w * BN * 128);
#pragma unroll
          for (int k = 0; k < 4; ++k) umma_ss_tf32(ts, dqa + (uint64_t)(k * 2), dka + (uint64_t)(k * 2), I_S, (j > 0 || k > 0) ? 1u : 0u);
        }
        if (NBX) {
          const uint64_t dqb = desc_sw64(pa + A_Q + NA * QA), dkb = desc_sw64(pk + K_B + w * BN * 64);
#pragma unroll
          for (int k = 0; k < 2; ++k) umma_ss_tf32(ts, dqb + (uint64_t)(k * 2), dkb + (uint64_t)(k * 2), I_S, 1u);
        }
#pragma unroll
        for (int j = 0; j < NA; ++j) {
          const uint64_t dda = desc_k128(pa + A_D + j * QA), dva = desc_k128(pa + A_VA + j * VA + w * BN * 128);
#pragma unroll
          for (int k = 0; k < 4; ++k) umma_ss_tf32(ts + 32, dda + (uint64_t)(k * 2), dva + (uint64_t)(k * 2), I_S, (j > 0 || k > 0) ? 1u : 0u);
        }
        if (NBX) {
          const uint64_t ddb = desc_sw64(pa + A_D + NA * QA), dvb = desc_sw64(pa + A_VB + w * BN * 64);
#pragma unroll
          for (int k = 0; k < 2; ++k) umma_ss_tf32(ts + 32, ddb + (uint64_t)(k * 2), dvb + (uint64_t)(k * 2), I_S, 1u);
        }
        tc_commit(&B.s_full[w]);
        tc_commit(&B.emptyA[g % STA]);
      }
      __syncwarp();
    };
    if (ng > 0) sdp(0);
    for (int g = 0; g < ng; ++g) {
      const int b = g % NDQ;
      // S(g + 1) as soon as its operands are in and warpgroup w holds S(g) in registers, but never ahead of a ready dQ(g)
      bool issued = g + 1 >= ng;
      while (!__shfl_sync(0xffffffffu, (int)mbar_test(&B.ds_full[w], g & 1), 0)) {
        if (!issued && __shfl_sync(0xffffffffu, (int)sdp_ready(g + 1), 0)) { sdp(g + 1); issued = true; }
        else __nanosleep(20);
      }
      if (g >= NDQ) mbar_wait(&B.dq_free[b], ((g / NDQ) - 1) & 1);         // step g - NDQ's dQ has been read out (and zeroed)
      tc_fence_after();
      // K_w as MN-major B: keys as rows (8 keys = 1 KB per K step), two 32-column atoms 8 KB apart
      const uint64_t dk = desc_mn32b(su + O_K + (g % STK) * SK + K_M + w * BN * 128, KC * 128);
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < BN / 8; ++k)
          umma_ts_tf32(tmem + T_DQ + b * DH, tmem + T_DS + w * 32 + k * 8, dk + (uint64_t)((k * 1024) >> 4), I_DQ, 1u);
        tc_commit(&B.ds_free[w]);
        tc_commit(&B.emptyK[g % STK]);
        tc_commit(&B.dq_full[b]);
      }
      __syncwarp();
      if (!issued) sdp(g + 1);
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dS warpgroups
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    if (PP && w == 1) named_bar_arrive(3, 256);                          // warpgroup 0 takes the first exp turn
    auto epi = [&](int g, int arow, int head) {                            // dQ of step g (rows arow .. arow + 127 of DQ, head): TMEM -> reductions
      const int b = g % NDQ;
      mbar_wait(&B.dq_full[b], (g / NDQ) & 1);
      tc_fence_after();
      // warpgroup 0: columns 0 .. EC0 - 1, warpgroup 1: EC0 .. DH - 1 (48: 32 + 16, 64: 32 + 32, 32: 16 + 16)
      const int c0 = w == 0 ? 0 : EC0, nc = w == 0 ? EC0 : DH - EC0;
      const uint32_t tq = trow + T_DQ + b * DH + c0;
      uint32_t v[32];
      if (nc == 32) tmem_ld32(tq, v);
      else tmem_ld16(tq, *reinterpret_cast<uint32_t(*)[16]>(v));
      tmem_wait_ld();
      {
        const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        tmem_st16(tq, z);
        if (nc == 32) tmem_st16(tq + 16, z);
        tmem_wait_st();
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dq_free[b]);
      float* drow = DQ + ((size_t)arow + r) * ldq + head * DH + c0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (k * 4 >= nc) break;
        red4(drow + 4 * k, __uint_as_float(v[4 * k]) * RSQD, __uint_as_float(v[4 * k + 1]) * RSQD, __uint_as_float(v[4 * k + 2]) * RSQD,
             __uint_as_float(v[4 * k + 3]) * RSQD);
      }
    };
    float db[BN];
    // positions advance incrementally: the runtime divisions of item_of run once per item, not per step
    // (a sample group starts at a multiple of AS, so the walk wraps where a + 1 is one: no group start kept in registers)
    auto advance = [&](int& pli, int& pi, int& pc, int& pm0, int& ph, int& pa) {
      if (++pi == AS) { pi = 0; if (++pli < my_items) { int ps0; item_of(pli, pc, pm0, ph, ps0); pa = samp(pc, ps0, 0); } }
      else pa = ((pa + 1) % AS == 0) ? pa + 1 - AS : pa + 1;
    };
    int li = 0, i = 0, c = 0, m0 = 0, head = 0, a = 0;
    if (ng > 0) { int s0; item_of(0, c, m0, head, s0); a = samp(c, s0, 0); }
    int nli = li, ni = i, nc_ = c, nm0 = m0, nh = head, na = a;          // the next step's position (LSE / D prefetch)
    float lse_n = 0.f, dd_n = 0.f;
    if (ng > 0) { const size_t ri = ((size_t)a * NHEAD + head) * L + m0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
    int e1r = 0, e1h = 0, e2r = 0, e2h = 0;                                // (DQ row, head) of steps g - 1 and g - 2
    const uint32_t sb = su + O_B + w * TBI;
    for (int g = 0; g < ng; ++g) {
      const float lse = lse_n, dd = dd_n;                                  // loaded one step ahead
      advance(nli, ni, nc_, nm0, nh, na);
      if (g + 1 < ng) { const size_t ri = ((size_t)na * NHEAD + nh) * L + nm0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
      if (i == 0) {
#pragma unroll
        for (int j = 0; j < BN; ++j) db[j] = 0.f;
        mbar_wait(&B.bfull, li & 1);
      }
      mbar_wait(&B.s_full[w], g & 1);
      if (g >= 1) mbar_wait(&B.ds_free[w], (g - 1) & 1);                   // dQ(g - 1) has consumed the previous dS
      tc_fence_after();
      if (PP) named_bar_sync(3 + w, 256);                                  // my exp turn
#pragma unroll
      for (int h = 0; h < 2; ++h) {                                        // halves of 16 keys: S / dP in, dS out
        uint32_t sv[16], dv[16];
        tmem_ld16(trow + T_S + w * 64 + h * 16, sv);
        tmem_ld16(trow + T_S + w * 64 + 32 + h * 16, dv);
        tmem_wait_ld();
        if (h == 1) {                                                      // S / dP read: the next S MMA may overwrite them
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {                                      // 4 keys of fp32 bias per 16-byte chunk
          const uint4 bw = lds128(sb + sw128(r, h * 4 + q));
          const float bb[4] = {__uint_as_float(bw.x), __uint_as_float(bw.y), __uint_as_float(bw.z), __uint_as_float(bw.w)};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = 4 * q + e;
            const float p = ex2f(fmaf(__uint_as_float(sv[j]), RSQD * LOG2E, fmaf(bb[e], LOG2E, -lse)));
            const float ds = p * (__uint_as_float(dv[j]) - dd);
            db[h * 16 + j] += ds;
            sv[j] = __float_as_uint(ds);
          }
        }
        tmem_st16(trow + T_DS + w * 32 + h * 16, sv);
      }
      if (PP) named_bar_arrive(4 - w, 256);                                // hand the MUFU over
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ds_full[w]);
      if (g >= 2) epi(g - 2, e2r, e2h);
      if (i == AS - 1) {
        // ---- the item's dbias: this thread's 32 keys of query row m0 + r, natural units: plain stores, or (sample groups)
        // reductions into the zeroed DB
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.bempty);
        float* brow = DB + ((size_t)head * L + m0 + r) * L + c * KC + w * BN;
        if (NS == 1) {
#pragma unroll
          for (int k = 0; k < BN / 4; ++k)
            *reinterpret_cast<float4*>(brow + 4 * k) = make_float4(db[4 * k], db[4 * k + 1], db[4 * k + 2], db[4 * k + 3]);
        } else {
#pragma unroll
          for (int k = 0; k < BN / 4; ++k) red4(brow + 4 * k, db[4 * k], db[4 * k + 1], db[4 * k + 2], db[4 * k + 3]);
        }
      }
      e2r = e1r; e2h = e1h; e1r = a * L + m0; e1h = head;
      advance(li, i, c, m0, head, a);
    }
    if (ng >= 2) epi(ng - 2, e2r, e2h);
    if (ng >= 1) epi(ng - 1, e1r, e1h);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
