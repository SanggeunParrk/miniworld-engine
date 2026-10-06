// attn_dkv_tf32.cu — the augmented pair-bias attention core's backward dK / dV pass for the fp32 path, sm_100a, TF32 tensor cores
// (the fp32 twin of attn_dkv.cu).
//
// A CTA owns (sample, head, 128 keys) and streams 32-query blocks; one key row per thread:
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32)     P^T = 2^(S^T log2 e / sqrt 48 + bias^T log2 e - LSE)     dS^T = P^T (dP^T - D)
//   dV += P^T dO, dK += dS^T q                  (TS MMAs: P^T / dS^T fp32 in TMEM)
// What the fp32 operands change against the bf16 kernel:
//   * dO and q are read twice per block: K-major (S^T, dP^T: 48-wide rows as a 32-column 128-B-swizzled box + a 16-column 64-B one)
//     and MN-major for dV / dK, which a tf32 MMA takes only in the 128-B swizzle with 32-B atoms (two 32-column boxes).
//   * the bias is read as given, [H, L(query), L(key)]: a thread's 32 values of one block sit one 128-B row apart, the 32 lanes of a
//     warp on consecutive words of each row (no transposed copy, no bank conflicts).
//   * 32-query blocks, 3 stages; K / V in one item slot, which doubles as the item's dK / dV staging once its MMAs are done (the
//     next item's K / V load waits for that store: deeper q / dO staging paid more than a second slot).
// The two warpgroups take alternate blocks of the same key tile and feed one dK / dV accumulator pair.
// TMEM: S^T[w] at w * 64, dP^T[w] at w * 64 + 32, P^T[w] at 128 + w * 64, dS^T[w] at 160 + w * 64, dK at 256, dV at 320.
//
// Head layouts (-DNHEAD -DDHP -DRSQDV, as the bf16 cores; none = 16 x 48): a DH-wide fp32 row is NA 32-column boxes in the
// 128-B swizzle and NBX 16-column ones in the 64-B swizzle (48 = 32 + 16, 32 = 32, 64 = 32 + 32); the MN-major q / dO NV
// 32-column atoms. smem (3 stages): 16 x 48 184 KB, 24 x 32 132 KB, 12 x 64 / 16 x 64 212 KB.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef PP
#define PP 1                             // the two dS warpgroups take turns on the exponentials (named bars 3, 4)
#endif
#ifndef ST
#define ST 3                             // q / dO / bias stages (3 with one K / V slot: 452 / 1554 us against 472 / 1706 for
#endif                                   // 2 stages with two slots, dkv + dqb at A = 48, L384 / L768)
#ifndef KVS
#define KVS 1                            // K / V item slots (each doubles as the item's dK / dV staging)
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
constexpr int BQ = 32, KM = 128, DH = DHP, DM = NHEAD * DHP;
constexpr int NA = DH / 32, NBX = (DH % 32) / 16, NV = (DH + 31) / 32;    // 128-B boxes, 64-B boxes, MN atoms of a head row
constexpr int KVA = KM * 128, KVB = KM * 64, TKV = NA * KVA + NBX * KVB;   // K (or V) of the item: 24 KB at 48 (box j at j KVA)
constexpr int SLOT = 2 * TKV;                                              // K | V
constexpr int QA = BQ * 128, QB = BQ * 64, QM = NV * BQ * 128, TB = KM * BQ * 4;
// q boxes j at S_QA + j QA, dO boxes at S_DA + j QA, then the 16-column boxes, the MN-major q / dO, the bias, LSE | D
constexpr int S_QA = 0, S_DA = NA * QA, S_QB = 2 * NA * QA, S_DB = S_QB + NBX * QB, S_QM = S_DB + NBX * QB, S_DM = S_QM + QM,
              S_BI = S_DM + QM;
constexpr int S_LD = S_BI + TB, STB = S_LD + 1024;                         // + LSE | D of the block's queries (128 B each)
constexpr int O_KV = 0, O_ST = KVS * SLOT, O_BAR = O_ST + ST * STB, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((S_QM % 1024) == 0 && (S_DM % 1024) == 0 && (STB % 1024) == 0 && (SLOT % 1024) == 0 && (TKV % 1024) == 0, "alignment");
constexpr uint32_t T_S = 0, T_P = 128, T_DK = 256, T_DV = 320;
constexpr uint32_t I_S = idesc_tf32(128, BQ), I_O = idesc_tf32(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = RSQDV;
static_assert(T_DV + DH <= 512 && T_DK + DH <= T_DV, "TMEM");

struct Bars {
  uint64_t full[ST], empty[ST], kvfull[KVS], kvempty[KVS], s_full[2], s_free[2], ds_full[2], ds_free[2], acc_full, acc_free;
  uint32_t tmem;
};
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
DEVI float lds32f(uint32_t a) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a)); return v; }

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_dkv_tf32_sm100(const __grid_constant__ CUtensorMap mqa, const __grid_constant__ CUtensorMap mqb,
                       const __grid_constant__ CUtensorMap mqm, const __grid_constant__ CUtensorMap mda,
                       const __grid_constant__ CUtensorMap mdb, const __grid_constant__ CUtensorMap mdm,
                       const __grid_constant__ CUtensorMap mka, const __grid_constant__ CUtensorMap mkb,
                       const __grid_constant__ CUtensorMap mva, const __grid_constant__ CUtensorMap mvb,
                       const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mdka,
                       const __grid_constant__ CUtensorMap mdkb, const __grid_constant__ CUtensorMap mdva,
                       const __grid_constant__ CUtensorMap mdvb, const float* __restrict__ LSE, const float* __restrict__ DD,
                       float* __restrict__ DQZ, int L, int A, int ldq) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kt = L / KM, nb = L / BQ;                                      // nb is a multiple of 4: every item starts on warpgroup 0
  const int items = A * NHEAD * kt;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nblk = my_items * nb;
  auto item_of = [&](int li, int& a, int& k0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    a = wi % A; const int r = wi / A;
    k0 = (r % kt) * KM; head = r / kt;
  };

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < KVS; ++b) { mbar_init(&B.kvfull[b], 1); mbar_init(&B.kvempty[b], 2); }   // released by the two epilogues
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4);
      mbar_init(&B.ds_full[b], 4); mbar_init(&B.ds_free[b], 1);
    }
    mbar_init(&B.acc_full, 2); mbar_init(&B.acc_free, 8);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4 && warp < 8) {                                             // dK / dV start at zero; every MMA accumulates
    const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t trow = tmem + ((uint32_t)(warp & 3) * 32 << 16);
#pragma unroll
    for (int c = 0; c < DH / 16; ++c) { tmem_st16(trow + T_DK + c * 16, z); tmem_st16(trow + T_DV + c * 16, z); }
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  // no setmaxnreg: the kernel compiles to 168 registers, so inc<224> gains the softmax nothing, and dec<56> spilled the control warps
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int G = 0;
      for (int li = 0; li < my_items; ++li) {
        int a, k0, head; item_of(li, a, k0, head);
        const int ks = li % KVS, qcol = head * DH;
        const uint32_t kv = su + O_KV + ks * SLOT;
        if (li >= KVS) mbar_wait(&B.kvempty[ks], ((li / KVS) - 1) & 1);
        mbar_expect_tx(&B.kvfull[ks], SLOT);
#pragma unroll
        for (int j = 0; j < NA; ++j) {
          tma_load_2d(kv + j * KVA, &mka, &B.kvfull[ks], qcol + 32 * j, a * L + k0);
          tma_load_2d(kv + TKV + j * KVA, &mva, &B.kvfull[ks], qcol + 32 * j, a * L + k0);
        }
        if (NBX) {
          tma_load_2d(kv + NA * KVA, &mkb, &B.kvfull[ks], qcol + 32 * NA, a * L + k0);
          tma_load_2d(kv + TKV + NA * KVA, &mvb, &B.kvfull[ks], qcol + 32 * NA, a * L + k0);
        }
        const size_t li0 = ((size_t)a * NHEAD + head) * L;
        for (int n = 0; n < nb; ++n, ++G) {
          const int s = G % ST, row = a * L + n * BQ;
          if (G >= ST) mbar_wait(&B.empty[s], ((G / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], 2 * (NA * QA + NBX * QB + QM) + TB + 2 * BQ * 4);
#pragma unroll
          for (int j = 0; j < NA; ++j) {
            tma_load_2d(st + S_QA + j * QA, &mqa, &B.full[s], qcol + 32 * j, row);
            tma_load_2d(st + S_DA + j * QA, &mda, &B.full[s], qcol + 32 * j, row);
          }
          if (NBX) {
            tma_load_2d(st + S_QB, &mqb, &B.full[s], qcol + 32 * NA, row);
            tma_load_2d(st + S_DB, &mdb, &B.full[s], qcol + 32 * NA, row);
          }
#pragma unroll
          for (int j = 0; j < NV; ++j) {
            tma_load_2d(st + S_QM + j * BQ * 128, &mqm, &B.full[s], qcol + 32 * j, row);
            tma_load_2d(st + S_DM + j * BQ * 128, &mdm, &B.full[s], qcol + 32 * j, row);
          }
#pragma unroll
          for (int j = 0; j < KM / 32; ++j) tma_load_2d(st + S_BI + j * BQ * 128, &mb, &B.full[s], k0 + j * 32, head * L + n * BQ);
          bulk_g2s(st + S_LD, LSE + li0 + n * BQ, BQ * 4, &B.full[s]);
          bulk_g2s(st + S_LD + 512, DD + li0 + n * BQ, BQ * 4, &B.full[s]);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves
    // warpgroup w (blocks G = w, w + 2, ...). dK / dV are shared: every MMA accumulates onto them, the epilogue zeroes them after the
    // read-out, so the two issuers need no ordering between each other.
    const int w = warp - 1;
    auto sdp_ready = [&](int G, int li, int n) {
      return (n > 1 || mbar_test(&B.kvfull[li % KVS], (li / KVS) & 1)) && mbar_test(&B.full[G % ST], (G / ST) & 1) &&
             (G < 2 || mbar_test(&B.s_free[w], ((G >> 1) - 1) & 1));
    };
    auto sdp = [&](int G, int li, int n) {                                 // S^T and dP^T of block G (item li, block n of the item)
      const int s = G % ST, ks = li % KVS;
      if (n <= 1) mbar_wait(&B.kvfull[ks], (li / KVS) & 1);
      mbar_wait(&B.full[s], (G / ST) & 1);
      if (G >= 2) mbar_wait(&B.s_free[w], ((G >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB, kv = su + O_KV + ks * SLOT;
      const uint32_t ts = tmem + T_S + w * 64;
      if (elect_one()) {
#pragma unroll
        for (int j = 0; j < NA; ++j) {
          const uint64_t dka = desc_k128(kv + j * KVA), dqa = desc_k128(st + S_QA + j * QA);
#pragma unroll
          for (int k = 0; k < 4; ++k) umma_ss_tf32(ts, dka + (uint64_t)(k * 2), dqa + (uint64_t)(k * 2), I_S, (j > 0 || k > 0) ? 1u : 0u);
        }
        if (NBX) {
          const uint64_t dkb = desc_sw64(kv + NA * KVA), dqb = desc_sw64(st + S_QB);
#pragma unroll
          for (int k = 0; k < 2; ++k) umma_ss_tf32(ts, dkb + (uint64_t)(k * 2), dqb + (uint64_t)(k * 2), I_S, 1u);
        }
#pragma unroll
        for (int j = 0; j < NA; ++j) {
          const uint64_t dva = desc_k128(kv + TKV + j * KVA), dda = desc_k128(st + S_DA + j * QA);
#pragma unroll
          for (int k = 0; k < 4; ++k) umma_ss_tf32(ts + 32, dva + (uint64_t)(k * 2), dda + (uint64_t)(k * 2), I_S, (j > 0 || k > 0) ? 1u : 0u);
        }
        if (NBX) {
          const uint64_t dvb = desc_sw64(kv + TKV + NA * KVA), ddb = desc_sw64(st + S_DB);
#pragma unroll
          for (int k = 0; k < 2; ++k) umma_ss_tf32(ts + 32, dvb + (uint64_t)(k * 2), ddb + (uint64_t)(k * 2), I_S, 1u);
        }
        tc_commit(&B.s_full[w]);
      }
      __syncwarp();
    };
    if (w < nblk) sdp(w, 0, w);
    for (int G = w, li = 0, n = w; G < nblk; G += 2) {                     // (li, n) advance incrementally (nb is even)
      const int s = G % ST;
      int li2 = li, n2 = n + 2;
      if (n2 >= nb) { n2 -= nb; ++li2; }
      // S^T(G + 2) as soon as its operands are in and warpgroup w holds S^T(G) in registers, but never ahead of a ready dV / dK(G)
      bool issued = G + 2 >= nblk;
      while (!__shfl_sync(0xffffffffu, (int)mbar_test(&B.ds_full[w], (G >> 1) & 1), 0)) {
        if (!issued && __shfl_sync(0xffffffffu, (int)sdp_ready(G + 2, li2, n2), 0)) { sdp(G + 2, li2, n2); issued = true; }
        else __nanosleep(20);
      }
      if (n <= 1 && li >= 1) mbar_wait(&B.acc_free, (li - 1) & 1);          // the previous item's dK / dV have been read out (and zeroed)
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB;
      // q / dO as MN-major B: queries as rows (8 queries = 1 KB per K step), two 32-column atoms 4 KB apart
      const uint64_t dq = desc_mn32b(st + S_QM, BQ * 128), ddo = desc_mn32b(st + S_DM, BQ * 128);
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < BQ / 8; ++k)
          umma_ts_tf32(tmem + T_DV, tmem + T_P + w * 64 + k * 8, ddo + (uint64_t)((k * 1024) >> 4), I_O, 1u);
#pragma unroll
        for (int k = 0; k < BQ / 8; ++k)
          umma_ts_tf32(tmem + T_DK, tmem + T_P + w * 64 + 32 + k * 8, dq + (uint64_t)((k * 1024) >> 4), I_O, 1u);
        tc_commit(&B.ds_free[w]);
        tc_commit(&B.empty[s]);
        if (n >= nb - 2) tc_commit(&B.acc_full);
      }
      __syncwarp();
      if (!issued) sdp(G + 2, li2, n2);
      li = li2; n = n2;
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ P^T / dS^T warpgroups
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    if (PP && w == 1) named_bar_arrive(3, 256);                          // warpgroup 0 takes the first exp turn
    for (int G = w, li = 0, n = w; G < nblk; G += 2, n += 2) {
      if (n >= nb) { n -= nb; ++li; }
      const int s = G % ST;
      const uint32_t st = su + O_ST + s * STB;
      const uint32_t sb = st + S_BI + (r >> 5) * (BQ * 128) + (r & 31) * 4, sl = st + S_LD;
      mbar_wait(&B.full[s], (G / ST) & 1);                                 // bias, LSE, D of this block
      mbar_wait(&B.s_full[w], (G >> 1) & 1);
      if (G >= 2) mbar_wait(&B.ds_free[w], ((G >> 1) - 1) & 1);            // dV / dK of this warpgroup's previous block are done
      tc_fence_after();
      if (PP) named_bar_sync(3 + w, 256);                                  // my exp turn
#pragma unroll
      for (int h = 0; h < 2; ++h) {                                        // halves of 16 queries: S^T / dP^T in, P^T / dS^T out
        uint32_t sv[16], dv[16];
        tmem_ld16(trow + T_S + w * 64 + h * 16, sv);
        tmem_ld16(trow + T_S + w * 64 + 32 + h * 16, dv);
        tmem_wait_ld();
        if (h == 1) {                                                      // S^T / dP^T read: the next S^T MMA may overwrite them
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int jq = h * 16 + j;
          const float lse = lds32f(sl + jq * 4), dd = lds32f(sl + 512 + jq * 4), b = lds32f(sb + jq * 128);
          const float p = ex2f(fmaf(__uint_as_float(sv[j]), RSQD * LOG2E, fmaf(b, LOG2E, -lse)));
          sv[j] = __float_as_uint(p);
          dv[j] = __float_as_uint(p * (__uint_as_float(dv[j]) - dd));
        }
        tmem_st16(trow + T_P + w * 64 + h * 16, sv);
        tmem_st16(trow + T_P + w * 64 + 32 + h * 16, dv);
      }
      if (PP) named_bar_arrive(4 - w, 256);                                // hand the MUFU over
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ds_full[w]);
      if (n >= nb - 2) {                                                   // this warpgroup's last block of the item
        // ---- the item's epilogue: warpgroup 0 writes dK (scaled by 1 / sqrt 48), warpgroup 1 dV, one key row per thread, staged in
        // the item's K (dK) or V (dV) slot: every MMA that read the slot is done once acc_full has fired
        int a, k0, head; item_of(li, a, k0, head);
        mbar_wait(&B.acc_full, li & 1);
        tc_fence_after();
        const float sc = w == 0 ? RSQD : 1.f;
        const uint32_t tc = w == 0 ? T_DK : T_DV;
        uint32_t v[DH];
#pragma unroll
        for (int c = 0; c < DH / 16; ++c) tmem_ld16(trow + tc + c * 16, *reinterpret_cast<uint32_t(*)[16]>(v + 16 * c));
        tmem_wait_ld();
        {
          const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
          for (int c = 0; c < DH / 16; ++c) tmem_st16(trow + tc + c * 16, z);
          tmem_wait_st();
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_free);
        if (DQZ) {                                                         // zero this tile of dQ for attn_dqb_tf32's reductions (it runs next)
          constexpr int C4 = DH / 4;                                       // 128 rows x C4 float4, lanes along rows, half per warpgroup
          float4* zq = reinterpret_cast<float4*>(DQZ + ((size_t)a * L + k0) * ldq + head * DH);   // dQ rows: ldq floats apart
#pragma unroll
          for (int k = 0; k < C4 / 2; ++k) {
            const int idx = w * 64 * C4 + k * 128 + (int)r, row = idx / C4, ch = idx % C4;
            zq[(size_t)row * (ldq / 4) + ch] = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        const uint32_t xa = su + O_KV + (li % KVS) * SLOT + w * TKV, xb = xa + NA * KVA;   // A boxes at xa + j KVA, the B box at xb
        auto put = [&](int q) {
          return make_uint4(__float_as_uint(__uint_as_float(v[4 * q]) * sc), __float_as_uint(__uint_as_float(v[4 * q + 1]) * sc),
                            __float_as_uint(__uint_as_float(v[4 * q + 2]) * sc), __float_as_uint(__uint_as_float(v[4 * q + 3]) * sc));
        };
        named_bar_sync(1 + w, 128);
#pragma unroll
        for (int j = 0; j < NA; ++j) {
#pragma unroll
          for (int q = 0; q < 8; ++q) sts128(xa + j * KVA + sw128(r, q), put(8 * j + q));
        }
        if (NBX) {
#pragma unroll
          for (int q = 0; q < 4; ++q) sts128(xb + sw64(r, q), put(8 * NA + q));
        }
        fence_proxy_async();
        named_bar_sync(1 + w, 128);
        if (r == 0) {
#pragma unroll
          for (int j = 0; j < NA; ++j) tma_store_2d(w == 0 ? &mdka : &mdva, xa + j * KVA, head * DH + 32 * j, a * L + k0);
          if (NBX) tma_store_2d(w == 0 ? &mdkb : &mdvb, xb, head * DH + 32 * NA, a * L + k0);
          tma_store_commit();
          tma_store_wait_read0();                                          // the slot may take the next item's K / V
          mbar_arrive(&B.kvempty[li % KVS]);
        }
      }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
