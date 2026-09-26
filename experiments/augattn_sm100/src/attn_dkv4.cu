// attn_dkv4.cu — attn_dkv.cu with FOUR dS warpgroups sharing each query block (16 query columns each) instead of two warpgroups on
// alternate blocks. Same fusion and arithmetic; the point is latency hiding: 4 dS warps per SM sub-partition instead of 2, and with
// every warpgroup on the same block the S^T / dP^T and P^T / dS^T tiles are double-buffered in TMEM.
//
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32)     P^T = 2^(S^T log2 e / sqrt 48 + bias^T log2 e - LSE)     dS^T = P^T (dP^T - D)
//   dV += P^T dO, dK += dS^T q                  (TS MMAs, one issuer)
// TMEM: S^T[b] at b * 128, dP^T[b] at b * 128 + 64, P^T[b] at 256 + b * 64 (bf16, 32 cols), dS^T[b] at 288 + b * 64, dK at 384, dV at 432.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ST
#define ST 4
#endif
constexpr int BQ = 64, KM = 128, DH = 48, DM = 768, NW = 4, QW = BQ / NW;  // QW = 16 query columns per warpgroup
constexpr int TQ = BQ * 128, TKV = KM * 128, TBT = KM * BQ * 2;
constexpr int STB = 2 * TQ + TBT + 1024;                                  // q | dO | bias^T | LSE, D
constexpr int O_KV = 0, O_ST = 2 * 2 * TKV, O_BAR = O_ST + ST * STB;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_P = 256, T_DK = 384, T_DV = 432;
constexpr uint32_t I_S = idesc_bf16(128, BQ), I_O = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = 0.14433756729740643f;
constexpr int NTH = 128 + NW * 128;

struct Bars {
  uint64_t full[ST], empty[ST], kvfull[2], kvempty[2], s_full[2], s_free[2], ds_full[2], ds_free[2], acc_full, acc_free;
  uint32_t tmem;
};
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
DEVI void tmem_ld16_(uint32_t a, uint32_t (&r)[16]) { tmem_ld16(a, r); }
DEVI void tmem_st8_(uint32_t a, const uint32_t (&r)[8]) { tmem_st8(a, r); }

extern "C" __global__ void __launch_bounds__(NTH, 1)
augattn_dkv4_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mbt, const float* __restrict__ LSE,
                   const float* __restrict__ DD, float* __restrict__ DK, float* __restrict__ DV, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kt = L / KM, nb = L / BQ;
  const int items = A * 16 * kt;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nblk = my_items * nb;
  auto item_of = [&](int li, int& a, int& k0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    a = wi % A; const int r = wi / A;
    k0 = (r % kt) * KM; head = r / kt;
  };

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.kvfull[b], 1); mbar_init(&B.kvempty[b], 1);
      mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4 * NW);
      mbar_init(&B.ds_full[b], 4 * NW); mbar_init(&B.ds_free[b], 1);
    }
    mbar_init(&B.acc_full, 1); mbar_init(&B.acc_free, 4 * NW);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int G = 0;
      for (int li = 0; li < my_items; ++li) {
        int a, k0, head; item_of(li, a, k0, head);
        const int ks = li & 1, qcol = head * DH;
        if (li >= 2) mbar_wait(&B.kvempty[ks], ((li >> 1) - 1) & 1);
        mbar_expect_tx(&B.kvfull[ks], 2 * KM * DH * 2);
        tma_load_2d(su + O_KV + ks * 2 * TKV, &mk, &B.kvfull[ks], qcol, a * L + k0);
        tma_load_2d(su + O_KV + ks * 2 * TKV + TKV, &mv, &B.kvfull[ks], qcol, a * L + k0);
        const size_t li0 = ((size_t)a * 16 + head) * L;
        for (int n = 0; n < nb; ++n, ++G) {
          const int s = G % ST;
          if (G >= ST) mbar_wait(&B.empty[s], ((G / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], 2 * BQ * DH * 2 + TBT + 2 * BQ * 4);
          tma_load_2d(st, &mq, &B.full[s], qcol, a * L + n * BQ);
          tma_load_2d(st + TQ, &mdo, &B.full[s], qcol, a * L + n * BQ);
          tma_load_2d(st + 2 * TQ, &mbt, &B.full[s], n * BQ, head * L + k0);
          bulk_g2s(st + 2 * TQ + TBT, LSE + li0 + n * BQ, BQ * 4, &B.full[s]);
          bulk_g2s(st + 2 * TQ + TBT + 512, DD + li0 + n * BQ, BQ * 4, &B.full[s]);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    auto sdp = [&](int G) {
      const int li = G / nb, n = G % nb, s = G % ST, b = G & 1, ks = li & 1;
      if (n == 0) mbar_wait(&B.kvfull[ks], (li >> 1) & 1);
      mbar_wait(&B.full[s], (G / ST) & 1);
      if (G >= 2) mbar_wait(&B.s_free[b], ((G >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB, kv = su + O_KV + ks * 2 * TKV;
      const uint64_t dk = desc_k128(kv), dv = desc_k128(kv + TKV), dq = desc_k128(st), ddo = desc_k128(st + TQ);
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 3; ++k) umma_ss(tmem + T_S + b * 128, dk + (uint64_t)(k * 2), dq + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < 3; ++k) umma_ss(tmem + T_S + b * 128 + 64, dv + (uint64_t)(k * 2), ddo + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
        tc_commit(&B.s_full[b]);
      }
      __syncwarp();
    };
    if (nblk > 0) sdp(0);
    if (nblk > 1) sdp(1);
    for (int G = 0; G < nblk; ++G) {
      const int li = G / nb, n = G % nb, s = G % ST, b = G & 1;
      mbar_wait(&B.ds_full[b], (G >> 1) & 1);
      if (n == 0 && li >= 1) mbar_wait(&B.acc_free, (li - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB;
      const uint64_t dq = desc_mn128(st, 8192), ddo = desc_mn128(st + TQ, 8192);
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          umma_ts(tmem + T_DV, tmem + T_P + b * 64 + k * 8, ddo + (uint64_t)(k * 2048 >> 4), I_O, (n > 0 || k > 0) ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          umma_ts(tmem + T_DK, tmem + T_P + b * 64 + 32 + k * 8, dq + (uint64_t)(k * 2048 >> 4), I_O, (n > 0 || k > 0) ? 1u : 0u);
        tc_commit(&B.ds_free[b]);
        tc_commit(&B.empty[s]);
        if (n == nb - 1) { tc_commit(&B.acc_full); tc_commit(&B.kvempty[li & 1]); }
      }
      __syncwarp();
      if (G + 2 < nblk) sdp(G + 2);
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ P^T / dS^T: warpgroup w takes query
    // columns w * 16 .. w * 16 + 15 of every block; one key row per thread
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const f2 CQL = mk2(RSQD * LOG2E, RSQD * LOG2E), L2E = mk2(LOG2E, LOG2E);
    for (int G = 0; G < nblk; ++G) {
      const int li = G / nb, n = G % nb, s = G % ST, b = G & 1;
      const uint32_t st = su + O_ST + s * STB, sb = st + 2 * TQ, sl = sb + TBT;
      mbar_wait(&B.full[s], (G / ST) & 1);
      mbar_wait(&B.s_full[b], (G >> 1) & 1);
      tc_fence_after();
      uint32_t sv[16], dv[16];
      tmem_ld16_(trow + T_S + b * 128 + w * QW, sv);
      tmem_ld16_(trow + T_S + b * 128 + 64 + w * QW, dv);
      const uint4 bw0 = lds128(sb + sw128(r, 2 * w)), bw1 = lds128(sb + sw128(r, 2 * w + 1));
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.s_free[b]);
      const uint32_t bb[8] = {bw0.x, bw0.y, bw0.z, bw0.w, bw1.x, bw1.y, bw1.z, bw1.w};
      uint32_t pp[8], pd[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = w * QW + 2 * j;
        const float2 lse = lds64f(sl + col * 4), dd = lds64f(sl + 512 + col * 4);
        const f2 x = fma2(mk2u(sv[2 * j], sv[2 * j + 1]), CQL, fma2(mk2(bf16lo(bb[j]), bf16hi(bb[j])), L2E, mk2(-lse.x, -lse.y)));
        const f2 p = mk2(ex2f(lo2(x)), ex2f(hi2(x)));
        const f2 ds = mul2(p, add2(mk2u(dv[2 * j], dv[2 * j + 1]), mk2(-dd.x, -dd.y)));
        pp[j] = pack_bf16(lo2(p), hi2(p));
        pd[j] = pack_bf16(lo2(ds), hi2(ds));
      }
      if (G >= 2) mbar_wait(&B.ds_free[b], ((G >> 1) - 1) & 1);           // dV / dK of block G - 2 have consumed this P / dS buffer
      tc_fence_after();
      tmem_st8_(trow + T_P + b * 64 + w * 8, pp);
      tmem_st8_(trow + T_P + b * 64 + 32 + w * 8, pd);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ds_full[b]);
      if (n == nb - 1) {
        // ---- the item's epilogue: warpgroups 0-1 write dK (x 1 / sqrt 48), 2-3 write dV; 24 columns each
        int a, k0, head; item_of(li, a, k0, head);
        mbar_wait(&B.acc_full, li & 1);
        tc_fence_after();
        const bool isk = w < 2;
        const float sc = isk ? RSQD : 1.f;
        const int c0 = (w & 1) * 24;
        float* orow = (isk ? DK : DV) + ((size_t)a * L + k0 + r) * DM + head * DH + c0;
        const uint32_t tc = (isk ? T_DK : T_DV) + c0;
        uint32_t v[16], u[8];
        tmem_ld16_(trow + tc, v);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]), "=r"(u[4]), "=r"(u[5]), "=r"(u[6]), "=r"(u[7]) : "r"(trow + tc + 16) : "memory");
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_free);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          *reinterpret_cast<float4*>(orow + 4 * k) = make_float4(__uint_as_float(v[4 * k]) * sc, __uint_as_float(v[4 * k + 1]) * sc,
                                                                 __uint_as_float(v[4 * k + 2]) * sc, __uint_as_float(v[4 * k + 3]) * sc);
#pragma unroll
        for (int k = 0; k < 2; ++k)
          *reinterpret_cast<float4*>(orow + 16 + 4 * k) = make_float4(__uint_as_float(u[4 * k]) * sc, __uint_as_float(u[4 * k + 1]) * sc,
                                                                      __uint_as_float(u[4 * k + 2]) * sc, __uint_as_float(u[4 * k + 3]) * sc);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
