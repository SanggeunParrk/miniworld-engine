// tsx_dxn.cu -- the Transition backward at SMALL M, second kernel: d_xn product + LayerNorm backward + residual + dgamma / dbeta
// (-DDIM=<D> -DHID=<H> -DKS=<1 | 2>), sm_100a, bf16, a programmatic dependent launch after tsx_gate.cu.
// One cluster per 128-row tile of D / 64 x KS CTAs; CTA (column block c, K half k):
//   d_xn partial = (k = 0: dA Wa, k = 1: dB Wb; KS = 1: both)   M 128, N 64, K H (2 H) through a TMA ring: A = the tile's [dA | dB]
//          rows (K-major), B = Wa / Wb in their own [H][D] layout (MN-major: no [Wa; Wb]^T copy). Halving K halves the per-SM TMA
//          intake that bounds this kernel; the two halves of a block meet in distributed shared memory (each CTA owns 32 of the 64
//          columns and receives its partner's partial of them), summed dA Wa + dB Wb in that order (deterministic).
//   d_xn = bf16(sum); the LayerNorm backward needs whole-row sums (ca = mean(xhat w), cb = mean(w), w = gamma d_xn): each CTA sums its
//          own columns, the cluster exchanges the per-row pairs through distributed shared memory, adds them in rank order, then
//          dx = bf16(bf16((w - xhat ca - cb) rstd) + dy) -- the tlnbwd_w.cu contract.
//   dgamma / dbeta: per-CTA column sums over its 128 rows (warp reduce-scatter, then the four lane quadrants in order) into a
//          [tiles][2 D] partial; the weight-gradient kernel (tsx_dw.cu, next on the stream) adds the tiles in order.
// Wa / Wb ring stages are loaded before griddepcontrol.wait (forward weights); [dA | dB], x, dy and the statistics after it.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef KS
#define KS 2
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NB = 64, NBLK = D_ / NB, HKB = H_ / 64;
constexpr int CLS = NBLK * KS, NKB = 2 * HKB / KS, OW = NB / KS, NG = OW / 32;   // cluster size, K-blocks per CTA, owned columns
static_assert((KS == 1 || KS == 2) && CLS <= 16, "one cluster per tile");
constexpr int STG = 6;
constexpr int KBT = ROWS * 128, WBT = 64 * 128;                // [dA | dB] K-block [128][64] 16 KB, weight block [64 k][64 cols] 8 KB
constexpr int O_A = 0, O_B = STG * KBT, O_RED = O_B + STG * WBT;  // partner's partial of the owned columns: [128][32] fp32
constexpr int O_RS = O_RED + (KS > 1 ? ROWS * 32 * 4 : 0);     // row sums: [NG column groups][128 rows] float2
constexpr int O_CP = O_RS + NG * ROWS * 8;                     // column partials: [4 lane quadrants][2][OW] fp32
constexpr int O_BAR = O_CP + 4 * 2 * OW * 4, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(128, NB, 0, 1);

struct Bars {
  uint64_t full[STG], empty[STG], acc_full, red_full;
  uint32_t tmem;
};

// warp reduce-scatter: lane l returns the sum over the warp of v[l]
DEVI float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    const bool up = (lane & off) != 0;
#pragma unroll
    for (int k = 0; k < off; ++k) {
      const float lo = v[k], hi = v[k + off];
      v[k] = up ? hi : lo; v[k + off] = up ? lo : hi;
    }
#pragma unroll
    for (int k = 0; k < off; ++k) v[k] += __shfl_xor_sync(0xffffffffu, v[k + off], off);
  }
  return v[0];
}
DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.aligned;" ::: "memory"); }   // release
DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }       // acquire
DEVI uint32_t cl_addr(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
DEVI float2 ld_cluster_f2(uint32_t addr, uint32_t rank) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(cl_addr(addr, rank)) : "memory");
  return v;
}
DEVI void st_cluster_v4(uint32_t addr, uint32_t rank, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(cl_addr(addr, rank)), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// maps: dab [M][2 H] (box 64 x 128); wa, wb [H][D] (box 64 cols x 64 rows, MN-major operand).
// part [tiles][2 D] fp32: this launch's dgamma | dbeta partials per tile.
extern "C" __global__ void __launch_bounds__(384, 1)
transition_sx_dxn(const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mwa,
                  const __grid_constant__ CUtensorMap mwb, const __nv_bfloat16* __restrict__ xg,
                  const __nv_bfloat16* __restrict__ dyg, const float* __restrict__ rstd, const float* __restrict__ c1g,
                  const float* __restrict__ gamma, __nv_bfloat16* __restrict__ dx, float* __restrict__ part) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cluster_rank(), nb = rank % NBLK, ks = rank / NBLK, tile = blockIdx.x / CLS, row0 = tile * ROWS;
  const int own0 = nb * NB + ks * OW;                          // owned output columns [own0, own0 + OW)
  const int kb0 = ks * NKB;                                    // first global K-block (of 2 H / 64: Wa's, then Wb's)

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.acc_full, 1); mbar_init(&B.red_full, 128);
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(&mdab); prefetch_map(&mwa); prefetch_map(&mwb);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), NB); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();                                      // every CTA of the cluster runs before any distributed-shared access
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      for (int kb = 0; kb < STG; ++kb) {                        // the weights first: forward tensors, not the gate's output
        const int g = kb0 + kb;
        mbar_expect_tx(&B.full[kb], KBT + WBT);
        tma_load_2d(su + O_B + kb * WBT, g < HKB ? &mwa : &mwb, &B.full[kb], nb * NB, (g % HKB) * 64);
      }
      pdl_wait();
      for (int kb = 0; kb < STG; ++kb) tma_load_2d(su + O_A + kb * KBT, &mdab, &B.full[kb], (kb0 + kb) * 64, row0);
      for (int kb = STG; kb < NKB; ++kb) {
        const int s = kb % STG, g = kb0 + kb;
        mbar_wait(&B.empty[s], ((kb / STG) - 1) & 1);
        mbar_expect_tx(&B.full[s], KBT + WBT);
        tma_load_2d(su + O_B + s * WBT, g < HKB ? &mwa : &mwb, &B.full[s], nb * NB, (g % HKB) * 64);
        tma_load_2d(su + O_A + s * KBT, &mdab, &B.full[s], g * 64, row0);
      }
    }
    pdl_wait();
    pdl_launch();
  } else if (warp == 1) {
    pdl_wait();
    pdl_launch();
    for (int kb = 0; kb < NKB; ++kb) {
      const int s = kb % STG;
      mbar_wait(&B.full[s], (kb / STG) & 1);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_A + s * KBT), db = desc_mn128(su + O_B + s * WBT, WBT);
      if (elect_one()) {
#pragma unroll
        for (int ks4 = 0; ks4 < 4; ++ks4)
          umma_ss(tmem, da + (uint64_t)(ks4 * 2), db + (uint64_t)((ks4 * 2048) >> 4), IDESC, (kb > 0 || ks4 > 0) ? 1u : 0u);
        tc_commit(&B.empty[s]);
        if (kb == NKB - 1) tc_commit(&B.acc_full);
      }
      __syncwarp();
    }
  } else {
    pdl_wait();
    pdl_launch();
  }

  // ------------------------------------------------------------------------------------------ epilogue: one row per thread, NG
  // groups of 32 owned columns (warps 4-7 the first, 8-11 the second)
  const bool epi = warp >= 4 && warp < 4 + 4 * NG;
  const int r = (warp & 3) * 32 + lane, g = (warp - 4) >> 2, c0 = own0 + g * 32;
  const size_t grow = (size_t)(row0 + r);
  const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
  float rs = 0.f, mean = 0.f;
  uint32_t np[16];                                             // d_xn in bf16, packed pairs
  uint4 xv[4], dyv[4];
  uint32_t v[32];
  if (epi) {
#pragma unroll
    for (int q = 0; q < 4; ++q) { xv[q] = ldg128(xg + grow * D_ + c0 + q * 8); dyv[q] = ldg128(dyg + grow * D_ + c0 + q * 8); }
    rs = rstd[grow];
    mean = c1g[grow] / rs;
    mbar_wait(&B.acc_full, 0);
    tc_fence_after();
    if (KS > 1) {                                              // the partner's half of this block -> its shared memory
      tmem_ld32(trow + (1 - ks) * OW, v);
      tmem_wait_ld();
      const uint32_t dst = su + O_RED + (uint32_t)(r * 128);
      const uint32_t partner = (uint32_t)(ks ? rank - NBLK : rank + NBLK);
#pragma unroll
      for (int i = 0; i < 8; ++i) st_cluster_v4(dst + 16 * i, partner, v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
      mbar_arrive_remote(&B.red_full, partner);                // release (cluster scope): the stores above are visible first
    }
    tmem_ld32(trow + ks * OW + g * 32, v);
    tmem_wait_ld();
    tc_fence_before();
  }
  float cg[32], cb[32];
  if (epi) {
    if (KS > 1) {
      mbar_wait_cl(&B.red_full, 0);                            // the partner's partial is here
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 t = lds128(su + O_RED + (uint32_t)(r * 128 + 16 * i));
        const uint32_t p4[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {                          // dA Wa + dB Wb, in that order
          const float mine = __uint_as_float(v[4 * i + e]), other = __uint_as_float(p4[e]);
          v[4 * i + e] = __float_as_uint(ks == 0 ? mine + other : other + mine);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) np[k] = pack_bf16(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
    float pa = 0.f, pb = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t x4[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w};
      const float4 ga = __ldg(reinterpret_cast<const float4*>(gamma + c0 + q * 8));
      const float4 gc = __ldg(reinterpret_cast<const float4*>(gamma + c0 + q * 8 + 4));
      const float g8[8] = {ga.x, ga.y, ga.z, ga.w, gc.x, gc.y, gc.z, gc.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 8 * q + 2 * e;
        const float n0 = bf16lo(np[i / 2]), n1 = bf16hi(np[i / 2]);
        const float x0 = (bf16lo(x4[e]) - mean) * rs, x1 = (bf16hi(x4[e]) - mean) * rs;
        const float w0 = g8[2 * e] * n0, w1 = g8[2 * e + 1] * n1;
        pa += x0 * w0 + x1 * w1; pb += w0 + w1;
        cg[i] = n0 * x0; cg[i + 1] = n1 * x1; cb[i] = n0; cb[i + 1] = n1;
      }
    }
    reinterpret_cast<float2*>(sm + O_RS)[g * ROWS + r] = make_float2(pa, pb);
  }
  cluster_arrive();                                            // this CTA's row sums are published; the column sums run meanwhile
  if (epi) {
    // dgamma | dbeta column partials: sum over the warp's 32 rows (lane l -> owned column g 32 + l), then the quadrants
    const float sg = reduce_scatter32(cg, lane), sb = reduce_scatter32(cb, lane);
    float* cp = reinterpret_cast<float*>(sm + O_CP);
    cp[((warp & 3) * 2 + 0) * OW + g * 32 + lane] = sg;
    cp[((warp & 3) * 2 + 1) * OW + g * 32 + lane] = sb;
    named_bar_sync(1, 128 * NG);
    if (tid < 128 + 2 * OW) {                                  // this tile's partial of the owned columns
      const int k = tid - 128, which = k / OW, c = k % OW;
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) s += cp[(q * 2 + which) * OW + c];
      part[(size_t)tile * 2 * D_ + which * D_ + own0 + c] = s;
    }
  }
  cluster_wait();                                              // every CTA's row sums are in its shared memory
  float ca = 0.f, cbv = 0.f;
  if (epi) {
    float pa = 0.f, pb = 0.f;
#pragma unroll
    for (int q = 0; q < CLS; ++q)
#pragma unroll
      for (int h = 0; h < NG; ++h) {
        const float2 t = ld_cluster_f2(su + O_RS + (h * ROWS + r) * 8, q);
        pa += t.x; pb += t.y;
      }
    ca = pa * (1.f / D_); cbv = pb * (1.f / D_);
  }
  cluster_arrive();                                            // done reading the peers' row sums
  if (epi) {
    uint32_t o[16];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t x4[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w}, d4[4] = {dyv[q].x, dyv[q].y, dyv[q].z, dyv[q].w};
      const float4 ga = __ldg(reinterpret_cast<const float4*>(gamma + c0 + q * 8));
      const float4 gc = __ldg(reinterpret_cast<const float4*>(gamma + c0 + q * 8 + 4));
      const float g8[8] = {ga.x, ga.y, ga.z, ga.w, gc.x, gc.y, gc.z, gc.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 8 * q + 2 * e;
        const float x0 = (bf16lo(x4[e]) - mean) * rs, x1 = (bf16hi(x4[e]) - mean) * rs;
        const float w0 = g8[2 * e] * bf16lo(np[i / 2]), w1 = g8[2 * e + 1] * bf16hi(np[i / 2]);
        const uint32_t t = pack_bf16((w0 - (x0 * ca + cbv)) * rs, (w1 - (x1 * ca + cbv)) * rs);
        o[4 * q + e] = pack_bf16(bf16lo(t) + bf16lo(d4[e]), bf16hi(t) + bf16hi(d4[e]));
      }
    }
    store_t(dx + (size_t)row0 * D_ + c0 + store_t_off<16>(r, lane, D_, 2), D_, o, lane);
  }
  cluster_wait();                                              // no CTA leaves while a peer may still read its row sums
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, NB); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
