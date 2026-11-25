// dcx_tf32.cu — the tail of the fp32 SWA atom block's few-tile backward in ONE kernel on sm_100a (B200 round 3 part 16): the qkvg
// projections' dx GEMM, the qkvg RMS-adaLN backward, the modulation's dc GEMM and silu' -- four dependent launches before
// (cuBLAS dx = dP W -> adaln_bwd1 -> cuBLAS g Wmod -> mod_dsilu, ~16.7 us at 1024 rows on 16 CTAs each):
//   dy = dP W                    (dP [M, 4C] TF32-valued, W = [Wqkv; Wg] [4C, C] TF32-valued)
//   xh = x rstd;  dxh = dy (1 + scale_q);  dq = dq1 + rstd (dxh - xh mean(dxh xh))          (one augment per atom: A = 1)
//   dmod[:, 0:C] = dy,  dmod[:, C:2C] = dy xh                                              (written; the other columns are inputs)
//   dc = (dmod Wmod) silu'(c),  silu'(c) = s (1 + c (1 - s)),  s = sigmoid(c)               [+ PREV] [+ dq: FOLD]
// One cluster of CL = 8 CTAs per 128-row tile; rank k owns the K slices k of both GEMMs and the rows 16 k .. 16 k + 15 of the tile:
//   1. dy partial = dP[:, 64 k ..] W[64 k ..]  (TMEM, K 64)  -> staged as 8 row blocks, block o sent to rank o (cp.async.bulk shared::cta ->
//      shared::cluster; the H100's DSMEM form, part 15: bulk copies, not per-thread DSMEM stores); rank k sums the 8 blocks of its rows in
//      rank order (deterministic) and runs the adaLN on its 16 rows;
//   2. dc partial (early, needs nothing of step 1) = dmod[:, 2C + 64 k ..] Wmod[2C + 64 k ..] (K 64), and (late) the adaLN's columns
//      dmod[:, 32 k .. 32 k + 31] Wmod[32 k ..] (K 32): every rank sends its 16 rows of those columns (2 KB) to rank o's operand tile
//      (an all-gather by bulk copies) -> the 8 partials of a row block summed in rank order on its owner, then silu'.
// Warps: 0 TMA producer (one box per lane), 1 MMA issuer, 2 TMEM allocator, 4-7 epilogue (thread = tile row = TMEM lane); the row
// stages (adaLN, silu') on all 8 warps, a warp per row, lane = four channels.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace dcx {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
// MN-major operand in the 128-B swizzle with 32-B atoms (dw_tf32.cu): SBO 512, LBO = distance between 32-element MN atoms
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
DEVI void bulk_s2c(uint32_t dst_cl, uint32_t src, uint32_t bytes, uint32_t bar_cl) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst_cl), "r"(src), "r"(bytes), "r"(bar_cl) : "memory");
}
DEVI void st_async(uint32_t dst_cl, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t bar_cl) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(dst_cl), "r"(a), "r"(b), "r"(c), "r"(d), "r"(bar_cl) : "memory");
}
DEVI void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEVI void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
// volatile loads: issued where written (the row stages read inputs long before the exchanges deliver; a plain load was sunk to its use)
DEVI float4 ld4f(const float* p) {
  float4 v;
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
DEVI float4 ld4b(const uint16_t* p) {
  uint2 u;
  asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" : "=r"(u.x), "=r"(u.y) : "l"(p));
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
DEVI float4 ld4(const void* p, size_t o, bool b16) {
  return b16 ? ld4b(reinterpret_cast<const uint16_t*>(p) + o) : ld4f(reinterpret_cast<const float*>(p) + o);
}
DEVI uint32_t bf2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
// store four channels in fp32 or bf16; returns the values as stored (bf16: rounded)
DEVI float4 st4(void* p, size_t o, float4 v, bool b16) {
  if (b16) {
    const uint32_t a = bf2(v.x, v.y), b = bf2(v.z, v.w);
    *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(p) + o) = make_uint2(a, b);
    return make_float4(__uint_as_float(a << 16), __uint_as_float(a & 0xffff0000u), __uint_as_float(b << 16), __uint_as_float(b & 0xffff0000u));
  }
  *reinterpret_cast<float4*>(reinterpret_cast<float*>(p) + o) = v;
  return v;
}
}  // namespace dcx

#ifndef CL
#define CL 8
#endif
#ifndef TRACE
#define TRACE 0
#endif
#ifndef STA
#define STA 0
#endif
#ifndef TRACE_CTA
#define TRACE_CTA 0
#endif
static_assert(CL == 8, "the resident layout below is the 8-CTA cluster's");
constexpr int C = 128, MODW = 6 * C;
constexpr int KD = 4 * C / CL;                 // dx K slice of a rank: 64
constexpr int KE = 4 * C / CL;                 // dc early K slice (modulation columns 2C .. 6C): 64
constexpr int KL = 2 * C / CL;                 // dc late K slice (the adaLN's columns 0 .. 2C): 32
constexpr int RW = 128 / CL;                   // rows a rank owns: 16
constexpr int KB = 128 * 128;                  // a [128 rows][32 fp32] SW128 operand tile: 16 KB
constexpr int BLK = RW * C * 4;                // one exchanged row block of a [128][128] fp32 partial: 8 KB
constexpr int GBLK = RW * 128;                 // one gathered block: RW rows x 32 fp32 = 2 KB
constexpr int O_DXA = 0, O_DXB = O_DXA + (KD / 32) * KB, O_EA = O_DXB + KD * 512, O_EB = O_EA + (KE / 32) * KB;
constexpr int O_LB = O_EB + KE * 512, O_GA = O_LB + KL * 512, O_RCV = O_GA + KB, O_BAR = O_RCV + CL * BLK;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(O_GA % 1024 == 0 && O_EA % 1024 == 0, "align");
static_assert(CL * BLK <= O_EA - O_DXA, "the partial's staging over the dx operands");
static_assert(CL * GBLK <= O_EB - O_EA, "the gather staging over the early dc operands");
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_DC = 0, T_DX = 128;

struct Bars {
  uint64_t fx, fe, fl, dxdone, edone, dcdone, rdx, rg, rdc;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 256, "barriers");

// flags: bit 0 x / dq bf16, bit 1 c bf16, bit 2 dc out bf16, bit 3 FOLD (dc out += dq: the stack's x and c are one tensor)
extern "C" __global__ void __launch_bounds__(256, 1)
swa_dcx_tf32_sm100(const __grid_constant__ CUtensorMap mdp, const __grid_constant__ CUtensorMap mw,
                   const __grid_constant__ CUtensorMap mdm, const __grid_constant__ CUtensorMap mwe,
                   const __grid_constant__ CUtensorMap mwl, const void* __restrict__ X, const float* __restrict__ RES,
                   const float* __restrict__ MOD, void* __restrict__ OUT, float* __restrict__ DMOD, const void* __restrict__ CIN,
                   void* __restrict__ DC, const float* __restrict__ PREV, int M, float eps, int flags
#if TRACE
                   , long long* TRb
#endif
                   ) {
#if TRACE
#define TRS(e, cond) do { if ((int)blockIdx.x < CL && (cond)) TRb[16 * (int)blockIdx.x + (e)] = clock64(); } while (0)
#else
#define TRS(e, cond) do { } while (0)
#endif
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cluster_rank();
  const int tile = (int)blockIdx.x / CL, row0 = tile * 128;
  const bool xb16 = flags & 1, cb16 = flags & 2, ob16 = flags & 4, fold = flags & 8;
  (void)M;

  if (tid == 0) {
    mbar_init(&B.fx, 1); mbar_init(&B.fe, 1); mbar_init(&B.fl, 1);
    mbar_init(&B.dxdone, 1); mbar_init(&B.edone, 1); mbar_init(&B.dcdone, 1);
    mbar_init(&B.rdx, 1); mbar_init(&B.rg, 1); mbar_init(&B.rdc, 1);
    fence_barrier_init();
    mbar_expect_tx(&B.rdx, CL * BLK);          // armed before the cluster barrier: every sender's copy lands on an armed barrier
    mbar_expect_tx(&B.rg, CL * GBLK);
    mbar_expect_tx(&B.rdc, CL * BLK);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  dcx::cl_arrive();                            // the barriers are initialized and armed (waited before the first remote operation)
  TRS(0, tid == 0);
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ---------------------------------------------------------------------------------------- TMA producer: one box per lane
    if (lane == 0) {
      mbar_expect_tx(&B.fx, (KD / 32) * KB + KD * 512);
      mbar_expect_tx(&B.fe, (KE / 32) * KB + KE * 512);
      mbar_expect_tx(&B.fl, KL * 512);
    }
    __syncwarp();
    // the weights first (made before this backward: W's TF32 form, Wmod), then the activations after the previous kernel
    if (lane == 1) tma_load_3d(su + O_DXB, &mw, &B.fx, 0, KD * (int)rank, 0);
    else if (lane == 2) tma_load_3d(su + O_EB, &mwe, &B.fe, 0, 2 * C + KE * (int)rank, 0);
    else if (lane == 3) tma_load_3d(su + O_LB, &mwl, &B.fl, 0, KL * (int)rank, 0);
    else if (lane >= 4 && lane < 4 + KD / 32) {
      const int j = lane - 4;
      pdl_wait();
      tma_load_2d(su + O_DXA + j * KB, &mdp, &B.fx, KD * (int)rank + 32 * j, row0);
    } else if (lane >= 8 && lane < 8 + KE / 32) {
      const int j = lane - 8;
      pdl_wait();
      tma_load_2d(su + O_EA + j * KB, &mdm, &B.fe, 2 * C + KE * (int)rank + 32 * j, row0);
    }
    __syncwarp();
  } else if (warp == 1) {
    // ---------------------------------------------------------------------------------------- MMA issuer
    const uint32_t id = dcx::idesc(128, 128, 0, 1);
    mbar_wait(&B.fx, 0);
    tc_fence_after();
    TRS(1, lane == 0);
    if (elect_one()) {
#pragma unroll
      for (int j = 0; j < KD / 32; ++j)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          dcx::mma(tmem + T_DX, desc_k128(su + O_DXA + j * KB) + (uint64_t)(2 * ks),
                   dcx::desc_mn32b(su + O_DXB + (uint32_t)(32 * j + 8 * ks) * 128u, KD * 128), id, (j | ks) ? 1u : 0u);
      tc_commit(&B.dxdone);
    }
    __syncwarp();
    mbar_wait(&B.fe, 0);
    tc_fence_after();
    TRS(10, lane == 0);
    if (elect_one()) {
#pragma unroll
      for (int j = 0; j < KE / 32; ++j)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          dcx::mma(tmem + T_DC, desc_k128(su + O_EA + j * KB) + (uint64_t)(2 * ks),
                   dcx::desc_mn32b(su + O_EB + (uint32_t)(32 * j + 8 * ks) * 128u, KE * 128), id, (j | ks) ? 1u : 0u);
      tc_commit(&B.edone);
    }
    __syncwarp();
  }

  dcx::cl_wait();                              // (every thread arrived above; the remote barriers are ready from here)
  TRS(12, tid == 0);

  // ------------------------------------------------------------------------------------------ the row stages' inputs (registers)
  pdl_wait();                                  // x / dq1 / mod / c come from the previous kernels
  const int rA = warp, rB = warp + 8;          // the two rows (of this rank's 16) a warp runs
  const int gA = row0 + (int)rank * RW + rA, gB = row0 + (int)rank * RW + rB;
  const size_t oA = (size_t)gA * C + 4 * lane, oB = (size_t)gB * C + 4 * lane;
  const float4 xA = dcx::ld4(X, oA, xb16), xB = dcx::ld4(X, oB, xb16);
  const float4 dA = dcx::ld4f(RES + oA), dB = dcx::ld4f(RES + oB);
  const float4 sA = dcx::ld4f(MOD + (size_t)gA * MODW + C + 4 * lane), sB = dcx::ld4f(MOD + (size_t)gB * MODW + C + 4 * lane);

  // ------------------------------------------------------------------------------------------ dy partial -> row blocks -> owners
  auto send_partial = [&](uint32_t tcol, uint64_t* bar) {
#if STA == 3
    // STA 3: staged by the row threads, then ALL 8 warps push: warp w sends owner (rank + w) % 8 its 16 rows (512-B rows, lane = 16-B
    // chunk, st.async) -- every link of the cluster busy at once (STA 2: 4 warps, owners in the same order on every sender)
    if (warp >= 4) {
      const int lq = warp & 3, r = lq * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16) + tcol;
      const uint32_t dst = su + O_DXA + (uint32_t)(r / RW) * BLK + (uint32_t)(r % RW) * (C * 4);
#pragma unroll 1
      for (int c = 0; c < C; c += 32) {
        uint32_t v[32];
        tmem_ld32(trow + c, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t ch = (uint32_t)(c / 4 + q);
          sts128(dst + ((ch ^ (uint32_t)(r & 31)) << 4), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
        }
      }
      tc_fence_before();
    }
    named_bar_sync(2, 256);
    TRS(tcol == T_DX ? 11 : 14, tid == 0);
    {
      const uint32_t o = (rank + (uint32_t)warp) % CL, rb = dcx::mapa(smem_u32(bar), o);
#pragma unroll 4
      for (int k = 0; k < RW; ++k) {
        const uint32_t pos = (uint32_t)k * (C * 4) + ((uint32_t)lane << 4);
        const uint4 u = lds128(su + O_DXA + o * BLK + pos);
        dcx::st_async(dcx::mapa(su + O_RCV + rank * BLK + pos, o), u.x, u.y, u.z, u.w, rb);
      }
    }
    return;
#elif STA == 2
    // STA 2: staged locally as below, then each warp pushes whole 512-B rows (lane = 16-B chunk) by st.async to the row's owner: one
    // contiguous remote segment per warp instruction (DSMEM wants coalesced segments: per-row-thread 16-B stores ran ~10 B/clk)
    if (warp >= 4) {
      const int lq = warp & 3, r = lq * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16) + tcol;
      const uint32_t dst = su + O_DXA + (uint32_t)(r / RW) * BLK + (uint32_t)(r % RW) * (C * 4);
#pragma unroll 1
      for (int c = 0; c < C; c += 32) {
        uint32_t v[32];
        tmem_ld32(trow + c, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t ch = (uint32_t)(c / 4 + q);
          sts128(dst + ((ch ^ (uint32_t)(r & 31)) << 4), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
        }
      }
      tc_fence_before();
      __syncwarp();                            // a warp pushes the rows it staged itself
      TRS(tcol == T_DX ? 11 : 14, tid == 128);
      const uint32_t rb0 = smem_u32(bar);
#pragma unroll 4
      for (int k = 0; k < 32; ++k) {
        const uint32_t r2 = (uint32_t)(lq * 32 + k), o = r2 / RW, pos = (r2 % RW) * (C * 4) + ((uint32_t)lane << 4);
        const uint4 u = lds128(su + O_DXA + o * BLK + pos);
        dcx::st_async(dcx::mapa(su + O_RCV + rank * BLK + pos, o), u.x, u.y, u.z, u.w, dcx::mapa(rb0, o));
      }
    }
    return;
#elif STA
    // STA: every row thread stores its row of the partial straight into its owner's receive slot (st.async, 16 B a store, completing
    // on the owner's mbarrier): no local staging, no bulk-copy queue
    if (warp >= 4) {
      const int lq = warp & 3, r = lq * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16) + tcol;
      const uint32_t o = (uint32_t)(r / RW);
      const uint32_t dst = dcx::mapa(su + O_RCV + rank * BLK + (uint32_t)(r % RW) * (C * 4), o), rb = dcx::mapa(smem_u32(bar), o);
#pragma unroll 1
      for (int c = 0; c < C; c += 32) {
        uint32_t v[32];
        tmem_ld32(trow + c, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t ch = (uint32_t)(c / 4 + q);
          dcx::st_async(dst + ((ch ^ (uint32_t)(r & 31)) << 4), v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3], rb);
        }
      }
      tc_fence_before();
    }
    return;
#endif
    if (warp >= 4) {
      const int lq = warp & 3, r = lq * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16) + tcol;
      const uint32_t dst = su + O_DXA + (uint32_t)(r / RW) * BLK + (uint32_t)(r % RW) * (C * 4);
#pragma unroll 1
      for (int c = 0; c < C; c += 32) {
        uint32_t v[32];
        tmem_ld32(trow + c, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t ch = (uint32_t)(c / 4 + q);                       // 16-B chunk of the row (0 .. 31)
          sts128(dst + ((ch ^ (uint32_t)(r & 31)) << 4), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
        }
      }
      tc_fence_before();
      fence_proxy_async();                     // the staged blocks are read by the bulk copies (async proxy)
      named_bar_sync(1, 128);
      TRS(tcol == T_DX ? 11 : 14, tid == 128);
      if (tid >= 128 && tid < 128 + CL) {
        const uint32_t o = (uint32_t)(tid - 128);
        dcx::bulk_s2c(dcx::mapa(su + O_RCV + rank * BLK, o), su + O_DXA + o * BLK, BLK, dcx::mapa(smem_u32(bar), o));
      }
    }
  };
  if (warp >= 4) { mbar_wait(&B.dxdone, 0); tc_fence_after(); }
  TRS(2, tid == 128);
  send_partial(T_DX, &B.rdx);

  // the sum of a row's 8 blocks in rank order (lane = 16-B chunk of the row)
  auto row_sum = [&](int rl) {
    const int r = (int)rank * RW + rl;
    const uint32_t off = (uint32_t)rl * (C * 4) + (((uint32_t)lane ^ (uint32_t)(r & 31)) << 4);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < CL; ++s) {
      const uint4 u = lds128(su + O_RCV + (uint32_t)s * BLK + off);
      a.x += __uint_as_float(u.x); a.y += __uint_as_float(u.y); a.z += __uint_as_float(u.z); a.w += __uint_as_float(u.w);
    }
    return a;
  };

  // ------------------------------------------------------------------------------------------ adaLN backward on this rank's rows
  mbar_wait_cl(&B.rdx, 0);
  TRS(3, tid == 0);
  float4 qA, qB;                               // dq as stored (FOLD adds it to dc)
  {
    mbar_wait(&B.edone, 0);                    // the early dc operands are read: their space takes the gather staging
    auto adaln = [&](int rl, float4 q, float4 d2, float4 sc, size_t o, int g) {
      const float4 dy = row_sum(rl);
      TRS(15, tid == 0 && rl == rA);
      const float rstd = 1.f / sqrtf(dcx::warp_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) * (1.f / C) + eps);
      const float4 xh = make_float4(q.x * rstd, q.y * rstd, q.z * rstd, q.w * rstd);
      const float4 dxh = make_float4(dy.x * (1.f + sc.x), dy.y * (1.f + sc.y), dy.z * (1.f + sc.z), dy.w * (1.f + sc.w));
      const float m = dcx::warp_sum(dxh.x * xh.x + dxh.y * xh.y + dxh.z * xh.z + dxh.w * xh.w) * (1.f / C);
      float4 r;
      r.x = d2.x + rstd * (dxh.x - xh.x * m); r.y = d2.y + rstd * (dxh.y - xh.y * m);
      r.z = d2.z + rstd * (dxh.z - xh.z * m); r.w = d2.w + rstd * (dxh.w - xh.w * m);
      const float4 rs = dcx::st4(OUT, o, r, xb16);
      const float4 dsh = make_float4(0.f + dy.x, 0.f + dy.y, 0.f + dy.z, 0.f + dy.w);
      const float4 dsc = make_float4(0.f + dy.x * xh.x, 0.f + dy.y * xh.y, 0.f + dy.z * xh.z, 0.f + dy.w * xh.w);
      const size_t om = (size_t)g * MODW + 4 * lane;
      *reinterpret_cast<float4*>(DMOD + om) = dsh;
      *reinterpret_cast<float4*>(DMOD + om + C) = dsc;
      // the gather staging: block o = this rank's rows of the late columns 32 o .. 32 o + 31 (o < 4: shift, o >= 4: scale), rows in the
      // destination tile's 128-B swizzle (tile row 16 rank + rl: phase rl & 7)
      const uint32_t ch = (uint32_t)(lane & 7), sw = (uint32_t)(rl & 7);
#if STA
      {
        const uint32_t off = O_GA + (rank * RW + (uint32_t)rl) * 128u + ((ch ^ sw) << 4), o1 = (uint32_t)(lane >> 3), o2 = 4u + o1;
        dcx::st_async(dcx::mapa(su + off, o1), __float_as_uint(dsh.x), __float_as_uint(dsh.y), __float_as_uint(dsh.z), __float_as_uint(dsh.w),
                      dcx::mapa(smem_u32(&B.rg), o1));
        dcx::st_async(dcx::mapa(su + off, o2), __float_as_uint(dsc.x), __float_as_uint(dsc.y), __float_as_uint(dsc.z), __float_as_uint(dsc.w),
                      dcx::mapa(smem_u32(&B.rg), o2));
        return rs;
      }
#endif
      const uint32_t base = su + O_EA + (uint32_t)rl * 128u + ((ch ^ sw) << 4);
      sts128(base + (uint32_t)(lane >> 3) * GBLK, make_uint4(__float_as_uint(dsh.x), __float_as_uint(dsh.y), __float_as_uint(dsh.z), __float_as_uint(dsh.w)));
      sts128(base + (uint32_t)(4 + (lane >> 3)) * GBLK,
             make_uint4(__float_as_uint(dsc.x), __float_as_uint(dsc.y), __float_as_uint(dsc.z), __float_as_uint(dsc.w)));
      return rs;
    };
    qA = adaln(rA, xA, dA, sA, oA, gA);
    TRS(13, tid == 0);
    qB = adaln(rB, xB, dB, sB, oB, gB);
  }
  fence_proxy_async();                         // the gather staging is read by the bulk copies; the receive area's reads before its reuse
  __syncthreads();
  TRS(4, tid == 0);
  if (!STA && tid < CL) {
    const uint32_t o = (uint32_t)tid;
    dcx::bulk_s2c(dcx::mapa(su + O_GA + rank * GBLK, o), su + O_EA + o * GBLK, GBLK, dcx::mapa(smem_u32(&B.rg), o));
  }
  if (warp == 1) {                             // the late dc MMAs (the MMA warp also runs rows: it issues them only here)
    mbar_wait(&B.fl, 0);
    mbar_wait_cl(&B.rg, 0);                    // every rank's rows of this rank's late columns have landed in the operand tile
    TRS(5, lane == 0);
    tc_fence_after();
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KL / 8; ++ks)
        dcx::mma(tmem + T_DC, desc_k128(su + O_GA) + (uint64_t)(2 * ks), dcx::desc_mn32b(su + O_LB + (uint32_t)(8 * ks) * 128u, KL * 128),
                 dcx::idesc(128, 128, 0, 1), 1u);
      tc_commit(&B.dcdone);
    }
    __syncwarp();
  }
  // c of this rank's rows (read while the late MMAs run)
  const float4 cA = dcx::ld4(CIN, oA, cb16), cB = dcx::ld4(CIN, oB, cb16);

  // ------------------------------------------------------------------------------------------ dc partial -> row blocks -> owners
  if (warp >= 4) { mbar_wait(&B.dcdone, 0); tc_fence_after(); }
  TRS(6, tid == 128);
  send_partial(T_DC, &B.rdc);
  mbar_wait_cl(&B.rdc, 0);
  TRS(7, tid == 0);
  auto dsilu = [&](int rl, float4 c, float4 q, size_t o) {
    const float4 g = row_sum(rl);
    const float sx = dcx::sigm(c.x), sy = dcx::sigm(c.y), sz = dcx::sigm(c.z), sw = dcx::sigm(c.w);
    float4 d = make_float4(g.x * sx * (1.f + c.x * (1.f - sx)), g.y * sy * (1.f + c.y * (1.f - sy)), g.z * sz * (1.f + c.z * (1.f - sz)),
                           g.w * sw * (1.f + c.w * (1.f - sw)));
    if (PREV != nullptr) {
      const float4 p = dcx::ld4f(PREV + o);
      d = make_float4(p.x + d.x, p.y + d.y, p.z + d.z, p.w + d.w);
    }
    if (fold) d = make_float4(q.x + d.x, q.y + d.y, q.z + d.z, q.w + d.w);
    dcx::st4(DC, o, d, ob16);
  };
  dsilu(rA, cA, qA, oA);
  dsilu(rB, cB, qB, oB);

  // every copy out of this CTA has landed (its receivers waited) before any CTA of the cluster exits
  TRS(8, tid == 0);
  tc_fence_before();
  cluster_sync();
  TRS(9, tid == 0);
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
