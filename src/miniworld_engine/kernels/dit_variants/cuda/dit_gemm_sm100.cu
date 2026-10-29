// dit_gemm_sm100.cu -- the token DiT blocks' GEMMs with their row passes in the epilogue, sm_100a, bf16 operands, fp32 accumulation.
// Two kernels, one per build (-DGLU or -DRES):
//
//   dit_glu_gemm_sm100  (-DGLU -DNOPS=<2|3> -DUNITS=<64|128> -DKDIM=<K> -DHID=<H>)
//       h = bf16(silu(a) * b [* u]), [a | b | u] = X [Wa; Wb; Wu]^T: the transition's expand GEMM with the (up-)GLU in its epilogue
//       (NOPS = 3: Boltz-2's up-gate). 2-CTA pairs (cta_group::2), persistent over (256-row tile pair, UNITS-unit hidden chunk)
//       items dealt round-robin over the pairs. The pair's B operand is the chunk's NOPS x UNITS weight rows, split by N: the leader
//       streams the first half, its peer the second (32-row TMA boxes), so each SM takes in its 128 X rows and half the weights.
//       Accumulators [a | b | u] double-buffered in tensor memory; two epilogue warpgroups (half the units each) stage h in shared
//       memory for TMA stores.
//
//   dit_res_gemm_sm100  (-DRES -DCL=8 -DK1=<K1> [-DK2=<K2>])
//       x += g1[tok] (A1 W1^T) [+ g2[tok] (A2 W2^T)]  (x fp32 [M, 768], g bf16 tables -- sigmoid values, or logits the epilogue takes the
//       sigmoid of -- tok = row % L), then either out = bf16(x) (a stack's last residual: x is not written back), or x written back
//       and, for nout = 1 / 2, out_i = LN(x) * f(ms_i[tok]) + mb_i[tok]: the residual + gate row pass and the next part's AdaLN in the
//       epilogue of the to_out / squeeze GEMM (the DiT variants' parallel blocks: both GEMMs into two accumulators).
//       One CL-CTA cluster per 128-row tile, a CTA per 768 / CL columns (1-CTA tcgen05.mma, M 128 x N 96 at CL 8); each CTA TMA-loads
//       16 of the tile's A rows and multicasts them to the cluster. The x tile and the gate tiles come in by TMA during the main loop
//       (by cp.async, row by row, where a tile's table rows wrap around L), the AdaLN tables into the ring once it is done; one row per
//       epilogue thread; the row's LayerNorm partials (mean, M2 of each CTA's columns) meet over distributed shared memory (st.async)
//       and are combined (Chan); x and the outputs leave by TMA stores.
//
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace {
DEVI uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
// 8 bytes into a CTA of the cluster, completing their transaction on that CTA's mbarrier (no release fence: a release at cluster scope
// waited for the thread's cp.async copies still in flight, microseconds)
DEVI void st_async_f2(uint32_t a, float x, float y, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               :: "r"(a), "f"(x), "f"(y), "r"(bar) : "memory");
}
// the epilogues' sigmoid: ex2 on MUFU, the reciprocal by Newton steps on the FMA pipe (one MUFU op an element instead of two)
DEVI float sig(float v) { return sigmoid_nr(v); }
}  // namespace

// =================================================================================================================== GLU
#if defined(GLU)
#ifndef NOPS
#define NOPS 2
#endif
#ifndef UNITS
#define UNITS 64
#endif
#ifndef KDIM
#define KDIM 768
#endif
#ifndef HID
#define HID 1536
#endif
constexpr int U = UNITS, NOP = NOPS, N = NOP * U, NH = N / 2, NKB = KDIM / 64, NCH = HID / U;
static_assert(N <= 256 && N % 32 == 0 && NH % 32 == 0 && HID % U == 0 && KDIM % 64 == 0, "GLU geometry");
constexpr int A_BYTES = 128 * 128;                   // 128 rows x 64 bf16 (one 128-B swizzled K-block)
constexpr int B_BYTES = NH * 128;
constexpr int STAGE = A_BYTES + B_BYTES;
constexpr int NST = N == 256 ? 5 : 6;
constexpr int OUT_BYTES = 128 * U * 2;               // h staging: [128 rows][U units] as U / 64 tiles of [128][64]
constexpr int O_OUT = NST * STAGE, O_BAR = O_OUT + 2 * OUT_BYTES;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(256, N);
constexpr uint32_t TCOLS = 2 * N <= 256 ? 256 : 512;

struct Bars {
  uint64_t full[NST], empty[NST], acc_full[2], acc_empty[2], staged[2], stage_free[2];
  uint32_t tmem;
};

// maps: x [M][K] (box 64 x 128); w [NOPS H][K] (box 64 x 32); h [M][H] (box 64 x 128)
extern "C" __global__ void __launch_bounds__(384, 1)
dit_glu_gemm_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw,
                   const __grid_constant__ CUtensorMap mh, int M) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  const int tiles = (M + 127) / 128, ntp = (tiles + 1) / 2;
  const int npairs = gridDim.x >> 1, pair = blockIdx.x >> 1, ntot = ntp * NCH;
  const int nitem = ntot > pair ? (ntot - pair + npairs - 1) / npairs : 0;
  auto item_j = [&](int q) { return (pair + q * npairs) % NCH; };
  auto item_tile = [&](int q) { return 2 * ((pair + q * npairs) / NCH) + crank; };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.acc_full[s], 1); mbar_init(&B.acc_empty[s], 16);   // the 8 epilogue warps of both CTAs
      mbar_init(&B.staged[s], 8); mbar_init(&B.stage_free[s], 1);
    }
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mw); prefetch_map(&mh);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), TCOLS); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef PDL
  pdl_wait();                                                  // x comes from the previous kernel
#endif
  if (warp == 0) {
    if (lane == 0) {
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int j = item_j(q), t = item_tile(q);
        const int row = (t < tiles ? t : tiles - 1) * 128;           // a CTA past the last tile reloads it (its rows are never stored)
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NST;
          if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * STAGE);
          const uint32_t base = su + s * STAGE;
          tma_load_2d_2sm(base, &mx, &B.full[s], kb * 64, row);
#pragma unroll
          for (int b = 0; b < NH / 32; ++b) {
            const int v = crank * NH + 32 * b;                       // virtual row of the pair's [a_j | b_j | u_j] operand
            tma_load_2d_2sm(base + A_BYTES + b * 4096, &mw, &B.full[s], kb * 64, (v / U) * HID + j * U + v % U);
          }
        }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int s = q & 1, u = q >> 1;
        if (q >= 2) mbar_wait_cl(&B.acc_empty[s], (u - 1) & 1);
        tc_fence_after();
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + sg * STAGE;
          const uint64_t da = desc_k128(base), db = desc_k128(base + A_BYTES);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2(tmem + s * N, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) tc_commit2_mc(&B.acc_full[s], 3);
          }
          __syncwarp();
        }
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {                                                  // the store thread
      for (int q = 0; q < nitem; ++q) {
        const int b = q & 1;
        mbar_wait(&B.staged[b], (q >> 1) & 1);
        const int t = item_tile(q);
        if (t < tiles) {
#pragma unroll
          for (int h = 0; h < U / 64; ++h) tma_store_2d(&mh, su + O_OUT + b * OUT_BYTES + h * 16384, item_j(q) * U + h * 64, t * 128);
          tma_store_commit();
          tma_store_wait_read0();
        }
        mbar_arrive(&B.stage_free[b]);
      }
    }
  } else if (warp >= 4) {
    // epilogue: warps 4-7 units [0, U / 2), warps 8-11 units [U / 2, U); TMEM lane quadrant = warp % 4
    const int quad = warp & 3, half = (warp - 4) >> 2;
    const uint32_t r = 32 * quad + lane, trow = tmem + ((uint32_t)(32 * quad) << 16);
    for (int q = 0; q < nitem; ++q) {
      const int s = q & 1, u = q >> 1, b = q & 1;
      mbar_wait(&B.acc_full[s], u & 1);
#ifdef PDL
      if (q == nitem - 1) pdl_launch();                        // the next kernel may set up while the last item's epilogue runs
#endif
      tc_fence_after();
      uint32_t hp[U / 4];                                             // this warpgroup's U / 2 units, bf16 pairs
#pragma unroll
      for (int st2 = 0; st2 < U / 64; ++st2) {
        const int c = half * (U / 2) + st2 * 32;
        uint32_t av[32], bv[32];
        tmem_ld32(trow + s * N + c, av);
        tmem_ld32(trow + s * N + U + c, bv);
#if NOPS == 3
        uint32_t uv[32];
        tmem_ld32(trow + s * N + 2 * U + c, uv);
#endif
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          float h0 = a0 * sig(a0) * __uint_as_float(bv[2 * k]), h1 = a1 * sig(a1) * __uint_as_float(bv[2 * k + 1]);
#if NOPS == 3
          h0 *= __uint_as_float(uv[2 * k]); h1 *= __uint_as_float(uv[2 * k + 1]);
#endif
          hp[st2 * 16 + k] = pack_bf16(h0, h1);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
      if (q >= 2) mbar_wait(&B.stage_free[b], ((q >> 1) - 1) & 1);
      // staging: U / 64 tiles of [128][64] (128-B swizzle); this warpgroup's units start at half * U / 2
#pragma unroll
      for (int qq = 0; qq < U / 16; ++qq) {
        const int unit = half * (U / 2) + qq * 8;
        *reinterpret_cast<uint4*>(sm + O_OUT + b * OUT_BYTES + (unit / 64) * 16384 + sw128(r, (unit % 64) / 8)) =
            make_uint4(hp[4 * qq], hp[4 * qq + 1], hp[4 * qq + 2], hp[4 * qq + 3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.staged[b]);
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, TCOLS); }
}
#endif  // GLU

// =================================================================================================================== RES
#if defined(RES)
#ifndef CL
#define CL 8
#endif
#ifndef K1
#define K1 768
#endif
#ifndef K2
#define K2 0
#endif
constexpr int D = 768, C = CL, NT = D / C, NS = NT / 32, NKB1 = K1 / 64, NKB2 = K2 / 64, NKB = NKB1 + NKB2;
constexpr bool DUAL = K2 > 0;
static_assert(K1 % 64 == 0 && K2 % 64 == 0 && D % C == 0 && NT % 32 == 0 && NT <= 128, "RES geometry");
constexpr int A_BYTES = 128 * 128, B_BYTES = NT * 128, STAGE = A_BYTES + B_BYTES;
// tiles of this CTA's 128 rows x NT columns as 32-column slabs, TMA box layouts: fp32 in the 128-B swizzle (16 KB a slab), bf16 in the
// 64-B one (8 KB)
constexpr int XSLAB = 16384, GSLAB = 8192, XT = NS * XSLAB, GT = NS * GSLAB;
constexpr int O_X = 0, O_G1 = O_X + XT, O_G2 = O_G1 + GT, O_RING = (DUAL ? O_G2 + GT : O_G2);   // x and the gates: loaded during the main loop
constexpr int O_STAT_BUDGET = C * 128 * 8 + 512 + 256;
constexpr int NST = (232448 - O_STAT_BUDGET - O_RING) / STAGE;
constexpr int RING = NST * STAGE;
constexpr int O_T = O_RING;                                    // after the main loop: ms0 | mb0 | ms1 | mb1 tiles in the ring
static_assert(NST >= 4 && 4 * GT <= RING, "RES shared memory");
constexpr int O_STAT = O_RING + RING;                          // float2 [C column groups][128 rows]
constexpr int O_TOK = O_STAT + C * 128 * 8;                    // int [128]: the table row of every tile row
constexpr int O_BAR = O_TOK + 512;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448 && O_RING % 1024 == 0, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(128, NT);
constexpr uint32_t TCOLS = 2 * NT <= 128 ? 128 : 256;
constexpr int HALF = NT / 2;                                   // columns per epilogue thread (two warpgroups split NT)
constexpr uint16_t CMASK = (uint16_t)((1u << C) - 1);

struct Bars {
  uint64_t full[NST], empty[NST], acc_full, stat_full, xg_full, tab_full;
  uint32_t tmem;
};

struct ResArgs {
  float* x;                    // [M, 768] fp32 residual (read; written back unless `fin`)
  const __nv_bfloat16* g1;     // gate logits [L, *] (row stride sg), sigmoid applied here
  const __nv_bfloat16* g2;
  const __nv_bfloat16* ms[2];  // AdaLN scale / shift tables [L, *] (row stride sms)
  const __nv_bfloat16* mb[2];
  long sg, sms;
  int M, L, nout, sig, fin;    // fin: write the stack's output (bf16 x, map mo0) instead of x and the LayerNorm outputs
  float eps;
  int sig_g;                   // the gate tables hold logits (sigmoid applied here) rather than sigmoid values
};

namespace {
DEVI void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
DEVI void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
// byte offset of (row r, column c) in a slab tile: fp32 (ESZ 4) 128-B swizzle, bf16 (ESZ 2) 64-B swizzle; c a multiple of 16 / ESZ
template <int ESZ>
DEVI uint32_t toff(int r, int c) {
  if constexpr (ESZ == 4) return (uint32_t)((c / 32) * XSLAB + r * 128 + ((((c % 32) / 4) ^ (r & 7)) << 4));
  else return (uint32_t)((c / 32) * GSLAB + r * 64 + ((((c % 32) / 8) ^ ((r >> 1) & 3)) << 4));
}
// a tile whose rows wrap around the token table (row0 % L + rows > L): rows by 16-B cp.async, consecutive threads on consecutive
// chunks of a row (coalesced), each tile row from table row tok[r]
DEVI void tile_load_rows(uint32_t dst, const __nv_bfloat16* base, long row_stride, int col0, const int* tok, int rows, int et) {
  constexpr int CPR = NT * 2 / 16;
  for (int e = et; e < rows * CPR; e += 256) {
    const int r = e / CPR, c = (e - r * CPR) * 8;
    cp_async16(dst + toff<2>(r, c), base + (long)tok[r] * row_stride + col0 + c);
  }
}
}  // namespace

// maps: a1 [M][K1], a2 [M][K2] (box 64 x 16: each CTA loads 16 rows and multicasts them to the cluster); w1 [768][K1], w2 [768][K2]
// (box 64 x NT); mx: x fp32 [M][768] (box 32 x 128, 128-B swizzle); mo0 / mo1: the bf16 outputs [M][768]; mg1, mg2, mt[4]: the gate and
// AdaLN tables [L][768] (box 32 x 128, 64-B swizzle; a table a call does not use gets any map)
extern "C" __global__ void __launch_bounds__(384, 1)
dit_res_gemm_sm100(const __grid_constant__ CUtensorMap ma1, const __grid_constant__ CUtensorMap mw1,
                   const __grid_constant__ CUtensorMap ma2, const __grid_constant__ CUtensorMap mw2,
                   const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mo0,
                   const __grid_constant__ CUtensorMap mo1, const __grid_constant__ CUtensorMap mg1,
                   const __grid_constant__ CUtensorMap mg2, const __grid_constant__ CUtensorMap mt0,
                   const __grid_constant__ CUtensorMap mt1, const __grid_constant__ CUtensorMap mt2,
                   const __grid_constant__ CUtensorMap mt3, const ResArgs p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  float2* stat = reinterpret_cast<float2*>(sm + O_STAT);
  int* stok = reinterpret_cast<int*>(sm + O_TOK);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cr = (int)cluster_rank();                          // the column group: columns [cr NT, (cr + 1) NT)
  const int row0 = (blockIdx.x / C) * 128;
  const int col0 = cr * NT;
  const int M = p.M;
  const int rows = M - row0 < 128 ? M - row0 : 128;
  const int tok0 = row0 % p.L;
  const bool wrap = tok0 + rows > p.L;                         // the tile's table rows are not one contiguous range: cp.async
  const bool ln = p.nout > 0 && !p.fin;

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], C); }   // a stage is free once every CTA's MMAs read it
    mbar_init(&B.acc_full, 1);
    mbar_init(&B.stat_full, 1);
    mbar_init(&B.xg_full, 1);
    mbar_init(&B.tab_full, 1);
    fence_barrier_init();
    prefetch_map(&ma1); prefetch_map(&mw1);
    if (DUAL) { prefetch_map(&ma2); prefetch_map(&mw2); }
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  if (wrap && tid >= 128 && tid < 256) stok[tid - 128] = (row0 + tid - 128) % p.L;
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef PDL
  pdl_wait();                                                  // the operands come from the previous kernels
#endif

  if (warp == 0) {
    if (lane == 0) {
      // this CTA's 16 rows of the shared A tile (a slice past M reloads the last valid one: its rows are never stored)
      int arow = row0 + 16 * cr;
      if (arow >= M) arow = M > 16 ? M - 16 : 0;
      for (int kb = 0; kb < NKB; ++kb) {
        const int s = kb % NST;
        if (kb >= NST) mbar_wait(&B.empty[s], ((kb / NST) - 1) & 1);
        mbar_expect_tx(&B.full[s], STAGE);
        const uint32_t base = su + O_RING + s * STAGE;
        const bool second = DUAL && kb >= NKB1;
        const int kk = second ? kb - NKB1 : kb;
        tma_load_2d_mc(base + 16 * cr * 128, second ? &ma2 : &ma1, &B.full[s], kk * 64, arow, CMASK);
        tma_load_2d(base + A_BYTES, second ? &mw2 : &mw1, &B.full[s], kk * 64, col0);
      }
    }
  } else if (warp == 1) {
    for (int kb = 0; kb < NKB; ++kb) {
      const int s = kb % NST;
      mbar_wait(&B.full[s], (kb / NST) & 1);
      tc_fence_after();
      const uint32_t base = su + O_RING + s * STAGE;
      const uint64_t da = desc_k128(base), db = desc_k128(base + A_BYTES);
      const bool second = DUAL && kb >= NKB1;
      const bool first_kb = kb == 0 || kb == NKB1;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ss(tmem + (second ? (uint32_t)NT : 0u), da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (!first_kb || ks > 0) ? 1u : 0u);
        tc_commit_mc(&B.empty[s], CMASK);
        if (kb == NKB - 1) tc_commit(&B.acc_full);
      }
      __syncwarp();
    }
  } else if (warp == 3) {
    if (lane == 0) {
      // the epilogue's tiles by TMA: x and the gates while the main loop runs, the AdaLN tables into the ring once it is done
      prefetch_map(&mx);
      mbar_expect_tx(&B.xg_full, XT + (wrap ? 0 : (DUAL ? 2 : 1) * GT));
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        tma_load_2d(su + O_X + s * XSLAB, &mx, &B.xg_full, col0 + 32 * s, row0);
        if (!wrap) {
          tma_load_2d(su + O_G1 + s * GSLAB, &mg1, &B.xg_full, col0 + 32 * s, tok0);
          if (DUAL) tma_load_2d(su + O_G2 + s * GSLAB, &mg2, &B.xg_full, col0 + 32 * s, tok0);
        }
      }
      if (ln && !wrap) {
        mbar_wait(&B.acc_full, 0);
        mbar_expect_tx(&B.tab_full, 2 * p.nout * GT);
#pragma unroll
        for (int s = 0; s < NS; ++s) {
          tma_load_2d(su + O_T + s * GSLAB, &mt0, &B.tab_full, col0 + 32 * s, tok0);
          tma_load_2d(su + O_T + GT + s * GSLAB, &mt1, &B.tab_full, col0 + 32 * s, tok0);
          if (p.nout > 1) {
            tma_load_2d(su + O_T + 2 * GT + s * GSLAB, &mt2, &B.tab_full, col0 + 32 * s, tok0);
            tma_load_2d(su + O_T + 3 * GT + s * GSLAB, &mt3, &B.tab_full, col0 + 32 * s, tok0);
          }
        }
      }
    }
  } else if (warp >= 4) {
    // Epilogue (warps 4-11, et = 0..255): one row per thread (its TMEM lane), half the CTA's columns per warpgroup. x += sigmoid(g1)
    // acc1 [+ sigmoid(g2) acc2], kept in registers and in the x tile (TMA-stored); the row's LayerNorm partials (mean, M2 of the CTA's
    // NT columns) go to every CTA of the cluster (st.async); each combines the C partials (Chan) and writes its columns of the
    // outputs, staged over the tables they were made from (TMA stores). Shared memory is read and written through plain C++ accesses
    // (no volatile asm) so the compiler can interleave the rows' arithmetic.
    const int et = tid - 128, quad = warp & 3, wg = (warp - 4) >> 2;
    const int r = 32 * quad + lane, c0 = wg * HALF;
    const uint32_t trow = tmem + ((uint32_t)(32 * quad) << 16);
    if (wrap) {
      tile_load_rows(su + O_G1, p.g1, p.sg, col0, stok, rows, et);
      if (DUAL) tile_load_rows(su + O_G2, p.g2, p.sg, col0, stok, rows, et);
      cp_async_commit();
    }
    mbar_wait(&B.acc_full, 0);
    tc_fence_after();
#ifdef PDL
    pdl_launch();
#endif
    if (wrap) {
      if (ln) {                                                // the tables into the ring the main loop is done with
#pragma unroll
        for (int o = 0; o < 2; ++o) {
          if (o >= p.nout) break;
          tile_load_rows(su + O_T + (2 * o) * GT, o ? p.ms[1] : p.ms[0], p.sms, col0, stok, rows, et);
          tile_load_rows(su + O_T + (2 * o + 1) * GT, o ? p.mb[1] : p.mb[0], p.sms, col0, stok, rows, et);
        }
        cp_async_commit();
        asm volatile("cp.async.wait_group 1;" ::: "memory");   // the gates (the tables may still be in flight)
      } else {
        cp_async_wait_all();
      }
      named_bar_sync(1, 256);
    }
    mbar_wait(&B.xg_full, 0);
    float xv[HALF];
    float s1 = 0.f;
#pragma unroll
    for (int k = 0; k < HALF / 16; ++k) {
      const int c = c0 + 16 * k;
      uint32_t a[16];
      tmem_ld16(trow + c, a);
#if K2 > 0
      uint32_t b2[16];
      tmem_ld16(trow + NT + c, b2);
#endif
      tmem_wait_ld();
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const uint4 g = *reinterpret_cast<const uint4*>(sm + O_G1 + toff<2>(r, c + 8 * v));
        const uint32_t gw[4] = {g.x, g.y, g.z, g.w};
#if K2 > 0
        const uint4 g2 = *reinterpret_cast<const uint4*>(sm + O_G2 + toff<2>(r, c + 8 * v));
        const uint32_t gw2[4] = {g2.x, g2.y, g2.z, g2.w};
#endif
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float4* xp = reinterpret_cast<float4*>(sm + O_X + toff<4>(r, c + 8 * v + 4 * h));
          const float4 xo = *xp;
          float t[4] = {xo.x, xo.y, xo.z, xo.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = 8 * v + 4 * h + e;                   // element of this 16-column chunk
            const uint32_t w = gw[2 * h + e / 2];
            const float gl = (e & 1) ? bf16hi(w) : bf16lo(w);
            t[e] += (p.sig_g ? sig(gl) : gl) * __uint_as_float(a[i]);
#if K2 > 0
            const uint32_t w2 = gw2[2 * h + e / 2];
            const float gl2 = (e & 1) ? bf16hi(w2) : bf16lo(w2);
            t[e] += (p.sig_g ? sig(gl2) : gl2) * __uint_as_float(b2[i]);
#endif
            xv[16 * k + i] = t[e];
            s1 += t[e];
          }
          if (!p.fin) *xp = make_float4(t[0], t[1], t[2], t[3]);
        }
      }
    }
    if (p.fin) {
      // the stack's output: bf16 x, staged in the first gate tile (this thread's own row and columns)
#pragma unroll
      for (int c = 0; c < HALF / 8; ++c)
        *reinterpret_cast<uint4*>(sm + O_G1 + toff<2>(r, c0 + 8 * c)) =
            make_uint4(pack_bf16(xv[8 * c], xv[8 * c + 1]), pack_bf16(xv[8 * c + 2], xv[8 * c + 3]),
                       pack_bf16(xv[8 * c + 4], xv[8 * c + 5]), pack_bf16(xv[8 * c + 6], xv[8 * c + 7]));
      fence_proxy_async();
      named_bar_sync(1, 256);
      if (et == 0) {
#pragma unroll
        for (int s = 0; s < NS; ++s) tma_store_2d(&mo0, su + O_G1 + s * GSLAB, col0 + 32 * s, row0);
        tma_store_commit();
        tma_store_wait_read0();
      }
    } else {
      fence_proxy_async();
      named_bar_sync(1, 256);                                  // the x tile is complete
      if (et == 0) {
#pragma unroll
        for (int s = 0; s < NS; ++s) tma_store_2d(&mx, su + O_X + s * XSLAB, col0 + 32 * s, row0);
        tma_store_commit();
      }
      if (ln) {
        // the row's partial over this warpgroup's columns; the two warpgroups' partials (n = HALF each) meet in this CTA's own stat
        // slot (no other CTA writes it): warpgroup 1 parks its partial there, warpgroup 0 combines and sends the row's partial
        // (n = NT) to every CTA of the cluster
        const float m_t = s1 * (1.f / (float)HALF);
        float q_t = 0.f;
#pragma unroll
        for (int i = 0; i < HALF; ++i) { const float d = xv[i] - m_t; q_t = fmaf(d, d, q_t); }
        if (wg == 1) stat[cr * 128 + r] = make_float2(m_t, q_t);
        if (et == 0) mbar_expect_tx(&B.stat_full, C * 128 * 8);
        named_bar_sync(1, 256);
        if (wg == 0) {
          const float2 o1 = stat[cr * 128 + r];
          const float mm = 0.5f * (m_t + o1.x), d0 = m_t - mm, d1 = o1.x - mm;
          const float qq = q_t + o1.y + (float)HALF * (d0 * d0 + d1 * d1);
          const uint32_t sa = smem_u32(&stat[cr * 128 + r]), ba = smem_u32(&B.stat_full);
#pragma unroll
          for (int c = 0; c < C; ++c) st_async_f2(mapa(sa, (uint32_t)c), mm, qq, mapa(ba, (uint32_t)c));
        }
        mbar_wait_cl(&B.stat_full, 0);
        float mean = 0.f;
#pragma unroll
        for (int i = 0; i < C; ++i) mean += stat[i * 128 + r].x;
        mean *= 1.f / (float)C;
        float m2 = 0.f;
#pragma unroll
        for (int i = 0; i < C; ++i) { const float2 v = stat[i * 128 + r]; const float d = v.x - mean; m2 += v.y + (float)NT * d * d; }
        const float rstd = rsqrtf(m2 * (1.f / (float)D) + p.eps);
#pragma unroll
        for (int i = 0; i < HALF; ++i) xv[i] = (xv[i] - mean) * rstd;
        if (wrap) {
          cp_async_wait_all();
          named_bar_sync(1, 256);                              // every thread's tables have landed
        } else {
          mbar_wait(&B.tab_full, 0);
        }
#pragma unroll
        for (int o = 0; o < 2; ++o) {
          if (o >= p.nout) break;
          uint8_t* tsc = sm + O_T + (2 * o) * GT;
          const uint8_t* tsh = sm + O_T + (2 * o + 1) * GT;
#pragma unroll
          for (int c = 0; c < HALF / 8; ++c) {
            const uint32_t ad = toff<2>(r, c0 + 8 * c);
            const uint4 sv = *reinterpret_cast<const uint4*>(tsc + ad), hv = *reinterpret_cast<const uint4*>(tsh + ad);
            const uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w}, hw[4] = {hv.x, hv.y, hv.z, hv.w};
            uint32_t ow[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              float s0 = bf16lo(sw[e]), s1v = bf16hi(sw[e]);
              if (p.sig) { s0 = sig(s0); s1v = sig(s1v); }
              ow[e] = pack_bf16(xv[8 * c + 2 * e] * s0 + bf16lo(hw[e]), xv[8 * c + 2 * e + 1] * s1v + bf16hi(hw[e]));
            }
            *reinterpret_cast<uint4*>(tsc + ad) = make_uint4(ow[0], ow[1], ow[2], ow[3]);   // over the scale table (own row)
          }
        }
        fence_proxy_async();
        named_bar_sync(1, 256);
        if (et == 0) {
#pragma unroll
          for (int s = 0; s < NS; ++s) {
            tma_store_2d(&mo0, su + O_T + s * GSLAB, col0 + 32 * s, row0);
            if (p.nout > 1) tma_store_2d(&mo1, su + O_T + 2 * GT + s * GSLAB, col0 + 32 * s, row0);
          }
          tma_store_commit();
        }
      }
      if (et == 0) tma_store_wait_read0();
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
#endif  // RES
