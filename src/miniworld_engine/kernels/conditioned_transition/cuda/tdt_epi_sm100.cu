// tdt_epi_sm100.cu -- the token DiT training block's few-row GEMMs on sm_100a (tcgen05, TMEM, TMA) with the neighbouring row pass
// in the epilogue, so the chain loses that pass's launch (one-sample training: M = 128 - 1024 rows, every chain kernel 2-9 us).
//
//   C[M, N] = A[M, K] op(B), bf16 operands, fp32 accumulation in TMEM. A cluster of KS CTAs shares one 128 x BN output tile, CTA s
//   summing K-blocks [s nk / KS, (s + 1) nk / KS) (at M = 128 a tile per CTA leaves most SMs idle and each one latency bound on its
//   K loop); the fp32 partial tiles meet in distributed shared memory, CTA r reduces rows [r 128 / KS, (r + 1) 128 / KS) in rank
//   order (bitwise repeatable) and applies the epilogue to them (coalesced: a thread owns 4 columns of a row).
//   A  [M, K] bf16, K contiguous                                   TMA box [64 (K), 128 (M)], 128-B swizzle
//   B  TRB 0: [N, K] bf16 (an nn.Linear weight as stored)          TMA box [64 (K), BN (N)] (EPI 3: two boxes of BN / 2 rows)
//      TRB 1: [K, N] bf16 (dX = dY W)                              TMA box [64 (N), 64 (K)], MN-major operand, BN a multiple of 64
//
// Epilogues (EPI); every value is the one the replaced cuBLAS GEMM + row kernel pair computes (the GEMM output rounded to bf16
// first, the row kernel's fp32 arithmetic):
//   0  plain: out = acc (+ bias), bf16 or fp32 (OUTF32)
//   1  res_c (forward):        z = bf16(acc); out = x1 + sigmoid(Gg[:, D + n] + bg2) z          (writes z and out)
//   2  swiglu_bwd (backward):  dh = bf16(acc); a, b from ab: dab = [dh b s (1 + a (1 - s)) | dh a s], h = a s b
//   3  swiglu (forward):       the tile is BN / 2 columns of a (rows of Wa) and the same of b (rows of Wb); a, b = bf16(acc):
//                              ab = [a | b], h = a s b                                           (s = sigmoid(a))
//
// Roles (192 threads): warp 0 TMA producer, warp 1 TMEM owner + MMA issuer, warps 2-5 drain the accumulator into the owners'
// staging rows (remote stores); all six warps reduce. PDL: barriers, TMEM, descriptor prefetch and the first NST weight boxes before
// griddepcontrol.wait (the weights do not depend on the previous kernel), the A loads after it.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef BN
#define BN 64
#endif
#ifndef TRB
#define TRB 0
#endif
#ifndef NST
#define NST 4
#endif
#ifndef EPI
#define EPI 0
#endif
#ifndef PDL
#define PDL 1
#endif
#ifndef OUTF32
#define OUTF32 0
#endif
#ifndef OUT_BF16
#define OUT_BF16 1      // EPI 1: the block output dtype (bf16 or fp32)
#endif

constexpr int BM = 128, BK = 64;
constexpr int A_BYTES = BM * BK * 2, B_BYTES = BN * BK * 2, STAGE = A_BYTES + B_BYTES;
constexpr int RLD = BN + 4;                                       // fp32 staging row stride (floats)
constexpr int O_RED = NST * STAGE, O_BAR = O_RED + BM * RLD * 4;
constexpr int SMEM_BYTES = 1024 + O_BAR + 256;
static_assert(STAGE % 1024 == 0 && SMEM_BYTES <= 232448, "shared memory budget");
static_assert(!TRB || BN % 64 == 0, "MN-major B: 64-column boxes");
static_assert(BN % 32 == 0 && BN >= 32 && BN <= 256, "tile width");
static_assert(EPI != 3 || (TRB == 0 && BN % 16 == 0), "swiglu forward: two K-major halves");
constexpr uint32_t IDESC = idesc_bf16(BM, BN, 0, TRB);
constexpr int TCOLS = BN <= 32 ? 32 : BN <= 64 ? 64 : BN <= 128 ? 128 : 256;

DEVI float sigm(float v) { return 1.f / (1.f + __expf(-v)); }
DEVI float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
DEVI uint32_t peer_addr(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
DEVI void st_cluster4(uint32_t addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared::cluster.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
DEVI uint2 bf4(float a, float b, float c, float d) { return make_uint2(pack_bf16(a, b), pack_bf16(c, d)); }
DEVI float4 ld_bf4(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y));
}

struct EpiArgs {
  void* out;          // EPI 0: C; 1: block output; 2: dab; 3: ab
  long ldo;           // row stride (elements) of out
  void* aux;          // EPI 1: z (bf16); 2: h (bf16); 3: h (bf16)
  long lda;
  const void* in0;    // EPI 0: bias (fp32) or null; 1: x1 (fp32); 2: ab (bf16)
  long ld0;
  const void* in1;    // EPI 1: Gg's gate-2 column block (bf16)
  long ld1;
  const float* vec;   // EPI 1: bg2
  int nh;             // EPI 2 / 3: the hidden width n D (b's column offset in ab / dab)
};

extern "C" __global__ void __launch_bounds__(192, 1)
tdt_epi_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, int M, int K, int ks, const EpiArgs p) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* sm = (unsigned char*)(((uintptr_t)raw + 1023) & ~(uintptr_t)1023);
  const uint32_t su = smem_u32(sm);
  float* red = reinterpret_cast<float*>(sm + O_RED);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t* empty = full + NST;
  uint64_t* done = empty + NST;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int rank = blockIdx.x % ks, tile = blockIdx.x / ks, m0 = blockIdx.y * BM;
#if EPI == 3
  const int n0 = tile * (BN / 2);                                  // a / b / h column of the tile
#else
  const int n0 = tile * BN;
#endif
  const int nk = K / BK, kb0 = rank * nk / ks, kb1 = (rank + 1) * nk / ks, nloc = kb1 - kb0;

  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
    mbar_init(done, 1);
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mb);
  }
  if (warp == 1) { tmem_alloc(smem_u32(tslot), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // (waited on before the first remote store)

  if (warp == 0) {                                                 // TMA producer
    auto load_b = [&](int s, int kb) {
        const uint32_t b = su + s * STAGE + A_BYTES;
#if TRB
#pragma unroll
        for (int j = 0; j < BN / 64; ++j) tma_load_2d(b + j * 8192, &mb, &full[s], n0 + 64 * j, kb * BK);
#elif EPI == 3
        tma_load_2d(b, &mb, &full[s], kb * BK, n0);                              // Wa rows
        tma_load_2d(b + (BN / 2) * 128, &mb, &full[s], kb * BK, p.nh + n0);      // Wb rows
#else
        tma_load_2d(b, &mb, &full[s], kb * BK, n0);
#endif
    };
    // (2026-11-17, part 6) the first NST stages issued by NST lanes in parallel: one thread issues about one TMA load per ~600 clk
    // whatever the box size, and a split-K CTA has only a few K-blocks -- its whole operand stream is these first stages
    const int pre = nloc < NST ? nloc : NST;
    if (lane < pre) { mbar_expect_tx(&full[lane], STAGE); load_b(lane, kb0 + lane); }   // weights: before the wait
#if PDL
    pdl_wait();
#endif
    if (lane < pre) tma_load_2d(su + lane * STAGE, &ma, &full[lane], (kb0 + lane) * BK, m0);
    if (lane == 0) {
      for (int i = NST; i < nloc; ++i) {
        const int s = i % NST;
        mbar_wait(&empty[s], ((i / NST) - 1) & 1);
        mbar_expect_tx(&full[s], STAGE);
        load_b(s, kb0 + i);
        tma_load_2d(su + s * STAGE, &ma, &full[s], (kb0 + i) * BK, m0);
      }
    }
    __syncwarp();
  } else if (warp == 1) {                                          // MMA issuer
    for (int i = 0; i < nloc; ++i) {
      const int s = i % NST;
      mbar_wait(&full[s], (i / NST) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t a = su + s * STAGE, b = a + A_BYTES;
#pragma unroll
        for (int k = 0; k < BK / 16; ++k) {
#if TRB
          umma_ss(tmem, desc_k128(a + k * 32), desc_mn128(b + k * 2048, 8192), IDESC, (i | k) != 0);
#else
          umma_ss(tmem, desc_k128(a + k * 32), desc_k128(b + k * 32), IDESC, (i | k) != 0);
#endif
        }
        tc_commit(&empty[s]);
        if (i == nloc - 1) tc_commit(done);
      }
      __syncwarp();
    }
  } else {                                                         // warps 2-5: accumulator rows -> their owner's slice
    const int q = warp % 4, row = q * 32 + lane, rpr = BM / ks, owner = row / rpr;
    const uint32_t dst = peer_addr(smem_u32(red) + (uint32_t)((rank * rpr + row % rpr) * RLD) * 4u, owner);
    mbar_wait(done, 0);
    tc_fence_after();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#pragma unroll
    for (int c = 0; c < BN; c += 32) {
      uint32_t v[32];
      tmem_ld32(tmem + ((uint32_t)(q * 32) << 16) + c, v);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 8; ++j) st_cluster4(dst + (uint32_t)(c + 4 * j) * 4u, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
    }
  }
  if (warp < 2) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#if PDL
  pdl_launch();
#endif
  tc_fence_before();
  cluster_sync();                                                  // every peer's partial rows of this CTA's slice are in
  {
    const int rpr = BM / ks;
#if EPI == 3
    constexpr int CW = BN / 2;                                     // a columns of the tile (b at + CW in the staging row)
#else
    constexpr int CW = BN;
#endif
    const int units = rpr * (CW / 4);
    for (int u = threadIdx.x; u < units; u += 192) {
      const int r = u / (CW / 4), c = (u % (CW / 4)) * 4;
      const long grow = m0 + rank * rpr + r;
      auto sum4 = [&](int col) {
        float4 acc = *reinterpret_cast<const float4*>(red + r * RLD + col);
        for (int pp = 1; pp < ks; ++pp) {                          // the K slices in rank order
          const float4 t = *reinterpret_cast<const float4*>(red + (pp * rpr + r) * RLD + col);
          acc.x += t.x; acc.y += t.y; acc.z += t.z; acc.w += t.w;
        }
        return acc;
      };
      const float4 acc = sum4(c);
      if (grow >= M) continue;
      const int n = n0 + c;
#if EPI == 0
      float4 o = acc;
      const float* bias = reinterpret_cast<const float*>(p.in0);
      if (bias != nullptr) { const float4 b4 = *reinterpret_cast<const float4*>(bias + n); o.x += b4.x; o.y += b4.y; o.z += b4.z; o.w += b4.w; }
#if OUTF32
      *reinterpret_cast<float4*>(reinterpret_cast<float*>(p.out) + grow * p.ldo + n) = o;
#else
      *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(p.out) + grow * p.ldo + n) = bf4(o.x, o.y, o.z, o.w);
#endif
#elif EPI == 1
      const float4 z = make_float4(rbf(acc.x), rbf(acc.y), rbf(acc.z), rbf(acc.w));
      const float4 x1 = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(p.in0) + grow * p.ld0 + n);
      const float4 g = ld_bf4(reinterpret_cast<const __nv_bfloat16*>(p.in1) + grow * p.ld1 + n);
      const float4 bg = *reinterpret_cast<const float4*>(p.vec + n);
      const float4 o = make_float4(x1.x + sigm(g.x + bg.x) * z.x, x1.y + sigm(g.y + bg.y) * z.y, x1.z + sigm(g.z + bg.z) * z.z,
                                   x1.w + sigm(g.w + bg.w) * z.w);
      *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(p.aux) + grow * p.lda + n) = bf4(z.x, z.y, z.z, z.w);
#if OUT_BF16
      *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(p.out) + grow * p.ldo + n) = bf4(o.x, o.y, o.z, o.w);
#else
      *reinterpret_cast<float4*>(reinterpret_cast<float*>(p.out) + grow * p.ldo + n) = o;
#endif
#elif EPI == 2
      const float dh[4] = {rbf(acc.x), rbf(acc.y), rbf(acc.z), rbf(acc.w)};
      const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(p.in0) + grow * p.ld0 + n;
      const float4 a4 = ld_bf4(ab), b4 = ld_bf4(ab + p.nh);
      const float a[4] = {a4.x, a4.y, a4.z, a4.w}, b[4] = {b4.x, b4.y, b4.z, b4.w};
      float da[4], db[4], hh[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const float s = sigm(a[t]), as = a[t] * s;
        da[t] = dh[t] * b[t] * s * (1.f + a[t] * (1.f - s));
        db[t] = dh[t] * as;
        hh[t] = as * b[t];
      }
      __nv_bfloat16* dab = reinterpret_cast<__nv_bfloat16*>(p.out) + grow * p.ldo + n;
      *reinterpret_cast<uint2*>(dab) = bf4(da[0], da[1], da[2], da[3]);
      *reinterpret_cast<uint2*>(dab + p.nh) = bf4(db[0], db[1], db[2], db[3]);
      *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(p.aux) + grow * p.lda + n) = bf4(hh[0], hh[1], hh[2], hh[3]);
#elif EPI == 3
      const float4 bacc = sum4(c + CW);
      const float a[4] = {rbf(acc.x), rbf(acc.y), rbf(acc.z), rbf(acc.w)}, b[4] = {rbf(bacc.x), rbf(bacc.y), rbf(bacc.z), rbf(bacc.w)};
      float hh[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) hh[t] = a[t] * sigm(a[t]) * b[t];
      __nv_bfloat16* ab = reinterpret_cast<__nv_bfloat16*>(p.out) + grow * p.ldo + n;
      *reinterpret_cast<uint2*>(ab) = bf4(a[0], a[1], a[2], a[3]);
      *reinterpret_cast<uint2*>(ab + p.nh) = bf4(b[0], b[1], b[2], b[3]);
      *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(p.aux) + grow * p.lda + n) = bf4(hh[0], hh[1], hh[2], hh[3]);
#endif
    }
  }
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
