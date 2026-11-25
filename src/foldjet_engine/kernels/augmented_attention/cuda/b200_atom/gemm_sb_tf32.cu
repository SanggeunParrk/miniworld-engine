// gemm_sb_tf32.cu — the transition's dh product of the fp32 path's backward with the SwiGLU backward as its epilogue (sm_100a, tcgen05
// kind::tf32; round 3 part 11, 2026-11-18). The H100 keeps dh on chip (tr_bwd90); the B200 wrote dh [M, 256] fp32 and read it back in
// f32_swiglu_bwd (2 KB a row, ~0.5 GB at 5120 atoms x 48 samples). Here:
//
//   dh[M, 256] = dt[:, 0 : K] . WsT[0 : 256, 0 : K]^T     (TMEM only)
//   with u = [a_0 | b_0 | a_1 | b_1] (the forward's raw [a | b], [M, 512]) and hidden column j = 128 hb + jj:
//     h  = silu(a) b                      -> hh [M, 256]   (the dWs operand)
//     da = dh b s (1 + a (1 - s))         -> dab[:, 256 hb + jj]
//     db = dh silu(a)                     -> dab[:, 256 hb + 128 + jj]     (s = sigmoid(a), rcp.approx as everywhere in the fp32 path)
//
// f32_swiglu_bwd's arithmetic, element for element: the outputs are its bits. Everything else is gemm_rg_tf32.cu: persistent (128-row tile)
// CTAs, TMA producer / MMA issuer / TMEM owner warps, A rounded to tf32 (RNA) in shared memory by warps 12..15, two TMEM accumulators of 256
// columns, the epilogue (warps 4..11: row quadrant qd, warpgroup wg takes hidden chunks wg, wg + 2, ...) loads the a / b chunks of u by
// TMA one chunk ahead and stores through a double-buffered 4 KB staging a warp.
// smem: 2 stages of A 16 KB | B 32 KB, 8 warps x (2 x 4 KB staging + 4 KB a + 4 KB b) + barriers = 229632 B.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int ST = 2, NT = 256, A_BYTES = 128 * 128, B_BYTES = NT * 128, STB = A_BYTES + B_BYTES;
constexpr int EPB = 4096, O_EP = ST * STB, O_U = O_EP + 8 * 2 * EPB, O_BAR = O_U + 8 * 2 * EPB, SMEM_BYTES = O_BAR + 256;
static_assert(STB % 1024 == 0 && O_EP % 1024 == 0 && O_U % 1024 == 0, "1 KB alignment of the 128-B-swizzled tiles");
static_assert(SMEM_BYTES == 229632, "keep b200_atom.KERNELS_TF32 in step");

struct Bars {
  uint64_t full[ST], rfull[ST], empty[ST], accfull[2], accfree[2], ul[8];
  uint32_t tmem;
};
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI uint32_t rna_tf32(uint32_t x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(__uint_as_float(x))); return r; }
DEVI float sigf(float x) { return rcpf(1.f + __expf(-x)); }          // (rows_tf32's)

DEVI void put_chunk(const CUtensorMap* map, uint32_t sbase, int& nst, int lane, int col, int row0, const float (&v)[32]) {
  const uint32_t buf = sbase + (uint32_t)(nst & 1) * EPB;
  if (lane == 0) tma_store_wait_read1();
  __syncwarp();
#pragma unroll
  for (int q = 0; q < 8; ++q)
    sts128(buf + sw128((uint32_t)lane, (uint32_t)q),
           make_uint4(__float_as_uint(v[4 * q]), __float_as_uint(v[4 * q + 1]), __float_as_uint(v[4 * q + 2]), __float_as_uint(v[4 * q + 3])));
  fence_proxy_async();
  __syncwarp();
  if (lane == 0) {
    tma_store_2d(map, buf, col, row0);
    tma_store_commit();
  }
  ++nst;
}

// ma: dt [M, K] (box 32 x 128); mb: WsT [256, K] (box 32 x 256); mu: u [M, 512], mdab: dab [M, 512], mhh: hh [M, 256] (box 32 x 32)
extern "C" __global__ void __launch_bounds__(512, 1)
atom_gemm_sb_tf32(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mu,
                  const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mhh, int M, int K) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tiles = M >> 7, nk = K >> 5;
  const int my = (int)blockIdx.x < tiles ? (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto tile_m0 = [&](int i) { return ((int)blockIdx.x + i * (int)gridDim.x) * 128; };

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.rfull[s], 4); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.accfull[b], 1); mbar_init(&B.accfree[b], 8); }
    for (int w = 0; w < 8; ++w) mbar_init(&B.ul[w], 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_wait();
  pdl_launch();

  if (warp == 0) {
    if (lane == 0) {
      const uint32_t tx = (uint32_t)A_BYTES + (uint32_t)B_BYTES;
      int g = 0;
      for (int i = 0; i < my; ++i) {
        const int m0 = tile_m0(i);
        for (int ks = 0; ks < nk; ++ks, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + s * STB;
          mbar_expect_tx(&B.full[s], tx);
          tma_load_2d(st, &ma, &B.full[s], 32 * ks, m0);
          tma_load_2d(st + A_BYTES, &mb, &B.full[s], 32 * ks, 0);
        }
      }
    }
  } else if (warp == 1) {
    const uint32_t idesc = idesc_tf32(128, NT);
    int g = 0;
    for (int i = 0; i < my; ++i) {
      const int ab = i & 1;
      if (i >= 2) mbar_wait(&B.accfree[ab], ((i >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t d = tmem + (uint32_t)(ab * NT);
      for (int ks = 0; ks < nk; ++ks, ++g) {
        const int s = g % ST;
        mbar_wait(&B.rfull[s], (g / ST) & 1);
        tc_fence_after();
        const uint32_t st = su + s * STB;
        const uint64_t da = desc_k128(st), db = desc_k128(st + A_BYTES);
        if (elect_one()) {
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) umma_ss_tf32(d, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), idesc, (ks | kk) ? 1u : 0u);
          tc_commit(&B.empty[s]);
        }
        __syncwarp();
      }
      if (elect_one()) tc_commit(&B.accfull[ab]);
      __syncwarp();
    }
  } else if (warp >= 12) {
    const int rt = tid - 384;
    int g = 0;
    for (int i = 0; i < my; ++i)
      for (int ks = 0; ks < nk; ++ks, ++g) {
        const int s = g % ST;
        const uint32_t st = su + s * STB;
        mbar_wait(&B.full[s], (g / ST) & 1);
#pragma unroll
        for (int q = 0; q < A_BYTES / 16 / 128; ++q) {
          const uint32_t ad = st + (uint32_t)(q * 128 + rt) * 16u;
          const uint4 v = lds128(ad);
          sts128(ad, make_uint4(rna_tf32(v.x), rna_tf32(v.y), rna_tf32(v.z), rna_tf32(v.w)));
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.rfull[s]);
      }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const uint32_t sbase = su + O_EP + (uint32_t)e * 2 * EPB;
    const uint32_t abuf = su + O_U + (uint32_t)e * 2 * EPB, bbuf = abuf + EPB;
    int nst = 0, q = 0;
    auto ucol = [](int c) { return 256 * (c >> 2) + 32 * (c & 3); };  // hidden chunk c (32 columns) -> its a columns in u / dab
    auto fetch = [&](int i, int c) {                                 // a and b of hidden chunk c of tile i
      if (lane == 0) {
        const int r0 = tile_m0(i) + 32 * qd;
        mbar_expect_tx(&B.ul[e], 2 * EPB);
        tma_load_2d(abuf, &mu, &B.ul[e], ucol(c), r0);
        tma_load_2d(bbuf, &mu, &B.ul[e], ucol(c) + 128, r0);
      }
    };
    if (my > 0) fetch(0, wg);
    for (int i = 0; i < my; ++i) {
      const int ab = i & 1;
      const int r0 = tile_m0(i) + 32 * qd;
      mbar_wait(&B.accfull[ab], (i >> 1) & 1);
      tc_fence_after();
      const uint32_t tacc = tmem + ((uint32_t)(32 * qd) << 16) + (uint32_t)(ab * NT);
      for (int c = wg; c < 8; c += 2) {
        float g[32];
        {
          uint32_t r[32];
          tmem_ld32(tacc + 32 * c, r);
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 32; ++j) g[j] = __uint_as_float(r[j]);
        }
        mbar_wait(&B.ul[e], q & 1);
        ++q;
        float av[32], bv[32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint4 a = lds128(abuf + sw128((uint32_t)lane, (uint32_t)j)), b = lds128(bbuf + sw128((uint32_t)lane, (uint32_t)j));
          av[4 * j] = __uint_as_float(a.x); av[4 * j + 1] = __uint_as_float(a.y); av[4 * j + 2] = __uint_as_float(a.z); av[4 * j + 3] = __uint_as_float(a.w);
          bv[4 * j] = __uint_as_float(b.x); bv[4 * j + 1] = __uint_as_float(b.y); bv[4 * j + 2] = __uint_as_float(b.z); bv[4 * j + 3] = __uint_as_float(b.w);
        }
        __syncwarp();                                                // every lane has read the buffers: the next chunk's loads may land
        if (c + 2 < 8) fetch(i, c + 2);
        else if (i + 1 < my) fetch(i + 1, wg);
        float h[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          const float s = sigf(av[j]), silu = av[j] * s;
          h[j] = silu * bv[j];
          const float db = g[j] * silu;
          av[j] = g[j] * bv[j] * s * (1.f + av[j] * (1.f - s));     // da
          bv[j] = db;
        }
        put_chunk(&mdab, sbase, nst, lane, ucol(c), r0, av);
        put_chunk(&mdab, sbase, nst, lane, ucol(c) + 128, r0, bv);
        put_chunk(&mhh, sbase, nst, lane, 32 * c, r0, h);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.accfree[ab]);
    }
    if (lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
