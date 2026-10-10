// gemm_rg_tf32.cu — the gated-residual product of the atom block's fp32 path (sm_100a, tcgen05 kind::tf32): the same product as
// gemm_tf32.cu's mode 1 specialised to N = 128, with the residual and the gate arriving by TMA instead of per-thread global loads.
//
//   out[:, j] = res[:, j] + gate[:, j] * acc[:, j],   acc[M, 128] = A[:, 0 : K] . B[0 : 128, 0 : K]^T       (+ acc itself into out2 when save2)
//
// One thread owns one row (one TMEM lane), so its res / gate values are 128 contiguous bytes of a row: a warp's load instruction touched 32
// different cache lines, 16 bytes each, and the epilogue ran at a third of the bandwidth of the same product without them (144 against 42 us at
// M = 196608). Here each epilogue warp loads its 32 x 32 chunk of res and of gate into shared memory with two TMA loads (128-B swizzle, the
// layout the output staging uses), issued one chunk ahead of its use, and reads them with ld.shared.
// Everything else is gemm_tf32.cu: persistent (128-row tile) CTAs, 4 warps' worth of TMA / MMA / TMEM roles, A rounded to tf32 (RNA) in
// shared memory by warps 12..15, two TMEM accumulators, a 32 x 32 chunk per TMA store.
// smem: 3 stages of A 16 KB | B 16 KB, 8 warps x (2 x 4 KB output staging + 4 KB res + 4 KB gate) + barriers = 229632 B.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int ST = 3, NT = 128, A_BYTES = 128 * 128, B_BYTES = NT * 128, STB = A_BYTES + B_BYTES;
constexpr int EPB = 4096, O_EP = ST * STB, O_RG = O_EP + 8 * 2 * EPB, O_BAR = O_RG + 8 * 2 * EPB, SMEM_BYTES = O_BAR + 256;
static_assert(STB % 1024 == 0 && O_EP % 1024 == 0 && O_RG % 1024 == 0 && A_BYTES % 1024 == 0, "1 KB alignment of the 128-B-swizzled tiles");
static_assert(SMEM_BYTES == 229632, "keep sm100_atom.KERNELS_TF32 in step");

struct Bars {
  uint64_t full[ST], rfull[ST], empty[ST], accfull[2], accfree[2], rg[8];
  uint32_t tmem;
};
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI uint32_t rna_tf32(uint32_t x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(__uint_as_float(x))); return r; }

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
DEVI void ld_chunk(uint32_t taddr, float (&v)[32]) {
  uint32_t r[32];
  tmem_ld32(taddr, r);
  tmem_wait_ld();
#pragma unroll
  for (int j = 0; j < 32; ++j) v[j] = __uint_as_float(r[j]);
}

extern "C" __global__ void __launch_bounds__(512, 1)
atom_gemm_rg_tf32(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mo,
                  const __grid_constant__ CUtensorMap mo2, const __grid_constant__ CUtensorMap mres, const __grid_constant__ CUtensorMap mgate,
                  int M, int K, int save2) {
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
    for (int w = 0; w < 8; ++w) mbar_init(&B.rg[w], 1);
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
    // epilogue: warp e (0..7) = row quadrant qd of warpgroup wg; chunks c = wg, wg + 2 of the 4 32-column chunks of every tile
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const uint32_t sbase = su + O_EP + (uint32_t)e * 2 * EPB;
    const uint32_t rbuf = su + O_RG + (uint32_t)e * 2 * EPB, gbuf = rbuf + EPB;
    int nst = 0, q = 0;                                              // q: the warp's chunk sequence (res / gate loads issued up to q)
    auto fetch = [&](int i, int c) {                                 // res and gate of chunk c of tile i into rbuf / gbuf
      if (lane == 0) {
        const int r0 = tile_m0(i) + 32 * qd;
        mbar_expect_tx(&B.rg[e], 2 * EPB);
        tma_load_2d(rbuf, &mres, &B.rg[e], 32 * c, r0);
        tma_load_2d(gbuf, &mgate, &B.rg[e], 32 * c, r0);
      }
    };
    if (my > 0) fetch(0, wg);
    for (int i = 0; i < my; ++i) {
      const int ab = i & 1;
      const int r0 = tile_m0(i) + 32 * qd;
      mbar_wait(&B.accfull[ab], (i >> 1) & 1);
      tc_fence_after();
      const uint32_t tacc = tmem + ((uint32_t)(32 * qd) << 16) + (uint32_t)(ab * NT);
      for (int c = wg; c < 4; c += 2) {
        float v[32];
        ld_chunk(tacc + 32 * c, v);
        if (save2) put_chunk(&mo2, sbase, nst, lane, 32 * c, r0, v);
        mbar_wait(&B.rg[e], q & 1);                                  // this chunk's res / gate have landed
        ++q;
        float rr[32], gg[32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint4 a = lds128(rbuf + sw128((uint32_t)lane, (uint32_t)j)), b = lds128(gbuf + sw128((uint32_t)lane, (uint32_t)j));
          rr[4 * j] = __uint_as_float(a.x); rr[4 * j + 1] = __uint_as_float(a.y); rr[4 * j + 2] = __uint_as_float(a.z); rr[4 * j + 3] = __uint_as_float(a.w);
          gg[4 * j] = __uint_as_float(b.x); gg[4 * j + 1] = __uint_as_float(b.y); gg[4 * j + 2] = __uint_as_float(b.z); gg[4 * j + 3] = __uint_as_float(b.w);
        }
#pragma unroll
        for (int j = 0; j < 32; ++j) v[j] = fmaf(gg[j], v[j], rr[j]);   // (consumes the shared-memory reads)
        __syncwarp();                                                // every lane has read the buffers: the next chunk's loads may land
        // the next chunk of this warp (this tile's c + 2, or the next tile's wg)
        if (c + 2 < 4) fetch(i, c + 2);
        else if (i + 1 < my) fetch(i + 1, wg);
        put_chunk(&mo, sbase, nst, lane, 32 * c, r0, v);
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
