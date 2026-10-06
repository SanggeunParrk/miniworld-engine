// gemm_tf32.cu — the TF32 tensor-core product of the atom block's fp32 path (sm_100a, tcgen05 kind::tf32), with the block's epilogues.
//
//   acc[M, n] = A[:, a_col0 : a_col0 + K] . B[b_row0 : b_row0 + n, b_col0 : b_col0 + K]^T      (fp32 operands, fp32 accumulation)
//
// A [M, *] and B [*, K'] are row-major fp32 (K-major operands), M a multiple of 128, K of 32, n of NT (128 or 256). Every projection of the
// block's fp32 path and every activation-gradient product is one launch; the weight gradients stay cuBLAS. Epilogues (``mode``):
//   0  out[:, ocol0 + j] = act(acc[:, j] + bias[b_row0 + j])  act = sigmoid on the 128-column blocks (b_row0 + j) / 128 set in sigmask
//                         (the conditioning tables: the AdaLN scales and the output gates leave as sigmoids); the blocks set in rndmask
//                         leave rounded to tf32 (RNA): the q / k / v columns, which only ever feed the attention's MMAs
//   1  out[:, j] = res[:, j] + gate[:, j] * (acc[:, j] (+ bias))  (+ the raw product into out2 when save2): the gated residuals
//                         a1 = a + so * (gated Wo^T) and out = a1 + st * (h Ws^T)
//   2  SwiGLU over an interleaved [Wa_0; Wb_0; Wa_1; Wb_1] pack (NT = 256: tile t = [a | b] of hidden columns 128 t ..):
//                         out[:, 128 t + j] = silu(acc[:, j]) * acc[:, 128 + j]  (+ the raw [a | b] into out2 when save2)
// Operand rounding: a kind::tf32 MMA reads an fp32 operand by dropping its low 13 mantissa bits (truncation: a one-sided error that
// accumulates coherently -- 1.6e-3 relative at the block output, 5x the cuBLAS TF32 path, which rounds). So every operand is rounded to
// nearest (cvt.rna.tf32) before the MMA reads it: the B operands (weights) in the host pack, the A operands here, in shared memory, by
// warps 12..15 between the TMA landing (full) and the MMA (rfull). The activations in global memory stay exact fp32.
// Persistent CTAs walk (128-row tile, NT-column tile) items, tile-major. Warp roles (512 threads): warp 0 lane 0 issues the TMA loads,
// warp 1 the MMAs (whole warp, elect_one), warp 2 owns TMEM, warps 4..11 are the epilogue (two warpgroups take alternate 32-column
// chunks; a thread owns one row = one TMEM lane) and store each 32 x 32 chunk through a per-warp double-buffered 4 KB staging
// (128-B swizzle) with a TMA store; warps 12..15 round the A slices (16 KB: 8 float4 per thread).
// Budgets. smem: 3 stages of A 16 KB (128 rows x 32 fp32) | B 32 KB (256 rows x 32 fp32) + 8 warps x 2 x 4 KB staging + barriers = 213248 B.
// TMEM: two accumulators of NT columns (512 allocated): the epilogue of tile i overlaps the MMAs of tile i + 1. Registers: <= 128
// (512 threads), the epilogue holds two 32-value chunks at most.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int ST = 3, A_BYTES = 128 * 128, B_BYTES = 256 * 128, STB = A_BYTES + B_BYTES;
constexpr int EPB = 4096, O_EP = ST * STB, O_BAR = O_EP + 8 * 2 * EPB, SMEM_BYTES = O_BAR + 256;
static_assert(STB % 1024 == 0 && O_EP % 1024 == 0 && A_BYTES % 1024 == 0, "1 KB alignment of the 128-B-swizzled tiles");
static_assert(SMEM_BYTES == 213248, "keep sm100_atom.KERNELS_TF32 in step");

struct Bars {
  uint64_t full[ST], rfull[ST], empty[ST], accfull[2], accfree[2];
  uint32_t tmem;
};
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI float sigf(float x) { return 1.f / (1.f + __expf(-x)); }
DEVI uint32_t rna_tf32(uint32_t x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(__uint_as_float(x))); return r; }

// one 32-row x 32-column fp32 chunk (v: this lane's row) -> staging buffer (nst & 1) of the warp -> TMA store at (col, row0)
DEVI void put_chunk(const CUtensorMap* map, uint32_t sbase, int& nst, int lane, int col, int row0, const float (&v)[32]) {
  const uint32_t buf = sbase + (uint32_t)(nst & 1) * EPB;
  if (lane == 0) tma_store_wait_read1();                             // the store that used this buffer two chunks ago has read it
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
atom_gemm_tf32(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mo,
               const __grid_constant__ CUtensorMap mo2, const float* __restrict__ bias, const float* __restrict__ res,
               const float* __restrict__ gate, int M, int N, int K, int NT, int a_col0, int b_row0, int b_col0, int ocol0, int sigmask,
               int mode, int save2, int res_ld, int gate_ld, int rndmask) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntn = N / NT, tiles = (M >> 7) * ntn, nk = K >> 5;
  const int my = (int)blockIdx.x < tiles ? (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto tile_of = [&](int i, int& m0, int& n0) {
    const int t = (int)blockIdx.x + i * (int)gridDim.x;
    m0 = (t / ntn) * 128;
    n0 = (t % ntn) * NT;
  };

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.rfull[s], 4); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.accfull[b], 1); mbar_init(&B.accfree[b], 8); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      const uint32_t tx = (uint32_t)A_BYTES + (uint32_t)NT * 128u;
      int g = 0;
      for (int i = 0; i < my; ++i) {
        int m0, n0;
        tile_of(i, m0, n0);
        for (int ks = 0; ks < nk; ++ks, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + s * STB;
          mbar_expect_tx(&B.full[s], tx);
          tma_load_2d(st, &ma, &B.full[s], a_col0 + 32 * ks, m0);
          tma_load_2d(st + A_BYTES, &mb, &B.full[s], b_col0 + 32 * ks, b_row0 + n0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    const uint32_t idesc = idesc_tf32(128, NT);
    int g = 0;
    for (int i = 0; i < my; ++i) {
      const int ab = i & 1;
      if (i >= 2) mbar_wait(&B.accfree[ab], ((i >> 1) - 1) & 1);      // the epilogue of tile i - 2 has read this accumulator
      tc_fence_after();
      const uint32_t d = tmem + (uint32_t)(ab * NT);
      for (int ks = 0; ks < nk; ++ks, ++g) {
        const int s = g % ST;
        mbar_wait(&B.rfull[s], (g / ST) & 1);                      // landed and rounded
        tc_fence_after();
        const uint32_t st = su + s * STB;
        const uint64_t da = desc_k128(st), db = desc_k128(st + A_BYTES);
        if (elect_one()) {
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)                             // K = 8 fp32 (32 B) per MMA
            umma_ss_tf32(d, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), idesc, (ks | kk) ? 1u : 0u);
          tc_commit(&B.empty[s]);
        }
        __syncwarp();
      }
      if (elect_one()) tc_commit(&B.accfull[ab]);
      __syncwarp();
    }
  } else if (warp >= 12) {
    // ------------------------------------------------------------------------------------------------ A-operand rounding (RNA to tf32)
    const int rt = tid - 384;                                        // 0..127
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
        fence_proxy_async();                                         // generic-proxy writes -> the MMA's (async-proxy) reads
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.rfull[s]);
      }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue: one row per thread
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const uint32_t sbase = su + O_EP + (uint32_t)e * 2 * EPB;
    int nst = 0;
    for (int i = 0; i < my; ++i) {
      int m0, n0;
      tile_of(i, m0, n0);
      const int ab = i & 1;
      mbar_wait(&B.accfull[ab], (i >> 1) & 1);
      tc_fence_after();
      const uint32_t tacc = tmem + ((uint32_t)(32 * qd) << 16) + (uint32_t)(ab * NT);
      const int r0 = m0 + 32 * qd, row = r0 + lane;
      if (mode == 2) {
        for (int c = wg; c < 4; c += 2) {                            // hidden columns 128 (n0 / 256) + 32 c ..
          float va[32], vb[32];
          ld_chunk(tacc + 32 * c, va);
          ld_chunk(tacc + 128 + 32 * c, vb);
          if (save2) {
            put_chunk(&mo2, sbase, nst, lane, n0 + 32 * c, r0, va);
            put_chunk(&mo2, sbase, nst, lane, n0 + 128 + 32 * c, r0, vb);
          }
#pragma unroll
          for (int j = 0; j < 32; ++j) va[j] = va[j] * sigf(va[j]) * vb[j];
          put_chunk(&mo, sbase, nst, lane, (n0 >> 1) + 32 * c, r0, va);
        }
      } else {
        for (int c = wg; c < (NT >> 5); c += 2) {
          float v[32];
          ld_chunk(tacc + 32 * c, v);
          const int nb = b_row0 + n0 + 32 * c;
          if (bias != nullptr) {
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] += __ldg(bias + nb + j);
          }
          if (mode == 0) {
            if ((sigmask >> (nb >> 7)) & 1) {
#pragma unroll
              for (int j = 0; j < 32; ++j) v[j] = sigf(v[j]);
            }
            if ((rndmask >> (nb >> 7)) & 1) {
#pragma unroll
              for (int j = 0; j < 32; ++j) v[j] = __uint_as_float(rna_tf32(__float_as_uint(v[j])));
            }
          } else {
            if (save2) put_chunk(&mo2, sbase, nst, lane, n0 + 32 * c, r0, v);
            const float4* rp = reinterpret_cast<const float4*>(res + (size_t)row * res_ld + n0 + 32 * c);
            const float4* gp = reinterpret_cast<const float4*>(gate + (size_t)row * gate_ld + n0 + 32 * c);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float4 rr = __ldg(rp + q), gg = __ldg(gp + q);
              v[4 * q] = fmaf(gg.x, v[4 * q], rr.x);
              v[4 * q + 1] = fmaf(gg.y, v[4 * q + 1], rr.y);
              v[4 * q + 2] = fmaf(gg.z, v[4 * q + 2], rr.z);
              v[4 * q + 3] = fmaf(gg.w, v[4 * q + 3], rr.w);
            }
          }
          put_chunk(&mo, sbase, nst, lane, ocol0 + n0 + 32 * c, r0, v);
        }
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
