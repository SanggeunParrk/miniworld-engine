// tdxln_w.cu — d_xn = [dA | dB] [Wa; Wb] fused with the LayerNorm backward + residual (the tbwd.cu contract), D = 256 (K = 2H = 2048),
// sm_100a, bf16 operands, fp32 accumulation:
//   n = bf16(d_xn), xhat = (x - mean) rstd, w = gamma n,  ca = mean(xhat w), cb = mean(w),  dx = bf16(bf16((w - xhat ca - cb) rstd) + dy)
//   dgamma, dbeta partials = sum over the CTA's rows of n xhat, n       (a second kernel sums the partials in a fixed order)
// d_xn never reaches HBM. Main loop as tgemm_nd.cu (2-CTA, M = 256, N = 256 split over the pair, 3-stage ring, TMEM double-buffered).
// Epilogue: warpgroup g (warps 4-7 / 12-15) takes the 64-column blocks g and g + 2; its x and dy blocks for the tile sit in four 16 KB
// slots (prefetched one tile ahead); pass 1 reads d_xn from TMEM once (kept in registers as bf16, the accumulator is released right
// away) and forms the row sums, exchanged between the warpgroups through shared memory, and the dgamma / dbeta terms (warp reduce-scatter
// per 32 columns, kept per lane across tiles); pass 2 writes dx over dy and TMA-stores it. SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 256, K_ = 2048, ROWS = 128, NKB = K_ / 64;
constexpr int KBT = ROWS * 128;
constexpr int BHALF = (D_ / 2) * 128;
constexpr int S_A = 0, S_B = KBT, STAGE = KBT + BHALF;         // 32 KB
constexpr int NST = 3;
constexpr int O_ST = 0, O_EP = NST * STAGE;                     // per warpgroup: x blocks 0, 1 | dy blocks 0, 1
constexpr int O_RED = O_EP + 2 * 4 * KBT;                       // row sums: [2 warpgroups][128 rows][2]
constexpr int O_BAR = O_RED + 2 * ROWS * 2 * 4;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(256, D_);

struct Bars {
  uint64_t full[NST];
  uint64_t empty[NST], acc_full[2];
  uint64_t acc_empty[2];
  uint64_t xy_full[2];                                          // local, per warpgroup: its x and dy blocks of a tile
  uint32_t tmem;
};

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

// maps: A = dab [M][2H], Bt = [Wa; Wb]^T [D][2H], x, dy, dx [M][D] (box 64 x 64); part [G * 4][2 D]
extern "C" __global__ void __launch_bounds__(512, 1)
transition_dxln_w_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mbt,
                        const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdy,
                        const __grid_constant__ CUtensorMap mdx, const float* __restrict__ rstd, const float* __restrict__ c1,
                        const float* __restrict__ gamma, float* __restrict__ part, int tiles) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acc_full[s], 1); mbar_init(&B.acc_empty[s], 16); mbar_init(&B.xy_full[s], 1); }
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mbt); prefetch_map(&mx); prefetch_map(&mdy); prefetch_map(&mdx);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    if (lane == 0) {
      int st = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = tile_of(i) * ROWS;
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NST;
          if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * STAGE);
          const uint32_t base = su + O_ST + s * STAGE;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            tma_load_2d_2sm(base + S_A + h * 8192, &ma, &B.full[s], kb * 64, row + h * 64);
            tma_load_2d_2sm(base + S_B + h * 8192, &mbt, &B.full[s], kb * 64, crank * 128 + h * 64);
          }
        }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      int st = 0;
      for (int i = 0; i < n_local; ++i) {
        const int s = i & 1, u = i >> 1;
        if (i >= 2) mbar_wait_cl(&B.acc_empty[s], (u - 1) & 1);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + sg * STAGE;
          const uint64_t da = desc_k128(base + S_A), db = desc_k128(base + S_B);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2(tmem + s * 256, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) tc_commit2_mc(&B.acc_full[s], 3);
          }
          __syncwarp();
        }
      }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    setmaxnreg_inc<152>();
    const int g = warp >= 12 ? 1 : 0;
    const bool lead = (warp & 3) == 0 && lane == 0;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t ep = su + O_EP + g * 4 * KBT;                 // x block k at ep + k KBT, dy block k at ep + (2 + k) KBT
    float* red = reinterpret_cast<float*>(sm + O_RED);          // [g][row][2]
    auto load_xy = [&](int i) {
      const int row = tile_of(i) * ROWS;
      mbar_expect_tx(&B.xy_full[g], 4 * KBT);
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          tma_load_2d(ep + k * KBT + h * 8192, &mx, &B.xy_full[g], (g + 2 * k) * 64, row + h * 64);
          tma_load_2d(ep + (2 + k) * KBT + h * 8192, &mdy, &B.xy_full[g], (g + 2 * k) * 64, row + h * 64);
        }
    };
    if (lead && n_local > 0) load_xy(0);
    float accg[4] = {0.f, 0.f, 0.f, 0.f}, accb[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < n_local; ++i) {
      const int s = i & 1, u = i >> 1, grow = tile_of(i) * ROWS + (int)r;
      const bool real = i < n_valid;
      const float rs = __ldg(rstd + grow), mean = __ldg(c1 + grow) / rs;
      mbar_wait(&B.acc_full[s], u & 1);
      tc_fence_after();
      mbar_wait(&B.xy_full[g], i & 1);
      float pa = 0.f, pb = 0.f;
      // pass 1, 32 columns at a time: n = bf16(d_xn) from TMEM, row sums, dgamma / dbeta terms (one reusable work array)
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int cb = g + 2 * k;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          uint32_t n16[16];
          {
            uint32_t v[32];
            tmem_ld32(trow + s * 256 + cb * 64 + hh * 32, v);
            tmem_wait_ld();
#pragma unroll
            for (int e = 0; e < 16; ++e) n16[e] = pack_bf16(__uint_as_float(v[2 * e]), __uint_as_float(v[2 * e + 1]));
          }
          float work[32];
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            const uint4 xv = lds128(ep + k * KBT + sw128(r, hh * 4 + qq));
            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
            const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma) + (cb * 64 + hh * 32 + qq * 8) / 4);
            const float4 g1 = __ldg(reinterpret_cast<const float4*>(gamma) + (cb * 64 + hh * 32 + qq * 8) / 4 + 1);
            const float gm[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint32_t nw = n16[qq * 4 + e];
              const float n0 = bf16lo(nw), n1 = bf16hi(nw);
              const float x0 = (bf16lo(xw[e]) - mean) * rs, x1 = (bf16hi(xw[e]) - mean) * rs;
              const float w0 = gm[2 * e] * n0, w1 = gm[2 * e + 1] * n1;
              pa += x0 * w0 + x1 * w1; pb += w0 + w1;
              work[qq * 8 + 2 * e] = n0 * x0; work[qq * 8 + 2 * e + 1] = n1 * x1;
            }
          }
          const float sg = reduce_scatter32(work, lane);
#pragma unroll
          for (int e = 0; e < 16; ++e) { work[2 * e] = bf16lo(n16[e]); work[2 * e + 1] = bf16hi(n16[e]); }
          const float sb = reduce_scatter32(work, lane);
          if (real) { accg[k * 2 + hh] += sg; accb[k * 2 + hh] += sb; }
        }
      }
      red[(g * ROWS + r) * 2] = pa; red[(g * ROWS + r) * 2 + 1] = pb;
      named_bar_sync(3, 256);                                  // both warpgroups' halves of every row
      const float ca = (red[r * 2] + red[(ROWS + r) * 2]) * (1.f / D_), cbv = (red[r * 2 + 1] + red[(ROWS + r) * 2 + 1]) * (1.f / D_);
      // pass 2: n again from TMEM (the buffer is released after it; the next tile accumulates in the other one)
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int cb = g + 2 * k;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          uint32_t n16[16];
          {
            uint32_t v[32];
            tmem_ld32(trow + s * 256 + cb * 64 + hh * 32, v);
            tmem_wait_ld();
#pragma unroll
            for (int e = 0; e < 16; ++e) n16[e] = pack_bf16(__uint_as_float(v[2 * e]), __uint_as_float(v[2 * e + 1]));
          }
          if (k == 1 && hh == 1) {
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
          }
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            const int q = hh * 4 + qq;
            const uint32_t off = sw128(r, q);
            const uint4 xv = lds128(ep + k * KBT + off), dv = lds128(ep + (2 + k) * KBT + off);
            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, dw[4] = {dv.x, dv.y, dv.z, dv.w};
            const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma) + (cb * 64 + q * 8) / 4);
            const float4 g1 = __ldg(reinterpret_cast<const float4*>(gamma) + (cb * 64 + q * 8) / 4 + 1);
            const float gm[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
            uint32_t o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint32_t nw = n16[qq * 4 + e];
              const float x0 = (bf16lo(xw[e]) - mean) * rs, x1 = (bf16hi(xw[e]) - mean) * rs;
              const float w0 = gm[2 * e] * bf16lo(nw), w1 = gm[2 * e + 1] * bf16hi(nw);
              const uint32_t t = pack_bf16((w0 - (x0 * ca + cbv)) * rs, (w1 - (x1 * ca + cbv)) * rs);
              o[e] = pack_bf16(bf16lo(t) + bf16lo(dw[e]), bf16hi(t) + bf16hi(dw[e]));
            }
            sts128(ep + (2 + k) * KBT + off, make_uint4(o[0], o[1], o[2], o[3]));
          }
        }
      }
      fence_proxy_async();
      named_bar_sync(3, 256);                                  // row sums read by all before the next tile overwrites them
      if (lead) {
        if (real) {
          const int row0 = tile_of(i) * ROWS;
#pragma unroll
          for (int k = 0; k < 2; ++k)
#pragma unroll
            for (int h = 0; h < 2; ++h) tma_store_2d(&mdx, ep + (2 + k) * KBT + h * 8192, (g + 2 * k) * 64, row0 + h * 64);
        }
        tma_store_commit();
        tma_store_wait_read0();
        if (i + 1 < n_local) load_xy(i + 1);
      }
    }
    float* prow = part + ((size_t)cta * 4 + (warp & 3)) * (2 * D_);
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int col = (g + 2 * k) * 64 + hh * 32 + lane;
        prow[col] = accg[k * 2 + hh]; prow[D_ + col] = accb[k * 2 + hh];
      }
    if (lead) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}

// dgamma | dbeta = sum over the partial rows (fixed order)
extern "C" __global__ void transition_dxln_w_reduce(const float* __restrict__ part, float* __restrict__ dgb, int nrows) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 2 * D_) return;
  float v = 0.f;
  for (int r = 0; r < nrows; ++r) v += part[(size_t)r * 2 * D_ + col];
  dgb[col] = v;
}
