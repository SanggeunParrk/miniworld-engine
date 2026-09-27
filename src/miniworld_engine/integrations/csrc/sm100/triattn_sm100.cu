// Triangle-attention forward for B200 (sm_100a): the cuEquivariance / opt_core `triangle_attention` op, D = 32, bf16 operands,
// fp32 softmax and accumulation, bf16 out -- the op the H100 kernel (opt_core kernels/triattn/cuda_sm90a) serves, on tcgen05 / TMEM.
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]        (no mask)
//
// A CTA task = one (b, h, 128-query tile) x R pair rows (the bias tile is shared by every pair row: R rows reuse each staged bias
// tile, as the H100 kernel's 3-row CTAs do).  Per 128-key tile j and row r (a "sub-step"):
//   MMA warp:      S = Q_r . K_{r,j}^T                (M = 128 q, N = 128 k, K = 32) into a TMEM buffer
//   softmax warps: x = S * scale*log2e + bias * log2e; p = 2^(x - m_r); l_r += p; P (bf16) written over the S buffer in TMEM
//   MMA warp:      O_r += P . V_{r,j}                  (A = P from TMEM, B = V MN-major, N = 32)
// The softmax offset m_r is the row maximum of the first key tile and is never moved afterwards (no running maximum, no O rescale):
// a later logit would have to exceed it by ~100 log2 units to overflow, which a guard counts (FLAGS) -- the H100 kernel's max-free
// softmax, with its offset fixed at the first tile.
// Warps: 0 = TMA producer (Q of the task's rows, K|V per sub-step, the bias tile per key tile), 1 = MMA issue, 2-5 / 6-9 = two softmax
// groups (rows 0, 2 / 1, 3 of the task: sub-step x uses S buffer x & 1 = r & 1) + their rows' epilogue.  A task always has R rows
// (rows past N are computed on row N-1 and not stored), so the sub-step parity is the row parity.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

namespace {
constexpr int D = 32, BM = 128, BN = 128, R = 4;
constexpr int QT = BM * D;                                 // Q tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][32], 64B swizzle (8 KiB)
constexpr int BT = BM * BN;                                // bias tile [2 key halves][128 q][64], 128B swizzle, bf16 (32 KiB)
constexpr int NKV = 5, NB = 2;
constexpr int THREADS = 448;                               // 0 TMA, 1 MMA, 2-5 / 6-9 softmax groups (even / odd rows), 10-13 epilogue
constexpr int SMEM = 1024 + (2 * R * QT + NKV * 2 * KVT + NB * BT) * 2 + 2 * 2 * R * BM * 4 + 512;   // + the row sums / offsets for the epilogue
constexpr int NS = 3;                                      // S/P TMEM buffers: with two, a group's buffer sat idle through PV + QK^T
constexpr int COL_S = 0, COL_O = NS * BN;                  // S/P [NS][128] | O [R rows][32] (single: the next task's first PV waits the drain)
constexpr uint32_t ID_S = idesc_bf16(128, BN, 0, 0);       // A = Q K-major, B = K K-major
constexpr uint32_t ID_O = idesc_bf16(128, D, 0, 1);        // A = P from TMEM, B = V MN-major
constexpr float L2E = 1.4426950408889634f;
static_assert(SMEM <= 232448, "one CTA per SM");

__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
#ifndef TA_POLY
#define TA_POLY 0        // half of the exponentials on the FMA pipe (the MUFU's 16 / clk / SM is the softmax's bound)
#endif
// 2^x on the FMA pipe, two at a time: n = round(x) by the 1.5 * 2^23 add, 2^f (|f| <= 1/2) by a degree-3 minimax polynomial
// (relative error 7.5e-5, well under P's bf16 rounding), n added into the exponent bits: (bits(t) << 23) == n << 23 exactly.
__device__ __forceinline__ float2 ex2_poly2(float2 x) {
  x.x = fmaxf(x.x, -126.f); x.y = fmaxf(x.y, -126.f);
  const float2 t = add2(x, make_float2(12582912.f, 12582912.f));
  const float2 rr = add2(t, make_float2(-12582912.f, -12582912.f));
  const float2 f = add2(x, make_float2(-rr.x, -rr.y));
  float2 pp = fma2(f, make_float2(0.05517097723f, 0.05517097723f), make_float2(0.24260970271f, 0.24260970271f));
  pp = fma2(f, pp, make_float2(0.69326096627f, 0.69326096627f));
  pp = fma2(f, pp, make_float2(0.99992816244f, 0.99992816244f));
  return make_float2(__int_as_float(__float_as_int(pp.x) + (__float_as_int(t.x) << 23)),
                     __int_as_float(__float_as_int(pp.y) + (__float_as_int(t.y) << 23)));
}


__global__ void __launch_bounds__(THREADS, 1) triattn_fwd_sm100(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][H*32] bf16 (the projection's layout), box (32, 128) at column h*32, 64B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*32]
    float* __restrict__ LSE,                             // [B, N, H, S] base-2 log-sum-exp for the backward, or nullptr
    int* __restrict__ FLAGS) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 tasks][R][QT]
  __nv_bfloat16* sKV = sQ + 2 * R * QT;                          // [NKV][K | V]
  __nv_bfloat16* sB = sKV + NKV * 2 * KVT;                       // [NB][BT]
  float* sL = reinterpret_cast<float*>(sB + NB * BT);            // [2 tasks][R][128] row sums
  float* sM = sL + 2 * R * BM;                                   // [2 tasks][R][128] row offsets
  uint64_t* bars = reinterpret_cast<uint64_t*>(sM + 2 * R * BM);
  uint64_t* qf = bars;               // [2]   the task's Q tiles landed
  uint64_t* qe = qf + 2;             // [2]   the task's last QK^T retired
  uint64_t* kvf = qe + 2;            // [NKV]
  uint64_t* kve = kvf + NKV;         // [NKV] the PV GEMM of the stage retired
  uint64_t* bf = kve + NKV;          // [NB]
  uint64_t* be = bf + NB;            // [NB]  count 8: both softmax groups are done with the bias tile
  uint64_t* sf = be + NB;            // [NS]  S of the sub-step ready
  uint64_t* pf = sf + NS;            // [NS]  count 4: P written (over S)
  uint64_t* of = pf + NS;            // [1]   the task's O complete
  uint64_t* oe = of + 1;             // [1]   count 4: the epilogue drained O
  uint64_t* lf = oe + 1;             // [2]   count 8: both groups wrote the task's row sums
  uint32_t* tslot = reinterpret_cast<uint32_t*>(lf + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < 2; ++k) { bar_init(qf + k, 1); bar_init(qe + k, 1); }
    for (int k = 0; k < NS; ++k) { bar_init(sf + k, 1); bar_init(pf + k, 4); }
    bar_init(of, 1); bar_init(oe, 4); bar_init(lf, 8); bar_init(lf + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(kvf + k, 1); bar_init(kve + k, 1); }
    for (int k = 0; k < NB; ++k) { bar_init(bf + k, 1); bar_init(be + k, 8); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  // task t: q tile fastest (the CTAs sharing a row group's K / V run together), then the row group, then (b, h)
  auto decode = [&](int t, int& bh, int& qt, int& g) { qt = t % QTILES; const int r = t / QTILES; g = r % NG; bh = r / NG; };
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H; n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row of q / k / v / o
  auto hcol = [&](int bh) -> int { return (bh % H) * D; };

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        const int nr = R, qb = lt & 1;
        if (lt >= 2) wait(qe + qb, ((lt >> 1) - 1) & 1);
        expect_tx(qf + qb, nr * QT * 2);
        for (int r = 0; r < nr; ++r) load_2d(&qmap, sQ + (qb * R + r) * QT, qf + qb, hcol(bh), row0(bh, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1);
          expect_tx(bf + bs, BT * 2);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sB + bs * BT + hh * BM * 64, bf + bs, j * BN + hh * 64, bh * S + qt * BM);
          for (int r = 0; r < nr; ++r, ++x) {
            const int st = x % NKV;
            if (x >= NKV) wait(kve + st, ((x / NKV) - 1) & 1);
            expect_tx(kvf + st, 2 * KVT * 2);
            const int rw = row0(bh, g * R + r) + j * BN;
            load_2d(&kmap, sKV + st * 2 * KVT, kvf + st, hcol(bh), rw);
            load_2d(&vmap, sKV + st * 2 * KVT + KVT, kvf + st, hcol(bh), rw);
          }
        }
      }
    }
  } else if (warp == 1) {
    // flattened sub-steps y (task k = y / SPT, key tile j, row r): QK^T runs up to NS - 1 sub-steps ahead of the PV GEMMs, so a
    // softmax group finds its next S ready when it hands over P (issued strictly one ahead, each group idled ~1K cycles per sub-step)
    const int SPT = NK * R;
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    auto qk = [&](int y) {
      const int k = y / SPT, w = y % SPT, j = w / R, r = w % R, st = y % NKV, b = y % NS, qb = k & 1;
      if (w == 0) wait(qf + qb, (k >> 1) & 1);
      wait(kvf + st, (y / NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* q = sQ + (qb * R + r) * QT;
        const __nv_bfloat16* kk = sKV + st * 2 * KVT;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + COL_S + b * BN, desc_k64(q + ks * 16), desc_k64(kk + ks * 16), ID_S, ks ? 1u : 0u);
        mma_commit(sf + b);
        if (w == SPT - 1) mma_commit(qe + qb);
      }
      __syncwarp();
      (void)j;
    };
    auto pv = [&](int y) {                       // O_r += P . V
      const int k = y / SPT, w = y % SPT, j = w / R, r = w % R, st = y % NKV, b = y % NS;
      wait(pf + b, (y / NS) & 1);
      if (w == 0 && k >= 1) wait(oe, (k - 1) & 1);   // the previous task's O was drained
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* v = sKV + st * 2 * KVT + KVT;
        const uint32_t o = tmem + COL_O + r * D;
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks)
          mma_ts(o, tmem + COL_S + b * BN + ks * 8, sdesc(sa(v + ks * 16 * D), KVT * 2, 512, 4), ID_O, (j | ks) ? 1u : 0u);
        mma_commit(kve + st);
        if (w == SPT - 1) mma_commit(of);
      }
      __syncwarp();
    };
    int y = 0;
    for (int p = 0; p < total; ++p) {
      while (y < total && y < p + NS) qk(y++);   // S buffer y % NS is free once PV(y - NS) is issued (the tensor pipe runs in order)
      pv(p);
    }
  } else if (warp < 10) {
    const int q = warp & 3, row = q * 32 + lane;   // TMEM lane = query row of the tile
    const int gi = (warp - 2) >> 2;                // this group's rows: gi, gi + 2
    int lt = 0, x = 0, jb = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      const int ob = lt & 1;
      float m[R / 2], l[R / 2];
#pragma unroll
      for (int r = 0; r < R / 2; ++r) { m[r] = 0.f; l[r] = 0.f; }
      for (int j = 0; j < NK; ++j, ++jb) {
        const int bs = jb % NB;
        wait(bf + bs, (jb / NB) & 1);
        const __nv_bfloat16* brow = sB + bs * BT + row * 64;
        for (int rr = 0; rr < R / 2; ++rr) {
          const int r = gi + 2 * rr, xs = x + r, b = xs % NS;
          wait(sf + b, (xs / NS) & 1);
          tc_fence_after();
          const uint32_t sb = tmem_at(tmem + COL_S + b * BN, q * 32, 0);
          float mr = m[rr], lr = 0.f;
          const float2 sc2 = make_float2(scl, scl), l2 = make_float2(L2E, L2E);
          // x = S * scale*log2e + bias * log2e for chunk c (32 keys), bias from the staged tile
          auto logits = [&](int c, float* s32, float2 nm0) {   // x = S * scale*log2e + bias * log2e + nm0 (two FFMA2 a pair)
            const __nv_bfloat16* bp = brow + (c >> 1) * BM * 64;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint4 u = *reinterpret_cast<const uint4*>(bp + ((((c & 1) * 4 + e) ^ (row & 7)) << 3));
              const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
              for (int k2 = 0; k2 < 4; ++k2) {
                const int i2 = e * 8 + k2 * 2;
                const float2 xx = fma2(bf2f(w[k2]), l2, fma2(make_float2(s32[i2], s32[i2 + 1]), sc2, nm0));
                s32[i2] = xx.x; s32[i2 + 1] = xx.y;
              }
            }
          };
          {
            float2 acc = make_float2(0.f, 0.f);
            float s32[2][32];
            tmem_ld32(sb, s32[0]);
            tmem_wait_ld();
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (c + 1 < 4) tmem_ld32(sb + (c + 1) * 32, s32[(c + 1) & 1]);
              float* sc = s32[c & 1];
              if (j == 0 && c == 0) {            // the row's offset: the first 32 keys' maximum, fixed for the task (overflow
                logits(c, sc, make_float2(0.f, 0.f));   // needs a later logit ~125 log2 units above it; the epilogue counts any)
                float m0 = sc[0], m1 = sc[1];
#pragma unroll
                for (int i2 = 2; i2 < 32; i2 += 2) { m0 = fmaxf(m0, sc[i2]); m1 = fmaxf(m1, sc[i2 + 1]); }
                mr = fmaxf(m0, m1);
#pragma unroll
                for (int i2 = 0; i2 < 32; ++i2) sc[i2] -= mr;
              } else {
                logits(c, sc, make_float2(-mr, -mr));
              }
              uint32_t pk[16];
#pragma unroll
              for (int k2 = 0; k2 < 16; ++k2) {
                const float2 xx = make_float2(sc[k2 * 2], sc[k2 * 2 + 1]);
                float2 pv;
                if (TA_POLY && (k2 & 1)) pv = ex2_poly2(xx);
                else pv = make_float2(ex2f(xx.x), ex2f(xx.y));
                acc = add2(acc, pv);
                pk[k2] = pack2(pv.x, pv.y);
              }
              tmem_wait_ld();                    // chunk c+1 landed (and every S column chunk c's P covers was read)
              tmem_st8(sb + c * 16, pk);         // P over S: chunk c's 16 columns cover S columns already read
              tmem_st8(sb + c * 16 + 8, pk + 8);
            }
            lr = acc.x + acc.y;
          }
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(pf + b);
          m[rr] = mr; l[rr] += lr;
        }
        x += R;
        __syncwarp();
        if (lane == 0) arrive(be + bs);
      }
      // hand the row sums to the epilogue warps
#pragma unroll
      for (int rr = 0; rr < R / 2; ++rr) { sL[((lt & 1) * R + gi + 2 * rr) * BM + row] = l[rr]; sM[((lt & 1) * R + gi + 2 * rr) * BM + row] = m[rr]; }
      __syncwarp();
      if (lane == 0) arrive(lf + (lt & 1));
    }
  } else if (warp < 14) {
    // ---- epilogue warps: O_r / l_r -> bf16 -> out, overlapping the next task's softmax ----
    const int q = warp & 3, row = q * 32 + lane;
    int lt = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      wait(of, lt & 1);
      wait(lf + (lt & 1), (lt >> 1) & 1);
      tc_fence_after();
      float o[R][32];
#pragma unroll
      for (int r = 0; r < R; ++r) tmem_ld32(tmem_at(tmem + COL_O + r * D, q * 32, 0), o[r]);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(oe);                 // O is free for the next task's PV GEMMs
      int bad = 0;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (g * R + r >= N) continue;            // a padding row
        const float lr = sL[((lt & 1) * R + r) * BM + row];
        const float inv = 1.f / lr;
        bad |= !(lr > 0.f) || !(lr < 3.0e38f);
        if (LSE != nullptr) {                     // base 2: lse = m + log2(l), [B, N, H, S]
          const int b_ = bh / H, h_ = bh % H;
          LSE[((size_t)(b_ * N + g * R + r) * H + h_) * S + qt * BM + row] = sM[((lt & 1) * R + r) * BM + row] + __log2f(lr);
        }
        uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (H * D) + hcol(bh));
#pragma unroll
        for (int c8 = 0; c8 < 4; ++c8) {
          uint4 w;
          w.x = pack2(o[r][c8 * 8 + 0] * inv, o[r][c8 * 8 + 1] * inv); w.y = pack2(o[r][c8 * 8 + 2] * inv, o[r][c8 * 8 + 3] * inv);
          w.z = pack2(o[r][c8 * 8 + 4] * inv, o[r][c8 * 8 + 5] * inv); w.w = pack2(o[r][c8 * 8 + 6] * inv, o[r][c8 * 8 + 7] * inv);
          dst[c8] = w;
        }
      }
      if (bad) atomicAdd(FLAGS, 1);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ================================================================================================================================
// Backward.  With P = 2^(x - lse) (x the forward's base-2 logit), dP = dO . V^T, Delta = rowsum(dO o O), dS = P o (dP - Delta):
//   dV = P^T dO,  dK = scale * dS^T Q,  dQ = scale * dS K,  dbias[b,h] = sum_n dS[b,n,h].
// The H100 kernel boundaries (a KV-owned dK/dV kernel, a query-owned dQ kernel, both recomputing P), built around two sm_100 facts:
//   * tcgen05 is fed from shared memory at ~67 B/clk/SM (measured: an ss MMA with a 128 x 32 A tile costs ~65 clk even at N = 32,
//     the same MMA with A in TMEM ~30 clk), so the operand that stays fixed for a whole task -- K / V here, Q / dO in the dQ kernel --
//     is copied once into TMEM and every per-stage MMA is a ts MMA that reads only its streaming B tile from shared memory;
//   * dS never goes to HBM (N x L^2 bf16: 3.6 GB at L = 768): each kernel recomputes P from the forward's LSE.
// dbias is summed in the dQ kernel, where the query sits on the TMEM lane: a thread holds 32 consecutive keys of one query, so the
// bias tile is read with 16-byte loads and the R-row partial sums go to an L2-resident fp32 [q][k] buffer with 16-byte reductions.
#ifndef TA_TL
#define TA_TL 0
#endif
#ifndef TA_NOLD
#define TA_NOLD 0
#endif
#ifndef TA_NOLDS
#define TA_NOLDS 0
#endif
#ifndef TA_NOST
#define TA_NOST 0
#endif
#ifndef TA_NOMATH
#define TA_NOMATH 0
#endif
#ifndef TA_SLEEP
#define TA_SLEEP 0
#endif
#ifndef TA_NOPF
#define TA_NOPF 0
#endif
#ifndef TA_NODKST
#define TA_NODKST 0
#endif
#ifndef TA_BIAS1
#define TA_BIAS1 0
#endif
#ifndef TA_QD1
#define TA_QD1 0
#endif
#ifndef TA_FAKEK
#define TA_FAKEK 0
#endif
#if TA_TL          // clock64 timeline of one CTA: [event][stage]
__device__ long long g_tl[24][256];
#define TL(e, y) do { if (blockIdx.x == TA_TL && (y) < 256) g_tl[e][y] = clock64(); } while (0)
#else
#define TL(e, y) do {} while (0)
#endif

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void reduce_add_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void reduce_add_2d(const void* map, const void* src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1) : "memory");
}
// 16 packed words -> row `row` of a [rows][32] bf16 tile with the 64-byte swizzle (the TMA store's layout)
__device__ __forceinline__ void row_sw64_st(__nv_bfloat16* tile, int row, const uint32_t* w) {
  unsigned char* base = reinterpret_cast<unsigned char*>(tile) + row * 64;
#pragma unroll
  for (int c = 0; c < 4; ++c)
    *reinterpret_cast<uint4*>(base + ((c ^ ((row >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
}
// row `row` of a [rows][32] bf16 tile with the 64-byte swizzle (16-byte chunk c at c ^ ((row >> 1) & 3)) -> 16 packed words
__device__ __forceinline__ void row_sw64(const __nv_bfloat16* tile, int row, uint32_t* w) {
  const unsigned char* base = reinterpret_cast<const unsigned char*>(tile) + row * 64;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 u = *reinterpret_cast<const uint4*>(base + ((c ^ ((row >> 1) & 3)) << 4));
    w[c * 4] = u.x; w[c * 4 + 1] = u.y; w[c * 4 + 2] = u.z; w[c * 4 + 3] = u.w;
  }
}

// ---- bias^T * log2e (fp32 [b, h, k, q]) for the KV kernel, whose threads own a key: one 32 x 32 tile per block
__global__ void __launch_bounds__(256) triattn_biasT(const __nv_bfloat16* __restrict__ bias, float* __restrict__ out, int S) {
  __shared__ float t[32][33];
  const int bh = blockIdx.z, q0 = blockIdx.y * 32, k0 = blockIdx.x * 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
  const __nv_bfloat16* src = bias + (size_t)bh * S * S;
#pragma unroll
  for (int i = ty; i < 32; i += 8) t[i][tx] = __bfloat162float(src[(size_t)(q0 + i) * S + k0 + tx]) * L2E;
  __syncthreads();
  float* dst = out + (size_t)bh * S * S;
#pragma unroll
  for (int i = ty; i < 32; i += 8) dst[(size_t)(k0 + i) * S + q0 + tx] = t[tx][i];
}

// ---- kernel 1: dK, dV.  Task = (b, h, 128-key tile) x KR = 2 pair rows, all queries in 64-wide sub-tiles; stage y = (sub-tile, row),
// y & 1 = the row = the grad group that owns it (the two groups alternate stages, as the forward's softmax groups do):
//   S^T MMA warp:  S^T = K_r . Q^T, dP^T = V_r . dO^T  (ss: K_r / V_r from shared; M = 128 k, N = 64 q, K = 32)
//   grad group r:  P^T, dS^T (bf16) -> its P^T | dS^T TMEM buffer
//   grad MMA warp: dV_r += P^T . dO, dK_r += dS^T . Q   (ts: A from TMEM, B MN-major; N = 32, K = 64)
// Every hand-off here (a barrier round trip, a group of MMA issues) costs ~100-300 clk, so the structure is chosen to keep them off
// the critical path and few per unit of work: 64-query stages (a group's fixed per-stage cost ~600 clk against ~700 of math, which
// the other group's math covers), each group's S^T | dP^T buffer is released as soon as the group has loaded it (the next S^T of
// that group goes out during its math), each group rewrites its P^T | dS^T buffer only a stage later, and the S^T and gradient GEMMs
// have their own issuing warps.  TMEM: S^T | dP^T of group g at 128 g (64 + 64 fp32), P^T | dS^T of group g at 256 + 64 g
// (32 + 32 bf16 pairs), dV_r / dK_r at 384 + 64 r.  K / V stay in shared memory (double-buffered across tasks).
constexpr int KR = 2;
constexpr int KB_QW = 64, KB_QT = KB_QW * D, KB_BT = BN * KB_QW;   // Q / dO sub-tile [64][32] (4 KiB), bias^T [2 halves][128 k][32 q] fp32
constexpr int KB_NQ = 6, KB_NB = 2, KB_THREADS = 352;              // 0 TMA, 1 S^T MMA, 2-5 / 6-9 grad groups (rows 0 / 1), 10 grad MMA
constexpr int KB_COL_P = 256, KB_COL_G = 384;
constexpr int KB_SMEM = 1024 + (2 * KR * 2 * KVT + KR * 2 * KVT + KB_NQ * 2 * KB_QT) * 2 + KB_NB * KB_BT * 4 + KB_NQ * 2 * KB_QW * 4 + 512;
constexpr uint32_t ID_ST = idesc_bf16(128, KB_QW, 0, 0);
constexpr uint32_t ID_G = idesc_bf16(128, D, 0, 1);
static_assert(KB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(KB_THREADS, 1) triattn_bwd_kv_sm100(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][H*32], box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias^T * log2e fp32 [B*H*S (k)][S (q)], box (32, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,   // [B, N, H, S]
    const __grid_constant__ CUtensorMap dkmap,           // dK / dV [B*N*S][H*32] bf16, box (32, 128), 64B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dvmap) {
  const int NK = S / BN, NG = (N + KR - 1) / KR, NQS = S / KB_QW, SPT = KR * NQS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 task buffers][KR][K | V]
  __nv_bfloat16* sO = sKV + 2 * KR * 2 * KVT;                    // [KR][dV | dK] out tiles for the TMA stores
  __nv_bfloat16* sQ = sO + KR * 2 * KVT;                         // [NQ][Q | dO]
  float* sB = reinterpret_cast<float*>(sQ + KB_NQ * 2 * KB_QT);  // [NB][2 q halves][128 k][32 q]
  float* sLD = sB + KB_NB * KB_BT;                               // [NQ][lse 64 | delta 64]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLD + KB_NQ * 2 * KB_QW);
  uint64_t* kvf = bars;              // [2]  a task buffer's K / V landed
  uint64_t* kve = kvf + 2;           // [2]  its last S^T GEMMs retired
  uint64_t* dfr = kve + 2;           // [KR] row r's dK / dV final
  uint64_t* de = dfr + KR;           // [KR] count 4: row r drained
  uint64_t* qf = de + KR;            // [NQ]
  uint64_t* qe = qf + KB_NQ;         // [NQ]
  uint64_t* bf = qe + KB_NQ;         // [NB]
  uint64_t* be = bf + KB_NB;         // [NB] count 8
  uint64_t* sf = be + KB_NB;         // [2]  group g's S^T | dP^T ready
  uint64_t* se = sf + 2;             // [2]  count 4: group g loaded it
  uint64_t* pf = se + 2;             // [2]  count 4: group g's P^T | dS^T written
  uint64_t* ge = pf + 2;             // [2]  group g's gradient GEMMs retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ge + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) {
      bar_init(kvf + i, 1); bar_init(kve + i, 1); bar_init(dfr + i, 1); bar_init(de + i, 4);
      bar_init(sf + i, 1); bar_init(se + i, 4); bar_init(pf + i, 4); bar_init(ge + i, 1);
    }
    for (int i = 0; i < KB_NQ; ++i) { bar_init(qf + i, 1); bar_init(qe + i, 1); }
    for (int i = 0; i < KB_NB; ++i) { bar_init(bf + i, 1); bar_init(be + i, 8); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const uint32_t sbase = sa(smb);                  // shared-window address of p: sbase + (p - smb), no S2R per use
  auto A = [&](const void* p) -> uint32_t { return sbase + (uint32_t)(reinterpret_cast<const unsigned char*>(p) - smb); };
  auto decode = [&](int t, int& kt, int& g, int& bh) { kt = t % NK; g = (t / NK) % NG; bh = t / (NK * NG); };
  auto nrow = [&](int g, int r) { return min(g * KR + r, N - 1); };   // rows past N compute on row N-1 and store nothing
  const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  const int total = mytasks * SPT;

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int kt, g, bh; decode(t, kt, g, bh);
        const int b_ = bh / H, h_ = bh % H, kb = lt & 1;
        if (lt >= 2) wait(A(kve + kb), ((lt >> 1) - 1) & 1);
        expect_tx(A(kvf + kb), KR * 2 * KVT * 2);
        for (int r = 0; r < KR; ++r) {
          const int rw = (b_ * N + nrow(g, r)) * S + kt * BN;
          load_2d(&kmap, A(sKV + (kb * KR + r) * 2 * KVT), A(kvf + kb), h_ * D, rw);
          load_2d(&vmap, A(sKV + (kb * KR + r) * 2 * KVT + KVT), A(kvf + kb), h_ * D, rw);
        }
        for (int w = 0; w < SPT; ++w, ++y) {
          const int qs = w >> 1, r = w & 1, sl = y % KB_NQ;
          if (r == 0) {
            const int jb = lt * NQS + qs, bs = jb % KB_NB;
            if (jb >= KB_NB) wait(A(be + bs), ((jb / KB_NB) - 1) & 1);
            if (TA_BIAS1 && jb >= KB_NB) arrive(A(bf + bs));   // timing only
            else {
            expect_tx(A(bf + bs), KB_BT * 4);
            for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, A(sB + bs * KB_BT + hh * BN * 32), A(bf + bs), qs * KB_QW + hh * 32, bh * S + kt * BN);
            }
          }
          if (y >= KB_NQ) wait(A(qe + sl), ((y / KB_NQ) - 1) & 1);
          if (TA_QD1 && y >= KB_NQ) { arrive(A(qf + sl)); continue; }   // timing only
          expect_tx(A(qf + sl), 2 * KB_QT * 2 + 2 * KB_QW * 4);
          const int rw = (b_ * N + nrow(g, r)) * S + qs * KB_QW;
          load_2d(&qmap, A(sQ + sl * 2 * KB_QT), A(qf + sl), h_ * D, rw);
          load_2d(&dmap, A(sQ + sl * 2 * KB_QT + KB_QT), A(qf + sl), h_ * D, rw);
          const size_t st = ((size_t)(b_ * N + nrow(g, r)) * H + h_) * S + qs * KB_QW;
          bulk_load(A(sLD + sl * 2 * KB_QW), LSE + st, KB_QW * 4, A(qf + sl));
          bulk_load(A(sLD + sl * 2 * KB_QW + KB_QW), DELTA + st, KB_QW * 4, A(qf + sl));
        }
      }
    }
  } else if (warp == 1 || warp == 10) {
    // Issue loops: the converged warp runs them, one elected lane's tcgen05 ops are predicated (no branch / warp sync per group);
    // ring slots and phase bits are incremental counters (no divisions); barrier addresses and descriptor bases precomputed.
    const uint32_t L = elect_one() ? 1u : 0u;
    constexpr uint64_t SLOT = 2 * KB_QT * 2 / 16, MAT = KB_QT * 2 / 16, KVM = KVT * 2 / 16, KVROW = 2 * KVT * 2 / 16;
    const uint32_t qfa = A(qf), qea = A(qe), sfa = A(sf), sea = A(se), pfa = A(pf), gea = A(ge), dea = A(de), dfa = A(dfr);
    int sl = 0, slp = 0, w = 0, lt = 0, gp = 0;          // ring slot / phase, task-local stage, task, phase of the group barriers
    if (warp == 1) {
      // S^T of a stage goes out once its Q | dO | lse | delta slot and its bias tile landed, so the grad group's S^T barrier implies
      // both (release / acquire through this thread): one barrier a stage for the group.
      const uint64_t dQ0 = desc_k64(sQ), dKV0 = desc_k64(sKV);
      const uint32_t bfa = A(bf), kvfa = A(kvf), kvea = A(kve);
      int bs = 0, bsp = 0;
      for (int y = 0; y < total; ++y) {
        const int r = w & 1, kb = lt & 1;
        TL(0, y);
        const uint32_t okq = probe(qfa + sl * 8, slp);
        const uint32_t oke = y >= 2 ? probe(sea + r * 8, gp ^ 1) : 1u;     // group r loaded its previous S^T
        const uint32_t okb = r == 0 ? probe(bfa + bs * 8, bsp) : 1u;
        if (w == 0) wait(kvfa + kb * 8, (lt >> 1) & 1);
        if (!okq) wait(qfa + sl * 8, slp);
        if (!oke) wait(sea + r * 8, gp ^ 1);
        if (!okb) wait(bfa + bs * 8, bsp);
        tc_fence_after();
        const uint64_t bq = dQ0 + sl * SLOT, ak = dKV0 + (kb * KR + r) * KVROW;
        const uint32_t d = tmem + r * 128;
        mma_ss_if(L, d, ak, bq, ID_ST, 0u);
        mma_ss_if(L, d, ak + 2, bq + 2, ID_ST, 1u);
        mma_ss_if(L, d + 64, ak + KVM, bq + MAT, ID_ST, 0u);
        mma_ss_if(L, d + 64, ak + KVM + 2, bq + MAT + 2, ID_ST, 1u);
        TL(1, y);
        mma_commit_if(L, sfa + r * 8);
        if (w == SPT - 1) mma_commit_if(L, kvea + kb * 8);  // the task's K / V buffer is free once these retire
        if (r == 1) { if (++bs == KB_NB) { bs = 0; bsp ^= 1; } gp ^= (y >= 1); }
        if (y == 0) gp = 0;
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    } else {
      const uint64_t dM0 = sdesc(sa(sQ), KB_QT * 2, 512, 4);
      const uint32_t tG = tmem + KB_COL_G;
      for (int p = 0; p < total; ++p) {
        const int r = w & 1;                               // = the group
        const bool first = w < 2, last = w >= SPT - 2;
        TL(2, p);
        const uint32_t okp = probe(pfa + r * 8, gp);
        if (first && lt >= 1) wait(dea + r * 8, (lt - 1) & 1);   // the row's accumulators of the previous task were drained
        if (!okp) wait(pfa + r * 8, gp);
        TL(3, p);
        tc_fence_after();
        const uint64_t bm = dM0 + sl * SLOT;
        const uint32_t acc = first ? 0u : 1u, a = tmem + KB_COL_P + r * 64, g0 = tG + r * 64;
#pragma unroll
        for (int kk = 0; kk < KB_QW / 16; ++kk) {
          mma_ts_if(L, g0, a + kk * 8, bm + MAT + kk * 64, ID_G, (acc | kk) ? 1u : 0u);        // dV_r += P^T dO
          mma_ts_if(L, g0 + 32, a + 32 + kk * 8, bm + kk * 64, ID_G, (acc | kk) ? 1u : 0u);    // dK_r += dS^T Q
        }
        mma_commit_if(L, qea + sl * 8);
        mma_commit_if(L, gea + r * 8);
        if (last) mma_commit_if(L, dfa + r * 8);
        if (r == 1) gp ^= 1;
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    }
    __syncwarp();
  } else if (warp < 10) {
    const int qq = warp & 3, k = qq * 32 + lane;   // TMEM lane = key row of the tile
    const int gi = (warp - 2) >> 2;                // this group's row
    const bool lead = qq == 2 && lane == 0;        // warps 2 / 6: the group's TMA-issuing thread
    const float2 sc2 = make_float2(scl, scl), neg1 = make_float2(-1.f, -1.f);
    const uint32_t tS = tmem_at(tmem + gi * 128, qq * 32, 0), tP = tmem_at(tmem + KB_COL_P + gi * 64, qq * 32, 0);
    int js = 0, lt = 0;                            // this group's stage count (phase of sf / se / pf / ge), task
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int kt, g, bh; decode(t, kt, g, bh);
      const int b_ = bh / H, h_ = bh % H;
      for (int qs = 0; qs < NQS; ++qs, ++js) {
        const int jb = lt * NQS + qs, bs = jb % KB_NB, y = lt * SPT + qs * 2 + gi, sl = y % KB_NQ;
        if (lane == 0 && qq == 2) TL(4 + gi * 6, y);
        wait(A(sf + gi), js & 1);                   // implies the bias tile and the lse / delta slot (see the S^T issuer)
        if (lane == 0 && qq == 2) TL(5 + gi * 6, y);
        tc_fence_after();
        const float* ld = sLD + sl * 2 * KB_QW;
#pragma unroll
        for (int c = 0; c < 2; ++c) {              // 32-query chunks (registers)
          float sv[32], dp[32], bl[32];
#if TA_NOLD
#pragma unroll
          for (int i = 0; i < 32; ++i) { sv[i] = __int_as_float(tS + i + c); dp[i] = sv[i] * 0.5f; }
#else
          tmem_ld32(tS + c * 32, sv);
          tmem_ld32(tS + 64 + c * 32, dp);
#endif
#if TA_NOLDS
#pragma unroll
          for (int i = 0; i < 32; ++i) bl[i] = sv[i] * 0.25f;
#else
          {
            const float* bp = sB + bs * KB_BT + c * BN * 32 + k * 32;
#pragma unroll
            for (int q4 = 0; q4 < 8; ++q4) {
              const float4 u = *reinterpret_cast<const float4*>(bp + ((q4 ^ (k & 7)) << 2));
              bl[q4 * 4] = u.x; bl[q4 * 4 + 1] = u.y; bl[q4 * 4 + 2] = u.z; bl[q4 * 4 + 3] = u.w;
            }
          }
#endif
          tmem_wait_ld();
          if (c == 1) {                             // both chunks loaded: the group's S^T buffer takes its next stage
            tc_fence_before();
            __syncwarp();
            if (lane == 0) { arrive(A(se + gi)); arrive(A(be + bs)); }
          }
          uint32_t pk[16], dk[16];
#if TA_NOMATH
#pragma unroll
          for (int i = 0; i < 16; ++i) { pk[i] = __float_as_uint(sv[2 * i] + bl[i]); dk[i] = __float_as_uint(dp[2 * i + 1]); }
          if (ld[c] == 1.2345f) pk[0] = 0;
#else
#pragma unroll
          for (int j = 0; j < 8; ++j) {
#if TA_NOLDS
            const float4 l4 = make_float4(bl[j], bl[j + 1], bl[j + 2], bl[j + 3]), d4 = make_float4(dp[j], dp[j + 8], dp[j + 16], dp[j + 24]);
#else
            const float4 l4 = *reinterpret_cast<const float4*>(ld + c * 32 + 4 * j);
            const float4 d4 = *reinterpret_cast<const float4*>(ld + KB_QW + c * 32 + 4 * j);
#endif
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              const int i = 4 * j + 2 * h2;
              const float2 lm = h2 ? make_float2(l4.z, l4.w) : make_float2(l4.x, l4.y);
              const float2 dm = h2 ? make_float2(d4.z, d4.w) : make_float2(d4.x, d4.y);
              // FADD2 has no negate: -lse / -delta go through FFMA2 with -1 (else two FADD negations a pair)
              const float2 x = fma2(lm, neg1, fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])));
              const float2 pp = make_float2(ex2f(x.x), ex2f(x.y));
              const float2 ds = mul2(pp, fma2(dm, neg1, make_float2(dp[i], dp[i + 1])));
              pk[i >> 1] = pack2(pp.x, pp.y);
              dk[i >> 1] = pack2(ds.x, ds.y);
            }
          }
#endif
          if (c == 0 && js >= 1) wait(A(ge + gi), (js - 1) & 1);   // this group's previous gradient GEMMs read the P^T | dS^T buffer
#if TA_NOST
          if (pk[3] == 0x12345u && dk[7] == 0x777u) {
#endif
          tmem_st8(tP + c * 16, pk); tmem_st8(tP + c * 16 + 8, pk + 8);
          tmem_st8(tP + 32 + c * 16, dk); tmem_st8(tP + 32 + c * 16 + 8, dk + 8);
#if TA_NOST
          }
#endif
        }
        if (lane == 0 && qq == 2) TL(6 + gi * 6, y);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(pf + gi));
        if (lane == 0 && qq == 2) TL(7 + gi * 6, y);
      }
      // ---- dK / dV of row gi ----
      wait(A(dfr + gi), lt & 1);
      tc_fence_after();
      float v[32], kk[32];
      tmem_ld32(tmem_at(tmem + KB_COL_G + gi * 64, qq * 32, 0), v);
      tmem_ld32(tmem_at(tmem + KB_COL_G + gi * 64 + 32, qq * 32, 0), kk);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(A(de + gi));           // the accumulators are free for the next task
      if (lead) bulk_wait_read<0>();               // the previous task's stores have read the out tiles
      named_sync(1 + gi, 128);
      {
        uint32_t wv[16], wk[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) { wv[i] = pack2(v[2 * i], v[2 * i + 1]); wk[i] = pack2(kk[2 * i] * scale, kk[2 * i + 1] * scale); }
        row_sw64_st(sO + gi * 2 * KVT, k, wv);
        row_sw64_st(sO + gi * 2 * KVT + KVT, k, wk);
      }
      fence_proxy_async();
      named_sync(1 + gi, 128);
      if (lead && g * KR + gi < N) {
        const int rw = (b_ * N + g * KR + gi) * S + kt * BN;
        store_2d(&dvmap, A(sO + gi * 2 * KVT), h_ * D, rw);
        store_2d(&dkmap, A(sO + gi * 2 * KVT + KVT), h_ * D, rw);
        bulk_commit();
      }
    }
    if (lead) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- kernel 2: dQ and dbias.  Task = (b, h, 128-query tile) x QR pair rows, all keys in 64-wide sub-tiles (stage = (sub-tile, row)).
//   MMA:   S = Q_r . K^T, dP = dO_r . V^T        (ts: A = Q_r / dO_r from TMEM, B = K / V K-major; M = 128 q, N = 64 k, K = 32)
//   grad:  dS (bf16) over dP; dbias summed over the QR rows in registers   (group gi: key columns 32 gi .. of the stage)
//   MMA:   dQ_r += dS . K                        (ts: A = dS from TMEM, B = K MN-major; N = 32)
constexpr int QR = 4;
constexpr int QB_KW = 64, QB_KT = QB_KW * D, QB_BT = BM * QB_KW;   // K / V sub-tile [64][32] (4 KiB), bias [128 q][64 k] 128B swizzle
constexpr int QB_NKV = 6, QB_NB = 2, QB_NS = 2, QB_THREADS = 320;
constexpr int QB_COL_G = QB_NS * 128;                               // dQ_r at 256 + 32 r
constexpr int QB_COL_A = QB_COL_G + QR * 32;                        // Q_r at 384 + 16 r, dO_r at 384 + 16 QR + 16 r
constexpr int QB_DBT = BM * 32;                                   // a group's dbias partial [128 q][32 k] fp32, 128B swizzle (16 KiB)
constexpr int QB_SMEM = 1024 + (QR * 2 * QT + QB_NKV * 2 * QB_KT + QB_NB * QB_BT) * 2 + QR * 2 * BM * 4 + 2 * 2 * QB_DBT * 4 + 256;
constexpr uint32_t ID_SQ = idesc_bf16(128, QB_KW, 0, 0);
static_assert(QB_COL_A + QR * 32 <= 512, "TMEM");
static_assert(QB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(QB_THREADS, 1) triattn_bwd_q_sm100(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][H*32], box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,
    const __grid_constant__ CUtensorMap dqmap,           // dQ [B*N*S][H*32] bf16, box (32, 128), 64B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dbmap) {         // dbias fp32 [B*H*S][S], zeroed; box (32, 128), 128B swizzle (TMA reduce-add)
  const int QTL = S / BM, NG = (N + QR - 1) / QR, NKS = S / QB_KW, SPT = QR * NKS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smb);    // [QR][Q | dO] staging for TMEM
  __nv_bfloat16* sKV = sA + QR * 2 * QT;                         // [NKV][K | V]
  __nv_bfloat16* sB = sKV + QB_NKV * 2 * QB_KT;                  // [NB][128 q][64 k]
  float* sLD = reinterpret_cast<float*>(sB + QB_NB * QB_BT);     // [QR][lse 128 | delta 128]
  float* sDB = sLD + QR * 2 * BM;                                // [2 groups][2 buffers][QB_DBT] dbias partials; the dQ out tiles
                                                                 // reuse a group's buffers at the task's end
  uint64_t* bars = reinterpret_cast<uint64_t*>(sDB + 2 * 2 * QB_DBT);
  uint64_t* af = bars;
  uint64_t* ae = af + 1;             // count 8
  uint64_t* at = ae + 1;             // count 8
  uint64_t* df = at + 1;
  uint64_t* kvf = df + 1;            // [NKV]
  uint64_t* kve = kvf + QB_NKV;      // [NKV]
  uint64_t* bf = kve + QB_NKV;       // [NB]
  uint64_t* be = bf + QB_NB;         // [NB] count 8
  uint64_t* sf = be + QB_NB;         // [NS]
  uint64_t* pf = sf + QB_NS;         // [NS] count 8
  uint32_t* tslot = reinterpret_cast<uint32_t*>(pf + QB_NS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    bar_init(af, 1); bar_init(ae, 8); bar_init(at, 8); bar_init(df, 1);
    for (int i = 0; i < QB_NKV; ++i) { bar_init(kvf + i, 1); bar_init(kve + i, 1); }
    for (int i = 0; i < QB_NB; ++i) { bar_init(bf + i, 1); bar_init(be + i, 8); }
    for (int i = 0; i < QB_NS; ++i) { bar_init(sf + i, 1); bar_init(pf + i, 8); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  if (warp >= 2 && (warp & 3) == 2 && lane == 0) bulk_wait<0>();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const uint32_t sbase = sa(smb);                  // shared-window address of p: sbase + (p - smb), no S2R per use
  auto A = [&](const void* p) -> uint32_t { return sbase + (uint32_t)(reinterpret_cast<const unsigned char*>(p) - smb); };
  auto decode = [&](int t, int& qt, int& g, int& bh) { qt = t % QTL; g = (t / QTL) % NG; bh = t / (QTL * NG); };
  auto nrow = [&](int g, int r) { return min(g * QR + r, N - 1); };

  if (warp == 0) {
    if (lane == 0) {
      auto staging = [&](int t) {                // Q / dO rows, lse, delta of task t -> staging
        int qt, g, bh; decode(t, qt, g, bh);
        const int b_ = bh / H, h_ = bh % H;
        expect_tx(A(af), QR * 2 * QT * 2 + QR * 2 * BM * 4);
        for (int r = 0; r < QR; ++r) {
          const int rw = (b_ * N + nrow(g, r)) * S + qt * BM;
          load_2d(&qmap, A(sA + r * 2 * QT), A(af), h_ * D, rw);
          load_2d(&dmap, A(sA + r * 2 * QT + QT), A(af), h_ * D, rw);
          const size_t st = ((size_t)(b_ * N + nrow(g, r)) * H + h_) * S + qt * BM;
          bulk_load(A(sLD + r * 2 * BM), LSE + st, BM * 4, A(af));
          bulk_load(A(sLD + r * 2 * BM + BM), DELTA + st, BM * 4, A(af));
        }
      };
      if ((int)blockIdx.x < ntasks) staging(blockIdx.x);
      const int st_at = min(SPT, QB_NKV) - 1;
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int qt, g, bh; decode(t, qt, g, bh);
        const int b_ = bh / H, h_ = bh % H;
        for (int w = 0; w < SPT; ++w, ++y) {
          if (w == st_at && t + (int)gridDim.x < ntasks) { wait(A(ae), lt & 1); staging(t + gridDim.x); }
          const int ks = w / QR, r = w % QR, sl = y % QB_NKV;
          if (r == 0) {
            const int jb = lt * NKS + ks, bs = jb % QB_NB;
            if (jb >= QB_NB) wait(A(be + bs), ((jb / QB_NB) - 1) & 1);
            expect_tx(A(bf + bs), QB_BT * 2);
            load_2d(&bmap, A(sB + bs * QB_BT), A(bf + bs), ks * QB_KW, bh * S + qt * BM);
          }
          if (y >= QB_NKV) wait(A(kve + sl), ((y / QB_NKV) - 1) & 1);
          expect_tx(A(kvf + sl), 2 * QB_KT * 2);
          const int rw = (b_ * N + nrow(g, r)) * S + ks * QB_KW;
          load_2d(&kmap, A(sKV + sl * 2 * QB_KT), A(kvf + sl), h_ * D, rw);
          load_2d(&vmap, A(sKV + sl * 2 * QB_KT + QB_KT), A(kvf + sl), h_ * D, rw);
        }
      }
    }
  } else if (warp == 1) {
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    const uint64_t dK0 = desc_k64(sKV), dM0 = sdesc(sa(sKV), QB_KT * 2, 512, 4);
    constexpr uint64_t SLOT = 2 * QB_KT * 2 / 16, MAT = QB_KT * 2 / 16;
    auto stt = [&](int y, int lt, int w) {
      const int r = w % QR, sl = y % QB_NKV, b = y % QB_NS;
      if (w == 0) wait(A(at), lt & 1);
      wait(A(kvf + sl), (y / QB_NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t bk = dK0 + sl * SLOT;
        const uint32_t d = tmem + b * 128, a = tmem + QB_COL_A + r * 16;
        mma_ts(d, a, bk, ID_SQ, 0u);
        mma_ts(d, a + 8, bk + 2, ID_SQ, 1u);
        mma_ts(d + 64, a + QR * 16, bk + MAT, ID_SQ, 0u);
        mma_ts(d + 64, a + QR * 16 + 8, bk + MAT + 2, ID_SQ, 1u);
        mma_commit(A(sf + b));
      }
      __syncwarp();
    };
    auto grad = [&](int y, int w) {
      const int kb = w / QR, r = w % QR, sl = y % QB_NKV, b = y % QB_NS;
      wait(A(pf + b), (y / QB_NS) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t bm = dM0 + sl * SLOT;
        const uint32_t acc = kb ? 1u : 0u, d = tmem + QB_COL_G + r * 32;
#pragma unroll
        for (int ks = 0; ks < QB_KW / 16; ++ks) {
          const uint32_t a = tmem + b * 128 + 64 + (ks >> 1) * 32 + (ks & 1) * 8;
          mma_ts(d, a, bm + ks * 64, ID_G, (acc | ks) ? 1u : 0u);
        }
        mma_commit(A(kve + sl));
        if (w == SPT - 1) mma_commit(A(df));
      }
      __syncwarp();
    };
    int y = 0, ylt = 0, yw = 0, pw = 0;
    for (int p = 0; p < total; ++p) {
      // a task's first S waits for its TMEM operands, which the grad warps copy only after draining the previous task's
      // accumulators: issue it after the previous task's last gradient GEMM
      while (y < total && y < p + QB_NS && (yw != 0 || y == p)) {
        stt(y, ylt, yw);
        ++y;
        if (++yw == SPT) { yw = 0; ++ylt; }
      }
      grad(p, pw);
      if (++pw == SPT) pw = 0;
    }
  } else {
    const int qq = warp & 3, row = qq * 32 + lane;   // TMEM lane = query row of the tile
    const int gi = (warp - 2) >> 2, c0 = gi * 32;    // key columns of a stage; rows gi, gi + 2 for the Q / dO copy and dQ drain
    const float2 sc2 = make_float2(scl, scl);
    const bool lead = (warp & 3) == 2 && lane == 0;   // the group's TMA-issuing thread (warps 2 / 6)
    float* myDB = sDB + gi * 2 * QB_DBT;
    int nb = 0;                                       // dbias buffers used by this group so far
    int lt = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int qt, g, bh; decode(t, qt, g, bh);
      const int b_ = bh / H;
      float lse[QR], dl[QR];
      {
        wait(A(af), lt & 1);
        const uint32_t ta = tmem_at(tmem + QB_COL_A, qq * 32, 0);
#pragma unroll
        for (int rr = 0; rr < QR / 2; ++rr) {
          const int r = gi + 2 * rr;
          uint32_t wq[16], wd[16];
          row_sw64(sA + r * 2 * QT, row, wq);
          row_sw64(sA + r * 2 * QT + QT, row, wd);
          tmem_st8(ta + r * 16, wq); tmem_st8(ta + r * 16 + 8, wq + 8);
          tmem_st8(ta + QR * 16 + r * 16, wd); tmem_st8(ta + QR * 16 + r * 16 + 8, wd + 8);
        }
#pragma unroll
        for (int r = 0; r < QR; ++r) { lse[r] = sLD[r * 2 * BM + row]; dl[r] = sLD[r * 2 * BM + BM + row]; }
        __syncwarp();
        if (lane == 0) arrive(A(ae));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(at));
      }
      for (int kb = 0; kb < NKS; ++kb) {
        const int jb = lt * NKS + kb, bs = jb % QB_NB;
        wait(A(bf + bs), (jb / QB_NB) & 1);
        float bl[32], dacc[32];
        {
          const __nv_bfloat16* brow = sB + bs * QB_BT + row * 64;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint4 u = *reinterpret_cast<const uint4*>(brow + (((gi * 4 + e) ^ (row & 7)) << 3));
            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
              const float2 f = bf2f(w4[k2]);
              bl[e * 8 + k2 * 2] = f.x * L2E; bl[e * 8 + k2 * 2 + 1] = f.y * L2E;
            }
          }
#pragma unroll
          for (int i = 0; i < 32; ++i) dacc[i] = 0.f;
        }
        __syncwarp();
        if (lane == 0) arrive(A(be + bs));
#pragma unroll
        for (int r = 0; r < QR; ++r) {
          const int y = lt * SPT + kb * QR + r, b = y % QB_NS;
          const bool valid = g * QR + r < N;
          wait(A(sf + b), (y / QB_NS) & 1);
          tc_fence_after();
          const uint32_t tb = tmem_at(tmem + b * 128, qq * 32, 0);
          float sv[32], dp[32];
          tmem_ld32(tb + c0, sv);
          tmem_ld32(tb + 64 + c0, dp);
          tmem_wait_ld();
          const float2 nl = make_float2(-lse[r], -lse[r]), nd = make_float2(-dl[r], -dl[r]);
          const float vm = valid ? 1.f : 0.f;       // padding rows add nothing to dbias
          const float2 vm2 = make_float2(vm, vm);
          uint32_t dk[16];
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const float2 x = add2(fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])), nl);
            const float2 pp = make_float2(ex2f(x.x), ex2f(x.y));
            const float2 ds = mul2(pp, add2(make_float2(dp[i], dp[i + 1]), nd));
            dk[i >> 1] = pack2(ds.x, ds.y);
            const float2 a2 = fma2(ds, vm2, make_float2(dacc[i], dacc[i + 1]));
            dacc[i] = a2.x; dacc[i + 1] = a2.y;
          }
          tmem_st8(tb + 64 + c0, dk); tmem_st8(tb + 64 + c0 + 8, dk + 8);
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(A(pf + b));
        }
        // the group's [128 q][32 k] partial -> shared -> one TMA reduce-add (per-thread 16-byte reductions touch 32 lines each)
        {
          float* buf = myDB + (nb & 1) * QB_DBT;
          if (lead) bulk_wait_read<1>();              // the reduce that used this buffer two rounds ago has read it
          named_sync(1 + gi, 128);
#pragma unroll
          for (int c = 0; c < 8; ++c)
            *reinterpret_cast<float4*>(buf + row * 32 + ((c ^ (row & 7)) << 2)) = make_float4(dacc[4 * c], dacc[4 * c + 1], dacc[4 * c + 2], dacc[4 * c + 3]);
          fence_proxy_async();
          named_sync(1 + gi, 128);
          if (lead) { reduce_add_2d(&dbmap, A(buf), kb * QB_KW + c0, bh * S + qt * BM); bulk_commit(); }
          ++nb;
        }
      }
      // ---- dQ of rows gi, gi + 2 ----
      wait(A(df), lt & 1);
      tc_fence_after();
      float a[QR / 2][32];
#pragma unroll
      for (int rr = 0; rr < QR / 2; ++rr) tmem_ld32(tmem_at(tmem + QB_COL_G + (gi + 2 * rr) * 32, qq * 32, 0), a[rr]);
      tmem_wait_ld();
      tc_fence_before();
      // dQ rows gi, gi + 2 -> the group's dbias buffers (both reduces retired) -> TMA stores
      __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(myDB);
      if (lead) bulk_wait_read<0>();
      named_sync(1 + gi, 128);
#pragma unroll
      for (int rr = 0; rr < QR / 2; ++rr) {
        uint32_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = pack2(a[rr][2 * i] * scale, a[rr][2 * i + 1] * scale);
        row_sw64_st(ob + rr * QT, row, w);
      }
      fence_proxy_async();
      named_sync(1 + gi, 128);
      if (lead) {
#pragma unroll
        for (int rr = 0; rr < QR / 2; ++rr) {
          const int r = gi + 2 * rr;
          if (g * QR + r < N) store_2d(&dqmap, A(ob + rr * QT), (bh % H) * D, (b_ * N + g * QR + r) * S + qt * BM);
        }
        bulk_commit();
      }
      nb = 1;                                         // buffer 0 holds the dQ tiles: the next partial goes to buffer 1, whose
                                                      // wait_read<1> may leave this store pending; buffer 0's then retires it
    }
  }
  if (warp >= 2 && (warp & 3) == 2 && lane == 0) bulk_wait<0>();
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

// q, k, v [B, N, S, H, 32] bf16 contiguous (the projection's layout: row (b, n, s) holds all heads); bias [B, H, S, S] (any float
// dtype; staged to bf16 when it is not); S % 128 == 0; no mask.
// -> (out [B, N, S, H, 32] bf16, lse fp32 [B, N, H, S] base 2 (empty unless want_lse), flags int32 [1]: rows whose offset overflowed)
std::vector<torch::Tensor> triattn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, double scale, bool want_lse) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 32] bf16");
  for (const auto* t : {&q, &k, &v}) TORCH_CHECK(t->is_contiguous() && t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16, "q, k, v: contiguous, same shape");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.numel() == B * H * S * S, "bias: [B, H, S, S]");
  auto b16 = bias.scalar_type() == torch::kBFloat16 ? bias.contiguous() : bias.to(torch::kBFloat16).contiguous();
  auto out = torch::empty_like(q);
  auto lse = want_lse ? torch::empty({B, N, H, S}, q.options().dtype(torch::kFloat32)) : torch::empty({0}, q.options().dtype(torch::kFloat32));
  auto flags = torch::zeros({1}, q.options().dtype(torch::kInt32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  CUtensorMap qm = make_map<2>(q.data_ptr(), {cols, rows}, {cols}, {D, BM}, CU_TENSOR_MAP_SWIZZLE_64B, "q");
  CUtensorMap km = make_map<2>(k.data_ptr(), {cols, rows}, {cols}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_64B, "k");
  CUtensorMap vm = make_map<2>(v.data_ptr(), {cols, rows}, {cols}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_64B, "v");
  CUtensorMap bm = make_map<2>(b16.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  const int ntasks = (int)(B * H * (S / BM) * ((N + R - 1) / R));
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_fwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  const float scl = (float)scale * L2E;
  triattn_fwd_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)H, (int)S, ntasks, scl, qm, km, vm, bm,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags};
}

// Backward of triattn_fwd.  q, k, v, dout [B, N, S, H, 32] bf16 (the projection layout); bias [B, H, S, S]; lse / delta fp32
// [B, N, H, S] (lse: triattn_fwd's base-2 LSE; delta = rowsum(dout o out)).
// -> (dq, dk, dv [B, N, S, H, 32] bf16, dbias fp32 [B, H, S, S], dS^T bf16 [B, N, H, S, S] scratch)
#if TA_TL
torch::Tensor triattn_tl() { auto t = torch::empty({24, 256}, torch::kInt64); C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), g_tl, sizeof(g_tl))); return t; }
#endif
std::vector<torch::Tensor> triattn_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor dout,
                                       torch::Tensor lse, torch::Tensor delta, double scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 32] bf16");
  for (const auto* t : {&q, &k, &v, &dout}) TORCH_CHECK(t->is_contiguous() && t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16, "q, k, v, dout: contiguous, same shape");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.numel() == B * H * S * S, "bias: [B, H, S, S]");
  for (const auto* t : {&lse, &delta}) TORCH_CHECK(t->scalar_type() == torch::kFloat32 && t->is_contiguous() && t->numel() == B * N * H * S, "lse / delta: fp32 [B, N, H, S]");
  auto b16 = bias.scalar_type() == torch::kBFloat16 ? bias.contiguous() : bias.to(torch::kBFloat16).contiguous();
  auto dq = torch::empty_like(q), dk = torch::empty_like(q), dv = torch::empty_like(q);
  auto db = torch::zeros({B, H, S, S}, q.options().dtype(torch::kFloat32));
  CUtensorMap dbm = make_map<2>(db.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "dbias",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  auto m2 = [&](const torch::Tensor& t, uint32_t box, const char* what) { return make_map<2>(t.data_ptr(), {cols, rows}, {cols}, {D, box}, CU_TENSOR_MAP_SWIZZLE_64B, what); };
  CUtensorMap q32 = m2(q, KB_QW, "q"), d32 = m2(dout, KB_QW, "dout"), k128 = m2(k, 128, "k"), v128 = m2(v, 128, "v");   // (q32: box KB_QW rows)
  CUtensorMap q128 = m2(q, 128, "q"), d128 = m2(dout, 128, "dout"), k64 = m2(k, 64, "k"), v64 = m2(v, 64, "v");
  CUtensorMap dkm = m2(dk, 128, "dk"), dvm = m2(dv, 128, "dv"), dqm = m2(dq, 128, "dq");
  auto bT = torch::empty({B, H, S, S}, q.options().dtype(torch::kFloat32));
  CUtensorMap bkv = make_map<2>(bT.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "bias^T",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap bq = make_map<2>(b16.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {QB_KW, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd_kv_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, KB_SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd_q_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, QB_SMEM));
    attr = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sms = num_sms(q.device().index());
  const float scl = (float)scale * L2E;
  triattn_biasT<<<dim3(S / 32, S / 32, B * H), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(b16.data_ptr<at::BFloat16>()), bT.data_ptr<float>(), (int)S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int n1 = (int)(B * H * (S / BN) * ((N + KR - 1) / KR));
  triattn_bwd_kv_sm100<<<std::min(n1, sms), KB_THREADS, KB_SMEM, stream>>>((int)N, (int)H, (int)S, n1, scl, (float)scale, q32, d32, k128, v128, bkv,
      lse.data_ptr<float>(), delta.data_ptr<float>(), dkm, dvm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int n2 = (int)(B * H * (S / BM) * ((N + QR - 1) / QR));
  triattn_bwd_q_sm100<<<std::min(n2, sms), QB_THREADS, QB_SMEM, stream>>>((int)N, (int)H, (int)S, n2, scl, (float)scale, q128, d128, k64, v64, bq,
      lse.data_ptr<float>(), delta.data_ptr<float>(), dqm, dbm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd", &triattn_fwd, "sm100 triangle-attention forward (D = 32, bf16, no mask): q/k/v/out [B, N, S, H, 32], optional base-2 LSE",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"), py::arg("want_lse") = false);
#if TA_TL
  m.def("triattn_tl", &triattn_tl);
#endif
  m.def("triattn_bwd", &triattn_bwd, "sm100 triangle-attention backward: (dq, dk, dv [B, N, S, H, 32] bf16, dbias fp32 [B, H, S, S])",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("dout"), py::arg("lse"), py::arg("delta"), py::arg("scale"));
}
