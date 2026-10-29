// Triangle attention for B200 (sm_100a) with 128-channel heads (and 96-channel heads, zero-padded to 128 in the packed weights: their
// padded q / k / v / g channels are 0, the softmax scale stays 1/sqrt(96)): the d_pair 384 / 512 widths at AF3's four heads (4 x 96,
// 4 x 128), or any heads x 128 <= 512.  bf16 operands, fp32 softmax and accumulation, bf16 out; the forward (with the base-2 LSE for
// training) and its backward (below the forward):
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]        (masked keys: bias = bf16 min)
//
// The head-dim-64 core's schedule (triattn_dh64_sm100.cu) planned for the tensor-memory budget of a 128-channel head: an accumulator row
// takes 128 of the 512 TMEM columns, so the forward keeps two pair rows a task but walks the keys in 64-wide sub-tiles (S / P buffers of
// 64 columns, four of them), the key-side backward owns one pair row a task with 32-query stages, and the query-side backward one pair
// row a task with Q | dO resident in TMEM.  Q / K / V tiles are two 128-byte-swizzled 64-column halves; the QK^T products step through
// both halves (K = 128), the PV / gradient products take N = 128 as one MN-major operand whose halves are LBO apart.
//
// Forward: a CTA task = (b, h, 128-query tile) x R = 2 pair rows; per 64-key sub-tile j and row r:
//   MMA warp:      S = Q_r . K_{r,j}^T     (M = 128, N = 64, K = 128: eight k-steps, both operands from shared)
//   softmax warps: x = S * scale*log2e + bias * log2e; p = 2^(x - m_r) (m_r the running offset, raised lazily past TA_TAU); P over S
//   MMA warp:      O_r += P . V_{r,j}       (A = P from TMEM, B = V MN-major [64 k][128], N = 128)
// TMEM: four 64-column S / P buffers + O of the two rows (2 x 128) = 512 columns.  Shared memory: Q of the two rows (2 x 32 KiB, each
// row refilled as soon as the task's last QK^T of that row retired), four K | V stages (4 x 32 KiB), two bias tiles (2 x 16 KiB) and the
// row sums: 226.5 KiB, so the dynamic base carries no alignment slack (the kernel traps if it is not 1024-aligned).
// Warps: 0 = TMA, 1 = MMA, 2-5 / 6-9 = the two softmax groups (row 0 / row 1 of the task), 10-13 = epilogue (O / l -> bf16 out).
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

#ifndef TA_TAU
#define TA_TAU 8.f       // a chunk raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
namespace {
constexpr int D = 128, DH = 64, BM = 128, BN = 64, R = 2;
constexpr int QH = BM * DH;                                // a Q half [128 q][64], 128B swizzle (16 KiB)
constexpr int QT = 2 * QH;                                 // a row's Q (32 KiB)
constexpr int KH = BN * DH;                                // a K or V half [64 k][64] (8 KiB)
constexpr int KVS = 4 * KH;                                // a stage: K (two halves) | V (two halves), 32 KiB
constexpr int BT = BM * BN;                                // bias tile [128 q][64 k] bf16, 128B swizzle (16 KiB)
constexpr int NKV = 4, NB = 2, NS = 4;                     // K | V stages, bias stages, S / P TMEM buffers (NKV >= NS: QK^T runs NS - 1 ahead)
constexpr int THREADS = 448;
constexpr int SMEM = (R * QT + NKV * KVS + NB * BT) * 2 + 2 * R * BM * 4 + 512;
constexpr int COL_S = 0, COL_O = NS * BN;                  // S/P [NS][64] | O [R rows][128]
static_assert(COL_O + R * D <= 512, "TMEM");
static_assert(NKV >= NS, "the MMA warp's QK^T look-ahead needs a K | V stage each");
constexpr uint32_t ID_S = idesc_bf16(128, BN, 0, 0);       // A = Q K-major, B = K K-major
constexpr uint32_t ID_O = idesc_bf16(128, D, 0, 1);        // A = P from TMEM, B = V MN-major (N = 128: two 64-wide blocks)
constexpr float L2E = 1.4426950408889634f;
static_assert(SMEM <= 232448, "one CTA per SM");

__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__global__ void __launch_bounds__(THREADS, 1) triattn_fwd128_sm100(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][.. H*128] bf16, box (64, 128) at column h*128 + 64 half, 128B swizzle
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (64, 64)
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*128]
    float* __restrict__ LSE,                             // [B, N, H, S] base-2 log-sum-exp (training), or nullptr
    int* __restrict__ FLAGS) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();                         // no alignment slack (see the header)
  unsigned char* smb = raw;
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);   // [R][2 halves][QH]
  __nv_bfloat16* sKV = sQ + R * QT;                              // [NKV][K h0 | K h1 | V h0 | V h1]
  __nv_bfloat16* sB = sKV + NKV * KVS;                           // [NB][BT]
  float* sL = reinterpret_cast<float*>(sB + NB * BT);            // [2 tasks][R][128] row sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sL + 2 * R * BM);
  uint64_t* qf = bars;               // [R]   row r's Q landed
  uint64_t* qe = qf + R;             // [R]   the task's last QK^T of row r retired
  uint64_t* kvf = qe + R;            // [NKV]
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
    for (int k = 0; k < R; ++k) { bar_init(qf + k, 1); bar_init(qe + k, 1); }
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
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H; n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row
  auto hcol = [&](int bh) -> int { return (bh % H) * D; };

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1);
          expect_tx(bf + bs, BT * 2);
          load_2d(&bmap, sB + bs * BT, bf + bs, j * BN, bh * S + qt * BM);
          for (int r = 0; r < R; ++r, ++x) {
            if (j == 0) {                                // row r's Q: its buffer is free once the previous task's last QK^T of row r retired
              if (lt >= 1) wait(qe + r, (lt - 1) & 1);
              expect_tx(qf + r, QT * 2);
              for (int hh = 0; hh < 2; ++hh)
                load_2d(&qmap, sQ + r * QT + hh * QH, qf + r, hcol(bh) + hh * DH, row0(bh, g * R + r) + qt * BM);
            }
            const int st = x % NKV;
            if (x >= NKV) wait(kve + st, ((x / NKV) - 1) & 1);
            expect_tx(kvf + st, KVS * 2);
            const int rw = row0(bh, g * R + r) + j * BN;
            for (int hh = 0; hh < 2; ++hh) {
              load_2d(&kmap, sKV + st * KVS + hh * KH, kvf + st, hcol(bh) + hh * DH, rw);
              load_2d(&vmap, sKV + st * KVS + (2 + hh) * KH, kvf + st, hcol(bh) + hh * DH, rw);
            }
          }
        }
      }
    }
  } else if (warp == 1) {
    // flattened sub-steps y (task k = y / SPT, key sub-tile j, row r): QK^T runs up to NS - 1 sub-steps ahead of the PV GEMMs
    const int SPT = NK * R;
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    constexpr uint64_t QHD = QH * 2 / 16, KHD = KH * 2 / 16, RQD = QT * 2 / 16, STD = KVS * 2 / 16;
    const uint64_t dQ0 = desc_k128(sQ), dK0 = desc_k128(sKV), dV0 = desc_mn128(sKV + 2 * KH, KH * 2);
    auto qk = [&](int y) {
      const int k = y / SPT, w = y % SPT, r = w % R, st = y % NKV, b = y % NS;
      if (w < R) wait(qf + r, k & 1);
      wait(kvf + st, (y / NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t q = dQ0 + r * RQD, kk = dK0 + st * STD;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + COL_S + b * BN, q + (ks >> 2) * QHD + 2 * (ks & 3), kk + (ks >> 2) * KHD + 2 * (ks & 3), ID_S, ks ? 1u : 0u);
        mma_commit(sf + b);
        if (w >= SPT - R) mma_commit(qe + r);    // the task's last QK^T of row r: its Q buffer may be refilled
      }
      __syncwarp();
    };
    auto pv = [&](int y) {                       // O_r += P . V
      const int k = y / SPT, w = y % SPT, j = w / R, r = w % R, st = y % NKV, b = y % NS;
      wait(pf + b, (y / NS) & 1);
      if (w == 0 && k >= 1) wait(oe, (k - 1) & 1);   // the previous task's O was drained
      tc_fence_after();
      if (elect_one()) {
        const uint64_t v = dV0 + st * STD;
        const uint32_t o = tmem + COL_O + r * D;
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks)
          mma_ts(o, tmem + COL_S + b * BN + ks * 8, v + ks * (16 * DH * 2 / 16), ID_O, (j | ks) ? 1u : 0u);
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
    const int r = (warp - 2) >> 2;                 // this group's row of the task
    int lt = 0, x = 0, jb = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      float m = -INFINITY, l = 0.f;                // no finite logit seen yet
      for (int j = 0; j < NK; ++j, ++jb) {
        const int bs = jb % NB;
        wait(bf + bs, (jb / NB) & 1);
        const __nv_bfloat16* brow = sB + bs * BT + row * 64;
        const int xs = x + r, b = xs % NS;
        wait(sf + b, (xs / NS) & 1);
        tc_fence_after();
        const uint32_t sb = tmem_at(tmem + COL_S + b * BN, q * 32, 0);
        float mr = m, lr = 0.f;
        const float2 sc2 = make_float2(scl, scl), l2 = make_float2(L2E, L2E);
        auto logits = [&](int c, float* s32, float2 nm0) {   // x = S * scale*log2e + bias * log2e + nm0
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint4 u = *reinterpret_cast<const uint4*>(brow + (((c * 4 + e) ^ (row & 7)) << 3));
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
          float lfall = 0.f, lf0 = 0.f;
          tmem_ld32(sb, s32[0]);
          tmem_wait_ld();
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            if (c == 0) tmem_ld32(sb + 32, s32[1]);
            float* sc = s32[c];
            const bool have = mr > -INFINITY;                     // -inf until a finite logit was seen
            const float nm = have ? -mr : 0.f;
            logits(c, sc, make_float2(nm, nm));
            float c0 = sc[0], c1 = sc[1];
#pragma unroll
            for (int i2 = 2; i2 < 32; i2 += 2) { c0 = fmaxf(c0, sc[i2]); c1 = fmaxf(c1, sc[i2 + 1]); }
            const float cm = fmaxf(c0, c1);                        // the chunk's maximum, relative to the offset (absolute if !have)
            const bool up = have ? cm > TA_TAU : cm > -1e30f;
            if (up) {                                              // raise the offset: this chunk and the running sum here; the
              if (have) {                                          // P chunk already in TMEM and O_r after the loop (lfall)
                const float f = ex2f(-cm);
                acc.x *= f; acc.y *= f;
                lfall -= cm;
              }
              mr = have ? mr + cm : cm;
#pragma unroll
              for (int i2 = 0; i2 < 32; ++i2) sc[i2] -= cm;
            }
            if (c == 0) lf0 = lfall;
            uint32_t pk[16];
#pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) {
              const float2 pv = make_float2(ex2f(sc[k2 * 2]), ex2f(sc[k2 * 2 + 1]));
              acc = add2(acc, pv);
              pk[k2] = pack2(pv.x, pv.y);
            }
            tmem_wait_ld();                    // chunk 1 landed (and every S column chunk c's P covers was read)
            tmem_st8(sb + c * 16, pk);         // P over S: chunk c's 16 columns cover S columns already read
            tmem_st8(sb + c * 16 + 8, pk + 8);
          }
          lr = acc.x + acc.y;
          if (__any_sync(0xffffffffu, lfall != 0.f)) {           // rare, warp-uniform (tcgen05.ld / st are warp-collective)
            // the sub-tile's P chunk written before a raise, and O_r (previous sub-tiles) and l_r, to the final offset. PV(previous
            // sub-tile, row r) = sub-step xs - R: waited on through its K/V stage barrier (R <= NKV)
            tmem_wait_st();
            {
              const float f = ex2f(lfall - lf0);
              float pw[16];
              tmem_ld16(sb, pw);
              tmem_wait_ld();
              uint32_t pq[16];
#pragma unroll
              for (int k2 = 0; k2 < 16; ++k2) {
                const float2 pp = bf2f(__float_as_uint(pw[k2]));
                pq[k2] = pack2(pp.x * f, pp.y * f);
              }
              tmem_st8(sb, pq);
              tmem_st8(sb + 8, pq + 8);
            }
            const float fo = ex2f(lfall);
            if (j > 0) {
              const int yp = xs - R;
              wait(kve + yp % NKV, (yp / NKV) & 1);
              tc_fence_after();
#pragma unroll 1
              for (int hh = 0; hh < D / 32; ++hh) {
                const uint32_t oa = tmem_at(tmem + COL_O + r * D + hh * 32, q * 32, 0);
                float ov[32];
                tmem_ld32(oa, ov);
                tmem_wait_ld();
                uint32_t oq[32];
#pragma unroll
                for (int k2 = 0; k2 < 32; ++k2) oq[k2] = __float_as_uint(ov[k2] * fo);
#pragma unroll
                for (int k2 = 0; k2 < 32; k2 += 8) tmem_st8(oa + k2, oq + k2);
              }
            }
            l *= fo;
          }
        }
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(pf + b);
        m = mr; l += lr;
        x += R;
        __syncwarp();
        if (lane == 0) arrive(be + bs);
      }
      sL[((lt & 1) * R + r) * BM + row] = l;      // hand the row sum to the epilogue warps
      if (LSE != nullptr) {                       // base 2: lse = m + log2(l) (offset 0 for a row with no finite logit), [B, N, H, S]
        int bh, qt, g; decode(t, bh, qt, g);
        if (g * R + r < N)
          LSE[((size_t)((bh / H) * N + g * R + r) * H + bh % H) * S + qt * BM + row] = (m > -INFINITY ? m : 0.f) + __log2f(l);
      }
      __syncwarp();
      if (lane == 0) arrive(lf + (lt & 1));
    }
  } else if (warp < 14) {
    // ---- epilogue warps: O_r / l_r -> bf16 -> out, overlapping the next task's softmax ----
    const int q = warp & 3, row = q * 32 + lane;
    int lt = 0, bad = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      wait(of, lt & 1);
      wait(lf + (lt & 1), (lt >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float lr = sL[((lt & 1) * R + r) * BM + row];
        const float inv = 1.f / lr;
        bad |= !(lr > 0.f) || !(lr < 3.0e38f);
        const bool store = g * R + r < N;          // else a padding row
        uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (H * D) + hcol(bh));
#pragma unroll
        for (int hh = 0; hh < D / 32; ++hh) {
          float o[32];
          tmem_ld32(tmem_at(tmem + COL_O + r * D + hh * 32, q * 32, 0), o);
          tmem_wait_ld();
          if (r == R - 1 && hh == D / 32 - 1) {    // O is free for the next task's PV GEMMs
            tc_fence_before();
            __syncwarp();
            if (lane == 0) arrive(oe);
          }
          if (store) {
#pragma unroll
            for (int c8 = 0; c8 < 4; ++c8) {
              uint4 w;
              w.x = pack2(o[c8 * 8 + 0] * inv, o[c8 * 8 + 1] * inv); w.y = pack2(o[c8 * 8 + 2] * inv, o[c8 * 8 + 3] * inv);
              w.z = pack2(o[c8 * 8 + 4] * inv, o[c8 * 8 + 5] * inv); w.w = pack2(o[c8 * 8 + 6] * inv, o[c8 * 8 + 7] * inv);
              dst[hh * 4 + c8] = w;
            }
          }
        }
      }
    }
    if (bad) atomicAdd(FLAGS, 1);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ================================================================================================================================
// Backward.  With P = 2^(x - lse) (x the forward's base-2 logit), dP = dO . V^T, Delta = rowsum(dO o O), dS = P o (dP - Delta):
//   dV = P^T dO,  dK = scale * dS^T Q,  dQ = scale * dS K,  dbias[b,h] = sum_n dS[b,n,h].
// The head-dim-64 core's two kernels (a key-owned dK / dV kernel and a query-owned dQ / dbias kernel, both recomputing P from the LSE)
// on the operands of a 128-channel head.  The per-score math does not depend on the head dim; the tensor-memory plan does: the key
// side's dV | dK take 256 columns, so its two gradient groups alternate 32-query stages (S^T | dP^T 2 x 64, P^T | dS^T 2 x 32 bf16);
// the query side owns one pair row a task (S | dP 2 x 128, dQ 128, Q | dO 2 x 64 bf16 columns).
// Delta arrives as four quarters per head [B, N, 4H, S] (the gate backward's per-32-column sums), added where it is read.
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void reduce_add_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
// 32 fp32 values (one row's 32 channels) -> bf16 x scale -> 64 bytes of global memory
__device__ __forceinline__ void store_row32(__nv_bfloat16* dst, const float* v, float s) {
  uint4* d = reinterpret_cast<uint4*>(dst);
#pragma unroll
  for (int c = 0; c < 4; ++c)
    d[c] = make_uint4(pack2(v[c * 8] * s, v[c * 8 + 1] * s), pack2(v[c * 8 + 2] * s, v[c * 8 + 3] * s),
                      pack2(v[c * 8 + 4] * s, v[c * 8 + 5] * s), pack2(v[c * 8 + 6] * s, v[c * 8 + 7] * s));
}

// ---- bias^T * log2e (fp32 [b, h, k, q]) for the key-side kernel, whose threads own a key: one 32 x 32 tile per block
__global__ void __launch_bounds__(256) triattn_biasT128(const __nv_bfloat16* __restrict__ bias, float* __restrict__ out, int S) {
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

// ---- kernel 1: dK, dV.  Task = (b, h, 128-key tile, pair row n), all queries in 32-wide stages, the two gradient groups taking
// alternate stages (stage w belongs to group w & 1):
//   S^T MMA warp:  S^T = K . Q^T, dP^T = V . dO^T    (ss: K / V of the task resident in shared; M = 128 k, N = 32 q, K = 128)
//   grad group g:  P^T, dS^T (bf16) -> its P^T | dS^T TMEM buffer
//   grad MMA warp: dV += P^T . dO, dK += dS^T . Q     (ts: A from TMEM, B MN-major [32 q][128]; N = 128, K = 32)
// TMEM: S^T | dP^T of group g at 64 g (32 + 32 fp32), P^T | dS^T of group g at 128 + 32 g (16 + 16 bf16 pairs), dV | dK at 256 | 384.
// Each stage has its own bias^T tile (one pair row a task: nothing to share it with).  Shared memory: K | V of two tasks (2 x 64 KiB),
// three Q | dO stages (3 x 16 KiB), three bias^T stages (3 x 16 KiB) and the stages' lse | delta: 226.1 KiB, no alignment slack.
// The drain: group 0 stores dV, group 1 dK (a thread a key row, 256 bytes of each, from registers in 32-channel pieces).
constexpr int KB_QW = 32, KB_QH = KB_QW * DH, KB_QS = 4 * KB_QH;   // a Q / dO half [32][64] (4 KiB); a stage Q h0 | Q h1 | dO h0 | dO h1
constexpr int KB_KH = 128 * DH, KB_KV = 4 * KB_KH;                 // a K / V half [128 k][64] (16 KiB); a task's K h0 | K h1 | V h0 | V h1
constexpr int KB_BT = 128 * KB_QW;                                 // bias^T [128 k][32 q] fp32, 128B swizzle (16 KiB)
constexpr int KB_NQ = 3, KB_NB = 3, KB_THREADS = 352;              // 0 TMA, 1 S^T MMA, 2-5 / 6-9 grad groups, 10 grad MMA
constexpr int KB_COL_P = 128, KB_COL_G = 256;
constexpr int KB_LD = 5 * KB_QW;                                   // a stage's lse | delta (four quarters), fp32
constexpr int KB_SMEM = (2 * KB_KV + KB_NQ * KB_QS) * 2 + KB_NB * KB_BT * 4 + KB_NQ * KB_LD * 4 + 256;
constexpr uint32_t ID_ST = idesc_bf16(128, KB_QW, 0, 0);
constexpr uint32_t ID_G = idesc_bf16(128, D, 0, 1);
static_assert(KB_SMEM <= 232448, "smem");
static_assert(KB_COL_G + 2 * D <= 512, "TMEM");

__global__ void __launch_bounds__(KB_THREADS, 1) triattn_bwd128_kv(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][.. H*128], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias^T * log2e fp32 [B*H*S (k)][S (q)], box (32, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,   // [B, N, H, S], [B, N, 4H, S]
    __nv_bfloat16* __restrict__ DK, __nv_bfloat16* __restrict__ DV, long ldo) {   // [B*N*S rows at stride ldo][H*128]
  const int NK = S / 128, NQS = S / KB_QW, SPT = NQS;     // NQS even: stage w's group is w & 1 in every task
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();                         // no alignment slack (see the header)
  unsigned char* smb = raw;
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 task buffers][K h0 | K h1 | V h0 | V h1]
  __nv_bfloat16* sQ = sKV + 2 * KB_KV;                           // [NQ][Q h0 | Q h1 | dO h0 | dO h1]
  float* sB = reinterpret_cast<float*>(sQ + KB_NQ * KB_QS);      // [NB][128 k][32 q]
  float* sLD = sB + KB_NB * KB_BT;                               // [NQ][lse 32 | delta 4 x 32]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLD + KB_NQ * KB_LD);
  uint64_t* kvf = bars;              // [2]  a task buffer's K / V landed
  uint64_t* kve = kvf + 2;           // [2]  its last S^T GEMMs retired
  uint64_t* dfr = kve + 2;           // [1]  dK / dV final
  uint64_t* de = dfr + 1;            // [1]  count 8: drained
  uint64_t* qf = de + 1;             // [NQ]
  uint64_t* qe = qf + KB_NQ;         // [NQ]
  uint64_t* bf = qe + KB_NQ;         // [NB]
  uint64_t* be = bf + KB_NB;         // [NB] count 4 (the stage's group)
  uint64_t* sf = be + KB_NB;         // [2]  group g's S^T | dP^T ready
  uint64_t* se = sf + 2;             // [2]  count 4: group g loaded it
  uint64_t* pf = se + 2;             // [2]  count 4: group g's P^T | dS^T written
  uint64_t* ge = pf + 2;             // [2]  group g's gradient GEMMs retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ge + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) {
      bar_init(kvf + i, 1); bar_init(kve + i, 1);
      bar_init(sf + i, 1); bar_init(se + i, 4); bar_init(pf + i, 4); bar_init(ge + i, 1);
    }
    bar_init(dfr, 1); bar_init(de, 8);
    for (int i = 0; i < KB_NQ; ++i) { bar_init(qf + i, 1); bar_init(qe + i, 1); }
    for (int i = 0; i < KB_NB; ++i) { bar_init(bf + i, 1); bar_init(be + i, 4); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const uint32_t sbase = sa(smb);
  auto A = [&](const void* p) -> uint32_t { return sbase + (uint32_t)(reinterpret_cast<const unsigned char*>(p) - smb); };
  auto decode = [&](int t, int& kt, int& n, int& bh) { kt = t % NK; n = (t / NK) % N; bh = t / (NK * N); };
  const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  const int total = mytasks * SPT;

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int kt, n, bh; decode(t, kt, n, bh);
        const int b_ = bh / H, h_ = bh % H, kb = lt & 1;
        if (lt >= 2) wait(A(kve + kb), ((lt >> 1) - 1) & 1);
        expect_tx(A(kvf + kb), KB_KV * 2);
        const int rw = (b_ * N + n) * S + kt * 128;
        for (int hh = 0; hh < 2; ++hh) {
          load_2d(&kmap, A(sKV + kb * KB_KV + hh * KB_KH), A(kvf + kb), h_ * D + hh * DH, rw);
          load_2d(&vmap, A(sKV + kb * KB_KV + (2 + hh) * KB_KH), A(kvf + kb), h_ * D + hh * DH, rw);
        }
        for (int w = 0; w < SPT; ++w, ++y) {
          const int bs = y % KB_NB, sl = y % KB_NQ;
          if (y >= KB_NB) wait(A(be + bs), ((y / KB_NB) - 1) & 1);
          expect_tx(A(bf + bs), KB_BT * 4);
          load_2d(&bmap, A(sB + bs * KB_BT), A(bf + bs), w * KB_QW, bh * S + kt * 128);
          if (y >= KB_NQ) wait(A(qe + sl), ((y / KB_NQ) - 1) & 1);
          expect_tx(A(qf + sl), KB_QS * 2 + KB_LD * 4);
          const int rq = (b_ * N + n) * S + w * KB_QW;
          for (int hh = 0; hh < 2; ++hh) {
            load_2d(&qmap, A(sQ + sl * KB_QS + hh * KB_QH), A(qf + sl), h_ * D + hh * DH, rq);
            load_2d(&dmap, A(sQ + sl * KB_QS + (2 + hh) * KB_QH), A(qf + sl), h_ * D + hh * DH, rq);
          }
          const size_t st = ((size_t)(b_ * N + n) * H + h_) * S + w * KB_QW;
          bulk_load(A(sLD + sl * KB_LD), LSE + st, KB_QW * 4, A(qf + sl));
          const size_t sd = ((size_t)(b_ * N + n) * 4 * H + 4 * h_) * S + w * KB_QW;
          for (int qd = 0; qd < 4; ++qd) bulk_load(A(sLD + sl * KB_LD + (1 + qd) * KB_QW), DELTA + sd + (size_t)qd * S, KB_QW * 4, A(qf + sl));
        }
      }
    }
  } else if (warp == 1 || warp == 10) {
    // converged issue loops, one elected lane's tcgen05 ops predicated; ring slots and phases as incremental counters
    const uint32_t L = elect_one() ? 1u : 0u;
    constexpr uint64_t SLOT = KB_QS * 2 / 16, QHD = KB_QH * 2 / 16, KHD = KB_KH * 2 / 16, KVBUF = KB_KV * 2 / 16;
    const uint32_t qfa = A(qf), qea = A(qe), sfa = A(sf), sea = A(se), pfa = A(pf), gea = A(ge), dea = A(de), dfa = A(dfr);
    int sl = 0, slp = 0, w = 0, lt = 0;
    if (warp == 1) {
      // S^T of a stage goes out once its Q | dO | lse | delta slot and its bias^T tile landed: the group's S^T barrier implies both
      const uint64_t dQ0 = desc_k128(sQ), dKV0 = desc_k128(sKV);
      const uint32_t bfa = A(bf), kvfa = A(kvf), kvea = A(kve);
      int bs = 0, bsp = 0;
      for (int y = 0; y < total; ++y) {
        const int r = y & 1, kb = lt & 1;
        const uint32_t okq = probe(qfa + sl * 8, slp);
        const uint32_t oke = y >= 2 ? probe(sea + r * 8, ((y >> 1) - 1) & 1) : 1u;   // group r loaded its previous S^T
        const uint32_t okb = probe(bfa + bs * 8, bsp);
        if (w == 0) wait(kvfa + kb * 8, (lt >> 1) & 1);
        if (!okq) wait(qfa + sl * 8, slp);
        if (!oke) wait(sea + r * 8, ((y >> 1) - 1) & 1);
        if (!okb) wait(bfa + bs * 8, bsp);
        tc_fence_after();
        const uint64_t bq = dQ0 + sl * SLOT, ak = dKV0 + kb * KVBUF;
        const uint32_t d = tmem + r * 64;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss_if(L, d, ak + (ks >> 2) * KHD + 2 * (ks & 3), bq + (ks >> 2) * QHD + 2 * (ks & 3), ID_ST, ks ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss_if(L, d + 32, ak + (2 + (ks >> 2)) * KHD + 2 * (ks & 3), bq + (2 + (ks >> 2)) * QHD + 2 * (ks & 3), ID_ST, ks ? 1u : 0u);
        mma_commit_if(L, sfa + r * 8);
        if (w == SPT - 1) mma_commit_if(L, kvea + kb * 8);  // the task's K / V buffer is free once these retire
        if (++bs == KB_NB) { bs = 0; bsp ^= 1; }
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    } else {
      const uint64_t dM0 = desc_mn128(sQ, KB_QH * 2);       // [32 q][128] MN-major: the two 64-wide halves KB_QH apart
      const uint32_t tG = tmem + KB_COL_G;
      for (int p = 0; p < total; ++p) {
        const int r = p & 1;                               // = the group
        const bool first = w == 0, last = w == SPT - 1;
        const uint32_t gpar = (p >> 1) & 1;
        const uint32_t okp = probe(pfa + r * 8, gpar);
        if (first && lt >= 1) wait(dea, (lt - 1) & 1);   // the previous task's accumulators were drained
        if (!okp) wait(pfa + r * 8, gpar);
        tc_fence_after();
        const uint64_t bm = dM0 + sl * SLOT;
        const uint32_t a = tmem + KB_COL_P + r * 32;
#pragma unroll
        for (int kk = 0; kk < KB_QW / 16; ++kk) {
          const uint32_t acc = (!first || kk) ? 1u : 0u;
          mma_ts_if(L, tG, a + kk * 8, bm + 2 * QHD + kk * 128, ID_G, acc);       // dV += P^T dO
          mma_ts_if(L, tG + D, a + 16 + kk * 8, bm + kk * 128, ID_G, acc);       // dK += dS^T Q
        }
        mma_commit_if(L, qea + sl * 8);
        mma_commit_if(L, gea + r * 8);
        if (last) mma_commit_if(L, dfa);
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    }
    __syncwarp();
  } else if (warp < 10) {
    const int qq = warp & 3, k = qq * 32 + lane;   // TMEM lane = key row of the tile
    const int gi = (warp - 2) >> 2;                // this group: stages gi, gi + 2, ...
    const float2 sc2 = make_float2(scl, scl), neg1 = make_float2(-1.f, -1.f);
    const uint32_t tS = tmem_at(tmem + gi * 64, qq * 32, 0), tP = tmem_at(tmem + KB_COL_P + gi * 32, qq * 32, 0);
    int js = 0, lt = 0;                            // this group's stage count (phase of sf / se / pf / ge), task
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int kt, n, bh; decode(t, kt, n, bh);
      const int b_ = bh / H, h_ = bh % H;
      for (int qs = gi; qs < NQS; qs += 2, ++js) {
        const int y = lt * SPT + qs, bs = y % KB_NB, sl = y % KB_NQ;
        wait(A(sf + gi), js & 1);                   // implies the bias^T tile and the lse / delta slot (see the S^T issuer)
        tc_fence_after();
        const float* ld = sLD + sl * KB_LD;
        float sv[32], dp[32], bl[32];
        tmem_ld32(tS, sv);
        tmem_ld32(tS + 32, dp);
        {
          const float* bp = sB + bs * KB_BT + k * 32;
#pragma unroll
          for (int q4 = 0; q4 < 8; ++q4) {
            const float4 u = *reinterpret_cast<const float4*>(bp + ((q4 ^ (k & 7)) << 2));
            bl[q4 * 4] = u.x; bl[q4 * 4 + 1] = u.y; bl[q4 * 4 + 2] = u.z; bl[q4 * 4 + 3] = u.w;
          }
        }
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(se + gi));          // the group's S^T buffer takes its next stage
        uint32_t pk[16], dk[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float4 l4 = *reinterpret_cast<const float4*>(ld + 4 * j);
          float4 dsum = *reinterpret_cast<const float4*>(ld + KB_QW + 4 * j);
#pragma unroll
          for (int qd = 1; qd < 4; ++qd) {
            const float4 dq = *reinterpret_cast<const float4*>(ld + (1 + qd) * KB_QW + 4 * j);
            dsum.x += dq.x; dsum.y += dq.y; dsum.z += dq.z; dsum.w += dq.w;
          }
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int i = 4 * j + 2 * h2;
            const float2 lm = h2 ? make_float2(l4.z, l4.w) : make_float2(l4.x, l4.y);
            const float2 dm = h2 ? make_float2(dsum.z, dsum.w) : make_float2(dsum.x, dsum.y);
            const float2 x = fma2(lm, neg1, fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])));
            const float2 pp = make_float2(ex2f(x.x), ex2f(x.y));
            const float2 ds = mul2(pp, fma2(dm, neg1, make_float2(dp[i], dp[i + 1])));
            pk[i >> 1] = pack2(pp.x, pp.y);
            dk[i >> 1] = pack2(ds.x, ds.y);
          }
        }
        if (js >= 1) {                              // this group's previous gradient GEMMs read the P^T | dS^T buffer
          wait(A(ge + gi), (js - 1) & 1);
          tc_fence_after();
        }
        tmem_st8(tP, pk); tmem_st8(tP + 8, pk + 8);
        tmem_st8(tP + 16, dk); tmem_st8(tP + 24, dk + 8);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        // the bias^T tile is released only now: an mbarrier arrive does not wait for this thread's outstanding shared loads (ptxas
        // issued it ahead of the bias LDS when it followed them directly, and the next TMA load overwrote rows the slowest warp had
        // not read yet); here every bias value has been consumed by the P^T | dS^T this thread stored
        if (lane == 0) { arrive(A(pf + gi)); arrive(A(be + bs)); }
      }
      // ---- the drain: group 0 dV, group 1 dK (x scale), a thread a key row, 32 channels at a time ----
      wait(A(dfr), lt & 1);
      tc_fence_after();
      __nv_bfloat16* dst = (gi ? DK : DV) + ((size_t)(b_ * N + n) * S + kt * 128 + k) * ldo + h_ * D;
      const float s = gi ? scale : 1.f;
      const uint32_t src = tmem_at(tmem + KB_COL_G + gi * D, qq * 32, 0);
#pragma unroll
      for (int c = 0; c < D / 32; ++c) {
        float a0[32];
        tmem_ld32(src + c * 32, a0);
        tmem_wait_ld();
        if (c == D / 32 - 1) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(A(de));            // the accumulators are free for the next task
        }
        store_row32(dst + c * 32, a0, s);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- kernel 2: dQ and dbias.  Task = (b, h, 128-query tile, pair row n), all keys in 64-wide sub-tiles.
//   MMA:   S = Q . K^T, dP = dO . V^T        (ts: A = Q / dO from TMEM, B = K / V K-major [64 k][128]; M = 128 q, N = 64 k, K = 128)
//   grad:  dS (bf16) over dP; dbias straight from the fp32 dS   (group gi: key columns 32 gi .. of the sub-tile)
//   MMA:   dQ += dS . K                      (ts: A = dS from TMEM, B = K MN-major [64 k][128]; N = 128)
// TMEM: S | dP buffers at 0 / 128, dQ at 256, Q at 384, dO at 448.  Group 0 copies Q, group 1 dO into TMEM; group gi drains dQ's
// columns 64 gi .. (straight from registers, a thread a query row).  Shared memory: the Q | dO staging (64 KiB, refilled for the next
// task once copied), three K | V stages (3 x 32 KiB), two bias tiles (2 x 16 KiB), lse | delta and one dbias buffer a group
// (2 x 16 KiB): 226.8 KiB, no alignment slack.  One pair row a task: dbias leaves per (row, sub-tile) by TMA reduce-add.
constexpr int QB_KW = 64, QB_KH = QB_KW * DH, QB_KS = 4 * QB_KH;   // a K / V half [64][64] (8 KiB); a stage K h0 | K h1 | V h0 | V h1
constexpr int QB_BT = BM * QB_KW;                                   // bias [128 q][64 k] 128B swizzle
constexpr int QB_NKV = 3, QB_NB = 2, QB_NS = 2, QB_THREADS = 320;
constexpr int QB_COL_G = QB_NS * 128;                               // dQ at 256
constexpr int QB_COL_A = QB_COL_G + D;                              // Q at 384, dO at 448
constexpr int QB_DBT = BM * 32;                                     // a group's dbias piece [128 q][32 k] fp32, 128B swizzle (16 KiB)
constexpr int QB_LD = 5 * BM;                                       // the row's lse | delta (four quarters)
constexpr int QB_SMEM = (2 * QT + QB_NKV * QB_KS + QB_NB * QB_BT) * 2 + QB_LD * 4 + 2 * QB_DBT * 4 + 256;
constexpr uint32_t ID_SQ = idesc_bf16(128, QB_KW, 0, 0);
static_assert(QB_COL_A + 2 * 64 <= 512, "TMEM");
static_assert(QB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(QB_THREADS, 1) triattn_bwd128_q(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][.. H*128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (64, 64), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,
    __nv_bfloat16* __restrict__ DQ, long ldo,            // [B*N*S rows at stride ldo][H*128]
    const __grid_constant__ CUtensorMap dbmap) {         // dbias fp32 [B*H*S][S], zeroed; box (32, 128), 128B swizzle (TMA reduce-add)
  const int QTL = S / BM, NKS = S / QB_KW, SPT = NKS;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();                         // no alignment slack (see the header)
  unsigned char* smb = raw;
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smb);    // [Q h0 | Q h1 | dO h0 | dO h1] staging for TMEM
  __nv_bfloat16* sKV = sA + 2 * QT;                              // [NKV][K h0 | K h1 | V h0 | V h1]
  __nv_bfloat16* sB = sKV + QB_NKV * QB_KS;                      // [NB][128 q][64 k]
  float* sDB = reinterpret_cast<float*>(sB + QB_NB * QB_BT);     // [2 groups][QB_DBT] dbias pieces (1024-aligned: swizzled TMA source)
  float* sLD = sDB + 2 * QB_DBT;                                 // [lse 128 | delta 4 x 128]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLD + QB_LD);
  uint64_t* af = bars;
  uint64_t* ae = af + 1;             // count 8: the staging was copied (the producer may refill it)
  uint64_t* at = ae + 1;             // count 8: Q / dO in TMEM
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
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const uint32_t sbase = sa(smb);
  auto A = [&](const void* p) -> uint32_t { return sbase + (uint32_t)(reinterpret_cast<const unsigned char*>(p) - smb); };
  auto decode = [&](int t, int& qt, int& n, int& bh) { qt = t % QTL; n = (t / QTL) % N; bh = t / (QTL * N); };

  if (warp == 0) {
    if (lane == 0) {
      auto staging = [&](int t) {                // Q / dO rows, lse, delta of task t -> staging
        int qt, n, bh; decode(t, qt, n, bh);
        const int b_ = bh / H, h_ = bh % H;
        expect_tx(A(af), 2 * QT * 2 + QB_LD * 4);
        const int rw = (b_ * N + n) * S + qt * BM;
        for (int hh = 0; hh < 2; ++hh) {
          load_2d(&qmap, A(sA + hh * QH), A(af), h_ * D + hh * DH, rw);
          load_2d(&dmap, A(sA + (2 + hh) * QH), A(af), h_ * D + hh * DH, rw);
        }
        const size_t st = ((size_t)(b_ * N + n) * H + h_) * S + qt * BM;
        bulk_load(A(sLD), LSE + st, BM * 4, A(af));
        const size_t sd = ((size_t)(b_ * N + n) * 4 * H + 4 * h_) * S + qt * BM;
        for (int qd = 0; qd < 4; ++qd) bulk_load(A(sLD + (1 + qd) * BM), DELTA + sd + (size_t)qd * S, BM * 4, A(af));
      };
      if ((int)blockIdx.x < ntasks) staging(blockIdx.x);
      const int st_at = min(SPT, QB_NKV) - 1;
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int qt, n, bh; decode(t, qt, n, bh);
        const int b_ = bh / H, h_ = bh % H;
        for (int w = 0; w < SPT; ++w, ++y) {
          if (w == st_at && t + (int)gridDim.x < ntasks) { wait(A(ae), lt & 1); staging(t + gridDim.x); }
          const int sl = y % QB_NKV, bs = y % QB_NB;
          if (y >= QB_NB) wait(A(be + bs), ((y / QB_NB) - 1) & 1);
          expect_tx(A(bf + bs), QB_BT * 2);
          load_2d(&bmap, A(sB + bs * QB_BT), A(bf + bs), w * QB_KW, bh * S + qt * BM);
          if (y >= QB_NKV) wait(A(kve + sl), ((y / QB_NKV) - 1) & 1);
          expect_tx(A(kvf + sl), QB_KS * 2);
          const int rw = (b_ * N + n) * S + w * QB_KW;
          for (int hh = 0; hh < 2; ++hh) {
            load_2d(&kmap, A(sKV + sl * QB_KS + hh * QB_KH), A(kvf + sl), h_ * D + hh * DH, rw);
            load_2d(&vmap, A(sKV + sl * QB_KS + (2 + hh) * QB_KH), A(kvf + sl), h_ * D + hh * DH, rw);
          }
        }
      }
    }
  } else if (warp == 1) {
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    const uint64_t dK0 = desc_k128(sKV), dM0 = desc_mn128(sKV, QB_KH * 2);
    constexpr uint64_t SLOT = QB_KS * 2 / 16, KHD = QB_KH * 2 / 16;
    auto stt = [&](int y, int lt, int w) {
      const int sl = y % QB_NKV, b = y % QB_NS;
      if (w == 0) wait(A(at), lt & 1);
      wait(A(kvf + sl), (y / QB_NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t bk = dK0 + sl * SLOT;
        const uint32_t d = tmem + b * 128, a = tmem + QB_COL_A;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) mma_ts(d, a + ks * 8, bk + (ks >> 2) * KHD + 2 * (ks & 3), ID_SQ, ks ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ts(d + 64, a + 64 + ks * 8, bk + (2 + (ks >> 2)) * KHD + 2 * (ks & 3), ID_SQ, ks ? 1u : 0u);
        mma_commit(A(sf + b));
      }
      __syncwarp();
    };
    auto grad = [&](int y, int w) {
      const int sl = y % QB_NKV, b = y % QB_NS;
      wait(A(pf + b), (y / QB_NS) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t bm = dM0 + sl * SLOT;
        const uint32_t acc = w ? 1u : 0u, d = tmem + QB_COL_G;
#pragma unroll
        for (int ks = 0; ks < QB_KW / 16; ++ks) {
          const uint32_t a = tmem + b * 128 + 64 + (ks >> 1) * 32 + (ks & 1) * 8;   // group (ks >> 1)'s dS columns
          mma_ts(d, a, bm + ks * 128, ID_G, (acc | ks) ? 1u : 0u);
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
    const int gi = (warp - 2) >> 2, c0 = gi * 32;    // key columns of a sub-tile; Q (group 0) / dO (group 1) for the copy
    const float2 sc2 = make_float2(scl, scl);
    const bool lead = (warp & 3) == 2 && lane == 0;   // the group's TMA-issuing thread (warps 2 / 6)
    float* myDB = sDB + gi * QB_DBT;
    int lt = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int qt, n, bh; decode(t, qt, n, bh);
      const int b_ = bh / H;
      float lse, dl;
      {
        wait(A(af), lt & 1);
        const uint32_t ta = tmem_at(tmem + QB_COL_A + gi * 64, qq * 32, 0);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {                    // Q (group 0) / dO (group 1): [128][64] 128B-swizzled halves -> 64 TMEM columns
          const unsigned char* base = reinterpret_cast<const unsigned char*>(sA + (2 * gi + hh) * QH) + row * 128;
          uint32_t wv[32];
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            const uint4 u = *reinterpret_cast<const uint4*>(base + ((c ^ (row & 7)) << 4));
            wv[c * 4] = u.x; wv[c * 4 + 1] = u.y; wv[c * 4 + 2] = u.z; wv[c * 4 + 3] = u.w;
          }
#pragma unroll
          for (int c = 0; c < 4; ++c) tmem_st8(ta + hh * 32 + c * 8, wv + c * 8);
        }
        lse = sLD[row];
        dl = sLD[BM + row] + sLD[2 * BM + row] + sLD[3 * BM + row] + sLD[4 * BM + row];
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(at));
      }
      for (int kb = 0; kb < NKS; ++kb) {
        const int y = lt * SPT + kb, bs = y % QB_NB, b = y % QB_NS;
        wait(A(bf + bs), (y / QB_NB) & 1);
        float bl[32];
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
        }
        wait(A(sf + b), (y / QB_NS) & 1);
        tc_fence_after();
        const uint32_t tb = tmem_at(tmem + b * 128, qq * 32, 0);
        float sv[32], dp[32];
        tmem_ld32(tb + c0, sv);
        tmem_ld32(tb + 64 + c0, dp);
        tmem_wait_ld();
        const float2 nl = make_float2(-lse, -lse), nd = make_float2(-dl, -dl);
        uint32_t dk[16];
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
          const float2 x = add2(fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])), nl);
          const float2 pp = make_float2(ex2f(x.x), ex2f(x.y));
          const float2 ds = mul2(pp, add2(make_float2(dp[i], dp[i + 1]), nd));
          dk[i >> 1] = pack2(ds.x, ds.y);
          sv[i] = ds.x; sv[i + 1] = ds.y;             // the fp32 dS: this row's dbias piece
        }
        tmem_st8(tb + 64 + c0, dk); tmem_st8(tb + 64 + c0 + 8, dk + 8);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        // the bias tile and (after the first sub-tile) the staging with lse / delta are released only once their values were consumed
        // by the dS this thread stored: an mbarrier arrive does not wait for outstanding shared loads
        if (lane == 0) {
          arrive(A(pf + b));
          arrive(A(be + bs));
          if (kb == 0) arrive(A(ae));
        }
        // the group's [128 q][32 k] piece -> shared -> one TMA reduce-add
        {
          if (lead) bulk_wait_read<0>();              // the previous reduce has read the buffer
          named_sync(1 + gi, 128);
#pragma unroll
          for (int c = 0; c < 8; ++c)
            *reinterpret_cast<float4*>(myDB + row * 32 + ((c ^ (row & 7)) << 2)) = make_float4(sv[4 * c], sv[4 * c + 1], sv[4 * c + 2], sv[4 * c + 3]);
          fence_proxy_async();
          named_sync(1 + gi, 128);
          if (lead) { reduce_add_2d(&dbmap, A(myDB), kb * QB_KW + c0, bh * S + qt * BM); bulk_commit(); }
        }
      }
      // ---- dQ columns 64 gi .. (x scale), a thread a query row ----
      wait(A(df), lt & 1);
      tc_fence_after();
      float a0[32], a1[32];
      const uint32_t src = tmem_at(tmem + QB_COL_G + gi * 64, qq * 32, 0);
      tmem_ld32(src, a0);
      tmem_ld32(src + 32, a1);
      tmem_wait_ld();
      tc_fence_before();
      __nv_bfloat16* dst = DQ + ((size_t)(b_ * N + n) * S + qt * BM + row) * ldo + (bh % H) * D + gi * 64;
      store_row32(dst, a0, scale);
      store_row32(dst + 32, a1, scale);
    }
  }
  if (warp >= 2 && (warp & 3) == 2 && lane == 0) bulk_wait<0>();
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

// q, k, v [B, N, S, H, 128] bf16 (rows (b, n, s) of all heads at one common stride, e.g. column slices of one projection output, or
// contiguous); bias [B, H, S, S] bf16 (masked keys already filled); S % 128 == 0.
// -> (out [B, N, S, H, 128] bf16 contiguous, lse fp32 [B, N, H, S] (empty unless want_lse), flags int32 [1]: rows whose sum left
//     (0, 3e38), a per-device counter never re-zeroed)
// out_: a contiguous bf16 tensor like q to write into (a group of samples' part of a batch's out), or empty (fresh)
// want_lse: also the base-2 LSE fp32 [B, N, H, S] for the backward (into lse_ when it is given)
std::vector<torch::Tensor> triattn_fwd128(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, double scale,
                                          torch::Tensor out_, bool want_lse, torch::Tensor lse_) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 128] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld,
                "q, k, v: same shape, [.., H, 128] dense per row, rows at one common stride");
  TORCH_CHECK(ld >= H * D && ld % 8 == 0, "q / k / v row stride: >= H*128 and a multiple of 8 elements");
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous() && bias.numel() == B * H * S * S, "bias: [B, H, S, S] bf16");
  const bool og = out_.defined() && out_.numel() > 0;
  if (og) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == q.numel(), "out: contiguous bf16 like q");
  auto out = og ? out_.view(q.sizes()) : torch::empty(q.sizes(), q.options());
  const bool lg = want_lse && lse_.defined() && lse_.numel() > 0;
  if (lg) TORCH_CHECK(lse_.scalar_type() == torch::kFloat32 && lse_.is_contiguous() && lse_.numel() == B * N * H * S, "lse: contiguous fp32 [B, N, H, S]");
  auto lse = lg ? lse_.view({B, N, H, S}) : want_lse ? torch::empty({B, N, H, S}, q.options().dtype(torch::kFloat32))
                                                       : torch::empty({0}, q.options().dtype(torch::kFloat32));
  static torch::Tensor flags_dev[64];
  torch::Tensor& flags = flags_dev[q.device().index()];
  if (!flags.defined()) flags = torch::zeros({1}, q.options().dtype(torch::kInt32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  CUtensorMap qm = make_map<2>(q.data_ptr(), {cols, rows}, {(uint64_t)ld}, {DH, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "q");
  CUtensorMap km = make_map<2>(k.data_ptr(), {cols, rows}, {(uint64_t)ld}, {DH, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "k");
  CUtensorMap vm = make_map<2>(v.data_ptr(), {cols, rows}, {(uint64_t)ld}, {DH, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "v");
  CUtensorMap bm = make_map<2>(bias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {BN, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  const long nt = B * H * (S / BM) * ((N + R - 1) / R);
  TORCH_CHECK(nt < (1L << 31) && rows < (1ull << 31), "too many tasks / rows for one launch");
  const int ntasks = (int)nt;
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_fwd128_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  triattn_fwd128_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)H, (int)S, ntasks, (float)scale * L2E, qm, km, vm,
      bm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags};
}

// Backward of triattn_fwd128.  q, k, v, dout [B, N, S, H, 128] bf16 (rows (b, n, s) of all heads, each tensor at its own row stride, a
// multiple of 8 elements: column slices of one projection output, or contiguous); bias [B, H, S, S] bf16 (masked keys filled);
// lse fp32 [B, N, H, S] (triattn_fwd128's); delta fp32 [B, N, 4H, S] = rowsum(dout o out) over each head's four 32-channel quarters.
// dq_out / dk_out / dv_out: [B*N*S, H*128] views to store into (dense rows at a 16-byte-multiple stride, e.g. column slices of one
// gradient buffer), or empty (fresh [B, N, S, H, 128]).
// -> (dq, dk, dv, dbias fp32 [B, H, S, S]: summed over n by fp32 reduce-adds)
std::vector<torch::Tensor> triattn_bwd128(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor dout,
                                          torch::Tensor lse, torch::Tensor delta, double scale, torch::Tensor dq_out, torch::Tensor dk_out,
                                          torch::Tensor dv_out) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 128] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  for (const auto* t : {&q, &k, &v, &dout}) {
    const long ld = t->stride(2);
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(1) == S * ld && t->stride(0) == N * S * ld && ld >= H * D && ld % 8 == 0,
                "q, k, v, dout: [B, N, S, H, 128] bf16, [.., H, 128] dense per row, rows at a 16-byte-multiple stride");
  }
  TORCH_CHECK(k.stride(2) == q.stride(2) && v.stride(2) == q.stride(2), "q, k, v: one row stride");
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous() && bias.numel() == B * H * S * S, "bias: [B, H, S, S] bf16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.is_contiguous() && lse.numel() == B * N * H * S, "lse: fp32 [B, N, H, S]");
  TORCH_CHECK(delta.scalar_type() == torch::kFloat32 && delta.is_contiguous() && delta.numel() == B * N * 4 * H * S, "delta: fp32 [B, N, 4H, S]");
  const bool given = dq_out.defined() && dq_out.numel() > 0;
  auto dq = given ? dq_out : torch::empty(q.sizes(), q.options()), dk = given ? dk_out : torch::empty(q.sizes(), q.options()),
       dv = given ? dv_out : torch::empty(q.sizes(), q.options());
  const long ldo = given ? dq.stride(0) : H * D;
  if (given)
    for (const auto* t : {&dq, &dk, &dv})
      TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->dim() == 2 && t->size(0) == B * N * S && t->size(1) == H * D && t->stride(1) == 1 &&
                  t->stride(0) == ldo && ldo % 8 == 0, "dq / dk / dv outputs: [rows, H*128] bf16, dense rows at one 16-byte-multiple stride");
  auto db = torch::zeros({B, H, S, S}, q.options().dtype(torch::kFloat32));
  auto bT = torch::empty({B, H, S, S}, q.options().dtype(torch::kFloat32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  auto m2 = [&](const torch::Tensor& t, uint32_t box, const char* what) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {(uint64_t)t.stride(2)}, {DH, box}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap q32 = m2(q, KB_QW, "q"), d32 = m2(dout, KB_QW, "dout"), k128 = m2(k, 128, "k"), v128 = m2(v, 128, "v");
  CUtensorMap q128 = m2(q, BM, "q"), d128 = m2(dout, BM, "dout"), k64 = m2(k, QB_KW, "k"), v64 = m2(v, QB_KW, "v");
  CUtensorMap bkv = make_map<2>(bT.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_128B,
                                "bias^T", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap bq = make_map<2>(bias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {QB_KW, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  CUtensorMap dbm = make_map<2>(db.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "dbias",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd128_kv, cudaFuncAttributeMaxDynamicSharedMemorySize, KB_SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd128_q, cudaFuncAttributeMaxDynamicSharedMemorySize, QB_SMEM));
    attr = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sms = num_sms(q.device().index());
  const float scl = (float)scale * L2E;
  auto bf = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  triattn_biasT128<<<dim3(S / 32, S / 32, B * H), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
                                                                   bT.data_ptr<float>(), (int)S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long n1 = B * H * (S / 128) * N, n2 = B * H * (S / BM) * N;
  TORCH_CHECK(n1 < (1L << 31) && n2 < (1L << 31) && rows < (1ull << 31), "too many tasks / rows for one launch");
  triattn_bwd128_kv<<<(int)std::min<long>(n1, sms), KB_THREADS, KB_SMEM, stream>>>((int)N, (int)H, (int)S, (int)n1, scl, (float)scale, q32, d32,
      k128, v128, bkv, lse.data_ptr<float>(), delta.data_ptr<float>(), bf(dk), bf(dv), ldo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  triattn_bwd128_q<<<(int)std::min<long>(n2, sms), QB_THREADS, QB_SMEM, stream>>>((int)N, (int)H, (int)S, (int)n2, scl, (float)scale, q128, d128,
      k64, v64, bq, lse.data_ptr<float>(), delta.data_ptr<float>(), bf(dq), ldo, dbm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd128", &triattn_fwd128, "sm100 triangle-attention forward, 128-channel heads (bf16): q/k/v/out [B, N, S, H, 128], optional base-2 LSE",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"), py::arg("out"), py::arg("want_lse") = false,
        py::arg("lse") = torch::Tensor());
  m.def("triattn_bwd128", &triattn_bwd128, "sm100 triangle-attention backward, 128-channel heads: (dq, dk, dv [.., H, 128] bf16, dbias fp32 [B, H, S, S])",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("dout"), py::arg("lse"), py::arg("delta"), py::arg("scale"),
        py::arg("dq_out"), py::arg("dk_out"), py::arg("dv_out"));
}
