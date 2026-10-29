// Triangle attention for B200 (sm_100a) with 64-channel heads: Chai-1's trunk and confidence pair attention (4 x 64 at d_pair 256),
// RoseTTAFold3's template stack (4 x 64 at d_pair 64), IntelliFold-2's trunk (8 x 64 at d_pair 512).  bf16 operands, fp32 softmax and
// accumulation, bf16 out; the forward (with the base-2 LSE for training) and its backward (below the forward):
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]        (masked keys: bias = bf16 min)
//
// The head-dim-32 core's schedule (triattn_sm100.cu) on the operands of a 64-channel head, as its own kernel (the d_pair-128 core stays
// as it is): a CTA task = (b, h, 128-query tile) x R = 2 pair rows; per 128-key tile j and row r:
//   MMA warp:      S = Q_r . K_{r,j}^T     (M = 128, N = 128, K = 64: four k-steps on 128-byte-swizzled [128][64] tiles)
//   softmax warps: x = S * scale*log2e + bias * log2e; p = 2^(x - m_r) (m_r the running maximum, raised lazily past TA_TAU); P over S
//   MMA warp:      O_r += P . V_{r,j}       (A = P from TMEM, B = V MN-major [128 k][64], N = 64)
// TMEM: three 128-column S / P buffers + O of the two rows (2 x 64) = 512 columns.  Shared memory: Q of two tasks (2 x 2 x 16 KiB), three
// K | V stages (3 x 32 KiB), two bias tiles (2 x 32 KiB) and the row sums: 226.5 KiB, so the dynamic base carries no alignment slack (the
// kernel traps if it is not 1024-aligned, as tri_wfront does).
// Warps: 0 = TMA, 1 = MMA, 2-5 / 6-9 = the two softmax groups (row 0 / row 1 of the task), 10-13 = epilogue (O / l -> bf16 out).
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

#ifndef TA_TAU
#define TA_TAU 8.f       // a chunk raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
namespace {
constexpr int D = 64, BM = 128, BN = 128, R = 2;
constexpr int QT = BM * D;                                 // Q tile [128 q][64], 128B swizzle (16 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][64], 128B swizzle (16 KiB)
constexpr int BT = BM * BN;                                // bias tile [2 key halves][128 q][64], 128B swizzle, bf16 (32 KiB)
constexpr int NKV = 3, NB = 2, NS = 3;                     // K | V stages, bias stages, S / P TMEM buffers (NKV >= NS: QK^T runs NS - 1 ahead)
constexpr int THREADS = 448;
constexpr int SMEM = (2 * R * QT + NKV * 2 * KVT + NB * BT) * 2 + 2 * R * BM * 4 + 512;
constexpr int COL_S = 0, COL_O = NS * BN;                  // S/P [NS][128] | O [R rows][64]
static_assert(COL_O + R * D <= 512, "TMEM");
static_assert(NKV >= NS, "the MMA warp's QK^T look-ahead needs a K | V stage each");
constexpr uint32_t ID_S = idesc_bf16(128, BN, 0, 0);       // A = Q K-major, B = K K-major
constexpr uint32_t ID_O = idesc_bf16(128, D, 0, 1);        // A = P from TMEM, B = V MN-major
constexpr float L2E = 1.4426950408889634f;
static_assert(SMEM <= 232448, "one CTA per SM");

__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__global__ void __launch_bounds__(THREADS, 1) triattn_fwd64_sm100(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][H*64] bf16, box (64, 128) at column h*64, 128B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*64]
    float* __restrict__ LSE,                             // [B, N, H, S] base-2 log-sum-exp (training), or nullptr
    int* __restrict__ FLAGS) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();                         // no alignment slack (see the header)
  unsigned char* smb = raw;
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 tasks][R][QT]
  __nv_bfloat16* sKV = sQ + 2 * R * QT;                          // [NKV][K | V]
  __nv_bfloat16* sB = sKV + NKV * 2 * KVT;                       // [NB][BT]
  float* sL = reinterpret_cast<float*>(sB + NB * BT);            // [2 tasks][R][128] row sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sL + 2 * R * BM);
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
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H; n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row
  auto hcol = [&](int bh) -> int { return (bh % H) * D; };

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        const int qb = lt & 1;
        if (lt >= 2) wait(qe + qb, ((lt >> 1) - 1) & 1);
        expect_tx(qf + qb, R * QT * 2);
        for (int r = 0; r < R; ++r) load_2d(&qmap, sQ + (qb * R + r) * QT, qf + qb, hcol(bh), row0(bh, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1);
          expect_tx(bf + bs, BT * 2);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sB + bs * BT + hh * BM * 64, bf + bs, j * BN + hh * 64, bh * S + qt * BM);
          for (int r = 0; r < R; ++r, ++x) {
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
    // flattened sub-steps y (task k = y / SPT, key tile j, row r): QK^T runs up to NS - 1 sub-steps ahead of the PV GEMMs
    const int SPT = NK * R;
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    auto qk = [&](int y) {
      const int k = y / SPT, w = y % SPT, r = w % R, st = y % NKV, b = y % NS, qb = k & 1;
      if (w == 0) wait(qf + qb, (k >> 1) & 1);
      wait(kvf + st, (y / NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* q = sQ + (qb * R + r) * QT;
        const __nv_bfloat16* kk = sKV + st * 2 * KVT;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + COL_S + b * BN, desc_k128(q + ks * 16), desc_k128(kk + ks * 16), ID_S, ks ? 1u : 0u);
        mma_commit(sf + b);
        if (w == SPT - 1) mma_commit(qe + qb);
      }
      __syncwarp();
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
          mma_ts(o, tmem + COL_S + b * BN + ks * 8, desc_mn128(v + ks * 16 * D, KVT * 2), ID_O, (j | ks) ? 1u : 0u);
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
          float lfall = 0.f, lfc[4];
          tmem_ld32(sb, s32[0]);
          tmem_wait_ld();
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if (c + 1 < 4) tmem_ld32(sb + (c + 1) * 32, s32[(c + 1) & 1]);
            float* sc = s32[c & 1];
            const bool have = mr > -INFINITY;                     // -inf until a finite logit was seen
            const float nm = have ? -mr : 0.f;
            logits(c, sc, make_float2(nm, nm));
            float c0 = sc[0], c1 = sc[1];
#pragma unroll
            for (int i2 = 2; i2 < 32; i2 += 2) { c0 = fmaxf(c0, sc[i2]); c1 = fmaxf(c1, sc[i2 + 1]); }
            const float cm = fmaxf(c0, c1);                        // the chunk's maximum, relative to the offset (absolute if !have)
            const bool up = have ? cm > TA_TAU : cm > -1e30f;
            if (up) {                                              // raise the offset: this chunk and the running sum here; the
              if (have) {                                          // P chunks already in TMEM and O_r after the loop (lfall)
                const float f = ex2f(-cm);
                acc.x *= f; acc.y *= f;
                lfall -= cm;
              }
              mr = have ? mr + cm : cm;
#pragma unroll
              for (int i2 = 0; i2 < 32; ++i2) sc[i2] -= cm;
            }
            lfc[c] = lfall;
            uint32_t pk[16];
#pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) {
              const float2 pv = make_float2(ex2f(sc[k2 * 2]), ex2f(sc[k2 * 2 + 1]));
              acc = add2(acc, pv);
              pk[k2] = pack2(pv.x, pv.y);
            }
            tmem_wait_ld();                    // chunk c+1 landed (and every S column chunk c's P covers was read)
            tmem_st8(sb + c * 16, pk);         // P over S: chunk c's 16 columns cover S columns already read
            tmem_st8(sb + c * 16 + 8, pk + 8);
          }
          lr = acc.x + acc.y;
          if (__any_sync(0xffffffffu, lfall != 0.f)) {           // rare, warp-uniform (tcgen05.ld / st are warp-collective)
            // the tile's P chunks written before a raise, and O_r (previous tiles) and l_r, to the final offset. PV(previous tile,
            // row r) = sub-step xs - R: waited on through its K/V stage barrier (R <= NS)
            tmem_wait_st();
#pragma unroll
            for (int c2 = 0; c2 < 3; ++c2) {
              const float f = ex2f(lfall - lfc[c2]);
              float pw[16];
              tmem_ld16(sb + c2 * 16, pw);
              tmem_wait_ld();
              uint32_t pq[16];
#pragma unroll
              for (int k2 = 0; k2 < 16; ++k2) {
                const float2 pp = bf2f(__float_as_uint(pw[k2]));
                pq[k2] = pack2(pp.x * f, pp.y * f);
              }
              tmem_st8(sb + c2 * 16, pq);
              tmem_st8(sb + c2 * 16 + 8, pq + 8);
            }
            const float fo = ex2f(lfall);
            if (j > 0) {
              const int yp = xs - R;
              wait(kve + yp % NKV, (yp / NKV) & 1);
              tc_fence_after();
#pragma unroll
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
// The head-dim-32 core's backward (triattn_sm100.cu: a key-owned dK / dV kernel and a query-owned dQ / dbias kernel, both
// recomputing P from the LSE; the operand fixed for a task in TMEM or resident, the streaming one from shared memory) on the operands
// of a 64-channel head.  The per-score math is the head-dim-32 kernels' (it does not depend on the head dim); what changes is the
// tensor-memory budget: an accumulator row of a 64-channel head takes 64 columns, so the key side owns ONE pair row a task (its dV | dK
// take 128 columns; the two gradient groups alternate its 64-query stages) and the query side two (dQ 2 x 64, Q | dO 2 x 64 bf16).
// Delta arrives as two halves per head [B, N, 2H, S] (the gate backward's per-32-column sums), added where it is read.
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
__global__ void __launch_bounds__(256) triattn_biasT64(const __nv_bfloat16* __restrict__ bias, float* __restrict__ out, int S) {
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

// ---- kernel 1: dK, dV.  Task = (b, h, 128-key tile, pair row n), all queries in 64-wide stages, the two gradient groups taking
// alternate stages (stage w belongs to group w & 1):
//   S^T MMA warp:  S^T = K . Q^T, dP^T = V . dO^T    (ss: K / V of the task resident in shared; M = 128 k, N = 64 q, K = 64)
//   grad group g:  P^T, dS^T (bf16) -> its P^T | dS^T TMEM buffer
//   grad MMA warp: dV += P^T . dO, dK += dS^T . Q     (ts: A from TMEM, B MN-major; N = 64, K = 64)
// TMEM: S^T | dP^T of group g at 128 g (64 + 64 fp32), P^T | dS^T of group g at 256 + 64 g (32 + 32 bf16 pairs), dV | dK at 384 | 448.
// Each stage has its own bias^T tile (one pair row a task: nothing to share it with).  The drain: group 0 stores dV, group 1 dK
// (a thread a key row, 128 bytes of each, straight from registers).
constexpr int KB_QW = 64, KB_QT = KB_QW * D, KB_BT = BN * KB_QW;   // Q / dO stage [64][64] (8 KiB); bias^T [2 halves][128 k][32 q] fp32
constexpr int KB_NQ = 3, KB_NB = 3, KB_THREADS = 352;              // 0 TMA, 1 S^T MMA, 2-5 / 6-9 grad groups, 10 grad MMA
constexpr int KB_COL_P = 256, KB_COL_G = 384;
constexpr int KB_LD = 3 * KB_QW;                                   // a stage's lse | delta (two halves), fp32
constexpr int KB_SMEM = 1024 + (2 * 2 * KVT + KB_NQ * 2 * KB_QT) * 2 + KB_NB * KB_BT * 4 + KB_NQ * KB_LD * 4 + 256;
constexpr uint32_t ID_ST = idesc_bf16(128, KB_QW, 0, 0);
constexpr uint32_t ID_G = idesc_bf16(128, D, 0, 1);
static_assert(KB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(KB_THREADS, 1) triattn_bwd64_kv(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][.. H*64], box (64, 64), 128B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias^T * log2e fp32 [B*H*S (k)][S (q)], box (32, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,   // [B, N, H, S], [B, N, 2H, S]
    __nv_bfloat16* __restrict__ DK, __nv_bfloat16* __restrict__ DV, long ldo) {   // [B*N*S rows at stride ldo][H*64]
  const int NK = S / BN, NQS = S / KB_QW, SPT = NQS;      // NQS even: stage w's group is w & 1 in every task
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 task buffers][K | V]
  __nv_bfloat16* sQ = sKV + 2 * 2 * KVT;                         // [NQ][Q | dO]
  float* sB = reinterpret_cast<float*>(sQ + KB_NQ * 2 * KB_QT);  // [NB][2 q halves][128 k][32 q]
  float* sLD = sB + KB_NB * KB_BT;                               // [NQ][lse 64 | delta 64 | delta 64]
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
        expect_tx(A(kvf + kb), 2 * KVT * 2);
        const int rw = (b_ * N + n) * S + kt * BN;
        load_2d(&kmap, A(sKV + kb * 2 * KVT), A(kvf + kb), h_ * D, rw);
        load_2d(&vmap, A(sKV + kb * 2 * KVT + KVT), A(kvf + kb), h_ * D, rw);
        for (int w = 0; w < SPT; ++w, ++y) {
          const int bs = y % KB_NB, sl = y % KB_NQ;
          if (y >= KB_NB) wait(A(be + bs), ((y / KB_NB) - 1) & 1);
          expect_tx(A(bf + bs), KB_BT * 4);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, A(sB + bs * KB_BT + hh * BN * 32), A(bf + bs), w * KB_QW + hh * 32, bh * S + kt * BN);
          if (y >= KB_NQ) wait(A(qe + sl), ((y / KB_NQ) - 1) & 1);
          expect_tx(A(qf + sl), 2 * KB_QT * 2 + KB_LD * 4);
          const int rq = (b_ * N + n) * S + w * KB_QW;
          load_2d(&qmap, A(sQ + sl * 2 * KB_QT), A(qf + sl), h_ * D, rq);
          load_2d(&dmap, A(sQ + sl * 2 * KB_QT + KB_QT), A(qf + sl), h_ * D, rq);
          const size_t st = ((size_t)(b_ * N + n) * H + h_) * S + w * KB_QW;
          const size_t sd = ((size_t)(b_ * N + n) * 2 * H + 2 * h_) * S + w * KB_QW;
          bulk_load(A(sLD + sl * KB_LD), LSE + st, KB_QW * 4, A(qf + sl));
          bulk_load(A(sLD + sl * KB_LD + KB_QW), DELTA + sd, KB_QW * 4, A(qf + sl));
          bulk_load(A(sLD + sl * KB_LD + 2 * KB_QW), DELTA + sd + S, KB_QW * 4, A(qf + sl));
        }
      }
    }
  } else if (warp == 1 || warp == 10) {
    // converged issue loops, one elected lane's tcgen05 ops predicated; ring slots and phases as incremental counters
    const uint32_t L = elect_one() ? 1u : 0u;
    constexpr uint64_t SLOT = 2 * KB_QT * 2 / 16, MAT = KB_QT * 2 / 16, KVM = KVT * 2 / 16, KVBUF = 2 * KVT * 2 / 16;
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
        const uint32_t d = tmem + r * 128;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) mma_ss_if(L, d, ak + 2 * ks, bq + 2 * ks, ID_ST, ks ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) mma_ss_if(L, d + 64, ak + KVM + 2 * ks, bq + MAT + 2 * ks, ID_ST, ks ? 1u : 0u);
        mma_commit_if(L, sfa + r * 8);
        if (w == SPT - 1) mma_commit_if(L, kvea + kb * 8);  // the task's K / V buffer is free once these retire
        if (++bs == KB_NB) { bs = 0; bsp ^= 1; }
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    } else {
      const uint64_t dM0 = desc_mn128(sQ, KB_QT * 2);
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
        const uint32_t a = tmem + KB_COL_P + r * 64;
#pragma unroll
        for (int kk = 0; kk < KB_QW / 16; ++kk) {
          const uint32_t acc = (!first || kk) ? 1u : 0u;
          mma_ts_if(L, tG, a + kk * 8, bm + MAT + kk * 128, ID_G, acc);          // dV += P^T dO
          mma_ts_if(L, tG + D, a + 32 + kk * 8, bm + kk * 128, ID_G, acc);      // dK += dS^T Q
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
    const uint32_t tS = tmem_at(tmem + gi * 128, qq * 32, 0), tP = tmem_at(tmem + KB_COL_P + gi * 64, qq * 32, 0);
    int js = 0, lt = 0;                            // this group's stage count (phase of sf / se / pf / ge), task
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int kt, n, bh; decode(t, kt, n, bh);
      const int b_ = bh / H, h_ = bh % H;
      for (int qs = gi; qs < NQS; qs += 2, ++js) {
        const int y = lt * SPT + qs, bs = y % KB_NB, sl = y % KB_NQ;
        wait(A(sf + gi), js & 1);                   // implies the bias^T tile and the lse / delta slot (see the S^T issuer)
        tc_fence_after();
        const float* ld = sLD + sl * KB_LD;
#pragma unroll
        for (int c = 0; c < 2; ++c) {              // 32-query chunks (registers)
          float sv[32], dp[32], bl[32];
          tmem_ld32(tS + c * 32, sv);
          tmem_ld32(tS + 64 + c * 32, dp);
          {
            const float* bp = sB + bs * KB_BT + c * BN * 32 + k * 32;
#pragma unroll
            for (int q4 = 0; q4 < 8; ++q4) {
              const float4 u = *reinterpret_cast<const float4*>(bp + ((q4 ^ (k & 7)) << 2));
              bl[q4 * 4] = u.x; bl[q4 * 4 + 1] = u.y; bl[q4 * 4 + 2] = u.z; bl[q4 * 4 + 3] = u.w;
            }
          }
          tmem_wait_ld();
          if (c == 1) {                             // both chunks loaded: the group's S^T buffer takes its next stage
            fence_proxy_async();                    // the bias^T tile's reads, before its release: an arrive alone does not wait for outstanding LDS
            tc_fence_before();
            __syncwarp();
            if (lane == 0) { arrive(A(se + gi)); arrive(A(be + bs)); }
          }
          uint32_t pk[16], dk[16];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float4 l4 = *reinterpret_cast<const float4*>(ld + c * 32 + 4 * j);
            const float4 da = *reinterpret_cast<const float4*>(ld + KB_QW + c * 32 + 4 * j);
            const float4 db = *reinterpret_cast<const float4*>(ld + 2 * KB_QW + c * 32 + 4 * j);
            const float2 d01 = add2(make_float2(da.x, da.y), make_float2(db.x, db.y));
            const float2 d23 = add2(make_float2(da.z, da.w), make_float2(db.z, db.w));
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              const int i = 4 * j + 2 * h2;
              const float2 lm = h2 ? make_float2(l4.z, l4.w) : make_float2(l4.x, l4.y);
              const float2 dm = h2 ? d23 : d01;
              const float2 x = fma2(lm, neg1, fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])));
              const float2 pp = make_float2(ex2f(x.x), ex2f(x.y));
              const float2 ds = mul2(pp, fma2(dm, neg1, make_float2(dp[i], dp[i + 1])));
              pk[i >> 1] = pack2(pp.x, pp.y);
              dk[i >> 1] = pack2(ds.x, ds.y);
            }
          }
          if (c == 0 && js >= 1) {                  // this group's previous gradient GEMMs read the P^T | dS^T buffer
            wait(A(ge + gi), (js - 1) & 1);
            tc_fence_after();
          }
          tmem_st8(tP + c * 16, pk); tmem_st8(tP + c * 16 + 8, pk + 8);
          tmem_st8(tP + 32 + c * 16, dk); tmem_st8(tP + 32 + c * 16 + 8, dk + 8);
        }
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(pf + gi));
      }
      // ---- the drain: group 0 dV, group 1 dK (x scale), a thread a key row ----
      wait(A(dfr), lt & 1);
      tc_fence_after();
      float a0[32], a1[32];
      const uint32_t src = tmem_at(tmem + KB_COL_G + gi * D, qq * 32, 0);
      tmem_ld32(src, a0);
      tmem_ld32(src + 32, a1);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(A(de));                // the accumulators are free for the next task
      __nv_bfloat16* dst = (gi ? DK : DV) + ((size_t)(b_ * N + n) * S + kt * BN + k) * ldo + h_ * D;
      const float s = gi ? scale : 1.f;
      store_row32(dst, a0, s);
      store_row32(dst + 32, a1, s);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- kernel 2: dQ and dbias.  Task = (b, h, 128-query tile) x QR = 2 pair rows, all keys in 64-wide sub-tiles (stage = (sub-tile, row)).
//   MMA:   S = Q_r . K^T, dP = dO_r . V^T        (ts: A = Q_r / dO_r from TMEM, B = K / V K-major; M = 128 q, N = 64 k, K = 64)
//   grad:  dS (bf16) over dP; dbias summed over the QR rows in registers   (group gi: key columns 32 gi .. of the stage)
//   MMA:   dQ_r += dS . K                        (ts: A = dS from TMEM, B = K MN-major; N = 64)
// TMEM: S | dP buffers at 0 / 128, dQ_r at 256 + 64 r, Q_r at 384 + 32 r, dO_r at 448 + 32 r.  Group gi copies row gi's Q / dO
// into TMEM and drains its dQ (straight from registers, a thread a query row).
constexpr int QR = 2;
constexpr int QB_KW = 64, QB_KT = QB_KW * D, QB_BT = BM * QB_KW;   // K / V sub-tile [64][64] (8 KiB), bias [128 q][64 k] 128B swizzle
constexpr int QB_NKV = 3, QB_NB = 2, QB_NS = 2, QB_THREADS = 320;
constexpr int QB_COL_G = QB_NS * 128;                               // dQ_r at 256 + 64 r
constexpr int QB_COL_A = QB_COL_G + QR * D;                         // Q_r at 384 + 32 r, dO_r at 384 + 32 QR + 32 r
constexpr int QB_DBT = BM * 32;                                     // a group's dbias partial [128 q][32 k] fp32, 128B swizzle (16 KiB)
constexpr int QB_LD = 3 * BM;                                       // a row's lse | delta (two halves)
constexpr int QB_SMEM = 1024 + (QR * 2 * QT + QB_NKV * 2 * QB_KT + QB_NB * QB_BT) * 2 + QR * QB_LD * 4 + 2 * 2 * QB_DBT * 4 + 256;
constexpr uint32_t ID_SQ = idesc_bf16(128, QB_KW, 0, 0);
static_assert(QB_COL_A + 2 * QR * 32 <= 512, "TMEM");
static_assert(QB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(QB_THREADS, 1) triattn_bwd64_q(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][.. H*64], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (64, 64), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,
    __nv_bfloat16* __restrict__ DQ, long ldo,            // [B*N*S rows at stride ldo][H*64]
    const __grid_constant__ CUtensorMap dbmap) {         // dbias fp32 [B*H*S][S], zeroed; box (32, 128), 128B swizzle (TMA reduce-add)
  const int QTL = S / BM, NG = (N + QR - 1) / QR, NKS = S / QB_KW, SPT = QR * NKS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smb);    // [QR][Q | dO] staging for TMEM
  __nv_bfloat16* sKV = sA + QR * 2 * QT;                         // [NKV][K | V]
  __nv_bfloat16* sB = sKV + QB_NKV * 2 * QB_KT;                  // [NB][128 q][64 k]
  float* sLD = reinterpret_cast<float*>(sB + QB_NB * QB_BT);     // [QR][lse 128 | delta 128 | delta 128]
  float* sDB = sLD + QR * QB_LD;                                 // [2 groups][2 buffers][QB_DBT] dbias partials
  uint64_t* bars = reinterpret_cast<uint64_t*>(sDB + 2 * 2 * QB_DBT);
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
  auto decode = [&](int t, int& qt, int& g, int& bh) { qt = t % QTL; g = (t / QTL) % NG; bh = t / (QTL * NG); };
  auto nrow = [&](int g, int r) { return min(g * QR + r, N - 1); };

  if (warp == 0) {
    if (lane == 0) {
      auto staging = [&](int t) {                // Q / dO rows, lse, delta of task t -> staging
        int qt, g, bh; decode(t, qt, g, bh);
        const int b_ = bh / H, h_ = bh % H;
        expect_tx(A(af), QR * 2 * QT * 2 + QR * QB_LD * 4);
        for (int r = 0; r < QR; ++r) {
          const int rw = (b_ * N + nrow(g, r)) * S + qt * BM;
          load_2d(&qmap, A(sA + r * 2 * QT), A(af), h_ * D, rw);
          load_2d(&dmap, A(sA + r * 2 * QT + QT), A(af), h_ * D, rw);
          const size_t st = ((size_t)(b_ * N + nrow(g, r)) * H + h_) * S + qt * BM;
          const size_t sd = ((size_t)(b_ * N + nrow(g, r)) * 2 * H + 2 * h_) * S + qt * BM;
          bulk_load(A(sLD + r * QB_LD), LSE + st, BM * 4, A(af));
          bulk_load(A(sLD + r * QB_LD + BM), DELTA + sd, BM * 4, A(af));
          bulk_load(A(sLD + r * QB_LD + 2 * BM), DELTA + sd + S, BM * 4, A(af));
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
    const uint64_t dK0 = desc_k128(sKV), dM0 = desc_mn128(sKV, QB_KT * 2);
    constexpr uint64_t SLOT = 2 * QB_KT * 2 / 16, MAT = QB_KT * 2 / 16;
    auto stt = [&](int y, int lt, int w) {
      const int r = w % QR, sl = y % QB_NKV, b = y % QB_NS;
      if (w == 0) wait(A(at), lt & 1);
      wait(A(kvf + sl), (y / QB_NKV) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t bk = dK0 + sl * SLOT;
        const uint32_t d = tmem + b * 128, a = tmem + QB_COL_A + r * 32;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) mma_ts(d, a + ks * 8, bk + 2 * ks, ID_SQ, ks ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) mma_ts(d + 64, a + QR * 32 + ks * 8, bk + MAT + 2 * ks, ID_SQ, ks ? 1u : 0u);
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
        const uint32_t acc = kb ? 1u : 0u, d = tmem + QB_COL_G + r * D;
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
    const int gi = (warp - 2) >> 2, c0 = gi * 32;    // key columns of a stage; row gi for the Q / dO copy and the dQ drain
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
        for (int m2 = 0; m2 < 2; ++m2) {                    // Q, then dO of row gi: [128][64] 128B-swizzled staging -> 32 TMEM columns
          const unsigned char* base = reinterpret_cast<const unsigned char*>(sA + gi * 2 * QT + m2 * QT) + row * 128;
          uint32_t wv[32];
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            const uint4 u = *reinterpret_cast<const uint4*>(base + ((c ^ (row & 7)) << 4));
            wv[c * 4] = u.x; wv[c * 4 + 1] = u.y; wv[c * 4 + 2] = u.z; wv[c * 4 + 3] = u.w;
          }
          const uint32_t col = m2 * QR * 32 + gi * 32;
#pragma unroll
          for (int c = 0; c < 4; ++c) tmem_st8(ta + col + c * 8, wv + c * 8);
        }
#pragma unroll
        for (int r = 0; r < QR; ++r) {
          lse[r] = sLD[r * QB_LD + row];
          dl[r] = sLD[r * QB_LD + BM + row] + sLD[r * QB_LD + 2 * BM + row];
        }
        fence_proxy_async();                        // the staging's reads (Q / dO, lse / delta), before its release: an arrive alone does not wait for outstanding LDS
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
        fence_proxy_async();                        // the bias tile's reads, before its release: an arrive alone does not wait for outstanding LDS
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
        // the group's [128 q][32 k] partial -> shared -> one TMA reduce-add
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
      // ---- dQ of row gi (x scale), a thread a query row ----
      wait(A(df), lt & 1);
      tc_fence_after();
      float a0[32], a1[32];
      const uint32_t src = tmem_at(tmem + QB_COL_G + gi * D, qq * 32, 0);
      tmem_ld32(src, a0);
      tmem_ld32(src + 32, a1);
      tmem_wait_ld();
      tc_fence_before();
      if (g * QR + gi < N) {
        __nv_bfloat16* dst = DQ + ((size_t)(b_ * N + g * QR + gi) * S + qt * BM + row) * ldo + (bh % H) * D;
        store_row32(dst, a0, scale);
        store_row32(dst + 32, a1, scale);
      }
    }
  }
  if (warp >= 2 && (warp & 3) == 2 && lane == 0) bulk_wait<0>();
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

// q, k, v [B, N, S, H, 64] bf16 (rows (b, n, s) of all heads at one common stride, e.g. column slices of one projection output, or
// contiguous); bias [B, H, S, S] bf16 (masked keys already filled); S % 128 == 0.
// -> (out [B, N, S, H, 64] bf16 contiguous, lse fp32 [B, N, H, S] (empty unless want_lse), flags int32 [1]: rows whose sum left
//     (0, 3e38), a per-device counter never re-zeroed)
// out_: a contiguous bf16 tensor like q to write into (a group of samples' part of a batch's out), or empty (fresh)
// want_lse: also the base-2 LSE fp32 [B, N, H, S] for the backward (into lse_ when it is given, a group of samples' part of a batch's)
std::vector<torch::Tensor> triattn_fwd64(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, double scale,
                                         torch::Tensor out_, bool want_lse, torch::Tensor lse_) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 64] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld,
                "q, k, v: same shape, [.., H, 64] dense per row, rows at one common stride");
  TORCH_CHECK(ld >= H * D && ld % 8 == 0, "q / k / v row stride: >= H*64 and a multiple of 8 elements");
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
  CUtensorMap qm = make_map<2>(q.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "q");
  CUtensorMap km = make_map<2>(k.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "k");
  CUtensorMap vm = make_map<2>(v.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "v");
  CUtensorMap bm = make_map<2>(bias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  const long nt = B * H * (S / BM) * ((N + R - 1) / R);
  TORCH_CHECK(nt < (1L << 31), "too many tasks for one launch");
  const int ntasks = (int)nt;
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_fwd64_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  triattn_fwd64_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)H, (int)S, ntasks, (float)scale * L2E, qm, km, vm,
      bm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags};
}

// Backward of triattn_fwd64.  q, k, v, dout [B, N, S, H, 64] bf16 (rows (b, n, s) of all heads, each tensor at its own row stride, a
// multiple of 8 elements: column slices of one projection output, or contiguous); bias [B, H, S, S] bf16 (masked keys filled);
// lse fp32 [B, N, H, S] (triattn_fwd64's); delta fp32 [B, N, 2H, S] = rowsum(dout o out) over each head's two 32-channel halves.
// dq_out / dk_out / dv_out: [B*N*S, H*64] views to store into (dense rows at a 16-byte-multiple stride, e.g. column slices of one
// gradient buffer), or empty (fresh [B, N, S, H, 64]).
// -> (dq, dk, dv, dbias fp32 [B, H, S, S]: summed over n by fp32 reduce-adds)
std::vector<torch::Tensor> triattn_bwd64(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor dout,
                                         torch::Tensor lse, torch::Tensor delta, double scale, torch::Tensor dq_out, torch::Tensor dk_out,
                                         torch::Tensor dv_out) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 64] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  for (const auto* t : {&q, &k, &v, &dout}) {
    const long ld = t->stride(2);
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(1) == S * ld && t->stride(0) == N * S * ld && ld >= H * D && ld % 8 == 0,
                "q, k, v, dout: [B, N, S, H, 64] bf16, [.., H, 64] dense per row, rows at a 16-byte-multiple stride");
  }
  TORCH_CHECK(k.stride(2) == q.stride(2) && v.stride(2) == q.stride(2), "q, k, v: one row stride");
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous() && bias.numel() == B * H * S * S, "bias: [B, H, S, S] bf16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.is_contiguous() && lse.numel() == B * N * H * S, "lse: fp32 [B, N, H, S]");
  TORCH_CHECK(delta.scalar_type() == torch::kFloat32 && delta.is_contiguous() && delta.numel() == B * N * 2 * H * S, "delta: fp32 [B, N, 2H, S]");
  const bool given = dq_out.defined() && dq_out.numel() > 0;
  auto dq = given ? dq_out : torch::empty(q.sizes(), q.options()), dk = given ? dk_out : torch::empty(q.sizes(), q.options()),
       dv = given ? dv_out : torch::empty(q.sizes(), q.options());
  const long ldo = given ? dq.stride(0) : H * D;
  if (given)
    for (const auto* t : {&dq, &dk, &dv})
      TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->dim() == 2 && t->size(0) == B * N * S && t->size(1) == H * D && t->stride(1) == 1 &&
                  t->stride(0) == ldo && ldo % 8 == 0, "dq / dk / dv outputs: [rows, H*64] bf16, dense rows at one 16-byte-multiple stride");
  auto db = torch::zeros({B, H, S, S}, q.options().dtype(torch::kFloat32));
  auto bT = torch::empty({B, H, S, S}, q.options().dtype(torch::kFloat32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  auto m2 = [&](const torch::Tensor& t, uint32_t box, const char* what) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {(uint64_t)t.stride(2)}, {D, box}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap q64 = m2(q, KB_QW, "q"), d64 = m2(dout, KB_QW, "dout"), k128 = m2(k, BN, "k"), v128 = m2(v, BN, "v");
  CUtensorMap q128 = m2(q, BM, "q"), d128 = m2(dout, BM, "dout"), k64 = m2(k, QB_KW, "k"), v64 = m2(v, QB_KW, "v");
  CUtensorMap bkv = make_map<2>(bT.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, BN}, CU_TENSOR_MAP_SWIZZLE_128B,
                                "bias^T", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap bq = make_map<2>(bias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {QB_KW, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  CUtensorMap dbm = make_map<2>(db.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "dbias",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd64_kv, cudaFuncAttributeMaxDynamicSharedMemorySize, KB_SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd64_q, cudaFuncAttributeMaxDynamicSharedMemorySize, QB_SMEM));
    attr = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sms = num_sms(q.device().index());
  const float scl = (float)scale * L2E;
  auto bf = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  triattn_biasT64<<<dim3(S / 32, S / 32, B * H), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
                                                                  bT.data_ptr<float>(), (int)S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long n1 = B * H * (S / BN) * N, n2 = B * H * (S / BM) * ((N + QR - 1) / QR);
  TORCH_CHECK(n1 < (1L << 31) && rows < (1ull << 31), "too many tasks / rows for one launch");
  triattn_bwd64_kv<<<(int)std::min<long>(n1, sms), KB_THREADS, KB_SMEM, stream>>>((int)N, (int)H, (int)S, (int)n1, scl, (float)scale, q64, d64,
      k128, v128, bkv, lse.data_ptr<float>(), delta.data_ptr<float>(), bf(dk), bf(dv), ldo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  triattn_bwd64_q<<<(int)std::min<long>(n2, sms), QB_THREADS, QB_SMEM, stream>>>((int)N, (int)H, (int)S, (int)n2, scl, (float)scale, q128, d128,
      k64, v64, bq, lse.data_ptr<float>(), delta.data_ptr<float>(), bf(dq), ldo, dbm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd64", &triattn_fwd64, "sm100 triangle-attention forward, 64-channel heads (bf16): q/k/v/out [B, N, S, H, 64], optional base-2 LSE",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"), py::arg("out"), py::arg("want_lse") = false,
        py::arg("lse") = torch::Tensor());
  m.def("triattn_bwd64", &triattn_bwd64, "sm100 triangle-attention backward, 64-channel heads: (dq, dk, dv [.., H, 64] bf16, dbias fp32 [B, H, S, S])",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("dout"), py::arg("lse"), py::arg("delta"), py::arg("scale"),
        py::arg("dq_out"), py::arg("dk_out"), py::arg("dv_out"));
}
