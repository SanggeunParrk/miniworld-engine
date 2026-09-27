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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd", &triattn_fwd, "sm100 triangle-attention forward (D = 32, bf16, no mask): q/k/v/out [B, N, S, H, 32], optional base-2 LSE",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"), py::arg("want_lse") = false);
}
