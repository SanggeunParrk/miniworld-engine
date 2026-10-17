// Triangle-attention core backward for H100 (sm_90a), D = 32, bf16 operands, fp32 accumulation.  With the forward's base-2 LSE,
// P = 2^(x - lse) (x = scale log2e q.k + log2e bias, the forward's logit), dP = dO V^T, Delta = rowsum(dO o O), dS = P o (dP - Delta):
//   dV = P^T dO,  dK = scale dS^T Q,  dQ = scale dS K,  dbias[b, h] = sum_n dS[b, n, h].
// The B200 kernel boundaries (b200_sources/triattn_sm100.cu): a key-owned dK / dV kernel and a query-owned dQ / dbias kernel, both
// recomputing P from the LSE (dS never reaches HBM).  Operands as the forward: q / k / v / dO [B, N, S, H, 32] (the projection
// layout), sbias = bf16(bias / scale) [B, H, S, S] -- the score accumulators start from it (bias^T for the key-owned kernel: the
// same tile through ldmatrix.trans), lse / delta fp32 [B, N, H, S].
#include <torch/extension.h>
#include "sm90.cuh"
using namespace sm90;

namespace {
constexpr int D = 32;
}  // namespace

// ---- delta[b, n, h, s] = sum_d dout . out (fp32): one thread per (row (b, n, s), h), 64 B of each
__global__ void __launch_bounds__(256) triattn_delta_kernel(const __nv_bfloat16* __restrict__ dout, const __nv_bfloat16* __restrict__ out,
                                                           float* __restrict__ delta, long rows, int S, int H, long ld_do, long ld_o) {
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= rows * H) return;
  const long row = i / H;
  const int h = (int)(i - row * H);
  const uint4* a = reinterpret_cast<const uint4*>(dout + row * ld_do + h * D);
  const uint4* b = reinterpret_cast<const uint4*>(out + row * ld_o + h * D);
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 x = a[c], y = b[c];
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 p = bf2f(xs[e]), q = bf2f(ys[e]); s = fmaf(p.x, q.x, fmaf(p.y, q.y, s)); }
  }
  const long bn = row / S, sidx = row - bn * S;
  delta[(bn * H + h) * S + sidx] = s;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// dK, dV.  Task = (b, h, 128-key tile) x KR = 2 pair rows; the persistent grid walks tasks with the key tile fastest, then the head.
// 384 threads: warpgroup 0 = TMA producer; warpgroups 1 / 2 own keys 0-63 / 64-127 of the tile, both rows.  The rows' K / V land once
// per task and live in registers as wgmma A fragments.  The queries stream in 64-wide steps (a stage = the bias^T tile shared by the
// rows, and per row Q, dO, lse, Delta); a sub-step (step s, row r):
//   G:  S^T = bias^T (ldmatrix.trans of the bias tile) + K_r Q^T,  dP^T = V_r dO^T       (RS wgmma, m64 keys x n64 queries)
//   X:  P^T = 2^(S^T scale log2e - lse),  dS^T = P^T o (dP^T - Delta)                     (lse / Delta per query = accumulator column)
//   M:  dV_r += P^T dO,  dK_r += dS^T Q                                                  (RS wgmma, dO / Q MN-major from shared)
// in FlashAttention-3 order: G(y), M(y - 1), wait for G(y), X(y), wait for M(y - 1) -- nothing left in flight at a loop back edge.
#ifndef TB_DESCOFF
#define TB_DESCOFF 0      // 1: descriptors built once and moved by offsets -- measured 7 % slower (bwd_kv, L768, clock-locked)
#endif
#ifndef TB_PP
#define TB_PP 1           // the consumer warpgroups take turns issuing their GEMMs (key side: -8 % at L384, -2 % at L768)
#endif
#ifndef TB_PPQ
#define TB_PPQ 0
#endif
#ifndef TB_TRACE
#define TB_TRACE 0        // experiments: phase timestamps of CTA 0, warp 0 of each consumer warpgroup (key-side kernel)
#endif
#if TB_TRACE
__device__ unsigned g_tkv[2][256][8];
#define TRK(k) do { if (blockIdx.x == 0 && (T >> 5) == 0 && lane == 0 && ys < 256) g_tkv[w][ys][k] = (unsigned)clock(); } while (0)
#else
#define TRK(k) do {} while (0)
#endif
namespace kv {
constexpr int BK = 128, BQ = 64, KR = 2, NS = 4;
constexpr int BTQ = BQ * BK;                              // bias tile [2 key halves][64 q][64 k], 128B swizzle (16 KiB)
constexpr int QT = BQ * D;                                // Q / dO step tile [64 q][32], 64B swizzle (4 KiB)
constexpr int KT = BK * D;                                // K / V tile [128 k][32] (8 KiB)
constexpr int STG = BTQ * 2 + KR * (2 * QT * 2 + 2 * BQ * 4);   // bytes of a stage: bias | per row (Q | dO | lse | Delta)
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + NS * STG + KR * 2 * KT * 2 + 256;
static_assert(SMEM <= 232448, "smem");
}  // namespace kv

__global__ void __launch_bounds__(kv::THREADS, 1) triattn_bwd_kv_sm90(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,             // q / dO [B*N*S][H*32] bf16, box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,             // k / v [B*N*S][H*32] bf16, box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,             // sbias [B*H*S][S] bf16, box (64, 64), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,   // [B, N, H, S]
    __nv_bfloat16* __restrict__ DK, __nv_bfloat16* __restrict__ DV, long ldo) {  // rows at stride ldo (H*32 when contiguous)
  using namespace kv;
  const int KTILES = S / BK, QS = S / BQ, NG = (N + KR - 1) / KR;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sS = sa(smb), sKV = sS + NS * STG, bar0 = sKV + KR * 2 * KT * 2;
  auto sBias = [&](int s) { return sS + s * STG; };
  auto sQr = [&](int s, int r) { return sS + s * STG + BTQ * 2 + r * (2 * QT * 2 + 2 * BQ * 4); };
  auto sDr = [&](int s, int r) { return sQr(s, r) + QT * 2; };
  auto sLr = [&](int s, int r) { return sQr(s, r) + 2 * QT * 2; };               // lse [64] | Delta [64]
  auto sK = [&](int r) { return sKV + r * 2 * KT * 2; };
  auto sV = [&](int r) { return sK(r) + KT * 2; };
  const uint32_t kvf = bar0, kve = bar0 + 8;                                     // K / V landed / count 8: fragments taken
  auto sf = [&](int s) { return bar0 + 16 + 8 * s; };
  auto se = [&](int s) { return bar0 + 16 + 8 * (NS + s); };                    // count 8
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sS));
    bar_init(bars, 1); bar_init(bars + 1, 8);
    for (int k = 0; k < NS; ++k) { bar_init(bars + 2 + k, 1); bar_init(bars + 2 + NS + k, 8); }
    bar_init_fence();
  }
  __syncthreads();
  auto decode = [&](int t, int& b, int& h, int& kt, int& g) {
    kt = t % KTILES; int r = t / KTILES; h = r % H; r /= H; g = r % NG; b = r / NG;
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&qmap); prefetch_map(&dmap); prefetch_map(&kmap); prefetch_map(&vmap); prefetch_map(&bmap);
      int lt = 0, st = 0, ph = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int b, h, kt, g; decode(t, b, h, kt, g);
        wait(kve, (lt & 1) ^ 1);
        expect_tx(kvf, KR * 2 * KT * 2);
        for (int r = 0; r < KR; ++r) {
          load_2d(&kmap, sK(r), kvf, h * D, row0(b, g * KR + r) + kt * BK);
          load_2d(&vmap, sV(r), kvf, h * D, row0(b, g * KR + r) + kt * BK);
        }
        for (int s = 0; s < QS; ++s) {
          wait(se(st), ph ^ 1);
          expect_tx(sf(st), STG);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sBias(st) + hh * BQ * 128, sf(st), kt * BK + hh * 64, (b * H + h) * S + s * BQ);
          for (int r = 0; r < KR; ++r) {
            const int n = min(g * KR + r, N - 1), rw = row0(b, n) + s * BQ;
            load_2d(&qmap, sQr(st, r), sf(st), h * D, rw);
            load_2d(&dmap, sDr(st, r), sf(st), h * D, rw);
            const long li = ((long)(b * N + n) * H + h) * S + s * BQ;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(sLr(st, r)), "l"(LSE + li), "r"(BQ * 4), "r"(sf(st)) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(sLr(st, r) + BQ * 4), "l"(DELTA + li), "r"(BQ * 4), "r"(sf(st)) : "memory");
          }
          if (++st == NS) { st = 0; ph ^= 1; }
        }
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;   // ldmatrix: this lane's key row
  // ldmatrix.trans of the bias tile: lane -> (q row within an 8-block pair, key chunk)
  const int tq = (lane & 7) + 8 * (lane >> 4), tch = 2 * wi + ((lane >> 3) & 1);
  // descriptors of the first stage's row-0 Q tile, moved by offsets (desc_off) for every stage / row / tile
#if TB_DESCOFF
  const uint64_t dK64 = desc_k64(sQr(0, 0)), dMN64 = desc_mn64(sQr(0, 0), QT * 2);
#define KDESC(off) desc_off(dK64, (off))
#define MDESC(off) desc_off(dMN64, (off))
#else
#define KDESC(off) desc_k64(sQr(0, 0) + (off))
#define MDESC(off) desc_mn64(sQr(0, 0) + (off), QT * 2)
#endif
  uint32_t ka[KR][2][4], va[KR][2][4];                   // K_r / V_r A fragments (the warp's 16 keys x 32 channels)
  float dk[KR][16], dv[KR][16];
  float st_[32], dp[32];                                 // S^T, then P^T; dP^T, then dS^T (fp32)
  uint32_t pa[16], sa_[16];                              // the previous sub-step's P^T / dS^T (bf16 A fragments)
  int st = 0, ph = 0, mst = 0;                           // stage cursor of the G phase; stage of the pending M phase
  int ys = 0;
#if TB_PP                                                // the two warpgroups take turns issuing their GEMMs
  int pp_left = 0;
  {
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    pp_left = mytasks * (QS * KR - 1);
  }
  if (w == 1 && pp_left > 0) named_arrive(1, 256);
#define PPS() named_sync(1 + w, 256)
#define PPA() do { if (!(w == 1 && pp_left == 1)) named_arrive(2 - w, 256); --pp_left; } while (0)
#else
#define PPS() do {} while (0)
#define PPA() do {} while (0)
#endif
  auto G = [&](int r) {
    if (r == 0) wait(sf(st), ph);
    const uint32_t bt = sBias(st) + w * BQ * 128;        // this warpgroup's key half of the bias tile
#pragma unroll
    for (int i2 = 0; i2 < 4; ++i2) {                      // query blocks 2 i2, 2 i2 + 1
      const int qr = 16 * i2 + tq;
      uint32_t r4[4];
      ldsm_x4_t(bt + qr * 128 + (((tch) ^ (qr & 7)) << 4), r4);
      // r4 = (key g, block 2 i2), (key g + 8, block 2 i2), (key g, block 2 i2 + 1), (key g + 8, block 2 i2 + 1)
      const float2 a0 = bf2f(r4[0]), a1 = bf2f(r4[1]), a2 = bf2f(r4[2]), a3 = bf2f(r4[3]);
      st_[8 * i2 + 0] = a0.x; st_[8 * i2 + 1] = a0.y; st_[8 * i2 + 2] = a1.x; st_[8 * i2 + 3] = a1.y;
      st_[8 * i2 + 4] = a2.x; st_[8 * i2 + 5] = a2.y; st_[8 * i2 + 6] = a3.x; st_[8 * i2 + 7] = a3.y;
    }
    wg_fence();
    const uint32_t qo = sQr(st, r) - sQr(0, 0);
    mma_rs_n64<0>(st_, ka[r][0], KDESC(qo), 1u);
    mma_rs_n64<0>(st_, ka[r][1], KDESC(qo + 32), 1u);
    mma_rs_n64<0>(dp, va[r][0], KDESC(qo + QT * 2), 0u);
    mma_rs_n64<0>(dp, va[r][1], KDESC(qo + QT * 2 + 32), 1u);
    wg_commit();
  };
  auto M = [&](int r) {                                   // dV_r += P^T dO, dK_r += dS^T Q (the pending sub-step's stage mst)
    const uint32_t mo = sQr(mst, r) - sQr(0, 0);
#pragma unroll
    for (int kk = 0; kk < BQ / 16; ++kk) {
      const uint32_t a[4] = {pa[4 * kk], pa[4 * kk + 1], pa[4 * kk + 2], pa[4 * kk + 3]};
      const uint32_t c[4] = {sa_[4 * kk], sa_[4 * kk + 1], sa_[4 * kk + 2], sa_[4 * kk + 3]};
      mma_rs_n32<1>(dv[r], a, MDESC(mo + QT * 2 + kk * 1024), 1u);
      mma_rs_n32<1>(dk[r], c, MDESC(mo + kk * 1024), 1u);
    }
    wg_commit();
  };
  auto X = [&](int r) {
    const uint32_t lb = sLr(st, r);
#pragma unroll
    for (int i = 0; i < 8; ++i) {                         // query columns 8i + 2 t4, +1
      const float2 ls = lds64f(lb + (8 * i + 2 * t4) * 4), dl = lds64f(lb + BQ * 4 + (8 * i + 2 * t4) * 4);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float l_ = (e & 1) ? ls.y : ls.x, d_ = (e & 1) ? dl.y : dl.x;
        const float p = ex2f(fmaf(st_[4 * i + e], scl, -l_));
        st_[4 * i + e] = p;
        dp[4 * i + e] = p * (dp[4 * i + e] - d_);
      }
    }
  };
  auto pack = [&]() {
#pragma unroll
    for (int i = 0; i < 16; ++i) { pa[i] = PACKP(st_[2 * i], st_[2 * i + 1]); sa_[i] = PACKP(dp[2 * i], dp[2 * i + 1]); }
  };
  auto release = [&](bool last_row) {                     // the pending M retired: its stage is free after the last row
    if (last_row) {
      __syncwarp();
      if (lane == 0) arrive(se(mst));
      if (++mst == NS) mst = 0;
    }
  };
  int lt = 0;
  for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
    int b, h, kt, g; decode(t, b, h, kt, g);
    wait(kvf, lt & 1);
#pragma unroll
    for (int r = 0; r < KR; ++r)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint32_t off = lrow * D * 2 + (((2 * ks + lcol8) ^ ((lrow >> 1) & 3)) << 4);
        ldsm_x4(sK(r) + off, ka[r][ks]);
        ldsm_x4(sV(r) + off, va[r][ks]);
      }
#pragma unroll
    for (int r = 0; r < KR; ++r) { fence_regs(ka[r][0]); fence_regs(ka[r][1]); fence_regs(va[r][0]); fence_regs(va[r][1]); }
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) arrive(kve);
#pragma unroll
    for (int r = 0; r < KR; ++r)
#pragma unroll
      for (int i = 0; i < 16; ++i) { dk[r][i] = 0.f; dv[r][i] = 0.f; }
    // sub-step (0, 0): no pending M
    G(0);
    wg_wait<0>();
    fence_regs(st_); fence_regs(dp);
    X(0);
    pack();
    for (int s = 0; s < QS; ++s) {
      // (s, 1): M(s, 0)
      TRK(0); PPS(); G(1); TRK(1);
      M(0); PPA(); TRK(2);
      wg_wait<1>(); TRK(3);
      fence_regs(st_); fence_regs(dp);
      X(1); TRK(4);
      if (++st == NS) { st = 0; ph ^= 1; }                // the G cursor moves to the next step
      wg_wait<0>(); TRK(5);
      fence_regs(dk[0]); fence_regs(dv[0]);
      release(false);
      pack(); TRK(6); ++ys;
      if (s + 1 < QS) {                                   // (s + 1, 0): M(s, 1)
        TRK(0); PPS(); G(0); TRK(1);
        M(1); PPA(); TRK(2);
        wg_wait<1>(); TRK(3);
        fence_regs(st_); fence_regs(dp);
        X(0); TRK(4);
        wg_wait<0>(); TRK(5);
        fence_regs(dk[1]); fence_regs(dv[1]);
        release(true);
        pack(); TRK(6); ++ys;
      }
    }
    M(1);                                                 // the last sub-step's M
    wg_wait<0>();
    fence_regs(dk[1]); fence_regs(dv[1]); fence_regs(dk[0]); fence_regs(dv[0]);
    release(true);
    // ---- epilogue: dK_r (x scale), dV_r -> bf16 -> [rows][H*32] (a quad transposes its words: 16 contiguous bytes a lane)
#pragma unroll
    for (int r = 0; r < KR; ++r) {
      const int n = g * KR + r;
      if (n >= N) continue;
      const long rbase = (long)row0(b, n) + kt * BK + 64 * w + 16 * wi + g8;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        uint32_t ink[4], inv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ink[i] = pack2(dk[r][4 * i + 2 * hf] * scale, dk[r][4 * i + 2 * hf + 1] * scale);
          inv[i] = pack2(dv[r][4 * i + 2 * hf], dv[r][4 * i + 2 * hf + 1]);
        }
        const uint4 vk = quad_transpose(ink, lane), vv = quad_transpose(inv, lane);
        const long off = (rbase + 8 * hf) * ldo + h * D + 8 * t4;
        *reinterpret_cast<uint4*>(DK + off) = vk;
        *reinterpret_cast<uint4*>(DV + off) = vv;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// dQ and dbias.  Task = (b, h, 128-query tile) x QR = 4 pair rows; warpgroups 1 / 2 own queries 0-63 / 64-127.  The rows' Q / dO stay
// in shared memory for the task (ldmatrix'd per use); lse / Delta of the lane's two query rows in registers.  The keys stream in
// 64-wide steps (a stage = the bias tile shared by the rows, and per row K, V); a sub-step (step j, row r):
//   G:  S = bias + Q_r K^T,  dP = dO_r V^T               (RS wgmma, m64 queries x n64 keys; the same accumulator layout as the forward)
//   X:  P = 2^(S scale log2e - lse),  dS = P o (dP - Delta),  dbias += dS  (fp32, summed over the QR rows in registers)
//   M:  dQ_r += dS K                                     (RS wgmma, K MN-major from shared)
// with the rows innermost, so after a step's last row the warpgroup's 64 x 64 dbias partial is complete: it goes through a fp32 staging
// tile (128B swizzle) and one TMA reduce-add into dbias [B, H, S, S] (fp32, L2-resident) -- S / QR partials per element in all.
namespace qk {
constexpr int BQ = 128, BK = 64, QR = 4, NS = 2;
constexpr int QT = BQ * D;                                // Q / dO task tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KT = BK * D;                                // K / V step tile [64 k][32] (4 KiB)
constexpr int BT = BQ * BK;                               // bias step tile [128 q][64 k], 128B swizzle (16 KiB)
constexpr int STG = BT * 2 + QR * 2 * KT * 2;             // bytes of a stage: bias | per row (K | V)
constexpr int DBT = 64 * 64 * 4;                          // a warpgroup's dbias staging tile, fp32 [2 halves][64 q][32 k] (16 KiB)
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + QR * 2 * QT * 2 + NS * STG + 2 * 2 * DBT + 256;
static_assert(SMEM <= 232448, "smem");
}  // namespace qk

__global__ void __launch_bounds__(qk::THREADS, 1) triattn_bwd_q_sm90(
    int N, int H, int S, int ntasks, int P, long slabrows, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,             // q / dO [B*N*S][H*32] bf16, box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,             // k / v, box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,             // sbias [B*H*S][S] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap dbmap,            // dbias [B*H*S][S] fp32, box (32, 64), 128B swizzle (TMA reduce-add)
    const float* __restrict__ LSE, const float* __restrict__ DELTA,   // [B, N, H, S]
    __nv_bfloat16* __restrict__ DQ, long ldo) {           // rows at stride ldo (H*32 when contiguous)
  using namespace qk;
  const int QTILES = S / BQ, KS = S / BK, NG = (N + QR - 1) / QR;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sQD = sa(smb), sS = sQD + QR * 2 * QT * 2, sDB = sS + NS * STG, bar0 = sDB + 2 * 2 * DBT;
  auto sQ = [&](int r) { return sQD + r * 2 * QT * 2; };
  auto sD = [&](int r) { return sQ(r) + QT * 2; };
  auto sBias = [&](int s) { return sS + s * STG; };
  auto sKr = [&](int s, int r) { return sS + s * STG + BT * 2 + r * 2 * KT * 2; };
  auto sVr = [&](int s, int r) { return sKr(s, r) + KT * 2; };
  const uint32_t qf = bar0, qe = bar0 + 8;                                       // Q / dO landed / count 8: the task is done with them
  auto sf = [&](int s) { return bar0 + 16 + 8 * s; };
  auto se = [&](int s) { return bar0 + 16 + 8 * (NS + s); };                    // count 8
  // dbias staging buffer (warpgroup w, buffer k): staged (count 4: the warpgroup's warps) / read by the reduce (the storer warp)
  auto dbf = [&](int w_, int k) { return bar0 + 16 + 8 * (2 * NS + 2 * w_ + k); };
  auto dbe = [&](int w_, int k) { return bar0 + 16 + 8 * (2 * NS + 4 + 2 * w_ + k); };
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sQD));
    bar_init(bars, 1); bar_init(bars + 1, 8);
    for (int k = 0; k < NS; ++k) { bar_init(bars + 2 + k, 1); bar_init(bars + 2 + NS + k, 8); }
    for (int k = 0; k < 4; ++k) { bar_init(bars + 2 + 2 * NS + k, 4); bar_init(bars + 2 + 2 * NS + 4 + k, 1); }
    bar_init_fence();
  }
  __syncthreads();
  // Deterministic dbias: a chunk c = (tile (b, h, qt), slab p) owns the pair-row groups [g0, g1) of that tile and is run by ONE CTA, in
  // order; its TMA reduce-adds go to slab p of dbias [P][B*H*S][S].  Every element then receives its adds from one thread in a fixed
  // order (the storer drains each task's reduces before the next), and the slabs are summed in a fixed order afterwards.
  auto decode = [&](int c, int& b, int& h, int& qt, int& g0, int& g1, int& p) {
    const int tile = c / P; p = c % P;
    qt = tile % QTILES; int r = tile / QTILES; h = r % H; b = r / H;
    g0 = (int)((long)p * NG / P); g1 = (int)((long)(p + 1) * NG / P);
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&qmap); prefetch_map(&dmap); prefetch_map(&kmap); prefetch_map(&vmap); prefetch_map(&bmap); prefetch_map(&dbmap);
      int lt = 0, st = 0, ph = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x) {
       int b, h, qt, g0, g1, p; decode(t, b, h, qt, g0, g1, p);
       for (int g = g0; g < g1; ++g, ++lt) {
        wait(qe, (lt & 1) ^ 1);
        expect_tx(qf, QR * 2 * QT * 2);
        for (int r = 0; r < QR; ++r) {
          load_2d(&qmap, sQ(r), qf, h * D, row0(b, g * QR + r) + qt * BQ);
          load_2d(&dmap, sD(r), qf, h * D, row0(b, g * QR + r) + qt * BQ);
        }
        for (int j = 0; j < KS; ++j) {
          wait(se(st), ph ^ 1);
          expect_tx(sf(st), STG);
          load_2d(&bmap, sBias(st), sf(st), j * BK, (b * H + h) * S + qt * BQ);
          for (int r = 0; r < QR; ++r) {
            load_2d(&kmap, sKr(st, r), sf(st), h * D, row0(b, g * QR + r) + j * BK);
            load_2d(&vmap, sVr(st, r), sf(st), h * D, row0(b, g * QR + r) + j * BK);
          }
          if (++st == NS) { st = 0; ph ^= 1; }
        }
       }
      }
    } else if (tid == 32) {                               // the storer: each step's dbias partials -> TMA reduce-adds
      int u = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x) {
       int b, h, qt, g0, g1, p; decode(t, b, h, qt, g0, g1, p);
       for (int g = g0; g < g1; ++g) {
        for (int j = 0; j < KS; ++j, ++u) {
          const int k = u & 1;
          for (int w_ = 0; w_ < 2; ++w_) {
            wait(dbf(w_, k), (u >> 1) & 1);
            const uint32_t buf = sDB + (w_ * 2 + k) * DBT;
            for (int half = 0; half < 2; ++half)
              reduce_add_2d(&dbmap, buf + half * 64 * 128, j * BK + half * 32, (long)p * slabrows + (b * H + h) * S + qt * BQ + 64 * w_);
          }
          bulk_commit();
          bulk_wait_read<0>();
          for (int w_ = 0; w_ < 2; ++w_) arrive(dbe(w_, k));
        }
        bulk_wait<0>();                                   // this task's reduces have landed before the next task's reach the same elements
       }
      }
      bulk_wait<0>();
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4, x7 = lane & 7;
  int dbu = 0;                                            // this warpgroup's flushes so far (buffer dbu & 1)
  const uint32_t qoff = lrow * D * 2;                     // this lane's ldmatrix row of the Q / dO tiles
  float dq[QR][16], lse[QR][2], dlt[QR][2];
  float rlive[QR];
  float s_[32], dp[32], db[32];                           // S then P; dP then dS; the step's dbias partial
  uint32_t da[16];                                        // the pending sub-step's dS (bf16 A fragments)
  int st = 0, ph = 0, mst = 0;                            // G stage cursor; pending M's stage
  auto frag = [&](uint32_t tile, int ks, uint32_t (&a)[4]) { ldsm_x4(tile + qoff + (((2 * ks + lcol8) ^ ((lrow >> 1) & 3)) << 4), a); };
  auto G = [&](int r) {
    if (r == 0) wait(sf(st), ph);
    const uint32_t bt = sBias(st) + lrow * 128;
#pragma unroll
    for (int i2 = 0; i2 < 4; ++i2) {                      // key blocks 2 i2, 2 i2 + 1
      uint32_t r4[4];
      ldsm_x4(bt + ((((2 * i2 + lcol8) & 7) ^ x7) << 4), r4);
      const float2 a0 = bf2f(r4[0]), a1 = bf2f(r4[1]), a2 = bf2f(r4[2]), a3 = bf2f(r4[3]);
      s_[8 * i2 + 0] = a0.x; s_[8 * i2 + 1] = a0.y; s_[8 * i2 + 2] = a1.x; s_[8 * i2 + 3] = a1.y;
      s_[8 * i2 + 4] = a2.x; s_[8 * i2 + 5] = a2.y; s_[8 * i2 + 6] = a3.x; s_[8 * i2 + 7] = a3.y;
    }
    uint32_t qa[2][4], oa[2][4];
    frag(sQ(r), 0, qa[0]); frag(sQ(r), 1, qa[1]); frag(sD(r), 0, oa[0]); frag(sD(r), 1, oa[1]);
    wg_fence();
    mma_rs_n64<0>(s_, qa[0], desc_k64(sKr(st, r)), 1u);
    mma_rs_n64<0>(s_, qa[1], desc_k64(sKr(st, r) + 32), 1u);
    mma_rs_n64<0>(dp, oa[0], desc_k64(sVr(st, r)), 0u);
    mma_rs_n64<0>(dp, oa[1], desc_k64(sVr(st, r) + 32), 1u);
    wg_commit();
  };
  auto M = [&](int r) {                                   // dQ_r += dS K (the pending sub-step, stage mst)
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const uint32_t a[4] = {da[4 * kk], da[4 * kk + 1], da[4 * kk + 2], da[4 * kk + 3]};
      mma_rs_n32<1>(dq[r], a, desc_mn64(sKr(mst, r) + kk * 1024, KT * 2), 1u);
    }
    wg_commit();
  };
  auto X = [&](int r) {
    const float live = rlive[r];                          // 0 for a padding row (computed on row N - 1): not part of dbias
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int hf = (i >> 1) & 1;                        // accumulator rows g (words 0, 1 of a block) / g + 8 (words 2, 3)
      const float p = ex2f(fmaf(s_[i], scl, -lse[r][hf]));
      const float d = p * (dp[i] - dlt[r][hf]);
      dp[i] = d;
      db[i] = fmaf(d, live, db[i]);
    }
  };
  auto pack = [&]() {
#pragma unroll
    for (int i = 0; i < 16; ++i) da[i] = PACKP(dp[2 * i], dp[2 * i + 1]);
  };
  // the step's dbias partial -> fp32 staging [2 halves][64 q][32 k] (128B swizzle: 16-byte chunk c of a row at c ^ (row & 7)) ->
  // one TMA reduce-add per half, issued by one thread; the staging buffer is double-buffered across steps
  auto flush = [&](int b, int h, int qt, int j) {
    const int k = dbu & 1;
    const uint32_t buf = sDB + (w * 2 + k) * DBT;
    wait(dbe(w, k), ((dbu >> 1) & 1) ^ 1);                // the reduce issued from this buffer two steps ago has read it
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int row = 16 * wi + g8 + 8 * hf, col = 8 * i + 2 * t4, half = col >> 5, c = (col & 31) >> 2;
        const uint32_t addr = buf + half * 64 * 128 + row * 128 + ((c ^ (row & 7)) << 4) + (col & 3) * 4;
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" :: "r"(addr), "f"(db[4 * i + 2 * hf]), "f"(db[4 * i + 2 * hf + 1]) : "memory");
      }
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) arrive(dbf(w, k));                    // the storer warp issues the reduce-add
    ++dbu;
#pragma unroll
    for (int i = 0; i < 32; ++i) db[i] = 0.f;
  };
  auto release = [&]() { __syncwarp(); if (lane == 0) arrive(se(mst)); if (++mst == NS) mst = 0; };
#if TB_PPQ                                               // ping-pong on the query side: measured 11-13 % slower, off
  int pp_left = 0;
  {
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    pp_left = mytasks * (KS * QR - 1);
  }
  if (w == 1 && pp_left > 0) named_arrive(1, 256);
#define PQS() named_sync(1 + w, 256)
#define PQA() do { if (!(w == 1 && pp_left == 1)) named_arrive(2 - w, 256); --pp_left; } while (0)
#else
#define PQS() do {} while (0)
#define PQA() do {} while (0)
#endif
  int lt = 0;
  for (int t = blockIdx.x; t < ntasks; t += gridDim.x) {
   int b, h, qt, g0, g1, p; decode(t, b, h, qt, g0, g1, p);
   for (int g = g0; g < g1; ++g, ++lt) {
#pragma unroll
    for (int r = 0; r < QR; ++r) {
      const long li = ((long)(b * N + min(g * QR + r, N - 1)) * H + h) * S + qt * BQ + 64 * w + 16 * wi + g8;
      lse[r][0] = LSE[li]; lse[r][1] = LSE[li + 8]; dlt[r][0] = DELTA[li]; dlt[r][1] = DELTA[li + 8];
      rlive[r] = g * QR + r < N ? 1.f : 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) dq[r][i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) db[i] = 0.f;
    wait(qf, lt & 1);
    G(0);                                                 // sub-step (0, 0): no pending M
    wg_wait<0>();
    fence_regs(s_); fence_regs(dp);
    X(0);
    pack();
    for (int j = 0; j < KS; ++j) {
#pragma unroll
      for (int r = 1; r < QR; ++r) {                      // (j, r): M(j, r - 1)
        PQS(); G(r);
        M(r - 1); PQA();
        wg_wait<1>();
        fence_regs(s_); fence_regs(dp);
        X(r);
        wg_wait<0>();
        fence_regs(dq[r - 1]);
        pack();
      }
      flush(b, h, qt, j);                                 // the step's dbias is complete
      if (++st == NS) { st = 0; ph ^= 1; }
      if (j + 1 < KS) {                                   // (j + 1, 0): M(j, QR - 1)
        PQS(); G(0);
        M(QR - 1); PQA();
        wg_wait<1>();
        fence_regs(s_); fence_regs(dp);
        X(0);
        wg_wait<0>();
        fence_regs(dq[QR - 1]);
        release();
        pack();
      }
    }
    M(QR - 1);
    wg_wait<0>();
#pragma unroll
    for (int r = 0; r < QR; ++r) fence_regs(dq[r]);
    release();
    __syncwarp();
    if (lane == 0) arrive(qe);                            // Q / dO of the task are no longer read (G phases done)
    // ---- epilogue: dQ_r x scale -> bf16 -> [rows][H*32]
#pragma unroll
    for (int r = 0; r < QR; ++r) {
      const int n = g * QR + r;
      if (n >= N) continue;
      const long rbase = (long)row0(b, n) + qt * BQ + 64 * w + 16 * wi + g8;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        uint32_t in[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) in[i] = pack2(dq[r][4 * i + 2 * hf] * scale, dq[r][4 * i + 2 * hf + 1] * scale);
        const uint4 v = quad_transpose(in, lane);
        *reinterpret_cast<uint4*>(DQ + (rbase + 8 * hf) * ldo + h * D + 8 * t4) = v;
      }
    }
   }
  }
}

namespace {
auto bf16p(const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }
void check_proj(const torch::Tensor& t, const torch::Tensor& q, const char* what) {
  const long S = q.size(2), N = q.size(1), ld = t.stride(2);
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.sizes() == q.sizes() && t.stride(4) == 1 && t.stride(3) == D &&
              t.stride(1) == S * ld && t.stride(0) == N * S * ld && ld % 8 == 0, what, ": [B, N, S, H, 32] bf16 projection layout");
}
CUtensorMap proj_map(const torch::Tensor& t, uint32_t rows_box, const char* what) {
  const long B = t.size(0), N = t.size(1), S = t.size(2), H = t.size(3);
  return make_map<2>(t.data_ptr(), {(uint64_t)(H * D), (uint64_t)(B * N * S)}, {(uint64_t)t.stride(2)}, {32, rows_box}, CU_TENSOR_MAP_SWIZZLE_64B, what);
}
}  // namespace

// delta = rowsum(dout o out) fp32 [B, N, H, S]; dout / out [B, N, S, H, 32] bf16 (projection layout)
torch::Tensor triattn_delta(torch::Tensor dout, torch::Tensor out) {
  check_proj(dout, dout, "dout"); check_proj(out, dout, "out");
  const long B = dout.size(0), N = dout.size(1), S = dout.size(2), H = dout.size(3);
  auto delta = torch::empty({B, N, H, S}, dout.options().dtype(torch::kFloat32));
  const long rows = B * N * S;
  triattn_delta_kernel<<<(int)((rows * H + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf16p(dout), bf16p(out), delta.data_ptr<float>(), rows, (int)S, (int)H, dout.stride(2), out.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return delta;
}

// dK, dV of the core: q, k, v, dout [B, N, S, H, 32] bf16; sbias bf16 [B, H, S, S] (bias / scale); lse, delta fp32 [B, N, H, S]
std::vector<torch::Tensor> triattn_bwd_kv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sbias, torch::Tensor dout,
                                          torch::Tensor lse, torch::Tensor delta, double scale, c10::optional<torch::Tensor> dk_dst,
                                          c10::optional<torch::Tensor> dv_dst) {
  check_proj(q, q, "q"); check_proj(k, q, "k"); check_proj(v, q, "v"); check_proj(dout, q, "dout");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % kv::BK == 0, "S % 128 == 0");
  TORCH_CHECK(sbias.scalar_type() == torch::kBFloat16 && sbias.is_contiguous() && sbias.numel() == B * H * S * S, "sbias bf16 [B, H, S, S]");
  for (const auto* t : {&lse, &delta})
    TORCH_CHECK(t->scalar_type() == torch::kFloat32 && t->is_contiguous() && t->numel() == B * N * H * S, "lse / delta fp32 [B, N, H, S]");
  auto dk = dk_dst.has_value() ? *dk_dst : torch::empty({B, N, S, H, D}, q.options());
  auto dv = dv_dst.has_value() ? *dv_dst : torch::empty({B, N, S, H, D}, q.options());
  check_proj(dk, q, "dk"); check_proj(dv, q, "dv");
  TORCH_CHECK(dk.stride(2) == dv.stride(2), "dk / dv: one row stride");
  CUtensorMap qm = proj_map(q, kv::BQ, "q"), dm = proj_map(dout, kv::BQ, "dout"), km = proj_map(k, kv::BK, "k"), vm = proj_map(v, kv::BK, "v");
  CUtensorMap bm = make_map<2>(sbias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "sbias");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd_kv_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::SMEM)); attr = true; }
  const long ntasks = B * H * ((N + kv::KR - 1) / kv::KR) * (S / kv::BK);
  const int grid = (int)std::min<long>(ntasks, num_sms(q.device().index()));
  triattn_bwd_kv_sm90<<<grid, kv::THREADS, kv::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)N, (int)H, (int)S, (int)ntasks, (float)(scale * 1.4426950408889634), (float)scale, qm, dm, km, vm, bm,
      lse.data_ptr<float>(), delta.data_ptr<float>(), bf16p(dk), bf16p(dv), (long)dk.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dk, dv};
}

// dQ and dbias of the core: q, k, v, dout [B, N, S, H, 32] bf16; sbias bf16 [B, H, S, S]; lse, delta fp32 [B, N, H, S];
// dbias: a zeroed fp32 [B, H, S, S] the partials are added into -> dq [B, N, S, H, 32] bf16
torch::Tensor triattn_bwd_q(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sbias, torch::Tensor dout, torch::Tensor lse,
                            torch::Tensor delta, double scale, torch::Tensor dbias, c10::optional<torch::Tensor> dq_dst) {
  check_proj(q, q, "q"); check_proj(k, q, "k"); check_proj(v, q, "v"); check_proj(dout, q, "dout");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % qk::BQ == 0, "S % 128 == 0");
  TORCH_CHECK(sbias.scalar_type() == torch::kBFloat16 && sbias.is_contiguous() && sbias.numel() == B * H * S * S, "sbias bf16 [B, H, S, S]");
  TORCH_CHECK(dbias.scalar_type() == torch::kFloat32 && dbias.is_contiguous() && dbias.numel() == B * H * S * S, "dbias fp32 [B, H, S, S]");
  for (const auto* t : {&lse, &delta})
    TORCH_CHECK(t->scalar_type() == torch::kFloat32 && t->is_contiguous() && t->numel() == B * N * H * S, "lse / delta fp32 [B, N, H, S]");
  auto dq = dq_dst.has_value() ? *dq_dst : torch::empty({B, N, S, H, D}, q.options());
  check_proj(dq, q, "dq");
  CUtensorMap qm = proj_map(q, qk::BQ, "q"), dm = proj_map(dout, qk::BQ, "dout"), km = proj_map(k, qk::BK, "k"), vm = proj_map(v, qk::BK, "v");
  CUtensorMap bm = make_map<2>(sbias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "sbias");
  const long tiles = B * H * (S / qk::BQ), NG = (N + qk::QR - 1) / qk::QR;
  const long P = std::max<long>(1, std::min<long>(NG, num_sms(q.device().index()) / tiles));   // slabs: enough chunks to fill the SMs
  torch::Tensor slabs = P > 1 ? torch::zeros({P, B, H, S, S}, dbias.options()) : dbias;
  CUtensorMap dbm = make_map<2>(slabs.data_ptr(), {(uint64_t)S, (uint64_t)(P * B * H * S)}, {(uint64_t)S}, {32, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "dbias",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd_q_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, qk::SMEM)); attr = true; }
  const long ntasks = tiles * P;
  const int grid = (int)std::min<long>(ntasks, num_sms(q.device().index()));
  triattn_bwd_q_sm90<<<grid, qk::THREADS, qk::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)N, (int)H, (int)S, (int)ntasks, (int)P, (long)(B * H * S), (float)(scale * 1.4426950408889634), (float)scale, qm, dm, km, vm, bm, dbm,
      lse.data_ptr<float>(), delta.data_ptr<float>(), bf16p(dq), (long)dq.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (P > 1) dbias.add_(slabs.sum(0));                    // fixed-order sum of the slabs (dbias arrives zeroed)
  return dq;
}

#if TB_TRACE
torch::Tensor triattn_trace_kv() {
  auto t = torch::empty({2, 256, 8}, torch::TensorOptions().dtype(torch::kInt32));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr<int>(), g_tkv, sizeof(g_tkv)));
  return t;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if TB_TRACE
  m.def("triattn_trace_kv", &triattn_trace_kv);
#endif
  m.def("triattn_bwd_q", &triattn_bwd_q, "sm90 triangle-attention backward, query side: dq [B, N, S, H, 32] bf16; dbias partials added into a fp32 [B, H, S, S]",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sbias"), py::arg("dout"), py::arg("lse"), py::arg("delta"), py::arg("scale"), py::arg("dbias"),
        py::arg("dq_dst") = py::none());
  m.def("triattn_delta", &triattn_delta, "rowsum(dout o out) fp32 [B, N, H, S]");
  m.def("triattn_bwd_kv", &triattn_bwd_kv, "sm90 triangle-attention backward, key side: (dk, dv) [B, N, S, H, 32] bf16",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sbias"), py::arg("dout"), py::arg("lse"), py::arg("delta"), py::arg("scale"),
        py::arg("dk_dst") = py::none(), py::arg("dv_dst") = py::none());
}
