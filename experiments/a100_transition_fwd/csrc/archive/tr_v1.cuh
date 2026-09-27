// tr_fwd_sm80.cuh -- pair Transition forward, A100 / sm_80, one kernel:
//
//   out = x + Ws (silu(Wa xn) * (Wb xn)),  xn = LN(x),  D = 128, H = 512, bf16 in / out, fp32 accumulation.
//
// Warp = 32 token rows (2 m16 tiles) for the whole hidden dimension: LN(x) stays in registers as the GEMM1 A fragments (64 regs) and the
// output accumulator (32 x 128 fp32, 128 regs) stays in registers across all 512 hidden units, so the [M][512] activation never leaves
// the warp.  Per 16 hidden units: GEMM1 (a | b, 64 MMAs) -> SwiGLU in the C fragments, which ARE the GEMM2 A fragment (m16n8 C = m16k16 A)
// -> GEMM2 (32 MMAs).  Weights stream from L2 through a 2-slot cp.async ring of 32-hidden chunks (24 KB), one CTA barrier per chunk.
//
// Host-side permutations (pack() in transition_a100.py) make every thread's x / out footprint 32 CONTIGUOUS columns of its rows:
//   * GEMM1 k order: mma k-step s, index kk <-> logical column 32 (kk % 8 / 2) + 4 s + 2 (kk / 8) + kk % 2, so thread q of a quad holds
//     columns [32 q, 32 q + 32) of rows g, g + 8 in its A fragments (a straight 4 x 16 B read per row, no shuffle);
//   * GEMM2 n order: physical output column 8 J + 2 q + e <-> logical 32 q + 2 J + e, so the accumulator of n8 tile J is word J of the same
//     64 B the thread read for the LayerNorm: the residual is a word-for-word add and the store is 4 x 16 B per row.
//   * Wa is pre-scaled by 1/2 (exact): silu(a) b = a' b (1 + tanh a'), one MUFU per hidden unit.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct TrParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* w;      // packed [16 chunks][W1: 16 granules x 64 rows x 16 B | W2: 4 granules x 128 rows x 16 B]
  const float4* gb;            // LN affine as float4 slots [gamma | beta][i][q] = columns 32 q + 4 i .. 32 q + 4 i + 3
  __nv_bfloat16* out;          // [T][128]
  int T, num_tiles;
  float eps;
};

struct TrCfg {
  static constexpr int D = 128, H = 512, BM = 128, NTHR = 128, MINB = 2, NWARP = 4;
  static constexpr int CH = 32, NCHUNK = H / CH;            // hidden units per ring slot, chunks per tile
  static constexpr int SLOT_W1 = 2 * CH * D * 2;            // 16 KB: [16 k-granules][64 rows (a | b of 2 x 16 hidden)][16 B]
  static constexpr int SLOT = SLOT_W1 + CH * D * 2;         // + 8 KB: [4 hidden-granules][128 out rows][16 B]
  static constexpr int NST = 2;
  static constexpr int XW = 32 * D * 2;                     // per-warp x staging, 8 KB
  static constexpr int SMEM_W = NST * SLOT, SMEM_X = NWARP * XW;
  static constexpr int NBAR = NST + NWARP;
  static constexpr int SMEM = SMEM_W + SMEM_X + NBAR * 8;
  static_assert(MINB * (SMEM + 1024) <= 167936, "sm_80 shared memory for two CTAs per SM");
};

// x staging row (256 B, 16 granules): granule G at G ^ (r & 1) ^ ((G >> 3) << 1).  A quarter-warp of the LN / residual read is rows
// {r, r + 1} x quads q = 0..3 at granule 4 q + i: bits (r0, G3, G2) = (r0, q1, q0) pick 8 distinct 16 B bank groups.
DEVI uint32_t swz_x(uint32_t r, uint32_t G) { return r * 256 + ((G ^ (r & 1u) ^ ((G >> 3) << 1)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) tr_fwd_kernel(const TrParams p) {
  constexpr int D = G::D, NT = G::NTHR, BM = G::BM, NCHUNK = G::NCHUNK;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sW = smem;
  uint8_t* sX = sW + G::SMEM_W;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sX + G::SMEM_X);
  const uint32_t sW_u = smem_u32(sW);
  const uint32_t barW = smem_u32(bars), barX = barW + 8 * G::NST;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g8 = lane >> 2, q = lane & 3;
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = n_iter * NCHUNK;

  if (tid == 0) {
    for (int s = 0; s < G::NST; ++s) mbar_init(barW + 8 * s, NT);
    for (int w = 0; w < G::NWARP; ++w) mbar_init(barX + 8 * w, 32);
  }
  __syncthreads();

  // ---- weight chunk c -> slot s: 1536 granules, 12 per thread, a straight copy of the host-packed chunk
  auto issue_w = [&](int c, int s) {
    const __nv_bfloat16* src = p.w + (size_t)c * (G::SLOT / 2) + tid * 8;
    const uint32_t dst = sW_u + s * G::SLOT + tid * 16;
#pragma unroll
    for (int i = 0; i < G::SLOT / 16 / NT; ++i) cp_async16(dst + i * NT * 16, src + i * NT * 8);
    cp_async_mbar_arrive(barW + 8 * s);
  };
  // ---- this warp's 32 rows of tile `tile` -> its x staging (zero-filled past T)
  const uint32_t xw = smem_u32(sX) + warp * G::XW;
  auto load_x = [&](int tile) {
    const int r0 = tile * BM + 32 * warp;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = lane + 32 * i, r = c >> 4, gr = c & 15;
      const bool ok = r0 + r < p.T;
      cp_async16(xw + swz_x(r, gr), p.x + (size_t)(ok ? r0 + r : 0) * D + gr * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barX + 8 * warp);
  };
  if (n_iter > 0) { load_x(blockIdx.x); issue_w(0, 0); }

  // ldmatrix lane addresses: matrix mi = lane / 8 -> (row half mi >> 1, granule half mi & 1)
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  const uint32_t w1_off = lgr * 1024 + lrow * 16;                         // + p * 512 (step rows) + 256 (b rows) + s * 2048 (k-step)
  const uint32_t w2_off = G::SLOT_W1 + lgr * 2048 + lrow * 16;            // + p * 4096 (step granules) + j * 256 (out rows)

  int u = 0;
  for (int it = 0; it < n_iter; ++it) {
    const int tile = (int)blockIdx.x + it * (int)gridDim.x;
    const int r0 = tile * BM + 32 * warp;

    // ---- LayerNorm of rows (16 mt + 8 hr + g8) straight into the A fragments: word w = 2 s + e2 of the thread's 64 B -> fa[mt][s][hr + 2 e2]
    uint32_t fa[2][8][4];
    mbar_wait(barX + 8 * warp, it & 1);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const int r = 16 * mt + 8 * hr + g8;
        uint32_t wv[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = lds128(xw + swz_x(r, 4 * q + i));
          wv[4 * i] = v.x; wv[4 * i + 1] = v.y; wv[4 * i + 2] = v.z; wv[4 * i + 3] = v.w;
        }
        float xv[32];
#pragma unroll
        for (int e = 0; e < 16; ++e) { xv[2 * e] = bf16lo(wv[e]); xv[2 * e + 1] = bf16hi(wv[e]); }
        float sm = 0.f;
#pragma unroll
        for (int e = 0; e < 32; ++e) sm += xv[e];
        const float mean = quad_sum(sm) * (1.f / D);
        float sq = 0.f;
#pragma unroll
        for (int e = 0; e < 32; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
        const float rstd = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float4 gv = __ldg(p.gb + i * 4 + q), bv = __ldg(p.gb + 32 + i * 4 + q);
          const float* gf = reinterpret_cast<const float*>(&gv);
          const float* bf = reinterpret_cast<const float*>(&bv);
          fa[mt][i][hr] = pack_bf16(fmaf(xv[4 * i] * rstd, gf[0], bf[0]), fmaf(xv[4 * i + 1] * rstd, gf[1], bf[1]));
          fa[mt][i][hr + 2] = pack_bf16(fmaf(xv[4 * i + 2] * rstd, gf[2], bf[2]), fmaf(xv[4 * i + 3] * rstd, gf[3], bf[3]));
        }
      }
    __syncwarp();
    if (it + 1 < n_iter) load_x(tile + (int)gridDim.x);       // the staging is free: the next tile's rows land under this tile

    float acc2[2][16][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc2[mt][j][e] = 0.f;

#pragma unroll 1
    for (int c = 0; c < NCHUNK; ++c, ++u) {
      // every warp retired chunk u - 1: its slot takes chunk u + 1; then wait for chunk u
      bar_sync(1, NT);
      if (u + 1 < total) issue_w((u + 1) % NCHUNK, (u + 1) & 1);
      mbar_wait(barW + 8 * (u & 1), (u >> 1) & 1);
      const uint32_t wb = sW_u + (u & 1) * G::SLOT;
#pragma unroll
      for (int ps = 0; ps < 2; ++ps) {
        float acc1[2][4][4];                                   // [mt][a lo, a hi, b lo, b hi][4]
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc1[mt][n][e] = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          uint32_t ba[4], bb[4];
          ldsm_x4(ba, wb + w1_off + ps * 512 + s * 2048);
          ldsm_x4(bb, wb + w1_off + ps * 512 + 256 + s * 2048);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc1[mt][0], fa[mt][s], ba[0], ba[1]);
            mma16816(acc1[mt][1], fa[mt][s], ba[2], ba[3]);
            mma16816(acc1[mt][2], fa[mt][s], bb[0], bb[1]);
            mma16816(acc1[mt][3], fa[mt][s], bb[2], bb[3]);
          }
        }
        uint32_t ha[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          float h[2][4];
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float av = acc1[mt][n][e], pr = av * acc1[mt][2 + n][e];
              h[n][e] = fmaf(tanh_approx(av), pr, pr);
            }
          ha[mt][0] = pack_bf16(h[0][0], h[0][1]);
          ha[mt][1] = pack_bf16(h[0][2], h[0][3]);
          ha[mt][2] = pack_bf16(h[1][0], h[1][1]);
          ha[mt][3] = pack_bf16(h[1][2], h[1][3]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t bs[4];
          ldsm_x4(bs, wb + w2_off + ps * 4096 + j * 256);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc2[mt][2 * j], ha[mt], bs[0], bs[1]);
            mma16816(acc2[mt][2 * j + 1], ha[mt], bs[2], bs[3]);
          }
        }
      }
    }

    // ---- residual (x re-read from L2) + store: word J of the thread's 64 B = accumulator tile J
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const int r = r0 + 16 * mt + 8 * hr + g8;
        if (r < p.T) {
          const uint4* src = reinterpret_cast<const uint4*>(p.x + (size_t)r * D + 32 * q);
          __nv_bfloat16* dst = p.out + (size_t)r * D + 32 * q;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 xv = __ldg(src + i);
            const uint32_t xw4[4] = {xv.x, xv.y, xv.z, xv.w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int J = 4 * i + k;
              o[k] = pack_bf16(acc2[mt][J][2 * hr] + bf16lo(xw4[k]), acc2[mt][J][2 * hr + 1] + bf16hi(xw4[k]));
            }
            stg128(dst + 8 * i, make_uint4(o[0], o[1], o[2], o[3]));
          }
        }
      }
  }
  cp_async_wait<0>();
}

}  // namespace a100
