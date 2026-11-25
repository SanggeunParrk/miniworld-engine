// wgrad.cuh -- the weight gradients of the AF3 local atom DiT block on A100 / sm_80 as one grouped split-M product: for every problem D = A^T B (A [M][Ka] and B [M][Kb] bf16, row strides lda / ldb;
// D [Ka][Kb] fp32, the weight's [out][in] layout) -- the products every activation-by-gradient pair of the backward asks for -- in a single launch, each operand read once from DRAM.
//   wgrad_kernel   a CTA computes a 128 x 128 tile of one problem over a range of rows (the split): 8 warps of 64 x 32, A and B tiles of 64 rows through a three-stage cp.async ring, both operands read
//                  as transposed fragments (ldmatrix.trans); the partial sums go to ``part`` [split][problem slab], one writer per element
//   wgrad_reduce   adds the splits in a fixed order (deterministic) into the weights' gradients, rounded to their dtype; plain copies (bias / LayerNorm-weight gradients) go through the same launch
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

constexpr int WG_MAXP = 12, WG_MAXS = 32, WG_RB = 64, WG_STAGE = 2 * WG_RB * 256, WG_SMEM = 3 * WG_STAGE;

struct WgProb {
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  long long lda, ldb, pofs;                          // row strides (elements) and the problem's offset in a split's slab
  int ka, kb;
};
struct WgParams {
  WgProb prob[WG_MAXP];
  int tile_prefix[WG_MAXP + 1];                      // first tile of each problem
  int nprob, M, splits, rps;                         // rps: rows per split
  float* part;
  long long slab;                                    // elements of one split's partial buffer
};

DEVI void ldsm_x4_trans(uint32_t (&r)[4], uint32_t addr) { ldsm_x4_t(r, addr); }

__global__ void __launch_bounds__(256, 1) wgrad_kernel(const __grid_constant__ WgParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int tile = (int)blockIdx.x / p.splits, split = (int)blockIdx.x % p.splits;
  int pi = 0;
  while (pi + 1 < p.nprob && tile >= p.tile_prefix[pi + 1]) ++pi;
  const WgProb pr = p.prob[pi];
  const int local = tile - p.tile_prefix[pi], ntb = pr.kb / 128, ta = local / ntb, tb = local % ntb;
  const int m0 = split * p.rps, m1 = min(p.M, m0 + p.rps), nch = (m1 - m0 + WG_RB - 1) / WG_RB;
  const __nv_bfloat16* ga = pr.a + 128 * ta;
  const __nv_bfloat16* gb = pr.b + 128 * tb;

  auto load = [&](int stage, int ch) {                                         // 64 rows of A and of B (128 columns each) -> stage
    const uint32_t sa = sb + stage * WG_STAGE, sbb = sa + WG_RB * 256;
#pragma unroll
    for (int i = tid; i < WG_RB * 16; i += 256) {
      const int r = i >> 4, c = i & 15, row = m0 + ch * WG_RB + r;
      const bool ok = ch < nch && row < m1;
      const size_t rr = ok ? (size_t)row : 0;
      cp_async16(sa + sw256(r, c), ga + rr * pr.lda + c * 8, ok ? 16u : 0u);
      cp_async16(sbb + sw256(r, c), gb + rr * pr.ldb + c * 8, ok ? 16u : 0u);
    }
  };
  load(0, 0); cp_async_commit();
  load(1, 1); cp_async_commit();

  const int wm = warp >> 2, wn = warp & 3;
  float acc[4][4][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }

  for (int ch = 0; ch < nch; ++ch) {
    cp_async_wait<1>();
    __syncthreads();
    load((ch + 2) % 3, ch + 2); cp_async_commit();
    const uint32_t sa = sb + (ch % 3) * WG_STAGE, sbb = sa + WG_RB * 256;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {                                           // 16 rows a step
      uint32_t af[4][4], bfg[2][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {                                         // A^T fragments: ka 64 wm + 16 mt .. , rows 16 ks ..
        const int row = 16 * ks + (lane & 7) + 8 * (lane >> 4), colch = (64 * wm + 16 * mt) / 8 + ((lane >> 3) & 1);
        ldsm_x4_trans(af[mt], sa + sw256(row, colch));
      }
#pragma unroll
      for (int np = 0; np < 2; ++np) {                                         // B fragments of the n tiles 2 np, 2 np + 1: kb 32 wn + 16 np ..
        const int row = 16 * ks + (lane & 7) + 8 * ((lane >> 3) & 1), colch = (32 * wn + 16 * np) / 8 + (lane >> 4);
        ldsm_x4_trans(bfg[np], sbb + sw256(row, colch));
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma16816(acc[mt][nt], af[mt], bfg[nt >> 1][2 * (nt & 1)], bfg[nt >> 1][2 * (nt & 1) + 1]);
    }
  }

  float* dst = p.part + (size_t)split * p.slab + pr.pofs;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r0 = 128 * ta + 64 * wm + 16 * mt + g8, c0 = 128 * tb + 32 * wn + 8 * nt + 2 * q4;
      *reinterpret_cast<float2*>(dst + (size_t)r0 * pr.kb + c0) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<float2*>(dst + (size_t)(r0 + 8) * pr.kb + c0) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
}

struct WgSeg {
  const float* src;
  void* dst;
  long long n, split_stride;
  int nsplit, bf16;
};
struct WgSegs {
  WgSeg seg[WG_MAXS];
};

__global__ void __launch_bounds__(256) wgrad_reduce_kernel(const __grid_constant__ WgSegs s) {
  const WgSeg g = s.seg[blockIdx.y];
  const long long i = ((long long)blockIdx.x * 256 + threadIdx.x) * 2;
  if (i >= g.n) return;
  float2 v = *reinterpret_cast<const float2*>(g.src + i);
  for (int k = 1; k < g.nsplit; ++k) {
    const float2 w = *reinterpret_cast<const float2*>(g.src + (long long)k * g.split_stride + i);
    v.x += w.x; v.y += w.y;
  }
  if (g.bf16) *reinterpret_cast<uint32_t*>(reinterpret_cast<__nv_bfloat16*>(g.dst) + i) = pack_bf16(v.x, v.y);
  else *reinterpret_cast<float2*>(reinterpret_cast<float*>(g.dst) + i) = v;
}

}  // namespace lr80
