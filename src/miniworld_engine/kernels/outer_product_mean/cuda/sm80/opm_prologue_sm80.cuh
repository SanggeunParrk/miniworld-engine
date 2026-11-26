// opm_prologue_sm80.cuh -- OuterProductMean forward, front: LayerNorm of the MSA rows and the left / right projections, written straight into the s-major
// operands of the outer-product GEMM.
//
//   y[s, i, :]  = bf16( LN(m[s, i, :]) )                          fp32 statistics, affine, one rounding (the module's bf16 LayerNorm output)
//   a[s, i, c]  = bf16( y . Wl[c, :] ) * mask[s, i]               (and b with Wr)  -- the module's `left` / `right`
//   A[s, i * 32 + c] = a[s, i, c],  B[s, j * 32 + e] = b[s, j, e]         both [S, L * 32] bf16 (a token's 32 channels are 64 contiguous bytes)
//
// so that O = A^T B (one cuBLAS call, K = S: the NT-class kernels, the fastest of the four operand layouts on this card) is the grouped outer product
// O[(i, c), (j, e)] = sum_s a[s, i, c] b[s, j, e].  A CTA loops over tiles of one MSA row s x 128 consecutive tokens (8 warps x 16 tokens: the tile is one contiguous
// block of m, of A, of B and of the statistics); the next tile's rows arrive by cp.async into a second buffer while the current one is processed.  The LayerNorm runs
// on the A fragments of the projection (ldmatrix, quad shuffles) and leaves y in registers; the two projections are one mma.sync product with the 64 stacked weight
// rows [Wl; Wr] as the B operand (n = channel), so a warp's accumulators are [16 tokens x 64 channels]: masked, rounded to bf16, staged as 64-byte token rows and stored
// as contiguous 16-byte vectors.  Training saves the LayerNorm statistics (mean, rstd).
#pragma once
#include "sm80_common.cuh"

namespace opm80 {

struct PrologueParams {
  const __nv_bfloat16* m;      // [S][L][CM]
  const uint8_t* mask;         // [S][L] (nonzero = valid) or nullptr
  const float* lnw;            // [CM]
  const float* lnb;            // [CM]
  const __nv_bfloat16* wl;     // [32][CM]
  const __nv_bfloat16* wr;     // [32][CM]
  __nv_bfloat16* a;            // [S][L * 32]
  __nv_bfloat16* b;            // [S][L * 32]
  float2* stats;               // [S][L] (mean, rstd) or nullptr
  int S, L, nti, ntile;        // nti = tiles per MSA row, ntile = S nti
  float eps;
};

template <int CM, int NB> struct PrologueCfg {                              // NB: m tile buffers (the prefetch runs NB - 1 tiles ahead)
  static constexpr int NTHR = 256, NTOK = 128, NCH = CM / 8;                 // 16-byte chunks per m row
  static constexpr int MB = NTOK * CM * 2, WB = 64 * CM * 2, OB = 2 * NTOK * 64;   // one m tile, stacked weights, the staged a | b tiles ([128][32] bf16 each)
  static constexpr int SMEM = NB * MB + WB + OB + 2 * CM * 4;
  static_assert(CM == 64 || CM == 128, "d_msa");
  static_assert(NB >= 2 && NB <= 4, "buffers");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

template <int CM, int NB>
DEVI void pl_load(const PrologueParams& p, uint32_t dst, int tl, int tid) {
  using C = PrologueCfg<CM, NB>;
  const int s = tl / p.nti, i0 = (tl % p.nti) * C::NTOK;
#pragma unroll
  for (int it = 0; it < C::NTOK * C::NCH / C::NTHR; ++it) {
    const int v = it * C::NTHR + tid, ch = v % C::NCH, t = v / C::NCH;
    const bool ok = i0 + t < p.L;
    cp_async16(dst + swzn<C::NCH>(t, ch), p.m + (ok ? ((long)s * p.L + i0 + t) * CM + ch * 8 : 0), ok ? 16u : 0u);
  }
}

template <int CM, int NB, bool SAVE_STATS, int NMINB>
__global__ void __launch_bounds__(256, NMINB) opm_prologue_kernel(const PrologueParams p) {
  using C = PrologueCfg<CM, NB>;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ uint8_t smask[C::NTOK];
  const uint32_t sm0 = smem_u32(smem), sw = sm0 + NB * C::MB, so_a = sw + C::WB, so_b = so_a + C::NTOK * 64;
  float* sgb = reinterpret_cast<float*>(smem + NB * C::MB + C::WB + C::OB);                  // gamma [CM], beta [CM]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;

  for (int u = tid; u < 64 * C::NCH; u += C::NTHR) {                                          // the stacked weights [Wl; Wr] -> shared memory (swizzled rows)
    const int row = u / C::NCH, ch = u % C::NCH;
    cp_async16(sw + swzn<C::NCH>(row, ch), (row < 32 ? p.wl + row * CM : p.wr + (row - 32) * CM) + ch * 8);
  }
  for (int k = tid; k < CM; k += C::NTHR) { sgb[k] = p.lnw[k]; sgb[CM + k] = p.lnb[k]; }
#pragma unroll
  for (int j = 0; j < NB - 1; ++j) {                                                          // the first NB - 1 tiles (group 0 also carries the weights)
    if (blockIdx.x + j * gridDim.x < p.ntile) pl_load<CM, NB>(p, sm0 + j * C::MB, blockIdx.x + j * gridDim.x, tid);
    cp_async_commit();
  }

  // the tile's mask bytes (one per token, 128 B) are fetched one tile ahead into a register of the first NTOK threads, so that no global load sits in front of the barrier
  auto fetch = [&](int tl2) -> uint32_t {                                                      // 0 for padded tokens, 1 without a mask
    if (tid >= C::NTOK || tl2 >= p.ntile) return 0u;
    const int i2 = (tl2 % p.nti) * C::NTOK + tid;
    if (i2 >= p.L) return 0u;
    return p.mask == nullptr ? 1u : (uint32_t)p.mask[(long)(tl2 / p.nti) * p.L + i2];
  };
  uint32_t mk = fetch(blockIdx.x);

#pragma unroll 1
  for (int tl = blockIdx.x, buf = 0; tl < p.ntile; tl += gridDim.x, buf = buf + 1 == NB ? 0 : buf + 1) {
    const int s = tl / p.nti, i0 = (tl % p.nti) * C::NTOK;
    const int nx = tl + (NB - 1) * gridDim.x;                                                 // prefetch into the buffer that was consumed last
    if (nx < p.ntile) pl_load<CM, NB>(p, sm0 + (buf == 0 ? NB - 1 : buf - 1) * C::MB, nx, tid);
    cp_async_commit();
    if (tid < C::NTOK) smask[tid] = (uint8_t)mk;
    mk = fetch(tl + gridDim.x);                                                                // consumed at the next iteration
    cp_async_wait<NB - 1>();
    __syncthreads();
    const uint32_t sm = sm0 + buf * C::MB;
    const int r0 = warp * 16 + gq, r1 = r0 + 8;

    // ---- LayerNorm of the warp's 16 tokens on the A fragments -> ya
    uint32_t xa[CM / 16][4];
#pragma unroll
    for (int ks = 0; ks < CM / 16; ++ks) ldsm_x4(xa[ks], sm + swzn<C::NCH>(warp * 16 + (lane & 15), 2 * ks + (lane >> 4)));
    float s0v = 0.f, s1v = 0.f;
#pragma unroll
    for (int ks = 0; ks < CM / 16; ++ks) {
      s0v += bf16lo(xa[ks][0]) + bf16hi(xa[ks][0]) + bf16lo(xa[ks][2]) + bf16hi(xa[ks][2]);
      s1v += bf16lo(xa[ks][1]) + bf16hi(xa[ks][1]) + bf16lo(xa[ks][3]) + bf16hi(xa[ks][3]);
    }
    const float mean0 = quad_sum(s0v) * (1.f / CM), mean1 = quad_sum(s1v) * (1.f / CM);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < CM / 16; ++ks) {
      float d;
      d = bf16lo(xa[ks][0]) - mean0; v0 = fmaf(d, d, v0); d = bf16hi(xa[ks][0]) - mean0; v0 = fmaf(d, d, v0);
      d = bf16lo(xa[ks][2]) - mean0; v0 = fmaf(d, d, v0); d = bf16hi(xa[ks][2]) - mean0; v0 = fmaf(d, d, v0);
      d = bf16lo(xa[ks][1]) - mean1; v1 = fmaf(d, d, v1); d = bf16hi(xa[ks][1]) - mean1; v1 = fmaf(d, d, v1);
      d = bf16lo(xa[ks][3]) - mean1; v1 = fmaf(d, d, v1); d = bf16hi(xa[ks][3]) - mean1; v1 = fmaf(d, d, v1);
    }
    const float rs0 = rsqrtf(quad_sum(v0) * (1.f / CM) + p.eps), rs1 = rsqrtf(quad_sum(v1) * (1.f / CM) + p.eps);
    if (SAVE_STATS && q == 0) {
      if (i0 + r0 < p.L) p.stats[(long)s * p.L + i0 + r0] = make_float2(mean0, rs0);
      if (i0 + r1 < p.L) p.stats[(long)s * p.L + i0 + r1] = make_float2(mean1, rs1);
    }
    uint32_t ya[CM / 16][4];
#pragma unroll
    for (int ks = 0; ks < CM / 16; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sgb + col), g1 = *reinterpret_cast<const float2*>(sgb + col + 8);
      const float2 b0 = *reinterpret_cast<const float2*>(sgb + CM + col), b1 = *reinterpret_cast<const float2*>(sgb + CM + col + 8);
      ya[ks][0] = pack_bf16(fmaf((bf16lo(xa[ks][0]) - mean0) * rs0, g0.x, b0.x), fmaf((bf16hi(xa[ks][0]) - mean0) * rs0, g0.y, b0.y));
      ya[ks][1] = pack_bf16(fmaf((bf16lo(xa[ks][1]) - mean1) * rs1, g0.x, b0.x), fmaf((bf16hi(xa[ks][1]) - mean1) * rs1, g0.y, b0.y));
      ya[ks][2] = pack_bf16(fmaf((bf16lo(xa[ks][2]) - mean0) * rs0, g1.x, b1.x), fmaf((bf16hi(xa[ks][2]) - mean0) * rs0, g1.y, b1.y));
      ya[ks][3] = pack_bf16(fmaf((bf16lo(xa[ks][3]) - mean1) * rs1, g1.x, b1.x), fmaf((bf16hi(xa[ks][3]) - mean1) * rs1, g1.y, b1.y));
    }

    // ---- projections: acc[nt] = y (16 tokens) x the stacked weight rows 8 nt ..  (n8 tiles 0-3: left, 4-7: right)
    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < CM / 16; ++ks) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4(bf, sw + swzn<C::NCH>(16 * np + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
        mma16816(acc[2 * np], ya[ks], bf[0], bf[1]);
        mma16816(acc[2 * np + 1], ya[ks], bf[2], bf[3]);
      }
    }

    // ---- mask, bf16, into the staged token rows (64 bytes: 4 chunks of 8 channels)
    const bool k0 = smask[r0] != 0, k1 = smask[r1] != 0;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      sts32(so_a + swz64(r0, nt) + 4 * q, pack_bf16(k0 ? acc[nt][0] : 0.f, k0 ? acc[nt][1] : 0.f));
      sts32(so_a + swz64(r1, nt) + 4 * q, pack_bf16(k1 ? acc[nt][2] : 0.f, k1 ? acc[nt][3] : 0.f));
      sts32(so_b + swz64(r0, nt) + 4 * q, pack_bf16(k0 ? acc[4 + nt][0] : 0.f, k0 ? acc[4 + nt][1] : 0.f));
      sts32(so_b + swz64(r1, nt) + 4 * q, pack_bf16(k1 ? acc[4 + nt][2] : 0.f, k1 ? acc[4 + nt][3] : 0.f));
    }
    __syncthreads();

    // ---- stores: 128 tokens x 64 B of A and of B, contiguous
#pragma unroll
    for (int it = 0; it < C::NTOK * 4 / C::NTHR; ++it) {
      const int u = it * C::NTHR + tid, t = u >> 2, ch = u & 3;
      if (i0 + t < p.L) {
        const long off = ((long)s * p.L + i0 + t) * 32 + ch * 8;
        stg128(p.a + off, lds128(so_a + swz64(t, ch)));
        stg128(p.b + off, lds128(so_b + swz64(t, ch)));
      }
    }
    __syncthreads();                                   // the staged tiles are rewritten by the next tile
  }
  cp_async_wait<0>();
}

}  // namespace opm80
