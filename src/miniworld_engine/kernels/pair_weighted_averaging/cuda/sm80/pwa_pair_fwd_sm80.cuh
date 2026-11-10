// pwa_pair_fwd_sm80.cuh -- MSA pair-weighted averaging forward, the pair side: LayerNorm of the pair, the 8-head bias projection, the key mask and the
// softmax over the keys, one CTA per query row i.
//
//   zn[i, j, :]  = bf16( LN(z[i, j, :]) )                   fp32 statistics (two-pass), affine, one rounding
//   logit[h]     = bf16( zn . Wb[h, :] )                    mma.sync, fp32 accumulation, the bf16 output of the module's `to_bias`
//   w[h, i, :]   = softmax_j( key mask ? logit : -1e30 )    fp32, written bf16 [8, L, L]
//
// A warp takes 16 keys at a time: the [16][DZ] tile is staged by cp.async (swizzled rows), read once into mma A fragments (ldmatrix) which stay in
// registers through the statistics (quad shuffles: a row is four lanes), the normalisation (straight into the A fragments of the projection) and the
// product with Wb^T (B fragments held in registers; N = 8 heads is one n8 tile).  The next tile's copy is issued as soon as the fragments are loaded.
// The logits of the whole row are staged in shared memory [8][L], then warp h takes the softmax of head h.
#pragma once
#include "sm80_common.cuh"

namespace pwa80 {

struct PairFwdParams {
  const __nv_bfloat16* z;      // [L][L][DZ]
  const uint8_t* mask;         // [L] key mask (nonzero = valid) or nullptr
  const float* lnw;            // [DZ]
  const float* lnb;            // [DZ]
  const __nv_bfloat16* wb;     // [8][DZ]
  __nv_bfloat16* w;            // [8][L][L]
  int L;
  float eps;
};

template <int DZ> struct PairFwdCfg {
  static constexpr int NWARP = 8, NTHR = 256, GJ = 16, NK = DZ / 16, NCHK = DZ / 8;
  static constexpr int TILE = GJ * DZ * 2;                      // one warp's [16][DZ] bf16 tile
  static constexpr int FIXED = NWARP * TILE + 2 * DZ * 4 + DZ * 16;   // tiles + gamma / beta + the B fragments of Wb^T ([NK][32 lanes] x 8 B)
  static_assert(DZ % 128 == 0 || DZ == 192, "d_pair");
};

template <int NCH> DEVI uint32_t tile_off(int r, int ch) { return swzn<NCH>(r, ch); }

template <int DZ> DEVI void pf_load_tile(uint32_t dst, const __nv_bfloat16* src_row0, int lane) {   // 16 rows x DZ*2 bytes, contiguous in global
  using C = PairFwdCfg<DZ>;
#pragma unroll
  for (int q = 0; q < C::GJ * C::NCHK / 32; ++q) {
    const int v = q * 32 + lane, r = v / C::NCHK, ch = v % C::NCHK;
    cp_async16(dst + tile_off<C::NCHK>(r, ch), src_row0 + (long)r * DZ + ch * 8);
  }
}

template <int DZ>
__global__ void __launch_bounds__(256, 1) pwa_pair_fwd_kernel(const PairFwdParams p) {
  using C = PairFwdCfg<DZ>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  float* sg = reinterpret_cast<float*>(smem + C::NWARP * C::TILE);           // gamma [DZ], beta [DZ]
  uint2* sbw = reinterpret_cast<uint2*>(sg + 2 * DZ);                       // B fragments of Wb^T: [ks][lane] = (n = head lane / 4, k = 16 ks + 2 (lane % 4) (+8)) pairs
  float* sb = reinterpret_cast<float*>(sbw + C::NK * 32);                   // logits [8][NP]
  const int NP = p.L + 4;
  const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  for (int k = tid; k < DZ; k += C::NTHR) { sg[k] = p.lnw[k]; sg[DZ + k] = p.lnb[k]; }
  for (int idx = tid; idx < C::NK * 32; idx += C::NTHR) {                    // (kept in shared memory: 2 NK registers a thread would not fit at d_pair 384)
    const int ks = idx >> 5, l = idx & 31, gg = l >> 2, qq = l & 3;
    sbw[idx] = make_uint2(*reinterpret_cast<const uint32_t*>(p.wb + gg * DZ + 16 * ks + 2 * qq), *reinterpret_cast<const uint32_t*>(p.wb + gg * DZ + 16 * ks + 2 * qq + 8));
  }
  __syncthreads();
  const uint32_t tbase = sbase + warp * C::TILE;
  const int ngroups = p.L / C::GJ;
  int gi = warp;
  if (gi < ngroups) pf_load_tile<DZ>(tbase, p.z + ((long)i * p.L + gi * C::GJ) * DZ, lane);
  cp_async_commit();
  for (; gi < ngroups; gi += C::NWARP) {
    cp_async_wait<0>();
    __syncwarp();
    uint32_t a[C::NK][4];
#pragma unroll
    for (int ks = 0; ks < C::NK; ++ks) ldsm_x4(a[ks], tbase + tile_off<C::NCHK>(lane & 15, 2 * ks + (lane >> 4)));
    __syncwarp();
    if (gi + C::NWARP < ngroups) pf_load_tile<DZ>(tbase, p.z + ((long)i * p.L + (gi + C::NWARP) * C::GJ) * DZ, lane);
    cp_async_commit();
    // row statistics: this lane holds DZ/4 values of row g (a0, a2) and of row g + 8 (a1, a3); the quad completes a row
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < C::NK; ++ks) {
      s0 += bf16lo(a[ks][0]) + bf16hi(a[ks][0]) + bf16lo(a[ks][2]) + bf16hi(a[ks][2]);
      s1 += bf16lo(a[ks][1]) + bf16hi(a[ks][1]) + bf16lo(a[ks][3]) + bf16hi(a[ks][3]);
    }
    const float m0 = quad_sum(s0) * (1.f / DZ), m1 = quad_sum(s1) * (1.f / DZ);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < C::NK; ++ks) {
      float d;
      d = bf16lo(a[ks][0]) - m0; v0 = fmaf(d, d, v0); d = bf16hi(a[ks][0]) - m0; v0 = fmaf(d, d, v0);
      d = bf16lo(a[ks][2]) - m0; v0 = fmaf(d, d, v0); d = bf16hi(a[ks][2]) - m0; v0 = fmaf(d, d, v0);
      d = bf16lo(a[ks][1]) - m1; v1 = fmaf(d, d, v1); d = bf16hi(a[ks][1]) - m1; v1 = fmaf(d, d, v1);
      d = bf16lo(a[ks][3]) - m1; v1 = fmaf(d, d, v1); d = bf16hi(a[ks][3]) - m1; v1 = fmaf(d, d, v1);
    }
    const float rs0 = rsqrtf(quad_sum(v0) * (1.f / DZ) + p.eps), rs1 = rsqrtf(quad_sum(v1) * (1.f / DZ) + p.eps);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < C::NK; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sg + col), g1 = *reinterpret_cast<const float2*>(sg + col + 8);
      const float2 b0 = *reinterpret_cast<const float2*>(sg + DZ + col), b1 = *reinterpret_cast<const float2*>(sg + DZ + col + 8);
      uint32_t zn[4];
      zn[0] = pack_bf16(fmaf((bf16lo(a[ks][0]) - m0) * rs0, g0.x, b0.x), fmaf((bf16hi(a[ks][0]) - m0) * rs0, g0.y, b0.y));
      zn[1] = pack_bf16(fmaf((bf16lo(a[ks][1]) - m1) * rs1, g0.x, b0.x), fmaf((bf16hi(a[ks][1]) - m1) * rs1, g0.y, b0.y));
      zn[2] = pack_bf16(fmaf((bf16lo(a[ks][2]) - m0) * rs0, g1.x, b1.x), fmaf((bf16hi(a[ks][2]) - m0) * rs0, g1.y, b1.y));
      zn[3] = pack_bf16(fmaf((bf16lo(a[ks][3]) - m1) * rs1, g1.x, b1.x), fmaf((bf16hi(a[ks][3]) - m1) * rs1, g1.y, b1.y));
      const uint2 bwk = sbw[ks * 32 + lane];
      mma16816(c, zn, bwk.x, bwk.y);
    }
    // c0, c1: (key g, heads 2q, 2q + 1); c2, c3: (key g + 8)
    const int j0 = gi * C::GJ;
    const bool k0 = p.mask == nullptr || p.mask[j0 + g] != 0, k1 = p.mask == nullptr || p.mask[j0 + g + 8] != 0;
    sb[(2 * q) * NP + j0 + g] = k0 ? round_bf16f(c[0]) : -1e30f;
    sb[(2 * q + 1) * NP + j0 + g] = k0 ? round_bf16f(c[1]) : -1e30f;
    sb[(2 * q) * NP + j0 + g + 8] = k1 ? round_bf16f(c[2]) : -1e30f;
    sb[(2 * q + 1) * NP + j0 + g + 8] = k1 ? round_bf16f(c[3]) : -1e30f;
  }
  cp_async_wait<0>();
  __syncthreads();
  {                                                                         // softmax over the keys of head `warp`
    const int h = warp;
    float mx = -3.0e38f;
    for (int j = lane; j < p.L; j += 32) mx = fmaxf(mx, sb[h * NP + j]);
    mx = warp_max(mx);
    float den = 0.f;
    for (int j = lane; j < p.L; j += 32) den += __expf(sb[h * NP + j] - mx);
    den = warp_sum(den);
    const float inv = 1.f / den;
    __nv_bfloat16* wr = p.w + ((long)h * p.L + i) * p.L;
    for (int j = lane; j < p.L; j += 32) wr[j] = __float2bfloat16_rn(__expf(sb[h * NP + j] - mx) * inv);
  }
}

}  // namespace pwa80
