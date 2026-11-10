// qkvg_fwd_sm80.cuh -- the first stage of the SWA atom DiT block, A100 / sm_80: the adaLN-modulated RMSNorm of the residual stream, the four projections q | k | v | g
// (128 -> 128 each), the per-head RMS norm and RoPE of q and k, written head-major.  The twin of the Triton ``_swa_qkvg_fwd_kernel`` with its rounding points:
//
//   x    = rn(rn(q rstd) (1 + scale_a) + shift_a)          rstd = 1 / sqrt(mean(q^2) + eps), fp32;  shift_a | scale_a = mod[mrow, 0:128] | mod[mrow, 128:256]
//   p_*  = rn(x W_*^T)                                      fp32 accumulate (q, k: rounded to bf16 before the head norm)
//   Q    = rn(rope(rn(p_q rstd_h)))   rstd_h = 1 / sqrt(mean_head(p_q^2) + qk_eps)         (same for K);  V = p_v;  G = p_g
//
// Q / K / V are written head-major [N][H][S][32], G row-major [M][128]; with ``save`` also x and the pre-norm p_q, p_k (row-major bf16).  mrow = (n % B) S + s is the
// row of the hoisted modulation (and of the RoPE angles) of the row r = n S + s.
//
// The structure is the TriangleAttention front's (f1_sm80.cuh): one persistent CTA per SM of NW warps; the 4 x [128][128] weights stay in shared memory in the kernel's
// row order (``f1_channel``: a thread's accumulator pair over the 4 n tiles of a group is 8 consecutive output channels, so a head's 32 channels are a quad's 8 + 8 + 8 + 8
// and the head norm is a quad reduction; the RoPE partner d + 16 is the lane two over); every warp runs its own loop over tiles of 32 rows, with the rows going from
// global memory into the A fragments (RMS norm and modulation on the registers).
#pragma once
#include "sm80_common.cuh"

namespace sw80 {

constexpr int QF_ROWS = 512;                         // packed weight rows: q | k | v | g, 128 outputs each

struct QkvgFwdParams {
  const __nv_bfloat16* x;                            // [M][128]
  const float* mod;                                  // [B S][768]
  const float* cos;                                  // [B S][16]
  const float* sin;
  const __nv_bfloat16* wqkv;                         // [384][128]
  const __nv_bfloat16* wg;                           // [128][128]
  __nv_bfloat16 *qh, *kh, *vh;                       // [N][4][S][32]
  __nv_bfloat16* g;                                  // [M][128]
  __nv_bfloat16 *xs, *pqs, *pks;                     // [M][128] or nullptr (save)
  int M, S, B;
  float eps, qk_eps;
};

// packed row s -> the output channel it holds: in a block of 64 rows the 8 n tiles are (og, j) = (nt >> 2, nt & 3) and column n of a tile holds channel 32 og + 8 (n >> 1) + 2 j + (n & 1)
DEVI int qf_channel(int s) { return (s & ~63) + 32 * ((s >> 5) & 1) + 8 * ((s >> 1) & 3) + 2 * ((s >> 3) & 3) + (s & 1); }
// byte offset of 16-byte chunk `chunk` of packed row `row` (rows of 256 B; the chunk index is XOR-ed with 4 on odd rows: the 2 rows x 4 chunks a quarter warp reads fall in 8 bank groups)
DEVI uint32_t qf_woff(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ ((row & 1u) << 2)) << 4); }

// NW warps per CTA, each looping over tiles of 16 MT rows
template <int NW_, int MT_>
struct QkvgCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MT = MT_, ROWS = 16 * MT_;
  static constexpr int SMEM = QF_ROWS * 256;
};

DEVI float rn(float x) { return round_bf16f(x); }

// 8 consecutive channels d = 8 q4 .. 8 q4 + 7 of a head: RoPE with the partner d +- 16 (the lane q4 ^ 2); cs / sn = cos / sin of f = d mod 16
DEVI void rope8(float (&x)[8], const float (&cs)[8], const float (&sn)[8], int q4) {
  float o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = __shfl_xor_sync(0xffffffffu, x[i], 2);
  if (q4 < 2) {                       // x1 = mine, x2 = the partner's:  y1 = x1 c - x2 s
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = fmaf(x[i], cs[i], -o[i] * sn[i]);
  } else {                            // x2 = mine, x1 = the partner's:  y2 = x2 c + x1 s
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = fmaf(x[i], cs[i], o[i] * sn[i]);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) qkvg_fwd_kernel(const QkvgFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- the weights into shared memory, once per CTA, in the kernel's row order
  for (int i = tid; i < QF_ROWS * 16; i += G::NTHR) {
    const int row = i >> 4, c = qf_channel(row);
    const __nv_bfloat16* src = c < 384 ? p.wqkv + (size_t)c * 128 : p.wg + (size_t)(c - 384) * 128;
    cp_async16(sb + qf_woff(row, i & 15), src + (i & 15) * 8);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wb[4];                                                // this thread's B-fragment address of channel chunk kb in row g8 of block 0
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = sb + qf_woff(g8, 4 * kb + q4);

  // tiles are dealt out round-robin over the CTAs (the first gridDim.x tiles go to the warp 0 of every CTA, ...): a problem with few tiles still uses every SM
  const int ntile = (p.M + G::ROWS - 1) / G::ROWS;
  for (int tile = warp * gridDim.x + blockIdx.x; tile < ntile; tile += G::NW * gridDim.x) {
    const int t0 = tile * G::ROWS;

    // ---- rows (mt, hh): r = t0 + 16 mt + g8 + 8 hh;  n, s, and the modulation row
    int rr[G::MT][2], mrow[G::MT][2], sidx[G::MT][2], nidx[G::MT][2];
    bool rok[G::MT][2];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = t0 + mt * 16 + g8 + 8 * hh;
        rr[mt][hh] = r;
        rok[mt][hh] = r < p.M;
        const int n = rok[mt][hh] ? r / p.S : 0, s = rok[mt][hh] ? r - n * p.S : 0;
        nidx[mt][hh] = n;
        sidx[mt][hh] = s;
        mrow[mt][hh] = (n % p.B) * p.S + s;
      }

    // ---- RMS norm + modulation on the registers: a thread holds 32 of the 128 channels of its row, a quad the whole row
    uint32_t a[G::MT][8][4];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint4 xr[4];
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
          xr[kb] = rok[mt][hh] ? ldg128(p.x + (size_t)rr[mt][hh] * 128 + 32 * kb + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        float ss = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[kb];
          float e;
          e = bf16lo(u.x); ss = fmaf(e, e, ss); e = bf16hi(u.x); ss = fmaf(e, e, ss);
          e = bf16lo(u.y); ss = fmaf(e, e, ss); e = bf16hi(u.y); ss = fmaf(e, e, ss);
          e = bf16lo(u.z); ss = fmaf(e, e, ss); e = bf16hi(u.z); ss = fmaf(e, e, ss);
          e = bf16lo(u.w); ss = fmaf(e, e, ss); e = bf16hi(u.w); ss = fmaf(e, e, ss);
        }
        const float rstd = 1.f / sqrtf(quad_sum(ss) * (1.f / 128.f) + p.eps);
        const float* mrp = p.mod + (size_t)mrow[mt][hh] * 768;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[kb];
          const float4 sh0 = *reinterpret_cast<const float4*>(mrp + 32 * kb + 8 * q4), sh1 = *reinterpret_cast<const float4*>(mrp + 32 * kb + 8 * q4 + 4);
          const float4 sc0 = *reinterpret_cast<const float4*>(mrp + 128 + 32 * kb + 8 * q4), sc1 = *reinterpret_cast<const float4*>(mrp + 128 + 32 * kb + 8 * q4 + 4);
          const float xv[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
          const float sh[8] = {sh0.x, sh0.y, sh0.z, sh0.w, sh1.x, sh1.y, sh1.z, sh1.w};
          const float sc[8] = {sc0.x, sc0.y, sc0.z, sc0.w, sc1.x, sc1.y, sc1.z, sc1.w};
          float y[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) y[i] = fmaf(xv[i] * rstd, 1.f + sc[i], sh[i]);
          const uint32_t n0 = pack_bf16(y[0], y[1]), n1 = pack_bf16(y[2], y[3]), n2 = pack_bf16(y[4], y[5]), n3 = pack_bf16(y[6], y[7]);
          a[mt][2 * kb][hh] = n0;      a[mt][2 * kb][2 + hh] = n1;
          a[mt][2 * kb + 1][hh] = n2;  a[mt][2 * kb + 1][2 + hh] = n3;
          if (p.xs != nullptr && rok[mt][hh])
            stg128(p.xs + (size_t)rr[mt][hh] * 128 + 32 * kb + 8 * q4, make_uint4(n0, n1, n2, n3));
        }
      }

    // ---- 8 blocks of 64 output channels: q | k | v | g, 2 heads (og) per block
#pragma unroll 1
    for (int blk = 0; blk < 8; ++blk) {
      float acc[G::MT][8][4];
#pragma unroll
      for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
      const uint32_t wblk = 64u * blk * 256u;
      uint4 bn = lds128_ro(wb[0] + wblk);                        // the B fragments, one n tile ahead of the mma that uses them
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint4 bc = bn;
          if (kb * 8 + nt < 31) bn = lds128_ro(wb[(kb * 8 + nt + 1) >> 3] + wblk + ((kb * 8 + nt + 1) & 7) * 8 * 256);
#pragma unroll
          for (int mt = 0; mt < G::MT; ++mt) {
            mma16816(acc[mt][nt], a[mt][2 * kb], bc.x, bc.y);
            mma16816(acc[mt][nt], a[mt][2 * kb + 1], bc.z, bc.w);
          }
        }

      // ---- epilogue: per head (og), row (mt, hh): 8 consecutive channels d = 8 q4 ..
      const int proj = blk >> 1;                                  // 0 q, 1 k, 2 v, 3 g
#pragma unroll
      for (int og = 0; og < 2; ++og)
#pragma unroll
        for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {          // every lane runs the quad reductions (a row >= M only skips its stores)
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) { v[2 * j] = acc[mt][4 * og + j][2 * hh]; v[2 * j + 1] = acc[mt][4 * og + j][2 * hh + 1]; }
            const int head = 2 * (blk & 1) + og;
            const int col = 64 * (blk & 1) + 32 * og + 8 * q4;      // channel within the projection
            const size_t row = (size_t)rr[mt][hh];
            if (proj >= 2) {                                          // v (head-major) and g (row-major): bf16 of the accumulators
              uint4 u;
              u.x = pack_bf16(v[0], v[1]); u.y = pack_bf16(v[2], v[3]); u.z = pack_bf16(v[4], v[5]); u.w = pack_bf16(v[6], v[7]);
              if (rok[mt][hh]) {
                if (proj == 2) stg128(p.vh + (((size_t)nidx[mt][hh] * 4 + head) * p.S + sidx[mt][hh]) * 32 + 8 * q4, u);
                else stg128(p.g + row * 128 + col, u);
              }
            } else {                                                  // q, k: round, head RMS norm, round, RoPE, round
              float pb[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) pb[i] = rn(v[i]);
              if (p.xs != nullptr && rok[mt][hh]) {
                uint4 u;
                u.x = pack_bf16(pb[0], pb[1]); u.y = pack_bf16(pb[2], pb[3]); u.z = pack_bf16(pb[4], pb[5]); u.w = pack_bf16(pb[6], pb[7]);
                stg128((proj == 0 ? p.pqs : p.pks) + row * 128 + col, u);
              }
              float ssq = 0.f;
#pragma unroll
              for (int i = 0; i < 8; ++i) ssq = fmaf(pb[i], pb[i], ssq);
              const float rh = 1.f / sqrtf(quad_sum(ssq) * (1.f / 32.f) + p.qk_eps);
              float x[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) x[i] = rn(pb[i] * rh);
              float cs[8], sn[8];
              const float* cp = p.cos + (size_t)mrow[mt][hh] * 16 + 8 * (q4 & 1);
              const float* sp = p.sin + (size_t)mrow[mt][hh] * 16 + 8 * (q4 & 1);
              {
                const float4 c0 = *reinterpret_cast<const float4*>(cp), c1 = *reinterpret_cast<const float4*>(cp + 4);
                const float4 s0 = *reinterpret_cast<const float4*>(sp), s1 = *reinterpret_cast<const float4*>(sp + 4);
                cs[0] = c0.x; cs[1] = c0.y; cs[2] = c0.z; cs[3] = c0.w; cs[4] = c1.x; cs[5] = c1.y; cs[6] = c1.z; cs[7] = c1.w;
                sn[0] = s0.x; sn[1] = s0.y; sn[2] = s0.z; sn[3] = s0.w; sn[4] = s1.x; sn[5] = s1.y; sn[6] = s1.z; sn[7] = s1.w;
              }
              rope8(x, cs, sn, q4);
              uint4 u;
              u.x = pack_bf16(x[0], x[1]); u.y = pack_bf16(x[2], x[3]); u.z = pack_bf16(x[4], x[5]); u.w = pack_bf16(x[6], x[7]);
              if (rok[mt][hh]) stg128((proj == 0 ? p.qh : p.kh) + (((size_t)nidx[mt][hh] * 4 + head) * p.S + sidx[mt][hh]) * 32 + 8 * q4, u);
            }
          }
    }
  }
}

}  // namespace sw80
