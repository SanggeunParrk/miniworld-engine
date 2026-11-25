// pre_fwd.cuh -- the attention input stage of the AF3 local atom DiT block on A100 / sm_80 (per row r of M = A N; the bf16 module's rounding points, rn = round to bf16):
//
//   xn = rn(LayerNorm(x))  (no affine, eps 1e-5);   x1 = rn(rn(s1 xn) + bi1)                                   (the AdaLN; s1 | bi1 | skv | bikv = mod[r, 0:512] from cond_fwd)
//   Q = rn(x1 Wq^T + bq),  G = rn(x1 Wg^T),  SG = rn(sigmoid(G))                                                (row-major bf16 [M, 128])
//   xkv = rn(rn(skv rn(LayerNorm(x1))) + bikv)                                                                  (the cross-attention mode's second AdaLN)
//   K = rn(xkv Wk^T),  V = rn(xkv Wv^T)
// With ``save`` it also writes x1 and xkv (the backward's operands of the weight gradients).
//
// The structure is the SWA qkvg stage's: one persistent CTA per SM of NW warps; the four [128][128] weights Wq | Wg | Wk | Wv (packed in the f1 row order) stay in shared memory; every warp
// runs its own loop over tiles of 16 MT rows; the rows go from global memory into the A fragments (LayerNorm and modulation on the registers); the accumulators of a block of 64 output
// channels are the epilogue's 8-channel pieces and the next product's A fragments.
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

constexpr int PF_ROWS_W = 512;                       // packed weight rows: Wq | Wg | Wk | Wv

struct PreFwdParams {
  const __nv_bfloat16* x;                            // [M][128]
  const __nv_bfloat16* mod;                          // [M][1024]
  const __nv_bfloat16* w;                            // [512][128]
  const float* bq;                                   // [128]
  __nv_bfloat16 *q, *k, *v, *sg;                     // [M][128]
  __nv_bfloat16 *x1s, *xkvs;                         // [M][128] or nullptr (save)
  int M;
  float eps;
  int cross;                                         // 0: K and V are projected from x1 itself (no second AdaLN)
  int prefetch;                                      // 1: the next tile's rows are requested into L2
};

template <int NW_, int MT_>
struct PreCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MT = MT_, ROWS = 16 * MT_;
  static constexpr int SMEM = PF_ROWS_W * 256 + 128 * 4;
};

template <int MT>
DEVI void pf_gemm_block(const uint32_t sb, const uint32_t (&wb)[4], const uint32_t (&a)[MT][8][4], float (&acc)[MT][8][4], int blk) {
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
  const uint32_t wblk = sb + 64u * blk * 256u;
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    uint4 b[8];                                                  // the B fragments of the 8 n tiles of k block kb
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) b[nt] = lds128_ro(wblk + nt * 8 * 256 + wb[kb]);
#pragma unroll
    for (int half = 0; half < 2; ++half)                         // the two k steps of the block: 16 independent accumulators between dependent products
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma16816(acc[mt][nt], a[mt][2 * kb + half], half ? b[nt].z : b[nt].x, half ? b[nt].w : b[nt].y);
  }
}

// the 8-channel piece (og, mt, hh) of an accumulator block: v[2 j + e] = acc[mt][4 og + j][2 hh + e]
template <int MT>
DEVI void pf_piece(const float (&acc)[MT][8][4], int mt, int og, int hh, float (&v)[8]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) { v[2 * j] = acc[mt][4 * og + j][2 * hh]; v[2 * j + 1] = acc[mt][4 * og + j][2 * hh + 1]; }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) pre_fwd_kernel(const PreFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* sbq = reinterpret_cast<const float*>(smem_raw + PF_ROWS_W * 256);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- the weights into shared memory, once per CTA, in the f1 row order
  for (int i = tid; i < PF_ROWS_W * 16; i += G::NTHR)
    cp_async16(sb + qf_woff(i >> 4, i & 15), p.w + (size_t)qf_channel(i >> 4) * 128 + (i & 15) * 8);
  cp_async_commit();
  for (int i = tid; i < 128; i += G::NTHR) reinterpret_cast<float*>(smem_raw + PF_ROWS_W * 256)[i] = p.bq[i];
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wb[4];                                                // this thread's B-fragment address of channel chunk kb in row g8 of block 0
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = qf_woff(g8, 4 * kb + q4);

  const int ntile = (p.M + G::ROWS - 1) / G::ROWS;
  for (int tile = warp * (int)gridDim.x + (int)blockIdx.x; tile < ntile; tile += G::NW * (int)gridDim.x) {
    const int t0 = tile * G::ROWS;
    if (p.prefetch) {
      const int nt = tile + G::NW * (int)gridDim.x;
      if (nt * G::ROWS < p.M) {
        pf_rows(p.x, 256, 0, 2, nt * G::ROWS, G::ROWS, p.M, lane);
        pf_rows(p.mod, 2048, 0, 8, nt * G::ROWS, G::ROWS, p.M, lane);
      }
    }
    int rr[G::MT][2];
    bool rok[G::MT][2];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        rr[mt][hh] = t0 + mt * 16 + g8 + 8 * hh;
        rok[mt][hh] = rr[mt][hh] < p.M;
      }

    // ---- x1 = AdaLN(x): LayerNorm statistics, modulation on the registers (packed bf16 arithmetic, one rounding per operation); the statistics of x1 for the key / value AdaLN
    uint32_t a[G::MT][8][4];
    float m1[G::MT][2], r1[G::MT][2];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const size_t row = (size_t)rr[mt][hh];
        float xv[4][8];
        float s = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = rok[mt][hh] ? ldg128(p.x + row * 128 + 32 * kb + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
          unpack8(u, xv[kb]);
#pragma unroll
          for (int i = 0; i < 8; ++i) s += xv[kb][i];
        }
        const float mean = quad_sum(s) * (1.f / 128.f);
        float vs = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
#pragma unroll
          for (int i = 0; i < 8; ++i) { xv[kb][i] -= mean; vs = fmaf(xv[kb][i], xv[kb][i], vs); }
        const float rstd = 1.f / sqrtf(quad_sum(vs) * (1.f / 128.f) + p.eps);
        float s1sum = 0.f;
        uint4 y1u[4];
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const __nv_bfloat16* mp = p.mod + row * 1024 + 32 * kb + 8 * q4;
          const uint4 sc = rok[mt][hh] ? ldg128(mp) : make_uint4(0u, 0u, 0u, 0u);
          const uint4 bi = rok[mt][hh] ? ldg128(mp + 128) : make_uint4(0u, 0u, 0u, 0u);
          uint4 xn, y;
          xn.x = pk2(xv[kb][0] * rstd, xv[kb][1] * rstd); xn.y = pk2(xv[kb][2] * rstd, xv[kb][3] * rstd);
          xn.z = pk2(xv[kb][4] * rstd, xv[kb][5] * rstd); xn.w = pk2(xv[kb][6] * rstd, xv[kb][7] * rstd);
          y.x = padd(pmul(sc.x, xn.x), bi.x); y.y = padd(pmul(sc.y, xn.y), bi.y);
          y.z = padd(pmul(sc.z, xn.z), bi.z); y.w = padd(pmul(sc.w, xn.w), bi.w);
          y1u[kb] = y;
          put_a(a[mt], kb, hh, y);
          if (p.x1s != nullptr && rok[mt][hh]) stg128(p.x1s + row * 128 + 32 * kb + 8 * q4, y);
#pragma unroll
          for (int e = 0; e < 4; ++e) { const uint32_t w = e == 0 ? y.x : e == 1 ? y.y : e == 2 ? y.z : y.w; s1sum += bf16lo(w) + bf16hi(w); }
        }
        const float mean1 = quad_sum(s1sum) * (1.f / 128.f);
        float v1 = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint32_t w[4] = {y1u[kb].x, y1u[kb].y, y1u[kb].z, y1u[kb].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float d0 = bf16lo(w[e]) - mean1, d1 = bf16hi(w[e]) - mean1;
            v1 = fmaf(d0, d0, fmaf(d1, d1, v1));
          }
        }
        const float rs1 = 1.f / sqrtf(quad_sum(v1) * (1.f / 128.f) + p.eps);
        r1[mt][hh] = rs1;
        m1[mt][hh] = mean1 * rs1;                                  // mean * rstd of x1
      }

    float acc[G::MT][8][4];
    // ---- Q (blocks 0, 1) and G (blocks 2, 3) from x1
#pragma unroll 1
    for (int blk = 0; blk < 4; ++blk) {
      pf_gemm_block<G::MT>(sb, wb, a, acc, blk);
      const bool isq = blk < 2;
#pragma unroll
      for (int og = 0; og < 2; ++og) {
        const int col = 64 * (blk & 1) + 32 * og + 8 * q4;
        float bq[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (isq) {
          const float4 b0 = *reinterpret_cast<const float4*>(sbq + col), b1 = *reinterpret_cast<const float4*>(sbq + col + 4);
          bq[0] = b0.x; bq[1] = b0.y; bq[2] = b0.z; bq[3] = b0.w; bq[4] = b1.x; bq[5] = b1.y; bq[6] = b1.z; bq[7] = b1.w;
        }
#pragma unroll
        for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            uint32_t o[4];
            const size_t row = (size_t)rr[mt][hh];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
              const float v0 = acc[mt][4 * og + jj][2 * hh], v1 = acc[mt][4 * og + jj][2 * hh + 1];
              if (isq) o[jj] = pk2(v0 + bq[2 * jj], v1 + bq[2 * jj + 1]);
              else { const uint32_t g = pk2(v0, v1); o[jj] = pk2(sigm(bf16lo(g)), sigm(bf16hi(g))); }
            }
            if (rok[mt][hh]) stg128((isq ? p.q : p.sg) + row * 128 + col, make_uint4(o[0], o[1], o[2], o[3]));
          }
      }
    }

    // ---- xkv = rn(rn(skv rn(LayerNorm(x1))) + bikv), in place over the fragments of x1
    if (p.cross) {
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const size_t row = (size_t)rr[mt][hh];
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 xu = get_a(a[mt], kb, hh);
          const __nv_bfloat16* mp = p.mod + row * 1024 + 256 + 32 * kb + 8 * q4;
          const uint4 sc = rok[mt][hh] ? ldg128(mp) : make_uint4(0u, 0u, 0u, 0u);
          const uint4 bi = rok[mt][hh] ? ldg128(mp + 128) : make_uint4(0u, 0u, 0u, 0u);
          const uint32_t xw[4] = {xu.x, xu.y, xu.z, xu.w};
          uint32_t xn[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) xn[e] = pk2(fmaf(bf16lo(xw[e]), r1[mt][hh], -m1[mt][hh]), fmaf(bf16hi(xw[e]), r1[mt][hh], -m1[mt][hh]));
          const uint4 u = make_uint4(padd(pmul(sc.x, xn[0]), bi.x), padd(pmul(sc.y, xn[1]), bi.y), padd(pmul(sc.z, xn[2]), bi.z), padd(pmul(sc.w, xn[3]), bi.w));
          put_a(a[mt], kb, hh, u);
          if (p.xkvs != nullptr && rok[mt][hh]) stg128(p.xkvs + row * 128 + 32 * kb + 8 * q4, u);
        }
      }
    }

    // ---- K (blocks 4, 5) and V (blocks 6, 7) from xkv
#pragma unroll 1
    for (int blk = 4; blk < 8; ++blk) {
      pf_gemm_block<G::MT>(sb, wb, a, acc, blk);
      __nv_bfloat16* dst = blk < 6 ? p.k : p.v;
#pragma unroll
      for (int og = 0; og < 2; ++og) {
        const int col = 64 * (blk & 1) + 32 * og + 8 * q4;
#pragma unroll
        for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            uint32_t o[4];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) o[jj] = pk2(acc[mt][4 * og + jj][2 * hh], acc[mt][4 * og + jj][2 * hh + 1]);
            if (rok[mt][hh]) stg128(dst + (size_t)rr[mt][hh] * 128 + col, make_uint4(o[0], o[1], o[2], o[3]));
          }
      }
    }
  }
}

}  // namespace lr80
