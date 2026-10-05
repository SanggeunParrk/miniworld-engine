// tail_bwd.cuh -- the edge tail backward on the A100 (sm_80), part 1: LayerNorm / dropout / residual backward and the three dX layers in ONE kernel.
//
// Operands of the forward (kept by the saveact policy): values, (mean, rstd) per row, the packed dropout words, the GELU derivatives d2 = gelu'(hid) and d1 = gelu'(pre) (fp16); the weight image of the backward (pack.cuh: the
// transposed weights in the f1 row order, so that the dX products chain through registers exactly as the forward's do).  Per tile of 16 rows, in registers (the Triton path's
// ``compute`` rounding points):
//
//   xhat = (values - mean) rstd         sc = go gamma          gv  = rn(rstd (sc - mean(sc) - xhat mean(sc xhat)))      the residual branch's gradient
//   G3   = rn(keep ? gv / (1 - p) : 0)  (the gradient of the third product's output)
//   G2   = rn((G3 W3) d2)               G1 = rn((G2 W2) d1)                 grad_edge = rn(gv + G1 W1e)
//
// G3, G2 and G1 also go to global memory: the weight-gradient kernel (dw.cuh) contracts them over the rows.  dgamma = sum go xhat and dbeta = sum go accumulate in registers over the
// CTA's tiles and leave as one [2][128] partial per CTA (summed in a fixed order afterwards: a replay is bit-identical).  Every product runs in two halves of 64 channels.
#pragma once
#include "tail_fwd.cuh"

namespace me80 {

struct TailBwdParams {
  const __nv_bfloat16* go;       // [rows][128] gradient of the output
  const __nv_bfloat16* values;   // [rows][128]
  const float2* stats;           // [rows] (mean, rstd)
  const uint32_t* keep;          // [rows][4] or nullptr (no dropout)
  const __nv_bfloat16* d2;       // [rows][128] fp16 bits: gelu'(hid)
  const __nv_bfloat16* d1;       // [rows][128] fp16 bits: gelu'(pre)
  const uint8_t* img;            // backward weight image
  const float* tab;              // [b2 | b3 | gamma | beta] (only gamma is read)
  __nv_bfloat16 *g3, *g2, *g1, *gedge;   // [rows][128]
  float* ln_part;                // [grid][2][128]: dgamma | dbeta partial of every CTA
  int rows;
  float scale;                   // 1 / (1 - p)
};

template <int NW, bool DROP>
__global__ void __launch_bounds__(NW * 32, 1) tail_bwd_kernel(const TailBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* tab = reinterpret_cast<const float*>(smem_raw + IMG_BYTES);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < IMG_BYTES / 16; i += NW * 32) cp_async16(sb + i * 16, p.img + (size_t)i * 16);
  cp_async_commit();
  {
    float* t = reinterpret_cast<float*>(smem_raw + IMG_BYTES);
    for (int i = tid; i < TAB_FLOATS; i += NW * 32) t[i] = p.tab[i];
  }
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wq[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wq[kb] = woff(g8, 4 * kb + q4);

  float dga[32], dba[32];                        // this thread's 32 channels: index 8 aa + 2 s + e <-> channel 32 aa + 8 q4 + 2 s + e
#pragma unroll
  for (int i = 0; i < 32; ++i) { dga[i] = 0.f; dba[i] = 0.f; }

  const int ntile = (p.rows + 15) >> 4;
  for (int tile = blockIdx.x * NW + warp; tile < ntile; tile += gridDim.x * NW) {
    int row[2], rr[2];
    bool ok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      row[hh] = tile * 16 + g8 + 8 * hh;
      ok[hh] = row[hh] < p.rows;
      rr[hh] = ok[hh] ? row[hh] : p.rows - 1;
    }

    // ---- LayerNorm backward (statistics from the forward), dropout backward
    uint32_t gvq[2][4][4];                       // gv as bf16 pairs: the residual that joins the last product's result
    uint32_t a[8][4];                            // the A fragments of the next product (G3 first)
    {
      uint4 go[2][4], vv[2][4];
      float2 st[2];
      uint32_t kw[2] = {0xffffffffu, 0xffffffffu};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const size_t o = (size_t)rr[hh] * D + 8 * q4;
#pragma unroll
        for (int aa = 0; aa < 4; ++aa) { go[hh][aa] = ldg128_stream(p.go + o + 32 * aa); vv[hh][aa] = ldg128_stream(p.values + o + 32 * aa); }
        st[hh] = p.stats[rr[hh]];
        if (DROP) kw[hh] = p.keep[(size_t)rr[hh] * 4 + q4];
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float mean = st[hh].x, rstd = st[hh].y;
        float s_sc = 0.f, s_scx = 0.f;
#pragma unroll
        for (int aa = 0; aa < 4; ++aa) {
          const float4 g0 = *reinterpret_cast<const float4*>(tab + 256 + 32 * aa + 8 * q4), g1 = *reinterpret_cast<const float4*>(tab + 256 + 32 * aa + 8 * q4 + 4);
          const float gam[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t gp = comp(go[hh][aa], s), vp = comp(vv[hh][aa], s);
            const float x0 = (bf16lo(vp) - mean) * rstd, x1 = (bf16hi(vp) - mean) * rstd;
            const float o0 = bf16lo(gp), o1 = bf16hi(gp);
            const float c0 = o0 * gam[2 * s], c1 = o1 * gam[2 * s + 1];
            s_sc += c0 + c1;
            s_scx = fmaf(c0, x0, fmaf(c1, x1, s_scx));
            if (ok[hh]) {
              dga[8 * aa + 2 * s] = fmaf(o0, x0, dga[8 * aa + 2 * s]);         dba[8 * aa + 2 * s] += o0;
              dga[8 * aa + 2 * s + 1] = fmaf(o1, x1, dga[8 * aa + 2 * s + 1]); dba[8 * aa + 2 * s + 1] += o1;
            }
          }
        }
        const float m1 = quad_sum(s_sc) * (1.f / 128.f), m2 = quad_sum(s_scx) * (1.f / 128.f);
#pragma unroll
        for (int aa = 0; aa < 4; ++aa) {
          const float4 g0 = *reinterpret_cast<const float4*>(tab + 256 + 32 * aa + 8 * q4), g1 = *reinterpret_cast<const float4*>(tab + 256 + 32 * aa + 8 * q4 + 4);
          const float gam[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
          uint32_t g3p[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t gp = comp(go[hh][aa], s), vp = comp(vv[hh][aa], s);
            const float x0 = (bf16lo(vp) - mean) * rstd, x1 = (bf16hi(vp) - mean) * rstd;
            const float c0 = bf16lo(gp) * gam[2 * s], c1 = bf16hi(gp) * gam[2 * s + 1];
            const float v0 = rstd * (c0 - m1 - x0 * m2), v1 = rstd * (c1 - m1 - x1 * m2);
            const uint32_t gvp = pack_bf16(v0, v1);
            gvq[hh][aa][s] = gvp;
            if (DROP) {
              const float d0 = ((kw[hh] >> (8 * aa + 2 * s)) & 1u) ? v0 * p.scale : 0.f;
              const float d1 = ((kw[hh] >> (8 * aa + 2 * s + 1)) & 1u) ? v1 * p.scale : 0.f;
              g3p[s] = pack_bf16(d0, d1);
            } else {
              g3p[s] = gvp;
            }
          }
          if (ok[hh]) stg128(p.g3 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(g3p[0], g3p[1], g3p[2], g3p[3]));
#pragma unroll
          for (int s = 0; s < 4; ++s) A_SLOT(a, aa, s, hh) = g3p[s];
        }
      }
    }

    // ---- dX3: G2 = rn((G3 W3) gelu'(hid))
    uint32_t a2[8][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 hv[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) hv[hh][j] = ldg128_stream(p.d2 + (size_t)rr[hh] * D + 8 * q4 + 32 * (2 * h + j));
      float acc[8][4];
      gemm_half(acc, a, sb + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          uint32_t g2p[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t hx = comp(hv[hh][j], s);
            g2p[s] = pack_bf16(acc[4 * j + s][2 * hh] * f16lo(hx), acc[4 * j + s][2 * hh + 1] * f16hi(hx));
          }
          if (ok[hh]) stg128(p.g2 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(g2p[0], g2p[1], g2p[2], g2p[3]));
#pragma unroll
          for (int s = 0; s < 4; ++s) A_SLOT(a2, aa, s, hh) = g2p[s];
        }
    }

    // ---- dX2: G1 = rn((G2 W2) gelu'(pre))
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 pv[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) pv[hh][j] = ldg128_stream(p.d1 + (size_t)rr[hh] * D + 8 * q4 + 32 * (2 * h + j));
      float acc[8][4];
      gemm_half(acc, a2, sb + LAYER_BYTES + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          uint32_t g1p[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t x = comp(pv[hh][j], s);
            g1p[s] = pack_bf16(acc[4 * j + s][2 * hh] * f16lo(x), acc[4 * j + s][2 * hh + 1] * f16hi(x));
          }
          if (ok[hh]) stg128(p.g1 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(g1p[0], g1p[1], g1p[2], g1p[3]));
#pragma unroll
          for (int s = 0; s < 4; ++s) A_SLOT(a, aa, s, hh) = g1p[s];
        }
    }

    // ---- dX1: grad_edge = rn(gv + G1 W1e)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float acc[8][4];
      gemm_half(acc, a, sb + 2 * LAYER_BYTES + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          uint32_t ge[4];
#pragma unroll
          for (int s = 0; s < 4; ++s)
            ge[s] = pack_bf16(bf16lo(gvq[hh][aa][s]) + acc[4 * j + s][2 * hh], bf16hi(gvq[hh][aa][s]) + acc[4 * j + s][2 * hh + 1]);
          if (ok[hh]) stg128(p.gedge + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(ge[0], ge[1], ge[2], ge[3]));
        }
    }
  }

  // ---- dgamma / dbeta of this warp: the 8 lanes of a column (the g lanes) are summed by shuffles, the lanes g = 0 write the warp's [2][128] sums to shared memory (the weight tiles are
  // dead once every warp is here), and the CTA's 256 threads add the NW warps in a fixed order: one [2][128] partial per CTA
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    dga[i] += __shfl_xor_sync(0xffffffffu, dga[i], 4); dba[i] += __shfl_xor_sync(0xffffffffu, dba[i], 4);
    dga[i] += __shfl_xor_sync(0xffffffffu, dga[i], 8); dba[i] += __shfl_xor_sync(0xffffffffu, dba[i], 8);
    dga[i] += __shfl_xor_sync(0xffffffffu, dga[i], 16); dba[i] += __shfl_xor_sync(0xffffffffu, dba[i], 16);
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem_raw);
  if (g8 == 0) {
    float* dst = red + warp * 256;
#pragma unroll
    for (int aa = 0; aa < 4; ++aa) {
      *reinterpret_cast<float4*>(dst + 32 * aa + 8 * q4) = make_float4(dga[8 * aa], dga[8 * aa + 1], dga[8 * aa + 2], dga[8 * aa + 3]);
      *reinterpret_cast<float4*>(dst + 32 * aa + 8 * q4 + 4) = make_float4(dga[8 * aa + 4], dga[8 * aa + 5], dga[8 * aa + 6], dga[8 * aa + 7]);
      *reinterpret_cast<float4*>(dst + 128 + 32 * aa + 8 * q4) = make_float4(dba[8 * aa], dba[8 * aa + 1], dba[8 * aa + 2], dba[8 * aa + 3]);
      *reinterpret_cast<float4*>(dst + 128 + 32 * aa + 8 * q4 + 4) = make_float4(dba[8 * aa + 4], dba[8 * aa + 5], dba[8 * aa + 6], dba[8 * aa + 7]);
    }
  }
  __syncthreads();
  for (int c = tid; c < 256; c += NW * 32) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w * 256 + c];
    p.ln_part[(size_t)blockIdx.x * 256 + c] = s;
  }
}

}  // namespace me80
