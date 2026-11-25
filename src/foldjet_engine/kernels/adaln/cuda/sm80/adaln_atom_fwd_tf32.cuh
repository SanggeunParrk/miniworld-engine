// adaln_atom_fwd_tf32.cuh -- AdaLN forward at the atom width (d_hidden = d_cond = 128), fp32 rows, TF32 tensor cores, A100 / sm_80, ONE kernel (the fp32 twin of ``adaln_atom_fwd.cuh``):
//
//   aff = LN(cond) w                                      fp32 statistics, the operand of the products rounded to TF32 (as the cuBLAS / Triton TF32 paths do)
//   S | B = aff [Ws | Wb]^T + 0                           two 128 x 128 products on mma.sync m16n8k8 tf32, fp32 accumulation
//   y = sigmoid(S + sb) (x - mean) rstd + B               LN(x) without affine, fp32 statistics, fp32 out
//
// The three row-sized streams (x, cond in, y out: 1.5 KB a row) are all the kernel moves.  One persistent CTA of 8 warps per SM; the weights Ws | Wb (fp32, rounded to TF32 once: 128 KB) stay in shared
// memory in the f1 row order (adaln_tf32.cuh), a warp loops over tiles of 16 consecutive rows (row r reads cond row r % P).  Used for inference: nothing is saved for a backward (the training step keeps the
// composition, whose backward reads the intermediates).
#pragma once
#include "adaln_tf32.cuh"

namespace adl {

struct AdalnAtomFwdTf32Params {
  const float* x;                // [M][128]
  const float* cond;             // [P][128]
  const float* lnw;              // [128]
  const float* ws;               // [128][128]  to_scale.weight [out][in]
  const float* wb;               // [128][128]  to_bias.weight
  const float* sb;               // [128]       to_scale.bias
  float* y;                      // [M][128]
  long M, P;
  float eps_x, eps_c;
};

template <int NW_>
struct AdalnAtomFwdTf32Cfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = 1;
  static constexpr int SMEM = 2 * 128 * 512;                             // Ws | Wb, fp32
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) adaln_atom_fwd_tf32_kernel(const AdalnAtomFwdTf32Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw), sws = sbase, swb = sbase + 128 * 512;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  load_weight128_tf32(sws, p.ws, tid, G::NTHR);
  load_weight128_tf32(swb, p.wb, tid, G::NTHR);
  __syncthreads();

  const long ntile = (p.M + 15) / 16;
  for (long tile = (long)blockIdx.x * G::NW + warp; tile < ntile; tile += (long)gridDim.x * G::NW) {
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = rr[hh] < p.M; }

    // ---- every input of the tile is requested before anything waits for it
    float4 uc[2][4][2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long crow = rok[hh] ? rr[hh] % p.P : 0;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) uc[hh][gq][h] = rok[hh] ? ldg_f4(p.cond + crow * 128 + 32 * gq + 8 * q4 + 4 * h) : zero4();
    }
    float4 ux[2][4][2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) ux[hh][gq][h] = rok[hh] ? ldg_f4(p.x + rr[hh] * 128 + 32 * gq + 8 * q4 + 4 * h) : zero4();

    // ---- aff = LN(cond) w: the A fragments of the two products (k step 4 gk + s: a0 / a1 = element s of rows g8 / g8 + 8, a2 / a3 = element s + 4)
    float meanc[2], rstdc[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) s += sum4(uc[hh][gq][0]) + sum4(uc[hh][gq][1]);
      meanc[hh] = quad_sum(s) / 128.f;
      float q = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 d = make_float4(uc[hh][gq][h].x - meanc[hh], uc[hh][gq][h].y - meanc[hh], uc[hh][gq][h].z - meanc[hh], uc[hh][gq][h].w - meanc[hh]);
          q += sum4(mul4(d, d));
        }
      rstdc[hh] = rsqrtf(quad_sum(q) / 128.f + p.eps_c);
    }
    uint32_t aa[16][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const float4 w0 = ldg_f4(p.lnw + 32 * gq + 8 * q4), w1 = ldg_f4(p.lnw + 32 * gq + 8 * q4 + 4);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float cv[8] = {uc[hh][gq][0].x, uc[hh][gq][0].y, uc[hh][gq][0].z, uc[hh][gq][0].w, uc[hh][gq][1].x, uc[hh][gq][1].y, uc[hh][gq][1].z, uc[hh][gq][1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          aa[4 * gq + s][hh] = to_tf32((cv[s] - meanc[hh]) * rstdc[hh] * wv[s]);
          aa[4 * gq + s][2 + hh] = to_tf32((cv[s + 4] - meanc[hh]) * rstdc[hh] * wv[s + 4]);
        }
      }
    }

    // ---- the statistics of the x rows
    float meanx[2], rstdx[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) s += sum4(ux[hh][gq][0]) + sum4(ux[hh][gq][1]);
      meanx[hh] = quad_sum(s) / 128.f;
      float q = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 d = make_float4(ux[hh][gq][h].x - meanx[hh], ux[hh][gq][h].y - meanx[hh], ux[hh][gq][h].z - meanx[hh], ux[hh][gq][h].w - meanx[hh]);
          q += sum4(mul4(d, d));
        }
      rstdx[hh] = rsqrtf(quad_sum(q) / 128.f + p.eps_x);
    }

    // ---- per output group of 32 channels (4 n tiles): S | B = aff [Ws | Wb]^T over the 16 k steps, then the gate and the sum
#pragma unroll
    for (int og = 0; og < 4; ++og) {
      float as[4][4], ab[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) { as[t][0] = as[t][1] = as[t][2] = as[t][3] = 0.f; ab[t][0] = ab[t][1] = ab[t][2] = ab[t][3] = 0.f; }
#pragma unroll
      for (int gk = 0; gk < 4; ++gk)
#pragma unroll
        for (int sp = 0; sp < 2; ++sp) {                                  // k steps 2 sp, 2 sp + 1 of group gk: one LDS.128 a tile and matrix holds (b0, b1) of both
          float4 bs[4], bb[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint32_t row = (4 * og + t) * 8 + g8, cg = 8 * gk + 2 * q4 + sp;
            bs[t] = lds_f4(sws + tf_woff(row, cg));
            bb[t] = lds_f4(swb + tf_woff(row, cg));
          }
#pragma unroll
          for (int ss = 0; ss < 2; ++ss)                                  // the 8 accumulators are independent: a tile's next MMA is 8 MMAs away
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              mma1688(as[t], aa[4 * gk + 2 * sp + ss], __float_as_uint(ss ? bs[t].z : bs[t].x), __float_as_uint(ss ? bs[t].w : bs[t].y));
              mma1688(ab[t], aa[4 * gk + 2 * sp + ss], __float_as_uint(ss ? bb[t].z : bb[t].x), __float_as_uint(ss ? bb[t].w : bb[t].y));
            }
        }
      const float4 sb0 = ldg_f4(p.sb + 32 * og + 8 * q4), sb1 = ldg_f4(p.sb + 32 * og + 8 * q4 + 4);
      const float sbv[8] = {sb0.x, sb0.y, sb0.z, sb0.w, sb1.x, sb1.y, sb1.z, sb1.w};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float xv[8] = {ux[hh][og][0].x, ux[hh][og][0].y, ux[hh][og][0].z, ux[hh][og][0].w, ux[hh][og][1].x, ux[hh][og][1].y, ux[hh][og][1].z, ux[hh][og][1].w};
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float g = sigmoidf(as[i >> 1][2 * hh + (i & 1)] + sbv[i]);
          o[i] = fmaf(g, (xv[i] - meanx[hh]) * rstdx[hh], ab[i >> 1][2 * hh + (i & 1)]);
        }
        if (rok[hh]) {
          stg_f4(p.y + rr[hh] * 128 + 32 * og + 8 * q4, make_float4(o[0], o[1], o[2], o[3]));
          stg_f4(p.y + rr[hh] * 128 + 32 * og + 8 * q4 + 4, make_float4(o[4], o[5], o[6], o[7]));
        }
      }
    }
  }
}

}  // namespace adl
