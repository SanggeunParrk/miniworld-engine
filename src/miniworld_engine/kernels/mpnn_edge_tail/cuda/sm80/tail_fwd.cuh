// tail_fwd.cuh -- the ProteinMPNN encoder edge tail on the A100 (sm_80), forward: the whole chain of the Triton path in one kernel (its rounding points; rn = round to bf16)
//
//   pre   = rn(rn(query[g] + rn(edge W1e^T)) + neighbor[idx])         act1 = rn(gelu(pre))
//   hid   = rn(act1 W2^T + b2)                                         act2 = rn(gelu(hid))
//   upd   = rn(act2 W3^T + b3)         dropped = rn(keep ? upd / (1 - p) : 0)        values = rn(edge + dropped)
//   out   = rn(layer_norm(values) * gamma + beta)                      statistics in fp32 over the 128 channels
//
// A persistent CTA per SM of NW warps; the three 128 x 128 weights (96 KiB, the packed image of pack.cuh) stay in shared memory for the CTA's life and every warp runs its own loop over
// tiles of 16 rows (no CTA barrier after the prologue).  A row's edge, query and gathered neighbour vectors go straight from global memory into the A fragments / epilogue operands
// (16-byte loads: the f1 permutation of common.cuh), each product's accumulators become the next product's A fragments in registers, and the epilogues write 16-byte vectors straight
// from the registers.  Every product is computed in two halves of 64 output channels (32 accumulator registers; the A fragments are read twice), which is what keeps the kernel under
// 170 registers.  With SAVE (training) the kernel also writes the backward's operands: act1 / act2 (the GELU outputs, bf16: the weight-gradient operands), d1 / d2 (the GELU derivatives at pre /
// hid, fp16: the chain backward multiplies by them, so neither GELU is evaluated again), values, the row statistics and the packed dropout decisions (one 32-bit word
// per row and quad lane, bit 8 a + 2 s + e of the thread's channel 32 a + 8 q + 2 s + e).
#pragma once
#include "common.cuh"

namespace me80 {

struct TailFwdParams {
  const __nv_bfloat16* edge;     // [rows][128]
  const __nv_bfloat16* query;    // [rows / K][128]  (the packed projection's query block, bias included)
  const __nv_bfloat16* nbr;      // [nodes][128]
  const int64_t* idx;            // [rows] flat neighbour indices
  const uint8_t* img;            // forward weight image (pack.cuh)
  const float* tab;              // b2 | b3 | gamma | beta
  const int64_t* seed;           // [1]
  __nv_bfloat16* out;            // [rows][128]
  __nv_bfloat16 *act1, *d1, *act2, *d2, *values;   // [rows][128] or nullptr (d1 / d2 hold fp16 bits)
  float2* stats;                 // [rows] (mean, rstd) or nullptr
  uint32_t* keep;                // [rows][4] or nullptr
  int rows, K;
  int row_base;                  // global index of this launch's first row (the dropout counter is keyed by the row of the whole tensor: a chunk of a replayed forward draws the same decisions)
  float eps, scale;              // scale = 1 / (1 - p)
  uint32_t thr;                  // keep threshold of the draw
};

// acc = A . W^T for 8 n tiles (64 output channels) of one packed weight tile (``base`` = the tile's address + 16384 h for half h): kb = 32-channel groups of k, n tiles in groups of 4 so a
// dependent mma is 4 issues away
DEVI void gemm_half(float (&acc)[8][4], const uint32_t (&a)[8][4], uint32_t base, const uint32_t (&wq)[4]) {
  // the B fragments of group (kb, n4) are loaded one group ahead of the mma that use them (the loads are volatile: they stay where they are written)
  uint4 bc[4], bn[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) bc[j] = lds128(base + j * 2048 + wq[0]);
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int kb = g >> 1, n4 = g & 1;
    if (g < 7) {
      const int kn = (g + 1) >> 1, nn = (g + 1) & 1;
#pragma unroll
      for (int j = 0; j < 4; ++j) bn[j] = lds128(base + (4 * nn + j) * 2048 + wq[kn]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (g < 2) mma16816_z(acc[4 * n4 + j], a[2 * kb], bc[j].x, bc[j].y);          // the first k step of the accumulator: C = 0
      else mma16816(acc[4 * n4 + j], a[2 * kb], bc[j].x, bc[j].y);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) mma16816(acc[4 * n4 + j], a[2 * kb + 1], bc[j].z, bc[j].w);
#pragma unroll
    for (int j = 0; j < 4; ++j) bc[j] = bn[j];
  }
}

DEVI uint32_t comp(const uint4& v, int s) { return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w; }
// the A fragment slots of the pair s of group aa, rows hh: a[2 aa + s / 2][2 (s % 2) + hh]
#define A_SLOT(a, aa, s, hh) a[2 * (aa) + ((s) >> 1)][2 * ((s) & 1) + (hh)]

template <int NW, bool SAVE, bool DROP>
__global__ void __launch_bounds__(NW * 32, 1) tail_fwd_kernel(const TailFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* tab = reinterpret_cast<const float*>(smem_raw + IMG_BYTES);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- the weights and the vectors, once per CTA
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
  const uint32_t dkey = DROP ? drop_key((unsigned long long)p.seed[0]) : 0u;

  const int ntile = (p.rows + 15) >> 4;
  for (int tile = blockIdx.x * NW + warp; tile < ntile; tile += gridDim.x * NW) {
    int row[2], rr[2];
    bool ok[2];
    const __nv_bfloat16 *qp[2], *np[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      row[hh] = tile * 16 + g8 + 8 * hh;
      ok[hh] = row[hh] < p.rows;
      rr[hh] = ok[hh] ? row[hh] : p.rows - 1;
    }

    // ---- the edge rows: the A fragments of product 1 and, kept in the same registers, the residual
    uint32_t a[8][4];
    {
      uint4 e[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const __nv_bfloat16* ep = p.edge + (size_t)rr[hh] * D + 8 * q4;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) e[hh][kb] = ldg128_stream(ep + 32 * kb);
        qp[hh] = p.query + (size_t)(rr[hh] / p.K) * D + 8 * q4;
        np[hh] = p.nbr + (size_t)p.idx[rr[hh]] * D + 8 * q4;
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          a[2 * kb][hh] = e[hh][kb].x;      a[2 * kb][2 + hh] = e[hh][kb].y;
          a[2 * kb + 1][hh] = e[hh][kb].z;  a[2 * kb + 1][2 + hh] = e[hh][kb].w;
        }
    }
    uint32_t res[2][4][4];                       // the residual (the edge pairs), a copy the compiler merges with the A fragments until they are overwritten
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int aa = 0; aa < 4; ++aa)
#pragma unroll
        for (int s = 0; s < 4; ++s) res[hh][aa][s] = A_SLOT(a, aa, s, hh);

    uint32_t kw[2] = {0u, 0u};                   // the packed keep decisions of the rows (bit 8 aa + 2 s + e), written with SAVE

    // ---- product 1 (two halves): pre = rn(rn(query + rn(acc)) + neighbour); act1 = rn(gelu(pre)) -> the A fragments of product 2
    uint32_t a2[8][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 qv[2][2], nv[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) { qv[hh][j] = ldg128(qp[hh] + 32 * (2 * h + j)); nv[hh][j] = ldg128(np[hh] + 32 * (2 * h + j)); }
      float acc[8][4];
      gemm_half(acc, a, sb + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          uint32_t pr[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t proj = pack_bf16(acc[4 * j + s][2 * hh], acc[4 * j + s][2 * hh + 1]);
            pr[s] = add_bf16x2(comp(nv[hh][j], s), add_bf16x2(comp(qv[hh][j], s), proj));
          }
          if (SAVE) {
            uint32_t gp[4], dp[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              float da, db;
              const float g0 = gelu_with_grad(bf16lo(pr[s]), da), g1 = gelu_with_grad(bf16hi(pr[s]), db);
              gp[s] = pack_bf16(g0, g1); dp[s] = pack_f16(da, db);
              A_SLOT(a2, aa, s, hh) = gp[s];
            }
            if (ok[hh]) {
              stg128_stream(p.act1 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(gp[0], gp[1], gp[2], gp[3]));
              stg128_stream(p.d1 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(dp[0], dp[1], dp[2], dp[3]));
            }
          } else {
#pragma unroll
            for (int s = 0; s < 4; ++s) A_SLOT(a2, aa, s, hh) = pack_bf16(gelu(bf16lo(pr[s])), gelu(bf16hi(pr[s])));
          }
        }
    }

    // ---- product 2: hid = rn(acc + b2); act2 = rn(gelu(hid))
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float acc[8][4];
      gemm_half(acc, a2, sb + LAYER_BYTES + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          const float4 b0 = *reinterpret_cast<const float4*>(tab + 32 * aa + 8 * q4), b1 = *reinterpret_cast<const float4*>(tab + 32 * aa + 8 * q4 + 4);
          const float bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
          uint32_t hp[4];
#pragma unroll
          for (int s = 0; s < 4; ++s) hp[s] = pack_bf16(acc[4 * j + s][2 * hh] + bias[2 * s], acc[4 * j + s][2 * hh + 1] + bias[2 * s + 1]);
          if (SAVE) {
            uint32_t gp[4], dp[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              float da, db;
              const float g0 = gelu_with_grad(bf16lo(hp[s]), da), g1 = gelu_with_grad(bf16hi(hp[s]), db);
              gp[s] = pack_bf16(g0, g1); dp[s] = pack_f16(da, db);
              A_SLOT(a, aa, s, hh) = gp[s];
            }
            if (ok[hh]) {
              stg128_stream(p.act2 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(gp[0], gp[1], gp[2], gp[3]));
              stg128_stream(p.d2 + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(dp[0], dp[1], dp[2], dp[3]));
            }
          } else {
#pragma unroll
            for (int s = 0; s < 4; ++s) A_SLOT(a, aa, s, hh) = pack_bf16(gelu(bf16lo(hp[s])), gelu(bf16hi(hp[s])));
          }
        }
    }

    // ---- product 3: upd = rn(acc + b3), dropout, residual: values = rn(edge + dropped)
    uint32_t vp[2][4][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float acc[8][4];
      gemm_half(acc, a, sb + 2 * LAYER_BYTES + h * 16384, wq);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int aa = 2 * h + j;
          const float4 b0 = *reinterpret_cast<const float4*>(tab + 128 + 32 * aa + 8 * q4), b1 = *reinterpret_cast<const float4*>(tab + 128 + 32 * aa + 8 * q4 + 4);
          const float bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            uint32_t up = pack_bf16(acc[4 * j + s][2 * hh] + bias[2 * s], acc[4 * j + s][2 * hh + 1] + bias[2 * s + 1]);
            if (DROP) {
              const uint32_t w = drop_word((uint32_t)(p.row_base + row[hh]) * 64u + (4 * aa + s) * 4 + q4, dkey);
              const bool k0 = (w & 0xffffu) < p.thr, k1 = (w >> 16) < p.thr;
              up = pack_bf16(k0 ? bf16lo(up) * p.scale : 0.f, k1 ? bf16hi(up) * p.scale : 0.f);
              if (SAVE) kw[hh] |= (k0 ? 1u : 0u) << (8 * aa + 2 * s) | (k1 ? 1u : 0u) << (8 * aa + 2 * s + 1);
            }
            vp[hh][aa][s] = add_bf16x2(res[hh][aa][s], up);
          }
          if (SAVE && ok[hh]) stg128_stream(p.values + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(vp[hh][aa][0], vp[hh][aa][1], vp[hh][aa][2], vp[hh][aa][3]));
        }
    }

    // ---- LayerNorm over the 128 channels of each row (the quad holds a row's 4 x 32), fp32 statistics
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s1 = 0.f;
#pragma unroll
      for (int aa = 0; aa < 4; ++aa)
#pragma unroll
        for (int s = 0; s < 4; ++s) s1 += bf16lo(vp[hh][aa][s]) + bf16hi(vp[hh][aa][s]);
      const float mean = quad_sum(s1) * (1.f / 128.f);
      float s2 = 0.f;
#pragma unroll
      for (int aa = 0; aa < 4; ++aa)
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const float d0 = bf16lo(vp[hh][aa][s]) - mean, d1 = bf16hi(vp[hh][aa][s]) - mean;
          s2 = fmaf(d0, d0, fmaf(d1, d1, s2));
        }
      const float rstd = rsqrtf_(quad_sum(s2) * (1.f / 128.f) + p.eps);
#pragma unroll
      for (int aa = 0; aa < 4; ++aa) {
        const float4 g0 = *reinterpret_cast<const float4*>(tab + 256 + 32 * aa + 8 * q4), g1 = *reinterpret_cast<const float4*>(tab + 256 + 32 * aa + 8 * q4 + 4);
        const float4 c0 = *reinterpret_cast<const float4*>(tab + 384 + 32 * aa + 8 * q4), c1 = *reinterpret_cast<const float4*>(tab + 384 + 32 * aa + 8 * q4 + 4);
        const float gam[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, bet[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        uint32_t op[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const float y0 = fmaf((bf16lo(vp[hh][aa][s]) - mean) * rstd, gam[2 * s], bet[2 * s]);
          const float y1 = fmaf((bf16hi(vp[hh][aa][s]) - mean) * rstd, gam[2 * s + 1], bet[2 * s + 1]);
          op[s] = pack_bf16(y0, y1);
        }
        if (ok[hh]) stg128(p.out + (size_t)row[hh] * D + 32 * aa + 8 * q4, make_uint4(op[0], op[1], op[2], op[3]));
      }
      if (SAVE && ok[hh]) {
        if (q4 == 0) p.stats[row[hh]] = make_float2(mean, rstd);
        if (DROP) stg32(p.keep + (size_t)row[hh] * 4 + q4, kw[hh]);
      }
    }
  }
}

}  // namespace me80
