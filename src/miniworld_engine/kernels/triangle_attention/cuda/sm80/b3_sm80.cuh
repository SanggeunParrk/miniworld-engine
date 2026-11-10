// b3_sm80.cuh -- the backward of the back (f3_sm80.cuh), A100 / sm_80: the output projection's input gradient and the gate's two gradients in one pass.
//
//   dy = bf16(dout ds)                 the residual's gradient through the dropout scale (training; dy = dout without dropout)
//   da = bf16(dy . Wo)                 to_out's input gradient
//   s = sigmoid(g);  a = bf16(s o)     the gated product (the operand of dWo = dy^T a: written for the weight-gradient GEMM)
//   do = bf16(da s)                    gradient of the attention output
//   dg = bf16(da o s (1 - s))          gradient of the gate pre-activation
//
// Same layout as the back: a persistent CTA per SM, a warp walks tiles of 16 tokens (one m16 tile), the tokens go from global memory straight into the
// A fragments (k order permuted so a thread holds 8 consecutive channels), the packed weight rows follow f1_channel so the accumulators hold 8 consecutive
// output channels per thread and every store is 16 bytes.  The product is da[t][ci] = sum_o dy[t][o] Wo[o][ci], so the B operand is Wo^T: the host hands
// the kernel Wo^T (``wot``) and the shared-memory fill gathers its rows exactly as the back gathers Wo's.  ``dout`` is read at the transposed position for
// the ending node; everything else is in the starting frame.
#pragma once
#include "f3_sm80.cuh"

namespace a100 {

struct B3Params {
  const __nv_bfloat16* dout;    // [T][128] the output's gradient, in the module's layout
  const __nv_bfloat16* ds;      // [Z][L][128] dropout scale or nullptr
  const __nv_bfloat16* o;       // [T][128] attention output (starting frame)
  const __nv_bfloat16* g;       // [T][ldg] gate pre-activation (columns 0 .. 127 of each row)
  const __nv_bfloat16* wot;     // [128][128] Wo^T: wot[ci][o] = Wo[o][ci]
  __nv_bfloat16* dg;            // [T][lddg] gate gradient (the gate columns of the [T, 512] gradient buffer)
  __nv_bfloat16* dov;           // [T][128] gradient of the attention output
  __nv_bfloat16* dy;            // [T][128] to_out's output gradient after the dropout scale (starting frame)
  __nv_bfloat16* a;             // [T][128] sigmoid(g) o, bf16
  float* delta;                 // [Z][4][L][L] sum over the head's 32 channels of o * do (the attention backward's row term) or nullptr
  long long ldg, lddg;
  unsigned ntile;               // T / 16
  int L, transposed;
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) b3_kernel(const B3Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- Wo^T into shared memory, once per CTA: packed row s holds the output channel f1_channel(s) of da
  for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(sb + f1_woff(i >> 4, i & 15), p.wot + (size_t)f1_channel(i >> 4) * 128 + (i & 15) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = sb + f1_woff(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    // 16 tokens of one pair row (L is a multiple of 128): problem z, row arow, columns b0 .. b0 + 15
    const unsigned t0 = tile * 16, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;
    size_t drow[2];                                              // the token's row in the module's layout (dout)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const unsigned r = g8 + 8 * hh;
      drow[hh] = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
    }

    // ---- loads: dout (and the dropout scale) for the A fragments; o and g for the epilogue
    uint4 dv[2][4], ov[2][4], gv[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const size_t tok = (size_t)t0 + g8 + 8 * hh;
        dv[hh][kb] = ldg128(p.dout + drow[hh] * 128 + 32 * kb + 8 * q4);
        ov[hh][kb] = ldg128(p.o + tok * 128 + 32 * kb + 8 * q4);
        gv[hh][kb] = ldg128(p.g + tok * p.ldg + 32 * kb + 8 * q4);
      }
    if (p.ds != nullptr) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 d = ldg128(p.ds + ((size_t)z * L + b0 + g8 + 8 * hh) * 128 + 32 * kb + 8 * q4);
          uint4& u = dv[hh][kb];
          u.x = mul_bf16x2(u.x, d.x); u.y = mul_bf16x2(u.y, d.y); u.z = mul_bf16x2(u.z, d.z); u.w = mul_bf16x2(u.w, d.w);
        }
    }

    // ---- dy as the A fragments (a0 / a2 of step 2 kb: x / y of row g8, a1 / a3: of row g8 + 8; step 2 kb + 1: z / w)
    uint32_t a[8][4];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4 u = dv[hh][kb];
        a[2 * kb][hh] = u.x;      a[2 * kb][2 + hh] = u.y;
        a[2 * kb + 1][hh] = u.z;  a[2 * kb + 1][2 + hh] = u.w;
      }

    // ---- da = dy . Wo: 16 n tiles of 8 channels
    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    uint4 bn = lds128_ro(wb[0]);
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 bc = bn;
        if (kb * 16 + nt < 63) bn = lds128_ro(wb[(kb * 16 + nt + 1) >> 4] + ((kb * 16 + nt + 1) & 15) * 8 * 256);
        mma16816(acc[nt], a[2 * kb], bc.x, bc.y);
        mma16816(acc[nt], a[2 * kb + 1], bc.z, bc.w);
      }

    // ---- epilogue: per (channel group, row): the gate's gradients from da, o and g, 16 bytes (8 channels) per thread and output
#pragma unroll
    for (int og = 0; og < 4; ++og)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const size_t tok = (size_t)t0 + g8 + 8 * hh;
        const uint4 ou = ov[hh][og], gu = gv[hh][og];
        const uint32_t ow[4] = {ou.x, ou.y, ou.z, ou.w}, gw[4] = {gu.x, gu.y, gu.z, gu.w};
        uint32_t d_o[4], d_g[4], av[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float da0 = round_bf16f(acc[4 * og + j][2 * hh]), da1 = round_bf16f(acc[4 * og + j][2 * hh + 1]);
          const float o0 = bf16lo(ow[j]), o1 = bf16hi(ow[j]);
          const float s0 = sigmoid(bf16lo(gw[j])), s1 = sigmoid(bf16hi(gw[j]));
          d_o[j] = pack_bf16(da0 * s0, da1 * s1);
          d_g[j] = pack_bf16(da0 * o0 * (s0 * (1.f - s0)), da1 * o1 * (s1 * (1.f - s1)));
          av[j] = pack_bf16(o0 * s0, o1 * s1);
        }
        const size_t col = 32 * og + 8 * q4;
        stg128(p.dov + tok * 128 + col, make_uint4(d_o[0], d_o[1], d_o[2], d_o[3]));
        stg128(p.dg + tok * p.lddg + col, make_uint4(d_g[0], d_g[1], d_g[2], d_g[3]));
        stg128(p.a + tok * 128 + col, make_uint4(av[0], av[1], av[2], av[3]));
        stg128(p.dy + tok * 128 + col, make_uint4(a[2 * og][hh], a[2 * og][2 + hh], a[2 * og + 1][hh], a[2 * og + 1][2 + hh]));
        if (p.delta != nullptr) {                                // delta[head og][token] = sum_d o do over the head's 32 channels (4 threads x 8): a quad sum
          float dl = 0.f;
#pragma unroll
          for (int j = 0; j < 4; ++j) dl = fmaf(bf16lo(ow[j]), bf16lo(d_o[j]), fmaf(bf16hi(ow[j]), bf16hi(d_o[j]), dl));
          dl = quad_sum(dl);
          if (q4 == 0) p.delta[(size_t)(z * 4 + og) * LL + rem0 + g8 + 8 * hh] = dl;
        }
      }
  }
}

}  // namespace a100
