// f3_sm80.cuh -- the back of the triangle attention, A100 / sm_80: the gate, the output projection and the residual in one pass.
//
//   a = bf16(sigmoid(g) o)         per token and channel: the module's sigmoid-gate statement, rounded to bf16 as the operand of the product
//   y = bf16(a . Wo^T)             to_out
//   out = bf16(res + y)            the residual add in bf16 (the module's ``pair + out``)
//
// Token t = (a, b) of the "starting" problem; ``transposed`` reads the residual from and writes the result to row (b L + a) (the ending node, with no
// transposing copy on either side).  The layout is that of the front (f1_sm80.cuh): a persistent CTA per SM, Wo resident in shared memory, every warp
// walking tiles (here of 16 tokens, one m16 tile) on its own; o and g go from global memory straight into the A fragments (the k order permuted so a
// thread holds 8 consecutive channels), the packed Wo rows follow f1_channel so the accumulators hold 8 consecutive output channels per thread and
// the residual / result are 16-byte accesses.  The product is a fifth of the traffic's time at most (32 KFLOP against 1 KiB per token): the
// kernel is a memory stream, the tensor pipe is mostly idle.
#pragma once
#include "f1_sm80.cuh"

namespace a100 {

struct F3Params {
  const __nv_bfloat16* o;       // [T][128] attention output, token-major
  const __nv_bfloat16* g;       // [T][ldg] gate pre-activation (columns 0 .. 127 of each row)
  const __nv_bfloat16* wo;      // [128][128] to_out.weight [out][in]
  const __nv_bfloat16* res;     // [T][128] residual, in the module's layout
  const __nv_bfloat16* ds;      // [Z][L][128] dropout scale (broadcast over the pair row; training) or nullptr
  __nv_bfloat16* out;           // [T][128] result, in the module's layout
  long long ldg;
  unsigned ntile;               // T / 16
  int L, transposed;            // T = Z L^2 tokens, L a multiple of 128
};

template <int NW_>
struct F3Cfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32;
  static constexpr int SMEM = 128 * 256;                        // Wo' (rows of 256 B)
};

DEVI uint32_t mul_bf16x2(uint32_t a, uint32_t b) { return pack_bf16(bf16lo(a) * bf16lo(b), bf16hi(a) * bf16hi(b)); }   // bf16(a b) per half: the framework's bf16 product

DEVI uint32_t gate_pair(uint32_t o, uint32_t g) {              // bf16(sigmoid(g) o) of two channels
  return pack_bf16(bf16lo(o) * sigmoid(bf16lo(g)), bf16hi(o) * sigmoid(bf16hi(g)));
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) f3_kernel(const F3Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- Wo into shared memory, once per CTA: packed row s holds output channel f1_channel(s)
  for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(sb + f1_woff(i >> 4, i & 15), p.wo + (size_t)f1_channel(i >> 4) * 128 + (i & 15) * 8);
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
    size_t drow[2];                                              // the token's row in the module's layout (residual, result)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const unsigned r = g8 + 8 * hh;
      drow[hh] = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
    }

    // ---- loads: o and g (A fragments), then the residual rows (needed only at the end)
    uint4 ov[2][4], gv[2][4], rv[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const size_t tok = (size_t)t0 + g8 + 8 * hh;
        ov[hh][kb] = ldg128(p.o + tok * 128 + 32 * kb + 8 * q4);
        gv[hh][kb] = ldg128(p.g + tok * p.ldg + 32 * kb + 8 * q4);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int og = 0; og < 4; ++og) rv[hh][og] = ldg128(p.res + drow[hh] * 128 + 32 * og + 8 * q4);

    // ---- the gate on the registers: the A fragments (a0 / a2 of step 2 kb: x / y of row g8, a1 / a3: of row g8 + 8; step 2 kb + 1: z / w)
    uint32_t a[8][4];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4 o = ov[hh][kb], g = gv[hh][kb];
        a[2 * kb][hh] = gate_pair(o.x, g.x);      a[2 * kb][2 + hh] = gate_pair(o.y, g.y);
        a[2 * kb + 1][hh] = gate_pair(o.z, g.z);  a[2 * kb + 1][2 + hh] = gate_pair(o.w, g.w);
      }

    // ---- the product: 16 n tiles of 8 output channels
    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    uint4 bn = lds128_ro(wb[0]);                                 // the B fragments, one n tile ahead of the mma that uses them
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 bc = bn;
        if (kb * 16 + nt < 63) bn = lds128_ro(wb[(kb * 16 + nt + 1) >> 4] + ((kb * 16 + nt + 1) & 15) * 8 * 256);
        mma16816(acc[nt], a[2 * kb], bc.x, bc.y);
        mma16816(acc[nt], a[2 * kb + 1], bc.z, bc.w);
      }

    // ---- epilogue: y in bf16 (x the dropout scale in bf16 in training), + the residual in bf16, 16 bytes (8 channels) per thread and row
#pragma unroll
    for (int og = 0; og < 4; ++og)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4 r = rv[hh][og];
        uint32_t y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) y[j] = pack_bf16(acc[4 * og + j][2 * hh], acc[4 * og + j][2 * hh + 1]);
        if (p.ds != nullptr) {                                  // out = res + bf16(y ds): ds[z][column][channel], the column being the token's second index
          const uint4 d = ldg128(p.ds + ((size_t)z * L + b0 + g8 + 8 * hh) * 128 + 32 * og + 8 * q4);
          y[0] = mul_bf16x2(y[0], d.x); y[1] = mul_bf16x2(y[1], d.y); y[2] = mul_bf16x2(y[2], d.z); y[3] = mul_bf16x2(y[3], d.w);
        }
        uint4 v;
        v.x = add_bf16x2(r.x, y[0]);
        v.y = add_bf16x2(r.y, y[1]);
        v.z = add_bf16x2(r.z, y[2]);
        v.w = add_bf16x2(r.w, y[3]);
        stg128(p.out + drow[hh] * 128 + 32 * og + 8 * q4, v);
      }
  }
}

}  // namespace a100
