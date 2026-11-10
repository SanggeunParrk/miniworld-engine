// rows_layout.cuh -- the operand layout shared by the row-stage kernels of the A100 AF3 local atom DiT block.
//
// A warp owns tiles of 16 MT rows. A thread (g8 = lane / 4, q4 = lane % 4) holds, of row g8 + 8 hh, the 8 channels 32 kb + 8 q4 .. + 7 of each 32-channel block kb as bf16 pairs: the
// A fragments of an m16n8k16 product with the contraction index permuted the same way on both operands (a[2 kb][hh] = channels +0,1; a[2 kb][2 + hh] = +2,3; a[2 kb + 1][hh] = +4,5;
// a[2 kb + 1][2 + hh] = +6,7), so a row's channels reach the registers from global memory in 16-byte pieces and a B fragment of 8 consecutive k is one 16-byte shared load.
// The weights are staged in the "f1" row order (qf_channel): the accumulators of a thread over the 4 n tiles of a group are 8 consecutive output channels, which are the A fragment
// pieces of the next product and one 16-byte store.
#pragma once
#include "rows_common.cuh"

namespace lr80 {

// packed row s -> the output channel it holds: in a block of 64 rows the 8 n tiles are (og, j) = (nt >> 2, nt & 3) and column n of a tile holds channel 32 og + 8 (n >> 1) + 2 j + (n & 1)
DEVI int qf_channel(int s) { return (s & ~63) + 32 * ((s >> 5) & 1) + 8 * ((s >> 1) & 3) + 2 * ((s >> 3) & 3) + (s & 1); }
// byte offset of 16-byte chunk `chunk` of packed row `row` (rows of 256 B; the chunk index is XOR-ed with 4 on odd rows: the 2 rows x 4 chunks a quarter warp reads fall in 8 bank groups)
DEVI uint32_t qf_woff(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ ((row & 1u) << 2)) << 4); }

// byte offset of 16-byte chunk `chunk` (0 .. 7) of row `row` of a [128][128 B] tile (the chunk index is XOR-ed with 4 on odd rows)
DEVI uint32_t w128_off(uint32_t row, uint32_t chunk) { return row * 128u + ((chunk ^ ((row & 1u) << 2)) << 4); }

DEVI float rn(float x) { return round_bf16f(x); }

// prefetches (into L2) ``lines`` consecutive 128-byte lines from byte offset ``off`` of each of the ``nrows`` rows (stride ``stride`` bytes) from ``row0`` of a tensor of M rows: the lanes of a warp share the work
DEVI void pf_rows(const void* base, size_t stride, int off, int lines, int row0, int nrows, int M, int lane) {
  for (int i = lane; i < nrows * lines; i += 32) {
    const int r = row0 + i / lines;
    if (r < M) prefetch_l2(reinterpret_cast<const char*>(base) + (size_t)r * stride + off + (i % lines) * 128);
  }
}
DEVI float bfl(uint32_t v) { return bf16lo(v); }
DEVI float bfh(uint32_t v) { return bf16hi(v); }

// the 8 floats of a uint4 of bf16
DEVI void unpack8(const uint4 u, float (&v)[8]) {
  v[0] = bf16lo(u.x); v[1] = bf16hi(u.x); v[2] = bf16lo(u.y); v[3] = bf16hi(u.y);
  v[4] = bf16lo(u.z); v[5] = bf16hi(u.z); v[6] = bf16lo(u.w); v[7] = bf16hi(u.w);
}
DEVI uint4 pack8(const float (&v)[8]) {
  return make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
}
// the A-fragment registers of the 8 channels of block kb of row half hh
DEVI void put_a(uint32_t (&a)[8][4], int kb, int hh, const uint4 u) {
  a[2 * kb][hh] = u.x;      a[2 * kb][2 + hh] = u.y;
  a[2 * kb + 1][hh] = u.z;  a[2 * kb + 1][2 + hh] = u.w;
}
DEVI uint4 get_a(const uint32_t (&a)[8][4], int kb, int hh) {
  return make_uint4(a[2 * kb][hh], a[2 * kb][2 + hh], a[2 * kb + 1][hh], a[2 * kb + 1][2 + hh]);
}
DEVI float rcp_approx(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float sigm(float x) { return rcp_approx(1.f + ex2f(x * -1.4426950408889634f)); }       // FMUL, MUFU.EX2, FADD, MUFU.RCP

// packed bf16x2 arithmetic with the module's rounding per operation (sm_80 has fma.rn.bf16x2 only): a b + (-0) = rn(a b) per half, a 1 + b = rn(a + b) per half
DEVI uint32_t pmul(uint32_t a, uint32_t b) { return fma_bf16x2(a, b, 0x80008000u); }
DEVI uint32_t padd(uint32_t a, uint32_t b) { return fma_bf16x2(a, 0x3f803f80u, b); }
DEVI uint32_t pk2(float lo, float hi) { return pack_bf16(lo, hi); }

}  // namespace lr80

namespace lr80 {

// LayerNorm (no affine) of a row half whose 4 x 8 values per thread (quad = the row) sit in xh: xh <- (x - mean) rstd (fp32), returns rstd
DEVI float ln_normalize(float (&xh)[4][8], float eps) {
  float s = 0.f;
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int i = 0; i < 8; ++i) s += xh[gq][i];
  const float mean = quad_sum(s) * (1.f / 128.f);
  float vs = 0.f;
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int i = 0; i < 8; ++i) { xh[gq][i] -= mean; vs = fmaf(xh[gq][i], xh[gq][i], vs); }
  const float rstd = 1.f / sqrtf(quad_sum(vs) * (1.f / 128.f) + eps);
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int i = 0; i < 8; ++i) xh[gq][i] *= rstd;
  return rstd;
}

// the backward of that LayerNorm for the gradient dxn (bf16 pairs [group][j]): out = rn(rstd (dxn - mean(dxn) - xh mean(dxn xh)))
DEVI void ln_backward(const float (&xh)[4][8], float rstd, const uint32_t (&dxn)[4][4], uint32_t (&out)[4][4]) {
  float sa = 0.f, sb = 0.f;
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float d0 = bf16lo(dxn[gq][e]), d1 = bf16hi(dxn[gq][e]);
      sa += d0 + d1;
      sb = fmaf(d0, xh[gq][2 * e], fmaf(d1, xh[gq][2 * e + 1], sb));
    }
  const float m1 = quad_sum(sa) * (1.f / 128.f), m2 = quad_sum(sb) * (1.f / 128.f);
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int e = 0; e < 4; ++e)
      out[gq][e] = pk2(rstd * (bf16lo(dxn[gq][e]) - m1 - xh[gq][2 * e] * m2), rstd * (bf16hi(dxn[gq][e]) - m1 - xh[gq][2 * e + 1] * m2));
}

DEVI uint32_t omin2(uint32_t s) { return pk2(1.f - bf16lo(s), 1.f - bf16hi(s)); }          // rn(1 - s) per half

}  // namespace lr80

namespace lr80 {

// Sums the NV values of every lane over the 8 lanes that share lane & 3 (lane bits 2, 3, 4: the rows g8 of an mma tile) and scatters the sums: after the call v[0 .. NV / 8) of a lane hold the sums of the
// values base .. base + NV / 8 - 1, base = NV / 2 b2 + NV / 4 b3 + NV / 8 b4 (b2, b3, b4 the lane's bits 2, 3, 4).  NV - NV / 8 shuffles per lane.
template <int NV>
DEVI void reduce_scatter_g8(float (&v)[NV], int lane) {
  static_assert(NV % 8 == 0, "NV must be a multiple of 8");
  const bool b2 = (lane & 4) != 0, b3 = (lane & 8) != 0, b4 = (lane & 16) != 0;
#pragma unroll
  for (int i = 0; i < NV / 2; ++i) {
    const float lo = v[i], hi = v[NV / 2 + i];
    v[i] = (b2 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b2 ? lo : hi, 4);
  }
#pragma unroll
  for (int i = 0; i < NV / 4; ++i) {
    const float lo = v[i], hi = v[NV / 4 + i];
    v[i] = (b3 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b3 ? lo : hi, 8);
  }
#pragma unroll
  for (int i = 0; i < NV / 8; ++i) {
    const float lo = v[i], hi = v[NV / 8 + i];
    v[i] = (b4 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b4 ? lo : hi, 16);
  }
}
DEVI int g8_base(int lane, int nv) { return (nv / 2) * ((lane >> 2) & 1) + (nv / 4) * ((lane >> 3) & 1) + (nv / 8) * ((lane >> 4) & 1); }

}  // namespace lr80
