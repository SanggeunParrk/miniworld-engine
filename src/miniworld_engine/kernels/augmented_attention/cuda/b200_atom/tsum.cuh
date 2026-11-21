// tsum.cuh — the 16-row tile schedule of the backward row kernels (tr_bwd_gate, post_bwd, pre_bwd) and, when the samples share one
// conditioning (the tables have mrows < M rows; table row = r % mrows), the gradient of the tables summed over the samples inside the kernels.
//
// Per-sample tables (mrows == M): the CTAs take the tiles round robin, the two warpgroups of a CTA alternating (the kernels' original schedule).
// Shared tables: the M / 16 tiles in the sample-inner order g = t A + a (table tile t = rows 16 t .. of the tables, sample a: rows
// a mrows + 16 t ..) are cut into contiguous ranges, one per warpgroup slot u = 2 blockIdx.x + w of the U = 2 gridDim.x: [lo(u), lo(u + 1)),
// lo(u) = floor(ntile u / U). A slot so runs through the samples of a table tile in order; each compute thread adds its 8-column chunks of the
// tile's table-gradient blocks (bf16 per sample, the values the per-sample path stores) into fp32 sums and, where a table tile's run ends,
// writes them to the partial slot (t, j = u - owner(t A)) of PART [mrows / 16][J][16][C] (fp32; owner(g) = the slot whose range holds tile
// g). sm100_atom_local's local_table_sum adds the partials of each table tile in slot order (deterministic) into the bf16 table gradient.
// SPDX-License-Identifier: Apache-2.0
#pragma once

struct TileSched {
  int u, U, ntile, A, mrows, lo, cnt;
  // w: the warpgroup; ntT: the CTA's round-robin tile count (per-sample tables)
  DEVI void init(int w, int ntT, int ntile_, int mrows_) {
    ntile = ntile_;
    mrows = mrows_;
    U = 2 * (int)gridDim.x;
    A = mrows < 16 * ntile ? 16 * ntile / mrows : 0;
    if (A) {
      u = 2 * (int)blockIdx.x + w;
      lo = lo_of(u);
      cnt = lo_of(u + 1) - lo;
    } else {
      u = w;
      lo = 0;
      cnt = (ntT - w + 1) / 2;
    }
  }
  DEVI int lo_of(int v) const { return (int)(((long long)ntile * v) / U); }
  DEVI int owner(int g) const { return (int)((((long long)g + 1) * U - 1) / ntile); }
  DEVI int row(int i) const {                                           // first row of this warpgroup's tile i
    if (!A) return 16 * ((int)blockIdx.x + (2 * i + u) * (int)gridDim.x);
    const int g = lo + i, t = g / A;
    return (g - t * A) * mrows + 16 * t;
  }
  DEVI bool run_end(int i) const { return i == cnt - 1 || (lo + i + 1) % A == 0; }   // (shared) the last tile of a table tile's run
  DEVI size_t slot(int i, int J) const {                                // (shared) the partial slot of tile i's run
    const int t = (lo + i) / A;
    return (size_t)t * J + (size_t)(u - owner(t * A));
  }
};

DEVI void tsum_add8(float* a, uint4 v) {                               // a[0..7] += the 8 bf16 of v
  a[0] += __uint_as_float(v.x << 16); a[1] += __uint_as_float(v.x & 0xffff0000u);
  a[2] += __uint_as_float(v.y << 16); a[3] += __uint_as_float(v.y & 0xffff0000u);
  a[4] += __uint_as_float(v.z << 16); a[5] += __uint_as_float(v.z & 0xffff0000u);
  a[6] += __uint_as_float(v.w << 16); a[7] += __uint_as_float(v.w & 0xffff0000u);
}
DEVI void tsum_put8(float* p, const float* a) {                        // 8 fp32 (32-B aligned)
  reinterpret_cast<float4*>(p)[0] = make_float4(a[0], a[1], a[2], a[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(a[4], a[5], a[6], a[7]);
}
// the thread's 8 columns of a table-gradient block held in TMEM (its lane, 8 columns from taddr): add / write out and clear
DEVI void tsum_tmem_add8(uint32_t taddr, uint4 v) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  float a[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) a[e] = __uint_as_float(r[e]);
  tsum_add8(a, v);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])), "r"(__float_as_uint(a[2])), "r"(__float_as_uint(a[3])),
                  "r"(__float_as_uint(a[4])), "r"(__float_as_uint(a[5])), "r"(__float_as_uint(a[6])), "r"(__float_as_uint(a[7])) : "memory");
}
DEVI void tsum_tmem_put8(uint32_t taddr, float* p) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  float a[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) a[e] = __uint_as_float(r[e]);
  tsum_put8(p, a);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}
DEVI void tsum_tmem_zero8(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}
