// adaln_tf32.cuh -- the fp32 (TF32 tensor cores) building blocks of the atom-width kernels, A100 / sm_80: mma.sync m16n8k8 tf32 -> fp32, the TF32 rounding, and the layouts of fp32 rows and weights.
//
// The row layout is the f1 order of adaln_mma.cuh with fp32 elements: lane = 4 g8 + q4 holds rows g8 and g8 + 8; of a row's 128 channels the 32 channels 32 gq + 8 q4 + 0 .. 7 of group gq = 0 .. 3: two
// 16-byte vectors (e = 0 .. 3 and e = 4 .. 7).  The A fragment of m16n8k8 is a0 = A[g8][q4], a1 = A[g8 + 8][q4], a2 = A[g8][q4 + 4], a3 = A[g8 + 8][q4 + 4]: the k index of a step is free, so the lane's own
// element e = s of its chunk is "k = q4" and its element e = s + 4 is "k = q4 + 4" of step s (s = 0 .. 3 inside a group of 32 channels, 16 steps for K = 128): the fragments are the loaded registers, no shuffle.  The
// B fragment b0 = B[q4][g8], b1 = B[q4 + 4][g8] of an output row o needs the same two elements of row o of the weight: a thread reads the 8 contiguous floats 32 gk + 8 q4 + 0 .. 7 of weight row o (two LDS.128
// serve the 4 steps of the group).  Weight rows are stored in the order ``qf_channel`` (adaln_mma.cuh), so the accumulators (c0, c1 = row g8, outputs 2 q4, 2 q4 + 1; c2, c3 = row g8 + 8) of the 4 n tiles of
// an output group are again the 8 contiguous channels of the lane's chunk: the output of one product is laid out as the input of the next.
#pragma once
#include "adaln_mma.cuh"

namespace adl {

// the value an m16n8k8 tf32 operand holds: round to nearest (ties away), low 13 bits cleared
ADL_DEVI uint32_t to_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}

ADL_DEVI void mma1688(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// shared-memory float vectors (the weights are read-only after the staging barrier)
ADL_DEVI float4 lds_f4(uint32_t a) {
  float4 v;
  asm("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
ADL_DEVI void sts_f4(uint32_t a, float4 v) {
  asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
ADL_DEVI void stg_f4(float* p, float4 v) {
  asm volatile("st.global.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// byte offset of 16-byte chunk `chunk` (0 .. 31) of fp32 weight row `row` (rows of 512 B; the chunk index is XOR-ed with 1 on odd rows: the two rows g8 = 2 i, 2 i + 1 a quarter warp reads
// fall on disjoint bank groups: chunks 8 gk + 2 q4 (+ 1) of an even row are the even / odd chunks of an odd row)
ADL_DEVI uint32_t tf_woff(uint32_t row, uint32_t chunk) { return row * 512u + ((chunk ^ (row & 1u)) << 4); }

// a 128 x 128 fp32 weight matrix W [out][in] into shared memory: rows in the f1 order, values rounded to TF32 (the MMA would truncate the low 13 bits: round to nearest like the cuBLAS / Triton TF32 paths).
// The 8 floats a lane reads of a row for a group of 32 inputs (elements e = 0 .. 7 of its chunk) are stored as the pairs (e0, e4) (e1, e5) | (e2, e6) (e3, e7): one LDS.128 is (b0, b1) of TWO consecutive k
// steps, so the kernels can interleave the MMAs of several n tiles per step (the accumulation of one tile is a dependent chain of 16 MMAs: a pair of independent accumulators is not enough to cover its latency).
ADL_DEVI void load_weight128_tf32(uint32_t dst, const float* w, int tid, int nthr) {
  for (int i = tid; i < 128 * 16; i += nthr) {
    const int row = i >> 4, jj = i & 15;
    const float* src = w + (size_t)qf_channel(row) * 128 + 8 * jj;
    float4 a = ldg_f4(src), b = ldg_f4(src + 4);
    a.x = __uint_as_float(to_tf32(a.x)); a.y = __uint_as_float(to_tf32(a.y)); a.z = __uint_as_float(to_tf32(a.z)); a.w = __uint_as_float(to_tf32(a.w));
    b.x = __uint_as_float(to_tf32(b.x)); b.y = __uint_as_float(to_tf32(b.y)); b.z = __uint_as_float(to_tf32(b.z)); b.w = __uint_as_float(to_tf32(b.w));
    sts_f4(dst + tf_woff(row, 2 * jj), make_float4(a.x, b.x, a.y, b.y));
    sts_f4(dst + tf_woff(row, 2 * jj + 1), make_float4(a.z, b.z, a.w, b.w));
  }
}

}  // namespace adl
