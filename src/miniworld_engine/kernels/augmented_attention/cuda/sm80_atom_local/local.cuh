// local.cuh — shared helpers of the AF3 windowed atom attention kernels (sm_100a, mma.sync m16n8k16 bf16).
//
// AF3 atom attention (Alg. 24 with the 32 x 128 trunking): atom i of query window w = i / 32 attends to the 128 atoms
// [32 w - 48, 32 w + 80) (clipped to [0, N)), with a per-window pair bias bias[h][w][i % 32][j - (32 w - 48)].
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__
constexpr int NH = 4, DH = 32, DM = 128, WQ = 32, WK = 128, KOFF = 48;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.17677669529663687f;           // 1 / sqrt(32)

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {   // zero-fills the 16 bytes when !pred
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

DEVI void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// A [rows][32 bf16] tile is 64-byte rows = 4 chunks of 16 bytes; the chunk is XORed with (row >> 1) & 3 (conflict-free ldmatrix).
DEVI uint32_t tile_addr(uint32_t base, int row, int chunk) { return base + row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

// A operand (16 rows from row0, k = 32 as two k-steps) <- tile[row][k]
DEVI void load_a(uint32_t a[2][4], uint32_t tile, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 2; ++ks)
    ldsm4(a[ks][0], a[ks][1], a[ks][2], a[ks][3], tile_addr(tile, row0 + (lane & 7) + ((lane >> 3) & 1) * 8, ks * 2 + (lane >> 4)));
}
// B operand "col" (B[k = d][n = row], storage tile[row][d]) for the n8 tiles 2 np, 2 np + 1 and k-step ks
DEVI void load_b_rows(uint32_t& b0, uint32_t& b1, uint32_t& b2, uint32_t& b3, uint32_t tile, int n0, int ks, int lane) {
  const int mi = lane >> 3;
  ldsm4(b0, b1, b2, b3, tile_addr(tile, n0 + (lane & 7) + (mi >> 1) * 8, ks * 2 + (mi & 1)));
}
// B operand (B[k = row][n = d], storage tile[row][d]) for the d tiles 2 dp, 2 dp + 1 and the 16 rows from k0
DEVI void load_b_cols(uint32_t& b0, uint32_t& b1, uint32_t& b2, uint32_t& b3, uint32_t tile, int k0, int dp, int lane) {
  const int mi = lane >> 3;
  ldsm4t(b0, b1, b2, b3, tile_addr(tile, k0 + (lane & 7) + (mi & 1) * 8, dp * 2 + (mi >> 1)));
}
DEVI void cp_async4(uint32_t dst, const void* src, bool pred) {   // 4-byte copy, zero-filled when !pred
  const int sz = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(sz) : "memory");
}
