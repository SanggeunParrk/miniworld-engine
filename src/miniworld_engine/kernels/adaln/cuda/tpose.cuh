// tpose.cuh -- feature-major (transposed) stores of 32-row tiles, the layout the TF32 weight-gradient GEMMs need.
//
// A TF32 wgmma GEMM reads both operands K-major. The weight gradient dW = G^T B contracts over the ROWS (M) of the activations, so a row-major G [M, P]
// is "MN-major" for it and cuBLAS runs its slower kernel (measured at M = 18432, d = 768: G^T B 430 us with G row-major, 295 us with G^T written
// feature-major [P, M]; 237 -> 145 and 246 -> 151 for the other two weight gradients of the ConditionedTransition / AdaLN step). The row kernels that
// produce the gradients therefore write them feature-major: a block collects a tile of TR = 32 rows x P features in shared memory
// (T[feature][row], row stride 33, rows rotated per feature so that the 32 lanes of a warp hit 32 banks, both on the write and on the read side)
// and then writes 32 consecutive rows (128 B) of every feature row of the output.
#pragma once
#include <cstdint>

namespace tpose {
constexpr int TR = 32;   // rows of a tile
constexpr int LD = 33;   // shared-memory row stride (floats)
// the rotation of the 32 row slots of feature f: lane L of a warp that holds the chunk of four features 4 (L + 32 k) .. + 3 for ONE row writes
// banks (f + L-dependent rot + row) mod 32 = 4 (L & 7) + (L >> 3) + const: 32 distinct banks
__device__ __forceinline__ int rot(int f) { return (f >> 5) & 3; }
__device__ __forceinline__ void put(float* T, int f, int r, float v) { T[f * LD + ((r + rot(f)) & 31)] = v; }
// flush the nf features of T (feature f of T goes to row d0 + f of D, columns m0 .. m0 + rows - 1) by one warp group of nwarps warps; the lane is the tile row
__device__ __forceinline__ void flush(const float* T, float* __restrict__ D, int64_t ldd, int d0, int nf, int m0, int rows, int warp, int nwarps, int lane) {
  const bool ok = lane < rows;
#pragma unroll 4
  for (int f = warp; f < nf; f += nwarps) {
    const float v = T[f * LD + ((lane + rot(f)) & 31)];
    if (ok) D[(int64_t)(d0 + f) * ldd + m0 + lane] = v;
  }
}
}  // namespace tpose
