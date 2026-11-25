// pack.cuh -- the weights of the three-layer chain in the shared-memory image the kernels copy straight into smem (one launch each, any parameter dtype).
//
//   forward image   layer l = 0 (edge block of the packed W1), 1 (W2), 2 (W3):  tile[r][c] = W_l[perm(r)][c]     r = packed (output) row, c = logical input channel
//   backward image  layer l = 0 (dX3), 1 (dX2), 2 (dX1):                       tile[r][c] = Wsrc_l[c][perm(r)]  the transposed weight (Wsrc = W3, W2, W1e), rows in the f1 order
//   Each tile is 128 rows of 256 B stored with the ``woff`` chunk swizzle; elements are rounded to bf16 (fp32 weights come from the autocast contract).
//   tab = [b2 | b3 | gamma | beta] fp32 (the two biases rounded through bf16 first, as the Triton path adds them), natural channel order.
#pragma once
#include "common.cuh"

namespace me80 {

struct PackParams {
  const void* w[3];          // forward: W1e (a slice of the packed projection: leading dimension w_ld[0]), W2, W3;  backward: W3, W2, W1e
  int w_ld[3];
  int w_fp32[3];             // 1: the matrix is fp32
  int tr[3];                 // 1: store the transposed matrix (the backward image), 0: the matrix itself (the forward image)
  const void* vec[4];        // b2, b3, gamma, beta
  int vec_fp32[4];
  uint8_t* img;              // 3 x 32 KiB
  float* tab;                // 512 floats
};

DEVI void load8(const void* p, int fp32, float (&v)[8]) {
  if (fp32) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    v[0] = bf16lo(r.x); v[1] = bf16hi(r.x); v[2] = bf16lo(r.y); v[3] = bf16hi(r.y); v[4] = bf16lo(r.z); v[5] = bf16hi(r.z); v[6] = bf16lo(r.w); v[7] = bf16hi(r.w);
  }
}
DEVI float load1(const void* base, int fp32, size_t idx) {
  return fp32 ? reinterpret_cast<const float*>(base)[idx] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[idx]);
}

__global__ void __launch_bounds__(256) pack_kernel(const PackParams p) {
  const int i = blockIdx.x * 256 + threadIdx.x;       // one 16-byte chunk of the image
  if (i < 3 * 2048) {
    const int l = i >> 11, rem = i & 2047, r = rem >> 4, c = rem & 15;
    float v[8];
    if (!p.tr[l]) {
      load8(reinterpret_cast<const char*>(p.w[l]) + ((size_t)perm(r) * p.w_ld[l] + 8 * c) * (p.w_fp32[l] ? 4 : 2), p.w_fp32[l], v);
    } else {
      const int col = perm(r);
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = load1(p.w[l], p.w_fp32[l], (size_t)(8 * c + j) * p.w_ld[l] + col);
    }
    uint4 o;
    o.x = pack_bf16(v[0], v[1]); o.y = pack_bf16(v[2], v[3]); o.z = pack_bf16(v[4], v[5]); o.w = pack_bf16(v[6], v[7]);
    *reinterpret_cast<uint4*>(p.img + l * LAYER_BYTES + woff(r, c)) = o;
  }
  if (i < TAB_FLOATS) {
    const int v = i >> 7, k = i & 127;
    float x = load1(p.vec[v], p.vec_fp32[v], k);
    if (v < 2) x = round_bf16f(x);                    // the biases enter the accumulator as bf16
    p.tab[i] = x;
  }
}

}  // namespace me80
