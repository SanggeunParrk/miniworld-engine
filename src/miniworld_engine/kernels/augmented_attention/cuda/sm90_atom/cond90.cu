// cond90.cu — the conditioning tables of the AF3 atom block on sm_90a (wgmma, register A operand; bf16 or -DTF32=1).
//
//   n = LN(c)                         (no affine; the three LN weights are folded into the weights: W' = W diag(gamma))
//   mod = [ s1 | bi1 | so | s2 | bi2 | st ]   [M, 768]    s1 = sigmoid(n W1s' + b), bi1 = n W1b', s2 / bi2 likewise from n W2*',
//                                                       so = sigmoid(c Wso + b), st = sigmoid(c Wst + b)
//   mkv = [ pre | shift ]            [M, 256]    n Wkv'  (the K / V AdaLN's modulation; its sigmoid and bias are applied by pre90)
// One CTA = 2 warpgroups = 128 rows; the eight 128 x 128 weights stream through shared memory in 64-row units (swizzled K-major tiles).
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;

// matrix p = [Ws1', Wb1', Ws2', Wb2', Wkv_pre', Wkv_shift', Wso, Wst]: output column base (mod unless p is 4 / 5) and bias index
__device__ constexpr int OUT_COL[8] = {0, 128, 384, 512, 0, 128, 256, 640};
__device__ constexpr int BIAS_IX[8] = {0, -1, 1, -1, -1, -1, 2, 3};

extern "C" __global__ __launch_bounds__(NTHR, MINB) void cond90_fwd(const T* __restrict__ c, const T* __restrict__ w, const float* __restrict__ bias,
                                                                T* __restrict__ mod, T* __restrict__ mkv, int M, float eps) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
#ifndef RING
#define RING 2
#endif
  auto issue = [&](int u) { load_w(base + (u % RING) * UNIT, w + (size_t)u * 64 * 128, 128, 64, 128, tid, NTHR); cp_commit(); };
#pragma unroll
  for (int r = 0; r < RING; ++r) issue(r);
  Tile x;
  ld_tile(x, c + row0 * 128, 128, 0, g, t);
  float m0, r0, m1, r1;
  ln_stats(x, eps, m0, r0, m1, r1);
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    x.f[nt][0] = (x.f[nt][0] - m0) * r0; x.f[nt][1] = (x.f[nt][1] - m0) * r0;
    x.f[nt][2] = (x.f[nt][2] - m1) * r1; x.f[nt][3] = (x.f[nt][3] - m1) * r1;
  }
  rna(x);
#if !TF32
#pragma unroll   // bf16: specialised units (335 -> 278 us); the fp32 build spills when unrolled (433 -> 1095 us)
#endif
  for (int u = 0; u < 16; ++u) {
    if (u == 12) { ld_tile(x, c + row0 * 128, 128, 0, g, t); rna(x); }  // the last two matrices read the raw conditioning
    cp_wait<RING - 1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm<64, KSTEPS>(acc, x, base + (u % RING) * UNIT, 0);
    __syncthreads();
    if (u + RING < 16) issue(u + RING); else cp_commit();
    const int p = u >> 1, half = u & 1, bi = BIAS_IX[p];
    T* dst = (p == 4 || p == 5) ? mkv : mod;
    const int ld = (p == 4 || p == 5) ? 256 : 768, col = OUT_COL[p] + 64 * half;
    float v[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) v[nt][e] = acc[4 * nt + e];
      if (bi >= 0) {
        const float2 b = colv(bias + bi * 128 + 64 * half, nt, t);
        v[nt][0] = sigm(v[nt][0] + b.x); v[nt][1] = sigm(v[nt][1] + b.y); v[nt][2] = sigm(v[nt][2] + b.x); v[nt][3] = sigm(v[nt][3] + b.y);
      }
    }
    st_nt<8, false>(v, dst + row0 * ld, ld, col, g, t);
  }
}

// Small row counts: grid.y splits the 16 weight units over CTAs (units u0 .. u0 + 16 / gridDim.y - 1 of this CTA; the products are independent),
// so a few 64-row tiles occupy many SMs instead of one SM walking all 16 units.
extern "C" __global__ __launch_bounds__(NTHR, 1) void cond90_split(const T* __restrict__ c, const T* __restrict__ w, const float* __restrict__ bias,
                                                                T* __restrict__ mod, T* __restrict__ mkv, int M, float eps) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  const int upc = 16 / (int)gridDim.y, u0 = (int)blockIdx.y * upc;
  auto issue = [&](int ul) { load_w(base + (ul & 1) * UNIT, w + (size_t)(u0 + ul) * 64 * 128, 128, 64, 128, tid, NTHR); cp_commit(); };
  issue(0);
  if (upc > 1) issue(1); else cp_commit();
  Tile x;
  ld_tile(x, c + row0 * 128, 128, 0, g, t);
  if (u0 < 12) {  // matrices 0 .. 5 read LN(c), 6 and 7 the raw conditioning
    float m0, r0, m1, r1;
    ln_stats(x, eps, m0, r0, m1, r1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      x.f[nt][0] = (x.f[nt][0] - m0) * r0; x.f[nt][1] = (x.f[nt][1] - m0) * r0;
      x.f[nt][2] = (x.f[nt][2] - m1) * r1; x.f[nt][3] = (x.f[nt][3] - m1) * r1;
    }
  }
  rna(x);
#pragma unroll 1
  for (int ul = 0; ul < upc; ++ul) {
    const int u = u0 + ul;
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm<64, KSTEPS>(acc, x, base + (ul & 1) * UNIT, 0);
    __syncthreads();
    if (ul + 2 < upc) issue(ul + 2); else cp_commit();
    const int p = u >> 1, half = u & 1;
    const int bi = p == 0 ? 0 : p == 2 ? 1 : p == 6 ? 2 : p == 7 ? 3 : -1;
    const int oc = p == 0 ? 0 : p == 1 ? 128 : p == 2 ? 384 : p == 3 ? 512 : p == 4 ? 0 : p == 5 ? 128 : p == 6 ? 256 : 640;
    T* dst = (p == 4 || p == 5) ? mkv : mod;
    const int ld = (p == 4 || p == 5) ? 256 : 768, col = oc + 64 * half;
    float v[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) v[nt][e] = acc[4 * nt + e];
      if (bi >= 0) {
        const float2 b = colv(bias + bi * 128 + 64 * half, nt, t);
        v[nt][0] = sigm(v[nt][0] + b.x); v[nt][1] = sigm(v[nt][1] + b.y); v[nt][2] = sigm(v[nt][2] + b.x); v[nt][3] = sigm(v[nt][3] + b.y);
      }
    }
    st_nt<8, false>(v, dst + row0 * ld, ld, col, g, t);
  }
}
