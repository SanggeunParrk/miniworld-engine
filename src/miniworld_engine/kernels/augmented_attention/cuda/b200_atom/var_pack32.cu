// var_pack32.cu — the B200 variant block's fp32 (TF32) TRAINING weight packs in ONE launch (2026-11-17; sm_100a, no tensor cores): var_pack.cu
// with fp32 outputs on the TF32 grid (cvt.rna: a kind::tf32 MMA truncates its operands). It replaces the ~12 launches of the torch pack
// (a cat, the rounding, a copy, a transpose per operand) that every captured training step repeats.
//
//   M[15]: Wsc1 Wbi1 Wsc2 Wbi2 Wos Wts | Wq Wk Wv Wg | Wo | Wa Wb | Ws | Wup   (the fp32 path's order: local_dit_b200._pack32)
//   BF  = Wmod [768][128] at row 0, Wp [512][128] at 768, Wo at 1280, Wu = [Wa_0; Wb_0; Wa_1; Wb_1] [512][128] at 1408 (the SwiGLU GEMM's
//         interleaved 128-row halves), Ws [128][256] at 1920, Wup [256][128] at 2176 (rows of 128 elements)
//   BT  = the transposes Ws^T [256][128] at row 0, Wu^T [128][512] at 256, Wo^T at 768, Wp^T [128][512] at 896, Wmod^T [128][768] at 1408
//         (, Wup^T [128][256] at 2176)
//   BG  = the conditioning backward's grouped operand [384][256]: row block g = Wmod^T[:, 256 g : 256 g + 256]
//   BU  (with Wup) = [Wu^T | Wup^T] [128][768]: the operand of Boltz-2's one dx2 GEMM of [d a | d b | d up]
//   VEC = nvec fp32 rows of 128 from V[16] (a null source: zeros), not rounded
// mmask / vmask: bit i set = source i is bf16 (else fp32). A CTA takes a 16-column strip of one [128][128] unit of the sources (8 CTAs per
// unit, as var_pack.cu), the last CTA writes VEC.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

static __device__ __forceinline__ float rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
static __device__ __forceinline__ float bf2f(uint32_t h) { return __uint_as_float(h << 16); }

extern "C" __global__ void __launch_bounds__(256)
atom_var_pack32(const void* m0, const void* m1, const void* m2, const void* m3, const void* m4, const void* m5, const void* m6, const void* m7,
                const void* m8, const void* m9, const void* m10, const void* m11, const void* m12, const void* m13, const void* m14,
                const void* v0, const void* v1, const void* v2, const void* v3, const void* v4, const void* v5, const void* v6, const void* v7,
                const void* v8, const void* v9, const void* v10, const void* v11, const void* v12, const void* v13, const void* v14,
                const void* v15, float* __restrict__ BF, float* __restrict__ VEC, float* __restrict__ BT, float* __restrict__ BG, uint32_t mmask,
                uint32_t vmask, int nvec, float* __restrict__ BU) {
#define MSEL(s) ((s) == 0 ? m0 : (s) == 1 ? m1 : (s) == 2 ? m2 : (s) == 3 ? m3 : (s) == 4 ? m4 : (s) == 5 ? m5 : (s) == 6 ? m6 : (s) == 7 ? m7 : \
                 (s) == 8 ? m8 : (s) == 9 ? m9 : (s) == 10 ? m10 : (s) == 11 ? m11 : (s) == 12 ? m12 : (s) == 13 ? m13 : m14)
#define VSEL(s) ((s) == 0 ? v0 : (s) == 1 ? v1 : (s) == 2 ? v2 : (s) == 3 ? v3 : (s) == 4 ? v4 : (s) == 5 ? v5 : (s) == 6 ? v6 : (s) == 7 ? v7 : \
                 (s) == 8 ? v8 : (s) == 9 ? v9 : (s) == 10 ? v10 : (s) == 11 ? v11 : (s) == 12 ? v12 : (s) == 13 ? v13 : (s) == 14 ? v14 : v15)
  const int nunit = m14 != nullptr ? 19 : 17, tid = threadIdx.x;
  if ((int)blockIdx.x >= 8 * nunit) {                                      // the vector pack
    for (int t = tid * 4; t < nvec * 128; t += 1024) {
      const int s = t >> 7, c = t & 127;
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      const void* const vs = VSEL(s);
      if (vs != nullptr) {
        if ((vmask >> s) & 1u) {
          const uint2 h = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(vs) + c));
          x = make_float4(bf2f(h.x & 0xffffu), bf2f(h.x >> 16), bf2f(h.y & 0xffffu), bf2f(h.y >> 16));
        } else {
          x = __ldg(reinterpret_cast<const float4*>(reinterpret_cast<const float*>(vs) + c));
        }
      }
      *reinterpret_cast<float4*>(VEC + t) = x;
    }
    return;
  }
  const int u = (int)blockIdx.x >> 3, c16 = 16 * ((int)blockIdx.x & 7);   // the unit and this CTA's 16-column strip of it
  // unit u: source s, its first element soff and row pitch sp; BF at fo, row pitch fp; BT at to, row pitch tp (elements)
  int s, soff = 0, sp = 128, fo, fp = 128, to, tp;
  if (u < 6)        { s = u;                                    fo = 128 * u * 128;          to = 1408 * 128 + 128 * u;       tp = 768; }
  else if (u < 10)  { s = u;                                    fo = (768 + 128 * (u - 6)) * 128;  to = 896 * 128 + 128 * (u - 6);  tp = 512; }
  else if (u == 10) { s = 10;                                   fo = 1280 * 128;             to = 768 * 128;                  tp = 128; }
  else if (u < 15)  { const int k = u - 11; s = 11 + (k & 1); soff = (k >> 1) * 16384; fo = (1408 + 128 * k) * 128; to = 256 * 128 + 128 * k; tp = 512; }
  else if (u < 17)  { const int c = u - 15; s = 13; soff = 128 * c; sp = 256; fo = 1920 * 128 + 128 * c; fp = 256; to = 128 * c * 128; tp = 128; }
  else              { const int k = u - 17; s = 14; soff = k * 16384; fo = (2176 + 128 * k) * 128; to = 2176 * 128 + 128 * k; tp = 256; }
  __shared__ __align__(16) float tile[128][16];
  const void* const ms = MSEL(s);
  const int r = tid & 127, h8 = 8 * (tid >> 7);                            // this thread: row r, columns c16 + h8 .. + 7
  float4 x0, x1;
  if ((mmask >> s) & 1u) {
    const uint4 h = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(ms) + soff + r * sp + c16 + h8));
    x0 = make_float4(bf2f(h.x & 0xffffu), bf2f(h.x >> 16), bf2f(h.y & 0xffffu), bf2f(h.y >> 16));
    x1 = make_float4(bf2f(h.z & 0xffffu), bf2f(h.z >> 16), bf2f(h.w & 0xffffu), bf2f(h.w >> 16));
  } else {
    const float* p = reinterpret_cast<const float*>(ms) + soff + r * sp + c16 + h8;
    x0 = __ldg(reinterpret_cast<const float4*>(p));
    x1 = __ldg(reinterpret_cast<const float4*>(p + 4));
    x0 = make_float4(rna_tf32(x0.x), rna_tf32(x0.y), rna_tf32(x0.z), rna_tf32(x0.w));
    x1 = make_float4(rna_tf32(x1.x), rna_tf32(x1.y), rna_tf32(x1.z), rna_tf32(x1.w));
  }
  *reinterpret_cast<float4*>(&tile[r][h8]) = x0;
  *reinterpret_cast<float4*>(&tile[r][h8 + 4]) = x1;
  *reinterpret_cast<float4*>(BF + fo + r * fp + c16 + h8) = x0;
  *reinterpret_cast<float4*>(BF + fo + r * fp + c16 + h8 + 4) = x1;
  __syncthreads();
  const int j = tid >> 4, r8 = 8 * (tid & 15);                             // X^T row c16 + j, columns r8 .. + 7 = the strip's column j, rows r8 ..
  const float4 y0 = make_float4(tile[r8][j], tile[r8 + 1][j], tile[r8 + 2][j], tile[r8 + 3][j]);
  const float4 y1 = make_float4(tile[r8 + 4][j], tile[r8 + 5][j], tile[r8 + 6][j], tile[r8 + 7][j]);
  float* bt = BT + to + (c16 + j) * tp + r8;
  *reinterpret_cast<float4*>(bt) = y0;
  *reinterpret_cast<float4*>(bt + 4) = y1;
  if (u < 6) {                                                             // BG: row block u / 2, column half u % 2
    float* bg = BG + (128 * (u >> 1) + c16 + j) * 256 + 128 * (u & 1) + r8;
    *reinterpret_cast<float4*>(bg) = y0;
    *reinterpret_cast<float4*>(bg + 4) = y1;
  }
  if (BU != nullptr && u >= 11 && (u < 15 || u >= 17)) {                   // BU: the Wu / Wup units' column blocks
    float* bu = BU + (c16 + j) * 768 + (u >= 17 ? 512 + 128 * (u - 17) : 128 * (u - 11)) + r8;
    *reinterpret_cast<float4*>(bu) = y0;
    *reinterpret_cast<float4*>(bu + 4) = y1;
  }
}
