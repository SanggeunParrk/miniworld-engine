// var_pack.cu — the B200 variant block's weight packs in ONE launch (2026-11-17; sm_100a, no tensor cores): the row kernels' stacked bf16
// operands, the fp32 vector pack and, for training, the backward's transposed bf16 A operands, straight from the parameters (fp32 master or
// bf16; a per-source dtype mask). It replaces a torch.cat per pack plus a cast (and one more cat for the transposes): the packs are part of
// every captured training step (a capture records them: the weights change between replays).
//
//   M[15]: Wsc1 Wbi1 Wos Wsc2 Wbi2 Wts | Wq Wk Wv Wg | Wo | Wa Wb | Ws | Wup   (row-major; [128][128] each, Wa / Wb / Wup [256][128],
//          Ws [128][256]; Wup may be null)
//   BF  = their flat concatenation as bf16: Wmod [768][128] at row 0, Wp [512][128] at 768, Wo at 1280, Wu = [Wa; Wb] [512][128] at 1408,
//         Ws [128][256] at 1920, Wup [256][128] at 2176 (rows of 128 elements)
//   BT  (do_t) = the transposes Ws^T [256][128] at row 0, Wu^T [128][512] at 256, Wo^T [128][128] at 768, Wp^T [128][512] at 896,
//         Wmod^T [128][768] at 1408 (, Wup^T [128][256] at 2176)
//   VEC = nvec fp32 rows of 128 from V[13] (a null source: zeros)
// mmask / vmask: bit i set = source i is bf16 (else fp32).
// The sources are [128][128] units (17, 19 with Wup); a CTA takes a 16-column strip of one unit (8 CTAs per unit, about one per SM): the
// strip through shared memory as bf16, out row-major (BF) and transposed (BT: 16 rows of X^T), 16-byte accesses. The last CTA writes VEC.
// Measured in a captured training step at 1024 atoms (the L2 flushed before each replay, so the weights come from DRAM): one CTA per whole
// unit took 18 - 22 us a pack, the strips 6 - 7 us (var_pack32: 13 -> 4 - 5.5 us).
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

static __device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  const __nv_bfloat16 a = __float2bfloat16_rn(lo), b = __float2bfloat16_rn(hi);
  return (uint32_t)(*reinterpret_cast<const uint16_t*>(&a)) | ((uint32_t)(*reinterpret_cast<const uint16_t*>(&b)) << 16);
}

extern "C" __global__ void __launch_bounds__(256)
atom_var_pack(const void* m0, const void* m1, const void* m2, const void* m3, const void* m4, const void* m5, const void* m6, const void* m7,
              const void* m8, const void* m9, const void* m10, const void* m11, const void* m12, const void* m13, const void* m14,
              const void* v0, const void* v1, const void* v2, const void* v3, const void* v4, const void* v5, const void* v6, const void* v7,
              const void* v8, const void* v9, const void* v10, const void* v11, const void* v12,
              uint16_t* __restrict__ BF, float* __restrict__ VEC, uint16_t* __restrict__ BT, uint32_t mmask, uint32_t vmask, int nvec, int do_t) {
#define MSEL(s) ((s) == 0 ? m0 : (s) == 1 ? m1 : (s) == 2 ? m2 : (s) == 3 ? m3 : (s) == 4 ? m4 : (s) == 5 ? m5 : (s) == 6 ? m6 : (s) == 7 ? m7 : \
                 (s) == 8 ? m8 : (s) == 9 ? m9 : (s) == 10 ? m10 : (s) == 11 ? m11 : (s) == 12 ? m12 : (s) == 13 ? m13 : m14)
#define VSEL(s) ((s) == 0 ? v0 : (s) == 1 ? v1 : (s) == 2 ? v2 : (s) == 3 ? v3 : (s) == 4 ? v4 : (s) == 5 ? v5 : (s) == 6 ? v6 : (s) == 7 ? v7 : \
                 (s) == 8 ? v8 : (s) == 9 ? v9 : (s) == 10 ? v10 : (s) == 11 ? v11 : v12)
  const int nunit = m14 != nullptr ? 19 : 17, tid = threadIdx.x;
  if ((int)blockIdx.x >= 8 * nunit) {                                      // the vector pack
    for (int t = tid * 4; t < nvec * 128; t += 1024) {
      const int s = t >> 7, c = t & 127;
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      const void* const vs = VSEL(s);
      if (vs != nullptr) {
        if ((vmask >> s) & 1u) {
          const uint2 h = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(vs) + c));
          x = make_float4(__uint_as_float((h.x & 0xffffu) << 16), __uint_as_float(h.x & 0xffff0000u), __uint_as_float((h.y & 0xffffu) << 16),
                          __uint_as_float(h.y & 0xffff0000u));
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
  else if (u < 15)  { const int k = u - 11; s = 11 + (k >> 1); soff = (k & 1) * 16384; fo = (1408 + 128 * k) * 128; to = 256 * 128 + 128 * k; tp = 512; }
  else if (u < 17)  { const int c = u - 15; s = 13; soff = 128 * c; sp = 256; fo = 1920 * 128 + 128 * c; fp = 256; to = 128 * c * 128; tp = 128; }
  else              { const int k = u - 17; s = 14; soff = k * 16384; fo = (2176 + 128 * k) * 128; to = 2176 * 128 + 128 * k; tp = 256; }
  __shared__ __align__(16) uint16_t tile[128][16];
  const void* const ms = MSEL(s);
  const int r = tid & 127, h8 = 8 * (tid >> 7);                            // this thread: row r, columns c16 + h8 .. + 7
  uint4 o;
  if ((mmask >> s) & 1u) {
    o = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(ms) + soff + r * sp + c16 + h8));
  } else {
    const float* p = reinterpret_cast<const float*>(ms) + soff + r * sp + c16 + h8;
    const float4 x0 = __ldg(reinterpret_cast<const float4*>(p)), x1 = __ldg(reinterpret_cast<const float4*>(p + 4));
    o = make_uint4(f2bf2(x0.x, x0.y), f2bf2(x0.z, x0.w), f2bf2(x1.x, x1.y), f2bf2(x1.z, x1.w));
  }
  *reinterpret_cast<uint4*>(&tile[r][h8]) = o;
  *reinterpret_cast<uint4*>(BF + fo + r * fp + c16 + h8) = o;
  if (!do_t) return;
  __syncthreads();
  const int j = tid >> 4, r8 = 8 * (tid & 15);                             // X^T row c16 + j, columns r8 .. + 7 = the strip's column j, rows r8 ..
  *reinterpret_cast<uint4*>(BT + to + (c16 + j) * tp + r8) =
      make_uint4((uint32_t)tile[r8][j] | ((uint32_t)tile[r8 + 1][j] << 16), (uint32_t)tile[r8 + 2][j] | ((uint32_t)tile[r8 + 3][j] << 16),
                 (uint32_t)tile[r8 + 4][j] | ((uint32_t)tile[r8 + 5][j] << 16), (uint32_t)tile[r8 + 6][j] | ((uint32_t)tile[r8 + 7][j] << 16));
}
