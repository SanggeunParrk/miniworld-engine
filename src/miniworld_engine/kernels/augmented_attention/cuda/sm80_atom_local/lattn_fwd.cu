// lattn_fwd.cu — AF3 windowed atom attention forward (32 queries x 128 keys per window), sm_100a, mma.sync.
//
//   S = q k^T / sqrt(32) + bias[h][w]      O = softmax(S) v      (bf16 O [A N, 128], row LSE [A, 4, N] in log2 units)
//
// One CTA = (query window w, head h, a chunk of the samples); 4 warps = 2 samples x 2 query tiles of 16 rows; the 128 keys of a window
// fit one softmax (no online rescaling). The window's bias [32][128] (fp32) stays in shared memory for all the CTA's samples; q / K / V
// tiles of the next sample pair are prefetched with cp.async. Keys outside [0, N) and keys with kmask == 0 get -inf.
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"

constexpr int BS = 136;                                           // bias row stride (floats): conflict-free float2 reads
constexpr int TQB = WQ * 64, TKB = WK * 64;                       // q tile, K / V tile bytes
constexpr int SLOT = TQB + 2 * TKB;                               // one sample: q | K | V
constexpr int O_BIAS = 0, O_KADD = WQ * BS * 4, O_ST = O_KADD + WK * 4;
constexpr int SMEM_BYTES = O_ST + 2 * 2 * SLOT;                   // 2 stages x 2 samples

extern "C" __global__ void __launch_bounds__(128)
local_attn_fwd(const __nv_bfloat16* __restrict__ gq, const __nv_bfloat16* __restrict__ gk, const __nv_bfloat16* __restrict__ gv,
               const float* __restrict__ gbias, const uint8_t* __restrict__ kmask, __nv_bfloat16* __restrict__ gO,
               float* __restrict__ LSE, int N, int A, int nwin, int psplit) {
  extern __shared__ __align__(1024) uint8_t sm[];
  float* sbias = reinterpret_cast<float*>(sm + O_BIAS);
  float* kadd = reinterpret_cast<float*>(sm + O_KADD);
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int w = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
  const int npair = (A + 1) >> 1;
  const int kbase = WQ * w - KOFF;

  // bias [32][128] of this (head, window) -> smem (fp32), key additive term -> smem (log2 domain: 0 or -inf)
  {
    const float* src = gbias + ((size_t)h * nwin + w) * (WQ * WK);
    for (int i = tid; i < WQ * WK / 4; i += 128) {
      const int r = i >> 5, c = (i & 31) * 4;
      *reinterpret_cast<float4*>(sbias + r * BS + c) = *reinterpret_cast<const float4*>(src + i * 4);
    }
    for (int j = tid; j < WK; j += 128) {
      const int key = kbase + j;
      kadd[j] = (key >= 0 && key < N && (kmask == nullptr || kmask[key])) ? 0.f : -INFINITY;
    }
  }

  auto issue = [&](int stage, int p) {                            // sample pair p -> stage
    const uint32_t st = su + O_ST + stage * 2 * SLOT;
    for (int idx = tid; idx < 2 * (WQ * 4 + 2 * WK * 4); idx += 128) {
      const int slot = idx / (WQ * 4 + 2 * WK * 4), rem = idx - slot * (WQ * 4 + 2 * WK * 4);
      const int a = 2 * p + slot;
      const bool ok_a = (p < npair) && a < A;
      const uint32_t sbase = st + slot * SLOT;
      if (rem < WQ * 4) {
        const int row = rem >> 2, ch = rem & 3, q = WQ * w + row;
        const bool ok = ok_a && q < N;
        cp_async16(tile_addr(sbase, row, ch), gq + ((size_t)a * N + (ok ? q : 0)) * DM + h * DH + ch * 8, ok);
      } else {
        const int r2 = rem - WQ * 4, which = r2 >= WK * 4, rr = which ? r2 - WK * 4 : r2;
        const int row = rr >> 2, ch = rr & 3, key = kbase + row;
        const bool ok = ok_a && key >= 0 && key < N;
        const __nv_bfloat16* src = (which ? gv : gk) + ((size_t)a * N + (ok ? key : 0)) * DM + h * DH + ch * 8;
        cp_async16(tile_addr(sbase + TQB + which * TKB, row, ch), src, ok);
      }
    }
    cp_commit();
  };

  int p = z;
  issue(0, p);
  __syncthreads();
  const int mt = warp & 1, sl = warp >> 1;
  const int r0 = lane >> 2, c0 = 2 * (lane & 3);
  for (int stage = 0; p < npair; p += psplit, stage ^= 1) {
    if (p + psplit < npair) { issue(stage ^ 1, p + psplit); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const int a = 2 * p + sl;
    if (a < A) {
      const uint32_t sbase = su + O_ST + stage * 2 * SLOT + sl * SLOT, tq = sbase, tk = sbase + TQB, tv = sbase + TQB + TKB;
      uint32_t qa[2][4];
      load_a(qa, tq, 16 * mt, lane);
      float s[16][4];
#pragma unroll
      for (int i = 0; i < 16; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
      for (int np = 0; np < 8; ++np)
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t b0, b1, b2, b3;
          load_b_rows(b0, b1, b2, b3, tk, 16 * np, ks, lane);
          mma16816(s[2 * np], qa[ks], b0, b1);
          mma16816(s[2 * np + 1], qa[ks], b2, b3);
        }
      // scores in log2 units, row maxima over the quad
      float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const int c = 8 * nt + c0;
        const float2 ka = *reinterpret_cast<const float2*>(kadd + c);
        const float2 b_lo = *reinterpret_cast<const float2*>(sbias + (16 * mt + r0) * BS + c);
        const float2 b_hi = *reinterpret_cast<const float2*>(sbias + (16 * mt + r0 + 8) * BS + c);
        s[nt][0] = fmaf(s[nt][0], QSCALE * LOG2E, b_lo.x * LOG2E + ka.x);
        s[nt][1] = fmaf(s[nt][1], QSCALE * LOG2E, b_lo.y * LOG2E + ka.y);
        s[nt][2] = fmaf(s[nt][2], QSCALE * LOG2E, b_hi.x * LOG2E + ka.x);
        s[nt][3] = fmaf(s[nt][3], QSCALE * LOG2E, b_hi.y * LOG2E + ka.y);
        m0 = fmaxf(m0, fmaxf(s[nt][0], s[nt][1]));
        m1 = fmaxf(m1, fmaxf(s[nt][2], s[nt][3]));
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
      }
      const float mm0 = m0 == -INFINITY ? 0.f : m0, mm1 = m1 == -INFINITY ? 0.f : m1;
      float l0 = 0.f, l1 = 0.f;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        s[nt][0] = ex2f(s[nt][0] - mm0);
        s[nt][1] = ex2f(s[nt][1] - mm0);
        s[nt][2] = ex2f(s[nt][2] - mm1);
        s[nt][3] = ex2f(s[nt][3] - mm1);
        l0 += s[nt][0] + s[nt][1];
        l1 += s[nt][2] + s[nt][3];
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o);
      }
      float o_acc[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) o_acc[i][0] = o_acc[i][1] = o_acc[i][2] = o_acc[i][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t pa[4] = {pack_bf16(s[2 * ks][0], s[2 * ks][1]), pack_bf16(s[2 * ks][2], s[2 * ks][3]),
                          pack_bf16(s[2 * ks + 1][0], s[2 * ks + 1][1]), pack_bf16(s[2 * ks + 1][2], s[2 * ks + 1][3])};
#pragma unroll
        for (int dp = 0; dp < 2; ++dp) {
          uint32_t b0, b1, b2, b3;
          load_b_cols(b0, b1, b2, b3, tv, 16 * ks, dp, lane);
          mma16816(o_acc[2 * dp], pa, b0, b1);
          mma16816(o_acc[2 * dp + 1], pa, b2, b3);
        }
      }
      const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
      const int q0 = WQ * w + 16 * mt + r0, q1 = q0 + 8;
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        if (q0 < N) *reinterpret_cast<uint32_t*>(gO + ((size_t)a * N + q0) * DM + h * DH + 8 * dt + c0) = pack_bf16(o_acc[dt][0] * i0, o_acc[dt][1] * i0);
        if (q1 < N) *reinterpret_cast<uint32_t*>(gO + ((size_t)a * N + q1) * DM + h * DH + 8 * dt + c0) = pack_bf16(o_acc[dt][2] * i1, o_acc[dt][3] * i1);
      }
      if ((lane & 3) == 0) {
        if (q0 < N) LSE[((size_t)a * NH + h) * N + q0] = l0 > 0.f ? mm0 + __log2f(l0) : 1e30f;
        if (q1 < N) LSE[((size_t)a * NH + h) * N + q1] = l1 > 0.f ? mm1 + __log2f(l1) : 1e30f;
      }
    }
    __syncthreads();
  }
}
