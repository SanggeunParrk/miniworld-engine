// attn_global_sm120.cuh -- the RTX 5090 (sm_120a) forward of the SWA atom DiT block's GLOBAL attention (every key j < seqused, no window), inference (no lse).
//
//   o[i] = softmax_j( scale q_i . k_j ) v_j       over 0 <= j < seqused, per (sample n, head h); a query i >= seqused writes 0
//
// Why: PyTorch's bundled FlashAttention-2 (hdim 32, 128 x 128 tiles, 4 warps) is what the composed block called for a global window; on the RTX 5090 it runs at ~85 TFLOPs
// (156 us for N 1, S 5120) and needs token-major Q / K / V copies (a transpose each).  Here: head-major [N][H][S][32] planes read as they are, an FA2-style online
// softmax over 64-key blocks on mma.sync m16n8k16 (the operand layouts of attn_fwd_sm80.cuh), and the keys SPLIT over two warps of a CTA: a CTA = 4 query groups (16 queries
// each, 64 queries) x 2 key halves (the odd / even 64-key blocks of every 128-key stage), merged through shared memory at the end.  A call of N 1, S 5120 has
// 4 x 5120 / 16 = 1280 query groups for 170 SMs (7.5 warps an SM); the split doubles the warps that hide the mma / exp latency.
#pragma once
#include "sm80_common.cuh"

namespace sw80 {

constexpr int AG_NQ = 4, AG_NKH = 2, AG_NW = AG_NQ * AG_NKH, AG_BQ = AG_NQ * 16, AG_BK = 64, AG_KP = 80;     // 80-byte rows: ldmatrix is conflict free
constexpr int AG_STAGE_ROWS = AG_NKH * AG_BK;                                                              // 128 key rows per stage
constexpr int AG_STAGE = 2 * AG_STAGE_ROWS * AG_KP;                                                        // K then V
constexpr int AG_SMEM = 2 * AG_STAGE;                                                                      // two stages (40 KB)
static_assert(AG_SMEM >= AG_NQ * 32 * 20 * 4, "the merge buffer reuses the stages");

struct AttnGParams {
  const __nv_bfloat16* q;       // [N][H][S][32]
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const int* seqused;           // [N]
  __nv_bfloat16* o;             // [N S][128]
  int S;
  float scl2;                   // 32^-0.5 log2 e
};

__global__ void __launch_bounds__(AG_NW * 32, 2) attn_global_fwd_kernel(const AttnGParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int qw = warp & (AG_NQ - 1), kh = warp / AG_NQ;
  const int nh = blockIdx.y, n = nh / 4, h = nh % 4;
  const int S = p.S, su = p.seqused[n], i0 = blockIdx.x * AG_BQ;
  const __nv_bfloat16* qp = p.q + (size_t)nh * S * 32;
  const __nv_bfloat16* kp = p.k + (size_t)nh * S * 32;
  const __nv_bfloat16* vp = p.v + (size_t)nh * S * 32;

  if (i0 >= su) {   // a tile of padding queries: zeros
    const int r = i0 + (tid >> 2);
    if (tid < AG_BQ * 4 && r < S) stg128(p.o + ((size_t)n * S + r) * 128 + h * 32 + (tid & 3) * 8, make_uint4(0, 0, 0, 0));
    return;
  }
  const int nit = (su + AG_STAGE_ROWS - 1) / AG_STAGE_ROWS;

  auto load = [&](int stage, int it) {
    const uint32_t sk = sb + stage * AG_STAGE, sv = sk + AG_STAGE_ROWS * AG_KP;
    for (int c = tid; c < AG_STAGE_ROWS * 4; c += AG_NW * 32) {
      const int row = c >> 2, part = c & 3, j = it * AG_STAGE_ROWS + row;
      const bool ok = j < su;
      const size_t jj = (size_t)(ok ? j : 0);
      cp_async16(sk + row * AG_KP + part * 16, kp + jj * 32 + part * 8, ok ? 16 : 0);
      cp_async16(sv + row * AG_KP + part * 16, vp + jj * 32 + part * 8, ok ? 16 : 0);
    }
  };
  load(0, 0);
  cp_async_commit();

  const int q0 = i0 + qw * 16, r0 = q0 + g8, r1 = r0 + 8;
  uint32_t qa[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int col = ks * 16 + 2 * q4;
    qa[ks][0] = r0 < su ? *reinterpret_cast<const uint32_t*>(qp + (size_t)r0 * 32 + col) : 0u;
    qa[ks][1] = r1 < su ? *reinterpret_cast<const uint32_t*>(qp + (size_t)r1 * 32 + col) : 0u;
    qa[ks][2] = r0 < su ? *reinterpret_cast<const uint32_t*>(qp + (size_t)r0 * 32 + col + 8) : 0u;
    qa[ks][3] = r1 < su ? *reinterpret_cast<const uint32_t*>(qp + (size_t)r1 * 32 + col + 8) : 0u;
  }

  float o[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;   // running max (raw score units), per-thread partial denominators

  for (int it = 0; it < nit; ++it) {
    if (it + 1 < nit) load((it + 1) & 1, it + 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const int jb = it * AG_STAGE_ROWS + kh * AG_BK;      // first key of this warp's block
    if (jb < su) {
      const uint32_t kbase = sb + (it & 1) * AG_STAGE + (kh * AG_BK) * AG_KP;
      const uint32_t vbase = kbase + AG_STAGE_ROWS * AG_KP;
      float s[8][4];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        uint32_t b[4];
        ldsm_x4(b, kbase + (8 * nt + (lane & 7)) * AG_KP + (lane >> 3) * 16);
        mma16816(s[nt], qa[0], b[0], b[1]);
        mma16816(s[nt], qa[1], b[2], b[3]);
      }
      if (jb + AG_BK > su) {   // the keys past seqused (their K rows are zero-filled, the scores must not count)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (jb + 8 * nt + 2 * q4 + (e & 1) >= su) s[nt][e] = -INFINITY;
      }
      float b0 = -INFINITY, b1 = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        b0 = fmaxf(b0, fmaxf(s[nt][0], s[nt][1]));
        b1 = fmaxf(b1, fmaxf(s[nt][2], s[nt][3]));
      }
      b0 = fmaxf(b0, __shfl_xor_sync(0xffffffffu, b0, 1)); b0 = fmaxf(b0, __shfl_xor_sync(0xffffffffu, b0, 2));
      b1 = fmaxf(b1, __shfl_xor_sync(0xffffffffu, b1, 1)); b1 = fmaxf(b1, __shfl_xor_sync(0xffffffffu, b1, 2));
      const float n0 = fmaxf(m0, b0), n1 = fmaxf(m1, b1);
      const float c0 = n0 == -INFINITY ? 0.f : n0, c1 = n1 == -INFINITY ? 0.f : n1;
      const float a0 = ex2f((m0 - c0) * p.scl2), a1 = ex2f((m1 - c1) * p.scl2);
      m0 = n0; m1 = n1;
      l0 *= a0; l1 *= a1;
#pragma unroll
      for (int i = 0; i < 4; ++i) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        s[nt][0] = ex2f((s[nt][0] - c0) * p.scl2); s[nt][1] = ex2f((s[nt][1] - c0) * p.scl2);
        s[nt][2] = ex2f((s[nt][2] - c1) * p.scl2); s[nt][3] = ex2f((s[nt][3] - c1) * p.scl2);
        l0 += s[nt][0] + s[nt][1];
        l1 += s[nt][2] + s[nt][3];
      }
#pragma unroll
      for (int kb = 0; kb < AG_BK / 16; ++kb) {
        const uint32_t a[4] = {pack_bf16(s[2 * kb][0], s[2 * kb][1]), pack_bf16(s[2 * kb][2], s[2 * kb][3]),
                               pack_bf16(s[2 * kb + 1][0], s[2 * kb + 1][1]), pack_bf16(s[2 * kb + 1][2], s[2 * kb + 1][3])};
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
          uint32_t b[4];
          const int m = lane >> 3;
          ldsm_x4_t(b, vbase + (16 * kb + ((m & 1) << 3) + (lane & 7)) * AG_KP + (nn * 16 + (m >> 1) * 8) * 2);
          mma16816(o[2 * nn], a, b[0], b[1]);
          mma16816(o[2 * nn + 1], a, b[2], b[3]);
        }
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- merge the two key halves of every query group: half 1 hands (m, l, o) to half 0 through shared memory
  float* mg = reinterpret_cast<float*>(smem_raw) + (qw * 32 + lane) * 20;
  if (kh == 1) {
    mg[0] = m0; mg[1] = m1; mg[2] = l0; mg[3] = l1;
#pragma unroll
    for (int i = 0; i < 4; ++i) { mg[4 + 4 * i] = o[i][0]; mg[5 + 4 * i] = o[i][1]; mg[6 + 4 * i] = o[i][2]; mg[7 + 4 * i] = o[i][3]; }
  }
  __syncthreads();
  if (kh == 1) return;
  const float bm0 = mg[0], bm1 = mg[1], bl0 = mg[2], bl1 = mg[3];
  const float t0 = fmaxf(m0, bm0), t1 = fmaxf(m1, bm1);
  const float u0 = t0 == -INFINITY ? 0.f : t0, u1 = t1 == -INFINITY ? 0.f : t1;
  const float x0 = ex2f((m0 - u0) * p.scl2), y0 = ex2f((bm0 - u0) * p.scl2), x1 = ex2f((m1 - u1) * p.scl2), y1 = ex2f((bm1 - u1) * p.scl2);
  const float L0 = quad_sum(x0 * l0 + y0 * bl0), L1 = quad_sum(x1 * l1 + y1 * bl1);
  const float il0 = L0 > 0.f ? 1.f / L0 : 1.f, il1 = L1 > 0.f ? 1.f / L1 : 1.f;
  __nv_bfloat16* orow0 = p.o + ((size_t)n * S + r0) * 128 + h * 32;
  __nv_bfloat16* orow1 = p.o + ((size_t)n * S + r1) * 128 + h * 32;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = 8 * nt + 2 * q4;
    const float v00 = (x0 * o[nt][0] + y0 * mg[4 + 4 * nt]) * il0, v01 = (x0 * o[nt][1] + y0 * mg[5 + 4 * nt]) * il0;
    const float v10 = (x1 * o[nt][2] + y1 * mg[6 + 4 * nt]) * il1, v11 = (x1 * o[nt][3] + y1 * mg[7 + 4 * nt]) * il1;
    if (r0 < S) stg32(orow0 + col, r0 < su ? pack_bf16(v00, v01) : 0u);
    if (r1 < S) stg32(orow1 + col, r1 < su ? pack_bf16(v10, v11) : 0u);
  }
}

}  // namespace sw80
