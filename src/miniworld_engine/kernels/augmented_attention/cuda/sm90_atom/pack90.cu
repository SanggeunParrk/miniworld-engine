// pack90.cu — every weight form of the AF3 atom block's row kernels (forward and backward packs) in one launch.
//
// The torch expression of the packs (cast / gamma fold / transpose / lane + slot permutations / cat: ~150 small kernels, ~190 us per call at
// fp32) becomes an index map: out[...] = cast(src[row][col] * scale[col]) with row / col the inverse of the lane permutation (``pcol``, 16 bytes
// per lane) and, for TF32, of the slot permutation [0 2 4 6 1 3 5 7] of the K axis. Block y = the output tensor; fp32 or bf16 sources (bit j of
// ``wbf``), tf32 (rounded to nearest, ties away) or bf16 outputs.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

struct Pack {
  const void* w[27];   // parameters in the ``_I`` order of sm90_atom/__init__.py
  void* out[16];       // cond wq... see JOBS
  uint32_t wbf;        // parameter i is bf16
  int tf32;
};

__device__ __forceinline__ float ldw(const Pack& p, int i, size_t idx) {
  return (p.wbf >> i) & 1 ? __uint_as_float((uint32_t)reinterpret_cast<const unsigned short*>(p.w[i])[idx] << 16)
                          : reinterpret_cast<const float*>(p.w[i])[idx];
}
// the axis map of ``_lperm``: out[L] = in[lp(L)] inside groups of gs
__device__ __forceinline__ int lp(int L, int gs) {
  const int g = L / gs, w = L - g * gs;
  return g * gs + 2 * (((w >> 1) & 3) * (gs >> 3) + (w >> 3)) + (w & 1);
}
// ``_kperm8``: out[i] = in[(i & 3) * 2 + (i >> 2)] inside groups of 8
__device__ __forceinline__ int kp(int c) { return (c & ~7) + ((c & 3) * 2 + ((c >> 2) & 1)); }

__device__ __forceinline__ void put(const Pack& p, void* out, size_t i, float v) {
  if (p.tf32) reinterpret_cast<float*>(out)[i] = __uint_as_float((__float_as_uint(v) + 0x1000u) & ~0x1FFFu);
  else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(v);
}

// parameter indices (the ``_I`` order)
enum { G1, SC1_W, SC1_B, BI1_W, WQ, BQ, WK, WV, LN_PAIR, WB, WG, WO, SO_W, SO_B, G2, SC2_W, SC2_B, BI2_W, WA, WB2, WD, ST_W, ST_B, GKV, KV_SC_W, KV_SC_B, KV_BI_W };

// conditioning matrix m of [Ws1', Wb1', Ws2', Wb2', Wkp', Wks', Wso, Wst]: source parameter and the K-axis scale parameter (-1: none)
__device__ constexpr int CM_W[8] = {SC1_W, BI1_W, SC2_W, BI2_W, KV_SC_W, KV_BI_W, SO_W, ST_W};
__device__ constexpr int CM_G[8] = {G1, G1, G2, G2, GKV, GKV, -1, -1};

__device__ __forceinline__ float cond_val(const Pack& p, int m, int r, int c) {   // W'[r][c] (fp32, before the cast)
  const float v = ldw(p, CM_W[m], (size_t)r * 128 + c);
  return CM_G[m] >= 0 ? v * ldw(p, CM_G[m], c) : v;
}

extern "C" __global__ __launch_bounds__(256) void pack90(const __grid_constant__ Pack p) {
  const int job = blockIdx.y;
  const int gs = p.tf32 ? 16 : 32;
  const int hb = p.tf32 ? 32 : 64, nch = 256 / hb;
  void* out = p.out[job];
  // column map of a prepped operand: the K axis in lane order (+ slot order for TF32)
  auto lc = [&](int c, int n) { (void)n; return p.tf32 ? lp(kp(c), gs) : lp(c, gs); };
  for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;; i += (size_t)gridDim.x * 256) {
    float v;
    switch (job) {
      case 0: {  // cond [8][128][128] (forward)
        if (i >= 8 * 16384) return;
        const int m = i >> 14, r = lp((i >> 7) & 127, gs), c = lc(i & 127, 128);
        v = cond_val(p, m, r, c);
        break;
      }
      case 1: {  // pre [512][128] = [Wq; Wg; Wk; Wv]
        if (i >= 512 * 128) return;
        const int r = lp(i >> 7, gs), c = lc(i & 127, 128);
        const int pi = r >> 7 == 0 ? WQ : r >> 7 == 1 ? WG : r >> 7 == 2 ? WK : WV;
        v = ldw(p, pi, (size_t)(r & 127) * 128 + c);
        break;
      }
      case 2: {  // wo [128][128]
        if (i >= 16384) return;
        v = ldw(p, WO, (size_t)lp(i >> 7, gs) * 128 + lc(i & 127, 128));
        break;
      }
      case 3: {  // wab [nch][2 hb][128]: chunk j = [Wa_j; Wb_j]
        if (i >= (size_t)nch * 2 * hb * 128) return;
        const int j = i / (2 * hb * 128), rem = i % (2 * hb * 128), r = lp(rem >> 7, gs), c = lc(rem & 127, 128);
        v = r < hb ? ldw(p, WA, (size_t)(j * hb + r) * 128 + c) : ldw(p, WB2, (size_t)(j * hb + r - hb) * 128 + c);
        break;
      }
      case 4: {  // wd [nch][128][hb]
        if (i >= (size_t)nch * 128 * hb) return;
        const int j = i / (128 * hb), rem = i % (128 * hb), r = lp(rem / hb, gs), c = lc(rem % hb, hb);
        v = ldw(p, WD, (size_t)r * 256 + j * hb + c);
        break;
      }
      case 5: {  // wabt [4 kb][2 half][64][128]
        if (i >= 4 * 2 * 64 * 128) return;
        const int kb = i >> 14, half = (i >> 13) & 1, r = lp((i >> 7) & 63, gs), hh = lc(i & 127, 128);
        const int hid = kb * 128 + hh;
        v = hid < 256 ? ldw(p, WA, (size_t)hid * 128 + half * 64 + r) : ldw(p, WB2, (size_t)(hid - 256) * 128 + half * 64 + r);
        break;
      }
      case 6: {  // wot [2][64][128]
        if (i >= 2 * 64 * 128) return;
        const int half = i >> 13, r = lp((i >> 7) & 63, gs), hh = lc(i & 127, 128);
        v = ldw(p, WO, (size_t)hh * 128 + half * 64 + r);
        break;
      }
      case 7: {  // wdt [nch][hb][128]
        if (i >= (size_t)nch * hb * 128) return;
        const int j = i / (hb * 128), rem = i % (hb * 128), r = lp(rem >> 7, gs), hh = lc(rem & 127, 128);
        v = ldw(p, WD, (size_t)hh * 256 + j * hb + r);
        break;
      }
      case 8: {  // wbt [4 (Wg Wq Wk Wv)][2][64][128]
        if (i >= 4 * 2 * 64 * 128) return;
        const int m = i >> 14, half = (i >> 13) & 1, r = lp((i >> 7) & 63, gs), hh = lc(i & 127, 128);
        const int pi = m == 0 ? WG : m == 1 ? WQ : m == 2 ? WK : WV;
        v = ldw(p, pi, (size_t)hh * 128 + half * 64 + r);
        break;
      }
      case 9: {  // wct [8][2][64][128]: W'^T tiles
        if (i >= 8 * 2 * 64 * 128) return;
        const int m = i >> 14, half = (i >> 13) & 1, r = lp((i >> 7) & 63, gs), hh = lc(i & 127, 128);
        v = cond_val(p, m, hh, half * 64 + r);
        break;
      }
      default: {  // the fp32 vectors: bias [512] = sc1_b | sc2_b | so_b | st_b, bq, bs, ln_pair [16], wb [64]
        float* o = reinterpret_cast<float*>(out);
        if (job == 10) { if (i >= 512) return; const int s = i >> 7; const int pi = s == 0 ? SC1_B : s == 1 ? SC2_B : s == 2 ? SO_B : ST_B; o[i] = ldw(p, pi, i & 127); }
        else if (job == 11) { if (i >= 128) return; o[i] = ldw(p, BQ, i); }
        else if (job == 12) { if (i >= 128) return; o[i] = ldw(p, KV_SC_B, i); }
        else if (job == 13) { if (i >= 16) return; o[i] = ldw(p, LN_PAIR, i); }
        else { if (i >= 64) return; o[i] = ldw(p, WB, i); }
        continue;
      }
    }
    put(p, out, i, v);
  }
}
