// finalize90.cu — the parameter gradients of the AF3 atom block from the weight-gradient GEMM outputs, in one launch.
//
// Inputs (fp32): dwgq [256,128] = [dWg; dWq], dwkv [256,128] = [dWk; dWv], dwo [128,128], dwab [512,128] = [dWa; dWb], dwd [128,256],
// dc1 / dc2 / dckv [256,128] = d W' of the gamma-folded conditioning weights (first / second AdaLN of the attention and transition, the K / V
// AdaLN), dwso / dwst [128,128], the column sums of the pre-activation gradients (sc1, bq, so, sc2, st, bs: [128] each), the pair-bias
// gradients (dgamma [16], dWb [4,16]) and the forward weights Ws1 / Wb1 / ... (for dgamma = sum_o dW' * W).
// Outputs: one tensor per parameter in its dtype (bit i of ``bf`` set: bf16). Block y = the output index (the ``_NAMES`` order).
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

struct Fin {
  const float *dwgq, *dwkv, *dwo, *dwab, *dwd, *dc1, *dc2, *dckv, *dwso, *dwst;
  const float *cs_sc1, *cs_bq, *cs_so, *cs_sc2, *cs_st, *cs_bs, *dgamma_pair, *dwb;
  const void *g1, *g2, *gkv, *sc1_w, *bi1_w, *sc2_w, *bi2_w, *kv_sc_w, *kv_bi_w;  // forward parameters (fp32 or bf16 by ``wbf`` bits)
  void* out[27];
  uint32_t bf;   // output i is bf16
  uint32_t wbf;  // forward parameter j (g1, g2, gkv, sc1_w, bi1_w, sc2_w, bi2_w, kv_sc_w, kv_bi_w) is bf16
};

__device__ __forceinline__ float ldp(const void* p, int i, bool bf) {
  return bf ? __uint_as_float((uint32_t)reinterpret_cast<const unsigned short*>(p)[i] << 16) : reinterpret_cast<const float*>(p)[i];
}
__device__ __forceinline__ void stp(void* p, int i, float v, bool bf) {
  if (bf) reinterpret_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
  else reinterpret_cast<float*>(p)[i] = v;
}

// dgamma[j] = sum_o dWp[o][j] W[o][j] over the 256 rows of two [128, 128] weights: threads (j = t & 127, half = t >> 7), halves added in order
__device__ __forceinline__ void gamma_grad(const float* dwp, const void* wa, bool wabf, const void* wb, bool wbbf, void* out, bool obf) {
  __shared__ float red[256];
  const int j = threadIdx.x & 127, hf = threadIdx.x >> 7;
  float v = 0.f;
#pragma unroll 16
  for (int r = 64 * hf; r < 64 * hf + 64; ++r) v += dwp[r * 128 + j] * ldp(wa, r * 128 + j, wabf) + dwp[(128 + r) * 128 + j] * ldp(wb, r * 128 + j, wbbf);
  red[threadIdx.x] = v;
  __syncthreads();
  if (threadIdx.x < 128) stp(out, threadIdx.x, red[threadIdx.x] + red[threadIdx.x + 128], obf);
}

extern "C" __global__ __launch_bounds__(256) void finalize90(const __grid_constant__ Fin f) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int o = blockIdx.y;
  const bool bf = (f.bf >> o) & 1;
  void* out = f.out[o];
  if (o == 0 || o == 14 || o == 23) {  // the three LayerNorm-weight gradients: one block each
    if (blockIdx.x == 0) {
      if (o == 0) gamma_grad(f.dc1, f.sc1_w, (f.wbf >> 3) & 1, f.bi1_w, (f.wbf >> 4) & 1, out, bf);
      else if (o == 14) gamma_grad(f.dc2, f.sc2_w, (f.wbf >> 5) & 1, f.bi2_w, (f.wbf >> 6) & 1, out, bf);
      else gamma_grad(f.dckv, f.kv_sc_w, (f.wbf >> 7) & 1, f.kv_bi_w, (f.wbf >> 8) & 1, out, bf);
    }
    return;
  }
  for (int i = blockIdx.x * 256 + threadIdx.x; i < 32768; i += gridDim.x * 256) {
    const int row = i >> 7, col = i & 127;   // [rows,128] views; vectors use i < 128, wd uses i < 32768 as [128,256]
    float v;
    bool ok = true;
    switch (o) {
      case 0: v = 0.f; if (i < 128) { for (int r = 0; r < 128; ++r) v += f.dc1[r * 128 + i] * ldp(f.sc1_w, r * 128 + i, (f.wbf >> 3) & 1) + f.dc1[(128 + r) * 128 + i] * ldp(f.bi1_w, r * 128 + i, (f.wbf >> 4) & 1); } else ok = false; break;
      case 1: ok = i < 16384; v = ok ? f.dc1[i] * ldp(f.g1, col, f.wbf & 1) : 0.f; break;
      case 2: ok = i < 128; v = ok ? f.cs_sc1[i] : 0.f; break;
      case 3: ok = i < 16384; v = ok ? f.dc1[16384 + i] * ldp(f.g1, col, f.wbf & 1) : 0.f; break;
      case 4: ok = i < 16384; v = ok ? f.dwgq[16384 + i] : 0.f; break;             // wq
      case 5: ok = i < 128; v = ok ? f.cs_bq[i] : 0.f; break;
      case 6: ok = i < 16384; v = ok ? f.dwkv[i] : 0.f; break;                       // wk
      case 7: ok = i < 16384; v = ok ? f.dwkv[16384 + i] : 0.f; break;               // wv
      case 8: ok = i < 16; v = ok ? f.dgamma_pair[i] : 0.f; break;
      case 9: ok = i < 64; v = ok ? f.dwb[i] : 0.f; break;
      case 10: ok = i < 16384; v = ok ? f.dwgq[i] : 0.f; break;                      // wg
      case 11: ok = i < 16384; v = ok ? f.dwo[i] : 0.f; break;
      case 12: ok = i < 16384; v = ok ? f.dwso[i] : 0.f; break;
      case 13: ok = i < 128; v = ok ? f.cs_so[i] : 0.f; break;
      case 14: v = 0.f; if (i < 128) { for (int r = 0; r < 128; ++r) v += f.dc2[r * 128 + i] * ldp(f.sc2_w, r * 128 + i, (f.wbf >> 5) & 1) + f.dc2[(128 + r) * 128 + i] * ldp(f.bi2_w, r * 128 + i, (f.wbf >> 6) & 1); } else ok = false; break;
      case 15: ok = i < 16384; v = ok ? f.dc2[i] * ldp(f.g2, col, (f.wbf >> 1) & 1) : 0.f; break;
      case 16: ok = i < 128; v = ok ? f.cs_sc2[i] : 0.f; break;
      case 17: ok = i < 16384; v = ok ? f.dc2[16384 + i] * ldp(f.g2, col, (f.wbf >> 1) & 1) : 0.f; break;
      case 18: v = f.dwab[i]; ok = true; break;                                       // wa [256,128]: rows 0..255 = i < 32768
      case 19: v = f.dwab[32768 + i]; break;                                          // wb2
      case 20: v = f.dwd[i]; break;                                                   // wd [128,256]
      case 21: ok = i < 16384; v = ok ? f.dwst[i] : 0.f; break;
      case 22: ok = i < 128; v = ok ? f.cs_st[i] : 0.f; break;
      case 23: v = 0.f; if (i < 128) { for (int r = 0; r < 128; ++r) v += f.dckv[r * 128 + i] * ldp(f.kv_sc_w, r * 128 + i, (f.wbf >> 7) & 1) + f.dckv[(128 + r) * 128 + i] * ldp(f.kv_bi_w, r * 128 + i, (f.wbf >> 8) & 1); } else ok = false; break;
      case 24: ok = i < 16384; v = ok ? f.dckv[i] * ldp(f.gkv, col, (f.wbf >> 2) & 1) : 0.f; break;
      case 25: ok = i < 128; v = ok ? f.cs_bs[i] : 0.f; break;
      default: ok = i < 16384; v = ok ? f.dckv[16384 + i] * ldp(f.gkv, col, (f.wbf >> 2) & 1) : 0.f; break;
    }
    if (ok) stp(out, i, v, bf);
  }
}
