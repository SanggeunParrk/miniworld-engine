// wprep_bwd_tf32.cu — the weight forms of the fp32 SWA atom block's sm_100a BACKWARD, both in ONE launch (the forward's forms are
// wprep_tf32.cu): the transposed weights that are K-major B operands of the backward kernels, rounded to the nearest TF32 (cvt.rna, low 13 bits
// cleared), into one packed fp32 buffer:
//   WdT [256, 128] = Wd^T      (ffn_bwd_gate_tf32: dh = dffn Wd)           floats      0 .. 32767
//   WoT [128, 128] = Wo^T      (oproj_bwd_fused_tf32: dgated = datt Wo)    floats  32768 .. 49151
// Why a kernel: the forms are cached per weight version, but inside a CUDA-graph capture the cache is scoped to the capture (every replay must
// see the weights as they are then), so they are rebuilt in every replay; as torch ops (transpose copy + two integer ops each) they were seven
// small kernels with launch gaps. One thread per element.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

constexpr int C = 128, NHID = 256, E_WD = NHID * C, E_END = E_WD + C * C;

__device__ __forceinline__ float rna_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r & 0xffffe000u);
}

extern "C" __global__ void __launch_bounds__(256)
swa_wprep_bwd_tf32_sm100(const float* __restrict__ wd, const float* __restrict__ wo, float* __restrict__ out) {
  const int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= E_END) return;
  if (e < E_WD) {
    const int h = e / C, c = e % C;                                        // WdT[h][c] = Wd[c][h]   (Wd [128, 256])
    out[e] = rna_tf32(wd[c * NHID + h]);
  } else {
    const int i = e - E_WD, j = i / C, c = i % C;                          // WoT[j][c] = Wo[c][j]   (Wo [128, 128])
    out[e] = rna_tf32(wo[c * C + j]);
  }
}
