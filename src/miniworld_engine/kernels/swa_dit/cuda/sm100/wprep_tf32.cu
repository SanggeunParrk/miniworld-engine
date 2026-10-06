// wprep_tf32.cu — the weight forms of the fp32 SWA atom block's sm_100a FORWARD (tf32_fwd.py), each in ONE launch:
//   swa_wprep_tf32_sm100   the block's four forms into one packed fp32 buffer, every element rounded to the nearest TF32:
//                            W   [512, 128] = [Wqkv; Wg]                       (qkvg_fwd_tf32's B operand)       floats      0 .. 65535
//                            WO  [128, 128] = Wo                                (ffn_fwd_tf32)                           65536 .. 81919
//                            WAB [512, 128] = rows per 32-unit hidden chunk j: [Wu[32 j ..]; Wu[256 + 32 j ..]]       81920 .. 147455
//                            WD  [128, 256] = Wd                                                                      147456 .. 180223
//   swa_round_tf32_sm100   dst = rna_tf32(src) over n4 float4s (Wmod for mod_fwd_tf32)
// Why a kernel: the forms are cached per weight version, but inside a CUDA-graph capture the cache is scoped to the capture
// (kernels/_capture.py: every replay must see the weights as they are then), so the forms are rebuilt in every replay unless the
// caller declares static weights. Built from torch ops they were ten small kernels per block call (cat, permute copy, the two
// integer ops of the rounding per form) -- ~2 us each in a graph, more than the block's attention at small sizes. Here: one launch,
// which also joins the PDL chain (it lets the next kernel launch at once and waits for the previous one before its stores; the
// forward kernels read the weights after their own griddepcontrol.wait).
// Rounding: cvt.rna.tf32.f32 (nearest, ties away from zero), the low 13 bits cleared -- the same value as tf32_fwd._round_tf32 and
// bwd_prep_tf32.cu. One thread per float4, block 256.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

__device__ __forceinline__ uint32_t rna_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r & 0xffffe000u;
}
__device__ __forceinline__ uint4 rna_tf32x4(float4 v) { return make_uint4(rna_tf32(v.x), rna_tf32(v.y), rna_tf32(v.z), rna_tf32(v.w)); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

constexpr int C = 128, NHID = 256;
constexpr int E_W = 0, E_WO = E_W + 4 * C * C, E_WAB = E_WO + C * C, E_WD = E_WAB + 2 * NHID * C, E_END = E_WD + C * NHID;
static_assert(E_END == 180224, "packed size (tf32_fwd.WPACK)");

extern "C" __global__ void __launch_bounds__(256)
swa_wprep_tf32_sm100(const float* __restrict__ wqkv, const float* __restrict__ wg, const float* __restrict__ wo,
                     const float* __restrict__ wu, const float* __restrict__ wd, float* __restrict__ out) {
  pdl_launch();
  const int e = 4 * (int)(blockIdx.x * blockDim.x + threadIdx.x);          // first packed element of this thread's float4
  if (e >= E_END) { pdl_wait(); return; }
  const float* src;
  if (e < E_WO) {
    src = e < 3 * C * C ? wqkv + e : wg + (e - 3 * C * C);
  } else if (e < E_WAB) {
    src = wo + (e - E_WO);
  } else if (e < E_WD) {
    const int r = (e - E_WAB) / C, col = (e - E_WAB) % C;                  // packed row r = 64 j + 32 half + w
    const int j = r >> 6, half = (r >> 5) & 1, w = r & 31;
    src = wu + (size_t)(half * NHID + 32 * j + w) * C + col;
  } else {
    src = wd + (e - E_WD);
  }
  const uint4 v = rna_tf32x4(*reinterpret_cast<const float4*>(src));
  pdl_wait();                                                              // the previous kernel may still read this buffer's memory
  *reinterpret_cast<uint4*>(out + e) = v;
}

extern "C" __global__ void __launch_bounds__(256)
swa_round_tf32_sm100(const float* __restrict__ src, float* __restrict__ dst, int n4) {
  pdl_launch();
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (i < n4) v = rna_tf32x4(reinterpret_cast<const float4*>(src)[i]);
  pdl_wait();
  if (i < n4) reinterpret_cast<uint4*>(dst)[i] = v;
}
