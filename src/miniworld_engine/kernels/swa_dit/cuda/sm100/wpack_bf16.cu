// wpack_bf16.cu — every bf16 weight form of the SWA atom block's sm_100a MASTER training (fp32 master weights, bf16 kernels) in ONE launch
// (round 3 part 8, B200 2026-11-17): the forward's forms (Wmod, [Wqkv; Wg] adjacent, Wo, Wu, Wd: sm100.master_pack's views) and the
// backward's transposes (Wd^T, Wo^T, Wu^T, [Wqkv; Wg]^T: sm100._w_bwd's), all plain round-to-nearest casts of the fp32 masters (the
// values of torch's casts). Was: six cast launches in the forward (master_pack's cat) and two cat launches at the head of the backward.
// Made once per step at the head of the forward, so every backward kernel may read its weights BEFORE its griddepcontrol.wait (the
// `_wpre` builds): the forms are complete long before.
// No programmatic dependent launch: launched plainly, it never triggers early, so the next kernel (which reads weights before its own
// wait) starts after its stores. One thread per 4 output elements, block 256.
//   bf16 elements:  Wmod [768,128] 0 | Wqkv [384,128] 98304 | Wg [128,128] 147456 | Wo [128,128] 163840 | Wu [512,128] 180224 |
//                   Wd [128,256] 245760 | Wd^T [256,128] 278528 | Wo^T [128,128] 311296 | Wu^T [128,512] 327680 |
//                   [Wqkv; Wg]^T [128,512] 393216 | end 458752
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int C = 128;
constexpr int E_WM = 0, E_WQKV = 98304, E_WG = 147456, E_WO = 163840, E_WU = 180224, E_WD = 245760, E_WDT = 278528, E_WOT = 311296,
              E_WUT = 327680, E_WT = 393216, E_END = 458752;

__device__ __forceinline__ uint32_t pk(float a, float b) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(a)) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(b)) << 16);
}

extern "C" __global__ void __launch_bounds__(256)
swa_wpack_bf16_sm100(const float* __restrict__ wmod, const float* __restrict__ wqkv, const float* __restrict__ wg,
                     const float* __restrict__ wo, const float* __restrict__ wu, const float* __restrict__ wd, __nv_bfloat16* __restrict__ out) {
  const int e = 4 * (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= E_END) return;
  float v[4];
  if (e < E_WDT) {                                                         // direct casts
    const float* src = e < E_WQKV ? wmod + e : e < E_WG ? wqkv + (e - E_WQKV) : e < E_WO ? wg + (e - E_WG) : e < E_WU ? wo + (e - E_WO)
                     : e < E_WD ? wu + (e - E_WU) : wd + (e - E_WD);
    const float4 f = *reinterpret_cast<const float4*>(src);
    v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
  } else {                                                                 // transposes: out [r][c0 .. c0 + 3] = src [c][r]
    int rel, ncol, sc;                                                     // ncol: the form's row length; sc: the source's row length
    const float* src;
    if (e < E_WOT) { rel = e - E_WDT; ncol = 128; sc = 256; src = wd; }
    else if (e < E_WUT) { rel = e - E_WOT; ncol = 128; sc = 128; src = wo; }
    else if (e < E_WT) { rel = e - E_WUT; ncol = 512; sc = 128; src = wu; }
    else { rel = e - E_WT; ncol = 512; sc = 128; src = nullptr; }
    const int r = rel / ncol, c0 = rel % ncol;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = c0 + i;
      v[i] = src != nullptr ? src[(size_t)c * sc + r] : (c < 3 * C ? wqkv[(size_t)c * C + r] : wg[(size_t)(c - 3 * C) * C + r]);
    }
  }
  *reinterpret_cast<uint2*>(out + e) = make_uint2(pk(v[0], v[1]), pk(v[2], v[3]));
}

// swa_save32_sm100 (round 3 part 8): master training of one sample with the bf16 forward and the TF32 backward ("hybrid") -- the
// forward's bf16 saves (Qh, Kh, Vh, G, O, q1, X, PQ, PK, Att, Y, FF: twelve tensors of n elements each) widened to the fp32 tensors the
// TF32 backward kernels read, in one launch. Exact (every bf16 value is an fp32 value). One thread per 8 elements of one tensor.
struct Ptr12 { const __nv_bfloat16* s[12]; float* d[12]; };
extern "C" __global__ void __launch_bounds__(256)
swa_save32_sm100(const Ptr12 p, int n8, int nt) {
  asm volatile("griddepcontrol.wait;" ::: "memory");                    // the saves come from the previous kernels (PDL launch)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n8 * nt) return;
  const int t = i / n8, e = 8 * (i - t * n8);
  const uint4 u = *reinterpret_cast<const uint4*>(p.s[t] + e);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
  float4 a, b;
  a.x = __uint_as_float(w[0] << 16); a.y = __uint_as_float(w[0] & 0xffff0000u);
  a.z = __uint_as_float(w[1] << 16); a.w = __uint_as_float(w[1] & 0xffff0000u);
  b.x = __uint_as_float(w[2] << 16); b.y = __uint_as_float(w[2] & 0xffff0000u);
  b.z = __uint_as_float(w[3] << 16); b.w = __uint_as_float(w[3] & 0xffff0000u);
  float4* d = reinterpret_cast<float4*>(p.d[t] + e);
  d[0] = a; d[1] = b;
}

// swa_save32_lw_sm100 (r3-swa part 19): the same widening OFF the dependent chain -- launched in the backward right after the FFN gate
// kernel (which reads y in bf16 in place and triggers its dependents only after its own griddepcontrol.wait, so the forward's saves are
// complete when this grid starts). It triggers its own dependents at once, widens (grid-stride: a bounded grid stays resident beside the
// gate kernel), and waits for the gate kernel only at its END: the kernel after it (the FFN adaLN backward, which waits for this grid)
// still runs after the gate kernel, and the widening runs under the gate kernel instead of in front of it.
extern "C" __global__ void __launch_bounds__(256)
swa_save32_lw_sm100(const Ptr12 p, int n8, int nt) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  constexpr int U = 4;                                                   // loads in flight a thread
  const int total = n8 * nt, stride = (int)(gridDim.x * blockDim.x);
  for (int i0 = (int)(blockIdx.x * blockDim.x + threadIdx.x); i0 < total; i0 += U * stride) {
    uint4 u[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      if (i < total) {
        const int t = i / n8, e = 8 * (i - t * n8);
        u[k] = *reinterpret_cast<const uint4*>(p.s[t] + e);
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      if (i < total) {
        const int t = i / n8, e = 8 * (i - t * n8);
        const uint32_t w[4] = {u[k].x, u[k].y, u[k].z, u[k].w};
        float4 a, b;
        a.x = __uint_as_float(w[0] << 16); a.y = __uint_as_float(w[0] & 0xffff0000u);
        a.z = __uint_as_float(w[1] << 16); a.w = __uint_as_float(w[1] & 0xffff0000u);
        b.x = __uint_as_float(w[2] << 16); b.y = __uint_as_float(w[2] & 0xffff0000u);
        b.z = __uint_as_float(w[3] << 16); b.w = __uint_as_float(w[3] & 0xffff0000u);
        float4* d = reinterpret_cast<float4*>(p.d[t] + e);
        d[0] = a; d[1] = b;
      }
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");                    // completes after the gate kernel before it
}

// swa_wpack_hybrid_sm100 (round 3 part 8): the weight forms of master training with the bf16 forward and the TF32 backward in ONE launch:
// the bf16 forward forms (out16: Wmod, Wqkv, Wg, Wo, Wu, Wd -- elements 0 .. E_WDT of the layout above) and, rounded to the nearest TF32
// (cvt.rna, low 13 bits cleared), the TF32 backward's forms (out32, floats): W = [Wqkv; Wg] [512,128] 0 | WO [128,128] 65536 |
// WAB [512,128] 81920 (rows per 32-unit hidden chunk j: [Wu[32 j ..]; Wu[256 + 32 j ..]]) | WD [128,256] 147456 (wprep_tf32.cu's pack) |
// WdT [256,128] 180224 | WoT [128,128] 212992 (wprep_bwd_tf32.cu's) | end 229376. Was three launches (wprep_tf32, wprep_bwd_tf32,
// swa_wpack_bf16) with ~2 us launch gaps at the head of the step. Plain launch (no early trigger: the next kernel reads Wmod before its wait).
__device__ __forceinline__ float rna32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r & 0xffffe000u);
}
constexpr int F_W = 0, F_WO = 65536, F_WAB = 81920, F_WD = 147456, F_WDT = 180224, F_WOT = 212992, F_END = 229376;

extern "C" __global__ void __launch_bounds__(256)
swa_wpack_hybrid_sm100(const float* __restrict__ wmod, const float* __restrict__ wqkv, const float* __restrict__ wg,
                       const float* __restrict__ wo, const float* __restrict__ wu, const float* __restrict__ wd,
                       __nv_bfloat16* __restrict__ out16, float* __restrict__ out32) {
  const int e = 4 * (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e < E_WDT) {                                                         // the bf16 forward forms
    const float* src = e < E_WQKV ? wmod + e : e < E_WG ? wqkv + (e - E_WQKV) : e < E_WO ? wg + (e - E_WG) : e < E_WU ? wo + (e - E_WO)
                     : e < E_WD ? wu + (e - E_WU) : wd + (e - E_WD);
    const float4 f = *reinterpret_cast<const float4*>(src);
    *reinterpret_cast<uint2*>(out16 + e) = make_uint2(pk(f.x, f.y), pk(f.z, f.w));
    return;
  }
  const int f = e - E_WDT;
  if (f >= F_END) return;
  float v[4];
  if (f < F_WDT) {                                                         // wprep_tf32's pack: 4 consecutive elements of one source row
    const float* src;
    if (f < F_WO) src = f < 3 * C * C ? wqkv + f : wg + (f - 3 * C * C);
    else if (f < F_WAB) src = wo + (f - F_WO);
    else if (f < F_WD) {
      const int r = (f - F_WAB) / C, col = (f - F_WAB) % C, j = r >> 6, half = (r >> 5) & 1, w = r & 31;
      src = wu + (size_t)(half * 256 + 32 * j + w) * C + col;
    } else src = wd + (f - F_WD);
    const float4 x = *reinterpret_cast<const float4*>(src);
    v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
  } else {                                                                 // wprep_bwd_tf32's transposes
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int g = f + i;
      if (g < F_WOT) { const int h = (g - F_WDT) / C, c = (g - F_WDT) % C; v[i] = wd[c * 256 + h]; }
      else { const int j = (g - F_WOT) / C, c = (g - F_WOT) % C; v[i] = wo[c * C + j]; }
    }
  }
  *reinterpret_cast<float4*>(out32 + f) = make_float4(rna32(v[0]), rna32(v[1]), rna32(v[2]), rna32(v[3]));
}
