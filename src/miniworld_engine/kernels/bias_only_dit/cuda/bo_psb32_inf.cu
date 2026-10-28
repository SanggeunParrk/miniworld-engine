// Bias-only token DiT, INFERENCE attention core with a pair bias of its own per sample, fp32 (TF32 tensor cores, mma.sync m16n8k8).
//
//   psb32i_fwd   og = sigmoid(g) (softmax(bias + key mask) v) per (head, sample)
//
// The forward of bo_psb32.cu without its backward bookkeeping, restructured for the inference shapes (L 128 .. 512; v of the
// (head, sample) stays resident in shared memory, so a CTA holds ONE tile of 16 query rows beside 114 KB of v at L = 512):
//   * no (max, 1 / sum) statistics are stored (nothing rebuilds P);
//   * the bias tiles of 16 rows are prefetched 2 (L < 384, two CTAs per SM) or 3 (one CTA per SM) tiles ahead by cp.async: the
//     training kernel keeps one tile in flight and idles the memory system while a tile is softmaxed and multiplied;
//   * the product P v uses all eight warps, 4 key quarters x 2 column halves (the training kernel runs three warps over every key),
//     the quarters' partial sums are added through shared memory in a fixed order (no atomics: bitwise repeatable).
// Operands enter shared memory TF32-rounded (cvt.rna) exactly as in bo_psb32.cu; accumulation, softmax and the gate are fp32.
// Layouts as bo_psb32.cu: bias [H A L, L] head-major, vg [A L, 2 H dh] = v | g, og [A L, H dh].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include "bo_ptx.cuh"

namespace {
template <typename T> T* P_(at::Tensor t) { return reinterpret_cast<T*>(t.data_ptr()); }

using namespace bop;
__device__ __forceinline__ float tf32f(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r);
}
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// four 8 x 4-word matrices; lane l gives the row address of row l % 8 of matrix l / 8 (the m16n8k8 A fragment of a TF32 tile)
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
constexpr float L2E = 1.4426950408889634f;
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ uint32_t f2u(float x) { return __float_as_uint(x); }

// softmax of one row of NCH float4 chunks per lane, in place; the normalised row is stored TF32-rounded (bo_psb32.cu's arithmetic)
template <int NCH, bool MASKED>
__device__ __forceinline__ void soft_row32(float* __restrict__ row, const float* __restrict__ Ma, int lane) {
  float4 v[NCH];
  float mx = -3.4e38f;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = lane + 32 * k;
    v[k] = *reinterpret_cast<const float4*>(row + ch * 4);
    if constexpr (MASKED) {
      const float4 m = *reinterpret_cast<const float4*>(Ma + ch * 4);
      v[k].x += m.x; v[k].y += m.y; v[k].z += m.z; v[k].w += m.w;
    }
    mx = fmaxf(mx, fmaxf(fmaxf(v[k].x, v[k].y), fmaxf(v[k].z, v[k].w)));
  }
  mx = warp_max(mx);
  const float mc = mx * L2E;
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    v[k].x = ex2(fmaf(v[k].x, L2E, -mc)); v[k].y = ex2(fmaf(v[k].y, L2E, -mc));
    v[k].z = ex2(fmaf(v[k].z, L2E, -mc)); v[k].w = ex2(fmaf(v[k].w, L2E, -mc));
    sum += (v[k].x + v[k].y) + (v[k].z + v[k].w);
  }
  const float inv = 1.f / warp_sum(sum);
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = lane + 32 * k;
    *reinterpret_cast<float4*>(row + ch * 4) = make_float4(tf32f(v[k].x * inv), tf32f(v[k].y * inv), tf32f(v[k].z * inv), tf32f(v[k].w * inv));
  }
}

// two rows (one warp, rows r0 and r1) softmaxed together: their reduction chains interleave
template <int NCH, bool MASKED>
__device__ __forceinline__ void soft_rows32x2(float* __restrict__ row0, float* __restrict__ row1, const float* __restrict__ Ma, int lane) {
  float4 v[2][NCH];
  float mx0 = -3.4e38f, mx1 = -3.4e38f;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = lane + 32 * k;
    v[0][k] = *reinterpret_cast<const float4*>(row0 + ch * 4);
    v[1][k] = *reinterpret_cast<const float4*>(row1 + ch * 4);
    if constexpr (MASKED) {
      const float4 m = *reinterpret_cast<const float4*>(Ma + ch * 4);
      v[0][k].x += m.x; v[0][k].y += m.y; v[0][k].z += m.z; v[0][k].w += m.w;
      v[1][k].x += m.x; v[1][k].y += m.y; v[1][k].z += m.z; v[1][k].w += m.w;
    }
    mx0 = fmaxf(mx0, fmaxf(fmaxf(v[0][k].x, v[0][k].y), fmaxf(v[0][k].z, v[0][k].w)));
    mx1 = fmaxf(mx1, fmaxf(fmaxf(v[1][k].x, v[1][k].y), fmaxf(v[1][k].z, v[1][k].w)));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  const float mc0 = mx0 * L2E, mc1 = mx1 * L2E;
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    v[0][k].x = ex2(fmaf(v[0][k].x, L2E, -mc0)); v[0][k].y = ex2(fmaf(v[0][k].y, L2E, -mc0));
    v[0][k].z = ex2(fmaf(v[0][k].z, L2E, -mc0)); v[0][k].w = ex2(fmaf(v[0][k].w, L2E, -mc0));
    v[1][k].x = ex2(fmaf(v[1][k].x, L2E, -mc1)); v[1][k].y = ex2(fmaf(v[1][k].y, L2E, -mc1));
    v[1][k].z = ex2(fmaf(v[1][k].z, L2E, -mc1)); v[1][k].w = ex2(fmaf(v[1][k].w, L2E, -mc1));
    s0 += (v[0][k].x + v[0][k].y) + (v[0][k].z + v[0][k].w);
    s1 += (v[1][k].x + v[1][k].y) + (v[1][k].z + v[1][k].w);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s0 += __shfl_xor_sync(0xffffffffu, s0, o);
    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
  }
  const float inv0 = 1.f / s0, inv1 = 1.f / s1;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = lane + 32 * k;
    *reinterpret_cast<float4*>(row0 + ch * 4) = make_float4(tf32f(v[0][k].x * inv0), tf32f(v[0][k].y * inv0), tf32f(v[0][k].z * inv0), tf32f(v[0][k].w * inv0));
    *reinterpret_cast<float4*>(row1 + ch * 4) = make_float4(tf32f(v[1][k].x * inv1), tf32f(v[1][k].y * inv1), tf32f(v[1][k].z * inv1), tf32f(v[1][k].w * inv1));
  }
}

constexpr int BM = 16;

// CTA = (sample, head, a slice of the 16-row blocks); 8 warps. Warp w: key quarter w & 3, column half w >> 2 (NTG = DH / 16 n8 tiles).
template <int DH, int STAGES, int MINB, bool MASKED>
__global__ void __launch_bounds__(256, MINB) psb32i_fwd_k(const float* __restrict__ BIAS, const bool* __restrict__ MASK_ALL, long mask_stride,
                                                        const float* __restrict__ VG, float* __restrict__ OG, int L, int A, int H, int nrb_per_cta) {
  constexpr int VP = DH + 8, VCH = DH / 4, NTG = DH / 16, NI = BM * DH / 256;
  extern __shared__ __align__(16) uint8_t sm_raw[];
  const int PS = L + 4;
  float* Ps = reinterpret_cast<float*>(sm_raw);           // STAGES x [BM][PS]
  float* Vs = Ps + STAGES * BM * PS;                      // [L][VP]
  float* Ma = Vs + (size_t)L * VP;                        // [L] additive key mask (MASKED)
  float* Red = Ma + L;                                    // [4][2][NTG][4][32] partial sums
  const int a = blockIdx.x, h = blockIdx.y;
  const int nrb = L / BM, rb0 = blockIdx.z * nrb_per_cta, rb1 = min(nrb, rb0 + nrb_per_cta);
  const long ha = (long)h * A + a;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long DA = (long)H * DH, ldv = 2 * DA;
  const int cpr = L / 4;

  auto load_tile = [&](int buf, int rb) {
    const float* brow = BIAS + (ha * L + (long)rb * BM) * L;
    float* dst = Ps + buf * BM * PS;
    for (int r = warp; r < BM; r += 8)
      for (int c = lane; c < cpr; c += 32) cp_async16(dst + r * PS + c * 4, brow + (long)r * L + c * 4);
  };
  load_tile(0, rb0);
  const float* vbase = VG + (long)a * L * ldv + h * DH;
  for (int idx = tid; idx < L * VCH; idx += 256) {
    const int r = idx / VCH, c = idx - r * VCH;
    cp_async16(Vs + r * VP + c * 4, vbase + (long)r * ldv + c * 4);
  }
  cp_commit();                                             // group 0: tile 0 and v
#pragma unroll
  for (int s = 1; s < STAGES - 1; ++s) {
    if (rb0 + s < rb1) load_tile(s, rb0 + s);
    cp_commit();                                           // group s: tile s (possibly empty)
  }
  if constexpr (MASKED) {
    const bool* mk = MASK_ALL + (long)a * mask_stride;
    for (int j = tid; j < L; j += 256) Ma[j] = mk[j] ? 0.f : -3.0e38f;
  }

  const int kq = warp & 3, cg = warp >> 2, g = lane >> 2, t = lane & 3, nch = L / 128, m = lane >> 3;
  const int kbeg = kq * (L / 4), kend = kbeg + L / 4;

  for (int rb = rb0; rb < rb1; ++rb) {
    cp_wait<STAGES - 2>();
    __syncthreads();                                       // tile rb (and v, mask) landed; every warp is done with iteration rb - 1
    if (rb == rb0) {                                       // v to TF32, once (made visible by the barrier below)
      for (int idx = tid; idx < L * VCH; idx += 256) {
        const int r = idx / VCH, c = idx - r * VCH;
        float4* p = reinterpret_cast<float4*>(Vs + r * VP + c * 4);
        const float4 x = *p;
        *p = make_float4(tf32f(x.x), tf32f(x.y), tf32f(x.z), tf32f(x.w));
      }
    }
    if (rb + STAGES - 1 < rb1) load_tile((rb + STAGES - 1 - rb0) % STAGES, rb + STAGES - 1);
    cp_commit();                                           // one group per iteration, empty past the end
    float gate[NI];                                        // the gate values of this thread's output elements, loaded now (their latency hides behind the softmax)
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const int idx = tid + 256 * j, r = idx / DH, c = idx - r * DH;
      gate[j] = VG[((long)a * L + (long)rb * BM + r) * ldv + DA + h * DH + c];
    }
    float* tile = Ps + ((rb - rb0) % STAGES) * BM * PS;
    {
      float* row0 = tile + warp * PS;
      float* row1 = tile + (warp + 8) * PS;
      switch (nch) {
        case 1: soft_rows32x2<1, MASKED>(row0, row1, Ma, lane); break;
        case 2: soft_rows32x2<2, MASKED>(row0, row1, Ma, lane); break;
        case 3: soft_rows32x2<3, MASKED>(row0, row1, Ma, lane); break;
        default: soft_rows32x2<4, MASKED>(row0, row1, Ma, lane); break;
      }
    }
    __syncthreads();

    float acc[NTG][4];
#pragma unroll
    for (int n = 0; n < NTG; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    const uint32_t pa = smem_u32(tile);
#pragma unroll 4
    for (int k0 = kbeg; k0 < kend; k0 += 8) {
      uint32_t a0, a1, a2, a3;
      ldsm4(pa + (((m & 1) * 8 + (lane & 7)) * PS + k0 + (m >> 1) * 4) * 4, a0, a1, a2, a3);
#pragma unroll
      for (int nt = 0; nt < NTG; ++nt) {
        const float* vp = Vs + (k0 + t) * VP + (cg * NTG + nt) * 8 + g;
        mma1688(acc[nt], a0, a1, a2, a3, f2u(vp[0]), f2u(vp[4 * VP]));
      }
    }
#pragma unroll
    for (int nt = 0; nt < NTG; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) Red[((((kq * 2 + cg) * NTG + nt) * 4) + e) * 32 + lane] = acc[nt][e];
    __syncthreads();
    // og tile [BM, DH]: the four key quarters added in a fixed order, gated, one coalesced store per element
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const int idx = tid + 256 * j, r = idx / DH, c = idx - r * DH;
      const int ntg = c >> 3, cc = c & 7;
      const int cgi = ntg / NTG, nt = ntg - cgi * NTG;
      const int e = (r >> 3) * 2 + (cc & 1), ln = (r & 7) * 4 + (cc >> 1);
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) s += Red[((((q * 2 + cgi) * NTG + nt) * 4) + e) * 32 + ln];
      const long row = (long)a * L + (long)rb * BM + r;
      OG[row * DA + h * DH + c] = s * sg(gate[j]);
    }
  }
}

constexpr size_t SMEM_MAX = 232448;
size_t fwd_smem(int L, int DH, int stages) {
  return (size_t)(stages * BM * (L + 4) + (size_t)L * (DH + 8) + L + 4 * 2 * (DH / 16) * 4 * 32) * 4;
}

template <int DH, int STAGES, int MINB, bool MASKED>
void launch_t(at::Tensor bias, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H, int nsm) {
  auto k = psb32i_fwd_k<DH, STAGES, MINB, MASKED>;
  const size_t smem = fwd_smem(L, DH, STAGES);
  static bool attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_MAX) == cudaSuccess;
  TORCH_CHECK(attr && smem <= SMEM_MAX, "psb32i_fwd: shared memory");
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, smem);
  const int slots = std::max(per, 1) * nsm, nrb = L / BM, pairs = A * H;
  int best = 1;
  long best_cost = -1;
  for (int sl = 1; sl <= nrb; ++sl) {
    const int per_cta = (nrb + sl - 1) / sl, ctas = pairs * ((nrb + per_cta - 1) / per_cta);
    const long cost = (long)((ctas + slots - 1) / slots) * (per_cta + 2);      // +2: the v / first-tile prologue
    if (best_cost < 0 || cost < best_cost) { best_cost = cost; best = sl; }
  }
  const int per_cta = (nrb + best - 1) / best;
  k<<<dim3(A, H, (nrb + per_cta - 1) / per_cta), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      P_<const float>(bias), mk, mstride, P_<const float>(vg), P_<float>(og), L, A, H, per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int DH, bool MASKED>
void launch_dh(at::Tensor bias, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H, int nsm) {
  // two CTAs per SM with a 2-deep ring where the shared memory allows it, else one CTA with a 3-deep ring
  if (2 * fwd_smem(L, DH, 2) <= 227 * 1024) launch_t<DH, 2, 2, MASKED>(bias, mk, mstride, vg, og, L, A, H, nsm);
  else launch_t<DH, 3, 1, MASKED>(bias, mk, mstride, vg, og, L, A, H, nsm);
}
void check_f32(const at::Tensor& t, const char* n) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.stride(-1) == 1, n, ": fp32 CUDA, unit column stride"); }
}  // namespace

// bias [H A L, L]; vg [A L, 2 H dh] (v | g); og [A L, H dh]; mask bool [L] or [A, L] or none. L a multiple of 128 up to 512, dh 32 or 48.
void psb32i_fwd(at::Tensor bias, c10::optional<at::Tensor> mask, at::Tensor vg, at::Tensor og, int64_t samples, int64_t heads, int64_t dh) {
  check_f32(bias, "bias"); check_f32(vg, "vg"); check_f32(og, "og");
  const int64_t L = bias.size(1), A = samples, H = heads;
  TORCH_CHECK(bias.is_contiguous() && bias.size(0) == H * A * L, "psb32i_fwd: bias [H A L, L]");
  TORCH_CHECK(L % 128 == 0 && L <= 512, "psb32i_fwd: L a multiple of 128, at most 512");
  TORCH_CHECK(vg.size(0) == A * L && vg.size(1) == 2 * H * dh && vg.stride(0) == 2 * H * dh && og.size(0) == A * L && og.size(1) == H * dh && og.stride(0) == H * dh,
              "psb32i_fwd: vg [A L, 2 H dh], og [A L, H dh]");
  const bool* mk = nullptr;
  long mstride = 0;
  if (mask.has_value()) {
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && (mask->numel() == L || mask->numel() == A * L), "psb32i_fwd: bool mask [L] or [A, L]");
    mk = mask->data_ptr<bool>();
    mstride = mask->numel() == L ? 0 : L;
  }
  const at::cuda::CUDAGuard g(bias.device());
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (dh == 48) { if (mk) launch_dh<48, true>(bias, mk, mstride, vg, og, (int)L, (int)A, (int)H, nsm); else launch_dh<48, false>(bias, mk, mstride, vg, og, (int)L, (int)A, (int)H, nsm); }
  else if (dh == 32) { if (mk) launch_dh<32, true>(bias, mk, mstride, vg, og, (int)L, (int)A, (int)H, nsm); else launch_dh<32, false>(bias, mk, mstride, vg, og, (int)L, (int)A, (int)H, nsm); }
  else TORCH_CHECK(false, "psb32i_fwd: head width 32 or 48");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("psb32i_fwd", &psb32i_fwd); }
