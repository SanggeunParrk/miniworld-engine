// Bias-only token DiT, attention core with a pair bias of its own per sample, fp32 (TF32 tensor cores, mma.sync m16n8k8): the fp32
// counterpart of bo_psb.cu in its recompute form -- the forward keeps two floats per softmax row (mc, 1 / sum) instead of P, the backward
// rebuilds P from the bias with the forward's own arithmetic.
//
//   psb32_fwd   bias -> P = softmax(bias + key mask) -> og = sigmoid(g) (P v)       bias tiles double-buffered by cp.async, softmaxed
//               in place (P rounded to TF32, the MMA's A operand), v of the (head, sample) resident (rounded to TF32 once)
//   psb32_bwd   dV = P^T dO and dbias = P o (dO v^T - D), one pass over the bias tiles of 128 keys x 32 queries
//
// Operands are rounded to TF32 (cvt.rna) when they enter shared memory, so the inner loops convert nothing; accumulation, softmax
// statistics and the bias gradient's elementwise part are fp32. Layouts as bo_psb.cu with fp32 elements. No atomics: bitwise repeatable.
// The v-resident forward needs L (DH + 8) 4 bytes of shared memory: L up to 512 (the host refuses larger L; the caller keeps the cuBLAS core).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

namespace {
template <typename T> T* P_(at::Tensor t) { return reinterpret_cast<T*>(t.data_ptr()); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float tf32f(float x) { return __uint_as_float(tf32(x)); }
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// four 8 x 4-word matrices (the b16 8 x 8 ldmatrix reads 32-bit words as pairs: thread l gets word l % 4 of row l / 4 -- the m16n8k8
// A fragment's (g, t) layout); lane l gives the row address of row l % 8 of matrix l / 8
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

// ------------------------------------------------------------------------------------------------------------------ forward
// softmax of one row of NCH float4 chunks per lane, in place; the normalised row is stored TF32-rounded
template <int NCH, bool MASKED>
__device__ __forceinline__ void soft_row32(float* __restrict__ row, const float* __restrict__ Ma, int lane, float2* __restrict__ st) {
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
  if (lane == 0) *st = make_float2(mc, inv);
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = lane + 32 * k;
    *reinterpret_cast<float4*>(row + ch * 4) = make_float4(tf32f(v[k].x * inv), tf32f(v[k].y * inv), tf32f(v[k].z * inv), tf32f(v[k].w * inv));
  }
}

// CTA = (sample, head, a slice of the query-row blocks of BM rows); 8 warps. The product og = P v runs on (BM / 16) x (DH / 16) warps.
template <int DH, int BM, bool MASKED>
__global__ void __launch_bounds__(256) psb32_fwd_k(const float* __restrict__ BIAS, float2* __restrict__ STATS, const bool* __restrict__ MASK_ALL,
                                                   long mask_stride, const float* __restrict__ VG, float* __restrict__ OG, int L, int A, int H,
                                                   int nrb_per_cta) {
  constexpr int VP = DH + 8, VCH = DH / 4, NWN = DH / 16, NMMA = (BM / 16) * NWN;
  extern __shared__ __align__(16) uint8_t sm_raw[];
  const int PS = L + 4;
  float* Ps = reinterpret_cast<float*>(sm_raw);           // 2 x [BM][PS]
  float* Vs = Ps + 2 * BM * PS;                           // [L][VP]
  float* Ma = Vs + (size_t)L * VP;                        // [L] (MASKED)
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
  cp_commit();
  const float* vbase = VG + (long)a * L * ldv + h * DH;
  for (int idx = tid; idx < L * VCH; idx += 256) {
    const int r = idx / VCH, c = idx - r * VCH;
    cp_async16(Vs + r * VP + c * 4, vbase + (long)r * ldv + c * 4);
  }
  cp_commit();
  if constexpr (MASKED) {
    const bool* mk = MASK_ALL + (long)a * mask_stride;
    for (int j = tid; j < L; j += 256) Ma[j] = mk[j] ? 0.f : -3.0e38f;
  }

  const int wm = warp & (BM / 16 - 1), wn = warp / (BM / 16);
  const bool mma_warp = warp < NMMA;
  const int g = lane >> 2, t = lane & 3, nch = L / 128;

  for (int rb = rb0; rb < rb1; ++rb) {
    cp_wait<0>();
    __syncthreads();
    if (rb == rb0) {                                       // v to TF32, once
      for (int idx = tid; idx < L * VCH; idx += 256) {
        const int r = idx / VCH, c = idx - r * VCH;
        float4* p = reinterpret_cast<float4*>(Vs + r * VP + c * 4);
        const float4 x = *p;
        *p = make_float4(tf32f(x.x), tf32f(x.y), tf32f(x.z), tf32f(x.w));
      }
    }
    if (rb + 1 < rb1) { load_tile((rb + 1 - rb0) & 1, rb + 1); cp_commit(); }
    float* tile = Ps + ((rb - rb0) & 1) * BM * PS;
    for (int r = warp; r < BM; r += 8) {
      float* row = tile + r * PS;
      float2* st = STATS + ha * L + (long)rb * BM + r;
      switch (nch) {
        case 1: soft_row32<1, MASKED>(row, Ma, lane, st); break;
        case 2: soft_row32<2, MASKED>(row, Ma, lane, st); break;
        case 3: soft_row32<3, MASKED>(row, Ma, lane, st); break;
        default: soft_row32<4, MASKED>(row, Ma, lane, st); break;
      }
    }
    __syncthreads();

    if (mma_warp) {
      float acc[2][4];
#pragma unroll
      for (int n = 0; n < 2; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
      const uint32_t pa = smem_u32(tile);
      const int m = lane >> 3;
#pragma unroll 4
      for (int k0 = 0; k0 < L; k0 += 8) {
        uint32_t a0, a1, a2, a3;
        ldsm4(pa + ((wm * 16 + (m & 1) * 8 + (lane & 7)) * PS + k0 + (m >> 1) * 4) * 4, a0, a1, a2, a3);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const float* vp = Vs + (k0 + t) * VP + wn * 16 + nt * 8 + g;
          mma1688(acc[nt], a0, a1, a2, a3, f2u(vp[0]), f2u(vp[4 * VP]));
        }
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const long row = (long)a * L + (long)rb * BM + wm * 16 + g + half * 8;
          const int col = h * DH + wn * 16 + nt * 8 + 2 * t;
          const float2 gg = *reinterpret_cast<const float2*>(VG + row * ldv + DA + col);
          *reinterpret_cast<float2*>(OG + row * DA + col) = make_float2(acc[nt][2 * half] * sg(gg.x), acc[nt][2 * half + 1] * sg(gg.y));
        }
    }
  }
}

// ----------------------------------------------------------------------------------------------------------------- backward
constexpr int BBM = 32, BBN = 128;

// CTA = (128 keys, sample, head); walks the L query rows in blocks of 32. Warp w: dP rows 16 (w & 1), keys 32 (w >> 1); dV keys 16 w.
template <int DH, bool MASKED>
__global__ void __launch_bounds__(256, 2) psb32_bwd_k(const float* __restrict__ BIAS, const float2* __restrict__ STATS, const bool* __restrict__ MASK_ALL,
                                                      long mask_stride, const float* __restrict__ DO, const float* __restrict__ VG,
                                                      const float* __restrict__ DD, float* __restrict__ DBIAS, float* __restrict__ DVG, int L, int A, int H) {
  constexpr int VB = DH + 4, OP = DH + 8, PP = BBN + 8, NV = DH / 8, KS = DH / 8, VCH = DH / 4;
  extern __shared__ __align__(16) uint8_t sm_raw[];
  float* Vs = reinterpret_cast<float*>(sm_raw);           // [BBN][VB] keys j0..j0+127, TF32
  float* Ps = Vs + BBN * VB;                              // 2 x [BBM][PP]
  float* Os = Ps + 2 * BBM * PP;                          // 2 x [BBM][OP]
  float* Ds = Os + 2 * BBM * OP;                          // 2 x [BBM]
  float2* Ss = reinterpret_cast<float2*>(Ds + 2 * BBM);   // 2 x [BBM]
  float* Mb = reinterpret_cast<float*>(Ss + 2 * BBM);     // [BBN]
  const int j0 = blockIdx.x * BBN, a = blockIdx.y, h = blockIdx.z;
  const long ha = (long)h * A + a;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long DA = (long)H * DH, ldv = 2 * DA;
  const int nib = L / BBM;

  const float* vbase = VG + ((long)a * L + j0) * ldv + h * DH;
  for (int idx = tid; idx < BBN * VCH; idx += 256) {
    const int r = idx / VCH, c = idx - r * VCH;
    cp_async16(Vs + r * VB + c * 4, vbase + (long)r * ldv + c * 4);
  }
  auto load_stage = [&](int s, int ib) {
    const int i0 = ib * BBM;
    const float* pg = BIAS + (ha * L + i0) * L + j0;
    for (int idx = tid; idx < BBM * (BBN / 4); idx += 256) {
      const int r = idx / (BBN / 4), c = idx % (BBN / 4);
      cp_async16(Ps + s * BBM * PP + r * PP + c * 4, pg + (long)r * L + c * 4);
    }
    const float* og = DO + ((long)a * L + i0) * DA + h * DH;
    for (int idx = tid; idx < BBM * VCH; idx += 256) {
      const int r = idx / VCH, c = idx - r * VCH;
      cp_async16(Os + s * BBM * OP + r * OP + c * 4, og + (long)r * DA + c * 4);
    }
    if (tid < BBM / 4) cp_async16(Ds + s * BBM + tid * 4, DD + ((long)a * H + h) * L + i0 + tid * 4);
    if (tid >= 32 && tid < 32 + BBM / 2) cp_async16(Ss + s * BBM + (tid - 32) * 2, STATS + ha * L + i0 + (tid - 32) * 2);
  };
  load_stage(0, 0);
  cp_commit();
  if constexpr (MASKED) {
    const bool* mk = MASK_ALL + (long)a * mask_stride + j0;
    if (tid < BBN) Mb[tid] = mk[tid] ? 0.f : -3.0e38f;
  }

  const int wm = warp & 1, wn = warp >> 1, g = lane >> 2, t = lane & 3;
  float dv[NV][4];
#pragma unroll
  for (int n = 0; n < NV; ++n) dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.f;

  for (int ib = 0; ib < nib; ++ib) {
    const int s = ib & 1;
    cp_wait<0>();
    __syncthreads();
    if (ib == 0) {                                          // v to TF32, once
      for (int idx = tid; idx < BBN * VCH; idx += 256) {
        const int r = idx / VCH, c = idx - r * VCH;
        float4* p = reinterpret_cast<float4*>(Vs + r * VB + c * 4);
        const float4 x = *p;
        *p = make_float4(tf32f(x.x), tf32f(x.y), tf32f(x.z), tf32f(x.w));
      }
    }
    if (ib + 1 < nib) { load_stage(s ^ 1, ib + 1); cp_commit(); }
    float* tile = Ps + s * BBM * PP;
    float* otile = Os + s * BBM * OP;
    {                                                       // bias tile -> TF32 P tile, dO tile -> TF32 (in place)
      const float2* st = Ss + s * BBM;
      for (int idx = tid; idx < BBM * (BBN / 4); idx += 256) {
        const int r = idx / (BBN / 4), c = idx % (BBN / 4);
        const float2 ms = st[r];
        float4 x = *reinterpret_cast<const float4*>(tile + r * PP + c * 4);
        if constexpr (MASKED) {
          const float4 m = *reinterpret_cast<const float4*>(Mb + c * 4);
          x.x += m.x; x.y += m.y; x.z += m.z; x.w += m.w;
        }
        *reinterpret_cast<float4*>(tile + r * PP + c * 4) =
            make_float4(tf32f(ex2(fmaf(x.x, L2E, -ms.x)) * ms.y), tf32f(ex2(fmaf(x.y, L2E, -ms.x)) * ms.y),
                        tf32f(ex2(fmaf(x.z, L2E, -ms.x)) * ms.y), tf32f(ex2(fmaf(x.w, L2E, -ms.x)) * ms.y));
      }
      for (int idx = tid; idx < BBM * VCH; idx += 256) {
        const int r = idx / VCH, c = idx - r * VCH;
        float4* p = reinterpret_cast<float4*>(otile + r * OP + c * 4);
        const float4 x = *p;
        *p = make_float4(tf32f(x.x), tf32f(x.y), tf32f(x.z), tf32f(x.w));
      }
    }
    __syncthreads();
    const float* dd = Ds + s * BBM;

    // dP [16, 32] per warp = dO [16, DH] v^T
    float dp[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) dp[n][0] = dp[n][1] = dp[n][2] = dp[n][3] = 0.f;
    const uint32_t oa = smem_u32(otile);
    const int m = lane >> 3;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldsm4(oa + ((wm * 16 + (m & 1) * 8 + (lane & 7)) * OP + ks * 8 + (m >> 1) * 4) * 4, a0, a1, a2, a3);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float* vp = Vs + (wn * 32 + n * 8 + g) * VB + ks * 8 + t;
        mma1688(dp[n], a0, a1, a2, a3, f2u(vp[0]), f2u(vp[4]));
      }
    }
    // dbias = P (dP - D): the result waits in registers until dV has read the P tile
    float xb[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = wm * 16 + g + half * 8, c = wn * 32 + n * 8 + 2 * t;
        const float2 p = *reinterpret_cast<const float2*>(tile + r * PP + c);
        const float d = dd[r];
        xb[n][2 * half] = p.x * (dp[n][2 * half] - d);
        xb[n][2 * half + 1] = p.y * (dp[n][2 * half + 1] - d);
      }
    // dV [128, DH] += P^T [128, 32] dO [32, DH]; warp w owns keys 16 w .. 16 w + 15
#pragma unroll
    for (int k0 = 0; k0 < BBM; k0 += 8) {
      const float* pp = tile + (k0 + t) * PP + warp * 16 + g;
      const uint32_t a0 = f2u(pp[0]), a1 = f2u(pp[8]), a2 = f2u(pp[4 * PP]), a3 = f2u(pp[4 * PP + 8]);
#pragma unroll
      for (int n = 0; n < NV; ++n) {
        const float* op = otile + (k0 + t) * OP + n * 8 + g;
        mma1688(dv[n], a0, a1, a2, a3, f2u(op[0]), f2u(op[4 * OP]));
      }
    }
    __syncthreads();                                        // the P tile is read: the gradient replaces it in place
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half)
        *reinterpret_cast<float2*>(tile + (wm * 16 + g + half * 8) * PP + wn * 32 + n * 8 + 2 * t) = make_float2(xb[n][2 * half], xb[n][2 * half + 1]);
    __syncthreads();
    float* og = DBIAS + (ha * L + ib * BBM) * L + j0;
    for (int idx = tid; idx < BBM * (BBN / 4); idx += 256) {
      const int r = idx / (BBN / 4), c = idx % (BBN / 4);
      *reinterpret_cast<float4*>(og + (long)r * L + c * 4) = *reinterpret_cast<const float4*>(tile + r * PP + c * 4);
    }
  }
#pragma unroll
  for (int n = 0; n < NV; ++n)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const long row = (long)a * L + j0 + warp * 16 + g + half * 8;
      *reinterpret_cast<float2*>(DVG + row * ldv + h * DH + n * 8 + 2 * t) = make_float2(dv[n][2 * half], dv[n][2 * half + 1]);
    }
}

constexpr size_t SMEM_MAX = 232448;
size_t fwd_smem(int L, int DH, int bm) { return (size_t)(2 * bm * (L + 4) + (size_t)L * (DH + 8) + L) * 4; }
size_t bwd_smem(int DH) {
  return (size_t)(BBN * (DH + 4) + 2 * BBM * (BBN + 8) + 2 * BBM * (DH + 8) + 2 * BBM + 4 * BBM + BBN) * 4;
}

template <int DH, int BM, bool MASKED> void launch_fwd_t(at::Tensor bias, at::Tensor stats, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H, int nsm) {
  auto k = psb32_fwd_k<DH, BM, MASKED>;
  const size_t smem = fwd_smem(L, DH, BM);
  static bool attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_MAX) == cudaSuccess;
  TORCH_CHECK(attr && smem <= SMEM_MAX, "psb32_fwd: shared memory");
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, smem);
  const int slots = std::max(per, 1) * nsm, nrb = L / BM, pairs = A * H;
  int best = 1;
  long best_cost = -1;
  for (int sl = 1; sl <= nrb; ++sl) {
    const int per_cta = (nrb + sl - 1) / sl, ctas = pairs * ((nrb + per_cta - 1) / per_cta);
    const long cost = (long)((ctas + slots - 1) / slots) * (per_cta + 2);
    if (best_cost < 0 || cost < best_cost) { best_cost = cost; best = sl; }
  }
  const int per_cta = (nrb + best - 1) / best;
  k<<<dim3(A, H, (nrb + per_cta - 1) / per_cta), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      P_<const float>(bias), P_<float2>(stats), mk, mstride, P_<const float>(vg), P_<float>(og), L, A, H, per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int DH> void launch_fwd(at::Tensor bias, at::Tensor stats, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H) {
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const bool use32 = fwd_smem(L, DH, 32) <= SMEM_MAX;
  if (use32) {
    if (mk) launch_fwd_t<DH, 32, true>(bias, stats, mk, mstride, vg, og, L, A, H, nsm);
    else launch_fwd_t<DH, 32, false>(bias, stats, mk, mstride, vg, og, L, A, H, nsm);
  } else {
    if (mk) launch_fwd_t<DH, 16, true>(bias, stats, mk, mstride, vg, og, L, A, H, nsm);
    else launch_fwd_t<DH, 16, false>(bias, stats, mk, mstride, vg, og, L, A, H, nsm);
  }
}
template <int DH, bool MASKED> void launch_bwd_t(at::Tensor bias, at::Tensor stats, const bool* mk, long mstride, at::Tensor dout, at::Tensor vg, at::Tensor dd,
                                                 at::Tensor dbias, at::Tensor dvg, int L, int A, int H) {
  auto k = psb32_bwd_k<DH, MASKED>;
  const size_t smem = bwd_smem(DH);
  static bool attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_MAX) == cudaSuccess;
  TORCH_CHECK(attr && smem <= SMEM_MAX, "psb32_bwd: shared memory");
  k<<<dim3(L / BBN, A, H), 256, smem, at::cuda::getCurrentCUDAStream()>>>(P_<const float>(bias), P_<const float2>(stats), mk, mstride, P_<const float>(dout),
                                                                         P_<const float>(vg), P_<const float>(dd), P_<float>(dbias), P_<float>(dvg), L, A, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_f32(const at::Tensor& t, const char* n) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.stride(-1) == 1, n, ": fp32 CUDA, unit column stride"); }
const bool* mask_args(const c10::optional<at::Tensor>& mask, int64_t L, int64_t A, long& mstride, const char* who) {
  if (!mask.has_value()) return nullptr;
  TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && (mask->numel() == L || mask->numel() == A * L), who, ": bool mask [L] or [A, L]");
  mstride = mask->numel() == L ? 0 : L;
  return mask->data_ptr<bool>();
}
}  // namespace

// og = sigmoid(g) (P v), P = softmax(bias + key mask); stats [H A L, 2] fp32 get (mc, 1 / sum). bias [H A L, L]; vg [A L, 2 H dh]; og [A L, H dh].
void psb32_fwd(at::Tensor bias, at::Tensor stats, c10::optional<at::Tensor> mask, at::Tensor vg, at::Tensor og, int64_t samples, int64_t heads, int64_t dh) {
  check_f32(bias, "bias"); check_f32(vg, "vg"); check_f32(og, "og");
  const int64_t L = bias.size(1), A = samples, H = heads;
  TORCH_CHECK(bias.is_contiguous() && bias.size(0) == H * A * L, "psb32_fwd: bias [H A L, L]");
  TORCH_CHECK(stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == 2 * H * A * L, "psb32_fwd: stats [H A L, 2] fp32");
  TORCH_CHECK(L % 128 == 0 && L <= 512, "psb32_fwd: L a multiple of 128, at most 512");
  TORCH_CHECK(vg.size(0) == A * L && vg.size(1) == 2 * H * dh && vg.stride(0) == 2 * H * dh && og.size(0) == A * L && og.size(1) == H * dh && og.stride(0) == H * dh,
              "psb32_fwd: vg [A L, 2 H dh], og [A L, H dh]");
  long mstride = 0;
  const bool* mk = mask_args(mask, L, A, mstride, "psb32_fwd");
  const at::cuda::CUDAGuard g(bias.device());
  if (dh == 48) launch_fwd<48>(bias, stats, mk, mstride, vg, og, (int)L, (int)A, (int)H);
  else if (dh == 32) launch_fwd<32>(bias, stats, mk, mstride, vg, og, (int)L, (int)A, (int)H);
  else if (dh == 64) launch_fwd<64>(bias, stats, mk, mstride, vg, og, (int)L, (int)A, (int)H);
  else TORCH_CHECK(false, "psb32_fwd: head width 32, 48 or 64");
}

// dV = P^T dO (into dvg's v columns) and dbias = P o (dO v^T - D), P rebuilt from the bias, the statistics and the key mask.
void psb32_bwd(at::Tensor bias, at::Tensor stats, c10::optional<at::Tensor> mask, at::Tensor dout, at::Tensor vg, at::Tensor dd, at::Tensor dbias,
               at::Tensor dvg, int64_t samples, int64_t heads, int64_t dh) {
  check_f32(bias, "bias"); check_f32(dout, "do"); check_f32(vg, "vg"); check_f32(dbias, "dbias"); check_f32(dvg, "dvg");
  const int64_t L = bias.size(1), A = samples, H = heads;
  TORCH_CHECK(bias.is_contiguous() && dbias.is_contiguous() && bias.size(0) == H * A * L && dbias.sizes() == bias.sizes(), "psb32_bwd: bias, dbias [H A L, L]");
  TORCH_CHECK(L % 128 == 0 && L <= 768, "psb32_bwd: L a multiple of 128, at most 768");
  TORCH_CHECK(dout.is_contiguous() && dout.size(0) == A * L && dout.size(1) == H * dh, "psb32_bwd: do [A L, H dh]");
  TORCH_CHECK(vg.size(0) == A * L && vg.stride(0) == 2 * H * dh && dvg.size(0) == A * L && dvg.stride(0) == 2 * H * dh && dvg.size(1) == 2 * H * dh, "psb32_bwd: vg, dvg [A L, 2 H dh]");
  TORCH_CHECK(dd.scalar_type() == at::kFloat && dd.is_contiguous() && dd.numel() == A * H * L, "psb32_bwd: dd [A, H, L] fp32");
  TORCH_CHECK(stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == 2 * H * A * L, "psb32_bwd: stats [H A L, 2] fp32");
  long mstride = 0;
  const bool* mk = mask_args(mask, L, A, mstride, "psb32_bwd");
  const at::cuda::CUDAGuard g(bias.device());
#define L_(DH)                                                                                                                                     \
  if (mk) launch_bwd_t<DH, true>(bias, stats, mk, mstride, dout, vg, dd, dbias, dvg, (int)L, (int)A, (int)H);                                     \
  else launch_bwd_t<DH, false>(bias, stats, nullptr, 0, dout, vg, dd, dbias, dvg, (int)L, (int)A, (int)H);
  if (dh == 48) { L_(48) }
  else if (dh == 32) { L_(32) }
  else if (dh == 64) { L_(64) }
  else TORCH_CHECK(false, "psb32_bwd: head width 32, 48 or 64");
#undef L_
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psb32_fwd", &psb32_fwd);
  m.def("psb32_bwd", &psb32_bwd);
}
