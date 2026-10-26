// dit_variants edge.cu -- the slid key window's EDGE windows of the atom DiT variants, forward and backward, in fp32.
//
// With ``key_window = "slide"`` the keys of window w are [start_w, start_w + 128) with start_w = min(max(32 w - 48, 0), max(n_real - 128, 0));
// that is the clipped window [32 w - 48, 32 w + 80) the mma.sync window attention reads for every window but five: windows 0, 1 and the
// last three real ones (``widx``). Those five are redone here over their own keys (``keys`` [5, 128], clamped to [0, N)) with their own
// bias (``ebias`` [5, 4, 32, 128], natural units, -inf off the valid keys), and the results replace the kernel's rows.
//
//   edge_bias_fwd / edge_bias_bwd   LN(z_e) gamma . Wb for the five windows, -inf where invalid; and its gradients
//   edge_attn_fwd                   o[a, 32 w + i, h 32 + d] = softmax_k(q k^T / sqrt(32) + bias) v for the five windows
//   edge_bwd_compute                before the main backward kernel: the edge windows' dq / dk / dv (fp32 side buffers), dbias (per
//                                   sample), and the edge rows of dO / D zeroed (the main kernel then contributes nothing for them)
//   edge_bwd_apply                  after it: dq / dk / dv of the side buffers added into the main kernel's outputs
// A window listed twice (fewer than five distinct windows) counts once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;
constexpr int WQ = 32, WK = 128, NH = 4, DH = 32, DM = 128, PC = 16;
__device__ __forceinline__ float4 zero4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

template <typename T> __device__ __forceinline__ float ld(const T* p) { return static_cast<float>(*p); }
template <> __device__ __forceinline__ float ld<bf>(const bf* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ void st(T* p, float v) { *p = static_cast<T>(v); }
template <> __device__ __forceinline__ void st<bf>(bf* p, float v) { *p = __float2bfloat16_rn(v); }

__device__ __forceinline__ bool dup_window(const int64_t* widx, int e) {
  for (int i = 0; i < e; ++i) if (widx[i] == widx[e]) return true;
  return false;
}

// ----------------------------------------------------------------------------------------------------------- bias
// one thread per (window e, query i, slot j) row of 16 channels
template <typename ZT>
__global__ void edge_bias_fwd_k(const ZT* __restrict__ Z, const float* __restrict__ GAMMA, const float* __restrict__ WB,
                                const uint8_t* __restrict__ VALID, float* __restrict__ EB, float eps, long rows) {
  const long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float z[PC], mean = 0.f;
#pragma unroll
  for (int c = 0; c < PC; ++c) { z[c] = ld(Z + r * PC + c); mean += z[c]; }
  mean *= 1.f / PC;
  float var = 0.f;
#pragma unroll
  for (int c = 0; c < PC; ++c) { z[c] -= mean; var += z[c] * z[c]; }
  const float rstd = rsqrtf(var * (1.f / PC) + eps);
  const long e = r / (WQ * WK), rem = r % (WQ * WK);          // rem = i * 128 + j
  const bool ok = VALID[r] != 0;
#pragma unroll
  for (int h = 0; h < NH; ++h) {
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < PC; ++c) acc += z[c] * rstd * GAMMA[c] * WB[h * PC + c];
    EB[(e * NH + h) * (WQ * WK) + rem] = ok ? acc : -INFINITY;
  }
}

// dz_e rows and per-block partial sums of dgamma [16] and dwb [4, 16]
template <typename ZT>
__global__ void __launch_bounds__(256) edge_bias_bwd_k(const ZT* __restrict__ Z, const float* __restrict__ GAMMA, const float* __restrict__ WB,
                                                       const float* __restrict__ DEB, ZT* __restrict__ DZ, float* __restrict__ PART, float eps, long rows) {
  __shared__ float red[8][PC + NH * PC];
  float dg[PC], dw[NH * PC];
#pragma unroll
  for (int c = 0; c < PC; ++c) dg[c] = 0.f;
#pragma unroll
  for (int c = 0; c < NH * PC; ++c) dw[c] = 0.f;
  for (long r = (long)blockIdx.x * 256 + threadIdx.x; r < rows; r += (long)gridDim.x * 256) {
    float z[PC], mean = 0.f;
#pragma unroll
    for (int c = 0; c < PC; ++c) { z[c] = ld(Z + r * PC + c); mean += z[c]; }
    mean *= 1.f / PC;
    float var = 0.f;
#pragma unroll
    for (int c = 0; c < PC; ++c) { z[c] -= mean; var += z[c] * z[c]; }
    const float rstd = rsqrtf(var * (1.f / PC) + eps);
    const long e = r / (WQ * WK), rem = r % (WQ * WK);
    float db[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) { db[h] = DEB[(e * NH + h) * (WQ * WK) + rem]; if (!(db[h] == db[h]) || isinf(db[h])) db[h] = 0.f; }
    float dxh[PC], m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int c = 0; c < PC; ++c) {
      const float xh = z[c] * rstd;
      float t = 0.f;
#pragma unroll
      for (int h = 0; h < NH; ++h) { t += db[h] * WB[h * PC + c]; dw[h * PC + c] += db[h] * xh * GAMMA[c]; }
      dg[c] += t * xh;
      dxh[c] = t * GAMMA[c];
      m1 += dxh[c]; m2 += dxh[c] * xh;
    }
    m1 *= 1.f / PC; m2 *= 1.f / PC;
#pragma unroll
    for (int c = 0; c < PC; ++c) st(DZ + r * PC + c, rstd * (dxh[c] - m1 - z[c] * rstd * m2));
  }
  // block sum of the 80 column sums
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
#pragma unroll
  for (int c = 0; c < PC + NH * PC; ++c) {
    float v = c < PC ? dg[c] : dw[c - PC];
    v = warp_sum(v);
    if (l == 0) red[w][c] = v;
  }
  __syncthreads();
  if (threadIdx.x < PC + NH * PC) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t += red[i][threadIdx.x];
    PART[(long)blockIdx.x * (PC + NH * PC) + threadIdx.x] = t;
  }
}

// ---------------------------------------------------------------------------------------------------------- attention
// 128 threads = one (sample, window, head). Shared tiles in fp32 with 36-float rows (16-byte aligned, conflict-free): Q [32], K [128],
// V [128], dO [32]; S / P / dS [32][132]. Register blocking everywhere: a thread owns one key's K / V row (scores, dP, dK, dV) or one
// output column d of 8 queries (P V, dS K), so every shared load feeds 4 or more FMAs.
constexpr int RS = 36, SS = 132;
constexpr int SM_Q = 0, SM_K = SM_Q + WQ * RS, SM_V = SM_K + WK * RS, SM_P = SM_V + WK * RS, SM_O = SM_P + WQ * SS, SM_S = SM_O + WQ * RS;
constexpr int SM_FWD = SM_O, SM_BWD = SM_S + WQ * SS;          // floats

template <typename AT>
__device__ __forceinline__ void load_tiles(const AT* Q, const AT* K, const AT* V, const int64_t* keys, const int64_t* widx, int a, int e, int h, long N,
                                           float* sq, float* sk, float* sv) {
  const long w = widx[e];
  for (int i = threadIdx.x; i < WQ * 8; i += 128) {                  // 8 float4 per 32-wide head row
    const int q = i / 8, c = (i % 8) * 4;
    *reinterpret_cast<float4*>(sq + q * RS + c) = V4<AT>::load(Q + ((long)a * N + w * WQ + q) * DM + h * DH + c);
  }
  for (int i = threadIdx.x; i < WK * 8; i += 128) {
    const int j = i / 8, c = (i % 8) * 4;
    const long row = (long)a * N + keys[e * WK + j];
    *reinterpret_cast<float4*>(sk + j * RS + c) = V4<AT>::load(K + row * DM + h * DH + c);
    *reinterpret_cast<float4*>(sv + j * RS + c) = V4<AT>::load(V + row * DM + h * DH + c);
  }
}

__device__ __forceinline__ float dot32(const float* row, const float (&r)[32]) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 v = *reinterpret_cast<const float4*>(row + 4 * i);
    acc += v.x * r[4 * i] + v.y * r[4 * i + 1] + v.z * r[4 * i + 2] + v.w * r[4 * i + 3];
  }
  return acc;
}

// scores -> probabilities in P [32][132] (thread = key slot; a row with no valid key gets zeros)
__device__ __forceinline__ void scores_probs(const float* sq, const float* sk, const float* EB, int e, int h, float* sP) {
  const int j = threadIdx.x;
  const float scale = 0.17677669529663687f;
  float kr[32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 v = *reinterpret_cast<const float4*>(sk + j * RS + 4 * i);
    kr[4 * i] = v.x; kr[4 * i + 1] = v.y; kr[4 * i + 2] = v.z; kr[4 * i + 3] = v.w;
  }
  const float* eb = EB + ((long)e * NH + h) * (WQ * WK);
  for (int q = 0; q < WQ; ++q) sP[q * SS + j] = dot32(sq + q * RS, kr) * scale + eb[q * WK + j];
  __syncthreads();
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;      // warp w owns rows w, w + 4, ...
  for (int q = w; q < WQ; q += 4) {
    float v[4], m = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = sP[q * SS + l + 32 * i]; m = fmaxf(m, v[i]); }
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = (m == -INFINITY) ? 0.f : __expf(v[i] - m); s += v[i]; }
    s = warp_sum(s);
    const float inv = s > 0.f ? 1.f / s : 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) sP[q * SS + l + 32 * i] = v[i] * inv;
  }
  __syncthreads();
}

// out[q][d] = sum_j M[q][j] B[j][d] for the warp's 8 queries (lane = d); M [32][132], B [128][36]
__device__ __forceinline__ void mat_times(const float* M, const float* B, int warp, int lane, float (&acc)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int j = 0; j < WK; j += 4) {
    const float b0 = B[j * RS + lane], b1 = B[(j + 1) * RS + lane], b2 = B[(j + 2) * RS + lane], b3 = B[(j + 3) * RS + lane];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 p = *reinterpret_cast<const float4*>(M + (8 * warp + i) * SS + j);
      acc[i] += p.x * b0 + p.y * b1 + p.z * b2 + p.w * b3;
    }
  }
}

template <typename AT>
__global__ void __launch_bounds__(128) edge_attn_fwd_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V,
                                                       const int64_t* __restrict__ keys, const int64_t* __restrict__ widx, const float* __restrict__ EB,
                                                       AT* __restrict__ O, long N) {
  extern __shared__ __align__(16) float sm[];
  const int h = blockIdx.x % NH, e = (blockIdx.x / NH) % 5, a = blockIdx.x / (NH * 5);
  load_tiles<AT>(Q, K, V, keys, widx, a, e, h, N, sm + SM_Q, sm + SM_K, sm + SM_V);
  __syncthreads();
  scores_probs(sm + SM_Q, sm + SM_K, EB, e, h, sm + SM_P);
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  float acc[8];
  mat_times(sm + SM_P, sm + SM_V, w, l, acc);
  const long wq = widx[e];
#pragma unroll
  for (int i = 0; i < 8; ++i) st(O + ((long)a * N + wq * WQ + 8 * w + i) * DM + h * DH + l, acc[i]);
}

// compute phase of the backward: dq_e [A, 5, 32, 128], dk_e / dv_e [A, 5, 128, 128] (fp32, columns h 32 + d), debias partial [A, 5, 4, 32, 128];
// zeroes the edge rows of dO (AT [A N, 128]) and of dd [A, 4, N]
template <typename AT>
__global__ void __launch_bounds__(128) edge_bwd_compute_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V, AT* __restrict__ DO,
                                                          float* __restrict__ DD, const int64_t* __restrict__ keys, const int64_t* __restrict__ widx,
                                                          const float* __restrict__ EB, float* __restrict__ DQE, float* __restrict__ DKE, float* __restrict__ DVE,
                                                          float* __restrict__ DBP, long N) {
  extern __shared__ __align__(16) float sm[];
  const int h = blockIdx.x % NH, e = (blockIdx.x / NH) % 5, a = blockIdx.x / (NH * 5);
  const long wq = widx[e];
  const bool dup = dup_window(widx, e);
  float* sq = sm + SM_Q; float* sk = sm + SM_K; float* sv = sm + SM_V; float* sP = sm + SM_P; float* sdo = sm + SM_O; float* sdS = sm + SM_S;
  load_tiles<AT>(Q, K, V, keys, widx, a, e, h, N, sq, sk, sv);
  for (int i = threadIdx.x; i < WQ * 8; i += 128) {
    const int q = i / 8, c = (i % 8) * 4;
    *reinterpret_cast<float4*>(sdo + q * RS + c) = dup ? zero4() : V4<AT>::load(DO + ((long)a * N + wq * WQ + q) * DM + h * DH + c);
  }
  __syncthreads();
  scores_probs(sq, sk, EB, e, h, sP);
  const int j = threadIdx.x, w = threadIdx.x / 32, l = threadIdx.x % 32;
  {   // dP[q][j] = sum_d dO[q][d] V[j][d]
    float vr[32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 v = *reinterpret_cast<const float4*>(sv + j * RS + 4 * i);
      vr[4 * i] = v.x; vr[4 * i + 1] = v.y; vr[4 * i + 2] = v.z; vr[4 * i + 3] = v.w;
    }
    for (int q = 0; q < WQ; ++q) sdS[q * SS + j] = dot32(sdo + q * RS, vr);
  }
  __syncthreads();
  for (int q = w; q < WQ; q += 4) {          // dS = P (dP - rowsum(P dP))
    float s = 0.f, p[4], d[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { p[i] = sP[q * SS + l + 32 * i]; d[i] = sdS[q * SS + l + 32 * i]; s += p[i] * d[i]; }
    s = warp_sum(s);
#pragma unroll
    for (int i = 0; i < 4; ++i) sdS[q * SS + l + 32 * i] = p[i] * (d[i] - s);
  }
  __syncthreads();
  const float scale = 0.17677669529663687f;
  const long eo = (long)(a * 5 + e);
  {   // dV[j] = sum_q P[q][j] dO[q];  dK[j] = scale sum_q dS[q][j] Q[q]    (thread = key slot j, 64 accumulators)
    float av[32], ak[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) { av[i] = 0.f; ak[i] = 0.f; }
    for (int q = 0; q < WQ; ++q) {
      const float p = sP[q * SS + j], ds = sdS[q * SS + j];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float4 o = *reinterpret_cast<const float4*>(sdo + q * RS + 4 * i), qq = *reinterpret_cast<const float4*>(sq + q * RS + 4 * i);
        av[4 * i] += p * o.x; av[4 * i + 1] += p * o.y; av[4 * i + 2] += p * o.z; av[4 * i + 3] += p * o.w;
        ak[4 * i] += ds * qq.x; ak[4 * i + 1] += ds * qq.y; ak[4 * i + 2] += ds * qq.z; ak[4 * i + 3] += ds * qq.w;
      }
    }
    float* pv = DVE + (eo * WK + j) * DM + h * DH;
    float* pk = DKE + (eo * WK + j) * DM + h * DH;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<float4*>(pv + 4 * i) = make_float4(av[4 * i], av[4 * i + 1], av[4 * i + 2], av[4 * i + 3]);
      *reinterpret_cast<float4*>(pk + 4 * i) = make_float4(ak[4 * i] * scale, ak[4 * i + 1] * scale, ak[4 * i + 2] * scale, ak[4 * i + 3] * scale);
    }
  }
  {   // dQ[q][d] = scale sum_j dS[q][j] K[j][d]
    float acc[8];
    mat_times(sdS, sk, w, l, acc);
#pragma unroll
    for (int i = 0; i < 8; ++i) DQE[(eo * WQ + 8 * w + i) * DM + h * DH + l] = acc[i] * scale;
  }
  // dbias partial: dS itself ([32][128] of (a, e, h))
  float* pb = DBP + ((eo * NH) + h) * (WQ * WK);
  for (int i = threadIdx.x; i < WQ * WK / 4; i += 128) {
    const int q = i / (WK / 4), c = (i % (WK / 4)) * 4;
    *reinterpret_cast<float4*>(pb + q * WK + c) = *reinterpret_cast<const float4*>(sdS + q * SS + c);
  }
  // the main kernel sees no gradient for the edge rows (a duplicate window leaves them to the block that owns the window)
  if (!dup) {
    for (int i = threadIdx.x; i < WQ * 8; i += 128) {
      const int q = i / 8, c = (i % 8) * 4;
      V4<AT>::store(DO + ((long)a * N + wq * WQ + q) * DM + h * DH + c, zero4());
    }
    for (int q = threadIdx.x; q < WQ; q += 128) DD[((long)a * NH + h) * N + wq * WQ + q] = 0.f;
  }
}

// apply phase: dq (rows of the window; the main kernel wrote zeros there), dk / dv (rows of the keys; windows may share keys, so one block owns
// all five windows of (a, head, 8 columns) and adds them one after the other)
template <typename AT>
__global__ void __launch_bounds__(128) edge_bwd_apply_k(AT* __restrict__ DKV, long ldd, const int64_t* __restrict__ keys, const int64_t* __restrict__ widx,
                                                        const float* __restrict__ DQE, const float* __restrict__ DKE, const float* __restrict__ DVE, long N) {
  const int dc = blockIdx.x % 4, h = (blockIdx.x / 4) % NH, a = blockIdx.x / (4 * NH);
  const int col = h * DH + dc * 8;
  for (int e = 0; e < 5; ++e) {
    if (dup_window(widx, e)) continue;
    const long w = widx[e], eo = (long)(a * 5 + e);
    for (int i = threadIdx.x; i < WQ * 2; i += 128) {                  // 32 rows x two float4
      const int q = i / 2, c = (i % 2) * 4;
      AT* p = DKV + ((long)a * N + w * WQ + q) * ldd + col + c;       // dq columns 0 ..
      const float4 cur = V4<AT>::load(p), add = *reinterpret_cast<const float4*>(DQE + (eo * WQ + q) * DM + col + c);
      V4<AT>::store(p, add4(cur, add));
    }
    for (int i = threadIdx.x; i < WK * 2; i += 128) {
      const int j = i / 2, c = (i % 2) * 4;
      const long row = (long)a * N + keys[e * WK + j];
      AT* pk = DKV + row * ldd + DM + col + c;
      AT* pv = DKV + row * ldd + 2 * DM + col + c;
      V4<AT>::store(pk, add4(V4<AT>::load(pk), *reinterpret_cast<const float4*>(DKE + (eo * WK + j) * DM + col + c)));
      V4<AT>::store(pv, add4(V4<AT>::load(pv), *reinterpret_cast<const float4*>(DVE + (eo * WK + j) * DM + col + c)));
    }
    __syncthreads();
  }
}

c10::cuda::CUDAStream S_() { return c10::cuda::getCurrentCUDAStream(); }
template <typename T> T* P(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* CP(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
bool is_bf(const at::Tensor& t) { return t.scalar_type() == at::kBFloat16; }
size_t smem_fwd() { return (size_t)SM_FWD * 4; }
size_t smem_bwd() { return (size_t)SM_BWD * 4; }
template <typename K> void prep(K kern, size_t bytes) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes); }
}  // namespace

at::Tensor edge_bias_fwd(at::Tensor z, at::Tensor gamma, at::Tensor wb, at::Tensor valid, double eps) {
  const c10::cuda::CUDAGuard guard(z.device());
  TORCH_CHECK(z.is_contiguous() && z.size(0) == 5 && z.size(1) == WQ && z.size(2) == WK && z.size(3) == PC, "edge_bias_fwd: z [5, 32, 128, 16]");
  auto eb = at::empty({5, NH, WQ, WK}, z.options().dtype(at::kFloat));
  const long rows = 5L * WQ * WK;
  if (is_bf(z)) edge_bias_fwd_k<bf><<<(rows + 255) / 256, 256, 0, S_()>>>(CP<bf>(z), CP<float>(gamma), CP<float>(wb), CP<uint8_t>(valid), P<float>(eb), (float)eps, rows);
  else edge_bias_fwd_k<float><<<(rows + 255) / 256, 256, 0, S_()>>>(CP<float>(z), CP<float>(gamma), CP<float>(wb), CP<uint8_t>(valid), P<float>(eb), (float)eps, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return eb;
}

std::vector<at::Tensor> edge_bias_bwd(at::Tensor z, at::Tensor gamma, at::Tensor wb, at::Tensor deb, double eps) {
  const c10::cuda::CUDAGuard guard(z.device());
  const long rows = 5L * WQ * WK;
  const int blocks = 40;
  auto dz = at::empty_like(z);
  auto part = at::empty({blocks, PC + NH * PC}, z.options().dtype(at::kFloat));
  if (is_bf(z)) edge_bias_bwd_k<bf><<<blocks, 256, 0, S_()>>>(CP<bf>(z), CP<float>(gamma), CP<float>(wb), CP<float>(deb), P<bf>(dz), P<float>(part), (float)eps, rows);
  else edge_bias_bwd_k<float><<<blocks, 256, 0, S_()>>>(CP<float>(z), CP<float>(gamma), CP<float>(wb), CP<float>(deb), P<float>(dz), P<float>(part), (float)eps, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, part};
}

void edge_attn_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor keys, at::Tensor widx, at::Tensor ebias, at::Tensor o) {
  const c10::cuda::CUDAGuard guard(q.device());
  const long N = q.size(1);
  const int A = q.size(0);
  const int blocks = A * 5 * NH;
  if (is_bf(q)) {
    prep(edge_attn_fwd_k<bf>, smem_fwd());
    edge_attn_fwd_k<bf><<<blocks, 128, smem_fwd(), S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), CP<int64_t>(keys), CP<int64_t>(widx), CP<float>(ebias), P<bf>(o), N);
  } else {
    prep(edge_attn_fwd_k<float>, smem_fwd());
    edge_attn_fwd_k<float><<<blocks, 128, smem_fwd(), S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), CP<int64_t>(keys), CP<int64_t>(widx), CP<float>(ebias),
                                                                P<float>(o), N);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void edge_bwd_compute(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dO, at::Tensor dd, at::Tensor keys, at::Tensor widx, at::Tensor ebias, at::Tensor dqe,
                      at::Tensor dke, at::Tensor dve, at::Tensor dbp) {
  const c10::cuda::CUDAGuard guard(q.device());
  const long N = q.size(1);
  const int A = q.size(0);
  const int blocks = A * 5 * NH;
  if (is_bf(q)) {
    prep(edge_bwd_compute_k<bf>, smem_bwd());
    edge_bwd_compute_k<bf><<<blocks, 128, smem_bwd(), S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), P<bf>(dO), P<float>(dd), CP<int64_t>(keys), CP<int64_t>(widx),
        CP<float>(ebias), P<float>(dqe), P<float>(dke), P<float>(dve), P<float>(dbp), N);
  } else {
    prep(edge_bwd_compute_k<float>, smem_bwd());
    edge_bwd_compute_k<float><<<blocks, 128, smem_bwd(), S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), P<float>(dO), P<float>(dd), CP<int64_t>(keys),
        CP<int64_t>(widx), CP<float>(ebias), P<float>(dqe), P<float>(dke), P<float>(dve), P<float>(dbp), N);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void edge_bwd_apply(at::Tensor dkv, at::Tensor keys, at::Tensor widx, at::Tensor dqe, at::Tensor dke, at::Tensor dve, int64_t A, int64_t N) {
  const c10::cuda::CUDAGuard guard(dkv.device());
  const int blocks = A * NH * 4;
  if (is_bf(dkv)) edge_bwd_apply_k<bf><<<blocks, 128, 0, S_()>>>(P<bf>(dkv), dkv.stride(0), CP<int64_t>(keys), CP<int64_t>(widx), CP<float>(dqe), CP<float>(dke),
                                                                   CP<float>(dve), N);
  else edge_bwd_apply_k<float><<<blocks, 128, 0, S_()>>>(P<float>(dkv), dkv.stride(0), CP<int64_t>(keys), CP<int64_t>(widx), CP<float>(dqe), CP<float>(dke),
                                                          CP<float>(dve), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_bias_fwd", &edge_bias_fwd);
  m.def("edge_bias_bwd", &edge_bias_bwd);
  m.def("edge_attn_fwd", &edge_attn_fwd);
  m.def("edge_bwd_compute", &edge_bwd_compute);
  m.def("edge_bwd_apply", &edge_bwd_apply);
}
