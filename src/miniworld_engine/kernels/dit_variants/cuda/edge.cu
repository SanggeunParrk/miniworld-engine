// dit_variants edge.cu -- the slid key window's EDGE and TAIL windows of the atom DiT variants, forward and backward, in fp32.
//
// With ``key_window = "slide"`` the keys of window w are [start_w, start_w + 128) with start_w = min(max(32 w - 48, 0), max(n_real - 128, 0));
// that is the clipped window [32 w - 48, 32 w + 80) the mma.sync window attention reads for every window but windows 0, 1, the
// last three real ones (``widx``) and every window past the last real one (the TAIL, below). The five edge windows are redone here over
// their own keys (``keys`` [5, 128], clamped to [0, N)) with their own
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
#include <type_traits>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;
constexpr int WQ = 32, WK = 128, NH = 4, DH = 32, DM = 128, PC = 16;

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


// ------------------------------------------------------------------------------------------------------------------ the tail
// The query windows past the last real one (w >= t0 = max(last + 1, 2), last = (n_real - 1) / 32: every query of them is a padding atom).
// Their slid key window is the same for all of them, keys [s, s + 128) with s = max(n_rule - 128, 0) (n_rule: the real count, or rounded up
// to a whole query window for slide_qblock), and a padding query sees every valid key (key < n_real; the conformer mask spares padding
// queries). How many there are is data-dependent, so the kernels read ``info`` = [t0, s, n_real] (int64, on the device) and do nothing
// for the windows below t0: no host read, capturable; a call without padding windows costs only the launches.
//
//   tail_info        info from the valid-key bytes (one CTA)
//   tail_bias_fwd    LN(z_w) gamma . Wb of the tail windows, -inf off the valid keys -> tb [nloop, 4, 32, 128] (rows below t0 unwritten)
//   tail_attn_fwd    o rows of the tail windows: softmax(q k^T / sqrt(32) + tb) v over the shared 128 keys, on tensor cores (K / V
//                    staged once per CTA; bf16 mma, or TF32 for fp32 operands as the window kernels)
//   tail_bwd         dq kernel: CTA = (window, head, sample chunk): dq rows into dqt, d tb summed over the chunk's samples (dtbp [SC, ...]);
//                    then dk / dv kernel: CTA = (sample, head, window chunk): dk / dv of the 128 shared keys over its windows in order
//                    (dkvt [WC, A, 128, 256] fp32), then those rows of dO / D zeroed (the main kernel then adds nothing for them)
//   tail_bwd_apply   after the main backward kernel: dqt rows and dkvt (its WC chunks summed in order) added into dq / dk / dv (one
//                    owner per element, deterministic)
//   tail_bias_bwd    the LN / projection backward of the tail windows (d tb = the sum of the SC partials, in order) added into dz;
//                    per-block partial sums of d gamma, d Wb
//   tail_dbias_sum   (A100 autograd) the SC partials summed into one d tb [1, nloop, 4, 32, 128]
template <typename ZT>
__global__ void tail_bias_fwd_k(const ZT* __restrict__ Z, const float* __restrict__ GAMMA, const float* __restrict__ WB,
                                const int64_t* __restrict__ INFO, float* __restrict__ TB, float eps, long rows) {
  const long t0 = INFO[0], s = INFO[1], nreal = INFO[2];
  for (long r = t0 * (WQ * WK) + (long)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (long)gridDim.x * blockDim.x) {
    float z[PC], mean = 0.f;
#pragma unroll
    for (int c = 0; c < PC; ++c) { z[c] = ld(Z + r * PC + c); mean += z[c]; }
    mean *= 1.f / PC;
    float var = 0.f;
#pragma unroll
    for (int c = 0; c < PC; ++c) { z[c] -= mean; var += z[c] * z[c]; }
    const float rstd = rsqrtf(var * (1.f / PC) + eps);
    const long e = r / (WQ * WK), rem = r % (WQ * WK);
    const bool ok = s + rem % WK < nreal;
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      float acc = 0.f;
#pragma unroll
      for (int c = 0; c < PC; ++c) acc += z[c] * rstd * GAMMA[c] * WB[h * PC + c];
      TB[(e * NH + h) * (WQ * WK) + rem] = ok ? acc : -INFINITY;
    }
  }
}

// the shared keys [s, s + 128) of sample a, head h (a key past N reads row N - 1: it is invalid, its bias -inf)
template <typename AT>
__device__ __forceinline__ void load_tail_kv(const AT* K, const AT* V, long s, int a, int h, long N, float* sk, float* sv) {
  for (int i = threadIdx.x; i < WK * 8; i += 128) {
    const int j = i / 8, c = (i % 8) * 4;
    const long row = (long)a * N + min(s + j, N - 1);
    *reinterpret_cast<float4*>(sk + j * RS + c) = V4<AT>::load(K + row * DM + h * DH + c);
    *reinterpret_cast<float4*>(sv + j * RS + c) = V4<AT>::load(V + row * DM + h * DH + c);
  }
}

template <typename AT>
__device__ __forceinline__ void load_rows(const AT* X, int a, long w, int h, long N, float* dst) {
  for (int i = threadIdx.x; i < WQ * 8; i += 128) {
    const int q = i / 8, c = (i % 8) * 4;
    *reinterpret_cast<float4*>(dst + q * RS + c) = V4<AT>::load(X + ((long)a * N + w * WQ + q) * DM + h * DH + c);
  }
}

// ---- the tail forward on tensor cores (mma.sync: bf16 m16n8k16, or TF32 m16n8k8 for fp32 operands; fp32 accumulation, softmax in fp32)
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t tf32_bits(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t ld32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }

// shared tiles of the tensor-core tail: bf16: K [128][40] and V^T [32][136] (bf16); fp32: K [128][36] and V [128][40] (tf32-rounded fp32)
constexpr int TK_BF = 40, TV_BF = 136, TK_F = 36, TV_F = 40;
constexpr size_t SMEM_TC_BF = (size_t)(WK * TK_BF + DH * TV_BF) * 2, SMEM_TC_F = (size_t)(WK * TK_F + WK * TV_F) * 4;

// CTA = (sample, head, chunk c) of 4 warps; warp pair p (warps 2p, 2p + 1: query rows 0-15 / 16-31) takes windows t0 + 2 c + p, then
// every 2 C-th. A warp owns 16 query rows x all 128 keys: S = Q K^T on mma, scale + bias, the row softmax in registers (quad shuffles),
// P (unnormalized, bf16 / tf32) V on mma, / row sum. A row with no valid key gets zeros.
template <typename AT>
__global__ void __launch_bounds__(128) tail_attn_fwd_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V,
                                                       const int64_t* __restrict__ INFO, const float* __restrict__ TB, AT* __restrict__ O, long N,
                                                       int nloop, int G, int C) {
  constexpr bool BF = std::is_same<AT, bf>::value;
  extern __shared__ __align__(16) unsigned char smraw[];
  const int h = blockIdx.x % NH, c = (blockIdx.x / NH) % C, a = blockIdx.x / (NH * C);
  const long t0 = INFO[0], s = INFO[1];
  if (t0 + 2 * c >= nloop) return;
  // stage the shared keys [s, s + 128) of (a, h) (a key past N reads row N - 1: invalid, bias -inf)
  for (int i = threadIdx.x; i < WK * DH; i += 128) {
    const int j = i / DH, d = i % DH;
    const long row = (long)a * N + min(s + j, N - 1);
    const long off = row * DM + h * DH + d;
    if constexpr (BF) {
      bf* ks = reinterpret_cast<bf*>(smraw);
      bf* vt = ks + WK * TK_BF;
      ks[j * TK_BF + d] = K[off];
      vt[d * TV_BF + j] = V[off];
    } else {
      float* ks = reinterpret_cast<float*>(smraw);
      float* vs = ks + WK * TK_F;
      ks[j * TK_F + d] = __uint_as_float(tf32_bits(K[off]));
      vs[j * TV_F + d] = __uint_as_float(tf32_bits(V[off]));
    }
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int p = warp / 2, mh = warp % 2;
  const float scale = 0.17677669529663687f;
  const float* tbg = TB + (long)(a % G) * nloop * NH * (WQ * WK);
  for (long win = t0 + 2 * c + p; win < nloop; win += 2 * (long)C) {
    const long r0 = (long)a * N + win * WQ + 16 * mh;            // this warp's first query row
    const AT* q0 = Q + (r0 + g) * DM + h * DH;
    const AT* q1 = q0 + 8 * DM;
    float S[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) S[nt][0] = S[nt][1] = S[nt][2] = S[nt][3] = 0.f;
    if constexpr (BF) {
      const bf* ks = reinterpret_cast<const bf*>(smraw);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const uint32_t a0 = ld32(q0 + kk * 16 + 2 * t), a1 = ld32(q1 + kk * 16 + 2 * t);
        const uint32_t a2 = ld32(q0 + kk * 16 + 2 * t + 8), a3 = ld32(q1 + kk * 16 + 2 * t + 8);
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
          const bf* kr = ks + (nt * 8 + g) * TK_BF + kk * 16 + 2 * t;
          mma16816(S[nt], a0, a1, a2, a3, ld32(kr), ld32(kr + 8));
        }
      }
    } else {
      const float* ks = reinterpret_cast<const float*>(smraw);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const uint32_t a0 = tf32_bits(ld(q0 + kk * 8 + t)), a1 = tf32_bits(ld(q1 + kk * 8 + t));
        const uint32_t a2 = tf32_bits(ld(q0 + kk * 8 + t + 4)), a3 = tf32_bits(ld(q1 + kk * 8 + t + 4));
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
          const float* kr = ks + (nt * 8 + g) * TK_F + kk * 8 + t;
          mma1688(S[nt], a0, a1, a2, a3, __float_as_uint(kr[0]), __float_as_uint(kr[4]));
        }
      }
    }
    // scale + bias, row max / sum (rows g and g + 8 of this warp's 16)
    const float* eb0 = tbg + ((long)win * NH + h) * (WQ * WK) + (16 * mh + g) * WK + 2 * t;
    const float* eb1 = eb0 + 8 * WK;
    float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float2 b0 = *reinterpret_cast<const float2*>(eb0 + nt * 8), b1 = *reinterpret_cast<const float2*>(eb1 + nt * 8);
      S[nt][0] = S[nt][0] * scale + b0.x; S[nt][1] = S[nt][1] * scale + b0.y;
      S[nt][2] = S[nt][2] * scale + b1.x; S[nt][3] = S[nt][3] * scale + b1.y;
      m0 = fmaxf(m0, fmaxf(S[nt][0], S[nt][1]));
      m1 = fmaxf(m1, fmaxf(S[nt][2], S[nt][3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
    }
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      S[nt][0] = m0 == -INFINITY ? 0.f : __expf(S[nt][0] - m0);
      S[nt][1] = m0 == -INFINITY ? 0.f : __expf(S[nt][1] - m0);
      S[nt][2] = m1 == -INFINITY ? 0.f : __expf(S[nt][2] - m1);
      S[nt][3] = m1 == -INFINITY ? 0.f : __expf(S[nt][3] - m1);
      l0 += S[nt][0] + S[nt][1];
      l1 += S[nt][2] + S[nt][3];
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      l0 += __shfl_xor_sync(0xffffffffu, l0, o);
      l1 += __shfl_xor_sync(0xffffffffu, l1, o);
    }
    // O = P V
    float Oa[4][4];
#pragma unroll
    for (int nd = 0; nd < 4; ++nd) Oa[nd][0] = Oa[nd][1] = Oa[nd][2] = Oa[nd][3] = 0.f;
    if constexpr (BF) {
      const bf* vt = reinterpret_cast<const bf*>(smraw) + WK * TK_BF;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint32_t a0 = pack_bf2(S[2 * kk][0], S[2 * kk][1]), a1 = pack_bf2(S[2 * kk][2], S[2 * kk][3]);
        const uint32_t a2 = pack_bf2(S[2 * kk + 1][0], S[2 * kk + 1][1]), a3 = pack_bf2(S[2 * kk + 1][2], S[2 * kk + 1][3]);
#pragma unroll
        for (int nd = 0; nd < 4; ++nd) {
          const bf* vr = vt + (nd * 8 + g) * TV_BF + kk * 16 + 2 * t;
          mma16816(Oa[nd], a0, a1, a2, a3, ld32(vr), ld32(vr + 8));
        }
      }
    } else {
      const float* vs = reinterpret_cast<const float*>(smraw) + WK * TK_F;
      const int src1 = (lane & ~3) | (t >> 1), src2 = src1 + 2;
      const bool odd = t & 1;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {          // keys 8 kk .. 8 kk + 7: S tile kk, re-laid as the A operand (cols t, t + 4)
        const float x0 = __shfl_sync(0xffffffffu, S[kk][0], src1), x1 = __shfl_sync(0xffffffffu, S[kk][1], src1);
        const float y0 = __shfl_sync(0xffffffffu, S[kk][2], src1), y1 = __shfl_sync(0xffffffffu, S[kk][3], src1);
        const float z0 = __shfl_sync(0xffffffffu, S[kk][0], src2), z1 = __shfl_sync(0xffffffffu, S[kk][1], src2);
        const float w0 = __shfl_sync(0xffffffffu, S[kk][2], src2), w1 = __shfl_sync(0xffffffffu, S[kk][3], src2);
        const uint32_t a0 = tf32_bits(odd ? x1 : x0), a1 = tf32_bits(odd ? y1 : y0), a2 = tf32_bits(odd ? z1 : z0), a3 = tf32_bits(odd ? w1 : w0);
#pragma unroll
        for (int nd = 0; nd < 4; ++nd) {
          const float* vr = vs + (kk * 8 + t) * TV_F + nd * 8 + g;
          mma1688(Oa[nd], a0, a1, a2, a3, __float_as_uint(vr[0]), __float_as_uint(vr[4 * TV_F]));
        }
      }
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    AT* o0 = O + (r0 + g) * DM + h * DH + 2 * t;
    AT* o1 = o0 + 8 * DM;
#pragma unroll
    for (int nd = 0; nd < 4; ++nd) {
      if constexpr (BF) {
        *reinterpret_cast<uint32_t*>(o0 + nd * 8) = pack_bf2(Oa[nd][0] * i0, Oa[nd][1] * i0);
        *reinterpret_cast<uint32_t*>(o1 + nd * 8) = pack_bf2(Oa[nd][2] * i1, Oa[nd][3] * i1);
      } else {
        *reinterpret_cast<float2*>(o0 + nd * 8) = make_float2(Oa[nd][0] * i0, Oa[nd][1] * i0);
        *reinterpret_cast<float2*>(o1 + nd * 8) = make_float2(Oa[nd][2] * i1, Oa[nd][3] * i1);
      }
    }
  }
}

// P (in sP) and dS = P (dP - rowsum(P dP)) (in sdS) of one (sample, window, head) whose Q / K / V / dO tiles are staged
__device__ __forceinline__ void tail_ds(const float* sq, const float* sk, const float* sv, const float* sdo, const float* tb, int win, int h,
                                        float* sP, float* sdS) {
  scores_probs(sq, sk, tb, win, h, sP);
  const int j = threadIdx.x, w = threadIdx.x / 32, l = threadIdx.x % 32;
  {
    float vr[32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 v = *reinterpret_cast<const float4*>(sv + j * RS + 4 * i);
      vr[4 * i] = v.x; vr[4 * i + 1] = v.y; vr[4 * i + 2] = v.z; vr[4 * i + 3] = v.w;
    }
    for (int q = 0; q < WQ; ++q) sdS[q * SS + j] = dot32(sdo + q * RS, vr);
  }
  __syncthreads();
  for (int q = w; q < WQ; q += 4) {
    float s = 0.f, p[4], d[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { p[i] = sP[q * SS + l + 32 * i]; d[i] = sdS[q * SS + l + 32 * i]; s += p[i] * d[i]; }
    s = warp_sum(s);
#pragma unroll
    for (int i = 0; i < 4; ++i) sdS[q * SS + l + 32 * i] = p[i] * (d[i] - s);
  }
  __syncthreads();
}

// CTA = (window, head, sample chunk sc): samples sc, sc + SC, ...: dq rows (dqt, AT) and d tb summed over them (dtbp[sc])
template <typename AT>
__global__ void __launch_bounds__(128) tail_bwd_dq_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V,
                                                     const AT* __restrict__ DO, const int64_t* __restrict__ INFO, const float* __restrict__ TB,
                                                     AT* __restrict__ DQT, float* __restrict__ DTBP, long N, int nloop, int A, int SC) {
  extern __shared__ __align__(16) float sm[];
  const int h = blockIdx.x % NH, sc = (blockIdx.x / NH) % SC, win = blockIdx.x / (NH * SC);
  const long t0 = INFO[0], s = INFO[1];
  if (win < t0) return;
  float* sq = sm + SM_Q; float* sk = sm + SM_K; float* sv = sm + SM_V; float* sP = sm + SM_P; float* sdo = sm + SM_O; float* sdS = sm + SM_S;
  const int j = threadIdx.x, w = threadIdx.x / 32, l = threadIdx.x % 32;
  const float scale = 0.17677669529663687f;
  float db[WQ];
#pragma unroll
  for (int q = 0; q < WQ; ++q) db[q] = 0.f;
  for (int a = sc; a < A; a += SC) {
    load_tail_kv<AT>(K, V, s, a, h, N, sk, sv);
    load_rows<AT>(Q, a, win, h, N, sq);
    load_rows<AT>(DO, a, win, h, N, sdo);
    __syncthreads();
    tail_ds(sq, sk, sv, sdo, TB, win, h, sP, sdS);
#pragma unroll
    for (int q = 0; q < WQ; ++q) db[q] += sdS[q * SS + j];
    float acc[8];
    mat_times(sdS, sk, w, l, acc);
#pragma unroll
    for (int i = 0; i < 8; ++i) st(DQT + ((long)a * N + (long)win * WQ + 8 * w + i) * DM + h * DH + l, acc[i] * scale);
    __syncthreads();                        // the next sample's tiles overwrite K / dS
  }
  float* pb = DTBP + (((long)sc * nloop + win) * NH + h) * (WQ * WK);
#pragma unroll
  for (int q = 0; q < WQ; ++q) pb[q * WK + j] = db[q];
}


// CTA = (sample, head, window chunk c of WC): dk / dv of the shared keys over the tail windows t0 + c, t0 + c + WC, ... in order (dkvt
// [WC, A, 128, 256] fp32: dk | dv; a chunk without a window writes zeros), then those windows' rows of dO (this head's columns) and D
// zeroed. Runs after tail_bwd_dq_k (which reads dO).
template <typename AT>
__global__ void __launch_bounds__(128) tail_bwd_kv_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V, AT* __restrict__ DO,
                                                     float* __restrict__ DD, const int64_t* __restrict__ INFO, const float* __restrict__ TB,
                                                     float* __restrict__ DKVT, long N, int nloop, int A, int WC) {
  extern __shared__ __align__(16) float sm[];
  const int h = blockIdx.x % NH, a = (blockIdx.x / NH) % A, wc = blockIdx.x / (NH * A);
  const long t0 = INFO[0], s = INFO[1];
  if (t0 >= nloop) return;
  float* sq = sm + SM_Q; float* sk = sm + SM_K; float* sv = sm + SM_V; float* sP = sm + SM_P; float* sdo = sm + SM_O; float* sdS = sm + SM_S;
  const int j = threadIdx.x;
  const float scale = 0.17677669529663687f;
  float av[32], ak[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) { av[i] = 0.f; ak[i] = 0.f; }
  if (t0 + wc < nloop) load_tail_kv<AT>(K, V, s, a, h, N, sk, sv);
  for (long win = t0 + wc; win < nloop; win += WC) {
    load_rows<AT>(Q, a, win, h, N, sq);
    load_rows<AT>(DO, a, win, h, N, sdo);
    __syncthreads();
    tail_ds(sq, sk, sv, sdo, TB, (int)win, h, sP, sdS);
    for (int q = 0; q < WQ; ++q) {
      const float p = sP[q * SS + j], ds = sdS[q * SS + j];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float4 o = *reinterpret_cast<const float4*>(sdo + q * RS + 4 * i), qq = *reinterpret_cast<const float4*>(sq + q * RS + 4 * i);
        av[4 * i] += p * o.x; av[4 * i + 1] += p * o.y; av[4 * i + 2] += p * o.z; av[4 * i + 3] += p * o.w;
        ak[4 * i] += ds * qq.x; ak[4 * i + 1] += ds * qq.y; ak[4 * i + 2] += ds * qq.z; ak[4 * i + 3] += ds * qq.w;
      }
    }
    __syncthreads();                        // the next window's tiles overwrite Q / dO
  }
  float* pk = DKVT + (((long)wc * A + a) * WK + j) * (2 * DM) + h * DH;
  float* pv = pk + DM;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *reinterpret_cast<float4*>(pk + 4 * i) = make_float4(ak[4 * i] * scale, ak[4 * i + 1] * scale, ak[4 * i + 2] * scale, ak[4 * i + 3] * scale);
    *reinterpret_cast<float4*>(pv + 4 * i) = make_float4(av[4 * i], av[4 * i + 1], av[4 * i + 2], av[4 * i + 3]);
  }
  for (long win = t0 + wc; win < nloop; win += WC) {
    for (int i = threadIdx.x; i < WQ * 8; i += 128) {
      const long row = win * WQ + i / 8;
      V4<AT>::store(DO + ((long)a * N + row) * DM + h * DH + (i % 8) * 4, zero4());
    }
    if (threadIdx.x < WQ) DD[((long)a * NH + h) * N + win * WQ + threadIdx.x] = 0.f;
  }
}

// CTA = (sample, head, 8 columns): the tail rows' dq (dqt) and the shared keys' dk / dv (the WC chunk partials of dkvt, summed in order)
// added into DKV = [dq | dk | dv] (ld ldd)
template <typename AT>
__global__ void __launch_bounds__(128) tail_bwd_apply_k(AT* __restrict__ DKV, long ldd, const int64_t* __restrict__ INFO, const AT* __restrict__ DQT,
                                                        const float* __restrict__ DKVT, long N, int nloop, int A, int WC) {
  const int dc = blockIdx.x % 4, h = (blockIdx.x / 4) % NH, a = blockIdx.x / (4 * NH);
  const long t0 = INFO[0], s = INFO[1];
  if (t0 >= nloop) return;
  const int col = h * DH + dc * 8;
  for (long i = threadIdx.x; i < (nloop - t0) * WQ * 2; i += 128) {
    const long row = (long)a * N + t0 * WQ + i / 2;
    const int c = (int)(i % 2) * 4;
    AT* p = DKV + row * ldd + col + c;
    V4<AT>::store(p, add4(V4<AT>::load(p), V4<AT>::load(DQT + row * DM + col + c)));
  }
  const long slab = (long)A * WK * (2 * DM);
  for (int i = threadIdx.x; i < WK * 2; i += 128) {
    const int j = i / 2, c = (i % 2) * 4;
    if (s + j >= N) continue;               // a key past N: invalid, zero gradient
    const long row = (long)a * N + s + j;
    const float* src = DKVT + ((long)a * WK + j) * (2 * DM) + col + c;
    float4 dk = *reinterpret_cast<const float4*>(src), dv = *reinterpret_cast<const float4*>(src + DM);
    for (int w = 1; w < WC; ++w) {
      dk = add4(dk, *reinterpret_cast<const float4*>(src + w * slab));
      dv = add4(dv, *reinterpret_cast<const float4*>(src + w * slab + DM));
    }
    AT* pk = DKV + row * ldd + DM + col + c;
    AT* pv = DKV + row * ldd + 2 * DM + col + c;
    V4<AT>::store(pk, add4(V4<AT>::load(pk), dk));
    V4<AT>::store(pv, add4(V4<AT>::load(pv), dv));
  }
}

// d tb = sum of the SC partials (in order); the LN / projection backward of the tail windows ADDED into DZ; per-block partials of d gamma, d Wb
template <typename ZT>
__global__ void __launch_bounds__(256) tail_bias_bwd_k(const ZT* __restrict__ Z, const float* __restrict__ GAMMA, const float* __restrict__ WB,
                                                       const float* __restrict__ DTBP, int SC, const int64_t* __restrict__ INFO, ZT* __restrict__ DZ,
                                                       float* __restrict__ PART, float eps, int nloop) {
  __shared__ float red[8][PC + NH * PC];
  float dg[PC], dw[NH * PC];
#pragma unroll
  for (int c = 0; c < PC; ++c) dg[c] = 0.f;
#pragma unroll
  for (int c = 0; c < NH * PC; ++c) dw[c] = 0.f;
  const long t0 = INFO[0], rows = (long)nloop * WQ * WK, slab = (long)nloop * NH * WQ * WK;
  for (long r = t0 * (WQ * WK) + (long)blockIdx.x * 256 + threadIdx.x; r < rows; r += (long)gridDim.x * 256) {
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
    for (int h = 0; h < NH; ++h) {
      float t = 0.f;
      for (int sc = 0; sc < SC; ++sc) t += DTBP[sc * slab + (e * NH + h) * (WQ * WK) + rem];
      db[h] = (!(t == t) || isinf(t)) ? 0.f : t;
    }
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
    for (int c = 0; c < PC; ++c) st(DZ + r * PC + c, ld(DZ + r * PC + c) + rstd * (dxh[c] - m1 - z[c] * rstd * m2));
  }
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

// info = [t0, s, n_real] of the tail from the valid-key bytes KM [n] (real atoms first; nullptr: n0 real atoms): one CTA, integer sums
__global__ void tail_info_k(const bool* __restrict__ KM, long n, long n0, int qblock, int64_t* __restrict__ INFO) {
  __shared__ unsigned long long tot;
  if (threadIdx.x == 0) tot = 0ULL;
  __syncthreads();
  unsigned long long c = 0ULL;
  if (KM != nullptr)
    for (long i = threadIdx.x; i < n; i += blockDim.x) c += KM[i] ? 1ULL : 0ULL;
  atomicAdd(&tot, c);
  __syncthreads();
  if (threadIdx.x == 0) {
    const long nreal = KM != nullptr ? (long)tot : n0;
    const long last = nreal >= 1 ? (nreal - 1) / WQ : -1;
    const long rule = qblock ? (nreal + WQ - 1) / WQ * WQ : nreal;
    INFO[0] = last + 1 > 2 ? last + 1 : 2;
    INFO[1] = rule - WK > 0 ? rule - WK : 0;
    INFO[2] = nreal;
  }
}

__global__ void tail_dbias_sum_k(const float* __restrict__ DTBP, int SC, const int64_t* __restrict__ INFO, float* __restrict__ DTB, int nloop) {
  const long per = (long)NH * WQ * WK, slab = (long)nloop * per;
  for (long r = INFO[0] * per + (long)blockIdx.x * blockDim.x + threadIdx.x; r < slab; r += (long)gridDim.x * blockDim.x) {
    float t = 0.f;
    for (int sc = 0; sc < SC; ++sc) t += DTBP[sc * slab + r];
    DTB[r] = t;
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


at::Tensor tail_info(c10::optional<at::Tensor> kmask, int64_t n0, bool qblock, at::Tensor like) {
  // [t0, s, n_real] int64 on the device (local_dit_h100.tail_info); kmask: bool [n] valid keys, real atoms first (None: n0 real atoms)
  const c10::cuda::CUDAGuard guard(like.device());
  auto info = at::empty({3}, like.options().dtype(at::kLong));
  const bool has = kmask.has_value() && kmask->defined();
  if (has) TORCH_CHECK(kmask->scalar_type() == at::kBool && kmask->is_contiguous() && kmask->device() == like.device(), "tail_info: kmask bool, contiguous");
  tail_info_k<<<1, 1024, 0, S_()>>>(has ? CP<bool>(*kmask) : nullptr, has ? kmask->numel() : 0, n0, qblock ? 1 : 0, P<int64_t>(info));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return info;
}

at::Tensor tail_bias_fwd(at::Tensor z, at::Tensor gamma, at::Tensor wb, at::Tensor info, double eps, at::Tensor tb) {
  // z [>= nloop, 32, 128, 16] (its first nloop windows), tb [nloop, 4, 32, 128] fp32 (a contiguous slot; rows below t0 left as they are)
  const c10::cuda::CUDAGuard guard(z.device());
  TORCH_CHECK(z.is_contiguous() && z.size(1) == WQ && z.size(2) == WK && z.size(3) == PC, "tail_bias_fwd: z [W, 32, 128, 16]");
  const long nloop = tb.size(0);
  TORCH_CHECK(tb.is_contiguous() && tb.scalar_type() == at::kFloat && tb.size(1) == NH && nloop <= z.size(0), "tail_bias_fwd: tb [nloop <= W, 4, 32, 128] fp32");
  TORCH_CHECK(info.scalar_type() == at::kLong && info.numel() == 3, "tail_bias_fwd: info int64 [t0, s, n_real]");
  const long rows = nloop * WQ * WK;
  const int blocks = (int)std::max(1L, std::min((rows + 255) / 256, 1024L));
  if (is_bf(z)) tail_bias_fwd_k<bf><<<blocks, 256, 0, S_()>>>(CP<bf>(z), CP<float>(gamma), CP<float>(wb), CP<int64_t>(info), P<float>(tb), (float)eps, rows);
  else tail_bias_fwd_k<float><<<blocks, 256, 0, S_()>>>(CP<float>(z), CP<float>(gamma), CP<float>(wb), CP<int64_t>(info), P<float>(tb), (float)eps, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tb;
}

void tail_attn_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor info, at::Tensor tb, at::Tensor o) {
  // q / k / v / o [A, N, 128]; tb [G, nloop, 4, 32, 128] (sample a reads pair a % G)
  const c10::cuda::CUDAGuard guard(q.device());
  const long N = q.size(1);
  const int A = q.size(0), G = tb.size(0), nloop = tb.size(1);
  TORCH_CHECK(tb.is_contiguous() && tb.dim() == 5 && A % G == 0 && (long)nloop * WQ <= N, "tail_attn_fwd: tb [G, nloop, 4, 32, 128]");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // about 6 CTAs per SM (from the shapes only), two windows per CTA per step
  const int C = std::max(1, std::min((nloop + 1) / 2, (6 * sms + A * NH - 1) / (A * NH)));
  const int blocks = A * NH * C;
  if (is_bf(q)) {
    prep(tail_attn_fwd_k<bf>, SMEM_TC_BF);
    tail_attn_fwd_k<bf><<<blocks, 128, SMEM_TC_BF, S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), CP<int64_t>(info), CP<float>(tb), P<bf>(o), N, nloop, G, C);
  } else {
    prep(tail_attn_fwd_k<float>, SMEM_TC_F);
    tail_attn_fwd_k<float><<<blocks, 128, SMEM_TC_F, S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), CP<int64_t>(info), CP<float>(tb), P<float>(o), N,
                                                                nloop, G, C);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tail_bwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dO, at::Tensor dd, at::Tensor info, at::Tensor tb, at::Tensor dqt, at::Tensor dkvt,
              at::Tensor dtbp) {
  // tb [1, nloop, 4, 32, 128]; dqt like q; dkvt [WC, A, 128, 256] fp32 (WC window chunks); dtbp [SC, nloop, 4, 32, 128] fp32
  const c10::cuda::CUDAGuard guard(q.device());
  const long N = q.size(1);
  const int A = q.size(0), nloop = tb.size(1), SC = dtbp.size(0), WC = dkvt.size(0);
  TORCH_CHECK(tb.size(0) == 1 && dtbp.size(1) == nloop && dkvt.size(1) == A, "tail_bwd: one pair; dtbp [SC, nloop, ...]; dkvt [WC, A, 128, 256]");
  const int b1 = nloop * NH * SC, b2 = A * NH * WC;
  if (is_bf(q)) {
    prep(tail_bwd_dq_k<bf>, smem_bwd());
    prep(tail_bwd_kv_k<bf>, smem_bwd());
    tail_bwd_dq_k<bf><<<b1, 128, smem_bwd(), S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), CP<bf>(dO), CP<int64_t>(info), CP<float>(tb), P<bf>(dqt), P<float>(dtbp),
                                                     N, nloop, A, SC);
    tail_bwd_kv_k<bf><<<b2, 128, smem_bwd(), S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), P<bf>(dO), P<float>(dd), CP<int64_t>(info), CP<float>(tb),
                                                     P<float>(dkvt), N, nloop, A, WC);
  } else {
    prep(tail_bwd_dq_k<float>, smem_bwd());
    prep(tail_bwd_kv_k<float>, smem_bwd());
    tail_bwd_dq_k<float><<<b1, 128, smem_bwd(), S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), CP<float>(dO), CP<int64_t>(info), CP<float>(tb),
                                                        P<float>(dqt), P<float>(dtbp), N, nloop, A, SC);
    tail_bwd_kv_k<float><<<b2, 128, smem_bwd(), S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), P<float>(dO), P<float>(dd), CP<int64_t>(info),
                                                        CP<float>(tb), P<float>(dkvt), N, nloop, A, WC);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tail_bwd_apply(at::Tensor dkv, at::Tensor info, at::Tensor dqt, at::Tensor dkvt, int64_t A, int64_t N, int64_t nloop) {
  const c10::cuda::CUDAGuard guard(dkv.device());
  const int blocks = A * NH * 4;
  if (is_bf(dkv)) tail_bwd_apply_k<bf><<<blocks, 128, 0, S_()>>>(P<bf>(dkv), dkv.stride(0), CP<int64_t>(info), CP<bf>(dqt), CP<float>(dkvt), N, (int)nloop, (int)A, (int)dkvt.size(0));
  else tail_bwd_apply_k<float><<<blocks, 128, 0, S_()>>>(P<float>(dkv), dkv.stride(0), CP<int64_t>(info), CP<float>(dqt), CP<float>(dkvt), N, (int)nloop, (int)A, (int)dkvt.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor tail_bias_bwd(at::Tensor z, at::Tensor gamma, at::Tensor wb, at::Tensor dtbp, at::Tensor info, at::Tensor dz, double eps) {
  // dz (z's shape and dtype) += the tail windows' LN / projection backward of sum_sc dtbp[sc]; -> per-block partials [4 x SMs, 80] (a fixed grid per device: the same partials, in the same order, every call)
  const c10::cuda::CUDAGuard guard(z.device());
  TORCH_CHECK(dz.is_contiguous() && dz.sizes() == z.sizes() && dz.scalar_type() == z.scalar_type(), "tail_bias_bwd: dz like z");
  const int blocks = 4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount, SC = dtbp.size(0), nloop = dtbp.size(1);
  TORCH_CHECK(dtbp.is_contiguous() && nloop <= z.size(0), "tail_bias_bwd: dtbp [SC, nloop <= W, 4, 32, 128]");
  auto part = at::empty({blocks, PC + NH * PC}, z.options().dtype(at::kFloat));
  if (is_bf(z)) tail_bias_bwd_k<bf><<<blocks, 256, 0, S_()>>>(CP<bf>(z), CP<float>(gamma), CP<float>(wb), CP<float>(dtbp), SC, CP<int64_t>(info), P<bf>(dz),
                                                              P<float>(part), (float)eps, nloop);
  else tail_bias_bwd_k<float><<<blocks, 256, 0, S_()>>>(CP<float>(z), CP<float>(gamma), CP<float>(wb), CP<float>(dtbp), SC, CP<int64_t>(info), P<float>(dz),
                                                        P<float>(part), (float)eps, nloop);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return part;
}

at::Tensor tail_dbias_sum(at::Tensor dtbp, at::Tensor info) {
  // [SC, nloop, 4, 32, 128] -> [1, nloop, 4, 32, 128] (the tail windows; rows below t0 unwritten)
  const c10::cuda::CUDAGuard guard(dtbp.device());
  const int SC = dtbp.size(0), nloop = dtbp.size(1);
  auto dtb = at::empty({1, nloop, NH, WQ, WK}, dtbp.options());
  const long n = (long)nloop * NH * WQ * WK;
  const int blocks = (int)std::max(1L, std::min((n + 255) / 256, 1024L));
  tail_dbias_sum_k<<<blocks, 256, 0, S_()>>>(CP<float>(dtbp), SC, CP<int64_t>(info), P<float>(dtb), nloop);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dtb;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_bias_fwd", &edge_bias_fwd);
  m.def("edge_bias_bwd", &edge_bias_bwd);
  m.def("edge_attn_fwd", &edge_attn_fwd);
  m.def("edge_bwd_compute", &edge_bwd_compute);
  m.def("edge_bwd_apply", &edge_bwd_apply);
  m.def("tail_bias_fwd", &tail_bias_fwd);
  m.def("tail_attn_fwd", &tail_attn_fwd);
  m.def("tail_bwd", &tail_bwd);
  m.def("tail_bwd_apply", &tail_bwd_apply);
  m.def("tail_bias_bwd", &tail_bias_bwd);
  m.def("tail_dbias_sum", &tail_dbias_sum);
  m.def("tail_info", &tail_info);
}
