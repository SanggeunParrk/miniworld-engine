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
// smem layout (floats): Q [32][33], K [128][33], V [128][33], S [32][129]
constexpr int QS = 33, KS = 33, SS = 129;
constexpr int SM_Q = 0, SM_K = SM_Q + WQ * QS, SM_V = SM_K + WK * KS, SM_S = SM_V + WK * KS, SM_FLOATS = SM_S + WQ * SS;

template <typename AT>
__device__ __forceinline__ void load_tiles(const AT* Q, const AT* K, const AT* V, const int64_t* keys, const int64_t* widx, int a, int e, int h, long N,
                                           float* sq, float* sk, float* sv) {
  const long w = widx[e];
  for (int i = threadIdx.x; i < WQ * DH; i += blockDim.x) {
    const int q = i / DH, d = i % DH;
    sq[q * QS + d] = ld(Q + ((long)a * N + w * WQ + q) * DM + h * DH + d);
  }
  for (int i = threadIdx.x; i < WK * DH; i += blockDim.x) {
    const int j = i / DH, d = i % DH;
    const long row = (long)a * N + keys[e * WK + j];
    sk[j * KS + d] = ld(K + row * DM + h * DH + d);
    sv[j * KS + d] = ld(V + row * DM + h * DH + d);
  }
}

// scores -> probabilities in S [32][129] (thread = key slot; rows with no valid key get zeros)
__device__ __forceinline__ void scores_probs(const float* sq, const float* sk, const float* EB, int e, int h, float* sS) {
  const int j = threadIdx.x;     // 128 threads
  const float scale = 0.17677669529663687f;
  for (int q = 0; q < WQ; ++q) {
    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) acc += sq[q * QS + d] * sk[j * KS + d];
    sS[q * SS + j] = acc * scale + EB[((long)e * NH + h) * (WQ * WK) + q * WK + j];
  }
  __syncthreads();
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;      // warp w owns rows w, w + 4, ...
  for (int q = w; q < WQ; q += 4) {
    float m = -INFINITY;
    for (int jj = l; jj < WK; jj += 32) m = fmaxf(m, sS[q * SS + jj]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float s = 0.f;
    for (int jj = l; jj < WK; jj += 32) {
      const float p = (m == -INFINITY) ? 0.f : __expf(sS[q * SS + jj] - m);
      sS[q * SS + jj] = p;
      s += p;
    }
    s = warp_sum(s);
    const float inv = s > 0.f ? 1.f / s : 0.f;
    for (int jj = l; jj < WK; jj += 32) sS[q * SS + jj] *= inv;
  }
  __syncthreads();
}

template <typename AT>
__global__ void __launch_bounds__(128) edge_attn_fwd_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V,
                                                       const int64_t* __restrict__ keys, const int64_t* __restrict__ widx, const float* __restrict__ EB,
                                                       AT* __restrict__ O, long N) {
  extern __shared__ float sm[];
  const int h = blockIdx.x % NH, e = (blockIdx.x / NH) % 5, a = blockIdx.x / (NH * 5);
  load_tiles<AT>(Q, K, V, keys, widx, a, e, h, N, sm + SM_Q, sm + SM_K, sm + SM_V);
  __syncthreads();
  scores_probs(sm + SM_Q, sm + SM_K, EB, e, h, sm + SM_S);
  const float* sS = sm + SM_S;
  const float* sv = sm + SM_V;
  const long w = widx[e];
  // out[q][d] = sum_j P[q][j] V[j][d]: 1024 outputs, 8 per thread
  for (int o = threadIdx.x; o < WQ * DH; o += 128) {
    const int q = o / DH, d = o % DH;
    float acc = 0.f;
    for (int j = 0; j < WK; ++j) acc += sS[q * SS + j] * sv[j * KS + d];
    st(O + ((long)a * N + w * WQ + q) * DM + h * DH + d, acc);
  }
}

// compute phase of the backward: dq_e [A, 5, 32, 128], dk_e / dv_e [A, 5, 128, 128] (fp32, columns h 32 + d), debias partial [A, 5, 4, 32, 128];
// zeroes the edge rows of dO (AT [A N, 128]) and of dd [A, 4, N]
template <typename AT>
__global__ void __launch_bounds__(128) edge_bwd_compute_k(const AT* __restrict__ Q, const AT* __restrict__ K, const AT* __restrict__ V, AT* __restrict__ DO,
                                                          float* __restrict__ DD, const int64_t* __restrict__ keys, const int64_t* __restrict__ widx,
                                                          const float* __restrict__ EB, float* __restrict__ DQE, float* __restrict__ DKE, float* __restrict__ DVE,
                                                          float* __restrict__ DBP, long N) {
  extern __shared__ float sm[];
  const int h = blockIdx.x % NH, e = (blockIdx.x / NH) % 5, a = blockIdx.x / (NH * 5);
  const long w = widx[e];
  const bool dup = dup_window(widx, e);
  float* sq = sm + SM_Q; float* sk = sm + SM_K; float* sv = sm + SM_V; float* sP = sm + SM_S;
  load_tiles<AT>(Q, K, V, keys, widx, a, e, h, N, sq, sk, sv);
  // dO tile of this head into registers-free smem: reuse the Q region after scores (Q is needed again for dK) -> separate tile
  __shared__ float sdo[WQ * QS];
  for (int i = threadIdx.x; i < WQ * DH; i += 128) {
    const int q = i / DH, d = i % DH;
    sdo[q * QS + d] = dup ? 0.f : ld(DO + ((long)a * N + w * WQ + q) * DM + h * DH + d);
  }
  __syncthreads();
  scores_probs(sq, sk, EB, e, h, sP);
  // dP[q][j] = sum_d dO[q][d] V[j][d] (thread = j), kept in a second [32][129] tile after P: reuse registers per q
  __shared__ float sdP[WQ * SS];
  {
    const int j = threadIdx.x;
    for (int q = 0; q < WQ; ++q) {
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < DH; ++d) acc += sdo[q * QS + d] * sv[j * KS + d];
      sdP[q * SS + j] = acc;
    }
  }
  __syncthreads();
  // dS = P (dP - rowsum(P dP)): rows by warps
  {
    const int wp = threadIdx.x / 32, l = threadIdx.x % 32;
    for (int q = wp; q < WQ; q += 4) {
      float s = 0.f;
      for (int jj = l; jj < WK; jj += 32) s += sP[q * SS + jj] * sdP[q * SS + jj];
      s = warp_sum(s);
      for (int jj = l; jj < WK; jj += 32) sdP[q * SS + jj] = sP[q * SS + jj] * (sdP[q * SS + jj] - s);   // dS in sdP
    }
  }
  __syncthreads();
  const float scale = 0.17677669529663687f;
  const long eo = (long)(a * 5 + e);
  // dV[j][d] = sum_q P[q][j] dO[q][d]; dK[j][d] = scale sum_q dS[q][j] Q[q][d]   (thread = key slot j)
  {
    const int j = threadIdx.x;
    for (int d = 0; d < DH; ++d) {
      float av = 0.f, ak = 0.f;
      for (int q = 0; q < WQ; ++q) { av += sP[q * SS + j] * sdo[q * QS + d]; ak += sdP[q * SS + j] * sq[q * QS + d]; }
      DVE[(eo * WK + j) * DM + h * DH + d] = dup ? 0.f : av;
      DKE[(eo * WK + j) * DM + h * DH + d] = dup ? 0.f : ak * scale;
    }
  }
  // dQ[q][d] = scale sum_j dS[q][j] K[j][d]
  for (int o = threadIdx.x; o < WQ * DH; o += 128) {
    const int q = o / DH, d = o % DH;
    float acc = 0.f;
    for (int j = 0; j < WK; ++j) acc += sdP[q * SS + j] * sk[j * KS + d];
    DQE[(eo * WQ + q) * DM + h * DH + d] = dup ? 0.f : acc * scale;
  }
  // dbias partial: dS itself ([32][128] of (a, e, h))
  for (int o = threadIdx.x; o < WQ * WK; o += 128) DBP[((eo * NH) + h) * (WQ * WK) + o] = dup ? 0.f : sdP[(o / WK) * SS + (o % WK)];
  // the main kernel sees no gradient for the edge rows (a duplicate window leaves them to the block that owns the window)
  if (!dup) {
    for (int i = threadIdx.x; i < WQ * DH; i += 128) {
      const int q = i / DH, d = i % DH;
      st(DO + ((long)a * N + w * WQ + q) * DM + h * DH + d, 0.f);
    }
    for (int q = threadIdx.x; q < WQ; q += 128) DD[((long)a * NH + h) * N + w * WQ + q] = 0.f;
  }
}

// apply phase: dq (rows of the window; the main kernel wrote zeros there), dk / dv (rows of the keys; windows may share keys, so one block
// owns all five windows of (a, h) and adds them one after the other)
template <typename AT>
__global__ void __launch_bounds__(128) edge_bwd_apply_k(AT* __restrict__ DKV, long ldd, const int64_t* __restrict__ keys, const int64_t* __restrict__ widx,
                                                        const float* __restrict__ DQE, const float* __restrict__ DKE, const float* __restrict__ DVE, long N) {
  const int h = blockIdx.x % NH, a = blockIdx.x / NH;
  for (int e = 0; e < 5; ++e) {
    if (dup_window(widx, e)) continue;
    const long w = widx[e], eo = (long)(a * 5 + e);
    for (int i = threadIdx.x; i < WQ * DH; i += 128) {
      const int q = i / DH, d = i % DH;
      AT* p = DKV + ((long)a * N + w * WQ + q) * ldd + h * DH + d;          // dq columns 0 ..
      st(p, ld(p) + DQE[(eo * WQ + q) * DM + h * DH + d]);
    }
    for (int i = threadIdx.x; i < WK * DH; i += 128) {
      const int j = i / DH, d = i % DH;
      const long row = (long)a * N + keys[e * WK + j];
      AT* pk = DKV + row * ldd + DM + h * DH + d;
      AT* pv = DKV + row * ldd + 2 * DM + h * DH + d;
      st(pk, ld(pk) + DKE[(eo * WK + j) * DM + h * DH + d]);
      st(pv, ld(pv) + DVE[(eo * WK + j) * DM + h * DH + d]);
    }
    __syncthreads();
  }
}

c10::cuda::CUDAStream S_() { return c10::cuda::getCurrentCUDAStream(); }
template <typename T> T* P(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* CP(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
bool is_bf(const at::Tensor& t) { return t.scalar_type() == at::kBFloat16; }
size_t smem_bytes() { return (size_t)SM_FLOATS * 4; }
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
    prep(edge_attn_fwd_k<bf>, smem_bytes());
    edge_attn_fwd_k<bf><<<blocks, 128, smem_bytes(), S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), CP<int64_t>(keys), CP<int64_t>(widx), CP<float>(ebias), P<bf>(o), N);
  } else {
    prep(edge_attn_fwd_k<float>, smem_bytes());
    edge_attn_fwd_k<float><<<blocks, 128, smem_bytes(), S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), CP<int64_t>(keys), CP<int64_t>(widx), CP<float>(ebias),
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
    prep(edge_bwd_compute_k<bf>, smem_bytes());
    edge_bwd_compute_k<bf><<<blocks, 128, smem_bytes(), S_()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), P<bf>(dO), P<float>(dd), CP<int64_t>(keys), CP<int64_t>(widx),
        CP<float>(ebias), P<float>(dqe), P<float>(dke), P<float>(dve), P<float>(dbp), N);
  } else {
    prep(edge_bwd_compute_k<float>, smem_bytes());
    edge_bwd_compute_k<float><<<blocks, 128, smem_bytes(), S_()>>>(CP<float>(q), CP<float>(k), CP<float>(v), P<float>(dO), P<float>(dd), CP<int64_t>(keys),
        CP<int64_t>(widx), CP<float>(ebias), P<float>(dqe), P<float>(dke), P<float>(dve), P<float>(dbp), N);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void edge_bwd_apply(at::Tensor dkv, at::Tensor keys, at::Tensor widx, at::Tensor dqe, at::Tensor dke, at::Tensor dve, int64_t A, int64_t N) {
  const c10::cuda::CUDAGuard guard(dkv.device());
  const int blocks = A * NH;
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
