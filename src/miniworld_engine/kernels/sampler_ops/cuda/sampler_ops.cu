// sampler_ops.cu -- the rigid-motion glue around an EDM denoiser step (fp32 coordinates [B, N, 3], one rigid motion per sample b).
//
//   augment_noise   coords -> centre (mask-weighted mean), random rotation (Gram-Schmidt of two normal vectors) and translation,
//                   mask, then re-noise: noisy = aug + magnitude * noise. The churn arithmetic (sigma_hat, the noise magnitude) runs on
//                   the device scalars sigma / sigma_next and sigma_hat is written to ``sc`` for the denoiser.
//   align_euler     Kabsch-align ``noisy`` onto ``denoised`` (weighted, Horn's quaternion method: the dominant eigenvector of the 4x4
//                   N(H) by 12 squarings, no SVD), then the Euler update  aligned + step_scale (sigma_next - sigma_hat) (aligned - denoised) / sigma_hat.
//   pool_atoms      the atom encoder's pooling: out[s, t] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)
//                  (x bf16 [S, Q, C] atom features in the queries layout, idx / g the gather of the token-atoms layout, m the atom mask),
//                  rounded to bf16: the gather, relu and masked mean as one pass over the atoms' rows.
//
// The first two are a pass over per-chunk partial sums ([B, C] blocks of 256 / 512 / 1024 atoms) and a pass that reduces the C partials of its sample
// again in every block, so no atomics and no host read; the kernels of an op are chained with programmatic dependent launch. The arithmetic is that of the torch path (utils.rigid_align, augmentation) in
// fp32; no fast-math, no contraction where the torch path rounds between operations.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cuda_bf16.h>

namespace {
constexpr int THREADS = 256;
constexpr int CHUNK = THREADS * 4;   // atoms per block
constexpr int PART = 16;             // floats per partial record

struct Churn {
  int window;                // 0: gamma_0 where sigma_next > gamma_min (AF3); 1: churn_gamma inside [sigma_min, sigma_max] (EDM)
  float gamma, gamma_min, sigma_min, sigma_max;
  float noise_scale, variance_floor, step_scale;
  int clamp;
};

// Programmatic dependent launch: a kernel of a chain starts its (input-only) prologue while the previous kernel drains, and ``pdl_wait`` blocks until that
// kernel is complete and its writes are visible. A kernel launched without the attribute (the first of a chain) makes ``pdl_wait`` a no-op.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t stream, bool dependent, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = dependent ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

// Atoms per block of the partial-sum passes: 1024 (K = 4 per thread) unless the launch is smaller than two waves of blocks, then halved down to 256.
int chunk_for(int64_t n, int64_t b) {
  int chunk = CHUNK;
  while (chunk > 256 && b * ((n + chunk - 1) / chunk) < 264) chunk /= 2;
  return chunk;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sums NV (<= 16) values per thread over the block; the result is in red[0..NV) after the call.
template <int NV>
__device__ __forceinline__ void block_sums(float (&v)[NV], float* red /* [NV][8] */) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const float s = warp_sum(v[k]);
    if (lane == 0) red[k * 8 + warp] = s;
  }
  __syncthreads();
  if (threadIdx.x < NV) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) s += red[threadIdx.x * 8 + w];
    red[threadIdx.x * 8] = s;
  }
  __syncthreads();
}

// sigma_hat and the noise magnitude of the churn step (the torch path's float32 arithmetic, operation for operation).
__device__ __forceinline__ void churn_scalars(const Churn& c, float sigma, float sigma_next, float& sigma_hat, float& magnitude) {
  const bool on = c.window ? (sigma >= c.sigma_min && sigma <= c.sigma_max) : (sigma_next > c.gamma_min);
  const float factor = on ? c.gamma : 1.0f;   // c.gamma holds float(1 + gamma)
  sigma_hat = __fmul_rn(sigma, factor);
  float variance = __fsub_rn(__fmul_rn(sigma_hat, sigma_hat), __fmul_rn(sigma, sigma));
  if (c.clamp) variance = fmaxf(variance, c.variance_floor);
  magnitude = __fmul_rn(c.noise_scale, __fsqrt_rn(variance));
}

// Sums the chunk partials of sample ``b`` in chunk order, one value per thread: values 0 .. N1 of the records of ``src1`` and (if N2 > 0) values 0 .. N2 of
// ``src2``; the result of value v is in ``out[v]``. The records are staged in shared memory (one round trip of independent loads per tile instead of a dependent
// chain over the chunks), then thread v adds its column in order, so the sum is the serial one of a single thread.
constexpr int TILE = 128;
template <int N1, int N2>
__device__ __forceinline__ void sum_partials(const float* src1, const float* src2, int b, int nchunks, float* buf /* [TILE][N1 + N2] */, float* out) {
  constexpr int NV = N1 + N2;
  float acc = 0.f;
  for (int c0 = 0; c0 < nchunks; c0 += TILE) {
    const int cnt = min(TILE, nchunks - c0);
    for (int e = threadIdx.x; e < cnt * NV; e += THREADS) {
      const int q = e / NV, v = e - q * NV;
      const size_t rec = ((size_t)b * nchunks + c0 + q) * PART;
      buf[e] = (v < N1) ? src1[rec + v] : src2[rec + v - N1];
    }
    __syncthreads();
    if (threadIdx.x < NV) {
#pragma unroll 16
      for (int q = 0; q < cnt; ++q) acc += buf[q * NV + threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.x < NV) out[threadIdx.x] = acc;
  __syncthreads();
}

// ---- pass 1: sum_n w, sum_n w p (one tensor) -> part[b][c][0..3]. K atoms per thread, CHUNK = THREADS * K atoms per block.
template <int K>
__global__ void moments_k(const float* __restrict__ p, const float* __restrict__ w, float* part, int n) {
  __shared__ float red[4 * 8];
  pdl_trigger();
  const int b = blockIdx.y, base = blockIdx.x * (THREADS * K);
  float wv[K], pv[K][3];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
    wv[k] = 0.f; pv[k][0] = pv[k][1] = pv[k][2] = 0.f;
    if (i < n) {
      wv[k] = w[i];
      const float* q = p + ((size_t)b * n + i) * 3;
      pv[k][0] = q[0]; pv[k][1] = q[1]; pv[k][2] = q[2];
    }
  }
  float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < K; ++k) {
    if (base + k * THREADS + threadIdx.x < n) {
      const float wi = wv[k];
      v[0] += wi; v[1] += wi * pv[k][0]; v[2] += wi * pv[k][1]; v[3] += wi * pv[k][2];
    }
  }
  block_sums<4>(v, red);
  if (threadIdx.x < 4) part[((size_t)b * gridDim.x + blockIdx.x) * PART + threadIdx.x] = red[threadIdx.x * 8];
}

// ---- augment + re-noise. The coordinates, mask and noise of the block's atoms and the rotation frame / churn scalars do not depend on the partials: they are
// read (and computed) before ``pdl_wait``; the partials of the sample are then summed by four threads (one value each, in chunk order).
template <int K>
__global__ void augment_noise_k(const float* __restrict__ p, const float* __restrict__ w, const float* __restrict__ part,
                                const float* __restrict__ rv, const float* __restrict__ tr, const float* __restrict__ noise,
                                const float* __restrict__ sigma, const float* __restrict__ sigma_next, Churn cfg, float* noisy,
                                float* sc, int n, int nchunks) {
  __shared__ float s_sum[4], s_rot[9], s_t[3], s_mag, s_buf[TILE * 4];
  const int b = blockIdx.y;
  float pv[K][3], nv[K][3], wv[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = blockIdx.x * (THREADS * K) + k * THREADS + threadIdx.x;
    if (i < n) {
      const size_t o = ((size_t)b * n + i) * 3;
      pv[k][0] = p[o]; pv[k][1] = p[o + 1]; pv[k][2] = p[o + 2];
      nv[k][0] = noise[o]; nv[k][1] = noise[o + 1]; nv[k][2] = noise[o + 2];
      wv[k] = w[i];
    }
  }
  if (threadIdx.x == 0) {
    const float* v = rv + b * 6;   // v0 = v[0..3), v1 = v[3..6)
    float n0 = fmaxf(sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1e-10f);
    const float e0x = v[0] / n0, e0y = v[1] / n0, e0z = v[2] / n0;
    const float d = v[3] * e0x + v[4] * e0y + v[5] * e0z;
    const float ux = v[3] - e0x * d, uy = v[4] - e0y * d, uz = v[5] - e0z * d;
    const float n1 = fmaxf(sqrtf(ux * ux + uy * uy + uz * uz), 1e-10f);
    const float e1x = ux / n1, e1y = uy / n1, e1z = uz / n1;
    s_rot[0] = e0x; s_rot[1] = e0y; s_rot[2] = e0z;
    s_rot[3] = e1x; s_rot[4] = e1y; s_rot[5] = e1z;
    s_rot[6] = e0y * e1z - e0z * e1y; s_rot[7] = e0z * e1x - e0x * e1z; s_rot[8] = e0x * e1y - e0y * e1x;
    s_t[0] = tr[b * 3]; s_t[1] = tr[b * 3 + 1]; s_t[2] = tr[b * 3 + 2];
    float sh, mag;
    churn_scalars(cfg, *sigma, *sigma_next, sh, mag);
    s_mag = mag;
    if (blockIdx.x == 0 && b == 0) { sc[0] = sh; sc[1] = mag; }
  }
  pdl_wait();
  sum_partials<4, 0>(part, part, b, nchunks, s_buf, s_sum);
  const float tot = fmaxf(s_sum[0], 1e-6f);
  const float c0 = s_sum[1] / tot, c1 = s_sum[2] / tot, c2 = s_sum[3] / tot;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = blockIdx.x * (THREADS * K) + k * THREADS + threadIdx.x;
    if (i >= n) continue;
    const size_t o = ((size_t)b * n + i) * 3;
    const float x = pv[k][0] - c0, y = pv[k][1] - c1, z = pv[k][2] - c2;
    const float wi = wv[k];
    float out[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float a = x * s_rot[j] + y * s_rot[3 + j] + z * s_rot[6 + j];
      out[j] = __fadd_rn(__fmul_rn(__fadd_rn(a, s_t[j]), wi), __fmul_rn(s_mag, nv[k][j]));
    }
    noisy[o] = out[0]; noisy[o + 1] = out[1]; noisy[o + 2] = out[2];
  }
}

// ---- Kabsch: moments of both tensors, covariance, rotation
template <int K>
__global__ void moments2_k(const float* __restrict__ m, const float* __restrict__ r, const float* __restrict__ w, float* part, int n) {
  __shared__ float red[7 * 8];
  pdl_trigger();
  const int b = blockIdx.y, base = blockIdx.x * (THREADS * K);
  float wv[K], mv[K][3], rv[K][3];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
    wv[k] = 0.f;
    if (i < n) {
      wv[k] = w[i];
      const size_t o = ((size_t)b * n + i) * 3;
      mv[k][0] = m[o]; mv[k][1] = m[o + 1]; mv[k][2] = m[o + 2];
      rv[k][0] = r[o]; rv[k][1] = r[o + 1]; rv[k][2] = r[o + 2];
    }
  }
  float v[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < K; ++k) {
    if (base + k * THREADS + threadIdx.x < n) {
      const float wi = wv[k];
      v[0] += wi;
      v[1] += wi * mv[k][0]; v[2] += wi * mv[k][1]; v[3] += wi * mv[k][2];
      v[4] += wi * rv[k][0]; v[5] += wi * rv[k][1]; v[6] += wi * rv[k][2];
    }
  }
  block_sums<7>(v, red);
  if (threadIdx.x < 7) part[((size_t)b * gridDim.x + blockIdx.x) * PART + threadIdx.x] = red[threadIdx.x * 8];
}

template <int K>
__global__ void cov_k(const float* __restrict__ m, const float* __restrict__ r, const float* __restrict__ w,
                      const float* __restrict__ part, float* cov, int n, int nchunks) {
  __shared__ float s_sum[8], red[9 * 8], s_buf[TILE * 7];
  pdl_trigger();
  const int b = blockIdx.y;
  float wv[K], mv[K][3], rv[K][3];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = blockIdx.x * (THREADS * K) + k * THREADS + threadIdx.x;
    if (i < n) {
      const size_t o = ((size_t)b * n + i) * 3;
      wv[k] = w[i];
      mv[k][0] = m[o]; mv[k][1] = m[o + 1]; mv[k][2] = m[o + 2];
      rv[k][0] = r[o]; rv[k][1] = r[o + 1]; rv[k][2] = r[o + 2];
    }
  }
  pdl_wait();
  sum_partials<7, 0>(part, part, b, nchunks, s_buf, s_sum);   // (sum w, sum w m_xyz, sum w r_xyz)
  const float tot = s_sum[0];
  float cm[3], cr[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) { cm[j] = s_sum[1 + j] / tot; cr[j] = s_sum[4 + j] / tot; }
  float v[9];
#pragma unroll
  for (int j = 0; j < 9; ++j) v[j] = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = blockIdx.x * (THREADS * K) + k * THREADS + threadIdx.x;
    if (i < n) {
      const float wi = wv[k];
      float a[3], c[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) { a[j] = (mv[k][j] - cm[j]) * wi; c[j] = rv[k][j] - cr[j]; }
#pragma unroll
      for (int p = 0; p < 3; ++p)
#pragma unroll
        for (int q = 0; q < 3; ++q) v[p * 3 + q] += a[p] * c[q];
    }
  }
  block_sums<9>(v, red);
  if (threadIdx.x < 9) cov[((size_t)b * gridDim.x + blockIdx.x) * PART + threadIdx.x] = red[threadIdx.x * 8];
}

// R (row-vector convention: aligned = a R) of the covariance h[9], by Horn's method; run by warp 0 (lanes 0..15 = the 4x4 entries).
__device__ void horn_rotation(const float* h, float* rot /* [9] shared out */, float* nm /* [16] shared */) {
  const int lane = threadIdx.x;   // warp 0 only
  if (lane == 0) {
    const float sxx = h[0], sxy = h[1], sxz = h[2], syx = h[3], syy = h[4], syz = h[5], szx = h[6], szy = h[7], szz = h[8];
    const float t[16] = {sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
                         syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
                         szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
                         sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz};
#pragma unroll
    for (int k = 0; k < 16; ++k) nm[k] = t[k];
  }
  __syncwarp();
  const int i = (lane >> 2) & 3, j = lane & 3;
  float e = nm[(lane & 15)];
  // shift = |N|_F; m = N + shift I
  float sq = (lane < 16) ? e * e : 0.f;
  sq = warp_sum(sq);
  const float shift = sqrtf(sq);
  float m = e + ((i == j) ? shift : 0.f);
  for (int it = 0; it < 12; ++it) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) s = fmaf(__shfl_sync(0xffffffffu, m, (i << 2) | k), __shfl_sync(0xffffffffu, m, (k << 2) | j), s);
    // max |s| over the 16 entries: every lane gathers all 16 and reduces locally (the shuffles overlap; the xor tree of shuffles was 5 dependent rounds)
    float sv[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) sv[q] = fabsf(__shfl_sync(0xffffffffu, s, q));
    float am = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3])), fmaxf(fmaxf(sv[4], sv[5]), fmaxf(sv[6], sv[7])));
    am = fmaxf(am, fmaxf(fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[10], sv[11])), fmaxf(fmaxf(sv[12], sv[13]), fmaxf(sv[14], sv[15]))));
    m = s / fmaxf(am, FLT_MIN);
  }
  // pick the column of largest norm
  float colsq = (lane < 16) ? m * m : 0.f;
  float norms[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    float t = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) t += __shfl_sync(0xffffffffu, colsq, (r << 2) | c);
    norms[c] = t;
  }
  int pick = 0;
#pragma unroll
  for (int c = 1; c < 4; ++c) if (norms[c] > norms[pick]) pick = c;
  float q[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) q[r] = __shfl_sync(0xffffffffu, m, (r << 2) | pick);
  const float qn = fmaxf(sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), FLT_MIN);
  if (lane == 0) {
    const float w = q[0] / qn, x = q[1] / qn, y = q[2] / qn, z = q[3] / qn;
    const float Q[9] = {w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (y * x + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
                        2 * (z * x - w * y), 2 * (z * y + w * x), w * w - x * x - y * y + z * z};
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
      for (int c = 0; c < 3; ++c) rot[a * 3 + c] = Q[c * 3 + a];   // transpose
  }
  __syncwarp();
}

template <int K>
__global__ void align_euler_k(const float* __restrict__ m, const float* __restrict__ r, const float* __restrict__ part,
                              const float* __restrict__ cov, const float* __restrict__ sc, const float* __restrict__ sigma_next,
                              float step_scale, float* out, int n, int nchunks) {
  __shared__ float s_sum[16], s_rot[9], s_nm[16], s_k, s_sh, s_buf[TILE * 16];
  const int b = blockIdx.y;
  float mv[K][3], rv[K][3];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = blockIdx.x * (THREADS * K) + k * THREADS + threadIdx.x;
    if (i < n) {
      const size_t o = ((size_t)b * n + i) * 3;
      mv[k][0] = m[o]; mv[k][1] = m[o + 1]; mv[k][2] = m[o + 2];
      rv[k][0] = r[o]; rv[k][1] = r[o + 1]; rv[k][2] = r[o + 2];
    }
  }
  if (threadIdx.x == 0) {
    const float sh = sc[0];
    s_sh = sh;
    s_k = __fmul_rn(step_scale, __fsub_rn(*sigma_next, sh));
  }
  pdl_wait();
  sum_partials<7, 9>(part, cov, b, nchunks, s_buf, s_sum);   // 0..6: the centre sums of the first pass, 7..15: the nine covariance sums
  if (threadIdx.x < 32) horn_rotation(s_sum + 7, s_rot, s_nm);
  __syncthreads();
  const float tot = s_sum[0];
  float cm[3], cr[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) { cm[j] = s_sum[1 + j] / tot; cr[j] = s_sum[4 + j] / tot; }
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = blockIdx.x * (THREADS * K) + k * THREADS + threadIdx.x;
    if (i >= n) continue;
    const size_t o = ((size_t)b * n + i) * 3;
    const float x = mv[k][0] - cm[0], y = mv[k][1] - cm[1], z = mv[k][2] - cm[2];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float al = __fadd_rn(x * s_rot[j] + y * s_rot[3 + j] + z * s_rot[6 + j], cr[j]);
      const float dir = __fdiv_rn(__fsub_rn(al, rv[k][j]), s_sh);
      out[o + j] = __fadd_rn(al, __fmul_rn(s_k, dir));
    }
  }
}

// ---- encoder pooling. One block per (token, sample); a thread owns 4 consecutive channels (C % 4 == 0). The atoms' (mask, gather mask, row) are loaded
// one atom per lane and handed to the warp by shuffles, so the rows of up to 32 atoms are requested at once (the loop of the generic kernel below is a
// dependent chain of two loads per atom).
__device__ __forceinline__ float ldf(const void* p, int code, long i) {
  switch (code) {
    case 0: return ((const float*)p)[i];
    case 1: return ((const bool*)p)[i] ? 1.f : 0.f;
    case 2: return (float)((const int*)p)[i];
    case 3: return (float)((const long long*)p)[i];
    default: return __bfloat162float(((const __nv_bfloat16*)p)[i]);
  }
}

template <int CODE>
__device__ __forceinline__ float ldt(const void* p, long i) {
  if constexpr (CODE == 0) return ((const float*)p)[i];
  else if constexpr (CODE == 1) return ((const bool*)p)[i] ? 1.f : 0.f;
  else if constexpr (CODE == 2) return (float)((const int*)p)[i];
  else return (float)((const long long*)p)[i];
}

constexpr int NO_ROW = INT_MIN;

template <int IC, int GC, int MC>
__global__ void pool_atoms_fast_k(const __nv_bfloat16* __restrict__ x, const void* idx, const void* gm, const void* am, __nv_bfloat16* out, int atoms,
                                  int tokens, int rows, int channels) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x * 4, lane = threadIdx.x & 31;
  const bool live = c < channels;
  const __nv_bfloat16* xs = x + (size_t)s * rows * channels + c;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, den = 0.f;
  for (int g0 = 0; g0 < atoms; g0 += 32) {
    float ml = 0.f;
    int rl = NO_ROW;
    if (g0 + lane < atoms) {
      const long k = (long)t * atoms + g0 + lane;
      ml = ldt<MC>(am, k);
      const float gl = ldt<GC>(gm, k);
      if (ml != 0.f && gl != 0.f) rl = (int)(long)ldt<IC>(idx, k);
    }
    const unsigned use = __ballot_sync(0xffffffffu, rl != NO_ROW);   // bit u: atom g0 + u is gathered
    uint2 raw[32];
    float mu[32];
#pragma unroll
    for (int u = 0; u < 32; ++u) {
      mu[u] = __shfl_sync(0xffffffffu, ml, u);
      const int r = __shfl_sync(0xffffffffu, rl, u);
      raw[u] = make_uint2(0u, 0u);
      if (live && r != NO_ROW) raw[u] = *reinterpret_cast<const uint2*>(xs + (size_t)r * channels);
    }
#pragma unroll
    for (int u = 0; u < 32; ++u) {
      den += mu[u];
      const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&raw[u].x), v1 = *reinterpret_cast<const __nv_bfloat162*>(&raw[u].y);
      if ((use >> u) & 1u) {
        acc[0] += mu[u] * fmaxf(__bfloat162float(v0.x), 0.f); acc[1] += mu[u] * fmaxf(__bfloat162float(v0.y), 0.f);
        acc[2] += mu[u] * fmaxf(__bfloat162float(v1.x), 0.f); acc[3] += mu[u] * fmaxf(__bfloat162float(v1.y), 0.f);
      }
    }
  }
  if (!live) return;
  const float inv = fmaxf(den, 1e-10f);
  __nv_bfloat162 o0 = __floats2bfloat162_rn(acc[0] / inv, acc[1] / inv), o1 = __floats2bfloat162_rn(acc[2] / inv, acc[3] / inv);
  uint2 packed;
  packed.x = *reinterpret_cast<unsigned*>(&o0); packed.y = *reinterpret_cast<unsigned*>(&o1);
  *reinterpret_cast<uint2*>(out + ((size_t)s * tokens + t) * channels + c) = packed;
}


// Wide variant (channels % 8 == 0, the one launched): a thread owns 8 consecutive channels (one 16-byte load per atom row), the rows of SUB atoms are requested at once; the atom loop runs over
// the real atom count (no 32-wide padding), the channel order of the sums is the atom order, exactly as pool_atoms_fast_k. SUB = 8 (was 4):
// a 24-atom token is 3 rounds of row loads instead of 6 (latency bound at L2-cold token counts; H100 L1024 9.7 -> 9.4 us, same bits).
template <int IC, int GC, int MC, int SUB = 8>
__global__ void pool_atoms_wide_k(const __nv_bfloat16* __restrict__ x, const void* idx, const void* gm, const void* am, __nv_bfloat16* out, int atoms,
                                  int tokens, int rows, int channels) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x * 8, lane = threadIdx.x & 31;
  const bool live = c < channels;
  const __nv_bfloat16* xs = x + (size_t)s * rows * channels + c;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, den = 0.f;
  for (int g0 = 0; g0 < atoms; g0 += 32) {
    float ml = 0.f;
    int rl = NO_ROW;
    if (g0 + lane < atoms) {
      const long k = (long)t * atoms + g0 + lane;
      ml = ldt<MC>(am, k);
      const float gl = ldt<GC>(gm, k);
      const int ri = (int)(long)ldt<IC>(idx, k);   // read with the masks (not behind them): one memory round trip, not two
      if (ml != 0.f && gl != 0.f) rl = ri;
    }
    const int nvalid = min(32, atoms - g0);
    for (int u0 = 0; u0 < nvalid; u0 += SUB) {
      uint4 raw[SUB];
      float mu[SUB];
      bool use[SUB];
#pragma unroll
      for (int u = 0; u < SUB; ++u) {
        mu[u] = __shfl_sync(0xffffffffu, ml, (u0 + u) & 31);
        const int r = __shfl_sync(0xffffffffu, rl, (u0 + u) & 31);
        use[u] = (u0 + u < nvalid) && r != NO_ROW;
        raw[u] = make_uint4(0u, 0u, 0u, 0u);
        if (live && use[u]) raw[u] = *reinterpret_cast<const uint4*>(xs + (size_t)r * channels);
      }
#pragma unroll
      for (int u = 0; u < SUB; ++u) {
        if (u0 + u >= nvalid) continue;
        den += mu[u];
        if (use[u]) {
          const unsigned w[4] = {raw[u].x, raw[u].y, raw[u].z, raw[u].w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            acc[2 * q] += mu[u] * fmaxf(__uint_as_float(w[q] << 16), 0.f);
            acc[2 * q + 1] += mu[u] * fmaxf(__uint_as_float(w[q] & 0xffff0000u), 0.f);
          }
        }
      }
    }
  }
  if (!live) return;
  const float inv = fmaxf(den, 1e-10f);
  uint4 packed;
  unsigned* pw = reinterpret_cast<unsigned*>(&packed);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __nv_bfloat162 o = __floats2bfloat162_rn(acc[2 * q] / inv, acc[2 * q + 1] / inv);
    pw[q] = *reinterpret_cast<unsigned*>(&o);
  }
  *reinterpret_cast<uint4*>(out + ((size_t)s * tokens + t) * channels + c) = packed;
}
__global__ void pool_atoms_k(const __nv_bfloat16* __restrict__ x, const void* idx, int idx_code, const void* gm, int gm_code, const void* am,
                             int am_code, __nv_bfloat16* out, int atoms, int tokens, int rows, int channels) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x * 4;
  if (c >= channels) return;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, den = 0.f;
  for (int a = 0; a < atoms; ++a) {
    const long k = (long)t * atoms + a;
    const float m = ldf(am, am_code, k);
    den += m;
    if (m == 0.f || ldf(gm, gm_code, k) == 0.f) continue;
    const long r = (long)ldf(idx, idx_code, k);
    const uint2 raw = *reinterpret_cast<const uint2*>(x + ((size_t)s * rows + r) * channels + c);
    const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), v1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    acc[0] += m * fmaxf(__bfloat162float(v0.x), 0.f); acc[1] += m * fmaxf(__bfloat162float(v0.y), 0.f);
    acc[2] += m * fmaxf(__bfloat162float(v1.x), 0.f); acc[3] += m * fmaxf(__bfloat162float(v1.y), 0.f);
  }
  const float inv = fmaxf(den, 1e-10f);
  __nv_bfloat162 o0 = __floats2bfloat162_rn(acc[0] / inv, acc[1] / inv), o1 = __floats2bfloat162_rn(acc[2] / inv, acc[3] / inv);
  uint2 packed;
  packed.x = *reinterpret_cast<unsigned*>(&o0); packed.y = *reinterpret_cast<unsigned*>(&o1);
  *reinterpret_cast<uint2*>(out + ((size_t)s * tokens + t) * channels + c) = packed;
}

int code_of(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return 0;
    case at::kBool: return 1;
    case at::kInt: return 2;
    case at::kLong: return 3;
    case at::kBFloat16: return 4;
    default: TORCH_CHECK(false, "pool_atoms: unsupported index / mask dtype ", t.scalar_type());
  }
}

// SUB of the wide kernel by the launch's CTA count (tokens x samples): 4 from POOL_SUB4_BLOCKS CTAs on, 8 below. H100 2026-11-12 (C 128, 24
// atoms per token, graph replay): SUB 4 / SUB 8 equal (+-0.7%) up to 640 CTAs, SUB 4 1.7-5.6% faster from 1024 on (N 16384 S 5: 16.8 vs
// 17.6 us; N 16384 S 48: 143.7 vs 151.8 us; engine 2.4.4 ran SUB 4). The sums run in the same atom order either way (same bits).
// The A100 build (sampler_ops_sm80.cu includes this file after defining the PDL launch attribute away) keeps SUB 8 everywhere.
#ifndef POOL_SUB4_BLOCKS
#ifdef cudaLaunchAttributeProgrammaticStreamSerialization
#define POOL_SUB4_BLOCKS 0x7fffffff
#else
#define POOL_SUB4_BLOCKS 1024
#endif
#endif
template <int IC, int GC, int MC>
void pool_fast(dim3 grid, unsigned threads, cudaStream_t s, const __nv_bfloat16* x, const at::Tensor& idx, const at::Tensor& gm, const at::Tensor& am,
               __nv_bfloat16* out, int A, int T, int R, int C) {
  if (C % 8 == 0 && (long)grid.x * grid.y >= (long)POOL_SUB4_BLOCKS)
    pool_atoms_wide_k<IC, GC, MC, 4><<<grid, (unsigned)(((C / 8) + 31) / 32 * 32), 0, s>>>(x, idx.data_ptr(), gm.data_ptr(), am.data_ptr(), out, A, T, R, C);
  else if (C % 8 == 0)
    pool_atoms_wide_k<IC, GC, MC><<<grid, (unsigned)(((C / 8) + 31) / 32 * 32), 0, s>>>(x, idx.data_ptr(), gm.data_ptr(), am.data_ptr(), out, A, T, R, C);
  else
    pool_atoms_fast_k<IC, GC, MC><<<grid, threads, 0, s>>>(x, idx.data_ptr(), gm.data_ptr(), am.data_ptr(), out, A, T, R, C);
}

template <int IC, int GC>
bool pool_by_mask(int mc, dim3 grid, unsigned threads, cudaStream_t s, const __nv_bfloat16* x, const at::Tensor& idx, const at::Tensor& gm,
                  const at::Tensor& am, __nv_bfloat16* out, int A, int T, int R, int C) {
  if (mc == 0) pool_fast<IC, GC, 0>(grid, threads, s, x, idx, gm, am, out, A, T, R, C);
  else if (mc == 1) pool_fast<IC, GC, 1>(grid, threads, s, x, idx, gm, am, out, A, T, R, C);
  else return false;
  return true;
}

template <int IC>
bool pool_by_gather(int gc, int mc, dim3 grid, unsigned threads, cudaStream_t s, const __nv_bfloat16* x, const at::Tensor& idx, const at::Tensor& gm,
                    const at::Tensor& am, __nv_bfloat16* out, int A, int T, int R, int C) {
  if (gc == 0) return pool_by_mask<IC, 0>(mc, grid, threads, s, x, idx, gm, am, out, A, T, R, C);
  if (gc == 1) return pool_by_mask<IC, 1>(mc, grid, threads, s, x, idx, gm, am, out, A, T, R, C);
  return false;
}

at::Tensor pool_atoms(at::Tensor x, at::Tensor idx, at::Tensor gather_mask, at::Tensor atom_mask) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 3, "x: contiguous bf16 cuda [S, Q, C]");
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && gather_mask.is_contiguous() && atom_mask.is_contiguous() && idx.dim() == 2 &&
              gather_mask.sizes() == idx.sizes() && atom_mask.sizes() == idx.sizes(), "idx / masks: contiguous [T, A]");
  const int64_t S = x.size(0), R = x.size(1), C = x.size(2), T = idx.size(0), A = idx.size(1);
  TORCH_CHECK(C % 4 == 0 && C / 4 <= 1024, "channels % 4 == 0 and <= 4096");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto out = at::empty({S, T, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const int ic = code_of(idx), gc = code_of(gather_mask), mc = code_of(atom_mask);
  const dim3 grid((unsigned)T, (unsigned)S);
  const unsigned threads = (unsigned)(((C / 4) + 31) / 32 * 32);
  bool done = false;
  if (ic == 3) done = pool_by_gather<3>(gc, mc, grid, threads, stream, xp, idx, gather_mask, atom_mask, op, (int)A, (int)T, (int)R, (int)C);
  else if (ic == 2) done = pool_by_gather<2>(gc, mc, grid, threads, stream, xp, idx, gather_mask, atom_mask, op, (int)A, (int)T, (int)R, (int)C);
  if (!done)
    pool_atoms_k<<<grid, (unsigned)(C / 4), 0, stream>>>(xp, idx.data_ptr(), ic, gather_mask.data_ptr(), gc, atom_mask.data_ptr(), mc, op, (int)A, (int)T,
                                                         (int)R, (int)C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---- the pooling's backward (training). d out[s, t] / d x[s, r] = sum over the slots (t, a) with idx[t, a] = r of w[t, a] [x[s, r] > 0], w = m / max(sum_a m, 1e-10)
// on a gathered slot (m != 0 and g != 0, as the forward kernels: relu(g x) = relu(x) for a 0 / 1 gather mask), 0 elsewhere. An atom row may be read by several slots
// (any idx), so the gradient is a gather, not a scatter: the slots are sorted by row (a stable sort of ``key``, done by the caller), and one warp per
// (row, sample) adds its rows' slots in slot order (fp32, fixed order: no atomics, bitwise reproducible).
//   prep: key[k] = idx[k] for a gathered slot k = t A + a, else R (sorted past every row), w[k] as above (the forward's fp32 sum of the token's atom mask).
__global__ void pool_atoms_prep_k(const void* idx, int idx_code, const void* gm, int gm_code, const void* am, int am_code, int* key, float* w,
                                  int atoms, int tokens, int rows) {
  const long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (long)tokens * atoms) return;
  const long t0 = (k / atoms) * atoms;
  float den = 0.f;
  for (int a = 0; a < atoms; ++a) den += ldf(am, am_code, t0 + a);
  const float m = ldf(am, am_code, k);
  const bool use = m != 0.f && ldf(gm, gm_code, k) != 0.f;
  key[k] = use ? (int)(long)ldf(idx, idx_code, k) : rows;
  w[k] = use ? __fdiv_rn(m, fmaxf(den, 1e-10f)) : 0.f;
}

constexpr int BWD_ROWS = 8;     // rows (warps) per block
constexpr int BWD_SAMPLES = 8;  // at most this many samples per block (the row's slot range is searched once per block); fewer when the rows alone do not fill the GPU

// The first k in [0, n) with v[k] >= x (n if none), v sorted ascending; the whole warp calls it and gets the same answer. A 32-ary search: each round
// the 32 lanes probe 32 evenly spaced keys (one load each, in parallel), so 3 072 slots take three rounds instead of twelve dependent loads.
__device__ __forceinline__ int warp_lower_bound(const int* __restrict__ v, int n, int x, int lane) {
  int lo = 0, hi = n;
  while (hi - lo > 32) {
    const int step = (hi - lo + 31) / 32, pos = lo + lane * step;
    const int cnt = __popc(__ballot_sync(0xffffffffu, pos < hi && __ldg(v + pos) < x));  // probes lo, lo + step, ... below x (a prefix: v is sorted)
    if (cnt == 0) return lo;
    const int nlo = lo + (cnt - 1) * step + 1;
    hi = min(hi, lo + cnt * step);
    lo = nlo;
  }
  const int pos = lo + lane;
  return lo + __popc(__ballot_sync(0xffffffffu, pos < hi && __ldg(v + pos) < x));
}

// The end of row x's run that starts at beg (the first k >= beg with v[k] != x): a run is usually a few slots, scanned 32 at a time.
__device__ __forceinline__ int warp_run_end(const int* __restrict__ v, int n, int beg, int x, int lane) {
  for (int k = beg;; k += 32) {
    const unsigned eq = __ballot_sync(0xffffffffu, k + lane < n && __ldg(v + k + lane) == x);
    if (eq != 0xffffffffu) return k + __popc(eq);  // the run's slots are a prefix of the 32 (v sorted)
  }
}

// grid (ceil(R / BWD_ROWS), ceil(S / spb)), spb <= BWD_SAMPLES samples per block (pool_atoms_bwd), block (32, BWD_ROWS): a lane owns 4 consecutive channels (C % 4 == 0), stepping 128 channels.
__global__ void __launch_bounds__(32 * BWD_ROWS) pool_atoms_bwd_k(const __nv_bfloat16* __restrict__ gout, const __nv_bfloat16* __restrict__ x,
                                                                  const int* __restrict__ skey, const long long* __restrict__ perm,
                                                                  const float* __restrict__ w, __nv_bfloat16* __restrict__ gx, int atoms, int tokens,
                                                                  int rows, int channels, int samples, int slots, int spb) {
  const int r = blockIdx.x * BWD_ROWS + threadIdx.y;
  if (r >= rows) return;
  const int beg = warp_lower_bound(skey, slots, r, threadIdx.x), end = warp_run_end(skey, slots, beg, r, threadIdx.x);
  const int s0 = blockIdx.y * spb, s1 = min(samples, s0 + spb);
  for (int s = s0; s < s1; ++s) {
    const __nv_bfloat16* gs = gout + (size_t)s * tokens * channels;
    for (int c = threadIdx.x * 4; c < channels; c += 128) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k = beg; k < end; ++k) {
        const long long slot = __ldg(perm + k);
        const float wk = __ldg(w + slot);
        const uint2 raw = *reinterpret_cast<const uint2*>(gs + (size_t)(slot / atoms) * channels + c);
        const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), v1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        acc[0] += wk * __bfloat162float(v0.x); acc[1] += wk * __bfloat162float(v0.y);
        acc[2] += wk * __bfloat162float(v1.x); acc[3] += wk * __bfloat162float(v1.y);
      }
      const size_t o = ((size_t)s * rows + r) * channels + c;
      const uint2 xr = *reinterpret_cast<const uint2*>(x + o);
      const __nv_bfloat162 x0 = *reinterpret_cast<const __nv_bfloat162*>(&xr.x), x1 = *reinterpret_cast<const __nv_bfloat162*>(&xr.y);
      const __nv_bfloat162 g0 = __floats2bfloat162_rn(__bfloat162float(x0.x) > 0.f ? acc[0] : 0.f, __bfloat162float(x0.y) > 0.f ? acc[1] : 0.f);
      const __nv_bfloat162 g1 = __floats2bfloat162_rn(__bfloat162float(x1.x) > 0.f ? acc[2] : 0.f, __bfloat162float(x1.y) > 0.f ? acc[3] : 0.f);
      uint2 packed;
      packed.x = *reinterpret_cast<const unsigned*>(&g0); packed.y = *reinterpret_cast<const unsigned*>(&g1);
      *reinterpret_cast<uint2*>(gx + o) = packed;
    }
  }
}

// (key int32 [T A], w fp32 [T A]) of the backward (see pool_atoms_prep_k); ``rows`` = R, the key of a slot that is not gathered.
std::vector<at::Tensor> pool_atoms_prep(at::Tensor idx, at::Tensor gather_mask, at::Tensor atom_mask, int64_t rows) {
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && gather_mask.is_contiguous() && atom_mask.is_contiguous() && idx.dim() == 2 &&
              gather_mask.sizes() == idx.sizes() && atom_mask.sizes() == idx.sizes(), "idx / masks: contiguous [T, A]");
  const int64_t T = idx.size(0), A = idx.size(1), n = T * A;
  TORCH_CHECK(n < INT_MAX && rows < INT_MAX, "pool_atoms_prep: too many slots / rows");
  const at::cuda::OptionalCUDAGuard guard(idx.device());
  auto key = at::empty({n}, idx.options().dtype(at::kInt));
  auto w = at::empty({n}, idx.options().dtype(at::kFloat));
  if (n > 0)
    pool_atoms_prep_k<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        idx.data_ptr(), code_of(idx), gather_mask.data_ptr(), code_of(gather_mask), atom_mask.data_ptr(), code_of(atom_mask), key.data_ptr<int>(),
        w.data_ptr<float>(), (int)A, (int)T, (int)rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {key, w};
}

// d x [S, R, C] bf16 of the upstream gradient ``gout`` [S, T, C] bf16: ``skey`` / ``perm`` the stable sort of ``key`` (int32 / int64 [T A]), ``w`` of pool_atoms_prep.
at::Tensor pool_atoms_bwd(at::Tensor gout, at::Tensor x, at::Tensor skey, at::Tensor perm, at::Tensor w, int64_t atoms) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 3, "x: contiguous bf16 cuda [S, Q, C]");
  TORCH_CHECK(gout.is_contiguous() && gout.scalar_type() == at::kBFloat16 && gout.dim() == 3 && gout.size(0) == x.size(0) && gout.size(2) == x.size(2),
              "gout: contiguous bf16 [S, T, C]");
  TORCH_CHECK(skey.is_contiguous() && skey.scalar_type() == at::kInt && perm.is_contiguous() && perm.scalar_type() == at::kLong &&
              w.is_contiguous() && w.scalar_type() == at::kFloat && perm.numel() == skey.numel() && w.numel() == skey.numel() &&
              skey.numel() == gout.size(1) * atoms, "skey int32 / perm int64 / w fp32 [T A]");
  const int64_t S = x.size(0), R = x.size(1), C = x.size(2), T = gout.size(1);
  TORCH_CHECK(C % 4 == 0, "channels % 4 == 0");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto gx = at::empty_like(x);
  if (gx.numel() == 0) return gx;
  // samples per block: up to BWD_SAMPLES (one slot search per row and block), fewer while the grid holds less than ~32 blocks (256 warps) per SM: a short
  // structure with many samples (1024 atoms x 48 samples) is otherwise latency bound on the serial sample loop. The sums do not depend on it.
  const int64_t row_blocks = (R + BWD_ROWS - 1) / BWD_ROWS, target = 32LL * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t want = std::max<int64_t>(1, (target + row_blocks - 1) / row_blocks);
  const int spb = (int)std::min<int64_t>(BWD_SAMPLES, std::max<int64_t>(1, (S + want - 1) / want));
  const dim3 grid((unsigned)row_blocks, (unsigned)((S + spb - 1) / spb)), block(32, BWD_ROWS);
  pool_atoms_bwd_k<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gout.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), skey.data_ptr<int>(),
      reinterpret_cast<const long long*>(perm.data_ptr<int64_t>()), w.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(gx.data_ptr()), (int)atoms, (int)T, (int)R, (int)C, (int)S,
      (int)skey.numel(), spb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gx;
}

void check(const at::Tensor& t, const char* name, int ndim) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat && t.dim() == ndim, name, ": contiguous fp32 cuda ", ndim, "-d");
}

std::vector<at::Tensor> augment_noise(at::Tensor coords, at::Tensor mask, at::Tensor rv, at::Tensor tr, at::Tensor noise,
                                      at::Tensor sigma, at::Tensor sigma_next, int64_t window, double gamma, double gamma_min,
                                      double sigma_min, double sigma_max, double noise_scale, double variance_floor, bool clamp) {
  check(coords, "coords", 3); check(mask, "mask", 1); check(noise, "noise", 3);
  const int64_t B = coords.size(0), N = coords.size(1);
  TORCH_CHECK(coords.size(2) == 3 && mask.size(0) == N && noise.sizes() == coords.sizes(), "shapes");
  TORCH_CHECK(rv.numel() == B * 6 && tr.numel() == B * 3 && rv.scalar_type() == at::kFloat && tr.scalar_type() == at::kFloat &&
              rv.is_contiguous() && tr.is_contiguous(), "rotation / translation draws");
  TORCH_CHECK(sigma.numel() == 1 && sigma_next.numel() == 1 && sigma.scalar_type() == at::kFloat && sigma_next.scalar_type() == at::kFloat, "sigma scalars");
  const at::cuda::OptionalCUDAGuard guard(coords.device());
  const int chunk = chunk_for(N, B);
  const int C = (int)((N + chunk - 1) / chunk);
  auto part = at::empty({B, C, PART}, coords.options());
  auto noisy = at::empty_like(coords);
  auto sc = at::empty({4}, coords.options());
  Churn cfg{(int)window, (float)(1.0 + gamma), (float)gamma_min, (float)sigma_min, (float)sigma_max, (float)noise_scale, (float)variance_floor, 0.f, clamp ? 1 : 0};
  auto s = at::cuda::getCurrentCUDAStream();
  dim3 grid(C, (unsigned)B);
  const float* cp = coords.data_ptr<float>();
  const float* mp = mask.data_ptr<float>();
  float* pp = part.data_ptr<float>();
#define MW_AUG(KK)                                                                                                                                    \
  launch(moments_k<KK>, grid, THREADS, s, false, cp, mp, pp, (int)N);                                                                                \
  launch(augment_noise_k<KK>, grid, THREADS, s, true, cp, mp, (const float*)pp, rv.data_ptr<float>(), tr.data_ptr<float>(), noise.data_ptr<float>(),  \
         sigma.data_ptr<float>(), sigma_next.data_ptr<float>(), cfg, noisy.data_ptr<float>(), sc.data_ptr<float>(), (int)N, C)
  if (chunk == 1024) { MW_AUG(4); } else if (chunk == 512) { MW_AUG(2); } else { MW_AUG(1); }
#undef MW_AUG
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {noisy, sc};
}

at::Tensor align_euler(at::Tensor noisy, at::Tensor denoised, at::Tensor mask, at::Tensor sc, at::Tensor sigma_next, double step_scale) {
  check(noisy, "noisy", 3); check(denoised, "denoised", 3); check(mask, "mask", 1);
  const int64_t B = noisy.size(0), N = noisy.size(1);
  TORCH_CHECK(denoised.sizes() == noisy.sizes() && mask.size(0) == N && sc.numel() >= 2 && sigma_next.numel() == 1, "shapes");
  const at::cuda::OptionalCUDAGuard guard(noisy.device());
  const int chunk = chunk_for(N, B);
  const int C = (int)((N + chunk - 1) / chunk);
  auto part = at::empty({B, C, PART}, noisy.options());
  auto cov = at::empty({B, C, PART}, noisy.options());
  auto out = at::empty_like(noisy);
  auto s = at::cuda::getCurrentCUDAStream();
  dim3 grid(C, (unsigned)B);
  const float* np = noisy.data_ptr<float>();
  const float* dp = denoised.data_ptr<float>();
  const float* mp = mask.data_ptr<float>();
  float* pp = part.data_ptr<float>();
  float* cvp = cov.data_ptr<float>();
#define MW_ALN(KK)                                                                                                                                    \
  launch(moments2_k<KK>, grid, THREADS, s, false, np, dp, mp, pp, (int)N);                                                                           \
  launch(cov_k<KK>, grid, THREADS, s, true, np, dp, mp, (const float*)pp, cvp, (int)N, C);                                                           \
  launch(align_euler_k<KK>, grid, THREADS, s, true, np, dp, (const float*)pp, (const float*)cvp, sc.data_ptr<float>(), sigma_next.data_ptr<float>(),  \
         (float)step_scale, out.data_ptr<float>(), (int)N, C)
  if (chunk == 1024) { MW_ALN(4); } else if (chunk == 512) { MW_ALN(2); } else { MW_ALN(1); }
#undef MW_ALN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("augment_noise", &augment_noise);
  m.def("align_euler", &align_euler);
  m.def("pool_atoms", &pool_atoms);
  m.def("pool_atoms_prep", &pool_atoms_prep);
  m.def("pool_atoms_bwd", &pool_atoms_bwd);
}
