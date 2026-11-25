// sampler_ops_sm100.cu -- the B200 (sm_100a) build of the sampler glue: sampler_ops.cu (the H100 source, compiled unchanged; its own module
// definition renamed away) plus single-launch variants for the B200, where a CUDA-graph replay pays ~2 us per kernel node (2026-11-17, measurer
// timing: one empty kernel 6.0 us, each further dependent kernel +2.0 us, PDL or not) and these ops are a few microseconds of work:
//
//   augment_noise   ONE thread-block-cluster kernel per sample (augment_noise_cl_k): the CTAs of a sample's cluster load their atoms, sum
//                   (w, w p) per CTA, exchange the CTA sums through distributed shared memory (every CTA adds the cluster's partials in rank
//                   order: the same total, bitwise, in every CTA, no atomics), then centre / rotate / re-noise their own atoms. Was 2 kernels.
//   align_euler     ONE cluster kernel (align_euler_cl_k): one-pass moments about atom 0 (sum w, w m', w r', w m' r'^T) -> ONE cluster exchange
//                   -> the centred covariance and Horn's rotation in thread 0's registers (horn_rotation_one: no shuffles, no division per
//                   squaring) -> the Euler update. Was 3 kernels.
//   pool_atoms      up to 1536 CTAs: the wide kernel with every atom row of a token requested at once (SUB 32: one dependent round trip after the index load
//                   instead of three); in training it also zeroes the backward's per-row counters.
//   pool_atoms bwd  no sort: pool_prep_b200_k writes the slot weights and buckets every gathered slot under its row (an atomic counter per row
//                   gives a position, CAP slots a row); pool_bwd_b200_k (PDL-chained) orders each row's few slots by slot index in registers and
//                   adds them in that order -- the stable sort's order, so the sums are those of the H100 kernel, bitwise, and reproducible.
//                   A row with more than CAP slots scans the slot keys in order instead (correct for any index, slow only for such rows).
//                   Was prep + torch.sort (4-11 kernels: radix passes, memsets, copies) + bwd.
// Sizes outside the single-launch contracts take sampler_ops.cu's kernels unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void sampler_ops_base_bindings(pybind11::module_& variable)
#include "sampler_ops.cu"
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")

namespace cg = cooperative_groups;

namespace {
constexpr int MAX_CLUSTER = 16;   // non-portable cluster size (sm_100: 16 CTAs)

template <typename... KArgs, typename... Args>
void launch_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block, int cluster_x, cudaStream_t stream, Args... args) {
  if (cluster_x > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster_x; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

// Horn's rotation in ONE thread's registers: horn_rotation's method (N(H) + |N|_F I squared 12 times, the column of largest norm as the
// quaternion) on the 10 distinct entries of the symmetric matrix, rescaled by a power of two (exact) from its trace before squarings 0, 3, 6, 9
// instead of a division by the largest entry after each (trace = sum of the eigenvalues >= the largest: three squarings of a matrix with
// trace in [1, 2) stay in [4^-8, 2^8]). ~530 instructions; the 16-entry version with a division per squaring was ~1400 (1.6 us on one thread).
__device__ __forceinline__ void horn_rotation_one(const float* h, float* rot) {
  const float sxx = h[0], sxy = h[1], sxz = h[2], syx = h[3], syy = h[4], syz = h[5], szx = h[6], szy = h[7], szz = h[8];
  float a00 = sxx + syy + szz, a01 = syz - szy, a02 = szx - sxz, a03 = sxy - syx, a11 = sxx - syy - szz, a12 = sxy + syx, a13 = szx + sxz,
        a22 = -sxx + syy - szz, a23 = syz + szy, a33 = -sxx - syy + szz;
  const float off = a01 * a01 + a02 * a02 + a03 * a03 + a12 * a12 + a13 * a13 + a23 * a23;
  const float shift = sqrtf(a00 * a00 + a11 * a11 + a22 * a22 + a33 * a33 + 2.f * off);
  a00 += shift; a11 += shift; a22 += shift; a33 += shift;
#pragma unroll
  for (int it = 0; it < 12; ++it) {
    if (it % 3 == 0) {
      const float tr = (a00 + a11) + (a22 + a33);
      const float sc = __int_as_float((254 << 23) - (__float_as_int(tr) & 0x7F800000));   // 2^-floor(log2 tr)
      a00 *= sc; a01 *= sc; a02 *= sc; a03 *= sc; a11 *= sc; a12 *= sc; a13 *= sc; a22 *= sc; a23 *= sc; a33 *= sc;
    }
    const float b00 = fmaf(a03, a03, fmaf(a02, a02, fmaf(a01, a01, a00 * a00)));
    const float b01 = fmaf(a03, a13, fmaf(a02, a12, fmaf(a01, a11, a00 * a01)));
    const float b02 = fmaf(a03, a23, fmaf(a02, a22, fmaf(a01, a12, a00 * a02)));
    const float b03 = fmaf(a03, a33, fmaf(a02, a23, fmaf(a01, a13, a00 * a03)));
    const float b11 = fmaf(a13, a13, fmaf(a12, a12, fmaf(a11, a11, a01 * a01)));
    const float b12 = fmaf(a13, a23, fmaf(a12, a22, fmaf(a11, a12, a01 * a02)));
    const float b13 = fmaf(a13, a33, fmaf(a12, a23, fmaf(a11, a13, a01 * a03)));
    const float b22 = fmaf(a23, a23, fmaf(a22, a22, fmaf(a12, a12, a02 * a02)));
    const float b23 = fmaf(a23, a33, fmaf(a22, a23, fmaf(a12, a13, a02 * a03)));
    const float b33 = fmaf(a33, a33, fmaf(a23, a23, fmaf(a13, a13, a03 * a03)));
    a00 = b00; a01 = b01; a02 = b02; a03 = b03; a11 = b11; a12 = b12; a13 = b13; a22 = b22; a23 = b23; a33 = b33;
  }
  const float m[16] = {a00, a01, a02, a03, a01, a11, a12, a13, a02, a12, a22, a23, a03, a13, a23, a33};
  int pick = 0;
  float best = -1.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float t = m[c] * m[c] + m[4 + c] * m[4 + c] + m[8 + c] * m[8 + c] + m[12 + c] * m[12 + c];
    if (t > best) { best = t; pick = c; }
  }
  float q[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) q[r] = m[r * 4 + pick];
  const float qn = fmaxf(sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), FLT_MIN);
  const float w = q[0] / qn, x = q[1] / qn, y = q[2] / qn, z = q[3] / qn;
  const float Q[9] = {w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (y * x + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
                      2 * (z * x - w * y), 2 * (z * y + w * x), w * w - x * x - y * y + z * z};
#pragma unroll
  for (int a = 0; a < 3; ++a)
#pragma unroll
    for (int c = 0; c < 3; ++c) rot[a * 3 + c] = Q[c * 3 + a];
}

// The cluster exchange of NV block sums (red[v * 8] of block_sums) in ONE cluster barrier: every CTA pushes its sums into slot [its rank][v] of every
// CTA's ``inbox`` (distributed shared memory stores), the caller runs cl.sync() (release / acquire: the stores visible, none read remotely after it,
// so no CTA can exit under a reader), then cluster_fold adds the slots in rank order -- the same total, bitwise, in every CTA. (Measured 2026-11-17:
// a cluster barrier after the loads is free in a graph replay, a pull exchange with a second barrier costs one 2 us replay step.)
template <int NV>
__device__ __forceinline__ void cluster_push(cg::cluster_group& cl, const float* red, float* inbox) {
  const int nc = (int)cl.num_blocks(), me = (int)cl.block_rank();
  for (int j = threadIdx.x; j < nc * NV; j += blockDim.x) {
    const int dst = j / NV, v = j - dst * NV;
    *cl.map_shared_rank(inbox + me * NV + v, dst) = red[v * 8];
  }
}
template <int NV>
__device__ __forceinline__ void cluster_fold(cg::cluster_group& cl, const float* inbox, float* out) {
  if (threadIdx.x < NV) {
    const int nc = (int)cl.num_blocks();
    float s = 0.f;
    for (int r = 0; r < nc; ++r) s += inbox[r * NV + threadIdx.x];
    out[threadIdx.x] = s;
  }
}

// The cluster's total of the block sums in out[0..NV) (threads 0..NV-1 write it; the caller's barrier publishes it). A one-CTA cluster has
// its total already: no exchange, no cluster barrier (the exchange measured ~0.6 us of device time; the same bits either way).
template <int NV>
__device__ __forceinline__ void cluster_total(cg::cluster_group& cl, const float* red, float* inbox, float* out) {
  if (cl.num_blocks() > 1) {
    cluster_push<NV>(cl, red, inbox);
    cl.sync();
    cluster_fold<NV>(cl, inbox, out);
  } else if (threadIdx.x < NV) {
    out[threadIdx.x] = 0.f + red[threadIdx.x * 8];
  }
}

// grid (nc, B), cluster (nc, 1, 1): CTA rank c of sample b owns atoms [c THREADS K, (c + 1) THREADS K) (augment_noise_k's mapping with CHUNK = THREADS K).
template <int K>
__global__ void __launch_bounds__(THREADS) augment_noise_cl_k(const float* __restrict__ p, const float* __restrict__ w, const float* __restrict__ rv,
                                                              const float* __restrict__ tr, const float* __restrict__ noise,
                                                              const float* __restrict__ sigma, const float* __restrict__ sigma_next, Churn cfg,
                                                              float* noisy, float* sc, int n) {
  __shared__ float red[4 * 8], inbox[MAX_CLUSTER * 4], s_sum[4], s_rot[9], s_t[3], s_mag;
  cg::cluster_group cl = cg::this_cluster();
  const int b = blockIdx.y, base = (int)cl.block_rank() * (THREADS * K);
  float pv[K][3], nv[K][3], wv[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
    wv[k] = 0.f; pv[k][0] = pv[k][1] = pv[k][2] = 0.f;
    if (i < n) {
      const size_t o = ((size_t)b * n + i) * 3;
      pv[k][0] = p[o]; pv[k][1] = p[o + 1]; pv[k][2] = p[o + 2];
      nv[k][0] = noise[o]; nv[k][1] = noise[o + 1]; nv[k][2] = noise[o + 2];
      wv[k] = w[i];
    }
  }
  if (threadIdx.x == 0) {
    const float* v = rv + b * 6;
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
    if (cl.block_rank() == 0 && b == 0) { sc[0] = sh; sc[1] = mag; }
  }
  float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const float wi = wv[k];
    v[0] += wi; v[1] += wi * pv[k][0]; v[2] += wi * pv[k][1]; v[3] += wi * pv[k][2];
  }
  block_sums<4>(v, red);
  cluster_total<4>(cl, red, inbox, s_sum);
  __syncthreads();
  const float tot = fmaxf(s_sum[0], 1e-6f);
  const float c0 = s_sum[1] / tot, c1 = s_sum[2] / tot, c2 = s_sum[3] / tot;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
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

template <int K>
__global__ void __launch_bounds__(THREADS) align_euler_cl_k(const float* __restrict__ m, const float* __restrict__ r, const float* __restrict__ w,
                                                            const float* __restrict__ sc, const float* __restrict__ sigma_next, float step_scale,
                                                            float* out, int n) {
  __shared__ float red[16 * 8], inbox[MAX_CLUSTER * 16], s_sum[16], s_rot[9], s_k, s_sh;
  cg::cluster_group cl = cg::this_cluster();
  const int b = blockIdx.y, base = (int)cl.block_rank() * (THREADS * K);
  // the reference point of the one-pass moments: atom 0 of each tensor (any point works; near the structure keeps the cancellation small)
  const size_t o0 = (size_t)b * n * 3;
  const float p0[3] = {m[o0], m[o0 + 1], m[o0 + 2]}, q0[3] = {r[o0], r[o0 + 1], r[o0 + 2]};
  float wv[K], mv[K][3], rv[K][3];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
    wv[k] = 0.f;
    mv[k][0] = mv[k][1] = mv[k][2] = rv[k][0] = rv[k][1] = rv[k][2] = 0.f;
    if (i < n) {
      wv[k] = w[i];
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
  {
    // (sum w, sum w m', sum w r', sum w m'_p r'_q) with m' = m - p0, r' = r - q0: one exchange instead of two
    float v[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const float wi = wv[k];
      float a[3], c[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) { a[j] = mv[k][j] - p0[j]; c[j] = rv[k][j] - q0[j]; }
      v[0] += wi;
#pragma unroll
      for (int j = 0; j < 3; ++j) { v[1 + j] += wi * a[j]; v[4 + j] += wi * c[j]; }
#pragma unroll
      for (int p = 0; p < 3; ++p)
#pragma unroll
        for (int q = 0; q < 3; ++q) v[7 + p * 3 + q] += (wi * a[p]) * c[q];
    }
    block_sums<16>(v, red);
  }
  cluster_total<16>(cl, red, inbox, s_sum);   // written by lanes 0..15 of warp 0
  if (threadIdx.x < 32) {
    __syncwarp();
    if (threadIdx.x == 0) {
      // the centred covariance sum_i w (m_i - cm)(r_i - cr)^T = sum w m' r'^T - tot dm dr^T, and its rotation, in thread 0's registers
      const float tot0 = s_sum[0];
      float h[9], rot0[9];
#pragma unroll
      for (int p = 0; p < 3; ++p)
#pragma unroll
        for (int q = 0; q < 3; ++q) h[p * 3 + q] = s_sum[7 + p * 3 + q] - tot0 * (s_sum[1 + p] / tot0) * (s_sum[4 + q] / tot0);
      horn_rotation_one(h, rot0);
#pragma unroll
      for (int j = 0; j < 9; ++j) s_rot[j] = rot0[j];
    }
  }
  __syncthreads();
  const float tot = s_sum[0];
  float cm[3], cr[3], rot[9];
#pragma unroll
  for (int j = 0; j < 3; ++j) { cm[j] = p0[j] + s_sum[1 + j] / tot; cr[j] = q0[j] + s_sum[4 + j] / tot; }
#pragma unroll
  for (int j = 0; j < 9; ++j) rot[j] = s_rot[j];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
    if (i >= n) continue;
    const size_t o = ((size_t)b * n + i) * 3;
    const float x = mv[k][0] - cm[0], y = mv[k][1] - cm[1], z = mv[k][2] - cm[2];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float al = __fadd_rn(x * rot[j] + y * rot[3 + j] + z * rot[6 + j], cr[j]);
      const float dir = __fdiv_rn(__fsub_rn(al, rv[k][j]), s_sh);
      out[o + j] = __fadd_rn(al, __fmul_rn(s_k, dir));
    }
  }
}


// Atoms per thread K and CTAs per sample nc = ceil(N / (THREADS K)) of the cluster kernels (measured 2026-11-17, measurer timing, S 1 / 5 / 48 x
// N 1024-16384): the smallest K in 1, 2, 4 whose cluster fits (nc <= 16) while the launch stays within g_cta_per_sm CTAs per SM (B nc; past
// that K 4 is 2-6 us faster at S 48: N 4096 align 12.1 vs 16.2, N 8192 14.1 vs 18.2); K 8 only when K 4 does not fit a cluster (2 us slower
// than K 4 where both fit). One CTA of K 4 / 8 for N <= 2048 (no exchange) measured 2 us SLOWER than the 4- / 8-CTA cluster (align N 1024 10.0
// vs 8.0, N 2048 12.1 vs 10.0). nc 0 when N needs more than 8 x 256 x 16 atoms a sample (the two- / three-kernel chain of sampler_ops.cu then).
int g_cta_per_sm = 4;
void cluster_shape(int64_t n, int64_t samples, int& K, int& nc) {
  const int64_t budget = (int64_t)g_cta_per_sm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  for (int k : {1, 2, 4}) {
    const int64_t c = (n + (int64_t)THREADS * k - 1) / ((int64_t)THREADS * k);
    if (c <= MAX_CLUSTER && c * samples <= budget) { K = k; nc = (int)c; return; }
  }
  for (int k : {4, 8}) {
    const int64_t c = (n + (int64_t)THREADS * k - 1) / ((int64_t)THREADS * k);
    if (c <= MAX_CLUSTER) { K = k; nc = (int)c; return; }
  }
  K = 8; nc = 0;
}

std::vector<at::Tensor> augment_noise_b200(at::Tensor coords, at::Tensor mask, at::Tensor rv, at::Tensor tr, at::Tensor noise, at::Tensor sigma,
                                           at::Tensor sigma_next, int64_t window, double gamma, double gamma_min, double sigma_min, double sigma_max,
                                           double noise_scale, double variance_floor, bool clamp) {
  check(coords, "coords", 3); check(mask, "mask", 1); check(noise, "noise", 3);
  const int64_t B = coords.size(0), N = coords.size(1);
  TORCH_CHECK(coords.size(2) == 3 && mask.size(0) == N && noise.sizes() == coords.sizes(), "shapes");
  TORCH_CHECK(rv.numel() == B * 6 && tr.numel() == B * 3 && rv.scalar_type() == at::kFloat && tr.scalar_type() == at::kFloat &&
              rv.is_contiguous() && tr.is_contiguous(), "rotation / translation draws");
  TORCH_CHECK(sigma.numel() == 1 && sigma_next.numel() == 1 && sigma.scalar_type() == at::kFloat && sigma_next.scalar_type() == at::kFloat, "sigma scalars");
  int K = 0, nc = 0;
  cluster_shape(N, B, K, nc);
  if (nc == 0 || B > 65535)
    return augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp);
  const at::cuda::OptionalCUDAGuard guard(coords.device());
  auto noisy = at::empty_like(coords);
  auto sc = at::empty({4}, coords.options());
  Churn cfg{(int)window, (float)(1.0 + gamma), (float)gamma_min, (float)sigma_min, (float)sigma_max, (float)noise_scale, (float)variance_floor, 0.f, clamp ? 1 : 0};
  auto s = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)nc, (unsigned)B);
#define MW_AUG(KK)                                                                                                                                   \
  launch_cluster(augment_noise_cl_k<KK>, grid, dim3(THREADS), nc, s, coords.data_ptr<float>(), mask.data_ptr<float>(), rv.data_ptr<float>(),        \
                 tr.data_ptr<float>(), noise.data_ptr<float>(), sigma.data_ptr<float>(), sigma_next.data_ptr<float>(), cfg, noisy.data_ptr<float>(), \
                 sc.data_ptr<float>(), (int)N)
  if (K == 1) { MW_AUG(1); } else if (K == 2) { MW_AUG(2); } else if (K == 4) { MW_AUG(4); } else { MW_AUG(8); }
#undef MW_AUG
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {noisy, sc};
}

at::Tensor align_euler_b200(at::Tensor noisy, at::Tensor denoised, at::Tensor mask, at::Tensor sc, at::Tensor sigma_next, double step_scale) {
  check(noisy, "noisy", 3); check(denoised, "denoised", 3); check(mask, "mask", 1);
  const int64_t B = noisy.size(0), N = noisy.size(1);
  TORCH_CHECK(denoised.sizes() == noisy.sizes() && mask.size(0) == N && sc.numel() >= 2 && sigma_next.numel() == 1, "shapes");
  int K = 0, nc = 0;
  cluster_shape(N, B, K, nc);
  if (nc == 0 || B > 65535) return align_euler(noisy, denoised, mask, sc, sigma_next, step_scale);
  const at::cuda::OptionalCUDAGuard guard(noisy.device());
  auto out = at::empty_like(noisy);
  auto s = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)nc, (unsigned)B);
#define MW_ALN(KK)                                                                                                                                  \
  launch_cluster(align_euler_cl_k<KK>, grid, dim3(THREADS), nc, s, noisy.data_ptr<float>(), denoised.data_ptr<float>(), mask.data_ptr<float>(),    \
                 sc.data_ptr<float>(), sigma_next.data_ptr<float>(), (float)step_scale, out.data_ptr<float>(), (int)N)
  if (K == 1) { MW_ALN(1); } else if (K == 2) { MW_ALN(2); } else if (K == 4) { MW_ALN(4); } else { MW_ALN(8); }
#undef MW_ALN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---- pooling forward: pool_atoms_wide_k's arithmetic (same atom order, same bits) with all of a token's atoms (<= 32 a round) requested at once;
// ``zero`` (training) is the backward's row counter array, zeroed by the sample-0 CTAs (rows / tokens entries each).
template <int IC, int GC, int MC>
__global__ void pool_atoms_b200_k(const __nv_bfloat16* __restrict__ x, const void* idx, const void* gm, const void* am, __nv_bfloat16* out, int atoms,
                                  int tokens, int rows, int channels, int* zero) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x * 8, lane = threadIdx.x & 31;
  if (zero != nullptr && s == 0)
    for (int i = t * blockDim.x + threadIdx.x; i < rows; i += tokens * blockDim.x) zero[i] = 0;
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
      const int ri = (int)(long)ldt<IC>(idx, k);
      if (ml != 0.f && gl != 0.f) rl = ri;
    }
    const int nvalid = min(32, atoms - g0);
    uint4 raw[32];
#pragma unroll
    for (int u = 0; u < 32; ++u) {
      const int r = __shfl_sync(0xffffffffu, rl, u);
      raw[u] = make_uint4(0u, 0u, 0u, 0u);
      if (live && u < nvalid && r != NO_ROW) raw[u] = *reinterpret_cast<const uint4*>(xs + (size_t)r * channels);
    }
#pragma unroll
    for (int u = 0; u < 32; ++u) {
      const float mu = __shfl_sync(0xffffffffu, ml, u);
      const int r = __shfl_sync(0xffffffffu, rl, u);
      if (u >= nvalid) continue;
      den += mu;
      if (r != NO_ROW) {
        const unsigned wq[4] = {raw[u].x, raw[u].y, raw[u].z, raw[u].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          acc[2 * q] += mu * fmaxf(__uint_as_float(wq[q] << 16), 0.f);
          acc[2 * q + 1] += mu * fmaxf(__uint_as_float(wq[q] & 0xffff0000u), 0.f);
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

template <int IC, int GC, int MC>
void pool_b200(dim3 grid, cudaStream_t s, const __nv_bfloat16* x, const at::Tensor& idx, const at::Tensor& gm, const at::Tensor& am, __nv_bfloat16* out,
               int A, int T, int R, int C, int* zero) {
  pool_atoms_b200_k<IC, GC, MC><<<grid, (unsigned)(((C / 8) + 31) / 32 * 32), 0, s>>>(x, idx.data_ptr(), gm.data_ptr(), am.data_ptr(), out, A, T, R, C, zero);
}

// [out [S, T, C] bf16, count [R] int32 (zeroed when ``train``, else empty)]
// pool_atoms_b200_k up to g_pool_b200_max CTAs (tokens x samples), sampler_ops.cu's wide kernel (SUB 8 / 4) above. Measured 2026-11-17 (measurer
// timing, us, base / this kernel): S1 N 8192 12.1 / 10.0, N 12288 12.1 / 10.0, S5 N 2048 (1280 CTAs) 12.1 / 10.0; equal (10.0) up to 896 CTAs;
// from 1920 CTAs on the base kernel wins (S5 N 3072 12.1 / 14.1, S1 N 16384 12.1 / 14.1, S48 N 8192 73.5 / 116.5: 32 rows a thread in flight
// lose to SUB 4 once the grid is several waves). Training above the limit: the base forward + a zero fill of the counters (S48 N 8192 182 / 225).
// Rejected: a shuffle-free forward (2 channels a thread, every thread reading the token's indices itself): 16.2 us at S1 N 1024 (10.0 here).
int64_t g_pool_b200_max = 1536;
std::vector<at::Tensor> pool_atoms_b200(at::Tensor x, at::Tensor idx, at::Tensor gather_mask, at::Tensor atom_mask, bool train) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 3, "x: contiguous bf16 cuda [S, Q, C]");
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && gather_mask.is_contiguous() && atom_mask.is_contiguous() && idx.dim() == 2 &&
              gather_mask.sizes() == idx.sizes() && atom_mask.sizes() == idx.sizes(), "idx / masks: contiguous [T, A]");
  const int64_t S = x.size(0), R = x.size(1), C = x.size(2), T = idx.size(0), A = idx.size(1);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto count = at::empty({train ? R : 0}, idx.options().dtype(at::kInt));
  const int ic = code_of(idx), gc = code_of(gather_mask), mc = code_of(atom_mask);
  const bool fast = C % 8 == 0 && C / 8 <= 1024 && (ic == 2 || ic == 3) && (gc == 0 || gc == 1) && (mc == 0 || mc == 1) && T > 0 && S <= 65535 &&
                    T * S <= g_pool_b200_max;
  if (!fast) {
    auto out = pool_atoms(x, idx, gather_mask, atom_mask);
    if (train) count.zero_();
    return {out, count};
  }
  auto out = at::empty({S, T, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  int* zp = train ? count.data_ptr<int>() : nullptr;
  const dim3 grid((unsigned)T, (unsigned)S);
#define MW_POOL(I, G, M) pool_b200<I, G, M>(grid, stream, xp, idx, gather_mask, atom_mask, op, (int)A, (int)T, (int)R, (int)C, zp)
  if (ic == 3) {
    if (gc == 1) { if (mc == 1) MW_POOL(3, 1, 1); else MW_POOL(3, 1, 0); }
    else { if (mc == 1) MW_POOL(3, 0, 1); else MW_POOL(3, 0, 0); }
  } else {
    if (gc == 1) { if (mc == 1) MW_POOL(2, 1, 1); else MW_POOL(2, 1, 0); }
    else { if (mc == 1) MW_POOL(2, 0, 1); else MW_POOL(2, 0, 0); }
  }
#undef MW_POOL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, count};
}

// ---- pooling backward without a sort
constexpr int CAP = 16;   // bucketed slots a row (a row with more scans the keys)

// key / w as pool_atoms_prep_k; a gathered slot k also takes position atomicAdd(count[row]) of its row's bucket (positions >= CAP are not stored).
__global__ void pool_prep_b200_k(const void* idx, int idx_code, const void* gm, int gm_code, const void* am, int am_code, int* count, int* bucket,
                                 int* key, float* w, int atoms, int tokens, int rows) {
  pdl_trigger();
  const long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (long)tokens * atoms) return;
  const long t0 = (k / atoms) * atoms;
  float den = 0.f;
  for (int a = 0; a < atoms; ++a) den += ldf(am, am_code, t0 + a);
  const float m = ldf(am, am_code, k);
  const bool use = m != 0.f && ldf(gm, gm_code, k) != 0.f;
  const int row = use ? (int)(long)ldf(idx, idx_code, k) : rows;
  key[k] = row;
  w[k] = use ? __fdiv_rn(m, fmaxf(den, 1e-10f)) : 0.f;
  if (use) {
    const int pos = atomicAdd(count + row, 1);
    if (pos < CAP) bucket[(size_t)row * CAP + pos] = (int)k;
  }
}

// grid / block as pool_atoms_bwd_k (warp per row, spb samples a block). The row's slots in ascending slot order (the stable sort's), each sum as there.
__global__ void __launch_bounds__(32 * BWD_ROWS) pool_bwd_b200_k(const __nv_bfloat16* __restrict__ gout, const __nv_bfloat16* __restrict__ x,
                                                                 const int* __restrict__ count, const int* __restrict__ bucket,
                                                                 const int* __restrict__ key, const float* __restrict__ w, __nv_bfloat16* __restrict__ gx,
                                                                 int atoms, int tokens, int rows, int channels, int samples, int slots, int spb) {
  const int r = blockIdx.x * BWD_ROWS + threadIdx.y, lane = threadIdx.x;
  if (r >= rows) return;
  pdl_wait();
  const int cnt = __ldcg(count + r);
  const int s0 = blockIdx.y * spb, s1 = min(samples, s0 + spb);
  if (cnt <= CAP) {
    const int my = lane < cnt ? __ldcg(bucket + (size_t)r * CAP + lane) : INT_MAX;
    int rank = 0;
#pragma unroll
    for (int j = 0; j < CAP; ++j) rank += (__shfl_sync(0xffffffffu, my, j) < my) ? 1 : 0;
    const float wl = lane < cnt ? __ldcg(w + my) : 0.f;
    // lane p holds the p-th slot (ascending) and its weight
    int sl = INT_MAX;
    float ws = 0.f;
#pragma unroll
    for (int j = 0; j < CAP; ++j) {
      const int sj = __shfl_sync(0xffffffffu, my, j);
      const float wj = __shfl_sync(0xffffffffu, wl, j);
      const int rj = __shfl_sync(0xffffffffu, rank, j);
      if (j < cnt && rj == lane) { sl = sj; ws = wj; }
    }
    const int tok = lane < cnt ? sl / atoms : 0;
    for (int s = s0; s < s1; ++s) {
      const __nv_bfloat16* gs = gout + (size_t)s * tokens * channels;
      for (int c = lane * 4; c < channels; c += 128) {
        const size_t o = ((size_t)s * rows + r) * channels + c;
        const uint2 xr = *reinterpret_cast<const uint2*>(x + o);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k0 = 0; k0 < cnt; k0 += 4) {
          uint2 raw[4];
          float wk[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int tk = __shfl_sync(0xffffffffu, tok, (k0 + u) & 31);
            wk[u] = __shfl_sync(0xffffffffu, ws, (k0 + u) & 31);
            raw[u] = make_uint2(0u, 0u);
            if (k0 + u < cnt) raw[u] = *reinterpret_cast<const uint2*>(gs + (size_t)tk * channels + c);
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            if (k0 + u >= cnt) break;
            const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&raw[u].x), v1 = *reinterpret_cast<const __nv_bfloat162*>(&raw[u].y);
            acc[0] += wk[u] * __bfloat162float(v0.x); acc[1] += wk[u] * __bfloat162float(v0.y);
            acc[2] += wk[u] * __bfloat162float(v1.x); acc[3] += wk[u] * __bfloat162float(v1.y);
          }
        }
        const __nv_bfloat162 x0 = *reinterpret_cast<const __nv_bfloat162*>(&xr.x), x1 = *reinterpret_cast<const __nv_bfloat162*>(&xr.y);
        const __nv_bfloat162 g0 = __floats2bfloat162_rn(__bfloat162float(x0.x) > 0.f ? acc[0] : 0.f, __bfloat162float(x0.y) > 0.f ? acc[1] : 0.f);
        const __nv_bfloat162 g1 = __floats2bfloat162_rn(__bfloat162float(x1.x) > 0.f ? acc[2] : 0.f, __bfloat162float(x1.y) > 0.f ? acc[3] : 0.f);
        uint2 packed;
        packed.x = *reinterpret_cast<const unsigned*>(&g0); packed.y = *reinterpret_cast<const unsigned*>(&g1);
        *reinterpret_cast<uint2*>(gx + o) = packed;
      }
    }
    return;
  }
  // more than CAP slots: every slot key in order, 32 at a time
  for (int s = s0; s < s1; ++s) {
    const __nv_bfloat16* gs = gout + (size_t)s * tokens * channels;
    for (int c = lane * 4; c < channels; c += 128) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < slots; k0 += 32) {
        unsigned hit = __ballot_sync(0xffffffffu, k0 + lane < slots && __ldcg(key + k0 + lane) == r);
        while (hit) {
          const int u = __ffs(hit) - 1;
          hit &= hit - 1;
          const int slot = k0 + u;
          const float wk = __ldcg(w + slot);
          const uint2 raw = *reinterpret_cast<const uint2*>(gs + (size_t)(slot / atoms) * channels + c);
          const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), v1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
          acc[0] += wk * __bfloat162float(v0.x); acc[1] += wk * __bfloat162float(v0.y);
          acc[2] += wk * __bfloat162float(v1.x); acc[3] += wk * __bfloat162float(v1.y);
        }
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

// d x of the pooling: ``count`` [R] int32 zeroed by pool_atoms_b200 (train) -- consumed here (the prep kernel counts into it).
at::Tensor pool_atoms_bwd_b200(at::Tensor gout, at::Tensor x, at::Tensor idx, at::Tensor gather_mask, at::Tensor atom_mask, at::Tensor count) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 3, "x: contiguous bf16 cuda [S, Q, C]");
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && gather_mask.is_contiguous() && atom_mask.is_contiguous() && idx.dim() == 2 &&
              gather_mask.sizes() == idx.sizes() && atom_mask.sizes() == idx.sizes(), "idx / masks: contiguous [T, A]");
  TORCH_CHECK(gout.is_contiguous() && gout.scalar_type() == at::kBFloat16 && gout.dim() == 3 && gout.size(0) == x.size(0) && gout.size(2) == x.size(2) &&
              gout.size(1) == idx.size(0), "gout: contiguous bf16 [S, T, C]");
  const int64_t S = x.size(0), R = x.size(1), C = x.size(2), T = idx.size(0), A = idx.size(1), n = T * A;
  TORCH_CHECK(count.is_contiguous() && count.scalar_type() == at::kInt && count.numel() == R, "count: int32 [R]");
  TORCH_CHECK(C % 4 == 0 && n < INT_MAX && R < INT_MAX / CAP, "channels % 4 == 0; slots / rows");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto gx = at::empty_like(x);
  if (gx.numel() == 0) return gx;
  auto key = at::empty({n}, idx.options().dtype(at::kInt));
  auto w = at::empty({n}, idx.options().dtype(at::kFloat));
  auto bucket = at::empty({R * CAP}, idx.options().dtype(at::kInt));
  auto s = at::cuda::getCurrentCUDAStream();
  if (n > 0)
    launch(pool_prep_b200_k, dim3((unsigned)((n + 255) / 256)), dim3(256), s, false, idx.data_ptr(), code_of(idx), gather_mask.data_ptr(),
           code_of(gather_mask), atom_mask.data_ptr(), code_of(atom_mask), count.data_ptr<int>(), bucket.data_ptr<int>(), key.data_ptr<int>(),
           w.data_ptr<float>(), (int)A, (int)T, (int)R);
  const int64_t row_blocks = (R + BWD_ROWS - 1) / BWD_ROWS, target = 32LL * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t want = std::max<int64_t>(1, (target + row_blocks - 1) / row_blocks);
  const int spb = (int)std::min<int64_t>(BWD_SAMPLES, std::max<int64_t>(1, (S + want - 1) / want));
  const dim3 grid((unsigned)row_blocks, (unsigned)((S + spb - 1) / spb)), block(32, BWD_ROWS);
  launch(pool_bwd_b200_k, grid, block, s, n > 0, reinterpret_cast<const __nv_bfloat16*>(gout.data_ptr()),
         reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (const int*)count.data_ptr<int>(), (const int*)bucket.data_ptr<int>(),
         (const int*)key.data_ptr<int>(), (const float*)w.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(gx.data_ptr()), (int)A, (int)T, (int)R,
         (int)C, (int)S, (int)n, spb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gx;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("augment_noise", &augment_noise_b200);
  m.def("align_euler", &align_euler_b200);
  m.def("pool_atoms", [](at::Tensor x, at::Tensor idx, at::Tensor gm, at::Tensor am) { return pool_atoms_b200(x, idx, gm, am, false)[0]; });
  m.def("pool_atoms_train", &pool_atoms_b200);
  m.def("pool_atoms_bwd_b200", &pool_atoms_bwd_b200);
  m.def("pool_atoms_prep", &pool_atoms_prep);
  m.def("pool_atoms_bwd", &pool_atoms_bwd);
  m.def("set_cta_per_sm", [](int64_t v) { g_cta_per_sm = (int)std::max<int64_t>(v, 1); });
  m.def("set_pool_b200_max", [](int64_t v) { g_pool_b200_max = v; });
}
