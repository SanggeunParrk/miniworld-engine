// sampler_ops_sm80.cu -- the A100 (sm_80) build of sampler_ops.cu: the same kernels, launched plainly (no programmatic dependent launch, which needs
// sm_90). sampler_ops.cu is compiled as it is (the H100 build's source stays byte-identical); this wrapper only turns its two PDL pieces into no-ops:
//   * the device side: ``asm volatile("griddepcontrol...")`` in pdl_wait / pdl_trigger becomes an empty asm statement (every header is included
//     first, so the macro reaches sampler_ops.cu's own two statements only);
//   * the host side: the launch attribute becomes cudaLaunchAttributeIgnore, so each kernel waits for the previous one by stream order.
// and adds one A100 variant: augment_noise of a small launch as ONE kernel (augment_noise_one_k below). Without PDL the two-pass chain (moments_k,
// augment_noise_k) pays two launch latencies and two cold round trips for a few microseconds of work; for a small structure every block instead
// sums the whole sample's weights and moments itself (the sample is a few hundred KB, read from L2 after the first block's misses) and then moves
// its own chunk. (A cooperative launch of the two passes with a grid barrier was measured slower: its launch costs more than the second kernel.)
// sampler_ops.cu's own module definition is renamed (unused) so that this file's module binds the dispatcher as ``augment_noise``.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cuda_bf16.h>

#ifndef MWE_SM120  // the RTX 5090 (sm_120a build, kernels/_sm120.py) has programmatic dependent launch: the H100 chain keeps it there
#define volatile(...) volatile("")
#define cudaLaunchAttributeProgrammaticStreamSerialization cudaLaunchAttributeIgnore
#endif
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void sampler_ops_base_bindings(pybind11::module_& variable)
#include "sampler_ops.cu"
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")
#ifndef MWE_SM120
#undef volatile
#undef cudaLaunchAttributeProgrammaticStreamSerialization
#endif

namespace {
//: The one-kernel path's bound on the atoms all its blocks read together (B * C * N): every block re-reads its sample, so the redundant L2 traffic
//: is B * C * N * 16 bytes (4 MB here, about a microsecond of L2 bandwidth); larger launches take the two-pass chain.
constexpr int64_t ONE_KERNEL_READS = 1 << 18;
constexpr int SUM_BATCH = 32;

// ---- augment + re-noise in one kernel: the block's own atoms (augment_noise_k's mapping) are loaded first, then the block sums w, w p over its whole
// sample (thread t takes atoms t, t + THREADS, ... ; one block tree), then the per-atom arithmetic of augment_noise_k.
template <int K>
__global__ void __launch_bounds__(THREADS) augment_noise_one_k(const float* __restrict__ p, const float* __restrict__ w, const float* __restrict__ rv,
                                                               const float* __restrict__ tr, const float* __restrict__ noise,
                                                               const float* __restrict__ sigma, const float* __restrict__ sigma_next, Churn cfg,
                                                               float* noisy, float* sc, int n) {
  __shared__ float red[4 * 8], s_rot[9], s_t[3], s_mag;
  const int b = blockIdx.y, base = blockIdx.x * (THREADS * K);
  float pv[K][3], nv[K][3], wv[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = base + k * THREADS + threadIdx.x;
    if (i < n) {
      const size_t o = ((size_t)b * n + i) * 3;
      pv[k][0] = p[o]; pv[k][1] = p[o + 1]; pv[k][2] = p[o + 2];
      nv[k][0] = noise[o]; nv[k][1] = noise[o + 1]; nv[k][2] = noise[o + 2];
      wv[k] = w[i];
    }
  }
  if (threadIdx.x == 0) {
    const float* r = rv + b * 6;   // v0 = r[0..3), v1 = r[3..6)
    float n0 = fmaxf(sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), 1e-10f);
    const float e0x = r[0] / n0, e0y = r[1] / n0, e0z = r[2] / n0;
    const float d = r[3] * e0x + r[4] * e0y + r[5] * e0z;
    const float ux = r[3] - e0x * d, uy = r[4] - e0y * d, uz = r[5] - e0z * d;
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
  // the whole sample, SUM_BATCH atoms per thread in flight at once (every load of a batch issued before its sums: the pass is latency bound)
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  const float* ps = p + (size_t)b * n * 3;
  for (int i0 = threadIdx.x; i0 < n; i0 += SUM_BATCH * THREADS) {
    float bw[SUM_BATCH], bx[SUM_BATCH], by[SUM_BATCH], bz[SUM_BATCH];
#pragma unroll
    for (int u = 0; u < SUM_BATCH; ++u) {
      const int i = i0 + u * THREADS;
      bw[u] = 0.f; bx[u] = by[u] = bz[u] = 0.f;
      if (i < n) { bw[u] = w[i]; bx[u] = ps[i * 3]; by[u] = ps[i * 3 + 1]; bz[u] = ps[i * 3 + 2]; }
    }
#pragma unroll
    for (int u = 0; u < SUM_BATCH; ++u) {
      if (i0 + u * THREADS < n) { v[0] += bw[u]; v[1] += bw[u] * bx[u]; v[2] += bw[u] * by[u]; v[3] += bw[u] * bz[u]; }
    }
  }
  block_sums<4>(v, red);   // its barriers also publish s_rot / s_t / s_mag
  const float tot = fmaxf(red[0], 1e-6f);
  const float c0 = red[8] / tot, c1 = red[16] / tot, c2 = red[24] / tot;
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

// augment_noise on the A100: one kernel when B * C * N <= ONE_KERNEL_READS, else the two-pass chain of sampler_ops.cu.
std::vector<at::Tensor> augment_noise_sm80(at::Tensor coords, at::Tensor mask, at::Tensor rv, at::Tensor tr, at::Tensor noise,
                                           at::Tensor sigma, at::Tensor sigma_next, int64_t window, double gamma, double gamma_min,
                                           double sigma_min, double sigma_max, double noise_scale, double variance_floor, bool clamp) {
  check(coords, "coords", 3); check(mask, "mask", 1); check(noise, "noise", 3);
  const int64_t B = coords.size(0), N = coords.size(1);
  const int chunk = chunk_for(N, B);
  const int C = (int)((N + chunk - 1) / chunk);
  if (B * C * N > ONE_KERNEL_READS || B > 65535)
    return augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale,
                         variance_floor, clamp);
  TORCH_CHECK(coords.size(2) == 3 && mask.size(0) == N && noise.sizes() == coords.sizes(), "shapes");
  TORCH_CHECK(rv.numel() == B * 6 && tr.numel() == B * 3 && rv.scalar_type() == at::kFloat && tr.scalar_type() == at::kFloat &&
              rv.is_contiguous() && tr.is_contiguous(), "rotation / translation draws");
  TORCH_CHECK(sigma.numel() == 1 && sigma_next.numel() == 1 && sigma.scalar_type() == at::kFloat && sigma_next.scalar_type() == at::kFloat, "sigma scalars");
  const at::cuda::OptionalCUDAGuard guard(coords.device());
  auto noisy = at::empty_like(coords);
  auto sc = at::empty({4}, coords.options());
  Churn cfg{(int)window, (float)(1.0 + gamma), (float)gamma_min, (float)sigma_min, (float)sigma_max, (float)noise_scale, (float)variance_floor, 0.f, clamp ? 1 : 0};
  const dim3 grid(C, (unsigned)B);
  auto s = at::cuda::getCurrentCUDAStream();
#define MW_ONE(KK)                                                                                                                                   \
  augment_noise_one_k<KK><<<grid, THREADS, 0, s>>>(coords.data_ptr<float>(), mask.data_ptr<float>(), rv.data_ptr<float>(), tr.data_ptr<float>(),   \
                                                   noise.data_ptr<float>(), sigma.data_ptr<float>(), sigma_next.data_ptr<float>(), cfg,             \
                                                   noisy.data_ptr<float>(), sc.data_ptr<float>(), (int)N)
  if (chunk == 1024) { MW_ONE(4); } else if (chunk == 512) { MW_ONE(2); } else { MW_ONE(1); }
#undef MW_ONE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {noisy, sc};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("augment_noise", &augment_noise_sm80);
  m.def("align_euler", &align_euler);
  m.def("pool_atoms", &pool_atoms);
  m.def("pool_atoms_prep", &pool_atoms_prep);
  m.def("pool_atoms_bwd", &pool_atoms_bwd);
}
