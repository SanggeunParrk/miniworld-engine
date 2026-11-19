// ieee_f32.cu -- IEEE fp32 GEMM (FFMA, no tensor cores, no TF32) with fused epilogues, and the row kernels of an fp32 Transition (A100 sm_80, built for
// sm_120a on the RTX 5090: static shared memory only, 34 KB a block, no TMA / cp.async.bulk).
//
// One 128 x 128 x 16 block tile (64 x 128 when a small M would leave SMs idle), 256 threads, an 8 x 8 register tile a thread, a double-buffered shared tile fed through registers (one barrier a k step):
//   A: LA_MK  [M, K] row major (lda)        or LA_KM [K, M] row major (the transposed operand of a weight gradient)
//   B: LB_NK  [N, K] row major (a weight [out, in])   or LB_KN [K, N] row major; rows below ``bsplit`` come from b0, the rest from b1 (the stacked [Wa; Wb] without a copy)
// Split-K (a weight gradient, a small M with a long K) sums its slabs in slab order.
// Epilogues:
//   EPI_STORE   c[m, n] = acc                      (split-K: slab z at c + z * slab, summed by reduce_slabs in a fixed order)
//   EPI_RES     c[m, n] = acc + r[m, n]
//   EPI_SWIGLU  the block's 128 columns are 64 of Wa and the same 64 of Wb (b0 = Wa, b1 = Wb, both NK): h = silu(a) * b -> c [M, H]; a | b -> aux [M, 2 H] when given
//   EPI_DAB     acc = dh [M, H]; with the saved a | b (r, ldr = 2 H): d a = dh b silu'(a), d b = dh silu(a) -> aux [M, 2 H] (d a | d b)
// The products and the sums are single-precision fused multiply-adds in a fixed order (k ascending inside a block, split-K slabs ascending): bitwise
// repeatable, no atomics. expf is the accurate one (no fast-math).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <vector>

#include "ieee_f32_gemm.cuh"

namespace {

// the SM count of the current device (the attribute query: torch's cudaDeviceProp is laid out by the CUDA it was built with, which need not be the toolkit nvcc compiles against here)
int sm_count() {
  int dev = 0, n = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  return n;
}

// The launch shape: a tile (128 x 128 by 256 threads, 64 x 128 or 64 x 64 by 128) and a split-K depth gz (the k range cut in gz slabs summed afterwards in slab order: bitwise
// repeatable).  Chosen by a cost model fitted to timings of every (tile, gz) over the shapes of a Protenix v1 fp32 training step on the RTX 5090 (170 SMs): the busiest SM's
// FFMA work (a SM alone with a CTA runs slower per FFMA than with several), a constant per launch, and for gz > 1 the slab round trip.  Depends on the shape and the SM count only.
struct Plan { int cfg; int gz; };
struct CfgInfo { int bm, bn; double s1, s2, c0; };  // us per million FFMA of the busiest SM (one CTA a SM / several), constant (us)
// B stored [K, N] (the backward GEMMs) and B stored [N, K] (the forward weights, transposed through shared memory) fit differently
constexpr CfgInfo CFGS_KN[3] = {{128, 128, 4.122, 4.504, 9.839}, {64, 128, 6.470, 5.704, -9.680}, {64, 64, 8.051, 6.132, -15.081}};
constexpr CfgInfo CFGS_NK[3] = {{128, 128, 6.422, 5.992, -11.417}, {64, 128, 9.152, 5.431, -9.661}, {64, 64, 11.966, 5.946, -8.770}};

Plan plan(long M, long N, long K, bool splittable, bool wide_only, bool kn) {
  const CfgInfo* CFGS = kn ? CFGS_KN : CFGS_NK;
  const int sms = sm_count();
  static const int gzs[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 85, 96, 128};
  Plan best{0, 1};
  double best_t = 1e30;
  for (int ci = 0; ci < 3; ++ci) {
    if (wide_only && CFGS[ci].bn != 128) continue;
    const long tiles = ((N + CFGS[ci].bn - 1) / CFGS[ci].bn) * ((M + CFGS[ci].bm - 1) / CFGS[ci].bm);
    for (int gz : gzs) {
      if (gz > 1 && (!splittable || K / gz < 64 || tiles * gz > 6000 || (double)gz * M * N > 268435456.0)) continue;
      const long ncta = tiles * gz;
      const long per = (ncta + sms - 1) / sms;
      const double ks = (double)((K + gz - 1) / gz);
      const double f = (double)per * CFGS[ci].bm * CFGS[ci].bn * ks / 1e6;
      double t = (per == 1 ? CFGS[ci].s1 : CFGS[ci].s2) * f + CFGS[ci].c0;
      if (ci != 0) t *= 1.08;  // hysteresis: the 128 x 128 tile unless another is clearly faster
      if (gz > 1) t += 2.5 + (gz + 1) * (double)M * N * 4.0 / 1.4e6;
      if (t < best_t) { best_t = t; best = {ci, gz}; }
    }
  }
  return best;
}

template <int LA, int LB, int EPI, int BMT, int BNT, int NT, int MINB>
void launch_cfg(Args& p, int gz) {
  const unsigned gx = (EPI == EPI_SWIGLU) ? (unsigned)((p.N + 63) / 64) : (unsigned)((p.N + BNT - 1) / BNT);
  const dim3 grid(gx, (unsigned)((p.M + BMT - 1) / BMT), (unsigned)gz);
  gemm_k<LA, LB, EPI, BMT, BNT, NT, MINB><<<grid, dim3(NT), 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int LA, int LB, int EPI>
void launch(Args& p, Plan pl) {
  if (pl.cfg == 0) launch_cfg<LA, LB, EPI, 128, 128, 256, 1>(p, pl.gz);
  else if (pl.cfg == 1) launch_cfg<LA, LB, EPI, 64, 128, 128, 3>(p, pl.gz);
  else if constexpr (EPI != EPI_SWIGLU) launch_cfg<LA, LB, EPI, 64, 64, 128, 4>(p, pl.gz);  // (the SwiGLU tile is 128 wide: plan() never picks the 64-wide one for it)
}
// out [M, N] = sum of the slabs [gz, M, N] (+ res) in slab order
__global__ void sum_slabs_k(const float* __restrict__ slabs, const float* __restrict__ res, float* __restrict__ out, int gz, long n4, long slab4, int cols4, long ldo) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 s = reinterpret_cast<const float4*>(slabs)[i];
  for (int z = 1; z < gz; ++z) {
    const float4 t = reinterpret_cast<const float4*>(slabs)[z * slab4 + i];
    s.x += t.x; s.y += t.y; s.z += t.z; s.w += t.w;
  }
  if (res != nullptr) {
    const float4 r = reinterpret_cast<const float4*>(res)[i];
    s.x += r.x; s.y += r.y; s.z += r.z; s.w += r.w;
  }
  *reinterpret_cast<float4*>(out + (i / cols4) * ldo + (i % cols4) * 4) = s;
}

at::Tensor sum_slabs(const at::Tensor& slabs, const at::Tensor* res, const c10::optional<at::Tensor>& into = c10::nullopt) {
  const int gz = (int)slabs.size(0);
  auto out = into.has_value() ? *into : at::empty({slabs.size(1), slabs.size(2)}, slabs.options());
  const long n4 = slabs.size(1) * slabs.size(2) / 4;
  sum_slabs_k<<<(unsigned)((n4 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(slabs.data_ptr<float>(), res ? res->data_ptr<float>() : nullptr, out.data_ptr<float>(), gz, n4, n4, (int)(slabs.size(2) / 4), out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void check_f32(const at::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat, what, ": a float32 cuda tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, what, ": 16-byte aligned");
}

}  // namespace

namespace {
template <int EPI>
void run_layout(Args& p, int64_t la, int64_t lb, Plan pl) {
  if (la == LA_MK && lb == LB_NK) launch<LA_MK, LB_NK, EPI>(p, pl);
  else if (la == LA_MK && lb == LB_KN) launch<LA_MK, LB_KN, EPI>(p, pl);
  else if (la == LA_KM && lb == LB_NK) launch<LA_KM, LB_NK, EPI>(p, pl);
  else launch<LA_KM, LB_KN, EPI>(p, pl);
}
// the k range of one split-K slab, a multiple of BK
int kper_of(int64_t K, int gz) { return (int)(((K + gz - 1) / gz + BK - 1) / BK * BK); }
}  // namespace

// out [M, N] = A B (+ bias[n]); la / lb: 0 = [M,K] / [N,K], 1 = [K,M] / [K,N]. Split-K (a weight gradient, a small M) is summed in slab order.
at::Tensor mm(at::Tensor a, int64_t lda, int64_t la, at::Tensor b, int64_t ldb, int64_t lb, int64_t M, int64_t N, int64_t K, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> into) {
  if (into.has_value()) {
    check_f32(*into, "out");
    TORCH_CHECK(into->dim() == 2 && into->size(0) == M && into->size(1) == N && into->stride(1) == 1 && into->stride(0) % 4 == 0, "out: [M, N] with unit column stride, row stride a multiple of 4");
  }
  check_f32(a, "a"); check_f32(b, "b");
  TORCH_CHECK(N % 4 == 0 && lda % 4 == 0 && ldb % 4 == 0 && (la == LA_KM || K % 4 == 0) && (lb == LB_KN || K % 4 == 0) && (la == LA_MK || M % 4 == 0),
              "ieee mm: N and the leading dimensions multiples of 4, K for a row-major k operand, M for a transposed A");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  Args p{};
  p.a = a.data_ptr<float>(); p.lda = lda; p.b0 = b.data_ptr<float>(); p.b1 = p.b0; p.ldb = ldb; p.bsplit = INT_MAX;
  p.M = (int)M; p.N = (int)N; p.K = (int)K;
  if (bias.has_value()) {  // the residual epilogue with a zero row stride: one pass, no split
    check_f32(*bias, "bias");
    TORCH_CHECK(bias->is_contiguous() && bias->numel() == N, "bias: [N]");
    auto out = into.has_value() ? *into : at::empty({M, N}, a.options());
    p.kper = (int)K; p.c = out.data_ptr<float>(); p.ldc = out.stride(0); p.r = bias->data_ptr<float>(); p.ldr = 0;
    run_layout<EPI_RES>(p, la, lb, plan(M, N, K, false, false, lb == LB_KN));
    return out;
  }
  Plan pl = plan(M, N, K, true, false, lb == LB_KN);
  p.kper = kper_of(K, pl.gz);
  pl.gz = (int)((K + p.kper - 1) / p.kper);
  if (pl.gz == 1) {  // straight into the result (``into``: a caller's buffer, a row stride allowed)
    auto res = into.has_value() ? *into : at::empty({M, N}, a.options());
    p.c = res.data_ptr<float>(); p.ldc = res.stride(0); p.slab = 0;
    run_layout<EPI_STORE>(p, la, lb, pl);
    return res;
  }
  auto out = at::empty({pl.gz, M, N}, a.options());
  p.c = out.data_ptr<float>(); p.ldc = N; p.slab = M * N;
  run_layout<EPI_STORE>(p, la, lb, pl);
  return sum_slabs(out, nullptr, into);
}

// out [M, N] = A [M, K] [b0; b1] ([K, N]: rows < b0.size(0) from b0), A MK, B KN
at::Tensor mm_stacked(at::Tensor a, at::Tensor b0, at::Tensor b1) {
  check_f32(a, "a"); check_f32(b0, "b0"); check_f32(b1, "b1");
  const int64_t M = a.size(0), K = a.size(1), N = b0.size(1), kb = b0.size(0);
  TORCH_CHECK(a.is_contiguous() && b0.is_contiguous() && b1.is_contiguous() && b1.size(1) == N && kb + b1.size(0) == K, "mm_stacked: shapes");
  TORCH_CHECK(N % 4 == 0 && K % 4 == 0, "mm_stacked: N, K multiples of 4");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  Args p{};
  p.a = a.data_ptr<float>(); p.lda = K; p.b0 = b0.data_ptr<float>(); p.b1 = b1.data_ptr<float>(); p.ldb = N; p.bsplit = (int)kb;
  p.M = (int)M; p.N = (int)N; p.K = (int)K;
  Plan pl = plan(M, N, K, true, false, true);
  p.kper = kper_of(K, pl.gz);
  pl.gz = (int)((K + p.kper - 1) / p.kper);
  auto out = at::empty({pl.gz, M, N}, a.options());
  p.c = out.data_ptr<float>(); p.ldc = N; p.slab = M * N;
  launch<LA_MK, LB_KN, EPI_STORE>(p, pl);
  return pl.gz == 1 ? out.view({M, N}) : sum_slabs(out, nullptr);
}

// h = silu(x Wa^T) (x Wb^T): x [M, K] MK, wa / wb [H, K]; returns (h [M, H], ab [M, 2H] or empty)
std::vector<at::Tensor> swiglu_fwd(at::Tensor x, at::Tensor wa, at::Tensor wb, bool save) {
  check_f32(x, "x"); check_f32(wa, "wa"); check_f32(wb, "wb");
  const int64_t M = x.size(0), K = x.size(1), H = wa.size(0);
  TORCH_CHECK(x.is_contiguous() && wa.is_contiguous() && wb.is_contiguous() && wa.sizes() == wb.sizes() && wa.size(1) == K, "swiglu_fwd: shapes");
  TORCH_CHECK(H % 4 == 0 && K % 4 == 0, "swiglu_fwd: H, K multiples of 4");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto h = at::empty({M, H}, x.options());
  auto ab = save ? at::empty({M, 2 * H}, x.options()) : at::empty({0}, x.options());
  Args p{};
  p.a = x.data_ptr<float>(); p.lda = K; p.b0 = wa.data_ptr<float>(); p.b1 = wb.data_ptr<float>(); p.ldb = K; p.bsplit = INT_MAX;
  p.c = h.data_ptr<float>(); p.ldc = H; p.aux = save ? ab.data_ptr<float>() : nullptr; p.ldaux = 2 * H;
  p.M = (int)M; p.N = (int)H; p.K = (int)K; p.kper = (int)K; p.H = (int)H;
  launch<LA_MK, LB_NK, EPI_SWIGLU>(p, plan(M, 2 * H, K, false, true, false));
  return {h, ab};
}

// y = x + h Ws^T: h [M, H] MK, ws [D, H] NK
at::Tensor gemm_res(at::Tensor h, at::Tensor ws, at::Tensor x) {
  check_f32(h, "h"); check_f32(ws, "ws"); check_f32(x, "x");
  const int64_t M = h.size(0), K = h.size(1), N = ws.size(0);
  TORCH_CHECK(h.is_contiguous() && ws.is_contiguous() && x.is_contiguous() && ws.size(1) == K && x.size(0) == M && x.size(1) == N, "gemm_res: shapes");
  TORCH_CHECK(N % 4 == 0 && K % 4 == 0, "gemm_res: multiples of 4");
  const at::cuda::OptionalCUDAGuard guard(h.device());
  Args p{};
  p.a = h.data_ptr<float>(); p.lda = K; p.b0 = ws.data_ptr<float>(); p.b1 = p.b0; p.ldb = K; p.bsplit = INT_MAX;
  p.M = (int)M; p.N = (int)N; p.K = (int)K;
  Plan pl = plan(M, N, K, true, false, false);
  p.kper = kper_of(K, pl.gz);
  pl.gz = (int)((K + p.kper - 1) / p.kper);
  if (pl.gz == 1) {
    auto y = at::empty({M, N}, h.options());
    p.c = y.data_ptr<float>(); p.ldc = N; p.r = x.data_ptr<float>(); p.ldr = N;
    launch<LA_MK, LB_NK, EPI_RES>(p, pl);
    return y;
  }
  auto slabs = at::empty({pl.gz, M, N}, h.options());  // split-K: the slabs, then their sum with the residual
  p.c = slabs.data_ptr<float>(); p.ldc = N; p.slab = M * N;
  launch<LA_MK, LB_NK, EPI_STORE>(p, pl);
  return sum_slabs(slabs, &x);
}

// dab [M, 2H] = the SwiGLU backward of dh = dy Ws: dy [M, D] MK, ws [D, H] KN, ab [M, 2H] the forward's a | b
at::Tensor dab_gemm(at::Tensor dy, at::Tensor ws, at::Tensor ab) {
  check_f32(dy, "dy"); check_f32(ws, "ws"); check_f32(ab, "ab");
  const int64_t M = dy.size(0), K = dy.size(1), H = ws.size(1);
  TORCH_CHECK(dy.is_contiguous() && ws.is_contiguous() && ab.is_contiguous() && ws.size(0) == K && ab.size(0) == M && ab.size(1) == 2 * H, "dab_gemm: shapes");
  TORCH_CHECK(H % 4 == 0 && K % 4 == 0, "dab_gemm: multiples of 4");
  const at::cuda::OptionalCUDAGuard guard(dy.device());
  auto dab = at::empty({M, 2 * H}, dy.options());
  Args p{};
  p.a = dy.data_ptr<float>(); p.lda = K; p.b0 = ws.data_ptr<float>(); p.b1 = p.b0; p.ldb = H; p.bsplit = INT_MAX;
  p.r = ab.data_ptr<float>(); p.ldr = 2 * H; p.aux = dab.data_ptr<float>(); p.ldaux = 2 * H;
  p.M = (int)M; p.N = (int)H; p.K = (int)K; p.kper = (int)K; p.H = (int)H;
  launch<LA_MK, LB_KN, EPI_DAB>(p, plan(M, H, K, false, false, true));
  return dab;
}

// ---------------------------------------------------------------------------------------------------- row kernels
// out [N] = sum over the P slabs of in [P, N], in slab order
__global__ void reduce_slabs_k(const float* __restrict__ in, float* __restrict__ out, int P, long N) {
  __shared__ float part[8][33];
  const long n = (long)blockIdx.x * 32 + threadIdx.x;
  float s = 0.f;
  if (n < N)
    for (int q = threadIdx.y; q < P; q += 8) s += in[(long)q * N + n];
  part[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.y == 0 && n < N) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) t += part[w][threadIdx.x];
    out[n] = t;
  }
}
at::Tensor reduce_slabs(at::Tensor in) {
  check_f32(in, "in");
  TORCH_CHECK(in.is_contiguous() && in.dim() >= 2, "reduce_slabs: contiguous [P, ...]");
  const at::cuda::OptionalCUDAGuard guard(in.device());
  const int P = (int)in.size(0);
  const long N = in.numel() / P;
  auto shape = in.sizes().vec();
  shape.erase(shape.begin());
  auto out = at::empty(shape, in.options());
  reduce_slabs_k<<<(unsigned)((N + 31) / 32), dim3(32, 8), 0, at::cuda::getCurrentCUDAStream()>>>(in.data_ptr<float>(), out.data_ptr<float>(), P, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

constexpr int LN_MAX = 8;  // float4 per lane: D <= 1024 (the kernels are instantiated per ceil(D / 128))

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// one warp a row: xn = (x - mean) rstd gamma + beta; (mean, rstd) [M, 2] saved when ``stats`` is given
template <int LN_MAXV>
__global__ void ln_fwd_k(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ xn,
                         float* __restrict__ stats, int M, int D, float eps) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float4 v[LN_MAXV];
  float s = 0.f;
#pragma unroll
  for (int q = 0; q < LN_MAXV; ++q) {
    const int c = (q * 32 + lane) * 4;
    v[q] = c < D ? *reinterpret_cast<const float4*>(x + (long)row * D + c) : f4zero();
    s += v[q].x + v[q].y + v[q].z + v[q].w;
  }
  const float mean = warp_sum(s) / (float)D;
  float t = 0.f;
#pragma unroll
  for (int q = 0; q < LN_MAXV; ++q) {
    const int c = (q * 32 + lane) * 4;
    if (c < D) {
      const float a = v[q].x - mean, b = v[q].y - mean, e = v[q].z - mean, f = v[q].w - mean;
      t += a * a + b * b + e * e + f * f;
    }
  }
  const float rstd = rsqrtf(warp_sum(t) / (float)D + eps);
#pragma unroll
  for (int q = 0; q < LN_MAXV; ++q) {
    const int c = (q * 32 + lane) * 4;
    if (c < D) {
      const float4 g = *reinterpret_cast<const float4*>(gamma + c), bb = *reinterpret_cast<const float4*>(beta + c);
      *reinterpret_cast<float4*>(xn + (long)row * D + c) =
          make_float4((v[q].x - mean) * rstd * g.x + bb.x, (v[q].y - mean) * rstd * g.y + bb.y, (v[q].z - mean) * rstd * g.z + bb.z,
                      (v[q].w - mean) * rstd * g.w + bb.w);
    }
  }
  if (stats != nullptr && lane == 0) { stats[2 * row] = mean; stats[2 * row + 1] = rstd; }
}

std::vector<at::Tensor> ln_fwd(at::Tensor x, at::Tensor gamma, at::Tensor beta, double eps, bool save) {
  check_f32(x, "x"); check_f32(gamma, "gamma"); check_f32(beta, "beta");
  TORCH_CHECK(x.is_contiguous() && x.dim() == 2 && x.size(1) % 4 == 0 && x.size(1) <= LN_MAX * 128, "ln_fwd: [M, D], D % 4 == 0, D <= 1024");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int M = (int)x.size(0), D = (int)x.size(1);
  auto xn = at::empty_like(x);
  auto stats = save ? at::empty({M, 2}, x.options()) : at::empty({0}, x.options());
  const int nv = (D + 127) / 128;
  const float* xp = x.data_ptr<float>(); const float* gp = gamma.data_ptr<float>(); const float* bp = beta.data_ptr<float>();
  float* xnp = xn.data_ptr<float>(); float* sp = save ? stats.data_ptr<float>() : nullptr;
  const unsigned gridn = (unsigned)((M + 7) / 8);
  auto st = at::cuda::getCurrentCUDAStream();
#define LNF(N) case N: ln_fwd_k<N><<<gridn, 256, 0, st>>>(xp, gp, bp, xnp, sp, M, D, (float)eps); break;
  switch (nv) { LNF(1) LNF(2) LNF(3) LNF(4) LNF(5) LNF(6) LNF(7) default: LNF(8) }
#undef LNF
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {xn, stats};
}

// dx = dy + LN backward of dxn (the residual branch), and the per-block partials of d gamma | d beta ([G, 2 D])
template <int LN_MAXV>
__global__ void __launch_bounds__(256) ln_bwd_k(const float* __restrict__ dxn, const float* __restrict__ x, const float* __restrict__ stats,
                                                const float* __restrict__ gamma, const float* __restrict__ dy, float* __restrict__ dx,
                                                float* __restrict__ part, int M, int D) {
  __shared__ __align__(16) float sg[8][LN_MAXV * 128];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float4 dg[LN_MAXV], db[LN_MAXV];
#pragma unroll
  for (int q = 0; q < LN_MAXV; ++q) { dg[q] = f4zero(); db[q] = f4zero(); }
  for (int row = blockIdx.x * 8 + warp; row < M; row += gridDim.x * 8) {
    const float mean = stats[2 * row], rstd = stats[2 * row + 1];
    float4 xh[LN_MAXV], dh[LN_MAXV];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int q = 0; q < LN_MAXV; ++q) {
      const int c = (q * 32 + lane) * 4;
      if (c < D) {
        const float4 xv = *reinterpret_cast<const float4*>(x + (long)row * D + c), g = *reinterpret_cast<const float4*>(gamma + c);
        const float4 d = *reinterpret_cast<const float4*>(dxn + (long)row * D + c);
        xh[q] = make_float4((xv.x - mean) * rstd, (xv.y - mean) * rstd, (xv.z - mean) * rstd, (xv.w - mean) * rstd);
        dg[q].x += d.x * xh[q].x; dg[q].y += d.y * xh[q].y; dg[q].z += d.z * xh[q].z; dg[q].w += d.w * xh[q].w;
        db[q].x += d.x; db[q].y += d.y; db[q].z += d.z; db[q].w += d.w;
        dh[q] = make_float4(d.x * g.x, d.y * g.y, d.z * g.z, d.w * g.w);
        s1 += dh[q].x + dh[q].y + dh[q].z + dh[q].w;
        s2 += dh[q].x * xh[q].x + dh[q].y * xh[q].y + dh[q].z * xh[q].z + dh[q].w * xh[q].w;
      }
    }
    const float m1 = warp_sum(s1) / (float)D, m2 = warp_sum(s2) / (float)D;
#pragma unroll
    for (int q = 0; q < LN_MAXV; ++q) {
      const int c = (q * 32 + lane) * 4;
      if (c < D) {
        const float4 r = *reinterpret_cast<const float4*>(dy + (long)row * D + c);
        *reinterpret_cast<float4*>(dx + (long)row * D + c) =
            make_float4(r.x + rstd * (dh[q].x - m1 - xh[q].x * m2), r.y + rstd * (dh[q].y - m1 - xh[q].y * m2),
                        r.z + rstd * (dh[q].z - m1 - xh[q].z * m2), r.w + rstd * (dh[q].w - m1 - xh[q].w * m2));
      }
    }
  }
#pragma unroll
  for (int q = 0; q < LN_MAXV; ++q) *reinterpret_cast<float4*>(&sg[warp][(q * 32 + lane) * 4]) = dg[q];
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 256) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) a += sg[w][c];
    part[(long)blockIdx.x * 2 * D + c] = a;
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < LN_MAXV; ++q) *reinterpret_cast<float4*>(&sg[warp][(q * 32 + lane) * 4]) = db[q];
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 256) {
    float b = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) b += sg[w][c];
    part[(long)blockIdx.x * 2 * D + D + c] = b;
  }
}

// (dx [M, D], partials [G, 2 D]): d gamma | d beta = reduce_slabs(partials)
std::vector<at::Tensor> ln_bwd(at::Tensor dxn, at::Tensor x, at::Tensor stats, at::Tensor gamma, at::Tensor dy) {
  check_f32(dxn, "dxn"); check_f32(x, "x"); check_f32(stats, "stats"); check_f32(gamma, "gamma"); check_f32(dy, "dy");
  TORCH_CHECK(dxn.is_contiguous() && x.is_contiguous() && dy.is_contiguous() && x.dim() == 2 && x.size(1) % 4 == 0 && x.size(1) <= LN_MAX * 128,
              "ln_bwd: [M, D] contiguous, D % 4 == 0, D <= 1024");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int M = (int)x.size(0), D = (int)x.size(1);
  auto dx = at::empty_like(x);
  const int sms = sm_count();
  const int G = std::max(1, std::min((M + 7) / 8, sms * 8));
  auto part = at::empty({G, 2 * D}, x.options());
  const int nv = (D + 127) / 128;
  const float* dxnp = dxn.data_ptr<float>(); const float* xp = x.data_ptr<float>(); const float* sp = stats.data_ptr<float>(); const float* gp = gamma.data_ptr<float>();
  const float* dyp = dy.data_ptr<float>(); float* dxp = dx.data_ptr<float>(); float* pp = part.data_ptr<float>();
  auto st = at::cuda::getCurrentCUDAStream();
#define LNB(N) case N: ln_bwd_k<N><<<(unsigned)G, 256, 0, st>>>(dxnp, xp, sp, gp, dyp, dxp, pp, M, D); break;
  switch (nv) { LNB(1) LNB(2) LNB(3) LNB(4) LNB(5) LNB(6) LNB(7) default: LNB(8) }
#undef LNB
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, part};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", &mm);
  m.def("mm_stacked", &mm_stacked);
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("gemm_res", &gemm_res);
  m.def("dab_gemm", &dab_gemm);
  m.def("reduce_slabs", &reduce_slabs);
  m.def("ln_fwd", &ln_fwd);
  m.def("ln_bwd", &ln_bwd);
}
