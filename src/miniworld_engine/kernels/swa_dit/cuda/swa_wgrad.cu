// ESMFold2 SWA atom block, the four weight-gradient GEMMs of one block's backward in ONE launch (CUDA sm_90a, bf16 operands, fp32
// accumulate, mma.sync m16n8k16, cp.async pipeline):
//   dWqkv, dWg  = dP^T x        [384, 128], [128, 128]   (dP [M, 4C] = dp_q | dp_k | dv | dG; the first 3C columns give dWqkv, the last C dWg)
//   dWo         = datt^T gated  [128, 128]
//   dWu         = dab^T y       [512, 128]
//   dWd         = dffn^T h      [128, 256]
// Each is a K = M (rows of the batch) reduction: 11 output tiles of 128 x 128. One cluster of 8 CTAs owns a tile and splits K 8 ways
// (contiguous row ranges); each CTA streams its [64 rows][128] slices of both operands through a 3-stage cp.async ring, accumulates the
// 128 x 128 tile in registers (8 warps of 32 x 64), parks it in shared memory, and after a cluster barrier every CTA sums 16 rows of
// the tile over the eight partials IN RANK ORDER through distributed shared memory (deterministic) and stores them as bf16.
// The result is the cuBLAS GEMM's up to summation order (fp32 accumulation, one rounding to bf16); nothing is staged through global.
// ``wgrad_f32`` is the same decomposition on fp32 operands: every staged element is split into hi + lo bf16 and a x b is taken as
// hi_a hi_b + hi_a lo_b + lo_a hi_b (fp32 accumulation, fp32 gradients), closer to fp32 than the TF32 GEMMs of cuBLAS.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace {
using bf16 = __nv_bfloat16;
constexpr int LDS = 136;                       // padded bf16 row stride of a [64][128] slice (272 B: ldmatrix conflict-free)
constexpr int KB = 64;                         // rows per pipeline stage
constexpr int NSTG = 3;
constexpr int XS = 132;                        // fp32 row stride of the parked accumulator tile
constexpr int NJOB = 11;
constexpr int CL = 8;                          // K splits per tile (cluster size)

struct Jobs {
  const bf16* A[NJOB];                         // [M, lda] slice start (the tile's 128 columns)
  const bf16* B[NJOB];
  bf16* out[NJOB];                             // [128 rows][ldo] tile start
  int lda[NJOB], ldb[NJOB], ldo[NJOB];
};

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(256) wgrad_kernel(Jobs jobs, int M, int kper) {
  extern __shared__ __align__(16) unsigned char smem[];
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mi = lane >> 3;
  const int rank = blockIdx.x % CL, job = blockIdx.y;
  const bf16* __restrict__ A = jobs.A[job];
  const bf16* __restrict__ B = jobs.B[job];
  const int lda = jobs.lda[job], ldb = jobs.ldb[job];
  const int k0 = rank * kper, k1 = min(M, k0 + kper);
  const int nk = (k1 > k0) ? (k1 - k0 + KB - 1) / KB : 0;
  bf16* sA = reinterpret_cast<bf16*>(smem);                  // NSTG x [KB][LDS]
  bf16* sB = sA + NSTG * KB * LDS;
  const int wm = (warp & 3) * 32, wn = (warp >> 2) * 64;

  auto load = [&](int it, int stage) {
    const int r = k0 + it * KB;
    bf16* da = sA + stage * KB * LDS;
    bf16* db = sB + stage * KB * LDS;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                            // 64 rows x 16 vectors of 16 B per operand
      const int i = tid + j * 256, row = i >> 4, v = i & 15;
      const bool ok = (r + row) < k1;
      const int rr = ok ? r + row : 0;
      cp16(sa(da + row * LDS + v * 8), A + (size_t)rr * lda + v * 8, ok);
      cp16(sa(db + row * LDS + v * 8), B + (size_t)rr * ldb + v * 8, ok);
    }
  };
  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  for (int s = 0; s < NSTG - 1; ++s) {
    if (s < nk) load(s, s);
    cp_commit();
  }
  for (int it = 0; it < nk; ++it) {
    cp_wait<NSTG - 2>();
    __syncthreads();
    if (it + NSTG - 1 < nk) load(it + NSTG - 1, (it + NSTG - 1) % NSTG);
    cp_commit();
    const bf16* a = sA + (it % NSTG) * KB * LDS;
    const bf16* b = sB + (it % NSTG) * KB * LDS;
    // fragments of k-step ks + 1 are loaded before the MMAs of k-step ks (register double buffer)
    uint32_t af[2][2][4], bb[2][4][4];
    auto frags = [&](int ks, int s) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int kk = ks * 16 + (mi >> 1) * 8 + (lane & 7), mm = wm + mt * 16 + (mi & 1) * 8;
        ldsm4t(sa(a + kk * LDS + mm), af[s][mt][0], af[s][mt][1], af[s][mt][2], af[s][mt][3]);
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int kk = ks * 16 + (mi & 1) * 8 + (lane & 7), nn = wn + np * 16 + (mi >> 1) * 8;
        ldsm4t(sa(b + kk * LDS + nn), bb[s][np][0], bb[s][np][1], bb[s][np][2], bb[s][np][3]);
      }
    };
    frags(0, 0);
#pragma unroll
    for (int ks = 0; ks < KB / 16; ++ks) {
      if (ks + 1 < KB / 16) frags(ks + 1, (ks + 1) & 1);
      const int s = ks & 1;
#pragma unroll
      for (int np = 0; np < 4; ++np)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma(acc[mt][2 * np], af[s][mt], bb[s][np][0], bb[s][np][1]);
          mma(acc[mt][2 * np + 1], af[s][mt], bb[s][np][2], bb[s][np][3]);
        }
    }
  }
  cp_wait<0>();
  __syncthreads();                                           // the ring is free: park the tile in it
  float* X = reinterpret_cast<float*>(smem);                 // [128][XS]
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int row = wm + mt * 16 + g, col = wn + nt * 8 + 2 * t;
      *reinterpret_cast<float2*>(X + row * XS + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<float2*>(X + (row + 8) * XS + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
  cluster.sync();
  {
    // rows 16 * rank .. +16 of the tile: thread -> (row = tid / 16, 8 contiguous columns)
    const int row = rank * 16 + (tid >> 4), col = (tid & 15) * 8;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = 0.f;
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      const float* p = cluster.map_shared_rank(X, r) + row * XS + col;
      const float4 v0 = *reinterpret_cast<const float4*>(p), v1 = *reinterpret_cast<const float4*>(p + 4);
      s[0] += v0.x; s[1] += v0.y; s[2] += v0.z; s[3] += v0.w;
      s[4] += v1.x; s[5] += v1.y; s[6] += v1.z; s[7] += v1.w;
    }
    uint4 o;
    o.x = pack(s[0], s[1]); o.y = pack(s[2], s[3]); o.z = pack(s[4], s[5]); o.w = pack(s[6], s[7]);
    *reinterpret_cast<uint4*>(jobs.out[job] + (size_t)row * jobs.ldo[job] + col) = o;
  }
  cluster.sync();                                            // peers are done reading this CTA's tile
}

// ------------------------------------------------------------------------------------------------------------------------------
// fp32 operands: every staged element is split into a hi and a lo bf16 (x = hi + lo, 2^-17) and a x b = hi_a hi_b + hi_a lo_b + lo_a hi_b
// on the bf16 tensor cores (the lo . lo term is below 2^-17): about 2^-16 relative per product, ~30x closer to fp32 than the TF32
// GEMMs of cuBLAS. KF-row stages arrive raw (fp32) by cp.async through a ring, one barrier before the planes are built and one before
// the MMAs (the planes are a single buffer: every warp has left the previous stage's MMAs before the first barrier).
constexpr int KF = 32, NRF = 3;
constexpr int FRAW = KF * 128 * 4;                           // bytes of one raw fp32 operand tile [KF][128]
constexpr int FPL = KF * LDS * 2;                            // bytes of one bf16 plane [KF][LDS]
struct JobsF {
  const float* A[NJOB];
  const float* B[NJOB];
  float* out[NJOB];
  int lda[NJOB], ldb[NJOB], ldo[NJOB];
};
__device__ __forceinline__ void split2f(float v0, float v1, uint32_t& hi, uint32_t& lo) {
  hi = pack(v0, v1);
  const float h0 = __uint_as_float(hi << 16), h1 = __uint_as_float(hi & 0xffff0000u);
  lo = pack(v0 - h0, v1 - h1);
}
__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(256) wgrad_f32_kernel(JobsF jobs, int M, int kper) {
  extern __shared__ __align__(16) unsigned char smem[];
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mi = lane >> 3;
  const int rank = blockIdx.x % CL, job = blockIdx.y;
  const float* __restrict__ A = jobs.A[job];
  const float* __restrict__ B = jobs.B[job];
  const int lda = jobs.lda[job], ldb = jobs.ldb[job];
  const int k0 = rank * kper, k1 = min(M, k0 + kper);
  const int nk = (k1 > k0) ? (k1 - k0 + KF - 1) / KF : 0;
  float* rawA = reinterpret_cast<float*>(smem);              // NRF x [KF][128]
  float* rawB = rawA + NRF * KF * 128;
  bf16* pA = reinterpret_cast<bf16*>(smem + 2 * NRF * FRAW); // hi, lo planes [KF][LDS]
  bf16* pB = pA + 2 * KF * LDS;
  const int wm = (warp & 3) * 32, wn = (warp >> 2) * 64;
  auto load = [&](int it) {
    const int r = k0 + it * KF, slot = it % NRF;
    float* da = rawA + slot * KF * 128;
    float* db = rawB + slot * KF * 128;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                            // 32 rows x 32 vectors of 16 B per operand
      const int i = tid + j * 256, row = i >> 5, v = i & 31;
      const bool ok = (r + row) < k1;
      const int rr = ok ? r + row : 0;
      cp16(sa(da + row * 128 + v * 4), A + (size_t)rr * lda + v * 4, ok);
      cp16(sa(db + row * 128 + v * 4), B + (size_t)rr * ldb + v * 4, ok);
    }
  };
  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  for (int s = 0; s < NRF - 1; ++s) {
    if (s < nk) load(s);
    cp_commit();
  }
  for (int it = 0; it < nk; ++it) {
    cp_wait<NRF - 2>();
    __syncthreads();
    if (it + NRF - 1 < nk) load(it + NRF - 1);
    cp_commit();
    const float* ra = rawA + (it % NRF) * KF * 128;
    const float* rb = rawB + (it % NRF) * KF * 128;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = tid + j * 256, row = i >> 5, v = i & 31;
      const float4 x = *reinterpret_cast<const float4*>(ra + row * 128 + v * 4), y = *reinterpret_cast<const float4*>(rb + row * 128 + v * 4);
      uint32_t h0, l0, h1, l1;
      split2f(x.x, x.y, h0, l0);
      split2f(x.z, x.w, h1, l1);
      *reinterpret_cast<uint2*>(pA + row * LDS + v * 4) = make_uint2(h0, h1);
      *reinterpret_cast<uint2*>(pA + (KF + row) * LDS + v * 4) = make_uint2(l0, l1);
      split2f(y.x, y.y, h0, l0);
      split2f(y.z, y.w, h1, l1);
      *reinterpret_cast<uint2*>(pB + row * LDS + v * 4) = make_uint2(h0, h1);
      *reinterpret_cast<uint2*>(pB + (KF + row) * LDS + v * 4) = make_uint2(l0, l1);
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < KF / 16; ++ks) {
      uint32_t af[2][2][4];
#pragma unroll
      for (int pl = 0; pl < 2; ++pl)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int kk = ks * 16 + (mi >> 1) * 8 + (lane & 7), mm = wm + mt * 16 + (mi & 1) * 8;
          ldsm4t(sa(pA + (pl * KF + kk) * LDS + mm), af[pl][mt][0], af[pl][mt][1], af[pl][mt][2], af[pl][mt][3]);
        }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int kk = ks * 16 + (mi & 1) * 8 + (lane & 7), nn = wn + np * 16 + (mi >> 1) * 8;
        uint32_t bh[4], bl[4];
        ldsm4t(sa(pB + kk * LDS + nn), bh[0], bh[1], bh[2], bh[3]);
        ldsm4t(sa(pB + (KF + kk) * LDS + nn), bl[0], bl[1], bl[2], bl[3]);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma(acc[mt][2 * np], af[0][mt], bh[0], bh[1]);
          mma(acc[mt][2 * np + 1], af[0][mt], bh[2], bh[3]);
          mma(acc[mt][2 * np], af[0][mt], bl[0], bl[1]);
          mma(acc[mt][2 * np + 1], af[0][mt], bl[2], bl[3]);
          mma(acc[mt][2 * np], af[1][mt], bh[0], bh[1]);
          mma(acc[mt][2 * np + 1], af[1][mt], bh[2], bh[3]);
        }
      }
    }
  }
  cp_wait<0>();
  __syncthreads();
  float* X = reinterpret_cast<float*>(smem);                 // [128][XS]
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int row = wm + mt * 16 + g, col = wn + nt * 8 + 2 * t;
      *reinterpret_cast<float2*>(X + row * XS + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<float2*>(X + (row + 8) * XS + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
  cluster.sync();
  {
    const int row = rank * 16 + (tid >> 4), col = (tid & 15) * 8;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = 0.f;
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      const float* p = cluster.map_shared_rank(X, r) + row * XS + col;
      const float4 v0 = *reinterpret_cast<const float4*>(p), v1 = *reinterpret_cast<const float4*>(p + 4);
      s[0] += v0.x; s[1] += v0.y; s[2] += v0.z; s[3] += v0.w;
      s[4] += v1.x; s[5] += v1.y; s[6] += v1.z; s[7] += v1.w;
    }
    float* o = jobs.out[job] + (size_t)row * jobs.ldo[job] + col;
    *reinterpret_cast<float4*>(o) = make_float4(s[0], s[1], s[2], s[3]);
    *reinterpret_cast<float4*>(o + 4) = make_float4(s[4], s[5], s[6], s[7]);
  }
  cluster.sync();
}

void check(const torch::Tensor& t, int64_t M, int64_t n, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.dim() == 2 && t.size(0) == M && t.size(1) == n,
              name, " must be a contiguous bf16 [", M, ", ", n, "] CUDA tensor");
}
}  // namespace

// dP [M, 512], x [M, 128], datt [M, 128], gated [M, 128], dab [M, 512], y [M, 128], dffn [M, 128], hh [M, 256]
// -> [dWqkv [384, 128], dWg [128, 128], dWo [128, 128], dWu [512, 128], dWd [128, 256]] (bf16)
std::vector<torch::Tensor> wgrad(torch::Tensor dP, torch::Tensor x, torch::Tensor datt, torch::Tensor gated, torch::Tensor dab,
                                 torch::Tensor y, torch::Tensor dffn, torch::Tensor hh) {
  const int64_t M = dP.size(0);
  check(dP, M, 512, "dP");
  check(x, M, 128, "x");
  check(datt, M, 128, "datt");
  check(gated, M, 128, "gated");
  check(dab, M, 512, "dab");
  check(y, M, 128, "y");
  check(dffn, M, 128, "dffn");
  check(hh, M, 256, "hh");
  auto o = dP.options();
  auto dwq = torch::empty({384, 128}, o), dwg = torch::empty({128, 128}, o), dwo = torch::empty({128, 128}, o), dwu = torch::empty({512, 128}, o), dwd = torch::empty({128, 256}, o);
  if (M == 0) {
    dwq.zero_(); dwg.zero_(); dwo.zero_(); dwu.zero_(); dwd.zero_();
    return {dwq, dwg, dwo, dwu, dwd};
  }
  Jobs jb;
  int j = 0;
  auto add = [&](const bf16* A, int lda, const bf16* B, int ldb, bf16* out, int ldo) {
    jb.A[j] = A; jb.B[j] = B; jb.out[j] = out; jb.lda[j] = lda; jb.ldb[j] = ldb; jb.ldo[j] = ldo;
    ++j;
  };
  auto P = [](const torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr()); };
  for (int i = 0; i < 3; ++i) add(P(dP) + i * 128, 512, P(x), 128, P(dwq) + (size_t)i * 128 * 128, 128);
  add(P(dP) + 3 * 128, 512, P(x), 128, P(dwg), 128);
  add(P(datt), 128, P(gated), 128, P(dwo), 128);
  for (int i = 0; i < 4; ++i) add(P(dab) + i * 128, 512, P(y), 128, P(dwu) + (size_t)i * 128 * 128, 128);
  for (int i = 0; i < 2; ++i) add(P(dffn), 128, P(hh) + i * 128, 256, P(dwd) + i * 128, 256);
  const int kper = static_cast<int>((((M + CL - 1) / CL) + KB - 1) / KB * KB);
  const size_t smem = std::max<size_t>(2 * NSTG * KB * LDS * sizeof(bf16), 128 * XS * sizeof(float));
  cudaFuncSetAttribute(wgrad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  wgrad_kernel<<<dim3(CL, NJOB), 256, smem, at::cuda::getCurrentCUDAStream()>>>(jb, static_cast<int>(M), kper);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dwq, dwg, dwo, dwu, dwd};
}

// fp32 operands (same shapes as ``wgrad``) -> [dWqkv [384, 128], dWg, dWo, dWu [512, 128], dWd [128, 256]] fp32 (bf16 hi / lo planes)
std::vector<torch::Tensor> wgrad_f32(torch::Tensor dP, torch::Tensor x, torch::Tensor datt, torch::Tensor gated, torch::Tensor dab,
                                     torch::Tensor y, torch::Tensor dffn, torch::Tensor hh) {
  const int64_t M = dP.size(0);
  auto chk = [&](const torch::Tensor& t, int64_t n, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous() && t.dim() == 2 && t.size(0) == M && t.size(1) == n,
                name, " must be a contiguous fp32 [", M, ", ", n, "] CUDA tensor");
  };
  chk(dP, 512, "dP"); chk(x, 128, "x"); chk(datt, 128, "datt"); chk(gated, 128, "gated");
  chk(dab, 512, "dab"); chk(y, 128, "y"); chk(dffn, 128, "dffn"); chk(hh, 256, "hh");
  auto o = dP.options();
  auto dwq = torch::empty({384, 128}, o), dwg = torch::empty({128, 128}, o), dwo = torch::empty({128, 128}, o), dwu = torch::empty({512, 128}, o),
       dwd = torch::empty({128, 256}, o);
  if (M == 0) {
    dwq.zero_(); dwg.zero_(); dwo.zero_(); dwu.zero_(); dwd.zero_();
    return {dwq, dwg, dwo, dwu, dwd};
  }
  JobsF jb;
  int j = 0;
  auto add = [&](const float* A, int lda, const float* B, int ldb, float* out, int ldo) {
    jb.A[j] = A; jb.B[j] = B; jb.out[j] = out; jb.lda[j] = lda; jb.ldb[j] = ldb; jb.ldo[j] = ldo;
    ++j;
  };
  auto P = [](const torch::Tensor& t) { return t.data_ptr<float>(); };
  for (int i = 0; i < 3; ++i) add(P(dP) + i * 128, 512, P(x), 128, P(dwq) + (size_t)i * 128 * 128, 128);
  add(P(dP) + 3 * 128, 512, P(x), 128, P(dwg), 128);
  add(P(datt), 128, P(gated), 128, P(dwo), 128);
  for (int i = 0; i < 4; ++i) add(P(dab) + i * 128, 512, P(y), 128, P(dwu) + (size_t)i * 128 * 128, 128);
  for (int i = 0; i < 2; ++i) add(P(dffn), 128, P(hh) + i * 128, 256, P(dwd) + i * 128, 256);
  const int kper = static_cast<int>((((M + CL - 1) / CL) + KF - 1) / KF * KF);
  const size_t smem = std::max<size_t>(2 * NRF * FRAW + 4 * FPL, 128 * XS * sizeof(float));
  cudaFuncSetAttribute(wgrad_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  wgrad_f32_kernel<<<dim3(CL, NJOB), 256, smem, at::cuda::getCurrentCUDAStream()>>>(jb, static_cast<int>(M), kper);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dwq, dwg, dwo, dwu, dwd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wgrad", &wgrad, "the weight-gradient GEMMs of the SWA atom block backward in one launch (bf16)");
  m.def("wgrad_f32", &wgrad_f32, "the same on fp32 operands (bf16 hi / lo planes, 3 MMAs), fp32 gradients");
}
