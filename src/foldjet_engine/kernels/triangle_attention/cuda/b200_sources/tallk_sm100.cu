// Tall-K weight-gradient GEMMs of the wide TriangleAttention backward on B200 (sm_100a), both in ONE launch:
//   out_p[m][n] = sum_k A_p[k][m] B_p[k][n]      (A_p [K_p, M_p], B_p [K_p, N_p] bf16 row-major; out fp32 [M_p, N_p])
// p = 0: G = Dall^T [xhat | 1] (K = the kernels' B Lp^2 rows), p = 1: dWo = dy_s^T u (K = the caller's B L^2 rows).
// cuBLAS ran each as a split-K GEMM plus a reduce kernel (four launches, latency bound at small L: ~20 us at L128 for ~2 GFLOP).
// Here a cluster of `split` CTAs shares one 128 x 128 output tile, each CTA a contiguous K range (tcgen05, both operands MN-major
// straight from the row-major tiles), and the partial tiles are added through distributed shared memory in rank order: the same
// bits every run, no partial buffers, one launch.  Warps: 0 TMA, 1 tcgen05, 2-5 drain (TMEM lane = output row).
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

namespace {
namespace tk {
constexpr int BM = 128, BN = 128, BK = 64, NS = 4, THREADS = 192;
constexpr int ABYTES = BM * BK * 2, BBYTES = BN * BK * 2;  // 16 KiB each
constexpr int RS = BN + 4;                                 // partial tile row stride (floats)
constexpr int SMEM = 1024 + NS * (ABYTES + BBYTES) + 256;
static_assert(BM * RS * 4 <= NS * (ABYTES + BBYTES), "the partial tile reuses the ring");
constexpr uint32_t ID = idesc_bf16(BM, BN, 1, 1);
}  // namespace tk

struct TkProb {
  CUtensorMap a, b;                                        // A [K][M], B [K][N] bf16, box (64, 64), 128B swizzle
  float* out;                                              // [M][N] fp32
  int M, N, K, mtiles, tiles;                              // tiles = mtiles x ntiles
};
struct TkArgs { TkProb p[2]; };

__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ float4 ld_cl4(uint32_t local, uint32_t rank) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(local), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}

__global__ void __launch_bounds__(tk::THREADS, 1) tallk_sm100(const __grid_constant__ TkArgs args, int split) {
  using namespace tk;
  const int pi = (int)blockIdx.y < args.p[0].tiles ? 0 : 1;
  const TkProb& P = args.p[pi];
  const int tile = (int)blockIdx.y - (pi ? args.p[0].tiles : 0);
  const int m0 = (tile % P.mtiles) * BM, n0 = (tile / P.mtiles) * BN;
  const int rank = (int)blockIdx.x;                        // cluster (split, 1, 1)
  const int nkb = (P.K + BK - 1) / BK, per = (nkb + split - 1) / split;
  const int kb0 = rank * per, kb1 = min(nkb, kb0 + per), nk = max(0, kb1 - kb0);
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sA = smb;
  unsigned char* sB = smb + NS * ABYTES;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sB + NS * BBYTES);
  uint64_t* full = bars;
  uint64_t* empty = full + NS;
  uint64_t* done = empty + NS;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    bar_init(done, 1);
    bar_init_fence();
    prefetch_map(&P.a); prefetch_map(&P.b);
  }
  if (warp == 1) tmem_alloc(tslot, BN);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % NS, k0 = (kb0 + i) * BK;
        if (i >= NS) wait(empty + s, ((i / NS) - 1) & 1);
        // a 64-wide half beyond M / N is not loaded: the MMA reads stale shared memory there, which only reaches output rows /
        // columns that are never stored
        const bool a2 = m0 + 64 < P.M, b2 = n0 + 64 < P.N;
        expect_tx(full + s, (ABYTES + BBYTES) / 2 + (a2 ? ABYTES / 2 : 0) + (b2 ? BBYTES / 2 : 0));
        unsigned char* a = sA + s * ABYTES;
        unsigned char* b = sB + s * BBYTES;
        load_2d(&P.a, a, full + s, m0, k0);
        if (a2) load_2d(&P.a, a + 64 * 64 * 2, full + s, m0 + 64, k0);
        load_2d(&P.b, b, full + s, n0, k0);
        if (b2) load_2d(&P.b, b + 64 * 64 * 2, full + s, n0 + 64, k0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % NS;
        wait(full + s, (i / NS) & 1);
        tc_fence_after();
        const unsigned char* a = sA + s * ABYTES;
        const unsigned char* b = sB + s * BBYTES;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks)
          mma_ss(tmem, desc_mn128(a + ks * 16 * 128, 64 * 64 * 2), desc_mn128(b + ks * 16 * 128, 64 * 64 * 2), ID, (i | ks) ? 1u : 0u);
        mma_commit(empty + s);
      }
      mma_commit(done);
    }
  }
  float* red = reinterpret_cast<float*>(smb);              // [BM][RS] partial tile, over the drained ring
  if (warp >= 2) {
    const int qq = warp & 3, r = qq * 32 + lane;
    if (nk > 0) { wait(done, 0); tc_fence_after(); }
#pragma unroll 1
    for (int cc = 0; cc < BN; cc += 32) {
      float v[32];
      if (nk > 0) {
        tmem_ld32(tmem_at(tmem + cc, qq * 32, 0), v);
        tmem_wait_ld();
      } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = 0.f;            // an empty K range (more ranks than K blocks): a zero partial
      }
      float4* dst = reinterpret_cast<float4*>(red + r * RS + cc);
#pragma unroll
      for (int g = 0; g < 8; ++g) dst[g] = make_float4(v[4 * g], v[4 * g + 1], v[4 * g + 2], v[4 * g + 3]);
    }
  }
  __syncwarp();
  cl_sync();                                               // every rank's partial tile is in its shared memory
  {
    const int rows = BM / split, q4 = BN / 4;
    const uint32_t base = sa(red);
    for (int e = tid; e < rows * q4; e += THREADS) {
      const int rr = rank * rows + e / q4, cq = (e % q4) * 4, m = m0 + rr, n = n0 + cq;
      const uint32_t off = base + (uint32_t)(rr * RS + cq) * 4u;
      float4 t[16];                                        // every rank's value in flight at once, then added in rank order
#pragma unroll
      for (int q = 0; q < 16; ++q) if (q < split) t[q] = ld_cl4(off, q);
      float4 s4 = t[0];
#pragma unroll
      for (int q = 1; q < 16; ++q)
        if (q < split) { s4.x += t[q].x; s4.y += t[q].y; s4.z += t[q].z; s4.w += t[q].w; }
      if (m < P.M) {
        float* o = P.out + (long)m * P.N + n;
        if (n + 3 < P.N && (P.N & 3) == 0) *reinterpret_cast<float4*>(o) = s4;
        else {
          const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
          for (int j = 0; j < 4; ++j) if (n + j < P.N) o[j] = sv[j];
        }
      }
    }
  }
  cl_sync();                                               // the peers are done reading this CTA's tile
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, BN);
}

TkProb prob(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& out) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == b.size(0) && a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
              "tallk: A [K, M], B [K, N] bf16");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 && a.stride(0) % 8 == 0 && b.stride(0) % 8 == 0 && a.size(1) % 8 == 0 && b.size(1) % 8 == 0,
              "tallk: rows contiguous, 16-byte aligned row strides and widths");
  TkProb p{};
  const uint64_t K = (uint64_t)a.size(0), M = (uint64_t)a.size(1), N = (uint64_t)b.size(1);
  p.a = make_map<2>(a.data_ptr(), {M, K}, {(uint64_t)a.stride(0)}, {64u, 64u}, CU_TENSOR_MAP_SWIZZLE_128B, "tallk A");
  p.b = make_map<2>(b.data_ptr(), {N, K}, {(uint64_t)b.stride(0)}, {64u, 64u}, CU_TENSOR_MAP_SWIZZLE_128B, "tallk B");
  out = torch::empty({(long)M, (long)N}, a.options().dtype(torch::kFloat32));
  p.out = out.data_ptr<float>();
  p.M = (int)M; p.N = (int)N; p.K = (int)K;
  p.mtiles = (int)((M + tk::BM - 1) / tk::BM);
  p.tiles = p.mtiles * (int)((N + tk::BN - 1) / tk::BN);
  return p;
}

// (a0^T b0, a1^T b1) fp32, one launch; a1 / b1 may be empty (one product)
std::vector<torch::Tensor> tallk(torch::Tensor a0, torch::Tensor b0, torch::Tensor a1, torch::Tensor b1, int64_t split_) {
  torch::Tensor o0, o1;
  TkArgs args{};
  args.p[0] = prob(a0, b0, o0);
  const bool two = a1.defined() && a1.numel() > 0;
  if (two) args.p[1] = prob(a1, b1, o1); else args.p[1].tiles = 0;
  const int tiles = args.p[0].tiles + args.p[1].tiles;
  int split = (int)split_;
  if (split <= 0) {                                        // fill the SMs: 16 ranks (non-portable cluster) while tiles are few
    split = 16;
    while (split > 2 && tiles * split > 160) split >>= 1;
  }
  TORCH_CHECK(split >= 1 && split <= 16 && (tk::BM % split) == 0, "tallk: split");
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tallk_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, tk::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tallk_sm100, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = split; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(split, tiles, 1);
  cfg.blockDim = dim3(tk::THREADS);
  cfg.dynamicSmemBytes = tk::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = at; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tallk_sm100, args, split));
  return {o0, two ? o1 : torch::Tensor()};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tallk", &tallk, "sm100 (a0^T b0, a1^T b1) fp32 in one launch: tall-K GEMMs, cluster split-K reduced in rank order (deterministic)",
        py::arg("a0"), py::arg("b0"), py::arg("a1") = torch::Tensor(), py::arg("b1") = torch::Tensor(), py::arg("split") = 0);
}
