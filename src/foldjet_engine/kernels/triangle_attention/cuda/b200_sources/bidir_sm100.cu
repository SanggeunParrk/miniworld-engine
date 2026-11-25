// Bias-only TriangleAttention (BidirectionalTriangleAttention and TriangleAttention(use_self_attention=False)) on B200 (sm_100a): the
// ending node's per-head contractions over the channel planes as tcgen05 GEMMs, one launch for every head and channel (the H100 /
// A100 path runs them as one cuBLAS batched GEMM per head, plus a torch sum per head for dP), and the softmax backward's row dot.
//
//   bo_pv  (AMN = 0, forward):   O[c][i][j]  = sum_k P[h][i][k] V[c][k][j]        A = P[h] K-major,  B = V[c] MN-major
//          (AMN = 1, backward):  dV[c][k][j] = sum_i P[h][i][k] dO[c][i][j]       A = P[h]^T MN-major, B = dO[c] MN-major
//          h = c / dh; a CTA = one channel x one 128 x N output tile; bf16 out (fp32 accumulation in TMEM, one rounding: what the
//          batched GEMM wrote).
//   bo_dp  dP[h][i][k] = sum_{c in h} sum_j dO[c][i][j] V[c][k][j]                 both K-major; fp32 out
//          the K loop runs over (channel, 64-column block); a cluster of `split` CTAs splits the head's channels and adds the partial
//          tiles through distributed shared memory in rank order (deterministic: the same bits every run).
//   bo_rowdot  s[r] = sum_k p[r][k] dp[r][k]   (the softmax backward's row term; one warp a row)
//
// Warps: 0 TMA, 1 tcgen05 issue, 2-5 drain (TMEM lane = output row). Four-stage ring of 64-wide K blocks.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

namespace {
namespace bo {
constexpr int BM = 128, BK = 64, NS = 4, THREADS = 192;
constexpr int ABYTES = BM * BK * 2;                       // 16 KiB
template <int N> constexpr int bbytes() { return N * BK * 2; }
template <int N> constexpr int smem() { return 1024 + NS * (ABYTES + bbytes<N>()) + 256; }
}  // namespace bo

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

// ------------------------------------------------------------------------------------------------------------------- bo_pv
template <int N, int AMN>
__global__ void __launch_bounds__(bo::THREADS, 1) bo_pv_sm100(
    int L, int dh, int mtiles,
    const __grid_constant__ CUtensorMap pmap,             // P [H][L][L] bf16, dims (L, L, H); box (64, 128, 1) or (AMN) (64, 64, 1)
    const __grid_constant__ CUtensorMap vmap,             // planes [nch][L][L] bf16, dims (L, L, nch); box (64, 64, 1)
    __nv_bfloat16* __restrict__ OUT) {                    // [nch][L][L] bf16
  using namespace bo;
  constexpr int BB = bbytes<N>();
  constexpr uint32_t ID = idesc_bf16(BM, N, AMN, 1);
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sA = smb;
  unsigned char* sB = smb + NS * ABYTES;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sB + NS * BB);
  uint64_t* full = bars;
  uint64_t* empty = full + NS;
  uint64_t* done = empty + NS;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ch = blockIdx.y, h = ch / dh;
  const int m0 = (blockIdx.x % mtiles) * BM, n0 = (blockIdx.x / mtiles) * N;
  const int nk = (L + BK - 1) / BK;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    bar_init(done, 1);
    bar_init_fence();
    prefetch_map(&pmap); prefetch_map(&vmap);
  }
  if (warp == 1) tmem_alloc(tslot, N);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  if (warp == 0) {
    if (lane == 0) {
      for (int kb = 0; kb < nk; ++kb) {
        const int s = kb % NS;
        if (kb >= NS) wait(empty + s, ((kb / NS) - 1) & 1);
        expect_tx(full + s, ABYTES + BB);
        unsigned char* a = sA + s * ABYTES;
        if (AMN) {
          load_3d(&pmap, a, full + s, m0, kb * BK, h);
          load_3d(&pmap, a + 64 * 64 * 2, full + s, m0 + 64, kb * BK, h);
        } else {
          load_3d(&pmap, a, full + s, kb * BK, m0, h);
        }
        unsigned char* b = sB + s * BB;
#pragma unroll
        for (int nb = 0; nb < N / 64; ++nb) load_3d(&vmap, b + nb * 64 * 64 * 2, full + s, n0 + nb * 64, kb * BK, ch);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      for (int kb = 0; kb < nk; ++kb) {
        const int s = kb % NS;
        wait(full + s, (kb / NS) & 1);
        tc_fence_after();
        const unsigned char* a = sA + s * ABYTES;
        const unsigned char* b = sB + s * BB;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
          const uint64_t ad = AMN ? desc_mn128(a + ks * 16 * 128, 64 * 64 * 2) : desc_k128(a + ks * 32);
          mma_ss(tmem, ad, desc_mn128(b + ks * 16 * 128, 64 * 64 * 2), ID, (kb | ks) ? 1u : 0u);
        }
        mma_commit(empty + s);
      }
      mma_commit(done);
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane, m = m0 + r;
    wait(done, 0);
    tc_fence_after();
    __nv_bfloat16* orow = OUT + ((long)ch * L + m) * L;
#pragma unroll 1
    for (int c0 = 0; c0 < N; c0 += 32) {
      float v[32];
      tmem_ld32(tmem_at(tmem + c0, qq * 32, 0), v);
      tmem_wait_ld();
      if (m < L) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int n = n0 + c0 + g * 8;
          if (n < L)
            *reinterpret_cast<uint4*>(orow + n) = make_uint4(pack2(v[g * 8], v[g * 8 + 1]), pack2(v[g * 8 + 2], v[g * 8 + 3]),
                                                             pack2(v[g * 8 + 4], v[g * 8 + 5]), pack2(v[g * 8 + 6], v[g * 8 + 7]));
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, N);
}

// ------------------------------------------------------------------------------------------------------------------- bo_dp
template <int N>
__global__ void __launch_bounds__(bo::THREADS, 1) bo_dp_sm100(
    int L, int dh, int split, int mtiles,
    const __grid_constant__ CUtensorMap amap,             // dO planes [nch][L][L] bf16, dims (L, L, nch); box (64, 128, 1)
    const __grid_constant__ CUtensorMap bmap,             // V planes, the same dims; box (64, N, 1)
    float* __restrict__ DP) {                             // [H][L][L] fp32
  using namespace bo;
  constexpr int BB = bbytes<N>();
  constexpr uint32_t ID = idesc_bf16(BM, N, 0, 0);
  constexpr int RS = N + 4;                               // the partial tile's row stride in shared (floats)
  static_assert(BM * RS * 4 <= NS * (ABYTES + BB), "the partial tile reuses the ring");
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sA = smb;
  unsigned char* sB = smb + NS * ABYTES;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sB + NS * BB);
  uint64_t* full = bars;
  uint64_t* empty = full + NS;
  uint64_t* done = empty + NS;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.z, rank = (int)blockIdx.x;      // cluster (split, 1, 1): blockIdx.x is the rank
  const int m0 = (blockIdx.y % mtiles) * BM, n0 = (blockIdx.y / mtiles) * N;
  const int dps = dh / split, c0 = h * dh + rank * dps;
  const int nj = (L + BK - 1) / BK, nk = dps * nj;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    bar_init(done, 1);
    bar_init_fence();
    prefetch_map(&amap); prefetch_map(&bmap);
  }
  if (warp == 1) tmem_alloc(tslot, N);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  if (warp == 0) {
    if (lane == 0) {
      for (int kb = 0; kb < nk; ++kb) {
        const int s = kb % NS, c = c0 + kb / nj, j0 = (kb % nj) * BK;
        if (kb >= NS) wait(empty + s, ((kb / NS) - 1) & 1);
        expect_tx(full + s, ABYTES + BB);
        load_3d(&amap, sA + s * ABYTES, full + s, j0, m0, c);
        load_3d(&bmap, sB + s * BB, full + s, j0, n0, c);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      for (int kb = 0; kb < nk; ++kb) {
        const int s = kb % NS;
        wait(full + s, (kb / NS) & 1);
        tc_fence_after();
        const unsigned char* a = sA + s * ABYTES;
        const unsigned char* b = sB + s * BB;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) mma_ss(tmem, desc_k128(a + ks * 32), desc_k128(b + ks * 32), ID, (kb | ks) ? 1u : 0u);
        mma_commit(empty + s);
      }
      mma_commit(done);
    }
  }
  float* red = reinterpret_cast<float*>(smb);              // [BM][RS] partial tile (split > 1), over the drained ring
  if (warp >= 2) {
    const int qq = warp & 3, r = qq * 32 + lane, m = m0 + r;
    wait(done, 0);
    tc_fence_after();
#pragma unroll 1
    for (int cc = 0; cc < N; cc += 32) {
      float v[32];
      tmem_ld32(tmem_at(tmem + cc, qq * 32, 0), v);
      tmem_wait_ld();
      if (split > 1) {
        float4* dst = reinterpret_cast<float4*>(red + r * RS + cc);
#pragma unroll
        for (int g = 0; g < 8; ++g) dst[g] = make_float4(v[4 * g], v[4 * g + 1], v[4 * g + 2], v[4 * g + 3]);
      } else if (m < L) {
        float* orow = DP + ((long)h * L + m) * L;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const int n = n0 + cc + 4 * g;
          if (n < L) *reinterpret_cast<float4*>(orow + n) = make_float4(v[4 * g], v[4 * g + 1], v[4 * g + 2], v[4 * g + 3]);
        }
      }
    }
  }
  __syncwarp();                                            // (warps 0 / 1: lane 0 ran the loops) converged for the aligned barrier
  if (split > 1) {
    cl_sync();                                           // every rank's partial tile is in its shared memory
    const int rows = BM / split, q4 = N / 4;
    const uint32_t base = sa(red);
    for (int e = tid; e < rows * q4; e += bo::THREADS) {
      const int rr = rank * rows + e / q4, cq = (e % q4) * 4, m = m0 + rr, n = n0 + cq;
      const uint32_t off = base + (uint32_t)(rr * RS + cq) * 4u;
      float4 t[8];                                         // every rank's value in flight at once, then added in rank order
#pragma unroll
      for (int q = 0; q < 8; ++q) if (q < split) t[q] = ld_cl4(off, q);
      float4 s4 = t[0];
#pragma unroll
      for (int q = 1; q < 8; ++q)
        if (q < split) { s4.x += t[q].x; s4.y += t[q].y; s4.z += t[q].z; s4.w += t[q].w; }
      if (m < L && n < L) *reinterpret_cast<float4*>(DP + ((long)h * L + m) * L + n) = s4;
    }
    cl_sync();                                             // the peers are done reading this CTA's tile
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, N);
}

// ------------------------------------------------------------------------------------------------------------------- bo_rowdot
__global__ void __launch_bounds__(256) bo_rowdot_k(const __nv_bfloat16* __restrict__ P, const float* __restrict__ DP, float* __restrict__ S,
                                                   long rows, int L) {
  const long r = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (r >= rows) return;
  const __nv_bfloat16* p = P + r * L;
  const float* d = DP + r * L;
  float s = 0.f;
  for (int k = lane * 8; k < L; k += 256) {
    const uint4 pv = __ldg(reinterpret_cast<const uint4*>(p + k));
    const float4 d0 = __ldg(reinterpret_cast<const float4*>(d + k)), d1 = __ldg(reinterpret_cast<const float4*>(d + k + 4));
    const float2 a = bf2f(pv.x), b = bf2f(pv.y), c = bf2f(pv.z), e = bf2f(pv.w);
    s += a.x * d0.x + a.y * d0.y + b.x * d0.z + b.y * d0.w + c.x * d1.x + c.y * d1.y + e.x * d1.z + e.y * d1.w;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) S[r] = s;
}

// ------------------------------------------------------------------------------------------------------------------- host
CUtensorMap planes_map(const torch::Tensor& t, long L, long n, uint32_t b0, uint32_t b1, const char* what) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.numel() == n * L * L, what, ": contiguous bf16 [n, L, L]");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0 && L % 8 == 0, what, ": 16-byte aligned, L % 8 == 0");
  return make_map<3>(t.data_ptr(), {(uint64_t)L, (uint64_t)L, (uint64_t)n}, {(uint64_t)L, (uint64_t)(L * L)}, {b0, b1, 1u},
                     CU_TENSOR_MAP_SWIZZLE_128B, what);
}

template <typename K> void smem_attr(K k, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// p [H, L, L] bf16 (the ending softmax), planes [nch, L L] bf16 (nch = H dh), out [nch, L L] bf16:
// trans = false: out[c] = P[h] planes[c]  (the forward's O_e); trans = true: out[c] = P[h]^T planes[c]  (the backward's dV_e)
void bo_pv(torch::Tensor p, torch::Tensor planes, int64_t dh, bool trans, torch::Tensor out) {
  const long L = p.size(-1), H = p.numel() / (L * L), nch = planes.numel() / (L * L);
  TORCH_CHECK(nch == H * dh && out.numel() == planes.numel() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16, "bo_pv: shapes");
  constexpr int N = 128;
  const CUtensorMap pm = planes_map(p, L, H, 64, trans ? 64 : 128, "P");
  const CUtensorMap vm = planes_map(planes, L, nch, 64, 64, "planes");
  const int mtiles = (int)((L + 127) / 128), ntiles = (int)((L + N - 1) / N);
  auto st = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const dim3 grid(mtiles * ntiles, (unsigned)nch);
  if (trans) {
    static bool init = false;
    if (!init) { smem_attr(bo_pv_sm100<N, 1>, bo::smem<N>()); init = true; }
    bo_pv_sm100<N, 1><<<grid, bo::THREADS, bo::smem<N>(), st>>>((int)L, (int)dh, mtiles, pm, vm, o);
  } else {
    static bool init = false;
    if (!init) { smem_attr(bo_pv_sm100<N, 0>, bo::smem<N>()); init = true; }
    bo_pv_sm100<N, 0><<<grid, bo::THREADS, bo::smem<N>(), st>>>((int)L, (int)dh, mtiles, pm, vm, o);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dO, V planes [H dh, L L] bf16 -> dp [H, L, L] fp32 (written): dp[h] = sum_{c in h} dO[c] V[c]^T, deterministic
void bo_dp(torch::Tensor dO, torch::Tensor v, int64_t dh, torch::Tensor dp, int64_t split_) {
  const long L = dp.size(-1), H = dp.numel() / (L * L), nch = v.numel() / (L * L);
  TORCH_CHECK(nch == H * dh && dO.numel() == v.numel() && dp.is_contiguous() && dp.scalar_type() == torch::kFloat32, "bo_dp: shapes");
  constexpr int N = 128;
  const CUtensorMap am = planes_map(dO, L, nch, 64, 128, "dO");
  const CUtensorMap bm = planes_map(v, L, nch, 64, N, "V");
  const int mtiles = (int)((L + 127) / 128), ntiles = (int)((L + N - 1) / N), tiles = (int)H * mtiles * ntiles;
  int split = (int)split_;
  if (split <= 0) {                                       // enough CTAs for the 148 SMs, a power of two dividing dh, <= 8 (portable)
    split = 1;
    while (split < 8 && dh % (2 * split) == 0 && tiles * split * 2 <= 160) split *= 2;
  }
  TORCH_CHECK(split >= 1 && split <= 8 && dh % split == 0 && (split & (split - 1)) == 0, "bo_dp: split");
  static bool init = false;
  if (!init) { smem_attr(bo_dp_sm100<N>, bo::smem<N>()); init = true; }
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = split; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(split, mtiles * ntiles, (unsigned)H);
  cfg.blockDim = dim3(bo::THREADS);
  cfg.dynamicSmemBytes = bo::smem<N>();
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = at; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, bo_dp_sm100<N>, (int)L, (int)dh, split, mtiles, am, bm, dp.data_ptr<float>()));
}

// p [P, L, L] bf16, dp [P, L, L] fp32 -> s [P, L, 1] fp32 = (p dp).sum(-1)
torch::Tensor bo_rowdot(torch::Tensor p, torch::Tensor dp) {
  const long L = p.size(-1), rows = p.numel() / L;
  TORCH_CHECK(p.is_contiguous() && dp.is_contiguous() && dp.numel() == p.numel() && L % 8 == 0 && dp.scalar_type() == torch::kFloat32 &&
              p.scalar_type() == torch::kBFloat16, "bo_rowdot: contiguous bf16 p, fp32 dp, L % 8 == 0");
  auto s = torch::empty({rows / L, L, 1}, dp.options());
  bo_rowdot_k<<<(unsigned)((rows + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(p.data_ptr<at::BFloat16>()), dp.data_ptr<float>(), s.data_ptr<float>(), rows, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return s;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bo_pv", &bo_pv, "sm100 bias-only ending contraction over every head's channels: out[c] = P[h] V[c] (or P[h]^T)", py::arg("p"),
        py::arg("planes"), py::arg("dh"), py::arg("trans"), py::arg("out"));
  m.def("bo_dp", &bo_dp, "sm100 bias-only ending dP[h] = sum_c dO[c] V[c]^T (fp32, deterministic cluster reduction)", py::arg("dO"), py::arg("v"),
        py::arg("dh"), py::arg("dp"), py::arg("split") = 0);
  m.def("bo_rowdot", &bo_rowdot, "sm100 (p dp).sum(-1) of the softmax backward", py::arg("p"), py::arg("dp"));
  m.def("bo_colsum", [](torch::Tensor part) { return sm100::colsum(part); },
        "sum of fp32 partials [n, ...] over dim 0 in a fixed order (the LayerNorm backward's d gamma / d beta partials), one launch", py::arg("part"));
}
