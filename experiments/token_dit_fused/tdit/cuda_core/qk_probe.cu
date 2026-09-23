// Validates the mechanics the fused attention core needs, in isolation:
//   * a TMA box that starts at a head's column offset inside the packed [M, 4D] q|k|v|g buffer (48 columns of interest,
//     loaded 64 wide: the extra 16 belong to the next field and are never used, since K = 48 is three k-steps of 16),
//   * 128-B swizzled wgmma descriptors for both operands,
//   * the transposed-B flag, which is what makes S = Q K^T work from a [N, K] tile,
//   * the m64nNk16 accumulator layout, written straight out to global.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }

template <int TA, int TB> TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, %35, %36; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
}

TMN_DEVI uint64_t dmn(uint32_t base, int ks, uint32_t lbo) { return smem_desc(base + ks * 2048, lbo, 1024, 1); }

// o = p v with B (v) MN-major: rows are the contraction index. ss form, so only the descriptor is under test.
template <int TB> TMN_DEVI void mma_ss48(float (&d)[24], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %26, 0; wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, p, 1, 1, 0, %27; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "l"(a), "l"(b), "r"(accumulate), "n"(TB));
}
// same, but A comes from registers packed out of an accumulator-shaped tile
TMN_DEVI void mma_rs48(float (&d)[24], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
TMN_DEVI uint32_t sw128(int row, int byte) { return row * 128 + ((((byte >> 4) ^ (row & 7))) << 4) + (byte & 15); }

// mode 0: ss with trans-b = 1 and LBO from `lbo`; mode 1: rs, A packed from the P tile in the accumulator layout
__global__ void __launch_bounds__(128, 1)
pv_kernel(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mv, float* __restrict__ OUT,
          int mode, int lbo) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* sp = sm;
  uint8_t* sv = sm + 8192;
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + 16384);
  const int tid = threadIdx.x;
  if (tid == 0) { mbar_init(bar, 1); fence_barrier_init(); }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect_tx(bar, 16384);
    tma_load_2d(sp, &mp, bar, 0, 0);
    tma_load_2d(sv, &mv, bar, 0, 0);
  }
  mbar_wait(bar, 0);
  float acc[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) acc[i] = 0.f;
  const uint32_t ap = smem_u32(sp), av = smem_u32(sv);
  const int lane = tid & 31, warp = tid >> 5;
  const int r0 = warp * 16 + (lane >> 2), c0 = 2 * (lane & 3);
  wgmma_fence();
  if (mode == 0) {
    for (int ks = 0; ks < 4; ++ks) mma_ss48<1>(acc, dsw(ap + ks * 32), dmn(av, ks, lbo), ks != 0);
  } else {
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t a[4];
      const uint8_t* prow = sm;
      for (int t = 0; t < 4; ++t) {                                 // a0 (r0, k), a1 (r0+8, k), a2 (r0, k+8), a3 (r0+8, k+8)
        const int rr = r0 + 8 * (t & 1), cc = 16 * ks + 8 * (t >> 1) + c0;
        a[t] = *reinterpret_cast<const uint32_t*>(prow + sw128(rr, cc * 2));
      }
      mma_rs48(acc, a, dmn(av, ks, lbo));
    }
  }
  wgmma_commit();
  wgmma_wait<0>();
  for (int j = 0; j < 6; ++j) {
    OUT[(size_t)r0 * 48 + 8 * j + c0] = acc[4 * j + 0];
    OUT[(size_t)r0 * 48 + 8 * j + c0 + 1] = acc[4 * j + 1];
    OUT[(size_t)(r0 + 8) * 48 + 8 * j + c0] = acc[4 * j + 2];
    OUT[(size_t)(r0 + 8) * 48 + 8 * j + c0 + 1] = acc[4 * j + 3];
  }
}

__global__ void __launch_bounds__(128, 1)
qk_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, float* __restrict__ OUT,
          int q_col, int k_col, int nk16) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* sq = sm;                                                 // [64][64] bf16
  uint8_t* sk = sm + 8192;
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + 16384);
  const int tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(bar, 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect_tx(bar, 16384);
    tma_load_2d(sq, &mq, bar, q_col, 0);
    tma_load_2d(sk, &mk, bar, k_col, 0);
  }
  mbar_wait(bar, 0);

  float acc[32];
  const uint32_t aq = smem_u32(sq), ak = smem_u32(sk);
  wgmma_fence();
  for (int ks = 0; ks < nk16; ++ks)                                 // K = 48 -> three k-steps of 16
    mma64<0, 0>(acc, dsw(aq + ks * 32), dsw(ak + ks * 32), ks != 0);
  wgmma_commit();
  wgmma_wait<0>();

  // m64nNk16 accumulator: warp w, lane l owns rows 16w + l/4 (+8), columns 8j + 2(l%4) + {0,1}
  const int lane = tid & 31, warp = tid >> 5;
  const int r0 = warp * 16 + (lane >> 2), c0 = 2 * (lane & 3);
  for (int j = 0; j < 8; ++j) {
    OUT[(size_t)r0 * 64 + 8 * j + c0] = acc[4 * j + 0];
    OUT[(size_t)r0 * 64 + 8 * j + c0 + 1] = acc[4 * j + 1];
    OUT[(size_t)(r0 + 8) * 64 + 8 * j + c0] = acc[4 * j + 2];
    OUT[(size_t)(r0 + 8) * 64 + 8 * j + c0 + 1] = acc[4 * j + 3];
  }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <array>
#include <map>
#include <mutex>

namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "no TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t bi, uint32_t bo) {
  static std::map<std::array<uint64_t, 6>, CUtensorMap> cache;
  static std::mutex lock;
  const std::array<uint64_t, 6> key{reinterpret_cast<uint64_t>(t.data_ptr()), (uint64_t)t.size(0), (uint64_t)t.size(1),
                                    (uint64_t)t.stride(0), bi, bo};
  std::lock_guard<std::mutex> g(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * 2};
  const cuuint32_t box[2] = {bi, bo}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "encode failed");
  return cache.emplace(key, map).first->second;
}
}  // namespace

torch::Tensor qk(torch::Tensor qkvg, int64_t q_col, int64_t k_col, int64_t kdim) {
  auto out = torch::zeros({64, 64}, qkvg.options().dtype(torch::kFloat32));
  const size_t smem = 16384 + 1024;
  cudaFuncSetAttribute(qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  qk_kernel<<<1, 128, smem, at::cuda::getCurrentCUDAStream()>>>(tile_map(qkvg, 64, 64), tile_map(qkvg, 64, 64),
                                                                out.data_ptr<float>(), (int)q_col, (int)k_col, (int)(kdim / 16));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
  return out;
}

torch::Tensor pv(torch::Tensor p, torch::Tensor v, int64_t mode, int64_t lbo) {
  auto out = torch::zeros({64, 48}, p.options().dtype(torch::kFloat32));
  const size_t smem = 16384 + 1024;
  cudaFuncSetAttribute(pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  pv_kernel<<<1, 128, smem, at::cuda::getCurrentCUDAStream()>>>(tile_map(p, 64, 64), tile_map(v, 64, 64),
                                                                out.data_ptr<float>(), (int)mode, (int)lbo);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk", &qk);
  m.def("pv", &pv);
}
