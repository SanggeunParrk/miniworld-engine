// ESMFold2 SWA atom block, the fp32 weight-gradient GEMMs of the backward in ONE launch on TMA + wgmma (CUDA sm_90a).
// Same decomposition as swa_wgrad_wg.cu (11 tiles of 128 x 128, K = M rows, clusters of 8 CTAs splitting K, deterministic DSMEM
// reduction) on fp32 operands: a TMA producer warp streams 32-row fp32 stages of A and B (4 x 4 KB boxes each, 128 B swizzle); the
// 256 consumer threads split every element into hi + lo bf16 planes in 128 B-swizzled MN-major shared memory and the two warpgroups run
// a x b = hi hi + hi lo + lo hi on wgmma m64n128k16 (fp32 accumulate, ~2^-16: 60x closer to fp32 than cuBLAS's TF32 GEMMs).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "pdl.cuh"

namespace {
using bf16 = __nv_bfloat16;
constexpr int NJOB = 11, CL = 8, ST = 4, KB = 32, XS = 132;
constexpr int STAGE_BYTES = 32768;                 // raw fp32: A 4 boxes x 4 KB, B 4 boxes x 4 KB
constexpr int PLANE_BYTES = 2 * 32768;             // 2 buffers x (A hi, A lo, B hi, B lo: 8 KB each)
constexpr int SMEM = ST * STAGE_BYTES + PLANE_BYTES + 2 * ST * 8 + 1024;

struct Maps { alignas(64) CUtensorMap m[8]; };
struct Jobs {
  int amap[NJOB], acol[NJOB], bmap[NJOB], bcol[NJOB], ldo[NJOB];
  float* out[NJOB];
};

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma2d(const void* map, void* dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
// MN-major, 128 B swizzle descriptor: start, leading offset 4 KB (64-wide block to block along MN), stride offset 1 KB (8 k rows)
__device__ __forceinline__ uint64_t desc(const void* p) {
  uint64_t d = static_cast<uint64_t>((sa(p) >> 4) & 0x3FFFu);
  d |= static_cast<uint64_t>(4096 >> 4) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(K) : "memory"); }

__device__ __forceinline__ void wgmma_ss(uint64_t da, uint64_t db, float* d) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void split4(float4 v, uint2& hi, uint2& lo) {
  hi.x = pack(v.x, v.y);
  hi.y = pack(v.z, v.w);
  const float h0 = __uint_as_float(hi.x << 16), h1 = __uint_as_float(hi.x & 0xffff0000u);
  const float h2 = __uint_as_float(hi.y << 16), h3 = __uint_as_float(hi.y & 0xffff0000u);
  lo.x = pack(v.x - h0, v.y - h1);
  lo.y = pack(v.z - h2, v.w - h3);
}

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(288, 1)
wgrad_wg_f32_kernel(const __grid_constant__ Maps maps, const Jobs jobs, int M, int kper) {
  extern __shared__ unsigned char smem_raw[];
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  unsigned char* smem = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  unsigned char* planes = smem + ST * STAGE_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE_BYTES + PLANE_BYTES);
  uint64_t* empty = full + ST;
  const int tid = threadIdx.x, rank = blockIdx.x % CL, job = blockIdx.y;
  const int k0 = rank * kper, k1 = min(M, k0 + kper);
  const int nk = (k1 > k0) ? (k1 - k0 + KB - 1) / KB : 0;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { bar_init(&full[s], 1); bar_init(&empty[s], 2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  pdl_gate();                                                // the operands are the previous kernels' outputs
  const int wg = tid >> 7;
  float acc[64];
  if (wg == 2) {
    if (tid == 256) {
      const void* ma = &maps.m[jobs.amap[job]];
      const void* mb = &maps.m[jobs.bmap[job]];
      for (int it = 0; it < nk; ++it) {
        const int s = it % ST;
        if (it >= ST) wait(&empty[s], ((it / ST) - 1) & 1);
        expect_tx(&full[s], STAGE_BYTES);
        unsigned char* d = smem + s * STAGE_BYTES;
        const int row = k0 + it * KB;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          tma2d(ma, d + j * 4096, &full[s], jobs.acol[job] + 32 * j, row);
          tma2d(mb, d + 16384 + j * 4096, &full[s], jobs.bcol[job] + 32 * j, row);
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    for (int it = 0; it < nk; ++it) {
      const int s = it % ST;
      unsigned char* pl = planes + (it & 1) * 32768;         // A hi, A lo, B hi, B lo: 8 KB each = 2 blocks of [32 k][64 elem]
      wait(&full[s], (it / ST) & 1);
      const unsigned char* raw = smem + s * STAGE_BYTES;
#pragma unroll
      for (int c = 0; c < 8; ++c) {                          // 2048 float4: operand (A / B), row k (32), chunk q (32)
        const int i = tid + c * 256, op = i >> 10, k = (i >> 5) & 31, q = i & 31;
        const int box = q >> 3, ch = q & 7;                  // 8 float4 (128 B) per box row
        const float4 v = *reinterpret_cast<const float4*>(raw + op * 16384 + box * 4096 + k * 128 + ((ch ^ (k & 7)) << 4));
        uint2 hi, lo;
        split4(v, hi, lo);
        const int m = 4 * q, blk = m >> 6, c16 = (m & 63) >> 3, half = (m >> 2) & 1;
        const int off = blk * 4096 + k * 128 + ((c16 ^ (k & 7)) << 4) + half * 8;
        *reinterpret_cast<uint2*>(pl + op * 16384 + off) = hi;
        *reinterpret_cast<uint2*>(pl + op * 16384 + 8192 + off) = lo;
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // generic writes -> async-proxy (wgmma) reads
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      if ((tid & 127) == 0) arrive(&empty[s]);               // the raw stage is consumed
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < KB / 16; ++ks) {
        const uint64_t ah = desc(pl + wg * 4096 + ks * 2048), al = desc(pl + 8192 + wg * 4096 + ks * 2048);
        const uint64_t bh = desc(pl + 16384 + ks * 2048), bl = desc(pl + 24576 + ks * 2048);
        wgmma_ss(al, bh, acc);
        wgmma_ss(ah, bl, acc);
        wgmma_ss(ah, bh, acc);
      }
      wg_commit();
      wg_wait<1>();                                          // the previous stage's planes are free again
    }
    wg_wait<0>();
  }
  __syncthreads();
  float* X = reinterpret_cast<float*>(smem);                 // [128][XS] over the raw ring (all TMA consumed)
  if (wg < 2) {
    const int t = tid & 127, w = t >> 5, lane = t & 31, g = lane >> 2, q = lane & 3;
    const int row = wg * 64 + w * 16 + g;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = 8 * j + 2 * q;
      *reinterpret_cast<float2*>(X + row * XS + col) = make_float2(acc[4 * j], acc[4 * j + 1]);
      *reinterpret_cast<float2*>(X + (row + 8) * XS + col) = make_float2(acc[4 * j + 2], acc[4 * j + 3]);
    }
  }
  cluster.sync();
  if (tid < 256) {
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

PFN_cuTensorMapEncodeTiled_v12000 tma_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    // CUDA 13 has no unversioned PFN name and deprecates cudaGetDriverEntryPoint: ask for the 12.0 ABI by version
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
#endif
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
CUtensorMap map2d(const torch::Tensor& t, int64_t M, int64_t cols) {   // [M][cols] fp32 -> box {32 cols, 32 rows}, 128 B swizzle
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous() && t.size(0) == M && t.size(1) == cols,
              "operand must be a contiguous fp32 [", M, ", ", cols, "] CUDA tensor");
  alignas(64) CUtensorMap m{};
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(M)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * 4};
  cuuint32_t box[2] = {32, 32}, el[2] = {1, 1};
  const CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, t.data_ptr(), dims, strides, box, el, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
}  // namespace

// same contract as swa_wgrad.cu ``wgrad_f32``
std::vector<torch::Tensor> wgrad_f32(torch::Tensor dP, torch::Tensor x, torch::Tensor datt, torch::Tensor gated, torch::Tensor dab,
                                 torch::Tensor y, torch::Tensor dffn, torch::Tensor hh) {
  const int64_t M = dP.size(0);
  auto o = dP.options();
  auto dwq = torch::empty({384, 128}, o), dwg = torch::empty({128, 128}, o), dwo = torch::empty({128, 128}, o), dwu = torch::empty({512, 128}, o),
       dwd = torch::empty({128, 256}, o);
  if (M == 0) {
    dwq.zero_(); dwg.zero_(); dwo.zero_(); dwu.zero_(); dwd.zero_();
    return {dwq, dwg, dwo, dwu, dwd};
  }
  Maps maps;
  maps.m[0] = map2d(dP, M, 512); maps.m[1] = map2d(x, M, 128); maps.m[2] = map2d(datt, M, 128); maps.m[3] = map2d(gated, M, 128);
  maps.m[4] = map2d(dab, M, 512); maps.m[5] = map2d(y, M, 128); maps.m[6] = map2d(dffn, M, 128); maps.m[7] = map2d(hh, M, 256);
  Jobs jb{};
  int j = 0;
  auto add = [&](int am, int ac, int bm, int bc, torch::Tensor& out, int64_t row0, int ldo, int col0) {
    jb.amap[j] = am; jb.acol[j] = ac; jb.bmap[j] = bm; jb.bcol[j] = bc; jb.ldo[j] = ldo;
    jb.out[j] = out.data_ptr<float>() + row0 * ldo + col0;
    ++j;
  };
  for (int i = 0; i < 3; ++i) add(0, i * 128, 1, 0, dwq, i * 128, 128, 0);
  add(0, 384, 1, 0, dwg, 0, 128, 0);
  add(2, 0, 3, 0, dwo, 0, 128, 0);
  for (int i = 0; i < 4; ++i) add(4, i * 128, 5, 0, dwu, i * 128, 128, 0);
  for (int i = 0; i < 2; ++i) add(6, 0, 7, i * 128, dwd, 0, 256, i * 128);
  const int kper = static_cast<int>((((M + CL - 1) / CL) + KB - 1) / KB * KB);
  cudaFuncSetAttribute(wgrad_wg_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  pdl_launch(wgrad_wg_f32_kernel, dim3(CL, NJOB), 288, SMEM, at::cuda::getCurrentCUDAStream(), true, maps, jb, static_cast<int>(M), kper);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dwq, dwg, dwo, dwu, dwd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wgrad_f32", &wgrad_f32, "fp32 weight gradients of the SWA atom block backward, one launch, TMA + wgmma on bf16 planes");
}
