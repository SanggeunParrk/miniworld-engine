// ESMFold2 SWA atom block, the hoisted adaLN modulation forward on TMA + wgmma (CUDA sm_90a, bf16):
//   mod[R, 768] fp32 = silu(c)[R, 128] . Wmod[768, 128]^T      (+ silu(c) bf16 kept for the backward when ``save``)
// CTA = (64-row tile, pair of 128-channel chunks): one TMA wave brings the c tile (2 boxes of [64 rows][64 k], 128 B swizzle) and the
// two Wmod chunks ([128 n][128 k] row-major = K-major B); the 128 threads (one warpgroup) apply silu in place on the swizzled tile
// (bf16 rounding as F.silu on bf16), then 8 x wgmma m64n128k16 per chunk, fp32 accumulators stored straight to global.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include "swa_ptx.cuh"
#include "pdl.cuh"

namespace {
using bf16 = __nv_bfloat16;
constexpr int MW = 768, C = 128;
constexpr int SMEM = 16384 + 2 * 32768 + 64 + 1024;

__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma2d(const void* map, void* dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
// K-major, 128 B swizzle: 8-row groups 1 KB apart
__device__ __forceinline__ uint64_t desc(const void* p) {
  uint64_t d = static_cast<uint64_t>((sa(p) >> 4) & 0x3FFFu);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}
__device__ __forceinline__ void wgmma_ss(uint64_t da, uint64_t db, float* d, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

__global__ void __launch_bounds__(128, 2) mod_fwd_wg_kernel(const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mw,
                                                            float* __restrict__ out, bf16* __restrict__ a_out, int R) {
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + 16384 + 65536);
  const int tid = threadIdx.x, r0 = blockIdx.x * 64, cp = blockIdx.y;       // chunks 2 cp, 2 cp + 1
  if (tid == 0) {
    bar_init(&bar[0], 1);
    bar_init(&bar[1], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    expect_tx(&bar[0], 16384 + 32768);
    tma2d(&mc, smem, &bar[0], 0, r0);
    tma2d(&mc, smem + 8192, &bar[0], 64, r0);
    tma2d(&mw, smem + 16384, &bar[0], 0, (2 * cp) * 128);                   // W chunk = 128 rows (n) x 128 k: 2 boxes of [128 n][64 k]
    tma2d(&mw, smem + 16384 + 16384, &bar[0], 64, (2 * cp) * 128);
    expect_tx(&bar[1], 32768);
    tma2d(&mw, smem + 16384 + 32768, &bar[1], 0, (2 * cp + 1) * 128);
    tma2d(&mw, smem + 16384 + 32768 + 16384, &bar[1], 64, (2 * cp + 1) * 128);
  }
  wait(&bar[0], 0);
  // silu in place on the swizzled c tile: 64 rows x 2 blocks x 8 chunks of 16 B = 1024 chunks / 128 threads
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = tid + i * 128, blk = idx >> 9, row = (idx >> 3) & 63, ph = idx & 7;
    uint4* p = reinterpret_cast<uint4*>(smem + blk * 8192 + row * 128 + (ph << 4));
    uint4 v = *p;
    uint32_t* u = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float lo = silu_f(__uint_as_float(u[e] << 16)), hi = silu_f(__uint_as_float(u[e] & 0xffff0000u));
      __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
      u[e] = *reinterpret_cast<uint32_t*>(&b);
    }
    *p = v;
    if (a_out != nullptr && cp == 0 && r0 + row < R)
      *reinterpret_cast<uint4*>(a_out + (size_t)(r0 + row) * C + blk * 64 + ((ph ^ (row & 7)) << 3)) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int row = r0 + w * 16 + g;
#pragma unroll
  for (int ch = 0; ch < 2; ++ch) {
    if (ch == 1) wait(&bar[1], 0);
    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const unsigned char* wb = smem + 16384 + ch * 32768;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int blk = ks >> 2, off = (ks & 3) * 32;
      wgmma_ss(desc(smem + blk * 8192 + off), desc(wb + blk * 16384 + off), acc, ks > 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const int cbase = (2 * cp + ch) * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = cbase + 8 * j + 2 * q;
      if (row < R) __stcs(reinterpret_cast<float2*>(out + (size_t)row * MW + col), make_float2(acc[4 * j], acc[4 * j + 1]));
      if (row + 8 < R) __stcs(reinterpret_cast<float2*>(out + (size_t)(row + 8) * MW + col), make_float2(acc[4 * j + 2], acc[4 * j + 3]));
    }
  }
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
CUtensorMap map2d(const torch::Tensor& t, int64_t rows, int64_t cols, uint32_t brows) {   // [rows][cols] bf16, box {64, brows}
  alignas(64) CUtensorMap m{};
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * 2};
  cuuint32_t box[2] = {64, brows}, el[2] = {1, 1};
  const CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, el, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
}  // namespace

std::vector<torch::Tensor> mod_fwd(torch::Tensor c, torch::Tensor w, bool save) {
  TORCH_CHECK(c.is_cuda() && c.scalar_type() == torch::kBFloat16 && c.is_contiguous() && c.dim() == 2 && c.size(1) == C, "c [R, 128] bf16");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous() && w.size(0) == MW && w.size(1) == C, "wmod [768, 128] bf16");
  const int R = static_cast<int>(c.size(0));
  auto out = torch::empty({R, MW}, c.options().dtype(torch::kFloat32));
  auto aout = save ? torch::empty({R, C}, c.options()) : torch::empty({0}, c.options());
  if (R == 0) return {out, aout};
  CUtensorMap mc = map2d(c, R, C, 64), mw = map2d(w, MW, C, 128);
  cudaFuncSetAttribute(mod_fwd_wg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  mod_fwd_wg_kernel<<<dim3((R + 63) / 64, 3), 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      mc, mw, out.data_ptr<float>(), save ? reinterpret_cast<bf16*>(aout.data_ptr()) : nullptr, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, aout};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mod_fwd", &mod_fwd, "adaLN modulation forward on TMA + wgmma (bf16)");
}
