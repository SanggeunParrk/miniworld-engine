// one-launch skeleton of the small-M Transition chain: ONE cluster of CS CTAs, one 128-row tile, D = DDIM, hidden H = 2 DDIM (n 2): only the data movement
// and synchronisation of the real design (weights slice -> smem by bulk copies, x slice in, xn all-gather by DSMEM bulk push, h all-gather by DSMEM bulk push,
// y slice out), with no arithmetic and with the destination buffers overlaid (an optimistic lower bound: a real kernel cannot hold them all).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#ifndef CS
#define CS 16
#endif
#ifndef DDIM
#define DDIM 384
#endif
constexpr int ROWS = 128, H = 2 * DDIM;
constexpr int WBYTES = 3 * DDIM * H * 2 / CS;          // this CTA's slice of Wa, Wb, Ws (bf16)
constexpr int XROWS = ROWS / CS, XBYTES = XROWS * DDIM * 2;
constexpr int HBYTES = ROWS * (H / CS) * 2;           // h slice of the hidden units a CTA owns
constexpr int YBYTES = ROWS * (DDIM / CS) * 4 / 2;     // bf16 y slice
constexpr int OFF_W = 0, XG = ROWS * DDIM * 2, SMEM = 232448 - 1024;
static_assert(XG + 0 <= SMEM, "");
constexpr int HG_BYTES = HBYTES * CS;
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su(b)), "r"(n)); }
__device__ __forceinline__ void mb_tx(uint64_t* b, int bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su(b)), "r"(bytes)); }
__device__ __forceinline__ void mb_wait(uint64_t* b, int par) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(su(b)), "r"(par));
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, int r) { uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o; }
__device__ __forceinline__ void cl_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ int cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__global__ void __launch_bounds__(128, 1) skel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ x, uint8_t* __restrict__ y, int stamps, long long* st) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar[4];
  const int rk = cl_rank(), t = threadIdx.x;
  long long t0 = clock64();
  if (t == 0) {
    for (int i = 0; i < 4; ++i) mb_init(&bar[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
    mb_tx(&bar[0], WBYTES); mb_tx(&bar[1], XBYTES); mb_tx(&bar[2], (CS - 1) * XBYTES); mb_tx(&bar[3], (CS - 1) * HBYTES);
  }
  cl_sync();
  if (t < 32) {                                           // one warp; every bulk copy issued by a lane of its own
    const int ln = t;
    for (int o = ln * 16384; o < WBYTES; o += 32 * 16384) {
      int n = min(16384, WBYTES - o);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(su(sm + OFF_W + o)), "l"(w + (size_t)rk * WBYTES + o), "r"(n), "r"(su(&bar[0])) : "memory");
    }
    if (ln == 31) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(su(sm + XG + rk * XBYTES)), "l"(x + (size_t)rk * XBYTES), "r"(XBYTES), "r"(su(&bar[1])) : "memory");
    mb_wait(&bar[1], 0);
    if (stamps && ln == 0) st[rk * 8 + 0] = clock64() - t0;
    if (ln < CS && ln != rk)
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(mapa(su(sm + XG + rk * XBYTES), ln)), "r"(su(sm + XG + rk * XBYTES)), "r"(XBYTES), "r"(mapa(su(&bar[2]), ln)) : "memory");
    mb_wait(&bar[2], 0);
    if (stamps && ln == 0) st[rk * 8 + 1] = clock64() - t0;
    mb_wait(&bar[0], 0);
    if (stamps && ln == 0) st[rk * 8 + 2] = clock64() - t0;
    if (ln < CS && ln != rk)
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(mapa(su(sm + rk * HBYTES), ln)), "r"(su(sm + rk * HBYTES)), "r"(HBYTES), "r"(mapa(su(&bar[3]), ln)) : "memory");
    mb_wait(&bar[3], 0);
    if (stamps && ln == 0) st[rk * 8 + 3] = clock64() - t0;
  }
  __syncthreads();
  for (int i = t; i < YBYTES / 16; i += 128) reinterpret_cast<uint4*>(y + (size_t)rk * YBYTES)[i] = reinterpret_cast<uint4*>(sm)[i];
  cl_sync();
  if (t == 0 && stamps) st[rk * 8 + 4] = clock64() - t0;
}
void run(torch::Tensor w, torch::Tensor x, torch::Tensor y, torch::Tensor st, int stamps) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(skel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(skel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CS); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = SMEM; cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at; at.id = cudaLaunchAttributeClusterDimension; at.val.clusterDim.x = CS; at.val.clusterDim.y = 1; at.val.clusterDim.z = 1;
  cfg.attrs = &at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, skel, (const uint8_t*)w.data_ptr(), (const uint8_t*)x.data_ptr(), (uint8_t*)y.data_ptr(), stamps, (long long*)st.data_ptr());
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
