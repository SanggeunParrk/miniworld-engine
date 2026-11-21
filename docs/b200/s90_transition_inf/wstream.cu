// wstream.cu -- weight-stream bound of a fused D 512 pair kernel: every CTA streams WBYTES of weights per tile (cyclic over a 1 MB L2-resident buffer, the expand + squeeze weights of one
// CTA's share) through a ring of NSTG stages of STG bytes with cp.async.bulk (one lane a stage), consuming a stage at a fixed MMA-time pace (clock64 spin of CLK per stage): reports us a tile
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#ifndef STG
#define STG 16384
#endif
#ifndef NSTG
#define NSTG 8
#endif
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__global__ void __launch_bounds__(128, 1) wstream(const uint8_t* w, long long wsize, int tiles, int per_tile, int spin, long long* out) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t full[NSTG], empty[NSTG];
  const int t = threadIdx.x;
  if (t == 0) { for (int s = 0; s < NSTG; ++s) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su(&full[s]))); asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su(&empty[s]))); } asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  const int total = tiles * per_tile;
  long long t0 = clock64();
  if (t == 0) {          // producer
    for (int q = 0; q < total; ++q) {
      const int s = q % NSTG;
      if (q >= NSTG) { asm volatile("{ .reg .pred p; W1: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W1; }" :: "r"(su(&empty[s])), "r"(((q / NSTG) - 1) & 1)); }
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su(&full[s])), "r"(STG));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(su(sm + s * STG)), "l"(w + ((long long)q * STG) % wsize), "r"(STG), "r"(su(&full[s])) : "memory");
    }
  } else if (t == 32) {  // consumer: waits a stage, spins `spin` clocks (the MMAs of the stage), frees it
    for (int q = 0; q < total; ++q) {
      const int s = q % NSTG;
      asm volatile("{ .reg .pred p; W2: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W2; }" :: "r"(su(&full[s])), "r"((q / NSTG) & 1));
      long long a = clock64(); while (clock64() - a < spin) {}
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su(&empty[s])));
    }
    out[blockIdx.x] = clock64() - t0;
  }
}
void run(torch::Tensor w, int tiles, int per_tile, int spin, torch::Tensor out, int nb) {
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(wstream, cudaFuncAttributeMaxDynamicSharedMemorySize, NSTG * STG); init = true; }
  wstream<<<nb, 128, NSTG * STG, c10::cuda::getCurrentCUDAStream()>>>((const uint8_t*)w.data_ptr(), (long long)w.numel(), tiles, per_tile, spin, (long long*)out.data_ptr());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
