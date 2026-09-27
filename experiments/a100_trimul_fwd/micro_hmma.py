"""HMMA.16816 (bf16 -> fp32) issue throughput on this card vs warps per SM sub-partition and independent accumulators per warp."""
import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
template <int NACC>
__global__ void k(float* out, int iters) {
  float acc[NACC][4];
  for (int i = 0; i < NACC; ++i) for (int e = 0; e < 4; ++e) acc[i][e] = 0.f;
  uint32_t a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float s = 0.f;
  for (int i = 0; i < NACC; ++i) for (int e = 0; e < 4; ++e) s += acc[i][e];
  if (s == 12345.f) out[0] = s;
}
double run(int warps_per_block, int nacc, int iters) {
  auto out = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto launch = [&]() {
    if (nacc == 1) k<1><<<sms, 32 * warps_per_block>>>(out.data_ptr<float>(), iters);
    else if (nacc == 2) k<2><<<sms, 32 * warps_per_block>>>(out.data_ptr<float>(), iters);
    else if (nacc == 4) k<4><<<sms, 32 * warps_per_block>>>(out.data_ptr<float>(), iters);
    else k<8><<<sms, 32 * warps_per_block>>>(out.data_ptr<float>(), iters);
  };
  launch(); cudaDeviceSynchronize();
  cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  double flops = 2.0 * 16 * 8 * 16 * (double)nacc * iters * warps_per_block * sms;
  return flops / (ms * 1e-3) / 1e12;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""
m = load_inline("micro_hmma", cpp_sources="", cuda_sources=src, extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
                build_directory=None, verbose=False, with_cuda=True)
for wpb in (4, 8, 16):
    print(f"warps/SM {wpb:2d} (per SMSP {wpb // 4}): " + "  ".join(f"nacc{n} {m.run(wpb, n, 20000):6.1f} TF" for n in (1, 2, 4, 8)))
