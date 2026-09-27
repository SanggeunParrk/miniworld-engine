import os, pathlib, torch
from torch.utils.cpp_extension import load_inline
src = r'''
__global__ void k(int* o) { extern __shared__ __align__(1024) unsigned char raw[]; if (threadIdx.x == 0) o[blockIdx.x] = (int)(static_cast<unsigned>(__cvta_generic_to_shared(raw)) & 1023u); }
torch::Tensor run(int smem) { auto o = torch::zeros({148}, torch::dtype(torch::kInt32).device(torch::kCUDA)); cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); k<<<148, 32, smem>>>(o.data_ptr<int>()); return o; }
'''
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "smalign"; d.mkdir(parents=True, exist_ok=True)
m = load_inline("smalign", cpp_sources="torch::Tensor run(int smem);", cuda_sources=src, functions=["run"], build_directory=str(d), extra_cuda_cflags=["-gencode=arch=compute_100a,code=sm_100a"])
for sm in (1024, 100000, 232448): print(sm, torch.unique(m.run(sm)).tolist())
