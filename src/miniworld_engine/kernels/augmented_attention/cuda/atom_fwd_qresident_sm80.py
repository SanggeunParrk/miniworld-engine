"""E257 private forward Q-fragment resident experiment; original kernels remain intact."""
from functools import lru_cache
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
@lru_cache(None)
def _ext():
 ensure_cuda_home()
 d=Path(__file__).resolve().parent
 return load_extension(name='atom_fwd_qresident257_sm80',sources=[str(d/'sm80_fwd_qresident257/ops.cu')],extra_include_paths=[str(d/'sm80'),str(d/'sm80_fwd_qresident257')],extra_cuda_cflags=[*host_flags(),'-O3','-std=c++17','-lineinfo','-Xptxas=-v','-gencode=arch=compute_80,code=sm_80','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT162_OPERATORS__'],extra_cflags=['-O3','-std=c++17'],verbose=True)
