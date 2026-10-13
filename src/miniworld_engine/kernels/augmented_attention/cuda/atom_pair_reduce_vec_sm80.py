"""E221 private original vector-eight-key BF partial reduction + original atom pair-bias backward."""
import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home,host_flags,load_extension
@functools.lru_cache(maxsize=1)
def _ext():
 ensure_cuda_home();directory=Path(__file__).parent/'sm80'
 return load_extension(name='atom_pair_reduce_vec_sm80',sources=[str(directory/'atom_pair_reduce_vec.cu')],extra_cuda_cflags=[*host_flags(),'-O3','-std=c++17','-lineinfo','-Xptxas=-v','-gencode=arch=compute_80,code=sm_80','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT162_OPERATORS__'],extra_cflags=['-O3','-std=c++17'],verbose=True)
def run(z,w,dbp,dz,pw,mask,eps=1e-5):
 if mask is None:mask=z.new_empty((0,),dtype=__import__('torch').bool)
 _ext().run(z,w,dbp,dz,pw,mask,dbp.shape[-1],z.shape[-2],eps)
