"""E168 private rolled atom dy@Wo and original A100 gate-backward final stores."""
import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent
    adaln = Path(__file__).parents[2] / 'adaln/cuda/sm80'
    return load_extension(name='atom_post_grad_rolled_sm80', sources=[str(directory / 'sm80/atom_post_grad_rolled.cu')], extra_include_paths=[str(adaln)], extra_cuda_cflags=[*host_flags(), '-O3', '-std=c++17', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_80,code=sm_80', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_OPERATORS__', '-U__CUDA_NO_BFLOAT162_OPERATORS__'], extra_cflags=['-O3', '-std=c++17'], verbose=True)

def run(dy, wo, o, gate, dob, dg, dog=None, cfg=1):
    if dog is None:
        dog = dy.new_empty((0,))
    _ext().run(dy, wo, o, gate, dob, dg, dog, cfg)
