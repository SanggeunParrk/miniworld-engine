"""Independent A100 gradient-join experiment with actual aligned vector loads/stores."""
import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    path = Path(__file__).parent
    return load_extension(name='atom_gradient_join_vector_sm80', sources=[str(path / 'atom_gradient_join_vector_sm80.cu')], extra_cuda_cflags=[*host_flags(), '-O3', '-std=c++17', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_80,code=sm_80'], extra_cflags=['-O3', '-std=c++17'], verbose=True)

def run(a, b, c, out, planes=2, cfg=0, intermediate=False):
    _ext().run(a, b, c, out, planes, cfg, intermediate)
