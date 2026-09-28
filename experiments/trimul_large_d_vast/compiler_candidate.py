"""Build the unchanged saved forward with the image's CUDA 13.1 compiler.
The explicit candidate keeps CUDA 12.8 runtime and math/launch configuration.
"""
import subprocess
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_saved_front_overlap import SavedOverlapFront
from wide_loader_warp_gp import LoaderWarpGP

def build(plan,component):
    nvcc='/usr/local/cuda/bin/nvcc';version=subprocess.check_output([nvcc,'--version'],text=True)
    assert 'release 13.1' in version,version
    old=T._compiler
    try:
        T._compiler=lambda:(nvcc,version)
        if component=='front':result=SavedOverlapFront(plan.baseline_front,plan.pre)
        elif component=='contract':
            assert (plan.p.D,plan.p.n)==(512,384)
            result=LoaderWarpGP(plan,False,True)
        else:raise ValueError(component)
    finally:T._compiler=old
    return result,version
