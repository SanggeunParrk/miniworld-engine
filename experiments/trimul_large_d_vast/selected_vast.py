"""Explicit large-D plan factory; no production dispatch or autograd registration.

Use one width per process with the frozen research snapshot and CUDA 12.8 runtime.
Calls require torch.no_grad(). Each plan owns activations and is stream-bound.
"""
import os
from pathlib import Path
import sys


def make_plan(leaves, mask, dropscale, dy):
    root = Path(os.environ.get('VAST_TRIMUL_RESEARCH_ROOT',
                               '/workspace/experiments/trimul-large-d/runs'))
    directories = ('trimul_cuda_widths_opt_20260923',
                   'trimul_forward_wide_20260923',
                   'trimul_backward_wide_20260923',
                   'trimul_d256_bwd_sol90_20260923',
                   'trimul_d256_bwd_sol90_stage2_20260923')
    for directory in directories:
        path = root / directory
        if not path.is_dir():
            raise FileNotFoundError(f'Missing frozen snapshot: {path}; see README.md')
        sys.path.insert(0, str(path))
    import torch
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as runtime
    width, length = leaves[0].shape[-1], leaves[0].shape[1]
    if width not in (256, 384, 512) or length not in (384, 768):
        raise ValueError('Only D256/384/512 at L384/768 is covered')
    os.environ.update(PREFIX_IMPL='blas', PREFIX_COPY='tma', CHECKPOINT_LN_THREADS='0')
    if width == 256:
        from d256_pool_checkpoint import Training, configure
        configure()
    elif width == 384:
        from wide_checkpoint23 import Training
    else:
        from wide_checkpoint24 import Training
    with torch.no_grad(), runtime.native_context(leaves[0].device):
        plan = Training(leaves, mask, dropscale, dy)
        if (width, length) == (512, 384):
            # Only the measured short-wide cell changes its input-weight split.
            from input_split import attach
            attach(plan, 4, 0)
            plan.artifacts.append(plan.dx.reduce_only.cubin)
        return plan
