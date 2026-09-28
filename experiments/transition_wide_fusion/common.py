"""Shared inputs and complete Transition forward/backward experiment contract."""
import hashlib
import os
from pathlib import Path

import torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide
from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd
from bench import capture, paired, rel


def inputs(d, length):
    torch.manual_seed(9100 + d + length)
    m = length * length
    dev = dict(device='cuda', dtype=torch.bfloat16)
    x, dy = (torch.randn((m, d), **dev) for _ in range(2))
    gamma = 1 + .2 * torch.randn(d, device='cuda')
    beta = .2 * torch.randn(d, device='cuda')
    wa, wb = (torch.randn((4*d, d), **dev) * d**-.5 for _ in range(2))
    ws = torch.randn((d, 4*d), **dev) * (4*d)**-.5
    return x, gamma, beta, wa, wb, ws, dy


def baseline(v):
    x, gamma, beta, wa, wb, ws, dy = v
    y, xn, rstd, c1, _ = wide._fwd_launch(x, gamma, beta, wa, wb, ws, 1e-5, True)
    grads = wide._bwd_launch(dy, x, xn, rstd, c1, gamma, wa, wb, ws, x.new_empty((1, 1)))
    # Autograd returns gradients in each parameter's original dtype.
    return (y, *grads[:3], *(g.to(torch.bfloat16) for g in grads[3:]))


def identity():
    root = Path(__file__).resolve().parents[2]
    files = list((root / 'src/miniworld_engine/kernels/transition').rglob('*.py'))
    files += list((root / 'src/miniworld_engine/kernels/transition/cuda').rglob('*.cu'))
    files += list((root / 'src/miniworld_engine/kernels/transition/cuda').rglob('*.cuh'))
    files += list(Path(__file__).parent.glob('*.py'))
    files += list(Path(__file__).parent.glob('*.cu'))
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, sms=p.multi_processor_count, torch=torch.__version__,
                cuda=torch.version.cuda, job=os.environ.get('SLURM_JOB_ID'),
                qos=os.environ.get('SLURM_JOB_QOS'), node=os.environ.get('SLURMD_NODENAME'),
                sources={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)})
