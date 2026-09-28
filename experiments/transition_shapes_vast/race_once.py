"""One complete full-shape F+B per native variant under racecheck.

Repeated and changed-state CUDA graphs are checked separately by bench.py under
normal execution, memcheck and synccheck. Repeating identical WGMMA kernels under
race instrumentation is prohibitively expensive and adds no new code coverage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.cuda import fused_sm90a, fused_wide_sm90a
from bench import rel

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, required=True, choices=(128, 256, 384, 512))
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
d = a.width
torch.manual_seed(1497 + d)
settings.configure(engine_backend='auto', transition_residual_fusion=True, transition_fused_sm90a=True)
mod = Transition(d, n=4, implementation='miniworld').cuda().bfloat16()
with torch.no_grad():
    for name, t in mod.named_parameters():
        if t.ndim == 2:
            t.normal_(std=t.shape[-1] ** -.5)
        elif name == 'ln_in.weight':
            t.copy_(1 + .2 * torch.randn_like(t))
        else:
            t.normal_(std=.2)
x = torch.randn((1, 768, 768, d), device='cuda', dtype=torch.bfloat16).requires_grad_()
dy = torch.randn_like(x)
leaves = (x, *mod.parameters())
names = ('y', 'dx', *dict(mod.named_parameters()))
backend = fused_sm90a if d == 128 else fused_wide_sm90a
entry = 'transition_fused_sm90a' if d == 128 else 'transition_wide_sm90a'
original = getattr(backend, entry)
calls = []
def observed(*args):
    calls.append(1)
    return original(*args)
setattr(backend, entry, observed)
record = {'D': d, 'L': 768, 'complete': False, 'protocol': 'one full F+B per save-h variant',
          'race_kernel_filter': os.environ.get('RACE_KERNEL_FILTER', 'all'),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'variants': {}}
for save_h in ((0, 1) if d >= 384 else (0,)):
    os.environ['MINIWORLD_TRANSITION_WIDE_SAVE_H'] = str(save_h)
    print('FULL_F+B', d, 'save_h', save_h, flush=True)
    y = mod(x)
    values = (y, *torch.autograd.grad(y, leaves, dy))
    torch.cuda.synchronize()
    assert all(bool(t.isfinite().all()) for t in values)
    assert len(calls) == save_h + 1
    if save_h == 0:
        baseline = tuple(t.detach().clone() for t in values)
        errors = {n: 0.0 for n in names}
    else:
        errors = {n: rel(g, w) for n, g, w in zip(names, values, baseline)}
        assert all(v < (.002 if n == 'squeeze.weight' else 1e-5) for n, v in errors.items()), errors
    record['variants'][str(save_h)] = errors
    print('PASS_VARIANT', d, save_h, flush=True)
record['complete'] = True
(a.out / f'D{d}-L768.json').write_text(json.dumps(record, indent=2))
print('PASS', d, flush=True)
