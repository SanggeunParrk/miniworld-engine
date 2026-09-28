"""Paired frozen pre-fix extension versus installed D128 input-slot barrier fix."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.cuda import fused_sm90a as native
from bench import capture, paired, rel

p = argparse.ArgumentParser()
p.add_argument('--length', type=int, choices=(384, 768), required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
name = 'transition_fused_sm90a_c132_r8_s1'
old_path = Path(os.environ['TORCH_EXTENSIONS_DIR']) / name / (name + '.so')
spec = importlib.util.spec_from_file_location(name, old_path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
settings.configure(engine_backend='auto', transition_residual_fusion=True, transition_fused_sm90a=True)
original = native._ext
new = original(132, 8, True)
torch.manual_seed(8192 + a.length)
module = Transition(128, implementation='miniworld').cuda().bfloat16()
with torch.no_grad():
    for name, t in module.named_parameters():
        if t.ndim == 2:
            t.normal_(std=t.shape[-1] ** -.5)
        elif name == 'ln_in.weight':
            t.copy_(1 + .2 * torch.randn_like(t))
        else:
            t.normal_(std=.2)
x = torch.randn(1, a.length, a.length, 128, device='cuda', dtype=torch.bfloat16).requires_grad_()
dy = torch.randn_like(x)
names = ['y', 'dx', *dict(module.named_parameters())]
def step():
    y = module(x)
    return (y, *torch.autograd.grad(y, (x, *module.parameters()), dy))

graphs, outputs = {}, {}
for key, ext in (('before', old), ('fixed', new)):
    native._ext = lambda *unused, _ext=ext: _ext
    graphs[key], outputs[key] = capture(step)
errors = {n: rel(g, w) for n, g, w in zip(names, outputs['fixed'], outputs['before'])}
assert all(torch.equal(g, w) for g, w in zip(outputs['fixed'], outputs['before'])), errors
times = paired(graphs, 150)
result = {'L': a.length, 'D': 128, 'complete': True, 'errors': errors, 'times': times,
          'extension_sha256': {k: hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
                               for k, e in (('before', old), ('fixed', new))}}
(a.out / f'D128-L{a.length}.json').write_text(json.dumps(result, indent=2))
print({k: t['median_ms'] for k, t in times.items()}, errors, flush=True)
native._ext = original
