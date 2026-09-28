"""Isolate wide native kernels from vendor GEMMs for sanitizer diagnosis.

The full module's numerical/graph checks and unfiltered memcheck/synccheck are
separate. This runs every wide forward/backward gate kernel at full L768, with
explicit synchronization to identify the failing launch if instrumentation fails.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as wide
from bench import rel

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, choices=(384, 512), required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
d, m = a.width, 768 * 768
torch.manual_seed(1497 + d)
x = torch.randn(m, d, device='cuda', dtype=torch.bfloat16)
dy = torch.randn_like(x)
gamma = 1 + .2 * torch.randn(d, device='cuda', dtype=torch.float32)
beta = .2 * torch.randn_like(gamma)
wa, wb = [(torch.randn(4*d, d, device='cuda') * d**-.5).bfloat16() for _ in range(2)]
ws = (torch.randn(d, 4*d, device='cuda') * (4*d)**-.5).bfloat16()
ext = wide._ext_for(x)
print('FWD', d, flush=True)
y, xn, rstd, c1, h = wide._fwd_launch(x, gamma, beta, wa, wb, ws, 1e-5, True)
torch.cuda.synchronize()
assert all(bool(t.isfinite().all()) for t in (y, xn, rstd, c1, h))
print('GATE_WITH_H', d, flush=True)
recomputed_h, dab = ext.gate(xn, dy, wide._pack(wa, wb, 128), ws.t().contiguous(), True)
torch.cuda.synchronize()
assert all(bool(t.isfinite().all()) for t in (recomputed_h, dab))
print('GATE_NO_H', d, flush=True)
_, dab_noh = ext.gate(xn, dy, wide._pack(wa, wb, 128), ws.t().contiguous(), False)
torch.cuda.synchronize()
assert torch.equal(dab, dab_noh), rel(dab, dab_noh)
r = {'D': d, 'L': 768, 'complete': True, 'scope': 'isolated native forward, squeeze, gate and no-h gate kernels',
     'source_sha256': os.environ.get('TRANSITION_DIAGNOSTIC_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'gate_dab_bitwise_equal': True, 'forward_vs_recomputed_h_rel': rel(h, recomputed_h)}
(a.out / f'D{d}-L768.json').write_text(json.dumps(r, indent=2))
print('PASS', d, flush=True)
