"""Run the backward a few times on one shape (for ncu): python prof_op.py 384 [flags...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402

L = int(sys.argv[1])
ext = TB.build(extra=sys.argv[2:])
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
bufs = {}
import os
mode = os.environ.get("PROF_MODE", "")
if mode == "eng":
    import copy
    from miniworld_engine.modules.exceptions import ImplementationType as I
    from miniworld_engine.modules.dispatch import resolve_transition
    emod = copy.deepcopy(mod); emod.implementation = I.MINIWORLD; emod._backend = resolve_transition(I.MINIWORLD); emod.train()
    xg = x.view(1, L, L, 128).clone().requires_grad_(True)
    dyg = torch.randn_like(xg)
    eparams = [xg] + list(emod.parameters())
fn = {"eng": lambda: torch.autograd.grad(emod(xg), eparams, dyg), "two": lambda: TB.backward_2k(ext, x, dy, pk, bufs), "seq": lambda: TB.backward_seq(ext, x, dy, pk, bufs)}.get(mode, lambda: TB.backward(ext, x, dy, pk, bufs))
for _ in range(4):
    fn()
torch.cuda.synchronize()
