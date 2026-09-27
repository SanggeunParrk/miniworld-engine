"""Engine baseline on this card: Transition(128, n=4) training step (fwd + bwd through autograd) with the default A100 dispatch,
timed as one CUDA graph (fwd alone, fwd + bwd), plus each gradient's rel-RMS against an fp32 autograd run of the same module.
    python baseline.py --length 384 768"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "a100_transition_fwd"))
import transition_a100 as TA  # noqa: E402
from bench_common import graph_ms, rel_rms  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", nargs="+", type=int, default=[384, 768])
p.add_argument("--out", default=None)
a = p.parse_args()
rec = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, rows=[])
for L in a.length:
    ref_mod, x = TA.fixture(L)
    x = x.view(1, L, L, 128)                                   # the engine keys its kernels on the unflattened pair shape
    mod = copy.deepcopy(ref_mod)
    mod.implementation = I.MINIWORLD
    from miniworld_engine.modules.dispatch import resolve_transition
    mod._backend = resolve_transition(I.MINIWORLD)
    mod.train()
    torch.manual_seed(7)
    dy = torch.randn_like(x)
    params = [mod.ln_in.weight, mod.ln_in.bias, mod.expand_a.weight, mod.expand_b.weight, mod.squeeze.weight]
    names = ["dx", "dgamma", "dbeta", "dWa", "dWb", "dWs"]
    # fp32 reference
    m32 = copy.deepcopy(ref_mod).float().train()
    x32 = x.float().requires_grad_(True)
    p32 = [m32.ln_in.weight, m32.ln_in.bias, m32.expand_a.weight, m32.expand_b.weight, m32.squeeze.weight]
    ref = torch.autograd.grad(m32(x32), [x32] + p32, dy.float())
    xg = x.clone().requires_grad_(True)
    step = lambda: torch.autograd.grad(mod(xg), [xg] + params, dy)  # noqa: E731
    got = step()
    rels = {n: rel_rms(g, r) for n, g, r in zip(names, got, ref)}
    with torch.no_grad():
        fwd_ms = graph_ms(lambda: mod(xg))[0]
    both_ms = graph_ms(step)[0]
    row = dict(L=L, fwd_us=fwd_ms * 1e3, fwd_bwd_us=both_ms * 1e3, bwd_us=(both_ms - fwd_ms) * 1e3, rel_rms=rels)
    print(f"L{L}: engine fwd {fwd_ms*1e3:.1f} us  fwd+bwd {both_ms*1e3:.1f} us  (bwd ~{(both_ms-fwd_ms)*1e3:.1f})  rel "
          + " ".join(f"{k} {v:.2e}" for k, v in rels.items()), flush=True)
    rec["rows"].append(row)
if a.out:
    Path(a.out).write_text(json.dumps(rec, indent=1))
