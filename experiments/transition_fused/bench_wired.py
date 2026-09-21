"""Time the wired dispatch: ``modules.Transition`` with the fused sm_90a path on and off.

``bench_module.py`` measures the kernels by monkeypatching the module's entry point. This one
measures what a training run actually gets -- the same module, the same settings object, the
only difference being ``transition_fused_sm90a``. If the two disagree, the wiring is the reason.

  python bench_wired.py --length 384 [--save records/wired-L384.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--save", default="")
a = p.parse_args()

from miniworld_engine import settings  # noqa: E402
from miniworld_engine.modules import Transition  # noqa: E402

D, L = 128, a.length
torch.manual_seed(2319)
base = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for param in base.parameters():
        if param.ndim == 2:
            param.normal_(std=D**-0.5)   # squeeze is zero-init, which makes four of the five
                                         # parameter gradients exactly zero in every backend
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn_like(x)


def build(fused: bool):
    settings.configure(engine_backend="triton", transition_residual_fusion=True,
                       transition_fused_sm90a=fused)
    mod = copy.deepcopy(base)

    def step():
        for t in [x, *mod.parameters()]:
            t.grad = None
        y = mod(x)
        y.backward(dy)
        return y

    return mod, step


def grads_of(mod, y):
    return {"out": y.detach().float(), "dx": x.grad.detach().float(),
            **{n: p.grad.detach().float() for n, p in mod.named_parameters()}}


results, timing = {}, {}
for name, fused in (("fused_sm90a", True), ("triton_residual", False)):
    mod, step = build(fused)
    y = step()
    torch.cuda.synchronize()
    results[name] = grads_of(mod, y)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    runs = []
    for _ in range(a.rounds):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(a.reps):
            step()
        en.record()
        torch.cuda.synchronize()
        runs.append(st.elapsed_time(en) * 1000.0 / a.reps)
    timing[name] = dict(median=statistics.median(runs), min=min(runs), samples=runs)
    print(f"RESULT {name:>16s}: {timing[name]['median']:8.1f} us (min {timing[name]['min']:.1f})",
          flush=True)

speedup = timing["triton_residual"]["median"] / timing["fused_sm90a"]["median"]
print(f"speedup: {speedup:.2f}x  (module forward + backward, L={L})", flush=True)

cmp = {}
for k, want in results["triton_residual"].items():
    got = results["fused_sm90a"][k]
    cmp[k] = float((got - want).norm() / want.norm().clamp_min(1e-20))
    print(f"  {k:>22s}: rel_rms vs triton {cmp[k]:.3e}", flush=True)

rec = dict(length=L, speedup=speedup, time_us=timing, rel_rms_vs_triton=cmp)
if a.save:
    out = Path(a.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("saved", out, flush=True)
