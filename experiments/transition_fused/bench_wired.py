"""Time the wired dispatch against everything it is meant to beat.

``bench_module.py`` measures the kernels by monkeypatching the module's entry point. This one
measures what a training run actually gets -- the same `modules.Transition`, the same settings
object, the only difference being ``transition_fused_sm90a`` -- and puts plain PyTorch next to
it, eager and compiled, built from the same weights through the engine's own reference.

  python bench_wired.py --length 384 [--no-compile] [--save records/wired-L384.json]
"""
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
p.add_argument("--no-compile", action="store_true", help="skip the torch.compile row")
p.add_argument("--save", default="")
a = p.parse_args()

from miniworld_engine import settings  # noqa: E402
from miniworld_engine.kernels.transition.reference import transition_pytorch  # noqa: E402
from miniworld_engine.modules import Transition  # noqa: E402

D, N, L = 128, 4, a.length
torch.manual_seed(2319)
base = Transition(D, n=N, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for param in base.parameters():
        if param.ndim == 2:
            param.normal_(std=D**-0.5)   # squeeze is zero-init, which makes four of the five
                                         # parameter gradients exactly zero in every backend
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn_like(x)

#: One name per gradient, so the engine module and the PyTorch one can be compared at all.
NAMES = ("ln_weight", "ln_bias", "expand_a", "expand_b", "squeeze")


class TorchTransition(torch.nn.Module):
    """The same op as `modules.Transition`, written in plain PyTorch ops, over the same
    weights. This is the baseline a reader means by "versus PyTorch": LayerNorm, two expand
    GEMMs, SwiGLU, the squeeze GEMM and the residual, each its own launch, with autograd
    building the backward out of its own kernels."""

    def __init__(self, src):
        super().__init__()
        clone = lambda t: torch.nn.Parameter(t.detach().clone())
        self.ln_weight, self.ln_bias = clone(src.ln_in.weight), clone(src.ln_in.bias)
        self.expand_a, self.expand_b = clone(src.expand_a.weight), clone(src.expand_b.weight)
        self.squeeze = clone(src.squeeze.weight)
        self.eps = src.ln_in.eps

    def forward(self, t):
        return transition_pytorch(t, self.ln_weight, self.ln_bias, self.expand_a,
                                  self.expand_b, self.squeeze, N, self.eps)


def engine(fused):
    settings.configure(engine_backend="triton", transition_residual_fusion=True,
                       transition_fused_sm90a=fused)
    mod = copy.deepcopy(base)
    named = dict(zip(NAMES, [mod.ln_in.weight, mod.ln_in.bias, mod.expand_a.weight,
                             mod.expand_b.weight, mod.squeeze.weight]))
    return mod, named


def pytorch(compiled):
    mod = TorchTransition(base)
    named = {n: getattr(mod, n) for n in NAMES}
    return (torch.compile(mod) if compiled else mod), named


def stepper(mod, named):
    def step():
        for t in [x, *named.values()]:
            t.grad = None
        y = mod(x)
        y.backward(dy)
        return y
    return step


BACKENDS = [("fused_sm90a", lambda: engine(True)), ("triton_residual", lambda: engine(False)),
            ("pytorch_eager", lambda: pytorch(False))]
if not a.no_compile:
    BACKENDS.append(("pytorch_compiled", lambda: pytorch(True)))

results, timing = {}, {}
for name, make in BACKENDS:
    mod, named = make()
    step = stepper(mod, named)
    y = step()
    torch.cuda.synchronize()
    results[name] = {"out": y.detach().float(), "dx": x.grad.detach().float(),
                     **{n: t.grad.detach().float() for n, t in named.items()}}
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
    print(f"RESULT {name:>18s}: {timing[name]['median']:8.1f} us "
          f"(min {timing[name]['min']:.1f})", flush=True)

fused_us = timing["fused_sm90a"]["median"]
speedup = {n: t["median"] / fused_us for n, t in timing.items() if n != "fused_sm90a"}
print(f"\nfused sm_90a at L={L}, module forward + backward:", flush=True)
for n, s in speedup.items():
    print(f"  {s:5.2f}x vs {n} ({timing[n]['median']:.1f} -> {fused_us:.1f} us)", flush=True)

cmp = {}
for ref in ("triton_residual", "pytorch_eager"):
    cmp[ref] = {k: float((results["fused_sm90a"][k] - v).norm() / v.norm().clamp_min(1e-20))
                for k, v in results[ref].items()}
    print(f"\n  rel_rms of fused vs {ref}:", flush=True)
    for k, v in cmp[ref].items():
        print(f"    {k:>10s}: {v:.3e}", flush=True)

rec = dict(length=L, speedup=speedup, time_us=timing, rel_rms=cmp)
if a.save:
    out = Path(a.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("\nsaved", out, flush=True)
