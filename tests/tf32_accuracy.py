"""Error of an engine module in fp32 (TF32 tensor cores) against an fp64 run, as a multiple of the PyTorch statements' own error.

The repo yardstick: the engine's error against an fp64 run of the same inputs and cotangent is at most 1.5x the error of the PyTorch
statements in the same dtype and matmul precision, for the output and for every gradient. ``measure`` returns one row per tensor
(output, each input gradient, each parameter gradient); ``worst`` and ``table`` condense them. The portable (Triton) kernels are the
subject: a caller wraps ``measure`` in ``portable_on()``.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch

from miniworld_engine.modules.exceptions import ImplementationType

YARDSTICK = 1.5
#: a tensor whose fp64 value is this small, relative to the largest of its kind, is judged absolutely (its error is noise: a LayerNorm
#: bias on the keys shifts every logit of a query by one constant, which the softmax cancels)
TINY = 0.02


@dataclass
class Row:
    name: str
    engine: float
    pytorch: float

    @property
    def ratio(self) -> float:
        return self.engine / max(self.pytorch, 1e-12)


@contextlib.contextmanager
def tf32(on: bool = True):
    """TF32 matmuls for fp32 on (``set_float32_matmul_precision("high")``, MiniWorld's recipe) or off."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if on else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def randomise(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            r = torch.randn(p.shape, generator=g)
            p.copy_((r / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.1 * r + (1.0 if "weight" in name and ("norm" in name or "ln" in name) else 0.0)).to(p.dtype))
    return module


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def measure(make, args, call=None, *, train: bool = True, live=None, seed: int = 1) -> list[Row]:
    """``make(implementation)`` builds the module (same seed, any weights); ``args`` are fp32 cuda tensors (floating ones get a gradient
    in training); ``call(module, xs)`` runs it (default ``module(*xs)``). Runs the fp64 PyTorch module (the gold), the PyTorch module in
    fp32 and the MINIWORLD module in fp32, all under TF32, with the same cotangent. ``live``: an index applied to the output (padded rows)."""
    call = call or (lambda m, xs: m(*xs))
    ref = make(ImplementationType.PYTORCH).cuda().double()
    served = make(ImplementationType.MINIWORLD).cuda()
    served.load_state_dict(ref.state_dict())
    own = make(ImplementationType.PYTORCH).cuda()
    own.load_state_dict(ref.state_dict())
    for m in (ref, served, own):
        m.train(train)
    g = torch.Generator(device="cuda").manual_seed(seed)
    out0 = None

    def run(module, dt):
        nonlocal out0
        xs = [t.to(dt).clone().requires_grad_(train) if t is not None and t.is_floating_point() else t for t in args]
        for p in module.parameters():
            p.grad = None
        with torch.enable_grad() if train else torch.no_grad():
            out = call(module, xs)
            if out0 is None:
                out0 = torch.randn(out.shape, device="cuda", generator=g)
            if train:
                out.backward(out0.to(out.dtype))
        sel = (lambda t: t) if live is None else (lambda t: t[live])
        grads = [x.grad for x in xs if isinstance(x, torch.Tensor) and x.requires_grad] if train else []
        params = {k: p.grad.double() for k, p in module.named_parameters() if p.grad is not None} if train else {}
        return sel(out.detach()), grads, params

    with tf32(True):
        want, base, got = run(ref, torch.float64), run(own, torch.float32), run(served, torch.float32)
    rows = [Row("out", _rel(got[0], want[0]), _rel(base[0], want[0]))]
    for i, (a, b, w) in enumerate(zip(got[1], base[1], want[1], strict=True)):
        rows.append(Row(f"d_arg{i}", _rel(a, w), _rel(b, w)))
    if want[2]:
        largest = max(w.norm() for w in want[2].values())
        for k, w in want[2].items():
            if w.norm() < TINY * largest:
                continue
            rows.append(Row(f"d_{k}", _rel(got[2][k], w), _rel(base[2][k], w)))
    return rows


def worst(rows: list[Row]) -> Row:
    return max(rows, key=lambda r: r.ratio)


def table(tag: str, rows: list[Row]) -> str:
    out, grads = rows[0], [r for r in rows[1:]]
    w = worst(rows)
    g = worst(grads) if grads else None
    s = f"TF32ACC {tag} out={out.engine:.2e}/{out.pytorch:.2e}={out.ratio:.2f}"
    if g:
        s += f" worst_grad[{g.name}]={g.engine:.2e}/{g.pytorch:.2e}={g.ratio:.2f}"
    return s + f" worst={w.ratio:.2f}"


def report(tag: str, rows: list[Row]) -> None:
    if os.environ.get("PARITY_REPORT") or os.environ.get("TF32ACC_REPORT"):
        print(table(tag, rows), flush=True)
        if os.environ.get("TF32ACC_VERBOSE"):
            for r in rows:
                print(f"   {r.name}: {r.engine:.2e} / {r.pytorch:.2e} = {r.ratio:.2f}", flush=True)
