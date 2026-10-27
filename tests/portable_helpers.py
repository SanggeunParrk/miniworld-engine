"""The one place the module-parity tests spell the internal A/B switch that puts the portable kernels on every card.

``settings.force_portable_kernels`` is the switch; if it is renamed again, this is the only line that changes. A test uses
``with portable_on():`` (or the ``portable`` fixture in the test module) and never names the setting itself.
"""
from __future__ import annotations

import contextlib
import os

import torch

from miniworld_engine import settings

_SWITCH = "force_portable_kernels"


@contextlib.contextmanager
def portable_on(on: bool = True):
    previous = settings.configure(**{_SWITCH: on})
    try:
        yield
    finally:
        settings.configure(**vars(previous))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error in fp64."""
    a, b = a.detach().double(), b.detach().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def report(tag: str, out, out_ref, out_gold, grads=(), grads_ref=(), grads_gold=()) -> None:
    """With ``PARITY_REPORT=1`` print the measured errors of one case (forward, worst gradient) next to the PyTorch statements' own."""
    if not os.environ.get("PARITY_REPORT"):
        return
    worst = lambda a, g: max((rel(x, y) for x, y in zip(a, g, strict=True) if y is not None and x is not None), default=float("nan"))   # noqa: E731
    print(f"PARITY {tag} fwd={rel(out, out_gold):.2e} fwd_pytorch={rel(out_ref, out_gold):.2e} "
          f"bwd={worst(grads, grads_gold):.2e} bwd_pytorch={worst(grads_ref, grads_gold):.2e}", flush=True)


def randomize_normalised(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else
                    1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
    return m


def run_extra_pair_bias(block, data, dtype, training):
    single, cond, pair, mask, extra, dy = data
    single, cond, pair, extra = (t.to(dtype).clone().requires_grad_(training) for t in (single, cond, pair, extra))
    with torch.enable_grad() if training else torch.no_grad():
        out = block(single, cond, pair, mask, extra_pair_bias=extra)
        if not training:
            return out, ()
        return out, torch.autograd.grad(out, [single, cond, pair, extra, *block.parameters()], dy.to(dtype), allow_unused=True)


def randomize_half_std(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5 * 0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def run_pair_init(fn, leaves, ids, bond, **kw):
    left, right, w_rel, w_bond = leaves
    asym, resi, tok, ent, sym = ids
    return fn(left, right, w_rel, w_bond, asym, resi, tok, ent, sym, bond, **kw)


@contextlib.contextmanager
def tf32_matmul_flag(on):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def relative_fp64(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def relative(a, b):
    return float(
        (a.detach().float() - b.detach().float()).norm()
        / b.detach().float().norm().clamp_min(1e-12)
    )


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


def poison_allocator(mb=512):
    """Fill the caching allocator's next blocks with NaN: a buffer the step reads before writing would show it."""
    junk = torch.full((mb << 18,), float("nan"), device="cuda")
    torch.cuda.synchronize()
    del junk


def relative_float(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
