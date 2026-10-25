"""Transition in fp32 on the portable kernels (TF32 tensor cores), forward and gradients, as the H100 generic kernels serve fp32: only where
torch allows TF32 (``torch.set_float32_matmul_precision`` not ``"highest"`` / ``allow_tf32``); with full-precision fp32 the call keeps the
PyTorch reference. Against the module's fp64 statements, held to the PyTorch statements' own TF32 error (x 2, floor 4e-3).
"""
from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F
from portable_helpers import portable_on, rel

from miniworld_engine.kernels.transition.triton import residual as residual_mod
from miniworld_engine.modules import Transition
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV = "cuda"


@pytest.fixture(autouse=True)
def portable():
    with portable_on():
        yield


@contextlib.contextmanager
def tf32(on: bool):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if on else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _module(d, n, implementation, seed=7):
    torch.manual_seed(seed)
    m = Transition(d, n=n, implementation=implementation).to(DEV)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else
                    1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
    return m


def _gold(m, x):
    """x + squeeze(silu(a) * b) in fp64 (the module's statements), differentiable in x and the parameters."""
    p = {k: v.detach().double().requires_grad_() for k, v in m.named_parameters()}
    d = x.shape[-1]
    xn = F.layer_norm(x, (d,), p["ln_in.weight"], p["ln_in.bias"], m.ln_in.eps)
    h = F.silu(xn @ p["expand_a.weight"].T) * (xn @ p["expand_b.weight"].T)
    return x + h @ p["squeeze.weight"].T, p


def _call(m, x, dy):
    x = x.clone().requires_grad_()
    out = m(x)
    grads = torch.autograd.grad(out, [x, *m.parameters()], dy)
    return out, grads


SHAPES = [(128, 4, (1, 96, 96)), (64, 2, (1, 80, 80)), (192, 2, (1, 70, 70)), (128, 4, (1, 300)), (256, 4, (1, 40, 40))]


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize(("d", "n", "lead"), SHAPES, ids=[f"d{s[0]}n{s[1]}x{'x'.join(map(str, s[2]))}" for s in SHAPES])
def test_fp32_matches_fp64_on_tf32(monkeypatch, d, n, lead, training):
    m = _module(d, n, I.MINIWORLD)
    ref = _module(d, n, I.PYTORCH)
    g = torch.Generator(device=DEV).manual_seed(4)
    x = torch.randn(*lead, d, device=DEV, generator=g)
    dy = torch.randn(*lead, d, device=DEV, generator=g)
    calls = []
    original = residual_mod.transition_residual
    monkeypatch.setattr("miniworld_engine.kernels.transition.triton.residual.transition_residual",
                        lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    with tf32(True):
        xd = x.double().requires_grad_()
        gold, pg = _gold(m, xd)
        if training:
            g_gold = torch.autograd.grad(gold, [xd, *pg.values()], dy.double())
        want_ref = None
        with torch.enable_grad() if training else torch.no_grad():
            if training:
                out_ref, g_ref = _call(ref, x, dy)
                out, g_eng = _call(m, x, dy)
            else:
                with torch.no_grad():
                    out_ref, out = ref(x), m(x)
    del want_ref
    assert calls, "the portable residual path did not run"
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    assert rel(out, gold) <= max(2 * rel(out_ref, gold), 4e-3), (rel(out, gold), rel(out_ref, gold))
    if training:
        for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_ref, strict=True)):
            assert rel(a, r) <= max(2 * rel(c, r), 4e-3), (i, rel(a, r), rel(c, r))


def test_full_precision_fp32_keeps_the_pytorch_reference(monkeypatch):
    """With TF32 off the call is the PyTorch statements (cuBLAS fp32), as on H100: no portable kernel launches."""
    m = _module(128, 4, I.MINIWORLD)
    monkeypatch.setattr("miniworld_engine.kernels.transition.triton.residual.transition_residual",
                        lambda *a, **k: pytest.fail("the portable fp32 path ran with TF32 off"))
    x = torch.randn(1, 64, 64, 128, device=DEV)
    with tf32(False), torch.no_grad():
        out = m(x)
    ref = _module(128, 4, I.PYTORCH)
    with tf32(False), torch.no_grad():
        torch.testing.assert_close(out, ref(x))


def test_bf16_is_unchanged():
    m = _module(128, 4, I.MINIWORLD).eval()
    ref = _module(128, 4, I.PYTORCH).eval()
    x = torch.randn(1, 64, 64, 128, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        assert rel(m(x), ref(x)) < 2e-2
