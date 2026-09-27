"""Preparation reuse must preserve live values and per-forward ownership."""

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as H
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("width", [64, 256, 384, 512])
@pytest.mark.parametrize("batched_mask", [False, True])
def test_packing_once_and_fp32_mask_owned(width, batched_mask, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    n, d = 384, width
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = [
        (torch.randn(h, k, device="cuda", dtype=x.dtype) * k**-0.5).requires_grad_()
        for h, k in [(2 * d, d)] * 4 + [(d, d), (d, 2 * d)]
    ]
    ln = [
        (
            torch.ones(c, device="cuda")
            if i % 2 == 0
            else torch.zeros(c, device="cuda")
        ).requires_grad_()
        for i, c in enumerate((d, d, 2 * d, 2 * d))
    ]
    # Already-FP32 callers must not make a custom-op output alias an input.
    mask = torch.ones(n, n, device="cuda", dtype=torch.float32)
    if batched_mask:
        mask = mask.unsqueeze(0)
    ds = torch.ones(n, d, device="cuda", dtype=x.dtype)
    pack, norm = W.pack_into, W.normalize_into
    calls = [0, 0]

    def pk(*a):
        calls[0] += 1
        return pack(*a)

    def nm(*a):
        calls[1] += 1
        return norm(*a)

    monkeypatch.setattr(W, "pack_into", pk)
    monkeypatch.setattr(W, "normalize_into", nm)
    args = (x, *w, *ln)
    y = H.bidirectional_trimul(*args, mask, ds)
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y, args, dy)
    assert calls == [1, int(d == 512)]
    assert all(torch.isfinite(g).all() for g in grads)

    monkeypatch.setattr(W, "pack_into", pack)
    monkeypatch.setattr(W, "normalize_into", norm)
    compiled = torch.compile(
        H.bidirectional_trimul, fullgraph=True, options={"triton.cudagraphs": False}
    )
    z = compiled(*args, mask, ds)
    actual = torch.autograd.grad(z, args, dy)
    for a, b in zip((z, *actual), (y, *grads), strict=True):
        error = (a.detach().float() - b.detach().float()).norm()
        assert error / b.detach().float().norm().clamp_min(1e-12) < 2e-6
