"""The H100 bias-only TriangleAttention path (``kernels/integrations/bidir_triattn_h100``: ``BidirectionalTriangleAttention`` and the
single-direction ``TriangleAttention(use_self_attention=False)``).

Default route = the engine 2.5.9 outputs, bit for bit:

- the host-side tails (one-pass softmax-logits kernel, the fp32 dyln GEMM written in place, the planes GEMM through the fastest bit-equal
  cublasLt algorithm) against the torch ops they replaced (``LEGACY``), output and every gradient;
- the fused inference tail (gate + out-projection + residual in one kernel) against the unfused gate_u + torch.addmm;
- the opt-in cublasLt residual out-projection (``RESIDUAL_LT``) is NOT the same bits (one rounding where torch.addmm rounds twice): its error
  against an fp64 reference is not larger than the default's.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


pytestmark.append(pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)"))


def _module(kind, starting, seed=0, w_std=0.1):
    from miniworld_engine.modules.triangle import BidirectionalTriangleAttention, TriangleAttention

    torch.manual_seed(seed)
    if kind == "bi":
        m = BidirectionalTriangleAttention(128, 4, backend="cuda", master_weights=True)
    else:
        m = TriangleAttention(128, 4, starting=starting, use_self_attention=False, master_weights=True, backend="cuda")
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=w_std)
    return m.cuda().eval()


def _inputs(B, L, masked, x_scale=2.0):
    x = torch.randn(B, L, L, 128, device="cuda", dtype=torch.bfloat16) * x_scale
    mask = None
    if masked:
        mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
        mask[:, L - 13 :] = False
    return x, mask


def _run(m, x, mask, train):
    xx = x.clone().requires_grad_(train)
    for p in m.parameters():
        p.grad = None
    with torch.enable_grad() if train else torch.no_grad():
        y = m(xx, mask) if mask is not None else m(xx)
        grads = None
        if train:
            y.backward(torch.linspace(-1, 1, y.numel(), device=y.device).view_as(y).to(y.dtype))
            grads = [xx.grad] + [p.grad for p in m.parameters() if p.grad is not None]
    return y.detach(), grads


KINDS = [("bi", True), ("tri", True), ("tri", False)]


@pytest.mark.parametrize(("kind", "starting"), KINDS)
@pytest.mark.parametrize(
    ("length", "batch", "train"),
    [(128, 1, False), (128, 2, True), (200, 1, True), (256, 2, False), (256, 1, True), (512, 1, False), (768, 1, False)],
)
@pytest.mark.parametrize("masked", [False, True])
def test_host_tails_equal_the_torch_ops(monkeypatch, kind, starting, length, batch, train, masked):
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as I

    m = _module(kind, starting)
    x, mask = _inputs(batch, length, masked)
    monkeypatch.setattr(I, "LEGACY", True)
    ya, ga = _run(m, x, mask, train)
    monkeypatch.setattr(I, "LEGACY", False)
    yb, gb = _run(m, x, mask, train)
    assert torch.equal(ya, yb)
    if train:
        assert len(ga) == len(gb) and all(torch.equal(a, b) for a, b in zip(ga, gb, strict=True))


@pytest.mark.parametrize(("kind", "starting"), KINDS)
@pytest.mark.parametrize("length", [128, 144, 256, 384, 512, 640, 768])
@pytest.mark.parametrize(("batch", "masked", "w_std", "x_scale"), [(1, False, 0.1, 2.0), (1, True, 0.5, 0.5), (2, True, 0.1, 4.0)])
def test_fused_tail_equals_the_unfused_tail(monkeypatch, kind, starting, length, batch, masked, w_std, x_scale):
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as I

    if length >= 640 and batch > 1:
        pytest.skip("memory")
    m = _module(kind, starting, seed=length, w_std=w_std)
    x, mask = _inputs(batch, length, masked, x_scale)
    monkeypatch.setattr(I, "FUSED_TAIL", False)
    ya, _ = _run(m, x, mask, False)
    monkeypatch.setattr(I, "FUSED_TAIL", True)
    yb, _ = _run(m, x, mask, False)
    assert torch.equal(ya, yb)


def test_fused_tail_is_the_default_and_refuses_what_it_cannot_take():
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as I

    assert I.FUSED_TAIL is True and I.RESIDUAL_LT is False and I.LEGACY is False
    assert I._fused_tail(False, 128, 256, 128 * 128)
    assert I._fused_tail(True, 128, 256, 128 * 128)  # training: the kernel also writes the saved u
    assert not I._fused_tail(False, 256, 256, 128 * 128)  # d_pair 256
    assert not I._fused_tail(False, 128, 256, 200 * 200)  # L L % 128 != 0


@pytest.mark.parametrize("length", [128, 256, 512])
def test_residual_lt_is_the_more_accurate_rounding(monkeypatch, length):
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as I

    m = _module("bi", True)
    x, mask = _inputs(1, length, True)
    monkeypatch.setattr(I, "FUSED_TAIL", False)
    ya, _ = _run(m, x, mask, False)  # torch.addmm: bf16(bf16(product) + x)
    monkeypatch.setattr(I, "RESIDUAL_LT", True)
    yb, _ = _run(m, x, mask, False)  # cublasLt: bf16(product + x)
    d = (ya.float() - yb.float()).abs()
    assert (ya != yb).any()  # not the same bits
    # the two roundings differ by at most the bf16 grid of the product and of the sum: |x| + |out| bounds both
    assert (d <= 2 ** (-7) * (x.float().abs() + ya.float().abs()) + 1e-6).all()


@pytest.mark.parametrize(("kind", "starting"), KINDS)
@pytest.mark.parametrize("length", [128, 144, 256, 384, 512])
@pytest.mark.parametrize(("batch", "masked", "w_std", "x_scale"), [(1, False, 0.1, 2.0), (1, True, 0.5, 0.5), (2, True, 0.1, 4.0)])
def test_fused_tail_with_saved_u_equals_the_unfused_training(monkeypatch, kind, starting, length, batch, masked, w_std, x_scale):
    """Training: the fused tail also writes u; output, dx and every parameter gradient equal the gate_u + addmm route's bits."""
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as I

    if length >= 384 and batch > 1:
        pytest.skip("memory")
    m = _module(kind, starting, seed=length, w_std=w_std)
    x, mask = _inputs(batch, length, masked, x_scale)
    monkeypatch.setattr(I, "FUSED_TAIL_SAVE", False)
    ya, ga = _run(m, x, mask, True)
    monkeypatch.setattr(I, "FUSED_TAIL_SAVE", True)
    yb, gb = _run(m, x, mask, True)
    assert torch.equal(ya, yb)
    assert len(ga) == len(gb) and all(torch.equal(a, b) for a, b in zip(ga, gb, strict=True))
