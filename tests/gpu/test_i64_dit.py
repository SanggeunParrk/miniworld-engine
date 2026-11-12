"""64-bit address arithmetic in the DiT-side Triton kernels (adaLN, conditioned transition, RMSNorm / RMSNorm-adaMod, q/k-norm RoPE).

Their launchers set a WIDE constexpr (``kernels/_wide.needs_wide``) that makes the column / row index int64 before it meets an activation
stride. Below 2**31 elements WIDE is off and the kernels compile to the code they always were. No served size reaches the bound: these are
token / atom activations ([samples, A or L, d] with d <= 3072 and at most ~1e8 elements at the requirement sizes), so a tensor past 2**31
elements cannot be built for them on an 80 GB card in a meaningful configuration. The tests instead force WIDE on at a small size
(``needs_wide`` patched to True in each launcher module) and require bitwise-equal outputs and gradients against WIDE off: the 64-bit
addresses must read and write exactly the same elements.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

LAUNCHERS = (
    "miniworld_engine.kernels.adaln.triton.inference",
    "miniworld_engine.kernels.adaln.triton.training",
    "miniworld_engine.kernels.conditioned_transition.triton.composed",
    "miniworld_engine.kernels.conditioned_transition.triton.fwd_saveact",
    "miniworld_engine.kernels.conditioned_transition.triton.inference",
    "miniworld_engine.kernels.conditioned_transition.triton.training",
    "miniworld_engine.kernels.rmsnorm.triton.main",
    "miniworld_engine.kernels.rope.triton.qk_norm",
)


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


requires_card = pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")


def _run(fn: Callable[[list[torch.Tensor]], list[torch.Tensor]], inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """``fn`` on fresh leaf copies of ``inputs``; the outputs plus every leaf's gradient (backward of sum(out * fixed weights))."""
    leaves = [t.detach().clone().requires_grad_(t.requires_grad) for t in inputs]
    outs = fn(leaves)
    res = [o.detach().clone() for o in outs]
    if any(t.requires_grad for t in leaves):
        gen = torch.Generator(device="cuda").manual_seed(7)
        loss = sum((o.float() * torch.randn(o.shape, device="cuda", generator=gen)).sum() for o in outs)
        loss.backward()  # type: ignore[union-attr]
        res += [t.grad.detach().clone() for t in leaves if t.grad is not None]
    return res


def _same_with_wide(monkeypatch, fn, inputs) -> None:
    off = _run(fn, inputs)
    for name in LAUNCHERS:
        monkeypatch.setattr(importlib.import_module(name), "needs_wide", lambda *a: True)
    on = _run(fn, inputs)
    assert len(on) == len(off)
    for i, (a, b) in enumerate(zip(off, on, strict=True)):
        # bitwise, except the parameter gradients some kernels reduce with atomics (order-dependent fp32 sums: a few ulp)
        err = float((a.float() - b.float()).abs().max())
        assert torch.equal(a, b) or err <= 1e-6 * float(a.float().abs().max()), (
            f"output / grad {i}: WIDE on differs from WIDE off (max {err:.3e})"
        )


def _randn(*shape, dtype=torch.bfloat16, grad=False, scale=1.0):
    t = (torch.randn(*shape, device="cuda") * scale).to(dtype)
    return t.requires_grad_(grad)


@requires_card
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_wide(monkeypatch, dtype) -> None:
    from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm

    torch.manual_seed(0)
    x = _randn(3, 257, 128, dtype=dtype, grad=True)
    w = _randn(128, dtype=torch.float32, grad=True)
    _same_with_wide(monkeypatch, lambda t: [triton_rmsnorm(t[0], t[1], 1e-5)], [x, w])


@requires_card
def test_rmsnorm_adamod_wide(monkeypatch) -> None:
    from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm_adamod

    torch.manual_seed(1)
    q = _randn(2, 300, 128, grad=True)
    c = _randn(2, 300, 128, grad=True)
    ws, wsh, wg = (_randn(128, 128, scale=0.05, grad=True) for _ in range(3))
    _same_with_wide(monkeypatch, lambda t: list(triton_rmsnorm_adamod(*t, eps=1e-5)), [q, c, ws, wsh, wg])


@requires_card
@pytest.mark.parametrize(("nx", "dtype"), [(128, torch.bfloat16), (768, torch.bfloat16), (768, torch.float32)])
def test_adaln_inference_wide(monkeypatch, nx, dtype) -> None:
    """d <= 256: the fused kernel; wide d: LN(cond) + the GEMM + gate kernel (16-bit, TF32) or the materialize path (strict fp32)."""
    from miniworld_engine.kernels.adaln.triton.inference import adaln_inference

    torch.manual_seed(2)
    nc = 128 if nx == 128 else 384
    x, cond = _randn(2, 260, nx, dtype=dtype), _randn(2, 260, nc, dtype=dtype)
    lnw = 1 + 0.1 * torch.randn(nc, device="cuda")
    sw, bw = _randn(nx, nc, dtype=dtype, scale=0.05), _randn(nx, nc, dtype=dtype, scale=0.05)
    sb = _randn(nx, dtype=dtype)
    for tf32 in (True, False) if dtype == torch.float32 else (True,):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            with torch.no_grad():
                _same_with_wide(monkeypatch, lambda t: [adaln_inference(*t, 1e-5, 1e-5)], [x, cond, lnw, sw, sb, bw])  # pyright: ignore[reportCallIssue]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
            monkeypatch.undo()


@requires_card
@pytest.mark.parametrize("nx", [128, 768])
@pytest.mark.parametrize("mode", ["cublas", "fused"])
def test_adaln_train_wide(monkeypatch, nx, mode) -> None:
    from miniworld_engine.kernels.adaln.triton import training

    torch.manual_seed(3)
    nc = 128 if nx == 128 else 384
    x, cond = _randn(2, 260, nx, grad=True), _randn(2, 260, nc, grad=True)
    lnw = (1 + 0.1 * torch.randn(nc, device="cuda")).requires_grad_(True)
    sw, bw = _randn(nx, nc, scale=0.05, grad=True), _randn(nx, nc, scale=0.05, grad=True)
    sb = _randn(nx, grad=True)
    training.set_forward_mode(mode)
    try:
        _same_with_wide(monkeypatch, lambda t: [training.adaln_train(*t, 1e-5, 1e-5)], [x, cond, lnw, sw, sb, bw])  # pyright: ignore[reportCallIssue]
    finally:
        training.set_forward_mode("auto")


def _ct_inputs(d: int, dc: int, m: int, dtype=torch.bfloat16, grad=False):
    nd = 2 * d
    return [
        _randn(m, d, dtype=dtype, grad=grad),
        _randn(m, dc, dtype=dtype, grad=grad),
        _randn(nd, d, dtype=dtype, scale=0.05, grad=grad),
        _randn(nd, d, dtype=dtype, scale=0.05, grad=grad),
        _randn(d, nd, dtype=dtype, scale=0.05, grad=grad),
        _randn(d, dc, dtype=dtype, scale=0.05, grad=grad),
        _randn(d, dtype=dtype, grad=grad),
    ]


@requires_card
def test_conditioned_transition_inference_wide(monkeypatch) -> None:
    """The fused b2b kernel (atom width 128) and the composed two-kernel path (token width 768)."""
    from miniworld_engine.kernels.conditioned_transition.triton.composed import cond_transition_inference_composed
    from miniworld_engine.kernels.conditioned_transition.triton.inference import cond_transition_inference

    torch.manual_seed(4)
    with torch.no_grad():
        _same_with_wide(monkeypatch, lambda t: [cond_transition_inference(*t, length=300)], _ct_inputs(128, 128, 600))
        monkeypatch.undo()
        _same_with_wide(monkeypatch, lambda t: [cond_transition_inference_composed(*t, length=300)], _ct_inputs(768, 384, 600))


@requires_card
@pytest.mark.parametrize(("d", "dc"), [(128, 128), (768, 384)])
def test_conditioned_transition_train_wide(monkeypatch, d, dc) -> None:
    """The fused training forwards (b2b saveact at the atom width, the composed saveact pair at the token width) and the backward."""
    from miniworld_engine.kernels.conditioned_transition.triton import training

    torch.manual_seed(5)
    training.set_forward_mode("fused")
    try:
        _same_with_wide(monkeypatch, lambda t: [training.cond_transition_train(*t, length=300)], _ct_inputs(d, dc, 600, grad=True))
    finally:
        training.set_forward_mode("auto")


@requires_card
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_qk_norm_rope_wide(monkeypatch, dtype) -> None:
    from miniworld_engine.kernels.rope.triton.qk_norm import qk_norm_rope_3d

    torch.manual_seed(6)
    n, s, h, d, half = 2, 96, 4, 32, 12
    qkv = _randn(n, s, 3, h, d, dtype=dtype)
    ang = torch.rand(1, s, half, device="cuda") * 6.28
    cos, sin = ang.cos(), ang.sin()

    def fn(t):
        q, k = t[0][:, :, 0], t[0][:, :, 1]  # strided views of one qkv buffer, as the module hands them
        return list(qk_norm_rope_3d(q, k, cos, sin))

    _same_with_wide(monkeypatch, fn, [qkv.requires_grad_(True)])
