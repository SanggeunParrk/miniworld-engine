"""The slid key window's TAIL on heavily padded atom inputs: every query window past the last real one (padding queries only) reads, in the
reference, the slid key window ``[R_rule - 128, R_rule)`` (``functional.slide_starts``; ``R_rule``: the real count, or rounded up to a whole
query window for ``slide_qblock``). ``cuda`` (H100 kernels/integrations/local_dit_h100, A100 kernels/dit_variants) recomputes those windows
on the tail kernels of ``dit_variants/cuda/edge.cu`` (window count read on the device). Compared with the ``pytorch`` reference at EVERY
position: the block output, and in training d single / d cond / d pair and every parameter gradient, with an upstream gradient on all rows
(padding rows included: callers read the residual stream there) and masked to the real rows (FoldForge's pattern). Also the ``triton``
backend (its window kernel reads the slid starts of every window)."""

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import atom_dit as AD

pytestmark = [pytest.mark.gpu]

#: (class, key_window): every block-local atom class on both slid windows
CASES = [
    (cls, kw)
    for cls in ("AtomDiTProtenix", "AtomDiTOpenFold3", "AtomDiTBoltz2", "AtomDiTRoseTTAFold3", "AtomDiTChai1")
    for kw in ("slide", "slide_qblock")
]
#: (N, real atoms): many padding windows; under 128 real atoms (the window begins at 0, keys past R masked); no padding window (control)
SIZES = [(1024, 700), (2048, 1536), (256, 100), (1024, 1000)]
TOL = {"float32": 5e-3, "master": 2e-2}  # ||ours - ref|| / ||ref||, the registry's bounds


def _backends() -> list[str]:
    return ["cuda", "triton"] if torch.cuda.is_available() else []


def _module(name: str, key_window: str, dtype: str, train: bool):
    m = getattr(AD, name)(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, key_window=key_window).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    return typing.cast("AD._LocalAtomDiT", C.cast(m, dtype).train(train))


def _inputs(m, n: int, real: int, dtype: str, samples: int) -> dict:
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=n, samples=samples, depth=None, dtype=act, device="cuda", generator=g)
    inp["mask"] = (torch.arange(n, device="cuda") < real)[None]
    return inp


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    den = ref.double().norm().item()
    return (got.double() - ref.double()).norm().item() / den if den > 0 else (got.double().norm().item())


def _run(m, backend: str, inp: dict, dtype: str, dy: torch.Tensor | None):
    """(output, {name: gradient}) of ``backend``; gradients when ``dy`` is given."""
    m.set_backend(backend)
    m.zero_grad(set_to_none=True)
    args = {k: v.detach().clone() for k, v in inp.items()}
    if dy is not None:
        for k in ("single", "cond", "pair"):
            args[k] = args[k].contiguous().requires_grad_(True)
    ctx = torch.enable_grad() if dy is not None else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master"):
        y = m(**args)
    grads = {}
    if dy is not None:
        (y.float() * dy).sum().backward()
        grads = {k: args[k].grad.float() for k in ("single", "cond", "pair")}
        grads.update({f"param {k}": p.grad.float() for k, p in m.named_parameters() if p.grad is not None})
    return y.detach().float(), grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(("name", "key_window"), CASES)
@pytest.mark.parametrize("dtype", ["float32", "master"])
def test_padded_inference_matches_reference(backend: str, name: str, key_window: str, dtype: str) -> None:
    m = _module(name, key_window, dtype, train=False)
    for n, real in SIZES:
        inp = _inputs(m, n, real, dtype, samples=5)
        m.set_backend(backend)
        why = m.unsupported(**inp)
        if why is not None:
            pytest.skip(f"{backend}: {why}")
        ref, _ = _run(m, "pytorch", inp, dtype, None)
        out, _ = _run(m, backend, inp, dtype, None)
        tail = (real - 1) // 32 * 32 + 32  # the first row of the first window past the last real one
        tol = TOL[dtype]
        assert _rel(out, ref) < tol, (n, real, "all rows", _rel(out, ref))
        assert _rel(out[..., :real, :], ref[..., :real, :]) < tol, (n, real, "real rows")
        if tail < n:
            assert _rel(out[..., tail:, :], ref[..., tail:, :]) < tol, (n, real, "tail rows", _rel(out[..., tail:, :], ref[..., tail:, :]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(("name", "key_window"), CASES)
@pytest.mark.parametrize("dtype", ["float32", "master"])
@pytest.mark.parametrize("masked_dy", [False, True], ids=["dy_all_rows", "dy_real_rows"])
def test_padded_training_matches_reference(backend: str, name: str, key_window: str, dtype: str, masked_dy: bool) -> None:
    m = _module(name, key_window, dtype, train=True)
    for n, real in SIZES[:3]:
        inp = _inputs(m, n, real, dtype, samples=2)
        m.set_backend(backend)
        if m.unsupported(**inp) is not None:
            pytest.skip(f"{backend}: {m.unsupported(**inp)}")
        g = torch.Generator(device="cuda").manual_seed(2)
        dy = torch.randn(2, 1, n, 128, device="cuda", generator=g)
        if masked_dy:
            dy = dy * inp["mask"][None, :, :, None]
        ref, gref = _run(m, "pytorch", inp, dtype, dy)
        out, got = _run(m, backend, inp, dtype, dy)
        tol = TOL[dtype]
        assert _rel(out, ref) < tol, (n, real, "output", _rel(out, ref))
        for k, r in gref.items():
            if r.abs().max() == 0:
                continue
            assert k in got, (n, real, k, "no gradient")
            assert _rel(got[k], r) < tol, (n, real, k, _rel(got[k], r))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("key_window", ["slide", "slide_qblock"])
def test_padded_training_is_deterministic(key_window: str) -> None:
    """Two training calls on the same padded inputs give bitwise the same output and gradients (the tail kernels reduce in fixed order)."""
    m = _module("AtomDiTOpenFold3", key_window, "master", train=True)
    inp = _inputs(m, 1024, 700, "master", samples=2)
    m.set_backend("cuda")
    if m.unsupported(**inp) is not None:
        pytest.skip(f"cuda: {m.unsupported(**inp)}")
    dy = torch.randn(2, 1, 1024, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    a = _run(m, "cuda", inp, "master", dy)
    b = _run(m, "cuda", inp, "master", dy)
    assert torch.equal(a[0], b[0])
    for k in a[1]:
        assert torch.equal(a[1][k], b[1][k]), k
