"""Smooth lDDT portable kernels (``kernels/smooth_lddt/triton``) on any card: value and coordinate gradient against the definition in
float64 and (on H100 / B200) against the CUDA kernels, masks and radii, B > 1, ragged N, bf16 input, bitwise determinism, CUDA-graph replay."""
from __future__ import annotations

import contextlib

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.smooth_lddt import smooth_lddt

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
DEV = "cuda"


@contextlib.contextmanager
def portable_kernels():
    """The one place the A/B switch is named: the portable kernels on this card, H100 / B200 included."""
    previous = settings.configure(force_portable_kernels=True)
    try:
        yield
    finally:
        settings.configure(force_portable_kernels=previous.force_portable_kernels)


def _case(a, b, n, seed=0, frac_nuc=0.2, frac_valid=0.9, noise=1.5, scale=1.5):
    g = torch.Generator().manual_seed(seed)
    gt = (torch.cumsum(torch.randn(b, n, 3, generator=g), dim=1) * scale).to(DEV)
    nuc = (torch.rand(b, n, generator=g) < frac_nuc).to(DEV)
    mask = (torch.rand(b, n, generator=g) < frac_valid).to(DEV)
    x = (gt.unsqueeze(0) + torch.randn(a, b, n, 3, generator=g).to(DEV) * noise).contiguous()
    return x, gt, nuc, mask


def _dense64(x, gt, nuc, mask):
    """AF3 Algorithm 27 in float64, the definition (autograd-differentiable): mean over (a, b) of 1 - lddt."""
    x, g = x.double(), gt.double()
    d_pred = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
    d_gt = torch.cdist(g, g, compute_mode="donot_use_mm_for_euclid_dist")
    diff = (d_pred - d_gt).abs()
    score = sum(torch.sigmoid(t - diff) for t in (0.5, 1.0, 2.0, 4.0)) / 4
    cut = torch.where(nuc, 30.0, 15.0).double().unsqueeze(-1)
    n = gt.shape[-2]
    keep = (d_gt < cut) & mask.unsqueeze(-1) & mask.unsqueeze(-2) & ~torch.eye(n, dtype=torch.bool, device=gt.device)
    return (1.0 - (score * keep).sum((-1, -2)) / keep.sum((-1, -2)).clamp(min=1)).mean()


def _run(fn, x, *rest):
    xk = x.clone().requires_grad_()
    loss = fn(xk, *rest)
    loss.backward()
    return loss.detach().double(), xk.grad


def _rel(a, e):
    return float((a.double() - e.double()).norm() / e.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize(("a", "b", "n", "kw"), [
    (3, 2, 200, {}), (4, 1, 500, {}), (2, 2, 1024, {}), (8, 1, 1000, {"frac_nuc": 0.0}), (8, 1, 1000, {"frac_nuc": 1.0}),
    (48, 1, 1000, {}), (2, 1, 3000, {}), (3, 3, 130, {"frac_valid": 0.3}), (2, 1, 777, {"scale": 0.4}), (2, 1, 777, {"scale": 6.0}),
    (2, 2, 2048, {"frac_valid": 1.0, "noise": 0.2}), (1, 1, 64, {}), (4, 1, 300, {"noise": 10.0}),
])
def test_value_and_gradient_match_the_definition_in_float64(a, b, n, kw):
    x, gt, nuc, mask = _case(a, b, n, **kw)
    with portable_kernels():
        lk, gk = _run(smooth_lddt, x, gt, nuc, mask)
    lr, gr = _run(_dense64, x.double(), gt, nuc, mask)
    assert abs(float(lk) - float(lr)) < 5e-7
    # the pair score has a kink at d_pred = d_gt (|.|): an fp32 distance that lands on the other side of it flips one pair's term, so against
    # float64 the gradient agrees to ~1e-4 (the CUDA kernels and the fp32 PyTorch reference show the same); the tight check is against the CUDA kernels below
    assert _rel(gk, gr) < 3e-4


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (9, 0)), reason="the CUDA kernels serve H100 / B200")
@pytest.mark.parametrize(("a", "b", "n"), [(3, 2, 200), (48, 1, 1000), (2, 1, 3000)])
def test_the_portable_kernels_agree_with_the_cuda_kernels(a, b, n):
    x, gt, nuc, mask = _case(a, b, n)
    lc, gc = _run(smooth_lddt, x, gt, nuc, mask)
    with portable_kernels():
        lp, gp = _run(smooth_lddt, x, gt, nuc, mask)
    assert abs(float(lc) - float(lp)) < 5e-7
    assert _rel(gp, gc) < 2e-6


def test_nothing_resolved_gives_unit_loss_and_zero_gradient():
    x, gt, nuc, mask = _case(2, 1, 256)
    mask[:] = False
    with portable_kernels():
        loss, grad = _run(smooth_lddt, x, gt, nuc, mask)
    assert float(loss) == pytest.approx(1.0)
    assert float(grad.abs().max()) == 0.0


def test_bf16_input_gets_a_bf16_gradient_of_the_fp32_value():
    x, gt, nuc, mask = _case(3, 1, 384)
    xb = x.to(torch.bfloat16).requires_grad_()
    with portable_kernels():
        loss = smooth_lddt(xb, gt, nuc, mask)
        loss.backward()
    ref = _dense64(xb.detach(), gt, nuc, mask)
    assert xb.grad.dtype == torch.bfloat16
    assert abs(float(loss) - float(ref)) < 5e-7


def test_value_and_gradient_are_bitwise_deterministic():
    x, gt, nuc, mask = _case(8, 2, 640)
    with portable_kernels():
        runs = [_run(smooth_lddt, x, gt, nuc, mask) for _ in range(4)]
    assert all(torch.equal(runs[0][0], r[0]) and torch.equal(runs[0][1], r[1]) for r in runs[1:])


def test_a_cuda_graph_replays_value_and_gradient():
    x, gt, nuc, mask = _case(6, 1, 512)
    with portable_kernels():
        xk = x.clone().requires_grad_()
        for _ in range(2):                                  # warm-up: the autotuner and the compiles run outside the capture
            xk.grad = None
            smooth_lddt(xk, gt, nuc, mask).backward()
        xk.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = smooth_lddt(xk, gt, nuc, mask)
            loss.backward()
        graph.replay()
        torch.cuda.synchronize()
    lr, gr = _run(_dense64, x.double(), gt, nuc, mask)
    assert abs(float(loss) - float(lr)) < 5e-7
    assert _rel(xk.grad, gr) < 3e-4


def test_the_registry_checkers_pass_at_their_declared_bands():
    from paths import registry_rows

    from miniworld_engine.autotune.run_all import (
        _running_dtype,
        check_one,
        declared_rtol,
    )

    rows = [r for r in registry_rows() if r["family"] == "smooth_lddt"]
    assert {r["kernel"] for r in rows} == {"smooth_lddt_prep_triton", "smooth_lddt_fwd_triton", "smooth_lddt_bwd_triton"}
    for r in rows:
        ok, detail = check_one(r["check"], declared_rtol(r, _running_dtype()))
        assert ok, f"{r['kernel']}: {detail}"
