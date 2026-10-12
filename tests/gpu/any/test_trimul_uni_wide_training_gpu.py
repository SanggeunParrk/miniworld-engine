"""Single-direction TriMul D256/D384 native training (h100_uni_wide_training): numerics, graphs, compile."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.trimul_inproj.cuda import h100_uni_wide_training as H

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
NAMES = ["y-x", "dx", "dWl", "dWlg", "dWr", "dWrg", "dWg", "dWp", "dgi", "dbi", "dgo", "dbo"]


def _hopper():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")


def _leaves(D, n, seed, requires_grad=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    x = r(1, n, n, D).to("cuda", torch.bfloat16)
    w = [(r(D, D) * D**-0.5).to("cuda", torch.bfloat16) for _ in range(6)]
    ln = [(1 + 0.1 * r(D)), 0.1 * r(D), (1 + 0.1 * r(D)), 0.1 * r(D)]
    leaves = [x, *w, *[t.to("cuda") for t in ln]]
    return [t.requires_grad_(requires_grad) for t in leaves]


def _reference(x, w, ln, pair_mask, ds, outgoing):
    """fp32 copy of TriangleMultiplication (PYTORCH) with a fixed drop_row scale."""
    D = x.shape[-1]
    wl, wlg, wr, wrg, wg, wp = w
    gi, bi, go, bo = ln
    z = F.layer_norm(x, (D,), gi, bi, 1e-5)
    m = pair_mask[None, :, :, None]
    left = torch.sigmoid(z @ wlg.t()) * (z @ wl.t()) * m
    right = torch.sigmoid(z @ wrg.t()) * (z @ wr.t()) * m
    eq = "bikd,bjkd->bijd" if outgoing else "bkid,bkjd->bijd"
    o = F.layer_norm(torch.einsum(eq, left, right), (D,), go, bo, 1e-5)
    return x + torch.sigmoid(z @ wg.t()) * (o @ wp.t()) * ds[None, None]


def _rel(a, b):
    a, b = a.detach().float(), b.detach().float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _inputs(D, n, masked, dropout, seed=0):
    torch.manual_seed(seed)
    if masked:
        keep = torch.ones(n, dtype=torch.bool, device="cuda")
        keep[n - n // 7:] = False
        keep[5] = False
        pm = (keep[:, None] & keep[None, :]).to(torch.bfloat16)
    else:
        pm = torch.ones(n, n, device="cuda", dtype=torch.bfloat16)
    if dropout:
        ds = (torch.rand(n, D, device="cuda") > 0.25).to(torch.bfloat16) / 0.75
    else:
        ds = torch.ones(n, D, device="cuda", dtype=torch.bfloat16)
    return pm, ds


@pytest.mark.parametrize("D", [256, 384])
@pytest.mark.parametrize("n", [384, 768])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(("masked", "dropout"), [(False, False), (True, True)])
def test_matches_fp32_reference(D, n, outgoing, masked, dropout):
    _hopper()
    leaves = _leaves(D, n, 3)
    pm, ds = _inputs(D, n, masked, dropout)
    dy = torch.randn_like(leaves[0])
    y = H.single_trimul(outgoing, *leaves, pm, ds)
    grads = torch.autograd.grad(y, leaves, dy)
    ref_leaves = [t.detach().float().requires_grad_() for t in leaves]
    yr = _reference(ref_leaves[0], ref_leaves[1:7], ref_leaves[7:], pm.float(), ds.float(), outgoing)
    gr = torch.autograd.grad(yr, ref_leaves, dy.float())
    x = leaves[0].detach().float()
    errors = [_rel(y.float() - x, yr - x)] + [_rel(a, b) for a, b in zip(grads, gr, strict=True)]
    for name, e in zip(NAMES, errors, strict=True):
        assert e < 1.2e-2, (name, e)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(g.dtype == t.dtype and g.shape == t.shape for g, t in zip(grads, leaves, strict=True))


@pytest.mark.parametrize("D", [256, 384])
def test_graph_replay_follows_changed_inputs_and_weights(D):
    _hopper()
    n = 384
    static = _leaves(D, n, 5)
    pm, ds = _inputs(D, n, True, True)
    dy = torch.randn_like(static[0])

    def step(args):
        y = H.single_trimul(True, *args, pm, ds)
        return [y.detach(), *torch.autograd.grad(y, args, dy)]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step(static)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(static)
    for seed in (11, 12):
        fresh = _leaves(D, n, seed)
        with torch.no_grad():
            for dst, src in zip(static, fresh, strict=True):
                dst.copy_(src)
        graph.replay()
        expected = step(fresh)
        for i, (a, b) in enumerate(zip(out, expected, strict=True)):
            if b.dtype == torch.float32:
                # LN affine sums are atomic across CTAs (order not fixed).
                assert _rel(a, b) < 1e-5, i
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("D", [256, 384])
def test_independent_forwards_and_nograd_match(D):
    _hopper()
    n = 384
    leaves = _leaves(D, n, 7)
    pm, ds = _inputs(D, n, True, True)
    dy = torch.randn_like(leaves[0])
    a = H.single_trimul(False, *leaves, pm, ds)
    b = H.single_trimul(False, *leaves, pm, ds)
    ga = torch.autograd.grad(a, leaves, dy)
    gb = torch.autograd.grad(b, leaves, dy)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for u, v in zip(ga, gb, strict=True):
        assert _rel(u, v) < 1e-5
    with torch.no_grad():
        c = H.single_trimul(False, *leaves, pm, ds)
    torch.testing.assert_close(c, a.detach(), rtol=0, atol=0)


def test_compile_fullgraph_matches_eager():
    _hopper()
    D, n = 256, 384
    leaves = _leaves(D, n, 9)
    pm = torch.ones(n, n, device="cuda", dtype=torch.bfloat16)
    _, ds = _inputs(D, n, False, True)
    dy = torch.randn_like(leaves[0])
    y = H.single_trimul(True, *leaves, pm, ds)
    grads = torch.autograd.grad(y, leaves, dy)
    compiled = torch.compile(H.single_trimul, fullgraph=True, options={"triton.cudagraphs": False})
    z = compiled(True, *leaves, pm, ds)
    actual = torch.autograd.grad(z, leaves, dy)
    for a, b in zip((z, *actual), (y, *grads), strict=True):
        assert _rel(a, b) < 2e-6


@pytest.mark.parametrize("D", [256, 384])
@pytest.mark.parametrize("outgoing", [True, False])
def test_module_wiring_matches_pytorch_reference(monkeypatch, D, outgoing):
    """TriangleMultiplication(D256/384) through integrations.trimul_h100.update matches its PYTORCH implementation."""
    _hopper()
    from miniworld_engine import settings
    from miniworld_engine.integrations import trimul_h100
    from miniworld_engine.modules.exceptions import ImplementationType as I
    from miniworld_engine.modules.triangle_multiplication.module import (
        TriangleMultiplication,
    )

    real = H.single_trimul
    calls = []

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(H, "single_trimul", spy)
    previous = settings.configure(engine_backend="auto",
                                  trimul_h100_training_widths=trimul_h100.TRAINING_WIDTHS)
    try:
        n = 384
        torch.manual_seed(21)
        m = TriangleMultiplication(D, outgoing=outgoing, implementation=I.MINIWORLD, p_drop=0.0).cuda().train()
        with torch.no_grad():
            for name, p in m.named_parameters():
                if p.ndim == 1:
                    p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p))
                else:
                    p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
                    p.data = p.data.to(torch.bfloat16)
        ref = TriangleMultiplication(D, outgoing=outgoing, implementation=I.PYTORCH, p_drop=0.0).cuda().train()
        with torch.no_grad():
            for p, q in zip(ref.parameters(), m.parameters(), strict=True):
                p.copy_(q.float())
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        mask[:, n - 40:] = False
        x = torch.randn(1, n, n, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)
        y = m(x, mask)
        grads = torch.autograd.grad(y, [x, *m.parameters()], dy)
        assert calls, "the single-direction wide native path was not taken"
        xr = x.detach().float().requires_grad_()
        yr = ref(xr, mask)
        gr = torch.autograd.grad(yr, [xr, *ref.parameters()], dy.float())
        xf = x.detach().float()
        assert _rel(y.float() - xf, yr - xf) < 1.2e-2
        for a, b in zip(grads, gr, strict=True):
            assert a.shape == b.shape
            assert _rel(a, b) < 1.2e-2
    finally:
        settings.configure(**{f: getattr(previous, f) for f in ("engine_backend", "trimul_h100_training_widths")})
