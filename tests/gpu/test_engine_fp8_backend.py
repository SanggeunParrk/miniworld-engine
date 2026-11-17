"""H100: the ``engine-fp8`` backend (engine 3.0.0). Selecting it on a module turns that module's FP8 route on (route call counts), within the
route's error; a module on ``cuda`` (the default engine) gives the same bytes before and after an ``engine-fp8`` call in the same process,
and every switch is back off after the call."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _fill(m: torch.nn.Module, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))


def _off() -> dict:
    from miniworld_engine import fp8

    return {k: v for k, v in fp8.status().items() if k != "guard"}


_OFF = {"triattn": {"pv": [], "qk": False}, "trimul": None, "opm": None, "pwa": False, "transition": None, "token_dit": [], "atom_dit": []}


def _case(kind: str):
    from miniworld_engine.modules.transition import Transition
    from miniworld_engine.modules.triangle import TriangleMultiplication

    n, d = 256, 128
    g = torch.Generator(device="cuda").manual_seed(1)
    if kind == "trimul":
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as route

        m = TriangleMultiplication(d, outgoing=True, master_weights=True, backend="cuda")
        x = torch.randn(1, n, n, d, device="cuda", generator=g).bfloat16()
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        mask[0, n - 20 :] = False
        args = (x, mask)
    else:
        from miniworld_engine.kernels.transition.cuda import fp8_sm90a as route

        m = Transition(d, 4, master_weights=True, backend="cuda")
        args = (torch.randn(1, n, n, d, device="cuda", generator=g).bfloat16(),)
    _fill(m, 7)
    return m.cuda().eval(), args, route


@pytest.mark.parametrize("kind", ["trimul", "transition"])
def test_engine_fp8_turns_the_route_on(kind: str) -> None:
    _needs_h100()
    m, args, route = _case(kind)
    assert route.POLICY is None and _off() == _OFF
    with torch.no_grad():
        y_cuda = m(*args)
        calls = route.GUARD_STATS["calls"]
        assert m.set_backend("engine-fp8").backend_name == "engine-fp8"
        y_fp8 = m(*args)
        fired = route.GUARD_STATS["calls"] - calls
        assert _off() == _OFF and route.POLICY is None  # restored after the call
        m.set_backend("cuda")
        calls = route.GUARD_STATS["calls"]
        y_cuda2 = m(*args)
        assert route.GUARD_STATS["calls"] == calls  # cuda: the route stays off
    assert fired >= 1, f"{kind}: engine-fp8 did not take the FP8 route"
    assert not torch.equal(y_fp8, y_cuda)
    assert _rel(y_fp8 - args[0], y_cuda - args[0]) < 0.06
    assert torch.equal(y_cuda2, y_cuda)  # the default engine: byte-identical after an engine-fp8 call


def test_engine_fp8_composite_and_default_bytes() -> None:
    """A Pairformer block selected as engine-fp8 runs its parts' FP8 routes; a block on cuda (same weights) in the same process gives the
    same bytes before and after it, and takes no FP8 route."""
    _needs_h100()
    from miniworld_engine import fp8
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as tr
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as tm
    from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig

    cfg = PairformerConfig(d_pair=128, model="af3")
    a = PairformerBlock(cfg, master_weights=True, backend="cuda")
    _fill(a, 3)
    a = a.cuda().eval()
    b = PairformerBlock(cfg, master_weights=True, backend="engine-fp8").cuda().eval()
    b.load_state_dict(a.state_dict())
    n = 256
    g = torch.Generator(device="cuda").manual_seed(2)
    pair = torch.randn(1, n, n, 128, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    fp8.set_model("af3")
    try:
        with torch.no_grad():
            y0 = a(pair, mask)
            c_tm, c_tr = tm.GUARD_STATS["calls"], tr.GUARD_STATS["calls"]
            y8 = b(pair, mask)
            assert tm.GUARD_STATS["calls"] > c_tm and tr.GUARD_STATS["calls"] > c_tr
            assert _off() == _OFF
            c_tm, c_tr = tm.GUARD_STATS["calls"], tr.GUARD_STATS["calls"]
            y1 = a(pair, mask)
            assert (tm.GUARD_STATS["calls"], tr.GUARD_STATS["calls"]) == (c_tm, c_tr)
    finally:
        fp8.set_model(None)
    assert torch.equal(y0, y1)
    assert not torch.equal(y8, y0) and torch.isfinite(y8.float()).all()
    assert _rel(y8 - pair, y0 - pair) < 0.1


def _train_sha1(m: torch.nn.Module, args, train: bool) -> str:
    import hashlib

    x = args[0].detach().clone().requires_grad_(True)
    m.train(train)
    m.zero_grad(set_to_none=True)
    y = m(x, *args[1:])
    g = torch.Generator(device="cuda").manual_seed(5)
    y.backward(torch.randn(y.shape, device="cuda", generator=g).to(y.dtype))
    h = hashlib.sha1()
    for t in (y, x.grad, *(p.grad for p in m.parameters() if p.grad is not None)):
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("train", [True, False], ids=["train_mode", "eval_grad_on"])
@pytest.mark.parametrize("kind", ["trimul", "transition"])
def test_engine_fp8_training_runs_the_default_path(kind: str, train: bool) -> None:
    """engine-fp8 accelerates inference only: a training call (train mode, or autograd on) gives the bytes of the cuda backend (output, input
    and parameter gradients) and takes no FP8 route."""
    _needs_h100()
    m, args, route = _case(kind)
    torch.manual_seed(0)
    ref = _train_sha1(m, args, train)
    calls = route.GUARD_STATS["calls"]
    m.set_backend("engine-fp8")
    torch.manual_seed(0)
    got = _train_sha1(m, args, train)
    assert route.GUARD_STATS["calls"] == calls and _off() == _OFF
    assert got == ref
