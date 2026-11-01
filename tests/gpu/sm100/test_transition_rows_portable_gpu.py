"""The B200 row-fused portable Transition (``kernels/transition/triton/rows.py``) against an fp64 reference and the split portable path.

It is what ``settings.force_portable_kernels`` runs on sm_100 at d = 64 / 128 (bf16): the error of every output and gradient must be no
worse than the split path's (the backward keeps d_xn in fp32 where the split path rounds it to bf16, so it is usually better), on ragged
row counts and both expansion factors.
"""
import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition import Transition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="B200 (sm_100) only")

NAMES = ("y", "dx", "dgamma", "dbeta", "dWa", "dWb", "dWs")


def _run(m, x, dy, train, use_rows, monkeypatch):
    import miniworld_engine.kernels.transition.triton.rows as rows

    with monkeypatch.context() as mp:
        if not use_rows:
            mp.setattr(rows, "serves", lambda *a, **k: False)
        params = [m.ln_in.weight, m.ln_in.bias, m.expand_a.weight, m.expand_b.weight, m.squeeze.weight]
        for p in params:
            p.grad = None
        xx = x.clone().requires_grad_(train)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if train:
                y = m(xx)
            else:
                with torch.no_grad():
                    y = m(xx)
        if not train:
            return [y.double()]
        y.backward(dy)
        return [y.double(), xx.grad.double()] + [p.grad.double() for p in params]


def _launched_rows(m, x, monkeypatch):
    import miniworld_engine.kernels.transition.triton.rows as rows

    calls = []
    real = rows.transition_rows
    with monkeypatch.context() as mp:
        mp.setattr(rows, "transition_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            m(x)
    return bool(calls)


@pytest.mark.parametrize("d,n,shape", [(128, 4, (1, 47, 48)), (64, 4, (1, 33, 40)), (128, 2, (2, 20, 24)), (64, 4, (1, 64, 64))])
@pytest.mark.parametrize("train", [False, True])
def test_rows_portable_matches_reference(d, n, shape, train, monkeypatch):
    settings.reset()
    settings.configure(force_portable_kernels=True)
    try:
        torch.manual_seed(0)
        m = Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().float()
        with torch.no_grad():
            m.squeeze.weight.normal_(std=(d * n) ** -0.5)
            m.ln_in.weight.uniform_(0.5, 1.5)
            m.ln_in.bias.uniform_(-0.2, 0.2)
        m.train(train)
        ref = Transition(d, n=n, implementation=ImplementationType.PYTORCH).cuda().double()
        ref.load_state_dict(m.state_dict())
        x = torch.randn(*shape, d, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        assert _launched_rows(m, x, monkeypatch), "the B200 portable path did not take the row-fused kernels"
        xr = x.double().requires_grad_(train)
        yr = ref._torch_forward(xr) + xr
        refs = [yr.detach()]
        if train:
            yr.backward(dy.double())
            refs += [xr.grad] + [p.grad for p in (ref.ln_in.weight, ref.ln_in.bias, ref.expand_a.weight, ref.expand_b.weight, ref.squeeze.weight)]
        got = _run(m, x, dy, train, True, monkeypatch)
        split = _run(m, x, dy, train, False, monkeypatch)
        for name, g, s, r in zip(NAMES, got, split, refs):
            e_rows = ((g - r).norm() / r.norm()).item()
            e_split = ((s - r).norm() / r.norm()).item()
            assert e_rows <= 1.1 * e_split + 1e-4, f"{name}: rows {e_rows:.3e} vs split {e_split:.3e}"
            assert e_rows < 1e-2, f"{name}: {e_rows:.3e}"
    finally:
        settings.reset()


def test_rows_fp32_inference_matches_fp64():
    """fp32 (TF32) inference at hidden = 4 d: the row-fused forward against fp64, no worse than the split path."""
    import miniworld_engine.kernels.transition.triton.rows as rows

    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    settings.reset()
    settings.configure(force_portable_kernels=True)
    try:
        for d, shape in ((128, (1, 37, 40)), (64, (1, 50, 50))):
            torch.manual_seed(0)
            m = Transition(d, n=4, implementation=ImplementationType.MINIWORLD).cuda().float().eval()
            with torch.no_grad():
                m.squeeze.weight.normal_(std=(d * 4) ** -0.5)
                m.ln_in.weight.uniform_(0.5, 1.5)
                m.ln_in.bias.uniform_(-0.2, 0.2)
            ref = Transition(d, n=4, implementation=ImplementationType.PYTORCH).cuda().double()
            ref.load_state_dict(m.state_dict())
            x = torch.randn(*shape, d, device="cuda")
            with torch.no_grad():
                yr = ref._torch_forward(x.double())
                calls = []
                real = rows.transition_rows
                rows.transition_rows = lambda *a, **k: calls.append(1) or real(*a, **k)
                try:
                    yrow = m(x).double() - x.double()
                finally:
                    rows.transition_rows = real
                assert calls, "fp32 inference did not take the row-fused forward"
                orig = rows.serves
                rows.serves = lambda *a, **k: False
                try:
                    ysplit = m(x).double() - x.double()
                finally:
                    rows.serves = orig
            e_rows, e_split = ((yrow - yr).norm() / yr.norm()).item(), ((ysplit - yr).norm() / yr.norm()).item()
            assert e_rows <= 1.1 * e_split + 1e-6 and e_rows < 1e-3, (d, e_rows, e_split)
    finally:
        settings.reset()
        torch.backends.cuda.matmul.allow_tf32 = old


def test_rows_serves_gate():
    import miniworld_engine.kernels.transition.triton.rows as rows
    from miniworld_engine.kernels.transition.triton.rows import serves

    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    wa = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    ws = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    assert serves(x, wa, ws, True) and serves(x, wa, ws, False)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        assert serves(x.float(), wa.float(), ws.float(), False)            # fp32 (TF32) inference at hidden = 4 d
        assert not serves(x.float(), wa.float(), ws.float(), True)         # fp32 training keeps the split path
        assert not serves(x.float(), wa[:256].float(), ws[:, :256].float(), False)   # hidden = 2 d, few rows: the split path
        big = torch.empty(rows.FP32_N2_MIN_ROWS, 128, device="cuda")
        assert serves(big, wa[:256].float(), ws[:, :256].float(), False)   # hidden = 2 d from FP32_N2_MIN_ROWS rows
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    assert not serves(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16), wa.new_empty(1024, 256), ws.new_empty(256, 1024), True)
    assert not serves(x, wa[:480], ws[:, :480], True)                       # hidden width not a multiple of 64
