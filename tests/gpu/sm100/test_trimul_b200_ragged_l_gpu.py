"""B200 TriMul inference at L % 16 != 0: padded to 16 on the host (integrations.trimul_b200._padded_inference; H100's
62acfe721), against the FP32 PyTorch statement of the module (helpers from the H100 long-L test)."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from miniworld_engine.integrations import trimul_b200
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def hopper():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    torch.compiler.reset()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False   # the reference is FP32
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.cuda.empty_cache()


def relative(a, b):
    return float((a.detach().float().to(b.device) - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def make(D, direction, seed=0):
    torch.manual_seed(seed)
    m = (BidirectionalTriangleMultiplication(D, implementation=ImplementationType.MINIWORLD, p_drop=0.25) if direction == 0 else
         TriangleMultiplication(D, outgoing=direction == 1, implementation=ImplementationType.MINIWORLD, p_drop=0.25)).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def inputs(B, L, D, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15
    ds = (torch.rand(B, 1, L, D, device="cuda", generator=g) > 0.25).to(torch.bfloat16) / 0.75
    return x, mask, ds


def _front(x, mrow, mcol, gi, bi, wl, wlg, wr, wrg):
    """Rows of x [B, r, L, D] -> gated left / right planes, channel-major [B, H, r, L] (FP32)."""
    xn = F.layer_norm(x.float(), (x.shape[-1],), gi, bi, 1e-5)
    pm = (mrow[:, :, None] & mcol[:, None, :])[..., None].float()
    left = torch.sigmoid(F.linear(xn, wlg)) * F.linear(xn, wl) * pm
    right = torch.sigmoid(F.linear(xn, wrg)) * F.linear(xn, wr) * pm
    return left.permute(0, 3, 1, 2).contiguous(), right.permute(0, 3, 1, 2).contiguous()


def _back(t, x, ds, gi, bi, go, bo, wg, wp):
    """Contraction rows t [B, H, r, L] and the pair rows x [B, r, L, D] -> the residual output rows (FP32)."""
    t = t.permute(0, 2, 3, 1)
    o = F.layer_norm(t, (t.shape[-1],), go, bo, 1e-5)
    xn = F.layer_norm(x.float(), (x.shape[-1],), gi, bi, 1e-5)
    upd = F.linear(o, wp) * torch.sigmoid(F.linear(xn, wg))
    return x.float() + (upd if ds is None else upd * ds.float())


def reference(m, direction, x, mask, ds, chunk=256):
    """FP32 statement of the module (the PyTorch reference of the residual update), in row chunks; differentiable."""
    p = dict(m.named_parameters())
    B, n, _, D = x.shape
    grad = torch.is_grad_enabled()
    run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if grad else (lambda f, *a: f(*a))
    mask = torch.ones(B, n, device=x.device, dtype=torch.bool) if mask is None else mask
    H = p["to_left.weight"].shape[0]
    left = torch.empty((B, H, n, n), device=x.device, dtype=torch.float32)
    right = torch.empty_like(left)
    for r in range(0, n, chunk):
        lc, rc = run(_front, x[:, r:r + chunk], mask[:, r:r + chunk], mask, p["ln_pair.weight"], p["ln_pair.bias"],
                     p["to_left.weight"], p["to_left_gate.weight"], p["to_right.weight"], p["to_right_gate.weight"])
        left[:, :, r:r + chunk] = lc
        right[:, :, r:r + chunk] = rc
        del lc, rc
    h = H // 2 if direction == 0 else H
    out_ = lambda a, b: torch.matmul(a, b.transpose(-1, -2))     # outgoing: t[i, j] = sum_k a[i, k] b[j, k]
    in_ = lambda a, b: torch.matmul(a.transpose(-1, -2), b)      # incoming: t[i, j] = sum_k a[k, i] b[k, j]
    if grad and direction == 0:                                  # the two halves written in place (no concatenated copy)
        tri = torch.empty_like(left)
        tri[:, :h] = out_(left[:, :h], right[:, :h])
        tri[:, h:] = in_(left[:, h:], right[:, h:])
    elif grad:
        tri = (out_ if direction == 1 else in_)(left, right)
    else:                                                        # no autograd: no concatenated copy
        tri = torch.empty_like(left)
        for b in range(B):
            if direction in (0, 1):
                tri[b, :h] = out_(left[b, :h], right[b, :h])
            if direction in (0, 2):
                tri[b, H - h:] = in_(left[b, H - h:], right[b, H - h:])
    del left, right
    y = torch.empty((B, n, n, D), device=x.device, dtype=torch.float32)
    for r in range(0, n, chunk):
        y[:, r:r + chunk] = run(_back, tri[:, :, r:r + chunk], x[:, r:r + chunk], ds, p["ln_pair.weight"], p["ln_pair.bias"],
                                p["ln_out.weight"], p["ln_out.bias"], p["to_gate.weight"], p["to_out.weight"])
    return y


RAGGED = [(D, B, d, L) for D, B in ((64, 1), (64, 3), (128, 1), (128, 3)) for d in (0, 1, 2) for L in (100, 203)]


@pytest.mark.parametrize("case", RAGGED, ids=[f"D{D}-B{B}-dir{d}-L{L}" for D, B, d, L in RAGGED])
def test_ragged_inference(case, monkeypatch):
    D, B, direction, L = case
    m = make(D, direction).eval()
    x, mask, _ = inputs(B, L, D)
    calls = []
    real = trimul_b200._padded_inference

    def wrapped(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(trimul_b200, "_padded_inference", wrapped)
    with torch.no_grad():
        assert trimul_b200.serves_inference(m, x, bidirectional=direction == 0)
        got = m(x, mask)
        assert calls, "the padded CUDA path was not reached"
        assert got.shape == x.shape and got.is_contiguous()
        assert torch.isfinite(got.float()).all()
        assert torch.equal(got, m(x, mask))
        err = relative(got, reference(m, direction, x, mask, None))
        assert err < 1e-2, err
        err0 = relative(m(x, None), reference(m, direction, x, None, None))
        assert err0 < 1e-2, err0
