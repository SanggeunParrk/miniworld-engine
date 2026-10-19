"""H100 TriMul beyond L1024 and at ragged L: the D64 / D128 CUDA paths (and the wide B200-style port) at the lengths MiniWorld's
inference rungs reach (1536 / 2048 / 2560) and at lengths that are not a multiple of 128.

Contract (``integrations.trimul_h100``): inference any L >= 16 (one sample pads to 16 inside K1 / K3; B samples at L % 16 != 0
are padded on the host), training L a multiple of 128 up to ``MAX_TRAINING_LENGTH``. Every case runs the module as MiniWorld
does (FP32 parameters, a bf16 pair, a token mask, row dropout in training) against the FP32 PyTorch statement of the module,
every gradient included. The reference is computed in row chunks (checkpointed in training) so that L2560 fits next to the
kernels on one 80 GB card."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from miniworld_engine.integrations import trimul_h100
from miniworld_engine.kernels.trimul_inproj.cuda import (
    h100_b200style,
    h100_d64_b200gate,
    h100_inference,
    h100_single,
    h100_training,
    h100_uni_d64_b200gate,
)
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
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
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


def counting(monkeypatch, module, name):
    calls = []
    fn = getattr(module, name)

    def wrapped(*a, **k):
        calls.append(1)
        return fn(*a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return calls


def native_entry(monkeypatch, D, direction, training):
    """The CUDA entry point a served call must reach (not the PyTorch statements)."""
    if not training:
        return counting(monkeypatch, h100_inference, "inference")
    if D == 64:
        return (counting(monkeypatch, h100_d64_b200gate, "bidirectional_trimul") if direction == 0 else
                counting(monkeypatch, h100_uni_d64_b200gate, "single_trimul"))
    if D == 128:
        return (counting(monkeypatch, h100_training, "bidirectional_trimul") if direction == 0 else
                counting(monkeypatch, h100_single, "single_trimul"))
    return counting(monkeypatch, h100_b200style, "trimul")


# --------------------------------------------------------------------------- contract
def test_contract():
    x = lambda B, L, D=64: torch.empty(B, L, L, D, device="cuda", dtype=torch.bfloat16)
    for D in (64, 128):
        for direction in (0, 1, 2):
            m = make(D, direction).eval()
            with torch.no_grad():
                for L in (48, 200, 520, 1040, 1536, 2048, 2560, 3000):
                    assert trimul_h100.serves_inference(m, x(1, L, D), bidirectional=direction == 0), (D, direction, L)
            m.train()
            for L in (1152, 1536, 2048, 2560, 3072, 4096):
                assert trimul_h100.serves(m, x(1, L, D)), (D, direction, L)
            for L in (1040, 1100, 3000):
                assert not trimul_h100.serves(m, x(1, L, D)), (D, direction, L)
    m = make(64, 0)
    assert trimul_h100.serves(m, x(4, 2560))
    assert not trimul_h100.serves(m, x(4, 1040))   # batched training: L % 128
    with torch.no_grad():
        m.eval()
        assert trimul_h100.serves_inference(m, x(4, 520), bidirectional=True)   # padded on the host
        assert trimul_h100.serves_inference(m, x(4, 2560), bidirectional=True)
    # the length bound (B200's 80 x 128) and the int32 token index B L^2 < 2^31, without allocating such pairs
    assert trimul_h100._training_length(10240)
    assert not trimul_h100._training_length(10368)
    big = lambda B, L: torch.empty(1, 1, 1, 1).expand(B, L, L, 64)
    assert trimul_h100._index_ok(big(20, 10240))
    assert not trimul_h100._index_ok(big(21, 10240))
    assert trimul_h100._index_ok(big(1, 46336))
    assert not trimul_h100._index_ok(big(1, 46352))


# --------------------------------------------------------------------------- inference
INFER = ([(D, B, d, L) for D, B in ((64, 1), (64, 4), (128, 1)) for d in (0, 1, 2) for L in (200, 1040, 2560)]
         + [(D, B, 0, L) for D, B in ((64, 1), (64, 4), (128, 1)) for L in (48, 520, 1536, 2048)])
INFER_IDS = [f"D{D}-B{B}-dir{d}-L{L}" for D, B, d, L in INFER]


@pytest.mark.parametrize("case", INFER, ids=INFER_IDS)
def test_inference(case, monkeypatch):
    D, B, direction, L = case
    m = make(D, direction).eval()
    x, mask, _ = inputs(B, L, D)
    calls = native_entry(monkeypatch, D, direction, False)
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=direction == 0)
        got = m(x, mask)
        assert calls, "the CUDA path was not reached"
        assert got.shape == x.shape
        assert got.is_contiguous()
        assert torch.isfinite(got.float()).all()
        assert torch.equal(got, m(x, mask))                      # bitwise run to run
        err = relative(got, reference(m, direction, x, mask, None))
        assert err < 1e-2, err
        got0 = m(x, None)
        err0 = relative(got0, reference(m, direction, x, None, None))
        assert err0 < 1e-2, err0
        if B > 1:                                                # one call = B calls of one sample
            want = torch.cat([m(x[i:i + 1], mask[i:i + 1]) for i in range(B)], 0)
            assert relative(got, want) < 1e-3, relative(got, want)


@pytest.mark.parametrize("case", [(64, 4, 0, 520), (64, 4, 0, 1536), (128, 1, 0, 2048), (128, 1, 1, 200)],
                         ids=["D64-B4-L520", "D64-B4-L1536", "D128-L2048", "D128-out-L200"])
def test_inference_graph_replay(case):
    D, B, direction, L = case
    m = make(D, direction).eval()
    x, mask, _ = inputs(B, L, D)
    with torch.no_grad():
        m(x, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = m(x, mask)
        x.normal_()
        mask.copy_(torch.rand_like(mask, dtype=torch.float) > 0.3)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, m(x, mask))


# --------------------------------------------------------------------------- training
def _train(m, x, mask, ds, dy, monkeypatch):
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    xx = x.detach().clone().requires_grad_()
    params = list(m.parameters())
    y = m(xx, mask)
    grads = torch.autograd.grad(y, [xx, *params], dy)
    return y.detach(), [g.detach().clone() for g in grads]


TRAIN = ([(D, 1, d, L) for D in (64, 128) for d in (0, 1, 2) for L in (1152, 1536, 2048, 2560)]
         + [(64, 4, d, 1152) for d in (0, 1, 2)]
         # beyond the rungs: the FP32 reference runs on a second GPU (``_reference_device``)
         + [(64, 1, 1, 4096), (64, 1, 0, 4096), (128, 1, 2, 3584), (128, 1, 0, 2816)])
TRAIN_IDS = [f"D{D}-B{B}-dir{d}-L{L}" for D, B, d, L in TRAIN]


def _reference_device(L):
    """Above L2560 the FP32 reference runs on a second GPU (it needs up to ~70 GB at D128 L2816 / 3584)."""
    if L <= 2560:
        return torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.device_count() < 2:
        pytest.skip("the FP32 reference beyond L2560 needs a second GPU")
    return torch.device("cuda", 1 - torch.cuda.current_device() if torch.cuda.current_device() < 2 else 0)


@pytest.mark.parametrize("case", TRAIN, ids=TRAIN_IDS)
def test_training(case, monkeypatch):
    D, B, direction, L = case
    m = make(D, direction).train()
    x, mask, ds = inputs(B, L, D)
    dy = torch.randn_like(x)
    calls = native_entry(monkeypatch, D, direction, True)
    assert trimul_h100.serves(m, x)
    y, grads = _train(m, x, mask, ds, dy, monkeypatch)
    assert calls, "the CUDA path was not reached"
    y2, grads2 = _train(m, x, mask, ds, dy, monkeypatch)        # bitwise run to run
    assert torch.equal(y, y2)
    assert all(torch.equal(a, b) for a, b in zip(grads, grads2, strict=True))
    del y2, grads2
    torch.cuda.empty_cache()
    dev = _reference_device(L)
    mr = m if dev == x.device else copy.deepcopy(m).to(dev)
    xr = x.detach().to(dev).requires_grad_()
    z = reference(mr, direction, xr, mask.to(dev), ds.to(dev))
    want = torch.autograd.grad(z, [xr, *mr.parameters()], dy.to(dev).float())
    errors = [relative(y, z)] + [relative(a, b) for a, b in zip(grads, want, strict=True)]
    assert errors[0] < 5e-3, errors
    assert max(errors[1:]) < 2e-2, errors


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_training_batch_l2560(direction, monkeypatch):
    """B = 4 at L2560: one call against four calls of one sample (each checked against the reference in ``test_training``)."""
    D, B, L = 64, 4, 2560
    m = make(D, direction).train()
    x, mask, ds = inputs(B, L, D)
    dy = torch.randn_like(x)
    y_b, g_b = _train(m, x, mask, ds, dy, monkeypatch)
    ys, gs = [], []
    for i in range(B):
        yi, gi = _train(m, x[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1], monkeypatch)
        ys.append(yi)
        gs.append(gi)
    assert relative(y_b, torch.cat(ys, 0)) < 1e-3
    assert relative(g_b[0], torch.cat([g[0] for g in gs], 0)) < 2e-3
    for k in range(1, len(g_b)):
        want = sum(g[k].float() for g in gs)
        assert relative(g_b[k], want) < 2e-3, (k, relative(g_b[k], want))


@pytest.mark.parametrize("case", [(64, 4, 0, 1536), (64, 1, 1, 2048), (128, 1, 0, 1536), (128, 1, 2, 1152)],
                         ids=["D64-B4-bidir-L1536", "D64-out-L2048", "D128-bidir-L1536", "D128-in-L1152"])
def test_training_graph_replay(case, monkeypatch):
    """A captured forward + backward replays with live inputs, masks, dropout scales and weights."""
    D, B, direction, L = case
    m = make(D, direction).train()
    x, mask, ds = inputs(B, L, D)
    dy = torch.randn_like(x)
    x = x.requires_grad_()
    monkeypatch.setattr(m, "_make_drop_row_scale", lambda pair, p: ds)
    params = list(m.parameters())

    def step():
        y = m(x, mask)
        return y, torch.autograd.grad(y, [x, *params], dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y, grads = step()
    with torch.no_grad():
        x.normal_()
        params[0].mul_(0.95)
        mask.copy_(torch.rand_like(mask, dtype=torch.float) > 0.3)
        ds.copy_((torch.rand_like(ds, dtype=torch.float) > 0.25).to(ds.dtype) / 0.75)
        dy.normal_()
    graph.replay()
    torch.cuda.synchronize()
    want_y, want_g = step()
    assert torch.equal(y, want_y)
    assert all(torch.equal(a, b) for a, b in zip(grads, want_g, strict=True))


# --------------------------------------------------------------------------- wide widths beyond L1024 (the B200-style port)
WIDE = [(256, d, L) for d in (0, 1, 2) for L in (1152, 1536)] + [(384, 0, 1152), (384, 1, 1152), (512, 0, 1152), (512, 2, 1152)]
WIDE_IDS = [f"D{D}-dir{d}-L{L}" for D, d, L in WIDE]


@pytest.mark.parametrize("case", WIDE, ids=WIDE_IDS)
def test_wide_inference(case, monkeypatch):
    D, direction, L = case
    m = make(D, direction).eval()
    x, mask, _ = inputs(1, L, D)
    with torch.no_grad():
        assert trimul_h100.serves_inference(m, x, bidirectional=direction == 0)
        got = m(x, mask)
        assert torch.isfinite(got.float()).all()
        assert torch.equal(got, m(x, mask))
        err = relative(got, reference(m, direction, x, mask, None))
        assert err < 1e-2, err


@pytest.mark.parametrize("case", WIDE, ids=WIDE_IDS)
def test_wide_training(case, monkeypatch):
    D, direction, L = case
    m = make(D, direction).train()
    x, mask, ds = inputs(1, L, D)
    dy = torch.randn_like(x)
    calls = native_entry(monkeypatch, D, direction, True)
    assert trimul_h100.serves(m, x)
    y, grads = _train(m, x, mask, ds, dy, monkeypatch)
    assert calls, "the CUDA path was not reached"
    y2, grads2 = _train(m, x, mask, ds, dy, monkeypatch)
    assert torch.equal(y, y2)
    # The port's backward sums its LayerNorm / gate partials with global atomics (b200_sources/wide_bwd.cu, as at L <= 1024):
    # its gradients agree run to run up to the FP32 summation order, not bitwise.
    assert max(relative(a, b) for a, b in zip(grads, grads2, strict=True)) < 1e-3
    del y2, grads2
    torch.cuda.empty_cache()
    xr = x.detach().clone().requires_grad_()
    params = list(m.parameters())
    z = reference(m, direction, xr, mask, ds)
    want = torch.autograd.grad(z, [xr, *params], dy.float())
    errors = [relative(y, z)] + [relative(a, b) for a, b in zip(grads, want, strict=True)]
    assert errors[0] < 5e-3, errors
    assert max(errors[1:]) < 2e-2, errors


def test_wide_port_determinism_l512(monkeypatch):
    """The existing property at L <= 1024 (D256 L512 runs the same port): y bitwise, gradients up to the summation order."""
    m = make(256, 0).train()
    x, mask, ds = inputs(1, 512, 256)
    dy = torch.randn_like(x)
    y, g = _train(m, x, mask, ds, dy, monkeypatch)
    y2, g2 = _train(m, x, mask, ds, dy, monkeypatch)
    assert torch.equal(y, y2)
    assert max(relative(a, b) for a, b in zip(g, g2, strict=True)) < 1e-3
    print("D256 L512 bitwise-equal gradients:", [torch.equal(a, b) for a, b in zip(g, g2, strict=True)])
