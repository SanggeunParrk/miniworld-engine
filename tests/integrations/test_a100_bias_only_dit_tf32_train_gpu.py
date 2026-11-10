"""The fp32 (TF32) training block of the bias-only token DiT on A100 (integrations/bias_only_dit_train.py with fp32 inputs and weights): the output and every gradient (single, cond, pair, all parameters) against an fp64 PyTorch
block, no worse than the PyTorch fp32 block with its GEMMs on TF32 tensor cores (1.5x + 1e-3: the B200 page's rule), the TF32 bias-gradient core against fp64 and bit-identical on reruns, CUDA-graph capture, what the gate declines."""

import contextlib
import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit_train as TR
from miniworld_engine.kernels.bias_only_dit.cuda import sm80
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.bias_only_dit import module as bo_module
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="A100 (sm_80)",
    ),
]
LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]
F64 = torch.float64


def relative(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


@contextlib.contextmanager
def tf32(on):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@pytest.fixture(autouse=True)
def policy():
    old = settings.configure(engine_backend="auto")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        settings.configure(**vars(old))


def blocks(seed=811, n_head=16, d_head=None):
    torch.manual_seed(seed)
    ref = randomize(
        BiasOnlyDiTBlock(
            n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH
        )
    ).cuda()
    fast = BiasOnlyDiTBlock(
        n_head=n_head, d_head=d_head, implementation=ImplementationType.MINIWORLD
    ).cuda()
    fast.load_state_dict(ref.state_dict())
    return ref, fast, copy.deepcopy(ref).double()


@contextlib.contextmanager
def no_module_path():
    def boom(*a, **k):
        raise AssertionError("the module's PyTorch composition ran")

    old = bo_module.BiasOnlyAttention.forward
    bo_module.BiasOnlyAttention.forward = boom
    try:
        yield
    finally:
        bo_module.BiasOnlyAttention.forward = old


def inputs(L, A, masked, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(A, 1, L, 768, device="cuda")
    c = torch.randn(A, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    dy = torch.randn(A, 1, L, 768, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    return x, c, p, dy, mask


def step(m, x, c, p, mask, dy, dtype):
    leaves = [t.detach().to(dtype).requires_grad_(True) for t in (x, c, p)]
    m.zero_grad(set_to_none=True)
    y = m(*leaves, mask)
    y.backward(dy.to(dtype))
    return [y.detach()] + [t.grad for t in leaves] + [q.grad for q in m.parameters()]


SHAPES = [(128, 8, False), (256, 16, True), (384, 6, True), (640, 4, False), (768, 3, True)]


@pytest.mark.parametrize(("L", "A", "masked"), SHAPES)
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_every_gradient_within_the_pytorch_tf32_error(L, A, masked, n_head, d_head):
    ref, fast, ref64 = blocks(seed=3, n_head=n_head, d_head=d_head)
    x, c, p, dy, mask = inputs(L, A, masked)
    xr, cr, pr = (t.clone().requires_grad_(True) for t in (x, c, p))
    assert TR.serves(fast, xr, cr, pr, mask)
    want = step(ref64, x, c, p, mask, dy, F64)
    with no_module_path():
        got = step(fast, x, c, p, mask, dy, torch.float32)
    with tf32(True):
        base = step(ref, x, c, p, mask, dy, torch.float32)
    names = ["out", "d single", "d cond", "d pair"] + [
        n for n, _ in fast.named_parameters()
    ]
    for n, g, b, w in zip(names, got, base, want, strict=True):
        assert g.dtype is torch.float32, n
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3, (
            n,
            relative(g, w),
            relative(b, w),
        )


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 256, 384, 640, 768])
def test_dpb32_matches_fp64_and_repeats(L, n_head, d_head):
    dev_A, H, DH = 6, n_head, d_head
    DA, M = H * DH, dev_A * L
    dpb = sm80.Dpb32(H, DH)
    for seed in range(3):
        torch.manual_seed(1000 * L + seed)
        P = torch.softmax(2 * torch.randn(H * L, L, device="cuda"), -1)
        do = torch.randn(M, DA, device="cuda")
        v = torch.randn(M, 2 * DA, device="cuda")[:, :DA]
        dd = torch.randn(dev_A, H, L, device="cuda")
        outs = []
        for _ in range(3):
            out = torch.empty(H * L, L, device="cuda")
            dpb(do, v, P, dd, out, dev_A)
            outs.append(out)
        assert all(torch.equal(outs[0], o) for o in outs[1:]), seed
        want = P.double().view(H, L, L) * (
            torch.einsum(
                "aihd,ajhd->hij",
                do.double().view(dev_A, L, H, DH),
                v.double().reshape(dev_A, L, H, DH),
            )
            - dd.double().sum(0)[:, :, None]
        )
        assert relative(outs[0].view(H, L, L), want) < 3e-3, seed


def test_graph_capture_replays_the_step():
    _, fast, _ = blocks(seed=7)
    L, A = 384, 8
    x, c, p, dy, _ = inputs(L, A, False, seed=1)
    xs, cs, ps = (t.clone().requires_grad_(True) for t in (x, c, p))

    def run():
        for t in (xs, cs, ps):
            t.grad = None
        for q in fast.parameters():
            if q.grad is not None:
                q.grad.zero_()
        fast(xs, cs, ps).backward(dy)
        return [t.grad.clone() for t in (xs, cs, ps)] + [
            q.grad.clone() for q in fast.parameters()
        ]

    eager = run()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    for t in (xs, cs, ps):
        t.grad = None
    for q in fast.parameters():
        q.grad = torch.zeros_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast(xs, cs, ps).backward(dy)
    for t in (xs, cs, ps):
        t.grad.zero_()
    for q in fast.parameters():
        q.grad.zero_()
    graph.replay()
    torch.cuda.synchronize()
    replay = [t.grad for t in (xs, cs, ps)] + [q.grad for q in fast.parameters()]
    for e, r in zip(eager, replay, strict=True):
        assert relative(r, e) < 1e-4


def test_what_the_gate_declines():
    _, fast, _ = blocks(seed=5)
    x, c, p, _, _ = inputs(128, 2, False)
    xr, cr, pr = (t.clone().requires_grad_(True) for t in (x, c, p))
    assert TR.serves(fast, xr, cr, pr)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert not TR.serves(fast, xr, cr, pr)
    with torch.no_grad():
        assert not TR.serves(fast, xr, cr, pr)
    x, c, p, _, _ = inputs(1024, 2, False)
    assert not TR.serves(fast, x.requires_grad_(True), c, p)  # L > 768
