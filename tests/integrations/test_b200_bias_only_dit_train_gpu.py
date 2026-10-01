"""The fused bias-only token DiT TRAINING path on B200 (integrations/bias_only_dit_train.py): the output and every gradient
against an fp32 PyTorch block, within the PyTorch bf16 block's own error; the attention's backward kernels against einsum;
CUDA graph capture, torch.compile, steady memory over steps, and what it declines."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit_train as TR
from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-30))


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


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        settings.configure(**vars(old))


def blocks(seed=3, n_head=16, d_head=None):
    torch.manual_seed(seed)
    ref = randomize(BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH)).cuda()
    fast = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.MINIWORLD).cuda()
    fast.load_state_dict(ref.state_dict())
    ref_bf = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH).cuda()
    ref_bf.load_state_dict(ref.state_dict())
    return ref, fast.to(torch.bfloat16), ref_bf.to(torch.bfloat16)


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


# L / A pick every branch: pv v tiles of 128 (A < 16), 256 (L256) and 192 (L384, L768); dpb key tiles of 128 and 256 (L768);
# every head layout (16 x 48, 24 x 32, 12 x 64, 16 x 64)
SHAPES = [(128, 8, False), (256, 16, True), (384, 48, True), (640, 4, False), (768, 16, True)]
LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]


@pytest.mark.parametrize(("L", "A", "masked"), SHAPES)
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_every_gradient_within_the_pytorch_bf16_error(L, A, masked, n_head, d_head):
    ref, fast, ref_bf = blocks(n_head=n_head, d_head=d_head)
    x, c, p, dy, mask = inputs(L, A, masked)
    xb, cb, pb = (t.bfloat16().requires_grad_(True) for t in (x, c, p))
    assert TR.serves(fast, xb, cb, pb, mask)
    want = step(ref, x, c, p, mask, dy, torch.float32)
    got = step(fast, x, c, p, mask, dy, torch.bfloat16)
    base = step(ref_bf, x, c, p, mask, dy, torch.bfloat16)
    names = ["out", "d single", "d cond", "d pair"] + [n for n, _ in fast.named_parameters()]
    # measured: ours / torch-bf16 error 0.96 - 0.99 over every gradient
    for n, g, b, w in zip(names, got, base, want, strict=True):
        assert relative(g, w) <= 1.1 * relative(b, w) + 1e-4, n


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 640, 768])
def test_attention_backward_kernels_match_einsum(L, n_head, d_head):
    A, H, DH = 6, n_head, d_head
    DA = H * DH
    M = A * L
    torch.manual_seed(L)
    do = torch.randn(M, DA, device="cuda").bfloat16()
    v = torch.randn(M, 2 * DA, device="cuda").bfloat16()[:, :DA]
    dd = torch.randn(A, H, L, device="cuda")
    # the training softmax writes P^T beside P: the same values, transposed
    bias = 2 * torch.randn(H * L, L, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(L, device="cuda") > 0.2
    P, Pt, P_rows = (torch.empty(H, L, L, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    C.softmax_t(bias, P.view(H * L, L), Pt.view(H * L, L), mask)
    C.softmax_rows(bias, P_rows.view(H * L, L), mask)
    assert torch.equal(P, P_rows)
    assert torch.equal(Pt.transpose(1, 2), P)
    dv = torch.empty(M, 2 * DA, device="cuda", dtype=torch.bfloat16)[:, :DA]
    C.PvGateCore(torch.cuda.current_device(), nh=H, dh=DH)(do, Pt.view(H * L, L), dv, A)
    dh, vh = do.float().view(A, L, H, DH), v.float().reshape(A, L, H, DH)
    assert relative(dv, torch.einsum("hij,aihd->ajhd", P.float(), dh).reshape(M, DA)) < 3e-3
    db = torch.empty(H * L, L, device="cuda", dtype=torch.bfloat16)
    C.DpbKernel(torch.cuda.current_device(), nh=H, dh=DH)(do, v, P.view(H * L, L), dd, db, A)
    want = P.float() * (torch.einsum("aihd,ajhd->hij", dh, vh) - dd.sum(0)[:, :, None])
    assert relative(db.view(H, L, L), want) < 3e-3


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_graph_capture_compile_and_steady_memory(n_head, d_head):
    _, fast, _ = blocks(seed=7, n_head=n_head, d_head=d_head)
    L, A = 384, 8
    x, c, p, dy, _ = inputs(L, A, False, seed=1)
    x, c, p, dy = (t.bfloat16() for t in (x, c, p, dy))
    xs, cs, ps = (t.clone().requires_grad_(True) for t in (x, c, p))

    def run(m):
        for t in (xs, cs, ps):
            t.grad = None
        m.zero_grad(set_to_none=False)
        for q in m.parameters():
            if q.grad is not None:
                q.grad.zero_()
        m(xs, cs, ps).backward(dy)
        return [t.grad.clone() for t in (xs, cs, ps)] + [q.grad.clone() for q in m.parameters()]

    eager = run(fast)
    # the bound launches reuse their argument blocks and the activations are allocated afresh each step: nothing may pile up
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    for _ in range(4):
        run(fast)
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved
    # a captured step replays to the eager gradients: not bit-exact (atomics and cuBLAS under capture may sum in another order,
    # so a bf16 gradient can round the other way here and there: measured 1.2e-4); a wrong launch would be O(1)
    for q in fast.parameters():
        q.grad = torch.zeros_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run(fast)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for t in (xs, cs, ps):
            t.grad = None
        for q in fast.parameters():
            q.grad.zero_()
        fast(xs, cs, ps).backward(dy)
    graph.replay()
    torch.cuda.synchronize()
    replayed = [t.grad for t in (xs, cs, ps)] + [q.grad for q in fast.parameters()]
    for g, e in zip(replayed, eager, strict=True):
        assert relative(g, e) < 2e-3
    # torch.compile keeps the two opaque ops (forward / backward) and gives the same step
    compiled = torch.compile(fast, options={"triton.cudagraphs": False})
    for g, e in zip(run(compiled), eager, strict=True):
        assert relative(g, e) < 2e-3


def test_declines_what_it_does_not_serve():
    _, fast, _ = blocks()
    x = torch.randn(4, 1, 384, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn(4, 1, 384, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    assert TR.serves(fast, x, c, p)
    assert not TR.serves(fast, x.float(), c.float(), p.float())                                  # bf16 only
    assert not TR.serves(fast, x[:, :, :200], c[:, :, :200], p[:, :200, :200])                    # L % 128
    assert not TR.serves(fast, x, c[:1], p)                                                       # a conditioning per sample
    assert not TR.serves(fast, x, c, p, torch.ones(4, 384, device="cuda", dtype=torch.bool))      # a mask per sample
    assert TR.serves(blocks(n_head=24)[1], x, c, p)                                               # 24 heads x 32
    assert TR.serves(blocks(n_head=12)[1], x, c, p)                                               # 12 x 64
    assert TR.serves(blocks(n_head=16, d_head=64)[1], x, c, p)                                    # 16 x 64
    assert not TR.serves(blocks(n_head=8)[1], x, c, p)                                            # 8 x 96: no kernels
    with torch.no_grad():
        assert not TR.serves(fast, x, c, p)                                                       # inference: the other path
