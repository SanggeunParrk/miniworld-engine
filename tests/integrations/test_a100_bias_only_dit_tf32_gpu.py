"""The fp32 (TF32) path of the bias-only token DiT on A100 (integrations/bias_only_dit.py with fp32 inputs: cuBLAS TF32 GEMMs, the family's fp32 rows and the TF32 ``mma.sync`` core of
kernels/bias_only_dit/cuda/sm80): the block's output against an fp64 reference, the core against its fp64 reference, eager / torch.compile / CUDA graph, that the TF32 core ran (and no Triton), what it declines.

Tolerance rationale (the B200 page's): a TF32 operand keeps 10 mantissa bits (2^-11 ~ 4.9e-4), so every product of the block carries O(1e-3) relative error whoever computes it. The bar is the PyTorch fp32
module with its GEMMs on the same TF32 tensor cores: the fused path is no worse than that against fp64 within a factor 1.5 plus a floor of 1e-3 (the core rounds its operands to nearest tf32, as cuBLAS does)."""

import contextlib
import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit as INF
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
    """(the PyTorch fp32 block, the engine's fp32 block, the PyTorch block in fp64), same weights."""
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


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events()]


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


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "S"), [(128, 1), (384, 5), (256, 3), (1024, 2)])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_inference_within_the_pytorch_tf32_error(L, S, shared, masked, n_head, d_head):
    ref, fast, ref64 = blocks(n_head=n_head, d_head=d_head)
    for m in (ref, fast, ref64):
        m.eval()
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = (
        torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384)
        if shared
        else torch.randn(S, 1, L, 384, device="cuda")
    )
    p = torch.randn(1, L, L, 128, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    with torch.no_grad():
        want = ref64(x.double(), c.double(), p.double(), mask)
        assert INF.serves(fast, x, c, p, mask)
        with no_module_path():
            got = fast(x, c, p, mask)
        with tf32(True):
            base = ref(x, c, p, mask)
    assert got.dtype is torch.float32
    for g, b, w in ((got, base, want), (got - x, base - x, want - x.double())):
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3
        assert relative(g, w) < 1e-2


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "S"), [(128, 1), (384, 5), (768, 3), (1024, 2)])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("sg", [None, 1, 2, 3, 4])
def test_core_against_fp64(L, S, gated, n_head, d_head, sg):
    torch.manual_seed(3)
    W = n_head * d_head
    core = sm80.PvGateTF32(n_head, d_head)
    vg = torch.randn(S * L, 2 * W, device="cuda")
    P = torch.softmax(2 * torch.randn(n_head * L, L, device="cuda"), -1)
    out = torch.empty(S * L, W, device="cuda")
    core(vg[:, :W], P, out, S, g=vg[:, W:] if gated else None, sg=sg)
    v = vg[:, :W].double().view(S, L, n_head, d_head)
    ref = torch.einsum("hij,ajhd->aihd", P.double().view(n_head, L, L), v)
    if gated:
        ref = ref * torch.sigmoid(vg[:, W:].double().view(S, L, n_head, d_head))
    assert relative(out, ref.reshape(S * L, W)) < 1e-3


def test_graph_compile_and_kernels():
    _, fast, _ = blocks(seed=5)
    fast.eval()
    L, S = 256, 5
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    with torch.no_grad():
        eager = fast(x, c, p)
        names = kernel_names(lambda: fast(x, c, p))
        assert any("pv_gate_tf32_kernel" in n for n in names), names
        assert not any("triton" in n.lower() for n in names)
        compiled = torch.compile(
            fast, fullgraph=True, options={"triton.cudagraphs": False}
        )
        assert relative(compiled(x, c, p), eager) < 1e-5
        xs, cs, ps = x.clone(), c.clone(), p.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fast(xs, cs, ps)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fast(xs, cs, ps)
        graph.replay()
        torch.cuda.synchronize()
        assert relative(out, eager) < 1e-5


def test_caches_follow_live_weights_and_pair():
    _, fast, ref64 = blocks(seed=5)
    fast.eval()
    ref64.eval()
    L = 256
    x = torch.randn(5, 1, L, 768, device="cuda")
    c = torch.randn(5, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    with torch.no_grad():
        got = fast(x, c, p)
        fast.transition.squeeze.weight.mul_(0.8)
        ref64.transition.squeeze.weight.mul_(0.8)
        assert not torch.equal(got, fast(x, c, p))
        assert relative(fast(x, c, p), ref64(x.double(), c.double(), p.double())) < 1e-2
        p.add_(0.5 * torch.randn_like(p))
        assert relative(fast(x, c, p), ref64(x.double(), c.double(), p.double())) < 1e-2


def test_what_the_gate_declines():
    _, fast, _ = blocks(seed=5)
    fast.eval()
    L = 256
    x = torch.randn(2, 1, L, 768, device="cuda")
    c = torch.randn(2, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    with torch.no_grad():
        assert INF.serves(fast, x, c, p)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not INF.serves(fast, x, c, p)
        with settings_off():
            assert not INF.serves(fast, x, c, p)
    assert not INF.serves(fast, x, c, p)  # autograd on: the training block's gate


@contextlib.contextmanager
def settings_off():
    old = settings.configure(bias_only_dit_core=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))
