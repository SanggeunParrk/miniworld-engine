"""The fp32 (TF32) path of the bias-only token DiT on B200 (kernels/bias_only_dit/cuda/tf32.py, integrations/bias_only_dit.py and
bias_only_dit_train.py with fp32 inputs): the block's output and every gradient against an fp64 reference, the kernels against
their fp64 references, eager / torch.compile / CUDA graph, that the TF32 CUDA kernels ran (and no Triton), and what it declines.

Tolerance rationale. A TF32 operand keeps 10 mantissa bits (unit roundoff 2^-11 ~ 4.9e-4), so every product of the block -- the
cuBLAS GEMMs and the attention core alike -- carries O(1e-3) relative error whoever computes it; IEEE fp32 (~1e-7) is not the bar.
The bar is the PyTorch fp32 module with its GEMMs on the same TF32 tensor cores (torch.backends.cuda.matmul.allow_tf32 = True):
the fused path must be no worse than that against fp64, within a factor 1.5 (different blockings round differently) plus a floor
of 2^-10 ~ 1e-3: a tcgen05 kind::tf32 MMA reads fp32 operands and drops their low 13 mantissa bits (truncation, mean -2^-11 per
operand) where cuBLAS may round to nearest -- the same TF32 products rounded the other legitimate way differ by that much (the
token DiT's fp32 tests take 1.5x + 3e-3). The reference is the module in fp64 (its LayerNorms compute in fp32, as the engine's LayerNorm
pins them: ~1e-7, negligible here). Kernel tests against fp64 einsum bound the TF32 products at a few 1e-3 and the exact-fp32
kernels (softmax, pair bias) at 1e-5."""

import contextlib
import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit as INF
from miniworld_engine.integrations import bias_only_dit_train as TR
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.bias_only_dit import module as bo_module
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
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
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
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
    ref = randomize(BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH)).cuda()
    fast = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.MINIWORLD).cuda()
    fast.load_state_dict(ref.state_dict())
    ref64 = copy.deepcopy(ref).double()
    return ref, fast, ref64


def kernel_names(fn):
    """The CUDA kernels ``fn`` launches (CUPTI records driver-API launches too)."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events()]


@contextlib.contextmanager
def no_module_path():
    """The block's PyTorch composition must not run (the fused path serves the whole block)."""
    def boom(*a, **k):
        raise AssertionError("the module's PyTorch composition ran")
    old = bo_module.BiasOnlyAttention.forward
    bo_module.BiasOnlyAttention.forward = boom
    try:
        yield
    finally:
        bo_module.BiasOnlyAttention.forward = old


# ------------------------------------------------------------------------------------------------------------ inference
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 768])
@pytest.mark.parametrize("S", [1, 5])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_inference_within_the_pytorch_tf32_error(L, S, shared, masked, n_head, d_head):
    ref, fast, ref64 = blocks(n_head=n_head, d_head=d_head)
    for m in (ref, fast, ref64):
        m.eval()
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
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
    # the output, and the block's update alone (the residual input is exact on every path)
    for g, b, w in ((got, base, want), (got - x, base - x, want - x.double())):
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3
        assert relative(g, w) < 1e-2


def test_inference_graph_compile_and_caches():
    _, fast, _ = blocks(seed=5)
    fast.eval()
    L, S = 256, 5
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    with torch.no_grad():
        eager = fast(x, c, p)
        names = kernel_names(lambda: fast(x, c, p))
        assert any("bo_pv_gate_tf32_sm100" in n for n in names), names
        assert not any("bo_pv_gate_inf_sm100" in n for n in names)
        assert not any("triton" in n.lower() for n in names), [n for n in names if "triton" in n.lower()]
        compiled = torch.compile(fast, fullgraph=True, options={"triton.cudagraphs": False})
        assert relative(compiled(x, c, p), eager) < 1e-5
        # a captured step replays to the eager output (cuBLAS may pick another algorithm under capture: not bit-exact)
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
        # new inputs copied in: the replay follows them (the P cache is keyed on the pair's version inside the capture too)
        x2 = torch.randn_like(x)
        xs.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        assert relative(out, fast(x2, c, p)) < 1e-5
        # a weight updated in place bumps its version: the pack misses
        fast.transition.squeeze.weight.mul_(0.8)
        new = fast(x, c, p)
        assert not torch.equal(new, eager)
        # a pair changed in place bumps its version: the hoisted attention weights miss
        ref = copy.deepcopy(fast)
        ref.implementation = ImplementationType.PYTORCH
        p.add_(0.5 * torch.randn_like(p))
        with tf32(True):
            assert relative(fast(x, c, p), ref(x, c, p)) < 1e-2


# ------------------------------------------------------------------------------------------------------------ training
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


# L / A pick the branches: pv groups of 1 to 8 samples, dpb key tiles of 128, 192 and 256; every head layout
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
    names = ["out", "d single", "d cond", "d pair"] + [n for n, _ in fast.named_parameters()]
    for n, g, b, w in zip(names, got, base, want, strict=True):
        assert g.dtype is torch.float32, n
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3, (n, relative(g, w), relative(b, w))


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_training_kernels_ran_graph_capture_compile_and_steady_memory(n_head, d_head):
    _, fast, _ = blocks(seed=7, n_head=n_head, d_head=d_head)
    L, A = 384, 8
    x, c, p, dy, _ = inputs(L, A, False, seed=1)
    xs, cs, ps = (t.clone().requires_grad_(True) for t in (x, c, p))

    def run(m):
        for t in (xs, cs, ps):
            t.grad = None
        for q in m.parameters():
            if q.grad is not None:
                q.grad.zero_()
        m(xs, cs, ps).backward(dy)
        return [t.grad.clone() for t in (xs, cs, ps)] + [q.grad.clone() for q in m.parameters()]

    eager = run(fast)
    names = kernel_names(lambda: run(fast))
    for k in ("bo_pv_gate_tf32_sm100", "bo_dpb_tf32_sm100"):
        assert any(k in n for n in names), (k, names)
    assert not any("triton" in n.lower() for n in names), [n for n in names if "triton" in n.lower()]
    assert not any("bo_pv_gate_inf_sm100" in n or "bo_dpb_sm100" in n for n in names)
    # the bound launches reuse their argument blocks and the activations are allocated afresh each step: nothing may pile up
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    for _ in range(4):
        run(fast)
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved
    # a captured step replays to the eager gradients (cuBLAS under capture may sum in another order: not bit-exact)
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
        assert relative(g, e) < 1e-4
    # torch.compile keeps the two opaque ops (forward / backward) and gives the same step
    compiled = torch.compile(fast, options={"triton.cudagraphs": False})
    for g, e in zip(run(compiled), eager, strict=True):
        assert relative(g, e) < 1e-4


def test_declines_what_it_does_not_serve():
    _, fast, _ = blocks()
    x = torch.randn(4, 1, 384, 768, device="cuda", requires_grad=True)
    c = torch.randn(4, 1, 384, 384, device="cuda")
    p = torch.randn(1, 384, 384, 128, device="cuda")
    assert TR.serves(fast, x, c, p)
    with torch.no_grad():
        assert INF.serves(fast, x, c, p)
        assert not INF.serves(fast, x, c.bfloat16(), p)                                       # mixed dtypes
        assert not INF.serves(fast, x.double(), c.double(), p.double())                        # fp64
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not INF.serves(fast, x, c, p)                                               # autocast: the module's casts
        bf = copy.deepcopy(fast).to(torch.bfloat16)
        assert not INF.serves(bf, x, c, p)                                                     # fp32 inputs, bf16 weights
    assert not TR.serves(fast, x, c, p.bfloat16())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert not TR.serves(fast, x, c, p)
    assert not TR.serves(fast, x[:, :, :200], c[:, :, :200], p[:, :200, :200])                 # L % 128
    assert not TR.serves(blocks(n_head=8)[1], x, c, p)                                         # 8 x 96: no kernels


# ------------------------------------------------------------------------------------------------------------ kernels
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("S", [1, 2, 3, 5, 8])
def test_core_and_softmax_match_fp64(L, S, n_head, d_head):
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    torch.manual_seed(L + S)
    H, DA = n_head, n_head * d_head
    M = S * L
    vg = torch.randn(M, 2 * DA, device="cuda")
    bias = 2 * torch.randn(H * L, L, device="cuda")
    mask = torch.rand(L, device="cuda") > 0.3
    P = torch.empty_like(bias)
    T.rows32().softmax_rows_cuda(bias, P, mask)
    want_p = torch.softmax(bias.double().masked_fill(~mask, torch.finfo(F64).min), -1)
    assert relative(P, want_p) < 1e-5
    core = T.PvGateCoreTF32(torch.cuda.current_device(), nh=H, dh=d_head)
    v, g = vg[:, :DA], vg[:, DA:]
    o = torch.einsum("hij,sjhd->sihd", P.double().view(H, L, L), v.double().view(S, L, H, d_head)).reshape(M, DA)
    a = torch.empty(M, DA, device="cuda")
    core(v, P, a, S, g=g)
    assert relative(a, torch.sigmoid(g.double()) * o) < 3e-3
    # ungated, into a strided view (the training backward's dV = P^T dO into the v half of dvg)
    out = torch.empty(M, 2 * DA, device="cuda")[:, :DA]
    core(v, P, out, S)
    assert relative(out, o) < 3e-3


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_every_sample_group_of_the_core(monkeypatch, n_head, d_head):
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    L, S, H, DA = 384, 9, n_head, n_head * d_head            # S = 9: a partial last group for every group size
    torch.manual_seed(1)
    v, g = torch.randn(S * L, DA, device="cuda"), torch.randn(S * L, DA, device="cuda")
    P = torch.softmax(torch.randn(H * L, L, device="cuda"), -1)
    o = torch.einsum("hij,sjhd->sihd", P.double().view(H, L, L), v.double().view(S, L, H, d_head)).reshape(S * L, DA)
    for sg in T._PV_GROUPS[d_head]:
        monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_SG", str(sg))
        a = torch.empty(S * L, DA, device="cuda")
        T.PvGateCoreTF32(torch.cuda.current_device(), nh=H, dh=d_head)(v, P, a, S, g=g)       # a new instance: no bound launch reuse
        assert relative(a, torch.sigmoid(g.double()) * o) < 3e-3, sg


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 640, 768])
def test_training_kernels_match_fp64(L, n_head, d_head):
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    from miniworld_engine.kernels.bias_only_dit.cuda.train import ext as bo_ext
    F, dev = T.rows32(), torch.cuda.current_device()
    A, H, DH = 6, n_head, d_head
    DA, M, R = H * DH, A * L, L * L
    torch.manual_seed(L)
    # the pair bias and its backward (exact fp32) against fp64 autograd
    z = torch.randn(R, 128, device="cuda")
    wf = torch.randn(H, 128, device="cuda") / 128 ** 0.5
    bias, pst = torch.empty(H, L, L, device="cuda"), torch.empty(R, 2, device="cuda")
    F.pair_bias_cuda(z, wf, bias, pst, 1e-5)
    z64, wf64 = z.double().requires_grad_(True), wf.double().requires_grad_(True)
    b64 = (torch.nn.functional.layer_norm(z64, (128,), eps=1e-5) @ wf64.t()).t().reshape(H, L, L)
    assert relative(bias, b64) < 1e-5
    db = torch.randn(H, L, L, device="cuda")
    b64.backward(db.double())
    dz = torch.empty_like(z)
    pwf = torch.zeros(F.partial_rows(M), H, 128, device="cuda")
    n = F.pair_bias_bwd_cuda(db, z, pst, wf, dz, pwf)
    assert relative(dz, z64.grad) < 1e-5
    assert relative(pwf[:n].sum(0), wf64.grad) < 1e-5
    # the training softmax writes P^T beside P: the same values, transposed
    mask = torch.rand(L, device="cuda") > 0.2
    P, Pt = torch.empty(H, L, L, device="cuda"), torch.empty(H, L, L, device="cuda")
    F.softmax_t_cuda(bias.view(H * L, L), P.view(H * L, L), Pt.view(H * L, L), mask)
    P_rows = torch.empty_like(P)
    F.softmax_rows_cuda(bias.view(H * L, L), P_rows.view(H * L, L), mask)
    assert torch.equal(P, P_rows)
    assert torch.equal(Pt.transpose(1, 2), P)
    # dV = P^T dO on the ungated core; the bias gradient
    do = torch.randn(M, DA, device="cuda")
    v = torch.randn(M, 2 * DA, device="cuda")[:, :DA]
    dd = torch.randn(A, H, L, device="cuda")
    dh, vh = do.double().view(A, L, H, DH), v.double().reshape(A, L, H, DH)
    dv = torch.empty(M, 2 * DA, device="cuda")[:, :DA]
    T.PvGateCoreTF32(dev, nh=H, dh=DH)(do, Pt.view(H * L, L), dv, A)
    assert relative(dv, torch.einsum("hij,aihd->ajhd", P.double(), dh).reshape(M, DA)) < 3e-3
    dbias = torch.empty(H * L, L, device="cuda")
    T.DpbKernelTF32(dev, nh=H, dh=DH)(do, v, P.view(H * L, L), dd, dbias, A)
    want = P.double() * (torch.einsum("aihd,ajhd->hij", dh, vh) - dd.double().sum(0)[:, :, None])
    assert relative(dbias.view(H, L, L), want) < 3e-3
    # the gate backward rows
    ao, gg = torch.randn(M, DA, device="cuda"), torch.randn(M, 2 * DA, device="cuda")[:, DA:]
    da = torch.randn(M, DA, device="cuda")
    do2, dg, dd2 = torch.empty(M, DA, device="cuda"), torch.empty(M, 2 * DA, device="cuda")[:, DA:], torch.empty(A, H, L, device="cuda")
    F.gate_bwd_cuda(da, ao, gg, do2, dg, dd2, L)
    s = torch.sigmoid(gg.double())
    assert relative(do2, da.double() * s) < 1e-6
    assert relative(dg, da.double() * ao.double() * (1 - s)) < 1e-6
    assert relative(dd2, (da.double() * ao.double()).view(A, L, H, DH).sum(-1).permute(0, 2, 1)) < 1e-6
    bo_ext()                                                     # the shared finalize / unfold extension builds
