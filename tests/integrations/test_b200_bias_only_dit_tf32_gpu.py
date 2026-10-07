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
pins them: ~1e-7, negligible here). Kernel tests against fp64 einsum bound the TF32 products at a few 1e-3 (the training pair bias on
TF32 mma.sync among them) and the exact-fp32 kernels (softmax, the FMA pair bias of inference, the split-fp32 training pair bias) at
1e-5."""

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
# The served fp32 inference step is the three-kernel one (test_inf3_* below); these two pin MINIWORLD_BIAS_ONLY_DIT_INF3=0, the
# 12-launch cuBLAS + rows step that stays selectable (and that a failed three-kernel build falls back to).
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 768])
@pytest.mark.parametrize("S", [1, 5])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_inference_within_the_pytorch_tf32_error(monkeypatch, L, S, shared, masked, n_head, d_head):
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3", "0")
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


@pytest.mark.parametrize("inf3_env", ["0", "1"])
def test_inference_graph_compile_and_caches(monkeypatch, inf3_env):
    """Eager / torch.compile / CUDA graph and the caches, for the 12-launch step (INF3=0) and the served three-kernel step."""
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3", inf3_env)
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
        rows = [n for n in names if any(r in n for r in DEFAULT_ROWS)]
        if inf3_env == "1":                                                  # front -> core -> tail, none of the 12-launch rows
            assert all(any(k in n for n in names) for k in INF3_NAMES) and not rows, names
        else:                                                                # no front / tail kernel
            assert not any(k in n for n in names for k in INF3_NAMES[::2]), names
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


@pytest.mark.parametrize("env", [{"MINIWORLD_BIAS_ONLY_DIT_TF32_GLU": "1"}, {"MINIWORLD_BIAS_ONLY_DIT_TF32_GLU_BWD": "0"},
                                 {"MINIWORLD_BIAS_ONLY_DIT_PV_BATCH": "0"}, {"MINIWORLD_BIAS_ONLY_DIT_F32_RESB_MINB": "2"},
                                 {"MINIWORLD_BIAS_ONLY_DIT_PAIR_EXACT": "1"}, {"MINIWORLD_BIAS_ONLY_DIT_WGRAD_SPLIT": "0"},
                                 {"MINIWORLD_BIAS_ONLY_DIT_WGRAD_SPLIT": "8"}])
def test_switches_keep_every_gradient_within_the_pytorch_tf32_error(monkeypatch, env):
    """Every A/B switch of the fp32 training step (the opt-in fused SwiGLU forward, the backward fusion off, the per-sample core,
    the row kernels' resident-block builds, the exact pair bias) gives a step within the same bound as the default."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    L, A, masked = 384, 6, True
    ref, fast, ref64 = blocks(seed=5)
    x, c, p, dy, mask = inputs(L, A, masked)
    want = step(ref64, x, c, p, mask, dy, F64)
    with no_module_path():
        got = step(fast, x, c, p, mask, dy, torch.float32)
    with tf32(True):
        base = step(ref, x, c, p, mask, dy, torch.float32)
    names = ["out", "d single", "d cond", "d pair"] + [n for n, _ in fast.named_parameters()]
    for n, g, b, w in zip(names, got, base, want, strict=True):
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3, (env, n, relative(g, w), relative(b, w))


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
    from tests.cuda_graph_nodes import launched_kernels      # the graph's own kernel nodes (the profiler drops driver-API launches)
    names = launched_kernels(lambda: run(fast))
    for k in ("bo_pv_gate_tf32_sm100", "bo_dpb32_sm100", "bo_glu_bwd_tf32_sm100", "pair_bias_tc_k", "pair_bias_bwd_tc_k",
              "pack_k"):
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


@pytest.mark.parametrize("batch", ["1", "0"])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_every_sample_group_of_the_core(monkeypatch, n_head, d_head, batch):
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_PV_BATCH", batch)      # one MMA for the group's samples (default) / one per sample
    L, S, H, DA = 384, 9, n_head, n_head * d_head            # S = 9: a partial last group for every group size
    torch.manual_seed(1)
    v, g = torch.randn(S * L, DA, device="cuda"), torch.randn(S * L, DA, device="cuda")
    P = torch.softmax(torch.randn(H * L, L, device="cuda"), -1)
    o = torch.einsum("hij,sjhd->sihd", P.double().view(H, L, L), v.double().view(S, L, H, d_head)).reshape(S * L, DA)
    for sg in T.pv_groups(d_head):
        monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_SG", str(sg))
        core = T.PvGateCoreTF32(torch.cuda.current_device(), nh=H, dh=d_head)                  # a new instance: no bound launch reuse
        a = torch.empty(S * L, DA, device="cuda")
        core(v, P, a, S, g=g)
        assert relative(a, torch.sigmoid(g.double()) * o) < 3e-3, sg
        out = torch.empty(S * L, DA, device="cuda")
        core(v, P, out, S)                                   # ungated (the training dV)
        assert relative(out, o) < 3e-3, sg
        for k in (core.kernel(sg, True), core.kernel(sg, False)):
            assert k.regs <= 128 and k.lmem == 0, (sg, k.regs, k.lmem)


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 256, 384, 512, 640, 768])
def test_dpb32_repeatable(L, n_head, d_head):
    """dpb32_sm100.cu (the fp32 port of dpb_sm100.cu): the bias gradient is a fixed-order sum, so reruns are bit-identical (its
    predecessor dpb_tf32.cu was not from L640: several items per CTA); within 3e-3 of fp64 on four input draws; no spills."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    dev, A, H, DH = torch.cuda.current_device(), 6, n_head, d_head
    DA, M = H * DH, A * L
    dpb = T.Dpb32(dev, nh=H, dh=DH)
    k = dpb.kernel()
    assert k.regs <= 128 and k.lmem == 0, (k.regs, k.lmem)
    for seed in range(4):
        torch.manual_seed(1000 * L + seed)
        P = torch.softmax(2 * torch.randn(H * L, L, device="cuda"), -1)
        do, v = torch.randn(M, DA, device="cuda"), torch.randn(M, 2 * DA, device="cuda")[:, :DA]
        dd = torch.randn(A, H, L, device="cuda")
        outs = []
        for _ in range(3):
            out = torch.empty(H * L, L, device="cuda")
            dpb(do, v, P, dd, out, A)
            outs.append(out)
        assert all(torch.equal(outs[0], o) for o in outs[1:]), seed
        want = P.double().view(H, L, L) * (torch.einsum("aihd,ajhd->hij", do.double().view(A, L, H, DH), v.double().reshape(A, L, H, DH))
                                           - dd.double().sum(0)[:, :, None])
        assert relative(outs[0].view(H, L, L), want) < 3e-3, seed


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [384, 640, 768])
def test_core_tf32_repeatable(L, n_head, d_head):
    """pv_gate_tf32.cu (double-buffered accumulators, several items per CTA from L640 at A = 6): reruns bit-identical, gated and
    ungated (the training dV)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    dev, A, H, DH = torch.cuda.current_device(), 6, n_head, d_head
    DA, M = H * DH, A * L
    core = T.PvGateCoreTF32(dev, nh=H, dh=DH)
    for seed in range(3):
        torch.manual_seed(2000 * L + seed)
        P = torch.softmax(2 * torch.randn(H * L, L, device="cuda"), -1)
        vg = torch.randn(M, 2 * DA, device="cuda")
        for g in (vg[:, DA:], None):
            outs = []
            for _ in range(3):
                out = torch.empty(M, DA, device="cuda")
                core(vg[:, :DA], P, out, A, g=g)
                outs.append(out)
            assert all(torch.equal(outs[0], o) for o in outs[1:]), (seed, g is None)


@pytest.mark.parametrize("M", [384, 1024, 18432])
def test_swiglu_gemms_match_fp64(M):
    """gemm_glu_tf32.cu: the expand GEMM with the SwiGLU in its epilogue (h, a | b) and the dh GEMM with the SwiGLU backward in its
    epilogue (da | db), TF32 products against fp64; M = 384: an odd number of 128-row tiles (a pair's missing tile)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    dev, D, H = torch.cuda.current_device(), 768, 1536
    torch.manual_seed(M)
    x, w = torch.randn(M, D, device="cuda"), torch.randn(2 * H, D, device="cuda") / D ** 0.5
    fwd, bwd = T.GemmGluTF32(dev), T.GemmGluTF32(dev, bwd=True)
    for k in (fwd.k, bwd.k):
        assert k.regs <= 128 and k.lmem == 0, (k.regs, k.lmem)
    ab, h = torch.empty(M, 2 * H, device="cuda"), torch.empty(M, H, device="cuda")
    fwd(x, w, ab, h)
    ab64 = x.double() @ w.double().t()
    a64, b64 = ab64[:, :H], ab64[:, H:]
    assert relative(ab, ab64) < 3e-3
    assert relative(h, torch.nn.functional.silu(a64) * b64) < 3e-3
    # the SwiGLU of the stored a | b is the fp32 row pass's (same formula; at most the last bit apart)
    h_rows = torch.empty_like(h)
    T.rows32().swiglu_cuda(ab, h_rows)
    assert relative(h, h_rows) < 1e-6
    dz, wsq = torch.randn(M, D, device="cuda"), torch.randn(D, H, device="cuda") / H ** 0.5
    dab = torch.empty_like(ab)
    bwd(dz, wsq.t().contiguous(), ab, dab)
    dh = dz.double() @ wsq.double()
    a, b = ab.double()[:, :H], ab.double()[:, H:]
    sa = torch.sigmoid(a)
    assert relative(dab[:, :H], dh * b * sa * (1 + a * (1 - sa))) < 3e-3
    assert relative(dab[:, H:], dh * a * sa) < 3e-3


def test_fp32_row_kernels_fit_128_registers():
    """Every kernel of the fp32 row extension (defaults and opt-in builds): at most 128 registers, no local memory (spills). The
    three-warp res_adaln_b_bwd is built for two 192-thread blocks per SM: up to 168 registers."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T
    cap = {"res_adaln_b_bwd_k": 168}
    bad = [(name, regs, lmem) for name, regs, lmem in T.rows32().func_attrs() if regs > cap.get(name, 128) or lmem]
    assert not bad, bad


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
    # the training step's pair bias and its backward on TF32 tensor cores (mma.sync): TF32 products (terms 1, the default) bound
    # as the other TF32 products; split fp32 (terms 3, MINIWORLD_BIAS_ONLY_DIT_PAIR_EXACT=1) as the exact kernels
    for terms, tol in ((1, 3e-3), (3, 1e-5)):
        bias_t, pst_t = torch.empty_like(bias), torch.empty_like(pst)
        F.pair_bias_tc_cuda(z, wf, bias_t, pst_t, 1e-5, terms)
        assert relative(bias_t, b64) < tol, terms
        assert relative(pst_t, pst) < 1e-6, terms
        dz_t = torch.empty_like(z)
        pwf_t = torch.zeros(F.partial_rows(M), H, 128, device="cuda")
        n_t = F.pair_bias_bwd_tc_cuda(db, z, pst, wf, dz_t, pwf_t, terms)
        assert relative(dz_t, z64.grad) < tol, terms
        assert relative(pwf_t[:n_t].sum(0), wf64.grad) < tol, terms
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
    dpb = T.Dpb32(dev, nh=H, dh=DH)
    dpb(do, v, P.view(H * L, L), dd, dbias, A)
    want = P.double() * (torch.einsum("aihd,ajhd->hij", dh, vh) - dd.double().sum(0)[:, :, None])
    rel = relative(dbias.view(H, L, L), want)
    if rel >= 3e-3:                                              # where: per (head, 128-query tile, 32-key piece), and a rerun
        again = torch.empty_like(dbias)
        dpb(do, v, P.view(H * L, L), dd, again, A)
        err = (dbias.view(H, L, L).double() - want).abs().view(H, L // 128, 128, L // 32, 32).amax((2, 4))
        top = torch.topk(err.flatten(), 6)
        where = [(int(i) // (err.shape[1] * err.shape[2]), int(i) // err.shape[2] % err.shape[1] * 128, int(i) % err.shape[2] * 32,
                  float(e)) for e, i in zip(top.values, top.indices)]
        raise AssertionError(f"dpb32 dbias rel {rel:.3e} (NJ {T.DPB32_NJ}); rerun bit-identical: "
                             f"{torch.equal(dbias, again)}, rerun rel {relative(again.view(H, L, L), want):.3e}; worst (head, i0, j0, "
                             f"max abs err): {where}; median tile err {float(err.median()):.3e}, |want| max {float(want.abs().max()):.3e}")
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


# ------------------------------------------------------------------------------------------------------------ three-kernel inference step
# the served fp32 inference step (MINIWORLD_BIAS_ONLY_DIT_INF3=0 turns it off): bo_front_tf32 -> pv_gate_tf32 (-DPDL_INF) ->
# bo_tail_tf32 per block, the conditioning tables hoisted
INF3_L = [128, 256, 384, 512, 640, 768]
INF3_NAMES = ("bo_front_tf32_sm100", "bo_pv_gate_tf32_sm100", "bo_tail_tf32_sm100")
#: the 12-launch step's row kernels, which the three-kernel step must not launch
DEFAULT_ROWS = ("adaln_in_rows", "resgate_adaln_rows", "resgate_out_rows", "swiglu_k")


@pytest.fixture
def inf3(monkeypatch):
    """The step on (explicitly, though it is the default), and a spy that the runner really took it (a failed build would fall
    back silently)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import runner as RN
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3", "1")
    calls = []
    orig = RN.FusedBiasOnlyDiT._step3

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(RN.FusedBiasOnlyDiT, "_step3", spy)
    return calls


def _tables(T, nb=2, seed=0):
    """Random hoisted tables [T, nb, 6, 768] in the step's layout (gate1, gate2, s1, s2 through a sigmoid; sh1, sh2 raw)."""
    torch.manual_seed(seed)
    tab = torch.randn(T, nb, 6, 768, device="cuda")
    tab[:, :, :4] = torch.sigmoid(tab[:, :, :4] + 1.0)
    return tab


def _ln64(x):
    x = x.double()
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", INF3_L)
@pytest.mark.parametrize("S", [1, 5])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_inf3_within_the_pytorch_tf32_error(inf3, L, S, shared, masked, n_head, d_head):
    """The three-kernel step against the fp64 module at the default inference step's bounds; every head layout, L 128..768 (both
    front clusters: 8 up to 18 row tiles, 4 above), masked / unmasked, shared / per-sample conditioning."""
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
    assert inf3, "the three-kernel step did not run"
    assert got.dtype is torch.float32
    for g, b, w in ((got, base, want), (got - x, base - x, want - x.double())):
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3, (relative(g, w), relative(b, w))
        assert relative(g, w) < 1e-2


@pytest.mark.parametrize("tail_cl", ["8", "6", "4"])
@pytest.mark.parametrize("cl", ["4", "6", "8"])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
@pytest.mark.parametrize("L", [384, 512, 768])
def test_inf3_every_cluster_size(inf3, monkeypatch, cl, tail_cl, n_head, d_head, L):
    """Every front cluster size (8 / 6 / 4; 6 only for 768 attention channels) and every tail (8, 6 with its two a | b passes, 4: the
    pair tail bo_tail2_tf32) forced at shapes where the default would pick another (A = 5: front 8 at L384, 6 at L512, 4 at L768;
    tail 6 at L512)."""
    if cl == "6" and d_head == 64:
        pytest.skip("front CL 6 splits 1536 v|g columns: 768 attention channels only")
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_CL", cl)
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL", tail_cl)
    ref, fast, ref64 = blocks(seed=17, n_head=n_head, d_head=d_head)
    for m in (ref, fast, ref64):
        m.eval()
    S = 5
    x, c = torch.randn(S, 1, L, 768, device="cuda"), torch.randn(S, 1, L, 384, device="cuda")
    p, mask = torch.randn(1, L, L, 128, device="cuda"), torch.rand(1, L, device="cuda") > 0.2
    with torch.no_grad():
        want = ref64(x.double(), c.double(), p.double(), mask)
        with no_module_path():
            got = fast(x, c, p, mask)
        with tf32(True):
            base = ref(x, c, p, mask)
    assert inf3
    assert relative(got - x, want - x.double()) <= 1.5 * relative(base - x, want - x.double()) + 1e-3


@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize("cl", [4, 6, 8])
@pytest.mark.parametrize(("M", "T"), [(128, 128), (640, 128), (1920, 1920), (3840, 768)])
def test_front_kernel_matches_fp64(monkeypatch, da, cl, M, T):
    if cl == 6 and da == 1024:
        pytest.skip("front CL 6: 768 attention channels only")
    """bo_front_tf32.cu alone: v | g = (LN(x) s1 + sh1) [Wv; Wg]^T against fp64 (TF32 products: 3e-3), on a table slice with a
    multi-block row stride; v leaves rounded to TF32 (it is the core's MMA operand), g as accumulated."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T32
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_CL", str(cl))
    torch.manual_seed(M + da + cl)
    x = 1.5 * torch.randn(M, 768, device="cuda") + 0.3
    tab = _tables(T, seed=M)
    w = T32.round_tf32(torch.randn(2 * da, 768, device="cuda") / 768 ** 0.5)
    vg = torch.full((M, 2 * da), float("nan"), device="cuda")
    xa = torch.full((M, 768), float("nan"), device="cuda")
    front = T32.FrontTF32(torch.cuda.current_device(), da)
    k = front.kernel(cl)
    assert k.lmem == 0 and k.regs <= 168, (k.regs, k.lmem)
    front(x, tab[:, 1], w, vg, T, xa=xa)
    tok = torch.arange(M, device="cuda") % T
    t64 = tab[:, 1].double()[tok]
    xa64 = _ln64(x) * t64[:, 2] + t64[:, 4]
    want = xa64 @ w.double().t()
    assert torch.isfinite(vg).all()
    assert relative(vg, want) < 3e-3, relative(vg, want)
    assert torch.equal(vg[:, :da], T32.round_tf32(vg[:, :da]))
    assert torch.isfinite(xa).all() and torch.equal(xa, T32.round_tf32(xa))     # the exchange scratch: every block written, TF32
    assert relative(xa, xa64) < 2e-3
    again = torch.empty_like(vg)
    front(x, tab[:, 1], w, again, T, xa=xa)
    assert torch.equal(again, vg)


@pytest.mark.parametrize("cl", [8, 6])
@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize(("M", "T"), [(128, 128), (640, 128), (1920, 1920), (3840, 768)])
def test_tail_kernel_matches_fp64(cl, da, M, T):
    """bo_tail_tf32.cu alone: out = x1 + gate2 (silu(xt Wa^T) (xt Wb^T)) Wsq^T, x1 = x + gate1 a Wo^T, xt = LN(x1) s2 + sh2, against
    fp64 -- the block update out - x within 3e-3 (TF32 products) -- and bit-identical on a rerun."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T32
    torch.manual_seed(M + da)
    dev = torch.cuda.current_device()
    a = T32.round_tf32(torch.randn(M, da, device="cuda"))
    x = 1.5 * torch.randn(M, 768, device="cuda") + 0.3
    tab = _tables(T, seed=M + 1)
    wo = T32.round_tf32(torch.randn(768, da, device="cuda") / da ** 0.5)
    wab = T32.round_tf32(torch.randn(3072, 768, device="cuda") / 768 ** 0.5)
    wsq = T32.round_tf32(torch.randn(768, 1536, device="cuda") / 1536 ** 0.5)
    tail = T32.TailTF32(dev, da)
    nc = 768 // cl
    wop, wsqp = T32.pack_pairs(wo, cl), T32.pack_pairs(wsq, cl)
    k0 = torch.arange(32, device="cuda")
    assert torch.equal(wop.view(cl, -1, 2, nc, 32)[3, 5, 1, 7], wo[nc * 3 + 7, 64 * 5 + 32 + k0])   # P[((c p) 2 + h) NC + n, k]
    k = tail.kernel(cl)
    assert k.lmem == 0 and k.regs <= 168, (k.regs, k.lmem)
    out = torch.full((M, 768), float("nan"), device="cuda")
    xt = torch.full((M, 768), float("nan"), device="cuda")              # row-major
    hs = torch.full((M * 48, 32), float("nan"), device="cuda")          # blocked k-block-major
    tail(a, x, tab[:, 0], wop, wab, wsqp, out, T, xt=xt, h=hs, cl=cl)
    tok = torch.arange(M, device="cuda") % T
    t64 = tab[:, 0].double()[tok]
    x1 = x.double() + t64[:, 0] * (a.double() @ wo.double().t())
    xt64 = _ln64(x1) * t64[:, 3] + t64[:, 5]
    ab = xt64 @ wab.double().t()
    h = torch.nn.functional.silu(ab[:, :1536]) * ab[:, 1536:]
    want = x1 + t64[:, 1] * (h @ wsq.double().t())
    assert torch.isfinite(out).all()
    assert relative(out - x, want - x.double()) < 3e-3, relative(out - x, want - x.double())
    assert torch.isfinite(xt).all() and torch.isfinite(hs).all()          # the exchange scratch: every block written
    rows = lambda t, nk: t.view(M // 128, nk, 128, 32).permute(0, 2, 1, 3).reshape(M, 32 * nk)   # noqa: E731  blocked -> rows
    assert relative(xt, xt64) < 2e-3 and relative(rows(hs, 48), h) < 3e-3
    again = torch.empty_like(out)
    tail(a, x, tab[:, 0], wop, wab, wsqp, again, T, xt=xt, h=hs, cl=cl)
    assert torch.equal(again, out)


@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize(("M", "T"), [(128, 128), (640, 128), (3200, 3200), (3840, 768)])
def test_tail2_kernel_matches_fp64(da, M, T):
    """bo_tail2_tf32.cu (the pair tail: two row tiles x 4 column groups per cluster of 8, tcgen05 cta_group::2) alone against fp64 at
    the CL 8 tail's bounds, its padded scratch fully written (an odd tile count: the missing tile writes its own padding rows and no
    output -- M 128, 640, 3200 = A 5 x L640), and bit-identical on a rerun."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T32
    torch.manual_seed(M + da + 7)
    dev = torch.cuda.current_device()
    a = T32.round_tf32(torch.randn(M, da, device="cuda"))
    x = 1.5 * torch.randn(M, 768, device="cuda") + 0.3
    tab = _tables(T, seed=M + 3)
    wo = T32.round_tf32(torch.randn(768, da, device="cuda") / da ** 0.5)
    wab = T32.round_tf32(torch.randn(3072, 768, device="cuda") / 768 ** 0.5)
    wsq = T32.round_tf32(torch.randn(768, 1536, device="cuda") / 1536 ** 0.5)
    tail = T32.TailTF32(dev, da)
    k = tail.kernel(T32.TAIL_PAIR)
    assert k.lmem == 0 and k.regs <= 168, (k.regs, k.lmem)
    Mp = T32.tail_rows(M, T32.TAIL_PAIR)
    out = torch.full((M, 768), float("nan"), device="cuda")
    xt = torch.full((Mp, 800), float("nan"), device="cuda")[:, :768]    # padded rows, as the runner's
    hs = torch.full((Mp * 48, 32), float("nan"), device="cuda")
    wop, wsqp = T32.pack_pairs(wo, 8), T32.pack_pairs(wsq, 8)
    tail(a, x, tab[:, 0], wop, wab, wsqp, out, T, xt=xt, h=hs, cl=T32.TAIL_PAIR)
    tok = torch.arange(M, device="cuda") % T
    t64 = tab[:, 0].double()[tok]
    x1 = x.double() + t64[:, 0] * (a.double() @ wo.double().t())
    xt64 = _ln64(x1) * t64[:, 3] + t64[:, 5]
    ab = xt64 @ wab.double().t()
    h = torch.nn.functional.silu(ab[:, :1536]) * ab[:, 1536:]
    want = x1 + t64[:, 1] * (h @ wsq.double().t())
    assert torch.isfinite(out).all()
    assert relative(out - x, want - x.double()) < 3e-3, relative(out - x, want - x.double())
    assert torch.isfinite(xt).all() and torch.isfinite(hs).all()          # every block of every tile, padding included
    rows = lambda t, nk: t.view(Mp // 128, nk, 128, 32).permute(0, 2, 1, 3).reshape(Mp, 32 * nk)[:M]   # noqa: E731
    assert relative(xt[:M], xt64) < 2e-3 and relative(rows(hs, 48), h) < 3e-3
    again = torch.empty_like(out)
    tail(a, x, tab[:, 0], wop, wab, wsqp, again, T, xt=xt, h=hs, cl=T32.TAIL_PAIR)
    assert torch.equal(again, out)


def _poison(mb=512):
    """Fill the caching allocator's next blocks with NaN: a buffer the step reads before writing would show it."""
    junk = torch.full((mb << 18,), float("nan"), device="cuda")
    torch.cuda.synchronize()
    del junk


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 640, 768])
@pytest.mark.parametrize("shared", [True, False])
def test_inf3_bit_identical_reruns_with_a_poisoned_allocator(inf3, n_head, d_head, L, shared):
    """Three reruns from scratch (the runner's buffers, tables, bound launches and P dropped; the allocator's free memory filled with
    NaN before each): finite and bit-identical -- fixed-order reductions, no atomics, nothing read before it is written."""
    _, fast, _ = blocks(seed=9, n_head=n_head, d_head=d_head)
    fast.eval()
    S = 5
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
    p, mask = torch.randn(1, L, L, 128, device="cuda"), torch.rand(1, L, device="cuda") > 0.2
    outs = []
    for _ in range(3):
        INF._RUNNERS.clear()
        _poison()
        with torch.no_grad():
            outs.append(fast(x, c, p, mask).clone())
        torch.cuda.synchronize()
    assert len(inf3) == 3
    assert torch.isfinite(outs[0]).all()
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
def test_inf3_graph_is_three_kernels_and_replays(inf3, static, n_head, d_head):
    """The captured step: with the weights and the conditioning declared static (the bench harness's inference mode) exactly the three
    kernels, replaying bit-identically to the eager call and following a new single copied in; without, the hoists (weight pack, P,
    conditioning tables) are recorded too and the three kernels close the graph. The default step's row kernels never appear."""
    from miniworld_engine.kernels import _capture
    from tests.cuda_graph_nodes import graph_kernels
    _, fast, _ = blocks(seed=5, n_head=n_head, d_head=d_head)
    fast.eval()
    L, S = 384, 5
    x, c, p = torch.randn(S, 1, L, 768, device="cuda"), torch.randn(S, 1, L, 384, device="cuda"), torch.randn(1, L, L, 128, device="cuda")
    xs, cs, ps = x.clone(), c.clone(), p.clone()
    ctx = contextlib.ExitStack()
    if static:
        ctx.enter_context(_capture.static_weights())
        ctx.enter_context(_capture.static_inputs())
    with ctx, torch.no_grad():
        eager = fast(x, c, p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fast(xs, cs, ps)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out = fast(xs, cs, ps)
        names = graph_kernels(graph)
        graph.replay()
        torch.cuda.synchronize()
        if static:
            assert len(names) == 3 and all(k in n for k, n in zip(INF3_NAMES, names, strict=True)), names
            assert torch.equal(out, eager)
        else:
            assert all(k in n for k, n in zip(INF3_NAMES, names[-3:], strict=True)), names
            assert sum(any(k in n for k in INF3_NAMES) for n in names) == 3, names
            assert relative(out, eager) < 1e-5
        assert not any(r in n for n in names for r in DEFAULT_ROWS), names
        x2 = torch.randn_like(x)
        xs.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        new = fast(x2, c, p)
        if static:
            assert torch.equal(out, new)
        else:
            assert relative(out, new) < 1e-5


def test_inf3_tables_follow_the_conditioning(inf3):
    """The hoisted tables are keyed on the conditioning tensor's version: an in-place change is seen by the next call; a later call
    with the same tensor launches none of the table kernels."""
    from tests.cuda_graph_nodes import launched_kernels
    ref, fast, _ = blocks(seed=21)
    for m in (ref, fast):
        m.eval()
    L, S = 256, 5
    x, c, p = torch.randn(S, 1, L, 768, device="cuda"), torch.randn(S, 1, L, 384, device="cuda"), torch.randn(1, L, L, 128, device="cuda")
    with torch.no_grad(), tf32(True):
        first = fast(x, c, p)
        c.mul_(0.5).add_(0.3)
        second = fast(x, c, p)
        assert not torch.equal(first, second)
        assert relative(second, ref(x, c, p)) < 1e-2
    from miniworld_engine.kernels import _capture
    with torch.no_grad(), _capture.static_weights(), _capture.static_inputs():
        fast(x, c, p)
        names = launched_kernels(lambda: fast(x, c, p))
    assert len(names) == 3 and all(k in n for k, n in zip(INF3_NAMES, names, strict=True)), names


def test_inf3_two_blocks_match_two_single_block_steps(inf3, monkeypatch):
    """FusedBiasOnlyDiT over two blocks (the ping-pong residual buffers, per-block table slices) against the 12-launch step's runner and
    against two single-block three-kernel runners."""
    from miniworld_engine.kernels.bias_only_dit.cuda.runner import FusedBiasOnlyDiT
    b1, b2 = blocks(seed=31)[1], blocks(seed=32)[1]
    L, S = 384, 5
    x, c, p = torch.randn(S, 1, L, 768, device="cuda"), torch.randn(S, 1, L, 384, device="cuda"), torch.randn(1, L, L, 128, device="cuda")
    with torch.no_grad():
        two = FusedBiasOnlyDiT([b1, b2], dtype=torch.float32)
        P = two.hoist(p.contiguous())
        got = two.step(x, c, P)
        r1, r2 = FusedBiasOnlyDiT([b1], dtype=torch.float32), FusedBiasOnlyDiT([b2], dtype=torch.float32)
        seq = r2.step(r1.step(x, c, r1.hoist(p.contiguous())), c, r2.hoist(p.contiguous()))
    assert len(inf3) == 3
    assert torch.isfinite(got).all()
    assert relative(got, seq) < 1e-3
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3", "0")              # the same runner, the 12-launch step
    with torch.no_grad():
        default = two.step(x, c, P)
    assert len(inf3) == 3
    assert relative(got, default) < 1e-2


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_inf3_kernels_do_not_spill(n_head, d_head):
    """No local memory in any three-kernel-step cubin: every front and tail cluster (the pair tail too) for the layout's width (__launch_bounds__(384,
    1): up to 168 registers), the PDL core for every sample group (<= 128 registers, its __launch_bounds__(256, 2)). The bar is the
    served cubins: the -DTRACE builds (bench_scripts/bo32_inf3_trace.py only) may spill a few bytes for the %globaltimer stamps
    (front CL 4: 4 B), which this does not check."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T32
    dev, da = torch.cuda.current_device(), n_head * d_head
    front, tail = T32.FrontTF32(dev, da), T32.TailTF32(dev, da)
    for k in [front.kernel(cl) for cl in front.clusters()] + [tail.kernel(cl) for cl in (*T32.TAIL_CLUSTERS, T32.TAIL_PAIR)]:
        assert k.lmem == 0 and k.regs <= 168, (k.regs, k.lmem)
    core = T32.PvGateCoreTF32(dev, nh=n_head, dh=d_head, pdl=True)
    for sg in T32.pv_groups(d_head):
        k = core.kernel(sg, True)
        assert k.lmem == 0 and k.regs <= 128, (sg, k.regs, k.lmem)


# ------------------------------------------------------------------------------------------------------------ the pair tail (A/B)
# bo_tail2_tf32 where it saves a round (A = 5: L640 / L768; the default, MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA=0 keeps CL 8 / CL 6)
@pytest.fixture
def pair_tail(inf3, monkeypatch):
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA", "1")
    return inf3


@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
@pytest.mark.parametrize("L", [640, 768])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_inf3_pair_tail_within_the_pytorch_tf32_error(pair_tail, L, shared, masked, n_head, d_head):
    """The three-kernel step with the pair tail against the fp64 module at the inf3 tests' bounds."""
    ref, fast, ref64 = blocks(seed=23, n_head=n_head, d_head=d_head)
    for m in (ref, fast, ref64):
        m.eval()
    S = 5
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    with torch.no_grad():
        want = ref64(x.double(), c.double(), p.double(), mask)
        with no_module_path():
            got = fast(x, c, p, mask)
        with tf32(True):
            base = ref(x, c, p, mask)
    assert pair_tail, "the three-kernel step did not run"
    for g, b, w in ((got, base, want), (got - x, base - x, want - x.double())):
        assert relative(g, w) <= 1.5 * relative(b, w) + 1e-3, (relative(g, w), relative(b, w))
        assert relative(g, w) < 1e-2


@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
@pytest.mark.parametrize("L", [640, 768])
def test_inf3_pair_tail_bit_identical_reruns_with_a_poisoned_allocator(pair_tail, n_head, d_head, L):
    """Three steps from scratch (runner, buffers, tables, bound launches dropped; free memory NaN-filled before each): finite and
    bit-identical -- the pair tail reads nothing before writing it, its padding rows included."""
    _, fast, _ = blocks(seed=29, n_head=n_head, d_head=d_head)
    fast.eval()
    S = 5
    x, c = torch.randn(S, 1, L, 768, device="cuda"), torch.randn(S, 1, L, 384, device="cuda")
    p, mask = torch.randn(1, L, L, 128, device="cuda"), torch.rand(1, L, device="cuda") > 0.2
    outs = []
    for _ in range(3):
        INF._RUNNERS.clear()
        _poison()
        with torch.no_grad():
            outs.append(fast(x, c, p, mask).clone())
        torch.cuda.synchronize()
    assert len(pair_tail) == 3
    assert torch.isfinite(outs[0]).all()
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@pytest.mark.parametrize("on", ["1", "0"])
@pytest.mark.parametrize(("L", "pair_L"), [(768, True), (640, True), (512, False), (384, False)])
def test_pair_tail_switch_selects_bo_tail2(inf3, monkeypatch, on, L, pair_L):
    """A = 5: the pair tail at L640 / L768 (25 / 30 tiles: one round of clusters of two tiles instead of two), the CL 6 / CL 8 tail at
    L512 / L384; MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA=0 keeps CL 8 / CL 6 everywhere. Read off the runner's choice (a profiler kernel list came back without kernel names late in
    this file's run)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32 as T32
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA", on)
    want = pair_L and on == "1"
    picked = []
    orig = T32.TailTF32.cluster

    def spy(self, n_tiles):
        cl = orig(self, n_tiles)
        picked.append(cl)
        return cl

    monkeypatch.setattr(T32.TailTF32, "cluster", spy)
    _, fast, _ = blocks(seed=3)
    fast.eval()
    S = 5
    x, c, p = torch.randn(S, 1, L, 768, device="cuda"), torch.randn(S, 1, L, 384, device="cuda"), torch.randn(1, L, L, 128, device="cuda")
    INF._RUNNERS.clear()
    with torch.no_grad():
        out = fast(x, c, p)
    torch.cuda.synchronize()
    assert inf3 and picked, (len(inf3), picked)
    assert torch.isfinite(out).all()
    assert all((cl == T32.TAIL_PAIR) is want for cl in picked), picked
