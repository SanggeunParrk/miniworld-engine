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


# ------------------------------------------------------------------------------------------------- the step's defaults and their switches
# The bf16 step's one-launch weight pack (MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1), dxt / dxa written into dG's d-shift columns
# (MINIWORLD_BIAS_ONLY_DIT_BWD_DG) and the CTA-pair bias gradient (MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR) are the defaults; each =0 restores
# the previous launches. The test script runs this file at the defaults and with all three at 0.
F64 = torch.float64
SWITCHES = ("MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1", "MINIWORLD_BIAS_ONLY_DIT_BWD_DG", "MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR")


def relative64(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _poison(mb=512):
    """Fill the caching allocator's next blocks with NaN: a buffer read before it is written would show it."""
    junk = torch.full((mb << 18,), float("nan"), device="cuda")
    torch.cuda.synchronize()
    del junk


SHAPES64 = [(128, 8, False), (384, 3, True), (640, 4, False), (768, 16, True)]


@pytest.mark.parametrize(("L", "A", "masked"), SHAPES64)
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_every_gradient_against_fp64(L, A, masked, n_head, d_head):
    """The step as the environment sets it (fused backward GEMMs on or off): the output and every gradient against the block in
    fp64, within 1.1x the PyTorch bf16 block's error + 1e-4 (the bound of the fp32-reference test above). L384 A3: an odd number of
    128-row tiles."""
    import copy
    ref, fast, ref_bf = blocks(n_head=n_head, d_head=d_head)
    ref64 = copy.deepcopy(ref).double()
    x, c, p, dy, mask = inputs(L, A, masked)
    want = step(ref64, x, c, p, mask, dy, F64)
    got = step(fast, x, c, p, mask, dy, torch.bfloat16)
    base = step(ref_bf, x, c, p, mask, dy, torch.bfloat16)
    names = ["out", "d single", "d cond", "d pair"] + [n for n, _ in fast.named_parameters()]
    for n, g, b, w in zip(names, got, base, want, strict=True):
        assert relative64(g, w) <= 1.1 * relative64(b, w) + 1e-4, (n, relative64(g, w), relative64(b, w))


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_steps_bit_identical_with_a_poisoned_allocator(n_head, d_head):
    """Twenty training steps (forward + backward) as the environment sets them, the allocator's free memory filled with NaN before
    each: every gradient finite and bit-identical across the steps."""
    _, fast, _ = blocks(seed=11, n_head=n_head, d_head=d_head)
    L, A = 384, 5
    x, c, p, dy, mask = inputs(L, A, True, seed=2)
    runs = []
    for _ in range(20):
        _poison()
        runs.append([t.clone() for t in step(fast, x, c, p, mask, dy, torch.bfloat16)])
        torch.cuda.synchronize()
    for i, g in enumerate(runs[0]):
        assert torch.isfinite(g).all(), i
        for r in runs[1:]:
            assert torch.equal(r[i], g), i


@pytest.mark.parametrize("off", [False, True])
def test_defaults_and_their_switches(monkeypatch, off):
    """At the defaults the weight pack is one pack16 launch, the LayerNorm backward rows get dxt / dxa as dG's own d-shift columns,
    and the L768 bias gradient runs on CTA pairs; with the three switches at 0, the torch pack, separate dxt / dxa and dpb_sm100."""
    from miniworld_engine.kernels.bias_only_dit.cuda import train as TRN
    for v in SWITCHES:
        if off:
            monkeypatch.setenv(v, "0")
        else:
            monkeypatch.delenv(v, raising=False)
    TR._PACKS.clear()
    TR._OPS.clear()
    rows, alias = [], []
    orig_ext = TRN.ext

    class _Ext:
        def __getattr__(self, n):
            f = getattr(orig_ext(), n)
            if n not in ("pack16_cuda", "res_adaln_b_bwd_cuda", "adaln_a_bwd_cuda"):
                return f

            def w(*a, **k):
                if n == "res_adaln_b_bwd_cuda":            # (dout, dxt, ..., dG = a[11]): dxt is dG[:, 3D:] itself
                    alias.append(a[1].data_ptr() == a[11].data_ptr() + 3 * 768 * 2)
                elif n == "adaln_a_bwd_cuda":              # (dxa, ..., dG = a[7]): dxa is dG[:, D:2D] itself
                    alias.append(a[0].data_ptr() == a[7].data_ptr() + 768 * 2)
                else:
                    rows.append(n)
                return f(*a, **k)
            return w

    monkeypatch.setattr(TRN, "ext", lambda: _Ext())
    _, fast, _ = blocks(seed=4)
    x, c, p, dy, mask = inputs(768, 2, True, seed=4)
    got = step(fast, x, c, p, mask, dy, torch.bfloat16)
    torch.cuda.synchronize()
    dpb = next(op for k, op in TR._OPS.items() if k[0] == "dpb")
    TR._PACKS.clear()
    assert all(torch.isfinite(g).all() for g in got)
    assert len(alias) == 2, alias
    if off:
        assert not rows and not any(alias) and dpb.last == "single", (rows, alias, dpb.last)
    else:
        assert rows == ["pack16_cuda"] and all(alias) and dpb.last == "pair", (rows, alias, dpb.last)


@pytest.mark.parametrize("fp32_norms", [False, True])
def test_one_launch_weight_pack_matches_the_torch_pack(monkeypatch, fp32_norms):
    """The one-launch pack (the default; MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1=0: the torch pack) gives the torch pack's tensors bit
    for bit (bf16 casts, the folded cond-LN and pair-LN weights, the concatenations, fp32 copies); with every parameter bf16 and with
    the LayerNorm weights in fp32 (the engine keeps them fp32 in a bf16 block)."""
    _, fast, _ = blocks(seed=6, n_head=24)
    if fp32_norms:
        for n, q in fast.named_parameters():
            if "ln_" in n:
                q.data = q.data.float()
    params = dict(zip(TR.NAMES, [fast.get_parameter(n) for n in TR.NAMES], strict=True))
    packs = []
    for on in ("0", "1"):
        monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1", on)
        TR._PACKS.clear()
        packs.append(dict(TR._pack(params, torch.device("cuda"))))
        torch.cuda.synchronize()
    TR._PACKS.clear()
    ref, one = packs
    for k, v in ref.items():
        assert one[k].dtype is v.dtype and one[k].shape == v.shape and torch.equal(one[k], v), k


# ------------------------------------------------------------------------------------------------- bias gradient on CTA pairs (dpbx2)
def _dpb_case(L, A, H, DH, seed):
    DA, M = H * DH, A * L
    g = torch.Generator(device="cuda").manual_seed(seed)
    do = torch.randn(M, DA, device="cuda", generator=g).bfloat16()
    v = torch.randn(M, 2 * DA, device="cuda", generator=g).bfloat16()[:, :DA]
    dd = torch.randn(A, H, L, device="cuda", generator=g)
    bias = 2 * torch.randn(H * L, L, device="cuda", generator=g).bfloat16()
    P, Pt = (torch.empty(H, L, L, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    C.softmax_t(bias, P.view(H * L, L), Pt.view(H * L, L), torch.rand(L, device="cuda", generator=g) > 0.2)
    return do, v, dd, P


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_dpb_pair_kernel_does_not_spill(n_head, d_head):
    k = C.DpbKernel(torch.cuda.current_device(), nh=n_head, dh=d_head).pair_kernel()
    assert k.lmem == 0 and k.regs <= 255, (k.regs, k.lmem)


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "A"), [(256, 6), (512, 5), (768, 6), (384, 6)])
def test_dpb_pair_matches_einsum(monkeypatch, L, A, n_head, d_head):
    """MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR=1 (forced): dpbx2_sm100.cu wherever 256 divides L (L384 keeps dpb_sm100.cu), within 3e-3 of
    fp64 einsum."""
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR", "1")
    H, DH = n_head, d_head
    do, v, dd, P = _dpb_case(L, A, H, DH, seed=L + A)
    op = C.DpbKernel(torch.cuda.current_device(), nh=H, dh=DH)
    db = torch.empty(H * L, L, device="cuda", dtype=torch.bfloat16)
    op(do, v, P.view(H * L, L), dd, db, A)
    torch.cuda.synchronize()
    assert op.last == ("pair" if L % 256 == 0 else "single")
    dh, vh = do.float().view(A, L, H, DH), v.float().reshape(A, L, H, DH)
    want = P.double() * (torch.einsum("aihd,ajhd->hij", dh.double(), vh.double()) - dd.double().sum(0)[:, :, None])
    assert relative64(db.view(H, L, L), want) < 3e-3, relative64(db.view(H, L, L), want)


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "A"), [(768, 48), (512, 7)])
def test_dpb_pair_bit_identical_with_a_poisoned_allocator(monkeypatch, L, A, n_head, d_head):
    """20 runs of dpbx2_sm100.cu, the allocator's free memory filled with NaN before each, dbias allocated afresh (the bound launch
    rebuilt every fifth run): finite and bit-identical."""
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR", "1")
    H, DH = n_head, d_head
    do, v, dd, P = _dpb_case(L, A, H, DH, seed=7)
    first, op = None, None
    for i in range(20):
        if i % 5 == 0:
            op = C.DpbKernel(torch.cuda.current_device(), nh=H, dh=DH)
        _poison()
        db = torch.empty(H * L, L, device="cuda", dtype=torch.bfloat16)
        op(do, v, P.view(H * L, L), dd, db, A)
        torch.cuda.synchronize()
        assert op.last == "pair"
        if first is None:
            first = db.clone()
            assert torch.isfinite(first).all()
        else:
            assert torch.equal(db, first), i


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "pair"), [(768, True), (512, False), (384, False)])
def test_dpb_pair_default_selection(monkeypatch, L, pair, n_head, d_head):
    """The default: CTA pairs where 256 divides L and the pairs' items fill the SM pairs -- L768 at the training lengths."""
    monkeypatch.delenv("MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR", raising=False)
    do, v, dd, P = _dpb_case(L, 2, n_head, d_head, seed=1)
    op = C.DpbKernel(torch.cuda.current_device(), nh=n_head, dh=d_head)
    db = torch.empty(n_head * L, L, device="cuda", dtype=torch.bfloat16)
    op(do, v, P.view(n_head * L, L), dd, db, 2)
    torch.cuda.synchronize()
    assert op.last == ("pair" if pair else "single"), op.last
