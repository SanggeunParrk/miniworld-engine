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


FUSED_ALL = {"MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL": "split3", "MINIWORLD_BIAS_ONLY_DIT_BWD_MID": "1", "MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN": "1"}


@pytest.mark.parametrize("fusions", ["defaults", "all"])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_graph_capture_compile_and_steady_memory(monkeypatch, n_head, d_head, fusions):
    """Steady memory: after two warm-up steps every further step leaves memory_reserved unchanged. The first step of a process
    allocates for good in the middle of the step (the parameters' .grad, the backward thread's cuBLAS workspaces, the kernels' item
    tables), so the allocator's layout after it differs and the second step may add segments once (bwdf7, the first test of a fresh
    process: 40 MiB, with or without the fused kernels); a per-step growth would show from the third step on. "all": every
    fused-backward kernel on (the tail, bo_bwd_mid, pair_bias_bwd_fin; off by default)."""
    if fusions == "all":
        for k, v in FUSED_ALL.items():
            monkeypatch.setenv(k, v)
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

    torch.cuda.synchronize()
    r0 = torch.cuda.memory_reserved()
    eager = run(fast)
    torch.cuda.synchronize()
    r1 = torch.cuda.memory_reserved()
    run(fast)                                                                         # the second warm-up step
    # the bound launches reuse their argument blocks and the activations are allocated afresh each step: nothing may pile up
    torch.cuda.synchronize()
    r2 = torch.cuda.memory_reserved()
    print(f"memory_reserved: first step +{(r1 - r0) >> 20} MiB, second step +{(r2 - r1) >> 20} MiB")
    st0, packs0 = torch.cuda.memory_stats(), {k: id(v[1]) for k, v in TR._PACKS.items()}
    steps = []
    for _ in range(4):
        run(fast)
        torch.cuda.synchronize()
        steps.append(torch.cuda.memory_reserved() - r2)
    st1, packs1 = torch.cuda.memory_stats(), {k: id(v[1]) for k, v in TR._PACKS.items()}
    seg = {p: st1[f"segment.{p}_pool.current"] - st0[f"segment.{p}_pool.current"] for p in ("small", "large")}
    assert steps == [0, 0, 0, 0], ("growth after steps 3..6", steps, "new segments", seg, "pack rebuilt", packs0 != packs1,
                                   "second step", r2 - r1)
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
# (MINIWORLD_BIAS_ONLY_DIT_BWD_DG), the CTA-pair bias gradient (MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR) and the fused backward
# (MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED: bo_bwd_tail + bo_wgrad) are the defaults; each =0 restores the previous launches. The test
# script runs this file at the defaults and with all four at 0.
F64 = torch.float64
SWITCHES = ("MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1", "MINIWORLD_BIAS_ONLY_DIT_BWD_DG", "MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR",
            "MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED", "MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN", "MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB",
            "MINIWORLD_BIAS_ONLY_DIT_BWD_PVPDL", "MINIWORLD_BIAS_ONLY_DIT_BWD_MID")


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
    and the L768 bias gradient runs on CTA pairs; with the three switches at 0, the torch pack, separate dxt / dxa and dpb_sm100. The
    fused backward is pinned off here (it replaces res_adaln_b_bwd: test_fused_backward_* below)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import train as TRN
    for v in SWITCHES[:3]:
        if off:
            monkeypatch.setenv(v, "0")
        else:
            monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED", "0")
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


@pytest.mark.parametrize("mid", ["0", "1"])
@pytest.mark.parametrize("fp32_norms", [False, True])
def test_one_launch_weight_pack_matches_the_torch_pack(monkeypatch, fp32_norms, mid):
    """The one-launch pack (the default; MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1=0: the torch pack) gives the torch pack's tensors bit
    for bit (bf16 casts, the folded cond-LN and pair-LN weights, the concatenations, fp32 copies); with every parameter bf16 and with
    the LayerNorm weights in fp32 (the engine keeps them fp32 in a bf16 block)."""
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_BWD_MID", mid)                  # mid: + bo_bwd_mid's transposed weights
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
    from miniworld_engine.kernels.bias_only_dit.cuda import bwd_fused as BWF
    assert set(one) == set(ref) and ("WnT3" in ref) == BWF.mid_on(), sorted(set(one) ^ set(ref))   # + bo_bwd_mid's transposes
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


# ------------------------------------------------------------------------------------------------- fused backward (bo_bwd_tail, bo_wgrad)
# MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED (default on): the 23 launches of a block's bf16 backward become 11 at the defaults -- bo_bwd_tail.cu x 3
# (res_c_bwd, the dh GEMM + SwiGLU backward, the dxt GEMM, res_adaln_b_bwd, the d(og) GEMM + gate backward; _BWD_TAIL=off: the seven),
# bo_pvdpb.cu (pv dV + dbias; _BWD_PVDPB), the dxa GEMM / adaln_a_bwd / dchat / dcg / cond_bwd (bo_bwd_mid.cu x 3 with _BWD_MID=1: 9),
# bo_wgrad.cu (the six weight gradients and the cond-LN unfold), pair_bias_bwd_fin_k (pair_bias_bwd + finalize; _BWD_PBFIN=0: two).
def _sig(t):
    return torch.sigmoid(t)


def _tail_case(L, A, H, DH, seed):
    """Random inputs of bo_bwd_tail.cu at the step's shapes (row statistics of x1 the real ones) and its fp64 reference."""
    from miniworld_engine.kernels.bias_only_dit.cuda import bwd_fused as BWF
    D, DA, M, bf = 768, H * DH, A * L, torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(seed)
    rn = lambda *s, sc=1.0: (sc * torch.randn(*s, device="cuda", generator=g)).to(bf)
    dout, z, x, y = rn(M, D), rn(M, D), rn(M, D), rn(M, D)
    Gg, G, ab, og, vg = rn(M, 2 * D), rn(M, 4 * D), rn(M, 4 * D), rn(M, DA), rn(M, 2 * DA)
    bg2, bs2, bg1 = (0.1 * torch.randn(D, device="cuda", generator=g) for _ in range(3))
    Wsq, Wa, Wb, Wo = rn(D, 2 * D, sc=D ** -0.5), rn(2 * D, D, sc=D ** -0.5), rn(2 * D, D, sc=D ** -0.5), rn(D, DA, sc=D ** -0.5)
    g1s = _sig(Gg[:, :D].double() + bg1.double())
    x1 = (x.double() + (g1s.float() * y.float()).double())
    mean, var = x1.mean(1), x1.var(1, unbiased=False)
    x1st = torch.stack([mean, (var + 1e-5).rsqrt()], 1).float().contiguous()
    ins = {"dout": dout, "z": z, "Gg": Gg, "ab": ab, "x": x, "y": y, "G": G, "x1st": x1st, "og": og, "vg": vg, "bg2": bg2, "bs2": bs2, "bg1": bg1,
               "WsqT": Wsq.t().contiguous(), "WaT": Wa.t().contiguous(), "WbT": Wb.t().contiguous(), "WoT": Wo.t().contiguous()}
    d = lambda t: t.double()
    s2g = _sig(d(Gg[:, D:]) + d(bg2))
    dz = d(dout) * s2g
    dg2 = d(dout) * d(z) * s2g * (1 - s2g)
    dh = dz @ d(Wsq)
    a, b = d(ab[:, :2 * D]), d(ab[:, 2 * D:])
    sa = _sig(a)
    dab = torch.cat([dh * b * sa * (1 + a * (1 - sa)), dh * a * sa], 1)
    dxt = dab @ torch.cat([d(Wa), d(Wb)])
    g1s = _sig(d(Gg[:, :D]) + d(bg1))
    xh = (d(x) + g1s * d(y) - d(x1st[:, :1])) * d(x1st[:, 1:])
    s2 = _sig(d(G[:, 2 * D:3 * D]) + d(bs2))
    ds2 = dxt * xh * s2 * (1 - s2)
    dxh = dxt * s2
    dx1 = d(dout) + d(x1st[:, 1:]) * (dxh - dxh.mean(1, keepdim=True) - xh * (dxh * xh).mean(1, keepdim=True))
    dy = dx1 * g1s
    dg1 = dx1 * d(y) * g1s * (1 - g1s)
    dog = dy @ d(Wo)
    sgt = _sig(d(vg[:, DA:]))
    want = {"dz": dz, "dg2": dg2, "dab": dab, "dxt": dxt, "ds2": ds2, "dx1": dx1, "dy": dy, "dg1": dg1, "do": dog * sgt, "dg": dog * d(og) * (1 - sgt),
                "dd": (dog * d(og)).view(A, L, H, DH).sum(3).permute(0, 2, 1), "bg2": dg2.sum(0), "bs2": ds2.sum(0), "bg1": dg1.sum(0)}
    return ins, want, BWF


def _run_tail(op, ins, A, L, H, DA):
    D, M, bf, nan = 768, A * L, torch.bfloat16, float("nan")
    full = lambda *s, dt=bf: torch.full(s, nan, device="cuda", dtype=dt)
    T = M // 128
    out = {"dz": full(M, D), "dGg": full(M, 2 * D), "dab": full(M, 4 * D), "dG": full(M, 4 * D), "dx1": full(M, D, dt=torch.float32),
               "dy": full(M, D), "do": full(M, DA), "dvg": full(M, 2 * DA), "dd": full(A, H, L, dt=torch.float32),
               "part": full(4, max(8 * 148, 4 * T), D, dt=torch.float32)}
    n = op(ins["dout"], ins["z"], ins["Gg"], ins["ab"], ins["x"], ins["y"], ins["G"], ins["x1st"], ins["og"], ins["vg"],
           ins["bg2"], ins["bs2"], ins["bg1"], ins["WsqT"], ins["WaT"], ins["WbT"], ins["WoT"], out["dz"], out["dGg"], out["dab"],
           out["dG"], out["dx1"], out["dy"], out["do"], out["dvg"], out["dd"], out["part"], L)
    assert n == 4 * T
    return out


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "A"), [(128, 1), (384, 3), (768, 2), (256, 24)])
def test_fused_backward_tail_kernel_against_fp64(L, A, n_head, d_head):
    """bo_bwd_tail.cu alone on random inputs against fp64: every output it writes (a NaN-filled buffer shows a missing write), the
    column-sum partials, bit-identical on a rerun; no local memory."""
    H, DA, D = n_head, n_head * d_head, 768
    ins, want, BWF = _tail_case(L, A, H, d_head, L + A + H)
    op = BWF.TailBwd(torch.cuda.current_device(), H, d_head)
    assert op.k.lmem == 0 and op.k.regs <= 168, (op.k.regs, op.k.lmem)
    out = _run_tail(op, ins, A, L, H, DA)
    T = A * L // 128
    got = {"dz": out["dz"], "dg2": out["dGg"][:, D:], "dab": out["dab"], "dxt": out["dG"][:, 3 * D:], "ds2": out["dG"][:, 2 * D:3 * D],
               "dx1": out["dx1"], "dy": out["dy"], "dg1": out["dGg"][:, :D], "do": out["do"], "dg": out["dvg"][:, DA:], "dd": out["dd"],
               "bg2": out["part"][0, :4 * T].sum(0), "bs2": out["part"][1, :4 * T].sum(0), "bg1": out["part"][2, :4 * T].sum(0)}
    bounds = {"dz": 4e-3, "dg2": 6e-3, "dab": 1.5e-2, "dxt": 1.5e-2, "ds2": 2e-2, "dx1": 1.5e-2, "dy": 1.5e-2, "dg1": 2e-2, "do": 2e-2, "dg": 2e-2, "dd": 2e-2,
                  "bg2": 6e-3, "bs2": 2e-2, "bg1": 2e-2}
    for k, w in want.items():
        assert torch.isfinite(got[k].float()).all(), k
        assert relative64(got[k], w) < bounds[k], (k, relative64(got[k], w))
    again = _run_tail(op, ins, A, L, H, DA)
    for k in ("dz", "dGg", "dab", "dx1", "dy", "do", "dd"):
        assert torch.equal(again[k], out[k]), k
    assert torch.equal(again["dG"][:, 2 * D:], out["dG"][:, 2 * D:]) and torch.equal(again["dvg"][:, DA:], out["dvg"][:, DA:])
    assert torch.equal(again["part"][:3, :4 * T], out["part"][:3, :4 * T])


@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize("M", [128, 384, 3072, 18432])
@pytest.mark.parametrize("f32", [False, True])
def test_fused_backward_wgrad_kernel_against_fp64(da, M, f32):
    """bo_wgrad.cu alone: the six weight gradients (bf16 or fp32 out) and the unfold (dWraw = dWn w, the cond-LN partials over 16
    rows) against fp64; bit-identical on a rerun; no local memory."""
    from miniworld_engine.kernels.bias_only_dit.cuda import bwd_fused as BWF
    D, DC, bf = 768, 384, torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(M + da)
    rn = lambda *s: torch.randn(*s, device="cuda", generator=g).to(bf)
    grads = (rn(M, D), rn(M, 4 * D), rn(M, D), rn(M, 2 * da), rn(M, 4 * D), rn(M, 2 * D))
    acts = (rn(M, 2 * D), rn(M, D), rn(M, da), rn(M, D), rn(M, DC), rn(M, DC))
    odt = torch.float32 if f32 else bf
    shapes = ((D, 2 * D), (4 * D, D), (D, da), (2 * da, D), (2 * D, DC))
    wraw = torch.randn(4 * D, DC, device="cuda", generator=g)
    w1, w2 = 1 + 0.1 * torch.randn(DC, device="cuda", generator=g), 1 + 0.1 * torch.randn(DC, device="cuda", generator=g)
    op = BWF.Wgrad(torch.cuda.current_device(), da)
    assert op.k.lmem == 0 and op.k.regs <= 168, (op.k.regs, op.k.lmem)

    def run():
        outs = tuple(torch.full(s, float("nan"), device="cuda", dtype=odt) for s in shapes)
        dwu = torch.full((4 * D, DC), float("nan"), device="cuda", dtype=odt)
        pw = torch.full((192, DC), float("nan"), device="cuda")
        op(grads, acts, outs, dwu, pw, wraw, w1, w2)
        return outs, dwu, pw
    outs, dwu, pw = run()
    ref = [grads[i].double().t() @ acts[i].double() for i in range(6)]
    for i, o in zip((0, 1, 2, 3, 5), outs, strict=True):
        assert torch.isfinite(o.float()).all(), i
        assert relative64(o, ref[i]) < 6e-3, (i, relative64(o, ref[i]))
    dwn = ref[4]
    w = torch.cat([w1.double().expand(2 * D, DC), w2.double().expand(2 * D, DC)])
    assert relative64(dwu, dwn * w) < 6e-3
    assert relative64(pw, (dwn * wraw.double()).view(192, 16, DC).sum(1)) < 1e-3
    o2, dwu2, pw2 = run()
    assert all(torch.equal(a, b) for a, b in zip(outs, o2, strict=True)) and torch.equal(dwu, dwu2) and torch.equal(pw, pw2)


@pytest.mark.parametrize("nh", [12, 16, 24])
@pytest.mark.parametrize("L", [128, 384, 768])
@pytest.mark.parametrize("odt", [torch.bfloat16, torch.float32])
def test_fused_backward_pbfin_kernel_against_two_launches(nh, L, odt):
    """pair_bias_bwd_fin_k (MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN) against pair_bias_bwd + finalize on the same inputs: d pair
    bit-identical (the same body), the finalize outputs equal up to the order of the fp32 partial sums, every output written
    (NaN-filled), the barrier counters back at zero, bit-identical over five reruns (the counters reused)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import bwd_fused as BWF
    from miniworld_engine.kernels.bias_only_dit.cuda import train as TRN
    BT = TRN.ext()
    D, DC, R, M, bf = 768, 384, L * L, 4 * L, torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(L + nh)
    rn = lambda *s: torch.randn(*s, device="cuda", generator=g)
    z, wf = rn(R, 128).to(bf), (0.1 * rn(nh, 128)).to(bf)
    pst = torch.empty(R, 2, device="cuda")
    BT.pair_bias_cuda(z, wf, torch.empty(nh * R, device="cuda", dtype=bf), pst, 1e-5)
    db = rn(nh * R).to(bf)
    rows = max(BT.partial_rows(M), 4 * (M // 128))
    part, pw, wb, wp = rn(4, rows, D), rn(192, DC), rn(nh, 128), rn(128)
    n = [4 * (M // 128)] * 3 + [M // 16]
    nan = float("nan")

    def outs():
        return (torch.full((R, 128), nan, device="cuda", dtype=bf), torch.full((BT.partial_rows(M), nh, 128), nan, device="cuda"),
                torch.full((4 * D + nh * 128,), nan, device="cuda", dtype=odt), torch.full((2 * DC + 128,), nan, device="cuda", dtype=odt))
    dz0, pwf0, out0, nout0 = outs()
    nwf = BT.pair_bias_bwd_cuda(db, z, pst, wf, dz0, pwf0)
    BT.finalize_cuda(part, n, pw, pwf0, nwf, wb, wp, out0, nout0)
    bar = BWF.pb_bar(torch.device("cuda"))
    runs = []
    for _ in range(5):
        dz, pwf, out, nout = outs()
        BT.pair_bias_bwd_fin_cuda(db, z, pst, wf, dz, pwf, part, n, pw, wb, wp, out, nout, bar)
        torch.cuda.synchronize()
        assert bar.tolist() == [0, 0]
        runs.append((dz, out, nout))
    dz, out, nout = runs[0]
    assert torch.equal(dz, dz0)
    tol = 4e-3 if odt == bf else 1e-5
    for a, b in ((out, out0), (nout, nout0)):
        assert torch.isfinite(a.float()).all()
        assert relative64(a, b) < tol, relative64(a, b)
    for r in runs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(r, runs[0], strict=True))


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "A"), [(128, 8), (384, 3), (512, 2), (768, 4), (768, 48)])
def test_fused_backward_pvdpb_kernel_bit_identical_to_two_launches(monkeypatch, L, A, n_head, d_head):
    """bo_pvdpb.cu (MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB) against PvGateCore (no gate) + DpbKernel (dpb_sm100) on the same inputs: the
    same bodies on the same virtual grids, so dV and dbias are bit-identical (and every element written: NaN-filled outputs), also
    on reruns. Where the bias gradient belongs on CTA pairs (L768) the op launches nothing and returns False; ``single`` forces the
    fused launch there (compared with DpbKernel at MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR=0)."""
    from miniworld_engine.kernels.bias_only_dit import cuda as C
    pair = C.dpb_pair_on(L, n_head, 2 if d_head == 32 else 1, torch.cuda.get_device_properties(0).multi_processor_count)
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR", "0")
    H, DA, M, bf = n_head, n_head * d_head, A * L, torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(L + A + H)
    do = torch.randn(M, DA, device="cuda", generator=g).to(bf)
    vg = torch.randn(M, 2 * DA, device="cuda", generator=g).to(bf)
    P = torch.softmax(torch.randn(H, L, L, device="cuda", generator=g), -1)
    Pb, Ptb = P.to(bf).view(H * L, L), P.transpose(1, 2).contiguous().to(bf).view(H * L, L)
    dd = torch.randn(A, H, L, device="cuda", generator=g)
    dev = torch.cuda.current_device()
    pv, dpb, both = C.PvGateCore(dev, nh=H, dh=d_head), C.DpbKernel(dev, nh=H, dh=d_head), C.PvDpb(dev, nh=H, dh=d_head, single=pair)
    nan = float("nan")
    dv0, db0 = torch.full((M, 2 * DA), nan, device="cuda", dtype=bf), torch.full((H * L, L), nan, device="cuda", dtype=bf)
    pv(do, Ptb, dv0[:, :DA], A)
    dpb(do, vg[:, :DA], Pb, dd, db0, A)
    for _ in range(3):
        dv, db = torch.full((M, 2 * DA), nan, device="cuda", dtype=bf), torch.full((H * L, L), nan, device="cuda", dtype=bf)
        assert both(do, Ptb, dv[:, :DA], vg[:, :DA], Pb, dd, db, A)
        torch.cuda.synchronize()
        assert dpb.last == "single"
        assert torch.isfinite(dv[:, :DA].float()).all() and torch.isfinite(db.float()).all()
        assert torch.equal(dv[:, :DA], dv0[:, :DA]) and torch.equal(db, db0)
        assert dv[:, DA:].isnan().all()                                                 # the gate half untouched
    if pair:                                                                            # at the step's default the op declines
        monkeypatch.delenv("MINIWORLD_BIAS_ONLY_DIT_DPB_PAIR")
        dv = torch.full((M, 2 * DA), nan, device="cuda", dtype=bf)
        assert not C.PvDpb(dev, nh=H, dh=d_head)(do, Ptb, dv[:, :DA], vg[:, :DA], Pb, dd, db, A)
        torch.cuda.synchronize()
        assert dv.isnan().all()


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "A"), [(128, 1), (384, 3), (768, 2), (256, 24)])
def test_fused_backward_mid_kernel_against_fp64(L, A, n_head, d_head):
    """bo_bwd_mid.cu (MINIWORLD_BIAS_ONLY_DIT_BWD_MID) alone on random inputs against fp64: dxa and ds1 in dG, dx, the dcg scratch, dc
    and the bs1 partials (NaN-filled outputs show a missing write), dG's transition half untouched; bit-identical on a rerun; no
    local memory."""
    from miniworld_engine.kernels.bias_only_dit.cuda import bwd_fused as BWF
    H, DA, D, DC, M, bf, nan = n_head, n_head * d_head, 768, 384, A * L, torch.bfloat16, float("nan")
    T = M // 128
    g = torch.Generator(device="cuda").manual_seed(L + A + H + 1)
    rn = lambda *s, sc=1.0: (sc * torch.randn(*s, device="cuda", generator=g)).to(bf)        # noqa: E731
    dvg, dGg, G, x, c = rn(M, 2 * DA), rn(M, 2 * D), rn(M, 4 * D), rn(M, D), rn(M, DC)
    dGt = rn(M, 2 * D)
    dx1 = torch.randn(M, D, device="cuda", generator=g)
    bs1 = 0.1 * torch.randn(D, device="cuda", generator=g)
    stat = lambda t: torch.stack([t.double().mean(1), (t.double().var(1, unbiased=False) + 1e-5).rsqrt()], 1).float().contiguous()  # noqa: E731
    xst, cst = stat(x), stat(c)
    Wv, Wgt = rn(DA, D, sc=D ** -0.5), rn(DA, D, sc=D ** -0.5)
    Ws0, Ws1, Wn = rn(D, DC, sc=DC ** -0.5), rn(D, DC, sc=DC ** -0.5), rn(4 * D, DC, sc=DC ** -0.5)
    W = {"WvT": Wv.t().contiguous(), "WgtT": Wgt.t().contiguous(), "WsT0": Ws0.t().contiguous(), "WsT1": Ws1.t().contiguous(),
         **{f"WnT{i}": Wn[i * D:(i + 1) * D].t().contiguous() for i in range(4)}}
    op = BWF.MidBwd(torch.cuda.current_device(), H, d_head)
    assert op.k.lmem == 0 and op.k.regs <= 168, (op.k.regs, op.k.lmem)

    def run():
        dG = torch.full((M, 4 * D), nan, device="cuda", dtype=bf)
        dG[:, 2 * D:] = dGt
        out = {"dG": dG, "dcg": torch.full((M, DC), nan, device="cuda", dtype=bf), "dx": torch.full((M, D), nan, device="cuda", dtype=bf),
               "dc": torch.full((M, DC), nan, device="cuda", dtype=bf),
               "part": torch.full((max(8 * 148, 4 * T), D), nan, device="cuda")}
        n = op(dvg, dGg, dG, G, x, xst, dx1, c, cst, W, bs1, out["dcg"], out["dx"], out["dc"], out["part"])
        assert n == 4 * T
        return out
    out = run()
    d = lambda t: t.double()                                                                  # noqa: E731
    dxa = d(dvg) @ torch.cat([d(Wv), d(Wgt)])
    xh = (d(x) - d(xst[:, :1])) * d(xst[:, 1:])
    s1 = torch.sigmoid(d(G[:, :D]) + d(bs1))
    ds1 = dxa * xh * s1 * (1 - s1)
    dxh = dxa * s1
    dx = d(dx1) + d(xst[:, 1:]) * (dxh - dxh.mean(1, keepdim=True) - xh * (dxh * xh).mean(1, keepdim=True))
    dchat = torch.cat([ds1, dxa, d(dGt)], 1) @ d(Wn)
    dcg = d(dGg) @ torch.cat([d(Ws0), d(Ws1)])
    ch = (d(c) - d(cst[:, :1])) * d(cst[:, 1:])
    dc = d(cst[:, 1:]) * (dchat - dchat.mean(1, keepdim=True) - ch * (dchat * ch).mean(1, keepdim=True)) + dcg
    got = {"dxa": out["dG"][:, D:2 * D], "ds1": out["dG"][:, :D], "dx": out["dx"], "dcg": out["dcg"], "dc": out["dc"],
           "bs1": out["part"][:4 * T].sum(0)}
    want = {"dxa": dxa, "ds1": ds1, "dx": dx, "dcg": dcg, "dc": dc, "bs1": ds1.sum(0)}
    bounds = {"dxa": 1.5e-2, "ds1": 2e-2, "dx": 1.5e-2, "dcg": 6e-3, "dc": 2e-2, "bs1": 2e-2}
    for k, w in want.items():
        assert torch.isfinite(got[k].float()).all(), k
        assert relative64(got[k], w) < bounds[k], (k, relative64(got[k], w))
    assert torch.equal(out["dG"][:, 2 * D:], dGt)
    again = run()
    for k in ("dG", "dcg", "dx", "dc"):
        assert torch.equal(again[k], out[k]), k
    assert torch.equal(again["part"][:4 * T], out["part"][:4 * T])


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("fused", "tail", "pbfin", "pvdpb", "mid"), [(None, None, None, None, None), ("1", "split3", "1", "1", "1"),
                                                                      ("1", "split3", "0", "0", "0"), ("1", "off", "1", "0", "1"),
                                                                      ("1", "split3", "1", "0", "0"), ("0", "split3", "1", "1", "1")])
def test_fused_backward_launch_count(monkeypatch, n_head, d_head, fused, tail, pbfin, pvdpb, mid):
    """The captured backward of one block (``_bwd`` under static weights: no repack) at L384: 11 kernel launches at the defaults
    (bo_bwd_tail x 3, bo_pvdpb, the five of dxa / adaln_a_bwd / dchat / dcg / cond_bwd, bo_wgrad, pair_bias_bwd_fin); 9 with
    bo_bwd_mid on (_BWD_MID=1); the per-step path launches 23 (BWD_FUSED=0)."""
    from miniworld_engine.kernels import _capture
    from tests.cuda_graph_nodes import graph_kernels
    from miniworld_engine.kernels.bias_only_dit.cuda import bwd_fused as BWF
    for k, v in (("FUSED", fused), ("TAIL", tail), ("PBFIN", pbfin), ("PVDPB", pvdpb), ("MID", mid)):
        if v is None:                                                               # the defaults
            monkeypatch.delenv(f"MINIWORLD_BIAS_ONLY_DIT_BWD_{k}", raising=False)
        else:
            monkeypatch.setenv(f"MINIWORLD_BIAS_ONLY_DIT_BWD_{k}", v)
    _, fast, _ = blocks(seed=8, n_head=n_head, d_head=d_head)
    L, A = 384, 4
    x, c, p, dy, mask = inputs(L, A, True, seed=8)
    x, c, p, dy = (t.bfloat16() for t in (x, c, p, dy))
    params = [fast.get_parameter(n) for n in TR.NAMES]
    with _capture.static_weights(), torch.no_grad():
        out, *saved = TR._fwd(x, c, p, mask, params)
        TR._bwd(x, c, p, mask, params, saved, dy)                                     # eager: packs, tables, counters
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            got = TR._bwd(x, c, p, mask, params, saved, dy)
        names = graph_kernels(graph)
        graph.replay()
        torch.cuda.synchronize()
        eager = TR._bwd(x, c, p, mask, params, saved, dy)
    for a, b in zip(got, eager, strict=True):                                         # cuBLAS may pick another algorithm under capture
        assert relative(a, b) < 2e-3
    if BWF.bwd_fused_on():
        # pair_bias_bwd_fin_k, bo_pvdpb: one launch for two (L384: the single-CTA bias gradient); bo_bwd_mid 3 for 5; the tail 3 for 7
        one, pd, md, tl = BWF.pbfin_on(), BWF.pvdpb_on(), BWF.mid_on(), BWF.tail_mode() == "split3"
        ours = {"bo_bwd_tail": 3 * tl, "res_c_bwd": int(not tl), "swiglu_bwd": int(not tl), "res_adaln_b_bwd": int(not tl),
                "gate_bwd": int(not tl), "bo_pvdpb": int(pd), "pv_gate": int(not pd), "bo_dpb": int(not pd), "bo_bwd_mid": 3 * md,
                "adaln_a_bwd": int(not md), "cond_bwd": int(not md), "bo_wgrad": 1, "pair_bias_bwd": 1, "pair_bias_bwd_fin": int(one),
                "finalize": int(not one)}
        assert all(sum(k in n for n in names) == c for k, c in ours.items()), names
        assert len(names) == 13 + 4 * (not tl) - one - pd - 2 * md, names
    else:
        assert not any("bo_bwd_tail" in n or "bo_wgrad" in n for n in names), names
        assert len(names) >= 20, names
