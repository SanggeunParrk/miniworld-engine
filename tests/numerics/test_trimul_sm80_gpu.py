"""The A100 (sm_80) hand-CUDA TriMul matches the fp32 PyTorch module at least as closely as the bf16 PyTorch module does, in
inference and training (output, dz and all ten parameter gradients, with the dropout row scale and a token mask), for both
single directions and the bidirectional block; the modules really dispatch to it; the gate only takes what it is built for."""
import copy

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
AMPERE = CUDA and torch.cuda.get_device_capability() == (8, 0)
needs_ampere = pytest.mark.skipif(not AMPERE, reason="the sm80 TriMul is sm_80 only")

P_DROP = 0.25


def _module(kind, seed=1234):
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    torch.manual_seed(seed)
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(128, implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    else:
        m = TriangleMultiplication(128, outgoing=kind == "outgoing", implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    m = m.cuda().bfloat16()
    with torch.no_grad():                        # the zero-initialised gates would make most gradients exactly zero
        for name, t in m.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -0.5)
            elif "weight" in name:
                t.copy_(1 + 0.1 * torch.randn_like(t))
            else:
                t.normal_(std=0.05)
    return m


def _inputs(n, seed=90323):
    torch.manual_seed(seed)
    z = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, n, device="cuda") > 0.1
    ds = ((torch.rand(1, 1, n, 128, device="cuda") > P_DROP).float() / (1 - P_DROP)).to(torch.bfloat16)
    dy = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16)
    return z, mask, ds, dy


def _run(module, z, mask, ds, dy, *, implementation, fp32=False, train=True):
    m = copy.deepcopy(module)
    from miniworld_engine.modules.dispatch import resolve_triangle_multiplication

    m.implementation, m._backend = implementation, resolve_triangle_multiplication(implementation)
    if fp32:
        m, z, ds, dy = m.float(), z.float(), ds.float(), dy.float()
    m.train(train)
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)        # the same row scale on every path
    zz = z.clone().requires_grad_(train)
    with torch.set_grad_enabled(train):
        y = m(zz, mask)
    res = {"out": y.detach().float()}
    if train:
        y.backward(dy)
        res["dz"] = zz.grad.detach().float()
        res.update({name: p.grad.detach().float() for name, p in m.named_parameters()})
    return res


def _rel(got, want):
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    z = torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 64, device="cuda", dtype=torch.bool)
    assert sm80.supports(z, 128, mask) is AMPERE
    assert not sm80.supports(z.float(), 128)
    assert not sm80.supports(z, 64)                                                             # d_hidden != 128
    assert not sm80.supports(torch.randn(1, 64, 64, 64, device="cuda", dtype=torch.bfloat16), 64)
    batch = torch.randn(2, 64, 64, 128, device="cuda", dtype=torch.bfloat16)
    assert sm80.supports(batch, 128, torch.ones(2, 64, device="cuda", dtype=torch.bool)) is AMPERE        # a batch: the integration runs it plane by plane
    assert not sm80.supports(batch, 128, mask)                                                  # a [1, L] mask for B = 2
    assert not sm80.supports(torch.randn(1, 40, 40, 128, device="cuda", dtype=torch.bfloat16), 128)   # L % 16
    assert not sm80.supports(z, 128, mask.float())
    assert not sm80.supports(z.cpu(), 128)


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    monkeypatch.setenv("MINIWORLD_TRIMUL_SM80", "0")
    assert not sm80.supports(torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.bfloat16), 128)


KINDS = ["bidir", "outgoing", "incoming"]


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [64, 384])
def test_training_is_no_less_accurate_than_the_bf16_module(kind, n):
    module = _module(kind)
    z, mask, ds, dy = _inputs(n)
    ref = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH, fp32=True)
    bf16 = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH)
    got = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    for name in ref:
        mine, base = _rel(got[name], ref[name]), _rel(bf16[name], ref[name])
        assert mine <= max(1.25 * base, 2e-3), f"{name}: sm80 {mine:.3e} vs bf16 module {base:.3e}"


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [64, 384])
def test_inference_is_no_less_accurate_than_the_bf16_module(kind, n):
    module = _module(kind)
    z, mask, ds, dy = _inputs(n)
    ref = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH, fp32=True, train=False)
    bf16 = _run(module, z, mask, ds, dy, implementation=ImplementationType.PYTORCH, train=False)
    got = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD, train=False)
    mine, base = _rel(got["out"], ref["out"]), _rel(bf16["out"], ref["out"])
    assert mine <= max(1.25 * base, 2e-3), f"sm80 {mine:.3e} vs bf16 module {base:.3e}"


@needs_ampere
@pytest.mark.parametrize("kind", ["bidir", "outgoing"])
def test_the_modules_actually_dispatch_to_it(kind, monkeypatch):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    calls = []
    original = sm80.trimul
    monkeypatch.setattr(sm80, "trimul", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    module = _module(kind)
    z, mask, ds, dy = _inputs(64)
    _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD, train=False)
    assert len(calls) == 2, "the sm80 path was not entered for both training and inference"


@needs_ampere
def test_no_mask_matches_an_all_true_mask():
    module = _module("bidir")
    z, mask, ds, dy = _inputs(64)
    full = torch.ones_like(mask)
    a = _run(module, z, None, ds, dy, implementation=ImplementationType.MINIWORLD)
    b = _run(module, z, full, ds, dy, implementation=ImplementationType.MINIWORLD)
    assert all(torch.equal(a[k], b[k]) for k in a)


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
def test_the_residual_is_the_kernels_own(kind, monkeypatch):
    """The module returns ``pair + drop_row(trimul(pair))`` and a block never adds the residual itself
    (``tests/compile/test_modules_own_their_residual.py``). A freshly initialised TriMul has a zero output projection, so its update
    is exactly zero: the output is the input bit for bit and the input's gradient exactly one (two would mean the residual is added
    twice, which is what a caller still writing ``x = x + module(x)`` gets)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    calls = []
    original = sm80.trimul
    monkeypatch.setattr(sm80, "trimul", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    torch.manual_seed(7)
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(128, implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    else:
        m = TriangleMultiplication(128, outgoing=kind == "outgoing", implementation=ImplementationType.MINIWORLD, p_drop=P_DROP)
    m = m.cuda().bfloat16().train()
    z = torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.rand(1, 64, device="cuda") > 0.1
    out = m(z, mask)
    assert calls, "the sm80 path was not entered"
    assert torch.equal(out, z)
    out.sum().backward()
    assert z.grad is not None
    assert torch.equal(z.grad, torch.ones_like(z))


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
def test_fp32_parameters_get_fp32_gradients_equal_to_the_bf16_ones(kind):
    """Mixed precision keeps fp32 master parameters under bf16 activations: the gradients come back in the parameters' dtype and layout
    (the LayerNorm ones fp32 from the fused finalize), equal to those of the same values held in bf16."""
    module = _module(kind)
    z, mask, ds, dy = _inputs(64)
    want = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    m = copy.deepcopy(module).float().train()                   # the same values in fp32; activations stay bf16
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    zz = z.clone().requires_grad_(True)
    m(zz, mask).backward(dy)
    for name, p in m.named_parameters():
        assert p.grad.dtype is torch.float32, name
        if p.numel() > 64:                                      # the kernels' fp32 accumulators, never rounded to bf16 (the cast is outside autograd)
            assert not torch.equal(p.grad, p.grad.bfloat16().float()), name
        assert _rel(p.grad.float(), want[name]) < 4e-3, name    # the bf16 parameters' gradient is this one rounded once (~2^-9 per element)


@needs_ampere
def test_replay_is_bit_identical():
    module = _module("bidir")
    z, mask, ds, dy = _inputs(128)
    a = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    b = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]


@needs_ampere
@pytest.mark.parametrize("ch", [128, 256])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("front", ["row_major", "in_out"])
def test_one_launch_pack_matches_its_torch_definition(ch, train, front):
    """``in_out``: the four front matrices stored [in, out] (strides 1, CH) as the bidirectional module does, read in place."""
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    torch.manual_seed(5)
    w = [torch.randn(ch, 128, device="cuda").bfloat16() for _ in range(4)]
    if front == "in_out":
        w = [t.t().contiguous().t() for t in w]
        assert all(t.stride() == (1, ch) for t in w)
    wg = torch.randn(128, 128, device="cuda").bfloat16()
    wo = torch.randn(128, ch, device="cuda").bfloat16()
    ln = [1 + 0.1 * torch.randn(128, device="cuda"), 0.1 * torch.randn(128, device="cuda"),
          1 + 0.1 * torch.randn(ch, device="cuda"), 0.1 * torch.randn(ch, device="cuda")]
    (wl, wlg, wr, wrg), (gi, bi, go, bo) = w, ln
    got = sm80._pack(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, train=train)
    want = sm80._pack_reference(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, train=train)
    assert got.keys() == want.keys()
    for key, ref in want.items():
        if not isinstance(ref, torch.Tensor):
            assert got[key] == ref, key
        elif ref.dtype is torch.float32 and ref.ndim == 1 and key not in ("g_in", "b_in"):     # reductions: order differs
            torch.testing.assert_close(got[key], ref, rtol=1e-5, atol=1e-5, msg=key)
        else:
            assert torch.equal(got[key].reshape(ref.shape), ref), key


@needs_ampere
@pytest.mark.parametrize("ch", [128, 256])
@pytest.mark.parametrize("front", ["row_major", "in_out"])
@pytest.mark.parametrize("ln_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("groups", "parts"), [(7, 42), (27, 216), (1, 1)])      # B7 joint, B7src + B8, a single partial
def test_finalize_sums_the_partials_and_undoes_the_k1_row_order(ch, front, ln_dtype, groups, parts):
    """One launch for the end of the backward: the fp32 partials of B7 summed in a fixed order, the K1 row order undone through the same
    table the pack uses, every gradient written in its parameter's strides, the LayerNorm gradients in the parameters' dtype."""
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    torch.manual_seed(11)
    dwp = torch.randn(groups, 4 * ch, 128, device="cuda")
    part = torch.randn(parts, 2, 128, device="cuda")
    rows = sm80._k1_rows(ch, dwp.device)[2]                       # left gate, left, right gate, right
    total = dwp.double().sum(0)                                   # [4 CH, 128], K1 row order
    want = [total[rows[1]], total[rows[0]], total[rows[3]], total[rows[2]]]   # W_l, W_lg, W_r, W_rg
    leaves = [torch.empty(ch, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    if front == "in_out":
        leaves = [t.t().contiguous().t() for t in leaves]
    got = [sm80._grad_like(t) for t in leaves]
    d_gi, d_bi = (torch.empty(128, device="cuda", dtype=ln_dtype) for _ in range(2))
    sm80._ext().finalize(dwp, part, *got, d_gi, d_bi)
    for g, leaf, w in zip(got, leaves, want, strict=True):
        assert g.stride() == leaf.stride()
        torch.testing.assert_close(g.double(), w, rtol=2 ** -7, atol=1e-4)           # within a bf16 rounding of the exact sum
    lnsum = part.double().sum(0)
    for g, w in zip((d_gi, d_bi), lnsum, strict=True):
        torch.testing.assert_close(g.double(), w, rtol=2 ** -7 if ln_dtype is torch.bfloat16 else 1e-5, atol=1e-4)


@needs_ampere
@pytest.mark.parametrize("kind", KINDS)
def test_training_under_torch_compile_matches_eager(kind):
    """A compiled caller checks every output of the opaque backward against its fake: the harness compiles the module, and a
    gradient laid out other than as ``empty_like(leaf)`` (the fake) failed there with a stride assertion, which no eager test sees."""
    module = _module(kind)
    n = 128
    z, mask, ds, dy = _inputs(n)
    eager = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)

    m = copy.deepcopy(module).train()
    m._make_drop_row_scale = lambda pair, p: ds.to(pair.dtype)
    zz = z.clone().requires_grad_(True)
    torch.compile(m, dynamic=False)(zz, mask).backward(dy)
    assert torch.equal(zz.grad.float(), eager["dz"])
    for name, p in m.named_parameters():
        assert _rel(p.grad.float(), eager[name]) < 1e-3, name


@needs_ampere
@pytest.mark.parametrize("kind", ["bidir", "outgoing"])
def test_a_batch_runs_plane_by_plane_and_equals_the_planes_run_alone(kind):
    """B = 2: the integration runs the kernels on one square plane per sample; the outputs, dz and the summed parameter gradients are those of the planes run
    alone (every kernel is deterministic, so the outputs and dz are bit-identical)."""
    n = 64
    module = _module(kind)
    torch.manual_seed(5)
    z = torch.randn(2, n, n, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(2, n, device="cuda") > 0.1
    ds = ((torch.rand(2, 1, n, 128, device="cuda") > P_DROP).float() / (1 - P_DROP)).to(torch.bfloat16)
    dy = torch.randn(2, n, n, 128, device="cuda", dtype=torch.bfloat16)
    both = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    singles = [_run(module, z[i:i + 1], mask[i:i + 1], ds[i:i + 1], dy[i:i + 1], implementation=ImplementationType.MINIWORLD) for i in range(2)]
    assert torch.equal(both["out"], torch.cat([s["out"] for s in singles]))
    assert torch.equal(both["dz"], torch.cat([s["dz"] for s in singles]))
    for name in both:
        if name not in ("out", "dz"):
            assert _rel(both[name], singles[0][name] + singles[1][name]) < 1e-2, name
