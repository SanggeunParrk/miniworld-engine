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
    assert not sm80.supports(torch.randn(2, 64, 64, 128, device="cuda", dtype=torch.bfloat16), 128)   # B > 1
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
def test_replay_is_bit_identical():
    module = _module("bidir")
    z, mask, ds, dy = _inputs(128)
    a = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    b = _run(module, z, mask, ds, dy, implementation=ImplementationType.MINIWORLD)
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]


@needs_ampere
@pytest.mark.parametrize("ch", [128, 256])
@pytest.mark.parametrize("train", [False, True])
def test_one_launch_pack_matches_its_torch_definition(ch, train):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    torch.manual_seed(5)
    w = [torch.randn(ch, 128, device="cuda").bfloat16() for _ in range(4)]
    wg = torch.randn(128, 128, device="cuda").bfloat16()
    wo = torch.randn(128, ch, device="cuda").bfloat16()
    ln = [1 + 0.1 * torch.randn(128, device="cuda"), 0.1 * torch.randn(128, device="cuda"),
          1 + 0.1 * torch.randn(ch, device="cuda"), 0.1 * torch.randn(ch, device="cuda")]
    got = sm80._pack(*w, wg, wo, *ln, train=train)
    want = sm80._pack_reference(*w, wg, wo, *ln, train=train)
    assert got.keys() == want.keys()
    for key, ref in want.items():
        if not isinstance(ref, torch.Tensor):
            assert got[key] == ref, key
        elif ref.dtype is torch.float32 and ref.ndim == 1 and key not in ("g_in", "b_in"):     # reductions: order differs
            torch.testing.assert_close(got[key], ref, rtol=1e-5, atol=1e-5, msg=key)
        else:
            assert torch.equal(got[key].reshape(ref.shape), ref), key
