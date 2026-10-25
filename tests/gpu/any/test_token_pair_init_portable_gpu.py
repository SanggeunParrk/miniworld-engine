"""The portable kernels of the token-pair initialisation (``kernels/token_pair_init/triton``) on any card: output and every gradient against the dense fp32
reference (the 139-wide one-hot Linear the model ran), ragged lengths, B > 1, bf16 streams, other r_max / s_max, bitwise determinism of the backward (also across every
candidate launch configuration), CUDA-graph capture. The A/B switch puts every call on the portable kernels (H100 / B200 included); the comparison with the CUDA kernels is in
``tests/gpu/sm90/test_pair_ops_portable_vs_cuda_gpu.py``."""
from __future__ import annotations

import pytest
import torch
from settings_helpers import forced_triton as portable_switch   # the internal A/B switch: the one place the tests name it

from miniworld_engine.kernels.token_pair_init import refusal, token_pair_init, token_pair_init_reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
DEV, P = "cuda", 128


@pytest.fixture(autouse=True)
def _exact_fp32_reference():
    """The dense reference is an fp32 Linear: pin full precision (TF32 would round its weights) and restore it."""
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(previous[0])
    torch.backends.cuda.matmul.allow_tf32 = previous[1]


def _case(b, l, seed=0, r_max=32, s_max=2, grad=True):
    g = torch.Generator().manual_seed(seed)
    n_rel = 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1

    def ids(hi, dtype=torch.int64):
        return torch.randint(0, hi, (b, l), generator=g).to(DEV, dtype)

    asym, resi, tok, ent, sym = ids(3), ids(90), ids(60), ids(2), ids(6, torch.int32)
    bond = (torch.rand(b, l, l, generator=g) < 0.05).to(DEV)
    left, right = torch.randn(b, l, P, generator=g).to(DEV), torch.randn(b, l, P, generator=g).to(DEV)
    w_rel, w_bond = (torch.randn(P, n_rel, generator=g) * 0.3).to(DEV), (torch.randn(P, 2, generator=g) * 0.3).to(DEV)
    leaves = [t.requires_grad_(grad) for t in (left, right, w_rel, w_bond)]
    return leaves, (asym, resi, tok, ent, sym), bond


def _run(fn, leaves, ids, bond, **kw):
    left, right, w_rel, w_bond = leaves
    asym, resi, tok, ent, sym = ids
    return fn(left, right, w_rel, w_bond, asym, resi, tok, ent, sym, bond, **kw)


def _rel(a, e):
    return float((a.double() - e.double()).norm() / e.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize(("b", "l", "kw"), [(1, 384, {}), (2, 130, {}), (1, 7, {}), (1, 1, {}), (3, 64, {}), (2, 50, dict(r_max=8, s_max=1)), (48, 24, {})])
def test_matches_the_dense_reference(b, l, kw):
    (left, right, w_rel, w_bond), ids, bond = _case(b, l, **({"r_max": kw["r_max"], "s_max": kw["s_max"]} if kw else {}))
    with portable_switch():
        assert refusal(left, right, w_rel, w_bond, bond, **kw) is None
        z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bond, **kw)
    ze = _run(token_pair_init_reference, (left, right, w_rel, w_bond), ids, bond, **kw)
    assert z.shape == ze.shape and z.dtype == torch.float32
    assert _rel(z, ze) < 1e-6, _rel(z, ze)
    dz = torch.randn_like(ze)
    with portable_switch():
        got = torch.autograd.grad(z, (left, right, w_rel, w_bond), dz)
    exp = torch.autograd.grad(ze, (left, right, w_rel, w_bond), dz)
    for name, a, e in zip(("left", "right", "w_rel", "w_bond"), got, exp, strict=True):
        assert a.shape == e.shape, name
        assert _rel(a, e) < 1e-5, (name, _rel(a, e))


def test_bond_dtypes_and_bf16_streams():
    (left, right, w_rel, w_bond), ids, bond = _case(1, 96, seed=1, grad=False)
    ze = _run(token_pair_init_reference, (left, right, w_rel, w_bond), ids, bond)
    with portable_switch():
        for bd in (bond, bond.to(torch.uint8), bond.long(), bond.float()):
            assert _rel(_run(token_pair_init, (left, right, w_rel, w_bond), ids, bd), ze) < 1e-6
        zb = _run(token_pair_init, (left.bfloat16(), right.bfloat16(), w_rel.bfloat16(), w_bond.bfloat16()), ids, bond)
    zr = _run(token_pair_init_reference, (left.bfloat16().float(), right.bfloat16().float(), w_rel.bfloat16().float(), w_bond.bfloat16().float()), ids, bond)
    assert zb.dtype == torch.float32 and _rel(zb, zr) < 1e-6


def test_backward_is_bitwise_deterministic_and_reproduces_across_calls():
    for b, l in [(1, 384), (2, 130), (1, 257)]:
        (left, right, w_rel, w_bond), ids, bond = _case(b, l, seed=5)
        dz = torch.randn(b, l, l, P, device=DEV)
        with portable_switch():
            runs = []
            for _ in range(3):
                z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bond)
                runs.append((z, *torch.autograd.grad(z, (left, right, w_rel, w_bond), dz)))
        for r in runs[1:]:
            for a, e in zip(r, runs[0], strict=True):
                assert torch.equal(a, e)


def test_cuda_graph_capture():
    (left, right, w_rel, w_bond), ids, bond = _case(1, 128, seed=2)
    dz = torch.randn(1, 128, 128, P, device=DEV)

    def step():
        z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bond)
        return z, torch.autograd.grad(z, (left, right, w_rel, w_bond), dz)

    with portable_switch():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                       # eager runs and warm-up (the autotuner runs here) on the stream the capture will use
            eager = step()
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            z, grads = step()
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(z, eager[0])
    for a, e in zip(grads, eager[1], strict=True):
        assert torch.equal(a, e)


def _tuners():
    from miniworld_engine.kernels.token_pair_init.triton import main

    return {"fwd": main._tpi_fwd, "dw": main._tpi_dw, "reduce": main._tpi_dw_reduce, "dx": main._tpi_dx}


@pytest.mark.parametrize("which", ["fwd", "dw", "reduce", "dx"])
def test_every_candidate_config_is_correct_and_gives_the_same_bits(which, monkeypatch):
    """Each config of the default set alone gives the right numbers; the table-gradient kernels (the summation order does not depend on the tile) the same bits as every other."""
    tuner = _tuners()[which]
    configs = list(tuner.configs)
    assert len(configs) >= 3
    (left, right, w_rel, w_bond), ids, bond = _case(2, 90, seed=9)
    dz = torch.randn(2, 90, 90, P, device=DEV)
    ze = _run(token_pair_init_reference, (left, right, w_rel, w_bond), ids, bond)
    exp = torch.autograd.grad(ze, (left, right, w_rel, w_bond), dz)
    first = None
    for config in configs:
        monkeypatch.setattr(tuner, "configs", [config])
        monkeypatch.setattr(tuner, "cache", {})
        with portable_switch():
            z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bond)
            got = torch.autograd.grad(z, (left, right, w_rel, w_bond), dz)
        assert _rel(z, ze) < 1e-6, config
        for a, e in zip(got, exp, strict=True):
            assert _rel(a, e) < 1e-5, config
        if which in ("dw", "reduce"):
            first = first or got
            assert torch.equal(got[2], first[2]) and torch.equal(got[3], first[3]), config
