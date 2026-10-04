"""Fused token-pair initialisation on B200 (``kernels/token_pair_init``): output and every gradient against the dense fp32
reference (the 139-wide one-hot Linear the model ran), ragged lengths, B > 1, CUDA-graph capture."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.token_pair_init import (
    refusal,
    token_pair_init,
    token_pair_init_reference,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]
DEV, P = "cuda", 128


def _case(b, l, seed=0, r_max=32, s_max=2, grad=True):
    g = torch.Generator().manual_seed(seed)
    n_rel = 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1

    def ids(hi, dtype=torch.int64):
        return torch.randint(0, hi, (b, l), generator=g).to(DEV, dtype)

    asym = ids(3)
    resi = ids(90)                                 # offsets beyond r_max (clamp) and equal residues both occur
    tok = ids(60)
    ent = ids(2)
    sym = ids(6, torch.int32)                      # offsets beyond s_max
    bond = (torch.rand(b, l, l, generator=g) < 0.05).to(DEV)
    left = torch.randn(b, l, P, generator=g).to(DEV)
    right = torch.randn(b, l, P, generator=g).to(DEV)
    w_rel = (torch.randn(P, n_rel, generator=g) * 0.3).to(DEV)
    w_bond = (torch.randn(P, 2, generator=g) * 0.3).to(DEV)
    leaves = [t.requires_grad_(grad) for t in (left, right, w_rel, w_bond)]
    return leaves, (asym, resi, tok, ent, sym), bond


def _run(fn, leaves, ids, bond, **kw):
    left, right, w_rel, w_bond = leaves
    asym, resi, tok, ent, sym = ids
    return fn(left, right, w_rel, w_bond, asym, resi, tok, ent, sym, bond, **kw)


def _rel(a, e):
    return float((a.double() - e.double()).norm() / e.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize(("b", "l"), [(1, 384), (2, 130), (1, 7), (1, 1), (3, 64)])
def test_matches_the_dense_reference(b, l):
    (left, right, w_rel, w_bond), ids, bond = _case(b, l)
    assert refusal(left, right, w_rel, w_bond, bond) is None
    z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bond)
    ze = _run(token_pair_init_reference, (left, right, w_rel, w_bond), ids, bond)
    assert z.shape == ze.shape
    assert z.dtype == torch.float32
    assert _rel(z, ze) < 1e-6, _rel(z, ze)
    dz = torch.randn_like(ze)
    got = torch.autograd.grad(z, (left, right, w_rel, w_bond), dz)
    exp = torch.autograd.grad(ze, (left, right, w_rel, w_bond), dz)
    for name, a, e in zip(("left", "right", "w_rel", "w_bond"), got, exp, strict=True):
        assert a.shape == e.shape, name
        assert _rel(a, e) < 1e-5, (name, _rel(a, e))


def test_bond_dtypes_and_ids():
    (left, right, w_rel, w_bond), ids, bond = _case(1, 96, seed=1, grad=False)
    ze = _run(token_pair_init_reference, (left, right, w_rel, w_bond), ids, bond)
    for bd in (bond, bond.to(torch.uint8), bond.long(), bond.float()):
        z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bd)
        assert _rel(z, ze) < 1e-6


def test_refusals():
    (left, right, w_rel, w_bond), _ids, bond = _case(1, 32, grad=False)
    assert "bond" in (refusal(left, right, w_rel, w_bond, None) or "")
    assert refusal(left.bfloat16(), right.bfloat16(), w_rel, w_bond, bond) is None     # bf16 streams: cast to fp32 in the op
    assert "fp32 or bf16" in (refusal(left.half(), right.half(), w_rel, w_bond, bond) or "")
    assert "128" in (refusal(left[..., :64], right[..., :64], w_rel, w_bond, bond) or "")


def test_cuda_graph_capture():
    (left, right, w_rel, w_bond), ids, bond = _case(1, 128, seed=2)
    dz = torch.randn(1, 128, 128, P, device=DEV)

    def step():
        z = _run(token_pair_init, (left, right, w_rel, w_bond), ids, bond)
        return z, torch.autograd.grad(z, (left, right, w_rel, w_bond), dz)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                       # eager runs and warm-up on the stream the capture will use
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
        assert _rel(a, e) < 1e-6


def test_bins_that_do_not_fit_eight_bits_are_refused():
    """The kernels pack three bins into 8 bits each (the largest is n_rel - 2): r_max = 64 does not fit and must not run silently wrong."""
    from miniworld_engine.kernels.token_pair_init.cuda import sm100

    (left, right, _, w_bond), _ids, bond = _case(1, 32, grad=False)
    w_rel = torch.randn(P, 2 * (2 * 64 + 2) + (2 * 2 + 2) + 1, device=DEV)
    assert "8 bits" in (refusal(left, right, w_rel, w_bond, bond, r_max=64) or "")
    with pytest.raises(ValueError, match="8 bits"):
        sm100.check_packable(64, 2)
    sm100.check_packable(32, 2)
