"""The fused SWA atom DiT block on B200 (``kernels/swa_dit/cuda/sm100``): every stage on the sm_100a kernels, its output and
every gradient (q, the five block weights, and the conditioning and adaLN weight through the hoisted modulation) no worse
against the fp32 reference than the Triton path's, the kernel variants each shape picks, CUDA-graph capture, and the
atom-count rule (a multiple of 128, else ValueError)."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.swa_dit.cuda import sm100
from miniworld_engine.kernels.swa_dit.interface import (
    swa_dit_block,
    swa_dit_hoist_modulation,
)
from miniworld_engine.kernels.swa_dit.reference import (
    swa_dit_block_reference,
    swa_dit_hoist_modulation_reference,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]
DEV, BF = "cuda", torch.bfloat16
C, H, HIDDEN, HW = 128, 4, 256, 64


def _case(a, b, s, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, BF)

    n = a * b
    leaves = [r(n, s, C), r(b, s, C), r(6 * C, C, scale=0.05), r(3 * C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5),
              r(C, C, scale=C ** -0.5), r(2 * HIDDEN, C, scale=C ** -0.5), r(C, HIDDEN, scale=HIDDEN ** -0.5)]
    angles = torch.randn(b, s, C // H // 2, generator=g) * 3.0
    lengths = [s, max(1, s - 17), 5, s - 1] * n                       # ragged, front-packed, one row shorter than the window
    return leaves, angles.cos().to(DEV), angles.sin().to(DEV), torch.tensor(lengths[:n], dtype=torch.int32, device=DEV)


def _fused(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    return swa_dit_block(q, swa_dit_hoist_modulation(c_base, wmod), cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW)


def _reference(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = swa_dit_hoist_modulation_reference(c_base, wmod)
    return swa_dit_block_reference(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW)


def _train(fn, leaves, dy, *args):
    live = [t.detach().clone().requires_grad_() for t in leaves]
    out = fn(live, *args)
    out.backward(dy.to(out.dtype))
    return [out.detach(), *(t.grad for t in live)]


def _rel(a, e):
    a, e = a.double(), e.double()
    return float((a - e).norm() / e.norm().clamp_min(1e-30))


NAMES = ("out", "q", "c_base", "w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")


@pytest.fixture
def spy(monkeypatch):
    calls = []
    for name in ("block_fwd", "block_bwd", "mod_fwd", "mod_bwd"):
        orig = getattr(sm100, name)
        monkeypatch.setattr(sm100, name, (lambda o, n: (lambda *a, **k: calls.append(n) or o(*a, **k)))(orig, name))
    return calls


# (A, B, S): A < 4 (the ATM = 32 builds), A = 5 / 8, A = 48 with and without dQ fused into the dK / dV pass
@pytest.mark.parametrize(("a", "b", "s"), [(3, 2, 256), (8, 2, 128), (48, 1, 256), (48, 1, 512)])
def test_training_matches_the_triton_path(a, b, s, spy, monkeypatch):
    leaves, cos, sin, seqused = _case(a, b, s)
    dy = torch.randn(a * b, s, C, device=DEV)
    truth = _train(_reference, [t.float() for t in leaves], dy, cos, sin, seqused, b)
    fused = _train(_fused, leaves, dy, cos, sin, seqused, b)
    assert sorted(set(spy)) == ["block_bwd", "block_fwd", "mod_bwd", "mod_fwd"], spy
    monkeypatch.setenv("MINIWORLD_SWA_DIT_SM100", "0")
    spy.clear()
    triton = _train(_fused, leaves, dy, cos, sin, seqused, b)
    assert spy == [], "the switch did not keep the Triton path"
    worst = []
    for name, f, t, e in zip(NAMES, fused, triton, truth, strict=True):
        ef, et = _rel(f, e), _rel(t, e)
        worst.append((ef / max(et, 1e-6), name, ef, et))
        assert torch.isfinite(f).all(), name
        assert f.dtype == t.dtype, name
        assert ef < 1.5 * et + 3e-3, f"{name}: sm_100a {ef:.2e} vs Triton {et:.2e}"
    print(f"training A{a} B{b} S{s} worst:", sorted(worst, reverse=True)[:2])


@pytest.mark.parametrize(("a", "b", "s"), [(1, 1, 256), (5, 1, 1024), (1, 3, 384)])
def test_inference_matches_the_triton_path(a, b, s, spy, monkeypatch):
    leaves, cos, sin, seqused = _case(a, b, s, seed=1)
    with torch.no_grad():
        truth = _reference([t.float() for t in leaves], cos, sin, seqused, b)
        fused = _fused(leaves, cos, sin, seqused, b)
        assert sorted(set(spy)) == ["block_fwd", "mod_fwd"], spy
        monkeypatch.setenv("MINIWORLD_SWA_DIT_SM100", "0")
        triton = _fused(leaves, cos, sin, seqused, b)
    ef, et = _rel(fused, truth), _rel(triton, truth)
    print(f"inference A{a} B{b} S{s}: sm_100a {ef:.2e} Triton {et:.2e}")
    assert ef < 1.5 * et + 3e-3, f"sm_100a {ef:.2e} vs Triton {et:.2e}"


def test_inference_is_cuda_graph_capturable():
    leaves, cos, sin, seqused = _case(5, 1, 256, seed=2)
    with torch.no_grad():
        eager = _fused(leaves, cos, sin, seqused, 1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _fused(leaves, cos, sin, seqused, 1)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _fused(leaves, cos, sin, seqused, 1)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def test_an_atom_count_off_the_128_grid_is_refused():
    leaves, cos, sin, seqused = _case(2, 1, 200, seed=3)
    with torch.no_grad(), pytest.raises(ValueError, match="multiple of 128"):
        _fused(leaves, cos, sin, seqused, 1)
