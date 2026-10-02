"""Global attention in the fused SWA atom DiT block on B200 (``kernels/swa_dit`` with ``interface.is_global``): the sm_100a
stages around FlashAttention-4. Output and every gradient are held to what the windowed block already achieves against the
fp32 reference on the same inputs (same bf16 rounding points), padding rows stay clean, and the block captures in a CUDA graph."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.swa_dit.interface import (
    GLOBAL_HALF_WINDOW,
    is_global,
    refusal,
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
C, H, HIDDEN = 128, 4, 256
BIG = 1_000_000                                                  # what a model's huge ``swa_window_size`` becomes
NAMES = ("out", "q", "c_base", "w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")


def _case(a, b, s, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, BF)

    n = a * b
    leaves = [r(n, s, C), r(b, s, C), r(6 * C, C, scale=0.05), r(3 * C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5),
              r(C, C, scale=C ** -0.5), r(2 * HIDDEN, C, scale=C ** -0.5), r(C, HIDDEN, scale=HIDDEN ** -0.5)]
    angles = torch.randn(b, s, C // H // 2, generator=g) * 3.0
    lengths = [s, max(1, s - 17), 5, s - 1] * n                  # ragged, front-packed, one row shorter than a window
    return leaves, angles.cos().to(DEV), angles.sin().to(DEV), torch.tensor(lengths[:n], dtype=torch.int32, device=DEV)


def _fused(leaves, cos, sin, seqused, b, hw):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    return swa_dit_block(q, swa_dit_hoist_modulation(c_base, wmod), cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=hw)


def _reference(leaves, cos, sin, seqused, b, hw):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = swa_dit_hoist_modulation_reference(c_base, wmod)
    return swa_dit_block_reference(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=hw)


def _train(fn, leaves, dy, *args):
    live = [t.detach().clone().requires_grad_() for t in leaves]
    out = fn(live, *args)
    out.backward(dy.to(out.dtype))
    return [out.detach(), *(t.grad for t in live)]


def _rel(a, e):
    a, e = a.double(), e.double()
    return float((a - e).norm() / e.norm().clamp_min(1e-30))


def test_global_window_values():
    assert is_global(-1)
    assert is_global(BIG)
    assert is_global(GLOBAL_HALF_WINDOW)
    assert not is_global(64)


def test_refusal_serves_global_in_bf16_only():
    leaves, cos, sin, seqused = _case(1, 1, 128)
    _, _, _, wqkv, wg, wo, wu, wd = leaves
    q = leaves[0]
    assert refusal(q, cos, sin, seqused, wqkv, wg, wo, wu, wd, n_head=H, half_window=BIG) is None
    assert refusal(q, cos, sin, seqused, wqkv, wg, wo, wu, wd, n_head=H, half_window=100) is not None     # neither 64 nor global
    f32 = [t.float() for t in (q, wqkv, wg, wo, wu, wd)]
    assert "global attention" in (refusal(f32[0], cos, sin, seqused, *f32[1:], n_head=H, half_window=BIG) or "")


# (A, B, S): the embedder's A = 1 (S = 4096 in the model), small A, A = 48
@pytest.mark.parametrize(("a", "b", "s"), [(1, 1, 256), (1, 2, 384), (3, 2, 256), (48, 1, 256)])
def test_training_matches_the_reference(a, b, s):
    leaves, cos, sin, seqused = _case(a, b, s)
    dy = torch.randn(a * b, s, C, device=DEV)
    truth_g = _train(_reference, [t.float() for t in leaves], dy, cos, sin, seqused, b, BIG)
    got = _train(_fused, leaves, dy, cos, sin, seqused, b, BIG)
    truth_w = _train(_reference, [t.float() for t in leaves], dy, cos, sin, seqused, b, 64)
    win = _train(_fused, leaves, dy, cos, sin, seqused, b, 64)
    worst = []
    for name, g_, w_, eg, ew in zip(NAMES, got, win, truth_g, truth_w, strict=True):
        assert torch.isfinite(g_).all(), name
        egl, ewl = _rel(g_, eg), _rel(w_, ew)
        worst.append((egl / max(ewl, 1e-6), name, egl, ewl))
        assert egl < 1.5 * ewl + 3e-3, f"{name}: global {egl:.2e} vs the windowed block {ewl:.2e}"
    print(f"global training A{a} B{b} S{s} worst vs windowed:", sorted(worst, reverse=True)[:2])


@pytest.mark.parametrize(("a", "b", "s"), [(1, 1, 256), (1, 3, 384), (5, 1, 1024)])
def test_inference_matches_the_reference(a, b, s):
    leaves, cos, sin, seqused = _case(a, b, s, seed=1)
    with torch.no_grad():
        got = _fused(leaves, cos, sin, seqused, b, BIG)
        truth = _reference([t.float() for t in leaves], cos, sin, seqused, b, BIG)
        win = _fused(leaves, cos, sin, seqused, b, 64)
        truth_w = _reference([t.float() for t in leaves], cos, sin, seqused, b, 64)
    eg, ew = _rel(got, truth), _rel(win, truth_w)
    print(f"global inference A{a} B{b} S{s}: global {eg:.2e} windowed {ew:.2e}")
    assert eg < 1.5 * ew + 3e-3, f"global {eg:.2e} vs the windowed block {ew:.2e}"


def test_padding_rows_stay_clean():
    """Rows at or past seqused: attention output zero, no NaN in the output or any gradient (FA4 skips those rows)."""
    leaves, cos, sin, seqused = _case(2, 1, 256, seed=4)
    seqused = torch.tensor([256, 100], dtype=torch.int32, device=DEV)
    dy = torch.randn(2, 256, C, device=DEV)
    res = _train(_fused, leaves, dy, cos, sin, seqused, 1, BIG)
    for name, t in zip(NAMES, res, strict=True):
        assert torch.isfinite(t).all(), name


def test_global_is_cuda_graph_capturable():
    leaves, cos, sin, seqused = _case(1, 1, 256, seed=2)
    dy = torch.randn(1, 256, C, device=DEV)

    def step():
        live = [t.detach().clone().requires_grad_() for t in leaves]
        out = _fused(live, cos, sin, seqused, 1, BIG)
        out.backward(dy.to(out.dtype))
        return out.detach(), live[0].grad

    eager = step()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, dq = step()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager[0])                              # the forward is deterministic
    assert _rel(dq, eager[1]) < 2e-2                               # FA4's dQ accumulates with atomics: summation order varies


@pytest.mark.parametrize("hw", [64, BIG])
def test_packed_weights_follow_replaced_weights(hw):
    """The block caches packed copies of the weights. A new set of weights allocated where the old set was freed (same shape,
    same version) must get its own packed copies: a fresh bf16 cast of an fp32 master weight every step looks exactly like this."""
    errs = []
    for seed in range(4):
        leaves, cos, sin, seqused = _case(1, 1, 256, seed=10 + seed)
        with torch.no_grad():
            got = _fused(leaves, cos, sin, seqused, 1, hw)
            truth = _reference([t.float() for t in leaves], cos, sin, seqused, 1, hw)
        errs.append(_rel(got, truth))
        del leaves, got
    assert max(errs) < 1e-2, errs


def test_refusal_follows_the_sm100_switches(monkeypatch):
    """Global attention has no Triton fallback: refusal() must say so when the sm_100a stages are switched off, instead of letting
    the block raise mid-step."""
    leaves, cos, sin, seqused = _case(1, 1, 128)
    _, _, _, wqkv, wg, wo, wu, wd = leaves
    args = (leaves[0], cos, sin, seqused, wqkv, wg, wo, wu, wd)
    assert refusal(*args, n_head=H, half_window=BIG) is None
    monkeypatch.setenv("MINIWORLD_SWA_DIT_SM100", "0")
    assert "sm_100a" in (refusal(*args, n_head=H, half_window=BIG) or "")
    assert refusal(*args, n_head=H, half_window=64) is None                 # the windowed block still has its Triton path


@pytest.mark.parametrize("hw", [64, BIG])
def test_packed_weights_are_refreshed_in_a_captured_graph(hw):
    """A graph captured after eager warm-up must still repack the weights every replay: an optimizer step between replays changes
    the weights in place, and a cache hit during capture would bake the warm-up copy in."""
    leaves, cos, sin, seqused = _case(1, 1, 256, seed=7)

    def run():
        with torch.no_grad():
            return _fused(leaves, cos, sin, seqused, 1, hw)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run()
    g.replay()
    torch.cuda.synchronize()
    before = out.clone()
    for w in leaves[3:]:                                         # an "optimizer step": every block weight changes in place
        w.mul_(1.5)
    g.replay()
    torch.cuda.synchronize()
    want = _reference([t.float() for t in leaves], cos, sin, seqused, 1, hw)
    assert _rel(out, want) < 1e-2, _rel(out, want)
    assert _rel(out, before) > 1e-2                              # the replay really used the new weights


def test_inference_tensor_weights_are_served():
    """Inference tensors carry no version counter; the pack cache must not read one."""
    with torch.inference_mode():
        leaves, cos, sin, seqused = _case(1, 1, 128, seed=8)
        got = _fused(leaves, cos, sin, seqused, 1, BIG)
        want = _reference([t.float() for t in leaves], cos, sin, seqused, 1, BIG)
    assert _rel(got, want) < 1e-2
