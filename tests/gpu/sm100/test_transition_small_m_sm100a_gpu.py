"""The B200 Transition at D = 384 and a few row tiles (the single track [B, L, 384] of a pairformer: ``fused_wide_sm100a`` round sx,
``sm100/widths/tsx_*.cu``): LayerNorm (+ the fp32-master casts) -> expand + SwiGLU -> squeeze forward, gate -> d_xn + LayerNorm
backward -> weight gradients + dgamma / dbeta.
Claims: the module really runs these kernels at every tile count up to the bound, for n = 2 and 4, in inference and training; output
and every gradient (fp32 masters and bf16 parameters) are no further from the fp32 PyTorch module than the bf16 PyTorch statements
are; every squeeze decomposition and expand chunk width agree; a repeated step is bit-identical (every reduction runs in a fixed
order); the training forward's output equals inference's."""

import copy

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as w
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.transition.module import Transition

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"
    ),
]


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _module(n, fp32_masters, seed=3):
    torch.manual_seed(seed)
    m = Transition(384, n, implementation=ImplementationType.MINIWORLD).cuda()
    if fp32_masters:
        m = m.float()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 2:  # the zero-init squeeze would make most gradients exactly zero
                p.normal_(std=p.shape[1] ** -0.5)
            else:
                p.copy_(1 + 0.2 * torch.randn_like(p))
    return m


def _step(mod, x, dy, *, fp32=False, autocast=True):
    mod = copy.deepcopy(mod)
    if fp32:
        mod, x, dy = mod.float(), x.float(), dy.float()
    xx = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and not fp32):
        y = mod(xx)
    y.backward(dy)
    return {"out": y.detach().float(), "dx": xx.grad.float(), **{k: p.grad.float() for k, p in mod.named_parameters()}}


def _spy(monkeypatch):
    names = []
    real = w._Launch.run

    def run(self, cubin, *a, **k):
        names.append(cubin)
        return real(self, cubin, *a, **k)

    monkeypatch.setattr(w._Launch, "run", run)
    return names


@pytest.mark.parametrize("fp32_masters", [True, False], ids=["fp32-masters", "bf16-params"])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("rows", [128, 384, 768, 896, 1536, 2048])
def test_small_m_path_runs_and_is_no_less_accurate_than_pytorch(rows, n, fp32_masters, monkeypatch):
    mod = _module(n, fp32_masters)
    x = torch.randn(1, rows, 384, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    ref = _step(mod, x, dy, fp32=True)
    pt = copy.deepcopy(mod)
    pt._backend = KernelBackend.PYTORCH
    pt.implementation = ImplementationType.PYTORCH
    base = _step(pt, x, dy)
    names = _spy(monkeypatch)
    got = _step(mod, x, dy)
    assert {"sx_ln", "sx_gate"} <= set(names), names
    assert all(any(c.startswith(p) for c in names) for p in ("sx_expand", "sx_sq", "sx_dxn", "sx_dw")), names
    for k in ref:
        assert torch.isfinite(got[k]).all(), k
        e, b = _rel(got[k], ref[k]), _rel(base[k], ref[k])
        assert e <= max(1.25 * b, 1e-6), f"{k}: kernels {e:.3e} vs PyTorch bf16 {b:.3e}"


@pytest.mark.parametrize("n", [2, 4])
def test_every_squeeze_and_expand_build_agrees(n, monkeypatch):
    mod = _module(n, False)
    x = torch.randn(1, 768, 384, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref = mod(x).float()
        outs = []
        for sq in w.SX_SQ:
            for ch in w.SX_CH:
                monkeypatch.setattr(w, "SX_SQ_FORCE", sq)
                monkeypatch.setattr(w, "_sx_ch", lambda h, tiles, nsm, ch=ch: ch)
                outs.append(((sq, ch), mod(x).float()))
    for key, o in outs:
        assert _rel(o, ref) < 2e-3, key  # fp32 sums in another order, one bf16 rounding of the output


@pytest.mark.parametrize("fp32_masters", [True, False], ids=["fp32-masters", "bf16-params"])
@pytest.mark.parametrize("n", [2, 4])
def test_every_weight_gradient_tile_width_agrees(n, fp32_masters, monkeypatch):
    """The weight-gradient kernel's output tile width (``SX_DW_NT``, picked to fit beside the d_xn clusters) changes only the tiling of
    the outputs, not any element's sum: every width gives the same gradients, bit for bit."""
    mod = _module(n, fp32_masters)
    x = torch.randn(1, 768, 384, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    names = _spy(monkeypatch)
    outs = {}
    for nt in w.SX_DW_NT:
        monkeypatch.setattr(w, "SX_DW_NT_FORCE", nt)
        names.clear()
        outs[nt] = _step(mod, x, dy)
        assert f"sx_dw{nt}" in names, names
    first = outs[w.SX_DW_NT[0]]
    for nt, got in outs.items():
        assert all(torch.equal(got[k], first[k]) for k in first), nt


@pytest.mark.parametrize("n", [2, 4])
def test_step_is_bit_identical(n):
    mod = _module(n, True)
    x = torch.randn(1, 768, 384, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    poison = torch.full((64 << 20,), float("nan"), device="cuda")  # scratch / partial buffers must not depend on old contents
    del poison
    first, second = _step(mod, x, dy), _step(mod, x, dy)
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_inference_output_equals_the_training_forward():
    mod = _module(2, True)
    x = torch.randn(1, 640, 384, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inf = mod(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tr = mod(x.clone().requires_grad_())
    assert torch.equal(inf, tr.detach())


def test_training_forward_reads_the_masters_of_this_step():
    """With fp32 masters the LayerNorm launch writes the bf16 copies and the expand (a programmatic dependent launch) must read them
    only after griddepcontrol.wait. A CUDA graph (where the launches are staged back to back) flips the masters in place right
    before the forward, as an optimizer step would write them; every replay's output is held to the fp32 module of that step: a copy
    read early would carry the previous step's sign."""
    mod = _module(2, True)
    x = torch.randn(1, 768, 384, device="cuda", dtype=torch.bfloat16)
    xx = x.clone().requires_grad_()
    mats = (mod.expand_a.weight, mod.expand_b.weight, mod.squeeze.weight)

    def step():
        with torch.no_grad():
            for p in mats:
                p.mul_(-1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mod(xx)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = step()
    for _ in range(12):
        graph.replay()
        torch.cuda.synchronize()
        with torch.no_grad():
            ref = copy.deepcopy(mod).float()(x.float())
        assert _rel(y.detach().float(), ref) < 1e-2
