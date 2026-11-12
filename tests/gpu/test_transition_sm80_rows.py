"""A100 (sm_80) Transition past one call's rows (FoldForge 5I28: the template pair stack at 4 x 2048^2 rows, the protenix confidence block's
pair transition at 5 x 1536^2). The one-kernel forward's per-CTA work list holds 253 tiles per SM and the wide chain's GEMM grid 65 535
tiles of 128 rows; the cuda backend raised ``fwd: too many work items per CTA`` / ``invalid configuration argument`` there. Inference now
runs one call per contiguous row block (``kernels.transition.cuda.transition_sm80``); a training call past the limit is refused with a
reason (the backward's work lists would drop rows)."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_fwd_sm80
from miniworld_engine.measure import config as C
from miniworld_engine.modules.primitives import BackendUnsupported
from miniworld_engine.modules.transition import Transition

pytestmark = [pytest.mark.gpu, pytest.mark.a100, pytest.mark.slow]


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _master(module: torch.nn.Module) -> typing.Any:
    module = module.cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    return C.cast(module, "master").eval()


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80) Transition")
@pytest.mark.parametrize(("d_hidden", "n"), [(128, 4), (256, 2)])  # fused_sm80 / the one-kernel forward; the wide chain
def test_transition_sm80_inference_past_one_call(d_hidden: int, n: int) -> None:
    m = _master(Transition(d_hidden=d_hidden, n=n))
    x = torch.randn(2, 2048, 2048, d_hidden, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)).bfloat16()
    assert x.numel() // d_hidden > fused_fwd_sm80.max_rows(x.device, x.dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m.set_backend("cuda")
        assert m.unsupported(x) is None
        y = m(x)
        m.set_backend("pytorch")
        ref = m(x)
    assert y.shape == ref.shape and bool(torch.isfinite(y).all())
    assert float((y.float() - ref.float()).abs().max() / ref.float().abs().max()) < 2e-2


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80) Transition")
def test_transition_sm80_training_past_one_call_refused() -> None:
    m = _master(Transition(d_hidden=64, n=2)).train()
    x = torch.randn(4, 2048, 2048, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    m.set_backend("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        why = m.unsupported(x)
        assert isinstance(why, str) and "rows with a gradient" in why
        with pytest.raises(BackendUnsupported):
            m(x)


def _train_grads(m: Transition, x: torch.Tensor, dy: torch.Tensor) -> list[torch.Tensor]:
    for p in m.parameters():
        p.grad = None
    xi = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(xi)
    y.backward(dy)
    return [y.detach(), typing.cast("torch.Tensor", xi.grad)] + [typing.cast("torch.Tensor", p.grad) for p in m.parameters()]


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80) Transition")
@pytest.mark.parametrize(("d_hidden", "n"), [(256, 2), (384, 2), (512, 4)])
def test_transition_sm80_train_overlap_backward_bits(d_hidden: int, n: int) -> None:
    """The saved-activation backward runs its weight-gradient GEMMs on a side stream (``fused_wide_sm80._bwd_saved_overlap``) and ``ln_bwd``
    keeps two rows in flight: the gradients equal the one-stream order of the same kernels bit for bit, run to run and under a graph."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    m = _master(Transition(d_hidden=d_hidden, n=n)).train()
    m.set_backend("cuda")
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, 96, 96, d_hidden, device="cuda", generator=g).bfloat16()
    dy = torch.randn(1, 96, 96, d_hidden, device="cuda", generator=g).bfloat16()
    assert fw._save_ab(96 * 96, d_hidden, d_hidden * n, True)
    first = _train_grads(m, x, dy)
    again = _train_grads(m, x, dy)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))

    # the one-stream order of the same launches (the pre-overlap backward), on the forward's saved tensors
    flat = x.reshape(-1, d_hidden).contiguous()
    wab16, ws16 = fw._master_cast_wab(m.expand_a.weight, m.expand_b.weight, m.squeeze.weight)
    h = d_hidden * n
    wa16, wb16 = wab16[:h], wab16[h:]
    _, xn, stats, hid, ab = fw._fwd_launch(flat, m.ln_in.weight, m.ln_in.bias, wa16, wb16, ws16, m.ln_in.eps, True, True)
    ext = fw._ext()
    dyf = dy.reshape(-1, d_hidden).contiguous()
    dab = ext.gate_dh(dyf, ws16.t().contiguous(), ab, fw._dh_cfg(flat.shape[0], d_hidden, h))
    dws = fw._mm_f32(dyf.t(), hid)
    dwab = fw._mm_f32(dab.t(), xn)
    dx, dgam, dbeta = ext.ln_bwd(fw._mm_f32(dab, fw._wab(wa16, wb16)), flat, stats, m.ln_in.weight.float(), dyf, torch.float32)
    ref = {"dx": dx, "ln_in.weight": dgam, "ln_in.bias": dbeta, "expand_a.weight": dwab[:h], "expand_b.weight": dwab[h:], "squeeze.weight": dws}
    names = ["y", "dx"] + [k for k, _ in m.named_parameters()]
    got = dict(zip(names, first, strict=True))
    assert torch.equal(got["dx"].reshape(-1, d_hidden), ref["dx"])
    for k in ("ln_in.weight", "ln_in.bias", "expand_a.weight", "expand_b.weight", "squeeze.weight"):
        assert torch.equal(got[k], ref[k].to(got[k].dtype)), k

    # captured: the side stream forks / joins inside the graph
    xs = x.detach().clone().requires_grad_(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _train_grads(m, x, dy)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for p in m.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ys = m(xs)
        ys.backward(dy)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(typing.cast("torch.Tensor", xs.grad), first[1])
    assert all(torch.equal(typing.cast("torch.Tensor", p.grad), r) for p, r in zip(m.parameters(), first[2:], strict=True))
