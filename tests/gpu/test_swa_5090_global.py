"""RTX 5090 (sm_120a): the SWA atom block's bf16 global attention on the engine's own forward kernel (``attn_global_sm120.cuh``) and the fused
qkvg / FFN stages around it, and the bf16 training forward on the fused stages (``composed.TRAIN_FUSED_SM120_MIN_ROWS``) -- forward and every gradient
against the block's PyTorch statements in float64, with padded keys (rows past ``seqused``) and a row count on both sides of the fused thresholds."""

from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine.modules.atom_dit import AtomDiTMiniWorld

pytestmark = [pytest.mark.gpu]


def _rel(x: torch.Tensor, gold: torch.Tensor) -> float:
    return float((x.double() - gold).abs().max() / gold.abs().max().clamp_min(1e-30))


def _case(half_window: int, length: int, samples: int, grad: bool, monkeypatch, train_min_rows: int | None = None):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the RTX 5090 (sm_120) path")
    from miniworld_engine.kernels.swa_dit.cuda.sm80 import composed

    if train_min_rows is not None:
        monkeypatch.setattr(composed, "TRAIN_FUSED_SM120_MIN_ROWS", train_min_rows)
    torch.manual_seed(0)
    m = AtomDiTMiniWorld(half_window=half_window).cuda()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            r = torch.randn(p.shape, generator=g)
            p.copy_(1.0 + 0.1 * r if p.ndim == 1 else r * p.shape[-1] ** -0.5)
    inputs = m.make_inputs(
        length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    x, cond, params = inputs["x"], inputs["cond"], inputs["attention_params"]
    assert int(params[2].min()) < x.shape[1], "the inputs must pad keys (seqused < S)"
    ref = copy.deepcopy(m).double().set_backend("pytorch")
    wide = tuple(t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in params)
    xr, cr = (t.detach().double().requires_grad_(grad) for t in (x, cond))
    out_ref = ref(xr, cr, wide)
    m.set_backend("cuda")
    xb, cb = (t.detach().to(torch.bfloat16).requires_grad_(grad) for t in (x, cond))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xb, cb, params)
    tol = 6e-2  # bf16 operands
    assert _rel(out, out_ref.detach()) < tol, f"output {_rel(out, out_ref.detach()):.3g}"
    if not grad:
        return
    up = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64).cuda()
    out_ref.backward(up)
    out.backward(up.to(out.dtype))
    for name, a, r in (("x", xb, xr), ("cond", cb, cr)):
        assert a.grad is not None and r.grad is not None, name
        assert _rel(a.grad, r.grad) < tol, f"d{name} {_rel(a.grad, r.grad):.3g}"
    for (name, p), (_, pr) in zip(m.named_parameters(), ref.named_parameters(), strict=True):
        if pr.grad is not None and pr.grad.abs().max() > 0:
            assert p.grad is not None, name
            assert _rel(p.grad, pr.grad) < tol, f"d{name} {_rel(p.grad, pr.grad):.3g}"


@pytest.mark.parametrize("length,samples", [(320, 1), (2560, 1), (1100, 5)])
def test_swa_block_global_inference(length, samples, monkeypatch):
    """Global-attention inference: the sm_120a forward kernel (composed stages below 2049 rows, the fused stages from there)."""
    _case(500000, length, samples, False, monkeypatch)


@pytest.mark.parametrize("half_window,length", [(500000, 2560), (64, 2560)])
def test_swa_block_train_fused_forward(half_window, length, monkeypatch):
    """Training (forward + every gradient) with the forward on the fused stages (threshold 0) for the window and the global attention."""
    _case(half_window, length, 1, True, monkeypatch, train_min_rows=0)


@pytest.mark.parametrize(
    "half_window,length,samples",
    [(64, 5120, 1), (64, 3000, 1), (500000, 4096, 1), (500000, 2048, 1), (500000, 1100, 5), (64, 1280, 2), (64, 8192, 1)],
)
def test_swa_block_train_streaming_backward(half_window, length, samples, monkeypatch):
    """Training backward on the K-split streaming dy / qkvg kernels (``bwd_ks_sm120.cuh``) and the grouped weight-gradient GEMM (``wgrad5_sm120.cu``; 3000 rows
    are not a multiple of 256: the cuBLAS products), every gradient against the block's PyTorch statements in float64."""
    _case(half_window, length, samples, True, monkeypatch)
