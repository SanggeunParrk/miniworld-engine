"""H100: the fused SWA atom block's bf16 global attention on the composed bf16 stages around FlashAttention-4
(``swa_dit.dispatch._h100_global_composed``) -- forward and every gradient against the block's PyTorch statements in float64, with
padded keys (rows past ``seqused``: FA4 leaves them alone, the composed path zeroes them first)."""

from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine.modules.atom_dit import AtomDiTMiniWorld

pytestmark = [pytest.mark.gpu]


def _rel(x: torch.Tensor, gold: torch.Tensor) -> float:
    return float((x.double() - gold).abs().max() / gold.abs().max().clamp_min(1e-30))


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("samples", [1, 3])
def test_swa_block_global_bf16_composed(grad, samples):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the H100 (sm_90) path")
    from miniworld_engine.kernels.swa_dit import dispatch

    torch.manual_seed(0)
    m = AtomDiTMiniWorld(half_window=500000).cuda()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            r = torch.randn(p.shape, generator=g)
            p.copy_(1.0 + 0.1 * r if p.ndim == 1 else r * p.shape[-1] ** -0.5)
    inputs = m.make_inputs(
        length=320, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    x, cond, params = inputs["x"], inputs["cond"], inputs["attention_params"]
    assert int(params[2].min()) < x.shape[1], "the inputs must pad keys (seqused < S)"
    ref = copy.deepcopy(m).double().set_backend("pytorch")
    wide = tuple(t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in params)
    xr, cr = (t.detach().double().requires_grad_(grad) for t in (x, cond))
    out_ref = ref(xr, cr, wide)
    m.set_backend("cuda")
    xb, cb = (t.detach().to(torch.bfloat16).requires_grad_(grad) for t in (x, cond))
    assert m.unsupported(xb, cb, params) is None, m.unsupported(xb, cb, params)
    assert dispatch._h100_global_composed(xb, m.attn.half_window), "bf16 global attention must take the composed stages"
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
