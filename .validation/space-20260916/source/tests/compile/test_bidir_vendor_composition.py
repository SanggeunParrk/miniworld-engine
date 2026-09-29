"""The vendor bidirectional composition must implement the shared-normalization formula."""
import sys
from types import ModuleType
from typing import Any, cast

import torch
import torch.nn.functional as F

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)


def test_cuequivariance_bidir_shares_output_norm_and_all_gradients(monkeypatch):
    calls = []
    norm = ModuleType("cuequivariance_ops_torch.fused_layer_norm_torch")
    gate = ModuleType("cuequivariance_ops_torch.gated_gemm_torch")

    def layer_norm(x, weight, bias, eps, layout):
        calls.append((layout, weight.numel()))
        if layout == "dbij->bijd":
            x = x.permute(1, 2, 3, 0)
        return F.layer_norm(x, (weight.numel(),), weight, bias, eps)

    def projection(x, gate_weight, project_weight, mask, transpose_out):
        assert transpose_out
        y = torch.sigmoid(F.linear(x, gate_weight)) * F.linear(x, project_weight)
        if mask is not None:
            y = y * mask[..., None]
        return y.permute(3, 0, 1, 2)

    cast("Any", norm).layer_norm_transpose = layer_norm
    cast("Any", gate).fused_sigmoid_gated_dual_gemm = projection
    monkeypatch.setitem(sys.modules, norm.__name__, norm)
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    torch.manual_seed(31)
    ref = BidirectionalTriangleMultiplication(8, p_drop=0, implementation=ImplementationType.PYTORCH)
    actual = BidirectionalTriangleMultiplication(8, p_drop=0, implementation=ImplementationType.CUEQUIVARIANCE)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(std=.3)
    actual.load_state_dict(ref.state_dict())
    x = torch.randn(2, 5, 5, 8, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1,1,0,1,1], [1,0,1,1,1]], dtype=torch.bool)
    y, yr = actual(x, mask), ref(xr, mask)
    torch.testing.assert_close(y, yr, atol=1e-6, rtol=1e-5)
    dy = torch.randn_like(y)
    y.backward(dy); yr.backward(dy)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-6, rtol=1e-5)
    for (name,p), (ref_name,rp) in zip(actual.named_parameters(), ref.named_parameters(), strict=True):
        assert name == ref_name
        assert p.grad is not None
        assert rp.grad is not None
        torch.testing.assert_close(p.grad, rp.grad, atol=1e-5, rtol=1e-4)
    assert calls == [("bijd->bijd",8), ("dbij->bijd",16)]
