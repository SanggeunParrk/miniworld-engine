"""CPU dispatch regressions: explicit memory selection must reach atomic backward."""

import pytest
import torch

from miniworld_engine.kernels.augmented_attention import interface
from miniworld_engine.kernels.augmented_attention.whole_op import (
    augmented_attention_pair_bias,
)
from miniworld_engine.modules.augmented_attention.module import (
    AugmentedAttentionPairBias,
)
from miniworld_engine.modules.exceptions import ImplementationType


@pytest.mark.parametrize("kernel_type", ["compute_efficient", "memory_efficient", None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_whole_op_routes_backend_and_preserves_layout(monkeypatch, kernel_type, dtype, masked):
    calls = []
    q = torch.randn(2, 1, 2, 5, 4, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    bias = torch.randn(1, 2, 5, 5, dtype=dtype, requires_grad=True)
    mask = torch.ones(2, 1, 5, dtype=torch.bool) if masked else None

    def backend(name):
        def run(qi, ki, vi, bi, mi):
            calls.append(name)
            torch.testing.assert_close(qi, q.transpose(2, 3))
            torch.testing.assert_close(ki, k.transpose(2, 3))
            torch.testing.assert_close(vi, v.transpose(2, 3))
            torch.testing.assert_close(bi, bias.permute(0, 2, 3, 1))
            assert mi is mask
            return qi + ki + vi + bi.sum()  # exercise wrapper autograd as well
        return run

    monkeypatch.setattr(interface, "_pair_bias_compute_efficient", backend("compute_efficient"))
    monkeypatch.setattr(interface, "_pair_bias_memory_efficient", backend("memory_efficient"))
    kwargs = {} if kernel_type is None else {"kernel_type": kernel_type}
    out = augmented_attention_pair_bias(q, k, v, bias, mask, **kwargs)
    assert calls == [kernel_type or "compute_efficient"]
    torch.testing.assert_close(out, q + k + v + bias.sum())
    out.sum().backward()
    for tensor in (q, k, v):
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
    torch.testing.assert_close(bias.grad, torch.full_like(bias, q.numel()))


@pytest.mark.parametrize("kernel_type", ["compute_efficient", "memory_efficient", None])
def test_module_routes_to_selected_backend(monkeypatch, kernel_type):
    calls = []

    def compute(q, k, v, bias, mask):
        calls.append("compute_efficient")
        return q

    def atomic(q, k, v, bias, mask):
        calls.append("memory_efficient")
        return q

    monkeypatch.setattr(interface, "_pair_bias_compute_efficient", compute)
    monkeypatch.setattr(interface, "_pair_bias_memory_efficient", atomic)
    kwargs = {} if kernel_type is None else {"kernel_type": kernel_type}
    module = AugmentedAttentionPairBias(8, 8, 4, 2, implementation=ImplementationType.MINIWORLD, **kwargs)
    q = torch.randn(2, 1, 5, 2, 4)
    bias = torch.randn(1, 5, 5, 2)
    assert module._kernel_attention_pair_bias(q, q, q, bias) is q
    assert calls == [kernel_type or "compute_efficient"]


@pytest.mark.parametrize("kernel_type", ["invalid"])
def test_invalid_kernel_type(kernel_type):
    q = torch.empty(2, 1, 2, 5, 4)
    bias = torch.empty(1, 2, 5, 5)
    with pytest.raises(ValueError, match="unknown kernel_type"):
        augmented_attention_pair_bias(q, q, q, bias, kernel_type=kernel_type)
    with pytest.raises(ValueError, match="unknown kernel_type"):
        AugmentedAttentionPairBias(8, 8, 4, 2, kernel_type=kernel_type)
