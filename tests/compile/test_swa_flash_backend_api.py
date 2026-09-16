"""Run the real SWA packing/call boundary against strict CPU FlashAttention mocks."""
from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "src/miniworld_engine/modules/swa_atom_attention/module.py"


def flash_core(backend: str) -> Callable[..., torch.Tensor]:
    tree = ast.parse(SOURCE.read_text())
    body: list[ast.stmt] = [node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name == "_flash_window_core"]
    namespace: dict[str, object] = {"torch": torch, "_flash_backend": lambda _device: backend}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return cast("Callable[..., torch.Tensor]", namespace["_flash_window_core"])


@pytest.mark.parametrize("backend", ["fa2", "fa4"])
@pytest.mark.parametrize("half_window", [-1, 2])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("mask", ["front", "holes_and_empty", "empty"])
def test_backend_specific_varlen_api_preserves_padding_and_gradients(
        monkeypatch, backend, half_window, grad_enabled, mask):
    n, s, heads, width = 2, 4, 1, 2
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    if mask == "holes_and_empty":
        valid = torch.tensor([[False, True, False, True], [False, False, False, False]])
    elif mask == "empty":
        valid = torch.zeros_like(valid)
    seqused = valid.sum(-1, dtype=torch.int32)
    cu = torch.arange(0, (n + 1) * s, s, dtype=torch.int32)
    q = torch.arange(n * s * heads * width, dtype=torch.float32).reshape(n, s, heads, width)
    q = q.requires_grad_()
    calls = []

    # FA4's installed signature intentionally has no max_seqlen_q/k and no **kwargs.
    def fa4(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None,
            seqused_q=None, seqused_k=None, page_table=None, softmax_scale=None,
            causal=False, window_size=(None, None), learnable_sink=None, softcap=0.0, pack_gqa=None):
        calls.append("fa4")
        torch.testing.assert_close(cu_seqlens_q, cu)
        torch.testing.assert_close(cu_seqlens_k, cu)
        torch.testing.assert_close(seqused_q, seqused)
        torch.testing.assert_close(seqused_k, seqused)
        assert q.shape == (8, 1, 2)
        assert q.dtype == torch.bfloat16
        assert softmax_scale == 0.5
        assert window_size == ((-1, -1) if half_window < 0 else (2, 2))
        return q * 2, torch.empty(0)  # exercise auxiliary tuple handling

    # FA2 keeps static capacity in both modes; cu_seqlens excludes unused storage.
    def fa2(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
        calls.append("fa2")
        assert max_seqlen_q == max_seqlen_k == s
        torch.testing.assert_close(cu_seqlens_q,
                                   torch.cat([seqused.new_zeros(1), seqused.cumsum(0, dtype=torch.int32)]))
        torch.testing.assert_close(cu_seqlens_k, cu_seqlens_q)
        assert q.shape == (n * s, 1, 2)
        assert q.dtype == torch.bfloat16
        assert softmax_scale == 0.5
        assert dropout_p == 0.0
        assert window_size == ((-1, -1) if half_window < 0 else (2, 2))
        return q * 2

    def unpad(tensor, mask):
        raise AssertionError("Neither mode may use dynamic nonzero packing")

    def pad(tensor, indices, batch, length):
        result = tensor.new_zeros((batch * length, *tensor.shape[1:]))
        return result.index_copy(0, indices, tensor).reshape(batch, length, *tensor.shape[1:])

    parent = ModuleType("flash_attn")
    parent.__dict__["__path__"] = []
    monkeypatch.setitem(sys.modules, "flash_attn", parent)
    for name, fn in (("flash_attn.cute", fa4), ("flash_attn.flash_attn_interface", fa2)):
        mod = ModuleType(name)
        mod.__dict__["flash_attn_varlen_func"] = fn
        monkeypatch.setitem(sys.modules, name, mod)
    padding = ModuleType("flash_attn.bert_padding")
    padding.__dict__.update(unpad_input=unpad, pad_input=pad,
                            index_first_axis=lambda value, indices: value[indices])
    monkeypatch.setitem(sys.modules, padding.__name__, padding)

    with torch.set_grad_enabled(grad_enabled):
        actual = flash_core(backend)(q, q, q, cu, seqused, s, valid, n, s, 0.5, half_window)
    expected = torch.where(valid[..., None, None], q * 2, torch.zeros_like(q))
    torch.testing.assert_close(actual, expected)
    if grad_enabled:
        actual.sum().backward()
        torch.testing.assert_close(q.grad, valid[..., None, None].expand_as(q).float() * 2)
    assert calls == [backend]
