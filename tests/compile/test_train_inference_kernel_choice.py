"""Which kernel a forward picks -- the training one that saves, or the inference one that does not.

Two families ask this, AdaptiveLayerNorm and ConditionedTransition, and they used to each answer
it themselves. Both answers were wrong, in opposite directions, and both failed silently:

  * neither asked whether gradients were being RECORDED, so a `mode=inference` module bench
    measured the training forward -- parameters always require grad and a module is in train mode
    until someone calls .eval(), and neither fact says anything about the call in front of it.
  * ConditionedTransition tested only `x.requires_grad`, so a detached x with a live conditioning
    tensor or a live WEIGHT took the inference path. Its kernels are `@opaque`, so the output has
    no grad_fn: the numbers come out right and nothing learns, with no error.

These run on the CPU -- the condition is the thing under test, not the kernels.
"""
from __future__ import annotations

import pytest
import torch

from miniworld_engine.modules.dispatch import needs_backward


class _M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2, 2))


def test_no_grad_means_no_saving() -> None:
    """The one that was costing a whole benchmark mode."""
    m = _M()
    x = torch.zeros(2, 2)
    assert needs_backward(m, x) is True          # train mode, gradients recorded
    with torch.no_grad():
        assert needs_backward(m, x) is False
    with torch.inference_mode():
        assert needs_backward(m, x) is False


def test_a_live_conditioning_tensor_counts() -> None:
    """The one that was losing gradients."""
    m = _M().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.zeros(2, 2)
    cond = torch.zeros(2, 2, requires_grad=True)
    assert needs_backward(m, x) is False
    assert needs_backward(m, x, cond) is True


def test_a_live_weight_counts() -> None:
    m = _M().eval()                               # parameters still require grad
    assert needs_backward(m, torch.zeros(2, 2)) is True


@pytest.mark.parametrize("module_path", [
    "miniworld_engine.modules.adaptive_layernorm.module",
    "miniworld_engine.modules.conditioned_transition.module",
])
def test_both_modules_ask_the_shared_condition(module_path: str) -> None:
    """Neither may grow its own copy again: that is how the two drifted apart."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module_path))
    assert "needs_backward(" in src, f"{module_path} does not use the shared condition"
    body = src.split("def forward", 1)[-1]
    assert "self.training or" not in body, f"{module_path} still hand-rolls the condition"
