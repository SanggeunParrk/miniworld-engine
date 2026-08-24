"""Public entry points for the ConditionedTransition tail family.

The post-AdaLN tail of ConditionedTransition: SwiGLU expand/squeeze plus the sigmoid gate,
fused. Two entries, because inference and training are different kernels rather than one
kernel with a flag -- the inference paths save nothing for backward, while training is a
``torch.autograd.Function`` whose forward stores what its backward needs.

Inference goes through ``triton/dispatch.py``, which routes by ``d_hidden`` between the fused
b2b and composed triton variants. That per-variant choice stays inside the family; this module
is the door callers name.
"""

from __future__ import annotations

from .triton.dispatch import cond_transition_inference_dispatch
from .triton.training import cond_transition_train

__all__ = [
    "cond_transition_inference_dispatch",
    "cond_transition_train",
]
