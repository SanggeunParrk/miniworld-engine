"""ConditionedTransition tail kernels (post-AdaLN: SwiGLU expand/squeeze + sigmoid gate)."""

from .triton import (
    ConditionedTransitionTail12345Function,
    ConditionedTransitionTailFunction,
    ConditionedTransitionTailFusedFunction,
    cond_transition_fwd_12_345,
    cond_transition_inference,
    cond_transition_inference_composed,
    cond_transition_inference_dispatch,
    cond_transition_train,
    cond_transition_train_12_345,
    cond_transition_train_fused,
)

__all__ = [
    "ConditionedTransitionTail12345Function",
    "ConditionedTransitionTailFunction",
    "ConditionedTransitionTailFusedFunction",
    "cond_transition_fwd_12_345",
    "cond_transition_inference",
    "cond_transition_inference_composed",
    "cond_transition_inference_dispatch",
    "cond_transition_train",
    "cond_transition_train_12_345",
    "cond_transition_train_fused",
]
