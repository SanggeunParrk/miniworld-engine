"""ConditionedTransition tail kernels (post-AdaLN: SwiGLU expand/squeeze + sigmoid gate)."""

from miniworld_engine.kernels.conditioned_transition.triton import (
    ConditionedTransitionTailFunction,
    cond_transition_inference,
    cond_transition_inference_composed,
    cond_transition_inference_dispatch,
    cond_transition_train,
)

__all__ = [
    "ConditionedTransitionTailFunction",
    "cond_transition_inference",
    "cond_transition_inference_composed",
    "cond_transition_inference_dispatch",
    "cond_transition_train",
]
