"""Transition adapter; the engine owns LayerNorm, SwiGLU and the residual."""

from miniworld_engine.architecture._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.modules import Transition as EngineTransition


class Transition(EngineTransition):
    """Return ``x + transition(x)`` with the historical checkpoint key names."""

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(d_hidden, n, implementation=to_engine_impl(implementation))
        # Preserve team-gm's construction precision. The model loader can then
        # cast the whole model to native BF16 while LayerNorm remains FP32.
        self.float()
