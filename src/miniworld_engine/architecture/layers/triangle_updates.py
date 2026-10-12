"""Triangle adapters with engine-owned residuals and broadcast dropout."""

from miniworld_engine.architecture._engine_impl import to_engine_impl
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.modules import (
    BidirectionalTriangleMultiplication as EngineBidirectionalTriangleMultiplication,
)
from miniworld_engine.modules import TriangleAttention as EngineTriangleAttention
from miniworld_engine.modules import (
    TriangleMultiplication as EngineTriangleMultiplication,
)


class TriangleMultiplication(EngineTriangleMultiplication):
    """Return the residual update; ``p_drop`` applies only to its branch."""

    def __init__(
        self,
        d_pair: int = 128,
        *,
        d_hidden: int | None = None,
        outgoing: bool = True,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__(
            d_pair,
            d_hidden=d_hidden,
            outgoing=outgoing,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )


class BidirectionalTriangleMultiplication(EngineBidirectionalTriangleMultiplication):
    """Outgoing and incoming contraction with one residual and row dropout."""

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int | None = None,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__(
            d_pair,
            d_hidden=d_hidden,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )


class TriangleAttention(EngineTriangleAttention):
    """Triangle attention with an engine-owned residual and row/column dropout."""

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__(
            d_pair,
            n_head,
            d_hidden=d_hidden,
            starting=starting,
            use_self_attention=use_self_attention,
            use_qk_norm=use_qk_norm,
            implementation=to_engine_impl(implementation),
            p_drop=p_drop,
        )
