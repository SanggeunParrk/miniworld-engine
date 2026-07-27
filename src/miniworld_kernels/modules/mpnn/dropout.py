"""Dropout module used only by ProteinMPNN encoder edge updates."""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.mpnn_edge_dropout import (
    EdgeDropoutBackend,
    edge_dropout,
)


class EdgeDropout(torch.nn.Dropout):
    """``nn.Dropout`` boundary with an opt-in compressed-mask backend.

    Keeping dispatch inside the module preserves its stable state-dict path,
    train/eval behavior, and module forward/backward hook boundary.
    The explicit ``bitpack`` training backend supports first-order gradients only.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        *,
        backend: EdgeDropoutBackend = "auto",
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        if backend not in {"auto", "pytorch", "bitpack"}:
            raise ValueError(f"unknown MPNN edge dropout backend: {backend!r}")
        self.backend: EdgeDropoutBackend = backend

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return edge_dropout(
            input,
            self.p,
            training=self.training,
            backend=self.backend,
            inplace=self.inplace,
        )

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, backend={self.backend!r}"


__all__ = ["EdgeDropout"]
