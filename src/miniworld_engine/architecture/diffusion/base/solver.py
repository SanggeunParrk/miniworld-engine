"""Abstract base solver and common type aliases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float
from pydantic import BaseModel

if TYPE_CHECKING:
    from miniworld_engine.architecture.diffusion.base.scheduler import (
        DiffusionScheduler,
    )


class DiffusionSolver(ABC):
    """Base class for defining a diffusion solver."""

    class SolverSchedulerConfig(BaseModel):
        """Configuration for the DiffusionScheduler class."""

        method: str = "EDM"

        # Add any additional configuration parameters here

    class SolverConfig(BaseModel):
        """Configuration for the DiffusionSolver class."""

        method: str = "Euler"
        seed: int = 0
        # Add any additional configuration parameters here

    def __init__(self, config: SolverConfig, scheduler: DiffusionScheduler) -> None:
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform one solver step."""


ModelFn = Callable[
    [Float[torch.Tensor, "... L 3"], Float[torch.Tensor, "..."]],
    Float[torch.Tensor, "... L 3"],
]


AtomChainMap = torch.Tensor | Mapping[Any, tuple[int, int]]


def _expand_to_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a 1D tensor with one value per batch item."""
    value = value.reshape(-1)
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        msg = f"Expected scalar or {batch_size} values, got shape {value.shape}."
        raise ValueError(msg)
    return value


def _chain_count(atom_chain_map: AtomChainMap) -> int:
    if isinstance(atom_chain_map, torch.Tensor):
        if atom_chain_map.numel() == 0:
            msg = "atom_chain_map must not be empty."
            raise ValueError(msg)
        return int(atom_chain_map.max().item()) + 1
    return len(atom_chain_map)
