"""Abstract base diffuser."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from miniworld_engine.architecture.diffusion.base.scheduler import (
        DiffusionScheduler,
    )


def _expand_to_trailing_dims(
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Broadcast per-sample scalars over coordinate dimensions."""
    if value.ndim > target.ndim:
        msg = f"Cannot broadcast shape {value.shape} to target shape {target.shape}."
        raise ValueError(msg)
    return value.reshape(*value.shape, *((1,) * (target.ndim - value.ndim)))


class Diffuser(ABC):
    """Base class for defining a diffusion model. (use solver when sampling)."""

    class DiffuserConfig(BaseModel):
        """Configuration for the Diffuser class."""

        method: str = "EDM"
        seed: int = 0
        translation_noise: float = 1.0
        # Add any additional configuration parameters here

    def __init__(
        self,
        config: DiffuserConfig,
        scheduler: DiffusionScheduler,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @torch.no_grad()
    def random_rotation_and_translation(
        self,
        x: Float[torch.Tensor, "... L 3"],
    ) -> Float[torch.Tensor, "... L 3"]:
        """Apply random rotation and translation to the input tensor."""
        if x.ndim < 2:
            msg = "Input tensor must have at least 2 dimensions."
            raise ValueError(msg)
        if x.shape[-1] != 3:
            msg = "Last dimension of input tensor must be of size 3."
            raise ValueError(msg)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])  # (AB, L, 3) or (B, L, 3)

        n = x.shape[0]
        rot_mats = torch.from_numpy(Rotation.random(n).as_matrix()).to(
            x.device,
            x.dtype,
        )
        translation = (
            torch.randn(n, 1, 3, device=x.device, dtype=x.dtype)
            * self.config.translation_noise
        )

        x = torch.bmm(x, rot_mats.transpose(-1, -2))
        x = x + translation
        return x.reshape(*x_shape)

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample noisy input and store preconditioning data."""

    @abstractmethod
    def cal_loss(self, *args: Any, **kwargs: Any) -> Float[torch.Tensor, 1]:
        """Compute loss between model output and ground truth."""
