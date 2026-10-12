from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from pydantic import BaseModel


# ruff: noqa: E501
class DiffusionScheduler(ABC):
    """Maps a time index t -> noise level(s), scaling factors, loss weights, etc."""

    class DiffusionSchedulerConfig(BaseModel):
        """Configuration for the DiffusionScheduler class."""

        method: str = "EDM"

    @abstractmethod
    def sample_noise(
        self,
        batch_size: int,
        uniform: bool = False,
    ) -> (
        Float[torch.Tensor, ...]
        | tuple[
            Float[torch.Tensor, ...],
            Float[torch.Tensor, ...],
            Float[torch.Tensor, ...],
        ]
    ):
        """Noise magnitude at time t."""

    @abstractmethod
    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
        sigma_translation: Float[torch.Tensor, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """How to scale the clean input into the noisy domain."""

    @abstractmethod
    def output_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """How to scale the model's raw prediction back to the data domain."""

    @abstractmethod
    def skip_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """How to scale the noixy x to the data domain."""

    @abstractmethod
    def loss_weight(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Weight to apply to loss at noise level sigma."""

    @abstractmethod
    def noise_condition(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Conditioning for the noise prediction."""

    @abstractmethod
    def sampling_time_steps(self, num_steps: int) -> Float[torch.Tensor, ...]:
        """Generate a schedule of time steps for sampling."""

    @abstractmethod
    def sampling_schedule(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)

    @abstractmethod
    def sampling_schedule_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt

    @abstractmethod
    def sampling_scale(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)

    @abstractmethod
    def sampling_scale_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factor derivatives for sampling."""
        # t -> dscale(t)/dt
