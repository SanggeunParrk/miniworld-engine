"""EDM scheduler."""

from __future__ import annotations

import torch
from jaxtyping import Float

from miniworld_engine.architecture.diffusion.base.scheduler import DiffusionScheduler
from miniworld_engine.architecture.diffusion.edm.schedules import PowerLawSchedule


class EDMScheduler(DiffusionScheduler):
    """EDM scheduler implementing DiffusionScheduler.

    Revised based on AF3 paper:
      - sigma_data set to 16.
      - Noise distribution: ln((1/sigma_data)*sigma) ~ N(P_mean, P_std).
      - Loss weight: (sigma^2+sigma_data^2)/(sigma*sigma_data)^2.
    """

    class EDMSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        """Configuration for the EDMScheduler class."""

        method: str = "AF3"
        P_mean: float = -1.2
        P_std: float = 1.5
        sigma_data: float = 16.0
        sigma_max: float = 160.0
        sigma_min: float = 4e-4
        rho: float = 7.0
        use_time_augmentation: bool = True

    def __init__(self, config: EDMSchedulerConfig) -> None:
        self.config = config

    def sample_noise(
        self,
        batch_size: int,
        uniform: bool = False,
    ) -> Float[torch.Tensor, ...]:
        """Sample noise from a exp Normal distribution."""
        if batch_size <= 0:
            msg = f"Batch size should be greater than 0, got {batch_size}"
            raise ValueError(msg)
        if uniform:
            num_uniform_steps = max(batch_size - 1, 1)
            return self.sampling_time_steps(num_uniform_steps)[:batch_size]
        if self.config.use_time_augmentation:
            u = torch.rand(batch_size)
        else:
            u = torch.rand(1).expand(batch_size)
        normal = torch.distributions.Normal(0.0, 1.0)

        sigma = self.config.sigma_data * torch.exp(
            self.config.P_mean + self.config.P_std * normal.icdf(u),
        )
        return torch.clamp(
            sigma,
            min=self.config.sigma_min * self.config.sigma_data,
            max=self.config.sigma_max * self.config.sigma_data,
        )

    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
        sigma_translation: Float[torch.Tensor, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """Compute the input scaling term."""
        del sigma_translation
        return 1.0 / torch.sqrt(sigma**2 + self.config.sigma_data**2)

    def output_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the output scaling term."""
        sd = self.config.sigma_data
        return (sigma * sd) / torch.sqrt(sigma**2 + sd**2)

    def skip_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the skip scaling term."""
        sd = self.config.sigma_data
        return sd**2 / (sigma**2 + sd**2)

    def loss_weight(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the loss weighting term."""
        sd = self.config.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def noise_condition(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the noise conditioning term."""
        return (sigma / self.config.sigma_data).log() / 4.0

    def sampling_time_steps(self, num_steps: int) -> Float[torch.Tensor, ...]:
        """Return steps + 1 AF3 levels ending at sigma_data * sigma_min."""
        schedule = PowerLawSchedule(
            PowerLawSchedule.Config(
                sigma_data=self.config.sigma_data,
                sigma_max=self.config.sigma_max,
                sigma_min=self.config.sigma_min,
                rho=self.config.rho,
                terminal_zero=False,
            )
        )
        return schedule(num_steps, dtype=torch.get_default_dtype())

    def sampling_schedule(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)
        return time_steps

    def sampling_schedule_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt
        return torch.ones_like(time_steps)

    def sampling_scale(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factor derivatives for sampling."""
        # t -> dscale(t)/dt
        return torch.zeros_like(time_steps)
