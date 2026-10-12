"""Decoupled x-prediction scheduler (VE noise, independent of edm/)."""

from __future__ import annotations

import torch
from jaxtyping import Float

from miniworld_engine.architecture.diffusion.base.scheduler import DiffusionScheduler


class DecoupledXPredScheduler(DiffusionScheduler):
    """Decoupled x-prediction scheduler (VE noise, independent of edm/) implementing DiffusionScheduler.

    Revised based on the AF3 paper:
      - sigma_data set to 16.
      - Noise distribution: ln((1/sigma_data)*sigma_y) ~ N(P_mean, P_std).
      - Rotation and translation noise use their own schedules derived from sigma_y.
    """

    class DecoupledXPredSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        """Configuration for the DecoupledXPredScheduler class."""

        method: str = "AF3"
        P_mean: float = -1.2
        P_std: float = 1.5
        sigma_data: float = 16.0

        sigma_y_max: float = 160.0
        sigma_y_min: float = 4e-4
        rho_y: float = 7.0
        # Phase boundaries in the actual sigma_y units after sigma_data scaling.
        sigma_y_phase_1_boundary: float = 64.0
        sigma_y_phase_2_boundary: float = 1e-2
        smooth_phase_transition: bool = True

        sigma_R_max: float = 3.0
        sigma_R_min: float = 0.01
        rho_R: float = 3.0

        sigma_T_max: float = 8.0
        sigma_T_min: float = 4e-6
        rho_T: float = 3.0

        use_time_augmentation: bool = True

    def __init__(self, config: DecoupledXPredSchedulerConfig) -> None:
        self.config = config
        sigma_y_max = self.config.sigma_y_max * self.config.sigma_data
        sigma_y_min = self.config.sigma_y_min * self.config.sigma_data
        if not (
            sigma_y_min
            < self.config.sigma_y_phase_2_boundary
            < self.config.sigma_y_phase_1_boundary
            < sigma_y_max
        ):
            msg = (
                "sigma_y phase boundaries must satisfy "
                "sigma_y_min < phase_2_boundary < phase_1_boundary < sigma_y_max "
                f"after sigma_data scaling; got min={sigma_y_min}, "
                f"phase_2={self.config.sigma_y_phase_2_boundary}, "
                f"phase_1={self.config.sigma_y_phase_1_boundary}, "
                f"max={sigma_y_max}."
            )
            raise ValueError(msg)

    @staticmethod
    def _interpolate_sigma(
        progress: Float[torch.Tensor, ...],
        sigma_start: float,
        sigma_end: float,
        rho: float,
    ) -> Float[torch.Tensor, ...]:
        """Interpolate between sigma endpoints in EDM-style transformed space."""
        sigma_start_r = sigma_start ** (1.0 / rho)
        sigma_end_r = sigma_end ** (1.0 / rho)
        return (sigma_start_r + progress * (sigma_end_r - sigma_start_r)) ** rho

    @staticmethod
    def _interpolation_progress(
        sigma: Float[torch.Tensor, ...],
        sigma_start: float,
        sigma_end: float,
        rho: float,
    ) -> Float[torch.Tensor, ...]:
        """Invert `_interpolate_sigma` for monotone decreasing schedules."""
        sigma_start_r = sigma_start ** (1.0 / rho)
        sigma_end_r = sigma_end ** (1.0 / rho)
        sigma_r = torch.clamp(sigma, min=0.0) ** (1.0 / rho)
        return torch.clamp(
            (sigma_r - sigma_start_r) / (sigma_end_r - sigma_start_r),
            min=0.0,
            max=1.0,
        )

    @staticmethod
    def _smoothstep(progress: Float[torch.Tensor, ...]) -> Float[torch.Tensor, ...]:
        """C1 transition curve with zero slope at both phase boundaries."""
        progress = torch.clamp(progress, min=0.0, max=1.0)
        return progress * progress * (3.0 - 2.0 * progress)

    @property
    def phase_1_boundary(self) -> float:
        """sigma_y above which sigma_R / sigma_T stay at their maxima.

        Default warm-start ``start_sigma_y`` for flexible-docking: at this
        sigma the schedule is still in phase 1 (max rigid-body noise) yet
        coordinate noise is finite, so the first solver step samples max
        R/T per combine-group while ``init_x0`` is preserved as signal.
        """
        return float(self.config.sigma_y_phase_1_boundary)

    def _sigma_y_phase_boundaries(self) -> tuple[float, float, float, float]:
        """Return the full sigma_y range and the two phase boundaries."""
        sigma_y_max = self.config.sigma_y_max * self.config.sigma_data
        sigma_y_min = self.config.sigma_y_min * self.config.sigma_data
        sigma_y_phase_1_end = self.config.sigma_y_phase_1_boundary
        sigma_y_phase_2_end = self.config.sigma_y_phase_2_boundary
        return sigma_y_max, sigma_y_min, sigma_y_phase_1_end, sigma_y_phase_2_end

    def convert_to_sigma_rt(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> tuple[Float[torch.Tensor, ...], Float[torch.Tensor, ...]]:
        """Convert sigma_y into a three-phase rigid-body schedule.

        Phase 1:
          - sigma_y decreases from sigma_y_max to the first sigma_y boundary
          - sigma_rotation / sigma_translation stay at their maxima
        Phase 2:
          - sigma_y traverses the middle sigma_y band
          - sigma_rotation / sigma_translation decay to their minima
        Phase 3:
          - sigma_rotation / sigma_translation stay near zero
          - sigma_y decreases to sigma_y_min for high-resolution refinement
        """
        eps_mask = sigma <= 1e-8
        sigma_y_max, sigma_y_min, sigma_y_phase_1_end, sigma_y_phase_2_end = (
            self._sigma_y_phase_boundaries()
        )
        sigma_y = torch.clamp(sigma, min=sigma_y_min, max=sigma_y_max)

        sigma_translation_max = self.config.sigma_T_max * self.config.sigma_data
        sigma_translation_min = self.config.sigma_T_min * self.config.sigma_data

        sigma_rotation = torch.full_like(sigma_y, self.config.sigma_R_min)
        sigma_translation = torch.full_like(sigma_y, sigma_translation_min)

        phase_1_mask = sigma_y >= sigma_y_phase_1_end
        phase_2_mask = (sigma_y < sigma_y_phase_1_end) & (
            sigma_y >= sigma_y_phase_2_end
        )

        sigma_rotation = torch.where(
            phase_1_mask,
            torch.full_like(sigma_rotation, self.config.sigma_R_max),
            sigma_rotation,
        )
        sigma_translation = torch.where(
            phase_1_mask,
            torch.full_like(sigma_translation, sigma_translation_max),
            sigma_translation,
        )

        if phase_2_mask.any():
            phase_2_progress = self._interpolation_progress(
                sigma_y[phase_2_mask],
                sigma_y_phase_1_end,
                sigma_y_phase_2_end,
                self.config.rho_y,
            )
            if self.config.smooth_phase_transition:
                phase_2_progress = self._smoothstep(phase_2_progress)
            sigma_rotation_phase_2 = self._interpolate_sigma(
                phase_2_progress,
                self.config.sigma_R_max,
                self.config.sigma_R_min,
                self.config.rho_R,
            )
            sigma_translation_phase_2 = self._interpolate_sigma(
                phase_2_progress,
                sigma_translation_max,
                sigma_translation_min,
                self.config.rho_T,
            )
            sigma_rotation = sigma_rotation.clone()
            sigma_translation = sigma_translation.clone()
            sigma_rotation[phase_2_mask] = sigma_rotation_phase_2
            sigma_translation[phase_2_mask] = sigma_translation_phase_2

        sigma_rotation = torch.where(
            eps_mask,
            torch.full_like(sigma_rotation, 1e-8),
            sigma_rotation,
        )
        sigma_translation = torch.where(
            eps_mask,
            torch.full_like(sigma_translation, 1e-8),
            sigma_translation,
        )
        return sigma_rotation, sigma_translation

    def sample_noise(
        self,
        batch_size: int,
        uniform: bool = False,
    ) -> tuple[
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
    ]:
        """Sample decoupled noise magnitudes for coordinates, rotation, and translation."""
        if batch_size <= 0:
            msg = f"Batch size should be greater than 0, got {batch_size}"
            raise ValueError(msg)
        if self.config.use_time_augmentation:
            u = torch.rand(batch_size)
        else:
            u = torch.rand(1).expand(batch_size)
        if uniform:
            num_uniform_steps = max(batch_size - 1, 1)
            sigma = self.sampling_time_steps(num_uniform_steps)[:batch_size]
        else:
            normal = torch.distributions.Normal(0.0, 1.0)
            sigma = self.config.sigma_data * torch.exp(
                self.config.P_mean + self.config.P_std * normal.icdf(u),
            )

        sigma_y = torch.clamp(
            sigma,
            min=self.config.sigma_y_min * self.config.sigma_data,
            max=self.config.sigma_y_max * self.config.sigma_data,
        )
        sigma_rotation, sigma_translation = self.convert_to_sigma_rt(sigma_y)
        return sigma_y, sigma_rotation, sigma_translation

    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
        sigma_translation: Float[torch.Tensor, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """Compute the input scaling term."""
        if sigma_translation is None:
            msg = "sigma_translation must be provided for decoupled EDM input scaling."
            raise ValueError(msg)
        return 1.0 / torch.sqrt(
            sigma**2 + sigma_translation**2 + self.config.sigma_data**2,
        )

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
        return sigma.log() / 4.0

    def sampling_time_steps(
        self,
        num_steps: int,
        start_sigma_y: float | None = None,
    ) -> Float[torch.Tensor, ...]:
        """Generate the EDM sigma_y schedule.

        Decoupled EDM keeps EDM's coordinate noise schedule/distribution and
        applies the three-phase policy only when converting sigma_y to
        sigma_rotation/sigma_translation.

        ``start_sigma_y``: when set, the schedule starts at this sigma_y
        instead of the default ``sigma_y_max * sigma_data`` (used by the
        flexible-docking warm start to skip the high-noise prefix).
        """
        time_steps = torch.empty(num_steps + 1)
        t = torch.linspace(0.0, 1.0, steps=num_steps)
        sigma_y_min_r = self.config.sigma_y_min ** (1.0 / self.config.rho_y)
        if start_sigma_y is None:
            sigma_y_max_r = self.config.sigma_y_max ** (1.0 / self.config.rho_y)
            scale = self.config.sigma_data
        else:
            if start_sigma_y <= 0.0:
                msg = f"start_sigma_y must be positive, got {start_sigma_y}."
                raise ValueError(msg)
            # ``start_sigma_y`` is already in scaled sigma_y units (the same
            # units the scheduler uses everywhere, e.g. for phase boundaries).
            sigma_y_max_r = (start_sigma_y / self.config.sigma_data) ** (
                1.0 / self.config.rho_y
            )
            scale = self.config.sigma_data
        time_steps[:-1] = (
            scale
            * (sigma_y_max_r + t * (sigma_y_min_r - sigma_y_max_r)) ** self.config.rho_y
        )
        time_steps[-1] = 0.0
        return time_steps

    def sampling_schedule(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise levels for sampling."""
        return time_steps

    def sampling_schedule_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise level derivatives for sampling."""
        return torch.ones_like(time_steps)

    def sampling_scale(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factors for sampling."""
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factor derivatives for sampling."""
        return torch.zeros_like(time_steps)
