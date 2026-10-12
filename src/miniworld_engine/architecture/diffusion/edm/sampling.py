"""Shared EDM sampling: churn, denoising, alignment and Euler integration.

Models supply a denoiser returning clean coordinates. Schedules, rigid motions,
and guidance are independent of model definitions.
Constructing a sampler never reseeds a process or changes backend settings.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from miniworld_engine.architecture.diffusion.base.solver import ModelFn

if TYPE_CHECKING:
    from collections.abc import Callable


def expand_to_coords(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast leading noise-level axes across coordinate dimensions."""
    if value.ndim > target.ndim:
        msg = f"Cannot broadcast shape {tuple(value.shape)} to {tuple(target.shape)}."
        raise ValueError(msg)
    return value.reshape(*value.shape, *(1,) * (target.ndim - value.ndim))


class PreconditionedDenoiser:
    """Adapt a network update to the sampler's clean-coordinate contract."""

    def __init__(
        self,
        network: ModelFn,
        scheduler: Any,
        *,
        condition_noise: bool = False,
        batch_noise: bool = True,
    ) -> None:
        self.network, self.scheduler = (network, scheduler)
        self.condition_noise, self.batch_noise = (condition_noise, batch_noise)

    def __call__(self, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.batch_noise:
            sigma = sigma.reshape(1).expand(noisy.shape[0])
        scheduler = self.scheduler
        scale = expand_to_coords(scheduler.input_scale(sigma), noisy)
        condition = scheduler.noise_condition(sigma) if self.condition_noise else sigma
        update = self.network(noisy * scale, condition)
        skip = expand_to_coords(scheduler.skip_scale(sigma), noisy)
        out = expand_to_coords(scheduler.output_scale(sigma), noisy)
        return skip * noisy + out * update


class EulerSampler:
    """One sampler for scalar or batched noise levels and arbitrary atom layouts."""

    class Config(BaseModel):
        """Numerical choices, independent of checkpoint/model names."""

        model_config = ConfigDict(extra="forbid", frozen=True)
        gamma_0: float = Field(default=0.8, ge=0)
        gamma_min: float = Field(default=1.0, ge=0)
        noise_scale: float = Field(default=1.003, ge=0)
        step_scale: float = Field(default=1.5, gt=0)
        clamp_variance: bool = False
        batched_sigma: bool = False
        #: EDM's own churn parameterisation: a CONSTANT gamma applied wherever
        #: the current sigma lies in the window, rather than AF3's fixed gamma_0
        #: switched on above a threshold. The caller computes the constant
        #: because it depends on the step count. None keeps AF3's form.
        churn_gamma: float | None = None
        churn_sigma_min: float = 0.0
        churn_sigma_max: float = float("inf")
        #: Floor of the re-noising variance. A no-churn step under AF3's form
        #: adds exactly nothing; a family that clamps above zero still adds a
        #: little, which matters once no-churn steps are most of the trajectory.
        variance_floor: float = 0.0

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def prepare(
        self,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        augment: Callable | None = None,
        sigma_dtype: torch.dtype | None = None,
        coordinate_dims: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repose and re-noise before the denoiser's compiled boundary."""
        if augment is not None:
            coords = augment(coords)
        config = self.config
        if config.churn_gamma is None:
            gamma = config.gamma_0 if sigma_next > config.gamma_min else 0.0
        else:
            inside = config.churn_sigma_min <= sigma <= config.churn_sigma_max
            gamma = config.churn_gamma if inside else 0.0
        sigma_hat = sigma * (1 + gamma)
        variance = sigma_hat**2 - sigma**2
        if config.clamp_variance or config.variance_floor:
            variance = variance.clamp(min=config.variance_floor)
        magnitude = config.noise_scale * torch.sqrt(variance)
        noisy = coords + magnitude * torch.randn(
            coords.shape, device=coords.device, dtype=coords.dtype, generator=generator
        )
        if config.batched_sigma:
            batch = coords.shape[: -(coordinate_dims + 1)]
            sigma_hat = sigma_hat.reshape((1,) * len(batch)).expand(batch)
        if sigma_dtype is not None:
            sigma_hat = sigma_hat.to(sigma_dtype)
        return (noisy, sigma_hat)

    def update(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        sigma_hat: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        align: Callable | None = None,
    ) -> torch.Tensor:
        """Advance the same Euler equation for every model."""
        if align is not None:
            noisy = align(noisy, denoised)
        direction = (noisy - denoised) / expand_to_coords(sigma_hat, noisy)
        dt = expand_to_coords(sigma_next - sigma_hat, noisy)
        return noisy + self.config.step_scale * dt * direction

    def step(
        self,
        denoise: ModelFn,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        augment: Callable | None = None,
        align: Callable | None = None,
        sigma_dtype: torch.dtype | None = None,
        coordinate_dims: int = 1,
        guidance: Callable | None = None,
        step_index: int = 0,
        num_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare noise, denoise or guide, then advance one Euler step."""
        noisy, sigma_hat = self.prepare(
            coords,
            sigma,
            sigma_next,
            generator=generator,
            augment=augment,
            sigma_dtype=sigma_dtype,
            coordinate_dims=coordinate_dims,
        )
        if guidance is not None:
            return (guidance(noisy, sigma_hat, sigma_next, step_index, num_steps), None)
        denoised = denoise(noisy, sigma_hat)
        result = self.update(noisy, denoised, sigma_hat, sigma_next, align=align)
        return (result, denoised)

    def integrate(
        self,
        denoise: ModelFn,
        coords: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        return_trajectory: bool = False,
        **step_options,
    ):
        """Integrate an existing iterate; this is the sole reverse-time loop."""
        if sigmas.ndim != 1 or len(sigmas) < 2:
            msg = "A noise schedule must contain at least two scalar levels"
            raise ValueError(msg)
        trajectory, denoised_trajectory = ([], [])
        for index, (sigma, sigma_next) in enumerate(itertools.pairwise(sigmas)):
            coords, denoised = self.step(
                denoise,
                coords,
                sigma,
                sigma_next,
                step_index=index,
                num_steps=len(sigmas) - 1,
                **step_options,
            )
            if return_trajectory:
                trajectory.append(coords.clone())
                denoised_trajectory.append(
                    None if denoised is None else denoised.clone()
                )
        return (
            (coords, trajectory, denoised_trajectory) if return_trajectory else coords
        )

    def sample(
        self,
        denoise: ModelFn,
        shape: tuple[int, ...],
        sigmas: torch.Tensor,
        *,
        device=None,
        dtype=torch.float32,
        generator=None,
        chunk_size: int | None = None,
        coordinate_dims: int = 1,
        initialize_all: bool = False,
        return_trajectory: bool = False,
        **step_options,
    ):
        """Draw initial noise and integrate sample chunks with a stable RNG order."""
        if len(shape) < coordinate_dims + 2 or shape[-1] != 3:
            msg = "Coordinates require a sample axis followed by atom axes and xyz"
            raise ValueError(msg)
        sample_axis = -(coordinate_dims + 2)
        count = shape[sample_axis]
        if count < 1 or (chunk_size is not None and chunk_size < 1):
            msg = "Sample and chunk counts must be positive"
            raise ValueError(msg)
        size = count if chunk_size is None else min(chunk_size, count)
        if return_trajectory and size != count:
            msg = "Chunked trajectories must be requested per chunk"
            raise ValueError(msg)

        def initial(local_shape):
            return sigmas[0] * torch.randn(
                local_shape, device=device, dtype=dtype, generator=generator
            )

        all_coords = initial(shape) if initialize_all else None
        chunks = []
        for start in range(0, count, size):
            current = min(size, count - start)
            local_shape = list(shape)
            local_shape[sample_axis] = current
            coords = (
                initial(tuple(local_shape))
                if all_coords is None
                else all_coords.narrow(sample_axis, start, current)
            )
            result = self.integrate(
                denoise,
                coords,
                sigmas,
                generator=generator,
                coordinate_dims=coordinate_dims,
                return_trajectory=return_trajectory,
                **step_options,
            )
            if return_trajectory:
                return result
            chunks.append(result)
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=sample_axis)
