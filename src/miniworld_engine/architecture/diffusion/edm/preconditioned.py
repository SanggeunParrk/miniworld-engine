"""EDM network preconditioning around the common Euler sampler."""

from __future__ import annotations

from functools import partial

import torch
from pydantic import Field

from miniworld_engine.architecture.diffusion.augmentation import random_rigid_motion
from miniworld_engine.architecture.diffusion.edm.sampling import (
    EulerSampler,
    PreconditionedDenoiser,
)
from miniworld_engine.architecture.diffusion.edm.schedules import cap_noise_schedule
from miniworld_engine.architecture.utils.transform import weighted_kabsch_align


class PreconditionedEulerSolver:
    """A scaled-coordinate network with optional rigid alignment and sigma cap."""

    class Config(EulerSampler.Config):
        augment_per_step: bool = True
        translation_noise: float = Field(default=1.0, ge=0)
        align_to_denoised: bool = True
        max_sigma: float | None = Field(default=256.0, gt=0)
        clamp_variance: bool = True

    def __init__(self, config: Config, scheduler) -> None:
        self.config, self.scheduler = config, scheduler
        self.sampler = EulerSampler(
            EulerSampler.Config.model_validate(
                {
                    name: getattr(config, name)
                    for name in EulerSampler.Config.model_fields
                }
            )
        )

    def solver_sigmas(self, num_steps: int, device=None) -> torch.Tensor:
        sigmas = self.scheduler.sampling_time_steps(num_steps).to(device)
        return cap_noise_schedule(sigmas, self.config.max_sigma)

    def _options(self, mask, generator):
        def align(noisy, denoised):
            weights = (
                mask.to(denoised.dtype)
                if mask is not None
                else denoised.new_ones(denoised.shape[:-1])
            )
            return weighted_kabsch_align(
                noisy.float(), denoised.float(), weights, None
            ).to(denoised.dtype)

        return {
            "augment": partial(
                random_rigid_motion,
                mask=mask,
                trans_scale=self.config.translation_noise,
                generator=generator,
            )
            if self.config.augment_per_step
            else None,
            "align": align if self.config.align_to_denoised else None,
        }

    def denoise_step(
        self, denoise_fn, coords, sigma, sigma_next, *, generator=None, mask=None
    ):
        return self.sampler.step(
            PreconditionedDenoiser(denoise_fn, self.scheduler),
            coords,
            sigma,
            sigma_next,
            generator=generator,
            **self._options(mask, generator),
        )

    def sample(
        self,
        denoise_fn,
        shape,
        num_steps,
        device=None,
        *,
        generator=None,
        mask=None,
        return_trajectory=False,
    ):
        result = self.sampler.sample(
            PreconditionedDenoiser(denoise_fn, self.scheduler),
            tuple(shape),
            self.solver_sigmas(num_steps, device),
            device=device,
            generator=generator,
            return_trajectory=return_trajectory,
            **self._options(mask, generator),
        )
        return result[:2] if return_trajectory else result
