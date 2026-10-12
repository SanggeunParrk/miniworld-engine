"""Legacy AF3 interface over the shared EDM sampler.

The constructor's historical seed/backend settings remain for existing callers.
New applications use EulerSampler or PreconditionedEulerSolver with an explicit
Generator; neither changes process-global settings.
"""

from __future__ import annotations

from functools import partial

import torch

from miniworld_engine.architecture.diffusion.augmentation import scipy_rigid_motion
from miniworld_engine.architecture.diffusion.base.solver import DiffusionSolver
from miniworld_engine.architecture.diffusion.edm.sampling import (
    EulerSampler,
    PreconditionedDenoiser,
)


class AF3Solver(DiffusionSolver):
    def __init__(self, config, scheduler):
        super().__init__(config, scheduler)
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5
        self.aug_translation = 1.0

    def _sampler(self):
        return EulerSampler(
            EulerSampler.Config(
                gamma_0=self.gamma_0,
                gamma_min=self.gamma_min,
                noise_scale=self._lambda,
                step_scale=self.step_scale,
            )
        )

    def _denoiser(self, model_fn):
        return PreconditionedDenoiser(
            model_fn, self.scheduler, condition_noise=True, batch_noise=False
        )

    @torch.no_grad()
    def _centre_random_augmentation(self, coords, mask=None):
        return scipy_rigid_motion(coords, mask, self.aug_translation)

    def step(self, model_fn, x, t_index, time_steps, mask=None):
        return self._sampler().step(
            self._denoiser(model_fn),
            x,
            self.scheduler.sampling_schedule(time_steps[t_index]),
            self.scheduler.sampling_schedule(time_steps[t_index + 1]),
            augment=partial(self._centre_random_augmentation, mask=mask),
        )

    def sample(
        self,
        model_fn,
        shape,
        num_steps,
        device,
        *,
        mask=None,
        return_intermediate=False,
    ):
        times = self.scheduler.sampling_time_steps(num_steps).to(device)
        sigmas = torch.stack([self.scheduler.sampling_schedule(t) for t in times])
        initial = torch.randn(shape, device=device) * sigmas[0]
        return self._sampler().integrate(
            self._denoiser(model_fn),
            initial,
            sigmas,
            augment=partial(self._centre_random_augmentation, mask=mask),
            return_trajectory=return_intermediate,
        )
