"""Power-law inference schedules and log-normal training noise."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field


def cap_noise_schedule(sigmas: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Start at a requested cap, preserving the released endpoint convention."""
    return (
        sigmas
        if cap is None
        else torch.cat((sigmas.new_full((1,), cap), sigmas[sigmas <= cap]))
    )


class PowerLawSchedule:
    """Construct a schedule while preserving its endpoint and grid conventions."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        sigma_data: float = Field(default=16.0, gt=0)
        sigma_max: float = Field(default=160.0, gt=0)
        sigma_min: float = Field(default=4e-4, gt=0)
        rho: float = Field(default=7.0, gt=0)
        grid: Literal["linspace", "arange"] = "linspace"
        terminal_zero: bool = True
        include_minimum_before_zero: bool = True
        max_sigma: float | None = Field(default=None, gt=0)

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def levels(self, times: torch.Tensor) -> torch.Tensor:
        config = self.config
        upper = config.sigma_max ** (1 / config.rho)
        lower = config.sigma_min ** (1 / config.rho)
        return config.sigma_data * (upper + times * (lower - upper)) ** config.rho

    def __call__(
        self, num_steps: int, device=None, dtype=torch.float32
    ) -> torch.Tensor:
        if num_steps < 1:
            msg = "num_steps must be positive"
            raise ValueError(msg)
        config = self.config
        count = (
            num_steps
            if config.terminal_zero and config.include_minimum_before_zero
            else num_steps + 1
        )
        if config.grid == "arange":
            times = torch.arange(count, device=device, dtype=dtype) * (1 / num_steps)
        else:
            times = torch.linspace(0, 1, count, device=device, dtype=dtype)
        levels = self.levels(times)
        if config.terminal_zero:
            if config.include_minimum_before_zero:
                levels = torch.cat((levels, levels.new_zeros(1)))
            else:
                levels[-1] = 0
        return cap_noise_schedule(levels, config.max_sigma)


class LogNormalNoise:
    """Noise draws without changing the caller's generator or global settings."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        mean: float = -1.2
        std: float = Field(default=1.5, ge=0)
        sigma_data: float = Field(default=16.0, gt=0)

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def __call__(self, shape, device=None, *, generator=None):
        noise = torch.randn(shape, device=device, generator=generator)
        return (
            noise * self.config.std + self.config.mean
        ).exp() * self.config.sigma_data


class InferenceNoiseScheduler(PowerLawSchedule):
    """Compatibility constructor; computation belongs to PowerLawSchedule."""

    def __init__(self, s_max=160.0, s_min=4e-4, rho=7, sigma_data=16.0):
        super().__init__(
            self.Config(
                sigma_max=s_max,
                sigma_min=s_min,
                rho=rho,
                sigma_data=sigma_data,
                grid="arange",
                include_minimum_before_zero=False,
            )
        )

    def __call__(self, N_step=200, device="cpu", dtype=torch.float32):
        return super().__call__(N_step, device, dtype)


class TrainingNoiseSampler(LogNormalNoise):
    """Compatibility constructor for existing training configurations."""

    def __init__(self, p_mean=-1.2, p_std=1.5, sigma_data=16.0):
        super().__init__(self.Config(mean=p_mean, std=p_std, sigma_data=sigma_data))

    def __call__(self, size, device="cpu", *, generator=None):
        return super().__call__(size, device, generator=generator)
