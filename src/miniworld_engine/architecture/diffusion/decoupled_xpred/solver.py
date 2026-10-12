"""VE x-prediction ODE solver (Decoupled coordinate + rigid-body SE(3)).

R/T are auxiliary model-input corruptions sampled fresh at each step from
their marginal at sigma_hat — they carry no state across steps, so they do
not appear in the public step/sample interface.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel

from miniworld_engine.architecture.diffusion.base.solver import (
    DiffusionSolver,
    ModelFn,
    _chain_count,
    _expand_to_batch,
)
from miniworld_engine.architecture.diffusion.decoupled_xpred.scheduler import (
    DecoupledXPredScheduler,
)
from miniworld_engine.architecture.utils.align import weighted_align
from miniworld_engine.architecture.utils.se3 import apply_chain_rt, sample_rigid


class XPredDecoupledSolver(DiffusionSolver):
    """EDM ODE solver with x-prediction for decoupled coordinate + R/T noise.

    The model outputs x0/sigma_data. The solver multiplies by sigma_data
    to recover x_pred in original coordinates for the ODE step.
    """

    class Config(BaseModel):
        """Configuration for XPredDecoupledSolver."""

        seed: int = 0
        gamma_0: float = 0.8
        gamma_min: float = 1.0
        noise_lambda: float = 1.003
        step_scale: float = 1.5

    def __init__(
        self,
        config: Config,
        scheduler: DecoupledXPredScheduler,
    ) -> None:
        self.config = config
        self.scheduler: DecoupledXPredScheduler = scheduler
        self._set_seed(config.seed)

    @property
    def sigma_data(self) -> float:
        """Data standard deviation from scheduler config."""
        return self.scheduler.config.sigma_data

    def _sample_rt(
        self,
        sigma: torch.Tensor,
        batch_size: int,
        chain_num: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample R/T marginal at given sigma."""
        sigma_r, sigma_t = self.scheduler.convert_to_sigma_rt(sigma)
        sigma_r = _expand_to_batch(sigma_r, batch_size)
        sigma_t = _expand_to_batch(sigma_t, batch_size)
        return sample_rigid(
            sigma_r,
            sigma_t,
            C=chain_num,
            device=device,
            dtype=dtype,
        )

    def _center_to_origin(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Translate each structure so the valid-atom centroid is at the origin."""
        if mask is None:
            centroid = x.mean(dim=-2, keepdim=True)
            return x - centroid

        mask = self._prepare_weight(x, mask)

        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        centroid = (x * mask.unsqueeze(-1)).sum(dim=-2, keepdim=True) / denom.unsqueeze(
            -1
        )
        x_centered = x - centroid
        return x_centered * mask.unsqueeze(-1)

    def _prepare_weight(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand the atom mask to match the current batch shape."""
        if mask is None:
            return torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)

        weight = mask.to(device=x.device, dtype=x.dtype)
        if weight.ndim == 1:
            weight = weight.unsqueeze(0).expand(x.shape[0], -1)
        elif weight.shape[0] == 1 and x.shape[0] > 1:
            weight = weight.expand(x.shape[0], -1)
        return weight

    def _align_to_prediction(
        self,
        y: torch.Tensor,
        x_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rigidly align the current iterate to the predicted x0 frame."""
        weight = self._prepare_weight(y, mask)
        y_aligned = weighted_align(y, x_pred, weight=weight)
        return torch.where(weight.unsqueeze(-1) > 0, y_aligned, y)

    def step(
        self,
        model_fn: ModelFn,
        y: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_to_combine: torch.Tensor,
        *,
        use_rt: bool = True,
        mask: torch.Tensor | None = None,
        update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One solver step on coordinates.

        R/T are sampled fresh from their marginal at sigma_hat (not stateful).
        Returns (y_next, x_pred, x_with_noise), where x_with_noise is the
        R/T-corrupted model input at sigma_hat (pre input-scaling).
        """
        sigma_i = self.scheduler.sampling_schedule(time_steps[t_index])
        sigma_next = self.scheduler.sampling_schedule(time_steps[t_index + 1])
        batch_size = y.shape[0]
        chain_num = _chain_count(atom_to_combine)

        # Stochastic noise injection (EDM Euler)
        gamma = self.config.gamma_0 if sigma_next > self.config.gamma_min else 0
        sigma_hat = sigma_i * (1 + gamma)
        # AF3 Algorithm 18 uses the pre-injection iterate (x_l) for the ODE
        # delta; keep a handle before overwriting y with x_noisy.
        y_pre_inject = y
        if gamma > 0:
            added_noise = (
                self.config.noise_lambda
                * (sigma_hat**2 - sigma_i**2) ** 0.5
                * torch.randn_like(y)
            )
            y = y + added_noise

        # Sample R/T marginal at sigma_hat for model-input corruption unless disabled.
        if use_rt:
            rotation_hat, translation_hat = self._sample_rt(
                sigma_hat,
                batch_size,
                chain_num,
                y.device,
                y.dtype,
            )
            x_with_noise = apply_chain_rt(
                y,
                rotation_hat,
                translation_hat,
                atom_to_combine,
            )
            _, sigma_t_hat = self.scheduler.convert_to_sigma_rt(sigma_hat)
        else:
            x_with_noise = y
            sigma_t_hat = torch.zeros_like(sigma_hat)

        # Model prediction
        t_emb = self.scheduler.noise_condition(sigma_hat)
        c_in = self.scheduler.input_scale(sigma_hat, sigma_t_hat)
        x_pred = model_fn(x_with_noise * c_in, t_emb) * self.sigma_data

        if update_rule == "ode":
            v = (y_pre_inject - x_pred) / sigma_hat
            y_next = y + self.config.step_scale * (sigma_next - sigma_hat) * v
        elif update_rule == "ode_aligned":
            y_aligned = self._align_to_prediction(y, x_pred, mask)
            v = (y_aligned - x_pred) / sigma_hat
            y_next = y_aligned + self.config.step_scale * (sigma_next - sigma_hat) * v
        elif update_rule == "x0_centered":
            x_pred = self._center_to_origin(x_pred, mask)
            is_last_step = t_index + 1 == time_steps.shape[0] - 1
            if is_last_step:
                y_next = x_pred
            else:
                noise_scale = (
                    1.0 - self.config.step_scale
                ) * sigma_hat + self.config.step_scale * sigma_next
                y_next = x_pred + noise_scale * torch.randn_like(y)
        else:
            msg = f"Unsupported update_rule: {update_rule}"
            raise ValueError(msg)

        return y_next, x_pred, x_with_noise

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelFn,
        shape: torch.Size,
        atom_to_combine: torch.Tensor,
        num_steps: int,
        device: torch.device,
        *,
        use_rt: bool = True,
        mask: torch.Tensor | None = None,
        update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered",
        return_intermediate: bool = False,
        combine_all: bool = False,
        init_x0: torch.Tensor | None = None,
        start_sigma_y: float | None = None,
    ) -> (
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | torch.Tensor
    ):
        """Sample from noise using the selected x-prediction update rule.

        When return_intermediate is True, returns
        (y_final, inter_traj, hat_list, input_traj) where input_traj[i] is the
        R/T-corrupted model input at step i (pre input-scaling).

        When combine_all is True, all atoms are treated as a single group so a
        single R/T is sampled and applied to the whole structure regardless of
        the chain layout in atom_to_combine.

        Flexible-docking warm start: pass ``init_x0`` (``(N_atom, 3)`` or
        ``(B, N_atom, 3)``, broadcastable to ``shape``) of known coords with
        each combine-group already centered at its centroid, plus
        ``start_sigma_y`` (default :attr:`scheduler.phase_1_boundary`). The
        time-step schedule is rebuilt from ``start_sigma_y`` instead of
        ``sigma_y_max``, and ``y0 = init_x0 + sigma_0 * randn`` replaces the
        usual full-noise init. The first step's ``apply_chain_rt`` then
        samples R/T from the marginal at ``sigma_hat ~ start_sigma_y``
        (phase-1 = max ``sigma_R`` / ``sigma_T``), so each group lands at a
        random pose while its internal coords are preserved.
        """
        if combine_all:
            if init_x0 is not None:
                msg = (
                    "init_x0 (flexible-docking warm start) is incompatible "
                    "with combine_all=True: combine_all zeroes the per-atom "
                    "group ids so all atoms move as one rigid body, which "
                    "defeats the per-group warm start."
                )
                raise ValueError(msg)
            atom_to_combine = torch.zeros_like(atom_to_combine)

        if init_x0 is None and start_sigma_y is not None:
            msg = (
                "start_sigma_y is only meaningful together with init_x0 — "
                "the standard solver always starts at sigma_y_max."
            )
            raise ValueError(msg)

        if init_x0 is not None:
            if start_sigma_y is None:
                start_sigma_y = self.scheduler.phase_1_boundary
            time_steps = self.scheduler.sampling_time_steps(
                num_steps,
                start_sigma_y=start_sigma_y,
            ).to(device)
        else:
            time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])

        if init_x0 is not None:
            init_x0 = init_x0.to(device=device, dtype=torch.float32)
            if init_x0.ndim == 2:
                init_x0 = init_x0.unsqueeze(0)
            if init_x0.shape[-2:] != shape[-2:]:
                msg = (
                    f"init_x0 shape {tuple(init_x0.shape)} does not match "
                    f"solver shape {tuple(shape)}; last two dims must agree."
                )
                raise ValueError(msg)
            init_x0 = init_x0.expand(shape)
            y = init_x0 + sigma_0 * torch.randn(shape, device=device)
        else:
            y = torch.randn(shape, device=device) * sigma_0
        trajectory: list[torch.Tensor] = []
        hat_list: list[torch.Tensor] = []
        input_list: list[torch.Tensor] = []

        for i in range(num_steps):
            y, x_pred, x_with_noise = self.step(
                model_fn,
                y,
                i,
                time_steps,
                atom_to_combine,
                use_rt=use_rt,
                mask=mask,
                update_rule=update_rule,
            )
            if return_intermediate:
                trajectory.append(y.clone())
                hat_list.append(x_pred.clone())
                input_list.append(x_with_noise.clone())

        if return_intermediate:
            return y, trajectory, hat_list, input_list
        return y
