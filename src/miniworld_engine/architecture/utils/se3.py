from __future__ import annotations

import math
from collections.abc import Mapping
from math import pi

import numpy as np
import torch
from torch import Tensor


def hat_so3(omega: Tensor) -> Tensor:
    """(B,3) -> (B,3,3) skew matrix."""
    wx, wy, wz = omega.unbind(-1)
    O = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    O[..., 0, 1] = -wz
    O[..., 0, 2] = wy
    O[..., 1, 0] = wz
    O[..., 1, 2] = -wx
    O[..., 2, 0] = -wy
    O[..., 2, 1] = wx
    return O


def rodrigues(omega: Tensor) -> Tensor:
    """Axis-angle to rotation matrix using Rodrigues. omega: (B,3)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = omega / theta
    K = hat_so3(k)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    I = I.expand(*omega.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    R = I + s * K + (1 - c) * (K @ K)
    # For very small angles, use 1st order approx
    small = (theta.squeeze(-1) < 1e-4)[..., None, None]
    R_small = I + hat_so3(omega)
    return torch.where(small, R_small, R)


def exp_se3(omega: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Exponential map from (omega,v) in se(3) to (R,T).

    omega, v: (B,3). Returns R: (B,3,3), T: (B,3).
    """
    B = omega.shape[0]
    R = rodrigues(omega)
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = hat_so3(omega / theta)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(B, 3, 3)
    s = torch.sin(theta)
    c = torch.cos(theta)
    A = s / theta
    Bcoef = (1 - c) / (theta**2)
    V = I + A[..., None] * K + Bcoef[..., None] * (K @ K)
    T = (V @ v.unsqueeze(-1)).squeeze(-1)
    return R, T


def apply_rt(R: Tensor, T: Tensor, pts: Tensor) -> Tensor:
    """Apply (R,T) to points. R:(B,3,3) or (3,3), T:(B,3) or (3,), pts:(...,3)."""
    if R.dim() == 2:
        return pts @ R.mT + T
    # batch over first dim of pts assumed to match R,T or be broadcastable
    return (pts @ R.transpose(-1, -2)) + T.unsqueeze(-2)


def apply_chain_rt(
    x: Tensor,
    R: Tensor,
    T: Tensor,
    atom_to_combine: Tensor | Mapping[object, tuple[int, int]],
    inverse: bool = False,
) -> Tensor:
    """Apply chain-specific RT. x:(B,N,3), R,T:(B,C,3,3)/(B,C,3), atom_to_combine: (B, N_atom)."""
    if isinstance(atom_to_combine, Mapping):
        output = []
        for cc, chain_ID in enumerate(atom_to_combine.keys()):
            atom_start, atom_end = atom_to_combine[chain_ID]
            x_rt = x[:, atom_start : atom_end + 1]
            if inverse:
                _R = R[:, cc].mT
                _T = torch.einsum("bij,bj->bi", R[:, cc].mT, -T[:, cc])
                x_rt = apply_rt(_R, _T, x_rt)
            else:
                x_rt = apply_rt(R[:, cc], T[:, cc], x_rt)
            output.append(x_rt)
        output = torch.cat(output, dim=1)
        return output

    # atom_to_combine: (B, N_atom) — group index per atom, values in [0, C)
    atom_to_combine = atom_to_combine.to(device=x.device, dtype=torch.long)
    if atom_to_combine.ndim == 1:
        atom_to_combine = atom_to_combine.expand(x.shape[0], -1)
    if atom_to_combine.shape != x.shape[:2]:
        msg = (
            f"atom_to_combine shape {atom_to_combine.shape} "
            f"does not match x shape {x.shape[:2]}."
        )
        raise ValueError(msg)
    # vectorized bounds check — stays on device, no GPU-CPU sync
    if atom_to_combine.numel() > 0 and (
        atom_to_combine.lt(0).any() or atom_to_combine.ge(R.shape[1]).any()
    ):
        msg = (
            f"atom_to_combine values must be in [0, {R.shape[1]}), "
            f"got min={atom_to_combine.min()}, max={atom_to_combine.max()}."
        )
        raise ValueError(msg)

    # gather per-atom RT: R_atom/T_atom shape (B, N_atom, 3, 3)/(B, N_atom, 3)
    batch_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1)  # (B, 1)
    R_atom = R[batch_idx, atom_to_combine]  # (B, N_atom, 3, 3)
    T_atom = T[batch_idx, atom_to_combine]  # (B, N_atom, 3)
    if inverse:
        R_atom = R_atom.transpose(-1, -2)  # R^T
        T_atom = torch.matmul(R_atom, -T_atom.unsqueeze(-1)).squeeze(-1)  # -R^T T
    return torch.matmul(x.unsqueeze(-2), R_atom.transpose(-1, -2)).squeeze(-2) + T_atom


def log_so3(R: Tensor) -> Tensor:
    """Log map for SO(3) (stable for near-identity). Returns axis-angle (3,)."""
    # Using formula with clamped trace
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((tr - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    # For small angles, use approximation
    small = theta < 1e-4
    if small.any():
        # skew = (R - R^T)/2, vee(skew) is approx omega
        skew = 0.5 * (R - R.transpose(-1, -2))
        omega_approx = torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
            dim=-1,
        )
        omega = omega_approx
    else:
        denom = 2 * torch.sin(theta)
        skew = (R - R.transpose(-1, -2)) / denom[..., None, None]
        omega = theta.unsqueeze(-1) * torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
            dim=-1,
        )
    return omega


def compose(R1: Tensor, T1: Tensor, R2: Tensor, T2: Tensor) -> tuple[Tensor, Tensor]:
    """(R1,T1)∘(R2,T2)."""
    R = R1 @ R2
    T = (R1 @ T2.unsqueeze(-1)).squeeze(-1) + T1
    return R, T


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues formula for exp map from so(3) to SO(3). omega: (B,C,3) axis-angle."""
    B, C = omega.shape[:2]
    omega = omega.view(-1, 3)
    theta = omega.norm(dim=-1, keepdim=True)  # (C,1)
    u = torch.where(theta > 0, omega / theta, torch.zeros_like(omega))
    ux, uy, uz = u.unbind(-1)

    K = torch.zeros(B * C, 3, 3, device=omega.device, dtype=omega.dtype)
    K[:, 0, 1] = -uz
    K[:, 0, 2] = uy
    K[:, 1, 0] = uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] = ux

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(B * C, 3, 3)
    th = theta.squeeze(-1)

    # small-angle safe sin/cos
    s = torch.where(th > 0, torch.sin(th), th - th**3 / 6 + th**5 / 120)
    c = torch.where(th > 0, torch.cos(th), 1 - th**2 / 2 + th**4 / 24)

    a = s.view(B * C, 1, 1)
    b = (1 - c).view(B * C, 1, 1)
    R = I + a * K + b * (K @ K)
    R = R.view(B, C, 3, 3)
    return R


class _IGSO3Sampler:
    """Direct sampler for the IGSO3 distribution via precomputed inverse CDF.

    Replaces the O(steps) Brownian-motion simulation with O(1) sampling
    using a 2D lookup table over (epsilon, omega).  The table is built
    lazily on first use and cached per device.
    """

    def __init__(
        self,
        n_eps: int = 512,
        n_omega: int = 4096,
        eps_min: float = 1e-4,
        eps_max: float = 1100.0,
        L_max: int = 1500,
        small_sigma_thresh: float = 0.01,
    ) -> None:
        self.n_eps = n_eps
        self.n_omega = n_omega
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.L_max = L_max
        self.small_sigma_thresh = small_sigma_thresh
        self._cdf_table: Tensor | None = None
        self._omega_grid: Tensor | None = None
        self._log_eps_grid: Tensor | None = None

    def _precompute(self, device: torch.device) -> None:
        """Build the 2D CDF table ``(n_eps, n_omega)`` once."""
        omega = torch.linspace(
            1e-6,
            pi,
            self.n_omega,
            dtype=torch.float64,
            device=device,
        )
        log_eps = torch.linspace(
            math.log(self.eps_min),
            math.log(self.eps_max),
            self.n_eps,
            dtype=torch.float64,
            device=device,
        )
        eps = torch.exp(log_eps)

        ls = torch.arange(self.L_max + 1, dtype=torch.float64, device=device)
        # coefficients: (n_eps, L+1)
        coeffs = (2 * ls + 1).unsqueeze(0) * torch.exp(
            -ls * (ls + 1) * eps.unsqueeze(1) / 2,
        )
        # characters: (n_omega, L+1)
        half_omega = omega.unsqueeze(1) / 2
        sin_half = torch.sin(half_omega).clamp_min(1e-12)
        characters = torch.sin((2 * ls + 1) * half_omega) / sin_half
        # Haar measure factor
        haar = (1 - torch.cos(omega)) / pi

        pdf = coeffs @ characters.T * haar.unsqueeze(0)  # (n_eps, n_omega)
        pdf = pdf.clamp_min(0)

        d_omega = omega[1] - omega[0]
        cdf = torch.cumsum(pdf * d_omega, dim=1)
        cdf = cdf / cdf[:, -1:].clamp_min(1e-30)

        self._omega_grid = omega.float()
        self._log_eps_grid = log_eps.float()
        self._cdf_table = cdf.float()

    def _ensure_table(self, device: torch.device) -> None:
        if self._cdf_table is None or self._cdf_table.device != device:
            self._precompute(device)

    @torch.no_grad()
    def sample(
        self,
        sigma_R: Tensor,
        C: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample ``(B, C, 3, 3)`` rotation matrices from IGSO3(sigma_R).

        * For sigma < ``small_sigma_thresh``: tangent-space Gaussian (exact
          to first order, covers the near-identity regime cheaply).
        * Otherwise: inverse-CDF lookup from the precomputed table.
        """
        if device is None:
            device = sigma_R.device
        B = sigma_R.shape[0]
        R = (
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, C, 3, 3)
            .clone()
        )

        small = sigma_R < self.small_sigma_thresh
        large = ~small

        # --- Small sigma: tangent-space Gaussian (exact to first order) ---
        if small.any():
            n_s = int(small.sum().item())
            std = sigma_R[small].to(dtype).view(n_s, 1, 1)
            omega_s = torch.randn(n_s, C, 3, device=device, dtype=dtype) * std
            R[small] = exp_so3(omega_s)

        # --- Larger sigma: inverse-CDF from precomputed table ---
        if large.any():
            self._ensure_table(device)
            sigma_l = sigma_R[large]
            n_l = sigma_l.shape[0]

            eps = (sigma_l**2).clamp(self.eps_min, self.eps_max)
            log_eps = torch.log(eps).float()
            idx = torch.searchsorted(self._log_eps_grid, log_eps).clamp(
                0,
                self.n_eps - 1,
            )
            cdf_rows = self._cdf_table[idx]  # (n_l, n_omega)

            u = torch.rand(n_l, C, device=device)
            omega_idx = torch.searchsorted(cdf_rows, u).clamp(0, self.n_omega - 1)
            angles = self._omega_grid[omega_idx]  # (n_l, C)

            axes = torch.randn(n_l, C, 3, device=device, dtype=dtype)
            axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            omega_vec = axes * angles.to(dtype).unsqueeze(-1)  # (n_l, C, 3)
            R[large] = exp_so3(omega_vec)

        return R


_igso3_sampler = _IGSO3Sampler()


def sample_rotation_heat(
    sigma_R: torch.Tensor,
    C: int,
    steps: int = 64,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample from IGSO3 (heat kernel on SO(3)) via precomputed inverse CDF.

    This replaces the previous O(steps) Brownian-motion simulation with
    O(1) direct sampling.  The ``steps`` parameter is kept for backward
    compatibility but is no longer used.
    """
    del steps  # no longer needed — kept for API compat
    return _igso3_sampler.sample(sigma_R, C, device=device, dtype=dtype)


def sample_translation(sigma_T: torch.Tensor, C: int) -> torch.Tensor:
    """Sample translation noise."""
    B = sigma_T.shape[0]
    sigma_T = sigma_T[:, None, None]  # (B,1,1)
    return torch.randn(B, C, 3, device=sigma_T.device, dtype=sigma_T.dtype) * sigma_T


def sample_rigid(
    sigma_R: torch.Tensor,
    sigma_T: torch.Tensor,
    C: int,
    steps_R: int = 64,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Sample (R,T) from SE(3) heat kernel with rotation variance sigma_R^2 and translation variance sigma_T^2."""
    R = sample_rotation_heat(sigma_R, C, steps=steps_R, device=device, dtype=dtype)
    T = sample_translation(sigma_T, C)
    return R, T


def se3_heat_step_sigma(
    R_t: Tensor,
    T_t: Tensor,
    sigma_R_t: Tensor,
    sigma_T_t: Tensor,  # (B,) or scalar
    sigma_R_s: Tensor,
    sigma_T_s: Tensor,  # target sigmas (B,) or scalar
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Given (R_t,T_t) at time t with noise levels sigma_R_t, sigma_T_t, compute (R_s,T_s) at time s with noise levels sigma_R_s, sigma_T_s by applying the SE(3) heat semigroup step. Handles both increasing and decreasing noise cases."""
    B = R_t.shape[0]
    device, dtype = R_t.device, R_t.dtype

    inc_R = sigma_R_s >= sigma_R_t  # forward (add variance)
    var_R = (sigma_R_s**2 - sigma_R_t**2).clamp_min(0.0)
    std_R = torch.sqrt(var_R + 0.0).view(B, 1, 1)  # (B,1,1)
    if inc_R.all():
        # forward: Brownian increment
        d_omega = torch.randn(B, R_t.shape[1], 3, device=device, dtype=dtype) * std_R
        R_s = R_t @ exp_so3(d_omega)  # right-invariant
    elif (~inc_R).all():
        # decreasing noise: deterministic shrink in small-angle algebra
        omega_t = log_so3(R_t)  # (B,C,3)
        scale = ((sigma_R_s + eps) / (sigma_R_t + eps)).view(B, 1, 1)
        R_s = exp_so3(scale * omega_t)  # around identity
    else:
        msg = "Mixed inc/dec batch not supported in this helper."
        raise ValueError(msg)
    # ---- Translation step (R^3 heat semigroup) ----
    inc_T = sigma_T_s >= sigma_T_t
    var_T = (sigma_T_s**2 - sigma_T_t**2).clamp_min(0.0)
    std_T = torch.sqrt(var_T + 0.0).view(B, 1, 1)  # (B,1,1)
    if inc_T.all():
        T_s = T_t + torch.randn_like(T_t) * std_T
    elif (~inc_T).all():
        scale_T = ((sigma_T_s + eps) / (sigma_T_t + eps)).view(B, 1, 1)
        T_s = scale_T * T_t
    else:
        msg = "Mixed inc/dec batch not supported in this helper."
        raise ValueError(msg)

    return R_s, T_s


def se3_heat_step_delta_sigma(
    R_t: Tensor,
    T_t: Tensor,
    sigma_R_t: Tensor,
    sigma_T_t: Tensor,  # current sigma (B,) or scalar
    d_sigma_R: Tensor,
    d_sigma_T: Tensor,  # Δσ can be + or -
    sigma_floor: float = 0.0,  # e.g., 1e-6
) -> tuple[Tensor, Tensor]:
    """Step from (R_t,T_t) with noise sigma_t to (R_s,T_s)."""
    device, dtype = R_t.device, R_t.dtype

    def _to(x: Tensor | float) -> Tensor:
        return torch.as_tensor(x, device=device, dtype=dtype).reshape(-1)

    sigma_R_t = _to(sigma_R_t)
    sigma_T_t = _to(sigma_T_t)
    d_sigma_R = _to(d_sigma_R)
    d_sigma_T = _to(d_sigma_T)

    B = R_t.shape[0]
    if sigma_R_t.numel() == 1:
        sigma_R_t = sigma_R_t.expand(B)
    if sigma_T_t.numel() == 1:
        sigma_T_t = sigma_T_t.expand(B)
    if d_sigma_R.numel() == 1:
        d_sigma_R = d_sigma_R.expand(B)
    if d_sigma_T.numel() == 1:
        d_sigma_T = d_sigma_T.expand(B)

    sigma_R_s = (sigma_R_t + d_sigma_R).clamp_min(sigma_floor)
    sigma_T_s = (sigma_T_t + d_sigma_T).clamp_min(sigma_floor)

    R_s, T_s = se3_heat_step_sigma(
        R_t,
        T_t,
        sigma_R_t,
        sigma_T_t,
        sigma_R_s,
        sigma_T_s,
    )
    return R_s, T_s


def SE3_oper(
    length: int,
    noise_level: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random SE(3) rotation matrices (length,3,3) and Gaussian noise (length,3)."""
    if rng is None:
        rng = np.random.default_rng()

    u1 = rng.random(size=length)
    u2 = rng.random(size=length)
    u3 = rng.random(size=length)

    r1 = np.sqrt(1 - u1)
    r2 = np.sqrt(u1)
    th1 = 2 * pi * u2
    th2 = 2 * pi * u3

    qw = r2 * np.cos(th2)
    qx = r1 * np.sin(th1)
    qy = r1 * np.cos(th1)
    qz = r2 * np.sin(th2)

    ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz

    R = np.empty((length, 3, 3), dtype=np.float32)

    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    noise = rng.standard_normal(size=(length, 3)).astype(np.float32) * noise_level

    return R, noise
