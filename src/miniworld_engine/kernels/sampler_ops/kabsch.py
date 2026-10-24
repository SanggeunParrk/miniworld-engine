"""Kabsch rotation by Horn's quaternion method: the dominant eigenvector of the symmetric 4x4 N(H) by repeated squaring of N + c I (no SVD,
no host read, so a CUDA-graph capture allows it). A proper rotation always results."""

from __future__ import annotations

import torch

#: ``(N + cI)^(2**SQUARINGS)``: the ratio of the two largest eigenvalues is raised to ``2**SQUARINGS``.
SQUARINGS = 12


def horn_matrix(h: torch.Tensor) -> torch.Tensor:
    """The symmetric 4x4 ``N`` of Horn (1987) from ``h[..., i, j] = sum_n w_n a[n, i] b[n, j]``."""
    sxx, sxy, sxz = h[..., 0, 0], h[..., 0, 1], h[..., 0, 2]
    syx, syy, syz = h[..., 1, 0], h[..., 1, 1], h[..., 1, 2]
    szx, szy, szz = h[..., 2, 0], h[..., 2, 1], h[..., 2, 2]
    rows = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def dominant_quaternion(n: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector of the largest eigenvalue of the symmetric 4x4 ``n`` (``[..., 4]``)."""
    # |eigenvalue| <= the trace norm of H <= sqrt(3) |H|_F, and the spectrum of N is the sums of the
    # signed singular values, so shifting by the Frobenius bound makes the matrix positive definite.
    shift = n.square().sum((-2, -1), keepdim=True).sqrt() * 1.0
    m = n + shift * torch.eye(4, dtype=n.dtype, device=n.device)
    for _ in range(SQUARINGS):
        m = m @ m
        m = m / m.abs().amax((-2, -1), keepdim=True).clamp_min(torch.finfo(n.dtype).tiny)
    # m ~ lambda^k q q^T: the column of largest norm is q up to scale (and sign).
    columns = m.transpose(-2, -1)
    norms = columns.square().sum(-1)
    pick = norms.argmax(-1, keepdim=True)
    q = torch.gather(columns, -2, pick[..., None].expand(*pick.shape, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(n.dtype).tiny)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (column-vector convention) of the unit quaternion ``q = (w, x, y, z)``."""
    w, x, y, z = q.unbind(-1)
    rows = [
        [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (y * x + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
        [2 * (z * x - w * y), 2 * (z * y + w * x), w * w - x * x - y * y + z * z],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def kabsch_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """``R`` with ``a @ R ~ b`` (row vectors) for ``covariance = a^T W b``; the same ``R`` as ``U D V^T`` of its SVD."""
    q = dominant_quaternion(horn_matrix(covariance))
    return quaternion_to_matrix(q).transpose(-2, -1)
