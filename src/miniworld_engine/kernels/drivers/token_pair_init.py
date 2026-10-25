"""Drivers for the ``token_pair_init`` family (portable kernels of the token-pair initialisation).

The kernels compute in fp32 (``left`` / ``right`` of a bf16 caller are cast by the autograd function before the launch); a driver therefore builds the streams at the
activation dtype and casts them, as the entry point does. The pair is ``[B, L, L, 128]`` fp32.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, driver_length, ragged

_B = 2
_L = ragged(driver_length(128))
_RMAX, _SMAX = 32, 2
_NREL = 2 * (2 * _RMAX + 2) + (2 * _SMAX + 2) + 1


def tpi_inputs(b: int = _B, n: int = _L):
    """``(left, right, tbl, ids, bond)`` as the launchers take them: ids [5, B*L] int32 (asym, residue, token, entity, sym), bond [B*L*L] uint8."""
    d = dev()
    ids = torch.stack([torch.randint(0, hi, (b, n), device=d, dtype=torch.int32) for hi in (3, 90, 60, 2, 6)]).reshape(5, b * n).contiguous()
    bond = (torch.rand(b * n * n, device=d) < 0.05).to(torch.uint8)
    left = torch.randn(b, n, 128, device=d, dtype=BF16).float().contiguous()
    right = torch.randn(b, n, 128, device=d, dtype=BF16).float().contiguous()
    return left, right, (torch.randn(_NREL + 2, 128, device=d) * 0.3), ids, bond


def token_pair_init_fwd_triton() -> None:
    from miniworld_engine.kernels.token_pair_init.triton import portable_forward

    left, right, tbl, ids, bond = tpi_inputs()
    portable_forward(left, right, tbl, ids, bond, _RMAX, _SMAX)


def _backward() -> None:
    from miniworld_engine.kernels.token_pair_init.triton import portable_backward

    _, _, _, ids, bond = tpi_inputs()
    portable_backward(torch.randn(_B, _L, _L, 128, device=dev()), ids, bond, _RMAX, _SMAX)


def token_pair_init_bwd_dx_triton() -> None:
    _backward()


def token_pair_init_dw_triton() -> None:
    _backward()


def token_pair_init_dw_reduce_triton() -> None:
    _backward()
