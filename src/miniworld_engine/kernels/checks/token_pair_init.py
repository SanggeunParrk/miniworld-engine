"""Numerical checks for the ``token_pair_init`` family (portable kernels), against the dense fp32 reference (`reference.token_pair_init_reference`): the 139-wide
one-hot Linear the model ran, evaluated in exact fp32 (TF32 off)."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _fixed, _no_tf32
from miniworld_engine.kernels.drivers import dev
from miniworld_engine.kernels.drivers.token_pair_init import _B, _L, _NREL, _RMAX, _SMAX, tpi_inputs


def _reference(left, right, tbl, ids, bond, dz):
    """(z, (dleft, dright, dtbl)) of the dense reference from the launchers' packed operands."""
    from miniworld_engine.kernels.token_pair_init.reference import token_pair_init_reference

    b, n = left.shape[:2]
    a, r, t, e, s = (ids[k].reshape(b, n).long() for k in range(5))
    leaves = [x.detach().clone().requires_grad_() for x in (left, right, tbl[:_NREL].t(), tbl[_NREL:].t())]
    z = token_pair_init_reference(*leaves, a, r, t, e, s, bond.reshape(b, n, n).long(), _RMAX, _SMAX)
    dl, dr, dwr, dwb = torch.autograd.grad(z, leaves, dz)
    return z.detach(), (dl, dr, torch.cat([dwr.t(), dwb.t()], 0))


def _both():
    from miniworld_engine.kernels.token_pair_init.triton import portable_backward, portable_forward

    left, right, tbl, ids, bond = tpi_inputs()
    dz = torch.randn(_B, _L, _L, 128, device=dev())
    with _no_tf32():
        z_ref, g_ref = _reference(left, right, tbl, ids, bond, dz)
    return portable_forward(left, right, tbl, ids, bond, _RMAX, _SMAX), portable_backward(dz, ids, bond, _RMAX, _SMAX), z_ref, g_ref


def token_pair_init_fwd_triton():
    _fixed()
    z, _, z_ref, _ = _both()
    return {"z": (z, z_ref)}


def token_pair_init_bwd_dx_triton():
    _fixed()
    _, g, _, g_ref = _both()
    return {"dleft": (g[0], g_ref[0]), "dright": (g[1], g_ref[1])}


def token_pair_init_dw_triton():
    _fixed()
    _, g, _, g_ref = _both()
    return {"dtbl": (g[2], g_ref[2])}


def token_pair_init_dw_reduce_triton():
    """The slice reduction on its own: random partials summed in slice order."""
    import triton

    from miniworld_engine.kernels.token_pair_init.triton import main

    _fixed()
    part = torch.randn(main.NBLK, _NREL + 2, 128, device=dev())
    total = (_NREL + 2) * 128
    out = torch.empty(_NREL + 2, 128, device=dev())
    main._tpi_dw_reduce[lambda meta: (triton.cdiv(total, meta["BLOCK"]),)](part, out, total, NBLK=main.NBLK, shape_key=main.token_key(_L))
    return {"dtbl": (out, part.sum(0))}
