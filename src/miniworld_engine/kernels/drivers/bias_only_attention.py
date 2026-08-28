"""Drivers for the ``bias_only_attention`` family.

The three attention families were one module (``drivers_attn.py``) and still share the
``L``/``H``/``D`` extents, which live in ``drivers/triangle_attention.py`` together with the
shape, grad and tile-alignment rationale for all three. ``DH``/``DP`` -- the gate-out GEMM's
contraction and output widths -- are this family's own and stay here.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    TensorKw,
    _grad,
    dev,
    driver_width,
    ragged,
)
from miniworld_engine.kernels.drivers.triangle_attention import D, H, L

# Both follow the swept width, and they have to move TOGETHER with the imported `H`: H is
# d_pair // head dim, so an H that says d_pair=384 beside a DP that says 128 describes no d_pair
# the model ever runs, and DH is folded into the shape key.
DH = ragged(driver_width(128))          # d_hidden: gate/out_r width == the gate-out GEMM's contraction
DP = ragged(driver_width(128), by=5)    # d_pair: the gate-out GEMM's output width N == wo.shape[0]


def _bias_only_vb() -> tuple[torch.Tensor, ...]:
    """v [1, H, L, L, D] and bias [1, H, L, L] -- bench_kernel_bias_attn."""
    kw: TensorKw = {"device": dev(), "dtype": BF16, "requires_grad": True}
    return torch.randn(1, H, L, L, D, **kw), torch.randn(1, H, L, L, **kw)


# ── bias_only_attention / triton / main.py ──────────────────────────────────────────────────


def bias_only_attention_fwd_triton() -> None:
    from miniworld_engine.kernels.bias_only_attention.triton.main import (
        TritonBiasOnlyAttentionFunction as Fn,
    )
    Fn.apply(*_bias_only_vb())


def _bias_only_backward() -> None:
    from miniworld_engine.kernels.bias_only_attention.triton.main import (
        TritonBiasOnlyAttentionFunction as Fn,
    )
    _grad(Fn.apply(*_bias_only_vb()))


def bias_only_attention_bwd_pre_triton() -> None:
    _bias_only_backward()


def bias_only_attention_bwd_triton() -> None:
    _bias_only_backward()


# ── bias_only_attention / triton / gate_out.py ──────────────────────────────────────────────


def _pair_key() -> int:
    """``shape_key`` for the two gate-out launchers, computed the way the module computes it.

    ``_fwd`` and ``_dgrad_epilogue`` are INNER launchers: by the time they are called the
    activation is the flattened ``(M, DH)`` matrix and L is gone, so per
    ``autotune/shape_key.py::length_of`` they cannot derive the key themselves -- both take it as
    ``shape_key=`` and both fall back to ``token_key(0)`` (the BOTTOM bucket, 128) when it is
    omitted. The drivers below used to omit it, which pinned every driver length to bucket 128.
    Calling the module's own ``_key_of`` on the PRE-flatten pair shape ``(1, L, L, DH)`` is exactly
    what ``_FusedGateOut.forward``/``backward`` do, so the driver now records the bucket production
    records at this L.
    """
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _key_of
    return _key_of((1, L, L, DH))


def gated_projection_gate_gemm_triton() -> None:
    # M = L*L rows, DH the contraction, DP the output width -- all three tile, all three ragged.
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd
    kw: TensorKw = {"device": dev(), "dtype": BF16}
    gate = torch.randn(L * L, DH, **kw)
    out_r = torch.randn(L * L, DH, **kw)
    wo = torch.randn(DP, DH, **kw)          # to_out.weight [d_pair, d_hidden]
    _fwd(gate, out_r, wo, shape_key=_pair_key())


def gated_projection_bwd_dx_triton() -> None:
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import (
        _dgrad_epilogue,
    )
    kw: TensorKw = {"device": dev(), "dtype": BF16}
    do2 = torch.randn(L * L, DP, **kw)      # grad wrt [M, N], N == d_pair
    wo = torch.randn(DP, DH, **kw)
    g2 = torch.randn(L * L, DH, **kw)
    r2 = torch.randn(L * L, DH, **kw)
    _dgrad_epilogue(do2, wo, g2, r2, shape_key=_pair_key())
