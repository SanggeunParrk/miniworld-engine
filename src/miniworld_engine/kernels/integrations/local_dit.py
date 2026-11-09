"""The AF3 atom transformer block's shared helpers: the parameter names, the padding / trunking of the atoms and the hoist of per-item
tables. The block itself (the H100 warp-level row kernels and window attention) is ``local_dit_h100``, which ``modules/atom_dit``
calls; the hoisted tables (conditioning, windowed pair bias, slid-window edges) are made once per tensor (pointer and in-place
version) and weights, scoped to the CUDA-graph capture (``kernels._capture``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels import _capture

# ------------------------------------------------------------------------------------------------------------ weights
EPS = 1e-5

#: The block's parameters, in the order the ops take and return them (every ``DiTBlock.named_parameters`` entry at atom widths).
WEIGHTS = (
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_scale.bias",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_query.weight",
    "attention.to_query.bias",
    "attention.to_key.weight",
    "attention.to_value.weight",
    "attention.ln_pair.weight",
    "attention.to_bias.weight",
    "attention.to_gate.weight",
    "attention.to_out.weight",
    "attention.to_scale.weight",
    "attention.to_scale.bias",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.to_bias.weight",
    "transition.expand_a.weight",
    "transition.expand_b.weight",
    "transition.squeeze.weight",
    "transition.to_scale.weight",
    "transition.to_scale.bias",
)
_GAMMA, _WBIAS = (
    WEIGHTS.index("attention.ln_pair.weight"),
    WEIGHTS.index("attention.to_bias.weight"),
)


def _padded(t, n):
    """[A, 1, N, d] -> [A, 1, n, d], zero rows appended (the tensor itself when n == N)."""
    return (
        t if t.shape[2] == n else torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[2]))
    )


def _unpadded(t, shape):
    """[A, 1, n, d] (a view of the kernels' rows) -> the caller's [A, 1, N, d]."""
    return t if t.shape == shape else t[:, :, : shape[2]].contiguous()


def _padded_length(n0: int) -> int:
    return (n0 + 127) // 128 * 128

QUERIES, KEYS = 32, 128

#: The parameters of the cross-attention mode's second AdaLN (keys / values), appended to ``WEIGHTS``.
CROSS_WEIGHTS = (
    "attention.ada_ln_kv.ln_cond.weight",
    "attention.ada_ln_kv.to_scale.weight",
    "attention.ada_ln_kv.to_scale.bias",
    "attention.ada_ln_kv.to_bias.weight",
)


def _keys(mask, n0, n):
    """The valid-key bytes [n] the attention kernels read (None: every key valid and no padding)."""
    if mask is not None:
        if n == n0:
            return mask[0].contiguous()
        return F.pad(mask[0], (0, n - n0)).contiguous()
    return None if n == n0 else (torch.arange(n, device=_device_of(mask)) < n0)


def _device_of(mask):
    return torch.cuda.current_device() if mask is None else mask.device


def _hoisted(store, tensor, key, build):
    """The table ``build()`` makes from ``tensor``, once per tensor (pointer, version, ``key``): ``_capture.hoisted_table``."""
    return _capture.hoisted_table(store, tensor, key, build, limit=64)


def _tkey(t):
    """A tensor's identity for a cache key: storage address, in-place version, shape, dtype, device."""
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _trunk(pair, n):
    """[1, nwin, 32, 128, 16] -> [n / 32, 32, 128, 16]: zero windows appended for the padded atoms."""
    z = pair[0]
    return (
        z
        if z.shape[0] == n // QUERIES
        else F.pad(z, (0, 0, 0, 0, 0, 0, 0, n // QUERIES - z.shape[0])).contiguous()
    )

