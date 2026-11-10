"""Private A1 row-grid inference, all other qualified E517 providers preserved."""

import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine import settings
from miniworld_engine.kernels.swa_dit.cuda import sm80
from miniworld_engine.kernels.swa_dit.cuda import (
    inference_rowgrid_ffn_sm80_a100 as rowgrid,
)
from miniworld_engine.kernels.swa_dit.cuda import inference_rt8_ffn_sm80_a100 as rt8

from miniworld_engine.kernels.swa_dit.cuda import inference_sm80_a100 as baseline

_ROWGRID = {(1, 1024): 3, (1, 2048): 3}
_RT8 = {(1, 2048): 3, (5, 1024): 3}


def _block_fake(x, c, cos, sin, seqused, wmod, wq, wg, wo, wu, wd):
    """Same shape and dtype as ``x`` ([A, S, 128] block output), freshly allocated."""
    return torch.empty_like(x)


@opaque(fake=_block_fake, name="swa_dit_inference_rowgrid_block_sm80")
def block(
    x: torch.Tensor,
    c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqused: torch.Tensor,
    wmod: torch.Tensor,
    wq: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
) -> torch.Tensor:
    """A100 SAVE=false inference block with the row-grid / rt8 FFN geometries; other shapes fall through to the baseline block."""
    a, s, _ = x.shape
    if (a, s) in _RT8:
        return rt8.forward(x, c, cos, sin, seqused, wmod, wq, wg, wo, wu, wd)
    # Keep the already-qualified actual E517 body/providers intact elsewhere.
    if (a, s) not in _ROWGRID:
        return baseline.block(x, c, cos, sin, seqused, wmod, wq, wg, wo, wu, wd)
    b = c.shape[0]
    eps = float(torch.finfo(torch.float32).eps)
    mod = sm80.mod_fwd(c.reshape(-1, 128).contiguous(), wmod, save=False)[0]
    qf = x.reshape(-1, 128)
    cos2, sin2 = (
        cos[:b].reshape(-1, 16).contiguous(),
        sin[:b].reshape(-1, 16).contiguous(),
    )
    qkv_kind, qkv_cfg, ffn_kind = baseline._GEOMETRY_KERNELS[(a, s)]
    assert qkv_kind == "singlehead" and qkv_cfg == 0 and ffn_kind == "tuned"
    qh, kh, vh, g = baseline.qkvg_extension().singlehead(
        qf, mod, cos2, sin2, wq, wg, s, b, eps, eps, False, qkv_cfg
    )
    o, _ = sm80.attn_fwd(
        qh, kh, vh, seqused, save_lse=not settings.current().a100_cleanup
    )
    out = rowgrid.ffn(qf, g, o, mod, wo, wu, wd, s, b, eps, _ROWGRID[(a, s)])
    return out.view(a, s, 128)


def eligible(x, c, ap, weights, n_head, half_window):
    # All physical dtype/layout/device/hooks/native-disable rules are original.
    return baseline.eligible(x, c, ap, weights, n_head, half_window)
