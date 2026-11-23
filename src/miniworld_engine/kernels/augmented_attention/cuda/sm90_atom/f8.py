"""EXPERIMENTAL, opt-in (never reached by default): the bf16 inference forward of the H100 atom block with its row-kernel GEMMs on e4m3 wgmma
(``pre90_f8.cu``, ``post90_f8.cu``, ``f8w90.cuh``). The window attention, the conditioning tables and the pair bias stay bf16.

Scales are powers of two and exact: the weights per 64-row unit / matrix (here, once per weight version), the activations per row inside the kernels from
the row's exact amax (dynamic: no calibration, no overflow guard -- an e4m3 cast cannot saturate). Served by
``kernels/integrations/local_dit_h100._fwd`` when ``local_dit_h100.FP8_ROWS`` is set (``MINIWORLD_ATOMDIT_FP8=1`` / ``pre`` / ``post`` at import, or ``enable_fp8``).
"""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._device import num_sms

from . import DS, NWARP, ROWS, _dev, cubin, pack

E4M3 = torch.float8_e4m3fn
_KERNELS = {
    "pre_f8": ("pre90_f8", 2 * 64 * 128 + 1024),
    "post_f8": ("post90_f8", 2 * (128 * 128 + 128 * 128) + 1024),
    "post_f8up": ("post90_f8up", 2 * (128 * 128 + 64 * 128 + 128 * 128) + 1024),  # + a Wu_j chunk (64 rows) per buffer
}
#: model options this route serves (``local_dit_h100.options`` plus RMS of ``local_dit_esm``: ESMFold2's RMSNorm AdaLNs and raw gates, built with
#: -DRMS=1 into both kernels; RF3's KQLN, the flat q / k LayerNorm, built into pre90_f8 with -DKQLN=1). UP (Boltz-2's up-gate transition) is
#: served by ``post90_f8up.cu`` (``post_f8up``): the post kernel with a Wu_j chunk beside each [Wa_j; Wb_j]; ``pre90_f8`` does not depend on it
OPTIONS = frozenset({"IDENT", "PARS", "NOOUT", "RMS", "KQLN", "UP"})
#: CTAs per SM each kernel's register allocation targets (__launch_bounds__ minimum blocks), by grid size: 1 gives 236 / 253 registers
#: (2 CTAs per SM), 3 caps them at 168 with a small spill. Measured (H100, AF3 block, A = 5, kernel times): pre90_f8 at 320 / 640 CTAs
#: 26.1 -> 19.5 / 43.0 -> 40.1 us with 3, at 1280 79.2 -> 85.8 us (so 1); post90_f8 3 only where it makes one wave of a grid that
#: takes two at 2 per SM (320 CTAs: 42.7 -> 36.9 us; 1280: 120.6 -> 131.5). A fixed value here (1 or 3) overrides the rule.
MINB: dict = {"pre_f8": None, "post_f8": None}


def _minb(kind: str, ctas: int) -> int:
    fixed = MINB[kind]
    if fixed is not None:
        return int(fixed)
    if kind == "pre_f8":
        return 3 if ctas <= 800 else 1
    sms = num_sms(torch.cuda.current_device())
    return 3 if 2 * sms < ctas <= 3 * sms else 1


def kernel_cubin(kind: str, options: tuple = (), minb: int = 1) -> str:
    """The cubin of ``kind`` under the model ``options`` and ``minb`` (the build :func:`kernel` loads; also the prebuild recipe's call)."""
    if (
        kind == "post_f8" and "UP" in options
    ):  # the up-gate post kernel (own source; one register target: its tile of accumulators needs all 255)
        kind, minb = "post_f8up", 1
    defs = tuple(
        f"-D{o}=1"
        for o in sorted(options)
        if o == "RMS" or (kind == "pre_f8" and o == "KQLN") or (kind in ("post_f8", "post_f8up") and o in ("PARS", "NOOUT"))
    )
    return cubin(_KERNELS[kind][0], False, NWARP, minb, defs)


@functools.lru_cache(maxsize=None)
def kernel(kind: str, device_index: int, options: tuple = (), minb: int = 1):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, smem = _KERNELS["post_f8up" if kind == "post_f8" and "UP" in options else kind]
    with torch.cuda.device(device_index):
        return driver.Kernel(kernel_cubin(kind, options, minb), stem, smem, pdl=True)


def _kperm32(t: torch.Tensor) -> torch.Tensor:
    """The K axis (last, a multiple of 32) in the e4m3 A-fragment slot order: slot 16 h + 4 t + 2 a + b of a k32 step holds logical column
    16 h + 8 a + 2 t + b (``f8w90.cuh``). View arithmetic only."""
    k = t.shape[-1]
    x = t.reshape(*t.shape[:-1], k // 32, 2, 2, 4, 2)
    return x.transpose(-3, -2).reshape(t.shape)


def _pow2(amax: torch.Tensor) -> torch.Tensor:
    """The largest power of two s with amax * s <= 448 (1 where amax is 0)."""
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    return torch.exp2(torch.floor(torch.log2(448.0 / safe)))


def _q(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _kperm32((w.float() * scale).to(E4M3)).contiguous()


def pack8(weights) -> dict:
    """e4m3 forms of the bf16 pack (``sm90_atom.pack``), K in slot order, each scaled by a power of two: pre90_f8 per 64-row unit (q lo / hi,
    gate, k, v: 8 scales), post90_f8 per Wo unit (2), per [Wa_j | Wb_j] half (8) and one for Wd; the inverse scales fp32 (``wsi_pre`` [8],
    ``wsi_post`` [11]). Cached per weight version / capture scope."""
    p = pack(weights, False)
    up = len(weights) > 27  # Boltz-2: the 28th leaf is W_u; p["wu"] [4][64][128]: chunk j of it beside [Wa_j; Wb_j]

    def build():
        pre, wo, wab, wd = p["pre"].float(), p["wo"].float(), p["wab"].float(), p["wd"].float()
        s_pre = _pow2(pre.view(8, 64 * 128).abs().amax(-1))  # [8]
        s_wo = _pow2(wo.view(2, 64 * 128).abs().amax(-1))  # [2]
        s_ab = _pow2(wab.view(4, 2, 64 * 128).abs().amax(-1))  # [4, 2]: chunk j, Wa_j / Wb_j
        s_d = _pow2(wd.abs().amax().reshape(1))  # [1]
        s_u = _pow2(p["wu"].float().view(4, 64 * 128).abs().amax(-1)) if up else None  # [4]: one scale per Wu_j chunk
        return {
            "pre8": _q(pre, s_pre.repeat_interleave(64)[:, None]),
            "wsi_pre": (1.0 / s_pre).contiguous(),
            "wo8": _q(wo, s_wo.repeat_interleave(64)[:, None]),
            "wab8": _q(wab, s_ab.repeat_interleave(64, dim=1)[:, :, None]),
            "wd8": _q(wd, s_d),
            "wsi_post": (1.0 / torch.cat([s_wo, s_ab.reshape(-1), s_d] + ([s_u] if up else []))).contiguous(),
            **({"wu8": _q(p["wu"].float(), s_u[:, None, None])} if up else {}),
            "bq": p["bq"],
            "bs": p["bs"],
        }

    # inference only (the route declines a saving forward): a graph captured with grad off reads the entry its eager warm-up made (as the B200 atom DiT FP8 pack)
    return _capture.param_forms("sm90_atom_pack_f8", list(weights), build, capture_reads_eager=not torch.is_grad_enabled())


def _split_for(tiles: int, choices: tuple = (1, 2, 4, 8)) -> int:
    from . import _split_for as split

    return split(tiles, choices)


def pre_fwd8(x, mod, mkv, p8, eps: float = 1e-5, nmod: int = 0, parallel_kv: int = 0, options: tuple = (), kq=None, kq_eps: float = 1e-5):
    """bf16 x [M, 128] -> q, sg, k, v [M, 128] bf16 (``sm90_atom.pre_fwd`` without save). KQLN: ``kq`` = [wq | bq | wk | bk] fp32 [4, 128]."""
    M = x.shape[0]
    new = lambda: torch.empty(M, DS, device=x.device, dtype=x.dtype)  # noqa: E731
    q, sg, k, v = new(), new(), new(), new()
    tiles = M // ROWS
    kqln = "KQLN" in options
    ns = _split_for(tiles, (1, 2, 4) if kqln else (1, 2, 4, 8))  # KQLN: a CTA holds both 64-column units of q (and of k)
    kernel("pre_f8", _dev(x), tuple(options), _minb("pre_f8", tiles * ns))(
        (tiles, ns, 1),
        (NWARP * 32, 1, 1),
        x,
        mod,
        mkv,
        p8["pre8"],
        p8["wsi_pre"],
        p8["bq"],
        p8["bs"],
        q,
        sg,
        k,
        v,
        int(M),
        int(nmod or M),
        float(eps),
        int(parallel_kv),
        kq,
        float(kq_eps),
    )
    return q, sg, k, v


def post_fwd8(a, o, sg, mod, p8, eps: float = 1e-5, nmod: int = 0, options: tuple = ()):
    """-> out [M, 128] bf16 (``sm90_atom.post_fwd`` without save). UP (Boltz-2): ``post90_f8up`` with the chunks of W_u (``p8["wu8"]``)."""
    M = a.shape[0]
    out = torch.empty(M, DS, device=a.device, dtype=a.dtype)
    kernel("post_f8", _dev(a), tuple(options), _minb("post_f8", M // ROWS))(
        (M // ROWS, 1, 1),
        (NWARP * 32, 1, 1),
        a,
        o,
        sg,
        mod,
        p8["wo8"],
        p8["wab8"],
        p8["wd8"],
        *((p8["wu8"],) if "UP" in options else ()),  # post90_f8up: wu8 between wd8 and the scales
        p8["wsi_post"],
        out,
        int(M),
        int(nmod or M),
        float(eps),
    )
    return out
