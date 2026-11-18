"""EXPERIMENTAL e4m3 placements of the H100 bf16 token DiT inference block (engine 3.0.0 line; opt-in, default OFF; docs/fp8.md "Token DiT").

Enable (environment, read at import; or set :data:`PLACES` / :data:`MIN_ROWS` at run time)::

    MINIWORLD_TOKEN_DIT_FP8=qkvg,expand      # any of qkvg, expand, out, squeeze ("1" = qkvg,expand; "all" = the four)
    MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS=0       # one row gate for every placement (default: MIN_ROWS_PER; fewer step rows A B L keep bf16)

Scales: activations get a DYNAMIC per-row scale (row max / 448, computed where the row is made: the AdaLN row passes for the qkvg / expand
inputs, a row pass for og / h), weights a static per-output-channel scale made once per weight pack. Nothing saturates by construction,
whatever the noise level of a diffusion step: no calibration, no overflow guard (the guard contract of the TriMul / TriAttn FP8 routes is
not needed here). Products: ``torch._scaled_mm`` with row-wise scales (bf16 out, fp32 accumulation; qkvg / out / squeeze) and
``gemm_swiglu_f8_sm90.cu`` (expand + SwiGLU, e4m3 wgmma). Default (PLACES empty): the 2.5.12 bf16 route runs unchanged.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.fp8 import env as _fp8_env
from miniworld_engine.kernels import _capture

_dir = Path(__file__).parent
E4M3 = torch.float8_e4m3fn
_ALL = ("qkvg", "expand", "out", "squeeze")


def _parse(v: str) -> frozenset[str]:
    v = v.strip().lower()
    if v in ("", "0", "off", "none"):
        return frozenset()
    if v in ("1", "on"):
        return frozenset(("qkvg", "expand"))
    if v == "all":
        return frozenset(_ALL)
    out = frozenset(p.strip() for p in v.split(",") if p.strip())
    bad = out - set(_ALL)
    if bad:
        raise ValueError(f"MINIWORLD_TOKEN_DIT_FP8: unknown placements {sorted(bad)} (use {', '.join(_ALL)})")
    return out


#: the e4m3 placements in force (empty: the default bf16 route)
PLACES: frozenset[str] = _parse(_fp8_env("MINIWORLD_TOKEN_DIT_FP8", "") or "")
#: Route call counters (``miniworld_engine.fp8.route_stats()``; plain ints at Python call time, never inside a kernel or graph replay):
#: ``calls`` = block calls that took at least one placement, one counter per placement (``expand`` / ``qkvg`` / ``out`` / ``squeeze``),
#: ``declined`` = block calls the route is on for (``PLACES`` set) that :func:`active` sent to the bf16 route (width / rows / fp32 / options).
STATS = {"calls": 0, "declined": 0, "expand": 0, "qkvg": 0, "out": 0, "squeeze": 0}
#: per placement, the fewest step rows M = A B L that take it (H100, CUDA graph, 2026-11-17, docs/fp8.md): the e4m3 expand + SwiGLU won at
#: every M measured (768 .. 10240); the row-wise cuBLAS e4m3 qkvg GEMM lost at M = 768 / 1536 / 1920 (+1 .. +2 %), won from 2560 on;
#: out / squeeze (a separate quantizing row pass) lost or tied everywhere -- their gate only bounds the loss
MIN_ROWS_PER: dict[str, int] = {"qkvg": 2560, "expand": 640, "out": 2560, "squeeze": 2560}
#: the expand of Boltz-2's up-gate block (n192 tiles, three e4m3 operands per tile): the kernel is 26-28 % faster than the bf16 up-gate kernel at every M
#: (11.4 -> 8.4 us at M = 1920, 26.6 -> 19.1 us at 5120), but the block in a CUDA graph (H100, 2026-11-18, A = 5) is slower below M = 2560 (1920: +2.7 %,
#: 1280: +3.2 %, 640: +5.6 %), level at 2560 (0.998), faster above (3840 0.991, 5120 0.965, 7680 0.954): gate 2560
MIN_ROWS_UP = 2560
#: one override for every placement (``MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS``; 0 = no gate), None: ``MIN_ROWS_PER``
MIN_ROWS: int | None = int(os.environ["MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS"]) if os.environ.get("MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS") else None

_F8_PACKS: dict = _capture.weight_cache("conditioned_transition.token_fp8_packs")


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    k = _dir.parent.parent
    return load_extension(
        name="token_dit_fp8_sm90",
        sources=[str(_dir / "gemm_swiglu_f8_sm90.cu"), str(_dir / "token_dit_fp8_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "--expt-relaxed-constexpr",
            f"-I{k / 'transition' / 'cuda' / 'anthropic_v5'}",
            f"-I{k / 'transition' / 'cuda' / 'wide' / 'kernels'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def active(
    M: int, D: int, fp32: bool, plain: bool, identity: bool = False, parallel: bool = False, up: bool = False
) -> frozenset[str]:
    """The placements this block call takes: bf16 default-option blocks (the CondFold route) at width 768, M % 128 == 0, each placement
    from its row gate (``MIN_ROWS`` or ``MIN_ROWS_PER``). ``identity``: Chai-1's option block (parallel attention / transition, identity
    AdaLN, no gating query; the CondFold route) takes ``expand`` (the transition's AdaLN comes out of the attention's LN pass as e4m3),
    ``out`` and ``squeeze``; not ``qkvg`` (its AdaLN scale is unbounded: no bound on the e4m3 error was measured, docs/fp8.md).
    ``parallel`` (any other CondFold block of the parallel schedule, RoseTTAFold3 with its k-q LayerNorm: ``plain`` False): the ``expand``
    placement only -- the transition reads the block input's AdaLN rows (quantised in one row pass), which does not depend on the attention part.
    ``up`` (Boltz-2's up-gate transition, ``plain`` then says the block is the default or Boltz-2's): the ``expand`` placement reads its
    own row gate ``MIN_ROWS_UP`` (its e4m3 kernel has the 192-wide tile); qkvg / out / squeeze keep the default block's gates."""
    if not PLACES or fp32 or D != 768 or M % 128 or not (plain or identity or parallel):
        return frozenset()
    places = PLACES if plain else (PLACES - {"qkvg"} if identity else PLACES & {"expand"})
    return frozenset(
        p for p in places if M >= (MIN_ROWS if MIN_ROWS is not None else MIN_ROWS_UP if up and p == "expand" else MIN_ROWS_PER[p])
    )


def quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (e4m3 [N, K], fp32 [N]): per output channel, scale = row max / 448."""
    wf = w.float()
    s = wf.abs().amax(1).clamp_min(1e-30) / 448.0
    return (wf / s[:, None]).to(E4M3).contiguous(), s.contiguous()


def weight_pack(key, wqkvg, wo, ws, wab_ht, bqkvg):
    """The e4m3 forms of the block's packed weights, once per weight version (``key``: the bf16 pack's key; scoped as every pack).
    ``wab_ht`` = (the expand weight packed for the SwiGLU kernel's tiles, ht)."""
    wab, ht = wab_ht

    def build():
        q8, qs = quant_weight(wqkvg)
        o8, os_ = quant_weight(wo)
        s8, ss = quant_weight(ws)
        a8, as_ = quant_weight(wab)
        # torch._scaled_mm takes B column-major: the [N, K] row-major weight transposed is exactly that
        return {
            "qkvg": (q8.t(), qs.view(1, -1)),
            "bqkvg": bqkvg.to(torch.bfloat16).contiguous(),
            "out": (o8.t(), os_.view(1, -1)),
            "squeeze": (s8.t(), ss.view(1, -1)),
            "expand": (a8, as_, ht),
        }

    return _capture.lookup(_F8_PACKS, ("f8", ht, *key), build, limit=_capture.BLOCK_CACHE_LIMIT)


def rows_q8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, D] -> (e4m3 [M, D], fp32 [M, 1]) with a dynamic per-row scale (one row pass)."""
    M, D = x.shape
    x8 = torch.empty((M, D), device=x.device, dtype=E4M3)
    s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    ext().quant_rows(x, x8, s, False)
    return x8, s


def mm(x8, sx, w, bias=None):
    """bf16 = (sx x8) (w8 sw)^T [+ bias] on the e4m3 tensor cores (row-wise scales, fp32 accumulation)."""
    w8, sw = w
    return torch._scaled_mm(x8, w8, sx, sw, bias=bias, out_dtype=torch.bfloat16)


__all__ = ["E4M3", "MIN_ROWS", "MIN_ROWS_PER", "MIN_ROWS_UP", "PLACES", "active", "ext", "mm", "quant_weight", "rows_q8", "weight_pack"]
