"""OPT-IN e4m3 placements of the RTX 5090 (sm_120a) token DiT inference block (docs/fp8.md "RTX 5090", token DiT): the ``token_dit`` slot of
``fp8.SM120_ROUTES``, the same interface and switch as the H100 module ``token_dit_fp8`` (``PLACES``, ``STATS``, ``MIN_ROWS``, ``_parse``)::

    MINIWORLD_TOKEN_DIT_FP8=expand          # (MINIWORLD_FP8=safe / engine-fp8 set this) any of qkvg, expand, out, squeeze; "1" = qkvg,expand; "all"
    MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS=0      # one row gate for every placement (default: MIN_ROWS_PER; fewer step rows A B L keep bf16)

The 5090 token DiT runs the sm_80 kernels built for sm_120a: the attention on ``integrations/augattn_sm80`` (AdaLN, the qkvg GEMM, the pair bias,
the mma.sync core, the out projection) and the transition on ``integrations/conditioned_transition_sm80`` (AdaLN rows, the expand GEMM, SwiGLU,
the squeeze GEMM, the gate). Their bf16 GEMMs are cuBLAS (137-185 TFLOPS at the token widths); the placements put them on cuBLASLt e4m3 with
row-wise scales (``torch._scaled_mm``: fp32 accumulation, bf16 out; 320-375 TFLOPS):

* ``expand``  (in ``safe``): the transition's AdaLN epilogue writes xa as e4m3 with a per-row scale (``adaln_q8``, the bf16 xa is never
  written), ``[a | b (| u)] = xa8 W8^T``, then the default SwiGLU pass (Boltz-2's up gate: silu(a) b u), the bf16 squeeze. Every token DiT class:
  TokenDiTProtenix / MiniWorld (default block), Boltz-2 (up-gated transition), RoseTTAFold3 (parallel block), Chai-1 (identity AdaLN).
* ``squeeze``: h = silu(a) b (u) quantised in the SwiGLU pass (``swiglu_q8``), the squeeze on e4m3.
* ``qkvg``: the attention's AdaLN output quantised (``quant_rows``), q | k | v | g on e4m3 with the bias in the GEMM; gated blocks without a
  q / k LayerNorm (not Chai-1, not RoseTTAFold3; MiniWorld's per-head RMSNorm follows the GEMM as before).
* ``out``: the gated core output og quantised, the out projection on e4m3 (not RoseTTAFold3).

Scales are the H100 recipe: activations a DYNAMIC per-row scale (row max / 448), weights a static per-output-channel scale (max / 448, once per
weight version): nothing saturates, no calibration, no overflow guard (``fp8.check()`` has nothing to read here). The pair bias (and its hoist,
``augattn_sm80`` TOKEN_BIAS), the attention core and every row pass stay as they are. bf16 inference only (fp32, training: the default route).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.fp8 import env as _fp8_env

from .token_dit_fp8 import _ALL, _parse

_dir = Path(__file__).parent / "sm80"
E4M3 = torch.float8_e4m3fn
BF = torch.bfloat16

#: the e4m3 placements in force (empty: the default bf16 route); the H100 module's variable
PLACES: frozenset[str] = _parse(_fp8_env("MINIWORLD_TOKEN_DIT_FP8", "") or "")
#: route call counters (``fp8.route_stats()["token_dit"]``, plain ints at Python call time): ``calls`` = token transition calls that took an
#: e4m3 placement (expand / squeeze), ``declined`` = token transition calls with a transition placement in ``PLACES`` sent to the bf16 route
#: (fp32, width, row gate); per placement the calls that took it (``qkvg`` / ``out``: attention calls).
STATS = {"calls": 0, "declined": 0, "expand": 0, "qkvg": 0, "out": 0, "squeeze": 0}
#: per placement, the fewest step rows M = A B L that take it (RTX 5090, same-process A-B, 6 rounds, docs/fp8.md "RTX 5090" token DiT): expand
#: 1.04-1.27x and qkvg 1.11-1.18x already at M 640 / 768 (the smallest measured); squeeze 0.98-0.99 and out 0.98-0.99 at 640, 1.02-1.05 from 1280
MIN_ROWS_PER: dict[str, int] = {"qkvg": 640, "expand": 640, "out": 1280, "squeeze": 1280}
#: Boltz-2's up-gated expand (its own GEMM shape [a | b | u])
MIN_ROWS_UP = 640
#: one override for every placement (``MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS``; 0 = no gate), None: ``MIN_ROWS_PER``
MIN_ROWS: int | None = int(os.environ["MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS"]) if os.environ.get("MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS") else None
#: the widths built (the token DiT's d = 768; its hidden 1536)
WIDTHS = (768,)


@functools.lru_cache(maxsize=1)
def ext():
    """The sm_120a extension ``token_f8_sm120`` (the e4m3 row passes; the GEMMs are torch._scaled_mm)."""
    from ..._nvcc import ensure_cuda_home, host_flags, load_extension
    from ..._sm120_builds import SM120

    ensure_cuda_home()
    return load_extension(
        name="token_f8_sm120",
        sources=[str(_dir / "token_f8_sm120.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", SM120, "-DMWE_SM120"],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def _sm120(t: torch.Tensor) -> bool:
    from ..._sm120_builds import is_sm120

    return t.is_cuda and is_sm120(t.device)


def _gate(p: str, M: int, up: bool = False) -> int:
    if MIN_ROWS is not None:
        return MIN_ROWS
    return MIN_ROWS_UP if up and p == "expand" else MIN_ROWS_PER[p]


def transition_places(x: torch.Tensor, identity: bool, parallel: bool, up: bool) -> frozenset[str]:
    """The transition placements (``expand`` / ``squeeze``) of a token block's transition call: bf16 x [M, 768] on an RTX 5090, each from its
    row gate. RoseTTAFold3's parallel block (``parallel`` and not ``identity``): ``expand`` only (the H100 route's rule)."""
    if not PLACES or x.dtype is not BF or x.shape[-1] not in WIDTHS or not _sm120(x):
        return frozenset()
    places = PLACES & {"expand", "squeeze"}
    if parallel and not identity:
        places &= {"expand"}
    M = x.numel() // x.shape[-1]
    return frozenset(p for p in places if M >= _gate(p, M, up))


def attn_places(m: int, d: int, dt: torch.dtype, gated: bool, flat: int, device: torch.device) -> frozenset[str]:
    """The attention placements (``qkvg`` / ``out``) of a token block's bf16 inference attention call (``augattn_sm80._fwd``): width 768, an
    RTX 5090, each from its row gate. ``qkvg``: gated, no flat q / k LayerNorm (``flat`` = the number of its leaves: 4 = RoseTTAFold3's LN,
    2 = MiniWorld's per-head RMSNorm, served); ``out``: no flat LayerNorm."""
    if not PLACES or dt is not BF or d not in WIDTHS or device.type != "cuda":
        return frozenset()
    from ..._sm120_builds import is_sm120

    if not is_sm120(device):
        return frozenset()
    places = PLACES & {"qkvg", "out"}
    if not gated or flat == 4:
        places -= {"qkvg"}
    if flat == 4:
        places -= {"out"}
    return frozenset(p for p in places if m >= _gate(p, m))


# ------------------------------------------------------------------------------------------------------------------ weights
def quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (e4m3 [N, K] seen as the column-major [K, N] operand of torch._scaled_mm, fp32 [1, N]): per output channel, scale = row max / 448."""
    wf = w.float()
    s = wf.abs().amax(1).clamp_min(1e-30) / 448.0
    return (wf / s[:, None]).to(E4M3).contiguous().t(), s.view(1, -1).contiguous()


def _forms(tag: str, tensors: tuple, build):
    from miniworld_engine.kernels.integrations.adaln_sm80 import cached

    return cached(tag, tensors, build)


def weight8(tag: str, *ws: torch.Tensor):
    """The e4m3 form of ``cat(ws)`` [N, K] (see :func:`quant_weight`), kept per weight version (``adaln_sm80.cached``: weakrefs + versions, capture scoped)."""
    return _forms(f"tdf8_{tag}", tuple(ws), lambda: quant_weight(torch.cat(ws) if len(ws) > 1 else ws[0]))


# ------------------------------------------------------------------------------------------------------------------ products
def rows_q8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, D] (unit column stride) -> (e4m3 [M, D], fp32 [M, 1]) with a dynamic per-row scale (one pass)."""
    M, D = x.shape
    x8 = torch.empty((M, D), device=x.device, dtype=E4M3)
    s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    ext().quant_rows(x, x8, s)
    return x8, s


def mm(x8: torch.Tensor, sx: torch.Tensor, w, bias: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 = (sx x8) (w8 sw)^T [+ bias] on cuBLASLt e4m3 (row-wise scales, fp32 accumulation)."""
    w8, sw = w
    return torch._scaled_mm(x8, w8, sx, sw, bias=bias, out_dtype=BF)


def qkvg(xs: torch.Tensor, wqkvg: torch.Tensor, bvec: torch.Tensor) -> torch.Tensor:
    """The attention's ``addmm(bvec, xs, wqkvg^T)`` on e4m3 (``qkvg`` placement)."""
    STATS["qkvg"] += 1
    b = bvec if bvec.dtype is BF else _forms("tdf8_bq", (bvec,), lambda: bvec.to(BF))
    return mm(*rows_q8(xs), weight8("qkvg", wqkvg), bias=b)


def out(og: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """The attention's ``og wo^T`` on e4m3 (``out`` placement)."""
    STATS["out"] += 1
    return mm(*rows_q8(og), weight8("out", wo))


# ------------------------------------------------------------------------------------------------------------------ the transition
def transition(module, xs: torch.Tensor, cs: torch.Tensor, eps: tuple, residual: bool, places: frozenset) -> torch.Tensor:
    """``conditioned_transition_sm80.inference`` / ``inference_up`` with the ``expand`` (and ``squeeze``) placements: xs [M, 768] bf16, cs [P, dc]
    the conditioning rows (P | M), ``x + update`` (``residual``) or the update."""
    from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct
    from miniworld_engine.kernels.integrations.adaln_sm80 import Branch, _mm_op, cast_params, is_identity

    ka, kt = ct._adaln(), ct._tail()
    p = ct._params(module)
    lnw, ws1, sb1, wb1, wa, wb, ws, wsc, bsc = p
    up = ct._up(module)
    identity = is_identity(module.ada_ln_in)
    STATS["calls"] += 1
    for q in places:
        STATS[q] += 1
    with torch.cuda.device(xs.device):
        sb1, ws, wsc, bsc = cast_params((sb1, ws, wsc, bsc), BF, True)
        wcat = _forms("tdf8_wcat", (ws1, wb1), lambda: torch.cat([ws1, wb1]).to(BF))
        br = Branch(xs.device)
        g = br.run(lambda: ct._gate_logits(cs, wsc, bsc), cs, wsc, bsc)  # the gate GEMM beside the chain below
        if identity:  # Chai-1: (1 + S) LN(x) + B on the raw conditioning
            sbt = _mm_op(cs, wcat.t())
            sb = sbt.new_empty(0)
        else:
            aff, _ = ka.cond_ln(cs, lnw, BF, eps[1])
            sbt, sb = _mm_op(aff, wcat.t()), sb1
        M = xs.shape[0]
        if "expand" in places:
            x8 = torch.empty((M, xs.shape[1]), device=xs.device, dtype=E4M3)
            sx = torch.empty((M, 1), device=xs.device, dtype=torch.float32)
            ext().adaln_q8(xs, sbt, sb, x8, sx, float(eps[0]))
            ab = mm(x8, sx, weight8("expand_up" if up is not None else "expand", wa, wb, *((up.weight,) if up is not None else ())))
        else:  # squeeze alone: the default bf16 AdaLN epilogue and expand GEMM
            xa = ka.epilogue_identity(xs, sbt, eps[0])[0] if identity else ka.epilogue(xs, sbt, sb, eps[0])[0]
            ab = _mm_op(xa, _forms("tdf8_wab", (wa, wb, *((up.weight,) if up is not None else ())), lambda: torch.cat([wa, wb, *((up.weight,) if up is not None else ())]).to(BF)).t())
        if "squeeze" in places:
            H = ws.shape[1]
            h8 = torch.empty((M, H), device=xs.device, dtype=E4M3)
            sh = torch.empty((M, 1), device=xs.device, dtype=torch.float32)
            ext().swiglu_q8(ab, h8, sh, up is not None)
            z = mm(h8, sh, weight8("squeeze", ws))
        else:
            h = kt.swiglu_up_fwd(ab) if up is not None else kt.swiglu_fwd(ab)
            z = _mm_op(h, ws.t())
        br.join()
        return kt.gate_res_fwd(xs if residual else None, z, g)


def prebuild_items():
    """The build of this route (``_prebuild --arch sm120``): the row-pass extension."""
    yield ext


__all__ = ["E4M3", "MIN_ROWS", "MIN_ROWS_PER", "MIN_ROWS_UP", "PLACES", "STATS", "_ALL", "_parse", "attn_places", "ext", "transition", "transition_places"]
