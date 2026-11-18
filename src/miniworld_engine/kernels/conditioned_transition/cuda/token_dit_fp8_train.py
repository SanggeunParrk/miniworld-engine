"""EXPERIMENTAL FP8 GEMM placements of the H100 bf16 token DiT TRAINING block (engine 3.1.0 line; opt-in, default OFF;
docs/fp8_training.md "Token DiT").

Enable (environment, read at import; or set :data:`PLACES` / :data:`GRAD_E5M2` at run time)::

    MINIWORLD_FP8_TRAIN=1                        # the training FP8 switch: this block takes RECOMMENDED (unless the variable below is set)
    MINIWORLD_TOKEN_DIT_FP8_TRAIN=transition     # explicit placements, a comma list of
                                                 #   forward  expand, squeeze, qkvg, out
                                                 #   dgrad    dexpand (dxt = dab Wab), dsqueeze (dh = dz Wsq), dqkvg (dxa = dqkvg Wqkvg), dout (dog = dy Wo)
                                                 # or a preset: fast (= expand,squeeze), transition, attention, all ("" / 0 / off: none)
    MINIWORLD_TOKEN_DIT_FP8_TRAIN_GRAD=e4m3      # the 8-bit format of the incoming gradients (dgrad A operands): e4m3 (default) or e5m2

Scales: every activation / gradient operand gets a DYNAMIC per-row (per-token) scale, row max / FMAX, made by the row pass that writes
it (``token_dit_fp8_train_rows.cu``); the weights a per-channel scale along the GEMM's output columns (made once per weight version,
stored in the bf16 pack). Diffusion training puts every sample at its own noise level, so a row's scale follows its own magnitude:
nothing saturates, there is no calibration and no overflow guard. Products: ``torch._scaled_mm`` (cuBLASLt, e4m3 / e5m2 x e4m3,
row-wise scales, fp32 accumulation, bf16 out). The weight gradients stay bf16 (cuBLAS, fp32 out): their contraction runs over the
tokens, an FP8 wgrad would need token-major (transposed) 8-bit copies of both operands with per-channel scales over all tokens
(docs/fp8_training.md: the extra passes cost more than the GEMM saves). With no placement the block runs the 2.5.14 bf16 route
unchanged (bit-identical).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent
E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
FWD = ("expand", "squeeze", "qkvg", "out")
DGRAD = ("dexpand", "dsqueeze", "dqkvg", "dout")
#: round 2: byte reductions that need no 8-bit GEMM (bit-identical outputs and gradients): nox = the block input is the residual, the
#: saved fp32 copy of it is dropped (docs/fp8_training.md "Token DiT round 2")
BYTES = ("nox",)
_ALL = FWD + DGRAD + BYTES
PRESETS = {
    "bytes": frozenset(BYTES),
    "transition": frozenset(("expand", "squeeze", "dexpand", "dsqueeze")),
    "attention": frozenset(("qkvg", "out", "dqkvg", "dout")),
    "all": frozenset(_ALL),
    # the placements that pay at every measured size (docs/fp8_training.md): the forward transition GEMMs
    "fast": frozenset(("expand", "squeeze")),
}
#: what ``MINIWORLD_FP8_TRAIN=1`` turns on for this block (the attention-side projections stay bf16: the inference q|k|v|g placement
#: moved AF3 FoldBench; docs/fp8_training.md)
RECOMMENDED = PRESETS["fast"]


def parse(v: str) -> frozenset[str]:
    v = v.strip().lower()
    if v in ("", "0", "off", "none"):
        return frozenset()
    out: set[str] = set()
    for p in (t.strip() for t in v.split(",") if t.strip()):
        if p in PRESETS:
            out |= PRESETS[p]
        elif p in _ALL:
            out.add(p)
        else:
            raise ValueError(f"MINIWORLD_TOKEN_DIT_FP8_TRAIN: unknown placement {p!r} (use {', '.join(_ALL)} or {', '.join(PRESETS)})")
    return frozenset(out)


def _from_env() -> frozenset[str]:
    own = os.environ.get("MINIWORLD_TOKEN_DIT_FP8_TRAIN")
    if own is not None:
        return parse(own)
    if os.environ.get("MINIWORLD_FP8_TRAIN", "").strip().lower() in ("1", "on", "true", "safe"):
        return RECOMMENDED
    return frozenset()


#: the FP8 placements in force (empty: the default bf16 route)
PLACES: frozenset[str] = _from_env()
#: incoming gradients (the dgrad A operands) as e5m2 instead of e4m3
GRAD_E5M2: bool = os.environ.get("MINIWORLD_TOKEN_DIT_FP8_TRAIN_GRAD", "e4m3").strip().lower() == "e5m2"
#: the fewest block rows M = A L (after padding) that take the placements; smaller calls keep bf16 (``MINIWORLD_TOKEN_DIT_FP8_TRAIN_MIN_ROWS``).
#: H100, CUDA graph fwd + bwd, 2026-11-18: `fast` won at M 18432 / 36864 (-2.2 / -1.1 % cool), was within the same-cubin control at
#: 3072 .. 12288
MIN_ROWS: int = int(os.environ.get("MINIWORLD_TOKEN_DIT_FP8_TRAIN_MIN_ROWS", "16384"))


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="token_dit_fp8_train_sm90",
        sources=[str(_dir / "token_dit_fp8_train_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a"), "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def active(M: int, D: int, fp32: bool, plain: bool, h100: bool) -> frozenset[str]:
    """The placements this block call takes: bf16 default-option blocks on H100 at width 768 (M = A L is a multiple of 128 after the
    block's padding)."""
    if not PLACES or fp32 or not plain or not h100 or D != 768:
        return frozenset()
    if M % 16 or M < MIN_ROWS:  # the byte reductions (no GEMM) apply at every size
        return PLACES & frozenset(BYTES)
    return PLACES


def q8(x: torch.Tensor, e5: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, C] (rows contiguous) -> (8-bit [M, C], fp32 [M, 1]) with a dynamic per-row scale."""
    M, C = x.shape
    x8 = torch.empty((M, C), device=x.device, dtype=E5M2 if e5 else E4M3)
    s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    ext().q8_rows(x, x8, s)
    return x8, s


def swiglu_q8(ab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """h = silu(a) b of the expand output [M, 2N] as e4m3 [M, N] with a per-row scale (h is not stored in bf16)."""
    M, N2 = ab.shape
    h8 = torch.empty((M, N2 // 2), device=ab.device, dtype=E4M3)
    s = torch.empty((M, 1), device=ab.device, dtype=torch.float32)
    ext().swiglu_q8(ab, h8, s)
    return h8, s


def swiglu_bwd_q8(dh, ab, dab, h, e5: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """swiglu_bwd (dab, h in bf16, written into ``dab`` / ``h``) plus dab as 8-bit with a per-row scale."""
    dab8 = torch.empty(dab.shape, device=dab.device, dtype=E5M2 if e5 else E4M3)
    s = torch.empty((dab.shape[0], 1), device=dab.device, dtype=torch.float32)
    ext().swiglu_bwd_q8(dh, ab, dab, h, dab8, s)
    return dab8, s


def _wq(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[R, C] bf16 rows -> the B operand of ``torch._scaled_mm`` for x @ w^T: (e4m3 [C, R] column-major, fp32 [1, R])."""
    w8, s = q8(w.contiguous())
    return w8.t(), s.view(1, -1)


def weights(W: dict, places: frozenset[str]) -> dict:
    """The e4m3 weight operands the placements need, made once per bf16 pack (an optimizer step makes a new pack; stored in it)."""
    f8 = W.get("_fp8_train")
    if f8 is None:
        f8 = W["_fp8_train"] = {}
    need = {
        "qkvg": ("qkvg", "Wqkvg", False),
        "out": ("out", "Wo", False),
        "expand": ("expand", "Wab", False),
        "squeeze": ("squeeze", "Wsq", False),
        "dqkvg": ("dqkvg", "Wqkvg", True),
        "dout": ("dout", "Wo", True),
        "dexpand": ("dexpand", "Wab", True),
        "dsqueeze": ("dsqueeze", "Wsq", True),
    }
    for p in places:
        k, wn, tr = need[p]
        if k not in f8:
            w = W[wn]
            # forward x @ W^T: per output channel of W; dgrad dY @ W: per input channel (the rows of W^T)
            f8[k] = _wq(w.t().contiguous() if tr else w)
    return f8


def mm(a8: torch.Tensor, sa: torch.Tensor, w: tuple[torch.Tensor, torch.Tensor], bias: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 [M, N] = (sa a8) (w8 sw) [+ bias] on the FP8 tensor cores (row-wise scales, fp32 accumulation)."""
    w8, sw = w
    return torch._scaled_mm(a8, w8, sa, sw, bias=bias, out_dtype=torch.bfloat16)


__all__ = [
    "DGRAD",
    "FWD",
    "GRAD_E5M2",
    "PLACES",
    "PRESETS",
    "RECOMMENDED",
    "active",
    "ext",
    "mm",
    "parse",
    "q8",
    "swiglu_bwd_q8",
    "swiglu_q8",
    "weights",
]
