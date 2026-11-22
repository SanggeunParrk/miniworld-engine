"""B200 (sm_100) LayerNorm / RMSNorm: the sm_100a row kernels (``cuda/sm100``) and the engine's plain-CUDA norm kernels, whose
extensions carry an sm_100 build (``general`` and ``pipe`` compile ``-gencode`` 80 / 90 / 100: no other card's build changes).

Per call one of (measured on the B200, kernel durations in a CUDA graph; see ``docs/b200/transition.md``):

* ``sm100`` / ``skew`` (``cuda/sm100``, sm_100a only): the sub-warp row kernels for the widths of ``SM100_ROWS``, their skewed form for the
  bf16 odd widths of ``SM100_SKEW`` (267 / 651 pair streams);
* ``pipe`` (``cuda/pipe``): rows of 144 .. 2048 bytes, a multiple of 16 -- the warp-per-row-group kernels with a cp.async ring;
* ``rows`` (``cuda/general/ln_rows.cu``): rows that are not a 16-byte multiple (267 / 451 / 651 / 831 / 833), 32 .. 1024 elements;
* ``general`` (``cuda/general``): every other width (64 bf16, 2560 ...).

:func:`refusal` narrows the route to the cases where it beats the Triton LayerNorm on this card (``SLOWER``): a call it refuses is
``unsupported`` for the ``"cuda"`` backend (never another backend's kernels).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.layernorm.cuda import general, pipe, sm100

#: (width, dtype, training) cells where the CUDA kernels lose to Triton on the B200: {(width, dtype, train): (min rows, max rows)} refused.
#: Measured 2026-11-16 (measure.speed, CUDA graph, L2 flushed; docs/b200/transition.md): the bf16 pair-stream inference forward from
#: 64-147k rows (pipe / general 6-25 % behind: the H100-tuned cp.async ring keeps too few bytes in flight here), the D 64 training
#: step (general kernels: bf16 below 410k rows, fp32 from 147k rows).
#: 2026-11-17 (part 2): bf16 256 served from 65 536 to 147 455 rows ([65536, 256] 20.2 us against Triton 22.0), bf16 128 up to 262 143 rows
#: ([147456, 128] 20.4 against 20.4).
#: 2026-11-17 (round 6): fp32 D 64 training served again (row kernels with the round-6 backward launch: [147456 / 589824, 64] 44.9 / 128.7 us
#: against Triton 45.1 / 130.9).
_BF16, _F32 = torch.bfloat16, torch.float32
NEVER = 1 << 62
SLOWER: dict[tuple[int, torch.dtype, bool], tuple[int, int]] = {
    # 2026-11-23: bf16 64 / 128 inference are served at every size (lean forward with early affine loads: [65536 .. 4194304, 64] 10.2 .. 166 us against Triton 10.2 .. 168, [327680 .. 5242880, 128] 34.7 .. 405 against 34.7 .. 416)
    (256, _BF16, False): (147_456, 1 << 62),
    (384, _BF16, False): (524_288, 1 << 62),  # 2026-11-23: [147456 / 262144, 384] level with Triton (45.0 / 71.6 us), [1048576, 384] 253.8 against 247.7
}


#: bf16 widths that run the sub-warp row kernels (sm100/ln_rows_sm100.cu, the B200 port branch's ln_rows.cu) instead of pipe / general:
#: measured on the B200 (CUDA graph; docs/b200/transition.md) eval [L^2, D] at L 256 / 768 / 2048: D 64 10.2 / 32.8 / 178 us (general 12.3 /
#: 38.9 / 219), D 128 14.4 / 57.4 / 350 (pipe 14.3 / 69.7 / 434), D 256 20.4 / 106.5 / 692 (pipe 22.5 / 133 / 879), D 384 28.7 / 190 / 1302 (pipe
#: 30.7 / 201 / 1360); training (forward + backward) D 64 L 256 / 768 22.7 / 96.3 (general 32.8 / 104.4), D 128 30.7 / 149.5 (pipe 30.7 / 163.9),
#: D 256 47.1 / 274 (pipe 49.2 / 320); D 384 training stays on pipe (59.4 / 369 against 69.6 / 461), D 512 is a tie.
#: {(width, dtype): ((training modes), min rows)}: below 65 536 rows the row kernels measured a timer step (2 us) behind pipe at D 128 / 384
#: (atom streams of 1-6k rows, [1, 128, 384] single), so they take the pair streams only there; D 64 at every size (general is slower).
#: 2026-11-17 (round 3, graph timing): fp32 768 at every size (the token streams the general kernels took with two column-pass launches:
#: training [1, L, 768] 16.3-24.4 -> 12.2-16.3 us, inference [5, L, 768] from L 640 12.2-20.4 -> 10.1-18.3; the forward launch by row
#: count, ``sm100.DIRECT_BELOW``). D 128 / 256 from 16 384 rows was tried and dropped: single timer steps either way (D 256 L 128
#: training 22.4 -> 24.4, [1, 16384, 128] inference 8.1 -> 10.1, [5, 7168, 128] 12.2 -> 10.1).
SM100_ROWS: dict[tuple[int, torch.dtype], tuple[tuple[bool, ...], int]] = {
    (64, _BF16): ((False, True), 1),
    (128, _BF16): ((False, True), 65_536),
    (256, _BF16): ((False, True), 65_536),
    (384, _BF16): ((False,), 65_536),
    (768, _F32): ((False, True), 1),
    (2560, _BF16): ((True,), 1),
    (512, _BF16): ((False,), 65_536),
    (64, _F32): ((True,), 1),
    (128, _F32): ((True,), 1),
    (256, _F32): ((True,), 65_536),
    (384, _F32): ((True,), 65_536),
    (512, _F32): ((True,), 65_536),
}


#: inference row counts from which a width of ``SM100_ROWS`` (or one it does not list for inference) takes the row kernels (round 6,
#: graph timing): [20480 / 35840, 128] fp32 12.1 / 14.1 -> 10.0 / 12.0 us against pipe, bf16 [35840, 128] 12.1 -> 10.0, fp32 [2949120, 128]
#: 452 -> 444 (the four-warp non-persistent forward; Triton 445); [16384, 128] level.
SM100_EVAL_FROM: dict[tuple[int, torch.dtype], int] = {(128, _F32): 16_384, (128, _BF16): 16_384}

#: bf16 odd widths (the 267 / 651 pair streams) on the skewed-row kernels (sm100/ln_skew_sm100.cuh, 2026-11-17: a row as aligned 16-byte
#: vectors in registers with a partial first / last vector, one row phase a CTA): {width: (min rows inference, min rows training)}.
#: Kernel durations on the B200 (us, L2 flushed, L 128 / 256 / 384 / 512 / 768; docs/b200/transition.md): 267 forward 11.8 / 23.9 / 42.7 /
#: 67.7 / 140 against the staged kernel's 11.1 / 26.6 / 49.3 / 79.7 / 165, backward 14.0 / 37.1 / 70.1 / 116 / 246 against the
#: warp-per-row kernel's 28.0 / 83.2 / 167 / 287 / 627 (+ a second column pass); 651 forward 22.6 / 49.0 / 90.4 / 149 / 317 against 18.1 /
#: 55.5 / 116 / 200 / 443, backward 26.5 / 74.2 / 150 / 257 / 565 against 44.5 / 134 / 278 / 481 / 1057. The L 128 forward is behind, so
#: inference starts at 65 536 rows; training (forward + backward) wins from L 128.
#: 2026-11-17 (part 2, cuda:b200_sm100 4): fp32 267 / 651 and the 451 / 831 / 833 token streams (both dtypes) train on them (their
#: inference stays on the staged / warp-per-row kernels: NEVER); training steps fp32 [16384, 267 / 651] 36.7 / 73.6 -> 30.5 / 57.1 us, [L, 831 / 833]
#: 20.4-24.4 -> 14.1-20.3 (docs/b200/transition.md, round 3 part 2).
SM100_SKEW: dict[tuple[int, torch.dtype], tuple[int, int]] = {
    (267, _BF16): (65_536, 16_384),
    (651, _BF16): (65_536, 16_384),
    (267, _F32): (NEVER, 1),
    (651, _F32): (NEVER, 1),
    (451, _BF16): (NEVER, 1),
    (831, _BF16): (NEVER, 1),
    (833, _BF16): (NEVER, 1),
    (451, _F32): (NEVER, 1),
    (831, _F32): (NEVER, 1),
    (833, _F32): (NEVER, 1),
}


def path(x: torch.Tensor, train: bool = False) -> str:
    """``"sm100"``, ``"skew"``, ``"pipe"``, ``"rows"`` or ``"general"`` for activation ``x`` (shape / dtype only)."""
    n = x.shape[-1]
    skew = SM100_SKEW.get((n, x.dtype))
    if skew is not None and x.numel() // n >= skew[bool(train)] and sm100.skew_supported(x):
        return "skew"
    modes, min_rows = SM100_ROWS.get((n, x.dtype), ((), 0))
    eval_from = SM100_EVAL_FROM.get((n, x.dtype))
    if not train and eval_from is not None:
        modes, min_rows = (*modes, False), eval_from
    if bool(train) in modes and x.numel() // n >= min_rows and sm100.supported(x):
        return "sm100"
    if pipe.supported(x):
        return "pipe"
    if x.dtype in general.DTYPES and (n * x.element_size()) % 16 != 0 and 32 <= n <= general.ROWS_MAX_N:
        return "rows"
    return "general"


def refusal(x: torch.Tensor, train: bool) -> str | None:
    """Why the B200 route does not take ``x`` (beyond the general kernels' widths, or a cell Triton is faster at), or ``None``."""
    if not general.supported(x):
        return f"width {x.shape[-1]} ({x.dtype}) is beyond the general CUDA LayerNorm's builds"
    rows = x.numel() // x.shape[-1]
    span = SLOWER.get((x.shape[-1], x.dtype, bool(train)))
    if span is not None and span[0] <= rows <= span[1]:
        return (
            f"B200: {rows} rows of width {x.shape[-1]} ({x.dtype}): the Triton LayerNorm is faster here (measured), the CUDA route refuses"
        )
    return None


def _needs_grad(*tensors: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)


def layernorm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension on the B200 kernels (:func:`path`); ``y`` in ``x``'s dtype, fp32 statistics."""
    p = path(x, _needs_grad(x, weight, bias))
    if p == "sm100":
        return sm100.layernorm(x, weight, bias, eps)
    if p == "skew":
        return sm100.layernorm_skew(x, weight, bias, eps)
    if p == "pipe":
        return pipe.layernorm_pipe(x, weight, bias, eps)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    grad = _needs_grad(x, weight, bias)
    if p == "rows":
        if not grad:
            return general._rows_fwd(x2, general._affine(weight, x2), general._affine(bias, x2), float(eps), False)[0].reshape(shape)
        return general._RowsLayerNorm.apply(x2, weight, bias, float(eps)).reshape(shape)
    if not grad:
        return general._fwd(x2, general._affine(weight, x2), general._affine(bias, x2), float(eps), False)[0].reshape(shape)
    return general._LayerNorm.apply(x2, weight, bias, float(eps)).reshape(shape)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension on the B200 kernels: ``pipe`` where it takes the rows, else the general kernels' RMS mode."""
    if pipe.supported(x):
        return pipe.rmsnorm_pipe(x, weight, eps)
    from miniworld_engine.kernels.rmsnorm.cuda import _fwd, _RMSNorm

    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    if not _needs_grad(x, weight):
        return _fwd(x2, general._affine(weight, x2), float(eps), False)[0].reshape(shape)
    return _RMSNorm.apply(x2, weight, float(eps)).reshape(shape)


__all__ = ["SLOWER", "layernorm", "path", "refusal", "rmsnorm"]
