"""B200 (sm_100a) Transition dispatch: the one entry ``modules.Transition``'s ``"cuda"`` backend calls on capability (10, 0).

Order (the B200 port's ``kernels/transition/whole_op.py``, branch ``b200/ladder-final``):

* ``fused_sm100a`` -- D 128, n 4, bf16 activations, whole 128-row tiles (one persistent forward, a two-role backward);
* ``fused_wide_sm100a`` -- the tuned builds of every other (D, n) in its ``SHAPES`` (D 64 / 128 / 256 / 384 / 512 with n 2 / 4, D 768
  with n 2), bf16 activations, whole 128-row tiles (D 384 at most 16 tiles: the small-M PDL chains);
* a bf16 call with a ragged row count is padded with zero rows to the next 128-row tile for those two (measured faster than the generic
  kernels wherever a tuned build exists) and cut back;
* ``fused_generic_sm100a`` -- any D a multiple of 64 up to 1024 with a hidden width a multiple of 128, any row count, bf16 or fp32 on
  TF32 tensor cores.

The weights may be bf16 or fp32 masters (cast inside, their gradients fp32). ``None`` when no build takes the call.
"""

from __future__ import annotations

import torch

from . import fp8_sm100a, fused_generic_sm100a, fused_small_sm100a, fused_sm100a, fused_wide_sm100a

#: The tuned builds in the order they are tried (each ``available`` is its own whole gate).
_TUNED = (
    (fused_sm100a, fused_sm100a.transition_fused_sm100a),
    (fused_wide_sm100a, fused_wide_sm100a.transition_wide_sm100a),
)

#: (D, n, training) cells that skip the tuned builds and run the generic kernels in a row range: {(D, n, train): (min rows, max rows)}
#: (measured: see docs/b200/transition.md).
GENERIC: dict[tuple[int, int, bool], tuple[int, int]] = {
    # D 768 n 2 single stream (rows 37 / 219, [1, L <= 2048, 768], master): the per-tile wide build is launch / weight bound at a few tiles;
    # generic 30-43 us eval (wide 59-65, Triton 55-68) and 61-76 us train (wide 168-174, Triton 129-135)
    (768, 2, False): (1, 2048),
    (768, 2, True): (1, 2048),
    # D 384 bf16 inference from 8 row tiles at n 4 and 16 at n 2 (rows 35 / 36): the generic kernels with the cuBLAS squeeze. Round 3 part 8
    # (2026-11-17): the small-M chain's squeeze (one 64-column block per CTA) loses there (graph, L2 flushed, us, chain -> generic:
    # [1, 1024 / 1536 / 2048, 384] n 4 20.5-22.4 / 24.5 / 28.7 -> 18.5 / 20.4 / 24.4, [1, 2976 / 3968, 384] n 4 32.6 / 34.7 -> 28.6 / 30.6 (generic
    # with the RESID squeeze 38.8 / 40.8, wide 59.2 / 55.2); n 2 L 2048 20.4 -> 18.4; kept on the chain: n 4 L 512-768 16.4-18.2 against
    # 18.4, n 2 L 768-1536 14.3-16.4 against 16.4-18.4). Part 18: fused_small_sm100a.INFER_CELLS (checked before this table) takes n 4
    # 8-12 tiles and n 2 16 tiles (and D 768 n 2 7-12 tiles) on the chain with the CTA-pair squeeze; the rest of these ranges stays here
    (384, 4, False): (1024, 4096),
    (384, 2, False): (1921, 4096),
}


#: (N1) the folded D 384 chain of ``fused_wide_sm100a`` ahead of ``fused_small_sm100a``'s chains (n 4 8 - 12 tiles, n 2 16 tiles); False: A/B
FOLD_FIRST = True


def _wants_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _padded(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(x itself, or its rows zero-padded to whole 128-row tiles as [rows', D]; the true row count)."""
    d = x.shape[-1]
    rows = x.numel() // d
    if rows % fused_sm100a.ROWS == 0 or x.dtype is not torch.bfloat16:
        return x, rows
    return torch.nn.functional.pad(x.reshape(rows, d), (0, 0, 0, -rows % fused_sm100a.ROWS)), rows


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether some sm_100a build takes the call (shape / dtype / card only: no build)."""
    xs, _ = _padded(x) if x.numel() else (x, 0)
    return any(k.supported(xs, wa, ws) for k, _ in _TUNED) or fused_generic_sm100a.supported(x, wa, ws)


def transition_sm100a(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps: float,
    _fp8: bool = True,
) -> torch.Tensor | None:
    """``x + W_s(silu(W_a LN(x)) * (W_b LN(x)))`` on the B200 kernels, or ``None`` when none takes the call."""
    tuned = _TUNED
    grad = _wants_grad(x, gamma, beta, wa, wb, ws)
    if x.dtype is torch.float32 and not fused_generic_sm100a._tf32_allowed():  # fp32 with TF32 off = IEEE products: the engine's IEEE fp32 CUDA chain (kernels/ieee_f32, built for sm_100a)
        from miniworld_engine.kernels import ieee_f32

        f = torch.float32
        w = [t.to(f) for t in (gamma, beta, wa, wb, ws)]  # a cast stays in autograd: a bf16 master's gradient comes back in its dtype
        if not ieee_f32.supported_transition(x, w[2], w[4], hopper=True):
            return None
        return ieee_f32.transition_ieee(x, w[0].contiguous(), w[1].contiguous(), w[2].contiguous(), w[3].contiguous(), w[4].contiguous(), float(eps))
    if _fp8 and fp8_sm100a.POLICY is not None and not grad and fp8_sm100a.serves(x, wa, ws):  # opt-in e4m3 route (docs/b200/fp8.md); POLICY is None by default
        return fp8_sm100a.inference(x, gamma, beta, wa, wb, ws, eps)
    if FOLD_FIRST and not grad and fused_wide_sm100a.fold_first(x, wa, ws):  # (N1) the LayerNorm-folded D 384 chain (2 launches) before the small-M chains (3 launches)
        return fused_wide_sm100a.transition_wide_sm100a(x, gamma, beta, wa, wb, ws, eps)
    if fused_small_sm100a.available(x, wa, ws, grad):  # the small-M chains of the cells the D 384 bf16 chains do not take
        return fused_small_sm100a.transition_small_sm100a(x, gamma, beta, wa, wb, ws, eps)
    # (N1) the zero-row pad (two kernels, ~4 us in a graph) only where a tuned build runs: the small-M chains take a ragged row count themselves,
    # the generic kernels any
    rows = x.numel() // x.shape[-1]
    cell = (x.shape[-1], wa.shape[0] // x.shape[-1], grad)
    if cell in GENERIC and GENERIC[cell][0] <= rows <= GENERIC[cell][1]:
        tuned = ()
    xs = _padded(x)[0] if tuned else x
    for kern, entry in tuned:
        if kern.available(xs, wa, ws):
            out = entry(xs, gamma, beta, wa, wb, ws, eps)
            return out if xs is x else out[:rows].reshape(x.shape)
    if fused_generic_sm100a.available(x, wa, ws):
        return fused_generic_sm100a.transition_generic_sm100a(x, gamma, beta, wa, wb, ws, eps)
    return None


__all__ = ["supported", "transition_sm100a"]
