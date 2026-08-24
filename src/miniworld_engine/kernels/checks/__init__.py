"""Helpers shared by more than one family's checkers.

``checks/<family>.py`` holds one reference per registry kernel, keyed by registry.csv's
``family`` column. What lives here instead is the machinery several of those modules need and
none of them owns: the fixed RNG seed, the TF32-off context that makes a reference GEMM true
fp32, the exact-fp32 matmuls, the softmax/log-sum-exp pieces the three attention families share,
and the LayerNorm forward/backward references the three layernorm families share. A helper only
one family uses stays in that family's module.
"""
from __future__ import annotations

import contextlib
import torch

from collections.abc import Callable, Sequence
from miniworld_engine.kernels.drivers import _ln_stats

def _fixed() -> None:
    """Fixed RNG stream per checker: a "WRONG NUMBERS" line reproduces on the next run."""
    torch.manual_seed(0)


@contextlib.contextmanager
def _no_tf32():
    """True-fp32 reference GEMMs (the kernels use tl.dot(input_precision="tf32"))."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

Pair = tuple[torch.Tensor, torch.Tensor]


# ── shared machinery ────────────────────────────────────────────────────────────────────────


def _grads(
    fn: Callable[..., torch.Tensor],
    inputs: Sequence[torch.Tensor],
    ref: Callable[..., torch.Tensor],
    names: Sequence[str],
) -> dict[str, Pair]:
    """Kernel grads vs fp32-autograd grads of ``ref``, on the same values and the same ``dy``.

    The kernel runs on bf16 leaves; the reference runs on fp32 copies of those exact bf16 values,
    so nothing but the arithmetic differs. One ``dy`` is drawn and both backwards get it.
    """
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(*leaves)
    dy = torch.randn_like(out)          # dense: see the module docstring on out.sum().backward()
    out.backward(dy)

    refs = [t.detach().float().requires_grad_(True) for t in inputs]
    ref(*refs).backward(dy.float())
    out_grads: dict[str, Pair] = {}
    for n, lf, rf in zip(names, leaves, refs):
        # A missing grad is a kernel bug, not a comparison to skip: `None` here means the
        # backward never touched that leaf. Name the side that dropped it.
        if lf.grad is None or rf.grad is None:
            missing = "kernel" if lf.grad is None else "reference"
            raise AssertionError(f"{n}: {missing} backward produced no grad")
        out_grads[n] = (lf.grad, rf.grad)
    return out_grads


def _rowsum(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """``delta[..., m] = sum_d o[..., m, d] * do[..., m, d]`` -- what every bwd_pre kernel emits."""
    return (o.float() * do.float()).sum(-1)


def _fp32_matmul() -> None:
    """fp32 has to mean fp32: tf32 would hand the reference the kernel's own 10-bit mantissa."""
    torch.backends.cuda.matmul.allow_tf32 = False


#: The base-2 conversion every one of these kernels hardcodes (``qk_scale *= 1.44269504``).
_LOG2E = 1.44269504


def _lse2(logits: torch.Tensor) -> torch.Tensor:
    """The saved ``m``: a base-2 log-sum-exp over the key axis -- NOT the running row max.

    Each forward runs its softmax in log2 space (``p = exp2(logits*log2e - m_i)``) and then, after
    the key loop closes, folds the denominator in: ``m_i += tl.math.log2(l_i)``. So the stored
    value is ``log2(sum_n 2**(logits*log2e)) == logsumexp(logits) * log2e``, and the backward that
    consumes it recovers ``p`` as ``exp2(logits*log2e - m)`` with no separate ``l``. Checking ``m``
    against the max alone would be wrong by ``log2(l)`` on every row -- and would still "look
    plausible", which is the whole reason the indirect constraint through the backward is weak.
    """
    return torch.logsumexp(logits, -1) * _LOG2E


def _fwd_saved(fn, inputs, m_at: int) -> Pair:
    """``(out, m)`` for a forward whose ``m`` is saved for the backward and never returned.

    ``forward`` returns ``out`` alone; ``m`` reaches the backward through
    ``ctx.save_for_backward``, so the autograd node is the only place a checker can see the value
    the backward will actually consume. ``m_at`` is that tensor's index in the save order, which
    differs per file and is named at each call site. Reaching the kernel through ``.apply`` rather
    than re-launching it keeps the checker on the driver's exact path -- same ``.contiguous()``
    copies, same strides, same grid.
    """
    out = fn(*inputs)
    return out, out.grad_fn.saved_tensors[m_at]

_EPS = 1e-5


# ── references ───────────────────────────────────────────────────────────────────────────────
#
# The math comes from the two reference modules the repo already declares as ground truth for
# these families -- ``layernorm.reference.layernorm_pytorch`` and
# ``layernorm_linear.reference.layernorm_linear_pytorch`` -- rather than a second hand-written
# F.layer_norm here. The only thing added is fp32 promotion and the saved statistics, which the
# reference functions do not return.
#
# Those imports are kept inside the functions, exactly as the drivers keep theirs: both parent
# packages pull triton modules into their ``__init__``, and a module-level import would run that
# when ``run_all`` merely resolves the checker -- turning one bad import into ten failed checks.


def _ln_fwd_ref(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 LayerNorm over the last axis of x -> (y, mean, rstd), stats as fp32 [M].

    ``_ln_stats`` is the same helper the drivers build mean/rstd with: ``rstd = 1/sqrt(var+eps)``
    over the *biased* variance (unbiased=False), which is the LayerNorm definition and what every
    kernel in this family stores.
    """
    from miniworld_engine.kernels.layernorm.reference import layernorm_pytorch

    xf = x.float()
    y = layernorm_pytorch(xf, None if w is None else w.float(), None if b is None else b.float(), eps)
    mean, rstd = _ln_stats(xf, eps)
    return y, mean, rstd


def _ln_bwd_ref(
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 autograd ground truth for LayerNorm backward -> (dx, dgamma, dbeta).

    ``beta`` is taken as zero: dbeta = sum_m dy[m] does not depend on its value, so a kernel
    that never sees a bias is still checked against the right dbeta.
    """
    from miniworld_engine.kernels.layernorm.reference import layernorm_pytorch

    xf = x.float().detach().requires_grad_(True)
    gf = w.float().detach().requires_grad_(True)
    bf = torch.zeros_like(gf).requires_grad_(True)
    layernorm_pytorch(xf, gf, bf, eps).backward(dy.float())
    assert xf.grad is not None and gf.grad is not None and bf.grad is not None
    return xf.grad, gf.grad, bf.grad


def _proj(xn, wa, wb) -> tuple[torch.Tensor, torch.Tensor]:
    """(a, b) = (xn @ wa^T, xn @ wb^T) in fp32 -- the kernels' two fp32 accumulators."""
    xf = xn.float()
    return xf @ wa.float().T, xf @ wb.float().T


def _f(t: torch.Tensor) -> torch.Tensor:
    """bf16 -> fp32, exactly (no value changes). Reference math runs here."""
    return t.detach().float()


def _exact_fp32_matmul() -> None:
    """Pin the reference's fp32 matmuls to real fp32.

    Every GEMM reference below runs ``fp32 @ fp32``. If TF32 were enabled the reference would
    silently drop its operands to 10 mantissa bits -- coarser than the bf16 kernel's fp32
    accumulation -- and a genuine kernel error of a few 1e-3 would be indistinguishable from the
    reference's own rounding. Torch's default for matmul is already False; this makes the
    requirement explicit and survives a process that changed it (the checkers run under
    ``--isolate``, one subprocess per kernel, so the write cannot leak into another kernel's run,
    and none of these kernels goes through an fp32 cuBLAS call itself).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
