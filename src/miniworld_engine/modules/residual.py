"""The fused residual add of parallel branches (FoldForge request R12).

A parallel stage whose parts own their residuals (each part returns ``y_i = x + delta_i``; Chai-1's MSA block: the MSA attention and the
MSA transition on the MSA, the two triangle multiplications and the pair transition on the pair, then the two triangle attentions) updates
``x`` by the sum of the parts' deltas::

    out = x + sum_i (y_i - x)

:func:`parallel_residual` / :class:`ParallelResidual` compute it. ``"pytorch"`` (the reference) runs the statement with the differences and
the sum in at least fp32 (``wide``) and one rounding to ``x``'s dtype; ``"triton"`` is the same in one pass over the tensors
(``kernels/parallel_residual/triton``: reads ``x`` and the branches once, writes once) for CUDA tensors of one dtype (bf16 / fp16 / fp32),
1 to 4 branches; its backward is torch statements. ``"cuda"`` is the same arithmetic in a hand-written kernel
(``kernels/parallel_residual/cuda``: one source, the H100 sm_90a and A100 sm_80 builds; 16-byte vector accesses, several vectors in
flight per thread, streaming cache hints) over the same range, with its backward's ``(1 - n) g`` in one small kernel too. The backward
is ``d out / d y_i = g`` (no kernel), ``d out / d x = (1 - n) g`` (fp32 product, one rounding). Ladder exception (user decision
2026-11-11, docs/ladder.md): at the bandwidth wall ``"cuda"`` may be slower than ``"triton"`` in eval (1-5 % at L >= 384); it beats
``"pytorch"`` everywhere and ``"triton"`` in training.

``PairformerBlock(parallel=True)`` keeps its own statements for its parallel schedule (its records are unchanged); a caller composing a
parallel stage from engine parts uses this op.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from miniworld_engine.modules.primitives import BackendModule, random_input, stream_shape, wide

MAX_BRANCHES = 4  # kernels/parallel_residual/triton/main.py MAX_BRANCHES


def _reference(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    w = wide(x, *branches)
    xw = x.to(w)
    total = branches[0].to(w) - xw
    for y in branches[1:]:
        total = total + (y.to(w) - xw)
    return (xw + total).to(x.dtype)


class _ParallelResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *branches):
        from miniworld_engine.kernels.parallel_residual.triton.main import parallel_residual_fwd

        ctx.n = len(branches)
        return parallel_residual_fwd(x.contiguous(), [y.contiguous() for y in branches])

    @staticmethod
    def backward(ctx, g):
        gx = (g.to(wide(g)) * (1 - ctx.n)).to(g.dtype)
        return (gx, *([g] * ctx.n))


class _ParallelResidualCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *branches):
        from miniworld_engine.kernels.parallel_residual import cuda as K

        ctx.n = len(branches)
        return K.forward(x.contiguous(), [y.contiguous() for y in branches])

    @staticmethod
    def backward(ctx, g):
        from miniworld_engine.kernels.parallel_residual import cuda as K

        return (K.backward(g, ctx.n), *([g] * ctx.n))


def _cuda_reason(x: torch.Tensor, branches: list[torch.Tensor]) -> str | None:
    """The Triton range (the kernels read the same tensors) on the cards the CUDA kernel is built for (H100 sm_90a, A100 sm_80, RTX 5090 sm_120a)."""
    from miniworld_engine.modules.pair_embed import _cuda_arch

    why = _triton_reason(x, branches) or _cuda_arch(x.device)
    return None if why is None else why.replace("the Triton kernel", "the CUDA kernel")


def _triton_reason(x: torch.Tensor, branches: list[torch.Tensor]) -> str | None:
    if not 1 <= len(branches) <= MAX_BRANCHES:
        return f"{len(branches)} branches (the Triton kernel reads 1 to {MAX_BRANCHES})"
    if not x.is_cuda:
        return "CUDA tensors only"
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {x.dtype} (bfloat16 / float16 / float32)"
    for y in branches:
        if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
            return f"every branch must have x's shape / dtype / device: {tuple(x.shape)} {x.dtype}, got {tuple(y.shape)} {y.dtype}"
    return None


def parallel_residual(x: torch.Tensor, *branches: torch.Tensor, backend: str = "pytorch") -> torch.Tensor:
    """``x + sum_i (y_i - x)``: the sum of the parallel parts' deltas added to ``x`` (each ``y_i`` a part's residual output). ``backend``:
    ``"pytorch"`` (any branch count, any device), ``"cuda"`` or ``"triton"`` (:class:`ParallelResidual`'s range; outside it
    ``BackendUnsupported``)."""
    if not branches:
        raise ValueError("parallel_residual needs at least one branch")
    if backend == "pytorch":
        return _reference(x, list(branches))
    if backend not in ("cuda", "triton"):
        raise ValueError(f"parallel_residual has no backend {backend!r}; available: pytorch, cuda, triton")
    why = (_cuda_reason if backend == "cuda" else _triton_reason)(x, list(branches))
    if why is not None:
        from miniworld_engine.modules.primitives import BackendUnsupported

        raise BackendUnsupported(f"parallel_residual backend {backend!r}: {why}")
    return (_ParallelResidualCudaFn if backend == "cuda" else _ParallelResidualFn).apply(x, *branches)


class ParallelResidual(BackendModule):
    """:func:`parallel_residual` as a module (no parameters): ``forward(x, y0, y1=None, y2=None, y3=None)`` -> ``x + sum_i (y_i - x)`` over
    the given branches. ``d`` (the channel width) and ``branches`` (the branch count) shape the measurement inputs only."""

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backends = ("pytorch", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {"cuda:5090_sm120": 1}  # 2026-11-12: the sm_120a build (RTX 5090)

    def __init__(self, d: int = 256, branches: int = 2, *, backend: str = "pytorch") -> None:
        super().__init__()
        if not 1 <= branches <= MAX_BRANCHES:
            raise ValueError(f"branches must be 1 .. {MAX_BRANCHES}, got {branches}")
        self.d, self.branches = d, branches
        self.set_backend(backend)

    @staticmethod
    def _branches(y0, y1, y2, y3) -> list[torch.Tensor]:
        return [y for y in (y0, y1, y2, y3) if y is not None]

    def _forward_pytorch(self, x, y0, y1=None, y2=None, y3=None) -> torch.Tensor:
        return _reference(x, self._branches(y0, y1, y2, y3))

    def _supports_cuda(self, x, y0, y1=None, y2=None, y3=None) -> str | None:
        return _cuda_reason(x, self._branches(y0, y1, y2, y3))

    def _forward_cuda(self, x, y0, y1=None, y2=None, y3=None) -> torch.Tensor:
        return _ParallelResidualCudaFn.apply(x, *self._branches(y0, y1, y2, y3))

    def _supports_triton(self, x, y0, y1=None, y2=None, y3=None) -> str | None:
        return _triton_reason(x, self._branches(y0, y1, y2, y3))

    def _forward_triton(self, x, y0, y1=None, y2=None, y3=None) -> torch.Tensor:
        return _ParallelResidualFn.apply(x, *self._branches(y0, y1, y2, y3))

    def extra_repr(self) -> str:
        return f"d={self.d}, branches={self.branches}, {super().extra_repr()}"

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``x`` on the row's stream (``token_pair`` [S, L, L, d], ``msa_token`` [S, D, L, d]) and the branches ``x + 0.1 N(0, 1)`` (a part's
        residual output: its input plus a small update)."""
        shape = stream_shape(stream, samples=samples, length=length, depth=depth, width=self.d)
        x = random_input(shape, dtype=torch.float32, device=device, generator=generator)
        out = {"x": x.to(dtype)}
        for i in range(self.branches):
            out[f"y{i}"] = (x + random_input(shape, dtype=torch.float32, device=device, generator=generator, scale=0.1)).to(dtype)
        return out


__all__ = ["ParallelResidual", "parallel_residual"]
