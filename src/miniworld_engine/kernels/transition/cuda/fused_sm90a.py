"""Fused sm_90a Transition: one kernel for the forward, one (plus a partial reduction) for the
backward.

This is the hand-CUDA path developed in ``experiments/transition_fused``. It replaces five
launches -- LayerNorm, expand-SwiGLU, squeeze+residual on the way forward, and the
squeeze/SwiGLU/LN backward chain on the way back -- with two, keeping the whole op resident in
shared memory and registers. Measured on an H100 SXM against the engine's own Triton residual
path, at the AF3 pair width:

    op        L=384          L=768
    forward   272 -> 130 us  1076 ->  500 us
    backward  821 -> 406 us  3196 -> 1605 us
    module   1109 -> 583 us  4106 -> 2307 us   (modules.Transition, forward + backward)

**It is Hopper-only and shape-specific by construction.** The tile sizes, the shared-memory
budget and the two-role CTA split are all written for ``d_hidden == 128`` with ``n == 4``
(hidden 512) in bf16, and the persistent grid is one CTA per SM with the SM count compiled in.
``supported()`` is the whole gate; everything it rejects keeps the existing path, unchanged.

Numerics differ from the Triton path in the last bf16 ulp, not in the contract: the statistics
are the same LayerNorm statistics, the intermediate roundings happen at the same places, and
``dgamma``/``dbeta`` are summed in a fixed CTA order (no atomics), so a replay is bit-identical.
"""

import functools
import os
import warnings
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent

#: Row tile of the persistent grid. ``M`` must be a whole number of these.
ROWS = 128
#: Hidden slice replicas in the backward's weight role: NDW = 8 * DW_REPL weight CTAs.
#: 8 measured fastest at both lengths; see ``experiments/transition_fused/records/``.
DW_REPL = 8


def _sm_count(device: torch.device | int | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=8)
def _ext(ctas: int, dw_repl: int, save: bool):
    """Build for a given SM count. ``ctas`` is in the module name, so a second device with a
    different multiprocessor count gets its own build rather than a silently wrong grid.

    ``save`` picks the forward variant. Writing ``xn`` and the LayerNorm statistics is inside
    the normalization epilogue; the training build writes them unconditionally, and the
    inference build keeps the stores in the code but skips them at run time. Compiling them
    out instead produced a smaller-register schedule that was 7-8 % slower than the training
    build; behind the runtime guard the inference forward is 15-16 % faster than that
    (bit-identical output), and it still allocates no M x 128 tensor it does not return.
    """
    ensure_cuda_home()
    return load_extension(
        name=f"transition_fused_sm90a_c{ctas}_r{dw_repl}_s{int(save)}",
        sources=[str(_dir / "transition_fused_sm90a.cu"),
                 str(_dir / "transition_fused_fwd_sm90a_kernel.cu"),
                 str(_dir / "transition_fused_bwd_sm90a_kernel.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"),
                           f"-I{_dir / 'anthropic_v5'}", f"-DNCTA={ctas}", f"-DDW_REPL={dw_repl}",
                           f"-DFWD_SAVE={int(save)}",
                           "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
        extra_cflags=["-std=c++17"], verbose=False,
    )


def _ext_for(x: torch.Tensor, save: bool = True):
    return _ext(_sm_count(x.device), DW_REPL, save)


@functools.lru_cache(maxsize=8)
def _is_hopper(index: int) -> bool:
    major, minor = torch.cuda.get_device_capability(index)
    return (major, minor) == (9, 0)


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether the fused kernels can run this call. Everything else keeps the current path.

    The requirements are the kernel's own, not a policy: sm_90 for wgmma/TMA, bf16 because the
    operand fragments are bf16, ``d_hidden == 128`` and hidden ``512`` because the tile shapes
    are literals, and a whole number of 128-row tiles because the persistent grid has no
    ragged-tail path.
    """
    if os.environ.get("MINIWORLD_TRANSITION_FUSED_SM90A", "1") == "0":
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    if not _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    if x.shape[-1] != 128 or wa.shape != (512, 128) or ws.shape != (128, 512):
        return False
    rows = 1
    for s in x.shape[:-1]:
        rows *= s
    return rows % ROWS == 0


_BUILD_FAILED = False


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build, both cached.

    A JIT build needs an nvcc that matches torch. Failing to find one is an environment
    problem, not a dispatch bug, so it warns once and keeps the existing path rather than
    taking a training run down.
    """
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(x, wa, ws):
        return False
    # FakeTensor dispatch records shapes only; it must not invoke nvcc or
    # wait for another process's extension lock. Launch wrappers have fakes.
    if _is_fake(x, wa, ws):
        return True
    try:
        # Build the variant this call will actually use, so inference never pays for the
        # training forward's build.
        _ext_for(x, torch.is_grad_enabled())
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"fused sm90a Transition unavailable, keeping the existing path: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


def _is_fake(*tensors) -> bool:
    """True under FakeTensorMode, where the launch must not be entered.

    `dev derive` runs every module under FakeTensorMode with ``compile_wrap="disable"``, so an
    opaque op's BODY runs on fake tensors. It intercepts Triton launches and records them; it
    cannot intercept a native extension, which then reads a data pointer that does not exist
    ("the tensor has a non-zero number of elements, but its data is not allocated yet"). That
    failed all 504 Transition units and 234 Pairformer units of the sm90 plan. This path has no
    autotuned Triton kernel, so recording nothing for it is the correct derivation.
    """
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


def _fwd_launch_fake(x, gamma, beta, wa, wb, wst, eps, save):
    """Output structure only. It may branch on ``save`` -- a compile-time argument -- and on
    nothing the GPU decides."""
    rows = x.shape[0] if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    return (torch.empty_like(x),
            torch.empty_like(x) if save else x.new_empty((1, x.shape[1])),
            torch.empty((rows,), **f32), torch.empty((rows,), **f32))


@opaque(fake=_fwd_launch_fake, name="transition_fused_fwd_sm90a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor,
                wb: torch.Tensor, wst: torch.Tensor, eps: float, save: bool,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The launch, split out of the autograd Function so Dynamo can trace up to it and past it.
    Returns (out, xn, rstd, c1), all freshly allocated."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, wst, eps, save)
    return tuple(_ext_for(x, save).transition_fused_fwd(x, gamma, beta, wa, wb, wst, eps, save))


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws):
    """Output structure only: the six gradients, each shaped like what it is a gradient of."""
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma),
            torch.empty_like(wa), torch.empty_like(wb), torch.empty_like(ws))


@opaque(fake=_bwd_launch_fake, name="transition_fused_bwd_sm90a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, rstd: torch.Tensor,
                c1: torch.Tensor, gamma: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                ws: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws)
    return tuple(_ext_for(x).transition_fused_bwd(dy, x, xn, rstd, c1, gamma, wa, wb, ws))


class _FusedTransitionSM90A(torch.autograd.Function):
    """``y = transition(x) + x`` with the residual folded into the squeeze epilogue.

    The forward saves ``xn`` and the LayerNorm statistics the backward needs. Recomputing ``xn``
    instead was measured and is slower: it saves 9 us of the forward and costs ~40 us in the
    backward's weight role, which reads it on the critical path.
    """

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        save = any(ctx.needs_input_grad)
        out, xn, rstd, c1 = _fwd_launch(
            flat, gf, bf, wa.contiguous(), wb.contiguous(), ws.t().contiguous(), float(eps), save)
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws)
        ctx.shape = shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, gf,
            wa.contiguous(), wb.contiguous(), ws.contiguous())
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt),
                dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None)


def transition_fused_sm90a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as the Triton ``transition_residual``.

    Call ``supported(x, wa, ws)`` first: this raises rather than falling back, so a dispatch bug
    shows up as an error instead of a silent slowdown.

    Inference is decided HERE, not inside the autograd Function: its ``forward`` always runs with
    grad mode off, and ``ctx.needs_input_grad`` only reflects ``requires_grad`` -- which module
    parameters keep under ``torch.no_grad()`` -- so deciding there sent every no_grad call through
    the training build, writing an M x 128 ``xn`` nobody reads (38 MB at L384, 151 MB at L768).
    """
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        out, _, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma.float().contiguous(),
                                   beta.float().contiguous(), wa.contiguous(), wb.contiguous(),
                                   ws.t().contiguous(), float(eps), False)
        return out.reshape(shape)
    return _FusedTransitionSM90A.apply(x, gamma, beta, wa, wb, ws, eps)
