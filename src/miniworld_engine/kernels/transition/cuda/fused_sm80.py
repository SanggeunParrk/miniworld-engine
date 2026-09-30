"""Fused sm_80 Transition: one kernel for the forward, two (plus the partial sums) for the backward.

The hand-CUDA path developed in ``experiments/a100_transition_fwd`` and ``experiments/a100_transition_bwd``.
Measured on an A100 80GB PCIe (300 W) against the engine's own Triton residual path, at the AF3 pair width
(CUDA-graph replay, same node):

    op                         L=384          L=768
    forward (inference)        392 -> 288 us  1514 -> 1100 us   (Anthropic ``pf`` kernel as the reference)
    training step (fwd + bwd)  2119 -> 1470 us  8301 -> 5700 us  (modules.Transition)

* forward: 8 warps x 32 rows, LayerNorm into registers, [a | b] GEMM per 32-hidden chunk through a cp.async ring, SwiGLU,
  squeeze in f16 with the residual as the accumulator's initial value.  Training also writes ``xn`` and (mean, rstd).
* backward kernel 1 (PW): per 64-unit hidden slice x row replica: a, b, dh of the slice from ``xn`` / ``dy``, the SwiGLU backward,
  and the slice's dWa, dWb, dWs^T (f32 partials); dA | dB leave as fragment-native blocks, h never leaves the SM.
* backward kernel 2 (X): d_xn = [dA | dB] [Wa; Wb] (f32), the LayerNorm backward and the residual -> dx, dgamma / dbeta partials.

Every product is computed once (16 M D H in the backward).  The card runs at its power cap under this load, so the split that wins
is the one with the least on-chip work per FLOP, not the fewest launches (a one-launch version measured 13 % slower); see the
experiments' READMEs.

**Ampere-only and shape-specific by construction**: sm_80, bf16, ``d_hidden == 128`` and hidden 512, rows a multiple of 256.
``supported()`` is the whole gate; everything it rejects keeps the existing path.  ``dgamma`` / ``dbeta`` / the weight gradients are
summed in a fixed order (no atomics), so a replay is bit-identical.
"""

import functools
import hashlib
import os
import warnings
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

#: Rows must be a whole number of these (the backward's 256-row tiles).
ROWS = 256
D, H, CH = 128, 512, 32


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_TRANSITION_SM80_FLAGS: extra -D flags for A/B experiments (their own build); PW_2DA=0 is not supported here (the wx
    # layout carries 0.5 Wa)
    extra = os.environ.get("MINIWORLD_TRANSITION_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"transition_fused_sm80{tag}",
        sources=[str(_dir / "transition_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", "-DPW_XN", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@functools.lru_cache(maxsize=8)
def _is_ampere(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """The kernels' own requirements: sm_80, bf16, D = 128 / hidden 512 (tile shapes are literals), whole 256-row tiles."""
    if os.environ.get("MINIWORLD_TRANSITION_FUSED_SM80", "1") == "0":
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    if not _is_ampere(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    if x.shape[-1] != D or wa.shape != (H, D) or ws.shape != (D, H):
        return False
    rows = 1
    for s in x.shape[:-1]:
        rows *= s
    return rows > 0 and rows % ROWS == 0


_BUILD_FAILED = False


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful (cached) build; a build failure warns once and keeps the existing path."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(x, wa, ws):
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"fused sm80 Transition unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


# ------------------------------------------------------------------------------------------------------------ weight layouts
def _k_perm(dev):
    """Physical GEMM1 k index 16 s + kk -> input column (thread q of a quad holds the 16 B vectors at 32 i + 8 q)."""
    s = torch.arange(8, device=dev).view(-1, 1)
    kk = torch.arange(16, device=dev).view(1, -1)
    return (32 * (s // 2) + 8 * ((kk % 8) // 2) + 4 * (s % 2) + 2 * (kk // 8) + kk % 2).reshape(-1)


def _o_perm(dev):
    """Physical GEMM2 output column 8 J + 2 q + e -> output column 32 (J / 4) + 8 q + 2 (J % 4) + e."""
    pcol = torch.arange(D, device=dev)
    J, q, e = pcol // 8, (pcol % 8) // 2, pcol % 2
    return 32 * (J // 4) + 8 * q + 2 * (J % 4) + e


def _layout_fwd(wa, wb, ws, dev):
    """Forward weight layout, applied to element codes: [16 chunks][W1 (0.5 Wa | Wb, k-permuted) | W2 (Ws, output-permuted)]."""
    kp = _k_perm(dev)
    wa_p, wb_p = wa[:, kp], wb[:, kp]
    ws_p = ws[_o_perm(dev)]
    nstep, nchunk = CH // 16, H // CH
    w1 = torch.stack([wa_p.view(nchunk, nstep, 16, D), wb_p.view(nchunk, nstep, 16, D)], 2).reshape(nchunk, 2 * CH, D)
    w1 = w1.view(nchunk, 2 * CH, 16, 8).transpose(1, 2).reshape(nchunk, -1)
    w2 = ws_p.view(D, nchunk, CH // 8, 8).permute(1, 2, 0, 3).reshape(nchunk, -1)
    return w1, w2


def _layout_bwd(wa, wb, ws, dev):
    """PW: per 64-unit slice [W1s 16 k-granules x 128 rows (0.5 Wa | Wb) | W3s = Ws^T slice 16 x 64]; X: [Wa | Wb] per chunk."""
    op = _o_perm(dev)
    nchunk = H // CH
    w1s = torch.stack([wa.view(8, 4, 16, D), wb.view(8, 4, 16, D)], 2).reshape(8, 128, D).view(8, 128, 16, 8).transpose(1, 2)
    w3s = ws.t().reshape(8, 64, 16, 8).transpose(1, 2)
    wdw = torch.cat([w1s.reshape(8, -1), w3s.reshape(8, -1)], 1).reshape(-1)
    xa = wa[:, op].reshape(nchunk, CH, 16, 8).transpose(1, 2).reshape(nchunk, -1)
    xb = wb[:, op].reshape(nchunk, CH, 16, 8).transpose(1, 2).reshape(nchunk, -1)
    return wdw, torch.cat([xa, xb], 1).reshape(-1)


N_W, N_WDW, N_WX = (H // CH) * (2 * CH * D + CH * D), 8 * (128 * D + 64 * D), (H // CH) * 2 * CH * D


@functools.lru_cache(maxsize=8)
def _pack_tables(dev):
    """Gather tables of the one-launch pack kernel: out16 = [w | wdw | wx], out32 = gb (LN affine float4 slots).
    idx16: element | source << 26 (Wa, Wb, Ws) | x0.5 << 28 | f16 << 29; idx32: element | beta << 26."""
    n = H * D
    code = lambda src: (torch.arange(n, device=dev, dtype=torch.int64) | (src << 26))  # noqa: E731
    half = 1 << 28
    wa, wb = code(0).view(H, D), code(1).view(H, D)
    ws = code(2).view(D, H)
    w1, w2 = _layout_fwd(wa | half, wb, ws | (1 << 29), dev)          # W1 carries 0.5 Wa; W2 runs in f16
    w = torch.cat([w1, w2], 1).reshape(-1)
    wdw, wx = _layout_bwd(wa | half, wb, ws, dev)                     # W1s carries 0.5 Wa
    _, wx = _layout_bwd(wa | half, wb, ws, dev)                       # X: 0.5 Wa (PW's dA tile holds 2 dA, build flag PW_2DA)
    idx16 = torch.cat([w, wdw, wx]).to(torch.int32).contiguous()
    assert idx16.numel() == N_W + N_WDW + N_WX
    s_, q_ = torch.arange(8, device=dev).view(-1, 1), torch.arange(4, device=dev).view(1, -1)
    c0 = (32 * (s_ // 2) + 8 * q_ + 4 * (s_ % 2)).reshape(-1, 1) + torch.arange(4, device=dev).view(1, -1)
    idx32 = torch.cat([c0.reshape(-1), c0.reshape(-1) | (1 << 26)]).to(torch.int32).contiguous()
    return idx16, idx32


def _pack(gamma, beta, wa, wb, ws):
    """All kernel weight layouts in one launch -> (w, gb, wdw, wx, gamma f32, beta f32)."""
    dev = wa.device
    idx16, idx32 = _pack_tables(dev)
    gf, bf = gamma.float().contiguous(), beta.float().contiguous()
    out16 = torch.empty(idx16.numel(), dtype=torch.bfloat16, device=dev)
    gb = torch.empty(idx32.numel(), dtype=torch.float32, device=dev)
    _pack_launch(wa.contiguous(), wb.contiguous(), ws.contiguous(), gf, bf, idx16, idx32, out16, gb)
    return out16[:N_W], gb, out16[N_W:N_W + N_WDW], out16[N_W + N_WDW:], gf, bf


# ------------------------------------------------------------------------------------------------------------------ launches
def _pack_launch_fake(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, idx16: torch.Tensor,
                      idx32: torch.Tensor, out16: torch.Tensor, gb: torch.Tensor) -> None:
    """Nothing: the op writes out16 and gb in place."""
    return None


@opaque(fake=_pack_launch_fake, name="transition_fused_pack_sm80", mutates_args=("out16", "gb"))
def _pack_launch(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, idx16: torch.Tensor,
                 idx32: torch.Tensor, out16: torch.Tensor, gb: torch.Tensor) -> None:
    """Packs the three weights (bf16 / fp16 tiles in out16) and the LayerNorm affine (gb) for the sm80 kernels, in place."""
    if _is_fake(wa, out16):
        return None
    _ext().pack(wa, wb, ws, gamma, beta, idx16, idx32, out16, gb)


def _fwd_launch_fake(x, w, gb, eps, save):
    """out like x; xn [rows, D] and fp32 stats [rows, 2] when save, else empty."""
    rows = x.shape[0] if save else 0
    return (torch.empty_like(x), x.new_empty((rows, D)) if save else x.new_empty((0,)),
            torch.empty((rows, 2), dtype=torch.float32, device=x.device))


@opaque(fake=_fwd_launch_fake, name="transition_fused_fwd_sm80")
def _fwd_launch(x: torch.Tensor, w: torch.Tensor, gb: torch.Tensor, eps: float, save: bool,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(out, xn, stats): xn / stats are empty unless ``save``."""
    if _is_fake(x, w):
        return _fwd_launch_fake(x, w, gb, eps, save)
    return tuple(_ext().fwd(x, w, gb, eps, save))


def _bwd_launch_fake(dy, x, xn, stats, wdw, wx, gamma, beta, eps, adt, wdt):
    """dx like x; dgamma, dbeta [D] in adt; dWa, dWb [H, D] and dWs [D, H] in wdt."""
    return (torch.empty_like(x), torch.empty((D,), dtype=adt, device=x.device), torch.empty((D,), dtype=adt, device=x.device),
            torch.empty((H, D), dtype=wdt, device=x.device), torch.empty((H, D), dtype=wdt, device=x.device),
            torch.empty((D, H), dtype=wdt, device=x.device))


@opaque(fake=_bwd_launch_fake, name="transition_fused_bwd_sm80")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, stats: torch.Tensor, wdw: torch.Tensor, wx: torch.Tensor,
                gamma: torch.Tensor, beta: torch.Tensor, eps: float, adt: torch.dtype, wdt: torch.dtype,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dgamma, dbeta in ``adt``, dWa, dWb, dWs in ``wdt``); dx carries the residual branch.  The partial sums are reduced in the
    kernel's fixed order and written in the parameters' dtypes (one launch for the sum, the 0.5, the dWs transpose and the casts)."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, stats, wdw, wx, gamma, beta, eps, adt, wdt)
    return tuple(_ext().bwd(dy, x, xn, stats, wdw, wx, gamma, beta, eps, int(os.environ.get("MINIWORLD_TRANSITION_SM80_NREP", "0")),
                            adt, wdt))


class _FusedTransitionSM80(torch.autograd.Function):
    """``y = transition(x) + x``.  The forward saves ``xn`` and (mean, rstd): the backward's PW kernel reads ``xn`` directly
    (normalising ``x`` in each of its 8 slice CTAs instead cost 240 us at L384)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        w, gb, wdw, wx, gf, bf = _pack(gamma, beta, wa, wb, ws)
        out, xn, stats = _fwd_launch(flat, w, gb, float(eps), True)
        ctx.save_for_backward(flat, xn, stats, wdw, wx, gf, bf)
        ctx.shape, ctx.eps = shape, float(eps)
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, stats, wdw, wx, gf, bf = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        wd = adt if adt in (torch.float32, torch.bfloat16) and adt == bwdt == sdt else torch.float32
        ad = gdt if gdt in (torch.float32, torch.bfloat16) and gdt == bdt else torch.float32
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, stats, wdw, wx, gf, bf,
                                                     ctx.eps, ad, wd)
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None)


def transition_fused_sm80(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as the Triton ``transition_residual``.  Call ``available(x, wa, ws)`` first.
    Inference (no grad needed) runs the forward without writing xn / stats."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        w, gb, _, _, _, _ = _pack(gamma, beta, wa, wb, ws)
        out, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), w, gb, float(eps), False)
        return out.reshape(shape)
    return _FusedTransitionSM80.apply(x, gamma, beta, wa, wb, ws, eps)
