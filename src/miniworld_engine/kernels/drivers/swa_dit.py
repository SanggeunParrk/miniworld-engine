"""Driver for the ``swa_dit`` family: the fused ESMFold2 SWA atom DiT block (``kernels/swa_dit``).

The shapes are the atom transformer's calls as the models make them: one batch element (B = 1) of S atoms, A augments on
the N = A*B rows, the adaLN modulation hoisted once per (b, atom) from the augment-invariant conditioning. d_atom 128 /
4 heads / SwiGLU hidden 256 / half window 64 are the only widths the kernels serve. S follows ``driver_length``; every
launch keys on ``atom_key(S, A=A, ...)`` -- the augment count exactly -- so each driver runs every A the models use:

  A = 1    MiniWorld's input feature embedder (``build_attention_params(..., num_aug=1)``), training and inference
  A = 48   diffusion training (``num_augment: 48`` in every MiniWorld phase-2 config)
  A = 5    diffusion evaluation samples (inference forward only)

Every bf16 Triton driver pins the hand-CUDA stages off (``settings.swa_dit_*_cuda=False``): on sm_90 they would otherwise
take the qkvg and FFN stages and the Triton kernel would never launch. The forward drivers run the inference forward
(SAVE=0) and the training forward (SAVE=1); ``swa_dit_swiglu_bwd_triton`` runs both FFN weight-gradient modes (DWOPS=1
and 0), so every key flag is built on both sides. The ``*_sm90_cuda`` drivers pin their stage on and raise when the
extension is not available, rather than quietly timing the Triton fallback.

The block dispatches on the activation dtype, so which kernels a driver reaches is the process's
``MINIWORLD_DRIVER_DTYPE``: the ``*_fp32_*`` rows are fp32 and the rest bf16, and each driver refuses to run at the
other precision rather than tune the other half's kernels under its name. The ``*_fp32_triton`` drivers pin the fp32
CUDA extension off; the ``*_fp32_sm90_cuda`` drivers pin it on and run under TF32-allowing matmul precision, the only
one it serves (every stage of ``cuda/swa_fp32.cu`` launches in each fp32 forward or backward, so they share drivers).
"""
from __future__ import annotations

import contextlib

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    DTYPE_MODE,
    aligned_only,
    dev,
    driver_length,
    ragged,
)

_S = ragged(driver_length(1024))
_A = 48
_B = 1
_C = aligned_only("swa_dit.d_atom", 128, "the CUDA tiles and the Triton tl.arange extents are d_atom 128 = 4 heads x 32")
_H = 4
_NHID = aligned_only("swa_dit.n_hidden", 256, "the SwiGLU hidden width the CUDA FFN tiles are written for")
_HALF_WINDOW = 64
#: Augment counts driven: the embedder's 1 and diffusion's training 48 everywhere, diffusion's 5 evaluation samples on
#: the inference forward (see the module docstring).
_TRAIN_AUGMENTS = (1, 48)
_INFER_AUGMENTS = (1, 5, 48)

#: The Triton kernels on every card: every hand-CUDA stage off.
_TRITON = {"swa_dit_qkvg_fwd_cuda": False, "swa_dit_ffn_fwd_cuda": False, "swa_dit_ffn_bwd_cuda": False,
           "swa_dit_attn_cuda": False, "swa_dit_oproj_bwd_cuda": False, "swa_dit_qkvg_bwd_cuda": False}


@contextlib.contextmanager
def _pinned(**pins):
    """``settings`` pinned for the duration and restored after: a driver that leaves a setting changed changes what every
    later driver in the same build process tunes."""
    from miniworld_engine import settings

    previous = settings.configure(**pins)
    try:
        yield
    finally:
        settings.configure(**{name: getattr(previous, name) for name in pins})


def _require(precision: str) -> None:
    """Refuse to drive a row at a precision it does not declare: the dtype decides which kernels the block launches."""
    if precision != DTYPE_MODE:
        msg = (f"this swa_dit row is {precision}; the process builds {DTYPE_MODE} activations, which launch the "
               f"other precision's kernels. Run it with MINIWORLD_DRIVER_DTYPE={precision}.")
        raise RuntimeError(msg)


def _inputs(dtype=BF16, *, a=_A, b=_B, s=_S, seed=0):
    """(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window) for ``interface.swa_dit_block``.

    q [a*b, s, 128]; mod [b*s, 768] fp32, hoisted from a conditioning [b, s, 128]; cos / sin [b, s, 16] fp32 (per batch
    element, as ``build_attention_params`` holds them before repeating over the augments); seqused [a*b] int32, ragged
    and front-packed; the five block weights.
    """
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_hoist_modulation

    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(dev(), dtype)

    n = a * b
    q = r(n, s, _C)
    c_base = r(b, s, _C)
    wmod = r(6 * _C, _C, scale=0.05)
    mod = swa_dit_hoist_modulation(c_base, wmod).detach()
    angles = torch.randn(b, s, _C // _H // 2, generator=g) * 3.0
    cos = angles.cos().to(dev(), torch.float32)
    sin = angles.sin().to(dev(), torch.float32)
    seqused = torch.tensor([max(1, s - (7 * i) % 29) for i in range(n)], dtype=torch.int32, device=dev())
    weights = (r(3 * _C, _C, scale=_C ** -0.5), r(_C, _C, scale=_C ** -0.5), r(_C, _C, scale=_C ** -0.5),
               r(2 * _NHID, _C, scale=_C ** -0.5), r(_C, _NHID, scale=_NHID ** -0.5))
    return (q, mod, cos, sin, seqused, *weights, b, _HALF_WINDOW)


def _require_cuda(which: str) -> None:
    """Raise unless the hand-CUDA extension for stage ``which`` is usable here."""
    from miniworld_engine.kernels.swa_dit.cuda.loader import ERRORS, extension, is_sm90

    if not is_sm90(dev()):
        msg = "sm90 (H100) only: the swa_dit CUDA kernels are sm_90a wgmma code"
        raise RuntimeError(msg)
    if extension(which) is None:
        msg = f"the swa_dit CUDA extension {which!r} did not build: {ERRORS.get(which)}"
        raise RuntimeError(msg)


def _forward(**pins) -> None:
    """The inference forward (SAVE=0) at every inference augment count, the training forward (SAVE=1) at the training
    ones."""
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block

    with _pinned(**pins):
        for a in _INFER_AUGMENTS:
            args = _inputs(a=a)
            with torch.no_grad():
                swa_dit_block(*args)
            if a in _TRAIN_AUGMENTS:
                swa_dit_block(args[0].detach().requires_grad_(), *args[1:])


def _backward(**pins) -> None:
    """Training forward + backward with every differentiable input live, at every training augment count."""
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block

    with _pinned(**pins):
        for a in _TRAIN_AUGMENTS:
            args = _inputs(a=a)
            q, mod = (t.detach().requires_grad_() for t in args[:2])
            weights = [w.detach().requires_grad_() for w in args[5:10]]
            out = swa_dit_block(q, mod, *args[2:5], *weights, *args[10:])
            out.backward(torch.randn_like(out))


def swa_dit_inproj_fwd_triton() -> None:
    _require("bf16")
    _forward(**_TRITON)


def swa_dit_softmax_fwd_triton() -> None:
    _require("bf16")
    _forward(**_TRITON)


def swa_dit_output_swiglu_fwd_triton() -> None:
    _require("bf16")
    _forward(**_TRITON)


def swa_dit_swiglu_bwd_triton() -> None:
    """DWOPS=1 (the materialised dW operands, the default) and DWOPS=0 (``swa_dit_ffn_dw="fused"``)."""
    _require("bf16")
    _backward(**_TRITON, swa_dit_ffn_dw="mat")
    _backward(**_TRITON, swa_dit_ffn_dw="fused")


def swa_dit_swiglu_dw_triton() -> None:
    _require("bf16")
    _backward(**_TRITON, swa_dit_ffn_dw="fused")


def swa_dit_output_bwd_triton() -> None:
    _require("bf16")
    _backward(**_TRITON)


def swa_dit_softmax_bwd_dq_triton() -> None:
    _require("bf16")
    _backward(**_TRITON)


def swa_dit_softmax_bwd_dkdv_triton() -> None:
    _require("bf16")
    _backward(**_TRITON)


def swa_dit_inproj_bwd_triton() -> None:
    _require("bf16")
    _backward(**_TRITON)


def swa_dit_inproj_fwd_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("qkvg")
    _forward(engine_backend="auto", swa_dit_qkvg_fwd_cuda=True, swa_dit_rows_impl="wgmma")


def swa_dit_output_swiglu_fwd_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("fwd")
    _forward(engine_backend="auto", swa_dit_ffn_fwd_cuda=True, swa_dit_rows_impl="wgmma")


def swa_dit_swiglu_bwd_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("bwd")
    _backward(engine_backend="auto", swa_dit_ffn_bwd_cuda=True, swa_dit_ffn_dw="mat", swa_dit_rows_impl="wgmma")


def swa_dit_softmax_fwd_sm90_cuda() -> None:
    """The window attention forward (inference and training forwards)."""
    _require("bf16")
    _require_cuda("attn")
    _forward(engine_backend="auto", swa_dit_attn_cuda=True)


def swa_dit_softmax_bwd_sm90_cuda() -> None:
    """The window attention backward (dq, dk, dv in one kernel), at both of its output layouts: into the CUDA qkvg
    backward's buffer, and head-major for the Triton qkvg backward."""
    _require("bf16")
    _require_cuda("attn")
    _backward(engine_backend="auto", swa_dit_attn_cuda=True, swa_dit_qkvg_bwd_cuda=True)
    _backward(engine_backend="auto", swa_dit_attn_cuda=True, swa_dit_qkvg_bwd_cuda=False)


def swa_dit_output_bwd_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("slab")
    _backward(engine_backend="auto", swa_dit_oproj_bwd_cuda=True)


def swa_dit_inproj_bwd_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("slab")
    _backward(engine_backend="auto", swa_dit_qkvg_bwd_cuda=True)


# The slab kernels of the stages that also have a team-gm wgmma kernel (``swa_dit_rows_impl`` pinned to "slab").

def swa_dit_inproj_fwd_rows_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("slab")
    _forward(engine_backend="auto", swa_dit_qkvg_fwd_cuda=True, swa_dit_rows_impl="slab")


def swa_dit_output_fwd_rows_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("slab")
    _forward(engine_backend="auto", swa_dit_ffn_fwd_cuda=True, swa_dit_rows_impl="slab")


def swa_dit_swiglu_fwd_rows_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("slab")
    _forward(engine_backend="auto", swa_dit_ffn_fwd_cuda=True, swa_dit_rows_impl="slab")


def swa_dit_swiglu_bwd_rows_sm90_cuda() -> None:
    _require("bf16")
    _require_cuda("slab")
    _backward(engine_backend="auto", swa_dit_ffn_bwd_cuda=True, swa_dit_ffn_dw="mat", swa_dit_rows_impl="slab")


# ---- the hoisted modulation and the weight-gradient GEMMs (sm_90a, cuda/swa_mod.cu, cuda/swa_wgrad.cu) ----------------

def _modulation(*, backward: bool) -> None:
    """``swa_dit_hoist_modulation`` on the sm_90a kernels: the forward, and the backward through c and Wmod."""
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_hoist_modulation

    g = torch.Generator(device="cpu").manual_seed(0)
    with _pinned(engine_backend="auto", swa_dit_mod_cuda=True):
        for b in (1, 3):
            c = (torch.randn(b, _S, _C, generator=g)).to(dev(), BF16).requires_grad_(backward)
            w = (torch.randn(6 * _C, _C, generator=g) * 0.05).to(dev(), BF16).requires_grad_(backward)
            mod = swa_dit_hoist_modulation(c, w)
            if backward:
                mod.backward(torch.randn(mod.shape, generator=g).to(dev(), torch.float32))


def swa_dit_gemm_fwd_sm90_cuda() -> None:
    """The modulation forward silu(c) Wmod^T (TMA + wgmma)."""
    _require("bf16")
    _require_cuda("modwg")
    _modulation(backward=False)


def swa_dit_gemm_bwd_sm90_cuda() -> None:
    """The modulation backward (dc and dWmod, one launch of two roles)."""
    _require("bf16")
    _require_cuda("mod")
    _modulation(backward=True)


#: Every bf16 stage on the hand-CUDA slab kernels, so the backward reaches the one-launch weight-gradient kernel (it serves up
#: to ``dispatch._WGRAD_MAX_ROWS`` rows: the A = 1 backward here, not the 48-augment one).
_WGRAD = {"engine_backend": "auto", "swa_dit_wgrad_cuda": True, "swa_dit_ffn_bwd_cuda": True, "swa_dit_oproj_bwd_cuda": True,
          "swa_dit_qkvg_bwd_cuda": True, "swa_dit_attn_cuda": True, "swa_dit_ffn_dw": "mat", "swa_dit_rows_impl": "slab"}


def swa_dit_dw_sm90_cuda() -> None:
    """The weight gradients on TMA + wgmma (``swa_wgrad_wg.cu``): every row count, so both training augment counts."""
    _require("bf16")
    _require_cuda("slab")
    _require_cuda("attn")
    _require_cuda("wgwg")
    _backward(**_WGRAD)


# ---- fp32 Triton (the window attention kernels are shared with bf16 and driven by its rows) -------------------------
#: The fp32 Triton kernels on every card: the fp32 CUDA extension off.
_TRITON_FP32 = {"swa_dit_fp32_cuda": False}


def swa_dit_inproj_fwd_fp32_triton() -> None:
    _require("fp32")
    _forward(**_TRITON_FP32)


def swa_dit_output_swiglu_fwd_fp32_triton() -> None:
    _require("fp32")
    _forward(**_TRITON_FP32)


def swa_dit_swiglu_bwd_fp32_triton() -> None:
    _require("fp32")
    _backward(**_TRITON_FP32)


def swa_dit_output_bwd_fp32_triton() -> None:
    _require("fp32")
    _backward(**_TRITON_FP32)


def swa_dit_inproj_bwd_fp32_triton() -> None:
    _require("fp32")
    _backward(**_TRITON_FP32)


# ---- fp32 CUDA (sm_90a, cuda/swa_fp32.cu): one extension, every stage; not autotuned (fixed tiles) ------------------
#: The fp32 CUDA extension pinned on.
_CUDA_FP32 = {"engine_backend": "auto", "swa_dit_fp32_cuda": True}


@contextlib.contextmanager
def tf32_allowed():
    """The fp32 CUDA stages serve only while torch's fp32 matmul precision allows TF32 ("high" / "medium", MiniWorld's
    trainer setting): "high" for the duration, the previous precision restored after."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _cuda_fp32(run) -> None:
    _require("fp32")
    _require_cuda("fp32")
    with tf32_allowed():
        run(**_CUDA_FP32)


def swa_dit_transpose_fp32_sm90_cuda() -> None:
    """prep_kernel: the weights as the fp32 GEMMs stream them, launched by every forward and backward."""
    _cuda_fp32(_backward)


def swa_dit_inproj_fwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_forward)


def swa_dit_softmax_fwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_forward)


def swa_dit_output_swiglu_fwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_forward)


def swa_dit_swiglu_bwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_backward)


def swa_dit_output_bwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_backward)


def swa_dit_softmax_bwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_backward)


def swa_dit_inproj_bwd_fp32_sm90_cuda() -> None:
    _cuda_fp32(_backward)


def swa_dit_dw_fp32_sm90_cuda() -> None:
    """The fp32 weight gradients on TMA + wgmma (``swa_wgrad_wg_f32.cu``), served up to ``dispatch._WGRAD_F32_WG_MAX_ROWS`` rows."""
    _require("fp32")
    _require_cuda("fp32")
    _require_cuda("wgwg32")
    with tf32_allowed():
        _backward(**_CUDA_FP32, swa_dit_wgrad_cuda=True)


def swa_dit_transpose_sm90_cuda() -> None:
    """The wgmma FFN kernels' four weight layouts in one launch: driven by an A = 48 backward (the rows where they serve)."""
    from miniworld_engine.kernels.swa_dit.interface import swa_dit_block

    _require("bf16")
    _require_cuda("pack")
    _require_cuda("bwd")
    _require_cuda("fwd")
    with _pinned(engine_backend="auto", swa_dit_ffn_fwd_cuda=True, swa_dit_ffn_bwd_cuda=True, swa_dit_rows_impl="wgmma"):
        args = _inputs(a=_A)
        q, mod = (t.detach().requires_grad_() for t in args[:2])
        weights = [w.detach().requires_grad_() for w in args[5:10]]
        out = swa_dit_block(q, mod, *args[2:5], *weights, *args[10:])
        out.backward(torch.randn_like(out))
