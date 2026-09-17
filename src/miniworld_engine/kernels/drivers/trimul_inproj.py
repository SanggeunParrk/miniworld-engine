"""Drivers for the ``trimul_inproj`` family -- and the shape block ``tm1``, ``tm2`` and
``gated_projection`` import.

The four families were one module (``drivers_trimul.py``) and still share
``D``/``L``/``IS_PAIR``/``M`` and the ``_x``/``_rows``/``_w``/``_bdll`` builders; the block lives
here because trimul_inproj has the most kernels reading it.

One function per kernel in ``.bench/driver_groups/trimul.tsv``; each launches its kernel once on
the current device and raises on failure. See ``drivers.py`` for the contract.

Shapes come from the repo, not from taste: the default ``D = 128`` is ``BenchConfig.d_pair`` and
``L = 64`` is ``BenchConfig.min_seq_len`` (benchmarks/runners/bench.py), which is also the
smallest L the trimul front/back kernels are documented as verified at (front_sm100.py: "verified
at L=64..1024"). The pair activation these kernels are written for is ``x (1, L, L, D)`` with
``M = L*L`` flattened rows -- exactly what ``bench_kernel_dual_gemm_epil`` /
``bench_kernel_gemm_gate`` build.

Tile alignment: both ``D`` and ``L`` go through ``drivers.ragged()``, so
``MINIWORLD_SHAPE_MODE=ragged`` drops them to 125 / 61 and every axis these kernels tile over
ends in a partial tile at once -- the channel/contraction axis D, both spatial axes of the pair
(L appears twice), and the flattened row count ``M = L*L`` (4096 -> 3721). Unset, the extents are
exactly the repo values above.

Every kernel import is LAZY (inside the driver). Some of these modules import ``quack`` at module
scope (tm1/cute/launch.py, trimul_inproj/cute/front_train_sm100.py); a top-level import here would
make one missing dependency take down all 25 drivers instead of the one it belongs to.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    both_level_is_pair,
    dev,
    driver_length,
    driver_width,
    norm_affine,
    ragged,
)

# Both extents go through ``ragged()`` (see drivers.py): unset MINIWORLD_SHAPE_MODE keeps the
# repo values, MINIWORLD_SHAPE_MODE=ragged subtracts 3 from each.
#
#   D  channel width. Ragged D perturbs the LN reduce axis of the in-projection kernels (which is
#      also the GEMM contraction K) *and* every weight/bias width that multiplies it, since
#      ``_w``/``_bdll``/``_rows`` all default to D and the h2 = 2*D / 4*D packed widths derive
#      from it. 128 -> 125.
#   L  sequence length. The activation is [B, L, L, D], so one perturbation makes BOTH spatial
#      axes ragged at once, and M = L*L (the flattened row count every kernel tiles over) goes
#      ragged with it: 64 -> 61, M 4096 -> 3721.
#: What the packed axis of this family carries. For most kernels here it IS d_pair; for the two
#: `width=pair_bidir` rows the unit hands over `2 * d_pair`, because that is the per-side hidden
#: width a bidirectional trimul meets and it is what lands in the bucket.
D = ragged(driver_width(128))  # BenchConfig.d_pair, or the derived width the unit declares
#: The width projected FROM. `_x()` builds the activation at it and `front_bwd_dW` reads it off
#: `WL.shape[0]`; it is NOT in the cache key, so it does not follow D onto the derived ladder.
_DIN = ragged(128)
L = ragged(driver_length(64))   # BenchConfig.min_seq_len
#: A level=both kernel meets 512 and below as a PAIR activation (1, L, L, D) flattening to
#: M = L*L, and 1024 and above as an ATOM activation (1, A, D) flattening to M = A -- see
#: ``drivers.both_level_is_pair``. Four kernels here are level=both; the other 21 are
#: level=token and are never driven above 512, so the same constant serves both.
IS_PAIR = both_level_is_pair(L)
M = L * L if IS_PAIR else L      # flattened pair rows, or atom rows A


def _x(c: int = D) -> torch.Tensor:
    """The activation these kernels take: pair (1, L, L, c) on the token side, atom (1, A, c) on
    the atom side. ``length_of`` reads shape[-2] either way, so both record the same shape_key.

    ``c`` defaults to the packed width, which is right wherever the kernel projects a square
    d_pair -> d_pair. The two `width=pair_bidir` drivers pass `_DIN`: there the packed axis is the
    per-side HIDDEN width and the activation is still d_pair wide, and `front_bwd_dW` reshapes it
    to `(M, WL.shape[0])` -- so leaving it at the packed width does not merely mis-shape the run,
    it raises.
    """
    if not IS_PAIR:
        return torch.randn(1, L, c, device=dev(), dtype=BF16)
    return torch.randn(1, L, L, c, device=dev(), dtype=BF16)


def _rows(n: int = D) -> torch.Tensor:
    """(M, n) -- the flattened (1, L, L, n) view."""
    return torch.randn(M, n, device=dev(), dtype=BF16)


def _w(n: int = D, din: int = D) -> torch.Tensor:
    """``(din, n)`` weight in x@W form.

    ``din`` is separate because ``front_bwd_dW`` reads the INPUT width off it -- `Din =
    WL.shape[0]` -- while the packed axis is the per-side hidden width `n`. A bidirectional trimul
    has `n = 2 * Din`, so a driver that ties the two can only ever build the square case.
    """
    return (torch.randn(din, n, device=dev(), dtype=BF16) * (din**-0.5)).contiguous()


def _bdll(c: int = D) -> torch.Tensor:
    """Channel-major buffer: pair (1, c, L, L) on the token side, atom (1, c, A) on the atom
    side -- both hold exactly M elements per channel, matching the flat drivers here."""
    if not IS_PAIR:
        return torch.randn(1, c, L, device=dev(), dtype=BF16)
    return torch.randn(1, c, L, L, device=dev(), dtype=BF16)


def _sm100() -> bool:
    """Is this the card whose merged-training cute paths pass `from_preact=True`?

    Lazy import so the module stays importable with no CUDA (the CPU suite imports every driver).
    """
    try:
        from miniworld_engine.modules.dispatch import is_sm100
        return is_sm100()
    except Exception:
        return False


# ── trimul_inproj: front / back (triton) ─────────────────────────────────────────────────────

_EPS = 1e-5   # trimul_back_triton's own default; checks/trimul_inproj.py uses the same


def trimul_outproj_layernorm_gemm_gate_triton() -> None:
    """back.py _back_kernel, via trimul_back_triton (LN_out + proj + gate), fp32 norm affine.

    This fused back is the INFERENCE-only path (``_uni_infer`` on A100/sm86,
    ``_forward_cute_free`` on H100 sm90; training uses ``_UniBackHalfTriton``). There is no
    residual flag left to drive both sides of -- ``residual`` is a required argument and the add
    is unconditional -- so one probe covers the kernel."""
    from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton

    # fp32, NOT BF16. `dtype_of_args` keys on the SET of float operand dtypes, and the norm
    # affine reaches this kernel as a tensor operand: the module holds it in
    # `primitives.LayerNorm`, whose `_Fp32ParamsMixin._apply` pins gamma/beta to fp32 through the
    # trunk's bulk `.to(bfloat16)` (bf16's ULP at 1.0 exceeds Adam's step, so a bf16 gamma never
    # trains). So production launches key `bfloat16+float32` while a bf16 driver recorded plain
    # `bfloat16` -- a different bucket, and every production call missed on the dtype axis alone
    # no matter which shapes or flags were built.
    ln_w, ln_b = norm_affine(D), norm_affine(D)
    # `eps` is POSITIONAL and required. Leaving it out made every unit of this op die with
    # "trimul_back_fused() is missing value for argument 'eps'" before a single config was timed,
    # so the op has never been tuned on any card -- a build failure that reads as a kernel that
    # simply has no cache. The op is registered through `@opaque`, so the miss is a torch.library
    # schema error at call time rather than a TypeError Python could have caught earlier.
    trimul_back_triton(_bdll(), _x(), _w(), _w(), ln_w, ln_b, _EPS, residual=_x())


def trimul_gemm_gate_mmajor_triton() -> None:
    """bidirectional.py _bidir_front_kernel, via bidir_front_triton -- both SAVE_PREACT sides.

    Per-side hidden H2 = 2*d_hidden = 2*D (module docstring: "H = 2*d_hidden, Din = d_pair").
    """
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
        bidir_front_triton,
    )

    h2 = 2 * D
    # BOTH values of SAVE_PREACT, which is in the autotune key (bidirectional.py:64). The default
    # is True, so driving one call built training only and left the whole INFERENCE side unbuilt:
    # `_uni_infer` (unidirectional.py:161) and `_bidir_infer` (bidirectional.py:332) both pass
    # `save_preact=False`, and the =0 kernel is a different program -- it skips the preact tensor
    # and its stores, so it does not want the =1 winner's tile either.
    for h in (h2, D):
        # BOTH hidden widths. `h2 = 2*D` is the BIDIRECTIONAL front (two directions packed into
        # one weight); the UNIDIRECTIONAL front feeds the same kernel with per-side hidden
        # `d_hidden`, which defaults to d_pair -- so its H2 equals K. `dev audit --replay`
        # measured the gap as (H2, K) pairs (128,128), (256,256), (384,384), (512,512): every
        # unidirectional launch, at every length, on the heuristic subset.
        bidir_front_triton(_x(), _w(h), _w(h), _w(h), _w(h), save_preact=True)   # training
        bidir_front_triton(_x(), _w(h), _w(h), _w(h), _w(h), save_preact=False)  # inference


def gated_projection_gate_res_triton() -> None:
    """gate_elem.py _gate_mul_infer_kernel via gate_elem_infer -- the INFERENCE gate store.

    One probe, because the kernel has no flags: no dropout (inference), no saved gate (nothing
    backpropagates through it). Its whole coverage is the shape ladder, and a missing shape shows
    up as a missing cache entry rather than as an unbuilt flag value.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_infer

    # x_n as _x(): gate_elem_infer documents (M,K) OR (B,L,L,K) and its ``_shape_key`` reads
    # ``length_of`` off a 4-D x_n; a 2-D x_n with no seq_len has no L in it and falls to
    # ``token_key(0)`` -> the smallest bucket (128). It flattens x_n itself, so the launch is
    # unchanged. seq_len=L is passed too, which is what every production caller does.
    # Launch sites: bidirectional.py `_bidir_infer`, cute/back_split{,_sm100}.py.
    gate_elem_infer(_x(), _rows(), _w(), _rows(), seq_len=L)


def gated_projection_gate_dropres_triton() -> None:
    """gate_elem.py _gate_mul_train_kernel via gate_elem_train -- the TRAINING gate store.

    One probe, for the same reason: the flags are gone. This kernel always writes the gate (the
    backward needs it) and always applies a drop scale (ones when the model's p_drop is 0), so
    there is no second side to drive. Launch sites: unidirectional.py / bidirectional.py training
    Functions, cute v6_training_merged / bidir_training.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_train

    # residual is [M,N] (the flattened module input pair), dropscale is [L,N] broadcast
    # over the i-index -- per gate_elem_train's docstring, not the 4-D x_n layout.
    ds = torch.rand(L, D, device=dev(), dtype=BF16)
    gate_elem_train(_x(), _rows(), _w(), _rows(), ds, seq_len=L)


def gated_projection_bwd_gate_dropres_triton() -> None:
    """gate_elem.py _gate_elem_bwd_ew_kernel, via gate_elem_bwd_ew. Both USE_DROPOUT, FROM_PREACT=0."""
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_bwd_ew

    # seq_len=L: every argument of gate_elem_bwd_ew is already flattened to (M, N) by contract,
    # so its docstring says seq_len "is the only place L can come from"; without it ``_shape_key``
    # returns ``token_key(0)`` -> the smallest bucket (128) at every length.
    # There is no USE_DROPOUT to drive both sides of: this is the TRAINING backward and every
    # training launch carries a drop scale (ones when the model's p_drop is 0).
    # FROM_PREACT is CARD-DEPENDENT and the branch has to be here, because the registry row is
    # `arch=sm80` and so this driver runs on every card. The =1 side is passed only by the sm100
    # merged-training paths (cute/bidir_training_sm100.py:82, cute/v6_training_merged_sm100.py:68),
    # which `dispatch` selects only there; below sm90 it is a program nothing can launch. Saying
    # "it must be driven on an sm100 build" and then not gating it is how a B200 cache ends up
    # missing half of its training backward.
    ds = torch.rand(L, D, device=dev(), dtype=BF16)
    gate_elem_bwd_ew(_rows(), _rows(), _rows(), ds, L)                        # FROM_PREACT=0
    if _sm100():
        gate_elem_bwd_ew(_rows(), _rows(), _rows(), ds, L, from_preact=True)  # FROM_PREACT=1


def trimul_bwd_gate_packed_triton() -> None:
    """back_fused.py _dconcat_kernel, via front_bwd_dW.

    The bucket is `pack(token_key(L), D=H)` where H is the PER-SIDE hidden width -- and
    `front_bwd_dW` says so itself: "Din = WL.shape[0] (= d_pair); may differ from H
    (bidirectional)". This used to say "Square single-dir (H = Din = D)" and drive H = d_pair,
    which is half of what a bidirectional trimul meets; `dev audit --replay` missed H = 1024
    (2 x 512) at three lengths. The registry row now says `width=pair_bidir` and H arrives here.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW

    # H -- the per-side hidden width, which is what the bucket carries -- comes from the unit.
    # Din, the width being projected FROM, stays the driver's pair width: `front_bwd_dW` reads it
    # off `WL.shape[0]` and it is not in the key, so it does not need its own ladder.
    front_bwd_dW(_bdll(D), _bdll(D), _bdll(4 * D), _x(_DIN),
                 _w(D, _DIN), _w(D, _DIN), _w(D, _DIN), _w(D, _DIN))


def trimul_bwd_gate_packed_recompute_triton() -> None:
    """back_fused.py _dconcat_sig_kernel, via front_bwd_dW_sig (sg = sigma(gate), (1,2D,L,L))."""
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import (
        front_bwd_dW_sig,
    )

    front_bwd_dW_sig(_bdll(), _bdll(), _bdll(), _bdll(), _bdll(2 * D), _x(_DIN),
                     _w(D, _DIN), _w(D, _DIN), _w(D, _DIN), _w(D, _DIN))


# ── trimul_inproj/cute: the two @triton.jit kernels living under cute/ ───────────────────────

def fused_preact_gemm_kernel() -> None:
    """FusedPreactGemmKernel.kernel, via fused_front_gemm (A (M,K); Bp/Bg (2H,K) -> lr, preact)."""
    from miniworld_engine.kernels.trimul_inproj.cute.front_fused_gemm_sm100 import (
        fused_front_gemm,
    )

    h = D
    b = (torch.randn(2 * h, D, device=dev(), dtype=BF16) * (D**-0.5)).contiguous()
    lr = torch.empty(2 * h, M, device=dev(), dtype=BF16)
    preact = torch.empty(4 * h, M, device=dev(), dtype=BF16)
    fused_front_gemm(_rows(), b, b.clone(), lr, preact)


def masked_front_sm90():
    """Actual per-side inference and stacked training projection contracts."""
    from miniworld_engine.kernels.trimul_inproj.cute.masked_front import masked_front
    a = _rows(D)
    mask = torch.ones(M, device=dev(), dtype=torch.bool)
    mask[::3] = False
    # Inference calls left/right separately, each with gate+value (2D).
    # Training stacks both sides (4D), or both sides and directions (8D).
    for projected, save in ((2 * D, False), (4 * D, True), (8 * D, True)):
        weight = torch.randn(D, projected, device=dev(), dtype=BF16)
        masked_front(a, weight, mask, save)
