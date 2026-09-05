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
D = ragged(driver_width(128))  # BenchConfig.d_pair
L = ragged(driver_length(64))   # BenchConfig.min_seq_len
#: A level=both kernel meets 512 and below as a PAIR activation (1, L, L, D) flattening to
#: M = L*L, and 1024 and above as an ATOM activation (1, A, D) flattening to M = A -- see
#: ``drivers.both_level_is_pair``. Four kernels here are level=both; the other 21 are
#: level=token and are never driven above 512, so the same constant serves both.
IS_PAIR = both_level_is_pair(L)
M = L * L if IS_PAIR else L      # flattened pair rows, or atom rows A


def _x() -> torch.Tensor:
    """The activation these kernels take: pair (1, L, L, D) on the token side, atom (1, A, D) on
    the atom side. ``length_of`` reads shape[-2] either way, so both record the same shape_key."""
    if not IS_PAIR:
        return torch.randn(1, L, D, device=dev(), dtype=BF16)
    return torch.randn(1, L, L, D, device=dev(), dtype=BF16)


def _rows(n: int = D) -> torch.Tensor:
    """(M, n) -- the flattened (1, L, L, n) view."""
    return torch.randn(M, n, device=dev(), dtype=BF16)


def _w(n: int = D) -> torch.Tensor:
    """(D, n) weight in x@W form."""
    return (torch.randn(D, n, device=dev(), dtype=BF16) * (D**-0.5)).contiguous()


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

def trimul_outproj_layernorm_gemm_gate_triton() -> None:
    """back.py _back_kernel, via trimul_back_triton (LN_out + proj + gate), fp32 norm affine.

    ADD_RESIDUAL=1 ONLY:
    this fused back is the INFERENCE-only path (``_uni_infer`` on A100/sm86, ``_forward_cute_free``
    on H100 sm90; training uses ``_UniBackHalfTriton``), and inference always fuses the module's
    UNCONDITIONAL pairformer residual (dropout is off outside training), so ADD_RESIDUAL is 1 in
    every production call. The no-residual bucket exists only for the manual ``_ADD_RESIDUAL=False``
    raw-op benchmark toggle -- not worth a committed cache entry -- so it is not driven here."""
    from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton

    # fp32, NOT BF16. `dtype_of_args` keys on the SET of float operand dtypes, and the norm
    # affine reaches this kernel as a tensor operand: the module holds it in
    # `primitives.LayerNorm`, whose `_Fp32ParamsMixin._apply` pins gamma/beta to fp32 through the
    # trunk's bulk `.to(bfloat16)` (bf16's ULP at 1.0 exceeds Adam's step, so a bf16 gamma never
    # trains). So production launches key `bfloat16+float32` while a bf16 driver recorded plain
    # `bfloat16` -- a different bucket, and every production call missed on the dtype axis alone
    # no matter which shapes or flags were built.
    ln_w, ln_b = norm_affine(D), norm_affine(D)
    trimul_back_triton(_bdll(), _x(), _w(), _w(), ln_w, ln_b, residual=_x())     # ADD_RESIDUAL=1


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


def gated_projection_gate_dropres_triton() -> None:
    """gate_elem.py _gate_mul_kernel, via gate_elem_triton -- the three reachable flag combos."""
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton

    # x_n as _x(): gate_elem_triton documents (M,K) OR (B,L,L,K) and its ``_shape_key`` reads
    # ``length_of`` off a 4-D x_n; a 2-D x_n with no seq_len has no L in it and falls to
    # ``token_key(0)`` -> the smallest bucket (128). It flattens x_n itself, so the launch is
    # unchanged. seq_len=L is passed too, which is what every production caller does.
    # ADD_RESIDUAL / USE_DROPOUT / SAVE_GATE are all in the key (gate_elem.py:47) and the
    # per-op build has no switch axis of its own, so whatever the driver does not call is never
    # built. The reachable combinations, from the production call sites:
    #   inference       bidirectional.py:343   residual, no gate out   -> 1,0,0
    #   training        unidirectional.py:83 / bidirectional.py:252    -> 1,0,1 and 1,1,1
    #                   (`return_gate=True`; dropscale carries pairformer's p_drop=0.25)
    # BOTH values of ADD_RESIDUAL. The module path is always =1 (`_ADD_RESIDUAL = True` at
    # module.py:212 / bidirectional.py:106), which is what replay measured -- 18 misses, all =1.
    # But `trimul_inproj/whole_op.py`, the public `ops` facade, calls the triton trimul without
    # `add_residual`, and `unidirectional.py:207` / `bidirectional.py:356` default it False, so
    # `residual_flat` is None and the =0 program launches. Below sm90 that facade resolves to
    # these very kernels, so =0 is reachable on this card, not just on cute.
    # residual is [M,N] (the flattened module input pair), dropscale is [L,N] broadcast
    # over the i-index -- per gate_elem_triton's docstring, not the 4-D x_n layout.
    res, ds = _rows(), torch.rand(L, D, device=dev(), dtype=BF16)
    gate_elem_triton(_x(), _rows(), _w(), seq_len=L)                                    # 0,0,0
    gate_elem_triton(_x(), _rows(), _w(), seq_len=L, return_gate=True)                  # 0,0,1
    gate_elem_triton(_x(), _rows(), _w(), seq_len=L, residual=res)                      # 1,0,0
    gate_elem_triton(_x(), _rows(), _w(), seq_len=L, residual=res, return_gate=True)    # 1,0,1
    gate_elem_triton(_x(), _rows(), _w(), seq_len=L, residual=res, dropscale=ds,
                     return_gate=True)                                                  # 1,1,1


def gated_projection_bwd_gate_dropres_triton() -> None:
    """gate_elem.py _gate_elem_bwd_ew_kernel, via gate_elem_bwd_ew. Both USE_DROPOUT, FROM_PREACT=0."""
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_bwd_ew

    # seq_len=L: every argument of gate_elem_bwd_ew is already flattened to (M, N) by contract,
    # so its docstring says seq_len "is the only place L can come from"; without it ``_shape_key``
    # returns ``token_key(0)`` -> the smallest bucket (128) at every length.
    # USE_DROPOUT: both, because pairformer runs p_drop=0.25 in training and 0 in inference
    # (unidirectional.py:105, bidirectional.py:273 pass `dropscale=ctx.dropscale`).
    # FROM_PREACT is CARD-DEPENDENT and the branch has to be here, because the registry row is
    # `arch=sm80` and so this driver runs on every card. The =1 side is passed only by the sm100
    # merged-training paths (cute/bidir_training_sm100.py:82, cute/v6_training_merged_sm100.py:68),
    # which `dispatch` selects only there; below sm90 it is a program nothing can launch. Saying
    # "it must be driven on an sm100 build" and then not gating it is how a B200 cache ends up
    # missing half of its training backward.
    ds = torch.rand(L, D, device=dev(), dtype=BF16)
    gate_elem_bwd_ew(_rows(), _rows(), _rows(), seq_len=L)                              # 0,0
    gate_elem_bwd_ew(_rows(), _rows(), _rows(), seq_len=L, dropscale=ds)                # 1,0
    if _sm100():
        gate_elem_bwd_ew(_rows(), _rows(), _rows(), from_preact=True, seq_len=L)        # 0,1
        gate_elem_bwd_ew(_rows(), _rows(), _rows(), from_preact=True, dropscale=ds,
                         seq_len=L)                                                     # 1,1


def trimul_bwd_gate_packed_triton() -> None:
    """back_fused.py _dconcat_kernel, via front_bwd_dW. Square single-dir (H = Din = D)."""
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW

    front_bwd_dW(_bdll(), _bdll(), _bdll(4 * D), _x(), _w(), _w(), _w(), _w())


def trimul_bwd_gate_packed_recompute_triton() -> None:
    """back_fused.py _dconcat_sig_kernel, via front_bwd_dW_sig (sg = sigma(gate), (1,2D,L,L))."""
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import (
        front_bwd_dW_sig,
    )

    front_bwd_dW_sig(_bdll(), _bdll(), _bdll(), _bdll(), _bdll(2 * D), _x(),
                     _w(), _w(), _w(), _w())


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
