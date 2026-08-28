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
import triton

from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.drivers import (
    BF16,
    both_level_is_pair,
    dev,
    driver_length,
    driver_width,
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


# ── trimul_inproj: front / back (triton) ─────────────────────────────────────────────────────

def trimul_gemm_gate_packed_mmajor_triton() -> None:
    """front.py _lr_kernel, via trimul_front_triton."""
    from miniworld_engine.kernels.trimul_inproj.triton.front import trimul_front_triton

    trimul_front_triton(_x(), _w(), _w(), _w(), _w(), _w())


def trimul_outproj_gemm_sigmoid_triton() -> None:
    """front.py _gate_kernel -- the second launch of the same trimul_front_triton front."""
    from miniworld_engine.kernels.trimul_inproj.triton.front import trimul_front_triton

    trimul_front_triton(_x(), _w(), _w(), _w(), _w(), _w())


def trimul_outproj_layernorm_gemm_gate_triton() -> None:
    """back.py _back_kernel, via trimul_back_triton (LN_out + proj + gate, no residual)."""
    from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton

    ln_w = torch.randn(D, device=dev(), dtype=BF16)
    ln_b = torch.randn(D, device=dev(), dtype=BF16)
    trimul_back_triton(_bdll(), _x(), _w(), _w(), ln_w, ln_b)


def trimul_gemm_gate_mmajor_triton() -> None:
    """bidirectional.py _bidir_front_kernel, via bidir_front_triton.

    Per-side hidden H2 = 2*d_hidden = 2*D (module docstring: "H = 2*d_hidden, Din = d_pair").
    """
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
        bidir_front_triton,
    )

    h2 = 2 * D
    bidir_front_triton(_x(), _w(h2), _w(h2), _w(h2), _w(h2))


def gated_projection_gate_dropres_triton() -> None:
    """gate_elem.py _gate_mul_kernel, via gate_elem_triton (no residual/dropout)."""
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton

    # x_n as _x(): gate_elem_triton documents (M,K) OR (B,L,L,K) and its ``_shape_key`` reads
    # ``length_of`` off a 4-D x_n; a 2-D x_n with no seq_len has no L in it and falls to
    # ``token_key(0)`` -> the smallest bucket (128). It flattens x_n itself, so the launch is
    # unchanged. seq_len=L is passed too, which is what every production caller does.
    gate_elem_triton(_x(), _rows(), _w(), seq_len=L)


def gated_projection_bwd_gate_dropres_triton() -> None:
    """gate_elem.py _gate_elem_bwd_ew_kernel, via gate_elem_bwd_ew."""
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_bwd_ew

    # seq_len=L: every argument of gate_elem_bwd_ew is already flattened to (M, N) by contract,
    # so its docstring says seq_len "is the only place L can come from"; without it ``_shape_key``
    # returns ``token_key(0)`` -> the smallest bucket (128) at every length.
    gate_elem_bwd_ew(_rows(), _rows(), _rows(), seq_len=L)


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


def trimul_bwd_gate_transpose_packed_triton() -> None:
    """back_fused.py _dconcat5_kernel, via front_bwd_dW_glogit (the NEGATIVE-RESULT launcher)."""
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import (
        front_bwd_dW_glogit,
    )

    front_bwd_dW_glogit(_bdll(), _bdll(), _bdll(4 * D), _x(),
                        _w(), _w(), _w(), _w(), _rows(), _w())


# ── trimul_inproj/cute: the two @triton.jit kernels living under cute/ ───────────────────────

def trimul_transpose_triton() -> None:
    """front_sm100.py _transpose_kernel, via _transpose_blld_to_bdll ((M,2D) -> (2D,M))."""
    from miniworld_engine.kernels.trimul_inproj.cute.front_sm100 import (
        _transpose_blld_to_bdll,
    )

    # seq_len=L as trimul_front_sm100 passes it: both arguments are already flattened (M rows),
    # so ``seq_len`` is the only place L can come from; without it the launcher keys on
    # ``token_key(0)`` -> the smallest bucket (128) at every length.
    out = torch.empty(2 * D, M, device=dev(), dtype=BF16)
    _transpose_blld_to_bdll(_rows(2 * D), out, seq_len=L)


def gated_projection_gate_packed_mmajor_triton() -> None:
    """front_train_sm100.py _glu_bdll_kernel: preact (4H,M) -> lr (2H,M).

    Launched as the v13 fallback in trimul_front_sm100_train launches it; that launcher is not
    used because its other half is the quack/sm100 front GEMM, which is not this kernel.
    """
    from miniworld_engine.kernels.trimul_inproj.cute.front_train_sm100 import (
        _glu_bdll_kernel,
    )

    h = D
    preact = torch.randn(4 * h, M, device=dev(), dtype=BF16)
    lr = torch.empty(2 * h, M, device=dev(), dtype=BF16)
    grid = lambda meta: (triton.cdiv(h * M, meta["BLOCK_E"]),)
    _glu_bdll_kernel[grid](preact, lr, H=h, M=M, shape_key=token_key(L, H=h))


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
