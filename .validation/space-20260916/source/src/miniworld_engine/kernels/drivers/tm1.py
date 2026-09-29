"""Drivers for the ``tm1`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``drivers_trimul.py``) and still
share ``D``/``L``/``IS_PAIR``/``M`` and the ``_x``/``_rows``/``_w``/``_bdll`` builders, which
live in ``drivers/trimul_inproj.py`` together with the shape and lazy-import rationale for all
four.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, aligned_only, dev, ragged
from miniworld_engine.kernels.drivers.trimul_inproj import D, L, _bdll, _rows, _w, _x

# ── tm1 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_gemm_gate_triton() -> None:
    """fused_sigmoid_gate_fwd_kernel, via triton_tm1."""
    from miniworld_engine.kernels.tm1.triton.main import triton_tm1

    # _x(), not _rows(): TritonTM1Function reads token_key(length_of(x.shape)) BEFORE its own
    # rearrange to (M, d), so a pre-flattened (M, d) gives it M = L*L -> clamped to 512.
    triton_tm1(_x(), _w(), _w(), _w(), _w())


def trimul_bwd_gate_recompute_triton() -> None:
    """fused_sigmoid_gate_bwd_kernel, via TritonTM1Function.backward."""
    from miniworld_engine.kernels.tm1.triton.main import triton_tm1

    # _x(): same reason as the forward -- ctx.original_shape is what the backward keys on.
    x = _x().requires_grad_()
    left, right = triton_tm1(x, _w(), _w(), _w(), _w())
    (left.sum() + right.sum()).backward()


def gated_projection_gate_inplace_flat_triton() -> None:
    """tm1/cute/launch.py _gate_mul_kernel, via _fused_gate_mul (proj *= sigmoid(gate))."""
    from miniworld_engine.kernels.tm1.cute.launch import _fused_gate_mul

    # seq_len=L as tm1_cute_forward passes it: the launcher keys on
    # ``token_key(seq_len if seq_len is not None else 0)``, so omitting it pins the key at the
    # SMALLEST bucket (128) regardless of the shape actually launched. bdll is [B, D, L, L], so
    # its own shape[-2] is L only by coincidence and the launcher does not read it.
    _fused_gate_mul(_bdll().contiguous(), _bdll().contiguous(), seq_len=L)


def gated_projection_gate_packed_flat_triton() -> None:
    """tm1/cute/launch.py _glu_wide_kernel, via _glu_wide (wide (1,2D,L,L) -> out (1,D,L,L))."""
    from miniworld_engine.kernels.tm1.cute.launch import _glu_wide

    out = torch.empty(1, D, L, L, device=dev(), dtype=BF16)
    _glu_wide(out, _bdll(2 * D), D, L)


def gated_persistent_gemm_kernel() -> None:
    """GatedPersistentGemmKernel.kernel, via gate_gemm (the tm1 bdll_sm100 entry point)."""
    from miniworld_engine.kernels.tm1.cute.sm100_gate_gemm_collective import gate_gemm

    gate_gemm(_rows(), _w(), _w())


def persistent_dense_gemm_kernel() -> None:
    """PersistentDenseGemmKernel.kernel, via the module's own ``run`` entry point.

    mnkl / majors / tiler are this file's own CLI defaults (prepare_parser + __main__), with
    bf16 operands. ``skip_ref_check`` stays False on purpose: ``run`` only calls the compiled
    kernel inside that branch (with it set, and benchmark off, it returns without launching).
    """
    import cutlass

    from miniworld_engine.kernels.tm1.cute import _blackwell_dense_gemm as bdg

    # m is the only one of the three that this kernel's own precondition leaves free, so it is
    # the one that carries the ragged tail here. n and k are pinned by
    # _blackwell_dense_gemm.py:1287 ``check_contiguous_16B_alignment`` -> ``num_major_elements %
    # num_contiguous_elements == 0``: with a_major="k"/b_major="k" the checked mode of A and B is
    # k, with c_major="n" the checked mode of C is n, and num_contiguous_elements is 16*8//16 = 8
    # for bf16, so k%8 and n%8 must be 0 or ``can_implement`` raises CantImplementError
    # (_blackwell_dense_gemm.py:1295) before anything launches. m is checked in no mode.
    n = aligned_only(
        "tm1.trimul_gemm_sm100_cute.n",
        256,
        "_blackwell_dense_gemm.py:1287 check_contiguous_16B_alignment(c_dtype, c_major=='m', "
        "(m,n,l)) reads mode 1 = n for c_major='n' and requires n % (16*8//16 == 8) == 0; "
        "otherwise line 1295 raises CantImplementError('Invalid tensor alignment: ...')",
    )
    k = aligned_only(
        "tm1.trimul_gemm_sm100_cute.k",
        512,
        "_blackwell_dense_gemm.py:1287 check_contiguous_16B_alignment for A and B reads mode 1 "
        "= k for a_major='k'/b_major='k' and requires k % (16*8//16 == 8) == 0; otherwise line "
        "1295 raises CantImplementError('Invalid tensor alignment: ...')",
    )
    bdg.run((ragged(256), n, k, 1), cutlass.BFloat16, cutlass.BFloat16, cutlass.Float32,
            "k", "k", "n", mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
            use_2cta_instrs=False, use_tma_store=True)
