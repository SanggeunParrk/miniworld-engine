"""Drivers for the trimul_inproj / tm1 / tm2 / gated_projection kernels.

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

from miniworld_engine.autotune.shape_key import both_key, token_key
from miniworld_engine.kernels.drivers import (
    BF16,
    aligned_only,
    both_level_is_pair,
    dev,
    driver_length,
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
D = ragged(128)  # BenchConfig.d_pair
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


# ── gated_projection/triton/main.py ──────────────────────────────────────────────────────────

def gated_projection_gate_triton() -> None:
    """sigmoid_gate_fwd_kernel, via TritonGatedProjectionFunction.

    ``_x()``, not ``_rows()``: the wrapper takes ``* hd`` and flattens to (M, hd) itself, and it
    reads ``both_key(length_of(original_shape))`` from the shape it was HANDED. Handing it the
    already-flattened (M, D) makes length_of return M = L*L, which clamps to the top bucket 8192
    at every L >= 91 -- the launched shape is identical either way, only the key differs.
    """
    from miniworld_engine.kernels.gated_projection.triton.main import triton_gated_projection

    triton_gated_projection(_x(), _x(), _w())


def gated_projection_bwd_gate_triton() -> None:
    """sigmoid_gate_bwd_kernel, launched as TritonGatedProjectionFunction.backward launches it.

    The autograd path is not used: that backward returns ``.float()`` grads for bf16 inputs.
    """
    from miniworld_engine.kernels.gated_projection.triton.main import (
        sigmoid_gate_bwd_kernel,
    )

    gate, x, grad_out = _rows(), _rows(), _rows()
    dgate, dx = torch.empty_like(gate), torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    sigmoid_gate_bwd_kernel[grid](gate, x, grad_out, dgate, dx, gate.stride(0), x.stride(0),
                                  M, D, shape_key=both_key(L))


def gated_projection_gate_flat_triton() -> None:
    """_sigmul_fwd, via bias_only_attention's sigmoid_gate_fused (one of its two callers)."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import sigmoid_gate_fused

    sigmoid_gate_fused(_x(), _x())


def gated_projection_bwd_gate_flat_triton() -> None:
    """_sigmul_bwd, via _SigmoidGate.backward."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import sigmoid_gate_fused

    gate, out = _x().requires_grad_(), _x().requires_grad_()
    sigmoid_gate_fused(gate, out).sum().backward()


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


# ── tm2 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_outproj_gemm_gate_triton() -> None:
    """fused_sigmoid_gate2_fwd_kernel, via triton_tm2."""
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    # _x(), not _rows(): TritonTM2Function reads token_key(length_of(x.shape)) before its own
    # rearrange, so a pre-flattened (M, d) gives it M = L*L -> clamped to 512.
    triton_tm2(_x(), _x(), _w(), _w())


def trimul_outproj_bwd_gate_recompute_triton() -> None:
    """fused_sigmoid_gate2_bwd_kernel, via TritonTM2Function.backward."""
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    # _x(): same reason as the forward -- ctx.original_shape is what the backward keys on.
    x, y = _x().requires_grad_(), _x().requires_grad_()
    triton_tm2(x, y, _w(), _w()).sum().backward()


def tm2_dual_kernel() -> None:
    """TM2DualKernel.kernel, via tm2_dual_from_scratch (weights in (N,K) form, as the bench).

    This driver builds its OWN x/W instead of ``_x()``/``_w()``: both of this kernel's extents are
    pinned by asserts the launcher writes down itself, so neither can carry the ragged tail. The
    two ``aligned_only`` calls below record that, quoting the asserts.
    """
    from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import tm2_dual_from_scratch

    d = aligned_only(
        "tm2.trimul_outproj_gemm_gate_sm90_cute.K",
        128,
        "tm2/cute/tm2_cute_kernel.py:66 `assert K % _TILE_K == 0, f\"K={K} must be divisible by "
        "TILE_K={_TILE_K}\"`, with `_TILE_K = 64` at tm2_cute_kernel.py:49 and "
        "`self.k_loop = K // _TILE_K` at :72 -- the K-stage count is an exact division, so the "
        "channel width K = d_pair must be a multiple of 64",
    )
    lseq = aligned_only(
        "tm2.trimul_outproj_gemm_gate_sm90_cute.M",
        64,
        "tm2/cute/tm2_cute_kernel.py:406 `assert M % tile_m == 0, f\"M={M} must be divisible by "
        "tile_m={tile_m}\"`; tile_m is chosen at :404 as the largest of (256,192,128,64) dividing "
        "M and falls back to 64, so M = L*L must be a multiple of 64. Perturbing L to 61 makes "
        "that assert the launcher's first statement to fail (AssertionError: M=3721 must be "
        "divisible by tile_m=64), which is the assert speaking, not a masked tail",
    )
    x1 = torch.randn(1, lseq, lseq, d, device=dev(), dtype=BF16)
    x2 = torch.randn(1, lseq, lseq, d, device=dev(), dtype=BF16)
    wg = (torch.randn(d, d, device=dev(), dtype=BF16) * (d**-0.5)).contiguous()
    wp = (torch.randn(d, d, device=dev(), dtype=BF16) * (d**-0.5)).contiguous()
    tm2_dual_from_scratch(x1, x2, wg, wp)


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
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidir_front_triton

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
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW_sig

    front_bwd_dW_sig(_bdll(), _bdll(), _bdll(), _bdll(), _bdll(2 * D), _x(),
                     _w(), _w(), _w(), _w())


def trimul_bwd_gate_transpose_packed_triton() -> None:
    """back_fused.py _dconcat5_kernel, via front_bwd_dW_glogit (the NEGATIVE-RESULT launcher)."""
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW_glogit

    front_bwd_dW_glogit(_bdll(), _bdll(), _bdll(4 * D), _x(),
                        _w(), _w(), _w(), _w(), _rows(), _w())


# ── trimul_inproj/cute: the two @triton.jit kernels living under cute/ ───────────────────────

def trimul_transpose_triton() -> None:
    """front_sm100.py _transpose_kernel, via _transpose_blld_to_bdll ((M,2D) -> (2D,M))."""
    from miniworld_engine.kernels.trimul_inproj.cute.front_sm100 import _transpose_blld_to_bdll

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
    grid = lambda meta: (triton.cdiv(h * M, meta["BLOCK_E"]),)  # noqa: E731
    _glu_bdll_kernel[grid](preact, lr, H=h, M=M, shape_key=token_key(L))


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
