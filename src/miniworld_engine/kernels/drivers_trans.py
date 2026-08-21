"""Drivers for the ``transition`` and ``triangle_multiplication`` kernels.

Every driver calls the launcher that the repo already uses for that kernel, so the argument
shapes are the ones the launcher documents, not invented ones:

* transition: ``benchmarks/runners/bench.py::bench_kernel_transition_b2b`` runs the op on
  ``x (1, L, L, D)`` with ``n = 4``, i.e. ``M = L*L`` rows of width ``K = D`` and
  ``ND = n*D``.  ``L = 64`` gives ``M = 4096``.  (``M % 128 == 0`` is what the sm90/sm100
  fused paths in ``TritonTransitionFusedFunction`` gate on, at fused.py:1105 and 1329 -- but
  these drivers call the Triton launchers directly and never reach that dispatch, so the row
  count is a free extent here.)
* ``transition_b2b_ktiled`` is only reached from ``TritonTransitionFusedFunction.forward``
  on the ``K > _B2B_MAX_K (=128)`` branch, so its driver uses ``K = K_LARGE (256), ND = 4*K``.
* triangle_multiplication: ``fused_triangle_multiplicative_update_dtv1`` flattens
  ``x (b, i, j, d)`` to ``(M = b*i*j, d)``; the input gate weight has ``2*d`` rows and the
  output gate weight ``d`` rows (both proofs are in the launcher comments).

Tile alignment
--------------
Every extent below goes through ``drivers.ragged()``, so ``MINIWORLD_SHAPE_MODE=ragged``
subtracts 3 from each and puts a partial tile at the end of every axis this family tiles:

* ``ROWS`` / ``TRIMUL_ROWS`` -- the M row count (BLOCK_M1 / BLOCK_E tails);
* ``K_SMALL`` / ``K_LARGE`` / ``TRIMUL_D`` -- the LN feature width, which is also the GEMM
  contraction extent (BLOCK_K / BLOCK_K_D tails), and it drags the weight rows with it;
* ``ND_SMALL = N_EXPAND * K_SMALL`` and ``2 * TRIMUL_D`` -- the expand/gate output width, the
  N axis of every expand GEMM and of the squeeze contraction (BLOCK_N / BLOCK_K_ND tails).

``N_EXPAND`` (=4) is NOT perturbed: it is the transition's expansion factor, part of the op
the bench defines (``n=4``), not a tile extent -- ND rides on ``K_SMALL`` instead.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import dev, driver_length, ragged, rows2d, vec

EPS = 1e-5
ROWS = ragged(driver_length(64) ** 2)  # M: L=64 pair rows (L*L); 4096 is a multiple of 128, ragged -> 4093
N_EXPAND = 4  # transition expansion factor n (bench: n=4) -- an op parameter, not a tile extent
K_SMALL = ragged(128)  # K: the AF3 transition d; ragged -> 125
K_LARGE = ragged(256)  # K for the ktiled kernel's K > _B2B_MAX_K(=128) reason to exist -> 253
ND_SMALL = N_EXPAND * K_SMALL  # expand/gate width for the K_SMALL paths: 512 -> 500


def _transition_operands(k: int = K_SMALL, n: int = N_EXPAND):
    """(x2, gamma, beta, wa, wb, ws) in nn.Linear layouts: wa/wb (ND, K), ws (K, ND)."""
    nd = n * k
    return (
        rows2d(ROWS, k), vec(k), vec(k),
        rows2d(nd, k), rows2d(nd, k), rows2d(k, nd),
    )


# --------------------------------------------------------------------------- transition


def transition_expand_swiglu_triton() -> None:
    """transition_fwd_kernel via TritonTransitionFunction.forward (kernels/transition/triton/main)."""
    from miniworld_engine.kernels.transition.triton.main import triton_transition

    x2, _, _, wa, wb, ws = _transition_operands()
    triton_transition(x2, wa, wb, ws, N_EXPAND)


def transition_fold_triton() -> None:
    """_fold_kernel via fold_swiglu_triton: Wa/Wb (ND, K), gamma/beta (K,)."""
    from miniworld_engine.kernels.transition.triton.fold import fold_swiglu_triton

    _, g, b, wa, wb, _ = _transition_operands()
    fold_swiglu_triton(wa, wb, g, b)


def transition_layernorm_expand_swiglu_triton() -> None:
    """_transition_expand_gate_kernel via transition_expand_gate (SAVE_XN=False, the default)."""
    from miniworld_engine.kernels.transition.triton.fused import transition_expand_gate

    x2, g, b, wa, wb, _ = _transition_operands()
    transition_expand_gate(x2, g, b, wa, wb, EPS)


def transition_fwd_b2b_triton() -> None:
    """_transition_b2b_kernel via transition_b2b at K_SMALL (<= _B2B_MAX_K), stats precomputed."""
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b

    x2, g, b, wa, wb, ws = _transition_operands(k=K_SMALL)
    transition_b2b(x2, g, b, wa, wb, ws, EPS, fuse_stats=False)


def transition_fwd_b2b_ktiled_triton() -> None:
    """_transition_b2b_ktiled_kernel via transition_b2b_ktiled at K_LARGE (its K > 128 path)."""
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b_ktiled

    x2, g, b, wa, wb, ws = _transition_operands(k=K_LARGE)
    transition_b2b_ktiled(x2, g, b, wa, wb, ws, EPS)


def transition_bwd_swiglu_recompute_triton() -> None:
    """_transition_expand_gatebwd_kernel with NORMALIZE/STORE_H/STACK_DAB = True/True/True --
    the Version A stacked launcher TritonTransitionFusedFunction.backward takes by default."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.triton.fused import (
        _transition_expand_gatebwd_stacked,
    )

    x2, g, b, wa, wb, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS)
    grad_expand = rows2d(ROWS, wa.shape[0])
    _transition_expand_gatebwd_stacked(x2, rstd, c1, g, b, wa, wb, grad_expand)


def layernorm_bwd_privatized_triton() -> None:
    """_transition_ln_bwd_kernel via _transition_ln_bwd. ``transition_lnbwd_cuda`` defaults to
    True and would route bf16/K<=512 to the hand-CUDA LN backward instead, so it is turned off
    for this launch; PRIVATIZE_DGDB keeps its default (True)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd

    x2, g, _, _, _, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS)
    previous = settings.current().transition_lnbwd_cuda
    settings.configure(transition_lnbwd_cuda=False)
    try:
        _transition_ln_bwd(torch.empty_like(x2).normal_(), x2, rstd, c1, g)
    finally:
        settings.configure(transition_lnbwd_cuda=previous)


def layernorm_fwd_recompute_foldstats_triton() -> None:
    """_xn_recompute_kernel via _xn_recompute (cute/fused.py backward's xn re-materialization)."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.cute.fused import _xn_recompute

    x2, g, b, _, _, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS)
    _xn_recompute(x2, rstd, c1, g, b)


def transition_bwd_epilogue_triton() -> None:
    """_grad_mul_kernel via _grad_mul_inplace: dA, dB, grad_expand all (M, ND) bf16 contiguous."""
    from miniworld_engine.kernels.transition.cute.gatebwd_sm100 import _grad_mul_inplace

    _grad_mul_inplace(rows2d(ROWS, ND_SMALL), rows2d(ROWS, ND_SMALL), rows2d(ROWS, ND_SMALL))


def transition_bwd_transpose_packed_triton() -> None:
    """_cdup_interleave_kernel via _cdup_interleave: grad_expand (M, ND) -> (M, 2*ND)."""
    from miniworld_engine.kernels.transition.cute.backward_gatebwd import _cdup_interleave

    _cdup_interleave(rows2d(ROWS, ND_SMALL))


def swiglu_gate_bwd_sm100() -> None:
    """SwiGLUGateBwdKernel.kernel via its host entry transition_expand_gatebwd_sm100.
    xn (M, K), wa/wb (ND, K), grad_expand (M, ND) bf16.

    This is a cutlass-DSL SM100 GEMM (``mma_tiler_mn=(128, 128)``, TMA store, operands marked
    ``assumed_align=16``; gatebwd_sm100.py:806-830), so its M/ND alignment needs are a plausible
    but UNPROVEN requirement -- nothing in the host entry asserts one. It is therefore left
    ragged rather than wrapped in ``aligned_only``: if the shape matters, the failure is the
    finding. It cannot be settled on the sm86 cards this sweep runs on, where the kernel does
    not launch at all for reasons that have nothing to do with the shape.
    """
    from miniworld_engine.kernels.transition.cute.gatebwd_sm100 import (
        transition_expand_gatebwd_sm100,
    )

    xn, _, _, wa, wb, _ = _transition_operands()
    transition_expand_gatebwd_sm100(xn, wa, wb, rows2d(ROWS, wa.shape[0]))


# ------------------------------------------------------- triangle_multiplication (dt-v1)

TRIMUL_ROWS = ragged(driver_length(128) ** 2)  # M = b*i*j for a (1, 128, 128, d) pair activation; ragged -> 16381
TRIMUL_D = ragged(128)  # d: the contraction width; the gate/proj weights have 2*d or d rows


def trimul_gemm_gate_saveact_triton() -> None:
    """_input_gated_gemm_kernel via _input_gemm_fwd. w_gate/w_proj have 2*D rows and the
    public entry passes TRANSPOSE_OUT=True; mask=None (APPLY_MASK=False)."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _input_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    _input_gemm_fwd(xn, rows2d(2 * TRIMUL_D, TRIMUL_D), rows2d(2 * TRIMUL_D, TRIMUL_D), None, True)


def trimul_outproj_gemm_gate_saveact_triton() -> None:
    """_output_gated_gemm_kernel via _output_gemm_fwd: x_normed/x_out (M, D), weights (D, D)."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _output_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    _output_gemm_fwd(xn, rows2d(TRIMUL_ROWS, TRIMUL_D),
                     rows2d(TRIMUL_D, TRIMUL_D), rows2d(TRIMUL_D, TRIMUL_D))


def gated_projection_bwd_gate_recompute_flat_triton() -> None:
    """_gated_gemm_bwd_elemwise_kernel via _elemwise_bwd_combined: grad/ab are the (2D, M)
    transposed input-path activations, sig_m is fp32 (the launcher's dtype contract)."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
        _elemwise_bwd_combined,
    )

    n = 2 * TRIMUL_D
    sig_m = torch.rand(n, TRIMUL_ROWS, device=dev(), dtype=torch.float32)
    _elemwise_bwd_combined(rows2d(n, TRIMUL_ROWS), rows2d(n, TRIMUL_ROWS), sig_m)


# ── the vendored transition_cuda extension ───────────────────────────────────────────────────
#
# `transition/cuda/transition_cuda_kernel.cu` holds three kernels the registry declares --
# `cast_kernel`, `swish_mul_kernel`, `transition_grad_kernel` -- and until now all three were
# reported `untested` with no driver, because nothing in the package imports the extension: it is
# built only by the standalone `transition/cuda/setup.py` as `transition_cuda_ext_v2`, and the
# `transition_cuda_b2b` setting refers to the *other* extensions loaded in
# `transition/cuda/__init__.py`. "No import path" is a reason a kernel cannot be reached, not a
# reason it cannot be tested: the sources are here, so load them the same way the sibling
# `__init__.py` loads its own, and drive them through the two functions the .cpp exports.
#
# The load is inside the driver, not at module scope, so a build failure is reported against these
# three kernels instead of breaking every other driver in this module at import.
#
# Shape contract, quoted from transition_cuda.cpp:45-68 -- x (M, N) contiguous fp32/bf16,
# expand_a/expand_b (nN, N), squeeze (N, nN), nN == n * N. Weights are nn.Linear-style (out, in).

_CUDA_N = 4  # the op's expansion factor, same as N_EXPAND


def _transition_cuda_ext():
    """JIT-build and return the vendored transition_cuda extension."""
    from pathlib import Path

    from torch.utils.cpp_extension import load

    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags

    ensure_cuda_home()
    d = Path(__file__).parent / "transition" / "cuda"
    return load(
        name="transition_cuda_ext_v2",
        sources=[str(d / "transition_cuda.cpp"), str(d / "transition_cuda_kernel.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "--use_fast_math",
                           *gencodes("80", "86", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"],
        extra_ldflags=["-lcublas"],
        verbose=False,
    )


def _transition_cuda_operands(dtype=torch.bfloat16):
    """(x, wa, wb, ws) at the driver's extents, matching the .cpp shape contract."""
    k = K_SMALL
    nk = _CUDA_N * k
    x = torch.randn(ROWS, k, device=dev(), dtype=dtype).contiguous()
    wa = torch.randn(nk, k, device=dev(), dtype=dtype).contiguous()
    wb = torch.randn(nk, k, device=dev(), dtype=dtype).contiguous()
    ws = torch.randn(k, nk, device=dev(), dtype=dtype).contiguous()
    return x, wa, wb, ws


def transition_cast_cuda() -> None:
    """`cast_kernel`, reached from the forward's fp32<->bf16 conversion around the cublas GEMM
    (kernel .cu:161 and :170). bf16 inputs are what make that path run at all."""
    ext = _transition_cuda_ext()
    ext.forward(*_transition_cuda_operands(torch.bfloat16), _CUDA_N)


def transition_swiglu_cuda() -> None:
    """`swish_mul_kernel` via `launch_swish_mul` (kernel .cu:325 in the forward)."""
    ext = _transition_cuda_ext()
    ext.forward(*_transition_cuda_operands(torch.bfloat16), _CUDA_N)


def transition_bwd_cuda() -> None:
    """`transition_grad_kernel` via `launch_transition_grad` (kernel .cu:411 in the backward).
    The .cpp additionally requires grad_output to be 2-D, contiguous, and to match x in shape
    and dtype (transition_cuda.cpp:108-115)."""
    ext = _transition_cuda_ext()
    x, wa, wb, ws = _transition_cuda_operands(torch.bfloat16)
    ext.backward(torch.randn_like(x).contiguous(), x, wa, wb, ws, _CUDA_N)
