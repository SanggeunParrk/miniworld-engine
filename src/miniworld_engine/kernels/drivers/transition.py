"""Drivers for the ``transition`` family.

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

``_pair_x`` is the one exception, and only in ragged mode: it must stay square for
``length_of`` to read L off it, so ``ragged()`` is applied to L (61) rather than to L*L. The M
tail is still partial -- 61*61 = 3721, and 3721 % 16 == 9 -- and in the default aligned mode it
is exactly ``ROWS``.

Shape key
---------
``shape_key`` is in nearly every one of these kernels' ``key=[...]``, and it is L -- never the
flattened row count M = L*L. Every launcher in this family is handed the already-flattened
(M, K) matrix, so it cannot read L off a tensor: ``autotune.shape_key.length_of`` says outright
that "an inner launcher that only receives the flattened (M, D) matrix CANNOT call this; its
caller must compute the key and pass it down". These drivers ARE that caller, so each one passes
``shape_key=SHAPE_KEY`` (transition) or ``seq_len=TRIMUL_L`` (trimul). Left unpassed, the
transition launchers fall back to ``both_key(M)`` = the clamped TOP bucket (8192 at any L >= 91)
and the trimul ones to ``token_key(0)`` = the clamped BOTTOM bucket (128), so every driver length
records the same bucket and a per-bucket sweep tunes one bucket over and over.

The same applies to the ``stats_triton`` LN-stats helper three of these drivers call to build
``rstd``/``c1``: it fires ``layernorm_stats_triton`` on the SAME activation at the same L, and
what a capture records is every op that fired, not just the driver's own. Left unpassed it
recorded ``shape_key=8192`` for that op at every driver length -- so it is passed here too,
exactly as ``transition_b2b`` / ``transition_expand_gate`` already forward it internally.
"""
from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.drivers import (
    BF16,
    both_level_is_pair,
    dev,
    driver_length,
    ragged,
    rows2d,
    vec,
)

EPS = 1e-5
L_PAIR = driver_length(64)  # L: the pair side length; the activation is (1, L, L, K) before flattening
# M = (ragged L)**2, not ragged(L**2): `_pair_x` below must stay square for `length_of`, so it
# flattens to ragged(L)**2 rows, and ROWS is what the flat drivers here build. Deriving them
# differently made the two disagree in ragged mode (4093 vs 3721). Aligned is 64*64 = 4096
# either way; ragged is 61*61 = 3721, still a partial tile in all five config sets (% 16 == 9).
#: A level=both kernel meets 512 and below as a PAIR activation (1, L, L, D) flattening to
#: M = L*L, and 1024 and above as an ATOM activation (1, A, D) flattening to M = A -- see
#: ``drivers.both_level_is_pair``. Squaring at every bucket builds shapes production never
#: presents: M = 67,108,864 at L=8192 where the model hands over 8,192. That is what OOM'd the
#: atom probes here and what left transition_fwd_b2b_ktiled at L=4096 measuring 16.7M rows at
#: ~420 s per config. The token-level kernels in this file are never driven above 512, so the
#: same constant serves them unchanged.
IS_PAIR = both_level_is_pair(L_PAIR)
ROWS = ragged(L_PAIR) ** 2 if IS_PAIR else ragged(L_PAIR)  # M: pair rows L*L, or atom rows A
#: What production records for this activation: ``both_key(rows_of(<pre-flatten shape>))``, which
#: is ROWS -- L*L on the pair side, A on the atom side. It used to be ``both_key(L_PAIR)``, and
#: that is what put a pair L=1024 (1,048,576 rows) and an atom A=1024 (1,024 rows) in one bucket.
#: Every launcher below is handed the already-flattened (M, K) matrix and takes the key from its
#: caller; passing it is what makes the sweep's unit (op, bucket) instead of (op, one bucket) N
#: times.
SHAPE_KEY = both_key(ROWS)
N_EXPAND = 4  # transition expansion factor n (bench: n=4) -- an op parameter, not a tile extent
K_SMALL = ragged(128)  # K: the AF3 transition d; ragged -> 125
K_LARGE = ragged(256)  # K for the ktiled kernel's K > _B2B_MAX_K(=128) reason to exist -> 253
ND_SMALL = N_EXPAND * K_SMALL  # expand/gate width for the K_SMALL paths: 512 -> 500


def _pair_x(k: int = K_SMALL) -> torch.Tensor:
    """x as the PRE-FLATTEN pair activation (1, L, L, K), for the one launcher that takes it.

    ``TritonTransitionFunction.forward`` does its own ``view(-1, d)`` and reads the shape key off
    ``x.shape`` before that (main.py:129), so it is the only entry here that can be given L at all
    -- it takes no ``shape_key=``. ``ragged()`` is applied to L rather than to L*L so the tensor
    stays square: the M tail is still partial (61*61 = 3721, 3721 % 16 == 9, so every one of the
    five config sets sees it), and in the default aligned mode M = L*L = ROWS exactly.

    On the ATOM side there is no pair to build -- production hands (1, A, D) -- so it returns the
    3-D activation instead, which flattens to M = A = ROWS. ``length_of`` reads shape[-2] either
    way, so both layouts record the same shape_key.
    """
    n = ragged(L_PAIR)
    if not IS_PAIR:
        return torch.randn(1, n, k, device=dev(), dtype=BF16)
    return torch.randn(1, n, n, k, device=dev(), dtype=BF16)


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

    _, _, _, wa, wb, ws = _transition_operands()
    # The pre-flatten (1, L, L, K) activation, not the flat (M, K): the launcher reads
    # both_key(rows_of(x.shape)) before its own view(-1, d), so a flat x makes it bucket M.
    triton_transition(_pair_x(), wa, wb, ws, N_EXPAND)


def transition_fold_triton() -> None:
    """_fold_kernel via fold_swiglu_triton: Wa/Wb (ND, K), gamma/beta (K,)."""
    from miniworld_engine.kernels.transition.triton.fold import fold_swiglu_triton

    _, g, b, wa, wb, _ = _transition_operands()
    fold_swiglu_triton(wa, wb, g, b)


def transition_layernorm_expand_swiglu_triton() -> None:
    """_transition_expand_gate_kernel via transition_expand_gate (SAVE_XN=False, the default)."""
    from miniworld_engine.kernels.transition.triton.fused import transition_expand_gate

    x2, g, b, wa, wb, _ = _transition_operands()
    transition_expand_gate(x2, g, b, wa, wb, EPS, shape_key=SHAPE_KEY)


def transition_fwd_b2b_triton() -> None:
    """_transition_b2b_kernel via transition_b2b at K_SMALL (<= _B2B_MAX_K), stats precomputed."""
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b

    x2, g, b, wa, wb, ws = _transition_operands(k=K_SMALL)
    transition_b2b(x2, g, b, wa, wb, ws, EPS, fuse_stats=False, shape_key=SHAPE_KEY)


def transition_fwd_b2b_ktiled_triton() -> None:
    """_transition_b2b_ktiled_kernel via transition_b2b_ktiled at K_LARGE (its K > 128 path)."""
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b_ktiled

    x2, g, b, wa, wb, ws = _transition_operands(k=K_LARGE)
    transition_b2b_ktiled(x2, g, b, wa, wb, ws, EPS, shape_key=SHAPE_KEY)


def transition_bwd_swiglu_recompute_triton() -> None:
    """_transition_expand_gatebwd_kernel with NORMALIZE/STORE_H/STACK_DAB = True/True/True --
    the Version A stacked launcher TritonTransitionFusedFunction.backward takes by default."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.triton.fused import (
        _transition_expand_gatebwd_stacked,
    )

    x2, g, b, wa, wb, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS, shape_key=SHAPE_KEY)
    grad_expand = rows2d(ROWS, wa.shape[0])
    _transition_expand_gatebwd_stacked(x2, rstd, c1, g, b, wa, wb, grad_expand,
                                       shape_key=SHAPE_KEY)


def layernorm_bwd_privatized_triton() -> None:
    """_transition_ln_bwd_kernel via _transition_ln_bwd. ``transition_lnbwd_cuda`` defaults to
    True and would route bf16/K<=512 to the hand-CUDA LN backward instead, so it is turned off
    for this launch; PRIVATIZE_DGDB keeps its default (True)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd

    x2, g, _, _, _, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS, shape_key=SHAPE_KEY)
    previous = settings.current().transition_lnbwd_cuda
    settings.configure(transition_lnbwd_cuda=False)
    try:
        _transition_ln_bwd(torch.empty_like(x2).normal_(), x2, rstd, c1, g,
                           shape_key=SHAPE_KEY)
    finally:
        settings.configure(transition_lnbwd_cuda=previous)


def layernorm_fwd_recompute_foldstats_triton() -> None:
    """_xn_recompute_kernel via _xn_recompute (cute/fused.py backward's xn re-materialization)."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.cute.fused import _xn_recompute

    x2, g, b, _, _, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS, shape_key=SHAPE_KEY)
    _xn_recompute(x2, rstd, c1, g, b, shape_key=SHAPE_KEY)


def transition_bwd_epilogue_triton() -> None:
    """_grad_mul_kernel via _grad_mul_inplace: dA, dB, grad_expand all (M, ND) bf16 contiguous."""
    from miniworld_engine.kernels.transition.cute.gatebwd_sm100 import _grad_mul_inplace

    _grad_mul_inplace(rows2d(ROWS, ND_SMALL), rows2d(ROWS, ND_SMALL), rows2d(ROWS, ND_SMALL),
                      shape_key=SHAPE_KEY)


def transition_bwd_transpose_packed_triton() -> None:
    """_cdup_interleave_kernel via _cdup_interleave: grad_expand (M, ND) -> (M, 2*ND)."""
    from miniworld_engine.kernels.transition.cute.backward_gatebwd import (
        _cdup_interleave,
    )

    _cdup_interleave(rows2d(ROWS, ND_SMALL), shape_key=SHAPE_KEY)


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
    transition_expand_gatebwd_sm100(xn, wa, wb, rows2d(ROWS, wa.shape[0]), shape_key=SHAPE_KEY)


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
    # `parents[1]`, not `parent`: this module used to be `kernels/drivers_trans.py`, where
    # `.parent` was the kernels package. It is now `kernels/drivers/transition.py`, one level
    # deeper, so `.parent` became `kernels/drivers/` and the sources resolved to
    # `kernels/drivers/transition/cuda/transition_cuda.cpp` -- a path that has never existed. It
    # imported fine and raised FileNotFoundError only when the driver actually ran.
    d = Path(__file__).parents[1] / "transition" / "cuda"
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
    (kernel .cu:161 and :170). bf16 inputs are what make that path run at all.

    ``torch.bfloat16`` outright, NOT ``BF16``/``MINIWORLD_DRIVER_DTYPE``: ``cast_tensor_to_dtype``
    returns its argument untouched when it is already the destination dtype (.cu:146), and the two
    instantiations it can reach are bf16->float and float->bf16 (.cu:157-173). An all-fp32 call
    therefore launches no cast_kernel at all, so driving this one in fp32 would report a kernel
    that never ran. The registry's ``bf16|fp32`` is about the PAIR the cast bridges, not about a
    dtype this kernel can be driven at on its own."""
    ext = _transition_cuda_ext()
    ext.forward(*_transition_cuda_operands(torch.bfloat16), _CUDA_N)


def transition_swiglu_cuda() -> None:
    """`swish_mul_kernel` via `launch_swish_mul` (kernel .cu:325 in the forward).

    ``BF16`` is the activation dtype, so ``MINIWORLD_DRIVER_DTYPE=fp32`` reaches
    ``swish_mul_kernel<float>``: the .cpp accepts float32 or bfloat16 (transition_cuda.cpp:29-37)
    and the .cu dispatches ``transition_forward<float>`` for an fp32 x (.cu:499)."""
    ext = _transition_cuda_ext()
    ext.forward(*_transition_cuda_operands(BF16), _CUDA_N)


def transition_bwd_cuda() -> None:
    """`transition_grad_kernel` via `launch_transition_grad` (kernel .cu:411 in the backward).
    The .cpp additionally requires grad_output to be 2-D, contiguous, and to match x in shape
    and dtype (transition_cuda.cpp:108-115). fp32 reaches ``transition_grad_kernel<float>`` by
    the same .cu:538 dispatch as the forward."""
    ext = _transition_cuda_ext()
    x, wa, wb, ws = _transition_cuda_operands(BF16)
    ext.backward(torch.randn_like(x).contiguous(), x, wa, wb, ws, _CUDA_N)
