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
  on the ``K > _B2B_MAX_K (=128)`` branch. Its driver uses the module-declared
  width and expansion ratio (standalone defaults: K=256, n=4).
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

``N_EXPAND`` (module-declared, default 4) is NOT perturbed: it is the transition's
expansion factor, not a tile extent -- ND rides on ``K_SMALL`` instead.

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

import os

import torch

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.drivers import (
    BF16,
    both_level_is_pair,
    dev,
    driver_heads,
    driver_length,
    driver_width,
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
ROWS = (8 * ragged(L_PAIR) if os.environ.get("MINIWORLD_DRIVER_SIDE") == "msa"
        else ragged(L_PAIR) ** 2 if IS_PAIR else ragged(L_PAIR))  # M: pair rows L*L, or atom rows A
#: What production records for this activation: ``both_key(rows_of(<pre-flatten shape>))``, which
#: is ROWS -- L*L on the pair side, A on the atom side. It used to be ``both_key(L_PAIR)``, and
#: that is what put a pair L=1024 (1,048,576 rows) and an atom A=1024 (1,024 rows) in one bucket.
#: Every launcher below is handed the already-flattened (M, K) matrix and takes the key from its
#: caller; passing it is what makes the sweep's unit (op, bucket) instead of (op, one bucket) N
#: times.
SHAPE_KEY = both_key(ROWS)
N_EXPAND = driver_heads(4)  # spare unit axis carries the actual module expansion ratio
# The plan restricts this path to K <= 128; drive its actual declared width.
# A fixed 128 silently collapsed template/MSA K=64 into the wrong cache key.
K_SMALL = ragged(driver_width(128))
#: The width the kernels DOWNSTREAM of the expansion see: `n * d_hidden`, which is what their
#: buckets carry. Their registry rows say `width=expand_nd` and the unit hands the expanded width
#: over here, so K follows from it rather than the other way round.
#:
#: This is why they were missing. The driver built `ND_SMALL = 4 * K_SMALL = 512` at every one of
#: the five widths the sweep declared, so five units wrote one bucket -- and `cases()` builds the
#: transition at d_hidden 128, 256 and 384, i.e. ND 512, 1024 and 1536. `dev audit --replay` asked
#: `transition_expand_swiglu_triton` and `transition_bwd_swiglu_recompute_triton` for 1024 and
#: 1536 and neither had ever been built. Same defect as triangle_attention's frozen head dim and
#: trimul's frozen per-side width: a kernel keys on a DERIVED width and the driver pinned it.
ND_DRIVEN = ragged(driver_width(4 * 128))
K_FROM_ND = max(16, ND_DRIVEN // N_EXPAND)
# The plan restricts ktiled probes to K > _B2B_MAX_K; honor the declared width exactly.
K_LARGE = ragged(driver_width(256))  # -> 253 at the default width
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
    if os.environ.get("MINIWORLD_DRIVER_SIDE") == "msa":
        return torch.randn(1, 8, n, k, device=dev(), dtype=BF16)
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
    """transition_fwd_kernel via TritonTransitionFunction.forward (kernels/transition/triton/main).

    At the EXPANDED width the unit declares (`width=expand_nd`), not at the frozen `K_SMALL`: this
    kernel folds ND into its key, so a driver pinned to one ND can only ever build one bucket.
    """
    from miniworld_engine.kernels.transition.triton.main import triton_transition

    _, _, _, wa, wb, ws = _transition_operands(k=K_FROM_ND)
    # The pre-flatten (1, L, L, K) activation, not the flat (M, K): the launcher reads
    # both_key(rows_of(x.shape)) before its own view(-1, d), so a flat x makes it bucket M.
    triton_transition(_pair_x(K_FROM_ND), wa, wb, ws, N_EXPAND)


def transition_layernorm_expand_swiglu_triton() -> None:
    """_transition_expand_gate_kernel via transition_expand_gate. SAVE_XN=0 ONLY.

    SAVE_XN is in the key, but every production caller passes `save_xn=False`:
    modules/transition/module.py:265 and :326, and kernels/transition/whole_op.py:79. The
    `save_xn=True` launch inside fused.py is reached only when that argument is already True, so
    nothing can turn it on. Driving it built a program the model never runs."""
    from miniworld_engine.kernels.transition.triton.fused import transition_expand_gate

    x2, g, b, wa, wb, _ = _transition_operands()
    transition_expand_gate(x2, g, b, wa, wb, EPS, shape_key=SHAPE_KEY)


def transition_fwd_b2b_triton() -> None:
    """_transition_b2b_kernel via transition_b2b at K_SMALL (<= _B2B_MAX_K), stats precomputed.

    BOTH values of HAS_LN. The no-LayerNorm form -- the bare SwiGLU FFN an adaLN-Zero block
    wants, reached through `triton_swiglu_ffn` -- is a separate compiled kernel and a separate
    cache bucket (HAS_LN is in the autotune key), so a config tuned with the LayerNorm in place
    says nothing about it. One driver and not a second registry row, because both are
    `_transition_b2b_kernel` and a config ladder belongs to a kernel.
    """
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b

    x2, g, b, wa, wb, ws = _transition_operands(k=K_SMALL)
    # ONE probe where two used to stand, because there is no longer an ADD_RESIDUAL to drive
    # both sides of. The residual follows from HAS_LN (see the note above `_transition_b2b_kernel`):
    # HAS_LN=1 is the Transition op and always adds it, HAS_LN=0 is the bare SwiGLU FFN and never
    # does. Both HAS_LN values are still driven, one probe each, and HAS_LN IS in the key.
    #
    # FUSE_STATS and SAVE_XN stay at 0: `settings.transition_fuse_stats` defaults False
    # (settings.py:201) and its only setter is `builder.SWITCHES`, which the per-op `build all`
    # never reads; `save_xn=True` has no caller (see transition_layernorm_expand_swiglu above).
    transition_b2b(x2, g, b, wa, wb, ws, EPS, fuse_stats=False, shape_key=SHAPE_KEY)
    empty = x2.new_empty(0)
    transition_b2b(x2, empty, empty, wa, wb, ws, 0.0, has_ln=False, shape_key=SHAPE_KEY)


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
        _transition_expand_gatebwd_savedxn,
        _transition_expand_gatebwd_stacked,
    )

    # At the expanded width the unit declares, like the forward: this kernel's bucket carries the
    # expand width, so `_transition_operands()` at the frozen K_SMALL built ND 512 for every rung.
    x2, g, b, wa, wb, _ = _transition_operands(k=K_FROM_ND)
    rstd, c1 = stats_triton(x2, EPS, shape_key=SHAPE_KEY)
    grad_expand = rows2d(ROWS, wa.shape[0])
    _transition_expand_gatebwd_stacked(x2, rstd, c1, g, b, wa, wb, grad_expand,
                                       shape_key=SHAPE_KEY)          # NORMALIZE=1, STORE_H=1
    # The Version B (saved-xn) backward is the SAME kernel with NORMALIZE=False -- it reads the
    # already-normalized xn as the GEMM operand instead of recomputing it -- and it varies STORE_H
    # on top of that (fused.py:1052-1077, 1093-1117). Both are keyed; neither was driven, so the
    # whole saved-xn half of this kernel ran on the heuristic subset.
    _transition_expand_gatebwd_savedxn(x2, wa, wb, grad_expand, store_h=True,
                                       shape_key=SHAPE_KEY)          # NORMALIZE=0, STORE_H=1
    # STORE_H=0 is NOT driven: fused.py:1052 defaults `store_h=True` and its caller takes that
    # default (fused.py:1573, implicitly). NORMALIZE=0 is,
    # because the saved-xn backward really runs it -- replay asked for (NORMALIZE=0, STORE_H=1)
    # six times.


def layernorm_bwd_foldstats_triton() -> None:
    """_transition_ln_bwd_kernel via _transition_ln_bwd. ``transition_lnbwd_cuda`` defaults to
    True and would route bf16/K<=512 to the hand-CUDA LN backward instead, so it is turned off
    for this launch; PRIVATIZE_DGDB is driven at BOTH values."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd

    x2, g, _, _, _, _ = _transition_operands()
    rstd, c1 = stats_triton(x2, EPS, shape_key=SHAPE_KEY)
    # PRIVATIZE_DGDB=1 only: `settings.transition_lnbwd_privatize` defaults True and nothing in
    # production sets it, so =0 is reachable only from a build-harness pin. Both pins are applied
    # only while the settings still declare them (v2.2.0 retires the legacy transition_* switches).
    current = settings.current()
    pins = {name: value for name, value in (("transition_lnbwd_cuda", False),
                                            ("transition_lnbwd_privatize", True))
            if hasattr(current, name)}
    previous = {name: getattr(current, name) for name in pins}
    if pins:
        settings.configure(**pins)
    try:
        _transition_ln_bwd(torch.empty_like(x2).normal_(), x2, rstd, c1, g,
                           shape_key=SHAPE_KEY)
    finally:
        # Restored: a driver that mutates global settings past its own return changes what
        # every LATER driver in the same build process tunes.
        if previous:
            settings.configure(**previous)


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

    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    # `parents[1]`, not `parent`: this module used to be `kernels/drivers_trans.py`, where
    # `.parent` was the kernels package. It is now `kernels/drivers/transition.py`, one level
    # deeper, so `.parent` became `kernels/drivers/` and the sources resolved to
    # `kernels/drivers/transition/cuda/transition_cuda.cpp` -- a path that has never existed. It
    # imported fine and raised FileNotFoundError only when the driver actually ran.
    d = Path(__file__).parents[1] / "transition" / "cuda"
    return load_extension(
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


def transition_fwd_b2b_sm90_cuda():
    from miniworld_engine.kernels.drivers import hopper
    return hopper.transition_fwd_b2b_sm90_cuda()


def transition_expand_gate_sm90_cuda():
    from miniworld_engine.kernels.drivers import hopper
    return hopper.transition_expand_gate_sm90_cuda()


def transition_bwd_gate_sm90_cuda():
    from miniworld_engine.kernels.drivers import hopper
    return hopper.transition_bwd_gate_sm90_cuda()


def transition_squeeze_residual_triton():
    """Same squeeze/residual launcher and row key used by the split Transition."""
    from miniworld_engine.kernels.transition.triton.residual import squeeze_residual
    h = rows2d(ROWS, N_EXPAND * K_SMALL)
    w = rows2d(K_SMALL, N_EXPAND * K_SMALL)
    r = rows2d(ROWS, K_SMALL)
    squeeze_residual(h, w, r, SHAPE_KEY)


def _fused_transition(backward):
    from miniworld_engine.autotune import policy
    from miniworld_engine.kernels.transition.cuda.fused_sm90a import _fwd_launch, _bwd_launch
    length = driver_length(384)
    rows = length * length if both_level_is_pair(length) else length
    x = torch.randn(rows, 128, device="cuda", dtype=BF16)
    gamma, beta = torch.ones(128, device="cuda"), torch.zeros(128, device="cuda")
    wa, wb = [torch.randn(512, 128, device="cuda", dtype=BF16) / 128 ** 0.5 for _ in range(2)]
    ws = torch.randn(128, 512, device="cuda", dtype=BF16) / 512 ** 0.5
    if backward:
        _, xn, rstd, c1 = _fwd_launch(x, gamma, beta, wa, wb, ws.T.contiguous(), 1e-5, True)
        return _bwd_launch(torch.randn_like(x), x, xn, rstd, c1, gamma, wa, wb, ws)
    for save in ((False,) if policy.mode() == "eval" else (True,) if policy.mode() == "train" else (False, True)):
        _fwd_launch(x, gamma, beta, wa, wb, ws.T.contiguous(), 1e-5, save)


def transition_fwd_residual_sm90_cuda():
    return _fused_transition(False)


def transition_bwd_residual_sm90_cuda():
    return _fused_transition(True)
