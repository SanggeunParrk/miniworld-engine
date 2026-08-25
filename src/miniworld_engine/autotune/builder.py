"""Dedicated autotune-cache builder: drives the PRODUCTION modules, not the benchmarks.

The cache used to be built by running ``benchmarks/runners/bench.py`` with capture switched on.
That ties what gets cached to what the benchmark suite happens to measure, and the two are not the
same thing: the suite covers eight kernel targets through its own wrappers, so of the 86 ops that
register themselves with the cache, only 23 ever fired during a build. The other 63 were not
"missing" in any way a build could notice -- no code path reached them, so there was nothing to
capture, and the gap only surfaced when production took one of those paths and fell back to the
full autotune grid (a multi-minute stall that reads as a hang).

Chasing that one dispatch switch at a time -- gate epilogue, LN+proj concat, dropout flag, the
hand-CUDA b2b forward -- finds one hole per incident and never converges. So this builds the cache
from the modules the model actually runs, at the settings it actually runs them with, and the
kernels each module dispatches to are captured by construction. ``miniworld-engine coverage`` then
checks the result against the registry, so a hole is a failed build rather than a slow forward six
weeks later.

Dispatch switches that pick between equally-correct implementations are still swept explicitly:
whichever side a given shape does not take contributes no entries, yet production reaches it at
other shapes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path
from queue import Empty, Queue

import torch

from miniworld_engine import build as build_matrix

BF16 = torch.bfloat16


@dataclasses.dataclass(frozen=True)
class Case:
    """One module to exercise, and how to feed it.

    Dimensions are declared with the module's OWN parameter names -- d_pair, d_single, d_hidden,
    n_head -- not as a single anonymous "width". They are not interchangeable: a kernel's cache
    bucket is built from the constexprs it was launched with, so d_hidden and n_head land in
    different buckets than d_pair even when the numbers coincide, and sweeping one scalar leaves
    the others at their defaults forever.

    dtype is an axis for the same reason: the cache key is (op, dtype, bucket) literally, so a
    bf16-only build leaves every fp32 bucket empty no matter how many shapes it visits.
    """

    name: str
    #: build the module: factory(width, p_drop) -- p_drop is ignored by modules without dropout
    factory: Callable[..., torch.nn.Module]   # factory(width, p_drop, impl)
    #: build its forward arguments for a given (batch, length, width)
    inputs: Callable[..., tuple]
    # The ladders below are the DEFAULT sweep, and they decide what the cache covers: an entry
    # exists only for a (dtype, bucket) some run actually produced, and the buckets key on the
    # shapes. A narrow sweep therefore does not merely cache less -- it leaves production shapes
    # with no entry at all, which is a full-grid fallback mid-forward (minutes) rather than a
    # slightly worse config. The warnings that prompted this widening named real buckets a build
    # had never visited: layernorm_linear_fwd K=128,N=520 and trimul_bidir_front GM=2,H2=256,K=128.
    #: constructor dimensions to sweep, each entry the module's own kwargs
    dims: tuple[dict, ...] = ()
    #: dtypes to build under -- part of the cache key, so each one is a separate set of entries
    dtypes: tuple[torch.dtype, ...] = (BF16,)
    #: lengths worth sweeping -- spans the trunk sizes the model runs, not just two points
    lengths: tuple[int, ...] = (256, 384, 512, 768, 1024)
    #: run a backward too -- training-only kernels are a large share of the registry
    train: bool = True
    #: input dtype. Per case, because the modules differ: the fused bf16 kernels want bf16, while
    #: ConditionedTransition keeps fp32 parameters (the bench runs it at precision=32) and a bf16
    #: input fails with "expected mat1 and mat2 to have the same dtype".
    dtype: torch.dtype = BF16
    #: implementations to build under. The registry holds several implementations of the same
    #: kernel -- tm1/tm2, *_miniworld_*, *_perf_*, adaln_fused3_* -- and a run only ever fires the
    #: one its dispatcher picks, so the rest can never be captured without asking for them by name.
    impls: tuple[str, ...] = ("miniworld",)
    #: dispatch switches this module consults. Whichever side a given shape does not take
    #: contributes no entries, yet production reaches it at other shapes -- so a build sweeps both.
    #: Dropping this axis when the builder replaced the bench-driven capture is what took coverage
    #: from 23 ops down to 20.
    switches: tuple[str, ...] = ()
    #: compute dtypes to sweep as a SEPARATE axis from ``dtypes``, for modules that take the
    #: compute dtype per CALL rather than at construction. Only AugmentedAttentionPairBias does:
    #: its attention core is cast independently of the module's own dtype, so (module fp32, core
    #: bf16) is a real production configuration that (module bf16) never visits -- different cache
    #: bucket, different tuned config. Empty means the module computes in its own dtype and the
    #: axis collapses, which is every other case.
    compute_dtypes: tuple[torch.dtype, ...] = ()


#: switch -> (values, modes it applies to). Swept independently, not as a cross product: each
#: switch selects among its own kernels, so one run per side covers them.
SWITCHES: dict[str, tuple[tuple, tuple[str, ...]]] = {
    "gate_backend": (("fused", "split"), ("eval", "train")),
    "infer_concat": ((True, False), ("eval",)),
    "p_drop": ((0.25,), ("train",)),
    # settings.py states the rule this table implements: "A build must capture BOTH sides of every
    # device-calibrated switch. The card picks one side for the shapes being swept, so the other
    # side's kernels never fire and never get captured -- yet they still run in production at other
    # shapes, then with no cached configs at all." Only three pins were being swept; every other
    # backend-selection setting below is the same kind of switch and was leaving its far side
    # uncaptured. Each entry names the OFF-DEFAULT value only -- the default side is already
    # covered by the unpinned unit, so pinning both would double the build for nothing.
    "ln_partial_reduction": ((True, False), ("eval", "train")),
    "ln_bwd_path": (("persistent", "partial", "atomic"), ("train",)),
    "ln_out_bwd_path": (("split", "fused"), ("train",)),
    "transition_force_split": ((True,), ("eval", "train")),
    "transition_cuda_b2b": ((False,), ("eval",)),
    "transition_fuse_stats": ((True,), ("eval", "train")),
    "transition_savedxn_split_bwd": ((True,), ("train",)),
    "transition_dab_lnbwd": ((True,), ("train",)),
    "transition_lnbwd_privatize": ((False,), ("train",)),
    "trimul_impl": (("triton", "cute"), ("eval", "train")),
}

#: switch name -> the ``settings`` field it pins, and how to parse the CLI string back to a value.
#: A table rather than an if/elif chain in the child: every switch added to SWITCHES must be
#: pinnable, and a chain lets one be added without its pin -- which silently produces a duplicate
#: of the default unit instead of the other side of the switch.
SWITCH_SETTINGS: dict[str, tuple[str, Callable[[str], object]]] = {
    "gate_backend": ("pin_gate_backend", str),
    "infer_concat": ("pin_infer_concat", lambda v: v == "True"),
    "ln_partial_reduction": ("pin_ln_partial_reduction", lambda v: v == "True"),
    "ln_bwd_path": ("layernorm_bwd_path", str),
    "ln_out_bwd_path": ("layernorm_out_bwd_path", str),
    "transition_force_split": ("transition_force_split", lambda v: v == "True"),
    "transition_cuda_b2b": ("transition_cuda_b2b", lambda v: v == "True"),
    "transition_fuse_stats": ("transition_fuse_stats", lambda v: v == "True"),
    "transition_savedxn_split_bwd": ("transition_savedxn_split_bwd", lambda v: v == "True"),
    "transition_dab_lnbwd": ("transition_dab_lnbwd", lambda v: v == "True"),
    "transition_lnbwd_privatize": ("transition_lnbwd_privatize", lambda v: v == "True"),
    "trimul_impl": ("trimul_impl", str),
}


def device_sm() -> str | None:
    """sm tag of the card this build will run on, or None when there is no CUDA.

    None means "do not filter": the caller is enumerating work off-GPU (a plan dump, a test), and
    dropping architecture-specific units there would understate the build rather than protect it.
    """
    if not torch.cuda.is_available():
        return None
    return build_matrix.sm_tag(torch.cuda.get_device_capability())


def skipped_units(selected: list[Case], sm: str | None) -> list[tuple[str, str]]:
    """``(label, reason)`` for every unit this card is not allowed to build. Reported, never
    silently dropped: a bucket no build visits is a full-grid fallback in production."""
    if sm is None:
        return []
    out = {}
    for case in selected:
        for impl in case.impls:
            for dtype in case.dtypes:
                dt = str(dtype).replace("torch.", "")
                ok, why = build_matrix.decide(sm, case.name, impl, dt)
                if not ok:
                    out[f"{case.name}[{impl}/{dt}]"] = why
    return sorted(out.items())


def _pair(batch: int, length: int, d: int, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(batch, length, length, d, device="cuda", dtype=dtype)


def _single(batch: int, length: int, d: int, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(batch, length, d, device="cuda", dtype=dtype)


def _mask(batch: int, length: int) -> torch.Tensor:
    return torch.ones(batch, length, dtype=torch.bool, device="cuda")


class _KernelModule(torch.nn.Module):
    """Wraps a kernel callable so a Case can drive it exactly like a production module.

    Some registered ops have no ``nn.Module`` that dispatches to them -- the tm1/tm2 Triton
    implementations and the ``*_miniworld_*`` / ``*_perf_*`` triangle-attention variants are
    alternative implementations kept for A/B measurement, reachable only through the public kernel
    API in ``miniworld_engine.kernels``. They register with the cache all the same, so a build that
    only drives modules can never produce their entries no matter how wide the module sweep gets.

    A wrapper rather than a second Case KIND: ``run_case`` already does the right thing with an
    ``nn.Module`` (train/eval, requires_grad on the inputs, backward on the summed output), and
    duplicating that for callables would be two code paths that have to stay in step.
    """

    def __init__(self, fn: Callable, weights: dict[str, torch.Tensor],
                 tail: tuple = ()) -> None:
        super().__init__()
        self._fn = fn
        # Trailing NON-tensor arguments (an eps, a flag) that the kernel takes AFTER its weights.
        # They cannot live in the Case's input tuple: forward() appends the weights after the
        # inputs, so an eps placed there lands in the weight slot and the weights shift one right.
        # That is how `layernorm_lowreg` was calling triton_layernorm_lowreg(x, 1e-5, w, b) against
        # a (x, weight, bias, eps) signature -- eps arrived as a tensor and the kernel failed to
        # compile. Keeping them out of the inputs is also right on its own: an eps is not something
        # run_case should be attaching requires_grad to.
        self._tail = tail
        # registered so .cuda()/.to(dtype) reach them and so autograd sees leaves for the backward
        for name, w in weights.items():
            self.register_parameter(name, torch.nn.Parameter(w))

    def forward(self, *args):
        out = self._fn(*args, *[p for _, p in self.named_parameters()], *self._tail)
        return out[0] if isinstance(out, tuple) else out


def _kernel_case(fn_path: tuple[str, str], weights: Callable[[dict, torch.dtype], dict],
                 tail: tuple = ()):
    """Case factory for a kernel driven through its public API. Imports lazily, like the modules."""
    def make(dims, p, impl, dt):
        import importlib

        mod, attr = fn_path
        fn = getattr(importlib.import_module(mod), attr)
        fn = fn.apply if hasattr(fn, "apply") else fn
        return _KernelModule(fn, weights(dims, dt), tail).cuda().to(dt)
    return make


def _w(*shapes: tuple[int | str, ...]):
    """Weight-dict builder for _kernel_case: positional names keep the kernel's argument order.

    An extent is an int, or a str naming one of the case's ``dims`` -- ``_w(("d", "d"))`` is a
    (d, d) weight whose d comes from the dims dict the case is instantiated with. The annotation
    said ``tuple[int, ...]``, which every caller violates and which produced 27 of this repo's
    type findings from one wrong word.
    """
    def build(dims: dict, dt: torch.dtype) -> dict:
        return {f"w{i}": torch.randn(*[dims.get(s, s) if isinstance(s, str) else s for s in shape],
                                     device="cuda", dtype=dt)
                for i, shape in enumerate(shapes)}
    return build


def _swa_params(length: int, dims: dict, dtype: torch.dtype) -> tuple:
    """``(cos, sin, seqused, cu_seqlens, max_seqlen, valid)`` for SWA3DRoPEAttention.forward.

    Its forward takes the rotary tables and the varlen packing as a prebuilt tuple (production
    computes them once per batch and reuses them across blocks), so the case has to supply the
    same tuple rather than a plain tensor. One sequence of ``length`` tokens, all valid.
    """
    head_dim = dims["d_model"] // dims["n_heads"]
    half = head_dim // 2
    cos = torch.ones(1, length, 1, half, device="cuda", dtype=dtype)
    sin = torch.zeros(1, length, 1, half, device="cuda", dtype=dtype)
    seqused = torch.tensor([length], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    valid = torch.ones(1, length, dtype=torch.bool, device="cuda")
    return (cos, sin, seqused, cu_seqlens, length, valid)


#: Every case name, declared. `cases()` cannot answer "is this a real case name?" cheaply: it
#: constructs the module classes, which imports every kernel, which is 2+ minutes -- so
#: `miniworld-engine build <typo>` spent all of it before saying "unknown case". Validating
#: against this list happens before the first import. `test_case_names_are_declared` asserts the
#: two agree, so it cannot drift into a second source of truth.
CASE_NAMES: tuple[str, ...] = (
    "transition",
    "triangle_multiplication",
    "triangle_multiplication_bidirectional",
    "triangle_attention_bidirectional",
    "triangle_attention_heads",
    "attention_pair_bias",
    "augmented_attention",
    "adaptive_layernorm",
    "conditioned_transition",
    "msa_pair_weighted_averaging",
    "outer_product_mean",
    "pairformer_block",
    "triangle_pair_attention",
    "tm1",
    "tm2",
    "gated_projection",
    "layernorm_linear_pair_bias",
    "swa_atom_attention",
    "layernorm_lowreg",
    "layernorm_transpose",
    "layernorm_linear_stats",
)


def cases() -> list[Case]:
    """Every production module worth driving, deferred so importing this module needs no GPU.

    Dimensions come from each module's declared defaults and the values the model actually runs --
    d_pair 128, d_single 384, tri-attention d_hidden 32 / n_head 4, OPM d_hidden 32 -- plus wider
    trunks. A ladder of round numbers (128/256/512) misses 384 and 32 entirely, and a bucket no
    build visits is a bucket production falls back to the full grid on.
    """
    from miniworld_engine.modules import (
        AdaptiveLayerNorm,
        AttentionPairBias,
        AugmentedAttentionPairBias,
        ConditionedTransition,
        MSAPairWeightedAveraging,
        OuterProductMean,
        PairformerBlock,
        PairformerConfig,
        Transition,
    )
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention
    from miniworld_engine.modules.triangle_attention import (
        BidirectionalTriangleAttention,
        TriangleAttention,
        TrianglePairAttention,
    )
    from miniworld_engine.modules.triangle_multiplication import (
        BidirectionalTriangleMultiplication,
        TriangleMultiplication,
    )

    def IT(i):
        return ImplementationType(i)

    PAIR_D = ({"d_pair": 128}, {"d_pair": 256}, {"d_pair": 384}, {"d_pair": 512})
    # d_hidden is a SEPARATE axis from d_pair, and leaving it at its default is what pinned
    # layernorm_linear_mmajor_bwd to a single bucket N=128 across an entire build: that op keys on
    # N = the projection width, which TriangleMultiplication takes from d_hidden, not d_pair. So
    # sweeping d_pair alone moves the pair tensor and never moves the bucket the kernel keys on.
    PAIR_HID = (
        {"d_pair": 128, "d_hidden": 128}, {"d_pair": 256, "d_hidden": 128},
        {"d_pair": 256, "d_hidden": 256}, {"d_pair": 384, "d_hidden": 256},
        {"d_pair": 512, "d_hidden": 512},
    )
    HID_D = ({"d_hidden": 128}, {"d_hidden": 256}, {"d_hidden": 384})
    BOTH = (BF16, torch.float32)

    return [
        Case("transition",
             lambda dims, p, i, dt: Transition(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, dims["d_hidden"], dt),),
             # bf16 only: the fused kernels are bf16, so an fp32 run falls to torch and there is
             # no autotuner to capture -- 12 fp32 units produced 0 ops each.
             # No "cuda" either: that extension is compiled for sm_90a and will not build on sm_86
             # ("Error building extension 'transition_b2b_cuda'"), so the unit can only fail here.
             dims=HID_D, impls=("miniworld", "triton"),
             switches=("transition_force_split", "transition_cuda_b2b",
                       "transition_fuse_stats", "transition_savedxn_split_bwd",
                       "transition_dab_lnbwd", "transition_lnbwd_privatize",
                       "ln_bwd_path", "ln_partial_reduction")),
        Case("triangle_multiplication",
             lambda dims, p, i, dt: TriangleMultiplication(
                 **dims, implementation=IT(i), p_drop=p).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=PAIR_HID, dtypes=BOTH,
             switches=("p_drop", "trimul_impl", "ln_out_bwd_path", "ln_partial_reduction"),
             impls=("miniworld", "triton", "cute")),
        Case("triangle_multiplication_bidirectional",
             lambda dims, p, i, dt: BidirectionalTriangleMultiplication(
                 **dims, implementation=IT(i), p_drop=p).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=PAIR_HID, switches=("p_drop", "trimul_impl", "ln_out_bwd_path")),
        Case("triangle_attention_bidirectional",
             lambda dims, p, i, dt: BidirectionalTriangleAttention(
                 **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=PAIR_D, switches=("gate_backend", "infer_concat"),
             impls=("miniworld", "triton")),
        # tri-attention buckets key on HEAD_DIM and H, which d_pair never moves
        Case("triangle_attention_heads",
             lambda dims, p, i, dt: TriangleAttention(
                 128, **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, 128, dt), _mask(b, l)),
             # d_hidden is the TOTAL qkv width and the head dim is d_hidden // n_head, which is
             # what lands in the HEAD_DIM bucket -- and tl.dot needs it >= 16, so d_hidden must be
             # at least 16*n_head. (d_hidden=32 with n_head=4 gives a head dim of 8 and fails to
             # compile: "Input shapes should have M >= 1, N >= 1 and K >= 16".)
             dims=({"n_head": 4, "d_hidden": 128}, {"n_head": 8, "d_hidden": 128},
                   {"n_head": 4, "d_hidden": 256}, {"n_head": 16, "d_hidden": 256}),
             lengths=(256, 384, 512), impls=("miniworld", "triton")),
        Case("attention_pair_bias",
             lambda dims, p, i, dt: AttentionPairBias(**dims).cuda().to(dt),
             lambda b, l, dims, dt: (_single(b, l, dims["d_single"], dt),
                                     _pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=({"d_single": 384, "d_pair": 128, "n_head": 8},
                   {"d_single": 768, "d_pair": 128, "n_head": 16},
                   {"d_single": 384, "d_pair": 256, "n_head": 8})),
        Case("augmented_attention",
             lambda dims, p, i, dt: AugmentedAttentionPairBias(
                 **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_single(2, l, dims["d_single"], dt).unsqueeze(1),
                                     _single(2, l, dims["d_cond"], dt).unsqueeze(1),
                                     _pair(1, l, dims["d_pair"], dt), _mask(1, l)),
             # module dtype x core dtype. The cross product is the point: the whole-op wrapper
             # runs the core in bf16 under an fp32 forward, so (fp32, bf16) is production, not a
             # corner -- and it keys to a different cache bucket than (bf16, bf16).
             dims=({"d_single": 384, "d_cond": 384, "d_pair": 128, "n_head": 16},
                   {"d_single": 768, "d_cond": 768, "d_pair": 128, "n_head": 16}),
             dtypes=BOTH, compute_dtypes=BOTH),
        Case("adaptive_layernorm",
             lambda dims, p, i, dt: AdaptiveLayerNorm(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_single(b, l, dims["d_hidden"], dt),
                                     _single(b, l, dims["d_cond"], dt)),
             dims=({"d_hidden": 384, "d_cond": 384}, {"d_hidden": 128, "d_cond": 128},
                   {"d_hidden": 768, "d_cond": 768}), dtypes=BOTH),
        Case("conditioned_transition",
             lambda dims, p, i, dt: ConditionedTransition(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_single(b, l, dims["d_hidden"], dt),
                                     _single(b, l, dims["d_cond"], dt)),
             # bf16 is reachable now that the module takes its dtype at construction instead of
             # pinning the four Linears to fp32; fp32 stays because the bench still runs it there.
             dims=({"d_hidden": 384, "d_cond": 384}, {"d_hidden": 128, "d_cond": 128}),
             dtypes=BOTH),
        Case("msa_pair_weighted_averaging",
             lambda dims, p, i, dt: MSAPairWeightedAveraging(
                 **dims, implementation=IT(i), p_drop=p).cuda().to(dt),
             lambda b, l, dims, dt: (torch.randn(b, 8, l, dims["d_msa"], device="cuda", dtype=dt),
                                     _pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=({"d_msa": 64, "d_pair": 128, "n_head": 8},
                   {"d_msa": 128, "d_pair": 128, "n_head": 8}),
             lengths=(256, 384, 512)),
        Case("outer_product_mean",
             lambda dims, p, i, dt: OuterProductMean(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (torch.randn(b, 8, l, dims["d_msa"], device="cuda", dtype=dt),
                                     torch.ones(b, 8, l, dtype=torch.bool, device="cuda")),
             dims=({"d_msa": 64, "d_pair": 128, "d_hidden": 32},
                   {"d_msa": 128, "d_pair": 128, "d_hidden": 32}),
             lengths=(256, 384, 512)),
        Case("pairformer_block",
             lambda dims, p, i, dt: PairformerBlock(
                 PairformerConfig(**dims, p_drop=p, n_block=1)).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=({"d_pair": 128, "d_hidden_tri_multi": 128, "d_hidden_tri_attention": 32},
                   {"d_pair": 256, "d_hidden_tri_multi": 128, "d_hidden_tri_attention": 32}),
             lengths=(256, 384), switches=("p_drop",)),
        # The two modules below are driven for COVERAGE, not because the current model calls them:
        # an op registers itself with the cache regardless of whether production reaches it today,
        # and an unbuilt op is a full-grid stall the day something starts reaching it. The audit
        # (build.audit, check "reach") is what turned these two up -- they were the only nn.Module
        # exports with no Case at all.
        Case("triangle_pair_attention",
             lambda dims, p, i, dt: TrianglePairAttention(
                 **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             dims=({"d_pair": 128, "n_head": 4}, {"d_pair": 256, "n_head": 8}),
             lengths=(256, 384, 512), impls=("miniworld", "triton")),
        # ---- kernels with no module that dispatches to them ------------------------------- #
        # Registered ops are built because they are registered, not because the current model
        # reaches them. Each entry below drives the op through its own public entry point.
        Case("tm1",
             _kernel_case(("miniworld_engine.kernels.tm1.triton.main", "triton_tm1"),
                          _w(("d", "d"), ("d", "d"), ("d", "d"), ("d", "d"))),
             lambda b, l, dims, dt: (_pair(b, l, dims["d"], dt),),
             dims=({"d": 128}, {"d": 256}), lengths=(256, 384, 512)),
        Case("tm2",
             _kernel_case(("miniworld_engine.kernels.tm2.triton.main", "triton_tm2"),
                          _w(("d", "d"), ("d", "d"))),
             lambda b, l, dims, dt: (_pair(b, l, dims["d"], dt), _pair(b, l, dims["d"], dt)),
             dims=({"d": 128}, {"d": 256}), lengths=(256, 384, 512)),
        Case("gated_projection",
             _kernel_case(("miniworld_engine.kernels.gated_projection.triton.main",
                           "TritonGatedProjectionFunction"), _w(("hd", "d"))),
             lambda b, l, dims, dt: (_pair(b, l, dims["hd"], dt), _pair(b, l, dims["hd"], dt)),
             dims=({"hd": 128, "d": 128}, {"hd": 256, "d": 128}), lengths=(256, 384)),
        Case("layernorm_linear_pair_bias",
             _kernel_case(("miniworld_engine.kernels.layernorm_linear.triton.pair_bias",
                           "triton_layer_norm_linear"), _w(("d",), ("n_head", "d"))),
             lambda b, l, dims, dt: (_pair(b, l, dims["d"], dt),),
             dims=({"d": 128, "n_head": 4}, {"d": 256, "n_head": 8}), lengths=(256, 384, 512)),
        Case("swa_atom_attention",
             lambda dims, p, i, dt: SWA3DRoPEAttention(**dims).cuda().to(dt),
             # forward takes (x, attention_params); the params tuple is built by the caller in
             # production, so the case supplies the same shape the atom encoder passes.
             lambda b, l, dims, dt: (_single(b, l, dims["d_model"], dt).squeeze(0),
                                     _swa_params(l, dims, dt)),
             dims=({"d_model": 128, "n_heads": 4}, {"d_model": 256, "n_heads": 8}),
             lengths=(256, 512), train=False),
        # forward-only kernel probes: no backward is registered for these, so `train=False`
        # (a train unit would only re-run the same forward and write the same entries).
        Case("layernorm_lowreg",
             _kernel_case(("miniworld_engine.kernels.layernorm.triton.lowreg",
                           "triton_layernorm_lowreg"), _w(("d",), ("d",)), tail=(1e-5,)),
             lambda b, l, dims, dt: (_pair(b, l, dims["d"], dt),),
             dims=({"d": 128}, {"d": 256}), lengths=(256, 384), train=False),
        Case("layernorm_transpose",
             _kernel_case(("miniworld_engine.kernels.layernorm.triton.transpose",
                           "layer_norm_transpose"), _w(("d",), ("d",))),
             lambda b, l, dims, dt: (_pair(b, l, dims["d"], dt),),
             dims=({"d": 128}, {"d": 256}), lengths=(256, 384), train=False),
        Case("layernorm_linear_stats",
             _kernel_case(("miniworld_engine.kernels.layernorm_linear.triton.stats",
                           "stats_triton"), _w()),
             lambda b, l, dims, dt: (_pair(b, l, dims["d"], dt).reshape(-1, dims["d"]), 1e-5),
             dims=({"d": 128}, {"d": 256}, {"d": 512}), lengths=(256, 384), train=False),
    ]


def run_case(case: Case, length: int, dim_index: int, *, train: bool, p_drop: float = 0.0,
             impl: str = "miniworld", dtype: torch.dtype = BF16,
             compute_dtype: torch.dtype | None = None) -> int:
    """Forward (and backward) once, so every kernel the module dispatches to fires. Returns 1 on
    success, 0 if the module could not run this shape (an unsupported width is data, not failure).

    ``compute_dtype`` is passed to the module's forward, not to its constructor -- that is the
    whole distinction the ``compute_dtypes`` axis exists to build (see Case.compute_dtypes)."""
    try:
        dims = case.dims[dim_index]
        module = case.factory(dims, p_drop, impl, dtype)
    except Exception as exc:
        print(f"    skip {case.name} dims#{dim_index}: build failed ({type(exc).__name__}: {exc})",
              flush=True)
        return 0
    module.train(train)
    args = case.inputs(1, length, dims, dtype)
    fwd_kwargs = {"compute_dtype": compute_dtype} if compute_dtype is not None else {}
    try:
        if train:
            args = tuple(
                a.detach().clone().requires_grad_(True)
                if torch.is_tensor(a) and a.is_floating_point() else a
                for a in args
            )
            out = module(*args, **fwd_kwargs)
            (out.float().sum() if torch.is_tensor(out) else out[0].float().sum()).backward()
        else:
            with torch.no_grad():
                module(*args, **fwd_kwargs)
        torch.cuda.synchronize()
    except Exception as exc:  # an unsupported shape must not stop the build
        print(f"    skip {case.name} dims#{dim_index} L={length} "
              f"{'train' if train else 'eval'}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return 0
    return 1


# --------------------------------------------------------------------------- #
# work decomposition and parallel execution
# --------------------------------------------------------------------------- #
# The unit of work is one (case, width, length, mode): it writes its own shard, so units never race
# on a file and a crash costs one unit rather than the build. Units run one per GPU, pulled from a
# shared queue -- capture time varies by more than an order of magnitude between them, so a fixed
# split leaves cards idle while one grinds through the tail. Parallelism lives HERE rather than in
# the command line front end, because the decomposition is a property of the build: anything that
# drives the builder gets the same behaviour, and the front end stays a thin argument parser.


@dataclasses.dataclass(frozen=True)
class Unit:
    case: str
    dim_index: int            # which entry of Case.dims -- named kwargs, not a bare number
    length: int
    train: bool
    dtype: str = "bfloat16"
    switch: str = ""
    value: object = None
    impl: str = "miniworld"
    #: compute dtype passed to forward, "" when the module computes in its own dtype
    compute: str = ""

    @property
    def label(self) -> str:
        pin = f" {self.switch}={self.value}" if self.switch else ""
        core = f"->{self.compute}" if self.compute else ""
        return (f"{self.case}[{self.impl}/{self.dtype}{core}] dims#{self.dim_index} "
                f"L={self.length} {'train' if self.train else 'eval'}{pin}")

    @property
    def stem(self) -> str:
        pin = f"-{self.switch}{self.value}" if self.switch else ""
        # the compute dtype is part of the identity: (fp32 module, bf16 core) and (fp32, fp32) are
        # different units writing different cache buckets, so they must not share a shard file.
        core = f"-core{self.compute}" if self.compute else ""
        return (f"{self.case}-{self.impl}-{self.dtype}{core}-dims{self.dim_index}-L{self.length}-"
                f"{'train' if self.train else 'eval'}{pin}")

    def cmd_args(self) -> list[str]:
        """This unit's own arguments to the child. The runner supplies --shard/--compile-jobs."""
        args = ["--case", self.case, "--dims", str(self.dim_index), "--length", str(self.length),
                "--dtype", self.dtype, "--mode", "train" if self.train else "eval",
                "--impl", self.impl]
        if self.compute:
            args += ["--compute-dtype", self.compute]
        if self.switch:
            args += ["--switch", self.switch, "--value", str(self.value)]
        return args

    def env(self) -> dict[str, str]:
        return {}


@dataclasses.dataclass(frozen=True)
class OpUnit:
    """One kernel at one shape bucket -- the unit a per-op tuning sweep is made of.

    A ``Unit`` drives a whole MODULE, so it re-tunes every op that module touches, and two units
    differing only in something an op does not key on land in the same (op, bucket) and pay the
    full grid again -- each unit being its own process. Measured: 3,385 units, of which 1,950 are
    one case, and a single 15,552-config op inside it costs 244 GPU-h of pure re-benching.

    Keyed on (op, L) there is no redundancy: 538 items, each tuning exactly one (op, bucket) once.
    Everything else -- the GPU pool, the O_EXCL claims, --resume, per-unit shards and logs, the
    compile-worker split, merge_shards -- is the SAME machinery, which is why this is a unit kind
    and not a second harness.
    """

    op: str
    length: int
    dtype: str = "bfloat16"
    #: "" for a token/atom kernel, "pair" or "atom" for a `level=both` one. A both-level kernel
    #: keys on ROWS, so its two sides are different buckets at the same length -- atom A=256 is
    #: 256 rows, pair L=256 is 65,536 -- and the side has to be said, not inferred.
    side: str = ""

    @property
    def bucket(self) -> int:
        """The ``shape_key`` value this unit's launch will record.

        Not ``length``. A `level=both` kernel keys on its ROW count, so a pair unit at L records
        ``both_key(L*L)`` -- and a coverage check that compares declared lengths against cached
        buckets would report every both-level op as a total miss.
        """
        from miniworld_engine.autotune.shape_key import both_key

        if self.side == "pair":
            return both_key(self.length * self.length)
        if self.side == "atom":
            return both_key(self.length)
        return self.length

    @property
    def label(self) -> str:
        tag = f" {self.side}" if self.side else ""
        return f"{self.op}[{self.dtype}]{tag} L={self.length}"

    @property
    def stem(self) -> str:
        tag = f"-{self.side}" if self.side else ""
        return f"op-{self.op}-{self.dtype}{tag}-L{self.length}"

    def cmd_args(self) -> list[str]:
        args = ["--op", self.op, "--dtype", self.dtype, "--length", str(self.length)]
        return args + (["--side", self.side] if self.side else [])

    def env(self) -> dict[str, str]:
        # The drivers read this at IMPORT time, like MINIWORLD_SHAPE_MODE -- their shape constants
        # are module-level and the kernels reach them through helpers that close over them, so a
        # per-call override would have to reach inside every driver module. Per-process does not.
        env = {"MINIWORLD_DRIVER_LENGTH": str(self.length)}
        if self.side:
            env["MINIWORLD_DRIVER_SIDE"] = self.side
        return env


def check(selected: list[Case]) -> list[str]:
    """Construct AND run every case once, at its smallest shape, reporting the ones that fail.

    A wrong constructor keyword or a wrong input dtype costs one failed unit per
    (width, length, mode) -- eight to twelve wasted GPU launches for a single mistake, reported
    minutes apart as if they were separate problems. Constructing is not enough to catch it: the
    dtype mismatch that cost thirteen units here only appears once something is pushed through the
    module. So the check does a real forward at the smallest shape, which takes seconds.
    """
    problems = []
    sm = device_sm()
    with _one_config_per_op():
        return _check_inner(selected, sm, problems)


@contextlib.contextmanager
def _one_config_per_op():
    """Pin every op to a single config for the duration.

    The preflight's job is "does this module construct and run", not "which config is fastest",
    and its docstring promises it "takes seconds". That held only while the config sets held one
    config per op. Pointed at a real search grid it inherits the same set as the build, so the
    forward triggers a full autotune sweep -- in the PARENT, on one card, with no compile
    fan-out, before a single unit is dispatched. Measured on configs/grid (205,266 configs):
    15 minutes in, zero units claimed, seven of eight GPUs still at 4 MiB.

    Truncating the live lists is the whole mechanism: `configs_for` hands each autotuner the list
    object itself, so shortening it in place reaches autotuners that already exist, and restoring
    it afterwards leaves the build's own config space untouched.
    """
    from miniworld_engine.autotune.configs import _LISTS

    saved = {op: list(live) for op, live in _LISTS.items()}
    for live in _LISTS.values():
        if len(live) > 1:
            del live[1:]
    try:
        yield
    finally:
        for op, live in _LISTS.items():
            live[:] = saved.get(op, live)


def _check_inner(selected: list[Case], sm, problems: list[str]) -> list[str]:
    for case in selected:
        dims, length, dt = case.dims[0], case.lengths[0], case.dtypes[0]
        dt_name = str(dt).replace("torch.", "")
        impls = [i for i in case.impls
                 if sm is None or build_matrix.allows(sm, case.name, i, dt_name)]
        if not impls:
            continue      # nothing this card can build; units() drops it too, so it is not a defect
        try:
            module = case.factory(dims, 0.0, impls[0], dt)
            module.eval()
            with torch.no_grad():
                module(*case.inputs(1, length, dims, dt))
            torch.cuda.synchronize()
        except Exception as exc:
            # OutOfResources is the autotuner working, not a broken case: a config that wants more
            # shared memory than the card has is exactly what the tuner is there to reject, and it
            # surfaces on any run wide enough to reach one. Treating it as a case defect stopped a
            # 104-unit build before a single unit launched.
            if type(exc).__name__ in {"OutOfResources", "CompileTimeAssertionFailure",
                                      "PTXASError"}:
                continue
            problems.append(f"{case.name}: {type(exc).__name__}: {exc}")
    return problems


def op_units(only: set[str] | None = None, config_dir: Path | None = None) -> list[OpUnit]:
    """One item per (triton op with a driver, DECLARED dtype, shape bucket of its declared level).

    The level comes from registry.csv and decides the bucket set, so a token kernel is never
    driven at an atom length and vice versa -- driving it there would tune a bucket the model
    never asks for while missing ones it does.

    Coverage is DECLARED -- registry.csv crossed with the level -- not incidental. That is the
    whole reason this unit kind exists for `build all`: driving modules reaches only the kernels
    some module happens to dispatch to, measured at 48 of the 91 triton kernels on an A6000, so 43
    declared kernels WITH WORKING DRIVERS were never tuned by a full build and nobody could see it
    from the build's own output.

    Eligibility is "the registry declares it and this config set has a grid for it". It must NOT
    be `registered_ops()`: that reflects which kernel modules THIS process happened to import, and
    the parent imports far fewer than the children do -- filtering on it silently dropped 8 more
    ops that have config files and drivers.

    dtype is declared too, in registry.csv's ``dtypes`` column: token kernels are bf16, atom and
    both are bf16|fp32. This used to emit bfloat16 for everything, so the fp32 half of 66 kernels
    was never driven -- and the coverage check counted (op, bucket) without dtype, so it reported
    527/527 and missing_pairs=0 over a cache that held one of the two declared precisions.
    """
    import csv

    from miniworld_engine.autotune.shape_key import (
        ATOM_SHAPES,
        BOTH_PAIR_LENGTHS,
        SHAPES_BY_LEVEL,
    )

    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    out = []
    for r in csv.DictReader(reg.open()):
        if r["backend"] != "triton" or not (r["driver"] or "").strip():
            continue
        if only and r["kernel"] not in only:
            continue
        if config_dir is not None and not (config_dir / f"{r['kernel']}.csv").is_file():
            continue          # this config set declares no grid for it
        # A `level=both` kernel is TWO work lists, not one. It keys on rows (shape_key.BOTH_ROWS),
        # so a pair L and an atom A of the same value are different buckets -- pair L=256 is
        # 65,536 rows, atom A=256 is 256 -- and driving one length list picks a side per length
        # and never builds the other. 4 pair + 6 atom = 10 buckets, which is exactly BOTH_ROWS.
        # Before this, 8 units covered 8 of the 10 and two of those 8 were the wrong side.
        if r["level"] == "both":
            sided = ([("pair", L) for L in BOTH_PAIR_LENGTHS]
                     + [("atom", A) for A in ATOM_SHAPES])
        else:
            sided = [("", L) for L in SHAPES_BY_LEVEL[r["level"]]]
        if not _keys_on_shape(Path(__file__).resolve().parents[2] / r["file"], r["symbol"]):
            # A kernel that does not key on shape_key has no per-shape cache to build, so driving
            # it at every length would tune one identical bucket N times. transition_fold_triton is
            # the only one today, and correctly so: it reads the WEIGHTS (Wa, Wb (N,K), gamma,
            # beta (K,)) and never touches the activation, so N and K are its whole shape.
            sided = sided[:1]
        alias = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
        dtypes = [alias.get(x, x) for x in (r.get("dtypes") or "bf16").split("|") if x]
        out.append([OpUnit(op=r["kernel"], length=length, dtype=dt, side=side)
                    for dt in dtypes for side, length in sided])
    # INTERLEAVE by op: emit every op's first shape, then every op's second, and so on.
    #
    # Grouped by op -- the obvious order -- is the worst possible one here. The runner hands
    # consecutive items to consecutive GPUs, so all 8 cards on a node get the SAME op at 8
    # different shapes, and a shape does not change the compile key: they race to compile the
    # identical 1440 keys, eight times over, all cold because none has landed in the shared
    # triton cache yet. Measured that way: 972 s of precompile per unit and 6 items finished in
    # an hour across 24 GPUs.
    #
    # Interleaved, the 8 cards get 8 DIFFERENT ops, so the 8 cold compiles are 8 different key
    # sets -- no duplication -- and by the time an op's second shape is picked up its keys are
    # already on disk, which is the warm path (0.56 vs 5.7 core-s per config).
    return [u for row in zip_longest(*out) for u in row if u is not None]


def _keys_on_shape(path: Path, symbol: str) -> bool:
    """Does this kernel's ``@triton.autotune(key=[...])`` include ``shape_key``?"""
    import ast

    try:
        tree = ast.parse(path.read_text())
    except OSError as exc:
        # Loud, because silent was how a wrong path went unnoticed: the file could not be read,
        # every op fell back to "keeps all shapes", and the item count came out unchanged -- which
        # is exactly what a working filter looks like from the outside.
        raise FileNotFoundError(
            f"registry names {path} but it cannot be read ({exc}); the shape-key check would "
            f"silently keep every shape for every op") from exc
    except SyntaxError:
        return True           # cannot tell -> keep every shape rather than drop work
    want = symbol.split(".")[-1]
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name != want:
            continue
        for dec in fn.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if getattr(dec.func, "attr", getattr(dec.func, "id", "")) != "autotune":
                continue
            for kw in dec.keywords:
                if kw.arg == "key" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    return any(getattr(e, "value", None) == "shape_key" for e in kw.value.elts)
    return True


def units(selected: list[Case]) -> list[Unit]:
    out = []
    sm = device_sm()
    for case in selected:
        # build/gpu_to_kernels/<sm>.csv, not a list trimmed in cases(): the sweep is shared across
        # cards, so dropping "cute" from Case.impls to protect sm_86 would also stop building it
        # on an H100, where it is the fastest path there is.
        allowed = {(i, str(d).replace("torch.", "")) for i in case.impls for d in case.dtypes
                   if sm is None or build_matrix.allows(
                       sm, case.name, i, str(d).replace("torch.", ""))}
        for di in range(len(case.dims)):
            for length in case.lengths:
                for dtype in case.dtypes:
                    dt = str(dtype).replace("torch.", "")
                    for train in ((False, True) if case.train else (False,)):
                        mode = "train" if train else "eval"
                        for impl in (i for i in case.impls if (i, dt) in allowed):
                            # "" = no compute-dtype argument at all, which is a DIFFERENT unit from
                            # passing the module's own dtype explicitly only in bookkeeping; cases
                            # without the axis keep their existing stems and stay resumable.
                            cores = tuple(str(c).replace("torch.", "")
                                          for c in case.compute_dtypes) or ("",)
                            for core in cores:
                                out.append(Unit(case.name, di, length, train, dt,
                                                impl=impl, compute=core))
                                for switch in case.switches:
                                    values, modes = SWITCHES[switch]
                                    if mode not in modes:
                                        continue
                                    out.extend(
                                        Unit(case.name, di, length, train, dt, switch, v, impl,
                                             compute=core)
                                        for v in values)
    return out


def _shard_has_entries(shard: Path) -> bool:
    """Did this unit actually PRODUCE something, or only leave a file behind?

    ``--resume`` used to test ``shard.exists()``, and a unit that captured nothing still writes its
    shard: ``dump_shard`` serializes an empty dict when the run skipped (an unsupported shape, or a
    kernel that died before any config was benched). So every zero-op unit was marked done forever
    -- 185 of them per shard dir in the build that prompted this. A resumed build inherited those
    holes and could never fill them, which is the failure mode resume exists to avoid.
    """
    try:
        data = json.loads(shard.read_text())
    except (OSError, ValueError):
        return False
    return any(isinstance(v, dict) and v.get("entries") for v in data.values())


def reclaim_orphans(shard_dir: Path) -> list[str]:
    """Delete claims whose unit produced nothing, so a restarted build can run them again.

    A claim is created with O_EXCL before a unit runs and removed if it produced no ops -- but a
    build that is KILLED (time limit, scancel, node failure) leaves one claim per in-flight unit,
    and nothing ever removes those. The next build then finds the claim, treats the unit as
    "claimed elsewhere", and skips it SILENTLY -- no log line, no failure, just a unit that is
    never built again.

    Not automatic on startup: the claim is also what lets a second node join an in-flight build
    against the same directory, and clearing claims unconditionally would steal that node's work.
    Liveness cannot be settled from the filesystem across nodes, so this stays an explicit
    operator action -- run it when restarting after a kill, not while another build is running.
    """
    freed = []
    for claim in sorted(shard_dir.glob("*.claim")):
        if not _shard_has_entries(shard_dir / f"{claim.stem}.json"):
            claim.unlink(missing_ok=True)
            freed.append(claim.stem)
    return freed


#: A single launch may not exceed this before the parent kills the whole unit process. A launched
#: CUDA kernel has no host-side cancellation and SIGALRM cannot interrupt cudaStreamSynchronize
#: (PEP 475), so tearing down the context by killing the process is the ONLY thing that stops one.
#: The compile guard already works exactly this way; this is the same guard for the bench half.
#: Measured need: with the full grid open, one config ({BM:16,BK:16,BN:16,warps:1,stages:1}) ran
#: 468 SECONDS -- 85% of that unit's benchmarking -- and no in-process check can shorten it,
#: because a launch's duration is only readable once it has already finished.
# `Unit | OpUnit`: `build_all` decomposes a per-op sweep into OpUnits and puts them on the
# same queue, so both kinds reach here. The annotation said `Unit` while every unit of the
# 922-unit sweep that produced the shipped cache was an OpUnit.
def _run_unit_subprocess(unit: Unit | OpUnit, device: int, shard_dir: Path, repo: Path,
                         compile_jobs: int, config_dir: Path | None = None,
                         fill_gaps: bool = False) -> dict:
    """One unit, in its own process on one card. Subprocess rather than thread: a capture can take
    the CUDA context down with it, and one dead unit must not end the build."""
    shard = shard_dir / f"{unit.stem}.json"
    # Claim the unit by creating its marker exclusively. --resume alone only filters at startup, so
    # two builds pointed at one shard dir would each take the whole list and run every unit twice;
    # an O_EXCL create is the one check that cannot race. Lets a second node join an in-flight build
    # simply by starting against the same directory.
    claim = shard_dir / f"{unit.stem}.claim"
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return {"label": unit.label, "gpu": device, "rc": 0, "ops": -1, "seconds": 0.0,
                "shard": str(shard), "log": "", "claimed_elsewhere": True}
    log = shard_dir / "logs" / f"gpu{device}-{unit.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)   # CUDA's own interface, not an engine switch
    cmd = [sys.executable, "-u", "-m", "miniworld_engine.autotune.builder",
           "--shard", str(shard), "--compile-jobs", str(compile_jobs), *unit.cmd_args()]
    # Paths rather than the parsed configs: the child re-reads the CSVs itself, so a unit's config
    # space is reproducible from its own command line (the same reason every other knob here is an
    # argument and not inherited shell state).
    if config_dir is not None:
        cmd += ["--config-dir", str(config_dir)]
    if fill_gaps:
        cmd += ["--fill-gaps"]
    env.update(unit.env())
    started = time.monotonic()
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=repo, stdout=handle, stderr=subprocess.STDOUT,
                              check=False, env=env)
    ops = 0
    if shard.exists():
        try:
            raw = json.loads(shard.read_text())
            ops = sum(1 for v in raw.values() if isinstance(v, dict) and v.get("entries"))
        except (OSError, ValueError):
            ops = 0
    # Read the log whatever `ops` says. A unit is `--op <one kernel>`, but its driver fires the
    # neighbouring kernels too, so the shard can hold entries for ops that ran while the DRIVEN
    # one was permanently skipped -- the child then returns rc=1 for `ran=0` with `ops=3` on
    # disk. Gating this read on `not ops` meant that case never set `skipped`, and the merge
    # reported "1 bad unit ... entries will be MISSING" against a card that had answered
    # correctly: `augmented_attention_bwd_split_triton[float32] L=4096` wants 153,600 B of shared
    # memory and an A6000 has 101,376.
    try:
        skipped = "[unit] SKIPPED-PERMANENT" in log.read_text()
    except OSError:
        skipped = False
    if not ops and not skipped:
        claim.unlink(missing_ok=True)  # nothing produced: let a later run retry this unit
    # a permanent skip KEEPS its claim: the shape will not fit on the next attempt either, and
    # releasing it made every resumed job re-claim the same OOMing units and produce nothing.
    return {"label": unit.label, "gpu": device, "rc": proc.returncode, "ops": ops,
            "skipped": skipped,
            "seconds": round(time.monotonic() - started, 1), "shard": str(shard), "log": str(log)}


def build_all(selected: list, shard_dir: Path, gpus: list[int], compile_jobs: int,
              resume: bool = False, reclaim: bool = False,
              config_dir: Path | None = None, fill_gaps: bool = False) -> list[dict]:
    """Run every unit of ``selected`` across ``gpus``. Returns one result record per unit."""
    import concurrent.futures as cf

    repo = Path(__file__).resolve().parents[3]
    shard_dir.mkdir(parents=True, exist_ok=True)

    # `selected` is either Cases (module units, decomposed here) or OpUnits (already the work
    # items). Everything downstream -- pool, claims, resume, shards, merge -- is shared; only the
    # three case-shaped preliminaries below differ.
    case_build = bool(selected) and isinstance(selected[0], Case)

    # DIVIDE the cores among the concurrent units. Each unit is its own process and works out its
    # own compile fan-out from `sched_getaffinity`, which reports the WHOLE allocation -- so every
    # unit assumes it owns the machine. Eight units on a 64-core node therefore asked for
    # min(32, 64) = 32 workers each: 256 compile processes, measured as 529 python processes and a
    # load average of 239. Compile is ~99% of build cost and ptxas is memory-hungry, so
    # oversubscribing it that hard costs throughput rather than buying any.
    if not compile_jobs:
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        compile_jobs = max(1, cores // max(1, len(gpus)))
        print(f"  [compile] {cores} cores / {len(gpus)} gpus -> {compile_jobs} compile workers "
              f"per unit ({compile_jobs * len(gpus)} total)", flush=True)

    # OpUnits carry no Case, and `check` is a per-case module smoke test, so there is nothing for
    # it to check -- a driver that cannot run its shape reports that as a skipped unit, which is
    # the same contract run_case has.
    broken = check(selected) if case_build else []
    if broken:
        print("cases that will not build -- fix these before running units:")
        for line in broken:
            print(f"  {line}")
        return [{"label": b.split(":")[0], "gpu": -1, "rc": 2, "ops": 0, "seconds": 0.0,
                 "shard": "", "log": ""} for b in broken]
    if reclaim:
        freed = reclaim_orphans(shard_dir)
        print(f"reclaimed {len(freed)} orphaned claim(s) from a killed build", flush=True)
        for stem in freed[:20]:
            print(f"    {stem}", flush=True)
    work = units(selected) if case_build else list(selected)
    if resume:
        work = [u for u in work if not _shard_has_entries(shard_dir / f"{u.stem}.json")]
    if not work:
        print("nothing to do (every unit already has a shard with entries)")
        return []

    queue: Queue = Queue()
    for unit in work:
        queue.put(unit)
    print(f"build: {len(selected)} case(s), {len(work)} unit(s), {len(gpus)} gpu(s) -> {shard_dir}",
          flush=True)
    sm = device_sm()
    # skipped_units reports which (case, impl, dtype) the build matrix denies on this card.
    # An OpUnit has no impl axis -- it drives one kernel through its driver -- so there is nothing
    # to report and nothing to skip.
    skipped = skipped_units(selected, sm) if case_build else []
    if skipped:
        print(f"  {sm}: {len(skipped)} case/impl/dtype combination(s) not built here "
              f"(build/gpu_to_kernels/{sm}.csv)", flush=True)
        for label, why in skipped:
            print(f"    - {label}: {why}", flush=True)

    def worker(device: int) -> list[dict]:
        got = []
        while True:
            try:
                unit = queue.get_nowait()
            except Empty:
                return got
            res = _run_unit_subprocess(unit, device, shard_dir, repo, compile_jobs,
                                       config_dir, fill_gaps)
            if res.get("claimed_elsewhere"):
                continue
            status = ("ok" if res["rc"] == 0 and res["ops"] else
                      "skip" if res.get("skipped") else
                      "EMPTY" if res["rc"] == 0 else "FAIL")
            print(f"  [gpu{device}] {status:5s} {res['label']}  {res['seconds']}s  {res['ops']} ops",
                  flush=True)
            got.append(res)

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in cf.as_completed([pool.submit(worker, g) for g in gpus]):
            results.extend(future.result())
    return results


def audit(selected: list[Case]) -> list[tuple]:
    """Replay the build matrix with capture OFF and return every lookup the cache did not serve.

    A build can only report what it captured; that says nothing about whether the result COVERS the
    work. This runs the same modules against the finished cache and collects the misses the engine
    already reports, so "no missing shapes" becomes a measurement rather than a claim: an empty
    result means every (op, dtype, bucket) this matrix reaches was found.

    Run it in a FRESH PROCESS, once. Two reasons, both of which made a before/after over one
    process report the before twice: the miss set accumulated and was never cleared (fixed here by
    clearing it), and triton's Autotuner memoises its choice per tuning key on the instance, so a
    second replay never consults the cache again -- measured, the second call returned in 0 s and
    named the same four misses a filled cache had just covered.
    """
    from miniworld_engine import settings
    from miniworld_engine.autotune.cache import cache_misses, clear_cache_misses

    clear_cache_misses()
    settings.configure(run_autotune=False, capture=False)   # use the cache, do not rebuild it
    for case in selected:
        for di in range(len(case.dims)):
            for length in case.lengths:
                for dtype in case.dtypes:
                    for train in ((False, True) if case.train else (False,)):
                        for impl in case.impls:
                            run_case(case, length, di, train=train, impl=impl, dtype=dtype)
    return sorted(cache_misses())


def _run_one_driver(op: str) -> int:
    """Launch one kernel through the driver registry.csv names for it. 1 on success, 0 if the
    shape is one this kernel cannot run (data, not failure -- same contract as run_case)."""
    import csv
    import importlib

    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    row = next((r for r in csv.DictReader(reg.open()) if r["kernel"] == op), None)
    if row is None or not (row.get("driver") or "").strip():
        print(f"    no driver for {op!r} in registry.csv", file=sys.stderr)
        return 0
    mod_name, _, fn_name = row["driver"].partition(":")
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name)
    except Exception as exc:
        print(f"    skip {op}: driver import failed ({type(exc).__name__}: {exc})", flush=True)
        return 0
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as exc:  # an unsupported shape must not stop the sweep
        # OOM and OutOfResources are PERMANENT facts about this card at this shape, not failures
        # to retry: the tensors do not fit, or the tiles want more smem than the SM has. Saying so
        # in a line the parent can read is what stops a resumed run from re-claiming them forever
        # and -- because the parent counts "did anything succeed?" -- from refusing to merge the
        # 526 units that did.
        perm = type(exc).__name__ in ("OutOfMemoryError", "OutOfResources") or \
            "out of memory" in str(exc).lower()
        print(f"    skip {op} at this shape: {type(exc).__name__}: "
              f"{str(exc).strip().splitlines()[0][:160]}", flush=True)
        if perm:
            print(f"  [unit] SKIPPED-PERMANENT {op}: shape does not fit this GPU", flush=True)
        return 0
    return 1


def _report_unit(shard: str) -> int:
    """Print everything a finished unit knows, then dump its shard. Returns ops dumped.

    ONE reporter for both unit kinds. They had diverged: each path was missing a different half of
    the diagnostics -- the ``--op`` path (which is what a sweep actually runs, since build_all
    decomposes into OpUnits) skipped one summary and the ``--case`` path never called
    ``record_errors``, so a capture that failed silently stayed silent there. Exactly the shape of
    bug that costs an afternoon later.
    """
    from miniworld_engine.autotune import capture

    print(capture.precompile_summary(), flush=True)
    print(capture.summary(), flush=True)
    n = capture.dump_shard(shard)
    errs = capture.record_errors()
    if errs:
        print(f"  [capture] recording failures: {errs}", flush=True)
    return n


def _child_main(argv: list[str] | None = None) -> int:
    """Entry point for ONE unit; the parent invokes this via -m."""
    import argparse

    from miniworld_engine import settings
    from miniworld_engine.autotune import capture

    ap = argparse.ArgumentParser(description="build one autotune-cache unit")
    ap.add_argument("--case", default="")
    ap.add_argument("--op", default="",
                    help="tune ONE kernel via its registry driver, at --length, instead of "
                         "driving a whole module")
    ap.add_argument("--dims", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=0)
    ap.add_argument("--side", default="", choices=("", "pair", "atom"),
                    help="which side of a `level=both` kernel to drive. It keys on rows, so pair "
                         "L and atom A of the same value are different buckets and the side "
                         "cannot be inferred from --length. Reaches the drivers as "
                         "MINIWORLD_DRIVER_SIDE, which they read at import.")
    ap.add_argument("--mode", choices=("eval", "train"), default="eval")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--compile-jobs", type=int, default=0)
    ap.add_argument("--impl", default="miniworld")
    ap.add_argument("--config-dir", default="",
                    help="directory of <op>.csv config files; every kernel's grid comes from here")
    ap.add_argument("--compute-dtype", default="",
                    help="dtype passed to forward() as compute_dtype; empty = do not pass one")
    ap.add_argument("--switch", default="")
    ap.add_argument("--value", default="")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="leave keys the cache already holds alone; full-grid only the misses. "
                         "See settings.Settings.fill_gaps")
    args = ap.parse_args(argv)

    if args.config_dir:
        from miniworld_engine.autotune.configs import use_config_dir
        use_config_dir(args.config_dir, require_all=False)
        # Reported after run_case, when the kernels have imported and registered: counting here
        # would always print 0/0, since nothing has asked for configs yet.
        print(f"  [config] set to {args.config_dir}", flush=True)
    settings.configure(run_autotune=True, capture=True, fill_gaps=args.fill_gaps,
                       compile_jobs=(args.compile_jobs or None))
    p_drop = 0.0
    if args.switch == "p_drop":
        p_drop = float(args.value)          # a module argument, not a settings pin
    elif args.switch:
        try:
            field, parse = SWITCH_SETTINGS[args.switch]
        except KeyError:
            print(f"switch {args.switch!r} has no settings pin -- add it to SWITCH_SETTINGS "
                  f"or the unit silently rebuilds the default side", file=sys.stderr)
            return 2
        settings.configure(**{field: parse(args.value)})
    if args.op:
        # ONE kernel at ONE shape, via its registry driver. The shape reached the drivers through
        # MINIWORLD_DRIVER_LENGTH (and, for a `level=both` kernel, MINIWORLD_DRIVER_SIDE) in the
        # environment, before any of them imported. `--side` is on the command line so the unit is
        # reproducible from it; the env var is what the drivers actually read.
        capture.install()
        n_done = capture.load_compile_state(args.shard)
        if n_done:
            print(f"  [resume] {n_done} compile(s) replayable", flush=True)
        ran = _run_one_driver(args.op)
        n = _report_unit(args.shard)
        print(f"unit ran={ran} ops={n}", flush=True)
        return 0 if ran else 1

    case = next((c for c in cases() if c.name == args.case), None)
    if case is None:
        print(f"unknown case {args.case!r}", file=sys.stderr)
        return 2

    capture.install()
    n_done = capture.load_compile_state(args.shard)
    if n_done:
        print(f"  [resume] {n_done} compile(s) replayable from an earlier attempt", flush=True)
    ran = run_case(case, args.length, args.dims, train=(args.mode == "train"), p_drop=p_drop,
                   impl=args.impl, dtype=getattr(torch, args.dtype),
                   compute_dtype=getattr(torch, args.compute_dtype) if args.compute_dtype else None)
    n = _report_unit(args.shard)
    print(f"unit ran={ran} ops={n}", flush=True)
    return 0 if ran else 1


if __name__ == "__main__":
    raise SystemExit(_child_main())
