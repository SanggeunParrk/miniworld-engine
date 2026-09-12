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
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path
from queue import Empty, Queue
from typing import ClassVar

import torch

from miniworld_engine import build as build_matrix
from miniworld_engine.autotune import triton_cache, width_evidence
from miniworld_engine.autotune.module_registry import module_rows

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
    #: lengths worth sweeping -- the UNION over this case's rows in registry_module.csv. Kept as
    #: a field because `audit` and the plan dump ask "what lengths does this case cover"; the
    #: sweep itself must use `lengths_for(dim_index)`, because a module's rows do NOT share a
    #: ladder. `conditioned_transition` is the DiT transition on BOTH sides: its (768, 384) row is
    #: the token DiT at token counts and its (128, 128) row is the atom DiT at atom counts, and
    #: sweeping the union would tune the atom widths at token lengths and vice versa -- which is
    #: exactly the hole `registry_module.csv` exists to close.
    lengths: tuple[int, ...] = (256, 384, 512, 768, 1024)
    #: per-dims length ladder, same order as `dims`. Empty means "every dims uses `lengths`".
    lengths_by_dim: tuple[tuple[int, ...], ...] = ()
    #: the stream each dims entry belongs to, same order as `dims` -- token_pair / atom_single /
    #: token_single / msa_token. Reported, so a unit says which activation it is driving.
    streams: tuple[str, ...] = ()
    rows: tuple = ()

    def augmentation_for(self, dim_index: int, train: bool) -> int:
        return self.rows[dim_index].augmentation("train" if train else "eval") if self.rows else 1

    def input_args(self, dim_index: int, length: int, dtype: torch.dtype, *, train: bool):
        return self.inputs(self.augmentation_for(dim_index, train), length,
                           self.dims[dim_index], dtype, self.stream_for(dim_index))

    def stream_for(self, dim_index: int) -> str:
        """Which activation THIS dims entry drives -- and therefore the rank of its input."""
        return self.streams[dim_index] if self.streams else "token_pair"

    def lengths_for(self, dim_index: int) -> tuple[int, ...]:
        """The lengths THIS dims entry runs at."""
        if self.lengths_by_dim:
            return self.lengths_by_dim[dim_index]
        return self.lengths
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
    "ln_bwd_path": (("persistent", "atomic", "cuda"), ("train",)),
    "ln_out_bwd_path": (("split", "fused"), ("train",)),
    "transition_force_split": ((True,), ("eval", "train")),
    "transition_cuda_b2b": ((False,), ("eval",)),
    "transition_fuse_stats": ((True,), ("eval", "train")),
    "transition_savedxn_split_bwd": ((True,), ("train",)),
    "transition_dab_lnbwd": ((True,), ("train",)),
    "transition_lnbwd_privatize": ((False,), ("train",)),
    "trimul_impl": (("triton", "cute"), ("eval", "train")),
    # The rest of settings.py's backend selectors. They were left out when SWITCHES was written
    # and each one is a set of kernels no unit ever reached: measured against the shipped A6000
    # cache, 26 kernels with working drivers were in the cache and in NO derived unit, and the
    # far side of these switches is where most of them live.
    "transition_large_d_training": (("triton", "cute"), ("train",)),
    "transition_cute_backward": (("cute",), ("train",)),
    "transition_gatebwd_wgmma": ((False,), ("train",)),
    "transition_lnbwd_cuda": ((False,), ("train",)),
    "layernorm_cuda_bwd": ((True,), ("train",)),
    "trimul_cute_dispatch": ((False,), ("eval", "train")),
    "trimul_train_front_fused": ((False,), ("train",)),
    "trimul_out_layout": (("bdll_direct", "bdll_direct_wide"), ("eval", "train")),
}

#: switch name -> the ``settings`` field it pins, and how to parse the CLI string back to a value.
#: A table rather than an if/elif chain in the child: every switch added to SWITCHES must be
#: pinnable, and a chain lets one be added without its pin -- which silently produces a duplicate
#: of the default unit instead of the other side of the switch.
SWITCH_SETTINGS: dict[str, tuple[str, Callable[[str], object]]] = {
    "gate_backend": ("pin_gate_backend", str),
    "infer_concat": ("pin_infer_concat", lambda v: v == "True"),
    "ln_bwd_path": ("layernorm_bwd_path", str),
    "ln_out_bwd_path": ("layernorm_out_bwd_path", str),
    "transition_force_split": ("transition_force_split", lambda v: v == "True"),
    "transition_cuda_b2b": ("transition_cuda_b2b", lambda v: v == "True"),
    "transition_fuse_stats": ("transition_fuse_stats", lambda v: v == "True"),
    "transition_savedxn_split_bwd": ("transition_savedxn_split_bwd", lambda v: v == "True"),
    "transition_dab_lnbwd": ("transition_dab_lnbwd", lambda v: v == "True"),
    "transition_lnbwd_privatize": ("transition_lnbwd_privatize", lambda v: v == "True"),
    "trimul_impl": ("trimul_impl", str),
    "transition_large_d_training": ("transition_large_d_training", str),
    "transition_cute_backward": ("transition_cute_backward", str),
    "transition_gatebwd_wgmma": ("transition_gatebwd_wgmma", lambda v: v == "True"),
    "transition_lnbwd_cuda": ("transition_lnbwd_cuda", lambda v: v == "True"),
    "layernorm_cuda_bwd": ("layernorm_cuda_bwd", lambda v: v == "True"),
    "trimul_cute_dispatch": ("trimul_cute_dispatch", lambda v: v == "True"),
    "trimul_train_front_fused": ("trimul_train_front_fused", lambda v: v == "True"),
    "trimul_out_layout": ("trimul_out_layout", str),
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


def _swa_params(length: int, dims: dict, dtype: torch.dtype, batch: int = 1) -> tuple:
    """``(cos, sin, seqused, cu_seqlens, max_seqlen, valid)`` for SWA3DRoPEAttention.forward.

    Its forward takes the rotary tables and the varlen packing as a prebuilt tuple (production
    computes them once per batch and reuses them across blocks), so the case has to supply the
    same tuple rather than a plain tensor. ``batch`` sequences of ``length`` tokens, all valid.
    """
    head_dim = dims["d_model"] // dims["n_heads"]
    half = head_dim // 2
    # Match build_attention_params / the native benchmark: one FP32 table per
    # position, shared across heads. Activation dtype must not alter the cache key.
    cos = torch.ones(batch, length, half, device="cuda", dtype=torch.float32)
    sin = torch.zeros(batch, length, half, device="cuda", dtype=torch.float32)
    seqused = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device="cuda")
    valid = torch.ones(batch, length, dtype=torch.bool, device="cuda")
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


def _shapes(module: str) -> dict:
    """Every number a Case sweeps, taken from ``registry_module.csv``.

    This is the whole point of that file. These used to be hand-written tuples on each Case, and
    a second set of hand-written tuples in ``op_units`` had to agree with them; they did not, and
    every disagreement was a bucket production reaches with no cache entry (``TOKEN_SHAPES``
    stopped at 512 while the sweep ran to 1024, ``MSA_WIDTHS`` held 64 while the config declares
    64 and 128). One file states the shapes now, ``dev derive`` says exactly which kernels those
    shapes launch, and neither is written by hand.

    Rows of one module must agree on everything except dims and lengths -- see
    ``test_a_modules_rows_agree_on_what_is_not_a_shape``. dims and lengths are per row precisely
    because they are what a row IS.
    """
    rows = [r for r in module_rows() if r.module == module]
    if not rows:
        msg = (f"registry_module.csv has no row for {module!r}. Every case takes its shapes from "
               f"that file; a case with no row would sweep nothing.")
        raise KeyError(msg)
    first = rows[0]
    switches = tuple(dict.fromkeys(name for r in rows for name, _v in r.options))
    return {
        "dims": tuple(r.dims for r in rows),
        "rows": tuple(rows),
        "lengths_by_dim": tuple(r.lengths for r in rows),
        "streams": tuple(r.stream for r in rows),
        "lengths": tuple(sorted({n for r in rows for n in r.lengths})),
        "dtypes": tuple(getattr(torch, d) for d in first.dtypes),
        "compute_dtypes": tuple(getattr(torch, d) for d in first.computes),
        "impls": first.impls,
        # p_drop is a constructor argument, not a settings pin, so it is not a Case switch --
        # `units` reads it off `case.switches` and SWITCHES, and SWITCHES has it.
        "switches": switches,
        "train": "train" in first.modes,
    }


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

    # The dims that used to be declared here -- PAIR_D, PAIR_HID, HID_D, BOTH -- are in
    # registry_module.csv now, with a `source` column saying where each number came from. They are
    # gone rather than kept alongside it, because two declarations of the same shapes is exactly
    # what this file was: `TOKEN_SHAPES` stopped at 512 while these ran to 1024, and the build
    # tuned one set while `--replay` asked for the other.
    #
    # Two facts they carried are worth keeping, and both are enforced elsewhere now:
    #   * d_hidden must EQUAL d_pair. Every fused trimul back half normalises over tri's channel
    #     axis and gates the pair over the same axis, so the two widths are one axis; the module
    #     raises on a mismatch, and `dev derive` reports the refusal rather than silently building
    #     a unit that launches nothing.
    #   * an asymmetric pair is a PyTorch-only shape, so it is not a build unit at all.

    return [
        Case("transition",
             lambda dims, p, i, dt: Transition(**dims, implementation=IT(i)).cuda().to(dt),
             # The stream decides the RANK: token_pair is (B, L, L, D) and token_single is
             # (B, L, D). Before this the token_single row built a pair too, so it produced
             # byte-identical units to the d_hidden=384 pair row -- 152 duplicate units, and the
             # single-stream buckets (rows = B*L, not B*L*L) still had no entry anywhere.
             lambda b, l, dims, dt, s: (
                 (_pair if s == "token_pair" else _single)(b, l, dims["d_hidden"], dt),),
             # bf16 only: the fused kernels are bf16, so an fp32 run falls to torch and there is
             # no autotuner to capture -- 12 fp32 units produced 0 ops each.
             # No "cuda" either: that extension is compiled for sm_90a and will not build on sm_86
             # ("Error building extension 'transition_b2b_cuda'"), so the unit can only fail here.
             **_shapes("transition")),
        Case("triangle_multiplication",
             lambda dims, p, i, dt: TriangleMultiplication(
                 **dims, implementation=IT(i), p_drop=p).cuda().to(dt),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("triangle_multiplication")),
        Case("triangle_multiplication_bidirectional",
             lambda dims, p, i, dt: BidirectionalTriangleMultiplication(
                 **dims, implementation=IT(i), p_drop=p).cuda().to(dt),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("triangle_multiplication_bidirectional")),
        Case("triangle_attention_bidirectional",
             lambda dims, p, i, dt: BidirectionalTriangleAttention(
                 **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("triangle_attention_bidirectional")),
        # tri-attention buckets key on HEAD_DIM and H, which d_pair never moves
        Case("triangle_attention_heads",
             lambda dims, p, i, dt: TriangleAttention(
                 128, **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_pair(b, l, 128, dt), _mask(b, l)),
             # d_hidden is the TOTAL qkv width and the head dim is d_hidden // n_head, which is
             # what lands in the HEAD_DIM bucket -- and tl.dot needs it >= 16, so d_hidden must be
             # at least 16*n_head. (d_hidden=32 with n_head=4 gives a head dim of 8 and fails to
             # compile: "Input shapes should have M >= 1, N >= 1 and K >= 16".
             **_shapes("triangle_attention_heads")),
        Case("attention_pair_bias",
             lambda dims, p, i, dt: AttentionPairBias(**dims).cuda().to(dt),
             lambda b, l, dims, dt, s: (_single(b, l, dims["d_single"], dt),
                                     _pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("attention_pair_bias")),
        Case("augmented_attention",
             lambda dims, p, i, dt: AugmentedAttentionPairBias(
                 **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_single(b, l, dims["d_single"], dt).unsqueeze(1),
                                     _single(b, l, dims["d_cond"], dt).unsqueeze(1),
                                     _pair(1, l, dims["d_pair"], dt), _mask(1, l)),
             # module dtype x core dtype. The cross product is the point: the whole-op wrapper
             # runs the core in bf16 under an fp32 forward, so (fp32, bf16) is production, not a
             # corner -- and it keys to a different cache bucket than (bf16, bf16).
             **_shapes("augmented_attention")),
        Case("adaptive_layernorm",
             lambda dims, p, i, dt: AdaptiveLayerNorm(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_single(b, l, dims["d_hidden"], dt).unsqueeze(1),
                                     _single(b, l, dims["d_cond"], dt).unsqueeze(1)),
             # Same two the model builds -- AdaptiveLayerNorm is constructed inside
             # ConditionedTransition and AugmentedAttentionPairBias with the block's own
             # (d_single, d_cond), so it sees 768/384 and 128/128 and nothing else.
             **_shapes("adaptive_layernorm")),
        # The two combinations the model builds, and only those. The model's config gives the
        # DiffusionTransformer its `(d_single, d_cond)` and the block passes them straight through
        # as `ConditionedTransition(d_hidden=d_single, d_cond=d_cond)`:
        #
        #     token_dit   d_single = d_single_token 768   d_cond = d_single 384    24 blocks
        #     atom_dit    d_single = d_single_atom  128   d_cond = d_single_atom 128   3 blocks
        #
        # Those are AlphaFold-3's own widths -- c_token 768, c_s 384, c_atom 128 -- so the token
        # side is NOT square, and a sweep of square pairs (384/384) tunes a shape the model never
        # builds while leaving 768/384, the 24-block half, with no entry at all.
        Case("conditioned_transition",
             lambda dims, p, i, dt: ConditionedTransition(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_single(b, l, dims["d_hidden"], dt).unsqueeze(1),
                                     _single(b, l, dims["d_cond"], dt).unsqueeze(1)),
             # bf16 is reachable now that the module takes its dtype at construction instead of
             # pinning the four Linears to fp32; fp32 stays because the bench still runs it there.
             **_shapes("conditioned_transition")),
        Case("msa_pair_weighted_averaging",
             lambda dims, p, i, dt: MSAPairWeightedAveraging(
                 **dims, implementation=IT(i), p_drop=p).cuda().to(dt),
             lambda b, l, dims, dt, s: (torch.randn(b, 8, l, dims["d_msa"], device="cuda", dtype=dt),
                                     _pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("msa_pair_weighted_averaging")),
        Case("outer_product_mean",
             lambda dims, p, i, dt: OuterProductMean(**dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (torch.randn(b, 8, l, dims["d_msa"], device="cuda", dtype=dt),
                                     torch.ones(b, 8, l, dtype=torch.bool, device="cuda")),
             **_shapes("outer_product_mean")),
        # `implementation` is a SECOND argument to PairformerBlock, not a config field, and it was
        # not being passed -- so every pairformer_block unit built the PyTorch reference and
        # launched no kernel at all. Silent: the unit succeeded, wrote an empty shard, and the
        # block's kernels were covered only incidentally by the leaf-module cases.
        Case("pairformer_block",
             lambda dims, p, i, dt: PairformerBlock(
                 PairformerConfig(**dims, p_drop=p, n_block=1),
                 implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("pairformer_block")),
        # The two modules below are driven for COVERAGE, not because the current model calls them:
        # an op registers itself with the cache regardless of whether production reaches it today,
        # and an unbuilt op is a full-grid stall the day something starts reaching it. The audit
        # (build.audit, check "reach") is what turned these two up -- they were the only nn.Module
        # exports with no Case at all.
        Case("triangle_pair_attention",
             lambda dims, p, i, dt: TrianglePairAttention(
                 **dims, implementation=IT(i)).cuda().to(dt),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d_pair"], dt), _mask(b, l)),
             **_shapes("triangle_pair_attention")),
        # ---- kernels with no module that dispatches to them ------------------------------- #
        # Registered ops are built because they are registered, not because the current model
        # reaches them. Each entry below drives the op through its own public entry point.
        Case("tm1",
             _kernel_case(("miniworld_engine.kernels.tm1.triton.main", "triton_tm1"),
                          _w(("d", "d"), ("d", "d"), ("d", "d"), ("d", "d"))),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d"], dt),),
             **_shapes("tm1")),
        Case("tm2",
             _kernel_case(("miniworld_engine.kernels.tm2.triton.main", "triton_tm2"),
                          _w(("d", "d"), ("d", "d"))),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d"], dt), _pair(b, l, dims["d"], dt)),
             **_shapes("tm2")),
        Case("gated_projection",
             _kernel_case(("miniworld_engine.kernels.gated_projection.triton.main",
                           "TritonGatedProjectionFunction"), _w(("hd", "d"))),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["hd"], dt), _pair(b, l, dims["hd"], dt)),
             **_shapes("gated_projection")),
        Case("layernorm_linear_pair_bias",
             _kernel_case(("miniworld_engine.kernels.layernorm_linear.triton.pair_bias",
                           "triton_layer_norm_linear"), _w(("d",), ("n_head", "d"))),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d"], dt),),
             **_shapes("layernorm_linear_pair_bias")),
        Case("swa_atom_attention",
             lambda dims, p, i, dt: SWA3DRoPEAttention(**dims).cuda().to(dt),
             # forward takes (x, attention_params); the params tuple is built by the caller in
             # production, so the case supplies the same shape the atom encoder passes.
             # (N, S, d) -- `forward` reads `n, s = x.shape[:2]` and views Wqkv(x) as
             # (n, s, 3, n_heads, head_dim). A squeeze(0) here handed it (S, d), so it took the
             # WIDTH as the sequence length and every unit died on "shape [...] is invalid for
             # input of size ..." -- at every length, in every build, since the case was written.
             lambda b, l, dims, dt, s: (_single(b, l, dims["d_model"], dt),
                                     _swa_params(l, dims, dt, b)),
             **_shapes("swa_atom_attention")),
        # forward-only kernel probes: no backward is registered for these, so `train=False`
        # (a train unit would only re-run the same forward and write the same entries).
        Case("layernorm_lowreg",
             _kernel_case(("miniworld_engine.kernels.layernorm.triton.lowreg",
                           "triton_layernorm_lowreg"), _w(("d",), ("d",)), tail=(1e-5,)),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d"], dt),),
             **_shapes("layernorm_lowreg")),
        Case("layernorm_transpose",
             _kernel_case(("miniworld_engine.kernels.layernorm.triton.transpose",
                           "layer_norm_transpose"), _w(("d",), ("d",))),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d"], dt),),
             **_shapes("layernorm_transpose")),
        Case("layernorm_linear_stats",
             _kernel_case(("miniworld_engine.kernels.layernorm_linear.triton.stats",
                           "stats_triton"), _w()),
             lambda b, l, dims, dt, s: (_pair(b, l, dims["d"], dt).reshape(-1, dims["d"]), 1e-5),
             **_shapes("layernorm_linear_stats")),
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
    args = case.input_args(dim_index, length, dtype, train=train)
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
    generation: str = ""

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
                f"{'train' if self.train else 'eval'}{pin}"
                f"{'-plan' + self.generation if self.generation else ''}")

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
    #: Base channel WIDTH to drive at. The shape key carries the whole shape (plan.md G5), so a
    #: bucket is a (rows, widths) pair and a sweep that varies only the length reaches exactly one
    #: width per op -- whichever its driver happens to build. That is the 363 uncovered lookups the
    #: module pass exists to reach. One number per unit, not one per axis: a driver derives its
    #: other widths from a base (`_DC`, `ND = n*D`, `NH = D//32`), so overriding the base moves them
    #: together, the way changing `d_pair` does in the model.
    width: int = 0
    #: "" for a token/atom kernel, "pair" or "atom" for a `level=both` one. A both-level kernel
    #: keys on ROWS, so its two sides are different buckets at the same length -- atom A=256 is
    #: 256 rows, pair L=256 is 65,536 -- and the side has to be said, not inferred.
    side: str = ""
    #: HEAD COUNT, for the kernels whose bucket carries `H` alongside `HEAD_DIM`.
    #:
    #: One width is not enough for those. `triangle_attention` packs `pack(shape_key, H=H,
    #: HEAD_DIM=D)` -- two independent axes -- and its driver derived the second from the first
    #: (`H = 128 // D`), which ties them: D=64 forces H=2 and D=16 forces H=8. `cases()` declares
    #: (n_head, d_hidden) of (4,128) (8,128) (4,256) (16,256), i.e. (H, HEAD_DIM) of (4,32) (8,16)
    #: (4,64) (16,16) -- and (4,64) and (16,16) are exactly the two that ratio cannot produce.
    #: `dev audit --replay` asked for both at every length and no unit had ever built either, and
    #: no amount of rebuilding could: the driver had no way to be told.
    #:
    #: 0 means "the driver decides", which is every op that does not key on a head count.
    heads: int = 0
    generation: str = ""

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
        from miniworld_engine.autotune.cache import _levels
        from miniworld_engine.autotune.shape_key import atom_key
        level = _levels().get(self.op)
        if level == "both":
            return both_key(self.length)
        if level == "atom":
            return atom_key(self.length)
        return self.length

    @property
    def label(self) -> str:
        tag = f" {self.side}" if self.side else ""
        w = f" D={self.width}" if self.width else ""
        return f"{self.op}[{self.dtype}]{tag} L={self.length}{w}"

    @property
    def stem(self) -> str:
        tag = f"-{self.side}" if self.side else ""
        w = f"-D{self.width}" if self.width else ""
        # The spare axis is part of the identity, like the width: two units differing only in it
        # write DIFFERENT buckets -- (H=4, HEAD_DIM=64) and (H=8, HEAD_DIM=64) are two keys -- so a
        # shared stem would have one shard overwrite the other and the sweep would silently build
        # half of what it planned. `test_the_op_sweep_drives_more_than_one_width` pins exactly this.
        h = f"-H{self.heads}" if self.heads else ""
        return (f"op-{self.op}-{self.dtype}{tag}-L{self.length}{w}{h}"
                f"{'-plan' + self.generation if self.generation else ''}")

    def cmd_args(self) -> list[str]:
        args = ["--op", self.op, "--dtype", self.dtype, "--length", str(self.length)]
        if self.heads:
            args += ["--heads", str(self.heads)]
        if self.width:
            args += ["--width", str(self.width)]
        return args + (["--side", self.side] if self.side else [])

    #: registry.csv's dtype names -> the driver env's. `MINIWORLD_DRIVER_DTYPE` takes the short
    #: spelling because that is what the drivers' own docstrings and `drivers.DTYPE_MODE` use.
    _DRIVER_DTYPE: ClassVar[dict[str, str]] = {"bfloat16": "bf16", "float32": "fp32"}

    def env(self) -> dict[str, str]:
        # The drivers read these at IMPORT time, like MINIWORLD_SHAPE_MODE -- their shape constants
        # are module-level and the kernels reach them through helpers that close over them, so a
        # per-call override would have to reach inside every driver module. Per-process does not.
        env = {"MINIWORLD_DRIVER_LENGTH": str(self.length)}
        if self.width:
            env["MINIWORLD_DRIVER_WIDTH"] = str(self.width)
        if self.heads:
            env["MINIWORLD_DRIVER_HEADS"] = str(self.heads)
        if self.side:
            env["MINIWORLD_DRIVER_SIDE"] = self.side
        # THE DTYPE, which used to be missing. registry.csv's `dtypes` column splits an op into one
        # unit per declared precision, and `--dtype` is on the child's command line so the unit is
        # reproducible from it -- but the op path never read that argument, and this never set the
        # variable the drivers actually consult. So every unit built at `drivers.DTYPE_MODE`'s
        # default, bf16, and the fp32 half of the sweep was 360 of `build trunk`'s 1,269 units
        # doing exactly what the bf16 half had already done. No fp32 entry has ever been built for
        # any kernel, on any card.
        try:
            env["MINIWORLD_DRIVER_DTYPE"] = self._DRIVER_DTYPE[self.dtype]
        except KeyError:
            # `dtypes` aliases fp16 -> float16 in op_units, and drivers.DTYPE_MODE takes bf16 or
            # fp32 only. Without this the mismatch surfaces as a bare KeyError out of env.update,
            # after the build has started, naming neither the kernel nor the column.
            msg = (f"{self.op}: registry.csv declares dtype {self.dtype!r}, which the drivers "
                   f"cannot build -- MINIWORLD_DRIVER_DTYPE is bf16 or fp32. Fix the dtypes column.")
            raise ValueError(msg) from None
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
        dims, length, dt = case.dims[0], case.lengths_for(0)[0], case.dtypes[0]
        dt_name = str(dt).replace("torch.", "")
        impls = [i for i in case.impls
                 if sm is None or build_matrix.allows(sm, case.name, i, dt_name)]
        if not impls:
            continue      # nothing this card can build; units() drops it too, so it is not a defect
        try:
            module = case.factory(dims, 0.0, impls[0], dt)
            module.eval()
            with torch.no_grad():
                module(*case.input_args(0, length, dt, train=False))
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


def op_units(only: set[str] | None = None, config_dir: Path | None = None, driver_widths=None,
             stack: str | None = None) -> list[OpUnit]:
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

    ``stack`` narrows the sweep to one half of the model. registry.csv's ``stack`` column says
    which side of the model each kernel is launched from -- ``trunk`` for the Pairformer / MSA /
    template stack, ``diffusion`` for the DiT stack (token_dit, atom_dit and the SWA atom
    transformer), ``both`` for a kernel each side launches. Asking for one INCLUDES ``both``: a
    kernel the trunk launches has to be tuned for a trunk build whether or not the diffusion side
    launches it too, and building one extra kernel costs time while missing one costs a cache miss
    in production.

    dtype is declared too, in registry.csv's ``dtypes`` column: token kernels are bf16, atom and
    both are bf16|fp32. This used to emit bfloat16 for everything, so the fp32 half of 66 kernels
    was never driven -- and the coverage check counted (op, bucket) without dtype, so it reported
    527/527 and missing_pairs=0 over a cache that held one of the two declared precisions.
    """
    import csv

    from miniworld_engine.autotune.shape_key import (
        ATOM_SHAPES,
        BOTH_PAIR_LENGTHS,
        DIT_ATOM_LENGTHS,
        DIT_TOKEN_LENGTHS,
        SHAPES_BY_LEVEL,
        TOKEN_SHAPES,
    )

    if stack is not None and stack not in ("trunk", "diffusion"):
        # Returning the `both` rows for an unrecognised name -- a third of the sweep -- and having
        # the CLI report it as a build is worse than not running.
        msg = f"op_units(stack={stack!r}): the halves are 'trunk' and 'diffusion'"
        raise ValueError(msg)

    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    # The WIDTHS to drive each op at. The shape key carries the whole shape now (plan.md G5), so a
    # sweep that varies only the length tunes one width per op -- whichever the driver happens to
    # build -- and every other width the model uses falls back to the grid at runtime. Measured on
    # an A6000: 363 such lookups across 42 of 91 ops
    # (docs/records/cache-coverage-replay-a6000.md), which is precisely what the module pass was
    # added to reach.
    #
    # These are the model's own widths, not DIM_BUCKETS. `cases()` states them -- d_pair 128,
    # d_single 384 -- and driving the two the model actually runs costs 2x, where the six declared
    # buckets would cost 6x to tune four widths nothing asks for. `dim_bucket` still refuses
    # anything outside DIM_BUCKETS at runtime, so an unlisted width is a loud miss, not a silent
    # one.
    # The WIDTH ladder per class. `level` says which LENGTH axis a kernel is on; `width` says which
    # side's channel width it sees, and the two are independent -- a kernel can be token-level and
    # still be driven at several d_pair values. Ladders come from what the model runs: d_pair walks
    # 128/256/512, d_single walks those plus 384 and 768 (`cases()` -- adaptive_layernorm at 768,
    # attention_pair_bias and augmented_attention at d_single 384 and 768).
    #
    # A kernel used on BOTH sides gets the union, because it meets both.
    #: An atom activation is (B, A, ATOM_D_MAX): the channel width is fixed, so a unit driving the
    #: atom STREAM has exactly one width and sweeping more of them tunes shapes production never
    #: presents. `width=atom` is how a row says it only ever sees that stream.
    ATOM_WIDTH = 128
    #: The SINGLE ladder is the three widths the model declares and no others: all four
    #: model configs (debug/small/medium/large) set d_single_atom=128, d_single=384,
    #: d_single_token=768 -- they differ in block counts, not widths -- and those are AlphaFold-3's
    #: c_atom, c_s and c_token. It used to also carry 256 and 512, which no config presents.
    #: The PAIR ladder keeps 128/256/512: d_pair is 128 in every config, and 256/512 are the
    #: headroom to sweep it.
    #
    #: The two halves are named separately because the headroom is not free and nothing said what
    #: it costs. PRESENTED are the widths the model actually runs -- AlphaFold-3's c_atom (128),
    #: c_s (384) and c_token (768), with d_pair at 128. HEADROOM is 256 and 512 on the pair side,
    #: kept so a config that widens d_pair finds a tuned cache instead of a miss.
    #
    #: Measured on the shipped registry: of 2,079 (op, dtype, side, length, width) units in
    #: `build all`, **514 -- 25% -- are at 256 or 512**, widths no model config presents. That is
    #: 25% of every full build's GPU time spent on shapes nothing asks for today. (It was 674 of
    #: 1,827, 37%, before the seven `head_dim` / `pair_bidir` rows stopped drawing from this
    #: ladder: their widths are derived, so 256 and 512 are no longer rungs they walk.) Whether to keep
    #: paying it is a decision about the future, not a fact about the code, so it is written here
    #: as a decision rather than buried in a literal. Dropping HEADROOM_PAIR makes `build all`
    #: 1,565 units; the cost of being wrong is a cache miss (a warning and a heuristic subset,
    #: `cache._miss`), not a failure.
    #: Literal, not `(ATOM_WIDTH,)`: a test reads these two declarations straight out of the
    #: source so the split cannot be folded away, and it can only read literals.
    #: Which `cases()` dims name each stream's channel width. This is the ONE mapping left, and it
    #: is knowledge -- `d_pair` is the pair stream and `d_single` the single stream, and no rule
    #: derives that. Everything downstream is arithmetic over `cases()`, which is why the ladders
    #: below are no longer tuples anyone edits.
    #:
    #: They were, and every miss this repository has recorded came from one drifting: TOKEN_SHAPES
    #: stopped at 512 while cases() ran to 1024; MSA_WIDTHS held 64 while cases() declares d_msa 64
    #: AND 128; the pair side had no 384, which the DiT hands the shared layernorms. Each was found
    #: by `dev audit --replay` -- a card, a finished cache, half an hour -- and patched by adding
    #: another tuple.
    STREAM_DIMS = {
        "atom": ("d_atom",),
        "pair": ("d_pair", "d_hidden_tri_multi"),
        "single": ("d_single", "d_cond"),
    }

    def _from_cases(*names: str) -> tuple[int, ...]:
        """Every value `cases()` declares under any of these dims names."""
        return tuple(sorted({v for c in cases() for d in c.dims
                             for k, v in d.items() if k in names and isinstance(v, int)}))

    #: The pair stream, from the model. `HEADROOM_PAIR` was (256, 512) beside a PRESENTED of
    #: (128,) on the argument that 26 bench.yaml files sweep d_pair 128/256/512 -- which is exactly
    #: what `cases()` declares, so the split had nothing left to say and is gone.
    PRESENTED = {"atom": (ATOM_WIDTH,),
                 "pair": _from_cases(*STREAM_DIMS["pair"]) or (128,),
                 "single": _from_cases(*STREAM_DIMS["single"]) or (128, 384, 768)}
    #: Widths the MSA stack presents that no other stream does. `cases()` builds
    #: `msa_pair_weighted_averaging` and `outer_product_mean` at `d_msa=64`, and both dispatch into
    #: the SHARED layernorm/transition kernels -- so 64 arrives at a `level=both` kernel as a
    #: channel width with no ladder rung of its own. `dev audit --replay` measured it directly:
    #: `layernorm_fwd_saveact_triton` missing `(rows=2048, N=64)` and `(4096, 64)`.
    MSA_WIDTHS = _from_cases("d_msa") or (64,)

    def _case_lengths() -> tuple[int, ...]:
        """Every length `cases()` runs, which is the WORK list.

        `TOKEN_SHAPES` and `DIT_TOKEN_LENGTHS` are KEY sets -- what `atom_key` floor-clamps into,
        deliberately disjoint so one clamp can serve both sides -- and they were also used as the
        work list. They are not the same thing and they did not agree: the key sets stop at 512 and
        768 while `cases()` runs to 1024, so a length production runs had no unit at all.
        """
        return tuple(sorted({int(L) for c in cases() for L in c.lengths}))

    CASE_LENGTHS = _case_lengths()
    #: Widths the TRANSITION's expansion presents, for the same reason MSA_WIDTHS exists: a shared
    #: kernel meets them and no stream ladder carries them. The transition expands its hidden width
    #: by `n` (4) before the SwiGLU, so every kernel downstream of that expansion sees `n*d_hidden`,
    #: and `cases()` builds the transition at d_hidden 128, 256 and 384 -- 512, 1024 and 1536.
    #:
    #: `dev audit --replay` measured all three consequences on an A6000 at once: the LN kernels the
    #: expanded activation flows through (`layernorm_fwd_saveact_strided_triton`,
    #: `layernorm_bwd_split_triton`, `layernorm_bwd_atomic_strided_triton`) each asked for a width
    #: 1024 they had no rung for, and `transition_expand_swiglu_triton` /
    #: `transition_bwd_swiglu_recompute_triton` -- which fold ND into their key -- asked for 1024
    #: and 1536 while their driver built 512 at every declared width.
    #:
    #: 512 is already on the pair ladder; it is listed anyway so the set says what it is rather
    #: than relying on an overlap that a change to HEADROOM_PAIR would silently break.
    EXPANDED_WIDTHS = (512, 1024, 1536)
    #: ...and only for the families that ARE the shared path. A `level=both` row is shared in the
    #: sense that both streams reach it; these three families are the normalisation and transition
    #: kernels that every OTHER family dispatches into, which is why they see widths belonging to
    #: no stream of their own. `gated_projection` and the two rmsnorm families are `level=both`
    #: too and asked for none of these -- their callers hand them their own stream's width -- so
    #: giving them the extras would be 300-odd units for buckets no measurement has requested.
    SHARED_EXTRA_FAMILIES = frozenset({"layernorm", "layernorm_linear", "transition"})
    #: The token side of a DiT family. 128 is NOT here: it is d_single_atom, the atom side's width,
    #: and pairing it with a token count builds a shape no config presents. 384 (d_cond, AF3's c_s)
    #: and 768 (d_single_token, c_token) are what the token blocks run.
    #:
    #: 512 WAS here, as headroom "on the same argument as HEADROOM_PAIR". The argument does not
    #: transfer. HEADROOM_PAIR earns its place because 26 `benchmarks/**/bench.yaml` files sweep
    #: `d_pair_values: [128, 256, 512]`, so a missing 512 would push published bench points onto
    #: the heuristic subset. NOTHING sweeps d_single: `builder.cases()` presents d_single 384/768
    #: and d_cond 128/384/768, and no bench.yaml has a d_single axis at all. So this rung was
    #: 17% of the whole build spent on a width no measurement and no model config asks for.
    DIT_TOKEN_WIDTHS = _from_cases("d_single", "d_cond") or (384, 768)
    #: Widths that are a kernel's own axis DERIVED from d_pair, not d_pair itself.
    #:
    #: `Case`'s docstring states the rule this implements: "Dimensions are declared with the
    #: module's OWN parameter names -- d_pair, d_single, d_hidden, n_head -- not as a single
    #: anonymous width. They are not interchangeable: a kernel's cache bucket is built from the
    #: constexprs it was launched with." The `width` column had only stream names, so a kernel
    #: whose bucket carries a DERIVED axis had no way to say so and was declared `pair`, which
    #: drives the wrong numbers entirely.
    #:
    #: Both entries are the set `cases()` already declares, read off the same Case rows a
    #: `dev audit --replay` drives -- so the declaration and the measurement cannot disagree:
    #:
    #:   head_dim: `Case("triangle_attention_heads")` sweeps (n_head, d_hidden) =
    #:     (4,128) (8,128) (4,256) (16,256), i.e. head dims 32, 16, 64, 16. The driver drove
    #:     `ragged(32)` alone, and `--replay` missed exactly 16 and 64, at every length.
    #:
    #:   pair_bidir: `front_bwd_dW` keys on H, the PER-SIDE hidden width, and says so --
    #:     "Din = WL.shape[0] (= d_pair); may differ from H (bidirectional)". A bidirectional
    #:     trimul meets 2*d_pair, so the ladder is the pair ladder doubled. Its driver says
    #:     "Square single-dir (H = Din = D)" in its own docstring: it drives one half of what the
    #:     row meets, and `--replay` missed H=1024 (2 x 512) at three lengths.
    #: (n_head, head_dim) PAIRS, not head dims. The two are independent axes of the same key --
    #: `pack(shape_key, H=H, HEAD_DIM=D)` -- so a ladder of head dims alone leaves the driver to
    #: invent the head count, and `H = 128 // D` is what it invented: it can produce (4,32) (8,16)
    #: (2,64) (16,8) and nothing else. `cases()` declares (4,32) (8,16) (4,64) (16,16); the last
    #: two are unreachable from that ratio at any width, which is what `--replay` measured.
    #:
    #: Read off `cases()` rather than listed, so the two cannot drift.
    HEAD_PAIRS = tuple(sorted({(d["n_head"], d["d_hidden"] // d["n_head"])
                               for c in cases() if c.name.startswith("triangle_attention")
                               for d in c.dims
                               if d.get("n_head") and d.get("d_hidden")
                               and d["d_hidden"] % d["n_head"] == 0}))
    HEAD_DIMS = tuple(sorted({d for _h, d in HEAD_PAIRS})) or (16, 32, 64)
    #: (d_hidden, d_pair) pairs for the gate-out GEMM, whose bucket is `pack(..., N=N, DH=DH)`:
    #: DH is the contraction and N the output width, and they are independent. Its driver took both
    #: from one `driver_width`, so it could only ever build the DIAGONAL -- (128,128) (256,256)
    #: (512,512) -- while `cases()` declares gated_projection at (hd, d) of (128,128) and
    #: (256,128). `--replay` asked for (256,128) and (512,256) and neither was reachable.
    GATE_OUT_PAIRS = tuple(sorted({(d["hd"], d["d"])
                                   for c in cases() if c.name == "gated_projection"
                                   for d in c.dims if d.get("hd") and d.get("d")}))
    #: (d_hidden, d_cond) for the DiT families. Their kernels key on both -- `pack(..., NX=NX,
    #: NC=NC)` and the `(D, ND)` / `(DC, ND)` variants -- and `drivers/conditioned_transition.py`
    #: derived the second from the first: `_DC_BASE = 384 if _D_BASE > 128 else 128`. That yields
    #: (128,128) and (768,384) and nothing else, which happens to be what `cases()` declares -- but
    #: it welds the pair to the WIDTH, so each pair was only ever built at the lengths that width's
    #: ladder rung carries. `--replay` asked for (128,128) at 256..768 and had it only at 1024+,
    #: and for (384,768) at 1024 and had it only at 128..768.
    DIT_PAIRS = tuple(sorted({(d["d_hidden"], d["d_cond"])
                              for c in cases()
                              if c.name in ("adaptive_layernorm", "conditioned_transition")
                              for d in c.dims if d.get("d_hidden") and d.get("d_cond")}))
    PAIR_BIDIR = tuple(sorted({2 * w for w in PRESENTED["pair"]}))
    #: One entry PER LINE, like LADDER: `test_the_builders_ladder_defines_exactly_these` reads the
    #: vocabulary straight out of this source and takes the first quoted name on each line, so a
    #: one-line dict declares only its first class to the test that exists to catch a typo.
    #: The rungs to drive when a row's `key_axis` names an axis that is NOT a stream width.
    #:
    #: Keyed by the axis name the KERNEL uses -- `pack(shape_key, H=H, HEAD_DIM=D)` -- not by a
    #: name coined here. The `width` column keeps saying which stream the kernel sees, which is
    #: what it has always meant; `key_axis` says which constexpr the bucket carries, and only that
    #: decides the ladder. Reading a registry row no longer requires knowing that `pair_bidir` was
    #: a word this file made up for `H`.
    AXIS_LADDERS = {
        "HEAD_DIM": HEAD_DIMS,      # d_hidden // n_head
        "H": PAIR_BIDIR,            # the per-side hidden width; 2 * d_pair on a bidirectional trimul
        "ND": EXPANDED_WIDTHS,      # n * d_hidden, the transition's expanded width
    }
    assert PRESENTED["atom"] == (ATOM_WIDTH,), "the atom stream has one width and it is ATOM_WIDTH"
    LADDER = {"atom": PRESENTED["atom"],
              "pair": PRESENTED["pair"],
              "single": PRESENTED["single"],
              # `both` is the fallback for a row with no side; the MSA widths reach a
              # both-level kernel through `_widths("atom")`, which is the side their row count
              # lands on -- not through this entry, which `_widths` never reads for such a row.
              # The union a SHARED kernel meets: every stream `cases()` declares, because every
              # family dispatches into these and hands them its own width. NOT the derived axes --
              # `EXPANDED_WIDTHS` is what the transition's own kernels key on after expanding, and
              # a shared layernorm is handed the width BEFORE that. Folding it in here added 1536
              # to every shared row, which no replay has ever asked for.
              # MSA_WIDTHS is NOT here: the MSA stack reaches these kernels through their ROW
              # count, on the non-pair side, and `_widths("atom")` already carries it. Putting it
              # in the union gave the pair and token sides a d_msa rung as well, which is an
              # activation neither stream has.
              "both": tuple(sorted(set(PRESENTED["pair"] + PRESENTED["single"])))}
    #: read once, not once per row -- 91 rows would open the same file 91 times.
    evidence = width_evidence.load()
    out = []
    for r in csv.DictReader(reg.open()):
        if r["backend"] != "triton" or not (r["driver"] or "").strip():
            continue
        # `developed` is a HAND-MAINTAINED judgement, not a rule derived from the benchmark tables,
        # and it has to be: bias_only_attention loses on time on every committed card and uses half
        # the memory at every length, so a rule reading either number alone gets it wrong. Every
        # `no` carries its reason in kernels/undeveloped.csv, which a test pins.
        if (r.get("developed") or "yes").strip() == "no":
            continue
        if only and r["kernel"] not in only:
            continue
        if stack and r.get("stack") not in (stack, "both"):
            continue
        if config_dir is not None and not (config_dir / f"{r['kernel']}.csv").is_file():
            continue          # this config set declares no grid for it
        # A `level=both` kernel is TWO work lists, not one. It keys on rows (shape_key.BOTH_ROWS),
        # so a pair L and an atom A of the same value are different buckets -- pair L=256 is
        # 65,536 rows, atom A=256 is 256 -- and driving one length list picks a side per length
        # and never builds the other. 4 pair + 6 atom = 10 buckets, which is exactly BOTH_ROWS.
        # Before this, 8 units covered 8 of the 10 and two of those 8 were the wrong side.
        # stripped: an unstripped "single " matches no ladder key and falls through to the union,
        # which is the exact failure test_width_column_selects_a_ladder exists to stop.
        klass = (r.get("width") or "both").strip() or "both"
        #: The axis this kernel's bucket carries, from registry.csv. Blank on a row whose key
        #: carries a plain stream width, which is most of them. `HEAD_DIM|H` names both axes of a
        #: two-axis key; the FIRST is the one whose ladder decides the widths, the rest ride in the
        #: unit's spare slot.
        _axes = [a for a in (r.get("key_axis") or "").split("|") if a]
        _axis = _axes[0] if _axes else ""
        #: ...and whether that axis is what the DRIVER asks `driver_width` for. Three families were
        #: changed to do that, because their kernels key on a quantity no stream produces and their
        #: drivers had pinned it: triangle_attention's head dim, the bidirectional trimul's
        #: per-side hidden width, the transition's expanded ND. Everywhere else the driver derives
        #: the axis from the stream width, so the stream ladder is still the right one.
        _axis_drives = bool(_axis) and r["family"] in ("triangle_attention", "trimul_inproj",
                                                       "transition")
        #: A row that IS the shared normalisation/transition path: every other family
        #: dispatches into it, so it meets their widths and not only its own stream's.
        _shared = r["level"] == "both" and r["family"] in SHARED_EXTRA_FAMILIES
        if r["level"] == "both":
            # WHICH sides comes from the row, not from the level. `level=both` says the kernel is
            # keyed on rows and driven per side; it does not say which streams the model runs it
            # on, and assuming pair+atom was wrong in both directions for the transition family.
            # AlphaFold-3's Transition is applied to the pair representation, to the single
            # representation at token granularity (`pairformer.transition_single`, d_single) and to
            # the MSA stack -- and never to atoms. So it was built at six atom lengths it never
            # sees and at none of the token shapes it does. Rows with no `sides` cell keep the old
            # pair+atom pair, which is right for layernorm (the DiT normalises atoms) and for
            # gated_projection until someone traces it.
            want = [x for x in (r.get("sides") or "pair|atom").split("|") if x]
            # The token side of a SHARED row runs at the lengths `cases()` runs, not at
            # TOKEN_SHAPES. `--replay` asked `layernorm_fwd_strided` and `_bwd_atomic_strided` for
            # (rows=1024, N=384): 1024 rows of a 384-wide TOKEN activation, which is the DiT token
            # track at length 1024 being normalised by the shared kernel. TOKEN_SHAPES stops at
            # 512, so the only rung at 1024 was the ATOM side -- a different activation, at the
            # atom width -- and the key was never built.
            # The WORK list is what `cases()` runs. TOKEN_SHAPES is the key set and stops short
            # of it; using it here is what left the shared layernorms with no unit at 768 or 1024.
            _tok_shared = tuple(sorted(set(TOKEN_SHAPES) | set(CASE_LENGTHS)))
            per = {"pair": [("pair", L) for L in BOTH_PAIR_LENGTHS],
                   "atom": [("atom", A) for A in ATOM_SHAPES],
                   "token": [("token", N) for N in _tok_shared]}
            sided = [u for side in want for u in per[side]]
        elif r["level"] == "atom":
            # Also two work lists -- see shape_key.DIT_TOKEN_LENGTHS. `level=atom` says which key
            # function; it does not say the kernel only ever sees ATOM COUNTS.
            #
            # This used to be gated on `klass == "single"`, on the argument that 384/768 are widths
            # only the token stream has. True about WIDTH, wrong about LENGTH -- and the two are
            # independent axes here. `cond_transition_fwd_b2b_triton` is `width=atom` (dispatch
            # routes it only at d <= 128) and was therefore driven at atom lengths ONLY, while the
            # 128/128 DiT block it serves is built at every length the sweep runs. Measured by
            # `dev audit --replay`: identical widths (DC=128, K=128, ND=256), built at L in
            # 1024..8192, asked for at L in 256..768. The width class still decides the WIDTH --
            # `_widths` reads `klass`, so a `width=atom` row stays pinned to 128 on both ladders.
            # The token side runs at every length `cases()` runs these families at, which is not
            # DIT_TOKEN_LENGTHS. That list stops at 768 because it is a KEY set -- `atom_key`
            # floor-clamps into it and the two side lists are disjoint so one clamp can serve both
            # -- and it was reused here as a WORK list. `cases()` builds adaptive_layernorm,
            # conditioned_transition and augmented_attention at 256..1024, so a token launch at
            # 1024 keys to `atom_key(1024)`, and the build drove that length on the ATOM side only,
            # at the atom width. `--replay` asked augmented_attention for (H=16, HEAD_DIM=24) and
            # (16, 48) at base 1024 -- token widths -- and had them only at 128..768.
            #
            # Driving both sides at 1024 collides with nothing: the key carries the widths too, and
            # the atom unit there is width 128 while the token units are 384/768.
            _tok = tuple(sorted(set(DIT_TOKEN_LENGTHS) | set(CASE_LENGTHS)))
            sided = ([("token", L) for L in _tok]
                     + [("atom", A) for A in DIT_ATOM_LENGTHS])
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
        # The ladder is per UNIT, not per op, because the width depends on which SIDE the unit
        # drives. An ATOM activation's channel width is fixed -- (B, A, 128), `interface.ATOM_D_MAX`
        # -- so sweeping 384 or 768 there builds a shape the model never presents. A token/pair
        # activation's width is exactly what varies. A `level=both` kernel therefore splits inside
        # one op: its pair units walk the ladder, its atom units are 128 and only 128.
        # The width comes from the `width` COLUMN, never from `level`. The two say different things:
        # `level` picks the key function (which length ladder a bucket is drawn from), while `width`
        # says which stream's channel width the kernel sees. This used to read
        # `atom_only = level == "atom"` -> one width, 128, and that silently capped three families
        # that key on `atom_key` but run on BOTH sides of the model. The model builds one
        # DiffusionTransformer block class 24 times at d_single=768/d_cond=384 (`token_dit`) and 3
        # times at 128/128 (`atom_dit`), and adaln, ConditionedTransition and
        # AugmentedAttentionPairBias are all constructed inside it -- adaln has no width guard at
        # all. So 768 was 24 of the model's 27 blocks and no unit ever built it: production opened a
        # drawer the builder never filled. Rows genuinely pinned to the atom width say `width=atom`
        # (cond_transition's b2b pair, which `dispatch.ATOM_D_MAX` routes only at d <= 128).
        def _widths(side: str, _k=klass, _lvl=r["level"], _shared=_shared,
                    _axis=_axis, _axis_drives=_axis_drives) -> tuple:
            if driver_widths:
                return tuple(driver_widths)
            # The axis ladder REPLACES the stream ladder only where the driver takes that axis as
            # its width -- `driver_width` returns the head dim itself, or the per-side hidden
            # width, or ND -- because then no stream rung is the right number and no side changes
            # that.
            #
            # It does NOT replace it just because the row declares an axis. `augmented_attention`
            # and `cond_transition` fold `HEAD_DIM`/`ND` into their keys too, but their drivers
            # DERIVE those from the stream width they are handed, and that width differs per side:
            # augmented_attention's head dim is 768/16 = 48 on the token side and 128/n_head on the
            # atom side. Handing both sides one axis ladder gave a `level=atom` row head dims
            # 16/32/64 on its ATOM side, which three tests reject and which the model never runs.
            # Their `key_axis` is a declaration of what the bucket carries, which is what makes the
            # unit's second axis and the driver's derivation checkable; it is not a ladder.
            if _axis in AXIS_LADDERS and _axis_drives:
                return AXIS_LADDERS[_axis]
            # A `level=both` row is driven once per SIDE, and the side names the stream outright,
            # so it decides the ladder: its atom units are a real atom activation (128 and only
            # 128) and its pair units a real pair one, whatever the row's own class says. The
            # class ladder is for rows with no side -- `level=token`/`atom` -- where the column is
            # the only thing that knows which stream the kernel sees.
            #
            # So the `width` cell is never READ on a `level=both` row. It is still required to say
            # `both` there, and a test pins the biconditional: unread is not free to be wrong, and
            # `level=both,width=pair` would be a row contradicting itself with nothing to catch it.
            if side == "atom":
                if _lvl != "both":
                    # A DiT row (`level=atom`) has a real atom stream, and it is 128 and only 128
                    # -- `test_a_dit_family_is_built_on_both_streams` pins that. The MSA widths
                    # below are for `level=both` rows only; letting them through here paired an
                    # atom length with a width the DiT never presents, for 17 families at once.
                    return (ATOM_WIDTH,)
                # ATOM_WIDTH plus the MSA widths. The non-pair side of a `level=both` kernel is
                # not only the atom stream: the MSA stack borrows these same shared LayerNorm /
                # transition kernels, and an MSA activation is (B, n_msa, n_token, d_msa) whose
                # ROW COUNT (n_msa*n_token) lands in exactly these buckets while its width is
                # `d_msa`. `cases()` builds `msa_pair_weighted_averaging` and `outer_product_mean`
                # at d_msa 64, and `dev audit --replay` measured the consequence directly --
                # `layernorm_fwd_saveact_triton` asked for (rows=2048, N=64) and (4096, 64) and
                # had neither, because 64 was a rung on no ladder at all.
                return tuple(sorted({ATOM_WIDTH, *MSA_WIDTHS}))
            if side == "pair":
                # Plus the transition's expanded widths on a `level=both` row. Those rows are the
                # SHARED layernorm/transition kernels, and the pair-side transition hands them
                # `n*d_hidden` -- 1024 is what `--replay` asked three of them for and none had.
                # A `level=pair`/`token` row keeps the plain pair ladder: it is one family's own
                # kernel, and the expansion is not on its stream.
                if _lvl == "both" and _shared:
                    # A shared kernel is handed whichever stream's width its CALLER has, so its
                    # pair side is not d_pair's ladder alone. This used to be a hand-written
                    # `SHARED_EXTRA["pair"] = (384,)`, added because `--replay` asked for exactly
                    # that; the union of the streams `cases()` declares says it without the list.
                    return LADDER["both"]
                return LADDER["pair"]
            if side == "token":
                # The class decides the WIDTH even here -- the comment above says so, and this
                # line used not to honour it. A `level=atom, width=atom` row (cond_transition's
                # two b2b kernels and rope_fwd) walks the token LENGTH ladder because the 128/128
                # DiT block runs at token lengths, but its channel width is still 128: the row
                # says `width=atom` precisely because `dispatch.ATOM_D_MAX` routes it only at
                # d <= 128. Returning the token widths gave those three rows 30 units each at
                # 384/512/768 -- shapes their own registry row says they never see, which build
                # and then store nothing.
                if _k == "atom":
                    return (ATOM_WIDTH,)
                if _shared:
                    return LADDER["both"]
                return DIT_TOKEN_WIDTHS
            return LADDER.get(_k, LADDER["both"])

        # A ladder is a guess that a different width is a different bucket. For 17 ops it is not:
        # `dev buckets` launched every (op, width) the plan contains and read the key back, and
        # they file every declared width into ONE. Building the other rungs re-times a bucket a
        # sibling unit already timed and overwrites it -- 274 of 2,079 units. Drop them here, where
        # the plan is made, so the work is never scheduled rather than skipped after the fact.
        #
        # Measured, not declared: the key is a packed integer built from constexprs that do not
        # exist until the launcher builds them (see `_cache_answers`), so a registry column saying
        # this would be a guess with nothing to check it against. An op the evidence does not
        # cover keeps its full ladder.
        # The LARGEST of a collapsed group, not the first. When two widths share a bucket they
        # share a winner, so exactly one of them decides what production gets at both -- and the
        # repo has already measured which one to pick: `_ROWS_SATURATE` exists because tuning the
        # adaLN kernels at 512 rows instead of their real row count chose a config that costs
        # production 1.53x. Tuning at the small end of a saturating bucket is the same mistake on
        # the other axis. Before this the winner was whichever unit the pool happened to finish
        # last, which was neither deterministic nor chosen.
        def _distinct(widths: tuple, _op=r["kernel"]) -> tuple:
            if len(widths) < 2 or not width_evidence.collapses(_op, widths, evidence):
                return widths
            return (max(widths),)

        # A `head_dim` row's bucket carries TWO axes, so its units carry the pair. Everything else
        # gets heads=0, which leaves the driver's own derivation alone.
        # HEAD_PAIRS is triangle_attention's (n_head, head_dim) list, so this may only fire on a
        # row whose driver takes the head dim as its width. `augmented_attention` declares
        # `key_axis=HEAD_DIM|H` too and DERIVES both from d_single; gating on the axis alone
        # intersected its single ladder (128/384/768) with triangle_attention's head dims and left
        # it with no units at all -- five ops silently dropped from the sweep.
        _pairs = (_axis == "HEAD_DIM" and _axis_drives)
        #: The gate-out GEMM's two widths are independent and its driver tied them, so these two
        #: rows carry the (d_hidden, d_pair) pair the same way a head_dim row carries (H, D). The
        #: SECOND number rides in `heads` -- it is the unit's spare axis, not a head count here,
        #: and `drivers/bias_only_attention.py` reads it under its own name.
        _gate_out = r["kernel"] in ("gated_projection_bwd_dx_triton",
                                    "gated_projection_gate_gemm_triton")
        #: The adaLN / conditioned-transition rows: their kernels key on d_hidden AND d_cond, and
        #: the driver derived the second from the first, which welded the pair to the width. The
        #: pair rides the same way as the others -- width is d_hidden, the spare axis is d_cond.
        _dit_pair = r["family"] in ("adaln", "conditioned_transition")

        def _axes(side: str, _pairs=_pairs, _gate_out=_gate_out, _dit_pair=_dit_pair) -> list:
            """(width, spare axis) per unit. The LADDER still decides the widths -- the pair only
            says what the second axis is for a width the model actually declares.

            Filtering by the ladder is not decoration. `_widths` is where the side and the registry
            `width` column are honoured, and returning a pair list outright ignored both: an
            atom-level row got a token width at an atom length, which three tests forbid and which
            `_widths`' own comment calls out ("gave those three rows 30 units each at 384/512/768
            -- shapes their own registry row says they never see").
            """
            ws = _distinct(tuple(_widths(side)))
            if _pairs:
                # only the (H, D) combinations `cases()` declares, not their cross product: the
                # cross would build head counts the model never pairs with that head dim.
                return [(d, h) for h, d in HEAD_PAIRS if d in ws]
            if _gate_out:
                pair = dict(GATE_OUT_PAIRS)
                return [(w, pair.get(w, 0)) for w in ws]
            if _dit_pair:
                pair = dict(DIT_PAIRS)
                return [(w, pair.get(w, 0)) for w in ws]
            return [(w, 0) for w in ws]

        out.append([OpUnit(op=r["kernel"], length=length, dtype=dt, side=side, width=w, heads=h)
                    for dt in dtypes for side, length in sided
                    for w, h in _axes(side)])
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
        if case.rows:
            from miniworld_engine.autotune import derive
            positions = {(r.stream, tuple(r.dims.items())): i for i, r in enumerate(case.rows)}
            for u in derive.units(list(case.rows), arch=sm):
                switch, value = u.option or ("", None)
                if switch and value is not None:
                    value = (float(value) if switch == "p_drop" else SWITCH_SETTINGS[switch][1](value))
                out.append(Unit(case.name, positions[(u.stream, u.dims)], u.length,
                                u.mode == "train", u.dtype, switch, value, u.impl, u.compute))
            continue
        # build/gpu_to_kernels/<sm>.csv, not a list trimmed in cases(): the sweep is shared across
        # cards, so dropping "cute" from Case.impls to protect sm_86 would also stop building it
        # on an H100, where it is the fastest path there is.
        allowed = {(i, str(d).replace("torch.", "")) for i in case.impls for d in case.dtypes
                   if sm is None or build_matrix.allows(
                       sm, case.name, i, str(d).replace("torch.", ""))}
        for di in range(len(case.dims)):
            # lengths_for(di), NOT case.lengths: a module's rows do not share a ladder. The union
            # would sweep the atom DiT (d_hidden=128) at token counts and the token DiT
            # (768/384) at atom counts -- twice the units, and every one of them a bucket
            # production never presents.
            for length in case.lengths_for(di):
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
        stat = shard.stat()
    except OSError:
        return False
    memo = _HAS_ENTRIES.get(shard)
    if memo is not None and memo[0] == (stat.st_size, stat.st_mtime_ns):
        return memo[1]
    answer = _read_has_entries(shard, stat.st_size)
    _HAS_ENTRIES[shard] = ((stat.st_size, stat.st_mtime_ns), answer)
    return answer


#: shard path -> ((size, mtime_ns), answer). One build asks this question three times over the same
#: directory -- the resume filter, `reclaim_orphans`, and the startup report.
#:
#: NOT because the scan is expensive. Measured on gpu04, gpu05 and the login node: 1,163 shards in
#: 2.4-3.8 s, i.e. 3.3 ms a file. An earlier version of this comment said 0.5 s a file and blamed a
#: 35-minute startup on it; that was a misreading of a py-spy sample. The 35 minutes were one
#: contended node (gpu03: 270 KB/s against 120 MB/s elsewhere, process in state D behind another
#: user's job on the same filesystem), and the identical build on gpu05 started in seconds.
#:
#: The memo and the `_has_entries` flag stay because doing three passes where one will do is worth
#: having, and because it is exactly what shortens the contended case -- not because the uncontended
#: one was ever slow.
_HAS_ENTRIES: dict[Path, tuple[tuple[int, int], bool]] = {}

#: A shard carrying no measurements is `{"_key_scheme": N, "_has_entries": false}`. Anything at or
#: below this cannot hold an entry, so its answer needs no read at all.
#:
#: It cannot be the WHOLE test: a shard whose every config scored +inf carries a full grid and no
#: entries -- 530 such configs in one A6000 unit -- and would read as finished on size alone.
_EMPTY_SHARD_BYTES = 96


def _read_has_entries(shard: Path, size: int) -> bool:
    """The uncached answer. Reads the head of the file when the shard declares the fact itself."""
    if size <= _EMPTY_SHARD_BYTES:
        return False
    try:
        with shard.open("rb") as fh:
            head = fh.read(256)
            # `dump_shard` writes `_has_entries` first for exactly this: the boolean is answerable
            # from the first few hundred bytes instead of megabytes. Shards written before that
            # field fall through to the full parse.
            if b'"_has_entries"' in head:
                return b'"_has_entries": true' in head or b'"_has_entries":true' in head
            fh.seek(0)
            data = json.loads(fh.read())
    except (OSError, ValueError):
        return False
    return any(isinstance(v, dict) and v.get("entries") for v in data.values())


def _shard_reusable(path: Path) -> bool:
    """A completed file can resume work only on its recorded GPU/compiler."""
    from miniworld_engine.autotune.shard import provenance_error

    if path.with_suffix(".failed").exists() or not _shard_has_entries(path):
        return False
    try:
        data = json.loads(path.read_text())
        return isinstance(data, dict) and provenance_error(data) is None
    except (OSError, ValueError, TypeError):
        return False


def _generation_for_work(config_dir: Path | None) -> str:
    """Keep stale shards and claims out of this GPU/source/grid generation."""
    import hashlib

    from miniworld_engine.autotune import plan
    from miniworld_engine.autotune.shard import provenance

    digest = hashlib.sha256()
    digest.update(plan.source_identity().encode())
    digest.update(json.dumps(provenance(), sort_keys=True).encode())
    if config_dir is not None:
        for path in sorted(config_dir.glob("*.csv")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _cache_ok_ops() -> set[str]:
    """Ops whose committed cache for THIS card the RUNTIME will actually serve.

    `dev cache-status`'s verdict is not that question. Its "OK" means "not stale enough to fail
    CI", and it says so in the reason: `config grid changed -- incremental build pending` and
    `build driver changed -- coverage may differ; rebuild the op` are both OK rows that still owe
    a build. On this repo today every one of the 51 OK A6000 rows carries the first one. The
    runtime reader is stricter -- a `config_space_hash` mismatch is a full miss (`cache.py`,
    "STALE (kernel config grid changed)") -- so treating those as done skips the op AND leaves
    every launch of it on the heuristic fallback, which is the exact failure this cache exists to
    prevent.

    So: fail CLOSED. Only a verdict of OK with NO reason at all counts, and the toolchain has to
    match too -- `env_identity` is deliberately kept out of the verdict (it must not fail CI on a
    different machine) but the runtime treats a mismatch as a miss, which is what makes the
    "second node, fresh clone" case wrong in the other direction. An unrecognised future reason
    string therefore costs a rebuild rather than a silently unusable cache.
    """
    from miniworld_engine.autotune import cache_status
    from miniworld_engine.autotune.cache import gpu_key

    return {r.op for r in cache_status.scan(gpu_substr=gpu_key())
            if r.verdict == "OK" and not r.reason and r.env_matches is True}


#: An entry is keyed `<dtype>|<bucket>`, and `dtype_of_args` names the SET of float operand dtypes
#: a launch carried: a bf16 launch whose norm affine is pinned fp32 records `bfloat16+float32`, a
#: true fp32 launch records `float32`. So a unit's dtype matches a label by these rules and not by
#: substring -- `float32 in "bfloat16+float32"` is true and would count a bf16 entry as fp32 cover.
def _label_serves_dtype(label: str, dtype: str) -> bool:
    return dtype_label_serves(label.split("|", 1)[0], dtype)


def dtype_label_serves(recorded: str, declared: str) -> bool:
    """Does a cache entry recorded under `recorded` answer a unit declared `declared`?

    The dtype half alone, so the same rule can be applied where the bucket has already been split
    off -- `build/audit.py`'s coverage check compares `(dtype, bucket)` pairs and was matching them
    exactly, which no mixed-operand kernel can ever satisfy: rmsnorm records `bfloat16+float32`
    against a declared `bfloat16`, so all 78 of its entries read as missing and the audit reported
    16 holes per op that a rebuild could never close.
    """
    if declared == "float32":
        return recorded == "float32"
    return recorded.startswith(declared)


def _cache_answers(unit: OpUnit, ok_ops: set[str]) -> bool:
    """Is this unit's (op, dtype) already tuned by a cache the runtime will serve?

    Op AND dtype, because dtype is the one axis of the entry key a planner can actually read. The
    key is `<dtype>|<bucket>`; the bucket half is a PACKED shape_key carrying constexprs that do
    not exist until the launcher builds them, so it cannot be reconstructed here -- but the dtype
    half is just `unit.dtype`. Ignoring it is self-perpetuating: 23 of this card's 74 caches hold
    no fp32 entry at all, so their fp32 units would be skipped, the hole would never be filled,
    and the next build would skip them again for the same reason.

    What is still coarser than the truth: `side` and `width`. Both are folded into the packed
    bucket, so a hole in one of them survives a plain `build all`; `--rebuild-cached` re-tunes the
    op from scratch and `dev audit` is what finds such holes.
    """
    if unit.op not in ok_ops:
        return False
    from miniworld_engine.autotune.cache import _load, gpu_key

    data = _load(unit.op, gpu_key())
    if not data:
        return False
    return any(_label_serves_dtype(k, unit.dtype) for k in data.get("entries", {}))


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
def visible_device(device: int) -> str:
    """Resolve a logical GPU through the parent allocation's CUDA visibility mask."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if device < 0:
            raise ValueError("GPU indices must be nonnegative")
        return str(device)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if device < 0 or device >= len(tokens) or tokens[device] == "-1":
        raise ValueError(f"GPU {device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    return tokens[device]


DEFAULT_UNIT_TIMEOUT_SECONDS = 7200.0


def _kill_unit_tree(pid: int) -> None:
    """Stop and kill this Linux unit's descendants, including setsid compilers.

    capture's compile guards create private sessions, so killpg alone misses them.
    Freeze parents before enumerating children (including thread children), keeping
    them from forking or reaping while we traverse. Check process start times before
    signalling again. This works on cluster Python builds without os.pidfd_open.
    No node-wide process scan or signalling of other build slots is needed.
    """
    frozen = []

    def stat(current: int) -> list[str]:
        return Path(f"/proc/{current}/stat").read_text().rsplit(") ", 1)[1].split()

    def freeze(current: int, parent: int | None = None) -> None:
        try:
            state = stat(current)
            if parent is not None and int(state[1]) != parent:
                return
            frozen.append((current, state[19]))  # starttime, /proc stat field 22
            os.kill(current, signal.SIGSTOP)
            deadline = time.monotonic() + 0.2
            while stat(current)[0] not in ("T", "t", "Z") and time.monotonic() < deadline:
                time.sleep(0.001)
            children = set()
            for task in Path(f"/proc/{current}/task").iterdir():
                with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                    children.update(map(int, (task / "children").read_text().split()))
            for child in children:
                freeze(child, current)
        except (FileNotFoundError, ProcessLookupError):
            return

    try:
        freeze(pid)
    finally:
        # Child-first: keep ancestry intact until every detached compiler is found.
        for current, started in reversed(frozen):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                if stat(current)[19] == started:
                    os.kill(current, signal.SIGKILL)


def _run_unit_process(cmd, *, cwd, stdout, stderr, check, env, timeout):
    """Bound the whole unit, including external compilers, in its own process group.

    subprocess.run(timeout=...) kills only the leader. A compiler descendant can
    otherwise keep the CUDA context / file descriptors alive after the slot is freed.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env,
                            start_new_session=True)
    stdout.write(f"[unit] pid={proc.pid} pgid={proc.pid}\n")
    stdout.flush()
    try:
        proc.wait(timeout=timeout)
    except BaseException:
        try:
            _kill_unit_tree(proc.pid)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    # Also reap any lingering compile workers from an otherwise completed unit.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    return subprocess.CompletedProcess(cmd, proc.returncode)


def _run_unit_subprocess(unit: Unit | OpUnit, device: int, shard_dir: Path, repo: Path,
                         compile_jobs: int, config_dir: Path | None = None,
                         fill_gaps: bool = False, share_card: bool = False,
                         keep_ir: bool = False, predict: bool = False,
                         bench_clear_mb: int = 0, bench_rep_ms: int = 0,
                         cores: str = "", rebuild_cached: bool = False,
                         unit_timeout_seconds: float = DEFAULT_UNIT_TIMEOUT_SECONDS) -> dict:
    """One unit, in its own process on one card. Subprocess rather than thread: a capture can take
    the CUDA context down with it, and one dead unit must not end the build."""
    if not math.isfinite(unit_timeout_seconds) or unit_timeout_seconds <= 0:
        raise ValueError("unit_timeout_seconds must be finite and positive")
    physical_device = visible_device(device)
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
    env["CUDA_VISIBLE_DEVICES"] = physical_device
    # Write cubin + metadata and not the five IR levels: 187 KB an entry becomes 71, and the A6000
    # rebuild's cache was 40 GB of a shared filesystem. See autotune/triton_cache.py.
    triton_cache.store_binary_only_env(env, keep_ir)
    cmd = [sys.executable, "-u", "-m", "miniworld_engine.autotune.builder",
           "--shard", str(shard), "--compile-jobs", str(compile_jobs), *unit.cmd_args()]
    if rebuild_cached:
        # On the command line, not in the environment: this is the flag that decides whether a
        # unit re-measures configs the cache already holds, and a unit has to be reproducible
        # from its own logged argv or the answer to "why did this take four hours" is unreadable.
        cmd.append("--rebuild-cached")
    if cores:
        # A slot's own cores, shared with nobody. A unit alternates between compiling on a pool of
        # processes and MEASURING on one thread, and the measurement is the build's product: with
        # the node's cores pooled, a unit that is measuring competes with every other unit's
        # compile workers, and each of those forks a child per chunk, so four units asking for 32
        # workers put ~256 runnable processes on 128 cores. Measured, one launch cost 21 ms inside
        # a loaded build against 329 us on an idle card -- and the timer's own step is 1.024 us,
        # so at that point configs stop being distinguishable from each other.
        #
        # The cost is real and is the other half of the trade: while a slot measures, its own
        # compile cores sit idle and no other slot may borrow them.
        cmd = ["taskset", "-c", cores, *cmd]
    if share_card:
        # Units sharing a card may compile at the same time -- that is the point -- but must not
        # MEASURE at the same time. One lock file per card, taken per tuning round; see
        # capture._bench_lock_acquire.
        cmd += ["--bench-lock", str(shard_dir / f"gpu{device}.benchlock")]
    if predict:
        cmd += ["--predict-unusable"]
    if bench_clear_mb and bench_rep_ms:
        cmd += ["--bench-clear-mb", str(bench_clear_mb), "--bench-rep-ms", str(bench_rep_ms)]
    # Paths rather than the parsed configs: the child re-reads the CSVs itself, so a unit's config
    # space is reproducible from its own command line (the same reason every other knob here is an
    # argument and not inherited shell state).
    if config_dir is not None:
        cmd += ["--config-dir", str(config_dir)]
    if fill_gaps:
        cmd += ["--fill-gaps"]
    env.update(unit.env())
    started = time.monotonic()
    timed_out = False
    with log.open("a") as handle:
        handle.write(f"\n[unit] START timeout={unit_timeout_seconds:g}s argv={cmd!r}\n")
        handle.flush()
        attempt_offset = handle.tell()
        try:
            proc = _run_unit_process(cmd, cwd=repo, stdout=handle, stderr=subprocess.STDOUT,
                                     check=False, env=env, timeout=unit_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(cmd, 124)
            handle.write(f"\n[unit] TIMEOUT after {unit_timeout_seconds:g}s; "
                         "unit process group killed; claim released for resume\n")
        except OSError as exc:
            proc = subprocess.CompletedProcess(cmd, 127)
            handle.write(f"\n[unit] LAUNCH-ERROR {exc}\n")
        except BaseException:
            claim.unlink(missing_ok=True)
            raise
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
        with log.open() as handle:
            handle.seek(attempt_offset)
            skipped = not timed_out and "[unit] SKIPPED-PERMANENT" in handle.read()
    except OSError:
        skipped = False
    failed_marker = shard.with_suffix(".failed")
    if proc.returncode != 0 and not skipped:
        # Do not rewrite partial measurements. This also prevents an older complete
        # shard from satisfying resume after a forced rebuild timed out.
        failed_marker.write_text(f"rc={proc.returncode} timed_out={timed_out}\n")
    else:
        failed_marker.unlink(missing_ok=True)
    if not skipped and (proc.returncode != 0 or not _shard_reusable(shard)):
        # A failed module can still have timings from its earlier kernels. Preserve those
        # timings, but release the claim so resume retries the unfinished module.
        claim.unlink(missing_ok=True)
    # a permanent skip KEEPS its claim: the shape will not fit on the next attempt either, and
    # releasing it made every resumed job re-claim the same OOMing units and produce nothing.
    return {"label": unit.label, "gpu": device, "rc": proc.returncode, "ops": ops,
            "skipped": skipped, "timed_out": timed_out,
            "unit_timeout_seconds": unit_timeout_seconds,
            "seconds": round(time.monotonic() - started, 1), "shard": str(shard), "log": str(log)}


def _core_slices(slots: int) -> list[str]:
    """One disjoint core list per slot, from the cores this job was actually given.

    MEASURED AND IT DOES NOT PAY, on the case it was written for. Two units, two cards, 48 cores:
    pooled 1655 s, pinned 1687 s -- 2% slower, and both arms chose configs whose measured times
    were identical. The reasoning that motivated it was sound and the arithmetic was not: at
    `cores / gpus` workers per unit the pool already fits the allocation exactly (24 + 24 = 48),
    so there is nothing to contend over, and pinning only takes away a slot's ability to borrow
    the other's cores while it measures.

    The contention that IS real comes from `_compile_chunk` forking a child per chunk, which
    doubles the process count for as long as a chunk runs -- and that happens inside a slot's own
    slice too, so a slice does not prevent it. Kept behind `--pin-cores`, default off, because the
    trade could go the other way on a node whose core count is not a multiple of its cards.


    Read from `sched_getaffinity`, not `cpu_count`: under Slurm the job owns a subset, and slicing
    the machine instead of the allocation would hand a slot cores it may not run on. Returns empty
    strings when there are fewer cores than slots, which leaves every slot unpinned -- pinning one
    core per slot would cost more than the contention.
    """
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return [""] * slots
    if slots < 2 or len(allowed) < 2 * slots:
        return [""] * slots
    per = len(allowed) // slots
    return [",".join(str(c) for c in allowed[i * per:(i + 1) * per]) for i in range(slots)]


def validate_build_gpus(gpus: list[int]) -> None:
    """A build publishes one device-model cache, so its workers must match it."""
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("build requires a nonempty list of distinct logical GPUs")
    target = torch.cuda.current_device()
    devices = set(gpus) | {target}
    models = {}
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        models[device] = (props.name, props.major, props.minor)
    if len(set(models.values())) != 1:
        raise ValueError(
            f"build workers must match the current GPU used for planning/merge: {models}. "
            "Run each GPU model separately with CUDA_VISIBLE_DEVICES selecting that model.")


def build_all(selected: list, shard_dir: Path, gpus: list[int], compile_jobs: int,
              resume: bool = False, reclaim: bool = False,
              config_dir: Path | None = None, fill_gaps: bool = False,
              units_per_gpu: int = 1, keep_ir: bool = False, predict: bool = False,
              bench_clear_mb: int = 0, bench_rep_ms: int = 0,
              pin_cores: bool = False, skip_cached: bool = True,
              unit_timeout_seconds: float = DEFAULT_UNIT_TIMEOUT_SECONDS) -> list[dict]:
    """Run every unit of ``selected`` across ``gpus``. Returns one result record per unit.

    ``units_per_gpu`` > 1 puts that many units on each card so their phases interleave. A unit
    alternates between compiling (a pool of processes, no GPU) and measuring (one GPU, one core),
    and on the A6000 rebuild those were 72% and 20% of the wall -- so during a fifth of every
    unit its whole compile pool sat idle, and during the rest the card did nothing. Units on the
    same card take that card's bench lock while they measure, so the overlap is compile-against-
    measure and never measure-against-measure.

    Measured twice, and the first measurement was wrong twice over. At 2 it first came out 34%
    SLOWER, from a pool-sizing bug -- `compile_jobs` divided the cores by the SLOTS, so each unit
    got half a pool and its compile phase took twice the wall -- on a unit set that spent 73% of
    its wall BENCHING, where a second unit has nothing to fill. Fixed (divide by the cards) and
    re-measured on compile-dominated units, four units on two cards, 48 cores either way:

        units-per-gpu 1    6097 s
        units-per-gpu 2    5793 s      5% faster

    Safe: four of six buckets chose the same config and the two that differed were 0.0% and 2.6%
    off, against a control -- the same settings run twice -- that disagreed about three of seven
    with a worst case of 12.5%.

    Five percent and not more, and the log says why: one unit spent 3,962 s waiting on the other's
    bench lock. The ceiling is the bench, ~18% of a unit's wall, and half of it went to queueing.
    Cheapening the bench (`bench_rep_ms`) should raise this ceiling; the two have not been
    measured together. See docs/operations/dispatch-cache.md.
    """
    import concurrent.futures as cf

    if not math.isfinite(unit_timeout_seconds) or unit_timeout_seconds <= 0:
        raise ValueError("unit_timeout_seconds must be finite and positive")
    validate_build_gpus(gpus)
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
        # Divide by the CARDS, not by the slots. At most one slot per card is measuring at any
        # moment -- that is what the bench lock enforces -- and a measuring slot needs about one
        # core, so the slots sharing a card are not compiling at the same time either. Dividing by
        # the slots instead halved every unit's pool, stretched its compile phase to twice the
        # wall, and left the card idle waiting for someone to finish: measured at 4436 s -> 5940 s
        # on 28 units, with the card busy 71% of the wall against 56%.
        compile_jobs = max(1, cores // max(1, len(gpus)))
        slots = max(1, len(gpus) * max(1, units_per_gpu))
        print(f"  [compile] {cores} cores / {len(gpus)} gpu(s) -> {compile_jobs} compile workers "
              f"per unit, {slots} unit slot(s)", flush=True)

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
    sm = device_sm()
    if case_build and fill_gaps and sm:
        from miniworld_engine.autotune import derive, plan
        try:
            evidence = plan.load(sm)
        except (OSError, ValueError):
            print("No current verified derivation: enumerating every module; shared tuning "
                  "rounds still prevent duplicate measurements.", flush=True)
        else:
            from miniworld_engine.autotune.cache import gpu_key
            report = derive.coverage(sm, gpu_key())
            missing = {f"{op}|{key}" for op, key in report["missing"]}
            before = len(work)
            work = plan.select(work, selected, evidence["units"], missing)
            import hashlib

            from miniworld_engine.autotune.cache import config_space_hash
            from miniworld_engine.autotune.configs import configs_for
            grids = [(op, config_space_hash(configs_for(op)))
                     for op in sorted({op for op, _key in report["missing"]})]
            generation = hashlib.sha256(
                repr((evidence["source_identity"], sorted(missing), grids)).encode()).hexdigest()[:12]
            work = [dataclasses.replace(u, generation=generation) for u in work]
            print(f"verified cache plan: {len(work)} of {before} module units cover "
                  f"{len(missing)} missing keys", flush=True)
    if work:
        generation = _generation_for_work(config_dir)
        work = [dataclasses.replace(u, generation=(
            f"{u.generation}-{generation}" if u.generation else generation)) for u in work]
    if resume:
        work = [u for u in work if not _shard_reusable(shard_dir / f"{u.stem}.json")]
    # A unit the shipped cache already answers is not work. Tested per ITEM with `isinstance`, not
    # from `case_build`: that flag reads `selected[0]` alone, so a mixed list would send a module
    # `Unit` -- which has no `.op` -- into a lookup expecting one. Only OpUnits name a single
    # (op, dtype) to look up; a module Unit re-tunes whatever its case touches.
    if skip_cached:
        ok_ops = _cache_ok_ops()
        before = len(work)
        work = [u for u in work
                if not (isinstance(u, OpUnit) and _cache_answers(u, ok_ops))]
        if before != len(work):
            print(f"skipping {before - len(work)} of {before} unit(s): a non-stale committed cache "
                  f"already answers them -- pass --rebuild-cached to redo them", flush=True)
    # `--resume` is on by default, so a claim with no shard now costs a unit that is never built
    # and never reported. Say it: the alternative is a build that looks complete and quietly
    # covers less than the last one did.
    orphans = [c.stem for c in sorted(shard_dir.glob("*.claim"))
               if not _shard_has_entries(shard_dir / f"{c.stem}.json")]
    if orphans and not reclaim:
        print(f"WARNING: {len(orphans)} claim(s) here have no shard -- units a killed build left "
              f"in flight. They are being SKIPPED. Re-run with --reclaim once no other build is "
              f"using this directory:", flush=True)
        for stem in orphans[:10]:
            print(f"    {stem}", flush=True)
        if len(orphans) > 10:
            print(f"    ... and {len(orphans) - 10} more", flush=True)
    if not work:
        print("nothing to do (every unit is already answered by a shard or the committed cache)")
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

    def worker(device: int, cores: str = "") -> list[dict]:
        got = []
        while True:
            try:
                unit = queue.get_nowait()
            except Empty:
                return got
            res = _run_unit_subprocess(unit, device, shard_dir, repo, compile_jobs,
                                       config_dir, fill_gaps, share_card=units_per_gpu > 1,
                                       keep_ir=keep_ir, predict=predict,
                                       bench_clear_mb=bench_clear_mb, bench_rep_ms=bench_rep_ms,
                                       cores=cores, unit_timeout_seconds=unit_timeout_seconds,
                                       # `--rebuild-cached` on the CHILD means "re-measure the
                                       # configs the cache already searched for this shape". That
                                       # is not what disabling the unit-level skip is for, and the
                                       # two were the same flag: the only way to reach a unit the
                                       # cache "answers" was to pay for re-measuring every config
                                       # it already holds. `--fill-gaps` splits them -- run the
                                       # unit, subtract what is measured, bench only the rest --
                                       # which is what a new WIDTH rung on an already-tuned op
                                       # needs, and what a rebuild should cost.
                                       rebuild_cached=not skip_cached and not fill_gaps)
            if res.get("claimed_elsewhere"):
                got.append(res)
                continue
            status = ("TIMEOUT" if res.get("timed_out") else "ok" if res["rc"] == 0 and res["ops"] else
                      "skip" if res.get("skipped") else
                      "EMPTY" if res["rc"] == 0 else "FAIL")
            print(f"  [gpu{device}] {status:5s} {res['label']}  {res['seconds']}s  {res['ops']} ops",
                  flush=True)
            got.append(res)

    results: list[dict] = []
    # One thread per unit SLOT, not per card: `units_per_gpu` slots share each card and pull from
    # the same queue, so a card whose unit is measuring still has a unit compiling.
    slots = [g for g in gpus for _ in range(max(1, units_per_gpu))]
    slices = _core_slices(len(slots)) if pin_cores else [""] * len(slots)
    if pin_cores and any(slices):
        print(f"  [cores] {len(slots)} slot(s), {slices[0].count(',') + 1} core(s) each, "
              f"shared with no other slot", flush=True)
    with cf.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for future in cf.as_completed([pool.submit(worker, g, c)
                                       for g, c in zip(slots, slices, strict=True)]):
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
    from miniworld_engine.autotune.cache import (
        cache_misses,
        clear_cache_misses,
    )
    from miniworld_engine.autotune.cache import (
        drop_cache_misses as _drop_cache_misses,
    )

    clear_cache_misses()
    settings.configure(run_autotune=False, capture=False)   # use the cache, do not rebuild it
    aborted = 0
    for case in selected:
        for di in range(len(case.dims)):
            for length in case.lengths_for(di):
                for dtype in case.dtypes:
                    for train in ((False, True) if case.train else (False,)):
                        for impl in case.impls:
                            # Snapshot BEFORE, and drop what this case added if it then died. A
                            # miss is a claim that production asks for a key the cache lacks, and
                            # a case that aborts is production doing no such thing: the lookups it
                            # made before the exception are for a shape this card never reaches.
                            #
                            # 31 of one A6000 replay's misses came from `cases()` forcing
                            # `implementation="cute"` on triangle_multiplication, every one of them
                            # followed immediately by `NotImplementedError: Gemm Sm80 is not
                            # implemented yet`. They cannot be built here -- the case that would
                            # capture them dies at the same line -- so counting them made the
                            # number unactionable and sent a 1,220-unit build after keys no unit
                            # could ever produce.
                            before = set(cache_misses())
                            if run_case(case, length, di, train=train, impl=impl, dtype=dtype):
                                continue
                            added = set(cache_misses()) - before
                            if added:
                                aborted += len(added)
                                _drop_cache_misses(added)
    if aborted:
        print(f"  ({aborted} lookup(s) ignored: recorded by a case that then failed to run -- "
              f"a shape this card does not reach)", flush=True)
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
        # ShapeKeyTooWide joins them: a width whose derived axis does not fit the shape-key
        # packing (ND2 = 8*base is 6,144 at base 768) cannot be keyed at all, so the unit is not
        # retryable either. See autotune/shape_key.pack.
        perm = type(exc).__name__ in ("OutOfMemoryError", "OutOfResources", "ShapeKeyTooWide") or \
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
    dropped = capture.over_budget()
    if dropped:
        print(f"  [launch-budget] abandoned {sum(dropped.values())} config(s) whose launch was "
              f"{capture._LAUNCH_BUDGET_X}x the round's fastest: "
              + ", ".join(f"{op}={n}" for op, n in sorted(dropped.items())), flush=True)
    skipped = capture.skipped_configs()
    if skipped:
        # Say it out loud. A unit that reuses the cache and one that re-measures the whole grid
        # produce the same shard and differ only in wall-clock, so without this line the single
        # most expensive property of a build is invisible until it is over.
        print(f"  [incremental] reused {sum(skipped.values())} already-measured config(s): "
              + ", ".join(f"{op}={n}" for op, n in sorted(skipped.items())), flush=True)
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
    ap.add_argument("--width", type=int, default=0,
                    help="base channel width for this unit. Reaches the drivers through "
                         "MINIWORLD_DRIVER_WIDTH before they import, like --length; it is on the "
                         "command line so the unit is reproducible from it.")
    ap.add_argument("--heads", type=int, default=0,
                    help="the unit's SECOND width axis, for a kernel whose bucket carries two: "
                         "n_head beside HEAD_DIM, d_cond beside d_hidden, d_pair beside d_hidden. "
                         "Reaches the drivers as MINIWORLD_DRIVER_HEADS. A driver that derived it "
                         "from the first axis could only ever build the pairs that one ratio "
                         "produces, and `cases()` declares pairs it does not.")
    # Every side `op_units` can emit. "token" was added when the DiT families and the level=both
    # rows were split by stream, and this list was not -- so every token unit died in argparse
    # before it reached a kernel, 3 seconds and 0 ops each. The parent process and the child have
    # to agree on the vocabulary; keeping the tuple here in step with `_widths` is the whole job.
    ap.add_argument("--side", default="", choices=("", "pair", "atom", "token"),
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
    ap.add_argument("--rebuild-cached", action="store_true",
                    help="re-measure configs this card's committed cache already searched for the "
                         "shape being built; off by default, which is what makes a rerun cheap")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="leave keys the cache already holds alone; full-grid only the misses. "
                         "See settings.Settings.fill_gaps")
    ap.add_argument("--predict-unusable", action="store_true",
                    help="probe a slice of each round first and skip the configs the probes prove "
                         "cannot pay off. See settings.Settings.predict_unusable")
    ap.add_argument("--bench-clear-mb", type=int, default=0,
                    help="MB zeroed before each timed iteration (0 = triton's 256). Must be set "
                         "together with --bench-rep-ms. See settings.Settings.bench_clear_mb")
    ap.add_argument("--bench-rep-ms", type=int, default=0,
                    help="ms of measurement per config (0 = triton's 100). Warmup scales 1:4.")
    ap.add_argument("--bench-lock", default="",
                    help="per-card lock file held while this unit MEASURES, so two units sharing "
                         "a card never measure at once. See settings.Settings.bench_lock")
    args = ap.parse_args(argv)

    if args.config_dir:
        from miniworld_engine.autotune.configs import use_config_dir
        use_config_dir(args.config_dir, require_all=False)
        # Reported after run_case, when the kernels have imported and registered: counting here
        # would always print 0/0, since nothing has asked for configs yet.
        print(f"  [config] set to {args.config_dir}"
              f"{'  [fill-gaps] keys the cache already holds are left alone' if args.fill_gaps else ''}",
              flush=True)
    # Every compile records the shared memory it needed, beside this unit's shard. It is the one
    # number that says WHY a config later scores +inf, and triton throws it away: it catches its
    # own OutOfResources inside `Autotuner._bench` and returns [inf, inf, inf], so shared-memory
    # overflow, a register-spill kill and a genuinely slow config all arrive identical. A build
    # that cannot tell them apart cannot decide what to stop compiling.
    #
    # The ENVIRONMENT, unlike every knob above it, and not for want of trying: the reading is taken
    # inside the spawned compile workers (`capture._compile_payload`), and a spawned process does
    # not get this one's argv. A setting cannot cross that boundary either -- `settings.configure`
    # runs here, in the parent. Derived from --shard rather than passed in, so it is still a
    # function of the command line and not of the shell.
    os.environ["MINIWORLD_SMEM_LOG"] = str(Path(args.shard).with_suffix(".smem"))
    settings.configure(run_autotune=True, capture=True, fill_gaps=args.fill_gaps,
                       compile_jobs=(args.compile_jobs or None),
                       predict_unusable=args.predict_unusable,
                       bench_lock=args.bench_lock,
                       bench_clear_mb=args.bench_clear_mb, bench_rep_ms=args.bench_rep_ms)
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
    capture.set_incremental(not args.rebuild_cached)
    capture.set_round_cache(str(Path(args.shard).parent / ".round-cache"))
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
