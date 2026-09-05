"""The per-op build must be able to reach what the module cases present.

There are two declarations of "what shapes the model runs" in `autotune/builder.py`:

* `cases()` -- module-level, one entry per production module with its real `dims`. This is what
  `dev audit --replay` drives (`builder.audit(builder.cases())`), so it is the repository's own
  definition of a covered workload.
* `op_units()`'s `LADDER` -- kernel-level, the channel widths `build all` sweeps. `build all` takes
  the per-op path, so this list alone decides what the shipped cache contains.

Nothing tied them together, and they drifted. Every consequence below was measured by
`dev audit --replay` against a cache `build all` had just reported complete:

* `cases()` builds `triangle_multiplication` and `triangle_attention_bidirectional` at
  `d_pair=384`; the pair ladder was `(128, 256, 512)`. `trimul_outproj_layernorm_gemm_gate_triton`
  and `layernorm_fwd_saveact_triton` had no `K=384` / `N=384` bucket at all.
* `cases()` builds the MSA modules at `d_msa=64`, which reaches the shared `level=both` LayerNorm
  kernels as a channel width. No ladder carried 64.

This test does not try to DERIVE one from the other -- the dim-name-to-stream mapping is real
knowledge and guessing it is how the last such "obvious" inference destroyed a cache. It asserts
the weaker, checkable thing: every width any case presents is a rung on some ladder. That is
exactly the invariant 384 and 64 broke, and it is one a new case cannot break silently.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import builder

#: dims keys that are NOT channel widths -- head COUNTS, which the driver derives a head dim from
#: rather than tiling over. `augmented_attention`'s heads are the reason this list is explicit:
#: the DiT fixes `n_head` and lets head_dim follow `d_single`, so 16 is a count, and 24 / 48 (the
#: widths it implies) are what the kernel actually sees.
NOT_A_WIDTH = frozenset({"n_head", "n_heads"})

#: width -> why no ladder drives it. A width only needs a rung if it reaches a kernel that KEYS on
#: shape -- `pack` folds a kernel's own tiled axes into the key, and a projection width that never
#: becomes one of them changes no bucket. The evidence for an entry here is a clean
#: `dev audit --replay`: if a width really were keyed and undriven, replay would ask for it.
NOT_DRIVEN: dict[int, str] = {
    32: "`outer_product_mean.d_hidden` and `pairformer_block.d_hidden_tri_attention` -- the small "
        "inner projection width of the OPM / tri-attention hidden GEMMs, which no keyed kernel "
        "tiles over: the A100 replay asked for no bucket at width 32, while it did ask for 64, "
        "384 and the augmented-attention head dims (all now driven)",
}


def _presented_widths() -> dict[int, list[str]]:
    """width -> the case names that present it."""
    out: dict[int, list[str]] = {}
    for case in builder.cases():
        for dims in case.dims:
            for name, value in dims.items():
                if name in NOT_A_WIDTH or not isinstance(value, int):
                    continue
                out.setdefault(value, []).append(f"{case.name}.{name}")
    return out


def _ladder_widths() -> set[int]:
    """Every rung `op_units` can drive, across all streams."""
    units = builder.op_units(None)
    if not units:
        pytest.skip("no op units on this checkout (no config set?)")
    return {u.width for u in units if getattr(u, "width", None)}


def test_cases_present_widths_at_all() -> None:
    """Guard the guard: a renamed dims key would make the sweep below vacuous."""
    assert _presented_widths(), "builder.cases() presents no integer dims"


def test_every_width_a_case_presents_is_on_some_ladder() -> None:
    ladder = _ladder_widths()
    presented = _presented_widths()
    missing = {w: sorted(set(who)) for w, who in presented.items()
               if w not in ladder and w not in NOT_DRIVEN}
    detail = "\n".join(f"    {w}: presented by {', '.join(who)}" for w, who in sorted(missing.items()))
    assert not missing, (
        f"widths `builder.cases()` presents that no `op_units` ladder drives:\n{detail}\n\n"
        f"ladder rungs: {sorted(ladder)}\n"
        f"`build all` runs the per-op path, so a width absent from every ladder is a bucket the "
        f"shipped cache can never hold -- while `dev audit --replay`, which drives `cases()`, asks "
        f"for it on every run. Add the rung in `op_units` (PRESENTED / HEADROOM_PAIR / MSA_WIDTHS), "
        f"or drop the width from the case if the model does not run it.")


def test_not_driven_entries_are_still_undriven() -> None:
    """A declaration the ladders have since caught up with is stale documentation; drop it."""
    overtaken = sorted(set(NOT_DRIVEN) & _ladder_widths())
    assert not overtaken, (
        f"NOT_DRIVEN still excuses widths the ladders now drive: {overtaken}. Remove the entry.")
