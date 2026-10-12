"""Model shapes are covered by module units; explicit driver probes keep their own axes."""
from __future__ import annotations

import os

import pytest
from tests.paths import ROOT, registry_rows

from miniworld_engine.autotune import builder

#: dims keys that are NOT channel widths -- head COUNTS, which the driver derives a head dim from
#: rather than tiling over. `augmented_attention`'s heads are the reason this list is explicit:
#: the DiT fixes `n_head` and lets head_dim follow `d_single`, so 16 is a count, and 24 / 48 (the
#: widths it implies) are what the kernel actually sees.
NOT_A_WIDTH = frozenset({"n_head", "n_heads", "n", "has_bias"})
# Exact leaf-only axes are driven by checkpoint_cases, not by every GEMM's
# legacy width ladder. Broadcasting norm width 833 into TriMul would be invalid.
EXACT_LEAF_AXES = frozenset({"d_norm", "d_expanded", "d_out"})

#: width -> why no ladder drives it. A width only needs a rung if it reaches a kernel that KEYS on
#: shape -- `pack` folds a kernel's own tiled axes into the key, and a projection width that never
#: becomes one of them changes no bucket. The evidence for an entry here is a clean
#: `dev audit --replay`: if a width really were keyed and undriven, replay would ask for it.
#: EMPTY. 32 was the last entry -- the OPM / tri-attention inner projection width, excused because
#: "the A100 replay asked for no bucket at width 32". It is driven now, as a rung of the
#: `head_dim` ladder: `triangle_attention`'s bucket carries `d_hidden // n_head`, and `cases()`
#: declares combinations giving 16, 32 and 64. The number is the same 32 for a different reason,
#: which is exactly why this test compares numbers and not stories -- a width on a ladder needs no
#: excuse, whatever put it there.
NOT_DRIVEN: dict[int, str] = {}


def _presented_widths() -> dict[int, list[str]]:
    """width -> the case names that present it."""
    out: dict[int, list[str]] = {}
    for case in builder.cases():
        for dims in case.dims:
            for name, value in dims.items():
                if name in NOT_A_WIDTH | EXACT_LEAF_AXES or not isinstance(value, int):
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


def test_every_model_row_is_preserved_by_the_module_plan(monkeypatch) -> None:
    from miniworld_engine.autotune import derive, plan

    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    cases = builder.cases()
    by_name = {case.name: case for case in cases}
    actual = {plan.label(unit, by_name[unit.case]) for unit in builder.units(cases)}
    expected = {unit.label for unit in derive.units(derive.module_rows(), arch="sm86")}
    assert actual == expected
    assert any("d_hidden=8" in label and "msa_pair_weighted_averaging[" in label
               for label in actual)


def test_not_driven_entries_are_still_undriven() -> None:
    """A declaration the ladders have since caught up with is stale documentation; drop it."""
    overtaken = sorted(set(NOT_DRIVEN) & _ladder_widths())
    assert not overtaken, (
        f"NOT_DRIVEN still excuses widths the ladders now drive: {overtaken}. Remove the entry.")


# --------------------------------------------------------------------------- #
# derived axes: a head dim is not a channel width, and the check above cannot see it
# --------------------------------------------------------------------------- #
#: family -> the driver module attribute holding the HEAD DIM it builds tensors at. Named, not
#: guessed: the check below asks the driver for the number instead of dividing widths by a list of
#: plausible head counts, because both defects here produced numbers a plausible-divisor rule
#: accepts (`triangle_attention`'s frozen 32 is 128 // 4).
HEAD_DIM_SYMBOL = {
    "triangle_attention": "D",
    "augmented_attention": "D",
}

_PROBE = r"""
import importlib, json, os, sys
out = {}
for fam, sym, widths in json.loads(sys.argv[1]):
    got = {}
    for w in widths:
        os.environ["MINIWORLD_DRIVER_WIDTH"] = str(w)
        drivers = importlib.reload(importlib.import_module("miniworld_engine.kernels.drivers"))
        mod = importlib.import_module("miniworld_engine.kernels.drivers." + fam)
        # the family module closes over the constants above, so it has to be reloaded after them
        mod = importlib.reload(mod)
        got[w] = getattr(mod, sym, None)
    out[fam] = got
print(json.dumps(out))
"""


def _packed_axes() -> dict[str, set[str]]:
    """registry file -> the axis names its kernels fold into the cache key."""
    import ast

    root = ROOT / "src" / "miniworld_engine" / "kernels"
    out: dict[str, set[str]] = {}
    for f in root.rglob("*.py"):
        if "notes" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except (OSError, SyntaxError):
            continue
        names: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) in ("pack", "token_key", "atom_key",
                                                           "both_key")):
                names |= {kw.arg for kw in node.keywords if kw.arg}
        if names:
            out[str(f.relative_to(ROOT / "src"))] = names
    return out


def _families_keyed_on_head_dim() -> dict[str, set[int]]:
    """family -> the widths `op_units` drives it at, for families that key on HEAD_DIM."""
    packed = _packed_axes()
    fam_of = {r["kernel"]: r["family"] for r in registry_rows()}
    keyed = {r["family"] for r in registry_rows()
             if (r.get("developed") or "yes").strip() != "no"
             and "HEAD_DIM" in packed.get(r["file"], set())}
    out: dict[str, set[int]] = {}
    for unit in builder.op_units(None):
        fam = fam_of.get(unit.op)
        if fam in keyed and unit.width:
            out.setdefault(fam, set()).add(unit.width)
    return out


def _declared_head_dims() -> dict[str, set[int]]:
    """case name -> the head dims that case's dims imply.

    `d_hidden // n_head`, or `d_single // n_head` where the case names no d_hidden -- the two forms
    the two families use. Computed from `cases()` rather than listed, so it cannot drift from it.
    """
    out: dict[str, set[int]] = {}
    for case in builder.cases():
        for dims in getattr(case, "dims", ()) or ():
            n = dims.get("n_head") or dims.get("n_heads")
            total = dims.get("d_hidden") or dims.get("d_single")
            if n and total and total % n == 0:
                out.setdefault(case.name, set()).add(total // n)
    return out


def _driven_head_dims(want: dict[str, set[int]]) -> dict[str, dict[int, int]]:
    """family -> {driven width: the head dim its driver builds at that width}.

    Imported in a child process at each width, because `MINIWORLD_DRIVER_WIDTH` is read when
    `drivers/__init__` is imported and the family modules close over the result -- the same route
    `build all` takes (`builder.py` sets that variable per unit).
    """
    import json
    import subprocess
    import sys

    spec = [[fam, HEAD_DIM_SYMBOL[fam], sorted(widths)] for fam, widths in sorted(want.items())]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([sys.executable, "-c", _PROBE, json.dumps(spec)],
                          capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, f"driver probe failed:\n{proc.stderr[-2000:]}"
    return {fam: {int(w): d for w, d in got.items()}
            for fam, got in json.loads(proc.stdout.splitlines()[-1]).items()}


def test_every_family_that_keys_on_head_dim_names_its_symbol() -> None:
    """Guard the guard: a family added or renamed without an entry would skip the check below."""
    missing = sorted(set(_families_keyed_on_head_dim()) - set(HEAD_DIM_SYMBOL))
    assert not missing, (
        f"{missing} fold HEAD_DIM into the cache key and HEAD_DIM_SYMBOL does not say which driver "
        f"attribute holds it, so nothing checks that they build the head dims `cases()` declares.")


def test_every_head_dim_a_case_declares_is_built_by_its_driver() -> None:
    """A kernel that folds HEAD_DIM into its key is tuned per head dim, and a head dim is
    `d_hidden // n_head` -- a number no stream's channel ladder contains.

    This has been the same defect twice, and the second time got past a test written for the first.
    `augmented_attention` had it first: its driver reused triangle_attention's `H = width // 32,
    D = 32`, which fixes the head DIM and derives the head COUNT, while the DiT does the opposite --
    so production ran head dims 24 and 48, the build made 32, and `dev audit --replay` measured 60
    misses across three kernels. `triangle_attention` then kept the constant `D = ragged(32)` while
    `cases()` declared 16, 32 and 64; replay measured 20 more misses across four kernels, found only
    by reading the packed keys back by hand.

    Nothing compared the two numbers. `test_every_width_a_case_presents_is_on_some_ladder` above
    compares CHANNEL widths, and a head dim is not one: 32 is on the ladder for its own reasons and
    the wrong 32 sat behind it. So this asks the DRIVER what it builds, at each width the build
    drives it at, rather than dividing widths by plausible head counts -- a rule like that accepts
    the frozen 32 (128 // 4) and would have passed both defects.
    """
    want = _declared_head_dims()
    driven = _driven_head_dims(_families_keyed_on_head_dim())
    if not driven:
        pytest.skip("no kernel folds HEAD_DIM into its key")

    bad = []
    for fam, per_width in sorted(driven.items()):
        declared: set[int] = set()
        for case_name, dims in want.items():
            if case_name.startswith(fam):
                declared |= dims
        if not declared:
            continue
        built = {d for d in per_width.values() if d}
        assert built, f"{fam}: the driver probe read no head dim from {HEAD_DIM_SYMBOL[fam]}"
        gap = sorted(declared - built)
        if gap:
            bad.append(f"{fam}: cases() declares head dims {sorted(declared)}, its driver builds "
                       f"{sorted(built)} across widths {sorted(per_width)} -- missing {gap}")
    assert not bad, (
        "a family folds HEAD_DIM into its cache key and never builds the head dims `cases()` "
        "declares, so every production launch at those dims misses the cache:\n  "
        + "\n  ".join(bad)
        + "\n\nFix the registry `width` column (a head dim is its own axis, not a channel "
          "stream: `width=head_dim`) or the driver's derivation -- not the ladder.")
