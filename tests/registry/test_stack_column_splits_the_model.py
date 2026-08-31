"""`build trunk` is only as good as the column it reads.

registry.csv's `stack` says which half of krystal launches a kernel: `trunk` for the Pairformer /
MSA / template stack, `diffusion` for token_dit / atom_dit / the SWA atom transformer, `both` for
one each side launches. `build trunk` sweeps trunk + both, `build diffusion` sweeps diffusion +
both -- so a row with a typo, or a blank one, silently drops a kernel out of BOTH builds and the
only symptom is a production cache miss months later.

The family agreement test is the one that catches a hand-edit: a family's kernels live in one
package and are launched by one module, so they belong to the same half. If a real exception ever
appears this test is where the decision gets recorded, deliberately, instead of one row quietly
disagreeing with its eleven siblings.
"""
from __future__ import annotations

from collections import defaultdict

from paths import registry_rows

KNOWN = {"trunk", "diffusion", "both"}


def _rows() -> list[dict]:
    return registry_rows()


def test_every_row_declares_a_known_stack() -> None:
    bad = sorted({(r["kernel"], r.get("stack")) for r in _rows()
                  if (r.get("stack") or "").strip() not in KNOWN})
    assert not bad, f"rows whose stack is blank or unknown (they fall out of every build): {bad}"


def test_the_cli_offers_exactly_the_two_halves() -> None:
    from miniworld_engine.cli import STACKS

    declared = {r["stack"] for r in _rows()}
    assert set(STACKS) | {"both"} == declared, (
        f"registry declares {sorted(declared)}, cli offers {sorted(STACKS)} (+ both)")


def test_a_family_does_not_straddle_the_two_halves() -> None:
    by_family: dict[str, set[str]] = defaultdict(set)
    for r in _rows():
        by_family[r["family"]].add(r["stack"])
    split = {f: sorted(v) for f, v in by_family.items() if len(v) > 1}
    assert not split, (
        "families whose rows disagree about which half launches them -- one package is launched "
        f"by one module, so this is a hand-edit slip unless it is deliberate: {split}")


def test_asking_for_a_half_includes_the_shared_kernels() -> None:
    """`both` is built by either half: missing a kernel costs more than building it twice."""
    from miniworld_engine.autotune.builder import op_units

    rows = {r["kernel"]: r for r in _rows()}
    trunk = {u.op for u in op_units(stack="trunk")}
    diffusion = {u.op for u in op_units(stack="diffusion")}
    assert trunk
    assert diffusion
    assert not [k for k in trunk if rows[k]["stack"] == "diffusion"]
    assert not [k for k in diffusion if rows[k]["stack"] == "trunk"]
    shared = {k for k, r in rows.items() if r["stack"] == "both" and r["backend"] == "triton"}
    built = {u.op for u in op_units()}
    assert shared & built <= trunk & diffusion, "a `both` kernel is missing from one of the halves"
    assert trunk | diffusion == built, "every built op belongs to at least one half"
