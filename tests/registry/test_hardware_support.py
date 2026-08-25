"""The README's hardware table is the registry's `arch` column, or it is fiction.

A kernel library is not portable by default, and until this file existed nothing in the repo said
which architectures it supports. The information was there -- as `sm90`/`sm100` in file names, as
`-gencode` lists in `kernels/<family>/cuda/setup.py`, as asserts inside individual checkers ("SM90
(H100) only") -- and a consumer had to read the source to assemble it.

`registry.csv` now declares it per kernel, and the README renders it. Rendering by hand is how a
table goes stale, so this file regenerates it and compares: change a kernel's `arch` without
touching the README and the diff shows up here.

The `arch` value is a MINIMUM, and the check below pins it to evidence rather than to taste: a
kernel whose name or file says `sm100` may not claim `sm80`, and a CuTeDSL kernel may not claim the
Triton floor.
"""
from __future__ import annotations

import collections
import csv
import re
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REGISTRY = REPO / "src/miniworld_engine/kernels/registry.csv"
README = REPO / "README.md"
BEGIN = "<!-- BEGIN GENERATED: hardware-support -->"
END = "<!-- END GENERATED: hardware-support -->"

#: arch -> the cards this repo has actually run on at that level. Committed result tables under
#: `benchmarks/**/results/<gpu>/` are the evidence for the sm80 row.
GPUS = {
    "sm80": "A100, A5000, A6000, RTX 4090",
    "sm90": "H100",
    "sm100": "B200",
}
ORDER = ["sm80", "sm90", "sm100"]


def _rows() -> list[dict]:
    with REGISTRY.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _render(rows: list[dict]) -> str:
    by_arch: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_arch[r["arch"]].append(r)
    out = ["| arch | GPUs | kernels | backends |", "|---|---|---|---|"]
    for arch in ORDER:
        group = by_arch.get(arch, [])
        if not group:
            continue
        backends = collections.Counter(r["backend"] for r in group)
        detail = ", ".join(f"{b} {n}" for b, n in backends.most_common())
        out.append(f"| **{arch}+** | {GPUS[arch]} | {len(group)} | {detail} |")
    return "\n".join(out)


def test_every_kernel_declares_an_arch() -> None:
    rows = _rows()
    assert rows, "registry.csv is empty"
    assert "arch" in rows[0], "registry.csv has no arch column"
    missing = [r["kernel"] for r in rows if r["arch"] not in ORDER]
    assert not missing, f"{missing} declare an arch outside {ORDER}"


def test_the_declared_arch_matches_the_evidence_in_the_name() -> None:
    """A name saying `sm100` has to be accounted for by one of the two columns.

    `arch` is the ENFORCED minimum -- below it the kernel is not launched. `tuned_for` is what it
    was written against. The name is evidence for one of them, and the check is which.

    They used to be one column, and the conflation cost coverage: three triton kernels live inside
    sm100-named cute modules (`_grad_mul_inplace` in `transition/cute/gatebwd_sm100.py` and two
    like it), and their drivers launch the triton kernel directly rather than building the cute
    pipeline around it. Gating on the module's name skipped them on sm86, where they run fine --
    measured, before this was relaxed: launch ok, check ok, `rel dA=1.99e-03` against a `1e-02`
    band. `driven` went 94 -> 97 and `skipped` 9 -> 6.
    """
    wrong = []
    for r in _rows():
        blob = f"{r['kernel']} {r['file']} {r['symbol']}"
        for marker in ("sm100", "sm90"):
            if not re.search(marker, blob):
                continue
            if marker in (r["arch"], r.get("tuned_for") or ""):
                break                      # accounted for by one column or the other
            wrong.append(
                f"{r['kernel']}: name says {marker}, arch={r['arch']} tuned_for="
                f"{r.get('tuned_for') or '(empty)'}. Set `arch` if the hardware is REQUIRED, or "
                f"`tuned_for` if it merely names what it was written against.")
            break
    assert not wrong, wrong


def test_tuned_for_is_only_used_to_relax_not_to_tighten() -> None:
    """`tuned_for` is informational. A row using it must still be launchable at its `arch`, so it
    may only sit ABOVE `arch` -- otherwise it is a second, quieter gate."""
    from miniworld_engine.autotune.run_all import _sm

    bad = [f"{r['kernel']}: arch={r['arch']} tuned_for={r['tuned_for']}"
           for r in _rows() if (r.get("tuned_for") or "").strip()
           and _sm(r["tuned_for"]) <= _sm(r["arch"])]
    assert not bad, (
        "`tuned_for` at or below `arch` says nothing the enforced column does not:\n  "
        + "\n  ".join(bad))


def test_no_cutedsl_kernel_claims_the_triton_floor() -> None:
    """CuTeDSL/quack GEMMs are Hopper-and-later. One claiming sm80 would put it in the row a
    consumer reads as "runs on my A6000"."""
    bad = [r["kernel"] for r in _rows() if r["backend"] == "cute" and r["arch"] == "sm80"]
    assert not bad, bad


def test_every_arch_above_the_floor_has_a_portable_fallback() -> None:
    """The promise the README makes: an unsupported card loses performance, not function.

    Every FAMILY that has an sm90/sm100 kernel must also have a kernel at the floor, so
    `modules/dispatch.py` has somewhere to fall back to.
    """
    rows = _rows()
    at_floor = {r["family"] for r in rows if r["arch"] == "sm80"}
    above = {r["family"] for r in rows if r["arch"] != "sm80"}
    # Not vacuous: there ARE kernels above the floor (9 today, across 5 families). If that ever
    # reaches zero this assertion stops meaning anything, so it fails instead.
    assert above, "no kernel declares an arch above the floor; this test now checks nothing"
    stranded = sorted(above - at_floor)
    assert not stranded, (
        f"families with no sm80 kernel at all: {stranded}. An op reachable only above the floor "
        f"makes the library non-functional on an A6000, not merely slower.")


def test_the_readme_table_is_what_the_registry_says() -> None:
    text = README.read_text()
    for marker in (BEGIN, END):
        assert marker in text, f"{marker} is gone from README.md; the table is no longer generated"
    block = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    expected = _render(_rows())
    assert block == expected, (
        "README.md's hardware table no longer matches registry.csv's arch column.\n"
        f"--- README has:\n{block}\n--- registry says:\n{expected}")
