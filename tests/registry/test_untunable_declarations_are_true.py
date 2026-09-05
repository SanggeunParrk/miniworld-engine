"""`kernels/untunable.csv` is a claim about the SOURCE, so check it against the source.

An exemption list is the most dangerous kind of documentation: it makes an audit go quiet, and a
quiet audit is indistinguishable from a passing one. Each row here says "the launcher pins this
kernel's config instead of tuning it", which is true only if some launcher calls the kernel through
`.fn[...]` -- triton's raw JIT handle, which bypasses the autotune wrapper. That is greppable, so
it is checked rather than trusted.

What this does NOT check is the RANGE: whether the bypass covers every width the build drives (the
`*` rows) or one bucket (the numbered ones). That needs the launcher's guard condition, which is
prose. The reason column carries it, and `test_every_untunable_row_explains_itself` makes the
prose mandatory.
"""
from __future__ import annotations

import csv
import re

import pytest
from paths import REGISTRY, ROOT

UNTUNABLE = REGISTRY.parent / "untunable.csv"
KERNELS = ROOT / "src" / "miniworld_engine" / "kernels"


@pytest.fixture(scope="module")
def rows():
    if not UNTUNABLE.is_file():
        pytest.skip("no untunable.csv in this checkout")
    with UNTUNABLE.open(newline="") as fh:
        out = list(csv.DictReader(fh))
    assert out, "untunable.csv exists but declares nothing"
    return out


@pytest.fixture(scope="module")
def registry():
    with REGISTRY.open(newline="") as fh:
        return {r["kernel"]: r for r in csv.DictReader(fh)}


def test_every_untunable_kernel_is_a_real_registry_row(rows, registry):
    unknown = sorted(r["kernel"] for r in rows if r["kernel"] not in registry)
    assert not unknown, f"untunable.csv names kernels registry.csv does not: {unknown}"


def test_every_untunable_kernel_really_has_a_bypassing_launcher(rows, registry):
    """The claim is `.fn[` on this kernel's symbol somewhere in its own file."""
    bad = []
    for r in rows:
        row = registry[r["kernel"]]
        src = ROOT / "src" / row["file"]
        symbol = row["symbol"].split(".")[-1]
        if not src.is_file():
            bad.append((r["kernel"], f"{row['file']} not readable"))
            continue
        if not re.search(rf"\b{re.escape(symbol)}\.fn\s*\[", src.read_text()):
            bad.append((r["kernel"], f"no `{symbol}.fn[` in {row['file']}"))
    assert not bad, (
        f"untunable.csv claims the launcher pins the config, but the bypass is not there: {bad}. "
        f"Either the launcher was changed to tune again -- drop the row, the audit should see the "
        f"op -- or the claim was never true.")


def test_no_bypassing_kernel_is_left_undeclared(rows, registry):
    """The other direction: a NEW `.fn[` bypass must be declared, or the audit fails forever.

    This is the check that would have caught the two rows here before they became a standing FAIL
    nobody could clear by building.
    """
    declared = {r["kernel"] for r in rows}
    found = []
    for kernel, row in registry.items():
        if row["backend"] != "triton" or (row.get("developed") or "yes").strip() == "no":
            continue
        src = ROOT / "src" / row["file"]
        symbol = row["symbol"].split(".")[-1]
        if not src.is_file():
            continue
        if re.search(rf"\b{re.escape(symbol)}\.fn\s*\[", src.read_text()) and kernel not in declared:
            found.append(kernel)
    assert not found, (
        f"these kernels are launched through `.fn[` -- bypassing the autotuner -- and are not in "
        f"untunable.csv: {sorted(found)}. Declare which buckets the bypass covers (`*` for all) "
        f"and why, or the coverage audit reports a hole no build can fill.")


def test_every_untunable_row_explains_itself(rows):
    """A row with no reasoning is an exemption nobody can review."""
    thin = [(r["kernel"], len(r.get("reason") or "")) for r in rows
            if len(r.get("reason") or "") < 200]
    assert not thin, (
        f"untunable.csv rows with no real reason: {thin}. Say WHICH launcher bypasses, under what "
        f"condition, and why that is right -- an audit exemption is read by whoever inherits it.")


def test_bucket_scopes_are_wildcards_or_integers(rows):
    bad = []
    for r in rows:
        for b in (r.get("buckets") or "*").split("|"):
            b = b.strip()
            if b != "*" and not b.isdigit():
                bad.append((r["kernel"], b))
    assert not bad, f"bucket scope must be `*` or an integer bucket base: {bad}"
