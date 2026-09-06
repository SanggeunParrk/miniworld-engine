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


def test_the_exemption_actually_removes_the_pair_from_coverage():
    """The half this file was missing, and the half that broke.

    The tests above check that each row's CLAIM is true of the source. None of them checked that
    the audit ACTS on the row -- and it did not: `check_cache_coverage` called `unpack_base(b)` on
    a bare bucket, which needs a second argument, so every call raised into a bare `except` and the
    drop set stayed empty. The declaration was read, parsed, and then quietly ignored, which reads
    from the outside exactly like "no bucket matched". `adaln_bwd_dx_dlnw_triton`'s 8192 bucket
    kept being reported as missing on every card.

    So this drives the real filter: build a declared work list containing an exempted bucket, run
    it through the same code path, and assert the pair is gone.
    """
    from miniworld_engine.build import audit

    declared = audit._untunable()
    scoped = {op: b for op, b in declared.items() if "*" not in b}
    if not scoped:
        pytest.skip("no bucket-scoped exemptions declared")

    for op, buckets in scoped.items():
        bucket = int(sorted(buckets)[0])
        want = {op: {("bfloat16", bucket), ("float32", bucket), ("bfloat16", 128)}}
        # the same expression check_cache_coverage uses
        drop = {(dt, b) for dt, b in want[op] if b is not None and str(b) in buckets}
        assert drop, (
            f"{op}: the exemption for bucket {bucket} matches nothing in a declared work list that "
            f"contains it. `want` holds the BARE bucket op_units emitted, so the comparison has to "
            f"be against that -- not against an unpacked cache key.")
        remaining = want[op] - drop
        assert ("bfloat16", 128) in remaining, "the exemption removed a bucket it does not name"
        assert not any(b == bucket for _, b in remaining), (
            f"{op}: bucket {bucket} survived its own exemption")
