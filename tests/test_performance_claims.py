"""A latency in a live doc must say what machine it came from.

`docs/library-standards.md` C1: a benchmark number is a claim about a machine, a config set, a
dtype, a compile mode and a version, and detached from those it is folklore. The committed CSVs
carry all of it per row. Prose does not, and prose is what people read.

This is enforced at FILE level, not per claim, and the reason is worth stating: the attribution for
an old number is often not recoverable. `docs/benchmarking-cautions.md` carried 17 latencies and
named a device once; back-filling the rest would mean guessing which card a 2026-07 trimul run
used, which is worse than saying it is unknown. So the rule is that a live doc making performance
claims must state the hardware behind them somewhere — which a provenance paragraph satisfies,
including one that says which numbers are unattributed and why.

DATED records are out of scope by design: `kernels/*/notes/**/v*.md` are per-version logs
of "on this date, this config measured this", and `docs/kernels/{naming-audit,tiling-audit}.md` say
in their own headers that they are records. A record does not go stale; a reference does.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Docs a reader takes as CURRENT. Anything here that quotes a latency or a speedup has to say on
#: what. Declared rather than globbed: the distinction between a reference and a record is a
#: judgement, and it belongs in one visible list instead of a path heuristic.
LIVE_DOCS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "benchmarks/RESULTS.md",
    "docs/benchmarks.md",
    "docs/benchmarking-cautions.md",
    "docs/operations/dispatch-cache.md",
)

#: A latency or a speedup. Deliberately narrow: version numbers, tolerances and counts are not
#: performance claims, and a rule that fires on `2.10.0` teaches people to ignore it.
# \u00d7 and \u00b5 spelled as escapes: written literally, ruff (RUF001) reads them as
# homoglyphs of `x` and `u`, which is a fair warning about identifiers and noise in a regex.
CLAIM = re.compile("\\d+\\.\\d+\\s*(?:ms|\u00b5s|us|x\\b|\u00d7)")
DEVICE = re.compile(r"A6000|A5000|A100|H100|B200|RTX\s*\d|sm\d{2,3}|4090", re.IGNORECASE)


@pytest.mark.parametrize("relpath", LIVE_DOCS)
def test_a_live_doc_with_performance_claims_names_its_hardware(relpath: str) -> None:
    path = REPO / relpath
    assert path.is_file(), f"{relpath} is in LIVE_DOCS but does not exist"
    text = path.read_text()
    claims = CLAIM.findall(text)
    if not claims:
        pytest.skip(f"{relpath} makes no performance claim")
    assert DEVICE.search(text), (
        f"{relpath} quotes {len(claims)} performance figure(s) ({claims[:4]}) and never names a "
        f"device. A latency without a machine is not a result -- add the hardware, or a provenance "
        f"paragraph saying which numbers are unattributed and why.")


def test_the_rule_is_not_vacuous() -> None:
    """At least one live doc must actually make a claim, or this file is asserting nothing."""
    with_claims = [d for d in LIVE_DOCS if CLAIM.search((REPO / d).read_text())]
    assert with_claims, f"no doc in LIVE_DOCS quotes a figure any more; is the list stale? {LIVE_DOCS}"


def test_dated_records_are_excluded_on_purpose_and_still_exist() -> None:
    """If the record trees vanish, the exclusion above is silently covering nothing -- and someone
    should notice that the reasoning in this file's docstring no longer applies."""
    records = [m for d in (REPO / "src/miniworld_engine/kernels").glob("*/notes")
           for m in d.rglob("*.md")]
    assert records, "no kernels/*/notes/**.md; the record/reference split needs revisiting"
