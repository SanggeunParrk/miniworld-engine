"""Which autotune-cache units a given GPU may build: one CSV per architecture in gpu_to_kernels/.

The rule this encodes is narrow and always the same: an implementation written for one architecture
produces NOTHING on another. It does not degrade, it raises -- quack/CuTe's ``Gemm Sm80 is not
implemented yet``, or the hand-CUDA extension failing to build off sm_90a. The unit still pays for
itself in full first, because the failure lands after the autotune grid has been compiled and
benchmarked: on sm_86 the ``cute`` units of a from-scratch build cost 7-14 GPU-hours per shard and
wrote zero entries.

One file per card rather than one table with an ``sm`` column, matching how ``autotune/data``
already keys its caches per GPU: the question people actually arrive with is "what does THIS card
build", and that should be one file to open, not a filter to run in your head. It also means adding
a card is adding a file -- no edit to a shared table that every other card also reads.

The cost of that split is duplication for anything true everywhere, so those rules live in
``_common.csv`` instead of being pasted into four files where they would drift.

THE FILES SUBTRACT. Everything ``builder.cases()`` enumerates is built unless a row says otherwise.
An allow-list would read as tidier and would be a trap: a newly added Case would silently go
unbuilt on every card until someone remembered to list it, and an unvisited bucket is not a smaller
cache -- it is a full-grid autotune stall inside a production forward, months later, nowhere near
this directory. A row may only ever mean "this card cannot run this", never "this is slow".
"""

from __future__ import annotations

import csv
import dataclasses
import functools
from pathlib import Path

RULES_DIR = Path(__file__).parent / "gpu_to_kernels"

#: architecture-independent rules, applied before the per-GPU file.
COMMON_STEM = "_common"

#: Lines before the header, kept as prose in the same file the rules live in. csv.reader has no
#: comment support, so they are stripped here; '>' rather than '#' so a spreadsheet shows them as
#: text instead of hiding them.
COMMENT_PREFIX = ">"

_FIELDS = ("case", "impl", "dtype", "build", "reason")


@dataclasses.dataclass(frozen=True)
class Rule:
    case: str
    impl: str
    dtype: str
    build: bool
    reason: str
    source: str = ""

    def matches(self, case: str, impl: str, dtype: str) -> bool:
        return all(pattern in ("*", value) for pattern, value in
                   ((self.case, case), (self.impl, impl), (self.dtype, dtype)))


def sm_tag(capability: tuple[int, int]) -> str:
    """(8, 6) -> 'sm_86'. The stem of this card's file in ``gpu_to_kernels/``.

    Underscored, unlike ``autotune.cache.gpu_key``'s ``sm86``: that key names a tuned-cache file
    per DEVICE ("NVIDIA RTX A6000 (sm86)"), this one names a policy file per ARCHITECTURE, and one
    A6000 cache is not interchangeable with an A5000's even though both are sm_86.
    """
    return f"sm_{capability[0]}{capability[1]}"


def known_gpus() -> list[str]:
    """sm tags with a file. A card absent from this list still builds -- everything."""
    return sorted(p.stem for p in RULES_DIR.glob("sm_*.csv"))


def _parse(path: Path) -> tuple[Rule, ...]:
    with path.open(newline="") as handle:
        lines = [ln for ln in handle if not ln.startswith(COMMENT_PREFIX)]
    out = []
    for lineno, row in enumerate(csv.DictReader(lines), start=2):
        missing = [f for f in _FIELDS if row.get(f) is None]
        if missing:
            raise ValueError(f"{path.name}:{lineno}: missing column(s) {', '.join(missing)}")
        decision = row["build"].strip().lower()
        if decision not in ("yes", "no"):
            raise ValueError(f"{path.name}:{lineno}: build must be yes/no, got {row['build']!r}")
        reason = row["reason"].strip()
        if decision == "no" and not reason:
            raise ValueError(f"{path.name}:{lineno}: a 'no' row must say why")
        out.append(Rule(row["case"].strip(), row["impl"].strip(), row["dtype"].strip(),
                        decision == "yes", reason, source=path.name))
    return tuple(out)


@functools.cache
def rules(sm: str) -> tuple[Rule, ...]:
    """``_common.csv`` then ``<sm>.csv``, in that order -- so a card can reopen a common deny.

    A missing per-GPU file is not an error. An unrecognised card must build everything rather than
    nothing: the only safe default here is the expensive one.
    """
    out = list(_parse(RULES_DIR / f"{COMMON_STEM}.csv"))
    per_gpu = RULES_DIR / f"{sm}.csv"
    if per_gpu.exists():
        out.extend(_parse(per_gpu))
    return tuple(out)


def decide(sm: str, case: str, impl: str, dtype: str) -> tuple[bool, str]:
    """``(build?, reason)`` for one unit. Last matching row wins; unmatched units are built."""
    verdict, why = True, ""
    for rule in rules(sm):
        if rule.matches(case, impl, dtype):
            verdict, why = rule.build, rule.reason
    return verdict, why


def allows(sm: str, case: str, impl: str, dtype: str) -> bool:
    return decide(sm, case, impl, dtype)[0]
