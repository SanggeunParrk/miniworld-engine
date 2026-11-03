"""What each release action does, derived from the code graph and nothing else (``miniworld-engine status``).

The graph (``build.repo_graph``) is run once and reduced to :class:`Facts`: the files that differ from the base, the test files it names,
the derivation units that recorded one of those files, the CUDA artifacts that rebuild, the benchmark modules it reaches, and what is owed on
cards this machine cannot run. Each ``*_scope`` function turns the facts into a :class:`Scope`: the items to run, the items already done at
identical inputs (the ledger, ``action_records``), the gaps. No function here takes a path, a filter or a family from the caller; the only way
to widen a scope is ``all_reason`` (``--all --reason``), which makes the scope the whole universe and is recorded.

This module imports no torch and runs no GPU code: it is pure data in, data out, so a unit test feeds it a synthetic graph.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath

from miniworld_engine.build import action_records as rec

SRC = "src/miniworld_engine"


@dataclasses.dataclass
class Facts:
    """The code graph's answer for one base and one accessible card, reduced to what the actions need."""

    arch: str
    base: str
    changed: list[str] = dataclasses.field(default_factory=list)                 # repo-relative files that differ from the base
    unit_files: dict[str, list[str]] = dataclasses.field(default_factory=dict)   # derivation unit label -> repo-relative files it ran
    stale_units: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # unit label -> files that no longer match its recorded token
    tests_cpu: list[str] = dataclasses.field(default_factory=list)
    tests_gpu: list[str] = dataclasses.field(default_factory=list)
    reaching: dict[str, list[str]] = dataclasses.field(default_factory=dict)     # node (test / benchmark) -> changed files that reach it
    bench_ops: list[str] = dataclasses.field(default_factory=list)               # benchmark modules the change reaches
    kernel_inputs: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # registry.csv kernel reached -> changed files reaching it
    owed: dict[str, list[str]] = dataclasses.field(default_factory=dict)         # other card -> lines (never done here, never claimed)
    universe: dict[str, list[str]] = dataclasses.field(default_factory=dict)     # "tests_cpu" / "tests_gpu" / "bench_ops": everything (for --all)
    graph: object = dataclasses.field(default=None, repr=False, compare=False)    # the repository graph it was computed on (reused)


@dataclasses.dataclass
class Scope:
    action: str
    arch: str
    base: str
    items: dict[str, str] = dataclasses.field(default_factory=dict)      # item -> digest of its inputs: what runs
    done: list[str] = dataclasses.field(default_factory=list)            # items already successful at identical inputs
    gaps: list[str] = dataclasses.field(default_factory=list)            # things the graph names that this action cannot map (never widened)
    owed: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    all_reason: str | None = None
    payload: dict = dataclasses.field(default_factory=dict)              # what the executor needs (case descriptors, row numbers, ...)

    @property
    def empty(self) -> bool:
        return not self.items

    def names(self) -> list[str]:
        return sorted(self.items)

    def summary(self) -> dict:
        return {"items": len(self.items), "done": len(self.done), "gaps": len(self.gaps), "names": self.names()[:200],
                "owed": {c: len(v) for c, v in self.owed.items()}, "all": self.all_reason}

    def block(self, tree: str = "", limit: int = 40) -> str:
        head = f"[{self.action}] arch {self.arch}  base {self.base}" + (f"  tree {tree[:12]}" if tree else "")
        lines = [head]
        if self.all_reason is not None:
            lines.append(f"  FULL RUN (--all): reason: {self.all_reason}")
        lines.append(f"  scope: {len(self.items)} to run, {len(self.done)} already done at identical inputs, {len(self.gaps)} not mappable")
        names = self.names()
        lines.extend(f"    {n}" for n in names[:limit])
        if len(names) > limit:
            lines.append(f"    ... {len(names) - limit} more")
        lines.extend(f"  gap: {g}" for g in self.gaps[:10])
        for card, items in sorted(self.owed.items()):
            lines.append(f"  UNVERIFIABLE-HERE (owed on {card}): " + "; ".join(items[:3]))
        if self.empty:
            lines.append("  nothing to do")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------ scope helpers
def _pending(inputs: dict[str, list[str]], ledger: dict[str, str], read: Callable[[str], str] | None, repo=None,
             extra: Callable[[str], str] | None = None) -> tuple[dict[str, str], list[str]]:
    todo: dict[str, str] = {}
    done: list[str] = []
    for item, files in inputs.items():
        digest = rec.content_digest(files, repo, read) + (extra(item) if extra else "")
        if ledger.get(item) == digest:
            done.append(item)
        else:
            todo[item] = digest
    return todo, sorted(done)


def _scope(action: str, facts: Facts, all_reason: str | None) -> Scope:
    return Scope(action, facts.arch, facts.base, all_reason=all_reason, owed=dict(facts.owed))


def _everything(scope: Scope, names: Iterable[str]) -> Scope:
    scope.items = dict.fromkeys(names, "all")
    return scope


def _named_or_owed(scope: Scope, named: Iterable[str], owed: Iterable[str], universe: Iterable[str]) -> Scope:
    """What the graph names for the diff since the action's last run, plus what that run left unfinished (failed, not run) and still exists."""
    alive = set(universe)
    scope.items = dict.fromkeys(named, "graph")
    scope.items.update({n: "owed" for n in owed if n in alive and n not in scope.items})
    return scope


def test_scope(facts: Facts, owed: Iterable[str] = (), all_reason: str | None = None) -> Scope:
    """The test files the graph names (CPU and GPU) for the changes since the last ``test`` run (``facts.base``), plus the files that run left
    failed or not run."""
    scope = _scope("test", facts, all_reason)
    scope.payload = {"cpu": [], "gpu": []}
    universe = [*facts.universe.get("tests_cpu", []), *facts.universe.get("tests_gpu", [])]
    if all_reason is not None:
        _everything(scope, universe)
    else:
        _named_or_owed(scope, [*facts.tests_cpu, *facts.tests_gpu], owed, universe)
    gpu = set(facts.universe.get("tests_gpu", [])) | set(facts.tests_gpu)
    for t in scope.names():
        scope.payload["gpu" if t in gpu else "cpu"].append(t)
    return scope


def derive_scope(facts: Facts, all_reason: str | None = None) -> Scope:
    """The derivation units whose recorded files no longer match (``dev derive`` reuses every other unit; ``--all`` re-records everything)."""
    scope = _scope("derive", facts, all_reason)
    if all_reason is not None:
        return _everything(scope, facts.unit_files)
    scope.items = {label: ",".join(files) for label, files in sorted(facts.stale_units.items())}
    return scope


def bench_scope(facts: Facts, owed: Iterable[str] = (), all_reason: str | None = None) -> Scope:
    """The benchmark modules the graph reaches for the changes since the last ``bench`` run, plus the ones that run left unmeasured."""
    scope = _scope("bench", facts, all_reason)
    if all_reason is not None:
        return _everything(scope, facts.universe.get("bench_ops", []))
    return _named_or_owed(scope, facts.bench_ops, owed, facts.universe.get("bench_ops", []))


# ------------------------------------------------------------------------------------------------------------------ unit <-> case mapping
def unit_label(call: dict, dims: dict) -> str:
    """``autotune.derive.DeriveUnit.label`` of a registry call (``registry_calls.Call.to_json()``): how a derivation unit names the call."""
    d = ";".join(f"{k}={v}" for k, v in dims.items())
    opt = f" {call['option'][0]}={call['option'][1]}" if call.get("option") else ""
    core = f"->{call['compute']}" if call.get("compute") else ""
    return (f"{call['module']}[{call['impl']}/{call['dtype']}{core}] stream={call['stream']} {d} "
            f"L={call['length']} A={call['augmentation']} {call['mode']}{opt}")


# ------------------------------------------------------------------------------------------------------------------ the bundle against the tree
@dataclasses.dataclass
class Rows:
    """The ``registry_module.csv`` rows of the accessible card and what the last verify learned about them."""

    keys: dict[int, str]                                                         # row number -> key (module/stream/dims/A), rows with calls here
    text: dict[str, str]                                                         # key -> digest of what the row declares
    recorded: dict[str, dict] = dataclasses.field(default_factory=dict)          # key -> {text, artifacts, served, inputs}: last clean verify
    reported_missing: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # key -> names the last verify found missing
    inputs: dict[str, str] = dataclasses.field(default_factory=dict)             # key -> digest of the files its derivation units ran, now

    def number(self, key: str) -> int:
        return next(n for n, k in self.keys.items() if k == key)


def stale_artifacts(artifacts: Iterable[dict], digest_of: Callable[[str], str | None]) -> dict[str, list[str]]:
    """Artifact name -> the source nodes (``pkg:...`` labels) whose recorded digest differs from the tree now. ``digest_of(label)`` is the
    tree's digest, ``"absent"`` for a file that is gone, None for a node the tree does not own (an external header): not judged."""
    out: dict[str, set[str]] = {}
    for a in artifacts:
        for label, recorded in (a.get("material") or {}).get("nodes") or ():
            now = digest_of(label)
            if now is not None and now != recorded:
                out.setdefault(a["name"], set()).add(label)
    return {n: sorted(v) for n, v in sorted(out.items())}


def owed_rows(rows: Rows, names: set[str], stale: dict[str, list[str]]) -> tuple[dict[str, str], list[str]]:
    """``(key -> why the bundle owes the row a coverage run, stale artifacts no recorded row produces)``. A row is owed when it is new or
    edited since its record, when an artifact it was served from is missing from the bundle or stale against the tree, or when the last
    verify found it missing artifacts (a dispatch change asked for a key the bundle never had)."""
    why: dict[str, str] = {}
    for key, text in rows.text.items():
        rec_ = rows.recorded.get(key)
        if rec_ is None:
            why[key] = "new row (no verify record)"
        elif rec_.get("text") != text:
            why[key] = "row edited since its verify record"
        elif missing := sorted(set(rec_["artifacts"]) - names):
            why[key] = "not in the bundle: " + ", ".join(missing)
        elif st := sorted(set(rec_["artifacts"]) & set(stale)):
            why[key] = "stale against the tree: " + ", ".join(st)
        if key in rows.reported_missing and key not in why:
            why[key] = "the last verify found missing: " + ", ".join(rows.reported_missing[key])
    produced = {n for r in rows.recorded.values() for n in r["artifacts"]}
    return why, sorted(n for n in stale if n not in produced)


def _row_digest(key: str, rows: Rows, why: str, stale: dict[str, list[str]], digest_of: Callable[[str], str | None]) -> str:
    """The inputs of a row's coverage run: what it declares, why it is owed, and the tree's digest of every stale node of its artifacts (the same
    edits are not built twice)."""
    rec_ = rows.recorded.get(key) or {"artifacts": []}
    nodes = sorted({f"{lab}={digest_of(lab)}" for n in rec_["artifacts"] for lab in stale.get(n, ())})
    return hashlib.sha256("\n".join([rows.text.get(key, ""), why, *nodes]).encode()).hexdigest()


def bundle_scope(facts: Facts, cases: list[dict], rows: Rows, names: set[str], stale: dict[str, list[str]], ledger: dict[str, str],
                 digest_of: Callable[[str], str | None] = lambda _: None, read=None, all_reason: str | None = None, repo=None, owed: Iterable[str] = ()) -> Scope:
    """What the bundle owes, from the bundle itself against the tree (not against the base): the registry rows :func:`owed_rows` names, and
    the ``registry.csv`` kernel rows the graph reached. A row runs all its coverage calls (an unchanged key is a prebuilt hit, so only the
    owed keys compile). ``--all``: every case."""
    scope = _scope("bundle", facts, all_reason)
    if all_reason is not None:
        _everything(scope, {f"row:{rows.keys[c['row']]}" if c["kind"] != "registry_kernel" else f"kernel:{c['kernel']}" for c in cases
                            if c["kind"] == "registry_kernel" or c["row"] in rows.keys})
        scope.payload = {"cases": cases}
        return scope
    why, orphans = owed_rows(rows, names, stale)
    inputs = {f"row:{k}": _row_digest(k, rows, w, stale, digest_of) for k, w in why.items()}
    for c in cases:
        if c["kind"] == "registry_kernel" and facts.kernel_inputs.get(c["kernel"]):
            inputs[f"kernel:{c['kernel']}"] = rec.content_digest(facts.kernel_inputs[c["kernel"]], repo, read)
        elif c["kind"] == "registry_kernel" and f"kernel:{c['kernel']}" in set(owed):
            inputs[f"kernel:{c['kernel']}"] = "owed"
    # a row owed by an earlier run (not run, or failing under --retry-failing) runs its coverage again although its artifacts are fresh
    inputs.update({k: "owed" for k in owed if k.startswith("row:") and k.removeprefix("row:") in rows.text and k not in inputs})
    scope.items = {k: d for k, d in inputs.items() if ledger.get(k) != d}
    scope.done = sorted(k for k, d in inputs.items() if ledger.get(k) == d)
    scope.gaps = [f"stale artifact no registry row produces (kept, not rebuilt; declare its row): {n} ({', '.join(stale[n][:2])})" for n in orphans]
    pick = {k.removeprefix("row:") for k in scope.items if k.startswith("row:")}
    kern = {k.removeprefix("kernel:") for k in scope.items if k.startswith("kernel:")}
    scope.payload = {"cases": [c for c in cases if (c["kind"] == "registry_kernel" and c["kernel"] in kern)
                               or (c["kind"] != "registry_kernel" and rows.keys.get(c["row"]) in pick)],
                     "why": {k: why.get(k.removeprefix("row:"), "owed by an earlier run (failing, or not reached)") for k in scope.items if k.startswith("row:")}, "orphans": orphans}
    return scope


def owed_verify(rows: Rows, fingerprint: Callable[[str], str] | None = None) -> dict[str, str]:
    """``key -> why`` the row must run again under an empty cache with JIT off, judged against its own clean record (not against the base):
    new or edited, the last verify found it missing artifacts, an artifact it is served from was rebuilt (``fingerprint(key)`` vs the
    record's ``served``), or a file its derivation units ran changed since (``inputs``)."""
    why: dict[str, str] = {}
    for key, text in rows.text.items():
        r = rows.recorded.get(key)
        if r is None:
            why[key] = "new row (no verify record)"
        elif r.get("text") != text:
            why[key] = "row edited since its verify record"
        elif key in rows.reported_missing:
            why[key] = "the last verify found missing: " + ", ".join(rows.reported_missing[key])
        elif fingerprint is not None and r.get("served") is not None and fingerprint(key) != r["served"]:
            why[key] = "an artifact it is served from was rebuilt"
        elif r.get("inputs") is not None and rows.inputs.get(key) is not None and rows.inputs[key] != r["inputs"]:
            why[key] = "a file its derivation units ran changed since its verify record"
    return why


def verify_scope(facts: Facts, cases: list[dict], rows: Rows, ledger: dict[str, str], fingerprint: Callable[[str], str] | None = None,
                 read=None, all_reason: str | None = None, repo=None, owed: Iterable[str] = (), bundle_fp: str = "") -> Scope:
    """The rows :func:`owed_verify` names, and the ``registry.csv`` kernel rows the graph reached. ``--all``: every row and kernel."""
    scope = _scope("verify", facts, all_reason)
    kernels = sorted({c["kernel"] for c in cases if c["kind"] == "registry_kernel"})
    if all_reason is not None:
        _everything(scope, [*(f"row:{k}" for k in rows.text), *(f"kernel:{k}" for k in kernels)])
        why = {}
    else:
        why = owed_verify(rows, fingerprint)
        # a row with no record is judged against the whole bundle (it has no artifact list yet): a rebuilt bundle reruns it, the same one does not
        inputs = {f"row:{k}": "|".join([rows.text[k], rows.inputs.get(k, ""), fingerprint(k) if fingerprint else "", w,
                                        bundle_fp if k not in rows.recorded else ""]) for k, w in why.items()}
        inputs.update({f"kernel:{k}": rec.content_digest(facts.kernel_inputs[k], repo, read) for k in kernels if facts.kernel_inputs.get(k)})
        inputs.update({f"kernel:{k}": "owed" for k in kernels if f"kernel:{k}" in set(owed) and f"kernel:{k}" not in inputs})
        scope.items = {k: d for k, d in inputs.items() if ledger.get(k) != d}
        scope.done = sorted(k for k, d in inputs.items() if ledger.get(k) == d)
    scope.payload = {"rows": sorted(rows.number(u.removeprefix("row:")) for u in scope.items if u.startswith("row:")),
                     "kernels": sorted(u.split(":", 1)[1] for u in scope.items if u.startswith("kernel:")),
                     "why": {u: why[u.removeprefix("row:")] for u in scope.items if u.startswith("row:") and u.removeprefix("row:") in why}}
    return scope


def row_inputs(rows_calls: dict[str, list[dict]], dims: dict[str, dict], unit_files: dict[str, list[str]], digest: Callable[[list[str]], str]) -> dict[str, str]:
    """``key -> digest of the files the row's derivation units ran`` (``unit_files``: unit label -> repo files, from the card's plan). A row
    with no derivation unit has no inputs (None: not judged on them)."""
    out = {}
    for key, calls in rows_calls.items():
        files = sorted({f for c in calls for f in unit_files.get(unit_label(c, dims[key]), ())})
        if files:
            out[key] = digest(files)
    return out


def bench_op_of(path: str) -> str | None:
    """``benchmarks/modules/<op>/...`` -> ``<op>``."""
    parts = PurePosixPath(path).parts
    return parts[2] if len(parts) > 2 and parts[:2] == ("benchmarks", "modules") else None


# ------------------------------------------------------------------------------------------------------------------ the other CUDA builds
def owed_replays(manifests: dict[str, dict]) -> dict[tuple[str, str], list[str]]:
    """``(build, python) -> artifact names that build lacks`` for every CUDA build of the arch (``<arch>-cu<XY>``) and every Python ABI any of
    them ships. The coverage runs in ONE environment (one CUDA build, one Python), so the keys it adds land in that build only; each other build
    owes a replay of their recipes. A cubin does not depend on the Python ABI (owed once per build, under its first ABI); an extension does."""
    def py(a: dict) -> str:
        return (a.get("material") or {}).get("python") or "-"

    cubins = {a["name"] for mf in manifests.values() for a in mf.get("artifacts", {}).values() if a["kind"] == "cubin"}
    exts = {a["name"] for mf in manifests.values() for a in mf.get("artifacts", {}).values() if a["kind"] != "cubin"}
    pythons = sorted({p for mf in manifests.values() for p in mf.get("pythons", [])})
    out: dict[tuple[str, str], list[str]] = {}
    for build, mf in sorted(manifests.items()):
        arts = list(mf.get("artifacts", {}).values())
        missing_cubins = cubins - {a["name"] for a in arts if a["kind"] == "cubin"}
        for i, p in enumerate(pythons):
            miss = (exts - {a["name"] for a in arts if a["kind"] != "cubin" and py(a) == p}) | (missing_cubins if i == 0 else set())
            if miss:
                out[(build, p)] = sorted(miss)
    return out
