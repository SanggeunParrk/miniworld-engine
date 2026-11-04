"""The release actions: ``miniworld-engine status | test | bundle | verify | derive | bench | wheel | release``. Documented in docs/actions.md.

Each action is graph-native: it runs the code graph first (``repo_graph.changes`` against the last uploaded tag), reduces it to a scope
(``action_scope``), prints the scope BEFORE doing anything, does nothing when the scope is empty, executes only the scope, and writes a record
(``action_records``) that the downstream actions read. The caller passes no path, filter or family; the only widening is ``--all --reason``.
The low-level commands (``dev bundle-kernels`` / ``bundle-verify`` / ``build-wheel`` / ``derive`` ...) stay as the internals these call.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from miniworld_engine.build import action_records as rec
from miniworld_engine.build import action_scope as sc

SRC = sc.SRC
ROW_ARTIFACTS = "tests/cpu/builder/data/registry_row_artifacts_{arch}.json"
PREV_NAMES = "tests/cpu/builder/data/prebuilt_names_prev_release.txt"
ALLOWED_DROPS = "tests/cpu/builder/data/prebuilt_names_allowed_drops.txt"
MANIFESTS = f"{SRC}/autotune/manifests"
PROVENANCE = "#provenance"


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------------------------ graph -> facts
def registry_units(card: str) -> list[str]:
    """The derivation unit labels ``registry_module.csv`` expands to on ``card`` (``autotune.derive.units``; no GPU)."""
    from miniworld_engine.autotune import derive, module_registry

    return [u.label for u in derive.units(module_registry.module_rows(), arch=card)]


def plan_stale(card: str, path: Path | None) -> dict[str, list[str]]:
    """``{unit label: why}`` for the units of ``card``'s plan at ``path`` that must be derived again: its recorded units whose files moved,
    the units the registry asks for that it never recorded (a new row), and the recorded units no row asks for any more (a removed row)."""
    from miniworld_engine.autotune import plan

    current = set(registry_units(card))
    try:
        data = json.loads(path.read_text()) if path is not None else None
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        return {k: ["<no derivation plan for this card>"] for k in sorted(current)}
    stale = plan.stale_units(data, labels=sorted(current))
    stale.update({k: ["<row removed from registry_module.csv>"] for k in sorted(set(data.get("units", {})) - current)})
    return stale


def graph_facts(base: str, repo: Path, accessible: str = "", graph=None) -> sc.Facts:
    """Run the repository graph against ``base`` (a release tag, a revision, or an action's snapshot tree) and reduce it to
    :class:`action_scope.Facts` for the accessible card. ``graph``: an already built repository graph of this tree (reused)."""
    from miniworld_engine.autotune import plan
    from miniworld_engine.build import cards as card_mod
    from miniworld_engine.build import repo_graph as rg

    result = rg.changes(base, graph=graph)
    g = result["graph"]
    here, _ = card_mod.accessible(accessible)
    card = card_mod.sort_cards(here)[0]
    changed = list(result["changed"])
    reach_by = {p: set(rg.reach(g, [p], lazy=True)) for p in changed if p in g.nodes}
    seen = rg.view(result, card, lazy=True)
    plans = rg.derivation_plans()
    unit_files, stale = {}, {}
    # the units the registry asks for NOW: a row added since the plan was derived has units the plan never recorded (stale: no files),
    # a row removed leaves recorded units the plan must drop (:func:`plan_stale`)
    if card in plans:
        data = json.loads(plans[card].read_text())
        unit_files = {label: [f"{SRC}/{f}" for f in files] for label, files in plan.unit_file_sets(data).items()}
    stale = {k: [f"{SRC}/{f}" if not f.startswith("<") else f for f in v] for k, v in plan_stale(card, plans.get(card)).items()}
    runnable, needs = rg.plan_tests(result, "all", frozenset(here))
    reaching: dict[str, list[str]] = {}
    for t in runnable:
        reaching[t] = sorted(p for p, nodes in reach_by.items() if t in nodes)
    bench_nodes = [n for n in seen if g.nodes[n] == "benchmark" and sc.bench_op_of(n)]
    ops = sorted({sc.bench_op_of(n) for n in bench_nodes})
    for op in ops:
        reaching[f"bench:{op}"] = sorted(p for p, nodes in reach_by.items() if any(sc.bench_op_of(n) == op for n in nodes & set(bench_nodes)))
    kernel_inputs = {}
    row_nodes = {n.removeprefix("registry-row:") for n in seen if n.startswith("registry-row:")}
    for k in row_nodes:
        files = [p for p, nodes in reach_by.items() if f"registry-row:{k}" in nodes]
        if k in (result.get("rows") or []):
            files.append(rg.REGISTRY_CSV)
        if files:
            kernel_inputs[k] = sorted(set(files))

    def runs_here(n: str) -> bool:
        return g.nodes[n] == "test" or card in card_mod.expand(g.cards.get(n, card_mod.ALL), g.known_cards)

    names = [n for n in g.nodes if Path(n).name.startswith("test_") and n.endswith(".py") and (repo / n).is_file()]
    universe = {"tests_cpu": sorted(n for n in names if g.nodes[n] == "test"),
                "tests_gpu": sorted(n for n in names if g.nodes[n] == "test-gpu" and runs_here(n)),
                "bench_ops": sorted({sc.bench_op_of(n) for n in g.nodes if g.nodes[n] == "benchmark" and sc.bench_op_of(n)})}
    owed = {}
    for c, tests in needs.items():
        owed[c] = [f"{len(tests)} GPU test file(s)", "CUDA build / retune / bench on that card"]
    for c in rg.reached_cards(result):
        if c not in here and c not in owed and rg.hardware_bound(g, rg.view(result, c, lazy=True)):
            owed[c] = ["hardware evidence (tests, CUDA build, retune, bench)"]
    return sc.Facts(arch=card, base=base, changed=changed, unit_files=unit_files, stale_units=stale,
                    tests_cpu=[t for t in runnable if g.nodes[t] == "test"],
                    tests_gpu=[t for t in runnable if g.nodes[t] == "test-gpu"], reaching=reaching, bench_ops=ops,
                    kernel_inputs=kernel_inputs, owed=owed, universe=universe, graph=g)


class Ctx:
    """Everything an action needs before it touches anything: base, tree, version, reason, and the graph facts (computed once, on demand)."""

    def __init__(self, args: argparse.Namespace, action: str = "", graph: bool = True) -> None:
        from miniworld_engine.build import cards as card_mod

        self.repo = rec.repo_root()
        self.reason = rec.require_reason(getattr(args, "all", False), getattr(args, "reason", None))
        self.tree = rec.tree_hash(self.repo)
        self.version = rec.declared_version(self.repo)
        self.records = rec.records_root()
        here, _ = card_mod.accessible("")
        self.arch = card_mod.sort_cards(here)[0]
        self.explicit_base = getattr(args, "base", "") or ""
        self.retry = bool(getattr(args, "retry_failing", False))   # rerun the recorded failures (an environment fix the graph cannot see)
        self.action = action
        if action:
            os.environ["MINIWORLD_ENGINE_ACTION"] = action     # the internals this run calls (and the processes it starts) may run
        self._graph = None
        self._facts: dict[str, sc.Facts] = {}
        # The snapshot this run starts from: what it judged. Saved as the action's state when it finishes, so the next run diffs from here.
        self.start = rec.snapshot_tree(self.repo) if graph and action else ""
        self.base, self.base_why = self.base_of(action) if action else (self.explicit_base or rec.release_base(self.repo), "release")
        self.facts = self.facts_for(self.base) if graph else None

    def base_of(self, action: str) -> tuple[str, str]:
        """``--base`` when given, else where ``action`` last left off (its state's snapshot), else the last release."""
        if self.explicit_base:
            return self.explicit_base, "--base"
        return rec.action_base(action, self.arch, self.repo, self.records)

    def facts_for(self, base: str) -> sc.Facts:
        """The graph's facts for ``base`` (the repository graph is built once per run)."""
        if base not in self._facts:
            self._facts[base] = graph_facts(base, self.repo, graph=self._graph)
            self._graph = self._facts[base].graph
        return self._facts[base]

    def owed(self, action: str = "") -> list[str]:
        """What the action's last run did not run: owed again whatever the diff says."""
        state = rec.load_state(action or self.action, self.arch, self.records)
        return [*state.get("owed", []), *(state.get("failing", []) if self.retry else [])]

    def failing(self, action: str = "") -> list[str]:
        """What failed and has not been rerun since (no change reached it: the same inputs would fail the same way)."""
        return rec.load_state(action or self.action, self.arch, self.records).get("failing", [])

    def still_failing(self, ran: Iterable[str], failed: Iterable[str], alive: Iterable[str] | None = None) -> list[str]:
        """The failing set after this run: the earlier failures this run did not rerun, and this run's own failures; with ``alive``, only
        items that still exist (a removed test file or registry row is not failing, it is gone)."""
        out = (set(self.failing()) - set(ran)) | set(failed)
        return sorted(out & set(alive) | set(failed) if alive is not None else out)

    def ledger(self, action: str) -> dict[str, str]:
        """Items already done at identical inputs; with ``--retry-failing`` the recorded failures are not among them."""
        done = rec.load_ledger(action, self.arch, self.records)
        if self.retry:
            done = {k: v for k, v in done.items() if k not in set(self.failing(action))}
        return done

    def finish(self, action: str, scope: sc.Scope | None, result: str, extra: dict | None = None, owed: list[str] | None = None,
               failing: list[str] | None = None) -> Path:
        """Write the run's record; with ``owed`` (the items not run), also the action's state: the next run starts from this run's snapshot,
        owes ``owed`` and remembers ``failing`` (rerun only when a change reaches them)."""
        failing = self.failing(action) if failing is None else failing
        if failing and result in ("pass", "nothing"):
            result = "fail"                                      # an unchanged failure is still a failure: not rerun, not forgotten
        path = rec.write_record(action, self.arch, self.tree, base=self.base, scope=scope.summary() if scope else {}, result=result,
                                version=self.version, extra={"base_why": self.base_why, "failing": failing, **(extra or {})},
                                all_reason=self.reason, root=self.records)
        if owed is not None and self.start:
            rec.save_state(action, self.arch, self.start, owed, self.records, failing=failing)
        if failing:
            say(f"[{action}] still failing, not rerun (no change reached them since they failed): {len(failing)}: " + ", ".join(failing[:8]))
        say(f"[{action}] result: {result}; record {path}")
        return path

    def nothing(self, action: str, scope: sc.Scope, extra: dict | None = None) -> int:
        say(f"[{action}] nothing to do (the graph names nothing this action owes since {self.base_why}; no GPU, no cache touched)")
        path = self.finish(action, scope, "nothing", extra, owed=[])
        return 0 if json.loads(Path(path).read_text())["result"] == "nothing" else 1


def _gpus() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g for g in env.split(",") if g != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [g.strip() for g in out.splitlines() if g.strip()]


# ------------------------------------------------------------------------------------------------------------------ test
def _junit_files(xml_path: Path, files: list[str]) -> dict[str, bool]:
    """file -> passed, from a junit xml (a file with no test case in it did not run: failed)."""
    status = dict.fromkeys(files)
    mods = {f: f[:-3].replace("/", ".") for f in files}
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return dict.fromkeys(files, False)
    for case in root.iter("testcase"):
        cls = case.get("classname", "")
        # a module-level skip (``pytest.skip(allow_module_level=True)``) is reported with no classname and the module as the name
        cls = cls or case.get("name", "")
        for f, mod in mods.items():
            if cls == mod or cls.startswith(mod + "."):
                bad = case.find("failure") is not None or case.find("error") is not None
                status[f] = (status[f] is not False) and not bad
    return {f: bool(v) for f, v in status.items()}


def cpu_shards(files: list[str], cpus: int | None = None) -> list[list[str]]:
    """The CPU test files split round-robin into one shard per two CPUs of this allocation (at most 8, at least 1; none without files): each
    shard is one pytest process, so the imports are paid once per shard and the shards run at the same time (one process ran 76 files in
    4.5 minutes)."""
    if cpus is None:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    n = max(1, min(8, cpus // 2, len(files)))
    return [files[i::n] for i in range(n)] if files else []


def run_tests(scope: sc.Scope, repo: Path, logdir: Path) -> dict[str, bool]:
    """CPU files in parallel pytest processes (`cpu_shards`: one per two CPUs of the allocation, at most 8); GPU files as a work queue over
    the allocation's GPUs, one file per GPU at a time. A line is
    printed as each file finishes. Returns file -> passed."""
    logdir.mkdir(parents=True, exist_ok=True)
    cpu, gpu_files = scope.payload["cpu"], scope.payload["gpu"]
    gpus = _gpus()
    results: dict[str, bool] = {}
    lock = threading.Lock()
    t0 = time.monotonic()

    def report(f: str, ok: bool, secs: float, where: str) -> None:
        with lock:
            results[f] = ok
            say(f"[test] {time.monotonic() - t0:6.0f}s {'PASS' if ok else 'FAIL'} {f} ({secs:.0f}s, {where}) [{len(results)}/{len(scope.items)}]")

    def run_cpu_shard(i: int, files: list[str]) -> None:
        xml, log = logdir / f"cpu{i}.xml", logdir / f"cpu{i}.log"
        started = time.monotonic()
        with log.open("w") as fh:
            rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *files], cwd=repo,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        per = _junit_files(xml, files)
        for f in files:
            report(f, per[f] and rc in (0, 1, 5), (time.monotonic() - started) / len(files), f"cpu {i}")
        if rc not in (0, 5):
            say(f"[test] cpu pytest exit {rc}; log {log}")

    def run_gpu_queue(gpu: str, queue: list[str]) -> None:
        while True:
            with lock:
                if not queue:
                    return
                f = queue.pop(0)
            log = logdir / (f.replace("/", "__") + ".log")
            started = time.monotonic()
            with log.open("w") as fh:
                rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f], cwd=repo, stdout=fh,
                                    stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}).returncode
            report(f, rc in (0, 5), time.monotonic() - started, f"gpu {gpu}")
            if rc not in (0, 5):
                say(f"[test]   log {log}")

    workers = []
    if gpu_files and not gpus:
        say(f"[test] {len(gpu_files)} GPU test file(s) in scope but this allocation shows no GPU: not run, not claimed")
    queue = list(gpu_files) if gpus else []
    shards = cpu_shards(cpu)
    with cf.ThreadPoolExecutor(max_workers=len(shards) + max(1, len(gpus))) as pool:
        workers.extend(pool.submit(run_cpu_shard, i, files) for i, files in enumerate(shards))
        workers.extend(pool.submit(run_gpu_queue, g, queue) for g in gpus)
        for w in workers:
            w.result()
    return results


def cmd_test(args: argparse.Namespace) -> int:
    ctx = Ctx(args, "test")
    scope = sc.test_scope(ctx.facts, ctx.owed(), all_reason=ctx.reason)
    say(f"[test] since {ctx.base_why} ({ctx.base[:12]}): {len(ctx.facts.changed)} changed file(s)")
    say(scope.block(ctx.tree))
    if scope.empty:
        return ctx.nothing("test", scope)
    results = run_tests(scope, ctx.repo, ctx.records / "logs" / f"test-{ctx.tree[:12]}")
    failed = sorted(f for f, ok in results.items() if not ok)
    not_run = sorted(set(scope.items) - set(results))
    say(f"[test] {len(results) - len(failed)} passed, {len(failed)} failed, {len(not_run)} not run")
    result = "fail" if failed else "partial" if not_run else "pass"
    ctx.finish("test", scope, result, {"failed": failed, "not_run": not_run}, owed=not_run,
               failing=ctx.still_failing(results, failed, alive=[*ctx.facts.universe.get("tests_cpu", []), *ctx.facts.universe.get("tests_gpu", [])]))
    return 0 if result == "pass" else 1


# ------------------------------------------------------------------------------------------------------------------ bundle
def prebuilt_manifests(root: Path | None, arch: str) -> list[Path]:
    from miniworld_engine.kernels import prebuilt

    return sorted((root or prebuilt.PREBUILT_DIR).glob(f"{arch}-cu*/{prebuilt.MANIFEST}"))


def bundle_digest(manifests: list[Path]) -> str:
    return rec.content_digest([str(m) for m in manifests], read=lambda p: rec.file_digest(Path(p))) if manifests else "none"


def manifest_names(manifests: list[Path]) -> set[str]:
    from miniworld_engine.build import kernel_bundle

    return set().union(*(kernel_bundle.artifact_names(json.loads(m.read_text())) for m in manifests)) if manifests else set()


def name_diff(names: set[str], repo: Path) -> dict:
    """The release-continuity diff: names the previous release carried that this bundle lacks, split into explained (allowlist) and not."""
    from miniworld_engine.build import kernel_bundle as kb

    prev = kb.read_name_list(repo / PREV_NAMES)
    allowed = kb.read_name_list(repo / ALLOWED_DROPS)
    return {"dropped": kb.dropped_names({"artifacts": {n: {"name": n} for n in names}}, prev, allowed), "allowed": sorted((prev - names) & allowed),
            "previous": len(prev)}


def prune_names(manifest: dict, drop: set[str]) -> tuple[dict, list[str]]:
    """A manifest without the artifacts of ``drop`` (names); also the paths to delete."""
    keep, gone = {}, []
    for key, a in manifest["artifacts"].items():
        if a["name"] in drop:
            gone.append(a["path"])
            gone.extend(str(Path(a["path"]).with_suffix(s)) for s in a.get("sidecars") or [])
        else:
            keep[key] = a
    return {**manifest, "artifacts": keep}, gone


def prune_reviewed(manifests: list[Path], names_file: Path, repo: Path) -> int:
    """``bundle --prune-reviewed FILE``: remove from the bundle the artifacts named in FILE. Every name must be in the reviewed allow-list
    (tests/cpu/builder/data/prebuilt_names_allowed_drops.txt: the previous release's names a maintainer has explained as dropped)."""
    from miniworld_engine.build import kernel_bundle as kb

    want = kb.read_name_list(names_file)
    allowed = kb.read_name_list(repo / ALLOWED_DROPS)
    if want - allowed:
        say(f"[bundle] --prune-reviewed refused: not in the reviewed list {ALLOWED_DROPS}: {sorted(want - allowed)}")
        return 2
    for mf in manifests:
        new, gone = prune_names(json.loads(mf.read_text()), want)
        for rel in gone:
            (mf.parent / rel).unlink(missing_ok=True)
        mf.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        say(f"[bundle] pruned {len(gone)} file(s) from {mf.parent.name}")
    return 0


def _dims(arch: str) -> tuple[list[dict], dict[int, dict]]:
    from miniworld_engine.build import bundle_coverage, registry_calls

    return bundle_coverage.cases(arch), {e.row: e.dims for e in registry_calls.expand(arch)}


def node_digest() -> callable:
    """``label -> the tree's digest`` for a code-graph node label of an artifact's key material: ``pkg:<path>`` is a file of this package
    (``absent`` when it is gone); any other label (an external header) is not the tree's: None."""
    from miniworld_engine.kernels import _codegraph as cg

    pkg = rec.repo_root() / SRC
    memo: dict[str, str | None] = {}

    def digest(label: str) -> str | None:
        if label not in memo:
            if not label.startswith("pkg:"):
                memo[label] = None
            else:
                path = pkg / label[4:]
                memo[label] = cg.parse_file(str(path)).digest if path.is_file() else "absent"
        return memo[label]
    return digest


def bundle_artifacts(arch: str) -> tuple[set[str], list[dict]]:
    """(names, every artifact entry with its key material) of the installed bundle of ``arch`` (all its CUDA builds)."""
    entries = [a for m in prebuilt_manifests(None, arch) for a in json.loads(m.read_text())["artifacts"].values()]
    return {a["name"] for a in entries}, entries


def read_row_record(arch: str, repo: Path) -> dict[str, dict]:
    """``tests/cpu/builder/data/registry_row_artifacts_<arch>.json``: row key -> {text, artifacts} of the rows the last verify ran clean."""
    data = repo / ROW_ARTIFACTS.format(arch=arch)
    if not data.is_file():
        return {}
    return {k: v for k, v in json.loads(data.read_text())["rows"].items() if isinstance(v, dict) and v.get("clean", True)}


def merge_row_record(arch: str, repo: Path, ran: dict[str, dict], inputs: dict[str, str] | None = None,
                     keep: set[str] | None = None) -> None:
    """Merge a verify's ``row_artifacts.json`` rows: a clean row replaces its record -- with ``served`` (the digest of the bundle artifacts it
    ran on: a later rebuild of one of them re-verifies it) and ``inputs`` (the digest of the files its derivation units ran: a later edit of
    one of them re-verifies it) -- a row that was not clean loses it (it is owed again), and with ``keep`` a row the registry no longer declares is dropped."""
    data = repo / ROW_ARTIFACTS.format(arch=arch)
    cur = json.loads(data.read_text())["rows"] if data.is_file() else {}
    sha = _artifact_shas(arch)
    for key, r in ran.items():
        if r.get("clean"):
            cur[key] = {"text": r["text"], "artifacts": r["artifacts"], "served": _served(r["artifacts"], sha),
                        **({"inputs": inputs[key]} if inputs and key in inputs else {})}
        else:
            cur.pop(key, None)
    if keep is not None:
        cur = {k: v for k, v in cur.items() if k in keep}
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps({"arch": arch, "rows": cur}, indent=1, sort_keys=True) + "\n")


def current_row_inputs(arch: str, repo: Path, exp) -> dict[str, str]:
    """``row key -> digest of the files its derivation units ran`` on the card's committed plan (``autotune/plans/<arch>``)."""
    from miniworld_engine.autotune import plan
    from miniworld_engine.build import repo_graph as rg

    plans = rg.derivation_plans()
    if arch not in plans:
        return {}
    unit_files = {label: [f"{SRC}/{f}" for f in files] for label, files in plan.unit_file_sets(json.loads(plans[arch].read_text())).items()}
    return sc.row_inputs({e.key: [c.to_json() for c in e.calls] for e in exp}, {e.key: e.dims for e in exp}, unit_files,
                         lambda files: rec.content_digest(files, repo))


def row_state(arch: str, repo: Path, records: Path) -> sc.Rows:
    from miniworld_engine.build import registry_calls

    exp = [e for e in registry_calls.expand(arch) if e.calls]
    last = rec.latest("verify", arch, results=("pass", "nothing", "fail"), root=records)
    return sc.Rows(keys={e.row: e.key for e in exp}, text={e.key: e.text_digest for e in exp}, recorded=read_row_record(arch, repo),
                   reported_missing=(last or {}).get("row_missing", {}), inputs=current_row_inputs(arch, repo, exp))


BUILD_ENVS = "build_envs.json"


def build_envs(records: Path) -> dict:
    """``<records>/build_envs.json``: the environment that builds each other CUDA build of the bundle, per Python ABI --
    ``{"sm90-cu130": {"cpython-312": {"python": ".../bin/python", "cuda_home": "...", "env": {"LD_LIBRARY_PATH": ""}}}}``. Machine-local (next
    to the records, outside the checkout): the paths are this machine's. ``tool_path``: more binaries the replay finds on PATH after the
    build's own (a pip CUDA has no ``cuobjdump``, which the register-budget patch reads the SASS with; without it the patch declines and the
    variant is never made)."""
    path = records / BUILD_ENVS
    return json.loads(path.read_text()) if path.is_file() else {}


def recorded_recipes(records: Path) -> dict[str, list[dict]]:
    """``name -> every recorded artifact entry with a recipe`` from the coverage runs of earlier ``bundle`` actions (their work directories)."""
    from miniworld_engine.build import kernel_bundle as kb

    out: dict[str, list[dict]] = {}
    for e in kb.read_records(sorted(records.glob("work/bundle-*/records/*.jsonl")), require_file=False).values():
        if e.get("recipe"):
            out.setdefault(e["name"], []).append(e)
    return out


def replay_code_digest() -> str:
    """The code a replay runs (this module and ``kernel_bundle``): a replay that failed is tried again once that code changes."""
    h = hashlib.sha256()
    for name in ("actions.py", "kernel_bundle.py"):
        h.update((Path(__file__).with_name(name)).read_bytes())
    return h.hexdigest()[:16]


def with_bases(entries: list[dict], recipes: dict[str, list[dict]]) -> list[dict]:
    """``entries`` plus the base cubin of every derived one (a register-budget variant is re-derived from its base, which a build that lacks
    only the variant does not lack: the replay skipped it, "its base cubin was not recorded")."""
    by_key = {e["key"]: e for es in recipes.values() for e in es}
    keys = {e["key"] for e in entries}
    bases = [by_key[k] for k in dict.fromkeys(e["recipe"].get("base_key") for e in entries if e["recipe"].get("fn") == "derived")
             if k in by_key and k not in keys]
    return bases + entries


def replay_builds(ctx: Ctx, arch: str) -> dict[str, str]:
    """Bring every other CUDA build and Python ABI of the bundle up to the names one of them holds (``action_scope.owed_replays``): replay the
    recorded recipes of exactly the names a build lacks, in that build's environment (``build_envs``). Returns ``"<build>/<python>" -> what is
    still owed`` (empty: every build carries every name)."""
    from miniworld_engine.kernels import prebuilt

    def manifests() -> dict[str, dict]:
        return {m.parent.name: json.loads(m.read_text()) for m in prebuilt_manifests(None, arch)}

    owed, envs, recipes = sc.owed_replays(manifests()), build_envs(ctx.records), recorded_recipes(ctx.records)
    for (build, py), names in owed.items():
        env = envs.get(build, {}).get(py)
        entries = with_bases([e for n in names for e in recipes.get(n, ())], recipes)
        if env is None or not entries:
            continue
        work = ctx.records / "work" / f"replay-{build}-{py}-{ctx.tree[:12]}"
        work.mkdir(parents=True, exist_ok=True)
        todo = work / "replay-these.jsonl"
        todo.write_text("".join(json.dumps(e) + "\n" for e in entries))
        python, cuda_home = env["python"], env["cuda_home"]
        say(f"[bundle] replay {len(names)} name(s) ({len(entries)} recipes) onto {build} / {py}")
        run_env = {**os.environ, **(env.get("env") or {}), "PYTHONPATH": str(ctx.repo / "src"), "CUDA_HOME": cuda_home,
                   "PATH": f"{Path(python).parent}:{cuda_home}/bin:{env.get('tool_path', '')}:/usr/bin:/bin", "MINIWORLD_ENGINE_ACTION": "bundle"}
        rc = subprocess.run([python, "-m", "miniworld_engine.cli", "dev", "bundle-kernels", "--arch", arch, "--out", str(work),
                             "--prebuilt-root", str(prebuilt.PREBUILT_DIR), "--keep", "--replay", str(todo), "--jobs", "32"],
                            cwd=ctx.repo, env=run_env).returncode
        say(f"[bundle] replay onto {build} / {py}: exit {rc}")
    left = {}
    for (build, py), names in sc.owed_replays(manifests()).items():
        why = ("no build environment in " + str(ctx.records / BUILD_ENVS) if build not in envs or py not in envs[build]
               else "no recorded recipe" if not any(n in recipes for n in names) else "the replay did not produce them")
        left[f"{build}/{py}"] = f"{len(names)} name(s) missing ({why}): " + ", ".join(names[:4])
    return left


def cmd_bundle(args: argparse.Namespace) -> int:
    from miniworld_engine.build import kernel_bundle as kb

    ctx = Ctx(args, "bundle")
    manifests = prebuilt_manifests(None, ctx.arch)
    if args.prune_reviewed:
        return prune_reviewed(manifests, Path(args.prune_reviewed), ctx.repo)
    cases, _ = _dims(ctx.arch)
    rows = row_state(ctx.arch, ctx.repo, ctx.records)
    digest = node_digest()
    names, entries = bundle_artifacts(ctx.arch)
    stale = sc.stale_artifacts(entries, digest)
    say(f"[bundle] the installed {ctx.arch} bundle: {len(names)} names, {len(stale)} stale against the tree"
        + (": " + ", ".join(list(stale)[:8]) if stale else ""))
    scope = sc.bundle_scope(ctx.facts, cases, rows, names, stale, ctx.ledger("bundle"), digest_of=digest, all_reason=ctx.reason, repo=ctx.repo,
                            owed=ctx.owed())
    say(scope.block(ctx.tree))
    for k, w in list(scope.payload.get("why", {}).items())[:20]:
        say(f"  {k}: {w}")
    if scope.empty:
        # no registry row owes a coverage run; the other CUDA builds / Python ABIs may still lack names this one has (a replay, no GPU)
        owed = {f"{b}/{p}": n for (b, p), n in sc.owed_replays(
            {m.parent.name: json.loads(m.read_text()) for m in prebuilt_manifests(None, ctx.arch)}).items()}
        if not owed:
            # the digest of the bundle as it stands: `release` matches it with the wheel's and verify's (a commit that changes no row, such as
            # the run_all manifest, gets a new tree with a "nothing" record, and the release must still find the bundle it was verified on)
            return ctx.nothing("bundle", scope, {"bundle_digest": bundle_digest(prebuilt_manifests(None, ctx.arch))})
        last = rec.latest("bundle", ctx.arch, results=("pass", "nothing", "fail"), root=ctx.records) or {}
        digest = bundle_digest(prebuilt_manifests(None, ctx.arch))
        code = replay_code_digest()
        if (last.get("bundle_digest") == digest and sorted(last.get("replays_left", {})) == sorted(owed) and last.get("replay_code") == code
                and not args.retry_failing):
            # the same builds still lack the same names on the same bundle, and the replay is the same code: it failed this way before; not
            # rerun until a change (or --retry-failing)
            say("[bundle] nothing to do: the other builds still owe what the last replay could not make (" + ", ".join(sorted(owed)) + ")")
            ctx.finish("bundle", scope, "fail", {"replays_left": last["replays_left"], "bundle_digest": digest, "replay_code": code}, owed=[])
            return 1
        left = replay_builds(ctx, ctx.arch)
        for k, v in left.items():
            say(f"[bundle]   still owed on {k}: {v}")
        ctx.finish("bundle", scope, "fail" if left else "pass",
                   {"replays_left": left, "bundle_digest": bundle_digest(prebuilt_manifests(None, ctx.arch)), "replay_code": code}, owed=[])
        return 1 if left else 0
    import shutil

    if shutil.which("ninja") is None:   # every owed key compiles through torch.utils.cpp_extension, which needs ninja on PATH
        say("[bundle] refused before building: ninja is not on PATH (put the environment's bin first), nothing was run")
        return 2
    out = ctx.records / "work" / f"bundle-{ctx.arch}-{ctx.tree[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    cases_file = out / "scope-cases.json"
    cases_file.write_text(json.dumps(scope.payload["cases"]))
    ns = argparse.Namespace(arch=ctx.arch, out=str(out), prebuilt_root="", coverage=[f"scoped:{cases_file}"], workers=args.workers, gpus="",
                            collect_only=False, keep=True, drop_stale=False, replay=[], jobs=32, only="")
    rc = kb.bundle_kernels(ns)
    bad_items: set[str] = set()
    log = out / "logs" / "engine_cases.jsonl"
    ran = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    seen = set()
    for c in ran:
        item = f"kernel:{c['kernel']}" if c["kind"] == "registry_kernel" else f"row:{rows.keys.get(c['row'], '?')}"
        seen.add(item)
        if c.get("status") != "ok":
            bad_items.add(item)
    bad_items |= set(scope.items) - seen                         # an item none of whose cases reported is not done
    # every item that ran is in the ledger, failed or not: the same inputs are not built again (a failure stays failing until a change)
    rec.save_ledger("bundle", ctx.arch, {k: d for k, d in scope.items.items() if k in seen}, ctx.records)
    names, entries = bundle_artifacts(ctx.arch)
    left = sc.stale_artifacts(entries, node_digest())
    still = sorted(set(left) - set(scope.payload.get("orphans", ())))
    diff = name_diff(names, ctx.repo)
    say(f"[bundle] items: {len(scope.items) - len(bad_items)} ok, {len(bad_items)} not ok; artifacts still stale against the tree: {len(still)}")
    for k in sorted(bad_items)[:20]:
        say(f"[bundle]   not ok: {k}")
    for n in still[:20]:
        say(f"[bundle]   still stale: {n} ({', '.join(left[n][:2])})")
    say(f"[bundle] names vs the previous release: {len(diff['dropped'])} dropped without an allow-list line" + (": " + ", ".join(diff["dropped"][:12]) if diff["dropped"] else ""))
    left = replay_builds(ctx, ctx.arch)          # the other CUDA builds / Python ABIs get the names this run added
    for k, v in left.items():
        say(f"[bundle]   still owed on {k}: {v}")
    ok = rc == 0 and not bad_items and not still and not left
    ctx.finish("bundle", scope, "pass" if ok else "fail",
               {"bundle_digest": bundle_digest(prebuilt_manifests(None, ctx.arch)), "names": sorted(names), "dropped": diff["dropped"],
                "not_ok": sorted(bad_items), "still_stale": still, "gaps": scope.gaps, "replays_left": left},
               owed=sorted(set(scope.items) - seen), failing=ctx.still_failing(scope.items, bad_items & seen,
                                                                                 alive=[*(f"row:{k}" for k in rows.text), *(f"kernel:{c['kernel']}" for c in cases if c["kind"] == "registry_kernel")]))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------ verify
def audit_rows(recorded: dict[str, list[str]], expected: set[str], have: set[str]) -> dict:
    """The cheap whole-bundle registry audit (no GPU): every registry row of the arch has a recorded artifact list, and every artifact it names
    is in the bundle."""
    no_record = sorted(expected - set(recorded))
    gone = {row: sorted(set(names) - have) for row, names in recorded.items() if row in expected and set(names) - have}
    return {"rows": len(expected), "no_record": no_record, "missing": gone, "stale_record": sorted(set(recorded) - expected),
            "ok": not no_record and not gone}


#: Modules whose rows are not CUDA holes when no engine kernel serves them (the maintainer's decisions, 2026-11-04): an external library is
#: fine, and ProteinMPNN gets no CUDA development (it stays on Triton on every card). Every other row without an engine kernel is a hole.
NO_ENGINE_KERNEL_EXEMPT = {
    "protein_mpnn": "no CUDA development for ProteinMPNN (Triton on every card)",
    "swa_atom_attention": "FlashAttention-4, an external library",
}


def audit_bundle(arch: str, repo: Path) -> dict:
    """Every registry row of ``arch`` has a clean verify record for what it declares now, and every artifact of it is in each CUDA build of the
    bundle. No GPU."""
    from miniworld_engine.build import registry_calls

    expected_text = {e.key: e.text_digest for e in registry_calls.expand(arch) if e.calls}
    recorded = {k: v["artifacts"] for k, v in read_row_record(arch, repo).items() if expected_text.get(k) == v.get("text")}
    expected = set(expected_text)
    manifests = prebuilt_manifests(None, arch)
    if not manifests:
        return {"rows": len(expected), "no_record": sorted(expected - set(recorded)), "missing": {"<bundle>": ["no prebuilt manifest for " + arch]}, "stale_record": [],
                "no_engine_kernel": [], "no_engine_kernel_exempt": {}, "ok": False}
    rep = audit_rows(recorded, expected, set())
    out = {"rows": len(expected), "no_record": rep["no_record"], "missing": {}, "stale_record": rep["stale_record"]}
    for mf in manifests:
        miss = audit_rows(recorded, expected, manifest_names([mf]))["missing"]
        for row, names in miss.items():
            out["missing"].setdefault(row, []).extend(f"{n} [{mf.parent.name}]" for n in names)
    out["ok"] = not out["no_record"] and not out["missing"]
    # the path classification of the card: a clean row that loaded NO bundle artifact ran on no engine kernel here (cuBLAS, an external
    # library, or the PyTorch statements). Registry rule: a row runs on CUDA on this card; such a row is a hole to look at, listed every time,
    # unless its module is exempt (NO_ENGINE_KERNEL_EXEMPT: listed apart, with the reason, never counted as a hole)
    bare = sorted(k for k, names in recorded.items() if not names)
    out["no_engine_kernel"] = [k for k in bare if k.split("/", 1)[0] not in NO_ENGINE_KERNEL_EXEMPT]
    out["no_engine_kernel_exempt"] = {k: NO_ENGINE_KERNEL_EXEMPT[k.split("/", 1)[0]] for k in bare if k.split("/", 1)[0] in NO_ENGINE_KERNEL_EXEMPT}
    return out


def _artifact_shas(arch: str) -> dict[str, list[str]]:
    sha: dict[str, list[str]] = {}
    for mf in prebuilt_manifests(None, arch):
        for a in json.loads(mf.read_text())["artifacts"].values():
            sha.setdefault(a["name"], []).append(a["sha256"])
    return sha


def _served(names: list[str], sha: dict[str, list[str]]) -> str:
    return hashlib.sha256("|".join(sorted(s for n in names for s in sha.get(n, []))).encode()).hexdigest()


def _fingerprint(arch: str, repo: Path):
    """``row key -> digest of the bundle artifacts the row is served from now`` (compared with the record's ``served``: a rebuilt artifact
    re-verifies its rows)."""
    recorded = read_row_record(arch, repo)
    sha = _artifact_shas(arch)

    def fp(key: str) -> str:
        return _served((recorded.get(key) or {}).get("artifacts", []), sha)
    return fp


def cmd_verify(args: argparse.Namespace) -> int:
    from miniworld_engine.build import registry_verify

    ctx = Ctx(args, "verify")
    cases, _ = _dims(ctx.arch)
    rows = row_state(ctx.arch, ctx.repo, ctx.records)
    scope = sc.verify_scope(ctx.facts, cases, rows, ctx.ledger("verify"), fingerprint=_fingerprint(ctx.arch, ctx.repo),
                            all_reason=ctx.reason, repo=ctx.repo, owed=ctx.owed(),
                            bundle_fp=bundle_digest(prebuilt_manifests(None, ctx.arch)))
    say(scope.block(ctx.tree))
    for k, w in list(scope.payload.get("why", {}).items())[:20]:
        say(f"  {k}: {w}")
    manifests = prebuilt_manifests(None, ctx.arch)
    extra = {"bundle_digest": bundle_digest(manifests)}
    rc = 0
    if not scope.empty:
        out = ctx.records / "work" / f"verify-{ctx.arch}-{ctx.tree[:12]}"
        ns = argparse.Namespace(wheels="", tree=str(ctx.repo / "src"), python=sys.executable, arch=ctx.arch, out=str(out), gpus=",".join(_gpus()) or "",
                                row=[str(r) for r in scope.payload["rows"]], kernel=scope.payload["kernels"], no_kernels=False,
                                env=[f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"], timeout=7200.0)
        if ctx.reason:
            ns.row, ns.kernel = [], []          # --all: the whole registry
        if not ns.gpus:
            say("[verify] no GPU in this allocation: the scoped rows cannot run here")
            rc = 2
        else:
            rc = registry_verify.verify(ns)
            summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {"units": []}
            ran = json.loads((out / "row_artifacts.json").read_text())["rows"] if (out / "row_artifacts.json").is_file() else {}
            merge_row_record(ctx.arch, ctx.repo, ran, rows.inputs, keep=set(rows.text))   # the row record is the state
            clean, attempted = [], []
            for u in summary["units"]:
                s = u.get("summary")
                item = f"row:{u['key']}" if u["kind"] == "row" else f"kernel:{u['kernel']}"
                if s and item in scope.items:
                    attempted.append(item)
                    if s["ok"] == s["calls"]:
                        clean.append(item)
            # every item that ran is in the ledger, clean or not: the same inputs are not verified twice (a failure stays failing until a
            # change reaches it: its inputs, or a rebuilt bundle for a row without a record)
            rec.save_ledger("verify", ctx.arch, {k: scope.items[k] for k in attempted}, ctx.records)
            extra["clean"], extra["attempted"] = sorted(clean), sorted(attempted)
            extra["row_missing"] = {k: r["missing"] for k, r in ran.items() if r.get("missing")}
    else:
        say("[verify] nothing to run on a GPU (the graph names no row or kernel that changed since the last verified inputs)")
    audit = audit_bundle(ctx.arch, ctx.repo)
    say(f"[verify] registry-row audit of the whole bundle (no GPU): {audit['rows']} rows, {len(audit['no_record'])} without a clean record of what "
        f"they declare now, {len(audit['missing'])} with artifacts missing from the bundle")
    for row in audit["no_record"][:10]:
        say(f"[verify]   no record: {row}")
    say(f"[verify] rows served by no engine kernel on {ctx.arch} (cuBLAS or PyTorch; a CUDA hole to look at): "
        f"{len(audit['no_engine_kernel'])}" + (": " + ", ".join(audit["no_engine_kernel"][:8]) if audit["no_engine_kernel"] else ""))
    for row, why in audit.get("no_engine_kernel_exempt", {}).items():
        say(f"[verify]   not a hole: {row} ({why})")
    for row, names in list(audit["missing"].items())[:10]:
        say(f"[verify]   {row}: missing {names[:4]}")
    extra["audit"] = {k: v for k, v in audit.items() if k != "missing"} | {"missing": len(audit["missing"])}
    ok = rc == 0 and audit["ok"]
    if not audit["ok"]:
        say("[verify] FAILED: the whole-bundle audit reports rows without a clean record or with missing artifacts "
            "(run `miniworld-engine bundle`, then verify again)")
    attempted = extra.get("attempted", [])
    ctx.finish("verify", scope, ("nothing" if scope.empty else "pass") if ok else "fail", extra,
               owed=sorted(set(scope.items) - set(attempted)), failing=ctx.still_failing(attempted, set(attempted) - set(extra.get("clean", [])),
                                        alive=[*(f"row:{k}" for k in rows.text), *(f"kernel:{c['kernel']}" for c in cases if c["kind"] == "registry_kernel")]))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------ derive
def cmd_derive(args: argparse.Namespace) -> int:
    from miniworld_engine.autotune import plan
    from miniworld_engine.build import repo_graph as rg

    ctx = Ctx(args, "derive")
    scope = sc.derive_scope(ctx.facts, ctx.reason)
    say(scope.block(ctx.tree))
    if scope.empty:
        return ctx.nothing("derive", scope)
    plans = rg.derivation_plans()
    target = plans[ctx.arch].with_name("registry_kernel.csv")
    cmd = [sys.executable, "-m", "miniworld_engine.cli", "dev", "derive", "--arch", ctx.arch, "--out", str(target)]
    if ctx.reason:
        cmd.append("--no-reuse")
    rc = subprocess.run(cmd, cwd=ctx.repo).returncode
    left = plan_stale(ctx.arch, plans[ctx.arch])
    say(f"[derive] exit {rc}; stale units after: {len(left)}")
    owed_page: list[str] = []
    if rc == 0 and not left:   # the sweep page is generated from the derived plans: it is part of what a derivation owes
        page = subprocess.run([sys.executable, "-m", "miniworld_engine.utils.viz.sweep_page"], cwd=ctx.repo).returncode
        # the page reads the plan of every card; one this machine cannot derive (stale after a registry edit) is owed there, not a failure here
        owed_page = sorted(c for c, p in plans.items() if c != ctx.arch and plan.status(p)["stale"])
        if page and owed_page:
            say(f"[derive] sweep page not regenerated: it needs the plan(s) of {', '.join(owed_page)}, stale and not derivable here "
                f"(UNVERIFIABLE-HERE, owed on {', '.join(owed_page)}: `miniworld-engine derive` there, then the page)")
        else:
            say(f"[derive] sweep page regenerated: exit {page}")
            rc = rc or page
    ctx.finish("derive", scope, "pass" if rc == 0 and not left else "fail", {"stale_after": len(left), "owed_on": owed_page})
    return 0 if rc == 0 and not left else 1


# ------------------------------------------------------------------------------------------------------------------ bench
def cmd_bench(args: argparse.Namespace) -> int:
    from miniworld_engine import cli

    ctx = Ctx(args, "bench")
    scope = sc.bench_scope(ctx.facts, ctx.owed(), all_reason=ctx.reason)
    unknown = sorted(op for op in scope.items if op not in cli.MODULE_TARGETS)
    for op in unknown:
        scope.items.pop(op)
        scope.gaps.append(f"benchmarks/modules/{op} is not a module bench target (not timed)")
    say(scope.block(ctx.tree))
    if scope.empty:
        return ctx.nothing("bench", scope)
    done, failed = {}, []
    for op in scope.names():
        rc = 0
        for mode in ("inference", "training"):
            ns = argparse.Namespace(target=op, impl="all", shards="~/.cache/miniworld-build", gpus="all", compile_jobs=0, unit_timeout_seconds=7200.0,
                                    resume=False, no_build=True, sweep_axis="seq_len", cudagraph="auto", compile="true", compile_wrap="", mode=mode,
                                    config_type=None)
            rc |= cli.cmd_bench_module(ns)
        if rc:
            failed.append(op)
        else:
            done[op] = scope.items[op]
    ctx.finish("bench", scope, "fail" if failed else "pass", {"failed": failed, "measured": sorted(done)}, owed=[],
               failing=ctx.still_failing(scope.items, failed, alive=ctx.facts.universe.get("bench_ops", [])))
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------------------------ wheel / release
def _sha(path: Path) -> str:
    return rec.file_digest(path)


def cmd_wheel(args: argparse.Namespace) -> int:
    from miniworld_engine.build import kernel_bundle as kb

    ctx = Ctx(args, graph=False)
    try:
        bundle = rec.require("bundle", ctx.arch, ctx.tree, ctx.version, ctx.records)
        verify = rec.require("verify", ctx.arch, ctx.tree, ctx.version, ctx.records)
    except rec.Refused as exc:
        say(f"[wheel] refused: {exc}")
        return 2
    manifests = prebuilt_manifests(None, ctx.arch)
    digest = bundle_digest(manifests)
    if verify.get("bundle_digest") != digest:
        say(f"[wheel] refused: the bundle changed since it was verified (verified {str(verify.get('bundle_digest'))[:12]}, now {digest[:12]}). Run: miniworld-engine verify")
        return 2
    diff = name_diff(manifest_names(manifests), ctx.repo)
    if diff["dropped"]:
        say(f"[wheel] refused: the bundle lacks {len(diff['dropped'])} name(s) the previous release carried, with no allow-list line: {diff['dropped'][:12]}. "
            "Run: miniworld-engine bundle (it keeps names), or review them into tests/cpu/builder/data/prebuilt_names_allowed_drops.txt")
        return 2
    out = ctx.records / "dist" / f"{ctx.version}-{ctx.tree[:12]}"
    before = {p.name for p in out.glob("*")} if out.is_dir() else set()
    rc = kb.build_wheel(argparse.Namespace(prebuilt_root="", out=str(out), strict=args.strict, split=args.split))
    files = sorted(p for p in out.glob("*") if p.suffix == ".whl" or p.name.endswith(".tar.gz"))
    extra = {"files": {p.name: {"path": str(p), "sha256": _sha(p)} for p in files}, "bundle_digest": digest, "names": sorted(manifest_names(manifests)),
             "python": f"{sys.version_info.major}.{sys.version_info.minor}", "bundle_record": bundle["_path"], "verify_record": verify["_path"],
             "new": sorted({p.name for p in files} - before)}
    ctx.finish("wheel", None, "pass" if rc == 0 and files else "fail", extra)
    return 0 if rc == 0 and files else 1


def release_evidence(repo: Path, version: str) -> list[str]:
    """Cards whose committed ``run_all`` manifest records this version from a clean tree (the rule ``test_the_released_version_has_run_on_a_card``
    asserts)."""
    import csv

    cards = []
    for f in sorted((repo / MANIFESTS).glob("*.csv")):
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["kernel"] == PROVENANCE and r["backend"] == version and r["file"] == "clean":
                    cards.append(f.stem)
                break
    return cards


def cmd_release(args: argparse.Namespace) -> int:
    ctx = Ctx(args, graph=False)
    try:
        wheels = [r for r in rec.read_records("wheel", ctx.arch, ctx.records) if r["tree"] == ctx.tree and r["version"] == ctx.version and r["result"] == "pass"]
        if not wheels:
            raise rec.Refused(f"no passing `wheel` record for tree {ctx.tree[:12]} version {ctx.version}: Run: miniworld-engine wheel")
        bundle = rec.require("bundle", ctx.arch, ctx.tree, ctx.version, ctx.records)
        verify = rec.require("verify", ctx.arch, ctx.tree, ctx.version, ctx.records)
    except rec.Refused as exc:
        say(f"[release] refused: {exc}")
        return 2
    files: dict[str, dict] = {}
    for w in wheels:
        if w.get("bundle_digest") != verify.get("bundle_digest") or w.get("bundle_digest") != bundle.get("bundle_digest"):
            say(f"[release] refused: wheel record {w['_path']} was not built from the verified bundle. Run: miniworld-engine wheel")
            return 2
        for name, f in w["files"].items():
            files.setdefault(name, f)
    bad = [n for n, f in files.items() if not Path(f["path"]).is_file() or _sha(Path(f["path"])) != f["sha256"]]
    if bad:
        say(f"[release] refused: built file(s) changed or are gone since the wheel record: {bad}. Run: miniworld-engine wheel")
        return 2
    diff = name_diff(set(wheels[0]["names"]), ctx.repo)
    if diff["dropped"]:
        say(f"[release] refused: names dropped since the previous release without an allow-list line: {diff['dropped'][:12]}")
        return 2
    cards = release_evidence(ctx.repo, ctx.version)
    if not cards:
        say(f"[release] refused: no committed run_all manifest records version {ctx.version} from a clean tree "
            "(tests/cpu/registry/test_a_release_has_been_run_on_a_card.py). Run: python -m miniworld_engine.autotune.run_all on a card, commit the manifest")
        return 2
    say(f"[release] version {ctx.version}, tree {ctx.tree[:12]}: {len(files)} file(s) from the verified bundle; "
        f"{len(diff['allowed'])} explained drop(s) vs the previous release's {diff['previous']} names; run_all on {', '.join(cards)}")
    for n, f in sorted(files.items()):
        say(f"  {n}  {f['sha256'][:12]}")
    if not args.confirm:
        say("[release] every check passed. Nothing uploaded: the maintainer uploads, with --confirm (an agent never does).")
        return 0
    rc = upload([f["path"] for f in files.values()])
    ctx.finish("release", None, "pass" if rc == 0 else "fail", {"uploaded": sorted(files)})
    return 0 if rc == 0 else 1


def upload(paths: list[str]) -> int:
    """twine upload (``release --confirm`` only)."""
    return subprocess.run([sys.executable, "-m", "twine", "upload", *paths]).returncode


# ------------------------------------------------------------------------------------------------------------------ status
def cmd_status(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    cases, _ = _dims(ctx.arch)
    rows = row_state(ctx.arch, ctx.repo, ctx.records)
    digest = node_digest()
    names, entries = bundle_artifacts(ctx.arch)
    stale = sc.stale_artifacts(entries, digest)
    base = {a: ctx.base_of(a) for a in ("test", "bundle", "verify", "bench")}
    f = {a: ctx.facts_for(b) for a, (b, _) in base.items()}
    scopes = [sc.test_scope(f["test"], ctx.owed("test")),
              sc.bundle_scope(f["bundle"], cases, rows, names, stale, ctx.ledger("bundle"), digest_of=digest, repo=ctx.repo, owed=ctx.owed("bundle")),
              sc.verify_scope(f["verify"], cases, rows, ctx.ledger("verify"), fingerprint=_fingerprint(ctx.arch, ctx.repo), repo=ctx.repo,
                              owed=ctx.owed("verify"), bundle_fp=bundle_digest(prebuilt_manifests(None, ctx.arch))),
              sc.derive_scope(ctx.facts), sc.bench_scope(f["bench"], ctx.owed("bench"))]
    say(f"status: version {ctx.version}, tree {ctx.tree[:12]}, accessible card {ctx.arch}")
    for a, (b, why) in base.items():
        say(f"  {a}: since {why} ({b[:12]}), {len(f[a].changed)} changed file(s), {len(ctx.owed(a))} owed from that run, "
            f"{len(ctx.failing(a))} still failing")
    say(f"bundle ({ctx.arch}): {len(names)} names in {len(prebuilt_manifests(None, ctx.arch))} CUDA build(s), {len(stale)} stale against the tree; "
        f"{len(rows.recorded)} of {len(rows.text)} registry rows have a clean verify record")
    for s in scopes:
        say(s.block(ctx.tree, limit=8))
        for k, w in list(s.payload.get("why", {}).items())[:8]:
            say(f"    why {k}: {w}")
    for action in ("bundle", "verify", "wheel"):
        r = rec.latest(action, ctx.arch, tree=ctx.tree, version=ctx.version, root=ctx.records)
        say(f"[{action}] record for this tree: " + (f"{r['result']} ({r['time']})" if r else f"none (run: miniworld-engine {action})"))
    audit = audit_bundle(ctx.arch, ctx.repo)
    say(f"[verify] registry-row audit: {audit['rows']} rows, {len(audit['no_record'])} unrecorded, {len(audit['missing'])} with missing artifacts, "
        f"{len(audit['no_engine_kernel'])} served by no engine kernel" + (": " + ", ".join(audit["no_engine_kernel"][:6]) if audit["no_engine_kernel"] else "")
        + f" ({len(audit.get('no_engine_kernel_exempt', {}))} more exempt: an external library or ProteinMPNN)")
    say("[release] needs: wheel record for this tree, names diff reviewed, run_all evidence for " + ctx.version + ": "
        + (", ".join(rec_cards) if (rec_cards := release_evidence(ctx.repo, ctx.version)) else "NONE yet"))
    return 0
