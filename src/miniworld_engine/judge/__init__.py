"""The judge: from the measurers' records and the human-managed versions, which requirement cases have a current result (a token) and
which are owed a measurement. It never measures; the measurers (miniworld_engine.measure) never decide. No hashes: every comparison
is of versions and plain values a person sets or can read.

What a result depends on, as recorded in it:

    case       the requirement case's values: module, model, stream, constructor kwargs, length, depth, samples, dtype, mode, backend,
               kind (accuracy | speed) and timing
    versions   the module class's ``version``, the ``version`` of every kernel requirements_kernel.csv lists for the class in that
               mode (a person bumps them when a change alters a result; a refactor, a comment or a rebuild bumps nothing) and the
               ``version`` of those kernel families' launch-config tables (the pretuner's merge increments it when a table changes)
    measurer   the measurer's version (``ACCURACY_VERSION`` / ``SPEED_VERSION``), the config file's (``CONFIG_VERSION``) and the
               prebuilder's (``PREBUILD_VERSION``: how the kernels were built)
    settings   seed, fixed standard and speed settings of the run, as values
    env        GPU capability, torch, CUDA, Triton versions

Verdict per case:

    carried    nothing above differs from now (or a measurer / env difference was answered "keep")
    owed       no token, a module or kernel version changed, or the settings changed: the driver measures it
    pending    the measurer / config / prebuild version or the environment changed and nobody answered yet: a person decides once, for every
               case with that change, whether the old results still stand (``sign ... keep``) or must be measured again
               (``sign ... remeasure``). Pending cases are neither carried nor re-run; they block a release only.

    python -m miniworld_engine.judge issue RECORDS...         records -> tokens (only records whose versions and settings are current)
    python -m miniworld_engine.judge owed [--plan DIR] [--redo failed ...]
    python -m miniworld_engine.judge sign PART OLD NEW keep|remeasure --reason TEXT
    python -m miniworld_engine.judge status
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import re
from pathlib import Path

import yaml

from miniworld_engine.measure import config as cfgmod

HERE = Path(__file__).resolve().parent
TOKENS = HERE / "tokens"                  # per card, one file per module class: tokens/<gpu>/<module>.jsonl, sorted by case
#: The cards results are kept for (``kernels._launch_config.gpu_key``) and their compute capability.
GPUS = {"h100_sm90": [9, 0], "a100_sm80": [8, 0]}
ANSWERS = HERE / "answers.jsonl"          # a person's answers: {"part", "old", "new", "verdict": keep | remeasure, "reason", "date"}
PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = {"accuracy": PACKAGE / "measure" / "configs" / "accuracy.yaml", "speed": PACKAGE / "measure" / "configs" / "speed.yaml"}
ENV_KEYS = ("capability", "torch", "cuda", "triton")      # the environment that can change a result
RESULT_KEYS = ("status", "reason", "light", "err_backend", "err_pytorch", "tol", "tensor", "fixed", "timed_as", "capture_error",
               "median_ms", "p10_ms", "p90_ms", "n", "speedup_vs_pytorch")


# ------------------------------------------------------------------------------------------------------------------------ the case
def case_key(kind: str, rec: dict) -> str:
    """The case, readable: what is measured, independent of how and with which versions."""
    kw = ";".join(f"{k}={v}" for k, v in sorted(rec["kwargs"].items()))
    parts = [kind, rec["module"], rec.get("model") or "-", rec.get("stream") or "-", kw or "-", f"L{rec['length']}",
             f"D{rec['depth']}", f"S{rec['samples']}", rec["dtype"], rec["mode"], rec["backend"]]
    if rec.get("timing"):
        parts.append(rec["timing"])
    return " | ".join(parts)


# --------------------------------------------------------------------------------------------------------------------------- now
class Now:
    """The current versions, settings and environment."""

    def __init__(self, gpu: str = cfgmod.DEFAULT_GPU) -> None:
        from miniworld_engine.measure import accuracy, speed

        if gpu not in GPUS:
            raise SystemExit(f"unknown card {gpu!r}; known: {', '.join(GPUS)}")
        self.gpu = gpu

        self.measurer = {"accuracy": accuracy.ACCURACY_VERSION, "speed": speed.SPEED_VERSION}
        self.config_version = cfgmod.CONFIG_VERSION
        from miniworld_engine.kernels._prebuild import PREBUILD_VERSION

        self.prebuild_version = PREBUILD_VERSION
        self.settings = {k: _plain(cfgmod.settings(cfgmod.load(p))) for k, p in CONFIGS.items()}
        self._versions: dict = {}
        self._env = None

    def versions(self, module: str, mode: str) -> dict:
        if (module, mode) not in self._versions:
            self._versions[(module, mode)] = cfgmod.versions(module, mode, self.gpu)
        return self._versions[(module, mode)]

    @property
    def env(self) -> dict:
        if self._env is None:
            import torch
            try:
                import triton
                tv = triton.__version__
            except Exception:  # absent
                tv = None
            # the target card: the judge runs on a CPU; results are kept per card (--gpu)
            self._env = {"capability": GPUS[self.gpu], "torch": torch.__version__, "cuda": torch.version.cuda, "triton": tv}
        return self._env


def _plain(obj):
    """Values as a record stores them (JSON round trip: tuples become lists)."""
    return json.loads(json.dumps(obj, default=str))


def differences(kind: str, tok: dict, now: Now) -> list[tuple[str, str, object, object]]:
    """(class, part, old, new) for every difference between a token and now. Class ``owed``: measure again; ``ask``: a person decides."""
    case, out = tok["case"], []
    old_v, cur_v = tok["versions"], now.versions(case["module"], case["mode"])
    if str(old_v["module"]) != str(cur_v["module"]):
        out.append(("owed", f"module:{case['module']}", old_v["module"], cur_v["module"]))
    b = case["backend"]
    o, n = (old_v.get("backends") or {}).get(b, "1"), (cur_v.get("backends") or {}).get(b, "1")   # tokens made before: "1"
    if str(o) != str(n):
        out.append(("owed", f"adapter:{b}", o, n))
    if case["dtype"] == "float32":                    # the TF32 rule only touches float32 matmuls
        o, n = (old_v.get("tf32") or {}).get(b, "1"), (cur_v.get("tf32") or {}).get(b, "1")
        if str(o) != str(n):
            out.append(("owed", f"tf32:{b}", o, n))
    if case["backend"] == "pytorch" or case["backend"] in cfgmod.EXTERNAL:
        old_v = {**old_v, "kernels": {}, "tables": {}}       # these backends run none of the engine's kernels
        cur_v = {**cur_v, "kernels": {}, "tables": {}}
    # only a kernel listed both then and now: a kernel entering or leaving the list is a correction of requirements_kernel.csv (which
    # kernels a class launches), not a change of any result; a class that starts launching another kernel bumps its own version
    for name in sorted(set(old_v["kernels"]) & set(cur_v["kernels"])):
        o, n = old_v["kernels"][name], cur_v["kernels"][name]
        if o != n:
            out.append(("owed", f"kernel:{name}", o, n))
    old_t = old_v.get("tables") or dict.fromkeys(cur_v.get("tables", {}), "1")   # tokens made before table versions: all "1"
    for fam in sorted(set(old_t) & set(cur_v.get("tables", {}))):   # the same rule for the families
        o, n = old_t[fam], cur_v["tables"][fam]
        if o != n:
            out.append(("owed", f"table:{fam}", o, n))
    if tok["settings"] != now.settings[kind]:
        out.append(("owed", "settings", tok["settings"], now.settings[kind]))
    if tok["measurer"] != now.measurer[kind]:
        out.append(("ask", f"measurer:{kind}", tok["measurer"], now.measurer[kind]))
    if tok["config_version"] != now.config_version:
        out.append(("ask", "config_version", tok["config_version"], now.config_version))
    if tok.get("prebuild_version", "1") != now.prebuild_version:
        out.append(("ask", "prebuild_version", tok.get("prebuild_version", "1"), now.prebuild_version))
    out.extend(("ask", f"env:{k}", tok["env"].get(k), now.env[k]) for k in ENV_KEYS if tok["env"].get(k) != now.env[k])
    return out


def answers() -> dict[tuple, str]:
    out = {}
    if ANSWERS.exists():
        for line in ANSWERS.read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                out[(a["part"], str(a["old"]), str(a["new"]))] = a["verdict"]     # the latest answer wins
    return out


def verdict(diffs, given) -> tuple[str, list]:
    owed, pending = [], []
    for cls, part, old, new in diffs:
        answer = given.get((part, str(old), str(new))) if cls == "ask" else None
        if answer == "keep":
            continue
        (owed if cls == "owed" or answer == "remeasure" else pending).append((part, old, new))
    return ("owed", owed) if owed else ("pending", pending) if pending else ("carried", [])


def card_of(env: dict) -> str:
    """The card key of a record's environment, the rule of ``kernels._launch_config.gpu_key`` (``h100_sm90``, ``a100_sm80``)."""
    name = str(env.get("gpu") or "").lower().replace("nvidia", "")
    model = next((t for t in re.split("[^a-z0-9]+", name) if any(c.isdigit() for c in t)), "gpu")
    cap = env.get("capability") or [0, 0]
    return f"{model}_sm{cap[0]}{cap[1]}"


# ------------------------------------------------------------------------------------------------------------------------ tokens
def load_tokens(gpu: str) -> dict[str, dict]:
    out = {}
    for f in sorted((TOKENS / gpu).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                out[t["case_key"]] = t
    return out


def save_tokens(tokens: dict[str, dict], gpu: str) -> None:
    d = TOKENS / gpu
    d.mkdir(parents=True, exist_ok=True)
    by_module = collections.defaultdict(list)
    for t in tokens.values():
        by_module[t["case"]["module"]].append(t)
    for module, ts in by_module.items():
        tmp = d / f".{module}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in sorted(ts, key=lambda t: t["case_key"])))
        tmp.replace(d / f"{module}.jsonl")


def issue(paths: list[str], now: Now) -> int:
    """Records -> tokens. A record becomes a token only if its module / kernel versions, measurer and config versions and settings
    equal the current ones (it was made by what is current); the token replaces any older token of the same case."""
    tokens, counts, stale = load_tokens(now.gpu), collections.Counter(), collections.Counter()
    files = sorted({f for p in paths for f in (glob.glob(os.path.join(p, "*.jsonl")) if os.path.isdir(p) else [p])
                    if not f.endswith(".plans.jsonl")})
    for path in files:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                meta = rec.get("meta") or {}
                kind = meta.get("kind")
                if kind not in CONFIGS or "versions" not in rec or "settings" not in meta:
                    counts["record without versions (older format)"] += 1
                    continue
                if (rec.get("gpu") or card_of(meta.get("env") or {})) != now.gpu:
                    counts[f"record of another card (not {now.gpu})"] += 1
                    continue
                key = case_key(kind, rec)
                tok = {"case_key": key, "kind": kind,
                       "case": {k: rec.get(k) for k in ("module", "model", "stream", "kwargs", "length", "depth", "samples", "dtype", "mode",
                                                       "backend", "timing", "row")},
                       "versions": rec["versions"], "measurer": meta["measurer_version"], "config_version": meta["config_version"],
                       "prebuild_version": meta.get("prebuild_version", "1"),
                       "settings": meta["settings"], "env": {k: (meta.get("env") or {}).get(k) for k in ENV_KEYS},
                       "result": {k: rec[k] for k in RESULT_KEYS if k in rec},
                       "evidence": {"host": (meta.get("env") or {}).get("host"), "gpu": (meta.get("env") or {}).get("gpu"),
                                    "record": os.path.relpath(path)},
                       "issued": datetime.datetime.now().isoformat(timespec="seconds")}  # noqa: DTZ005 -- local wall-clock time, as every token so far
                if kind == "accuracy" and rec["backend"] not in ("pytorch", *cfgmod.EXTERNAL):
                    tok["result"]["pass"] = rec.get("light") == "GREEN"
                diffs = [d for d in differences(kind, tok, now) if not d[1].startswith("env:")]
                if diffs:
                    counts["not current (made by other versions or settings)"] += 1
                    for _, part, _, _ in diffs:
                        stale[part.split(":")[0]] += 1
                    continue
                tokens[key] = tok
                counts["issued"] += 1
    save_tokens(tokens, now.gpu)
    for k, v in sorted(counts.items()):
        print(f"[judge] issue: {v:6d} {k}")
    for k, v in sorted(stale.items()):
        print(f"[judge]   not current because of {k}: {v}")
    return 0


# --------------------------------------------------------------------------------------------------------------------------- owed
def required(kind: str):
    """Every (case, backend[, timing]) the canonical config of ``kind`` asks for, record-shaped."""
    cfg = cfgmod.load(CONFIGS[kind])
    timings = list(cfgmod.speed_settings(cfg).get("timing", ["graph"])) if kind == "speed" else [None]   # the rule of measure/speed.py
    for case in cfgmod.cases(cfg):
        head = {"row": case.row, "module": case.module, "model": case.model, "stream": case.stream, "kwargs": case.kwargs,
                "length": case.length, "depth": case.depth, "samples": case.samples, "dtype": case.dtype, "mode": case.mode}
        for backend in cfgmod.backends_of(cfg, case):
            for timing in timings:
                yield {**head, "backend": backend, "timing": timing}


def owed(now: Now, plan_dir: str | None, redo: tuple[str, ...] = (), again: tuple[str, ...] = (), again_modules: tuple[str, ...] = (),
         again_kinds: tuple[str, ...] = (), n_groups: int = 4) -> int:
    """``redo``: result statuses (e.g. ``failed``) measured again although current (after a measurer fix that changes no other result,
    so its version stays). ``again``: backends measured again although current (a person asked for it). ``again_modules`` (optionally of ``again_kinds`` only):
    module classes measured again although current (a measurer fix that changes only their results)."""
    tokens, given = load_tokens(now.gpu), answers()
    tally, reasons, questions = collections.Counter(), collections.Counter(), collections.Counter()
    todo = collections.defaultdict(lambda: {"rows": set(), "lengths": set(), "backends": set(), "dtypes": set(), "modes": set(), "n": 0})
    seen = set()
    for kind in CONFIGS:
        for rec in required(kind):
            key = case_key(kind, rec)
            seen.add(key)
            tok = tokens.get(key)
            if tok is None:
                state, why = "owed", [("no token", None, None)]
            elif rec["module"] in again_modules and (not again_kinds or kind in again_kinds):
                state, why = "owed", [(f"again {rec['module']}", None, None)]
            elif rec["backend"] in again:
                state, why = "owed", [(f"again {rec['backend']}", None, None)]
            elif tok["result"].get("status") in redo:
                state, why = "owed", [(f"redo {tok['result'].get('status')}", None, None)]
            else:
                state, why = verdict(differences(kind, tok, now), given)
            tally[(kind, state)] += 1
            if state == "owed":
                for part, old, new in why:
                    reasons[(kind, part if old is None else f"{part} {old}->{new}")] += 1
                t = todo[(kind, rec["module"])]
                t["rows"].add(rec["row"]); t["lengths"].add(rec["length"]); t["backends"].add(rec["backend"])
                t["dtypes"].add(rec["dtype"]); t["modes"].add(rec["mode"]); t["n"] += 1
            elif state == "pending":
                for part, old, new in why:
                    questions[(part, str(old), str(new))] += 1
    for kind in CONFIGS:
        print(f"[judge] {kind:8s} carried {tally[(kind, 'carried')]:6d}  owed {tally[(kind, 'owed')]:6d}  pending {tally[(kind, 'pending')]:6d}")
    for (kind, why), n in sorted(reasons.items()):
        print(f"[judge]   owed {kind:8s} because of {why}: {n}")
    for (part, old, new), n in sorted(questions.items()):
        print(f"[judge]   QUESTION {part}: {old} -> {new} ({n} cases). Do the old results still stand?  "
              f"judge sign {part} {old} {new} keep|remeasure --reason ...")
    unused = sum(1 for k in tokens if k not in seen)
    if unused:
        print(f"[judge]   {unused} tokens belong to cases no longer required (kept, unused)")
    if plan_dir:
        write_plan(plan_dir, todo, n_groups)
    return 0


def write_plan(plan_dir: str, todo: dict, n_groups: int = 4) -> None:
    """The owed run: one config per (kind, module) selecting the owed rows / lengths / backends / dtypes / modes (a superset of the
    owed cases when they do not form a product: the extra cases are measured again, which is harmless), in ``n_groups`` balanced groups
    (one measurement job each, run in parallel)."""
    d = Path(plan_dir)
    (d / "cfg").mkdir(parents=True, exist_ok=True)
    jobs = []
    for (kind, module), t in sorted(todo.items()):
        base = yaml.safe_load(CONFIGS[kind].read_text())
        base.update(modules=[module], rows=sorted(t["rows"]), lengths=sorted(t["lengths"]), backends=sorted(t["backends"] | {"pytorch"}),
                    dtypes=sorted(t["dtypes"]), modes=sorted(t["modes"]))
        path = d / "cfg" / f"{kind}.{module}.yaml"
        path.write_text(yaml.safe_dump(base, sort_keys=False))
        jobs.append((t["n"], kind, module, str(path)))
    groups, load = [[] for _ in range(n_groups)], [0] * n_groups
    for n, kind, module, path in sorted(jobs, reverse=True):
        i = load.index(min(load))
        groups[i].append(f"{kind} {module} {path}")
        load[i] += n
    for i, g in enumerate(groups, 1):
        (d / f"G{i}").write_text("".join(line + "\n" for line in g))
    print(f"[judge] plan: {len(jobs)} (kind, module) runs in {plan_dir}, groups of {load} owed cases")


# ------------------------------------------------------------------------------------------------------------------- sign, status
def sign(part: str, old: str, new: str, answer: str, reason: str) -> int:
    entry = {"part": part, "old": old, "new": new, "verdict": answer, "reason": reason,
             "date": datetime.datetime.now().isoformat(timespec="seconds")}  # noqa: DTZ005 -- local wall-clock time, as every answer so far
    with open(ANSWERS, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"[judge] answered {part}: {old} -> {new} = {answer}")
    return 0


def status(gpu: str) -> int:
    tokens = load_tokens(gpu)
    c = collections.Counter((t["kind"], t["case"]["backend"], t["result"].get("status")) for t in tokens.values())
    for (kind, backend, st), n in sorted(c.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        print(f"[judge] {kind:8s} {backend:10s} {st!s:12s} {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m miniworld_engine.judge", description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default=cfgmod.DEFAULT_GPU, choices=tuple(GPUS), help="the card whose tokens to use (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("issue", help="turn records made with the current versions into tokens")
    p.add_argument("records", nargs="+", help="record files or directories of *.jsonl")
    p = sub.add_parser("owed", help="verdict of every required case")
    p.add_argument("--plan", default=None, help="write the owed run (configs + groups) into this directory")
    p.add_argument("--groups", type=int, default=4, help="parallel measurement groups (GPUs) of the plan (default %(default)s)")
    p.add_argument("--redo", nargs="*", default=(), help="result statuses to measure again (e.g. failed)")
    p.add_argument("--again", nargs="*", default=(), help="backends to measure again although current (e.g. pytorch cueq anthropic)")
    p.add_argument("--again-module", nargs="*", default=(), help="module classes to measure again although current")
    p.add_argument("--again-kind", nargs="*", default=(), choices=tuple(CONFIGS), help="restrict --again-module to these kinds")
    p = sub.add_parser("sign", help="answer a question for every case with that (part, old, new)")
    p.add_argument("part")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("answer", choices=("keep", "remeasure"))
    p.add_argument("--reason", required=True)
    sub.add_parser("status", help="counts of the tokens")
    a = ap.parse_args(argv)
    if a.cmd == "issue":
        return issue(a.records, Now(a.gpu))
    if a.cmd == "owed":
        return owed(Now(a.gpu), a.plan, tuple(a.redo), tuple(a.again), tuple(a.again_module), tuple(a.again_kind), a.groups)
    if a.cmd == "sign":
        return sign(a.part, a.old, a.new, a.answer, a.reason)
    return status(a.gpu)
