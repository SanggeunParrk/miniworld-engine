"""``miniworld-engine dev bundle-verify``: the release gate. ONE command proves the wheels cover ``registry_module.csv``.

It installs the given wheels (unzips them: ``--wheels DIR``) or takes an already unzipped tree (``--tree DIR``), then runs the SAME
expansion bundle coverage uses (:mod:`miniworld_engine.build.registry_calls`) -- every row of the installed package's
``registry_module.csv``, and the kernel-level rows of ``registry.csv`` -- in fresh processes, one unit per free GPU as a work queue,
under the conditions of a user's first call:

* an EMPTY engine kernel cache per unit (``XDG_CACHE_HOME`` -> ``<out>/cache/<unit>``), so nothing is served from a cache;
* ``MINIWORLD_ENGINE_JIT_FALLBACK=0``: an artifact the bundle lacks raises ``PrebuiltMissing`` instead of compiling;
* no ``CUTLASS_PATH`` and an environment (``PATH``) without nvcc.

A line per unit is printed as it finishes (calls, ok, missing, error, the names of the missing artifacts); the exit status is nonzero
when any call is missing, errored, or never ran. Rows the arch cannot run are reported as ``not an <arch> row`` with the registry text.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

MODULE = "miniworld_engine.build.registry_calls"


def python_tag(python: str) -> str:
    return subprocess.check_output([python, "-c", "import sys; print('cp%d%d' % sys.version_info[:2])"], text=True).strip()


def stage_wheels(wheels: Path, dest: Path, python: str) -> Path:
    """Unzip the engine wheel and the kernel wheels of this Python's ABI into one tree (what ``pip install`` puts in site-packages)."""
    tag = python_tag(python)
    picked = [w for w in sorted(wheels.glob("*.whl")) if w.name.endswith("-py3-none-any.whl") or f"-{tag}-{tag}-" in w.name]
    if not any(w.name.startswith("miniworld_engine-") for w in picked):
        raise SystemExit(f"no miniworld_engine wheel in {wheels}")
    dest.mkdir(parents=True, exist_ok=True)
    for w in picked:
        with zipfile.ZipFile(w) as z:
            z.extractall(dest)
        print(f"[bundle-verify] installed {w.name}", flush=True)
    return dest


def unit_env(tree: Path, cache: Path, gpu: str, extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("CUTLASS_PATH", "CUDA_HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "XDG_CACHE_HOME")}
    env.update(PATH="/usr/bin:/bin", LD_LIBRARY_PATH="", PYTHONPATH=str(tree), XDG_CACHE_HOME=str(cache), CUDA_VISIBLE_DEVICES=gpu,
               MINIWORLD_ENGINE_JIT_FALLBACK="0")
    env.update(extra)
    return env


def describe(python: str, tree: Path, arch: str, env: dict[str, str]) -> dict:
    out = subprocess.run([python, "-m", MODULE, "expand", "--arch", arch, "--json"], env=env, cwd=tree.parent, text=True,
                         capture_output=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def read_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_unit(uid: str, cmd: list[str], total: int, env: dict, out: Path, timeout: float) -> list[dict]:
    """Run one unit's calls to the end: a process that dies (sticky CUDA error, crash, timeout) is restarted after the call it died in."""
    results = out / "results" / f"{uid}.jsonl"
    log = out / "logs" / f"{uid}.log"
    results.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    results.unlink(missing_ok=True)
    start = 0
    while True:
        with log.open("a") as fh:
            fh.write(f"=== start {start}\n")
            fh.flush()
            proc = subprocess.Popen([*cmd, "--start", str(start)] if "run-calls" in cmd else cmd, env=env, cwd=out,
                                    stdout=fh, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                rc = -9
        done = read_results(results)
        start = len(done)
        if rc == 0 or start >= total:
            return done
        if rc != 3:   # died without recording the call it was in
            with results.open("a") as fh:
                fh.write(json.dumps({"i": start, "label": f"call #{start}", "status": "error", "detail": f"process exited {rc}",
                                     "missing": [], "seconds": 0}) + "\n")
            start += 1
            if start >= total:
                return read_results(results)


def summarize(done: list[dict], total: int) -> dict:
    count = collections.Counter(r["status"] for r in done)
    names = collections.Counter(n for r in done for n in r["missing"])
    errors = collections.Counter(r["detail"] for r in done if r["status"] == "error")
    served = sorted({a.rsplit("@", 1)[0] for r in done for a in r.get("artifacts", []) if a.endswith("@prebuilt")})
    return {"calls": total, "ran": len(done), "ok": count["ok"], "missing": count["missing"], "error": count["error"], "artifacts": served,
            "oom": count["oom"], "unbundled": count["unbundled"],
            "never_ran": total - len(done), "names": dict(names), "errors": dict(errors),
            "first_missing": next((r["label"] for r in done if r["status"] == "missing"), ""),
            "notrun": dict(collections.Counter(r["detail"] for r in done if r["status"] in ("oom", "unbundled"))),
            "first_error": next((r["label"] for r in done if r["status"] == "error"), "")}


def table(rows: list[dict]) -> str:
    lines = ["| unit | calls | ok | missing | error | oom | unbundled | missing artifacts / errors / not run |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        if r.get("not_arch"):
            if r["kind"] == "kernel":   # 170 Triton rows share three reasons: one line per reason
                grouped[r["not_arch"]].append(r["id"].removeprefix("kernel "))
            else:
                lines.append(f"| {r['id']} | - | - | - | - | - | - | not an {r['arch']} row: {r['not_arch']} |")
            continue
        s = r["summary"]
        what = ", ".join(sorted(s["names"])) or ""
        for key in ("errors", "notrun"):
            if s[key]:
                what += (" ; " if what else "") + "; ".join(f"{k[:160]} (x{v})" for k, v in list(s[key].items())[:2])
        if s["never_ran"]:
            what += f" ; {s['never_ran']} never ran"
        lines.append(f"| {r['id']} | {s['calls']} | {s['ok']} | {s['missing']} | {s['error']} | {s['oom']} | {s['unbundled']} | {what} |")
    lines.extend(f"| kernel rows not callable on this arch ({len(names)}) | - | - | - | - | - | - | {reason}: {', '.join(names)} |"
                 for reason, names in grouped.items())
    return "\n".join(lines)


def verify(args) -> int:
    started = time.monotonic()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = args.python or sys.executable
    tree = stage_wheels(Path(args.wheels).resolve(), out / "installed", python) if args.wheels else Path(args.tree).resolve()
    extra = dict(item.split("=", 1) for item in args.env)
    gpus = [g for g in (args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",") if g != ""]
    arch = args.arch if args.arch.startswith("sm") else f"sm{args.arch}"
    data = describe(python, tree, arch, unit_env(tree, out / "cache" / "expand", "", extra))
    units: list[dict] = [{"kind": "row", "uid": f"r{r['row']:03d}", "id": r["id"], "key": r.get("key", r["id"]), "text": r.get("text", ""),
                      "calls": r["calls"], "not_arch": r["not_arch"], "excluded": r["excluded"], "arch": arch, "row": r["row"]}
                         for r in data["rows"]]
    if not args.no_kernels:
        units.extend({"kind": "kernel", "uid": "k-" + k["kernel"], "id": f"kernel {k['kernel']}", "calls": 1, "not_arch": k["reason"],
                          "excluded": [], "arch": arch, "kernel": k["kernel"], "backend": k["backend"]}
                     for k in data["kernels"])
    kernels = set(getattr(args, "kernel", None) or [])
    if args.row or kernels:   # a subset (``miniworld-engine verify`` names it from the graph): row numbers (1-based), the word "kernels", kernel names
        keep = {f"r{int(n):03d}" for n in args.row if n != "kernels"}
        units = [u for u in units if u["uid"] in keep or (u["kind"] == "kernel" and ("kernels" in args.row or u["kernel"] in kernels))]
    work = sorted((u for u in units if not u["not_arch"]), key=lambda u: -u["calls"])
    for u in units:
        if u["not_arch"]:
            u["summary"] = None
    print(f"[bundle-verify] {arch}: {sum(1 for u in units if u['kind'] == 'row')} registry rows, "
          f"{sum(u['calls'] for u in work if u['kind'] == 'row')} calls, {len(work)} units on {len(gpus)} GPU(s); "
          f"python {python}; tree {tree}", flush=True)
    for u in units:
        if u["not_arch"] and u["kind"] == "row":
            print(f"[bundle-verify] {u['id']}: not an {arch} row: {u['not_arch']}", flush=True)
    q: queue.Queue = queue.Queue()
    for u in work:
        q.put(u)
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                u = q.get_nowait()
            except queue.Empty:
                return
            env = unit_env(tree, out / "cache" / u["uid"], gpu, extra)
            (out / "cache" / u["uid"]).mkdir(parents=True, exist_ok=True)
            cmd = [python, "-m", MODULE]
            if u["kind"] == "row":
                cmd += ["run-calls", "--arch", arch, "--row", str(u["row"]), "--out", str(out / "results" / f"{u['uid']}.jsonl")]
            else:
                cmd += ["run-kernel", "--arch", arch, "--kernel", u["kernel"], "--out", str(out / "results" / f"{u['uid']}.jsonl")]
            t0 = time.monotonic()
            done = run_unit(u["uid"], cmd, u["calls"], env, out, args.timeout)
            u["summary"] = s = summarize(done, u["calls"])
            built = sorted(p.name for p in (out / "cache" / u["uid"]).rglob("*") if p.suffix in (".so", ".cubin"))
            u["built"] = built
            with lock:
                print(f"{datetime.datetime.now().astimezone():%H:%M:%S} gpu{gpu} {time.monotonic() - t0:6.0f}s {u['id']}: calls={s['calls']} ok={s['ok']} "
                      f"PrebuiltMissing={s['missing']} Error={s['error']}" + (f" oom={s['oom']}" if s["oom"] else "") + (f" unbundled={s['unbundled']}" if s["unbundled"] else "") + (f" never_ran={s['never_ran']}" if s["never_ran"] else "")
                      + (f" missing={sorted(s['names'])}" if s["names"] else "")
                      + (f" first_error={next(iter(s['errors']))!r}" if s["errors"] else "")
                      + (f" BUILT_IN_CACHE={built}" if built else ""), flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rowu = [u for u in units if u["kind"] == "row"]
    keru = [u for u in units if u["kind"] == "kernel"]

    def tot(us):
        ran = [u for u in us if u["summary"]]
        return {"units": len(ran), "calls": sum(u["summary"]["calls"] for u in ran), "ok": sum(u["summary"]["ok"] for u in ran),
                "missing": sum(u["summary"]["missing"] for u in ran), "error": sum(u["summary"]["error"] for u in ran),
                "never_ran": sum(u["summary"]["never_ran"] for u in ran), "oom": sum(u["summary"]["oom"] for u in ran),
                "unbundled": sum(u["summary"]["unbundled"] for u in ran),
                "units_clean": sum(1 for u in ran if u["summary"]["ok"] == u["summary"]["calls"]),
                "not_arch": sum(1 for u in us if not u["summary"])}

    totals = {"rows": tot(rowu), "kernels": tot(keru),
              "missing_artifacts": sorted({n for u in units if u["summary"] for n in u["summary"]["names"]}),
              "seconds": round(time.monotonic() - started)}
    (out / "summary.json").write_text(json.dumps({"totals": totals, "units": units}, indent=1) + "\n")
    # the artifact names every registry row was served from, per stable row key (module/stream/dims) with the digest of what the row declared:
    # tests/cpu/builder/data/registry_row_artifacts_<arch>.json holds the clean ones (``miniworld-engine verify`` merges them in)
    (out / "row_artifacts.json").write_text(json.dumps({"arch": arch, "rows": {
        u["key"]: {"text": u["text"], "artifacts": u["summary"]["artifacts"], "clean": u["summary"]["ok"] == u["summary"]["calls"],
                   "missing": sorted(u["summary"]["names"])}
        for u in rowu if u["summary"]}}, indent=1, sort_keys=True) + "\n")
    text = table(rowu + keru)
    (out / "table.md").write_text(text + "\n")
    print(text)
    print(f"[bundle-verify] rows: {totals['rows']}\n[bundle-verify] kernels: {totals['kernels']}\n"
          f"[bundle-verify] missing artifacts: {totals['missing_artifacts'] or 'none'}", flush=True)
    bad = any(t["missing"] or t["error"] or t["never_ran"] for t in (totals["rows"], totals["kernels"]))
    print("[bundle-verify] " + ("FAILED: the wheels do not cover what registry_module.csv declares" if bad else
                                "OK: every declared call that ran was served from the bundle (0 missing, 0 error)"), flush=True)
    notrun = [(t["oom"], t["unbundled"]) for t in (totals["rows"], totals["kernels"])]
    if any(a or b for a, b in notrun):
        print(f"[bundle-verify] NOT RUN (reported, not failures): {sum(a for a, _ in notrun)} calls out of memory at a declared shape, "
              f"{sum(b for _, b in notrun)} kernel units that need MathDx (never bundled)", flush=True)
    return 1 if bad else 0
