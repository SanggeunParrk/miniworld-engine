"""``miniworld-engine dev bundle-kernels`` / ``dev build-wheel``: the engine's prebuilt kernels, built and shipped.

1. **Coverage.** Coverage drivers run with the recorder on (``MINIWORLD_ENGINE_BUNDLE_RECORD``), a JIT root of the
   build's own (``<out>/cache``) and ``MINIWORLD_ENGINE_BUNDLE_ARCH`` (multi-arch ``gencodes`` requests compile only
   the bundle's arch). Every artifact a driver's processes resolve -- built, cached or already prebuilt -- is recorded
   with its key and the recipe that rebuilds it. Drivers:

   * ``engine`` (default): :mod:`miniworld_engine.build.bundle_coverage`, dealt to ``--workers`` processes per GPU
     (different artifacts compile in parallel; a worker that needs an artifact another one is building waits on its
     lock). Heavy shapes run last, one per GPU.
   * ``pytest:<paths>``: a test run (one process per test file), e.g. a release smoke set.
   * ``cmd:<command>``: any command, e.g. MiniWorld's ``python scripts/prebuild_engine.py --mode all``.

   The bundle being refreshed is itself installed while the drivers run (it is ``<prebuilt root>/<arch>``, inside the
   package), so an artifact whose key did not change is a prebuilt hit and is not compiled again: a refresh compiles
   only new keys.

2. **Replay** (``--replay <records.jsonl>``): rebuild the recorded artifacts from their recipes in this environment --
   another Python ABI, another torch / CUDA -- without running a GPU driver: CPU only, ``--jobs`` processes.

3. **Update** ``<prebuilt root>/<arch>-cu<XY>/`` (default under ``src/miniworld_engine/_prebuilt/``, not in git; one
   directory per arch and CUDA build):
   ``manifest.json`` (key -> path, kind, name, size, sha256, key material) and ``artifacts/``. Artifacts of this
   Python ABI (and every cubin) that the run did not reach are dropped (``--keep`` keeps them); other ABIs' extensions
   are kept.

4. **Wheel** (``dev build-wheel``): the engine's own wheel (setuptools, then the prebuilt artifacts of this Python ABI
   added), tagged ``cpXY-cpXY-manylinux_2_28_x86_64`` and pinned to the torch minor version the artifacts were built
   against; :func:`check_binaries` is the auditwheel-style check (only torch / CUDA / system libraries needed, glibc
   and libstdc++ symbol versions within manylinux_2_28).

See docs/kernels/prebuilt-kernels.md.
"""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import multiprocessing
import os
import platform
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from miniworld_engine.kernels import prebuilt

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[1]
#: manylinux_2_28 (AlmaLinux 8): the newest symbol versions a binary may need from the system.
MANYLINUX = "manylinux_2_28_x86_64"
MAX_SYMBOL_VERSIONS = {"GLIBC": (2, 28), "GLIBCXX": (3, 4, 25), "CXXABI": (1, 3, 11), "GCC": (8, 0, 0)}
#: Libraries an artifact may link: the manylinux system set, and what torch's own wheel provides.
ALLOWED_LIBS = re.compile(
    r"^(libc|libm|libdl|librt|libpthread|libgcc_s|libstdc\+\+|ld-linux-x86-64|libutil)\.so[.\d]*$"
    r"|^lib(c10|c10_cuda|torch|torch_cpu|torch_cuda|torch_python|shm)\.so$"
    r"|^lib(cudart|cublas|cublasLt|cuda|nvrtc|cusparse|cufft|curand|cusolver|nvJitLink)\.so[.\d]*$")


def log(message: str) -> None:
    print(f"[bundle-kernels] {message}", flush=True)


# --------------------------------------------------------------------------------------------------- environment
def driver_env(out: Path, arch: str, record: Path, gpu: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env[prebuilt.RECORD_ENV] = str(record)
    env["MINIWORLD_ENGINE_BUNDLE_ARCH"] = arch
    env["XDG_CACHE_HOME"] = str(out / "cache")
    env.setdefault("MINIWORLD_ENGINE_JIT_FALLBACK", "1")
    src = str(PKG_ROOT.parent)
    env["PYTHONPATH"] = os.pathsep.join([src, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def visible_gpus(spec: str) -> list[str]:
    if spec:
        return [g.strip() for g in spec.split(",") if g.strip()]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        return [g.strip() for g in vis.split(",") if g.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


# --------------------------------------------------------------------------------------------------- engine driver
class _Worker:
    def __init__(self, index: int, gpu: str, env: dict, logs: Path):
        self.index, self.gpu = index, gpu
        self.log = (logs / f"worker{index:03d}.log").open("a")
        self.proc = subprocess.Popen([sys.executable, "-m", "miniworld_engine.build.bundle_coverage", "serve"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     env=env, bufsize=1)
        self.case: dict | None = None
        self.started = 0.0

    def give(self, case: dict) -> None:
        self.case, self.started = case, time.monotonic()
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(case) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            assert self.proc.stdin is not None
            self.proc.stdin.close()
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=60)
        if self.proc.poll() is None:
            self.proc.kill()
        self.log.close()


def _interleave(cases: list[dict]) -> list[dict]:
    """Round-robin over (kind, module, width row, mode, dtype): the workers start on different modules, so they compile
    different artifacts at once instead of all waiting on the first module's."""
    groups: dict[tuple, list[dict]] = {}
    for c in cases:
        groups.setdefault((c["kind"], c.get("case"), c.get("dim"), c["train"], c["dtype"]), []).append(c)
    out: list[dict] = []
    queues = list(groups.values())
    while queues:
        out += [q.pop(0) for q in queues]
        queues = [q for q in queues if q]
    return out


def run_engine_driver(out: Path, arch: str, gpus: list[str], workers: int, case_filter: str = "") -> dict:
    """Deal ``bundle_coverage.cases()`` to ``workers`` processes over ``gpus``; heavy cases run last, one per GPU."""
    from miniworld_engine.build import bundle_coverage

    cases = bundle_coverage.cases()
    if case_filter:
        cases = [c for c in cases if case_filter in json.dumps(c)]
    light = _interleave([c for c in cases if not c["heavy"]])
    heavy = _interleave([c for c in cases if c["heavy"]])
    results: dict[int, dict] = {}
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record = out / "records" / "engine.jsonl"

    def phase(queue: list[dict], per_gpu: int) -> list[dict]:
        sel = selectors.DefaultSelector()
        pool: list[_Worker] = []
        retry: list[dict] = []
        queue = list(queue)
        total, done, t0 = len(queue), 0, time.monotonic()

        def spawn(i: int) -> _Worker:
            w = _Worker(i, gpus[i % len(gpus)], driver_env(out, arch, record, gpus[i % len(gpus)]), logs)
            sel.register(w.proc.stdout, selectors.EVENT_READ, w)
            return w

        pool = [spawn(i) for i in range(min(len(gpus) * per_gpu, max(1, len(queue))))]
        for w in pool:
            if queue:
                w.give(queue.pop(0))
        while any(w.case is not None for w in pool):
            for key, _ in sel.select(timeout=30):
                w: _Worker = key.data
                line = w.proc.stdout.readline() if w.proc.stdout else ""
                if not line:  # the worker died: its case failed, start a new worker
                    sel.unregister(key.fileobj)
                    if w.case is not None:
                        results[w.case["id"]] = {"id": w.case["id"], "status": "crash",
                                                 "detail": f"worker exited {w.proc.poll()}", "seconds": 0}
                        done += 1
                    w.close()
                    nw = spawn(w.index)
                    pool[pool.index(w)] = nw
                    if queue:
                        nw.give(queue.pop(0))
                    continue
                if not line.startswith("RESULT "):
                    continue
                r = json.loads(line[7:])
                results[r["id"]] = r
                done += 1
                if r["status"] == "oom" and w.case is not None and not w.case.get("retried"):
                    retry.append(dict(w.case, retried=True))
                w.case = None
                if queue:
                    w.give(queue.pop(0))
                if done % 50 == 0 or done == total:
                    log(f"  {done}/{total} cases ({time.monotonic() - t0:.0f} s)")
        for w in pool:
            with contextlib.suppress(Exception):
                sel.unregister(w.proc.stdout)
            w.close()
        return retry

    log(f"engine driver: {len(light)} cases on {len(gpus)} GPU(s) x {workers} workers, then {len(heavy)} heavy")
    retry = phase(light, workers)
    phase(heavy + retry, 1)
    summary: dict[str, int] = {}
    for r in results.values():
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    (out / "logs" / "engine_cases.jsonl").write_text(
        "".join(json.dumps({**c, **results.get(c["id"], {})}) + "\n" for c in cases))
    log(f"engine driver done: {summary}")
    return summary


def run_pytest_driver(out: Path, arch: str, gpus: list[str], workers: int, paths: list[str], name: str) -> int:
    """One pytest process per test file, ``workers`` per GPU at a time."""
    files: list[str] = []
    for p in paths:
        path = Path(p)
        files += sorted(str(f) for f in path.rglob("test_*.py")) if path.is_dir() else [str(path)]
    record = out / "records" / f"{name}.jsonl"
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, Path]] = []
    failed = 0
    slots = max(1, len(gpus) * workers)
    for i, f in enumerate(files):
        while len(running) >= slots:
            for proc, lf in list(running):
                if proc.poll() is not None:
                    running.remove((proc, lf))
                    failed += proc.returncode not in (0, 5)
            time.sleep(1)
        lf = logs / f"{name}-{Path(f).stem}.log"
        env = driver_env(out, arch, record, gpus[i % len(gpus)])
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f]
        log(f"  {' '.join(cmd[3:])} (GPU {gpus[i % len(gpus)]})")
        running.append((subprocess.Popen(cmd, cwd=REPO_ROOT if (REPO_ROOT / "tests").is_dir() else None, env=env,
                                         stdout=lf.open("w"), stderr=subprocess.STDOUT), lf))
    for proc, _ in running:
        proc.wait()
        failed += proc.returncode not in (0, 5)
    log(f"pytest driver {name}: {len(files)} files, {failed} with failures (see {logs})")
    return failed


def run_cmd_driver(out: Path, arch: str, command: str, name: str) -> int:
    record = out / "records" / f"{name}.jsonl"
    lf = out / "logs" / f"{name}.log"
    lf.parent.mkdir(parents=True, exist_ok=True)
    log(f"  {command}")
    rc = subprocess.call(command, shell=True, env=driver_env(out, arch, record), stdout=lf.open("w"),
                         stderr=subprocess.STDOUT)
    log(f"command driver {name}: exit {rc} (log {lf})")
    return rc


# --------------------------------------------------------------------------------------------------- records
def read_records(paths: list[Path], require_file: bool = True) -> dict[str, dict]:
    """key -> record (an artifact recorded by several processes is one entry)."""
    entries: dict[str, dict] = {}
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not require_file or Path(e["path"]).is_file():
                entries.setdefault(e["key"], e)
    return entries


def stale_keys(entries: dict[str, dict]) -> dict[str, list[str]]:
    """key -> the package files that changed since the artifact was built, for the entries this checkout can no longer key.

    An artifact's key is a hash of the normalized sources it reached (``material["nodes"]``, labels ``pkg:<path>`` for this
    package's own files). When any of those files differs now, the loader of THIS tree computes another key and never asks
    for the recorded one: the artifact is superseded (a recording that ran against an older tree, or ``--keep``). Files
    outside the package (``file:`` labels) cannot be checked here and never make an entry stale."""
    from miniworld_engine.kernels import _codegraph as cg

    now: dict[str, str | None] = {}
    stale: dict[str, list[str]] = {}
    for key, e in entries.items():
        changed = []
        for label, digest in (e.get("material") or {}).get("nodes") or []:
            if not label.startswith("pkg:"):
                continue
            if label not in now:
                path = cg.PKG_ROOT / label[4:]
                now[label] = cg.parse_file(str(path)).digest if path.is_file() else None
            if now[label] != digest:
                changed.append(label[4:] + ("" if now[label] else " (missing)"))
        if changed:
            stale[key] = changed
    return stale


def drop_stale(entries: dict[str, dict]) -> dict[str, dict]:
    """``entries`` without the superseded ones (:func:`stale_keys`); each dropped artifact is logged."""
    stale = stale_keys(entries)
    for key, changed in sorted(stale.items(), key=lambda kv: entries[kv[0]]["name"]):
        log(f"  stale, not bundled: {entries[key]['name']} {key[:12]} (changed: {', '.join(changed[:2])})")
    return {k: e for k, e in entries.items() if k not in stale}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_commit() -> str:
    """``<sha>`` of the engine checkout (``-dirty`` when tracked files under the package are modified), or ""."""
    try:
        sha = subprocess.run(["git", "-C", str(PKG_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PKG_ROOT), "status", "--porcelain", "--untracked-files=no", "--",
                                str(PKG_ROOT)], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return sha + ("-dirty" if dirty else "")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def _abi_tag(cache_tag: str) -> str:
    """``cpython-312`` -> ``cp312``."""
    m = re.match(r"cpython-(\d+)", cache_tag or "")
    return f"cp{m.group(1)}" if m else _safe(cache_tag or "py")


def variant_name(arch: str, cuda: str) -> str:
    """``sm90`` + CUDA ``12.9`` -> ``sm90-cu129``: the directory of one arch / CUDA build under ``_prebuilt/``."""
    return f"{arch}-cu{str(cuda).replace('.', '')}"


def _entry_python(entry: dict) -> str:
    return ((entry.get("material") or {}).get("python")) or "-"


# --------------------------------------------------------------------------------------------------- update
def update_prebuilt(entries: dict[str, dict], root: Path, arch: str, *, keep: bool = False) -> Path:
    """Merge recorded artifacts into ``root/<arch>``: copy the new ones, keep unchanged keys in place, drop artifacts of
    this Python ABI (and cubins) the run did not reach unless ``keep``; write the manifest."""
    from miniworld_engine.kernels import _codegraph as cg

    versions = cg._versions()
    # one directory per arch and CUDA build (a wheel carries several: torch picks the one built for its own CUDA)
    target = root / variant_name(arch, versions["cuda"])
    old: dict = {}
    mf = target / prebuilt.MANIFEST
    if mf.is_file():
        old = json.loads(mf.read_text())
        if (old.get("torch"), old.get("cuda")) != (versions["torch"], versions["cuda"]) \
                or old.get("format") != prebuilt.MANIFEST_FORMAT or old.get("key_scheme") != cg.KEY_SCHEME:
            raise SystemExit(f"{target} holds a bundle for torch {old.get('torch')} / CUDA {old.get('cuda')} "
                             f"(format {old.get('format')}); this is torch {versions['torch']} / CUDA {versions['cuda']}. "
                             "Use another --prebuilt-root, or remove it.")
    artifacts: dict[str, dict] = dict(old.get("artifacts", {}))
    here = cg.python_abi()
    if not keep:
        for key, a in list(artifacts.items()):
            in_scope = a["kind"] == "cubin" or _entry_python(a) == here
            if in_scope and key not in entries:
                with contextlib.suppress(OSError):
                    (target / a["path"]).unlink()
                    for s in a.get("sidecars") or []:
                        (target / a["path"]).with_suffix(s).unlink(missing_ok=True)
                del artifacts[key]
    copied = 0
    for key, e in sorted(entries.items(), key=lambda kv: (kv[1]["kind"], kv[1]["name"])):
        if "<mathdx>" in json.dumps((e.get("material") or {}).get("flags")):
            # NVIDIA MathDx / cuBLASDx: not redistributed in compiled form; such kernels JIT-build where they are used
            log(f"  not bundled (compiles MathDx): {e['name']}")
            continue
        src = Path(e["path"])
        if e["kind"] == "ext":
            rel = Path("artifacts") / "ext" / _abi_tag(_entry_python(e)) / f"{_safe(e['name'])}-{key[:16]}" / src.name
        else:
            rel = Path("artifacts") / "cubin" / _safe(e["name"].split("/", 1)[0]) / \
                f"{_safe(e['name'].split('/', 1)[-1])}-{key[:16]}.cubin"
        dst = target / rel
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        sidecars = []
        for suffix in e.get("sidecars") or []:
            s, d = src.with_suffix(suffix), dst.with_suffix(suffix)
            if s.is_file():
                if s.resolve() != d.resolve():
                    shutil.copy2(s, d)
                sidecars.append(suffix)
        artifacts[key] = {
            "kind": e["kind"], "name": e["name"], "path": rel.as_posix(), "size": dst.stat().st_size,
            "sha256": _sha256(dst), "sidecars": sidecars, "material": e.get("material"),
            **({"module": e["module"], "python_module": e.get("python_module", True)} if e["kind"] == "ext" else {}),
        }
    # empty directories left by dropped artifacts
    for d in sorted((target / "artifacts").glob("**/"), key=lambda p: -len(p.parts)) if (target / "artifacts").is_dir() else []:
        with contextlib.suppress(OSError):
            d.rmdir()
    manifest = {
        "format": prebuilt.MANIFEST_FORMAT,
        "key_scheme": cg.KEY_SCHEME,
        "arch": arch,
        "torch": versions["torch"],
        "cuda": versions["cuda"],
        "pythons": sorted({_entry_python(a) for a in artifacts.values() if a["kind"] == "ext"}),
        "engine": prebuilt.engine_version(),
        "engine_commit": engine_commit(),
        "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "artifacts": artifacts,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / prebuilt.MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    log(f"{target}: {len(artifacts)} artifacts ({copied} copied in, {len(old.get('artifacts', {}))} before)")
    return target


# --------------------------------------------------------------------------------------------------- replay
def keyed_recipe(entry: dict) -> dict:
    """``entry``'s recipe with the ``extra_ldflags`` its key was computed from. torch's ``load`` appends its own link
    flags (``-L<torch>/lib -lc10 ... -lcudart``) to the list it is given, in place, and records up to 2.3.1 stored
    that list after the call: replayed as recorded, every extension with ``extra_ldflags`` (the SWA atom DiT, the
    token DiT's owned GEMM) keyed differently from the one a model asks for. The key material holds the flags as
    they were passed."""
    recipe = entry["recipe"]
    flags = (entry.get("material") or {}).get("flags") or {}
    if (recipe.get("fn") != "ext" or "extra_ldflags" not in recipe.get("kwargs", {})
            or any("<" in f for f in flags.get("ldflags", ["<"]))):  # absent, or machine-free placeholders
        return recipe
    return {**recipe, "kwargs": {**recipe["kwargs"], "extra_ldflags": list(flags["ldflags"])}}


def rebase_recipe(recipe: dict, recorded_jit: str | None, jit: str | None, package: Path = PKG_ROOT) -> dict:
    """``recipe`` with the recording's checkout and JIT root replaced by this one's: a replay compiles THIS tree's
    sources (and writes generated files under this JIT root), so it keys them as this tree's own builds would. (A
    recipe still carries the generated TriMul header and ``compile_text`` texts of the recording: refresh the records
    with a coverage run after editing the sources they are generated from.)"""
    text = json.dumps(recipe)
    m = re.search(r'(?:"|-I)(/[^"]*?/miniworld_engine)/(?:kernels|integrations|modules)/', text)
    for old, new in ((recipe.get("jit") or recorded_jit, jit), (m.group(1) if m else None, str(package))):
        if old and new and old != new:
            text = text.replace(old, new)
    return json.loads(text)


def _replay_one(task: dict) -> dict:
    """Rebuild one recorded artifact here (a worker process). Returns {old, key, path} or {old, error}."""
    from miniworld_engine.kernels import _codegraph as cg
    from miniworld_engine.kernels import _nvcc

    r = rebase_recipe(task["recipe"], task.get("jit"), cg.context().jit)
    try:
        if r["fn"] == "derived":
            base = _replay_one({"recipe": task["base"], "old": r["base_key"], "jit": task.get("jit")})
            if "error" in base:
                return {"old": task["old"], "error": "base: " + base["error"]}
            module, fn = r["replay"][0].split(":")
            import importlib

            out = getattr(importlib.import_module(module), fn)(Path(base["path"]), *r["replay"][1])
            return {"old": task["old"], "path": str(out)}
        if r["fn"] == "cubin":
            for p, t in (r.get("overlay") or {}).items():
                cg.publish(p, t)
            out = _nvcc.cubin(r["name"], r["flags"], source=r["source"], text=r["text"], family=r["family"],
                              toolchain=r["toolchain"], ptxas_log=r["ptxas_log"], timeout=r["timeout"])
            return {"old": task["old"], "path": str(out)}
        if r["inline"]:
            ext = _nvcc.load_inline_extension(r["name"], **r["kwargs"])
        else:
            ext = _nvcc.load_extension(r["name"], r["sources"], **r["kwargs"])
        return {"old": task["old"], "path": getattr(ext, "__file__", str(ext))}
    except Exception as exc:  # one failed rebuild is reported, the rest go on
        import traceback

        return {"old": task["old"], "error": f"{type(exc).__name__}: {str(exc)[-300:]}",
                "detail": f"{exc}\n{traceback.format_exc()}"}


def replay(records: list[Path], out: Path, root: Path, arch: str, jobs: int, only: str = "", *, stale: bool = True) -> Path:
    """Rebuild every recorded recipe in this environment, ``jobs`` at a time; returns the new records file.
    ``stale=False`` drops the recordings this checkout can no longer key (:func:`stale_keys`) first."""
    recorded = read_records(records, require_file=False)
    recipes = {k: e for k, e in (recorded if stale else drop_stale(recorded)).items() if e.get("recipe")}
    if only:
        recipes = {k: e for k, e in recipes.items() if only in e["kind"] or only in e["name"]}
    record = out / "records" / f"replay-{_abi_tag(sys.implementation.cache_tag)}.jsonl"
    if record.exists():  # one replay, one record: never pack an earlier run's artifacts
        record.rename(record.with_suffix(f".{int(time.time())}.prev"))
    env = driver_env(out, arch, record)
    os.environ.update({k: env[k] for k in (prebuilt.RECORD_ENV, "MINIWORLD_ENGINE_BUNDLE_ARCH", "XDG_CACHE_HOME",
                                           "MINIWORLD_ENGINE_JIT_FALLBACK")})
    os.environ.setdefault("MINIWORLD_ENGINE_TARGET_ARCH", arch)
    # no GPU here: torch must not try to detect the arch of an extension that names none
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{int(arch[2:]) // 10}.{int(arch[2:]) % 10}")
    # the bundle being built is the only one in view: its own artifacts are reused (another ABI's cubins, an
    # interrupted run), nothing from the package's own _prebuilt/ or elsewhere
    os.environ["MINIWORLD_ENGINE_PREBUILT_DIR"] = str(root)
    os.environ.pop(prebuilt.BUNDLE_ENV, None)
    def jit_of(e: dict) -> str | None:
        """The JIT root the recording ran with (older records do not name it: take it from the artifact's path)."""
        if e["recipe"].get("jit"):
            return e["recipe"]["jit"]
        m = re.match(r"(.*/miniworld_engine_jit)/", e["path"])
        return m.group(1) if m else None

    tasks = []
    for key, e in recipes.items():
        r = e["recipe"]
        if r["fn"] == "derived":
            base = recipes.get(r["base_key"])
            if base is None:
                log(f"  skip {e['name']}: its base cubin was not recorded")
                continue
            tasks.append({"old": key, "recipe": keyed_recipe(e), "base": keyed_recipe(base), "jit": jit_of(base)})
        else:
            tasks.append({"old": key, "recipe": keyed_recipe(e), "jit": jit_of(e)})
    tasks.sort(key=lambda t: t["recipe"]["fn"] == "derived")
    log(f"replaying {len(tasks)} recipes with {jobs} processes")
    started, errors = time.monotonic(), []
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        for i, res in enumerate(pool.map(_replay_one, tasks), 1):
            if "error" in res:
                errors.append(res)
                name = recipes[res["old"]]["name"]
                detail = out / "logs" / f"replay-failed-{_safe(name)}.log"
                detail.write_text(res.get("detail", res["error"]))
                log(f"  FAILED {name}: {res['error']} (full output: {detail})")
            if i % 20 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)} ({time.monotonic() - started:.0f} s)")
    # A recipe recorded without a CUTLASS checkout (the stand-in include path) cannot build; it resolves from the artifact a sibling
    # recipe (the same kernel recorded by another Python ABI, replayed here to the same key) has built, which a parallel pool may not
    # have finished when the stand-in task ran. Once the pool is done, try those again, one at a time.
    again = [res for res in errors if res["error"].startswith("CutlassMissing")]
    for res in again:
        task = next(t for t in tasks if t["old"] == res["old"])
        retried = _replay_one(task)
        if "error" not in retried:
            errors.remove(res)
            log(f"  {recipes[res['old']]['name']}: resolved from a sibling recipe's artifact")
    log(f"replay: {len(tasks) - len(errors)} rebuilt / loaded, {len(errors)} failed, {time.monotonic() - started:.0f} s")
    return record


# --------------------------------------------------------------------------------------------------- command
def bundle_kernels(args) -> int:
    started = time.monotonic()
    arch = args.arch if args.arch.startswith("sm") else f"sm{args.arch}"
    out = Path(args.out).resolve()
    root = Path(args.prebuilt_root).resolve() if args.prebuilt_root else prebuilt.PREBUILT_DIR
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    from miniworld_engine.kernels import _codegraph as cg

    timings: dict[str, float] = {}
    if args.replay:
        t0 = time.monotonic()
        rec = replay([Path(p) for p in args.replay], out, root, arch, args.jobs, args.only,
                     stale=not getattr(args, "drop_stale", False))
        timings["replay"] = time.monotonic() - t0
        records = [rec]
        keep = args.keep  # artifacts of this ABI (and cubins) the replay did not produce are pruned
    else:
        device_arch = cg.target_arch()
        if device_arch not in ("cpu", arch):
            log(f"this GPU is {device_arch}, not {arch}: a bundle is built on its own arch")
            return 2
        os.environ["MINIWORLD_ENGINE_PREBUILT_DIR"] = str(root)  # the drivers see the bundle being refreshed
        if not args.collect_only:
            for old in (out / "records").glob("*.jsonl"):
                old.rename(old.with_suffix(f".{int(time.time())}.prev"))
            gpus = visible_gpus(args.gpus)
            for i, spec in enumerate(args.coverage or ["engine"]):
                t0 = time.monotonic()
                if spec == "engine" or spec.startswith("engine:"):
                    run_engine_driver(out, arch, gpus, args.workers, spec.partition(":")[2])
                elif spec.startswith("pytest:"):
                    run_pytest_driver(out, arch, gpus, args.workers, spec[7:].split(","), f"pytest{i}")
                elif spec.startswith("cmd:"):
                    run_cmd_driver(out, arch, spec[4:], f"cmd{i}")
                else:
                    log(f"unknown coverage driver {spec!r} (engine | pytest:<paths> | cmd:<command>)")
                    return 2
                timings[spec[:40]] = time.monotonic() - t0
        records = sorted((out / "records").glob("*.jsonl"))
        keep = args.keep
    entries = read_records(records)
    if getattr(args, "drop_stale", False):
        entries = drop_stale(entries)
    if not entries:
        log("no artifacts were recorded")
        return 1
    target = update_prebuilt(entries, root, arch, keep=keep)
    manifest = json.loads((target / prebuilt.MANIFEST).read_text())
    total = sum(a["size"] for a in manifest["artifacts"].values())
    kinds: dict[str, int] = {}
    for a in manifest["artifacts"].values():
        k = a["kind"] if a["kind"] == "cubin" else f"ext-{_abi_tag(_entry_python(a))}"
        kinds[k] = kinds.get(k, 0) + 1
    origins: dict[str, int] = {}
    for e in entries.values():
        origins[e["origin"]] = origins.get(e["origin"], 0) + 1
    summary = {"seconds": round(time.monotonic() - started, 1), "phases": {k: round(v, 1) for k, v in timings.items()},
               "recorded": len(entries), "origins": origins, "bundle": kinds, "bytes": total, "target": str(target)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done in {summary['seconds']:.0f} s: {summary}")
    return 0


# --------------------------------------------------------------------------------------------------- wheel
def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text))


def check_binaries(paths: list[Path]) -> list[str]:
    """auditwheel-style findings for shared objects: a NEEDED library outside :data:`ALLOWED_LIBS`, an RPATH /
    RUNPATH, or a symbol version newer than manylinux_2_28 allows. Empty list = pass."""
    problems = []
    for so in paths:
        dyn = subprocess.run(["readelf", "-dW", str(so)], capture_output=True, text=True, check=False).stdout
        problems += [f"{so.name}: needs {lib}" for lib in re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
                     if not ALLOWED_LIBS.match(lib)]
        for kind, value in re.findall(r"\((RPATH|RUNPATH)\)\s+Library \w+: \[([^\]]+)\]", dyn):
            problems.append(f"{so.name}: {kind} {value}")
        ver = subprocess.run(["readelf", "-VW", str(so)], capture_output=True, text=True, check=False).stdout
        needs = set(re.findall(r"Name: ((?:GLIBC|GLIBCXX|CXXABI|GCC)_[\d.]+)", ver))
        for need in sorted(needs):
            family, _, number = need.partition("_")
            if _version_tuple(number) > MAX_SYMBOL_VERSIONS[family]:
                problems.append(f"{so.name}: needs {need} (manylinux_2_28 allows up to "
                                f"{family}_{'.'.join(map(str, MAX_SYMBOL_VERSIONS[family]))})")
    return problems


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


#: ``--split``: where the engine's kernel wheels exist (one per Python ABI the release replays), so an install elsewhere does
#: not ask for a wheel no index carries; there the engine JIT-builds as from an sdist.
KERNEL_WHEEL_MARKER = ('sys_platform == "linux" and platform_machine == "x86_64" and '
                       '(python_version == "3.10" or python_version == "3.12")')


def kernel_dist_name(variant: str) -> str:
    """``sm90-cu129`` -> ``miniworld-engine-kernels-sm90-cu129``: the distribution that carries one ``_prebuilt/`` variant."""
    return f"miniworld-engine-kernels-{variant}"


def _write_wheel(path: Path, files: list[tuple[str, bytes]], dist_info: str, metadata: str, tag: str,
                 purelib: bool) -> None:
    """A wheel from (name, bytes) pairs plus its ``METADATA`` / ``WHEEL`` / ``RECORD``."""
    record = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        wheel_meta = (f"Wheel-Version: 1.0\nGenerator: miniworld-engine build-wheel\n"
                      f"Root-Is-Purelib: {'true' if purelib else 'false'}\nTag: {tag}\n")
        for n, data in [*files, (f"{dist_info}/METADATA", metadata.encode()), (f"{dist_info}/WHEEL", wheel_meta.encode())]:
            z.writestr(n, data)
            record.append(f"{n},{_record_hash(data)},{len(data)}")
        record.append(f"{dist_info}/RECORD,,")
        z.writestr(f"{dist_info}/RECORD", "\n".join(record) + "\n")


def build_split_wheels(base: Path, variants: list[tuple[str, list[tuple[str, bytes]], str]], major_minor: str,
                       tag: str, out: Path) -> list[Path]:
    """``--split``: one platform wheel per prebuilt variant, named ``miniworld-engine-kernels-<variant>``, whose files sit
    at ``miniworld_engine/_prebuilt/<variant>/`` (pip installs them into the engine package, where ``prebuilt`` looks:
    no loader change), pinned to the engine version and the artifacts' torch minor; and the pure engine wheel, which
    requires every kernel wheel at the same version where they exist (``KERNEL_WHEEL_MARKER``). Each file is one CUDA
    build's kernels, so each stays under PyPI's per-file limit; an install still pulls the same set as one wheel did."""
    with zipfile.ZipFile(base) as zin:
        names = zin.namelist()
        dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        version = dist_info.removeprefix("miniworld_engine-").removesuffix(".dist-info")
        meta = zin.read(f"{dist_info}/METADATA").decode()
        lic = [(n, zin.read(n)) for n in names if n.startswith(f"{dist_info}/licenses/")]
        engine_files = [(n, zin.read(n)) for n in names
                        if n not in (f"{dist_info}/METADATA", f"{dist_info}/WHEEL", f"{dist_info}/RECORD")]
    wheels = []
    requires = []
    for variant, files, builds in variants:
        dist = kernel_dist_name(variant)
        norm = dist.replace("-", "_")
        k_info = f"{norm}-{version}.dist-info"
        k_meta = (f"Metadata-Version: 2.4\nName: {dist}\nVersion: {version}\n"
                  f"Summary: miniworld-engine prebuilt kernels for {variant} ({builds}); installed by miniworld-engine=={version}\n"
                  f"License-Expression: MIT AND Apache-2.0 AND BSD-3-Clause\n"
                  + "".join(f"License-File: {n.split('/licenses/', 1)[1]}\n" for n, _ in lic)
                  + f"Requires-Python: >=3.10\nRequires-Dist: torch=={major_minor}.*\n"
                  f"Description-Content-Type: text/markdown\n\n"
                  f"Prebuilt GPU kernels of [miniworld-engine](https://pypi.org/project/miniworld-engine/) {version} for "
                  f"`{variant}` ({builds}). Installed by `pip install miniworld-engine=={version}`; not used on its own.\n")
        wheel = out / f"{norm}-{version}-{tag}-{tag}-{MANYLINUX}.whl"
        _write_wheel(wheel, [*files, *((n.replace(f"{dist_info}/", f"{k_info}/", 1), d) for n, d in lic)], k_info, k_meta,
                     f"{tag}-{tag}-{MANYLINUX}", purelib=False)
        wheels.append(wheel)
        requires.append(f"Requires-Dist: {dist}=={version}; {KERNEL_WHEEL_MARKER}\n")
    # the engine: the pure wheel as setuptools built it, plus the kernel requirements (before the first blank line: the
    # description follows it)
    head, sep, body = meta.partition("\n\n")
    engine = out / f"miniworld_engine-{version}-py3-none-any.whl"
    _write_wheel(engine, engine_files, dist_info, head + "\n" + "".join(requires).rstrip("\n") + sep + body,
                 "py3-none-any", purelib=True)
    return [engine, *wheels]


def build_wheel(args) -> int:
    """The engine's platform wheel: setuptools builds the pure wheel, then the prebuilt artifacts of this Python ABI
    go in as package data, the torch requirement is pinned to the artifacts' torch minor, and the tag says
    ``cpXY-cpXY-manylinux_2_28_x86_64``. ``--split``: ``build_split_wheels``."""
    from miniworld_engine.kernels import _codegraph as cg

    root = Path(args.prebuilt_root).resolve() if args.prebuilt_root else prebuilt.PREBUILT_DIR
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    here = cg.python_abi()
    tag = _abi_tag(here)
    with tempfile.TemporaryDirectory() as tmp:
        # setuptools copies the package into REPO_ROOT/build/lib and never deletes from it: a file removed from src/ since
        # the last build would ship, and change the installed kernels' native-cache identity (2.4.0 caught four deleted .cu)
        for stale in (REPO_ROOT / "build" / "lib", *(REPO_ROOT / "build").glob("bdist.*")):
            shutil.rmtree(stale, ignore_errors=True)
        log(f"building the pure wheel of {REPO_ROOT} with {sys.executable}")
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO_ROOT), "--no-deps", "--no-build-isolation",
                        "-q", "-w", tmp], check=True)
        base = next(Path(tmp).glob("miniworld_engine-*.whl"))
        manifests = sorted(root.glob(f"*/{prebuilt.MANIFEST}"))
        if not manifests:
            log(f"no prebuilt kernels under {root}: run `dev bundle-kernels` first")
            return 1
        torch_versions = set()
        extra: list[tuple[str, bytes]] = []
        variants: list[tuple[str, list[tuple[str, bytes]], str]] = []
        binaries: list[Path] = []
        unpacked = Path(tmp) / "unpacked"
        for mf in manifests:
            manifest = json.loads(mf.read_text())
            torch_versions.add((manifest["torch"], manifest["cuda"]))
            arch_dir = mf.parent
            keep = {k: a for k, a in manifest["artifacts"].items()
                    if a["kind"] == "cubin" or _entry_python(a) == here}
            dropped = len(manifest["artifacts"]) - len(keep)
            variants.append((arch_dir.name, [], f"torch {manifest['torch']} / CUDA {manifest['cuda']}"))
            manifest = dict(manifest, artifacts=keep, pythons=[here])
            prefix = f"miniworld_engine/_prebuilt/{arch_dir.name}"
            extra.append((f"{prefix}/{prebuilt.MANIFEST}", (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()))
            variants[-1][1].append(extra[-1])
            for a in keep.values():
                for rel in [a["path"], *[str(Path(a["path"]).with_suffix(s)) for s in a.get("sidecars") or []]]:
                    data = (arch_dir / rel).read_bytes()
                    extra.append((f"{prefix}/{rel}", data))
                    variants[-1][1].append(extra[-1])
                    if rel.endswith(".so"):
                        p = unpacked / arch_dir.name / rel
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(data)
                        binaries.append(p)
            log(f"{arch_dir.name}: {len(keep)} artifacts for {tag} ({dropped} of other Python ABIs left out)")
        minors = {".".join(t.split("+")[0].split(".")[:2]) for t, _ in torch_versions}
        if len(minors) != 1:
            log(f"the bundles were built against different torch minor versions: {sorted(torch_versions)}")
            return 1
        major_minor = next(iter(minors))
        builds = ", ".join(f"{t} / CUDA {c}" for t, c in sorted(torch_versions))
        problems = check_binaries(binaries)
        for p in problems:
            log(f"  binary check: {p}")
        if getattr(args, "split", False):
            wheels = build_split_wheels(base, variants, major_minor, tag, out)
            for w in wheels:
                size = w.stat().st_size
                log(f"wheel: {w} ({size / 1e6:.1f} MB)" + ("  over PyPI's 100 MB file limit" if size > 100e6 else ""))
            if problems:
                log(f"  {len(problems)} binary check finding(s): the {MANYLINUX} tag is not honest until they are fixed")
                return 3 if args.strict else 0
            log(f"  binary check: {len(binaries)} shared objects, no findings")
            return 0
        with zipfile.ZipFile(base) as zin:
            names = zin.namelist()
            dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
            version = dist_info.removeprefix("miniworld_engine-").removesuffix(".dist-info")
            wheel = out / f"miniworld_engine-{version}-{tag}-{tag}-{MANYLINUX}.whl"
            record: list[str] = []
            with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for n in names:
                    if n.startswith(f"{dist_info}/") and n.rsplit("/", 1)[-1] in ("RECORD", "WHEEL", "METADATA"):
                        continue
                    if n.startswith("miniworld_engine/_prebuilt/"):
                        continue
                    data = zin.read(n)
                    zout.writestr(zin.getinfo(n), data)
                    record.append(f"{n},{_record_hash(data)},{len(data)}")
                for n, data in extra:
                    zout.writestr(n, data)
                    record.append(f"{n},{_record_hash(data)},{len(data)}")
                meta = zin.read(f"{dist_info}/METADATA").decode()
                meta, count = re.subn(r"^Requires-Dist: torch[^;\n]*$", f"Requires-Dist: torch=={major_minor}.*",
                                      meta, count=1, flags=re.MULTILINE)
                if not count:
                    meta = meta.replace("\n\n", f"\nRequires-Dist: torch=={major_minor}.*\n\n", 1)
                wheel_meta = (f"Wheel-Version: 1.0\nGenerator: miniworld-engine build-wheel\nRoot-Is-Purelib: false\n"
                              f"Tag: {tag}-{tag}-{MANYLINUX}\n")
                for n, text in (("METADATA", meta), ("WHEEL", wheel_meta)):
                    data = text.encode()
                    zout.writestr(f"{dist_info}/{n}", data)
                    record.append(f"{dist_info}/{n},{_record_hash(data)},{len(data)}")
                record.append(f"{dist_info}/RECORD,,")
                zout.writestr(f"{dist_info}/RECORD", "\n".join(record) + "\n")
    size = wheel.stat().st_size
    log(f"wheel: {wheel} ({size / 1e6:.1f} MB; {builds} -> requires torch=={major_minor}.*)")
    if size > 100e6:
        log("  over PyPI's 100 MB file limit")
    if problems:
        log(f"  {len(problems)} binary check finding(s): the {MANYLINUX} tag is not honest until they are fixed")
        return 3 if args.strict else 0
    log(f"  binary check: {len(binaries)} shared objects, no findings")
    return 0


# --------------------------------------------------------------------------------------------------- code graph
def known_artifacts(extra: list[str] = ()) -> list[dict]:
    """Every artifact whose key material is on this machine: the local cache's sidecars, the installed bundles, and
    ``extra`` manifests / record files."""
    from miniworld_engine import settings

    found: list[dict] = []
    root = settings.engine_jit_root()
    for side in list(root.glob("*/*.json")) + list(root.glob("torch_extensions/*/artifact.json")):
        with contextlib.suppress(OSError, ValueError, KeyError):
            data = json.loads(side.read_text())
            found.append({"name": data["name"], "key": data["key"], "material": data["material"],
                          "where": str(side.parent if side.name == "artifact.json" else side.with_suffix(".cubin"))})
    manifests = [b.root / prebuilt.MANIFEST for b in prebuilt.bundles()] + [Path(p) for p in extra]
    for m in manifests:
        with contextlib.suppress(OSError, ValueError):
            if m.suffix == ".jsonl":
                found.extend({"name": e["name"], "key": e["key"], "material": e.get("material"), "where": str(m)}
                             for e in read_records([m]).values())
                continue
            data = json.loads(m.read_text())
            for key, a in data.get("artifacts", {}).items():
                found.append({"name": a["name"], "key": key, "material": a.get("material"), "where": str(m)})
    return found


def file_label(path: Path) -> str:
    from miniworld_engine.kernels import _codegraph as cg

    p = os.path.abspath(path)
    rel = cg._under(p, cg.context().package)
    return f"pkg:{rel}" if rel is not None else f"file:{path.name}"


def code_graph(args) -> int:
    from miniworld_engine.kernels import _codegraph as cg

    known = known_artifacts(args.manifest)
    if args.affected:
        path = Path(args.affected)
        label = file_label(path)
        digest = cg.parse_file(os.path.abspath(path)).digest if path.is_file() else None
        hits = []
        for a in known:
            nodes = {n[0]: n[1] for n in (a.get("material") or {}).get("nodes") or []}
            if label in nodes:
                hits.append((a, "rebuilds" if nodes[label] != digest else "unchanged (same normalized content)"))
        print(f"{label}: reached by {len(hits)} of {len(known)} known artifacts")
        for a, state in sorted(hits, key=lambda h: h[0]["name"]):
            print(f"  {a['name']:60s} {a['key'][:16]}  {state}")
        return 0
    target = args.target
    if target and Path(target).is_file():
        path = Path(target)
        graph = cg.closure([path], [os.path.abspath(p) for p in args.include] + [str(path.parent)])
        print(f"graph of {file_label(path)} (-I {', '.join(args.include) or '(its directory)'}):")
        for n in graph.nodes:
            print(f"  {n.label:70s} {n.digest[:16]}")
        for spelling in graph.external:
            print(f"  ext:{spelling}")
        for frm, spelling, to in graph.edges:
            print(f'    {frm} --#include "{spelling}"--> {to}')
        users = [a for a in known if file_label(path) in dict(map(tuple, (a.get("material") or {}).get("nodes") or []))]
        print(f"reached by {len(users)} known artifact(s): " + ", ".join(sorted({a["name"] for a in users})))
        return 0
    matches = [a for a in known if target and (a["name"] == target or a["key"].startswith(target) or target in a["name"])]
    if not matches:
        print(f"no known artifact matches {target!r} ({len(known)} known: local cache, installed bundles, --manifest)")
        return 1
    for a in matches:
        m = a.get("material") or {}
        print(f"{a['name']}  key {a['key']}  ({a['where']})")
        for label, digest in m.get("nodes", []):
            now = ""
            if label.startswith("pkg:"):
                p = cg.PKG_ROOT / label[4:]
                cur = cg.parse_file(str(p)).digest if p.is_file() else None
                now = "" if cur == digest else ("  CHANGED" if cur else "  MISSING")
            print(f"  {label:70s} {digest[:16]}{now}")
        for spelling in m.get("external", []):
            print(f"  ext:{spelling}")
        print(f"  flags: {json.dumps(m.get('flags'))}")
        print(f"  arch {m.get('arch')}  torch {m.get('torch')}  cuda {m.get('cuda')}  python {m.get('python')}")
    return 0
