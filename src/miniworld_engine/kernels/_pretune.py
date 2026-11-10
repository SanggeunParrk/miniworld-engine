"""Pretune the launch-config tables: time every candidate config of every kernel launch the requirements reach, on the GPU.

    python -m miniworld_engine.kernels._pretune --candidates                      # CPU, once: write kernels/<family>/configs/candidates.json
    python -m miniworld_engine.kernels._pretune [--run DIR] [--config YAML] [--modules M ...] [--rows N ...]
                                                [--deadline EPOCH] [--hard-deadline EPOCH]          # GPU: tune (one process per GPU)
    python -m miniworld_engine.kernels._pretune --status [--run DIR]                 # units done / claimed / failed, keys tuned
    python -m miniworld_engine.kernels._pretune --merge [--run DIR ...] [--dry-run] # CPU: shards -> kernels/<family>/configs/<gpu>.json

**Candidates** (``--candidates``, imported once, then a reviewed file): per configured Triton op, its decorator ``default`` (every
entry of a default list), then the old engine's compact space ``autotune/configs/default/<op>.csv`` (read from git ``HEAD``), then,
for an op with at most :data:`GRID_SLICE_BELOW` such configs, the old ``grid/<op>.csv``'s ``num_warps`` x ``num_stages`` values around
each of them (nearest first), then every config the current ``<gpu>.json`` tables hold for the op (measured winners); deduplicated,
at most :data:`MAX_CANDIDATES`. A CSV whose columns are not exactly the decorator default's tile arguments is stale and not used. An op
with no (usable) CSV gets a generated space around each default: every tile argument (``BLOCK_*`` / ``GROUP_*``) halved and doubled
one at a time, ``num_warps`` halved and doubled, ``num_stages`` -1 / +1, then the product of the tile halvings / doublings with
``num_warps`` x {1/2, 1, 2} while the list stays under the cap. Written to ``kernels/<family>/configs/candidates.json`` as
``{op: [{"kwargs", "num_warps", "num_stages", ["maxnreg"]}, ...]}``. Native ``choose`` sites pass their own candidates.

**What is tuned = what the requirements reach.** The work list is every case of the measurement config (``measure/configs/speed.yaml``
by default: every requirements row, every length, min depth, min samples, the row's dtypes and modes) for the engine backends
(``cuda``, ``triton``) the case's class has. A case's step (eval: the forward under ``no_grad``; train: forward + backward, the measurer's
own ``make_step``) runs once, eagerly, with this module installed as ``_launch_config.PRETUNE``: at every configured launch whose
``(gpu, op, table key)`` no shard holds yet, every candidate that survives the kernel's ``prune`` is launched once (a compile error, out of
resources or any exception: recorded as failed), its outputs compared with the first candidate's (the decorator default when it runs;
every tensor argument after the launch except the ones the kernel declares ``scratch`` (``@configured(..., scratch=...)``: per-block
partials / split slots whose layout follows the config), ``max|x - ref| / max|ref|`` within 2e-2 for 16-bit and 1e-3 for 32-bit
tensors (equal values, infinities included, and NaN against NaN do not differ), integers exact: a config that computes something
else is recorded failed as ``mismatch``), then timed with ``triton.testing.do_bench`` (L2 flushed,
warmup 25 ms, rep 100 ms, median; the kernel's ``reset_to_zero`` / ``restore_value`` honoured, ``_launch_config``'s own timer); the
ranking goes to the shard at once and the launch proceeds with the winner. A native ``choose(..., run=)`` site is tuned the same way
over its candidates (dims key, exact); a candidate whose extension is not prebuilt is skipped (``not prebuilt``: nothing is compiled;
``_prebuild_cuda`` builds every candidate of every site).
Nothing is timed under CUDA-graph capture or a torch.compile trace. The table key is the one ``_launch_config`` looks up
(``<dtypes>|<shape>[|<flags>]``): the packed ``shape_key`` is decoded by recording what ``_shape_key.pack`` / ``*_key`` were called with.

**Work units, resume, parallel.** A unit is one case (all its engine backends): ``r<row>-<n>`` (the n-th case of requirements data row
<row>). A process claims a unit by creating ``<run>/claims/<unit>.json`` exclusively and ends it with ``<run>/done/<unit>.json``. A claim
whose process is gone (same job: pid dead; other job: not in ``squeue``) is taken over; a unit claimed three times is marked failed. Keys
already in any shard are never re-timed (shards are re-read before every unit). Shards: ``<run>/shards/<job>_<host>_<pid>.jsonl``,
one JSON per line: ``{"t": "try", ...}`` before each candidate (a process that dies inside a candidate leaves it last: the next attempt
skips that candidate as ``crashed``), ``{"t": "key", "gpu", "device", "family", "op", "key", "ranked": [{config, "ms"}, ...],
"failed": [{config, "reason"}], "candidates", "unit", "secs"}`` per tuned key, ``{"t": "end" | "killed"}``. A process starts no unit
after ``--deadline`` and abandons the current one at ``--hard-deadline`` (claim released, keys kept); a CUDA error that poisons the context
exits with code 3 (the job script restarts the process). Parallel: one 1-GPU job per GPU, all on one ``--run`` directory.

**Merge** (``--merge``): every ``key`` record of the shards -> for each family ``kernels/<family>/configs/<gpu>.json``: the top
:data:`TOP_K` per key (``ms`` kept), re-tuned keys replaced (an equal key spelled with the other bucket name, ``M`` / ``L``, keeps the
table's spelling), the others kept. The file's integer ``version`` (an old file without one counts as 1; a new file starts at 1) is
incremented by one when, and only when, the merge changed its parsed content. Prints per family: keys added / changed / kept, old -> new
version.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

KERNELS = Path(__file__).resolve().parent
REPO = KERNELS.parents[2]
#: Human-managed version of how the pretuner ranks configs (candidates rule, timer, verification): bump it when a change can make it
#: pick other configs, not for refactors.
PRETUNE_VERSION = "1"
DEFAULT_RUN = Path.home() / "miniworld-engine-scratch" / "redesign" / "pretune" / "run"
DEFAULT_CONFIG = KERNELS.parent / "measure" / "configs" / "speed.yaml"
MAX_CANDIDATES = 64
GRID_SLICE_BELOW = 8
TOP_K = 4
ENGINE_BACKENDS = ("cuda", "triton")
TOLERANCE = {16: 2e-2, 32: 1e-3, 64: 1e-6}
#: Above this many bytes of tensor arguments the outputs are not compared (the reference copy would not fit comfortably).
VERIFY_MAX_BYTES = 8 << 30
#: Threads that compile a key's candidates before they are launched one by one (a job has 8 CPUs).
PRECOMPILE_THREADS = 8
_META = ("num_warps", "num_stages", "maxnreg")


def log(msg: str) -> None:
    print(f"[pretune {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===================================================================================================================== candidates
def _value(raw: str):
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return {"True": True, "False": False, "None": None}.get(raw, raw)


def _cfg_dict(kwargs: dict, num_warps, num_stages, maxnreg=None) -> dict:
    out = {"kwargs": dict(kwargs), "num_warps": int(num_warps), "num_stages": int(num_stages)}
    if maxnreg is not None:
        out["maxnreg"] = int(maxnreg)
    return out


def _from_triton(c) -> dict:
    return _cfg_dict(c.kwargs, c.num_warps, c.num_stages, getattr(c, "maxnreg", None))


def _ident(c: dict) -> str:
    return json.dumps({k: v for k, v in c.items() if k not in ("ms", "reason")}, sort_keys=True)


def _git_show(path: str) -> str | None:
    r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _old_csv(op: str, kind: str) -> list[dict] | None:
    text = _git_show(f"src/miniworld_engine/autotune/configs/{kind}/{op}.csv")
    return None if text is None else list(csv.DictReader(io.StringIO(text)))


def _pow2_neighbours(v: int) -> list[int]:
    return [x for x in (v // 2, v * 2) if x >= 1 and x != v]


def _generated(defaults: list[dict]) -> list[dict]:
    out: list[dict] = []
    for d in defaults:
        tiles = [k for k, v in d["kwargs"].items() if k.startswith(("BLOCK", "GROUP")) and isinstance(v, int) and v >= 1]
        w, s = d["num_warps"], d["num_stages"]
        for k in tiles:
            out.extend(_cfg_dict({**d["kwargs"], k: v}, w, s, d.get("maxnreg")) for v in _pow2_neighbours(d["kwargs"][k]))
        out.extend(_cfg_dict(d["kwargs"], w2, s, d.get("maxnreg")) for w2 in _pow2_neighbours(w) if w2 <= 16)
        out.extend(_cfg_dict(d["kwargs"], w, s2, d.get("maxnreg")) for s2 in (s - 1, s + 1) if s2 >= 1)
        choices = [[d["kwargs"][k], *_pow2_neighbours(d["kwargs"][k])] for k in tiles]
        for combo in itertools.product(*choices):
            out.extend(_cfg_dict({**d["kwargs"], **dict(zip(tiles, combo, strict=True))}, w2, s, d.get("maxnreg"))
                       for w2 in (w, *_pow2_neighbours(w)) if w2 <= 16)
    return out


def _csv_configs(rows: list[dict], tile_names: set[str]) -> list[dict] | None:
    if not rows:
        return None
    cols = set(rows[0]) - set(_META)
    if cols != tile_names:
        return None
    out = []
    for r in rows:
        v = {k: _value(x) for k, x in r.items()}
        out.append(_cfg_dict({k: v[k] for k in sorted(cols)}, v.get("num_warps", 4), v.get("num_stages", 2), v.get("maxnreg")))
    return out


def _grid_slice(base: list[dict], grid_rows: list[dict] | None) -> list[dict]:
    axes = {r["axis"]: [_value(x) for x in r["values"].split()] for r in grid_rows or []}
    warps, stages = axes.get("num_warps", []), axes.get("num_stages", [])
    out = []
    for c in base:
        combos = sorted(itertools.product(warps or [c["num_warps"]], stages or [c["num_stages"]]),
                        key=lambda ws: (abs(math.log2(ws[0] / c["num_warps"])) + abs(ws[1] - c["num_stages"]), ws))
        out += [_cfg_dict(c["kwargs"], w, s, c.get("maxnreg")) for w, s in combos]
    return out


def configured_kernels() -> tuple[dict, list[str]]:
    """op -> its ``Configured`` (every kernel module under kernels/ imported), and the modules that failed to import."""
    import importlib

    from miniworld_engine.kernels._launch_config import Configured

    found, failed = {}, []
    for path in sorted(KERNELS.rglob("*.py")):
        rel = path.relative_to(KERNELS.parent.parent).with_suffix("")
        if path.parent == KERNELS or "__pycache__" in rel.parts or "gen" in rel.parts:
            continue
        name = ".".join(rel.parts).removesuffix(".__init__")
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # reported
            failed.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        for v in vars(mod).values():
            if isinstance(v, Configured):
                found.setdefault(v.op, v)
    return found, failed


def _defaults(k) -> list[dict]:
    return [_from_triton(c) for c in k.default]


def import_candidates() -> int:
    """Write every family's ``configs/candidates.json`` (see the module docstring)."""
    from miniworld_engine.kernels._launch_config import table

    ops, failed = configured_kernels()
    for f in failed:
        log(f"import failed: {f}")
    by_family: dict[str, dict] = {}
    for op, k in sorted(ops.items()):
        defaults = _defaults(k)
        tiles = set(defaults[0]["kwargs"])
        old = _csv_configs(_old_csv(op, "default") or [], tiles)
        how = "csv"
        if old is None:
            how = "generated"
            extra = _generated(defaults)
        else:
            extra = old
            if len(old) <= GRID_SLICE_BELOW:
                how = "csv+grid"
                extra = old + _grid_slice(old, _old_csv(op, "grid"))
        measured = [c for cfgs in table(k.family, "h100_sm90").get(op, {}).values() for c in cfgs]
        measured = [_cfg_dict(c["kwargs"], c.get("num_warps", 4), c.get("num_stages", 2), c.get("maxnreg")) for c in measured]
        out, seen = [], set()
        for c in defaults + extra[: max(0, MAX_CANDIDATES - len(defaults))] + measured:
            if set(c["kwargs"]) == tiles and _ident(c) not in seen and (len(out) < MAX_CANDIDATES or c in defaults):
                seen.add(_ident(c))
                out.append(c)
        by_family.setdefault(k.family, {})[op] = out
        log(f"{k.family:24s} {op:52s} {len(out):3d} candidates ({how}; {len(defaults)} default, {len(measured)} from the table)")
    for family, ops_ in sorted(by_family.items()):
        path = KERNELS / family / "configs" / "candidates.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["{"]
        for i, (op, cfgs) in enumerate(sorted(ops_.items())):
            body = ",\n".join("  " + json.dumps(c) for c in cfgs)
            lines.append(f" {json.dumps(op)}: [\n{body}\n ]" + ("," if i < len(ops_) - 1 else ""))
        lines.append("}")
        path.write_text("\n".join(lines) + "\n")
        log(f"wrote {path.relative_to(REPO)}: {len(ops_)} ops, {sum(map(len, ops_.values()))} configs")
    log(f"{len(ops)} configured ops, {len(failed)} modules failed to import")
    return 0


def candidates_of(k) -> list[dict]:
    """The decorator default(s), then the family's candidates.json list for the op (deduplicated)."""
    path = KERNELS / k.family / "configs" / "candidates.json"
    listed = json.loads(path.read_text()).get(k.op, []) if path.is_file() else []
    out, seen = [], set()
    for c in _defaults(k) + listed:
        if _ident(c) not in seen:
            seen.add(_ident(c))
            out.append(c)
    return out


# ======================================================================================================== shape-key labels
_AXES: dict[int, tuple[int, dict]] = {}      # packed shape_key -> (bucket, width axes)
_BASE_NAME: dict[int, str] = {}              # bucket -> "L" (token_key / atom_key) or "M" (both_key)


def install_shape_recorder() -> None:
    """Wrap ``_shape_key``'s pack / token_key / atom_key / both_key so a packed key can be spelled as the table spells it."""
    import sys as _sys

    from miniworld_engine.kernels import _shape_key as sk

    if getattr(sk, "_pretune_recording", False):
        return
    orig = {n: getattr(sk, n) for n in ("pack", "token_key", "atom_key", "both_key")}

    def pack(base, **axes):
        v = orig["pack"](base, **axes)
        if axes:
            _AXES[v] = (int(base), {k: int(x) for k, x in axes.items()})
        return v

    def keyed(name, letter):
        f = orig[name]

        def g(n, **axes):
            v = f(n, **axes)
            _BASE_NAME[_AXES[v][0] if axes else v] = letter
            return v
        g.__name__ = name
        return g

    new = {"pack": pack, "token_key": keyed("token_key", "L"), "atom_key": keyed("atom_key", "L"), "both_key": keyed("both_key", "M")}
    for mod in list(_sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("miniworld_engine"):
            for n, f in orig.items():
                if getattr(mod, n, None) is f:
                    setattr(mod, n, new[n])
    sk._pretune_recording = True


def table_key(key: tuple) -> str | None:
    """``Configured.run``'s lookup key -> the table's key string (None: the packed shape could not be decoded)."""
    from miniworld_engine.kernels._launch_config import _parse, label
    from miniworld_engine.kernels._shape_key import BASE_NAMES, pack

    dtype, sk, flags = key
    shape = ""
    if sk is not None:
        sk = int(sk)
        if sk in _AXES:
            base, axes = _AXES[sk]
        elif sk < 1 << 24:
            base, axes = sk, {}
        else:
            return None
        name = _BASE_NAME.get(base, "M")
        if name in axes:
            name = next((n for n in BASE_NAMES if n not in axes), None)
            if name is None:
                return None
        shape = label({name: base, **axes})
    text = f"{dtype}|{shape}" + (f"|{label(dict(flags))}" if flags else "")
    # the round trip _launch_config does: the spelled key must index the same launch
    _, fields, _ = _parse(text)
    fields = dict(fields)
    b = next((fields.pop(n) for n in BASE_NAMES if n in fields), None)
    if (None if b is None else pack(b, **fields)) != (None if sk is None else sk):
        return None
    return text


# ============================================================================================================== ownership
_HOST, _PID, _JOB = socket.gethostname().split(".")[0], os.getpid(), os.environ.get("SLURM_JOB_ID", "nojob")
_SQUEUE: dict[str, tuple[float, bool]] = {}


def alive(job: str, host: str, pid: int) -> bool:
    if job == _JOB and host == _HOST:
        if pid == _PID:
            return True
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    if job == "nojob":
        return host != _HOST  # another host's interactive process: assume alive
    hit = _SQUEUE.get(job)
    if hit is None or time.time() - hit[0] > 30:
        try:
            r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True, timeout=30)
            state = r.stdout.strip()
            hit = (time.time(), state in ("RUNNING", "COMPLETING", "PENDING", "CONFIGURING"))
        except Exception:  # no squeue: assume alive
            hit = (time.time(), True)
        _SQUEUE[job] = hit
    return hit[1]


def _parse_owner(stem: str) -> tuple[str, str, int]:
    job, host, pid = stem.rsplit("_", 2)
    return job, host, int(pid)


# ===================================================================================================================== the hook
class _Deadline(BaseException):
    """The hard deadline passed (or SIGTERM): abandon the unit, keep what is tuned."""


def _healthy() -> bool:
    import torch
    try:
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ''}"


class NotPrebuilt(RuntimeError):
    pass


@contextlib.contextmanager
def _no_build():
    """Any compile of a CUDA artifact raises NotPrebuilt (pretuning a native candidate never builds it)."""
    from miniworld_engine.kernels import _nvcc

    orig = _nvcc.building

    @contextlib.contextmanager
    def refuse(name, kind):
        raise NotPrebuilt(f"{name} ({kind}) is not prebuilt")
        yield  # pragma: no cover

    _nvcc.building = refuse
    try:
        yield
    finally:
        _nvcc.building = orig


def _floats_of(tree) -> list:
    import torch
    from torch.utils import _pytree

    return [t for t in _pytree.tree_leaves(tree) if isinstance(t, torch.Tensor)]


def _mismatch(live: dict, ref: dict) -> str | None:
    import torch

    for n, r in ref.items():
        t = live[n]
        if t.shape != r.shape:
            return f"{n}: shape {tuple(t.shape)} vs {tuple(r.shape)}"
        if t.is_floating_point():
            a, b = t.float(), r.float()
            # equal values (equal infinities too: inf - inf is NaN) and NaN against NaN do not differ
            same = (a == b) | (torch.isnan(a) & torch.isnan(b))
            d = torch.where(same, torch.zeros_like(a), (a - b).abs().nan_to_num(nan=math.inf))
            scale = b.abs().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).max().item() if b.numel() else 0.0
            err = d.max().item() / scale if scale > 0 else (d.max().item() if d.numel() else 0.0)
            tol = TOLERANCE.get(t.element_size() * 8, 1e-3)
            if not err <= tol:
                return f"{n}: max|x - ref| / max|ref| = {err:.3g} > {tol:g}"
        elif not torch.equal(t, r):
            return f"{n}: integer values differ"
    return None


class Pretuner:
    """Installed as ``_launch_config.PRETUNE[0]``: tunes every (gpu, op, key) not in the shards at its first launch."""

    def __init__(self, run: Path, *, hard_deadline: float | None):
        import torch

        self.run, self.hard_deadline = run, hard_deadline
        (run / "shards").mkdir(parents=True, exist_ok=True)
        self.shard_path = run / "shards" / f"{_JOB}_{_HOST}_{_PID}.jsonl"
        self.shard = open(self.shard_path, "a")  # noqa: SIM115 -- kept open for the run, closed by close()
        self.device = torch.cuda.get_device_name(0)
        self.results: dict[tuple, dict] = {}     # (gpu, op, key) -> key record
        self.crash_lines: dict[tuple, set] = {}  # (gpu, op, key) -> candidate idents recorded as crashed
        self.last: dict[str, dict | None] = {}   # shard file -> its last "try" record if nothing followed it
        self.ended: dict[str, bool] = {}
        self.offsets: dict[str, int] = {}
        self.drops: dict[tuple, float] = {}       # (gpu, op, key) -> time of its latest drop (records before it are superseded)
        self.unit = None
        self.new_keys = 0
        self.reload()

    # ---------------------------------------------------------------------------------------------------------------- shards
    def write(self, rec: dict) -> None:
        self.shard.write(json.dumps(rec, default=str) + "\n")
        self.shard.flush()

    def reload(self) -> None:
        for path in sorted((self.run / "shards").glob("*.jsonl")):
            p = str(path)
            off = self.offsets.get(p, 0)
            with open(path, "rb") as fh:
                fh.seek(off)
                data = fh.read()
            end = data.rfind(b"\n") + 1
            self.offsets[p] = off + end
            for line in data[:end].splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self._take(p, rec)

    def _take(self, p: str, rec: dict) -> None:
        t = rec.get("t")
        if t == "try":
            self.last[p] = rec
            return
        self.last[p] = None
        if t == "key":
            ident = (rec["gpu"], rec["op"], rec["key"])
            if rec.get("time", 0) > self.drops.get(ident, -1):
                self.results[ident] = rec
        elif t == "drop":
            ident = (rec["gpu"], rec["op"], rec["key"])
            self.drops[ident] = max(self.drops.get(ident, -1), rec["time"])
            if ident in self.results and self.results[ident].get("time", 0) < rec["time"]:
                del self.results[ident]
        elif t == "crash":
            self.crash_lines.setdefault((rec["gpu"], rec["op"], rec["key"]), set()).add(rec["cand"])
        elif t in ("end", "killed"):
            self.ended[p] = True

    def crashed(self, ident: tuple) -> set:
        out = set(self.crash_lines.get(ident, ()))
        for p, rec in self.last.items():
            if rec is None or p == str(self.shard_path) or (rec["gpu"], rec["op"], rec["key"]) != ident or self.ended.get(p):
                continue
            if not alive(*_parse_owner(Path(p).stem)):
                out.add(rec["cand"])
        return out

    # ------------------------------------------------------------------------------------------------------------- common
    def _skip(self) -> bool:
        import torch
        return torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()

    def _check_deadline(self) -> None:
        if self.hard_deadline is not None and time.time() > self.hard_deadline:
            raise _Deadline("hard deadline")

    def _record(self, ident, family, ranked, failed, n, started, note=None) -> dict:
        first_ms = next((c["ms"] for c in ranked if c.get("_first")), math.nan)
        ranked = sorted(({k: v for k, v in c.items() if k != "_first"} for c in ranked), key=lambda c: c["ms"])
        rec = {"t": "key", "gpu": ident[0], "device": self.device, "family": family, "op": ident[1], "key": ident[2],
               "ranked": ranked, "failed": failed, "candidates": n, "unit": self.unit, "secs": round(time.time() - started, 2),
               "pretune_version": PRETUNE_VERSION, "time": time.time(), **({"note": note} if note else {})}
        self.write(rec)
        self.results[ident] = rec
        self.new_keys += 1
        best = ranked[0] if ranked else None
        reasons = sorted({f["reason"].split(":")[0] for f in failed})
        log(f"  + {ident[1]} {ident[2]}: {n} candidates, {len(failed)} failed{' ' + str(reasons) if reasons else ''}; "
            + (f"best {best['ms']:.4f} ms {json.dumps({k: v for k, v in best.items() if k != 'ms'})} (first {first_ms:.4f} ms)"
               if best else "NONE RAN") + f" [{rec['secs']:.1f} s]")
        return rec

    def _try(self, ident, cand_id: str) -> None:
        self._check_deadline()
        self.write({"t": "try", "gpu": ident[0], "op": ident[1], "key": ident[2], "cand": cand_id})

    def _fatal(self, ident, cand_id: str, exc: BaseException) -> None:
        self.write({"t": "crash", "gpu": ident[0], "op": ident[1], "key": ident[2], "cand": cand_id, "reason": _reason(exc)})
        log(f"CUDA context poisoned by {ident[1]} {ident[2]} candidate {cand_id}: {_reason(exc)}; exiting (3)")
        self.shard.close()
        os._exit(3)

    # ----------------------------------------------------------------------------------------------------- configured (Triton)
    def configured(self, k, gpu, key, named, kwargs, every, launch, time_one):
        import torch

        from miniworld_engine.kernels._launch_config import _cfg

        if self._skip():
            return None
        tk = table_key(key)
        if tk is None:
            log(f"  ! {k.op}: shape_key {key[1]} could not be spelled as a table key; normal lookup")
            return None
        ident = (gpu, k.op, tk)
        if ident in self.results:
            ranked = [_cfg(c) for c in self.results[ident]["ranked"]]
            if ranked and k.prune:
                try:
                    ranked = list(k.prune(ranked, named, **kwargs))
                except Exception:
                    ranked = []
            return ranked[0] if ranked else None

        started = time.time()
        dicts = candidates_of(k)
        cands = [_cfg(c) for c in dicts]
        if k.prune:
            try:
                kept = list(k.prune(cands, named, **kwargs))
            except Exception:  # one at a time
                kept = []
                for c in cands:
                    with contextlib.suppress(Exception):
                        kept += list(k.prune([c], named, **kwargs))
            cands = [c for c in cands if c in kept]
        crashed = self.crashed(ident)
        tensors = {n: v for n, v in every.items() if isinstance(v, torch.Tensor)}
        # Every tensor argument as the caller handed it (S): restored before each candidate's checked launch and before the
        # launch with the winner, whatever the kernel declares (an undeclared accumulate / in-place write would otherwise
        # compare a candidate against the previous one's leftovers). Too large to copy twice: no S, no comparison, the
        # declared reset_to_zero / restore_value only.
        verify = sum(t.numel() * t.element_size() for t in tensors.values()) <= VERIFY_MAX_BYTES
        torch.cuda.empty_cache()
        try:
            snap = {n: t.clone() for n, t in tensors.items()} if verify else None
        except torch.cuda.OutOfMemoryError:
            snap, verify = None, False
            torch.cuda.empty_cache()
        saved = {n: every[n].clone() for n in k.restore_value} if snap is None else {}

        def restore():
            if snap is not None:
                for n, t in snap.items():
                    tensors[n].copy_(t)
            else:
                for n in k.reset_to_zero:
                    every[n].zero_()
                for n, t in saved.items():
                    every[n].copy_(t)

        # compile every candidate first, in parallel threads (ptxas runs as a subprocess); errors surface again at the launch
        todo = [c for c in cands if _ident(_from_triton(c)) not in crashed]
        if PRECOMPILE_THREADS > 1 and len(todo) > 1:
            import concurrent.futures as cf

            def warm(c):
                with contextlib.suppress(Exception):  # the checked launch below records it
                    launch(c, compile_only=True)
            self._check_deadline()
            with cf.ThreadPoolExecutor(PRECOMPILE_THREADS) as ex:
                list(ex.map(warm, todo))
        ref, ranked, failed = None, [], []
        first = True
        for c in cands:
            d = _from_triton(c)
            cid = _ident(d)
            if cid in crashed:
                failed.append({**d, "reason": "crashed: the process died in this candidate"})
                continue
            self._try(ident, cid)
            try:
                restore()
                launch(c)
                torch.cuda.synchronize()
                bad = None
                if verify:
                    if ref is None:
                        # the declared scratch (config-dependent intermediates) is restored like every argument, not compared
                        ref = {n: t.clone() for n, t in tensors.items() if n not in k.scratch}
                    else:
                        bad = _mismatch(tensors, ref)
                if bad is not None:
                    failed.append({**d, "reason": f"mismatch: {bad}"})
                    continue
                torch.cuda.empty_cache()
                ms = time_one(c)
                ranked.append({**d, "ms": round(float(ms), 6), **({"_first": True} if first else {})})
                first = False
            except _Deadline:
                restore()
                raise
            except Exception as exc:  # recorded
                if not _healthy():
                    self._fatal(ident, cid, exc)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                failed.append({**d, "reason": _reason(exc)})
        ref = None
        restore()
        snap = saved = None
        torch.cuda.empty_cache()
        rec = self._record(ident, k.family, ranked, failed, len(cands), started,
                           note=None if verify else "outputs not compared (tensor arguments above VERIFY_MAX_BYTES)")
        return _cfg(rec["ranked"][0]) if rec["ranked"] else None

    # ---------------------------------------------------------------------------------------------------------- choose (native)
    def choose(self, family, op, gpu, dtype, dims, candidates, run):
        import torch

        from miniworld_engine.kernels._launch_config import _bench, label

        if self._skip():
            return None
        tk = f"{dtype}|{label(dict(dims))}"
        ident = (gpu, op, tk)
        if ident in self.results:
            live = [c["kwargs"] for c in self.results[ident]["ranked"] if c["kwargs"] in candidates]
            return live[0] if live else None
        started = time.time()
        crashed = self.crashed(ident)
        cands, seen = [], set()
        for c in candidates:
            if _ident({"kwargs": c}) not in seen:
                seen.add(_ident({"kwargs": c}))
                cands.append(dict(c))
        ref, ranked, failed, first = None, [], [], True
        for c in cands:
            d = {"kwargs": c}
            cid = _ident(d)
            if cid in crashed:
                failed.append({**d, "reason": "crashed: the process died in this candidate"})
                continue
            self._try(ident, cid)
            try:
                with _no_build():
                    out = run(c)
                    torch.cuda.synchronize()
                    outs = _floats_of(out)
                    if ref is None:
                        ref = {i: t.clone() for i, t in enumerate(outs)}
                        bad = None
                    else:
                        bad = _mismatch(dict(enumerate(outs)), ref) if len(outs) == len(ref) else "output count differs"
                    del out, outs
                    if bad is not None:
                        failed.append({**d, "reason": f"mismatch: {bad}"})
                        continue
                    ms = _bench(lambda c=c: run(c))
                ranked.append({**d, "ms": round(float(ms), 6), **({"_first": True} if first else {})})
                first = False
            except _Deadline:
                raise
            except NotPrebuilt as exc:
                failed.append({**d, "reason": f"not prebuilt: {exc}"})
            except Exception as exc:  # recorded
                if not _healthy():
                    self._fatal(ident, cid, exc)
                failed.append({**d, "reason": _reason(exc)})
        ref = None
        rec = self._record(ident, family, ranked, failed, len(cands), started)
        return rec["ranked"][0]["kwargs"] if rec["ranked"] else None


# ===================================================================================================================== units
def units(cfg) -> list[tuple[str, object]]:
    from miniworld_engine.measure import config

    out, per_row = [], {}
    for case in config.cases(cfg):
        n = per_row.get(case.row, 0)
        per_row[case.row] = n + 1
        out.append((f"r{case.row:03d}-{n:03d}", case))
    return out


def _excl_write(path: Path, rec: dict) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh)
    return True


def claim(run: Path, uid: str, label: str) -> tuple[bool, int]:
    """(claimed, attempt). A dead owner's claim is taken over (rename first: only one process wins it)."""
    path = run / "claims" / f"{uid}.json"
    me = {"job": _JOB, "host": _HOST, "pid": _PID, "time": time.time(), "label": label, "attempt": 1}
    if _excl_write(path, me):
        return True, 1
    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, 0
    if alive(str(old["job"]), old["host"], int(old["pid"])):
        return False, 0
    grave = path.with_suffix(f".dead-{_PID}")
    try:
        os.rename(path, grave)
    except OSError:
        return False, 0
    grave.unlink(missing_ok=True)
    me["attempt"] = int(old.get("attempt", 1)) + 1
    return (_excl_write(path, me), me["attempt"])


def run_unit(hook: Pretuner, cfg, case, *, device) -> dict:
    import copy

    import torch

    from miniworld_engine.measure import config
    from miniworld_engine.measure.speed import make_step
    from miniworld_engine.modules.primitives import BackendUnsupported

    status = {}
    backends = [b for b in config.backends_of(cfg, case) if b in ENGINE_BACKENDS]
    if not backends:
        return {"status": "no engine backend"}
    base = None
    try:
        base = config.build(case, device=device)
        for backend in backends:
            before = hook.new_keys
            t0 = time.time()
            module = inputs = step = None
            try:
                module = copy.deepcopy(base)
                inputs = config.inputs(case, module, device=device)
                step = make_step(module, inputs, backend=backend, train=case.mode == "train", autocast=config.autocast(case),
                                 seed=config.seed(case), compile_mode=None)
                step()
                torch.cuda.synchronize()
                status[backend] = "ok"
            except BackendUnsupported as exc:
                status[backend] = f"unsupported: {str(exc)[:120]}"
            except torch.cuda.OutOfMemoryError:
                status[backend] = "skipped: out of memory"
            except _Deadline:
                raise
            except Exception as exc:  # recorded
                status[backend] = f"failed: {_reason(exc)}"
                if os.environ.get("PRETUNE_TRACEBACK"):
                    traceback.print_exc()
                if not _healthy():
                    log(f"CUDA context poisoned in the step of {case.label} {backend}; exiting (3)")
                    os._exit(3)
            finally:
                module = inputs = step = None
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            status[backend] += f" ({hook.new_keys - before} new keys, {time.time() - t0:.1f} s)"
    except torch.cuda.OutOfMemoryError:
        status["build"] = "skipped: out of memory"
    except _Deadline:
        raise
    except Exception as exc:  # recorded
        status["build"] = f"failed: {_reason(exc)}"
    finally:
        base = None
        torch.cuda.empty_cache()
    return status


def tune(args) -> int:
    install_shape_recorder()   # before any kernel module binds _shape_key's functions
    import torch

    from miniworld_engine.kernels import _launch_config
    from miniworld_engine.measure import config

    run = Path(args.run)
    for d in ("claims", "done", "shards"):
        (run / d).mkdir(parents=True, exist_ok=True)
    cfg = config.load(args.config)
    cfg = config.replace(cfg, backends=list(ENGINE_BACKENDS), modules=args.modules or None, rows=args.rows or None,
                         dtypes=args.dtypes or None, modes=args.modes or None)
    work = units(cfg)
    if args.list:
        for uid, case in work:
            print(uid, case.label)
        log(f"{len(work)} units")
        return 0
    dev = torch.device("cuda")
    hook = Pretuner(run, hard_deadline=args.hard_deadline)

    def on_term(signum, frame):
        hook.write({"t": "killed", "signal": signum})
        raise _Deadline("SIGTERM")
    signal.signal(signal.SIGTERM, on_term)
    _launch_config.PRETUNE[0] = hook
    log(f"run {run}; {len(work)} units; device {hook.device}; job {_JOB} pid {_PID}; shard {hook.shard_path.name}; "
        f"{len(hook.results)} keys already tuned")
    started, ran = time.time(), 0
    code = 0
    try:
        for uid, case in work:
            if (run / "done" / f"{uid}.json").exists():
                continue
            if args.deadline is not None and time.time() > args.deadline:
                log("deadline: no new unit")
                code = 4
                break
            ok, attempt = claim(run, uid, case.label)
            if not ok:
                continue
            if attempt > 3:
                _excl_write(run / "done" / f"{uid}.json", {"label": case.label, "status": {"unit": "failed: crashed 3 times"}})
                log(f"{uid} {case.label}: crashed 3 times; marked failed")
                continue
            hook.reload()
            hook.unit = uid
            t0, before = time.time(), hook.new_keys
            try:
                status = run_unit(hook, cfg, case, device=dev)
            except _Deadline as exc:
                (run / "claims" / f"{uid}.json").unlink(missing_ok=True)
                log(f"{uid} {case.label}: abandoned ({exc}); {hook.new_keys - before} keys kept")
                code = 4
                break
            _excl_write(run / "done" / f"{uid}.json", {"label": case.label, "status": status, "secs": round(time.time() - t0, 1),
                                                       "new_keys": hook.new_keys - before, "job": _JOB})
            ran += 1
            log(f"{uid} {case.label}: {hook.new_keys - before} new keys in {time.time() - t0:.1f} s | "
                + "; ".join(f"{b}: {s}" for b, s in status.items()))
    finally:
        _launch_config.PRETUNE[0] = None
        hook.write({"t": "end"})
        hook.shard.close()
    log(f"{ran} units, {hook.new_keys} new keys in {time.time() - started:.0f} s; exit {code}")
    return code


# ===================================================================================================================== status
def _records(runs: list[Path]) -> list[dict]:
    out = []
    for run in runs:
        for path in sorted((run / "shards").glob("*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("t") in ("key", "drop"):
                    out.append(rec)
    drops: dict[tuple, float] = {}
    for r in out:
        if r["t"] == "drop":
            ident = (r["gpu"], r["op"], r["key"])
            drops[ident] = max(drops.get(ident, -1), r["time"])
    # a key record older than a drop of its key is superseded
    return [r for r in out if r["t"] == "key" and r.get("time", 0) > drops.get((r["gpu"], r["op"], r["key"]), -1)]


def drop(args) -> int:
    """Supersede key records (every key with a ``mismatch`` failure, and ``--drop-key OP KEY`` pairs) and reopen their units, so
    the next tune run times them again. Appends ``{"t": "drop"}`` lines to ``shards/drop_cpu_<pid>.jsonl``."""
    run = Path(args.run)
    recs = _records([run])
    want = {tuple(x) for x in args.drop_key or ()}
    hit = [r for r in recs if (args.drop_mismatch and any(f["reason"].startswith("mismatch") for f in r["failed"]))
           or (r["op"], r["key"]) in want]
    now = time.time()
    with open(run / "shards" / f"drop_cpu_{_PID}.jsonl", "a") as fh:
        fh.writelines(json.dumps({"t": "drop", "gpu": r["gpu"], "op": r["op"], "key": r["key"], "time": now}) + "\n" for r in hit)
    reopened = set()
    for r in hit:
        uid = r.get("unit")
        if uid and (run / "done" / f"{uid}.json").exists():
            (run / "done" / f"{uid}.json").unlink()
            (run / "claims" / f"{uid}.json").unlink(missing_ok=True)
            reopened.add(uid)
        log(f"drop {r['op']} {r['key']} (unit {uid}): "
            + "; ".join(sorted({f["reason"][:80] for f in r["failed"] if f["reason"].startswith(("mismatch", "OutOfMemory"))}))[:300])
    missing = want - {(r["op"], r["key"]) for r in hit}
    for m in sorted(missing):
        log(f"no record for {m}")
    log(f"{len(hit)} key records dropped, {len(reopened)} units reopened")
    return 0


def status(args) -> int:
    run = Path(args.run)
    done = [json.loads(p.read_text()) for p in sorted((run / "done").glob("*.json"))]
    claims = {p.stem for p in (run / "claims").glob("*.json")}
    failed = [d for d in done if any(str(s).startswith("failed") for s in d["status"].values())]
    recs = _records([run])
    keys = {(r["gpu"], r["op"], r["key"]) for r in recs}
    log(f"{run}: {len(done)} units done ({len(failed)} with a failure), {len(claims - {p.stem for p in (run / 'done').glob('*.json')})} "
        f"claimed and open; {len(keys)} keys tuned ({len(recs)} records), "
        f"{sum(not r['ranked'] for r in recs)} with no candidate that ran")
    for d in failed:
        log(f"  failed: {d['label']}: {d['status']}")
    return 0


# ===================================================================================================================== merge
def _canon(key: str) -> tuple:
    from miniworld_engine.kernels._launch_config import _parse
    from miniworld_engine.kernels._shape_key import BASE_NAMES

    dtype, shape, flags = _parse(key)
    shape = dict(shape)
    base = next((shape.pop(n) for n in BASE_NAMES if n in shape), None)
    return dtype, base, tuple(sorted(shape.items())), flags


def _entry(c: dict) -> dict:
    out = {"kwargs": c["kwargs"]}
    for k in _META:
        if c.get(k) is not None:
            out[k] = c[k]
    out["ms"] = c["ms"]
    return out


def _dump_table(data: dict) -> str:
    lines = ["{"]
    lines.extend(f" {json.dumps(k)}: {json.dumps(data[k])}," for k in ("gpu", "source", "version"))
    lines.append(' "ops": {')
    ops = data["ops"]
    for i, op in enumerate(sorted(ops)):
        lines.append(f"  {json.dumps(op)}: {{")
        keys = sorted(ops[op])
        for j, key in enumerate(keys):
            body = ",\n".join("    " + json.dumps(c) for c in ops[op][key])
            lines.append(f"   {json.dumps(key)}: [\n{body}\n   ]" + ("," if j < len(keys) - 1 else ""))
        lines.append("  }" + ("," if i < len(ops) - 1 else ""))
    lines.append(" }")
    lines.append("}")
    return "\n".join(lines) + "\n"


SOURCE = ("Pretuned by kernels/_pretune.py (PRETUNE_VERSION {v}): every candidate timed at the launches the requirements reach "
          "(triton.testing.do_bench, L2 flushed, median), top {k} kept, fastest first. Keys not re-tuned: converted from the old "
          "autotune/data/<op>/*sm90*.json (git HEAD). Keys: '<dtypes>|<shape bucket>[|<flags>]'. Shape: M = row bucket, L = length "
          "bucket, other names = exact width axes.")


def merge(args) -> int:
    runs = [Path(r) for r in (args.run_list or [str(DEFAULT_RUN)])]
    recs = _records(runs)
    best: dict[tuple, dict] = {}
    for r in recs:  # one record per (gpu, family, op, key): the one that timed most candidates, then the latest
        ident = (r["gpu"], r["family"], r["op"], r["key"])
        old = best.get(ident)
        if old is None or len(r["ranked"]) >= len(old["ranked"]):
            best[ident] = r
    empty = sorted(f"{i[2]} {i[3]}" for i, r in best.items() if not r["ranked"])
    groups: dict[tuple, list[dict]] = {}
    for (gpu, family, _, _), r in best.items():
        if r["ranked"]:
            groups.setdefault((gpu, family), []).append(r)
    log(f"{len(recs)} key records from {len(runs)} run(s): {len(best)} keys, {len(empty)} with no candidate that ran")
    total = {"added": 0, "changed": 0, "same": 0, "kept": 0}
    for (gpu, family), rs in sorted(groups.items()):
        path = KERNELS / family / "configs" / f"{gpu}.json"
        if path.is_file():
            old = json.loads(path.read_text())
            version = int(old.get("version") or 1)
        else:
            old, version = None, 0
        ops = json.loads(json.dumps((old or {}).get("ops", {})))
        added = changed = same = 0
        touched = set()
        for r in sorted(rs, key=lambda r: (r["op"], r["key"])):
            table = ops.setdefault(r["op"], {})
            spelled = {_canon(k): k for k in table}
            key = spelled.get(_canon(r["key"]), r["key"])
            new = [_entry(c) for c in r["ranked"][:TOP_K]]
            if key not in table:
                added += 1
            elif table[key] != new:
                changed += 1
            else:
                same += 1
            table[key] = new
            touched.add((r["op"], key))
        kept = sum(len(v) for v in ops.values()) - len(touched)
        changed_file = old is None or ops != old.get("ops")
        new_version = version + 1 if changed_file else max(version, 1)
        data = {"gpu": (old or {}).get("gpu") or f"{rs[0]['device']} (sm{gpu.rsplit('_sm', 1)[1]})",
                "source": SOURCE.format(v=PRETUNE_VERSION, k=TOP_K) if changed_file else old.get("source", ""),
                "version": new_version, "ops": ops}
        log(f"{family:24s} {gpu}: {added:4d} added, {changed:4d} changed, {same:3d} re-tuned unchanged, {kept:4d} kept; "
            f"version {version if old is not None else '-'} -> {new_version}" + ("" if changed_file else " (content unchanged)"))
        for k, v in (("added", added), ("changed", changed), ("same", same), ("kept", kept)):
            total[k] += v
        if not args.dry_run and (changed_file or old.get("version") != new_version):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_dump_table(data))
    log(f"total: {total}" + (" (dry run: nothing written)" if args.dry_run else ""))
    for e in empty:
        log(f"  no candidate ran: {e}")
    return 0


# ===================================================================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m miniworld_engine.kernels._pretune", description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", action="store_true", help="write kernels/<family>/configs/candidates.json (CPU)")
    ap.add_argument("--merge", action="store_true", help="merge the shards into the tables (CPU)")
    ap.add_argument("--status", action="store_true", help="units and keys of a run")
    ap.add_argument("--drop-mismatch", action="store_true", help="supersede every key with a mismatch failure; reopen its unit")
    ap.add_argument("--drop-key", nargs=2, action="append", metavar=("OP", "KEY"), help="supersede this key; reopen its unit")
    ap.add_argument("--list", action="store_true", help="print the units; tune nothing")
    ap.add_argument("--dry-run", action="store_true", help="--merge: print, write nothing")
    ap.add_argument("--run", dest="run_list", action="append", default=None,
                    help=f"run directory (default {DEFAULT_RUN}); --merge takes several")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="measurement config whose cases are the work list")
    ap.add_argument("--modules", nargs="*", default=None)
    ap.add_argument("--rows", nargs="*", type=int, default=None)
    ap.add_argument("--dtypes", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: start no unit after it")
    ap.add_argument("--hard-deadline", type=float, default=None, help="epoch seconds: abandon the current unit after it")
    args = ap.parse_args(argv)
    args.run = (args.run_list or [str(DEFAULT_RUN)])[0]
    if args.candidates:
        return import_candidates()
    if args.merge:
        return merge(args)
    if args.drop_mismatch or args.drop_key:
        return drop(args)
    if args.status:
        return status(args)
    return tune(args)


if __name__ == "__main__":
    sys.exit(main())
