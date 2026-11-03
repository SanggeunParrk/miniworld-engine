"""The calls ``registry_module.csv`` declares: ONE expansion, shared by bundle coverage and the release verify.

``registry_module.csv`` is the contract of what the engine serves. A release must bundle the kernels of every call it
declares, and the release check must make exactly those calls under the installed wheels. Two hand-written lists (the
coverage cases and the test files a check happened to run) drifted apart and produced a loop of ``test fails -> add a
kernel -> rebuild the wheel -> verify``. This module is the single place that turns a registry row into calls:

* :func:`expand` -- every row of the CSV into its :class:`Call` s: ``length x impl x dtype x compute x mode x option``
  (options one at a time against the default, as ``autotune.derive.units`` does: each option selects among its own
  kernels, and only on the side of the pass where it acts -- ``builder.SWITCHES``). A (impl, dtype) the arch's rule files
  (``build/gpu_to_kernels``) refuse is NOT dropped silently: it is listed in the row's ``excluded`` with the rule's
  reason, and a row with no call left is a ``not an <arch> row`` with the registry text as evidence.
* :func:`kernel_units` -- the kernel-level rows of ``registry.csv`` (hand-CUDA kernels of the arch with a driver).
* :func:`run_call` -- one call, built the way ``autotune.builder`` builds the module (same ``Case`` factory, same inputs,
  same augmentation, same pins), but errors PROPAGATE (the builder prints ``skip`` and returns 0).
* a command line for the verify's worker processes::

      python -m miniworld_engine.build.registry_calls expand --arch sm90 [--json]
      python -m miniworld_engine.build.registry_calls run-calls --arch sm90 --row N --out results.jsonl [--start K]
      python -m miniworld_engine.build.registry_calls run-kernel --kernel NAME --out results.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

#: A CUDA error that leaves the process unusable (the context is poisoned): the caller restarts the process.
STICKY = re.compile(r"CUDA error|illegal memory access|device-side assert|an illegal instruction|unspecified launch failure",
                    re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class Call:
    """One declared call: a registry row at one (length, impl, dtype, compute, mode, option)."""

    row: int                     # 1-based data row of registry_module.csv
    module: str
    stream: str
    dim_index: int               # index among the rows of this module (the builder's ``Case.dims`` index)
    length: int
    impl: str
    dtype: str
    compute: str                 # "" = no compute_dtype argument
    mode: str                    # eval = forward under no_grad, train = forward + backward
    option: tuple[str, str] | None
    augmentation: int
    direction: str = ""          # "" or the module's other direction: outgoing / incoming (TriMul), starting / ending (TriAttn)

    @property
    def label(self) -> str:
        opt = f" {self.option[0]}={self.option[1]}" if self.option else ""
        core = f"->{self.compute}" if self.compute else ""
        way = f" {self.direction}" if self.direction else ""
        return f"r{self.row:03d} {self.module}[{self.impl}/{self.dtype}{core}] L={self.length} A={self.augmentation} {self.mode}{opt}{way}"

    def to_json(self) -> dict:
        return {**dataclasses.asdict(self), "option": list(self.option) if self.option else None}

    @staticmethod
    def from_json(data: dict) -> Call:
        data = dict(data)
        data["option"] = tuple(data["option"]) if data.get("option") else None
        return Call(**data)


@dataclasses.dataclass
class RowExpansion:
    """What one registry row expands to on one arch."""

    row: int
    module: str
    stream: str
    dims: dict
    registry_text: str           # the row as the CSV states it: impls, dtypes, modes, ...
    calls: list[Call]
    excluded: list[str]          # "impl/dtype: reason" for every combination the arch rules refuse
    augmentation: str = ""       # "<eval>|<train>": two rows may share module, stream and dims and differ only here

    @property
    def key(self) -> str:
        """The row's identity without its line number (``module/stream/dims/A=<eval>|<train>``): stable when other rows are added or
        removed."""
        dims = ";".join(f"{k}={v}" for k, v in self.dims.items())
        return f"{self.module}/{self.stream}/{dims}/A={self.augmentation}"

    @property
    def id(self) -> str:
        return f"r{self.row:03d} {self.key}"

    @property
    def text_digest(self) -> str:
        """Digest of what the row declares (lengths, impls, dtypes, modes, options, augmentation): an edited row is a new row."""
        import hashlib

        return hashlib.sha256(f"{self.key}|{self.registry_text}".encode()).hexdigest()[:16]

    @property
    def not_arch_reason(self) -> str:
        """Non-empty when the row yields no call on this arch ('not an <arch> row')."""
        if self.calls:
            return ""
        return f"registry says {self.registry_text}; " + ("; ".join(self.excluded) or "no combination declared")


@dataclasses.dataclass
class KernelUnit:
    """One row of ``registry.csv``: callable on this arch (``reason`` empty) or not an <arch> kernel (with the reason)."""

    kernel: str
    backend: str
    arch: str
    driver: str
    reason: str
    dtypes: str = ""


#: Modules the pairformer / trunk runs in BOTH directions with the same registry row: module -> (constructor flag, name of the non-default
#: direction). The builder's factories construct the default one (outgoing / starting); production also runs incoming / ending.
DIRECTIONS = {"triangle_multiplication": ("outgoing", "incoming"), "triangle_attention_heads": ("starting", "ending"),
              "triangle_attention_checkpoint": ("starting", "ending")}


def _sm(arch: str) -> str:
    return "sm_" + arch.removeprefix("sm_").removeprefix("sm")


def expand(arch: str = "sm90", rows=None) -> list[RowExpansion]:
    """Every row of ``registry_module.csv`` (or ``rows``) into its calls for ``arch``; one entry per row, none dropped."""
    from miniworld_engine import build as build_matrix
    from miniworld_engine.autotune.builder import switch_applies
    from miniworld_engine.autotune.module_registry import module_rows

    rows = module_rows() if rows is None else rows
    sm = _sm(arch)
    seen: dict[str, int] = {}
    out: list[RowExpansion] = []
    for number, row in enumerate(rows, 1):
        dim_index = seen.get(row.module, 0)
        seen[row.module] = dim_index + 1
        text = (f"lengths={'|'.join(map(str, row.lengths))} impls={'|'.join(row.impls)} dtypes={'|'.join(row.dtypes)} "
                f"modes={'|'.join(row.modes)} computes={'|'.join(row.computes) or '-'} "
                f"options={'|'.join(f'{k}={v}' for k, v in row.options) or '-'} "
                f"augmentation={'|'.join(str(row.augmentation(m)) for m in row.modes)}")
        excluded: list[str] = []
        allowed: set[tuple[str, str]] = set()
        for impl, dtype in itertools.product(row.impls, row.dtypes):
            ok, why = build_matrix.decide(sm, row.module, impl, dtype)
            if ok:
                allowed.add((impl, dtype))
            else:
                excluded.append(f"{impl}/{dtype}: {why}")
        calls: list[Call] = []
        for length, impl, dtype, mode in itertools.product(row.lengths, row.impls, row.dtypes, row.modes):
            if (impl, dtype) not in allowed:
                continue
            for compute in row.computes or ("",):
                for option in (None, *row.options):
                    if option is not None:
                        name = option[0]
                        if not switch_applies(name, mode, dtype):
                            continue   # the option acts on the other side of the pass (or on fp32 only): a duplicate of the default call
                        if name == "trimul_impl":
                            ok, why = build_matrix.decide(sm, "triangle_multiplication", option[1], dtype)
                            if not ok:
                                excluded.append(f"{name}={option[1]}/{dtype}: {why}")
                                continue
                    for direction in ("", *([DIRECTIONS[row.module][1]] if row.module in DIRECTIONS else [])):
                        calls.append(Call(number, row.module, row.stream, dim_index, length, impl, dtype, compute, mode,
                                          option, row.augmentation(mode), direction))
        out.append(RowExpansion(number, row.module, row.stream, dict(row.dims), text, calls, sorted(set(excluded)),
                                f"{row.augmentation('eval')}|{row.augmentation('train')}"))
    return out


def kernel_units(arch: str = "sm90") -> list[KernelUnit]:
    """The kernel-level rows of ``registry.csv`` for ``arch``: the developed hand-CUDA kernels with a driver are callable; every other
    row is listed with the reason it is not an ``arch`` kernel (Triton rows: the H100 / B200 path is hand-CUDA, Triton is the fallback of
    the other GPUs; no developed driver: an A/B alternative no module launches)."""
    import csv

    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry" / "registry.csv"
    out = []
    with reg.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["backend"] != "cuda":
                reason = f"backend={r['backend']}, tuned for {r['arch']}: the {arch} path is hand-CUDA (docs/gpus/h100/dispatch.md)"
            elif r["arch"] != arch:
                reason = f"hand-CUDA kernel of {r['arch']} (registry.csv arch column), not {arch}"
            elif r["developed"] != "yes":
                reason = f"developed={r['developed']}: an A/B alternative no module launches"
            elif not (r.get("driver") or "").strip():
                reason = "registry.csv names no driver for it"
            else:
                reason = ""
            out.append(KernelUnit(r["kernel"], r["backend"], r["arch"], r["driver"], reason, r["dtypes"]))
    return out


# ------------------------------------------------------------------------------------------------ running
def install_miss_recorder() -> list[str]:
    """Record every artifact the bundle lacked, even when the caller catches the ``PrebuiltMissing`` (some loaders fall back silently)."""
    from miniworld_engine.kernels import _nvcc

    misses: list[str] = []
    original = _nvcc._before_build

    def recording(kind: str, name: str, key: str) -> None:
        try:
            original(kind, name, key)
        except Exception:
            misses.append(name)
            raise

    _nvcc._before_build = recording

    resolved_original = _nvcc._resolved

    def resolving(kind: str, name: str, key: str, path, origin: str, *rest, **extra) -> None:
        RESOLVED.append((name, origin))
        resolved_original(kind, name, key, path, origin, *rest, **extra)

    _nvcc._resolved = resolving
    return misses


#: (artifact name, origin) of every artifact this process resolved (origin ``prebuilt`` = served by the bundle).
RESOLVED: list[tuple[str, str]] = []
_CASES: dict = {}


def _case(name: str):
    if not _CASES:
        from miniworld_engine.autotune.builder import cases

        _CASES.update({c.name: c for c in cases()})
    return _CASES[name]


def run_call(call: Call) -> None:
    """One call, built as ``autotune.builder`` builds the module; forward under no_grad (eval) or forward + backward (train).
    Every exception propagates."""
    import torch

    from miniworld_engine import settings
    from miniworld_engine.autotune.builder import SWITCH_SETTINGS

    case = _case(call.module)
    dims = dict(case.dims[call.dim_index])
    if call.direction:   # the module's other direction: the constructor argument is the flag
        dims[DIRECTIONS[call.module][0]] = False   # incoming / ending
    from miniworld_engine.autotune.module_registry import torch_dtypes

    dtype, act = torch_dtypes(call.dtype)          # parameters, activations (``master``: fp32 parameters, bf16 activations)
    train = call.mode == "train"
    p_drop, pins, torch_pin = 0.0, {}, contextlib.nullcontext()
    if call.option:
        name, value = call.option
        if name == "p_drop":
            p_drop = float(value)
        elif name in SWITCH_SETTINGS:
            field, parse = SWITCH_SETTINGS[name]
            pins[field] = parse(value)
        else:                                       # a torch-level switch: the fp32 matmul precision
            from miniworld_engine.autotune.builder import torch_switch

            torch_pin = torch_switch(name, value)
    if call.impl == "portable":   # an A/B label: the miniworld module under the portable family (builder.run_case)
        pins["force_portable_kernels"] = True
    previous = settings.configure(**pins) if pins else None
    try:
        with torch_pin:
            module = case.factory(dims, p_drop, call.impl, dtype)
            module.train(train)
            args = case.input_args(call.dim_index, call.length, act, train=train)
            kwargs = {"compute_dtype": getattr(torch, call.compute)} if call.compute else {}
            if train:
                args = tuple(a.detach().clone().requires_grad_(True) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
                out = module(*args, **kwargs)
                (out.float().sum() if torch.is_tensor(out) else out[0].float().sum()).backward()
            else:
                with torch.no_grad():
                    module(*args, **kwargs)
            torch.cuda.synchronize()
    finally:
        if previous is not None:
            settings.configure(**dataclasses.asdict(previous))
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def run_driver(unit: KernelUnit) -> None:
    """One kernel-level row: its registry driver, at the driver's own default shape (``autotune.builder._run_one_driver``)."""
    import importlib

    import torch

    # the drivers read MINIWORLD_DRIVER_DTYPE when their module is imported: an fp32-only row (a bf16 process launches the other precision's
    # kernels) sets it, every other row clears it, and the module is imported afresh (a coverage worker runs rows of both kinds)
    if unit.dtypes == "fp32":
        os.environ["MINIWORLD_DRIVER_DTYPE"] = "fp32"
    else:
        os.environ.pop("MINIWORLD_DRIVER_DTYPE", None)
    mod, _, fn = unit.driver.partition(":")
    sys.modules.pop(mod, None)
    getattr(importlib.import_module(mod), fn)()
    torch.cuda.synchronize()


def out_of_memory(exc: BaseException) -> bool:
    """An out-of-memory that is not ``torch.cuda.OutOfMemoryError``: a CUDA extension's allocation (``AcceleratorError: CUDA error: out
    of memory``) or a Triton launch's (``Triton Error [CUDA]: out of memory``). The same verdict: the shape did not fit, nothing was
    verified at it."""
    text = str(exc)
    return "CUDA error: out of memory" in text or "Triton Error [CUDA]: out of memory" in text


def classify(exc: BaseException) -> str:
    """missing: the bundle lacks an artifact. oom: the declared shape does not fit the card (nothing was verified at it). unbundled: the
    artifact compiles NVIDIA MathDx / cuBLASDx, which is never redistributed (it JIT-builds where used). error: anything else."""
    import torch

    from miniworld_engine.kernels import _nvcc, prebuilt

    if isinstance(exc, (prebuilt.PrebuiltMissing, _nvcc.CutlassMissing)):
        return "missing"
    if isinstance(exc, torch.cuda.OutOfMemoryError) or out_of_memory(exc):
        return "oom"
    if "cuBLASDx headers not found" in str(exc):
        return "unbundled"
    return "error"


def _record(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _configure() -> list[str]:
    from miniworld_engine import settings

    settings.configure(portable_on_hopper_blackwell=False, force_portable_kernels=False)
    return install_miss_recorder()


def run_calls(arch: str, row: int, start: int, out: Path) -> int:
    """Run the calls of one row from index ``start``; one JSON line per call appended to ``out``. Exit 3 after a sticky CUDA error
    (the caller restarts at the next index), 0 when the row is done."""
    misses = _configure()
    exp = expand(arch)[row - 1]
    for i in range(start, len(exp.calls)):
        call = exp.calls[i]
        before, seen, t0 = len(misses), len(RESOLVED), time.perf_counter()
        status, detail = "ok", ""
        try:
            run_call(call)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            status = classify(exc)
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            detail = f"{type(exc).__name__}: {first}"[:500]
            if status == "error":
                import traceback

                traceback.print_exc()
        new = misses[before:]
        if new and status == "ok":
            status, detail = "missing", "a loader swallowed the miss"
        _record(out, {"i": i, "label": call.label, "status": status, "detail": detail, "missing": sorted(set(new)),
                      "artifacts": sorted({f"{n}@{o}" for n, o in RESOLVED[seen:]}),
                      "seconds": round(time.perf_counter() - t0, 2)})
        print(f"{status:7s} {call.label}" + (f"  {detail}" if detail else ""), flush=True)
        if status != "ok" and STICKY.search(detail):
            return 3
    return 0


def run_kernel(arch: str, kernel: str, out: Path) -> int:
    misses = _configure()
    unit = next(u for u in kernel_units(arch) if u.kernel == kernel)
    t0, status, detail = time.perf_counter(), "ok", ""
    try:
        run_driver(unit)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        status = classify(exc)
        detail = f"{type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0]}"[:500]
        if status == "error":
            import traceback

            traceback.print_exc()
    if misses and status == "ok":
        status, detail = "missing", "a loader swallowed the miss"
    _record(out, {"i": 0, "label": f"kernel {kernel}", "status": status, "detail": detail, "missing": sorted(set(misses)),
                  "seconds": round(time.perf_counter() - t0, 2)})
    print(f"{status:7s} kernel {kernel}  {detail}", flush=True)
    return 0


def describe(arch: str) -> dict:
    """The expansion as JSON-able data (the verify reads it from the INSTALLED package)."""
    rows = expand(arch)
    return {
        "arch": arch,
        "rows": [{"row": e.row, "id": e.id, "key": e.key, "text": e.text_digest, "module": e.module, "stream": e.stream, "calls": len(e.calls),
                  "registry": e.registry_text, "not_arch": e.not_arch_reason, "excluded": e.excluded} for e in rows],
        "kernels": [dataclasses.asdict(k) for k in kernel_units(arch)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand")
    e.add_argument("--arch", default="sm90")
    e.add_argument("--json", action="store_true")
    c = sub.add_parser("run-calls")
    c.add_argument("--arch", default="sm90")
    c.add_argument("--row", type=int, required=True)
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--out", required=True)
    k = sub.add_parser("run-kernel")
    k.add_argument("--arch", default="sm90")
    k.add_argument("--kernel", required=True)
    k.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "expand":
        data = describe(args.arch)
        if args.json:
            print(json.dumps(data))
        else:
            for r in data["rows"]:
                print(f"{r['id']:70s} calls={r['calls']:5d}" + (f"  NOT {args.arch}: {r['not_arch']}" if r["not_arch"] else ""))
            print(f"rows={len(data['rows'])} calls={sum(r['calls'] for r in data['rows'])}")
        return 0
    if args.cmd == "run-calls":
        return run_calls(args.arch, args.row, args.start, Path(args.out))
    return run_kernel(args.arch, args.kernel, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
