"""Derive the kernel work list from ``registry_module.csv`` by RUNNING the modules on fake tensors.

The old plan was a hand-written ladder per axis in :mod:`builder`: the registry named WHICH
constexpr a kernel buckets on, and a tuple in Python named the VALUES. Those tuples drifted from
what the modules actually launch -- ``TOKEN_SHAPES`` stopped at 512 while the sweep ran to 1024,
``MSA_WIDTHS`` held 64 while the config declares 64 and 128 -- and every drift is a bucket
production reaches with no cache entry. 146 replay misses came out of that one split.

What replaces it is not a better ladder. It is not reading the ladder off the source either:
dispatch turns on values a parser cannot know (``H = 128 // D if D <= 128 else 1``;
``self.training``; ``is_sm86(x.device)``), which is why the hand-written tuples existed in the
first place. So this module RUNS each module row, and records the ``(op, dtype, bucket)`` of every
Triton launch that happens. Python decides the branches, because Python is the thing that decides
them in production too.

Running it does not cost a real forward:

* every tensor is a :class:`FakeTensor` -- shapes and dtypes propagate through the real launcher
  code, nothing is allocated, and an 8192-atom pair activation is as cheap as a 128-token one;
* the Triton launch itself is replaced by a recorder that reads the autotuner's own ``key=[...]``
  and returns, so no config is compiled and no kernel runs.

Two things it still needs a GPU for, and neither is compute:

* ``compile_wrap`` must be ``disable``. The default registers every kernel entry point as a
  ``torch.library`` custom op with a fake implementation, and under ``FakeTensorMode`` torch
  dispatches to THAT -- correct output shapes, launcher never entered, zero ops recorded.
* a CUDA device has to exist. ``.cuda()`` initialises a real context, and the autograd engine
  refuses to start a device thread without one ("hasPrimaryContext expects a valid device index").
  Forward-only derivation runs fine on a CPU node; the backward half does not.

Because dispatch reads the compute capability, the derived list is PER ARCH -- the same module row
yields different kernels on sm_86 and sm_90. That is a property of the code, not of this module.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import itertools
import os
from pathlib import Path

import torch

from miniworld_engine.autotune.module_registry import (
    REGISTRY_MODULE as REGISTRY_MODULE,
)
from miniworld_engine.autotune.module_registry import (
    STREAM_LADDERS as STREAM_LADDERS,
)
from miniworld_engine.autotune.module_registry import (
    ModuleRow as ModuleRow,
)
from miniworld_engine.autotune.module_registry import (
    module_rows as module_rows,
)

REGISTRY_KERNEL = Path(__file__).resolve().parents[1] / "kernels" / "registry_kernel.csv"


@dataclasses.dataclass(frozen=True)
class DeriveUnit:
    """One module invocation to record: everything that can change which kernel fires."""

    module: str
    stream: str
    length: int
    dims: tuple[tuple[str, int], ...]
    impl: str
    dtype: str
    compute: str
    mode: str
    option: tuple[str, str] | None

    @property
    def label(self) -> str:
        d = ";".join(f"{k}={v}" for k, v in self.dims)
        opt = f" {self.option[0]}={self.option[1]}" if self.option else ""
        core = f"->{self.compute}" if self.compute else ""
        return (f"{self.module}[{self.impl}/{self.dtype}{core}] {d} "
                f"L={self.length} {self.mode}{opt}")


def units(rows: list[ModuleRow]) -> list[DeriveUnit]:
    """Expand every row into the invocations to record.

    Options are swept ONE AT A TIME against the defaults, not as a cross product: each option
    selects among its own kernels, so pinning two at once records nothing the two single pins do
    not, and the cross product is what made the unit count explode.
    """
    from miniworld_engine.autotune.builder import SWITCHES

    def applies(option, mode: str) -> bool:
        """A switch selects among kernels on ONE side of the pass.

        ``ln_bwd_path`` pinned during an eval unit changes nothing and records exactly what the
        unpinned unit already did -- a duplicate that costs a unit and covers no bucket. SWITCHES
        already says which modes each switch reaches, so the gate reads it rather than restating
        it here.
        """
        if option is None:
            return True
        name = option[0]
        if name == "p_drop":
            return mode == "train"
        return mode in SWITCHES[name][1]

    out = []
    for row in rows:
        for length, impl, dtype, mode in itertools.product(
                row.lengths, row.impls, row.dtypes, row.modes):
            computes = row.computes or ("",)
            for compute in computes:
                for option in (None, *row.options):
                    if not applies(option, mode):
                        continue
                    out.append(DeriveUnit(
                        module=row.module, stream=row.stream, length=length,
                        dims=tuple(row.dims.items()), impl=impl, dtype=dtype,
                        compute=compute, mode=mode, option=option))
    return out


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def install_module_apply() -> None:
    """``nn.Module._apply`` reassigns parameters instead of swapping them.

    torch takes the ``swap_tensors`` path unconditionally for a ``FakeTensor`` parameter
    (``isinstance(param, FakeTensor)`` in ``Module._apply``), and ``swap_tensors`` refuses a tensor
    that has a weakref on it -- which every fake tensor does, because the fake-tensor converter
    holds one. So ``.cuda()`` on a module built under ``FakeTensorMode`` raises "Couldn't swap
    LayerNorm.weight" and no case can be constructed at all. Neither
    ``set_swap_module_params_on_conversion`` nor ``set_overwrite_module_params_on_conversion``
    turns the branch off; the isinstance check is not gated on them.
    """
    from torch import nn

    def _apply(self, fn, recurse=True):
        if recurse:
            for mod in self.children():
                mod._apply(fn)
        for key, param in self._parameters.items():
            if param is None:
                continue
            with torch.no_grad():
                applied = fn(param)
            self._parameters[key] = nn.Parameter(applied, requires_grad=param.requires_grad)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    nn.Module._apply = _apply


def _arg_names(kernel) -> list[str]:
    """Triton moved ``arg_names`` between the wrapper and the wrapped function across versions."""
    for obj in (kernel, getattr(kernel, "fn", None), getattr(kernel, "base_fn", None)):
        names = getattr(obj, "arg_names", None)
        if names:
            return list(names)
    return []


def install_recorder(sink: list) -> None:
    """Every Triton launch appends ``(op, dtype, bucket)`` to ``sink`` and returns without running.

    Recorded at the autotuner rather than at each kernel: ``bucket_of_autotuner`` reads the
    autotuner's own ``key=[...]``, which is the same list Triton re-tunes on, so the bucket
    recorded here is by construction the bucket the cache will be asked for. A kernel with no
    autotuner has no cache entry to build, so it is recorded by name only.
    """
    import triton.runtime.autotuner as autotuner_mod
    import triton.runtime.jit as jit_mod

    from miniworld_engine.autotune.cache import bucket_of_autotuner, dtype_of_args
    from miniworld_engine.autotune.configs import op_of

    def tuned_run(self, *args, **kwargs):
        nargs = dict(zip(_arg_names(self), args, strict=False))
        nargs.update(kwargs)
        op = op_of(getattr(self, "configs", None) or [])
        if op is None:
            sink.append((f"<untracked:{getattr(self.base_fn, '__name__', '?')}>", "", ""))
            return
        try:
            bucket = bucket_of_autotuner(self, nargs, None)
        except Exception as exc:                       # a key the fake args cannot answer
            bucket = f"<unresolved:{type(exc).__name__}>"
        sink.append((op, dtype_of_args(nargs), bucket))
        return

    def plain_run(self, *args, **kwargs):
        # No autotuner => no config grid => nothing for the cache to hold. Recorded so the report
        # can say the launch happened, never written to registry_kernel.csv.
        sink.append((f"<unautotuned:{self.fn.__name__}>", "", ""))
        return

    autotuner_mod.Autotuner.run = tuned_run
    if hasattr(autotuner_mod, "Heuristics"):
        autotuner_mod.Heuristics.run = tuned_run
    jit_mod.JITFunction.run = plain_run


@contextlib.contextmanager
def _pin(option: tuple[str, str] | None):
    """Apply one option for the duration of a unit, then put the settings back."""
    from miniworld_engine import settings
    from miniworld_engine.autotune.builder import SWITCH_SETTINGS

    if option is None:
        yield 0.0
        return
    name, raw = option
    if name == "p_drop":
        yield float(raw)
        return
    field, parse = SWITCH_SETTINGS[name]
    before = getattr(settings.current(), field)
    settings.configure(**{field: parse(raw)})
    try:
        yield 0.0
    finally:
        settings.configure(**{field: before})


def record(unit: DeriveUnit, cases: dict) -> tuple[list, str | None]:
    """Run one unit on fake tensors. Returns ``(launches, error)``; ``error`` is not a failure.

    A module that cannot take a shape is DATA -- ``d_hidden != d_pair`` is refused by every fused
    trimul back half, and a head dim below 16 will not compile -- so the reason is carried out and
    reported rather than raised.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = cases[unit.module]
    dims = dict(unit.dims)
    dtype = getattr(torch, unit.dtype)
    compute = getattr(torch, unit.compute) if unit.compute else None
    sink: list = []
    install_recorder(sink)

    with _pin(unit.option) as p_drop, FakeTensorMode(allow_non_fake_inputs=True):
        try:
            module = case.factory(dims, p_drop, unit.impl, dtype)
            module.train(unit.mode == "train")
            args = case.inputs(1, unit.length, dims, dtype, unit.stream)
            kw = {"compute_dtype": compute} if compute is not None else {}
            if unit.mode == "train":
                args = tuple(
                    a.detach().clone().requires_grad_(True)
                    if torch.is_tensor(a) and a.is_floating_point() else a
                    for a in args)
                out = module(*args, **kw)
                out = out[0] if isinstance(out, tuple) else out
                out.float().sum().backward()
            else:
                with torch.no_grad():
                    module(*args, **kw)
        except Exception as exc:
            return sink, f"{type(exc).__name__}: {exc}"
    return sink, None


def require_environment() -> None:
    """Fail loudly on the two things that make the derivation silently record nothing."""
    if os.environ.get("MINIWORLD_COMPILE_WRAP") != "disable":
        msg = ("derive needs MINIWORLD_COMPILE_WRAP=disable. With the default 'custom_op' every "
               "kernel entry point is a torch.library op with a fake implementation, so under "
               "FakeTensorMode torch dispatches to the fake one: shapes come out right, the "
               "launcher is never entered, and every unit records zero kernels.")
        raise RuntimeError(msg)
    if not torch.cuda.is_available():
        msg = ("derive needs a CUDA device to exist (it runs no kernel and allocates no memory). "
               "Dispatch reads the compute capability, .cuda() initialises a context, and the "
               "autograd engine will not start a device thread without one.")
        raise RuntimeError(msg)


def install_no_calibration() -> None:
    """Stop the runtime dispatch calibrators from persisting anything during a derivation.

    ``layernorm/dispatch.py`` and ``bias_only_attention/dispatch.py`` time both paths on first use
    and write the winner to ``autotune/data/<name>_dispatch/<gpu>.json``, in-repo. Under a
    derivation both of those inputs are wrong: the launches are stubbed, so every "timing" is the
    cost of returning None, and ``target_arch`` moves the file name -- a first run wrote
    ``NVIDIA_RTX_A6000_sm860``, ``_sm900``, ``_sm1000`` and ``_sm100`` and, worse, overwrote the
    REAL ``_sm86`` files with times measured off stubs.
    """
    from miniworld_engine.kernels.bias_only_attention import dispatch as bias_dispatch
    from miniworld_engine.kernels.layernorm import dispatch as ln_dispatch

    # Both names: layernorm's persister is `store`, bias_only's is `_store`, and patching only
    # the public one let the bias_only calibrator keep writing -- an `_sm100` file appeared on the
    # very next run after this guard was added.
    for mod in (ln_dispatch, bias_dispatch):
        for name in ("store", "_store"):
            if hasattr(mod, name):
                setattr(mod, name, lambda *a, **k: None)


def target_arch(sm: str) -> None:
    """Derive FOR ``sm`` instead of for the card this runs on.

    Dispatch reads the compute capability at almost every branch that picks between kernels
    (``is_sm90``, ``is_sm86``, ``get_device_capability(x.device)[0] == 10``), so the kernel set a
    module row yields is a function of the arch. Nothing here runs a kernel, so the card only has
    to EXIST -- which means one A6000 can derive the sm_90 and sm_100 lists too, and a kernel that
    appears on no arch at all is a kernel nothing can reach rather than one this card missed.

    Patching ``torch.cuda.get_device_capability`` rather than ``dispatch.capability``: 14 call
    sites across the kernel layer read the torch function directly, and half a patch would give a
    module the target arch at the module layer and the real one inside the launcher.
    """
    digits = sm.removeprefix("sm_").removeprefix("sm")
    if not digits.isdigit() or len(digits) < 2:
        msg = f"arch {sm!r} is not an sm tag; expected sm86 / sm_86 / sm90 / sm100"
        raise ValueError(msg)
    # The LAST digit is the minor: sm_86 is (8, 6), sm_90 is (9, 0), sm_100 is (10, 0). Splitting
    # on a "." instead read "86" as major 86, which is >= 10 and therefore sm100 to every
    # `is_sm100` gate -- so all three arch runs derived the same list and the mistake looked like
    # "the arch does not matter".
    cap = (int(digits[:-1]), int(digits[-1]))
    torch.cuda.get_device_capability = lambda *a, **k: cap


# --------------------------------------------------------------------------- #
# the whole traversal, and its output file
# --------------------------------------------------------------------------- #
KERNEL_HEADER = ("kernel", "arch", "dtype", "bucket", "modules", "streams", "lengths", "units")


def derive_all(rows: list[ModuleRow] | None = None, *, on_unit=None,
               arch: str | None = None) -> tuple[dict, list]:
    """Run every unit and collect ``(kernel, dtype, bucket) -> evidence``.

    Returns ``(entries, skipped)``. ``skipped`` holds ``(unit, reason)`` for every module that
    refused its shape -- reported, never dropped, because a shape the modules refuse is a shape the
    old hand-written ladders were quietly asking the build to tune.
    """
    from miniworld_engine.autotune.builder import cases

    require_environment()
    if arch:
        target_arch(arch)
    install_no_calibration()
    install_module_apply()
    rows = module_rows() if rows is None else rows
    case_by_name = {c.name: c for c in cases()}

    entries: dict = {}
    skipped: list = []
    work = units(rows)
    for i, unit in enumerate(work, 1):
        if unit.module not in case_by_name:
            skipped.append((unit, f"no case builds {unit.module!r}"))
            continue
        launches, error = record(unit, case_by_name)
        for op, dtype, bucket in launches:
            ev = entries.setdefault((op, dtype, bucket), {
                "modules": set(), "streams": set(), "lengths": set(), "units": 0})
            ev["modules"].add(unit.module)
            ev["streams"].add(unit.stream)
            ev["lengths"].add(unit.length)
            ev["units"] += 1
        if error:
            skipped.append((unit, error))
        if on_unit is not None:
            on_unit(i, len(work), unit, launches, error)
    return entries, skipped


def sm_tag() -> str:
    """The arch this derivation is FOR. Part of every row: dispatch reads it, so the kernel set
    a module row yields is not the same on sm_86 and sm_90."""
    from miniworld_engine import build as build_matrix

    if not torch.cuda.is_available():
        return "unknown"
    return build_matrix.sm_tag(torch.cuda.get_device_capability())


def write_kernel_registry(entries: dict, arch: str, path: Path = REGISTRY_KERNEL) -> int:
    """Write ``registry_kernel.csv``: one row per ``(kernel, arch, dtype, bucket)`` to build.

    Rows carry the evidence that produced them -- which module rows reached this bucket, on which
    stream, at which lengths -- so a bucket that looks wrong can be traced back to the line of
    ``registry_module.csv`` that asked for it without re-running anything.
    """
    rows = []
    for (op, dtype, bucket), ev in sorted(entries.items()):
        if op.startswith("<"):        # no autotuner, or a key the recorder could not resolve
            continue
        rows.append({
            "kernel": op, "arch": arch, "dtype": dtype, "bucket": bucket,
            "modules": "|".join(sorted(ev["modules"])),
            "streams": "|".join(sorted(ev["streams"])),
            "lengths": "|".join(str(n) for n in sorted(ev["lengths"])),
            "units": ev["units"],
        })
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(KERNEL_HEADER))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def normalise_arch(sm: str) -> str:
    """``sm_86`` and ``sm86`` are the same architecture spelled by two callers.

    ``build_matrix.sm_tag`` underscores ("sm_86" names a policy file in gpu_to_kernels/) and
    ``cache.gpu_key`` does not ("NVIDIA RTX A6000 (sm86)" names a device). registry.csv's `arch`
    column and registry_kernel.csv both use the bare form. Passing the underscored one through
    made `uncovered_kernels` raise -- and would have made `kernel_rows` return NOTHING, which
    reads as "no module reaches any kernel" and would have put all 61 into the driver sweep.
    """
    return sm.replace("_", "")


def kernel_rows(arch: str, path: Path = REGISTRY_KERNEL) -> list[dict]:
    """The derived work list for one arch. Empty is an error, not an answer."""
    want = normalise_arch(arch)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    mine = [r for r in rows if normalise_arch(r["arch"]) == want]
    if not mine:
        have = sorted({r["arch"] for r in rows})
        msg = (f"registry_kernel.csv has no rows for arch {arch!r} (it holds {have}). "
               f"Run `miniworld-engine dev derive` on that card, or with --arch {want}.")
        raise KeyError(msg)
    return mine


def coverage(arch: str, gpu_key_name: str, data_dir: Path | None = None) -> dict:
    """Compare a built cache against ``registry_kernel.csv`` -- offline, on any machine.

    This is what ``dev audit --replay`` was for, without the card and the half hour. Replay drove
    the modules on a GPU and reported the keys the cache could not answer; the derivation drove
    the SAME dispatch on fake tensors and wrote down every key that will be asked for, so the
    comparison is now two file reads. It is also strictly better than replay was: replay could
    only report what it happened to reach in one run, and its misses had to be re-classified by
    hand every time (31 of 146 turned out to be aborted cases that asked for nothing).

    ``missing`` is a bucket production will reach with no entry -- a full-grid autotune inside a
    forward. ``extra`` is an entry nothing reaches on this arch: dead weight, and, when a kernel
    is entirely extra, usually a wrong ``arch`` declaration in registry.csv.
    """
    import json

    root = data_dir or (Path(__file__).resolve().parents[1] / "autotune" / "data")
    want = {(r["kernel"], f'{r["dtype"]}|{r["bucket"]}') for r in kernel_rows(arch)}
    have = set()
    for op_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = op_dir / f"{gpu_key_name}.json"
        if not f.exists():
            continue
        for key in json.loads(f.read_text()).get("entries", {}):
            have.add((op_dir.name, key))
    return {
        "arch": arch, "gpu": gpu_key_name,
        "want": len(want), "have": len(have),
        "missing": sorted(want - have), "extra": sorted(have - want),
    }


#: sm tag -> the arch tags whose kernels this card can also run. A kernel declared sm80 runs on
#: sm86; one declared sm90 does not.
_ARCH_ORDER = ("sm80", "sm86", "sm90", "sm100")


def uncovered_kernels(arch: str, registry: Path | None = None) -> set[str]:
    """Buildable kernels on ``arch`` that NO module row reaches.

    These are the alternative implementations kept for A/B -- the atomic attention backward, the
    m-major layernorm backward, the recompute variants -- plus anything a module stopped calling.
    They register with the cache all the same, so an unbuilt one is a full-grid stall the day
    something starts reaching it, and no module sweep can ever produce them.

    They each have a DRIVER, which is the only way to reach them, so `build all` runs the module
    sweep and then the driver sweep narrowed to exactly this set. Narrowed to it, and not run
    whole: the driver sweep's own width ladders are hand-written, and running them for a kernel
    the modules already cover is how the build ended up with two disagreeing statements of the
    same shapes in the first place.
    """
    reg = registry or (Path(__file__).resolve().parents[1] / "kernels" / "registry.csv")
    ceiling = _ARCH_ORDER.index(normalise_arch(arch))
    with reg.open(newline="") as fh:
        buildable = {
            r["kernel"] for r in csv.DictReader(fh)
            if r["backend"] == "triton"
            and (r.get("driver") or "").strip()
            and (r.get("developed") or "").strip() == "yes"
            and _ARCH_ORDER.index(normalise_arch((r.get("arch") or "sm80").strip()
                                                 or "sm80")) <= ceiling
        }
    reached = {r["kernel"] for r in kernel_rows(arch)}
    return buildable - reached
