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
    augmentation: int = 1

    @property
    def label(self) -> str:
        d = ";".join(f"{k}={v}" for k, v in self.dims)
        opt = f" {self.option[0]}={self.option[1]}" if self.option else ""
        core = f"->{self.compute}" if self.compute else ""
        return (f"{self.module}[{self.impl}/{self.dtype}{core}] stream={self.stream} {d} "
                f"L={self.length} A={self.augmentation} {self.mode}{opt}")


def units(rows: list[ModuleRow], arch: str | None = None) -> list[DeriveUnit]:
    """Expand every row into the invocations to record.

    Options are swept ONE AT A TIME against the defaults, not as a cross product: each option
    selects among its own kernels, so pinning two at once records nothing the two single pins do
    not, and the cross product is what made the unit count explode.
    """
    from miniworld_engine import build as build_matrix
    from miniworld_engine.autotune.builder import SWITCHES

    sm = "sm_" + normalise_arch(arch).removeprefix("sm") if arch else None

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
            if sm and not build_matrix.allows(sm, row.module, impl, dtype):
                continue
            computes = row.computes or ("",)
            for compute in computes:
                for option in (None, *row.options):
                    if not applies(option, mode):
                        continue
                    if (sm and option and option[0] == "trimul_impl"
                            and not build_matrix.allows(sm, "triangle_multiplication", option[1], dtype)):
                        continue
                    out.append(DeriveUnit(
                        module=row.module, stream=row.stream, length=length,
                        dims=tuple(row.dims.items()), impl=impl, dtype=dtype,
                        compute=compute, mode=mode, option=option,
                        augmentation=row.augmentation(mode)))
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
    """Run one declared unit and return launches plus any error that invalidates its evidence."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    case = cases[unit.module]
    dims = dict(unit.dims)
    dtype = getattr(torch, unit.dtype)
    compute = getattr(torch, unit.compute) if unit.compute else None
    sink: list = []
    install_recorder(sink)
    # FakeTensor data pointers are all zero; never reuse another unit's folded weights.
    import sys
    inference = sys.modules.get("miniworld_engine.kernels.adaln.triton.inference")
    if inference is not None:
        inference._LNFOLD_CACHE.clear()

    # A ShapeEnv, so a data-dependent shape becomes an unbacked symint instead of an exception.
    # Without one, `aten.nonzero` raises DynamicOutputShapeException and takes the WHOLE module
    # down: SWA packs its varlen batch with a nonzero over the valid mask, so every swa unit died
    # there and `rmsnorm_bwd_triton`, which SWA's QK-norm backward launches, looked like a kernel
    # no module reaches. It was two lines from being reached.
    with _pin(unit.option) as p_drop, FakeTensorMode(allow_non_fake_inputs=True,
                                                    shape_env=ShapeEnv()):
        try:
            module = case.factory(dims, p_drop, unit.impl, dtype)
            module.train(unit.mode == "train")
            args = case.inputs(unit.augmentation, unit.length, dims, dtype, unit.stream)
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
    from miniworld_engine import settings
    from miniworld_engine.kernels.bias_only_attention import dispatch as bias_dispatch
    from miniworld_engine.kernels.layernorm import dispatch as ln_dispatch

    # Fake launches cannot provide timing evidence, including in-memory choices.
    # Explicit backend pins still exercise each declared alternative.
    settings.configure(layernorm_dispatch="off", biasonly_dispatch="off")

    # Both names: layernorm's persister is `store`, bias_only's is `_store`, and patching only
    # the public one let the bias_only calibrator keep writing -- an `_sm100` file appeared on the
    # very next run after this guard was added.
    for mod in (ln_dispatch, bias_dispatch):
        for name in ("store", "_store"):
            if hasattr(mod, name):
                setattr(mod, name, lambda *a, **k: None)


def install_native_recorders() -> None:
    """Native CUDA has no autotune grid; model its output without compiling or launching it.

    Keep the Python dispatch intact so the other declared pins still trace the Triton
    alternatives. Calling a pybind extension on FakeTensors can fail into a different
    dispatch branch, or wait on a compiler lock, neither of which is valid evidence.
    """
    from miniworld_engine.kernels.layernorm import cuda as ln_cuda
    from miniworld_engine.kernels.transition import cuda as transition_cuda

    # Ampere builds must retain the lean Triton dependency set. The CuTe
    # recorder is needed only when the target architecture can dispatch CuTe.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        import cutlass.cute as cute
        import quack.cache as quack_cache
        import quack.cute_dsl_utils as cute_utils

        # Preserve Python buffers and nested Triton calls; suppress binary JIT,
        # disk-cache reads/writes and occupancy kernels on fake inputs.
        quack_cache.CACHE_ENABLED = False
        # Shape-only recorder: no compiled native callable is returned.
        cute.compile = lambda *args, **kwargs: (lambda *args, **kwargs: None)  # ty: ignore[invalid-assignment]
        multiprocessors = torch.cuda.get_device_properties(0).multi_processor_count
        cute_utils.get_max_active_clusters = lambda cluster_size, device_capacity=None, device_id=0: max(  # ty: ignore[invalid-assignment]
            # Replace the lru-cached native occupancy query for fake dispatch.
            1, multiprocessors // cluster_size)
    # Some pure tensor helpers carry @torch.compile. Their eager bodies are what
    # the recorder needs; Dynamo would create another, incompatible FakeTensorMode.
    torch._dynamo.config.disable = True

    def require_fake(x):
        from torch._subclasses.fake_tensor import is_fake
        if not is_fake(x):
            raise RuntimeError("native derivation recorder received a real tensor")

    class TransitionExtension:
        def transition_b2b_fwd(self, x, rstd, c1, g, beta, wa, wb, ws, residual=True):
            require_fake(x)
            return x.new_empty((x.shape[0], ws.shape[0]))

        def transition_expand_gate_fwd(self, x, rstd, c1, g, beta, wa, wb):
            require_fake(x)
            return x.new_empty((x.shape[0], wa.shape[0]))

        def transition_expand_gatebwd_wgmma(self, x, rstd, c1, g, beta, wa, wb, grad):
            require_fake(x)
            m, k = x.shape
            nd = wa.shape[0]
            return x.new_empty((m, nd)), x.new_empty((m, 2 * nd)), x.new_empty((m, k))

    def transition_extension(name: str, width, config):
        if name not in ("b2b", "expand_gate", "gatebwd"):
            raise ValueError(f"no native derivation contract for {name}")
        return TransitionExtension()

    # Fake extension supplies shape functions instead of a pybind module.
    transition_cuda._ext = transition_extension  # ty: ignore[invalid-assignment]

    # FlashAttention is an external kernel, with no MiniWorld autotune grid.
    # Retain its Q/K/V gradient connections so the surrounding RMSNorm/rope
    # backwards are recorded, without importing/launching architecture-specific FA.
    from miniworld_engine.modules.swa_atom_attention import module as swa

    class FlashShape(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            require_fake(q)
            if q.shape != k.shape or q.shape != v.shape:
                raise ValueError("SWA flash shape recorder requires equal Q/K/V shapes")
            return torch.empty_like(q)

        @staticmethod
        def backward(ctx, grad):
            return torch.empty_like(grad), torch.empty_like(grad), torch.empty_like(grad)

    def flash_shape(q, k, v, *args, **kwargs):
        return FlashShape.apply(q, k, v)

    # The generic fake autograd function intentionally replaces the native signature.
    swa._flash_window_core = flash_shape  # ty: ignore[invalid-assignment]

    def ln_backward(dy, x, weight, mean, rstd, row_scale=None):
        from torch._subclasses.fake_tensor import is_fake
        if not is_fake(x):
            raise RuntimeError("native derivation recorder received a real tensor")
        if dy.dtype != x.dtype or weight.dtype != x.dtype:
            raise RuntimeError("CUDA LayerNorm requires matching input/weight dtypes")
        if not 1 <= x.shape[-1] <= 1024:
            raise RuntimeError("CUDA LayerNorm supports 1 <= N <= 1024")
        return (torch.empty_like(dy.contiguous()), torch.empty_like(weight.contiguous()),
                torch.empty_like(weight.contiguous()))

    # Shape-only outputs replace a native extension wrapper during derivation.
    ln_cuda.layer_norm_bwd_cuda = ln_backward  # ty: ignore[invalid-assignment]


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
    def capability(device: torch.device | str | int | None = None) -> tuple[int, int]:
        return cap

    from unittest.mock import patch

    patch.object(torch.cuda, "get_device_capability", capability).start()


# --------------------------------------------------------------------------- #
# the whole traversal, and its output file
# --------------------------------------------------------------------------- #
KERNEL_HEADER = ("kernel", "arch", "dtype", "bucket", "modules", "streams", "lengths", "units", "shapes")


def derive_all(rows: list[ModuleRow] | None = None, *, on_unit=None,
               arch: str | None = None, per_unit: Path | None = None,
               workers: int = 1) -> tuple[dict, list]:
    """Run every unit and collect ``(kernel, dtype, bucket) -> evidence``.

    Returns ``(entries, errors)``. Any failed declared invocation prevents publication of a
    complete registry. GPU support exclusions belong in the shared unit enumeration.
    """
    from miniworld_engine.autotune.builder import cases

    if workers > 1:
        return _parallel_derive(rows, arch, per_unit, workers)
    require_environment()
    if arch:
        target_arch(arch)
    install_no_calibration()
    install_module_apply()
    install_native_recorders()
    rows = module_rows() if rows is None else rows
    case_by_name = {c.name: c for c in cases()}

    entries: dict = {}
    skipped: list = []
    #: unit label -> the keys it produced. Written only when asked: it answers "which axis of the
    #: sweep earns its units", which is not a question a build needs but is exactly the question
    #: to ask when 6,526 units produce 833 buckets.
    per_unit_keys: dict = {}
    work = units(rows, arch=arch or sm_tag())
    for i, unit in enumerate(work, 1):
        if unit.module not in case_by_name:
            skipped.append((unit, f"no case builds {unit.module!r}"))
            continue
        launches, error = record(unit, case_by_name)
        unresolved = [x for x in launches if x[0].startswith("<untracked:")
                      or str(x[2]).startswith("<")]
        if unresolved and not error:
            error = f"unresolved cache launches: {unresolved}"
        if error:
            skipped.append((unit, error))
            if on_unit is not None:
                on_unit(i, len(work), unit, launches, error)
            continue
        for op, dtype, bucket in launches:
            ev = entries.setdefault((op, dtype, bucket), {
                "modules": set(), "streams": set(), "lengths": set(), "units": 0})
            ev["modules"].add(unit.module)
            ev["streams"].add(unit.stream)
            ev["lengths"].add(unit.length)
            ev.setdefault("shapes", {}).setdefault(unit.stream, set()).add(unit.length)
            ev["units"] += 1
        if per_unit is not None:
            per_unit_keys[unit.label] = sorted({f"{op}|{dt}|{b}" for op, dt, b in launches})
        if on_unit is not None:
            on_unit(i, len(work), unit, launches, error)
    if per_unit is not None:

        from miniworld_engine._atomic import write_json
        write_json(per_unit, {"schema": 1, "arch": normalise_arch(arch or sm_tag()),
                             "complete": not skipped, "units": per_unit_keys,
                             "errors": [{"unit": u.label, "reason": e} for u, e in skipped]})
    return entries, skipped


def _derive_chunk(args):
    torch.set_num_threads(1)
    rows, arch, path = args
    entries, errors = derive_all(rows, arch=arch, per_unit=path)
    return entries, errors, path


def _parallel_derive(rows, arch, per_unit, workers):
    """Separate processes isolate monkeypatches and share only the CUDA context requirement."""
    import json
    import multiprocessing
    import tempfile

    from miniworld_engine._atomic import write_json

    require_environment()
    rows = module_rows() if rows is None else rows
    # Balance by invocation count, not CSV row count.
    chunks = [[] for _ in range(min(workers, len(rows)))]
    sizes = [0] * len(chunks)
    for row in sorted(rows, key=lambda r: len(units([r])), reverse=True):
        i = min(range(len(chunks)), key=lambda n: sizes[n])
        chunks[i].append(row)
        sizes[i] += len(units([row]))
    entries, errors, evidence = {}, [], {}
    with tempfile.TemporaryDirectory(prefix="miniworld-derive-") as td:
        tasks = [(chunk, arch, Path(td) / f"{i}.json") for i, chunk in enumerate(chunks)]
        with multiprocessing.get_context("spawn").Pool(len(chunks)) as pool:
            for found, failed, path in pool.imap_unordered(_derive_chunk, tasks):
                errors.extend(failed)
                recorded = json.loads(path.read_text())["units"]
                if set(recorded) & set(evidence):
                    raise ValueError("duplicate derivation unit identities")
                evidence.update(recorded)
                for key, ev in found.items():
                    if key not in entries:
                        entries[key] = ev
                        continue
                    target = entries[key]
                    for name in ("modules", "streams", "lengths"):
                        target[name].update(ev[name])
                    target["units"] += ev["units"]
                    for stream, lengths in ev.get("shapes", {}).items():
                        target.setdefault("shapes", {}).setdefault(stream, set()).update(lengths)
                print(f"derive: {len(evidence)} invocations recorded, {len(errors)} errors", flush=True)
    if per_unit is not None:
        write_json(per_unit, {"schema": 1, "arch": normalise_arch(arch or sm_tag()),
                             "complete": not errors, "units": evidence,
                             "errors": [{"unit": u.label, "reason": e} for u, e in errors]})
    return entries, errors


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
    import json
    rows = []
    for (op, dtype, bucket), ev in sorted(entries.items()):
        if str(bucket).startswith("<"):
            raise ValueError(f"unresolved cache key for {op}: {bucket}")
        if op.startswith("<"):        # no autotuner
            continue
        rows.append({
            "kernel": op, "arch": arch, "dtype": dtype, "bucket": bucket,
            "modules": "|".join(sorted(ev["modules"])),
            "streams": "|".join(sorted(ev["streams"])),
            "lengths": "|".join(str(n) for n in sorted(ev["lengths"])),
            "units": ev["units"],
            "shapes": json.dumps({s: sorted(lengths) for s, lengths in
                                  sorted(ev.get("shapes", {}).items())}),
        })
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent,
                                     prefix=path.name, suffix=".tmp", delete=False) as fh:
        temporary = Path(fh.name)
        try:
            w = csv.DictWriter(fh, fieldnames=list(KERNEL_HEADER), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
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


def registry_path(arch: str) -> Path:
    """Prefer the plan for this exact source revision; retain the shipped-plan fallback."""
    from miniworld_engine.autotune import plan
    for path in plan.registry_candidates(arch):
        if plan.is_verified(arch, path):
            return path
    return REGISTRY_KERNEL


def kernel_rows(arch: str, path: Path | None = None) -> list[dict]:
    """The derived work list for one arch. Empty is an error, not an answer."""
    want = normalise_arch(arch)
    path = path if path is not None else registry_path(arch)
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

    ``missing`` names required keys without a usable current entry. ``extra`` is outside this
    declared module plan; driver or alternate-shape measurements remain preserved.
    """
    import json

    from miniworld_engine.autotune import cache, cache_status
    from miniworld_engine.autotune.configs import configs_for

    root = data_dir or (Path(__file__).resolve().parents[1] / "autotune" / "data")
    want = {(r["kernel"], f'{r["dtype"]}|{r["bucket"]}') for r in kernel_rows(arch)}
    have = set()
    usable = set()
    invalid = {}
    wanted_ops = {op for op, _key in want}
    for op_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = op_dir / f"{gpu_key_name}.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            invalid[op_dir.name] = "unreadable cache"
            continue
        for key in data.get("entries", {}):
            have.add((op_dir.name, key))
        if op_dir.name not in wanted_ops:
            continue
        try:
            configs = configs_for(op_dir.name)
            identity = cache_status._current_op_identity(op_dir.name)
            reason = ("cannot resolve kernel identity" if identity is None else
                      cache.measurement_mismatch(op_dir.name, data, identity))
            if not reason and data.get("config_space_hash") != cache.config_space_hash(configs):
                reason = "config grid changed"
            if reason:
                invalid[op_dir.name] = reason
                continue
            live = {cache._sig(c) for c in configs}
            implementation = cache_status._current_implementation_identity(op_dir.name)
            keys = set(data.get("entries", {})) | set(data.get("measurements", {}))
            for key in keys:
                ranked = cache.runtime_candidates(data, key, implementation)
                if isinstance(ranked, list) and any(
                        isinstance(c, dict) and "kwargs" in c
                        and cache._sig_from_dict(c) in live for c in ranked):
                    usable.add((op_dir.name, key))
        except (KeyError, TypeError, ValueError) as exc:
            invalid[op_dir.name] = f"invalid cache: {exc}"
    return {
        "arch": arch, "gpu": gpu_key_name,
        "want": len(want), "have": len(have),
        "missing": sorted(want - usable), "extra": sorted(have - want),
        "usable": len(want & usable), "invalid": invalid,
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
    from miniworld_engine.autotune.native import build_ops_for_arch
    native = build_ops_for_arch(normalise_arch(arch))
    ceiling = _ARCH_ORDER.index(normalise_arch(arch))
    with reg.open(newline="") as fh:
        buildable = {
            r["kernel"] for r in csv.DictReader(fh)
            if (r["backend"] == "triton" or r["kernel"] in native)
            and (r.get("driver") or "").strip()
            and ((r.get("developed") or "").strip() == "yes" or r["kernel"] in native)
            and _ARCH_ORDER.index(normalise_arch((r.get("arch") or "sm80").strip()
                                                 or "sm80")) <= ceiling
        }
    reached = {r["kernel"] for r in kernel_rows(arch)}
    # Triton derivation does not certify native config coverage. Their drivers
    # must run even when a module happens to call a native default.
    return (buildable - reached) | (buildable & native)
