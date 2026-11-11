"""Launch configs: per-GPU tables of measured top-K configs; rank 1 by default; opt-in in-process tuning.

Table   ``kernels/<family>/configs/<gpu_key>.json`` = ``{"gpu", "source", "version", "ops": {op: {key: [cfg, ...]}}}``, fastest
        first; written by ``_pretune --merge`` (``version``: its change counter, not read here).
        ``key`` = ``"<dtypes>|<shape>[|<flags>]"``: the sorted ``+``-joined floating dtypes of the tensor arguments; the decoded
        ``shape_key`` (``M=640,N=128``: M a row / L a length bucket, see ``_shape_key``; other names exact widths) or a native
        kernel's dims; the kernel's other int/bool key arguments. ``cfg`` = ``{"kwargs", "num_warps", "num_stages", ["maxnreg"],
        "ms"}`` (``ms``: what it was ranked by, reference only). A new GPU = new files, nothing else.
gpu_key ``<model>_sm<major><minor>``, model = the first alphanumeric token of the lowercased device name that holds a digit:
        "NVIDIA H100 80GB HBM3" -> ``h100_sm90``, "NVIDIA B200" -> ``b200_sm100``, "NVIDIA RTX A6000" -> ``a6000_sm86``.
        SKUs of one model and arch (H100 SXM / PCIe) share a table.
Default rank 1 of the exact (dtype, bucket, flags) entry that the kernel's prune keeps. No entry, unknown GPU, no CUDA -> the
        kernel's own default, warned once per (op, gpu). Nothing is measured: deterministic, CUDA-graph capture and torch.compile safe.
tune    ``tune(True)``: the first call at each EXACT shape (tensor shapes / strides / dtypes, scalar arguments) times the stored
        top-K with ``triton.testing.do_bench`` and keeps the winner in memory for the process. Inside CUDA-graph capture or a
        torch.compile trace nothing is timed: the winner already found at that exact shape, else rank 1 (warned once). So: warm
        up eagerly, then capture.
"""

from __future__ import annotations

import functools
import json
import math
import re
import typing
import warnings
from pathlib import Path

import torch
import triton

from miniworld_engine.kernels._shape_key import BASE_NAMES, pack

_ROOT, _META = Path(__file__).parent, ("num_warps", "num_stages", "maxnreg")
_TUNE, _WON, _WARNED = [False], {}, set()
#: The pretuner (``kernels/_pretune.py``) while it drives the modules, else None: asked first at every configured launch / ``choose``
#: with a ``run``; it returns the config to launch, or None for the normal table lookup.
PRETUNE: list[typing.Any] = [None]


def tune(on: bool = True) -> bool:
    """The one switch for in-process tuning of the stored top-K. Returns the previous state."""
    prev, _TUNE[0] = _TUNE[0], bool(on)
    return prev


@functools.cache
def _gpu_key(index: int) -> str:
    name = torch.cuda.get_device_name(index).lower().replace("nvidia", "")
    model = next((t for t in re.split("[^a-z0-9]+", name) if any(c.isdigit() for c in t)), "gpu")
    return "{}_sm{}{}".format(model, *torch.cuda.get_device_capability(index))


def gpu_key(device=None) -> str:
    device = None if device is None else torch.device(device)
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        return "cpu"
    return _gpu_key(torch.cuda.current_device() if device is None or device.index is None else device.index)


@functools.cache
def table(family: str, gpu: str) -> dict:
    path = _ROOT / family / "configs" / f"{gpu}.json"
    return json.loads(path.read_text())["ops"] if path.is_file() else {}


def _warn(op, gpu, why, kind="miss"):
    if (op, gpu, kind) not in _WARNED:
        _WARNED.add((op, gpu, kind))
        warnings.warn(f"[launch_config] {op} on {gpu}: {why}", RuntimeWarning, stacklevel=3)


def label(d: dict) -> str:
    return ",".join(f"{k}={int(v)}" for k, v in sorted(d.items()))


@functools.cache
def _parse(key: str):
    dtype, shape, *flags = key.split("|")
    fields = lambda s: tuple((k, int(v)) for k, v in (kv.split("=") for kv in s.split(",") if kv))
    return dtype, fields(shape), fields(flags[0] if flags else "")


def _cfg(c: dict) -> triton.Config:
    kwargs = c["kwargs"] if "kwargs" in c else {k: v for k, v in c.items() if k not in (*_META, "ms")}
    return triton.Config(dict(kwargs), **{k: c[k] for k in _META if c.get(k) is not None})


def _bench(fn) -> float:
    from triton.testing import do_bench
    return typing.cast(float, do_bench(fn, return_mode="median"))   # one value for return_mode="median"


def _best(exact, cands, time_one, after=None):
    """Rank 1, or (tune on) the fastest of ``cands`` at this exact call, memoised; never timed in capture / tracing."""
    if not _TUNE[0] or len(cands) < 2:
        return cands[0]
    if exact not in _WON:
        if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
            _warn(exact[0], exact[1], "tune is on but this exact shape was not warmed up before capture: rank 1", "capture")
            return cands[0]
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError
        ms = []
        for c in cands:
            try:
                ms.append(time_one(c))
            except (OutOfResources, CompileTimeAssertionFailure, PTXASError):  # noqa: PERF203 -- per candidate: one failure must not stop the timing
                ms.append(math.inf)
        _WON[exact] = cands[ms.index(min(ms))]
        if after is not None:
            after()
    return _WON[exact]


class Configured:
    """A ``@triton.jit`` kernel launched with one explicit config from the tables; ``kernel[grid](...)`` as before."""

    def __init__(self, fn, op, key, default, prune=None, reset_to_zero=(), restore_value=(), scratch=()):
        self.fn, self.op, self.key, self.prune, self.arg_names = fn, op, list(key), prune, fn.arg_names
        self.reset_to_zero, self.restore_value = list(reset_to_zero), list(restore_value)
        self.scratch = tuple(scratch)  # read by the pretuner only (see configured)
        self.default = [_cfg(c) for c in ([default] if isinstance(default, dict) else default)]
        jit = fn
        while not hasattr(jit, "module"):
            jit = jit.fn
        self.family, self._index = jit.module.split(".")[2], {}

    def _entries(self, gpu):
        if gpu not in self._index:
            idx = self._index[gpu] = {}
            for key, cfgs in table(self.family, gpu).get(self.op, {}).items():
                dtype, shape, flags = _parse(key)
                shape = dict(shape)
                base = next((shape.pop(n) for n in BASE_NAMES if n in shape), None)
                idx[(dtype, None if base is None else pack(base, **shape), flags)] = [_cfg(c) for c in cfgs]
        return self._index[gpu]

    @property
    def configs(self):
        """Every config this kernel can launch with on the current device: its default plus the table's entries."""
        out = list(self.default)
        for cfgs in self._entries(gpu_key()).values():
            out += [c for c in cfgs if c not in out]
        return out

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(self, *args, grid, warmup=False, **kwargs):
        named = dict(zip(self.arg_names, args, strict=False))
        every = {**named, **kwargs}
        tensors = [v for v in every.values() if isinstance(v, torch.Tensor)]
        gpu = gpu_key(tensors[0].device if tensors else None)
        dtype = "+".join(sorted({str(t.dtype).removeprefix("torch.") for t in tensors if t.dtype.is_floating_point})) or "any"
        flags = tuple(sorted((k, int(every[k])) for k in self.key if k != "shape_key" and isinstance(every.get(k), (bool, int))))
        key = (dtype, every.get("shape_key") if "shape_key" in self.key else None, flags)

        def launch(c, compile_only=False):  # compile_only: compile, do not launch (the pretuner compiles candidates in parallel)
            return self.fn.run(*args, grid=grid, warmup=warmup or compile_only, **kwargs, **c.all_kwargs())

        def time_one(c):
            saved = {n: every[n].clone() for n in self.restore_value}

            def call():
                for n in self.reset_to_zero:
                    every[n].zero_()
                launch(c)
                for n, t in saved.items():
                    every[n].copy_(t)
            return _bench(call)

        if PRETUNE[0] is not None:
            chosen = PRETUNE[0].configured(self, gpu, key, named, kwargs, every, launch, time_one)
            if chosen is not None:
                return launch(chosen)
        cands = self._entries(gpu).get(key, [])
        if cands and self.prune:
            try:
                cands = list(self.prune(cands, named, **kwargs))
            except Exception:  # a prune that refuses every stored config
                cands = []
        if not cands:
            _warn(self.op, gpu, f"no table entry for {key}; using the kernel's default config")
            cands = (list(self.prune(self.default, named, **kwargs)) if self.prune else self.default)[:1]

        zero = lambda: [every[n].zero_() for n in self.reset_to_zero]  # outputs as an untimed launch finds them
        exact = (self.op, gpu, *((tuple(v.shape), v.stride(), v.dtype) if isinstance(v, torch.Tensor) else
                                 v if isinstance(v, (int, float, bool, str, type(None))) else type(v).__name__
                                 for v in every.values()))
        return launch(_best(exact, cands, time_one, zero))


def configured(op, *, key, default, prune=None, reset_to_zero=(), restore_value=(), scratch=()):
    """Decorator for a ``@triton.jit`` kernel. ``default``: the kernel's own config (a dict of tile kwargs + num_warps /
    num_stages), or a list of them that ``prune`` (an ``early_config_prune``) takes the first valid one from. ``scratch``: names of
    tensor arguments whose contents after the launch are config-dependent intermediates (per-block partials / split slots whose
    layout follows the tile config, summed by a later kernel); the pretuner does not compare them between candidates. The launch
    ignores it."""
    return lambda fn: Configured(fn, op, key, default, prune, reset_to_zero, restore_value, scratch)


@functools.cache
def _ranked(family, gpu, op, dtype, dims, near):
    """The table entry at exactly ``dims``, else at the nearest measured dims: same dtype and non-``near`` fields, least
    summed |log ratio| over ``near``."""
    entries = table(family, gpu).get(op, {})
    exact = entries.get(f"{dtype}|{label(dict(dims))}")
    if exact is not None or not near:
        return exact
    want = dict(dims)

    def distance(key):
        kd, shape, _ = _parse(key)
        shape = dict(shape)
        if kd != dtype or shape.keys() != want.keys() or any(shape[k] != want[k] for k in want if k not in near):
            return math.inf
        return sum(abs(math.log(max(1, shape[k]) / max(1, want[k]))) for k in near)
    best = min(entries, key=distance, default=None)
    return entries[best] if best is not None and distance(best) < math.inf else None


@torch.compiler.disable
def choose(family, op, dims, dtype, candidates, *, device=None, near=("M",), run=None):
    """A native (CUDA) kernel's config: the ranked table entry for ``dims`` (exact, else nearest, see ``_ranked``) kept where
    it is one of ``candidates``; else ``candidates[0]``, warned once. Tune on and ``run(config)`` given: the survivors are timed."""
    return _choose(family, op, dims, dtype, candidates, device=device, near=near, run=run)


def _choose(family, op, dims, dtype, candidates, *, device=None, near=("M",), run=None):
    if not candidates:
        raise ValueError(f"{op}: no configuration supports this workload")
    gpu, dtype = gpu_key(device), str(dtype).removeprefix("torch.")
    dims = tuple(sorted((k, int(v)) for k, v in dims.items()))
    if PRETUNE[0] is not None and run is not None:
        chosen = PRETUNE[0].choose(family, op, gpu, dtype, dims, candidates, run)
        if chosen is not None:
            return dict(chosen)
    live = [dict(c["kwargs"]) for c in _ranked(family, gpu, op, dtype, dims, tuple(near)) or () if c["kwargs"] in candidates]
    if not live:
        _warn(op, gpu, f"no table entry for {dtype}|{label(dict(dims))}; using the kernel's default config")
        return dict(candidates[0])
    if run is None:
        return live[0]
    return _best((op, gpu, dims, dtype), live, lambda c: _bench(lambda: run(c)))
