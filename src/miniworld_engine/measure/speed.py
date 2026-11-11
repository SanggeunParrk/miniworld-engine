"""Speed of one module backend: build the module, warm it up, time its step precisely. Module-agnostic (no per-module code).

Usage::

    python -m miniworld_engine.measure.speed CONFIG.yaml --out /path/speed.jsonl

The config is the accuracy config (:mod:`miniworld_engine.measure.config`: modules or ``all``, backends, row / length / dtype / mode
filters) plus the speed keys ``timing`` (list of timing modes, default ``[graph]``), ``warmup`` (default 5), ``iterations`` (default 100),
``flush_l2`` (default true), ``compile_mode`` (default ``"default"``). The measurer only measures: one record per (case, backend, timing
mode) to stdout and to a JSONL file; it judges nothing, caches nothing, skips nothing.

Timing modes
------------
What a module costs inside a model is its GPU time. Two things blur that in a plain eager loop: the CPU (Python, autograd, dispatch,
Dynamo guards, the launches) when the GPU outruns it, and the unfused PyTorch glue a model run through ``torch.compile`` would fuse. So:

``graph`` (the default)
    The step (eval: the forward under ``no_grad``; train: forward + backward) runs ``warmup`` times eagerly on a side stream, is captured
    once into a ``torch.cuda.CUDAGraph`` on that stream with static inputs, replayed ``warmup`` times, then replayed ``iterations`` times,
    each replay between two CUDA events. A replay launches the recorded kernels with no Python, no dispatch and no launch gaps: the
    number is the backend's kernel time plus the inter-kernel gaps a graph still has, never CPU overhead. The glue runs as eager kernels.
``compile+graph``
    The backend's forward is first ``torch.compile``-d (``dynamic=False``, mode ``compile_mode``; never ``reduce-overhead`` /
    ``max-autotune``: their own cudagraph trees would nest in ours and replay copies), warmed up until it no longer compiles, captured once
    and discarded (code that branches on "is the stream capturing" compiles that variant there), then the same
    capture with ``error_on_recompile`` set. Inductor fuses the PyTorch glue around every backend (an opaque kernel is a graph break: the
    glue on both sides is still compiled), so a backend is credited with what it adds over a compiled reference, not with fusion the
    reference would also get. Under a manual graph a graph break costs nothing (no guards, no re-entry), which is why the graph is kept
    on top: eager-mode "compiled slowdowns" were the CPU (guards, custom-op dispatch), not the kernels.
``eager`` (diagnosis only, ``diagnosis_only: true`` in the record)
    The same step called directly, each call between two events. It includes whatever CPU time the GPU waits for; it is never a speed
    result. It exists to check that ``graph`` equals eager on a kernel-bound case and to see CPU-bound ones.

A backend whose step cannot be captured is timed eagerly instead: its record keeps the timing mode, says ``timed_as: eager`` and holds the
capture error (``capture_error``); an eager time includes the CPU time the GPU waits for, so it is an upper bound. One that cannot run
even eagerly (or cannot be compiled) is reported ``failed`` with the exception text. A case that
does not fit in the card's memory is reported ``skipped`` (an out-of-memory case is not a failure of the backend). A
backend whose ``_supports_<name>`` refuses the case is reported ``unsupported`` with its reason.

The policies
------------
- Range check outside the region: ``module.unsupported(**inputs)`` runs once, in the step's own grad / autocast context, on the same
  static inputs; the step then calls ``_forward_<backend>`` directly. A check may synchronise with the host (a mask count), which a
  capture cannot contain, and it is CPU work the graph excludes anyway.
- Train step: every floating leaf input and every parameter requires grad; the upstream gradient is a fixed seeded tensor per floating
  output; ``.grad`` is set to ``None`` at the start of the step, *inside* the captured region. That is the production step
  (``zero_grad(set_to_none=True)``, PyTorch's default, then backward): the captured backward allocates its gradients from the graph's
  pool and writes them, with no zeroing kernel and no accumulation add. Zeroing outside the graph would capture ``grad += g`` instead
  (one extra add per tensor, not the production step); ``zero_()`` inside would add memsets. No optimizer step.
- ``master``: fp32 parameters, bf16 activations (``config.DTYPES``), the step's forward under ``torch.autocast("cuda", bfloat16)`` inside the captured region (the backward
  runs outside autocast, as it does in training).
- L2: ``flush_l2: true`` (default) writes a 256 MB buffer (5x the H100's 50 MB L2) before every timed call, outside its events: every
  call starts cold (weights and inputs from HBM), as Triton's ``do_bench`` (the old engine's timer) does. Cold is the reproducible
  choice that favours no backend; ``false`` gives warm-L2 numbers for small working sets.
- Queueing: before the timed loop the GPU is held by ``torch.cuda._sleep`` long enough (estimated from the warm replays) for the CPU to
  enqueue every flush / event / replay, so no replay waits for its launch.
- Statistics: the median, p10 and p90 of the per-call times (ms). Speedup vs ``pytorch`` = pytorch median / backend median, in the same
  timing mode and case.
- Clocks are not locked (no root on the cluster): boost / thermal state varies run to run; compare numbers from one job.
- Seeds: the module is built after ``torch.manual_seed(seed)`` once per case and deep-copied for every backend (same weights); the inputs
  come from ``make_inputs`` with a ``torch.Generator`` seeded the same way for every backend; the upstream gradients from a seeded
  generator. TF32: every backend runs with TF32 products allowed in float32 PyTorch matmuls (the engine's fp32 kernels compute
  in TF32 too, and the accuracy measurer judges against the same reference; measure/config.py ``TF32_BACKENDS``). The flags of the run are in each record (``tf32_matmul``).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback

import torch
from torch.utils import _pytree

from miniworld_engine.modules.primitives import BackendUnsupported

TIMING_MODES = ("graph", "compile+graph", "eager")
#: Human-managed version of this measurer's meaning (how a step is timed: graph capture, L2 flush, statistics, the TF32 rule): bump it
#: when a change can alter a speed record, not for refactors. Recorded in ``meta`` with the config's version, the measurement settings and the
#: environment.
SPEED_VERSION = "3"
_FLUSH_BYTES = 256 * 1024 * 1024


# ---------------------------------------------------------------------------------------------------------------------------- the step
def _floats(tree) -> list[torch.Tensor]:
    return [t for t in _pytree.tree_leaves(tree) if isinstance(t, torch.Tensor) and t.is_floating_point()]


def _context(train: bool, autocast: torch.dtype | None):
    stack = contextlib.ExitStack()
    stack.enter_context(torch.enable_grad() if train else torch.no_grad())
    if autocast is not None:
        stack.enter_context(torch.autocast("cuda", dtype=autocast))
    return stack


def make_step(module, inputs: dict, *, backend: str, train: bool, autocast: torch.dtype | None, seed: int, compile_mode: str | None):
    """The callable one replay runs, or raise ``BackendUnsupported`` with the backend's reason.

    ``compile_mode`` None: the backend's forward as is; a string: ``torch.compile(forward, dynamic=False, mode=...)``."""
    from miniworld_engine.measure.config import TF32_BACKENDS

    allow = backend in TF32_BACKENDS   # every backend (user decision 2026-11-10, see TF32_RULE); one left out: fp32 without TF32
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    module.set_backend(backend)
    module.train(train)
    params = [p for p in module.parameters() if p.is_floating_point()] if train else []
    # the accuracy measurer's rule: a top-level floating input is a differentiable activation; a tensor inside a nested structure
    # (e.g. the SWA attention_params tuple: fp32 RoPE cos/sin, sequence lengths) is a constant and gets no gradient
    leaves = [v for v in inputs.values() if isinstance(v, torch.Tensor) and v.is_floating_point() and v.is_leaf] if train else []
    for t in leaves:
        t.requires_grad_(True)
    with _context(train, autocast):
        reason = module.unsupported(**inputs) if backend != "pytorch" else None
    if reason is not None:
        raise BackendUnsupported(reason)
    forward = getattr(module, f"_forward_{backend}")
    if compile_mode is not None:
        forward = torch.compile(forward, dynamic=False, mode=None if compile_mode == "default" else compile_mode)

    if not train:
        def step():
            with _context(False, autocast):
                return forward(**inputs)
        return step

    with _context(True, autocast):
        outs = [t for t in _floats(forward(**inputs)) if t.requires_grad]
    if not outs:
        raise RuntimeError("train step: no floating output requires grad")
    g = torch.Generator(device=outs[0].device).manual_seed(seed + 1)
    upstream = [torch.randn(o.shape, generator=g, device=o.device, dtype=torch.float32).to(o.dtype) for o in outs]
    del outs
    targets = params + leaves

    def step():
        for t in targets:
            t.grad = None            # set_to_none inside the region: the captured backward writes fresh gradients
        with _context(True, autocast):
            out = [t for t in _floats(forward(**inputs)) if t.requires_grad]
        torch.autograd.backward(out, upstream)
        return out
    return step


# ------------------------------------------------------------------------------------------------------------------------- the timer
class _Timer:
    def __init__(self, device, flush_l2: bool):
        self.device = device
        self.flush = torch.empty(_FLUSH_BYTES, dtype=torch.int8, device=device) if flush_l2 else None
        self._cycles_per_ms = None

    def cycles_per_ms(self) -> float:
        if self._cycles_per_ms is None:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(1_000_000)
            s.record()
            torch.cuda._sleep(10_000_000)
            e.record()
            torch.cuda.synchronize(self.device)
            self._cycles_per_ms = 10_000_000 / max(s.elapsed_time(e), 1e-3)
        return self._cycles_per_ms

    def run(self, fn, iterations: int, *, head_start_ms: float = 0.0) -> list[float]:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        torch.cuda.synchronize(self.device)
        if head_start_ms > 0:
            torch.cuda._sleep(int(min(head_start_ms, 2000.0) * self.cycles_per_ms()))
        for s, e in zip(starts, ends, strict=True):
            if self.flush is not None:
                self.flush.zero_()
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize(self.device)
        return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]


def _stats(ms: list[float]) -> dict:
    v = sorted(ms)

    def q(p):
        x = p * (len(v) - 1)
        lo, hi = math.floor(x), math.ceil(x)
        return v[lo] + (v[hi] - v[lo]) * (x - lo)
    return {"median_ms": q(0.5), "p10_ms": q(0.1), "p90_ms": q(0.9), "n": len(v)}


def _reason(exc: BaseException) -> str:
    """The exception and the ones it replaced (a failed capture's end masks the error raised inside the capture)."""
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen and len(parts) < 4:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {str(exc).strip()[:600]}")
        exc = exc.__cause__ or exc.__context__
    return " <- ".join(parts)


def _counter(group: str) -> int:
    from torch._dynamo.utils import counters
    return int(sum(counters[group].values()))


def _capture(step, stream) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    return graph


def measure(step, timer: _Timer, *, timing: str, warmup: int, iterations: int) -> dict:
    """Time ``step`` in one timing mode (see the module docstring). Exceptions propagate (the caller records them).

    Dynamo's counters are expected cleared before ``make_step`` (``run`` does): the compile statistics count from there."""
    extra = {}
    side = torch.cuda.Stream(timer.device)
    side.wait_stream(torch.cuda.current_stream(timer.device))
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
        if timing == "compile+graph":
            for _ in range(10):               # until a call compiles nothing new
                n = _counter("stats")
                step()
                if _counter("stats") == n:
                    break
            else:
                raise RuntimeError("torch.compile still compiling after the warmup")
    torch.cuda.current_stream(timer.device).wait_stream(side)
    torch.cuda.synchronize(timer.device)

    if timing == "eager":
        return _stats(timer.run(step, iterations))

    import torch._dynamo.config as dynamo_config

    if timing == "compile+graph":
        # A throwaway capture first: code that branches on "is the stream capturing" compiles its capture variant here; the kept
        # capture below must then compile nothing (error_on_recompile).
        del_graph = _capture(step, side)
        torch.cuda.synchronize(timer.device)
        del del_graph
    recompile = dynamo_config.error_on_recompile
    dynamo_config.error_on_recompile = True
    try:
        graph = _capture(step, side)
        torch.cuda.synchronize(timer.device)
    finally:
        dynamo_config.error_on_recompile = recompile
    if timing == "compile+graph":
        from torch._dynamo.utils import counters
        extra = {"compiled_graphs": int(counters["stats"]["unique_graphs"]), "graph_breaks": _counter("graph_break")}
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(timer.device)
    # the head start: how long the CPU takes to enqueue one timed call, times the calls, with margin
    t0 = time.perf_counter()
    for _ in range(3):
        if timer.flush is not None:
            timer.flush.zero_()
        graph.replay()
    enqueue_ms = (time.perf_counter() - t0) * 1e3 / 3
    torch.cuda.synchronize(timer.device)
    result = _stats(timer.run(graph.replay, iterations, head_start_ms=1.5 * enqueue_ms * iterations + 1.0))
    del graph
    return {**result, **extra}


# --------------------------------------------------------------------------------------------------------------------------- the driver
def _environment(device) -> dict:
    return {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "host": platform.node(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def run(cfg, out_path: str, *, device: str = "cuda") -> list[dict]:
    """Measure every case of the config, every backend, every timing mode; print a table and write JSONL records to ``out_path``."""
    from miniworld_engine.measure import config

    speed = config.speed_settings(cfg)
    timings = list(speed.get("timing", ["graph"]))
    bad = [t for t in timings if t not in TIMING_MODES]
    if bad:
        raise ValueError(f"unknown timing modes {bad}; choose from {TIMING_MODES}")
    warmup, iterations = int(speed.get("warmup", 5)), int(speed.get("iterations", 100))
    flush_l2, compile_mode = bool(speed.get("flush_l2", True)), str(speed.get("compile_mode", "default"))
    if compile_mode in ("reduce-overhead", "max-autotune"):
        raise ValueError(f"compile_mode {compile_mode!r} uses Inductor's own CUDA graphs; use 'default' or 'max-autotune-no-cudagraphs'")
    dev = torch.device(device)
    timer = _Timer(dev, flush_l2)
    env = {**_environment(dev), "flush_l2": flush_l2, "warmup": warmup, "iterations": iterations, "compile_mode": compile_mode}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records: list[dict] = []
    meta = config.run_meta(cfg, "speed", SPEED_VERSION)
    with open(out_path, "a") as sink:
        print(f"[speed] {env}", flush=True)
        print(f"[speed] records -> {out_path}", flush=True)
        for case in config.cases(cfg):
            base = config.build(case, device=dev)          # seeded: one set of weights for every backend
            for timing in timings:
                ref = None
                for backend in config.backends_of(cfg, case):
                    rec = {**config.describe(case, cfg), "meta": meta, "backend": backend, "timing": timing}
                    if timing == "eager":
                        rec["diagnosis_only"] = True
                    module = inputs = step = None
                    try:
                        if backend not in type(base).backends:
                            raise BackendUnsupported(f"{type(base).__name__} has no backend {backend!r}")
                        torch._dynamo.reset()
                        torch._dynamo.utils.counters.clear()
                        module = copy.deepcopy(base)
                        inputs = config.inputs(case, module, device=dev)   # make_inputs with a generator seeded per case
                        step = make_step(module, inputs, backend=backend, train=case.mode == "train", autocast=config.autocast(case),
                                         seed=config.seed(case), compile_mode=compile_mode if timing == "compile+graph" else None)
                        rec.update(status="ok", tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                   **measure(step, timer, timing=timing, warmup=warmup, iterations=iterations))
                    except BackendUnsupported as exc:
                        rec.update(status="unsupported", reason=str(exc))
                    except Exception as exc:  # recorded
                        # out of memory: the case does not fit on the card, it is skipped (not a failure of the backend)
                        rec.update(status="skipped" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed", reason=_reason(exc))
                        if os.environ.get("SPEED_TRACEBACK"):
                            traceback.print_exc()
                        if rec["status"] == "failed" and timing != "eager":
                            # A backend whose step cannot be captured (a host copy / sync inside) is timed eagerly instead: the record
                            # keeps its timing mode and says ``timed_as: eager`` with the capture error. Eager includes the CPU time
                            # the GPU waits for, so it is an upper bound, not comparable one-to-one with a captured time.
                            step = module = inputs = None
                            torch._dynamo.reset()
                            torch.cuda.synchronize(dev)
                            try:
                                module = copy.deepcopy(base)
                                inputs = config.inputs(case, module, device=dev)
                                step = make_step(module, inputs, backend=backend, train=case.mode == "train",
                                                 autocast=config.autocast(case), seed=config.seed(case), compile_mode=None)
                                timed = measure(step, timer, timing="eager", warmup=warmup, iterations=iterations)
                                rec.update(status="ok", timed_as="eager", capture_error=rec.pop("reason"),
                                           tf32_matmul=torch.backends.cuda.matmul.allow_tf32, **timed)
                            except Exception as exc2:  # recorded: not even the eager step runs
                                rec["reason"] = f"{rec.get('reason', '')} ; eager: {_reason(exc2)}"
                    finally:
                        step = module = inputs = None
                        torch._dynamo.reset()
                        torch.cuda.synchronize(dev)
                        torch.cuda.empty_cache()
                    if backend == "pytorch" and rec["status"] == "ok":
                        ref = rec["median_ms"]
                    if rec["status"] == "ok" and ref is not None:
                        rec["speedup_vs_pytorch"] = ref / rec["median_ms"]
                    rec["env"] = env
                    records.append(rec)
                    sink.write(json.dumps(rec, default=str) + "\n")
                    sink.flush()
                    print(_line(rec), flush=True)
            del base
            torch.cuda.empty_cache()
    print(f"[speed] {len(records)} records -> {out_path}", flush=True)
    return records


def _line(rec: dict) -> str:
    head = f"{rec.get('label', rec.get('module'))} | {rec['backend']:<9} | {rec['timing']:<13}"
    if rec["status"] != "ok":
        return f"{head} | {rec['status']}: {rec.get('reason', '')[:160]}"
    sp = rec.get("speedup_vs_pytorch")
    line = f"{head} | {rec['median_ms']:9.4f} ms  p10 {rec['p10_ms']:9.4f}  p90 {rec['p90_ms']:9.4f} | "
    line += f"x{sp:.3f} vs pytorch" if sp is not None else "-"
    if "compiled_graphs" in rec:
        line += f" | compiled_graphs={rec['compiled_graphs']} breaks={rec['graph_breaks']}"
    return line + (" | DIAGNOSIS ONLY" if rec.get("diagnosis_only") else "")


def main(argv=None) -> int:
    from miniworld_engine.measure import config

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("config", help="an accuracy-format config with the speed keys (see the module docstring)")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 -- a local-time file name
    p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "miniworld-speed", f"speed-{stamp}.jsonl"),
                   help="JSONL output path (default: under the system temp dir, never the repo)")
    p.add_argument("--timing", nargs="+", choices=TIMING_MODES, help="override the config's timing modes")
    args = p.parse_args(argv)
    cfg = config.load(args.config)
    if args.timing:
        cfg = config.with_speed(cfg, timing=args.timing)
    if not torch.cuda.is_available():
        print("[speed] no CUDA device", file=sys.stderr)
        return 2
    run(cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
