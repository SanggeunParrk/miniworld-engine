"""Accuracy of module backends against a float64 gold run. Module-agnostic: no per-module code; the inputs come from ``make_inputs``.

Usage::

    python -m miniworld_engine.measure.accuracy CONFIG.yaml --out /scratch/dir/or/file.jsonl [--module X ...] [--backend B ...]

Per case (:mod:`miniworld_engine.measure.config`) the module is built once (random seeded weights, the case dtype) and deep-copied:

- gold: the ``pytorch`` backend in float64 -- the same (dtype-rounded) parameters and inputs cast up to float64 (``module.double()``:
  a parameter the module pins to fp32 under bf16, the LayerNorm affine, becomes fp64 too; every statement the modules compute "in at least
  fp32" (``primitives.wide``) runs in fp64), no autocast;
- ``pytorch``: the reference at the case dtype (``master``: fp32 parameters, forward under ``torch.autocast("cuda", bfloat16)``), with
  TF32 products allowed in float32 (``torch.backends.cuda.matmul.allow_tf32``); every other backend runs with TF32 off wherever it
  calls PyTorch's own matmuls;
- every other backend of the config the class has, the same way.

Measured tensors: the forward output(s) and, in ``train``, the gradient of every floating input and every parameter, for one random
upstream gradient (seeded, the same for every run). The error of a tensor is ``max|x - gold| / max|gold|`` (``max|x - gold|`` where the
gold is all zero). Dropout and other training noise (``p_drop``, ``nn.Dropout``, ``coordinate_noise``) is switched off in every run: its
masks are random draws, not numerics.

The light, per tensor: ``tol = max(err_pytorch, fixed[dtype])``; ``err_backend <= tol`` GREEN, ``<= 5 tol`` YELLOW, beyond RED (a NaN /
inf error is RED). A call's light is its worst tensor's (largest ``err_backend / tol``); the record names that tensor. External
backends (``external`` in the config) are measured the same way; their light is informational.

The measurer only measures: one record per (case, backend) to stdout and to the JSONL file; it judges, caches and skips nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict

import torch

from miniworld_engine.measure import config as cfgmod
from miniworld_engine.modules.primitives import BackendUnsupported

YELLOW_FACTOR = 5.0
#: Human-managed version of this measurer's meaning (the light's rule, the tensors measured, the gold, the inputs): bump it when a change
#: can alter an accuracy record, not for refactors. It is recorded in ``meta`` with the config's version, the measurement settings and the environment.
ACCURACY_VERSION = "1"


# ------------------------------------------------------------------------------------------------------------------------- tensors
def _map_floats(tree, fn):
    """``tree`` (a tensor, or tuples / lists / dicts of them and other values) with ``fn`` applied to every floating tensor."""
    if isinstance(tree, torch.Tensor):
        return fn(tree) if tree.is_floating_point() else tree
    if isinstance(tree, (tuple, list)):
        return type(tree)(_map_floats(t, fn) for t in tree)
    if isinstance(tree, dict):
        return {k: _map_floats(v, fn) for k, v in tree.items()}
    return tree


def _outputs(tree, name: str = "out") -> list[tuple[str, torch.Tensor]]:
    """The floating tensors of a forward result (a tensor, tuples / lists / dicts / dataclasses of them), named by their path."""
    if isinstance(tree, torch.Tensor):
        return [(name, tree)] if tree.is_floating_point() else []
    if isinstance(tree, (tuple, list)):
        return [o for i, t in enumerate(tree) for o in _outputs(t, f"{name}[{i}]")]
    if isinstance(tree, dict):
        return [o for k, t in tree.items() for o in _outputs(t, f"{name}.{k}")]
    if dataclasses.is_dataclass(tree):
        return [o for f in dataclasses.fields(tree) for o in _outputs(getattr(tree, f.name), f"{name}.{f.name}")]
    return []


def _quiet(module: torch.nn.Module) -> None:
    """Switch off the training noise (dropout, coordinate noise): random draws are not numerics."""
    for sub in module.modules():
        if isinstance(sub, torch.nn.Dropout):
            sub.p = 0.0
        for name in ("p_drop", "coordinate_noise"):
            if isinstance(getattr(sub, name, None), float):
                setattr(sub, name, 0.0)


def _rel(x: torch.Tensor | None, gold: torch.Tensor | None) -> float:
    if gold is None and x is None:
        return 0.0
    if gold is None:
        gold = torch.zeros_like(x, dtype=torch.float64)
    x = torch.zeros_like(gold) if x is None else x
    diff = (x.double() - gold.double()).abs().max().item()
    den = gold.double().abs().max().item()
    return diff / den if den > 0 else diff


# ----------------------------------------------------------------------------------------------------------------------------- runs
def _run(module, inputs: dict, *, train: bool, autocast: torch.dtype | None, upstream: dict | None, generator: torch.Generator | None):
    """One call: ``{tensor name: value}`` (outputs; in train also ``grad:<input>`` / ``grad:<parameter>``) and the upstream gradients
    (drawn from ``generator`` on the first, gold, run)."""
    inputs = {k: _map_floats(v, lambda t: t.detach().clone()) for k, v in inputs.items()}
    if train:
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v.requires_grad_(True)
    ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else contextlib.nullcontext()
    with torch.set_grad_enabled(train), ctx:
        outs = _outputs(module(**inputs))
    values = {name: t.detach() for name, t in outs}
    if not train:
        return values, upstream
    if upstream is None:
        upstream = {name: torch.randn(t.shape, generator=generator, device=t.device, dtype=torch.float64) for name, t in outs}
    diff = [(t, upstream[name].to(t.dtype)) for name, t in outs if t.requires_grad]
    torch.autograd.backward([t for t, _ in diff], [g for _, g in diff])
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            values[f"grad:{k}"] = v.grad
    for name, p in module.named_parameters():
        values[f"grad:{name}"] = p.grad
    return values, upstream


def _reference_precision() -> None:
    """PyTorch at its fastest fp32 (TF32 tensor-core products allowed) for the ``pytorch`` reference runs, whatever a library set
    before: the engine's fp32 kernels compute their products in TF32 too, so the reference they are judged against does the same.
    The gold run is float64 (no TF32)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _backend_precision() -> None:
    """The other backends run with PyTorch's own fp32 math (no TF32) wherever they call it, whatever a library set before."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _light(err: float, tol: float) -> str:
    if not math.isfinite(err) or err > YELLOW_FACTOR * tol:
        return "RED"
    return "GREEN" if err <= tol else "YELLOW"


def _verdict(be: dict, pt: dict, gold: dict, fixed: float) -> dict:
    """The per-tensor rule over every tensor; the worst tensor decides."""
    worst = None
    for name, g in gold.items():
        e_be, e_pt = _rel(be.get(name), g), _rel(pt.get(name), g)
        tol = max(e_pt, fixed) if math.isfinite(e_pt) else math.inf
        ratio = e_be / tol if math.isfinite(e_be) else math.inf
        if worst is None or ratio > worst[0]:
            worst = (ratio, name, e_be, e_pt, tol)
    if worst is None:
        return {"status": "error", "reason": "the forward returned no floating tensor"}
    _, name, e_be, e_pt, tol = worst
    return {"status": "measured", "err_backend": e_be, "err_pytorch": e_pt, "tol": tol, "light": _light(e_be, tol), "tensor": name,
            "tensors": len(gold)}


def measure_case(case, cfg, *, device) -> list[dict]:
    """Every backend of the config on one case: one record each (the ``pytorch`` backend measured against itself too)."""
    head = {**cfgmod.describe(case, cfg), "meta": cfgmod.run_meta(cfg, "accuracy", ACCURACY_VERSION)}
    try:
        backends = cfgmod.backends_of(cfg, case)
    except Exception as exc:  # recorded: no such class
        return [{**head, "backend": "pytorch", "external": False, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}]
    fixed = float(cfg.fixed[case.dtype])
    train, autocast = case.mode == "train", cfgmod.autocast(case)
    records = []
    try:
        base = cfgmod.build(case, device=device)
        _quiet(base)
        inputs = cfgmod.inputs(case, base, device=device)
        _reference_precision()
        gold_module = cfgmod.clone(base).double()
        gold_inputs = {k: _map_floats(v, lambda t: t.double()) for k, v in inputs.items()}
        gen = torch.Generator(device=torch.device(device)).manual_seed(case.seed + 1)
        gold, upstream = _run(gold_module, gold_inputs, train=train, autocast=None, upstream=None, generator=gen)
        del gold_module, gold_inputs
        _reference_precision()
        ref, _ = _run(cfgmod.clone(base), inputs, train=train, autocast=autocast, upstream=upstream, generator=None)
    except Exception as exc:  # recorded: the case could not be measured at all
        reason = f"reference: {type(exc).__name__}: {exc}"[:600]
        if os.environ.get("ACCURACY_TRACEBACK"):
            traceback.print_exc()
        torch.cuda.empty_cache()
        # the float64 gold (or the reference) does not fit in memory: the case cannot be judged, it is skipped, not a failure
        status = "skipped" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"
        return [{**head, "backend": b, "external": b in cfg.external, "status": status, "reason": reason} for b in backends]
    for backend in backends:
        rec = {**head, "backend": backend, "external": backend in cfg.external, "fixed": fixed}
        t0 = time.perf_counter()
        module = None
        try:
            module = cfgmod.clone(base).set_backend(backend)
            probe = {k: v.detach().requires_grad_(train) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                     for k, v in inputs.items()}
            ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else contextlib.nullcontext()
            with torch.set_grad_enabled(train), ctx:
                why = module.unsupported(**probe) if backend != "pytorch" else None
            if why is not None:
                raise BackendUnsupported(why)
            (_reference_precision if backend == "pytorch" else _backend_precision)()
            values, _ = _run(module, inputs, train=train, autocast=autocast, upstream=upstream, generator=None)
            rec.update(_verdict(values, ref, gold, fixed))
        except BackendUnsupported as exc:
            rec.update(status="unsupported", reason=str(exc)[:600])
        except Exception as exc:  # recorded, never measured another way
            rec.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:600])
            if os.environ.get("ACCURACY_TRACEBACK"):
                traceback.print_exc()
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        records.append(rec)
        del module
    del base, inputs, gold, ref
    torch.cuda.empty_cache()
    return records


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def _line(rec: dict) -> str:
    depth = f"D{rec['depth']}" if rec.get("depth") else "-"
    head = (f"r{rec['row']:<4} {rec['module']:<34} L{rec['length']:<5} {depth:<6} S{rec['samples']:<3} {rec['dtype']:<8} "
            f"{rec['mode']:<5} {rec['backend']:<9}")
    if rec["status"] != "measured":
        return f"{head} {rec['status']}: {rec.get('reason', '')[:200]}"
    light = rec["light"] + (" (info)" if rec["external"] else "")
    return f"{head} err {rec['err_backend']:.3e}  pytorch {rec['err_pytorch']:.3e}  tol {rec['tol']:.3e}  {light:<14} {rec['tensor']}"


def summary(records: list[dict]) -> str:
    """Counts per (module, backend): GREEN / YELLOW / RED / unsupported / skipped (reference out of memory) / error."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in records:
        counts[(r["module"], r["backend"])][r.get("light") if r["status"] == "measured" else r["status"]] += 1
    cols = ("GREEN", "YELLOW", "RED", "unsupported", "skipped", "error")
    lines = [f"{'module':<34} {'backend':<9} " + " ".join(f"{c:>11}" for c in cols)]
    for (module, backend), c in counts.items():
        lines.append(f"{module:<34} {backend:<9} " + " ".join(f"{c[k]:>11}" for k in cols))
    return "\n".join(lines)


def run(cfg, out_path: str) -> list[dict]:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records: list[dict] = []
    print(f"[accuracy] records -> {out_path}", flush=True)
    print(f"[accuracy] fixed standard {cfg.fixed}; light: tol = max(err_pytorch, fixed), GREEN <= tol < YELLOW <= {YELLOW_FACTOR:g} tol < RED",
          flush=True)
    with open(out_path, "a") as sink:
        for case in cfgmod.cases(cfg):
            for rec in measure_case(case, cfg, device=cfg.device):
                records.append(rec)
                sink.write(json.dumps(rec, default=str) + "\n")
                sink.flush()
                print(_line(rec), flush=True)
    print(f"[accuracy] {len(records)} records -> {out_path}\n{summary(records)}", flush=True)
    return records


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("config", help="a measurement config (.yaml / .toml, see miniworld_engine.measure.config)")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 -- a local-time file name
    p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "miniworld-accuracy"),
                   help="a .jsonl file, or a directory for accuracy-<time>.jsonl (default: under the system temp dir, never the repo)")
    p.add_argument("--module", nargs="+", help="override the config's modules")
    p.add_argument("--backend", nargs="+", help="override the config's backends")
    p.add_argument("--rows", nargs="+", type=int, help="override the config's rows")
    p.add_argument("--lengths", help="override the config's lengths: min | all | 128,256")
    args = p.parse_args(argv)
    cfg = cfgmod.load(args.config)
    lengths = args.lengths if args.lengths in (None, "min", "all") else [int(v) for v in args.lengths.split(",")]
    cfg = cfgmod.replace(cfg, modules=args.module, backends=args.backend, rows=args.rows, lengths=lengths)
    out = args.out if args.out.endswith(".jsonl") else os.path.join(args.out, f"accuracy-{stamp}.jsonl")
    if not torch.cuda.is_available():
        print("[accuracy] no CUDA device", file=sys.stderr)
        return 2
    run(cfg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
