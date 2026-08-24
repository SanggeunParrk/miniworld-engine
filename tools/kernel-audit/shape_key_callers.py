"""Which call sites hand `length_of` an ALREADY-FLATTENED shape?

`length_of` returns shape[-2] and its docstring states the contract outright: it must be given the
activation BEFORE it is flattened, and "an inner launcher that only receives the flattened (M, D)
matrix therefore CANNOT call this; its caller must compute the key and pass it down."

A violating call site does not fail. It returns M instead of L, and for a pair kernel M = L*L, so
`both_key(L*L)` clamps to the top bucket at any L >= 91: every shape lands in one entry. Measured
on gated_projection_gate_triton, eight units at L=128..8192 all recorded shape_key=8192.

Wrap `length_of`, run every driver, and record (caller, ndim, argument, result) so the violations
are a list of file:line rather than a guess.
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _rebind(mod: Any, fn: Callable[..., Any]) -> None:
    """Install `fn` as `mod.length_of`.

    A deliberate monkeypatch, and the reason it needs a helper: a module attribute has a
    fixed declared type, so `mod.length_of = spy` is something a type checker is right to
    reject. Replacing it is the whole point of this probe, so the dynamic write lives here
    and nowhere else.
    """
    mod.length_of = fn


def main() -> int:

    from miniworld_engine import settings
    from miniworld_engine.autotune import shape_key
    from miniworld_engine.autotune.builder import _run_one_driver

    seen: dict[tuple, dict] = {}
    orig = shape_key.length_of

    def spy(shape):
        dims = tuple(shape)
        # the frame that CALLED length_of, skipping this wrapper
        fr = next((f for f in reversed(traceback.extract_stack()[:-1])
                   if "shape_key.py" not in f.filename), None)
        where = f"{Path(fr.filename).relative_to(REPO / 'src')}:{fr.lineno}" if fr else "?"
        out = orig(shape)
        rec = seen.setdefault((where, len(dims)),
                              {"where": where, "ndim": len(dims), "n": 0, "shapes": set()})
        rec["n"] += 1
        rec["shapes"].add(dims[:4])
        return out

    _rebind(shape_key, spy)
    # the kernels imported `length_of` by value, so rebind it in every module that holds one
    for name, mod in list(sys.modules.items()):
        if name.startswith("miniworld_engine") and getattr(mod, "length_of", None) is orig:
            _rebind(mod, spy)
    settings.configure(run_autotune=True, capture=True)

    def _rebind_all():
        # a first call imports the kernel module, which binds length_of by value then
        for name, mod in list(sys.modules.items()):
            if name.startswith("miniworld_engine") and getattr(mod, "length_of", None) is orig:
                _rebind(mod, spy)

    mode = sys.argv[1] if len(sys.argv) > 1 else "driver"
    ran = 0
    if mode == "module":
        # The PRODUCTION path: modules, not drivers. Which one violates the contract is the whole
        # question -- a driver that hands over a flattened (M, D) says nothing about what the
        # module does, and the fix is in a completely different place depending on the answer.
        import torch as _t

        from miniworld_engine.autotune.builder import cases, run_case
        cs = list(cases())
        for i, c in enumerate(cs, 1):
            _rebind_all()
            for li, length in enumerate(c.lengths[:2]):
                try:
                    ran += run_case(c, length, 0, train=(li == 0 and c.train),
                                    dtype=_t.bfloat16)
                except Exception as exc:  # noqa: PERF203 -- a case that cannot run is reported, not fatal
                    print(f"    {c.name} L={length}: {type(exc).__name__}", flush=True)
            if i % 5 == 0:
                print(f"  {i}/{len(cs)} cases, {len(seen)} call sites", flush=True)
        print(f"\nmodule runs that succeeded: {ran}")
    else:
        reg = [r for r in csv.DictReader(
            (REPO / "src/miniworld_engine/kernels/registry.csv").open())
            if r["backend"] == "triton" and (r["driver"] or "").strip()]
        for i, r in enumerate(reg, 1):
            _rebind_all()
            ran += _run_one_driver(r["kernel"])
            if i % 20 == 0:
                print(f"  {i}/{len(reg)} drivers, {len(seen)} call sites", flush=True)
        print(f"\ndrivers that ran: {ran}/{len(reg)}")

    rows = sorted(seen.values(), key=lambda d: (d["ndim"], d["where"]))
    bad = [d for d in rows if d["ndim"] < 3]
    print(f"length_of call sites reached: {len(rows)}")
    print(f"  ndim >= 3 (pre-flatten, correct): {len(rows) - len(bad)}")
    print(f"  ndim == 2 (ALREADY FLATTENED -> returns M, not L): {len(bad)}")
    for d in bad:
        print(f"    {d['where']:60s} calls={d['n']:4d} shapes={sorted(d['shapes'])[:2]}")
    (REPO / f".bench/shape_key_callers-{mode}.json").write_text(json.dumps(
        [{**d, "shapes": sorted(d["shapes"])} for d in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
