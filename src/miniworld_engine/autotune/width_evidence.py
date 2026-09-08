"""Which of a row's declared widths actually reach its cache key.

`op_units` crosses every kernel with a ladder of channel widths, on the assumption that a
different width is a different bucket and therefore a unit worth building. For 17 of the 91 triton
ops that is false, and a sweep of every (op, width) the build plans measured it directly: they file
EVERY declared width into ONE bucket. 274 of the build's 2,079 units -- 13.2% -- were re-timing a
bucket a sibling unit had already timed, and each overwrote the last.

Two different reasons, and neither is visible from the registry row:

* the kernel's key carries no channel width at all. `gated_projection_gate_flat_triton` keys on
  `token_key(M)` and nothing else, and `drivers/gated_projection.py` never calls `driver_width` --
  so its six declared widths are six identical units.
* the key carries one, but the driver cannot move it. `triangle_attention/triton/atomic.py` raises
  `ValueError("Only support D=32")` for any other head dim, and `transition_b2b` is dispatched only
  at `K <= 128`, so driving either at the swept width would tune a bucket the dispatcher never
  routes to it.

This is a MEASUREMENT, not a declaration, which is why it is a generated file rather than a
registry column: the question "does this width reach the key" is answered by launching the driver
and reading the key back, and any hand-written answer would be a guess that drifts. `dev buckets`
regenerates it; `op_units` reads it to drop the duplicate units; the default when an op is absent
is the full ladder, so a new kernel is never silently narrowed.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Beside the registry, because it says the same kind of thing about the same rows.
EVIDENCE = Path(__file__).resolve().parent.parent / "kernels" / "width_evidence.json"


def load(path: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """``{op: {width: [entry key, ...]}}`` as `dev buckets` last measured it, or ``{}``."""
    p = path or EVIDENCE
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def collapses(op: str, widths: tuple[int, ...], data: dict | None = None) -> bool:
    """Do ALL of `widths` file into the same bucket for `op`?

    Every width has to have been measured. A partially measured op keeps its full ladder: the
    unmeasured rung is exactly the one that might be the distinct bucket, and dropping it on the
    strength of the others is how a hole gets built in.
    """
    per = (data if data is not None else load()).get(op)
    if not per or len(widths) < 2:
        return False
    seen = []
    for w in widths:
        got = per.get(str(w))
        if not got:
            return False
        seen.append(tuple(sorted(got)))
    return len(set(seen)) == 1


def measure(ops: list[tuple[str, str, tuple[int, ...]]], out: Path | None = None) -> dict:
    """Launch each (op, driver ref, widths) and record the entry key its launch files under.

    One import per width, in this process, because `MINIWORLD_DRIVER_WIDTH` is read when
    `drivers/__init__` is imported and the driver modules close over the result -- the same route
    `build all` takes. The launch runs for real: the key is a packed integer built from constexprs
    that only exist once the launcher has built them, so there is nothing to read without one.
    """
    import collections
    import importlib
    import os
    import sys
    import traceback

    from miniworld_engine.autotune import capture

    seen: dict[str, set] = collections.defaultdict(set)
    from triton.runtime.autotuner import Autotuner
    prev = Autotuner.run

    def run(self, *args, **kwargs):
        try:
            op = capture._op_name(self)
            if op:
                nargs = dict(self.nargs) if getattr(self, "nargs", None) else None
                if nargs is None:
                    nargs = dict(zip([p.name for p in self.fn.params], args, strict=False))
                    nargs.update(kwargs)
                seen[op].add(capture._entry_key(self, kwargs, nargs))
        except Exception:      # a probe must never be the reason a build fails
            pass
        return prev(self, *args, **kwargs)

    Autotuner.run = run
    result: dict[str, dict[str, list[str]]] = {}
    try:
        for op, ref, widths in ops:
            mod_name, fn_name = ref.split(":", 1)
            if not mod_name.startswith("miniworld_engine"):
                mod_name = "miniworld_engine.kernels." + mod_name
            for w in sorted(widths):
                os.environ["MINIWORLD_DRIVER_WIDTH"] = str(w) if w else ""
                for m in [m for m in sys.modules if m.startswith("miniworld_engine.kernels.drivers")]:
                    del sys.modules[m]
                seen.clear()
                try:
                    getattr(importlib.import_module(mod_name), fn_name)()
                except Exception as exc:
                    print(f"  probe FAILED {op} width={w}: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    continue
                got = sorted(seen.get(op, ()))
                if got:
                    result.setdefault(op, {})[str(w)] = got
                print(f"  {op} width={w} -> {got}", flush=True)
    finally:
        Autotuner.run = prev
    if out is not None:
        out.write_text(json.dumps(dict(sorted(result.items())), indent=1, sort_keys=True) + "\n")
    return result
