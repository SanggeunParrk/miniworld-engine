"""How many shape buckets does the build matrix actually reach, per op?

Sizing the full sweep needs this and nothing else can supply it. The grid is 205,266 configs, but
an autotuner re-benches its WHOLE list for every distinct value of its ``key=[...]``, so the real
work is sum_op grid(op) * buckets(op).

Two wrong answers were available:
  * the Cartesian product of each kernel's key list -> 25.6M. A gross overcount: N/K/ND/DC move
    together (ND = 4D, DC fixed per model) and several boolean keys only ever take one value here.
  * the committed caches -> 1 bucket per op. That was an artefact, not a measurement: the bucket
    derivation had been dead since fcd3c7a and every entry was recorded as ``any|any``.

So measure it. Run the real build matrix with a ONE-config grid -- the bucket a launch lands in
does not depend on which config is benched, so a trivial grid gives the exact bucket set at a
fraction of the cost -- and count distinct (dtype, bucket) per op.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> int:
    import torch

    from miniworld_engine import settings
    from miniworld_engine.autotune import capture
    from miniworld_engine.autotune.builder import SWITCH_SETTINGS, cases, run_case, units

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cs = list(cases())
    us = units(cs)
    if limit:
        # Stride, do not truncate: units are emitted case-major, so the first N are all one case.
        us = us[:: max(1, len(us) // limit)][:limit]
    print(f"units {len(us)} of {len(units(cs))}", flush=True)
    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name} sm{p.major}{p.minor}", flush=True)

    settings.configure(run_autotune=True, capture=True)
    capture.install()
    by_case = {c.name: c for c in cs}

    ok = err = 0
    fails: collections.Counter = collections.Counter()
    t0 = time.perf_counter()
    for i, u in enumerate(us):
        try:
            # run_case takes the dim INDEX, not the dims dict, and it RETURNS 0/1 rather than
            # raising -- an unsupported width is data, not failure. Counting the call as success
            # reported "ran 300, failed 0" for a run where all 300 skipped and nothing was
            # captured.
            #
            # The SWITCH axis has to be applied too. Ignoring it (as the first version did)
            # collapses 3385 units onto their default branch: every boolean autotune key --
            # USE_DROPOUT, HAS_ROWSCALE, NORMALIZE, ADD_RESIDUAL -- then looks single-valued, and
            # the bucket count comes out an undercount rather than a measurement. p_drop is a
            # module ARGUMENT; everything else is a settings pin.
            p_drop, restore = 0.0, None
            if u.switch == "p_drop":
                p_drop = float(u.value)
            elif u.switch:
                field, parse = SWITCH_SETTINGS[u.switch]
                # configure() returns the PREVIOUS settings. builder.py gets away without
                # restoring because it runs every unit in its own process; this loop does not, so
                # an unrestored pin would leak into every later unit and silently rebuild the
                # wrong side of the switch.
                restore = settings.configure(**{field: parse(str(u.value))})
            try:
                ran = run_case(by_case[u.case], u.length, u.dim_index,
                               train=u.train, p_drop=p_drop, impl=u.impl,
                               dtype=getattr(torch, u.dtype),
                               compute_dtype=getattr(torch, u.compute) if u.compute else None)
            finally:
                if restore is not None:
                    settings.configure(**dataclasses.asdict(restore))
            if ran:
                ok += 1
            else:
                err += 1
                fails[f"{u.case}: skipped"] += 1
        except Exception as exc:                                  # noqa: BLE001
            err += 1
            fails[f"{u.case}: {type(exc).__name__}"] += 1
            if err <= 3:
                traceback.print_exc()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(us)} ok={ok} err={err} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\nran {ok}, failed {err}, {time.perf_counter() - t0:.0f}s")
    for k, v in fails.most_common(10):
        print(f"  {v:4d}  {k}")

    rows = []
    for op, slot in sorted(capture._CAPTURE.items()):             # noqa: SLF001
        buckets = sorted({b for _, b in slot["entries"]})
        dtypes = sorted({d for d, _ in slot["entries"]})
        rows.append({"op": op, "n_buckets": len(buckets), "n_dtypes": len(dtypes),
                     "buckets": buckets[:12]})
    print(f"\nops captured {len(rows)}")
    if rows:
        b = sorted(r["n_buckets"] for r in rows)
        print(f"buckets/op  min {b[0]}  median {b[len(b) // 2]}  max {b[-1]}  "
              f"mean {sum(b) / len(b):.1f}  total {sum(b)}")
        print(f"\n{'op':46s} {'bkt':>4s} {'dt':>3s}  buckets")
        for r in sorted(rows, key=lambda r: -r["n_buckets"])[:20]:
            print(f"{r['op']:46s} {r['n_buckets']:4d} {r['n_dtypes']:3d}  {r['buckets'][:4]}")
    with (Path(__file__).resolve().parents[2] / ".bench/bucketcount.json").open("w") as fh:
        json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
