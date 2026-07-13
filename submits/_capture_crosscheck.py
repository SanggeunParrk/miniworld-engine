"""Cross-check the capture builder against the hand-built pilot caches: for every op present in
BOTH the runtime cache (capture output) and the shipped cache (hand-built, committed), compare
top-1 config per shared (dtype|bucket). Broad agreement validates the capture path. A mismatch
on a near-tie is not fatal (measurement noise flips close configs) but is reported. One-off."""
import json
import os
from pathlib import Path

RT = Path(os.environ["RT"])
DST = Path(os.environ["DST"])


def load(root, op):
    files = list((root / op).glob("*.json")) if (root / op).exists() else []
    return {f.name: json.loads(f.read_text()) for f in files}


def top1(entry):
    e = entry[0]
    return (tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in e["kwargs"].items())),
            e["num_warps"], e["num_stages"])


rt_ops = {p.name for p in RT.iterdir()} if RT.exists() else set()
dst_ops = {p.name for p in DST.iterdir()} if DST.exists() else set()
shared_ops = sorted(rt_ops & dst_ops)
if not shared_ops:
    print(f"  no op overlap between runtime ({sorted(rt_ops)}) and shipped ({sorted(dst_ops)})")
    raise SystemExit(0)

for op in shared_ops:
    rt = load(RT, op)
    dst = load(DST, op)
    for gpu in sorted(set(rt) & set(dst)):
        r_entries = rt[gpu].get("entries", {})
        d_entries = dst[gpu].get("entries", {})
        shared = sorted(set(r_entries) & set(d_entries))
        if not shared:
            print(f"  {op} [{gpu}]: no shared buckets (rt={len(r_entries)} shipped={len(d_entries)})")
            continue
        agree = 0
        for key in shared:
            rt_t1, d_t1 = top1(r_entries[key]), top1(d_entries[key])
            same = rt_t1 == d_t1
            agree += same
            mark = "OK " if same else "DIFF"
            rt_ms = r_entries[key][0].get("ms")
            d_ms = d_entries[key][0].get("ms")
            print(f"  {mark} {op} [{key}]: capture={rt_t1[0]} ({rt_ms}) hand={d_t1[0]} ({d_ms})")
        print(f"  => {op} [{gpu}]: {agree}/{len(shared)} shared buckets agree on top-1")
