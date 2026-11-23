"""Per module of fp8_pair.py records: cases, fired cases, fired cases slower than 0.99, and the worst fired cases (ratio default / FP8; < 1 = FP8 slower)."""

import collections
import json
import statistics
import sys

by = collections.defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "ok":
            by["(" + r.get("status", "?") + ")"].append(r)
            continue
        by[r["module"]].append(r)
for m, rs in sorted(by.items()):
    ok = [r for r in rs if "ratio_default_over_fp8" in r]
    fired = [r for r in ok if r["fp8_route_fired"]]
    slow = [r for r in fired if r["ratio_default_over_fp8"] < 0.99]
    print(f"{m}: cases {len(rs)}, fired {len(fired)}, fired slower than 0.99: {len(slow)}", end="")
    if fired:
        v = [r["ratio_default_over_fp8"] for r in fired]
        print(f"; fired min {min(v):.3f} median {statistics.median(v):.3f} max {max(v):.3f}")
    else:
        print()
    for r in sorted(slow, key=lambda r: r["ratio_default_over_fp8"])[:12]:
        print(f"    {r['ratio_default_over_fp8']:.3f}  {r['label'][:120]}")
    if not ok and rs:
        print("   ", rs[0].get("reason", "")[:200])
