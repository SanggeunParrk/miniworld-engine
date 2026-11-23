import csv
import glob
import json
import statistics
import sys
from collections import defaultdict

f8dir, notrdir, refdir, out = sys.argv[1:5]


def load(root, pat):
    c = defaultdict(list)
    st = {}
    for f in glob.glob(f"{root}/F3/rtx5090/{pat}/*/cells.jsonl"):
        for line in open(f):
            r = json.loads(line)
            st[(r["model"], r["n"])] = r["status"]
            if r.get("status") == "ok" and r.get("steady_s"):
                c[(r["model"], r["n"])].append(r["steady_s"])
    return c, st


f8, s8 = load(f8dir, "engine_fp8_rep*")
nt, snt = load(notrdir, "engine_fp8_rep*")
bf, sbf = load(refdir, "engine_rep*")
models = ["af3", "boltz2", "chai1", "intellifold2", "opendde", "openfold3", "protenix1", "protenix2", "rosettafold3"]
ns = [1, 2, 4, 8, 12, 16]
rows = []
md = ["| model | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
for m in models:
    cells = []
    for n in ns:
        k = (m, n)
        if not (nt.get(k) and bf.get(k)):
            cells.append(sbf.get(k, "-") if sbf.get(k) == "oom" else "-")
            rows.append({"model": m, "n": n, "status": sbf.get(k, "missing")})
            continue
        b, a, w = statistics.median(bf[k]), statistics.median(nt[k]), statistics.median(f8[k]) if f8.get(k) else float("nan")
        cells.append(f"**{a / b:.3f}** ({w / b:.3f})")
        rows.append({"model": m, "n": n, "status": "ok", "bf16_s": round(b, 4), "fp8_safe_no_transition_s": round(a, 4), "fp8_safe_with_transition_s": round(w, 4),
                     "ratio_no_transition": round(a / b, 4), "ratio_with_transition": round(w / b, 4), "reps_bf16": len(bf[k]), "reps_fp8_no_tr": len(nt[k]),
                     "reps_fp8_with_tr": len(f8.get(k, []))})
    md.append(f"| {m} | " + " | ".join(cells) + " |")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model", "n", "status", "bf16_s", "fp8_safe_no_transition_s", "fp8_safe_with_transition_s", "ratio_no_transition",
                                       "ratio_with_transition", "reps_bf16", "reps_fp8_no_tr", "reps_fp8_with_tr"])
    w.writeheader()
    w.writerows(rows)
print("\n".join(md))
