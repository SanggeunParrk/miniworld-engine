"""sass_top.py <tag> [reason] [n]: top SASS instructions by warp stall samples (optionally of one reason: wait, short_sb, math, mio, barrier, long_sb)"""
import csv, os, sys, collections
tag = sys.argv[1]; reason = sys.argv[2] if len(sys.argv) > 2 else None; n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = list(csv.reader(open(os.path.expanduser(f"~/.cache/miniworld-a100/ncu/{tag}.sass.csv"))))
hd = rows[1]; data = [r for r in rows[2:] if len(r) > 5]
iS, iE, iA = hd.index("Source"), hd.index("Instructions Executed"), hd.index("Address")
col = hd.index("stall_" + reason) if reason else hd.index("Warp Stall Sampling (All Samples)")
num = lambda x: float(x) if x not in ("", None) else 0.0
tot = sum(num(r[col]) for r in data)
print(f"total samples ({reason or 'all'}): {tot:.0f}")
by = collections.Counter()
for r in data:
    tok = r[iS].split(); op = (tok[1] if tok and tok[0].startswith("@") else (tok[0] if tok else "?")).split(".")[0]
    by[op] += num(r[col])
print("by opcode:", ", ".join(f"{o} {100*v/tot:.1f}%" for o, v in by.most_common(10)))
for r in sorted(data, key=lambda r: -num(r[col]))[:n]:
    print(r[iA][-5:], r[iE].rjust(9), f"{100*num(r[col])/tot:5.1f}%", r[iS][:90])
