"""Top stall lines from `ncu --page source --csv --print-source=cuda,sass`: source-line rows carry the aggregated counters."""
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Line No")
hdr = rows[hdr_i]
ci = {}
for k, n in enumerate(hdr):
    ci.setdefault(n, k)
stall_cols = [n for n in ci if n.startswith("stall_") and "Not Issued" not in n]
lines = []
fname = rows[0][1].split("/")[-1] if rows and rows[0] and rows[0][0] == "File Name" else "?"
for r in rows[hdr_i + 1:]:
    if r and r[0] == "File Name":
        fname = r[1].split("/")[-1]; continue
    if not r or not r[0] or not r[0].isdigit():
        continue
    if int(r[0]) == 1:
        fid = locals().get("fid", -1) + 1
        fname = f"file{fid}"
    try: s = int(r[ci["Warp Stall Sampling (All Samples)"]])
    except ValueError: continue
    reasons = {}
    for n in stall_cols:
        try: reasons[n[6:]] = int(float(r[ci[n]]))
        except ValueError: pass
    lines.append((s, f"{fname}:{r[0]}", r[1].strip()[:88], reasons))
total = sum(l[0] for l in lines)
print("total samples", total)
for s, ln, txt, rs in sorted(lines, reverse=True)[:int(sys.argv[2])]:
    top = sorted(((v, n) for n, v in rs.items() if v), reverse=True)[:3]
    print(f"{s / total * 100:5.1f}%  {ln:<16s} {txt:88s} " + " ".join(f"{n}:{v * 100 // max(1, s)}%" for v, n in top))
