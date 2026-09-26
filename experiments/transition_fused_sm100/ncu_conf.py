"""SASS instructions with excess shared-memory wavefronts (bank conflicts) from a --print-source=cuda,sass CSV export."""
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Line No")
hdr = rows[hdr_i]
ci = {}
for k, n in enumerate(hdr): ci.setdefault(n, k)
# the SASS half of the table starts at the second "Source" column
sass = [k for k, n in enumerate(hdr) if n == "Source"][1]
exc = ci["L1 Wavefronts Shared Excessive"]; wav = ci["L1 Wavefronts Shared"]; ex_n = ci["Instructions Executed"]
out = []; cur_line = None
for r in rows[hdr_i + 1:]:
    if r and r[0].isdigit(): cur_line = (r[0], r[1].strip()[:70]); continue
    if len(r) <= exc or not r[2].startswith("0x"): continue
    try: e = int(r[exc]); w = int(r[wav])
    except ValueError: continue
    if e > 0: out.append((e, w, r[sass].strip()[:60], cur_line))
tot = sum(o[0] for o in out)
print("total excess wavefronts", tot)
for e, w, s, cl in sorted(out, reverse=True)[:15]:
    print(f"{e:9d} ({e * 100 // tot:2d}%) of {w:9d}  {s:60s}  near line {cl}")
