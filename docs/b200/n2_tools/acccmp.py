import json, sys
def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("backend") != "cuda": continue
        d[(r["label"], r.get("tensor", r.get("name", "")))] = r
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
print("records", len(a), len(b)); r0 = next(iter(a.values())); print("keys", sorted(r0.keys())[:30])
worse = 0
for k in a:
    ra, rb = a[k], b.get(k)
    if rb is None: continue
    ea, eb = ra.get("err"), rb.get("err")
    if ea is None or eb is None: continue
    tol = ra.get("tol") or rb.get("tol")
    flag = ""
    if ea > eb * 1.05 + 1e-12: flag = "worse"; worse += 1
    if ea != eb or flag:
        print(f"{k[0][:100]:100s} {k[1]:12s} new {ea:.3e} base {eb:.3e} tol {tol} {ra.get('status')}/{rb.get('status')} {flag}")
print("worse", worse, "of", len(a))
