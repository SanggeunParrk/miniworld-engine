# cmp.py NEW.jsonl BASE.jsonl : per-case median ms new vs base (cuda), summary by (dtype, mode) and the cases beyond +-2 %
import collections, json, math, sys
def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("backend") != "cuda" or r.get("status") != "ok": continue
        d[r["label"]] = r
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
rows = []
for k, r in a.items():
    if k in b:
        rows.append((r["dtype"], r["mode"], r["length"], r["kwargs"].get("features"), r["kwargs"].get("swa_window_size"), r["kwargs"].get("token_pool"), r["median_ms"], b[k]["median_ms"]))
g = collections.defaultdict(list)
for t in rows: g[(t[0], t[1])].append(t[6] / t[7])
print("group: n  geomean(new/base)  best  worst")
for k, xs in sorted(g.items()):
    print(f"{k[0]:9s} {k[1]:5s} {len(xs):3d}  {math.exp(sum(map(math.log, xs)) / len(xs)):.3f}  {min(xs):.3f}  {max(xs):.3f}")
print("cases beyond +-2 %% (new/base):")
for t in sorted(rows, key=lambda t: t[6] / t[7]):
    r = t[6] / t[7]
    if r > 1.02 or (r < 0.94 and t[1] == "train" and t[2] in (3072, 4096, 6144, 8192)):
        print(f"  {t[0]:8s} {t[1]:5s} L{t[2]:<5d} {t[3]:9s} w{t[4]:<8d} {t[5]:6s} new {t[6]:.4f} base {t[7]:.4f} ratio {r:.3f}")
