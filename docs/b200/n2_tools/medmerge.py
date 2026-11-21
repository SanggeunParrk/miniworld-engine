import json, sys, statistics, collections
out = sys.argv[1]; files = sys.argv[2:]
recs = collections.defaultdict(list); first = {}
for f in files:
    for line in open(f):
        r = json.loads(line)
        if "median_ms" not in r or r.get("status") != "ok": continue
        k = r["label"] + "|" + str(r.get("backend"))
        recs[k].append(r["median_ms"]); first.setdefault(k, r)
with open(out, "w") as fh:
    for k, v in recs.items():
        r = dict(first[k]); r["median_ms"] = statistics.median(v); r["n_runs"] = len(v); fh.write(json.dumps(r) + "\n")
print(len(recs), "cases merged from", len(files), "files")
