import collections, json, statistics, sys
by = collections.defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    by[(r["module"], r["status"])].append(r)
for (m, st), rs in sorted(by.items()):
    if st != "ok":
        print(m, st, len(rs), rs[0]["label"][:110], "|", rs[0].get("reason", "")[:90])
        continue
    v = [r["ratio_default_over_fp8"] for r in rs]
    print(f"{m} ok {len(rs)}: min {min(v):.3f} median {statistics.median(v):.3f} max {max(v):.3f}; below 0.97: {sum(x < 0.97 for x in v)}; eager-stats fired {sum(1 for r in rs if r['route_stats'].get('opm',{}).get('calls',0)+r['route_stats'].get('pwa',{}).get('calls',0)+r['route_stats'].get('pwa',{}).get('wide_calls',0) > 0)}")
    for r in sorted(rs, key=lambda r: r["ratio_default_over_fp8"])[:4]:
        print(f"    {r['ratio_default_over_fp8']:.3f} {r['label'][:110]}  stats={ {k: v for k, v in r['route_stats'].items() if k in ('opm','pwa')} }")
