import json,sys,glob,collections
root="/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/n2-merge/src/miniworld_engine/judge/tokens/h100_sm90"
h={}
for f in glob.glob(root+"/*.jsonl"):
    for l in open(f):
        t=json.loads(l);c=t["case"];r=t.get("result",{})
        if t.get("kind")=="speed" and r.get("status")=="ok" and c["backend"]=="cuda" and (r.get("timed_as") or "graph")=="graph":
            h[(c["module"],c["dtype"],c["mode"],c.get("length"),json.dumps(c["kwargs"],sort_keys=True))]=r["median_ms"]
rows=[]
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("backend")!="cuda" or "module" not in r or r.get("status")!="ok": continue
    k=(r["module"],r["dtype"],r["mode"],r.get("length"),json.dumps(r["kwargs"],sort_keys=True))
    hv=h.get(k)
    rows.append((r["module"][:5],r["mode"],json.dumps(r["kwargs"],sort_keys=True).replace(chr(34),"")[:70],r.get("length"),round(r["median_ms"]*1000,1),round(hv*1000,1) if hv else None, round(r["median_ms"]/hv,3) if hv else None))
for x in sorted(rows,key=lambda x:(x[0],x[1],x[2],x[3])): print(*x)
