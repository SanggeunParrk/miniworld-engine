"""Summarize completed/partial result files without treating numerical rejects as wins."""
import collections, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
results=root/'.bench/vast-20260927/results/trimul-d128-config-20260927'
summary={}
status_path=results/"isolated-status.json"
statuses=json.loads(status_path.read_text()) if status_path.exists() else []
status_by_file={"isolated-%s-L%d.json"%(r["label"],r["length"]):r["returncode"] for r in statuses}
for path in sorted(results.glob('*-L*.json')):
 data=json.loads(path.read_text())
 if not isinstance(data,dict) or 'rows' not in data:continue
 valid=[]
 for row in data['rows']:
  if row.get('status')!='measured':continue
  checks=[row.get('strict',{}),row.get('changed_graph',{}),row.get('independent_reference',{})]
  checks+=list(row.get('edge_cases',{}).values())
  if all(v['passed'] for group in checks for v in group.values()):valid.append(row)
 summary[path.name]={'complete':data['complete'],'count':len(data['rows']),
 'statuses':dict(collections.Counter(r.get('status','pending') for r in data['rows'])),
 'execution_status':status_by_file.get(path.name, 'terminated_by_owner' if path.name=='followup-L384.json' else None),
 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'best':[dict(label=r['config']['label'],config=r['config'],speedup=r['speedup'],
 baseline_us=r['times']['baseline']['median_us'],candidate_us=r['times']['candidate']['median_us'])
 for r in sorted(valid,key=lambda r:r['speedup'],reverse=True)[:5]]}
Path(__file__).with_name('summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,d in summary.items():
 print(name,d['complete'],d['count'],d['statuses'])
 for r in d['best'][:3]:print(r['label'],round(r['speedup'],5),round(r['baseline_us'],2),round(r['candidate_us'],2))
