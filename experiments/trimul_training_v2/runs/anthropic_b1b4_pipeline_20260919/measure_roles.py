from measure_experiment import *
records=[]
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n);ref=baseline(d,dy,s)
  plans={k:Experiment(d,dy,s,132,2,k) for k in ['dual_ratio12','dual_ratio12_roleprobe']}
  for k,p in plans.items():check(p(),ref)
  times=paired_events({k:capture(p) for k,p in plans.items()})
  allroles=[]
  for _ in range(20):
   p=plans['dual_ratio12_roleprobe'];p();torch.cuda.synchronize();t=p.timestamps.cpu().tolist();assert all(row[1]>=row[0] for row in t)
   allroles.append({role:dict(median_us=statistics.median((row[1]-row[0])/1000 for row in t if row[2]==flag),max_us=max((row[1]-row[0])/1000 for row in t if row[2]==flag),completion_us=max(row[1] for row in t if row[2]==flag)/1000-min(row[0] for row in t)/1000) for role,flag in [('DW',1),('DX',0)]})
  summary={role:{k:statistics.median(x[role][k] for x in allroles) for k in ['median_us','max_us','completion_us']} for role in ['DW','DX']}
  records.append(dict(L=n,times=times,roles=summary,samples=allroles));print('RESULT',n,{k:v['median_us'] for k,v in times.items()},summary,flush=True)
  (R/'roles-profile-results.json').write_text(json.dumps(records,indent=2))
