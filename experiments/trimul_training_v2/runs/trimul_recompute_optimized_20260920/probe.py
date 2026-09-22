import argparse,gc,json,torch
import plans as P
import compare_cueq_training as Q
from compare_all_training import BoundB1
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('b1','b7'),required=True);ap.add_argument('--length',type=int,default=384);ap.add_argument('--name',default='probe');ap.add_argument('--configs',required=True)
args=ap.parse_args();cfgs=json.loads(args.configs)
with torch.no_grad():
 a=Q.setup(args.length);d=a['d']
 if args.kind=='b1':
  old=BoundB1(d,a['dy'],a['s'],132,2,'dual_ln_prefetch');ref=tuple(t.clone() for t in old());names=('dg','dwg','dt','dgo','dbo','dwp');limits=(0,5e-4,2e-5,5e-6,5e-6,5e-4)
 else:ref=Q.baseline(a);names=('dx','dwl','dwlg','dwr','dwrg','dgi','dbi');limits=(2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6)
 result=[]
 for c in cfgs:
  P.DEFINES=tuple(sorted(c.get('defines',{}).items()));P.build.cache_clear();kw={k:v for k,v in c.items() if k!='defines'}
  try:
   p=P.B1(d,a['dy'],a['s'][0].saved_tensors[11],**kw) if args.kind=='b1' else P.B7(d,a['dy'],a['dl'],a['dr'],a['dg'],**kw)
   out=p();torch.cuda.synchronize();es={k:dict(rel=Q.rel(x,y),finite=bool(torch.isfinite(x).all()),limit=lim) for k,x,y,lim in zip(names,out,ref,limits)}
   valid=all(x['finite'] and x['rel']<=x['limit'] for x in es.values());row=dict(config=c,errors=es,valid=valid)
   if valid:
    g,out=Q.capture_outputs(p);row['time']=Q.pool([Q.paired({'new':g},iterations=100) for _ in range(2)])['new'];del g
   
   if dict(P.DEFINES).get('PROFILE_ROLES'):row['cycles']=p.counts[2:].cpu().tolist()
   del out,p;gc.collect()
  except RuntimeError as e:row=dict(config=c,valid=False,error=str(e))
  print('PROBE',{k:({kk:vv for kk,vv in v.items() if kk!='samples_us'} if k=='time' else v) for k,v in row.items()},flush=True);result.append(row)
  (R/('%s-%s-L%d.json'%(args.name,args.kind,args.length))).write_text(json.dumps(result,indent=2))
