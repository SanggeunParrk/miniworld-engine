from pathlib import Path
import argparse,gc,json,sys,torch
import role_plan as R
import compare_cueq_training as Q
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'trimul_ln_only_save_20260921'));import ln_save_core as LN
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--configs',required=True);ap.add_argument('--name',default='probe');args=ap.parse_args()
with torch.no_grad():
 a=Q.setup(args.length);d=a['d'];_,_,xn,_=LN.forward(d,1);ref=Q.baseline(a);rows=[]
 for config in json.loads(args.configs):
  c=dict(config);saved=c.pop('saved',False)
  try:
   plan=R.Plan(d,a['dy'],a['dl'],a['dr'],a['dg'],xn=xn if saved else None,**c);out=plan();torch.cuda.synchronize()
   names=('dx','dwl','dwlg','dwr','dwrg','dgi','dbi');limits=(2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6)
   es={k:dict(rel=Q.rel(x,y),finite=bool(torch.isfinite(x).all()),limit=lim) for k,x,y,lim in zip(names,out,ref,limits)};valid=all(v['finite'] and v['rel']<=v['limit'] for v in es.values());row=dict(config=config,valid=valid,errors=es)
   if valid:
    g,_=Q.capture_outputs(plan);row['time']=Q.pool([Q.paired({'new':g},iterations=100) for _ in range(2)])['new'];del g
   del out,plan;gc.collect()
  except RuntimeError as e:row=dict(config=config,valid=False,error=str(e))
  print('PROBE',{k:({kk:vv for kk,vv in v.items() if kk!='samples_us'} if k=='time' else v) for k,v in row.items()},flush=True);rows.append(row);(P/('%s-L%d.json'%(args.name,args.length))).write_text(json.dumps(rows,indent=2))
