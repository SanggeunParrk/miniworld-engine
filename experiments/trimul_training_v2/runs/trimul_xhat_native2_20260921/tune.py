from pathlib import Path
import argparse,gc,itertools,json,torch
import bench as B
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args();n=args.length
records=[]
def record(x):
 records.append(x);(R/('tune-L%d.json'%n)).write_text(json.dumps(records,indent=2));print('TUNE',x,flush=True)
def times(fn,control):
 g,_=B.Q.capture_outputs(fn);r=B.Q.paired({'control':control,'candidate':g},iterations=60);del g;return {k:v['median_us'] for k,v in r.items()}
with torch.no_grad():
 a=B.Q.setup(n);d=a['d'];m=B.Replacement(a,1);m.audit=True;y,k=m.forward();tri=m.audit_tri;m.audit=False
 yc,sc=B.RC.output(d,tri,ln=3,stats=2,method=1);reference=(yc.clone(),{k:v.clone() for k,v in sc.items() if v is not None})
 control,_=B.Q.capture_outputs(lambda:B.RC.output(d,tri,ln=3,stats=2,method=1))
 for tiles,slots,acc,regs,serial in itertools.product(((1,64),(2,64),(1,128)),(2,4,6,8),(1,2),(0,1),(0,1)):
  cfg=(*tiles,slots,acc,regs,serial)
  try:B.RC.I.k3_smem(cfg)
  except ValueError:continue
  try:
   y,s=B.RC.output(d,tri,ln=3,stats=2,method=1,cfg=cfg);torch.cuda.synchronize()
   exact=torch.equal(y,reference[0]) and all(torch.equal(s[key],val) for key,val in reference[1].items())
   if not exact:record(dict(scope='k3',config=cfg,valid=False));continue
   t=times(lambda:B.RC.output(d,tri,ln=3,stats=2,method=1,cfg=cfg),control)
   record(dict(scope='k3',config=cfg,valid=True,times=t,ratio=t['candidate']/t['control']))
  except Exception as e:record(dict(scope='k3',config=cfg,valid=False,error=str(e)[-1500:]))
 best=min((r for r in records if r['scope']=='k3' and r['valid']),key=lambda r:r['ratio']);(R/('fwd-selected-L%d.json'%n)).write_text(json.dumps(best,indent=2))
 del control,reference,sc,yc;gc.collect()
 m.backward(k);ref=tuple(t.clone() for t in m.p1());control,_=B.Q.capture_outputs(m.p1)
 for count,unroll in itertools.product((96,112,120,128,132),(1,2,4,8)):
  cfg=dict(count=count,defines=dict(GATE_PHASE=1,LOWREG=1,STREAM_LN=1,B1_STREAM_AFFINE_UNROLL=unroll,XHAT_FP32=1))
  try:
   p=B.RP.Plan(dict(d,x=k[-1]),a['dy'],k[1],k[3],**cfg);v=p();torch.cuda.synchronize()
   errs=[float((x.float()-r.float()).norm()/r.float().norm().clamp_min(1e-20)) for x,r in zip(v,ref)]
   exact=[torch.equal(x,r) for x,r in zip(v,ref)];valid=exact[0] and exact[2] and max(errs)<=5e-4
   if not valid:record(dict(scope='b1',config=cfg,valid=False,errors=errs));continue
   t=times(p,control);record(dict(scope='b1',config=cfg,valid=True,errors=errs,times=t,ratio=t['candidate']/t['control'],cubin=p.k.unit.cubin_path))
  except Exception as e:record(dict(scope='b1',config=cfg,valid=False,error=str(e)[-1500:]))
 best=min((r for r in records if r['scope']=='b1' and r['valid']),key=lambda r:r['ratio']);(R/('bwd-selected-L%d.json'%n)).write_text(json.dumps(best,indent=2))
 print('TUNING_DONE',n,flush=True)
