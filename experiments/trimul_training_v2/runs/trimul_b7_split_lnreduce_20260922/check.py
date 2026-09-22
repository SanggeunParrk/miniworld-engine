from pathlib import Path
import sys,importlib.util,json,torch,platform,os
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
with torch.no_grad():
 a,m=H.setup(384);refplan=m.p7;dl,dr,dg,dy=refplan.inputs[:4]
 S=module('atomic_fast',R.parent/'trimul_b7_nextrow_20260921/gate_policy.py');fast=S.Training(a);fast()
 d=refplan.d;refplan.bind(dl,dr,dg,dy,xn=refplan.xn);fast.p7.d=d;fast.p7.mask=refplan.mask;fast.p7.bind(dl,dr,dg,dy,xn=refplan.xn)
 A=module('candidate_plan',R/'role_plan.py');cfg=dict(fast.p7.cfg);cfg.pop('saved');cfg.update(json.loads(os.environ.get('CONFIG','{}')));p=A.Plan(d,dy,dl,dr,dg,refplan.xn,split=True,**cfg);p.mask=refplan.mask;p.bind(dl,dr,dg,dy,refplan.xn)
 plans={'split':fast.p7,'atomic':p};limits=[2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6]
 records=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),L=384,limits=limits,checks=[])
 ref=tuple(x.clone() for x in refplan());torch.cuda.synchronize()
 for n,plan in plans.items():
  out=plan();torch.cuda.synchronize();errs=[H.rel(x,y) for x,y in zip(out,ref)];print('CHECK',n,errs,flush=True);records['checks'].append(dict(name=n,relative_l2=errs));
  if not all(e<=l for e,l in zip(errs,limits)):
   (R/('check-'+os.environ.get('TAG','default')+'.json')).write_text(json.dumps(records,indent=2));raise RuntimeError('accuracy '+n)
 graphs={};outs={}
 for n,plan in plans.items():graphs[n],outs[n]=H.Q.capture_outputs(plan)
 for case in range(3):
  if case:
   dl.mul_(.93);dr.mul_(1.03);refplan.xn.mul_(.96);refplan.mask.copy_(1-refplan.mask);d['w1'].mul_(1.002)
   for w in d['wt'][:4]:w.mul_(1.002)
   d['x'].mul_(.99);d['gi'].mul_(1.01);dg.mul_(.94);dy.mul_(1.02)
  ref=tuple(x.clone() for x in refplan());torch.cuda.synchronize()
  for n,plan in plans.items():
   plan.partw.fill_(float('nan'));plan.partln.fill_(float('nan'));graphs[n].replay();graphs[n].replay();torch.cuda.synchronize()
   errs=[H.rel(x,y) for x,y in zip(outs[n],ref)];valid=all(e<=l for e,l in zip(errs,limits));zero=all(torch.count_nonzero(c).item()==0 for c in plan.counts)
   print('CASE',case,n,valid,zero,errs,flush=True);records['checks'].append(dict(case=case,name=n,relative_l2=errs,valid=valid,counts_zero=zero));assert valid and zero
 t=H.Q.pool([H.Q.paired(graphs,iterations=60) for _ in range(3)])
 records['times']={n:v['median_us'] for n,v in t.items()};records['config']=cfg;print('TIMES',records['times'],flush=True);(R/('check-'+os.environ.get('TAG','default')+'.json')).write_text(json.dumps(records,indent=2))

 for name,plan in plans.items():
  for index,role in enumerate(("dw","dx")):
   g,_=H.Q.capture_outputs(H.single(plan,index));tt=H.Q.pool([H.Q.paired({role:g},iterations=100) for _ in range(2)]);print("ROLE",name,role,tt[role]["median_us"],flush=True);records.setdefault("roles",{})[name+"_"+role]=tt[role]["median_us"]
 (R/("check-"+os.environ.get("TAG","default")+".json")).write_text(json.dumps(records,indent=2))
