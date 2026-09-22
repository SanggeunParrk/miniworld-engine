from pathlib import Path
import importlib.util,sys,json,torch,os
R=Path(__file__).resolve().parent

def load(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
P=load('ln_validation_policy',R.parent/'trimul_full_latest_20260922/policy.py');Q=P.Q
J=load('ln_debug_joint',R/'joint_debug/plan.py');S=load('ln_debug_split',R/'split_debug/role_plan.py');F=load('ln_debug_fixed',R/'fixed_debug/plan.py')
def rel(x,y):return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))
def clone(o):return o[0].clone(),tuple(t.clone() for t in o[1])
record=dict(job=os.environ.get('SLURM_JOB_ID'),cases={})
def save():(R/'diagnose.json').write_text(json.dumps(record,indent=2))
with torch.no_grad():
 a=Q.setup(384);d=a['d'];old=P.Historical(a);new=P.Latest(a);contract=P.Regression(a)
 print('REF_P7',type(old.p7),old.p7.cfg,flush=True)
 for name,m,impl in [('split',old,S),('joint',new,J)]:
  p=m.p7;dl,dr,dg,dy=p.inputs[:4]
  if name=='split':dbg=impl.Plan(d,dy,dl,dr,dg,xn=p.xn,split=True,**{k:v for k,v in p.cfg.items() if k!='saved'})
  else:
   os.environ.update(P.SELECTION['environment']);dbg=impl.Plan(d,dy,dl,dr,dg,xn=p.xn,**P.SELECTION['kwargs'])
  dbg.mask=p.mask
  if name=='split':old_debug=P.Regression(a);old_debug.p7=dbg
  else:new_debug=P.Latest(a);new_debug.p7=dbg
 fixed=P.Latest(a);p=fixed.p7;dl,dr,dg,dy=p.inputs[:4];os.environ.update(P.SELECTION['environment']);fp=F.Plan(d,dy,dl,dr,dg,xn=p.xn,**P.SELECTION['kwargs']);fp.mask=p.mask;fixed.p7=fp
 models={'contract':contract,'split':old,'joint':new,'split_debug':old_debug,'joint_debug':new_debug,'fixed_debug':fixed}
 for case in range(3):
  if case:
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);d['wp'].mul_(.93);d['go'].mul_(1.07);d['go'][0]=0;d['bo'].add_(.017);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
  outs={name:clone(m()) for name,m in models.items()};row={}
  for name,o in outs.items():
   row[name]=dict(dgamma_vs_split=rel(o[1][7],outs['contract'][1][7]),dbeta_vs_split=rel(o[1][8],outs['contract'][1][8]),dx_vs_split=rel(o[1][0],outs['contract'][1][0]))
  x1=old_debug.p7.debug_dxn;z1=old_debug.p7.debug_xhat;x2=new_debug.p7.debug_dxn;z2=new_debug.p7.debug_xhat
  row['fixed_intermediate']=dict(dxn=rel(fixed.p7.debug_dxn,x1),dxn_changed=int((fixed.p7.debug_dxn!=x1).sum()),xhat=rel(fixed.p7.debug_xhat,z1))
  row['intermediate']=dict(dxn=rel(x2,x1),dxn_changed=int((x2!=x1).sum()),xhat=rel(z2,z1),xhat_changed=int((z2!=z1).sum()))
  for name,m in [('split_debug',old_debug),('joint_debug',new_debug),('fixed_debug',fixed)]:
   p=m.p7;x=p.debug_dxn.double();z=p.debug_xhat.double();dg=(x*z).sum(0);db=x.sum(0)
   row[name].update(dgamma_vs_fp64sum=rel(outs[name][1][7],dg),dbeta_vs_fp64sum=rel(outs[name][1][8],db))
   row[name]['fp64sum_vs_split']=dict(dgamma=rel(dg,outs['contract'][1][7]),dbeta=rel(db,outs['contract'][1][8]))
   del x,z
  idx=(x2!=x1).nonzero();row['changed_samples']=[dict(row=int(i),c=int(j),split=float(x1[i,j]),joint=float(x2[i,j])) for i,j in idx[:20]]
  torch.save(dict(index=idx,split=x1[idx[:,0],idx[:,1]],joint=x2[idx[:,0],idx[:,1]]),R/('changed-%d.pt'%case))
  record['cases'][str(case)]=row;save();print('CASE',case,json.dumps(row),flush=True)
 record['complete']=True;save()
