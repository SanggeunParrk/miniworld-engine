print("START sweep",flush=True)
from pathlib import Path
import sys,os,importlib.util,json,torch,platform,argparse
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
p=argparse.ArgumentParser();p.add_argument('--configs',default='0:64,1:64,2:64,3:64');p.add_argument('--rounds',type=int,default=3);p.add_argument('--iterations',type=int,default=80);p.add_argument('--cases',type=int,default=5);args=p.parse_args()
tag='reg'+os.environ.get('B7_REG_CAP','96')+'g'+os.environ.get('B7_GROUPS','6')+'u'+os.environ.get('B7_C','10')+'-configs-'+args.configs.replace(',','_').replace(':','-')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
plan=module("integrated_candidate",R/"plan.py")
with torch.no_grad():
 print("SETUP begin",flush=True)
 a,m=H.setup(384);print("SETUP done",flush=True);refplan=m.p7;dl,dr,dg,dy=refplan.inputs[:4];d=refplan.d
 S=module('new_selected',R.parent/'trimul_b7_unconstrained_audit_20260922/selected.py')
 plans={'split':S.make(d,dy,dl,dr,dg,refplan.xn,refplan.mask)}
 os.environ['B7_CONSUMERS']='10';os.environ['B7_PRODUCER_REGS']='64';os.environ['B7_RING_DEPTH']='12'
 os.environ['B7_HW_CLUSTER']='2';os.environ['B7_MULTICAST']='0'
 old=module('current_selected',R.parent/'trimul_b7_ring_cluster_schedule_20260922/plan.py')
 q=old.Plan(d,dy,dl,dr,dg,xn=refplan.xn,clusters=10,mode=52);q.mask=refplan.mask;q.bind(dl,dr,dg,dy,xn=refplan.xn);plans['ring_baseline']=q
 for spec in args.configs.split(','):
  os.environ['B7_CONSUMERS']=os.environ.get('B7_C','10')
  variant,regs=map(int,spec.split(':'));os.environ['B7_PACK']=str(variant);os.environ['B7_PRODUCER_REGS']=str(regs)
  q=plan.Plan(d,dy,dl,dr,dg,xn=refplan.xn,clusters=int(os.environ.get("B7_GROUPS","6")),mode=52)
  q.mask=refplan.mask;q.bind(dl,dr,dg,dy,xn=refplan.xn);plans[f'v{variant}p{regs}']=q
 limits=[2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6]
 rec=dict(source_sha256={n:getattr(q,'source_sha256',{}) for n,q in plans.items()},L=384,args=vars(args),host=platform.node(),gpu=torch.cuda.get_device_name(),limits=limits,checks={},scope='B7-B12, all init/reduction included, same mutated inputs and alternating graph events')
 graphs={};outs={}
 for k,q in plans.items():graphs[k],outs[k]=H.Q.capture_outputs(q)
 for case in range(args.cases):
  if case:
   dl.mul_(.93);dr.mul_(1.03);refplan.xn.mul_(.96);refplan.mask.copy_(1-refplan.mask);d['w1'].mul_(1.002)
   for w in d['wt'][:4]:w.mul_(1.002)
   d['x'].mul_(.99);d['gi'].mul_(1.01);dg.mul_(.94);dy.mul_(1.02)
  ref=tuple(x.clone() for x in refplan());torch.cuda.synchronize();rec['checks'][str(case)]={}
  for n,q in plans.items():
   q.partw.fill_(float('nan'));q.partln.fill_(float('nan'));
   if hasattr(q,'ring'):q.ring.fill_(255)
   eager=tuple(x.clone() for x in q());graphs[n].replay();graphs[n].replay();torch.cuda.synchronize()
   errs=[H.rel(x,y) for x,y in zip(outs[n],ref)];ee=[H.rel(x,y) for x,y in zip(eager,ref)]
   cs=q.counts if isinstance(q.counts,list) else [q.counts];zero=all(torch.count_nonzero(c).item()==0 for c in cs)
   zero=zero and (not hasattr(q,'flags') or torch.count_nonzero(q.flags).item()==0)
   finite=all(torch.isfinite(x).all().item() for x in (*outs[n],*eager))
   exact=all(torch.equal(x,y) for x,y in zip(eager,outs[n]))
   valid=finite and zero and exact and all(e<=l for e,l in zip(errs,limits)) and all(e<=l for e,l in zip(ee,limits))
   rec['checks'][str(case)][n]=dict(valid=valid,graph_relative_l2=errs,eager_relative_l2=ee,counts_flags_zero=zero,finite=finite,eager_graph_exact=exact)
   print('CASE',case,n,valid,errs,flush=True)
  (R/(tag+'.json')).write_text(json.dumps(rec,indent=2))
 good={n:graphs[n] for n in plans if all(c[n]['valid'] for c in rec['checks'].values())};assert len(good)==len(plans)
 rounds=[H.Q.paired(good,iterations=args.iterations) for _ in range(args.rounds)]
 rec['rounds']=rounds;rec['times']={n:v['median_us'] for n,v in H.Q.pool(rounds).items()}
 print('TIMES',rec['times'],flush=True);(R/(tag+'.json')).write_text(json.dumps(rec,indent=2))
