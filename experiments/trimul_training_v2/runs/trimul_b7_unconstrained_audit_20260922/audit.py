from pathlib import Path
import sys,os,importlib.util,json,torch,platform,hashlib
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
with torch.no_grad():
 a,m=H.setup(384);refplan=m.p7;dl,dr,dg,dy=refplan.inputs[:4]
 S=module('audit_fast',R.parent/'trimul_b7_nextrow_20260921/gate_policy.py');fast=S.Training(a);fast()
 d=refplan.d;refplan.bind(dl,dr,dg,dy,xn=refplan.xn);fast.p7.d=d;fast.p7.mask=refplan.mask;fast.p7.bind(dl,dr,dg,dy,xn=refplan.xn)
 plans={'split':fast.p7};sources={}
 variants=[('reduce_one_slice','trimul_b7_split_reduce_20260922',dict(slices=1,dxctas=256)),('parallel_ln_reduce','trimul_b7_split_lnreduce_20260922',dict(slices=1,dxctas=256))]
 os.environ['B7_CONSUMERS']='10'
 for label,folder,opts in variants:
  path=R.parent/folder;pmod=module('audit_'+label,path/'role_plan.py');cfg=dict(fast.p7.cfg);cfg.pop('saved');cfg.update(opts);p=pmod.Plan(d,dy,dl,dr,dg,xn=refplan.xn,split=True,**cfg);p.mask=refplan.mask;p.bind(dl,dr,dg,dy,xn=refplan.xn);plans[label]=p
  sources[label]={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in path.iterdir() if f.suffix in ('.cu','.inc','.py')}
  assert p.inputs[4].data_ptr()==refplan.inputs[4].data_ptr()
 limits=[2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6]
 records=dict(L=384,host=platform.node(),gpu=torch.cuda.get_device_name(),limits=limits,sources=sources,checks={},scope='L384 B7-B12 only; two cooperative launches per plan, including reductions; unchanged accuracy limits')
 graphs={};outs={}
 for k,p in plans.items():graphs[k],outs[k]=H.Q.capture_outputs(p)
 for case in range(5):
  if case:
   dl.mul_(.93);dr.mul_(1.03);refplan.xn.mul_(.96);refplan.mask.copy_(1-refplan.mask);d['w1'].mul_(1.002)
   for w in d['wt'][:4]:w.mul_(1.002)
   d['x'].mul_(.99);d['gi'].mul_(1.01);dg.mul_(.94);dy.mul_(1.02)
  ref=tuple(x.clone() for x in refplan());torch.cuda.synchronize();records['checks'][str(case)]={}
  for n,p in plans.items():
   p.partw.fill_(float('nan'));p.partln.fill_(float('nan'));eager=tuple(x.clone() for x in p());graphs[n].replay();graphs[n].replay();torch.cuda.synchronize()
   errs=[H.rel(x,y) for x,y in zip(outs[n],ref)];eerrs=[H.rel(x,y) for x,y in zip(eager,ref)];counts=p.counts if isinstance(p.counts,list) else [p.counts];zero=all(torch.count_nonzero(c).item()==0 for c in counts)
   if hasattr(p,'flags'):zero=zero and torch.count_nonzero(p.flags).item()==0
   valid=all(e<=l for e,l in zip(errs,limits)) and all(e<=l for e,l in zip(eerrs,limits)) and zero
   records['checks'][str(case)][n]=dict(valid=valid,graph_relative_l2=errs,eager_relative_l2=eerrs,counts_flags_zero=zero,bitwise_eager_graph=all(torch.equal(x,y) for x,y in zip(eager,outs[n])))
   print('CASE',case,n,valid,errs,flush=True)
  (R/'audit-L384.json').write_text(json.dumps(records,indent=2))
 good=[n for n in plans if all(c[n]['valid'] for c in records['checks'].values())];assert 'split' in good
 g={n:graphs[n] for n in good};trials=[H.Q.paired(g,iterations=150) for _ in range(8)];t=H.Q.pool(trials);records['rounds']=trials
 records['timing_inputs']='identical case 4 in one process, alternating CUDA graph timing';records['times']={n:v['median_us'] for n,v in t.items()};records['target_us']=252.
 print('TIMES',records['times'],flush=True);(R/'audit-L384.json').write_text(json.dumps(records,indent=2))

 for n,p in plans.items():
  records.setdefault('role_times',{})[n]={}
  for index,role in enumerate(('dw','dx')):
   graph,_=H.Q.capture_outputs(H.single(p,index));tt=H.Q.pool([H.Q.paired({role:graph},iterations=150) for _ in range(4)]);records['role_times'][n][role]=tt[role]['median_us']
 (R/'audit-L384.json').write_text(json.dumps(records,indent=2));print('ROLE_TIMES',records['role_times'],flush=True)
