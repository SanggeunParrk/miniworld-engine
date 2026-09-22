from pathlib import Path
import argparse,importlib.util,json,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
spec=importlib.util.spec_from_file_location('b7_dw_mask',R/'role_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
 a,m=H.setup(n);p0=m.p7;cfg={k:v for k,v in p0.cfg.items() if k!='saved'};cfg.update(resident=0,dxctas=264)
 dl,dr,dg,dy=p0.inputs[:4]
 p=RP.Plan(m.d,dy,dl,dr,dg,xn=p0.xn,split=True,skip_restore=1,overlap=0,ln_mode=2,xn_early=0,mask_hoist=1,dw_mask_early=1,weight_prefetch=0,dx_pc=1,dw_pair=1,nextrow=2 if n==768 else 0,**cfg);p.mask=p0.mask;p.bind(dl,dr,dg,dy,xn=p0.xn)
 # Only the changed dX launch; dW uses the validated early-mask path.
 p.units=p.units[1:];p.params=p.params[1:];p.counts=p.counts[1:]
 def run():
  values=p();return tuple(values[i] for i in (0,5,6))
 g,o=H.Q.capture_outputs(run);records=[]
 for case in range(3):
  if case:
   dl.mul_(.93);dr.mul_(1.03);p.xn.mul_(.96);a['mask'].copy_(1-a['mask']);a['d']['w1'].mul_(1.002)
   # Keep the packed front weights and their transposed views consistent.
   for wt in a['d']['wt'][:4]:wt.mul_(1.002)
   a['d']['x'].mul_(.99);a['d']['gi'].mul_(1.01);dg.mul_(.94);dy.mul_(1.02)
  refall=p0();ref=tuple(refall[i].clone() for i in (0,5,6))
  for replay in range(2):
   g.replay();torch.cuda.synchronize();assert torch.equal(o[0],ref[0]) and all(H.rel(x,y)<=5e-6 for x,y in zip(o[1:],ref[1:])),(case,replay,[H.rel(x,y) for x,y in zip(o,ref)])
  assert all(torch.count_nonzero(c)==0 for c in p.counts)
  records.append(dict(case=case,replays=2,dx_bit_exact=True,LN_relative_l2=[H.rel(x,y) for x,y in zip(o[1:],ref[1:])],counters_zero=True));print('STRESS_DX_PASS',n,case,flush=True)
 (R/('stress-dx-L%d.json'%n)).write_text(json.dumps(records,indent=2))
