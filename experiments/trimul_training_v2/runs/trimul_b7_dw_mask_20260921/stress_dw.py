from pathlib import Path
import argparse,importlib.util,json,sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
spec=importlib.util.spec_from_file_location('b7_dw_mask',R/'role_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
 a,m=H.setup(n);p0=m.p7;cfg={k:v for k,v in p0.cfg.items() if k!='saved'}
 dl,dr,dg,dy=p0.inputs[:4]
 p=RP.Plan(m.d,dy,dl,dr,dg,xn=p0.xn,split=True,skip_restore=1,overlap=0,ln_mode=2,xn_early=1,mask_hoist=1,dw_mask_early=1,**cfg);p.mask=p0.mask;p.bind(dl,dr,dg,dy,xn=p0.xn)
 # Only the changed dW launch; dX already has full memcheck/racecheck.
 p.units=p.units[:1];p.params=p.params[:1];p.counts=p.counts[:1]
 g,o=H.Q.capture_outputs(lambda:tuple(p()[1:5]));records=[]
 for case in range(3):
  if case:
   dl.mul_(.93);dr.mul_(1.03);p.xn.mul_(.96);a['mask'].copy_(1-a['mask']);a['d']['w1'].mul_(1.002)
  ref=tuple(x.clone() for x in p0()[1:5])
  for replay in range(2):
   g.replay();torch.cuda.synchronize();assert all(torch.equal(x,y) for x,y in zip(o,ref)),(case,replay)
  assert all(torch.count_nonzero(c)==0 for c in p.counts)
  records.append(dict(case=case,replays=2,dW_bit_exact=True,counters_zero=True));print('STRESS_DW_PASS',n,case,flush=True)
 (R/('stress-dw-L%d.json'%n)).write_text(json.dumps(records,indent=2))
