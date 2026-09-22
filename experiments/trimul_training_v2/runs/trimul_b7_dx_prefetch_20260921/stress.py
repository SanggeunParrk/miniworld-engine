from pathlib import Path
import argparse,json,torch
from gate_policy import Training,B
R=Path(__file__).resolve().parent;Q=B.BASE.OLD.Q
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
 a=Q.setup(n);base=B.Training(a);candidate=Training(a);base();candidate();p0=base.p7;p=candidate.p7
 # Keep captured input addresses stable; update their contents and parameter
 # weights, then compare complete fresh B7 outputs after every graph replay.
 dl,dr,dg,dy=p.inputs[:4];p0.bind(dl,dr,dg,dy,xn=p.xn);p0.mask=a['mask'];p0.bind(dl,dr,dg,dy,xn=p.xn)
 g,o=Q.capture_outputs(p);records=[]
 for case in range(3):
  if case:
   dl.mul_(.93);dr.mul_(1.03);dg.mul_(.97);dy.mul_(.99);p.xn.mul_(.96);a['mask'].copy_(1-a['mask']);a['d']['gi'].mul_(1.01)
   # These maps read packed weights directly, with no pointer rebinding.
   a['d']['w1'].mul_(1.002)
  ref=tuple(x.clone() for x in p0())
  for replay in range(5):
   g.replay();torch.cuda.synchronize();assert all(torch.equal(x,y) for x,y in zip(o,ref)),(case,replay)
  for counts in p.counts:assert torch.count_nonzero(counts)==0
  records.append(dict(case=case,replays=5,all_seven_bit_exact=True,counters_zero=True));print('STRESS_PASS',n,case,flush=True)
 (R/('stress-L%d.json'%n)).write_text(json.dumps(records,indent=2))
