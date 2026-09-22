from pathlib import Path
import argparse,json,hashlib,torch
from policy import Training,B,BASE
R=Path(__file__).resolve().parent;Q=BASE.OLD.Q
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
with torch.no_grad():
 a=Q.setup(args.length);base=B.Training(a);candidate=Training(a)
 _,k=base.forward();base.backward(k)
 # Bind both B1 plans to identical forward buffers.
 candidate.p1.bind(a['dy'],k[1],k[3]);candidate.p1.d['x'].copy_(k[-1])
 p=candidate.p1;records=[]
 graph,outs=Q.capture_outputs(p)
 for case in range(3):
  if case==1:
   a['dy'].mul_(.91);p.d['go'][0]=0;p.d['ds'].copy_(p.d['ds'].roll(1,0))
  if case==2:
   p.xhat.mul_(.97);a['dy'].add_(.003);p.d['go'].mul_(1.03)
  ref=tuple(t.clone() for t in base.p1())
  p.partw.fill_(float('nan'));p.partln.fill_(float('nan'))
  eager=tuple(t.clone() for t in p())
  for _ in range(3):graph.replay()
  torch.cuda.synchronize()
  exact=[torch.equal(x,y) for x,y in zip(outs,ref)]
  eager_exact=[torch.equal(x,y) for x,y in zip(eager,ref)]
  assert all(exact+eager_exact) and torch.count_nonzero(p.counts).item()==0,(case,exact,eager_exact)
  assert all(torch.isfinite(t).all().item() for t in outs)
  records.append(dict(case=case,exact=exact,eager_exact=eager_exact,counters_zero=True))
 cubin=Path(p.k.unit.cubin_path)
 result=dict(length=args.length,checks=records,cubin=str(cubin),sha256=hashlib.sha256(cubin.read_bytes()).hexdigest())
 (R/args.output).write_text(json.dumps(result,indent=2));print('PROBE_PASS',args.length,result['sha256'],flush=True)
