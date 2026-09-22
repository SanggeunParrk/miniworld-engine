from warp_plan import *
from integrate_front import backward_full
from check_front import LIMITS
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--lengths',type=int,nargs='+',default=[384,768]);args=ap.parse_args()
def bench(gs):
 blocks=[paired(gs) for _ in range(3)];ts={}
 for k in gs:
  vals=sorted(v for b in blocks for v in b[k]['samples_us']);ts[k]=dict(median_us=statistics.median(vals),p90_us=vals[int(.9*(len(vals)-1))],samples_us=vals)
 return ts,blocks
with torch.no_grad():
 for n in args.lengths:
  a=setup(n);p=WarpPlan(a,count=264,splits=13,source='front_twocta_kindwg');old=Plan(a);ref=baseline(a);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
  assert all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items()),es
  ts,bs=bench({'baseline':capture(lambda:baseline(a)),'previous_selected':capture(old),'kindwg':capture(p)})
  (R/f'paired-kindwg-L{n}.json').write_text(json.dumps(dict(L=n,scope='B7-B12',dropout=.25,errors=es,times=ts,blocks=bs),indent=2));print('PAIRED',n,{k:v['median_us'] for k,v in ts.items()},flush=True)
  fullref=C.backward(a['d'],a['s'],a['dy']);out=backward_full(a,p);es=[rel(x,y) for x,y in zip(out,fullref)];assert max(es)<5e-4,es
  ts,bs=bench({'baseline':capture(lambda:C.backward(a['d'],a['s'],a['dy'])),'kindwg':capture(lambda:backward_full(a,p))})
  (R/f'full-kindwg-L{n}.json').write_text(json.dumps(dict(L=n,scope='full backward, unchanged B1-B6 prefix',dropout=.25,errors=es,times=ts,blocks=bs),indent=2));print('FULL',n,{k:v['median_us'] for k,v in ts.items()},flush=True)
