"""Diagnostic: per-role completion times from dual_ws3e_probe globaltimer stamps (us)."""
import argparse,statistics
from dual_experiment import *
from check_experiment import change_inputs,check
a=argparse.ArgumentParser();a.add_argument('--source',default='dual_ws3e_probe');a.add_argument('--lengths',type=int,nargs='+',default=[384,768]);a.add_argument('--iters',type=int,default=20);args=a.parse_args()
with torch.no_grad():
 for n in args.lengths:
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n);ref=baseline(d,dy,s)
  p=Experiment(d,dy,s,132,2,args.source);print('CHECK',n,check(p(),ref),flush=True)
  dw=int(open(R/(args.source+'.cu')).read().split('#define DW_RATIO ')[1].split('\n')[0]);dx=int(open(R/(args.source+'.cu')).read().split('#define DX_RATIO ')[1].split('\n')[0]);ndw=132*dw//(dw+dx)
  rows=[]
  for _ in range(args.iters):
   p();torch.cuda.synchronize();t=p.timestamps.flatten()[:132*12].reshape(132,12).cpu().tolist();rows.append(t)
  def med(f):return statistics.median(f(t) for t in rows)
  for name,ctas in (('DW',range(0,ndw)),('DX',range(ndw,132))):
   b1=med(lambda t:statistics.median((t[c][1]-t[c][0])/1e3 for c in ctas));iss=med(lambda t:statistics.median((t[c][2]-t[c][0])/1e3 for c in ctas));cons=med(lambda t:statistics.median((t[c][3]-t[c][0])/1e3 for c in ctas));consmax=med(lambda t:max((t[c][3]-t[c][0])/1e3 for c in ctas));end=med(lambda t:max((t[c][4]-t[c][0])/1e3 for c in ctas))
   tiles=n*n//64;per=tiles/len(ctas)
   print(f'ROLE L{n} {name} ctas={len(ctas)} tiles/cta={per:.2f} | B1 warps done {b1:.1f}us ({b1/per:.3f}/tile) | issuer done {iss:.1f} | consumers done med {cons:.1f} max {consmax:.1f} ({cons/per:.3f}/tile) | body end {end:.1f}',flush=True)
