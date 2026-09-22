"""Diagnostic: end-phase frontier deltas from dual_*_probe stamps (st[3]=consumers done, st[4]=after role ctasync, st[5]=after partln+fence ctasync, st[6]=after grid ticket, st[7]=after reduce, st[8]=after reset)."""
import argparse,statistics
from dual_experiment import *
from check_experiment import change_inputs
a=argparse.ArgumentParser();a.add_argument('--source',default='dual_ws3f_probe48');a.add_argument('--lengths',type=int,nargs='+',default=[384]);a.add_argument('--iters',type=int,default=20);args=a.parse_args()
with torch.no_grad():
 for n in args.lengths:
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n);p=Experiment(d,dy,s,132,2,args.source);p();torch.cuda.synchronize()
  fr=[]
  for _ in range(args.iters):
   p();torch.cuda.synchronize();t=p.timestamps.flatten()[:132*12].reshape(132,12).cpu().tolist()
   t0=min(r[0] for r in t);mx=lambda i:max(r[i] for r in t if r[i]>0)-t0
   fr.append([mx(3)/1e3,mx(4)/1e3,mx(5)/1e3,mx(6)/1e3,mx(7)/1e3,mx(8)/1e3])
  med=[statistics.median(f[i] for f in fr) for i in range(6)]
  print('ENDPHASE L%d frontier(us): consumers_done %.1f | after_role_sync %.1f | after_partln_fence %.1f | after_grid_ticket %.1f | after_reduce %.1f | after_reset %.1f'%(n,*med),flush=True)
