"""Fixed-contract replay validation; new outputs never replace earlier evidence."""
from ring_plan import *
import argparse
LIMITS=dict(dx=2e-5,dWL=5e-4,dWLg=5e-4,dWR=5e-4,dWRg=5e-4,dgamma=5e-6,dbeta=5e-6)
def check(p,a):
 out=p.outputs;ref=baseline(a);torch.cuda.synchronize();e=errors(out,ref)
 assert all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in e.items()),e
 assert torch.equal(p.counts,torch.zeros_like(p.counts)),p.counts[:2]
 return e
def change(a,seed):
 torch.manual_seed(seed)
 for k in ('dl','dr','dg','dy'):a[k].copy_(torch.randn_like(a[k]))
 a['mask'].copy_((torch.rand_like(a['mask'].float())>.2).to(a['mask'].dtype))
 # Saved values update in place. Preserve valid positive inverse std.
 a['mu'].add_(torch.randn_like(a['mu'])*.001);a['rs'].mul_(1.001)
 a['d']['gi'].add_(torch.randn_like(a['d']['gi'])*.001)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='front_ring96_cache3');ap.add_argument('--lengths',type=int,nargs='+',default=[64,384,768]);ap.add_argument('--counts',type=int,nargs='+',default=[264]);ap.add_argument('--parts',type=int,nargs='+',default=[2]);ap.add_argument('--dropouts',type=float,nargs='+',default=[0,.25]);ap.add_argument('--splits',type=int,default=20);ap.add_argument('--output',required=True);args=ap.parse_args();records=[]
 with torch.no_grad():
  for n in args.lengths:
   for drop in args.dropouts:
    a=setup(n,drop)
    for count in args.counts:
     for part in args.parts:
      sp=max(1,args.splits*count//264);p=RingPlan(a,count,sp,part,source=args.source)
      p();torch.cuda.synchronize();es=[check(p,a)]
      g=capture(p)
      for replay in range(2):
       change(a,20260921+n+replay);g.replay();torch.cuda.synchronize();es.append(check(p,a))
      # For direct-weight maps this is a real in-place parameter update with
      # no W_stack repacking. Packed-control paths explicitly refresh.
      a['wlg'].mul_(1.001);a['wl'].mul_(.999);p.refresh_weights()
      g.replay();torch.cuda.synchronize();es.append(check(p,a))
      rec=dict(L=n,dropout=drop,count=count,splits=sp,part=part,source=args.source,limits=LIMITS,errors=es,counters_zero=True)
      records.append(rec);(R/args.output).write_text(json.dumps(records,indent=2));print('PASS',n,drop,count,sp,part,flush=True)
      del g,p
 print('COMPLETE',len(records),'cases',len(records)*4,'comparisons',flush=True)
