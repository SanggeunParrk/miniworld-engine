from core import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from bench_k3 import graph,paired
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);plans={str(r):Plan(d,dy,s,64,64,1,r,1,2,0,0,2) for r in (0,1,2)};gs={k:graph(p) for k,p in plans.items()}
  ctx,_,_=s;xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mean,rs,gate,proj=ctx.saved_tensors
  dp,dg=B.gate_elem_bwd_ew(dy.reshape(n*n,128),proj,gate,d['ds'],n)
  gs['b1']=graph(lambda:B.gate_elem_bwd_ew(dy.reshape(n*n,128),proj,gate,d['ds'],n));gs['dwg']=graph(lambda:torch.mm(xn.reshape(n*n,128).t(),dg));gs['dwp']=graph(lambda:torch.mm(dp.t(),norm));gs['dnorm']=graph(lambda:torch.mm(dp,wp));gs['baseline']=graph(lambda:baseline(d,dy,s))
  tt=paired(gs,reps=80,rounds=12);print('RESULT',n,{k:round(v['median_us'],2) for k,v in tt.items()},flush=True);(R/f'probe-{n}.json').write_text(json.dumps(tt,indent=2))
