import argparse,json,torch
import save_cost_core as C
import compare_cueq_training as Q
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);a=p.parse_args();n=a.length
with torch.no_grad():
 d=Q.setup(n)['d'];ref,kept=C.LN.S.forward(d);ref=ref.clone();ab,tri=kept[:2]
 for pg,method in ((False,0),(True,0),(True,1)):
  y,pre=C.front(d,pg,method);torch.cuda.synchronize();print('FRONT',pg,method,torch.equal(y,ab),pre.shape if pre is not None else None,flush=True);assert torch.equal(y,ab)
 for ln,stats,pg,method in ((0,0,False,0),(1,0,False,0),(1,1,False,0),(1,3,False,0),(1,3,True,0),(1,3,True,1),(3,3,True,0)):
  y,s=C.output(d,tri,ln,stats,pg,method);torch.cuda.synchronize();print('OUTPUT',ln,stats,pg,method,torch.equal(y,ref),{k:list(v.shape) for k,v in s.items() if v is not None},flush=True);assert torch.equal(y,ref)
 print('PROBE_DONE',flush=True)
