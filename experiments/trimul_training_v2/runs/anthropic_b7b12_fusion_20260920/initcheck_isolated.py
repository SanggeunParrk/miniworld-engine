"""Initcheck fixture: host-initialized inputs, no upstream forward/B1-B6.

Only checks internal uninitialized reads. Mathematical equivalence is covered
separately by check_front.py. Avoids reading TMA-stored dx in another kernel.
"""
from front_plan import *
import argparse,faulthandler
faulthandler.dump_traceback_later(30,repeat=True)
ap=argparse.ArgumentParser();ap.add_argument('--part',type=int,default=2);args=ap.parse_args()
with torch.no_grad():
 torch.manual_seed(20260920);n=64;m=n*n
 def init(shape,fp32=False):return torch.randn(shape,dtype=torch.float32).to(torch.float32 if fp32 else torch.bfloat16).cuda()
 d=dict(n=n,x=init((1,n,n,128)),gi=init((128,),True))
 a=dict(d=d,dl=init((1,256,n,n)),dr=init((1,256,n,n)),dg=init((m,128)),dy=init((1,n,n,128)),pre=init((1024,m)),xn=init((m,128)),mu=init((m,),True),rs=torch.ones(m).cuda(),mask=torch.ones(m,dtype=torch.bfloat16).cuda())
 for k in ('wl','wlg','wr','wrg'):a[k]=init((128,256))
 a['wg']=init((128,128));p=Plan(a,part=args.part)
 for _ in range(2):p();torch.cuda.synchronize()
 assert p.counts[:2].cpu().tolist()==[0,0]
 print('ISOLATED_INITCHECK_FINISHED',args.part,flush=True)
 faulthandler.cancel_dump_traceback_later()
