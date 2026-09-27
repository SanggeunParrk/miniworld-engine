import argparse,itertools,json,traceback
from pathlib import Path
import torch
from miniworld_engine.kernels.layernorm.cute.tma_backward import prepare,config_rejection
p=argparse.ArgumentParser();p.add_argument('--part',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
cases=[]
for dtype,layout,n,bk,bm,nw,ns in itertools.product((torch.bfloat16,torch.float32),('col','row'),(128,137,256,384),(64,128,256,512),(8,32),(1,8),(1,3)):
 if (n,bk) not in ((128,64),(137,64),(137,256),(256,256),(384,128),(384,512)):continue
 cases.append((dtype,layout,n,bk,bm,nw,ns))
if a.smoke: cases=[(torch.bfloat16,'col',256,256,32,8,2),(torch.bfloat16,'col',137,64,8,1,3),(torch.float32,'row',137,256,8,8,1)]
else: cases=cases[a.part::2]
rows=[]
for dtype,layout,n,bk,bm,nw,ns in cases:
 c=dict(BLOCK_M1=bm,BLOCK_K=bk,num_warps=nw,num_stages=ns)
 row=dict(dtype=str(dtype),layout=layout,n=n,config=c)
 try:
  m=263; pitch=((m if layout=='col' else n)+7)//8*8
  strides=(1,pitch) if layout=='col' else (pitch,1)
  x=torch.empty_strided((m,n),strides,device='cuda',dtype=dtype).normal_();dy=torch.empty_strided((m,n),strides,device='cuda',dtype=dtype).normal_();w=torch.randn(n,device='cuda')
  mean=x.float().mean(1);rs=(x.float().var(1,unbiased=False)+1e-5).rsqrt();out=torch.empty_strided((m,n),strides,device='cuda',dtype=dtype)
  dw=torch.empty((7,n),device='cuda');db=torch.empty_like(dw)
  reason=config_rejection(c,n=n,itemsize=x.element_size(),m_major=layout=='col',smem_limit=torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
  if reason: row['skip']=reason
  else:
   fn=prepare(x,dy,w,mean,rs,out,dw,db,c);fn();torch.cuda.synchronize()
   xhat=(x.float()-mean[:,None])*rs[:,None];wd=dy.float()*w
   refs=[((wd-((wd*xhat).mean(1)[:,None]*xhat+wd.mean(1)[:,None]))*rs[:,None]).to(dtype),(dy.float()*xhat).sum(0),dy.float().sum(0)]
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):fn()
   out.fill_(float('nan'));dw.fill_(float('nan'));db.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
   errs=[]
   for v,r in zip((out,dw.sum(0),db.sum(0)),refs):
    assert torch.isfinite(v).all()
    err=((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item();errs.append(err)
    assert err < (0.004 if dtype==torch.bfloat16 else 3e-6),(err,errs)
   row.update(passed=True,errors=errs)
 except Exception as e:row.update(passed=False,error=traceback.format_exc());print(row,flush=True);raise
 rows.append(row);print(json.dumps(row),flush=True)
 Path(__file__).with_name(f'check-{a.part}.json').write_text(json.dumps(rows,indent=2))
