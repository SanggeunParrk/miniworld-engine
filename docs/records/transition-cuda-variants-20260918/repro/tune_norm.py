import gc,hashlib,json,statistics
from pathlib import Path
import torch,triton
from miniworld_engine.kernels.transition.cuda.variants import norm_extension
root=Path(__file__).parent;ext=norm_extension();result={}
def bench(f,rep=8):return statistics.median(triton.testing.do_bench_cudagraph(f,rep=rep) for _ in range(3))
for d in (128,256,384,512):
 torch.manual_seed(918);x=torch.randn(129,d,device='cuda',dtype=torch.bfloat16);g=torch.rand(d,device='cuda')+.5;b=torch.randn_like(g)*.1;g[0]=0
 y,mu,rs=ext.forward(x,g,b,1e-5,4);dy=torch.randn_like(x);dr=torch.randn_like(x)
 xf=x.float().requires_grad_();gg=g.clone().requires_grad_();bb=b.clone().requires_grad_()
 yf=torch.nn.functional.layer_norm(xf,(d,),gg,bb,1e-5);dx,dg,db=torch.autograd.grad(yf,(xf,gg,bb),dy.float());ref=(dx.bfloat16()+dr,dg,db)
 errors={}
 for channels in (1,4,16,32):
  got=ext.backward(dy,x,g,mu,rs,dr,4,4,8,256,channels)
  err=[((a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-12)).item() for a,e in zip(got,ref)]
  assert max(err)<.002,err;errors[channels]=err
 del xf,gg,bb,yf,ref,got,dx,dg,db;gc.collect()
 x=torch.randn(384**2,d,device='cuda',dtype=torch.bfloat16);dy=torch.randn_like(x);dr=torch.randn_like(x);y,mu,rs=ext.forward(x,g,b,1e-5,4)
 rows=[]
 for w in (4,8):
  fwd=bench(lambda:ext.forward(x,g,b,1e-5,w))
  for waves in (2,4,8):
   for tx in (8,16):
    for threads in (128,256):
     for channels in (1,4,16,32):
      c=[w,waves,tx,threads,channels];back=bench(lambda:ext.backward(dy,x,g,mu,rs,dr,*c))
      rows.append(dict(config=c,fwd_ms=fwd,bwd_ms=back,total_ms=fwd+back))
 finalists=sorted(rows,key=lambda r:r['total_ms'])[:8]
 for r in finalists:
  c=r['config'];r['fwd_ms']=bench(lambda:ext.forward(x,g,b,1e-5,c[0]),40);r['bwd_ms']=bench(lambda:ext.backward(dy,x,g,mu,rs,dr,*c),40);r['total_ms']=r['fwd_ms']+r['bwd_ms']
 control=[4,4,8,256,32];control_ms=bench(lambda:ext.backward(dy,x,g,mu,rs,dr,*control),40)
 result[str(d)]=dict(D=d,L=384,source=ext.__file__,errors=errors,rows=rows,best_norm=min(finalists,key=lambda r:r['total_ms']),control_config=control,control_bwd_ms=control_ms)
 (root/'norm-selections.json').write_text(json.dumps(result,indent=2)+'\n');print('BEST',d,json.dumps(result[str(d)]['best_norm']),'CONTROL',control_ms,flush=True)
 del x,y,dy,dr,mu,rs;gc.collect();torch.cuda.empty_cache()
