import argparse,json,statistics,gc
from pathlib import Path
import torch,triton
from large_candidates import candidates
from miniworld_engine.kernels.transition.cuda.variants import extension,norm_extension
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);a=p.parse_args();v=a.variant;root=Path(__file__).parent
bench=lambda f:statistics.median(triton.testing.do_bench_cudagraph(f,rep=40) for _ in range(3))
rel=lambda a,b:((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()
for d in (384,512):
 path=root/f'tune-{v}-D{d}.json';data=json.loads(path.read_text());assert 'best_norm' in data
 torch.manual_seed(234);m=384**2;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);g=torch.rand(d,device='cuda')+.5;beta=torch.randn_like(g)*.1;xn,_,_=norm_extension().forward(x,g,beta,1e-5,4)
 wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5;dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype)
 small=xn[:129].contiguous();res=x[:129].contiguous();grad=dh[:129].contiguous();aa=small.float()@wa.float().T;bb=small.float()@wb.float().T;sig=aa.sigmoid();h=(aa*sig*bb).bfloat16();yr=(h.float()@ws.float().T).bfloat16()+res;da=(grad.float()*bb*(sig+aa*sig*(1-sig))).bfloat16();db=(grad.float()*aa*sig).bfloat16()
 configs=list(candidates(v,d))
 for k in ('best_forward','best_backward'):
  if data[k]['config'] not in configs:configs.append(data[k]['config'])
 rows=[]
 for c in configs:
  e=extension(v,d,c);r=dict(config=c,status='ok',extension=e.__file__,resources=e.resources());errors={}
  if e.resources()['forward_smem']<=232448:
   y=e.forward(small,res,wa,wb,ws);errors['y']=rel(y,yr);r['fwd_ms']=bench(lambda:e.forward(xn,x,wa,wb,ws))
  if e.resources()['backward_smem']<=232448:
   hh,dab=e.gate_backward(small,wa,wb,grad);errors.update(h=rel(hh,h),da=rel(dab[:,:4*d],da),db=rel(dab[:,4*d:],db));r['gate_bwd_ms']=bench(lambda:e.gate_backward(xn,wa,wb,dh))
  assert max(errors.values())<.02,errors;r['errors']=errors;rows.append(r);print(d,v,json.dumps(r),flush=True)
 data['large_followup']=rows;data['best_forward']=min((r for r in rows if 'fwd_ms' in r),key=lambda r:r['fwd_ms']);data['best_backward']=min((r for r in rows if 'gate_bwd_ms' in r),key=lambda r:r['gate_bwd_ms']);data['config_scope']+='; large-D MG2/S1 follow-up with incumbent remeasurement'
 path.write_text(json.dumps(data,indent=2)+'\n');del x,xn,wa,wb,ws,dh,small,res,grad,aa,bb,sig,h,yr,da,db;gc.collect();torch.cuda.empty_cache()
