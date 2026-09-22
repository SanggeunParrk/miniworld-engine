import argparse,json,time
from pathlib import Path
import torch,triton
from miniworld_engine.kernels.trimul_inproj.cute.parity_front import launch_front,front_sm90,front_config_rejection
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import _bidir_front_kernel
p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--output',default='front/results.json');args=p.parse_args()
torch.manual_seed(419)
def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()
def compare(m,k,h2,maskmode,save,bk,bh,stages,bm=64):
 c=dict(BLOCK_M1=bm,BLOCK_K_D=bk,BLOCK_K_H2=bh,num_warps=8,num_stages=stages)
 a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.2
 w=torch.randn(k,4*h2,device='cuda',dtype=torch.bfloat16)*.2
 mask=None if maskmode=='none' else ((torch.rand(m,device='cuda')>.25).to(torch.bfloat16) if maskmode=='binary' else torch.rand(m,device='cuda').to(torch.bfloat16))
 out=torch.empty(2*h2,m,device='cuda',dtype=torch.bfloat16);pre=torch.empty(4*h2,m,device='cuda',dtype=torch.bfloat16) if save else None
 tlout=torch.empty_like(out);tlpre=torch.empty(4*h2,m,device='cuda',dtype=torch.bfloat16)
 def tc(): _bidir_front_kernel.fn[(triton.cdiv(m,bm),)](a,w,tlout[:h2],tlout[h2:],tlpre,mask,m,m,K=k,H2=h2,shape_key=0,SAVE_PREACT=save,**c)
 def cc(): launch_front(a,w,out,pre,mask,c)
 print('BEGIN',m,k,h2,maskmode,save,c,flush=True)
 tc();torch.cuda.synchronize();print('TRITON_DONE',flush=True)
 cc();torch.cuda.synchronize();print('CUTE_DONE',flush=True)
 errors={'out':rel(out,tlout)}
 if save:errors['preact']=rel(pre,tlpre)
 # Reference reproduces all intended BF16 boundaries, including non-binarymask.
 raw=a.float()@w.float();value=raw[:,0::2].sigmoid()*raw[:,1::2]
 if mask is not None:value=value.to(torch.bfloat16).float()*mask.float()[:,None]
 errors['torch_out']=rel(out,value.to(torch.bfloat16).T)
 if save:errors['torch_preact']=rel(pre,raw.to(torch.bfloat16).T)
 assert max(errors.values())<1e-4,errors
 if mask is not None: assert not out[:,mask==0].count_nonzero().item()
 result=dict(m=m,k=k,h2=h2,mask=maskmode,save=save,config=c,errors=errors)
 print(json.dumps(result),flush=True)
 return result
cases=[(144,96,40,'fractional',True,32,32,2)] if args.quick else [
 (144,96,40,'fractional',True,32,32,2),
 (256,128,64,'none',True,16,16,1),
 (256,128,64,'binary',True,64,64,4),
 (256,128,64,'fractional',False,32,32,3),
 (16384,128,256,'fractional',True,32,64,2),
 (147456,128,256,'binary',True,64,64,4),
 (144,96,40,'fractional',True,32,32,2,128),
 (147456,128,256,'binary',True,64,64,1,128),
]
results=[compare(*case) for case in cases]
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
Path(args.output).write_text(json.dumps(results,indent=2))
