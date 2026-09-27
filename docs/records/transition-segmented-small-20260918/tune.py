import argparse,csv,itertools,json,socket,time
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.segmented_b2b import launch
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
settings.configure(engine_backend='triton',autotune_miss_cap=24)
root=Path(__file__).parent;d=a.width;m=384**2
torch.manual_seed(782)
x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
g=torch.rand(d,device='cuda');g[0]=0;b=torch.randn_like(g)*.1
wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5
wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5
rs,c1=stats_triton(x,1e-5)
xn=((x.float()*rs[:,None]-c1[:,None])*g+b).bfloat16()
small=x[:257];refxn=xn[:257];ha=refxn.float()@wa.float().T;hb=refxn.float()@wb.float().T
ref=((ha*torch.sigmoid(ha)*hb).bfloat16().float()@ws.float().T).bfloat16()+small
out=torch.empty_like(x);saved=torch.empty_like(x);empty=x.new_empty(0)
configs=[dict(BM=bm,BN=bn,BK=d,BO=bo,num_warps=w,num_stages=s)
 for bm,bn,bo,w,s in itertools.product((32,64,128),(32,64,128),(d//4,d//2),(4,8),(1,2,3))
 if bm*d<255*32*w]
with (root/f'search-D{d}.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(configs[0]));writer.writeheader();writer.writerows(configs)
result={'node':socket.gethostname(),'D':d,'L':384,'affine_dtype':'float32','rows':[]}
path=root/f'tune-D{d}.json'
for mode,norm,save in [('separate',False,False),('fused_inference',True,False),('fused_training',True,True)]:
 for i,c in enumerate(configs):
  row=dict(mode=mode,config=c);start=time.monotonic()
  try:
   y,zn,k=launch(small if norm else refxn,small,g,b,rs[:257],c1[:257],wa,wb,ws,config=c,normalize=norm,save_xn=save)
   torch.cuda.synchronize();rel=((y.float()-ref.float()).norm()/ref.float().norm()).item()
   assert torch.isfinite(y).all().item() and rel<.02,rel
   if save:torch.testing.assert_close(zn,refxn,atol=.015625,rtol=.01)
   def fn():return launch(x if norm else xn,x,g,b,rs,c1,wa,wb,ws,config=c,normalize=norm,save_xn=save,out=out,xn_out=saved if save else empty)
   _,_,k=fn();torch.cuda.synchronize()
   ms=triton.testing.do_bench_cudagraph(fn,rep=25)
   row.update(status='ok',ms=ms,relative_frobenius=rel,registers=k.n_regs,spills=k.n_spills,shared_bytes=k.metadata.shared,wgmma=k.asm['ptx'].count('wgmma.mma_async'),mma_sync=k.asm['ptx'].count('mma.sync'))
  except Exception as e:
   row.update(status='failed',error=repr(e)[-1800:])
   if 'illegal memory' in str(e):result['rows'].append(row);path.write_text(json.dumps(result,indent=2));raise
  row['seconds']=time.monotonic()-start;result['rows'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
  print(mode,i+1,len(configs),json.dumps(row),flush=True)
selected={}
for mode in ['separate','fused_inference','fused_training']:
 ok=sorted((r for r in result['rows'] if r['mode']==mode and r['status']=='ok'),key=lambda r:r['ms']);assert ok
 selected[mode]=ok[0]
result['selected']=selected;path.write_text(json.dumps(result,indent=2)+'\n')
(root/f'selected-D{d}.json').write_text(json.dumps(selected,indent=2)+'\n')
print('SELECTED',json.dumps(selected),flush=True)
