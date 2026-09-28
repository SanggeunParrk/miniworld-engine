import argparse, csv, gc, itertools, json, socket, time
from pathlib import Path
import torch
import triton
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.wide_b2b import launch

p=argparse.ArgumentParser(); p.add_argument('--width',type=int,required=True)
p.add_argument('--length',type=int,default=384); a=p.parse_args()
settings.configure(engine_backend='triton',autotune_miss_cap=24)
root=Path(__file__).parent; d=a.width; m=a.length**2
torch.manual_seed(123)
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)/d**.5
wb=torch.randn_like(wa)/d**.5
ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/(4*d)**.5
x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
var,mean=torch.var_mean(x.float(),dim=-1,correction=0)
rs=torch.rsqrt(var+1e-5);c1=mean*rs
gamma=torch.ones(d,device='cuda');beta=torch.zeros_like(gamma)
xn=x
empty=x.new_empty(0); out=torch.empty_like(x)
small=xn[:257].contiguous(); residual=x[:257].contiguous()
norm=((small.float()*rs[:257,None]-c1[:257,None])*gamma+beta).bfloat16()
ha=norm.float()@wa.float().T; hb=norm.float()@wb.float().T
h=(ha*torch.sigmoid(ha)*hb).bfloat16()
ref=((h.float()@ws.float().T).bfloat16()+residual)
refnorm=ref.float().norm()
prior=[]
for name in (f'tune-D{d}.json',f'tune-more-D{d}.json',f'tune-refine-D{d}.json'):
 prior.extend(r for r in json.loads((root/name).read_text())['rows'] if r['status']=='ok')
configs=[]
for r in sorted(prior,key=lambda r:r['ms']):
 if r['config'] not in configs:configs.append(r['config'])
 if len(configs)==8:break
(root/f'selected-D{d}.json').write_text(json.dumps(dict(config=configs[0]),indent=2)+'\n')
with (root/f'search_ln-D{d}.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(configs[0]));w.writeheader();w.writerows(configs)
result=dict(node=socket.gethostname(),D=d,L=a.length,mode='normalization fused into b2b, full output width',rows=[])
path=root/f'tune-ln-D{d}.json'
for i,c in enumerate(configs):
    row=dict(config=c);t=time.monotonic()
    try:
        y,_,kernel=launch(small,residual,gamma,beta,rs[:257],c1[:257],wa,wb,ws,config=c,normalize=True)
        torch.cuda.synchronize()
        rel=((y.float()-ref.float()).norm()/refnorm).item()
        assert torch.isfinite(y).all().item() and rel<.02,(rel,c)
        def fn():return launch(xn,x,gamma,beta,rs,c1,wa,wb,ws,config=c,out=out,xn_out=empty,normalize=True)
        _,_,kernel=fn();torch.cuda.synchronize()
        ms=triton.testing.do_bench_cudagraph(fn,rep=30)
        ptx=kernel.asm['ptx']
        row.update(status='ok',ms=ms,relative_frobenius=rel,registers=kernel.n_regs,
                   spills=kernel.n_spills,shared_bytes=kernel.metadata.shared,
                   wgmma=ptx.count('wgmma.mma_async'),mma_sync=ptx.count('mma.sync'),
                   cp_async=ptx.count('cp.async'),tma=ptx.count('cp.async.bulk'))
    except Exception as e:
        row.update(status='failed',error=repr(e)[:1200])
    row['seconds']=time.monotonic()-t
    result['rows'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
    print(i+1,len(configs),json.dumps(row),flush=True)
    gc.collect()
ok=sorted((r for r in result['rows'] if r['status']=='ok'),key=lambda r:r['ms'])
result['best']=ok[:8];path.write_text(json.dumps(result,indent=2)+'\n')
print('BEST',json.dumps(ok[:8]),flush=True)

(root/f'selected-ln-D{d}.json').write_text(json.dumps(dict(config=ok[0]['config']),indent=2)+'\n')
