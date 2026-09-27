import argparse,csv,itertools,json,socket,time
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.wide_b2b import launch
from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd,transition_b2b_fwd_saved
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
settings.configure(engine_backend='auto',autotune_miss_cap=24)
root=Path(__file__).parent;d=a.width;m=384**2
torch.manual_seed(678)
x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
gamma=torch.rand(d,device='cuda',dtype=torch.bfloat16)+.5
beta=torch.randn_like(gamma)*.1
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)/d**.5
wb=torch.randn_like(wa)/d**.5
ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/d**.5
rs,c1=stats_triton(x,1e-5)
small=x[:256];sr,sc=rs[:256],c1[:256]
reference,xn_ref=transition_b2b_fwd_saved(small,sr,sc,gamma,beta,wa,wb,ws)
out=torch.empty_like(x);saved=torch.empty_like(x);empty=x.new_empty(0)
configs=[dict(BM=bm,BN=bn,BK=d,num_warps=w,num_stages=s)
         for bm,bn,w,s in itertools.product((16,32,64,128),(32,64,128),(4,8),(1,2,3))
         if bm*d<=255*32*w]
with (root/f'search-D{d}.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(configs[0]));w.writeheader();w.writerows(configs)
result=dict(node=socket.gethostname(),D=d,L=384,scope='identical full-K b2b fusion; kernel only',rows=[])
path=root/f'tune-D{d}.json'
for save in (False,True):
 for i,c in enumerate(configs):
    row=dict(config=c,save_xn=save);t=time.monotonic()
    try:
        y,xn,kernel=launch(small,small,gamma,beta,sr,sc,wa,wb,ws,config=c,normalize=True,save_xn=save)
        torch.cuda.synchronize()
        rel=((y.float()-reference.float()).norm()/reference.float().norm()).item()
        assert torch.isfinite(y).all().item() and rel<.02,rel
        if save:
            xn_rel=((xn.float()-xn_ref.float()).norm()/xn_ref.float().norm()).item()
            assert xn_rel<.005,xn_rel
            row['xn_relative_frobenius']=xn_rel
        def fn():return launch(x,x,gamma,beta,rs,c1,wa,wb,ws,config=c,normalize=True,save_xn=save,
                               out=out,xn_out=saved if save else empty)
        _,_,kernel=fn();torch.cuda.synchronize()
        ms=triton.testing.do_bench_cudagraph(fn,rep=30)
        ptx=kernel.asm['ptx']
        row.update(status='ok',ms=ms,relative_frobenius=rel,registers=kernel.n_regs,spills=kernel.n_spills,
                   shared_bytes=kernel.metadata.shared,wgmma=ptx.count('wgmma.mma_async'),
                   mma_sync=ptx.count('mma.sync'),cp_async=ptx.count('cp.async'),tma=ptx.count('cp.async.bulk'))
    except Exception as e:
        row.update(status='failed',error=repr(e)[:1500])
        if 'illegal memory access' in str(e):
            result['rows'].append(row);path.write_text(json.dumps(result,indent=2)+'\n');raise
    row['seconds']=time.monotonic()-t;result['rows'].append(row)
    path.write_text(json.dumps(result,indent=2)+'\n');print(save,i+1,len(configs),json.dumps(row),flush=True)
selected={}
for save,name in ((False,'inference'),(True,'training')):
 ok=sorted([r for r in result['rows'] if r['status']=='ok' and r['save_xn']==save],key=lambda r:r['ms'])
 selected[name]=ok[0]
result['selected']=selected;path.write_text(json.dumps(result,indent=2)+'\n')
(root/f'selected-D{d}.json').write_text(json.dumps(selected,indent=2)+'\n')
print('SELECTED',json.dumps(selected),flush=True)
