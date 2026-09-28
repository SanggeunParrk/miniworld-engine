import argparse,gc,hashlib,json,socket,statistics,time
from pathlib import Path
import torch,triton
from configs import seeds,smem
from miniworld_engine import settings
from miniworld_engine.kernels.transition.cuda.variants import extension,norm_extension
from miniworld_engine.kernels.transition.triton.segmented_b2b import launch as triton_launch
from miniworld_engine.kernels.transition.triton.fused import _transition_expand_gatebwd_savedxn_stacked
from miniworld_engine.autotune.shape_key import both_key

p=argparse.ArgumentParser();p.add_argument('--variant',required=True);a=p.parse_args();root=Path(__file__).parent
settings.configure(engine_backend='triton',autotune_miss_cap=24)
def bench(fn):return statistics.median(triton.testing.do_bench_cudagraph(fn,rep=40) for _ in range(3))
def relative(x,y):return ((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12)).item()
for d in [128,256,384,512]:
    torch.manual_seed(234);m=384**2;v=a.variant
    x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);g=torch.rand(d,device='cuda')+.5;b=torch.randn_like(g)*.1
    norm=norm_extension();xn,mu,rs=norm.forward(x,g,b,1e-5,4)
    wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5
    dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype);small=xn[:129].contiguous();res=x[:129].contiguous();gd=dh[:129].contiguous()
    aa=small.float()@wa.float().T;bb=small.float()@wb.float().T;sg=aa.sigmoid();hh=(aa*sg*bb).bfloat16()
    yy=(hh.float()@ws.float().T).bfloat16()+res;da=(gd.float()*bb*(sg+aa*sg*(1-sg))).bfloat16();db=(gd.float()*aa*sg).bfloat16()
    data=dict(node=socket.gethostname(),device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,triton_version=triton.__version__,source_sha256=hashlib.sha256(Path(__import__('miniworld_engine.kernels.transition.cuda.variants',fromlist=['x']).__file__).with_name('transition_variants_kernel.cu').read_bytes()).hexdigest(),D=d,L=384,variant=v,config_scope='bounded seed sweep; not full public grid',rows=[],triton=[])
    path=root/f'tune-{v}-D{d}.json'
    def save():path.write_text(json.dumps(data,indent=2)+'\n')
    # Compare the CURRENT Triton full-K implementation, not the old pre-hoist results.
    bk=(1<<(d-1).bit_length()) if v=='full_k' else 64;bo=64 if d==128 else 128;empty=x.new_empty(0)
    tc=[dict(BM=bm,BN=bn,BK=bk,BO=bo,num_warps=8,num_stages=st) for bm,bn,st in [(64,32,1),(64,32,2),(64,64,2),(64,128,3),(128,32,3)]]
    for c in tc:
        r=dict(config=c)
        try:
            y,_,k=triton_launch(small,res,empty,empty,empty,empty,wa,wb,ws,config=c)
            torch.cuda.synchronize();err=relative(y,yy);assert err<.02,err
            r.update(status='ok',relative=err,ms=bench(lambda:triton_launch(xn,x,empty,empty,empty,empty,wa,wb,ws,config=c)),registers=k.n_regs,spills=k.n_spills,shared=k.metadata.shared)
        except Exception as e:r.update(status='failed',error=str(e)[-1500:])
        data['triton'].append(r);save();print('TRITON',d,v,json.dumps(r),flush=True)
    gate=lambda:_transition_expand_gatebwd_savedxn_stacked(xn,wa,wb,dh,shape_key=both_key(m))
    gate();data['triton_gate_ms']=bench(gate);save()
    for c in seeds(v,d):
        r=dict(config=c);started=time.monotonic()
        if min(smem(v,d,c),smem(v,d,c,True))>232448:
            r.update(status='resource_excluded',reason='shared memory >232448 bytes',forward_smem=smem(v,d,c),backward_smem=smem(v,d,c,True))
        else:
            try:
                e=extension(v,d,c);errors={}
                r.update(status='ok',resources=e.resources(),extension=e.__file__)
                if smem(v,d,c)<=232448:
                    out=e.forward(small,res,wa,wb,ws);torch.cuda.synchronize();errors['y']=relative(out,yy)
                    assert errors['y']<.02,errors
                    r['fwd_ms']=bench(lambda:e.forward(xn,x,wa,wb,ws))
                else:r['forward_excluded']='shared memory >232448 bytes'
                if smem(v,d,c,True)<=232448:
                    h,dab=e.gate_backward(small,wa,wb,gd);torch.cuda.synchronize()
                    errors.update(h=relative(h,hh),da=relative(dab[:,:4*d],da),db=relative(dab[:,4*d:],db))
                    assert max(errors.values())<.02,errors
                    r['gate_bwd_ms']=bench(lambda:e.gate_backward(xn,wa,wb,dh))
                else:r['backward_excluded']='shared memory >232448 bytes'
                r['errors']=errors
            except Exception as e:
                r.update(status='failed',error=str(e)[-2500:])
                if 'illegal memory' in str(e):data['rows'].append(r);save();raise
        r['seconds']=time.monotonic()-started;data['rows'].append(r);save();print('CUDA',d,v,json.dumps(r),flush=True)
    ok=[r for r in data['rows'] if r['status']=='ok'];assert ok
    data['best_forward']=min((r for r in ok if 'fwd_ms' in r),key=lambda r:r['fwd_ms']);data['best_backward']=min((r for r in ok if 'gate_bwd_ms' in r),key=lambda r:r['gate_bwd_ms'])
    tok=[r for r in data['triton'] if r['status']=='ok'];data['best_triton_forward']=min(tok,key=lambda r:r['ms'])
    # LN has no per-config compilation; tune the shared native fwd/bwd on actual pair rows.
    dxn=torch.randn_like(x);dr=torch.randn_like(x);data['norm']=[]
    for w in [4,8]:
        fwd_ms=bench(lambda:norm.forward(x,g,b,1e-5,w))
        for waves in [2,4,8]:
            for tx in [8,16]:
                for reduce in [128,256]:
                    nc=[w,waves,tx,reduce,32]
                    ms=bench(lambda:norm.backward(dxn,x,g,mu,rs,dr,*nc))
                    data['norm'].append(dict(config=nc,fwd_ms=fwd_ms,bwd_ms=ms,total_ms=fwd_ms+ms))
    data['best_norm']=min(data['norm'],key=lambda r:r['total_ms']);save()
    print('BEST',d,v,json.dumps({k:data[k] for k in ['best_forward','best_backward','best_norm','best_triton_forward','triton_gate_ms']}),flush=True)
    del xn,mu,rs,x,dh,small,res,gd,dxn,dr;gc.collect();torch.cuda.empty_cache()
