import json,statistics
from pathlib import Path
from bench import load
import torch,triton
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel,dual_shape_key
root=Path(__file__).parent
import cutlass.cute as cute,subprocess
original_compile=cute.compile
dump=root/'compiled';dump.mkdir(exist_ok=True)
def keep(*a,**kw):
 kw['options']='--keep-cubin --keep-ptx --dump-dir='+str(dump)
 fn=original_compile(*a,**kw)
 binary=fn.__cubin__
 if binary:
  if isinstance(binary,str):binary=Path(binary).read_bytes()
  p=dump/(str(len(list(dump.glob('[0-9]*.cubin'))))+'.cubin');p.write_bytes(binary)
  resource=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(p)],text=True)
  print('RESOURCE '+resource.replace('\n',' '),flush=True)
 return fn
cute.compile=keep
base=load('baseline'); import argparse
ap=argparse.ArgumentParser();ap.add_argument('--variant',default='specialized');ap.add_argument('--short',action='store_true');ap.add_argument('--one',action='store_true');args=ap.parse_args()
new=load(args.variant)
kw=dict(device='cuda',dtype=torch.bfloat16)
L=384;M=L*L;kg=128;kp=1024;n=128
torch.manual_seed(93)
g=torch.randn(M,kg,**kw);f=torch.randn(kp,M,**kw).t();w=torch.randn(n,kg,**kw).t();v=torch.randn(kp,n,**kw);y=torch.empty(M,n,**kw)
best=dict(BLOCK_M1=64,BLOCK_N=128,BLOCK_K=64,GROUP_M=1,num_warps=4,num_stages=3)
def tri():
 _input_dual_bwd_kernel.fn[(triton.cdiv(M,64),)](g,f,w,v,y,M,kg,kp,n,*g.stride(),*f.stride(),*w.stride(),*v.stride(),shape_key=dual_shape_key(L,kg,kp,n),**best)
 return y
ref=tri();results=[]
for bk,st in ([(64,3)] if args.one else [(64,3),(64,2),(64,4)] if args.short else [(64,3),(64,2),(64,4),(32,3),(32,4),(128,2)]):
 c=dict(best,BLOCK_K=bk,num_warps=8,num_stages=st)
 funcs={'triton_strong':tri,'retained_strong':lambda:base.input_dual_bwd_sm90_impl(g,f,w,v,L,best),'retained_matched':lambda:base.input_dual_bwd_sm90_impl(g,f,w,v,L,c),'specialized':lambda:new.input_dual_bwd_sm90_impl(g,f,w,v,L,c)}
 for name,fn in funcs.items():
  z=fn();torch.cuda.synchronize();assert torch.equal(z,ref),(name,c,((z-ref).float().norm()/ref.float().norm()).item())
 times={k:[] for k in funcs}
 for r in range(3):
  keys=list(funcs);keys=keys[r:]+keys[:r]
  for name in keys:times[name].append(triton.testing.do_bench_cudagraph(funcs[name],rep=100))
 row=dict(config=c,median_ms={k:statistics.median(t) for k,t in times.items()},times_ms=times);print(json.dumps(row),flush=True);results.append(row);(root/(args.variant+'_probe.json')).write_text(json.dumps(results,indent=2))
