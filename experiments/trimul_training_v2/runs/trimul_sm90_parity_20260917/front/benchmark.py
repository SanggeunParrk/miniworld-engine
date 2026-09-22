import argparse,json,time,torch,triton
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cute.parity_front import launch_front,front_sm90,front_config_rejection
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import _bidir_front_kernel
from miniworld_engine.autotune.trimul_sm90_config import partition_configs
from miniworld_engine import settings
settings.configure(run_autotune=False)
parser=argparse.ArgumentParser();parser.add_argument("--m128",action="store_true");args=parser.parse_args()
resultfile="front/benchmark-m128.json" if args.m128 else "front/benchmark.json"
torch.manual_seed(29)
m,k,h2=384*384,128,256
a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1;w=torch.randn(k,4*h2,device='cuda',dtype=torch.bfloat16)*.1
mask=(torch.rand(m,device='cuda')>.15).to(torch.bfloat16)
out=torch.empty(2*h2,m,device='cuda',dtype=torch.bfloat16);pre=torch.empty(4*h2,m,device='cuda',dtype=torch.bfloat16)
tout=torch.empty_like(out);tpre=torch.empty_like(pre)
base=dict(BLOCK_M1=64,BLOCK_K_D=64,BLOCK_K_H2=64,num_warps=8,num_stages=4)
def tc():_bidir_front_kernel.fn[(triton.cdiv(m,base['BLOCK_M1']),)](a,w,tout[:h2],tout[h2:],tpre,mask,m,m,K=k,H2=h2,shape_key=0,SAVE_PREACT=True,**base)
tc();torch.cuda.synchronize()
def rel(x,y):return ((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12)).item()
def bench(fn):return triton.testing.do_bench_cudagraph(fn,rep=50)
triton_ms=bench(tc)
configs,rejected=partition_configs('trimul_inproj_gemm_gate_mmajor_sm90_cute',lambda c:front_config_rejection(c,m=m,k=k,h2=h2))
if args.m128:configs=[c for c in configs if c["BLOCK_M1"]==128]
results=[]
for i,c in enumerate(configs):
 start=time.monotonic()
 try:
  fn=lambda:launch_front(a,w,out,pre,mask,c)
  fn();torch.cuda.synchronize()
  errors=[rel(out,tout),rel(pre,tpre)];assert max(errors)<1e-4,errors
  ms=bench(fn)
  row=dict(config=c,ms=ms,errors=errors,status='ok',compile_check_s=time.monotonic()-start)
 except Exception as e:
  row=dict(config=c,status='failed',reason=str(e))
 results.append(row);print(i,json.dumps(row),flush=True)
 Path(resultfile).write_text(json.dumps(dict(triton_fixed_config=base,triton_fixed_ms=triton_ms,cute_results=results,rejected=rejected),indent=2))
print('SUMMARY',json.dumps(dict(triton_ms=triton_ms,best=min((r for r in results if r['status']=='ok'),key=lambda r:r['ms']))),flush=True)
# Opaque compiled path and CUDA graph with explicit config and default resolver.
c=min((r for r in results if r['status']=='ok'),key=lambda r:r['ms'])['config']
for args in [(c['BLOCK_M1'],c['BLOCK_K_H2'],c['BLOCK_K_D'],8,c['num_stages']),(0,0,0,0,0)]:
 fn=torch.compile(lambda a,w,mask:front_sm90(a,w,mask,True,*args),fullgraph=True,dynamic=False)
 x,p=fn(a,w,mask);torch.cuda.synchronize();assert rel(x,tout)<1e-4 and rel(p,tpre)<1e-4
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):x,p=fn(a,w,mask)
 graph.replay();torch.cuda.synchronize();assert rel(x,tout)<1e-4 and rel(p,tpre)<1e-4
print('COMPILE_GRAPH_PASS',flush=True)
