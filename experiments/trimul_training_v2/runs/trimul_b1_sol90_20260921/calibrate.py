import torch,triton,triton.language as tl,json,platform,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
@triton.jit
def stream(a,b,c,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);tl.store(c+i,tl.load(a+i,i<N,0)+tl.load(b+i,i<N,0),i<N)
def us(f):
 for _ in range(10):f()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):f()
 vals=[]
 for _ in range(7):
  s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record()
  for _ in range(60):g.replay()
  e.record();e.synchronize();vals.append(s.elapsed_time(e)*1000/60)
 return sorted(vals)[len(vals)//2],vals
result=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),stream=[],gemm=[],description='Saturated streaming two-read one-write FP32 add, total working set above L2; BF16 dense cuBLAS matmul. Measured references, not fundamental upper bounds.')
with torch.no_grad():
 for n in (1<<24,1<<26,1<<28):
  a=torch.randn(n,device='cuda');b=torch.randn_like(a);c=torch.empty_like(a)
  for block in (1024,4096,8192):
   for warps in (4,8):
    med,vals=us(lambda:stream[(triton.cdiv(n,block),)](a,b,c,n,block,num_warps=warps));row=dict(N=n,block=block,warps=warps,median_us=med,samples_us=vals,TBps=12*n/med/1e6);result['stream'].append(row);print(row,flush=True)
  del a,b,c
 for n in (4096,8192):
  a=torch.randn(n,n,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a);c=torch.empty_like(a)
  med,vals=us(lambda:torch.mm(a,b,out=c));result['gemm'].append(dict(N=n,median_us=med,samples_us=vals,TFps=2*n**3/med/1e6));del a,b,c
result['smi']=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,clocks.sm,clocks.mem,power.limit','--format=csv'],stdout=subprocess.PIPE,text=True).stdout
(R/'calibration-node01.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
