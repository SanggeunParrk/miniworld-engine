import torch,json
from miniworld_engine import settings
settings.configure(engine_backend='triton',compile_wrap='custom_op')
from miniworld_engine.kernels.transition.triton.main import triton_transition
from triton.testing import do_bench_cudagraph
r=[]
for m,d in [(384**2,128),(384,384),(768,768)]:
 h=torch.randn(m,4*d,device='cuda',dtype=torch.bfloat16);w=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/(4*d)**.5;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
 old=lambda:h@w.T+x
 new=lambda:torch.addmm(x,h,w.T)
 a=old();b=new();print('addmm',m,d,do_bench_cudagraph(old),do_bench_cudagraph(new),'rel',((a-b).float().norm()/a.float().norm()).item(),flush=True)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  new();torch.cuda.synchronize()
 print([e.name for e in p.events() if str(e.device_type).endswith('CUDA')],flush=True)
x=torch.randn(2,3,128,device='cuda',dtype=torch.bfloat16).transpose(0,1).requires_grad_()
wa=torch.randn(128,512,device='cuda',dtype=torch.bfloat16).T;wb=wa.clone();ws=torch.randn(128,512,device='cuda',dtype=torch.bfloat16)
try:
 y=triton_transition(x,wa,wb,ws,4);print('noncontig x passed')
except Exception as e: print('noncontig x',type(e).__name__,str(e))
x=x.contiguous()
y=triton_transition(x,wa,wb,ws,4);ref=triton_transition(x,wa.contiguous(),wb.contiguous(),ws,4)
print('noncontig weight rel',((y-ref).float().norm()/ref.float().norm()).item(),flush=True)
