import torch,triton,json
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.triton.segmented_residual import _kernel
from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b
settings.configure(engine_backend='triton',autotune_miss_cap=24)
m=Transition(256,implementation='triton').cuda().eval()
with torch.no_grad():m.squeeze.weight.normal_(std=256**-.5)
x=torch.randn(1,384,384,256,device='cuda',dtype=torch.bfloat16)
with torch.no_grad():
 for _ in range(3):y=m(x)
 print('SELECTED',_kernel.best_config,flush=True)
 print('CACHE',[(str(k),str(v)) for k,v in _kernel.cache.items()],flush=True)
 print('MS',triton.testing.do_bench_cudagraph(lambda:m(x),rep=150),flush=True)
