import argparse,json,socket
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.autotune.hopper_cuda_config import candidates
from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd,transition_b2b_fwd_saved
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
settings.configure(autotune_miss_cap=24)
root=Path(__file__).parent;d=a.width;m=384**2
torch.manual_seed(678)
x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
gamma=torch.rand(d,device='cuda',dtype=torch.bfloat16)+.5
beta=torch.randn_like(gamma)*.1
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)/d**.5
wb=torch.randn_like(wa)/d**.5
ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/d**.5
rs,c1=stats_triton(x,1e-5)
reference=transition_b2b_fwd(x,rs,c1,gamma,beta,wa,wb,ws)
result=dict(node=socket.gethostname(),D=d,L=384,rows=[]);selected={}
for save,mode in ((False,'inference'),(True,'training')):
 for config in candidates('b2b',d):
    fn=transition_b2b_fwd_saved if save else transition_b2b_fwd
    def call():return fn(x,rs,c1,gamma,beta,wa,wb,ws,config=config)
    observed=call();observed=observed[0] if save else observed
    torch.testing.assert_close(observed,reference,atol=0,rtol=0)
    ms=triton.testing.do_bench_cudagraph(call,rep=50)
    row=dict(config=config,mode=mode,ms=ms);result['rows'].append(row)
    print(json.dumps(row),flush=True)
 selected[mode]=min((r for r in result['rows'] if r['mode']==mode),key=lambda r:r['ms'])
(root/f'tune-cuda-D{d}.json').write_text(json.dumps(result,indent=2)+'\n')
(root/f'selected-cuda-D{d}.json').write_text(json.dumps(selected,indent=2)+'\n')
