"""Complete compiled module with matched dropout RNG and every gradient."""
import json,os,pathlib
import torch
from build import extension
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward
old=bias_backward._extension();new=extension();rows=[]
for L in (384,768):
 for starting in (True,False):
    torch.manual_seed(1012)
    m=TriangleAttention(128,n_head=4,starting=starting,implementation='miniworld',p_drop=.25).cuda().bfloat16()
    m.ln_pair.float();m.train()
    with torch.no_grad():
        for n,t in m.named_parameters():
            if t.ndim==2:t.normal_(std=128**-.5)
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x);mask=torch.rand(1,L,device='cuda')>.1
    c=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def step():
        y=c(x,mask);return (y,*torch.autograd.grad(y,(x,*m.parameters()),dy))
    bias_backward._EXT=old;step()
    for seed in (12345,67890):
        bias_backward._EXT=old;torch.manual_seed(seed);before=step()
        bias_backward._EXT=new;torch.manual_seed(seed);after=step()
        assert all(bool(t.isfinite().all()) for t in after)
        assert torch.equal(before[0],after[0])
        errs=[float((u.float()-v.float()).norm()/u.float().norm().clamp_min(1e-8)) for u,v in zip(before,after)]
        assert max(errs)<.005,(L,starting,errs)
        rows.append(dict(length=L,starting=starting,seed=seed,relative_l2=errs));print('PASS',rows[-1],flush=True)
    del c,m,x,dy,mask,before,after;torch.compiler.reset();torch.cuda.empty_cache()
root=pathlib.Path(__file__).resolve().parent
(root/'results'/f'dropout-{os.getenv("SLURM_JOB_ID")}.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')
