"""FP64 oracle and exact dK/dV checks for compact bias partials."""
import argparse,json,os,pathlib
import torch
from build import extension
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward

p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=64)
p.add_argument('--sanitize',action='store_true');a=p.parse_args()
old=bias_backward._extension();new=extension();L=a.length
torch.manual_seed(92799)
def tensor():
    return torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4)
def rel(x,y):return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))
rows=[]
for kind in (['mixed'] if a.sanitize else ['none','mixed','one_key','all_masked','zero']):
 for magnitude in ([1] if a.sanitize else [1,65536]):
    q,k,v=[tensor() for _ in range(3)];v.mul_(4)
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.3
    if kind=='mixed':b[...,::7]=torch.finfo(b.dtype).min
    if kind=='one_key':b[...,1:]=torch.finfo(b.dtype).min
    if kind=='all_masked':b.fill_(torch.finfo(b.dtype).min)
    if kind=='zero':q.zero_();k.zero_();v.zero_()
    dy=tensor();dy.mul_(magnitude)
    # Exact attention probabilities and statistics, sliced by outer row to
    # avoid materializing O(L^3) storage in sanitizer workloads.
    lse=torch.empty((1,4,L,L),device='cuda',dtype=torch.float32)
    delta=torch.empty_like(lse);target=torch.zeros_like(b,dtype=torch.float64)
    for i in range(L):
        qq=q[:,:,i].double();kk=k[:,:,i].double();vv=v[:,:,i].double();dd=dy[:,:,i].double()
        logits=qq@kk.transpose(-2,-1)*(32**-.5)+b.double()
        prob=logits.softmax(-1)
        if kind=='all_masked':prob.zero_()
        oo=prob@vv
        lse[:,:,i]=logits.logsumexp(-1)*1.4426950408889634
        delta[:,:,i]=(oo*dd).sum(-1)
        target+=(prob*((dd@vv.transpose(-2,-1))-(oo*dd).sum(-1,keepdim=True)))
    before=old.backward(q,k,v,b,lse.contiguous(),delta.contiguous(),dy,8)
    after=new.backward(q,k,v,b,lse.contiguous(),delta.contiguous(),dy,8)
    torch.cuda.synchronize()
    assert all(bool(t.isfinite().all()) for t in after)
    assert torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])
    err=rel(after[2],target);prior=rel(before[2],target);inc=rel(after[2],before[2])
    # One-key softmax has mathematically zero dS; compare absolute noise there.
    if kind=='one_key':
        assert float(after[2].abs().max()) < magnitude*.001
    else:assert err<.02 and inc<.005,(kind,err,inc)
    row=dict(kind=kind,magnitude=magnitude,old_fp64_rel=prior,new_fp64_rel=err,incremental_rel=inc,dkdv_bitwise=True,max_db=float(after[2].abs().max()))
    rows.append(row);print('PASS',row,flush=True)
mode='sanitize' if a.sanitize else 'full'
path=pathlib.Path(__file__).resolve().parent/'results'/f'stress-{mode}-L{L}-{os.getenv("SLURM_JOB_ID","manual")}.json'
path.write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')
