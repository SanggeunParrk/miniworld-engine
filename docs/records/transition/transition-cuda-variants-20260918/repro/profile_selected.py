import argparse,json,torch
from pathlib import Path
from miniworld_engine.kernels.transition.cuda.variants import extension,norm_extension
p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True);p.add_argument('--variant',default='full_k');a=p.parse_args();root=Path(__file__).parent;d=a.d;r=json.loads((root/f'tune-{a.variant}-D{d}.json').read_text())
ef=extension(a.variant,d,r['best_forward']['config']);eb=extension(a.variant,d,r['best_backward']['config'])
m=384**2;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);g=torch.ones(d,device='cuda');beta=torch.zeros_like(g);xn,_,_=norm_extension().forward(x,g,beta,1e-5,4)
wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5;dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype)
f=lambda:ef.forward(xn,x,wa,wb,ws);b=lambda:eb.gate_backward(xn,wa,wb,dh)
for _ in range(5):f();b()
torch.cuda.synchronize();print(json.dumps(dict(D=d,variant=a.variant,forward=r['best_forward'],backward=r['best_backward'])),flush=True)
torch.cuda.cudart().cudaProfilerStart();f();b();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
