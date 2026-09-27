"""Dump actual compiled artifacts and provide a narrowly filtered NCU target."""
import argparse, json, socket
from pathlib import Path
import torch
from miniworld_engine.kernels.transition.triton.wide_b2b import launch

p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
root=Path(__file__).parent;d=a.width;m=384**2
c=json.loads((root/f'selected-D{d}.json').read_text())['config']
torch.manual_seed(514)
x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
xn=torch.nn.functional.layer_norm(x.float(),(d,)).bfloat16()
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)/d**.5
wb=torch.randn_like(wa)/d**.5
ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/(4*d)**.5
empty=x.new_empty(0);out=torch.empty_like(x)
for _ in range(2):
 y,_,kernel=launch(xn,x,empty,empty,empty,empty,wa,wb,ws,config=c,out=out,xn_out=empty)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
launch(xn,x,empty,empty,empty,empty,wa,wb,ws,config=c,out=out,xn_out=empty)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
for suffix in ('ptx','ttgir','cubin'):
 data=kernel.asm[suffix];path=root/f'best-D{d}.{suffix}'
 if isinstance(data,bytes):path.write_bytes(data)
 else:path.write_text(data)
record=dict(node=socket.gethostname(),D=d,L=384,config=c,registers=kernel.n_regs,
            spills=kernel.n_spills,shared_bytes=kernel.metadata.shared,
            torch=torch.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name())
(root/f'kernel-D{d}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
