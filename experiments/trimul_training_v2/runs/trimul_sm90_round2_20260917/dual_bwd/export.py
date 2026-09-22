import argparse,json,subprocess
from pathlib import Path
import cutlass.cute as cute
import torch
from bench import load
p=argparse.ArgumentParser();p.add_argument('variant');args=p.parse_args()
root=Path(__file__).parent;folder=root/'cubin'/args.variant;folder.mkdir(parents=True,exist_ok=True)
orig=cute.compile;captured=[]
def keep(*a,**kw):
 kw['options']='--keep-ptx --keep-cubin --dump-dir='+str(folder)
 fn=orig(*a,**kw);captured.append(fn);return fn
cute.compile=keep
mod=load(args.variant);kw=dict(dtype=torch.bfloat16,device='cuda');M=384**2
G=torch.randn(M,128,**kw);F=torch.randn(1024,M,**kw).t();W=torch.randn(128,128,**kw).t();V=torch.randn(1024,128,**kw)
c=dict(BLOCK_M1=64,BLOCK_N=128,BLOCK_K=64,GROUP_M=1,num_warps=4,num_stages=3)
mod.input_dual_bwd_sm90_impl(G,F,W,V,384,c);torch.cuda.synchronize()
fn=captured[-1];binary=fn.__cubin__
if isinstance(binary,str):binary=Path(binary).read_bytes()
if binary is None:binary=list(folder.glob('*.cubin'))[-1].read_bytes()
f=folder/'kernel.cubin';f.write_bytes(binary)
for opt,name in [('--dump-sass','sass'),('--dump-resource-usage','resources.txt')]:
 text=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump',opt,str(f)],text=True);(folder/('kernel.'+name)).write_text(text)
print(json.dumps(dict(variant=args.variant,resources=(folder/'kernel.resources.txt').read_text(),kernel_info=fn.kernel_info),default=str))
