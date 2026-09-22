import sys,json,subprocess,hashlib
from pathlib import Path
import torch
import cutlass.cute as cute
orig_compile=cute.compile
def keep(*a,**kw):
 kw['options']='--keep-ptx --keep-cubin --generate-line-info'
 return orig_compile(*a,**kw)
cute.compile=keep
import projection_prefetch_initial as mod
L=int(sys.argv[1]);m=L*L;n=128;kp=256;kg=128;kw=dict(device='cuda',dtype=torch.bfloat16)
torch.manual_seed(123);a=torch.randn(m,kp,**kw)*.2;x=torch.randn(m,kg,**kw)*.2;wp=torch.randn(n,kp,**kw)*.2;wg=torch.randn(kg,n,**kw)*.2;r=torch.randn(m,n,**kw);ds=(torch.rand(L,n,device='cuda')>.2).bfloat16()*1.25
args=(a,x,wp,wg,r,ds,L);cfg=dict(BLOCK_M1=64,BLOCK_N=64,BLOCK_K=64,GROUP_M=4,num_warps=4,num_stages=2)
for _ in range(5):out=mod.output_f567_impl(*args,cfg)
torch.cuda.synchronize()
compiled=list(mod._COMPILE_CACHE.values())[-1];binary=compiled.__cubin__;binary=Path(binary).read_bytes() if isinstance(binary,str) else binary
Path(f'prefetch-L{L}.cubin').write_bytes(binary)
Path(f'prefetch-L{L}.resources.txt').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',f'prefetch-L{L}.cubin'],text=True))
Path(f'prefetch-L{L}.sass').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',f'prefetch-L{L}.cubin'],text=True))
Path(f'prefetch-L{L}.json').write_text(json.dumps({'L':L,'config':cfg,'sha256':hashlib.sha256(binary).hexdigest()},indent=2))
torch.cuda.cudart().cudaProfilerStart();out=mod.output_f567_impl(*args,cfg);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
