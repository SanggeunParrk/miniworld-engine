import json,statistics,subprocess,hashlib
from pathlib import Path
import torch,triton
import cutlass.cute as cute
oldcompile=cute.compile
def keep(*a,**kw):kw['options']='--keep-ptx --keep-cubin';return oldcompile(*a,**kw)
cute.compile=keep
import projection_prefetch_initial as mod
import dropout_tma as dropout
import committed_checkpoint as base
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel
root=Path(__file__).parent;result={}
def cfg(bn=64,bk=64,s=2,gm=4):return dict(BLOCK_M1=64,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=gm,num_warps=4,num_stages=s)
for seed,L in ((s,l) for s in (901,234) for l in (384,768)):
 m=L*L;n=128;kp=256;kg=128;kw=dict(device='cuda',dtype=torch.bfloat16);torch.manual_seed(seed)
 a=torch.randn(m,kp,**kw)*.2;x=torch.randn(m,kg,**kw)*.2;wp=torch.randn(n,kp,**kw)*.2;wg=torch.randn(kg,n,**kw)*.2;r=torch.randn(m,n,**kw);ds=(torch.rand(L,n,device='cuda')>.2).bfloat16()*1.25
 args=(a,x,wp,wg,r,ds,L);y,p,g=(torch.empty_like(r) for _ in range(3))
 c=cfg(gm=1);cb=cfg() if L==384 else cfg(128)
 tc=dict(cfg(s=3),BLOCK_M1=128,num_warps=8)
 def tri(c):return _output_f567_kernel.fn[(triton.cdiv(m,c['BLOCK_M1'])*triton.cdiv(n,c['BLOCK_N']),)](a,x,wp,wg,p,g,y,r,ds,m,L,kp,kg,n,*wp.stride(),*wg.stride(),shape_key=0,**c)
 out=mod.output_f567_impl(*args,c);tri(tc);torch.cuda.synchronize()
 errors=[((u.float()-v.float()).norm()/v.float().norm()).item() for u,v in zip(out,(y,p,g))];assert max(errors)<1e-4
 fns={'cute':lambda:mod.output_f567_impl(*args,c),'group4':lambda:mod.output_f567_impl(*args,cfg(gm=4)),'cute_previous':lambda:base.output_f567_impl(*args,cb),'triton_bm128':lambda:tri(tc),'triton_old':lambda:tri(cfg(s=3,gm=8)),'triton_m64':lambda:tri(cfg(s=3)),'triton_matched':lambda:tri(c)}
 samples={k:[] for k in fns}
 for rnd in range(5):
  keys=list(fns);keys=keys[rnd%len(keys):]+keys[:rnd%len(keys)]
  if rnd%2:keys=keys[::-1]
  for name in keys:samples[name].append(triton.testing.do_bench_cudagraph(fns[name],rep=65))
 medians={k:statistics.median(v) for k,v in samples.items()};tkey=min((k for k in medians if k.startswith('triton_')),key=lambda k:medians[k]);tconfigs={'triton_bm128':tc,'triton_old':cfg(s=3,gm=8),'triton_m64':cfg(s=3),'triton_matched':c}
 compiled=list(mod._COMPILE_CACHE.values())[-1];folder=root/f'group1-final-{seed}-L{L}';folder.mkdir(exist_ok=True);binary=compiled.__cubin__;binary=Path(binary).read_bytes() if isinstance(binary,str) else binary;(folder/'kernel.cubin').write_bytes(binary)
 resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(folder/'kernel.cubin')],text=True);(folder/'resources.txt').write_text(resources);sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(folder/'kernel.cubin')],text=True);(folder/'kernel.sass').write_text(sass)
 row={'cute_config':c,'triton_config':tconfigs[tkey],'triton_winner':tkey,'samples_ms':samples,'medians_ms':medians,'speedup':medians[tkey]/medians['cute'],'relative_l2':errors,'resources':resources,'cubin_sha256':hashlib.sha256(binary).hexdigest(),'has_hgmma':'HGMMA' in sass,'has_tma':'UTMALDG' in sass or 'UTMASTG' in sass};result[f'{seed}-{L}']=row;print(L,json.dumps(row),flush=True);(root/'group1-final-results.json').write_text(json.dumps(result,indent=2))
 del a,x,wp,wg,r,ds,args,y,p,g,out;torch.cuda.empty_cache()
