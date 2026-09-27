import json,re,subprocess,statistics
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.segmented_b2b import launch
from miniworld_engine.kernels.transition.triton.fused import _transition_expand_gatebwd_savedxn_stacked as gate, _transition_expand_gatebwd_kernel as kg
from miniworld_engine.autotune.shape_key import both_key
settings.configure(engine_backend='triton',autotune_miss_cap=24)
root=Path(__file__).parent;prior=root.parent/'transition_cuda_variants_20260918';rows=[]
ptxas=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
if not ptxas.exists():ptxas=Path('/usr/local/cuda-12.9/bin/ptxas')
(root/'ptxas-version.txt').write_text(subprocess.check_output([str(ptxas),'--version'],text=True))
def save(k,name,cfg,ms):
 (root/f'{name}.ptx').write_text(k.asm['ptx']);(root/f'{name}.cubin').write_bytes(k.asm['cubin'])
 for ext in ('ttgir','llir'):
  if ext in k.asm:(root/f'{name}.{ext}').write_text(k.asm[ext])
 sass=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',str(root/f'{name}.cubin')],text=True);(root/f'{name}.sass').write_text(sass)
 resources=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-resource-usage',str(root/f'{name}.cubin')],text=True);(root/f'{name}.resources.txt').write_text(resources)
 p=subprocess.run([str(ptxas),'-v','-arch=sm_90a',str(root/f'{name}.ptx'),'-o',str(root/f'{name}-reassembled.cubin')],capture_output=True,text=True);(root/f'{name}.ptxas.txt').write_text(p.stdout+p.stderr)
 row=dict(name=name,config=cfg,ms=ms,registers=k.n_regs,spills=k.n_spills,metadata=str(k.metadata),ptxas_returncode=p.returncode,opcodes={op:len(re.findall(r'\b'+op+r'\b',sass)) for op in ('HGMMA','HMMA','UTMALDG','LDGSTS','LDL','STL','BAR','WARPGROUP')});rows.append(row);print(json.dumps(row),flush=True)
 (root/'triton-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
bench=lambda f:statistics.median(triton.testing.do_bench_cudagraph(f,rep=30) for _ in range(3))
for d in (128,256,384,512):
 m=384**2;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype);wb=torch.randn_like(wa);ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype);dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype);z=x.new_empty(0)
 for v in ('streamed_k','full_k'):
  c=json.loads((prior/f'tune-{v}-D{d}.json').read_text())['best_triton_forward']['config']
  y,_,k=launch(x,x,z,z,z,z,wa,wb,ws,config=c)
  save(k,f'triton-{v}-D{d}-fwd',c,bench(lambda:launch(x,x,z,z,z,z,wa,wb,ws,config=c)))
 gate(x,wa,wb,dh,shape_key=both_key(m));bc=kg.best_config;print('GATE_CONFIG',d,str(bc),flush=True)
 h=torch.empty_like(dh);dab=torch.empty(m,8*d,device='cuda',dtype=x.dtype)
 c={**bc.kwargs,'num_warps':bc.num_warps,'num_stages':bc.num_stages}
 k=kg.fn[(triton.cdiv(m,c['BLOCK_M1'])*triton.cdiv(4*d,c['BLOCK_N']),)](x,x,x,x,x,wa,wb,dh,h,dab,dab,dab,x,m,4*d,d,both_key(m),d,1,d,1,d,1,4*d,1,4*d,1,8*d,1,d,1,NORMALIZE=False,STORE_H=True,STACK_DAB=True,**c)
 save(k,f'triton-D{d}-gate',c,bench(lambda:gate(x,wa,wb,dh,shape_key=both_key(m))))
 del x,wa,wb,ws,dh,z,y,h,dab;torch.cuda.empty_cache()
