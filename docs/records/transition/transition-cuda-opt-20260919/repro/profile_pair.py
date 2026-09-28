import argparse,json,torch
from pathlib import Path
from loader import extension
from baseline_loader import old_extension
from miniworld_engine.kernels.transition.triton.segmented_b2b import launch
from miniworld_engine.kernels.transition.triton.fused import _transition_expand_gatebwd_savedxn_stacked as gate
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine import settings
p=argparse.ArgumentParser();p.add_argument('--file',required=True);p.add_argument('--d',type=int,required=True);p.add_argument('--variant',default='full_k');a=p.parse_args();r=Path(__file__).parent;d=a.d
settings.configure(engine_backend='triton',autotune_miss_cap=24)
rows=[x for x in json.loads((r/a.file).read_text()) if x.get('status')=='ok' and x['D']==d and x['variant']==a.variant];t=min(rows,key=lambda x:x['ms']['new']);rec=json.loads((r.parent/'transition_cuda_variants_20260918'/f'tune-{a.variant}-D{d}.json').read_text());new=extension(t);old=old_extension(a.variant,d,rec['best_'+t['direction']]['config'])
m=384**2;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5;dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype);z=x.new_empty(0)
if t['direction']=='forward':
 calls=[lambda:old.forward(x,x,wa,wb,ws),lambda:new.forward(x,x,wa,wb,ws),lambda:launch(x,x,z,z,z,z,wa,wb,ws,config=rec['best_triton_forward']['config'])]
else:calls=[lambda:old.gate_backward(x,wa,wb,dh),lambda:new.gate_backward(x,wa,wb,dh),lambda:gate(x,wa,wb,dh,shape_key=both_key(m))]
for _ in range(5):
 for f in calls:f()
torch.cuda.synchronize();print('ORDER old,new,triton',json.dumps(t),flush=True)
torch.cuda.cudart().cudaProfilerStart()
for f in calls:f();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
