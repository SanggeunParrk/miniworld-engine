import json
from pathlib import Path
import torch
from miniworld_engine.autotune import native
from miniworld_engine.autotune.trimul_sm90_config import declared_configs, partition_for_bucket
from miniworld_engine.kernels.checks import trimul_inproj as checks
original=native.choose_config;records=[]
def observe(op,candidates,**kwargs):
    expected,rejected=partition_for_bucket(op,kwargs['bucket'])
    assert candidates==expected,(op,len(candidates),len(expected))
    records.append({'op':op,'declared':len(declared_configs(op)),'feasible':len(expected),'rejected':rejected})
    return original(op,candidates,**kwargs)
native.choose_config=observe
torch.backends.cuda.matmul.allow_tf32=False
errors={}
for name in ('trimul_parity_front_sm90','trimul_parity_f567_sm90','trimul_parity_dual_bwd_sm90'):
    outputs=getattr(checks,name)();errors[name]={}
    for label,(actual,ref) in outputs.items():
        err=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-8))
        errors[name][label]=err;assert err<1e-4,(name,label,err)
Path('module/build-contract.json').write_text(json.dumps({'records':records,'relative_l2':errors,'passed':True},indent=2)+'\n')
print(json.dumps({'counts':[{k:r[k] for k in ('op','declared','feasible')} for r in records],'errors':errors}),flush=True)
