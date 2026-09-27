"""Check that every reported Engine 2 measurement exercised the intended module route."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
expected={
 'trimul':{'inference':['tmn_k1_','tmn_k3_'],'training':['b1_fused','b7_joint']},
 'transition':{'inference':['transition_fwd_fused'],'training':['transition_bwd_fused']},
 'block':{'inference':['tmn_k1_','tmn_k3_','transition_fwd_fused'],'training':['b1_fused','b7_joint','transition_bwd_fused']},
 'single':{'inference':['tmn_k1_','tmn_k3_'],'training':[]},
 'opm':{'inference':['opm_epilogue_kernel'],'training':['opm_dgrad_kernel','opm_dwo_kernel']},
 'pwa':{'inference':['pwa_fwd2_kernel'],'training':['pwa_glue3_kernel']},
 'msa':{'inference':['opm_epilogue_kernel','pwa_fwd2_kernel','tmn_k1_','tmn_k3_','transition_fwd_fused','wide_d64_fwd_kernel'],
        'training':['opm_dgrad_kernel','pwa_glue3_kernel','b1_fused','b7_joint','transition_bwd_fused','wide_d64_bwd_kernel']},
 'dit':{'inference':['_attn_fwd_gated2','_resgate_adaln_rows_kernel'],'training':['_attn_bwd']},
}
checks=[]
for L in (384,768):
 for arm in ('pytorch','engine1','engine2'):
  d=json.loads((R/f'msa-{L}-{arm}.json').read_text())
  assert d['msa_depth']==1024
for module in ('opm','pwa'):
 for L in (384,768):
  for arm in ('pytorch','engine1','engine2'):
   d=json.loads((R/f'{module}-{L}-{arm}.json').read_text())
   assert d['msa_depth']==1024,(module,L,arm,'incorrect MSA depth')
   for mode in ('inference','training'):
    result=d['modes'][mode]
    assert result['finite'] and result['ms']>0,(module,L,arm,mode)
for module,modes in expected.items():
 for L in (384,768):
  d=json.loads((R/f'{module}-{L}-engine2.json').read_text())
  for mode,names in modes.items():
   result=d['modes'][mode]
   assert 'error' not in result and result['finite'] and result['ms']>0,(module,L,mode)
   kernels=result['kernels']
   assert kernels,(module,L,mode,'missing kernel trace')
   for name in names:assert any(name in k for k in kernels),(module,L,mode,name)
   checks.append(dict(module=module,L=L,mode=mode,ms=result['ms'],expected_kernels=names))
(R/'dispatch-audit.json').write_text(json.dumps(dict(passed=len(checks),checks=checks),indent=2)+'\n')
print('PASS',len(checks),'module/mode/shape dispatch checks')
