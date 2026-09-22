"""Retest near-neighbor role balance and small changes after LN prefetch."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'dual_ln_prefetch.cu').read_text()
variants={}
for dw in (36,42,44,48):
 variants[f'dual_pref_dw{dw}']=s.replace('#define DW_RATIO 10',f'#define DW_RATIO {dw}').replace('#define DX_RATIO 23',f'#define DX_RATIO {132-dw}')
for stride in (7,13):
 p=s.replace('// Independent CTA roles;',f'__device__ __forceinline__ int logical_block(){{return (blockIdx.x*{stride})%UCOUNT;}}\n// Independent CTA roles;')
 p=p.replace('int split=blockIdx.x;', 'int split=logical_block();').replace('int split=blockIdx.x-DWCOUNT;', 'int split=logical_block()-DWCOUNT;')
 p=p.replace('const bool dw=blockIdx.x<DWCOUNT;int split=dw?blockIdx.x:blockIdx.x-DWCOUNT;',
             'const int logical=logical_block();const bool dw=logical<DWCOUNT;int split=dw?logical:logical-DWCOUNT;')
 variants[f'dual_pref_place{stride}']=p
for name,p in variants.items():
 (r/(name+'.cu')).write_text('// Experiment: '+name+'\n'+p)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
