"""Keep role counts/work identical; permute their physical CTA placement."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_balanced.cu').read_text()
for stride in (5,7,13,17):
 name=f'dual_place{stride}'
 s=base
 marker='// Independent CTA roles;'
 s=s.replace(marker,f'__device__ __forceinline__ int logical_block(){{return (blockIdx.x*{stride})%UCOUNT;}}\n'+marker)
 s=s.replace('int split=blockIdx.x;', 'int split=logical_block();')
 s=s.replace('int split=blockIdx.x-DWCOUNT;', 'int split=logical_block()-DWCOUNT;')
 s=s.replace('const bool dw=blockIdx.x<DWCOUNT;int split=dw?blockIdx.x:blockIdx.x-DWCOUNT;',
             'const int logical=logical_block();const bool dw=logical<DWCOUNT;int split=dw?logical:logical-DWCOUNT;')
 # Require the mapping to be a permutation for either supported launch count.
 import math
 assert all(math.gcd(stride,n)==1 for n in (66,132))
 (r/(name+'.cu')).write_text('// Experiment: bijective CTA role placement.\n'+s)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
