from pathlib import Path
import json
r=Path(__file__).resolve().parent
s=(r/'dual_stage32.cu').read_text()
def save(name,src):
 (r/(name+'.cu')).write_text(src)
 (r/(name+'.launch.json')).write_text(json.dumps({'row_tile':32}))
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,*args,**kwargs):\n        super().__init__(*args,source="'+name+'",**kwargs)\n')
n=(r/'dual_n32.cu').read_text();mma=n[n.index('TMN_DEVI void mma_dgrad'):n.index('// LN row reductions')]
a=s.index('TMN_DEVI void mma_dgrad');b=s.index('#define ROW_TILE')
n=s[:a]+mma+s[b:]
n=n.replace('qn<2','qn<4').replace('int n=wi*2+qn;uint8_t* sw=sm+131072+n*16384;','int n=wi*4+qn;uint8_t* sw=sm+131072+(n/2)*16384+(n%2)*4096;').replace('float acc[32]={}','float acc[16]={}').replace('for(int q=0;q<4;++q){uint32_t fx','for(int q=0;q<2;++q){uint32_t fx').replace('n*64+q*16','n*32+q*16')
save('dual_stage32_n32',n)
for name,src in [('dual_stage32_plain',s),('dual_stage32_n32_plain',n)]:
 src=src.replace('const int mask_period=(32*UCOUNT)%p.L==0?1:((64*UCOUNT)%p.L==0?2:0);','const int mask_period=0;')
 save(name,src)
