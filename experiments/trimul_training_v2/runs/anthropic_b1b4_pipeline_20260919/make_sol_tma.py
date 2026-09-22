"""TMA completion-scope and cache-policy experiments; old sources untouched."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_balanced.cu').read_text()

def read_wait(s):
    old='tma_store_commit();tma_store_wait_all();}'
    assert s.count(old)==1
    s=s.replace(old,'tma_store_commit();tma_store_wait_read<0>();}')
    s=s.replace('sync_group(); // TMA finished consuming this slot before cluster reuse.',
                'sync_group(); // TMA has read this slot; global writes may overlap next tile.')
    old=' for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]'
    assert old in s
    return s.replace(old,''' // Complete outstanding dtri writes before the global publication protocol.
 if(threadIdx.x%128==0)tma_store_wait_all();
 sync_group();
''' + old)

def stream_policy(s):
    helper='''// Streaming-only inputs are not reused by the other CTA role.
TMN_DEVI void tma_load_stream(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){
 uint64_t policy;
 asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;" :: "r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1),"l"(policy):"memory");
}
'''
    s=s.replace('struct MaskCycle',helper+'struct MaskCycle') if False else s
    marker='// Independent CTA roles; each barrier is armed before its own TMA loads.'
    s=s.replace(marker,helper+marker)
    for address in ('s+32768+c*8192,&p.proj','s+49152+c*8192,&p.xn',
                    's+65536+c*8192,&p.norm','s+32768,&p.tri'):
        assert 'tma_load_2d('+address in s
        s=s.replace('tma_load_2d('+address,'tma_load_stream('+address)
    return s

for name,s in [('dual_tma_read',read_wait(base)),
               ('dual_tma_stream',stream_policy(base)),
               ('dual_tma_read_stream',stream_policy(read_wait(base)))]:
    (r/(name+'.cu')).write_text('// Experiment: '+name+'\n'+s)
    (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
