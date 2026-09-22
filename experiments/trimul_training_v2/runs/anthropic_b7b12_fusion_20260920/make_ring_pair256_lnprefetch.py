"""Overlap the first LN row tile with the final projection WGMMA.

Slot0 is no longer consumed after projection step14. LN row0 can therefore
reuse it while step15 consumes slot1. Barrier2 has two transactions per row
pair (gate phase0 and LN row1 phase1), while barrier3 alternates for LN row0.
"""
from pathlib import Path
R = Path(__file__).resolve().parent
base = 'front_ring_pair256_lnq4_w128'
s = (R / (base + '.cu')).read_text()
a = s.index('TMN_DEVI void pair_ln_load(')
b = s.index('TMN_DEVI void pair_ln(', a)
s = s[:a] + '''TMN_DEVI void pair_ln_load(const Params& p,uint8_t* sm,uint64_t* b,int row,int sub){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b,32768);
 for(int c=0;c<2;++c){
  tma_load_2d(sm+sub*32768+c*8192,&p.x,b,c*64,row+sub*64);
  tma_load_2d(sm+sub*32768+16384+c*8192,&p.res,b,c*64,row+sub*64);
 }
}
''' + s[b:]
old = 'pair_gate_load(p,sm,b+2,row);mbar_wait(b+2,round&1);'
assert old in s
s = s.replace(old, 'pair_gate_load(p,sm,b+2,row);mbar_wait(b+2,0);')
old = 'if(step<14)pair_front_load(p,sm,b,tile,step+2);'
assert old in s
s = s.replace(old, old + '\n    if(step==14)pair_ln_load(p,sm,b+3,row,0);')
s = s.replace('pair_ln_load(p,sm,b+3,row);', 'pair_ln_load(p,sm,b+2,row,1);')
s = s.replace('mbar_wait(b+3,round&1);allsync();pair_ln(',
              'mbar_wait(b+3,round&1);mbar_wait(b+2,1);allsync();pair_ln(')
name = 'front_ring_pair256_lnprefetch'
(R / (name + '.cu')).write_text(s)
(R / (name + '.launch.json')).write_text((R / (base + '.launch.json')).read_text())
print(name)
