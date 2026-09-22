"""Replace dX GEMM full-CTA waits with two per-stage empty barriers.

One arrival per warpgroup after WGMMA wait0. Only the TMA issuer waits before
refilling a stage. A group transition drains both stages before rearming the
ready barriers, so no warpgroup can lap the other's empty-barrier phase.
The final CTA barrier protects LN scratch overlapping the last P operands.
"""
from pathlib import Path
import json
R = Path(__file__).resolve().parent
for base in ('front_prefetch_lnpair_storepipe_ctasync',
             'front_ring96_cache3', 'front_ring112_wait256'):
    s = (R/(base+'.cu')).read_text()
    helper = '''// Both consumers retire a WGMMA stage before its next TMA overwrite.
TMN_DEVI void retire_stage(uint64_t* bar,int slot){
 if(threadIdx.x%128==0)mbar_arrive(bar+4+slot);
}
TMN_DEVI void drain_stages(uint64_t* bar){
 if(threadIdx.x==0){mbar_wait(bar+4,1);mbar_wait(bar+5,1);}
}
'''
    pos = s.index('TMN_DEVI void input_role(')
    s = s[:pos]+helper+s[pos:]
    start = s.index('TMN_DEVI void input_role(')
    end = s.index('TMN_DEVI void reduce_at(',start)
    body = s[start:end]
    a = 'wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_g(p,sm,bar,row,side,h+2);'
    b = 'wgmma_wait<0>();fence_regs(acc);retire_stage(bar,slot);if(h<2){if(threadIdx.x==0)mbar_wait(bar+4+slot,h/2);load_g(p,sm,bar,row,side,h+2);}'
    assert a in body,base
    body = body.replace(a,b)
    a = 'wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_p(p,sm,bar,side,h+2);'
    b = 'wgmma_wait<0>();fence_regs(acc);retire_stage(bar,slot);if(h<2){if(threadIdx.x==0)mbar_wait(bar+4+slot,h/2);load_p(p,sm,bar,side,h+2);}'
    assert a in body,base
    body = body.replace(a,b)
    body = body.replace('   load_g(p,sm,bar,row,side,0);', '   drain_stages(bar);load_g(p,sm,bar,row,side,0);')
    body = body.replace('   load_p(p,sm,bar,side,0);', '   drain_stages(bar);load_p(p,sm,bar,side,0);')
    a = '  if(tile+DXCOUNT<p.tiles)issue_gate('
    assert a in body,base
    body = body.replace(a,'  allsync(); // LN scratch overlaps the retired P2/P3 cache.\n'+a)
    s = s[:start]+body+s[end:]
    s = s.replace('__shared__ uint64_t bar[4];','__shared__ uint64_t bar[6];')
    a = 'for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();'
    assert a in s,base
    s = s.replace(a,'for(int i=0;i<4;++i)mbar_init(bar+i,1);mbar_init(bar+4,2);mbar_init(bar+5,2);fence_barrier_init();')
    name = base+'_stage_release'
    (R/(name+'.cu')).write_text(s)
    (R/(name+'.launch.json')).write_text((R/(base+'.launch.json')).read_text())
    print(name)
