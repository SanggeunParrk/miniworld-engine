from pathlib import Path
p=Path(__file__).resolve().parent
for base in ['front_ring96_cache3','front_prefetch_lnpair_storepipe','front_ring_paircta_u8']:
 s=(p/(base+'.cu')).read_text();a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
 # Two resident operand stages. Retire h-1 before reusing its stage for h+1.
 # Same accumulator and K order; do not access accumulator registers until
 # the final wait_group0 of each G/P sequence.
 v=v.replace('fence_regs(acc);wgmma_fence();','if(h==0){fence_regs(acc);wgmma_fence();}')
 row='tile*64' if 'paircta' in base else 'row'
 old='});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_g(p,sm,bar,%s,side,h+2);'%row
 new='});wgmma_commit();if(h>0){if(h==3){wgmma_wait<0>();fence_regs(acc);}else wgmma_wait<1>();allsync();if(h<3)load_g(p,sm,bar,%s,side,h+1);}'%row
 assert old in v,(base,'G');v=v.replace(old,new)
 old='});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_p(p,sm,bar,side,h+2);'
 new='});wgmma_commit();if(h>0){if(h==3){wgmma_wait<0>();fence_regs(acc);}else wgmma_wait<1>();allsync();if(h<3)load_p(p,sm,bar,side,h+1);}'
 assert old in v,(base,'P');v=v.replace(old,new)
 if 'paircta' not in base:v=v.replace('if(side==1&&h==1)load_ln_next','if(side==1&&h==2)load_ln_next')
 s=s[:a]+v+s[b:];name=base+'_mmapipeline';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
