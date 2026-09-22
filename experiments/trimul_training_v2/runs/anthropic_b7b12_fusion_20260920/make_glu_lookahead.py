"""Overlap next-tile B7 with current-tile dW WGMMA in the current two-CTA design.

Unlike the historical H128/246-register overlap prototype, this uses H64,
64 accumulator registers and the selected112KiB double buffer. The next GLU
writes only the other stage. The accumulator remains live and fenced until
wait_group0; current-stage TMA refill follows that wait and a CTA rendezvous.
"""
from pathlib import Path
R=Path(__file__).resolve().parent
for base in ('front_ring96_cache3','front_prefetch_lnpair_storepipe'):
 original=(R/(base+'.cu')).read_text()
 a=original.index('TMN_DEVI void weight_role(')
 b=original.index('TMN_DEVI void load_g(',a)
 body=original[a:b]
 old='for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r)'
 body=body.replace(old,'if(split<p.tiles){mbar_wait(b,0);glu_small(p,sm,sm+40960,sm+49152,split*64);}\n '+old)
 old='mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);'
 assert old in body
 body=body.replace(old,'')
 old='});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();'
 assert old in body
 overlap='''});wgmma_commit();
  if(tile+DW_SPLITS<p.tiles){int next=slot^1;uint8_t* ns=sm+next*DW_SLOT;
   mbar_wait(b+next,((r+1)/2)&1);glu_small(p,ns,ns+40960,ns+49152,(tile+DW_SPLITS)*64);
  }
  wgmma_wait<0>();fence_regs(acc);allsync();'''
 body=body.replace(old,overlap)
 for unroll in (8,4,2):
  modes=('deferred','early') if 'ring' in base else ('deferred',)
  for mode in modes:
   kernel=body
   if mode=='early':
    kernel=kernel.replace('if(r>0)ring_finish(p,tile-DW_SPLITS,group);','')
    kernel=kernel.replace('});wgmma_commit();\n','});wgmma_commit();ring_finish(p,tile,group);\n')
    kernel=kernel.replace('if(rounds>0)ring_finish(p,split+(rounds-1)*DW_SPLITS,group);','')
   s=original[:a]+kernel+original[b:]
   s=s.replace('#pragma unroll 8','#pragma unroll '+str(unroll))
   name=base+'_glu_ahead_u'+str(unroll)+'_'+mode
   (R/(name+'.cu')).write_text(s)
   (R/(name+'.launch.json')).write_text((R/(base+'.launch.json')).read_text())
   print(name)
