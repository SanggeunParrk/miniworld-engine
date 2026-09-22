from pathlib import Path
p=Path(__file__).resolve().parent
for base in ['front_ring96_cache3','front_prefetch_lnpair_storepipe']:
 for mode in ['early','pipeline']:
  s=(p/(base+'.cu')).read_text();a=s.index('TMN_DEVI void glu_small');b=s.index('TMN_DEVI void load_dw',a)
  f=s[a:b].replace('glu_small(', 'glu_premask(').replace('int row){','int row,uint32_t mask){').replace('uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);','')
  s=s[:b]+f+s[b:]
  a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b]
  old='mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);'
  assert old in v
  if mode=='early':
   new='uint32_t premask=*reinterpret_cast<const uint32_t*>(p.mask+tile*64+(threadIdx.x%32)*2);asm volatile(""::"r"(premask):"memory");mbar_wait(b+slot,(r/2)&1);glu_premask(p,s,s+40960,s+49152,tile*64,premask);'
  else:
   marker='if(split<p.tiles)load_dw';assert marker in v
   init='''uint32_t mask0=split<p.tiles?*reinterpret_cast<const uint32_t*>(p.mask+split*64+(threadIdx.x%32)*2):0;
 uint32_t mask1=split+DW_SPLITS<p.tiles?*reinterpret_cast<const uint32_t*>(p.mask+(split+DW_SPLITS)*64+(threadIdx.x%32)*2):0;
 asm volatile(""::"r"(mask0),"r"(mask1):"memory");
 '''
   v=v.replace(marker,init+marker,1)
   new='''mbar_wait(b+slot,(r/2)&1);glu_premask(p,s,s+40960,s+49152,tile*64,slot?mask1:mask0);
 if(tile+2*DW_SPLITS<p.tiles){uint32_t nextmask=*reinterpret_cast<const uint32_t*>(p.mask+(tile+2*DW_SPLITS)*64+(threadIdx.x%32)*2);asm volatile(""::"r"(nextmask):"memory");if(slot)mask1=nextmask;else mask0=nextmask;}
 '''
  v=v.replace(old,new);s=s[:a]+v+s[b:]
  name=base+'_maskreg_'+mode;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
