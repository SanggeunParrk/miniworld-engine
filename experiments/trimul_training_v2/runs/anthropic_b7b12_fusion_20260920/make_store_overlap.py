from pathlib import Path
p=Path(__file__).resolve().parent
for base in ['front_prefetch_lnpair','front_prefetch_reduceln','front_ring96_cache3','front_ring128_cache3']:
 s=(p/(base+'.cu')).read_text();a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
 v=v.replace('tma_store_commit();tma_store_wait_all();','tma_store_commit();')
 v=v.replace('  allsync(); // TMA store cannot read a slot overwritten by next tile\'s gate loads.','  // dx store overlaps the next gate contraction; drained before front buffers refill.')
 marker='  for(int side=0;side<2;++side){';assert marker in v;v=v.replace(marker,'  if(threadIdx.x==0&&round>0)tma_store_wait_all();allsync();\n'+marker,1)
 v=v.replace(' p.partln[split*256+threadIdx.x]=running;',' if(threadIdx.x==0)tma_store_wait_all();allsync();\n p.partln[split*256+threadIdx.x]=running;')
 s=s[:a]+v+s[b:];name=base+'_storepipe';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
