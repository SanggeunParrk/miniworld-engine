from pathlib import Path
p=Path(__file__).resolve().parent;base='front_prefetch_lnpair_storepipe'
for mode in ['start','both']:
 s=(p/(base+'.cu')).read_text()
 old='if(threadIdx.x==0&&round>0)tma_store_wait_all();allsync();';assert old in s
 s=s.replace(old,'if(threadIdx.x==0&&round>0)tma_store_wait_all(); // same thread then issues G TMA; its mbarrier gates consumers')
 if mode=='both':
  old='if(threadIdx.x==0)tma_store_wait_all();allsync();\n p.partln';assert old in s
  s=s.replace(old,'if(threadIdx.x==0)tma_store_wait_all(); // role-exit publication barrier follows\n p.partln')
 name=base+'_lesssync_'+mode;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
