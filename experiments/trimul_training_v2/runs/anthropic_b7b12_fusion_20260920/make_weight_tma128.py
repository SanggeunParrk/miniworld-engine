from pathlib import Path
import json
p=Path(__file__).resolve().parent
for base in ['front_prefetch_lnpair_storepipe','front_ring96_cache3_storepipe_acq32','front_ring96_queue']:
 s=(p/(base+'.cu')).read_text()
 for old,new in [('for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg,b+slot,h*64,n*64);','tma_load_2d(s+24576,side?&p.wrg:&p.wlg,b+slot,h*64,0);'),('for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b+slot,h*64,n*64);','tma_load_2d(s,side?&p.wr:&p.wl,b+slot,h*64,0);')]:
  assert old in s,(base,old);s=s.replace(old,new)
 name=base+'_wtma128';(p/(name+'.cu')).write_text(s)
 cfg=json.loads((p/(base+'.launch.json')).read_text());cfg['weight_tma_rows']=128
 (p/(name+'.launch.json')).write_text(json.dumps(cfg))
