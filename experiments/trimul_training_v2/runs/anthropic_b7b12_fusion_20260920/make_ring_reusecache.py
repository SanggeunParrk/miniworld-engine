from pathlib import Path
p=Path(__file__).resolve().parent
base='front_ring96_cache3_storepipe_acq32'
src=(p/(base+'.cu')).read_text()
for mode in ['xnlast','wlast','reuse','ringnormal']:
 s=src
 if mode in ['xnlast','reuse']:
  s=s.replace('tma_load_2d(s+24576+n*8192,&p.xn','tma_last(s+24576+n*8192,&p.xn')
 if mode in ['wlast','reuse']:
  s=s.replace('tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg','tma_last(s+24576+n*8192,side?&p.wrg:&p.wlg')
  s=s.replace('tma_load_2d(s+n*8192,side?&p.wr:&p.wl','tma_last(s+n*8192,side?&p.wr:&p.wl')
  s=s.replace('tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate','tma_last(sm+16384+n*16384+k*8192,&p.wgate')
 if mode=='ringnormal':
  s=s.replace('tma_last(s+16384,&p.ring','tma_load_2d(s+16384,&p.ring').replace('tma_last(sm+81920+h*8192,&p.ring','tma_load_2d(sm+81920+h*8192,&p.ring')
 name='front_ring_reuse_'+mode
 (p/(name+'.cu')).write_text(s)
 (p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
