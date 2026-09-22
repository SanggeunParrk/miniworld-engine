"""Release each hidden group's G/P slot after its TMA read completes.

The selected ring couples all eight producer groups to the last right-group
consumer. Per-group generations allow the left producers to advance earlier.
The streaming variant additionally waits for readiness at each load, rather
than waiting for all eight groups before starting the row tile.
"""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
for base in ('front_ring96_cache3','front_ring112_wait256'):
 for mode in ('free','stream'):
  s=(R/(base+'.cu')).read_text()
  old='ring_wait(p.counts+2+8*RING_TILES+slot,tile-RING_TILES+1)'
  assert old in s
  s=s.replace(old,'ring_wait(p.counts+2+8*RING_TILES+slot*8+group,tile-RING_TILES+1)')
  old='if(side==1&&h==3&&threadIdx.x==0)ring_publish(p.counts+2+8*RING_TILES+tile%RING_TILES,tile+1);'
  assert old in s
  s=s.replace(old,'if(threadIdx.x==0)ring_publish(p.counts+2+8*RING_TILES+(tile%RING_TILES)*8+side*4+h,tile+1);')
  s=s.replace('i<9*RING_TILES','i<16*RING_TILES')
  if mode=='stream':
   assert 'allsync();ring_ready(p,tile);' in s
   s=s.replace('allsync();ring_ready(p,tile);','allsync();')
   a=s.index('TMN_DEVI void load_g(');b=s.index('TMN_DEVI void load_p(',a)
   f=s[a:b]
   old='if(threadIdx.x)return;int slot=h&1;'
   assert old in f
   f=f.replace(old,'if(threadIdx.x)return;ring_wait(p.counts+2+((row/64)%RING_TILES)*8+side*4+h,row/64+1);int slot=h&1;')
   s=s[:a]+f+s[b:]
  name=base+'_group_'+mode
  (R/(name+'.cu')).write_text(s)
  cfg=json.loads((R/(base+'.launch.json')).read_text());cfg['extra_counts']=16*cfg['ring_tiles']
  (R/(name+'.launch.json')).write_text(json.dumps(cfg))
  print(name)
