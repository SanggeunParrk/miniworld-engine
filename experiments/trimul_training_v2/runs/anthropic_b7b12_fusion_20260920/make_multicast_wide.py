"""Share saved x_n over 4/8 dW channel groups with one multicast transaction.

The asynchronous cluster-ready protocol is inherited from the measured 2-CTA
prototype. Every dW cluster processes identical row coordinates. Runtime
occupancy validation prevents an oversubscribed cooperative launch.
"""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
for base in ('front_ring96_cache3','front_prefetch_lnpair_storepipe'):
 source=base+'_mcastasync_xn'
 original=(R/(source+'.cu')).read_text()
 for size in (4,8):
  s=original.replace('blockIdx.x&1','blockIdx.x&%d'%(size-1))
  s=s.replace('(unsigned short)3','(unsigned short)%d'%((1<<size)-1))
  s=s.replace('__cluster_dims__(2,1,1)','__cluster_dims__(%d,1,1)'%size)
  s=s.replace('mbar_init(bar+i+4,2)','mbar_init(bar+i+4,%d)'%size)
  if base=='front_prefetch_lnpair_storepipe':
   s=s.replace('#pragma unroll 8','#pragma unroll 4')
  name=base+'_mcastasync_xn'+str(size)
  (R/(name+'.cu')).write_text(s)
  config=json.loads((R/(source+'.launch.json')).read_text())
  config['cluster_size']=size
  (R/(name+'.launch.json')).write_text(json.dumps(config))
  print(name)
