from pathlib import Path
import json
R=Path(__file__).resolve().parent
s=(R/'front_ring96_cache3.cu').read_text()
start=s.index('TMN_DEVI void load_g(');end=s.index('TMN_DEVI void reduce_at(',start)
s=s[:start]+(R/'ring_ws_input.cuh').read_text()+'\n'+s[end:]
s=s.replace('__shared__ uint64_t bar[4];','__shared__ WsBarriers bars;uint64_t* bar=bars.tx;')
a='for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();'
assert a in s
s=s.replace(a,'for(int i=0;i<6;++i)mbar_init(bars.tx+i,1);for(int i=0;i<5;++i)mbar_init(bars.empty+i,1);fence_barrier_init();')
a='if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else input_role(p,sm,bar,gamma);'
assert a in s
s=s.replace(a,'''if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else {
  if(threadIdx.x<128){setmaxnreg_dec<40>();ws_producer(p,sm,&bars);setmaxnreg_inc<128>();}
  else{setmaxnreg_inc<216>();ws_consumer(p,sm,&bars,gamma);setmaxnreg_dec<128>();}
 }''')
name='front_ring_ws216'
(R/(name+'.cu')).write_text(s)
cfg=json.loads((R/'front_ring96_cache3.launch.json').read_text())
cfg.update(weight_tma_rows=128,gate_tma_rows=128)
(R/(name+'.launch.json')).write_text(json.dumps(cfg))
print(name)
