from pathlib import Path
import json
R=Path(__file__).resolve().parent
s=(R/'front_ring96_cache3.cu').read_text()
a=s.index('TMN_DEVI void load_g(');b=s.index('TMN_DEVI void reduce_at(',a)
s=s[:a].replace('#pragma unroll 8','#pragma unroll 2')+(R/'ring_pair256_input.cuh').read_text()+'\n'+s[b:]
name='front_ring_pair256'
(R/(name+'.cu')).write_text(s)
c=json.loads((R/'front_ring96_cache3.launch.json').read_text())
c.update(weight_tma_rows=128,gate_tma_rows=128)
(R/(name+'.launch.json')).write_text(json.dumps(c))
print(name)
