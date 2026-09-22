"""Logical traffic of the selected ring kernel, not an algorithmic SoL proof."""
from pathlib import Path
import json
r=Path(__file__).resolve().parent;rows=[]
for n in (384,768):
 m=n*n;tiles=m//64;sp=20;dx_ctas=104
 per_tile=dict(dw_preactivation=8*16384,dw_upstream=8*8192,dw_xn=8*16384,ring_write=8*16384,ring_read=8*16384,dx_projection_weights=4*128*256*2,gate_dgrad_input=64*128*2,gate_weights=128*128*2,ln_input_and_residual=2*64*128*2,dx_write=64*128*2)
 traffic={k:v*tiles for k,v in per_tile.items()}
 traffic.update(dw_partial_write_and_read=2*8*sp*2*16384*4,ln_partial_write_and_read=2*dx_ctas*256*4,weight_output=4*128*256*2,ln_parameter_output=256*4)
 flops=2*m*(128*1024+128*1024+128*128)
 rows.append(dict(L=n,tiles=tiles,logical_bytes=traffic,logical_bytes_subtotal=sum(traffic.values()),tensor_flops=flops,scope='Named tensor loads/stores; excludes mask/stat/gamma/counters, polls, cache overfetch, write allocation and hardware-sector effects. Not compulsory traffic of all possible implementations.'))
(r/'ring-logical-traffic.json').write_text(json.dumps(rows,indent=2))
for v in rows:print(v['L'],'logical GB',v['logical_bytes_subtotal']/1e9,'tensor GFLOP',v['tensor_flops']/1e9)
