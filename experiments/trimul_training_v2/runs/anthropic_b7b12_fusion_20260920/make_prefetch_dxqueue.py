from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'front_prefetch_lnpair.cu').read_text()
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
v=v.replace('int split=blockIdx.x-DWCOUNT,','__shared__ int queue_next;int split=blockIdx.x-DWCOUNT,')
v=v.replace('for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){','for(int tile=split;tile<p.tiles;tile=queue_next,++round){')
v=v.replace('  if(tile+DXCOUNT<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);','  if(threadIdx.x==0)queue_next=DXCOUNT+atomicAdd(p.counts+2,1u);allsync();if(queue_next<p.tiles)issue_gate(p,sm,bar+2,queue_next*64);')
s=s[:a]+v+s[b:];s=s.replace('atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);','atomicExch(p.counts+2,0u);atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);')
name='front_prefetch_dxqueue';(p/(name+'.cu')).write_text(s);c=json.loads((p/'front_kindprefetch.launch.json').read_text());c['extra_counts']=1;(p/(name+'.launch.json')).write_text(json.dumps(c))
