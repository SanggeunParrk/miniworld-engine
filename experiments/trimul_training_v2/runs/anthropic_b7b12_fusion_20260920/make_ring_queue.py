from pathlib import Path
import json
p=Path(__file__).resolve().parent;base='front_ring96_cache3_storepipe_acq32'
s=(p/(base+'.cu')).read_text()
old='if(split<p.tiles)issue_gate(p,sm,bar+2,split*64);int round=0;for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){'
new='''__shared__ int queued_tile;
 if(threadIdx.x==0)queued_tile=atomicAdd(p.counts+2+9*RING_TILES,1u);allsync();int first=queued_tile;
 if(first<p.tiles)issue_gate(p,sm,bar+2,first*64);int round=0;for(int tile=first;tile<p.tiles;tile=queued_tile,++round){'''
assert old in s;s=s.replace(old,new)
old='if(tile+DXCOUNT<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);'
new='if(threadIdx.x==0)queued_tile=atomicAdd(p.counts+2+9*RING_TILES,1u);allsync();if(queued_tile<p.tiles)issue_gate(p,sm,bar+2,queued_tile*64);'
assert old in s;s=s.replace(old,new)
s=s.replace('i<9*RING_TILES;i+=','i<9*RING_TILES+1;i+=')
name='front_ring96_queue';(p/(name+'.cu')).write_text(s)
cfg=json.loads((p/(base+'.launch.json')).read_text());cfg['extra_counts']+=1
(p/(name+'.launch.json')).write_text(json.dumps(cfg))
