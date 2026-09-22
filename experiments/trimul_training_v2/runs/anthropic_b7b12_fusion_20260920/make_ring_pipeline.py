from pathlib import Path
p=Path(__file__).resolve().parent
for w in [128,256]:
 s=(p/f'front_ring{w}.cu').read_text();a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b]
 v=v.replace('glu_small(p,s,s+40960,s+49152,tile*64);ring_begin(p,s,tile,group);','glu_small(p,s,s+40960,s+49152,tile*64);if(r>0)ring_finish(p,tile-DW_SPLITS,group);ring_begin(p,s,tile,group);')
 v=v.replace('fence_regs(acc);allsync();ring_finish(p,tile,group);','fence_regs(acc);allsync();')
 ix=v.rfind('\n }\n}');assert ix>=0;v=v[:ix]+v[ix:].replace('\n }\n}','\n }\n if(rounds>0)ring_finish(p,split+(rounds-1)*DW_SPLITS,group);\n}',1)
 s=s[:a]+v+s[b:];name=f'front_ring{w}_pipe';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/f'front_ring{w}.launch.json').read_text())
