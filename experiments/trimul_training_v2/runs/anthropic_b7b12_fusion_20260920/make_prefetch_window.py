from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'front_kindprefetch.cu').read_text()
old=(p/'front_softw128_g1.cu').read_text();a=old.index('TMN_DEVI void window_sync');b=old.index('TMN_DEVI void weight_role',a);helper=old[a:b]
a=s.index('TMN_DEVI void weight_role');s=s[:a]+helper+s[a:]
a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b]
v=v.replace('for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){','int tile=split;for(int window=0;window*WINDOW<p.tiles;++window){int limit=min(p.tiles,(window+1)*WINDOW);for(;tile<limit;tile+=DW_SPLITS,++r){')
ix=v.rfind('\n }\n}');assert ix>=0;v=v[:ix]+'\n }window_sync(p,window+1);\n }\n}\n'+v[ix+len('\n }\n}'):];s=s[:a]+v+s[b:]
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
v=v.replace('for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){','int tile=split;for(int window=0;window*WINDOW<p.tiles;++window){int limit=min(p.tiles,(window+1)*WINDOW);for(;tile<limit;tile+=DXCOUNT,++round){')
v=v.replace('\n p.partln[split*256+threadIdx.x]=running;','\n window_sync(p,window+1);}\n p.partln[split*256+threadIdx.x]=running;');s=s[:a]+v+s[b:]
s=s.replace('atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);','for(int i=0;i<2*((p.tiles+WINDOW-1)/WINDOW);++i)atomicExch(p.counts+2+i,0u);atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);')
for w in [128,256,512,1024]:
 for gap in [0,1]:
  name='front_prefetch_w%d_g%d'%(w,gap);(p/(name+'.cu')).write_text('#define WINDOW %d\n#define WINDOW_GAP %d\n'%(w,gap)+s)
  c=json.loads((p/'front_kindprefetch.launch.json').read_text());c['extra_counts']=2*((9216+w-1)//w);(p/(name+'.launch.json')).write_text(json.dumps(c))
