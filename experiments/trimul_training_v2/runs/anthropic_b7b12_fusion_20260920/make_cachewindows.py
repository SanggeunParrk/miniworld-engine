from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_twocta_kindwg.cu').read_text()
a=s.index('TMN_DEVI void weight_role')
s=s[:a]+'''TMN_DEVI void window_sync(const Params& p,int completed){
 allsync();if(threadIdx.x==0){atomicAdd(p.counts+2,1u);while(atomicAdd(p.counts+2,0u)<unsigned(UCOUNT*completed))__nanosleep(32);}allsync();
}
'''+s[a:]
# Preserve each CTA's original tile sequence and K-reduction order exactly.
a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);q=s[a:b]
q=q.replace('for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){','int tile=split;for(int window=0;window*WINDOW<p.tiles;++window){int limit=min(p.tiles,(window+1)*WINDOW);for(;tile<limit;tile+=DW_SPLITS,++r){')
ix=q.rfind('\n }\n}');assert ix>=0;q=q[:ix]+'\n }window_sync(p,window+1);\n }\n}\n'+q[ix+len('\n }\n}'):];s=s[:a]+q+s[b:]
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);q=s[a:b]
q=q.replace('for(int tile=split;tile<p.tiles;tile+=DXCOUNT){','int tile=split;for(int window=0;window*WINDOW<p.tiles;++window){int limit=min(p.tiles,(window+1)*WINDOW);for(;tile<limit;tile+=DXCOUNT){')
q=q.replace('\n p.partln[split*256+threadIdx.x]=running;','\n window_sync(p,window+1);}\n p.partln[split*256+threadIdx.x]=running;');s=s[:a]+q+s[b:]
s=s.replace('atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);','atomicExch(p.counts+2,0u);atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);')
for window in [256,512,1024,2048]:
 name='front_window%d'%window;(p/(name+'.cu')).write_text('#define WINDOW %d\n'%window+s);(p/(name+'.launch.json')).write_text('{"wgrad_slices":2,"threads":256,"shared":114688,"max_ctas_per_sm":2,"extra_counts":1}\n')
