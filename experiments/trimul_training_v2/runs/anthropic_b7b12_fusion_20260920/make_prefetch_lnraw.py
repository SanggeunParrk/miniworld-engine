from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_prefetch_lnpair.cu').read_text()
for kind in ['raw','halfxhat']:
 v=s.replace('s1[2]={},s2[2]={};','s1[2]={},s2[2]={};'+('uint32_t raw[16];' if kind=='raw' else 'float xhat[16];'))
 if kind=='raw':
  a=v.index('    float xa=');b=v.index('    float ha=',a)
  v=v[:a]+'''    raw[q*4+j]=pair_get(lnsm+wi*8192,r,c);
    float xa=__fmul_rn(__fsub_rn(bf16lo(raw[q*4+j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(raw[q*4+j]),mu[rr]),rs[rr]);
'''+v[b:]
  for name,r,offset,half in [('xaa','ra',0,'lo'),('xab','ra',0,'hi'),('xba','rb',1,'lo'),('xbb','rb',1,'hi')]:
   col='c' if half=='lo' else 'c+1'
   v=v.replace('get(lnsm+wi*8192,%s,%s)'%(r,col),'bf16%s(raw[q*4+j+%d])'%(half,offset))
 else:
  v=v.replace('    float ha=','    if((j&1)==0){xhat[q*4+(j>>1)*2]=xa;xhat[q*4+(j>>1)*2+1]=xb;}\n    float ha=')
  a=v.index('    float xaa=');b=v.index('\n    float xba=',a)
  v=v[:a]+'    float xaa=xhat[q*4+pair*2],xab=xhat[q*4+pair*2+1];'+v[b:]
 name='front_prefetch_ln'+kind+'pair';(p/(name+'.cu')).write_text(v);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
