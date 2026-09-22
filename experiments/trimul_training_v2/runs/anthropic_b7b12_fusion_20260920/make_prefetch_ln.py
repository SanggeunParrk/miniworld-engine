from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'front_kindprefetch.cu').read_text()
for cache in [False,True]:
 for pair in [False,True]:
  if not cache and not pair:continue
  v=s
  if cache:
   v=v.replace('s1[2]={},s2[2]={};','s1[2]={},s2[2]={};float xhat[32];')
   v=v.replace('float ha=acc[q*8+j*2]', 'xhat[q*8+j*2]=xa;xhat[q*8+j*2+1]=xb;float ha=acc[q*8+j*2]')
   start=v.index('    float xaa=');end=v.index('    float da=',start)
   v=v[:start]+'    float xaa=xhat[q*8+j*2],xab=xhat[q*8+j*2+1],xba=xhat[q*8+j*2+2],xbb=xhat[q*8+j*2+3];\n'+v[end:]
  if pair:
   start=v.index('    put(lnsm+wi*8192,ra,c,');end=v.index('    float dga=',start)
   v=v[:start]+'''    uint32_t resa=pair_get(lnsm+16384+wi*8192,ra,c),resb=pair_get(lnsm+16384+wi*8192,rb,c);
    *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(ra,c*2))=pack_bf16(bf16lo(outa)+bf16lo(resa),bf16hi(outa)+bf16hi(resa));
    *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(rb,c*2))=pack_bf16(bf16lo(outb)+bf16lo(resb),bf16hi(outb)+bf16hi(resb));
'''+v[end:]
  name='front_prefetch_ln%s%s'%('reg' if cache else '', 'pair' if pair else '')
  (p/(name+'.cu')).write_text(v);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
