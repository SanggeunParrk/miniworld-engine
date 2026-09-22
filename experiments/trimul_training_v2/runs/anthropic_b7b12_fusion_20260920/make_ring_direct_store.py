from pathlib import Path
import json
p=Path(__file__).resolve().parent;src=(p/'front_ring96_cache3_storepipe.cu').read_text()
for hint in [False,True]:
 s=src.replace(' const __nv_bfloat16* mask;',' __nv_bfloat16* ring_data;const __nv_bfloat16* mask;')
 helper='TMN_DEVI void ring_word(__nv_bfloat16* dst,uint32_t v){'
 if hint:helper+='asm volatile("{.reg .b64 policy;createpolicy.fractional.L2::evict_last.b64 policy,1.0;st.global.L2::cache_hint.u32 [%0],%1,policy;}"::"l"(dst),"r"(v):"memory");'
 else:helper+='*reinterpret_cast<uint32_t*>(dst)=v;'
 helper+='}\n';a=s.index('TMN_DEVI void glu_small');s=s[:a]+helper+s[a:]
 a=s.index('TMN_DEVI void glu_small');b=s.index('TMN_DEVI void load_dw',a);v=s[a:b];v=v.replace('uint32_t g=pack_bf16','uint32_t g=pack_bf16')
 v=v.replace(' }fence_proxy_async();allsync();','  int group=blockIdx.x%8;size_t off=((group/4)*512+(group%4)*64+c)*(RING_TILES*64)+(row%(RING_TILES*64))+r;ring_word(p.ring_data+off,g);ring_word(p.ring_data+off+256*(RING_TILES*64),pp);\n }fence_proxy_async();allsync();');s=s[:a]+v+s[b:]
 a=s.index('TMN_DEVI void ring_begin');b=s.index('TMN_DEVI void ring_ready',a)
 s=s[:a]+'''TMN_DEVI void ring_begin(const Params& p,uint8_t* s,int tile,int group){if(threadIdx.x==0&&tile>=RING_TILES)ring_wait(p.counts+2+8*RING_TILES+tile%RING_TILES,tile-RING_TILES+1);allsync();}
TMN_DEVI void ring_finish(const Params& p,int tile,int group){__threadfence();allsync();if(threadIdx.x==0)ring_publish(p.counts+2+(tile%RING_TILES)*8+group,tile+1);}
'''+s[b:]
 s=s.replace('glu_small(p,s,s+40960,s+49152,tile*64);if(r>0)ring_finish(p,tile-DW_SPLITS,group);ring_begin(p,s,tile,group);','ring_begin(p,s,tile,group);glu_small(p,s,s+40960,s+49152,tile*64);')
 a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b].replace('fence_regs(acc);allsync();','fence_regs(acc);allsync();ring_finish(p,tile,group);').replace(' if(rounds>0)ring_finish(p,split+(rounds-1)*DW_SPLITS,group);','');s=s[:a]+v+s[b:]
 name='front_ring96_direct'+('_last' if hint else '');(p/(name+'.cu')).write_text(s);c=json.loads((p/'front_ring96_cache3.launch.json').read_text());c['ring_direct_store']=True;(p/(name+'.launch.json')).write_text(json.dumps(c))
