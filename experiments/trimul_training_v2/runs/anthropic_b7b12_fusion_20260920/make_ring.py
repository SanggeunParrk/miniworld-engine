from pathlib import Path
import json
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_lnpair.cu').read_text()
for window in [128,256]:
 s='#define RING_TILES %d\n'%window+src;s=s.replace('wgate,x,res,dx;','wgate,x,res,dx,ring;')
 a=s.index('TMN_DEVI void weight_role')
 helper=r'''TMN_DEVI void ring_wait(const unsigned* ptr,unsigned want){unsigned got;do{asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");if(got<want)__nanosleep(32);}while(got<want);}
TMN_DEVI void ring_publish(unsigned* ptr,unsigned value){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(ptr),"r"(value):"memory");}
TMN_DEVI void ring_begin(const Params& p,uint8_t* s,int tile,int group){if(threadIdx.x)return;int slot=tile%RING_TILES;if(tile>=RING_TILES)ring_wait(p.counts+2+8*RING_TILES+slot,tile-RING_TILES+1);int h=(group/4)*512+(group%4)*64;store2d(&p.ring,s+40960,slot*64,h);store2d(&p.ring,s+49152,slot*64,h+256);tma_store_commit();}
TMN_DEVI void ring_finish(const Params& p,int tile,int group){if(threadIdx.x)return;tma_store_wait_all();ring_publish(p.counts+2+(tile%RING_TILES)*8+group,tile+1);}
TMN_DEVI void ring_ready(const Params& p,int tile){if(threadIdx.x<8)ring_wait(p.counts+2+(tile%RING_TILES)*8+threadIdx.x,tile+1);allsync();}
''';s=s[:a]+helper+s[a:]
 a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b];v=v.replace('glu_small(p,s,s+40960,s+49152,tile*64);','glu_small(p,s,s+40960,s+49152,tile*64);ring_begin(p,s,tile,group);')
 v=v.replace('fence_regs(acc);allsync();','fence_regs(acc);allsync();ring_finish(p,tile,group);');s=s[:a]+v+s[b:]
 a=s.index('TMN_DEVI void load_g');b=s.index('TMN_DEVI void load_p',a);v=s[a:b]
 v=v.replace('mbar_arrive_expect_tx(b+slot,40960);','mbar_arrive_expect_tx(b+slot,32768);')
 v=v.replace('tma_load_2d(s,&p.pre,b+slot,row,side*512+h*128);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h*64);','tma_load_2d(s+16384,&p.ring,b+slot,row%(RING_TILES*64),side*512+h*64);tma_load_2d(sm+81920+h*8192,&p.ring,b+slot,row%(RING_TILES*64),side*512+256+h*64);');s=s[:a]+v+s[b:]
 a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
 v=v.replace('  allsync();\n  for(int side=0;side<2;++side){','  allsync();ring_ready(p,tile);\n  for(int side=0;side<2;++side){')
 v=v.replace('glu_small(p,s,s+16384,sm+81920+h*8192,row);','')
 v=v.replace('  if(tile+DXCOUNT<p.tiles)issue_gate','  if(threadIdx.x==0)ring_publish(p.counts+2+8*RING_TILES+tile%RING_TILES,tile+1);\n  if(tile+DXCOUNT<p.tiles)issue_gate');s=s[:a]+v+s[b:]
 s=s.replace(' for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);',' for(int i=blockIdx.x*256+threadIdx.x;i<9*RING_TILES;i+=UCOUNT*256)p.counts[2+i]=0;\n for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);')
 name='front_ring%d'%window;(p/(name+'.cu')).write_text(s);c=json.loads((p/'front_kindprefetch.launch.json').read_text());c['extra_counts']=9*window;c['ring_tiles']=window;(p/(name+'.launch.json')).write_text(json.dumps(c))
