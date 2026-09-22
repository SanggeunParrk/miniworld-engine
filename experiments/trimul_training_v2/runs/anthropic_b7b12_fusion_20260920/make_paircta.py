"""Merge two CTAs locally: H128 dW, two64-row dX tiles, four WGs.

Same per-output accumulation order as ring96/sp20. Reuse saved x_n between
the two hidden halves and projection weights between the two row tiles.
"""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'front_ring96_cache3.cu').read_text()
s=s.replace('#include "warp_primitives.cuh"','#define allsync allsync_256\n#include "warp_primitives.cuh"\n#undef allsync\nTMN_DEVI void allsync(){named_bar_sync(0,512);}')
s=s.replace('DWCOUNT=8*DW_SPLITS','DWCOUNT=4*DW_SPLITS').replace('DW_SLOT=57344,DX_SLOT=40960','DW_SLOT=98304,DX_SLOT=32768')
a=s.index('TMN_DEVI void glu_small');b=s.index('TMN_DEVI void load_dw',a);v=s[a:b].replace('q*256','q*512').replace('s+16384,c,r','s+32768,c,r');s=s[:a]+v+s[b:]
a=s.index('TMN_DEVI void load_dw');b=s.index('\nTMN_DEVI void ring_wait',a)
s=s[:a]+'''TMN_DEVI void load_dw(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/2,h=(group%2)*128;mbar_arrive_expect_tx(b+slot,65536);
 tma_first(s,&p.pre,b+slot,row,side*512+h*2);tma_first(s+32768,side?&p.dr:&p.dl,b+slot,row,h);
 for(int n=0;n<2;++n)tma_load_2d(s+49152+n*8192,&p.xn,b+slot,n*64,row);
}
'''+s[b:]
a=s.index('TMN_DEVI void ring_begin');b=s.index('TMN_DEVI void load_g',a)
s=s[:a]+'''TMN_DEVI void ring_begin(const Params& p,uint8_t* s,int tile,int group){
 if(threadIdx.x)return;int slot=tile%RING_TILES;if(tile>=RING_TILES)ring_wait(p.counts+2+8*RING_TILES+slot,tile-RING_TILES+1);
 int h=(group/2)*512+(group%2)*128;for(int k=0;k<2;++k){ring_store_last(&p.ring,s+65536+k*8192,slot*64,h+k*64);ring_store_last(&p.ring,s+81920+k*8192,slot*64,h+256+k*64);}tma_store_commit();
}
TMN_DEVI void ring_finish(const Params& p,int tile,int group){if(threadIdx.x)return;tma_store_wait_all();for(int k=0;k<2;++k)ring_publish(p.counts+2+(tile%RING_TILES)*8+group*2+k,tile+1);}
TMN_DEVI void ring_ready(const Params& p,int tile){if(threadIdx.x<16){int t=tile+threadIdx.x/8;ring_wait(p.counts+2+(t%RING_TILES)*8+threadIdx.x%8,t+1);}allsync();}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x%4,split=blockIdx.x/4,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[64]={};
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+65536,s+81920,tile*64);if(r>0)ring_finish(p,tile-DW_SPLITS,group);ring_begin(p,s,tile,group);
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_weight128(acc,smem_desc(smem_u32(s+65536+(wi%2)*16384+(wi/2)*8192+k*32),16,1024,1),smem_desc(smem_u32(s+49152+k*2048),8192,1024,1),r%segmentRounds>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(tile+2*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){float* out=p.partw+(((group*2+wi/2)*DW_SPLITS+split)*2+r/segmentRounds)*16384+(wi%2)*8192;
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}
 }
 if(rounds>0)ring_finish(p,split+(rounds-1)*DW_SPLITS,group);
}
'''+s[b:]
a=s.index('TMN_DEVI void load_g');b=s.index('TMN_DEVI void input_role',a)
s=s[:a]+'''TMN_DEVI void load_g(const Params& p,uint8_t* sm,uint64_t* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,49152);
 for(int m=0;m<2;++m){int r=(row+m*64)%(RING_TILES*64);tma_last(s+m*8192,&p.ring,b+slot,r,side*512+h*64);tma_last(sm+65536+h*16384+m*8192,&p.ring,b+slot,r,side*512+256+h*64);}
 for(int n=0;n<2;++n)tma_load_2d(s+16384+n*8192,side?&p.wrg:&p.wlg,b+slot,h*64,n*64);
}
TMN_DEVI void load_p(const Params& p,uint8_t* sm,uint64_t* b,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b+slot,h*64,n*64);
}
TMN_DEVI void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,65536);
 for(int k=0;k<2;++k){for(int m=0;m<2;++m)tma_load_2d(sm+m*16384+k*8192,&p.dg,bar,k*64,row+m*64);
  for(int n=0;n<2;++n)tma_load_2d(sm+32768+n*16384+k*8192,&p.wgate,bar,k*64,n*64);}
}
TMN_DEVI void load_ln_next(const Params& p,uint8_t* sm,uint64_t* bar,int row){if(threadIdx.x)return;mbar_arrive_expect_tx(bar,65536);for(int m=0;m<2;++m)for(int c=0;c<2;++c){tma_load_2d(sm+m*32768+c*8192,&p.x,bar,c*64,row+m*64);tma_load_2d(sm+m*32768+16384+c*8192,&p.res,bar,c*64,row+m*64);}}
'''+s[b:]
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
v=v.replace('wi=threadIdx.x/128,','wi=(threadIdx.x/128)%2,mg=threadIdx.x/256,')
v=v.replace('if(split<p.tiles)issue_gate(p,sm,bar+2,split*64);','if(split*2<p.tiles)issue_gate(p,sm,bar+2,split*128);')
v=v.replace('for(int tile=split;tile<p.tiles;tile+=DXCOUNT','for(int tile=split*2;tile<p.tiles;tile+=DXCOUNT*2')
v=v.replace('int row=tile*64;','int row=(tile+mg)*64;')
v=v.replace('sm+(k/4)*8192+(k%4)*32','sm+mg*16384+(k/4)*8192+(k%4)*32').replace('sm+16384+wi*16384+(k/4)*8192','sm+32768+wi*16384+(k/4)*8192')
v=v.replace('load_g(p,sm,bar,row,','load_g(p,sm,bar,tile*64,')
v=v.replace('ring_publish(p.counts+2+8*RING_TILES+tile%RING_TILES,tile+1);','{ring_publish(p.counts+2+8*RING_TILES+tile%RING_TILES,tile+1);ring_publish(p.counts+2+8*RING_TILES+(tile+1)%RING_TILES,tile+2);}')
v=v.replace('s+16384+k*2048','s+mg*8192+k*2048').replace('s+24576+wi*8192+k*32','s+16384+wi*8192+k*32')
v=v.replace('sm+81920+h*8192+k*2048','sm+65536+h*16384+mg*8192+k*2048')
v=v.replace('load_ln_next(p,sm+65536,bar+3,row)','load_ln_next(p,sm+131072,bar+3,tile*64)')
v=v.replace('if(tile+DXCOUNT<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);','if(tile+DXCOUNT*2<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT*2)*64);')
v=v.replace('uint8_t* lnsm=sm+65536;','uint8_t* lnsm=sm+131072+mg*32768;')
v=v.replace('reinterpret_cast<float*>(lnsm+36864)','reinterpret_cast<float*>(sm+73728+mg*1024)').replace('reinterpret_cast<float*>(lnsm+32768)','reinterpret_cast<float*>(sm+65536+mg*4096)')
v=v.replace('tmp[threadIdx.x]','tmp[threadIdx.x%256]').replace('tmp[256+threadIdx.x]','tmp[256+threadIdx.x%256]').replace('tmp[512+threadIdx.x]','tmp[512+threadIdx.x%256]').replace('tmp[768+threadIdx.x]','tmp[768+threadIdx.x%256]')
v=v.replace('if(threadIdx.x==0){store2d(&p.dx,lnsm,0,row);store2d(&p.dx,lnsm+8192,64,row);tma_store_commit();tma_store_wait_all();}','if(threadIdx.x==0){for(int m=0;m<2;++m){store2d(&p.dx,sm+131072+m*32768,0,(tile+m)*64);store2d(&p.dx,sm+131072+m*32768+8192,64,(tile+m)*64);}tma_store_commit();tma_store_wait_all();}')
v=v.replace('p.partln[split*256+threadIdx.x]','p.partln[split*512+threadIdx.x]')
s=s[:a]+v+s[b:]
s=s.replace('b<DXCOUNT;++b','b<DXCOUNT*2;++b').replace('__launch_bounds__(256,2)','__launch_bounds__(512,1)').replace('blockIdx.x*256+threadIdx.x','blockIdx.x*512+threadIdx.x').replace('i+=UCOUNT*256','i+=UCOUNT*512')
for unroll in [8,4,2]:
 name='front_ring_paircta_u%d'%unroll;v=s.replace('#pragma unroll 8','#pragma unroll '+str(unroll));(p/(name+'.cu')).write_text(v)
 cfg=dict(wgrad_slices=2,threads=512,shared=196608,max_ctas_per_sm=1,extra_counts=864,ring_tiles=96,front_hidden_tile=128)
 (p/(name+'.launch.json')).write_text(json.dumps(cfg))
