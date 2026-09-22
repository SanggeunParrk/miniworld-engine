from pathlib import Path
import json
p=Path(__file__).resolve().parent;kind=(p/'front_twocta_kindwg.cu').read_text();warp=(p/'front_warp.cu').read_text();a=kind.index('TMN_DEVI void weight_role');b=kind.index('TMN_DEVI void load_g',a)
s=kind[:a];s=s.replace('constexpr int DW_SLOT=57344,DX_SLOT=40960;','constexpr int DW_SLOT=57344,DX_SLOT=40960;\nTMN_DEVI void sync128(){named_bar_sync(0,128);}\n#define allsync sync128')
s=s.replace('q<8;++q','q<16;++q').replace('tid+q*256','tid+q*128').replace('#pragma unroll 2','#pragma unroll 2')
# dW one WG owns gate + projection, N128 each. Single shared input/output stage.
s+=r'''
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x%8,split=blockIdx.x/8,lane=threadIdx.x%32,w=threadIdx.x/32;float acc[2][64]={};int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){load_dw(p,sm,b,0,tile*64,group);mbar_wait(b,r&1);glu_small(p,sm,sm+40960,sm+49152,tile*64);
  static_for<2>([&](auto qi){constexpr int kind=decltype(qi)::value;fence_regs(acc[kind]);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc[kind],smem_desc(smem_u32(sm+40960+kind*8192+k*32),16,1024,1),smem_desc(smem_u32(sm+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc[kind]);});allsync();
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){static_for<2>([&](auto ki){constexpr int kind=decltype(ki)::value;float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+kind*8192;static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[kind][q*4],acc[kind][q*4+1]);stg64f(out+(rr+8)*128+c,acc[kind][q*4+2],acc[kind][q*4+3]);});});}
 }
}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=threadIdx.x,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8,round=0;float running_g=0,running_b=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;uint32_t packed[32];float acc[64]={};
  if(tid==0){mbar_arrive_expect_tx(bar+1,49152);for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar+1,k*64,row);for(int c=0;c<2;++c)tma_load_2d(sm+16384+k*16384+c*8192,&p.wgate,bar+1,k*64,c*64);}}
  mbar_wait(bar+1,0);{float gate[64]={};fence_regs(gate);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_gate128(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+(k/4)*16384+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}allsync();
  for(int side=0;side<2;++side){
   for(int h=0;h<4;++h){if(tid==0){mbar_arrive_expect_tx(bar,40960);tma_load_2d(sm,&p.pre,bar,row,side*512+h*128);tma_load_2d(sm+16384,side?&p.dr:&p.dl,bar,row,h*64);for(int c=0;c<2;++c)tma_load_2d(sm+24576+c*8192,side?&p.wrg:&p.wlg,bar,h*64,c*64);}
    mbar_wait(bar,h&1);glu_small(p,sm,sm+16384,sm+40960+h*8192,row);fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+16384+k*2048),16,1024,1),smem_desc(smem_u32(sm+24576+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
   }
   for(int h=0;h<4;++h){if(tid==0){mbar_arrive_expect_tx(bar,16384);for(int c=0;c<2;++c)tma_load_2d(sm+c*8192,side?&p.wr:&p.wl,bar,h*64,c*64);}mbar_wait(bar,h&1);fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+40960+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(sm+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();}
  }
'''
a=warp.index('  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]');b=warp.index('TMN_DEVI void reduce_at',a);tail=warp[a:b]
tail=tail.replace('mbar_wait(b->ready,ph);','if(tid==0){mbar_arrive_expect_tx(bar+1,32768);for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,bar+1,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,bar+1,c*64,row);}}mbar_wait(bar+1,1);')
tail=tail.replace('csync()','allsync()').replace('release(b,0);','allsync();')
s+=tail;s+=kind[kind.index('TMN_DEVI void reduce_at'):];s=s.replace('__launch_bounds__(256,2)','__launch_bounds__(128,3)').replace('blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256','blockIdx.x*128+threadIdx.x;i<131328;i+=UCOUNT*128')
(p/'front_onewg.cu').write_text(s);(p/'front_onewg.launch.json').write_text(json.dumps(dict(direct_weights=True,threads=128,shared=73728,max_ctas_per_sm=3,wgrad_slices=2)))
