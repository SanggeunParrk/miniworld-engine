from pathlib import Path
import json
p=Path(__file__).resolve().parent;src=(p/'front_onewg_lowreg.cu').read_text();s=src[:src.index('TMN_DEVI void glu_small')];s=s.replace('DWCOUNT=8*DW_SPLITS','DWCOUNT=16*DW_SPLITS').replace('DW_SLOT=57344,DX_SLOT=40960','DW_SLOT=36864,DX_SLOT=20480')
ops=','.join('%%%d'%i for i in range(16));outs=','.join('"+f"(d[%d])'%i for i in range(16))
s+='''TMN_DEVI void mma_w32(float (&d)[16],uint64_t a,uint64_t b,int scale){asm volatile("{.reg .pred p;setp.ne.b32 p,%%18,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%s},%%16,%%17,p,1,1,1,0;}" : %s : "l"(a),"l"(b),"r"(scale));}\n'''%(ops,outs)
s+=r'''
TMN_DEVI void glu32(const Params& p,uint8_t* s,uint8_t* g,uint8_t* pp,int row){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll 2
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*128,c=i/32,r=(i%32)*2;uint32_t dy=pair_get(s+8192,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r),masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));*reinterpret_cast<uint32_t*>(g+swz128(c,r*2))=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));*reinterpret_cast<uint32_t*>(pp+swz128(c,r*2))=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);}
 fence_proxy_async();allsync();
}
TMN_DEVI void load_dw32(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/8,h=(group%8)*32;mbar_arrive_expect_tx(b+slot,28672);tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);tma_load_2d(s+8192,side?&p.dr:&p.dl,b+slot,row,h);for(int c=0;c<2;++c)tma_load_2d(s+12288+c*8192,&p.xn,b+slot,c*64,row);}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* bar){
 int group=blockIdx.x%16,split=blockIdx.x/16,tid=threadIdx.x,lane=tid%32,w=tid/32,r=0;int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,seg=(rounds+3)/4;float acc[4][16]={};
 // A short validation shape still writes all four segments, including empty ones.
 for(int i=tid;i<4*8192;i+=128)p.partw[(group*DW_SPLITS+split)*4*8192+i]=0.f;
 if(split<p.tiles)load_dw32(p,sm,bar,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw32(p,sm,bar,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(bar+slot,(r/2)&1);glu32(p,s,s+28672,s+32768,tile*64);
  static_for<4>([&](auto qi){constexpr int part=decltype(qi)::value;constexpr int kind=part/2,cb=part%2;fence_regs(acc[part]);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_w32(acc[part],smem_desc(smem_u32(s+12288+cb*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+28672+kind*4096+k*32),16,1024,1),r%seg>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc[part]);});allsync();if(tile+2*DW_SPLITS<p.tiles)load_dw32(p,sm,bar,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%seg==0||tile+DW_SPLITS>=p.tiles){static_for<4>([&](auto qi){constexpr int part=decltype(qi)::value;float* out=p.partw+((group*DW_SPLITS+split)*4+r/seg)*8192+(part/2)*4096+(part%2)*2048;static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=w*16+lane/4,c=j*8+2*(lane%4);stg64f(out+rr*32+c,acc[part][j*4],acc[part][j*4+1]);stg64f(out+(rr+8)*32+c,acc[part][j*4+2],acc[part][j*4+3]);});});}
 }
}
TMN_DEVI void load_g32(const Params& p,uint8_t* sm,uint64_t* bar,int row,int side,int h){if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(bar+slot,20480);tma_load_2d(s,&p.pre,bar+slot,row,side*512+h*64);tma_load_2d(s+8192,side?&p.dr:&p.dl,bar+slot,row,h*32);for(int c=0;c<2;++c)tma_load_2d(s+12288+c*4096,side?&p.wrg:&p.wlg,bar+slot,h*32,c*64);}
TMN_DEVI void load_p32(const Params& p,uint8_t* sm,uint64_t* bar,int side,int h){if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(bar+slot,8192);for(int c=0;c<2;++c)tma_load_2d(s+c*4096,side?&p.wr:&p.wl,bar+slot,h*32,c*64);}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=threadIdx.x,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8,round=0;float running_g=0,running_b=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;uint32_t packed[32];float acc[64]={};
  if(tid==0){mbar_arrive_expect_tx(bar+2,49152);for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar+2,k*64,row);for(int c=0;c<2;++c)tma_load_2d(sm+16384+k*16384+c*8192,&p.wgate,bar+2,k*64,c*64);}}
  mbar_wait(bar+2,ph);{float gate[64]={};fence_regs(gate);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_gate128(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+(k/4)*16384+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}allsync();
  for(int side=0;side<2;++side){load_g32(p,sm,bar,row,side,0);load_g32(p,sm,bar,row,side,1);
   for(int h=0;h<8;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,(h/2)&1);glu32(p,s,s+8192,sm+40960+h*4096,row);fence_regs(acc);wgmma_fence();static_for<2>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+8192+k*2048),16,1024,1),smem_desc(smem_u32(s+12288+k*32),16,512,2),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h+2<8)load_g32(p,sm,bar,row,side,h+2);}
   load_p32(p,sm,bar,side,0);load_p32(p,sm,bar,side,1);
   for(int h=0;h<8;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,(h/2)&1);fence_regs(acc);wgmma_fence();static_for<2>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+40960+h*4096+k*2048),16,1024,1),smem_desc(smem_u32(s+k*32),16,512,2),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h+2<8)load_p32(p,sm,bar,side,h+2);}
  }
'''
a=src.index('  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]');b=src.index('TMN_DEVI void reduce_at',a);ln=src[a:b].replace('bar+1,32768','bar+3,32768').replace('&p.x,bar+1','&p.x,bar+3').replace('&p.res,bar+1','&p.res,bar+3').replace('mbar_wait(bar+1,1);','mbar_wait(bar+3,ph);');s+=ln
s+=r'''
TMN_DEVI void reduce_at(const Params& p,int i){if(i<131072){int group=i/8192,j=i%8192,kind=j/4096,z=j%4096;float v=0;for(int q=0;q<DW_SPLITS*4;++q)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*4+q)*8192+j];int out=(group/8)*2+(kind==0?1:0),c=z/32,h=(group%8)*32+z%32;p.dw[(out*128+c)*256+h]=__float2bfloat16_rn(v);}else if(i<131328){int c=i-131072;float v=0;for(int q=0;q<DXCOUNT;++q)v+=reinterpret_cast<volatile float*>(p.partln)[q*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}}
extern "C" __global__ __launch_bounds__(128,3) void front_b7b12(__grid_constant__ const Params p){extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[4];__shared__ float gamma[128];gamma[threadIdx.x]=p.gamma[threadIdx.x];if(threadIdx.x==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}allsync();if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else input_role(p,sm,bar,gamma);__threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();for(int i=blockIdx.x*128+threadIdx.x;i<131328;i+=UCOUNT*128)reduce_at(p,i);__threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
''';(p/'front_onewg_h32.cu').write_text(s);(p/'front_onewg_h32.launch.json').write_text(json.dumps(dict(direct_weights=True,threads=128,shared=73728,max_ctas_per_sm=3,wgrad_slices=4)))
