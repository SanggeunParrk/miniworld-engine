from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_lnpair.cu').read_text()
a=src.index('TMN_DEVI void weight_role');b=src.index('TMN_DEVI void load_g',a)
body=r'''
TMN_DEVI void load_pre3(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){if(threadIdx.x)return;uint8_t* s=sm+slot*24576;int side=group/4,h=(group%4)*64;mbar_arrive_expect_tx(b+slot,24576);tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h);}
TMN_DEVI void load_xn2(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row){if(threadIdx.x)return;uint8_t* s=sm+73728+slot*16384;mbar_arrive_expect_tx(b+3+slot,16384);for(int n=0;n<2;++n)tma_load_2d(s+n*8192,&p.xn,b+3+slot,n*64,row);}
TMN_DEVI void glu_inplace3(const Params& p,uint8_t* s,int row){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2),gg[8],pp[8];
 static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;unsigned i=tid+q*256,c=i/32,r=(i%32)*2;uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r),masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));gg[q]=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));pp[q]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);});
 allsync();
 static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;unsigned i=tid+q*256,c=i/32,r=(i%32)*2;*reinterpret_cast<uint32_t*>(s+swz128(c,r*2))=gg[q];*reinterpret_cast<uint32_t*>(s+8192+swz128(c,r*2))=pp[q];});fence_proxy_async();allsync();
}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x%8,split=blockIdx.x/8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[64]={};
 for(int q=0;q<3;++q)if(split+q*DW_SPLITS<p.tiles)load_pre3(p,sm,b,q,(split+q*DW_SPLITS)*64,group);
 for(int q=0;q<2;++q)if(split+q*DW_SPLITS<p.tiles)load_xn2(p,sm,b,q,(split+q*DW_SPLITS)*64);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int ps=r%3,xs=r&1;uint8_t* s=sm+ps*24576;uint8_t* xn=sm+73728+xs*16384;
  mbar_wait(b+ps,(r/3)&1);mbar_wait(b+3+xs,(r/2)&1);glu_inplace3(p,s,tile*64);
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc,smem_desc(smem_u32(s+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(xn+k*2048),8192,1024,1),r%segmentRounds>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(tile+3*DW_SPLITS<p.tiles)load_pre3(p,sm,b,ps,(tile+3*DW_SPLITS)*64,group);if(tile+2*DW_SPLITS<p.tiles)load_xn2(p,sm,b,xs,(tile+2*DW_SPLITS)*64);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+wi*8192;static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}
 }
}
'''
s=src[:a]+body+src[b:];s=s.replace('uint64_t bar[4]','uint64_t bar[5]').replace('i<4;++i)mbar_init','i<5;++i)mbar_init');name='front_prefetch_dwthree';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
