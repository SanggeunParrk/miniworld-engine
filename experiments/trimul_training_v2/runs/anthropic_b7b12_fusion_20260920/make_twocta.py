from pathlib import Path
p=Path(__file__).resolve().parent
old=(p/'front_roundonly.cu').read_text();warp=(p/'front_rows128_p48c224.cu').read_text()
prefix=warp[:warp.index('struct Barriers')].replace('constexpr int DW_STAGES=5,DW_SLOT=40960,DX_SLOT=114688;','constexpr int DW_SLOT=57344,DX_SLOT=40960;').replace('static_assert(DXCOUNT>0 && DW_PREFETCH<DW_STAGES,"invalid persistent partition");','static_assert(DXCOUNT>0,"invalid partition");')
prim=(p/'front_primitives.cuh').read_text();mma=prim[prim.index('TMN_DEVI void mma_dgrad'):].replace('mma_dgrad','mma_weight64').replace('p, 1, 1, 0, 0','p, 1, 1, 0, 1')
store=old[old.index('TMN_DEVI void store2d'):old.index('// Each pair')]
body=r'''
TMN_DEVI void glu_small(const Params& p,uint8_t* s,uint8_t* gout,uint8_t* pout,int row){
 unsigned tid=threadIdx.x;float ma=__bfloat162float(p.mask[row+(tid%32)*2]),mb=__bfloat162float(p.mask[row+(tid%32)*2+1]);
 #pragma unroll 2
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*256,c=i/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);
  uint32_t masked=pack_bf16(bf16lo(dy)*ma,bf16hi(dy)*mb);float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(gout+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(pout+swz128(c,r*2))=pp;
 }fence_proxy_async();allsync();
}
TMN_DEVI void load_dw(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/4,h=(group%4)*64;mbar_arrive_expect_tx(b+slot,40960);
 tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,&p.xn,b+slot,n*64,row);
}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x/DW_SPLITS,split=blockIdx.x%DW_SPLITS,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[2][32]={};
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;fence_regs(acc[n]);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_weight64(acc[n],smem_desc(smem_u32(s+40960+n*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+wi*8192+k*2048),16,1024,1),r%segmentRounds>0||k>0);
  });});wgmma_commit();wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);allsync();
  if(tile+2*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+n*8192;
   static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=wi*64+q*8+2*(lane%4);stg64f(out+rr*128+c,acc[n][q*4],acc[n][q*4+1]);stg64f(out+(rr+8)*128+c,acc[n][q*4+2],acc[n][q*4+3]);});});}
 }
}
TMN_DEVI void load_g(const Params& p,uint8_t* sm,uint64_t* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,40960);
 tma_load_2d(s,&p.pre,b+slot,row,side*512+h*128);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h*64);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg,b+slot,h*64,n*64);
}
TMN_DEVI void load_p(const Params& p,uint8_t* sm,uint64_t* b,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b+slot,h*64,n*64);
}
'''
gate=old[old.index('TMN_DEVI void issue_gate'):old.index('TMN_DEVI void issue_dx')]
a=old.index('TMN_DEVI void input_role');b=old.index('  // B10 outputs',a)
inhead=old[a:old.index('  allsync();issue_dx',a)]
inbody=r'''
  allsync();
  for(int side=0;side<2;++side){
   load_g(p,sm,bar,row,side,0);load_g(p,sm,bar,row,side,1);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,1^(h/2)^slot);glu_small(p,s,s+16384,sm+81920+h*8192,row);
    fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+24576+wi*8192+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_g(p,sm,bar,row,side,h+2);
   }
   load_p(p,sm,bar,side,0);load_p(p,sm,bar,side,1);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,1^(h/2)^slot);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+wi*8192+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_p(p,sm,bar,side,h+2);
   }
  }
'''
intail=old[b:old.index('TMN_DEVI void reduce_at',b)]
reduce=warp[warp.index('TMN_DEVI void reduce_at'):warp.index('extern "C" __global__ __launch_bounds__')]
main=old[old.index('extern "C" __global__ __launch_bounds__'):].replace('__launch_bounds__(256,1)','__launch_bounds__(256,2)')
(p/'front_twocta.cu').write_text(prefix+mma+store+body+gate+inhead+inbody+intail+reduce+main)
(p/'front_twocta.launch.json').write_text('{"wgrad_slices":2,"threads":256,"shared":114688,"max_ctas_per_sm":2}\n')
