from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'front_rows512_r40_216.cu').read_text();a=s.index('constexpr int FRONT_SLOT');b=s.index('TMN_DEVI void dx_consumer',a)
body=r'''
constexpr int FRONT_SLOT=114688;
TMN_DEVI void load_gate128(const Params& p,uint8_t* sm,Barriers* b,int row){
 if(threadIdx.x)return;uint8_t* s=sm+FRONT_SLOT;mbar_arrive_expect_tx(b->tx+1,65536);
 for(int m=0;m<2;++m)for(int k=0;k<2;++k)tma_load_2d(s+m*16384+k*8192,&p.dg,b->tx+1,k*64,row+m*64);
 for(int k=0;k<2;++k)for(int n=0;n<2;++n)tma_load_2d(s+32768+k*16384+n*8192,&p.wgate,b->tx+1,k*64,n*64);
}
TMN_DEVI void load_front(const Params& p,uint8_t* sm,Barriers* b,int group,int row){
 if(threadIdx.x)return;int slot=group&1,side=group/4,h=group%4;uint8_t* s=sm+slot*FRONT_SLOT;mbar_arrive_expect_tx(b->tx+slot,81920);
 for(int m=0;m<2;++m){tma_load_2d(s+m*16384,&p.pre,b->tx+slot,row+m*64,side*512+h*128);tma_load_2d(s+32768+m*8192,side?&p.dr:&p.dl,b->tx+slot,row+m*64,h*64);}
 for(int kind=0;kind<2;++kind)for(int n=0;n<2;++n)tma_load_2d(s+49152+kind*16384+n*8192,side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg),b->tx+slot,h*64,n*64);
}
TMN_DEVI void load_ln128(const Params& p,uint8_t* sm,Barriers* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx,65536);
 for(int m=0;m<2;++m)for(int c=0;c<2;++c){tma_load_2d(sm+m*16384+c*8192,&p.x,b->tx,c*64,row+m*64);tma_load_2d(sm+32768+m*16384+c*8192,&p.res,b->tx,c*64,row+m*64);}
}
TMN_DEVI void glu_front(const Params& p,uint8_t* sm,int group,int row){
 unsigned tid=threadIdx.x;uint8_t* s=sm+(group&1)*FRONT_SLOT;
 float ma0=__bfloat162float(p.mask[row+(tid%32)*2]),mb0=__bfloat162float(p.mask[row+(tid%32)*2+1]);
 float ma1=__bfloat162float(p.mask[row+64+(tid%32)*2]),mb1=__bfloat162float(p.mask[row+64+(tid%32)*2+1]);
 #pragma unroll 1
 for(unsigned q=0;q<16;++q){unsigned i=tid+q*256,m=i/2048,c=(i%2048)/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+32768+m*8192,c,r),gl=pair_get(s+m*16384,c*2,r),pr=pair_get(s+m*16384,c*2+1,r);
  uint32_t masked=pack_bf16(bf16lo(dy)*(m?ma1:ma0),bf16hi(dy)*(m?mb1:mb0));float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
  uint32_t gg=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(s+81920+m*8192+swz128(c,r*2))=gg;*reinterpret_cast<uint32_t*>(s+98304+m*8192+swz128(c,r*2))=pp;
 }
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Barriers* b){
 int split=blockIdx.x-DWCOUNT,round=0,tiles=p.tiles/2;
 for(int tile=split;tile<tiles;tile+=DXCOUNT,++round){int row=tile*128,ph=round&1;
  if(round==0)load_gate128(p,sm,b,row);
  if(threadIdx.x==0){if(round)mbar_wait(b->empty,ph^1);load_front(p,sm,b,0,row);mbar_wait(b->empty+1,ph);load_front(p,sm,b,1,row);}
  for(int group=0;group<8;++group){int slot=group&1,phase=ph^((group/2+slot)&1);
   mbar_wait(b->tx+slot,phase);glu_front(p,sm,group,row);ready(p,b,slot);
   if(threadIdx.x==0){mbar_wait(b->empty+slot,phase);if(group<6)load_front(p,sm,b,group+2,row);else if(slot==0)load_ln128(p,sm,b,row);else if(tile+DXCOUNT<tiles)load_gate128(p,sm,b,(tile+DXCOUNT)*128);}
  }
 }
}
'''
s=s[:a]+body+s[b:]
a=s.index('  for(int side=0;side<2;++side)',s.index('TMN_DEVI void dx_consumer'));b=s.index('  static_for<32>',a)
s=s[:a]+r'''
  // Experimental accumulation order: Lg64,Lp64,... then right. Rounding boundaries are unchanged.
  for(int group=0;group<8;++group){int slot=group&1;mbar_wait(b->ready+slot,(group/2)&1);uint8_t* s=sm+slot*FRONT_SLOT;
   fence_regs(acc);wgmma_fence();
   static_for<2>([&](auto ni){constexpr int kind=decltype(ni)::value;static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+81920+kind*16384+wi*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+49152+kind*16384+k*32),16,1024,1),group>0||kind>0||k>0);});});
   wgmma_commit();wgmma_wait<0>();fence_regs(acc);dxrelease(b,slot);
  }
'''+s[b:]
s=s.replace('reinterpret_cast<float*>(sm+PGRAD)','reinterpret_cast<float*>(sm+65536)')
(p/'front_stream512.cu').write_text(s);(p/'front_stream512.launch.json').write_text('{"wgrad_slices":2,"threads":512}\n')
