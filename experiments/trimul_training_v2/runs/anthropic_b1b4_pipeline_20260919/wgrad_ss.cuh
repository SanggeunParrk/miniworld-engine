// SM90 BF16 WGMMA SS, both operands MN-major (transpose flags 1,1).
// Instruction signature follows NVIDIA CUTLASS MMA_64x64x16_F32BF16BF16_SS.
TMN_DEVI void mma_ss(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 1, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void wgrad_ss(const Params& p,uint8_t* sm,uint64_t* bar,int* last){
 int tid=threadIdx.x%128,lane=tid%32,w=tid/32,b=(blockIdx.x-p.tiles)*WGROUPS+threadIdx.x/128,tile=b/SPLITS,split=b%SPLITS;
 bool wg=tile<4;int t=wg?tile:tile-4,ncols=wg?2:4,mch=(t/ncols)*64,nch=(t%ncols)*64;
 uint8_t *sy=sm,*sg=sm+8192,*sp=sm+16384,*sx=sm+24576;
 float acc[32]={};int chunks=(p.M+63)/64,first=(chunks*split)/SPLITS,end=(chunks*(split+1))/SPLITS;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}sync_group();
 auto load_tile = [&](int it,uint8_t* buf,uint64_t* barrier){int row=it*64,cc=wg?nch:mch;
  if(tid==0){mbar_arrive_expect_tx(barrier,wg?32768:24576);tma_load_2d(buf,&p.dy,barrier,cc,row);tma_load_2d(buf+8192,&p.gate,barrier,cc,row);if(wg)tma_load_2d(buf+16384,&p.proj,barrier,cc,row);tma_load_2d(buf+24576,wg?&p.xn:&p.norm,barrier,wg?mch:nch,row);}
 };
 auto transform_tile = [&](int it,uint8_t* buf){uint8_t *sy=buf,*sg=buf+8192,*sp=buf+16384;int row=it*64,cc=wg?nch:mch;
  int j0=row%p.L;
  for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint32_t y=pair_get(sy,r,c),g=pair_get(sg,r,c),v=wg?pair_get(sp,r,c):0,ds=ldg32(p.ds+jr*128+cc+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
   pair_put(sy,r,c,pack_bf16(wg?((ya*bf16lo(v))*ga)*(1.f-ga):ya*ga,wg?((yb*bf16hi(v))*gb)*(1.f-gb):yb*gb));
  }
 };

 if(WSS>=2){if(tid==0){mbar_init(bar+1,1);fence_barrier_init();}sync_group();}
 if(first<end)load_tile(first,sm,bar);
 for(int it=first;it<end;++it){
  int stage=WSS>=2?(it-first)&1:0;uint8_t* buf=sm+stage*32768;uint64_t* barrier=bar+stage;
  sync_group();mbar_wait(barrier,((it-first)/(WSS>=2?2:1))&1);
  transform_tile(it,buf);sync_group();fence_proxy_async();sync_group();
  uint8_t* sa=wg?buf+24576:buf;uint8_t* sb=wg?buf:buf+24576;
  fence_regs(acc);wgmma_fence();
  static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   uint64_t da=smem_desc(smem_u32(sa+k*2048),16,1024,1),db=smem_desc(smem_u32(sb+k*2048),16,1024,1);
   mma_ss(acc,da,db,it>first||k>0);
  });wgmma_commit();
  if(WSS>=2&&it+1<end)load_tile(it+1,sm+(1-stage)*32768,bar+1-stage);
  wgmma_wait<0>();fence_regs(acc);fence_proxy_async();sync_group();
  if(WSS==1&&it+1<end)load_tile(it+1,sm,bar);
 }
 float* part=p.partw+(tile*SPLITS+split)*4096;
#pragma unroll
 for(int q=0;q<8;++q){int r=w*16+lane/4,c=q*8+2*(lane%4);part[r*64+c]=acc[4*q];part[r*64+c+1]=acc[4*q+1];part[(r+8)*64+c]=acc[4*q+2];part[(r+8)*64+c+1]=acc[4*q+3];}
 if(ticket(p.counts+tile,SPLITS,last)){
  for(int i=tid;i<4096;i+=128){float v=0;for(int s=0;s<SPLITS;++s)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*SPLITS+s)*4096+i];int r=mch+i/64,c=nch+i%64;(wg?p.dwg:p.dwp)[r*(wg?128:256)+c]=__float2bfloat16_rn(v);}
  sync_group();if(tid==0)atomicExch(p.counts+tile,0u);
 }
}
