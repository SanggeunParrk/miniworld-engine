TMN_DEVI void wp_load(const Params& p,uint8_t* sm,uint64_t* bar,int row){if(threadIdx.x==0){mbar_arrive_expect_tx(bar,65536);
#pragma unroll
for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.dy,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.gate,bar,c*64,row);}
#pragma unroll
for(int c=0;c<4;++c)tma_load_2d(sm+32768+c*8192,&p.norm,bar,c*64,row);}}
TMN_DEVI void flat_wp(const Params& p,uint8_t* sm,uint64_t* bars){
 const int tid=threadIdx.x%128,lane=tid%32,w=tid/32,wi=threadIdx.x/128,split=blockIdx.x/2;
 const int first=p.tiles*split/UCOUNT,end=p.tiles*(split+1)/UCOUNT;
 if(threadIdx.x==0){for(int i=0;i<2;++i)mbar_init(bars+i,1);fence_barrier_init();}allsync();
 wp_load(p,sm,bars,first*64);if(first+1<end)wp_load(p,sm+65536,bars+1,(first+1)*64);
 float acc[2][64]={};
 for(int it=first;it<end;++it){int st=(it-first)&1,phase=((it-first)/2)&1,row=it*64;uint8_t* b=sm+st*65536;allsync();mbar_wait(bars+st,phase);
  int j0=row%p.L;
  for(int i=threadIdx.x;i<4096;i+=256){int c=i/2048*64+2*(i%32),r=i%2048/32,jr=j0+r;if(jr>=p.L)jr-=p.L;uint8_t* sy=b+(c/64)*8192;uint32_t y=pair_get(sy,r,c%64),g=pair_get(b+16384+(c/64)*8192,r,c%64),ds=ldg32(p.ds+jr*128+c);pair_put(sy,r,c%64,pack_bf16((bf16lo(y)*bf16lo(ds))*bf16lo(g),(bf16hi(y)*bf16hi(ds))*bf16hi(g)));}
  fence_proxy_async();allsync();
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;uint8_t *sa=b+wi*8192,*sb=b+32768+n*16384;fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);allsync();if(it+2<end)wp_load(p,b,bars+st,(it+2)*64);
 }
 float* part=p.partw+((1+wi)*UCOUNT+split)*16384;
 static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;
#pragma unroll
 for(int q=0;q<16;++q){int r=w*16+lane/4,c=q*8+2*(lane%4)+n*128;part[r*256+c]=acc[n][4*q];part[r*256+c+1]=acc[n][4*q+1];part[(r+8)*256+c]=acc[n][4*q+2];part[(r+8)*256+c+1]=acc[n][4*q+3];}});
}
