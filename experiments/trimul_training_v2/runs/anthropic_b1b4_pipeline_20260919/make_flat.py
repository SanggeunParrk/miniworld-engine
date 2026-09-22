from pathlib import Path
r=Path(__file__).resolve().parent;s=(r/'unified_v1.cu').read_text();head=s[:s.index('TMN_DEVI void uload')].replace('named_bar_sync(0,512)','named_bar_sync(0,256)')
dg=s[s.index('TMN_DEVI void udgrad'):s.index('template<bool WEIGHT>')]
a=dg.index('#pragma unroll\n for(int n=0;n<4;++n)');b=dg.index('  float acc[32]',a)
dg=dg[:a]+'''#pragma unroll
 for(int qn=0;qn<4;++qn){int n=(qn+3)%4;sw=sm+(n==0?131072:n==1?65536:n==2?81920:147456);
'''+dg[b:]
body=r'''
TMN_DEVI void flat_load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int phase,bool wp,bool first){
 if(threadIdx.x==0){mbar_arrive_expect_tx(bar,wp?65536:(first?163840:114688));
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.dy,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.gate,bar,c*64,row);
   if(!wp){tma_load_2d(sm+32768+c*8192,&p.proj,bar,c*64,row);tma_load_2d(sm+49152+c*8192,&p.xn,bar,c*64,row);}}
  if(wp){for(int c=0;c<4;++c)tma_load_2d(sm+65536+c*8192,&p.norm,bar,c*64,row);}
  else{tma_load_2d(sm+98304,&p.tri,bar,row,0);
#pragma unroll
   for(int n=0;n<4;++n){if(first||n==3){uint8_t* sw=sm+(n==0?131072:n==1?65536:n==2?81920:147456);for(int k=0;k<2;++k)tma_load_2d(sw+k*8192,&p.wp,bar,k*64,n*64);}}
  }
 }
 allsync();mbar_wait(bar,phase);int j0=row%p.L;
 for(int i=threadIdx.x;i<4096;i+=256){int c=i/2048*64+2*(i%32),rr=i%2048/32,jr=j0+rr;if(jr>=p.L)jr-=p.L;uint8_t* sy=sm+(c/64)*8192,*sg=sm+16384+(c/64)*8192,*sp=sm+32768+(c/64)*8192;
  uint32_t y=pair_get(sy,rr,c%64),g=pair_get(sg,rr,c%64),ds=ldg32(p.ds+jr*128+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
  if(!wp){uint32_t v=pair_get(sp,rr,c%64);uint32_t vdg=pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb));pair_put(sg,rr,c%64,vdg);stg32(p.dg+(size_t)(row+rr)*128+c,vdg);}
  pair_put(sy,rr,c%64,pack_bf16(ya*ga,yb*gb));
 }
 fence_proxy_async();allsync();
}
extern "C" __global__ __launch_bounds__(256,1) void flat_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar;
 const int split=blockIdx.x/2,role=blockIdx.x%2,wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 if(threadIdx.x==0){mbar_init(&bar,1);fence_barrier_init();}
 for(int i=threadIdx.x;i<2048;i+=256)reinterpret_cast<float*>(sm+180224)[i]=0;allsync();
 int first=p.tiles*split/UCOUNT,end=p.tiles*(split+1)/UCOUNT,phase=0;
 float acc[2][64]={};
 for(int it=first;it<end;++it){flat_load(p,sm,&bar,it*64,phase,role==1,it==first);phase^=1;
  if(role==0&&wi==0){udgrad(p,sm,&bar,it*64);}else{
   static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;uint8_t* sa=role==0?sm+49152+n*8192:sm+wi*8192;uint8_t* sb=role==0?sm+16384:sm+65536+n*16384;
    fence_regs(acc[n]);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});wgmma_commit();
   });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);
  }
  allsync();
 }
 if(role==0&&wi==0){float* red=reinterpret_cast<float*>(sm+180224);for(int j=tid;j<512;j+=128){float v=0;for(int w=0;w<4;++w)v+=red[w*512+j];p.partln[split*512+j]=v;}}
 else{
  int tile=role==0?0:1+wi;float* part=p.partw+(tile*UCOUNT+split)*16384;
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;
#pragma unroll
   for(int q=0;q<16;++q){int rr=w*16+lane/4+(role==0?n*64:0),c=q*8+2*(lane%4)+(role==0?0:n*128),stride=role==0?128:256;part[rr*stride+c]=acc[n][4*q];part[rr*stride+c+1]=acc[n][4*q+1];part[(rr+8)*stride+c]=acc[n][4*q+2];part[(rr+8)*stride+c+1]=acc[n][4*q+3];}
  });
 }
}
'''
(r/'flat.cu').write_text(head+dg+body+s[s.index('extern "C" __global__ void unified_reduce'):])
a=(r/'unified.py').read_text().replace("source=R/'unified.cu'","source=R/'flat.cu'").replace("kernel('unified_b1b4')","kernel('flat_b1b4')").replace('(512,1,1)','(256,1,1)').replace('self.grid=count','self.grid=count*2');(r/'flat.py').write_text(a)
a=(r/'check_reduce.py').read_text().replace('from unified import','from flat import').replace('for n in (384,768):','for n in (64,384,768):').replace('for part in (0,1):','for part in (1,):').replace('132,part','min(66,n*n//64),part');(r/'check_flat.py').write_text(a)
