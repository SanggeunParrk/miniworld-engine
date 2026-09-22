from pathlib import Path
r=Path(__file__).resolve().parent;s=(r/'unified_v1.cu').read_text();head=s[:s.index('template<bool WEIGHT>')].replace('named_bar_sync(0,512)','named_bar_sync(0,384)').replace('i<4096;i+=512','i<4096;i+=384')
body=r'''
template<bool WEIGHT> TMN_DEVI void urun3(const Params& p,uint8_t* sm,uint64_t* bars){
 const int tid=threadIdx.x%128,lane=tid%32,w=tid/32,wi=__shfl_sync(0xffffffff,threadIdx.x/128,0);
 const int first=p.tiles*blockIdx.x/UCOUNT,end=p.tiles*(blockIdx.x+1)/UCOUNT;
 float acc[3][64];if constexpr(WEIGHT){for(int n=0;n<3;++n)for(int q=0;q<64;++q)acc[n][q]=0;}
 int phase=0;
 for(int it=first;it<end;++it){uload(p,sm,bars,it*64,phase);phase^=1;
  if constexpr(!WEIGHT){udgrad(p,sm,bars+1,it*64);if(threadIdx.x==0){mbar_init(bars+1,1);fence_barrier_init();}}
  else{
   static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=(wi-1)*3+n;
    uint8_t* sa=t<2?sm+49152+t*8192:sm+((t-2)/2)*8192;
    uint8_t* sb=t<2?sm+16384:sm+65536+((t-2)%2)*16384;
    fence_regs(acc[n]);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});wgmma_commit();
   });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);
  }
  allsync();
 }
 if constexpr(WEIGHT){
  static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=(wi-1)*3+n,tile=t<2?0:1+(t-2)/2;
   float* part=p.partw+(tile*UCOUNT+blockIdx.x)*16384;
#pragma unroll
   for(int q=0;q<16;++q){int r=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;part[r*stride+c]=acc[n][4*q];part[r*stride+c+1]=acc[n][4*q+1];part[(r+8)*stride+c]=acc[n][4*q+2];part[(r+8)*stride+c+1]=acc[n][4*q+3];}
  });
 }else{
  float* red=reinterpret_cast<float*>(sm+180224);
  for(int j=tid;j<512;j+=128){float v=0;for(int w=0;w<4;++w)v+=red[w*512+j];p.partln[blockIdx.x*512+j]=v;}
 }
}
extern "C" __global__ __launch_bounds__(384,1) void unified3_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bars[2];
 if(threadIdx.x==0){for(int i=0;i<2;++i)mbar_init(bars+i,1);fence_barrier_init();}
 for(int i=threadIdx.x;i<2048;i+=384)reinterpret_cast<float*>(sm+180224)[i]=0;
 allsync();const int wg=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
 if(wg==0){setmaxnreg_dec<80>();urun3<false>(p,sm,bars);}
 else{setmaxnreg_inc<216>();urun3<true>(p,sm,bars);}
}
'''
(r/'unified3.cu').write_text(head+body+s[s.index('extern "C" __global__ void unified_reduce'):])
p=r/'unified3.py';a=(r/'unified.py').read_text().replace("source=R/'unified.cu'","source=R/'unified3.cu'").replace("kernel('unified_b1b4')","kernel('unified3_b1b4')").replace('(512,1,1)','(384,1,1)');p.write_text(a)
p=r/'check_unified3.py';a=(r/'check_reduce.py').read_text().replace('from unified import','from unified3 import').replace('for part in (0,1):','for part in (1,):');p.write_text(a)
