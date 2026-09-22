from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'front_roundonly.cu').read_text();prefix=s[:s.index('TMN_DEVI void write_glu')]
body=r'''
TMN_DEVI void load_prod(const Params& p,uint8_t* sm,uint64_t* b,int group,int row){
 if(threadIdx.x)return;int slot=group&1;uint8_t* s=sm+slot*49152;mbar_arrive_expect_tx(b+slot,49152);
 tma_load_2d(s,&p.pre,b+slot,row,(group/2)*512+(group%2)*256);
 tma_load_2d(s+32768,group>=2?&p.dr:&p.dl,b+slot,row,(group%2)*128);
}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b){
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT){int row=tile*64;load_prod(p,sm,b,0,row);load_prod(p,sm,b,1,row);
  float ma=__bfloat162float(p.mask[row+(threadIdx.x%32)*2]),mb=__bfloat162float(p.mask[row+(threadIdx.x%32)*2+1]);
  for(int group=0;group<4;++group){int slot=group&1;mbar_wait(b+slot,group/2);uint8_t* s=sm+slot*49152;
   #pragma unroll 4
   for(int q=0;q<16;++q){int i=threadIdx.x+q*256,c=i/32,r=(i%32)*2;uint32_t g,pr;glu_pair(p,s,i,row,ma,mb,g,pr);
    int h=(group%2)*128+c,side=group/2;
    *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+h)*p.M+row+r)=g;
    *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+256+h)*p.M+row+r)=pr;
   }allsync();if(group<2)load_prod(p,sm,b,group+2,row);
  }
 }
}
extern "C" __global__ __launch_bounds__(256,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[2];
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}allsync();producer(p,sm,bar);
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){}
'''
(p/'front_producer_bench.cu').write_text(prefix+body);(p/'front_producer_bench.launch.json').write_text('{"direct_weights":true}\n')
