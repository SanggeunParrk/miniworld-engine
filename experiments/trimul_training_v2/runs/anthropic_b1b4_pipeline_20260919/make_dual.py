from pathlib import Path
r=Path(__file__).resolve().parent;s=(r/'unified_v1.cu').read_text();head=s[:s.index('TMN_DEVI void uload')].replace('named_bar_sync(0,512)','named_bar_sync(0,256)')
dg=(r/'dgrad_stats.cuh').read_text().replace('void udgrad_stats','void dual_dgrad').replace('const int tid=threadIdx.x,lane=tid%32,w=tid/32,','const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,').replace('sm+147456','sm+196608').replace('float* mus=stats+128','float* mus=stats+256').replace('if(tid<64){mus[tid]','if(threadIdx.x<64){mus[tid]').replace('gam[tid]=p.gamma[tid];gam[tid+128]=p.gamma[tid+128];sync_group();','gam[threadIdx.x]=p.gamma[threadIdx.x];allsync();')
dg=dg.replace('for(int qn=0;qn<4;++qn){int n=(qn+3)%4;uint8_t* sw=sm+(n==0?131072:n==1?65536:n==2?81920:147456);','for(int qn=0;qn<2;++qn){int n=wi*2+qn;uint8_t* sw=sm+131072+n*16384;')
dg=dg.replace('stats[ra*2]','stats[wi*128+ra*2]').replace('stats[ra*2+1]','stats[wi*128+ra*2+1]').replace('stats[rb*2]','stats[wi*128+rb*2]').replace('stats[rb*2+1]','stats[wi*128+rb*2+1]')
dg=dg.replace('s2[1];}sync_group();','s2[1];}allsync();').replace('sm+180224','sm+229376').replace('b<16;++b){int c=b*16','b<8;++b){int c=wi*128+b*16')
for v in ('r*2+1','r*2','(r+1)*2+1','(r+1)*2'):
 dg=dg.replace('stats['+v+']','(stats['+v+']+stats[128+'+v+'])')
dg=dg.replace('if(tid==0){for(int c=0;c<256;c+=16)','if(tid==0){for(int c=wi*128;c<(wi+1)*128;c+=16)')
body=r'''
TMN_DEVI void dual_load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int phase,bool first){
 if(threadIdx.x==0){mbar_arrive_expect_tx(bar,first?196608:131072);
#pragma unroll
 for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.dy,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.gate,bar,c*64,row);tma_load_2d(sm+32768+c*8192,&p.proj,bar,c*64,row);tma_load_2d(sm+49152+c*8192,&p.xn,bar,c*64,row);}
#pragma unroll
 for(int c=0;c<4;++c)tma_load_2d(sm+65536+c*8192,&p.norm,bar,c*64,row);
 tma_load_2d(sm+98304,&p.tri,bar,row,0);
 if(first)for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sm+131072+n*16384+k*8192,&p.wp,bar,k*64,n*64);
 }
 allsync();mbar_wait(bar,phase);int j0=row%p.L;
 for(int i=threadIdx.x;i<4096;i+=256){int c=i/2048*64+2*(i%32),r=i%2048/32,jr=j0+r;if(jr>=p.L)jr-=p.L;uint8_t* sy=sm+(c/64)*8192,*sg=sm+16384+(c/64)*8192,*sp=sm+32768+(c/64)*8192;
 uint32_t y=pair_get(sy,r,c%64),g=pair_get(sg,r,c%64),v=pair_get(sp,r,c%64),ds=ldg32(p.ds+jr*128+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);uint32_t z=pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb));pair_put(sg,r,c%64,z);stg32(p.dg+(size_t)(row+r)*128+c,z);pair_put(sy,r,c%64,pack_bf16(ya*ga,yb*gb));}
 fence_proxy_async();allsync();
}
extern "C" __global__ __launch_bounds__(256,1) void dual_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar;
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,split=blockIdx.x;
 if(threadIdx.x==0){mbar_init(&bar,1);fence_barrier_init();}
 for(int i=threadIdx.x;i<512;i+=256)reinterpret_cast<float*>(sm+229376)[i]=0;allsync();
 int first=p.tiles*split/UCOUNT,end=p.tiles*(split+1)/UCOUNT,phase=0;
 float acc[3][64]={};
 for(int it=first;it<end;++it){dual_load(p,sm,&bar,it*64,phase,it==first);phase^=1;
  static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n;uint8_t* sa=t<2?sm+49152+t*8192:sm+((t-2)/2)*8192;uint8_t* sb=t<2?sm+16384:sm+65536+((t-2)%2)*16384;fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);
  dual_dgrad(p,sm,&bar,it*64);allsync();
 }
 static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n,tile=t<2?0:1+(t-2)/2;float* part=p.partw+(tile*UCOUNT+split)*16384;
#pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;part[rr*stride+c]=acc[n][4*q];part[rr*stride+c+1]=acc[n][4*q+1];part[(rr+8)*stride+c]=acc[n][4*q+2];part[(rr+8)*stride+c+1]=acc[n][4*q+3];}});
 for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];
}
'''
(r/'dual.cu').write_text(head+dg+body+s[s.index('extern "C" __global__ void unified_reduce'):])
a=(r/'unified.py').read_text().replace("source=R/'unified.cu'","source=R/'dual.cu'").replace("kernel('unified_b1b4')","kernel('dual_b1b4')").replace('(512,1,1)','(256,1,1)').replace('188416','231424');(r/'dual.py').write_text(a)
a=(r/'tune_flat.py').read_text().replace('from flat import','from dual import').replace('for n in (384,768):','for n in (64,384,768):').replace('for count in (66,132,264,528):','for count in ((4,) if n<384 else (66,132,264)):').replace('flat-tune-','dual-tune-');(r/'check_dual.py').write_text(a)
