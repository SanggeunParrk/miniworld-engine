// D >= 384 forward, first half: LayerNorm inside the SwiGLU expand GEMM (h = silu(xn Wa^T) * xn Wb^T), xn / rstd / c1 saved
// behind a runtime flag. See kernels/ln_swiglu_gemm.cu. KD is the channel width.
#define KD WIDE_D
#define SWP 1                                             // software-pipelined epilogue: -6 % (D384) / -8 % (D512) at fixed clocks
#define ln_swiglu_gemm wide_lnsg_kernel
#include "kernels/ln_swiglu_gemm.cu"
#include "launch_util.cuh"

void wide_lnsg_launch(const CUtensorMap& mx, const CUtensorMap& mw1p, const CUtensorMap& mh, const float* gamma, const float* beta,
                      __nv_bfloat16* xn, float* rstd, float* c1, int M, int H, float eps, int save, int ctas, cudaStream_t stream
#if MW_SAVE_AB
                      ,__nv_bfloat16* saved_a,__nv_bfloat16* saved_b
#endif
                      ) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_lnsg_kernel), SMEM_BYTES, "ln_swiglu_gemm"), true);
  (void)ready;
  wide_lnsg_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mx, mw1p, mh, gamma, beta, xn, rstd, c1, M, H, eps, save
#if MW_SAVE_AB
    ,saved_a,saved_b
#endif
    );
}

#if MW_SPLIT_FWD
// B200 D512 boundary: LN materializes xn; expand writes h/a/b with TMA stores.
// Namespace isolates this second instance of the shared WGMMA pipeline.
namespace mw_split {
#undef WAIT0
#define WAIT0 1
#define TBK 32
#define NSTAGE 4
#define STG_HALF 1
#define REG_C 200
#define MW_EXPAND_ONLY 1
#define gate_gemm wide_expand_saved_kernel
#define tma_store_2d split_tma_store_2d
#include "kernels/gate_gemm2.cu"
#undef tma_store_2d
}
void wide_expand_saved_launch(const CUtensorMap& mxn,const CUtensorMap& mw1,
                             const CUtensorMap& mh,const CUtensorMap& ma,const CUtensorMap& mb,
                             int M,int D,int H,int ctas,cudaStream_t stream){
  static const bool ready=(wide_smem_optin(reinterpret_cast<const void*>(mw_split::wide_expand_saved_kernel),
                                          mw_split::SMEM_BYTES,"expand saved"),true);
  (void)ready;
  mw_split::wide_expand_saved_kernel<<<ctas,384,mw_split::SMEM_BYTES,stream>>>(mxn,mxn,mw1,mw1,mh,mh,M,D,H,ma,mb);
}
#endif

#if MW_FUSED_D384
// B200's entire D384 forward, with a Hopper register accumulator instead of TMEM.
// One consumer warpgroup owns 64 rows. 192 output registers plus 32 gate
// accumulators fit its 248-register allocation; two 72-KiB weight stages and
// a 48-KiB in-place LN/residual tile fit shared memory. No global h in inference.
namespace mw_f384 {
constexpr int WD=384, HH=1536, RC=64, CH=32, NC=48, NS=2;
constexpr int AW=49152, SLOT=73728, XB=NS*SLOT, BAR=XB+49152, BYTES=BAR+256;
static_assert(BYTES<=231424,"Hopper shared budget");
TMN_DEVI uint64_t desc(uint32_t x,int lbo=16){return smem_desc(x,lbo,1024,1);}
TMN_DEVI float sum32(float x){for(int s=16;s;s>>=1)x+=__shfl_xor_sync(0xffffffff,x,s);return x;}
#if MW_FUSED_D384_MC
TMN_DEVI void arrive_peer(uint64_t* p,int peer){
 uint32_t addr;asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(addr):"r"(smem_u32(p)),"r"(peer));
 asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(addr):"memory");
}
TMN_DEVI void mc_load(void* p,const CUtensorMap* map,uint64_t* bar,int x,int y){
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
  ::"r"(smem_u32(p)),"l"(map),"r"(smem_u32(bar)),"r"(x),"r"(y),"h"((uint16_t)3):"memory");
}
#endif
TMN_DEVI void f384_mma64ss(float* d, uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a), "l"(b));
}
TMN_DEVI void f384_mma256rs(float* d, const uint32_t* a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, 1, 1, 1, 1;"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "l"(b));
}
TMN_DEVI void f384_mma128rs(float* d, const uint32_t* a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 1;"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "l"(b));
}

template<bool SAVE>
__global__ __launch_bounds__(256,1)
#if MW_FUSED_D384_MC
__cluster_dims__(2,1,1)
#endif
void wide_fused_d384_kernel(const __grid_constant__ CUtensorMap mx,const __grid_constant__ CUtensorMap mwa,
 const __grid_constant__ CUtensorMap mwb,const __grid_constant__ CUtensorMap mws,const __grid_constant__ CUtensorMap mo,
 const float* gamma,const float* beta,__nv_bfloat16* xn,float* rs,float* c1,
 __nv_bfloat16* sh,__nv_bfloat16* sa,__nv_bfloat16* sb,int M,float eps){
 constexpr bool save=SAVE;
 extern __shared__ __align__(1024) uint8_t sm[];
 int tid=threadIdx.x,wg=tid>>7,wt=tid&127,warp=wt>>5,lane=tid&31,cta=blockIdx.x;
 int count=(M/RC>cta)?(M/RC-cta+gridDim.x-1)/gridDim.x:0;
 uint32_t su=smem_u32(sm);
 auto bars=reinterpret_cast<uint64_t*>(sm+BAR);
 auto full=bars;auto free=bars+NS;auto xl=bars+2*NS;auto xf=xl+1;auto xr=xl+2;auto done=xl+3;
#if MW_FUSED_D384_MC
 auto ready=done+1;int rank=cta&1,peer=rank^1;
#endif
 if(!tid){
  for(int j=0;j<NS;++j){mbar_init(full+j,1);
#if MW_FUSED_D384_MC
   mbar_init(free+j,2);mbar_init(ready+j,2);
#else
   mbar_init(free+j,1);
#endif
  }
  mbar_init(xl,1);mbar_init(xf,1);mbar_init(xr,1);mbar_init(done,1);fence_barrier_init();
 }
 __syncthreads();
#if MW_FUSED_D384_MC
 asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");
#endif
 if(wg==0){
  setmaxnreg_dec<64>();
  if(tid==0 && count){
  auto loadx=[&](int i,uint64_t* bar){
   mbar_arrive_expect_tx(bar,49152);
   for(int q=0;q<6;++q)tma_load_2d(sm+XB+q*8192,&mx,bar,q*64,(cta+i*gridDim.x)*RC);
  };
  auto loadw=[&](int seq){
   int s=seq%NS,j=seq%NC;if(seq>=NS)mbar_wait(free+s,(seq/NS-1)&1);
   auto p=sm+s*SLOT;mbar_arrive_expect_tx(full+s,SLOT);
#if MW_FUSED_D384_MC
   mbar_arrive(ready+s);arrive_peer(ready+s,peer);mbar_wait(ready+s,(seq/NS)&1);
   if(rank==0)for(int q=0;q<6;++q){
    mc_load(p+q*8192,&mwa,full+s,q*64,j*CH);
    mc_load(p+q*8192+4096,&mwb,full+s,q*64,j*CH);
    mc_load(p+AW+q*4096,&mws,full+s,q*64,j*CH);
   }
#else
   for(int q=0;q<6;++q){
    tma_load_2d(p+q*8192,&mwa,full+s,q*64,j*CH);
    tma_load_2d(p+q*8192+4096,&mwb,full+s,q*64,j*CH);
    tma_load_2d(p+AW+q*4096,&mws,full+s,q*64,j*CH);
   }
#endif
  };
  loadx(0,xl);for(int j=0;j<NS;++j)loadw(j);
  for(int i=0;i<count;++i){
   for(int j=NS;j<NC;++j)loadw(i*NC+j);
   mbar_wait(xf,i&1);loadx(i,xr);
   if(i+1<count)for(int j=0;j<NS;++j)loadw((i+1)*NC+j);
   mbar_wait(done,i&1);if(i+1<count)loadx(i+1,xl);
  }
  }
  __syncthreads();
#if MW_FUSED_D384_MC
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");
#endif
  return;
 }
 setmaxnreg_inc<248>();
 int seq=0;
 for(int i=0;i<count;++i){
  int row0=(cta+i*gridDim.x)*RC;
  mbar_wait(xl,i&1);
  {
   for(int r=warp;r<RC;r+=4){
    float x[12],sum=0.f,var=0.f;
    #pragma unroll
    for(int v=0;v<3;++v){
     int c=v*128+lane*4;
     uint32_t off=XB+(c>>6)*8192+swz128(r,(c&63)*2);
     uint2 u=*reinterpret_cast<uint2*>(sm+off);
     x[4*v]=bf16lo(u.x);x[4*v+1]=bf16hi(u.x);
     x[4*v+2]=bf16lo(u.y);x[4*v+3]=bf16hi(u.y);
     #pragma unroll
     for(int e=0;e<4;++e)sum+=x[4*v+e];
    }
    float mean=sum32(sum)/WD;
    #pragma unroll
    for(int e=0;e<12;++e){float z=x[e]-mean;var+=z*z;}
    float inv=rsqrtf(sum32(var)/WD+eps);
    #pragma unroll
    for(int v=0;v<3;++v){
     int c=v*128+lane*4;
     uint2 u;
     u.x=pack_bf16((x[4*v]-mean)*inv*gamma[c]+beta[c],(x[4*v+1]-mean)*inv*gamma[c+1]+beta[c+1]);
     u.y=pack_bf16((x[4*v+2]-mean)*inv*gamma[c+2]+beta[c+2],(x[4*v+3]-mean)*inv*gamma[c+3]+beta[c+3]);
     uint32_t off=XB+(c>>6)*8192+swz128(r,(c&63)*2);
     *reinterpret_cast<uint2*>(sm+off)=u;
     if(save)*reinterpret_cast<uint2*>(xn+(size_t)(row0+r)*WD+c)=u;
    }
    if(save&&!lane){rs[row0+r]=inv;c1[row0+r]=mean*inv;}
   }
  }
  fence_proxy_async();named_bar_sync(1,128);
  float acc[192];
  #pragma unroll
  for(int e=0;e<192;++e)acc[e]=0.f;
  for(int j=0;j<NC;++j,++seq){
   int s=seq%NS;uint32_t slot=su+s*SLOT;
   mbar_wait(full+s,(seq/NS)&1);
   float ab[32];
   #pragma unroll
   for(int e=0;e<32;++e)ab[e]=0.f;
   fence_regs(ab);fence_regs(acc);wgmma_fence();
   #pragma unroll
   for(int k=0;k<24;++k)f384_mma64ss(ab,desc(su+XB+(k>>2)*8192+(k&3)*32),desc(slot+(k>>2)*8192+(k&3)*32));
   wgmma_commit();wgmma_wait<1>();fence_regs(acc);
   if(j&&wt==0){mbar_arrive(free+(seq-1)%NS);
#if MW_FUSED_D384_MC
    arrive_peer(free+(seq-1)%NS,peer);
#endif
   }
   wgmma_wait<0>();fence_regs(ab);
   if(j==NC-1&&wt==0)mbar_arrive(xf);
   uint32_t h[2][4];
   #pragma unroll
   for(int g=0;g<4;++g){
    #pragma unroll
    for(int r=0;r<2;++r){
     int a=4*g+2*r,b=a+16;
     float a0=ab[a],a1=ab[a+1],b0=ab[b],b1=ab[b+1];
     uint32_t hp=pack_bf16(a0*sigmoid_(a0)*b0,a1*sigmoid_(a1)*b1);
     h[g>>1][2*(g&1)+r]=hp;
     if(save){
      size_t off=(size_t)(row0+16*warp+(lane>>2)+8*r)*HH+j*CH+8*g+2*(lane&3);
      *reinterpret_cast<uint32_t*>(sh+off)=hp;
      *reinterpret_cast<uint32_t*>(sa+off)=pack_bf16(a0,a1);
      *reinterpret_cast<uint32_t*>(sb+off)=pack_bf16(b0,b1);
     }
    }
   }
   fence_regs(acc);wgmma_fence();
   #pragma unroll
   for(int k=0;k<2;++k){
    f384_mma256rs(acc,h[k],desc(slot+AW+k*2048,4096));
    f384_mma128rs(acc+128,h[k],desc(slot+AW+16384+k*2048,4096));
   }
   wgmma_commit();
  }
  wgmma_wait<0>();fence_regs(acc);
  if(wt==0){mbar_arrive(free+(seq-1)%NS);
#if MW_FUSED_D384_MC
   arrive_peer(free+(seq-1)%NS,peer);
#endif
  }
  mbar_wait(xr,i&1);
  int mi=lane>>3,r=16*warp+8*(mi&1)+(lane&7);
  #pragma unroll
  for(int g=0;g<24;++g){
   int col=8*(2*g+(mi>>1));uint32_t ad=su+XB+(col>>6)*8192+swz128(r,(col&63)*2);
   uint32_t x[4];ldsm_x4(x,ad);
   stsm_x4(ad,pack_bf16(bf16lo(x[0])+acc[8*g],bf16hi(x[0])+acc[8*g+1]),
    pack_bf16(bf16lo(x[1])+acc[8*g+2],bf16hi(x[1])+acc[8*g+3]),
    pack_bf16(bf16lo(x[2])+acc[8*g+4],bf16hi(x[2])+acc[8*g+5]),
    pack_bf16(bf16lo(x[3])+acc[8*g+6],bf16hi(x[3])+acc[8*g+7]));
  }
  fence_proxy_async();named_bar_sync(1,128);
  if(!wt){
   for(int q=0;q<6;++q)tma_store_2d(&mo,sm+XB+q*8192,q*64,row0);
   tma_store_commit();tma_store_wait_read<0>();mbar_arrive(done);
  }
 }
 if(!wt)tma_store_wait_all();
 __syncthreads();
#if MW_FUSED_D384_MC
 // Neither CTA can retire while its peer still accesses its shared barriers.
 asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");
#endif
}
}
void wide_fused_d384_launch(const CUtensorMap& mx,const CUtensorMap& mwa,const CUtensorMap& mwb,const CUtensorMap& mws,
 const CUtensorMap& mo,const float* gamma,const float* beta,__nv_bfloat16* xn,float* rs,float* c1,
 __nv_bfloat16* h,__nv_bfloat16* a,__nv_bfloat16* b,int M,float eps,int save,cudaStream_t stream){
 static const bool ready=(wide_smem_optin(reinterpret_cast<const void*>(mw_f384::wide_fused_d384_kernel<true>),mw_f384::BYTES,"D384 fused train"),
  wide_smem_optin(reinterpret_cast<const void*>(mw_f384::wide_fused_d384_kernel<false>),mw_f384::BYTES,"D384 fused infer"),true);
 (void)ready;
 if(save)mw_f384::wide_fused_d384_kernel<true><<<WIDE_SMS,256,mw_f384::BYTES,stream>>>(mx,mwa,mwb,mws,mo,gamma,beta,xn,rs,c1,h,a,b,M,eps);
 else mw_f384::wide_fused_d384_kernel<false><<<WIDE_SMS,256,mw_f384::BYTES,stream>>>(mx,mwa,mwb,mws,mo,gamma,beta,xn,rs,c1,h,a,b,M,eps);
}
#endif
