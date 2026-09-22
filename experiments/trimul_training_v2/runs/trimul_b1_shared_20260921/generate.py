from pathlib import Path
R=Path(__file__).resolve().parent
old=R.parent/'trimul_split_bwd_20260921'
s=(old/'b1_fused.cu').read_text()
# Retain trusted fragment arithmetic/mask and dgrad; replace CTA role dispatch.
s=s[:s.index('TMN_DEVI void weight_role')]
s=s.replace('static_assert(DXCOUNT>0,"need DX roles");','')
s += r'''
// Each CTA owns rows and all three weight-gradient accumulators.
// The opposite raw slot becomes normalized output + dProj staging until
// weight consumers finish. It is then reused by the next asynchronous TMA load.
TMN_DEVI void prepare_shared(const Params& p,uint8_t* sm,uint64_t* bars,int slot,int row,int phase,const FragmentMask& mask,int mi){
 const int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 uint8_t* x=sm+16384+slot*49152;
 uint8_t* norm=sm+16384+(1-slot)*49152;
 float* ps=reinterpret_cast<float*>(sm+212992);
 if(wi==1){
  LnStats st=normalize_tile<16,true,B1_LN_SERIAL>(x+16384,norm,ps+256,ps+512);
  if(lane%4==0){int ra=w*16+lane/4,rb=ra+8;float* mu=reinterpret_cast<float*>(sm+225280);mu[ra]=st.mA;mu[rb]=st.mB;mu[64+ra]=st.rA;mu[64+rb]=st.rB;}
 }
 // Input x_n is already saved by forward.
 float acc[32];uint32_t gate[16];
 recompute_gemm<128,16384>(acc,x,sm+114688+wi*8192);
 #pragma unroll
 for(int j=0;j<16;++j)gate[j]=pack_bf16(math::sigmoid(math::round_bf16(acc[j*2])),math::sigmoid(math::round_bf16(acc[j*2+1])));
 fence_proxy_async();allsync();
 recompute_gemm<256,16384>(acc,norm,sm+147456+wi*8192);
 mbar_wait(bars+2,phase);
 #pragma unroll
 for(int q=0;q<4;++q){uint32_t dy[4],dg[4],dp[4];
  uint32_t off=swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16);
  ldsm_x4(dy,smem_u32(sm+wi*8192)+off);
  #pragma unroll
  for(int j=0;j<4;++j){int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+r)%TRAIN_L;uint32_t ds;
   if(mask.cached){int bit=q*8+j*2;uint32_t bits=mask_bits(mask,mi);ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
   else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+wi*64+c);
   float a=bf16lo(dy[j])*bf16lo(ds),b=bf16hi(dy[j])*bf16hi(ds),ga=bf16lo(gate[q*4+j]),gb=bf16hi(gate[q*4+j]);
   dp[j]=pack_bf16(a*ga,b*gb);
   dg[j]=pack_bf16(((a*math::round_bf16(acc[q*8+j*2]))*ga)*(1.f-ga),((b*math::round_bf16(acc[q*8+j*2+1]))*gb)*(1.f-gb));
  }
  stsm_x4(smem_u32(sm+wi*8192)+off,dg[0],dg[1],dg[2],dg[3]);
  stsm_x4(smem_u32(norm+32768+wi*8192)+off,dp[0],dp[1],dp[2],dp[3]);
 }
 fence_proxy_async();allsync();
 if(threadIdx.x==0){dg_store(&p.dgmap,sm,0,row);dg_store(&p.dgmap,sm+8192,64,row);tma_store_commit();}
}
TMN_DEVI void weight_gemm(float (&acc)[64],uint8_t* sa,uint8_t* sb,bool add){
 fence_regs(acc);wgmma_fence();
 static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),add||k>0);});
 wgmma_commit();wgmma_wait<0>();fence_regs(acc);
}
TMN_DEVI void store_weight(const Params& p,float (&acc)[64],int kind){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 float* out=p.partw+blockIdx.x*49152+kind*16384;
 #pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(kind==0?wi*64:0),c=q*8+2*(lane%4)+(kind==0?0:wi*128),stride=kind==0?128:256;
  stg64f(out+rr*stride+c,acc[4*q],acc[4*q+1]);stg64f(out+(rr+8)*stride+c,acc[4*q+2],acc[4*q+3]);}
}
TMN_DEVI void shared_role(const Params& p,uint8_t* sm,uint64_t* bars){
 int wi=threadIdx.x/128,round=0,mi=0;
 FragmentMask mask=fragment_mask<UCOUNT>(p,blockIdx.x);
 float wg[64]={},wp0[64]={},wp1[64]={};
 if(blockIdx.x<p.tiles)load_raw(p,sm,bars,0,blockIdx.x*64);
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);load_dy(p,sm,bars+2,tile*64);
  prepare_shared(p,sm,bars,slot,tile*64,round&1,mask,mi);
  uint8_t* x=sm+16384+slot*49152;
  uint8_t* norm=sm+16384+(1-slot)*49152;
  weight_gemm(wg,x+wi*8192,sm,round>0);
  weight_gemm(wp0,norm+32768,norm+wi*16384,round>0);
  weight_gemm(wp1,norm+40960,norm+wi*16384,round>0);
  if(threadIdx.x==0)tma_store_wait_all();allsync();
  for(int j=threadIdx.x;j<4096;j+=256)reinterpret_cast<uint32_t*>(sm)[j]=reinterpret_cast<uint32_t*>(norm+32768)[j];
  fence_proxy_async();allsync();
  // All consumers have released the opposite slot; overlap its next TMA load
  // with dTri/LN backward on the current raw tri tile.
  if(tile+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,(tile+UCOUNT)*64);
  pipeline_dgrad(p,sm,tile*64,slot);allsync();if(++mi==mask.period)mi=0;
 }
 store_weight(p,wg,0);store_weight(p,wp0,1);store_weight(p,wp1,2);
 for(int j=threadIdx.x;j<512;j+=256)p.partln[blockIdx.x*512+j]=reinterpret_cast<float*>(sm+225792)[j];
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<49152){float v=0;for(int b=0;b<UCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partw)[b*49152+i];
  (i<16384?p.dwg:p.dwp-16384)[i]=__float2bfloat16_rn(v);
 }else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<UCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
extern "C" __global__ __launch_bounds__(256,1) void b1_fused(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bars[4];
 if(threadIdx.x==0){for(int b=0;b<4;++b)mbar_init(bars+b,1);fence_barrier_init();mbar_arrive_expect_tx(bars+3,98304);
  for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+114688+k*16384+c*8192,&p.wg,bars+3,c*64,k*64);
  for(int c=0;c<2;++c)for(int k=0;k<4;++k)tma_load_2d(sm+147456+k*16384+c*8192,&p.wp,bars+3,c*64,k*64);
 }
 float* ps=reinterpret_cast<float*>(sm+212992);for(int c=threadIdx.x;c<768;c+=256)ps[c]=c<128?p.gi[c]:c<256?p.bi[c-128]:c<512?p.gamma[c-256]:p.bo[c-512];
 for(int j=threadIdx.x;j<512;j+=256)reinterpret_cast<float*>(sm+225792)[j]=0.f;allsync();mbar_wait(bars+3,0);
 shared_role(p,sm,bars);
#if PART_ONLY==2
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void b1_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
'''
(R/'b1_fused.cu').write_text(s)
for n in ('common_recompute.cuh','ln_recompute.cuh','b1_pipeline_math.inc'):(R/n).write_bytes((old/n).read_bytes())
