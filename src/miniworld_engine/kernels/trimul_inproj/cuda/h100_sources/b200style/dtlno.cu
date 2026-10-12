// Experimental dpr @ Wp + output-LN backward. Rounded dor never reaches HBM.
#include "tmn_kernels.cuh"
#include "wgmma_n.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf=__nv_bfloat16;
#ifndef TOKENS
#define TOKENS 128
#endif
constexpr int D=WIDTH,H=HIDDEN,BN=BLOCKN,NS=STAGES,MT=TOKENS,NWG=MT/64,NT=128*(NWG+1);
constexpr int KB=D/64,NA=BN/2,AB=MT*128,WB=BN*128,ST=AB+WB,OB=BN*128;
constexpr int RED=NWG*4*2*BN*4, SUM=NWG*2*BN*4;
static_assert(MT==128 || (MT==64 && BN==128), "fused output tile");
struct Params {
  CUtensorMap dp, wt, tri, dt;
  const float *mu,*rs,*s1,*s2,*go;
  float* part;
  int M,tiles;
};
TMN_DEVI uint32_t read_word(uint32_t addr) {
  uint32_t v; asm volatile("ld.shared.b32 %0,[%1];":"=r"(v):"r"(addr)); return v;
}
TMN_DEVI float recip(float v) { float x; asm("rcp.approx.ftz.f32 %0,%1;":"=f"(x):"f"(v)); return x; }
template<int N> TMN_DEVI void dot(float (&ac)[N/2],uint64_t a,uint64_t b,int acc) {
  if constexpr(N==128) mw_mma128<0>(ac,a,b,acc);
  else if constexpr(N==192) mw_mma192<0>(ac,a,b,acc);
  else mw_mma256<0>(ac,a,b,acc);
}
TMN_DEVI void store2(const CUtensorMap* map,const void* src,int x,int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               ::"l"(map),"r"(smem_u32(src)),"r"(x),"r"(y):"memory");
}
extern "C" __global__ __launch_bounds__(NT,MT==64?2:1)
void mw_b200style_dtlno(__grid_constant__ const Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* ring=sm; uint8_t* out=ring+NS*ST;
  float* red=reinterpret_cast<float*>(out+NWG*OB);
  float* sums=reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(red)+RED);
  uint64_t* full=reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sums)+SUM);
  uint64_t* empty=full+NS; uint64_t* tf=empty+NS;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,wg=tid/128;
  for(int j=tid;j<NWG*2*BN;j+=NT) sums[j]=0.f;
  if(tid==0) {
    for(int s=0;s<NS;++s) { mbar_init(full+s,1); mbar_init(empty+s,4*NWG); }
    for(int s=0;s<NWG;++s) mbar_init(tf+s,1);
    fence_barrier_init();
    tma_prefetch_desc(&p.dp); tma_prefetch_desc(&p.wt);
    tma_prefetch_desc(&p.tri); tma_prefetch_desc(&p.dt);
  }
  __syncthreads();
  const int ch0=blockIdx.y*BN;
  if(wg==0) {
    if constexpr(MT==64) setmaxnreg_dec<32>(); else setmaxnreg_dec<40>();
    if(warp==0 && lane==0) {
      int it=0;
      for(int tile=blockIdx.x;tile<p.tiles;tile+=gridDim.x)
      for(int k=0;k<KB;++k,++it) {
        const int s=it%NS,phase=it/NS;
        if(phase) mbar_wait(empty+s,(phase-1)&1);
        mbar_arrive_expect_tx(full+s,ST);
        tma_load_2d(ring+s*ST,&p.dp,full+s,k*64,tile*MT);
        tma_load_2d(ring+s*ST+AB,&p.wt,full+s,k*64,ch0);
      }
    }
    return;
  }
  if constexpr(MT==64) setmaxnreg_inc<192>(); else setmaxnreg_inc<232>();
  const int cw=wg-1,wiw=warp%4,rowa=16*wiw+lane/4,rowb=rowa+8;
  const int q2=2*(lane%4),idx=lane/8,rr=lane%8;
  const bool elect=wiw==0 && lane==0;
  uint8_t* stage=out+cw*OB;
  const uint32_t stage_u=smem_u32(stage);
  const uint32_t po=swz128(8*(idx>>1)+rr,(16*wiw+8*(idx&1))*2);
  auto release=[&](int it) { fence_proxy_async();__syncwarp();if(lane==0)mbar_arrive(empty+it%NS); };
  int it=0,ph=0;
  for(int tile=blockIdx.x;tile<p.tiles;tile+=gridDim.x,++ph) {
    const int tok=tile*MT+cw*64;
    if(elect) {
      tma_store_wait_read<0>(); mbar_arrive_expect_tx(tf+cw,OB);
#pragma unroll
      for(int j=0;j<BN/32;++j) tma_load_2d(stage+j*4096,&p.tri,tf+cw,tok,ch0+j*32);
    }
    float ac[NA];
#pragma unroll
    for(int j=0;j<NA;++j) ac[j]=0.f;
    for(int k=0;k<KB;++k,++it) {
      const int s=it%NS;
      mbar_wait(full+s,(it/NS)&1);
      const uint32_t a=smem_u32(ring)+s*ST+cw*8192,b=smem_u32(ring)+s*ST+AB;
      fence_regs(ac); wgmma_fence();
#pragma unroll
      for(int q=0;q<4;++q) {
        dot<BN>(ac,smem_desc(a+q*32,16,1024,1),smem_desc(b+q*32,16,1024,1),k>0||q>0);
      }
      wgmma_commit();
      if(k>0) { wgmma_wait<1>();fence_regs(ac);release(it-1); }
    }
    wgmma_wait<0>();fence_regs(ac);release(it-1);
    mbar_wait(tf+cw,ph&1);
    const float ma=p.mu[tok+rowa],mb=p.mu[tok+rowb],ra=p.rs[tok+rowa],rb=p.rs[tok+rowb];
    const float a1=ra*p.s1[tok+rowa]/H,b1=rb*p.s1[tok+rowb]/H;
    const float a2=ra*ra*p.s2[tok+rowa]/H,b2=rb*rb*p.s2[tok+rowb]/H;
    const float ia=recip(ra),ib=recip(rb);
#pragma unroll
    for(int half=0;half<BN/32;++half) {
      uint32_t pk[4][2];
      uint32_t tv[2][4];
      ldsm_x4_t(tv[0],stage_u+half*4096+po);
      ldsm_x4_t(tv[1],stage_u+half*4096+po+2048);
#pragma unroll
      for(int q=0;q<4;++q) {
        const int g=half*4+q,c=8*g+q2;
        float oa[2],ob[2];
#pragma unroll
        for(int j=0;j<2;++j) {
          const uint32_t va=tv[q/2][2*(q%2)],vb=tv[q/2][2*(q%2)+1];
          const float ta=(j?bf16hi(va):bf16lo(va))-ma;
          const float tb=(j?bf16hi(vb):bf16lo(vb))-mb;
          const float da=math::round_bf16(ac[4*g+j]),db=math::round_bf16(ac[4*g+2+j]);
          const float gamma=p.go[ch0+c+j];
          oa[j]=gamma*da-a1-ta*a2; ob[j]=gamma*db-b1-tb*b2;
          float sg=fmaf(da,ta,db*tb),sb=da*ia+db*ib;
#pragma unroll
          for(int off=4;off<32;off*=2) { sg+=__shfl_xor_sync(0xffffffffu,sg,off);sb+=__shfl_xor_sync(0xffffffffu,sb,off); }
          if(lane<4) { red[((cw*4+wiw)*2)*BN+c+j]=sg; red[((cw*4+wiw)*2+1)*BN+c+j]=sb; }
        }
        pk[q][0]=pack_bf16(oa[0],oa[1]);pk[q][1]=pack_bf16(ob[0],ob[1]);
      }
      named_bar_sync(1+cw,128);
      const uint32_t base=stage_u+half*4096;
      stsm_x4_t(base+po,pk[0][0],pk[0][1],pk[1][0],pk[1][1]);
      stsm_x4_t(base+po+2048,pk[2][0],pk[2][1],pk[3][0],pk[3][1]);
      fence_proxy_async();named_bar_sync(1+cw,128);
      if(elect) store2(&p.dt,stage+half*4096,tok,ch0+half*32);
    }
    if(elect) tma_store_commit();
    named_bar_sync(1+cw,128);
    for(int c=tid%128;c<BN;c+=128) {
      float sg=0.f,sb=0.f;
#pragma unroll
      for(int w=0;w<4;++w) { sg+=red[((cw*4+w)*2)*BN+c];sb+=red[((cw*4+w)*2+1)*BN+c]; }
      sums[(cw*2)*BN+c]+=sg;sums[(cw*2+1)*BN+c]+=sb;
    }
    named_bar_sync(1+cw,128);
  }
  for(int c=tid%128;c<BN;c+=128) {
    const size_t row=(blockIdx.x*NWG+cw)*2*(size_t)H;
    p.part[row+ch0+c]=sums[(cw*2)*BN+c];
    p.part[row+H+ch0+c]=sums[(cw*2+1)*BN+c];
  }
  if(elect) tma_store_wait_all();
}
