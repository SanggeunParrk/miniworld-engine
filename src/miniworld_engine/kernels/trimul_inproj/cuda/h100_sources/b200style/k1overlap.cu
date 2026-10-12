// B200-style independent compute/epilogue work, implemented with Hopper WGMMA.
// An FP32 shared mailbox preserves preactivation precision. Its lifetime ends
// only after the epilogue's TMA stores have finished reading the reused buffer.
#include "tmn_kernels.cuh"
#include "wgmma_n.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=HIDDEN,BN=BLOCKN,NS=STAGES,MT=128,KB=D/64,NB=4*H/BN;
constexpr int NA=BN/2,NP=BN/2,AB=MT*128,A_BYTES=KB*AB,SLOT=BN*128;
constexpr int W_BYTES=NS*SLOT,C_BYTES=NA*128*4,SMEM=A_BYTES+W_BYTES+2*C_BYTES+512;
static_assert((BN==64 || BN==128) && SMEM+1024<=232448 && NS>=2,"overlap layout");
struct Params {
  CUtensorMap x,w,ab,da,dc,xn;
  const bf* mask; const float *gamma,*beta,*xs; float* stats;
  int M,L,tiles,emit_xn;
};
TMN_DEVI uint32_t word(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0,[%1];":"=r"(v):"r"(a)); return v; }
TMN_DEVI void store2(const CUtensorMap* map,const void* src,int c0,int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      ::"l"(map),"r"(smem_u32(src)),"r"(c0),"r"(c1):"memory");
}
TMN_DEVI void dot64(float (&d)[32],uint64_t a,uint64_t b,int acc) {
  asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
    :"l"(a),"l"(b),"r"(acc));
}
template<int N> TMN_DEVI void dot(float (&v)[N/2],uint64_t a,uint64_t b,int acc) {
  if constexpr(N==64) dot64(v,a,b,acc); else mw_mma128<0>(v,a,b,acc);
}
extern "C" __global__ __launch_bounds__(640,1)
void KERNEL_NAME(__grid_constant__ const Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* sa=sm; uint8_t* sw=sa+A_BYTES; uint8_t* sc=sw+W_BYTES;
  uint64_t* af=reinterpret_cast<uint64_t*>(sc+2*C_BYTES);
  uint64_t* ae=af+1; uint64_t* wf=ae+1; uint64_t* we=wf+NS;
  uint64_t* cf=we+NS; uint64_t* ce=cf+2; uint64_t* df=ce+2;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,wg=tid/128;
  if(tid==0) {
    mbar_init(af,1); mbar_init(ae,8);
    for(int s=0;s<NS;++s) { mbar_init(wf+s,1); mbar_init(we+s,8); }
    for(int s=0;s<2;++s) { mbar_init(cf+s,4); mbar_init(ce+s,4); mbar_init(df+s,1); }
    fence_barrier_init(); tma_prefetch_desc(&p.x); tma_prefetch_desc(&p.w);
    if(BACKWARD) { tma_prefetch_desc(&p.da); tma_prefetch_desc(&p.dc); if(p.emit_xn)tma_prefetch_desc(&p.xn); }
    else tma_prefetch_desc(&p.ab);
  }
  __syncthreads();
  const int ni=(p.tiles-blockIdx.x+gridDim.x-1)/gridDim.x;
  if(wg==0) {
    setmaxnreg_dec<24>();
    if(warp==0 && lane==0) {
      for(int i=0;i<ni;++i) {
        if(i)mbar_wait(ae,(i-1)&1);
        mbar_arrive_expect_tx(af,A_BYTES);
        for(int k=0;k<KB;++k)tma_load_2d(sa+k*AB,&p.x,af,k*64,(blockIdx.x+i*gridDim.x)*MT);
      }
    } else if(warp==1 && lane==0) {
      for(int it=0;it<ni*NB*KB;++it) {
        const int s=it%NS,u=it/NS;
        if(u)mbar_wait(we+s,(u-1)&1);
        mbar_arrive_expect_tx(wf+s,SLOT);
        tma_load_2d(sw+s*SLOT,&p.w,wf+s,(it%KB)*64,((it/KB)%NB)*BN);
      }
    }
    return;
  }
  if(wg<3) {
    setmaxnreg_dec<88>();
    const int cw=wg-1,ct=tid%128;
    const uint32_t dest=smem_u32(sc+cw*C_BYTES);
    auto release=[&](int it) { fence_proxy_async(); __syncwarp(); if(lane==0)mbar_arrive(we+it%NS); };
    int it=0,item=0;
    for(int ti=0;ti<ni;++ti) {
      const int tok0=(blockIdx.x+ti*gridDim.x)*MT;
      mbar_wait(af,ti&1);
      for(int r=(warp-4)*8+lane/4;r<MT;r+=64) {
        float mu=0.f,rs=1.f;
        auto addr=[&](int k,int c) { return smem_u32(sa)+k*AB+swz128(r,c*2); };
        if(BACKWARD) { mu=p.xs[2*(tok0+r)];rs=p.xs[2*(tok0+r)+1]; }
        else {
          for(int k=0;k<KB;++k)for(int c=2*(lane&3);c<64;c+=8) { uint32_t v=word(addr(k,c));mu+=bf16lo(v)+bf16hi(v); }
          mu=quad_sum(mu)/D; float vv=0.f;
          for(int k=0;k<KB;++k)for(int c=2*(lane&3);c<64;c+=8) { uint32_t v=word(addr(k,c));float a=bf16lo(v)-mu,b=bf16hi(v)-mu;vv=fmaf(a,a,vv);vv=fmaf(b,b,vv); }
          rs=rsqrtf(quad_sum(vv)/D+1e-5f);
          if(p.stats && !(lane&3)) {p.stats[2*(tok0+r)]=mu;p.stats[2*(tok0+r)+1]=rs;}
        }
        for(int k=0;k<KB;++k)for(int c=2*(lane&3);c<64;c+=8) { const uint32_t a=addr(k,c),v=word(a);const int ch=k*64+c;sts32(a,pack_bf16(math::ln_affine(bf16lo(v),mu,rs,p.gamma[ch],p.beta[ch]),math::ln_affine(bf16hi(v),mu,rs,p.gamma[ch+1],p.beta[ch+1]))); }
      }
      fence_proxy_async();named_bar_sync(1,256);
      if(BACKWARD && p.emit_xn && tid==128) { for(int k=0;k<KB;++k)store2(&p.xn,sa+k*AB,k*64,tok0);tma_store_commit(); }
      for(int block=0;block<NB;++block,++item) {
        float ac[NA];
#pragma unroll
        for(int j=0;j<NA;++j)ac[j]=0.f;
        for(int k=0;k<KB;++k,++it) {
          const int s=it%NS;mbar_wait(wf+s,(it/NS)&1);
          const uint32_t a=smem_u32(sa)+k*AB+cw*8192,b=smem_u32(sw)+s*SLOT;
          fence_regs(ac);wgmma_fence();
#pragma unroll
          for(int q=0;q<4;++q)dot<BN>(ac,smem_desc(a+q*32,16,1024,1),smem_desc(b+q*32,16,1024,1),k>0 || q>0);
          wgmma_commit();if(k>0){wgmma_wait<1>();fence_regs(ac);release(it-1);}
        }
        wgmma_wait<0>();fence_regs(ac);release(it-1);
        if(item)mbar_wait(ce+cw,(item-1)&1);
#pragma unroll
        for(int j=0;j<NA;++j)sts32(dest+4*(j*128+ct),__float_as_uint(ac[j]));
        __syncwarp();if(lane==0)mbar_arrive(cf+cw);
      }
      if(BACKWARD && p.emit_xn && tid==128)tma_store_wait_read<0>();
      fence_proxy_async();__syncwarp();if(lane==0)mbar_arrive(ae);
    }
    if(tid==128 && BACKWARD && p.emit_xn)tma_store_wait_all();
    return;
  }
  setmaxnreg_inc<152>();
  const int cw=wg-3,ct=tid%128,wiw=warp%4,idx=lane/8,rr=lane%8;
  const int rowa=16*wiw+lane/4,rowb=rowa+8;
  const bool elect=wiw==0 && lane==0;
  uint8_t* stage=sc+cw*C_BYTES;const uint32_t su=smem_u32(stage);
  const uint32_t plane_off=swz128(8*(idx>>1)+rr,(16*wiw+8*(idx&1))*2);
  int item=0;
  for(int ti=0;ti<ni;++ti) {
    const int tok0=(blockIdx.x+ti*gridDim.x)*MT+cw*64;
    const float ma=p.mask?__bfloat162float(p.mask[tok0+rowa]):1.f,mb=p.mask?__bfloat162float(p.mask[tok0+rowb]):1.f;
    for(int block=0;block<NB;++block,++item) {
      mbar_wait(cf+cw,item&1);float ac[NA];
#pragma unroll
      for(int j=0;j<NA;++j)ac[j]=__uint_as_float(word(su+4*(j*128+ct)));
      if(BACKWARD) {
        fence_proxy_async();named_bar_sync(2+cw,128);
        if(elect){mbar_arrive_expect_tx(df+cw,NP*128);tma_load_2d(stage,&p.da,df+cw,tok0,block*NP);}
        mbar_wait(df+cw,item&1);uint32_t da[NP/16][4];
#pragma unroll
        for(int q=0;q<NP/16;++q)ldsm_x4_t(da[q],su+(q/2)*4096+plane_off+(q%2)*2048);
        uint32_t pk[NP/8][4];
#pragma unroll
        for(int q=0;q<NP/8;++q) {
          const int g=q+4*(q/4),pp=g+4;
#pragma unroll
          for(int e=0;e<2;++e) {
            const uint32_t dd=da[q/2][2*(q%2)+e];const float mask=e?mb:ma;
            const float d0=bf16lo(dd)*mask,d1=bf16hi(dd)*mask;
            const float s0=math::sigmoid(ac[4*g+2*e]),s1=math::sigmoid(ac[4*g+2*e+1]);
            pk[q][2*e]=pack_bf16(d0*s0*ac[4*pp+2*e]*(1.f-s0),d1*s1*ac[4*pp+2*e+1]*(1.f-s1));
            pk[q][2*e+1]=pack_bf16(d0*s0,d1*s1);
          }
        }
        named_bar_sync(2+cw,128);
        const int tr=16*wiw+8*(idx&1)+rr,cb=8*(idx>>1);
#pragma unroll
        for(int half=0;half<BN/64;++half) {
          const uint32_t base=su+half*8192;const int q=half*4;
          stsm_x4(base+swz128(tr,2*cb),pk[q][0],pk[q][2],pk[q+1][0],pk[q+1][2]);
          stsm_x4(base+swz128(tr,2*(cb+16)),pk[q+2][0],pk[q+2][2],pk[q+3][0],pk[q+3][2]);
          stsm_x4(base+swz128(tr,2*(cb+32)),pk[q][1],pk[q][3],pk[q+1][1],pk[q+1][3]);
          stsm_x4(base+swz128(tr,2*(cb+48)),pk[q+2][1],pk[q+2][3],pk[q+3][1],pk[q+3][3]);
        }
        fence_proxy_async();named_bar_sync(2+cw,128);
        if(elect){for(int half=0;half<BN/64;++half)store2(&p.dc,stage+half*8192,block*BN+half*64,tok0);tma_store_commit();tma_store_wait_read<0>();}
      } else {
        uint32_t pk[NP/8][2];
#pragma unroll
        for(int q=0;q<NP/8;++q) {const int g=q+4*(q/4),pp=g+4;pk[q][0]=pack_bf16(math::gate(ac[4*g],ac[4*pp],ma),math::gate(ac[4*g+1],ac[4*pp+1],ma));pk[q][1]=pack_bf16(math::gate(ac[4*g+2],ac[4*pp+2],mb),math::gate(ac[4*g+3],ac[4*pp+3],mb));}
        named_bar_sync(2+cw,128);
#pragma unroll
        for(int half=0;half<NP/32;++half){const int q=half*4;const uint32_t base=su+half*4096;stsm_x4_t(base+plane_off,pk[q][0],pk[q][1],pk[q+1][0],pk[q+1][1]);stsm_x4_t(base+plane_off+2048,pk[q+2][0],pk[q+2][1],pk[q+3][0],pk[q+3][1]);}
        fence_proxy_async();named_bar_sync(2+cw,128);
        if(elect){for(int half=0;half<NP/32;++half)store2(&p.ab,stage+half*4096,tok0,block*NP+half*32);tma_store_commit();tma_store_wait_read<0>();}
      }
      fence_proxy_async();named_bar_sync(2+cw,128);__syncwarp();if(lane==0)mbar_arrive(ce+cw);
    }
  }
  if(elect)tma_store_wait_all();
}
