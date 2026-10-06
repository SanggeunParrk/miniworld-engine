
#if defined(ATOM_HOPPER)
#include "sm100.cuh"
using namespace atom90;
extern "C" __global__ __launch_bounds__(160,1)
void atom_cond_bwd_sm90(const __nv_bfloat16* dm,const __nv_bfloat16* c,
 const __grid_constant__ CUtensorMap mw,const float* g1,const float* g2,
 __nv_bfloat16* dc,__nv_bfloat16* cn1,__nv_bfloat16* cn2,float* dgparts,int M,float eps) {
 extern __shared__ __align__(1024) uint8_t sm[];
 constexpr int NST=3,TEMP=98304,RED=TEMP+65536,BAR=RED+4096;
 uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
 int t=threadIdx.x,l=t&31,w=t>>5,nt=(M+63)/64;
 int tiles=blockIdx.x<nt?(nt-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x):0;
 if(t==0){for(int j=0;j<6;++j)init(bars+j);init_fence();}__syncthreads();
 if(t>=128){
   if(t==128)for(int it=0;it<tiles;++it)for(int p=0;p<6;++p){
     int g=it*6+p,st=g%NST;
     if(g>=NST)bwait(bars+NST+st,((g/NST)-1)&1);
     expect(bars+st,32768);
     for(int k=0;k<2;++k)load2(sm+st*32768+k*16384,&mw,bars+st,128*p+64*k,0);
   }
   return;
 }
 int r0=w*16+l/4,cq=2*(l&3);
 for(int it=0;it<tiles;++it){
   int tile=int(blockIdx.x)+it*int(gridDim.x),row=64*tile;float acc[64];
   for(int p=0;p<6;++p){
     int dblock=p<2?p:p==2?4:p==3?2:p==4?3:5;
     uint32_t fa[32];
     #pragma unroll
     for(int i=0;i<16;++i)
       #pragma unroll
       for(int rr=0;rr<2;++rr){
         int r=row+r0+8*rr,cx=8*i+cq;
         fa[4*(i>>1)+2*(i&1)+rr]=r<M?*reinterpret_cast<const uint32_t*>(dm+size_t(r)*768+dblock*128+cx):0;
       }
     int g=it*6+p,st=g%NST;bwait(bars+st,(g/NST)&1);
     const uint8_t* sw=sm+st*32768;fence();
     #pragma unroll
     for(int k=0;k<8;++k){
       uint64_t d=desc(sw+(k>>2)*16384+(k&3)*32);
       if(p!=1&&p!=4&&k==0)mma_rs_n128_first(fa[0],fa[1],fa[2],fa[3],d,acc);
       else mma_rs_n128(fa[4*k],fa[4*k+1],fa[4*k+2],fa[4*k+3],d,acc,1);
     }
     commit();wait();bsync();
     if(t==0)asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sa(bars+NST+st)):"memory");
     if(p==1||p==2||p==4||p==5){
       int b=p==1?0:p==2?1:p==4?2:3;
       #pragma unroll
       for(int i=0;i<16;++i)
         #pragma unroll
         for(int rr=0;rr<2;++rr)*reinterpret_cast<uint32_t*>(sm+TEMP+b*16384+off(r0+8*rr,8*i+cq))=pack(acc[4*i+2*rr],acc[4*i+2*rr+1]);
     }
   }
   float mu[2]={0,0},vv[2]={0,0};
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr){
       int r=row+r0+8*rr,cx=8*i+cq;
       float2 x=r<M?unpack(*reinterpret_cast<const uint32_t*>(c+size_t(r)*128+cx)):make_float2(0,0);
       acc[4*i+2*rr]=x.x;acc[4*i+2*rr+1]=x.y;mu[rr]+=x.x+x.y;
     }
   mu[0]=sum4(mu[0])/128.f;mu[1]=sum4(mu[1])/128.f;
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr){float x=acc[4*i+2*rr]-mu[rr],y=acc[4*i+2*rr+1]-mu[rr];vv[rr]+=x*x+y*y;}
   vv[0]=rsqrtf(sum4(vv[0])/128.f+eps);vv[1]=rsqrtf(sum4(vv[1])/128.f+eps);
   float ma[2][2]={{0,0},{0,0}},mb[2][2]={{0,0},{0,0}},gg1[32]={0},gg2[32]={0};
   #pragma unroll
   for(int i=0;i<16;++i){
     float2 a1=*reinterpret_cast<const float2*>(g1+8*i+cq),a2=*reinterpret_cast<const float2*>(g2+8*i+cq);
     #pragma unroll
     for(int rr=0;rr<2;++rr){
       int pos=off(r0+8*rr,8*i+cq);
       float2 d1=unpack(*reinterpret_cast<uint32_t*>(sm+TEMP+pos)),d2=unpack(*reinterpret_cast<uint32_t*>(sm+TEMP+32768+pos));
       float x=(acc[4*i+2*rr]-mu[rr])*vv[rr],y=(acc[4*i+2*rr+1]-mu[rr])*vv[rr];
       acc[4*i+2*rr]=x;acc[4*i+2*rr+1]=y;
       gg1[2*i]+=d1.x*x;gg1[2*i+1]+=d1.y*y;gg2[2*i]+=d2.x*x;gg2[2*i+1]+=d2.y*y;
       d1.x*=a1.x;d1.y*=a1.y;d2.x*=a2.x;d2.y*=a2.y;
       ma[0][rr]+=d1.x+d1.y;mb[0][rr]+=d1.x*x+d1.y*y;
       ma[1][rr]+=d2.x+d2.y;mb[1][rr]+=d2.x*x+d2.y*y;
     }
   }
   for(int k=0;k<2;++k)for(int rr=0;rr<2;++rr){ma[k][rr]=sum4(ma[k][rr])/128.f;mb[k][rr]=sum4(mb[k][rr])/128.f;}
   #pragma unroll
   for(int i=0;i<16;++i){
     int cx=8*i+cq;float2 a1=*reinterpret_cast<const float2*>(g1+cx),a2=*reinterpret_cast<const float2*>(g2+cx);
     #pragma unroll
     for(int rr=0;rr<2;++rr){
       int r=row+r0+8*rr,pos=off(r0+8*rr,cx);
       float2 d1=unpack(*reinterpret_cast<uint32_t*>(sm+TEMP+pos)),d2=unpack(*reinterpret_cast<uint32_t*>(sm+TEMP+32768+pos));
       float2 do1=unpack(*reinterpret_cast<uint32_t*>(sm+TEMP+16384+pos)),dt=unpack(*reinterpret_cast<uint32_t*>(sm+TEMP+49152+pos));
       float x=acc[4*i+2*rr],y=acc[4*i+2*rr+1];
       float dx10=rn(vv[rr]*(d1.x*a1.x-ma[0][rr]-x*mb[0][rr])),dx11=rn(vv[rr]*(d1.y*a1.y-ma[0][rr]-y*mb[0][rr]));
       float dx20=rn(vv[rr]*(d2.x*a2.x-ma[1][rr]-x*mb[1][rr])),dx21=rn(vv[rr]*(d2.y*a2.y-ma[1][rr]-y*mb[1][rr]));
       if(r<M){
         *reinterpret_cast<uint32_t*>(dc+size_t(r)*128+cx)=pack(rn(rn(dt.x+dx20)+do1.x)+dx10,rn(rn(dt.y+dx21)+do1.y)+dx11);
         *reinterpret_cast<uint32_t*>(cn1+size_t(r)*128+cx)=pack(a1.x*x,a1.y*y);
         *reinterpret_cast<uint32_t*>(cn2+size_t(r)*128+cx)=pack(a2.x*x,a2.y*y);
       }
     }
   }
   float* red=reinterpret_cast<float*>(sm+RED);
   #pragma unroll
   for(int i=0;i<32;++i){
     float u=gg1[i],v=gg2[i];
     u+=__shfl_xor_sync(~0u,u,4);u+=__shfl_xor_sync(~0u,u,8);u+=__shfl_xor_sync(~0u,u,16);
     v+=__shfl_xor_sync(~0u,v,4);v+=__shfl_xor_sync(~0u,v,8);v+=__shfl_xor_sync(~0u,v,16);
     int cx=8*(i/2)+cq+(i&1);
     if(l<4){red[w*256+cx]=u;red[w*256+128+cx]=v;}
   }
   bsync();
   float u=0,v=0;
   for(int ww=0;ww<4;++ww){u+=red[ww*256+t];v+=red[ww*256+128+t];}
   dgparts[size_t(tile)*256+t]=u;dgparts[size_t(tile)*256+128+t]=v;
   bsync();
 }
}

extern "C" __global__ void atom_column_partials_sm90(const __nv_bfloat16* dm,const __nv_bfloat16* dp,float* part,int M){
 __shared__ float s[16][16];
 const int x=threadIdx.x&15,y=threadIdx.x>>4,c=blockIdx.x*16+x,shard=blockIdx.y;
 const int lo=int((long long)M*shard/4),hi=int((long long)M*(shard+1)/4);
 float v=0;
 for(int r=lo+y;r<hi;r+=16)v+=__bfloat162float(c<768?dm[size_t(r)*768+c]:dp[size_t(r)*512+c-768]);
 s[y][x]=v;__syncthreads();
 if(y==0){float sum=0;for(int j=0;j<16;++j)sum+=s[j][x];part[shard*896+c]=sum;}
}
extern "C" __global__ void atom_column_finish_sm90(const float* part,const float* dgpart,
 float* dbias,float* dbq,float* dg,int ntile){
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<896){float v=0;for(int j=0;j<4;++j)v+=part[j*896+c];if(c<768)dbias[c]=v;else dbq[c-768]=v;}
 if(c<256){float v=0;for(int j=0;j<ntile;++j)v+=dgpart[size_t(j)*256+c];dg[c]=v;}
}

#else
// cond_bwd.cu — the atom DiT block's conditioning backward on sm_100a (dmod in the backward layout [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]):
//   dcn_k = rn(dsc_k Wsc_k + dbi_k Wbi_k)  (k = 1, 2; one fp32 accumulation);  dco = rn(dos Wos);  dct = rn(dts Wts)
//   cn_k = rn(g_k xhat):  dx_k = rn(LN_bwd(dcn_k g_k)),  d g_k += sum_rows dcn_k xhat
//   dc = the bf16 sum of dct, dx_2, dco, dx_1 (autograd's order);  cn_1, cn_2 written for the dWmod GEMMs
// Structure (post_fwd.cu's frame): transposed 16-row tiles; Wmod^T (128 x 768, the six blocks in the forward order) in TMEM as bf16 A
// operands (384 columns); the two compute warpgroups take alternate tiles, each with its own TMA producer warp, MMA warp, 2-stage ring
// and four accumulators (M = 128 channels, N = 16 rows): dcn1^T, dcn2^T, dco^T, dct^T.
// Stage (44 KB): dmod [16][768] | c -> dc | dcn1 | dcn2 | dco -> cn1 | dct -> cn2.
// TMEM: Wmod^T block j at 64 j; warpgroup w's accumulators at 384 + 64 w + 16 q.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NSW
#define NSW 2
#endif
#ifndef ORD
#define ORD 0                              // the order of dc's four bf16 additions
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_DM = 0, O_C = 6 * T_, O_N1 = 7 * T_, O_N2 = 8 * T_, O_CO = 9 * T_, O_CT = 10 * T_, STG = 11 * T_;
constexpr int O_BAR = (2 * NSW * STG > 6 * 32768 ? 2 * NSW * STG : 6 * 32768), O_RED = O_BAR + 512;
constexpr int SMEM = O_RED + 8 * 256 * 4;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_W = 0, T_ACC = 384;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, infull[2][NSW], infree[2][NSW], stfull[2][NSW], dfull[2], dfree[2];
  uint32_t tmem;
};
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_cond_bwd_sm100(const __grid_constant__ CUtensorMap mdmod, const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mwt,
                    const __grid_constant__ CUtensorMap mdc, const __grid_constant__ CUtensorMap mcn1, const __grid_constant__ CUtensorMap mcn2,
                    const float* __restrict__ G1, const float* __restrict__ G2, float* __restrict__ DG, int ntile, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto row0 = [&](int T) { return R * ((int)blockIdx.x + T * (int)gridDim.x); };
  auto stage = [&](int w, int i) { return su + (uint32_t)((w * NSW + i % NSW) * STG); };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1);
    for (int w = 0; w < 2; ++w) {
      for (int s = 0; s < NSW; ++s) { mbar_init(&B.infull[w][s], 1); mbar_init(&B.infree[w][s], 1); mbar_init(&B.stfull[w][s], 1); }
      mbar_init(&B.dfull[w], 1); mbar_init(&B.dfree[w], 4);
    }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA producers (warp 0: warpgroup 0, 2: 1)
    const int w = warp >> 1;
    if (lane == 0) {
      if (w == 0) {
        mbar_expect_tx(&B.wfull, 6 * 32768);                              // Wmod^T units: forward block j = columns 128 j ..
        for (int i = 0; i < 24; ++i) tma_load_2d(su + (i >> 2) * 32768 + (i & 3) * 8192, &mwt, &B.wfull, 32 * i, 0);
      }
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // dmod comes from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 14 * KB);
        for (int kb = 0; kb < 12; ++kb) tma_load_2d(st + O_DM + kb * KB, &mdmod, bar, 64 * kb, r0);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(st + O_C + kb * KB, &mc, bar, 64 * kb, r0);
      }
    } else if (lane == 1) {
      // the TMA stores of this warpgroup's tiles: issued (and waited for) off the compute threads, which go on to the next tile's row work
      pdl_wait();
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        const uint32_t st = stage(w, i);
        mbar_wait(&B.stfull[w][s], (i / NSW) & 1);
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mdc, st + O_C + kb * KB, 64 * kb, r0);
          tma_store_2d(&mcn1, st + O_CO + kb * KB, 64 * kb, r0);
          tma_store_2d(&mcn2, st + O_CT + kb * KB, 64 * kb, r0);
        }
        tma_store_commit();
        tma_store_wait_read0();                                            // (a short wait: the stores only have to read the stage)
        mbar_arrive(&B.infree[w][s]);                                      // this tile's stage is free for tile i + NSW
      }
      tma_store_wait0();
    }
  } else if (warp == 1 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1: warpgroup 0, 3: 1)
    const int w = warp >> 1;
    if (w == 0) {
      mbar_wait(&B.wfull, 0);
      tc_fence_after();
      if (elect_one()) {
        for (int u = 0; u < 6; ++u)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * u + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    // accumulator q: (weight block, dmod block) pairs -- q 0: (0, 0) + (1, 1); q 1: (3, 2) + (4, 3); q 2: (2, 4); q 3: (5, 5)
    auto mma = [&](uint32_t st, int q, int wb, int db, bool first) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
        umma_ts(ta + 16 * q, tmem + T_W + 64 * wb + ks * 8, desc_k128(st + O_DM + (2 * db + (ks >> 2)) * KB) + (uint64_t)((ks & 3) * 2), I_16,
                (first && ks == 0) ? 0u : 1u);
    };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      if (i > 0) mbar_wait(&B.dfree[w], (i - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        mma(st, 0, 0, 0, true); mma(st, 0, 1, 1, false);
        mma(st, 1, 3, 2, true); mma(st, 1, 4, 3, false);
        mma(st, 2, 2, 4, true);
        mma(st, 3, 5, 5, true);
        tc_commit(&B.dfull[w]);
      }
      __syncwarp();
    }
  } else {
    // ------------------------------------------------------------------------------------------------ compute warpgroups
    const int w = (warp - 4) >> 2, qw = warp & 3, ct = tid & 127;
    const uint32_t lb = (uint32_t)qw * 32;
    const bool leader = qw == 0 && lane == 0;
    const uint32_t ta = tmem + T_ACC + 64 * w;
    const int ar = ct >> 3, ak = ct & 7;
    const uint32_t aoff = sw128((uint32_t)ar, (uint32_t)ak);
    const uint32_t fj = (uint32_t)(8 * ((lane >> 3) & 1) + (lane & 7)), fh = (uint32_t)(lane >> 4);
    auto faddr = [&](uint32_t base, uint32_t ch) { return base + (ch >> 6) * KB + fj * 128u + ((((ch & 63u) >> 3) ^ (fj & 7u)) << 4); };
    float g1[16], g2[16], dg1[16], dg2[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      g1[e] = __ldg(G1 + c); g2[e] = __ldg(G2 + c); dg1[e] = 0.f; dg2[e] = 0.f;
    }
    const uint32_t stq[4] = {O_N1, O_N2, O_CO, O_CT};
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      // ---- P1: the four accumulators -> rn -> row-major staging
      mbar_wait(&B.dfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < 2; ++L) {
        uint32_t v[4][8];
#pragma unroll
        for (int q = 0; q < 4; ++q) tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 16 * q, v[q]);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint32_t r[4];
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            r[mi] = pack_bf16(__uint_as_float(v[q][k]), __uint_as_float(v[q][k + 1]));
          }
          stsm4t(faddr(st + stq[q], lb + 16 * L + 8 * fh), r);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dfree[w]);
      named_bar_sync(1 + w, 128);
      // ---- P2: row stage: LayerNorm recompute, the two LayerNorm backwards, dc; cn1 / cn2 over dco / dct
      {
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 cu = lds128(st + O_C + kb * KB + aoff);
          const uint32_t w4[4] = {cu.x, cu.y, cu.z, cu.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { x[8 * kb + 2 * e] = bf16lo(w4[e]); x[8 * kb + 2 * e + 1] = bf16hi(w4[e]); }
        }
        float sum = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) sum += x[e];
        sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
        const float mean = sum * (1.f / 128.f);
        float var = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) { x[e] -= mean; var += x[e] * x[e]; }
        var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
        const float rstd = rsqrtf(var * (1.f / 128.f) + eps);
#pragma unroll
        for (int e = 0; e < 16; ++e) x[e] *= rstd;                          // xhat
        float d1[16], d2[16], a1 = 0.f, b1 = 0.f, a2 = 0.f, b2 = 0.f;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 n1 = lds128(st + O_N1 + kb * KB + aoff), n2 = lds128(st + O_N2 + kb * KB + aoff);
          const uint32_t p4[4] = {n1.x, n1.y, n1.z, n1.w}, q4[4] = {n2.x, n2.y, n2.z, n2.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            const float u0 = bf16lo(p4[e]), u1 = bf16hi(p4[e]), v0 = bf16lo(q4[e]), v1 = bf16hi(q4[e]);
            dg1[k] += u0 * x[k]; dg1[k + 1] += u1 * x[k + 1]; dg2[k] += v0 * x[k]; dg2[k + 1] += v1 * x[k + 1];
            d1[k] = u0 * g1[k]; d1[k + 1] = u1 * g1[k + 1]; d2[k] = v0 * g2[k]; d2[k + 1] = v1 * g2[k + 1];
            a1 += d1[k] + d1[k + 1]; b1 += d1[k] * x[k] + d1[k + 1] * x[k + 1];
            a2 += d2[k] + d2[k + 1]; b2 += d2[k] * x[k] + d2[k + 1] * x[k + 1];
          }
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
          a1 += __shfl_xor_sync(~0u, a1, o); b1 += __shfl_xor_sync(~0u, b1, o); a2 += __shfl_xor_sync(~0u, a2, o); b2 += __shfl_xor_sync(~0u, b2, o);
        }
        a1 *= 1.f / 128.f; b1 *= 1.f / 128.f; a2 *= 1.f / 128.f; b2 *= 1.f / 128.f;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 co = lds128(st + O_CO + kb * KB + aoff), cq = lds128(st + O_CT + kb * KB + aoff);
          const uint32_t o4[4] = {co.x, co.y, co.z, co.w}, t4[4] = {cq.x, cq.y, cq.z, cq.w};
          uint32_t dc[4], c1[4], c2[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            const uint32_t x1 = pack_bf16(rstd * (d1[k] - a1 - x[k] * b1), rstd * (d1[k + 1] - a1 - x[k + 1] * b1));
            const uint32_t x2 = pack_bf16(rstd * (d2[k] - a2 - x[k] * b2), rstd * (d2[k + 1] - a2 - x[k + 1] * b2));
            dc[e] = ORD == 0 ? badd2(badd2(badd2(t4[e], x2), o4[e]), x1) : ORD == 1 ? badd2(badd2(badd2(x1, o4[e]), x2), t4[e])
                  : ORD == 2 ? badd2(badd2(badd2(t4[e], o4[e]), x2), x1) : badd2(badd2(badd2(x2, t4[e]), x1), o4[e]);
            c1[e] = pack_bf16(g1[k] * x[k], g1[k + 1] * x[k + 1]);
            c2[e] = pack_bf16(g2[k] * x[k], g2[k + 1] * x[k + 1]);
          }
          sts128(st + O_C + kb * KB + aoff, make_uint4(dc[0], dc[1], dc[2], dc[3]));
          sts128(st + O_CO + kb * KB + aoff, make_uint4(c1[0], c1[1], c1[2], c1[3]));
          sts128(st + O_CT + kb * KB + aoff, make_uint4(c2[0], c2[1], c2[2], c2[3]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.stfull[w][i % NSW]);                      // the store thread of this warpgroup takes it from here
    }
    float* red = reinterpret_cast<float*>(sm + O_RED);                      // [8 warps][256]
    const int cw = warp - 4;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      float v1 = dg1[e], v2 = dg2[e];
      v1 += __shfl_xor_sync(~0u, v1, 8); v1 += __shfl_xor_sync(~0u, v1, 16);
      v2 += __shfl_xor_sync(~0u, v2, 8); v2 += __shfl_xor_sync(~0u, v2, 16);
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      if ((lane >> 3) == 0) { red[cw * 256 + c] = v1; red[cw * 256 + 128 + c] = v2; }
    }
    named_bar_sync(3, 256);
    {
      const int c = tid - 128;
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) v += red[k * 256 + c];
      atomicAdd(DG + c, v);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

#endif // ATOM_HOPPER
