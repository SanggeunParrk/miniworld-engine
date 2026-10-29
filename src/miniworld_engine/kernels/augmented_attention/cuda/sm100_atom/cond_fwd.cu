
#if defined(ATOM_HOPPER)
#include "sm100.cuh"
using namespace atom90;
// Three conditioning GEMMs per half reuse the same normalized tile and keep
// the half's weights resident across a persistent row schedule.
extern "C" __global__ __launch_bounds__(160, 1)
void atom_cond_fwd_sm90(const __grid_constant__ CUtensorMap mc,
 const __grid_constant__ CUtensorMap mw,const float* __restrict__ g1,
 const float* __restrict__ g2,const float* __restrict__ bias,
 __nv_bfloat16* __restrict__ out,int M,float eps) {
 extern __shared__ __align__(1024) uint8_t sm[];
 constexpr int X=0,W=32768,BAR=W+3*32768;
 uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
 const int t=threadIdx.x,lane=t&31,warp=t>>5,hk=blockIdx.y;
 const int nt=(M+63)/64, tiles=blockIdx.x<nt?(nt-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x):0;
 if(t==0){for(int j=0;j<7;++j)init(bars+j);init_fence();}
 __syncthreads();
 if(t>=128) {
   if(t==128) {
     for(int p=0;p<3;++p) {
       expect(bars+p,32768);
       for(int k=0;k<2;++k)load2(sm+W+p*32768+k*16384,&mw,bars+p,64*k,384*hk+128*p);
     }
     for(int it=0;it<tiles;++it) {
       const int st=it&1,row=64*(int(blockIdx.x)+it*int(gridDim.x));
       if(it>=2)bwait(bars+5+st,((it>>1)-1)&1);
       expect(bars+3+st,16384);
       for(int k=0;k<2;++k)load2(sm+X+st*16384+k*8192,&mc,bars+3+st,64*k,row);
     }
   }
   return;
 }
 const int r0=warp*16+lane/4,cq=(lane&3)*2;
 for(int it=0;it<tiles;++it) {
   const int st=it&1,row=64*(int(blockIdx.x)+it*int(gridDim.x));
   bwait(bars+3+st,(it>>1)&1);
   uint32_t raw[32],norm[32];float xx[64],s[2]={0.f,0.f},v[2]={0.f,0.f};
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr) {
       const uint32_t x=*reinterpret_cast<uint32_t*>(sm+X+st*16384+off(r0+8*rr,8*i+cq));
       const float2 f=unpack(x);xx[4*i+2*rr]=f.x;xx[4*i+2*rr+1]=f.y;
       raw[4*(i>>1)+2*(i&1)+rr]=x;s[rr]+=f.x+f.y;
     }
   s[0]=sum4(s[0])/128.f;s[1]=sum4(s[1])/128.f;
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr) {
       const float a=xx[4*i+2*rr]-s[rr],b=xx[4*i+2*rr+1]-s[rr];v[rr]+=a*a+b*b;
     }
   v[0]=rsqrtf(sum4(v[0])/128.f+eps);v[1]=rsqrtf(sum4(v[1])/128.f+eps);
   const float* gamma=hk?g2:g1;
   #pragma unroll
   for(int i=0;i<16;++i) {
     const float2 g=*reinterpret_cast<const float2*>(gamma+8*i+cq);
     #pragma unroll
     for(int rr=0;rr<2;++rr)
       norm[4*(i>>1)+2*(i&1)+rr]=pack((xx[4*i+2*rr]-s[rr])*v[rr]*g.x,(xx[4*i+2*rr+1]-s[rr])*v[rr]*g.y);
   }
   bsync();
   if(t==0)asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sa(bars+5+st)):"memory");
   for(int p=0;p<3;++p) {
     bwait(bars+p,0);
     uint32_t input[32];
     #pragma unroll
     for(int j=0;j<32;++j)input[j]=p==2?raw[j]:norm[j];
     float acc[64];matmul128(input,sm+W+p*32768,acc);
     #pragma unroll
     for(int i=0;i<16;++i) {
       const int c=384*hk+128*p+8*i+cq;
       const float2 b=*reinterpret_cast<const float2*>(bias+c);
       #pragma unroll
       for(int rr=0;rr<2;++rr) {
         float x=rn(acc[4*i+2*rr]+b.x),y=rn(acc[4*i+2*rr+1]+b.y);
         if(p!=1){x=sig(x);y=sig(y);}
         const int r=row+r0+8*rr;
         if(r<M)*reinterpret_cast<uint32_t*>(out+size_t(r)*768+c)=pack(x,y);
       }
     }
   }
 }
}
extern "C" __global__ void atom_pack_sm90(
 const void* p0,const void* p1,const void* p2,const void* p3,const void* p4,const void* p5,
 const void* p6,const void* p7,const void* p8,const void* p9,const void* p10,const void* p11,
 const void* p12,const void* p13,const void* p14,const void* p15,const void* p16,const void* p17,
 const void* p18,const void* p19,const void* p20,const void* p21,const void* p22,
 __nv_bfloat16* wc,__nv_bfloat16* wp,__nv_bfloat16* wo,__nv_bfloat16* wu,__nv_bfloat16* wd,
 float* b,float* g1,float* g2,float* bq,int bfmask) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 auto value=[&](const void* p,int j,int n) {return (bfmask&(1<<n))?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[j]):reinterpret_cast<const float*>(p)[j];};
 if(i<98304) {
   const int h=i/16384,j=i%16384;
   const void* p=h==0?p1:h==1?p3:h==2?p12:h==3?p15:h==4?p17:p21;
   const int ix=h==0?1:h==1?3:h==2?12:h==3?15:h==4?17:21;
   wc[i]=__float2bfloat16_rn(value(p,j,ix));
 }
 if(i<65536) {
   const int h=i/16384,j=i%16384;const void* p=h==0?p4:h==1?p6:h==2?p7:p10;
   const int ix=h==0?4:h==1?6:h==2?7:10;
   wp[i]=__float2bfloat16_rn(value(p,j,ix));
   wu[i]=__float2bfloat16_rn(value(i<32768?p18:p19,i%32768,i<32768?18:19));
 }
 if(i<16384)wo[i]=__float2bfloat16_rn(value(p11,i,11));
 if(i<32768)wd[i]=__float2bfloat16_rn(value(p20,i,20));
 if(i<768) {
   const int h=i/128,j=i%128;const void* p=h==0?p2:h==2?p13:h==3?p16:p22;
   const int ix=h==0?2:h==2?13:h==3?16:22;
   b[i]=(h==1||h==4)?0.f:value(p,j,ix);
 }
 if(i<128){g1[i]=value(p0,i,0);g2[i]=value(p14,i,14);bq[i]=value(p5,i,5);}
}

extern "C" __global__ __launch_bounds__(160, 1)
void atom_cond_fwd_single_sm90(const __grid_constant__ CUtensorMap mc,
 const __grid_constant__ CUtensorMap mw,const float* __restrict__ g1,
 const float* __restrict__ g2,const float* __restrict__ bias,
 __nv_bfloat16* __restrict__ out,int M,float eps) {
 extern __shared__ __align__(1024) uint8_t sm[];
 constexpr int X=0,W=16384,BAR=W+3*32768;
 uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
 const int t=threadIdx.x,lane=t&31,warp=t>>5,hk=blockIdx.y;
 const int nt=(M+63)/64, tiles=blockIdx.x<nt?(nt-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x):0;
 if(t==0){for(int j=0;j<5;++j)init(bars+j);init_fence();}
 __syncthreads();
 if(t>=128) {
   if(t==128) {
     for(int p=0;p<3;++p) {
       expect(bars+p,32768);
       for(int k=0;k<2;++k)load2(sm+W+p*32768+k*16384,&mw,bars+p,64*k,384*hk+128*p);
     }
     for(int it=0;it<tiles;++it) {
       const int st=0,row=64*(int(blockIdx.x)+it*int(gridDim.x));
       if(it)bwait(bars+4,(it-1)&1);
       expect(bars+3+st,16384);
       for(int k=0;k<2;++k)load2(sm+X+st*16384+k*8192,&mc,bars+3+st,64*k,row);
     }
   }
   return;
 }
 const int r0=warp*16+lane/4,cq=(lane&3)*2;
 for(int it=0;it<tiles;++it) {
   const int st=0,row=64*(int(blockIdx.x)+it*int(gridDim.x));
   bwait(bars+3,it&1);
   uint32_t raw[32],norm[32];float xx[64],s[2]={0.f,0.f},v[2]={0.f,0.f};
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr) {
       const uint32_t x=*reinterpret_cast<uint32_t*>(sm+X+st*16384+off(r0+8*rr,8*i+cq));
       const float2 f=unpack(x);xx[4*i+2*rr]=f.x;xx[4*i+2*rr+1]=f.y;
       raw[4*(i>>1)+2*(i&1)+rr]=x;s[rr]+=f.x+f.y;
     }
   s[0]=sum4(s[0])/128.f;s[1]=sum4(s[1])/128.f;
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr) {
       const float a=xx[4*i+2*rr]-s[rr],b=xx[4*i+2*rr+1]-s[rr];v[rr]+=a*a+b*b;
     }
   v[0]=rsqrtf(sum4(v[0])/128.f+eps);v[1]=rsqrtf(sum4(v[1])/128.f+eps);
   const float* gamma=hk?g2:g1;
   #pragma unroll
   for(int i=0;i<16;++i) {
     const float2 g=*reinterpret_cast<const float2*>(gamma+8*i+cq);
     #pragma unroll
     for(int rr=0;rr<2;++rr)
       norm[4*(i>>1)+2*(i&1)+rr]=pack((xx[4*i+2*rr]-s[rr])*v[rr]*g.x,(xx[4*i+2*rr+1]-s[rr])*v[rr]*g.y);
   }
   bsync();
   if(t==0)asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sa(bars+4)):"memory");
   for(int p=0;p<3;++p) {
     bwait(bars+p,0);
     uint32_t input[32];
     #pragma unroll
     for(int j=0;j<32;++j)input[j]=p==2?raw[j]:norm[j];
     float acc[64];matmul128(input,sm+W+p*32768,acc);
     #pragma unroll
     for(int i=0;i<16;++i) {
       const int c=384*hk+128*p+8*i+cq;
       const float2 b=*reinterpret_cast<const float2*>(bias+c);
       #pragma unroll
       for(int rr=0;rr<2;++rr) {
         float x=rn(acc[4*i+2*rr]+b.x),y=rn(acc[4*i+2*rr+1]+b.y);
         if(p!=1){x=sig(x);y=sig(y);}
         const int r=row+r0+8*rr;
         if(r<M)*reinterpret_cast<uint32_t*>(out+size_t(r)*768+c)=pack(x,y);
       }
     }
   }
 }
}
#else
// cond_fwd.cu — the atom DiT block's conditioning projections on sm_100a (per row r of M = A N, c bf16 [M, 128]; the bf16 module's
// rounding points):
//   cn_k = rn(LayerNorm_{g_k}(c))  (weight only, eps 1e-5; the attention's and the transition's AdaLN, k = 1, 2)
//   mod = [ s1 | bi1 | so | s2 | bi2 | st ]  bf16 [M, 768]:
//     s1 = rn(sigmoid(rn(cn1 Wsc1^T + bsc1)))   bi1 = rn(cn1 Wbi1^T)   so = rn(sigmoid(rn(c Wos^T + bos)))   (attention: AdaLN scale / shift, output gate)
//     s2 = rn(sigmoid(rn(cn2 Wsc2^T + bsc2)))   bi2 = rn(cn2 Wbi2^T)   st = rn(sigmoid(rn(c Wts^T + bts)))   (transition: the same)
//   The module only ever uses the four scales through their sigmoids, so they leave as rn(sigmoid) (the same rounding points, and what
//   autograd's sigmoid backward reads); the consumers (pre_fwd, post_fwd) then need no transcendental for them.
// Structure (pre_fwd.cu's frame): blockIdx.y = the half k (blocks 3 k .. 3 k + 2: LayerNorm g_k), persistent over 32-row tiles; the half's
// three weights (3 x 128 x 128) in TMEM as bf16 A operands, p^T = W_p x^T (M = 128 output channels, N = 32 rows; x = cn_k for p < 2, the
// raw c tile for p = 2) into double-buffered accumulators. Rings: c tiles x NI (freed by the MMAs' commit), cn_k x 2, outputs x 2.
// Threads: the LayerNorm eight threads per row (16 channels each); the epilogue in the mma-fragment layout (tcgen05.ld 16x256b,
// warpgroup w: rows 16 w ..), + bias, sigmoid, rows paired into bf16x2, stored transposed by stmatrix.trans; TMA stores.
// TMEM: W_p at 64 p (192 columns); accumulator set b at 256 + 128 b, projection p at + 32 p.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NI
#define NI 4
#endif
#ifndef RMS
#define RMS 0                      // ESMFold2: every table from rn(silu(c)), no sigmoid (scales identity-centred by their bias 1); gridDim.y = 3,
#endif                             // the third half the K / V AdaLN's [scale | shift] into mout2 (its third projection is not stored)
constexpr int C = 128;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // [32][128] row-major tile (8 KB)
constexpr int O_IN = 0, O_X = NI * T_, O_O = O_X + 2 * T_, OST = 3 * T_, O_BAR = O_O + 2 * OST;
constexpr int SMEM = O_BAR + 256;
static_assert(SMEM <= 232448, "shared memory");
static_assert(O_BAR >= 3 * 32768, "the weight units stage through the rings");
constexpr uint32_t T_W = 0, T_ACC = 256;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t wfull, wfree, infull[NI], infree[NI], xfull[2], ofull[2], ofree[2], accfull[2], accfree[2];
  uint32_t tmem;
};
DEVI float sigm(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_cond_fwd_sm100(const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mout,
                    const __grid_constant__ CUtensorMap mout2, const float* __restrict__ G1, const float* __restrict__ G2, const float* __restrict__ BIAS, int ntile, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, hk = blockIdx.y;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto row0 = [&](int T) { return 32 * ((int)blockIdx.x + T * (int)gridDim.x); };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1);
    for (int i = 0; i < NI; ++i) { mbar_init(&B.infull[i], 1); mbar_init(&B.infree[i], 1); }
    for (int i = 0; i < 2; ++i) {
      mbar_init(&B.xfull[i], 2); mbar_init(&B.ofull[i], 2); mbar_init(&B.ofree[i], 1); mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 2);
    }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      mbar_expect_tx(&B.wfull, 3 * 32768);                                // W_p: K-major SW64 [128][32] atoms, 32 KB per projection
      for (int p = 0; p < 3; ++p)
        for (int ka = 0; ka < 4; ++ka) tma_load_2d(su + p * 32768 + ka * 8192, &mw, &B.wfull, 32 * ka, 384 * hk + 128 * p);
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // c comes from the previous kernels
      for (int T = 0; T < ntT; ++T) {
        const int s = T % NI;
        if (T >= NI) mbar_wait(&B.infree[s], ((T / NI) - 1) & 1);
        mbar_expect_tx(&B.infull[s], 2 * KBLK);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_IN + s * T_ + kb * KBLK, &mc, &B.infull[s], 64 * kb, row0(T));
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    mbar_wait(&B.wfull, 0);
    tc_fence_after();
    if (elect_one()) {
      for (int p = 0; p < 3; ++p)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * p + ks * 8, desc_sw64(su + p * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
      tc_commit(&B.wfree);
    }
    __syncwarp();
    for (int T = 0; T < ntT; ++T) {
      const int ab = T & 1, s = T % NI;
      if (T >= 2) mbar_wait(&B.accfree[ab], ((T >> 1) - 1) & 1);
      mbar_wait(&B.xfull[ab], (T >> 1) & 1);                               // cn_k written; the c tile has landed (the LayerNorm read it)
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int p = 0; p < 3; ++p) {
          const uint32_t xa = (RMS || p < 2) ? su + O_X + ab * T_ : su + O_IN + s * T_;
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_ACC + 128 * ab + 32 * p, tmem + T_W + 64 * p + ks * 8, desc_k128(xa + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2), I_32,
                    ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.accfull[ab]);
        tc_commit(&B.infree[s]);                                           // the c tile (and cn_k) are free once these MMAs are done
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores
    if (lane == 0) {
      pdl_wait();
      for (int T = 0; T < ntT; ++T) {
        const int ob = T & 1;
        mbar_wait(&B.ofull[ob], (T >> 1) & 1);
        for (int p = 0; p < 3; ++p)
          for (int kb = 0; kb < 2; ++kb) {
            if (RMS && hk == 2) {
              if (p < 2) tma_store_2d(&mout2, su + O_O + ob * OST + p * T_ + kb * KBLK, 128 * p + 64 * kb, row0(T));
            } else {
              tma_store_2d(&mout, su + O_O + ob * OST + p * T_ + kb * KBLK, 384 * hk + 128 * p + 64 * kb, row0(T));
            }
          }
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.ofree[ob]);
      }
      tma_store_wait0();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ compute threads
    const int ct = tid - 128, wg = ct >> 7, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    // LayerNorm: row ar = ct / 8 (rows 16 wg ..), 16 channels: 16-B chunk ak of both 64-channel blocks
    const int ar = ct >> 3, ak = ct & 7;
    const uint32_t aoff = sw128((uint32_t)ar, (uint32_t)ak);
    const float* G = hk ? G2 : G1;
    float gk[16];
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
#pragma unroll
      for (int e = 0; e < 8; ++e) gk[8 * kb + e] = RMS ? 1.f : __ldg(G + 64 * kb + 8 * ak + e);
    // epilogue: lane i addresses row fj = 16 wg + 8 ((i >> 3) & 1) + (i & 7) of channel group 8 (i >> 4) of a 16-lane half
    const uint32_t fj = (uint32_t)(16 * wg + 8 * ((lane >> 3) & 1) + (lane & 7)), fh = (uint32_t)(lane >> 4);
    float bias[3][4];                                                      // channel lb + 16 L + 8 h + lane / 4 of projection p: [p][2 L + h]
#pragma unroll
    for (int p = 0; p < 3; ++p)
#pragma unroll
      for (int i = 0; i < 4; ++i) bias[p][i] = __ldg(BIAS + 384 * hk + 128 * p + lb + 8 * i + (lane >> 2));
    auto xphase = [&](int T) {
      const int s = T % NI, xb = T & 1;
      const uint32_t st = su + O_IN + s * T_, xs = su + O_X + xb * T_;
      mbar_wait(&B.infull[s], (T / NI) & 1);
      uint4 u[2];
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) u[kb] = lds128(st + kb * KBLK + aoff);
      float x[16];
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const uint32_t w4[4] = {u[kb].x, u[kb].y, u[kb].z, u[kb].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[8 * kb + 2 * e] = bf16lo(w4[e]); x[8 * kb + 2 * e + 1] = bf16hi(w4[e]); }
      }
#if RMS
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {                                     // rn(silu(c))
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int k = 8 * kb + 2 * e;
          o[e] = pack_bf16(x[k] * sigm(x[k]), x[k + 1] * sigm(x[k + 1]));
        }
        sts128(xs + kb * KBLK + aoff, make_uint4(o[0], o[1], o[2], o[3]));
      }
      (void)eps; (void)gk;
#else
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) sum += x[e];
      sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
      const float mean = sum * (1.f / C);
      float var = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) { const float d = x[e] - mean; var += d * d; }
      var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
      const float rstd = rsqrtf(var * (1.f / C) + eps);
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int k = 8 * kb + 2 * e;
          o[e] = pack_bf16((x[k] - mean) * rstd * gk[k], (x[k + 1] - mean) * rstd * gk[k + 1]);
        }
        sts128(xs + kb * KBLK + aoff, make_uint4(o[0], o[1], o[2], o[3]));
      }
#endif
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.xfull[xb]);
    };
    auto epi = [&](int T) {
      const int ab = T & 1, ob = T & 1;
      mbar_wait(&B.accfull[ab], (T >> 1) & 1);
      if (T >= 2) mbar_wait(&B.ofree[ob], ((T >> 1) - 1) & 1);
      tc_fence_after();
      uint32_t va[3][2][8];                                                // all six accumulator reads in flight, one wait
#pragma unroll
      for (int p = 0; p < 3; ++p)
#pragma unroll
        for (int L = 0; L < 2; ++L) tmem_ld16x256b2(tmem + ((lb + 16 * L) << 16) + T_ACC + 128 * ab + 32 * p + 16 * wg, va[p][L]);
      tmem_wait_ld();
#pragma unroll
      for (int p = 0; p < 3; ++p) {
        const bool gate = !RMS && p != 1;
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          uint32_t r[4];
          const uint32_t (&v)[8] = va[p][L];
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            const float b = bias[p][2 * L + (mi >> 1)];
            const uint32_t y2 = pack_bf16(__uint_as_float(v[k]) + b, __uint_as_float(v[k + 1]) + b);
            r[mi] = gate ? sig_rn2(y2) : y2;
          }
          const uint32_t ch = lb + 16 * L + 8 * fh;
          stsm4t(su + O_O + ob * OST + p * T_ + (ch >> 6) * KBLK + fj * 128u + ((((ch & 63u) >> 3) ^ (fj & 7u)) << 4), r);
        }
      }
      tc_fence_before();
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) { mbar_arrive(&B.accfree[ab]); mbar_arrive(&B.ofull[ob]); }
    };
    // The two warpgroups are staggered by half an iteration: warpgroup 0 runs tile T + 1's LayerNorm and then tile T's epilogue, warpgroup 1 tile
    // T's epilogue and then tile T + 2's LayerNorm, so one's sigmoids (MUFU) overlap the other's row work.
    if (wg == 1) {
      if (ntT > 0) xphase(0);
      if (ntT > 1) xphase(1);
      for (int T = 0; T < ntT; ++T) {
        epi(T);
        if (T + 2 < ntT) xphase(T + 2);
      }
    } else {
      if (ntT > 0) xphase(0);
      for (int T = 0; T < ntT; ++T) {
        if (T + 1 < ntT) xphase(T + 1);
        epi(T);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

#endif // ATOM_HOPPER
