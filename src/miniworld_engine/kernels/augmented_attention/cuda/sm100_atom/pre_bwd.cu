
#if defined(ATOM_HOPPER)
#include "sm100.cuh"
using namespace atom90;
extern "C" __global__ __launch_bounds__(160,1)
void atom_pre_bwd_sm90(const __nv_bfloat16* dp,const __nv_bfloat16* a,const __nv_bfloat16* mod,
 const __nv_bfloat16* residual,const __grid_constant__ CUtensorMap mw,
 __nv_bfloat16* ds,__nv_bfloat16* dm,int M,float eps) {
 extern __shared__ __align__(1024) uint8_t sm[];
 constexpr int NST=3,BAR=NST*32768;
 uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
 int t=threadIdx.x,l=t&31,w=t>>5,nt=(M+63)/64;
 int tiles=blockIdx.x<nt?(nt-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x):0;
 if(t==0){for(int j=0;j<6;++j)init(bars+j);init_fence();}__syncthreads();
 if(t>=128){
   if(t==128)for(int it=0;it<tiles;++it)for(int p=0;p<4;++p){
     int g=it*4+p,st=g%NST;
     if(g>=NST)bwait(bars+NST+st,((g/NST)-1)&1);
     expect(bars+st,32768);
     for(int k=0;k<2;++k)load2(sm+st*32768+k*16384,&mw,bars+st,128*(3-p)+64*k,0);
   }
   return;
 }
 int r0=w*16+l/4,cq=2*(l&3);
 for(int it=0;it<tiles;++it){
   int row=64*(int(blockIdx.x)+it*int(gridDim.x));
   float dx[64],acc[64];
   for(int p=0;p<4;++p){
     uint32_t fa[32];
     #pragma unroll
     for(int i=0;i<16;++i)
       #pragma unroll
       for(int rr=0;rr<2;++rr){
         int r=row+r0+8*rr,c=8*i+cq;
         fa[4*(i>>1)+2*(i&1)+rr]=r<M?*reinterpret_cast<const uint32_t*>(dp+size_t(r)*512+128*(3-p)+c):0;
       }
     int g=it*4+p,st=g%NST;bwait(bars+st,(g/NST)&1);matmul128(fa,sm+st*32768,acc);
     bsync();if(t==0)asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sa(bars+NST+st)):"memory");
     #pragma unroll
     for(int i=0;i<64;++i)dx[i]=p==0?rn(acc[i]):rn(dx[i]+rn(acc[i]));
   }
   float mu[2]={0,0},vv[2]={0,0};
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr){
       int r=row+r0+8*rr,c=8*i+cq;
       float2 x=r<M?unpack(*reinterpret_cast<const uint32_t*>(a+size_t(r)*128+c)):make_float2(0,0);
       acc[4*i+2*rr]=x.x;acc[4*i+2*rr+1]=x.y;mu[rr]+=x.x+x.y;
     }
   mu[0]=sum4(mu[0])/128.f;mu[1]=sum4(mu[1])/128.f;
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr){float x=acc[4*i+2*rr]-mu[rr],y=acc[4*i+2*rr+1]-mu[rr];vv[rr]+=x*x+y*y;}
   vv[0]=rsqrtf(sum4(vv[0])/128.f+eps);vv[1]=rsqrtf(sum4(vv[1])/128.f+eps);
   float m1[2]={0,0},m2[2]={0,0};
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr){
       int r=row+r0+8*rr,c=8*i+cq;
       float2 s=r<M?unpack(*reinterpret_cast<const uint32_t*>(mod+size_t(r)*768+c)):make_float2(0,0);
       float xh0=(acc[4*i+2*rr]-mu[rr])*vv[rr],xh1=(acc[4*i+2*rr+1]-mu[rr])*vv[rr];
       float d0=dx[4*i+2*rr],d1=dx[4*i+2*rr+1];
       if(r<M){
         *reinterpret_cast<uint32_t*>(dm+size_t(r)*768+c)=pack(rn(rn(d0*rn(xh0))*rn(1.f-s.x))*s.x,rn(rn(d1*rn(xh1))*rn(1.f-s.y))*s.y);
         *reinterpret_cast<uint32_t*>(dm+size_t(r)*768+128+c)=pack(d0,d1);
       }
       d0=rn(d0*s.x);d1=rn(d1*s.y);
       dx[4*i+2*rr]=d0;dx[4*i+2*rr+1]=d1;acc[4*i+2*rr]=xh0;acc[4*i+2*rr+1]=xh1;
       m1[rr]+=d0+d1;m2[rr]+=d0*xh0+d1*xh1;
     }
   for(int rr=0;rr<2;++rr){m1[rr]=sum4(m1[rr])/128.f;m2[rr]=sum4(m2[rr])/128.f;}
   #pragma unroll
   for(int i=0;i<16;++i)
     #pragma unroll
     for(int rr=0;rr<2;++rr){
       int r=row+r0+8*rr,c=8*i+cq;
       if(r<M){
         float2 da=unpack(*reinterpret_cast<const uint32_t*>(residual+size_t(r)*128+c));
         *reinterpret_cast<uint32_t*>(ds+size_t(r)*128+c)=pack(da.x+rn(vv[rr]*(dx[4*i+2*rr]-m1[rr]-acc[4*i+2*rr]*m2[rr])),
                                                              da.y+rn(vv[rr]*(dx[4*i+2*rr+1]-m1[rr]-acc[4*i+2*rr+1]*m2[rr])));
       }
     }
 }
}

extern "C" __global__ void atom_grad_pack_sm90(const float* p0,const float* p1,const float* p2,const float* p3,const float* p4,const float* p5,const float* p6,const float* p7,const float* p8,const float* p9,const float* p10,const float* p11,const float* p12,const float* p13,const float* p14,const float* p15,const float* p16,const float* p17,const float* p18,const float* p19,const float* p20,const float* p21,const float* p22,void* q0,void* q1,void* q2,void* q3,void* q4,void* q5,void* q6,void* q7,void* q8,void* q9,void* q10,void* q11,void* q12,void* q13,void* q14,void* q15,void* q16,void* q17,void* q18,void* q19,void* q20,void* q21,void* q22,int bfmask) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<128){float v=p0[i];if(bfmask&(1<<0))reinterpret_cast<__nv_bfloat16*>(q0)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q0)[i]=v;}
 if(i<16384){float v=p1[i];if(bfmask&(1<<1))reinterpret_cast<__nv_bfloat16*>(q1)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q1)[i]=v;}
 if(i<128){float v=p2[i];if(bfmask&(1<<2))reinterpret_cast<__nv_bfloat16*>(q2)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q2)[i]=v;}
 if(i<16384){float v=p3[i];if(bfmask&(1<<3))reinterpret_cast<__nv_bfloat16*>(q3)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q3)[i]=v;}
 if(i<16384){float v=p4[i];if(bfmask&(1<<4))reinterpret_cast<__nv_bfloat16*>(q4)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q4)[i]=v;}
 if(i<128){float v=p5[i];if(bfmask&(1<<5))reinterpret_cast<__nv_bfloat16*>(q5)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q5)[i]=v;}
 if(i<16384){float v=p6[i];if(bfmask&(1<<6))reinterpret_cast<__nv_bfloat16*>(q6)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q6)[i]=v;}
 if(i<16384){float v=p7[i];if(bfmask&(1<<7))reinterpret_cast<__nv_bfloat16*>(q7)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q7)[i]=v;}
 if(i<16){float v=p8[i];if(bfmask&(1<<8))reinterpret_cast<__nv_bfloat16*>(q8)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q8)[i]=v;}
 if(i<64){float v=p9[i];if(bfmask&(1<<9))reinterpret_cast<__nv_bfloat16*>(q9)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q9)[i]=v;}
 if(i<16384){float v=p10[i];if(bfmask&(1<<10))reinterpret_cast<__nv_bfloat16*>(q10)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q10)[i]=v;}
 if(i<16384){float v=p11[i];if(bfmask&(1<<11))reinterpret_cast<__nv_bfloat16*>(q11)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q11)[i]=v;}
 if(i<16384){float v=p12[i];if(bfmask&(1<<12))reinterpret_cast<__nv_bfloat16*>(q12)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q12)[i]=v;}
 if(i<128){float v=p13[i];if(bfmask&(1<<13))reinterpret_cast<__nv_bfloat16*>(q13)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q13)[i]=v;}
 if(i<128){float v=p14[i];if(bfmask&(1<<14))reinterpret_cast<__nv_bfloat16*>(q14)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q14)[i]=v;}
 if(i<16384){float v=p15[i];if(bfmask&(1<<15))reinterpret_cast<__nv_bfloat16*>(q15)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q15)[i]=v;}
 if(i<128){float v=p16[i];if(bfmask&(1<<16))reinterpret_cast<__nv_bfloat16*>(q16)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q16)[i]=v;}
 if(i<16384){float v=p17[i];if(bfmask&(1<<17))reinterpret_cast<__nv_bfloat16*>(q17)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q17)[i]=v;}
 if(i<32768){float v=p18[i];if(bfmask&(1<<18))reinterpret_cast<__nv_bfloat16*>(q18)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q18)[i]=v;}
 if(i<32768){float v=p19[i];if(bfmask&(1<<19))reinterpret_cast<__nv_bfloat16*>(q19)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q19)[i]=v;}
 if(i<32768){float v=p20[i];if(bfmask&(1<<20))reinterpret_cast<__nv_bfloat16*>(q20)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q20)[i]=v;}
 if(i<16384){float v=p21[i];if(bfmask&(1<<21))reinterpret_cast<__nv_bfloat16*>(q21)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q21)[i]=v;}
 if(i<128){float v=p22[i];if(bfmask&(1<<22))reinterpret_cast<__nv_bfloat16*>(q22)[i]=__float2bfloat16_rn(v);else reinterpret_cast<float*>(q22)[i]=v;}
}

#else
// pre_bwd.cu — the atom DiT block's attention-input backward on sm_100a (the bf16 module's autograd rounding points):
//   dx1 = the bf16 sum of rn(dq Wq), rn(dk Wk), rn(dv Wv), rn(dg Wg)          (dP = [dq | dk | dv | dg] [M, 512])
//   x1 = rn(rn(s1 xn) + bi1), xn = rn(LN(a)):  dbi1 = dx1;  ds1 = rn(dx1 xn);  dxn = rn(dx1 s1);  dsc1 = rn(rn(ds1 rn(1 - s1)) s1)
//   d single = rn(DA + rn(LN_bwd(dxn)))                                     (DA: post_bwd's residual path)
// -> d single [M, 128]; dmod blocks 0 (dsc1), 1 (dbi1) of the backward layout; d bsc1 into DBIAS[0:128], d bq = sum_rows dq into DBQ.
// Structure (post_fwd.cu's frame): transposed 16-row tiles; W^T = [Wq; Wk; Wv; Wg]^T (128 x 512) in TMEM as the bf16 A operand (256
// columns); the two compute warpgroups take alternate tiles, each with its own TMA producer warp, MMA warp, NSW-stage ring and four
// accumulators dx_p^T = W_p^T dP_p^T (M = 128 channels, N = 16 rows, K = 128).
// Stage (32 KB): dP [16][512] | a | s1 -> dsc1 | DA -> d single | dx1.
// Cross-attention mode (-DCROSS=1, ``pre_bwd_kv``): the keys and values come from a second AdaLN of x1, xkv = LN(x1) sigmoid(pre + bs) + shift
// (mkv = [pre | shift]), so the K / V projections' input gradient is that AdaLN's: with the real Wk, Wv in W^T, projections 1 and 2 give
//   dxkv = rn(dk Wk + dv Wv)   (one fp32 sum, one rounding: a K = 256 product)
// which goes through the AdaLN's backward in the row stage (x1 and pre are staged: two more tiles) into dx1_kv, added to dx1 = rn(dg Wg + dq Wq)
// in bf16 before AdaLN 1's backward; dmkv = [dpre | dxkv] is stored and d bs reduced. 44-KB stages, two per warpgroup. This replaces a K = 256
// cuBLAS GEMM, a separate K / V AdaLN backward kernel and the dx1_kv round trip of the separate formulation.
// -DPARALLEL=1 (AlphaFold 3, xkv = LN(a) sigmoid(pre + bs) + shift): the K / V AdaLN's backward reads the staged a (x1 is not loaded) and its
// LN backward is added to d single instead of dx1.
// TMEM: W^T 0 (256 columns, projection p at 64 p); warpgroup w's accumulators at 256 + 64 w + 16 p.
// mrows: the rows of the conditioning tables (mod, mkv): M, or N when the samples share one conditioning (table row = r % mrows)
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CROSS
#define CROSS 0
#endif
#ifndef PARALLEL
#define PARALLEL 0                         // CROSS: the K / V AdaLN of the raw atoms a (AlphaFold 3): its LN backward goes to d single
#endif
#ifndef NSW
#define NSW (CROSS ? 2 : 3)                // input stages per warpgroup
#endif
#ifndef ORD
#define ORD 0                              // the order of the four bf16 additions (autograd's)
#endif
#ifndef RMS
#define RMS 0                              // ESMFold2 (with CROSS, PARALLEL): RMSNorm AdaLNs, the scales raw (1 + s: their tables hold s + 1)
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_DP = 0, O_A = 4 * T_, O_S1 = 5 * T_, O_DA = 6 * T_, O_X = 7 * T_, O_DXKV = 8 * T_, O_X1 = 9 * T_, O_MP = 10 * T_,
              STG = (CROSS ? 11 : 8) * T_;
constexpr int O_BAR = (2 * NSW * STG > 4 * 32768 ? 2 * NSW * STG : 4 * 32768), O_RED = O_BAR + 512;
constexpr int NRED = CROSS ? 384 : 256;                                    // per-warp partial sums: d bsc1 | d bq (| d bs)
constexpr int SMEM = O_RED + 8 * NRED * 4;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_W = 0, T_ACC = 256;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, infull[2][NSW], infree[2][NSW], stfull[2][NSW], dxfull[2], dxfree[2];
  uint32_t tmem;
};
DEVI float sigm_x(float x) { return RMS ? x : __fdividef(1.f, 1.f + __expf(-x)); }   // the K / V AdaLN's scale (RMS: raw)
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t bsub2(uint32_t a, uint32_t b) { uint32_t r; asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t sig_bwd2(uint32_t g, uint32_t y) { return bmul2(bmul2(g, bsub2(0x3F803F80u, y)), y); }   // torch's bf16 sigmoid backward
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_pre_bwd_sm100(const __grid_constant__ CUtensorMap mdp, const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mmod,
                   const __grid_constant__ CUtensorMap mda, const __grid_constant__ CUtensorMap mwt, const __grid_constant__ CUtensorMap mds,
                   const __grid_constant__ CUtensorMap mdmod, const __grid_constant__ CUtensorMap mx1, const __grid_constant__ CUtensorMap mmkv,
                   const __grid_constant__ CUtensorMap mdmkv, float* __restrict__ DBIAS, float* __restrict__ DBQ, float* __restrict__ DBS,
                   const float* __restrict__ BS, int ntile, float eps, int mrows) {
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
      mbar_init(&B.dxfull[w], 1); mbar_init(&B.dxfree[w], 4);
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
        mbar_expect_tx(&B.wfull, 4 * 32768);                              // W^T units: projection p = columns 128 p ..
        for (int i = 0; i < 16; ++i) tma_load_2d(su + (i >> 2) * 32768 + (i & 3) * 8192, &mwt, &B.wfull, 32 * i, 0);
      }
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // dP / DA come from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, (CROSS ? (PARALLEL ? 16 : 18) : 14) * KB);
        if (CROSS)
          for (int kb = 0; kb < 2; ++kb) {
            if (!PARALLEL) tma_load_2d(st + O_X1 + kb * KB, &mx1, bar, 64 * kb, r0);
            tma_load_2d(st + O_MP + kb * KB, &mmkv, bar, 64 * kb, r0 % mrows);        // the pre-sigmoid scale: mkv's columns 0 .. 127
          }
        for (int kb = 0; kb < 8; ++kb) tma_load_2d(st + O_DP + kb * KB, &mdp, bar, 64 * kb, r0);
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_2d(st + O_A + kb * KB, &ma, bar, 64 * kb, r0);
          tma_load_2d(st + O_S1 + kb * KB, &mmod, bar, 64 * kb, r0 % mrows);        // s1: the forward mod's block 0
          tma_load_2d(st + O_DA + kb * KB, &mda, bar, 64 * kb, r0);
        }
      }
    } else if (lane == 1) {
      // the TMA stores of this warpgroup's tiles: issued (and waited for) off the compute threads, which go on to the next tile's row work
      pdl_wait();
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        const uint32_t st = stage(w, i);
        mbar_wait(&B.stfull[w][s], (i / NSW) & 1);
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mds, st + O_DA + kb * KB, 64 * kb, r0);
          tma_store_2d(&mdmod, st + O_S1 + kb * KB, 64 * kb, r0);          // dsc1 -> block 0
          tma_store_2d(&mdmod, st + O_X + kb * KB, 128 + 64 * kb, r0);     // dbi1 = dx1 -> block 1
          if (CROSS) {
            tma_store_2d(&mdmkv, st + O_MP + kb * KB, 64 * kb, r0);        // dmkv = [dpre | dxkv]
            tma_store_2d(&mdmkv, st + O_DXKV + kb * KB, 128 + 64 * kb, r0);
          }
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
        for (int u = 0; u < 4; ++u)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * u + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      if (i > 0) mbar_wait(&B.dxfree[w], (i - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int p = 0; p < 4; ++p)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(ta + 16 * p, tmem + T_W + 64 * p + ks * 8, desc_k128(st + O_DP + (2 * p + (ks >> 2)) * KB) + (uint64_t)((ks & 3) * 2), I_16,
                    ks > 0 ? 1u : 0u);
        tc_commit(&B.dxfull[w]);
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
    float bs1[16], bq[16], bsk[16], bsr[16];                               // d bsc1, d bq, d bs over this thread's rows (channels 8 ak + e, 64 + 8 ak + e), bs
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      bs1[e] = 0.f; bq[e] = 0.f; bsk[e] = 0.f;
      bsr[e] = CROSS ? __ldg(BS + 64 * (e >> 3) + 8 * ak + (e & 7)) : 0.f;
    }
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      // ---- P1: dx1 = the bf16 sum of the four projections' input gradients -> row-major
      mbar_wait(&B.dxfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < 2; ++L) {
        uint32_t v[4][8], r[4], r2[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 16 * p, v[p]);
        tmem_wait_ld();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int k = 4 * (mi & 1) + 2 * (mi >> 1);
          uint32_t g[4];
#pragma unroll
          for (int p = 0; p < 4; ++p) g[p] = pack_bf16(__uint_as_float(v[p][k]), __uint_as_float(v[p][k + 1]));
          if (CROSS) {                                                     // dx1 = rn(dg Wg + dq Wq); dxkv = rn(dk Wk + dv Wv)
            r[mi] = badd2(g[3], g[0]);
            r2[mi] = pack_bf16(__uint_as_float(v[1][k]) + __uint_as_float(v[2][k]), __uint_as_float(v[1][k + 1]) + __uint_as_float(v[2][k + 1]));
          } else {
            // autograd's accumulation order of x1's four gradients (q 0, k 1, v 2, g 3)
            r[mi] = ORD == 0 ? badd2(badd2(badd2(g[3], g[2]), g[1]), g[0]) : ORD == 1 ? badd2(badd2(badd2(g[0], g[1]), g[2]), g[3])
                  : ORD == 2 ? badd2(badd2(badd2(g[2], g[1]), g[0]), g[3]) : badd2(badd2(badd2(g[3], g[0]), g[1]), g[2]);
          }
        }
        stsm4t(faddr(st + O_X, lb + 16 * L + 8 * fh), r);
        if (CROSS) stsm4t(faddr(st + O_DXKV, lb + 16 * L + 8 * fh), r2);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxfree[w]);
      named_bar_sync(1 + w, 128);
      // ---- P2: row stage
      uint32_t kvo[2][4];                                                  // dx1_kv (bf16), CROSS only
      if (CROSS) {
        float y[16], dy[16], dxn[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 yu = lds128(st + (PARALLEL ? O_A : O_X1) + kb * KB + aoff), du = lds128(st + O_DXKV + kb * KB + aoff);
          const uint32_t y4[4] = {yu.x, yu.y, yu.z, yu.w}, d4[4] = {du.x, du.y, du.z, du.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            y[8 * kb + 2 * e] = bf16lo(y4[e]); y[8 * kb + 2 * e + 1] = bf16hi(y4[e]);
            dy[8 * kb + 2 * e] = bf16lo(d4[e]); dy[8 * kb + 2 * e + 1] = bf16hi(d4[e]);
          }
        }
        float sum = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) sum += y[e];
        sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
        const float mean = RMS ? 0.f : sum * (1.f / 128.f);
        float var = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) { y[e] -= mean; var += y[e] * y[e]; }
        var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
        const float rstd = rsqrtf(var * (1.f / 128.f) + eps);
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 pu = lds128(st + O_MP + kb * KB + aoff);
          const uint32_t p4[4] = {pu.x, pu.y, pu.z, pu.w};
          uint32_t dp[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float d2[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const int k = 8 * kb + 2 * e + j;
              y[k] *= rstd;                                                 // LN(x1)
              const float sk = sigm_x((j ? bf16hi(p4[e]) : bf16lo(p4[e])) + bsr[k]);
              d2[j] = RMS ? dy[k] * y[k] : dy[k] * y[k] * sk * (1.f - sk);  // d pre-sigmoid scale (RMS: of the raw one)
              dxn[k] = dy[k] * sk;
              bsk[k] += d2[j];
              s1 += dxn[k];
              s2 += dxn[k] * y[k];
            }
            dp[e] = pack_bf16(d2[0], d2[1]);
          }
          sts128(st + O_MP + kb * KB + aoff, make_uint4(dp[0], dp[1], dp[2], dp[3]));   // dpre over the staged pre
        }
        s1 += __shfl_xor_sync(~0u, s1, 1); s1 += __shfl_xor_sync(~0u, s1, 2); s1 += __shfl_xor_sync(~0u, s1, 4);
        s2 += __shfl_xor_sync(~0u, s2, 1); s2 += __shfl_xor_sync(~0u, s2, 2); s2 += __shfl_xor_sync(~0u, s2, 4);
        const float m1 = RMS ? 0.f : s1 * (1.f / 128.f), m2 = s2 * (1.f / 128.f);
#pragma unroll
        for (int kb = 0; kb < 2; ++kb)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            kvo[kb][e] = pack_bf16(rstd * (dxn[k] - m1 - y[k] * m2), rstd * (dxn[k + 1] - m1 - y[k + 1] * m2));
          }
      }
      {
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 a = lds128(st + O_A + kb * KB + aoff), dq = lds128(st + O_DP + kb * KB + aoff);
          const uint32_t w4[4] = {a.x, a.y, a.z, a.w}, q4[4] = {dq.x, dq.y, dq.z, dq.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            x[8 * kb + 2 * e] = bf16lo(w4[e]); x[8 * kb + 2 * e + 1] = bf16hi(w4[e]);
            bq[8 * kb + 2 * e] += bf16lo(q4[e]); bq[8 * kb + 2 * e + 1] += bf16hi(q4[e]);
          }
        }
        float sum = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) sum += x[e];
        sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
        const float mean = RMS ? 0.f : sum * (1.f / 128.f);
        float var = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) { x[e] -= mean; var += x[e] * x[e]; }
        var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
        const float rstd = rsqrtf(var * (1.f / 128.f) + eps);
        float dxn[16], s1 = 0.f, s2 = 0.f;
        uint32_t dsc[8];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          uint4 dxu = lds128(st + O_X + kb * KB + aoff);
          const uint4 gu = lds128(st + O_S1 + kb * KB + aoff);
          if (CROSS && !PARALLEL) {                                      // dx1 += dx1_kv (bf16), and dbi1 = the sum
            dxu.x = badd2(dxu.x, kvo[kb][0]); dxu.y = badd2(dxu.y, kvo[kb][1]); dxu.z = badd2(dxu.z, kvo[kb][2]); dxu.w = badd2(dxu.w, kvo[kb][3]);
            sts128(st + O_X + kb * KB + aoff, dxu);
          }
          const uint32_t d4[4] = {dxu.x, dxu.y, dxu.z, dxu.w}, g4[4] = {gu.x, gu.y, gu.z, gu.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            x[k] *= rstd; x[k + 1] *= rstd;
            dsc[4 * kb + e] = RMS ? bmul2(d4[e], pack_bf16(x[k], x[k + 1])) : sig_bwd2(bmul2(d4[e], pack_bf16(x[k], x[k + 1])), g4[e]);
            const uint32_t dn = bmul2(d4[e], g4[e]);
            dxn[k] = bf16lo(dn); dxn[k + 1] = bf16hi(dn);
            s1 += dxn[k] + dxn[k + 1];
            s2 += dxn[k] * x[k] + dxn[k + 1] * x[k + 1];
            bs1[k] += bf16lo(dsc[4 * kb + e]); bs1[k + 1] += bf16hi(dsc[4 * kb + e]);
          }
        }
        s1 += __shfl_xor_sync(~0u, s1, 1); s1 += __shfl_xor_sync(~0u, s1, 2); s1 += __shfl_xor_sync(~0u, s1, 4);
        s2 += __shfl_xor_sync(~0u, s2, 1); s2 += __shfl_xor_sync(~0u, s2, 2); s2 += __shfl_xor_sync(~0u, s2, 4);
        const float m1 = RMS ? 0.f : s1 * (1.f / 128.f), m2 = s2 * (1.f / 128.f);
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 dau = lds128(st + O_DA + kb * KB + aoff);
          const uint32_t y4[4] = {dau.x, dau.y, dau.z, dau.w};
          uint32_t ds[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            ds[e] = badd2(y4[e], pack_bf16(rstd * (dxn[k] - m1 - x[k] * m2), rstd * (dxn[k + 1] - m1 - x[k + 1] * m2)));
            if (CROSS && PARALLEL) ds[e] = badd2(ds[e], kvo[kb][e]);      // + the K / V AdaLN's LN backward (of a as well)
          }
          sts128(st + O_DA + kb * KB + aoff, make_uint4(ds[0], ds[1], ds[2], ds[3]));
          sts128(st + O_S1 + kb * KB + aoff, make_uint4(dsc[4 * kb], dsc[4 * kb + 1], dsc[4 * kb + 2], dsc[4 * kb + 3]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.stfull[w][i % NSW]);                      // the store thread of this warpgroup takes it from here
    }
    float* red = reinterpret_cast<float*>(sm + O_RED);                      // [8 warps][NRED]
    const int cw = warp - 4;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      float v1 = bs1[e], vq = bq[e], vk = bsk[e];
      v1 += __shfl_xor_sync(~0u, v1, 8); v1 += __shfl_xor_sync(~0u, v1, 16);
      vq += __shfl_xor_sync(~0u, vq, 8); vq += __shfl_xor_sync(~0u, vq, 16);
      vk += __shfl_xor_sync(~0u, vk, 8); vk += __shfl_xor_sync(~0u, vk, 16);
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      if ((lane >> 3) == 0) {
        red[cw * NRED + c] = v1; red[cw * NRED + 128 + c] = vq;
        if (CROSS) red[cw * NRED + 256 + c] = vk;
      }
    }
    named_bar_sync(3, 256);
    for (int c = tid - 128; c < NRED; c += 256) {
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) v += red[k * NRED + c];
      atomicAdd(c < 128 ? DBIAS + c : (c < 256 ? DBQ + (c - 128) : DBS + (c - 256)), v);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

#endif // ATOM_HOPPER
