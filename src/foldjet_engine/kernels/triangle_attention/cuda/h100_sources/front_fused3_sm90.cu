// (copied from the h100/cuda-sweep b200style path; the bias is written as bf16(bias * inv_scale), masked keys = bf16 lowest)
// Hopper front stage v3: LayerNorm + [Wq;Wk;Wv;Wg;Wb] projection in one kernel.
// v2 re-streamed the whole weight matrix from L2 for every 64-row tile, which made
// L2 bandwidth the limit at C>=256. Here one CTA owns 128 rows: two warpgroups
// (64 rows each) walk the same 128-column weight blocks through one shared TMA ring,
// so each weight byte is fetched once per 128 rows. LN is computed once per tile into
// shared memory; Q/K/V/G leave through TMA stores, the head bias is transposed in
// shared memory so each head writes contiguous keys.
// HD = 32 H (heads padded to 32 channels) or 16 H (16-channel heads kept compact for the fp16 core's 3-D loads).  ``flip`` (bit 1 of tr):
// the ending node on AlphaFold 3's bias orientation -- the bias stored as the rows have it with the key mask on the transposed axis (each
// masked row whole), which the fp16 core reads transposed (bias_transposed; what h100_sm90.flip_bias did as three more kernels).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "fa3_utils.h"
#include <cuda_fp16.h>
#ifdef MW_F3_VTW
#include <cuda_fp8.h>
// MW_F3_VTW (1 = hi plane, 2 = hi + lo; opt-in FP8 PV route, default builds do not define it): V leaves the front as the FP8 core's e4m3 planes of V^T
// ([B, N, H, planes, dh, S], the keys of each group of 32 permuted by vt_pos), made from the fp16-rounded V by vt_quant_k's arithmetic (static per-head
// scale: f * vinv, cvt.rn.satfinite; the lo plane is the x16 residual).  Values past 448 saturate and set the int flag `ovf`.  The V plane of `p` is not written.
__device__ __forceinline__ int vt_pos(int k) {   // position (inside a group of 32) of the key k: the PV wgmma's k32 A-fragment order
  const int blk = k >> 3, rem = k & 7;
  return 16 * (blk >> 1) + 4 * (rem >> 1) + 2 * (blk & 1) + (rem & 1);
}
#endif
using namespace cute;
#ifdef MW_F3_Y16
// Y16: the fp16 inference route of the width-generic path, bit for bit: LN as ln_fwd_k (lanes per row, fmaf), fp16 y and weights,
// fp16 q / k / v / g and fp16 bias (rounded before the scale) as the cuBLAS fp16 GEMMs + front_bias_k make them
using Element=cutlass::half_t;
#else
using Element=cutlass::bfloat16_t;
#endif
using XE=cutlass::bfloat16_t;   // the pair x is bf16 in both
#ifdef MW_F3_Y16
#define SCL 1.0f
#define WSCALAR torch::kHalf
#else
#define SCL inv_scale
#define WSCALAR torch::kBFloat16
#endif
#ifndef MW_F3_PF
#define MW_F3_PF 4   // Y16 LayerNorm: rows per thread whose global loads are issued together
#endif
#ifndef MW_F3_PF2
#define MW_F3_PF2 2   // bf16 LayerNorm: rows per thread whose global loads are issued together
#endif
#ifndef MW_F3_STAGES
#define MW_F3_STAGES 4
#endif
constexpr int BN=128;
#ifndef MW_F3_SLOTS
#define MW_F3_SLOTS 1
#endif
constexpr int SLOTS=MW_F3_SLOTS;
// d_pair 64: one k tile, three weight blocks; two stages (a smaller footprint) were 2-4 % faster at L128-1024 (same bits)
constexpr int stages_for(int C){return C<=64?2:MW_F3_STAGES;}
#ifdef MW_F3_MINB_FORCE
constexpr int minb_for(int){return MW_F3_MINB_FORCE;}
#else
constexpr int minb_for(int C){return C<=128?2:1;}
#endif
using TileLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<Int<BN>,_64>{}));
using GS=Shape<int32_t,int32_t>;
using ST=Stride<int64_t,_1>;
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},ST{}),WL{},Shape<Int<BN>,_64>{},_1{}));
using OS=Shape<int32_t,int32_t,int32_t>;
using OStride=Stride<int64_t,_1,int64_t>;
using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),OS{},OStride{}),TileLayout{},Shape<_64,_64>{},_1{}));

template<int C,int HD> struct Shared {
  array_aligned<Element,128*C,1024> z;                 // [wg][k tile][64 x 64]
  array_aligned<Element,BN*64,1024> w[stages_for(C)];  // shared weight ring
  array_aligned<Element,64*BN,1024> st[2*SLOTS];
  cutlass::arch::ClusterTransactionBarrier full[stages_for(C)];
  cutlass::arch::ClusterBarrier empty[stages_for(C)];
};

template<int C,int HD> __global__ __launch_bounds__(256,minb_for(C))
void front_v3(CUTE_GRID_CONSTANT TW const tw,CUTE_GRID_CONSTANT TO const to,
              XE const* x,float const* lnw,float const* lnb,bool const* mask,
              Element* bias,float* stats,int rows,int L,int H,float eps,float inv_scale,int tr,
              int h16,int crt
#ifdef MW_F3_VTW
              ,unsigned char* vt,float const* vinv,int* ovf,int dh
#endif
              ) {      // h16: q / k / v and the bias leave as fp16 bits (the fp16 inference core; the TMA store moves bytes), g as bf16
  constexpr int NK=C/64,COLS=4*HD+64,NB=(COLS+BN-1)/BN,STAGES=stages_for(C),total=NB*NK;
  extern __shared__ char memory[]; auto& s=*reinterpret_cast<Shared<C,HD>*>(memory);
  int tid=threadIdx.x,wgi=tid/128,wt=tid%128,base=blockIdx.x*128;
  if(tid==0) {
    for(int i=0;i<STAGES;++i){s.full[i].init(1);s.empty[i].init(2);}
    prefetch_tma_descriptor(tw.get_tma_descriptor());prefetch_tma_descriptor(to.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  auto wg=tw.get_tma_tensor(make_shape(COLS,C));
  auto issue=[&](int item) {
    int stage=item%STAGES;
    if(item>=STAGES) s.empty[stage].wait(((item/STAGES)-1)&1);
    s.full[stage].arrive_and_expect_tx(BN*64*sizeof(Element));
    auto ww=local_tile(wg,Shape<Int<BN>,_64>{},make_coord(item/NK,item%NK));
    auto sw=make_tensor(make_smem_ptr(s.w[stage].data()),WL{});
    auto t=tw.get_slice(_0{});
    copy(tw.with(reinterpret_cast<uint64_t&>(s.full[stage])),t.partition_S(ww),t.partition_D(sw));
  };
  if(tid==0) for(int i=0;i<STAGES && i<total;++i) issue(i);

  // ---- LayerNorm: T lanes per row, 16-byte chunks interleaved across the lanes ----
  auto al=as_position_independent_swizzle_layout(TileLayout{});
#ifdef MW_F3_Y16
  {
    // ln_fwd_k's arithmetic per row: LPR lanes a row, lane `sub` takes 16-byte vectors sub, sub+LPR, ..; sums in vector then element
    // order, xor tree LPR/2 .. 1, /C by a run-time C (fast-math division as the glue's), fmaf affine, cvt.rn.f16x2
    constexpr int LPR=C<=64?8:(C<=128?16:32),NV=C/8,NA=(NV+LPR-1)/LPR,RP=256/LPR;
    int sub=tid%LPR,rg=tid/LPR;
    constexpr int RPT=128/RP,G=RPT<MW_F3_PF?RPT:MW_F3_PF;   // rows a thread normalises, rows whose loads are in flight together
    for(int r0=0;r0<RPT;r0+=G) {
    uint4 pre[G][NA];
    #pragma unroll
    for(int gi=0;gi<G;++gi) {
      int64_t row=base+rg+RP*(r0+gi),prow=row;
      if(tr&1){int64_t ll=int64_t(L)*L,bb=row/ll,rem=row-bb*ll,ii=rem/L,jj=rem-ii*L;prow=bb*ll+jj*L+ii;}
      #pragma unroll
      for(int a=0;a<NA;++a) if(sub+LPR*a<NV) pre[gi][a]=*reinterpret_cast<uint4 const*>(x+prow*C+(sub+LPR*a)*8);
    }
    #pragma unroll
    for(int gi=0;gi<G;++gi) {
      const int r=rg+RP*(r0+gi);
      float v[NA][8],sm=0.f;
      #pragma unroll
      for(int a=0;a<NA;++a) {
        int idx=sub+LPR*a;
        if(idx<NV) {
          uint4 vv=pre[gi][a];
          uint32_t w4[4]={vv.x,vv.y,vv.z,vv.w};
          #pragma unroll
          for(int e=0;e<4;++e){v[a][2*e]=__uint_as_float(w4[e]<<16);v[a][2*e+1]=__uint_as_float(w4[e]&0xffff0000u);}
          #pragma unroll
          for(int e=0;e<8;++e) sm+=v[a][e];
        }
      }
      #pragma unroll
      for(int o=LPR/2;o>0;o>>=1) sm+=__shfl_xor_sync(0xffffffffu,sm,o);
      const float mean=sm/crt;
      float q=0.f;
      #pragma unroll
      for(int a=0;a<NA;++a) if(sub+LPR*a<NV) {
        #pragma unroll
        for(int e=0;e<8;++e){const float d=v[a][e]-mean;q=fmaf(d,d,q);}
      }
      #pragma unroll
      for(int o=LPR/2;o>0;o>>=1) q+=__shfl_xor_sync(0xffffffffu,q,o);
      const float rstd=rsqrtf(q/crt+eps);
      int rw=r/64,rr=r%64;
      #pragma unroll
      for(int a=0;a<NA;++a) {
        int idx=sub+LPR*a;
        if(idx<NV) {
          int col=idx*8;float o[8];
          #pragma unroll
          for(int e=0;e<8;++e) o[e]=fmaf((v[a][e]-mean)*rstd,__ldg(lnw+col+e),__ldg(lnb+col+e));
          uint4 vec;uint32_t* pw=reinterpret_cast<uint32_t*>(&vec);
          #pragma unroll
          for(int e=0;e<4;++e) asm("cvt.rn.f16x2.f32 %0, %1, %2;":"=r"(pw[e]):"f"(o[2*e+1]),"f"(o[2*e]));
          *reinterpret_cast<uint4*>(s.z.data()+(rw*NK+col/64)*4096+al(make_coord(rr,col%64)))=vec;
        }
      }
    }
    }
  }
#else
  {
    constexpr int T=C<256?4:8,RP=256/T,CH=C/8,PER=CH/T;
    int sub=tid%T,rg=tid/T;
    // the rows' global loads are issued MW_F3_PF2 rows at a time (a thread had 2 loads in flight per row; the HBM latency then dominated)
    constexpr int RPT=128/RP,G=RPT<MW_F3_PF2?RPT:MW_F3_PF2;
#ifdef MW_F3_ABL_NOLN
    for(int r0=0;r0<0;r0+=G) {
#else
    for(int r0=0;r0<RPT;r0+=G) {
#endif
    uint4 pre[G][PER];
    #pragma unroll
    for(int gi=0;gi<G;++gi) {
      int64_t row=base+rg+RP*(r0+gi),prow=row;
      if(tr&1){int64_t ll=int64_t(L)*L,bb=row/ll,rem=row-bb*ll,ii=rem/L,jj=rem-ii*L;prow=bb*ll+jj*L+ii;}   // ending node: physical row (b, j, i)
      #pragma unroll
      for(int i=0;i<PER;++i) pre[gi][i]=*reinterpret_cast<uint4 const*>(x+prow*C+(sub+T*i)*8);
    }
    #pragma unroll
    for(int gi=0;gi<G;++gi) {
      const int r=rg+RP*(r0+gi);int64_t row=base+r;float vals[PER*8],mean=0.f,var=0.f;
      #pragma unroll
      for(int i=0;i<PER;++i) {
        uint4 vv=pre[gi][i];
        auto const* v=reinterpret_cast<Element const*>(&vv);
        #pragma unroll
        for(int e=0;e<8;++e){vals[i*8+e]=float(v[e]);mean+=vals[i*8+e];}
      }
      #pragma unroll
      for(int d=1;d<T;d<<=1) mean+=__shfl_xor_sync(0xffffffff,mean,d);
      mean/=C;
      #pragma unroll
      for(int i=0;i<PER*8;++i){float d=vals[i]-mean;var+=d*d;}
      #pragma unroll
      for(int d=1;d<T;d<<=1) var+=__shfl_xor_sync(0xffffffff,var,d);
      float inv=rsqrtf(var/C+eps);
      if(sub==0 && stats){stats[row*2]=mean;stats[row*2+1]=inv;}
      int rw=r/64,rr=r%64;
      #pragma unroll
      for(int i=0;i<PER;++i) {
        int col=(sub+T*i)*8;uint4 vec;auto* o=reinterpret_cast<Element*>(&vec);
        #pragma unroll
        for(int e=0;e<8;++e) o[e]=Element((vals[i*8+e]-mean)*inv*__ldg(lnw+col+e)+__ldg(lnb+col+e));
        *reinterpret_cast<uint4*>(s.z.data()+(rw*NK+col/64)*4096+al(make_coord(rr,col%64)))=vec;
      }
    }
    }
  }
#endif
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  // ---- projection: both warpgroups consume every ring stage ----
  using TileMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<BN>,_64>>()));
  TileMMA mma;auto thr=mma.get_slice(wt);
  auto acc=partition_fragment_C(mma,Shape<_64,Int<BN>>{});
  auto coord=thr.partition_C(make_identity_tensor(Shape<_64,Int<BN>>{}));
  auto og=to.get_tma_tensor(make_shape(rows,HD,4));
  auto tile=to.get_slice(_0{});
  Element* stg=s.st[wgi*SLOTS].data();
  bool leader=wt==0;
  int it=0;
  for(int blk=0;blk<NB;++blk) {
    clear(acc);
    for(int kc=0;kc<NK;++kc,++it) {
      int stage=it%STAGES;
      s.full[stage].wait((it/STAGES)&1);
      auto aa=make_tensor(make_smem_ptr(s.z.data()+(wgi*NK+kc)*4096),TileLayout{});
      auto ww=make_tensor(make_smem_ptr(s.w[stage].data()),WL{});
      auto ra=thr.partition_fragment_A(aa);auto rb=thr.partition_fragment_B(ww);
#ifndef MW_F3_ABL_NOMMA
      flash::gemm<false,1>(mma,ra,rb,acc);
#endif
      // Item it-1's WGMMA has retired in every warp of this warpgroup: release its stage.
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
      if(leader && it>0) s.empty[(it-1)%STAGES].arrive(uint32_t(0),uint32_t(1));
      if(tid==0 && it>0 && it-1+STAGES<total) issue(it-1+STAGES);
    }
    warpgroup_wait<0>();warpgroup_fence_operand(acc);
    int col0=blk*BN;
    stg=s.st[wgi*SLOTS+blk%SLOTS].data();
    // The previous block's TMA store has had a whole block of MMA time to drain.
#ifndef MW_F3_ABL_NOSTOREWAIT
    if(leader) tma_store_wait<SLOTS-1>();
#endif
    cutlass::arch::NamedBarrier::sync(128,1+wgi);
#ifdef MW_F3_ABL_NOEPI
    if(false) {
#else
    if(col0<4*HD) {
#endif
#ifdef MW_F3_VTW
      unsigned char* vs8=reinterpret_cast<unsigned char*>(stg);   // V^T staging [plane][channel][64 keys], 64-byte rows, words XOR-swizzled by 4 * ((channel / 2) mod 4)
      bool vovf=false;
      const bool vblk=(col0/HD)==2;   // this block holds the V columns (they start the block: 2 HD is a multiple of 128)
      if(vblk) {
        // The accumulator holds rows (16 wi + g8, + 8) x columns 8 i + 2 t4 + {0, 1}.  Key position p = 16 wi + 4 (g8 / 2) + 2 hf + (g8 & 1): the 4 keys of word 4 wi + g8 / 2
        // are the rows of the lane pair (g8 even, odd) at both halves; the pair swaps bytes (one shuffle) so each lane holds one whole word of one channel (lane parity picks
        // channel 2 t4 + parity), a single 32-bit store.
        const int lane=wt&31,wi=wt>>5,g8=lane>>2,t4=lane&3,odd=g8&1,pw=4*wi+(g8>>1);
        const uint32_t sel=odd?0x3715u:0x6240u;
        #pragma unroll
        for(int i=0;i<HD/8;++i) {
          const int chA=8*i+2*t4;
          const float inv=__ldg(vinv+(dh==32?chA>>5:chA>>4));
          uint32_t S=0,S2=0;
          float xs[2][2];
          #pragma unroll
          for(int hf=0;hf<2;++hf) {
            const float2 f=__half22float2(__halves2half2(__float2half_rn(acc(4*i+2*hf)),__float2half_rn(acc(4*i+2*hf+1))));
            xs[hf][0]=f.x*inv;xs[hf][1]=f.y*inv;
            vovf|=(fabsf(xs[hf][0])>448.f)|(fabsf(xs[hf][1])>448.f);
            const __nv_fp8x2_storage_t qv=__nv_cvt_float2_to_fp8x2(make_float2(xs[hf][0],xs[hf][1]),__NV_SATFINITE,__NV_E4M3);
            S|=uint32_t(qv)<<(16*hf);
          }
          if(MW_F3_VTW==2) {
            #pragma unroll
            for(int hf=0;hf<2;++hf) {
              const unsigned short q16=(unsigned short)((S>>(16*hf))&0xffffu);
              uint32_t hu;asm("cvt.rn.f16x2.e4m3x2 %0, %1;":"=r"(hu):"h"(q16));
              const float h0=__half2float(__ushort_as_half((unsigned short)(hu&0xffffu))),h1=__half2float(__ushort_as_half((unsigned short)(hu>>16)));
              unsigned short r2;asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;":"=h"(r2):"f"((xs[hf][1]-h1)*16.f),"f"((xs[hf][0]-h0)*16.f));
              S2|=uint32_t(r2)<<(16*hf);
            }
          }
          const int ch=chA+odd,sw=4*t4;
          const uint32_t P=__shfl_xor_sync(0xffffffffu,S,4);
          *reinterpret_cast<uint32_t*>(vs8+ch*64+(((pw)^sw)<<2))=__byte_perm(S,P,sel);
          if(MW_F3_VTW==2) {
            const uint32_t P2=__shfl_xor_sync(0xffffffffu,S2,4);
            *reinterpret_cast<uint32_t*>(vs8+HD*64+ch*64+(((pw)^sw)<<2))=__byte_perm(S2,P2,sel);
          }
        }
      }
#endif
      #pragma unroll
      for(int j=0;j<size(acc);j+=2) {
        int rr=get<0>(coord(j)),cc=get<1>(coord(j));
        uint32_t word;
#ifdef MW_F3_VTW
        if((col0+cc)/HD==2) continue;   // V: made below as V^T planes
#endif
#ifdef MW_F3_Y16
        if(true) word=uint32_t(__half_as_ushort(__float2half_rn(acc(j))))|(uint32_t(__half_as_ushort(__float2half_rn(acc(j+1))))<<16);
#else
        if(h16 && (col0+cc)/HD<3) word=uint32_t(__half_as_ushort(__float2half_rn(acc(j))))|(uint32_t(__half_as_ushort(__float2half_rn(acc(j+1))))<<16);
#endif
        else {Element lo=Element(acc(j)),hi=Element(acc(j+1));word=uint32_t(lo.raw())|(uint32_t(hi.raw())<<16);}
        *reinterpret_cast<uint32_t*>(stg+(cc/64)*4096+al(make_coord(rr,cc%64)))=word;
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
#ifdef MW_F3_VTW
      if(vblk) {
        // this warpgroup's 64 keys of V^T: 16-byte chunks, 64 contiguous keys per (head, plane, channel)
        if(__any_sync(0xffffffffu,vovf) && (wt&31)==0) atomicOr(ovf,1);
        const int64_t row0=base+wgi*64,ll=int64_t(L)*L,bq=row0/ll,iq=(row0/L)%L,j0=row0%L;
        for(int u=wt;u<HD*4*MW_F3_VTW;u+=128) {
          const int pl=u/(HD*4),ch=(u>>2)%HD,c16=u&3;
          const uint4 val=*reinterpret_cast<const uint4*>(vs8+pl*HD*64+ch*64+(((4*c16)^(4*((ch>>1)&3)))<<2));
          const int64_t orow=(((((bq*L+iq)*H+ch/dh)*MW_F3_VTW+pl)*dh+ch%dh)*L)+j0+16*c16;
          *reinterpret_cast<uint4*>(vt+orow)=val;
        }
      }
#endif
      if(leader) {
        #pragma unroll
        for(int chunk=0;chunk<BN/64;++chunk) {
          int col=col0+chunk*64;
#ifdef MW_F3_VTW
          if(col/HD==2) continue;   // V left as V^T planes
#endif
          auto dst=local_tile(og(_,_,col/HD),Shape<_64,_64>{},make_coord(base/64+wgi,(col%HD)/64));
          auto src=make_tensor(make_smem_ptr(stg+chunk*4096),TileLayout{});
          copy(to,tile.partition_S(src),tile.partition_D(dst));
        }
        tma_store_arrive();
      }
    } else {
#ifdef MW_F3_ABL_NOEPI
      if(false)
#endif
      {
      // Head bias: stage [head][64 keys], then each head writes its 64 contiguous keys.
      #pragma unroll
      for(int j=0;j<size(acc);j+=2) {
        int rr=get<0>(coord(j)),cc=get<1>(coord(j));
        auto put=[&](int c,float v){
#ifdef MW_F3_Y16
          // the bias GEMM's fp16 output, then front_bias_k: float(half) * inv_scale, clamped, rounded
          if(h16) reinterpret_cast<uint16_t*>(stg)[c*64+rr]=__half_as_ushort(__float2half_rn(fmaxf(fminf(__half2float(__float2half_rn(v))*inv_scale,6.0e4f),-6.0e4f)));
#else
          if(h16) reinterpret_cast<uint16_t*>(stg)[c*64+rr]=__half_as_ushort(__float2half_rn(fmaxf(fminf(v,6.0e4f),-6.0e4f)));
#endif
          else stg[c*64+rr]=Element(v);
        };
        if(cc<H) put(cc,acc(j)*SCL);
        if(cc+1<H) put(cc+1,acc(j+1)*SCL);
      }
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
      if(wt<H*8) {
        int h=wt/8,c=wt%8;
        int64_t row0=base+wgi*64;int b=row0/(int64_t(L)*L),q=(row0/L)%L,key0=row0%L;
        uint4 vec=*reinterpret_cast<uint4 const*>(stg+h*64+c*8);
        auto* v=reinterpret_cast<Element*>(&vec);
        if(tr&2) {
          // flip (the ending node on AlphaFold 3's bias orientation): token (q, key) is the transpose's element, bias^T[b, h, key, q]. The
          // bias is stored as the rows have it (the core reads its tiles transposed: triattn_fwd_h16(bias_transposed=True)), with the
          // key mask on the transposed axis: row q masked whole. (Writing the transpose here took 16-bit stores at stride L: AF3's ending
          // node was 1.3 us slower than the standard one at L128.)
          if(mask && !mask[b*L+q]) {
            #pragma unroll
            for(int e=0;e<8;++e) reinterpret_cast<uint16_t*>(v)[e]=h16?uint16_t(0xFC00u):uint16_t(0xFF80u);   // -inf (fp16 / bf16)
          }
        } else if(mask) {
          #pragma unroll
          for(int e=0;e<8;++e) if(!mask[b*L+key0+c*8+e]) {
            if(h16) reinterpret_cast<uint16_t*>(v)[e]=0xFC00u;    // fp16 -inf
            else v[e]=Element(-3.3895313892515355e38f);
          }
        }
        *reinterpret_cast<uint4*>(bias+((int64_t(b)*H+h)*L+q)*L+key0+c*8)=vec;
      }
      }
    }
  }
  if(leader) tma_store_wait<0>();
}

#ifdef MW_F3_VTW
struct VtArgs{unsigned char* vt;float const* vinv;int* ovf;int dh;};
static VtArgs g_vt;
#endif
template<int C,int HD> void launch(torch::Tensor x,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor weights,torch::Tensor mask,
                            torch::Tensor p,torch::Tensor bias,torch::Tensor stats,int L,int H,float eps,float inv_scale,int tr,int h16) {
  int rows=x.size(0);
  auto gw=make_tensor(make_gmem_ptr((Element const*)weights.data_ptr()),make_shape(4*HD+64,C),ST{C,_1{}});
  auto tw=make_tma_copy(SM90_TMA_LOAD{},gw,WL{},Shape<Int<BN>,_64>{},_1{});
  auto go=make_tensor(make_gmem_ptr((Element*)p.data_ptr()),make_shape(rows,HD,4),OStride{HD,_1{},int64_t(rows)*HD});
  auto to=make_tma_copy(SM90_TMA_STORE{},go,TileLayout{},Shape<_64,_64>{},_1{});
  size_t smem=sizeof(Shared<C,HD>);
  C10_CUDA_CHECK(cudaFuncSetAttribute(front_v3<C,HD>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  front_v3<C,HD><<<rows/128,256,smem,at::cuda::getCurrentCUDAStream()>>>(tw,to,(XE*)x.data_ptr(),lnw.data_ptr<float>(),lnb.data_ptr<float>(),
    mask.numel()?mask.data_ptr<bool>():nullptr,(Element*)bias.data_ptr(),stats.numel()?stats.data_ptr<float>():nullptr,rows,L,H,eps,inv_scale,tr,h16,C
#ifdef MW_F3_VTW
    ,g_vt.vt,g_vt.vinv,g_vt.ovf,g_vt.dh
#endif
    );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> front_forward(torch::Tensor x,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor weights,torch::Tensor mask,int L,int H,int HD,double eps,double inv_scale,bool trans,bool h16,bool flip
#ifdef MW_F3_VTW
    ,c10::optional<torch::Tensor> vinv,c10::optional<torch::Tensor> ovf
#endif
    ) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.dim()==2,"contiguous BF16 matrix required");
  c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0),B=rows/(L*L);
  TORCH_CHECK(L>0 && L%128==0 && rows==B*L*L && (HD==32*H || HD==16*H) && H<=16 && HD%64==0,"invalid pair/head dimensions");
  TORCH_CHECK(weights.device()==x.device() && weights.scalar_type()==WSCALAR && weights.is_contiguous() && weights.sizes()==torch::IntArrayRef({4*HD+64,C}),"packed weights mismatch");
  for(auto t:{lnw,lnb}) TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==C,"FP32 LN affines required");
  TORCH_CHECK(mask.device()==x.device() && mask.is_contiguous() && mask.scalar_type()==torch::kBool && (mask.numel()==0 || mask.sizes()==torch::IntArrayRef({B,L})),"mask mismatch");
  // h16: planes 0-2 of p and the bias hold fp16 bits (the caller views them as fp16), plane 3 (g) bf16
  auto p=torch::empty({4,rows,HD},x.options()),bias=torch::empty({B,H,L,L},h16?x.options().dtype(torch::kHalf):x.options());
  auto stats=torch::empty({0,2},x.options().dtype(torch::kFloat32));
#ifdef MW_F3_VTW
  TORCH_CHECK(h16 && vinv.has_value() && ovf.has_value() && vinv->is_cuda() && vinv->scalar_type()==torch::kFloat32 && vinv->numel()==H && ovf->is_cuda() && ovf->scalar_type()==torch::kInt32,"VT planes: h16, vinv fp32 [H], ovf int32 flag");
  auto vt=torch::empty({B,L,H,MW_F3_VTW,HD/H,L},x.options().dtype(torch::kFloat8_e4m3fn));
  g_vt={reinterpret_cast<unsigned char*>(vt.data_ptr()),vinv->data_ptr<float>(),ovf->data_ptr<int>(),HD/H};
#endif
  const int key=C*1000+HD;
  switch(key){
    case 64064: launch<64,64>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    case 64128: launch<64,128>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    case 128128: launch<128,128>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    case 128256: launch<128,256>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    case 256256: launch<256,256>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    case 384384: launch<384,384>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    case 512512: launch<512,512>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps),float(inv_scale),int(trans)|(int(flip)<<1),int(h16));break;
    default: TORCH_CHECK(false,"unsupported (d_pair, heads x 32) pair");
  }
#ifdef MW_F3_VTW
  return {p,bias,stats,vt};
#else
  return {p,bias,stats};
#endif
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("front",&front_forward,py::arg("x"),py::arg("lnw"),py::arg("lnb"),py::arg("weights"),py::arg("mask"),py::arg("L"),py::arg("H"),py::arg("HD"),
        py::arg("eps"),py::arg("inv_scale"),py::arg("trans"),py::arg("h16")=false,py::arg("flip")=false
#ifdef MW_F3_VTW
        ,py::arg("vinv")=py::none(),py::arg("ovf")=py::none()
#endif
        );
}
