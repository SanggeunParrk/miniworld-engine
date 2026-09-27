#define WGRAD_SLICES 2
// SPDX-License-Identifier: Apache-2.0
// Training extension of Anthropic v5: two GLU/TMA producer WGs, one MMA WG.
// Width-parameterised (MW_C = c_z): C = 128 keeps the validated D = 128 tiling exactly (CNW == 1); wider C walks the channel
// axis in 128-wide chunks (CNW = C / 128), which keeps every accumulator, tile and register count at the D = 128 shape and
// pays one extra pass over the weight tiles per chunk. Every chunk's dx_n stays in registers (acc[CNW][32]) because the
// LayerNorm backward needs sums over ALL C channels; that is what lifts the occupancy hint to one CTA per SM, which the
// 224 KB of dynamic shared memory already forced. x is read twice per row tile (sums, then the output) instead.
#include "warp_primitives.cuh"
#ifndef MW_C
#define MW_C 128
#endif
#define MW_CNW (MW_C / 128)            // preprocessor twin of CNW: the guards below pick the C = 128 path or the wide one
constexpr int CDIM = MW_C;                 // pair width (channels)
constexpr int CNW = CDIM / 128;            // 128-channel chunks: 1 at C=128, 2 at 256, 3 at 384, 4 at 512
constexpr int HS = 2 * CDIM;               // per-side hidden width (gate | projection share it)
constexpr int HCH = HS / 64;               // 64-wide hidden chunks per side (4 at C=128)
constexpr int PREW = 4 * CDIM;             // preactivation columns per side (gate and projection interleaved)
constexpr int DWGROUPS = 2 * HCH * CNW;    // dW roles: side x hidden chunk x channel chunk (8 at C=128)
static_assert(CDIM % 128 == 0, "channel width in 128-channel chunks");
#ifndef DW_DEPTH                           // dW ring slots. Two leaves a single tile group in flight (~27 GB/s per SM at
#if MW_CNW > 1                             // the measured latency); the wide widths have the shared memory for more.
#define DW_DEPTH 3
#else
#define DW_DEPTH 2
#endif
#endif
#ifndef DW_BLOCKS
#define DW_BLOCKS 2
#endif
#ifndef DW_GLU_AHEAD                       // evaluating the next slot's GLU under the current wgmma measured WORSE
#define DW_GLU_AHEAD 0                     // (649 vs 568 us at C = 256): two CTAs per SM already overlap the two pipes.
#endif
#ifndef DW_CSPAN                           // channel chunks a dW CTA covers at once. Two share one preactivation tile and
#define DW_CSPAN 1                         // one GLU evaluation between them, at the cost of a second accumulator.
#endif
#ifndef DW_THREADS                         // 512 lets the four warpgroups be (gate|projection) x (two channel chunks), so
#define DW_THREADS 256                     // DW_CSPAN = 2 costs no occupancy: 16 warps per SM either way.
#endif
static_assert(DW_THREADS==256||DW_THREADS==512,"dW block is one or two warpgroup pairs");
static_assert(DW_THREADS==256||DW_CSPAN==2,"the 512-thread dW block exists to cover two channel chunks");
#ifndef M2_GLU_SEP                         // 1: the GLU derivative gets its own buffer (one barrier per call)
#define M2_GLU_SEP 1
#endif
#ifndef GW_DEPTH                           // dx weight ring slots; at C = 256 the shared memory buys either a third slot
#if MW_C == 256                            // or the GLU's own output buffer, not both.
#define GW_DEPTH (M2_GLU_SEP?2:3)
#else
#define GW_DEPTH 2
#endif
#endif
#ifndef MW_DBG_SPIN
#define MW_DBG_SPIN 2000000u
#endif
#ifndef MW_DBG                             // bounded waits: on a stall, record the site in p.counts instead of hanging
#define MW_DBG 0
#endif
#ifndef M2_SKIP                            // hang bisect only: bit 0 drops the gate contraction, bit 1 the LayerNorm
#define M2_SKIP 0
#endif
#ifndef M2_WGRING                          // 1: a weight ring per warpgroup; 0: one shared ring with four issuers
#define M2_WGRING 1
#endif
#ifndef M2_WG_SYNC                         // 1: warpgroup-scoped barrier before re-arming its own ring; 0: CTA-wide
#define M2_WG_SYNC 1
#endif
#ifndef MW_MROWS                           // 64-row tiles per dx CTA: 2 feeds one weight tile to 128 rows instead of 64,
#if MW_C == 256                            // which halves the dominant read. Wider than 256 the doubled accumulator spills.
#define MW_MROWS 2
#else
#define MW_MROWS 1
#endif
#endif
#ifndef DX_BLOCKS                          // two CTAs per SM for the wide dx role means fitting C/2 channels of
#define DX_BLOCKS 1                        // accumulator plus its working set into 128 registers; only C = 256 is close
#endif
#ifndef NISSUE                             // TMA issuers per tile group: one issuer thread saturates near 27 GB/s per SM,
#if MW_CNW > 1                             // which is exactly where the wide dW role sits. Each issuer arrives for its own
#define NISSUE 4                           // bytes, so the barrier counts NISSUE arrivals.
#else
#define NISSUE 1
#endif
#endif
#ifndef ROLE_ONLY                          // 1: dW role only, 2: dx role only. Timing ablation; the outputs are then partial.
#define ROLE_ONLY 0
#endif
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef DW_SPLITS
#if MW_C == 128
#define DW_SPLITS 8
#elif MW_C == 256
#define DW_SPLITS 2
#else
#define DW_SPLITS 1
#endif
#endif
#ifndef DW_CTAS                            // CTAs in the dW role; jobs (group x tile split) are handed out round robin
#if MW_C == 384
#define DW_CTAS 72
#else
#define DW_CTAS 64
#endif
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
#ifndef DW_PREFETCH
#define DW_PREFETCH 3
#endif
constexpr int DWCGROUPS=DWGROUPS/DW_CSPAN;             // one job covers DW_CSPAN consecutive channel chunks
constexpr int DWJOBS=DWCGROUPS*DW_SPLITS;
static_assert(CNW%DW_CSPAN==0,"channel chunks must divide evenly over the dW jobs");
#if ROLE_ONLY == 1                         // the wide widths launch the two roles separately: each then owns the whole grid,
constexpr int DWCOUNT=UCOUNT,DXCOUNT=1;    // and the dW kernel can run two CTAs per SM, which is where its latency hides.
#elif ROLE_ONLY == 2
constexpr int DWCOUNT=0,DXCOUNT=UCOUNT;
#else
constexpr int DWCOUNT=(CNW==1)?DWJOBS:DW_CTAS,DXCOUNT=UCOUNT-DWCOUNT;
#endif
constexpr int DW_XN=24576,DW_GLU=DW_XN+DW_CSPAN*16384;   // [pre 16384][d(sigma) 8192][xn per chunk][GLU gate|proj]
constexpr int DW_SLOT=DW_GLU+16384,DX_SLOT=40960;
constexpr int DW_SMEM=DW_DEPTH*DW_SLOT;
static_assert(DXCOUNT>0,"invalid partition");
// A hot `while(!mbar_try_wait(...)){}` at one CTA per SM was the source of an intermittent "unspecified launch failure"
// under CUDA-graph replay (about one in a hundred replays; never in eager launches, and never with any extra instruction
// in the loop). The suspend-time hint form parks the warp between checks instead of spinning on the barrier.
TMN_DEVI void mbar_wait_hint(uint64_t* bar,uint32_t phase){
 uint32_t ok;
 do{asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
                 :"=r"(ok):"r"(smem_u32(bar)),"r"(phase),"r"(1000u):"memory");}while(!ok);
}
#if MW_DBG
TMN_DEVI void dbg_wait(uint64_t* bar,uint32_t ph,unsigned int* counts,int site){
 unsigned n=0;while(!mbar_try_wait(bar,ph)){if(++n>MW_DBG_SPIN){if(threadIdx.x%128==0)atomicExch(counts+blockIdx.x*4+site,1u+threadIdx.x/128);return;}}
}
#define MW_WAIT(b,ph,site) dbg_wait((b),(ph),p.counts,(site))
#else
#define MW_WAIT(b,ph,site) mbar_wait_hint((b),(ph))
#endif
TMN_DEVI int issuer(){int w=threadIdx.x/32;return (threadIdx.x%32==0&&w<NISSUE)?w:-1;}
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;
};
TMN_DEVI void mma_weight64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

template<int NT=256>
TMN_DEVI void glu_small(const Params& p,uint8_t* s,uint8_t* gout,uint8_t* pout,int row){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll
 for(unsigned q=0;q<2048/NT;++q){unsigned i=tid+q*NT,c=i/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);
  uint32_t masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(gout+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(pout+swz128(c,r*2))=pp;
 }fence_proxy_async();named_bar_sync(0,NT);
}
TMN_DEVI void load_dw(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 int q=issuer();if(q<0)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/(HCH*CNW),rest=group%(HCH*CNW),h=(rest/CNW)*64,cs=rest%CNW;
#if NISSUE == 1
 mbar_arrive_expect_tx(b+slot,24576+DW_CSPAN*16384);
 tma_load_2d(s,&p.pre,b+slot,row,side*PREW+h*2);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h);
 for(int n=0;n<2*DW_CSPAN;++n)tma_load_2d(s+DW_XN+n*8192,&p.xn,b+slot,(cs+(n>>1))*128+(n&1)*64,row);
#else
 mbar_arrive_expect_tx(b+slot,q==0?16384:(q==1?8192:(DW_CSPAN*8192)));
 if(q==0)tma_load_2d(s,&p.pre,b+slot,row,side*PREW+h*2);
 else if(q==1)tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h);
 else if(q<4)for(int i=0;i<DW_CSPAN;++i){int n=(q-2)+2*i;tma_load_2d(s+DW_XN+n*8192,&p.xn,b+slot,(cs+i)*128+(q-2)*64,row);}
#endif
}

TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int wg=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,cbits=0;   // one phase bit per ring slot; it carries across jobs
#if DW_THREADS == 512
 int wi=wg&1,cspan0=wg>>1;                 // warpgroup = (gate | projection) x (channel chunk)
 constexpr int CPERWG=1;
#else
 int wi=wg,cspan0=0;
 constexpr int CPERWG=DW_CSPAN;
#endif
 for(int job=blockIdx.x;job<DWJOBS;job+=DWCOUNT){
 int group=(job%DWCGROUPS)*DW_CSPAN,split=job/DWCGROUPS;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[CPERWG][64]={};
 for(int i=0;i<DW_DEPTH;++i)if(split+i*DW_SPLITS<p.tiles)load_dw(p,sm,b,i,(split+i*DW_SPLITS)*64,group);   // every slot is filled before the first wait
#if DW_GLU_AHEAD
 // The GLU derivative costs about as much as the wgmma chain it feeds, so the next tile's GLU is evaluated while this
 // tile's wgmma is still in flight. Only the ring slot differs between the two, so nothing aliases.
 if(split<p.tiles){int ph=cbits&1;cbits^=1;mbar_wait(b,ph);glu_small(p,sm,sm+DW_GLU,sm+DW_GLU+8192,split*64);}
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r%DW_DEPTH;uint8_t* s=sm+slot*DW_SLOT;
  static_for<CPERWG>([&](auto ci){constexpr int c=decltype(ci)::value;fence_regs(acc[c]);});wgmma_fence();
  static_for<CPERWG>([&](auto ci){constexpr int c=decltype(ci)::value;
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_weight128(acc[c],smem_desc(smem_u32(s+DW_GLU+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+DW_XN+(cspan0+c)*16384+k*2048),8192,1024,1),r%segmentRounds>0||k>0);});
  });wgmma_commit();
  int next=tile+DW_SPLITS;
  if(next<p.tiles){int ns=(r+1)%DW_DEPTH;int ph=(cbits>>ns)&1;cbits^=1<<ns;
   mbar_wait(b+ns,ph);glu_small<DW_THREADS>(p,sm+ns*DW_SLOT,sm+ns*DW_SLOT+DW_GLU,sm+ns*DW_SLOT+DW_GLU+8192,next*64);}
  wgmma_wait<0>();
  static_for<CPERWG>([&](auto ci){constexpr int c=decltype(ci)::value;fence_regs(acc[c]);});named_bar_sync(0,DW_THREADS);
  if(tile+DW_DEPTH*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+DW_DEPTH*DW_SPLITS)*64,group);
#else
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r%DW_DEPTH;uint8_t* s=sm+slot*DW_SLOT;
  int ph=(cbits>>slot)&1;cbits^=1<<slot;MW_WAIT(b+slot,ph,slot);glu_small<DW_THREADS>(p,s,s+DW_GLU,s+DW_GLU+8192,tile*64);
  static_for<CPERWG>([&](auto ci){constexpr int c=decltype(ci)::value;fence_regs(acc[c]);});wgmma_fence();
  static_for<CPERWG>([&](auto ci){constexpr int c=decltype(ci)::value;
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_weight128(acc[c],smem_desc(smem_u32(s+DW_GLU+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+DW_XN+(cspan0+c)*16384+k*2048),8192,1024,1),r%segmentRounds>0||k>0);});
  });wgmma_commit();wgmma_wait<0>();
  static_for<CPERWG>([&](auto ci){constexpr int c=decltype(ci)::value;fence_regs(acc[c]);});named_bar_sync(0,DW_THREADS);
  if(tile+DW_DEPTH*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+DW_DEPTH*DW_SPLITS)*64,group);
#endif
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){
   static_for<CPERWG>([&](auto ci){constexpr int cc=decltype(ci)::value;
    float* out=p.partw+(((group+cspan0+cc)*DW_SPLITS+split)*2+r/segmentRounds)*16384+wi*8192;
    static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);
     stg64f(out+rr*128+c,acc[cc][q*4],acc[cc][q*4+1]);stg64f(out+(rr+8)*128+c,acc[cc][q*4+2],acc[cc][q*4+3]);});});}
 }
 if(job+DWCOUNT<DWJOBS){named_bar_sync(0,DW_THREADS);}              // the next job refills the same slots
 }
}
TMN_DEVI void load_g(const Params& p,uint8_t* sm,uint64_t* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,40960);
 tma_load_2d(s,&p.pre,b+slot,row,side*512+h*128);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h*64);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg,b+slot,h*64,n*64);
}
TMN_DEVI void load_p(const Params& p,uint8_t* sm,uint64_t* b,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b+slot,h*64,n*64);
}
TMN_DEVI void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate,bar,k*64,n*64);}
}
TMN_DEVI void load_ln_next(const Params& p,uint8_t* sm,uint64_t* bar,int row){if(threadIdx.x)return;mbar_arrive_expect_tx(bar,32768);for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,bar,c*64,row);}}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma){
 int split=blockIdx.x-DWCOUNT,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;float running=0;
 if(split<p.tiles)issue_gate(p,sm,bar+2,split*64);int round=0;for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;uint32_t gate_packed[16];float acc[32]={};
  // Per row tile: bar0 gate+2channel+LN transactions (4); bar1 2channels.
  // Both phase parities return to0 before advancing to the next row tile.
  mbar_wait(bar+2,round&1);
  {float gate[32]={};
  fence_regs(gate);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
  // B9 rounds separately before B10's accumulation/add.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;gate_packed[j]=pack_bf16(gate[j*2],gate[j*2+1]);});}

  allsync();
  if(threadIdx.x==0&&round>0)tma_store_wait_all();allsync();
  for(int side=0;side<2;++side){
   load_g(p,sm,bar,row,side,0);load_g(p,sm,bar,row,side,1);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,h/2);glu_small(p,s,s+16384,sm+81920+h*8192,row);
    fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+24576+wi*8192+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_g(p,sm,bar,row,side,h+2);
   }
   load_p(p,sm,bar,side,0);load_p(p,sm,bar,side,1);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+wi*8192+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_p(p,sm,bar,side,h+2);if(side==1&&h==1)load_ln_next(p,sm+65536,bar+3,row);
   }
  }
  if(tile+DXCOUNT<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);
  // B10 outputs BF16 dx_n. Keep it in registers through B11/B12.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;acc[j*2]=__bfloat162float(__float2bfloat16_rn(acc[j*2]+bf16lo(gate_packed[j])));acc[j*2+1]=__bfloat162float(__float2bfloat16_rn(acc[j*2+1]+bf16hi(gate_packed[j])));});
  uint8_t* lnsm=sm+65536;mbar_wait(bar+3,round&1);
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1),r=rr?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,r,c),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,r,c+1),mu[rr]),rs[rr]);
    float ha=acc[q*8+j*2]*gamma[wi*64+c],hb=acc[q*8+j*2+1]*gamma[wi*64+c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
#if DEBUG_SAVE
    p.debugxn[(size_t)(row+r)*128+wi*64+c]=__float2bfloat16_rn(acc[q*8+j*2]);p.debugxn[(size_t)(row+r)*128+wi*64+c+1]=__float2bfloat16_rn(acc[q*8+j*2+1]);
#endif
   });
  });
  s1[0]=quad_sum(s1[0])/128.f;s1[1]=quad_sum(s1[1])/128.f;s2[0]=quad_sum(s2[0])/128.f;s2[1]=quad_sum(s2[1])/128.f;
  float* stats=reinterpret_cast<float*>(lnsm+36864);float* tmp=reinterpret_cast<float*>(lnsm+32768);
  if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
  allsync(); // Publish both64-channel halves before full128-channel LN reduction.
  float c1[2]={stats[ra*2]+stats[128+ra*2],stats[rb*2]+stats[128+rb*2]},c2[2]={stats[ra*2+1]+stats[128+ra*2+1],stats[rb*2+1]+stats[128+rb*2+1]};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair,gc=wi*64+c;
    float xaa=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,ra,c),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,ra,c+1),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,rb,c),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,rb,c+1),mu[1]),rs[1]);
    float da=acc[q*8+j*2],db=acc[q*8+j*2+1],dc=acc[q*8+j*2+2],dd=acc[q*8+j*2+3],ga=gamma[gc],gb=gamma[gc+1];
    uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    uint32_t resa=pair_get(lnsm+16384+wi*8192,ra,c),resb=pair_get(lnsm+16384+wi*8192,rb,c);
    *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(ra,c*2))=pack_bf16(bf16lo(outa)+bf16lo(resa),bf16hi(outa)+bf16hi(resa));
    *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(rb,c*2))=pack_bf16(bf16lo(outb)+bf16lo(resb),bf16hi(outb)+bf16hi(resb));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+gc]=dga;tmp[w*256+gc+1]=dgb;tmp[w*256+128+gc]=dba;tmp[w*256+128+gc+1]=dbb;}
   });
  });
  allsync(); // All warp parameter partials and dx stores published.
  running+=(tmp[threadIdx.x]+tmp[256+threadIdx.x])+(tmp[512+threadIdx.x]+tmp[768+threadIdx.x]);
  fence_proxy_async();allsync();
  if(threadIdx.x==0){store2d(&p.dx,lnsm,0,row);store2d(&p.dx,lnsm+8192,64,row);tma_store_commit();}
  // dx store overlaps the next gate contraction; drained before front buffers refill.
 }
 if(threadIdx.x==0)tma_store_wait_all();allsync();
 p.partln[split*256+threadIdx.x]=running;
}
#if MW_CNW > 1
// ---- wide path (C >= 256) --------------------------------------------------------------------------------------------
// Shared memory, in three regions: the (side, hidden chunk) tiles, the weight tiles for one 128-channel chunk, and a tail
// region that carries the gate operands, then the LayerNorm staging, then the parameter partials.
#if MW_MROWS == 2
constexpr int GBAR=2+(M2_WGRING?2:1)*GW_DEPTH;
#if M2_WGRING && ROLE_ONLY != 1                         // dx only: the dW kernel's own ring lives at the same indices
#define MW_WG_RING_BAR(i) ((i)>=2&&(i)<GBAR)               // per-warpgroup weight rings have a single issuer
#endif         // 0,1 h ring | weight ring(s) | gate x2 | x,res
#else
constexpr int GBAR=2+GW_DEPTH;                         // 0,1 h ring | 2.. weights | GBAR gate | GBAR+1 x,res
#endif
#if MW_MROWS == 1
constexpr int GH_BASE=0,GH_SLOT=40960;                 // pre | d(sigma) | GLU gate out | GLU projection out
constexpr int GW_BASE=2*GH_SLOT,GW_SLOT=32768;         // gate weights (2 x 64ch) | projection weights (2 x 64ch)
constexpr int GL_BASE=GW_BASE+GW_DEPTH*GW_SLOT;
constexpr int GL_TMP=49152,GL_STATS=GL_TMP+32*CDIM;    // gate operands (or the LayerNorm staging) below, partials above
static_assert(GL_BASE+GL_STATS+1024<=229376,"wide dx role does not fit in shared memory");
static_assert(NISSUE==4,"the wide loaders split their tiles over exactly four issuers");

TMN_DEVI void gload_h(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int side,int h){
 int q=issuer();if(q<0)return;uint8_t* s=sm+GH_BASE+slot*GH_SLOT;
 mbar_arrive_expect_tx(b+slot,q==0?16384:(q==1?8192:0));
 if(q==0)tma_load_2d(s,&p.pre,b+slot,row,side*PREW+h*128);
 else if(q==1)tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h*64);
}
TMN_DEVI void gload_w(const Params& p,uint8_t* sm,uint64_t* b,int slot,int side,int h,int cs){
 int q=issuer();if(q<0)return;uint8_t* s=sm+GW_BASE+slot*GW_SLOT;mbar_arrive_expect_tx(b+slot,q<4?8192:0);
 if(q<2)tma_load_2d(s+q*8192,side?&p.wrg:&p.wlg,b+slot,h*64,cs*128+q*64);
 else if(q<4)tma_load_2d(s+16384+(q-2)*8192,side?&p.wr:&p.wl,b+slot,h*64,cs*128+(q-2)*64);
}
TMN_DEVI void gload_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row,int cs,int kc){
 int q=issuer();if(q<0)return;uint8_t* s=sm+GL_BASE;mbar_arrive_expect_tx(bar,q<2?16384:8192);
 if(q<2)tma_load_2d(s+q*8192,&p.dg,bar,kc*128+q*64,row);                    // q 0,1 take a dg half as well
 {int k=q&1,n=q>>1;tma_load_2d(s+16384+n*16384+k*8192,&p.wgate,bar,kc*128+k*64,cs*128+n*64);}
}
TMN_DEVI void gload_x(const Params& p,uint8_t* sm,uint64_t* bar,int row,int cs,bool with_res){
 int q=issuer();if(q<0)return;uint8_t* s=sm+GL_BASE;mbar_arrive_expect_tx(bar,(with_res||q<2)?8192:0);
 if(q<2)tma_load_2d(s+q*8192,&p.x,bar,cs*128+q*64,row);
 else if(with_res)tma_load_2d(s+16384+(q-2)*8192,&p.res,bar,cs*128+(q-2)*64,row);
}

TMN_DEVI void input_role_wide(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma){
 int split=blockIdx.x-DWCOUNT,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;
 constexpr int TCOUNT=2*HCH,UCNT=TCOUNT*CNW;
 float* tmp=reinterpret_cast<float*>(sm+GL_BASE+GL_TMP);float* stats=reinterpret_cast<float*>(sm+GL_BASE+GL_STATS);
 float runacc[CNW]={};int cg=0,cx=0,wbits=0;           // HCH is even, so the h ring keeps its phase; the weight ring counts its own
 static_assert(HCH%2==0,"h ring phase must repeat every row tile");
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT){
  int row=tile*64;float acc[CNW][32]={};
  auto issue_h=[&](int t){if(t<TCOUNT)gload_h(p,sm,bar,t&1,row,t/HCH,t%HCH);};
  auto issue_w=[&](int u){if(u<UCNT)gload_w(p,sm,bar+2,u%GW_DEPTH,(u/CNW)/HCH,(u/CNW)%HCH,u%CNW);};
  issue_h(0);issue_h(1);for(int i=0;i<GW_DEPTH;++i)issue_w(i);      // the loop refills the slot it just consumed
  for(int t=0;t<TCOUNT;++t){uint8_t* s=sm+GH_BASE+(t&1)*GH_SLOT;
   mbar_wait(bar+(t&1),(t/2)&1);glu_small(p,s,s+24576,s+32768,row);
   static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;int u=t*CNW+cs,slot=u%GW_DEPTH;
    uint8_t* ws=sm+GW_BASE+slot*GW_SLOT;
    int ph=(wbits>>slot)&1;wbits^=1<<slot;mbar_wait(bar+2+slot,ph);
    wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;      // d(sigma) x gate weights
     mma_input64(acc[cs],smem_desc(smem_u32(s+24576+k*2048),16,1024,1),smem_desc(smem_u32(ws+wi*8192+k*32),16,1024,1),t>0||k>0);});
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;      // GLU projection x projection weights
     mma_input64(acc[cs],smem_desc(smem_u32(s+32768+k*2048),16,1024,1),smem_desc(smem_u32(ws+16384+wi*8192+k*32),16,1024,1),1);});
    wgmma_commit();wgmma_wait<0>();allsync();issue_w(u+GW_DEPTH);
   });
   issue_h(t+2);
  }
  wgmma_wait<0>();static_for<CNW>([&](auto csi){fence_regs(acc[decltype(csi)::value]);});allsync();
  // B9: the gate contraction, rounded to BF16 before it is added, exactly as the C = 128 path adds it.
  static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;float gate[32]={};
   for(int kc=0;kc<CNW;++kc){gload_gate(p,sm,bar+GBAR,row,cs,kc);mbar_wait(bar+GBAR,(cg++)&1);
    fence_regs(gate);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;         // K = 128 per group: half as many waits
     mma_dgrad(gate,smem_desc(smem_u32(sm+GL_BASE+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+GL_BASE+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),kc>0||k>0);});
    wgmma_commit();wgmma_wait<0>();fence_regs(gate);allsync();
   }
   static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;uint32_t g=pack_bf16(gate[j*2],gate[j*2+1]);
    acc[cs][j*2]=__bfloat162float(__float2bfloat16_rn(acc[cs][j*2]+bf16lo(g)));
    acc[cs][j*2+1]=__bfloat162float(__float2bfloat16_rn(acc[cs][j*2+1]+bf16hi(g)));});
  });
  // B11/B12: the LayerNorm backward needs the row sums over all C channels, so x is walked twice.
  uint8_t* lnsm=sm+GL_BASE;
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;
   gload_x(p,sm,bar+GBAR+1,row,cs,false);mbar_wait(bar+GBAR+1,(cx++)&1);
   static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
    static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1),r=rr?rb:ra;
     float xa=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,r,c),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,r,c+1),mu[rr]),rs[rr]);
     float ha=acc[cs][q*8+j*2]*gamma[cs*128+wi*64+c],hb=acc[cs][q*8+j*2+1]*gamma[cs*128+wi*64+c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;});});
   allsync();
  });
  s1[0]=quad_sum(s1[0])/float(CDIM);s1[1]=quad_sum(s1[1])/float(CDIM);s2[0]=quad_sum(s2[0])/float(CDIM);s2[1]=quad_sum(s2[1])/float(CDIM);
  if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
  allsync();
  float c1[2]={stats[ra*2]+stats[128+ra*2],stats[rb*2]+stats[128+rb*2]},c2[2]={stats[ra*2+1]+stats[128+ra*2+1],stats[rb*2+1]+stats[128+rb*2+1]};
  static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;
   gload_x(p,sm,bar+GBAR+1,row,cs,true);mbar_wait(bar+GBAR+1,(cx++)&1);
   static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
    static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair,gc=cs*128+wi*64+c;
     float xaa=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,ra,c),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,ra,c+1),mu[0]),rs[0]);
     float xba=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,rb,c),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,rb,c+1),mu[1]),rs[1]);
     float da=acc[cs][q*8+j*2],db=acc[cs][q*8+j*2+1],dc=acc[cs][q*8+j*2+2],dd=acc[cs][q*8+j*2+3],ga=gamma[gc],gb=gamma[gc+1];
     uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
     uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
     uint32_t resa=pair_get(lnsm+16384+wi*8192,ra,c),resb=pair_get(lnsm+16384+wi*8192,rb,c);
     *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(ra,c*2))=pack_bf16(bf16lo(outa)+bf16lo(resa),bf16hi(outa)+bf16hi(resa));
     *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(rb,c*2))=pack_bf16(bf16lo(outb)+bf16lo(resb),bf16hi(outb)+bf16hi(resb));
     float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
     for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
     if(lane<4){tmp[w*2*CDIM+gc]=dga;tmp[w*2*CDIM+gc+1]=dgb;tmp[w*2*CDIM+CDIM+gc]=dba;tmp[w*2*CDIM+CDIM+gc+1]=dbb;}});});
   fence_proxy_async();allsync();
   if(threadIdx.x==0){store2d(&p.dx,lnsm,cs*128,row);store2d(&p.dx,lnsm+8192,cs*128+64,row);tma_store_commit();tma_store_wait_all();}
   allsync();
  });
  static_for<CNW>([&](auto ki){constexpr int k=decltype(ki)::value;int c=k*256+threadIdx.x;   // 2C values over 256 threads
   runacc[k]+=(tmp[c]+tmp[2*CDIM+c])+(tmp[4*CDIM+c]+tmp[6*CDIM+c]);});
  allsync();
 }
 static_for<CNW>([&](auto ki){constexpr int k=decltype(ki)::value;p.partln[split*2*CDIM+k*256+threadIdx.x]=runacc[k];});
}
#endif
// ---- wide dx role, 128-row blocks ------------------------------------------------------------------------------------
// Same arithmetic as input_role_wide, but each CTA carries two 64-row tiles at once, so one weight tile feeds 128 rows
// instead of 64. The weight tiles are the dominant read at these widths (4 x HS x C bf16 per row tile at M = 64), and
// this halves them. The accumulator doubles to acc[2][CNW][32], which the one-CTA-per-SM register budget allows, and the
// GLU is written over the preactivation tile it consumed to pay for the second row tile's shared memory.
#if MW_MROWS == 2
static_assert(NISSUE==4,"the 128-row loaders split their tiles over exactly four issuers");
constexpr int M2H_BASE=0,M2H_HALF=24576,M2H_SLOT=2*M2H_HALF;   // per half: [pre -> GLU gate | GLU projection][d(sigma)]
constexpr int M2H_SIZE=2*M2H_SLOT;
// The h ring is idle once the main loop ends, so the epilogue borrows it: first the gate operands double buffered
// (2 x [wgate 32768][dg 16384]), then x and res for all C channels of one row tile at once (65536). That removes the
// serial round trips the epilogue used to pay -- one wait per row tile instead of four -- and frees a weight ring slot.
constexpr int M2G_SLOT=49152,M2X_RES=32768;
constexpr int M2W_BASE=M2H_SIZE,M2W_HALF=16384,M2W_SLOT=2*M2W_HALF;   // one half per warpgroup, ringed per warpgroup
constexpr int M2G_BASE=M2W_BASE+GW_DEPTH*M2W_SLOT;     // GLU output: 2 h slots x 2 row halves x (gate | projection)
constexpr int M2G_SIZE=(M2_GLU_SEP?2*16384:0);   // one buffer: the GLU output is consumed in the same step
constexpr int M2T_BASE=M2G_BASE+M2G_SIZE,M2T_STATS=32*CDIM;
static_assert(2*M2G_SLOT<=M2H_SIZE&&2*M2X_RES<=M2H_SIZE,"the epilogue does not fit in the h ring");
static_assert(M2T_BASE+M2T_STATS+2048<=229376,"128-row dx role does not fit in shared memory");

TMN_DEVI void gload_h2(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int side,int h){
 int q=issuer();if(q<0)return;uint8_t* s=sm+M2H_BASE+slot*M2H_SLOT+(q>>1)*M2H_HALF;int r=row+(q>>1)*64;
 mbar_arrive_expect_tx(b+slot,(q&1)?8192:16384);
 if(q&1)tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,r,h*64);
 else tma_load_2d(s,&p.pre,b+slot,r,side*PREW+h*128);
}
// Each warpgroup reads only its own 64 channels of a weight tile, so it gets its own ring, its own barriers and its own
// issuer. Nothing in the weight path is then shared between the warpgroups, and the CTA-wide barrier per mma group --
// 24% of the dx kernel's issue slots went to barrier stalls -- drops to one per hidden chunk.
TMN_DEVI void gload_w2(const Params& p,uint8_t* sm,uint64_t* b,int wi,int slot,int side,int h,int cs){
#if M2_WGRING
 if(threadIdx.x!=(unsigned)wi*128)return;uint8_t* s=sm+M2W_BASE+(wi*GW_DEPTH+slot)*M2W_HALF;
 uint64_t* bar=b+wi*GW_DEPTH+slot;mbar_arrive_expect_tx(bar,M2W_HALF);
 tma_load_2d(s,side?&p.wrg:&p.wlg,bar,h*64,cs*128+wi*64);
 tma_load_2d(s+8192,side?&p.wr:&p.wl,bar,h*64,cs*128+wi*64);
#else
 int q=issuer();if(q<0)return;uint8_t* s=sm+M2W_BASE+slot*M2W_SLOT;mbar_arrive_expect_tx(b+slot,q<4?8192:0);
 if(q<2)tma_load_2d(s+q*8192,side?&p.wrg:&p.wlg,b+slot,h*64,cs*128+q*64);
 else if(q<4)tma_load_2d(s+16384+(q-2)*8192,side?&p.wr:&p.wl,b+slot,h*64,cs*128+(q-2)*64);
#endif
}
TMN_DEVI void gload_gate2(const Params& p,uint8_t* sm,uint64_t* bar,int slot,int row,int cs,int kc){
 int q=issuer();if(q<0)return;uint8_t* s=sm+slot*M2G_SLOT;                   // [wgate 4 x 8192][dg 2 x 8192]
 mbar_arrive_expect_tx(bar+slot,q<2?16384:8192);
 if(q<2)tma_load_2d(s+32768+q*8192,&p.dg,bar+slot,kc*128+q*64,row);
 {int k=q&1,n=q>>1;tma_load_2d(s+n*16384+k*8192,&p.wgate,bar+slot,kc*128+k*64,cs*128+n*64);}
}
// x and res for every channel of one 64-row tile, in one transaction: [x: cs, n][res: cs, n]
TMN_DEVI void gload_xres(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 int q=issuer();if(q<0)return;int mine=0;
 for(int i=q;i<2*CNW;i+=NISSUE)mine+=16384;
 mbar_arrive_expect_tx(bar,mine);
 for(int i=q;i<2*CNW;i+=NISSUE){tma_load_2d(sm+i*8192,&p.x,bar,i*64,row);
  tma_load_2d(sm+M2X_RES+i*8192,&p.res,bar,i*64,row);}
}
// With M2_GLU_SEP the derivative lands in its own buffer, so the loop is one pass and one barrier. In place (the
// fallback) every read has to be staged in registers and published after an extra barrier, because the tile being
// written is the tile being read.
TMN_DEVI void glu_ip(const Params& p,uint8_t* s,int row,uint8_t* out){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
#if !M2_GLU_SEP
 uint32_t og[8],op[8];
#endif
 #pragma unroll
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*256,c=i/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);
  uint32_t masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));
  float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));
  uint32_t pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
#if M2_GLU_SEP
  *reinterpret_cast<uint32_t*>(out+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(out+8192+swz128(c,r*2))=pp;
#else
  og[q]=g;op[q]=pp;
#endif
 }
#if !M2_GLU_SEP
 allsync();
 #pragma unroll
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*256,c=i/32,r=(i%32)*2;
  *reinterpret_cast<uint32_t*>(out+swz128(c,r*2))=og[q];*reinterpret_cast<uint32_t*>(out+8192+swz128(c,r*2))=op[q];
 }
#endif
 fence_proxy_async();allsync();
}

TMN_DEVI void input_role_m2(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma){
 int split=blockIdx.x-DWCOUNT,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;
 constexpr int TCOUNT=2*HCH,UCNT=TCOUNT*CNW;
 float* tmp=reinterpret_cast<float*>(sm+M2T_BASE);float* stats=reinterpret_cast<float*>(sm+M2T_BASE+M2T_STATS);
 float runacc[CNW]={};int cg=0,cx=0,wbits=0;
 static_assert(HCH%2==0,"h ring phase must repeat every row block");
 for(int blk=split;blk<(p.tiles>>1);blk+=DXCOUNT){
  int row=blk*128;float acc[2][CNW][32]={};
  auto issue_h=[&](int t){if(t<TCOUNT)gload_h2(p,sm,bar,t&1,row,t/HCH,t%HCH);};
  auto issue_w=[&](int u){if(u<UCNT)gload_w2(p,sm,bar+2,wi,u%GW_DEPTH,(u/CNW)/HCH,(u/CNW)%HCH,u%CNW);};
  issue_h(0);issue_h(1);for(int i=0;i<GW_DEPTH;++i)issue_w(i);
  for(int t=0;t<TCOUNT;++t){uint8_t* s=sm+M2H_BASE+(t&1)*M2H_SLOT;
#if M2_GLU_SEP
   uint8_t* go=sm+M2G_BASE;constexpr int GOSTRIDE=16384;                  // its own buffer: one pass, one barrier
#else
   uint8_t* go=s;constexpr int GOSTRIDE=M2H_HALF;
#endif
   MW_WAIT(bar+(t&1),(t/2)&1,0);glu_ip(p,s,row,go);glu_ip(p,s+M2H_HALF,row+64,go+GOSTRIDE);
   static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;int u=t*CNW+cs,slot=u%GW_DEPTH;
#if M2_WGRING
    uint8_t* ws=sm+M2W_BASE+(wi*GW_DEPTH+slot)*M2W_HALF;int wb=2+wi*GW_DEPTH+slot,wg=0,wp=8192;
#else
    uint8_t* ws=sm+M2W_BASE+slot*M2W_SLOT;int wb=2+slot,wg=wi*8192,wp=16384+wi*8192;
#endif
    int ph=(wbits>>slot)&1;wbits^=1<<slot;MW_WAIT(bar+wb,ph,1);
    wgmma_fence();
    static_for<2>([&](auto hi){constexpr int hf=decltype(hi)::value;uint8_t* a=go+hf*GOSTRIDE;
     static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;      // d(sigma) x gate weights
      mma_input64(acc[hf][cs],smem_desc(smem_u32(a+k*2048),16,1024,1),smem_desc(smem_u32(ws+wg+k*32),16,1024,1),t>0||k>0);});
     static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;      // GLU projection x projection weights
      mma_input64(acc[hf][cs],smem_desc(smem_u32(a+8192+k*2048),16,1024,1),smem_desc(smem_u32(ws+wp+k*32),16,1024,1),1);});
    });
    // The ring is private to the warpgroup, so the CTA barrier becomes a 128-thread named one -- but it cannot be
    // dropped: re-arming a barrier while a sibling warp still spins on the old phase flips it twice and hangs.
    wgmma_commit();wgmma_wait<0>();
#if M2_WGRING && M2_WG_SYNC
    named_bar_sync(1+wi,128);
#else
    allsync();
#endif
    issue_w(u+GW_DEPTH);
   });
   allsync();issue_h(t+2);                                // both warpgroups are done with this h slot
  }
  static_for<CNW>([&](auto csi){fence_regs(acc[0][decltype(csi)::value]);fence_regs(acc[1][decltype(csi)::value]);});
  // B9: the gate contraction, double buffered over the h ring the main loop has finished with.
#if !(M2_SKIP & 1)
  constexpr int GSTEPS=2*CNW;                                              // (row tile, channel chunk) pairs
  // step i = ((row half * CNW) + channel chunk) * CNW + contraction chunk, alternating buffers
  auto gstep=[&](int i){if(i<GSTEPS*CNW)gload_gate2(p,sm,bar+GBAR,i&1,row+((i/CNW)/CNW)*64,(i/CNW)%CNW,i%CNW);};
  gstep(0);
  static_for<2>([&](auto hi){constexpr int hf=decltype(hi)::value;
   static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;float gate[32]={};
    for(int kc=0;kc<CNW;++kc){int i=(hf*CNW+cs)*CNW+kc,slot=i&1;uint8_t* gs=sm+slot*M2G_SLOT;
     gstep(i+1);
     MW_WAIT(bar+GBAR+slot,(cg>>slot)&1,2);cg^=1<<slot;
     fence_regs(gate);wgmma_fence();
     static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
      mma_dgrad(gate,smem_desc(smem_u32(gs+32768+(k/4)*8192+(k%4)*32),16,1024,1),
                     smem_desc(smem_u32(gs+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),kc>0||k>0);});
     wgmma_commit();wgmma_wait<0>();fence_regs(gate);allsync();
    }
    static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;uint32_t g=pack_bf16(gate[j*2],gate[j*2+1]);
     acc[hf][cs][j*2]=__bfloat162float(__float2bfloat16_rn(acc[hf][cs][j*2]+bf16lo(g)));
     acc[hf][cs][j*2+1]=__bfloat162float(__float2bfloat16_rn(acc[hf][cs][j*2+1]+bf16hi(g)));});
   });
  });
#endif
#if !(M2_SKIP & 2)
  // B11/B12: x and res for a whole row tile arrive in one transaction, and both LayerNorm passes read them from there.
#if !(M2_SKIP & 32)
  gload_xres(p,sm,bar+GBAR+2,row);
#endif
#if M2_SKIP & 8
  if(threadIdx.x==0)tma_store_wait_all();                                  // bisect: drain before every reuse
#endif
  static_for<2>([&](auto hi){constexpr int hf=decltype(hi)::value;int r0=row+hf*64;
#if !(M2_SKIP & 32)
   MW_WAIT(bar+GBAR+2,(cx++)&1,3);
#endif
   float mu[2]={p.mean[r0+ra],p.mean[r0+rb]},rs[2]={p.rs[r0+ra],p.rs[r0+rb]},s1[2]={},s2[2]={};
#if !(M2_SKIP & 128)
   static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;uint8_t* xs=sm+cs*16384+wi*8192;
    static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
     static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1),r=rr?rb:ra;
      float xa=__fmul_rn(__fsub_rn(get(xs,r,c),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(get(xs,r,c+1),mu[rr]),rs[rr]);
      float ha=acc[hf][cs][q*8+j*2]*gamma[cs*128+wi*64+c],hb=acc[hf][cs][q*8+j*2+1]*gamma[cs*128+wi*64+c+1];
      s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;});});
   });
#endif
   s1[0]=quad_sum(s1[0])/float(CDIM);s1[1]=quad_sum(s1[1])/float(CDIM);s2[0]=quad_sum(s2[0])/float(CDIM);s2[1]=quad_sum(s2[1])/float(CDIM);
   if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
   allsync();
   float c1[2]={stats[ra*2]+stats[128+ra*2],stats[rb*2]+stats[128+rb*2]},c2[2]={stats[ra*2+1]+stats[128+ra*2+1],stats[rb*2+1]+stats[128+rb*2+1]};
#if !(M2_SKIP & 16)
   static_for<CNW>([&](auto csi){constexpr int cs=decltype(csi)::value;uint8_t* xs=sm+cs*16384+wi*8192;
    static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
     static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair,gc=cs*128+wi*64+c;
      float xaa=__fmul_rn(__fsub_rn(get(xs,ra,c),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(xs,ra,c+1),mu[0]),rs[0]);
      float xba=__fmul_rn(__fsub_rn(get(xs,rb,c),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(xs,rb,c+1),mu[1]),rs[1]);
      float da=acc[hf][cs][q*8+j*2],db=acc[hf][cs][q*8+j*2+1],dc=acc[hf][cs][q*8+j*2+2],dd=acc[hf][cs][q*8+j*2+3],ga=gamma[gc],gb=gamma[gc+1];
      uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
      uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
      uint32_t resa=pair_get(sm+M2X_RES+cs*16384+wi*8192,ra,c),resb=pair_get(sm+M2X_RES+cs*16384+wi*8192,rb,c);
      *reinterpret_cast<uint32_t*>(xs+swz128(ra,c*2))=pack_bf16(bf16lo(outa)+bf16lo(resa),bf16hi(outa)+bf16hi(resa));
      *reinterpret_cast<uint32_t*>(xs+swz128(rb,c*2))=pack_bf16(bf16lo(outb)+bf16lo(resb),bf16hi(outb)+bf16hi(resb));
      float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
      for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
      if(lane<4){tmp[w*2*CDIM+gc]=dga;tmp[w*2*CDIM+gc+1]=dgb;tmp[w*2*CDIM+CDIM+gc]=dba;tmp[w*2*CDIM+CDIM+gc+1]=dbb;}});});
   });
#endif
   fence_proxy_async();allsync();
#if !(M2_SKIP & 4)
   if(threadIdx.x==0){for(int i=0;i<2*CNW;++i)store2d(&p.dx,sm+i*8192,i*64,r0);tma_store_commit();}
#endif
#if !(M2_SKIP & 256)
   static_for<CNW>([&](auto ki){constexpr int k=decltype(ki)::value;int c=k*256+threadIdx.x;   // 2C values over 256 threads
    runacc[k]+=(tmp[c]+tmp[2*CDIM+c])+(tmp[4*CDIM+c]+tmp[6*CDIM+c]);});
#endif
#if M2_SKIP & 4
   allsync();
#if !(M2_SKIP & 32)
   if(hf==0)gload_xres(p,sm,bar+GBAR+2,row+64);
#endif
#else
   if(threadIdx.x==0)tma_store_wait_all();allsync();
#if !(M2_SKIP & 32)
   if(hf==0)gload_xres(p,sm,bar+GBAR+2,row+64);
#endif
#endif
  });
#endif
 }
 static_for<CNW>([&](auto ki){constexpr int k=decltype(ki)::value;p.partln[split*2*CDIM+k*256+threadIdx.x]=runacc[k];});
}
#endif
#endif

#ifndef MW_WG_RING_BAR
#define MW_WG_RING_BAR(i) 0
#endif
TMN_DEVI void reduce_at(const Params& p,int i){
 constexpr int DWVALS=DWGROUPS*16384, LNVALS=2*CDIM;
 if(i<DWVALS){int group=i/16384,j=i%16384,kind=j/8192,z=j%8192;float v=0;
  for(int b=0;b<DW_SPLITS*2;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+b)*16384+j];
  int side=group/(HCH*CNW),rest=group%(HCH*CNW),hc=rest/CNW,cs=rest%CNW;
  int out=side*2+(kind==0?1:0),rr=cs*128+z%128,c=hc*64+z/128;p.dw[(out*CDIM+rr)*HS+c]=__float2bfloat16_rn(v);
 }else if(i<DWVALS+LNVALS){int c=i-DWVALS;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*LNVALS+c];(c<CDIM?p.dgam:p.dbeta)[c%CDIM]=v;}
}
#if MW_CNW == 1
#define MW_BARS (DW_DEPTH>4?DW_DEPTH:4)
#define MW_MINBLOCKS 2
#elif ROLE_ONLY == 1
#define MW_BARS (DW_DEPTH>4?DW_DEPTH:4)
#define MW_MINBLOCKS DW_BLOCKS             // the dW role needs 64 accumulator registers, so it can keep two CTAs per SM
#else
#define MW_BARS (DW_DEPTH>GBAR+3?DW_DEPTH:GBAR+3)
#define MW_MINBLOCKS DX_BLOCKS             // the wide dx role holds C channels of accumulator, usually one CTA per SM
#endif
#if ROLE_ONLY == 1
#define MW_BLOCK DW_THREADS                // the dW-only kernel may be two warpgroup pairs
#else
#define MW_BLOCK 256
#endif
extern "C" __global__ __launch_bounds__(MW_BLOCK,MW_MINBLOCKS)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[MW_BARS];__shared__ float gamma[CDIM];for(int g=threadIdx.x;g<CDIM;g+=blockDim.x)gamma[g]=p.gamma[g];
  // the per-warpgroup weight rings have one issuer, every other barrier counts all NISSUE of them
 if(threadIdx.x==0){for(int i=0;i<MW_BARS;++i)mbar_init(bar+i,MW_WG_RING_BAR(i)?1:NISSUE);fence_barrier_init();}
#if ROLE_ONLY != 1
 allsync();
#endif
 #if MW_CNW == 1
#define MW_DX_ROLE input_role
#elif MW_MROWS == 2
#define MW_DX_ROLE input_role_m2
#else
#define MW_DX_ROLE input_role_wide
#endif
#if ROLE_ONLY == 1
 named_bar_sync(0,DW_THREADS);weight_role(p,sm,bar);
#elif ROLE_ONLY == 2
 MW_DX_ROLE(p,sm,bar,gamma);
#else
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else MW_DX_ROLE(p,sm,bar,gamma);
#endif
#if PART_ONLY == 2
 __threadfence();allsync(); // Publish all partial writers before completion ticket.
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<DWGROUPS*16384+2*CDIM;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}   // grid: ceil((DWGROUPS*16384 + 2*CDIM)/256)
