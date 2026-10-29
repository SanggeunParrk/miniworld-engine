
#if defined(ATOM_HOPPER)
#include "../sm100_atom/sm100.cuh"
using namespace atom90;
// Two adjacent 32-query windows share a 160-key union. Three TMA stages
// keep Q/K/V local; register softmax and WGMMA PV never materialize S or P.
extern "C" __global__ __launch_bounds__(160, 1)
void local_attn_fwd_sm90(const __grid_constant__ CUtensorMap mq,
 const __grid_constant__ CUtensorMap mk,const __grid_constant__ CUtensorMap mv,
 const float* __restrict__ bias,const uint8_t* __restrict__ mask,
 __nv_bfloat16* __restrict__ out,float* __restrict__ lse,int N,int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int TB=8192, KV=TB, VB=KV+3*TB, BAR=VB+3*TB;
  uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
  const int t=threadIdx.x, lane=t&31, warp=t>>5;
  const int h=blockIdx.y%4, a=blockIdx.y/4, q0=64*blockIdx.x, key0=q0-48;
  if(t==0) { for(int j=0;j<4;++j)init(bars+j); init_fence(); }
  __syncthreads();
  if(t>=128) {
    if(t==128) {
      expect(bars,TB); load3(sm,&mq,bars,h*32,q0,a);
      #pragma unroll
      for(int j=0;j<3;++j) {
        expect(bars+j+1,2*TB);
        load3(sm+KV+j*TB,&mk,bars+j+1,h*32,key0+64*j,a);
        load3(sm+VB+j*TB,&mv,bars+j+1,h*32,key0+64*j,a);
      }
    }
    return;
  }
  const int r0=warp*16+lane/4, c0=(lane&3)*2;
  uint32_t qa[8];
  bwait(bars,0);
  #pragma unroll
  for(int k=0;k<2;++k)
    #pragma unroll
    for(int hi=0;hi<2;++hi)
      #pragma unroll
      for(int rr=0;rr<2;++rr)
        qa[k*4+2*hi+rr]=*reinterpret_cast<uint32_t*>(sm+off(r0+8*rr,16*k+8*hi+c0));
  float oo[16]={0.f}, mm[2]={-INFINITY,-INFINITY}, ll[2]={0.f,0.f};
  #pragma unroll
  for(int j=0;j<3;++j) {
    bwait(bars+j+1,0);
    const uint8_t* sk=sm+KV+j*TB; const uint8_t* sv=sm+VB+j*TB;
    float ss[32]; fence();
    mma_rs_n64_first(qa[0],qa[1],qa[2],qa[3],desc(sk),ss);
    mma_rs_n64(qa[4],qa[5],qa[6],qa[7],desc(sk+32),ss,1);
    commit();wait();
    float nm[2]={-INFINITY,-INFINITY};
    #pragma unroll
    for(int i=0;i<8;++i)
      #pragma unroll
      for(int rr=0;rr<2;++rr) {
        const int q=q0+r0+8*rr, w=q/32, lo=32*w-48;
        #pragma unroll
        for(int e=0;e<2;++e) {
          const int k=key0+64*j+8*i+c0+e;
          const bool ok=q<N && k>=0 && k<N && k>=lo && k<lo+128 && (!mask || mask[k]);
          float s=-INFINITY;
          if(ok) s=fmaf(ss[4*i+2*rr+e],0.17677669529663687f,
             bias[((size_t(h)*(N/32)+w)*32+(q&31))*128+(k-lo)])*1.4426950408889634f;
          ss[4*i+2*rr+e]=s; nm[rr]=fmaxf(nm[rr],s);
        }
      }
    #pragma unroll
    for(int rr=0;rr<2;++rr) {
      nm[rr]=fmaxf(mm[rr],max4(nm[rr]));
      const float safe=nm[rr]==-INFINITY ? 0.f:nm[rr];
      const float alpha=mm[rr]==-INFINITY?0.f:s100::ex2f(mm[rr]-safe);
      ll[rr]*=alpha;
      #pragma unroll
      for(int i=0;i<4;++i) { oo[4*i+2*rr]*=alpha;oo[4*i+2*rr+1]*=alpha; }
      mm[rr]=safe;
    }
    uint32_t pp[16]; float sum[2]={0.f,0.f};
    #pragma unroll
    for(int i=0;i<8;++i)
      #pragma unroll
      for(int rr=0;rr<2;++rr) {
        float p0=s100::ex2f(ss[4*i+2*rr]-mm[rr]);
        float p1=s100::ex2f(ss[4*i+2*rr+1]-mm[rr]);
        sum[rr]+=p0+p1;
        pp[4*(i>>1)+2*(i&1)+rr]=pack(p0,p1);
      }
    ll[0]+=sum4(sum[0]);ll[1]+=sum4(sum[1]);
    fence();
    #pragma unroll
    for(int k=0;k<4;++k) pv(oo,pp+4*k,desc(sv+k*2048));
    commit();wait();
  }
  #pragma unroll
  for(int rr=0;rr<2;++rr) {
    const int q=q0+r0+8*rr;
    if(q<N) {
      const float inv=ll[rr]>0?1.f/ll[rr]:0.f;
      #pragma unroll
      for(int i=0;i<4;++i)
        *reinterpret_cast<uint32_t*>(out+(size_t(a)*N+q)*128+h*32+8*i+c0)=pack(oo[4*i+2*rr]*inv,oo[4*i+2*rr+1]*inv);
      if((lane&3)==0) lse[(size_t(a)*4+h)*N+q]=ll[rr]>0?mm[rr]+__log2f(ll[rr]):1e30f;
    }
  }
}
#else
// lattn_fwd.cu — AF3 windowed atom attention forward (32 queries x 128 keys per window) on tcgen05 (sm_100a).
//
//   S = q k^T / sqrt(32) + bias[h][w]      O = softmax(S) v      (bf16 O [A N, 128], row LSE [A, 4, N] in log2 units)
//
// A CTA owns 128 queries [q0, q0 + 128) (windows 4 c .. 4 c + 3), one head and a range of samples (SP CTAs split the samples); they see the
// 224 keys [q0 - 48, q0 + 176). TMEM lane = query: S = Q K^T [128 x 224] in two buffers (cols 0..223 / 224..447: the next sample's S runs
// during this one's softmax), O = P V [128 x 32] cols 448..479. Warp w < 16 (lane quadrant w & 3 = one window, band quarter w >> 2) reads
// its 32 band cells (bias from shared memory, log2 domain, key mask folded in as -inf); the quadrant's four warps exchange row maxima and
// sums through shared memory; P (packed bf16, out-of-band cells zero) goes over the S buffer as the TMEM A operand of O += P V (B = the V
// tile, MN-major). The previous sample's O / l is drained (band quarters 0, 1: shared memory + TMA store; quarter 2: LSE) while this
// sample's P is formed. Warp 16 issues the TMA loads (3 stages, refilled when a sample's MMAs commit) and the MMAs.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
#include "slide.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QN = 128, KN = 224, NST = 3, BSTR = 33;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_K = QN * 64, T_V = T_K + KN * 64;
constexpr int STAGE = 36864;
static_assert(T_V + KN * 64 <= STAGE && STAGE % 1024 == 0, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 16 * 32 * BSTR * 4, O_R = O_O + 16 * 1024, O_BAR = O_R + 2 * 2 * 16 * 32 * 4;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int C_O = 448;
constexpr uint32_t TX = QN * 64 + 2 * KN * 64;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_fwd(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                  const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
                  float* __restrict__ LSE, int N, int A, int nwin, int SP, int SPE) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ int sred[32];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 2, *oready = bars + NST + 3, *empty = bars + NST + 4;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);
  float* red = reinterpret_cast<float*>(sm + O_R);                   // [kind 2][parity 2][warp 16][lane 32]: row max / row sum partials
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nedge = 2 * NH * SPE;                                    // the slid window's edge CTAs come first (slide.cuh; SPE = 0: none)
  const bool edge = (int)blockIdx.x < nedge;
  const int sp = edge ? SPE : SP, bx = (int)blockIdx.x - (edge ? 0 : nedge);
  const int pair = bx / sp, part = bx - pair * sp;
  const int jc = pair >> 2, h = pair & 3;                            // (an edge CTA: jc = its group)
  const int nr = SPE > 0 ? slide_nreal(kmask, N, sred) : N;
  int wf = 0, kbe = 0;
  const bool live = !edge || slide_group(jc, nr, wf, kbe);
  const int q0 = edge ? 32 * wf : 128 * jc, kb = edge ? kbe : q0 - KOFF;
  const int a_lo = (int)((long)A * part / sp), a_hi = (int)((long)A * (part + 1) / sp);
  const int nit = live ? a_hi - a_lo : 0;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) mbar_init(&full[i], 1);
    mbar_init(&sready[0], 1); mbar_init(&sready[1], 1); mbar_init(tready, 16); mbar_init(oready, 1);
    for (int i = 0; i < NST; ++i) mbar_init(&empty[i], 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = (q0 >> 5) + qd;
  const int qdb = edge ? 0 : qd;                                     // the window's first band cell (an edge CTA: its keys start the range)
  const bool keep = SPE == 0 || (edge ? slide_mine(jc, wq, nr) : !slide_moved(wq, nr));   // this CTA writes the window's O / LSE
  if (!ctl) {
    const int key = kb + 32 * qdb + 32 * qt + lane;
    const bool kok = key >= 0 && key < N && (kmask == nullptr || kmask[key]);
    const float* src = gbias + (((size_t)h * nwin + (wq < nwin ? wq : 0)) * WQ) * WK + 32 * qt + lane;
#pragma unroll 8
    for (int qq = 0; qq < 32; ++qq) {
      float v = 0.f;
      if (wq < nwin) v = kok ? src[qq * WK] * LOG2E : -INFINITY;
      sbias[(warp * 32 + lane) * BSTR + qq] = v;
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_s = idesc_bf16(128, KN, 0, 0), id_o = idesc_bf16(128, 32, 0, 1);

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
  };
  auto mma_s = [&](int it) {                                         // S of sample it into buffer it & 1
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_sw64(st + T_Q), dk = desc_sw64(st + T_K);
    if (elect_one()) {
      umma_ss(tm + 224 * (it & 1), dq, dk, id_s, 0);
      umma_ss(tm + 224 * (it & 1), dq + 2, dk + 2, id_s, 1);
      tc_commit(&sready[it & 1]);
    }
    __syncwarp();
  };
  auto mma_o = [&](int it) {                                         // O = P V (A = packed bf16 P in buffer it & 1, B = V tile MN-major)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dv = desc_sw64(st + T_V);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KN / 16; ++ks) umma_ts(tm + C_O, tm + 224 * (it & 1) + 8 * ks, dv + (uint64_t)(ks * 1024 >> 4), id_o, ks);
      tc_commit(oready);
      tc_commit(&empty[it % NST]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
    if (nit > 1) mma_s(1);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);
      tc_fence_after();
      mma_o(it);                                                     // commits oready and empty[stage of it]
      if (it + 2 < nit) mma_s(it + 2);
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  } else {

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int x0 = 32 * qdb + 32 * qt;
  const float* sb = sbias + warp * 32 * BSTR + lane;
  const int qrow = q0 + 32 * qd + lane;
  auto drain = [&](int itd, float mm) {                              // O / l (band quarters 0, 1) and LSE (quarter 2) of sample itd
    if (!keep) return;
    const int par = itd & 1, a = a_lo + itd;
    const float* rsum = red + ((1 * 2 + par) * 16) * 32;
    const float lt = rsum[qd * 32 + lane] + rsum[(qd + 4) * 32 + lane] + rsum[(qd + 8) * 32 + lane] + rsum[(qd + 12) * 32 + lane];
    if (qt < 2) {
      uint32_t v[16];
      tmem_ld16(tl + C_O + 16 * qt, v);
      tmem_wait_ld();
      const float il = lt > 0.f ? 1.f / lt : 0.f;
      uint8_t* ob = sm + O_O + warp * 1024;
      if (lane == 0) tma_store_wait_read0();
      __syncwarp();
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        uint4 o;
        o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * il, __uint_as_float(v[8 * g + 1]) * il);
        o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * il, __uint_as_float(v[8 * g + 3]) * il);
        o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * il, __uint_as_float(v[8 * g + 5]) * il);
        o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * il, __uint_as_float(v[8 * g + 7]) * il);
        *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_3d(&tm_o, su + O_O + warp * 1024, h * DH + 16 * qt, q0 + 32 * qd, a);
        tma_store_commit();
      }
    } else if (qt == 2 && qrow < N) {
      LSE[((size_t)a * NH + h) * N + qrow] = lt > 0.f ? mm + __log2f(lt) : 1e30f;
    }
  };
  float mm_prev = 0.f;
  for (int it = 0; it < nit; ++it) {
    const int b = it & 1, par = it & 1;
    mbar_wait(&sready[b], (it >> 1) & 1);
    tc_fence_after();
    float sv[32];
    {
      uint32_t r0[16], r1[16];
      tmem_ld16(tl + 224 * b + x0, r0);
      tmem_ld16(tl + 224 * b + x0 + 16, r1);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        sv[j] = fmaf(__uint_as_float(r0[j]), QSCALE * LOG2E, sb[j * BSTR]);
        sv[16 + j] = fmaf(__uint_as_float(r1[j]), QSCALE * LOG2E, sb[(16 + j) * BSTR]);
      }
    }
    float m = sv[0];
#pragma unroll
    for (int j = 1; j < 32; ++j) m = fmaxf(m, sv[j]);
    float* rmax = red + ((0 * 2 + par) * 16) * 32;
    float* rsum = red + ((1 * 2 + par) * 16) * 32;
    rmax[warp * 32 + lane] = m;
    tc_fence_before();
    named_bar_sync(1 + qd, 128);
    tc_fence_after();
    m = fmaxf(fmaxf(rmax[qd * 32 + lane], rmax[(qd + 4) * 32 + lane]), fmaxf(rmax[(qd + 8) * 32 + lane], rmax[(qd + 12) * 32 + lane]));
    const float mm = m == -INFINITY ? 0.f : m;
    uint32_t pw[16];
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j += 2) {
      const float p0 = ex2f(sv[j] - mm), p1 = ex2f(sv[j + 1] - mm);
      l += p0 + p1;
      pw[j >> 1] = pack_bf16(p0, p1);
    }
    rsum[warp * 32 + lane] = l;
    tmem_st16(tl + 224 * b + x0 / 2, pw);
    if (qt < 3) tmem_st16z(tl + 224 * b + (qt < qdb ? 16 * qt : 16 * qt + 64));
    if (it >= 1) {                                                   // the previous sample's O (its MMAs ran during this softmax)
      mbar_wait(oready, (it - 1) & 1);
      tc_fence_after();
      drain(it - 1, mm_prev);
    }
    mm_prev = mm;
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
  }
  if (nit > 0) {
    mbar_wait(oready, (nit - 1) & 1);
    tc_fence_after();
    drain(nit - 1, mm_prev);
  }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}

#endif // ATOM_HOPPER
