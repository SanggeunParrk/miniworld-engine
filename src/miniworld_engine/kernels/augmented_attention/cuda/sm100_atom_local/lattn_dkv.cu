#if defined(ATOM_HOPPER)
#include "../sm100_atom/sm100.cuh"
using namespace atom90;
// Key-centric Hopper lowering: 64 keys see six adjacent query windows.
// TMA stages hold their 192-query union; WGMMA retains dK/dV accumulators.
// Every valid dBias cell has one writer per sample, followed by fixed-order
// sample reduction. No atomic gradient accumulation or TMEM is needed.
extern "C" __global__ __launch_bounds__(160, 1)
void local_attn_dkv_sm90(const __grid_constant__ CUtensorMap mq,
 const __grid_constant__ CUtensorMap mk,const __grid_constant__ CUtensorMap mv,
 const __grid_constant__ CUtensorMap mdo,const float* __restrict__ bias,
 const uint8_t* __restrict__ mask,const float* __restrict__ lse,
 const float* __restrict__ dd,__nv_bfloat16* __restrict__ out,
 float* __restrict__ db,int ldd,int N,int A) {
 extern __shared__ __align__(1024) uint8_t sm[];
 constexpr int TB=8192,KB=0,VB=TB,QB=2*TB,DOB=5*TB,BAR=8*TB;
 uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
 const int t=threadIdx.x,lane=t&31,warp=t>>5;
 const int h=blockIdx.y%4,a=blockIdx.y/4,k0=64*blockIdx.x,query0=k0-64;
 if(t==0){for(int j=0;j<4;++j)init(bars+j);init_fence();}
 __syncthreads();
 if(t>=128){
  if(t==128){
   expect(bars,2*TB);load3(sm+KB,&mk,bars,h*32,k0,a);load3(sm+VB,&mv,bars,h*32,k0,a);
   #pragma unroll
   for(int j=0;j<3;++j){
    expect(bars+j+1,2*TB);
    load3(sm+QB+j*TB,&mq,bars+j+1,h*32,query0+64*j,a);
    load3(sm+DOB+j*TB,&mdo,bars+j+1,h*32,query0+64*j,a);
   }
  }
  return;
 }
 const int r0=warp*16+lane/4,c0=(lane&3)*2;
 uint32_t ka[8],va[8];bwait(bars,0);
 #pragma unroll
 for(int kk=0;kk<2;++kk)
  #pragma unroll
  for(int hi=0;hi<2;++hi)
   #pragma unroll
   for(int rr=0;rr<2;++rr){
    const int ix=kk*4+2*hi+rr,offx=off(r0+8*rr,16*kk+8*hi+c0);
    ka[ix]=*reinterpret_cast<uint32_t*>(sm+KB+offx);
    va[ix]=*reinterpret_cast<uint32_t*>(sm+VB+offx);
   }
 float dk[16]={0.f},dv[16]={0.f};
 #pragma unroll
 for(int j=0;j<3;++j){
  bwait(bars+j+1,0);
  const uint8_t* sq=sm+QB+j*TB;const uint8_t* sd=sm+DOB+j*TB;
  float ss[32],dp[32];fence();
  mma_rs_n64_first(ka[0],ka[1],ka[2],ka[3],desc(sq),ss);
  mma_rs_n64(ka[4],ka[5],ka[6],ka[7],desc(sq+32),ss,1);
  mma_rs_n64_first(va[0],va[1],va[2],va[3],desc(sd),dp);
  mma_rs_n64(va[4],va[5],va[6],va[7],desc(sd+32),dp,1);
  commit();wait();
  uint32_t ds[16],pp[16];
  #pragma unroll
  for(int i=0;i<8;++i)
   #pragma unroll
   for(int rr=0;rr<2;++rr){
    const int k=k0+r0+8*rr;float dsval[2],pval[2];
    #pragma unroll
    for(int e=0;e<2;++e){
     const int q=query0+64*j+8*i+c0+e,w=q/32,lo=32*w-48;
     const bool window=q>=0&&q<N&&k>=lo&&k<lo+128;
     const bool ok=window&&k<N&&(!mask||mask[k]);
     float p=0.f,value=0.f;
     if(ok){
      const size_t row=(size_t(a)*4+h)*N+q;
      const float score=fmaf(ss[4*i+2*rr+e],0.17677669529663687f,
        bias[((size_t(h)*(N/32)+w)*32+(q&31))*128+(k-lo)]);
      p=s100::ex2f(score*1.4426950408889634f-lse[row]);
      value=p*(dp[4*i+2*rr+e]-dd[row]);
     }
     if(window&&k<N)
      db[(((size_t(a)*4+h)*(N/32)+w)*32+(q&31))*128+(k-lo)]=value;
     dsval[e]=value;pval[e]=p;
    }
    const int ix=4*(i>>1)+2*(i&1)+rr;
    ds[ix]=pack(dsval[0],dsval[1]);pp[ix]=pack(pval[0],pval[1]);
   }
  fence();
  #pragma unroll
  for(int kk=0;kk<4;++kk){pv(dk,ds+4*kk,desc(sq+kk*2048));pv(dv,pp+4*kk,desc(sd+kk*2048));}
  commit();wait();
 }
 #pragma unroll
 for(int rr=0;rr<2;++rr){
  const int k=k0+r0+8*rr;
  if(k<N)
   #pragma unroll
   for(int i=0;i<4;++i){
    const size_t ix=(size_t(a)*N+k)*ldd+h*32+8*i+c0;
    *reinterpret_cast<uint32_t*>(out+ix+128)=
      pack(dk[4*i+2*rr]*0.17677669529663687f,dk[4*i+2*rr+1]*0.17677669529663687f);
    *reinterpret_cast<uint32_t*>(out+ix+256)=pack(dv[4*i+2*rr],dv[4*i+2*rr+1]);
   }
 }
}
extern "C" __global__
void local_bias_samples_sm90(const float* __restrict__ partial,float* __restrict__ out,int N,int A) {
 const size_t count=size_t(4)*(N/32)*32*128;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x){
  const int w=(i/4096)%(N/32),key=32*w-48+(i%128);
  float value=0.f;
  // Out-of-domain slots have no key-centric CTA and are never read.
  if(key>=0&&key<N)for(int a=0;a<A;++a)value+=partial[size_t(a)*count+i];
  out[i]=value;
 }
}
#else
// lattn_dkv.cu — dK, dV and dbias of the AF3 windowed atom attention on tcgen05 (sm_100a), key-centric.
//
// A CTA owns 128 keys [kb, kb + 128) with kb = 128 j - 112 (so every TMEM lane quadrant's 32 keys see the same 4 query windows), one head and
// a range of samples (SP CTAs split the samples; SP = 1 writes dbias directly). Its keys see the 7 windows W0 .. W0 + 6 (W0 = 4 j - 5,
// 224 queries from q0 = 32 W0). TMEM lane = key:
//   S^T = K Q^T [128 x 224] cols 0..223,  dP^T = V dO^T cols 224..447,  dK += dS^T Q cols 448..479,  dV += P^T dO cols 480..511.
// Warp w (lane quadrant w & 3, band quarter w >> 2 = one query window) reads its 32 x 32 band cells of S^T / dP^T, computes
// P = exp2(S c + bias - LSE) and dS = P (dP - D), adds dS into the dbias cells it holds in registers (summed over the samples), and -- after
// the quadrant's four warps have read -- writes P^T and dS^T as packed bf16 over S^T (cols 0..111 / 112..223, out-of-band cells zero): the
// TMEM A operands of dV += P^T dO and dK += dS^T Q. Each warp drains its 32 x 16 piece of dK or dV through shared memory with a TMA store
// (the first chunk, whose start row is negative, stores with plain 16-byte rows). Warp 15 also issues the TMA loads (3 stages) and the MMAs.
// The per-window bias (log2 domain, key mask folded in as -inf) sits in shared memory for all the samples.
// Slid key window (SPE > 0, slide.cuh): the clipped-window CTAs mask the moved windows' cells and skip their dbias; the last 8 SPE CTAs
// (edge: 128 slid keys x the group's windows) write those windows' dbias and their dK / dV as an fp32 partial that lslide.cu adds.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
#include "slide.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QR = 224, KR = 128, NST = 3;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QR * 64, T_K = 2 * QR * 64, T_V = T_K + KR * 64, T_L = T_V + KR * 64, T_D = T_L + QR * 4;
constexpr int STAGE = 47104;
static_assert(T_D + QR * 4 <= STAGE, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 512 * 32 * 4, O_BAR = O_O + 2 * KR * 64;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int C_S = 0, C_DP = 224, C_DK = 448, C_DV = 480, C_PP = 0, C_PS = 112;
constexpr uint32_t TX = 2 * QR * 64 + 2 * KR * 64 + 2 * QR * 4;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_dkv(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask, __nv_bfloat16* __restrict__ dkv,
                float* __restrict__ dbias, float* __restrict__ spart,
                int ldd, int N, int A, int nwin, int SP, int SPE) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ int sred[32];
  __shared__ int sofs[2][512];                                       // per thread: its dbias offset (-1: none), its edge partial offset (registers: at the cap)
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 1, *dready = bars + NST + 2;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nmain = (int)gridDim.x - 2 * NH * SPE;                  // the slid window's edge CTAs come last (slide.cuh; SPE = 0: none)
  const bool edge = (int)blockIdx.x >= nmain;
  const int sp = edge ? SPE : SP, bx = (int)blockIdx.x - (edge ? nmain : 0);
  const int pair = bx / sp, part = bx - pair * sp;
  const int jc = pair >> 2, h = pair & 3;                            // (an edge CTA: jc = its group)
  const int nr = SPE > 0 ? slide_nreal(kmask, N, sred) : N;
  int wf = 0, kbe = 0;
  const bool live = !edge || slide_group(jc, nr, wf, kbe);
  // an edge CTA: the group's 128 slid keys, its windows from wf (every key sees all of them: band offset 0)
  const int kb = edge ? kbe : 128 * jc - 112, W0 = edge ? wf : 4 * jc - 5, q0 = 32 * W0;
  const int a_lo = (int)((long)A * part / sp), a_hi = (int)((long)A * (part + 1) / sp);
  const int nit = live ? a_hi - a_lo : 0;
  if (edge && bx == 0 && tid == 0) reinterpret_cast<int*>(spart)[2 * A * NH * 64 * KR] = nr;   // n_real for lslide.cu, after the partial
  const bool ctl = warp == 15;

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) mbar_init(&full[i], 1);
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int qdb = edge ? 0 : qd;                                     // the quadrant's first band cell
  const int r = 32 * qd + lane, key = kb + r;
  const int wq = W0 + qdb + qt;                                      // this warp's band quarter = one window, queries 32 wq ..
  const bool kok = key >= 0 && key < N && (kmask == nullptr || kmask[key]);
  const bool wok = wq >= 0 && wq < nwin;
  // this CTA's window (its cells and dbias); the others' cells are masked: a moved window in a clipped-window CTA, an unmoved one in an edge CTA
  const bool wmine = SPE == 0 ? wok : edge ? slide_mine(jc, wq, nr) : wok && !slide_moved(wq, nr);
  const int rr = key - (edge ? kb : 32 * wq - KOFF);                 // the key's column in window wq's 128 keys
  {
    for (int jj = 0; jj < 32; ++jj) {
      float v = 0.f;
      if (wok) v = kok && wmine ? gbias[(((size_t)h * nwin + wq) * WQ + jj) * WK + rr] * LOG2E : -INFINITY;
      sbias[(warp * 32 + jj) * 32 + lane] = v;
    }
    if (tid < 512) {
      sofs[0][tid] = key >= 0 && key < N && wmine ? ((h * nwin + wq) * WQ) * WK + rr : -1;
      sofs[1][tid] = (((jc * A + a_lo) * NH + h) * 64 + (qt < 2 ? 0 : 32) + 16 * (qt & 1)) * KR + r;
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_s = idesc_bf16(128, QR, 0, 0), id_d = idesc_bf16(128, 32, 0, 1);

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_DO, &tm_do, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_L, &tm_l, &full[s], q0, h, a);
    tma_load_3d(st + T_D, &tm_d, &full[s], q0, h, a);
  };
  auto mma_s = [&](int it) {                                         // S^T, dP^T of sample it
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dk = desc_sw64(st + T_K), dq = desc_sw64(st + T_Q), dv = desc_sw64(st + T_V), ddo = desc_sw64(st + T_DO);
    if (elect_one()) {
      umma_ss(tm + C_S, dk, dq, id_s, 0);
      umma_ss(tm + C_DP, dv, ddo, id_s, 0);
      umma_ss(tm + C_S, dk + 2, dq + 2, id_s, 1);
      umma_ss(tm + C_DP, dv + 2, ddo + 2, id_s, 1);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dV += P^T dO, dK += dS^T Q (A = packed bf16 in TMEM)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t ddo = desc_sw64(st + T_DO), dq = desc_sw64(st + T_Q);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < QR / 16; ++ks) {
        const uint64_t boff = (uint64_t)(ks * 1024 >> 4);
        umma_ts(tm + C_DV, tm + C_PP + 8 * ks, ddo + boff, id_d, ks);
        umma_ts(tm + C_DK, tm + C_PS + 8 * ks, dq + boff, id_d, ks);
      }
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST - 1 && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) { mma_s(0); if (elect_one()) tc_commit(sready); __syncwarp(); }
  }

  auto drain = [&](int itd) {                                        // per warp: 32 keys x 16 columns of dK (quarters 0, 1) / dV (2, 3)
    uint32_t v[16];
    const int col = (qt < 2 ? C_DK : C_DV) + 16 * (qt & 1);
    tmem_ld16(tm + ((uint32_t)(32 * qd) << 16) + col, v);
    tmem_wait_ld();
    const float sc = qt < 2 ? QSCALE : 1.f;
    if (edge) {        // fp32 partial [group][A][head][dK 32 | dV 32 channels][128 keys] (lslide.cu adds it): coalesced across the lanes (keys)
      float* d = spart + (size_t)sofs[1][tid] + (size_t)itd * NH * 64 * KR;
#pragma unroll
      for (int j = 0; j < 16; ++j) d[j * KR] = __uint_as_float(v[j]) * sc;
      return;
    }
    uint8_t* ob = sm + O_O + warp * 1024;                             // [32 rows][32 B], no swizzle
    if (lane == 0) tma_store_wait_read0();                            // this warp's previous store has read the tile
    __syncwarp();
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint4 o;
      o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * sc, __uint_as_float(v[8 * g + 1]) * sc);
      o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * sc, __uint_as_float(v[8 * g + 3]) * sc);
      o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * sc, __uint_as_float(v[8 * g + 5]) * sc);
      o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * sc, __uint_as_float(v[8 * g + 7]) * sc);
      *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
    }
    fence_proxy_async();
    __syncwarp();
    const int kr = kb + 32 * qd;
    if (lane == 0) {
      if (kr >= 0) {
        tma_store_3d(&tm_o, su + O_O + warp * 1024, (qt < 2 ? 128 : 256) + h * DH + 16 * (qt & 1), kr, a_lo + itd);
        tma_store_commit();
      }
    }
    if (kr < 0) {                                                    // the first chunk's first quadrants (negative start row)
      const int kk = kb + r;
      if (kk >= 0 && kk < N) {
        __nv_bfloat16* d = dkv + ((size_t)(a_lo + itd) * N + kk) * ldd + (qt < 2 ? 128 : 256) + h * DH + 16 * (qt & 1);
        *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(ob + lane * 32);
        *reinterpret_cast<uint4*>(d + 8) = *reinterpret_cast<const uint4*>(ob + lane * 32 + 16);
      }
    }
  };

  float dbacc[32];
#pragma unroll
  for (int jj = 0; jj < 32; ++jj) dbacc[jj] = 0.f;
  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int x0 = 32 * qdb + 32 * qt;                                 // chunk-relative query column of the band quarter
  const float* sb = sbias + warp * 32 * 32 + lane;
  for (int it = 0; it < nit; ++it) {
    const int s = it % NST;
    const float* sl = reinterpret_cast<const float*>(sm + s * STAGE + T_L);
    const float* sd = reinterpret_cast<const float*>(sm + s * STAGE + T_D);
    mbar_wait(sready, it & 1);                                       // S / dP of this sample (and the previous sample's dV / dK) done
    tc_fence_after();
    if (ctl && lane == 0 && it + NST - 1 < nit) load(it + NST - 1);  // stage of sample it - 1 is free
    uint32_t pw[16], dw[16];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint32_t sv[16], dv[16];
      tmem_ld16(tl + C_S + x0 + 16 * c, sv);
      tmem_ld16(tl + C_DP + x0 + 16 * c, dv);
      tmem_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 16; jj += 2) {
        const int x = x0 + 16 * c + jj, b = 16 * c + jj;
        const float2 l2 = *reinterpret_cast<const float2*>(sl + x), d2 = *reinterpret_cast<const float2*>(sd + x);
        const float p0 = ex2f(fmaf(__uint_as_float(sv[jj]), QSCALE * LOG2E, sb[b * 32]) - l2.x);
        const float p1 = ex2f(fmaf(__uint_as_float(sv[jj + 1]), QSCALE * LOG2E, sb[(b + 1) * 32]) - l2.y);
        const float t0 = p0 * (__uint_as_float(dv[jj]) - d2.x), t1 = p1 * (__uint_as_float(dv[jj + 1]) - d2.y);
        dbacc[b] += t0; dbacc[b + 1] += t1;
        pw[b >> 1] = pack_bf16(p0, p1); dw[b >> 1] = pack_bf16(t0, t1);
      }
    }
    tc_fence_before();
    named_bar_sync(1 + qd, 128);                                     // the quadrant's four warps have read S^T / dP^T
    tc_fence_after();
    tmem_st16(tl + C_PP + x0 / 2, pw);
    tmem_st16(tl + C_PS + x0 / 2, dw);
    if (qt < 3) {                                                    // out-of-band packed columns of this quadrant: [0, 16 qd) U [16 qd + 64, 112)
      const int z = qt < qdb ? 16 * qt : 16 * qt + 64;
      tmem_st16z(tl + C_PP + z);
      tmem_st16z(tl + C_PS + z);
    }
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    if (ctl) {
      mbar_wait(tready, it & 1);
      tc_fence_after();
      mma_d(it);
      if (elect_one()) tc_commit(dready);
      __syncwarp();
      if (it + 1 < nit) {
        mma_s(it + 1);
        if (elect_one()) tc_commit(sready);
        __syncwarp();
      }
    }
    mbar_wait(dready, it & 1);                                       // this sample's dK / dV, while the next S / dP run
    tc_fence_after();
    drain(it);
  }
  if (lane == 0) tma_store_wait0();
  const int dofs = sofs[0][tid];
  if (dofs >= 0) {
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) {
      float* dst = dbias + (size_t)dofs + jj * WK;
      if (sp == 1) *dst = dbacc[jj]; else atomicAdd(dst, dbacc[jj]);
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
#endif
