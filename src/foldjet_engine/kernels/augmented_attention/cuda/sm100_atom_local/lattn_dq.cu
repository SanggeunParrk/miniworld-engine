#if defined(ATOM_HOPPER)
#include "../sm100_atom/sm100.cuh"
using namespace atom90;
// Query-centric Hopper lowering: two 32-query windows retain the union of
// their keys in three TMA stages. Scores/dP/dS remain register fragments.
extern "C" __global__ __launch_bounds__(160, 1)
void local_attn_dq_sm90(const __grid_constant__ CUtensorMap mq,
 const __grid_constant__ CUtensorMap mk,const __grid_constant__ CUtensorMap mv,
 const __grid_constant__ CUtensorMap mdo,const float* __restrict__ bias,
 const uint8_t* __restrict__ mask,const float* __restrict__ lse,
 const float* __restrict__ dd,__nv_bfloat16* __restrict__ out,int ldd,int N,int A) {
 extern __shared__ __align__(1024) uint8_t sm[];
 constexpr int TB=8192,QB=0,DOB=TB,KB=2*TB,VB=5*TB,BAR=8*TB;
 uint64_t* bars=reinterpret_cast<uint64_t*>(sm+BAR);
 const int t=threadIdx.x,lane=t&31,warp=t>>5;
 const int h=blockIdx.y%4,a=blockIdx.y/4,q0=64*blockIdx.x,key0=q0-48;
 if(t==0){for(int j=0;j<4;++j)init(bars+j);init_fence();}
 __syncthreads();
 if(t>=128){
  if(t==128){
   expect(bars,2*TB);load3(sm+QB,&mq,bars,h*32,q0,a);load3(sm+DOB,&mdo,bars,h*32,q0,a);
   #pragma unroll
   for(int j=0;j<3;++j){
    expect(bars+j+1,2*TB);
    load3(sm+KB+j*TB,&mk,bars+j+1,h*32,key0+64*j,a);
    load3(sm+VB+j*TB,&mv,bars+j+1,h*32,key0+64*j,a);
   }
  }
  return;
 }
 const int r0=warp*16+lane/4,c0=(lane&3)*2;
 uint32_t qa[8],doa[8];bwait(bars,0);
 #pragma unroll
 for(int k=0;k<2;++k)
  #pragma unroll
  for(int hi=0;hi<2;++hi)
   #pragma unroll
   for(int rr=0;rr<2;++rr){
    const int ix=k*4+2*hi+rr,offx=off(r0+8*rr,16*k+8*hi+c0);
    qa[ix]=*reinterpret_cast<uint32_t*>(sm+QB+offx);
    doa[ix]=*reinterpret_cast<uint32_t*>(sm+DOB+offx);
   }
 float dq[16]={0.f};
 #pragma unroll
 for(int j=0;j<3;++j){
  bwait(bars+j+1,0);
  const uint8_t* sk=sm+KB+j*TB;const uint8_t* sv=sm+VB+j*TB;
  float ss[32],dp[32];fence();
  mma_rs_n64_first(qa[0],qa[1],qa[2],qa[3],desc(sk),ss);
  mma_rs_n64(qa[4],qa[5],qa[6],qa[7],desc(sk+32),ss,1);
  mma_rs_n64_first(doa[0],doa[1],doa[2],doa[3],desc(sv),dp);
  mma_rs_n64(doa[4],doa[5],doa[6],doa[7],desc(sv+32),dp,1);
  commit();wait();
  uint32_t ds[16];
  #pragma unroll
  for(int i=0;i<8;++i)
   #pragma unroll
   for(int rr=0;rr<2;++rr){
    const int q=q0+r0+8*rr,w=q/32,lo=32*w-48;
    float values[2];
    #pragma unroll
    for(int e=0;e<2;++e){
     const int k=key0+64*j+8*i+c0+e;
     const bool ok=q<N&&k>=0&&k<N&&k>=lo&&k<lo+128&&(!mask||mask[k]);
     float value=0.f;
     if(ok){
      const size_t row=(size_t(a)*4+h)*N+q;
      const float score=fmaf(ss[4*i+2*rr+e],0.17677669529663687f,
        bias[((size_t(h)*(N/32)+w)*32+(q&31))*128+(k-lo)]);
      value=s100::ex2f(score*1.4426950408889634f-lse[row])*(dp[4*i+2*rr+e]-dd[row]);
     }
     values[e]=value;
    }
    ds[4*(i>>1)+2*(i&1)+rr]=pack(values[0],values[1]);
   }
  fence();
  #pragma unroll
  for(int k=0;k<4;++k)pv(dq,ds+4*k,desc(sk+k*2048));
  commit();wait();
 }
 #pragma unroll
 for(int rr=0;rr<2;++rr){
  const int q=q0+r0+8*rr;
  if(q<N)
   #pragma unroll
   for(int i=0;i<4;++i)
    *reinterpret_cast<uint32_t*>(out+(size_t(a)*N+q)*ldd+h*32+8*i+c0)=
      pack(dq[4*i+2*rr]*0.17677669529663687f,dq[4*i+2*rr+1]*0.17677669529663687f);
 }
}
#else
// lattn_dq.cu — dQ of the AF3 windowed atom attention on tcgen05 (sm_100a), the query-centric mirror of lattn_dkv.cu.
//
//   P = exp2(S - LSE)    dP = dO V^T    dS = P (dP - D)    dQ = dS K / sqrt(32)     (bf16 into dq[row, h * 32 + d], row stride ldd)
//
// A CTA owns 128 queries [q0, q0 + 128) (windows 4 c .. 4 c + 3), one head and a range of samples (SP CTAs split the samples); they see
// the 224 keys [q0 - 48, q0 + 176). TMEM lane = query:  S = Q K^T [128 x 224] cols 0..223,  dP = dO V^T cols 224..447,  dQ cols 448..479.
// Warp w (lane quadrant w & 3 = one window, band quarter w >> 2) reads its 32 x 32 band cells, computes dS = exp2(S c + bias - LSE) (dP - D)
// and -- after the quadrant's warps have read -- writes dS as packed bf16 over S (cols 0..111, out-of-band cells zero): the TMEM A operand
// of dQ += dS K (B = the K tile, MN-major). Warps with band quarter < 2 drain 32 x 16 of dQ each through shared memory with a TMA store.
// Warp 16 issues the TMA loads (3 stages, refilled when a sample's MMAs commit) and the MMAs; the window's bias (log2 domain, key mask folded in) stays in shared memory.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QN = 128, KN = 224, NST = 3;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QN * 64, T_K = 2 * QN * 64, T_V = T_K + KN * 64, T_L = T_V + KN * 64, T_D = T_L + QN * 4;
constexpr int STAGE = 46080;
static_assert(T_D + QN * 4 <= STAGE && STAGE % 1024 == 0, "stage");
constexpr int BSTR = 33;                                            // padded bias rows: conflict-free both ways
constexpr int O_B = NST * STAGE, O_O = O_B + 16 * 32 * BSTR * 4, O_BAR = O_O + 16 * 1024;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int C_S = 0, C_DP = 224, C_DQ = 448, C_PS = 0;
constexpr uint32_t TX = 2 * QN * 64 + 2 * KN * 64 + 2 * QN * 4;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_dq(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                 const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                 const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
                 int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 1, *dready = bars + NST + 2, *empty = bars + NST + 3;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) mbar_init(&full[i], 1);
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1);
    for (int i = 0; i < NST; ++i) mbar_init(&empty[i], 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;                                        // this quadrant's window; lane = query qq of it
  if (!ctl) {                                                                  // lane = key offset jj (coalesced rows of the bias), loop over the queries
    const int key = 32 * wq - KOFF + 32 * qt + lane;
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
  const uint32_t id_s = idesc_bf16(128, KN, 0, 0), id_d = idesc_bf16(128, 32, 0, 1);

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
  auto mma_s = [&](int it) {
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_sw64(st + T_Q), dk = desc_sw64(st + T_K), ddo = desc_sw64(st + T_DO), dv = desc_sw64(st + T_V);
    if (elect_one()) {
      umma_ss(tm + C_S, dq, dk, id_s, 0);
      umma_ss(tm + C_DP, ddo, dv, id_s, 0);
      umma_ss(tm + C_S, dq + 2, dk + 2, id_s, 1);
      umma_ss(tm + C_DP, ddo + 2, dv + 2, id_s, 1);
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dQ = dS K (A = packed bf16 dS in TMEM, B = K tile MN-major)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dk = desc_sw64(st + T_K);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KN / 16; ++ks) umma_ts(tm + C_DQ, tm + C_PS + 8 * ks, dk + (uint64_t)(ks * 1024 >> 4), id_d, ks);
      tc_commit(dready);
      tc_commit(&empty[it % NST]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);
      tc_fence_after();
      mma_d(it);                                                     // commits dready and empty[stage of it]
      if (it + 1 < nit) mma_s(it + 1);
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  } else {

  auto drain = [&](int itd) {                                        // warps with qt < 2: 32 queries x 16 columns of dQ
    if (qt >= 2) return;
    uint32_t v[16];
    tmem_ld16(tm + ((uint32_t)(32 * qd) << 16) + C_DQ + 16 * qt, v);
    tmem_wait_ld();
    uint8_t* ob = sm + O_O + warp * 1024;
    if (lane == 0) tma_store_wait_read0();
    __syncwarp();
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint4 o;
      o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * QSCALE, __uint_as_float(v[8 * g + 1]) * QSCALE);
      o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * QSCALE, __uint_as_float(v[8 * g + 3]) * QSCALE);
      o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * QSCALE, __uint_as_float(v[8 * g + 5]) * QSCALE);
      o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * QSCALE, __uint_as_float(v[8 * g + 7]) * QSCALE);
      *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
    }
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) {
      tma_store_3d(&tm_o, su + O_O + warp * 1024, h * DH + 16 * qt, q0 + 32 * qd, a_lo + itd);
      tma_store_commit();
    }
  };

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int x0 = 32 * qd + 32 * qt;                                  // chunk-relative key column of the band quarter
  const float* sb = sbias + warp * 32 * BSTR + lane;
  for (int it = 0; it < nit; ++it) {
    const int s = it % NST;
    mbar_wait(sready, it & 1);
    tc_fence_after();
    const float lse = reinterpret_cast<const float*>(sm + s * STAGE + T_L)[32 * qd + lane];
    const float dd = reinterpret_cast<const float*>(sm + s * STAGE + T_D)[32 * qd + lane];
    uint32_t dw[16];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint32_t sv[16], dv[16];
      tmem_ld16(tl + C_S + x0 + 16 * c, sv);
      tmem_ld16(tl + C_DP + x0 + 16 * c, dv);
      tmem_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 16; jj += 2) {
        const int b = 16 * c + jj;
        const float p0 = ex2f(fmaf(__uint_as_float(sv[jj]), QSCALE * LOG2E, sb[b * BSTR]) - lse);
        const float p1 = ex2f(fmaf(__uint_as_float(sv[jj + 1]), QSCALE * LOG2E, sb[(b + 1) * BSTR]) - lse);
        dw[b >> 1] = pack_bf16(p0 * (__uint_as_float(dv[jj]) - dd), p1 * (__uint_as_float(dv[jj + 1]) - dd));
      }
    }
    tc_fence_before();
    named_bar_sync(1 + qd, 128);
    tc_fence_after();
    tmem_st16(tl + C_PS + x0 / 2, dw);
    if (qt < 3) tmem_st16z(tl + C_PS + (qt < qd ? 16 * qt : 16 * qt + 64));
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    mbar_wait(dready, it & 1);
    tc_fence_after();
    drain(it);
  }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
#endif
