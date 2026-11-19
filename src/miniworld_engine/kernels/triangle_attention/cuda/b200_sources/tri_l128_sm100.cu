// TriangleAttention inference at L = 128 on B200 (sm_100a): the whole update of a pair row in ONE CTA (front + core + tail fused).
//
// At L128 every kernel of the three-kernel chain (tri_front -> triattn_fwd -> tri_tail) runs one 128-row tile a CTA, so each one is
// its own start-up / load / drain latency for ~1 us of work (round 3 parts 3 / 6: the core alone is two thirds skeleton).  Here a CTA
// owns one pair row (b, n) of 128 tokens and does, with the intermediates in shared memory / TMEM only:
//   phase A  y = LN(x) (in place), q | k | v | g = y . W^T (four 128 x HD GEMMs into all 512 TMEM columns), q / k / v drained into the
//            core's operand layouts ([128][32] 64B-swizzled per head), g kept in TMEM (fp32);
//   phase B  per head: S = Q K^T (TMEM), x = S scale log2e + bias log2e, exact row maximum (one key tile: no running maximum, no
//            overflow guard), P = 2^(x - m) (bf16, over S in TMEM), O_h = P V (ts MMA), l = sum P;
//   phase C  u = sigmoid(bf16 g) o bf16(O / l) (the tail's bits), y = x + u . Wo^T, stored in the caller's layout.
// The bias needs every row of the pair, so it is a first, light launch (tri_l128_bias: LN + the four bias dot products, the front's
// arithmetic and bits); the fused kernel is its programmatic dependent and waits for it only before loading its bias tiles.
// Warps: 0 TMA, 1 tcgen05 issue, 2-9 LayerNorm (A) then two softmax groups (B: heads g, g + 2), 10-13 drain (A) / epilogue (C).
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

namespace {
constexpr float L2E = 1.4426950408889634f;
#ifndef L1_P1ST
#define L1_P1ST 1       // 1: the softmax's first pass stores the logits over S (0: the second pass recomputes them from S and the bias: d_pair 128
                        // L128 kernel 23.1 vs 22.1 us a module -- the second pass, MUFU-bound, is the longer one)
#endif
#define TLM(i)
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
#ifndef L1_POLY
#define L1_POLY 4       // this many of a chunk's 16 exponential pairs on the FMA pipe (the softmax's second pass is MUFU-bound)
#endif
__device__ __forceinline__ float2 ex2_poly2(float2 x) {   // triattn_sm100.cu's degree-3 2^x on the FMA pipe, two at a time
  x.x = fmaxf(x.x, -126.f); x.y = fmaxf(x.y, -126.f);
  const float2 t = add2(x, make_float2(12582912.f, 12582912.f));
  const float2 rr = add2(t, make_float2(-12582912.f, -12582912.f));
  const float2 f = add2(x, make_float2(-rr.x, -rr.y));
  float2 pp = fma2(f, make_float2(0.05517097723f, 0.05517097723f), make_float2(0.24260970271f, 0.24260970271f));
  pp = fma2(f, pp, make_float2(0.69326096627f, 0.69326096627f));
  pp = fma2(f, pp, make_float2(0.99992816244f, 0.99992816244f));
  return make_float2(__int_as_float(__float_as_int(pp.x) + (__float_as_int(t.x) << 23)),
                     __int_as_float(__float_as_int(pp.y) + (__float_as_int(t.y) << 23)));
}
__device__ __forceinline__ float sigmoid_t(float x) {   // 0.5 + 0.5 tanh(x / 2): the tail's form
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}
__device__ __forceinline__ void tmem_st32f(uint32_t taddr, const float* v) {
  uint32_t w[8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = __float_as_uint(v[q * 8 + i]);
    tmem_st8(taddr + q * 8, w);
  }
}

template <int CX, int HD> struct LC {
  static constexpr int BM = 128, NH = HD / 32, QT = BM * 32;
  static constexpr int XT = BM * CX, BT = BM * BM, WT = 4 * HD * CX;
  static constexpr int THREADS = 448;
  // shared memory (bf16 elements): R0 = x / y (A), bias stage 0 (B); R1 = bias stage 1; RW = W4 (A), then Q | V | K | Wo (B),
  // then the residual x over Q, u over K, the output staging over V (C)
  static constexpr int R0 = XT > BT ? XT : BT, R1 = BT, RW = WT > 3 * NH * QT + CX * HD ? WT : 3 * NH * QT + CX * HD;
  static constexpr int OFF_Q = 0, OFF_V = NH * QT, OFF_K = 2 * NH * QT, OFF_WO = 3 * NH * QT;
  static constexpr int SMEM = 1024 + (R0 + R1 + RW) * 2 + (6 * CX + NH * BM) * 4 + 512;
  // training: a staging tile pair for the saved g / o of a head ([128][32] 64B-swizzled each), after the barriers
  static constexpr int STG = ((R0 + R1 + RW) * 2 + (6 * CX + NH * BM) * 4 + 512 + 1023) / 1024 * 1024, SMEM_TR = 1024 + STG + 2 * QT * 2;
  static_assert(SMEM_TR <= 232448, "smem (training)");
  static constexpr int COL_S0 = HD == 128 ? 0 : 256, COL_S1 = COL_S0 + 128, COL_O = 2 * HD, COL_G = 3 * HD;   // HD 128: S over q | k
  static constexpr uint32_t ID_P = idesc_bf16(128, HD, 0, 0), ID_S = idesc_bf16(128, 128, 0, 0), ID_O = idesc_bf16(128, 32, 0, 1),
                            ID_Y = idesc_bf16(128, CX, 0, 0);
  // phase B's regions over W4 ([CX / 64][4 HD][64]): Q on q | k weight rows (free once the q | k GEMMs retired), V and Wo on v | g rows
  // or beyond W4 (free once every GEMM retired), K on q | k rows or beyond W4; the residual x fits Q, the output staging V
  static_assert((CX == 64 || CX == 128) && (HD == 64 || HD == 128) && CX <= HD, "tri_l128: d_pair 64 / 128, heads x 32 = 64 / 128");
  static_assert(SMEM <= 232448, "smem");
};

// ---- the pair bias: tri_front_sm100's LayerNorm + bias arithmetic, two threads a row (the same order: the same bits)
template <int CX>
__global__ void __launch_bounds__(256) tri_l128_bias(const __nv_bfloat16* __restrict__ X, const float* __restrict__ LNW,
                                                     const float* __restrict__ LNB, const float* __restrict__ WB, const __nv_bfloat16* __restrict__ WB16,
                                                     const int NHB,
                                                     const bool* __restrict__ MASK, __nv_bfloat16* __restrict__ BIAS, const long R, const int L,
                                                     const int geo, const int BT, const int KL, const float eps) {
  pdl_trigger();
  if (threadIdx.x == 0) TLM(20);
  constexpr int NC = CX / 16, NHM = 4;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, hf = lane >> 4;
  const long row = (long)blockIdx.x * 128 + warp * 16 + (lane & 15);
  if (row >= R) return;
  const long er = row_at(row, L, geo);
  float2 x[NC * 4];
  if (er >= 0) {
    const uint4* p = reinterpret_cast<const uint4*>(X + er * CX + hf * (CX / 2));
#pragma unroll
    for (int c8 = 0; c8 < NC; ++c8) {
      const uint4 u = __ldg(p + c8);
      x[c8 * 4 + 0] = bf2f(u.x); x[c8 * 4 + 1] = bf2f(u.y); x[c8 * 4 + 2] = bf2f(u.z); x[c8 * 4 + 3] = bf2f(u.w);
    }
  } else {
#pragma unroll
    for (int i = 0; i < NC * 4; ++i) x[i] = make_float2(0.f, 0.f);
  }
  float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
  for (int i = 0; i < NC * 4; i += 2) { s0 = add2(s0, x[i]); s1 = add2(s1, x[i + 1]); }
  float sum = s0.x + s0.y + s1.x + s1.y;
  sum += __shfl_xor_sync(0xffffffffu, sum, 16);
  const float mean = sum * (1.f / CX);
  const float2 nm = make_float2(-mean, -mean);
  float2 v0 = make_float2(0.f, 0.f), v1 = v0;
#pragma unroll
  for (int i = 0; i < NC * 4; i += 2) {
    x[i] = add2(x[i], nm); x[i + 1] = add2(x[i + 1], nm);
    v0 = fma2(x[i], x[i], v0); v1 = fma2(x[i + 1], x[i + 1], v1);
  }
  float var = v0.x + v0.y + v1.x + v1.y;
  var += __shfl_xor_sync(0xffffffffu, var, 16);
  const float rstd = rsqrtf(var * (1.f / CX) + eps);
  const float2 rs = make_float2(rstd, rstd);
  float bacc[NHM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c8 = 0; c8 < NC; ++c8) {
    const int c0 = hf * (CX / 2) + c8 * 8;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int c = c0 + 2 * e;
      const float2 yv = fma2(mul2(x[c8 * 4 + e], rs), make_float2(__ldg(LNW + c), __ldg(LNW + c + 1)), make_float2(__ldg(LNB + c), __ldg(LNB + c + 1)));
      const float2 yb = bf2f(pack2(yv.x, yv.y));
#pragma unroll
      for (int h = 0; h < NHM; ++h) {
        const float w0 = h >= NHB ? 0.f : WB != nullptr ? __ldg(WB + h * CX + c) : __bfloat162float(WB16[h * CX + c]);
        const float w1 = h >= NHB ? 0.f : WB != nullptr ? __ldg(WB + h * CX + c + 1) : __bfloat162float(WB16[h * CX + c + 1]);
        bacc[h] = fmaf(yb.x, w0, fmaf(yb.y, w1, bacc[h]));
      }
    }
  }
#pragma unroll
  for (int h = 0; h < NHM; ++h) bacc[h] += __shfl_xor_sync(0xffffffffu, bacc[h], 16);
  if (hf == 0) {
    const int kx = (int)(row % L);
    const long bj = row / L;
    const int j = (int)(bj % L), b = (int)(bj / L);
    const int qi = BT ? kx : j, ki = BT ? j : kx;
    const bool keep = ki < KL && (MASK == nullptr || MASK[(long)b * KL + ki]);
#pragma unroll
    for (int h = 0; h < NHM; ++h)
      if (h < NHB)
        BIAS[(((long)b * NHB + h) * L + qi) * L + ki] = keep ? __float2bfloat16_rn(bacc[h]) : __float2bfloat16_rn(-3.3895313892515355e38f);
  }
  if (threadIdx.x == 0) TLM(21);
}

struct L1Save { __nv_bfloat16* p[5]; };

// GB: the CTAs make the bias rows of their own tokens (the front's LN warps' arithmetic) and meet at a grid barrier (every CTA
// co-resident: B L <= SMs, a cooperative launch) before loading the bias tiles; else the bias comes from tri_l128_bias (PDL wait).
template <int CX, int HD, bool GB, bool TR = false>
__global__ void __launch_bounds__(LC<CX, HD>::THREADS, 1) tri_l128_sm100(
    const int L, const int geo, const float eps, const float scl,   // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap xmap,             // x rows (the caller's layout), box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,             // W4 [4 HD][CX], box (64, 256), 128B swizzle
    const __grid_constant__ CUtensorMap omap,             // Wo [CX][HD], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap bmap,             // bias [B NH L][L], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,             // out rows (the caller's layout), box (32, 32), 64B swizzle
    const __grid_constant__ CUtensorMap xrmap,            // x rows in the output's geometry (the residual), box (64, 128), 128B swizzle
    const int geoy,                                       // the output's geometry (Chai-1's untransposed ending node: geo without the trans bit)
    const __grid_constant__ CUtensorMap sqmap,            // TR: the saved q / k / v [R][HD], box (32, 32), 64B swizzle (a warp's rows of a head)
    const __grid_constant__ CUtensorMap skmap, const __grid_constant__ CUtensorMap svmap,
    const __grid_constant__ CUtensorMap sgmap, const __grid_constant__ CUtensorMap somap,   // TR: the saved g / o, box (32, 32), 64B swizzle
    const float* __restrict__ LNW, const float* __restrict__ LNB,
    const float* __restrict__ WB, const __nv_bfloat16* __restrict__ WB16,   // the bias rows [NH][CX]: fp32, or (WB null) bf16
    const bool* __restrict__ MASK, __nv_bfloat16* __restrict__ BIAS, const int BTR, const int KL,
    const float* __restrict__ GBp, const float* __restrict__ OBp,         // gate bias [HD] / out bias [CX] fp32 (RoseTTAFold3), or null
    const int GROUND,                                                      // 1: g rounded to bf16 before the sigmoid (tri_tail), 0: fp32 (tri_tailg)
    unsigned long long* __restrict__ GBAR,     // GB: [256] arrival counters, slot gridDim.x (monotonic: a launch waits for the next multiple)
    const L1Save SAVE,                                  // TR: q, k, v, g, o [R][HD] bf16 (the chain's saved intermediates, kernel rows)
    float* __restrict__ LSEo,                           // TR: [B, L, NH, 128] base-2 log-sum-exp (the core's)
    const __nv_bfloat16* __restrict__ DROP,             // TR: [B][Lr][CX] dropout scale or null (tri_tail's)
    float2* __restrict__ STATS) {                      // TR: (mean, rstd) [R] for the wide backward, or null
  using T = LC<CX, HD>;
  constexpr int BM = T::BM, NH = T::NH, QT = T::QT, XT = T::XT, BT = T::BT;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sR0 = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sR1 = sR0 + T::R0;
  __nv_bfloat16* sW = sR1 + T::R1;
  __nv_bfloat16* sQ = sW + T::OFF_Q;
  __nv_bfloat16* sV = sW + T::OFF_V;
  __nv_bfloat16* sK = sW + T::OFF_K;
  __nv_bfloat16* sWo = sW + T::OFF_WO;
  float* sP = reinterpret_cast<float*>(sW + T::RW);      // lnw [CX] | lnb [CX] | Wb [4][CX]
  float* sInv = sP + 6 * CX;                             // [NH][128] 1 / l
  uint64_t* bars = reinterpret_cast<uint64_t*>(sInv + NH * BM);
  uint64_t *wf = bars, *xf = bars + 1, *yf = bars + 2, *af0 = bars + 3, *af1 = bars + 4, *qkvf = bars + 5, *wof = bars + 6;
  uint64_t *bfb = bars + 7, *beb = bars + 9, *sf = bars + 11, *pf = bars + 13;   // [2] each
  uint64_t *qkd = bars + 15, *xrf = bars + 16, *ofin = bars + 17, *uf = bars + 18, *accf = bars + 20;   // uf [2]: u halves
  uint64_t *bw = bars + 21, *vf = bars + 22, *ofh = bars + 23, *lf = bars + 23 + NH;                     // ofh, lf [NH]
  uint64_t* wf1 = bars + 23 + 2 * NH;                    // wf: Wq | Wk landed, wf1: Wv | Wg
  uint64_t* qsd = bars + 24 + 2 * NH;                    // TR: the q / k / v save stores have read their tiles (4 epilogue warps)
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 25 + 2 * NH);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int t = blockIdx.x, b = t / L;
  if (tid == 0) {
    TLM(0);
    bar_init(wf, 1); bar_init(wf1, 1); bar_init(xf, 1); bar_init(yf, 8); bar_init(af0, 1); bar_init(af1, 1); bar_init(qkvf, 4); bar_init(wof, 1);
    for (int i = 0; i < 2; ++i) { bar_init(bfb + i, 1); bar_init(beb + i, 4); bar_init(sf + i, 1); bar_init(pf + i, 4); bar_init(uf + i, 4); }
    bar_init(qkd, 1); bar_init(xrf, 1); bar_init(ofin, 1); bar_init(accf, 1); bar_init(bw, 8); bar_init(vf, 4); bar_init(qsd, 4);
    for (int h = 0; h < NH; ++h) { bar_init(ofh + h, 1); bar_init(lf + h, 4); }
    bar_init_fence();
    // the x tile and the weights first (before the block-wide set-up): their HBM latency is the head of the chain
    expect_tx(xf, XT * 2);
    for (int kh = 0; kh < CX / 64; ++kh) load_rows(&xmap, sR0 + kh * BM * 64, xf, kh * 64, t * BM, L, geo);
    for (int p = 0; p < 2; ++p) {                        // Wq | Wk first: the q | k GEMMs start without waiting for Wv | Wg
      expect_tx(p ? wf1 : wf, 2 * HD * CX * 2);
      for (int kh = 0; kh < CX / 64; ++kh) load_2d(&wmap, sW + kh * 4 * HD * 64 + p * 2 * HD * 64, p ? wf1 : wf, kh * 64, p * 2 * HD);
    }
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0) TLM(1);
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      unsigned long long target = 0;
      if (GB) {                                         // this CTA's bias rows are written (bw: every LN warp fenced them): arrive
        wait(bw, 0);
        __threadfence();
        const unsigned long long old = atomicAdd(GBAR + gridDim.x, 1ull);
        target = (old / gridDim.x + 1) * gridDim.x;
      }
      wait(af1, 0);                                     // every projection GEMM retired: W4's v | g rows and the y tile are free
      TLM(4);
      expect_tx(wof, CX * HD * 2);
      for (int kh = 0; kh < HD / 64; ++kh) load_2d(&omap, sWo + kh * CX * 64, wof, kh * 64, 0);
      if (GB) {
        unsigned long long now;
        do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(now) : "l"(GBAR + gridDim.x) : "memory"); } while (now < target);
        asm volatile("fence.proxy.async.global;" ::: "memory");
      } else {
        pdl_wait();                                     // the bias launch complete
      }
      TLM(5);
      for (int h = 0; h < NH; ++h) {
        const int st = h & 1;
        if (h >= 2) wait(beb + st, ((h >> 1) - 1) & 1);
        __nv_bfloat16* stg = st ? sR1 : sR0;
        expect_tx(bfb + st, BT * 2);
        for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, stg + hh * BM * 64, bfb + st, hh * 64, (b * NH + h) * L);
      }
      wait(qkd, 0);                                     // every QK^T retired: the Q region takes the residual x
      if constexpr (TR) wait(qsd, 0);
      TLM(10);
      expect_tx(xrf, XT * 2);
      for (int kh = 0; kh < CX / 64; ++kh) load_rows(&xrmap, sQ + kh * BM * 64, xrf, kh * 64, t * BM, L, geoy);
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      TLM(2);
      wait(yf, 0);
      TLM(3);
      tc_fence_after();
#pragma unroll
      for (int ten = 0; ten < 4; ++ten) {
#pragma unroll
        for (int ks = 0; ks < CX / 16; ++ks) {
          const int kh = ks >> 2, kk = (ks & 3) * 16;
          mma_ss(tmem + ten * HD, desc_k128(sR0 + kh * BM * 64 + kk), desc_k128(sW + kh * 4 * HD * 64 + ten * HD * 64 + kk), T::ID_P, ks ? 1u : 0u);
        }
        if (ten == 1) mma_commit(af0);
        if (ten == 1) wait(wf1, 0);
      }
      mma_commit(af1);
      wait(qkvf, 0);                                    // Q and K drained (V: vf, before the first PV)
      TLM(6);
      tc_fence_after();
      auto qk = [&](int h) {
        const int st = h & 1;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
          mma_ss(tmem + (st ? T::COL_S1 : T::COL_S0), desc_k64(sQ + h * QT + ks * 16), desc_k64(sK + h * QT + ks * 16), T::ID_S, ks ? 1u : 0u);
        mma_commit(sf + st);
        if (h == NH - 1) mma_commit(qkd);
      };
      qk(0);
      if (NH > 1) qk(1);
      for (int h = 0; h < NH; ++h) {
        const int st = h & 1;
        wait(pf + st, (h >> 1) & 1);
        TLM(7 + (h < 2 ? h : 2));
        if (h == 0) wait(vf, 0);
        tc_fence_after();
        const uint32_t sb = tmem + (st ? T::COL_S1 : T::COL_S0);
#pragma unroll
        for (int ks = 0; ks < BM / 16; ++ks)
          mma_ts(tmem + T::COL_O + h * 32, sb + ks * 8, sdesc(sa(sV + h * QT + ks * 16 * 32), QT * 2, 512, 4), T::ID_O, ks ? 1u : 0u);
        mma_commit(ofh + h);
        if (h + 2 < NH) qk(h + 2);                      // its S buffer: PV(h) is ahead of it in the tensor pipe
      }
      mma_commit(ofin);
      TLM(11);
      wait(wof, 0);
#pragma unroll
      for (int ks = 0; ks < HD / 16; ++ks) {
        const int kh = ks >> 2, kk = (ks & 3) * 16;
        if ((ks & 3) == 0) { wait(uf + kh, 0); tc_fence_after(); }   // u half kh (heads 2 kh, 2 kh + 1) written
        if (ks == 0) TLM(12);
        mma_ss(tmem + 0, desc_k128(sK + kh * BM * 64 + kk), desc_k128(sWo + kh * CX * 64 + kk), T::ID_Y, ks ? 1u : 0u);
      }
      mma_commit(accf);
    }
  } else if (warp < 10) {
    {  // ---- phase A: LayerNorm in place (tri_front_sm100's arithmetic), two threads a row
      constexpr int NC = CX / 16;
      const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
      auto xoff = [&](int c8) {
        const int c = hf * (CX / 2) + c8 * 8;
        return (c >> 6) * BM * 64 + r * 64 + ((((c & 63) >> 3) ^ (r & 7)) << 3);
      };
      for (int i = tid - 64; i < (GB ? 6 : 2) * CX; i += 256) {
        const int j = i - 2 * CX;
        sP[i] = i < CX ? LNW[i] : i < 2 * CX ? LNB[i - CX] : j >= NH * CX ? 0.f : WB != nullptr ? WB[j] : __bfloat162float(WB16[j]);
      }
      named_sync(2, 256);                              // the LN parameters (loaded while x / W are in flight)
      wait(xf, 0);
      __nv_bfloat16* xt = sR0;
      float2 x[NC * 4];
#pragma unroll
      for (int c8 = 0; c8 < NC; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xt + xoff(c8));
        x[c8 * 4 + 0] = bf2f(u.x); x[c8 * 4 + 1] = bf2f(u.y); x[c8 * 4 + 2] = bf2f(u.z); x[c8 * 4 + 3] = bf2f(u.w);
      }
      float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
      for (int i = 0; i < NC * 4; i += 2) { s0 = add2(s0, x[i]); s1 = add2(s1, x[i + 1]); }
      float sum = s0.x + s0.y + s1.x + s1.y;
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      const float mean = sum * (1.f / CX);
      const float2 nm = make_float2(-mean, -mean);
      float2 v0 = make_float2(0.f, 0.f), v1 = v0;
#pragma unroll
      for (int i = 0; i < NC * 4; i += 2) {
        x[i] = add2(x[i], nm); x[i + 1] = add2(x[i + 1], nm);
        v0 = fma2(x[i], x[i], v0); v1 = fma2(x[i + 1], x[i + 1], v1);
      }
      float var = v0.x + v0.y + v1.x + v1.y;
      var += __shfl_xor_sync(0xffffffffu, var, 16);
      const float rstd = rsqrtf(var * (1.f / CX) + eps);
      if constexpr (TR) if (STATS != nullptr && hf == 0) STATS[(long)t * BM + r] = make_float2(mean, rstd);
      const float2 rs = make_float2(rstd, rstd);
      float bacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c8 = 0; c8 < NC; ++c8) {
        const int c0 = hf * (CX / 2) + c8 * 8;
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = c0 + 2 * e;
          const float2 yv = fma2(mul2(x[c8 * 4 + e], rs), make_float2(sP[c], sP[c + 1]), make_float2(sP[CX + c], sP[CX + c + 1]));
          o[e] = pack2(yv.x, yv.y);
          if (GB) {
            const float2 yb = bf2f(o[e]);
#pragma unroll
            for (int h = 0; h < 4; ++h) bacc[h] = fmaf(yb.x, sP[2 * CX + h * CX + c], fmaf(yb.y, sP[2 * CX + h * CX + c + 1], bacc[h]));
          }
        }
        *reinterpret_cast<uint4*>(xt + xoff(c8)) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      fence_proxy_async();                             // y for the projection GEMMs first, the bias rows after
      __syncwarp();
      if (lane == 0) arrive(yf);
      if (GB) {                                        // the bias rows of this tile's tokens (tri_front_sm100's bias drain)
#pragma unroll
        for (int h = 0; h < 4; ++h) bacc[h] += __shfl_xor_sync(0xffffffffu, bacc[h], 16);
        if (hf == 0) {
          const long rw = (long)t * BM + r;
          const int kx = (int)(rw % L);
          const long bj = rw / L;
          const int j = (int)(bj % L), bb = (int)(bj / L);
          const int qi = BTR ? kx : j, ki = BTR ? j : kx;
          const bool keep = ki < KL && (MASK == nullptr || MASK[(long)bb * KL + ki]);
#pragma unroll
          for (int h = 0; h < NH; ++h)
            BIAS[(((long)bb * NH + h) * L + qi) * L + ki] = keep ? __float2bfloat16_rn(bacc[h]) : __float2bfloat16_rn(-3.3895313892515355e38f);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncwarp();
        if (lane == 0) arrive(bw);
      }
    }
    // ---- phase B: softmax group grp (warps 2-5 / 6-9) takes heads grp, grp + 2 on S buffer grp
    const int grp = (warp - 2) >> 2, qd = warp & 3, row = qd * 32 + lane;
    const float2 sc2 = make_float2(scl, scl), l2 = make_float2(L2E, L2E);
    for (int h = grp; h < NH; h += 2) {
      const int st = h & 1;
      const uint32_t par = (h >> 1) & 1;
      const uint32_t sb = tmem_at(tmem + (st ? T::COL_S1 : T::COL_S0), qd * 32, 0);
      const __nv_bfloat16* brow = (st ? sR1 : sR0) + row * 64;
      wait(bfb + st, par);
      if (warp == 2 && lane == 0) TLM(h == 0 ? 16 : 20);
      wait(sf + st, par);
      tc_fence_after();
      if (warp == 2 && lane == 0) TLM(h == 0 ? 17 : 21);
      float m0 = -INFINITY, m1 = -INFINITY;
      {
        float s[2][32];
        tmem_ld32(sb, s[0]);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          tmem_wait_ld();
          if (c + 1 < 4) tmem_ld32(sb + (c + 1) * 32, s[(c + 1) & 1]);
          float* sc = s[c & 1];
          const __nv_bfloat16* bp = brow + (c >> 1) * BM * 64;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint4 u = *reinterpret_cast<const uint4*>(bp + ((((c & 1) * 4 + e) ^ (row & 7)) << 3));
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
              const int i2 = e * 8 + k2 * 2;
              const float2 xx = fma2(bf2f(w[k2]), l2, mul2(make_float2(sc[i2], sc[i2 + 1]), sc2));
              sc[i2] = xx.x; sc[i2 + 1] = xx.y;
              m0 = fmaxf(m0, xx.x); m1 = fmaxf(m1, xx.y);
            }
          }
          if (L1_P1ST) tmem_st32f(sb + c * 32, sc);       // the logits over S (else pass 2 recomputes them)
        }
      }
      if (L1_P1ST) {
        fence_proxy_async();                             // the bias stage's reads before its release to the TMA warp
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(beb + st);
        tc_fence_after();
      }
      if (warp == 2 && lane == 0) TLM(h == 0 ? 18 : 22);
      float mr = fmaxf(m0, m1);
      if (!(mr > -1e30f)) mr = 0.f;                      // every key masked: -inf - -inf would be NaN (l = 0 as in the core)
      const float2 nm = make_float2(-mr, -mr);
      float2 acc = make_float2(0.f, 0.f);
      {
        float s[2][32];
        tmem_ld32(sb, s[0]);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          tmem_wait_ld();
          if (c + 1 < 4) tmem_ld32(sb + (c + 1) * 32, s[(c + 1) & 1]);
          float* sc = s[c & 1];
          if (!L1_P1ST) {
            const __nv_bfloat16* bp = brow + (c >> 1) * BM * 64;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint4 u = *reinterpret_cast<const uint4*>(bp + ((((c & 1) * 4 + e) ^ (row & 7)) << 3));
              const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
              for (int k2 = 0; k2 < 4; ++k2) {
                const int i2 = e * 8 + k2 * 2;
                const float2 xx = fma2(bf2f(w[k2]), l2, fma2(make_float2(sc[i2], sc[i2 + 1]), sc2, nm));
                sc[i2] = xx.x; sc[i2 + 1] = xx.y;
              }
            }
          }
          uint32_t pk[16];
#pragma unroll
          for (int k2 = 0; k2 < 16; ++k2) {
            const float2 xx = L1_P1ST ? add2(make_float2(sc[k2 * 2], sc[k2 * 2 + 1]), nm) : make_float2(sc[k2 * 2], sc[k2 * 2 + 1]);
            const float2 pv = (L1_POLY && ((k2 * L1_POLY) / 16 != ((k2 + 1) * L1_POLY) / 16)) ? ex2_poly2(xx) : make_float2(ex2f(xx.x), ex2f(xx.y));
            acc = add2(acc, pv);
            pk[k2] = pack2(pv.x, pv.y);
          }
          tmem_st8(sb + c * 16, pk);                     // P over S: chunk c's 16 columns cover logits already read
          tmem_st8(sb + c * 16 + 8, pk + 8);
        }
      }
      if (warp == 2 && lane == 0) TLM(h == 0 ? 19 : 23);
      if (!L1_P1ST) fence_proxy_async();                 // the bias stage's reads before its release to the TMA warp
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { arrive(pf + st); if (!L1_P1ST) arrive(beb + st); }
      sInv[h * BM + row] = 1.f / (acc.x + acc.y);
      if constexpr (TR) LSEo[((long)t * NH + h) * BM + row] = mr + __log2f(acc.x + acc.y);
      __syncwarp();
      if (lane == 0) arrive(lf + h);
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;
    auto put64 = [&](__nv_bfloat16* tile, const uint32_t* w) {   // row r of a [128][32] 64B-swizzled tile
      unsigned char* rowp = reinterpret_cast<unsigned char*>(tile) + r * 64;
#pragma unroll
      for (int c = 0; c < 4; ++c)
        *reinterpret_cast<uint4*>(rowp + ((c ^ ((r >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
    };
    auto save = [&](int ten, int h, const uint32_t* w) {   // TR: row r, head h of saved tensor ten (16-byte stores, the row's 64 bytes)
      uint4* p = reinterpret_cast<uint4*>(SAVE.p[ten] + ((long)t * BM + r) * HD + h * 32);
#pragma unroll
      for (int c = 0; c < 4; ++c) p[c] = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
    };
    auto drain = [&](int ten, __nv_bfloat16* dst) {
      float v[2][32];
      tmem_ld32(tmem_at(tmem + ten * HD, qq * 32, 0), v[0]);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        tmem_wait_ld();
        if (h + 1 < NH) tmem_ld32(tmem_at(tmem + ten * HD + (h + 1) * 32, qq * 32, 0), v[(h + 1) & 1]);
        uint32_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = pack2(v[h & 1][2 * i], v[h & 1][2 * i + 1]);
        put64(dst + h * QT, w);
      }
    };
    // ---- phase A: q, k (after their GEMMs), v (after every GEMM) into the core's operand tiles
    wait(af0, 0);
    tc_fence_after();
    drain(0, sQ);
    drain(1, sK);
    fence_proxy_async();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) arrive(qkvf);
    if constexpr (TR) if (lane == 0) {          // the saved q, k: this warp's 32 rows of every head, by TMA from the operand tiles
      for (int h = 0; h < NH; ++h) {
        store_2d(&sqmap, sQ + h * QT + qq * 32 * 32, h * 32, t * BM + qq * 32);
        store_2d(&skmap, sK + h * QT + qq * 32 * 32, h * 32, t * BM + qq * 32);
      }
      bulk_commit();
    }
    wait(af1, 0);
    tc_fence_after();
    drain(2, sV);
    fence_proxy_async();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) arrive(vf);
    if constexpr (TR) if (lane == 0) {
      for (int h = 0; h < NH; ++h) store_2d(&svmap, sV + h * QT + qq * 32 * 32, h * 32, t * BM + qq * 32);
      bulk_commit();
      bulk_wait_read<0>();                       // before the residual x (over Q) and u (over K) take the tiles
      arrive(qsd);
    }
    if constexpr (TR) __syncwarp();
    // ---- phase C: u = sigmoid(g) o (O / l) over K (the out-projection's A tile, [HD / 64][128][64] 128B-swizzled)
#pragma unroll 1
    for (int h = 0; h < NH; ++h) {             // head h as soon as its PV retired and its row sums are written
      wait(ofh + h, 0);
      wait(lf + h, 0);
      tc_fence_after();
      float o[32], g[32];
      tmem_ld32(tmem_at(tmem + T::COL_O + h * 32, qq * 32, 0), o);
      tmem_ld32(tmem_at(tmem + T::COL_G + h * 32, qq * 32, 0), g);
      tmem_wait_ld();
      const float inv = sInv[h * BM + r];
      uint32_t w[16];
      if constexpr (TR) {
        uint32_t gw[16], ow[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) { gw[i] = pack2(g[2 * i], g[2 * i + 1]); ow[i] = pack2(o[2 * i] * inv, o[2 * i + 1] * inv); }
        __nv_bfloat16* sg = reinterpret_cast<__nv_bfloat16*>(smb + T::STG);   // g | o staging of this head (the previous head's stores read)
        if (h > 0) { if (lane == 0) bulk_wait_read<0>(); __syncwarp(); }
        put64(sg, gw);
        put64(sg + QT, ow);
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          store_2d(&sgmap, sg + qq * 32 * 32, h * 32, t * BM + qq * 32);
          store_2d(&somap, sg + QT + qq * 32 * 32, h * 32, t * BM + qq * 32);
          bulk_commit();
        }
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float2 gv = make_float2(g[2 * i], g[2 * i + 1]);
        if (GBp != nullptr) { gv.x += __ldg(GBp + h * 32 + 2 * i); gv.y += __ldg(GBp + h * 32 + 2 * i + 1); }
        if (GROUND) gv = bf2f(pack2(gv.x, gv.y));
        const float2 ov = bf2f(pack2(o[2 * i] * inv, o[2 * i + 1] * inv));
        w[i] = pack2(sigmoid_t(gv.x) * ov.x, sigmoid_t(gv.y) * ov.y);
      }
      __nv_bfloat16* rowp = sK + ((h * 32) >> 6) * BM * 64 + r * 64;
#pragma unroll
      for (int c = 0; c < 4; ++c)
        *reinterpret_cast<uint4*>(rowp + (((((h * 32) & 63) >> 3) + c) ^ (r & 7)) * 8) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
      if (h % 2 == 1 || h == NH - 1) {            // a 64-channel half of u complete: its out-projection K steps may go
        fence_proxy_async();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(uf + (h >> 1));
      }
    }
    // ---- y = x + u . Wo^T -> staging over V -> TMA stores in the caller's layout (a warp's 32 rows, a chunk as soon as it is packed;
    // the residual words are read before the wait for the out-projection, the TMEM loads double-buffered)
    wait(xrf, 0);
    uint32_t xw[CX / 32][16];
#pragma unroll
    for (int ch = 0; ch < CX / 32; ++ch) {
      const __nv_bfloat16* xr = sQ + (ch >> 1) * BM * 64 + r * 64;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((((ch & 1) * 4 + c) ^ (r & 7)) << 3));
        xw[ch][c * 4] = u.x; xw[ch][c * 4 + 1] = u.y; xw[ch][c * 4 + 2] = u.z; xw[ch][c * 4 + 3] = u.w;
      }
    }
    const __nv_bfloat16* drp = nullptr;
    if constexpr (TR) if (DROP != nullptr) drp = DROP + drop_row((long)t * BM + r, L, geo, false) * CX;
    wait(accf, 0);
    if (warp == 10 && lane == 0) TLM(13);
    tc_fence_after();
    float v[2][32];
    tmem_ld32(tmem_at(tmem, qq * 32, 0), v[0]);
#pragma unroll
    for (int ch = 0; ch < CX / 32; ++ch) {
      uint32_t dw[16];
      if constexpr (TR) if (drp != nullptr) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { const uint4 u = __ldg(reinterpret_cast<const uint4*>(drp + ch * 32) + c); dw[c * 4] = u.x; dw[c * 4 + 1] = u.y; dw[c * 4 + 2] = u.z; dw[c * 4 + 3] = u.w; }
      }
      tmem_wait_ld();
      if (ch + 1 < CX / 32) tmem_ld32(tmem_at(tmem + (ch + 1) * 32, qq * 32, 0), v[(ch + 1) & 1]);
      uint32_t w[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float a0 = v[ch & 1][2 * i], a1 = v[ch & 1][2 * i + 1];
        if (OBp != nullptr) { a0 += __ldg(OBp + ch * 32 + 2 * i); a1 += __ldg(OBp + ch * 32 + 2 * i + 1); }
        if constexpr (TR) if (drp != nullptr) { const float2 sv = bf2f(dw[i]); a0 *= sv.x; a1 *= sv.y; }
        const float2 xv = bf2f(xw[ch][i]);
        w[i] = pack2(a0 + xv.x, a1 + xv.y);
      }
      put64(sV + ch * QT, w);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) { store_rows(&ymap, sV + ch * QT + qq * 32 * 32, ch * 32, t * BM + qq * 32, L, geoy); bulk_commit(); }
    }
    if (lane == 0) {
      bulk_wait_read<0>();
      if (warp == 10) TLM(14);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TLM(15);
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

inline bool gb_enabled() {   // MINIWORLD_TRIATTN_B200_L128_GB=0: the separate bias launch always (A/B)
  static const int on = [] { const char* e = std::getenv("MINIWORLD_TRIATTN_B200_L128_GB"); return (e == nullptr || e[0] != '0') ? 1 : 0; }();
  return on != 0;
}

template <int CX, int HD>
static torch::Tensor launch_l128(const torch::Tensor& x, const torch::Tensor& lw, const torch::Tensor& lb, double eps, const torch::Tensor& W,
                                 const float* wb, const __nv_bfloat16* wb16, const torch::Tensor& wo, const float* gbp, const float* obp,
                                 bool ground, const torch::Tensor& mask, long B, long L, int geo, int geoy, bool bt, double scale,
                                 const torch::Tensor& out_, std::vector<torch::Tensor>* tro = nullptr, const torch::Tensor* drop = nullptr,
                                 float2* stp = nullptr) {
  using T = LC<CX, HD>;
  constexpr int NH = HD / 32;
  const long R = B * L * L, Lr = geo_rows_len(L, geo), Rr = B * Lr * Lr;
  TORCH_CHECK(L == 128, "tri_l128: L = 128");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() == Rr * CX, "x: [B, Lr, Lr, CX] bf16");
  TORCH_CHECK(W.scalar_type() == torch::kBFloat16 && W.is_contiguous() && W.size(0) >= 4 * HD && W.size(1) == CX, "W [>= 4 HD, CX] bf16");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == CX * HD, "wo [CX, HD] bf16");
  const bool given = out_.defined() && out_.numel() > 0;
  if (given) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == Rr * CX, "out: like x");
  auto out = given ? out_.view({Rr, CX}) : torch::empty({Rr, CX}, x.options());
  auto bias = torch::empty({B, NH, L, L}, x.options());
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * Lr, "mask [B, Lr]"); mp = mk.data_ptr<bool>(); }
  const float* lwp = lw.data_ptr<float>();
  const float* lbp = lb.data_ptr<float>();
  auto* bp = reinterpret_cast<__nv_bfloat16*>(bias.data_ptr<at::BFloat16>());
  auto st = at::cuda::getCurrentCUDAStream();
  const bool gb = (B * L <= num_sms(x.device().index())) && gb_enabled();   // every CTA co-resident: the bias behind a grid barrier
  if (!gb) {
    tri_l128_bias<CX><<<(unsigned)(R / 128), 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), lwp, lbp, wb, wb16,
                                                            NH, mp, bp, R, (int)L, geo, bt ? 1 : 0, (int)Lr, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  CUtensorMap xm = make_rows_map(x.data_ptr(), CX, R, CX, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap wm = make_map<2>(W.data_ptr(), {(uint64_t)CX, (uint64_t)(4 * HD)}, {(uint64_t)CX}, {64, (uint32_t)(2 * HD)}, CU_TENSOR_MAP_SWIZZLE_128B, "W");
  CUtensorMap om = make_map<2>(wo.data_ptr(), {(uint64_t)HD, (uint64_t)CX}, {(uint64_t)HD}, {64, (uint32_t)CX}, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  CUtensorMap bm = make_map<2>(bias.data_ptr(), {(uint64_t)L, (uint64_t)(B * NH * L)}, {(uint64_t)L}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  CUtensorMap ym = make_rows_map(out.data_ptr(), CX, R, CX, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "out", L, geoy);   // a warp's 32 rows
  CUtensorMap xrm = geoy == geo ? xm : make_rows_map(x.data_ptr(), CX, R, CX, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geoy);
  // training (tro): the saved intermediates q | k | v | g | o [5][R][HD], lse [B, L, NH, 128] fp32, the bias, and the dropout scale
  L1Save sv = {};
  CUtensorMap sqm = xm, skm = xm, svm = xm, sgm = xm, som = xm;   // (inference: unused)
  float* lsp = nullptr;
  const __nv_bfloat16* dp = nullptr;
  if (tro != nullptr) {
    std::vector<torch::Tensor> save;
    for (int i = 0; i < 5; ++i) { save.push_back(torch::empty({R, (long)HD}, x.options())); sv.p[i] = reinterpret_cast<__nv_bfloat16*>(save[i].data_ptr<at::BFloat16>()); }
    auto lse = torch::empty({B, L, (long)NH, L}, x.options().dtype(torch::kFloat32));
    lsp = lse.data_ptr<float>();
    if (drop != nullptr && drop->numel() > 0) {
      TORCH_CHECK(drop->scalar_type() == torch::kBFloat16 && drop->is_contiguous() && drop->numel() == B * Lr * CX, "drop: [B, Lr, CX] bf16");
      dp = reinterpret_cast<const __nv_bfloat16*>(drop->data_ptr<at::BFloat16>());
    }
    *tro = save; tro->push_back(lse); tro->push_back(bias);
    sqm = make_map<2>(save[0].data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_64B, "q save");
    skm = make_map<2>(save[1].data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_64B, "k save");
    svm = make_map<2>(save[2].data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_64B, "v save");
    sgm = make_map<2>(save[3].data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_64B, "g save");
    som = make_map<2>(save[4].data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_64B, "o save");
  }
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_l128_sm100<CX, HD, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_TR));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_l128_sm100<CX, HD, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_TR));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_l128_sm100<CX, HD, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_l128_sm100<CX, HD, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    attr = true;
  }
  static torch::Tensor ctr[64];
  torch::Tensor& gbar = ctr[x.device().index()];
  if (!gbar.defined()) gbar = torch::zeros({256}, x.options().dtype(torch::kInt64));
  auto* gp = reinterpret_cast<unsigned long long*>(gbar.data_ptr<int64_t>());
  const float scl = (float)(scale * L2E);
  if (gb) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.gridDim = dim3((unsigned)(B * L)); cfg.blockDim = dim3(T::THREADS); cfg.dynamicSmemBytes = tro != nullptr ? T::SMEM_TR : T::SMEM; cfg.stream = st;
    cfg.attrs = at; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tro != nullptr ? tri_l128_sm100<CX, HD, true, true> : tri_l128_sm100<CX, HD, true, false>, (int)L, geo, (float)eps, scl, xm, wm, om, bm, ym, xrm, geoy, sqm, skm, svm, sgm, som, lwp, lbp, wb, wb16,
                                      (const bool*)mp, bp, bt ? 1 : 0, (int)Lr, gbp, obp, ground ? 1 : 0, gp, sv, lsp, dp, stp));
  } else {
    launch_pdl(tro != nullptr ? tri_l128_sm100<CX, HD, false, true> : tri_l128_sm100<CX, HD, false, false>, dim3((unsigned)(B * L)), dim3(T::THREADS), tro != nullptr ? T::SMEM_TR : T::SMEM, st, (int)L, geo, (float)eps, scl, xm, wm, om,
               bm, ym, xrm, geoy, sqm, skm, svm, sgm, som, lwp, lbp, wb, wb16, (const bool*)mp, bp, bt ? 1 : 0, (int)Lr, gbp, obp, ground ? 1 : 0, gp, sv, lsp, dp, stp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

static const float* opt_f32(const torch::Tensor& t, long n, const char* what) {
  if (!t.defined() || t.numel() == 0) return nullptr;
  TORCH_CHECK(t.scalar_type() == torch::kFloat32 && t.is_contiguous() && t.numel() == n, what, ": fp32 contiguous");
  return t.data_ptr<float>();
}

// TriangleAttention inference at L = 128 (whole update in one CTA a pair row).  x the caller's [B, Lr, Lr, CX] bf16 (Lr <= 128);
// lnw / lnb fp32 [CX]; W [Wq; Wk; Wv; Wg (; ...)] [>= 4 HD, CX] bf16 (HD = heads x 32: 16-channel heads zero-padded); the bias rows wb fp32
// [heads, CX] or (wb empty) bf16 rows 4 HD .. of W; wo [CX, HD] bf16; gb fp32 [HD] / ob fp32 [CX] or empty; ground: g rounded to bf16
// before the sigmoid (tri_tail's bits; else fp32 as tri_tailg); mask [B, Lr] bool or empty -> y in the caller's layout (into out if given)
torch::Tensor tri_l128(torch::Tensor x, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor W, torch::Tensor wb, torch::Tensor wo,
                       torch::Tensor gb, torch::Tensor ob, bool ground, int64_t HD, torch::Tensor mask, int64_t B, int64_t L, bool trans, bool trans_out,
                       bool bt, int64_t lr, double scale, torch::Tensor out) {
  const long CX = x.size(-1);
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous();
  const int geo = rows_geo(trans, L, lr), geoy = rows_geo(trans_out, L, lr);
  const float* wbp = nullptr;
  const __nv_bfloat16* wb16 = nullptr;
  if (wb.defined() && wb.numel() > 0) {
    TORCH_CHECK(wb.scalar_type() == torch::kFloat32 && wb.is_contiguous() && wb.numel() == (HD / 32) * CX, "wb fp32 [heads, CX]");
    wbp = wb.data_ptr<float>();
  } else {
    TORCH_CHECK(W.size(0) >= 4 * HD + HD / 32, "W: the bias rows after 4 HD");
    wb16 = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>()) + 4 * HD * CX;
  }
  const float* gbp = opt_f32(gb, HD, "gb");
  const float* obp = opt_f32(ob, CX, "ob");
#define L1_GO(cx, hd) if (CX == cx && HD == hd) return launch_l128<cx, hd>(x, lw, lb, eps, W, wbp, wb16, wo, gbp, obp, ground, mask, B, L, geo, geoy, bt, scale, out);
  L1_GO(128, 128) L1_GO(64, 128) L1_GO(64, 64)
#undef L1_GO
  TORCH_CHECK(false, "tri_l128: (d_pair, heads x 32) in (128, 128), (64, 128), (64, 64)");
  return {};
}

// Training at L = 128, d_pair 128 x 4 x 32 (no gate / out biases): the inference kernel that also stores what the chain's backward reads.
// -> (y, q, k, v, g, o [R, 128] bf16, lse [B, L, 4, 128], bias [B, 4, L, L]); drop [B, Lr, 128] bf16 or empty (y = x + drop o update)
std::vector<torch::Tensor> tri_l128_train(torch::Tensor x, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor W, torch::Tensor wb,
                                          torch::Tensor wo, int64_t HD, torch::Tensor mask, int64_t B, int64_t L, bool trans, int64_t lr, double scale,
                                          torch::Tensor drop, torch::Tensor out, torch::Tensor stats) {
  const long CX = x.size(-1);
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous();
  const int geo = rows_geo(trans, L, lr);
  const float* wbp = nullptr;
  const __nv_bfloat16* wb16 = nullptr;
  if (wb.defined() && wb.numel() > 0) {
    TORCH_CHECK(wb.scalar_type() == torch::kFloat32 && wb.is_contiguous() && wb.numel() == (HD / 32) * CX, "wb fp32 [heads, CX]");
    wbp = wb.data_ptr<float>();
  } else {
    TORCH_CHECK(W.size(0) >= 4 * HD + HD / 32, "W: the bias rows after 4 HD");
    wb16 = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>()) + 4 * HD * CX;
  }
  float2* stp = nullptr;
  if (stats.defined() && stats.numel() > 0) {
    TORCH_CHECK(stats.scalar_type() == torch::kFloat32 && stats.is_contiguous() && stats.numel() == 2 * B * L * L, "stats fp32 [R, 2]");
    stp = reinterpret_cast<float2*>(stats.data_ptr<float>());
  }
  std::vector<torch::Tensor> tro;
  torch::Tensor y;
#define L1_TR(cx, hd) if (CX == cx && HD == hd) y = launch_l128<cx, hd>(x, lw, lb, eps, W, wbp, wb16, wo, nullptr, nullptr, true, mask, B, L, geo, geo, false, scale, out, &tro, &drop, stp); else
  L1_TR(128, 128) L1_TR(64, 128) L1_TR(64, 64) TORCH_CHECK(false, "tri_l128_train: (d_pair, heads x 32) in (128, 128), (64, 128), (64, 64)");
#undef L1_TR
  tro.insert(tro.begin(), y);
  return tro;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_l128_train", &tri_l128_train, "sm100 TriangleAttention training forward at L = 128 (d_pair 128): (y, q, k, v, g, o, lse, bias)");
  m.def("tri_l128", &tri_l128, "sm100 TriangleAttention inference at L = 128: LN + projections + core + gate + out-projection in one CTA a pair row",
        py::arg("x"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("W"), py::arg("wb"), py::arg("wo"), py::arg("gb"), py::arg("ob"),
        py::arg("ground"), py::arg("HD"), py::arg("mask"), py::arg("B"), py::arg("L"), py::arg("trans"), py::arg("trans_out"), py::arg("bt"), py::arg("lr"),
        py::arg("scale"), py::arg("out"));
}
