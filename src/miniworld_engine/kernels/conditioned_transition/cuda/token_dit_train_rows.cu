// Token DiT TRAINING row kernels (CUDA): every elementwise / row step of the fused training block
// (integrations/token_dit_train.py), forward and backward. The GEMMs between them are cuBLAS, the attention core sm_100a.
//
// Layout: a block of NT = width / 4 threads walks rpb consecutive rows (bf16: RPB = 8; fp32: one resident wave of blocks, the
// rows dealt evenly -- row_grid); a thread always owns the same 4 columns, so the per-column gradient sums (biases, qk-norm
// weights) accumulate in registers across the block's rows and leave as one row of a [blocks, width] partial buffer that the host
// sums. Row statistics are block sums. x / x1 / dx are fp32 (the residual stream), every
// GEMM operand AT (bf16 on the bf16 path, fp32 on the fp32 path), the block's input / output dtype T (bf16 or fp32).
//
//   forward   cond_prep    c -> c_hat = LN(c) bf16, c bf16, (mean, rstd)                          (d_cond = 384)
//             adaln_a      xa = sigmoid(G[:, 0:D] + bs1) LN(x) + G[:, D:2D]                       (D = 768)
//             qknorm       qn = RMS(q) wq, kn = RMS(k) wk per 48-wide head (or copies), v copied, rq / rk saved
//             gate_o       og = sigmoid(g) o
//             res_adaln_b  x1 = x + sigmoid(Gg[:, 0:D] + bg1) y;  xt = sigmoid(G[:, 2D:3D] + bs2) LN(x1) + G[:, 3D:4D]
//             res_c        out = x1 + sigmoid(Gg[:, D:2D] + bg2) z
//             pair_ln      p_hat = LN(pair) bf16 and (mean, rstd)                                  (d_pair = 128)
//             pair_bias    (fp32 path) LN(pair) folded straight into the head-major bias, key mask as -inf
//   backward  res_c_bwd, swiglu_bwd, res_adaln_b_bwd, gate_o_bwd, qknorm_bwd, adaln_a_bwd, cond_bwd, unfold_lnw, pair_ln_bwd,
//             pair_bias_bwd (fp32 path: dpair and the dWf partials in one pass)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

// The model width and head dim are build-time: -DTD_D (768 | 1024) -DTD_HD (48 | 32 | 64); one extension per layout
#ifndef TD_D
#define TD_D 768
#endif
#ifndef TD_HD
#define TD_HD 48
#endif
constexpr int D = TD_D, NT = D / 4, DC = 384, NTC = DC / 4, RPB = 8, HD = TD_HD;
constexpr int HT = HD / 4, NHH = D / HD;                                   // threads per head, heads
static_assert(D % HD == 0 && NT % 32 == 0, "token DiT train rows: layout");

__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 sig4(float4 a) { return make_float4(sigm(a.x), sigm(a.y), sigm(a.z), sigm(a.w)); }
__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }
template <typename AT> __device__ __forceinline__ float4 rnd(float4 a) { return a; }   // the value a store of AT keeps
template <> __device__ __forceinline__ float4 rnd<bf>(float4 a) {                      // (sums use what was stored)
  return make_float4(__bfloat162float(__float2bfloat16_rn(a.x)), __bfloat162float(__float2bfloat16_rn(a.y)),
                     __bfloat162float(__float2bfloat16_rn(a.z)), __bfloat162float(__float2bfloat16_rn(a.w)));
}
__device__ __forceinline__ float4 dsig4(float4 v, float4 s) {   // v * s * (1 - s)
  return make_float4(v.x * s.x * (1.f - s.x), v.y * s.y * (1.f - s.y), v.z * s.z * (1.f - s.z), v.w * s.w * (1.f - s.w));
}
// Column sums leave a block as one row of a [blocks, width] partial buffer (plain stores); the host sums the rows. Atomics
// from every block onto the same 768 addresses serialised in L2 (res_c_bwd 105 us instead of ~20 at L384, A48).
__device__ __forceinline__ void part_store(float* part, long stride, float4 v) {
  V4<float>::store(part + (long)blockIdx.x * stride + threadIdx.x * 4, v);
}

// ------------------------------------------------------------------------------------------------------------- forward
template <typename CT, typename AT>
__global__ void __launch_bounds__(NTC) cond_prep_k(const CT* __restrict__ C, AT* __restrict__ CHAT, AT* __restrict__ CBF,
    float2* __restrict__ CST, int M, int rpb, float eps) {
  __shared__ float red[NTC / 32];
  const int col = threadIdx.x * 4;
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 c = V4<CT>::load(C + r * DC + col);
    const float mean = block_sum<NTC>(sum4(c), red) / DC;
    const float4 d = make_float4(c.x - mean, c.y - mean, c.z - mean, c.w - mean);
    const float rstd = rsqrtf(block_sum<NTC>(sum4(mul4(d, d)), red) / DC + eps);
    V4<AT>::store(CHAT + r * DC + col, make_float4(d.x * rstd, d.y * rstd, d.z * rstd, d.w * rstd));
    if (CBF) V4<AT>::store(CBF + r * DC + col, c);                       // none: the GEMM reads c itself (fp32 c, fp32 operands)
    if (threadIdx.x == 0) CST[r] = make_float2(mean, rstd);
  }
}

// X in the residual's dtype XT: the block's input itself (x = single, never copied first) or the fp32 residual; XO, when given,
// receives the fp32 residual (the backward and res_adaln_b read it)
template <typename XT, typename AT>
__global__ void __launch_bounds__(NT) adaln_a_k(const XT* __restrict__ X, float* __restrict__ XO, const AT* __restrict__ G, long sg,
    const float* __restrict__ BS, AT* __restrict__ XA, float2* __restrict__ XST, int M, int rpb, float eps) {
  __shared__ float red[NT / 32];
  const int col = threadIdx.x * 4;
  const float4 bs = V4<float>::load(BS + col);
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 x = V4<XT>::load(X + r * D + col);
    if (XO) V4<float>::store(XO + r * D + col, x);
    const float mean = block_sum<NT>(sum4(x), red) / D;
    const float4 d = make_float4(x.x - mean, x.y - mean, x.z - mean, x.w - mean);
    const float rstd = rsqrtf(block_sum<NT>(sum4(mul4(d, d)), red) / D + eps);
    const float4 s = sig4(add4(V4<AT>::load(G + r * sg + col), bs)), sh = V4<AT>::load(G + r * sg + D + col);
    V4<AT>::store(XA + r * D + col, make_float4(s.x * d.x * rstd + sh.x, s.y * d.y * rstd + sh.y, s.z * d.z * rstd + sh.z,
                                                s.w * d.w * rstd + sh.w));
    if (threadIdx.x == 0) XST[r] = make_float2(mean, rstd);
  }
}

// Per-head (HD columns = HT threads) sum through shared memory: part holds NT partials.
__device__ __forceinline__ float head_sum(float v, float* part) {
  part[threadIdx.x] = v;
  __syncthreads();
  const int h0 = (threadIdx.x / HT) * HT;
  float t = 0.f;
#pragma unroll
  for (int j = 0; j < HT; ++j) t += part[h0 + j];
  __syncthreads();
  return t;
}

template <typename AT>
__global__ void __launch_bounds__(NT) qknorm_k(const AT* __restrict__ QKVG, const float* __restrict__ WQ,
    const float* __restrict__ WK, AT* __restrict__ QN, AT* __restrict__ KN, AT* __restrict__ VC, float* __restrict__ RQK,
    int M, int rpb, float eq, float ek, int qk) {
  __shared__ float part[NT];
  const int col = threadIdx.x * 4, h = threadIdx.x / HT;
  const float4 wq = qk ? V4<float>::load(WQ + col % HD) : make_float4(1.f, 1.f, 1.f, 1.f);
  const float4 wk = qk ? V4<float>::load(WK + col % HD) : make_float4(1.f, 1.f, 1.f, 1.f);
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const AT* row = QKVG + r * 4 * D;
    float4 q = V4<AT>::load(row + col), k = V4<AT>::load(row + D + col);
    if (qk) {
      const float rq = rsqrtf(head_sum(sum4(mul4(q, q)), part) / HD + eq);
      const float rk = rsqrtf(head_sum(sum4(mul4(k, k)), part) / HD + ek);
      q = mul4(make_float4(q.x * rq, q.y * rq, q.z * rq, q.w * rq), wq);
      k = mul4(make_float4(k.x * rk, k.y * rk, k.z * rk, k.w * rk), wk);
      if (threadIdx.x % HT == 0) { RQK[r * 2 * NHH + h] = rq; RQK[r * 2 * NHH + NHH + h] = rk; }
    }
    V4<AT>::store(QN + r * D + col, q);
    V4<AT>::store(KN + r * D + col, k);
    if (VC) V4<AT>::store(VC + r * D + col, V4<AT>::load(row + 2 * D + col));   // none: the core reads v from qkvg
  }
}

template <typename AT>
__global__ void __launch_bounds__(NT) gate_o_k(const float* __restrict__ O, const AT* __restrict__ QKVG, AT* __restrict__ OG,
    int M, int rpb) {
  const int col = threadIdx.x * 4;
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    V4<AT>::store(OG + r * D + col, mul4(sig4(V4<AT>::load(QKVG + r * 4 * D + 3 * D + col)), V4<float>::load(O + r * D + col)));
  }
}

template <typename AT>
__global__ void __launch_bounds__(NT) res_adaln_b_k(const float* __restrict__ X, const AT* __restrict__ Y,
    const AT* __restrict__ GG, long sgg, const float* __restrict__ BG1, const AT* __restrict__ G, long sg,
    const float* __restrict__ BS2, float* __restrict__ X1, AT* __restrict__ XT, float2* __restrict__ X1ST, int M, int rpb, float eps) {
  __shared__ float red[NT / 32];
  const int col = threadIdx.x * 4;
  const float4 bg = V4<float>::load(BG1 + col), bs = V4<float>::load(BS2 + col);
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 x1 = add4(V4<float>::load(X + r * D + col),
                           mul4(sig4(add4(V4<AT>::load(GG + r * sgg + col), bg)), V4<AT>::load(Y + r * D + col)));
    V4<float>::store(X1 + r * D + col, x1);
    const float mean = block_sum<NT>(sum4(x1), red) / D;
    const float4 d = make_float4(x1.x - mean, x1.y - mean, x1.z - mean, x1.w - mean);
    const float rstd = rsqrtf(block_sum<NT>(sum4(mul4(d, d)), red) / D + eps);
    const float4 s = sig4(add4(V4<AT>::load(G + r * sg + 2 * D + col), bs)), sh = V4<AT>::load(G + r * sg + 3 * D + col);
    V4<AT>::store(XT + r * D + col, make_float4(s.x * d.x * rstd + sh.x, s.y * d.y * rstd + sh.y, s.z * d.z * rstd + sh.z,
                                                s.w * d.w * rstd + sh.w));
    if (threadIdx.x == 0) X1ST[r] = make_float2(mean, rstd);
  }
}

template <typename OT, typename AT>
__global__ void __launch_bounds__(NT) res_c_k(const float* __restrict__ X1, const AT* __restrict__ Z, const AT* __restrict__ GG,
    long sgg, const float* __restrict__ BG2, OT* __restrict__ OUT, int M, int rpb) {
  const int col = threadIdx.x * 4;
  const float4 bg = V4<float>::load(BG2 + col);
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    V4<OT>::store(OUT + r * D + col, add4(V4<float>::load(X1 + r * D + col),
        mul4(sig4(add4(V4<AT>::load(GG + r * sgg + D + col), bg)), V4<AT>::load(Z + r * D + col))));
  }
}

// pair rows, 128 wide: one warp per row, 8 rows per block
template <typename ZT, typename AT>
__global__ void __launch_bounds__(256) pair_ln_k(const ZT* __restrict__ Z, long sz, AT* __restrict__ PH, float2* __restrict__ PST,
    long R, float eps) {
  const long r = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (r >= R) return;
  const float4 z = V4<ZT>::load(Z + r * sz + lane * 4);
  const float mean = warp_sum(sum4(z)) / 128.f;
  const float4 d = make_float4(z.x - mean, z.y - mean, z.z - mean, z.w - mean);
  const float rstd = rsqrtf(warp_sum(sum4(mul4(d, d))) / 128.f + eps);
  V4<AT>::store(PH + r * 128 + lane * 4, make_float4(d.x * rstd, d.y * rstd, d.z * rstd, d.w * rstd));
  if (lane == 0) PST[r] = make_float2(mean, rstd);
}

// ------------------------------------------------------------------------------------------------------------ backward
template <typename OT, typename AT>
__global__ void __launch_bounds__(NT) res_c_bwd_k(const OT* __restrict__ DOUT, const AT* __restrict__ Z, const AT* __restrict__ GG,
    long sgg, const float* __restrict__ BG2, AT* __restrict__ DZ, AT* __restrict__ DGG, long sdg, float* __restrict__ PG2, long sp, int M, int rpb) {
  const int col = threadIdx.x * 4;
  const float4 bg = V4<float>::load(BG2 + col);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 dout = V4<OT>::load(DOUT + r * D + col), z = V4<AT>::load(Z + r * D + col);
    const float4 s = sig4(add4(V4<AT>::load(GG + r * sgg + D + col), bg));
    V4<AT>::store(DZ + r * D + col, mul4(dout, s));
    const float4 dg = rnd<AT>(dsig4(mul4(dout, z), s));
    V4<AT>::store(DGG + r * sdg + D + col, dg);
    acc = add4(acc, dg);
  }
  part_store(PG2, sp, acc);
}

// h = silu(a) b: da = dh b s (1 + a (1 - s)), db = dh a s (s = sigmoid(a)); h is recomputed for the squeeze weight gradient.
template <typename AT>
__global__ void __launch_bounds__(384) swiglu_bwd_k(const AT* __restrict__ DH, const AT* __restrict__ AB, AT* __restrict__ DAB,
    AT* __restrict__ Hh, int M, int N) {
  const long r = blockIdx.x;
  for (int c = threadIdx.x * 4; c < N; c += blockDim.x * 4) {
    const float4 dh = V4<AT>::load(DH + r * N + c), a = V4<AT>::load(AB + r * 2 * N + c), b = V4<AT>::load(AB + r * 2 * N + N + c);
    const float4 s = sig4(a);
    const float4 as = mul4(a, s);
    V4<AT>::store(DAB + r * 2 * N + c, make_float4(dh.x * b.x * s.x * (1.f + a.x * (1.f - s.x)), dh.y * b.y * s.y * (1.f + a.y * (1.f - s.y)),
                                                   dh.z * b.z * s.z * (1.f + a.z * (1.f - s.z)), dh.w * b.w * s.w * (1.f + a.w * (1.f - s.w))));
    V4<AT>::store(DAB + r * 2 * N + N + c, mul4(dh, as));
    V4<AT>::store(Hh + r * N + c, mul4(as, b));
  }
}

template <typename OT, typename AT>
__global__ void __launch_bounds__(NT) res_adaln_b_bwd_k(const OT* __restrict__ DOUT, const AT* DXT, long sxt, int dxt_in_dg,
    const float* __restrict__ X1, const float2* __restrict__ X1ST, const AT* __restrict__ G, long sg, const float* __restrict__ BS2,
    const AT* __restrict__ GG, long sgg, const float* __restrict__ BG1, const AT* __restrict__ Y, float* __restrict__ DX1,
    AT* __restrict__ DY, AT* DG, long sdg, AT* __restrict__ DGG, long sdgg, float* __restrict__ PS2,
    float* __restrict__ PG1, long sp, int M, int rpb) {
  __shared__ float red[NT / 32];
  const int col = threadIdx.x * 4;
  const float4 bs = V4<float>::load(BS2 + col), bg = V4<float>::load(BG1 + col);
  float4 acc_s = make_float4(0.f, 0.f, 0.f, 0.f), acc_g = acc_s;
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 dxt = V4<AT>::load(DXT + r * sxt + col), x1 = V4<float>::load(X1 + r * D + col);
    const float2 st = X1ST[r];
    const float4 xh = make_float4((x1.x - st.x) * st.y, (x1.y - st.x) * st.y, (x1.z - st.x) * st.y, (x1.w - st.x) * st.y);
    const float4 s2 = sig4(add4(V4<AT>::load(G + r * sg + 2 * D + col), bs));
    if (!dxt_in_dg) V4<AT>::store(DG + r * sdg + 3 * D + col, dxt);     // dxt_in_dg: the dxt GEMM wrote it there
    const float4 ds2 = rnd<AT>(dsig4(mul4(dxt, xh), s2));
    V4<AT>::store(DG + r * sdg + 2 * D + col, ds2);
    acc_s = add4(acc_s, ds2);
    const float4 dxh = mul4(dxt, s2);
    const float m1 = block_sum<NT>(sum4(dxh), red) / D, m2 = block_sum<NT>(sum4(mul4(dxh, xh)), red) / D;
    const float4 dout = V4<OT>::load(DOUT + r * D + col);
    const float4 dx1 = make_float4(dout.x + st.y * (dxh.x - m1 - xh.x * m2), dout.y + st.y * (dxh.y - m1 - xh.y * m2),
                                   dout.z + st.y * (dxh.z - m1 - xh.z * m2), dout.w + st.y * (dxh.w - m1 - xh.w * m2));
    V4<float>::store(DX1 + r * D + col, dx1);
    const float4 g1 = sig4(add4(V4<AT>::load(GG + r * sgg + col), bg));
    V4<AT>::store(DY + r * D + col, mul4(dx1, g1));
    const float4 dg1 = rnd<AT>(dsig4(mul4(dx1, V4<AT>::load(Y + r * D + col)), g1));
    V4<AT>::store(DGG + r * sdgg + col, dg1);
    acc_g = add4(acc_g, dg1);
  }
  part_store(PS2, sp, acc_s);
  part_store(PG1, sp, acc_g);
}

// og = sigmoid(g) o: dO = dog s (bf16, the core's input), D = rowsum_head(dO o) (the core backward's prep), dg = dog o s (1 - s)
template <typename AT>
__global__ void __launch_bounds__(NT) gate_o_bwd_k(const AT* __restrict__ DOG, const float* __restrict__ O, const AT* __restrict__ QKVG,
    AT* __restrict__ DOB, float* __restrict__ DD, AT* __restrict__ DQKVG, int M, int rpb, int L) {
  __shared__ float part[NT];
  const int col = threadIdx.x * 4, h = threadIdx.x / HT;
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 dog = V4<AT>::load(DOG + r * D + col), o = V4<float>::load(O + r * D + col);
    const float4 s = sig4(V4<AT>::load(QKVG + r * 4 * D + 3 * D + col));
    const float4 d_o = mul4(dog, s);
    V4<AT>::store(DOB + r * D + col, d_o);
    const float dsum = head_sum(sum4(mul4(d_o, o)), part);
    if (threadIdx.x % HT == 0) DD[((r / L) * NHH + h) * L + r % L] = dsum;
    V4<AT>::store(DQKVG + r * 4 * D + 3 * D + col, dsig4(mul4(dog, o), s));
  }
}

// qn = q rq wq: dq = rq (dqn wq - q rq mean_head(dqn wq q rq)); same for k; dv copied. AT into dqkvg[:, 0:3D]. dbq (column sums
// of dq) and dwq, dwk (dqn * q rq summed over rows and heads) accumulate per block.
template <typename AT>
__global__ void __launch_bounds__(NT) qknorm_bwd_k(const float* DQ, const float* DK, const float* DV, long sq, long sk, long sv,
    int alias, const AT* __restrict__ QKVG, const float* __restrict__ RQK, const float* __restrict__ WQ, const float* __restrict__ WK,
    AT* DQKVG, float* __restrict__ PBQ, float* __restrict__ DWQK, long sp, long sw, int M, int rpb, int qk) {
  __shared__ float part[NT];
  const int col = threadIdx.x * 4, h = threadIdx.x / HT;
  const float4 wq = qk ? V4<float>::load(WQ + col % HD) : make_float4(1.f, 1.f, 1.f, 1.f);
  const float4 wk = qk ? V4<float>::load(WK + col % HD) : make_float4(1.f, 1.f, 1.f, 1.f);
  float4 acc_b = make_float4(0.f, 0.f, 0.f, 0.f), acc_wq = acc_b, acc_wk = acc_b;
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const float4 dn = V4<float>::load(t == 0 ? DQ + r * sq + col : DK + r * sk + col);
      float4 dx = dn;
      if (qk) {
        const float4 x = V4<AT>::load(QKVG + r * 4 * D + t * D + col);
        const float rr = RQK[r * 2 * NHH + t * NHH + h];
        const float4 xh = make_float4(x.x * rr, x.y * rr, x.z * rr, x.w * rr);
        const float4 dxh = mul4(dn, t == 0 ? wq : wk);
        const float m = head_sum(sum4(mul4(dxh, xh)), part) / HD;
        dx = make_float4(rr * (dxh.x - xh.x * m), rr * (dxh.y - xh.y * m), rr * (dxh.z - xh.z * m), rr * (dxh.w - xh.w * m));
        if (t == 0) acc_wq = add4(acc_wq, mul4(dn, xh)); else acc_wk = add4(acc_wk, mul4(dn, xh));
      }
      const float4 dxb = rnd<AT>(dx);
      if (qk || !(alias & (1 << t))) V4<AT>::store(DQKVG + r * 4 * D + t * D + col, dxb);   // aliased and unchanged: already there
      if (t == 0) acc_b = add4(acc_b, dxb);
    }
    if (!(alias & 4)) V4<AT>::store(DQKVG + r * 4 * D + 2 * D + col, V4<float>::load(DV + r * sv + col));
  }
  part_store(PBQ, sp, acc_b);
  if (qk) { part_store(DWQK, sw, acc_wq); part_store(DWQK + D, sw, acc_wk); }
}

template <typename OT, typename AT>
__global__ void __launch_bounds__(NT) adaln_a_bwd_k(const AT* DXA, long sxa, int dxa_in_dg, const float* __restrict__ X,
    const float2* __restrict__ XST, const AT* __restrict__ G, long sg, const float* __restrict__ BS1, const float* __restrict__ DX1,
    OT* __restrict__ DX, AT* DG, long sdg, float* __restrict__ PS1, long sp, int M, int rpb) {
  __shared__ float red[NT / 32];
  const int col = threadIdx.x * 4;
  const float4 bs = V4<float>::load(BS1 + col);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 dxa = V4<AT>::load(DXA + r * sxa + col), x = V4<float>::load(X + r * D + col);
    const float2 st = XST[r];
    const float4 xh = make_float4((x.x - st.x) * st.y, (x.y - st.x) * st.y, (x.z - st.x) * st.y, (x.w - st.x) * st.y);
    const float4 s = sig4(add4(V4<AT>::load(G + r * sg + col), bs));
    if (!dxa_in_dg) V4<AT>::store(DG + r * sdg + D + col, dxa);         // dxa_in_dg: the dxa GEMM wrote it there
    const float4 ds1 = rnd<AT>(dsig4(mul4(dxa, xh), s));
    V4<AT>::store(DG + r * sdg + col, ds1);
    acc = add4(acc, ds1);
    const float4 dxh = mul4(dxa, s);
    const float m1 = block_sum<NT>(sum4(dxh), red) / D, m2 = block_sum<NT>(sum4(mul4(dxh, xh)), red) / D;
    const float4 dx1 = V4<float>::load(DX1 + r * D + col);
    V4<OT>::store(DX + r * D + col, make_float4(dx1.x + st.y * (dxh.x - m1 - xh.x * m2), dx1.y + st.y * (dxh.y - m1 - xh.y * m2),
                                                 dx1.z + st.y * (dxh.z - m1 - xh.z * m2), dx1.w + st.y * (dxh.w - m1 - xh.w * m2)));
  }
  part_store(PS1, sp, acc);
}

template <typename CT, typename OT, typename AT>
__global__ void __launch_bounds__(NTC) cond_bwd_k(const AT* __restrict__ DCHAT, const AT* __restrict__ DCG, const CT* __restrict__ C,
    const float2* __restrict__ CST, OT* __restrict__ DCO, int M, int rpb) {
  __shared__ float red[NTC / 32];
  const int col = threadIdx.x * 4;
  for (long r = (long)blockIdx.x * rpb, re = min((long)M, r + rpb); r < re; ++r) {
    const float4 dch = V4<AT>::load(DCHAT + r * DC + col), c = V4<CT>::load(C + r * DC + col);
    const float2 st = CST[r];
    const float4 ch = make_float4((c.x - st.x) * st.y, (c.y - st.x) * st.y, (c.z - st.x) * st.y, (c.w - st.x) * st.y);
    const float m1 = block_sum<NTC>(sum4(dch), red) / DC, m2 = block_sum<NTC>(sum4(mul4(dch, ch)), red) / DC;
    const float4 dcg = V4<AT>::load(DCG + r * DC + col);
    V4<OT>::store(DCO + r * DC + col, make_float4(st.y * (dch.x - m1 - ch.x * m2) + dcg.x, st.y * (dch.y - m1 - ch.y * m2) + dcg.y,
                                                  st.y * (dch.z - m1 - ch.z * m2) + dcg.z, st.y * (dch.w - m1 - ch.w * m2) + dcg.w));
  }
}

// The cond-LN weights are folded into the conditioning GEMM (Wn = W diag(w), four [768, 384] blocks). A block takes 16 rows of one
// 768-row group: dW[i] = dWn[i] w, and dw += dWn[i] W[i] summed over the group's rows (w1 for groups 0, 1; w2 for 2, 3).
__global__ void __launch_bounds__(NTC) unfold_lnw_k(const float* __restrict__ DWN, const float* __restrict__ WS1,
    const float* __restrict__ WB1, const float* __restrict__ WS2, const float* __restrict__ WB2, const float* __restrict__ W1,
    const float* __restrict__ W2, float* __restrict__ DW, float* __restrict__ DW12) {
  const int col = threadIdx.x * 4, i0 = blockIdx.x * 16, grp = i0 / D;
  const float* Wg = grp == 0 ? WS1 : grp == 1 ? WB1 : grp == 2 ? WS2 : WB2;
  const float4 lw = V4<float>::load((grp < 2 ? W1 : W2) + col);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = i0; i < i0 + 16; ++i) {
    const float4 d = V4<float>::load(DWN + (long)i * DC + col);
    V4<float>::store(DW + (long)i * DC + col, mul4(d, lw));
    acc = add4(acc, mul4(d, V4<float>::load(Wg + (long)(i % D) * DC + col)));
  }
  atomicAdd(DW12 + (grp / 2) * DC + col, acc.x); atomicAdd(DW12 + (grp / 2) * DC + col + 1, acc.y);
  atomicAdd(DW12 + (grp / 2) * DC + col + 2, acc.z); atomicAdd(DW12 + (grp / 2) * DC + col + 3, acc.w);
}

template <typename ZT>
__global__ void __launch_bounds__(256) pair_ln_bwd_k(const float* __restrict__ DXH, const ZT* __restrict__ Z, long sz,
    const float2* __restrict__ PST, ZT* __restrict__ DZ, long R) {
  const long r = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (r >= R) return;
  const float2 st = PST[r];
  const float4 z = V4<ZT>::load(Z + r * sz + lane * 4), dxh = V4<float>::load(DXH + r * 128 + lane * 4);
  const float4 xh = make_float4((z.x - st.x) * st.y, (z.y - st.x) * st.y, (z.z - st.x) * st.y, (z.w - st.x) * st.y);
  const float m1 = warp_sum(sum4(dxh)) / 128.f, m2 = warp_sum(sum4(mul4(dxh, xh))) / 128.f;
  V4<ZT>::store(DZ + r * 128 + lane * 4, make_float4(st.y * (dxh.x - m1 - xh.x * m2), st.y * (dxh.y - m1 - xh.y * m2),
                                                     st.y * (dxh.z - m1 - xh.z * m2), st.y * (dxh.w - m1 - xh.w * m2)));
}

// ------------------------------------------------------------------------------------------------- pair bias (fp32 path)
// The pair bias in one pass each way (the fp32 path; bf16 keeps pair_ln + a GEMM): forward LN of the 128-wide pair rows and
// bias[h, r] = sum_c LN(z)[r, c] Wf[h, c] (Wf = Wb diag(wp), the LN weight folded) with the key mask as -inf, no LN(z) written;
// backward recomputes LN(z) from the saved (mean, rstd), dxh = sum_h dbias[h, r] Wf[h, :] -> the LN backward, and dWf = dbias LN(z)
// as per-block partial rows [blocks, NHH 128] the host sums. Persistent blocks walk tiles of PT rows: a warp per row for the row
// work, the three small products on TF32 tensor cores (mma.sync m16n8k8, fp32 accumulation -- the TF32 the cuBLAS GEMMs they
// replace ran; on the FMA pipe with shared-memory operands the products were the kernels' whole cost, 269 / 404 us at L768).
// Heads are padded to HPAD (a multiple of 16) with zero weights / zero dbias rows. Shared-memory row strides are chosen so every
// fragment load is conflict-free (132: lanes (g, t) -> bank 4g + t; 136: lanes (t, g) -> bank 8t + g).
constexpr int PT = 32, HPAD = ((NHH + 15) / 16) * 16;

__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
// d += A (16 x 8, row) B (8 x 8, col): a0 (g, t), a1 (g + 8, t), a2 (g, t + 4), a3 (g + 8, t + 4); b0 (k t, n g), b1 (k t + 4, n g);
// d0 (g, 2t), d1 (g, 2t + 1), d2 (g + 8, 2t), d3 (g + 8, 2t + 1); g = lane / 4, t = lane % 4
__device__ __forceinline__ void mma_tf32(float (&d)[4], float a0, float a1, float a2, float a3, float b0, float b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(tf32r(a0)), "r"(tf32r(a1)), "r"(tf32r(a2)), "r"(tf32r(a3)), "r"(tf32r(b0)), "r"(tf32r(b1)));
}

template <typename ZT>
__global__ void __launch_bounds__(256) pair_bias_k(const ZT* __restrict__ Z, long sz, const float* __restrict__ WF,
    const bool* __restrict__ MASK, float* __restrict__ BIAS, float2* __restrict__ PST, long R, int L, float eps) {
  constexpr int S = 132;                                                   // A (rows g, cols t) and B (heads g, cols t): bank 4g + t
  __shared__ __align__(16) float wf[HPAD * S];
  __shared__ __align__(16) float ph[PT * S];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
  for (int i = threadIdx.x; i < HPAD * 32; i += 256) {
    const int h = i / 32, c4 = i % 32;
    *reinterpret_cast<float4*>(wf + h * S + 4 * c4) = h < NHH ? reinterpret_cast<const float4*>(WF)[h * 32 + c4] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  const long ntile = (R + PT - 1) / PT;
  for (long tl = blockIdx.x; tl < ntile; tl += gridDim.x) {
    const long r0 = tl * PT;
    __syncthreads();                                                       // wf in; the previous tile's LN(z) consumed
    for (int j = warp; j < PT; j += 8) {
      const long r = r0 + j;
      float4 xh = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < R) {
        const float4 z = V4<ZT>::load(Z + r * sz + lane * 4);
        const float mean = warp_sum(sum4(z)) / 128.f;
        const float4 d = make_float4(z.x - mean, z.y - mean, z.z - mean, z.w - mean);
        const float rstd = rsqrtf(warp_sum(sum4(mul4(d, d))) / 128.f + eps);
        xh = make_float4(d.x * rstd, d.y * rstd, d.z * rstd, d.w * rstd);
        if (lane == 0) PST[r] = make_float2(mean, rstd);
      }
      *reinterpret_cast<float4*>(ph + j * S + lane * 4) = xh;
    }
    __syncthreads();
    // bias^T [32 rows x HPAD] = LN(z) [32 x 128] Wf^T [128 x HPAD]: (row tile, head tile) items, one per warp
    constexpr int NIT = 2 * (HPAD / 8);
    if (warp < NIT) {
      const int mt = warp & 1, nt = warp >> 1;
      const float* pa = ph + (mt * 16 + g) * S + t;
      const float* pb = wf + (nt * 8 + g) * S + t;
      float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k0 = 0; k0 < 128; k0 += 8) mma_tf32(d, pa[k0], pa[8 * S + k0], pa[k0 + 4], pa[8 * S + k0 + 4], pb[k0], pb[k0 + 4]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const long r = r0 + mt * 16 + g + (e >= 2 ? 8 : 0);
        const int h = nt * 8 + 2 * t + (e & 1);
        if (r < R && h < NHH) BIAS[(long)h * R + r] = (MASK != nullptr && !MASK[r % L]) ? -INFINITY : d[e];   // r = query L + key
      }
    }
  }
}

template <typename ZT>
__global__ void __launch_bounds__(256) pair_bias_bwd_k(const float* __restrict__ DB, const ZT* __restrict__ Z, long sz,
    const float2* __restrict__ PST, const float* __restrict__ WF, ZT* __restrict__ DZ, float* __restrict__ PART, long R) {
  constexpr int WS = 136, XS = 136, TS = HPAD + 4, AS = PT + 4;
  __shared__ __align__(16) float wf[HPAD * WS];                           // Wf [heads][128]: B of dxh (k = head t, n = column g)
  __shared__ __align__(16) float xs[PT * XS];                             // dxh, then LN(z), of the tile's rows: B of dWf
  __shared__ float dbt[PT * TS];                                           // dbias [rows][heads]: A of dxh
  __shared__ float dba[HPAD * AS];                                         // dbias [heads][rows]: A of dWf
  constexpr int NACC = (HPAD / 16) * 16 / 8;                               // dWf (head tile, column tile) accumulators per warp
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
  for (int i = threadIdx.x; i < HPAD * 32; i += 256) {
    const int h = i / 32, c4 = i % 32;
    *reinterpret_cast<float4*>(wf + h * WS + 4 * c4) = h < NHH ? reinterpret_cast<const float4*>(WF)[h * 32 + c4] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float acc[NACC][4];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q][0] = acc[q][1] = acc[q][2] = acc[q][3] = 0.f;
  const long ntile = (R + PT - 1) / PT;
  for (long tl = blockIdx.x; tl < ntile; tl += gridDim.x) {
    const long r0 = tl * PT;
    __syncthreads();                                                       // wf in; the previous tile's db / xs consumed
    for (int i = threadIdx.x; i < HPAD * PT; i += 256) {                   // dbias[:, tile], 32 consecutive rows per head
      const int h = i / PT, j = i % PT;
      const long r = r0 + j;
      const float v = (h < NHH && r < R) ? DB[(long)h * R + r] : 0.f;
      dba[h * AS + j] = v;
      dbt[j * TS + h] = v;
    }
    __syncthreads();
    // dxh [32 x 128] = dbias^T [32 x HPAD] Wf [HPAD x 128]: 2 row tiles x 16 column tiles, 4 per warp
    for (int it = warp; it < 32; it += 8) {
      const int mt = it & 1, nt = it >> 1;
      const float* pa = dbt + (mt * 16 + g) * TS + t;
      const float* pb = wf + t * WS + nt * 8 + g;
      float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k0 = 0; k0 < HPAD; k0 += 8)
        mma_tf32(d, pa[k0], pa[8 * TS + k0], pa[k0 + 4], pa[8 * TS + k0 + 4], pb[k0 * WS], pb[(k0 + 4) * WS]);
      float* px = xs + (mt * 16 + g) * XS + nt * 8 + 2 * t;
      *reinterpret_cast<float2*>(px) = make_float2(d[0], d[1]);
      *reinterpret_cast<float2*>(px + 8 * XS) = make_float2(d[2], d[3]);
    }
    __syncthreads();
    // the LN backward, a warp per row; the row's LN(z) then replaces its dxh in xs
    for (int j = warp; j < PT; j += 8) {
      const long r = r0 + j;
      float4 xh = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < R) {
        const float2 st = PST[r];
        const float4 z = V4<ZT>::load(Z + r * sz + lane * 4);
        xh = make_float4((z.x - st.x) * st.y, (z.y - st.x) * st.y, (z.z - st.x) * st.y, (z.w - st.x) * st.y);
        const float4 dxh = *reinterpret_cast<const float4*>(xs + j * XS + lane * 4);
        const float m1 = warp_sum(sum4(dxh)) / 128.f, m2 = warp_sum(sum4(mul4(dxh, xh))) / 128.f;
        V4<ZT>::store(DZ + r * 128 + lane * 4, make_float4(st.y * (dxh.x - m1 - xh.x * m2), st.y * (dxh.y - m1 - xh.y * m2),
                                                           st.y * (dxh.z - m1 - xh.z * m2), st.y * (dxh.w - m1 - xh.w * m2)));
      }
      __syncwarp();
      *reinterpret_cast<float4*>(xs + j * XS + lane * 4) = xh;
    }
    __syncthreads();
    // dWf [HPAD x 128] += dbias [HPAD x 32] LN(z) [32 x 128]: (head tile, column tile) accumulators, NACC per warp, over the tile
#pragma unroll
    for (int q = 0; q < NACC; ++q) {
      const int it = warp + 8 * q, hm = it >> 4, cn = it & 15;
      const float* pa = dba + (hm * 16 + g) * AS + t;
      const float* pb = xs + t * XS + cn * 8 + g;
#pragma unroll
      for (int k0 = 0; k0 < PT; k0 += 8)
        mma_tf32(acc[q], pa[k0], pa[8 * AS + k0], pa[k0 + 4], pa[8 * AS + k0 + 4], pb[k0 * XS], pb[(k0 + 4) * XS]);
    }
  }
#pragma unroll
  for (int q = 0; q < NACC; ++q) {
    const int it = warp + 8 * q, hm = it >> 4, cn = it & 15, c = cn * 8 + 2 * t;
    float* prow = PART + (long)blockIdx.x * NHH * 128;
    const int h0 = hm * 16 + g, h1 = h0 + 8;
    if (h0 < NHH) *reinterpret_cast<float2*>(prow + h0 * 128 + c) = make_float2(acc[q][0], acc[q][1]);
    if (h1 < NHH) *reinterpret_cast<float2*>(prow + h1 * 128 + c) = make_float2(acc[q][2], acc[q][3]);
  }
}

// -------------------------------------------------------------------------------------------------------------- host
#define TDT_DISPATCH(T, NAME, ...)                                                                   \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    TORCH_CHECK((T) == at::kBFloat16, "fp32 or bf16");                                                 \
    using NAME = bf; return __VA_ARGS__();                                                           \
  }()

template <typename C> C* P(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }
template <typename C> const C* CP(const at::Tensor& t) { return reinterpret_cast<const C*>(t.data_ptr()); }
template <typename C> C* OP(const c10::optional<at::Tensor>& t) { return t ? P<C>(*t) : nullptr; }
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
unsigned blocks(int64_t M) { return (unsigned)((M + RPB - 1) / RPB); }
// every GEMM operand of one call shares the dtype of `ref` (bf16 or fp32) and has unit column stride
void actc(const at::Tensor& t, const at::Tensor& ref, const char* n) {
  TORCH_CHECK(t.scalar_type() == ref.scalar_type() && t.stride(-1) == 1, n, ": rows of the operand dtype");
}
void f32c(const at::Tensor& t, const char* n) { TORCH_CHECK(t.scalar_type() == at::kFloat && t.is_contiguous(), n, ": contiguous fp32"); }

int nsm() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
// resident blocks per SM of a kernel at `threads` threads (cached per kernel)
template <typename K> int occupancy(K* kern, int threads) {
  static std::mutex mu;
  static std::unordered_map<const void*, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(reinterpret_cast<const void*>(kern));
  if (it != cache.end()) return it->second;
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, threads, 0));
  return cache[reinterpret_cast<const void*>(kern)] = std::max(n, 1);
}

// The row grid of the D / d_cond-wide kernels. bf16: RPB rows per block, ceil(M / RPB) blocks, as always. fp32: one wave -- every
// block resident at once (SM count x the kernel's occupancy), the rows dealt evenly, ceil(M / blocks) each: the RPB grid left
// the last wave 56 % full at L384 (2304 blocks over 1480 slots) and the kernels at 5-5.7 TB/s. `fixed` > 0: exactly that many
// blocks (the rows of a partial buffer, every one of which is written, zeros by blocks without rows).
struct RowGrid { unsigned grid; int rpb; };
template <typename AT, typename K> RowGrid row_grid(K* kern, int threads, int64_t M, int64_t fixed = 0) {
  if (!std::is_same<AT, float>::value) {
    TORCH_CHECK(fixed == 0 || fixed == (int64_t)blocks(M), "partial buffer rows: ceil(M / 8) on the bf16 path");
    return {blocks(M), RPB};
  }
  const int64_t g = fixed > 0 ? fixed : std::max<int64_t>(1, std::min<int64_t>(M, (int64_t)nsm() * occupancy(kern, threads)));
  const int rpb = (int)std::max<int64_t>(1, (M + g - 1) / g);
  return {(unsigned)(fixed > 0 ? fixed : (M + rpb - 1) / rpb), rpb};
}

// rows of the backward's shared partial buffer (res_c_bwd, res_adaln_b_bwd, qknorm_bwd, adaln_a_bwd): bf16 ceil(M / 8); fp32 the
// SM count x the lowest occupancy of the four, so every block of each is resident at once
int64_t part_rows(int64_t M, bool fp32) {
  if (!fp32) return blocks(M);
  const int o = std::min(std::min(occupancy(res_c_bwd_k<float, float>, NT), occupancy(res_adaln_b_bwd_k<float, float>, NT)),
                         std::min(occupancy(qknorm_bwd_k<float>, NT), occupancy(adaln_a_bwd_k<float, float>, NT)));
  return std::max<int64_t>(1, std::min<int64_t>(M, (int64_t)nsm() * o));
}

void cond_prep(at::Tensor c, at::Tensor chat, c10::optional<at::Tensor> cbf, at::Tensor cst, double eps) {
  const int64_t M = chat.size(0);
  TORCH_CHECK(c.is_contiguous() && c.numel() == M * DC, "cond_prep: contiguous [M, 384]");
  if (cbf) actc(*cbf, chat, "cbf");
  const at::cuda::CUDAGuard g(c.device());
  TDT_DISPATCH(chat.scalar_type(), AT, [&] {
    TDT_DISPATCH(c.scalar_type(), CT, [&] {
      const RowGrid rg = row_grid<AT>(cond_prep_k<CT, AT>, NTC, M);
      cond_prep_k<CT, AT><<<rg.grid, NTC, 0, S()>>>(CP<CT>(c), P<AT>(chat), OP<AT>(cbf), P<float2>(cst), (int)M, rg.rpb, (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x: the fp32 residual, or (xo given) the block's input in its own dtype, whose fp32 copy the kernel writes to xo on the way
void adaln_a(at::Tensor x, at::Tensor G, at::Tensor bs, at::Tensor xa, at::Tensor xst, double eps, c10::optional<at::Tensor> xo) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_contiguous() && x.size(1) == D, "adaln_a: contiguous x [M, 768]");
  if (xo) f32c(*xo, "xo"); else f32c(x, "x");
  actc(xa, G, "xa"); f32c(bs, "bs");
  const at::cuda::CUDAGuard g(x.device());
  float* xop = xo ? P<float>(*xo) : nullptr;
  TDT_DISPATCH(G.scalar_type(), AT, [&] {
    TDT_DISPATCH(x.scalar_type(), XT, [&] {
      const RowGrid rg = row_grid<AT>(adaln_a_k<XT, AT>, NT, M);
      adaln_a_k<XT, AT><<<rg.grid, NT, 0, S()>>>(CP<XT>(x), xop, CP<AT>(G), G.stride(0), CP<float>(bs), P<AT>(xa), P<float2>(xst),
                                                 (int)M, rg.rpb, (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// vc none: v is not copied (the core reads it as a column view of qkvg)
void qknorm(at::Tensor qkvg, at::Tensor wq, at::Tensor wk, at::Tensor qn, at::Tensor kn, c10::optional<at::Tensor> vc, at::Tensor rqk,
            double eq, double ek, bool qk) {
  const int64_t M = qkvg.size(0);
  TORCH_CHECK(qkvg.is_contiguous() && qkvg.size(1) == 4 * D, "qknorm: qkvg [M, 3072]");
  actc(qn, qkvg, "qn"); actc(kn, qkvg, "kn");
  if (vc) actc(*vc, qkvg, "vc");
  const at::cuda::CUDAGuard g(qkvg.device());
  TDT_DISPATCH(qkvg.scalar_type(), AT, [&] {
    const RowGrid rg = row_grid<AT>(qknorm_k<AT>, NT, M);
    qknorm_k<AT><<<rg.grid, NT, 0, S()>>>(CP<AT>(qkvg), CP<float>(wq), CP<float>(wk), P<AT>(qn), P<AT>(kn), OP<AT>(vc), P<float>(rqk),
                                          (int)M, rg.rpb, (float)eq, (float)ek, qk ? 1 : 0);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_o(at::Tensor o, at::Tensor qkvg, at::Tensor og) {
  const int64_t M = o.size(0);
  actc(og, qkvg, "og");
  const at::cuda::CUDAGuard g(o.device());
  TDT_DISPATCH(qkvg.scalar_type(), AT, [&] {
    const RowGrid rg = row_grid<AT>(gate_o_k<AT>, NT, M);
    gate_o_k<AT><<<rg.grid, NT, 0, S()>>>(CP<float>(o), CP<AT>(qkvg), P<AT>(og), (int)M, rg.rpb);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_adaln_b(at::Tensor x, at::Tensor y, at::Tensor Gg, at::Tensor bg1, at::Tensor G, at::Tensor bs2, at::Tensor x1,
                 at::Tensor xt, at::Tensor x1st, double eps) {
  const int64_t M = x.size(0);
  f32c(x, "x"); actc(Gg, y, "Gg"); actc(G, y, "G"); actc(xt, y, "xt");
  const at::cuda::CUDAGuard g(x.device());
  TDT_DISPATCH(y.scalar_type(), AT, [&] {
    const RowGrid rg = row_grid<AT>(res_adaln_b_k<AT>, NT, M);
    res_adaln_b_k<AT><<<rg.grid, NT, 0, S()>>>(CP<float>(x), CP<AT>(y), CP<AT>(Gg), Gg.stride(0), CP<float>(bg1), CP<AT>(G), G.stride(0),
                                               CP<float>(bs2), P<float>(x1), P<AT>(xt), P<float2>(x1st), (int)M, rg.rpb, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_c(at::Tensor x1, at::Tensor z, at::Tensor Gg, at::Tensor bg2, at::Tensor out) {
  const int64_t M = x1.size(0);
  actc(Gg, z, "Gg");
  const at::cuda::CUDAGuard g(x1.device());
  TDT_DISPATCH(z.scalar_type(), AT, [&] {
    TDT_DISPATCH(out.scalar_type(), OT, [&] {
      const RowGrid rg = row_grid<AT>(res_c_k<OT, AT>, NT, M);
      res_c_k<OT, AT><<<rg.grid, NT, 0, S()>>>(CP<float>(x1), CP<AT>(z), CP<AT>(Gg), Gg.stride(0), CP<float>(bg2), P<OT>(out), (int)M,
                                               rg.rpb);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pair_ln(at::Tensor z, at::Tensor ph, at::Tensor pst, double eps) {
  const int64_t R = z.size(0);
  TORCH_CHECK(z.size(1) == 128 && z.stride(1) == 1 && ph.is_contiguous(), "pair_ln: [R, 128]");
  const at::cuda::CUDAGuard g(z.device());
  TDT_DISPATCH(ph.scalar_type(), AT, [&] {
    TDT_DISPATCH(z.scalar_type(), ZT, [&] {
      pair_ln_k<ZT, AT><<<(unsigned)((R + 7) / 8), 256, 0, S()>>>(CP<ZT>(z), z.stride(0), P<AT>(ph), P<float2>(pst), R, (float)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// blocks of the persistent pair-bias kernels (and rows of pair_bias_bwd's dWf partial buffer)
int64_t pair_grid(int64_t R) {
  const int64_t tiles = (R + PT - 1) / PT;
  return std::max<int64_t>(1, std::min<int64_t>(tiles, (int64_t)nsm() * occupancy(pair_bias_bwd_k<float>, 256)));
}

// z [R, 128] (fp32 or bf16 rows), wf [NHH, 128] fp32, mask [L] bool or none -> bias [NHH, R] fp32 (head-major, masked keys -inf),
// pst [R, 2] (mean, rstd)
void pair_bias(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, at::Tensor bias, at::Tensor pst, int64_t L, double eps) {
  const int64_t R = z.size(0);
  TORCH_CHECK(z.size(1) == 128 && z.stride(1) == 1, "pair_bias: z [R, 128]");
  f32c(wf, "wf"); f32c(bias, "bias"); f32c(pst, "pst");
  TORCH_CHECK(wf.numel() == NHH * 128 && bias.numel() == NHH * R && R == L * L, "pair_bias: wf [heads, 128], bias [heads, L, L]");
  if (mask) TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "pair_bias: mask [L] bool");
  const at::cuda::CUDAGuard g(z.device());
  const bool* mp = mask ? reinterpret_cast<const bool*>(mask->data_ptr()) : nullptr;
  TDT_DISPATCH(z.scalar_type(), ZT, [&] {
    const int64_t grid = std::max<int64_t>(1, std::min<int64_t>((R + PT - 1) / PT, (int64_t)nsm() * occupancy(pair_bias_k<ZT>, 256)));
    pair_bias_k<ZT><<<(unsigned)grid, 256, 0, S()>>>(CP<ZT>(z), z.stride(0), CP<float>(wf), mp, P<float>(bias), P<float2>(pst), R,
                                                     (int)L, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// db [NHH, R] fp32 (the core's dbias), z / pst as pair_bias got / gave -> dz [R, 128] (z's dtype), part [pair_grid(R), NHH 128] fp32
// (per-block dWf partials; the host sums the rows)
void pair_bias_bwd(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf, at::Tensor dz, at::Tensor part) {
  const int64_t R = z.size(0);
  TORCH_CHECK(z.size(1) == 128 && z.stride(1) == 1 && dz.is_contiguous() && dz.scalar_type() == z.scalar_type(), "pair_bias_bwd: z / dz");
  f32c(db, "db"); f32c(pst, "pst"); f32c(wf, "wf"); f32c(part, "part");
  TORCH_CHECK(db.numel() == NHH * R && part.size(1) == NHH * 128, "pair_bias_bwd: db [heads, R], part [blocks, heads 128]");
  const at::cuda::CUDAGuard g(z.device());
  TDT_DISPATCH(z.scalar_type(), ZT, [&] {
    pair_bias_bwd_k<ZT><<<(unsigned)part.size(0), 256, 0, S()>>>(CP<float>(db), CP<ZT>(z), z.stride(0), CP<float2>(pst), CP<float>(wf),
                                                                P<ZT>(dz), P<float>(part), R);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_c_bwd(at::Tensor dout, at::Tensor z, at::Tensor Gg, at::Tensor bg2, at::Tensor dz, at::Tensor dGg, at::Tensor pg2) {
  const int64_t M = z.size(0);
  actc(Gg, z, "Gg"); actc(dz, z, "dz"); actc(dGg, z, "dGg");
  const at::cuda::CUDAGuard g(z.device());
  TDT_DISPATCH(z.scalar_type(), AT, [&] {
    TDT_DISPATCH(dout.scalar_type(), OT, [&] {
      const RowGrid rg = row_grid<AT>(res_c_bwd_k<OT, AT>, NT, M, pg2.size(0));
      res_c_bwd_k<OT, AT><<<rg.grid, NT, 0, S()>>>(CP<OT>(dout), CP<AT>(z), CP<AT>(Gg), Gg.stride(0), CP<float>(bg2), P<AT>(dz), P<AT>(dGg),
                                                   dGg.stride(0), P<float>(pg2), pg2.stride(0), (int)M, rg.rpb);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_bwd(at::Tensor dh, at::Tensor ab, at::Tensor dab, at::Tensor h) {
  const int64_t M = dh.size(0), N = dh.size(1);
  TORCH_CHECK(N % 4 == 0 && ab.size(1) == 2 * N && dh.is_contiguous() && ab.is_contiguous(), "swiglu_bwd");
  actc(ab, dh, "ab"); actc(dab, dh, "dab"); actc(h, dh, "h");
  const at::cuda::CUDAGuard g(dh.device());
  TDT_DISPATCH(dh.scalar_type(), AT, [&] {
    swiglu_bwd_k<AT><<<(unsigned)M, 384, 0, S()>>>(CP<AT>(dh), CP<AT>(ab), P<AT>(dab), P<AT>(h), (int)M, (int)N);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// is t the column block [col, col + D) of the [M, 4D] buffer `buf` (same rows)?
bool col_view_of(const at::Tensor& t, const at::Tensor& buf, int64_t col) {
  return t.data_ptr() == static_cast<void*>(static_cast<char*>(buf.data_ptr()) + col * buf.element_size()) && t.stride(0) == buf.stride(0);
}

// dxt: [M, D] rows (any row stride), possibly dG's column block 3D (the dxt GEMM wrote it there: no copy)
void res_adaln_b_bwd(at::Tensor dout, at::Tensor dxt, at::Tensor x1, at::Tensor x1st, at::Tensor G, at::Tensor bs2, at::Tensor Gg,
                     at::Tensor bg1, at::Tensor y, at::Tensor dx1, at::Tensor dy, at::Tensor dG, at::Tensor dGg, at::Tensor ps2,
                     at::Tensor pg1) {
  const int64_t M = x1.size(0);
  actc(G, dxt, "G"); actc(Gg, dxt, "Gg"); actc(y, dxt, "y"); actc(dy, dxt, "dy"); actc(dG, dxt, "dG"); actc(dGg, dxt, "dGg");
  TORCH_CHECK(ps2.size(0) == pg1.size(0), "res_adaln_b_bwd: partial rows");
  const int in_dg = col_view_of(dxt, dG, 3 * D) ? 1 : 0;
  const at::cuda::CUDAGuard g(x1.device());
  TDT_DISPATCH(dxt.scalar_type(), AT, [&] {
    TDT_DISPATCH(dout.scalar_type(), OT, [&] {
      const RowGrid rg = row_grid<AT>(res_adaln_b_bwd_k<OT, AT>, NT, M, ps2.size(0));
      res_adaln_b_bwd_k<OT, AT><<<rg.grid, NT, 0, S()>>>(CP<OT>(dout), CP<AT>(dxt), dxt.stride(0), in_dg, CP<float>(x1), CP<float2>(x1st),
          CP<AT>(G), G.stride(0), CP<float>(bs2), CP<AT>(Gg), Gg.stride(0), CP<float>(bg1), CP<AT>(y), P<float>(dx1), P<AT>(dy), P<AT>(dG),
          dG.stride(0), P<AT>(dGg), dGg.stride(0), P<float>(ps2), P<float>(pg1), ps2.stride(0), (int)M, rg.rpb);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_o_bwd(at::Tensor dog, at::Tensor o, at::Tensor qkvg, at::Tensor dob, at::Tensor dd, at::Tensor dqkvg, int64_t L) {
  const int64_t M = o.size(0);
  actc(qkvg, dog, "qkvg"); actc(dob, dog, "dob"); actc(dqkvg, dog, "dqkvg");
  const at::cuda::CUDAGuard g(o.device());
  TDT_DISPATCH(dog.scalar_type(), AT, [&] {
    const RowGrid rg = row_grid<AT>(gate_o_bwd_k<AT>, NT, M);
    gate_o_bwd_k<AT><<<rg.grid, NT, 0, S()>>>(CP<AT>(dog), CP<float>(o), CP<AT>(qkvg), P<AT>(dob), P<float>(dd), P<AT>(dqkvg), (int)M,
                                              rg.rpb, (int)L);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dq / dk / dv: fp32 [M, D] rows (any row stride), each possibly dqkvg's own column block (the core wrote it there): read and
// rewritten in place by the thread that owns it, and not rewritten at all where it is unchanged
void qknorm_bwd(at::Tensor dq, at::Tensor dk, at::Tensor dv, at::Tensor qkvg, at::Tensor rqk, at::Tensor wq, at::Tensor wk,
                at::Tensor dqkvg, at::Tensor pbq, at::Tensor dwqk, bool qk) {
  const int64_t M = qkvg.size(0);
  actc(dqkvg, qkvg, "dqkvg");
  for (const at::Tensor* t : {&dq, &dk, &dv})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->stride(1) == 1 && t->size(0) == M && t->size(1) == D, "qknorm_bwd: fp32 [M, D] rows");
  const bool f32 = dqkvg.scalar_type() == at::kFloat;
  const int alias = f32 ? ((col_view_of(dq, dqkvg, 0) ? 1 : 0) | (col_view_of(dk, dqkvg, D) ? 2 : 0) | (col_view_of(dv, dqkvg, 2 * D) ? 4 : 0))
                        : 0;
  const at::cuda::CUDAGuard g(qkvg.device());
  TDT_DISPATCH(qkvg.scalar_type(), AT, [&] {
    const RowGrid rg = row_grid<AT>(qknorm_bwd_k<AT>, NT, M, pbq.size(0));
    qknorm_bwd_k<AT><<<rg.grid, NT, 0, S()>>>(CP<float>(dq), CP<float>(dk), CP<float>(dv), dq.stride(0), dk.stride(0), dv.stride(0), alias,
                                              CP<AT>(qkvg), CP<float>(rqk), CP<float>(wq), CP<float>(wk), P<AT>(dqkvg), P<float>(pbq),
                                              P<float>(dwqk), pbq.stride(0), dwqk.stride(0), (int)M, rg.rpb, qk ? 1 : 0);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dxa: [M, D] rows (any row stride), possibly dG's column block D (the dxa GEMM wrote it there: no copy)
void adaln_a_bwd(at::Tensor dxa, at::Tensor x, at::Tensor xst, at::Tensor G, at::Tensor bs1, at::Tensor dx1, at::Tensor dx,
                 at::Tensor dG, at::Tensor ps1) {
  const int64_t M = x.size(0);
  actc(G, dxa, "G"); actc(dG, dxa, "dG");
  const int in_dg = col_view_of(dxa, dG, D) ? 1 : 0;
  const at::cuda::CUDAGuard g(x.device());
  TDT_DISPATCH(dxa.scalar_type(), AT, [&] {
    TDT_DISPATCH(dx.scalar_type(), OT, [&] {
      const RowGrid rg = row_grid<AT>(adaln_a_bwd_k<OT, AT>, NT, M, ps1.size(0));
      adaln_a_bwd_k<OT, AT><<<rg.grid, NT, 0, S()>>>(CP<AT>(dxa), dxa.stride(0), in_dg, CP<float>(x), CP<float2>(xst), CP<AT>(G), G.stride(0),
                                                     CP<float>(bs1), CP<float>(dx1), P<OT>(dx), P<AT>(dG), dG.stride(0), P<float>(ps1),
                                                     ps1.stride(0), (int)M, rg.rpb);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cond_bwd(at::Tensor dchat, at::Tensor dcg, at::Tensor c, at::Tensor cst, at::Tensor dc) {
  const int64_t M = dchat.size(0);
  actc(dcg, dchat, "dcg");
  const at::cuda::CUDAGuard g(c.device());
  TDT_DISPATCH(dchat.scalar_type(), AT, [&] {
    TDT_DISPATCH(c.scalar_type(), CT, [&] {
      TDT_DISPATCH(dc.scalar_type(), OT, [&] {
        const RowGrid rg = row_grid<AT>(cond_bwd_k<CT, OT, AT>, NTC, M);
        cond_bwd_k<CT, OT, AT><<<rg.grid, NTC, 0, S()>>>(CP<AT>(dchat), CP<AT>(dcg), CP<CT>(c), CP<float2>(cst), P<OT>(dc), (int)M, rg.rpb);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unfold_lnw(at::Tensor dwn, at::Tensor ws1, at::Tensor wb1, at::Tensor ws2, at::Tensor wb2, at::Tensor w1, at::Tensor w2,
                at::Tensor dw, at::Tensor dw12) {
  TORCH_CHECK(dwn.size(0) == 4 * D && dwn.size(1) == DC, "unfold_lnw: dWn [3072, 384]");
  const at::cuda::CUDAGuard g(dwn.device());
  unfold_lnw_k<<<4 * D / 16, NTC, 0, S()>>>(CP<float>(dwn), CP<float>(ws1), CP<float>(wb1), CP<float>(ws2), CP<float>(wb2), CP<float>(w1),
                                            CP<float>(w2), P<float>(dw), P<float>(dw12));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pair_ln_bwd(at::Tensor dxh, at::Tensor z, at::Tensor pst, at::Tensor dz) {
  const int64_t R = z.size(0);
  const at::cuda::CUDAGuard g(z.device());
  TDT_DISPATCH(z.scalar_type(), ZT, [&] {
    pair_ln_bwd_k<ZT><<<(unsigned)((R + 7) / 8), 256, 0, S()>>>(CP<float>(dxh), CP<ZT>(z), z.stride(0), CP<float2>(pst), P<ZT>(dz), R);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cond_prep", &cond_prep);
  m.def("adaln_a", &adaln_a);
  m.def("qknorm", &qknorm);
  m.def("gate_o", &gate_o);
  m.def("res_adaln_b", &res_adaln_b);
  m.def("res_c", &res_c);
  m.def("pair_ln", &pair_ln);
  m.def("pair_bias", &pair_bias);
  m.def("pair_bias_bwd", &pair_bias_bwd);
  m.def("pair_grid", &pair_grid);
  m.def("part_rows", &part_rows);
  m.def("res_c_bwd", &res_c_bwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("res_adaln_b_bwd", &res_adaln_b_bwd);
  m.def("gate_o_bwd", &gate_o_bwd);
  m.def("qknorm_bwd", &qknorm_bwd);
  m.def("adaln_a_bwd", &adaln_a_bwd);
  m.def("cond_bwd", &cond_bwd);
  m.def("unfold_lnw", &unfold_lnw);
  m.def("pair_ln_bwd", &pair_ln_bwd);
}
