// dit_variants train_rows.cu -- the row passes of the DiT block variants' TRAINING step (Boltz-2, RoseTTAFold3, Chai-1; token DiT D = 768
// and atom DiT D = 128), forward and backward. The GEMMs between them are cuBLAS, the attention cores the engine's sm_90a kernels.
//
// Conventions
//   rows           row = a * L + tok: the A samples of a token's / atom's row; the conditioning tables are per token (shared by the samples).
//   tab            [L, K * D] table of the conditioning (per column block c: scale / shift / gate logits), dtype TT (= AT).
//   dtab partials  [S, L, K * D] fp32: the gradient of the tables. A token-major kernel (one block = one token x a split of the samples,
//                  `unit = s * L + tok`) accumulates its columns over its samples in registers and writes ONE row: no atomics, a fixed
//                  order (the host sums the S splits).  Every kernel writes its own column blocks.
//   sigmoid form   f(scale logit) = sigmoid(.) (Boltz-2, RoseTTAFold3); identity form (Chai-1): the table already holds W s + 1.
//   dtypes         X / OUT of the block: XT (bf16 or fp32); operands of the GEMMs and saved activations AT (bf16 or fp32 = TF32 GEMMs);
//                  the residual stream between the two halves of a sequential block, and every reduction, fp32.
//
//   forward   ln_mod_fwd     stats = LN(x); out_i = LN(x) f(sc_i) + sh_i  (one to three outputs)
//             qkv_prep       q, k, v of the attention core's layout (scale, bias-free; kq-norm: LayerNorm over all D columns + affine)
//             res_ln_fwd     x' = x + sigmoid(gA) ya; xt = LN(x') f(scT) + shT                      (sequential block)
//             res_out_fwd    out = base + sigmoid(g1) y1 [+ sigmoid(g2) y2]
//             glu_fwd        h = silu(g) v [u]
//   backward  res_bwd        dy_i = sigmoid(g_i) dout; dg_i += dout y_i sigmoid'(g_i)
//             glu_bwd        d[g | v | u] of h = silu(g) v [u]
//             res_adaln_bwd  the sequential block's middle: dx' = dout + LN-bwd(dxt f(scT)); dya = sigmoid(gA) dx'; dscT, dshT, dgA
//             gate_o_bwd     og = sigmoid(g) o: do, dg and D = rowsum_head(dog og)
//             qkv_bwd        the attention core's dq / dk / dv back to the projections' outputs (kq-norm backward), column sums
//             adaln_bwd      dx = pass + LN-bwd(sum_i da_i f(sc_i)); dsc_i, dsh_i
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 sub4(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __forceinline__ float4 zero4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
__device__ __forceinline__ float4 fma4(float4 a, float4 b, float4 c) {
  return make_float4(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y), fmaf(a.z, b.z, c.z), fmaf(a.w, b.w, c.w));
}
__device__ __forceinline__ float rna(float x) { unsigned y; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "f"(x)); return __uint_as_float(y); }
__device__ __forceinline__ float4 rna4(float4 a) { return make_float4(rna(a.x), rna(a.y), rna(a.z), rna(a.w)); }
__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }
__device__ __forceinline__ float4 sig4(float4 a) { return make_float4(sigm(a.x), sigm(a.y), sigm(a.z), sigm(a.w)); }
// scale factor f and its derivative d f / d logit, both 4-wide
template <bool SIG> __device__ __forceinline__ float4 sc_f(float4 v) { if constexpr (SIG) return sig4(v); else return v; }
template <bool SIG> __device__ __forceinline__ float4 sc_df(float4 f) {
  if constexpr (SIG) return make_float4(f.x * (1.f - f.x), f.y * (1.f - f.y), f.z * (1.f - f.z), f.w * (1.f - f.w));
  else return make_float4(1.f, 1.f, 1.f, 1.f);
}

// Row geometry: a 768-wide row is a block of NT = 192 threads; a 128-wide row (NT = 32) is a warp, 8 rows to a 256-thread block.
template <int NT>
struct Geo {
  static constexpr int RPB = NT == 32 ? 8 : 1, THREADS = NT * RPB;
  static __device__ __forceinline__ long row() { return NT == 32 ? (long)blockIdx.x * RPB + threadIdx.x / 32 : (long)blockIdx.x; }
  static __device__ __forceinline__ int col() { return (NT == 32 ? threadIdx.x % 32 : threadIdx.x) * 4; }
  static __device__ __forceinline__ bool lead() { return NT == 32 ? (threadIdx.x % 32) == 0 : threadIdx.x == 0; }
  static long blocks(long rows) { return (rows + RPB - 1) / RPB; }
};
template <int NT>
__device__ __forceinline__ float row_sum(float v, float* red) {
  if constexpr (NT == 32) return warp_sum(v); else return block_sum<NT>(v, red);
}

// LayerNorm (no affine) of a row: x -> xhat, with mean and rstd
template <int NT>
__device__ __forceinline__ float4 ln_row(float4 x, float eps, float* red, float& mean, float& rstd) {
  constexpr int D = NT * 4;
  mean = row_sum<NT>(sum4(x), red) / D;
  const float4 d = make_float4(x.x - mean, x.y - mean, x.z - mean, x.w - mean);
  rstd = rsqrtf(row_sum<NT>(sum4(mul4(d, d)), red) / D + eps);
  return mul4(d, rstd);
}

// ----------------------------------------------------------------------------------------------------------- forward
struct ModP {
  const void* sc[3];   // table column blocks (already offset to the block's first column; row stride sTab)
  const void* sh[3];
  void* out[3];
  int n;
};

template <int NT, typename XT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) ln_mod_fwd_k(const XT* __restrict__ X, float2* __restrict__ ST, ModP m, int L,
                                                                  long sTab, float eps, long M, bool sig) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int tok = row % L, col = Geo<NT>::col();
  float mean, rstd;
  const float4 xh = ln_row<NT>(V4<XT>::load(X + row * D + col), eps, red, mean, rstd);
  if (Geo<NT>::lead()) ST[row] = make_float2(mean, rstd);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    if (i < m.n) {
      const float4 s = V4<AT>::load(static_cast<const AT*>(m.sc[i]) + tok * sTab + col);
      const float4 h = V4<AT>::load(static_cast<const AT*>(m.sh[i]) + tok * sTab + col);
      const float4 f = sig ? sig4(s) : s;
      V4<AT>::store(static_cast<AT*>(m.out[i]) + row * D + col, fma4(xh, f, h));
    }
  }
}

// q (+ k, v) of the attention core's layout. Reads q / k / v rows (row strides sq, sk, sv), writes QO / KO / VO [M, D] contiguous
// (VO null: v is not copied). q is scaled by `scale`; kq: q and k are LayerNorm'd over the D columns with an fp32 affine first.
template <int NT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) qkv_prep_k(const AT* __restrict__ Q, long sq, const AT* __restrict__ K, long sk,
    const AT* __restrict__ V, long sv, AT* __restrict__ QO, AT* __restrict__ KO, AT* __restrict__ VO, float4* __restrict__ STQK,
    const float* __restrict__ WQ, const float* __restrict__ BQ, const float* __restrict__ WK, const float* __restrict__ BK, float scale,
    float eps, bool kq, bool tf32, long M) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int col = Geo<NT>::col();
  float4 q = V4<AT>::load(Q + row * sq + col), k = V4<AT>::load(K + row * sk + col);
  if (kq) {
    float mq, rq, mk, rk;
    const float4 qh = ln_row<NT>(q, eps, red, mq, rq), kh = ln_row<NT>(k, eps, red, mk, rk);
    q = fma4(qh, V4<float>::load(WQ + col), V4<float>::load(BQ + col));
    k = fma4(kh, V4<float>::load(WK + col), V4<float>::load(BK + col));
    if (Geo<NT>::lead()) STQK[row] = make_float4(mq, rq, mk, rk);
  }
  float4 qo = mul4(q, scale), vv = VO ? V4<AT>::load(V + row * sv + col) : zero4();
  if (tf32) { qo = rna4(qo); k = rna4(k); vv = rna4(vv); }
  V4<AT>::store(QO + row * D + col, qo);
  V4<AT>::store(KO + row * D + col, k);
  if (VO) V4<AT>::store(VO + row * D + col, vv);
}

// x' = x + sigmoid(gA) ya (fp32, stored); xt = LN(x') f(scT) + shT, stats of x' saved
template <int NT, typename XT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) res_ln_fwd_k(const XT* __restrict__ X, const AT* __restrict__ YA, const AT* __restrict__ GA,
    const AT* __restrict__ SC, const AT* __restrict__ SH, long sTab, float* __restrict__ XP, AT* __restrict__ XT_, float2* __restrict__ ST,
    int L, float eps, long M, bool sig) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int tok = row % L, col = Geo<NT>::col();
  float4 x = V4<XT>::load(X + row * D + col);
  x = fma4(sig4(V4<AT>::load(GA + tok * sTab + col)), V4<AT>::load(YA + row * D + col), x);
  V4<float>::store(XP + row * D + col, x);
  float mean, rstd;
  const float4 xh = ln_row<NT>(x, eps, red, mean, rstd);
  if (Geo<NT>::lead()) ST[row] = make_float2(mean, rstd);
  const float4 s = V4<AT>::load(SC + tok * sTab + col), h = V4<AT>::load(SH + tok * sTab + col);
  V4<AT>::store(XT_ + row * D + col, fma4(xh, sig ? sig4(s) : s, h));
}

// out = base + sigmoid(g1) y1 [+ sigmoid(g2) y2]
template <int NT, typename BT, typename XT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) res_out_fwd_k(const BT* __restrict__ BASE, const AT* __restrict__ Y1, const AT* __restrict__ G1,
    const AT* __restrict__ Y2, const AT* __restrict__ G2, long sTab, XT* __restrict__ OUT, int L, long M) {
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int tok = row % L, col = Geo<NT>::col();
  float4 x = V4<BT>::load(BASE + row * D + col);
  x = fma4(sig4(V4<AT>::load(G1 + tok * sTab + col)), V4<AT>::load(Y1 + row * D + col), x);
  if (Y2) x = fma4(sig4(V4<AT>::load(G2 + tok * sTab + col)), V4<AT>::load(Y2 + row * D + col), x);
  V4<XT>::store(OUT + row * D + col, x);
}

// og = sigmoid(g) o, in place on o (g row stride sg)
template <int NT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) gate_fwd_k(const AT* __restrict__ O, const AT* __restrict__ G, long sg, AT* __restrict__ OG, long M) {
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  if (row >= M) return;
  const int col = Geo<NT>::col();
  V4<AT>::store(OG + row * D + col, mul4(V4<AT>::load(O + row * D + col), sig4(V4<AT>::load(G + row * sg + col))));
}

template <typename AT, bool UP>
__global__ void __launch_bounds__(256) glu_fwd_k(const AT* __restrict__ AB, AT* __restrict__ H, int N, long n4) {
  const int per = N / 4;
  const long sab = (UP ? 3L : 2L) * N;
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long row = e / per;
    const int c = (int)(e - row * per) * 4;
    const float4 g = V4<AT>::load(AB + row * sab + c), v = V4<AT>::load(AB + row * sab + N + c);
    float4 o = make_float4(g.x * sigm(g.x) * v.x, g.y * sigm(g.y) * v.y, g.z * sigm(g.z) * v.z, g.w * sigm(g.w) * v.w);
    if constexpr (UP) o = mul4(o, V4<AT>::load(AB + row * sab + 2 * N + c));
    V4<AT>::store(H + row * N + c, o);
  }
}

// ---------------------------------------------------------------------------------------------------------- backward
// Token-major geometry: unit = s * L + tok; the block (warp) of the unit walks the samples a = s, s + S, ...
template <int NT>
struct TGeo {
  static __device__ __forceinline__ long unit() { return Geo<NT>::row(); }
};

__device__ __forceinline__ void part_put(float* part, long unit, long sPart, int off, int col, float4 v) {
  V4<float>::store(part + unit * sPart + off + col, v);
}

// dy_i = sigmoid(g_i) dout, dg_i += dout y_i sigmoid(g_i) (1 - sigmoid(g_i))      (one or two terms)
template <int NT, typename XT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) res_bwd_k(const XT* __restrict__ DOUT, const AT* __restrict__ Y1, const AT* __restrict__ G1,
    const AT* __restrict__ Y2, const AT* __restrict__ G2, long sTab, AT* __restrict__ DY1, AT* __restrict__ DY2, float* __restrict__ part,
    long sPart, int off1, int off2, int L, int S, int A) {
  constexpr int D = NT * 4;
  const long u = TGeo<NT>::unit();
  if (u >= (long)L * S) return;
  const int tok = u % L, s = u / L, col = Geo<NT>::col();
  const float4 g1 = sig4(V4<AT>::load(G1 + tok * sTab + col));
  const float4 g2 = Y2 ? sig4(V4<AT>::load(G2 + tok * sTab + col)) : zero4();
  float4 a1 = zero4(), a2 = zero4();
  for (int a = s; a < A; a += S) {
    const long row = (long)a * L + tok;
    const float4 d = V4<XT>::load(DOUT + row * D + col);
    const float4 y1 = V4<AT>::load(Y1 + row * D + col);
    V4<AT>::store(DY1 + row * D + col, mul4(d, g1));
    a1 = fma4(mul4(d, y1), mul4(g1, sub4(make_float4(1.f, 1.f, 1.f, 1.f), g1)), a1);
    if (Y2) {
      const float4 y2 = V4<AT>::load(Y2 + row * D + col);
      V4<AT>::store(DY2 + row * D + col, mul4(d, g2));
      a2 = fma4(mul4(d, y2), mul4(g2, sub4(make_float4(1.f, 1.f, 1.f, 1.f), g2)), a2);
    }
  }
  part_put(part, u, sPart, off1, col, a1);
  if (Y2) part_put(part, u, sPart, off2, col, a2);
}

// h = silu(g) v [u] backward: dab = [dg | dv | du]
template <typename AT, bool UP>
__global__ void __launch_bounds__(256) glu_bwd_k(const AT* __restrict__ AB, const AT* __restrict__ DH, AT* __restrict__ DAB, int N, long n4) {
  const int per = N / 4;
  const long sab = (UP ? 3L : 2L) * N;
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long row = e / per;
    const int c = (int)(e - row * per) * 4;
    const float4 g = V4<AT>::load(AB + row * sab + c), v = V4<AT>::load(AB + row * sab + N + c), dh = V4<AT>::load(DH + row * N + c);
    float4 u = make_float4(1.f, 1.f, 1.f, 1.f);
    if constexpr (UP) u = V4<AT>::load(AB + row * sab + 2 * N + c);
    float dg[4], dv[4], du[4];
    const float gg[4] = {g.x, g.y, g.z, g.w}, vv[4] = {v.x, v.y, v.z, v.w}, uu[4] = {u.x, u.y, u.z, u.w}, hh[4] = {dh.x, dh.y, dh.z, dh.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float sg = sigm(gg[i]), sl = gg[i] * sg;                      // silu(g)
      const float dsl = sg * (1.f + gg[i] * (1.f - sg));                    // silu'(g)
      dg[i] = hh[i] * vv[i] * uu[i] * dsl;
      dv[i] = hh[i] * sl * uu[i];
      du[i] = hh[i] * sl * vv[i];
    }
    V4<AT>::store(DAB + row * sab + c, make_float4(dg[0], dg[1], dg[2], dg[3]));
    V4<AT>::store(DAB + row * sab + N + c, make_float4(dv[0], dv[1], dv[2], dv[3]));
    if constexpr (UP) V4<AT>::store(DAB + row * sab + 2 * N + c, make_float4(du[0], du[1], du[2], du[3]));
  }
}

// The sequential block's middle: dx' = dout + LN-bwd(dxt f(scT)); dya = sigmoid(gA) dx'; dscT, dshT, dgA
template <int NT, typename XT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) res_adaln_bwd_k(const XT* __restrict__ DOUT, const AT* __restrict__ DT, const float* __restrict__ XP,
    const float2* __restrict__ ST, const AT* __restrict__ SC, const AT* __restrict__ GA, long sTab, const AT* __restrict__ YA, float* __restrict__ DXP,
    AT* __restrict__ DYA, float* __restrict__ part, long sPart, int offSc, int offSh, int offG, int L, int S, int A, bool sig) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long u = TGeo<NT>::unit();
  if (u >= (long)L * S) return;
  const int tok = u % L, s = u / L, col = Geo<NT>::col();
  const float4 sc = V4<AT>::load(SC + tok * sTab + col);
  const float4 f = sig ? sig4(sc) : sc, df = sig ? make_float4(f.x * (1.f - f.x), f.y * (1.f - f.y), f.z * (1.f - f.z), f.w * (1.f - f.w))
                                                 : make_float4(1.f, 1.f, 1.f, 1.f);
  const float4 ga = sig4(V4<AT>::load(GA + tok * sTab + col));
  float4 asc = zero4(), ash = zero4(), ag = zero4();
  for (int a = s; a < A; a += S) {
    const long row = (long)a * L + tok;
    const float2 st = ST[row];
    const float4 xp = V4<float>::load(XP + row * D + col);
    const float4 xh = mul4(make_float4(xp.x - st.x, xp.y - st.x, xp.z - st.x, xp.w - st.x), st.y);
    const float4 dt = V4<AT>::load(DT + row * D + col);
    const float4 dxh = mul4(dt, f);
    asc = fma4(mul4(dt, xh), df, asc);
    ash = add4(ash, dt);
    const float m1 = row_sum<NT>(sum4(dxh), red) / D, m2 = row_sum<NT>(sum4(mul4(dxh, xh)), red) / D;
    const float4 dln = mul4(sub4(sub4(dxh, make_float4(m1, m1, m1, m1)), mul4(xh, m2)), st.y);
    const float4 dxp = add4(V4<XT>::load(DOUT + row * D + col), dln);
    V4<float>::store(DXP + row * D + col, dxp);
    V4<AT>::store(DYA + row * D + col, mul4(dxp, ga));
    ag = fma4(mul4(dxp, V4<AT>::load(YA + row * D + col)), mul4(ga, sub4(make_float4(1.f, 1.f, 1.f, 1.f), ga)), ag);
  }
  part_put(part, u, sPart, offSc, col, asc);
  part_put(part, u, sPart, offSh, col, ash);
  part_put(part, u, sPart, offG, col, ag);
}

// og = sigmoid(g) o: do = sigmoid(g) dog, dg = dog og (1 - sigmoid(g)), dd[a, head, tok] = sum over the head's channels of dog og.
// Ungated: g null, `OG` is o: do = dog, dd = sum dog o. HD = channels per head; the head's threads are HD / 4 of the row's.
template <int NT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) gate_o_bwd_k(const AT* __restrict__ DOG, const AT* __restrict__ OG, const AT* __restrict__ G, long sg,
    AT* __restrict__ DO, AT* __restrict__ DG, long sdg, float* __restrict__ DD, int L, int HD, bool tf32, long M) {
  __shared__ float red[NT * Geo<NT>::RPB];
  constexpr int D = NT * 4;
  const long row = Geo<NT>::row();
  const bool live = row < M;
  const int col = Geo<NT>::col();
  const int slot = threadIdx.x;
  float p = 0.f;
  if (live) {
    const float4 dog = V4<AT>::load(DOG + row * D + col), og = V4<AT>::load(OG + row * D + col);
    const float4 prod = mul4(dog, og);
    p = sum4(prod);
    if (G) {
      const float4 sgm = sig4(V4<AT>::load(G + row * sg + col));
      V4<AT>::store(DO + row * D + col, tf32 ? rna4(mul4(dog, sgm)) : mul4(dog, sgm));
      V4<AT>::store(DG + row * sdg + col, mul4(prod, sub4(make_float4(1.f, 1.f, 1.f, 1.f), sgm)));
    } else {
      V4<AT>::store(DO + row * D + col, tf32 ? rna4(dog) : dog);
    }
  }
  red[slot] = p;
  __syncthreads();
  const int tpr = HD / 4, heads = D / HD;                          // threads per head, heads
  if (live) {
    const int r0 = NT == 32 ? (threadIdx.x / 32) * 32 : 0;          // this row's first thread slot
    const int lt = NT == 32 ? threadIdx.x % 32 : threadIdx.x;
    if (lt < heads) {
      float t = 0.f;
      for (int i = 0; i < tpr; ++i) t += red[r0 + lt * tpr + i];
      const long a = row / L, tok = row % L;
      DD[((long)a * heads + lt) * L + tok] = t;
    }
  }
}

// The attention core's dq, dk, dv (inputs, any of fp32 / AT, row strides) -> the projections' outputs' gradients, written to
// DQ_O / DK_O / DV_O (AT, row strides). kq: LayerNorm backward over the D columns with the fp32 affine (WQ, WK), q / k from the
// raw GEMM output (QR / KR, stats STQK). Column sums of the block's rows leave as one partial row:
//   [dbq] and, with kq, [dwq | dbq_ln | dwk | dbk_ln | dbq].
template <int NT, typename IT, typename AT>
__global__ void __launch_bounds__(NT) qkv_bwd_k(const IT* __restrict__ DQ, long sdq, const IT* __restrict__ DK, long sdk, const IT* __restrict__ DV,
    long sdv, const AT* __restrict__ QR, long sqr, const AT* __restrict__ KR, long skr, const float4* __restrict__ STQK,
    const float* __restrict__ WQ, const float* __restrict__ WK, AT* __restrict__ DQO, long sqo, AT* __restrict__ DKO, long sko,
    AT* __restrict__ DVO, long svo, float* __restrict__ part, long sPart, int rpb, bool kq, long M) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const int col = threadIdx.x * 4;
  float4 awq = zero4(), abq = zero4(), awk = zero4(), abk = zero4(), aq = zero4();
  float4 wq = zero4(), wk = zero4();
  if (kq) { wq = V4<float>::load(WQ + col); wk = V4<float>::load(WK + col); }
  const bool store = DKO != nullptr;                               // no outputs: only the column sums of dq
  for (long r = (long)blockIdx.x * rpb, re = std::min<long>(M, r + rpb); r < re; ++r) {
    float4 dq = V4<IT>::load(DQ + r * sdq + col), dk = store ? V4<IT>::load(DK + r * sdk + col) : zero4();
    if (!store) { aq = add4(aq, dq); continue; }
    if (kq) {
      const float4 st = STQK[r];
      const float4 qh = mul4(sub4(V4<AT>::load(QR + r * sqr + col), make_float4(st.x, st.x, st.x, st.x)), st.y);
      const float4 kh = mul4(sub4(V4<AT>::load(KR + r * skr + col), make_float4(st.z, st.z, st.z, st.z)), st.w);
      awq = fma4(dq, qh, awq); abq = add4(abq, dq);
      awk = fma4(dk, kh, awk); abk = add4(abk, dk);
      const float4 gq = mul4(dq, wq), gk = mul4(dk, wk);
      const float m1q = block_sum<NT>(sum4(gq), red) / D, m2q = block_sum<NT>(sum4(mul4(gq, qh)), red) / D;
      const float m1k = block_sum<NT>(sum4(gk), red) / D, m2k = block_sum<NT>(sum4(mul4(gk, kh)), red) / D;
      dq = mul4(sub4(sub4(gq, make_float4(m1q, m1q, m1q, m1q)), mul4(qh, m2q)), st.y);
      dk = mul4(sub4(sub4(gk, make_float4(m1k, m1k, m1k, m1k)), mul4(kh, m2k)), st.w);
    }
    aq = add4(aq, dq);
    V4<AT>::store(DQO + r * sqo + col, dq);
    V4<AT>::store(DKO + r * sko + col, dk);
    V4<AT>::store(DVO + r * svo + col, V4<IT>::load(DV + r * sdv + col));
  }
  float* p = part + (long)blockIdx.x * sPart + col;
  if (kq) {
    V4<float>::store(p, awq); V4<float>::store(p + D, abq); V4<float>::store(p + 2 * D, awk); V4<float>::store(p + 3 * D, abk);
    V4<float>::store(p + 4 * D, aq);
  } else {
    V4<float>::store(p, aq);
  }
}

// dx = pass + LN-bwd(sum_i da_i f(sc_i)); dsc_i = sum da_i xhat f'(sc_i); dsh_i = sum da_i       (one to three inputs)
struct AdP {
  const void* da[3];
  long sda[3];
  const void* sc[3];
  int offSc[3], offSh[3];
  int n;
};

template <int NT, typename BT, typename XT, typename AT>
__global__ void __launch_bounds__(Geo<NT>::THREADS) adaln_bwd_k(const XT* __restrict__ X, const float2* __restrict__ ST, const BT* __restrict__ PASS, AdP p,
    long sTab, XT* __restrict__ DX, float* __restrict__ part, long sPart, int L, int S, int A, bool sig) {
  __shared__ float red[NT / 32];
  constexpr int D = NT * 4;
  const long u = TGeo<NT>::unit();
  if (u >= (long)L * S) return;
  const int tok = u % L, s = u / L, col = Geo<NT>::col();
  float4 f[3], df[3], asc[3], ash[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    asc[i] = zero4(); ash[i] = zero4(); f[i] = zero4(); df[i] = zero4();
    if (i < p.n) {
      const float4 sc = V4<AT>::load(static_cast<const AT*>(p.sc[i]) + tok * sTab + col);
      f[i] = sig ? sig4(sc) : sc;
      df[i] = sig ? make_float4(f[i].x * (1.f - f[i].x), f[i].y * (1.f - f[i].y), f[i].z * (1.f - f[i].z), f[i].w * (1.f - f[i].w))
                  : make_float4(1.f, 1.f, 1.f, 1.f);
    }
  }
  for (int a = s; a < A; a += S) {
    const long row = (long)a * L + tok;
    const float2 st = ST[row];
    const float4 x = V4<XT>::load(X + row * D + col);
    const float4 xh = mul4(make_float4(x.x - st.x, x.y - st.x, x.z - st.x, x.w - st.x), st.y);
    float4 dxh = zero4();
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (i < p.n) {
        const float4 da = V4<AT>::load(static_cast<const AT*>(p.da[i]) + row * p.sda[i] + col);
        dxh = fma4(da, f[i], dxh);
        asc[i] = fma4(mul4(da, xh), df[i], asc[i]);
        ash[i] = add4(ash[i], da);
      }
    }
    const float m1 = row_sum<NT>(sum4(dxh), red) / D, m2 = row_sum<NT>(sum4(mul4(dxh, xh)), red) / D;
    const float4 dln = mul4(sub4(sub4(dxh, make_float4(m1, m1, m1, m1)), mul4(xh, m2)), st.y);
    V4<XT>::store(DX + row * D + col, add4(V4<BT>::load(PASS + row * D + col), dln));
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    if (i < p.n) {
      part_put(part, u, sPart, p.offSc[i], col, asc[i]);
      part_put(part, u, sPart, p.offSh[i], col, ash[i]);
    }
  }
}


// ---------------------------------------------------------------------------------------------------- pair LayerNorm rows
// LayerNorm of C-wide pair rows (C = 128 or 256, no affine), one warp per row: out = (z - mean) rstd
template <int C, typename ZT, typename OT>
__global__ void __launch_bounds__(256) pair_ln_fwd_k(const ZT* __restrict__ Z, OT* __restrict__ OUT, long R, float eps) {
  constexpr int K = C / 128;
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 z[K];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { z[k] = V4<ZT>::load(Z + row * C + k * 128 + lane * 4); s += sum4(z[k]); }
  const float mean = warp_sum(s) / C;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { z[k] = sub4(z[k], make_float4(mean, mean, mean, mean)); q += sum4(mul4(z[k], z[k])); }
  const float r = rsqrtf(warp_sum(q) / C + eps);
#pragma unroll
  for (int k = 0; k < K; ++k) V4<OT>::store(OUT + row * C + k * 128 + lane * 4, mul4(z[k], r));
}

// dz = rstd (dxhat - mean(dxhat) - xhat mean(dxhat xhat)), the statistics recomputed from z
template <int C, typename ZT, typename OT>
__global__ void __launch_bounds__(256) pair_ln_bwd_k(const ZT* __restrict__ Z, const OT* __restrict__ DXH, ZT* __restrict__ DZ, long R, float eps) {
  constexpr int K = C / 128;
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 z[K], g[K];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    z[k] = V4<ZT>::load(Z + row * C + k * 128 + lane * 4);
    g[k] = V4<OT>::load(DXH + row * C + k * 128 + lane * 4);
    s += sum4(z[k]);
  }
  const float mean = warp_sum(s) / C;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { z[k] = sub4(z[k], make_float4(mean, mean, mean, mean)); q += sum4(mul4(z[k], z[k])); }
  const float r = rsqrtf(warp_sum(q) / C + eps);
  float a = 0.f, b = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { z[k] = mul4(z[k], r); a += sum4(g[k]); b += sum4(mul4(g[k], z[k])); }
  a = warp_sum(a) / C;
  b = warp_sum(b) / C;
#pragma unroll
  for (int k = 0; k < K; ++k)
    V4<ZT>::store(DZ + row * C + k * 128 + lane * 4, mul4(sub4(sub4(g[k], make_float4(a, a, a, a)), mul4(z[k], b)), r));
}
// ------------------------------------------------------------------------------------------------------------- host
c10::cuda::CUDAStream S_() { return c10::cuda::getCurrentCUDAStream(); }
template <typename T> T* P(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* CP(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> const T* CPo(const c10::optional<at::Tensor>& t) { return (t && t->defined()) ? CP<T>(*t) : nullptr; }
template <typename T> T* Po(const c10::optional<at::Tensor>& t) { return (t && t->defined()) ? P<T>(*t) : nullptr; }

bool is_bf(const at::Tensor& t) { return t.scalar_type() == at::kBFloat16; }

// dtype dispatch: XT in {bf16, fp32}, AT in {bf16, fp32}; D in {768, 128}
#define DISPATCH_D(D, ...)                                                   \
  if ((D) == 768) { constexpr int NT = 192; __VA_ARGS__ }                    \
  else if ((D) == 128) { constexpr int NT = 32; __VA_ARGS__ }                \
  else TORCH_CHECK(false, "dit_variants train rows: D must be 768 or 128");
#define DISPATCH_A(T, ...)                                                   \
  if (is_bf(T)) { using AT = bf; __VA_ARGS__ } else { using AT = float; __VA_ARGS__ }
#define DISPATCH_X(T, ...)                                                   \
  if (is_bf(T)) { using XT = bf; __VA_ARGS__ } else { using XT = float; __VA_ARGS__ }

template <int NT> dim3 rgrid(long rows) { return dim3((unsigned)Geo<NT>::blocks(rows)); }
template <int NT> dim3 rblock() { return dim3(Geo<NT>::THREADS); }
long flat_grid(long n4) { return std::min<long>((n4 + 255) / 256, 132L * 16); }

}  // namespace

// out_i = LN(x) f(sc_i) + sh_i; tab [L, K D]; sc_cols / sh_cols: the column blocks (multiples of D) of each output; st [M, 2] fp32
void ln_mod_fwd(at::Tensor x, at::Tensor tab, std::vector<int64_t> sc_cols, std::vector<int64_t> sh_cols, std::vector<at::Tensor> outs,
                at::Tensor st, int64_t L, double eps, bool sig) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int D = x.size(1);
  const long M = x.size(0), sTab = tab.size(1);
  TORCH_CHECK(outs.size() >= 1 && outs.size() <= 3 && sc_cols.size() == outs.size() && sh_cols.size() == outs.size(), "ln_mod_fwd: outputs");
  ModP m{};
  m.n = (int)outs.size();
  const size_t es = tab.element_size();
  for (int i = 0; i < m.n; ++i) {
    m.sc[i] = reinterpret_cast<const char*>(tab.data_ptr()) + sc_cols[i] * es;
    m.sh[i] = reinterpret_cast<const char*>(tab.data_ptr()) + sh_cols[i] * es;
    m.out[i] = outs[i].data_ptr();
  }
  DISPATCH_D(D, DISPATCH_X(x, DISPATCH_A(tab,
    ln_mod_fwd_k<NT, XT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<XT>(x), P<float2>(st), m, (int)L, sTab, (float)eps, M, sig);
  )))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qkv_prep(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo, at::Tensor ko, c10::optional<at::Tensor> vo,
              c10::optional<at::Tensor> stqk, c10::optional<at::Tensor> wq, c10::optional<at::Tensor> bq, c10::optional<at::Tensor> wk,
              c10::optional<at::Tensor> bk, double scale, double eps, bool kq, bool tf32) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int D = qo.size(1);
  const long M = qo.size(0);
  TORCH_CHECK(q.stride(1) == 1 && k.stride(1) == 1 && v.stride(1) == 1, "qkv_prep: row-contiguous inputs");
  DISPATCH_D(D, DISPATCH_A(q,
    qkv_prep_k<NT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<AT>(q), q.stride(0), CP<AT>(k), k.stride(0), CP<AT>(v), v.stride(0), P<AT>(qo),
        P<AT>(ko), Po<AT>(vo), Po<float4>(stqk), CPo<float>(wq), CPo<float>(bq), CPo<float>(wk), CPo<float>(bk), (float)scale, (float)eps, kq, tf32, M);
  ))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_ln_fwd(at::Tensor x, at::Tensor ya, at::Tensor tab, int64_t g_col, int64_t sc_col, int64_t sh_col, at::Tensor xp, at::Tensor xt, at::Tensor st,
                int64_t L, double eps, bool sig) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int D = x.size(1);
  const long M = x.size(0), sTab = tab.size(1);
  const size_t es = tab.element_size();
  auto tb = [&](int64_t c) { return reinterpret_cast<const char*>(tab.data_ptr()) + c * es; };
  DISPATCH_D(D, DISPATCH_X(x, DISPATCH_A(tab,
    res_ln_fwd_k<NT, XT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<XT>(x), CP<AT>(ya), reinterpret_cast<const AT*>(tb(g_col)),
        reinterpret_cast<const AT*>(tb(sc_col)), reinterpret_cast<const AT*>(tb(sh_col)), sTab, P<float>(xp), P<AT>(xt), P<float2>(st), (int)L,
        (float)eps, M, sig);
  )))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// base: fp32 (the sequential block's x') or the block input's dtype
void res_out_fwd(at::Tensor base, at::Tensor y1, at::Tensor tab, int64_t g1_col, c10::optional<at::Tensor> y2, int64_t g2_col, at::Tensor out, int64_t L) {
  const c10::cuda::CUDAGuard guard(base.device());
  const int D = base.size(1);
  const long M = base.size(0), sTab = tab.size(1);
  const size_t es = tab.element_size();
  auto tb = [&](int64_t c) { return reinterpret_cast<const char*>(tab.data_ptr()) + c * es; };
  const bool base32 = base.scalar_type() == at::kFloat;
  DISPATCH_D(D, DISPATCH_X(out, DISPATCH_A(tab, {
    if (base32)
      res_out_fwd_k<NT, float, XT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<float>(base), CP<AT>(y1), reinterpret_cast<const AT*>(tb(g1_col)),
          CPo<AT>(y2), reinterpret_cast<const AT*>(tb(g2_col)), sTab, P<XT>(out), (int)L, M);
    else
      res_out_fwd_k<NT, XT, XT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<XT>(base), CP<AT>(y1), reinterpret_cast<const AT*>(tb(g1_col)),
          CPo<AT>(y2), reinterpret_cast<const AT*>(tb(g2_col)), sTab, P<XT>(out), (int)L, M);
  })))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_fwd(at::Tensor o, at::Tensor g, at::Tensor og) {
  const c10::cuda::CUDAGuard guard(o.device());
  const int D = o.size(1);
  const long M = o.size(0);
  TORCH_CHECK(g.stride(1) == 1, "gate_fwd: g rows");
  DISPATCH_D(D, DISPATCH_A(o,
    gate_fwd_k<NT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<AT>(o), CP<AT>(g), g.stride(0), P<AT>(og), M);
  ))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void glu_fwd(at::Tensor ab, at::Tensor h, bool up) {
  const c10::cuda::CUDAGuard guard(ab.device());
  const int N = h.size(1);
  const long n4 = h.numel() / 4;
  DISPATCH_A(ab, {
    if (up) glu_fwd_k<AT, true><<<flat_grid(n4), 256, 0, S_()>>>(CP<AT>(ab), P<AT>(h), N, n4);
    else glu_fwd_k<AT, false><<<flat_grid(n4), 256, 0, S_()>>>(CP<AT>(ab), P<AT>(h), N, n4);
  })
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void glu_bwd(at::Tensor ab, at::Tensor dh, at::Tensor dab, bool up) {
  const c10::cuda::CUDAGuard guard(ab.device());
  const int N = dh.size(1);
  const long n4 = dh.numel() / 4;
  DISPATCH_A(ab, {
    if (up) glu_bwd_k<AT, true><<<flat_grid(n4), 256, 0, S_()>>>(CP<AT>(ab), CP<AT>(dh), P<AT>(dab), N, n4);
    else glu_bwd_k<AT, false><<<flat_grid(n4), 256, 0, S_()>>>(CP<AT>(ab), CP<AT>(dh), P<AT>(dab), N, n4);
  })
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_bwd(at::Tensor dout, at::Tensor y1, at::Tensor tab, int64_t g1_col, at::Tensor dy1, c10::optional<at::Tensor> y2, int64_t g2_col,
             c10::optional<at::Tensor> dy2, at::Tensor part, int64_t off1, int64_t off2, int64_t L, int64_t S) {
  const c10::cuda::CUDAGuard guard(dout.device());
  const int D = dout.size(1);
  const long M = dout.size(0), sTab = tab.size(1), sPart = part.size(2);
  const int A = M / L;
  const size_t es = tab.element_size();
  auto tb = [&](int64_t c) { return reinterpret_cast<const char*>(tab.data_ptr()) + c * es; };
  DISPATCH_D(D, DISPATCH_X(dout, DISPATCH_A(tab,
    res_bwd_k<NT, XT, AT><<<rgrid<NT>(L * S), rblock<NT>(), 0, S_()>>>(CP<XT>(dout), CP<AT>(y1), reinterpret_cast<const AT*>(tb(g1_col)), CPo<AT>(y2),
        reinterpret_cast<const AT*>(tb(g2_col)), sTab, P<AT>(dy1), Po<AT>(dy2), P<float>(part), sPart, (int)off1, (int)off2, (int)L, (int)S, A);
  )))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_adaln_bwd(at::Tensor dout, at::Tensor dt, at::Tensor xp, at::Tensor st, at::Tensor tab, int64_t sc_col, int64_t g_col, at::Tensor ya,
                   at::Tensor dxp, at::Tensor dya, at::Tensor part, int64_t off_sc, int64_t off_sh, int64_t off_g, int64_t L, int64_t S, bool sig) {
  const c10::cuda::CUDAGuard guard(dout.device());
  const int D = dout.size(1);
  const long M = dout.size(0), sTab = tab.size(1), sPart = part.size(2);
  const int A = M / L;
  const size_t es = tab.element_size();
  auto tb = [&](int64_t c) { return reinterpret_cast<const char*>(tab.data_ptr()) + c * es; };
  DISPATCH_D(D, DISPATCH_X(dout, DISPATCH_A(tab,
    res_adaln_bwd_k<NT, XT, AT><<<rgrid<NT>(L * S), rblock<NT>(), 0, S_()>>>(CP<XT>(dout), CP<AT>(dt), CP<float>(xp), CP<float2>(st),
        reinterpret_cast<const AT*>(tb(sc_col)), reinterpret_cast<const AT*>(tb(g_col)), sTab, CP<AT>(ya), P<float>(dxp), P<AT>(dya), P<float>(part), sPart,
        (int)off_sc, (int)off_sh, (int)off_g, (int)L, (int)S, A, sig);
  )))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_o_bwd(at::Tensor dog, at::Tensor og, c10::optional<at::Tensor> g, at::Tensor dO, c10::optional<at::Tensor> dg, at::Tensor dd, int64_t L, int64_t hd, bool tf32) {
  const c10::cuda::CUDAGuard guard(dog.device());
  const int D = dog.size(1);
  const long M = dog.size(0);
  const long sg = (g && g->defined()) ? g->stride(0) : 0, sdg = (dg && dg->defined()) ? dg->stride(0) : 0;
  DISPATCH_D(D, DISPATCH_A(dog,
    gate_o_bwd_k<NT, AT><<<rgrid<NT>(M), rblock<NT>(), 0, S_()>>>(CP<AT>(dog), CP<AT>(og), CPo<AT>(g), sg, P<AT>(dO), Po<AT>(dg), sdg, P<float>(dd), (int)L,
        (int)hd, tf32, M);
  ))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// rows per block of qkv_bwd (the partial rows: ceil(M / rpb)). Without dko only the column sums of dq are made ([dbq] per block).
void qkv_bwd(at::Tensor dq, c10::optional<at::Tensor> dk, c10::optional<at::Tensor> dv, c10::optional<at::Tensor> qr, c10::optional<at::Tensor> kr,
             c10::optional<at::Tensor> stqk, c10::optional<at::Tensor> wq, c10::optional<at::Tensor> wk, c10::optional<at::Tensor> dqo,
             c10::optional<at::Tensor> dko, c10::optional<at::Tensor> dvo, at::Tensor part, int64_t rpb, bool kq) {
  const c10::cuda::CUDAGuard guard(dq.device());
  const int D = dq.size(1);
  const long M = dq.size(0), sPart = part.size(1);
  const long blocks = (M + rpb - 1) / rpb;
  const bool store = dko && dko->defined();
  TORCH_CHECK(part.size(0) >= blocks, "qkv_bwd: partial rows");
  TORCH_CHECK(!kq || store, "qkv_bwd: kq needs the outputs");
  auto s0 = [](const c10::optional<at::Tensor>& t) -> long { return (t && t->defined()) ? t->stride(0) : 0L; };
  const bool in_bf = is_bf(dq);
  const at::Tensor& adt = store ? *dko : dq;
  const int NTX = D == 768 ? 192 : 32;
  TORCH_CHECK(D == 768 || D == 128, "qkv_bwd: D");
#define QB_LAUNCH(NTV, ITV, ATV)                                                                                                          \
  qkv_bwd_k<NTV, ITV, ATV><<<blocks, NTV, 0, S_()>>>(CP<ITV>(dq), dq.stride(0), CPo<ITV>(dk), s0(dk), CPo<ITV>(dv), s0(dv), CPo<ATV>(qr), s0(qr),     \
      CPo<ATV>(kr), s0(kr), CPo<float4>(stqk), CPo<float>(wq), CPo<float>(wk), Po<ATV>(dqo), s0(dqo), Po<ATV>(dko), s0(dko), Po<ATV>(dvo), s0(dvo),   \
      P<float>(part), sPart, (int)rpb, kq, M);
  if (is_bf(adt)) {
    if (NTX == 192) { if (in_bf) { QB_LAUNCH(192, bf, bf) } else { QB_LAUNCH(192, float, bf) } }
    else { if (in_bf) { QB_LAUNCH(32, bf, bf) } else { QB_LAUNCH(32, float, bf) } }
  } else {
    if (NTX == 192) { if (in_bf) { QB_LAUNCH(192, bf, float) } else { QB_LAUNCH(192, float, float) } }
    else { if (in_bf) { QB_LAUNCH(32, bf, float) } else { QB_LAUNCH(32, float, float) } }
  }
#undef QB_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// das: the GEMM results da_i [M, D] (row strides), sc_cols / off_sc / off_sh per input; pass: fp32 or XT [M, D]
void adaln_bwd(at::Tensor x, at::Tensor st, at::Tensor pass, std::vector<at::Tensor> das, at::Tensor tab, std::vector<int64_t> sc_cols,
               std::vector<int64_t> off_sc, std::vector<int64_t> off_sh, at::Tensor dx, at::Tensor part, int64_t L, int64_t S, bool sig) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int D = x.size(1);
  const long M = x.size(0), sTab = tab.size(1), sPart = part.size(2);
  const int A = M / L;
  AdP p{};
  p.n = (int)das.size();
  TORCH_CHECK(p.n >= 1 && p.n <= 3 && sc_cols.size() == das.size() && off_sc.size() == das.size() && off_sh.size() == das.size(), "adaln_bwd: inputs");
  const size_t es = tab.element_size();
  for (int i = 0; i < p.n; ++i) {
    TORCH_CHECK(das[i].stride(1) == 1, "adaln_bwd: da rows");
    p.da[i] = das[i].data_ptr();
    p.sda[i] = das[i].stride(0);
    p.sc[i] = reinterpret_cast<const char*>(tab.data_ptr()) + sc_cols[i] * es;
    p.offSc[i] = (int)off_sc[i];
    p.offSh[i] = (int)off_sh[i];
  }
  const bool pass32 = pass.scalar_type() == at::kFloat;
  DISPATCH_D(D, DISPATCH_X(x, DISPATCH_A(tab, {
    if (pass32)
      adaln_bwd_k<NT, float, XT, AT><<<rgrid<NT>(L * S), rblock<NT>(), 0, S_()>>>(CP<XT>(x), CP<float2>(st), CP<float>(pass), p, sTab, P<XT>(dx),
          P<float>(part), sPart, (int)L, (int)S, A, sig);
    else
      adaln_bwd_k<NT, XT, XT, AT><<<rgrid<NT>(L * S), rblock<NT>(), 0, S_()>>>(CP<XT>(x), CP<float2>(st), CP<XT>(pass), p, sTab, P<XT>(dx),
          P<float>(part), sPart, (int)L, (int)S, A, sig);
  })))
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int C, typename ZT, typename OT> void pln_fwd_launch(const at::Tensor& z, const at::Tensor& out, double eps) {
  const long r = z.size(0);
  pair_ln_fwd_k<C, ZT, OT><<<(r + 7) / 8, 256, 0, S_()>>>(CP<ZT>(z), P<OT>(out), r, (float)eps);
}
template <int C, typename ZT, typename OT> void pln_bwd_launch(const at::Tensor& z, const at::Tensor& dxh, const at::Tensor& dz, double eps) {
  const long r = z.size(0);
  pair_ln_bwd_k<C, ZT, OT><<<(r + 7) / 8, 256, 0, S_()>>>(CP<ZT>(z), CP<OT>(dxh), P<ZT>(dz), r, (float)eps);
}

// out = LN(z) rows [R, C] (C 128 | 256); z / out bf16 or fp32 in any combination
void pair_ln_fwd(at::Tensor z, at::Tensor out, double eps) {
  const c10::cuda::CUDAGuard guard(z.device());
  const int64_t c = z.size(1);
  TORCH_CHECK((c == 128 || c == 256) && z.is_contiguous() && out.is_contiguous() && out.sizes() == z.sizes(), "pair_ln_fwd: contiguous [rows, 128 | 256]");
  const bool zb = is_bf(z), ob = is_bf(out);
#define PLN(C) if (zb) { if (ob) pln_fwd_launch<C, bf, bf>(z, out, eps); else pln_fwd_launch<C, bf, float>(z, out, eps); } \
               else { if (ob) pln_fwd_launch<C, float, bf>(z, out, eps); else pln_fwd_launch<C, float, float>(z, out, eps); }
  if (c == 128) { PLN(128) } else { PLN(256) }
#undef PLN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dz (z's dtype) = LN backward of the gradient dxh (bf16 or fp32) of the normalised rows
void pair_ln_bwd(at::Tensor z, at::Tensor dxh, at::Tensor dz, double eps) {
  const c10::cuda::CUDAGuard guard(z.device());
  const int64_t c = z.size(1);
  TORCH_CHECK((c == 128 || c == 256) && z.is_contiguous() && dxh.is_contiguous() && dz.is_contiguous() && dxh.sizes() == z.sizes() && dz.sizes() == z.sizes(),
              "pair_ln_bwd: contiguous [rows, 128 | 256]");
  const bool zb = is_bf(z), ob = is_bf(dxh);
#define PLN(C) if (zb) { if (ob) pln_bwd_launch<C, bf, bf>(z, dxh, dz, eps); else pln_bwd_launch<C, bf, float>(z, dxh, dz, eps); } \
               else { if (ob) pln_bwd_launch<C, float, bf>(z, dxh, dz, eps); else pln_bwd_launch<C, float, float>(z, dxh, dz, eps); }
  if (c == 128) { PLN(128) } else { PLN(256) }
#undef PLN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out [cols, rows] = in [rows, cols]^T, fp32 or bf16, tiled through shared memory
template <typename T>
__global__ void __launch_bounds__(256) transpose_k(const T* __restrict__ IN, T* __restrict__ OUT, long rows, long cols) {
  __shared__ T tile[32][33];
  const long c0 = (long)blockIdx.x * 32, r0 = (long)blockIdx.y * 32;
  const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
  for (int i = ty; i < 32; i += 8) {
    const long r = r0 + i, c = c0 + tx;
    if (r < rows && c < cols) tile[i][tx] = IN[r * cols + c];
  }
  __syncthreads();
  for (int i = ty; i < 32; i += 8) {
    const long c = c0 + i, r = r0 + tx;
    if (r < rows && c < cols) OUT[c * rows + r] = tile[tx][i];
  }
}

void transpose2d(at::Tensor in, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(in.device());
  TORCH_CHECK(in.dim() == 2 && in.stride(1) == 1 && in.stride(0) == in.size(1) && out.is_contiguous() && out.size(0) == in.size(1) && out.size(1) == in.size(0) &&
              in.scalar_type() == out.scalar_type(), "transpose2d: contiguous [rows, cols] -> [cols, rows]");
  const long rows = in.size(0), cols = in.size(1);
  dim3 grid((cols + 31) / 32, (rows + 31) / 32);
  if (is_bf(in)) transpose_k<bf><<<grid, 256, 0, S_()>>>(CP<bf>(in), P<bf>(out), rows, cols);
  else transpose_k<float><<<grid, 256, 0, S_()>>>(CP<float>(in), P<float>(out), rows, cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod_fwd", &ln_mod_fwd);
  m.def("qkv_prep", &qkv_prep);
  m.def("res_ln_fwd", &res_ln_fwd);
  m.def("res_out_fwd", &res_out_fwd);
  m.def("gate_fwd", &gate_fwd);
  m.def("glu_fwd", &glu_fwd);
  m.def("glu_bwd", &glu_bwd);
  m.def("res_bwd", &res_bwd);
  m.def("res_adaln_bwd", &res_adaln_bwd);
  m.def("gate_o_bwd", &gate_o_bwd);
  m.def("qkv_bwd", &qkv_bwd);
  m.def("adaln_bwd", &adaln_bwd);
  m.def("pair_ln_fwd", &pair_ln_fwd);
  m.def("pair_ln_bwd", &pair_ln_bwd);
  m.def("transpose2d", &transpose2d);
}
