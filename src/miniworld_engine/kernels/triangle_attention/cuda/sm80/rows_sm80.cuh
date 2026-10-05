// rows_sm80.cuh -- the row kernels of the A100 (sm_80) triangle attention at any width: d_pair a multiple of 64 up to 512, H heads of 32 channels (a 16-channel head
// is zero-padded to 32 inside the packed weights, as the B200 path does), bf16 activations.  They surround cuBLAS: the front is ``ln_rows`` (the input LayerNorm, writing
// the normalised input with a constant column) + one GEMM against the packed weights + ``bias_planes``; the back is ``gate_rows`` + one GEMM + ``out_rows`` (the dropout
// scale and the residual); the training backward adds ``dy_rows``, ``gate_bwd_rows``, ``ln_bwd_rows`` and the weight-gradient finalizer.  All are memory streams.
//
// Frames: token t = (z L + a) L + b of the "starting" problem.  ``transposed`` (the ending node) reads / writes the module's tensors at row (b L + a).
// Notation: D = d_pair, DHp = H * 32 (the padded hidden width), N = 4 DHp + Hpad (the GEMM's output columns: q | k | v | g, then the bias heads padded to a multiple of 8).
//
// Folding (as f1_sm80.cuh): xh = (x - mean) rstd rounded to bf16, [q | k | v | g | bias] = [xh | 1 | 1 | 0 ...] . [W diag(gamma) | b_hi | b_lo | 0 ...]^T with b = W beta
// split into two bf16 (hi + lo) so the shift reaches the fp32 accumulator to 16 mantissa bits: the GEMM is a plain GEMM, with K = D + 8.
#pragma once
#include "f3_sm80.cuh"
#include "wgrad_sm80.cuh"

namespace a100 {

DEVI float bf16_min_f() { return -3.3895313892515355e38f; }

// ------------------------------------------------------------------------------------------------------------------------ weights
// Packed row r of the GEMM's weights: r < 4 DHp: projection p = r / DHp (q, k, v, g), head h, channel c (a padded channel or head: a zero row); r >= 4 DHp: bias head r - 4 DHp.
struct WPackParams {
  const __nv_bfloat16* w[5];    // wq, wk, wv, wg [H hd][D]; wb [H][D]
  const float* gamma;           // [D] fp32 (folded)
  const float* beta;            // [D] fp32 (folded)
  __nv_bfloat16* out;           // folded: [NR][D + 8]; plain: [NR][D] (NR = 4 DHp: the four projections only)
  int D, H, hd, DHp, NR, folded;
};

__global__ void __launch_bounds__(256) w_pack_kernel(const WPackParams p) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= p.NR) return;
  const size_t ld = p.folded ? (size_t)p.D + 8 : (size_t)p.D;
  const __nv_bfloat16* src = nullptr;
  if (row < 4 * p.DHp) {
    const int pj = row / p.DHp, r = row - pj * p.DHp, h = r >> 5, c = r & 31;
    if (h < p.H && c < p.hd) src = p.w[pj] + (size_t)(h * p.hd + c) * p.D;
  } else {
    const int h = row - 4 * p.DHp;
    if (h < p.H) src = p.w[4] + (size_t)h * p.D;
  }
  float dotb = 0.f;
  for (int c2 = lane; c2 < p.D / 2; c2 += 32) {
    uint32_t u = 0u;
    if (src != nullptr) {
      const uint32_t raw = reinterpret_cast<const uint32_t*>(src)[c2];
      if (p.folded) {
        const float w0 = bf16lo(raw), w1 = bf16hi(raw);
        const float2 g = reinterpret_cast<const float2*>(p.gamma)[c2], b = reinterpret_cast<const float2*>(p.beta)[c2];
        u = pack_bf16(w0 * g.x, w1 * g.y);
        dotb = fmaf(w1, b.y, fmaf(w0, b.x, dotb));
      } else {
        u = raw;
      }
    }
    reinterpret_cast<uint32_t*>(p.out + (size_t)row * ld)[c2] = u;
  }
  if (p.folded) {
    dotb = warp_sum(dotb);                                       // the butterfly order is fixed: replays are bit-identical
    if (lane < 8) {                                              // columns D .. D + 7: b_hi, b_lo, 0 ...
      const __nv_bfloat16 hi = __float2bfloat16_rn(dotb);
      const __nv_bfloat16 lo = __float2bfloat16_rn(dotb - __bfloat162float(hi));
      p.out[(size_t)row * ld + p.D + lane] = lane == 0 ? hi : (lane == 1 ? lo : __float2bfloat16_rn(0.f));
    }
  }
}

// Wo [D][H hd] -> the padded [D][DHp] (zero columns for the padded channels of a 16-channel head).
__global__ void __launch_bounds__(256) wo_pad_kernel(const __nv_bfloat16* __restrict__ wo, __nv_bfloat16* __restrict__ out, int D, int H, int hd) {
  const int DHp = H * 32;
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)D * DHp) return;
  const int o = (int)(i / DHp), j = (int)(i - (size_t)o * DHp), h = j >> 5, c = j & 31;
  out[i] = c < hd ? wo[(size_t)o * (H * hd) + h * hd + c] : __float2bfloat16_rn(0.f);
}

// ------------------------------------------------------------------------------------------------------------------------ LayerNorm rows
struct LnRowsParams {
  const __nv_bfloat16* x;       // [T][D], the module's layout
  __nv_bfloat16* xh;            // [T][D + 8]: bf16((x - mean) rstd) | 1 | 1 | 0 ... (the starting frame)
  float* stats;                 // [T][2] (mean, rstd) or nullptr
  unsigned T;
  int L, transposed;
  float eps;
};

// A row is NCH = D / 8 chunks of 8 channels; TPR threads take a row (NCH when it divides 32, else a whole warp), CPT chunks each.
template <int D>
struct RowShape {
  static constexpr int NCH = D / 8;
  static constexpr int TPR = (NCH <= 32 && 32 % NCH == 0) ? NCH : 32;
  static constexpr int CPT = (NCH + TPR - 1) / TPR;
  static constexpr int RPW = 32 / TPR;                           // rows per warp step
};

DEVI float group_sum(float v, int width) {                       // sum over the aligned group of `width` lanes (a power of two)
  for (int o = width >> 1; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

DEVI size_t module_row(size_t t, unsigned L, int transposed) {  // the token's row in the module's layout
  if (!transposed) return t;
  const size_t LL = (size_t)L * L, z = t / LL, rem = t - z * LL, a = rem / L, b = rem - a * L;
  return z * LL + b * L + a;
}

template <int D>
__global__ void __launch_bounds__(256) ln_rows_kernel(const LnRowsParams p) {
  using S = RowShape<D>;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sub = lane / S::TPR, tl = lane - sub * S::TPR;
  for (unsigned grp = blockIdx.x * 8 + warp;; grp += gridDim.x * 8) {
    const size_t t = (size_t)grp * S::RPW + sub;
    if ((size_t)grp * S::RPW >= p.T) break;
    const bool valid = t < p.T;
    uint4 v[S::CPT];
    const __nv_bfloat16* xr = p.x + module_row(valid ? t : 0, p.L, p.transposed) * D;
#pragma unroll
    for (int i = 0; i < S::CPT; ++i) {
      const int ch = tl + i * S::TPR;
      v[i] = (ch < S::NCH) ? ldg128(xr + ch * 8) : make_uint4(0u, 0u, 0u, 0u);
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < S::CPT; ++i)
      s += (bf16lo(v[i].x) + bf16hi(v[i].x)) + (bf16lo(v[i].y) + bf16hi(v[i].y)) + (bf16lo(v[i].z) + bf16hi(v[i].z)) + (bf16lo(v[i].w) + bf16hi(v[i].w));
    const float mean = group_sum(s, S::TPR) * (1.f / D);
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < S::CPT; ++i) {
      if (tl + i * S::TPR >= S::NCH) continue;
      float d;
      d = bf16lo(v[i].x) - mean; q = fmaf(d, d, q);  d = bf16hi(v[i].x) - mean; q = fmaf(d, d, q);
      d = bf16lo(v[i].y) - mean; q = fmaf(d, d, q);  d = bf16hi(v[i].y) - mean; q = fmaf(d, d, q);
      d = bf16lo(v[i].z) - mean; q = fmaf(d, d, q);  d = bf16hi(v[i].z) - mean; q = fmaf(d, d, q);
      d = bf16lo(v[i].w) - mean; q = fmaf(d, d, q);  d = bf16hi(v[i].w) - mean; q = fmaf(d, d, q);
    }
    const float rstd = rsqrtf(group_sum(q, S::TPR) * (1.f / D) + p.eps);
    if (!valid) continue;
    __nv_bfloat16* xo = p.xh + t * (D + 8);
#pragma unroll
    for (int i = 0; i < S::CPT; ++i) {
      const int ch = tl + i * S::TPR;
      if (ch >= S::NCH) continue;
      stg128(xo + ch * 8, make_uint4(pack_bf16((bf16lo(v[i].x) - mean) * rstd, (bf16hi(v[i].x) - mean) * rstd),
                                     pack_bf16((bf16lo(v[i].y) - mean) * rstd, (bf16hi(v[i].y) - mean) * rstd),
                                     pack_bf16((bf16lo(v[i].z) - mean) * rstd, (bf16hi(v[i].z) - mean) * rstd),
                                     pack_bf16((bf16lo(v[i].w) - mean) * rstd, (bf16hi(v[i].w) - mean) * rstd)));
    }
    if (tl == 0) {
      stg128(xo + D, make_uint4(0x3f803f80u, 0u, 0u, 0u));       // the constant columns: 1, 1, 0 ... (bf16)
      if (p.stats != nullptr) *reinterpret_cast<float2*>(p.stats + 2 * t) = make_float2(mean, rstd);
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------ bias planes
struct BiasPlanesParams {
  const __nv_bfloat16* y;       // [T][ldy]: columns col0 .. col0 + Hpad hold the pre-mask bias heads
  const uint8_t* mask;          // [Z][L] key mask (non-zero = real key) or nullptr
  __nv_bfloat16* bias;          // [Z][H][L][L]
  long long ldy;
  unsigned T;
  int col0, H, L;
};

// bias[z][h][a][b] = keep(z, b) ? y[t][col0 + h] : bf16 min (the module's masked_fill); one thread per token, the stores of a warp are 64 consecutive bytes of a plane.
__global__ void __launch_bounds__(256) bias_planes_kernel(const BiasPlanesParams p) {
  const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (t >= p.T) return;
  const size_t LL = (size_t)p.L * p.L, z = t / LL, rem = t - z * LL, b = rem % p.L;
  const bool keep = p.mask == nullptr || p.mask[z * p.L + b] != 0;
  const __nv_bfloat16 neg = __float2bfloat16_rn(bf16_min_f());
  for (int g = 0; g * 8 < p.H; ++g) {
    const uint4 u = *reinterpret_cast<const uint4*>(p.y + t * p.ldy + p.col0 + 8 * g);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int h = 8 * g + j;
      if (h < p.H) {
        const uint32_t word = w[j >> 1];
        p.bias[((z * p.H + h) * LL) + rem] = keep ? __ushort_as_bfloat16((unsigned short)((j & 1) ? (word >> 16) : (word & 0xffffu))) : neg;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------ the back's rows
// a = bf16(sigmoid(g) o) over [T][DHp]; g is the gate columns of the GEMM's output (row stride ldg).  Block = (DHp / 8 chunks, 8 tokens).
struct GateRowsParams {
  const __nv_bfloat16* o;       // [T][DHp]
  const __nv_bfloat16* g;       // [T][ldg]
  __nv_bfloat16* a;             // [T][DHp]
  long long ldg;
  unsigned T;
  int DHp;
};

__global__ void __launch_bounds__(512) gate_rows_kernel(const GateRowsParams p) {
  const size_t t = (size_t)blockIdx.x * 8 + threadIdx.y;
  if (t >= p.T) return;
  const int c = threadIdx.x;
  const uint4 ov = ldg128(p.o + t * p.DHp + c * 8), gv = ldg128(p.g + t * p.ldg + c * 8);
  stg128(p.a + t * p.DHp + c * 8, make_uint4(gate_pair(ov.x, gv.x), gate_pair(ov.y, gv.y), gate_pair(ov.z, gv.z), gate_pair(ov.w, gv.w)));
}

// out = bf16(res + bf16(y ds)): y [T][D] (the starting frame), res / out in the module's layout, ds [Z][L][D] indexed by the token's second index (or nullptr).  Block = (D / 8, 8).
struct OutRowsParams {
  const __nv_bfloat16* y;
  const __nv_bfloat16* res;
  const __nv_bfloat16* ds;
  __nv_bfloat16* out;
  unsigned T;
  int L, D, transposed;
};

__global__ void __launch_bounds__(512) out_rows_kernel(const OutRowsParams p) {
  const size_t t = (size_t)blockIdx.x * 8 + threadIdx.y;
  if (t >= p.T) return;
  const int c = threadIdx.x;
  const size_t row = module_row(t, p.L, p.transposed);
  uint4 y = ldg128(p.y + t * p.D + c * 8);
  if (p.ds != nullptr) {
    const size_t LL = (size_t)p.L * p.L, z = t / LL, b = (t - z * LL) % p.L;
    const uint4 d = ldg128(p.ds + (z * p.L + b) * p.D + c * 8);
    y = make_uint4(mul_bf16x2(y.x, d.x), mul_bf16x2(y.y, d.y), mul_bf16x2(y.z, d.z), mul_bf16x2(y.w, d.w));
  }
  const uint4 r = ldg128(p.res + row * p.D + c * 8);
  stg128(p.out + row * p.D + c * 8, make_uint4(add_bf16x2(r.x, y.x), add_bf16x2(r.y, y.y), add_bf16x2(r.z, y.z), add_bf16x2(r.w, y.w)));
}

// ------------------------------------------------------------------------------------------------------------------------ the backward's rows
// dy = bf16(dout ds) in the starting frame: dout in the module's layout (read at the transposed position for the ending node), ds [Z][L][D] or nullptr.  Block = (D / 8, 8).
struct DyRowsParams {
  const __nv_bfloat16* dout;
  const __nv_bfloat16* ds;
  __nv_bfloat16* dy;
  unsigned T;
  int L, D, transposed;
};

__global__ void __launch_bounds__(512) dy_rows_kernel(const DyRowsParams p) {
  const size_t t = (size_t)blockIdx.x * 8 + threadIdx.y;
  if (t >= p.T) return;
  const int c = threadIdx.x;
  uint4 v = ldg128(p.dout + module_row(t, p.L, p.transposed) * p.D + c * 8);
  if (p.ds != nullptr) {
    const size_t LL = (size_t)p.L * p.L, z = t / LL, b = (t - z * LL) % p.L;
    const uint4 d = ldg128(p.ds + (z * p.L + b) * p.D + c * 8);
    v = make_uint4(mul_bf16x2(v.x, d.x), mul_bf16x2(v.y, d.y), mul_bf16x2(v.z, d.z), mul_bf16x2(v.w, d.w));
  }
  stg128(p.dy + t * p.D + c * 8, v);
}

// The gate's gradients (b3_sm80.cuh's epilogue) from da = bf16(dy Wo) [T][DHp], o and g:  s = sigmoid(g),  do = bf16(da s),  dg = bf16(da o s (1 - s)),  a = bf16(o s)  and
// delta[z][h][a][b] = sum over the head's 32 channels of o do (the attention backward's row term).  Block = (DHp / 8 chunks, 8 tokens): a head is 4 consecutive threads.
struct GateBwdParams {
  const __nv_bfloat16* da;      // [T][DHp]
  const __nv_bfloat16* o;       // [T][DHp]
  const __nv_bfloat16* g;       // [T][ldg]
  __nv_bfloat16* dg;            // [T][lddg]
  __nv_bfloat16* dov;           // [T][DHp]
  __nv_bfloat16* a;             // [T][DHp]
  float* delta;                 // [Z][H][L][L] or nullptr
  long long ldg, lddg;
  unsigned T;
  int DHp, L, H;
};

__global__ void __launch_bounds__(512) gate_bwd_rows_kernel(const GateBwdParams p) {
  extern __shared__ float sdelta[];                              // [H][8 tokens]
  const size_t t = (size_t)blockIdx.x * 8 + threadIdx.y;
  const int c = threadIdx.x;
  const bool valid = t < p.T;
  const size_t tt = valid ? t : 0;
  const uint4 du = ldg128(p.da + tt * p.DHp + c * 8), ou = ldg128(p.o + tt * p.DHp + c * 8), gu = ldg128(p.g + tt * p.ldg + c * 8);
  const uint32_t dw[4] = {du.x, du.y, du.z, du.w}, ow[4] = {ou.x, ou.y, ou.z, ou.w}, gw[4] = {gu.x, gu.y, gu.z, gu.w};
  uint32_t d_o[4], d_g[4], av[4];
  float dl = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float da0 = bf16lo(dw[j]), da1 = bf16hi(dw[j]);
    const float o0 = bf16lo(ow[j]), o1 = bf16hi(ow[j]);
    const float s0 = sigmoid(bf16lo(gw[j])), s1 = sigmoid(bf16hi(gw[j]));
    d_o[j] = pack_bf16(da0 * s0, da1 * s1);
    d_g[j] = pack_bf16(da0 * o0 * (s0 * (1.f - s0)), da1 * o1 * (s1 * (1.f - s1)));
    av[j] = pack_bf16(o0 * s0, o1 * s1);
    dl = fmaf(o0, bf16lo(d_o[j]), fmaf(o1, bf16hi(d_o[j]), dl));
  }
  if (valid) {
    stg128(p.dov + t * p.DHp + c * 8, make_uint4(d_o[0], d_o[1], d_o[2], d_o[3]));
    stg128(p.dg + t * p.lddg + c * 8, make_uint4(d_g[0], d_g[1], d_g[2], d_g[3]));
    stg128(p.a + t * p.DHp + c * 8, make_uint4(av[0], av[1], av[2], av[3]));
  }
  if (p.delta != nullptr) {
    dl = quad_sum(dl);
    if ((c & 3) == 0) sdelta[(c >> 2) * 8 + threadIdx.y] = dl;
    __syncthreads();
    const int n = p.H * 8;
    const int lin = threadIdx.y * blockDim.x + threadIdx.x;
    if (lin < n) {                                               // (head, token of the block): 8 consecutive tokens of a head are one 32-byte sector
      const int h = lin >> 3, k = lin & 7;
      const size_t tk = (size_t)blockIdx.x * 8 + k;
      if (tk < p.T) {
        const size_t LL = (size_t)p.L * p.L, z = tk / LL, rem = tk - z * LL;
        p.delta[(z * p.H + h) * LL + rem] = sdelta[h * 8 + k];
      }
    }
  }
}

// The front's input gradient: dxn = bf16(D W) [T][D] (the GEMM; D = [dq | dk | dv | dg]) plus the bias heads' term  sum_h db[h] Wb[h]  (fp32), then the LayerNorm backward and the
// residual (f1/b1_sm80.cuh's arithmetic):  dxh = dxn gamma,  xh = (x - mean) rstd,  dx = bf16(rstd (dxh - mean(dxh) - xh mean(dxh xh))),  dpair = bf16(dout + dx).
struct LnBwdParams {
  const __nv_bfloat16* dxn;     // [T][D] the starting frame
  const __nv_bfloat16* db;      // [Z][H][L][L] the pair bias' gradient planes
  const __nv_bfloat16* wb;      // [H][D]
  const __nv_bfloat16* x;       // [T][D] the module's layout
  const float* stats;           // [T][2]
  const __nv_bfloat16* dout;    // [T][D] the module's layout
  const float* gamma;           // [D] fp32
  __nv_bfloat16* dpair;         // [T][D] the module's layout
  unsigned T;
  int L, H, transposed;
};

template <int D>
__global__ void __launch_bounds__(256) ln_bwd_rows_kernel(const LnBwdParams p) {
  using S = RowShape<D>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* wbs = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // [H][D]
  for (int i = threadIdx.x; i < p.H * D / 8; i += 256) reinterpret_cast<uint4*>(wbs)[i] = reinterpret_cast<const uint4*>(p.wb)[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sub = lane / S::TPR, tl = lane - sub * S::TPR;
  const size_t LL = (size_t)p.L * p.L;
  for (unsigned grp = blockIdx.x * 8 + warp;; grp += gridDim.x * 8) {
    const size_t t = (size_t)grp * S::RPW + sub;
    if ((size_t)grp * S::RPW >= p.T) break;
    const bool valid = t < p.T;
    const size_t tt = valid ? t : 0, row = module_row(tt, p.L, p.transposed), z = tt / LL, rem = tt - z * LL;
    uint4 xv[S::CPT], dv[S::CPT], gv[S::CPT];
#pragma unroll
    for (int i = 0; i < S::CPT; ++i) {
      const int ch = tl + i * S::TPR;
      const bool ok = ch < S::NCH;
      xv[i] = ok ? ldg128(p.x + row * D + ch * 8) : make_uint4(0u, 0u, 0u, 0u);
      dv[i] = ok ? ldg128(p.dxn + tt * D + ch * 8) : make_uint4(0u, 0u, 0u, 0u);
    }
    const float2 st = *reinterpret_cast<const float2*>(p.stats + 2 * tt);
    const float mean = st.x, rstd = st.y;
    float dbh[16];
#pragma unroll
    for (int h = 0; h < 16; ++h) dbh[h] = h < p.H ? __bfloat162float(p.db[(z * p.H + h) * LL + rem]) : 0.f;
    float dxh[S::CPT][8], xh[S::CPT][8];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int i = 0; i < S::CPT; ++i) {
      const int ch = tl + i * S::TPR;
      const bool ok = ch < S::NCH;
      const uint32_t dw[4] = {dv[i].x, dv[i].y, dv[i].z, dv[i].w}, xw[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
      float d[8], gm[8];
      const float4 g0 = ok ? *reinterpret_cast<const float4*>(p.gamma + ch * 8) : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 g1 = ok ? *reinterpret_cast<const float4*>(p.gamma + ch * 8 + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
      gm[0] = g0.x; gm[1] = g0.y; gm[2] = g0.z; gm[3] = g0.w; gm[4] = g1.x; gm[5] = g1.y; gm[6] = g1.z; gm[7] = g1.w;
#pragma unroll
      for (int j = 0; j < 4; ++j) { d[2 * j] = bf16lo(dw[j]); d[2 * j + 1] = bf16hi(dw[j]); }
      if (ok) {
#pragma unroll
        for (int h = 0; h < 16; ++h) {
          if (h < p.H) {
            const uint4 wv = *reinterpret_cast<const uint4*>(wbs + h * D + ch * 8);
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) { d[2 * j] = fmaf(dbh[h], bf16lo(ww[j]), d[2 * j]); d[2 * j + 1] = fmaf(dbh[h], bf16hi(ww[j]), d[2 * j + 1]); }
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        xh[i][2 * j] = (bf16lo(xw[j]) - mean) * rstd;
        xh[i][2 * j + 1] = (bf16hi(xw[j]) - mean) * rstd;
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        dxh[i][e] = ok ? d[e] * gm[e] : 0.f;
        s1 += dxh[i][e];
        s2 = fmaf(dxh[i][e], ok ? xh[i][e] : 0.f, s2);
      }
    }
    const float m1 = group_sum(s1, S::TPR) * (1.f / D), m2 = group_sum(s2, S::TPR) * (1.f / D);
    if (!valid) continue;
#pragma unroll
    for (int i = 0; i < S::CPT; ++i) {
      const int ch = tl + i * S::TPR;
      if (ch >= S::NCH) continue;
      const uint4 ou = ldg128(p.dout + row * D + ch * 8);
      const uint32_t ow[4] = {ou.x, ou.y, ou.z, ou.w};
      uint32_t r[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        r[j] = add_bf16x2(ow[j], pack_bf16(rstd * (dxh[i][2 * j] - m1 - xh[i][2 * j] * m2), rstd * (dxh[i][2 * j + 1] - m1 - xh[i][2 * j + 1] * m2)));
      stg128(p.dpair + row * D + ch * 8, make_uint4(r[0], r[1], r[2], r[3]));
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------ parameter gradients
// From G = D^T [xh | 1 | 1 | 0 ...] (fp32 [NG][D + 8], NG = 4 DHp + H: the four projections' padded rows, then the bias heads):
//   dW[o][c] = G[o][c] gamma_c + G[o][D] beta_c   (the real rows only: head h channel c < hd of projection p)
//   dgamma_c = sum_o W[o][c] G[o][c]              dbeta_c = sum_o W[o][c] G[o][D]
// and dWo [D][H hd] from Go [D][DHp] (the padded columns dropped).  Blocks: [0, NW) one per real weight row, then D blocks (channel), then D blocks (dWo rows).
struct WgradWParams {
  const float* g;               // [NG][D + 8]
  const float* go;              // [D][DHp] fp32 (dWo before the cast)
  const __nv_bfloat16* w[5];    // wq, wk, wv, wg [H hd][D]; wb [H][D]
  __nv_bfloat16* dw[6];         // dwq .. dwb [.. ][D] (bf16), dwo [D][H hd]
  const void* gamma;            // [D] fp32 or bf16 (the LayerNorm parameters' dtype)
  const void* beta;
  void* dgamma;
  void* dbeta;
  int D, H, hd, DHp, ln_bf16, slot;    // slot = channels per head slot of the hidden layout: 32 (heads padded to 32) or hd (native)
};

__global__ void __launch_bounds__(256) wgrad_w_kernel(const WgradWParams p) {
  const int NW = 4 * p.H * p.hd + p.H;
  const int b = blockIdx.x, tid = threadIdx.x;
  const size_t ldg = (size_t)p.D + 8;
  auto grow = [&](int r) -> int {                                // real weight row r (p, o) -> its row of G
    if (r < 4 * p.H * p.hd) { const int pj = r / (p.H * p.hd), o = r - pj * p.H * p.hd; return pj * p.DHp + (o / p.hd) * p.slot + o % p.hd; }
    return 4 * p.DHp + (r - 4 * p.H * p.hd);
  };
  if (b < NW) {
    const int pj = b < 4 * p.H * p.hd ? b / (p.H * p.hd) : 4, o = b < 4 * p.H * p.hd ? b - pj * p.H * p.hd : b - 4 * p.H * p.hd;
    const float* gr = p.g + (size_t)grow(b) * ldg;
    for (int c = tid; c < p.D; c += 256)
      p.dw[pj][(size_t)o * p.D + c] = __float2bfloat16_rn(gr[c] * ld_param(p.gamma, p.ln_bf16, c) + gr[p.D] * ld_param(p.beta, p.ln_bf16, c));
    return;
  }
  if (b < NW + p.D) {                                            // dgamma / dbeta of channel col: the threads sum rows tid, tid + 256, ... (a fixed tree: deterministic)
    const int col = b - NW;
    __shared__ float red[2][256];
    float a = 0.f, s = 0.f;
    for (int r = tid; r < NW; r += 256) {
      const int pj = r < 4 * p.H * p.hd ? r / (p.H * p.hd) : 4, o = r < 4 * p.H * p.hd ? r - pj * p.H * p.hd : r - 4 * p.H * p.hd;
      const float wv = __bfloat162float(p.w[pj][(size_t)o * p.D + col]);
      const float* gr = p.g + (size_t)grow(r) * ldg;
      a = fmaf(wv, gr[col], a);
      s = fmaf(wv, gr[p.D], s);
    }
    red[0][tid] = a;
    red[1][tid] = s;
    __syncthreads();
#pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
      if (tid < stride) { red[0][tid] += red[0][tid + stride]; red[1][tid] += red[1][tid + stride]; }
      __syncthreads();
    }
    if (tid == 0) {
      st_param(p.dgamma, p.ln_bf16, col, red[0][0]);
      st_param(p.dbeta, p.ln_bf16, col, red[1][0]);
    }
    return;
  }
  const int o = b - NW - p.D;                                    // dWo row o
  for (int j = tid; j < p.H * p.hd; j += 256) p.dw[5][(size_t)o * (p.H * p.hd) + j] = __float2bfloat16_rn(p.go[(size_t)o * p.DHp + (j / p.hd) * p.slot + j % p.hd]);
}

}  // namespace a100
