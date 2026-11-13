// rows_f32_sm120.cuh -- the fp32 inference block's row passes on the RTX 5090 (MWE_SM120 build only; integrations/bias_only_dit_sm80.py
// _block_eval_sm120_f32). The GEMMs between them are cuBLAS (TF32); these passes fold the block's cond-only work into ONE batched GEMM and
// merge the attention's residual gate with the transition's AdaLN:
//
//   cond3         C3[0] = LN(c) w1, C3[1] = LN(c) w2, C3[2] = c          one read of c for both AdaLNs' cond LayerNorms and the gate GEMM's
//                                                                         operand: the three [M, dc] operands of one strided-batched GEMM
//   resgate_adaln out = x + sigmoid(g + bs) y;  xa = sigmoid(S + sb) LN(out) + B    the attention's output gate + residual and the
//                                                                         transition's AdaLN in one pass (out is written: the transition's residual)
//   gate_res_b    y = x + sigmoid(g + b) z                               the transition's output gate (g a strided column block of the GEMM)
//
// One warp owns a row (the LayerNorm rows; no block barriers): lane l holds the float4 vectors v = 0 .. D / 128 - 1 at columns 4 (32 v + l);
// the LayerNorm arithmetic is that of adaln_rows.cuh (cond_ln / adaln_epi): the same sums in the same order. Sigmoids are 1 / (1 + exp(-v)).
#pragma once
#include <cuda_runtime.h>

namespace r120 {

#define R120_DEVI __device__ __forceinline__
constexpr int RW = 4;  // warps (rows) per block

R120_DEVI float sigm(float v) { return 1.f / (1.f + __expf(-v)); }
R120_DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
R120_DEVI float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
R120_DEVI void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
R120_DEVI float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
R120_DEVI float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
R120_DEVI float4 scale4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
R120_DEVI float4 sig4(float4 a) { return make_float4(sigm(a.x), sigm(a.y), sigm(a.z), sigm(a.w)); }
R120_DEVI float sum4(float4 a) { return a.x + a.y + a.z + a.w; }
R120_DEVI int vcol(int v, int lane) { return (v * 32 + lane) * 4; }

// C3 [3][M][D]: LN(c) w1, LN(c) w2, c
template <int D>
__global__ void __launch_bounds__(RW * 32) cond3_kernel(const float* __restrict__ C, long sc, const float* __restrict__ W1, const float* __restrict__ W2, float* __restrict__ C3,
                                                        long M, float eps) {
  constexpr int NV = D / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= M) return;
  float4 c[NV];
  float s = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    c[v] = ld4(C + row * sc + vcol(v, lane));
    st4(C3 + 2 * M * D + row * D + vcol(v, lane), c[v]);
    s += sum4(c[v]);
  }
  const float mean = warp_sum(s) / D;
  float q = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) { c[v] = make_float4(c[v].x - mean, c[v].y - mean, c[v].z - mean, c[v].w - mean); q += sum4(mul4(c[v], c[v])); }
  const float rstd = rsqrtf(warp_sum(q) / D + eps);
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const float4 h = scale4(c[v], rstd);
    st4(C3 + row * D + vcol(v, lane), mul4(h, ld4(W1 + vcol(v, lane))));
    st4(C3 + M * D + row * D + vcol(v, lane), mul4(h, ld4(W2 + vcol(v, lane))));
  }
}

// out = x + sigmoid(g + bs) y; xa = sigmoid(S + sb) LN(out) + B, S | B = a row of SBT [M, 2 D] (row stride ssb); g a row of G (row stride sg)
template <int D>
__global__ void __launch_bounds__(RW * 32) resgate_adaln_kernel(const float* __restrict__ Y, const float* __restrict__ G, long sg, const float* __restrict__ BS,
                                                                const float* __restrict__ X, const float* __restrict__ SBT, long ssb, const float* __restrict__ SB,
                                                                float* __restrict__ OUT, float* __restrict__ XA, long M, float eps) {
  constexpr int NV = D / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= M) return;
  const float* t = SBT + row * ssb;
  float4 x[NV], sc[NV], bi[NV];
  float s = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {  // every load of the row before the first sum
    const int c = vcol(v, lane);
    const float4 y = ld4(Y + row * D + c), g = add4(ld4(G + row * sg + c), ld4(BS + c)), r = ld4(X + row * D + c);
    const float4 gs = sig4(g);
    x[v] = make_float4(fmaf(gs.x, y.x, r.x), fmaf(gs.y, y.y, r.y), fmaf(gs.z, y.z, r.z), fmaf(gs.w, y.w, r.w));
    sc[v] = ld4(t + c);
    bi[v] = ld4(t + D + c);
    st4(OUT + row * D + c, x[v]);
    s += sum4(x[v]);
  }
  const float mean = warp_sum(s) / D;
  float q = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) { x[v] = make_float4(x[v].x - mean, x[v].y - mean, x[v].z - mean, x[v].w - mean); q += sum4(mul4(x[v], x[v])); }
  const float rstd = rsqrtf(warp_sum(q) / D + eps);
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const float4 g = sig4(add4(sc[v], ld4(SB + vcol(v, lane))));
    st4(XA + row * D + vcol(v, lane), add4(mul4(g, scale4(x[v], rstd)), bi[v]));
  }
}

// Y [M, D] = X + sigmoid(G + b) Z (G row stride sg)
template <int D>
__global__ void __launch_bounds__(256) gate_res_b_kernel(const float* __restrict__ X, const float* __restrict__ Z, const float* __restrict__ G, long sg,
                                                         const float* __restrict__ B, float* __restrict__ Y, long M) {
  constexpr int D4 = D / 4;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < M * D4; i += (long)gridDim.x * 256) {
    const long row = i / D4;
    const int c = (int)(i - row * D4) * 4;
    const float4 gz = mul4(sig4(add4(ld4(G + row * sg + c), ld4(B + c))), ld4(Z + row * D + c));
    st4(Y + row * D + c, add4(ld4(X + row * D + c), gz));
  }
}

}  // namespace r120

namespace r120 {

// ---- pair_softmax: P [H, L, L] = softmax over the keys of the pair bias rstd (z . W' - mean sum W') (masked keys 0), fp32, in ONE pass over the pair: a block owns a
// query row i, writes the row's H x L biases to shared memory (never to global memory) and turns them into P there. The pair bias of pair_bias_gen (aux_sm80.cuh:
// the LayerNorm of the C channels, mean / rstd = rsqrt(max(E x^2 - mean^2, 0) + eps), the H dot products with the folded weight W' = Wb diag(ln w)) and the softmax of
// softmax_rows (bias_only_dit_f32_rows.cu: masked keys at -3e38, __expf, the sum pairwise per float4, 1 / sum) -- only the order of the channel sums differs (a warp
// owns an element: lane l takes channels 4 (32 v + l) .. + 3, the H + 2 sums then reduced across the lanes).
constexpr int PS_H = 16, PS_WP = 20;  // heads; the padded W' row (20 floats: a warp's 16-B reads of 8 consecutive rows hit distinct bank quads)
template <int C>
constexpr int ps_smem(int L) { return (PS_H * L + C * PS_WP + PS_H) * 4; }

template <int C, int U>
__global__ void __launch_bounds__(256, 1) pair_softmax_kernel(const float* __restrict__ Z, const float* __restrict__ WCM, const bool* __restrict__ MASK, float* __restrict__ P, int L,
                                                               float eps) {
  constexpr int H = PS_H, NV = C / 128;
  extern __shared__ __align__(16) float pssm[];
  float* S = pssm;                       // [H][L] the row's biases
  float* Wp = S + H * L;                 // [C][PS_WP] W'
  float* sw = Wp + C * PS_WP;            // [H] sum_c W'
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int i = blockIdx.x;
  for (int k = tid; k < C * H; k += 256) Wp[(k / H) * PS_WP + k % H] = WCM[k];
  __syncthreads();
  if (tid < H) {
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += Wp[c * PS_WP + tid];
    sw[tid] = s;
  }
  __syncthreads();
  const int head = ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
  const float* zi = Z + (size_t)i * L * C;
  for (int j0 = warp * U; j0 < L; j0 += 8 * U) {
    float4 x[U][NV];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int v = 0; v < NV; ++v) x[u][v] = __ldg(reinterpret_cast<const float4*>(zi + (size_t)(j0 + u) * C + 4 * (32 * v + lane)));
    float d[U][H], s[U], s2[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      s[u] = s2[u] = 0.f;
#pragma unroll
      for (int h = 0; h < H; ++h) d[u][h] = 0.f;
    }
#pragma unroll
    for (int v = 0; v < NV; ++v)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float4* wr = reinterpret_cast<const float4*>(Wp + (4 * (32 * v + lane) + e) * PS_WP);
        float w[H];
#pragma unroll
        for (int q = 0; q < H / 4; ++q) { const float4 t = wr[q]; w[4 * q] = t.x; w[4 * q + 1] = t.y; w[4 * q + 2] = t.z; w[4 * q + 3] = t.w; }
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const float xv = e == 0 ? x[u][v].x : e == 1 ? x[u][v].y : e == 2 ? x[u][v].z : x[u][v].w;
          s[u] += xv;
          s2[u] = fmaf(xv, xv, s2[u]);
#pragma unroll
          for (int h = 0; h < H; ++h) d[u][h] = fmaf(xv, w[h], d[u][h]);
        }
      }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      // reduce-scatter of the 16 head sums over the 32 lanes: after it lane l holds head `head` (lanes l, l ^ 1 the same value)
#pragma unroll
      for (int o = 16, n = 8; o >= 2; o >>= 1, n >>= 1) {
        const bool up = lane & o;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          if (k < n) {
            const float keep = up ? d[u][n + k] : d[u][k], give = up ? d[u][k] : d[u][n + k];
            d[u][k] = keep + __shfl_xor_sync(0xffffffffu, give, o);
          }
        }
      }
      d[u][0] += __shfl_xor_sync(0xffffffffu, d[u][0], 1);
#pragma unroll
      for (int o = 16; o; o >>= 1) { s[u] += __shfl_xor_sync(0xffffffffu, s[u], o); s2[u] += __shfl_xor_sync(0xffffffffu, s2[u], o); }
      const float invc = 1.f / (float)C, mean = s[u] * invc, rstd = rsqrtf(fmaxf(s2[u] * invc - mean * mean, 0.f) + eps);
      if (!(lane & 1)) S[head * L + j0 + u] = rstd * (d[u][0] - mean * sw[head]);
    }
  }
  __syncthreads();
  // the softmax of the row's H x L biases: warp w takes heads w and w + 8
  constexpr float NEG = -3.0e38f;
  const int nch = L / 4;
#pragma unroll 1
  for (int hh = 0; hh < 2; ++hh) {
    const int h = warp + 8 * hh;
    const float* b = S + h * L;
    float* out = P + ((size_t)h * L + i) * L;
    float mx = NEG;
    float4 vv[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int ch = lane + 32 * c;
      vv[c] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (ch < nch) {
        float4 e = *reinterpret_cast<const float4*>(b + ch * 4);
        if (MASK) {
          const uchar4 mk = *reinterpret_cast<const uchar4*>(MASK + ch * 4);
          if (!mk.x) e.x = NEG;
          if (!mk.y) e.y = NEG;
          if (!mk.z) e.z = NEG;
          if (!mk.w) e.w = NEG;
        }
        vv[c] = e;
        mx = fmaxf(mx, fmaxf(fmaxf(e.x, e.y), fmaxf(e.z, e.w)));
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c)
      if (lane + 32 * c < nch) {
        vv[c] = make_float4(__expf(vv[c].x - mx), __expf(vv[c].y - mx), __expf(vv[c].z - mx), __expf(vv[c].w - mx));
        sum += (vv[c].x + vv[c].y) + (vv[c].z + vv[c].w);
      }
    const float inv = 1.f / warp_sum(sum);
#pragma unroll
    for (int c = 0; c < 8; ++c)
      if (lane + 32 * c < nch) st4(out + (lane + 32 * c) * 4, make_float4(vv[c].x * inv, vv[c].y * inv, vv[c].z * inv, vv[c].w * inv));
  }
}

}  // namespace r120
