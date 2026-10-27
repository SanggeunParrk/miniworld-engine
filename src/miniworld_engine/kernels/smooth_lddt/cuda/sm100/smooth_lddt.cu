// Smooth lDDT (AlphaFold 3 Algorithm 27) of the diffusion loss: forward value and analytic coordinate gradient (fp32).
//
//   lddt(a, b) = sum_{i != j valid, d_gt(i, j) < cut_i} mean_{t in {.5, 1, 2, 4}} sigmoid(t - |d_pred(i, j) - d_gt(i, j)|) / n_pairs(b)
//   cut_i = 30 A for a nucleotide atom, 15 A otherwise; n_pairs is a property of the ground truth only (the augments a are rigid
//   motions of one structure), so it, the pair sets and the per-tile skip flags are built once per call by slddt_prep.
//
// No [N, N] float tensor exists. The pair set of the ground truth is kept as bit masks (N x N bits):
//   FM[b, i, j]  = both atoms valid, i != j, d_gt(i, j) < cut_i               (the pairs of the loss, directed)
//   UM[b, i, j]  = FM[b, i, j] | FM[b, j, i]                                  (symmetric: every pair that carries gradient to i or to j)
// A pair outside the radius costs nothing in forward / backward: one thread owns one atom row i, walks over the 128-atom tiles holding a
// set bit, and visits only the set bits of its 4 words per tile (the coordinates of the tile are staged in shared memory). Distances are
// computed from the coordinates; the four sigmoids share one exponential:
//   sigmoid(t - x) = 1 / (1 + exp(x) * exp(-t)),   x = |d_pred - d_gt| >= 0,   exp(x) = inf is fine (1 / inf = 0).
//
//   slddt_prep   grid (B, NB, NB): FM / UM words of the tile, its flags (bit 0: FM has a bit, bit 1: UM has a bit), per-row pair counts
//   slddt_fwd    grid (A * B, NB): per (augment, batch, row block) partial sum of the pair scores
//   slddt_bwd    grid (A * B, NB): d(sum of scores)/d(x_pred) for the CTA's 128 rows. Every atom collects the gradient of BOTH roles it
//                plays in a pair (first and second index, weight [g < cut_i] + [g < cut_j]), so there are no float atomics and the result
//                is deterministic.
//
// Layout: x [A, B, N, 3] fp32, g [B, N, 3] fp32, nuc [B, N] int8, tile [B, NB, NB] int8, FM / UM [B, N, N / 32] uint32; N % 128 == 0.
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int BLK = 128;
constexpr float C05 = 0.6065306597f;  // exp(-0.5)
constexpr float C1 = 0.3678794412f;   // exp(-1)
constexpr float C2 = 0.1353352832f;   // exp(-2)
constexpr float C4 = 0.0183156389f;   // exp(-4)
constexpr float NUC_CUT = 30.f, OTHER_CUT = 15.f;

__device__ __forceinline__ float rcp1p(float ex, float c) { return __fdividef(1.f, 1.f + ex * c); }

// With w = exp(-x) in (0, 1] and c_t = exp(-t), sigmoid(t - x) = w / (w + c_t), so the four-threshold score and its derivative are rational
// functions of w with one denominator Q(w) = prod_t (w + c_t) (all coefficients positive: no cancellation, no overflow for large x):
//   score   = 0.25 * w * Q'(w) / Q(w)
//   -dscore/dx = 0.25 * sum_t s_t (1 - s_t) = 0.25 * w * N(w) / Q(w)^2,   N(w) = sum_t c_t prod_{s != t} (w + c_s)^2
// One exponential and one reciprocal per pair instead of one exponential and four reciprocals (the MUFU pipe was the busiest).
__device__ __forceinline__ float rat_q(float w) {
    float q = w + 1.12806102f;
    q = fmaf(q, w, 0.375327923f);
    q = fmaf(q, w, 0.036699476f);
    return fmaf(q, w, 0.00055308437f);
}
__device__ __forceinline__ float rat_dq(float w) {
    float q = fmaf(w, 4.f, 3.38418307f);
    q = fmaf(q, w, 0.750655846f);
    return fmaf(q, w, 0.036699476f);
}
__device__ __forceinline__ float rat_n(float w) {
    float n = fmaf(1.12806102f, w, 1.50131169f);
    n = fmaf(n, w, 0.753688085f);
    n = fmaf(n, w, 0.174446344f);
    n = fmaf(n, w, 0.0193895544f);
    n = fmaf(n, w, 0.000830352031f);
    return fmaf(n, w, 2.02979066e-05f);
}
// sum over the four thresholds of sigmoid(t - x) / 4 for w = exp(-x)
__device__ __forceinline__ float pair_score(float w) { return 0.25f * w * rat_dq(w) * __fdividef(1.f, rat_q(w)); }

extern "C" __global__ void __launch_bounds__(BLK)
slddt_prep(const float* __restrict__ G, const int8_t* __restrict__ NUC, const int8_t* __restrict__ M, int8_t* __restrict__ TILE,
           int* __restrict__ ROWCNT, uint32_t* __restrict__ FM, uint32_t* __restrict__ UM, int B, int N, int NB) {
    const int b = blockIdx.x, ib = blockIdx.y, jb = blockIdx.z, t = threadIdx.x;
    __shared__ float sg[BLK * 3];
    __shared__ int sm[BLK];
    __shared__ float scut2[BLK];
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + t, j = jb * BLK + t;
    sg[t * 3] = gb[j * 3]; sg[t * 3 + 1] = gb[j * 3 + 1]; sg[t * 3 + 2] = gb[j * 3 + 2];
    sm[t] = M[b * N + j];
    const float cj = NUC[b * N + j] ? NUC_CUT : OTHER_CUT;
    scut2[t] = cj * cj;
    __syncthreads();
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    const float ci = NUC[b * N + i] ? NUC_CUT : OTHER_CUT;
    const float ci2 = ci * ci;
    const bool mi = M[b * N + i] != 0;
    const int W = N / 32;
    const size_t row = ((size_t)b * N + i) * W + (size_t)jb * 4;
    int c = 0;
    bool anyf = false, anyu = false;
#pragma unroll 1
    for (int w = 0; w < 4; ++w) {
        uint32_t f = 0u, u = 0u;
        if (mi) {
            for (int kk = 0; kk < 32; ++kk) {
                const int k = w * 32 + kk;
                if (!sm[k] || jb * BLK + k == i) continue;
                const float h0 = gi0 - sg[k * 3], h1 = gi1 - sg[k * 3 + 1], h2 = gi2 - sg[k * 3 + 2];
                const float g2 = h0 * h0 + h1 * h1 + h2 * h2;
                const bool f1 = g2 < ci2, f2 = g2 < scut2[k];
                f |= (uint32_t)f1 << kk;
                u |= (uint32_t)(f1 | f2) << kk;
            }
        }
        FM[row + w] = f;
        UM[row + w] = u;
        c += __popc(f);
        anyf |= f != 0u;
        anyu |= u != 0u;
    }
    if (c) atomicAdd(ROWCNT + (size_t)b * N + i, c);
    const int af = __syncthreads_or(anyf), au = __syncthreads_or(anyu);
    if (t == 0) TILE[((size_t)b * NB + ib) * NB + jb] = (int8_t)((af != 0) | ((au != 0) << 1));
}

extern "C" __global__ void __launch_bounds__(BLK)
slddt_fwd(const float* __restrict__ X, const float* __restrict__ G, const int8_t* __restrict__ TILE, const uint32_t* __restrict__ FM,
          float* __restrict__ OUT, int B, int N, int NB) {
    const int ab = blockIdx.x, ib = blockIdx.y, b = ab % B, t = threadIdx.x;
    __shared__ float sx0[BLK], sx1[BLK], sx2[BLK], sg0[BLK], sg1[BLK], sg2[BLK];
    __shared__ float red[BLK / 32];
    const float* xb = X + (size_t)ab * N * 3;
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + t, W = N / 32;
    const float xi0 = xb[i * 3], xi1 = xb[i * 3 + 1], xi2 = xb[i * 3 + 2];
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    float acc = 0.f;
    for (int jb = 0; jb < NB; ++jb) {
        if (!(TILE[((size_t)b * NB + ib) * NB + jb] & 1)) continue;  // uniform over the CTA
        __syncthreads();
        const int j = jb * BLK + t;
        sx0[t] = xb[j * 3]; sx1[t] = xb[j * 3 + 1]; sx2[t] = xb[j * 3 + 2];
        sg0[t] = gb[j * 3]; sg1[t] = gb[j * 3 + 1]; sg2[t] = gb[j * 3 + 2];
        __syncthreads();
        const uint4 fw = *reinterpret_cast<const uint4*>(FM + ((size_t)b * N + i) * W + (size_t)jb * 4);
        const uint32_t words[4] = {fw.x, fw.y, fw.z, fw.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t f = words[w];
            while (f) {
                const uint32_t low = f & (0u - f);
                const int k = w * 32 + __popc(low - 1u);
                f ^= low;
                const float h0 = gi0 - sg0[k], h1 = gi1 - sg1[k], h2 = gi2 - sg2[k];
                const float e0 = xi0 - sx0[k], e1 = xi1 - sx1[k], e2 = xi2 - sx2[k];
                const float diff = fabsf(sqrtf(e0 * e0 + e1 * e1 + e2 * e2) - sqrtf(h0 * h0 + h1 * h1 + h2 * h2));
                acc += pair_score(__expf(-diff));
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((t & 31) == 0) red[t >> 5] = acc;
    __syncthreads();
    if (t == 0) {
        float s = 0.f;
        for (int w = 0; w < BLK / 32; ++w) s += red[w];
        OUT[(size_t)ab * NB + ib] = s;
    }
}

extern "C" __global__ void __launch_bounds__(BLK)
slddt_bwd(const float* __restrict__ X, const float* __restrict__ G, const int8_t* __restrict__ NUC, const int8_t* __restrict__ TILE,
          const uint32_t* __restrict__ UM, const float* __restrict__ GO, float* __restrict__ GX, int B, int N, int NB) {
    const int ab = blockIdx.x, ib = blockIdx.y, b = ab % B, t = threadIdx.x;
    __shared__ float sx0[BLK], sx1[BLK], sx2[BLK], sg0[BLK], sg1[BLK], sg2[BLK], scut2[BLK];
    const float* xb = X + (size_t)ab * N * 3;
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + t, W = N / 32;
    const float xi0 = xb[i * 3], xi1 = xb[i * 3 + 1], xi2 = xb[i * 3 + 2];
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    const float ci = NUC[b * N + i] ? NUC_CUT : OTHER_CUT;
    const float ci2 = ci * ci;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    for (int jb = 0; jb < NB; ++jb) {
        if (!(TILE[((size_t)b * NB + ib) * NB + jb] & 2)) continue;
        __syncthreads();
        const int j = jb * BLK + t;
        sx0[t] = xb[j * 3]; sx1[t] = xb[j * 3 + 1]; sx2[t] = xb[j * 3 + 2];
        sg0[t] = gb[j * 3]; sg1[t] = gb[j * 3 + 1]; sg2[t] = gb[j * 3 + 2];
        const float cj = NUC[b * N + j] ? NUC_CUT : OTHER_CUT;
        scut2[t] = cj * cj;
        __syncthreads();
        const uint4 uw = *reinterpret_cast<const uint4*>(UM + ((size_t)b * N + i) * W + (size_t)jb * 4);
        const uint32_t words[4] = {uw.x, uw.y, uw.z, uw.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t u = words[w];
            while (u) {
                const uint32_t low = u & (0u - u);
                const int k = w * 32 + __popc(low - 1u);
                u ^= low;
                const float h0 = gi0 - sg0[k], h1 = gi1 - sg1[k], h2 = gi2 - sg2[k];
                const float g2 = h0 * h0 + h1 * h1 + h2 * h2;
                const float wgt = (g2 < ci2 ? 1.f : 0.f) + (g2 < scut2[k] ? 1.f : 0.f);
                const float e0 = xi0 - sx0[k], e1 = xi1 - sx1[k], e2 = xi2 - sx2[k];
                const float d = sqrtf(e0 * e0 + e1 * e1 + e2 * e2);
                const float df = d - sqrtf(g2);
                const float ew = __expf(-fabsf(df));
                const float qv = rat_q(ew);
                const float sgn = df > 0.f ? 1.f : (df < 0.f ? -1.f : 0.f);
                const float coef = wgt * (-sgn) * (0.25f * ew * rat_n(ew)) * __fdividef(1.f, qv * qv * fmaxf(d, 1e-6f));
                a0 += coef * e0; a1 += coef * e1; a2 += coef * e2;
            }
        }
    }
    const float go = GO[ab];
    float* gx = GX + (size_t)ab * N * 3;
    gx[i * 3] = a0 * go; gx[i * 3 + 1] = a1 * go; gx[i * 3 + 2] = a2 * go;
}

// ---- whole-structure variants -------------------------------------------------------------------------------------------------------
// When the coordinates of one augment fit in shared memory (N * 32 B), a CTA of 512 threads stages them once and every thread walks its
// whole neighbour list across ALL tiles instead of tile by tile: the lanes of a warp are no longer synchronised on the tile boundaries, so
// the imbalance of the neighbour counts costs once per row (max over the warp of the row total) rather than once per tile. One augment's
// x and g are float4 (x, y, z, cut^2 of the atom) so a neighbour is two 128-bit shared loads. Same masks, same partial-sum layout
// [A * B, N / 128] as the tile kernels (a block of 128 rows = 4 warps).
constexpr int FBLK = 512;   // threads of the default whole-structure CTA; slddt_*_full_{128,256} are the same kernels with smaller CTAs

template <int FB, int S>
__device__ __forceinline__ void fwd_full_impl(const float* __restrict__ X, const float* __restrict__ G, const uint32_t* __restrict__ FM, float* __restrict__ OUT,
               int B, int N, int NB) {
    extern __shared__ float4 sm4[];
    float4* sx = sm4;
    float4* sg = sm4 + N;
    __shared__ float red[FB / 32];
    const int ab = blockIdx.x, ib = blockIdx.y, b = ab % B, t = threadIdx.x;
    const float* xb = X + (size_t)ab * N * 3;
    const float* gb = G + (size_t)b * N * 3;
    for (int p = t; p < N; p += FB) {
        sx[p] = make_float4(xb[p * 3], xb[p * 3 + 1], xb[p * 3 + 2], 0.f);
        sg[p] = make_float4(gb[p * 3], gb[p * 3 + 1], gb[p * 3 + 2], 0.f);
    }
    __syncthreads();
    constexpr int ROWS = FB / S;               // rows of the CTA: S threads share a row, thread s of S walks the 4-word groups q with q % S == s
    const int i = ib * ROWS + t / S, sp = t % S, W = N / 32;
    float acc = 0.f;
    if (i < N) {
        const float4 xi = sx[i], gi = sg[i];
        const uint4* row = reinterpret_cast<const uint4*>(FM + ((size_t)b * N + i) * W);
        for (int q = sp; q < W / 4; q += S) {
            const uint4 fw = row[q];
            if (!(fw.x | fw.y | fw.z | fw.w)) continue;
            const uint32_t words[4] = {fw.x, fw.y, fw.z, fw.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t f = words[w];
                while (f) {
                    const uint32_t low = f & (0u - f);
                    const int k = q * 128 + w * 32 + __popc(low - 1u);
                    f ^= low;
                    const float4 xj = sx[k], gj = sg[k];
                    const float h0 = gi.x - gj.x, h1 = gi.y - gj.y, h2 = gi.z - gj.z;
                    const float e0 = xi.x - xj.x, e1 = xi.y - xj.y, e2 = xi.z - xj.z;
                    const float diff = fabsf(sqrtf(e0 * e0 + e1 * e1 + e2 * e2) - sqrtf(h0 * h0 + h1 * h1 + h2 * h2));
                    acc += pair_score(__expf(-diff));
                }
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((t & 31) == 0) red[t >> 5] = acc;
    __syncthreads();
    if (t < ROWS / BLK) {                      // a partial per 128 rows = 4 * S warps
        const int blk = ib * (ROWS / BLK) + t;
        float s = 0.f;
        for (int w = 0; w < 4 * S; ++w) s += red[t * 4 * S + w];
        if (blk < NB) OUT[(size_t)ab * NB + blk] = s;
    }
}

template <int FB>
__device__ __forceinline__ void bwd_full_impl(const float* __restrict__ X, const float* __restrict__ G, const int8_t* __restrict__ NUC, const uint32_t* __restrict__ UM,
               const float* __restrict__ GO, float* __restrict__ GX, int B, int N, int NB) {
    extern __shared__ float4 sm4[];
    float4* sx = sm4;      // x, y, z, cut^2 of the atom
    float4* sg = sm4 + N;
    const int ab = blockIdx.x, ib = blockIdx.y, b = ab % B, t = threadIdx.x;
    const float* xb = X + (size_t)ab * N * 3;
    const float* gb = G + (size_t)b * N * 3;
    for (int p = t; p < N; p += FB) {
        const float c = NUC[(size_t)b * N + p] ? NUC_CUT : OTHER_CUT;
        sx[p] = make_float4(xb[p * 3], xb[p * 3 + 1], xb[p * 3 + 2], c * c);
        sg[p] = make_float4(gb[p * 3], gb[p * 3 + 1], gb[p * 3 + 2], 0.f);
    }
    __syncthreads();
    const int i = ib * FB + t, W = N / 32;
    if (i >= N) return;
    const float4 xi = sx[i], gi = sg[i];
    const float ci2 = xi.w;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    const uint4* row = reinterpret_cast<const uint4*>(UM + ((size_t)b * N + i) * W);
    for (int q = 0; q < W / 4; ++q) {
        const uint4 uw = row[q];
        if (!(uw.x | uw.y | uw.z | uw.w)) continue;
        const uint32_t words[4] = {uw.x, uw.y, uw.z, uw.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t u = words[w];
            while (u) {
                const uint32_t low = u & (0u - u);
                const int k = q * 128 + w * 32 + __popc(low - 1u);
                u ^= low;
                const float4 xj = sx[k], gj = sg[k];
                const float h0 = gi.x - gj.x, h1 = gi.y - gj.y, h2 = gi.z - gj.z;
                const float g2 = h0 * h0 + h1 * h1 + h2 * h2;
                const float wgt = (g2 < ci2 ? 1.f : 0.f) + (g2 < xj.w ? 1.f : 0.f);
                const float e0 = xi.x - xj.x, e1 = xi.y - xj.y, e2 = xi.z - xj.z;
                const float d = sqrtf(e0 * e0 + e1 * e1 + e2 * e2);
                const float df = d - sqrtf(g2);
                const float ew = __expf(-fabsf(df));
                const float qv = rat_q(ew);
                const float sgn = df > 0.f ? 1.f : (df < 0.f ? -1.f : 0.f);
                const float coef = wgt * (-sgn) * (0.25f * ew * rat_n(ew)) * __fdividef(1.f, qv * qv * fmaxf(d, 1e-6f));
                a0 += coef * e0; a1 += coef * e1; a2 += coef * e2;
            }
        }
    }
    const float go = GO[ab];
    float* gx = GX + (size_t)ab * N * 3;
    gx[i * 3] = a0 * go; gx[i * 3 + 1] = a1 * go; gx[i * 3 + 2] = a2 * go;
}

// The whole-structure kernels at other CTA sizes. Forward: FB threads, S of them per row (the row's 4-word groups are dealt out round-robin), so that a launch
// with few rows (A48 x N1024 = 49k rows = 11 warps per SM) still keeps the SM's schedulers fed; partials stay one per 128 rows. Backward: FB threads, one per row.
// A launch of A * B augments of N atoms has A * B * N * S / FB CTAs; when that is fewer than the card has SMs the smaller / split CTAs fill it.
#define SLDDT_FWD_FULL(FB, S)                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(FB) slddt_fwd_full_##FB##s##S(const float* __restrict__ X, const float* __restrict__ G,                \
                                                                               const uint32_t* __restrict__ FM, float* __restrict__ OUT, int B,        \
                                                                               int N, int NB) {                                                         \
        fwd_full_impl<FB, S>(X, G, FM, OUT, B, N, NB);                                                                                                 \
    }
#define SLDDT_BWD_FULL(FB)                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(FB) slddt_bwd_full_##FB(                                                                              \
        const float* __restrict__ X, const float* __restrict__ G, const int8_t* __restrict__ NUC, const uint32_t* __restrict__ UM,                     \
        const float* __restrict__ GO, float* __restrict__ GX, int B, int N, int NB) {                                                                  \
        bwd_full_impl<FB>(X, G, NUC, UM, GO, GX, B, N, NB);                                                                                            \
    }
SLDDT_FWD_FULL(128, 1)
SLDDT_FWD_FULL(256, 1)
SLDDT_FWD_FULL(512, 1)
SLDDT_FWD_FULL(256, 2)
SLDDT_FWD_FULL(512, 2)
SLDDT_FWD_FULL(512, 4)
SLDDT_BWD_FULL(128)
SLDDT_BWD_FULL(256)
SLDDT_BWD_FULL(512)
