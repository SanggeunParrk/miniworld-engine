// Smooth lDDT (AlphaFold 3 Algorithm 27) of the diffusion loss: forward value and analytic coordinate gradient (fp32).
//
//   lddt(a, b) = sum_{i != j valid, d_gt(i, j) < cut_i} mean_{t in {.5, 1, 2, 4}} sigmoid(t - |d_pred(i, j) - d_gt(i, j)|) / n_pairs(b)
//   cut_i = 30 A for a nucleotide atom, 15 A otherwise; n_pairs is a property of the ground truth only (the augments a are rigid
//   motions of one structure), so it and the per-tile skip flags are built once per call by slddt_prep.
//
// No [N, N] tensor exists. One thread owns one atom i (a row) and walks over the atoms j of a 128-atom tile staged in shared memory;
// distances come from the coordinates. The four sigmoids share one exponential:
//   sigmoid(t - x) = 1 / (1 + exp(x) * exp(-t)),   x = |d_pred - d_gt| >= 0,   exp(x) = inf is fine (1 / inf = 0).
// Tiles holding no pair inside the radius (flag from slddt_prep) are skipped by the whole CTA.
//
//   slddt_prep   grid (B, NB, NB): per-tile "any pair inside the radius" flags and per-row pair counts (integer atomics), from the ground truth
//   slddt_fwd    grid (ceil(A / GA) * B, NB): per (augment, batch, row block) partial sum of the pair scores
//   slddt_bwd    grid (ceil(A / GA) * B, NB): d(sum of scores)/d(x_pred) for the CTA's 128 rows. Every atom collects the gradient of BOTH roles
//                it plays in a pair (first and second index), so there are no float atomics and the result is deterministic.
//
// A thread owns one atom row i for GA augments (default 1): with GA > 1 everything that depends on the ground truth only (distance, radius
// test, mask) is done once per pair and shared by the augments.
//
// Layout: x [A, B, N, 3] fp32, g [B, N, 3] fp32, nuc / mask [B, N] int8, tile [B, NB, NB] int8; N % 128 == 0 (the host pads).
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int BLK = 128;
#ifndef GA
#define GA 1
#endif
// GA augments per thread (the host launches ceil(A / GA) * B CTAs per row block); set by the host build flag -DGA. Measured on a B200
// (A = 48, N = 4096): 1 -> 3.7 ms, 2 -> 5.3, 4 -> 6.5, 8 -> 11.9 for forward + backward: sharing the ground-truth work does not pay for the
// fewer warps (SFU / shared-memory latency stops being hidden), so the default is one augment per thread.
constexpr float C05 = 0.6065306597f;  // exp(-0.5)
constexpr float C1 = 0.3678794412f;   // exp(-1)
constexpr float C2 = 0.1353352832f;   // exp(-2)
constexpr float C4 = 0.0183156389f;   // exp(-4)
constexpr float NUC_CUT = 30.f, OTHER_CUT = 15.f;

__device__ __forceinline__ float rcp1p(float ex, float c) { return __fdividef(1.f, 1.f + ex * c); }

extern "C" __global__ void __launch_bounds__(BLK)
slddt_prep(const float* __restrict__ G, const int8_t* __restrict__ NUC, const int8_t* __restrict__ M, int8_t* __restrict__ TILE,
           int* __restrict__ ROWCNT, int B, int N, int NB) {
    const int b = blockIdx.x, ib = blockIdx.y, jb = blockIdx.z, t = threadIdx.x;
    __shared__ float sg[BLK * 3];
    __shared__ int sm[BLK];
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + t, j = jb * BLK + t;
    sg[t * 3] = gb[j * 3]; sg[t * 3 + 1] = gb[j * 3 + 1]; sg[t * 3 + 2] = gb[j * 3 + 2];
    sm[t] = M[b * N + j];
    __syncthreads();
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    const float cut = NUC[b * N + i] ? NUC_CUT : OTHER_CUT;
    int c = 0;
    if (M[b * N + i]) {
        for (int k = 0; k < BLK; ++k) {
            if (!sm[k] || jb * BLK + k == i) continue;
            const float h0 = gi0 - sg[k * 3], h1 = gi1 - sg[k * 3 + 1], h2 = gi2 - sg[k * 3 + 2];
            c += (h0 * h0 + h1 * h1 + h2 * h2 < cut * cut) ? 1 : 0;
        }
    }
    if (c) atomicAdd(ROWCNT + (size_t)b * N + i, c);
    const int any = __syncthreads_or(c > 0);
    if (t == 0) TILE[((size_t)b * NB + ib) * NB + jb] = (int8_t)(any != 0);
}

extern "C" __global__ void __launch_bounds__(BLK)
slddt_fwd(const float* __restrict__ X, const float* __restrict__ G, const int8_t* __restrict__ NUC, const int8_t* __restrict__ M,
          const int8_t* __restrict__ TILE, float* __restrict__ OUT, int A, int B, int N, int NB) {
    const int b = blockIdx.x % B, a0 = (blockIdx.x / B) * GA, ib = blockIdx.y, t = threadIdx.x;
    const int na = min(GA, A - a0);
    __shared__ float sx[GA][BLK * 3];
    __shared__ float sg[BLK * 3];
    __shared__ int sm[BLK];
    __shared__ float red[GA][BLK / 32];
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + t;
    float xi[GA][3], acc[GA];
#pragma unroll
    for (int a = 0; a < GA; ++a) {
        acc[a] = 0.f;
        if (a < na) {
            const float* xb = X + ((size_t)(a0 + a) * B + b) * N * 3;
            xi[a][0] = xb[i * 3]; xi[a][1] = xb[i * 3 + 1]; xi[a][2] = xb[i * 3 + 2];
        } else {
            xi[a][0] = xi[a][1] = xi[a][2] = 0.f;
        }
    }
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    const bool mi = M[b * N + i] != 0;
    const float cut = NUC[b * N + i] ? NUC_CUT : OTHER_CUT;
    const float cut2 = cut * cut;
    for (int jb = 0; jb < NB; ++jb) {
        if (!TILE[((size_t)b * NB + ib) * NB + jb]) continue;  // uniform over the CTA
        __syncthreads();
        const int j = jb * BLK + t;
#pragma unroll
        for (int a = 0; a < GA; ++a) {
            if (a < na) {
                const float* xb = X + ((size_t)(a0 + a) * B + b) * N * 3;
                sx[a][t * 3] = xb[j * 3]; sx[a][t * 3 + 1] = xb[j * 3 + 1]; sx[a][t * 3 + 2] = xb[j * 3 + 2];
            }
        }
        sg[t * 3] = gb[j * 3]; sg[t * 3 + 1] = gb[j * 3 + 1]; sg[t * 3 + 2] = gb[j * 3 + 2];
        sm[t] = M[b * N + j];
        __syncthreads();
        if (!mi) continue;
        for (int k = 0; k < BLK; ++k) {
            if (!sm[k] || jb * BLK + k == i) continue;
            const float h0 = gi0 - sg[k * 3], h1 = gi1 - sg[k * 3 + 1], h2 = gi2 - sg[k * 3 + 2];
            const float g2 = h0 * h0 + h1 * h1 + h2 * h2;
            if (g2 >= cut2) continue;
            const float g = sqrtf(g2);
#pragma unroll
            for (int a = 0; a < GA; ++a) {
                if (a < na) {
                    const float e0 = xi[a][0] - sx[a][k * 3], e1 = xi[a][1] - sx[a][k * 3 + 1], e2 = xi[a][2] - sx[a][k * 3 + 2];
                    const float ex = __expf(fabsf(sqrtf(e0 * e0 + e1 * e1 + e2 * e2) - g));
                    acc[a] += 0.25f * (rcp1p(ex, C05) + rcp1p(ex, C1) + rcp1p(ex, C2) + rcp1p(ex, C4));
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GA; ++a) {
        float v = acc[a];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((t & 31) == 0) red[a][t >> 5] = v;
    }
    __syncthreads();
    if (t < na) {
        float s = 0.f;
        for (int w = 0; w < BLK / 32; ++w) s += red[t][w];
        OUT[((size_t)(a0 + t) * B + b) * NB + ib] = s;
    }
}

extern "C" __global__ void __launch_bounds__(BLK)
slddt_bwd(const float* __restrict__ X, const float* __restrict__ G, const int8_t* __restrict__ NUC, const int8_t* __restrict__ M,
          const int8_t* __restrict__ TILE, const float* __restrict__ GO, float* __restrict__ GX, int A, int B, int N, int NB) {
    const int b = blockIdx.x % B, a0 = (blockIdx.x / B) * GA, ib = blockIdx.y, t = threadIdx.x;
    const int na = min(GA, A - a0);
    __shared__ float sx[GA][BLK * 3];
    __shared__ float sg[BLK * 3];
    __shared__ int sm[BLK], sn[BLK];
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + t;
    float xi[GA][3], ga[GA][3];
#pragma unroll
    for (int a = 0; a < GA; ++a) {
        ga[a][0] = ga[a][1] = ga[a][2] = 0.f;
        if (a < na) {
            const float* xb = X + ((size_t)(a0 + a) * B + b) * N * 3;
            xi[a][0] = xb[i * 3]; xi[a][1] = xb[i * 3 + 1]; xi[a][2] = xb[i * 3 + 2];
        } else {
            xi[a][0] = xi[a][1] = xi[a][2] = 0.f;
        }
    }
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    const bool mi = M[b * N + i] != 0;
    const float cut_i = NUC[b * N + i] ? NUC_CUT : OTHER_CUT;
    for (int jb = 0; jb < NB; ++jb) {
        if (!(TILE[((size_t)b * NB + ib) * NB + jb] | TILE[((size_t)b * NB + jb) * NB + ib])) continue;
        __syncthreads();
        const int j = jb * BLK + t;
#pragma unroll
        for (int a = 0; a < GA; ++a) {
            if (a < na) {
                const float* xb = X + ((size_t)(a0 + a) * B + b) * N * 3;
                sx[a][t * 3] = xb[j * 3]; sx[a][t * 3 + 1] = xb[j * 3 + 1]; sx[a][t * 3 + 2] = xb[j * 3 + 2];
            }
        }
        sg[t * 3] = gb[j * 3]; sg[t * 3 + 1] = gb[j * 3 + 1]; sg[t * 3 + 2] = gb[j * 3 + 2];
        sm[t] = M[b * N + j];
        sn[t] = NUC[b * N + j];
        __syncthreads();
        if (!mi) continue;
        for (int k = 0; k < BLK; ++k) {
            if (!sm[k] || jb * BLK + k == i) continue;
            const float h0 = gi0 - sg[k * 3], h1 = gi1 - sg[k * 3 + 1], h2 = gi2 - sg[k * 3 + 2];
            const float g2 = h0 * h0 + h1 * h1 + h2 * h2;
            const float cut_j = sn[k] ? NUC_CUT : OTHER_CUT;
            const float w = (g2 < cut_i * cut_i ? 1.f : 0.f) + (g2 < cut_j * cut_j ? 1.f : 0.f);
            if (w == 0.f) continue;
            const float g = sqrtf(g2);
#pragma unroll
            for (int a = 0; a < GA; ++a) {
                if (a < na) {
                    const float e0 = xi[a][0] - sx[a][k * 3], e1 = xi[a][1] - sx[a][k * 3 + 1], e2 = xi[a][2] - sx[a][k * 3 + 2];
                    const float d = sqrtf(e0 * e0 + e1 * e1 + e2 * e2);
                    const float df = d - g;
                    const float ex = __expf(fabsf(df));
                    const float s1 = rcp1p(ex, C05), s2 = rcp1p(ex, C1), s3 = rcp1p(ex, C2), s4 = rcp1p(ex, C4);
                    const float dsc = 0.25f * (s1 * (1.f - s1) + s2 * (1.f - s2) + s3 * (1.f - s3) + s4 * (1.f - s4));
                    const float sgn = df > 0.f ? 1.f : (df < 0.f ? -1.f : 0.f);
                    const float coef = w * (-sgn) * dsc / fmaxf(d, 1e-6f);
                    ga[a][0] += coef * e0; ga[a][1] += coef * e1; ga[a][2] += coef * e2;
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GA; ++a) {
        if (a < na) {
            const float go = GO[(size_t)(a0 + a) * B + b];
            float* gx = GX + ((size_t)(a0 + a) * B + b) * N * 3;
            gx[i * 3] = ga[a][0] * go; gx[i * 3 + 1] = ga[a][1] * go; gx[i * 3 + 2] = ga[a][2] * go;
        }
    }
}
