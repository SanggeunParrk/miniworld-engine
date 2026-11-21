// smooth_lddt_sm100.cu -- the B200 (sm_100a) cubin of the smooth lDDT kernels (2026-11-17): the shared source smooth_lddt.cu (compiled
// unchanged for H100 / A100 / RTX 5090) plus one B200-only kernel, slddt_prep4. Only the sm_100a build compiles this file (sm100/__init__.py
// cubin), so the other cards' cubins and their keys are untouched.
#include "smooth_lddt.cu"

// slddt_prep with four threads per row (512 threads, thread t: row t % 128, mask word t / 128 of the tile), the same
// words, counts (integer atomics) and tile flags. At small N the grid (B, NB, NB) is a few dozen CTAs and the one-thread-per-row loop over 128
// columns is latency bound (N 1024: 64 CTAs, 10 us); four threads a row cut the serial loop to 32 columns.
extern "C" __global__ void __launch_bounds__(4 * BLK)
slddt_prep4(const float* __restrict__ G, const int8_t* __restrict__ NUC, const int8_t* __restrict__ M, int8_t* __restrict__ TILE,
            int* __restrict__ ROWCNT, uint32_t* __restrict__ FM, uint32_t* __restrict__ UM, int B, int N, int NB) {
    const int b = blockIdx.x, ib = blockIdx.y, jb = blockIdx.z, t = threadIdx.x, r = t % BLK, w = t / BLK;
    __shared__ float sg[BLK * 3];
    __shared__ int sm[BLK];
    __shared__ float scut2[BLK];
    const float* gb = G + (size_t)b * N * 3;
    const int i = ib * BLK + r;
    if (t < BLK) {
        const int j = jb * BLK + t;
        sg[t * 3] = gb[j * 3]; sg[t * 3 + 1] = gb[j * 3 + 1]; sg[t * 3 + 2] = gb[j * 3 + 2];
        sm[t] = M[b * N + j];
        const float cj = NUC[b * N + j] ? NUC_CUT : OTHER_CUT;
        scut2[t] = cj * cj;
    }
    __syncthreads();
    const float gi0 = gb[i * 3], gi1 = gb[i * 3 + 1], gi2 = gb[i * 3 + 2];
    const float ci = NUC[b * N + i] ? NUC_CUT : OTHER_CUT;
    const float ci2 = ci * ci;
    const bool mi = M[b * N + i] != 0;
    const int W = N / 32;
    const size_t row = ((size_t)b * N + i) * W + (size_t)jb * 4;
    uint32_t f = 0u, u = 0u;
    if (mi) {
#pragma unroll 4
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
    const int c = __popc(f);
    if (c) atomicAdd(ROWCNT + (size_t)b * N + i, c);
    const int af = __syncthreads_or(f != 0u), au = __syncthreads_or(u != 0u);
    if (t == 0) TILE[((size_t)b * NB + ib) * NB + jb] = (int8_t)((af != 0) | ((au != 0) << 1));
}
