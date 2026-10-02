// Token-pair initialisation of the input feature embedder, forward and backward (d_pair = 128, fp32).
//
//   z[b,i,j,:] = left[b,i,:] + right[b,j,:] + Wrel[:, c1] + Wrel[:, 66 + c2] + Wrel[:, 132 + c3] + same_entity * Wrel[:, 138]
//                + Wbond[:, bond[b,i,j]]
//
// where (c1, c2, c3, same_entity) are the relative-position classes of the token pair (i, j) -- AlphaFold 3's relative
// position encoding: c1 the clipped residue offset (66 classes, "other chain" the last), c2 the clipped token offset (same
// chain and residue only), c3 the clipped chain-symmetry offset (6 classes). The reference builds the 139-wide one-hot per
// pair and runs a Linear over it; a one-hot times a matrix is a row gather, so this kernel never materialises it.
//
//   forward   one pass writing z (the only large tensor)
//   backward  one pass over dz: dleft (row sums), dright (column sums, global vector atomics), and the class-bin
//             gradients dW[bin][:] -- run-length accumulated in registers, added per CTA in shared memory, then to global once.
//
// tbl [141][128] = [Wrel^T ; Wbond^T] (bins 0..138 relative position, 139/140 bond); ids [5][B*L] int32 = asym, residue,
// token, entity, sym; bond [B*L*L] uint8 (0/1).
//
// Both kernels first turn the ids of a CTA's rows into one packed word per pair (the four bins and the bond), in shared
// memory, with all threads: pk = bin1 | bin2 << 8 | bin3 << 16 | same_entity << 24 | bond << 25 (every bin < 256: the largest,
// bin3 = 4 r_max + 2 s_max + 5 = n_rel - 2, must be <= 255; the host checks it).
// The backward's class-bin sums are run-length accumulated in registers -- a bin changes rarely along j (the clipped offsets
// are constant away from the diagonal; chain / entity / bond bins are near-constant) -- and a run is added to the CTA's
// shared-memory gradient only when its bin changes (ids that jump around -- a shuffled residue index -- make every position a
// run: correct, but slow).
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int P = 128;
constexpr int NWARP = 8;
#ifndef TPI_RI
#define TPI_RI 2
#endif
constexpr int RI = TPI_RI;                  // rows of z per CTA in the backward

__device__ __forceinline__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ void operator+=(float4& a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

struct Row {
    int asym, res, tok, ent, sym;
};

__device__ __forceinline__ Row load_row(const int* ids, int BL, int idx) {
    return Row{ids[idx], ids[BL + idx], ids[2 * BL + idx], ids[3 * BL + idx], ids[4 * BL + idx]};
}

// bins of the pair (i, j): c1, 2R+2 + c2, 2(2R+2) + c3, and the same-entity flag (its own bin, NREL - 1)
__device__ __forceinline__ void pair_bins(const Row& a, const Row& b, int rmax, int smax, int& bin1, int& bin2, int& bin3, int& se) {
    const bool same_chain = a.asym == b.asym, same_res = a.res == b.res, same_ent = a.ent == b.ent;
    const int nr = 2 * rmax + 2, ns = 2 * smax + 2;
    int dres = min(max(a.res - b.res + rmax, 0), 2 * rmax);
    int dtok = min(max(a.tok - b.tok + rmax, 0), 2 * rmax);
    int dch = min(max(a.sym - b.sym + smax, 0), 2 * smax);
    bin1 = same_chain ? dres : nr - 1;
    bin2 = nr + ((same_chain && same_res) ? dtok : nr - 1);
    bin3 = 2 * nr + (same_ent ? dch : ns - 1);
    se = same_ent;
}


__device__ __forceinline__ uint32_t pack_pair(const Row& a, const Row& b, uint8_t bond, int rmax, int smax) {
    int b1, b2, b3, se;
    pair_bins(a, b, rmax, smax, b1, b2, b3, se);
    return (uint32_t)b1 | ((uint32_t)b2 << 8) | ((uint32_t)b3 << 16) | ((uint32_t)se << 24) | ((uint32_t)bond << 25);
}

// pk[r][j] for the rows i0 .. i0 + nrows - 1 of batch element b (smem), all threads
__device__ __forceinline__ void pack_rows(uint32_t* pk, const int* ids, const uint8_t* bond, int BL, int L, int b, int i0, int nrows,
                                          int rmax, int smax) {
    for (int idx = threadIdx.x; idx < nrows * L; idx += blockDim.x) {
        const int r = idx / L, j = idx % L, ri = b * L + i0 + r;
        pk[idx] = pack_pair(load_row(ids, BL, ri), load_row(ids, BL, b * L + j), bond[(size_t)ri * L + j], rmax, smax);
    }
}

#ifndef TPI_UNR
#define TPI_UNR 4
#endif
constexpr int UNR = TPI_UNR;                      // positions in flight per warp in the forward

extern "C" __global__ void __launch_bounds__(256)
tpi_fwd(const float* __restrict__ left, const float* __restrict__ right, const float* __restrict__ tbl, const int* __restrict__ ids,
        const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax, int rows_per_cta) {
    extern __shared__ __align__(16) float sm[];
    const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1, nbin = nrel + 2;
    uint32_t* pk = (uint32_t*)(sm + nbin * P);       // [L]
    for (int idx = threadIdx.x; idx < nbin * (P / 4); idx += blockDim.x) ((float4*)sm)[idx] = ((const float4*)tbl)[idx];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
    const int tiles = (L + rows_per_cta - 1) / rows_per_cta;
    const int b = blockIdx.x / tiles, i0 = (blockIdx.x % tiles) * rows_per_cta;
    const float4* T = (const float4*)sm;
    for (int i = i0; i < min(i0 + rows_per_cta, L); i++) {
        __syncthreads();                             // tables loaded (first row) / previous row's pk consumed
        pack_rows(pk, ids, bond, BL, L, b, i, 1, rmax, smax);
        __syncthreads();
        const int ri = b * L + i;
        const float4 lf = ((const float4*)(left + (size_t)ri * P))[lane];
        float4* orow = (float4*)(out + (size_t)ri * L * P);
        for (int j0 = warp * UNR; j0 < L; j0 += NWARP * UNR) {
            float4 o[UNR];
#pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int j = min(j0 + u, L - 1);
                o[u] = lf + ((const float4*)(right + (size_t)(b * L + j) * P))[lane];
            }
#pragma unroll
            for (int u = 0; u < UNR; u++) {
                const uint32_t w = pk[min(j0 + u, L - 1)];
                o[u] += T[(w & 255) * 32 + lane];
                o[u] += T[((w >> 8) & 255) * 32 + lane];
                o[u] += T[((w >> 16) & 255) * 32 + lane];
                if ((w >> 24) & 1) o[u] += T[(nrel - 1) * 32 + lane];
                o[u] += T[(nrel + ((w >> 25) & 1)) * 32 + lane];
            }
#pragma unroll
            for (int u = 0; u < UNR; u++)
                if (j0 + u < L) __stcs(orow + (size_t)(j0 + u) * 32 + lane, o[u]);
        }
    }
}

struct Run {
    int bin;
    float4 acc;
};

// a finished run is added to the CTA's class-bin gradient in shared memory (all warps meet on a few hot bins; one global
// atomic per warp and run would serialise on them)
__device__ __forceinline__ void flush_run(float* accs, const Run& r, int lane) {
    float* d = accs + r.bin * P + 4 * lane;
    atomicAdd(d + 0, r.acc.x); atomicAdd(d + 1, r.acc.y); atomicAdd(d + 2, r.acc.z); atomicAdd(d + 3, r.acc.w);
}

// add v to run `r` of bin `bin` (< 0: nothing to add); the run is flushed when its bin changes
__device__ __forceinline__ void push_run(float* accs, Run& r, int bin, float4 v, int lane) {
    if (bin < 0) return;
    if (bin == r.bin) {
        r.acc += v;
    } else {
        if (r.bin >= 0) flush_run(accs, r, lane);
        r.bin = bin;
        r.acc = v;
    }
}

extern "C" __global__ void __launch_bounds__(256)
tpi_bwd(const float* __restrict__ g, const int* __restrict__ ids, const uint8_t* __restrict__ bond, float* __restrict__ dleft,
        float* __restrict__ dright, float* __restrict__ dtbl, int B, int L, int rmax, int smax) {
    extern __shared__ __align__(16) float sm[];
    const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1, nbin = nrel + 2;
    float* acc = sm;                                 // [nbin][P] class-bin gradient of this CTA
    float4* red = (float4*)(sm + nbin * P);          // [NWARP][RI][32] per-warp row sums
    uint32_t* pk = (uint32_t*)(red + NWARP * RI * 32);   // [RI][L]
    for (int idx = threadIdx.x; idx < nbin * P; idx += blockDim.x) acc[idx] = 0.f;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
    const int tiles = (L + RI - 1) / RI;
    const int b = blockIdx.x / tiles, i0 = (blockIdx.x % tiles) * RI;
    const int nrows = min(RI, L - i0);
    pack_rows(pk, ids, bond, BL, L, b, i0, nrows, rmax, smax);
    __syncthreads();
    float4 sum_i[RI];
    Run run[RI][5];                                  // per row: the bins change slowly along j, not across the rows
#pragma unroll
    for (int r = 0; r < RI; r++) {
        sum_i[r] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < 5; k++) run[r][k] = Run{-1, make_float4(0.f, 0.f, 0.f, 0.f)};
    }
    // dz is read through a two-deep register pipeline: the loads of position j + 2 NWARP are in flight while j is processed
    constexpr int DEPTH = 2;
    float4 gq[DEPTH][RI];
#pragma unroll
    for (int d = 0; d < DEPTH; d++) {
        const int j = warp + d * NWARP;
#pragma unroll
        for (int r = 0; r < RI; r++)
            if (j < L && r < nrows) gq[d][r] = __ldg((const float4*)(g + ((size_t)(b * L + i0 + r) * L + j) * P) + lane);
    }
    for (int j = warp, it = 0; j < L; j += NWARP, it++) {  // interleaved: the diagonal band (changing bins) spreads over the warps
        float4 cur[RI];
#pragma unroll
        for (int r = 0; r < RI; r++) cur[r] = gq[0][r];
#pragma unroll
        for (int d = 0; d + 1 < DEPTH; d++) {
#pragma unroll
            for (int r = 0; r < RI; r++) gq[d][r] = gq[d + 1][r];
        }
        {
            const int jn = j + DEPTH * NWARP;
#pragma unroll
            for (int r = 0; r < RI; r++)
                if (jn < L && r < nrows) gq[DEPTH - 1][r] = __ldg((const float4*)(g + ((size_t)(b * L + i0 + r) * L + jn) * P) + lane);
        }
        float4 col = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < RI; r++) {
            if (r >= nrows) continue;
            const float4 gv = cur[r];
            sum_i[r] += gv;
            col += gv;
            const uint32_t w = pk[r * L + j];
            push_run(acc, run[r][0], (int)(w & 255), gv, lane);
            push_run(acc, run[r][1], (int)((w >> 8) & 255), gv, lane);
            push_run(acc, run[r][2], (int)((w >> 16) & 255), gv, lane);
            push_run(acc, run[r][3], ((w >> 24) & 1) ? nrel - 1 : -1, gv, lane);
            push_run(acc, run[r][4], nrel + (int)((w >> 25) & 1), gv, lane);
        }
        atomicAdd((float4*)(dright + (size_t)(b * L + j) * P) + lane, col);
    }
#pragma unroll
    for (int r = 0; r < RI; r++) {
#pragma unroll
        for (int k = 0; k < 5; k++)
            if (run[r][k].bin >= 0) flush_run(acc, run[r][k], lane);
        red[(warp * RI + r) * 32 + lane] = sum_i[r];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < RI * 32; idx += blockDim.x) {
        const int r = idx / 32, l = idx % 32;
        if (r >= nrows) continue;
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int w = 0; w < NWARP; w++) s += red[(w * RI + r) * 32 + l];
        ((float4*)(dleft + (size_t)(b * L + i0 + r) * P))[l] = s;
    }
    for (int idx = threadIdx.x; idx < nbin * 32; idx += blockDim.x) {
        const float4 v = ((const float4*)acc)[idx];
        if (v.x != 0.f || v.y != 0.f || v.z != 0.f || v.w != 0.f) atomicAdd((float4*)dtbl + idx, v);
    }
}
