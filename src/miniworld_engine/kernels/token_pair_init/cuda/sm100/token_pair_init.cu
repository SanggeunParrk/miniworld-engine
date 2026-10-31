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
//   forward   one pass writing z (the only large tensor): per row the table part is recomputed only where the bins change along j
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

#ifndef TPI_RF
#define TPI_RF 2
#endif
constexpr int RF = TPI_RF;                       // rows of z per tile in the forward

// Forward: a CTA takes tiles of RF rows i of one batch element (grid-stride over the B * ceil(L / RF) tiles). Each warp walks a
// contiguous run of positions j and serves the tile's RF rows from one register copy of right[j]; per row it keeps t = left[i] +
// the table rows of the pair's bins, recomputed only when the packed word changes along j (the clipped offsets are constant away
// from the diagonal, the chain / entity / bond bins nearly so: a warp-uniform branch). A position of a row costs one add and one
// streaming store; the table is read through L1 (__ldg) on a change only, and only the packed words live in shared memory.
extern "C" __global__ void __launch_bounds__(256)
tpi_fwd(const float* __restrict__ left, const float* __restrict__ right, const float* __restrict__ tbl, const int* __restrict__ ids,
        const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax) {
    extern __shared__ __align__(16) uint32_t pk[];   // [RF][L]
    const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
    const int tiles = (L + RF - 1) / RF;
    const float4* T = (const float4*)tbl;
    const int span = (L + NWARP - 1) / NWARP;
    const int jb = min(L, warp * span), je = min(L, jb + span);
    for (int tile = blockIdx.x; tile < B * tiles; tile += gridDim.x) {
        const int b = tile / tiles, i0 = (tile % tiles) * RF, nrows = min(RF, L - i0);
        __syncthreads();                             // the previous tile's words consumed
        pack_rows(pk, ids, bond, BL, L, b, i0, nrows, rmax, smax);
        __syncthreads();
        float4 lf[RF], t[RF];
        uint32_t last[RF];
#pragma unroll
        for (int r = 0; r < RF; r++) {
            lf[r] = r < nrows ? __ldg((const float4*)(left + (size_t)(b * L + i0 + r) * P) + lane) : make_float4(0.f, 0.f, 0.f, 0.f);
            t[r] = lf[r];
            last[r] = 0xffffffffu;                   // no packed word is all ones (bit 26 up is zero)
        }
        for (int j0 = jb; j0 < je; j0 += UNR) {
            float4 rv[UNR];
#pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int j = min(j0 + u, je - 1);
                rv[u] = __ldg((const float4*)(right + (size_t)(b * L + j) * P) + lane);
            }
#pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int j = j0 + u;
                if (j >= je) break;
#pragma unroll
                for (int r = 0; r < RF; r++) {
                    if (r >= nrows) break;
                    const uint32_t w = pk[r * L + j];
                    if (w != last[r]) {
                        last[r] = w;
                        float4 v = lf[r] + __ldg(T + (w & 255) * 32 + lane);
                        v += __ldg(T + ((w >> 8) & 255) * 32 + lane);
                        v += __ldg(T + ((w >> 16) & 255) * 32 + lane);
                        if ((w >> 24) & 1) v += __ldg(T + (nrel - 1) * 32 + lane);
                        t[r] = v + __ldg(T + (nrel + ((w >> 25) & 1)) * 32 + lane);
                    }
                    __stcs((float4*)(out + ((size_t)(b * L + i0 + r) * L + j) * P) + lane, t[r] + rv[u]);
                }
            }
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
