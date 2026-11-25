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
//   backward  one pass over dz by private-table streams (no atomics), then a fixed-order reduce launch: dleft (row sums), dright (column sums)
//             and the class-row gradients dW[bin][:] -- bitwise repeatable (tpi_bwd / tpi_bwd_w + tpi_bwd_red below).
//
// tbl [141][128] = [Wrel^T ; Wbond^T] (bins 0..138 relative position, 139/140 bond); ids [5][B*L] int32 = asym, residue,
// token, entity, sym; bond [B*L*L] uint8 (0/1).
//
// tpi_fwd first turns the ids of a CTA's rows into one packed word per pair (the four bins and the bond), in shared
// memory, with all threads: pk = bin1 | bin2 << 8 | bin3 << 16 | same_entity << 24 | bond << 25 (every bin < 256: the largest,
// bin3 = 4 r_max + 2 s_max + 5 = n_rel - 2, must be <= 255; the host checks it).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int P = 128;
constexpr int NWARP = 8;

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

__device__ __forceinline__ int ld_id(const void* p, int wide, long k) {
    return wide ? (int)reinterpret_cast<const long long*>(p)[k] : reinterpret_cast<const int*>(p)[k];
}

// The ids of a token row: from the packed [5][B L] int32 tensor (the prep's), or from the caller's five tensors as they are (int32 or int64
// elements, ``wide`` bit k for tensor k).
struct IdsPacked {
    const int* ids;
    int BL;
    __device__ __forceinline__ Row operator()(int idx) const { return load_row(ids, BL, idx); }
};
struct IdsRaw {
    const void *asym, *res, *tok, *ent, *sym;
    int wide;
    __device__ __forceinline__ Row operator()(int idx) const {
        return Row{ld_id(asym, wide & 1, idx), ld_id(res, (wide >> 1) & 1, idx), ld_id(tok, (wide >> 2) & 1, idx), ld_id(ent, (wide >> 3) & 1, idx),
                   ld_id(sym, (wide >> 4) & 1, idx)};
    }
};

// pk[r][j] for the rows i0 .. i0 + nrows - 1 of batch element b (smem), all threads
template <class Ids>
__device__ __forceinline__ void pack_rows(uint32_t* pk, const Ids& ids, const uint8_t* bond, int L, int b, int i0, int nrows, int rmax, int smax) {
    for (int idx = threadIdx.x; idx < nrows * L; idx += blockDim.x) {
        const int r = idx / L, j = idx % L, ri = b * L + i0 + r;
        pk[idx] = pack_pair(ids(ri), ids(b * L + j), bond[(size_t)ri * L + j], rmax, smax);
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
        pack_rows(pk, IdsPacked{ids, BL}, bond, L, b, i0, nrows, rmax, smax);
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

// Forward in ONE launch (the inference call, and the forward of a training step on fp32 weights): no ids / table prep before it. One CTA per SM
// stages TWO kinds of table rows in shared memory (fp32): the 2 (2 rmax + 2) - 1 MERGED rows M[a] = T[residue class] + T[token class] (the token
// class is "other" unless the residue offset is 0 in the same chain: a < nr = residue class a with token "other", a >= nr = residue offset 0 with
// token class a) and the 4 (2 smax + 2) COMBINED rows C[c3 + ns (same_entity + 2 bond)] = (T[chain class c3] + same_entity T[138]) + T[bond row],
// where T = [Wrel^T ; Wbond^T]: a pair's table part is TWO rows, where the five-row sum per position made the kernel compute-bound on ids that change
// at every position (2026-11-17: ~65 warp instructions per pair at L 384 -- 4.1 TB/s of z where a fill writes 7.3; three rows: 24.6 us, two: 22.5 at
// L 384 under the measurer -- the shared-memory reads share the LSU with the z stores). The ids are read as they are (int32 or int64, ``wide`` bit k
// for id tensor k), left / right as fp32 or bf16.
// Work items are (b, row pair, 16 positions): lane = (row r, position) packs its pair's word, then the warp walks the 16 positions, one register copy of
// right[j] serving both rows; the items are split EVENLY over the warps of the grid (contiguous ranges). ``pipe`` (the host sets it where the grid has
// at least two warps per item: a warp's one item is latency, not bandwidth): a whole item looks up every pair's rows without branches, the reads
// of the next positions issuing ahead of this one's adds and stores (L 128: 14.3 -> 12.3 us).
//   z = ((left + M) + C) + right       (tpi_fwd adds the five rows one by one: the same sum to the last bit or two)
#ifndef TPI_W_WARPS
#define TPI_W_WARPS 16                               // 16 warps: 128 registers a thread (the first item's loads are held across the staging)
#endif
constexpr int WW = TPI_W_WARPS;
constexpr int JB = 16;                               // positions per work item (x 2 rows = 32 pairs, one per lane)
#ifndef TPI_W_UNR
#define TPI_W_UNR 8
#endif
constexpr int UW = TPI_W_UNR;                        // positions per group of right[j] loads (JB / 2: two groups per item)

// channels 4 lane .. 4 lane + 3 of row ``row`` of left / right [B L, 128]: fp32 as stored, or bf16 widened (exact) -- the bf16 autocast call reads its
// inputs as they are instead of two cast launches before the kernel
__device__ __forceinline__ float4 ld_lr(const void* p, int bf, size_t row, int lane) {
    if (!bf) return __ldg(reinterpret_cast<const float4*>(p) + row * 32 + lane);
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p) + row * 32 + lane);
    return make_float4(__uint_as_float((u.x & 0xffffu) << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float((u.y & 0xffffu) << 16),
                       __uint_as_float(u.y & 0xffff0000u));
}

// z stores: streaming (st.global.cs; plain and L1::no_allocate stores measured the same, 2026-11-17)
__device__ __forceinline__ void z_store(float4* p, float4 v) {
    __stcs(p, v);
}

// -DTPI_TRACE (a measuring build, never planned): one more argument, ev [grid][WW][10] uint64 -- per warp %globaltimer at entry (0) and exit (9) and
// clock64 at: 1 entry, 2 first loads issued, 3 weights in shared memory, 4 table rows built, 5 after the barrier, 6 first item's first group
// written, 7 first item written, 8 last item written.
#ifdef TPI_TRACE
#define TPI_TRACE_ARG , unsigned long long* __restrict__ ev
#define TPI_EV0()                                                                                     \
    unsigned long long tev[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};                                      \
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tev[0]))
#define TPI_EV(k) tev[k] = clock64()
#define TPI_DUMP()                                                                                    \
    do {                                                                                              \
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tev[9]));                                     \
        if (lane == 0)                                                                                \
            for (int k = 0; k < 10; k++) ev[((size_t)blockIdx.x * WW + warp) * 10 + k] = tev[k];      \
    } while (0)
#else
#define TPI_TRACE_ARG
#define TPI_EV0()
#define TPI_EV(k)
#define TPI_DUMP()
#endif
extern "C" __global__ void __launch_bounds__(WW * 32, 1)
tpi_fwd_w(const void* __restrict__ left, const void* __restrict__ right, const float* __restrict__ w_rel, const float* __restrict__ w_bond,
          const void* __restrict__ id_asym, const void* __restrict__ id_res, const void* __restrict__ id_tok, const void* __restrict__ id_ent,
          const void* __restrict__ id_sym, int wide, const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax, int lr_bf16, int pipe TPI_TRACE_ARG) {
    extern __shared__ __align__(16) float4 Ts[];     // [NM + NC][32] float4: the merged rows, the combined rows; then the raw weights
    const int nr = 2 * rmax + 2, ns = 2 * smax + 2, nrel = 2 * nr + ns + 1, NC = 4 * ns, NM = 2 * nr - 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NI = (L + 1) >> 1, NJ = (L + JB - 1) / JB;
    const long long I = (long long)B * NI * NJ, G = (long long)gridDim.x * WW, g = (long long)blockIdx.x * WW + warp;
    const int it0 = (int)(g * I / G), it1 = (int)((g + 1) * I / G);   // the host keeps I < 2^31: the items in 32-bit arithmetic from here
    const IdsRaw ids{id_asym, id_res, id_tok, id_ent, id_sym, wide};
    struct Item {
        int b, i0, nrows, j0, nj;
    };
    auto item = [&](int it) -> Item {
        const unsigned jb = (unsigned)it % (unsigned)NJ, rest = (unsigned)it / (unsigned)NJ;
        const int ip = (int)(rest % (unsigned)NI), b = (int)(rest / (unsigned)NI);
        return Item{b, 2 * ip, min(2, L - 2 * ip), (int)jb * JB, min(JB, L - (int)jb * JB)};
    };
    // lane's pair of an item: row i0 + (lane >> 4), position j0 + (lane & 15). Its ids and bond are LOADED by ld_pair and turned into the
    // packed word (residue row | token row << 8 | combined row << 16) by pair_word later: a load is not followed by a use of its value
    // before the other loads of the step are issued
    struct Pair {
        Row a, c;
        uint32_t bond;                               // 0xffffffff: no pair (past the item, the rows or the positions)
    };
    auto ld_pair = [&](int it) -> Pair {
        Pair p;
        p.bond = 0xffffffffu;
        if (it < it1) {
            const Item q = item(it);
            const int i = q.i0 + (lane >> 4), j = q.j0 + (lane & 15);
            if (i < L && j < L) {
                const int ri = q.b * L + i;
                p.a = ids(ri);
                p.c = ids(q.b * L + j);
                p.bond = bond[(size_t)ri * L + j];
            }
        }
        return p;
    };
    auto pair_word = [&](const Pair& p) -> uint32_t {
        if (p.bond == 0xffffffffu) return 0xffffffffu;
        const uint32_t w = pack_pair(p.a, p.c, (uint8_t)p.bond, rmax, smax);
        const uint32_t c = ((w >> 16) & 255u) - 2u * nr + (uint32_t)ns * ((w >> 24) & 3u);   // chain class + ns (same entity + 2 bond)
        const uint32_t b1 = w & 255u, b2 = (w >> 8) & 255u;   // the residue x token row: b1, or the token class where the residue offset is 0
        return (b1 == (uint32_t)rmax ? b2 : b1) | ((uint32_t)(NM + c) << 16);
    };
    auto ld_left = [&](const Item& q, float4* lf) {
#pragma unroll
        for (int r = 0; r < 2; r++) lf[r] = r < q.nrows ? ld_lr(left, lr_bf16, (size_t)(q.b * L + q.i0 + r), lane) : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    auto ld_right = [&](const Item& q, int u0, float4* rv) {
#pragma unroll
        for (int u = 0; u < UW; u++) rv[u] = ld_lr(right, lr_bf16, (size_t)(q.b * L + q.j0 + min(u0 + u, q.nj - 1)), lane);
    };
    // Every load is issued one step before its value is needed, and no value is used before the loads beside it are issued, so a warp
    // waits ONE memory round trip where it waited one per step (2026-11-17: the kernel was a constant ~8 us of serial round trips at every L
    // -- the L2 is cold: ids, then left, then four groups of right, then the weights -- before its ~7 TB/s of z):
    //   * the weights' bulk copies, the first item's ids and bond, its left rows and first right group, and the second item's ids and bond are
    //     all in flight before the first of them is used (the table staging);
    //   * inside an item, the second right group before the first is written, the next item's left / first right group before the second,
    //     and the ids / bond of the item after the next.
    TPI_EV0();
    TPI_EV(1);
    // The weights reach shared memory as they are by bulk copies (cp.async.bulk, issued first by one thread: whole lines at the copy engine's
    // rate, nothing held in registers), and ONE pass builds the merged and the combined rows from that raw copy -- one barrier where the per-lane
    // transposing loads took two (2026-11-17, -DTPI_TRACE at L 384: per-lane loads of 32 floats of a channel, each straddling two lines, kept the
    // warps waiting on the SM's outstanding misses, and four warps per CTA took two round trips; L 384 19.35 -> 18.53 us a launch back to back).
    //   raw: w_rel [128][nrel] at float offset (NM + NC) * 128, w_bond [128][2] behind it; the mbarrier behind both.
    float* raw = reinterpret_cast<float*>(Ts + (NM + NC) * 32);
    const float* rawb = raw + P * nrel;
    uint64_t* bar = reinterpret_cast<uint64_t*>(raw + P * (nrel + 2));
    const uint32_t bar_a = (uint32_t)__cvta_generic_to_shared(bar);
    if (threadIdx.x == 0) {
        const uint32_t nw = (uint32_t)(P * nrel * 4), nb = (uint32_t)(P * 2 * 4);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_a) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_a), "r"(nw + nb) : "memory");
        const uint32_t dst = (uint32_t)__cvta_generic_to_shared(raw);
        constexpr uint32_t CH = 16384;               // bytes per copy (several in flight)
        for (uint32_t o = 0; o < nw; o += CH)
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst + o),
                         "l"(reinterpret_cast<const char*>(w_rel) + o), "r"(min(CH, nw - o)), "r"(bar_a)
                         : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst + nw), "l"(w_bond),
                     "r"(nb), "r"(bar_a)
                     : "memory");
    }
    const Pair p0 = ld_pair(it0);
    Item q{0, 0, 0, 0, 0};
    float4 lf[2], rva[UW], rvb[UW];
    if (it0 < it1) {
        q = item(it0);
        ld_left(q, lf);
        ld_right(q, 0, rva);
    }
    const Pair p1 = ld_pair(it0 + 1);                // the second item's ids / bond: in flight behind the first item's loads and the weights
    TPI_EV(2);
    __syncthreads();                                 // the mbarrier's init seen by every thread
    {
        uint32_t ok = 0;
        while (!ok)
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                         : "=r"(ok)
                         : "r"(bar_a), "r"(0u)
                         : "memory");
    }
    TPI_EV(3);
    {   // TWO table rows per pair: the merged residue x token rows M[a] = T[residue class] + T[token class] (a < nr: residue class a, token "other";
        // a >= nr: residue offset 0, token class a) and the combined rows, both from the raw copy, a thread per channel (raw reads at a stride of
        // nrel words: odd, no bank conflict; the row stores consecutive words). No class rows: nothing reads them.
        float* Tm = reinterpret_cast<float*>(Ts);
        const int ch = threadIdx.x & (P - 1);
        // the row both halves share is read once per thread (token "other" for a < nr, residue offset 0 for a >= nr); four rows in flight
        const float* rc = raw + ch * nrel;
        const float to = rc[2 * nr - 1], r0 = rc[rmax];
        const int a0 = threadIdx.x / P, st = WW * 32 / P;
        int a = a0;
        for (; a + 3 * st < NM; a += 4 * st) {
            float v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) v[k] = rc[a + k * st];
#pragma unroll
            for (int k = 0; k < 4; k++) Tm[(a + k * st) * P + ch] = a + k * st < nr ? v[k] + to : r0 + v[k];
        }
        for (; a < NM; a += st) Tm[a * P + ch] = a < nr ? rc[a] + to : r0 + rc[a];
        for (int c = threadIdx.x / P; c < NC; c += WW * 32 / P) {
            const int c3 = c % ns, sb = c / ns, r3 = 2 * nr + c3, re = nrel - 1;
            float v = raw[ch * nrel + r3];
            if (sb & 1) v += raw[ch * nrel + re];
            Tm[(NM + c) * P + ch] = v + rawb[ch * 2 + (sb >> 1)];
        }
    }
    TPI_EV(4);
    __syncthreads();                                 // the table rows: nothing of the ids is waited for here (a CTA-wide wait for the slowest warp)
    uint32_t cur = pair_word(p0), nxt = 0xffffffffu;
    TPI_EV(5);
    for (int it = it0; it < it1; ++it) {
        const Pair pn = ld_pair(it + 2);
        const int b = q.b, i0 = q.i0, nrows = q.nrows, j0 = q.j0, nj = q.nj;
        float4 t[2];
        uint32_t last[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            t[r] = lf[r];
            last[r] = 0xffffffffu;                   // no packed word is all ones
        }
        // left + the pair's two table rows (merged residue x token, combined)
        auto look = [&](const float4& l, uint32_t w) -> float4 {
            return (l + Ts[(w & 0xffffu) * 32 + lane]) + Ts[(w >> 16) * 32 + lane];
        };
        auto group = [&](int u0, const float4* rv) {
            if (pipe && nj == JB && nrows == 2) {    // a whole item: no branches, every pair's rows looked up (the reads of the next
                uint32_t wv[2][UW];                  // positions issue ahead of the adds and stores of this one)
#pragma unroll
                for (int u = 0; u < UW; u++)
#pragma unroll
                    for (int r = 0; r < 2; r++) wv[r][u] = __shfl_sync(0xffffffffu, cur, r * 16 + u0 + u);
#pragma unroll
                for (int u = 0; u < UW; u++)
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        const uint32_t w = wv[r][u];
                        t[r] = look(lf[r], w);
                        z_store(reinterpret_cast<float4*>(out + ((size_t)(b * L + i0 + r) * L + j0 + u0 + u) * P) + lane, t[r] + rv[u]);
                    }
#pragma unroll
                for (int r = 0; r < 2; r++) last[r] = wv[r][UW - 1];
                return;
            }
#pragma unroll
            for (int u = 0; u < UW; u++) {
                if (u0 + u >= nj) break;
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    if (r >= nrows) break;
                    const uint32_t w = __shfl_sync(0xffffffffu, cur, r * 16 + u0 + u);
                    if (w != last[r]) {
                        last[r] = w;
                        t[r] = look(lf[r], w);
                    }
                    z_store(reinterpret_cast<float4*>(out + ((size_t)(b * L + i0 + r) * L + j0 + u0 + u) * P) + lane, t[r] + rv[u]);
                }
            }
        };
        if (nj > UW) ld_right(q, UW, rvb);           // the second group's loads, in flight while the first group is written
        group(0, rva);
        if (it == it0) {
            nxt = pair_word(p1);                     // the second item's word: its ids were issued before the staging (in flight until here)
            TPI_EV(6);
        }
        Item qn = q;
        float4 lfn[2];
        const bool more = it + 1 < it1;
        if (more) {                                  // the next item's left rows and first group, in flight while the second group is written
            qn = item(it + 1);
            ld_left(qn, lfn);
            ld_right(qn, 0, rva);
        }
        if (nj > UW) group(UW, rvb);
        if (it == it0) TPI_EV(7);
        if (more) {
#pragma unroll
            for (int r = 0; r < 2; r++) lf[r] = lfn[r];
        }
        q = qn;
        cur = nxt;
        nxt = pair_word(pn);
    }
    TPI_EV(8);
    TPI_DUMP();
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// Backward (2026-11-17, bitwise repeatable: no atomics). dz [B][L][L][128] fp32 is read ONCE by "streams": a stream is 4 / BCL warps of a CTA that walk
// the same run of units -- a unit is BC positions j of one row i -- each warp on its own slice of the 128 channels (a lane holds BCL channels), all of them
// adding into ONE private class table of the stream in shared memory (disjoint columns per warp: plain load / add / store, in the order of the walk).
//
//   * TWO table rows per pair, where the forward's sum has five: the residue and token classes are one class (the token class is "other" unless the
//     residue offset is 0 in the same chain: rows 0 .. nr-1 = residue class with token class "other", rows nr .. nr + 2 rmax = token class with residue
//     class rmax), and chain class x same-entity x bond another (same entity <=> chain class != "other": 2 ns rows). The CTA expands its streams' tables
//     to the nbin class rows of the forward's table at the end (fixed order) and writes them to wpart [grid][nbin][128].
//   * The units are ordered (b, column chunk, row) and split EVENLY and STATICALLY over the streams of the grid (contiguous runs): a stream walks DOWN
//     the rows of a column chunk, the column sums of its run in registers (stored once per chunk and stream: cpart [streams][K][BC][128]) and the row
//     sum of each unit (rpart [units][128]: dz / BC).
//   * tpi_bwd_red adds the partials in a fixed order: dleft[b, i] over the column chunks, dright[b, j] over the streams of its chunk, the table
//     gradient over the CTAs; dleft / dright in fp32 or bf16 (the fp32 sums rounded once).
// The loads of a unit (dz, ids, bond) are issued one unit ahead, the first before the table is zeroed.
#ifndef TPI_BCL
#define TPI_BCL 1                                    // channels per lane: 1 / 2 / 4 -> 4 / 2 / 1 warps per stream
#endif
#ifndef TPI_BC
#define TPI_BC 32                                    // positions per unit (a divisor of 32: lane % BC makes the pair's class)
#endif
#ifndef TPI_BST
#define TPI_BST 3                                    // streams per CTA at most (three 73 KB tables fill the SM's shared memory at r_max 32)
#endif
constexpr int BCL = TPI_BCL, BC = TPI_BC, BSW = 4 / BCL;
#if TPI_BCL == 1
typedef float bv_t;
__device__ __forceinline__ bv_t bv_zero() { return 0.f; }
#elif TPI_BCL == 2
typedef float2 bv_t;
__device__ __forceinline__ bv_t bv_zero() { return make_float2(0.f, 0.f); }
__device__ __forceinline__ void operator+=(float2& a, float2 b) { a.x += b.x; a.y += b.y; }
#else
typedef float4 bv_t;
__device__ __forceinline__ bv_t bv_zero() { return make_float4(0.f, 0.f, 0.f, 0.f); }
#endif

// first unit of stream s: floor(s U / S) in 32-bit arithmetic (q = U / S, r = U % S; s r < S^2)
__device__ __forceinline__ uint32_t stream_begin(uint32_t s, uint32_t q, uint32_t r, uint32_t S) { return s * q + (s * r) / S; }

template <class Ids>
__device__ __forceinline__ void bwd_body(const float* __restrict__ g, const Ids& ids, const uint8_t* __restrict__ bond, float* __restrict__ rpart,
                                         float* __restrict__ cpart, float* __restrict__ wpart, int B, int L, int rmax, int smax, int nst, int K) {
    extern __shared__ __align__(16) float sm[];      // [nst][NT][128]: the streams' class tables
    const int nr = 2 * rmax + 2, ns = 2 * smax + 2, nrel = 2 * nr + ns + 1, nbin = nrel + 2, NA = 2 * nr - 1, NT = NA + 2 * ns;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, st = warp / BSW, chan = ((warp % BSW) * 32 + lane) * BCL, pl = lane % BC;
    const uint32_t NC = (uint32_t)(L + BC - 1) / BC, U = (uint32_t)B * NC * (uint32_t)L, S = gridDim.x * (uint32_t)nst;
    const uint32_t s = blockIdx.x * (uint32_t)nst + st, uq = U / S, ur = U % S;
    const uint32_t u0 = stream_begin(s, uq, ur, S), u1 = stream_begin(s + 1, uq, ur, S);
    float* Tp = sm + (size_t)st * NT * P + chan;      // this lane's columns of the stream's table
    struct Unit {
        uint32_t ch, b, i, j0;
        int nj;
    };
    auto unit = [&](uint32_t u) -> Unit {
        const uint32_t ch = u / (uint32_t)L, b = ch / NC, jc = ch - b * NC;
        return Unit{ch, b, u - ch * (uint32_t)L, jc * BC, min(BC, L - (int)(jc * BC))};
    };
    auto ld_g = [&](const Unit& q, bv_t* gv) {
        const float* gp = g + ((size_t)(q.b * L + q.i) * L + q.j0) * P + chan;
#pragma unroll
        for (int p = 0; p < BC; p++) gv[p] = p < q.nj ? __ldg(reinterpret_cast<const bv_t*>(gp + (size_t)p * P)) : bv_zero();
    };
    auto ld_col = [&](const Unit& q) -> Row { return ids((int)(q.b * L + q.j0) + min(pl, q.nj - 1)); };
    // the lane's pair of the unit, (i, j0 + lane % BC): its two table rows as byte offsets of the stream's table
    auto words = [&](const Row& a, const Row& c, uint32_t bd, uint32_t& wa, uint32_t& wb) {
        int b1, b2, b3, se;
        pair_bins(a, c, rmax, smax, b1, b2, b3, se);
        wa = (uint32_t)(b1 == rmax ? b2 : b1) * (P * 4);
        wb = (uint32_t)(NA + (b3 - 2 * nr) + ns * (int)bd) * (P * 4);
    };
    bv_t gc[BC], col[BC];
    Unit q{0, 0, 0, 0, 0};
    Row a{0, 0, 0, 0, 0}, c{0, 0, 0, 0, 0};
    uint32_t bd = 0;
    const bool any = st < nst && u0 < u1;
    if (any) {
        q = unit(u0);
        ld_g(q, gc);
        a = ids((int)(q.b * L + q.i));
        c = ld_col(q);
        bd = bond[(size_t)(q.b * L + q.i) * L + q.j0 + min(pl, q.nj - 1)];
    }
    if (st < nst)
        for (int r = 0; r < NT; r++) *reinterpret_cast<bv_t*>(Tp + r * P) = bv_zero();
    if (any) {
        uint32_t wa, wb;
        words(a, c, bd, wa, wb);
        const uint32_t ch0 = q.ch;
#pragma unroll
        for (int p = 0; p < BC; p++) col[p] = bv_zero();
        for (uint32_t u = u0; u < u1; ++u) {
            const bool more = u + 1 < u1;
            Unit qn = q;
            bv_t gn[BC];
            Row an = a, cn = c;
            uint32_t bdn = 0;
            if (more) {                              // the next unit's loads, in flight while this one is added
                qn = unit(u + 1);
                ld_g(qn, gn);
                an = ids((int)(qn.b * L + qn.i));
                if (qn.ch != q.ch) cn = ld_col(qn);
                bdn = bond[(size_t)(qn.b * L + qn.i) * L + qn.j0 + min(pl, qn.nj - 1)];
            }
            bv_t rs = bv_zero();
            auto pair = [&](int p) {                 // position p of the unit: its two table rows, its column sum, the row sum
                const uint32_t oa = __shfl_sync(0xffffffffu, wa, p), ob = __shfl_sync(0xffffffffu, wb, p);
                const bv_t v = gc[p];
                bv_t* ta = reinterpret_cast<bv_t*>(reinterpret_cast<char*>(Tp) + oa);
                bv_t* tb = reinterpret_cast<bv_t*>(reinterpret_cast<char*>(Tp) + ob);
                bv_t x = *ta, y = *tb;
                x += v;
                y += v;
                *ta = x;
                *tb = y;
                col[p] += v;
                rs += v;
            };
            if (q.nj == BC) {                        // a whole unit: straight-line code
#pragma unroll
                for (int p = 0; p < BC; p++) pair(p);
            } else {
#pragma unroll
                for (int p = 0; p < BC; p++)
                    if (p < q.nj) pair(p);
            }
            *reinterpret_cast<bv_t*>(rpart + (size_t)u * P + chan) = rs;
            if (!more || qn.ch != q.ch) {            // the stream leaves the column chunk: its column sums
                float* cp = cpart + ((size_t)(s * K + (q.ch - ch0)) * BC) * P + chan;
#pragma unroll
                for (int p = 0; p < BC; p++) {
                    if (p < q.nj) *reinterpret_cast<bv_t*>(cp + (size_t)p * P) = col[p];
                    col[p] = bv_zero();
                }
            }
            if (more) {
                words(an, cn, bdn, wa, wb);
#pragma unroll
                for (int p = 0; p < BC; p++) gc[p] = gn[p];
                a = an;
                c = cn;
                q = qn;
            }
        }
    }
    __syncthreads();
    // the CTA's tables as the nbin class rows of the forward's table: a warp per class row, a lane per four channels (fixed order: streams, then
    // rows; four loads in flight)
    const float4* T4 = reinterpret_cast<const float4*>(sm);
    const int nw = blockDim.x >> 5;
    for (int bin = warp; bin < nbin; bin += nw) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        auto add = [&](int r0, int r1) {             // rows r0 .. r1 - 1 of every stream
            for (int t = 0; t < nst; t++) {
                int r = r0;
                for (; r + 4 <= r1; r += 4) {
                    float4 v[4];
#pragma unroll
                    for (int k = 0; k < 4; k++) v[k] = T4[((size_t)t * NT + r + k) * 32 + lane];
#pragma unroll
                    for (int k = 0; k < 4; k++) acc += v[k];
                }
                for (; r < r1; r++) acc += T4[((size_t)t * NT + r) * 32 + lane];
            }
        };
        if (bin < nr) {                              // residue class: offset 0 in the same chain is the sum of the token-class rows
            if (bin == rmax) add(nr, nr + 2 * rmax + 1);
            else add(bin, bin + 1);
        } else if (bin < 2 * nr) {                   // token class: "other" is every pair but residue offset 0
            if (bin == 2 * nr - 1) {
                add(0, rmax);
                add(rmax + 1, nr);
            } else {
                add(bin, bin + 1);
            }
        } else if (bin < 2 * nr + ns) {              // chain class (both bond states)
            add(NA + bin - 2 * nr, NA + bin - 2 * nr + 1);
            add(NA + ns + bin - 2 * nr, NA + ns + bin - 2 * nr + 1);
        } else if (bin == nrel - 1) {                // same entity: every chain class but "other"
            add(NA, NA + ns - 1);
            add(NA + ns, NA + 2 * ns - 1);
        } else {                                     // bond 0 / 1
            add(NA + (bin - nrel) * ns, NA + (bin - nrel) * ns + ns);
        }
        reinterpret_cast<float4*>(wpart)[((size_t)blockIdx.x * nbin + bin) * 32 + lane] = acc;
    }
}

extern "C" __global__ void __launch_bounds__(BSW * 32 * TPI_BST, 1)
tpi_bwd(const float* __restrict__ g, const int* __restrict__ ids, const uint8_t* __restrict__ bond, float* __restrict__ rpart,
        float* __restrict__ cpart, float* __restrict__ wpart, int B, int L, int rmax, int smax, int nst, int K) {
    bwd_body(g, IdsPacked{ids, B * L}, bond, rpart, cpart, wpart, B, L, rmax, smax, nst, K);
}

// The backward on the caller's five id tensors as they are (no ids prep before it: the forward was tpi_fwd_w).
extern "C" __global__ void __launch_bounds__(BSW * 32 * TPI_BST, 1)
tpi_bwd_w(const float* __restrict__ g, const void* __restrict__ id_asym, const void* __restrict__ id_res, const void* __restrict__ id_tok,
          const void* __restrict__ id_ent, const void* __restrict__ id_sym, int wide, const uint8_t* __restrict__ bond, float* __restrict__ rpart,
          float* __restrict__ cpart, float* __restrict__ wpart, int B, int L, int rmax, int smax, int nst, int K) {
    bwd_body(g, IdsRaw{id_asym, id_res, id_tok, id_ent, id_sym, wide}, bond, rpart, cpart, wpart, B, L, rmax, smax, nst, K);
}

__device__ __forceinline__ void st_grad(void* d, int bf, size_t idx, float v) {
    if (bf) reinterpret_cast<__nv_bfloat16*>(d)[idx] = __float2bfloat16_rn(v);
    else reinterpret_cast<float*>(d)[idx] = v;
}
// The fixed-order sums of tpi_bwd's partials: block x < B L: dleft row x; < 2 B L: dright row x - B L; then one class row of the table gradient
// each. ``G`` / ``nst``: the grid and streams per CTA of the tpi_bwd launch. A thread's terms are LOADED RB at a time and then added in order
// (2026-11-17: a load consumed by the add behind it is one memory round trip per term, ~0.4 us each here -- 148 CTA terms were ~60 us).
constexpr int RB = 32;

extern "C" __global__ void __launch_bounds__(P)
tpi_bwd_red(const float* __restrict__ rpart, const float* __restrict__ cpart, const float* __restrict__ wpart, void* __restrict__ dleft,
            void* __restrict__ dright, float* __restrict__ dtbl, int B, int L, int nbin, int G, int nst, int K, int dl_bf, int dr_bf) {
    const uint32_t NC = (uint32_t)(L + BC - 1) / BC, U = (uint32_t)B * NC * (uint32_t)L, S = (uint32_t)G * (uint32_t)nst;
    const int ch = threadIdx.x, BL = B * L;
    float acc = 0.f, v[RB];
    if ((int)blockIdx.x < BL) {
        const uint32_t b = blockIdx.x / (uint32_t)L, i = blockIdx.x % (uint32_t)L;
        for (uint32_t j0 = 0; j0 < NC; j0 += RB) {
#pragma unroll
            for (int k = 0; k < RB; k++) v[k] = j0 + k < NC ? rpart[((size_t)(b * NC + j0 + k) * L + i) * P + ch] : 0.f;
#pragma unroll
            for (int k = 0; k < RB; k++) acc += v[k];
        }
        st_grad(dleft, dl_bf, (size_t)blockIdx.x * P + ch, acc);
    } else if ((int)blockIdx.x < 2 * BL) {
        const uint32_t x = blockIdx.x - BL, b = x / (uint32_t)L, j = x % (uint32_t)L, cc = b * NC + j / BC, p = j % BC;
        const uint32_t ua = cc * (uint32_t)L, ub = ua + L, uq = U / S, ur = U % S;
        uint32_t s = (uint32_t)(((unsigned long long)ua * S) / U);
        while (s > 0 && stream_begin(s, uq, ur, S) > ua) --s;
        while (stream_begin(s + 1, uq, ur, S) <= ua) ++s;   // the stream that holds the chunk's first unit
        // stream s holds units [a0, a0 + n): n = uq (+ 1 where the remainder carries); its first column chunk c0 = a0 / L -- all advanced
        // without a division (the divisions per term were the reduce launch: ~100 instructions x 111 streams per row at L 128)
        uint32_t a0 = stream_begin(s, uq, ur, S), rem = (s * ur) % S, c0 = a0 / (uint32_t)L, m0 = a0 - c0 * (uint32_t)L;
        bool more = true;
        while (more) {
#pragma unroll
            for (int k = 0; k < RB; k++) {
                v[k] = 0.f;
                while (more) {                       // the next stream with units in this chunk
                    if (s >= S || a0 >= ub) {
                        more = false;
                        break;
                    }
                    const bool carry = rem + ur >= S;
                    const uint32_t n = uq + (carry ? 1u : 0u), sk = s, ck = c0;
                    rem = carry ? rem + ur - S : rem + ur;
                    a0 += n;
                    m0 += n;
                    while (m0 >= (uint32_t)L) {
                        m0 -= (uint32_t)L;
                        ++c0;
                    }
                    ++s;
                    if (n == 0) continue;
                    v[k] = cpart[((size_t)(sk * K + (cc - ck)) * BC + p) * P + ch];
                    break;
                }
            }
#pragma unroll
            for (int k = 0; k < RB; k++) acc += v[k];
        }
        st_grad(dright, dr_bf, (size_t)x * P + ch, acc);
    } else {
        const int bin = blockIdx.x - 2 * BL;
        for (int t0 = 0; t0 < G; t0 += RB) {
#pragma unroll
            for (int k = 0; k < RB; k++) v[k] = t0 + k < G ? wpart[((size_t)(t0 + k) * nbin + bin) * P + ch] : 0.f;
#pragma unroll
            for (int k = 0; k < RB; k++) acc += v[k];
        }
        dtbl[(size_t)bin * P + ch] = acc;
    }
}
