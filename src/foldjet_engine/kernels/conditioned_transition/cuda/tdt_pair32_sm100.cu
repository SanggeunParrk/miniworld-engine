// tdt_pair32_sm100.cu -- fp32 (kind::tf32) GEMM on a CTA PAIR (cta_group::2) for the token DiT's few-row fp32 inference steps, sm_100a
// (round 3 part 9, 2026-11-18). C[M, N] = A[M, K] W[N, K]^T with a row-local epilogue, fp32 operands (the tensor cores read the TF32
// bits of each value, as cuBLAS's TF32 kernels do), fp32 accumulation.
//
// One 2-CTA cluster per (row tile of 2 HR rows, N chunk of 2 NH weight rows). CTA r loads rows [HR r, HR r + HR) of the tile's A and
// weight rows [NH r, NH r + NH) of the chunk; the leader issues the M = 2 HR, N = 2 NH products (cta_group::2) and the whole K lands in
// tensor memory: no K split, no partial exchange. Per SM the operand intake is (HR + NH) K 4 bytes for an HR x 2 NH output (a one-SM
// tile of the same output reads (HR + 2 NH) K 4).
//   HR 64 (M = 128 pair, measured by tsx_pairtest.cu): CTA r holds ITS 64 rows x the whole N folded over its 128 lanes -- lanes 0-63
//     the leader's NH weight rows' columns, lanes 64-127 the peer's (same rows, same column addresses); NH tensor-memory columns.
//   HR 128 (M = 256 pair): CTA r holds its 128 rows x the whole N (columns [0, NH) the leader's half, [NH, 2 NH) the peer's).
// Operand boxes: 3-D TMA (32 fp32 of K, rows, KBX K-blocks), KBX K-blocks per box and ring stage -- the per-SM BOX rate, not the bytes,
// paces these small-M kernels (Transition part 14), so a stage is one A box + one W box. Lane s of the producer warp issues ring stage s;
// the weight boxes of the first ring round are issued before griddepcontrol.wait (they do not depend on the previous kernel).
//
// EPI 0  out[:, n] = acc + bias[n] (bias may be null); columns n < rnac rounded to TF32 (cvt.rna: the TF32 attention core's operands)
// EPI 1  h = silu(a) b u (Boltz-2's up-gate SwiGLU): each NH-row weight half is [Wa | Wb | Wu] of U = NH / 3 hidden units (the packed
//        weight, ``pair32.pack_swiglu``); h of unit (half index) U + u; rounded to TF32 (the squeeze GEMM's operand)
// EPI 2  h = silu(a) b: halves [Wa | Wb] of U = NH / 2 units (else as EPI 1)
// EPI 3  out = x + sigmoid(gl[row % P]) acc (the block's last residual, ``resgate_out_rows``)
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef HR
#define HR 64            // rows per CTA (the pair's M is 2 HR)
#endif
#ifndef NH
#define NH 128           // weight rows per CTA (the pair's N is 2 NH)
#endif
#ifndef KD
#define KD 768           // K
#endif
#ifndef KBX
#define KBX 2            // K-blocks of 32 per TMA box / ring stage
#endif
#ifndef EPI
#define EPI 0
#endif
#ifndef NG
#define NG 1             // epilogue warpgroups (column chunks split between them)
#endif
#ifndef MAXSTG
#define MAXSTG 8
#endif

#ifndef NOLOAD
#define NOLOAD 0     // (timing builds) no operand loads
#endif
#ifndef NOEPI
#define NOEPI 0
#endif
#ifndef NOTLD
#define NOTLD 0
#endif
#ifndef NOXP
#define NOXP 0
#endif
#ifndef NOSTORE
#define NOSTORE 0    // (timing builds) no output stores
#endif
#ifndef NOMMA
#define NOMMA 0      // (timing builds) no MMAs
#endif
constexpr int NKB = KD / 32, NS = NKB / KBX;
constexpr int NMMA = 2 * NH, MMA_M = 2 * HR;
static_assert(HR == 64 || HR == 128, "M = 128 or 256 pair");
static_assert(NKB % KBX == 0 && KD % 32 == 0, "whole stages");
static_assert(NMMA % 16 == 0 && NMMA >= 32 && NMMA <= 256 && NH % 8 == 0, "pair N");
constexpr int AKB = HR * 128, WKB = NH * 128;                     // bytes of one K-block slab (128-B rows of 32 fp32)
constexpr int AST = KBX * AKB, WST = KBX * WKB;
constexpr int CAP0 = (232448 - 2048) / (AST + WST);
constexpr int CAP = CAP0 < MAXSTG ? CAP0 : MAXSTG;
constexpr int STG = NS < CAP ? NS : CAP;
static_assert(STG >= 1 && STG <= 32, "ring stages");
constexpr uint32_t TXB = NOLOAD ? 0u : (uint32_t)(2 * (AST + WST));
constexpr int O_A = 0, O_W = STG * AST, O_BAR = O_W + STG * WST, SMEM_BYTES = 1024 + O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr int DCOLS = HR == 64 ? NH : NMMA;                       // tensor-memory columns of the product
constexpr uint32_t TCOLS = DCOLS <= 32 ? 32 : DCOLS <= 64 ? 64 : DCOLS <= 128 ? 128 : 256;
constexpr uint32_t IDESC = idesc_tf32(MMA_M, NMMA);
constexpr int UC = EPI == 1 ? NH / 3 : EPI == 2 ? NH / 2 : NH;    // output columns of one weight half
constexpr int CH = UC % 32 == 0 ? 32 : 16;                        // epilogue column chunk
static_assert(EPI == 0 || EPI == 3 || (UC * (EPI == 1 ? 3 : 2) == NH), "SwiGLU halves");
static_assert(UC % CH == 0 && UC % 16 == 0, "column chunks");

struct Bars {
  uint64_t full[STG], empty[STG], mma_done;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 256, "barrier block");

DEVI void tma_load_3d_2sm(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void umma_tf32_2(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void cluster_sync_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }
DEVI float rna(float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return __uint_as_float(r); }
DEVI void xchg16(uint4& keep, uint4& give, bool up, int m) {     // (tsx_store.cuh)
  uint4 t = up ? keep : give;
  t.x = __shfl_xor_sync(0xffffffffu, t.x, m); t.y = __shfl_xor_sync(0xffffffffu, t.y, m);
  t.z = __shfl_xor_sync(0xffffffffu, t.z, m); t.w = __shfl_xor_sync(0xffffffffu, t.w, m);
  if (up) keep = t; else give = t;
}
DEVI float sigm(float v) { return 1.f / (1.f + __expf(-v)); }
template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r); else tmem_ld16(taddr, r);
}

#ifndef STAMP
#define STAMP 0
#endif
#if STAMP
#define ST(i) do { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) :: "memory"); e.stp[blockIdx.x * 8 + (i)] = t_; } while (0)
#else
#define ST(i) do { } while (0)
#endif
struct Epi {
  float* out; long ldo;            // EPI 0 / 3: [M, N]; EPI 1 / 2: h [M, N / (3 or 2)]
  const float* bias; int rnac;     // EPI 0
  const float* x; long ldx;        // EPI 3: the residual rows
  const float* gl; long ldg; int P;  // EPI 3: the gate rows (row % P)
  unsigned long long* stp;         // -DSTAMP=1 (timing builds): %globaltimer stamps [CTA][8]
};

// maps: a [M][K] fp32 as (32, M, K / 32), box (32, HR, KBX); w [N][K] fp32 as (32, N, K / 32), box (32, NH, KBX).
extern "C" __global__ void __launch_bounds__(128 + 128 * NG, 1)
tdt_pair32(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw, int M, int npn, Epi e) {
  extern __shared__ uint8_t raw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)raw + 1023) & ~(uintptr_t)1023);
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x >> 1, rank = (int)cluster_rank();
  const bool leader = rank == 0;
  const int tile = pair / npn, j = pair % npn;
  const int arow = tile * MMA_M + rank * HR, wrow = j * NMMA + rank * NH;

  if (tid == 0) ST(0);
  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.mma_done, 1);
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mw);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), TCOLS); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();                                        // the peer's TMA completes on the leader's barriers
  tc_fence_after();
  if (tid == 0) ST(1);
  const uint32_t tmem = B.tmem;
  if (warp == 0 && lane < STG) {                                 // the weights of ring stage `lane` before the dependency wait
    if (leader) mbar_expect_tx(&B.full[lane], TXB);
    if (!NOLOAD) tma_load_3d_2sm(su + O_W + lane * WST, &mw, &B.full[lane], 0, wrow, lane * KBX);
  }
  pdl_wait();
  if (tid == 0) ST(2);
  pdl_launch();

  if (warp == 0) {
    if (lane < STG)
      for (int st = lane; st < NS; st += STG) {
        if (st >= STG) {
          mbar_wait(&B.empty[lane], ((st / STG) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[lane], TXB);
          if (!NOLOAD) tma_load_3d_2sm(su + O_W + lane * WST, &mw, &B.full[lane], 0, wrow, st * KBX);
        }
        if (!NOLOAD) tma_load_3d_2sm(su + O_A + lane * AST, &ma, &B.full[lane], 0, arow, st * KBX);
      }
  } else if (warp == 1) {
    if (leader) {
#pragma unroll 1
      for (int st = 0; st < NS; ++st) {
        const int s = st % STG;
        mbar_wait(&B.full[s], (st / STG) & 1);
        if (lane == 0 && st == 0) ST(3);
        if (lane == 0 && st == NS - 1) ST(4);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int i = 0; i < KBX; ++i) {
            const uint32_t a = su + O_A + s * AST + i * AKB, w = su + O_W + s * WST + i * WKB;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              if (!NOMMA) umma_tf32_2(tmem, desc_k128(a + ks * 32), desc_k128(w + ks * 32), IDESC, (st > 0 || i > 0 || ks > 0) ? 1u : 0u);
          }
          if (NS > STG) tc_commit2_mc(&B.empty[s], 3);
          if (st == NS - 1) tc_commit2_mc(&B.mma_done, 3);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    const int g = (warp - 4) >> 2, q = warp & 3, L = q * 32 + lane;     // tensor-memory lane L
    const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
    const int row = HR == 64 ? tile * MMA_M + rank * HR + (L & 63) : arow + L;
    mbar_wait(&B.mma_done, 0);
    if (warp == 4 && lane == 0) ST(5);
    tc_fence_after();
    constexpr int NSEG = HR == 64 ? 1 : 2, NCH = UC / CH;
#pragma unroll 1
    for (int seg = 0; seg < (NOEPI ? 0 : NSEG); ++seg) {
      const int half = HR == 64 ? (L >> 6) : seg;                  // which CTA's weight rows these columns are
      const uint32_t tc0 = trow + seg * NH;
      const int hidx = j * 2 + half;                                // weight half index along N
#pragma unroll 1
      for (int c = g; c < NCH; c += NG) {
        const int cc = c * CH;
        uint32_t av[CH];
        if (!NOTLD) tld<CH>(tc0 + cc, av); else for (int i = 0; i < CH; ++i) av[i] = lane * i + cc;
#if EPI == 1 || EPI == 2
        uint32_t bv[CH];
        tld<CH>(tc0 + UC + cc, bv);
#if EPI == 1
        uint32_t uv[CH];
        tld<CH>(tc0 + 2 * UC + cc, uv);
#endif
#endif
        tmem_wait_ld();
        // the thread's row segment (CH values), transposed inside groups of C = CH / 4 lanes (tsx_store.cuh's butterfly): lane k of a
        // group then holds 16-byte chunk k of the group's C rows, so every global access below is C rows x 16 C contiguous bytes (a
        // warp storing one row segment per lane wrote 32 rows x 16 B per instruction: ~10 us of a 640 x 768 output, measured)
        constexpr int C = CH / 4;
#if EPI == 1 || EPI == 2
        const int n0 = hidx * UC + cc;
#pragma unroll
        for (int k = 0; k < CH; ++k) {
          const float a = __uint_as_float(av[k]);
          float h = a * sigm(a) * __uint_as_float(bv[k]);
#if EPI == 1
          h *= __uint_as_float(uv[k]);
#endif
          av[k] = __float_as_uint(rna(h));
        }
#else
        const int n0 = hidx * NH + cc;
#endif
        uint4 xv[C];
#pragma unroll
        for (int i = 0; i < C; ++i) xv[i] = make_uint4(av[4 * i], av[4 * i + 1], av[4 * i + 2], av[4 * i + 3]);
        const int kq = lane & (C - 1);
#pragma unroll
        for (int m = 1; m < (NOXP ? 1 : C); m <<= 1)
#pragma unroll
          for (int i = 0; i < C; ++i)
            if (!(i & m)) xchg16(xv[i], xv[i | m], (kq & m) != 0, m);
        const int r0 = row - (L & (C - 1)), col = n0 + kq * 4;     // the group's first row, this lane's four columns
#if EPI == 0
        const float4 bb = e.bias ? __ldg(reinterpret_cast<const float4*>(e.bias + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
        const bool rn = n0 < e.rnac;
#endif
#pragma unroll
        for (int t = 0; t < C; ++t) {
          const int r = r0 + t;
          if (r < M) {
            float4 o = make_float4(__uint_as_float(xv[t].x), __uint_as_float(xv[t].y), __uint_as_float(xv[t].z), __uint_as_float(xv[t].w));
#if EPI == 0
            o.x += bb.x; o.y += bb.y; o.z += bb.z; o.w += bb.w;
            if (rn) { o.x = rna(o.x); o.y = rna(o.y); o.z = rna(o.z); o.w = rna(o.w); }
#elif EPI == 3
            const float4 xx = *reinterpret_cast<const float4*>(e.x + (long)r * e.ldx + col);
            const float4 gg = *reinterpret_cast<const float4*>(e.gl + (long)(r % e.P) * e.ldg + col);
            o.x = xx.x + sigm(gg.x) * o.x; o.y = xx.y + sigm(gg.y) * o.y; o.z = xx.z + sigm(gg.z) * o.z; o.w = xx.w + sigm(gg.w) * o.w;
#endif
            if (!NOSTORE) *reinterpret_cast<float4*>(e.out + (long)r * e.ldo + col) = o;
            else if (o.x == 1234.5f) e.out[0] = o.y;
          }
        }
      }
    }
    if (warp == 4 && lane == 0) ST(6);
    tc_fence_before();
  }
  __syncthreads();
  cluster_sync_relaxed();                                        // the leader's products read the peer's shared memory until mma_done
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, TCOLS); }
  if (tid == 0) ST(7);
}

extern "C" __device__ int tdt_pair32_smem_bytes = SMEM_BYTES;
