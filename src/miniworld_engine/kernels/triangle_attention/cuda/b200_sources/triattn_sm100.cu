// Triangle-attention forward for B200 (sm_100a): the cuEquivariance / opt_core `triangle_attention` op, D = 32, bf16 operands,
// fp32 softmax and accumulation, bf16 out -- the op the H100 kernel (opt_core kernels/triattn/cuda_sm90a) serves, on tcgen05 / TMEM.
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]        (no mask)
//
// A CTA task = one (b, h, 128-query tile) x R pair rows (the bias tile is shared by every pair row: R rows reuse each staged bias
// tile, as the H100 kernel's 3-row CTAs do).  Per 128-key tile j and row r (a "sub-step"):
//   MMA warp:      S = Q_r . K_{r,j}^T                (M = 128 q, N = 128 k, K = 32) into a TMEM buffer
//   softmax warps: x = S * scale*log2e + bias * log2e; p = 2^(x - m_r); l_r += p; P (bf16) written over the S buffer in TMEM
//   MMA warp:      O_r += P . V_{r,j}                  (A = P from TMEM, B = V MN-major, N = 32)
// The softmax offset m_r starts at the maximum of the row's first 32 keys and is raised only when a tile's exponentials sum past 2^64
// (a logit ~57+ log2 units above it): a rare, warp-uniform branch after the tile's chunk loop recomputes the tile from q / k, raises
// the offset to the tile maximum and rescales l_r and O_r, so p < 2^64 for any spread of the logits (an offset that never moved
// overflowed to inf / NaN on trained weights). The steady state carries no running maximum. FLAGS counts rows whose sum left
// (0, 3e38).
// Warps: 0 = TMA producer (Q of the task's rows, K|V per sub-step, the bias tile per key tile), 1 = MMA issue, 2-5 / 6-9 = two softmax
// groups (rows 0, 2 / 1, 3 of the task: sub-step x uses S buffer x & 1 = r & 1) + their rows' epilogue.  A task always has R rows
// (rows past N are computed on row N-1 and not stored), so the sub-step parity is the row parity.
#include <torch/extension.h>
#include "sm100.cuh"
#include <cuda_fp16.h>
using namespace sm100;

// Per-length schedule knobs (the host builds one variant per configuration and picks it by L; see b200_triattn.py):
#ifndef TA_R
#define TA_R 4           // forward: pair rows per CTA task (2 / 4: the two softmax groups take alternate rows; 3: three groups, one row each)
#endif
#ifndef TA_NKV
#define TA_NKV 5         // forward: K | V ring stages
#endif
#ifndef TA_PSHIFT
#define TA_PSHIFT 1      // forward: a finite overflowing tile sum shifts P by a power of two instead of recomputing S (see the overflow guard)
#endif
#ifndef TA_QR
#define TA_QR 4          // backward, query side: pair rows per task (even: the two grad groups take rows gi, gi + 2, ...)
#endif
#ifndef TA_HB
#define TA_HB 0          // forward (inference): the staged bias is bf16(bias * log2 e) (the front writes it so), added by mixed fp32 + bf16 adds
#endif
#ifndef TA_CK
#define TA_CK 32         // forward: keys a softmax chunk (32, or 16: half the S registers a thread; the same bits)
#endif
#ifndef TA_D16
#define TA_D16 0         // forward (inference): 16-channel heads (channels 16 .. 31 of q / k zero) whose V carries 1.0 in channel 16: one QK^T MMA,
                         // the row sum l = O[:, 16] from the PV GEMM (no FADD2 a pair, no per-tile overflow guard: checked at the drain)
#endif
#ifndef TA_D16C
#define TA_D16C 0        // (with TA_D16) compact 16-channel heads in global memory: q / k / v / out [.., H, 16] (the H100 fp16 core's layout).
                         // q / k / v load through 3-D maps [rows][H][16] whose 32-channel box zero-fills channels 16 .. 31 (the smem tiles
                         // are the padded ones); the PV issue warp writes v's channel 16 = 1.0 into each landed V tile (the row sum);
                         // out stores channels 0 .. 15.  Half the bytes through the front's stores, these loads and the tail's o loads.
#endif
static_assert(!TA_D16C || TA_D16, "TA_D16C needs TA_D16");
#ifndef TA_ONES
#define TA_ONES 0        // (B200 2026-11-18 part 12, inference, 32-channel heads) the row sum on the tensor core, the H100 fp16 core's
                         // [O | l] += P [V | 1]: after a sub-step's PV, eight ts MMAs M128 N16 K16 with B = a ones tile accumulate l_r in
                         // TMEM columns COL_L + 16 r (no FADD2 a pair, no per-tile overflow guard: a bad row is recomputed exactly at the
                         // drain, as TA_D16).  TMEM: two rows (480 + 32 columns) only; with three rows only as the timing ceiling TA_ONESX
                         // (row 2 shares row 0's sum: WRONG results)
#endif
#ifndef TA_ONESX
#define TA_ONESX 0
#endif
#ifndef TA_PH
#define TA_PH 0          // (experiment, timing only, WRONG results) P packed as fp16 and the PV MMA reading fp16 operands
#endif
static_assert(!TA_ONES || (TA_HB && !TA_D16), "TA_ONES: the inference (HB) build of 32-channel heads");
#ifndef TA_SPLIT
#define TA_SPLIT 1        // forward: 1 = two issue warps (1: QK^T, 14: PV), 0 = one warp issues both (see the issue loops)
#endif
#ifndef TA_PVI
#define TA_PVI 0          // (experiment) QK^T(y) waits for PV(y - NS) to be ISSUED (tcgen05 fences + an mbarrier), not retired
#endif
namespace {
constexpr int D = 32, BM = 128, BN = 128, R = TA_R;
constexpr int DG = TA_D16C ? 16 : 32;                     // channels a head in global q / k / v / out (the smem tiles are D wide)
static_assert(R >= 2 && R <= 4, "R");
// R = 3 (B200 2026-11-17): three softmax groups, one pair row and one S / P buffer each (group = row = buffer), and each group drains
// its own row (no epilogue warps, no row-sum handoff): three softmax warps a sub-partition keep its one MUFU unit (a warp instruction
// per 8 clk, the bound of the op: 128 exponentials a row and tile) busy while a group waits for its PV + next QK^T.
constexpr bool G3 = R == 3;
constexpr int NGR = G3 ? 3 : 2, RPG = R / NGR;             // softmax groups; pair rows a group (rows gi, gi + NGR, ...)
static_assert(R % NGR == 0, "R");
constexpr int QT = BM * D;                                 // Q tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][32], 64B swizzle (8 KiB)
constexpr int BT = BM * BN;                                // bias tile [2 key halves][128 q][64], 128B swizzle, bf16 (32 KiB)
#ifndef TA_S16
#define TA_S16 0         // (experiment) S accumulated in fp16 (kind::f16, f16 accumulator: q / k fp16): 64 TMEM columns a buffer, six buffers
#endif
#ifndef TA_H16T
#define TA_H16T 0        // (experiment, timing only, WRONG results; needs TA_S16, TA_PH): packed fp16 softmax loop
#endif
#ifndef TA_NB
#define TA_NB 2          // bias tile ring stages
#endif
constexpr int NKV = TA_NKV, NB = TA_NB;
constexpr int THREADS = G3 ? 480 : 448 + 32 * TA_SPLIT;   // 0 TMA, 1 MMA (QK^T), 2-5 / 6-9 softmax groups (even / odd rows), 10-13 epilogue (R = 3: the third group), 14 MMA (PV)
static_assert(!G3 || TA_SPLIT, "R = 3 needs the two issue warps");
constexpr int ONT = TA_ONES ? 16 * D : 0;                 // (TA_ONES) the ones tile: B [16 rows][32] K-major 64B-swizzled, all 1.0 (1 KiB)
constexpr int SMEM = 1024 + (2 * R * QT + NKV * 2 * KVT + NB * BT + ONT) * 2 + 2 * 2 * R * BM * 4 + 2 * R * BM * 4 + 512;   // + the row sums / offsets for the epilogue
constexpr int NS = TA_S16 ? 6 : 3;                                      // S/P TMEM buffers: with two, a group's buffer sat idle through PV + QK^T
constexpr int BNS = TA_S16 ? BN / 2 : BN;                  // TMEM columns of one S / P buffer
constexpr int COL_S = 0, COL_O = NS * BNS;                  // S/P [NS][128] | O [R rows][32] (single: the next task's first PV waits the drain)
constexpr int COL_L = 512 - 32;                            // (TA_ONES) l_r: 16 columns a row (two rows)
constexpr bool NOSUM = TA_D16 || TA_ONES;                  // the row sum comes from the tensor core: no FADD2, no per-tile guard
static_assert(!TA_ONES || COL_O + R * D <= COL_L, "TA_ONES: TMEM");
// TA_ONES with three rows (48 sum columns do not fit beside 3 S + 3 O): a two-slot ring -- sub-step y's ones MMAs write its partial sum into
// slot y & 1 (no accumulate) and commit lfu[row] (per row: a group waits on its own commits in order -- one barrier per slot let a
// group wait two phases ahead and read a stale slot); the row's group reads it after the tile, adds it in fp32 (key-tile order) and arrives
// lem[y & 1], which the PV warp waits before the ones MMAs of y + 2.  TA_ONESX: the timing ceiling without the ring (WRONG sums).
constexpr bool ORING = TA_ONES && G3 && !TA_ONESX;
__host__ __device__ constexpr int lslot(int r) { return r & 1; }
constexpr uint32_t ID_S = TA_S16 ? ((uint32_t)(BN >> 3) << 17 | (uint32_t)(128 >> 4) << 24) : idesc_bf16(128, BN, 0, 0);       // A = Q K-major, B = K K-major
constexpr uint32_t ID_O = TA_PH ? (idesc_bf16(128, D, 0, 1) & ~((1u << 7) | (1u << 10))) : idesc_bf16(128, D, 0, 1);   // A = P from TMEM, B = V MN-major
constexpr uint32_t ID_L = idesc_bf16(128, 16, 0, 0);       // (TA_ONES) A = P from TMEM, B = the ones tile K-major
constexpr float L2E = 1.4426950408889634f;
constexpr int CK = TA_CK, NCK = BN / CK;                   // softmax chunk: keys, chunks a tile
static_assert(CK == 16 || CK == 32, "CK");
static_assert(!TA_S16 || (CK == 16 && TA_HB && !TA_D16), "TA_S16: the 16-key chunk HB build of 32-channel heads");
static_assert(SMEM <= 232448, "one CTA per SM");

// TA_S16: eight 32-bit columns = 16 fp16 scores, converted in place (descending: word k -> floats 2k, 2k + 1)
__device__ __forceinline__ void cv16(float* a) {
#pragma unroll
  for (int k = 7; k >= 0; --k) { const uint32_t w = __float_as_uint(a[k]); a[2 * k] = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu))); a[2 * k + 1] = __half2float(__ushort_as_half((unsigned short)(w >> 16))); }
}
__device__ __forceinline__ void tmem_ldc(uint32_t taddr, float* v) { if constexpr (CK == 32) tmem_ld32(taddr, v); else tmem_ld16(taddr, v); }
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// fp32 + one bf16 half of a 32-bit word (sm_100 mixed-precision add: one FHADD.BF16 with a half select, no unpack)
__device__ __forceinline__ float badd_lo(uint32_t w, float x) {
  float d; asm("{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, %1;\nadd.rn.f32.bf16 %0, lo, %2;\n}\n" : "=f"(d) : "r"(w), "f"(x)); return d;
}
__device__ __forceinline__ float badd_hi(uint32_t w, float x) {
  float d; asm("{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, %1;\nadd.rn.f32.bf16 %0, hi, %2;\n}\n" : "=f"(d) : "r"(w), "f"(x)); return d;
}
// x = S * scale*log2e + bias * log2e + nm for a bf16 bias pair: two FFMA2 (+ the unpack), or (TA_HB, the bias pre-scaled) FFMA2 + two FHADD
__device__ __forceinline__ float2 logit2(uint32_t w, float s0, float s1, float2 sc2, float2 nm) {
#if TA_HB
  const float2 t = fma2(make_float2(s0, s1), sc2, nm);
  return make_float2(badd_lo(w, t.x), badd_hi(w, t.y));
#else
  return fma2(bf2f(w), make_float2(L2E, L2E), fma2(make_float2(s0, s1), sc2, nm));
#endif
}
template <bool B> struct BC { static constexpr bool value = B; };
__device__ __forceinline__ float2 logit2q(uint32_t w, float s0, float s1) { return make_float2(badd_lo(w, s0), badd_hi(w, s1)); }
#ifndef TA_POLY
#define TA_POLY 0        // forward: this many of a chunk's 16 exponential pairs as an FMA-pipe polynomial (0: all on the MUFU; with the two issue
                         // warps, L768: 4 / 8 / 12 of 16 took 700 / 834 / 900 us against 625 -- the softmax warps are bound by their own issue)
#endif
#ifndef TA_QS
#define TA_QS 0          // (HB core) q arrives pre-scaled by scale * log2 e (the host scales the q weight rows) and the offset starts at 0 (provisional): the
                         // softmax loop's logit is x = S + bias (two FHADD, no FFMA2) while every row of the warp still has offset 0; scl = 1 for the rare paths
#endif
constexpr float TA_QT = 24.f;   // TA_QS: the log2-domain band of a row's first-32-key maximum that keeps offset 0
#ifndef TA_PSUB
#define TA_PSUB 1        // (HB core) the polynomial exponential's x - round(x) as one FFMA2 instead of two negating FADDs + FADD2
#endif
#ifndef TA_NOKV
#define TA_NOKV 0        // (experiment) K | V loaded only into the first ring stages
#endif
#ifndef TA_NOBIAS
#define TA_NOBIAS 0      // (experiment) bias tiles loaded only into the first ring stages
#endif
#ifndef TA_NOMMA
#define TA_NOMMA 0       // (experiment) no QK^T / PV MMAs (the commits only)
#endif
#ifndef TA_FNOMATH
#define TA_FNOMATH 0     // (experiment) neither logits nor exponentials: the softmax warps only move S / P
#endif
#ifndef TA_FNOLD
#define TA_FNOLD 0      // (experiment) the softmax warps do not load S from TMEM
#endif
#ifndef TA_FNOST
#define TA_FNOST 0      // (experiment) the softmax warps do not store P to TMEM
#endif
#ifndef TA_FNOLOOP
#define TA_FNOLOOP 0    // (experiment) the softmax warps do no arithmetic at all
#endif
#ifndef TA_NOEX
#define TA_NOEX 0        // (experiment) no exponential: the MUFU-free ceiling of the forward
#endif
// forward: a tile whose exponentials sum past TA_PMAX raises the row's offset (so p, l_r and O_r stay ~2^64 at most: far from
// overflow); under TA_PMIN a provisional offset (every key so far masked) takes the tile maximum
constexpr float TA_PMAX = 0x1p64f, TA_PMIN = 0x1p-64f;
#ifndef TB_POLY
#define TB_POLY 0        // the same split in both backward kernels (each recomputes P: one ex2 per score)
#endif
// 2^x on the FMA pipe, two at a time: n = round(x) by the 1.5 * 2^23 add, 2^f (|f| <= 1/2) by a degree-3 minimax polynomial
// (relative error 7.5e-5, well under P's bf16 rounding), n added into the exponent bits: (bits(t) << 23) == n << 23 exactly.
__device__ __forceinline__ float2 ex2_poly2(float2 x) {
  x.x = fmaxf(x.x, -126.f); x.y = fmaxf(x.y, -126.f);
  const float2 t = add2(x, make_float2(12582912.f, 12582912.f));
  const float2 rr = add2(t, make_float2(-12582912.f, -12582912.f));
#if TA_PSUB
  const float2 f = fma2(rr, make_float2(-1.f, -1.f), x);   // x - rr in one FFMA2 (the -rr negations were two FADDs of their own); the same bits
#else
  const float2 f = add2(x, make_float2(-rr.x, -rr.y));
#endif
  float2 pp = fma2(f, make_float2(0.05517097723f, 0.05517097723f), make_float2(0.24260970271f, 0.24260970271f));
  pp = fma2(f, pp, make_float2(0.69326096627f, 0.69326096627f));
  pp = fma2(f, pp, make_float2(0.99992816244f, 0.99992816244f));
  return make_float2(__int_as_float(__float_as_int(pp.x) + (__float_as_int(t.x) << 23)),
                     __int_as_float(__float_as_int(pp.y) + (__float_as_int(t.y) << 23)));
}


__device__ __forceinline__ void tmem_st1(uint32_t taddr, uint32_t v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n" :: "r"(taddr), "r"(v) : "memory");
}
__device__ __forceinline__ void tmem_st4(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
__device__ __forceinline__ void rwait(uint64_t* b, uint32_t par, int tag, int y) {
  wait(b, par);
}
__device__ __forceinline__ float tmem_ld1(uint32_t taddr) {
  uint32_t r; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(r) : "r"(taddr) : "memory"); return __uint_as_float(r);
}
__device__ __forceinline__ uint32_t pack2p(float a, float b) {   // P's pair: bf16 (fp16 under the TA_PH experiment)
#if TA_PH
  uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r;
#else
  return pack2(a, b);
#endif
}

#define FT0
#define FT(i)
#define FTW(base)

// TA_D16: a row whose PV-summed l left [2^-100, 3e38) (or whose O is not finite) -- a logit ~120+ log2 units above the offset (the first
// 32 keys' maximum), or a provisional offset over a row whose first 32 keys are masked -- is recomputed here exactly: two passes over
// the keys from q / k / v / the pre-scaled bias in global memory (CUDA cores; rare, slow), fp32 throughout.  -> ob (bf16 pairs, channels
// 16 .. 31 zero)
// (TA_ONES: the same for 32-channel heads, C = 32)
template <int C>
__device__ __noinline__ void d16_exact(const __nv_bfloat16* __restrict__ qp, const __nv_bfloat16* __restrict__ kp,
                                      const __nv_bfloat16* __restrict__ vp, const __nv_bfloat16* __restrict__ bp, int LD, int S, float scl,
                                      uint32_t* ob) {
  float qv[C];
#pragma unroll
  for (int c = 0; c < C; ++c) qv[c] = __bfloat162float(qp[c]);
  auto logit = [&](int kk) {
    const __nv_bfloat16* kr = kp + (size_t)kk * LD;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) s = fmaf(qv[c], __bfloat162float(kr[c]), s);
    return s * scl + __bfloat162float(bp[kk]);
  };
  float m = -INFINITY;
  for (int kk = 0; kk < S; ++kk) m = fmaxf(m, logit(kk));
  float l = 0.f, o[C];
#pragma unroll
  for (int c = 0; c < C; ++c) o[c] = 0.f;
  if (m > -INFINITY) {
    for (int kk = 0; kk < S; ++kk) {
      const float p = exp2f(logit(kk) - m);
      const __nv_bfloat16* vr = vp + (size_t)kk * LD;
      l += p;
#pragma unroll
      for (int c = 0; c < C; ++c) o[c] = fmaf(p, __bfloat162float(vr[c]), o[c]);
    }
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
  for (int k2 = 0; k2 < 16; ++k2) ob[k2] = k2 < C / 2 ? pack2(o[2 * k2] * inv, o[2 * k2 + 1] * inv) : 0u;
}
constexpr int CX = TA_D16 ? 16 : 32;   // channels of the exact recompute / the drain's finiteness check

__global__ void __launch_bounds__(THREADS, 1) triattn_fwd_sm100(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][H*32] bf16 (the projection's layout), box (32, 128) at column h*32, 64B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*32]
    float* __restrict__ LSE,                             // [B, N, H, S] base-2 log-sum-exp for the backward, or nullptr
    int* __restrict__ FLAGS,
    const __nv_bfloat16* __restrict__ QG,                // q / k [B*N*S rows at stride LD][H*32]: the overflow guard's recompute
    const __nv_bfloat16* __restrict__ KG, int LD,
    const __nv_bfloat16* __restrict__ VG,                // (TA_D16) v at the stride LD and the bias [B*H*S][S]: the exact recompute of a bad row
    const __nv_bfloat16* __restrict__ BG) {
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 tasks][R][QT]
  __nv_bfloat16* sKV = sQ + 2 * R * QT;                          // [NKV][K | V]
  __nv_bfloat16* sB = sKV + NKV * 2 * KVT;                       // [NB][BT]
  __nv_bfloat16* sOnes = sB + NB * BT;                           // (TA_ONES) [16][32] ones (1 KiB, 1024-aligned)
  float* sL = reinterpret_cast<float*>(sOnes + ONT);             // [2 tasks][R][128] row sums
  float* sM = sL + 2 * R * BM;                                   // [2 tasks][R][128] row offsets
  float* sRM = sM + 2 * R * BM;                                  // [R][128] the softmax rows' offsets
  float* sRL = sRM + R * BM;                                     // [R][128] their running sums; -1: the offset is provisional
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRL + R * BM);
  uint64_t* qf = bars;               // [2]   the task's Q tiles landed
  uint64_t* qe = qf + 2;             // [2]   the task's last QK^T retired
  uint64_t* kvf = qe + 2;            // [NKV]
  uint64_t* kve = kvf + NKV;         // [NKV] the PV GEMM of the stage retired
  uint64_t* bf = kve + NKV;          // [NB]
  uint64_t* be = bf + NB;            // [NB]  count 8: both softmax groups are done with the bias tile
  uint64_t* sf = be + NB;            // [NS]  S of the sub-step ready
  uint64_t* pf = sf + NS;            // [NS]  count 4: P written (over S)
  uint64_t* of = pf + NS;            // [1]   the task's O complete
  uint64_t* oe = of + 1;             // [1]   count 4: the epilogue drained O
  uint64_t* lf = oe + 1;             // [2]   count 8: both groups wrote the task's row sums
  uint64_t* rof = lf + 2;            // [R]   (R = 3) the row's O complete: its group drains it
  uint64_t* lfu = rof + R;           // [R]   (ORING) the row's partial sum written (per row: a group waits its own commits in order)
  uint64_t* lem = lfu + R;           // [2]   count 4: the slot was read
  uint64_t* pvi = lem + 2;           // [NKV] (TA_PVI) the PV MMAs of the stage were issued
  uint32_t* tslot = reinterpret_cast<uint32_t*>(pvi + NKV);
  FT0
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < 2; ++k) { bar_init(qf + k, 1); bar_init(qe + k, 1); }
    for (int k = 0; k < NS; ++k) { bar_init(sf + k, 1); bar_init(pf + k, 4); }
    bar_init(of, 1); bar_init(oe, 4); bar_init(lf, 8); bar_init(lf + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(kvf + k, 1); bar_init(kve + k, 1); bar_init(pvi + k, 1); }
    for (int k = 0; k < NB; ++k) { bar_init(bf + k, 1); bar_init(be + k, 4 * NGR); }
    for (int k = 0; k < R; ++k) bar_init(rof + k, 1);
    for (int k = 0; k < R; ++k) bar_init(lfu + k, 1);
    for (int k = 0; k < 2; ++k) bar_init(lem + k, 4);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  if constexpr (TA_ONES) {
    for (int i = tid; i < ONT / 2; i += THREADS) reinterpret_cast<uint32_t*>(sOnes)[i] = 0x3f803f80u;
    fence_proxy_async();                         // the MMAs (async proxy) read the ones
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  // PDL (inference chain): set-up above overlaps the front's last tiles; q / k / v / bias and every store come after the wait
  pdl_wait();
  pdl_trigger();
  // task t: q tile fastest (the CTAs sharing a row group's K / V run together), then the row group, then (b, h)
  auto decode = [&](int t, int& bh, int& qt, int& g) { qt = t % QTILES; const int r = t / QTILES; g = r % NG; bh = r / NG; };
  auto row0 = [&](int bh, int n) -> int { const int b = bh / H; n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row of q / k / v / o
  auto hcol = [&](int bh) -> int { return (bh % H) * DG; };   // the head's first channel in global q / k / v / out
  // a [128 rows][32] 64B-swizzled tile of head bh (TA_D16C: the 3-D map's box zero-fills channels 16 .. 31)
  auto ldt = [&](const CUtensorMap* m, void* dst, uint64_t* bar, int bh, int rw) {
    if (TA_D16C) load_3d(m, dst, bar, 0, bh % H, rw); else load_2d(m, dst, bar, hcol(bh), rw);
  };

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, x = 0, jb = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int bh, qt, g; decode(t, bh, qt, g);
        const int nr = R, qb = lt & 1;
        FT(4); if (lt >= 2) wait(qe + qb, ((lt >> 1) - 1) & 1); FT(0);
        expect_tx(qf + qb, nr * QT * 2);
        for (int r = 0; r < nr; ++r) ldt(&qmap, sQ + (qb * R + r) * QT, qf + qb, bh, row0(bh, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j, ++jb) {
          const int bs = jb % NB;
          FT(4); if (jb >= NB) wait(be + bs, ((jb / NB) - 1) & 1); FT(1);
          if (TA_NOBIAS && jb >= NB) arrive(bf + bs); else {
          expect_tx(bf + bs, BT * 2);
          load_3d(&bmap, sB + bs * BT, bf + bs, 0, bh * S + qt * BM, j * 2); }   // both 64-key halves in one box (3-D map)
          for (int r = 0; r < nr; ++r, ++x) {
            const int st = x % NKV;
            FT(4); if (x >= NKV) wait(kve + st, ((x / NKV) - 1) & 1); FT(2);
            if (TA_NOKV && x >= NKV) { arrive(kvf + st); continue; }
            expect_tx(kvf + st, 2 * KVT * 2);
            const int rw = row0(bh, g * R + r) + j * BN;
            ldt(&kmap, sKV + st * 2 * KVT, kvf + st, bh, rw);
            ldt(&vmap, sKV + st * 2 * KVT + KVT, kvf + st, bh, rw);
          }
        }
      }
    }
  } else if (warp == 1 || warp == 14) {
    // flattened sub-steps y (task k = y / SPT, key tile j, row r): QK^T runs up to NS - 1 sub-steps ahead of the PV GEMMs, so a
    // softmax group finds its next S ready when it hands over P (issued strictly one ahead, each group idled ~1K cycles per sub-step).
    // B200 2026-11-17 (clock64 buckets of a CTA, L768): one warp issuing both streams was the kernel's critical path -- ~1500 clk of
    // issue a sub-step (ten MMAs, two commits, the index divisions) against ~1150 clk of softmax work, the softmax groups waiting on
    // S a quarter of the time.  Now the indices are carried (no divisions: 661 -> 638 us) and, TA_SPLIT, warp 14 issues the PV
    // stream while warp 1 issues QK^T(y) once PV(y - NS) has retired (the K / V stage barrier kve: the two streams are ordered by
    // barriers, not by one thread's issue order; a commit tracks the issuing thread's own MMAs): 638 -> 625 us, the softmax groups
    // wait ~5 %.  The same bits.
    const int SPT = NK * R;
    const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int total = mytasks * SPT;
    // the sub-step indices of an issue stream, carried
    struct Ctr {
      int k = 0, w = 0, j = 0, r = 0, st = 0, b = 0, stp = 0, bp = 0;
      __device__ __forceinline__ void next(int spt) {
        if (++r == R) { r = 0; ++j; }
        if (++w == spt) { w = 0; j = 0; ++k; }
        if (++st == NKV) { st = 0; stp ^= 1; }
        if (++b == NS) { b = 0; bp ^= 1; }
      }
    } cq, cp;
    auto qk = [&]() {
      const int k = cq.k, w = cq.w, j = cq.j, r = cq.r, st = cq.st, b = cq.b, qb = k & 1, stp = cq.stp;
      cq.next(SPT);
      FT(4);
      if (w == 0) wait(qf + qb, (k >> 1) & 1);
      FT(3);
      wait(kvf + st, stp);
      FT(0);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* q = sQ + (qb * R + r) * QT;
        const __nv_bfloat16* kk = sKV + st * 2 * KVT;
        if (!TA_NOMMA)
#pragma unroll
        for (int ks = 0; ks < (TA_D16 ? 1 : D / 16); ++ks)
          mma_ss(tmem + COL_S + b * BNS, desc_k64(q + ks * 16), desc_k64(kk + ks * 16), ID_S, ks ? 1u : 0u);
        mma_commit(sf + b);
        if (w == SPT - 1) mma_commit(qe + qb);
      }
      __syncwarp();
      (void)j;
    };
    int yl = 0;                                  // (ORING) the PV stream's sub-step
    auto pv = [&]() {                            // O_r += P . V
      const int k = cp.k, w = cp.w, j = cp.j, r = cp.r, st = cp.st, b = cp.b, bp = cp.bp;
#if TA_D16C
      {                                          // v's channel 16 = 1.0 in the landed V tile (the TMA zero-filled 16 .. 31): l = P . ones
        const int stp = cp.stp;
        wait(kvf + st, stp);
        unsigned char* vb = reinterpret_cast<unsigned char*>(sKV + st * 2 * KVT + KVT);
#pragma unroll
        for (int i = 0; i < BN / 32; ++i) {
          const int rr = lane + 32 * i;        // 64B swizzle: 16-byte chunk 2 (channels 16 .. 23) of row rr at chunk 2 ^ ((rr >> 1) & 3)
          *reinterpret_cast<uint16_t*>(vb + rr * 64 + ((2 ^ ((rr >> 1) & 3)) << 4)) = 0x3f80u;
        }
        fence_proxy_async();
        __syncwarp();
      }
#endif
      cp.next(SPT);
      FT(4);
      wait(pf + b, bp);
      FT(1);
      if (!G3 && w == 0 && k >= 1) wait(oe, (k - 1) & 1);   // the previous task's O was drained (R = 3: the row's group arrives pf after its drain)
      FT(2);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* v = sKV + st * 2 * KVT + KVT;
        const uint32_t o = tmem + COL_O + r * D;
        if (!TA_NOMMA)
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks)
          mma_ts(o, tmem + COL_S + b * BNS + ks * 8, sdesc(sa(v + ks * 16 * D), KVT * 2, 512, 4), ID_O, (j | ks) ? 1u : 0u);
        if (ORING && yl >= 2) { rwait(lem + (yl & 1), ((yl >> 1) - 1) & 1, 2, yl); tc_fence_after(); }   // the slot's y - 2 sum was read
        if (TA_ONES && !TA_NOMMA)                // l_r += P . ones (N = 16: every column the row sum)
#pragma unroll
          for (int ks = 0; ks < BN / 16; ++ks)
            mma_ts(tmem + COL_L + (ORING ? (yl & 1) : lslot(r)) * 16, tmem + COL_S + b * BNS + ks * 8, desc_k64(sOnes), ID_L, ((ORING ? 0 : j) | ks) ? 1u : 0u);
        if (TA_PVI) { tc_fence_before(); arrive(pvi + st); }   // the stage's PV MMAs are issued: QK^T(y + NS) may follow in pipe order
        if (ORING) mma_commit(lfu + r);
        mma_commit(kve + st);
        if (G3) { if (w >= SPT - R) mma_commit(rof + r); }
        else if (w == SPT - 1) mma_commit(of);
      }
      __syncwarp();
      ++yl;
    };
#if TA_SPLIT
    if (warp == 1) {                             // S buffer y % NS is free once PV(y - NS) retired (its K / V stage barrier)
      Ctr cw;
      for (int y = 0; y < total; ++y) {
        if (y >= NS) { FT(4); if (TA_PVI) { wait(pvi + cw.st, cw.stp); tc_fence_after(); } else wait(kve + cw.st, cw.stp); FT(5); cw.next(SPT); }
        qk();
      }
    } else {
      for (int p = 0; p < total; ++p) pv();
    }
#else
    int y = 0;
    for (int p = 0; p < total; ++p) {
      for (; y < total && y < p + NS; ++y) qk();   // S buffer y % NS is free once PV(y - NS) is issued (the tensor pipe runs in order)
      pv();
    }
#endif
  } else if (warp < 2 + 4 * NGR) {
    const int q = warp & 3, row = q * 32 + lane;   // TMEM lane = query row of the tile
    const int gi = (warp - 2) >> 2;                // this group's rows: gi, gi + NGR, ...
    int lt = 0, x = 0, jb = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      const int ob = lt & 1;
      float lacc = 0.f;                          // (ORING) the row's sum, from the ring's partials
      // the rows' offset and sum (-1: provisional offset) live in shared memory, each thread its own slots: the rare overflow
      // branch below then has no register results, and the steady-state loop keeps its schedule (with them in registers: +3 %)
#pragma unroll
      for (int rr = 0; rr < RPG; ++rr) { const int o = (gi + NGR * rr) * BM + row; sRM[o] = 0.f; sRL[o] = 0.f; }
      for (int j = 0; j < NK; ++j, ++jb) {
        const int bs = jb % NB;
        FT(4);
        wait(bf + bs, (jb / NB) & 1);
        FT(0);
        const __nv_bfloat16* brow = sB + bs * BT + row * 64;
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
          const int r = gi + NGR * rr, xs = x + r, b = xs % NS;
          FT(4);
          wait(sf + b, (xs / NS) & 1);
          FT(1);
          tc_fence_after();
          const uint32_t sb = tmem_at(tmem + COL_S + b * BNS, q * 32, 0);
          float* const mslot = sRM + r * BM + row;
          float* const lslot = sRL + r * BM + row;
          float mr = *mslot, lr = 0.f;
          bool rare;
          const float2 sc2 = make_float2(scl, scl);
          // x = S * scale*log2e + bias * log2e for chunk c (32 keys), bias from the staged tile
          auto logits = [&](int c, float* s32, float2 nm0) {   // x = S * scale*log2e + bias * log2e + nm0 (keys CK c ..)
            const int k0 = c * CK;
            const __nv_bfloat16* bp = brow + (k0 >> 6) * BM * 64;
#pragma unroll
            for (int e = 0; e < CK / 8; ++e) {
              const uint4 u = *reinterpret_cast<const uint4*>(bp + (((((k0 & 63) >> 3) + e) ^ (row & 7)) << 3));
              const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
              for (int k2 = 0; k2 < 4; ++k2) {
                const int i2 = e * 8 + k2 * 2;
                const float2 xx = logit2(w[k2], s32[i2], s32[i2 + 1], sc2, nm0);
                s32[i2] = xx.x; s32[i2 + 1] = xx.y;
              }
            }
          };
          auto logitsq = [&](int c, float* s32) {   // TA_QS: x = S + bias for chunk c (offset 0)
            const int k0 = c * CK;
            const __nv_bfloat16* bp = brow + (k0 >> 6) * BM * 64;
#pragma unroll
            for (int e = 0; e < CK / 8; ++e) {
              const uint4 u = *reinterpret_cast<const uint4*>(bp + (((((k0 & 63) >> 3) + e) ^ (row & 7)) << 3));
              const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
              for (int k2 = 0; k2 < 4; ++k2) {
                const int i2 = e * 8 + k2 * 2;
                const float2 xx = logit2q(w[k2], s32[i2], s32[i2 + 1]);
                s32[i2] = xx.x; s32[i2 + 1] = xx.y;
              }
            }
          };
          if (TA_H16T) {   // (experiment, timing only, WRONG results) the packed fp16 loop: HFMA2 logits, HSUB2 offset, ex2.f16x2, HADD2 sum, P as fp16 pairs
            const __half2 sch = __float2half2_rn(0.01f), mh = __float2half2_rn(*mslot);
            __half2 acc2 = __float2half2_rn(0.f);
            uint32_t sw[2][8];
            tmem_ld8(sb, reinterpret_cast<float*>(sw[0]));
            tmem_wait_ld();
#pragma unroll
            for (int c = 0; c < NCK; ++c) {
              if (c + 1 < NCK) tmem_ld8(sb + (c + 1) * 8, reinterpret_cast<float*>(sw[(c + 1) & 1]));
              const int k0 = c * CK;
              const __nv_bfloat16* bp = brow + (k0 >> 6) * BM * 64;
              uint32_t bw[8];
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const uint4 u = *reinterpret_cast<const uint4*>(bp + (((((k0 & 63) >> 3) + e) ^ (row & 7)) << 3));
                bw[e * 4] = u.x; bw[e * 4 + 1] = u.y; bw[e * 4 + 2] = u.z; bw[e * 4 + 3] = u.w;
              }
              uint32_t pk[8];
#pragma unroll
              for (int k2 = 0; k2 < 8; ++k2) {
                __half2 h = __hfma2(*reinterpret_cast<const __half2*>(&sw[c & 1][k2]), sch, *reinterpret_cast<const __half2*>(&bw[k2]));
                h = __hsub2(h, mh);
                uint32_t hb = *reinterpret_cast<uint32_t*>(&h), pb;
                asm("ex2.approx.f16x2 %0, %1;" : "=r"(pb) : "r"(hb));
                acc2 = __hadd2(acc2, *reinterpret_cast<__half2*>(&pb));
                pk[k2] = pb;
              }
              tmem_wait_ld();
#pragma unroll
              for (int s8 = 0; s8 < 1; ++s8) tmem_st8(sb + c * 8 + s8 * 8, pk + s8 * 8);
            }
            lr = __half2float(acc2.x) + __half2float(acc2.y);
          } else
          {
          auto chunk_loop = [&](auto fast_t) {
            constexpr bool FAST = decltype(fast_t)::value;
            // chunks of CK keys, the next chunk's TMEM load in flight while this one is worked (CK 16 halves the S registers; the
            // offset still comes from the first 32 keys and l_r sums the pairs in key order: the same bits for any CK)
            float2 acc = make_float2(0.f, 0.f);
            float s32[2][CK];
            if (TA_FNOLD) { for (int i2 = 0; i2 < CK; ++i2) s32[0][i2] = s32[1][i2] = 0.f; } else
            if (TA_S16) tmem_ld8(sb, s32[0]); else tmem_ldc(sb, s32[0]);
            tmem_wait_ld();
            if (TA_S16) cv16(s32[0]);
#pragma unroll
            for (int c = 0; c < NCK; ++c) {
              if (c + 1 < NCK && !TA_FNOLD) { if (TA_S16) tmem_ld8(sb + (c + 1) * (CK / 2), s32[(c + 1) & 1]); else tmem_ldc(sb + (c + 1) * CK, s32[(c + 1) & 1]); }
              float* sc = s32[c & 1];
              if (j == 0 && c == 0) {            // the row's offset: the first 32 keys' maximum (raised after a tile, if need be)
                if constexpr (CK == 16) tmem_wait_ld();   // keys 16 .. 31 (chunk 1) too
                if (TA_S16) cv16(s32[1]);
                logits(0, s32[0], make_float2(0.f, 0.f));
                if constexpr (CK == 16) logits(1, s32[1], make_float2(0.f, 0.f));
                float m0 = s32[0][0], m1 = s32[0][1];
#pragma unroll
                for (int i2 = 2; i2 < 32; i2 += 2) { m0 = fmaxf(m0, s32[i2 / CK][i2 % CK]); m1 = fmaxf(m1, s32[i2 / CK][i2 % CK + 1]); }
                mr = fmaxf(m0, m1);
                if (!(mr > -1e30f)) { mr = 0.f; *lslot = -1.f; }   // the first 32 keys all masked: -inf - -inf would be NaN
                if (TA_QS && mr < TA_QT && mr > -TA_QT) mr = 0.f;   // TA_QS: a row whose first 32 keys stay inside +-TA_QT keeps offset 0 (the later tiles then take the FHADD-only loop)
                *mslot = mr;
#pragma unroll
                for (int i2 = 0; i2 < 32; ++i2) s32[i2 / CK][i2 % CK] -= mr;
              } else if (CK == 16 && j == 0 && c == 1) {
                // (chunk 1's logits were made with chunk 0's)
              } else if (!TA_FNOMATH) {
                if constexpr (FAST) logitsq(c, sc); else logits(c, sc, make_float2(-mr, -mr));
              }
              uint32_t pk[CK / 2];
#pragma unroll
              for (int k2 = 0; k2 < CK / 2; ++k2) {
                const float2 xx = make_float2(sc[k2 * 2], sc[k2 * 2 + 1]);
                float2 pv;
                if (TA_FNOLOOP) pv = make_float2(1.f, 1.f); else
                if (TA_POLY && ((k2 * TA_POLY) / 16 != ((k2 + 1) * TA_POLY) / 16)) pv = ex2_poly2(xx);   // TA_POLY of 16 pairs, spread
                else if (TA_NOEX || TA_FNOMATH) pv = fma2(xx, make_float2(1e-3f, 1e-3f), make_float2(1.f, 1.f));
                else pv = make_float2(ex2f(xx.x), ex2f(xx.y));
                if (TA_FNOLOOP) { pk[k2] = 0x3f803f80u; continue; }
                if (!NOSUM) acc = add2(acc, pv);
                pk[k2] = pack2p(pv.x, pv.y);
              }
              tmem_wait_ld();                    // chunk c+1 landed (and every S column chunk c's P covers was read)
              if (TA_S16 && c + 1 < NCK && !(j == 0 && c == 0)) cv16(s32[(c + 1) & 1]);
              if (!TA_FNOST) {
#pragma unroll
              for (int s8 = 0; s8 < CK / 16; ++s8) tmem_st8(sb + c * (CK / 2) + s8 * 8, pk + s8 * 8); }   // P over S: columns already read
            }
            lr = TA_FNOLOOP ? 128.f : acc.x + acc.y;
          };
          if constexpr (TA_QS) { if (j > 0 && __all_sync(0xffffffffu, mr == 0.f)) chunk_loop(BC<true>{}); else chunk_loop(BC<false>{}); }
          else chunk_loop(BC<false>{});
          }
          {
            // Overflow guard, after the chunk loop (a running maximum or a branch inside the loop cost 4-15 %: the loop is one
            // schedule).  A tile whose sum passed TA_PMAX (or is inf / NaN) has its raw S recomputed from q / k in global memory
            // (the first half of the S buffer is already P, which may hold inf), the offset raised to the tile maximum, P
            // rewritten and l_r, O_r rescaled.  A row whose first 32 keys were all masked has the provisional offset 0 (l slot
            // -1) and takes the tile maximum once a tile under it sums below TA_PMIN with a finite bias (underflow, not a mask).
            // Measured on the steady state (the branch is never taken there), each of these cost 1-3 %: a 32-wide body (it spilled
            // loop invariants: the body works 8 columns at a time in loops not unrolled), the branch's results in registers (the
            // row state is in shared memory instead), a shared-memory load before the vote (it reads only this tile's sum; a sum
            // under TA_PMIN -- a masked tile, keys far below the offset -- enters too and mostly leaves at the bias check).
            rare = NOSUM ? false : __any_sync(0xffffffffu, !(lr <= TA_PMAX) || !(lr >= TA_PMIN));
            if (rare) {                                             // rare, warp-uniform (tcgen05.ld / st are warp-collective)
              const float l0 = *lslot;
              const bool prov = l0 < 0.f;
              const float lo = fmaxf(l0, 0.f);
              const bool hi = !(lr <= TA_PMAX);                     // (also inf / NaN)
              bool need = hi || (prov && !(lr >= TA_PMIN));         // (prov && !need: a sum in range, offset 0 is as good as any)
              if (!hi && need) {                                    // provisional: is any key of this tile unmasked for the row?
                bool fin = false;
#pragma unroll 1
                for (int e = 0; e < 16; ++e) {
                  const uint4 u = *reinterpret_cast<const uint4*>(brow + (e >> 3) * BM * 64 + (e & 7) * 8);
                  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                  for (int k2 = 0; k2 < 4; ++k2) fin |= (w[k2] & 0xffffu) != 0xff80u || (w[k2] >> 16) != 0xff80u;
                }
                need = fin;
              }
              if (!__any_sync(0xffffffffu, need)) {
                *lslot = prov && !(lr >= TA_PMIN) ? l0 : lo + lr;   // a masked tile keeps a provisional offset
              } else {
                float f = 1.f;
#if TA_PSHIFT
                // B200 2026-11-17: a sum past TA_PMAX that is still finite (< 2^127: then every p <= the sum is finite too) raises the
                // offset by the sum's binary exponent e instead of recomputing S: P (bf16, in TMEM) is scaled by 2^-e, which is exact,
                // so P is the same as with the offset m + e from the start; l_r, O_r by 2^-e below.  The recompute from q / k (a
                // 128-key fp32 dot loop a row) took a wide-spread case 11x the steady state (d_pair 256 L128: core 14 -> 158 us).
                if (__all_sync(0xffffffffu, !need || (hi && lr < 0x1p127f))) {
                  const int e = need ? (int)((__float_as_uint(lr) >> 23) & 0xffu) - 127 : 0;
                  const float sc = __uint_as_float((uint32_t)(127 - e) << 23);   // 2^-e exactly (1 for a row that needs nothing)
                  tmem_wait_st();
#pragma unroll 1
                  for (int i = 0; i < BN / 16; ++i) {               // P of keys 16 i .. 16 i + 15: packed columns 8 i .. 8 i + 7
                    float t[8];
                    tmem_ld8(sb + i * 8, t);
                    tmem_wait_ld();
                    uint32_t pk[8];
#pragma unroll
                    for (int k2 = 0; k2 < 8; ++k2) {
                      const float2 pv = bf2f(__float_as_uint(t[k2]));
                      pk[k2] = pack2(pv.x * sc, pv.y * sc);
                    }
                    tmem_st8(sb + i * 8, pk);
                  }
                  if (need) {
                    f = prov ? 0.f : sc;                            // provisional: everything summed so far is exactly 0
                    *lslot = lo * f + lr * sc;
                    *mslot = mr + (float)e;
                  } else {
                    *lslot = prov && !(lr >= TA_PMIN) ? l0 : lo + lr;
                  }
                } else
#endif
                {
                int bh, qt, g; decode(t, bh, qt, g);              // (not kept live through the softmax loop: registers)
                tmem_wait_st();
                const uint4* qp = reinterpret_cast<const uint4*>(QG + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (size_t)LD + hcol(bh));
                const __nv_bfloat16* kp = KG + ((size_t)row0(bh, g * R + r) + j * BN) * (size_t)LD + hcol(bh);
                const uint4 q0 = qp[0], q1 = qp[1], q2 = qp[2], q3 = qp[3];
                const uint32_t qw[16] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
#pragma unroll 1
                for (int kk = 0; kk < BN; ++kk) {                   // raw S = q . k (fp32) over the S buffer, a key at a time
                  const uint4* kr = reinterpret_cast<const uint4*>(kp + (size_t)kk * LD);
                  float2 a = make_float2(0.f, 0.f);
#pragma unroll
                  for (int e = 0; e < 4; ++e) {
                    const uint4 u = kr[e];
                    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) a = fma2(bf2f(w[k2]), bf2f(qw[e * 4 + k2]), a);
                  }
                  tmem_st1(sb + kk, __float_as_uint(a.x + a.y));
                }
                tmem_wait_st();
                // x = S * scale*log2e + bias * log2e + nm for keys 8 i .. 8 i + 7
                auto logits8 = [&](int i, float* t, float nm) {
                  const uint4 u = *reinterpret_cast<const uint4*>(brow + (i >> 3) * BM * 64 + (((i & 7) ^ (row & 7)) << 3));
                  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                  for (int k2 = 0; k2 < 4; ++k2) {
                    const float2 xx = logit2(w[k2], t[2 * k2], t[2 * k2 + 1], sc2, make_float2(nm, nm));
                    t[2 * k2] = xx.x; t[2 * k2 + 1] = xx.y;
                  }
                };
                float tm = -INFINITY;                               // the tile maximum, relative to the offset
#pragma unroll 1
                for (int i = 0; i < BN / 8; ++i) {
                  float t[8];
                  tmem_ld8(sb + i * 8, t);
                  tmem_wait_ld();
                  logits8(i, t, -mr);
#pragma unroll
                  for (int k2 = 0; k2 < 8; ++k2) tm = fmaxf(tm, t[k2]);
                }
                float mn = mr;
                if (need && tm > -1e30f) {
                  f = prov ? 0.f : ex2f(-tm);                       // provisional: everything summed so far is exactly 0
                  mn = mr + tm;
                }
                float2 a2 = make_float2(0.f, 0.f);
#pragma unroll 1
                for (int i = 0; i < BN / 8; ++i) {                  // P of keys 8 i .. covers S columns 4 i .. 4 i + 3, already read
                  float t[8];
                  tmem_ld8(sb + i * 8, t);
                  tmem_wait_ld();
                  logits8(i, t, -mn);
                  uint32_t pk[4];
#pragma unroll
                  for (int k2 = 0; k2 < 4; ++k2) {
                    const float2 pv = make_float2(ex2f(t[2 * k2]), ex2f(t[2 * k2 + 1]));
                    a2 = add2(a2, pv);
                    pk[k2] = pack2(pv.x, pv.y);
                  }
                  tmem_st4(sb + i * 4, pk);
                }
                const float a = a2.x + a2.y;
                *lslot = prov && !(a >= TA_PMIN) ? -1.f : lo * f + a;   // still provisional: every key so far masked
                *mslot = mn;
                }
                // O_r: this task's from key tile 1 on (tile 0's PV overwrote it); the PV of this tile waits for pf below, and
                // PV(previous tile, row r) = sub-step xs - R is complete when R > NS (issued before QK^T(xs), whose commit we
                // waited on), else waited on through its K/V stage barrier
                if (j > 0) {
                  if constexpr (R <= NS) {
                    const int yp = xs - R;
                    wait(kve + yp % NKV, (yp / NKV) & 1);
                    tc_fence_after();
                  }
                  const uint32_t oa = tmem_at(tmem + COL_O + r * D, q * 32, 0);
#pragma unroll 1
                  for (int i = 0; i < D / 8; ++i) {
                    float ov[8];
                    tmem_ld8(oa + i * 8, ov);
                    tmem_wait_ld();
                    uint32_t oq[8];
#pragma unroll
                    for (int k2 = 0; k2 < 8; ++k2) oq[k2] = __float_as_uint(ov[k2] * f);
                    tmem_st8(oa + i * 8, oq);
                  }
                }
              }
            }
          }
          FT(2);
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(pf + b);
          FT(3);
          if (!NOSUM && !rare) *lslot = fmaxf(*lslot, 0.f) + lr;            // (clears a provisional offset: the sum was in range)

        }
        x += R;
        __syncwarp();
        if (lane == 0) arrive(be + bs);
        if constexpr (ORING) {                   // this tile's partial row sum (sub-step y = x - R + gi) from the ring
          const int y = x - R + gi;
          rwait(lfu + gi, jb & 1, 1, y);   // (jb: this group's tile count = its waits on lfu[gi] so far)
          __syncwarp();
          tc_fence_after();
          const float lp = tmem_ld1(tmem_at(tmem + COL_L + (y & 1) * 16, q * 32, 0));
          tmem_wait_ld();
          lacc += lp;
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(lem + (y & 1));
        }
      }
      if constexpr (G3) {
        // R = 3: the group drains its own row (O_r / l_r -> bf16 -> out); the next task's first PV of the row waits for this group's
        // pf arrive after its next tile, which comes after this drain in program order (tcgen05.ld waited, fenced before that arrive)
        const int r = gi;
        FT(4);
        wait(rof + r, lt & 1);
        FT(0);
        tc_fence_after();
        float o[32];
        tmem_ld32(tmem_at(tmem + COL_O + r * D, q * 32, 0), o);
        const float mo = sRM[r * BM + row];
#if TA_D16
        tmem_wait_ld();
        const float lr = o[16];                   // P . ones (the PV GEMM's channel 16)
        const float inv = 1.f / lr;
#elif TA_ONES
        const float lr = ORING ? lacc : tmem_ld1(tmem_at(tmem + COL_L + lslot(r) * 16, q * 32, 0));   // P . ones (the ones MMAs)
        if (!ORING) tmem_wait_ld();
        const float inv = 1.f / lr;
#else
        const float lr = fmaxf(sRL[r * BM + row], 0.f);
        const float inv = 1.f / lr;
        tmem_wait_ld();
#endif
        uint32_t ob[16];
#pragma unroll
        for (int k2 = 0; k2 < 16; ++k2) ob[k2] = (TA_D16 && k2 >= 8) ? 0u : pack2(o[2 * k2] * inv, o[2 * k2 + 1] * inv);
        if (g * R + r < N) {
#if TA_D16 || TA_ONES
          float sa = 0.f;
#pragma unroll
          for (int c = 0; c < CX; ++c) sa += fabsf(o[c]);
          if (!(lr >= 0x1p-100f) || !(lr < 3.0e38f) || !(sa < 3.0e38f)) {
            const size_t qr = (size_t)row0(bh, g * R + r);
            d16_exact<CX>(QG + (qr + qt * BM + row) * LD + hcol(bh), KG + qr * LD + hcol(bh), VG + qr * LD + hcol(bh),
                      BG + ((size_t)bh * S + qt * BM + row) * S, LD, S, scl, ob);
            if (TA_ONES) atomicAdd(FLAGS, 1);   // (TA_ONES: FLAGS counts the rows recomputed)
          }
#else
          if (!(lr > 0.f) || !(lr < 3.0e38f)) atomicAdd(FLAGS, 1);
#endif
          if (LSE != nullptr) {
            const int b_ = bh / H, h_ = bh % H;
            LSE[((size_t)(b_ * N + g * R + r) * H + h_) * S + qt * BM + row] = (mo > -INFINITY ? mo : 0.f) + __log2f(lr);
          }
          uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (H * DG) + hcol(bh));
#pragma unroll
          for (int c8 = 0; c8 < DG / 8; ++c8) dst[c8] = make_uint4(ob[c8 * 4], ob[c8 * 4 + 1], ob[c8 * 4 + 2], ob[c8 * 4 + 3]);
        }
      } else {
      // hand the row sums to the epilogue warps
#pragma unroll
      for (int rr = 0; rr < R / 2; ++rr) {
        const int r = gi + 2 * rr;
        sL[((lt & 1) * R + r) * BM + row] = fmaxf(sRL[r * BM + row], 0.f); sM[((lt & 1) * R + r) * BM + row] = sRM[r * BM + row];
      }
      __syncwarp();
      if (lane == 0) arrive(lf + (lt & 1));
      }
    }
  } else if (!G3 && warp < 14) {
    // ---- epilogue warps: O_r / l_r -> bf16 -> out, overlapping the next task's softmax ----
    const int q = warp & 3, row = q * 32 + lane;
    int lt = 0;
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int bh, qt, g; decode(t, bh, qt, g);
      FT(4);
      wait(of, lt & 1);
      FT(0);
      wait(lf + (lt & 1), (lt >> 1) & 1);
      FT(1);
      tc_fence_after();
      // O_r / l_r packed to bf16 a row at a time (64 registers for the task's 4 rows, not 128 fp32: the 448-thread kernel's
      // 128-register cap made the old all-rows-in-fp32 drain spill)
      uint32_t ob[R][16];
      int bad = 0;
      int fix = 0;                               // (TA_D16) rows to recompute exactly
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float o[32];
        tmem_ld32(tmem_at(tmem + COL_O + r * D, q * 32, 0), o);
#if TA_D16
        tmem_wait_ld();
        const float lr = o[16];                   // P . ones (the PV GEMM's channel 16)
#elif TA_ONES
        const float lr = tmem_ld1(tmem_at(tmem + COL_L + lslot(r) * 16, q * 32, 0));
        tmem_wait_ld();
#else
        const float lr = sL[((lt & 1) * R + r) * BM + row];
#endif
        const float inv = 1.f / lr;
        if (g * R + r < N) bad |= !(lr > 0.f) || !(lr < 3.0e38f);
        tmem_wait_ld();
#pragma unroll
        for (int k2 = 0; k2 < 16; ++k2) ob[r][k2] = (TA_D16 && k2 >= 8) ? 0u : pack2(o[2 * k2] * inv, o[2 * k2 + 1] * inv);
#if TA_D16 || TA_ONES
        {
          float sa = 0.f;
#pragma unroll
          for (int c = 0; c < CX; ++c) sa += fabsf(o[c]);
          if (!(lr >= 0x1p-100f) || !(lr < 3.0e38f) || !(sa < 3.0e38f)) fix |= 1 << r;
        }
#endif
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(oe);                 // O is free for the next task's PV GEMMs
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (g * R + r >= N) continue;            // a padding row
#if TA_D16 || TA_ONES
        if (fix & (1 << r)) {
          const size_t qr = (size_t)row0(bh, g * R + r);
          d16_exact<CX>(QG + (qr + qt * BM + row) * LD + hcol(bh), KG + qr * LD + hcol(bh), VG + qr * LD + hcol(bh),
                    BG + ((size_t)bh * S + qt * BM + row) * S, LD, S, scl, ob[r]);
          if (TA_ONES) atomicAdd(FLAGS, 1);   // (TA_ONES: FLAGS counts the rows recomputed)
        }
#endif
        if (LSE != nullptr) {                     // base 2: lse = m + log2(l), [B, N, H, S]
          const int b_ = bh / H, h_ = bh % H;
          const float lr = sL[((lt & 1) * R + r) * BM + row];
          const float mo = sM[((lt & 1) * R + r) * BM + row];
          LSE[((size_t)(b_ * N + g * R + r) * H + h_) * S + qt * BM + row] = (mo > -INFINITY ? mo : 0.f) + __log2f(lr);
        }
        uint4* dst = reinterpret_cast<uint4*>(OUT + ((size_t)row0(bh, g * R + r) + qt * BM + row) * (H * DG) + hcol(bh));
#pragma unroll
        for (int c8 = 0; c8 < DG / 8; ++c8) dst[c8] = make_uint4(ob[r][c8 * 4], ob[r][c8 * 4 + 1], ob[r][c8 * 4 + 2], ob[r][c8 * 4 + 3]);
      }
      if (bad && !NOSUM) atomicAdd(FLAGS, 1);
    }
  }
  FT(4);
  if (warp == 0) FTW(16); else if (warp == 1) FTW(0); else if (warp == 2) FTW(8); else if (warp == 10) FTW(24); else if (warp == 6) FTW(32);
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ================================================================================================================================
// Backward.  With P = 2^(x - lse) (x the forward's base-2 logit), dP = dO . V^T, Delta = rowsum(dO o O), dS = P o (dP - Delta):
//   dV = P^T dO,  dK = scale * dS^T Q,  dQ = scale * dS K,  dbias[b,h] = sum_n dS[b,n,h].
// The H100 kernel boundaries (a KV-owned dK/dV kernel, a query-owned dQ kernel, both recomputing P), built around two sm_100 facts:
//   * tcgen05 is fed from shared memory at ~67 B/clk/SM (measured: an ss MMA with a 128 x 32 A tile costs ~65 clk even at N = 32,
//     the same MMA with A in TMEM ~30 clk), so the operand that stays fixed for a whole task -- K / V here, Q / dO in the dQ kernel --
//     is copied once into TMEM and every per-stage MMA is a ts MMA that reads only its streaming B tile from shared memory;
//   * dS never goes to HBM (N x L^2 bf16: 3.6 GB at L = 768): each kernel recomputes P from the forward's LSE.
// dbias is summed in the dQ kernel, where the query sits on the TMEM lane: a thread holds 32 consecutive keys of one query, so the
// bias tile is read with 16-byte loads and the R-row partial sums go to an L2-resident fp32 [q][k] buffer with 16-byte reductions.
#ifndef TA_TL
#define TA_TL 0
#endif
#ifndef TA_SLEEP
#define TA_SLEEP 0
#endif
#ifndef TA_BIAS1
#define TA_BIAS1 0
#endif
#ifndef TA_QD1
#define TA_QD1 0
#endif
#ifndef TA_FAKEK
#define TA_FAKEK 0
#endif
#if TA_TL          // clock64 timeline of one CTA: [event][stage]
__device__ long long g_tl[24][256];
#define TL(e, y) do { if (blockIdx.x == TA_TL && (y) < 256) g_tl[e][y] = clock64(); } while (0)
#else
#define TL(e, y) do {} while (0)
#endif

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void reduce_add_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void reduce_add_2d(const void* map, const void* src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1) : "memory");
}
// 16 packed words -> row `row` of a [rows][32] bf16 tile with the 64-byte swizzle (the TMA store's layout)
__device__ __forceinline__ void row_sw64_st(__nv_bfloat16* tile, int row, const uint32_t* w) {
  unsigned char* base = reinterpret_cast<unsigned char*>(tile) + row * 64;
#pragma unroll
  for (int c = 0; c < 4; ++c)
    *reinterpret_cast<uint4*>(base + ((c ^ ((row >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
}
// row `row` of a [rows][32] bf16 tile with the 64-byte swizzle (16-byte chunk c at c ^ ((row >> 1) & 3)) -> 16 packed words
__device__ __forceinline__ void row_sw64(const __nv_bfloat16* tile, int row, uint32_t* w) {
  const unsigned char* base = reinterpret_cast<const unsigned char*>(tile) + row * 64;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 u = *reinterpret_cast<const uint4*>(base + ((c ^ ((row >> 1) & 3)) << 4));
    w[c * 4] = u.x; w[c * 4 + 1] = u.y; w[c * 4 + 2] = u.z; w[c * 4 + 3] = u.w;
  }
}

// ---- dbias = the query-side kernel's slabs summed in slab order (deterministic): element (bh, q, k) of unit u = bh * (S / 128) + q / 128
// has n_u slabs (the CTAs that ran the unit's NG tasks, `per` consecutive tasks a CTA)
__global__ void __launch_bounds__(256) triattn_db_slabs(const float* __restrict__ SL, float* __restrict__ DB, int S, int NG, int per, long total, long n4) {
  pdl_trigger();   // round 8: PDL at S = 128 (launch_pdl_if): the next kernel may launch; this one waits for the previous grid
  pdl_wait();
  // a block = 32 float4 columns x 8 slab lanes: lane g sums slabs g, g + 8, ... (loads in flight), lane 0 adds the lane sums in lane
  // order -- a fixed order every run (B200 2026-11-17: a thread walking every slab in series was latency-bound, 7 us at L128)
  const int col = threadIdx.x & 31, lane = threadIdx.x >> 5;
  const long i = (long)blockIdx.x * 32 + col;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  const long e = i * 4, ss = (long)S * S;
  int n = 0;
  if (i < n4) {
    n = (NG + per - 1) / per;                          // every unit: one slab a part of its row groups
    int p = lane;
    for (; p + 24 < n; p += 32) {
      float4 v[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) v[k] = __ldg(reinterpret_cast<const float4*>(SL + (p + 8 * k) * total + e));
#pragma unroll
      for (int k = 0; k < 4; ++k) { a.x += v[k].x; a.y += v[k].y; a.z += v[k].z; a.w += v[k].w; }
    }
    for (; p < n; p += 8) { const float4 v = __ldg(reinterpret_cast<const float4*>(SL + p * total + e)); a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
  }
  __shared__ float4 part[8][32];
  part[lane][col] = a;
  __syncthreads();
  if (lane == 0 && i < n4) {
    float4 r = part[0][col];
#pragma unroll
    for (int g = 1; g < 8; ++g) { const float4 v = part[g][col]; r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w; }
    *reinterpret_cast<float4*>(DB + e) = r;
  }
}

// ---- S90 (training, wide path, starting-node bias orientation): the slab sum with the head backward's A block written directly: dbias bf16 into
// rows [R][ldo] (row (b, j, k): its H heads, then zeros -- tri_db_rows' output) so tri_db_rows' launch and its fp32 round trip are gone.  Blocks
// [0, nb_sum) = triattn_db_slabs' sum (the same fixed order, the same bits); the later blocks write the zero columns [H, 64) of every row
// (16-byte stores from the first multiple of 8 columns, 2-byte stores before it).
__global__ void __launch_bounds__(256) triattn_db_slabs_rows(const float* __restrict__ SL, __nv_bfloat16* __restrict__ OUT, long ldo, int H, int S, int NG, int per,
                                                             long total, long n4, long nb_sum, long R) {
  pdl_trigger();
  pdl_wait();
  if ((long)blockIdx.x >= nb_sum) {
    const int z0 = (H + 7) & ~7;
    const long gt = ((long)blockIdx.x - nb_sum) * 256 + threadIdx.x, gs = ((long)gridDim.x - nb_sum) * 256;
    const int np = (64 - z0) / 8, ns = z0 - H;
    for (long t = gt; t < R * np; t += gs) {
      const long row = t / np;
      *reinterpret_cast<uint4*>(OUT + row * ldo + z0 + (t % np) * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
    if (ns > 0)
      for (long t = gt; t < R * ns; t += gs) OUT[(t / ns) * ldo + H + (t % ns)] = __float2bfloat16_rn(0.f);
    return;
  }
  const int col = threadIdx.x & 31, lane = threadIdx.x >> 5;
  const long i = (long)blockIdx.x * 32 + col;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  const long e = i * 4, ss = (long)S * S;
  int n = 0;
  if (i < n4) {
    n = (NG + per - 1) / per;
    int p = lane;
    for (; p + 24 < n; p += 32) {
      float4 v[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) v[k] = __ldg(reinterpret_cast<const float4*>(SL + (p + 8 * k) * total + e));
#pragma unroll
      for (int k = 0; k < 4; ++k) { a.x += v[k].x; a.y += v[k].y; a.z += v[k].z; a.w += v[k].w; }
    }
    for (; p < n; p += 8) { const float4 v = __ldg(reinterpret_cast<const float4*>(SL + p * total + e)); a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
  }
  __shared__ float4 part[8][32];
  part[lane][col] = a;
  __syncthreads();
  if (lane == 0 && i < n4) {
    float4 r = part[0][col];
#pragma unroll
    for (int g = 1; g < 8; ++g) { const float4 v = part[g][col]; r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w; }
    const long bh = e / ss, pos = e % ss, row = (bh / H) * ss + pos;     // row (b, j, k) = b S^2 + j S + k = b S^2 + pos
    const int h = (int)(bh % H);
    OUT[(row + 0) * ldo + h] = __float2bfloat16_rn(r.x);
    OUT[(row + 1) * ldo + h] = __float2bfloat16_rn(r.y);
    OUT[(row + 2) * ldo + h] = __float2bfloat16_rn(r.z);
    OUT[(row + 3) * ldo + h] = __float2bfloat16_rn(r.w);
  }
}

// ---- delta[b, n, h, s] = sum_d dout . out (fp32), the softmax backward's row term: one thread per (b, n, s, h), 64 B of each
__global__ void __launch_bounds__(256) triattn_delta(const __nv_bfloat16* __restrict__ dout, const __nv_bfloat16* __restrict__ out,
                                                     float* __restrict__ delta, int N, int H, int S, long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int h = (int)(i % H);
  const long bns = i / H;
  const int s = (int)(bns % S);
  const long bn = bns / S;
  const uint4* a = reinterpret_cast<const uint4*>(dout + i * D);
  const uint4* b = reinterpret_cast<const uint4*>(out + i * D);
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 x = a[c], y = b[c];
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 p = bf2f(xs[e]), q = bf2f(ys[e]); acc = fmaf(p.x, q.x, fmaf(p.y, q.y, acc)); }
  }
  delta[(bn * H + h) * S + s] = acc;
}

// ---- bias^T * log2e (fp32 [b, h, k, q]) for the KV kernel, whose threads own a key: one 32 x 32 tile per block
__global__ void __launch_bounds__(256) triattn_biasT(const __nv_bfloat16* __restrict__ bias, float* __restrict__ out, int S) {
  __shared__ float t[32][33];
  const int bh = blockIdx.z, q0 = blockIdx.y * 32, k0 = blockIdx.x * 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
  const __nv_bfloat16* src = bias + (size_t)bh * S * S;
#pragma unroll
  for (int i = ty; i < 32; i += 8) t[i][tx] = __bfloat162float(src[(size_t)(q0 + i) * S + k0 + tx]) * L2E;
  __syncthreads();
  float* dst = out + (size_t)bh * S * S;
#pragma unroll
  for (int i = ty; i < 32; i += 8) dst[(size_t)(k0 + i) * S + q0 + tx] = t[tx][i];
}

// ---- kernel 1: dK, dV.  Task = (b, h, 128-key tile) x KR = 2 pair rows, all queries in 64-wide sub-tiles; stage y = (sub-tile, row),
// y & 1 = the row = the grad group that owns it (the two groups alternate stages, as the forward's softmax groups do):
//   S^T MMA warp:  S^T = K_r . Q^T, dP^T = V_r . dO^T  (ss: K_r / V_r from shared; M = 128 k, N = 64 q, K = 32)
//   grad group r:  P^T, dS^T (bf16) -> its P^T | dS^T TMEM buffer
//   grad MMA warp: dV_r += P^T . dO, dK_r += dS^T . Q   (ts: A from TMEM, B MN-major; N = 32, K = 64)
// Every hand-off here (a barrier round trip, a group of MMA issues) costs ~100-300 clk, so the structure is chosen to keep them off
// the critical path and few per unit of work: 64-query stages (a group's fixed per-stage cost ~600 clk against ~700 of math, which
// the other group's math covers), each group's S^T | dP^T buffer is released as soon as the group has loaded it (the next S^T of
// that group goes out during its math), each group rewrites its P^T | dS^T buffer only a stage later, and the S^T and gradient GEMMs
// have their own issuing warps.  TMEM: S^T | dP^T of group g at 128 g (64 + 64 fp32), P^T | dS^T of group g at 256 + 64 g
// (32 + 32 bf16 pairs), dV_r / dK_r at 384 + 64 r.  K / V stay in shared memory (double-buffered across tasks).
constexpr int KR = 2;
constexpr int KB_QW = 64, KB_QT = KB_QW * D, KB_BT = BN * KB_QW;   // Q / dO sub-tile [64][32] (4 KiB), bias^T [2 halves][128 k][32 q] fp32
constexpr int KB_NQ = 6, KB_NB = 2, KB_THREADS = 352;              // 0 TMA, 1 S^T MMA, 2-5 / 6-9 grad groups (rows 0 / 1), 10 grad MMA
constexpr int KB_COL_P = 256, KB_COL_G = 384;
constexpr int KB_SMEM = 1024 + (2 * KR * 2 * KVT + KR * 2 * KVT + KB_NQ * 2 * KB_QT) * 2 + KB_NB * KB_BT * 4 + KB_NQ * 2 * KB_QW * 4 + 512;
constexpr uint32_t ID_ST = idesc_bf16(128, KB_QW, 0, 0);
constexpr uint32_t ID_G = idesc_bf16(128, D, 0, 1);
static_assert(KB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(KB_THREADS, 1) triattn_bwd_kv_sm100(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][H*32], box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias^T * log2e fp32 [B*H*S (k)][S (q)], box (32, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,   // [B, N, H, S]
    const __grid_constant__ CUtensorMap dkmap,           // dK / dV [B*N*S][H*32] bf16, box (32, 128), 64B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dvmap) {
  pdl_trigger();   // round 8: PDL at S = 128 (launch_pdl_if): the next kernel may launch; this one waits for the previous grid
  pdl_wait();
  const int NK = S / BN, NG = (N + KR - 1) / KR, NQS = S / KB_QW, SPT = KR * NQS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 task buffers][KR][K | V]
  __nv_bfloat16* sO = sKV + 2 * KR * 2 * KVT;                    // [KR][dV | dK] out tiles for the TMA stores
  __nv_bfloat16* sQ = sO + KR * 2 * KVT;                         // [NQ][Q | dO]
  float* sB = reinterpret_cast<float*>(sQ + KB_NQ * 2 * KB_QT);  // [NB][2 q halves][128 k][32 q]
  float* sLD = sB + KB_NB * KB_BT;                               // [NQ][lse 64 | delta 64]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLD + KB_NQ * 2 * KB_QW);
  uint64_t* kvf = bars;              // [2]  a task buffer's K / V landed
  uint64_t* kve = kvf + 2;           // [2]  its last S^T GEMMs retired
  uint64_t* dfr = kve + 2;           // [KR] row r's dK / dV final
  uint64_t* de = dfr + KR;           // [KR] count 4: row r drained
  uint64_t* qf = de + KR;            // [NQ]
  uint64_t* qe = qf + KB_NQ;         // [NQ]
  uint64_t* bf = qe + KB_NQ;         // [NB]
  uint64_t* be = bf + KB_NB;         // [NB] count 8
  uint64_t* sf = be + KB_NB;         // [2]  group g's S^T | dP^T ready
  uint64_t* se = sf + 2;             // [2]  count 4: group g loaded it
  uint64_t* pf = se + 2;             // [2]  count 4: group g's P^T | dS^T written
  uint64_t* ge = pf + 2;             // [2]  group g's gradient GEMMs retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ge + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) {
      bar_init(kvf + i, 1); bar_init(kve + i, 1); bar_init(dfr + i, 1); bar_init(de + i, 4);
      bar_init(sf + i, 1); bar_init(se + i, 4); bar_init(pf + i, 4); bar_init(ge + i, 1);
    }
    for (int i = 0; i < KB_NQ; ++i) { bar_init(qf + i, 1); bar_init(qe + i, 1); }
    for (int i = 0; i < KB_NB; ++i) { bar_init(bf + i, 1); bar_init(be + i, 8); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const uint32_t sbase = sa(smb);                  // shared-window address of p: sbase + (p - smb), no S2R per use
  auto A = [&](const void* p) -> uint32_t { return sbase + (uint32_t)(reinterpret_cast<const unsigned char*>(p) - smb); };
  auto decode = [&](int t, int& kt, int& g, int& bh) { kt = t % NK; g = (t / NK) % NG; bh = t / (NK * NG); };
  auto nrow = [&](int g, int r) { return min(g * KR + r, N - 1); };   // rows past N compute on row N-1 and store nothing
  const int mytasks = (int)blockIdx.x < ntasks ? (ntasks - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  const int total = mytasks * SPT;

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
        int kt, g, bh; decode(t, kt, g, bh);
        const int b_ = bh / H, h_ = bh % H, kb = lt & 1;
        if (lt >= 2) wait(A(kve + kb), ((lt >> 1) - 1) & 1);
        expect_tx(A(kvf + kb), KR * 2 * KVT * 2);
        for (int r = 0; r < KR; ++r) {
          const int rw = (b_ * N + nrow(g, r)) * S + kt * BN;
          load_2d(&kmap, A(sKV + (kb * KR + r) * 2 * KVT), A(kvf + kb), h_ * D, rw);
          load_2d(&vmap, A(sKV + (kb * KR + r) * 2 * KVT + KVT), A(kvf + kb), h_ * D, rw);
        }
        for (int w = 0; w < SPT; ++w, ++y) {
          const int qs = w >> 1, r = w & 1, sl = y % KB_NQ;
          if (r == 0) {
            const int jb = lt * NQS + qs, bs = jb % KB_NB;
            if (jb >= KB_NB) wait(A(be + bs), ((jb / KB_NB) - 1) & 1);
            if (TA_BIAS1 && jb >= KB_NB) arrive(A(bf + bs));   // timing only
            else {
            expect_tx(A(bf + bs), KB_BT * 4);
            for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, A(sB + bs * KB_BT + hh * BN * 32), A(bf + bs), qs * KB_QW + hh * 32, bh * S + kt * BN);
            }
          }
          if (y >= KB_NQ) wait(A(qe + sl), ((y / KB_NQ) - 1) & 1);
          if (TA_QD1 && y >= KB_NQ) { arrive(A(qf + sl)); continue; }   // timing only
          expect_tx(A(qf + sl), 2 * KB_QT * 2 + 2 * KB_QW * 4);
          const int rw = (b_ * N + nrow(g, r)) * S + qs * KB_QW;
          load_2d(&qmap, A(sQ + sl * 2 * KB_QT), A(qf + sl), h_ * D, rw);
          load_2d(&dmap, A(sQ + sl * 2 * KB_QT + KB_QT), A(qf + sl), h_ * D, rw);
          const size_t st = ((size_t)(b_ * N + nrow(g, r)) * H + h_) * S + qs * KB_QW;
          bulk_load(A(sLD + sl * 2 * KB_QW), LSE + st, KB_QW * 4, A(qf + sl));
          bulk_load(A(sLD + sl * 2 * KB_QW + KB_QW), DELTA + st, KB_QW * 4, A(qf + sl));
        }
      }
    }
  } else if (warp == 1 || warp == 10) {
    // Issue loops: the converged warp runs them, one elected lane's tcgen05 ops are predicated (no branch / warp sync per group);
    // ring slots and phase bits are incremental counters (no divisions); barrier addresses and descriptor bases precomputed.
    const uint32_t L = elect_one() ? 1u : 0u;
    constexpr uint64_t SLOT = 2 * KB_QT * 2 / 16, MAT = KB_QT * 2 / 16, KVM = KVT * 2 / 16, KVROW = 2 * KVT * 2 / 16;
    const uint32_t qfa = A(qf), qea = A(qe), sfa = A(sf), sea = A(se), pfa = A(pf), gea = A(ge), dea = A(de), dfa = A(dfr);
    int sl = 0, slp = 0, w = 0, lt = 0, gp = 0;          // ring slot / phase, task-local stage, task, phase of the group barriers
    if (warp == 1) {
      // S^T of a stage goes out once its Q | dO | lse | delta slot and its bias tile landed, so the grad group's S^T barrier implies
      // both (release / acquire through this thread): one barrier a stage for the group.
      const uint64_t dQ0 = desc_k64(sQ), dKV0 = desc_k64(sKV);
      const uint32_t bfa = A(bf), kvfa = A(kvf), kvea = A(kve);
      int bs = 0, bsp = 0;
      for (int y = 0; y < total; ++y) {
        const int r = w & 1, kb = lt & 1;
        TL(0, y);
        const uint32_t okq = probe(qfa + sl * 8, slp);
        const uint32_t oke = y >= 2 ? probe(sea + r * 8, gp ^ 1) : 1u;     // group r loaded its previous S^T
        const uint32_t okb = r == 0 ? probe(bfa + bs * 8, bsp) : 1u;
        if (w == 0) wait(kvfa + kb * 8, (lt >> 1) & 1);
        if (!okq) wait(qfa + sl * 8, slp);
        if (!oke) wait(sea + r * 8, gp ^ 1);
        if (!okb) wait(bfa + bs * 8, bsp);
        tc_fence_after();
        const uint64_t bq = dQ0 + sl * SLOT, ak = dKV0 + (kb * KR + r) * KVROW;
        const uint32_t d = tmem + r * 128;
        mma_ss_if(L, d, ak, bq, ID_ST, 0u);
        mma_ss_if(L, d, ak + 2, bq + 2, ID_ST, 1u);
        mma_ss_if(L, d + 64, ak + KVM, bq + MAT, ID_ST, 0u);
        mma_ss_if(L, d + 64, ak + KVM + 2, bq + MAT + 2, ID_ST, 1u);
        TL(1, y);
        mma_commit_if(L, sfa + r * 8);
        if (w == SPT - 1) mma_commit_if(L, kvea + kb * 8);  // the task's K / V buffer is free once these retire
        if (r == 1) { if (++bs == KB_NB) { bs = 0; bsp ^= 1; } gp ^= (y >= 1); }
        if (y == 0) gp = 0;
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    } else {
      const uint64_t dM0 = sdesc(sa(sQ), KB_QT * 2, 512, 4);
      const uint32_t tG = tmem + KB_COL_G;
      for (int p = 0; p < total; ++p) {
        const int r = w & 1;                               // = the group
        const bool first = w < 2, last = w >= SPT - 2;
        TL(2, p);
        const uint32_t okp = probe(pfa + r * 8, gp);
        if (first && lt >= 1) wait(dea + r * 8, (lt - 1) & 1);   // the row's accumulators of the previous task were drained
        if (!okp) wait(pfa + r * 8, gp);
        TL(3, p);
        tc_fence_after();
        const uint64_t bm = dM0 + sl * SLOT;
        const uint32_t acc = first ? 0u : 1u, a = tmem + KB_COL_P + r * 64, g0 = tG + r * 64;
#pragma unroll
        for (int kk = 0; kk < KB_QW / 16; ++kk) {
          mma_ts_if(L, g0, a + kk * 8, bm + MAT + kk * 64, ID_G, (acc | kk) ? 1u : 0u);        // dV_r += P^T dO
          mma_ts_if(L, g0 + 32, a + 32 + kk * 8, bm + kk * 64, ID_G, (acc | kk) ? 1u : 0u);    // dK_r += dS^T Q
        }
        mma_commit_if(L, qea + sl * 8);
        mma_commit_if(L, gea + r * 8);
        if (last) mma_commit_if(L, dfa + r * 8);
        if (r == 1) gp ^= 1;
        if (++sl == KB_NQ) { sl = 0; slp ^= 1; }
        if (++w == SPT) { w = 0; ++lt; }
      }
    }
    __syncwarp();
  } else if (warp < 10) {
    const int qq = warp & 3, k = qq * 32 + lane;   // TMEM lane = key row of the tile
    const int gi = (warp - 2) >> 2;                // this group's row
    const bool lead = qq == 2 && lane == 0;        // warps 2 / 6: the group's TMA-issuing thread
    const float2 sc2 = make_float2(scl, scl), neg1 = make_float2(-1.f, -1.f);
    const uint32_t tS = tmem_at(tmem + gi * 128, qq * 32, 0), tP = tmem_at(tmem + KB_COL_P + gi * 64, qq * 32, 0);
    int js = 0, lt = 0;                            // this group's stage count (phase of sf / se / pf / ge), task
    for (int t = blockIdx.x; t < ntasks; t += gridDim.x, ++lt) {
      int kt, g, bh; decode(t, kt, g, bh);
      const int b_ = bh / H, h_ = bh % H;
      for (int qs = 0; qs < NQS; ++qs, ++js) {
        const int jb = lt * NQS + qs, bs = jb % KB_NB, y = lt * SPT + qs * 2 + gi, sl = y % KB_NQ;
        if (lane == 0 && qq == 2) TL(4 + gi * 6, y);
        wait(A(sf + gi), js & 1);                   // implies the bias tile and the lse / delta slot (see the S^T issuer)
        if (lane == 0 && qq == 2) TL(5 + gi * 6, y);
        tc_fence_after();
        const float* ld = sLD + sl * 2 * KB_QW;
#pragma unroll
        for (int c = 0; c < 2; ++c) {              // 32-query chunks (registers)
          float sv[32], dp[32], bl[32];
          tmem_ld32(tS + c * 32, sv);
          tmem_ld32(tS + 64 + c * 32, dp);
          {
            const float* bp = sB + bs * KB_BT + c * BN * 32 + k * 32;
#pragma unroll
            for (int q4 = 0; q4 < 8; ++q4) {
              const float4 u = *reinterpret_cast<const float4*>(bp + ((q4 ^ (k & 7)) << 2));
              bl[q4 * 4] = u.x; bl[q4 * 4 + 1] = u.y; bl[q4 * 4 + 2] = u.z; bl[q4 * 4 + 3] = u.w;
            }
          }
          tmem_wait_ld();
          if (c == 1) {                             // both chunks loaded: the group's S^T buffer takes its next stage
            fence_proxy_async();                    // the bias^T tile's reads, before its release: an arrive alone does not wait for outstanding LDS
            tc_fence_before();
            __syncwarp();
            if (lane == 0) { arrive(A(se + gi)); arrive(A(be + bs)); }
          }
          uint32_t pk[16], dk[16];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float4 l4 = *reinterpret_cast<const float4*>(ld + c * 32 + 4 * j);
            const float4 d4 = *reinterpret_cast<const float4*>(ld + KB_QW + c * 32 + 4 * j);
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              const int i = 4 * j + 2 * h2;
              const float2 lm = h2 ? make_float2(l4.z, l4.w) : make_float2(l4.x, l4.y);
              const float2 dm = h2 ? make_float2(d4.z, d4.w) : make_float2(d4.x, d4.y);
              // FADD2 has no negate: -lse / -delta go through FFMA2 with -1 (else two FADD negations a pair)
              const float2 x = fma2(lm, neg1, fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])));
              const float2 pp = (TB_POLY && h2) ? ex2_poly2(x) : make_float2(ex2f(x.x), ex2f(x.y));
              const float2 ds = mul2(pp, fma2(dm, neg1, make_float2(dp[i], dp[i + 1])));
              pk[i >> 1] = pack2(pp.x, pp.y);
              dk[i >> 1] = pack2(ds.x, ds.y);
            }
          }
          if (c == 0 && js >= 1) {                  // this group's previous gradient GEMMs read the P^T | dS^T buffer
            wait(A(ge + gi), (js - 1) & 1);
            tc_fence_after();
          }
          tmem_st8(tP + c * 16, pk); tmem_st8(tP + c * 16 + 8, pk + 8);
          tmem_st8(tP + 32 + c * 16, dk); tmem_st8(tP + 32 + c * 16 + 8, dk + 8);
        }
        if (lane == 0 && qq == 2) TL(6 + gi * 6, y);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(pf + gi));
        if (lane == 0 && qq == 2) TL(7 + gi * 6, y);
      }
      // ---- dK / dV of row gi ----
      wait(A(dfr + gi), lt & 1);
      tc_fence_after();
      float v[32], kk[32];
      tmem_ld32(tmem_at(tmem + KB_COL_G + gi * 64, qq * 32, 0), v);
      tmem_ld32(tmem_at(tmem + KB_COL_G + gi * 64 + 32, qq * 32, 0), kk);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(A(de + gi));           // the accumulators are free for the next task
      if (lead) bulk_wait_read<0>();               // the previous task's stores have read the out tiles
      named_sync(1 + gi, 128);
      {
        uint32_t wv[16], wk[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) { wv[i] = pack2(v[2 * i], v[2 * i + 1]); wk[i] = pack2(kk[2 * i] * scale, kk[2 * i + 1] * scale); }
        row_sw64_st(sO + gi * 2 * KVT, k, wv);
        row_sw64_st(sO + gi * 2 * KVT + KVT, k, wk);
      }
      fence_proxy_async();
      named_sync(1 + gi, 128);
      if (lead && g * KR + gi < N) {
        const int rw = (b_ * N + g * KR + gi) * S + kt * BN;
        store_2d(&dvmap, A(sO + gi * 2 * KVT), h_ * D, rw);
        store_2d(&dkmap, A(sO + gi * 2 * KVT + KVT), h_ * D, rw);
        bulk_commit();
      }
    }
    if (lead) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- kernel 2: dQ and dbias.  Task = (b, h, 128-query tile) x QR pair rows, all keys in 64-wide sub-tiles (stage = (sub-tile, row)).
//   MMA:   S = Q_r . K^T, dP = dO_r . V^T        (ts: A = Q_r / dO_r from TMEM, B = K / V K-major; M = 128 q, N = 64 k, K = 32)
//   grad:  dS (bf16) over dP; dbias summed over the QR rows in registers   (group gi: key columns 32 gi .. of the stage)
//   MMA:   dQ_r += dS . K                        (ts: A = dS from TMEM, B = K MN-major; N = 32)
constexpr int QR = TA_QR;
static_assert(QR % 2 == 0 && QR >= 2 && QR <= 4, "QR");
constexpr int QB_KW = 64, QB_KT = QB_KW * D, QB_BT = BM * QB_KW;   // K / V sub-tile [64][32] (4 KiB), bias [128 q][64 k] 128B swizzle
constexpr int QB_NKV = 6, QB_NB = 2, QB_NS = 2, QB_THREADS = 320;
constexpr int QB_COL_G = QB_NS * 128;                               // dQ_r at 256 + 32 r
constexpr int QB_COL_A = QB_COL_G + QR * 32;                        // Q_r at 384 + 16 r, dO_r at 384 + 16 QR + 16 r
constexpr int QB_DBT = BM * 32;                                   // a group's dbias partial [128 q][32 k] fp32, 128B swizzle (16 KiB)
constexpr int QB_SMEM = 1024 + (QR * 2 * QT + QB_NKV * 2 * QB_KT + QB_NB * QB_BT) * 2 + QR * 2 * BM * 4 + 2 * 2 * QB_DBT * 4 + 256;
constexpr uint32_t ID_SQ = idesc_bf16(128, QB_KW, 0, 0);
static_assert(QB_COL_A + QR * 32 <= 512, "TMEM");
static_assert(QB_SMEM <= 232448, "smem");

__global__ void __launch_bounds__(QB_THREADS, 1) triattn_bwd_q_sm100(
    int N, int H, int S, int ntasks, float scl, float scale,
    const __grid_constant__ CUtensorMap qmap,            // q / dO [B*N*S][H*32], box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap dmap,
    const __grid_constant__ CUtensorMap kmap,            // k / v, box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // bias bf16 [B*H*S][S], box (64, 128), 128B swizzle
    const float* __restrict__ LSE, const float* __restrict__ DELTA,
    const __grid_constant__ CUtensorMap dqmap,           // dQ [B*N*S][H*32] bf16, box (32, 128), 64B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dbmap,           // dbias slabs fp32 [P][B*H*S][S]; box (32, 128), 128B swizzle (see below)
    const int per, const int bhs) {                      // tasks a CTA (a contiguous range), B*H*S
  pdl_trigger();   // round 8: PDL at S = 128 (launch_pdl_if): the next kernel may launch; this one waits for the previous grid
  pdl_wait();
  const int QTL = S / BM, NG = (N + QR - 1) / QR, NKS = S / QB_KW, SPT = QR * NKS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smb);    // [QR][Q | dO] staging for TMEM
  __nv_bfloat16* sKV = sA + QR * 2 * QT;                         // [NKV][K | V]
  __nv_bfloat16* sB = sKV + QB_NKV * 2 * QB_KT;                  // [NB][128 q][64 k]
  float* sLD = reinterpret_cast<float*>(sB + QB_NB * QB_BT);     // [QR][lse 128 | delta 128]
  float* sDB = sLD + QR * 2 * BM;                                // [2 groups][2 buffers][QB_DBT] dbias partials; the dQ out tiles
                                                                 // reuse a group's buffers at the task's end
  uint64_t* bars = reinterpret_cast<uint64_t*>(sDB + 2 * 2 * QB_DBT);
  uint64_t* af = bars;
  uint64_t* ae = af + 1;             // count 8
  uint64_t* at = ae + 1;             // count 8
  uint64_t* df = at + 1;
  uint64_t* kvf = df + 1;            // [NKV]
  uint64_t* kve = kvf + QB_NKV;      // [NKV]
  uint64_t* bf = kve + QB_NKV;       // [NB]
  uint64_t* be = bf + QB_NB;         // [NB] count 8
  uint64_t* sf = be + QB_NB;         // [NS]
  uint64_t* pf = sf + QB_NS;         // [NS] count 8
  uint32_t* tslot = reinterpret_cast<uint32_t*>(pf + QB_NS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    bar_init(af, 1); bar_init(ae, 8); bar_init(at, 8); bar_init(df, 1);
    for (int i = 0; i < QB_NKV; ++i) { bar_init(kvf + i, 1); bar_init(kve + i, 1); }
    for (int i = 0; i < QB_NB; ++i) { bar_init(bf + i, 1); bar_init(be + i, 8); }
    for (int i = 0; i < QB_NS; ++i) { bar_init(sf + i, 1); bar_init(pf + i, 8); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  if (warp >= 2 && (warp & 3) == 2 && lane == 0) bulk_wait<0>();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const uint32_t sbase = sa(smb);                  // shared-window address of p: sbase + (p - smb), no S2R per use
  auto A = [&](const void* p) -> uint32_t { return sbase + (uint32_t)(reinterpret_cast<const unsigned char*>(p) - smb); };
  const int NP = (NG + per - 1) / per;                 // parts of a unit's row groups
  // CTA c = ((bh NP + part) QTL + qt) runs rows part * per .. of unit (bh, qt): the CTAs of one part step through the same rows at once
  // (K / V shared in L2 across the query tiles, as the round-robin order did); virtual task t = c per + j
  auto decode = [&](int t, int& qt, int& g, int& bh) { const int c = t / per; qt = c % QTL; g = ((c / QTL) % NP) * per + t % per; bh = c / (QTL * NP); };
  // deterministic dbias: each CTA runs one part (up to `per` row groups) of one (b, h, query tile) unit and writes the part's slab of it
  // (its first task stores, the later ones reduce-add after the earlier writes completed), and triattn_db_slabs sums the NP slabs in a
  // fixed order: every bit the same each run
  const int part = ((int)blockIdx.x / QTL) % NP, t0 = (int)blockIdx.x * per, t1 = t0 + min(per, NG - part * per);
  auto nrow = [&](int g, int r) { return min(g * QR + r, N - 1); };

  if (warp == 0) {
    if (lane == 0) {
      auto staging = [&](int t) {                // Q / dO rows, lse, delta of task t -> staging
        int qt, g, bh; decode(t, qt, g, bh);
        const int b_ = bh / H, h_ = bh % H;
        expect_tx(A(af), QR * 2 * QT * 2 + QR * 2 * BM * 4);
        for (int r = 0; r < QR; ++r) {
          const int rw = (b_ * N + nrow(g, r)) * S + qt * BM;
          load_2d(&qmap, A(sA + r * 2 * QT), A(af), h_ * D, rw);
          load_2d(&dmap, A(sA + r * 2 * QT + QT), A(af), h_ * D, rw);
          const size_t st = ((size_t)(b_ * N + nrow(g, r)) * H + h_) * S + qt * BM;
          bulk_load(A(sLD + r * 2 * BM), LSE + st, BM * 4, A(af));
          bulk_load(A(sLD + r * 2 * BM + BM), DELTA + st, BM * 4, A(af));
        }
      };
      if (t0 < t1) staging(t0);
      const int st_at = min(SPT, QB_NKV) - 1;
      int lt = 0, y = 0;
      for (int t = t0; t < t1; ++t, ++lt) {
        int qt, g, bh; decode(t, qt, g, bh);
        const int b_ = bh / H, h_ = bh % H;
        for (int w = 0; w < SPT; ++w, ++y) {
          if (w == st_at && t + 1 < t1) { wait(A(ae), lt & 1); staging(t + 1); }
          const int ks = w / QR, r = w % QR, sl = y % QB_NKV;
          if (r == 0) {
            const int jb = lt * NKS + ks, bs = jb % QB_NB;
            if (jb >= QB_NB) wait(A(be + bs), ((jb / QB_NB) - 1) & 1);
            expect_tx(A(bf + bs), QB_BT * 2);
            load_2d(&bmap, A(sB + bs * QB_BT), A(bf + bs), ks * QB_KW, bh * S + qt * BM);
          }
          if (y >= QB_NKV) wait(A(kve + sl), ((y / QB_NKV) - 1) & 1);
          expect_tx(A(kvf + sl), 2 * QB_KT * 2);
          const int rw = (b_ * N + nrow(g, r)) * S + ks * QB_KW;
          load_2d(&kmap, A(sKV + sl * 2 * QB_KT), A(kvf + sl), h_ * D, rw);
          load_2d(&vmap, A(sKV + sl * 2 * QB_KT + QB_KT), A(kvf + sl), h_ * D, rw);
        }
      }
    }
  } else if (warp == 1) {
    // Issue loop: the converged warp runs it and one elected lane's tcgen05 ops are predicated (an `if (elect_one())` block per MMA
    // group cost a branch / warp sync and a uniform-register waterfall per MMA: ~400 clk to issue a stage's dQ GEMMs, on the
    // critical path of the two S buffers); ring slots and phases are incremental counters (no divisions by QB_NKV).
    const uint32_t L = elect_one() ? 1u : 0u;
    const int mytasks = max(0, t1 - t0);
    const int total = mytasks * SPT;
    const uint64_t dK0 = desc_k64(sKV), dM0 = sdesc(sa(sKV), QB_KT * 2, 512, 4);
    constexpr uint64_t SLOT = 2 * QB_KT * 2 / 16, MAT = QB_KT * 2 / 16;
    static_assert(QB_NS == 2, "S buffers: y & 1");
    const uint32_t ata = A(at), kvfa = A(kvf), kvea = A(kve), sfa = A(sf), pfa = A(pf), dfa = A(df);
    const uint32_t tA = tmem + QB_COL_A, tG = tmem + QB_COL_G;
    int y = 0, ylt = 0, yw = 0, ysl = 0, yslp = 0;   // S / dP GEMMs: stage, task, stage in task, K | V slot and its phase
    int pw = 0, psl = 0, pb = 0, pbp = 0;            // dQ GEMMs: stage in task, K | V slot, S buffer and its pf phase
    for (int p = 0; p < total; ++p) {
      // a task's first S waits for its TMEM operands, which the grad warps copy only after draining the previous task's
      // accumulators: issue it after the previous task's last gradient GEMM
      while (y < total && y < p + QB_NS && (yw != 0 || y == p)) {
        const int r = yw % QR, b = y & 1;
        if (yw == 0) wait(ata, ylt & 1);
        wait(kvfa + ysl * 8, yslp);
        tc_fence_after();
        const uint64_t bk = dK0 + ysl * SLOT;
        const uint32_t d = tmem + b * 128, a = tA + r * 16;
        mma_ts_if(L, d, a, bk, ID_SQ, 0u);
        mma_ts_if(L, d, a + 8, bk + 2, ID_SQ, 1u);
        mma_ts_if(L, d + 64, a + QR * 16, bk + MAT, ID_SQ, 0u);
        mma_ts_if(L, d + 64, a + QR * 16 + 8, bk + MAT + 2, ID_SQ, 1u);
        mma_commit_if(L, sfa + b * 8);
        ++y;
        if (++ysl == QB_NKV) { ysl = 0; yslp ^= 1; }
        if (++yw == SPT) { yw = 0; ++ylt; }
      }
      const int kb = pw / QR, r = pw % QR;
      wait(pfa + pb * 8, pbp);
      tc_fence_after();
      const uint64_t bm = dM0 + psl * SLOT;
      const uint32_t acc = kb ? 1u : 0u, d = tG + r * 32, a0 = tmem + pb * 128 + 64;
#pragma unroll
      for (int ks = 0; ks < QB_KW / 16; ++ks)
        mma_ts_if(L, d, a0 + (ks >> 1) * 32 + (ks & 1) * 8, bm + ks * 64, ID_G, (acc | ks) ? 1u : 0u);
      mma_commit_if(L, kvea + psl * 8);
      if (pw == SPT - 1) mma_commit_if(L, dfa);
      if (++psl == QB_NKV) psl = 0;
      pbp ^= pb; pb ^= 1;                              // pf phase (p / 2) & 1 of buffer p & 1
      if (++pw == SPT) pw = 0;
    }
    __syncwarp();
  } else {
    const int qq = warp & 3, row = qq * 32 + lane;   // TMEM lane = query row of the tile
    const int gi = (warp - 2) >> 2, c0 = gi * 32;    // key columns of a stage; rows gi, gi + 2 for the Q / dO copy and dQ drain
    const float2 sc2 = make_float2(scl, scl);
    const bool lead = (warp & 3) == 2 && lane == 0;   // the group's TMA-issuing thread (warps 2 / 6)
    float* myDB = sDB + gi * 2 * QB_DBT;
    int nb = 0;                                       // dbias buffers used by this group so far
    int lt = 0;
    for (int t = t0; t < t1; ++t, ++lt) {
      int qt, g, bh; decode(t, qt, g, bh);
      const int b_ = bh / H;
      float lse[QR], dl[QR];
      const int slab = part;                               // this CTA's slab of its unit
      const bool first = t == t0;                         // its first task (the unit's only CTA of this part): stores
      const int dbrow = slab * bhs + bh * S + qt * BM;
      {
        wait(A(af), lt & 1);
        const uint32_t ta = tmem_at(tmem + QB_COL_A, qq * 32, 0);
#pragma unroll
        for (int rr = 0; rr < QR / 2; ++rr) {
          const int r = gi + 2 * rr;
          uint32_t wq[16], wd[16];
          row_sw64(sA + r * 2 * QT, row, wq);
          row_sw64(sA + r * 2 * QT + QT, row, wd);
          tmem_st8(ta + r * 16, wq); tmem_st8(ta + r * 16 + 8, wq + 8);
          tmem_st8(ta + QR * 16 + r * 16, wd); tmem_st8(ta + QR * 16 + r * 16 + 8, wd + 8);
        }
#pragma unroll
        for (int r = 0; r < QR; ++r) { lse[r] = sLD[r * 2 * BM + row]; dl[r] = sLD[r * 2 * BM + BM + row]; }
        fence_proxy_async();                        // the staging's reads (Q / dO, lse / delta), before its release: an arrive alone does not wait for outstanding LDS
        __syncwarp();
        if (lane == 0) arrive(A(ae));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(A(at));
      }
      for (int kb = 0; kb < NKS; ++kb) {
        const int jb = lt * NKS + kb, bs = jb % QB_NB;
        wait(A(bf + bs), (jb / QB_NB) & 1);
        float bl[32], dacc[32];
        {
          const __nv_bfloat16* brow = sB + bs * QB_BT + row * 64;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint4 u = *reinterpret_cast<const uint4*>(brow + (((gi * 4 + e) ^ (row & 7)) << 3));
            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
              const float2 f = bf2f(w4[k2]);
              bl[e * 8 + k2 * 2] = f.x * L2E; bl[e * 8 + k2 * 2 + 1] = f.y * L2E;
            }
          }
#pragma unroll
          for (int i = 0; i < 32; ++i) dacc[i] = 0.f;
        }
        fence_proxy_async();                        // the bias tile's reads, before its release: an arrive alone does not wait for outstanding LDS
        __syncwarp();
        if (lane == 0) arrive(A(be + bs));
#pragma unroll
        for (int r = 0; r < QR; ++r) {
          const int y = lt * SPT + kb * QR + r, b = y % QB_NS;
          const bool valid = g * QR + r < N;
          wait(A(sf + b), (y / QB_NS) & 1);
          tc_fence_after();
          const uint32_t tb = tmem_at(tmem + b * 128, qq * 32, 0);
          float sv[32], dp[32];
          tmem_ld32(tb + c0, sv);
          tmem_ld32(tb + 64 + c0, dp);
          tmem_wait_ld();
          const float2 nl = make_float2(-lse[r], -lse[r]), nd = make_float2(-dl[r], -dl[r]);
          const float vm = valid ? 1.f : 0.f;       // padding rows add nothing to dbias
          const float2 vm2 = make_float2(vm, vm);
          uint32_t dk[16];
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const float2 x = add2(fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])), nl);
            const float2 pp = (TB_POLY && (i & 2)) ? ex2_poly2(x) : make_float2(ex2f(x.x), ex2f(x.y));
            const float2 ds = mul2(pp, add2(make_float2(dp[i], dp[i + 1]), nd));
            dk[i >> 1] = pack2(ds.x, ds.y);
            const float2 a2 = fma2(ds, vm2, make_float2(dacc[i], dacc[i + 1]));
            dacc[i] = a2.x; dacc[i + 1] = a2.y;
          }
          tmem_st8(tb + 64 + c0, dk); tmem_st8(tb + 64 + c0 + 8, dk + 8);
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(A(pf + b));
        }
        // the group's [128 q][32 k] partial -> shared -> one TMA reduce-add (per-thread 16-byte reductions touch 32 lines each)
        {
          float* buf = myDB + (nb & 1) * QB_DBT;
          // the reduce that used this buffer two rounds ago has read it; after the CTA's first task in the unit every older write of
          // the lead is complete (wait, not wait_read): the previous task's write of this stage landed before this reduce-add -- a fixed
          // order (a bulk_wait<0> at each task's start instead: bwd_q L640 546 -> 704 us)
          if (lead) { if (first) bulk_wait_read<1>(); else bulk_wait<1>(); }
          named_sync(1 + gi, 128);
#pragma unroll
          for (int c = 0; c < 8; ++c)
            *reinterpret_cast<float4*>(buf + row * 32 + ((c ^ (row & 7)) << 2)) = make_float4(dacc[4 * c], dacc[4 * c + 1], dacc[4 * c + 2], dacc[4 * c + 3]);
          fence_proxy_async();
          named_sync(1 + gi, 128);
          if (lead) {
            if (first) store_2d(&dbmap, A(buf), kb * QB_KW + c0, dbrow); else reduce_add_2d(&dbmap, A(buf), kb * QB_KW + c0, dbrow);
            bulk_commit();
          }
          ++nb;
        }
      }
      // ---- dQ of rows gi, gi + 2 ----
      wait(A(df), lt & 1);
      tc_fence_after();
      float a[QR / 2][32];
#pragma unroll
      for (int rr = 0; rr < QR / 2; ++rr) tmem_ld32(tmem_at(tmem + QB_COL_G + (gi + 2 * rr) * 32, qq * 32, 0), a[rr]);
      tmem_wait_ld();
      tc_fence_before();
      // dQ rows gi, gi + 2 -> the group's dbias buffers (both reduces retired) -> TMA stores
      __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(myDB);
      if (lead) bulk_wait_read<0>();
      named_sync(1 + gi, 128);
#pragma unroll
      for (int rr = 0; rr < QR / 2; ++rr) {
        uint32_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = pack2(a[rr][2 * i] * scale, a[rr][2 * i + 1] * scale);
        row_sw64_st(ob + rr * QT, row, w);
      }
      fence_proxy_async();
      named_sync(1 + gi, 128);
      if (lead) {
#pragma unroll
        for (int rr = 0; rr < QR / 2; ++rr) {
          const int r = gi + 2 * rr;
          if (g * QR + r < N) store_2d(&dqmap, A(ob + rr * QT), (bh % H) * D, (b_ * N + g * QR + r) * S + qt * BM);
        }
        bulk_commit();
      }
      nb = 1;                                         // buffer 0 holds the dQ tiles: the next partial goes to buffer 1, whose
                                                      // wait_read<1> may leave this store pending; buffer 0's then retires it
    }
  }
  if (warp >= 2 && (warp & 3) == 2 && lane == 0) bulk_wait<0>();
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
}  // namespace

// q, k, v [B, N, S, H, 32] bf16 contiguous (the projection's layout: row (b, n, s) holds all heads); bias [B, H, S, S] (any float
// dtype; staged to bf16 when it is not); S % 128 == 0; no mask.
// (q / k / v rows may sit at a common stride: column slices of one projection output)
// -> (out [B, N, S, H, 32] bf16 contiguous, lse fp32 [B, N, H, S] base 2 (empty unless want_lse), flags int32 [1]: rows whose offset overflowed,
//     a per-device counter accumulated over every call since the extension loaded -- never re-zeroed, so no memset per call)
// out_ / lse_: tensors to write into (a group of samples' part of a batch's out [B, N, S, H, 32] / lse [B, N, H, S], contiguous), or
// empty (fresh tensors)
std::vector<torch::Tensor> triattn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, double scale, bool want_lse,
                                       torch::Tensor out_, torch::Tensor lse_) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == DG,
              TA_D16C ? "q: [B, N, S, H, 16] bf16 (the compact build)" : "q: [B, N, S, H, 32] bf16");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  // q / k / v may be column slices of one projection output: rows (b, n, s) at a common stride ld >= H*DG (16-byte multiple)
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16 && t->stride(4) == 1 && t->stride(3) == DG &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld,
                "q, k, v: same shape, [.., H, DG] dense per row, rows at one common stride");
  TORCH_CHECK(ld >= H * DG && ld % 8 == 0, "q / k / v row stride: >= H*DG and a multiple of 8 elements");
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.numel() == B * H * S * S, "bias: [B, H, S, S]");
  auto b16 = bias.scalar_type() == torch::kBFloat16 ? bias.contiguous() : bias.to(torch::kBFloat16).contiguous();
  const bool og = out_.defined() && out_.numel() > 0, lg = want_lse && lse_.defined() && lse_.numel() > 0;
  if (og) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == q.numel(), "out: contiguous bf16 like q");
  if (lg) TORCH_CHECK(lse_.scalar_type() == torch::kFloat32 && lse_.is_contiguous() && lse_.numel() == B * N * H * S, "lse: contiguous fp32 [B, N, H, S]");
  auto out = og ? out_.view(q.sizes()) : torch::empty(q.sizes(), q.options());
  auto lse = lg ? lse_.view({B, N, H, S}) : want_lse ? torch::empty({B, N, H, S}, q.options().dtype(torch::kFloat32))
                                                       : torch::empty({0}, q.options().dtype(torch::kFloat32));
  static torch::Tensor flags_dev[64];
  torch::Tensor& flags = flags_dev[q.device().index()];
  if (!flags.defined()) flags = torch::zeros({1}, q.options().dtype(torch::kInt32));
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * DG);
  auto qkv_map = [&](const torch::Tensor& t, uint32_t box, const char* what) {
    if (TA_D16C) {   // [rows][H][16]; the 32-channel box zero-fills 16 .. 31
      return make_map<3>(t.data_ptr(), {16, (uint64_t)H, rows}, {16, (uint64_t)ld}, {D, 1, box}, CU_TENSOR_MAP_SWIZZLE_64B, what);
    }
    return make_map<2>(t.data_ptr(), {cols, rows}, {(uint64_t)ld}, {D, box}, CU_TENSOR_MAP_SWIZZLE_64B, what);
  };
  CUtensorMap qm = qkv_map(q, BM, "q");
  CUtensorMap km = qkv_map(k, BN, "k");
  CUtensorMap vm = qkv_map(v, BN, "v");
  // bias [B*H*S][S] as [S / 64 key blocks][rows][64]: a key tile's two 64-key halves in one box
  CUtensorMap bm = make_map<3>(b16.data_ptr(), {64, (uint64_t)(B * H * S), (uint64_t)(S / 64)}, {(uint64_t)S, 64}, {64, BM, 2},
                               CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  const int ntasks = (int)(B * H * (S / BM) * ((N + R - 1) / R));
  const int grid = std::min(ntasks, num_sms(q.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_fwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  const float scl = TA_QS ? 1.f : (float)scale * L2E;
  TORCH_CHECK(!((TA_D16 || TA_ONES) && want_lse), "the TA_D16 / TA_ONES builds are inference only (no LSE)");
  launch_pdl(triattn_fwd_sm100, dim3(grid), dim3(THREADS), SMEM, at::cuda::getCurrentCUDAStream(), (int)N, (int)H, (int)S, ntasks, scl, qm, km, vm, bm,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()), (int)ld,
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b16.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags};
}

// Backward of triattn_fwd.  q, k, v, dout [B, N, S, H, 32] bf16 (the projection layout); bias [B, H, S, S]; lse / delta fp32
// [B, N, H, S] (lse: triattn_fwd's base-2 LSE; delta = rowsum(dout o out)).
// -> (dq, dk, dv [B, N, S, H, 32] bf16, dbias fp32 [B, H, S, S], dS^T bf16 [B, N, H, S, S] scratch)
#if TA_TL
torch::Tensor triattn_tl() { auto t = torch::empty({24, 256}, torch::kInt64); C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), g_tl, sizeof(g_tl))); return t; }
#endif
// delta = rowsum(dout o out) fp32 [B, N, H, S] from dout / out [B, N, S, H, 32] bf16
torch::Tensor triattn_delta_rows(torch::Tensor dout, torch::Tensor out) {
  TORCH_CHECK(dout.is_contiguous() && out.is_contiguous() && dout.sizes() == out.sizes() && dout.dim() == 5 && dout.size(4) == D, "dout / out: [B, N, S, H, 32]");
  const long B = dout.size(0), N = dout.size(1), S = dout.size(2), H = dout.size(3), total = B * N * S * H;
  auto delta = torch::empty({B, N, H, S}, dout.options().dtype(torch::kFloat32));
  triattn_delta<<<(unsigned)((total + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      delta.data_ptr<float>(), (int)N, (int)H, (int)S, total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return delta;
}

// B200 training tails round 8: the core backward's three launches as programmatic dependents (entry trigger +
// wait) at S = 128 when the caller asks (pdl: the wide path; module training L128 d_pair 64 x 2 x 32 85.5 -> 83.5 us; slower on the d_pair 128
// path (+1 us) and at L256 / L768)
template <typename... P, typename... A>
inline void launch_pdl_if(bool on, void (*kern)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, A&&... args) {
  if (on) { launch_pdl(kern, grid, block, smem, st, std::forward<A>(args)...); return; }
  kern<<<grid, block, smem, st>>>(std::forward<A>(args)...);
}

std::vector<torch::Tensor> triattn_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor dout,
                                       torch::Tensor lse, torch::Tensor delta, double scale, torch::Tensor db_ws,
                                       torch::Tensor dq_out, torch::Tensor dk_out, torch::Tensor dv_out, torch::Tensor bias_t, bool raw_slabs, bool pdl, int which, torch::Tensor db_rows) {   // which: 0 both kernels, 1 the key side only, 2 the query side + slab sum only (S90: two graph branches)
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 5 && q.size(4) == D, "q: [B, N, S, H, 32] bf16");
  for (const auto* t : {&q, &k, &v, &dout}) TORCH_CHECK(t->is_contiguous() && t->sizes() == q.sizes() && t->scalar_type() == torch::kBFloat16, "q, k, v, dout: contiguous, same shape");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  TORCH_CHECK(bias.numel() == B * H * S * S, "bias: [B, H, S, S]");
  for (const auto* t : {&lse, &delta}) TORCH_CHECK(t->scalar_type() == torch::kFloat32 && t->is_contiguous() && t->numel() == B * N * H * S, "lse / delta: fp32 [B, N, H, S]");
  auto b16 = bias.scalar_type() == torch::kBFloat16 ? bias.contiguous() : bias.to(torch::kBFloat16).contiguous();
  // dq_out / dk_out / dv_out: optional [rows, H*32] views (dense rows at a 16-byte-multiple stride, e.g. column slices of one
  // gradient buffer) the kernels store into; empty -> fresh [B, N, S, H, 32] tensors
  const bool given = dq_out.defined() && dq_out.numel() > 0;
  auto dq = given ? dq_out : torch::empty_like(q), dk = given ? dk_out : torch::empty_like(q), dv = given ? dv_out : torch::empty_like(q);
  if (given)
    for (const auto* t : {&dq, &dk, &dv})
      TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->dim() == 2 && t->size(0) == B * N * S && t->size(1) == H * D && t->stride(1) == 1 &&
                  t->stride(0) % 8 == 0, "dq / dk / dv outputs: [rows, H*32] bf16, dense rows at a 16-byte-multiple stride");
  // db_ws: a fp32 buffer of B*H*S*S the dbias is written into (every element; its contents are not read), or fresh
  auto db = (db_ws.defined() && db_ws.numel() == B * H * S * S) ? db_ws.view({B, H, S, S}) : torch::empty({B, H, S, S}, q.options().dtype(torch::kFloat32));
  // the query-side kernel's deterministic dbias slabs (see triattn_bwd_q_sm100): CTA c runs `per` consecutive tasks; a (b, h, query tile)
  // unit of NG tasks is touched by at most ceil(NG / per) + 1 CTAs = slabs
  const int sms_ = num_sms(q.device().index());
  const int NGq = (int)((N + QR - 1) / QR), n2 = (int)(B * H * (S / BM) * NGq);
  int per2 = (n2 + sms_ - 1) / sms_;                 // the fewest rows a CTA with B H QTL ceil(NG / per) CTAs <= the SMs
  while (per2 < NGq && B * H * (S / BM) * ((NGq + per2 - 1) / per2) > sms_) ++per2;
  const int nslab = (NGq + per2 - 1) / per2, grid2 = (int)(B * H * (S / BM) * nslab);
  auto slabs = torch::empty({(long)nslab * B * H * S, S}, q.options().dtype(torch::kFloat32));
  CUtensorMap dbm = make_map<2>(slabs.data_ptr(), {(uint64_t)S, (uint64_t)(nslab * B * H * S)}, {(uint64_t)S}, {32, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "dbias",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  const uint64_t rows = (uint64_t)(B * N * S), cols = (uint64_t)(H * D);
  auto m2 = [&](const torch::Tensor& t, uint32_t box, const char* what) { return make_map<2>(t.data_ptr(), {cols, rows}, {cols}, {D, box}, CU_TENSOR_MAP_SWIZZLE_64B, what); };
  CUtensorMap q32 = m2(q, KB_QW, "q"), d32 = m2(dout, KB_QW, "dout"), k128 = m2(k, 128, "k"), v128 = m2(v, 128, "v");   // (q32: box KB_QW rows)
  CUtensorMap q128 = m2(q, 128, "q"), d128 = m2(dout, 128, "dout"), k64 = m2(k, 64, "k"), v64 = m2(v, 64, "v");
  auto mo = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {(uint64_t)(given ? t.stride(0) : (long)cols)}, {D, 128}, CU_TENSOR_MAP_SWIZZLE_64B, what); };
  CUtensorMap dkm = mo(dk, "dk"), dvm = mo(dv, "dv"), dqm = mo(dq, "dq");
  // bias_t: bias^T * log2e fp32 [B, H, S (k), S (q)] already made by the caller (B200 tri_bwd_pre: one launch with the dropout scale and
  // the dbias zero fill), or empty (made here by triattn_biasT)
  const bool given_t = bias_t.defined() && bias_t.numel() == B * H * S * S && bias_t.scalar_type() == torch::kFloat32 && bias_t.is_contiguous();
  auto bT = given_t ? bias_t : torch::empty({B, H, S, S}, q.options().dtype(torch::kFloat32));
  CUtensorMap bkv = make_map<2>(bT.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {32, BN}, CU_TENSOR_MAP_SWIZZLE_128B, "bias^T",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap bq = make_map<2>(b16.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {QB_KW, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "bias");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd_kv_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, KB_SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_bwd_q_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, QB_SMEM));
    attr = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sms = num_sms(q.device().index());
  const float scl = (float)scale * L2E;
  if (!given_t) {
    triattn_biasT<<<dim3(S / 32, S / 32, B * H), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(b16.data_ptr<at::BFloat16>()), bT.data_ptr<float>(), (int)S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int n1 = (int)(B * H * (S / BN) * ((N + KR - 1) / KR));
  if (which != 2)
  launch_pdl_if(pdl && S == 128, triattn_bwd_kv_sm100, dim3(std::min(n1, sms)), dim3( KB_THREADS),  KB_SMEM,  stream, (int)N, (int)H, (int)S, n1, scl, (float)scale, q32, d32, k128, v128, bkv,
      lse.data_ptr<float>(), delta.data_ptr<float>(), dkm, dvm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (which != 1) {
  launch_pdl_if(pdl && S == 128, triattn_bwd_q_sm100, dim3(grid2), dim3( QB_THREADS),  QB_SMEM,  stream, (int)N, (int)H, (int)S, n2, scl, (float)scale, q128, d128, k64, v64, bq,
      lse.data_ptr<float>(), delta.data_ptr<float>(), dqm, dbm, per2, (int)(B * H * S));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (which == 1) return {dq, dk, dv, db};
  // raw_slabs: the slabs come back unsummed with (NG, per) on the host -- the caller's consumer sums them in the same order
  // (tri_head_bwd: the fused d_pair 128 path, one launch fewer)
  if (raw_slabs) return {dq, dk, dv, slabs, torch::tensor({(long)NGq, (long)per2}, torch::kLong)};
  if (db_rows.defined() && db_rows.numel() > 0) {   // (S90) the slab sum writes the head backward's dbias rows [R][64] directly (starting-node orientation)
    TORCH_CHECK(db_rows.scalar_type() == torch::kBFloat16 && db_rows.dim() == 2 && db_rows.size(0) == B * S * S && db_rows.size(1) == 64 && db_rows.stride(1) == 1 && db_rows.stride(0) % 8 == 0 && H <= 16,
                "db_rows: bf16 [B S^2, 64] rows at a 16-byte-multiple stride");
    const long n4 = B * H * S * S / 4, nbs = (n4 + 31) / 32, Rr = B * S * S;
    const long nzb = std::min<long>(1024, (Rr * 8 + 255) / 256);
    launch_pdl_if(pdl && S == 128, triattn_db_slabs_rows, dim3((unsigned)(nbs + nzb)), dim3(256), 0, stream, slabs.data_ptr<float>(),
                  reinterpret_cast<__nv_bfloat16*>(db_rows.data_ptr<at::BFloat16>()), (long)db_rows.stride(0), (int)H, (int)S, NGq, per2, B * H * S * S, n4, nbs, Rr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv, db};
  }
  {
    const long n4 = B * H * S * S / 4;
    launch_pdl_if(pdl && S == 128, triattn_db_slabs, dim3((unsigned)((n4 + 31) / 32)), dim3( 256),  0,  stream, slabs.data_ptr<float>(), db.data_ptr<float>(), (int)S, NGq, per2,
                                                                   B * H * S * S, n4);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd", &triattn_fwd, "sm100 triangle-attention forward (D = 32, bf16, no mask): q/k/v/out [B, N, S, H, 32], optional base-2 LSE",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("scale"), py::arg("want_lse") = false,
        py::arg("out"), py::arg("lse"));
#if TA_TL
  m.def("triattn_tl", &triattn_tl);
#endif
  m.def("triattn_delta", &triattn_delta_rows, "rowsum(dout o out) fp32 [B, N, H, S]");
  m.def("triattn_bwd", &triattn_bwd, "sm100 triangle-attention backward: (dq, dk, dv [B, N, S, H, 32] bf16, dbias fp32 [B, H, S, S])",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("dout"), py::arg("lse"), py::arg("delta"), py::arg("scale"),
        py::arg("db_ws"), py::arg("dq_out"), py::arg("dk_out"), py::arg("dv_out"), py::arg("bias_t") = torch::Tensor(), py::arg("raw_slabs") = false, py::arg("pdl") = false, py::arg("which") = 0, py::arg("db_rows") = torch::Tensor());
}
