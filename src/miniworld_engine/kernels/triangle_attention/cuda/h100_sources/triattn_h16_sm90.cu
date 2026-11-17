// Triangle-attention core for H100 (sm_90a), inference, fp16 operands: D = 32, q / k / v / sbias fp16, out bf16.
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]
//
// The schedule of triattn_sm90.cu: a CTA task = one (b, h, 128-query tile) x 2 pair rows; warpgroup 0 = TMA producer (one thread),
// warpgroups 1 / 2 = consumers of queries 0-63 / 64-127 of both rows; per 128-key tile the bias tile is staged once for both rows; a
// consumer alternates the rows (G = QK^T of one row issued, PV of the other row issued, X = the exponentials of the first while that PV runs).
// The score GEMM accumulates in fp16 (the accumulator registers are fp16 pairs), which takes most of the per-logit work off the loop
// (it is issue- and latency-bound: the MUFU runs ~55 % of the time in the bf16 kernel):
//   - the bias fragment (ldmatrix of the fp16 sbias tile) IS the accumulator's initial value: no bf16 -> fp32 unpack per logit;
//   - x = s * scale log2(e) - m (HFMA2) and p = 2^x (MUFU.EX2 .f16x2) run on pairs;
//   - p is already the A fragment of [O | l] += P [V | 1] (fp16 x fp16 -> fp32): no fp32 -> 16-bit pack;
//   - the offset (TA_SPEC 2): the row maximum is taken only when needed -- a row without a finite logit yet, or a warp one of whose lanes sees a
//     logit past its row's offset + TA_TAU log2 units (each thread's own maximum, a depth-4 tree of HMNMX2, and one warp vote); the shuffles of
//     the full row maximum were a third of the exponential phase.  The offset rises lazily (o and l rescaled), so p <= 2^(TA_TAU + 1) always:
//     fp16 cannot overflow.  TA_SPEC 1 (as in triattn_sm90.cu: no maximum after a row's first tile, the row-sum increment checked afterwards,
//     a task with an fp16 overflow recomputed by a second, exact launch) is ~8 % faster on random inputs, but on Protenix trunk inputs it
//     recomputed ~5 % of the tasks and came out level with TA_SPEC 2 (2026-10-22).
// Precision: q / k / v in fp16 carry 3 more mantissa bits than bf16 (the module's front rounds them from fp32 either way); the score sum is
// rounded to fp16 (11 bits) once per k16 step, P as well (bf16 P has 8); measured against an fp32 reference about a third of the bf16
// core's error.  Range: |q . k + bias / scale| < 65504; masked keys -inf.  Offsets are fp16 values (the exponentials subtract exactly them).
// Heads of 16 channels (DH 16): q / k / v / gate / out / u are compact [.., H, 16] in global memory; the loads read them through a 3-D map
// [rows][H][16] whose 32-channel box zero-fills the 16 channels past the dimension, so the math is the D = 32 kernel's on zero-padded heads
// (bit-identical results) at half the bytes through the front's stores, these loads and the out-projection.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include "sm90.cuh"
using namespace sm90;

#ifndef TA_NKV
#define TA_NKV 4          // K | V ring stages (one per (key tile, row))
#endif
#ifndef TA_NB
#define TA_NB 2           // bias ring stages (one per key tile)
#endif
#ifndef TA_ORDER1_MINS
#define TA_ORDER1_MINS 2048   // from this S on the CTAs running together work on one head (its bias stays L2-resident across the row groups)
#endif
#ifndef TA_BIAS_L2
#define TA_BIAS_L2 (20 << 20) // below TA_ORDER1_MINS: the CTAs running together work on the largest group of heads whose bias (hg S^2 fp16) fits
#endif                        // this many bytes (all H heads up to 4 x 1536^2: 8 heads at L 1536 thrashed the L2 -- 32 GB of DRAM reads per call)
#ifndef TA_HG
#define TA_HG 0               // experiments: a fixed head group (a divisor of H) instead
#endif
#ifndef TA_F8PV
#define TA_F8PV 0         // opt-in FP8 variant (2026-11-14): P and V on the e4m3 tensor-core path (see the F8PV notes below); the extension is built with -DTA_F8PV=1
#endif
#ifndef TA_F8QK
#define TA_F8QK 0         // F8PV only (experimental, 2026-11-17): q and k arrive as e4m3 (per-head power-of-two scales sq, 1 / sq: their product is 1, so the bias and the softmax scale are unchanged); S += Q K^T is one m64n128k32 e4m3 wgmma with the fp16 accumulator (K tile 32-byte rows, 32-byte swizzle)
#endif
#ifndef TA_F8HL
#define TA_F8HL 0         // F8PV only: V as two e4m3 planes (hi + lo/16: the rounding of the first, again in e4m3), PV on m64n72k32 -- V error ~2^-8 instead of 2^-4
#endif
#if TA_F8PV
#define TA_TAU 0.5f       // F8PV: p = 2^(logit - m + 8): the largest probability is 2^8; p = 2^(x + 8) must stay below 448 (2^8.8): the offset rises as soon as a logit exceeds it by 0.5 log2 units
#endif
#ifndef TA_TAU
#define TA_TAU 8.f        // exact form: a tile raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
#ifndef TA_SPEC
#define TA_SPEC 2         // 2: the vote (see above); 1: the speculative offset with a fix list; 0: the full row maximum of every tile
#endif
#ifndef TA_SPEC_MINS
#define TA_SPEC_MINS 256  // TA_SPEC from this sequence length on
#endif
#ifndef TA_TS
#define TA_TS 8.f         // TA_SPEC 1: a tile whose row-sum increment exceeds 2^TA_TS raises the row's offset
#endif
#ifndef TA_MARGIN
#define TA_MARGIN 8.f     // TA_SPEC 1: the offset sits this many log2 units above the largest logit seen (room before an fp16 probability overflows;
                         // the small ones go subnormal: 12 already lost more accuracy than bf16)
#endif
#ifndef TA_DYN
#define TA_DYN 1          // dynamic task schedule (a global counter)
#endif
#ifndef KM_MAXB
#define KM_MAXB 64        // the most batch elements whose live key tiles the kernel counts from the key mask itself (KM)
#endif
namespace {
constexpr int D = 32, BM = 128, BN = 128, R = 2, NKV = TA_NKV, NB = TA_NB;
constexpr int QT = BM * D;                                 // Q tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][32], 64B swizzle (8 KiB)
constexpr int STG = 3 * KVT;                               // a K | V stage: K, V, ones (24 KiB)
constexpr int BT = BM * BN;                                // bias tile: fp16 [2 key halves][128 q][64], 128B swizzle (32 KiB)
constexpr int THREADS = 384;
constexpr int NO = TA_F8HL ? 36 : 20;                       // accumulator floats per row: [O | l] (n40), or [O_hi | O_lo | l] (n72)
constexpr int LC = TA_F8HL ? 32 : 16;                       // the row sums' index (row g; row g + 8: LC + 2)
constexpr int VROWS = TA_F8HL ? 64 : 32;                    // V^T rows per (row n, head) in global memory
constexpr int QB = TA_F8QK ? 1 : 2;                          // bytes per q / k element (F8QK: e4m3)
constexpr int SMEM = 1024 + (NB * BT + R * QT + NKV * STG) * 2 + 256;
static_assert(SMEM <= 232448, "one CTA per SM");

#define H16_U8(b) "+r"(d[b]), "+r"(d[b + 1]), "+r"(d[b + 2]), "+r"(d[b + 3]), "+r"(d[b + 4]), "+r"(d[b + 5]), "+r"(d[b + 6]), "+r"(d[b + 7])
// S += Q . K^T with an fp16 accumulator: m64n128k16, A = Q from registers (fp16 pairs), B = K from shared (K-major).  d = 32 fp16 pairs:
// d[2 i + h] = row 8 h + lane / 4 of the warp's 16, columns 8 i + 2 (lane % 4) + {0, 1}.
__device__ __forceinline__ void mma_h16_n128(uint32_t (&d)[32], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
               : H16_U8(0), H16_U8(8), H16_U8(16), H16_U8(24) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// F8QK: S += Q . K^T on e4m3, fp16 accumulator: m64n128k32, A = Q from registers (4 x 4 e4m3: the k16 fp16 fragment layout of the same bytes), B = K from shared (K-major, 32-byte rows)
__device__ __forceinline__ void mma_f8qk_n128(uint32_t (&d)[32], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
               : H16_U8(0), H16_U8(8), H16_U8(16), H16_U8(24) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// [O | l] += P [V | 1]: m64n40k16, fp16 x fp16 -> fp32, A = P from registers, B = [V | ones] from shared (MN-major).
__device__ __forceinline__ void mma_h16_n40(float (&d)[20], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n40k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
               "%15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 1;\n}\n"
               : SM90_D16(0), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// [O | l] += P [V | 1] for 16-channel heads: m64n24k16 (columns 0..15 = V, 16..23 = ones).
__device__ __forceinline__ void mma_h16_n24(float (&d)[12], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 1;\n}\n"
               : SM90_D8(0), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// ablation only (TA_ABL 1024): the 32-channel PV without the ones block (m64n32, row sums stay 0 -- WRONG results, time only)
__device__ __forceinline__ void mma_h16_n32(float (&d)[20], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 " SM90_L16 ", {%16, %17, %18, %19}, %20, p, 1, 1, 1;\n}\n"
               : SM90_D16(0) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// F8PV: [O | l] += P [V | 1] on e4m3: m64n40k32, fp32 accumulation, A = P from registers (4 x 4 e4m3), B = [V^T; ones] K-major from shared (40 rows of 128 keys).
__device__ __forceinline__ void mma_f8_n40(float (&d)[20], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
               "%15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1;\n}\n"
               : SM90_D16(0), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// F8PV, heads of 16 channels (TA_F8N24): [O | l] += P [V | 1] as m64n24k32 (16 value channels + one 8-column ones block), fp32 accumulation
__device__ __forceinline__ void mma_f8_n24(float (&d)[12], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1;\n}\n"
               : SM90_D8(0), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
#if TA_F8HL
__device__ __forceinline__ void mma_f8_n72(float (&d)[36], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\n}\n"
               : SM90_D32(0), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
#endif
__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) { uint32_t r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t hex2(uint32_t a) { uint32_t r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(a)); return r; }
// 2^x on f16x2 pairs without the MUFU (FMA / ALU pipes): x clamped at -15 (results below 2^-14.5 flush to 0), j = round(x) by the 1551 shift,
// f = x - j in [-0.5, 0.5], 2^f by a degree-3 polynomial (Chebyshev-economised Taylor, ~1.2e-4 relative before the fp16 roundings), times 2^j
// whose fp16 exponent field is the low bits of the shifted sum (15 + j for j >= -14, 0 for j = -15).
__device__ __forceinline__ uint32_t hex2_poly(uint32_t x) {
  uint32_t t, jf, f, p, s, r;
  asm("max.f16x2 %0, %1, %2;" : "=r"(x) : "r"(x), "r"(0xCB80CB80u));                 // -15
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(t) : "r"(x), "r"(0x660F660Fu));              // + 1551: rounds x to an integer
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(jf) : "r"(t), "r"(0xE60FE60Fu));             // - 1551 (exact)
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(f) : "r"(x), "r"(jf));
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(p) : "r"(0x2B1B2B1Bu), "r"(f), "r"(0x33C433C4u));   // c3 f + c2
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(p) : "r"(p), "r"(f), "r"(0x398C398Cu));            // .. f + c1
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(p) : "r"(p), "r"(f), "r"(0x3C003C00u));            // .. f + c0
  s = (t << 10) & 0x7C007C00u;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(p), "r"(s));
  return r;
}
#ifndef TA_PP
#define TA_PP 0           // 1: FlashAttention-3 pingpong: the two consumer warpgroups issue their wgmma groups strictly alternately (named barriers 2 / 3), so one warpgroup
                          // exponentiates while the other's tensor work runs.  Pure scheduling (bit-identical).
#endif
#ifndef TA_ONE
#define TA_ONE 0          // compile experiments: instantiate one kernel only
#endif
#ifndef TA_FR
#define TA_FR 0           // 1: the bias stage is released after the QK^T wgmma issue instead of before it (fence.proxy + arrive off the issue path).  Scheduling only (bit-identical).
#endif
#ifndef TA_SW
#define TA_SW 2           // software pipelined sub-step (step_sw: the QK^T of the next sub-step is issued right after the PV of this one retired, the exponentials of this one are all done before): 0 off, 1 all heads, 2 the 16-channel heads only (default: -4 .. -7 % on them, level on 32-channel heads, bit-identical)
#endif
#ifndef TA_SWF
#define TA_SWF 1
#endif
#ifndef TA_SWN
#define TA_SWN 0          // TA_SW: pairs (of 16) exponentiated before the next QK^T is issued (> 0: ptxas serialises the wgmma stage, 17-20 % slower)
#endif
#ifndef TA_N24
#define TA_N24 1          // 16-channel heads: [O | l] += P [V | 1] as m64n24 (16 value channels + one 8-column ones block, V and ones in 32-byte-swizzle MN-major atoms) instead of n40 (32 zero-padded channels + ones).  Bit-identical (the extra columns were exact zeros)
#endif
#ifndef TA_K16
#define TA_K16 1          // 16-channel heads with TA_N24: K read as the 16 channels only (32-byte swizzle, 4 KiB instead of 8 KiB per stage); the k16 step reads the same values (bit-identical)
#endif
#ifndef TA_F8N24
#define TA_F8N24 1        // F8PV, heads of 16 channels: PV as m64n24k32 (the 16 zero-padded channels and their MMA columns dropped: bit-identical)
#endif
#ifndef TA_PVH
#define TA_PVH 0          // TA_SW: the first TA_PVH of the 8 k16 steps of a row's PV are issued as soon as their probabilities are done (under the exponentials of the rest and the next QK^T), the others at the next sub-step. Same accumulation order (bit-identical)
#endif
#ifndef TA_ABL
#define TA_ABL 0         // ablation bit mask (time only, WRONG results): 1 no QK wgmma, 2 no PV wgmma (the probabilities stay live), 4 no exp, 8 no scale/offset HFMA2, 16 no max machinery, 32 no bias ldmatrix, 128 no X phase
#endif
#ifndef TA_POLY
#define TA_POLY 8         // pairs i (of 16 per row) whose exponentials run as hex2_poly: TA_POLY of every 16 (1, 2, 4, 8 or 16), the rest on the
                          // MUFU.  8 (every other pair; sustained, power-capped, c256 / 8 heads): L512 level, 768 / 1024 -0.5 %, 1536 -2.7 %,
                          // 2048 -3.3 %, accuracy level; 2 / 4: level or +1.5 %; 16 (all): +12 %
#endif
#ifndef TA_POLY16
#define TA_POLY16 8       // as TA_POLY for the heads of 16 channels (d_pair 64 x 4 heads): every other pair on the polynomial (0 = all on the MUFU).
                          // Their score step is short, but half the pairs on the FMA pipes is -7 .. -12 % of the core at H4 L2048 (2026-11-16);
                          // 4 / 16: level or slower
#endif
#ifndef TA_RC
#define TA_RC 1           // 1 (2.5.13): the last key tile peeled out of the consumer loop, the gate prefetch after the loop (see the loop).  Same
                          // operations in the same order: bit-identical to 0 (the 2.5.12 loop).  Sustained core 64:4:16 +7-11 %, 32-channel heads 0-+5 %
#endif
template <int NP> __device__ __forceinline__ uint32_t hexp(uint32_t a, int i) {   // NP of every 16 pairs on the polynomial (1, 2, 4, 8 or 16; 0: none)
  if (TA_ABL & 4) return a;
  if (NP > 0 && i % (16 / (NP > 0 ? NP : 1)) == (16 / (NP > 0 ? NP : 1)) - 1) return hex2_poly(a);
  return hex2(a);
}
__device__ __forceinline__ float hlo(uint32_t a) { return __half2float(__ushort_as_half((unsigned short)(a & 0xffffu))); }
__device__ __forceinline__ float hhi(uint32_t a) { return __half2float(__ushort_as_half((unsigned short)(a >> 16))); }
__device__ __forceinline__ uint32_t hdup(float x) { const unsigned short h = __half_as_ushort(__float2half_rn(x)); return (uint32_t)h | ((uint32_t)h << 16); }
__device__ __forceinline__ float hround(float x) { return __half2float(__float2half_rn(x)); }
}  // namespace

#ifndef TA_CL
#define TA_CL 0           // 1: clusters of two CTAs on adjacent query tiles of the same (b, h, row group): the K / V tiles are loaded once for both (rank 0 loads K, rank 1 loads V,
                          // each multicast to both), the stage is released by both CTAs' consumers (remote arrive).  The data and every arithmetic op are the same (bit-identical);
                          // only the L2 -> SM traffic of K / V halves.  32-channel heads, S % 256 == 0.
#endif
namespace {
__device__ __forceinline__ uint32_t cta_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory"); }
__device__ __forceinline__ void arrive_rank(uint32_t bar, uint32_t rank) {    // arrive on the barrier at the same offset in CTA `rank`
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}\n" :: "r"(bar), "r"(rank) : "memory");
}
__device__ __forceinline__ void arrive_rank_cta(uint32_t bar, uint32_t rank) {   // as CUTLASS' ClusterBarrier::arrive(cta_id): the default (cta-scope release) semantics
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n" :: "r"(bar), "r"(rank) : "memory");
}
__device__ __forceinline__ void st_remote_i32(uint32_t addr, uint32_t rank, int v) {
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nst.shared::cluster.s32 [ra], %2;\n}\n" :: "r"(addr), "r"(rank), "r"(v) : "memory");
}
__device__ __forceinline__ void wait_cl(uint32_t bar, uint32_t parity) {   // acquire at cluster scope (the remote store before the remote arrive is visible)
  asm volatile("{\n.reg .pred p;\nWAITC_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra WAITC_%=;\n}\n" :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_2d_mc(const void* map, uint32_t dst, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
}  // namespace

// SPEC: the speculative offset (fix list written to FIX); else the exact form, which with FIX set runs the tasks of that list (static schedule).
// ORD1: task order query tile, head in its group of hg, row group, head group (the CTAs running together share the bias of hg heads; hg = 1:
// one head), else query tile, head, row group.
template <bool SPEC, bool ORD1, int DH, bool POLY, bool BTR, bool CL = false>   // CL: see TA_CL.  BTR: the bias stored [B, H, key, query] (AF3's ending node from the one-kernel front), read transposed; POLY: TA_POLY (32-channel heads) / TA_POLY16 (16-channel heads) of every 16 exponential pairs on the FMA pipes; DH: channels per head in q / k / v / gate / out / U: 32, or 16 (compact; the core pads to 32)
__global__ void __launch_bounds__(THREADS, 1) triattn_h16_sm90(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    int hg,                                              // ORD1: heads per group (a divisor of H)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][H*32] fp16 (rows may sit at a wider stride), box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // sbias [B*H*S][S] fp16, box (64, 128), 128B swizzle
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*32] contiguous, or nullptr (only U is wanted)
    float* __restrict__ LSE,                             // [B, N, H, S] base-2 log-sum-exp, or nullptr
    int* __restrict__ FLAGS,
    const __nv_bfloat16* __restrict__ GATE,              // optional gate g (rows at stride ldg): U = sigmoid(g) o bf16(O), or nullptr; bf16, or fp16 bits
    int gate_h16,                                        // (gate_h16: the width-generic front's fp16 GEMM output)
    __nv_bfloat16* __restrict__ U, long ldg,
    int* __restrict__ SCHED,                             // zeroed task counter (dynamic schedule), or nullptr: static
    int* __restrict__ FIX,                               // fix list: FIX[0] = count (zeroed), FIX[1..] = task ids (written by SPEC, read by the exact form)
    const int* __restrict__ KT,                          // live key tiles of each batch element [B] (every key past them is masked in every row), or nullptr: all
    const unsigned char* __restrict__ KM
#if TA_F8PV
    , const __grid_constant__ CUtensorMap vtmap,        // V^T e4m3 [B*N*H*32 rows][S keys] (keys permuted inside each group of 32: see vt_perm), box (128, 32), 128B swizzle
    const float* __restrict__ VS                         // per (batch, head) scale of V^T: V = V^T-values * VS [B*H]
#endif
  ) {              // or the key mask itself (bool [B][S], 16-byte aligned, B <= KM_MAXB): the live key tiles
                                                         // counted by the producer warpgroup once per CTA (no glue kernels before the launch)
  constexpr bool VOTE = SPEC && TA_SPEC == 2;    // TA_SPEC 2: no fix list (see X)
  constexpr bool N24 = (DH != D) && TA_N24 && !TA_F8PV;       // PV as m64n24 (see TA_N24); the V tile is then 16 channels in the 32-byte swizzle
  constexpr bool K16 = N24 && TA_K16;   // (F8PV: dh 16 keeps the zero-padded 32-channel K and the 3-D V^T box)
  constexpr bool F8N24 = TA_F8PV && !TA_F8HL && TA_F8N24 && DH != D;   // F8PV, heads of 16: PV as m64n24 (V^T rows 0-15, ones at rows 16-23)
  constexpr int OW = TA_F8PV ? (F8N24 ? 12 : NO) : ((DH == D || !N24) ? 20 : 12);     // accumulator registers per row: [O | l] columns, the row sums in the last two pairs
  constexpr int LS = TA_F8PV ? (F8N24 ? 8 : LC) : OW - 4;                       // o[r][LS] = row g sum, o[r][LS + 2] = row g + 8
#if TA_F8PV
  constexpr bool SWM = false;   // F8PV: no software-pipelined sub-step (tried, level or worse: docs/fp8.md); TA_SW does not apply
#else
  constexpr bool SWM = (TA_SW == 2) ? (DH != D) : (TA_SW == 1);   // the software-pipelined sub-step (step_sw): TA_SW 2 = the 16-channel heads only
#endif
  constexpr bool LIST = SPEC && !VOTE;
  const bool LISTM = !SPEC && FIX != nullptr;
  const int nt = LISTM ? *reinterpret_cast<volatile int*>(FIX) : ntasks;
  if (LISTM && blockIdx.x == 0 && threadIdx.x == 0 && nt > 0) atomicAdd(FLAGS + 1, nt);     // telemetry: tasks the exact launch recomputed
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  // shared layout: bias ring [NB][BT] | Q [R][QT] | K | V ring [NKV][K | V | ones] | barriers
  const uint32_t sB = sa(smb), sQ = sB + NB * BT * 2, sKV = sQ + R * QT * 2, bar0 = sKV + NKV * STG * 2;
  const uint32_t qf = bar0, qe = bar0 + 8;                  // the task's Q tiles landed / count 8: every consumer warp holds its Q fragments
  auto kvf = [&](int s) { return bar0 + 16 + 8 * s; };     // [NKV]
  auto kve = [&](int s) { return bar0 + 16 + 8 * (NKV + s); };                // count 8: the stage's MMAs retired in every consumer warp
  auto bfu = [&](int s) { return bar0 + 16 + 8 * (2 * NKV + s); };            // [NB]
  auto bem = [&](int s) { return bar0 + 16 + 8 * (2 * NKV + NB + s); };       // count 8: every consumer warp read its bias fragments
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  if (TA_ABL & (768 | 65536)) {   // traffic ablation: the bytes the skipped loads would have written stay zero
    for (int i = tid; i < (NB * BT + R * QT + NKV * STG) / 2; i += THREADS) reinterpret_cast<uint32_t*>(smb)[i] = 0u;
    __syncthreads();
  }
#if TA_F8PV
  for (int i = tid; i < NKV * 256; i += THREADS) {          // F8PV: the 8 ones rows (e4m3 1.0 = 0x38) behind each stage's 32 V^T rows (1 KiB)
    const int s = i / 256, e = i % 256;
    reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (s * STG + KVT) * 2 + 128 * (F8N24 ? 16 : VROWS))[e] = 0x38383838u;
  }
#else
  for (int i = tid; i < NKV * KVT / 2; i += THREADS) {     // the ones blocks: fp16 pairs (1, 1)
    const int s = i / (KVT / 2), e = i % (KVT / 2);
    reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (s * STG + 2 * KVT) * 2)[e] = 0x3C003C00u;
  }
#endif
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sB));
    bar_init(bars, 1); bar_init(bars + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(bars + 2 + k, 1); bar_init(bars + 2 + NKV + k, CL ? 16 : 8); }
    for (int k = 0; k < NB; ++k) { bar_init(bars + 2 + 2 * NKV + k, 1); bar_init(bars + 2 + 2 * NKV + NB + k, 8); }
    if (CL) bar_init(bars + 2 + 2 * NKV + 2 * NB, 1);                              // the task-pair hand-over from rank 0
    bar_init_fence();
    reinterpret_cast<int*>(smb + (bar0 - sB) + 244)[0] = 0; reinterpret_cast<int*>(smb + (bar0 - sB) + 244)[1] = 0;     // the fix-list counters
  }
  fence_proxy_async();
  if (CL) cluster_sync_all();                    // every barrier of the pair initialised before a remote arrive / multicast
  else __syncthreads();
  const uint32_t crank = CL ? cta_rank() : 0u;
  auto decode = [&](int t, int& b, int& h, int& qt, int& g) {
    qt = t % QTILES; int r = t / QTILES;
    if (ORD1) { const int hl = r % hg; r /= hg; g = r % NG; r /= NG; h = (r % (H / hg)) * hg + hl; b = r / (H / hg); }
    else        { h = r % H; r /= H; g = r % NG; b = r / NG; }
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row; a padding row reads row N - 1
  // dynamic schedule: the producer takes each next task from the global counter and hands it to the consumers in this slot (written before
  // the Q tiles' arrive, read after its completion; -1 ends the CTA)
  volatile int* slot = reinterpret_cast<volatile int*>(smb + (bar0 - sB) + 240);
  volatile int* nkslot = reinterpret_cast<volatile int*>(smb + (bar0 - sB) + 252);   // the task's key tiles, handed over the same way (KT: no global load on the consumers' path between tasks)

  // a head's [128 rows][32 channels] tile of q / k / v: 2-D at column h * 32, or (dh 16: heads of 16 channels stored compact) the 3-D map
  // [rows][H][16] whose box of 32 channels reads the head's 16 and zero-fills the 16 past the dimension (padded channels are 0, as the
  // padded weights made them)
  auto load_head = [&](const CUtensorMap* map, uint32_t dst, uint32_t bar, int h, int row) {
    if (DH == D) load_2d(map, dst, bar, h * D, row);
    else asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
                      :: "r"(dst), "l"(map), "r"(bar), "r"(0), "r"(h), "r"(row) : "memory");
  };
#if TA_F8PV
#define BIGV vtmap
#else
#define BIGV vmap
#endif
  __shared__ int kts[KM_MAXB];                       // KM: 1 + the last attended key of each batch element (0: none)
  if (wg == 0) {
    reg_dealloc<40>();
    int lt = 0, st = 0, kph = 0, bs = 0, bph = 0;
    // one key tile of a task: its bias tile and the K / V tiles of both rows
    auto issue_tile = [&](int b, int h, int qt, int g, int j) {
      wait(bem(bs), bph ^ 1);
      expect_tx(bfu(bs), (TA_ABL & 512) ? BT : BT * 2);
      for (int hh = 0; hh < ((TA_ABL & 512) ? 1 : 2); ++hh)   // key halves [128 queries][64 keys], or (BTR) query halves [128 keys][64 queries]
        if (BTR) load_2d(&bmap, sB + (bs * BT + hh * BM * 64) * 2, bfu(bs), qt * BM + hh * 64, (b * H + h) * S + j * BN);
        else load_2d(&bmap, sB + (bs * BT + hh * BM * 64) * 2, bfu(bs), j * BN + hh * 64, (b * H + h) * S + qt * BM);
      if (++bs == NB) { bs = 0; bph ^= 1; }
      for (int r = 0; r < R; ++r) {
        wait(kve(st), kph ^ 1);
        const bool skip = (TA_ABL & 256) && r == 1;
#if TA_F8PV
        expect_tx(kvf(st), ((TA_ABL & 65536) ? 0 : KVT * QB) + ((TA_ABL & 256) ? 0 : (F8N24 ? 16 : VROWS) * BN));   // (ablation only: 65536 no K loads, 256 no V^T loads, 512 half the bias)
#else
        expect_tx(kvf(st), skip ? 0 : (N24 ? (K16 ? KVT * 2 : KVT * 2 + KVT) : 2 * KVT * 2));
#endif
        const int rw = row0(b, g * R + r) + j * BN;
#if TA_F8PV   // (F8PV: no cluster pair, no ablation skips)
        if (!(TA_ABL & 65536)) load_head(&kmap, sKV + st * STG * 2, kvf(st), h, rw);
#pragma unroll
        for (int pl = 0; pl < (TA_ABL & 256 ? 0 : VROWS / 32); ++pl) {   // the V^T planes (F8HL: hi, lo) of the row: 32 rows each
          const int bnh = (b * N + min(g * R + r, N - 1)) * H + h;
          if (DH == D) load_2d(&vtmap, sKV + (st * STG + KVT) * 2 + pl * 4096, kvf(st), j * BN, (bnh * (VROWS / 32) + pl) * 32);
          else asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"   // (heads of 16: the box's rows 16 - 31 read as zeros)
                            :: "r"(sKV + (st * STG + KVT) * 2 + pl * 4096), "l"(&vtmap), "r"(kvf(st)), "r"(j * BN), "r"(0), "r"(bnh * (VROWS / 32) + pl) : "memory");
        }
#else
        if (CL) {   // rank 0: the K tile, rank 1: the V tile, each to both CTAs of the pair (both expect all 16 KiB on their own barrier)
          if (crank == 0) load_2d_mc(&kmap, sKV + st * STG * 2, kvf(st), h * D, rw, 3);
          else            load_2d_mc(&vmap, sKV + (st * STG + KVT) * 2, kvf(st), h * D, rw, 3);
        } else if (!skip) {
          load_head(&kmap, sKV + st * STG * 2, kvf(st), h, rw);
          load_head(&vmap, sKV + (st * STG + KVT) * 2, kvf(st), h, rw);
        }
#endif
        if (++st == NKV) { st = 0; kph ^= 1; }
      }
    };
    // KM: the first task's Q tiles and first key tile (needed whatever the count) are requested before the mask is read; its Q barrier
    // takes the bytes now and its arrival once the count is handed over (below), so the scan's load latency overlaps theirs
    const bool pre = KM != nullptr && (int)blockIdx.x < nt;
    if (tid == 0) prefetch_map(&qmap), prefetch_map(&kmap), prefetch_map(&BIGV), prefetch_map(&bmap);
    if (pre && tid == 0) {
      int b, h, qt, g; decode(LISTM ? FIX[1 + blockIdx.x] : (int)blockIdx.x, b, h, qt, g);
      asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" :: "r"(qf), "r"(R * QT * QB) : "memory");
      for (int r = 0; r < R; ++r) load_head(&qmap, sQ + r * QT * 2, qf, h, row0(b, g * R + r) + qt * BM);
      issue_tile(b, h, qt, g, 0);
    }
    if (KM != nullptr) {
      // what key_tiles() computes, by the 128 producer threads: 16-byte chunks of the mask, the last set byte of each, a max over the 8 lanes
      // of a 128-key group (S % 128 == 0: one batch element), one shared atomic per group
      const int nb = ntasks / (H * NG * QTILES), chunks = nb * S / 16;
      for (int b = tid; b < nb; b += 128) kts[b] = 0;
      named_sync(1, 128);
      for (int c0 = 0; c0 < chunks; c0 += 128) {
        const int c = c0 + tid;
        int last = 0;
        if (c < chunks) {
          const uint4 v = __ldg(reinterpret_cast<const uint4*>(KM) + c);
          const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (w4[i] != 0u) last = (c * 16) % S + 4 * i + (31 - __clz(w4[i])) / 8 + 1;
        }
        last = max(last, __shfl_xor_sync(0xffffffffu, last, 1));
        last = max(last, __shfl_xor_sync(0xffffffffu, last, 2));
        last = max(last, __shfl_xor_sync(0xffffffffu, last, 4));
        if ((tid & 7) == 0 && c < chunks && last > 0) atomicMax(&kts[(c * 16) / S], last);
      }
      named_sync(1, 128);
    }
    if (tid == 0) {
      int pph = 0;
      // next task; CL: the pair index is taken by rank 0 from the counter and handed to rank 1 (remote store + remote arrive), the pair's tasks are 2 u + rank
      auto next_task = [&](int t) -> int {
        if (!CL || !SCHED) return SCHED ? (int)gridDim.x + atomicAdd(SCHED, 1) : t + (int)gridDim.x;
        int u;
        if (crank == 0) { u = (int)(gridDim.x >> 1) + atomicAdd(SCHED, 1); st_remote_i32(bar0 + 232, 1, u); arrive_rank(bar0 + 16 + 8 * (2 * NKV + 2 * NB), 1); }
        else { wait_cl(bar0 + 16 + 8 * (2 * NKV + 2 * NB), pph); pph ^= 1; u = *reinterpret_cast<volatile int*>(smb + (bar0 - sB) + 232); }
        return 2 * u + (int)crank;
      };
      for (int t = blockIdx.x; t < nt; t = next_task(t), ++lt) {
        int b, h, qt, g; decode(LISTM ? FIX[1 + t] : t, b, h, qt, g);
        const int nk = KM != nullptr ? min(NK, max(1, (kts[b] + BN - 1) / BN))
                     : KT != nullptr ? min(NK, max(1, __ldg(KT + b))) : NK;   // the dead trailing key tiles are not streamed
        const bool first = pre && lt == 0;               // its Q tiles and key tile 0 already requested
        if (!first) wait(qe, (lt & 1) ^ 1);
        if (SCHED) *slot = t;
        if (KT != nullptr || KM != nullptr) *nkslot = nk;
        if (first) {
          arrive(qf);                                    // (the bytes were expected before the scan)
        } else {
          expect_tx(qf, R * QT * QB);
          for (int r = 0; r < R; ++r) load_head(&qmap, sQ + r * QT * 2, qf, h, row0(b, g * R + r) + qt * BM);
        }
        for (int j = first ? 1 : 0; j < nk; ++j) issue_tile(b, h, qt, g, j);
      }
      if (SCHED) {                                   // the end: once the consumers took the last task's Q, a task-free phase of qf
        wait(qe, (lt & 1) ^ 1);
        *slot = -1;
        arrive(qf);
      }
    }
    if (CL) cluster_sync_all();                    // no CTA of the pair leaves while the other may still multicast into it / arrive on it
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  if (TA_PP && w == 1) named_arrive(2, 256);       // warpgroup 0 issues first
  const int qrow = 64 * w + 16 * wi;               // this warp's first query row in the tile
  // ldmatrix lane geometry: matrices (rows +0 / +8) x (columns +0 / +8) of a 16 x 16 block
  const int lrow = qrow + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4, x7 = lane & 7;
  const uint32_t qlane = sQ + lrow * D * QB, blane = sB + lrow * 128;
  // BTR (the bias tile stored [key][query]: AF3's ending node, the pair-as-stored bias): query half w, key row 8 (lane / 16) + lane % 8 of a
  // 16-key block, the 16-byte chunk of queries 16 wi + 8 ((lane / 8) % 2) (128-byte swizzle: chunk ^ row % 8)
  const uint32_t tlane = sB + w * BM * 128 + (8 * lcol8 + x7) * 128 + (((2 * wi + ((lane >> 3) & 1)) ^ x7) << 4);
  const uint64_t kd0 = (K16 || TA_F8QK) ? gdesc(sKV, 16, 256, SW32) : desc_k64(sKV), vd0 = TA_F8PV ? desc_k128(sKV + KVT * 2) : (N24 ? gdesc(sKV + KVT * 2, KVT * 2, 256, SW32) : desc_mn64(sKV + KVT * 2, KVT * 2));   // built once: a wgmma operand costs one 32-bit add
  const uint32_t scl2 = hdup(scl);
  // cursors: the K | V stage and bias stage the next G phase reads; the stage whose PV was issued last (released once it retired)
  int gst = 0, gph = 0, bs = 0, bph = 0, xst = 0, bad = 0, ovf = 0;
  uint32_t qa[R][2][4];
  float o[R][OW], m[R][2];                          // [O | l] per row (columns 32..39 = the row sums: o[16] row g, o[18] row g + 8); the offsets
  float lprev[R][2];                                // the row sums before the tile in flight (the increment test of the speculative offset)
  uint32_t sc[R][32];                               // per row: the score accumulator (fp16 pairs), then its probabilities = the A fragments of its PV
  // G phase of (tile j, row r): bias fragment -> sc[r] (the fp16 pairs ldmatrix returns are the accumulator's), S += Q_r K^T issued
  auto G = [&](int r) {
    if (r == 0) wait(bfu(bs), bph);
    const uint32_t btile = blane + bs * BT * 2;
    const uint32_t ttile = tlane + bs * BT * 2;
#pragma unroll
    for (int i2 = 0; i2 < 8; ++i2) {
      uint32_t r4[4];
      if (TA_ABL & 32) { r4[0] = r4[1] = r4[2] = r4[3] = 0u; }
      else if (BTR) ldsm_x4_t(ttile + i2 * 16 * 128, r4);   // the [key][query] tile: the transposed 8 x 8 matrices are the same fragments
      else ldsm_x4(btile + (i2 >> 2) * BM * 128 + ((((2 * i2 + lcol8) & 7) ^ x7) << 4), r4);
      // r4 = (row g, block 2 i2), (row g + 8, block 2 i2), (row g, block 2 i2 + 1), (row g + 8, block 2 i2 + 1)
#pragma unroll
      for (int e = 0; e < 4; ++e) sc[r][4 * i2 + e] = r4[e];
    }
    auto relb = [&]() {
      if (!TA_FR) fence_regs(sc[r]);               // the ldmatrix data is in registers before the stage is released (an arrive does not
      fence_proxy_async();                          // wait for this thread's loads: the next TMA would race them)
      __syncwarp();
      if (lane == 0) arrive(bem(bs));
      if (++bs == NB) { bs = 0; bph ^= 1; }
    };
    if (r == R - 1 && !TA_FR) relb();
    wait(kvf(gst), gph);
    if (TA_PP) named_sync(2 + w, 256);              // pingpong: my turn on the tensor core
    wg_fence();
    if (TA_F8QK && !(TA_ABL & 1)) mma_f8qk_n128(sc[r], qa[r][0], desc_off(kd0, gst * STG * 2));
    if (!TA_F8QK && !(TA_ABL & 1)) mma_h16_n128(sc[r], qa[r][0], desc_off(kd0, gst * STG * 2));
    if (!TA_F8QK && DH == D && !(TA_ABL & 1)) mma_h16_n128(sc[r], qa[r][1], desc_off(kd0, gst * STG * 2 + 32));   // (heads of 16: channels 16-31 are zeros, their k16 step adds exact zeros)
    wg_commit();
    if (r == R - 1 && TA_FR) relb();                // TA_FR: the stage is released after the wgmma issue (the ldmatrix results were consumed by it)
    if (++gst == NKV) { gst = 0; gph ^= 1; }
  };
  // PV of the previous sub-step (row r, stage xst): [O_r | l_r] += P_r [V | 1], issued and committed
  auto PV = [&](int r, int lo = 0, int hi = BN / 16) {
    if (SWM && TA_SWF) wg_fence();                          // TA_SW: the probabilities (A) were written after the last fence, with the next QK^T in flight
#if TA_F8PV
    // P is already e4m3 bytes (X): the A fragment of a k32 step is sc[r][4 kk .. 4 kk + 3]; V^T rows are K-major, a k32 step = +32 bytes in the 128-byte row
#pragma unroll
    for (int kk = 0; kk < BN / 32; ++kk) {
      const uint32_t a[4] = {sc[r][4 * kk], sc[r][4 * kk + 1], sc[r][4 * kk + 2], sc[r][4 * kk + 3]};
      if (TA_ABL & 2) { asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])); continue; }   // PV ablated (time only)
#if TA_F8HL
      mma_f8_n72(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 32));
#else
      if constexpr (F8N24) mma_f8_n24(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 32)); else mma_f8_n40(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 32));
#endif
    }
#else
#pragma unroll
    for (int kk = lo; kk < hi; ++kk) {
      const uint32_t a[4] = {sc[r][4 * kk], sc[r][4 * kk + 1], sc[r][4 * kk + 2], sc[r][4 * kk + 3]};
      if (TA_ABL & 2) asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));   // PV ablated: the probabilities stay live (else ptxas drops the exponentials too)
      if (!(TA_ABL & 2)) { if constexpr (N24) mma_h16_n24(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 512)); else if ((TA_ABL & 1024) != 0) mma_h16_n32(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 1024)); else mma_h16_n40(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 1024)); }
    }
#endif
    wg_commit();
  };
  auto release = [&]() {
    __syncwarp();
    if (lane == 0) { arrive(kve(xst)); if (CL) arrive_rank_cta(kve(xst), crank ^ 1u); }
    if (++xst == NKV) xst = 0;
  };
  // X phase of row r on sc[r] (its QK^T retired; o[r] idle: the PV in flight is the other row's): the offsets, the exponentials in place.
  // Exact form, and a row without a finite logit yet: the tile's row maximum (a tree of HMNMX2) raises the offset lazily.
  uint32_t nmr0 = 0, nmr1 = 0;                       // the offsets of the row in X (set by Xpre, used by Xexp)
  uint32_t nbr0 = 0, nbr1 = 0;                       // F8PV: the same, shifted by +8 (8 - m)
#if TA_F8PV
  // F8PV: the probabilities of the k32 groups [k0, k1) of row r (4 pairs = 8 registers each) as e4m3 bytes, in place: the A fragment of that group's PV step ends up in sc[r][4 kk .. 4 kk + 3]
  // (groups in increasing order: a group writes registers below the inputs of every later group).  p = 2^(x + 8) from the logit: see Xpre.
  // (ablation only, TA_ABL: 4 the whole conversion, 8 the scale / offset HFMA2, 4096 the clamp, 8192 the 8 x + 56 HFMA2, 2048 the PRMT pack; the probabilities stay live)
  auto F8conv = [&](int r, int k0, int k1) {
    if (TA_ABL & (4 | 128)) return;
#pragma unroll
    for (int kk = k0; kk < k1; ++kk) {
#pragma unroll
      for (int i = 4 * kk; i < 4 * kk + 4; ++i) {
        uint32_t x0 = (TA_ABL & 8) ? sc[r][2 * i] : hfma2(sc[r][2 * i], scl2, nbr0), x1 = (TA_ABL & 8) ? sc[r][2 * i + 1] : hfma2(sc[r][2 * i + 1], scl2, nbr1);
        if (!(TA_ABL & 4096)) { x0 = hmax2(x0, 0xC700C700u); x1 = hmax2(x1, 0xC700C700u); }   // x >= -7
        if (!(TA_ABL & 8192)) { x0 = hfma2(x0, 0x48004800u, 0x64386438u); x1 = hfma2(x1, 0x48004800u, 0x64386438u); }                           // 8 x + 56 + 1024
        sc[r][2 * i] = x0; sc[r][2 * i + 1] = x1;
      }
      if (!(TA_ABL & 2048)) {
        uint32_t pk[4];   // k32 step kk: a0 = row g keys (blocks 4 kk, 4 kk + 1), a1 = row g + 8, a2 / a3 = blocks 4 kk + 2, 4 kk + 3
#pragma unroll
        for (int j = 0; j < 4; ++j) asm("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(pk[j]) : "r"(sc[r][8 * kk + (j & 1) + 4 * (j >> 1)]), "r"(sc[r][8 * kk + (j & 1) + 4 * (j >> 1) + 2]));
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[r][4 * kk + j] = pk[j];
      }
    }
  };
#endif
  auto Xpre = [&](int r) {
    if (TA_ABL & 128) return;
    uint32_t t0[8], t1[8];
    auto tree = [&]() {                            // this thread's maximum of each of its two rows (no shuffle): a depth-4 tree of HMNMX2
#pragma unroll
      for (int i = 0; i < 8; ++i) { t0[i] = hmax2(sc[r][4 * i], sc[r][4 * i + 2]); t1[i] = hmax2(sc[r][4 * i + 1], sc[r][4 * i + 3]); }
#pragma unroll
      for (int n = 4; n >= 1; n >>= 1)
#pragma unroll
        for (int i = 0; i < n; ++i) { t0[i] = hmax2(t0[i], t0[i + n]); t1[i] = hmax2(t1[i], t1[i + n]); }
    };
    bool exact;
    if (TA_ABL & 16) exact = false;
    else if (!SPEC) { tree(); exact = true; }
    else if (VOTE) {                               // the exact path only when a lane of the warp sees a logit past its row's offset + TA_TAU (or a row has none yet)
      tree();
      if (TA_ABL & 131072) { asm volatile("" :: "r"(t0[0]), "r"(t1[0])); exact = false; } else {   // ablation only: the tree alone
      const float p0 = fmaxf(hlo(t0[0]), hhi(t0[0])) * scl, p1 = fmaxf(hlo(t1[0]), hhi(t1[0])) * scl;
      exact = __any_sync(0xffffffffu, !(m[r][0] > -INFINITY) | !(m[r][1] > -INFINITY) | (p0 > m[r][0] + TA_TAU) | (p1 > m[r][1] + TA_TAU));
      if (TA_ABL & 32768) { asm volatile("" :: "r"((int)exact)); exact = false; }   // ablation only: the vote runs, the exact body never
      }
    } else {
      exact = __any_sync(0xffffffffu, !(m[r][0] > -INFINITY) | !(m[r][1] > -INFINITY));
      if (exact) tree();
    }
    if (exact) {
      float m0, m1;
      m0 = fmaxf(hlo(t0[0]), hhi(t0[0])); m1 = fmaxf(hlo(t1[0]), hhi(t1[0]));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m0 *= scl; m1 *= scl;                         // -inf when the tile's 128 keys are all masked
      const bool u0 = m0 > m[r][0] + TA_TAU, u1 = m1 > m[r][1] + TA_TAU;     // m = -inf until a finite logit was seen
      if (__any_sync(0xffffffffu, u0 | u1)) {
        const float n0 = u0 ? hround(m0 + (LIST ? TA_MARGIN : 0.f)) : m[r][0], n1 = u1 ? hround(m1 + (LIST ? TA_MARGIN : 0.f)) : m[r][1];
        const float f0 = u0 ? ex2f(m[r][0] - n0) : 1.f, f1 = u1 ? ex2f(m[r][1] - n1) : 1.f;     // 2^(-inf) = 0 scales an empty sum
#pragma unroll
        for (int i = 0; i < OW / 4; ++i) { o[r][4 * i] *= f0; o[r][4 * i + 1] *= f0; o[r][4 * i + 2] *= f1; o[r][4 * i + 3] *= f1; }
        m[r][0] = n0; m[r][1] = n1;
        if (LIST) { lprev[r][0] = o[r][LS]; lprev[r][1] = o[r][LS + 2]; }
      }
    }
    nmr0 = hdup(m[r][0] > -INFINITY ? -m[r][0] : 0.f); nmr1 = hdup(m[r][1] > -INFINITY ? -m[r][1] : 0.f);
#if TA_F8PV
    // p = 2^(x + 8) directly as e4m3 bits: byte = round(8 (x + 8) + 56) - 64 ... = round(8 x + 56) with x already shifted by +8 (nm = 8 - m): the exponent field is the integer part
    // of the logit, the 3 mantissa bits the fraction (2^f ~ 1 + f, at most 6 % high mid-octave -- a smooth shape error the quantization noise already exceeds); logits below -7 give 0.
    // The magic add leaves the integer in the low byte of each half of the fp16 pair (1024 + y): two HFMA2 and one PRMT per four probabilities, no MUFU
    // (F8PV does the whole row here; Xexp is empty)
    nbr0 = hdup(m[r][0] > -INFINITY ? 8.f - m[r][0] : 0.f); nbr1 = hdup(m[r][1] > -INFINITY ? 8.f - m[r][1] : 0.f);
#endif
  };
  // the exponentials of the pairs [lo, hi) of row r (in place), with the offsets Xpre left
  auto Xexp = [&](int r, int lo, int hi) {
#if TA_F8PV
    F8conv(r, lo / 4, hi / 4);                       // F8PV: the e4m3 probabilities of the k32 groups [lo / 4, hi / 4) (the pairs are counted in 16ths of the row)
    return;
#endif
    if (TA_ABL & 128) return;
#pragma unroll
    for (int i = lo; i < hi; ++i) { sc[r][2 * i] = hexp<POLY ? (DH == D ? TA_POLY : TA_POLY16) : 0>((TA_ABL & 8) ? sc[r][2 * i] : hfma2(sc[r][2 * i], scl2, nmr0), i); sc[r][2 * i + 1] = hexp<POLY ? (DH == D ? TA_POLY : TA_POLY16) : 0>((TA_ABL & 8) ? sc[r][2 * i + 1] : hfma2(sc[r][2 * i + 1], scl2, nmr1), i); }
  };
  auto X = [&](int r) { Xpre(r); Xexp(r, 0, 16); };
  // after the PV of a tile of row rr retired: its row-sum increment is at least its largest probability; past 2^TA_TS the offset rises by the
  // power of two that brings it back to about 1 (o and l scaled exactly); a non-finite one (an fp16 probability overflowed) cannot be repaired
  // here: the task goes on the fix list
  auto check = [&](int rr) {
    const float i0 = o[rr][LS] - lprev[rr][0], i1 = o[rr][LS + 2] - lprev[rr][1];
    const float lim = exp2f(TA_TS);
    if (__any_sync(0xffffffffu, !(i0 <= lim) | !(i1 <= lim))) {
      const bool b0 = !(i0 <= lim), b1 = !(i1 <= lim);
      const bool f0 = b0 && i0 < 3.0e38f, f1 = b1 && i1 < 3.0e38f;
      ovf |= (b0 && !f0) || (b1 && !f1);
      const float n0 = f0 ? hround(m[rr][0] + ceilf(__log2f(i0)) + TA_MARGIN) : m[rr][0], n1 = f1 ? hround(m[rr][1] + ceilf(__log2f(i1)) + TA_MARGIN) : m[rr][1];
      const float g0 = ex2f(m[rr][0] - n0), g1 = ex2f(m[rr][1] - n1);
#pragma unroll
      for (int i = 0; i < OW / 4; ++i) { o[rr][4 * i] *= g0; o[rr][4 * i + 1] *= g0; o[rr][4 * i + 2] *= g1; o[rr][4 * i + 3] *= g1; }
      m[rr][0] = n0; m[rr][1] = n1;
    }
    lprev[rr][0] = o[rr][LS]; lprev[rr][1] = o[rr][LS + 2];
  };
  // one pipelined sub-step (tile j, row r): QK^T of row r and PV of the previous sub-step (row 1 - r) issued, the exponentials of row r
  // run while that PV executes; nothing is left in flight at the end
  auto step = [&](int r) {
    G(r);
    PV(1 - r);
    if (TA_PP) named_arrive(2 + (w ^ 1), 256);      // .. then the other warpgroup's
    wg_wait<1>();
    fence_regs(sc[r]);
    X(r);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    if (LIST) check(1 - r);
    release();
  };
  // TA_SW (software pipelined sub-step; scheduling only, bit-identical; EXPERIMENT, not faster -- see docs of the 2026-11-14 SOL run): on entry the QK^T of (tile,
  // row r) has landed.  PV of the previous row is issued, the tree / vote / offsets of row r and the first TA_SWN pairs of its exponentials run while that PV
  // executes, then (PV retired, stage released) the QK^T of the NEXT sub-step is issued into the other row's scores and the remaining pairs run under it; it is
  // waited for at the end, before the next sub-step's PV is issued.  ptxas serialises the wgmma stage (C7515) whenever non-wgmma code writes the exponentials
  // while a PV is in flight and the next QK^T follows it (TA_SWN > 0): 17-20 % slower; TA_SWN = 0 compiles clean and is level with the default.
  auto step_sw = [&](auto FIRST, auto LAST, int r) {
    constexpr bool last = decltype(LAST)::value;
    constexpr bool first = decltype(FIRST)::value;
    constexpr int PVH = (SWM && TA_PVH > 0 && TA_PVH < BN / 16) ? TA_PVH : 0;
    constexpr int SWNP = TA_SWN;
    if (!first) PV(1 - r, PVH);
    Xpre(r);
    Xexp(r, 0, SWNP);
    if (!first) {
      wg_wait<0>();
      fence_regs(o[0]); fence_regs(o[1]);
      if (LIST) check(1 - r);
      release();
    }
    if (!last) G(1 - r);
    if (PVH > 0 && !last) {
      Xexp(r, SWNP, 2 * PVH);
      PV(r, 0, PVH);                                    // the first part of this row's PV, queued behind the QK^T
      Xexp(r, 2 * PVH, 16);
      wg_wait<1>(); fence_regs(sc[1 - r]);               // the QK^T retired (groups retire in order)
    } else {
      Xexp(r, SWNP, 16);
      if (!last) { wg_wait<0>(); fence_regs(sc[1 - r]); }
    }
  };
  int lt = 0;
  for (int t = blockIdx.x;; t += SCHED ? 0 : (int)gridDim.x, ++lt) {
    if (SCHED == nullptr && t >= nt) break;
    wait(qf, lt & 1);
    if (SCHED) {
      t = *slot;
      if (t < 0) break;
    }
    int b, h, qt, g; decode(LISTM ? FIX[1 + t] : t, b, h, qt, g);
    const int nk = (KT != nullptr || KM != nullptr) ? *nkslot : NK;   // (the producer's count: it streams these tiles)
    // Q fragments of the warp's 16 rows (A of the QK^T wgmma), two k16 blocks per pair row
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        if (TA_F8QK) {   // e4m3 Q: [128 q][32 bytes], 32-byte swizzle (chunk ^ (row / 4) % 2); one k32 fragment
          if (ks == 0) ldsm_x4(qlane + r * QT * 2 + ((lcol8 ^ ((lrow >> 2) & 1)) << 4), qa[r][0]);
          else qa[r][1][0] = qa[r][1][1] = qa[r][1][2] = qa[r][1][3] = 0u;
        } else ldsm_x4(qlane + r * QT * 2 + (((2 * ks + lcol8) ^ ((lrow >> 1) & 3)) << 4), qa[r][ks]);
      }
#pragma unroll
    for (int r = 0; r < R; ++r) { fence_regs(qa[r][0]); fence_regs(qa[r][1]); }   // loaded before sQ is released
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) arrive(qe);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int i = 0; i < OW; ++i) o[r][i] = 0.f;
      m[r][0] = m[r][1] = -INFINITY;               // no finite logit seen yet
      lprev[r][0] = lprev[r][1] = 0.f;
    }
    ovf = 0;
    G(0);                                            // sub-step (0, 0): no previous PV
    if (TA_PP) named_arrive(2 + (w ^ 1), 256);
    wg_wait<0>();
    fence_regs(sc[0]);
    if (!SWM) X(0);
    uint4 gpre[R][2];                               // the gate's values of this lane's output columns, loaded under the last key tile
    if (SWM) step_sw(std::true_type{}, std::false_type{}, 0);
    auto gate_pre = [&]() {
      if (GATE != nullptr) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int n = min(g * R + r, N - 1);
#pragma unroll
          for (int half = 0; half < 2; ++half)
            gpre[r][half] = 8 * t4 < DH ? __ldg(reinterpret_cast<const uint4*>(GATE + ((size_t)row0(b, n) + qt * BM + qrow + g8 + 8 * half) * ldg + h * DH + 8 * t4)) : make_uint4(0, 0, 0, 0);
        }
      }
    };
    if (TA_RC) {
      // TA_RC: the last key tile peeled.  The loop below has no path from the end of one row's sub-step to a read of the other row's
      // probabilities other than through their redefinition (the j + 1 < nk test inside the loop gave ptxas such a path: the row whose PV
      // had retired stayed live through the exponentials of the other, 32 registers), and the gate prefetch is defined after the loop
      // (it was loop-carried: 16 registers live through every sub-step).  Same operations in the same order (bit-identical).  nk >= 1.
      for (int j = 0; j + 1 < nk; ++j) {
        if (SWM) { step_sw(std::false_type{}, std::false_type{}, 1); step_sw(std::false_type{}, std::false_type{}, 0); }
        else { step(1); step(0); }
      }
      gate_pre();
      if (SWM) step_sw(std::false_type{}, std::true_type{}, 1);
      else step(1);
    } else
    for (int j = 0; j < nk; ++j) {
      if (j == nk - 1) gate_pre();
      if (SWM) {
        if (j + 1 < nk) { step_sw(std::false_type{}, std::false_type{}, 1); step_sw(std::false_type{}, std::false_type{}, 0); }
        else step_sw(std::false_type{}, std::true_type{}, 1);
      } else {
        step(1);
        if (j + 1 < nk) step(0);
      }
    }
    if (TA_PP) named_sync(2 + w, 256);
    PV(1);                                           // the last sub-step's PV
    if (TA_PP) named_arrive(2 + (w ^ 1), 256);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    if (LIST) check(1);
    release();
    // ---- epilogue: O_r / l_r -> bf16 -> out (a quad transposes its 4 x 4 words so each lane stores 16 contiguous bytes); LSE
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = g * R + r;
      if (n >= N) continue;                          // a padding row: computed on row N - 1, not stored
      const float l0 = o[r][LS], l1 = o[r][LS + 2];
      bad |= !(l0 > 0.f) || !(l0 < 3.0e38f) || !(l1 > 0.f) || !(l1 < 3.0e38f);
      ovf |= !(l0 < 3.0e38f) || !(l1 < 3.0e38f);
      #if TA_F8PV
      const float vsc = VS[b * H + h], inv0 = vsc / l0, inv1 = vsc / l1;   // the row sum is of the e4m3 probabilities (ones rows of the PV): P V8 / l is normalised, times the V scale
#else
      const float inv0 = 1.f / l0, inv1 = 1.f / l1;
#endif
      const int q0 = qt * BM + qrow + g8;
      if (LSE != nullptr && t4 == 0) {
        float* lp = LSE + ((size_t)(b * N + n) * H + h) * S + q0;
        lp[0] = (m[r][0] > -INFINITY ? m[r][0] : 0.f) + __log2f(l0);
        lp[8] = (m[r][1] > -INFINITY ? m[r][1] : 0.f) + __log2f(l1);
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float inv = half ? inv1 : inv0;
        uint32_t in[4], outw[4];
#pragma unroll
#if TA_F8HL
        for (int i = 0; i < 4; ++i) in[i] = pack2((o[r][4 * i + 2 * half] + o[r][16 + 4 * i + 2 * half] * (1.f / 16.f)) * inv, (o[r][4 * i + 2 * half + 1] + o[r][16 + 4 * i + 2 * half + 1] * (1.f / 16.f)) * inv);
#else
        for (int i = 0; i < 4; ++i) in[i] = 4 * i < LS ? pack2(o[r][4 * i + 2 * half] * inv, o[r][4 * i + 2 * half + 1] * inv) : 0u;   // (channels past the head are zero: pack2(+0, +0))
#endif
        // lane t4 holds channels 8i + 2 t4 (i = 0..3); after the transpose it holds channels 8 t4 .. 8 t4 + 7
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int sidx = (t4 - k) & 3, didx = (t4 + k) & 3;
          uint32_t send = in[0];
#pragma unroll
          for (int i = 1; i < 4; ++i) send = sidx == i ? in[i] : send;
          const uint32_t recv = __shfl_sync(0xffffffffu, send, (lane & ~3) | didx);
#pragma unroll
          for (int i = 0; i < 4; ++i) if (didx == i) outw[i] = recv;
        }
        const size_t orow = (size_t)row0(b, n) + q0 + 8 * half;
        if (OUT != nullptr && 8 * t4 < DH && !((TA_ABL & 16384) && S > 0)) *reinterpret_cast<uint4*>(OUT + orow * (H * DH) + h * DH + 8 * t4) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
        if (GATE != nullptr && 8 * t4 < DH) {
          const uint4 gv = gpre[r][half];
          const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
          uint32_t uw[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 gg = gate_h16 ? make_float2(hlo(gw[i]), hhi(gw[i])) : bf2f(gw[i]), oo = bf2f(outw[i]);
            uw[i] = pack2(oo.x / (1.f + __expf(-gg.x)), oo.y / (1.f + __expf(-gg.y)));
          }
          *reinterpret_cast<uint4*>(U + orow * (H * DH) + h * DH + 8 * t4) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
        }
      }
    }
    if (LIST) {                                      // a tile the speculative offset could not absorb: the task goes on the fix list
      // every consumer warp adds 1 (+256 when one of its lanes overflowed) to a shared-memory counter per task parity; the warp that brings it
      // to 8 pushes the task once and resets the counter (no warp waits: the others are at most a tile behind)
      volatile int* cnt = reinterpret_cast<volatile int*>(smb + (bar0 - sB) + 244 + 4 * (lt & 1));
      const int mine = __any_sync(0xffffffffu, ovf) ? 256 : 0;
      if (lane == 0) {
        const int old = atomicAdd(const_cast<int*>(cnt), 1 + mine);
        if ((old & 255) == 7) {
          if (((old + 1 + mine) >> 8) != 0) { const int i = atomicAdd(FIX, 1); FIX[1 + i] = t; }
          *cnt = 0;
        }
      }
    }
  }
  if (bad) atomicAdd(FLAGS, 1);
  if (CL) cluster_sync_all();
}

namespace {
torch::Tensor& flags_for(const torch::Device& dev) {
  static torch::Tensor flags[16];
  auto& f = flags[dev.index()];
  if (!f.defined()) f = torch::zeros({2}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  return f;
}
}  // namespace

// q, k, v [B, N, S, H, 32] fp16 (the projection layout; rows may sit at a common wider stride); sbias fp16 [B, H, S, S] contiguous = bias / scale
// (masked keys -inf); S % 128 == 0.  -> (out [B, N, S, H, 32] bf16, lse fp32 [B, N, H, S] base 2 (empty unless want_lse), flags int32 [2], per-device
// counters accumulated over every call: [0] rows whose sum left (0, 3e38) -- a row with every key masked --, [1] tasks the speculative offset
// could not absorb and the exact launch recomputed; u = sigmoid(gate) o out when `gate` is given).  `key_tiles`: int32 [B], the 128-key tiles of
// each batch element up to its last attended key (`key_tiles` in h100_sm90.py); the tiles past them, whose keys are masked (-inf) in every row,
// are neither loaded nor computed: their exponentials are exact zeros, so the result is the same bits
// (`key_mask`, bool [B, S]: the same tiles counted inside the kernel from the mask, B <= KM_MAXB)
// `bias_transposed`: sbias holds bias[b, h, key, query] (the transpose of the layout above), read through ldmatrix.trans
std::vector<torch::Tensor> triattn_fwd_h16(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sbias, double scale, bool want_lse,
                                           c10::optional<torch::Tensor> gate, bool want_out,
                                           c10::optional<torch::Tensor> key_tiles, c10::optional<torch::Tensor> key_mask,
                                           bool bias_transposed
#if TA_F8PV
                                           , torch::Tensor vt, torch::Tensor vscale
#endif
                                           ) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 5 && (q.size(4) == D || q.size(4) == 16), "q [B, N, S, H, 32 | 16]");
  const long dh = q.size(4);   // 16: heads of 16 channels stored compact (zero-padded to 32 by the loads)
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % BM == 0 && S > 0, "S % 128 == 0");
  const long ld = q.stride(2);
  TORCH_CHECK(!TA_F8QK || (dh == D && q.scalar_type() == torch::kFloat8_e4m3fn && k.scalar_type() == torch::kFloat8_e4m3fn), "F8QK: q / k float8_e4m3fn, 32-channel heads");
  for (const auto* t : {&q, &k, &v}) {
    TORCH_CHECK(t->scalar_type() == (TA_F8QK ? torch::kFloat8_e4m3fn : torch::kHalf) && t->sizes() == q.sizes() && t->stride(4) == 1 && t->stride(3) == dh &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld && ld % 8 == 0 && ld >= H * dh,
                "q / k / v: [B, N, S, H, 32 | 16] fp16 rows at one common stride");
  }
  TORCH_CHECK(sbias.is_cuda() && sbias.scalar_type() == torch::kHalf && sbias.is_contiguous() && sbias.numel() == B * H * S * S,
              "sbias: fp16 [B, H, S, S] contiguous (bias / scale)");
  const uint64_t rows = (uint64_t)(B * N * S);
  auto m2 = [&](const torch::Tensor& t, const char* what) {
    if (dh != D)   // [rows][H][16], box (32, 1, 128): the 16 channels past the dimension read as zeros
      return make_map<3>(t.data_ptr(), {(uint64_t)dh, (uint64_t)H, rows}, {(uint64_t)dh, (uint64_t)ld}, {32, 1, 128}, CU_TENSOR_MAP_SWIZZLE_64B,
                         what, CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
    return make_map<2>(t.data_ptr(), {(uint64_t)(H * D), rows}, {(uint64_t)ld}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_64B, what,
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT16); };
#if TA_F8PV
  TORCH_CHECK(vt.is_cuda() && vt.scalar_type() == torch::kFloat8_e4m3fn && vt.is_contiguous() && vt.numel() == B * N * H * VROWS / 32 * dh * S,
              "vt: float8_e4m3fn [B, N, H, dh, S] contiguous (keys permuted inside each group of 32)");
  TORCH_CHECK(vscale.is_cuda() && vscale.scalar_type() == torch::kFloat32 && vscale.is_contiguous() && vscale.numel() == B * H, "vscale: fp32 [B, H]");
  CUtensorMap qm = m2(q, "q"), km = m2(k, "k"), vm = km;   // (v is not read: the fp16 operand of the shape contract)
  if (TA_F8QK) {   // e4m3 q / k: [rows][H * 32 bytes], box (32 bytes, 128 rows), 32-byte swizzle
    qm = make_map<2>(q.data_ptr(), {(uint64_t)(H * D), rows}, {(uint64_t)ld}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_32B, "q8", CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
    km = make_map<2>(k.data_ptr(), {(uint64_t)(H * D), rows}, {(uint64_t)ld}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_32B, "k8", CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
    vm = km;
  }
  CUtensorMap vtm = dh == D ? make_map<2>(vt.data_ptr(), {(uint64_t)S, (uint64_t)(B * N * H * VROWS)}, {(uint64_t)S}, {128, 32}, CU_TENSOR_MAP_SWIZZLE_128B, "vt",
                                          CU_TENSOR_MAP_DATA_TYPE_UINT8, 1)
                           : make_map<3>(vt.data_ptr(), {(uint64_t)S, 16, (uint64_t)(B * N * H * (VROWS / 32))}, {(uint64_t)S, 16 * (uint64_t)S}, {128, (TA_F8N24 && !TA_F8HL) ? 16u : 32u, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "vt",
                                         CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
#else
  CUtensorMap qm = m2(q, "q"), km = m2(k, "k"), vm = m2(v, "v");
  if (dh != D && TA_N24 && TA_K16)   // K likewise: the 16 channels, 32-byte swizzle (K-major atoms of 8 keys x 32 bytes)
    km = make_map<3>(k.data_ptr(), {(uint64_t)dh, (uint64_t)H, rows}, {(uint64_t)dh, (uint64_t)ld}, {16, 1, 128}, CU_TENSOR_MAP_SWIZZLE_32B, "k",
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  if (dh != D && TA_N24)   // PV as n24: V read as the 16 channels only, in the 32-byte swizzle (MN-major atoms of 8 keys x 16 channels)
    vm = make_map<3>(v.data_ptr(), {(uint64_t)dh, (uint64_t)H, rows}, {(uint64_t)dh, (uint64_t)ld}, {16, 1, 128}, CU_TENSOR_MAP_SWIZZLE_32B, "v",
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
#endif
  CUtensorMap bm = make_map<2>(sbias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B,
                               "sbias", CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  TORCH_CHECK(!key_tiles.has_value() || (key_tiles->is_cuda() && key_tiles->scalar_type() == torch::kInt32 && key_tiles->is_contiguous() &&
              key_tiles->numel() == B), "key_tiles: int32 [B] contiguous");
  const int* kt = key_tiles.has_value() ? key_tiles->data_ptr<int>() : nullptr;
  // key_mask (instead of key_tiles): bool [B, S], the kernel counts the live key tiles itself
  TORCH_CHECK(!key_mask.has_value() || (!key_tiles.has_value() && key_mask->is_cuda() && key_mask->scalar_type() == torch::kBool &&
              key_mask->is_contiguous() && key_mask->numel() == B * S && B <= KM_MAXB && reinterpret_cast<uintptr_t>(key_mask->data_ptr()) % 16 == 0),
              "key_mask: bool [B, S] contiguous, 16-byte aligned, B <= KM_MAXB (else key_tiles)");
  const unsigned char* km_ptr = key_mask.has_value() ? reinterpret_cast<const unsigned char*>(key_mask->data_ptr()) : nullptr;
  const bool has_gate = gate.has_value();
  const bool store_out = want_out || !has_gate;
  TORCH_CHECK(!has_gate || ((gate->scalar_type() == torch::kBFloat16 || gate->scalar_type() == torch::kHalf) && gate->is_cuda() && gate->sizes() == q.sizes() && gate->stride(4) == 1 &&
              gate->stride(3) == dh && gate->stride(2) % 8 == 0 && gate->stride(1) == S * gate->stride(2) && gate->stride(0) == N * S * gate->stride(2)),
              "gate: [B, N, S, H, 32 | 16] bf16 or fp16, rows at one stride");
  const auto bf = q.options().dtype(torch::kBFloat16);
  auto out = torch::empty({store_out ? B : 0, N, S, H, dh}, bf);
  auto u = torch::empty({has_gate ? B : 0, N, S, H, dh}, bf);
  auto lse = torch::empty({want_lse ? B : 0, N, H, S}, q.options().dtype(torch::kFloat32));
  auto& flags = flags_for(q.device());
  const long ntasks = B * H * ((N + R - 1) / R) * (S / BM);
  const int grid = (int)std::min<long>(ntasks, num_sms(q.device().index()));
  const bool dyn = TA_DYN && ntasks >= 16L * grid;
  auto sched = torch::zeros({dyn ? 1 : 0}, q.options().dtype(torch::kInt32));
  const bool spec = TA_SPEC >= 1 && S >= TA_SPEC_MINS, list = spec && TA_SPEC == 1;
  auto fix = torch::empty({list ? 1 + ntasks : 0}, q.options().dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (list) C10_CUDA_CHECK(cudaMemsetAsync(fix.data_ptr<int>(), 0, sizeof(int), stream));
  // head groups: one head from TA_ORDER1_MINS on, else the most heads whose bias fits TA_BIAS_L2 (all of them: ORD0)
  int hg = (int)H;
  if (TA_HG > 0 && H % TA_HG == 0) hg = TA_HG;
  else if (S >= TA_ORDER1_MINS) hg = 1;
  else while (hg > 1 && (H % hg != 0 || (double)hg * S * S * 2 > (double)(TA_BIAS_L2))) --hg;
  const bool ord1 = hg < H;
  auto launch = [&](auto kernel, int* sch, int* fx) {
    kernel<<<grid, THREADS, SMEM, stream>>>(
      (int)N, (int)H, (int)S, (int)ntasks, (float)(scale * 1.4426950408889634), hg, qm, km, vm, bm,
      store_out ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()) : nullptr, want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>(),
      has_gate ? reinterpret_cast<const __nv_bfloat16*>(gate->data_ptr()) : nullptr, (int)(has_gate && gate->scalar_type() == torch::kHalf),
      has_gate ? reinterpret_cast<__nv_bfloat16*>(u.data_ptr<at::BFloat16>()) : nullptr, has_gate ? (long)gate->stride(2) : 0L, sch, fx, kt, km_ptr
#if TA_F8PV
      , vtm, vscale.data_ptr<float>()
#endif
    );
  };
  // CL: the same launch as a cluster of two CTAs (adjacent query tiles share the K / V loads)
  auto launch_cl = [&](auto kernel, int* sch) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(grid & ~1)); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = SMEM; cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    if (getenv("TA_CL_DEBUG")) { int nc = -1; cudaError_t e = cudaOccupancyMaxActiveClusters(&nc, kernel, &cfg); fprintf(stderr, "TA_CL grid %d max active clusters %d (%s)\n", (int)cfg.gridDim.x, nc, cudaGetErrorString(e)); }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
      (int)N, (int)H, (int)S, (int)ntasks, (float)(scale * 1.4426950408889634), hg, qm, km, vm, bm,
      store_out ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()) : nullptr, want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>(),
      has_gate ? reinterpret_cast<const __nv_bfloat16*>(gate->data_ptr()) : nullptr, (int)(has_gate && gate->scalar_type() == torch::kHalf),
      has_gate ? reinterpret_cast<__nv_bfloat16*>(u.data_ptr<at::BFloat16>()) : nullptr, has_gate ? (long)gate->stride(2) : 0L, sch, (int*)nullptr, kt, km_ptr));
  };
  // the head width is a template argument: the 32-channel kernels are the ones without the compact form, instruction for instruction
  auto run = [&](auto dhc, auto btc) {
    constexpr int DH = decltype(dhc)::value;
    constexpr bool BT = decltype(btc)::value;
    // the polynomial exponentials: TA_POLY of every 16 pairs on the 32-channel heads, TA_POLY16 on the heads of 16 channels
    constexpr bool POLY = true;
    if constexpr (TA_ONE) {   // compile experiments only: one kernel (VOTE, ORD1, 32-channel heads, bias as stored)
      if constexpr (DH == D && !BT) {
        cudaFuncSetAttribute(triattn_h16_sm90<true, true, DH, POLY, BT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(triattn_h16_sm90<true, false, DH, POLY, BT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        launch(ord1 ? triattn_h16_sm90<true, true, DH, POLY, BT> : triattn_h16_sm90<true, false, DH, POLY, BT>, dyn ? sched.data_ptr<int>() : nullptr, nullptr);
      }
      return;
    }
    static bool attr = false;
    if (!attr) {
      for (auto kk : {triattn_h16_sm90<false, false, DH, POLY, BT>, triattn_h16_sm90<false, true, DH, POLY, BT>, triattn_h16_sm90<true, false, DH, POLY, BT>,
                      triattn_h16_sm90<true, true, DH, POLY, BT>})
        C10_CUDA_CHECK(cudaFuncSetAttribute(kk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      attr = true;
    }
    if constexpr (TA_CL && DH == D) {
      if (spec && !list && (S / BM) % 2 == 0 && grid >= 2) {
        static bool attr_cl = false;
        if (!attr_cl) {
          for (auto kk : {triattn_h16_sm90<true, false, DH, POLY, BT, true>, triattn_h16_sm90<true, true, DH, POLY, BT, true>})
            C10_CUDA_CHECK(cudaFuncSetAttribute(kk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
          attr_cl = true;
        }
        launch_cl(ord1 ? triattn_h16_sm90<true, true, DH, POLY, BT, true> : triattn_h16_sm90<true, false, DH, POLY, BT, true>, dyn ? sched.data_ptr<int>() : nullptr);
        return;
      }
    }
    if (spec) {
      launch(ord1 ? triattn_h16_sm90<true, true, DH, POLY, BT> : triattn_h16_sm90<true, false, DH, POLY, BT>, dyn ? sched.data_ptr<int>() : nullptr,
             list ? fix.data_ptr<int>() : nullptr);
      if (list) {
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        launch(ord1 ? triattn_h16_sm90<false, true, DH, POLY, BT> : triattn_h16_sm90<false, false, DH, POLY, BT>, nullptr, fix.data_ptr<int>());
      }
    } else {
      launch(ord1 ? triattn_h16_sm90<false, true, DH, POLY, BT> : triattn_h16_sm90<false, false, DH, POLY, BT>, dyn ? sched.data_ptr<int>() : nullptr, nullptr);
    }
  };
  using T_ = std::true_type;
  using F_ = std::false_type;
  if (dh == D) bias_transposed ? run(std::integral_constant<int, D>{}, T_{}) : run(std::integral_constant<int, D>{}, F_{});
  else bias_transposed ? run(std::integral_constant<int, 16>{}, T_{}) : run(std::integral_constant<int, 16>{}, F_{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags, u};
}


#if TA_F8PV
// ---- V^T e4m3 for the F8PV core (a standalone pass until the fronts write it directly) ----
// v [B, N, S, H, 32] fp16 (rows at stride ld) -> vt [B, N, H, 32, S] e4m3 (v / vscale), the keys of each group of 32 permuted so a lane's accumulator columns
// 2 t4 + {0, 1} of the 8-column blocks 4 s .. 4 s + 3 land on the k32 A fragment's positions 4 t4 + {0 .. 3} / 16 + 4 t4 + {0 .. 3}; vscale [B, H] = amax / 448
// (amax over N, S and the head's 32 channels: vt_amax; or `fixed` > 0: that amax for every head, no first pass).
#include <cuda_fp8.h>
namespace {
__device__ __forceinline__ int vt_pos(int k) {   // position (inside a group of 32) of the key k
  const int blk = k >> 3, rem = k & 7;
  return 16 * (blk >> 1) + 4 * (rem >> 1) + 2 * (blk & 1) + (rem & 1);
}
constexpr int VQ_TILE = 128;
// grid (S / 128, N, B), blockDim = a multiple of C / 8 (<= 256: a thread's c8, hence its head, is fixed across idx); amax: fp32 [B * H] bit patterns (non-negative floats order as integers), zeroed
__global__ void __launch_bounds__(256) vt_amax_k(const __half* __restrict__ v, long ld, int N, int S, int H, int dh, unsigned* __restrict__ amax) {
  const int C = H * dh, jt = blockIdx.x, n = blockIdx.y, b = blockIdx.z;
  const __half* base = v + ((size_t)(b * N + n) * S + jt * VQ_TILE) * ld;
  float mx = 0.f;
  const int nth = blockDim.x;
  for (int idx = threadIdx.x; idx < VQ_TILE * C / 8; idx += nth) {
    const int key = idx / (C / 8), c8 = idx % (C / 8);
    const uint4 x = *reinterpret_cast<const uint4*>(base + (size_t)key * ld + c8 * 8);
    const __half2* h2 = reinterpret_cast<const __half2*>(&x);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(h2[i]); a = fmaxf(a, fmaxf(fabsf(f.x), fabsf(f.y))); }
    mx = fmaxf(mx, a);
  }
  __shared__ unsigned sm[16];
  if (threadIdx.x < 16) sm[threadIdx.x] = 0u;
  __syncthreads();
  atomicMax(&sm[((threadIdx.x % (C / 8)) * 8) / dh], __float_as_uint(mx));
  __syncthreads();
  if (threadIdx.x < H) atomicMax(amax + b * H + threadIdx.x, sm[threadIdx.x]);
}
__global__ void __launch_bounds__(256) vt_quant_k(const __half* __restrict__ v, long ld, int N, int S, int H, int dh, const unsigned* __restrict__ amax, float fixed, const float* __restrict__ vinv, int* __restrict__ ovf,
                                                  __nv_fp8_storage_t* __restrict__ vt, float* __restrict__ vscale) {
  extern __shared__ __align__(16) unsigned char sh[];   // [PLANES][C][VQ_TILE + 4] e4m3 (PLANES 2: hi, lo)
  constexpr int PLANES = TA_F8HL ? 2 : 1;
  const int C = H * dh, LDS = VQ_TILE + 4, jt = blockIdx.x, n = blockIdx.y, b = blockIdx.z;
  const __half* base = v + ((size_t)(b * N + n) * S + jt * VQ_TILE) * ld;
  bool bad = false;
  for (int idx = threadIdx.x; idx < VQ_TILE * C / 8; idx += 256) {
    const int key = idx / (C / 8), c8 = idx % (C / 8), h = (c8 * 8) / dh;
    const float inv = vinv != nullptr ? vinv[h] : 448.f / fmaxf(fixed > 0.f ? fixed : __uint_as_float(amax[b * H + h]), 1e-30f);
    const uint4 x = *reinterpret_cast<const uint4*>(base + (size_t)key * ld + c8 * 8);
    const __half2* h2 = reinterpret_cast<const __half2*>(&x);
    const int p = (key & ~31) + vt_pos(key & 31);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h2[i]);
      bad |= (fabsf(f.x * inv) > 448.f) | (fabsf(f.y * inv) > 448.f);
      const __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(make_float2(f.x * inv, f.y * inv), __NV_SATFINITE, __NV_E4M3);
      sh[(c8 * 8 + 2 * i) * LDS + p] = (unsigned char)(q & 0xFF);
      sh[(c8 * 8 + 2 * i + 1) * LDS + p] = (unsigned char)(q >> 8);
#if TA_F8HL   // the residual of the first rounding (x16: |residual| <= 16 at the largest values), rounded again to e4m3
      {
        const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(q, __NV_E4M3);
        const uint32_t hu = (uint32_t)hr.x | ((uint32_t)hr.y << 16);
        const float2 hq = make_float2(hlo(hu), hhi(hu));
        const __nv_fp8x2_storage_t r2 = __nv_cvt_float2_to_fp8x2(make_float2((f.x * inv - hq.x) * 16.f, (f.y * inv - hq.y) * 16.f), __NV_SATFINITE, __NV_E4M3);
        sh[C * LDS + (c8 * 8 + 2 * i) * LDS + p] = (unsigned char)(r2 & 0xFF);
        sh[C * LDS + (c8 * 8 + 2 * i + 1) * LDS + p] = (unsigned char)(r2 >> 8);
      }
#endif
    }
  }
  if (ovf != nullptr && __any_sync(0xffffffffu, bad) && (threadIdx.x & 31) == 0) atomicOr(ovf, 1);
  __syncthreads();
  for (int idx = threadIdx.x; idx < PLANES * C * (VQ_TILE / 16); idx += 256) {
    const int pl = idx / (C * (VQ_TILE / 16)), idx0 = idx % (C * (VQ_TILE / 16));
    const int ch = idx0 / (VQ_TILE / 16), c16 = idx0 % (VQ_TILE / 16);
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned char* s = sh + (pl * C + ch) * LDS + c16 * 16 + 4 * i;
      w[i] = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
    }
    const int h = ch / dh, d = ch % dh;
    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(vt) + ((size_t)(((b * N + n) * H + h) * PLANES + pl) * dh + d) * S + jt * VQ_TILE + c16 * 16) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  if (jt == 0 && n == 0 && threadIdx.x < H) vscale[b * H + threadIdx.x] = vinv != nullptr ? 1.f / vinv[threadIdx.x] : (fixed > 0.f ? fixed : __uint_as_float(amax[b * H + threadIdx.x])) / 448.f;
}
}  // namespace

// v: fp16 [B, N, S, H, 32] (rows at one stride), S % 128 == 0, H <= 16; fixed_amax > 0: no amax pass (every head's amax taken as that value)
std::vector<torch::Tensor> vt_quant(torch::Tensor v, double fixed_amax, c10::optional<torch::Tensor> vinv, c10::optional<torch::Tensor> ovf) {
  TORCH_CHECK(v.is_cuda() && v.dim() == 5 && (v.size(4) == 32 || v.size(4) == 16) && v.scalar_type() == torch::kHalf && v.stride(4) == 1 && v.stride(3) == v.size(4) && v.size(3) <= 16 &&
              v.size(2) % VQ_TILE == 0 && v.stride(2) % 8 == 0 && v.stride(1) == v.size(2) * v.stride(2) && v.stride(0) == v.size(1) * v.stride(1), "v: [B, N, S, H<=16, 32 | 16] fp16");
  const int B = v.size(0), N = v.size(1), S = v.size(2), H = v.size(3), dh = v.size(4);
#if TA_F8HL
  auto vt = torch::empty({B, N, H, 2, dh, S}, v.options().dtype(torch::kFloat8_e4m3fn));
#else
  auto vt = torch::empty({B, N, H, dh, S}, v.options().dtype(torch::kFloat8_e4m3fn));
#endif
  auto vscale = torch::empty({B, H}, v.options().dtype(torch::kFloat32));
  // vinv: fp32 [H] = 1 / scale per head (static scale: no amax pass; values past 448 saturate and set the int32 flag `ovf`)
  TORCH_CHECK(!vinv.has_value() || (vinv->is_cuda() && vinv->scalar_type() == torch::kFloat32 && vinv->is_contiguous() && vinv->numel() == H), "vinv: fp32 [H]");
  TORCH_CHECK(!ovf.has_value() || (ovf->is_cuda() && ovf->scalar_type() == torch::kInt32 && ovf->numel() >= 1), "ovf: int32 flag");
  const bool stat = vinv.has_value();
  auto amax = torch::zeros({stat ? 0 : B * H}, v.options().dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(S / VQ_TILE, N, B);
  if (fixed_amax <= 0 && !stat) vt_amax_k<<<grid, (256 / (H * dh / 8)) * (H * dh / 8), 0, stream>>>(reinterpret_cast<const __half*>(v.data_ptr()), (long)v.stride(2), N, S, H, dh, reinterpret_cast<unsigned*>(amax.data_ptr<int>()));
  const int sm = (TA_F8HL ? 2 : 1) * H * dh * (VQ_TILE + 4);
  if (sm > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(vt_quant_k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
  vt_quant_k<<<grid, 256, sm, stream>>>(reinterpret_cast<const __half*>(v.data_ptr()), (long)v.stride(2), N, S, H, dh, reinterpret_cast<const unsigned*>(amax.data_ptr<int>()),
                                       (float)fixed_amax, stat ? vinv->data_ptr<float>() : nullptr, stat && ovf.has_value() ? ovf->data_ptr<int>() : nullptr,
                                       reinterpret_cast<__nv_fp8_storage_t*>(vt.data_ptr()), vscale.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {vt, vscale};
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd_h16", &triattn_fwd_h16, "sm90 triangle-attention forward, inference, fp16 operands (D = 32): q/k/v [B, N, S, H, 32] fp16, "
        "sbias = bias / scale fp16 [B, H, S, S] -> out bf16, optional base-2 LSE; with `gate` also u = sigmoid(gate) o out",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sbias"), py::arg("scale"), py::arg("want_lse"), py::arg("gate") = py::none(),
        py::arg("want_out") = true, py::arg("key_tiles") = py::none(), py::arg("key_mask") = py::none(),
        py::arg("bias_transposed") = false
#if TA_F8PV
        , py::arg("vt"), py::arg("vscale")
#endif
        );
#if TA_F8PV
  m.def("vt_quant", &vt_quant, "V^T e4m3 (keys permuted inside each group of 32) and the per-(batch, head) scale of v [B, N, S, H, 32] fp16", py::arg("v"), py::arg("fixed_amax") = 0.0, py::arg("vinv") = py::none(), py::arg("ovf") = py::none());
#endif
}
