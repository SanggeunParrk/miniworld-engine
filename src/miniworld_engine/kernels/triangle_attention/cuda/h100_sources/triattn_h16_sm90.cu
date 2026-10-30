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
namespace {
constexpr int D = 32, BM = 128, BN = 128, R = 2, NKV = TA_NKV, NB = TA_NB;
constexpr int QT = BM * D;                                 // Q tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][32], 64B swizzle (8 KiB)
constexpr int STG = 3 * KVT;                               // a K | V stage: K, V, ones (24 KiB)
constexpr int BT = BM * BN;                                // bias tile: fp16 [2 key halves][128 q][64], 128B swizzle (32 KiB)
constexpr int THREADS = 384;
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
// [O | l] += P [V | 1]: m64n40k16, fp16 x fp16 -> fp32, A = P from registers, B = [V | ones] from shared (MN-major).
__device__ __forceinline__ void mma_h16_n40(float (&d)[20], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n40k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
               "%15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 1;\n}\n"
               : SM90_D16(0), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
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
#ifndef TA_POLY
#define TA_POLY 8         // pairs i (of 16 per row) whose exponentials run as hex2_poly: TA_POLY of every 16 (1, 2, 4, 8 or 16), the rest on the
                          // MUFU.  8 (every other pair; sustained, power-capped, c256 / 8 heads): L512 level, 768 / 1024 -0.5 %, 1536 -2.7 %,
                          // 2048 -3.3 %, accuracy level; 2 / 4: level or +1.5 %; 16 (all): +12 %
#endif
__device__ __forceinline__ uint32_t hexp(uint32_t a, int i) {
  if (TA_POLY > 0 && i % (16 / (TA_POLY > 0 ? TA_POLY : 1)) == (16 / (TA_POLY > 0 ? TA_POLY : 1)) - 1) return hex2_poly(a);
  return hex2(a);
}
__device__ __forceinline__ float hlo(uint32_t a) { return __half2float(__ushort_as_half((unsigned short)(a & 0xffffu))); }
__device__ __forceinline__ float hhi(uint32_t a) { return __half2float(__ushort_as_half((unsigned short)(a >> 16))); }
__device__ __forceinline__ uint32_t hdup(float x) { const unsigned short h = __half_as_ushort(__float2half_rn(x)); return (uint32_t)h | ((uint32_t)h << 16); }
__device__ __forceinline__ float hround(float x) { return __half2float(__float2half_rn(x)); }
}  // namespace

// SPEC: the speculative offset (fix list written to FIX); else the exact form, which with FIX set runs the tasks of that list (static schedule).
// ORD1: task order query tile, head in its group of hg, row group, head group (the CTAs running together share the bias of hg heads; hg = 1:
// one head), else query tile, head, row group.
template <bool SPEC, bool ORD1>
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
    int* __restrict__ FIX) {                             // fix list: FIX[0] = count (zeroed), FIX[1..] = task ids (written by SPEC, read by the exact form)
  constexpr bool VOTE = SPEC && TA_SPEC == 2;    // TA_SPEC 2: no fix list (see X)
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
  for (int i = tid; i < NKV * KVT / 2; i += THREADS) {     // the ones blocks: fp16 pairs (1, 1)
    const int s = i / (KVT / 2), e = i % (KVT / 2);
    reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (s * STG + 2 * KVT) * 2)[e] = 0x3C003C00u;
  }
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sB));
    bar_init(bars, 1); bar_init(bars + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(bars + 2 + k, 1); bar_init(bars + 2 + NKV + k, 8); }
    for (int k = 0; k < NB; ++k) { bar_init(bars + 2 + 2 * NKV + k, 1); bar_init(bars + 2 + 2 * NKV + NB + k, 8); }
    bar_init_fence();
    reinterpret_cast<int*>(smb + (bar0 - sB) + 244)[0] = 0; reinterpret_cast<int*>(smb + (bar0 - sB) + 244)[1] = 0;     // the fix-list counters
  }
  fence_proxy_async();
  __syncthreads();
  auto decode = [&](int t, int& b, int& h, int& qt, int& g) {
    qt = t % QTILES; int r = t / QTILES;
    if (ORD1) { const int hl = r % hg; r /= hg; g = r % NG; r /= NG; h = (r % (H / hg)) * hg + hl; b = r / (H / hg); }
    else        { h = r % H; r /= H; g = r % NG; b = r / NG; }
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row; a padding row reads row N - 1
  // dynamic schedule: the producer takes each next task from the global counter and hands it to the consumers in this slot (written before
  // the Q tiles' arrive, read after its completion; -1 ends the CTA)
  volatile int* slot = reinterpret_cast<volatile int*>(smb + (bar0 - sB) + 240);

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&qmap); prefetch_map(&kmap); prefetch_map(&vmap); prefetch_map(&bmap);
      int lt = 0, st = 0, kph = 0, bs = 0, bph = 0;
      for (int t = blockIdx.x; t < nt; t = SCHED ? (int)gridDim.x + atomicAdd(SCHED, 1) : t + (int)gridDim.x, ++lt) {
        int b, h, qt, g; decode(LISTM ? FIX[1 + t] : t, b, h, qt, g);
        wait(qe, (lt & 1) ^ 1);
        if (SCHED) *slot = t;
        expect_tx(qf, R * QT * 2);
        for (int r = 0; r < R; ++r) load_2d(&qmap, sQ + r * QT * 2, qf, h * D, row0(b, g * R + r) + qt * BM);
        for (int j = 0; j < NK; ++j) {
          wait(bem(bs), bph ^ 1);
          expect_tx(bfu(bs), BT * 2);
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sB + (bs * BT + hh * BM * 64) * 2, bfu(bs), j * BN + hh * 64, (b * H + h) * S + qt * BM);
          if (++bs == NB) { bs = 0; bph ^= 1; }
          for (int r = 0; r < R; ++r) {
            wait(kve(st), kph ^ 1);
            expect_tx(kvf(st), 2 * KVT * 2);
            const int rw = row0(b, g * R + r) + j * BN;
            load_2d(&kmap, sKV + st * STG * 2, kvf(st), h * D, rw);
            load_2d(&vmap, sKV + (st * STG + KVT) * 2, kvf(st), h * D, rw);
            if (++st == NKV) { st = 0; kph ^= 1; }
          }
        }
      }
      if (SCHED) {                                   // the end: once the consumers took the last task's Q, a task-free phase of qf
        wait(qe, (lt & 1) ^ 1);
        *slot = -1;
        arrive(qf);
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int qrow = 64 * w + 16 * wi;               // this warp's first query row in the tile
  // ldmatrix lane geometry: matrices (rows +0 / +8) x (columns +0 / +8) of a 16 x 16 block
  const int lrow = qrow + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4, x7 = lane & 7;
  const uint32_t qlane = sQ + lrow * D * 2, blane = sB + lrow * 128;
  const uint64_t kd0 = desc_k64(sKV), vd0 = desc_mn64(sKV + KVT * 2, KVT * 2);   // built once: a wgmma operand costs one 32-bit add
  const uint32_t scl2 = hdup(scl);
  // cursors: the K | V stage and bias stage the next G phase reads; the stage whose PV was issued last (released once it retired)
  int gst = 0, gph = 0, bs = 0, bph = 0, xst = 0, bad = 0, ovf = 0;
  uint32_t qa[R][2][4];
  float o[R][20], m[R][2];                          // [O | l] per row (columns 32..39 = the row sums: o[16] row g, o[18] row g + 8); the offsets
  float lprev[R][2];                                // the row sums before the tile in flight (the increment test of the speculative offset)
  uint32_t sc[R][32];                               // per row: the score accumulator (fp16 pairs), then its probabilities = the A fragments of its PV
  // G phase of (tile j, row r): bias fragment -> sc[r] (the fp16 pairs ldmatrix returns are the accumulator's), S += Q_r K^T issued
  auto G = [&](int r) {
    if (r == 0) wait(bfu(bs), bph);
    const uint32_t btile = blane + bs * BT * 2;
#pragma unroll
    for (int i2 = 0; i2 < 8; ++i2) {
      uint32_t r4[4];
      ldsm_x4(btile + (i2 >> 2) * BM * 128 + ((((2 * i2 + lcol8) & 7) ^ x7) << 4), r4);
      // r4 = (row g, block 2 i2), (row g + 8, block 2 i2), (row g, block 2 i2 + 1), (row g + 8, block 2 i2 + 1)
#pragma unroll
      for (int e = 0; e < 4; ++e) sc[r][4 * i2 + e] = r4[e];
    }
    if (r == R - 1) {
      fence_regs(sc[r]);                            // the ldmatrix data is in registers before the stage is released (an arrive does not
      fence_proxy_async();                          // wait for this thread's loads: the next TMA would race them)
      __syncwarp();
      if (lane == 0) arrive(bem(bs));
      if (++bs == NB) { bs = 0; bph ^= 1; }
    }
    wait(kvf(gst), gph);
    wg_fence();
    mma_h16_n128(sc[r], qa[r][0], desc_off(kd0, gst * STG * 2));
    mma_h16_n128(sc[r], qa[r][1], desc_off(kd0, gst * STG * 2 + 32));
    wg_commit();
    if (++gst == NKV) { gst = 0; gph ^= 1; }
  };
  // PV of the previous sub-step (row r, stage xst): [O_r | l_r] += P_r [V | 1], issued and committed
  auto PV = [&](int r) {
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const uint32_t a[4] = {sc[r][4 * kk], sc[r][4 * kk + 1], sc[r][4 * kk + 2], sc[r][4 * kk + 3]};
      mma_h16_n40(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 1024));
    }
    wg_commit();
  };
  auto release = [&]() {
    __syncwarp();
    if (lane == 0) arrive(kve(xst));
    if (++xst == NKV) xst = 0;
  };
  // X phase of row r on sc[r] (its QK^T retired; o[r] idle: the PV in flight is the other row's): the offsets, the exponentials in place.
  // Exact form, and a row without a finite logit yet: the tile's row maximum (a tree of HMNMX2) raises the offset lazily.
  auto X = [&](int r) {
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
    if (!SPEC) { tree(); exact = true; }
    else if (VOTE) {                               // the exact path only when a lane of the warp sees a logit past its row's offset + TA_TAU (or a row has none yet)
      tree();
      const float p0 = fmaxf(hlo(t0[0]), hhi(t0[0])) * scl, p1 = fmaxf(hlo(t1[0]), hhi(t1[0])) * scl;
      exact = __any_sync(0xffffffffu, !(m[r][0] > -INFINITY) | !(m[r][1] > -INFINITY) | (p0 > m[r][0] + TA_TAU) | (p1 > m[r][1] + TA_TAU));
    } else {
      exact = __any_sync(0xffffffffu, !(m[r][0] > -INFINITY) | !(m[r][1] > -INFINITY));
      if (exact) tree();
    }
    if (exact) {
      float m0 = fmaxf(hlo(t0[0]), hhi(t0[0])), m1 = fmaxf(hlo(t1[0]), hhi(t1[0]));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m0 *= scl; m1 *= scl;                         // -inf when the tile's 128 keys are all masked
      const bool u0 = m0 > m[r][0] + TA_TAU, u1 = m1 > m[r][1] + TA_TAU;     // m = -inf until a finite logit was seen
      if (__any_sync(0xffffffffu, u0 | u1)) {
        const float n0 = u0 ? hround(m0 + (LIST ? TA_MARGIN : 0.f)) : m[r][0], n1 = u1 ? hround(m1 + (LIST ? TA_MARGIN : 0.f)) : m[r][1];
        const float f0 = u0 ? ex2f(m[r][0] - n0) : 1.f, f1 = u1 ? ex2f(m[r][1] - n1) : 1.f;     // 2^(-inf) = 0 scales an empty sum
#pragma unroll
        for (int i = 0; i < 5; ++i) { o[r][4 * i] *= f0; o[r][4 * i + 1] *= f0; o[r][4 * i + 2] *= f1; o[r][4 * i + 3] *= f1; }
        m[r][0] = n0; m[r][1] = n1;
        if (LIST) { lprev[r][0] = o[r][16]; lprev[r][1] = o[r][18]; }
      }
    }
    const uint32_t nm0 = hdup(m[r][0] > -INFINITY ? -m[r][0] : 0.f), nm1 = hdup(m[r][1] > -INFINITY ? -m[r][1] : 0.f);
#pragma unroll
    for (int i = 0; i < 16; ++i) { sc[r][2 * i] = hexp(hfma2(sc[r][2 * i], scl2, nm0), i); sc[r][2 * i + 1] = hexp(hfma2(sc[r][2 * i + 1], scl2, nm1), i); }
  };
  // after the PV of a tile of row rr retired: its row-sum increment is at least its largest probability; past 2^TA_TS the offset rises by the
  // power of two that brings it back to about 1 (o and l scaled exactly); a non-finite one (an fp16 probability overflowed) cannot be repaired
  // here: the task goes on the fix list
  auto check = [&](int rr) {
    const float i0 = o[rr][16] - lprev[rr][0], i1 = o[rr][18] - lprev[rr][1];
    const float lim = exp2f(TA_TS);
    if (__any_sync(0xffffffffu, !(i0 <= lim) | !(i1 <= lim))) {
      const bool b0 = !(i0 <= lim), b1 = !(i1 <= lim);
      const bool f0 = b0 && i0 < 3.0e38f, f1 = b1 && i1 < 3.0e38f;
      ovf |= (b0 && !f0) || (b1 && !f1);
      const float n0 = f0 ? hround(m[rr][0] + ceilf(__log2f(i0)) + TA_MARGIN) : m[rr][0], n1 = f1 ? hround(m[rr][1] + ceilf(__log2f(i1)) + TA_MARGIN) : m[rr][1];
      const float g0 = ex2f(m[rr][0] - n0), g1 = ex2f(m[rr][1] - n1);
#pragma unroll
      for (int i = 0; i < 5; ++i) { o[rr][4 * i] *= g0; o[rr][4 * i + 1] *= g0; o[rr][4 * i + 2] *= g1; o[rr][4 * i + 3] *= g1; }
      m[rr][0] = n0; m[rr][1] = n1;
    }
    lprev[rr][0] = o[rr][16]; lprev[rr][1] = o[rr][18];
  };
  // one pipelined sub-step (tile j, row r): QK^T of row r and PV of the previous sub-step (row 1 - r) issued, the exponentials of row r
  // run while that PV executes; nothing is left in flight at the end
  auto step = [&](int r) {
    G(r);
    PV(1 - r);
    wg_wait<1>();
    fence_regs(sc[r]);
    X(r);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    if (LIST) check(1 - r);
    release();
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
    // Q fragments of the warp's 16 rows (A of the QK^T wgmma), two k16 blocks per pair row
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) ldsm_x4(qlane + r * QT * 2 + (((2 * ks + lcol8) ^ ((lrow >> 1) & 3)) << 4), qa[r][ks]);
#pragma unroll
    for (int r = 0; r < R; ++r) { fence_regs(qa[r][0]); fence_regs(qa[r][1]); }   // loaded before sQ is released
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) arrive(qe);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int i = 0; i < 20; ++i) o[r][i] = 0.f;
      m[r][0] = m[r][1] = -INFINITY;               // no finite logit seen yet
      lprev[r][0] = lprev[r][1] = 0.f;
    }
    ovf = 0;
    G(0);                                            // sub-step (0, 0): no previous PV
    wg_wait<0>();
    fence_regs(sc[0]);
    X(0);
    uint4 gpre[R][2];                                // the gate's values of this lane's output columns, loaded under the last key tile
    for (int j = 0; j < NK; ++j) {
      if (GATE != nullptr && j == NK - 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int n = min(g * R + r, N - 1);
#pragma unroll
          for (int half = 0; half < 2; ++half)
            gpre[r][half] = __ldg(reinterpret_cast<const uint4*>(GATE + ((size_t)row0(b, n) + qt * BM + qrow + g8 + 8 * half) * ldg + h * D + 8 * t4));
        }
      }
      step(1);
      if (j + 1 < NK) step(0);
    }
    PV(1);                                           // the last sub-step's PV
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    if (LIST) check(1);
    release();
    // ---- epilogue: O_r / l_r -> bf16 -> out (a quad transposes its 4 x 4 words so each lane stores 16 contiguous bytes); LSE
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = g * R + r;
      if (n >= N) continue;                          // a padding row: computed on row N - 1, not stored
      const float l0 = o[r][16], l1 = o[r][18];
      bad |= !(l0 > 0.f) || !(l0 < 3.0e38f) || !(l1 > 0.f) || !(l1 < 3.0e38f);
      ovf |= !(l0 < 3.0e38f) || !(l1 < 3.0e38f);
      const float inv0 = 1.f / l0, inv1 = 1.f / l1;
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
        for (int i = 0; i < 4; ++i) in[i] = pack2(o[r][4 * i + 2 * half] * inv, o[r][4 * i + 2 * half + 1] * inv);
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
        if (OUT != nullptr) *reinterpret_cast<uint4*>(OUT + orow * (H * D) + h * D + 8 * t4) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
        if (GATE != nullptr) {
          const uint4 gv = gpre[r][half];
          const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
          uint32_t uw[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 gg = gate_h16 ? make_float2(hlo(gw[i]), hhi(gw[i])) : bf2f(gw[i]), oo = bf2f(outw[i]);
            uw[i] = pack2(oo.x / (1.f + __expf(-gg.x)), oo.y / (1.f + __expf(-gg.y)));
          }
          *reinterpret_cast<uint4*>(U + orow * (H * D) + h * D + 8 * t4) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
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
// could not absorb and the exact launch recomputed; u = sigmoid(gate) o out when `gate` is given)
std::vector<torch::Tensor> triattn_fwd_h16(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sbias, double scale, bool want_lse,
                                           c10::optional<torch::Tensor> gate, bool want_out) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 5 && q.size(4) == D, "q [B, N, S, H, 32]");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % BM == 0 && S > 0, "S % 128 == 0");
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v}) {
    TORCH_CHECK(t->scalar_type() == torch::kHalf && t->sizes() == q.sizes() && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld && ld % 8 == 0 && ld >= H * D,
                "q / k / v: [B, N, S, H, 32] fp16 rows at one common stride");
  }
  TORCH_CHECK(sbias.is_cuda() && sbias.scalar_type() == torch::kHalf && sbias.is_contiguous() && sbias.numel() == B * H * S * S,
              "sbias: fp16 [B, H, S, S] contiguous (bias / scale)");
  const uint64_t rows = (uint64_t)(B * N * S);
  auto m2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)(H * D), rows}, {(uint64_t)ld}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_64B, what,
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT16); };
  CUtensorMap qm = m2(q, "q"), km = m2(k, "k"), vm = m2(v, "v");
  CUtensorMap bm = make_map<2>(sbias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B,
                               "sbias", CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  const bool has_gate = gate.has_value();
  const bool store_out = want_out || !has_gate;
  TORCH_CHECK(!has_gate || ((gate->scalar_type() == torch::kBFloat16 || gate->scalar_type() == torch::kHalf) && gate->is_cuda() && gate->sizes() == q.sizes() && gate->stride(4) == 1 &&
              gate->stride(3) == D && gate->stride(2) % 8 == 0 && gate->stride(1) == S * gate->stride(2) && gate->stride(0) == N * S * gate->stride(2)),
              "gate: [B, N, S, H, 32] bf16 or fp16, rows at one stride");
  const auto bf = q.options().dtype(torch::kBFloat16);
  auto out = torch::empty({store_out ? B : 0, N, S, H, D}, bf);
  auto u = torch::empty({has_gate ? B : 0, N, S, H, D}, bf);
  auto lse = torch::empty({want_lse ? B : 0, N, H, S}, q.options().dtype(torch::kFloat32));
  auto& flags = flags_for(q.device());
  static bool attr = false;
  if (!attr) {
    for (auto kk : {triattn_h16_sm90<false, false>, triattn_h16_sm90<false, true>, triattn_h16_sm90<true, false>, triattn_h16_sm90<true, true>})
      C10_CUDA_CHECK(cudaFuncSetAttribute(kk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    attr = true;
  }
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
      has_gate ? reinterpret_cast<__nv_bfloat16*>(u.data_ptr<at::BFloat16>()) : nullptr, has_gate ? (long)gate->stride(2) : 0L, sch, fx);
  };
  if (spec) {
    launch(ord1 ? triattn_h16_sm90<true, true> : triattn_h16_sm90<true, false>, dyn ? sched.data_ptr<int>() : nullptr, list ? fix.data_ptr<int>() : nullptr);
    if (list) {
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      launch(ord1 ? triattn_h16_sm90<false, true> : triattn_h16_sm90<false, false>, nullptr, fix.data_ptr<int>());
    }
  } else {
    launch(ord1 ? triattn_h16_sm90<false, true> : triattn_h16_sm90<false, false>, dyn ? sched.data_ptr<int>() : nullptr, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags, u};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_fwd_h16", &triattn_fwd_h16, "sm90 triangle-attention forward, inference, fp16 operands (D = 32): q/k/v [B, N, S, H, 32] fp16, "
        "sbias = bias / scale fp16 [B, H, S, S] -> out bf16, optional base-2 LSE; with `gate` also u = sigmoid(gate) o out",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sbias"), py::arg("scale"), py::arg("want_lse"), py::arg("gate") = py::none(),
        py::arg("want_out") = true);
}
