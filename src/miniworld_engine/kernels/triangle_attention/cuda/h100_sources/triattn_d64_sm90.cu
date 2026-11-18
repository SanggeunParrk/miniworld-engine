// triattn_d64_sm90.cu -- triangle-attention core for heads of 64 channels (AF3 d_pair 512 x 8 heads, RoseTTAFold3's 4 x 64 heads, Chai-1's
// d_pair 256 trunk / confidence heads), inference, sm_90a, wgmma + TMA (raw PTX, sm90.cuh), fp16 operands:
//
//   u[b,n,s,h,:] = sigmoid(g[b,n,s,h,:] (+ gb[h,:])) * softmax_k( scale q[b,n,s,h,:] . k[b,n,k,h,:] + bias[b,h,s,k] ) @ v[b,n,k,h,:]
//
// Operands: proj [B N S][4 HD] fp16 (HD = 64 H): q at column h 64, k at HD + h 64, v at 2 HD + h 64, g at 3 HD + h 64; sbias [B, H, S, S]
// fp16 = bias / scale (masked keys -inf), shared by the N = S pair rows; u [B N S][HD] bf16.
//
// The design of the 32-channel fp16 core (triattn_h16_sm90.cu) at head width 64: a CTA task = one (b, h, 128-query tile) x R = 2 pair rows;
// 384 threads: warpgroup 0 = TMA producer (one thread), warpgroups 1 / 2 = consumers of queries 0-63 / 64-127 of both rows.  Per BN-key tile
// the producer stages the bias tile once (shared by the two rows) and K | V of each row; a consumer alternates the rows (G = the bias
// fragment as the fp16 score accumulator's initial value + S += Q_r K^T issued; the PV of the other row issued; X = the exponentials of
// row r while that PV runs):
//   - the score GEMM accumulates in fp16 (m64nBNk16.f16): no bf16 -> fp32 unpack of the bias per logit, half the accumulator registers;
//   - x = s scale log2(e) - m (HFMA2) and p = 2^x (MUFU.EX2 .f16x2, every other pair on the FMA pipes' polynomial) run on pairs: the MUFU
//     work per logit is a quarter of the fp32 softmax's -- at head width 64 the fp32 exponentials alone were as long as the tensor work;
//   - p is already the A fragment of [O | l] += P [V | 1] (m64n72k16, fp16 x fp16 -> fp32; columns 64..71 = one shared block of ones,
//     reached through each V descriptor's leading-byte offset): no fp32 -> 16-bit pack, the row sums from the tensor core;
//   - the row offset rises lazily (o and l rescaled, p <= 2^(TE_TAU + 1)): the full row maximum (shuffles) only when a lane of the warp
//     sees a logit past its row's offset + TE_TAU, or a row has no finite logit yet.
// The Q tiles of a task travel through the K | V ring (no separate Q buffer: room for a fourth stage); each Q stage also carries the task id
// and its live key tiles for the consumers (the producer takes tasks from a global counter: dynamic schedule).
// Precision: q / k / v / sbias in fp16 (11-bit mantissa, made from the fp32 GEMM accumulators: the bf16 core's operands had 8 bits), the score
// sum rounded to fp16 once per k16 step, P in fp16.  Range: |q . k + bias / scale| < 65504 (the front clamps the bias at +-6e4).
// KT[b] (key_tiles in BN-key tiles) bounds sample b's key tiles: the tiles past its last attended key are -inf in every row (p = 0), so
// skipping them gives the same result.  A row without an attended key gives u = 0.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "sm90.cuh"
using namespace sm90;

#ifndef TE_BN
#define TE_BN 128         // keys per tile (128 or 64)
#endif
#ifndef TE_NKV
#define TE_NKV (TE_BN == 128 ? 4 : 8)     // K | V ring stages (one per (key tile, row); a task's Q tiles take R * 16 KiB of them)
#endif
#ifndef TE_NB
#define TE_NB (TE_BN == 128 ? 2 : 3)      // bias ring stages (one per key tile)
#endif
#ifndef TE_F8PV
#define TE_F8PV 0         // opt-in FP8 variant: P and V on the e4m3 tensor-core path (the 32-channel core's F8PV design, triattn_h16_sm90.cu): V arrives as e4m3 V^T planes (vt_quant), p = 2^(x + 8) as e4m3 bytes; built with -DTE_F8PV=1
#endif
#if TE_F8PV
#define TE_TAU 0.5f       // F8PV: p = 2^(logit - m + 8) must stay below 448: the offset rises as soon as a logit exceeds it by 0.5 log2 units
#endif
#ifndef TE_TAU
#define TE_TAU 8.f        // a tile raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
#ifndef TE_POLY
#define TE_POLY 8         // exponential pairs of every 16 on the FMA-pipe polynomial (0, 1, 2, 4, 8, 16), the rest on the MUFU
#endif
#ifndef TE_BIAS_L2
#define TE_BIAS_L2 (20 << 20)   // the CTAs running together work on the largest group of heads whose bias (hg S^2 fp16) fits this many bytes
#endif
#ifndef TE_DYN
#define TE_DYN 1          // dynamic task schedule (a global counter)
#endif
#ifndef TE_GPF
#define TE_GPF 1          // the epilogue's gate rows prefetched into L1 before the last sub-step
#endif
#ifndef TE_PP
#define TE_PP 0           // 1: FlashAttention-3 pingpong of the two consumer warpgroups' wgmma issue (named barriers)
#endif
namespace {
typedef __nv_bfloat16 bf16;
constexpr int D = 64, BM = 128, BN = TE_BN, R = 2, NKV = TE_NKV, NB = TE_NB;
static_assert(BN == 128 || BN == 64, "TE_BN 128 or 64");
constexpr int QT = BM * D;                                 // Q tile [128 q][64], 128B swizzle (16 KiB)
constexpr int KVT = BN * D;                                // K or V tile [BN k][64], 128B swizzle
constexpr int STG = 2 * KVT;                               // a K | V stage (elements)
constexpr int RPS = STG / QT;                              // Q tiles a stage holds (2 at BN 128, 1 at BN 64)
constexpr int QST = R / RPS;                               // stages a task's Q tiles take
constexpr int BT = BM * BN;                                // bias tile: fp16 [BN / 64 key halves][128 q][64], 128B swizzle
constexpr int ONES = 16 * 64;                              // the ones block: [16 k][64] fp16 (2 KiB), the PV's columns 64..71
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + (NB * BT + NKV * STG + ONES) * 2 + 256;
static_assert(SMEM <= 232448, "one CTA per SM");
static_assert(8 * (2 * NKV + 2 * NB) + 8 * NKV <= 256, "barriers + stage info");

#define H16_U8(b) "+r"(d[b]), "+r"(d[b + 1]), "+r"(d[b + 2]), "+r"(d[b + 3]), "+r"(d[b + 4]), "+r"(d[b + 5]), "+r"(d[b + 6]), "+r"(d[b + 7])
// S += Q . K^T with an fp16 accumulator, A = Q from registers (fp16 pairs), B = K from shared (K-major); d[2 i + h] = row 8 h + lane / 4 of
// the warp's 16, columns 8 i + 2 (lane % 4) + {0, 1}
__device__ __forceinline__ void mma_qk(uint32_t (&d)[32], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
               : H16_U8(0), H16_U8(8), H16_U8(16), H16_U8(24) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
__device__ __forceinline__ void mma_qk(uint32_t (&d)[16], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
               : H16_U8(0), H16_U8(8) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// [O | l] += P [V | 1]: m64n72k16, fp16 x fp16 -> fp32, A = P from registers, B = V (MN-major) with the ones block as columns 64..71
__device__ __forceinline__ void mma_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
               "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
               "{%36, %37, %38, %39}, %40, p, 1, 1, 1;\n}\n"
               : SM90_D32(0), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
#if TE_F8PV
// F8PV: [O | l] += P [V | 1] on e4m3: m64n72k32, fp32 accumulation, A = P from registers (4 x 4 e4m3), B = [V^T (64 rows); ones (8 rows)] K-major from shared (72 rows of 128 keys)
__device__ __forceinline__ void mma_f8_n72(float (&d)[36], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\n}\n"
               : SM90_D32(0), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
#endif
// a descriptor moved by `bytes` with its leading-byte offset shortened by as much: the block at start + LBO (the ones) stays put
__device__ __forceinline__ uint64_t desc_mv(uint64_t d, uint32_t bytes) {
  uint64_t r;
  const uint32_t u = bytes >> 4;
  asm("{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, %1;\nadd.u32 lo, lo, %2;\nmov.b64 %0, {lo, hi};\n}\n" : "=l"(r) : "l"(d), "r"(u - (u << 16)));
  return r;
}
__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) { uint32_t r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t hmin2(uint32_t a, uint32_t b) { uint32_t r; asm("min.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t hex2(uint32_t a) { uint32_t r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(a)); return r; }
// 2^x on f16x2 pairs on the FMA / ALU pipes (triattn_h16_sm90.cu's hex2_poly): x clamped at -15 (2^-15 flushes to 0), j = round(x),
// 2^(x - j) by a degree-3 polynomial (~1.2e-4 relative), times 2^j through the exponent field
__device__ __forceinline__ uint32_t hex2_poly(uint32_t x) {
  uint32_t t, jf, f, p, s, r;
  asm("max.f16x2 %0, %1, %2;" : "=r"(x) : "r"(x), "r"(0xCB80CB80u));
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(t) : "r"(x), "r"(0x660F660Fu));
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(jf) : "r"(t), "r"(0xE60FE60Fu));
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(f) : "r"(x), "r"(jf));
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(p) : "r"(0x2B1B2B1Bu), "r"(f), "r"(0x33C433C4u));
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(p) : "r"(p), "r"(f), "r"(0x398C398Cu));
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(p) : "r"(p), "r"(f), "r"(0x3C003C00u));
  s = (t << 10) & 0x7C007C00u;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(p), "r"(s));
  return r;
}
template <int NP> __device__ __forceinline__ uint32_t hexp(uint32_t a, int i) {   // NP of every 16 pairs on the polynomial
  if (NP > 0 && i % (16 / (NP > 0 ? NP : 1)) == (16 / (NP > 0 ? NP : 1)) - 1) return hex2_poly(a);
  return hex2(a);
}
__device__ __forceinline__ float hlo(uint32_t a) { return __half2float(__ushort_as_half((unsigned short)(a & 0xffffu))); }
__device__ __forceinline__ float hhi(uint32_t a) { return __half2float(__ushort_as_half((unsigned short)(a >> 16))); }
__device__ __forceinline__ uint32_t hdup(float x) { const unsigned short h = __half_as_ushort(__float2half_rn(x)); return (uint32_t)h | ((uint32_t)h << 16); }
__device__ __forceinline__ float hround(float x) { return __half2float(__float2half_rn(x)); }
}  // namespace

template <bool ORD1>      // ORD1: task order (query tile, head in its group of hg, row group, head group); else (query tile, head, row group)
__global__ void __launch_bounds__(THREADS, 1) triattn_d64_sm90(
    int N, int H, int S, int ntasks, float scl,          // scl = scale log2(e)
    int hg,
    const __grid_constant__ CUtensorMap qmap,            // proj [B N S][4 H 64] fp16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap kvmap,           // the same tensor, box (64, BN)
    const __grid_constant__ CUtensorMap bmap,            // sbias [B H S][S] fp16, box (64, 128), 128B swizzle
#if TE_F8PV
    const __grid_constant__ CUtensorMap vtmap,           // V^T e4m3 [B N H 64 rows][S keys] (keys permuted inside each group of 32: vt_quant), box (128, 64), 128B swizzle
    const float* __restrict__ VS,                        // per (b, h) V scale fp32 [B H]
#endif
    const __half* __restrict__ P, long ld,               // proj (the gate's columns), its row stride
    const float* __restrict__ GB,                        // gate bias fp32 [H 64], or nullptr
    bf16* __restrict__ U,                                // [B N S][H 64]
    const int* __restrict__ KT,                          // [B] live BN-key tiles per sample, or nullptr (S / BN)
    int* __restrict__ SCHED) {                           // zeroed task counter (dynamic schedule), or nullptr: static
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R, HD = H * D;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  // shared layout: bias ring [NB][BT] | K | V ring [NKV][K | V] | ones | barriers + stage info
  const uint32_t sB = sa(smb), sKV = sB + NB * BT * 2, sOnes = sKV + NKV * STG * 2, bar0 = sOnes + ONES * 2;
  auto kvf = [&](int s) { return bar0 + 8 * s; };
  auto kve = [&](int s) { return bar0 + 8 * (NKV + s); };                    // count 8: the stage's reads retired in every consumer warp
  auto bfu = [&](int s) { return bar0 + 8 * (2 * NKV + s); };
  auto bem = [&](int s) { return bar0 + 8 * (2 * NKV + NB + s); };           // count 8: every consumer warp read its bias fragments
  volatile int2* info = reinterpret_cast<volatile int2*>(smb + (bar0 - sB) + 8 * (2 * NKV + 2 * NB));   // [NKV]: (task, key tiles) of a Q stage
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  for (int i = tid; i < ONES / 2; i += THREADS) reinterpret_cast<uint32_t*>(smb + (sOnes - sB))[i] = 0x3C003C00u;   // fp16 pairs (1, 1)
#if TE_F8PV
  for (int i = tid; i < NKV * 256; i += THREADS) {          // F8PV: the 8 ones rows (e4m3 1.0 = 0x38) behind each stage's 64 V^T rows (1 KiB)
    const int s = i / 256, e = i % 256;
    reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (s * STG + KVT) * 2 + 128 * 64)[e] = 0x38383838u;
  }
#endif
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sB));
    for (int k = 0; k < NKV; ++k) { bar_init(bars + k, 1); bar_init(bars + NKV + k, 8); }
    for (int k = 0; k < NB; ++k) { bar_init(bars + 2 * NKV + k, 1); bar_init(bars + 2 * NKV + NB + k, 8); }
    bar_init_fence();
  }
  fence_proxy_async();                                      // the ones are read by the tensor core (async proxy)
  __syncthreads();
  auto decode = [&](int t, int& b, int& h, int& qt, int& g) {
    qt = t % QTILES; int r = t / QTILES;
    if (ORD1) { const int hl = r % hg; r /= hg; g = r % NG; r /= NG; h = (r % (H / hg)) * hg + hl; b = r / (H / hg); }
    else      { h = r % H; r /= H; g = r % NG; b = r / NG; }
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&qmap); prefetch_map(&kvmap); prefetch_map(&bmap);
      int st = 0, kph = 0, bs = 0, bph = 0;
#if TE_F8PV
      bool qdirty[NKV];   // F8PV: a stage that carried Q tiles lost its ones rows (the Q tiles fill the whole stage): rewritten before its next K / V^T load
#pragma unroll
      for (int s = 0; s < NKV; ++s) qdirty[s] = false;
#endif
      auto adv = [&]() { if (++st == NKV) { st = 0; kph ^= 1; } };
      for (int t = blockIdx.x; t < ntasks; t = SCHED ? (int)gridDim.x + atomicAdd(SCHED, 1) : t + (int)gridDim.x) {
        int b, h, qt, g; decode(t, b, h, qt, g);
        const int nk = KT != nullptr ? min(NK, max(1, __ldg(KT + b))) : NK;
        for (int qs = 0; qs < QST; ++qs) {             // the Q tiles of both rows, with the task's id and key tiles
          wait(kve(st), kph ^ 1);
          info[st].x = t; info[st].y = nk;
          expect_tx(kvf(st), RPS * QT * 2);
          for (int rr = 0; rr < RPS; ++rr) load_2d(&qmap, sKV + (st * STG + rr * QT) * 2, kvf(st), h * D, row0(b, g * R + qs * RPS + rr) + qt * BM);
#if TE_F8PV
          qdirty[st] = true;
#endif
          adv();
        }
        for (int j = 0; j < nk; ++j) {
          wait(bem(bs), bph ^ 1);
          expect_tx(bfu(bs), BT * 2);
          for (int hh = 0; hh < BN / 64; ++hh) load_2d(&bmap, sB + (bs * BT + hh * BM * 64) * 2, bfu(bs), j * BN + hh * 64, (b * H + h) * S + qt * BM);
          if (++bs == NB) { bs = 0; bph ^= 1; }
          for (int r = 0; r < R; ++r) {
            wait(kve(st), kph ^ 1);
#if TE_F8PV
            if (qdirty[st]) {
              uint32_t* ones = reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (st * STG + KVT) * 2 + 128 * 64);
              for (int e = 0; e < 256; ++e) ones[e] = 0x38383838u;
              fence_proxy_async();
              qdirty[st] = false;
            }
            expect_tx(kvf(st), KVT * 2 + 64 * BN);                 // K (fp16) + the V^T tile (64 rows x BN keys, e4m3)
            const int rw = row0(b, g * R + r) + j * BN;
            load_2d(&kvmap, sKV + st * STG * 2, kvf(st), HD + h * D, rw);
            load_2d(&vtmap, sKV + (st * STG + KVT) * 2, kvf(st), j * BN, ((b * N + min(g * R + r, N - 1)) * H + h) * 64);
#else
            expect_tx(kvf(st), STG * 2);
            const int rw = row0(b, g * R + r) + j * BN;
            load_2d(&kvmap, sKV + st * STG * 2, kvf(st), HD + h * D, rw);
            load_2d(&kvmap, sKV + (st * STG + KVT) * 2, kvf(st), 2 * HD + h * D, rw);
#endif
            adv();
          }
        }
      }
      wait(kve(st), kph ^ 1);                          // the end: a task-free stage
      info[st].x = -1;
      arrive(kvf(st));
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  if (TE_PP && w == 1) named_arrive(2, 256);
  const int qrow = 64 * w + 16 * wi;               // this warp's first query row in the tile
  const int lrow = qrow + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4, x7 = lane & 7;
  const uint32_t blane = sB + lrow * 128, qlane = sKV + lrow * 128;
  const uint64_t kd0 = desc_k128(sKV);
  #if TE_F8PV
  const uint64_t vd0 = desc_k128(sKV + KVT * 2);         // F8PV: V^T rows K-major (128 keys = 128 bytes per row), the ones rows right behind the 64 V^T rows
#else
  const uint64_t vd0 = gdesc(sKV + KVT * 2, sOnes - (sKV + KVT * 2), 1024, SW128);
#endif   // V of stage 0, its columns 64..71 the ones block
  const uint32_t scl2 = hdup(scl);
  int gst = 0, gph = 0, bs = 0, bph = 0, xst = 0;
  uint32_t qa[R][4][4];
  float o[R][36], m[R][2];                         // [O | l] per row (o[32] = row g's sum, o[34] = row g + 8's); the offsets (log2 units)
  uint32_t sc[R][BN / 4];                          // per row: the score accumulator (fp16 pairs), then its probabilities = its PV's A fragments
  auto G = [&](int r) {
    if (r == 0) wait(bfu(bs), bph);
    const uint32_t btile = blane + bs * BT * 2;
#pragma unroll
    for (int i2 = 0; i2 < BN / 16; ++i2) {
      uint32_t r4[4];
      ldsm_x4(btile + (i2 >> 2) * BM * 128 + ((((2 * i2 + lcol8) & 7) ^ x7) << 4), r4);
#pragma unroll
      for (int e = 0; e < 4; ++e) sc[r][4 * i2 + e] = r4[e];
    }
    if (r == R - 1) {
      fence_regs(sc[r]);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(bem(bs));
      if (++bs == NB) { bs = 0; bph ^= 1; }
    }
    wait(kvf(gst), gph);
    if (TE_PP) named_sync(2 + w, 256);
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma_qk(sc[r], qa[r][ks], desc_off(kd0, gst * STG * 2 + 32 * ks));
    wg_commit();
    if (++gst == NKV) { gst = 0; gph ^= 1; }
  };
  auto PV = [&](int r) {
#if TE_F8PV
    // P is already e4m3 bytes (X): the A fragment of a k32 step is sc[r][4 kk .. 4 kk + 3]; a k32 step = +32 bytes in the 128-byte V^T row
#pragma unroll
    for (int kk = 0; kk < BN / 32; ++kk) {
      const uint32_t a[4] = {sc[r][4 * kk], sc[r][4 * kk + 1], sc[r][4 * kk + 2], sc[r][4 * kk + 3]};
      mma_f8_n72(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 32));
    }
#else
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const uint32_t a[4] = {sc[r][4 * kk], sc[r][4 * kk + 1], sc[r][4 * kk + 2], sc[r][4 * kk + 3]};
      mma_pv(o[r], a, desc_mv(vd0, xst * STG * 2 + kk * 2048));
    }
#endif
    wg_commit();
  };
  auto release = [&]() {
    __syncwarp();
    if (lane == 0) arrive(kve(xst));
    if (++xst == NKV) xst = 0;
  };
  // X phase of row r on sc[r] (its QK^T retired; o[r] idle: the PV in flight is the other row's): the offsets, the exponentials in place
  auto X = [&](int r) {
    constexpr int NT = BN / 16;
    uint32_t t0[NT], t1[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) { t0[i] = hmax2(sc[r][4 * i], sc[r][4 * i + 2]); t1[i] = hmax2(sc[r][4 * i + 1], sc[r][4 * i + 3]); }
#pragma unroll
    for (int n = NT / 2; n >= 1; n >>= 1)
#pragma unroll
      for (int i = 0; i < n; ++i) { t0[i] = hmax2(t0[i], t0[i + n]); t1[i] = hmax2(t1[i], t1[i + n]); }
    const float p0 = fmaxf(hlo(t0[0]), hhi(t0[0])) * scl, p1 = fmaxf(hlo(t1[0]), hhi(t1[0])) * scl;
    if (__any_sync(0xffffffffu, !(m[r][0] > -INFINITY) | !(m[r][1] > -INFINITY) | (p0 > m[r][0] + TE_TAU) | (p1 > m[r][1] + TE_TAU))) {
      float m0 = fmaxf(hlo(t0[0]), hhi(t0[0])), m1 = fmaxf(hlo(t1[0]), hhi(t1[0]));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m0 *= scl; m1 *= scl;                         // -inf when the tile's keys are all masked
      const bool u0 = m0 > m[r][0] + TE_TAU, u1 = m1 > m[r][1] + TE_TAU;     // m = -inf until a finite logit was seen
      if (__any_sync(0xffffffffu, u0 | u1)) {
        const float n0 = u0 ? hround(m0) : m[r][0], n1 = u1 ? hround(m1) : m[r][1];
        const float f0 = u0 ? ex2f(m[r][0] - n0) : 1.f, f1 = u1 ? ex2f(m[r][1] - n1) : 1.f;     // 2^(-inf) = 0 scales an empty sum
#pragma unroll
        for (int i = 0; i < 9; ++i) { o[r][4 * i] *= f0; o[r][4 * i + 1] *= f0; o[r][4 * i + 2] *= f1; o[r][4 * i + 3] *= f1; }
        m[r][0] = n0; m[r][1] = n1;
      }
    }
#if TE_F8PV
    // p = 2^(x + 8) directly as e4m3 bits (triattn_h16_sm90.cu F8conv): x = logit - m + 8, clamped at -7, byte = round(8 x + 56); written in place as the PV steps' A fragments
    const uint32_t nb0 = hdup(m[r][0] > -INFINITY ? 8.f - m[r][0] : 0.f), nb1 = hdup(m[r][1] > -INFINITY ? 8.f - m[r][1] : 0.f);
#pragma unroll
    for (int kk = 0; kk < BN / 32; ++kk) {
#pragma unroll
      for (int i = 4 * kk; i < 4 * kk + 4; ++i) {
        uint32_t x0 = hfma2(sc[r][2 * i], scl2, nb0), x1 = hfma2(sc[r][2 * i + 1], scl2, nb1);
        x0 = hmax2(x0, 0xC700C700u); x1 = hmax2(x1, 0xC700C700u);                       // x >= -7
        x0 = hmin2(x0, 0x48604860u); x1 = hmin2(x1, 0x48604860u);                       // x <= 8.75: byte <= 126 = 448 (127 is the e4m3 NaN; with large fp16 logits the offset's rounding can put x past 8.875)
        x0 = hfma2(x0, 0x48004800u, 0x64386438u); x1 = hfma2(x1, 0x48004800u, 0x64386438u);   // 8 x + 56 + 1024
        sc[r][2 * i] = x0; sc[r][2 * i + 1] = x1;
      }
      uint32_t pk[4];   // k32 step kk: a0 = row g keys (blocks 4 kk, 4 kk + 1), a1 = row g + 8, a2 / a3 = blocks 4 kk + 2, 4 kk + 3
#pragma unroll
      for (int j = 0; j < 4; ++j) asm("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(pk[j]) : "r"(sc[r][8 * kk + (j & 1) + 4 * (j >> 1)]), "r"(sc[r][8 * kk + (j & 1) + 4 * (j >> 1) + 2]));
#pragma unroll
      for (int j = 0; j < 4; ++j) sc[r][4 * kk + j] = pk[j];
    }
#else
    const uint32_t nm0 = hdup(m[r][0] > -INFINITY ? -m[r][0] : 0.f), nm1 = hdup(m[r][1] > -INFINITY ? -m[r][1] : 0.f);
#pragma unroll
    for (int i = 0; i < BN / 8; ++i) {
      sc[r][2 * i] = hexp<TE_POLY>(hfma2(sc[r][2 * i], scl2, nm0), i);
      sc[r][2 * i + 1] = hexp<TE_POLY>(hfma2(sc[r][2 * i + 1], scl2, nm1), i);
    }
#endif
  };
  auto step = [&](int r) {
    G(r);
    PV(1 - r);
    if (TE_PP) named_arrive(2 + (w ^ 1), 256);
    wg_wait<1>();
    fence_regs(sc[r]);
    X(r);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    release();
  };
  for (;;) {
    wait(kvf(gst), gph);
    const int t = info[gst].x, nk = info[gst].y;
    if (t < 0) break;
    int b, h, qt, g; decode(t, b, h, qt, g);
    // Q fragments of the warp's 16 rows of both pair rows (A of the QK^T wgmma), four k16 blocks each, from the task's Q stage(s)
#pragma unroll
    for (int qs = 0; qs < QST; ++qs) {
      if (qs > 0) wait(kvf(gst), gph);
#pragma unroll
      for (int rr = 0; rr < RPS; ++rr)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          ldsm_x4(qlane + (gst * STG + rr * QT) * 2 + (((2 * ks + lcol8) ^ x7) << 4), qa[qs * RPS + rr][ks]);
#pragma unroll
      for (int rr = 0; rr < RPS; ++rr)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) fence_regs(qa[qs * RPS + rr][ks]);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(kve(gst));
      if (++gst == NKV) { gst = 0; gph ^= 1; }
      if (++xst == NKV) xst = 0;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int i = 0; i < 36; ++i) o[r][i] = 0.f;
      m[r][0] = m[r][1] = -INFINITY;
    }
    G(0);                                          // sub-step (0, 0): no previous PV
    if (TE_PP) named_arrive(2 + (w ^ 1), 256);
    wg_wait<0>();
    fence_regs(sc[0]);
    X(0);
    // the last key tile peeled (as TA_RC in triattn_h16_sm90.cu: no loop path keeps a retired row's probabilities live)
    for (int j = 0; j + 1 < nk; ++j) { step(1); step(0); }
    if (TE_GPF) {   // the gate rows of the epilogue into L1 under the last sub-step (lane: pair row lane / 16, query row lane % 16 of the warp)
      const long grow = (long)row0(b, g * R + (lane >> 4)) + qt * BM + qrow + (lane & 15);
      asm volatile("prefetch.global.L1 [%0];\n" :: "l"(P + grow * ld + 3 * HD + h * D));
    }
    step(1);
    if (TE_PP) named_sync(2 + w, 256);
    PV(1);                                         // the last sub-step's PV
    if (TE_PP) named_arrive(2 + (w ^ 1), 256);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    release();
    // ---- epilogue: u = sigmoid(g (+ gb)) O / l in fp32, bf16; a quad transposes its words so each lane stores 16 contiguous bytes
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = g * R + r;
      if (n >= N) continue;                      // a padding row: computed on row N - 1, not stored
      const float l0 = o[r][32], l1 = o[r][34];  // accumulator columns 64..71 = P 1: the row sums
#if TE_F8PV
      const float vsc = VS[b * H + h];   // the row sum is of the e4m3 probabilities (the ones rows of the PV): P V8 / l is normalised, times the V scale
      const float inv0 = l0 > 0.f ? vsc / l0 : 0.f, inv1 = l1 > 0.f ? vsc / l1 : 0.f;
#else
      const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
#endif
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float inv = half ? inv1 : inv0;
        const long row = (long)row0(b, n) + qt * BM + qrow + g8 + 8 * half;
        const __half* gp = P + row * ld + 3 * HD + h * D;
#pragma unroll
        for (int grp = 0; grp < 2; ++grp) {
          uint32_t in[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int blk = 4 * grp + i, col = 8 * blk + 2 * t4;
            const uint32_t gw = *reinterpret_cast<const uint32_t*>(gp + col);
            float ga = hlo(gw), gb = hhi(gw);
            if (GB != nullptr) { ga += GB[h * D + col]; gb += GB[h * D + col + 1]; }
            in[i] = pack2(o[r][4 * blk + 2 * half] * inv / (1.f + __expf(-ga)), o[r][4 * blk + 2 * half + 1] * inv / (1.f + __expf(-gb)));
          }
          *reinterpret_cast<uint4*>(U + row * HD + h * D + 32 * grp + 8 * t4) = quad_transpose(in, lane);
        }
      }
    }
  }
}

// proj [B S S, 4 H 64] fp16 contiguous (q | k | v | g), sbias [B, H, S, S] fp16 contiguous (= bias / scale, masked keys -inf), S % 128 == 0;
// gate_bias fp32 [H 64] or empty; key_tiles int32 [B] (BN-key tiles, h100_sm90.key_tiles(mask, S, BN)) or None -> u [B S S, H 64] bf16
at::Tensor triattn_d64_forward(at::Tensor proj, at::Tensor sbias, at::Tensor gate_bias, int64_t H, double scale,
                                std::optional<at::Tensor> key_tiles
#if TE_F8PV
                                , at::Tensor vt, at::Tensor vscale
#endif
                                ) {
  TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::kHalf && proj.is_contiguous() && proj.dim() == 2 && proj.size(1) == 4 * H * D,
              "proj: contiguous fp16 [B S S, 4 H 64]");
  TORCH_CHECK(sbias.scalar_type() == at::kHalf && sbias.is_contiguous() && sbias.dim() == 4 && sbias.size(1) == H && sbias.size(2) == sbias.size(3),
              "sbias: contiguous fp16 [B, H, S, S]");
  const long B = sbias.size(0), S = sbias.size(2), N = S;
  TORCH_CHECK(S % BM == 0 && S > 0 && proj.size(0) == B * N * S, "S a multiple of 128, proj rows B S S");
  TORCH_CHECK(gate_bias.numel() == 0 || (gate_bias.scalar_type() == at::kFloat && gate_bias.is_contiguous() && gate_bias.numel() == H * D),
              "gate_bias: fp32 [H 64] or empty");
  const int* kt = nullptr;
  if (key_tiles.has_value()) {
    TORCH_CHECK(key_tiles->scalar_type() == at::kInt && key_tiles->is_contiguous() && key_tiles->numel() == B && key_tiles->is_cuda(),
                "key_tiles: int32 [B] on the device");
    kt = key_tiles->data_ptr<int>();
  }
  const c10::cuda::OptionalCUDAGuard guard(proj.device());
  const long ld = 4 * H * D;
  const uint64_t rows = (uint64_t)(B * N * S);
  TORCH_CHECK(rows < (1ull << 31) && (uint64_t)(B * H * S) < (1ull << 31), "row coordinates beyond int32");
  const auto f16 = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  CUtensorMap qm = make_map<2>(proj.data_ptr(), {(uint64_t)ld, rows}, {(uint64_t)ld}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "q", f16);
  CUtensorMap kvm = make_map<2>(proj.data_ptr(), {(uint64_t)ld, rows}, {(uint64_t)ld}, {64, (uint32_t)BN}, CU_TENSOR_MAP_SWIZZLE_128B, "k | v", f16);
  CUtensorMap bm = make_map<2>(sbias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "sbias", f16);
#if TE_F8PV
  TORCH_CHECK(vt.is_cuda() && vt.scalar_type() == at::kFloat8_e4m3fn && vt.is_contiguous() && vt.numel() == B * N * H * D * S, "vt: float8_e4m3fn [B, N, H, 64, S] contiguous (keys permuted inside each group of 32)");
  TORCH_CHECK(vscale.is_cuda() && vscale.scalar_type() == at::kFloat && vscale.is_contiguous() && vscale.numel() == B * H, "vscale: fp32 [B, H]");
  CUtensorMap vtm = make_map<2>(vt.data_ptr(), {(uint64_t)S, (uint64_t)(B * N * H * D)}, {(uint64_t)S}, {128, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "vt", CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
#endif
  auto u = at::empty({proj.size(0), H * D}, proj.options().dtype(at::kBFloat16));
  static bool attr = false;
  if (!attr) {
    for (auto k : {triattn_d64_sm90<false>, triattn_d64_sm90<true>})
      C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    attr = true;
  }
  const long ntasks = B * H * ((N + R - 1) / R) * (S / BM);
  TORCH_CHECK(ntasks < (1L << 31), "too many tasks");
  const int grid = (int)std::min<long>(ntasks, num_sms(proj.device().index()));
  const bool dyn = TE_DYN && ntasks >= 16L * grid;
  auto sched = at::zeros({dyn ? 1 : 0}, proj.options().dtype(at::kInt));
  // head groups: the most heads whose bias fits TE_BIAS_L2 run together (all of them: ORD0)
  int hg = (int)H;
  while (hg > 1 && (H % hg != 0 || (double)hg * S * S * 2 > (double)(TE_BIAS_L2))) --hg;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto kernel = hg < H ? triattn_d64_sm90<true> : triattn_d64_sm90<false>;
  kernel<<<grid, THREADS, SMEM, stream>>>((int)N, (int)H, (int)S, (int)ntasks, (float)(scale * 1.4426950408889634), hg, qm, kvm, bm,
#if TE_F8PV
                                          vtm, vscale.data_ptr<float>(),
#endif
                                          reinterpret_cast<const __half*>(proj.data_ptr()), ld,
                                          gate_bias.numel() ? gate_bias.data_ptr<float>() : nullptr, reinterpret_cast<bf16*>(u.data_ptr()), kt,
                                          dyn ? sched.data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return u;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_d64_forward", &triattn_d64_forward, "sm90 triangle-attention inference core for heads of 64 channels (wgmma, fp16 operands "
        "and scores): proj [B S S, 4 H 64] fp16 (q | k | v | g), sbias = bias / scale [B, H, S, S] fp16 -> u = sigmoid(g (+ gate_bias)) o "
        "[B S S, H 64] bf16", py::arg("proj"), py::arg("sbias"), py::arg("gate_bias"), py::arg("H"), py::arg("scale"),
#if TE_F8PV
        py::arg("key_tiles"), py::arg("vt"), py::arg("vscale"));
#else
        py::arg("key_tiles") = py::none());
#endif
  m.attr("KEY_TILE") = BN;
}
