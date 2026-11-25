// Triangle-attention core for H100 (sm_90a), D = 32, bf16 operands, fp32 softmax and accumulation, bf16 out:
//
//   out[b,n,h,q,:] = softmax_k( scale * q[b,n,h,q,:] . k[b,n,h,k,:] + bias[b,h,q,k] ) @ v[b,n,h,k,:]
//
// q / k / v / out are in the projection layout [B, N, S, H, 32] (row (b, n, s) holds every head), the layout the module's front writes
// and its tail reads; the bias is shared by the N pair rows and arrives PRE-DIVIDED by the scale: sbias = bf16(bias / scale)
// [B, H, S, S] (masked keys -inf), which the module's front writes directly.
//
// Forward.  A CTA task = one (b, h, 128-query tile) x 2 pair rows.  The persistent grid walks tasks with the query tile fastest, then
// the head, so the CTAs running at once read the same pair rows' K / V (whole 256-byte rows, every head) through L2.  384 threads:
// warpgroup 0 = TMA producer (one thread), warpgroups 1 / 2 = consumers, queries 0-63 / 64-127 of the tile, both rows of the task.  Per
// 128-key tile j the producer stages the bias tile once (shared by the rows) and K | V of each row.  A consumer's sub-step (tile j,
// row r) has two phases:
//   G: the warpgroup's 64 x 128 bias fragment (ldmatrix) initialises a score accumulator (the H100 technique of Anthropic's opt_core
//      `triattn_m1`: no per-logit bias add), and S += Q_r K_rj^T is issued (RS wgmma: Q_r lives in registers for the whole task);
//   X: p = 2^(S * scale * log2e - m_r) with the row's running maximum as the offset m_r, raised lazily (only when a tile's maximum
//      exceeds it by more than TA_TAU = 8 log2 units; o_r and l_r are then rescaled, which is safe because they are idle during the X phase
//      of their own row), so p <= 2^8 for any spread of the logits; P rounded to bf16 in registers = the A fragment of
//      [O_r | l_r] += P [V_rj | 1] (one m64n40 RS wgmma per 16 keys: the row sums come from the tensor core, from the same bf16 P),
//      left in flight.  (An earlier version fixed m_r at the maximum of the first 32 keys and overflowed to inf / NaN when a later
//      logit exceeded it by ~126 log2 units, which trained weights do.)
// The sub-steps are software-pipelined in each warpgroup (two score accumulators): the G phase of sub-step y + 1 is issued before
// the X phase of y, so the QK^T GEMM and its latency hide under the exponentials; the MUFU (16 ex2 / clk / SM) is the bound.
// With TA_PP the two consumer warpgroups also take turns on the G phase (FlashAttention-3's scheduler barrier).
// Shared per K | V stage: [K | V | ones] (the ones block is the 33rd-40th columns of the PV B operand, written once).
#include <torch/extension.h>
#include "sm90.cuh"
using namespace sm90;

#ifndef TA_FB
#define TA_FB 0          // forward (measured slower, 2026-10-22: more traffic, lower clock under the 700 W cap): the bias staged once per call as fp32 in the score accumulator's register order (stage_bias): a G phase initialises
                         // its accumulator with 16 LDS.128, no bf16 -> fp32 unpack per logit (~1 instruction per exponential of the issue-bound loop)
#endif
#ifndef TA_NKV
#define TA_NKV (TA_FB ? 3 : 4)   // forward: K | V ring stages (one per (key tile, row)); 3 with the fp32 bias tiles (shared memory)
#endif
#ifndef TA_DOFF
#define TA_DOFF 1        // forward: wgmma descriptors built once per CTA, moved per use by one add (desc_off), not re-assembled per wgmma
#endif
#ifndef TA_NB
#define TA_NB 2          // forward: bias ring stages (one per key tile)
#endif
#ifndef TA_PP
#define TA_PP 0          // forward: consumer ping-pong on the G phase
#endif
#ifndef TA_ORDER
#define TA_ORDER 2       // forward: task order, fastest first: 0 = query tile, head, row group, b; 1 = query tile, row group, head, b (the CTAs running together work on ONE
                         // head: its bias [S, S] stays L2-resident across the row groups; 0 keeps the four heads of the same K / V rows together); 2 = 1 from S >= TA_ORDER1_MINS
#endif
#ifndef TA_ORDER1_MINS
#define TA_ORDER1_MINS 2048    // the bias of all heads (4 S^2 x 2 bytes) no longer fits the L2 beside the K / V rows
#endif
#ifndef TA_SPEC
#define TA_SPEC 1        // forward: 1 = speculative offset: after a row's first finite tile no maximum is taken before the exponentials; the tile's row-sum increment
                         // (from the tensor core) tells afterwards whether the offset must rise, and a tile no raise can repair (a logit more than ~110 log2 units above
                         // the offset: non-finite sum) goes on a fix list that a second, exact launch (SAFE) recomputes.  0 = the row maximum of every tile (the SAFE form)
#endif
#ifndef TA_TS
#define TA_TS 40.f       // TA_SPEC 1: a tile whose row-sum increment exceeds 2^TA_TS raises the row's offset (o, l scaled by the power of two)
#endif
#ifndef TA_SPEC_MINS
#define TA_SPEC_MINS 512 // TA_SPEC 1 from this sequence length on (below it the second launch costs about what the maximum pass saves)
#endif
#ifndef TA_NOLIST
#define TA_NOLIST 0      // experiments: no fix list and no second launch (wrong results for an overflowing tile)
#endif
#ifndef TA_TOVF
#define TA_TOVF 110.f    // TA_SPEC 1: an increment above 2^TA_TOVF (or non-finite) is not trusted: o may have overflowed -> fix list
#endif
#ifndef TA_DYN
#define TA_DYN 1         // forward: dynamic task schedule (a global counter; the static round robin lets long launches drift apart)
#endif
namespace {
constexpr int D = 32, BM = 128, BN = 128, R = 2, NKV = TA_NKV, NB = TA_NB;
constexpr int QT = BM * D;                                 // Q tile [128 q][32], 64B swizzle (8 KiB)
constexpr int KVT = BN * D;                                // K or V tile [128 k][32], 64B swizzle (8 KiB)
constexpr int STG = 3 * KVT;                               // a K | V stage: K, V, ones (24 KiB)
constexpr int BT = BM * BN;                                // bias tile: bf16 [2 key halves][128 q][64], 128B swizzle (32 KiB), or fp32 in register order (TA_FB, 64 KiB)
constexpr int BTB = BT * (TA_FB ? 4 : 2);                 // its bytes
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + NB * BTB + (R * QT + NKV * STG) * 2 + 256;
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace

#ifndef TB_WGSYNC
#define TB_WGSYNC 0       // experiments: a warpgroup barrier at the start of every sub-step
#endif
#ifndef TA_TAU
#define TA_TAU 8.f        // forward: a tile raises a row's offset only when its maximum exceeds it by more than this many log2 units
#endif
#ifndef TA_POLY
#define TA_POLY 0         // forward: exponentials (of every 16) computed on the FMA pipe instead of the MUFU
#endif
#ifndef TA_TRACE
#define TA_TRACE 0        // experiments: phase timestamps of CTA 0, warp 0 of each consumer warpgroup
#endif
#if TA_TRACE
__device__ unsigned g_trace[2][256][8];
#define TRACE(k) do { if (blockIdx.x == 0 && wi == 0 && lane == 0 && xs < 256) g_trace[w][xs][k] = (unsigned)clock(); } while (0)
#else
#define TRACE(k) do {} while (0)
#endif
template <bool SPEC, bool ORD1>      // ORD1: the task order of TA_ORDER (query tile, row group, head); SPEC: the speculative offset (fix list written to FIX); else the exact form (row maximum of every tile), which with FIX set runs the tasks of that list (static schedule)
__global__ void __launch_bounds__(THREADS, 1) triattn_fwd_sm90(
    int N, int H, int S, int ntasks, float scl,          // scl = scale * log2(e)
    const __grid_constant__ CUtensorMap qmap,            // q [B*N*S][H*32] bf16 (rows may sit at a wider stride), box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap bmap,            // sbias [B*H*S][S] bf16, box (64, 128), 128B swizzle
    const float* __restrict__ FB,                        // TA_FB: the staged bias [B H][S / 128][S / 128][2][16][128][4] fp32 (stage_bias), else unused
    __nv_bfloat16* __restrict__ OUT,                     // [B*N*S][H*32] contiguous, or nullptr (only U is wanted)
    float* __restrict__ LSE,                             // [B, N, H, S] base-2 log-sum-exp for the backward, or nullptr
    int* __restrict__ FLAGS,
    const __nv_bfloat16* __restrict__ GATE,              // optional gate g (rows at stride ldg, columns h * 32 + c): U = sigmoid(g) o bf16(O), or nullptr
    __nv_bfloat16* __restrict__ U, long ldg,             // [B*N*S][H*32] contiguous
    int* __restrict__ SCHED,                             // zeroed task counter (dynamic schedule, see TA_DYN), or nullptr: static
    int* __restrict__ FIX) {                             // fix list: FIX[0] = count (zeroed), FIX[1..] = task ids (written by the speculative form, read by SAFE)
  const bool LISTM = !SPEC && FIX != nullptr;
  const int nt = LISTM ? *reinterpret_cast<volatile int*>(FIX) : ntasks;
  if (LISTM && blockIdx.x == 0 && threadIdx.x == 0 && nt > 0) atomicAdd(FLAGS + 1, nt);     // telemetry: tasks the exact launch recomputed
  const int QTILES = S / BM, NK = S / BN, NG = (N + R - 1) / R;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  // shared layout: bias ring [NB][BT] | Q [R][QT] | K | V ring [NKV][K | V | ones] | barriers
  const uint32_t sB = sa(smb), sQ = sB + NB * BTB, sKV = sQ + R * QT * 2, bar0 = sKV + NKV * STG * 2;
  const uint32_t qf = bar0, qe = bar0 + 8;                  // the task's Q tiles landed / count 8: every consumer warp holds its Q fragments
  auto kvf = [&](int s) { return bar0 + 16 + 8 * s; };     // [NKV]
  auto kve = [&](int s) { return bar0 + 16 + 8 * (NKV + s); };                // count 8: the stage's MMAs retired in every consumer warp
  auto bfu = [&](int s) { return bar0 + 16 + 8 * (2 * NKV + s); };            // [NB]
  auto bem = [&](int s) { return bar0 + 16 + 8 * (2 * NKV + NB + s); };       // count 8: every consumer warp read its bias fragments
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  for (int i = tid; i < NKV * KVT / 2; i += THREADS) {     // the ones blocks: bf16 pairs (1, 1)
    const int s = i / (KVT / 2), e = i % (KVT / 2);
    reinterpret_cast<uint32_t*>(smb + (sKV - sB) + (s * STG + 2 * KVT) * 2)[e] = 0x3F803F80u;
  }
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sB));
    bar_init(bars, 1); bar_init(bars + 1, 8);
    for (int k = 0; k < NKV; ++k) { bar_init(bars + 2 + k, 1); bar_init(bars + 2 + NKV + k, 8); }
    for (int k = 0; k < NB; ++k) { bar_init(bars + 2 + 2 * NKV + k, 1); bar_init(bars + 2 + 2 * NKV + NB + k, 8); }
    bar_init_fence();
    reinterpret_cast<int*>(smb + (bar0 - sB) + 244)[0] = 0; reinterpret_cast<int*>(smb + (bar0 - sB) + 244)[1] = 0;     // the fix-list counters
  }
  fence_proxy_async();                                      // the ones blocks are read by the tensor core (async proxy)
  __syncthreads();
  // task t: q tile fastest, then the head (the CTAs running together share whole K / V rows), then the row group, then b
  auto decode = [&](int t, int& b, int& h, int& qt, int& g) {
    qt = t % QTILES; int r = t / QTILES;
    if (ORD1) { g = r % NG; r /= NG; h = r % H; b = r / H; }
    else        { h = r % H; r /= H; g = r % NG; b = r / NG; }
  };
  auto row0 = [&](int b, int n) -> int { n = min(n, N - 1); return (b * N + n) * S; };   // first (b, n) row
  // dynamic schedule: the producer takes each next task from the global counter (in launch order, so the CTAs running together stay
  // on neighbouring tasks however long the launch) and hands it to the consumers in this slot (written before the Q tiles' arrive,
  // read after its completion; -1 ends the CTA)
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
          expect_tx(bfu(bs), BTB);
#if TA_FB
          {                                                // one bulk copy: the staged tile is contiguous
            const float* src = FB + ((size_t)((b * H + h) * QTILES + qt) * NK + j) * BT;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(sB + bs * BTB), "l"(src), "r"((uint32_t)BTB), "r"(bfu(bs)) : "memory");
          }
#else
          for (int hh = 0; hh < 2; ++hh) load_2d(&bmap, sB + (bs * BT + hh * BM * 64) * 2, bfu(bs), j * BN + hh * 64, (b * H + h) * S + qt * BM);
#endif
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
#if TA_DOFF
  const uint64_t kd0 = desc_k64(sKV), vd0 = desc_mn64(sKV + KVT * 2, KVT * 2);   // built once: a wgmma operand costs one 32-bit add (desc_off)
#endif
#if TA_PP
  const int mytasks = (int)blockIdx.x < nt ? (nt - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  int left = mytasks * NK * R;                      // G phases of this CTA still to run
  if (w == 1 && left > 0) named_arrive(1, 256);    // warpgroup 0 takes the first G phase
#endif
  // cursors: the K | V stage and bias stage the next G phase reads; the stage whose PV was issued last (released once it retired)
  int gst = 0, gph = 0, bs = 0, bph = 0, xst = 0, bad = 0, xs = 0;
  uint32_t qa[R][2][4];
  float o[R][20], m[R][2];
  float lprev[R][2];                                   // the row sums before the tile in flight (the increment test of the speculative offset)
  int ovf = 0;                                      // this task: a tile the speculative offset could not absorb
  float s[64];                                      // the score accumulator, then the sub-step's probabilities (fp32)
  uint32_t p[32];                                   // the previous sub-step's P (bf16 pairs): the A fragment of its PV GEMM
  // G phase of (tile j, row r): bias fragment -> s, S += Q_r K^T issued and committed
  auto G = [&](int j, int r) {
    TRACE(0);
    if (r == 0) wait(bfu(bs), bph);
#if TA_FB
    {                                              // the staged tile holds this thread's 64 accumulator values as 16 float4, warpgroup-contiguous
      const uint32_t base = sB + bs * BTB + ((w * 16) * 128 + T) * 16;
#pragma unroll
      for (int e4 = 0; e4 < 16; ++e4) {
        const float4 v = lds128(base + e4 * 128 * 16);
        s[4 * e4] = v.x; s[4 * e4 + 1] = v.y; s[4 * e4 + 2] = v.z; s[4 * e4 + 3] = v.w;
      }
    }
#else
    const uint32_t btile = blane + bs * BT * 2;
#pragma unroll
    for (int i2 = 0; i2 < 8; ++i2) {
      uint32_t r4[4];
      ldsm_x4(btile + (i2 >> 2) * BM * 128 + ((((2 * i2 + lcol8) & 7) ^ x7) << 4), r4);
      // r4 = (row g, block 2 i2), (row g + 8, block 2 i2), (row g, block 2 i2 + 1), (row g + 8, block 2 i2 + 1)
      const float2 a0 = bf2f(r4[0]), a1 = bf2f(r4[1]), a2 = bf2f(r4[2]), a3 = bf2f(r4[3]);
      s[8 * i2 + 0] = a0.x; s[8 * i2 + 1] = a0.y; s[8 * i2 + 2] = a1.x; s[8 * i2 + 3] = a1.y;
      s[8 * i2 + 4] = a2.x; s[8 * i2 + 5] = a2.y; s[8 * i2 + 6] = a3.x; s[8 * i2 + 7] = a3.y;
    }
#endif
    if (r == R - 1) {
      fence_regs(s);                             // the ldmatrix data is in registers before the stage is released (an arrive
      fence_proxy_async();
      __syncwarp();                              // does not wait for this thread's loads: the next TMA would race them)
      if (lane == 0) arrive(bem(bs));
      if (++bs == NB) { bs = 0; bph ^= 1; }
    }
    TRACE(1);
    wait(kvf(gst), gph);
    TRACE(2);
    wg_fence();
#if TA_DOFF
    mma_rs_n128<0>(s, qa[r][0], desc_off(kd0, gst * STG * 2), 1u);
    mma_rs_n128<0>(s, qa[r][1], desc_off(kd0, gst * STG * 2 + 32), 1u);
#else
    const uint32_t kaddr = sKV + gst * STG * 2;
    mma_rs_n128<0>(s, qa[r][0], desc_k64(kaddr), 1u);
    mma_rs_n128<0>(s, qa[r][1], desc_k64(kaddr + 32), 1u);
#endif
    wg_commit();
    if (++gst == NKV) { gst = 0; gph ^= 1; }
    TRACE(3);
  };
  // PV of the previous sub-step (row r, stage xst): [O_r | l_r] += P [V | 1], issued and committed
  auto PV = [&](int r) {
    const uint32_t vaddr = sKV + (xst * STG + KVT) * 2;
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const uint32_t a[4] = {p[4 * kk], p[4 * kk + 1], p[4 * kk + 2], p[4 * kk + 3]};
#if TA_DOFF
      mma_rs_n40<1>(o[r], a, desc_off(vd0, xst * STG * 2 + kk * 1024), 1u);
#else
      mma_rs_n40<1>(o[r], a, desc_mn64(vaddr + kk * 1024, KVT * 2), 1u);
#endif
    }
    wg_commit();
  };
  // after a wait that retired the PV of stage xst: release it
  auto release = [&]() {
    __syncwarp();
    if (lane == 0) arrive(kve(xst));
    if (++xst == NKV) xst = 0;
  };
  // X phase of (tile j, row r) on s (its QK^T retired): the row offset and the exponentials, in place.  The offset is the running
  // maximum of the row, raised only when a tile's maximum exceeds it by more than TA_TAU log2 units (the lazy rescale of
  // FlashAttention-4): every p is then <= 2^TA_TAU, whatever the spread of the logits, and the first tile fixes the offset exactly.  The
  // accumulators o[r] are idle here (the PV in flight belongs to the other pair row), so a rescale is a plain multiply of o[r] and l_r.
  auto X = [&](int j, int r) {
    TRACE(5);
    if (!SPEC || __any_sync(0xffffffffu, !(m[r][0] > -INFINITY) | !(m[r][1] > -INFINITY)))      // speculative form: the tile's maximum only for a row without a finite logit yet
    {
    float m0 = fmaxf(s[0], s[1]), m1 = fmaxf(s[2], s[3]);
#pragma unroll
    for (int i = 1; i < 16; ++i) { m0 = fmaxf(m0, fmaxf(s[4 * i], s[4 * i + 1])); m1 = fmaxf(m1, fmaxf(s[4 * i + 2], s[4 * i + 3])); }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m0 *= scl; m1 *= scl;                         // -inf when the tile's 128 keys are all masked
    const bool u0 = m0 > m[r][0] + TA_TAU, u1 = m1 > m[r][1] + TA_TAU;     // m = -inf until a finite logit was seen
    if (__any_sync(0xffffffffu, u0 | u1)) {
      const float n0 = u0 ? m0 : m[r][0], n1 = u1 ? m1 : m[r][1];
      const float f0 = u0 ? ex2f(m[r][0] - n0) : 1.f, f1 = u1 ? ex2f(m[r][1] - n1) : 1.f;     // 2^(-inf) = 0 scales an empty sum
#pragma unroll
      for (int i = 0; i < 5; ++i) { o[r][4 * i] *= f0; o[r][4 * i + 1] *= f0; o[r][4 * i + 2] *= f1; o[r][4 * i + 3] *= f1; }
      m[r][0] = n0; m[r][1] = n1;
    }
    }
    const float nm0 = m[r][0] > -INFINITY ? -m[r][0] : 0.f, nm1 = m[r][1] > -INFINITY ? -m[r][1] : 0.f;   // all keys masked so far: -inf - -inf would be NaN
#pragma unroll
    for (int i = 0; i < 16; ++i) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {             // TA_POLY of every 16 exponentials on the FMA pipe
        const float xx = fmaf(s[4 * i + e], scl, e < 2 ? nm0 : nm1);
        s[4 * i + e] = ((4 * i + e) % 16) < TA_POLY ? ex2_poly(xx) : ex2f(xx);
      }
    }
    TRACE(6);
  };
  auto pack = [&]() {
#pragma unroll
    for (int i = 0; i < 16; ++i) { p[2 * i] = PACKP(s[4 * i], s[4 * i + 1]); p[2 * i + 1] = PACKP(s[4 * i + 2], s[4 * i + 3]); }
    TRACE(7);
    ++xs;
  };
  // after the PV of a tile of row rr retired: its row-sum increment is at least its largest probability; a tile that exceeded the offset by more than
  // TA_TS log2 units raises it by the power of two that brings the increment back to about 1 (o and l scaled exactly); a non-finite sum (a logit more
  // than ~110 units above the offset) cannot be repaired here: the task goes on the fix list
  auto check = [&](int rr) {
    const float i0 = o[rr][16] - lprev[rr][0], i1 = o[rr][18] - lprev[rr][1];
    const float lim = exp2f(TA_TS), lov = exp2f(TA_TOVF);
    if (__any_sync(0xffffffffu, !(i0 <= lim) | !(i1 <= lim))) {
      const bool b0 = !(i0 <= lim), b1 = !(i1 <= lim);
      const bool f0 = b0 && i0 <= lov, f1 = b1 && i1 <= lov;           // increments that are absorbed by a raise
      ovf |= (b0 && !f0) || (b1 && !f1);
      const float d0 = f0 ? ceilf(__log2f(i0)) : 0.f, d1 = f1 ? ceilf(__log2f(i1)) : 0.f;
      const float g0 = ex2f(-d0), g1 = ex2f(-d1);
#pragma unroll
      for (int i = 0; i < 5; ++i) { o[rr][4 * i] *= g0; o[rr][4 * i + 1] *= g0; o[rr][4 * i + 2] *= g1; o[rr][4 * i + 3] *= g1; }
      m[rr][0] += d0; m[rr][1] += d1;
    }
    lprev[rr][0] = o[rr][16]; lprev[rr][1] = o[rr][18];
  };
  // one pipelined sub-step (tile j, row r): QK^T of this sub-step and PV of the previous one (row 1 - r) are both issued, the
  // exponentials run while the PV executes, and nothing is left in flight at the end (ptxas cannot follow a wgmma across a loop
  // back edge and serializes them all when one is)
  auto step = [&](int j, int r) {
#if TB_WGSYNC
    named_sync(3 + w, 128);
#endif
#if TA_PP
    named_sync(1 + w, 256);
#endif
    G(j, r);
    PV(1 - r);
#if TA_PP
    if (!(w == 1 && left == 1)) named_arrive(2 - w, 256);
    --left;
#endif
    wg_wait<1>();                                // QK(j, r) retired; the PV may still run
    fence_regs(s);
    TRACE(4);
    X(j, r);
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    if (SPEC) check(1 - r);
    release();
    pack();
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
      m[r][0] = m[r][1] = -INFINITY;             // no finite logit seen yet
      lprev[r][0] = lprev[r][1] = 0.f;
      ovf = 0;
    }
    {                                              // sub-step (0, 0): no previous PV
#if TA_PP
      named_sync(1 + w, 256);
#endif
      G(0, 0);
#if TA_PP
      if (!(w == 1 && left == 1)) named_arrive(2 - w, 256);
      --left;
#endif
      wg_wait<0>();
      fence_regs(s);
      X(0, 0);
      pack();
    }
    uint4 gpre[R][2];                              // the gate's values of this lane's output columns, loaded under the last key tile
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
      step(j, 1);
      if (j + 1 < NK) step(j + 1, 0);
    }
    PV(1);                                         // the last sub-step's PV
    wg_wait<0>();
    fence_regs(o[0]); fence_regs(o[1]);
    if (SPEC) check(1);
    release();
    // ---- epilogue: O_r / l_r -> bf16 -> out (a quad transposes its 4 x 4 words so each lane stores 16 contiguous bytes); LSE
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = g * R + r;
      if (n >= N) continue;                      // a padding row: computed on row N - 1, not stored
      const float l0 = o[r][16], l1 = o[r][18];  // accumulator columns 32..39 = P 1: the row sums
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
            const float2 gg = bf2f(gw[i]), oo = bf2f(outw[i]);
            uw[i] = pack2(oo.x / (1.f + __expf(-gg.x)), oo.y / (1.f + __expf(-gg.y)));
          }
          *reinterpret_cast<uint4*>(U + orow * (H * D) + h * D + 8 * t4) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
        }
      }
    }
    if (SPEC && !TA_NOLIST) {                      // a tile the speculative offset could not absorb: the task goes on the fix list (the exact launch recomputes it)
      // every consumer warp adds 1 (+256 when one of its lanes overflowed) to a shared-memory counter per task parity; the warp that brings it to 8 pushes the
      // task once and resets the counter (no warp waits: the others are at most a tile behind, so a counter is not reused before it was reset)
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

// TA_FB: the bias of a call in the score accumulator's register order. Per (b h, query tile, key tile) a contiguous [2][16][128][4] fp32 block:
// consumer warpgroup w, float4 e4, thread T, element sub hold the m64n128 accumulator element s[4 e4 + sub] of thread T of warpgroup w --
// row 64 w + 16 (T / 32) + (T % 32) / 4 + 8 (sub / 2), column 8 e4 + 2 (T % 4) + sub % 2 -- so a thread's 16 LDS.128 of one e4 are 512
// contiguous bytes per warp (no bank conflict). One thread per float4: the writes are contiguous.
__global__ void __launch_bounds__(256) stage_bias(const __nv_bfloat16* __restrict__ SB, float* __restrict__ FB, int S, long n4) {
  const int QTILES = S / BM, NKT = S / BN;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n4; i += (long)gridDim.x * 256) {
    const int T = (int)(i & 127), e4 = (int)((i >> 7) & 15), w = (int)((i >> 11) & 1);
    const long tile = i >> 12;
    const int kt = (int)(tile % NKT), qt = (int)((tile / NKT) % QTILES);
    const long bh = tile / ((long)NKT * QTILES);
    const int r0 = qt * BM + 64 * w + 16 * (T >> 5) + ((T & 31) >> 2), c0 = kt * BN + 8 * e4 + 2 * (T & 3);
    const __nv_bfloat16* src = SB + bh * S * S;
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(src + (long)r0 * S + c0);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(src + (long)(r0 + 8) * S + c0);
    reinterpret_cast<float4*>(FB)[i] = make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
  }
}

namespace {
torch::Tensor& flags_for(const torch::Device& dev) {
  static torch::Tensor flags[16];
  auto& f = flags[dev.index()];
  if (!f.defined()) f = torch::zeros({2}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  return f;
}
}  // namespace

// q, k, v [B, N, S, H, 32] bf16 (the projection layout: row (b, n, s) holds all heads; rows may sit at a common wider stride);
// sbias bf16 [B, H, S, S] contiguous = bias / scale (masked keys -inf); S % 128 == 0.
// -> (out [B, N, S, H, 32] bf16 contiguous, lse fp32 [B, N, H, S] base 2 (empty unless want_lse), flags int32 [2], per-device counters accumulated over
//     every call: [0] rows whose sum left (0, 3e38) -- a row with every key masked --, [1] tasks (128 queries x 2 pair rows of a head) that the
//     speculative softmax offset could not absorb and the exact launch recomputed (TA_SPEC))
std::vector<torch::Tensor> triattn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sbias, double scale, bool want_lse,
                                       c10::optional<torch::Tensor> gate, bool want_out) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 5 && q.size(4) == D, "q [B, N, S, H, 32]");
  const long B = q.size(0), N = q.size(1), S = q.size(2), H = q.size(3);
  TORCH_CHECK(S % BM == 0 && S > 0, "S % 128 == 0");
  const long ld = q.stride(2);
  for (const auto* t : {&q, &k, &v}) {
    TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->sizes() == q.sizes() && t->stride(4) == 1 && t->stride(3) == D &&
                t->stride(2) == ld && t->stride(1) == S * ld && t->stride(0) == N * S * ld && ld % 8 == 0 && ld >= H * D,
                "q / k / v: [B, N, S, H, 32] bf16 rows at one common stride");
  }
  TORCH_CHECK(sbias.is_cuda() && sbias.scalar_type() == torch::kBFloat16 && sbias.is_contiguous() && sbias.numel() == B * H * S * S,
              "sbias: bf16 [B, H, S, S] contiguous (bias / scale)");
  const uint64_t rows = (uint64_t)(B * N * S);
  auto m2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)(H * D), rows}, {(uint64_t)ld}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_64B, what); };
  CUtensorMap qm = m2(q, "q"), km = m2(k, "k"), vm = m2(v, "v");
  CUtensorMap bm = make_map<2>(sbias.data_ptr(), {(uint64_t)S, (uint64_t)(B * H * S)}, {(uint64_t)S}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "sbias");
  auto fb = torch::empty({TA_FB ? (long)B * H * S * S : 0}, q.options().dtype(torch::kFloat32));   // TA_FB: the staged bias (stage_bias)
  const bool has_gate = gate.has_value();
  const bool store_out = want_out || !has_gate;
  TORCH_CHECK(!has_gate || (gate->scalar_type() == torch::kBFloat16 && gate->is_cuda() && gate->sizes() == q.sizes() && gate->stride(4) == 1 &&
              gate->stride(3) == D && gate->stride(2) % 8 == 0 && gate->stride(1) == S * gate->stride(2) && gate->stride(0) == N * S * gate->stride(2)),
              "gate: [B, N, S, H, 32] bf16, rows at one stride");
  auto out = torch::empty({store_out ? B : 0, N, S, H, D}, q.options());
  auto u = torch::empty({has_gate ? B : 0, N, S, H, D}, q.options());
  auto lse = torch::empty({want_lse ? B : 0, N, H, S}, q.options().dtype(torch::kFloat32));
  auto& flags = flags_for(q.device());
  static bool attr = false;
  if (!attr) {
    for (auto k : {triattn_fwd_sm90<false, false>, triattn_fwd_sm90<false, true>, triattn_fwd_sm90<true, false>, triattn_fwd_sm90<true, true>})
      C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    attr = true;
  }
  const long ntasks = B * H * ((N + R - 1) / R) * (S / BM);
  const int grid = (int)std::min<long>(ntasks, num_sms(q.device().index()));
  // the dynamic schedule's counter (a zeroed int per call); a launch of a few tasks per CTA keeps the static one
  const bool dyn = TA_DYN && !TA_PP && ntasks >= 16L * grid;
  auto sched = torch::zeros({dyn ? 1 : 0}, q.options().dtype(torch::kInt32));
  // speculative offset (TA_SPEC): the tasks whose tiles it could not absorb (a logit ~110 log2 units above the row's offset) are listed by the
  // first launch, and a second, exact launch (row maximum of every tile) recomputes them; an empty list costs the second launch's start and exit
  const bool spec = TA_SPEC == 1 && S >= TA_SPEC_MINS;
  auto fix = torch::empty({spec ? 1 + ntasks : 0}, q.options().dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (spec) C10_CUDA_CHECK(cudaMemsetAsync(fix.data_ptr<int>(), 0, sizeof(int), stream));
  if (TA_FB) {
    const long n4 = (long)B * H * S * S / 4;
    stage_bias<<<(int)std::min<long>((n4 + 255) / 256, 132L * 32), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(sbias.data_ptr<at::BFloat16>()), fb.data_ptr<float>(), (int)S, n4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto launch = [&](auto kernel, int* sch, int* fx) {
    kernel<<<grid, THREADS, SMEM, stream>>>(
      (int)N, (int)H, (int)S, (int)ntasks, (float)(scale * 1.4426950408889634), qm, km, vm, bm, TA_FB ? fb.data_ptr<float>() : nullptr,
      store_out ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()) : nullptr, want_lse ? lse.data_ptr<float>() : nullptr, flags.data_ptr<int>(),
      has_gate ? reinterpret_cast<const __nv_bfloat16*>(gate->data_ptr<at::BFloat16>()) : nullptr,
      has_gate ? reinterpret_cast<__nv_bfloat16*>(u.data_ptr<at::BFloat16>()) : nullptr, has_gate ? (long)gate->stride(2) : 0L, sch, fx);
  };
  int* fixp = spec ? fix.data_ptr<int>() : nullptr;
  const bool ord1 = TA_ORDER == 1 || (TA_ORDER == 2 && S >= TA_ORDER1_MINS);
  if (spec) {
    launch(ord1 ? triattn_fwd_sm90<true, true> : triattn_fwd_sm90<true, false>, dyn ? sched.data_ptr<int>() : nullptr, fixp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!TA_NOLIST) launch(ord1 ? triattn_fwd_sm90<false, true> : triattn_fwd_sm90<false, false>, nullptr, fixp);
  } else {
    launch(ord1 ? triattn_fwd_sm90<false, true> : triattn_fwd_sm90<false, false>, dyn ? sched.data_ptr<int>() : nullptr, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, flags, u};
}

#if TA_TRACE
torch::Tensor triattn_trace() {
  auto t = torch::empty({2, 256, 8}, torch::TensorOptions().dtype(torch::kInt32));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr<int>(), g_trace, sizeof(g_trace)));
  return t;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if TA_TRACE
  m.def("triattn_trace", &triattn_trace);
#endif
  m.def("triattn_fwd", &triattn_fwd, "sm90 triangle-attention forward (D = 32, bf16): q/k/v/out [B, N, S, H, 32], sbias = bias / scale "
        "bf16 [B, H, S, S], optional base-2 LSE; with `gate` also u = sigmoid(gate) o out (and `out` only if want_out)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sbias"), py::arg("scale"), py::arg("want_lse"), py::arg("gate") = py::none(),
        py::arg("want_out") = true);
}
