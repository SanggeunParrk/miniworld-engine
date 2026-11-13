// TriangleAttention module kernels around the attention core for H100 (sm_90a), d_pair 128, 4 heads x 32 -- the B200 path's kernel
// boundaries (b200_sources/triattn_mod_sm100.cu) on wgmma / TMA.
//
// front:  y = LN(x) (never stored); q | k | v | g = y . W^T  (four 128 x 128 projections, bf16, the [rows][128] projection layout the
//         attention core reads); sbias[b, h, i, j] = (y . Wb[h]) / scale, bf16, masked keys (mask[b, j] false) = -inf -- the core's
//         bias operand, head-major.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include "sm90.cuh"
using namespace sm90;

namespace {
constexpr int C = 128;                                    // d_pair
__device__ __forceinline__ uint32_t pack2h(float a, float b) {       // fp16 pair, a -> low half
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
constexpr int NH = 4;                                     // heads (bias outputs)
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// front.  Tile = 128 rows (one (b, i), 128 consecutive j).  384 threads: warpgroup 0 = TMA producer (the weights once, then the x
// tiles through a two-stage ring); warpgroups 1 / 2 = rows 0-63 / 64-127 of the tile.  A consumer warp ldmatrix-loads its 16 rows of
// x straight into the wgmma A-fragment layout, releases the stage, and normalises in registers (a row's 128 channels sit on the 4
// lanes of a quad: the statistics are two quad sums); the normalised fragments are the A operand of every output chunk (RS wgmma):
// q, k, v, g as four m64n128 chains over the resident [Wq; Wk; Wv; Wg], the bias as one m64n16 chain over Wb (zero-padded to 16
// rows).  The kernel is HBM-bound (it writes four times what it reads): outputs go straight from the accumulators to global memory,
// a quad transposing its words so every lane stores 16 contiguous bytes.
namespace fr {
constexpr int BM = 128, NX = 2;
constexpr int XT = BM * C;                                // x tile [2 halves][128][64], 128B swizzle (32 KiB)
constexpr int WT = 4 * C * C;                             // [Wq; Wk; Wv; Wg] [2 halves][512][64] (128 KiB)
constexpr int WBT = 16 * C;                               // Wb [2 halves][16][64] (4 KiB)
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + (WT + WBT + NX * XT) * 2 + 2 * C * 4 + 256;
static_assert(SMEM <= 232448, "smem");
}  // namespace fr

// NOUT: 4 (q, k, v, g), or 3 (q, k, v: the inference tail tri_tail_sm90<true> projects the gate itself from x)
template <int NOUT>
__global__ void __launch_bounds__(fr::THREADS, 1) tri_front_sm90(
    int L, int ntiles, float eps, float inv_scale, int tr,
    int h16,                                              // q / k / v and the bias in fp16 (the operands of the fp16 core, triattn_h16_sm90.cu); g stays bf16
    const __grid_constant__ CUtensorMap xmap,             // x [R][128] bf16, box (64, 128), 128B swizzle (tr: the 4-D transposed pair map)
    const __grid_constant__ CUtensorMap wmap,             // W4 [512][128] bf16, box (64, 256), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,            // Wb16 [16][128] bf16, box (64, 16), 128B swizzle
    const float* __restrict__ LNW, const float* __restrict__ LNB,
    const bool* __restrict__ MASK,                        // [B][L] key mask or nullptr
    const __grid_constant__ CUtensorMap qmap,             // q / k / v / g [R][128] bf16, box (64, 64), 128B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap gmap,
    void* __restrict__ BIAS) {                            // [B][4][L][L] bf16, or fp16 (h16)
  using namespace fr;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sW = sa(smb), sWb = sW + WT * 2, sX = sWb + WBT * 2, sGB = sX + NX * XT * 2, bar0 = sGB + 2 * C * 4;
  float* gb = reinterpret_cast<float*>(smb + (sGB - sW));                    // gamma [128] | beta [128]
  const uint32_t wf = bar0;
  auto xf = [&](int s) { return bar0 + 8 + 8 * s; };
  auto xe = [&](int s) { return bar0 + 8 + 8 * (NX + s); };                 // count 8: every consumer warp holds its fragments
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  if (tid < 2 * C) gb[tid] = tid < C ? LNW[tid] : LNB[tid - C];
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sW));
    bar_init(bars, 1);
    for (int k = 0; k < NX; ++k) { bar_init(bars + 1 + k, 1); bar_init(bars + 1 + NX + k, 8); }
    bar_init_fence();
  }
  __syncthreads();

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&xmap); prefetch_map(&wmap); prefetch_map(&wbmap);
      expect_tx(wf, (WT + WBT) * 2);
      for (int kh = 0; kh < 2; ++kh) {
        for (int nb = 0; nb < 2; ++nb) load_2d(&wmap, sW + (kh * 4 * C + nb * 256) * 64 * 2, wf, kh * 64, nb * 256);
        load_2d(&wbmap, sWb + kh * 16 * 64 * 2, wf, kh * 64, 0);
      }
      int st = 0, ph = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        wait(xe(st), ph ^ 1);
        expect_tx(xf(st), XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_pair(&xmap, sX + (st * XT + kh * BM * 64) * 2, xf(st), kh * 64, t * BM, L, tr);
        if (++st == NX) { st = 0; ph ^= 1; }
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
  wait(wf, 0);
  int st = 0, ph = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    wait(xf(st), ph);
    const uint32_t xs = sX + st * XT * 2;                      // the x stage; its rows are this tile's output staging afterwards
    // ldmatrix address of 16-byte chunk c16 (of the 256-byte row) for this lane's row: the x tile's [2 halves][128][64] 128B swizzle
    auto chunk = [&](int c16) { return xs + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
    uint32_t xa[8][4];
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) ldsm_x4(chunk(2 * kb + lcol8), xa[kb]);
    // LayerNorm of rows g (words 0, 2 of each block) and g + 8 (words 1, 3), fp32
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const float2 a = bf2f(xa[kb][0]), b = bf2f(xa[kb][2]), c = bf2f(xa[kb][1]), d = bf2f(xa[kb][3]);
      s0 += (a.x + a.y) + (b.x + b.y); s1 += (c.x + c.y) + (d.x + d.y);
    }
    const float mu0 = quad_sum(s0) * (1.f / C), mu1 = quad_sum(s1) * (1.f / C);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const float2 a = bf2f(xa[kb][0]), b = bf2f(xa[kb][2]), c = bf2f(xa[kb][1]), d = bf2f(xa[kb][3]);
      v0 += (a.x - mu0) * (a.x - mu0) + (a.y - mu0) * (a.y - mu0) + (b.x - mu0) * (b.x - mu0) + (b.y - mu0) * (b.y - mu0);
      v1 += (c.x - mu1) * (c.x - mu1) + (c.y - mu1) * (c.y - mu1) + (d.x - mu1) * (d.x - mu1) + (d.y - mu1) * (d.y - mu1);
    }
    const float r0 = rsqrtf(quad_sum(v0) * (1.f / C) + eps), r1 = rsqrtf(quad_sum(v1) * (1.f / C) + eps);
    uint32_t ya[8][4];
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {                        // channels 16 kb + 8 hf + 2 t4 + {0, 1}
        const int k = 16 * kb + 8 * hf + 2 * t4;
        const float2 gm = *reinterpret_cast<const float2*>(gb + k), bt = *reinterpret_cast<const float2*>(gb + C + k);
        const float2 a = bf2f(xa[kb][2 * hf]), c = bf2f(xa[kb][2 * hf + 1]);
        ya[kb][2 * hf] = pack2(fmaf((a.x - mu0) * r0, gm.x, bt.x), fmaf((a.y - mu0) * r0, gm.y, bt.y));
        ya[kb][2 * hf + 1] = pack2(fmaf((c.x - mu1) * r1, gm.x, bt.x), fmaf((c.y - mu1) * r1, gm.y, bt.y));
      }
    }
    const long row = (long)t * BM + 64 * w + 16 * wi + g8;      // this lane's first row; its second is row + 8
    const CUtensorMap* omaps[4] = {&qmap, &kmap, &vmap, &gmap};
    float accs[2][64];                                         // chunk c + 1's GEMM runs while chunk c is stored
    auto issue = [&](float (&acc)[64], int c) {
      wg_fence();
#pragma unroll
      for (int kb = 0; kb < 8; ++kb)
        mma_rs_n128<0>(acc, ya[kb], desc_k128(sW + ((kb >> 2) * 4 * C + c * C) * 128 + (kb & 3) * 32), kb ? 1u : 0u);
      wg_commit();
    };
    issue(accs[0], 0);
#pragma unroll
    for (int c = 0; c < NOUT; ++c) {
      float (&acc)[64] = accs[c & 1];
      if (c + 1 < NOUT) { issue(accs[(c + 1) & 1], c + 1); wg_wait<1>(); } else wg_wait<0>();
      fence_regs(acc);
      // stage the warpgroup's 64 rows of chunk c over its rows of the x stage (stmatrix, the TMA store's 128B swizzle), then one
      // thread stores them; the previous chunk's store must have finished reading the staging first
      if (c > 0) {
        if (T == 0) bulk_wait_read<0>();
        named_sync(3 + w, 128);
      }
#pragma unroll
      if (h16 && c < 3) {
#pragma unroll
        for (int i2 = 0; i2 < 8; ++i2)
          stsm_x4(chunk(2 * i2 + lcol8), pack2h(acc[8 * i2], acc[8 * i2 + 1]), pack2h(acc[8 * i2 + 2], acc[8 * i2 + 3]),
                  pack2h(acc[8 * i2 + 4], acc[8 * i2 + 5]), pack2h(acc[8 * i2 + 6], acc[8 * i2 + 7]));
      } else {
#pragma unroll
        for (int i2 = 0; i2 < 8; ++i2)
          stsm_x4(chunk(2 * i2 + lcol8), pack2(acc[8 * i2], acc[8 * i2 + 1]), pack2(acc[8 * i2 + 2], acc[8 * i2 + 3]),
                  pack2(acc[8 * i2 + 4], acc[8 * i2 + 5]), pack2(acc[8 * i2 + 6], acc[8 * i2 + 7]));
      }
      fence_proxy_async();
      named_sync(3 + w, 128);
      if (T == 0) {
        for (int kh = 0; kh < 2; ++kh) store_2d(omaps[c], xs + kh * BM * 128 + 64 * w * 128, kh * 64, t * BM + 64 * w);
        bulk_commit();
      }
    }
    {                                                          // the bias: columns 0..3 of an m64n16 chain over Wb
      float acc[8];
      wg_fence();
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) mma_rs_n16<0>(acc, ya[kb], desc_k128(sWb + (kb >> 2) * 16 * 128 + (kb & 3) * 32), kb ? 1u : 0u);
      wg_commit();
      wg_wait<0>();
      fence_regs(acc);
      if (t4 < 2) {                                            // lanes t4 = 0 / 1 hold heads 0, 1 / 2, 3
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const long r = row + 8 * hf, bi = r / L;
          const int j = (int)(r - bi * L);
          const long b = bi / L, i = bi - b * L;
          const bool keep = MASK == nullptr || MASK[b * L + j];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int h = 2 * t4 + e;
            const float val = keep ? acc[2 * hf + e] * inv_scale : -INFINITY;
            const long at = ((b * NH + h) * L + i) * L + j;
            if (h16) reinterpret_cast<__half*>(BIAS)[at] = __float2half_rn(val > -INFINITY ? fmaxf(fminf(val, 6.0e4f), -6.0e4f) : val);   // finite values clamped to fp16's range
            else reinterpret_cast<__nv_bfloat16*>(BIAS)[at] = __float2bfloat16_rn(val);
          }
        }
      }
    }
    // release the stage once the last store has read it
    if (T == 0) bulk_wait_read<0>();
    named_sync(3 + w, 128);
    if (lane == 0) arrive(xe(st));
    if (++st == NX) { st = 0; ph ^= 1; }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// tail:  out = x + drop o ((sigmoid(g) o o) . Wo^T [+ bo])   (the gate, the out-projection, its bias, the dropout scale and the residual
//        in one pass, one rounding; drop [B][L (the row's last index j)][128] bf16 = the module's broadcast dropout scale, or none;
//        bo [128] fp32 = RoseTTAFold3's out-projection bias, or none: added after the bf16 sum it was a second rounding at x's scale).
// Tile = 128 rows.  Warpgroup 0 = TMA producer (Wo once; g | o | x per tile through a two-stage ring); warpgroups 1 / 2 = rows
// 0-63 / 64-127.  A consumer warp ldmatrix-loads g, o and x in the wgmma A-fragment layout -- which is also the accumulator
// layout of the same rows and columns -- forms u = sigmoid(g) o o in registers as the A operand of the m64n128 chain over Wo, adds
// drop o acc + x in place, and stages the result over its rows of the g tile for one TMA store.
// GLN (inference): the gate is not read but projected here, g = LN(x) . Wg^T in fp32 (x is loaded for the residual anyway): the front
// skips g (tri_front_sm90<3>), which saves a [R][128] bf16 store and its load (2 of the module's ~10 passes over the pair at d_pair 128),
// and g is no longer rounded to bf16 before the sigmoid.  Wg is resident next to Wo; a stage holds o | x.
namespace tl {
constexpr int BM = 128, NX = 2;
constexpr int XT = BM * C;                                // one [2 halves][128][64] 128B-swizzled tile (32 KiB)
constexpr int THREADS = 384;
template <bool GLN> struct Cfg {
  static constexpr int NA = GLN ? 2 : 3;                  // tiles per stage: o | x, or g | o | x
  static constexpr int NW = GLN ? 2 : 1;                  // resident weights: Wo | Wg, or Wo
  static constexpr int SMEM = 1024 + (NW * C * C + NX * NA * XT) * 2 + 2 * C * 4 + 256;
  static_assert(SMEM <= 232448, "smem");
};
}  // namespace tl

__device__ __forceinline__ float sigmoid_t(float x) {   // 0.5 + 0.5 tanh(x / 2): one MUFU op
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}

template <bool GLN>
__global__ void __launch_bounds__(tl::THREADS, 1) tri_tail_sm90(
    // tr: the ending node.  1: the tiles are rows of the transposed pair (x / out through the 4-D maps, g / o as stored); 2: the tiles
    // are rows of the pair as stored (x / out plain) and o is read transposed through its 4-D map -- one strided stream instead of two
    // (the out store among them), for calls without a dropout scale (its rows are indexed in the transposed order)
    int L, int ntiles, const __nv_bfloat16* __restrict__ DROP, int tr, const float* __restrict__ BO, const float* __restrict__ GB,
    const __grid_constant__ CUtensorMap gmap,             // g / o / x [R][128] bf16, box (64, 128), 128B swizzle (GLN: gmap unused)
    const __grid_constant__ CUtensorMap omap,
    const __grid_constant__ CUtensorMap xmap,
    const __grid_constant__ CUtensorMap wmap,             // Wo [128][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,             // out [R][128] bf16, box (64, 64), 128B swizzle (TMA store)
    const __grid_constant__ CUtensorMap wgmap,            // GLN: Wg [128][128] bf16, box (64, 128), 128B swizzle
    const float* __restrict__ LNW, const float* __restrict__ LNB, float eps) {   // GLN: the LayerNorm (fp32 [128] each)
  using namespace tl;
  constexpr int NA = Cfg<GLN>::NA, NW = Cfg<GLN>::NW;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sW = sa(smb), sWg = sW + C * C * 2, sX = sW + NW * C * C * 2, sGB = sX + NX * NA * XT * 2, bar0 = sGB + 2 * C * 4;
  float* gbeta = reinterpret_cast<float*>(smb + (sGB - sW));                 // GLN: gamma [128] | beta [128]
  const uint32_t wf = bar0;
  auto xf = [&](int s) { return bar0 + 8 + 8 * s; };
  auto xe = [&](int s) { return bar0 + 8 + 8 * (NX + s); };                 // count 8: every consumer warp is done with the stage
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  if (GLN && tid < 2 * C) gbeta[tid] = tid < C ? LNW[tid] : LNB[tid - C];
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sW));
    bar_init(bars, 1);
    for (int k = 0; k < NX; ++k) { bar_init(bars + 1 + k, 1); bar_init(bars + 1 + NX + k, 8); }
    bar_init_fence();
  }
  __syncthreads();

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&omap); prefetch_map(&xmap); prefetch_map(&wmap); prefetch_map(&ymap);
      if (GLN) prefetch_map(&wgmap); else prefetch_map(&gmap);
      expect_tx(wf, NW * C * C * 2);
      for (int kh = 0; kh < 2; ++kh) {
        load_2d(&wmap, sW + kh * C * 128, wf, kh * 64, 0);
        if (GLN) load_2d(&wgmap, sWg + kh * C * 128, wf, kh * 64, 0);
      }
      int st = 0, ph = 0;
      for (int bi = blockIdx.x / (L / BM); bi < ntiles / (L / BM); bi += gridDim.x / (L / BM)) {   // fixed j block per CTA
        const int t = bi * (L / BM) + (int)blockIdx.x % (L / BM);
        wait(xe(st), ph ^ 1);
        expect_tx(xf(st), NA * XT * 2);
        const CUtensorMap* maps[3] = {GLN ? &omap : &gmap, GLN ? &xmap : &omap, &xmap};
        for (int a = 0; a < NA; ++a)
          for (int kh = 0; kh < 2; ++kh)
            load_pair(maps[a], sX + ((st * NA + a) * XT + kh * BM * 64) * 2, xf(st), kh * 64, t * BM, L, a == NA - 1 ? tr == 1 : a == NA - 2 ? tr == 2 : 0);
        if (++st == NX) { st = 0; ph ^= 1; }
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
  wait(wf, 0);
  // the dropout scale of this lane's rows, cached: a CTA keeps one j block, and drop depends on (b, j) only
  uint32_t dfr[16][2];
  long dfb = -1;
  auto load_drop = [&](long b, long j) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const __nv_bfloat16* p = DROP + (b * L + j + 8 * hf) * C + 2 * t4;
#pragma unroll
      for (int i = 0; i < 16; ++i) dfr[i][hf] = *reinterpret_cast<const uint32_t*>(p + 8 * i);
    }
    dfb = b;
  };
  int st = 0, ph = 0;
  for (int bi = blockIdx.x / (L / BM); bi < ntiles / (L / BM); bi += gridDim.x / (L / BM)) {   // fixed j block per CTA
    const int t = bi * (L / BM) + (int)blockIdx.x % (L / BM);
    wait(xf(st), ph);
    // stage tiles: [g |] o | x; the output is staged over this warp's rows of the first one
    const uint32_t ss = sX + st * NA * XT * 2, os = GLN ? ss : ss + XT * 2, xs = os + XT * 2;
    auto chunk = [&](uint32_t base, int c16) { return base + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
    // x in the A-fragment layout = the accumulator layout of the same rows and columns (the A-fragment words of block kb are the
    // accumulator column blocks 2 kb (words 0, 1) and 2 kb + 1 (words 2, 3))
    uint32_t xa[8][4];
    uint32_t ua[8][4];
    float acc[64];
    if constexpr (GLN) {
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) ldsm_x4(chunk(xs, 2 * kb + lcol8), xa[kb]);
      // y = LN(x) of rows g (words 0, 2 of each block) and g + 8 (words 1, 3), fp32 statistics: the front's normalisation, operation for
      // operation (the same bf16 y, so the same g up to the GEMM's summation order)
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        const float2 a = bf2f(xa[kb][0]), b = bf2f(xa[kb][2]), c = bf2f(xa[kb][1]), d = bf2f(xa[kb][3]);
        s0 += (a.x + a.y) + (b.x + b.y); s1 += (c.x + c.y) + (d.x + d.y);
      }
      const float mu0 = quad_sum(s0) * (1.f / C), mu1 = quad_sum(s1) * (1.f / C);
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        const float2 a = bf2f(xa[kb][0]), b = bf2f(xa[kb][2]), c = bf2f(xa[kb][1]), d = bf2f(xa[kb][3]);
        v0 += (a.x - mu0) * (a.x - mu0) + (a.y - mu0) * (a.y - mu0) + (b.x - mu0) * (b.x - mu0) + (b.y - mu0) * (b.y - mu0);
        v1 += (c.x - mu1) * (c.x - mu1) + (c.y - mu1) * (c.y - mu1) + (d.x - mu1) * (d.x - mu1) + (d.y - mu1) * (d.y - mu1);
      }
      const float r0 = rsqrtf(quad_sum(v0) * (1.f / C) + eps), r1 = rsqrtf(quad_sum(v1) * (1.f / C) + eps);
      // ya in ua's registers (the gate chain's A operand; ua is formed after the chain retired)
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {                        // channels 16 kb + 8 hf + 2 t4 + {0, 1}
          const int k = 16 * kb + 8 * hf + 2 * t4;
          const float2 gm = *reinterpret_cast<const float2*>(gbeta + k), bt = *reinterpret_cast<const float2*>(gbeta + C + k);
          const float2 a = bf2f(xa[kb][2 * hf]), c = bf2f(xa[kb][2 * hf + 1]);
          ua[kb][2 * hf] = pack2(fmaf((a.x - mu0) * r0, gm.x, bt.x), fmaf((a.y - mu0) * r0, gm.y, bt.y));
          ua[kb][2 * hf + 1] = pack2(fmaf((c.x - mu1) * r1, gm.x, bt.x), fmaf((c.y - mu1) * r1, gm.y, bt.y));
        }
      }
      wg_fence();
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) mma_rs_n128<0>(acc, ua[kb], desc_k128(sWg + (kb >> 2) * C * 128 + (kb & 3) * 32), kb ? 1u : 0u);
      wg_commit();
      wg_wait<0>();
      fence_regs(acc);
      // u = sigmoid(g [+ gb]) o o, word e of block kb = accumulator columns 16 kb + 8 (e >> 1) + 2 t4 + {0, 1} of row g + 8 (e & 1)
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        uint32_t oo[4];
        ldsm_x4(chunk(os, 2 * kb + lcol8), oo);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int ai = 8 * kb + 4 * (e >> 1) + 2 * (e & 1);
          float2 a = make_float2(acc[ai], acc[ai + 1]);
          const float2 b = bf2f(oo[e]);
          if (GB != nullptr) {
            const float2 gv = __ldg(reinterpret_cast<const float2*>(GB + 16 * kb + 8 * (e >> 1) + 2 * t4));
            a.x += gv.x;
            a.y += gv.y;
          }
          ua[kb][e] = pack2(sigmoid_t(a.x) * b.x, sigmoid_t(a.y) * b.y);
        }
      }
    } else {
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        uint32_t gg[4], oo[4];
        ldsm_x4(chunk(ss, 2 * kb + lcol8), gg);
        ldsm_x4(chunk(os, 2 * kb + lcol8), oo);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 a = bf2f(gg[e]);
          const float2 b = bf2f(oo[e]);
          if (GB != nullptr) {                              // word e: columns 16 kb + 8 (e >> 1) + 2 t4, + 1
            const float2 gv = __ldg(reinterpret_cast<const float2*>(GB + 16 * kb + 8 * (e >> 1) + 2 * t4));
            a.x += gv.x;
            a.y += gv.y;
          }
          ua[kb][e] = pack2(sigmoid_t(a.x) * b.x, sigmoid_t(a.y) * b.y);
        }
      }
    }
    wg_fence();
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) mma_rs_n128<0>(acc, ua[kb], desc_k128(sW + (kb >> 2) * C * 128 + (kb & 3) * 32), kb ? 1u : 0u);
    wg_commit();
    // the residual and the dropout scale meanwhile
    if constexpr (!GLN) {
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) ldsm_x4(chunk(xs, 2 * kb + lcol8), xa[kb]);
    }
    const long row = (long)t * BM + 64 * w + 16 * wi + g8;
    if (DROP != nullptr && bi / L != dfb) load_drop(bi / L, row - (long)bi * L);
    wg_wait<0>();
    fence_regs(acc);
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const float2 xv = bf2f(xa[i >> 1][2 * (i & 1) + hf]);
        float2 dv = make_float2(1.f, 1.f);
        if (DROP != nullptr) dv = bf2f(dfr[i][hf]);
        if (BO != nullptr) {                              // columns 8 i + 2 t4, + 1 of the accumulator block
          const float2 bv = __ldg(reinterpret_cast<const float2*>(BO + 8 * i + 2 * t4));
          acc[4 * i + 2 * hf] += bv.x;
          acc[4 * i + 2 * hf + 1] += bv.y;
        }
        acc[4 * i + 2 * hf] = fmaf(acc[4 * i + 2 * hf], dv.x, xv.x);
        acc[4 * i + 2 * hf + 1] = fmaf(acc[4 * i + 2 * hf + 1], dv.y, xv.y);
      }
    // stage over this warp's rows of the stage's first tile (its own rows: no other warp reads them) and store
#pragma unroll
    for (int i2 = 0; i2 < 8; ++i2)
      stsm_x4(chunk(ss, 2 * i2 + lcol8), pack2(acc[8 * i2], acc[8 * i2 + 1]), pack2(acc[8 * i2 + 2], acc[8 * i2 + 3]),
              pack2(acc[8 * i2 + 4], acc[8 * i2 + 5]), pack2(acc[8 * i2 + 6], acc[8 * i2 + 7]));
    fence_proxy_async();
    named_sync(3 + w, 128);
    if (T == 0) {
      for (int kh = 0; kh < 2; ++kh) store_pair(&ymap, ss + kh * BM * 128 + 64 * w * 128, kh * 64, t * BM + 64 * w, L, tr == 1);
      bulk_commit();
      bulk_wait_read<0>();
    }
    named_sync(3 + w, 128);
    if (lane == 0) arrive(xe(st));
    if (++st == NX) { st = 0; ph ^= 1; }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// gate backward:  dyd = dy o drop;  du = dyd . Wo;  s = sigmoid(g);  do = du o s;  dg = du o o o s (1 - s);
//                 delta[b, i, h, j] = sum_{c in h} do . o (the core backward's row term, from the bf16 do it reads);
//                 dWo += dyd^T . (s o o)   (accumulated in registers over the CTA's tiles, added into the fp32 dWo once at the end);
//                 seed = dy (the residual's share of dpair; head_bwd adds dx into it).
// Tile = 128 rows.  Warpgroup 0 = TMA producer (Wo once; dy | g | o per tile, two stages); warpgroups 1 / 2 own rows 0-63 / 64-127 for
// the elementwise work and out-columns 0-63 / 64-127 of dWo.  The stage is reused as the staging of every output: dyd over dy (the
// dWo GEMM's A, MN-major), u = s o o over o (its B), do over g, then dg over dyd once the dWo GEMM has read it.
namespace gb {
constexpr int BM = 128, NX = 2;
constexpr int XT = BM * C;                                // a [2 halves][128][64] 128B-swizzled tile (32 KiB)
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + (C * C + NX * 3 * XT) * 2 + 256;
static_assert(SMEM <= 232448, "smem");
}  // namespace gb

__global__ void __launch_bounds__(gb::THREADS, 1) tri_gate_bwd_sm90(
    int L, int ntiles, const __nv_bfloat16* __restrict__ DROP, int tr,
    const __grid_constant__ CUtensorMap dymap,            // dy / g / o [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap gmap,
    const __grid_constant__ CUtensorMap omap,
    const __grid_constant__ CUtensorMap wmap,             // Wo [128][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap domap,            // do / dg / seed [R][128] bf16, box (64, 64), 128B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dgmap,
    const __grid_constant__ CUtensorMap seedmap,
    float* __restrict__ DELTA,                            // [B, L, 4, L]
    float* __restrict__ DWO) {                            // [grid][128][128] fp32 partials, one slot per CTA
  using namespace gb;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sW = sa(smb), sX = sW + C * C * 2, bar0 = sX + NX * 3 * XT * 2;
  const uint32_t wf = bar0;
  auto xf = [&](int s) { return bar0 + 8 + 8 * s; };
  auto xe = [&](int s) { return bar0 + 8 + 8 * (NX + s); };                 // count 8
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sW));
    bar_init(bars, 1);
    for (int k = 0; k < NX; ++k) { bar_init(bars + 1 + k, 1); bar_init(bars + 1 + NX + k, 8); }
    bar_init_fence();
  }
  __syncthreads();

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&dymap); prefetch_map(&gmap); prefetch_map(&omap); prefetch_map(&wmap);
      expect_tx(wf, C * C * 2);
      for (int kh = 0; kh < 2; ++kh) load_2d(&wmap, sW + kh * C * 128, wf, kh * 64, 0);
      int st = 0, ph = 0;
      for (int bi = blockIdx.x / (L / BM); bi < ntiles / (L / BM); bi += gridDim.x / (L / BM)) {   // fixed j block per CTA
    const int t = bi * (L / BM) + (int)blockIdx.x % (L / BM);
        wait(xe(st), ph ^ 1);
        expect_tx(xf(st), 3 * XT * 2);
        const CUtensorMap* maps[3] = {&dymap, &gmap, &omap};
        for (int a = 0; a < 3; ++a)
          for (int kh = 0; kh < 2; ++kh)
            load_pair(maps[a], sX + ((st * 3 + a) * XT + kh * BM * 64) * 2, xf(st), kh * 64, t * BM, L, a == 0 ? tr : 0);
        if (++st == NX) { st = 0; ph ^= 1; }
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
  float dwo[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) dwo[i] = 0.f;
  wait(wf, 0);
  uint32_t dfr[16][2];                                   // the cached dropout scale (see tri_tail_sm90)
  long dfb = -1;
  auto load_drop = [&](long b, long j) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const __nv_bfloat16* p = DROP + (b * L + j + 8 * hf) * C + 2 * t4;
#pragma unroll
      for (int i = 0; i < 16; ++i) dfr[i][hf] = *reinterpret_cast<const uint32_t*>(p + 8 * i);
    }
    dfb = b;
  };
  int st = 0, ph = 0;
  for (int bi = blockIdx.x / (L / BM); bi < ntiles / (L / BM); bi += gridDim.x / (L / BM)) {   // fixed j block per CTA
    const int t = bi * (L / BM) + (int)blockIdx.x % (L / BM);
    wait(xf(st), ph);
    const uint32_t ys = sX + st * 3 * XT * 2, gs = ys + XT * 2, os = gs + XT * 2;
    auto chunk = [&](uint32_t base, int c16) { return base + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
    auto store_rows = [&](const CUtensorMap* map, uint32_t base, int pr) {   // this warpgroup's 64 rows of a staged tile
      for (int kh = 0; kh < 2; ++kh) store_pair(map, base + kh * BM * 128 + 64 * w * 128, kh * 64, t * BM + 64 * w, L, pr);
      bulk_commit();
    };
    if (T == 0) store_rows(&seedmap, ys, tr);                 // the residual's share of dpair, straight from the stage
    const long row = (long)t * BM + 64 * w + 16 * wi + g8;
    if (DROP != nullptr && bi / L != dfb) load_drop(bi / L, row - (long)bi * L);
    // dyd = dy o drop in the A-fragment layout (word e of block kb: row g + 8 (e & 1), channels 16 kb + 8 (e >> 1) + 2 t4, +1)
    uint32_t ya[8][4];
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      ldsm_x4(chunk(ys, 2 * kb + lcol8), ya[kb]);
      if (DROP != nullptr) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 y = bf2f(ya[kb][e]), dd = bf2f(dfr[2 * kb + (e >> 1)][e & 1]);
          ya[kb][e] = pack2(y.x * dd.x, y.y * dd.y);
        }
      }
    }
    // du = dyd . Wo (B = Wo MN-major: [out = k][in = n], the n halves 16 KiB apart)
    float du[64];
    wg_fence();
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) mma_rs_n128<1>(du, ya[kb], desc_mn128(sW + kb * 2048, C * 128), kb ? 1u : 0u);
    wg_commit();
    // stage dyd over dy (the seed store must have read it)
    if (T == 0) bulk_wait_read<0>();
    named_sync(3 + w, 128);
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) stsm_x4(chunk(ys, 2 * kb + lcol8), ya[kb][0], ya[kb][1], ya[kb][2], ya[kb][3]);
    wg_wait<0>();
    fence_regs(du);
    // s, do, dg, u; delta from the bf16 do
    uint32_t dga[8][4];
    float dl[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      uint32_t gg[4], oo[4], da[4], ua[4];
      ldsm_x4(chunk(gs, 2 * kb + lcol8), gg);
      ldsm_x4(chunk(os, 2 * kb + lcol8), oo);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 2 * kb + (e >> 1), hf = e & 1;      // accumulator block and row half of this word
        const float2 gv = bf2f(gg[e]), ov = bf2f(oo[e]);
        const float s0 = sigmoid_t(gv.x), s1 = sigmoid_t(gv.y);
        const float u0 = du[4 * i + 2 * hf], u1 = du[4 * i + 2 * hf + 1];
        da[e] = pack2(u0 * s0, u1 * s1);
        dga[kb][e] = pack2(u0 * ov.x * s0 * (1.f - s0), u1 * ov.y * s1 * (1.f - s1));
        ua[e] = pack2(s0 * ov.x, s1 * ov.y);
        const float2 dr_ = bf2f(da[e]);
        dl[hf][i >> 2] = fmaf(dr_.x, ov.x, fmaf(dr_.y, ov.y, dl[hf][i >> 2]));   // head = block / 4
      }
      stsm_x4(chunk(gs, 2 * kb + lcol8), da[0], da[1], da[2], da[3]);   // do over g (this warp's rows only)
      stsm_x4(chunk(os, 2 * kb + lcol8), ua[0], ua[1], ua[2], ua[3]);   // u over o
    }
#pragma unroll
    for (int hf = 0; hf < 2; ++hf)
#pragma unroll
      for (int h = 0; h < 4; ++h) dl[hf][h] = quad_sum(dl[hf][h]);
    if (t4 == 0) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const long r = row + 8 * hf, bi = r / L, j = r - bi * L;
#pragma unroll
        for (int h = 0; h < 4; ++h) DELTA[(bi * 4 + h) * L + j] = dl[hf][h];
      }
    }
    fence_proxy_async();
    named_sync(5, 256);                                   // both warpgroups' dyd / u are staged: the dWo GEMM reads every row
    if (T == 0) store_rows(&domap, gs, 0);
    // dWo[out 64 w .., in] += dyd^T u  (A = dyd MN-major over the tile's rows, B = u MN-major)
    wg_fence();
#pragma unroll
    for (int kb = 0; kb < 8; ++kb)
      mma_ss_n128<1, 1>(dwo, desc_mn128(ys + w * BM * 128 + kb * 2048, BM * 128), desc_mn128(os + kb * 2048, BM * 128), 1u);
    wg_commit();
    wg_wait<0>();
    fence_regs(dwo);
    named_sync(5, 256);                                   // both dWo GEMMs have read dyd: dg may go over it
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) stsm_x4(chunk(ys, 2 * kb + lcol8), dga[kb][0], dga[kb][1], dga[kb][2], dga[kb][3]);
    fence_proxy_async();
    named_sync(3 + w, 128);
    if (T == 0) { store_rows(&dgmap, ys, 0); bulk_wait_read<0>(); }
    named_sync(3 + w, 128);
    if (lane == 0) arrive(xe(st));
    if (++st == NX) { st = 0; ph ^= 1; }
  }
  // this CTA's dWo share -> its slot of the partials (summed in a fixed order by tri_grad_reduce: deterministic)
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int o = 64 * w + 16 * wi + g8 + 8 * hf, c = 8 * i + 2 * t4;
      *reinterpret_cast<float2*>(DWO + (long)blockIdx.x * C * C + o * C + c) = make_float2(dwo[4 * i + 2 * hf], dwo[4 * i + 2 * hf + 1]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// head backward:  gy = sum_t d_t . W_t + sum_h db_h Wb_h   (the five projections' input gradient: q | k | v | g on the tensor core, the
//                 4-wide bias one on the CUDA cores in fp32);  LN backward from the recomputed statistics;  dpair += dx (a bf16 TMA
//                 reduce-add into the seed gate_bwd wrote).  The weight / LN-parameter gradients are tri_wgrad's.
// Tile = 128 rows.  Warpgroup 0 = TMA producer (W4 once; per tile x, then the d_t as [128 rows][64] half-tiles through a ring);
// warpgroups 1 / 2 = rows 0-63 / 64-127: SS wgmma chains (A = the d half-tile, B = W4 MN-major) into one m64n128 accumulator, then
// the row statistics as quad sums; dx is staged over the x stage (stmatrix) for the reduce-add.
namespace hb {
constexpr int BM = 128, NX = 2, NR = 2;
constexpr int XT = BM * C, HT = BM * 64;                  // x tile (32 KiB), a d half-tile [128][64] (16 KiB), 128B swizzle
constexpr int THREADS = 384;
// no 1 KiB alignment slack: the kernel checks the dynamic window is 1 KiB aligned (it is: no static shared memory)
constexpr int SMEM = (4 * C * C + NR * HT + NX * XT) * 2 + (NH + 1) * C * 4 + 256;
static_assert(SMEM <= 232448, "smem");
}  // namespace hb

__global__ void __launch_bounds__(hb::THREADS, 1) tri_head_bwd_sm90(
    int L, int ntiles, float eps, int tr,
    const __grid_constant__ CUtensorMap d0map,            // dq / dk / dv / dg [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map,
    const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap,             // x [R][128], box (64, 128)
    const __grid_constant__ CUtensorMap wmap,             // W4 [512][128], box (64, 256)
    const __grid_constant__ CUtensorMap pmap,             // dpair [R][128] bf16, box (64, 64), 128B swizzle (TMA reduce-add)
    const float* __restrict__ DB,                         // dbias fp32 [B][4][L][L]
    const float* __restrict__ WB, const float* __restrict__ LNW) {   // Wb fp32 [4][128], gamma [128]
  using namespace hb;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();
  const uint32_t sW = sa(raw), sR = sW + 4 * C * C * 2, sX = sR + NR * HT * 2, sP = sX + NX * XT * 2, bar0 = sP + (NH + 1) * C * 4;
  float* prm = reinterpret_cast<float*>(raw + (sP - sW));                    // Wb [4][128] | gamma [128]
  const uint32_t wf = bar0;
  auto xf = [&](int s) { return bar0 + 8 + 8 * s; };
  auto xe = [&](int s) { return bar0 + 8 + 8 * (NX + s); };                 // count 8
  auto rf = [&](int s) { return bar0 + 8 + 8 * (2 * NX + s); };
  auto re = [&](int s) { return bar0 + 8 + 8 * (2 * NX + NR + s); };        // count 2 (one per consumer warpgroup)
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  for (int i = tid; i < (NH + 1) * C; i += THREADS) prm[i] = i < NH * C ? WB[i] : LNW[i - NH * C];
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(raw + (bar0 - sW));
    bar_init(bars, 1);
    for (int k = 0; k < NX; ++k) { bar_init(bars + 1 + k, 1); bar_init(bars + 1 + NX + k, 8); }
    for (int k = 0; k < NR; ++k) { bar_init(bars + 1 + 2 * NX + k, 1); bar_init(bars + 1 + 2 * NX + NR + k, 2); }
    bar_init_fence();
  }
  __syncthreads();

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      const CUtensorMap* dm[4] = {&d0map, &d1map, &d2map, &d3map};
      for (int a = 0; a < 4; ++a) prefetch_map(dm[a]);
      prefetch_map(&xmap); prefetch_map(&wmap);
      expect_tx(wf, 4 * C * C * 2);
      for (int kh = 0; kh < 2; ++kh)
        for (int nb = 0; nb < 2; ++nb) load_2d(&wmap, sW + (kh * 4 * C + nb * 256) * 128, wf, kh * 64, nb * 256);
      int xs = 0, xph = 0, rs = 0, rph = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        wait(xe(xs), xph ^ 1);
        expect_tx(xf(xs), XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_pair(&xmap, sX + (xs * XT + kh * BM * 64) * 2, xf(xs), kh * 64, t * BM, L, tr);
        if (++xs == NX) { xs = 0; xph ^= 1; }
        for (int a = 0; a < 4; ++a)
          for (int kh = 0; kh < 2; ++kh) {
            wait(re(rs), rph ^ 1);
            expect_tx(rf(rs), HT * 2);
            load_2d(dm[a], sR + rs * HT * 2, rf(rs), kh * 64, t * BM);
            if (++rs == NR) { rs = 0; rph ^= 1; }
          }
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
  wait(wf, 0);
  int xs = 0, xph = 0, rs = 0, rph = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    // + the bias projection's share, fp32: db of this lane's two rows (4 heads each)
    const long row = (long)t * BM + 64 * w + 16 * wi + g8;
    float dbv[2][NH];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const long r = row + 8 * hf, bi = r / L, j = r - bi * L, b = bi / L, i = bi - b * L;
#pragma unroll
      for (int h = 0; h < NH; ++h) dbv[hf][h] = DB[((b * NH + h) * L + i) * L + j];
    }
    // gy = sum over the eight (t, half) K blocks: A = d half-tile rows 64 w .. (K-major), B = W4 rows o = 128 t + 64 half .. (MN-major)
    float gy[64];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        wait(rf(rs), rph);
        const uint32_t dt = sR + rs * HT * 2 + 64 * w * 128;
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_ss_n128<0, 1>(gy, desc_k128(dt + kk * 32), desc_mn128(sW + (a * C + kh * 64 + 16 * kk) * 128, 4 * C * 128),
                            (a | kh | kk) ? 1u : 0u);
        wg_commit();
        wg_wait<0>();
        fence_regs(gy);
        if (T == 0) arrive(re(rs));                        // the warpgroup's GEMM on this half-tile retired
        if (++rs == NR) { rs = 0; rph ^= 1; }
      }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = 8 * i + 2 * t4;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const float2 wv = *reinterpret_cast<const float2*>(prm + h * C + c);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          gy[4 * i + 2 * hf] = fmaf(dbv[hf][h], wv.x, gy[4 * i + 2 * hf]);
          gy[4 * i + 2 * hf + 1] = fmaf(dbv[hf][h], wv.y, gy[4 * i + 2 * hf + 1]);
        }
      }
    }
    // LayerNorm backward: x in the accumulator layout (A-fragment words of block kb = accumulator blocks 2 kb (0, 1), 2 kb + 1 (2, 3))
    wait(xf(xs), xph);
    const uint32_t xst = sX + xs * XT * 2;
    auto chunk = [&](int c16) { return xst + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
    uint32_t xa[8][4];
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) ldsm_x4(chunk(2 * kb + lcol8), xa[kb]);
    float sx[2] = {0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 8; ++kb)
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 v = bf2f(xa[kb][e]); sx[e & 1] += v.x + v.y; }
    const float mu[2] = {quad_sum(sx[0]) * (1.f / C), quad_sum(sx[1]) * (1.f / C)};
    float vx[2] = {0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 8; ++kb)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 v = bf2f(xa[kb][e]);
        vx[e & 1] += (v.x - mu[e & 1]) * (v.x - mu[e & 1]) + (v.y - mu[e & 1]) * (v.y - mu[e & 1]);
      }
    const float rs_[2] = {rsqrtf(quad_sum(vx[0]) * (1.f / C) + eps), rsqrtf(quad_sum(vx[1]) * (1.f / C) + eps)};
    // dxhat = gy o gamma (in gy); the two row means
    float m1[2] = {0.f, 0.f}, m2[2] = {0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 8; ++kb)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 2 * kb + (e >> 1), hf = e & 1, c = 8 * i + 2 * t4;
        const float2 gm = *reinterpret_cast<const float2*>(prm + NH * C + c), v = bf2f(xa[kb][e]);
        const float d0 = gy[4 * i + 2 * hf] * gm.x, d1 = gy[4 * i + 2 * hf + 1] * gm.y;
        gy[4 * i + 2 * hf] = d0; gy[4 * i + 2 * hf + 1] = d1;
        m1[hf] += d0 + d1;
        m2[hf] += d0 * (v.x - mu[hf]) * rs_[hf] + d1 * (v.y - mu[hf]) * rs_[hf];
      }
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) { m1[hf] = quad_sum(m1[hf]) * (1.f / C); m2[hf] = quad_sum(m2[hf]) * (1.f / C); }
    // dx = rstd (dxhat - mean(dxhat) - xhat mean(dxhat xhat)), staged over the x stage (this warp's rows) for the reduce-add
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      uint32_t o4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 2 * kb + (e >> 1), hf = e & 1;
        const float2 v = bf2f(xa[kb][e]);
        const float x0 = (v.x - mu[hf]) * rs_[hf], x1 = (v.y - mu[hf]) * rs_[hf];
        o4[e] = pack2(rs_[hf] * (gy[4 * i + 2 * hf] - m1[hf] - x0 * m2[hf]), rs_[hf] * (gy[4 * i + 2 * hf + 1] - m1[hf] - x1 * m2[hf]));
      }
      stsm_x4(chunk(2 * kb + lcol8), o4[0], o4[1], o4[2], o4[3]);
    }
    fence_proxy_async();
    named_sync(3 + w, 128);
    if (T == 0) {
      for (int kh = 0; kh < 2; ++kh) reduce_add_pair(&pmap, xst + kh * BM * 128 + 64 * w * 128, kh * 64, t * BM + 64 * w, L, tr);
      bulk_commit();
      bulk_wait_read<0>();
    }
    named_sync(3 + w, 128);
    if (lane == 0) arrive(xe(xs));
    if (++xs == NX) { xs = 0; xph ^= 1; }
  }
  if (T == 0) bulk_wait<0>();
}

// ---------------------------------------------------------------------------------------------------------------------------------
// weight / LN-parameter gradients.  With xhat the normalized rows (y = gamma o xhat + beta) and d_t the four projections' output
// gradients, everything reduces to accumulations the tensor core and column sums can do:
//   M_t = xhat^T d_t  [128 c][128 o],  cs_t = 1^T d_t,  B_h = sum_rows db_h xhat,  cdb_h = sum_rows db_h
// and then (tri_param_finish, the B200 path's algebra):  dW_t[o][c] = gamma[c] M_t[c][o] + beta[c] cs_t[o];
//   dgamma = sum_t diag(W_t^T M_t^T) + sum_h Wb_h o B_h;  dbeta = sum_t cs_t W_t + sum_h cdb_h Wb_h;  dWb_h = gamma o B_h + beta cdb_h.
// H100 has no tensor memory: four 128 x 128 fp32 accumulators do not fit one CTA's registers, so a CTA owns a PAIR of the projections
// (blockIdx & 1: q, k or v, g) over a strided share of the row tiles (x is read by both pairs).  Warpgroup 0 = TMA producer (x | d_a
// | d_b per tile, two stages); warpgroups 1 / 2: the rows' statistics and xhat (bf16, over x in place: the GEMMs' A, MN-major), then
// M_a / M_b for channels c = 0-63 / 64-127 (SS wgmma, 64 + 64 registers); meanwhile a thread per (projection, 8-column chunk, 16-row
// group) sums its d columns with 16-byte loads (and, in pair 0, B_h / cdb_h from xhat and db), reduced over the row groups at the end.
namespace wg {
constexpr int BM = 128, NX = 2;
constexpr int SLOT = 2 * C * C + 6 * C + 8;          // a CTA's partials: M_a | M_b | cs_a | cs_b | B [4][128] | cdb per warp [8]
constexpr int XT = BM * C;
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + NX * 3 * XT * 2 + 2 * 8 * C * 2 + 256;
static_assert(SMEM <= 232448, "smem");
}  // namespace wg

__global__ void __launch_bounds__(wg::THREADS, 1) tri_wgrad_sm90(
    int L, int ntiles, float eps, int tr,
    const __grid_constant__ CUtensorMap d0map, const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map, const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap,             // [R][128] bf16, box (64, 128), 128B swizzle
    const float* __restrict__ DB,                         // dbias [B, 4, L, L]
    float* __restrict__ MT) {                             // [grid][SLOT] fp32 partials, one slot per CTA (tri_grad_reduce sums them)
  using namespace wg;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  // stages [NX][x | d_a | d_b] | ones [8][128] | db^T [2 halves][8 heads (4 used)][64] (the B_h GEMM's K-major B) | barriers
  const uint32_t sX = sa(smb), sOnes = sX + NX * 3 * XT * 2, sDBT = sOnes + 8 * C * 2, bar0 = sDBT + 8 * C * 2;
  auto xf = [&](int s) { return bar0 + 8 * s; };
  auto xe = [&](int s) { return bar0 + 8 * (NX + s); };                     // count 8
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int pair = blockIdx.x & 1, splits = gridDim.x >> 1, split = blockIdx.x >> 1;
  for (int i = tid; i < 8 * C; i += THREADS) {             // ones | db^T (rows 0..3 hi, 4..7 lo, rewritten every tile)
    reinterpret_cast<uint16_t*>(smb + (sOnes - sX))[i] = 0x3F80u;
    reinterpret_cast<uint16_t*>(smb + (sDBT - sX))[i] = 0u;
  }
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sX));
    for (int k = 0; k < NX; ++k) { bar_init(bars + k, 1); bar_init(bars + NX + k, 8); }
    bar_init_fence();
  }
  fence_proxy_async();
  __syncthreads();

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      const CUtensorMap* dm[4] = {&d0map, &d1map, &d2map, &d3map};
      const CUtensorMap* maps[3] = {&xmap, dm[2 * pair], dm[2 * pair + 1]};
      for (int a = 0; a < 3; ++a) prefetch_map(maps[a]);
      int st = 0, ph = 0;
      for (int t = split; t < ntiles; t += splits) {
        wait(xe(st), ph ^ 1);
        expect_tx(xf(st), 3 * XT * 2);
        for (int a = 0; a < 3; ++a)
          for (int kh = 0; kh < 2; ++kh)
            load_pair(maps[a], sX + ((st * 3 + a) * XT + kh * BM * 64) * 2, xf(st), kh * 64, t * BM, L, a == 0 ? tr : 0);
        if (++st == NX) { st = 0; ph ^= 1; }
      }
    }
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
  const int T2 = tid - 128, dh = T2 >> 6, dr = (T2 & 63) * 2;   // db^T writer: head, row pair of the tile
  float acc[2][64], cs[2][4], bh[4], cdb = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) { acc[0][i] = 0.f; acc[1][i] = 0.f; }
#pragma unroll
  for (int i = 0; i < 4; ++i) { cs[0][i] = 0.f; cs[1][i] = 0.f; bh[i] = 0.f; }
  int st = 0, ph = 0;
  for (int t = split; t < ntiles; t += splits) {
    if (pair == 0) {                                      // db of the tile's 128 rows, 4 heads -> bf16 db^T (and cdb in fp32)
      const long row0 = (long)t * BM, bi = row0 / L, j0 = row0 - bi * L, b = bi / L, i = bi - b * L;
      const float2 v = *reinterpret_cast<const float2*>(DB + ((b * NH + dh) * L + i) * L + j0 + dr);
      cdb += v.x + v.y;
      const int half = dr >> 6, c = (dr & 63) >> 3;
      // hi in row dh, the bf16 residual in row dh + 4: B_h = hi + lo columns, near-fp32 (a bf16-rounded db would amplify the
      // order noise of its fp32 reduce-adds across rounding boundaries)
      const uint32_t hi = pack2(v.x, v.y);
      const float2 hf = bf2f(hi);
      const uint32_t lo = pack2(v.x - hf.x, v.y - hf.y);
      const uint32_t rb = sDBT + half * 8 * 128 + (dr & 7) * 2;
      asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(rb + dh * 128 + ((c ^ dh) << 4)), "r"(hi) : "memory");
      asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(rb + (dh + 4) * 128 + ((c ^ (dh + 4)) << 4)), "r"(lo) : "memory");
    }
    wait(xf(st), ph);
    const uint32_t xs = sX + st * 3 * XT * 2, das = xs + XT * 2, dbs = das + XT * 2;
    auto chunk = [&](int c16) { return xs + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
    // xhat of this warp's 16 rows (bf16) over x
    {
      uint32_t xa[8][4];
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) ldsm_x4(chunk(2 * kb + lcol8), xa[kb]);
      float sx[2] = {0.f, 0.f}, vx[2] = {0.f, 0.f};
#pragma unroll
      for (int kb = 0; kb < 8; ++kb)
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(xa[kb][e]); sx[e & 1] += v.x + v.y; }
      const float mu[2] = {quad_sum(sx[0]) * (1.f / C), quad_sum(sx[1]) * (1.f / C)};
#pragma unroll
      for (int kb = 0; kb < 8; ++kb)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 v = bf2f(xa[kb][e]);
          vx[e & 1] += (v.x - mu[e & 1]) * (v.x - mu[e & 1]) + (v.y - mu[e & 1]) * (v.y - mu[e & 1]);
        }
      const float rs[2] = {rsqrtf(quad_sum(vx[0]) * (1.f / C) + eps), rsqrtf(quad_sum(vx[1]) * (1.f / C) + eps)};
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        uint32_t o4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 v = bf2f(xa[kb][e]);
          o4[e] = pack2((v.x - mu[e & 1]) * rs[e & 1], (v.y - mu[e & 1]) * rs[e & 1]);
        }
        stsm_x4(chunk(2 * kb + lcol8), o4[0], o4[1], o4[2], o4[3]);
      }
    }
    fence_proxy_async();
    named_sync(5, 256);                                   // every row's xhat (and db^T) is staged
    // M_a, M_b [c 64 w ..][o] += xhat^T d  (A = xhat MN-major over the rows, B = d MN-major);
    // cs_a, cs_b [o 64 w ..] += d^T 1  (A = d MN-major, B = ones);  pair 0: B_h [c 64 w ..] += xhat^T db  (B = db^T K-major)
    wg_fence();
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const uint64_t a = desc_mn128(xs + w * BM * 128 + kb * 2048, BM * 128);
      mma_ss_n128<1, 1>(acc[0], a, desc_mn128(das + kb * 2048, BM * 128), 1u);
      mma_ss_n128<1, 1>(acc[1], a, desc_mn128(dbs + kb * 2048, BM * 128), 1u);
      const uint64_t ones = desc_k128(sOnes + (kb & 3) * 32);
      mma_ss_n8<1, 0>(cs[0], desc_mn128(das + w * BM * 128 + kb * 2048, BM * 128), ones, 1u);
      mma_ss_n8<1, 0>(cs[1], desc_mn128(dbs + w * BM * 128 + kb * 2048, BM * 128), ones, 1u);
      if (pair == 0) mma_ss_n8<1, 0>(bh, a, desc_k128(sDBT + (kb >> 2) * 8 * 128 + (kb & 3) * 32), 1u);
    }
    wg_commit();
    wg_wait<0>();
    fence_regs(acc[0]); fence_regs(acc[1]); fence_regs(cs[0]); fence_regs(cs[1]); fence_regs(bh);
    named_sync(5, 256);                                   // both warpgroups' GEMMs retired: the stage and db^T may be refilled
    if (lane == 0) arrive(xe(st));
    if (++st == NX) { st = 0; ph ^= 1; }
  }
  // M shares -> MT; cs / B_h (every column of an n8 product is the same sum: lanes t4 = 0) and cdb -> VEC
#pragma unroll
  for (int a = 0; a < 2; ++a) {
    float* mt = MT + (long)blockIdx.x * SLOT + a * C * C;
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int c = 64 * w + 16 * wi + g8 + 8 * hf, o = 8 * i + 2 * t4;
        *reinterpret_cast<float2*>(mt + c * C + o) = make_float2(acc[a][4 * i + 2 * hf], acc[a][4 * i + 2 * hf + 1]);
      }
  }
  const int r0 = 64 * w + 16 * wi + g8;
  float* vec = MT + (long)blockIdx.x * SLOT + 2 * C * C;   // cs_a | cs_b | B [4][128] | cdb per warp [8]
  if (t4 == 0) {
#pragma unroll
    for (int a = 0; a < 2; ++a) { vec[a * C + r0] = cs[a][0]; vec[a * C + r0 + 8] = cs[a][2]; }
  }
  if (pair == 0) {                                        // B_h = column h (hi) + column h + 4 (lo): lanes t4 and t4 + 2
#pragma unroll
    for (int e = 0; e < 4; ++e) bh[e] += __shfl_xor_sync(0xffffffffu, bh[e], 2);
  }
  if (pair == 0 && t4 < 2) {                              // B_h: columns h = 2 t4, 2 t4 + 1
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      vec[2 * C + (2 * t4 + e) * C + r0] = bh[e];
      vec[2 * C + (2 * t4 + e) * C + r0 + 8] = bh[2 + e];
    }
  }
  if (pair == 0) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) cdb += __shfl_xor_sync(0xffffffffu, cdb, o);   // a warp's 32 threads share one head
    if (lane == 0) vec[6 * C + (T2 >> 5)] = cdb;          // per warp (two warps a head)
  }
}

// the partial slots of tri_wgrad_sm90 ([2 splits][SLOT], pair = slot & 1) and tri_gate_bwd_sm90 ([ggrid][C*C]) summed in a fixed
// order -> MT [4][128 c][128 o] | VEC (cs [4][128] | B [4][128] | cdb [4]) | dWo [128][128], fp32: deterministic parameter gradients
__global__ void __launch_bounds__(256) tri_grad_reduce(const float* __restrict__ WS, int splits, const float* __restrict__ GS, int ggrid,
                                                     float* __restrict__ MT, float* __restrict__ VEC, float* __restrict__ DWO) {
  // 32 consecutive elements (coalesced) x 8 groups of splits (k, k + 8, ...), summed in a fixed order through shared memory
  using wg::SLOT;
  __shared__ float red[8][32];
  const int x = threadIdx.x & 31, y = threadIdx.x >> 5;
  const long i = (long)blockIdx.x * 32 + x;
  const long NM = 4L * C * C, NV = 8L * C + NH;
  float s = 0.f;
  if (i < NM) {
    const int t = (int)(i / (C * C)); const long e = i - (long)t * C * C;
    for (int k = y; k < splits; k += 8) s += WS[(2L * k + t / 2) * SLOT + (t & 1) * C * C + e];
  } else if (i < NM + 4 * C) {
    const int j = (int)(i - NM), t = j / C, o = j % C;
    for (int k = y; k < splits; k += 8) s += WS[(2L * k + t / 2) * SLOT + 2 * C * C + (t & 1) * C + o];
  } else if (i < NM + 8 * C) {
    const int j = (int)(i - NM - 4 * C);
    for (int k = y; k < splits; k += 8) s += WS[(2L * k) * SLOT + 2 * C * C + 2 * C + j];
  } else if (i < NM + NV) {
    const int h = (int)(i - NM - 8 * C);
    for (int k = y; k < splits; k += 8) s += WS[(2L * k) * SLOT + 2 * C * C + 6 * C + 2 * h] + WS[(2L * k) * SLOT + 2 * C * C + 6 * C + 2 * h + 1];
  } else if (i < NM + NV + C * C) {
    const long e = i - NM - NV;
    for (int g = y; g < ggrid; g += 8) s += GS[(long)g * C * C + e];
  }
  red[y][x] = s;
  __syncthreads();
  if (y != 0) return;
  float tot = 0.f;
#pragma unroll
  for (int q = 0; q < 8; ++q) tot += red[q][x];
  if (i < NM) MT[i] = tot;
  else if (i < NM + 4 * C) VEC[i - NM] = tot;
  else if (i < NM + 8 * C) VEC[4 * C + (i - NM - 4 * C)] = tot;
  else if (i < NM + NV) VEC[8 * C + (i - NM - 8 * C)] = tot;
  else if (i < NM + NV + C * C) DWO[i - NM - NV] = tot;
}

__device__ __forceinline__ void st_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void st_out(float* p, float v) { *p = v; }

// the B200 path's parameter algebra (arch-neutral): OutT = the parameters' dtype (bf16, or fp32 master weights)
template <typename OutT> struct Outs { OutT* dw[4]; OutT* gamma; OutT* beta; OutT* wb; OutT* wo; };
template <typename OutT>
__global__ void __launch_bounds__(256) tri_param_finish(const float* __restrict__ MT, const float* __restrict__ VEC,
    const __nv_bfloat16* __restrict__ W4, const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WBF,
    Outs<OutT> O, const float* __restrict__ DWO) {
  const int tid = threadIdx.x;
  if (blockIdx.x >= 64 + C) {                      // blocks 192..207: gate_bwd's dWo fp32 -> the parameters' dtype
    const int i0 = (blockIdx.x - 64 - C) * 1024;
    for (int i = tid; i < 1024; i += 256) st_out(O.wo + i0 + i, DWO[i0 + i]);
    return;
  }
  const float* cs = VEC; const float* bh = VEC + 4 * C; const float* cdb = VEC + 8 * C;
  if (blockIdx.x < 64) {
    __shared__ float tile[32][33];
    const int tt = blockIdx.x >> 4, c0 = ((blockIdx.x >> 2) & 3) * 32, o0 = (blockIdx.x & 3) * 32;
    for (int i = tid; i < 32 * 32; i += 256) { const int cc = i >> 5, oo = i & 31; tile[cc][oo] = MT[((long)tt * C + c0 + cc) * C + o0 + oo]; }
    __syncthreads();
    for (int i = tid; i < 32 * 32; i += 256) {
      const int oo = i >> 5, cc = i & 31, c = c0 + cc, o = o0 + oo;
      st_out(O.dw[tt] + (long)o * C + c, LNW[c] * tile[cc][oo] + LNB[c] * cs[tt * C + o]);
    }
    return;
  }
  const int c = blockIdx.x - 64;
  __shared__ float red[2][8];
  float dgam = 0.f, dbet = 0.f;
  for (int i = tid; i < 4 * C; i += 256) {        // (t, o)
    const int tt = i >> 7, o = i & (C - 1);
    const float w = __bfloat162float(W4[((long)tt * C + o) * C + c]);
    dgam = fmaf(w, MT[((long)tt * C + c) * C + o], dgam);
    dbet = fmaf(w, cs[tt * C + o], dbet);
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) { dgam += __shfl_xor_sync(0xffffffffu, dgam, o); dbet += __shfl_xor_sync(0xffffffffu, dbet, o); }
  if ((tid & 31) == 0) { red[0][tid >> 5] = dgam; red[1][tid >> 5] = dbet; }
  __syncthreads();
  if (tid == 0) {
    float g = 0.f, b = 0.f;
    for (int w = 0; w < 8; ++w) { g += red[0][w]; b += red[1][w]; }
    for (int h = 0; h < NH; ++h) {
      g = fmaf(WBF[h * C + c], bh[h * C + c], g);
      b = fmaf(WBF[h * C + c], cdb[h], b);
      st_out(O.wb + h * C + c, LNW[c] * bh[h * C + c] + LNB[c] * cdb[h]);
    }
    st_out(O.gamma + c, g);
    st_out(O.beta + c, b);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// wide front (inference):  d_pair CW = 256, 8 heads x 32 -- the front above at a width whose [Wq; Wk; Wv; Wg] (512 KiB) no longer fits
// one CTA.  The CTAs come in groups of four, CTA 4 n + p holding projection p's weights (q, k, v, g: [256 out][CW] bf16, 128 KiB,
// resident) and walking the same 64-row tiles as the other three of its group (so x is read from HBM once and from L2 three times);
// the g CTA also holds Wb ([8][CW]) and writes the head-major bias.  Each of them normalises the rows itself (the LayerNorm is ~5 %
// of the issue slots here; storing y would cost an HBM round trip).  384 threads: warpgroup 0 = TMA producer (the weights once, then
// each consumer's x tiles into that consumer's own stage), warpgroups 1 / 2 = alternate 64-row tiles of the CTA.  A consumer warp
// ldmatrix-loads its 16 rows of x into the wgmma A-fragment layout, releases the stage as soon as the fragments are in registers
// (the producer refills it under the GEMMs), normalises in registers (statistics = quad sums) and runs two m64n128 RS chains (the
// projection's two column halves) back to back; the first half is stored while the second computes.  The packed weight rows are
// permuted (tri_pack_wide) so that a lane's accumulator columns are 8 consecutive output columns per 32-column group: every lane
// stores 16 contiguous bytes straight from registers (a warp's store = 8 rows x 64 contiguous bytes, whole sectors), no staging.
#ifndef FW_MC
#define FW_MC 0       // 1: the four CTAs of a group form a cluster; each loads one K block of the x tile multicast to all four
#endif
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ void arrive_rank(uint32_t bar, uint32_t rank) {    // arrive on the barrier at the same offset in CTA `rank`
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}\n"
               :: "r"(bar), "r"(rank) : "memory");
}
__device__ __forceinline__ void load_pair_mc(const void* map, uint32_t dst, uint32_t bar, int c0, int row, int L, int tr, uint16_t mask) {
  if (!tr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(row), "h"(mask) : "memory");
    return;
  }
  int i, j, b; pair_coord(row, L, i, j, b);
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4, %5, %6}], [%2], %7;\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(j), "r"(i), "r"(b), "h"(mask) : "memory");
}
#ifndef FW_PF
#define FW_PF 0       // the producer prefetches x tile k + FW_PF into L2 (0: off)
#endif
__device__ __forceinline__ void l2_prefetch_pair(const void* map, int c0, int row, int L, int tr) {
  if (!tr) {
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];\n" :: "l"(map), "r"(c0), "r"(row) : "memory");
    return;
  }
  int i, j, b; pair_coord(row, L, i, j, b);
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];\n" :: "l"(map), "r"(c0), "r"(j), "r"(i), "r"(b) : "memory");
}
#ifndef FW_ABL
#define FW_ABL 0      // ablations (wrong output by design): 1 = no stores, 2 = no GEMM, 4 = no LayerNorm, 8 = no x loads
#endif
#ifndef FW_LAG
#define FW_LAG 3      // the four producers of a group stay within this many tiles of each other (a counter per group in global memory):
#endif                // free-running, the group's CTAs drifted apart and x left the L2 between their reads (1.44 GB of DRAM reads for
                      // 0.54 GB of x at L1024, 2026-11-13).  0: free-running
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void red_release_gpu(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
namespace fw {
constexpr int BM = 64;                                    // rows per consumer tile
constexpr int NG = 256;                                   // output columns per CTA (one projection)
constexpr int HB = 8;                                     // bias rows (heads)
constexpr int THREADS = 384;
template <int CW> struct Cfg {
  static constexpr int KB = CW / 64;                      // 64-channel K blocks (one 128B-swizzled TMA box each)
  static constexpr int XT = BM * CW;                      // x stage [KB][64 rows][64]
  static constexpr int WT = NG * CW;                      // W [KB][256 rows][64]
  static constexpr int WBT = HB * CW;                     // Wb [KB][8][64]
  static constexpr int SMEM = 1024 + (WT + WBT + 2 * XT) * 2 + 3 * CW * 4 + 256;
  static_assert(SMEM <= 232448, "smem");
};
struct Outs { void* p[4]; long ldo; };                    // q, k, v, g and their row stride (elements)
}  // namespace fw

template <int CW>
__global__ void __launch_bounds__(fw::THREADS, 1) tri_front_wide_sm90(
    int L, int ntiles, float eps, float inv_scale, int tr, int h16, int nh,
    const __grid_constant__ CUtensorMap xmap,             // x [R][CW] bf16, box (64, 64), 128B swizzle (tr: the 4-D transposed pair map)
    const __grid_constant__ CUtensorMap wmap,             // packed W [4 * 256][CW] bf16 (tri_pack_wide's row order), box (64, 256)
    const __grid_constant__ CUtensorMap wbmap,            // Wb8 [8][CW] bf16, box (64, 8)
    const float* __restrict__ FOLD,                       // [2][4 CW + 8] fp32: S1 | S0 of tri_pack_wide (the folded LayerNorm)
    const bool* __restrict__ MASK,                        // [B][L] key mask or nullptr
    const float* __restrict__ GBIAS,                      // [CW] fp32 gate bias (RoseTTAFold3) or nullptr
    fw::Outs outs,                                        // q / k / v ([R][CW], fp16 when h16, else bf16), g ([R][CW] bf16)
    void* __restrict__ BIAS,                              // [B][nh][L][L] bf16, or fp16 (h16)
    int* __restrict__ SYNC) {                             // [groups] zeroed tile counters (FW_LAG), or nullptr
  using namespace fw;
  using Q = Cfg<CW>;
  constexpr int KB = Q::KB, XT = Q::XT, WT = Q::WT, WBT = Q::WBT;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sW = sa(smb), sWb = sW + WT * 2, sX = sWb + WBT * 2, sGB = sX + 2 * XT * 2, bar0 = sGB + 3 * CW * 4;
  float* gb = reinterpret_cast<float*>(smb + (sGB - sW));                    // this projection's S1 [CW] | S0 [CW] (+ gate bias) | the heads' S1 [8] | S0 [8]
  const uint32_t wf = bar0;
  auto xf = [&](int s) { return bar0 + 8 + 8 * s; };
  auto xe = [&](int s) { return bar0 + 24 + 8 * s; };                       // count 4: every warp of the consumer holds its fragments
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int grp = blockIdx.x & 3, G = gridDim.x >> 2, idx = blockIdx.x >> 2;
  constexpr int NS = 4 * CW + HB;                          // FOLD: S1 [NS] | S0 [NS]; entries = the q | k | v | g output columns, then the heads
  for (int c = tid; c < 2 * CW + 2 * HB; c += THREADS)
    gb[c] = c < CW ? FOLD[grp * CW + c]
          : c < 2 * CW ? FOLD[NS + grp * CW + c - CW] + (grp == 3 && GBIAS != nullptr ? GBIAS[c - CW] : 0.f)
          : c < 2 * CW + HB ? FOLD[4 * CW + c - 2 * CW] : FOLD[NS + 4 * CW + c - 2 * CW - HB];
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sW));
    bar_init(bars, 1);
    for (int k = 0; k < 2; ++k) { bar_init(bars + 1 + k, 1); bar_init(bars + 3 + k, FW_MC ? 16 : 4); }
    bar_init_fence();
  }
  __syncthreads();
#if FW_MC
  cluster_sync_all();                                      // every barrier of the cluster initialised before a remote arrive / multicast
#endif

  if (wg == 0) {
    reg_dealloc<40>();
    if (tid == 0) {
      prefetch_map(&xmap); prefetch_map(&wmap); prefetch_map(&wbmap);
      expect_tx(wf, (WT + (grp == 3 ? WBT : 0)) * 2);
      for (int kb = 0; kb < KB; ++kb) {
        load_2d(&wmap, sW + kb * NG * 64 * 2, wf, kb * 64, grp * NG);
        if (grp == 3) load_2d(&wbmap, sWb + kb * HB * 64 * 2, wf, kb * 64, 0);
      }
      int k = 0;
      for (int t = idx; t < ntiles; t += G, ++k) {
#if FW_ABL & 8
        break;
#endif
        const int s = k & 1;
        wait(xe(s), ((k >> 1) & 1) ^ 1);
        if (SYNC != nullptr && k >= FW_LAG)                // every producer of the group has issued tile k - FW_LAG
          while (ld_acquire_gpu(SYNC + idx) < 4 * (k - FW_LAG + 1)) __nanosleep(64);
        expect_tx(xf(s), XT * 2);
        if (SYNC != nullptr) red_release_gpu(SYNC + idx, 1);
#if FW_MC
        for (int kb = grp; kb < KB; kb += 4) load_pair_mc(&xmap, sX + (s * XT + kb * BM * 64) * 2, xf(s), kb * 64, t * BM, L, tr, 0xF);
#else
        for (int kb = 0; kb < KB; ++kb) load_pair(&xmap, sX + (s * XT + kb * BM * 64) * 2, xf(s), kb * 64, t * BM, L, tr);
#endif
#if FW_PF
        if (grp == 0 && t + FW_PF * G < ntiles)
          for (int kb = 0; kb < KB; ++kb) l2_prefetch_pair(&xmap, kb * 64, (t + FW_PF * G) * BM, L, tr);
#endif
      }
    }
#if FW_MC
    cluster_sync_all();                                    // no CTA leaves while another may still write into it
#endif
    return;
  }

  reg_alloc<232>();
  const int w = wg - 1, T = tid - 128 * wg, wi = T >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int lrow = 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
  const bool half_out = h16 && grp < 3;
  wait(wf, 0);
  int k = w;
  for (int t = idx + w * G; t < ntiles; t += 2 * G, k += 2) {
#if !(FW_ABL & 8)
    wait(xf(w), (k >> 1) & 1);
#endif
    const uint32_t xs = sX + w * XT * 2;
    // ldmatrix address of 16-byte chunk c16 (of the row's CW / 8) for this lane's row: [KB][64 rows][64] 128B swizzle
    auto chunk = [&](int c16) { return xs + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
    uint32_t xa[4 * KB][4];
#pragma unroll
    for (int kb = 0; kb < 4 * KB; ++kb) ldsm_x4(chunk(2 * kb + lcol8), xa[kb]);
    // every fragment is in registers: hand the stage back to the producer
    fence_regs(xa[4 * KB - 1]);
    fence_proxy_async();
    __syncwarp();
#if !(FW_ABL & 8)
#if FW_MC
    if (lane < 4) arrive_rank(xe(w), lane);                // the stage of all four CTAs (each is filled by all four)
#else
    if (lane == 0) arrive(xe(w));
#endif
#endif
    const long row = (long)t * BM + 16 * wi + g8;              // this lane's first row; its second is row + 8
    float accs[2][64];
    float accb[4];
    // commit groups: [bias (g CTA)], half 0, half 1
    if (grp == 3) {
      wg_fence();
#pragma unroll
      for (int kb = 0; kb < 4 * KB; ++kb) mma_rs_n8(accb, xa[kb], desc_k128(sWb + (kb >> 2) * HB * 128 + (kb & 3) * 32), kb ? 1u : 0u);
      wg_commit();
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      wg_fence();
#pragma unroll
      for (int kb = 0; kb < 4 * KB; ++kb)
#if FW_ABL & 2
        for (int j = 0; j < 4; ++j) accs[c][4 * kb + j] = __uint_as_float(xa[kb][j] ^ c);
#else
        mma_rs_n128<0>(accs[c], xa[kb], desc_k128(sW + ((kb >> 2) * NG + c * 128) * 128 + (kb & 3) * 32), kb ? 1u : 0u);
#endif
      wg_commit();
    }
    // the row statistics while the chains run: rows g (words 0, 2 of each block) and g + 8 (words 1, 3), fp32
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int kb = 0; kb < 4 * KB; ++kb) {
      const float2 a = bf2f(xa[kb][0]), b = bf2f(xa[kb][2]), c = bf2f(xa[kb][1]), d = bf2f(xa[kb][3]);
      s0 += (a.x + a.y) + (b.x + b.y); s1 += (c.x + c.y) + (d.x + d.y);
    }
    const float mu0 = quad_sum(s0) * (1.f / CW), mu1 = quad_sum(s1) * (1.f / CW);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int kb = 0; kb < 4 * KB; ++kb) {
      const float2 a = bf2f(xa[kb][0]), b = bf2f(xa[kb][2]), c = bf2f(xa[kb][1]), d = bf2f(xa[kb][3]);
      v0 += (a.x - mu0) * (a.x - mu0) + (a.y - mu0) * (a.y - mu0) + (b.x - mu0) * (b.x - mu0) + (b.y - mu0) * (b.y - mu0);
      v1 += (c.x - mu1) * (c.x - mu1) + (c.y - mu1) * (c.y - mu1) + (d.x - mu1) * (d.x - mu1) + (d.y - mu1) * (d.y - mu1);
    }
#if FW_ABL & 4
    const float rs[2] = {1.f, 1.f}, nm[2] = {0.f * (mu0 + v0), 0.f * (mu1 + v1)};
#else
    // the folded LayerNorm: out = r (x . Wf^T - mu S1) + S0 per row (Wf = W diag(gamma), S1 = Wf 1, S0 = W beta: tri_pack_wide), so x
    // is the GEMM operand as loaded -- no normalised copy of the tile; r = rsqrt(var + eps), nm = -mu r
    const float rs[2] = {rsqrtf(quad_sum(v0) * (1.f / CW) + eps), rsqrtf(quad_sum(v1) * (1.f / CW) + eps)};
    const float nm[2] = {-mu0 * rs[0], -mu1 * rs[1]};
#endif
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      float (&acc)[64] = accs[c];
      if (c == 0) wg_wait<1>(); else wg_wait<0>();
      fence_regs(acc);
      if (c == 0 && grp == 3) {                                // the bias: lane t4 holds heads 2 t4, 2 t4 + 1 of rows g, g + 8
        fence_regs(accb);
        if (2 * t4 < nh) {
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const long r = row + 8 * hf, bi = r / L;
            const int j = (int)(r - bi * L);
            const long b = bi / L, i = bi - b * L;
            const bool keep = MASK == nullptr || MASK[b * L + j];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int h = 2 * t4 + e;
              const float pb = fmaf(rs[hf], accb[2 * hf + e], fmaf(nm[hf], gb[2 * CW + h], gb[2 * CW + HB + h]));
              const float val = keep ? pb * inv_scale : -INFINITY;
              const long at = ((b * nh + h) * L + i) * L + j;
              if (h16) reinterpret_cast<__half*>(BIAS)[at] = __float2half_rn(val > -INFINITY ? fmaxf(fminf(val, 6.0e4f), -6.0e4f) : val);
              else reinterpret_cast<__nv_bfloat16*>(BIAS)[at] = __float2bfloat16_rn(val);
            }
          }
        }
      }
      // accumulator column 8 i + 2 t4 + e holds output column 32 (i / 4) + 8 t4 + 2 (i % 4) + e of this half (tri_pack_wide's order):
      // the lane's words of 32-column group m are 8 consecutive columns
      __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(outs.p[grp]);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int col = c * 128 + 32 * m + 8 * t4;
        float k1[8], k0[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { k1[e] = gb[col + e]; k0[e] = gb[CW + col + e]; }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          uint32_t wv[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float a = fmaf(rs[hf], acc[16 * m + 4 * q + 2 * hf], fmaf(nm[hf], k1[2 * q], k0[2 * q]));
            const float b = fmaf(rs[hf], acc[16 * m + 4 * q + 2 * hf + 1], fmaf(nm[hf], k1[2 * q + 1], k0[2 * q + 1]));
            wv[q] = half_out ? pack2h(a, b) : pack2(a, b);
          }
#if FW_ABL & 1
          if (wv[0] == 0x7fc17fc1u && wv[1] == 0x7fc17fc1u)
#endif
          *reinterpret_cast<uint4*>(out + (row + 8 * hf) * outs.ldo + col) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
      }
    }
  }
#if FW_MC
  cluster_sync_all();
#endif
}

namespace {
auto bf16ptr(const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }
auto bf16c(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }
}  // namespace

// x [R, 128] bf16 (R = B * L * L rows (b, i, j)); lnw / lnb fp32 [128]; w4 = [Wq; Wk; Wv; Wg] [512, 128] bf16; wb16 [16, 128] bf16 (Wb
// in rows 0..3, zeros below); mask [B, L] bool or empty -> (q, k, v, g [R, 128] bf16, sbias [B, 4, L, L] bf16 = bias / scale, masked
// keys -inf)
std::vector<torch::Tensor> tri_front(torch::Tensor x, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor w4, torch::Tensor wb16,
                                     torch::Tensor mask, int64_t B, int64_t L, double scale, bool trans, bool h16, bool gate) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.size(-1) == C, "x: [R, 128] bf16");
  const long R = x.numel() / C;
  TORCH_CHECK(R == B * L * L && L % fr::BM == 0, "rows = B * L * L, L % 128 == 0");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.size(0) == 4 * C && w4.size(1) == C, "w4 [512, 128] bf16");
  TORCH_CHECK(wb16.scalar_type() == torch::kBFloat16 && wb16.is_contiguous() && wb16.size(0) == 16 && wb16.size(1) == C, "wb16 [16, 128] bf16");
  auto opt = x.options();
  const auto hopt = h16 ? opt.dtype(torch::kHalf) : opt;
  auto q = torch::empty({R, C}, hopt), k = torch::empty({R, C}, hopt), v = torch::empty({R, C}, hopt), g = torch::empty({gate ? R : 0, C}, opt);
  auto bias = torch::empty({B, NH, L, L}, hopt);
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous();
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  auto pm = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    if (!trans) return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what);
    return make_map<4>(t.data_ptr(), {(uint64_t)C, (uint64_t)L, (uint64_t)L, (uint64_t)(rows / ((uint64_t)L * L))},
                       {(uint64_t)L * C, (uint64_t)C, (uint64_t)L * L * C}, {64, br, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap xm = pm(x, R, 128, "x"), wm = m2(w4, 4 * C, 256, "w4"), wbm = m2(wb16, 16, 16, "wb16");
  CUtensorMap qm = m2(q, R, 64, "q"), km = m2(k, R, 64, "k"), vm = m2(v, R, 64, "v"), gm = m2(gate ? g : q, R, 64, "g");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_sm90<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, fr::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_sm90<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, fr::SMEM));
    attr = true;
  }
  const int ntiles = (int)(R / fr::BM);
  (gate ? tri_front_sm90<4> : tri_front_sm90<3>)<<<std::min(ntiles, num_sms(x.device().index())), fr::THREADS, fr::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, (float)eps, (float)(1.0 / scale), (int)trans, (int)h16, xm, wm, wbm, lw.data_ptr<float>(), lb.data_ptr<float>(), mp,
      qm, km, vm, gm, bias.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, v, g, bias};
}
// out = x + drop o ((sigmoid(g [+ gb]) o o) . Wo^T [+ bo]); g, o, x [R, 128] bf16, wo [128, 128] bf16; drop [B, L, 128] bf16 or empty;
// `out`: a contiguous bf16 [R, 128] to write into (a chunk's rows of a batch's output), else a new tensor; bo / gb: fp32 [128] or none
// (RoseTTAFold3's out-projection / gate biases, in fp32 before the one rounding of each).  With `wg` ([128, 128] bf16, Wg) and the
// LayerNorm (`lnw` / `lnb` fp32 [128], `eps`): g = LN(x) . Wg^T is projected here (fp32, never stored) and `g` is not read (may be empty)
torch::Tensor tri_tail(torch::Tensor g, torch::Tensor o, torch::Tensor x, torch::Tensor wo, torch::Tensor drop, int64_t L, bool trans,
                       c10::optional<torch::Tensor> out_into, c10::optional<torch::Tensor> bo, c10::optional<torch::Tensor> gb,
                       c10::optional<torch::Tensor> wg, c10::optional<torch::Tensor> lnw, c10::optional<torch::Tensor> lnb, double eps) {
  for (const auto* b : {&bo, &gb, &lnw, &lnb})
    if (b->has_value())
      TORCH_CHECK((*b)->is_cuda() && (*b)->scalar_type() == torch::kFloat32 && (*b)->is_contiguous() && (*b)->numel() == C,
                  "bo / gb / lnw / lnb: fp32 [128]");
  const bool gln = wg.has_value();
  TORCH_CHECK(!gln || (lnw.has_value() && lnb.has_value() && wg->is_cuda() && wg->scalar_type() == torch::kBFloat16 && wg->is_contiguous() &&
                       wg->numel() == C * C), "wg [128, 128] bf16 with lnw / lnb");
  for (const auto* t : {&o, &x}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == x.numel(), "o / x: [R, 128] bf16");
  if (!gln) TORCH_CHECK(g.is_cuda() && g.scalar_type() == torch::kBFloat16 && g.is_contiguous() && g.numel() == x.numel(), "g: [R, 128] bf16");
  const long R = x.numel() / C;
  TORCH_CHECK(R % tl::BM == 0 && L % tl::BM == 0, "rows % 128, L % 128");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == C * C, "wo [128, 128] bf16");
  const bool dr = drop.defined() && drop.numel() > 0;
  if (dr) TORCH_CHECK(drop.scalar_type() == torch::kBFloat16 && drop.is_contiguous() && drop.numel() * L == R * C, "drop: [B, L, 128] bf16");
  if (out_into.has_value())
    TORCH_CHECK(out_into->is_cuda() && out_into->scalar_type() == torch::kBFloat16 && out_into->is_contiguous() && out_into->numel() == R * C,
                "out: contiguous bf16 [R, 128]");
  auto out = out_into.has_value() ? out_into->view({R, C}) : torch::empty({R, C}, x.options());
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  auto pm = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what, bool tp) {
    if (!tp) return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what);
    return make_map<4>(t.data_ptr(), {(uint64_t)C, (uint64_t)L, (uint64_t)L, (uint64_t)(rows / ((uint64_t)L * L))},
                       {(uint64_t)L * C, (uint64_t)C, (uint64_t)L * L * C}, {64, br, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  const int tr = !trans ? 0 : (gln && !dr) ? 2 : 1;        // see tri_tail_sm90
  CUtensorMap gm = m2(gln ? o : g, R, 128, "g"), om = pm(o, R, 128, "o", tr == 2), xm = pm(x, R, 128, "x", tr == 1), wm = m2(wo, C, 128, "wo"),
              ym = pm(out, R, 64, "out", tr == 1), wgm = m2(gln ? *wg : wo, C, 128, "wg");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_tail_sm90<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, tl::Cfg<false>::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_tail_sm90<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, tl::Cfg<true>::SMEM));
    attr = true;
  }
  const int ntiles = (int)(R / tl::BM);
  const int jb = (int)(L / tl::BM), grid = std::max(jb, std::min(ntiles, num_sms(x.device().index())) / jb * jb);   // a multiple of the j blocks
  (gln ? tri_tail_sm90<true> : tri_tail_sm90<false>)<<<grid, tl::THREADS, gln ? tl::Cfg<true>::SMEM : tl::Cfg<false>::SMEM,
                                                       at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, dr ? reinterpret_cast<const __nv_bfloat16*>(drop.data_ptr<at::BFloat16>()) : nullptr, tr,
      bo.has_value() ? bo->data_ptr<float>() : nullptr, gb.has_value() ? gb->data_ptr<float>() : nullptr, gm, om, xm, wm, ym, wgm,
      gln ? lnw->data_ptr<float>() : nullptr, gln ? lnb->data_ptr<float>() : nullptr, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// dy, g, o [R, 128] bf16 (rows (b, i, j)), wo [128, 128] bf16, drop [B, L, 128] bf16 or empty
// -> (do, dg [R, 128] bf16, delta fp32 [B, L, 4, L], seed [R, 128] bf16 = dy, dWo partials fp32 [grid, 128 * 128] for tri_wgrad)
std::vector<torch::Tensor> tri_gate_bwd(torch::Tensor dy, torch::Tensor g, torch::Tensor o, torch::Tensor wo, torch::Tensor drop,
                                        int64_t B, int64_t L, bool trans) {
  for (const auto* t : {&dy, &g, &o}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == B * L * L * C, "dy / g / o: [R, 128] bf16");
  const long R = B * L * L;
  TORCH_CHECK(L % gb::BM == 0, "L % 128");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == C * C, "wo [128, 128] bf16");
  const bool dr = drop.defined() && drop.numel() > 0;
  if (dr) TORCH_CHECK(drop.scalar_type() == torch::kBFloat16 && drop.is_contiguous() && drop.numel() == B * L * C, "drop: [B, L, 128] bf16");
  auto opt = dy.options();
  auto dO = torch::empty({R, C}, opt), dG = torch::empty({R, C}, opt), seed = torch::empty({R, C}, opt);
  auto delta = torch::empty({B, L, 4, L}, opt.dtype(torch::kFloat32));
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  auto pm = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    if (!trans) return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what);
    return make_map<4>(t.data_ptr(), {(uint64_t)C, (uint64_t)L, (uint64_t)L, (uint64_t)(rows / ((uint64_t)L * L))},
                       {(uint64_t)L * C, (uint64_t)C, (uint64_t)L * L * C}, {64, br, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap ym = pm(dy, R, 128, "dy"), gm = m2(g, R, 128, "g"), om = m2(o, R, 128, "o"), wm = m2(wo, C, 128, "wo");
  CUtensorMap dom = m2(dO, R, 64, "do"), dgm = m2(dG, R, 64, "dg"), sm_ = pm(seed, R, 64, "seed");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_gate_bwd_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, gb::SMEM)); attr = true; }
  const int ntiles = (int)(R / gb::BM);
  const int jb = (int)(L / gb::BM), grid = std::max(jb, std::min(ntiles, num_sms(dy.device().index())) / jb * jb);
  auto dwo = torch::empty({grid, C * C}, opt.dtype(torch::kFloat32));
  tri_gate_bwd_sm90<<<grid, gb::THREADS, gb::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, dr ? reinterpret_cast<const __nv_bfloat16*>(drop.data_ptr<at::BFloat16>()) : nullptr, (int)trans, ym, gm, om, wm, dom, dgm, sm_,
      delta.data_ptr<float>(), dwo.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dG, delta, seed, dwo};
}

// dq, dk, dv, dg [R, 128] bf16; db fp32 [B, 4, L, L]; x [R, 128] (the LN input); w4 [512, 128] bf16; wb [4, 128], lnw [128] fp32;
// dpair [R, 128] bf16 (gate_bwd's seed; dx is added in place)
void tri_head_bwd(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                  torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, double eps, torch::Tensor dpair, int64_t B, int64_t L, bool trans) {
  const long R = B * L * L;
  for (const auto* t : {&dq, &dk, &dv, &dg, &x, &dpair}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.size(0) == 4 * C, "w4 [512, 128] bf16");
  auto wbf = wb.to(torch::kFloat32).contiguous(), lw = lnw.to(torch::kFloat32).contiguous();
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, R, 128, "dq"), m1 = m2(dk, R, 128, "dk"), m2_ = m2(dv, R, 128, "dv"), m3 = m2(dg, R, 128, "dg");
  auto pmk = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    if (!trans) return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what);
    return make_map<4>(t.data_ptr(), {(uint64_t)C, (uint64_t)L, (uint64_t)L, (uint64_t)(rows / ((uint64_t)L * L))},
                       {(uint64_t)L * C, (uint64_t)C, (uint64_t)L * L * C}, {64, br, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap xm = pmk(x, R, 128, "x");
  CUtensorMap wm = m2(w4, 4 * C, 256, "w4"), pm = pmk(dpair, R, 64, "dpair");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_head_bwd_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, hb::SMEM)); attr = true; }
  const int ntiles = (int)(R / hb::BM);
  tri_head_bwd_sm90<<<std::min(ntiles, num_sms(x.device().index())), hb::THREADS, hb::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, (float)eps, (int)trans, m0, m1, m2_, m3, xm, wm, pm, db.data_ptr<float>(), wbf.data_ptr<float>(), lw.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -> (dW_q|k|v|g [4, 128, 128] ([o, c] each: the Linear weight layout), dgamma | dbeta | dWb [6, 128], dWo [128, 128]) in fp32 when
// fp32_out (master parameters) else bf16; dwo_part: gate_bwd's dWo partials [grid, 128 * 128]
std::vector<torch::Tensor> tri_wgrad(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                                     double eps, int64_t B, int64_t L, torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, torch::Tensor lnb,
                                     torch::Tensor dwo_part, bool fp32_out, bool trans) {
  const long R = B * L * L;
  for (const auto* t : {&dq, &dk, &dv, &dg, &x}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  TORCH_CHECK(dwo_part.scalar_type() == torch::kFloat32 && dwo_part.is_contiguous() && dwo_part.numel() % (C * C) == 0, "dwo_part fp32 [grid, 128 * 128]");
  auto fopt = x.options().dtype(torch::kFloat32);
  auto mt = torch::empty({4L * C * C}, fopt), vec = torch::empty({8L * C + NH}, fopt), dwo32 = torch::empty({C * C}, fopt);
  auto m2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, (uint64_t)R}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, "dq"), m1 = m2(dk, "dk"), m2_ = m2(dv, "dv"), m3 = m2(dg, "dg");
  CUtensorMap xm = trans ? make_map<4>(x.data_ptr(), {(uint64_t)C, (uint64_t)L, (uint64_t)L, (uint64_t)B}, {(uint64_t)L * C, (uint64_t)C, (uint64_t)L * L * C},
                                       {64, 128, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "x")
                         : m2(x, "x");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wgrad_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, wg::SMEM)); attr = true; }
  const int ntiles = (int)(R / wg::BM);
  const int splits = std::max(1, std::min(ntiles, num_sms(x.device().index()) / 2));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto slots = torch::empty({2L * splits * wg::SLOT}, fopt);
  tri_wgrad_sm90<<<2 * splits, wg::THREADS, wg::SMEM, stream>>>((int)L, ntiles, (float)eps, (int)trans, m0, m1, m2_, m3, xm, db.data_ptr<float>(),
                                                                  slots.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long nred = 4L * C * C + 8L * C + NH + C * C;
  tri_grad_reduce<<<(int)((nred + 31) / 32), 256, 0, stream>>>(slots.data_ptr<float>(), splits, dwo_part.data_ptr<float>(),
      (int)(dwo_part.numel() / (C * C)), mt.data_ptr<float>(), vec.data_ptr<float>(), dwo32.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous(), wbf = wb.to(torch::kFloat32).contiguous();
  auto oopt = fp32_out ? x.options().dtype(torch::kFloat32) : x.options();
  torch::Tensor o8[8] = {torch::empty({C, C}, oopt), torch::empty({C, C}, oopt), torch::empty({C, C}, oopt), torch::empty({C, C}, oopt),
                         torch::empty({C}, oopt), torch::empty({C}, oopt), torch::empty({NH, C}, oopt), torch::empty({C, C}, oopt)};
  const auto* w4p = reinterpret_cast<const __nv_bfloat16*>(w4.data_ptr<at::BFloat16>());
  if (fp32_out) {
    Outs<float> O{{o8[0].data_ptr<float>(), o8[1].data_ptr<float>(), o8[2].data_ptr<float>(), o8[3].data_ptr<float>()}, o8[4].data_ptr<float>(),
                  o8[5].data_ptr<float>(), o8[6].data_ptr<float>(), o8[7].data_ptr<float>()};
    tri_param_finish<float><<<64 + C + 16, 256, 0, stream>>>(mt.data_ptr<float>(), vec.data_ptr<float>(), w4p, lw.data_ptr<float>(),
        lb.data_ptr<float>(), wbf.data_ptr<float>(), O, dwo32.data_ptr<float>());
  } else {
    Outs<__nv_bfloat16> O{{bf16ptr(o8[0]), bf16ptr(o8[1]), bf16ptr(o8[2]), bf16ptr(o8[3])}, bf16ptr(o8[4]), bf16ptr(o8[5]), bf16ptr(o8[6]),
                          bf16ptr(o8[7])};
    tri_param_finish<__nv_bfloat16><<<64 + C + 16, 256, 0, stream>>>(mt.data_ptr<float>(), vec.data_ptr<float>(), w4p, lw.data_ptr<float>(),
        lb.data_ptr<float>(), wbf.data_ptr<float>(), O, dwo32.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o8[0], o8[1], o8[2], o8[3], o8[4], o8[5], o8[6], o8[7]};
}

// one launch: [Wq; Wk; Wv; Wg] -> w4 [512, 128] bf16, Wb -> wb16 [16, 128] (rows >= 4 zero), Wo -> wo16 [128, 128] bf16 (inputs fp32 or bf16)
__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
__global__ void __launch_bounds__(256) tri_pack_k(const T* __restrict__ wq, const T* __restrict__ wk, const T* __restrict__ wv, const T* __restrict__ wg,
                                                  const T* __restrict__ wb, const T* __restrict__ wo, __nv_bfloat16* __restrict__ w4,
                                                  __nv_bfloat16* __restrict__ wb16, __nv_bfloat16* __restrict__ wo16) {
  const long n4 = 4L * C * C, nb = 16L * C, no = (long)C * C;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n4 + nb + no; i += (long)gridDim.x * 256) {
    float v;
    if (i < n4) {
      const int t = (int)(i / (C * C)); const long e = i - (long)t * C * C;
      const T* src = t == 0 ? wq : t == 1 ? wk : t == 2 ? wv : wg;
      v = to_f32(src[e]);
      w4[i] = __float2bfloat16_rn(v);
    } else if (i < n4 + nb) {
      const long j = i - n4;
      v = j < (long)NH * C ? to_f32(wb[j]) : 0.f;
      wb16[j] = __float2bfloat16_rn(v);
    } else {
      const long j = i - n4 - nb;
      wo16[j] = __float2bfloat16_rn(to_f32(wo[j]));
    }
  }
}
std::vector<torch::Tensor> tri_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo) {
  const auto dt = wq.scalar_type();
  for (const auto* t : {&wq, &wk, &wv, &wg, &wo}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == dt && t->is_contiguous() && t->numel() == (long)C * C, "w [128, 128]");
  TORCH_CHECK(wb.scalar_type() == dt && wb.is_contiguous() && wb.numel() == (long)NH * C, "wb [4, 128]");
  TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kBFloat16, "fp32 or bf16 parameters");
  auto bopt = wq.options().dtype(torch::kBFloat16);
  auto w4 = torch::empty({4 * C, C}, bopt), wb16 = torch::empty({16, C}, bopt), wo16 = torch::empty({C, C}, bopt);
  const long total = 4L * C * C + 16L * C + (long)C * C;
  const int grid = (int)std::min<long>((total + 255) / 256, 1024);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dt == torch::kFloat32)
    tri_pack_k<float><<<grid, 256, 0, stream>>>(wq.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), wg.data_ptr<float>(), wb.data_ptr<float>(),
        wo.data_ptr<float>(), bf16ptr(w4), bf16ptr(wb16), bf16ptr(wo16));
  else
    tri_pack_k<__nv_bfloat16><<<grid, 256, 0, stream>>>(bf16c(wq), bf16c(wk), bf16c(wv), bf16c(wg), bf16c(wb), bf16c(wo), bf16ptr(w4),
        bf16ptr(wb16), bf16ptr(wo16));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w4, wb16, wo16};
}

// ---------------------------------------------------------------------------------------------------------------------------------
// wide front (d_pair 256, 8 heads x 32), inference.
// x [R, 256] bf16 (rows (b, i, j)); fold fp32 [2, 1032], wpk [1024, 256] bf16, wb8 [8, 256] bf16 = tri_pack_wide(..., lnw, lnb): the
// LayerNorm folded into the weights ([Wq; Wk; Wv; Wg] diag(gamma), rows permuted within each 128-row half; Wb diag(gamma), zero rows
// below nh) and its per-output constants; mask [B, L] bool or empty; gb fp32 [256] gate bias or none
// -> (q, k, v [R, 256] fp16 when h16 else bf16, g [R, 256] bf16, sbias [B, nh, L, L] fp16 / bf16 = bias / scale, masked keys -inf)
std::vector<torch::Tensor> tri_front_wide(torch::Tensor x, torch::Tensor fold, double eps, torch::Tensor wpk,
                                          torch::Tensor wb8, torch::Tensor mask, int64_t B, int64_t L, int64_t nh, double scale, bool trans,
                                          bool h16, c10::optional<torch::Tensor> gbias, bool packed) {
  constexpr int CW = 256;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.size(-1) == CW, "x: [R, 256] bf16");
  const long R = x.numel() / CW;
  TORCH_CHECK(R == B * L * L && L % fw::BM == 0, "rows = B * L * L, L % 64 == 0");
  TORCH_CHECK(nh >= 1 && nh <= fw::HB, "1 <= heads <= 8");
  TORCH_CHECK(wpk.scalar_type() == torch::kBFloat16 && wpk.is_contiguous() && wpk.size(0) == 4 * fw::NG && wpk.size(1) == CW, "wpk [1024, 256] bf16");
  TORCH_CHECK(wb8.scalar_type() == torch::kBFloat16 && wb8.is_contiguous() && wb8.size(0) == fw::HB && wb8.size(1) == CW, "wb8 [8, 256] bf16");
  if (gbias.has_value())
    TORCH_CHECK(gbias->is_cuda() && gbias->scalar_type() == torch::kFloat32 && gbias->is_contiguous() && gbias->numel() == CW, "gb: fp32 [256]");
  auto opt = x.options();
  const auto hopt = h16 ? opt.dtype(torch::kHalf) : opt;
  // packed: q | k | v | g as column blocks of one [R, 4 CW] buffer (row stride 4 CW)
  torch::Tensor q, k, v, g;
  if (packed) {
    auto all = torch::empty({R, 4 * CW}, hopt);
    q = all.narrow(1, 0, CW); k = all.narrow(1, CW, CW); v = all.narrow(1, 2 * CW, CW);
    g = all.narrow(1, 3 * CW, CW);
    if (h16) g = g.view(torch::kBFloat16);
  } else {
    q = torch::empty({R, CW}, hopt); k = torch::empty({R, CW}, hopt); v = torch::empty({R, CW}, hopt); g = torch::empty({R, CW}, opt);
  }
  auto bias = torch::empty({B, nh, L, L}, hopt);
  TORCH_CHECK(fold.is_cuda() && fold.scalar_type() == torch::kFloat32 && fold.is_contiguous() && fold.numel() == 2 * (4 * CW + fw::HB),
              "fold: fp32 [2, 4 * 256 + 8] (tri_pack_wide)");
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)CW, rows}, {(uint64_t)CW}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap xm = !trans ? m2(x, R, fw::BM, "x")
                          : make_map<4>(x.data_ptr(), {(uint64_t)CW, (uint64_t)L, (uint64_t)L, (uint64_t)B},
                                        {(uint64_t)L * CW, (uint64_t)CW, (uint64_t)L * L * CW}, {64, (uint32_t)fw::BM, 1, 1},
                                        CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap wm = m2(wpk, 4 * fw::NG, fw::NG, "wpk"), wbm = m2(wb8, fw::HB, fw::HB, "wb8");
  using Q = fw::Cfg<CW>;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_wide_sm90<CW>, cudaFuncAttributeMaxDynamicSharedMemorySize, Q::SMEM)); attr = true; }
  const int ntiles = (int)(R / fw::BM);
  // groups of four CTAs (q, k, v, g), at most one CTA per SM, each consumer warpgroup with a tile
  const int groups = std::max(1, std::min(num_sms(x.device().index()) / 4, (ntiles + 1) / 2));
  fw::Outs outs{{q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr()}, (long)q.stride(0)};
  auto sync = torch::zeros({FW_LAG > 0 && !FW_MC ? groups : 0}, opt.dtype(torch::kInt32));   // the groups' tile counters (FW_LAG)
#if FW_MC
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(4 * groups); cfg.blockDim = dim3(fw::THREADS); cfg.dynamicSmemBytes = Q::SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 4; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tri_front_wide_sm90<CW>,
      (int)L, ntiles, (float)eps, (float)(1.0 / scale), (int)trans, (int)h16, (int)nh, xm, wm, wbm, fold.data_ptr<float>(),
      mp, gbias.has_value() ? gbias->data_ptr<float>() : nullptr, outs, bias.data_ptr(), nullptr));
#else
  tri_front_wide_sm90<CW><<<4 * groups, fw::THREADS, Q::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, (float)eps, (float)(1.0 / scale), (int)trans, (int)h16, (int)nh, xm, wm, wbm, fold.data_ptr<float>(),
      mp, gbias.has_value() ? gbias->data_ptr<float>() : nullptr, outs, bias.data_ptr(), FW_LAG > 0 ? sync.data_ptr<int>() : nullptr);
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, v, g, bias};
}

// the wide pack, LayerNorm folded in (y = (x - mu) r gamma + beta, so y . W^T = r (x . Wf^T - mu S1) + S0 with Wf = W diag(gamma),
// S1 = Wf 1 (of the bf16 Wf the GEMM reads), S0 = W beta): [Wq; Wk; Wv; Wg] [C, C] each -> wpk [4 C, C] bf16 = Wf in the wide front's row order (within each 128-row half, packed row
// n = 8 i + 2 t + e holds output row 32 (i / 4) + 8 t + 2 (i % 4) + e), Wb [nh, C] -> wb8 [8, C] = Wb diag(gamma) (rows >= nh zero), Wo -> wo16 [C, C] bf16;
// tri_fold_wide_k: fold [2][4 C + 8] fp32 = S1 | S0 per output column of q | k | v | g, then per head (sums in fp64, one thread each)
template <typename T>
__global__ void __launch_bounds__(256) tri_pack_wide_k(const T* __restrict__ wq, const T* __restrict__ wk, const T* __restrict__ wv,
                                                       const T* __restrict__ wg, const T* __restrict__ wb, const T* __restrict__ wo,
                                                       const float* __restrict__ gam, int Cw, int nh,
                                                       __nv_bfloat16* __restrict__ wpk, __nv_bfloat16* __restrict__ wb8,
                                                       __nv_bfloat16* __restrict__ wo16) {
  const long cc = (long)Cw * Cw, n4 = 4 * cc, nb = (long)fw::HB * Cw;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n4 + nb + cc; i += (long)gridDim.x * 256) {
    if (i < n4) {
      const int t = (int)(i / cc);
      const long e = i - (long)t * cc;
      const int prow = (int)(e / Cw), c = (int)(e - (long)prow * Cw);
      const int n = prow & 127, ii = n >> 3, tt = (n >> 1) & 3, ee = n & 1;
      const int orow = (prow & ~127) + 32 * (ii >> 2) + 8 * tt + 2 * (ii & 3) + ee;
      const T* src = t == 0 ? wq : t == 1 ? wk : t == 2 ? wv : wg;
      wpk[i] = __float2bfloat16_rn(to_f32(src[(long)orow * Cw + c]) * gam[c]);
    } else if (i < n4 + nb) {
      const long j = i - n4;
      wb8[j] = __float2bfloat16_rn(j < (long)nh * Cw ? to_f32(wb[j]) * gam[j % Cw] : 0.f);
    } else {
      const long j = i - n4 - nb;
      wo16[j] = __float2bfloat16_rn(to_f32(wo[j]));
    }
  }
}
template <typename T>
__global__ void __launch_bounds__(256) tri_fold_wide_k(const T* __restrict__ wq, const T* __restrict__ wk, const T* __restrict__ wv,
                                                       const T* __restrict__ wg, const T* __restrict__ wb, const float* __restrict__ gam,
                                                       const float* __restrict__ bet, int Cw, int nh, float* __restrict__ fold) {
  const int ns = 4 * Cw + fw::HB, n = blockIdx.x * 256 + threadIdx.x;
  if (n >= ns) return;
  const int t = n / Cw, row = n - t * Cw;                    // t = 4: the heads
  double s1 = 0.0, s0 = 0.0;
  if (t < 4 || row < nh) {
    const T* src = (t == 0 ? wq : t == 1 ? wk : t == 2 ? wv : t == 3 ? wg : wb) + (long)row * Cw;
    for (int c = 0; c < Cw; ++c) {
      const float w = to_f32(src[c]);
      s1 += (double)__bfloat162float(__float2bfloat16_rn(w * gam[c]));
      s0 += (double)w * (double)bet[c];
    }
  }
  fold[n] = (float)s1;
  fold[ns + n] = (float)s0;
}
std::vector<torch::Tensor> tri_pack_wide(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo,
                                         torch::Tensor lnw, torch::Tensor lnb) {
  const auto dt = wq.scalar_type();
  const long Cw = wq.size(1);
  const int nh = (int)wb.size(0);
  TORCH_CHECK(Cw % 128 == 0 && wq.size(0) == Cw, "wq [C, C], C % 128 == 0");
  for (const auto* t : {&wq, &wk, &wv, &wg, &wo}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == dt && t->is_contiguous() && t->numel() == Cw * Cw, "w [C, C]");
  TORCH_CHECK(wb.scalar_type() == dt && wb.is_contiguous() && nh <= fw::HB && wb.numel() == nh * Cw, "wb [nh <= 8, C]");
  TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kBFloat16, "fp32 or bf16 parameters");
  for (const auto* t : {&lnw, &lnb})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->is_contiguous() && t->numel() == Cw, "lnw / lnb: fp32 [C]");
  auto bopt = wq.options().dtype(torch::kBFloat16);
  auto wpk = torch::empty({4 * Cw, Cw}, bopt), wb8 = torch::empty({fw::HB, Cw}, bopt), wo16 = torch::empty({Cw, Cw}, bopt);
  auto fold = torch::empty({2, 4 * Cw + fw::HB}, wq.options().dtype(torch::kFloat32));
  const float *gp = lnw.data_ptr<float>(), *bp = lnb.data_ptr<float>();
  const int fgrid = (int)((4 * Cw + fw::HB + 255) / 256);
  const long total = 5 * Cw * Cw + fw::HB * Cw;
  const int grid = (int)std::min<long>((total + 255) / 256, 2048);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dt == torch::kFloat32) {
    tri_pack_wide_k<float><<<grid, 256, 0, stream>>>(wq.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), wg.data_ptr<float>(),
        wb.data_ptr<float>(), wo.data_ptr<float>(), gp, (int)Cw, nh, bf16ptr(wpk), bf16ptr(wb8), bf16ptr(wo16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    tri_fold_wide_k<float><<<fgrid, 256, 0, stream>>>(wq.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), wg.data_ptr<float>(),
        wb.data_ptr<float>(), gp, bp, (int)Cw, nh, fold.data_ptr<float>());
  } else {
    tri_pack_wide_k<__nv_bfloat16><<<grid, 256, 0, stream>>>(bf16c(wq), bf16c(wk), bf16c(wv), bf16c(wg), bf16c(wb), bf16c(wo), gp, (int)Cw, nh,
        bf16ptr(wpk), bf16ptr(wb8), bf16ptr(wo16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    tri_fold_wide_k<__nv_bfloat16><<<fgrid, 256, 0, stream>>>(bf16c(wq), bf16c(wk), bf16c(wv), bf16c(wg), bf16c(wb), gp, bp, (int)Cw, nh,
        fold.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wpk, wb8, wo16, fold};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_front_wide", &tri_front_wide, "sm90 TriangleAttention front at d_pair 256 / 8 heads x 32 (inference): LN + q|k|v|g + head-major bias / scale",
        py::arg("x"), py::arg("fold"), py::arg("eps"), py::arg("wpk"), py::arg("wb8"), py::arg("mask"), py::arg("B"),
        py::arg("L"), py::arg("nh"), py::arg("scale"), py::arg("trans") = false, py::arg("h16") = true, py::arg("gb") = py::none(), py::arg("packed") = false);
  m.def("tri_pack_wide", &tri_pack_wide, "sm90 TriangleAttention wide weight pack, LayerNorm folded: (wpk [4C, C], wb8 [8, C], wo16 [C, C] bf16, fold [2, 4C + 8] fp32)");
  m.def("tri_wgrad", &tri_wgrad, "sm90 TriangleAttention parameter gradients: (dWq, dWk, dWv, dWg [128,128], dgamma, dbeta [128], dWb [4,128], dWo [128,128])");
  m.def("tri_pack", &tri_pack, "sm90 TriangleAttention weight pack: (w4 [512,128] bf16, wb16 [16,128] bf16, wo16 [128,128] bf16) in one launch");
  m.def("tri_head_bwd", &tri_head_bwd, "sm90 TriangleAttention head backward: projection dgrad + LN backward, dx added into dpair");
  m.def("tri_gate_bwd", &tri_gate_bwd, "sm90 TriangleAttention gate backward: (do, dg, delta, seed); dWo added into a fp32 accumulator");
  m.def("tri_tail", &tri_tail, "sm90 TriangleAttention tail: x + drop o ((sigmoid(g [+ gb]) o o) . Wo^T [+ bo])", py::arg("g"),
        py::arg("o"), py::arg("x"), py::arg("wo"), py::arg("drop"), py::arg("L"), py::arg("trans") = false, py::arg("out") = py::none(),
        py::arg("bo") = py::none(), py::arg("gb") = py::none(), py::arg("wg") = py::none(), py::arg("lnw") = py::none(),
        py::arg("lnb") = py::none(), py::arg("eps") = 1e-5);
  m.def("tri_front", &tri_front, "sm90 TriangleAttention front: LN + q|k|v|g projections + head-major (masked) bias / scale",
        py::arg("x"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("w4"), py::arg("wb16"), py::arg("mask"), py::arg("B"),
        py::arg("L"), py::arg("scale"), py::arg("trans") = false, py::arg("h16") = false, py::arg("gate") = true);
}
