// ESMFold2 SWA atom DiT block, fp32 (CUDA sm_90a): every stage of the fp32 fused block (kernels/swa_dit, the CUDA twin of
// triton/forward_fp32.py + backward_fp32.py), C = 128, H = 4, D = 32, SwiGLU hidden 256, half window 64.
//
// Precision. The residual stream, the norms, the modulation, the gates, RoPE, the SwiGLU nonlinearity and every saved
// tensor are fp32. GEMM operands are rounded to tf32 with cvt.rna (round to nearest, ties away -- what cuBLAS's TF32
// does), never truncated: the weights once per call by prep_kernel, the activations as they become wgmma operands.
//   Q / K / V / gate and out projections, and their dgrads   one tf32 pass (the per-op path's cuBLAS TF32)
//   the SwiGLU FFN GEMMs (y Wu^T, h Wd^T; a|b recomputed, dffn Wd, [da|db] Wab)
//                                                             split weights: x_tf32 * (W_hi + W_lo), W = W_hi + W_lo
//   window attention                                          bf16 q / k / v / P / dO / dS, fp32 accumulate + softmax
// Splitting the weights (two tf32 passes) takes the FFN from 1.6x the Triton path's weight-gradient error with one
// pass down to the per-op path's level; splitting the activations too (tf32x3) would halve the rest but costs every
// FFN activation a second operand. Weight gradients are NOT here: dispatch.py takes them as torch GEMMs over the saved
// fp32 operands, following the process's matmul precision.
//
// Row kernels (qkvg forward, out-projection + FFN forward, FFN backward, out-projection backward, qkvg backward): a
// 64-row tile (SP augments x AT atoms of one batch element) runs on a PAIR of CTAs of one warpgroup each, persistent
// over tiles. The pair splits the tile's GEMM work -- the four projections two and two, alternate 32-wide FFN hidden
// chunks, or output column halves -- so each CTA streams half of the weights through its own ring of 16 KB TMA stages
// (128 B swizzle, K-major; thread 0 refills a slot once the wgmmas that read it have retired, running ahead into the
// next tile). Row tensors move by TMA both ways (boxes {32 fp32 | 64 bf16, AT atoms, 1, SP augments} of their
// [A][B][S][width] view) and the threads touch them only in shared memory; thread 0 prefetches the next tile's inputs
// as soon as a buffer is free. Operands a CTA needs whole (x, y, dffn, datt, gated) it builds whole; a sum split over K
// between the pair (FFN down-projection, dy, dx) and the row statistics after it cross by st.async into the partner's
// shared memory, completing bytes on the partner's mbarrier (no cluster barriers on the way).
//
// GEMM mechanics. A tf32 wgmma A operand in registers (RS) holds, per k8 step, rows g / g+8 at k = t and k = t + 4
// (g = lane / 4, t = lane % 4); an accumulator holds rows g / g+8 at columns 2t, 2t+1 of every n8 block. So an
// accumulator-layout value IS an A operand once the GEMM's K dimension is permuted inside each group of 8: logical k ->
// physical column perm8(k) = {0,2,4,6,1,3,5,7}[k] -- prep_kernel permutes those weights' K. An SS A operand is a TMA
// tile (natural K) rounded to tf32 in place.
//
// The d shift / scale / gate modulation gradients are summed over a tile's augments in shared memory and added into
// dmod [B*S, 6C] with red.global.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>

namespace {
constexpr int C = 128, H = 4, D = 32, NHID = 256, MODW = 6 * C, HWIN = 64;
constexpr int STAGE = 16384;

// ---------------------------------------------------------------- wgmma / TMA / barrier primitives
namespace wg {
__device__ __forceinline__ uint64_t desc(const void* p) {  // K-major, 128 B swizzle, 8-row groups 1024 B apart
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(64) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
#include "gen/wgmma_tf32.inc"
__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(K) : "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
}  // namespace wg

namespace tma {
__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, void* dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void load_4d(const void* map, void* dst, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void load_5d(const void* map, void* dst, uint64_t* bar, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}
__device__ __forceinline__ void store_4d(const void* map, const void* src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void store_5d(const void* map, const void* src, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}
__device__ __forceinline__ void store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void store_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(K) : "memory"); }
__device__ __forceinline__ void store_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
}  // namespace tma

// cluster of two CTAs: a barrier over both (only at start and end), distributed shared memory addresses
namespace cl {
__device__ __forceinline__ void sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t peer(const void* p, int other) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(tma::sa(p)), "r"(other)); return r;
}
}  // namespace cl

__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ void red2(float* p, float a, float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(p), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ float2 ld2(const float* p) { return __ldg(reinterpret_cast<const float2*>(p)); }
__device__ __forceinline__ uint32_t packb(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void stb2(__nv_bfloat16* p, float a, float b) { *reinterpret_cast<uint32_t*>(p) = packb(a, b); }
__device__ __forceinline__ float qsum(float v) {        // sum over the 4 threads of a quad (one row of an n8 block)
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
#define U(x) __float_as_uint(x)

// ---------------------------------------------------------------- tiles
// A tile is SP augments x AT atoms (SP * AT = 64) of one batch element b: tile row r -> augment a0 + r / AT, atom
// s0 + r % AT, sequence row n = a * B + b, flattened row n * S + s, modulation row b * S + s. SP = 8 at A >= 8 (so the
// modulation gradient is pre-summed over 8 augments in the tile), down to 1 at A = 1.
struct Geo {
  int S, A, B, SP, AT, nab, nag, ntile;
};
struct Row {
  long row;                                                 // flattened [N * S] row
  int mrow, n, s;
  bool ok;
};
__device__ __forceinline__ void tile_coords(const Geo& G, int tile, int& b, int& a0, int& s0) {
  const int ab = tile % G.nab, r = tile / G.nab, ag = r % G.nag;
  b = r / G.nag; a0 = ag * G.SP; s0 = ab * G.AT;
}
__device__ __forceinline__ Row row_of(const Geo& G, int b, int a0, int s0, int r) {
  Row o;
  const int a = a0 + r / G.AT, s = s0 + r % G.AT;
  o.ok = a < G.A && s < G.S;
  o.n = (a < G.A ? a : G.A - 1) * G.B + b; o.s = s < G.S ? s : G.S - 1;
  o.row = (long)o.n * G.S + o.s; o.mrow = b * G.S + o.s;
  return o;
}

// Fragment helpers. v[4i + 2rr + e] = (row g + 8rr, column col0 + 8i + 2t + e), i < NI n8 blocks (the accumulator layout).
template <int NI, class F> __device__ __forceinline__ void for_frag(F f) {
#pragma unroll
  for (int i = 0; i < NI; ++i)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) f(i, rr);
}
template <int NI> __device__ __forceinline__ void load_mod(float* v, const float* mod, const Row* R, int t, int col0) {
  for_frag<NI>([&](int i, int rr) {
    const float2 x = R[rr].ok ? ld2(mod + (long)R[rr].mrow * MODW + col0 + 8 * i + 2 * t) : make_float2(0.f, 0.f);
    v[4 * i + 2 * rr] = x.x; v[4 * i + 2 * rr + 1] = x.y;
  });
}
// accumulator layout -> tf32 A fragments in place (k-step i: rows g, g+8 at logical k = t, t+4 = physical cols 2t, 2t+1)
template <int NI> __device__ __forceinline__ void frag_inplace(float* v) {
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const float a = v[4 * i], b = v[4 * i + 1], c = v[4 * i + 2], d = v[4 * i + 3];
    v[4 * i] = __uint_as_float(tf32r(a)); v[4 * i + 1] = __uint_as_float(tf32r(c));
    v[4 * i + 2] = __uint_as_float(tf32r(b)); v[4 * i + 3] = __uint_as_float(tf32r(d));
  }
}
// partial row sums (this thread's columns, quad-reduced) of a * b for rows g, g + 8
template <int NI> __device__ __forceinline__ void row_dot(const float* a, const float* b, float& s0, float& s1) {
  s0 = 0.f; s1 = 0.f;
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    s0 += a[4 * i] * b[4 * i] + a[4 * i + 1] * b[4 * i + 1];
    s1 += a[4 * i + 2] * b[4 * i + 2] + a[4 * i + 3] * b[4 * i + 3];
  }
  s0 = qsum(s0); s1 = qsum(s1);
}
// fp32 tile of 32-column blocks ([64 rows][128 B] each, 128 B swizzle -- the TMA box layout): byte offset of (row, col)
__device__ __forceinline__ int offa(int r, int k) { return (k >> 5) * 8192 + r * 128 + ((((k & 31) >> 2) ^ (r & 7)) << 4) + (k & 3) * 4; }

// ---------------------------------------------------------------- weight rings
// Each CTA streams its own stage sequence: NSW stages per tile, the same for every tile, through NST slots.
// `issue(gc)` (thread 0) loads stage gc % NSW into slot gc % NST; wait(gc) returns the slot.
struct WRing {
  unsigned char* slot;
  uint64_t* full;
  int nst;
  __device__ __forceinline__ unsigned char* wait(int gc) const {
    tma::wait(full + gc % nst, (gc / nst) & 1);
    return slot + (gc % nst) * STAGE;
  }
};

// ================================================================ prep: the weights as the GEMMs stream them
// perm8: logical k -> physical column inside each group of 8 (see the header). hi = rna(w), lo = rna(w - hi).
__device__ __forceinline__ int perm8(int k) { const int j = k & 7; return (k & ~7) + (j < 4 ? 2 * j : 2 * (j - 4) + 1); }

// Forward (fwd = 1):
//   W1  [512][128]  rows [Wq; Wk; Wv; Wg], K = in                      x Wp^T                (qkvg)
//   Wo  [128][128]  K = in                                              gated Wo^T            (out-projection)
//   Wab hi, lo [512][128]  row 64j + r: Wu[32j + r] (r < 32), Wu[256 + 32j + r - 32]; K = in      y [Wa_j; Wb_j]^T
//   Wd  hi, lo [128][256]  K = hidden                                    h Wd^T
// Backward (fwd = 0):
//   Wab hi, lo [512][128]  as above but K NATURAL (y is an SS operand straight from its TMA tile)
//   WdT hi, lo [256][128]  WdT[j][c] = Wd[c][j], K NATURAL (dffn is an SS operand in the TMA tile layout)
//   WabT hi, lo [128][512] WabT[c][64j + k] = [Wa_j; Wb_j][perm8(k)][c]  [da | db]_j [Wa_j; Wb_j]
//   WoT [128][128]  WoT[c][k] = Wo[perm8(k)][c]                          datt Wo
//   W1T [128][512]  W1T[c][k] = W1[perm8(k)][c]                          dP [Wq; Wk; Wv; Wg]
// (K permuted with perm8 wherever the A operand comes from registers; the FFN groups hold hi and lo)
constexpr int F_W1 = 0, F_WO = F_W1 + 512 * C, F_WABH = F_WO + C * C, F_WABL = F_WABH + 512 * C, F_WDH = F_WABL + 512 * C,
              F_WDL = F_WDH + C * NHID, F_END = F_WDL + C * NHID;
constexpr int B_WABH = 0, B_WABL = B_WABH + 512 * C, B_WDTH = B_WABL + 512 * C, B_WDTL = B_WDTH + NHID * C,
              B_WABTH = B_WDTL + NHID * C, B_WABTL = B_WABTH + C * 512, B_WOT = B_WABTL + C * 512, B_W1T = B_WOT + C * C,
              B_END = B_W1T + C * 512;

__device__ __forceinline__ float wab_src(const float* wu, int row, int c) {   // Wab[row][c], chunk j = row / 64
  const int j = row >> 6, r = row & 63;
  return wu[(long)(r < 32 ? 32 * j + r : NHID + 32 * j + r - 32) * C + c];
}
__device__ __forceinline__ float w1_src(const float* wqkv, const float* wg, int o, int c) {
  return o < 3 * C ? wqkv[(long)o * C + c] : wg[(long)(o - 3 * C) * C + c];
}

__global__ void prep_kernel(float* out, const float* __restrict__ wqkv, const float* __restrict__ wg, const float* __restrict__ wo,
                            const float* __restrict__ wu, const float* __restrict__ wd, int fwd) {
  const int total = fwd ? F_END : B_END;
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
    float v;
    bool lo = false;
    if (fwd) {
      if (e < F_WO) { const int o = e / C, k = e % C; v = w1_src(wqkv, wg, o, perm8(k)); }
      else if (e < F_WABH) { const int x = e - F_WO, o = x / C, k = x % C; v = wo[o * C + perm8(k)]; }
      else if (e < F_WDH) { const int x = (e - F_WABH) % (512 * C), o = x / C, k = x % C; v = wab_src(wu, o, perm8(k)); lo = e >= F_WABL; }
      else { const int x = (e - F_WDH) % (C * NHID), o = x / NHID, k = x % NHID; v = wd[o * NHID + perm8(k)]; lo = e >= F_WDL; }
    } else {
      if (e < B_WDTH) { const int x = e % (512 * C), o = x / C, k = x % C; v = wab_src(wu, o, k); lo = e >= B_WABL; }
      else if (e < B_WABTH) { const int x = (e - B_WDTH) % (NHID * C), j = x / C, k = x % C; v = wd[k * NHID + j]; lo = e >= B_WDTL; }
      else if (e < B_WOT) {
        const int x = (e - B_WABTH) % (C * 512), c = x / 512, k = x % 512;
        v = wab_src(wu, (k & ~63) + perm8(k & 63), c); lo = e >= B_WABTL;
      }
      else if (e < B_W1T) { const int x = e - B_WOT, c = x / C, k = x % C; v = wo[perm8(k) * C + c]; }
      else { const int x = e - B_W1T, c = x / 512, k = x % 512; v = w1_src(wqkv, wg, perm8(k), c); }
    }
    const uint32_t h = tf32r(v);
    out[e] = __uint_as_float(lo ? tf32r(v - __uint_as_float(h)) : h);
  }
}


// ================================================================ row kernels: one warpgroup per CTA, CTA pairs per tile
constexpr int TPB = 128;
__device__ __forceinline__ void bar1() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

struct TileC {
  int b, a0, s0;
};
// fp32 row-tile block cb (32 columns) of a [64 rows][32] box at buf + i * 8 KB
__device__ __forceinline__ void ld_rows(const CUtensorMap* m, unsigned char* dst, uint64_t* bar, int cb0, int nb, const TileC& T) {
  for (int i = 0; i < nb; ++i) tma::load_4d(m, dst + i * 8192, bar, 32 * (cb0 + i), T.s0, T.b, T.a0);
}
__device__ __forceinline__ void st_rows(const CUtensorMap* m, const unsigned char* src, int cb0, int nb, const TileC& T) {
  for (int i = 0; i < nb; ++i) tma::store_4d(m, src + i * 8192, 32 * (cb0 + i), T.s0, T.b, T.a0);
}
// bf16 rows: 64-column boxes
__device__ __forceinline__ void ld_rows_bf(const CUtensorMap* m, unsigned char* dst, uint64_t* bar, int cb0, int nb, const TileC& T) {
  for (int i = 0; i < nb; ++i) tma::load_4d(m, dst + i * 8192, bar, 64 * (cb0 + i), T.s0, T.b, T.a0);
}
__device__ __forceinline__ void st_rows_bf(const CUtensorMap* m, const unsigned char* src, int cb0, int nb, const TileC& T) {
  for (int i = 0; i < nb; ++i) tma::store_4d(m, src + i * 8192, 64 * (cb0 + i), T.s0, T.b, T.a0);
}
// head-major bf16 [A][B][H][S][32]: head h box [64 rows][64 B] (64 B swizzle) at buf + h * 4 KB
__device__ __forceinline__ void ld_hm(const CUtensorMap* m, unsigned char* dst, uint64_t* bar, const TileC& T) {
  for (int h = 0; h < H; ++h) tma::load_5d(m, dst + h * 4096, bar, 0, T.s0, h, T.b, T.a0);
}
__device__ __forceinline__ void st_hm(const CUtensorMap* m, const unsigned char* src, const TileC& T) {
  for (int h = 0; h < H; ++h) tma::store_5d(m, src + h * 4096, 0, T.s0, h, T.b, T.a0);
}
__device__ __forceinline__ int offbf(int r, int c) { return (c >> 6) * 8192 + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2; }
__device__ __forceinline__ int offh(int r, int c) { return (c >> 5) * 4096 + r * 64 + (((((c & 31) >> 3) ^ (r >> 1)) & 3) << 4) + (c & 7) * 2; }

// smem tile <-> accumulator-layout fragments; columns c0 + 8i + 2t (+1), rows g0, g0 + 8 (buffer-relative columns)
template <int NI> __device__ __forceinline__ void sm_ld(float* v, const unsigned char* buf, int g0, int t, int c0) {
  for_frag<NI>([&](int i, int rr) {
    const float2 x = *reinterpret_cast<const float2*>(buf + offa(g0 + 8 * rr, c0 + 8 * i + 2 * t));
    v[4 * i + 2 * rr] = x.x; v[4 * i + 2 * rr + 1] = x.y;
  });
}
template <int NI> __device__ __forceinline__ void sm_st(unsigned char* buf, const float* v, int g0, int t, int c0) {
  for_frag<NI>([&](int i, int rr) {
    *reinterpret_cast<float2*>(buf + offa(g0 + 8 * rr, c0 + 8 * i + 2 * t)) = make_float2(v[4 * i + 2 * rr], v[4 * i + 2 * rr + 1]);
  });
}
template <int NI> __device__ __forceinline__ void sm_ld_bf(float* v, const unsigned char* buf, int g0, int t, int c0) {
  for_frag<NI>([&](int i, int rr) {
    const uint32_t u = *reinterpret_cast<const uint32_t*>(buf + offbf(g0 + 8 * rr, c0 + 8 * i + 2 * t));
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(&u);
    v[4 * i + 2 * rr] = __low2float(x); v[4 * i + 2 * rr + 1] = __high2float(x);
  });
}
template <int NI> __device__ __forceinline__ void sm_st_bf(unsigned char* buf, const float* v, int g0, int t, int c0) {
  for_frag<NI>([&](int i, int rr) {
    *reinterpret_cast<uint32_t*>(buf + offbf(g0 + 8 * rr, c0 + 8 * i + 2 * t)) = packb(v[4 * i + 2 * rr], v[4 * i + 2 * rr + 1]);
  });
}
__device__ __forceinline__ void sm_ld_hm(float* v, const unsigned char* buf, int g0, int t) {
  for_frag<16>([&](int i, int rr) {
    const uint32_t u = *reinterpret_cast<const uint32_t*>(buf + offh(g0 + 8 * rr, 8 * i + 2 * t));
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(&u);
    v[4 * i + 2 * rr] = __low2float(x); v[4 * i + 2 * rr + 1] = __high2float(x);
  });
}
__device__ __forceinline__ void sm_st_hm(unsigned char* buf, const float* v, int g0, int t) {
  for_frag<16>([&](int i, int rr) {
    *reinterpret_cast<uint32_t*>(buf + offh(g0 + 8 * rr, 8 * i + 2 * t)) = packb(v[4 * i + 2 * rr], v[4 * i + 2 * rr + 1]);
  });
}

// Round a TMA-loaded fp32 tile (NB 32-column blocks, this thread's fragment slots) to tf32 in place: it then serves as a
// wgmma SS operand with K in natural order.
template <int NB> __device__ __forceinline__ void round_tile(unsigned char* buf, int g0, int t) {
  for_frag<4 * NB>([&](int i, int rr) {
    float2* p = reinterpret_cast<float2*>(buf + offa(g0 + 8 * rr, 8 * i + 2 * t));
    const float2 x = *p;
    *p = make_float2(__uint_as_float(tf32r(x.x)), __uint_as_float(tf32r(x.y)));
  });
}
// write fp32 values (accumulator layout, NI blocks from column c0) rounded to tf32 into an SS tile at natural positions
template <int NI> __device__ __forceinline__ void sm_st_tf32(unsigned char* buf, const float* v, int g0, int t, int c0) {
  for_frag<NI>([&](int i, int rr) {
    *reinterpret_cast<float2*>(buf + offa(g0 + 8 * rr, c0 + 8 * i + 2 * t)) =
        make_float2(__uint_as_float(tf32r(v[4 * i + 2 * rr])), __uint_as_float(tf32r(v[4 * i + 2 * rr + 1])));
  });
}

// ---- pair exchanges without cluster barriers: st.async into the partner's shared memory, completing bytes on the
// partner's mbarrier. Buffers and barriers are double-buffered by tile parity; a CTA only writes a tile-k+2 buffer after
// it has received the partner's tile-k+1 data at the same point, which the partner sends after consuming tile k's.
namespace cl {
__device__ __forceinline__ void st_async2(uint32_t raddr, float a, float b, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n"
               :: "r"(raddr), "f"(a), "f"(b), "r"(rbar) : "memory");
}
}  // namespace cl
// Exchange of an accumulator split over K between the pair: acc[64] = this CTA's partial of a [64][128] output; own[32]
// <- the pair's sum for this CTA's column half (n8 blocks 8r ..). xb: [16][128] float2 receive buffer of this parity.
__device__ __forceinline__ void xacc(const float* acc, float* own, const float2* xb, uint64_t* xbar, uint32_t rxb, uint32_t rxbar,
                                      int r, int tid, uint32_t parity) {
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float a = r ? acc[2 * k] : acc[32 + 2 * k], b = r ? acc[2 * k + 1] : acc[32 + 2 * k + 1];
    cl::st_async2(rxb + 8 * (k * 128 + tid), a, b, rxbar);
  }
  tma::wait(xbar, parity);
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float2 p = xb[k * 128 + tid];
    own[2 * k] = (r ? acc[32 + 2 * k] : acc[2 * k]) + p.x;
    own[2 * k + 1] = (r ? acc[32 + 2 * k + 1] : acc[2 * k + 1]) + p.y;
  }
}
// Row sums over the pair's column halves: partial (s0, s1) of rows g0, g0 + 8 -> totals. xr: [64] floats receive buffer.
__device__ __forceinline__ float2 xrow(float s0, float s1, const float* xr, uint64_t* bar, uint32_t rxr, uint32_t rbar, int g0, int t,
                                        uint32_t parity) {
  if (t == 0) cl::st_async2(rxr + 4 * 2 * g0, s0, s1, rbar);                 // slot pair (2 g0, 2 g0 + 1)
  tma::wait(bar, parity);
  return make_float2(s0 + xr[2 * g0], s1 + xr[2 * g0 + 1]);
}

// d-modulation block col0 += sum over the tile's augments of v (this CTA's columns col0 + c0 + .., NI blocks, zero on
// invalid rows). SP >= 4: plain stores into dbuf [64 rows][72] (all 128 threads call it), summed per atom; else one
// red.global per value pair.
template <int NI> __device__ __forceinline__ void dmod_add(float* dmod, float* dbuf, const float* v, const Row* R, const Geo& G,
                                                           const TileC& T, int t, int tid, int g0, int col0, int c0) {
  if (G.SP < 4) {
    for_frag<NI>([&](int i, int rr) {
      if (R[rr].ok) red2(dmod + (long)R[rr].mrow * MODW + col0 + c0 + 8 * i + 2 * t, v[4 * i + 2 * rr], v[4 * i + 2 * rr + 1]);
    });
    return;
  }
  for_frag<NI>([&](int i, int rr) {
    *reinterpret_cast<float2*>(dbuf + (g0 + 8 * rr) * 72 + 8 * i + 2 * t) = make_float2(v[4 * i + 2 * rr], v[4 * i + 2 * rr + 1]);
  });
  bar1();
  constexpr int NP = 4 * NI;                               // column pairs
  for (int p = tid; p < G.AT * NP; p += TPB) {
    const int at = p / NP, cp = p % NP;
    float2 s = make_float2(0.f, 0.f);
    for (int a = 0; a < G.SP; ++a) {
      const float2 x = *reinterpret_cast<const float2*>(dbuf + (a * G.AT + at) * 72 + 2 * cp);
      s.x += x.x; s.y += x.y;
    }
    if (T.s0 + at < G.S) red2(dmod + (long)(T.b * G.S + T.s0 + at) * MODW + col0 + c0 + 2 * cp, s.x, s.y);
  }
  bar1();
}

// Row-kernel prologue: identity, pair tiling, this CTA's weight ring. `issue(gc)` goes in between the two halves.
#define ROW_IDENT(NST_, BAR_OFF_)                                                                                   \
  extern __shared__ __align__(1024) unsigned char smem[];                                                             \
  const int tid = threadIdx.x, lane = tid & 31, t = lane & 3, g0 = ((tid >> 5) << 4) + (lane >> 2);                  \
  const int r = (int)(blockIdx.x & 1), pair = (int)(blockIdx.x >> 1), npair = (int)(gridDim.x >> 1);                 \
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + (BAR_OFF_));                                                    \
  unsigned char* slots = smem;                                                                                        \
  const WRing ring{slots, full, (NST_)};                                                                              \
  const Geo& G = args.geo;                                                                                            \
  const int ntT = pair < G.ntile ? (G.ntile - pair + npair - 1) / npair : 0;                                         \
  auto tile_of = [&](int k, TileC& T) { tile_coords(G, pair + k * npair, T.b, T.a0, T.s0); };
#define ROW_START(NST_, NSW_, NBAR_)                                                                                \
  const int total = ntT * (NSW_);                                                                                     \
  if (tid == 0) {                                                                                                     \
    for (int i = 0; i < (NST_) + (NBAR_); ++i) tma::bar_init(full + i, 1);                                           \
    tma::bar_init_fence();                                                                                            \
  }                                                                                                                   \
  __syncthreads();                                                                                                    \
  if (tid == 0) for (int gc = 0; gc < (NST_) && gc < total; ++gc) issue(gc);                                         \
  auto rel = [&](int gc) { bar1(); if (tid == 0 && gc + (NST_) < total) issue(gc + (NST_)); };

// this CTA's column half (n8 blocks 8r ..) of a full-row accumulator-layout array, static register indices
__device__ __forceinline__ void own_half(const float* a, float* o, int r) {
#pragma unroll
  for (int k = 0; k < 32; ++k) o[k] = r ? a[32 + k] : a[k];
}

// ================================================================ ① qkvg forward
// rank r: projections 2r, 2r + 1 (q, k | v, g) of the tile -- independent halves, no exchange.
struct QkvgFwdArgs {
  const float *mod, *cs, *sn;
  Geo geo;
  float eps, qk_eps;
  int save;
};
constexpr int Q1_NSW = 8, Q1_NST = 6;
constexpr int Q1_QB = Q1_NST * STAGE, Q1_SP = Q1_QB + 32768, Q1_SH = Q1_SP + 32768, Q1_BAR = Q1_SH + 16384,
              Q1_BYTES = Q1_BAR + (Q1_NST + 1) * 8;

__global__ void __launch_bounds__(TPB, 1) qkvg_fwd_kernel(QkvgFwdArgs args, const __grid_constant__ CUtensorMap mw1,
                                                          const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mx,
                                                          const __grid_constant__ CUtensorMap mpq, const __grid_constant__ CUtensorMap mpk,
                                                          const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mQ,
                                                          const __grid_constant__ CUtensorMap mK, const __grid_constant__ CUtensorMap mV) {
  ROW_IDENT(Q1_NST, Q1_BAR)
  auto issue = [&](int gc) {
    const int s = gc % Q1_NSW, sl = gc % Q1_NST;
    tma::expect_tx(full + sl, STAGE);
    tma::load_2d(&mw1, slots + sl * STAGE, full + sl, (s & 3) * 32, (2 * r + (s >> 2)) * 128);
  };
  ROW_START(Q1_NST, Q1_NSW, 1)
  unsigned char *qb = smem + Q1_QB, *sp = smem + Q1_SP, *sh = smem + Q1_SH;
  uint64_t* qfull = full + Q1_NST;
  auto load_q = [&](int k) { TileC T; tile_of(k, T); tma::expect_tx(qfull, 32768); ld_rows(&mq, qb, qfull, 0, 4, T); };
  if (tid == 0 && ntT > 0) load_q(0);
  const bool xsave = r == 0 && args.save;
  for (int k = 0; k < ntT; ++k) {
    TileC T; tile_of(k, T);
    const Row R[2] = {row_of(G, T.b, T.a0, T.s0, g0), row_of(G, T.b, T.a0, T.s0, g0 + 8)};
    tma::wait(qfull, k & 1);
    float x[64];
    sm_ld<16>(x, qb, g0, t, 0);
    {
      float s0_, s1_; row_dot<16>(x, x, s0_, s1_);
      const float r0 = 1.f / sqrtf(s0_ * (1.f / C) + args.eps), r1 = 1.f / sqrtf(s1_ * (1.f / C) + args.eps);
      float m[64];
      load_mod<16>(m, args.mod, R, t, C);                  // scale_a
#pragma unroll
      for (int e = 0; e < 64; ++e) x[e] *= (((e >> 1) & 1) ? r1 : r0) * (1.f + m[e]);
      load_mod<16>(m, args.mod, R, t, 0);                  // shift_a
#pragma unroll
      for (int e = 0; e < 64; ++e) x[e] += m[e];
    }
    if (xsave) sm_st<16>(qb, x, g0, t, 0);                 // x over q, in place (each thread its own slots)
    wg::proxy_fence();
    bar1();
    if (tid == 0) {
      if (xsave) { st_rows(&mx, qb, 0, 4, T); tma::store_commit(); }
      else if (k + 1 < ntT) load_q(k + 1);
    }
    frag_inplace<16>(x);
    float2 cv[2][2], sv[2][2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
#pragma unroll
      for (int ib = 0; ib < 2; ++ib) {
        cv[rr][ib] = r == 0 ? ld2(args.cs + (long)R[rr].mrow * (D / 2) + 8 * ib + 2 * t) : make_float2(0.f, 0.f);
        sv[rr][ib] = r == 0 ? ld2(args.sn + (long)R[rr].mrow * (D / 2) + 8 * ib + 2 * t) : make_float2(0.f, 0.f);
      }
    float acc[64];
#pragma unroll 1
    for (int pp = 0; pp < 2; ++pp) {
      const int p = 2 * r + pp, gc0 = k * Q1_NSW + 4 * pp;
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        unsigned char* ws = ring.wait(gc0 + kb);
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int ks = 4 * kb + s;
          const uint64_t d = wg::desc(ws + s * 32);
          if (ks == 0) wg::tf32_rs_n128_first(U(x[0]), U(x[1]), U(x[2]), U(x[3]), d, acc);
          else wg::tf32_rs_n128(U(x[4 * ks]), U(x[4 * ks + 1]), U(x[4 * ks + 2]), U(x[4 * ks + 3]), d, acc);
        }
        wg::commit();
        if (kb > 0) { wg::wait<1>(); rel(gc0 + kb - 1); }
      }
      wg::wait<0>();
      rel(gc0 + 3);
      if (tid == 0) {                                      // staging (and, after the x store, the q buffer) free again
        tma::store_wait_read<0>();
        if (xsave && pp == 0 && k + 1 < ntT) load_q(k + 1);
      }
      bar1();
      if (p == 3) {
        sm_st<16>(sp, acc, g0, t, 0);
      } else {
        if (p < 2) {
          if (args.save) sm_st<16>(sp, acc, g0, t, 0);
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
              float ss = 0.f;
#pragma unroll
              for (int ib = 0; ib < 4; ++ib) {
                const float u = acc[4 * (4 * h + ib) + 2 * rr], v = acc[4 * (4 * h + ib) + 2 * rr + 1];
                ss += u * u + v * v;
              }
              const float rq = 1.f / sqrtf(qsum(ss) * (1.f / D) + args.qk_eps);
#pragma unroll
              for (int ib = 0; ib < 2; ++ib)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                  const int i1 = 4 * (4 * h + ib) + 2 * rr + e, i2 = 4 * (4 * h + ib + 2) + 2 * rr + e;
                  const float x1 = acc[i1] * rq, x2 = acc[i2] * rq;
                  const float c_ = e ? cv[rr][ib].y : cv[rr][ib].x, s_ = e ? sv[rr][ib].y : sv[rr][ib].x;
                  acc[i1] = x1 * c_ - x2 * s_; acc[i2] = x2 * c_ + x1 * s_;
                }
            }
        }
        sm_st_hm(sh, acc, g0, t);
      }
      wg::proxy_fence();
      bar1();
      if (tid == 0) {
        if (p == 3) st_rows(&mg, sp, 0, 4, T);
        else {
          if (p < 2 && args.save) st_rows(p == 0 ? &mpq : &mpk, sp, 0, 4, T);
          st_hm(p == 0 ? &mQ : p == 1 ? &mK : &mV, sh, T);
        }
        tma::store_commit();
      }
    }
  }
  if (tid == 0) tma::store_wait_all();
}

// ================================================================ ③ out-projection + gated residual + FFN forward
// Both CTAs of the pair build the whole att / q1 / y (att on the full Wo: 64 KB more weights, no row-statistics
// exchange); rank r then runs FFN hidden chunks j = 2c + r and owns output columns 64r..; the down-projection partials
// meet by st.async.
struct FfnFwdArgs {
  const float* mod;
  Geo geo;
  float eps;
  int save;
};
constexpr int F4_NSW = 4 + 6 * 4, F4_NST = 4;
constexpr int F4_GB = F4_NST * STAGE, F4_OB = F4_GB + 32768, F4_QB = F4_OB + 16384, F4_XB = F4_QB + 32768, F4_ST = F4_XB + 32768,
              F4_BAR = F4_ST + 16384, F4_BYTES = F4_BAR + (F4_NST + 4) * 8;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(TPB, 1)
    oproj_ffn_fwd_kernel(FfnFwdArgs args, const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwabh,
                          const __grid_constant__ CUtensorMap mwabl, const __grid_constant__ CUtensorMap mwdh,
                          const __grid_constant__ CUtensorMap mwdl, const __grid_constant__ CUtensorMap mq,
                          const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo,
                          const __grid_constant__ CUtensorMap mout, const __grid_constant__ CUtensorMap mq1,
                          const __grid_constant__ CUtensorMap matt, const __grid_constant__ CUtensorMap my,
                          const __grid_constant__ CUtensorMap mff) {
  ROW_IDENT(F4_NST, F4_BAR)
  auto issue = [&](int gc) {
    const int s = gc % F4_NSW, sl = gc % F4_NST;
    unsigned char* dst = slots + sl * STAGE;
    uint64_t* bar = full + sl;
    tma::expect_tx(bar, STAGE);
    if (s < 4) { tma::load_2d(&mwo, dst, bar, 32 * s, 0); return; }
    const int u = s - 4, c = u / 6, q = u % 6, j = 2 * c + r;
    if (q < 4) { tma::load_2d(&mwabh, dst, bar, 32 * q, 64 * j); tma::load_2d(&mwabl, dst + 8192, bar, 32 * q, 64 * j); }
    else tma::load_2d(q == 4 ? &mwdh : &mwdl, dst, bar, 32 * j, 0);
  };
  ROW_START(F4_NST, F4_NSW, 4)
  unsigned char *gb = smem + F4_GB, *ob = smem + F4_OB, *qb = smem + F4_QB, *stg = smem + F4_ST;
  const float2* xb = reinterpret_cast<const float2*>(smem + F4_XB);
  uint64_t *gfull = full + F4_NST, *qfull = full + F4_NST + 1, *xbar = full + F4_NST + 2;
  const uint32_t rxb = cl::peer(smem + F4_XB, r ^ 1), rxbar = cl::peer(xbar, r ^ 1);
  auto load_go = [&](int k) {
    TileC T; tile_of(k, T);
    tma::expect_tx(gfull, 32768 + 16384);
    ld_rows(&mg, gb, gfull, 0, 4, T);
    ld_rows_bf(&mo, ob, gfull, 0, 2, T);
  };
  auto load_q = [&](int k) { TileC T; tile_of(k, T); tma::expect_tx(qfull, 32768); ld_rows(&mq, qb, qfull, 0, 4, T); };
  cl::sync();                                              // the partner's barriers exist before any st.async reaches them
  if (tid == 0 && ntT > 0) { load_go(0); load_q(0); }
  const int cw = 64 * r;
  for (int k = 0; k < ntT; ++k) {
    TileC T; tile_of(k, T);
    const Row R[2] = {row_of(G, T.b, T.a0, T.s0, g0), row_of(G, T.b, T.a0, T.s0, g0 + 8)};
    const int gc0 = k * F4_NSW, par = k & 1;
    if (tid == 0) tma::expect_tx(xbar + par, 16384);       // this tile's incoming partial
    float acc[64];
    {                                                      // att = (sigmoid(g) * o) Wo^T, whole row
      float ga[64], ov[64];
      tma::wait(gfull, k & 1);
      sm_ld<16>(ga, gb, g0, t, 0);
      sm_ld_bf<16>(ov, ob, g0, t, 0);
      bar1();
      if (tid == 0 && k + 1 < ntT) load_go(k + 1);
#pragma unroll
      for (int e = 0; e < 64; ++e) ga[e] = sigm(ga[e]) * ov[e];
      frag_inplace<16>(ga);
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        unsigned char* ws = ring.wait(gc0 + kb);
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int ks = 4 * kb + s;
          const uint64_t d = wg::desc(ws + s * 32);
          if (ks == 0) wg::tf32_rs_n128_first(U(ga[0]), U(ga[1]), U(ga[2]), U(ga[3]), d, acc);
          else wg::tf32_rs_n128(U(ga[4 * ks]), U(ga[4 * ks + 1]), U(ga[4 * ks + 2]), U(ga[4 * ks + 3]), d, acc);
        }
        wg::commit();
        if (kb > 0) { wg::wait<1>(); rel(gc0 + kb - 1); }
      }
      wg::wait<0>();
      rel(gc0 + 3);
    }
    float y[64];
    {                                                      // q1 = q + gate_a * att (kept in qb); y = rmsnorm(q1) * (1 + scale_f) + shift_f
      float m[64];
      tma::wait(qfull, k & 1);
      sm_ld<16>(y, qb, g0, t, 0);
      load_mod<16>(m, args.mod, R, t, 2 * C);
#pragma unroll
      for (int e = 0; e < 64; ++e) y[e] += m[e] * acc[e];
      if (tid == 0) tma::store_wait_read<0>();             // the last tile's stores have read stg
      bar1();
      sm_st<16>(qb, y, g0, t, 0);                          // q1 over q (each thread its own slots)
      if (args.save) { own_half(acc, m, r); sm_st<8>(stg, m, g0, t, 0); }
      wg::proxy_fence();
      bar1();
      if (tid == 0 && args.save) { st_rows(&matt, stg, 2 * r, 2, T); st_rows(&mq1, qb + 16384 * r, 2 * r, 2, T); tma::store_commit(); }
      float s0_, s1_; row_dot<16>(y, y, s0_, s1_);
      const float r0 = 1.f / sqrtf(s0_ * (1.f / C) + args.eps), r1 = 1.f / sqrtf(s1_ * (1.f / C) + args.eps);
      load_mod<16>(m, args.mod, R, t, 4 * C);
#pragma unroll
      for (int e = 0; e < 64; ++e) y[e] *= (((e >> 1) & 1) ? r1 : r0) * (1.f + m[e]);
      load_mod<16>(m, args.mod, R, t, 3 * C);
#pragma unroll
      for (int e = 0; e < 64; ++e) y[e] += m[e];
      if (args.save) {
        own_half(y, m, r);
        if (tid == 0) tma::store_wait_read<0>();
        bar1();
        sm_st<8>(stg, m, g0, t, 0);
        wg::proxy_fence();
        bar1();
        if (tid == 0) { st_rows(&my, stg, 2 * r, 2, T); tma::store_commit(); }
      }
      frag_inplace<16>(y);
    }
#pragma unroll 1
    for (int c = 0; c < 4; ++c) {
      const int gcc = gc0 + 4 + 6 * c;
      float ab[32];
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {                     // a | b = y [Wa_j; Wb_j]^T, split tf32 weights (RS)
        unsigned char* ws = ring.wait(gcc + kb);
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int ks = 4 * kb + s;
          const uint64_t bh = wg::desc(ws + s * 32), bl = wg::desc(ws + 8192 + s * 32);
          if (ks == 0) wg::tf32_rs_n64_first(U(y[0]), U(y[1]), U(y[2]), U(y[3]), bh, ab);
          else wg::tf32_rs_n64(U(y[4 * ks]), U(y[4 * ks + 1]), U(y[4 * ks + 2]), U(y[4 * ks + 3]), bh, ab);
          wg::tf32_rs_n64(U(y[4 * ks]), U(y[4 * ks + 1]), U(y[4 * ks + 2]), U(y[4 * ks + 3]), bl, ab);
        }
        wg::commit();
        if (kb == 0 && c > 0) { wg::wait<1>(); rel(gcc - 2); rel(gcc - 1); }
        if (kb > 0) { wg::wait<1>(); rel(gcc + kb - 1); }
      }
      wg::wait<0>();
      rel(gcc + 3);
      float hv[16];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float a = ab[4 * i + e], bb = ab[16 + 4 * i + e]; hv[4 * i + e] = a * sigm(a) * bb; }
      frag_inplace<4>(hv);
      unsigned char* wh = ring.wait(gcc + 4);
      unsigned char* wl = ring.wait(gcc + 5);
      wg::fence();
#pragma unroll
      for (int s = 0; s < 4; ++s) {                        // ffn += h Wd_j^T, split tf32 weights (RS)
        const uint64_t dh = wg::desc(wh + s * 32), dl = wg::desc(wl + s * 32);
        if (c == 0 && s == 0) wg::tf32_rs_n128_first(U(hv[0]), U(hv[1]), U(hv[2]), U(hv[3]), dh, acc);
        else wg::tf32_rs_n128(U(hv[4 * s]), U(hv[4 * s + 1]), U(hv[4 * s + 2]), U(hv[4 * s + 3]), dh, acc);
        wg::tf32_rs_n128(U(hv[4 * s]), U(hv[4 * s + 1]), U(hv[4 * s + 2]), U(hv[4 * s + 3]), dl, acc);
      }
      wg::commit();
    }
    wg::wait<0>();
    rel(gc0 + F4_NSW - 2); rel(gc0 + F4_NSW - 1);
    float fo[32];
    xacc(acc, fo, xb + par * 2048, xbar + par, rxb + par * 16384, rxbar + 8 * par, r, tid, (k >> 1) & 1);
    {                                                      // out = q1 + gate_f * ffn, this column half
      float q1[32], m[32];
      sm_ld<8>(q1, qb, g0, t, cw);
      load_mod<8>(m, args.mod, R, t, 5 * C + cw);
#pragma unroll
      for (int e = 0; e < 32; ++e) q1[e] += m[e] * fo[e];
      if (tid == 0) tma::store_wait_read<0>();
      bar1();
      sm_st<8>(qb, q1, g0, t, cw);
      if (args.save) sm_st<8>(stg, fo, g0, t, 0);
      wg::proxy_fence();
      bar1();
      if (tid == 0) {
        st_rows(&mout, qb + 16384 * r, 2 * r, 2, T);
        if (args.save) st_rows(&mff, stg, 2 * r, 2, T);
        tma::store_commit();
        if (k + 1 < ntT) { tma::store_wait_read<0>(); load_q(k + 1); }
      }
    }
  }
  if (tid == 0) tma::store_wait_all();
  cl::sync();
}

// ================================================================ ④a FFN backward -> dq1
// Both CTAs of the pair build the whole dffn and y as tf32 SS tiles (dffn over the loaded dq2, y rounded in its own TMA
// tile, K natural); rank r runs hidden chunks j = 2c + r: a | b recomputed, dh = dffn Wd[:, j], h / da / db, and its
// partial dy; the dy halves meet by st.async, the row statistics of the RMSNorm backward likewise.
struct FfnBwdArgs {
  const float* mod;
  float* dmod;
  Geo geo;
  float eps;
};
constexpr int B4_NSW = 10 * 4, B4_NST = 4;
constexpr int B4_DF = B4_NST * STAGE, B4_YS = B4_DF + 32768, B4_XB = B4_YS + 32768, B4_Q1 = B4_XB + 32768, B4_ST = B4_Q1 + 16384,
              B4_XR = B4_ST + 40960, B4_BAR = B4_XR + 2048, B4_BYTES = B4_BAR + (B4_NST + 11) * 8;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(TPB, 1)
    ffn_bwd_kernel(FfnBwdArgs args, const __grid_constant__ CUtensorMap mwabh, const __grid_constant__ CUtensorMap mwabl,
                    const __grid_constant__ CUtensorMap mwdth, const __grid_constant__ CUtensorMap mwdtl,
                    const __grid_constant__ CUtensorMap mwabth, const __grid_constant__ CUtensorMap mwabtl,
                    const __grid_constant__ CUtensorMap mdq2, const __grid_constant__ CUtensorMap mffn,
                    const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mq1,
                    const __grid_constant__ CUtensorMap mdffn, const __grid_constant__ CUtensorMap mhh,
                    const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mdq1) {
  ROW_IDENT(B4_NST, B4_BAR)
  auto issue = [&](int gc) {                               // per chunk: Wab kb 0..3 (hi | lo), WdT hi, WdT lo, WabT kb0 hi, lo, kb1 hi, lo
    const int s = gc % B4_NSW, sl = gc % B4_NST, j = 2 * (s / 10) + r, q = s % 10;
    unsigned char* dst = slots + sl * STAGE;
    uint64_t* bar = full + sl;
    tma::expect_tx(bar, STAGE);
    if (q < 4) { tma::load_2d(&mwabh, dst, bar, 32 * q, 64 * j); tma::load_2d(&mwabl, dst + 8192, bar, 32 * q, 64 * j); }
    else if (q < 6) { for (int kb = 0; kb < 4; ++kb) tma::load_2d(q == 5 ? &mwdtl : &mwdth, dst + kb * 4096, bar, 32 * kb, 32 * j); }
    else tma::load_2d(((q - 6) & 1) ? &mwabtl : &mwabth, dst, bar, 64 * j + 32 * ((q - 6) >> 1), 0);
  };
  ROW_START(B4_NST, B4_NSW, 11)
  unsigned char *df = smem + B4_DF, *ys = smem + B4_YS, *q1b = smem + B4_Q1, *stg = smem + B4_ST;
  const float2* xb = reinterpret_cast<const float2*>(smem + B4_XB);
  float* dbuf = reinterpret_cast<float*>(stg + 16384);
  const float* xr = reinterpret_cast<const float*>(smem + B4_XR);
  // barriers: dq2, y, ffn (own), q1 (own), dq2 (own reload), xb[2], xr[2 points][2 parities]
  uint64_t *fdq = full + B4_NST, *fy = fdq + 1, *fff = fdq + 2, *fq1 = fdq + 3, *fdq2 = fdq + 4, *xbar = fdq + 5, *xrbar = fdq + 7;
  const uint32_t rxb = cl::peer(smem + B4_XB, r ^ 1), rxbar = cl::peer(xbar, r ^ 1);
  const uint32_t rxr = cl::peer(smem + B4_XR, r ^ 1), rxrbar = cl::peer(xrbar, r ^ 1);
  auto load_dq = [&](int k) { TileC T; tile_of(k, T); tma::expect_tx(fdq, 32768); ld_rows(&mdq2, df, fdq, 0, 4, T); };
  auto load_y = [&](int k) { TileC T; tile_of(k, T); tma::expect_tx(fy, 32768); ld_rows(&my, ys, fy, 0, 4, T); };
  auto load_ff = [&](int k) { TileC T; tile_of(k, T); tma::expect_tx(fff, 16384); ld_rows(&mffn, stg, fff, 2 * r, 2, T); };
  auto load_q1 = [&](int k) { TileC T; tile_of(k, T); tma::expect_tx(fq1, 16384); ld_rows(&mq1, q1b, fq1, 2 * r, 2, T); };
  cl::sync();
  if (tid == 0 && ntT > 0) { load_dq(0); load_y(0); load_ff(0); load_q1(0); }
  const int cw = 64 * r;
  for (int k = 0; k < ntT; ++k) {
    TileC T; tile_of(k, T);
    const Row R[2] = {row_of(G, T.b, T.a0, T.s0, g0), row_of(G, T.b, T.a0, T.s0, g0 + 8)};
    const int gc0 = k * B4_NSW, par = k & 1;
    const uint32_t xpar = (k >> 1) & 1;
    if (tid == 0) {
      tma::expect_tx(xbar + par, 16384);
      tma::expect_tx(xrbar + 2 * par, 256); tma::expect_tx(xrbar + 2 * par + 1, 256);
    }
    {                                                      // d gate_f += dq2 * ffn (own half); dffn = dq2 * gate_f -> SS + DFFN
      float v[64], m[64];
      tma::wait(fdq, k & 1);
      tma::wait(fff, k & 1);
      sm_ld<16>(v, df, g0, t, 0);
      sm_ld<8>(m, stg, g0, t, 0);
      {
        float o[32];
        own_half(v, o, r);
#pragma unroll
        for (int e = 0; e < 32; ++e) m[e] *= o[e];
      }
      dmod_add<8>(args.dmod, dbuf, m, R, G, T, t, tid, g0, 5 * C, cw);            // (bar1 inside when pre-summed)
      load_mod<16>(m, args.mod, R, t, 5 * C);
#pragma unroll
      for (int e = 0; e < 64; ++e) v[e] *= m[e];
      bar1();                                              // everyone has read stg (ffn) and df (dq2)
      own_half(v, m, r);
      sm_st<8>(stg, m, g0, t, 0);                          // DFFN staging (own half)
      sm_st_tf32<16>(df, v, g0, t, 0);                     // dffn SS tile over dq2
      tma::wait(fy, k & 1);
      round_tile<4>(ys, g0, t);                            // y as a tf32 SS tile, in place
      wg::proxy_fence();
      bar1();
      if (tid == 0) { st_rows(&mdffn, stg, 2 * r, 2, T); tma::store_commit(); }
    }
    float dy[64];
#pragma unroll 1
    for (int c = 0; c < 4; ++c) {
      const int gcc = gc0 + 10 * c, j = 2 * c + r;
      if (c > 0) {                                         // chunk c-1's dy GEMM still reads its [da | db] registers
        wg::wait<0>();
        rel(gcc - 4); rel(gcc - 3); rel(gcc - 2); rel(gcc - 1);
      }
      float ab[32], dh[16];
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {                     // a | b recomputed = y [Wa_j; Wb_j]^T (SS, split weights)
        unsigned char* ws = ring.wait(gcc + kb);
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint64_t a = wg::desc(ys + kb * 8192 + s * 32);
          const uint64_t bh = wg::desc(ws + s * 32), bl = wg::desc(ws + 8192 + s * 32);
          if (kb == 0 && s == 0) wg::tf32_ss_n64_first(a, bh, ab); else wg::tf32_ss_n64(a, bh, ab);
          wg::tf32_ss_n64(a, bl, ab);
        }
        wg::commit();
        if (kb > 0) { wg::wait<1>(); rel(gcc + kb - 1); }
      }
      {                                                    // dh = dffn Wd[:, j] (SS, split weights)
        unsigned char* wh = ring.wait(gcc + 4);
        unsigned char* wl = ring.wait(gcc + 5);
        wg::fence();
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
          const uint64_t a = wg::desc(df + (ks >> 2) * 8192 + (ks & 3) * 32);
          const uint64_t bh = wg::desc(wh + (ks >> 2) * 4096 + (ks & 3) * 32), bl = wg::desc(wl + (ks >> 2) * 4096 + (ks & 3) * 32);
          if (ks == 0) wg::tf32_ss_n32_first(a, bh, dh); else wg::tf32_ss_n32(a, bh, dh);
          wg::tf32_ss_n32(a, bl, dh);
        }
        wg::commit();
      }
      wg::wait<0>();
      rel(gcc + 3); rel(gcc + 4); rel(gcc + 5);
      float hv[16];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float a = ab[4 * i + e], bb = ab[16 + 4 * i + e], dhv = dh[4 * i + e];
          const float sa = sigm(a);
          hv[4 * i + e] = a * sa * bb;
          ab[4 * i + e] = dhv * bb * sa * (1.f + a * (1.f - sa));            // da
          ab[16 + 4 * i + e] = dhv * a * sa;                                  // db
        }
      if (tid == 0) tma::store_wait_read<0>();             // the previous staging has been read
      bar1();
      sm_st<4>(stg, hv, g0, t, 0);                         // [h | da | db] chunk boxes, 8 KB each
      sm_st<4>(stg + 8192, ab, g0, t, 0);
      sm_st<4>(stg + 16384, ab + 16, g0, t, 0);
      wg::proxy_fence();
      bar1();
      if (tid == 0) {
        tma::store_4d(&mhh, stg, 32 * j, T.s0, T.b, T.a0);
        tma::store_4d(&mdab, stg + 8192, 32 * j, T.s0, T.b, T.a0);
        tma::store_4d(&mdab, stg + 16384, NHID + 32 * j, T.s0, T.b, T.a0);
        tma::store_commit();
      }
      frag_inplace<8>(ab);
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {                     // dy += [da | db] [Wa_j; Wb_j] (RS, split weights)
        unsigned char* wh = ring.wait(gcc + 6 + 2 * kb);
        unsigned char* wl = ring.wait(gcc + 7 + 2 * kb);
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int ks = 4 * kb + s;
          const uint64_t bh = wg::desc(wh + s * 32), bl = wg::desc(wl + s * 32);
          if (c == 0 && ks == 0) wg::tf32_rs_n128_first(U(ab[0]), U(ab[1]), U(ab[2]), U(ab[3]), bh, dy);
          else wg::tf32_rs_n128(U(ab[4 * ks]), U(ab[4 * ks + 1]), U(ab[4 * ks + 2]), U(ab[4 * ks + 3]), bh, dy);
          wg::tf32_rs_n128(U(ab[4 * ks]), U(ab[4 * ks + 1]), U(ab[4 * ks + 2]), U(ab[4 * ks + 3]), bl, dy);
        }
        wg::commit();
      }
    }
    wg::wait<0>();
    rel(gc0 + 36); rel(gc0 + 37); rel(gc0 + 38); rel(gc0 + 39);
    bar1();
    if (tid == 0) {                                        // the SS tiles are free: next tile's dq2 / y; dq2 own half again
      if (k + 1 < ntT) { load_dq(k + 1); load_y(k + 1); }
      tma::store_wait_read<0>();
      tma::expect_tx(fdq2, 16384); ld_rows(&mdq2, stg, fdq2, 2 * r, 2, T);
    }
    bar1();                                                // the staging stores have read stg: dbuf is writable
    float d[32];
    xacc(dy, d, xb + par * 2048, xbar + par, rxb + par * 16384, rxbar + 8 * par, r, tid, xpar);
    {                                                      // RMSNorm/adaLN backward -> dq1; d scale_f, d shift_f
      float x[32], m[32];
      tma::wait(fq1, k & 1);
      sm_ld<8>(x, q1b, g0, t, 0);
      float s0_, s1_; row_dot<8>(x, x, s0_, s1_);
      float2 ss = xrow(s0_, s1_, xr + 128 * (2 * par), xrbar + 2 * par, rxr + 512 * (2 * par), rxrbar + 16 * par, g0, t, xpar);
      const float r0 = 1.f / sqrtf(ss.x * (1.f / C) + args.eps), r1 = 1.f / sqrtf(ss.y * (1.f / C) + args.eps);
#pragma unroll
      for (int e = 0; e < 32; ++e) { x[e] *= ((e >> 1) & 1) ? r1 : r0; m[e] = x[e] * d[e]; }
      tma::wait(fdq2, k & 1);
      dmod_add<8>(args.dmod, dbuf, m, R, G, T, t, tid, g0, 4 * C, cw);             // d scale_f
      dmod_add<8>(args.dmod, dbuf, d, R, G, T, t, tid, g0, 3 * C, cw);             // d shift_f
      load_mod<8>(m, args.mod, R, t, 4 * C + cw);
#pragma unroll
      for (int e = 0; e < 32; ++e) d[e] *= 1.f + m[e];                                // dxh
      row_dot<8>(d, x, s0_, s1_);
      ss = xrow(s0_, s1_, xr + 128 * (2 * par + 1), xrbar + 2 * par + 1, rxr + 512 * (2 * par + 1), rxrbar + 8 * (2 * par + 1),
                 g0, t, xpar);
      const float p0 = ss.x * (1.f / C), p1 = ss.y * (1.f / C);
      sm_ld<8>(m, stg, g0, t, 0);                          // dq2, own half
#pragma unroll
      for (int e = 0; e < 32; ++e) {
        const bool hi = (e >> 1) & 1;
        d[e] = m[e] + (hi ? r1 : r0) * (d[e] - x[e] * (hi ? p1 : p0));            // dq1
      }
      sm_st<8>(q1b, d, g0, t, 0);                          // each thread overwrites its own q1 slots
      wg::proxy_fence();
      bar1();
      if (tid == 0) {
        st_rows(&mdq1, q1b, 2 * r, 2, T);
        tma::store_commit();
        if (k + 1 < ntT) { tma::store_wait_read<0>(); load_ff(k + 1); load_q1(k + 1); }
      }
    }
  }
  if (tid == 0) tma::store_wait_all();
  cl::sync();
}

// ================================================================ ④b out-projection backward
// rank r: output columns 64r..; datt is built whole in registers (the RS operand of dgated = datt Wo[:, own half]); the
// WoT rows 64r.. stay resident (32 KB). No exchange, so no cluster.
struct OprojBwdArgs {
  const float* mod;
  float *dmod, *dv;
  Geo geo;
};
constexpr int O4_W = 0, O4_D1 = 32768, O4_AT = O4_D1 + 32768, O4_G = O4_AT + 16384, O4_O = O4_G + 16384, O4_DB = O4_O + 8192,
              O4_BAR = O4_DB + 64 * 72 * 4, O4_BYTES = O4_BAR + 2 * 8;

__global__ void __launch_bounds__(TPB, 1)
    oproj_bwd_kernel(OprojBwdArgs args, const __grid_constant__ CUtensorMap mwot, const __grid_constant__ CUtensorMap mdq1,
                      const __grid_constant__ CUtensorMap matt, const __grid_constant__ CUtensorMap mg,
                      const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mdatt,
                      const __grid_constant__ CUtensorMap mgated, const __grid_constant__ CUtensorMap mdg,
                      const __grid_constant__ CUtensorMap mdout) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, t = lane & 3, g0 = ((tid >> 5) << 4) + (lane >> 2);
  const int r = (int)(blockIdx.x & 1), pair = (int)(blockIdx.x >> 1), npair = (int)(gridDim.x >> 1);
  const Geo& G = args.geo;
  const int ntT = pair < G.ntile ? (G.ntile - pair + npair - 1) / npair : 0;
  auto tile_of = [&](int k, TileC& T) { tile_coords(G, pair + k * npair, T.b, T.a0, T.s0); };
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + O4_BAR);
  uint64_t *wfull = bars, *ifull = bars + 1;
  unsigned char *wb = smem + O4_W, *d1b = smem + O4_D1, *atb = smem + O4_AT, *gb = smem + O4_G, *obf = smem + O4_O;
  float* dbuf = reinterpret_cast<float*>(smem + O4_DB);
  auto load_i = [&](int k) {
    TileC T; tile_of(k, T); tma::expect_tx(ifull, 32768 + 16384 * 2 + 8192);
    ld_rows(&mdq1, d1b, ifull, 0, 4, T); ld_rows(&matt, atb, ifull, 2 * r, 2, T);
    ld_rows(&mg, gb, ifull, 2 * r, 2, T); ld_rows_bf(&mo, obf, ifull, r, 1, T);
  };
  if (tid == 0) { tma::bar_init(wfull, 1); tma::bar_init(ifull, 1); tma::bar_init_fence(); }
  __syncthreads();
  if (tid == 0) {
    tma::expect_tx(wfull, 32768);
    for (int kb = 0; kb < 4; ++kb) tma::load_2d(&mwot, wb + kb * 8192, wfull, 32 * kb, 64 * r);
    if (ntT > 0) load_i(0);
  }
  tma::wait(wfull, 0);
  const int cw = 64 * r;
  for (int k = 0; k < ntT; ++k) {
    TileC T; tile_of(k, T);
    const Row R[2] = {row_of(G, T.b, T.a0, T.s0, g0), row_of(G, T.b, T.a0, T.s0, g0 + 8)};
    float d[64], m[64];
    tma::wait(ifull, k & 1);
    sm_ld<16>(d, d1b, g0, t, 0);                           // dq1 (whole row)
    {
      float o[32];
      own_half(d, o, r);
      sm_ld<8>(m, atb, g0, t, 0);
#pragma unroll
      for (int e = 0; e < 32; ++e) m[e] *= o[e];
    }
    dmod_add<8>(args.dmod, dbuf, m, R, G, T, t, tid, g0, 2 * C, cw);               // d gate_a
    load_mod<16>(m, args.mod, R, t, 2 * C);
#pragma unroll
    for (int e = 0; e < 64; ++e) d[e] *= m[e];                                        // datt (whole row)
    own_half(d, m, r);
    bar1();                                                // everyone has read d1b
    sm_st<8>(d1b, m, g0, t, cw);                           // datt staging over dq1 (own half, own slots)
    frag_inplace<16>(d);
    float acc[32];
    wg::fence();
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {                      // dgated[:, own half] = datt Wo[:, own half]
      const uint64_t bd = wg::desc(wb + (ks >> 2) * 8192 + (ks & 3) * 32);
      if (ks == 0) wg::tf32_rs_n64_first(U(d[0]), U(d[1]), U(d[2]), U(d[3]), bd, acc);
      else wg::tf32_rs_n64(U(d[4 * ks]), U(d[4 * ks + 1]), U(d[4 * ks + 2]), U(d[4 * ks + 3]), bd, acc);
    }
    wg::commit();
    wg::wait<0>();
    float gv[32], ov[32];
    sm_ld<8>(gv, gb, g0, t, 0);
    sm_ld_bf<8>(ov, obf, g0, t, 0);
    float dvs[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    for_frag<8>([&](int i, int rr) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int kk = 4 * i + 2 * rr + e;
        const float so = sigm(gv[kk]), dgt = acc[kk], o_ = ov[kk];
        const float dov = __bfloat162float(__float2bfloat16_rn(dgt * so));
        gv[kk] = so * o_;                                  // gated
        m[kk] = dgt * o_ * so * (1.f - so);                // dG
        ov[kk] = dov;                                      // dO (bf16 value)
        dvs[rr][i >> 2] += dov * o_;
      }
    });
    bar1();                                                // atb / gb / obf read by everyone
    sm_st<8>(gb, gv, g0, t, 0);
    sm_st<8>(atb, m, g0, t, 0);
    sm_st_bf<8>(obf, ov, g0, t, 0);
    wg::proxy_fence();
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
#pragma unroll
      for (int hl = 0; hl < 2; ++hl) {
        const float s = qsum(dvs[rr][hl]);
        if (t == 0 && R[rr].ok) args.dv[((long)R[rr].n * H + 2 * r + hl) * G.S + R[rr].s] = s;
      }
    bar1();
    if (tid == 0) {
      st_rows(&mdatt, d1b + 16384 * r, 2 * r, 2, T);
      st_rows(&mgated, gb, 2 * r, 2, T); st_rows(&mdg, atb, 2 * r, 2, T); st_rows_bf(&mdout, obf, r, 1, T);
      tma::store_commit();
      if (k + 1 < ntT) { tma::store_wait_read<0>(); load_i(k + 1); }
    }
  }
  if (tid == 0) tma::store_wait_all();
}

// ================================================================ ④c qkvg backward
// rank r: pieces 2r, 2r + 1 of dP (q, k | v, g); the dx partials meet by st.async, then the RMSNorm/adaLN backward on
// this CTA's column half (row statistics exchanged the same way).
struct QkvgBwdArgs {
  const float *mod, *cs, *sn;
  float* dmod;
  Geo geo;
  float eps, qk_eps;
};
constexpr int Q4_NSW = 8, Q4_NST = 4;
constexpr int Q4_A = Q4_NST * STAGE, Q4_B = Q4_A + 16384, Q4_P = Q4_B + 16384, Q4_P2 = Q4_P + 32768, Q4_QB = Q4_P2 + 32768,
              Q4_D1 = Q4_QB + 16384, Q4_XB = Q4_D1 + 16384, Q4_XR = Q4_XB + 32768, Q4_BAR = Q4_XR + 2048,
              Q4_BYTES = Q4_BAR + (Q4_NST + 9) * 8;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(TPB, 1)
    qkvg_bwd_kernel(QkvgBwdArgs args, const __grid_constant__ CUtensorMap mw1t, const __grid_constant__ CUtensorMap mdQ,
                     const __grid_constant__ CUtensorMap mdK, const __grid_constant__ CUtensorMap mdV,
                     const __grid_constant__ CUtensorMap mpq, const __grid_constant__ CUtensorMap mpk,
                     const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mq,
                     const __grid_constant__ CUtensorMap mdq1, const __grid_constant__ CUtensorMap mdp,
                     const __grid_constant__ CUtensorMap mdq) {
  ROW_IDENT(Q4_NST, Q4_BAR)
  auto issue = [&](int gc) {
    const int s = gc % Q4_NSW, sl = gc % Q4_NST;
    tma::expect_tx(full + sl, STAGE);
    tma::load_2d(&mw1t, slots + sl * STAGE, full + sl, 128 * (2 * r + (s >> 2)) + 32 * (s & 3), 0);
  };
  ROW_START(Q4_NST, Q4_NSW, 9)
  // rank 0: A = dQ (hm), B = dK (hm), P = pq, P2 = pk;  rank 1: A = dV (hm), P = dG, P2 = dP_v staging
  unsigned char *ab_ = smem + Q4_A, *bb_ = smem + Q4_B, *pb = smem + Q4_P, *p2b = smem + Q4_P2, *qb = smem + Q4_QB, *d1b = smem + Q4_D1;
  const float2* xb = reinterpret_cast<const float2*>(smem + Q4_XB);
  const float* xr = reinterpret_cast<const float*>(smem + Q4_XR);
  uint64_t *f0 = full + Q4_NST, *f1 = f0 + 1, *f2 = f0 + 2, *xbar = f0 + 3, *xrbar = f0 + 5;
  const uint32_t rxb = cl::peer(smem + Q4_XB, r ^ 1), rxbar = cl::peer(xbar, r ^ 1);
  const uint32_t rxr = cl::peer(smem + Q4_XR, r ^ 1), rxrbar = cl::peer(xrbar, r ^ 1);
  auto load_p0 = [&](int k) {                              // piece 2r's inputs
    TileC T; tile_of(k, T);
    if (r == 0) { tma::expect_tx(f0, 16384 + 32768); ld_hm(&mdQ, ab_, f0, T); ld_rows(&mpq, pb, f0, 0, 4, T); }
    else { tma::expect_tx(f0, 16384); ld_hm(&mdV, ab_, f0, T); }
  };
  auto load_p1 = [&](int k) {                              // piece 2r + 1's inputs
    TileC T; tile_of(k, T);
    if (r == 0) { tma::expect_tx(f1, 16384 + 32768); ld_hm(&mdK, bb_, f1, T); ld_rows(&mpk, p2b, f1, 0, 4, T); }
    else { tma::expect_tx(f1, 32768); ld_rows(&mdg, pb, f1, 0, 4, T); }
  };
  auto load_q = [&](int k) {
    TileC T; tile_of(k, T); tma::expect_tx(f2, 32768);
    ld_rows(&mq, qb, f2, 2 * r, 2, T); ld_rows(&mdq1, d1b, f2, 2 * r, 2, T);
  };
  cl::sync();
  if (tid == 0 && ntT > 0) { load_p0(0); load_p1(0); load_q(0); }
  const int cw = 64 * r;
  for (int k = 0; k < ntT; ++k) {
    TileC T; tile_of(k, T);
    const Row R[2] = {row_of(G, T.b, T.a0, T.s0, g0), row_of(G, T.b, T.a0, T.s0, g0 + 8)};
    const int par = k & 1;
    const uint32_t xpar = (k >> 1) & 1;
    if (tid == 0) {
      tma::expect_tx(xbar + par, 16384);
      tma::expect_tx(xrbar + 2 * par, 256); tma::expect_tx(xrbar + 2 * par + 1, 256);
    }
    float2 cv[2][2], sv[2][2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
#pragma unroll
      for (int ib2 = 0; ib2 < 2; ++ib2) {
        cv[rr][ib2] = r == 0 ? ld2(args.cs + (long)R[rr].mrow * (D / 2) + 8 * ib2 + 2 * t) : make_float2(0.f, 0.f);
        sv[rr][ib2] = r == 0 ? ld2(args.sn + (long)R[rr].mrow * (D / 2) + 8 * ib2 + 2 * t) : make_float2(0.f, 0.f);
      }
    float acc[64], v[64];
#pragma unroll 1
    for (int pp = 0; pp < 2; ++pp) {
      const int p = 2 * r + pp, gc0 = k * Q4_NSW + 4 * pp;
      tma::wait(pp == 0 ? f0 : f1, k & 1);
      unsigned char* pbuf = r == 0 ? (pp == 0 ? pb : p2b) : (pp == 0 ? p2b : pb);   // where dP_p is staged
      if (p < 3) sm_ld_hm(v, pp == 0 ? ab_ : bb_, g0, t);
      else sm_ld<16>(v, pb, g0, t, 0);
      if (p < 2) {                                         // unrope, then the per-head RMSNorm backward on the saved projection
        float pv[64];
        sm_ld<16>(pv, pbuf, g0, t, 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
#pragma unroll
            for (int ib2 = 0; ib2 < 2; ++ib2)
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int i1 = 4 * (4 * h + ib2) + 2 * rr + e, i2 = 4 * (4 * h + ib2 + 2) + 2 * rr + e;
                const float c_ = e ? cv[rr][ib2].y : cv[rr][ib2].x, s_ = e ? sv[rr][ib2].y : sv[rr][ib2].x;
                const float y1 = v[i1], y2 = v[i2];
                v[i1] = y1 * c_ + y2 * s_; v[i2] = y2 * c_ - y1 * s_;
              }
            float ss = 0.f;
#pragma unroll
            for (int ib2 = 0; ib2 < 4; ++ib2)
#pragma unroll
              for (int e = 0; e < 2; ++e) { const float u = pv[4 * (4 * h + ib2) + 2 * rr + e]; ss += u * u; }
            const float rq = 1.f / sqrtf(qsum(ss) * (1.f / D) + args.qk_eps);
            float dot = 0.f;
#pragma unroll
            for (int ib2 = 0; ib2 < 4; ++ib2)
#pragma unroll
              for (int e = 0; e < 2; ++e) { const int kk = 4 * (4 * h + ib2) + 2 * rr + e; dot += v[kk] * pv[kk] * rq; }
            dot = qsum(dot) * (1.f / D);
#pragma unroll
            for (int ib2 = 0; ib2 < 4; ++ib2)
#pragma unroll
              for (int e = 0; e < 2; ++e) { const int kk = 4 * (4 * h + ib2) + 2 * rr + e; v[kk] = rq * (v[kk] - pv[kk] * rq * dot); }
          }
      }
      if (p == 2 && tid == 0) tma::store_wait_read<0>();   // p2b (dP_v staging) read by the last tile's store
      bar1();                                              // (everyone has read this piece's inputs)
      sm_st<16>(pbuf, v, g0, t, 0);                        // dP_p staging (in place for q, k, g)
      wg::proxy_fence();
      bar1();
      if (tid == 0) {
        st_rows(&mdp, pbuf, 4 * p, 4, T);
        tma::store_commit();
      }
      frag_inplace<16>(v);
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        unsigned char* ws = ring.wait(gc0 + kb);
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int ks = 4 * kb + s;
          const uint64_t d = wg::desc(ws + s * 32);
          if (pp == 0 && ks == 0) wg::tf32_rs_n128_first(U(v[0]), U(v[1]), U(v[2]), U(v[3]), d, acc);
          else wg::tf32_rs_n128(U(v[4 * ks]), U(v[4 * ks + 1]), U(v[4 * ks + 2]), U(v[4 * ks + 3]), d, acc);
        }
        wg::commit();
        if (kb > 0) { wg::wait<1>(); rel(gc0 + kb - 1); }
      }
      wg::wait<0>();
      rel(gc0 + 3);
    }
    if (tid == 0) { tma::store_wait_read<0>(); if (k + 1 < ntT) load_p0(k + 1); }
    bar1();                                                // the dP stores have read their staging
    float dx[32];
    xacc(acc, dx, xb + par * 2048, xbar + par, rxb + par * 16384, rxbar + 8 * par, r, tid, xpar);
    {                                                      // d scale_a, d shift_a, RMSNorm/adaLN backward -> dq
      float x[32], m[32];
      float* dbuf = reinterpret_cast<float*>(r == 0 ? p2b : pb);   // the second piece's staging (refilled at tile end)
      tma::wait(f2, k & 1);
      sm_ld<8>(x, qb, g0, t, 0);
      float s0_, s1_; row_dot<8>(x, x, s0_, s1_);
      float2 ss = xrow(s0_, s1_, xr + 128 * (2 * par), xrbar + 2 * par, rxr + 512 * (2 * par), rxrbar + 16 * par, g0, t, xpar);
      const float r0 = 1.f / sqrtf(ss.x * (1.f / C) + args.eps), r1 = 1.f / sqrtf(ss.y * (1.f / C) + args.eps);
#pragma unroll
      for (int e = 0; e < 32; ++e) { x[e] *= ((e >> 1) & 1) ? r1 : r0; m[e] = x[e] * dx[e]; }
      dmod_add<8>(args.dmod, dbuf, m, R, G, T, t, tid, g0, C, cw);                 // d scale_a
      dmod_add<8>(args.dmod, dbuf, dx, R, G, T, t, tid, g0, 0, cw);                // d shift_a
      load_mod<8>(m, args.mod, R, t, C + cw);
#pragma unroll
      for (int e = 0; e < 32; ++e) dx[e] *= 1.f + m[e];
      row_dot<8>(dx, x, s0_, s1_);
      ss = xrow(s0_, s1_, xr + 128 * (2 * par + 1), xrbar + 2 * par + 1, rxr + 512 * (2 * par + 1), rxrbar + 8 * (2 * par + 1),
                 g0, t, xpar);
      const float p0 = ss.x * (1.f / C), p1 = ss.y * (1.f / C);
      sm_ld<8>(m, d1b, g0, t, 0);
#pragma unroll
      for (int e = 0; e < 32; ++e) {
        const bool hi = (e >> 1) & 1;
        dx[e] = m[e] + (hi ? r1 : r0) * (dx[e] - x[e] * (hi ? p1 : p0));
      }
      sm_st<8>(d1b, dx, g0, t, 0);
      wg::proxy_fence();
      bar1();
      if (tid == 0) {
        st_rows(&mdq, d1b, 2 * r, 2, T);
        tma::store_commit();
        if (k + 1 < ntT) { tma::store_wait_read<0>(); load_q(k + 1); load_p1(k + 1); }
      }
    }
  }
  if (tid == 0) tma::store_wait_all();
  cl::sync();
}
// ================================================================ ② window attention (bf16 mma.sync m16n8k16, fp32 softmax)
// One CTA per (64-position tile t, n * H + h). Shared memory holds the 192 positions [64t - 64, 64t + 128) of Q, K, V
// (and dO in the backward), bf16 rows of 64 B with a 64 B swizzle (16 B chunk ^= (row / 2) % 4) so ldmatrix is
// conflict-free; out-of-range positions are zero-filled. A position p is allowed for query i iff |i - p| <= 64,
// p < seqused and i < seqused.
constexpr int AR = 192;                                     // positions staged per tile
__device__ __forceinline__ int offq(int r, int c) { return r * 64 + ((((c >> 3) ^ (r >> 1)) & 3) << 4) + (c & 7) * 2; }

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  const uint32_t d = tma::sa(dst);
  const int n = ok ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tma::sa(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tma::sa(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// A fragments (16 rows x 32 cols = 2 k16 steps) of rows r0.. of a staged [AR][32] tile
__device__ __forceinline__ void lda(uint32_t (*a)[4], const unsigned char* base, int r0, int lane) {
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) ldsm4(a[kk], base + offq(r0 + (lane & 15), 16 * kk + 8 * (lane >> 4)));
}
// S[16 x 16n] += A (16 x 32) . X[p0 .. p0 + 16n)^T : B operand from rows of a staged tile (non-transposed ldmatrix)
template <int NB> __device__ __forceinline__ void mma_rowsT(float (*s)[4], const uint32_t (*a)[4], const unsigned char* base, int p0, int lane) {
#pragma unroll
  for (int nb = 0; nb < NB; nb += 2)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t bq[4];
      ldsm4(bq, base + offq(p0 + 8 * nb + (lane & 7) + ((lane >> 4) << 3), 16 * kk + 8 * ((lane >> 3) & 1)));
      mma16816(s[nb], a[kk], bq[0], bq[1]);
      mma16816(s[nb + 1], a[kk], bq[2], bq[3]);
    }
}
// O[16 x 32] += P (16 x 16*KS, bf16 A fragments pa[ks]) . X[p0 .. p0 + 16KS) : B operand via transposed ldmatrix
template <int KS> __device__ __forceinline__ void mma_rows(float (*o)[4], const uint32_t (*pa)[4], const unsigned char* base, int p0, int lane) {
#pragma unroll
  for (int ks = 0; ks < KS; ++ks)
#pragma unroll
    for (int nb = 0; nb < 4; nb += 2) {
      uint32_t bv[4];
      ldsm4t(bv, base + offq(p0 + 16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), 8 * nb + 8 * (lane >> 4)));
      mma16816(o[nb], pa[ks], bv[0], bv[1]);
      mma16816(o[nb + 1], pa[ks], bv[2], bv[3]);
    }
}
__device__ __forceinline__ void stage_rows(unsigned char* dst, const __nv_bfloat16* src, long stride, int p0, int S, int tid, int nthr) {
  for (int e = tid; e < AR * 4; e += nthr) {               // AR rows x 4 chunks of 16 B
    const int r = e >> 2, ch = e & 3, p = p0 + r;
    const bool ok = p >= 0 && p < S;
    cp16(dst + offq(r, 8 * ch), src + (long)(ok ? p : 0) * stride + 8 * ch, ok);
  }
}

struct AttnArgs {
  const __nv_bfloat16 *Q, *K, *V;                           // head-major [N, H, S, D]
  const int* sequ;
  const __nv_bfloat16* dout;                                // [N * S, C] (backward)
  const float *lse_in, *dvv;                                // [N, H, S] (backward)
  __nv_bfloat16* O;                                         // [N * S, C]
  float* lse;
  __nv_bfloat16 *dQ, *dK, *dV;
  int S;
  float scale;
};

__global__ void __launch_bounds__(128) attn_fwd_kernel(AttnArgs args) {
  __shared__ __align__(128) unsigned char sq[64 * 64], sk[AR * 64], sv[AR * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = blockIdx.x, nh = blockIdx.y, n = nh / H, h = nh % H;
  const int S = args.S, su = args.sequ[n];
  const int i0 = 64 * tile, k0 = i0 - HWIN;
  const long hb = (long)nh * S * D;
  for (int e = tid; e < 64 * 4; e += 128) {
    const int r = e >> 2, ch = e & 3, p = i0 + r;
    const bool ok = p < su;                                // padding queries score 0, as the Triton kernel loads them
    cp16(sq + offq(r, 8 * ch), args.Q + hb + (long)(ok ? p : 0) * D + 8 * ch, ok);
  }
  stage_rows(sk, args.K + hb, D, k0, S, tid, 128);
  stage_rows(sv, args.V + hb, D, k0, S, tid, 128);
  cp_wait_all();
  __syncthreads();
  // warp w: queries i0 + 16w + (g, g + 8); keys k0 + [16w, 16w + 144)
  uint32_t qa[2][4];
  lda(qa, sq, 16 * warp, lane);
  constexpr int NB = 18;
  float s[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
  mma_rowsT<NB>(s, qa, sk, 16 * warp, lane);
  const float sl2 = args.scale * 1.4426950408889634f;
  float mx[2] = {-INFINITY, -INFINITY};
  const int qi[2] = {i0 + 16 * warp + g, i0 + 16 * warp + g + 8};
#pragma unroll
  for (int nb = 0; nb < NB; ++nb)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int rr = e >> 1, key = k0 + 16 * warp + 8 * nb + 2 * t + (e & 1), dd = qi[rr] - key;
      const bool ok = key >= 0 && key < su && dd <= HWIN && dd >= -HWIN;
      s[nb][e] = ok ? s[nb][e] * sl2 : -INFINITY;
      mx[rr] = fmaxf(mx[rr], s[nb][e]);
    }
  float l[2] = {0.f, 0.f};
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffffu, mx[rr], 1));
    mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffffu, mx[rr], 2));
  }
  uint32_t pa[NB / 2][4];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    float pv[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int rr = e >> 1;
      pv[e] = mx[rr] == -INFINITY ? 0.f : exp2f(s[nb][e] - mx[rr]);
      l[rr] += pv[e];
    }
    pa[nb >> 1][(nb & 1) * 2] = packb(pv[0], pv[1]);
    pa[nb >> 1][(nb & 1) * 2 + 1] = packb(pv[2], pv[3]);
  }
  float o[4][4];
#pragma unroll
  for (int nb = 0; nb < 4; ++nb) o[nb][0] = o[nb][1] = o[nb][2] = o[nb][3] = 0.f;
  mma_rows<NB / 2>(o, pa, sv, 16 * warp, lane);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) l[rr] = qsum(l[rr]);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = qi[rr];
    if (i >= S) continue;
    const bool live = i < su && l[rr] > 0.f;
    const float inv = live ? 1.f / l[rr] : 0.f;
#pragma unroll
    for (int nb = 0; nb < 4; ++nb)
      stb2(args.O + ((long)n * S + i) * C + h * D + 8 * nb + 2 * t, o[nb][2 * rr] * inv, o[nb][2 * rr + 1] * inv);
    if (t == 0) args.lse[(long)nh * S + i] = l[rr] > 0.f ? mx[rr] * 0.6931471805599453f + logf(l[rr]) : 0.f;
  }
}

// Backward. Warps 0-3: dQ for queries 64t + 16w + (g, g+8) over keys 64t - 64 + [16w, 16w + 144), 48 keys at a time.
// Warps 4-7: dK, dV for keys 64t + 16w' + (g, g+8) over queries 64t - 64 + [16w', 16w' + 144).
constexpr int AB_BYTES = 4 * AR * 64 + 2 * AR * 4;

__global__ void __launch_bounds__(256) attn_bwd_kernel(AttnArgs args) {
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char *sq = smem, *sk = smem + AR * 64, *sv = smem + 2 * AR * 64, *sdo = smem + 3 * AR * 64;
  float* slse = reinterpret_cast<float*>(smem + 4 * AR * 64);
  float* sdv = slse + AR;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = blockIdx.x, nh = blockIdx.y, n = nh / H, h = nh % H;
  const int S = args.S, su = args.sequ[n];
  const int p0 = 64 * tile - HWIN;
  const long hb = (long)nh * S * D;
  stage_rows(sq, args.Q + hb, D, p0, S, tid, 256);
  stage_rows(sk, args.K + hb, D, p0, S, tid, 256);
  stage_rows(sv, args.V + hb, D, p0, S, tid, 256);
  stage_rows(sdo, args.dout + (long)n * S * C + h * D, C, p0, S, tid, 256);
  for (int r = tid; r < AR; r += 256) {
    const int p = p0 + r;
    const bool ok = p >= 0 && p < S && p < su;
    slse[r] = ok ? args.lse_in[(long)nh * S + p] : 0.f;
    sdv[r] = ok ? args.dvv[(long)nh * S + p] : 0.f;
  }
  cp_wait_all();
  __syncthreads();
  const float sl2 = args.scale * 1.4426950408889634f;
  constexpr float L2E = 1.4426950408889634f;
  const int w = warp & 3;
  if (warp < 4) {
    // ---------------- dQ
    const int qr0 = 64 + 16 * w;                           // staged row of the warp's first query
    const int qi[2] = {p0 + qr0 + g, p0 + qr0 + g + 8};
    uint32_t qa[2][4], da[2][4];
    lda(qa, sq, qr0, lane);
    lda(da, sdo, qr0, lane);
    float lse2[2], dvr[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) { lse2[rr] = slse[qr0 + g + 8 * rr] * L2E; dvr[rr] = sdv[qr0 + g + 8 * rr]; }
    float dq[4][4];
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) dq[nb][0] = dq[nb][1] = dq[nb][2] = dq[nb][3] = 0.f;
#pragma unroll 1
    for (int kc = 0; kc < 3; ++kc) {
      const int kr0 = 16 * w + 48 * kc;                    // staged row of the chunk's first key
      float s[6][4], dp[6][4];
#pragma unroll
      for (int nb = 0; nb < 6; ++nb)
#pragma unroll
        for (int e = 0; e < 4; ++e) { s[nb][e] = 0.f; dp[nb][e] = 0.f; }
      mma_rowsT<6>(s, qa, sk, kr0, lane);
      mma_rowsT<6>(dp, da, sv, kr0, lane);
      uint32_t pa[3][4];
#pragma unroll
      for (int nb = 0; nb < 6; ++nb) {
        float ds[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int rr = e >> 1, key = p0 + kr0 + 8 * nb + 2 * t + (e & 1), dd = qi[rr] - key;
          const bool ok = key >= 0 && key < su && qi[rr] < su && dd <= HWIN && dd >= -HWIN;
          const float p = ok ? exp2f(s[nb][e] * sl2 - lse2[rr]) : 0.f;
          ds[e] = p * (dp[nb][e] - dvr[rr]);
        }
        pa[nb >> 1][(nb & 1) * 2] = packb(ds[0], ds[1]);
        pa[nb >> 1][(nb & 1) * 2 + 1] = packb(ds[2], ds[3]);
      }
      mma_rows<3>(dq, pa, sk, kr0, lane);
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int i = qi[rr];
      if (i >= S) continue;
#pragma unroll
      for (int nb = 0; nb < 4; ++nb)
        stb2(args.dQ + hb + (long)i * D + 8 * nb + 2 * t, dq[nb][2 * rr] * args.scale, dq[nb][2 * rr + 1] * args.scale);
    }
  } else {
    // ---------------- dK, dV
    const int kr0 = 64 + 16 * w;
    const int kj[2] = {p0 + kr0 + g, p0 + kr0 + g + 8};
    uint32_t ka[2][4], va[2][4];
    lda(ka, sk, kr0, lane);
    lda(va, sv, kr0, lane);
    float dk[4][4], dv[4][4];
#pragma unroll
    for (int nb = 0; nb < 4; ++nb)
#pragma unroll
      for (int e = 0; e < 4; ++e) { dk[nb][e] = 0.f; dv[nb][e] = 0.f; }
#pragma unroll 1
    for (int qc = 0; qc < 3; ++qc) {
      const int qr0 = 16 * w + 48 * qc;
      float s[6][4], dp[6][4];
#pragma unroll
      for (int nb = 0; nb < 6; ++nb)
#pragma unroll
        for (int e = 0; e < 4; ++e) { s[nb][e] = 0.f; dp[nb][e] = 0.f; }
      mma_rowsT<6>(s, ka, sq, qr0, lane);                  // S^T = K Q^T
      mma_rowsT<6>(dp, va, sdo, qr0, lane);                // dP^T = V dO^T
      uint32_t pa[3][4], dsa[3][4];
#pragma unroll
      for (int nb = 0; nb < 6; ++nb) {
        float p[4], ds[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int rr = e >> 1, qr = qr0 + 8 * nb + 2 * t + (e & 1), qi = p0 + qr, dd = qi - kj[rr];
          const bool ok = qi >= 0 && qi < su && kj[rr] < su && dd <= HWIN && dd >= -HWIN;
          p[e] = ok ? exp2f(s[nb][e] * sl2 - slse[qr] * L2E) : 0.f;
          ds[e] = p[e] * (dp[nb][e] - sdv[qr]);
        }
        pa[nb >> 1][(nb & 1) * 2] = packb(p[0], p[1]);
        pa[nb >> 1][(nb & 1) * 2 + 1] = packb(p[2], p[3]);
        dsa[nb >> 1][(nb & 1) * 2] = packb(ds[0], ds[1]);
        dsa[nb >> 1][(nb & 1) * 2 + 1] = packb(ds[2], ds[3]);
      }
      mma_rows<3>(dv, pa, sdo, qr0, lane);                 // dV += P^T dO
      mma_rows<3>(dk, dsa, sq, qr0, lane);                 // dK += dS^T Q
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int j = kj[rr];
      if (j >= S) continue;
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        stb2(args.dK + hb + (long)j * D + 8 * nb + 2 * t, dk[nb][2 * rr] * args.scale, dk[nb][2 * rr + 1] * args.scale);
        stb2(args.dV + hb + (long)j * D + 8 * nb + 2 * t, dv[nb][2 * rr], dv[nb][2 * rr + 1]);
      }
    }
  }
}

// ================================================================ host
// ================================================================ host
PFN_cuTensorMapEncodeTiled tma_encode() {
  static PFN_cuTensorMapEncodeTiled fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr; cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(p);
  }
  return fn;
}
// fp32 [rows][cols] -> box (32 fp32 = 128 B, brows rows), 128 B swizzle
CUtensorMap map_w(const float* ptr, uint64_t rows, uint64_t cols, uint32_t brows) {
  alignas(64) CUtensorMap m{};
  uint64_t gdim[2] = {cols, rows}; uint64_t gstride[1] = {cols * 4};
  uint32_t bdim[2] = {32, brows}; uint32_t estride[2] = {1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<float*>(ptr), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
int num_sms(int dev) { static int s = 0; if (s == 0) cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }

Geo make_geo(long A, long B, long S, int SP) {
  Geo g; g.S = (int)S; g.A = (int)A; g.B = (int)B; g.SP = SP; g.AT = 64 / SP;
  g.nab = (int)((S + g.AT - 1) / g.AT); g.nag = (int)((A + SP - 1) / SP); g.ntile = g.nab * g.nag * (int)B;
  return g;
}
int bwd_sp(long A) { return A >= 8 ? 8 : A >= 4 ? 4 : A >= 2 ? 2 : 1; }

void check_rows(const torch::Tensor& t, long M, long width, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat && t.is_contiguous() && t.numel() == M * width, name, ": fp32 [",
              M, ", ", width, "] contiguous");
}
template <class K> void smem_attr(K kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
const __nv_bfloat16* bfp(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
__nv_bfloat16* bfp_mut(torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }

// row tensors as [A][B][S][W] (row n * S + s, n = a * B + b): fp32 boxes {32, AT, 1, SP}, bf16 boxes {64, AT, 1, SP}
CUtensorMap map_rows(const torch::Tensor& t, const Geo& g, long W) {
  alignas(64) CUtensorMap m{};
  TORCH_CHECK(t.scalar_type() == torch::kFloat && t.is_contiguous() && t.numel() == (long)g.A * g.B * g.S * W, "row tensor map: fp32 [N*S, ", W, "]");
  uint64_t gdim[4] = {(uint64_t)W, (uint64_t)g.S, (uint64_t)g.B, (uint64_t)g.A};
  uint64_t gstride[3] = {(uint64_t)W * 4, (uint64_t)g.S * W * 4, (uint64_t)g.B * g.S * W * 4};
  uint32_t bdim[4] = {32, (uint32_t)g.AT, 1, (uint32_t)g.SP}; uint32_t estride[4] = {1, 1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, t.data_ptr(), gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(rows) failed: ", (int)r);
  return m;
}
CUtensorMap map_rows_bf(const torch::Tensor& t, const Geo& g) {
  alignas(64) CUtensorMap m{};
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.numel() == (long)g.A * g.B * g.S * C, "bf16 [N*S, 128]");
  uint64_t gdim[4] = {(uint64_t)C, (uint64_t)g.S, (uint64_t)g.B, (uint64_t)g.A};
  uint64_t gstride[3] = {(uint64_t)C * 2, (uint64_t)g.S * C * 2, (uint64_t)g.B * g.S * C * 2};
  uint32_t bdim[4] = {64, (uint32_t)g.AT, 1, (uint32_t)g.SP}; uint32_t estride[4] = {1, 1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(rows bf16) failed: ", (int)r);
  return m;
}
// head-major bf16 [N, H, S, D] as [A][B][H][S][32]: boxes {32, AT, 1, 1, SP}, 64 B swizzle
CUtensorMap map_hm(const torch::Tensor& t, const Geo& g) {
  alignas(64) CUtensorMap m{};
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.numel() == (long)g.A * g.B * g.S * C, "bf16 [N, H, S, D]");
  uint64_t gdim[5] = {(uint64_t)D, (uint64_t)g.S, (uint64_t)H, (uint64_t)g.B, (uint64_t)g.A};
  uint64_t gstride[4] = {(uint64_t)D * 2, (uint64_t)g.S * D * 2, (uint64_t)H * g.S * D * 2, (uint64_t)g.B * H * g.S * D * 2};
  uint32_t bdim[5] = {D, (uint32_t)g.AT, 1, 1, (uint32_t)g.SP}; uint32_t estride[5] = {1, 1, 1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, t.data_ptr(), gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(head-major) failed: ", (int)r);
  return m;
}
int pair_grid(const Geo& g, int dev) { return 2 * std::max(1, std::min(g.ntile, num_sms(dev) / 2)); }
}  // namespace

// The weights as the GEMMs stream them (fwd: F_* layout, else B_*). Wqkv [384, 128], Wg / Wo [128, 128], Wu [512, 128],
// Wd [128, 256], fp32.
torch::Tensor prep(const torch::Tensor& wqkv, const torch::Tensor& wg, const torch::Tensor& wo, const torch::Tensor& wu,
                   const torch::Tensor& wd, bool fwd) {
  TORCH_CHECK(wqkv.sizes() == torch::IntArrayRef({3 * C, C}) && wg.sizes() == torch::IntArrayRef({C, C}) &&
              wo.sizes() == torch::IntArrayRef({C, C}) && wu.sizes() == torch::IntArrayRef({2 * NHID, C}) &&
              wd.sizes() == torch::IntArrayRef({C, NHID}), "weights: Wqkv [384, 128], Wg / Wo [128, 128], Wu [512, 128], Wd [128, 256]");
  for (const auto* w : {&wqkv, &wg, &wo, &wu, &wd})
    TORCH_CHECK(w->is_cuda() && w->scalar_type() == torch::kFloat && w->is_contiguous(), "weights: fp32 contiguous");
  auto out = torch::empty({fwd ? F_END : B_END}, wqkv.options());
  prep_kernel<<<2 * num_sms(wqkv.device().index()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), wqkv.data_ptr<float>(), wg.data_ptr<float>(), wo.data_ptr<float>(), wu.data_ptr<float>(),
      wd.data_ptr<float>(), fwd ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Forward. q [N, S, C]; mod [B*S, 768]; cos / sin [B*S, 16]; seqused [N] int32; weights fp32.
// Returns [out [N*S, C]] or, with save, [out, Q, K, V (bf16 [N, H, S, D]), G, O (bf16 [N*S, C]), lse [N, H, S], q1, x, pq, pk,
// att, y, ffn] -- the layout of the Triton fp32 forward.
std::vector<torch::Tensor> fwd(torch::Tensor q, torch::Tensor mod, torch::Tensor cs, torch::Tensor sn, torch::Tensor sequ,
                               torch::Tensor wqkv, torch::Tensor wg, torch::Tensor wo, torch::Tensor wu, torch::Tensor wd,
                               int64_t B, int64_t half_window, double eps, double qk_eps, bool save) {
  TORCH_CHECK(half_window == HWIN, "the fp32 CUDA block serves half_window 64");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == C, "q [N, S, 128]");
  const long N = q.size(0), S = q.size(1), M = N * S, A = N / B;
  TORCH_CHECK(A * B == N && S > 0, "N % B, S > 0");
  check_rows(q, M, C, "q");
  check_rows(mod, B * S, MODW, "mod");
  check_rows(cs, B * S, D / 2, "cos");
  check_rows(sn, B * S, D / 2, "sin");
  TORCH_CHECK(sequ.is_cuda() && sequ.scalar_type() == torch::kInt && sequ.numel() == N && sequ.is_contiguous(), "seqused int32 [N]");
  const c10::cuda::CUDAGuard guard(q.device());
  const int dev = q.device().index();
  auto W = prep(wqkv, wg, wo, wu, wd, true);
  const float* w = W.data_ptr<float>();
  auto f32 = q.options(), bf = q.options().dtype(torch::kBFloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  const Geo geo = make_geo(A, B, S, bwd_sp(A));
  auto Qh = torch::empty({N, H, S, D}, bf), Kh = torch::empty_like(Qh), Vh = torch::empty_like(Qh);
  auto G = torch::empty({M, C}, f32);
  torch::Tensor X, PQ, PK, Q1, Att, Y, FF;
  if (save) {
    X = torch::empty({M, C}, f32); PQ = torch::empty_like(X); PK = torch::empty_like(X);
    Q1 = torch::empty_like(X); Att = torch::empty_like(X); Y = torch::empty_like(X); FF = torch::empty_like(X);
  }
  const CUtensorMap mG = map_rows(G, geo, C);
  {
    static bool attr = false;
    if (!attr) { smem_attr(qkvg_fwd_kernel, Q1_BYTES); attr = true; }
    QkvgFwdArgs a{mod.data_ptr<float>(), cs.data_ptr<float>(), sn.data_ptr<float>(), geo, (float)eps, (float)qk_eps, save ? 1 : 0};
    qkvg_fwd_kernel<<<pair_grid(geo, dev), TPB, Q1_BYTES, stream>>>(
        a, map_w(w + F_W1, 512, C, 128), map_rows(q, geo, C), save ? map_rows(X, geo, C) : mG, save ? map_rows(PQ, geo, C) : mG,
        save ? map_rows(PK, geo, C) : mG, mG, map_hm(Qh, geo), map_hm(Kh, geo), map_hm(Vh, geo));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto O = torch::empty({M, C}, bf);
  auto lse = torch::empty({N, H, S}, f32);
  {
    AttnArgs a{};
    a.Q = bfp(Qh); a.K = bfp(Kh); a.V = bfp(Vh); a.sequ = sequ.data_ptr<int>();
    a.O = bfp_mut(O); a.lse = lse.data_ptr<float>(); a.S = (int)S; a.scale = 1.f / sqrtf((float)D);
    attn_fwd_kernel<<<dim3((unsigned)((S + 63) / 64), (unsigned)(N * H)), 128, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto out = torch::empty({M, C}, f32);
  {
    static bool attr = false;
    if (!attr) { smem_attr(oproj_ffn_fwd_kernel, F4_BYTES); attr = true; }
    FfnFwdArgs a{mod.data_ptr<float>(), geo, (float)eps, save ? 1 : 0};
    const CUtensorMap mout = map_rows(out, geo, C);
    oproj_ffn_fwd_kernel<<<pair_grid(geo, dev), TPB, F4_BYTES, stream>>>(
        a, map_w(w + F_WO, C, C, 128), map_w(w + F_WABH, 512, C, 64), map_w(w + F_WABL, 512, C, 64), map_w(w + F_WDH, C, NHID, 128),
        map_w(w + F_WDL, C, NHID, 128), map_rows(q, geo, C), mG, map_rows_bf(O, geo), mout, save ? map_rows(Q1, geo, C) : mout,
        save ? map_rows(Att, geo, C) : mout, save ? map_rows(Y, geo, C) : mout, save ? map_rows(FF, geo, C) : mout);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (!save) return {out};
  return {out, Qh, Kh, Vh, G, O, lse, Q1, X, PQ, PK, Att, Y, FF};
}

// Backward from the forward's saved tensors. Returns [dq [N*S, C], dmod [B*S, 768], dP [N*S, 512], datt, gated [N*S, C],
// dab [N*S, 512], hh [N*S, 256], dffn [N*S, C]] -- the weight gradients are GEMMs over these (dispatch.py).
std::vector<torch::Tensor> bwd(torch::Tensor dy, torch::Tensor q, torch::Tensor mod, torch::Tensor cs, torch::Tensor sn,
                               torch::Tensor sequ, torch::Tensor wqkv, torch::Tensor wg, torch::Tensor wo, torch::Tensor wu,
                               torch::Tensor wd, torch::Tensor qh, torch::Tensor kh, torch::Tensor vh, torch::Tensor g,
                               torch::Tensor o, torch::Tensor lse, torch::Tensor q1, torch::Tensor pq, torch::Tensor pk,
                               torch::Tensor att, torch::Tensor y, torch::Tensor ffn, int64_t B, int64_t half_window, double eps,
                               double qk_eps) {
  TORCH_CHECK(half_window == HWIN, "the fp32 CUDA block serves half_window 64");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == C, "q [N, S, 128]");
  const long N = q.size(0), S = q.size(1), M = N * S, A = N / B;
  TORCH_CHECK(A * B == N && S > 0, "N % B, S > 0");
  for (const auto* t : {&dy, &q, &g, &q1, &pq, &pk, &att, &y, &ffn}) check_rows(*t, M, C, "row tensor");
  check_rows(mod, B * S, MODW, "mod");
  check_rows(cs, B * S, D / 2, "cos");
  check_rows(sn, B * S, D / 2, "sin");
  check_rows(lse, N * H, S, "lse");
  TORCH_CHECK(sequ.is_cuda() && sequ.scalar_type() == torch::kInt && sequ.numel() == N && sequ.is_contiguous(), "seqused int32 [N]");
  for (const auto* t : {&qh, &kh, &vh, &o})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == M * C, "bf16 attention operands");
  const c10::cuda::CUDAGuard guard(q.device());
  const int dev = q.device().index();
  auto W = prep(wqkv, wg, wo, wu, wd, false);
  const float* w = W.data_ptr<float>();
  auto f32 = q.options(), bf = q.options().dtype(torch::kBFloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  const Geo geo = make_geo(A, B, S, bwd_sp(A));
  auto dmod = torch::zeros({B * S, MODW}, f32);
  auto dq1 = torch::empty({M, C}, f32), dffn = torch::empty({M, C}, f32), hh = torch::empty({M, NHID}, f32);
  auto dab = torch::empty({M, 2 * NHID}, f32), datt = torch::empty({M, C}, f32), gated = torch::empty({M, C}, f32);
  auto dG = torch::empty({M, C}, f32), Dv = torch::empty({N, H, S}, f32);
  auto dO = torch::empty({M, C}, bf);
  const CUtensorMap mdq1 = map_rows(dq1, geo, C), mg = map_rows(g, geo, C), mdg = map_rows(dG, geo, C), matt = map_rows(att, geo, C);
  {
    static bool attr = false;
    if (!attr) { smem_attr(ffn_bwd_kernel, B4_BYTES); attr = true; }
    FfnBwdArgs a{mod.data_ptr<float>(), dmod.data_ptr<float>(), geo, (float)eps};
    ffn_bwd_kernel<<<pair_grid(geo, dev), TPB, B4_BYTES, stream>>>(
        a, map_w(w + B_WABH, 512, C, 64), map_w(w + B_WABL, 512, C, 64), map_w(w + B_WDTH, NHID, C, 32), map_w(w + B_WDTL, NHID, C, 32),
        map_w(w + B_WABTH, C, 512, 128), map_w(w + B_WABTL, C, 512, 128), map_rows(dy, geo, C), map_rows(ffn, geo, C),
        map_rows(y, geo, C), map_rows(q1, geo, C), map_rows(dffn, geo, C), map_rows(hh, geo, NHID), map_rows(dab, geo, 2 * NHID), mdq1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    static bool attr = false;
    if (!attr) { smem_attr(oproj_bwd_kernel, O4_BYTES); attr = true; }
    OprojBwdArgs a{mod.data_ptr<float>(), dmod.data_ptr<float>(), Dv.data_ptr<float>(), geo};
    oproj_bwd_kernel<<<pair_grid(geo, dev), TPB, O4_BYTES, stream>>>(
        a, map_w(w + B_WOT, C, C, 64), mdq1, matt, mg, map_rows_bf(o, geo), map_rows(datt, geo, C), map_rows(gated, geo, C), mdg,
        map_rows_bf(dO, geo));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto dQh = torch::empty({N, H, S, D}, bf), dKh = torch::empty_like(dQh), dVh = torch::empty_like(dQh);
  {
    AttnArgs a{};
    a.Q = bfp(qh); a.K = bfp(kh); a.V = bfp(vh); a.sequ = sequ.data_ptr<int>();
    a.dout = bfp(dO); a.lse_in = lse.data_ptr<float>(); a.dvv = Dv.data_ptr<float>();
    a.dQ = bfp_mut(dQh); a.dK = bfp_mut(dKh); a.dV = bfp_mut(dVh); a.S = (int)S; a.scale = 1.f / sqrtf((float)D);
    static bool attr = false;
    if (!attr) { smem_attr(attn_bwd_kernel, AB_BYTES); attr = true; }
    attn_bwd_kernel<<<dim3((unsigned)((S + 63) / 64), (unsigned)(N * H)), 256, AB_BYTES, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto dq = torch::empty({M, C}, f32), dP = torch::empty({M, 4 * C}, f32);
  {
    static bool attr = false;
    if (!attr) { smem_attr(qkvg_bwd_kernel, Q4_BYTES); attr = true; }
    QkvgBwdArgs a{mod.data_ptr<float>(), cs.data_ptr<float>(), sn.data_ptr<float>(), dmod.data_ptr<float>(), geo, (float)eps, (float)qk_eps};
    qkvg_bwd_kernel<<<pair_grid(geo, dev), TPB, Q4_BYTES, stream>>>(
        a, map_w(w + B_W1T, C, 512, 128), map_hm(dQh, geo), map_hm(dKh, geo), map_hm(dVh, geo), map_rows(pq, geo, C),
        map_rows(pk, geo, C), mdg, map_rows(q, geo, C), mdq1, map_rows(dP, geo, 4 * C), map_rows(dq, geo, C));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dq, dmod, dP, datt, gated, dab, hh, dffn};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "ESMFold2 SWA atom DiT block forward, fp32 (tf32 wgmma, split-tf32 FFN, bf16 window attention)");
  m.def("bwd", &bwd, "ESMFold2 SWA atom DiT block backward, fp32 (weight gradients excluded)");
  m.def("prep", &prep, "the weights as the fp32 block's GEMMs stream them (tf32-rounded / split, K-permuted, transposed)");
}
