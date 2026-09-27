// TriangleAttention module kernels around the attention core for B200 (sm_100a) -- the H100 kernel boundaries (LN + projections,
// gate + out-projection, gate backward + delta, projection dgrad + LN backward), developed on tcgen05 / TMEM / TMA.
//
// front:  y = LN(x) (never stored); q | k | v | g = y . W^T  (four 128 x 128 projections, bf16, the [rows][H*32] projection layout
//         the attention kernels read); bias[b, h, j, k] = y . Wb[h]  (head-major for the attention, masked keys = -inf)
#include <torch/extension.h>
#include <cstring>
#include "sm100.cuh"
using namespace sm100;

namespace {
constexpr int C = 128;                                    // d_pair
constexpr int NH = 4;                                     // heads (bias outputs)

namespace fr {
// Tile = 128 rows (tokens).  warp 0: TMA (x tiles, the weights once); warp 1: tcgen05 issue (q|k then v|g, two 256-column TMEM
// halves so the drain of one overlaps the other's GEMMs); warps 2-9: LayerNorm in place (two threads a row) + the bias
// projection on the CUDA cores (4 outputs a row); warps 10-13: drain (TMEM lane = row) -> two 64B-swizzled [128][32] staging tiles
// -> TMA stores.  (The staging must not be the tile's x buffer: the v|g GEMMs still read y there while q|k drain.)  The weights
// stay resident: [2 K halves][512 outputs][64]; the x buffer is released by the tile's last GEMM commit.
constexpr int BM = 128, NST = 2;
constexpr int XT = BM * C;                                // x / y tile [2 halves][128][64] (32 KiB)
constexpr int WT = 4 * C * C;                             // q|k|v|g weights (128 KiB)
constexpr int THREADS = 448;
constexpr int OT = BM * 32;                               // a staging tile [128][32] (8 KiB)
constexpr int SMEM = 1024 + (NST * XT + WT + 2 * OT) * 2 + (2 * C + NH * C) * 4 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, 128, 0, 0);
static_assert(SMEM <= 232448, "smem");
}  // namespace fr

__global__ void __launch_bounds__(fr::THREADS, 1) tri_front_sm100(
    int R, int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap xmap,             // x [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,             // W [512][128] bf16, box (64, 256), 128B swizzle
    const __grid_constant__ CUtensorMap qmap,             // q / k / v / g [R][128], box (32, 128), 64B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap gmap,
    const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WB,   // [128], [128], [4][128]
    const bool* __restrict__ MASK,                        // [B][L] key mask or nullptr
    __nv_bfloat16* __restrict__ BIAS) {                   // [B][4][L][L]
  using namespace fr;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][XT]
  __nv_bfloat16* sW = sX + NST * XT;                             // [2 K halves][512][64]
  __nv_bfloat16* sO = sW + WT;                                   // [2][OT] staging
  float* sP = reinterpret_cast<float*>(sO + 2 * OT);             // lnw [128] | lnb [128] | Wb [4][128]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sP + 2 * C + NH * C);
  uint64_t* xf = bars;               // [NST] x landed
  uint64_t* yf = xf + NST;           // [NST] count 8: y written
  uint64_t* xe = yf + NST;           // [NST] count 1: the tile's GEMMs retired (the buffer takes the tile after next)
  uint64_t* af = xe + NST;           // [2]   accumulator half ready
  uint64_t* ae = af + 2;             // [2]   count 4: drained
  uint64_t* wf = ae + 2;             // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { bar_init(xf + i, 1); bar_init(yf + i, 8); bar_init(xe + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  for (int i = tid; i < 2 * C + NH * C; i += blockDim.x) sP[i] = i < C ? LNW[i] : i < 2 * C ? LNB[i - C] : WB[i - 2 * C];
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, WT * 2);
      for (int kh = 0; kh < 2; ++kh)
        for (int nh = 0; nh < 2; ++nh) load_2d(&wmap, sW + kh * 4 * C * 64 + nh * 256 * 64, wf, kh * 64, nh * 256);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST;
        if (lt >= NST) wait(xe + st, ((lt / NST) - 1) & 1);
        expect_tx(xf + st, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&xmap, sX + st * XT + kh * BM * 64, xf + st, kh * 64, t * BM);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST;
        wait(yf + st, (lt / NST) & 1);
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          if (lt >= 1) wait(ae + half, (lt - 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int tt = 0; tt < 2; ++tt) {
            const int ten = half * 2 + tt;
#pragma unroll
            for (int ks = 0; ks < C / 16; ++ks) {
              const int kh = ks >> 2, kk = (ks & 3) * 16;
              mma_ss(tmem + ten * 128, desc_k128(y + kh * BM * 64 + kk), desc_k128(sW + kh * 4 * C * 64 + ten * C * 64 + kk), IDESC, ks ? 1u : 0u);
            }
          }
          mma_commit(af + half);
        }
        mma_commit(xe + st);
      }
    }
  } else if (warp < 10) {
    // ---- LayerNorm: row r, channel half hf (64 channels = 8 chunks of 16 B in the half's 128B-swizzled [128][64] tile) ----
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST;
      wait(xf + st, (lt / NST) & 1);
      __nv_bfloat16* xr = sX + st * XT + hf * BM * 64 + r * 64;
      float2 x[32];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        x[c8 * 4 + 0] = bf2f(u.x); x[c8 * 4 + 1] = bf2f(u.y); x[c8 * 4 + 2] = bf2f(u.z); x[c8 * 4 + 3] = bf2f(u.w);
      }
      float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
      for (int i = 0; i < 32; i += 2) { s0 = add2(s0, x[i]); s1 = add2(s1, x[i + 1]); }
      float sum = s0.x + s0.y + s1.x + s1.y;
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      const float mean = sum * (1.f / C);
      const float2 nm = make_float2(-mean, -mean);
      float2 v0 = make_float2(0.f, 0.f), v1 = v0;
#pragma unroll
      for (int i = 0; i < 32; i += 2) {
        x[i] = add2(x[i], nm); x[i + 1] = add2(x[i + 1], nm);
        v0 = fma2(x[i], x[i], v0); v1 = fma2(x[i + 1], x[i + 1], v1);
      }
      float var = v0.x + v0.y + v1.x + v1.y;
      var += __shfl_xor_sync(0xffffffffu, var, 16);
      const float rstd = rsqrtf(var * (1.f / C) + eps);
      const float2 rs = make_float2(rstd, rstd);
      float bacc[NH] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const int c0 = hf * 64 + c8 * 8;
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = c0 + 2 * e;
          const float2 yv = fma2(mul2(x[c8 * 4 + e], rs), make_float2(sP[c], sP[c + 1]), make_float2(sP[C + c], sP[C + c + 1]));
          o[e] = pack2(yv.x, yv.y);
          const float2 yb = bf2f(o[e]);           // the bias projection sees the bf16 y the GEMMs see
#pragma unroll
          for (int h = 0; h < NH; ++h) bacc[h] = fmaf(yb.x, sP[2 * C + h * C + c], fmaf(yb.y, sP[2 * C + h * C + c + 1], bacc[h]));
        }
        *reinterpret_cast<uint4*>(xr + ((c8 ^ (r & 7)) << 3)) = make_uint4(o[0], o[1], o[2], o[3]);
      }
#pragma unroll
      for (int h = 0; h < NH; ++h) bacc[h] += __shfl_xor_sync(0xffffffffu, bacc[h], 16);
      if (hf == 0) {
        const long row = (long)t * BM + r;       // (b, j, k)
        const int kx = (int)(row % L);
        const long bj = row / L;
        const int j = (int)(bj % L), b = (int)(bj / L);
        const bool keep = MASK == nullptr || MASK[(long)b * L + kx];
#pragma unroll
        for (int h = 0; h < NH; ++h)
          BIAS[(((long)b * NH + h) * L + j) * L + kx] = keep ? __float2bfloat16_rn(bacc[h]) : __float2bfloat16_rn(-3.3895313892515355e38f);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yf + st);
    }
  } else {
    // ---- drain: TMEM lane = row; each output tensor in four 32-column chunks through two staging tiles ----
    const int qq = warp & 3, r = qq * 32 + lane;
    const bool leader = warp == 10 && lane == 0;
    const CUtensorMap* maps[4] = {&qmap, &kmap, &vmap, &gmap};
    int nst = 0;                                  // stores issued
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
#pragma unroll 1
      for (int half = 0; half < 2; ++half) {
        wait(af + half, lt & 1);
        tc_fence_after();
#pragma unroll 1
        for (int tt = 0; tt < 2; ++tt) {
          const int ten = half * 2 + tt;
#pragma unroll 1
          for (int ch = 0; ch < 4; ++ch) {
            float v[32];
            tmem_ld32(tmem_at(tmem + ten * 128 + ch * 32, qq * 32, 0), v);
            tmem_wait_ld();
            if (tt == 1 && ch == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + half); }
            __nv_bfloat16* buf = sO + (nst & 1) * OT;
            if (leader && nst >= 2) bulk_wait_read<1>();   // the store two back (this tile) has read it
            named_sync(1, 128);
            uint32_t w[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) w[i] = pack2(v[2 * i], v[2 * i + 1]);
            unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + r * 64;
#pragma unroll
            for (int c = 0; c < 4; ++c)
              *reinterpret_cast<uint4*>(rowp + ((c ^ ((r >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
            fence_proxy_async();
            named_sync(1, 128);
            if (leader) { store_2d(maps[ten], buf, ch * 32, t * BM); bulk_commit(); }
            ++nst;
          }
        }
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
// ---------------------------------------------------------------------------------------------------------------------------------
// tail:  out = x + (sigmoid(g) o o) . Wo^T   (the gate, the out-projection and the residual in one pass)
// Tile = 128 rows.  warp 0: TMA (g | o in a two-stage ring, the residual x single-buffered, Wo once); warp 1: tcgen05 (N = 128,
// K = 128; double-buffered TMEM); warps 2-9: u = sigmoid(g) o o written over g (the GEMM's A tile), two threads a row; warps 10-13:
// drain + x -> 64B-swizzled [128][32] staging -> TMA stores.
__device__ __forceinline__ void reduce_add_2d_bf(const void* map, const void* src, int c0, int c1) {   // bf16 add into global
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ float sigmoid_t(float x) {   // 0.5 + 0.5 tanh(x / 2): one MUFU op
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}
namespace tl {
constexpr int BM = 128, NST = 2;
constexpr int XT = BM * C;                                // [2 halves][128][64] 128B-swizzled (32 KiB)
constexpr int OT = BM * 32;
constexpr int THREADS = 448;
constexpr int SMEM = 1024 + (NST * 2 * XT + XT + C * C + 2 * OT) * 2 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, 128, 0, 0);
static_assert(SMEM <= 232448, "smem");
}  // namespace tl

__global__ void __launch_bounds__(tl::THREADS, 1) tri_tail_sm100(
    int ntiles,
    const __grid_constant__ CUtensorMap gmap,             // g / o / x [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap omap,
    const __grid_constant__ CUtensorMap xmap,
    const __grid_constant__ CUtensorMap wmap,             // Wo [128][128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap) {           // out [R][128], box (32, 128), 64B swizzle (TMA stores)
  using namespace tl;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][g | o]
  __nv_bfloat16* sX = sG + NST * 2 * XT;                         // residual tile
  __nv_bfloat16* sW = sX + XT;                                   // [2 K halves][128][64]
  __nv_bfloat16* sO = sW + C * C;                                // [2][OT]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sO + 2 * OT);
  uint64_t* gf = bars;               // [NST] g | o landed
  uint64_t* uf = gf + NST;           // [NST] count 8: u written
  uint64_t* ge = uf + NST;           // [NST] the GEMM retired (the stage takes the tile after next)
  uint64_t* xf = ge + NST;           // [1]   residual landed
  uint64_t* xe = xf + 1;             // [1]   count 4: residual read
  uint64_t* af = xe + 1;             // [2]
  uint64_t* ae = af + 2;             // [2]   count 4
  uint64_t* wf = ae + 2;             // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { bar_init(gf + i, 1); bar_init(uf + i, 8); bar_init(ge + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init(xf, 1); bar_init(xe, 4); bar_init(wf, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, C * C * 2);
      for (int kh = 0; kh < 2; ++kh) load_2d(&wmap, sW + kh * C * 64, wf, kh * 64, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST;
        if (lt >= NST) wait(ge + st, ((lt / NST) - 1) & 1);
        expect_tx(gf + st, 2 * XT * 2);
        for (int kh = 0; kh < 2; ++kh) {
          load_2d(&gmap, sG + st * 2 * XT + kh * BM * 64, gf + st, kh * 64, t * BM);
          load_2d(&omap, sG + st * 2 * XT + XT + kh * BM * 64, gf + st, kh * 64, t * BM);
        }
        if (lt >= 1) wait(xe, (lt - 1) & 1);
        expect_tx(xf, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&xmap, sX + kh * BM * 64, xf, kh * 64, t * BM);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST, b = lt & 1;
        wait(uf + st, (lt / NST) & 1);
        if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* u = sG + st * 2 * XT;
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
          const int kh = ks >> 2, kk = (ks & 3) * 16;
          mma_ss(tmem + b * 128, desc_k128(u + kh * BM * 64 + kk), desc_k128(sW + kh * C * 64 + kk), IDESC, ks ? 1u : 0u);
        }
        mma_commit(af + b);
        mma_commit(ge + st);
      }
    }
  } else if (warp < 10) {
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST;
      wait(gf + st, (lt / NST) & 1);
      __nv_bfloat16* gr = sG + st * 2 * XT + hf * BM * 64 + r * 64;
      const __nv_bfloat16* orow = gr + XT;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const int off = (c8 ^ (r & 7)) << 3;
        const uint4 a = *reinterpret_cast<const uint4*>(gr + off), o = *reinterpret_cast<const uint4*>(orow + off);
        const uint32_t aa[4] = {a.x, a.y, a.z, a.w}, oo[4] = {o.x, o.y, o.z, o.w};
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 gv = bf2f(aa[e]), ov = bf2f(oo[e]);
          w[e] = pack2(sigmoid_t(gv.x) * ov.x, sigmoid_t(gv.y) * ov.y);
        }
        *reinterpret_cast<uint4*>(gr + off) = make_uint4(w[0], w[1], w[2], w[3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(uf + st);
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;
    const bool leader = warp == 10 && lane == 0;
    int nst = 0, lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int b = lt & 1;
      wait(af + b, (lt >> 1) & 1);
      wait(xf, lt & 1);
      tc_fence_after();
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        float v[32];
        tmem_ld32(tmem_at(tmem + b * 128 + ch * 32, qq * 32, 0), v);
        // residual: channels 32 ch .. 32 ch + 31 of row r = 16-byte chunks 4 (ch & 1) .. +3 of half ch >> 1
        const __nv_bfloat16* xr = sX + (ch >> 1) * BM * 64 + r * 64;
        uint32_t xw[16];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint4 u = *reinterpret_cast<const uint4*>(xr + ((((ch & 1) * 4 + c) ^ (r & 7)) << 3));
          xw[c * 4] = u.x; xw[c * 4 + 1] = u.y; xw[c * 4 + 2] = u.z; xw[c * 4 + 3] = u.w;
        }
        tmem_wait_ld();
        if (ch == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) { arrive(ae + b); arrive(xe); } }
        uint32_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) { const float2 xv = bf2f(xw[i]); w[i] = pack2(v[2 * i] + xv.x, v[2 * i + 1] + xv.y); }
        __nv_bfloat16* buf = sO + (nst & 1) * OT;
        if (leader && nst >= 2) bulk_wait_read<1>();
        named_sync(1, 128);
        unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + r * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c)
          *reinterpret_cast<uint4*>(rowp + ((c ^ ((r >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
        fence_proxy_async();
        named_sync(1, 128);
        if (leader) { store_2d(&ymap, buf, ch * 32, t * BM); bulk_commit(); }
        ++nst;
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}
// ---------------------------------------------------------------------------------------------------------------------------------
// gate backward:  du = dy . Wo;  s = sigmoid(g);  do = du o s;  dg = du o o o s (1 - s);  delta[b, i, h, j] = sum_{c in h} do . o;
//                 dWo += dy^T . (s o o)   (accumulated in TMEM over the CTA's tiles, added into the fp32 dWo once at the end)
// Tile = 128 rows.  warp 0: TMA; warp 1: tcgen05 (du: A = dy K-major, B = Wo MN-major; dWo: A = dy^T, B = u^T, both MN-major
// straight from the row tiles); warps 2-5 / 6-9: two groups, one 64-channel half each (a 32-channel chunk = one head: delta
// falls out per chunk); they write do into its own tile, dg over g and u = s o o over o, and TMA-store do / dg.
// Buffers are single (dy / g / o / do: 128 KiB + Wo): each input of the next tile is loaded as soon as its buffer frees.
#ifndef GB_NOSEED
#define GB_NOSEED 0
#endif
#ifndef GB_NOTICKET
#define GB_NOTICKET 0
#endif
namespace gb {
constexpr int BM = 128, XT = BM * C;
constexpr int THREADS = 320;
constexpr int SMEM = 1024 + (4 * XT + C * C) * 2 + 512;
constexpr uint32_t ID_DU = idesc_bf16(128, 128, 0, 1);   // A = dy K-major, B = Wo MN-major (N = in contiguous)
constexpr uint32_t ID_DW = idesc_bf16(128, 128, 1, 1);   // A = dy^T, B = u^T: both MN-major
static_assert(SMEM <= 232448, "smem");
}  // namespace gb

__global__ void __launch_bounds__(gb::THREADS, 1) tri_gate_bwd_sm100(
    int L, int ntiles,
    const __grid_constant__ CUtensorMap dymap,            // dy / g / o [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap gmap,
    const __grid_constant__ CUtensorMap omap,
    const __grid_constant__ CUtensorMap wmap,             // Wo [128 out][128 in], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap domap,            // do / dg [R][128], box (64, 128), 128B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dgmap,
    const __grid_constant__ CUtensorMap seedmap,          // dpair [R][128] <- dy (the residual's share; head_bwd adds dx), box (64, 128), 128B
    float* __restrict__ DELTA,                            // [B, L (i), 4, L (j)]
    float* __restrict__ DWO,                              // [128 out][128 in] fp32, accumulated (zeroed)
    unsigned* __restrict__ TICKET,                        // zeroed; the last CTA converts dWo and resets it
    __nv_bfloat16* __restrict__ DWO16) {
  using namespace gb;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sDy = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 halves][128][64] each
  __nv_bfloat16* sG = sDy + XT;                                  // g -> dg
  __nv_bfloat16* sO = sG + XT;                                   // o -> u
  __nv_bfloat16* sD = sO + XT;                                   // do
  __nv_bfloat16* sW = sD + XT;                                   // Wo
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + C * C);
  uint64_t* af = bars;               // dy | o landed
  uint64_t* bfb = af + 1;            // g landed
  uint64_t* wf = bfb + 1;            // Wo landed
  uint64_t* duf = wf + 1;            // du ready
  uint64_t* due = duf + 1;           // count 8: du read
  uint64_t* uf = due + 1;            // count 8: u written (both halves)
  uint64_t* dwd = uf + 1;            // the tile's dWo GEMM retired (dy / u buffers free)
  uint64_t* sd = dwd + 1;            // count 2: the do / dg stores read their tiles (g / do buffers free)
  uint64_t* fin = sd + 1;            // the last dWo GEMM retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(fin + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    bar_init(af, 1); bar_init(bfb, 1); bar_init(wf, 1); bar_init(duf, 1); bar_init(due, 8); bar_init(uf, 8);
    bar_init(dwd, 1); bar_init(sd, 2); bar_init(fin, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;                    // du at 0, dWo at 128
  const int mytiles = (int)blockIdx.x < ntiles ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, C * C * 2);
      for (int kh = 0; kh < 2; ++kh) load_2d(&wmap, sW + kh * C * 64, wf, kh * 64, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        if (lt >= 1) {
          // the previous tile's dy seeds dpair: stored here, where this thread waits for its dWo GEMM anyway (the dy tile landed
          // long ago); the reload below waits for the store to have read it
          wait(af, (lt - 1) & 1);
          if (!GB_NOSEED) for (int kh = 0; kh < 2; ++kh) store_2d(&seedmap, sDy + kh * BM * 64, kh * 64, (t - (int)gridDim.x) * BM);
          bulk_commit();
          wait(dwd, (lt - 1) & 1);
          bulk_wait_read<0>();
        }
        expect_tx(af, 2 * XT * 2);
        for (int kh = 0; kh < 2; ++kh) {
          load_2d(&dymap, sDy + kh * BM * 64, af, kh * 64, t * BM);
          load_2d(&omap, sO + kh * BM * 64, af, kh * 64, t * BM);
        }
        if (lt >= 1) wait(sd, (lt - 1) & 1);
        expect_tx(bfb, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&gmap, sG + kh * BM * 64, bfb, kh * 64, t * BM);
      }
      if (lt >= 1) {                                // the last tile's seed
        wait(af, (lt - 1) & 1);
        const int tl = blockIdx.x + (lt - 1) * (int)gridDim.x;
        if (!GB_NOSEED) for (int kh = 0; kh < 2; ++kh) store_2d(&seedmap, sDy + kh * BM * 64, kh * 64, tl * BM);
        bulk_commit();
        bulk_wait<0>();
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        wait(af, lt & 1);
        if (lt >= 1) wait(due, (lt - 1) & 1);      // the previous du was read
        tc_fence_after();
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {       // du = dy . Wo  (K = out: 16 rows of Wo per step)
          const int kh = ks >> 2, kk = (ks & 3) * 16;
          mma_ss(tmem, desc_k128(sDy + kh * BM * 64 + kk), desc_mn128(sW + ks * 16 * 64, C * 128), ID_DU, ks ? 1u : 0u);
        }
        mma_commit(duf);
        wait(uf, lt & 1);
        tc_fence_after();
#pragma unroll
        for (int ks = 0; ks < BM / 16; ++ks)        // dWo += dy^T . u  (K = rows)
          mma_ss(tmem + 128, desc_mn128(sDy + ks * 16 * 64, BM * 128), desc_mn128(sO + ks * 16 * 64, BM * 128), ID_DW, (lt | ks) ? 1u : 0u);
        mma_commit(dwd);
        if (lt == mytiles - 1) mma_commit(fin);
      }
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;   // TMEM lane = row
    const int gi = (warp - 2) >> 2;                // channel half
    const bool lead = qq == 2 && lane == 0;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      wait(duf, lt & 1);
      wait(bfb, lt & 1);
      tc_fence_after();
      float du[2][32];
      tmem_ld32(tmem_at(tmem + gi * 64, qq * 32, 0), du[0]);
      tmem_ld32(tmem_at(tmem + gi * 64 + 32, qq * 32, 0), du[1]);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(due);
      if (lt >= 1 && lead) bulk_wait_read<0>();    // the previous tile's do store read sD (dg left with g's buffer)
      named_sync(1 + gi, 128);
      const long row = (long)t * BM + r;           // (b, i, j)
      const int j = (int)(row % L);
      const long bi = row / L;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int c = gi * 2 + cc;                 // head
        const int boff = gi * BM * 64 + r * 64;
        float dsum = 0.f;
#pragma unroll
        for (int e4 = 0; e4 < 4; ++e4) {
          const int off = boff + (((cc * 4 + e4) ^ (r & 7)) << 3);
          const uint4 ga = *reinterpret_cast<const uint4*>(sG + off), oa = *reinterpret_cast<const uint4*>(sO + off);
          const uint32_t gw[4] = {ga.x, ga.y, ga.z, ga.w}, ow[4] = {oa.x, oa.y, oa.z, oa.w};
          uint32_t dow[4], dgw[4], uw[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 gv = bf2f(gw[e]), ov = bf2f(ow[e]);
            const float s0 = sigmoid_t(gv.x), s1 = sigmoid_t(gv.y);
            const float u0 = du[cc][e4 * 8 + e * 2], u1 = du[cc][e4 * 8 + e * 2 + 1];
            const float d0 = u0 * s0, d1 = u1 * s1;
            dsum = fmaf(d0, ov.x, fmaf(d1, ov.y, dsum));
            dow[e] = pack2(d0, d1);
            dgw[e] = pack2(u0 * ov.x * s0 * (1.f - s0), u1 * ov.y * s1 * (1.f - s1));
            uw[e] = pack2(s0 * ov.x, s1 * ov.y);
          }
          *reinterpret_cast<uint4*>(sD + off) = make_uint4(dow[0], dow[1], dow[2], dow[3]);
          *reinterpret_cast<uint4*>(sG + off) = make_uint4(dgw[0], dgw[1], dgw[2], dgw[3]);
          *reinterpret_cast<uint4*>(sO + off) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
        }
        DELTA[(bi * 4 + c) * L + j] = dsum;
      }
      fence_proxy_async();
      named_sync(1 + gi, 128);
      __syncwarp();
      if (lane == 0) arrive(uf);
      if (lead) {
        store_2d(&domap, sD + gi * BM * 64, gi * 64, t * BM);
        store_2d(&dgmap, sG + gi * BM * 64, gi * 64, t * BM);
        bulk_commit();
        bulk_wait_read<0>();
        arrive(sd);                                // (both groups: count 2) g / do buffers free
      }
    }
    // ---- dWo: TMEM lane = out, 128 in columns; group 0 adds them into the fp32 dWo ----
    if (gi == 0 && mytiles > 0) {
      wait(fin, 0);
      tc_fence_after();
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        float v[32];
        tmem_ld32(tmem_at(tmem + 128 + ch * 32, qq * 32, 0), v);
        tmem_wait_ld();
        float* dst = DWO + (long)r * C + ch * 32;
#pragma unroll
        for (int i = 0; i < 32; i += 4)
          asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(dst + i), "f"(v[i]), "f"(v[i + 1]), "f"(v[i + 2]), "f"(v[i + 3]) : "memory");
      }
    }
    if (lead) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
  (void)TICKET; (void)DWO16;                        // (a last-CTA bf16 conversion here cost 15 us: tri_param_finish does it)
}
// ---------------------------------------------------------------------------------------------------------------------------------
// head backward:  gy = sum_t d_t . W_t + sum_h db_h Wb_h   (the five projections' input gradient: q | k | v | g via tcgen05, the
//                 4-wide bias one on the CUDA cores);  LN backward from the recomputed statistics;  dpair += dx (a bf16 TMA
//                 reduce-add into the seed gate_bwd wrote, in place).  The weight / LN-parameter gradients are tri_wgrad's.
// Tile = 128 rows.  warp 0: TMA (x double-buffered; the d_t half-tiles [128][64] in a ring; W once); warp 1: tcgen05 (M = 128 rows,
// N = 128, K = 512 into one of two gy accumulators); warps 2-5 / 6-9: two groups, one 64-channel half each, a row a thread (TMEM
// lane); the row sums go between the groups through shared.  dx overwrites x (the reduce-add's source).
struct HbParams { float lnw[C], lnb[C], wb[NH][C]; };
__constant__ HbParams c_hb;                                // uniform constant-bank operands (filled by an async D2D copy: graph-safe)
namespace hb {
constexpr int BM = 128, XT = BM * C, HT = BM * 64;         // x tile (32 KiB), a d half-tile [128][64] (16 KiB)
constexpr int NR = 2, NX = 2;                              // d ring, x buffers
constexpr int THREADS = 320;
// no 1024 B alignment slack: the dynamic shared window starts 1 KiB aligned on this part (measured; the kernel traps otherwise)
constexpr int SMEM = (4 * C * C + NR * HT + NX * XT) * 2 + 2 * 2 * BM * 4 + 256;
constexpr uint32_t ID_DG = idesc_bf16(128, 128, 0, 1);    // A = d_t K-major, B = W_t MN-major
static_assert(SMEM <= 232448, "smem");
}  // namespace hb

__global__ void __launch_bounds__(hb::THREADS, 1) tri_head_bwd_sm100(
    int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap d0map,            // dq / dk / dv / dg [R][128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map,
    const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap,             // x [R][128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,             // W4 [512][128], box (64, 256), 128B swizzle
    const __grid_constant__ CUtensorMap rmap,             // dpair [R][128] (the seed; reduce-add target), box (64, 128), 128B swizzle
    const float* __restrict__ DB) {                       // dbias [B, 4, L, L] fp32
  using namespace hb;
  extern __shared__ __align__(1024) unsigned char raw[];
  if (sa(raw) & 1023u) __trap();
  unsigned char* smb = raw;
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(smb);    // [2 c halves][512 o][64]
  __nv_bfloat16* sD = sW + 4 * C * C;                            // [NR][128][64]
  __nv_bfloat16* sX = sD + NR * HT;                              // [NX] x -> dx   [2 halves][128][64]
  float* sRow = reinterpret_cast<float*>(sX + NX * XT);          // [2 groups][2][128] row partial sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRow + 2 * 2 * BM);
  uint64_t* df = bars;               // [NR]
  uint64_t* de = df + NR;            // [NR]
  uint64_t* xf = de + NR;            // [NX]
  uint64_t* xe = xf + NX;            // [NX] the dx reduce read the tile
  uint64_t* af = xe + NX;            // [2]  gy ready
  uint64_t* ae = af + 2;             // [2]  count 8: gy read
  uint64_t* wf = ae + 2;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NR; ++i) { bar_init(df + i, 1); bar_init(de + i, 1); }
    for (int i = 0; i < NX; ++i) { bar_init(xf + i, 1); bar_init(xe + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 8); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const CUtensorMap* dmaps[4] = {&d0map, &d1map, &d2map, &d3map};

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, 4 * C * C * 2);
      for (int kh = 0; kh < 2; ++kh)
        for (int nh = 0; nh < 2; ++nh) load_2d(&wmap, sW + kh * 4 * C * 64 + nh * 256 * 64, wf, kh * 64, nh * 256);
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int xb = lt % NX;
        if (lt >= NX) wait(xe + xb, ((lt / NX) - 1) & 1);
        expect_tx(xf + xb, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&xmap, sX + xb * XT + kh * BM * 64, xf + xb, kh * 64, t * BM);
        for (int q8 = 0; q8 < 8; ++q8, ++y) {      // (tensor q8 >> 1, K half q8 & 1)
          const int sl = y % NR;
          if (y >= NR) wait(de + sl, ((y / NR) - 1) & 1);
          expect_tx(df + sl, HT * 2);
          load_2d(dmaps[q8 >> 1], sD + sl * HT, df + sl, (q8 & 1) * 64, t * BM);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int b = lt & 1;
        if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
        for (int q8 = 0; q8 < 8; ++q8, ++y) {
          const int sl = y % NR, ten = q8 >> 1, kh = q8 & 1;
          wait(df + sl, (y / NR) & 1);
          tc_fence_after();
          const __nv_bfloat16* dt = sD + sl * HT;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)            // gy += d_t[:, 64 kh + 16 kk ..] . W_t[64 kh + 16 kk .., :]
            mma_ss(tmem + b * 128, desc_k128(dt + kk * 16), desc_mn128(sW + (ten * C + kh * 64 + kk * 16) * 64, 4 * C * 128), ID_DG, (q8 | kk) ? 1u : 0u);
          mma_commit(de + sl);
        }
        mma_commit(af + b);
      }
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;   // TMEM lane = row
    const int gi = (warp - 2) >> 2;                // channel half
    const bool leader = warp == 2 && lane == 0;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int xb = lt % NX, b = lt & 1;
      __nv_bfloat16* xr = sX + xb * XT + gi * BM * 64 + r * 64;
      const long row = (long)t * BM + r;           // (b, r1, r2): dbias[b, h, r1, r2]
      const long bb = row / ((long)L * L), rr = row % ((long)L * L);
      float dbv[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) dbv[h] = DB[(bb * NH + h) * (long)L * L + rr];
      wait(xf + xb, (lt / NX) & 1);
      // ---- statistics (the two halves' partial sums through shared) ----
      float xv[64];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); xv[c8 * 8 + e * 2] = v.x; xv[c8 * 8 + e * 2 + 1] = v.y; }
      }
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 64; ++i) { s1 += xv[i]; s2 = fmaf(xv[i], xv[i], s2); }
      sRow[(gi * 2) * BM + r] = s1; sRow[(gi * 2 + 1) * BM + r] = s2;
      named_sync(1, 256);
      s1 += sRow[((gi ^ 1) * 2) * BM + r]; s2 += sRow[((gi ^ 1) * 2 + 1) * BM + r];
      named_sync(1, 256);                           // (the buffer is reused for the second exchange)
      const float mean = s1 * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int i = 0; i < 64; ++i) xv[i] = (xv[i] - mean) * rstd;   // xhat
      // ---- gy = TMEM + db . Wb; the row sums ----
      wait(af + b, (lt >> 1) & 1);
      tc_fence_after();
      float gy[64];
      tmem_ld32(tmem_at(tmem + b * 128 + gi * 64, qq * 32, 0), gy);
      tmem_ld32(tmem_at(tmem + b * 128 + gi * 64 + 32, qq * 32, 0), gy + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(ae + b);
      float sg = 0.f, sgx = 0.f;
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        const int c = gi * 64 + i;
        float v = gy[i];
#pragma unroll
        for (int h = 0; h < NH; ++h) v = fmaf(dbv[h], c_hb.wb[h][c], v);
        gy[i] = v * c_hb.lnw[c];
        sg += gy[i]; sgx = fmaf(gy[i], xv[i], sgx);
      }
      sRow[(gi * 2) * BM + r] = sg; sRow[(gi * 2 + 1) * BM + r] = sgx;
      named_sync(1, 256);
      sg += sRow[((gi ^ 1) * 2) * BM + r]; sgx += sRow[((gi ^ 1) * 2 + 1) * BM + r];
      const float mg = sg * (1.f / C), mgx = sgx * (1.f / C);
      // ---- dx over x (in place) -> one TMA reduce-add into dpair ----
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = c8 * 8 + e * 2;
          w[e] = pack2(rstd * (gy[i] - mg - xv[i] * mgx), rstd * (gy[i + 1] - mg - xv[i + 1] * mgx));
        }
        *reinterpret_cast<uint4*>(xr + ((c8 ^ (r & 7)) << 3)) = make_uint4(w[0], w[1], w[2], w[3]);
      }
      fence_proxy_async();
      named_sync(1, 256);
      if (leader) {
        for (int kh = 0; kh < 2; ++kh) reduce_add_2d_bf(&rmap, sX + xb * XT + kh * BM * 64, kh * 64, t * BM);
        bulk_commit();
        bulk_wait_read<0>();
        arrive(xe + xb);
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// weight / LN-parameter gradients.  With xhat the normalized rows (y = gamma o xhat + beta) and d_t the four projections' output
// gradients, everything reduces to accumulations the tensor core and a column sum can do:
//   M_t = xhat^T d_t (tcgen05: M = 128 c, N = 64 o per d half-tile, K = 128 rows; four [128][128] accumulators = all of TMEM),
//   cs_t = 1^T d_t, B_h = sum_rows db_h xhat, cdb_h = sum_rows db_h         (CUDA cores)
// and then (host-side algebra on [128][128] matrices):  dW_t^T = gamma o M_t + beta (x) cs_t;
//   dgamma = sum_t diag(W_t^T M_t^T) + sum_h Wb_h o B_h;  dbeta = sum_t cs_t W_t + sum_h cdb_h Wb_h;  dWb_h = gamma o B_h + beta cdb_h.
// Tile = 128 rows.  warp 0: TMA (x double-buffered, the d half-tiles in a ring, db rows); warp 1: tcgen05; warps 2-5: a row a thread
// (statistics, xhat over x in place = the GEMMs' A tile); warps 6-9: a channel a thread (column sums of the d half-tiles, B_h, cdb).
#ifndef WG_NOCS
#define WG_NOCS 0
#endif
#ifndef WG_NOBH
#define WG_NOBH 0
#endif
#ifndef WG_NOMMA
#define WG_NOMMA 0
#endif
namespace wg {
constexpr int BM = 128, XT = BM * C, HT = BM * 64;
constexpr int NR = 4, NX = 2;
constexpr int THREADS = 320;
constexpr int SMEM = 1024 + (NX * XT + NR * HT) * 2 + NX * BM * NH * 4 + 256;
constexpr uint32_t IDESC = idesc_bf16(128, 64, 1, 1);     // A = xhat^T, B = d_t^T (a 64-wide half): both MN-major
static_assert(SMEM <= 232448, "smem");
}  // namespace wg

__global__ void __launch_bounds__(wg::THREADS, 1) tri_wgrad_sm100(
    int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap d0map, const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map, const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap,
    const float* __restrict__ DB,                         // dbias [B, 4, L, L]
    float* __restrict__ MT,                               // [4][128 c][128 o] fp32, accumulated (zeroed)
    float* __restrict__ VEC,                              // cs [4][128] | B [4][128] | cdb [4], accumulated (zeroed)
    unsigned* __restrict__ TICKET,                        // zeroed; the last CTA finishes and resets it
    const __nv_bfloat16* __restrict__ W4, const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WBF,
    __nv_bfloat16* __restrict__ DW,                       // dW_q|k|v|g [4][128 o][128 c]
    __nv_bfloat16* __restrict__ DGB,                      // dgamma [128] | dbeta [128] | dWb [4][128]
    const int want_finish) {
  using namespace wg;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);   // [NX] x -> xhat
  __nv_bfloat16* sD = sX + NX * XT;                              // [NR][128][64]
  float* sDb = reinterpret_cast<float*>(sD + NR * HT);           // [NX][128 rows][4]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sDb + NX * BM * NH);
  uint64_t* xf = bars;               // [NX]
  uint64_t* hf = xf + NX;            // [NX] count 4: xhat written
  uint64_t* he = hf + NX;            // [NX] count 5: the tile's GEMMs retired (commit) + the channel warps read xhat / db
  uint64_t* df = he + NX;            // [NR]
  uint64_t* de = df + NR;            // [NR] count 5: the GEMM + the channel warps
  uint64_t* fin = de + NR;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(fin + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NX; ++i) { bar_init(xf + i, 1); bar_init(hf + i, 4); bar_init(he + i, 5); }
    for (int i = 0; i < NR; ++i) { bar_init(df + i, 1); bar_init(de + i, 5); }
    bar_init(fin, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const CUtensorMap* dmaps[4] = {&d0map, &d1map, &d2map, &d3map};
  const int mytiles = (int)blockIdx.x < ntiles ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int xb = lt % NX;
        if (lt >= NX) wait(he + xb, ((lt / NX) - 1) & 1);
        expect_tx(xf + xb, XT * 2 + BM * NH * 4);
        for (int kh = 0; kh < 2; ++kh) load_2d(&xmap, sX + xb * XT + kh * BM * 64, xf + xb, kh * 64, t * BM);
        {                                           // the tile's db rows: 4 planes x 128 contiguous floats (512 B each)
          const long row0 = (long)t * BM, bb = row0 / ((long)L * L), rr = row0 % ((long)L * L);
          for (int h = 0; h < NH; ++h)
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(sa(sDb + (xb * NH + h) * BM)), "l"(DB + (bb * NH + h) * (long)L * L + rr), "r"(BM * 4), "r"(sa(xf + xb)) : "memory");
        }
        for (int q8 = 0; q8 < 8; ++q8, ++y) {
          const int sl = y % NR;
          if (y >= NR) wait(de + sl, ((y / NR) - 1) & 1);
          expect_tx(df + sl, HT * 2);
          load_2d(dmaps[q8 >> 1], sD + sl * HT, df + sl, (q8 & 1) * 64, t * BM);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int xb = lt % NX;
        wait(hf + xb, (lt / NX) & 1);
        for (int q8 = 0; q8 < 8; ++q8, ++y) {
          const int sl = y % NR, ten = q8 >> 1, kh = q8 & 1;
          wait(df + sl, (y / NR) & 1);
          tc_fence_after();
#pragma unroll
          for (int ks = 0; ks < (WG_NOMMA ? 0 : BM / 16); ++ks)      // M_t[:, 64 kh ..] += xhat^T . d_t-half   (K = rows)
            mma_ss(tmem + ten * 128 + kh * 64, desc_mn128(sX + xb * XT + ks * 16 * 64, BM * 128), desc_mn128(sD + sl * HT + ks * 16 * 64, BM * 128),
                   IDESC, (lt | ks) ? 1u : 0u);
          mma_commit(de + sl);
        }
        mma_commit(he + xb);
        if (lt == mytiles - 1) mma_commit(fin);
      }
    }
  } else if (warp < 6) {
    // ---- a row a thread: statistics, xhat (bf16) over x ----
    const int r = (warp - 2) * 32 + lane;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int xb = lt % NX;
      wait(xf + xb, (lt / NX) & 1);
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c8 = 0; c8 < 16; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(sX + xb * XT + (c8 >> 3) * BM * 64 + r * 64 + (((c8 & 7) ^ (r & 7)) << 3));
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); s1 += v.x + v.y; s2 = fmaf(v.x, v.x, fmaf(v.y, v.y, s2)); }
      }
      const float mean = s1 * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int c8 = 0; c8 < 16; ++c8) {
        __nv_bfloat16* p = sX + xb * XT + (c8 >> 3) * BM * 64 + r * 64 + (((c8 & 7) ^ (r & 7)) << 3);
        const uint4 u = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); o[e] = pack2((v.x - mean) * rstd, (v.y - mean) * rstd); }
        *reinterpret_cast<uint4*>(p) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(hf + xb);
    }
  } else {
    // ---- a channel a thread: column sums of the d half-tiles; B_h = sum db_h xhat_c, cdb_h ----
    const int c = (warp - 6) * 32 + lane;          // 0..127
    float cs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   // (tensor, half) of column c & 63 ... per half-tile column c % 64, row half c / 64
    float bh[NH] = {0.f, 0.f, 0.f, 0.f}, cdb[NH] = {0.f, 0.f, 0.f, 0.f};
    const int col = c & 63, rh = c >> 6;           // a half-tile's column, and which 64 of its rows this thread sums
    int lt = 0, y = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int xb = lt % NX;
      wait(hf + xb, (lt / NX) & 1);                 // xhat written (and x / db landed)
      const __nv_bfloat16* xh = sX + xb * XT + (c >> 6) * BM * 64;
      const float* dbp = sDb + xb * NH * BM;
#pragma unroll 4
      for (int rr = 0; rr < (WG_NOBH ? 0 : BM); ++rr) {
        const float xv = __bfloat162float(xh[rr * 64 + ((((c & 63) >> 3) ^ (rr & 7)) << 3) + (c & 7)]);
#pragma unroll
        for (int h = 0; h < NH; ++h) { const float d = dbp[h * BM + rr]; bh[h] = fmaf(d, xv, bh[h]); if (c == 0) cdb[h] += d; }
      }
      __syncwarp();
      if (lane == 0) arrive(he + xb);
      for (int q8 = 0; q8 < 8; ++q8, ++y) {
        const int sl = y % NR;
        wait(df + sl, (y / NR) & 1);
        const __nv_bfloat16* dt = sD + sl * HT;
        float sacc = 0.f;
#pragma unroll 8
        for (int rr = rh * 64; rr < rh * 64 + (WG_NOCS ? 0 : 64); ++rr) sacc += __bfloat162float(dt[rr * 64 + (((col >> 3) ^ (rr & 7)) << 3) + (col & 7)]);
        cs[q8] += sacc;
        __syncwarp();
        if (lane == 0) arrive(de + sl);
      }
    }
    // cs[q8] holds rows rh*64.. of column col of (tensor q8 >> 1, half q8 & 1): o = (q8 & 1) * 64 + col
#pragma unroll
    for (int q8 = 0; q8 < 8; ++q8) atomicAdd(VEC + (q8 >> 1) * C + (q8 & 1) * 64 + col, cs[q8]);
#pragma unroll
    for (int h = 0; h < NH; ++h) atomicAdd(VEC + 4 * C + h * C + c, bh[h]);
    if (c == 0) {
#pragma unroll
      for (int h = 0; h < NH; ++h) atomicAdd(VEC + 8 * C + h, cdb[h]);
    }
    // ---- the CTA's M_t: TMEM lane = c, 512 columns -> global ----
    if (mytiles > 0) {
      wait(fin, 0);
      tc_fence_after();
      const int qq = warp & 3, rl = qq * 32 + lane;
#pragma unroll 1
      for (int cq = 0; cq < 16; ++cq) {
        float v[32];
        tmem_ld32(tmem_at(tmem + cq * 32, qq * 32, 0), v);
        tmem_wait_ld();
        float* dst = MT + ((long)(cq >> 2) * C + rl) * C + (cq & 3) * 32;
#pragma unroll
        for (int i = 0; i < 32; i += 4)
          asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(dst + i), "f"(v[i]), "f"(v[i + 1]), "f"(v[i + 2]), "f"(v[i + 3]) : "memory");
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
  (void)want_finish; (void)TICKET; (void)W4; (void)LNW; (void)LNB; (void)WBF; (void)DW; (void)DGB;
}

// The parameter gradients from tri_wgrad's sums (the algebra in its header comment), one launch: blocks 0..63 write dW_t[o][c] through
// 32 x 32 shared transposes (M_t is [c][o]); blocks 64..191 one channel c each (dgamma, dbeta, dWb[:, c]), a block-wide reduction.
__global__ void __launch_bounds__(256) tri_param_finish(const float* __restrict__ MT, const float* __restrict__ VEC,
    const __nv_bfloat16* __restrict__ W4, const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WBF,
    __nv_bfloat16* __restrict__ DW, __nv_bfloat16* __restrict__ DGB, const float* __restrict__ DWO, __nv_bfloat16* __restrict__ DWO16) {
  const int tid = threadIdx.x;
  if (blockIdx.x >= 64 + C) {                      // blocks 192..207: gate_bwd's dWo fp32 -> bf16
    const int i0 = (blockIdx.x - 64 - C) * 1024;
    for (int i = tid; i < 1024; i += 256) if (DWO != nullptr) DWO16[i0 + i] = __float2bfloat16_rn(DWO[i0 + i]);
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
      DW[((long)tt * C + o) * C + c] = __float2bfloat16_rn(LNW[c] * tile[cc][oo] + LNB[c] * cs[tt * C + o]);
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
      DGB[2 * C + h * C + c] = __float2bfloat16_rn(LNW[c] * bh[h * C + c] + LNB[c] * cdb[h]);
    }
    DGB[c] = __float2bfloat16_rn(g);
    DGB[C + c] = __float2bfloat16_rn(b);
  }
}

}  // namespace

// dq, dk, dv, dg [R, 128] bf16; db fp32 [B, 4, L, L]; x [R, 128] (the LN input); w4 [512, 128]; wb [4, 128]; dpair [R, 128] bf16 (the
// seed gate_bwd wrote; dx is added in place)
void tri_head_bwd(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                  torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor dpair, int64_t B, int64_t L) {
  const long R = B * L * L;
  for (const auto* t : {&dq, &dk, &dv, &dg, &x, &dpair}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.size(0) == 4 * C, "w4 [512, 128] bf16");
  auto prm = torch::cat({lnw.to(torch::kFloat32).reshape(-1), lnb.to(torch::kFloat32).reshape(-1), wb.to(torch::kFloat32).reshape(-1)}).contiguous();
  TORCH_CHECK(prm.numel() * 4 == (long)sizeof(HbParams), "lnw / lnb / wb sizes");
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_hb, prm.data_ptr<float>(), sizeof(HbParams), 0, cudaMemcpyDeviceToDevice, stream));
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, R, 128, "dq"), m1 = m2(dk, R, 128, "dk"), m2_ = m2(dv, R, 128, "dv"), m3 = m2(dg, R, 128, "dg"), xm = m2(x, R, 128, "x");
  CUtensorMap wm = m2(w4, 4 * C, 256, "w4"), rm = m2(dpair, R, 128, "dpair");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_head_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, hb::SMEM)); attr = true; }
  const int ntiles = (int)(R / hb::BM);
  tri_head_bwd_sm100<<<std::min(ntiles, num_sms(x.device().index())), hb::THREADS, hb::SMEM, stream>>>(
      (int)L, ntiles, (float)eps, m0, m1, m2_, m3, xm, wm, rm, db.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// finish: -> (dW_q|k|v|g [4, 128, 128] bf16 ([o, c] each, the Linear weight layout), dgamma | dbeta | dWb [6, 128] bf16);
// raw (finish = false): -> (M [4, 128 c, 128 o], cs [4, 128], B [4, 128], cdb [4]) fp32; see tri_wgrad_sm100
std::vector<torch::Tensor> tri_wgrad(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                                     double eps, int64_t B, int64_t L, torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, torch::Tensor lnb, bool finish,
                                     torch::Tensor dwo32) {
  const long R = B * L * L;
  for (const auto* t : {&dq, &dk, &dv, &dg, &x}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  auto opt = x.options().dtype(torch::kFloat32);
  auto mt = torch::zeros({4, C, C}, opt), vec = torch::zeros({8 * C + NH}, opt);
  static torch::Tensor ticket;
  if (!ticket.defined() || ticket.device() != x.device()) ticket = torch::zeros({1}, x.options().dtype(torch::kInt32));
  auto dw = torch::empty({4, C, C}, x.options()), dgb = torch::empty({2 + NH, C}, x.options());
  torch::Tensor lw, lb, wbf;
  if (finish) {
    TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.numel() == 4 * C * C, "w4 [512, 128] bf16");
    lw = lnw.to(torch::kFloat32).contiguous(); lb = lnb.to(torch::kFloat32).contiguous(); wbf = wb.to(torch::kFloat32).contiguous();
  }
  auto m2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, (uint64_t)R}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, "dq"), m1 = m2(dk, "dk"), m2_ = m2(dv, "dv"), m3 = m2(dg, "dg"), xm = m2(x, "x");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wgrad_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, wg::SMEM)); attr = true; }
  const int ntiles = (int)(R / wg::BM);
  tri_wgrad_sm100<<<std::min(ntiles, num_sms(x.device().index())), wg::THREADS, wg::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, (float)eps, m0, m1, m2_, m3, xm, db.data_ptr<float>(), mt.data_ptr<float>(), vec.data_ptr<float>(),
      reinterpret_cast<unsigned*>(ticket.data_ptr<int>()), finish ? reinterpret_cast<const __nv_bfloat16*>(w4.data_ptr<at::BFloat16>()) : nullptr,
      finish ? lw.data_ptr<float>() : nullptr, finish ? lb.data_ptr<float>() : nullptr, finish ? wbf.data_ptr<float>() : nullptr,
      reinterpret_cast<__nv_bfloat16*>(dw.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dgb.data_ptr<at::BFloat16>()), 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (finish) {
    const bool wo_ = dwo32.numel() == C * C;
    auto dwo16 = torch::empty({wo_ ? C : 0, C}, x.options());
    tri_param_finish<<<64 + C + (wo_ ? 16 : 0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(mt.data_ptr<float>(), vec.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(w4.data_ptr<at::BFloat16>()), lw.data_ptr<float>(), lb.data_ptr<float>(), wbf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dw.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dgb.data_ptr<at::BFloat16>()),
        wo_ ? dwo32.data_ptr<float>() : nullptr, wo_ ? reinterpret_cast<__nv_bfloat16*>(dwo16.data_ptr<at::BFloat16>()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dw, dgb, dwo16};
  }
  return {mt, vec.slice(0, 0, 4 * C).view({4, C}), vec.slice(0, 4 * C, 8 * C).view({NH, C}), vec.slice(0, 8 * C, 8 * C + NH)};
}

// dy, g, o [R, 128] bf16 (rows (b, i, j)), wo [128, 128] -> (do, dg [R, 128] bf16, delta fp32 [B, L, 4, L], dWo fp32 [128, 128] (tri_wgrad's
// finish converts it),
// dpair [R, 128] bf16 = dy: the seed head_bwd adds dx into)
std::vector<torch::Tensor> tri_gate_bwd(torch::Tensor dy, torch::Tensor g, torch::Tensor o, torch::Tensor wo, int64_t B, int64_t L) {
  for (const auto* t : {&dy, &g, &o}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == B * L * L * C, "dy / g / o: [R, 128] bf16");
  const long R = B * L * L;
  TORCH_CHECK(L % gb::BM == 0, "L % 128");
  auto opt = dy.options();
  auto dO = torch::empty({R, C}, opt), dG = torch::empty({R, C}, opt);
  auto delta = torch::empty({B, L, 4, L}, opt.dtype(torch::kFloat32));
  auto dwo = torch::zeros({C, C}, opt.dtype(torch::kFloat32));
  auto dwo16 = torch::empty({C, C}, opt), seed = torch::empty({R, C}, opt);
  static torch::Tensor ticket;
  if (!ticket.defined() || ticket.device() != dy.device()) ticket = torch::zeros({1}, opt.dtype(torch::kInt32));
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap dym = m2(dy, R, "dy"), gm = m2(g, R, "g"), om = m2(o, R, "o"), wm = m2(wo, C, "wo"), dom = m2(dO, R, "do"), dgm = m2(dG, R, "dg"), sm_ = m2(seed, R, "seed");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_gate_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, gb::SMEM)); attr = true; }
  const int ntiles = (int)(R / gb::BM);
  tri_gate_bwd_sm100<<<std::min(ntiles, num_sms(dy.device().index())), gb::THREADS, gb::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, dym, gm, om, wm, dom, dgm, sm_, delta.data_ptr<float>(), dwo.data_ptr<float>(), reinterpret_cast<unsigned*>(ticket.data_ptr<int>()),
      reinterpret_cast<__nv_bfloat16*>(dwo16.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dG, delta, dwo, seed};
}

// out = x + (sigmoid(g) o o) . Wo^T; g, o, x [R, 128] bf16, wo [128, 128] bf16
torch::Tensor tri_tail(torch::Tensor g, torch::Tensor o, torch::Tensor x, torch::Tensor wo) {
  for (const auto* t : {&g, &o, &x}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == g.numel(), "g / o / x: [R, 128] bf16");
  const long R = g.numel() / C;
  TORCH_CHECK(R % tl::BM == 0, "rows % 128");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == C * C, "wo [128, 128] bf16");
  auto out = torch::empty({R, C}, g.options());
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap gm = m2(g, R, "g"), om = m2(o, R, "o"), xm = m2(x, R, "x"), wm = m2(wo, C, "wo");
  CUtensorMap ym = make_map<2>(out.data_ptr(), {(uint64_t)C, (uint64_t)R}, {(uint64_t)C}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_64B, "out");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_tail_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, tl::SMEM)); attr = true; }
  const int ntiles = (int)(R / tl::BM);
  tri_tail_sm100<<<std::min(ntiles, num_sms(g.device().index())), tl::THREADS, tl::SMEM, at::cuda::getCurrentCUDAStream()>>>(ntiles, gm, om, xm, wm, ym);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// x [R, 128] bf16 (R = B * L * L rows (b, j, k)); w4 = [Wq; Wk; Wv; Wg] [512, 128] bf16; wb [4, 128]; mask [B, L] bool or empty
// -> (q, k, v, g [R, 128] bf16, bias [B, 4, L, L] bf16 with masked keys at -inf)
std::vector<torch::Tensor> tri_front(torch::Tensor x, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor w4, torch::Tensor wb,
                                     torch::Tensor mask, int64_t B, int64_t L) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.size(-1) == C, "x: [R, 128] bf16");
  const long R = x.numel() / C;
  TORCH_CHECK(R == B * L * L && L % fr::BM == 0, "rows = B * L * L, L % 128 == 0");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.size(0) == 4 * C && w4.size(1) == C, "w4 [512, 128] bf16");
  auto opt = x.options();
  auto q = torch::empty({R, C}, opt), k = torch::empty({R, C}, opt), v = torch::empty({R, C}, opt), g = torch::empty({R, C}, opt);
  auto bias = torch::empty({B, NH, L, L}, opt);
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous(), wbf = wb.to(torch::kFloat32).contiguous();
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap xm = m2(x, R, 128, "x"), wm = m2(w4, 4 * C, 256, "w4");
  auto o2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, (uint64_t)R}, {(uint64_t)C}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_64B, what); };
  CUtensorMap qm = o2(q, "q"), km = o2(k, "k"), vm = o2(v, "v"), gm = o2(g, "g");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, fr::SMEM)); attr = true; }
  const int ntiles = (int)(R / fr::BM);
  tri_front_sm100<<<std::min(ntiles, num_sms(x.device().index())), fr::THREADS, fr::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)R, (int)L, ntiles, (float)eps, xm, wm, qm, km, vm, gm, lw.data_ptr<float>(), lb.data_ptr<float>(), wbf.data_ptr<float>(), mp,
      reinterpret_cast<__nv_bfloat16*>(bias.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, v, g, bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_head_bwd", &tri_head_bwd, "sm100 TriangleAttention head backward: projection dgrad + LN backward + residual (added into dpair in place)",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("dg"), py::arg("db"), py::arg("x"), py::arg("w4"), py::arg("wb"), py::arg("lnw"), py::arg("lnb"),
        py::arg("eps"), py::arg("dpair"), py::arg("B"), py::arg("L"));
  m.def("tri_wgrad", &tri_wgrad, "sm100 TriangleAttention parameter gradients: finish -> (dW_q|k|v|g [4,128,128], dgamma|dbeta|dWb [6,128]) bf16; raw -> (M, cs, B, cdb)",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("dg"), py::arg("db"), py::arg("x"), py::arg("eps"), py::arg("B"), py::arg("L"),
        py::arg("w4"), py::arg("wb"), py::arg("lnw"), py::arg("lnb"), py::arg("finish") = true, py::arg("dwo32") = torch::Tensor());
  m.def("tri_gate_bwd", &tri_gate_bwd, "sm100 TriangleAttention gate backward: (do, dg, delta, dWo)", py::arg("dy"), py::arg("g"), py::arg("o"), py::arg("wo"), py::arg("B"), py::arg("L"));
  m.def("tri_tail", &tri_tail, "sm100 TriangleAttention tail: x + (sigmoid(g) o o) . Wo^T", py::arg("g"), py::arg("o"), py::arg("x"), py::arg("wo"));
  m.def("tri_front", &tri_front, "sm100 TriangleAttention front: LN + q|k|v|g projections + head-major (masked) bias",
        py::arg("x"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("w4"), py::arg("wb"), py::arg("mask"), py::arg("B"), py::arg("L"));
}
