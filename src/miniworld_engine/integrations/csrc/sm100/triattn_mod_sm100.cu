// TriangleAttention module kernels around the attention core for B200 (sm_100a) -- the H100 kernel boundaries (LN + projections,
// gate + out-projection, gate backward + delta, projection dgrad + LN backward), developed on tcgen05 / TMEM / TMA.
//
// front:  y = LN(x) (never stored); q | k | v | g = y . W^T  (four 128 x 128 projections, bf16, the [rows][H*32] projection layout
//         the attention kernels read); bias[b, h, j, k] = y . Wb[h]  (head-major for the attention, masked keys = -inf)
#include <torch/extension.h>
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
    float* __restrict__ DELTA,                            // [B, L (i), 4, L (j)]
    float* __restrict__ DWO) {                            // [128 out][128 in] fp32, accumulated
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
        if (lt >= 1) wait(dwd, (lt - 1) & 1);
        expect_tx(af, 2 * XT * 2);
        for (int kh = 0; kh < 2; ++kh) {
          load_2d(&dymap, sDy + kh * BM * 64, af, kh * 64, t * BM);
          load_2d(&omap, sO + kh * BM * 64, af, kh * 64, t * BM);
        }
        if (lt >= 1) wait(sd, (lt - 1) & 1);
        expect_tx(bfb, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&gmap, sG + kh * BM * 64, bfb, kh * 64, t * BM);
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
}
// ---------------------------------------------------------------------------------------------------------------------------------
// head backward:  gy = sum_t d_t . W_t + sum_h db_h Wb_h   (the five projections' input gradient: q | k | v | g via tcgen05, the
//                 4-wide bias one on the CUDA cores);  LN backward from the recomputed statistics;  dpair = dres + dx (a bf16 TMA
//                 reduce-add into dres, in place);  dgamma / dbeta (warp reduce-scatter + shared accumulators);  y = LN(x) for the
//                 weight gradients (cuBLAS, as the H100 split).
// Tile = 128 rows.  warp 0: TMA (the d_t half-tiles [128][64] in a ring, x, W once); warp 1: tcgen05 (M = 128 rows, N = 128, K = 512;
// double-buffered TMEM); warps 2-5: one row a thread (TMEM lane): three passes over the row (statistics; gy + the row sums + the
// dgamma / dbeta terms; y and dx), y through [128][32] staging tiles by TMA store, dx over x in place (the reduce-add's source).
namespace hb {
constexpr int BM = 128, XT = BM * C, HT = BM * 64;         // x tile (32 KiB), a d half-tile [128][64] (16 KiB)
constexpr int NR = 3;                                      // d ring
constexpr int OT = BM * 32;
constexpr int THREADS = 192;
constexpr int SMEM = 1024 + (4 * C * C + NR * HT + XT + 2 * OT) * 2 + 2 * C * 4 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, 128, 0, 1);    // A = d_t K-major, B = W_t MN-major
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
    const __grid_constant__ CUtensorMap rmap,             // dres [R][128] (reduce-add target, becomes dpair), box (64, 128), 128B
    const __grid_constant__ CUtensorMap ymap,             // y [R][128], box (32, 128), 64B swizzle (TMA stores)
    const float* __restrict__ DB,                         // dbias [B, 4, L, L] fp32
    const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WB,
    float* __restrict__ DGAM, float* __restrict__ DBET) { // [128] fp32, accumulated
  using namespace hb;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(smb);    // [2 c halves][512 o][64]
  __nv_bfloat16* sD = sW + 4 * C * C;                            // [NR][128][64]
  __nv_bfloat16* sX = sD + NR * HT;                              // [2 halves][128][64]
  __nv_bfloat16* sY = sX + XT;                                   // [2][OT]
  float* sGB = reinterpret_cast<float*>(sY + 2 * OT);            // dgamma | dbeta partials [2][128]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sGB + 2 * C);
  uint64_t* df = bars;               // [NR]
  uint64_t* de = df + NR;            // [NR]
  uint64_t* xf = de + NR;            // x landed
  uint64_t* xe = xf + 1;             // count 1: dx reduce read the tile
  uint64_t* af = xe + 1;             // [2]
  uint64_t* ae = af + 2;             // [2] count 4
  uint64_t* wf = ae + 2;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NR; ++i) { bar_init(df + i, 1); bar_init(de + i, 1); }
    bar_init(xf, 1); bar_init(xe, 1); bar_init(wf, 1);
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init_fence();
  }
  for (int i = tid; i < 2 * C; i += blockDim.x) sGB[i] = 0.f;
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
        if (lt >= 1) wait(xe, (lt - 1) & 1);
        expect_tx(xf, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&xmap, sX + kh * BM * 64, xf, kh * 64, t * BM);
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
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)            // K = o: 16 rows of W_t per step
            mma_ss(tmem + b * 128, desc_k128(sD + sl * HT + kk * 16), desc_mn128(sW + (ten * C + kh * 64 + kk * 16) * 64, 4 * C * 128), IDESC,
                   (q8 | kk) ? 1u : 0u);
          mma_commit(de + sl);
        }
        mma_commit(af + b);
      }
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;
    const bool leader = warp == 2 && lane == 0;
    float gam[1] = {0.f};
    (void)gam;
    int lt = 0, nst = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int b = lt & 1;
      wait(xf, lt & 1);
      // ---- pass 1: statistics of x's row ----
      const __nv_bfloat16* xr0 = sX + r * 64;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c8 = 0; c8 < 16; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr0 + (c8 >> 3) * BM * 64 + (((c8 & 7) ^ (r & 7)) << 3));
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); s1 += v.x + v.y; s2 = fmaf(v.x, v.x, fmaf(v.y, v.y, s2)); }
      }
      const float mean = s1 * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
      const long row = (long)t * BM + r;           // (b, r1, r2): dbias[b, h, r1, r2]
      const long bb = row / ((long)L * L), rr = row % ((long)L * L);
      float dbv[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) dbv[h] = DB[(bb * NH + h) * (long)L * L + rr];
      wait(af + b, (lt >> 1) & 1);
      tc_fence_after();
      // ---- pass 2: gy = TMEM + db . Wb; the row sums of gamma gy and gamma gy xhat; dgamma / dbeta terms reduce-scattered ----
      float sg = 0.f, sgx = 0.f;
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        float gy[32];
        tmem_ld32(tmem_at(tmem + b * 128 + ch * 32, qq * 32, 0), gy);
        const __nv_bfloat16* xr = sX + (ch >> 1) * BM * 64 + r * 64;
        float xh[32];
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) {
          const uint4 u = *reinterpret_cast<const uint4*>(xr + ((((ch & 1) * 4 + c4) ^ (r & 7)) << 3));
          const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); xh[c4 * 8 + e * 2] = (v.x - mean) * rstd; xh[c4 * 8 + e * 2 + 1] = (v.y - mean) * rstd; }
        }
        tmem_wait_ld();
        float pg[32], pb[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const int c = ch * 32 + i;
          float v = gy[i];
#pragma unroll
          for (int h = 0; h < NH; ++h) v = fmaf(dbv[h], __ldg(WB + h * C + c), v);
          const float gv = v * __ldg(LNW + c);
          sg += gv; sgx = fmaf(gv, xh[i], sgx);
          pb[i] = v; pg[i] = v * xh[i];
        }
        // reduce-scatter over the warp's 32 rows: lane l ends with channel ch * 32 + l's sums
#pragma unroll
        for (int w5 = 16; w5 >= 1; w5 >>= 1) {
#pragma unroll
          for (int i = 0; i < w5; ++i) {
            const bool up = (lane & w5) != 0;
            const float sendg = up ? pg[i] : pg[i + w5], sendb = up ? pb[i] : pb[i + w5];
            const float rg = __shfl_xor_sync(0xffffffffu, sendg, w5), rb = __shfl_xor_sync(0xffffffffu, sendb, w5);
            pg[i] = (up ? pg[i + w5] : pg[i]) + rg;
            pb[i] = (up ? pb[i + w5] : pb[i]) + rb;
          }
        }
        atomicAdd(sGB + ch * 32 + lane, pg[0]);
        atomicAdd(sGB + C + ch * 32 + lane, pb[0]);
      }
      const float mg = sg * (1.f / C), mgx = sgx * (1.f / C);
      // ---- pass 3: y (staging -> TMA store) and dx (over x, in place) ----
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        float gy[32];
        tmem_ld32(tmem_at(tmem + b * 128 + ch * 32, qq * 32, 0), gy);
        tmem_wait_ld();
        if (ch == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + b); }
        __nv_bfloat16* xr = sX + (ch >> 1) * BM * 64 + r * 64;
        uint32_t yw[16], dw[16];
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) {
          const int off = (((ch & 1) * 4 + c4) ^ (r & 7)) << 3;
          const uint4 u = *reinterpret_cast<const uint4*>(xr + off);
          const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 v = bf2f(w[e]);
            const int i = c4 * 8 + e * 2, c = ch * 32 + i;
            float xh0 = (v.x - mean) * rstd, xh1 = (v.y - mean) * rstd;
            float g0 = gy[i], g1 = gy[i + 1];
#pragma unroll
            for (int h = 0; h < NH; ++h) { g0 = fmaf(dbv[h], __ldg(WB + h * C + c), g0); g1 = fmaf(dbv[h], __ldg(WB + h * C + c + 1), g1); }
            const float w0 = __ldg(LNW + c), w1 = __ldg(LNW + c + 1);
            yw[c4 * 4 + e] = pack2(fmaf(xh0, w0, __ldg(LNB + c)), fmaf(xh1, w1, __ldg(LNB + c + 1)));
            dw[c4 * 4 + e] = pack2(rstd * (g0 * w0 - mg - xh0 * mgx), rstd * (g1 * w1 - mg - xh1 * mgx));
          }
          *reinterpret_cast<uint4*>(xr + off) = make_uint4(dw[c4 * 4], dw[c4 * 4 + 1], dw[c4 * 4 + 2], dw[c4 * 4 + 3]);
        }
        __nv_bfloat16* buf = sY + (nst & 1) * OT;
        if (leader && nst >= 2) bulk_wait_read<1>();
        named_sync(1, 128);
        unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + r * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c)
          *reinterpret_cast<uint4*>(rowp + ((c ^ ((r >> 1) & 3)) << 4)) = make_uint4(yw[c * 4], yw[c * 4 + 1], yw[c * 4 + 2], yw[c * 4 + 3]);
        fence_proxy_async();
        named_sync(1, 128);
        if (leader) { store_2d(&ymap, buf, ch * 32, t * BM); bulk_commit(); }
        ++nst;
      }
      if (leader) {                                 // dpair = dres + dx
        for (int kh = 0; kh < 2; ++kh) reduce_add_2d_bf(&rmap, sX + kh * BM * 64, kh * 64, t * BM);
        bulk_commit();
        bulk_wait_read<0>();
        arrive(xe);
      }
    }
    named_sync(1, 128);
    if (tid >= 64 && tid < 64 + 2 * C / 2) {}      // (placeholder)
    for (int i = tid - 64; i < 2 * C; i += 128) { if (i >= 0) atomicAdd((i < C ? DGAM + i : DBET + i - C), sGB[i]); }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}
}  // namespace

// dq, dk, dv, dg [R, 128] bf16; db fp32 [B, 4, L, L]; x [R, 128] (the LN input); w4 [512, 128]; wb [4, 128]; dres [R, 128] bf16 (modified in
// place: becomes dpair = dres + LN_bwd(...)) -> (y [R, 128] bf16, dgamma, dbeta fp32 [128])
std::vector<torch::Tensor> tri_head_bwd(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                                        torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor dres,
                                        int64_t B, int64_t L) {
  const long R = B * L * L;
  for (const auto* t : {&dq, &dk, &dv, &dg, &x, &dres}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.size(0) == 4 * C, "w4 [512, 128] bf16");
  auto opt = x.options();
  auto y = torch::empty({R, C}, opt);
  auto dgam = torch::zeros({C}, opt.dtype(torch::kFloat32)), dbet = torch::zeros({C}, opt.dtype(torch::kFloat32));
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous(), wbf = wb.to(torch::kFloat32).contiguous();
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, R, 128, "dq"), m1 = m2(dk, R, 128, "dk"), m2_ = m2(dv, R, 128, "dv"), m3 = m2(dg, R, 128, "dg"), xm = m2(x, R, 128, "x");
  CUtensorMap wm = m2(w4, 4 * C, 256, "w4"), rm = m2(dres, R, 128, "dres");
  CUtensorMap ym = make_map<2>(y.data_ptr(), {(uint64_t)C, (uint64_t)R}, {(uint64_t)C}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_64B, "y");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_head_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, hb::SMEM)); attr = true; }
  const int ntiles = (int)(R / hb::BM);
  tri_head_bwd_sm100<<<std::min(ntiles, num_sms(x.device().index())), hb::THREADS, hb::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, (float)eps, m0, m1, m2_, m3, xm, wm, rm, ym, db.data_ptr<float>(), lw.data_ptr<float>(), lb.data_ptr<float>(), wbf.data_ptr<float>(),
      dgam.data_ptr<float>(), dbet.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, dgam, dbet};
}

// dy, g, o [R, 128] bf16 (rows (b, i, j)), wo [128, 128] -> (do, dg [R, 128] bf16, delta fp32 [B, L, 4, L], dWo fp32 [128, 128])
std::vector<torch::Tensor> tri_gate_bwd(torch::Tensor dy, torch::Tensor g, torch::Tensor o, torch::Tensor wo, int64_t B, int64_t L) {
  for (const auto* t : {&dy, &g, &o}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == B * L * L * C, "dy / g / o: [R, 128] bf16");
  const long R = B * L * L;
  TORCH_CHECK(L % gb::BM == 0, "L % 128");
  auto opt = dy.options();
  auto dO = torch::empty({R, C}, opt), dG = torch::empty({R, C}, opt);
  auto delta = torch::empty({B, L, 4, L}, opt.dtype(torch::kFloat32));
  auto dwo = torch::zeros({C, C}, opt.dtype(torch::kFloat32));
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap dym = m2(dy, R, "dy"), gm = m2(g, R, "g"), om = m2(o, R, "o"), wm = m2(wo, C, "wo"), dom = m2(dO, R, "do"), dgm = m2(dG, R, "dg");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_gate_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, gb::SMEM)); attr = true; }
  const int ntiles = (int)(R / gb::BM);
  tri_gate_bwd_sm100<<<std::min(ntiles, num_sms(dy.device().index())), gb::THREADS, gb::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, dym, gm, om, wm, dom, dgm, delta.data_ptr<float>(), dwo.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dG, delta, dwo};
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
  m.def("tri_head_bwd", &tri_head_bwd, "sm100 TriangleAttention head backward: projection dgrad + LN backward + residual (in place into dres) -> (y, dgamma, dbeta)",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("dg"), py::arg("db"), py::arg("x"), py::arg("w4"), py::arg("wb"), py::arg("lnw"), py::arg("lnb"),
        py::arg("eps"), py::arg("dres"), py::arg("B"), py::arg("L"));
  m.def("tri_gate_bwd", &tri_gate_bwd, "sm100 TriangleAttention gate backward: (do, dg, delta, dWo)", py::arg("dy"), py::arg("g"), py::arg("o"), py::arg("wo"), py::arg("B"), py::arg("L"));
  m.def("tri_tail", &tri_tail, "sm100 TriangleAttention tail: x + (sigmoid(g) o o) . Wo^T", py::arg("g"), py::arg("o"), py::arg("x"), py::arg("wo"));
  m.def("tri_front", &tri_front, "sm100 TriangleAttention front: LN + q|k|v|g projections + head-major (masked) bias",
        py::arg("x"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("w4"), py::arg("wb"), py::arg("mask"), py::arg("B"), py::arg("L"));
}
