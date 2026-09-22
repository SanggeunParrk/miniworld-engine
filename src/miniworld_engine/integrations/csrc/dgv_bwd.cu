// Fused PWA backward tail (sm_90a): reads the [M][512] dgv buffer ONCE and produces
//   dy    = dgv . Wgv                     (kept in registers, never stored)
//   dWgv += dgv^T . y                     [512][64] fp32, accumulated in registers per block, reduced with atomics at the end
//   dm    = LayerNorm_bwd(dy; x, gamma) + dout      [M][64] bf16   (the residual gradient folded in)
//   dgamma = sum_rows dy * xhat,  dbeta = sum_rows dy
// A 384-thread block: two consumer warpgroups share a 64-row tile (dy columns split, dWgv m-tiles alternated) and a
// producer warpgroup (one thread) streams the tiles; the block walks a persistent tile list.  Per tile the 512
// columns of dgv come in 8 k-blocks of [64 rows][64] (8 KiB, 128-byte swizzle) through an NST-deep TMA
// pipeline with full/empty mbarriers.  Each k-block feeds two wgmma chains: the dy GEMM (both warpgroups,
// 32 output columns each, A K-major) and the dWgv m-tile that k-block IS (the warpgroup kb%2 owns it,
// A = the same tile read MN-major i.e. transposed, B = the y tile MN-major).  y / x / dout are double-
// buffered one tile ahead.  The LayerNorm backward runs on the register dy with row sums exchanged in smem.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace {
constexpr int D = 64, KD = 512, KB = KD / 64;
constexpr int BM = 64;
constexpr int TILE = 64 * 64;
constexpr int THREADS = 384;                             // two consumer warpgroups + one producer warpgroup

namespace wg {
__device__ __forceinline__ int off(int r, int kc) { return r * 64 + ((kc ^ (r & 7)) << 3); }
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(64) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
// both K-major, n = 32: the dy GEMM (a warpgroup owns 32 of the 64 output columns)
__device__ __forceinline__ void mma_m64n32k16_kk(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %16, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15},\n"
      " %17, %18, p, 1, 1, 0, 0;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
// A MN-major (the dgv tile read transposed), B MN-major (the y tile): the dWgv m-tile
__device__ __forceinline__ void mma_m64n64k16_mm(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31},\n"
      " %33, %34, p, 1, 1, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}

__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory"); }
__device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n" ::: "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
}  // namespace wg

namespace tma {
__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra WAIT_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
}  // namespace tma

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float2 unpack2(const __nv_bfloat16* p) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p)); }

// smem carve-up (bf16 elements from a 1 KiB-aligned base)
constexpr int FRED = 0, FSTAT = FRED + 2 * 64 * 2, FEND = FSTAT + 64 * 2;          // floats after SEND
template <int NST> struct SMX {
  static constexpr int SDG = 0, SW = SDG + NST * TILE, SY = SW + KB * TILE, SX = SY + 2 * TILE, SDO = SX + 2 * TILE, SDM = SDO + 2 * TILE, SEND = SDM + TILE;
  static constexpr int NBAR = 2 * NST + 4;
  static constexpr int BYTES = SEND * 2 + FEND * 4 + NBAR * 8 + 1024;
};

// PARTIAL: each block writes its own dWgv slab [512][64] (summed on the host) instead of atomics into one
template <int NST, bool PARTIAL>
__global__ void __launch_bounds__(THREADS, 1) dgv_bwd_kernel(const float* __restrict__ lnw, float eps, int ntile, float* __restrict__ dW, float* __restrict__ dln,
                                                             const __grid_constant__ CUtensorMap dgmap, const __grid_constant__ CUtensorMap ymap,
                                                             const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap domap,
                                                             const __grid_constant__ CUtensorMap dmmap, const __grid_constant__ CUtensorMap wmap) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  unsigned char* sm = smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  using L = SMX<NST>;
  __nv_bfloat16* sDG = reinterpret_cast<__nv_bfloat16*>(sm) + L::SDG;
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(sm) + L::SW;
  __nv_bfloat16* sY = reinterpret_cast<__nv_bfloat16*>(sm) + L::SY;
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(sm) + L::SX;
  __nv_bfloat16* sDO = reinterpret_cast<__nv_bfloat16*>(sm) + L::SDO;
  __nv_bfloat16* sDM = reinterpret_cast<__nv_bfloat16*>(sm) + L::SDM;
  float* fred = reinterpret_cast<float*>(sm + L::SEND * 2) + FRED;      // [2 wg][64 rows][2]
  float* fstat = reinterpret_cast<float*>(sm + L::SEND * 2) + FSTAT;    // [64 rows][2]: mean, rstd
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + L::SEND * 2 + FEND * 4);
  uint64_t* empty = full + NST;
  uint64_t* fullT = empty + NST;                                      // [2]
  uint64_t* doneT = fullT + 2;                                        // consumers finished a tile's epilogue (its y/x/dout buffer is free)
  uint64_t* wbar = doneT + 1;
  const int tid = threadIdx.x, wgi = tid >> 7, lane = tid & 31, warp = tid >> 5;
  auto csync = [&]() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); };   // the two consumer warpgroups only

  auto tile_of = [&](int T) { return (int)blockIdx.x + T * (int)gridDim.x; };
  constexpr int LA = NST - 1;                              // k-blocks in flight: the producer simply blocks on the ring's empty barriers
  auto issueK = [&](int g) {                       // k-block g (tile g / KB, k-block g % KB) into stage g % NST
    const int t = tile_of(g / KB);
    if (t >= ntile) return;
    const int st = g % NST;
    if (g >= NST) tma::wait(empty + st, ((g / NST) - 1) & 1);
    tma::expect_tx(full + st, TILE * 2);
    tma::load_2d(&dgmap, tma::sa(sDG + st * TILE), full + st, (g % KB) * 64, t * BM);
  };
  auto issueT = [&](int T) {                       // tile T's y / x / dout into buffer T & 1
    const int t = tile_of(T);
    if (t >= ntile) return;
    const int b = T & 1;
    if (T >= 2) tma::wait(doneT, T & 1);           // buffer b was used by tile T-2: wait for that epilogue (phase T-2 -> parity (T-2)&1 = T&1)
    tma::expect_tx(fullT + b, 3 * TILE * 2);
    tma::load_2d(&ymap, tma::sa(sY + b * TILE), fullT + b, 0, t * BM);
    tma::load_2d(&xmap, tma::sa(sX + b * TILE), fullT + b, 0, t * BM);
    tma::load_2d(&domap, tma::sa(sDO + b * TILE), fullT + b, 0, t * BM);
  };
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { tma::bar_init(full + i, 1); tma::bar_init(empty + i, 8); }
    tma::bar_init(fullT, 1); tma::bar_init(fullT + 1, 1); tma::bar_init(doneT, 1); tma::bar_init(wbar, 1);
    tma::bar_init_fence();
    wg::proxy_fence();
  }
  __syncthreads();
  if (tid >= 256) {
    // ===================== producer warpgroup (one thread) =====================
    wg::reg_dealloc();
    if (tid == 256) {
      tma::expect_tx(wbar, KB * TILE * 2);
      for (int kb = 0; kb < KB; ++kb) tma::load_2d(&wmap, tma::sa(sW + kb * TILE), wbar, kb * 64, 0);
      issueT(0); issueT(1);
      for (int g = 0; g < LA; ++g) issueK(g);
      for (int T = 0; tile_of(T) < ntile; ++T) {
        for (int kb = 0; kb < KB; ++kb) issueK(T * KB + kb + LA);
        issueT(T + 2);
      }
    }
    return;
  }
  wg::reg_alloc();
  // accumulator geometry: rows r0 / r0+8 of the 64-row tile; dy columns wgi*32 + i*8 + (lane&3)*2 + j (i < 4)
  const int r0 = (warp & 3) * 16 + (lane >> 2), cq = (lane & 3) * 2;
  float gam[8];
  #pragma unroll
  for (int i = 0; i < 4; ++i) { gam[i * 2] = lnw[wgi * 32 + i * 8 + cq]; gam[i * 2 + 1] = lnw[wgi * 32 + i * 8 + cq + 1]; }
  float wacc[4][32];
  #pragma unroll
  for (int q = 0; q < 4; ++q)
    #pragma unroll
    for (int i = 0; i < 32; ++i) wacc[q][i] = 0.f;
  float dga[8], dbe[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { dga[i] = 0.f; dbe[i] = 0.f; }
  tma::wait(wbar, 0);

  for (int T = 0;; ++T) {
    const int t = tile_of(T);
    if (t >= ntile) break;
    const int b = T & 1;
    tma::wait(fullT + b, (T >> 1) & 1);
    {                                              // row statistics of x: 4 threads per row, 16 values each
      const int r = tid >> 2, q = tid & 3;
      float xv[16];
      #pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint4 u = *reinterpret_cast<const uint4*>(sX + b * TILE + wg::off(r, q * 2 + c));
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&u);
        #pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(p2[k]); xv[c * 8 + k * 2] = f.x; xv[c * 8 + k * 2 + 1] = f.y; }
      }
      float s = 0.f;
      #pragma unroll
      for (int k = 0; k < 16; ++k) s += xv[k];
      s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
      const float mean = s * (1.f / D);
      float ss = 0.f;
      #pragma unroll
      for (int k = 0; k < 16; ++k) { const float d = xv[k] - mean; ss += d * d; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      if (q == 0) { fstat[r * 2] = mean; fstat[r * 2 + 1] = rsqrtf(ss * (1.f / D) + eps); }
    }
    float dyacc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) dyacc[i] = 0.f;
    #pragma unroll
    for (int kb = 0; kb < KB; ++kb) {                 // fully unrolled: wacc[kb >> 1] must stay in registers
      const int g = T * KB + kb, st = g % NST;
      tma::wait(full + st, (g / NST) & 1);
#ifdef FENCE_PER_KBLOCK
      wg::fence();
#else
      if (kb == 0) wg::fence();
#endif
      const __nv_bfloat16* a = sDG + st * TILE;
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks)
        wg::mma_m64n32k16_kk(wg::desc(a + wg::off(0, ks * 2)), wg::desc(sW + kb * TILE + wgi * 32 * 64 + wg::off(0, ks * 2)), dyacc, 1);
      if ((kb & 1) == wgi) {
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          wg::mma_m64n64k16_mm(wg::desc(a + wg::off(ks * 16, 0)), wg::desc(sY + b * TILE + wg::off(ks * 16, 0)), wacc[kb >> 1], 1);
      }
      wg::commit();
      if (kb > 0) { wg::wait<1>(); if (lane == 0) tma::arrive(empty + ((g - 1) % NST)); }
    }
    wg::wait<0>();
    if (lane == 0) tma::arrive(empty + ((T * KB + KB - 1) % NST));
    // ---- LayerNorm backward on the register dy ----
    float gdy[16], xh[16];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 x0 = unpack2(sX + b * TILE + wg::off(r0, wgi * 4 + i) + cq), x1 = unpack2(sX + b * TILE + wg::off(r0 + 8, wgi * 4 + i) + cq);
      gdy[i * 4 + 0] = gam[i * 2] * dyacc[i * 4 + 0]; gdy[i * 4 + 1] = gam[i * 2 + 1] * dyacc[i * 4 + 1];
      gdy[i * 4 + 2] = gam[i * 2] * dyacc[i * 4 + 2]; gdy[i * 4 + 3] = gam[i * 2 + 1] * dyacc[i * 4 + 3];
      xh[i * 4 + 0] = x0.x; xh[i * 4 + 1] = x0.y; xh[i * 4 + 2] = x1.x; xh[i * 4 + 3] = x1.y;
    }
    csync();                                        // fstat visible
    const float m0 = fstat[r0 * 2], rs0 = fstat[r0 * 2 + 1], m1 = fstat[(r0 + 8) * 2], rs1 = fstat[(r0 + 8) * 2 + 1];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { xh[i * 4 + 0] = (xh[i * 4 + 0] - m0) * rs0; xh[i * 4 + 1] = (xh[i * 4 + 1] - m0) * rs0; xh[i * 4 + 2] = (xh[i * 4 + 2] - m1) * rs1; xh[i * 4 + 3] = (xh[i * 4 + 3] - m1) * rs1; }
    float a1 = 0.f, a2 = 0.f, b1 = 0.f, b2 = 0.f;   // row r0: sum gdy, sum gdy*xhat; row r0+8
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      a1 += gdy[i * 4 + 0] + gdy[i * 4 + 1]; a2 += gdy[i * 4 + 0] * xh[i * 4 + 0] + gdy[i * 4 + 1] * xh[i * 4 + 1];
      b1 += gdy[i * 4 + 2] + gdy[i * 4 + 3]; b2 += gdy[i * 4 + 2] * xh[i * 4 + 2] + gdy[i * 4 + 3] * xh[i * 4 + 3];
    }
    a1 += __shfl_xor_sync(0xffffffffu, a1, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    a2 += __shfl_xor_sync(0xffffffffu, a2, 1); a2 += __shfl_xor_sync(0xffffffffu, a2, 2);
    b1 += __shfl_xor_sync(0xffffffffu, b1, 1); b1 += __shfl_xor_sync(0xffffffffu, b1, 2);
    b2 += __shfl_xor_sync(0xffffffffu, b2, 1); b2 += __shfl_xor_sync(0xffffffffu, b2, 2);
    if ((lane & 3) == 0) {
      fred[(wgi * 64 + r0) * 2] = a1; fred[(wgi * 64 + r0) * 2 + 1] = a2;
      fred[(wgi * 64 + r0 + 8) * 2] = b1; fred[(wgi * 64 + r0 + 8) * 2 + 1] = b2;
    }
    if (tid == 0) tma::wait_read<0>();              // the previous tile's dm store has read the staging
    csync();
    {
      const float S1a = (fred[r0 * 2] + fred[(64 + r0) * 2]) * (1.f / D), S2a = (fred[r0 * 2 + 1] + fred[(64 + r0) * 2 + 1]) * (1.f / D);
      const float S1b = (fred[(r0 + 8) * 2] + fred[(64 + r0 + 8) * 2]) * (1.f / D), S2b = (fred[(r0 + 8) * 2 + 1] + fred[(64 + r0 + 8) * 2 + 1]) * (1.f / D);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 d0 = unpack2(sDO + b * TILE + wg::off(r0, wgi * 4 + i) + cq), d1 = unpack2(sDO + b * TILE + wg::off(r0 + 8, wgi * 4 + i) + cq);
        const float o00 = rs0 * (gdy[i * 4 + 0] - S1a - xh[i * 4 + 0] * S2a) + d0.x, o01 = rs0 * (gdy[i * 4 + 1] - S1a - xh[i * 4 + 1] * S2a) + d0.y;
        const float o10 = rs1 * (gdy[i * 4 + 2] - S1b - xh[i * 4 + 2] * S2b) + d1.x, o11 = rs1 * (gdy[i * 4 + 3] - S1b - xh[i * 4 + 3] * S2b) + d1.y;
        *reinterpret_cast<uint32_t*>(sDM + wg::off(r0, wgi * 4 + i) + cq) = pack2(o00, o01);
        *reinterpret_cast<uint32_t*>(sDM + wg::off(r0 + 8, wgi * 4 + i) + cq) = pack2(o10, o11);
        dbe[i * 2] += dyacc[i * 4 + 0] + dyacc[i * 4 + 2]; dbe[i * 2 + 1] += dyacc[i * 4 + 1] + dyacc[i * 4 + 3];
        dga[i * 2] += dyacc[i * 4 + 0] * xh[i * 4 + 0] + dyacc[i * 4 + 2] * xh[i * 4 + 2];
        dga[i * 2 + 1] += dyacc[i * 4 + 1] * xh[i * 4 + 1] + dyacc[i * 4 + 3] * xh[i * 4 + 3];
      }
    }
    wg::proxy_fence();
    csync();
    if (tid == 0) { tma::store_2d(&dmmap, tma::sa(sDM), 0, t * BM); tma::commit(); tma::arrive(doneT); }
  }
  if (tid == 0) tma::wait_all();
  // ---- block reductions out: dWgv m-tiles (rows mt*64.., cols i*8 + cq + j) and the LayerNorm parameter gradients ----
  float* dWb = PARTIAL ? dW + (long)blockIdx.x * (KD * D) : dW;
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int mt = q * 2 + wgi;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float* p0 = dWb + (mt * 64 + r0) * 64 + i * 8 + cq;
      if (PARTIAL) {
        *reinterpret_cast<float2*>(p0) = make_float2(wacc[q][i * 4 + 0], wacc[q][i * 4 + 1]);
        *reinterpret_cast<float2*>(p0 + 8 * 64) = make_float2(wacc[q][i * 4 + 2], wacc[q][i * 4 + 3]);
      } else {
        atomicAdd(p0, wacc[q][i * 4 + 0]); atomicAdd(p0 + 1, wacc[q][i * 4 + 1]);
        atomicAdd(p0 + 8 * 64, wacc[q][i * 4 + 2]); atomicAdd(p0 + 8 * 64 + 1, wacc[q][i * 4 + 3]);
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float a = dga[i], c = dbe[i];
    a += __shfl_xor_sync(0xffffffffu, a, 4); a += __shfl_xor_sync(0xffffffffu, a, 8); a += __shfl_xor_sync(0xffffffffu, a, 16);
    c += __shfl_xor_sync(0xffffffffu, c, 4); c += __shfl_xor_sync(0xffffffffu, c, 8); c += __shfl_xor_sync(0xffffffffu, c, 16);
    if (lane < 4) { const int col = wgi * 32 + (i >> 1) * 8 + cq + (i & 1); atomicAdd(dln + col, a); atomicAdd(dln + 64 + col, c); }
  }
}

PFN_cuTensorMapEncodeTiled tma_encode() {
  static PFN_cuTensorMapEncodeTiled fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(p);
  }
  return fn;
}
CUtensorMap map2d(const torch::Tensor& t, long rows, long cols, int box_cols, int box_rows, const char* what) {
  alignas(64) CUtensorMap m{};
  uint64_t gdim[2] = {(uint64_t)cols, (uint64_t)rows};
  uint64_t gstride[1] = {(uint64_t)cols * 2};
  uint32_t bdim[2] = {(uint32_t)box_cols, (uint32_t)box_rows};
  uint32_t estride[2] = {1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", what, ") failed: ", (int)r);
  return m;
}
}  // namespace

template <int NST, bool PARTIAL>
void launch_t(const torch::Tensor& lnw, double eps, int ntile, int grid, float* dW, float* dln, const CUtensorMap& dgmap, const CUtensorMap& ymap,
              const CUtensorMap& xmap, const CUtensorMap& domap, const CUtensorMap& dmmap, const CUtensorMap& wmap) {
  constexpr int BYTES = SMX<NST>::BYTES;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(dgv_bwd_kernel<NST, PARTIAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, dgv_bwd_kernel<NST, PARTIAL>, THREADS, BYTES);
    TORCH_WARN("dgv_bwd_kernel<", NST, ",", PARTIAL, ">: ", BYTES, " B smem -> ", nb, " blocks/SM");
    attr = true;
  }
  dgv_bwd_kernel<NST, PARTIAL><<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(lnw.data_ptr<float>(), (float)eps, ntile, dW, dln, dgmap, ymap, xmap, domap, dmmap, wmap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> dgv_bwd(torch::Tensor dgv, torch::Tensor y, torch::Tensor x, torch::Tensor dout, torch::Tensor wgvT, torch::Tensor lnw, double eps, int64_t blocks_per_sm, int64_t nst, bool partial) {
  const long M = x.numel() / D;
  TORCH_CHECK(dgv.is_cuda() && dgv.scalar_type() == torch::kBFloat16 && dgv.is_contiguous() && dgv.numel() == M * KD, "dgv: [M, 512] bf16 contiguous");
  for (const auto* t : {&y, &x, &dout}) TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == M * D, "y/x/dout: [M, 64] bf16 contiguous");
  TORCH_CHECK(wgvT.scalar_type() == torch::kBFloat16 && wgvT.is_contiguous() && wgvT.dim() == 2 && wgvT.size(0) == D && wgvT.size(1) == KD, "wgvT: [64, 512] bf16");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat && lnw.is_contiguous() && lnw.numel() == D, "lnw: fp32[64]");
  auto dm = torch::empty_like(x);
  auto dln = torch::zeros({2 * D}, x.options().dtype(torch::kFloat));
  CUtensorMap dgmap = map2d(dgv, M, KD, 64, BM, "dgv"), ymap = map2d(y, M, D, 64, BM, "y"), xmap = map2d(x, M, D, 64, BM, "x"),
              domap = map2d(dout, M, D, 64, BM, "dout"), dmmap = map2d(dm, M, D, 64, BM, "dm"), wmap = map2d(wgvT, D, KD, 64, D, "wgvT");
  static int sms = 0;
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.device().index());
  const int ntile = (int)((M + BM - 1) / BM);
  const int grid = (int)std::min<long>(ntile, (long)sms * blocks_per_sm);
  auto dW = partial ? torch::empty({grid, KD, D}, x.options().dtype(torch::kFloat)) : torch::zeros({KD, D}, x.options().dtype(torch::kFloat));
  float* dWp = dW.data_ptr<float>(); float* dlp = dln.data_ptr<float>();
  switch (nst * 2 + (partial ? 1 : 0)) {
    case 16: launch_t<8, false>(lnw, eps, ntile, grid, dWp, dlp, dgmap, ymap, xmap, domap, dmmap, wmap); break;
    case 17: launch_t<8, true>(lnw, eps, ntile, grid, dWp, dlp, dgmap, ymap, xmap, domap, dmmap, wmap); break;
    case 24: launch_t<12, false>(lnw, eps, ntile, grid, dWp, dlp, dgmap, ymap, xmap, domap, dmmap, wmap); break;
    case 25: launch_t<12, true>(lnw, eps, ntile, grid, dWp, dlp, dgmap, ymap, xmap, domap, dmmap, wmap); break;
    case 8: launch_t<4, false>(lnw, eps, ntile, grid, dWp, dlp, dgmap, ymap, xmap, domap, dmmap, wmap); break;
    case 9: launch_t<4, true>(lnw, eps, ntile, grid, dWp, dlp, dgmap, ymap, xmap, domap, dmmap, wmap); break;
    default: TORCH_CHECK(false, "nst must be 4, 8 or 12");
  }
  if (partial) dW = dW.sum(0);
  return {dm, dW, dln.narrow(0, 0, D), dln.narrow(0, D, D)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("dgv_bwd", &dgv_bwd, "fused dgv -> (dm = LN_bwd(dgv.Wgv) + dout, dWgv, dgamma, dbeta)",
          py::arg("dgv"), py::arg("y"), py::arg("x"), py::arg("dout"), py::arg("wgvT"), py::arg("lnw"), py::arg("eps") = 1e-5, py::arg("blocks_per_sm") = 1, py::arg("nst") = 8, py::arg("partial") = false);
}
