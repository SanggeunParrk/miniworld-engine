// PWA backward glue from the saved o, warp-specialized (sm_90a), with the proj_o weight gradient folded in:  out[s,i,:] = msa[s,i,:] + sum_h (sigmoid(y[s,i,:] Wg_h^T) .* o_h[s,i,:]) Wo_h,
//   o_h[s,i,:] = sum_j w[h,i,j] v[h][j][s*C + :]      (v head-major), optionally saving o (natural [S][N][HC]).
// A 384-thread block: two CONSUMER warpgroups own the two i-tiles (64 rows each) of one i-pair for one s-pair
// and a PRODUCER warpgroup (one thread) streams, per (head, j-chunk), a 24 KB stage = {W chunk for consumer 0,
// W chunk for consumer 1, the shared v chunk} through a 3-deep TMA ring with full/empty mbarriers.  Sharing the
// v chunk halves the L2 re-read of v (the old kernel: 6 CTAs per s-pair each streamed all of v_h).  Per head a
// consumer runs six m64n64k16 wgmma (A = W chunk K-major, B = v chunk MN-major), then the gate GEMM against its
// y tiles (TMA, double-buffered per tile), sigmoid x o packed into register-A fragments, and the out-projection
// as an RS wgmma accumulated over heads; per-head Wg_h / Wo_h come by TMA into parity buffers.  The residual
// is TMA-loaded into the output staging at the tile start and added in registers before the TMA store.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace {
constexpr int H = 8, C = 32, D = 64, HC = H * C;
constexpr int BS = 2, BJ = 64, NB = BS * C;
constexpr int TILE = 64 * 64;                            // 8 KiB bf16 tile
constexpr int THREADS = 384;

namespace wg {
constexpr uint32_t SBO = 64;
__device__ __forceinline__ int off(int r, int kc) { return r * 64 + ((kc ^ (r & 7)) << 3); }
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(SBO) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
__device__ __forceinline__ uint64_t desc_mn(const void* p) { return desc(p); }
__device__ __forceinline__ int off64(int r, int kc) { return r * 32 + ((kc ^ ((r >> 1) & 3)) << 3); }
__device__ __forceinline__ uint64_t desc64(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(32) << 32);
  d |= (static_cast<uint64_t>(2) << 62);
  return d;
}
// A K-major [m][k], B MN-major [k][n]: the contraction of a weight row block with a head-major activation tile
__device__ __forceinline__ void mma_m64n64k16_kmn(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31},\n"
      " %33, %34, p, 1, 1, 0, 1;\n"
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
// both K-major, n = 32: the per-head gate / output projections
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
// A from registers (the previous accumulator, packed to bf16), B K-major: the out-projection
__device__ __forceinline__ void mma_m64n64k16_rs(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31},\n"
      " {%33, %34, %35, %36}, %37, p, 1, 1, 0;\n"
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
      : "r"(accumulate), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db)
      : "memory");
}

// A from registers (four 32-bit fragments), B K-major, n = 32
__device__ __forceinline__ void mma_m64n32k16_rs(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %16, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15},\n"
      " {%17, %18, %19, %20}, %21, p, 1, 1, 0;\n"
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
      : "r"(accumulate), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db)
      : "memory");
}
// A MN-major (a [64 i][64 d] dout tile read transposed: M = d), B MN-major (a [64 i][32 c] go tile, 64B swizzle): the dWo partial
__device__ __forceinline__ void mma_m64n32k16_mm(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %16, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15},\n"
      " %17, %18, p, 1, 1, 1, 1;\n"
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
// n = 16: each consumer warpgroup owns 16 of the 32 output channels of every head's dWo partial (identical code paths, no divergence)
__device__ __forceinline__ void mma_m64n16k16_mm(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %8, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7},\n"
      " %9, %10, p, 1, 1, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory"); }
__device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n" ::: "memory"); }
}  // namespace wg

namespace tma {
__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ int sw128(int row, int col64) { return row * 64 + (((col64 >> 3) ^ (row & 7)) << 3) + (col64 & 7); }
template <int BI>
__device__ __forceinline__ int sw64nat(int tok, int si, int c) {
  const int line = si * BI + tok;
  return line * 32 + ((((c >> 3) ^ ((line >> 1) & 3))) << 3) + (c & 7);
}
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void load_3d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n" :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_3d(const void* map, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];\n" :: "l"(map), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void load_2d_mc(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
__device__ __forceinline__ void arrive_remote(uint64_t* b, uint32_t rank) {   // arrive on the same-offset barrier of CTA `rank` in the cluster (CUTLASS ClusterBarrier::arrive form)
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n" :: "r"(sa(b)), "r"(rank) : "memory");
}
__device__ __forceinline__ void wait_cluster(uint64_t* b, uint32_t parity) {   // a barrier that receives remote arrivals: cluster-scope acquire
  asm volatile("{\n.reg .pred p;\nWAITC_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra WAITC_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
}  // namespace tma

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}

// smem (bf16 elements from a 1 KiB-aligned base)
template <int NST> struct SMX {
  static constexpr int SOR = 0;                                     // [NST][2 consumers][TILE]  o tiles, natural layout [2 s][64 tok][32 c], 64B swizzle
  static constexpr int SDO = SOR + NST * 2 * TILE;                  // [2 tile parity][2 consumers][2 s][TILE] dout tiles (K-major [64 i][64 d])
  static constexpr int SWG = SDO + 8 * TILE;                        // [2 head parity][32*64] Wg_h
  static constexpr int SWOT = SWG + 2 * 32 * 64;                    // [2 head parity][32*64] Wo^T_h
  static constexpr int SDOUT = SWOT + 2 * 32 * 64;                  // [2 head parity][2 consumers][TILE] do staging (head-major tile [64 i][64 (s,c)])
  static constexpr int SDGP = SDOUT + 4 * TILE;                     // [2 head parity][2 consumers][TILE] dgp staging (natural, 64B swizzle)
  static constexpr int SGO = SDGP + 4 * TILE;                       // [2 head parity][2 consumers][TILE] go tiles (natural, 64B swizzle): the dWo B operand
  static constexpr int SEND = SGO + 4 * TILE;
  static constexpr int NBAR = 2 * NST + 2 + 1 + 2 + 2 + 2;          // full, empty, fullD[2], doneT, fullW[2], wfree[2], goready[2]
  static constexpr int BYTES = SEND * 2 + NBAR * 8 + 1024;
};

// Per 128 i x 2 s tile: consumer warpgroup c owns i-tile c.  Per head: gate GEMM (RS, y in registers) and du GEMM
// (A = the dout tile, B = Wo^T_h) -> do = du.g (head-major, TMA), dgp = du.o.g.(1-g) (natural, TMA into dgv's first
// half), go = g.o kept in shared memory only: the warpgroup owning head h (h & 1 == c) accumulates
// dWo_h[d][c] += sum_{s,i} dout[s,i,d] go[s,i,c] over BOTH i-tiles with A = dout tile read MN-major (transposed) and
// B = the go tiles (MN-major, 64B swizzle), in registers across the block's tiles; per-block fp32 slabs are summed
// on the host.  The producer warpgroup streams o tiles (one 16 KB stage per head, one issuing thread per slot), the
// dout tiles per tile (parity buffers behind a doneT barrier) and the per-head weights (parity buffers).
template <int NST>
__global__ void __launch_bounds__(THREADS, 1) pwa_glue3_kernel(int N, int S, int ntile, float* __restrict__ dWo,
    const __grid_constant__ CUtensorMap omap,     // o    natural [S][N][HC]     3-D box (32 c, 64 tok, 2 s), 64B swizzle (load)
    const __grid_constant__ CUtensorMap dmap,     // dout [S*N][D]              box (64, 64) (load)
    const __nv_bfloat16* __restrict__ Yg,         // y    [S*N][D]              (read as RS fragments)
    const __grid_constant__ CUtensorMap gmap,     // wg   [HC][D]               box (64, 32)
    const __grid_constant__ CUtensorMap wotmap,   // wo^T [HC][D]               box (64, 32)
    const __grid_constant__ CUtensorMap domap,    // do   head-major [H*N][S*C] box (64, 64) (store)
    const __grid_constant__ CUtensorMap dgpmap,
    const __nv_bfloat16* __restrict__ drop_mask, float drop_scale) { // dgp  natural, row stride rs, 3-D box (32 c, 64 tok, 2 s), 64B swizzle (store)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  unsigned char* smb = smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  __nv_bfloat16* sm = reinterpret_cast<__nv_bfloat16*>(smb);
  using L = SMX<NST>;
  __nv_bfloat16* sOR = sm + L::SOR;
  __nv_bfloat16* sDOb = sm + L::SDO;
  __nv_bfloat16* sWGb = sm + L::SWG;
  __nv_bfloat16* sWOTb = sm + L::SWOT;
  __nv_bfloat16* sDOUTb = sm + L::SDOUT;
  __nv_bfloat16* sDGPb = sm + L::SDGP;
  __nv_bfloat16* sGOb = sm + L::SGO;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smb + L::SEND * 2);
  uint64_t* full = bars;                 // [NST]
  uint64_t* empty = full + NST;          // [NST], count 8 (consumer warps)
  uint64_t* fullD = empty + NST;         // [2] tile parity: dout tiles landed
  uint64_t* doneT = fullD + 2;           // [1], count 2: both consumers are done with a tile (dout tiles free)
  uint64_t* fullW = doneT + 1;           // [2] head parity
  uint64_t* wfree = fullW + 2;           // [2], count 2
  uint64_t* goready = wfree + 2;         // [2] head parity, count 2: both consumers' go tiles of a head are in shared memory
  const int tid = threadIdx.x, lane = tid & 31;
  const int NI2 = N / 128;
  auto tile_sp = [&](int t) { return t / NI2; };
  auto tile_ip = [&](int t) { return t % NI2; };

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { tma::bar_init(full + i, 1); tma::bar_init(empty + i, 8); }
    for (int i = 0; i < 2; ++i) { tma::bar_init(fullD + i, 1); tma::bar_init(fullW + i, 1); tma::bar_init(wfree + i, 2); }
    tma::bar_init(doneT, 2); tma::bar_init(goready, 2); tma::bar_init(goready + 1, 2);
    tma::bar_init_fence();
    wg::proxy_fence();
  }
  __syncthreads();

  if (tid >= 256) {
    // ===================== producer: warp p lane 0 owns o-stage slot p; warp 0 also streams dout tiles and weights =====================
    static_assert(NST <= 4, "one producer warp per ring slot");
    wg::reg_dealloc();
    const int pw = (tid - 256) >> 5;
    if (lane == 0 && pw < NST) {
      auto issue_dout = [&](int T) {
        const int t = blockIdx.x + T * gridDim.x;
        if (t >= ntile) return;
        const int b = T & 1, sp = tile_sp(t), ip = tile_ip(t);
        if (T >= 2) tma::wait(doneT, (T - 2) & 1);                   // buffer b was used by tile T-2: wait for phase T-2 of the single doneT barrier
        tma::expect_tx(fullD + b, 4 * TILE * 2);
        for (int c = 0; c < 2; ++c)
          for (int si = 0; si < BS; ++si)
            tma::load_2d(&dmap, tma::sa(sDOb + ((b * 2 + c) * 2 + si) * TILE), fullD + b, 0, (sp * BS + si) * N + ip * 128 + c * 64);
      };
      if (pw == 0) { issue_dout(0); issue_dout(1); }
      int gs = 0;
      for (int T = 0;; ++T) {
        const int t = blockIdx.x + T * gridDim.x;
        if (t >= ntile) break;
        const int sp = tile_sp(t), ip = tile_ip(t), s0 = sp * BS;
        for (int h = 0; h < H; ++h, ++gs) {
          const int gh = T * H + h, hb = gh & 1;
          if (pw == 0) {
            if (gh >= 2) tma::wait(wfree + hb, ((gh >> 1) - 1) & 1);
            tma::expect_tx(fullW + hb, 2 * 32 * 64 * 2);
            tma::load_2d(&gmap, tma::sa(sWGb + hb * 32 * 64), fullW + hb, 0, h * C);
            tma::load_2d(&wotmap, tma::sa(sWOTb + hb * 32 * 64), fullW + hb, 0, h * C);
          }
          const int st = gs % NST;
          if (st != pw) continue;
          if (gs >= NST) tma::wait(empty + st, ((gs / NST) - 1) & 1);
          tma::expect_tx(full + st, 2 * TILE * 2);
          for (int c = 0; c < 2; ++c) tma::load_3d(&omap, tma::sa(sOR + (st * 2 + c) * TILE), full + st, h * C, ip * 128 + c * 64, s0);
        }
        if (pw == 0) issue_dout(T + 2);
      }
    }
  } else {
  // ===================== consumer c =====================
  wg::reg_alloc();
  const int c = tid >> 7, wtid = tid & 127, wl = wtid >> 5, oc = 1 - c;
  auto wg_sync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(1 + c) : "memory"); };
  const int r_acc = wl * 16 + (lane >> 2), cq = (lane & 3) * 2;

  uint32_t yf[2][16];
  auto load_y = [&](int T) {
    const int t = blockIdx.x + T * gridDim.x;
    if (t >= ntile) return;
    const int sp = tile_sp(t), ip = tile_ip(t), i0 = ip * 128 + c * 64, s0 = sp * BS;
#pragma unroll
    for (int si = 0; si < BS; ++si) {
      const uint32_t* base = reinterpret_cast<const uint32_t*>(Yg + ((long)(s0 + si) * N + i0) * D);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        yf[si][ks * 4 + 0] = __ldg(base + (r_acc * D + ks * 16 + cq) / 2);
        yf[si][ks * 4 + 1] = __ldg(base + ((r_acc + 8) * D + ks * 16 + cq) / 2);
        yf[si][ks * 4 + 2] = __ldg(base + (r_acc * D + ks * 16 + 8 + cq) / 2);
        yf[si][ks * 4 + 3] = __ldg(base + ((r_acc + 8) * D + ks * 16 + 8 + cq) / 2);
      }
    }
  };
  float gacc[2][16], duacc[2][16];
  // dWo partials: this warpgroup owns channels [16c, 16c+16) of every head (named arrays selected by a switch, never indexed)
  float wa0[8], wa1[8], wa2[8], wa3[8], wa4[8], wa5[8], wa6[8], wa7[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { wa0[i] = 0.f; wa1[i] = 0.f; wa2[i] = 0.f; wa3[i] = 0.f; wa4[i] = 0.f; wa5[i] = 0.f; wa6[i] = 0.f; wa7[i] = 0.f; }
  auto dwo_issue = [&](int hh, int hbb, const __nv_bfloat16* dc, const __nv_bfloat16* dother) {
    // dWo_hh[:, 16c:16c+16] over both i-tiles and both s: A = dout tile transposed (MN-major), B = the go tile's 16-channel half
    // (MN-major, 64B swizzle: the +32 B start offset selects the half; the swizzle is address-based)
    wg::fence();
#define DWO_GEMMS(wa)                                                                                                            \
    _Pragma("unroll")                                                                                                            \
    for (int q = 0; q < 2; ++q) {                                                                                                \
      const __nv_bfloat16* dq = q == 0 ? dc : dother;                                                                            \
      const __nv_bfloat16* gq = sGOb + (hbb * 2 + (q == 0 ? c : oc)) * TILE + c * 16;                                            \
      _Pragma("unroll")                                                                                                          \
      for (int si = 0; si < BS; ++si)                                                                                            \
        _Pragma("unroll")                                                                                                        \
        for (int ks = 0; ks < 4; ++ks)                                                                                           \
          wg::mma_m64n16k16_mm(wg::desc(dq + si * TILE + wg::off(ks * 16, 0)), wg::desc64(gq + si * 64 * C + ks * 16 * C), wa, 1); \
    }
    switch (hh) { case 0: DWO_GEMMS(wa0) break; case 1: DWO_GEMMS(wa1) break; case 2: DWO_GEMMS(wa2) break; case 3: DWO_GEMMS(wa3) break;
                  case 4: DWO_GEMMS(wa4) break; case 5: DWO_GEMMS(wa5) break; case 6: DWO_GEMMS(wa6) break; default: DWO_GEMMS(wa7) break; }
#undef DWO_GEMMS
    wg::commit();
  };
  int gs = 0;
  load_y(0);
  for (int T = 0;; ++T) {
    const int t = blockIdx.x + T * gridDim.x;
    if (t >= ntile) break;
    const int b = T & 1, sp = tile_sp(t), ip = tile_ip(t), i0 = ip * 128 + c * 64, s0 = sp * BS;
    const __nv_bfloat16* sDOc = sDOb + (b * 2 + c) * 2 * TILE;     // my dout tiles [2 s][TILE]
    const __nv_bfloat16* sDOo = sDOb + (b * 2 + oc) * 2 * TILE;    // the other consumer's (for the dWo partial)
    tma::wait(fullD + b, (T >> 1) & 1);
    if (drop_mask != nullptr) {
      // Each consumer owns its two dres tiles. Mask in shared memory once, before
      // both du and dWo consume them; no [S,N,D] masked gradient goes through HBM.
      auto* tile = const_cast<__nv_bfloat16*>(sDOc);
#pragma unroll
      for (int v = wtid; v < BS * TILE / 2; v += 128) {
        const int si = v / (TILE / 2), rc = (v % (TILE / 2)) * 2;
        const int row = rc / D, col = rc % D;
        auto* ptr = reinterpret_cast<__nv_bfloat162*>(tile + si * TILE + tma::sw128(row, col));
        const float2 value = __bfloat1622float2(*ptr);
        const float2 keep = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(drop_mask + (i0 + row) * D + col));
        *ptr = __float22bfloat162_rn(make_float2((value.x * keep.x) * drop_scale, (value.y * keep.y) * drop_scale));
      }
      wg::proxy_fence();
      wg_sync();
    }
#pragma unroll 1
    for (int h = 0; h < H; ++h, ++gs) {
      const int gh = T * H + h, hb = gh & 1, st = gs % NST;
      const __nv_bfloat16* wgt = sWGb + hb * 32 * 64;
      const __nv_bfloat16* wot = sWOTb + hb * 32 * 64;
      // ---- gate (RS) and du (SS) projections for both s ----
      tma::wait(fullW + hb, (gh >> 1) & 1);
      wg::fence();
#pragma unroll
      for (int si = 0; si < BS; ++si)
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          wg::mma_m64n32k16_rs(yf[si][ks * 4 + 0], yf[si][ks * 4 + 1], yf[si][ks * 4 + 2], yf[si][ks * 4 + 3], wg::desc(wgt + wg::off(0, ks * 2)), gacc[si], ks == 0 ? 0 : 1);
          wg::mma_m64n32k16_kk(wg::desc(sDOc + si * TILE + wg::off(0, ks * 2)), wg::desc(wot + wg::off(0, ks * 2)), duacc[si], ks == 0 ? 0 : 1);
        }
      wg::commit();
      if (h > 0) {                                                 // the previous head's dWo GEMM, once both consumers' go tiles of it are in place:
        tma::wait(goready + ((gh - 1) & 1), ((gh - 1) >> 1) & 1); // it retires under this head's wait<0>
        dwo_issue(h - 1, (gh - 1) & 1, sDOc, sDOo);
      }
      tma::wait(full + st, (gs / NST) & 1);                        // this head's o tiles
      const __nv_bfloat16* so = sOR + (st * 2 + c) * TILE;
      wg::wait<0>();
      if (wtid == 0) tma::wait_read<1>();                          // head h-2's do / dgp stores have read this parity's staging (h-1's may still be in flight)
      wg_sync();
      __nv_bfloat16* sGO = sGOb + (hb * 2 + c) * TILE;
      __nv_bfloat16* sDOUT = sDOUTb + (hb * 2 + c) * TILE;
      __nv_bfloat16* sDGP = sDGPb + (hb * 2 + c) * TILE;
      // ---- elementwise: do (head-major staging), dgp (natural staging), go (natural, kept for the dWo GEMM) ----
#pragma unroll
      for (int si = 0; si < BS; ++si)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int cc = nt * 8 + cq;                                // channel within the head
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int rr = r_acc + half * 8;
            const float2 ov = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so + tma::sw64nat<64>(rr, si, cc)));
            const float g0 = 1.f / (1.f + __expf(-gacc[si][nt * 4 + half * 2])), g1 = 1.f / (1.f + __expf(-gacc[si][nt * 4 + half * 2 + 1]));
            const float d0 = duacc[si][nt * 4 + half * 2], d1 = duacc[si][nt * 4 + half * 2 + 1];
            const int at = tma::sw128(rr, si * C + cc), bt = tma::sw64nat<64>(rr, si, cc);
            *reinterpret_cast<uint32_t*>(sDOUT + at) = pack2(d0 * g0, d1 * g1);                                                  // do
            *reinterpret_cast<uint32_t*>(sDGP + bt) = pack2(d0 * ov.x * g0 * (1.f - g0), d1 * ov.y * g1 * (1.f - g1));           // dgp
            *reinterpret_cast<uint32_t*>(sGO + bt) = pack2(g0 * ov.x, g1 * ov.y);                                                // go
          }
        }
      wg::proxy_fence();
      wg_sync();
      if (wtid == 0) {
        tma::store_2d(&domap, tma::sa(sDOUT), s0 * C, h * N + i0);
        tma::store_3d(&dgpmap, tma::sa(sDGP), h * C, i0, s0);
        tma::commit();
      }
      if (lane == 0) tma::arrive(empty + st);                      // the o tile was read with plain loads, all retired at the barrier
      if (wtid == 0) { tma::arrive(goready + hb); tma::arrive(wfree + hb); }   // my go tile of this head is in place; Wg / Wo^T of this head are free
    }
    // the last head's dWo GEMM (both go tiles in place), retired before the tile ends
    tma::wait(goready + ((T * H + H - 1) & 1), ((T * H + H - 1) >> 1) & 1);
    dwo_issue(H - 1, (T * H + H - 1) & 1, sDOc, sDOo);
    wg::wait<0>();
    if (wtid == 0) tma::arrive(doneT);                             // this tile's dout tiles are free (both consumers arrive)
    load_y(T + 1);
  }
  if (wtid == 0) tma::wait_all();
  // ---- this block's dWo partial slab [64 d][HC]: head 2q + c, rows d = r_acc / r_acc + 8, cols i*8 + cq + j ----
  float* slab = dWo + (long)blockIdx.x * (D * HC);
  auto dump = [&](const float* wa, int h) {                        // m64n16 accumulator: rows r_acc / r_acc+8, cols i*8 + cq + j (i < 2)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float* p0 = slab + r_acc * HC + h * C + c * 16 + i * 8 + cq;
      *reinterpret_cast<float2*>(p0) = make_float2(wa[i * 4 + 0], wa[i * 4 + 1]);
      *reinterpret_cast<float2*>(p0 + 8 * HC) = make_float2(wa[i * 4 + 2], wa[i * 4 + 3]);
    }
  };
  dump(wa0, 0); dump(wa1, 1); dump(wa2, 2); dump(wa3, 3); dump(wa4, 4); dump(wa5, 5); dump(wa6, 6); dump(wa7, 7);
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
CUtensorMap map2d(void* base, long rows, long cols, int box_cols, int box_rows, CUtensorMapSwizzle sw, const char* what) {
  alignas(64) CUtensorMap m{};
  uint64_t gdim[2] = {(uint64_t)cols, (uint64_t)rows};
  uint64_t gstride[1] = {(uint64_t)cols * 2};
  uint32_t bdim[2] = {(uint32_t)box_cols, (uint32_t)box_rows};
  uint32_t estride[2] = {1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", what, ") failed: ", (int)r);
  return m;
}
CUtensorMap map_nat(void* base, long row_stride, int N, int S, int bi) {   // natural [S][N][row_stride] bf16, HC live columns, box (32 c, bi tok, 2 s), 64B swizzle
  alignas(64) CUtensorMap m{};
  uint64_t gdim[3] = {(uint64_t)HC, (uint64_t)N, (uint64_t)S};
  uint64_t gstride[2] = {(uint64_t)row_stride * 2, (uint64_t)N * row_stride * 2};
  uint32_t bdim[3] = {32, (uint32_t)bi, BS};
  uint32_t estride[3] = {1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (natural) failed: ", (int)r);
  return m;
}
}  // namespace

template <int NST>
void launch_glue3(int N, int S, int ntile, int grid, float* dWo, const CUtensorMap& omap, const CUtensorMap& dmap, const __nv_bfloat16* yp,
                  const CUtensorMap& gmap, const CUtensorMap& wotmap, const CUtensorMap& domap, const CUtensorMap& dgpmap, const __nv_bfloat16* drop_mask, float drop_scale) {
  constexpr int BYTES = SMX<NST>::BYTES;
  static bool attr = false;                                        // one per instantiation
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_glue3_kernel<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
    int nb = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pwa_glue3_kernel<NST>, THREADS, BYTES));
    TORCH_WARN("pwa_glue3_kernel<", NST, ">: ", BYTES, " B smem -> ", nb, " blocks/SM");
    attr = true;
  }
  pwa_glue3_kernel<NST><<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(N, S, ntile, dWo, omap, dmap, yp, gmap, wotmap, domap, dgpmap, drop_mask, drop_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// glue from the saved o: (do head-major [H][N][S*C], dgp (into dgv[..., :HC] when given, else [S][N][HC]), dWo fp32 [D][HC])
std::vector<torch::Tensor> pwa_glue3(torch::Tensor o, torch::Tensor y, torch::Tensor dout, torch::Tensor wgw, torch::Tensor wot, c10::optional<torch::Tensor> dgv, int64_t nst, int64_t blocks_per_sm, c10::optional<torch::Tensor> drop_mask, double drop_scale) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.dim() == 3 && o.size(2) == HC, "o: [S, N, HC] bf16");
  const int S = (int)o.size(0), N = (int)o.size(1);
  TORCH_CHECK(N % 128 == 0 && S % BS == 0, "N must be a multiple of 128 and S even");
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}), "y: [S, N, D]");
  TORCH_CHECK(dout.is_contiguous() && dout.sizes() == torch::IntArrayRef({S, N, D}), "dout: [S, N, D]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}), "wg: [HC, D]");
  TORCH_CHECK(wot.is_contiguous() && wot.sizes() == torch::IntArrayRef({HC, D}), "wot: Wo^T [HC, D]");
  if (drop_mask.has_value()) {
    TORCH_CHECK(drop_mask->device() == o.device() && drop_mask->scalar_type() == torch::kBFloat16
                && drop_mask->is_contiguous() && drop_mask->sizes() == torch::IntArrayRef({N, D}),
                "drop_mask: contiguous bf16 [N, D] on o's device");
  }
  const auto* mask_ptr = drop_mask.has_value() ? reinterpret_cast<const __nv_bfloat16*>(drop_mask->data_ptr<at::BFloat16>()) : nullptr;
  auto d_o = torch::empty({H, N, (long)S * C}, o.options());
  torch::Tensor dgp; long rs = HC;
  if (dgv.has_value()) {
    TORCH_CHECK(dgv->is_contiguous() && dgv->sizes() == torch::IntArrayRef({S, N, 2 * HC}), "dgv: [S, N, 2*HC]");
    dgp = *dgv; rs = 2 * HC;
  } else {
    dgp = torch::empty({S, N, HC}, o.options());
  }
  static int sms = 0;
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, o.device().index());
  const int ntile = (S / BS) * (N / 128);
  const int grid = (int)std::min<long>(ntile, (long)sms * blocks_per_sm);
  auto dWo = torch::empty({grid, D, HC}, o.options().dtype(torch::kFloat));
  CUtensorMap omap = map_nat(o.data_ptr(), HC, N, S, 64), dmap = map2d(dout.data_ptr(), (long)S * N, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "dout"),
              gmap = map2d(wgw.data_ptr(), HC, D, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B, "wg"), wotmap = map2d(wot.data_ptr(), HC, D, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B, "wot"),
              domap = map2d(d_o.data_ptr(), (long)H * N, (long)S * C, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "do"), dgpmap = map_nat(dgp.data_ptr(), rs, N, S, 64);
  const __nv_bfloat16* yp = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const int so_unused = 0; (void)so_unused;
  if (nst == 3) launch_glue3<3>(N, S, ntile, grid, dWo.data_ptr<float>(), omap, dmap, yp, gmap, wotmap, domap, dgpmap, mask_ptr, (float)drop_scale);
  else if (nst == 2) launch_glue3<2>(N, S, ntile, grid, dWo.data_ptr<float>(), omap, dmap, yp, gmap, wotmap, domap, dgpmap, mask_ptr, (float)drop_scale);
  else TORCH_CHECK(false, "nst must be 2 or 3 (the double-buffered staging leaves room for at most three o-stage slots)");
  return {d_o, dgp, dWo.sum(0)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("pwa_glue3", &pwa_glue3, "PWA backward glue from the saved o with the dWo partial folded in: (do head-major, dgp, dWo fp32 [64][256])",
          py::arg("o"), py::arg("y"), py::arg("dout"), py::arg("wg"), py::arg("wot"), py::arg("dgv") = py::none(), py::arg("nst") = 2, py::arg("blocks_per_sm") = 1, py::arg("drop_mask") = py::none(), py::arg("drop_scale") = 1.0);
}
