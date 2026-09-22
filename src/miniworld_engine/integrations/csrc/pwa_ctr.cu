// PWA contraction kernel (sm_90a): o[s,i,h,:] = sum_j w[h,i,j] v[s,j,h,:] with the epilogue fused.
//
// MODE 0 (forward):  out[s,i,:] = sum_h (sigmoid(y[s,i,:] Wg_h^T) .* o_h[s,i,:]) Wo_h     -- gate + proj_o
// MODE 1 (bwd glue): du = dout[s,i,:] Wo_h, g = sigmoid(y Wg_h^T)  ->  do = du.*g, dgp = du.*o.*g.*(1-g), go = g.*o
//                    all three head-major, so dv (MODE 2 on do with w^T) and dw (a batched GEMM) read them as-is
// MODE 2 (plain):    the contraction alone, natural-layout out -- dv with W := w^T and V := do
// MODE 3 (glue-o):   MODE 1's epilogue over a SAVED o (natural layout) -- no contraction at all.  The
//                    forward writes o when asked (201 MB); the backward then never touches v or w.
// MODE 0 also adds the residual (out += msa) so the module's forward is one launch.
//
// v is HEAD-MAJOR, V[h][j][s*C + c]: for one head the contraction is a plain GEMM  W_h[i][j] x V_h[j][(s,c)]
// and a (j, s-pair) tile of it is one 2-D block with 128-byte rows -- the canonical MN-major wgmma
// operand -- so no permute and no gather anywhere.  A CTA owns 64 i x 2 s and loops over the 8 heads;
// per head it runs the 384-deep contraction in 6 chunks of 64 j through a two-stage cp.async pipeline,
// then the gate GEMM against the y tile it staged once, and the out-projection as a register-A wgmma
// (the accumulator's C layout IS the next mma's A layout, packed to bf16) accumulating over heads.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace {
constexpr int H = 8, C = 32, D = 64, HC = H * C;
constexpr int BS = 2, BJ = 64, NB = BS * C;              // tile: (64 * NWG) i x 2 s (n = 64 = two 32-channel runs)
constexpr int TILE = 64 * 64;                            // a [64 rows][64] bf16 tile = 8 KiB
static_assert(D == 64 && NB == 64 && BJ == 64, "all tiles are 64 wide: one 128-byte swizzle block");

namespace wg {
constexpr uint32_t SBO = 64;
// element offset of chunk kc (8 values) of row r in a [rows][64] tile, 128-byte XOR swizzle
__device__ __forceinline__ int off(int r, int kc) { return r * 64 + ((kc ^ (r & 7)) << 3); }
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(SBO) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
__device__ __forceinline__ uint64_t desc_mn(const void* p) { return desc(p); }   // one 64-wide MN block: LBO = 1 too
// 64-byte swizzle, K-major, rows of 32 elements: chunk (2 bits) ^= (row >> 1) & 3, 8-row groups 512 B apart
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

__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
}  // namespace wg

namespace tma {
__device__ __forceinline__ int sw128(int row, int col64) {
  return row * 64 + (((col64 >> 3) ^ (row & 7)) << 3) + (col64 & 7);
}
// natural-layout output tile [2 s][BI tok][32 c] under TMA's 64-byte swizzle (CUTLASS Swizzle<2,4,3>: the
// 16-byte chunk index, address bits 4-5, XOR-ed with bits 7-8 -- for 64-byte lines, line >> 1)
template <int BI>
__device__ __forceinline__ int sw64nat(int tok, int si, int c) {
  const int line = si * BI + tok;
  return line * 32 + ((((c >> 3) ^ ((line >> 1) & 3))) << 3) + (c & 7);
}
__device__ __forceinline__ void store_3d(const void* map, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
}  // namespace tma

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}

// shared memory carve-up, in bf16 elements from a 1 KiB-aligned base.  NWG warpgroups each own 64 i
// rows: the A-side tiles (W chunk, y, dout, outputs) are [64*NWG][64], the B-side ones [64][64].
// Three operand stages for the contraction modes: the prefetch of stage t+1 is issued at the top of
// iteration t into the buffer group t-2 read, which the previous iteration's wait_group 1 has retired,
// so one wgmma group stays in flight across the barrier with no hazard.  (Two stages + wait_group 1
// raced the prefetch against the in-flight group; two stages + a full wait cost 25%.)  The head
// weights are single-buffered: the next head's are fetched at the END of this head's epilogue.
template <int MODE, int NWG>
struct SM {
  static constexpr int TA = NWG * TILE;
  static constexpr bool CTR = MODE != 3;             // has the contraction (A/B operand stages)
  static constexpr int NST = CTR ? 3 : 2;            // operand stages (MODE 3: o tiles, plain loads, no hazard)
  static constexpr int A = 0;                        // [NST][TA]   W chunks   (K-major [i][j])        / MODE 3: o tiles
  static constexpr int B = A + NST * TA;             // [NST][TILE] V chunks   (MN-major [j][(s,c)])
  static constexpr int Y = B + (CTR ? NST * TILE : 0); // [BS][TA]  y tiles    (K-major [i][d])      (MODE 0, 1, 3)
  static constexpr int WG = Y + (MODE == 2 ? 0 : BS * TA);          // [32*64] Wg_h
  static constexpr int WO = WG + (MODE == 2 ? 0 : 32 * 64);          // [64*32] Wo_h (MODE 0, [d][c], 64B swizzle) / [32*64] Wo^T_h (MODE 1, 3)
  static constexpr int DO = WO + (MODE == 2 ? 0 : 32 * 64);          // [BS][TA] dout tiles (MODE 1, 3)
  // Output staging is DEDICATED, never an operand buffer: a TMA store reads its tile asynchronously and
  // the only wait is at the next epilogue, while an operand buffer is refilled two stages later.
  // Aliasing the two produced wrong dv/dw at the full shape and passed every small test.
  static constexpr int OUT = DO + (MODE == 1 || MODE == 3 ? BS * TA : 0);   // MODE 1: [3][TA] do/dgp/go; MODE 3: [2][TA] dgp/go (do rides on the consumed o tile); MODE 2: [TA] out
  static constexpr int END = OUT + (MODE == 1 ? 3 * TA : MODE == 3 ? 2 * TA : MODE == 2 ? TA : 0);
  static constexpr int BYTES = END * 2;              // no alignment slack: the base is checked instead (MODE 0 is at 76 KiB exactly, three per SM)
};

template <int MODE, int NWG>
__global__ __launch_bounds__(128 * NWG, NWG == 1 ? (MODE == 2 ? 4 : MODE == 0 ? 3 : 2) : (MODE == 2 ? 3 : MODE == 0 ? 2 : 1)) void pwa_ctr_kernel(
    const __nv_bfloat16* __restrict__ W,     // [H][N][N]      softmax weights (MODE 2: transposed per head)
    const __nv_bfloat16* __restrict__ V,     // [H][N][S*C]    head-major activation (v, or do for dv); MODE 3: o, natural [S][N][HC]
    const __nv_bfloat16* __restrict__ RES,   // [S][N][D]      the residual input msa                 (MODE 0)
    int save_o,                              // MODE 0: also write o (natural layout) through MAP1
    const __nv_bfloat16* __restrict__ Y,     // [S][N][D]      LayerNorm output            (MODE 0, 1)
    const __nv_bfloat16* __restrict__ DOUT,  // [S][N][D]      gradient of the update      (MODE 1)
    const __nv_bfloat16* __restrict__ WG,    // [HC][D]        proj_g.weight               (MODE 0, 1)
    const __nv_bfloat16* __restrict__ WO,    // [D][HC] proj_o.weight (MODE 0) / [HC][D] its transpose (MODE 1)
    int N, int S,
    const __grid_constant__ CUtensorMap MAP0,   // MODE 0: out [S*N][D] 2-D box (64, BI); MODE 1: do, head-major [H*N][S*C] 2-D;
                                                //   MODE 2: out in the natural [S][N][HC] layout, 3-D box (32 c, BI tok, 2 s)
    const __grid_constant__ CUtensorMap MAP1,   // MODE 1/3: dgp, natural 3-D; MODE 0: o (when save_o)
    const __grid_constant__ CUtensorMap MAP2) { // MODE 1/3: go,  natural 3-D
  constexpr int BI = 64 * NWG, THREADS = 128 * NWG;
  using L = SM<MODE, NWG>;
  constexpr int TA = L::TA;
  extern __shared__ char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  if (sbase & 1023u) __trap();                          // the swizzles are address-based: 1 KiB alignment, no slack allocated
  __nv_bfloat16* sm = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  constexpr int NST = L::NST;
  __nv_bfloat16* sA = sm + L::A;
  __nv_bfloat16* sB = sm + L::B;
  __nv_bfloat16* sY = sm + L::Y;
  __nv_bfloat16* sWG = sm + L::WG;
  __nv_bfloat16* sWO = sm + L::WO;
  __nv_bfloat16* sDO = sm + L::DO;
  __nv_bfloat16* sOUT = sm + L::OUT;

  const int tid = threadIdx.x, lane = tid & 31, wl = (tid >> 5) & 3;
  const int wgi = NWG == 1 ? 0 : (tid >> 7);            // compile-time zero for one warpgroup
  const int i0 = blockIdx.x * BI, s0 = blockIdx.y * BS;
  const long SC = (long)S * C;

  // ---- staging: every tile is 512 sixteen-byte chunks, four per thread, in swizzled [row][64] form
  auto stage_w = [&](int h, int kj, __nv_bfloat16* dst) {          // [BI][64]: BI*8 chunks
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int v = q * THREADS + tid, r = v >> 3, kc = v & 7;
      __pipeline_memcpy_async(dst + wg::off(r, kc), W + ((long)h * N + i0 + r) * N + kj * BJ + kc * 8, 16);
    }
  };
  auto stage_v = [&](int h, int kj, __nv_bfloat16* dst) {          // [64][64]: 512 chunks
#pragma unroll
    for (int q = 0; q < 4 / NWG; ++q) {
      const int v = q * THREADS + tid, r = v >> 3, kc = v & 7;
      __pipeline_memcpy_async(dst + wg::off(r, kc), V + ((long)h * N + kj * BJ + r) * SC + (long)s0 * C + kc * 8, 16);
    }
  };
  auto stage_o = [&](int h, __nv_bfloat16* dst) {                   // MODE 3: o[s0..+2][i0..+BI][h*C..+32] -> sw64nat tile
#pragma unroll
    for (int q = 0; q < BI * BS * 4 / THREADS; ++q) {
      const int v = q * THREADS + tid, r = v >> 3, si = (v >> 2) & 1, kc = v & 3;
      __pipeline_memcpy_async(dst + tma::sw64nat<BI>(r, si, kc * 8), V + ((long)(s0 + si) * N + i0 + r) * HC + h * C + kc * 8, 16);
    }
  };
  auto stage_head = [&](int h) {                       // Wg_h and Wo_h into the parity buffers
    if (MODE == 2) return;
    __nv_bfloat16* dg = sWG;
    __nv_bfloat16* dco = sWO;
#pragma unroll
    for (int q = 0; q < 256 / THREADS; ++q) {
      const int v = q * THREADS + tid, r = v >> 3, kc = v & 7;            // Wg_h: 32 rows c x 8 chunks of d
      __pipeline_memcpy_async(dg + wg::off(r, kc), WG + ((long)h * C + r) * D + kc * 8, 16);
    }
    if (MODE == 0) {
#pragma unroll
      for (int q = 0; q < 256 / THREADS; ++q) {
        const int v = q * THREADS + tid, r = v >> 2, kc = v & 3;          // Wo_h: 64 rows d x 4 chunks of c, 64-byte rows
        __pipeline_memcpy_async(dco + wg::off64(r, kc), WO + (long)r * HC + h * C + kc * 8, 16);
      }
    } else {
#pragma unroll
      for (int q = 0; q < 256 / THREADS; ++q) {
        const int v = q * THREADS + tid, r = v >> 3, kc = v & 7;          // Wo^T_h: 32 rows c x 8 chunks of d
        __pipeline_memcpy_async(dco + wg::off(r, kc), WO + ((long)h * C + r) * D + kc * 8, 16);
      }
    }
  };
  // the y (and dout) tiles, once: [BI i][64 d] for each of the two s
  if (MODE != 2) {
#pragma unroll
    for (int si = 0; si < BS; ++si)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int v = q * THREADS + tid, r = v >> 3, kc = v & 7;
        __pipeline_memcpy_async(sY + si * TA + wg::off(r, kc), Y + ((long)(s0 + si) * N + i0 + r) * D + kc * 8, 16);
        if (MODE == 1 || MODE == 3)
          __pipeline_memcpy_async(sDO + si * TA + wg::off(r, kc), DOUT + ((long)(s0 + si) * N + i0 + r) * D + kc * 8, 16);
      }
  }
  stage_head(0);
  if (MODE == 3) {
    stage_o(0, sA);
  } else {
    stage_w(0, 0, sA);
    stage_v(0, 0, sB);
  }
  __pipeline_commit();
  const int nkj0 = MODE == 3 ? 1 : N / BJ;
  __pipeline_commit();                                 // (an empty group keeps the wait_prior accounting uniform)

  float oacc[32], gacc[16], duacc[16], out0[32], out1[32];
#pragma unroll
  for (int t = 0; t < 32; ++t) { oacc[t] = 0.f; out0[t] = 0.f; out1[t] = 0.f; }
#pragma unroll
  for (int t = 0; t < 16; ++t) { gacc[t] = 0.f; duacc[t] = 0.f; }
  const int r_acc = wgi * 64 + wl * 16 + (lane >> 2);  // this thread's accumulator rows r_acc and r_acc + 8 (of BI)
  const int wgo = wgi * 64 * 64;                       // this warpgroup's 64-row slice of an A-side tile

  const int nkj = nkj0;
  const int nstage = H * nkj;
#pragma unroll 1
  for (int t = 0; t < nstage; ++t) {
    const int h = t / nkj, kj = t - h * nkj, buf = t % NST;
    if (MODE == 3) {
      // o tiles: two buffers, prefetch at the top (they are read with plain loads, retired at the barrier).
      // The buffer being refilled carried the previous head's `do` tile out through TMA: wait for that read.
      if (t + 1 < nstage) {
        if (t >= 1 && tid == 0) tma::wait_read();
        __syncthreads();
        stage_o(t + 1, sA + ((t + 1) & 1) * TA);
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
    } else {
      // stage t+1 goes into the buffer group t-2 read, retired by the previous iteration's wait_group 1
      if (MODE == 0 && save_o && kj == 0 && t > 0) {
        // ... unless it still holds the previous head's o tile, in flight to global through TMA
        if (tid == 0) tma::wait_read();
        __syncthreads();
      }
      if (t + 1 < nstage) {
        const int h1 = (t + 1) / nkj, kj1 = (t + 1) - h1 * nkj;
        stage_w(h1, kj1, sA + ((t + 1) % NST) * TA);
        stage_v(h1, kj1, sB + ((t + 1) % NST) * TILE);
      }
      __pipeline_commit();
      // stage t must be complete.  The most recent group is stage t+1; right after an epilogue the one
      // before it is the next head's weights, needed only at the next epilogue -- let both stay in flight.
      if ((MODE == 0 || MODE == 1) && kj == 0 && t > 0) __pipeline_wait_prior(2); else __pipeline_wait_prior(1);
    }
    __syncthreads();
    const __nv_bfloat16* a = sA + buf * TA + wgo;
    const __nv_bfloat16* b = sB + buf * TILE;
    if (MODE == 3) {
      // the saved o tile, read in the accumulator's own layout
      const __nv_bfloat16* so = sA + buf * TA;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int si = nt >> 2, c = (nt & 3) * 8 + (lane & 3) * 2;
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so + tma::sw64nat<BI>(r_acc, si, c)));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so + tma::sw64nat<BI>(r_acc + 8, si, c)));
        oacc[nt * 4 + 0] = lo.x; oacc[nt * 4 + 1] = lo.y; oacc[nt * 4 + 2] = hi.x; oacc[nt * 4 + 3] = hi.y;
      }
    } else {
    wg::fence();
#pragma unroll
    for (int ks = 0; ks < BJ / 16; ++ks)
      wg::mma_m64n64k16_kmn(wg::desc(a + wg::off(0, ks * 2)), wg::desc_mn(b + ks * 16 * 64), oacc, (kj == 0 && ks == 0) ? 0 : 1);
    wg::commit();
    }
    // Leave this chunk's group in flight: the next iteration issues its cp.async and reaches its barrier
    // while the tensor pipe is still busy.  Only the PREVIOUS group must be complete -- it is the one
    // whose operand buffers the next prefetch overwrites.  The epilogue below reads the accumulator, so
    // it drains fully first.
    if (MODE != 3) { if (kj == nkj - 1) wg::wait(); else wg::wait1(); }
    if (MODE == 0 && save_o && kj == nkj - 1) {
      // o for the backward: this head's tile in the natural layout, staged on the just-consumed A tile
      __nv_bfloat16* so = sA + buf * TA;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int si = nt >> 2, c = (nt & 3) * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(so + tma::sw64nat<BI>(r_acc, si, c)) = pack2(oacc[nt * 4 + 0], oacc[nt * 4 + 1]);
        *reinterpret_cast<uint32_t*>(so + tma::sw64nat<BI>(r_acc + 8, si, c)) = pack2(oacc[nt * 4 + 2], oacc[nt * 4 + 3]);
      }
      wg::proxy_fence();
      __syncthreads();
      if (tid == 0) {
        tma::store_3d(&MAP1, static_cast<uint32_t>(__cvta_generic_to_shared(so)), h * C, i0, s0);
        tma::commit();
      }
    }
    if (MODE == 2 && kj == nkj - 1) {
      // ---- plain: the accumulator tile [64 rows][64 (s,c)] -> bf16 swizzled over this stage's B tile
      // (just consumed; the prefetch went to the other buffer) -> one TMA store to the head-major output
      if (tid == 0) tma::wait_read();                    // the previous head's store must be done reading it
      __syncthreads();
      __nv_bfloat16* so = sOUT;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int si = nt >> 2, c = (nt & 3) * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(so + tma::sw64nat<BI>(r_acc, si, c)) = pack2(oacc[nt * 4 + 0], oacc[nt * 4 + 1]);
        *reinterpret_cast<uint32_t*>(so + tma::sw64nat<BI>(r_acc + 8, si, c)) = pack2(oacc[nt * 4 + 2], oacc[nt * 4 + 3]);
      }
      wg::proxy_fence();
      __syncthreads();
      if (tid == 0) {
        tma::store_3d(&MAP0, static_cast<uint32_t>(__cvta_generic_to_shared(so)), h * C, i0, s0);
        tma::commit();
      }
    }
    if ((MODE == 1 || MODE == 3) && kj == nkj - 1) {
      __syncthreads();                                 // nobody may still be in the previous epilogue's weight reads
      // ---- glue: gate and du projections per s, then the three elementwise outputs into a 3-tile
      // staging area, one TMA store each per head
      const __nv_bfloat16* wgt = sWG;
      const __nv_bfloat16* wot = sWO;
      __nv_bfloat16* sDOut = MODE == 3 ? sA + buf * TA : sOUT + 2 * TA;   // MODE 3: over this head's consumed o tile
      if (tid == 0) tma::wait_read();                    // the previous head's stores (dgp/go staging is reused)
      __syncthreads();
#pragma unroll
      for (int si = 0; si < BS; ++si) {
        wg::fence();
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          wg::mma_m64n32k16_kk(wg::desc(sY + si * TA + wgo + wg::off(0, ks * 2)), wg::desc(wgt + wg::off(0, ks * 2)), gacc, ks == 0 ? 0 : 1);
          wg::mma_m64n32k16_kk(wg::desc(sDO + si * TA + wgo + wg::off(0, ks * 2)), wg::desc(wot + wg::off(0, ks * 2)), duacc, ks == 0 ? 0 : 1);
        }
        wg::commit();
        wg::wait();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float* o = oacc + (si * 4 + nt) * 4;
          const int c = si * C + nt * 8 + (lane & 3) * 2;                 // column in the [64][64] (s,c) tile
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int rr = r_acc + half * 8;
            const float g0 = 1.f / (1.f + __expf(-gacc[nt * 4 + half * 2])), g1 = 1.f / (1.f + __expf(-gacc[nt * 4 + half * 2 + 1]));
            const float d0 = duacc[nt * 4 + half * 2], d1 = duacc[nt * 4 + half * 2 + 1];
            const float o0 = o[half * 2], o1 = o[half * 2 + 1];
            const int at = tma::sw128(rr, c), bt = tma::sw64nat<BI>(rr, si, c - si * C);
            *reinterpret_cast<uint32_t*>(sDOut + at) = pack2(d0 * g0, d1 * g1);                                            // do,  head-major
            *reinterpret_cast<uint32_t*>(sOUT + 0 * TA + bt) = pack2(d0 * o0 * g0 * (1.f - g0), d1 * o1 * g1 * (1.f - g1)); // dgp, natural
            *reinterpret_cast<uint32_t*>(sOUT + 1 * TA + bt) = pack2(g0 * o0, g1 * o1);                                    // go,  natural
          }
        }
      }
      wg::proxy_fence();
      __syncthreads();
      if (tid == 0) {
        tma::store_2d(&MAP0, static_cast<uint32_t>(__cvta_generic_to_shared(sDOut)), s0 * C, h * N + i0);
        tma::store_3d(&MAP1, static_cast<uint32_t>(__cvta_generic_to_shared(sOUT + 0 * TA)), h * C, i0, s0);
        tma::store_3d(&MAP2, static_cast<uint32_t>(__cvta_generic_to_shared(sOUT + 1 * TA)), h * C, i0, s0);
        tma::commit();
      }
      __syncthreads();                                 // every warp is past this head's weight tiles
      if (h + 1 < H) { stage_head(h + 1); __pipeline_commit(); }
    }
    if (MODE == 0 && kj == nkj - 1) {
      // ---- epilogue of head h: gate, multiply, out-projection, one s at a time
      const __nv_bfloat16* wgt = sWG;
      const __nv_bfloat16* wot = sWO;
#pragma unroll
      for (int si = 0; si < BS; ++si) {
        wg::fence();
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          wg::mma_m64n32k16_kk(wg::desc(sY + si * TA + wgo + wg::off(0, ks * 2)), wg::desc(wgt + wg::off(0, ks * 2)), gacc, ks == 0 ? 0 : 1);
        wg::commit();
        wg::wait();
        // u = sigmoid(gate) * o on the 32 columns of this s, straight into A fragments
        uint32_t af[8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float* o = oacc + (si * 4 + nt) * 4;
          const float* g = gacc + nt * 4;
          const float u0 = o[0] / (1.f + __expf(-g[0])), u1 = o[1] / (1.f + __expf(-g[1]));
          const float u2 = o[2] / (1.f + __expf(-g[2])), u3 = o[3] / (1.f + __expf(-g[3]));
          af[nt * 2 + 0] = pack2(u0, u1);      // row r,     k = nt*8 + (lane&3)*2
          af[nt * 2 + 1] = pack2(u2, u3);      // row r + 8
        }
        float* out = si == 0 ? out0 : out1;
        wg::fence();
        // k-step 0: n tiles 0,1 -> a0 a1 a2 a3 = (nt0 r), (nt0 r+8), (nt1 r), (nt1 r+8)
        wg::mma_m64n64k16_rs(af[0], af[1], af[2], af[3], wg::desc64(wot + wg::off64(0, 0)), out, h == 0 ? 0 : 1);
        wg::mma_m64n64k16_rs(af[4], af[5], af[6], af[7], wg::desc64(wot + wg::off64(0, 2)), out, 1);
        wg::commit();
        wg::wait();
      }
      __syncthreads();                                 // every warp is past this head's weight tiles
      if (h + 1 < H) { stage_head(h + 1); __pipeline_commit(); }
    }
    __syncthreads();
  }

  __pipeline_wait_prior(0);
  if (MODE == 0) {
    // ---- the residual: the msa tiles come in over the dead A stages (coalesced, swizzled), are added
    // in registers, and the sum goes back out through the same tile: out = msa + update
    if (save_o && tid == 0) tma::wait_read();
    __syncthreads();
    __nv_bfloat16* sO = sA;                            // 2 x [BI][64], aliases the A stages
#pragma unroll
    for (int si = 0; si < BS; ++si)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int v = q * THREADS + tid, r = v >> 3, kc = v & 7;
        __pipeline_memcpy_async(sO + si * TA + wg::off(r, kc), RES + ((long)(s0 + si) * N + i0 + r) * D + kc * 8, 16);
      }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
#pragma unroll
    for (int si = 0; si < BS; ++si) {
      float* out = si == 0 ? out0 : out1;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = nt * 8 + (lane & 3) * 2;
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sO + si * TA + tma::sw128(r_acc, c)));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sO + si * TA + tma::sw128(r_acc + 8, c)));
        // the stock module rounds the update to bf16 before the residual add
        const float u0 = __bfloat162float(__float2bfloat16(out[nt * 4 + 0])), u1 = __bfloat162float(__float2bfloat16(out[nt * 4 + 1]));
        const float u2 = __bfloat162float(__float2bfloat16(out[nt * 4 + 2])), u3 = __bfloat162float(__float2bfloat16(out[nt * 4 + 3]));
        *reinterpret_cast<uint32_t*>(sO + si * TA + tma::sw128(r_acc, c)) = pack2(lo.x + u0, lo.y + u1);
        *reinterpret_cast<uint32_t*>(sO + si * TA + tma::sw128(r_acc + 8, c)) = pack2(hi.x + u2, hi.y + u3);
      }
    }
    wg::proxy_fence();
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int si = 0; si < BS; ++si)
        tma::store_2d(&MAP0, static_cast<uint32_t>(__cvta_generic_to_shared(sO + si * TA)), 0, (s0 + si) * N + i0);
      tma::commit();
    }
  }
  if (tid == 0) tma::wait_all();
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
}  // namespace

namespace {
CUtensorMap map2d(const torch::Tensor& t, long rows, long cols, int box_rows) {   // [rows][cols] bf16, box 64 x box_rows, 128B swizzle
  alignas(64) CUtensorMap m{};
  uint64_t gdim[2] = {(uint64_t)cols, (uint64_t)rows};
  uint64_t gstride[1] = {(uint64_t)cols * 2};
  uint32_t bdim[2] = {64, (uint32_t)box_rows};
  uint32_t estride[2] = {1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
CUtensorMap map_nat(void* base, long row_stride, int N, int S, int bi) {   // natural [S][N][row_stride] bf16 with HC live columns at base, box (32 c, bi tok, 2 s), 64B swizzle
  alignas(64) CUtensorMap m{};
  uint64_t gdim[3] = {(uint64_t)HC, (uint64_t)N, (uint64_t)S};
  uint64_t gstride[2] = {(uint64_t)row_stride * 2, (uint64_t)N * row_stride * 2};
  uint32_t bdim[3] = {32, (uint32_t)bi, BS};
  uint32_t estride[3] = {1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (token-major) failed: ", (int)r);
  return m;
}
const __nv_bfloat16* bp(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }

struct Args {
  const __nv_bfloat16 *w = nullptr, *v = nullptr, *res = nullptr, *y = nullptr, *dout = nullptr, *wg = nullptr, *wo = nullptr;
  int save_o = 0, N = 0, S = 0;
};
template <int MODE, int NWG>
void launch_t(const Args& a, const CUtensorMap& m0, const CUtensorMap& m1, const CUtensorMap& m2) {
  constexpr int BYTES = SM<MODE, NWG>::BYTES;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(pwa_ctr_kernel<MODE, NWG>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pwa_ctr_kernel<MODE, NWG>, 128 * NWG, BYTES);
    TORCH_WARN("pwa_ctr_kernel<", MODE, ",", NWG, ">: ", BYTES, " B smem -> ", nb, " blocks/SM");
    attr = true;
  }
  TORCH_CHECK(a.N % (64 * NWG) == 0, "N must be a multiple of ", 64 * NWG);
  dim3 grid(a.N / (64 * NWG), a.S / BS);
  pwa_ctr_kernel<MODE, NWG><<<grid, 128 * NWG, BYTES, at::cuda::getCurrentCUDAStream()>>>(
      a.w, a.v, a.res, a.save_o, a.y, a.dout, a.wg, a.wo, a.N, a.S, m0, m1, m2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int MODE>
void launch(int nwg, const Args& a, const CUtensorMap& m0, const CUtensorMap& m1, const CUtensorMap& m2) {
  if (nwg == 2) launch_t<MODE, 2>(a, m0, m1, m2);
  else launch_t<MODE, 1>(a, m0, m1, m2);
}
void check_common(const torch::Tensor& w16, const torch::Tensor& vhm, int& N, int& S) {
  TORCH_CHECK(w16.is_cuda() && w16.scalar_type() == torch::kBFloat16 && w16.is_contiguous() && w16.dim() == 3 && w16.size(0) == H, "w16: [H, N, N] bf16");
  N = (int)w16.size(1);
  TORCH_CHECK(vhm.is_contiguous() && vhm.dim() == 3 && vhm.size(0) == H && vhm.size(1) == N && vhm.size(2) % C == 0, "vhm: [H, N, S*C]");
  S = (int)(vhm.size(2) / C);
  TORCH_CHECK(N % 64 == 0 && S % BS == 0, "tiles of 64 i x ", BS, " s");
}
}  // namespace

std::vector<torch::Tensor> pwa_fwd(torch::Tensor w16, torch::Tensor vhm, torch::Tensor y, torch::Tensor wgw, torch::Tensor wow,
                                   torch::Tensor msa, int64_t nwg, bool save_o) {
  int N, S; check_common(w16, vhm, N, S);
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}), "y: [S, N, D]");
  TORCH_CHECK(msa.is_contiguous() && msa.sizes() == torch::IntArrayRef({S, N, D}), "msa: [S, N, D]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}), "wg: [HC, D]");
  TORCH_CHECK(wow.is_contiguous() && wow.sizes() == torch::IntArrayRef({D, HC}), "wo: [D, HC]");
  auto out = torch::empty({S, N, D}, y.options());                  // msa + update
  auto o = save_o ? torch::empty({S, N, HC}, y.options()) : torch::empty({0}, y.options());
  CUtensorMap m0 = map2d(out, (long)S * N, D, 64 * (int)nwg);
  CUtensorMap m1 = save_o ? map_nat(o.data_ptr(), HC, N, S, 64 * (int)nwg) : m0;
  Args a; a.w = bp(w16); a.v = bp(vhm); a.res = bp(msa); a.save_o = save_o ? 1 : 0; a.y = bp(y); a.wg = bp(wgw); a.wo = bp(wow); a.N = N; a.S = S;
  launch<0>((int)nwg, a, m0, m1, m0);
  return {out, o};
}

std::vector<torch::Tensor> pwa_glue(torch::Tensor w16, torch::Tensor vhm, torch::Tensor y, torch::Tensor dout,
                                    torch::Tensor wgw, torch::Tensor wot, int64_t nwg) {
  int N, S; check_common(w16, vhm, N, S);
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}), "y: [S, N, D]");
  TORCH_CHECK(dout.is_contiguous() && dout.sizes() == torch::IntArrayRef({S, N, D}), "dout: [S, N, D]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}), "wg: [HC, D]");
  TORCH_CHECK(wot.is_contiguous() && wot.sizes() == torch::IntArrayRef({HC, D}), "wot: Wo^T [HC, D]");
  auto d_o = torch::empty_like(vhm);                                    // head-major: dv and dw read it as-is
  auto dgp = torch::empty({S, N, HC}, vhm.options()), go = torch::empty({S, N, HC}, vhm.options());   // natural layout: one GEMM each downstream
  CUtensorMap m0 = map2d(d_o, (long)H * N, (long)S * C, 64 * (int)nwg), m1 = map_nat(dgp.data_ptr(), HC, N, S, 64 * (int)nwg), m2 = map_nat(go.data_ptr(), HC, N, S, 64 * (int)nwg);
  Args a; a.w = bp(w16); a.v = bp(vhm); a.y = bp(y); a.dout = bp(dout); a.wg = bp(wgw); a.wo = bp(wot); a.N = N; a.S = S;
  launch<1>((int)nwg, a, m0, m1, m2);
  return {d_o, dgp, go};
}

std::vector<torch::Tensor> pwa_glue_o(torch::Tensor o, torch::Tensor y, torch::Tensor dout, torch::Tensor wgw, torch::Tensor wot, int64_t nwg,
                                      c10::optional<torch::Tensor> dgv) {   // dgv: [S][N][2*HC]; dgp lands in its first HC columns
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.dim() == 3 && o.size(2) == HC, "o: [S, N, HC] bf16");
  const int S = (int)o.size(0), N = (int)o.size(1);
  TORCH_CHECK(N % 64 == 0 && S % BS == 0, "tiles of 64 i x ", BS, " s");
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}), "y: [S, N, D]");
  TORCH_CHECK(dout.is_contiguous() && dout.sizes() == torch::IntArrayRef({S, N, D}), "dout: [S, N, D]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}), "wg: [HC, D]");
  TORCH_CHECK(wot.is_contiguous() && wot.sizes() == torch::IntArrayRef({HC, D}), "wot: Wo^T [HC, D]");
  auto d_o = torch::empty({H, N, (long)S * C}, o.options());
  auto go = torch::empty({S, N, HC}, o.options());
  torch::Tensor dgp;
  long rs = HC;
  if (dgv.has_value()) {
    TORCH_CHECK(dgv->is_contiguous() && dgv->sizes() == torch::IntArrayRef({S, N, 2 * HC}), "dgv: [S, N, 2*HC]");
    dgp = *dgv; rs = 2 * HC;
  } else {
    dgp = torch::empty({S, N, HC}, o.options());
  }
  CUtensorMap m0 = map2d(d_o, (long)H * N, (long)S * C, 64 * (int)nwg), m1 = map_nat(dgp.data_ptr(), rs, N, S, 64 * (int)nwg), m2 = map_nat(go.data_ptr(), HC, N, S, 64 * (int)nwg);
  Args a; a.v = bp(o); a.y = bp(y); a.dout = bp(dout); a.wg = bp(wgw); a.wo = bp(wot); a.N = N; a.S = S;
  launch<3>((int)nwg, a, m0, m1, m2);
  return {d_o, dgp, go};
}

torch::Tensor pwa_plain(torch::Tensor w16, torch::Tensor xhm, int64_t nwg, c10::optional<torch::Tensor> dgv) {   // dgv: dv lands in columns HC..2HC
  int N, S; check_common(w16, xhm, N, S);
  torch::Tensor out;
  CUtensorMap m0;
  if (dgv.has_value()) {
    TORCH_CHECK(dgv->is_contiguous() && dgv->sizes() == torch::IntArrayRef({S, N, 2 * HC}), "dgv: [S, N, 2*HC]");
    out = *dgv;
    m0 = map_nat(reinterpret_cast<char*>(dgv->data_ptr()) + HC * 2, 2 * HC, N, S, 64 * (int)nwg);
  } else {
    out = torch::empty({S, N, HC}, xhm.options());                    // natural layout
    m0 = map_nat(out.data_ptr(), HC, N, S, 64 * (int)nwg);
  }
  Args a; a.w = bp(w16); a.v = bp(xhm); a.N = N; a.S = S;
  launch<2>((int)nwg, a, m0, m0, m0);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pwa_fwd", &pwa_fwd, "PWA forward: contraction + gate + proj_o + residual (-> msa + update), optionally saving o", py::arg("w16"), py::arg("vhm"), py::arg("y"), py::arg("wg"), py::arg("wo"), py::arg("msa"), py::arg("nwg") = 1, py::arg("save_o") = false);
  m.def("pwa_glue_o", &pwa_glue_o, "PWA backward glue from a saved o (no contraction)", py::arg("o"), py::arg("y"), py::arg("dout"), py::arg("wg"), py::arg("wot"), py::arg("nwg") = 1, py::arg("dgv") = py::none());
  m.def("pwa_glue", &pwa_glue, "PWA backward: contraction + du/gate glue -> (do head-major, dgp, go natural)", py::arg("w16"), py::arg("vhm"), py::arg("y"), py::arg("dout"), py::arg("wg"), py::arg("wot"), py::arg("nwg") = 2);
  m.def("pwa_plain", &pwa_plain, "PWA contraction alone (dv with w^T), natural layout out", py::arg("w16"), py::arg("xhm"), py::arg("nwg") = 2, py::arg("dgv") = py::none());
}
