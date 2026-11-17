// PWA gate / out pass (sm_90a), for a contraction done elsewhere (cuBLAS bmm(w, v) -> o head-major [H][N][S*C]):
//   out[s,i,:] = msa[s,i,:] + sum_h (o_h .* sigmoid(y[s,i,:] Wg_h^T)) Wo_h        (the epilogue of pwa_fwd3, unchanged)
// Same 384-thread block as pwa_fwd3: two consumer warpgroups own the two 64-row i-tiles of one i-pair for one s-pair; a producer
// thread streams, per head, one o tile per consumer ([64 i][64 = 2 s x 32 c], 128-byte swizzle, the shape of pwa_fwd3's v tile)
// through an NST-deep ring plus the head's Wg_h / Wo_h.  The consumer reads its o fragment from the tile in accumulator layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include "sm90_ptx.cuh"

namespace {
constexpr int H = 8, C = 32, D = 64, HC = H * C;
constexpr int BS = 2, BJ = 64, NB = BS * C;
constexpr int TILE = 64 * 64;                            // 8 KiB bf16 tile
constexpr int THREADS = 384;
constexpr int JPS = 2;                                   // j-chunks per stage (N % 128 == 0 -> an even number of j-chunks per head)
constexpr int STG = JPS * 3 * TILE;                        // stage: [JPS][W c0 | W c1 | v] = 48 KiB

namespace wg {
using csrc_sm90::wg::commit;
using csrc_sm90::wg::fence;
using csrc_sm90::wg::mma_m64n32k16_kk;
using csrc_sm90::wg::mma_m64n32k16_rs;
using csrc_sm90::wg::mma_m64n64k16_kmn;
using csrc_sm90::wg::mma_m64n64k16_rs;
using csrc_sm90::wg::proxy_fence;
using csrc_sm90::wg::reg_alloc;
using csrc_sm90::wg::reg_dealloc;
using csrc_sm90::wg::wait;
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

}  // namespace wg

namespace tma {
using csrc_sm90::tma::arrive;
using csrc_sm90::tma::arrive_remote;
using csrc_sm90::tma::bar_init;
using csrc_sm90::tma::bar_init_fence;
using csrc_sm90::tma::cluster_rank;
using csrc_sm90::tma::cluster_sync;
using csrc_sm90::tma::commit;
using csrc_sm90::tma::expect_tx;
using csrc_sm90::tma::load_2d;
using csrc_sm90::tma::load_2d_mc;
using csrc_sm90::tma::sa;
using csrc_sm90::tma::store_2d;
using csrc_sm90::tma::store_3d;
using csrc_sm90::tma::wait;
using csrc_sm90::tma::wait_all;
using csrc_sm90::tma::wait_cluster;
using csrc_sm90::tma::wait_read;
__device__ __forceinline__ int sw128(int row, int col64) { return row * 64 + (((col64 >> 3) ^ (row & 7)) << 3) + (col64 & 7); }
template <int BI>
__device__ __forceinline__ int sw64nat(int tok, int si, int c) {
  const int line = si * BI + tok;
  return line * 32 + ((((c >> 3) ^ ((line >> 1) & 3))) << 3) + (c & 7);
}
}  // namespace tma

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}


// smem (bf16 elements from a 1 KiB-aligned base)
template <int NST> struct SMG {
  static constexpr int SRING = 0;                                   // [NST][2 consumers][TILE]: the o tiles of one head
  static constexpr int SWG = SRING + NST * 2 * TILE;                // [2 head parity][32*64] Wg_h
  static constexpr int SWO = SWG + 2 * 32 * 64;                     // [2 head parity][64*32] Wo_h (64B swizzle)
  static constexpr int SOUT = SWO + 2 * 64 * 32;                    // [2 consumers][2 s][TILE] output staging (residual in, sum out)
  static constexpr int SEND = SOUT + 4 * TILE;
  static constexpr int NBAR = 2 * NST + 2 + 2 + 2;                  // full, empty, fullW[2], wfree[2], resB[2]
  static constexpr int BYTES = SEND * 2 + NBAR * 8 + 1024;
};

template <int NST>
__global__ void __launch_bounds__(THREADS, 1) pwa_gateout_kernel(int N, int S, int ntile,
    const __nv_bfloat16* __restrict__ Yg,        // y   [S*N][D]        (read as RS fragments)
    const __nv_bfloat16* __restrict__ DMASK,     // drop_msa keep-mask [N][D] (0/1, shared over s) or nullptr
    float dscale,                                // 1 / (1 - p)
    const __grid_constant__ CUtensorMap otmap,   // o   [H*N][S*C]      box (64 (s,c), 64 i)
    const __grid_constant__ CUtensorMap rmap,    // msa [S*N][D]        box (64, 64)
    const __grid_constant__ CUtensorMap gmap,    // wg  [HC][D]         box (64, 32)
    const __grid_constant__ CUtensorMap omap_w,  // wo  [D][HC]         box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap outmap) { // out [S*N][D]       box (64, 64)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  unsigned char* smb = smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  __nv_bfloat16* sm = reinterpret_cast<__nv_bfloat16*>(smb);
  using L = SMG<NST>;
  __nv_bfloat16* sRing = sm + L::SRING;
  __nv_bfloat16* sWGb = sm + L::SWG;
  __nv_bfloat16* sWOb = sm + L::SWO;
  __nv_bfloat16* sOUTb = sm + L::SOUT;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smb + L::SEND * 2);
  uint64_t* full = bars;                 // [NST]
  uint64_t* empty = full + NST;          // [NST], count 8 (the consumer warps)
  uint64_t* fullW = empty + NST;         // [2] head parity
  uint64_t* wfree = fullW + 2;           // [2], count 2
  uint64_t* resB = wfree + 2;            // [2] per consumer: residual tiles landed
  const int tid = threadIdx.x, lane = tid & 31;
  const int NI2 = N / 128;                                       // i-pairs
  auto tile_sp = [&](int t) { return t / NI2; };
  auto tile_ip = [&](int t) { return t % NI2; };

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { tma::bar_init(full + i, 1); tma::bar_init(empty + i, 8); }
    for (int i = 0; i < 2; ++i) { tma::bar_init(fullW + i, 1); tma::bar_init(wfree + i, 2); tma::bar_init(resB + i, 1); }
    tma::bar_init_fence();
    wg::proxy_fence();
  }
  __syncthreads();

  if (tid >= 256) {
    // ===================== producer: one thread =====================
    wg::reg_dealloc();
    if (tid == 256) {
      int gs = 0;                                                    // global head-stage counter
      for (int T = 0;; ++T) {
        const int t = (int)blockIdx.x + T * (int)gridDim.x;
        if (t >= ntile) break;
        const int sp = tile_sp(t), ip = tile_ip(t), i0 = ip * 128;
        for (int h = 0; h < H; ++h, ++gs) {
          const int gh = T * H + h, hb = gh & 1;
          if (gh >= 2) tma::wait(wfree + hb, ((gh >> 1) - 1) & 1);
          tma::expect_tx(fullW + hb, (32 * 64 + 64 * 32) * 2);
          tma::load_2d(&gmap, tma::sa(sWGb + hb * 32 * 64), fullW + hb, 0, h * C);
          tma::load_2d(&omap_w, tma::sa(sWOb + hb * 64 * 32), fullW + hb, h * C, 0);
          const int st = gs % NST;
          if (gs >= NST) tma::wait(empty + st, ((gs / NST) - 1) & 1);
          tma::expect_tx(full + st, 2 * TILE * 2);
          for (int cc = 0; cc < 2; ++cc)
            tma::load_2d(&otmap, tma::sa(sRing + (st * 2 + cc) * TILE), full + st, sp * NB, h * N + i0 + cc * 64);
        }
      }
    }
  } else {

  // ===================== consumer c =====================
  wg::reg_alloc();
  const int c = tid >> 7, wtid = tid & 127, wl = wtid >> 5;
  auto wg_sync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(1 + c) : "memory"); };
  const int r_acc = wl * 16 + (lane >> 2);
  __nv_bfloat16* sOUT = sOUTb + c * 2 * TILE;
  float ov[32], gacc0[16], gacc1[16], out0[32], out1[32];
  uint32_t yf[2][16];                                            // y tiles of this tile as RS A-fragments: [s][k-step*4 + {r, r+8, r k+8, r+8 k+8}]
  const int cq = (lane & 3) * 2;
  auto load_y = [&](int T) {                                     // plain global loads straight into the fragment layout
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
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
  int gs = 0;
  load_y(0);
  for (int T = 0;; ++T) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    if (t >= ntile) break;
    const int sp = tile_sp(t), ip = tile_ip(t), i0 = ip * 128 + c * 64, s0 = sp * BS;
#pragma unroll
    for (int q = 0; q < 32; ++q) { out0[q] = 0.f; out1[q] = 0.f; }
    for (int h = 0; h < H; ++h, ++gs) {
      const int gh = T * H + h, hb = gh & 1;
      if (h == H / 2 && wtid == 0) {                               // residual tiles into the output staging, once the previous tile's out store
        tma::wait_read<0>();                                       // has finished reading it
        tma::expect_tx(resB + c, 2 * TILE * 2);
        for (int si = 0; si < BS; ++si) tma::load_2d(&rmap, tma::sa(sOUT + si * TILE), resB + c, 0, (s0 + si) * N + i0);
      }
      tma::wait(fullW + hb, (gh >> 1) & 1);
      const __nv_bfloat16* wgt = sWGb + hb * 32 * 64;
      const __nv_bfloat16* wot = sWOb + hb * 64 * 32;
      // gate GEMMs of this head (both s), A = the y fragments in registers
      wg::fence();
#pragma unroll
      for (int si = 0; si < BS; ++si)
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          wg::mma_m64n32k16_rs(yf[si][ks * 4 + 0], yf[si][ks * 4 + 1], yf[si][ks * 4 + 2], yf[si][ks * 4 + 3], wg::desc(wgt + wg::off(0, ks * 2)), si == 0 ? gacc0 : gacc1, ks == 0 ? 0 : 1);
      wg::commit();
      if (h > 0) {                                                 // the previous head's out-projection group is older than this gate group: once it
        wg::wait<1>();                                             // has retired every warp is past that head's Wo_h / Wg_h tiles
        wg_sync();
        if (wtid == 0) tma::arrive(wfree + ((gh - 1) & 1));
      }
      // ---- o fragment of this head from the cuBLAS tile (accumulator layout: rows r_acc / r_acc + 8, cols nt * 8 + cq) ----
      const int st = gs % NST;
      tma::wait(full + st, (gs / NST) & 1);
      const __nv_bfloat16* so_t = sRing + (st * 2 + c) * TILE;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so_t + tma::sw128(r_acc, nt * 8 + cq)));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so_t + tma::sw128(r_acc + 8, nt * 8 + cq)));
        ov[nt * 4 + 0] = lo.x; ov[nt * 4 + 1] = lo.y; ov[nt * 4 + 2] = hi.x; ov[nt * 4 + 3] = hi.y;
      }
      __syncwarp();
      if (lane == 0) tma::arrive(empty + st);
      wg::wait<0>();                                               // the gate accumulators are complete
      // ---- multiply by the gate and out-project, one s at a time ----
      wg::fence();
#pragma unroll
      for (int si = 0; si < BS; ++si) {
        uint32_t af[8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float* o = ov + (si * 4 + nt) * 4;
          const float* g = (si == 0 ? gacc0 : gacc1) + nt * 4;
          const float u0 = o[0] / (1.f + __expf(-g[0])), u1 = o[1] / (1.f + __expf(-g[1]));   // exactly the reference kernel's arithmetic
          const float u2 = o[2] / (1.f + __expf(-g[2])), u3 = o[3] / (1.f + __expf(-g[3]));
          af[nt * 2 + 0] = pack2(u0, u1);
          af[nt * 2 + 1] = pack2(u2, u3);
        }
        float* out = si == 0 ? out0 : out1;
        wg::mma_m64n64k16_rs(af[0], af[1], af[2], af[3], wg::desc64(wot + wg::off64(0, 0)), out, 1);
        wg::mma_m64n64k16_rs(af[4], af[5], af[6], af[7], wg::desc64(wot + wg::off64(0, 2)), out, 1);
      }
      wg::commit();                                                  // NOT waited: it retires under the next head's gate GEMM
      if (h == H - 1) {
        wg::wait<0>();
        wg_sync();                                                   // every warp is past this head's weight tiles
        if (wtid == 0) tma::arrive(wfree + hb);
        load_y(T + 1);                                               // the y fragments are free: prefetch the next tile's into them
      }
    }
    // ---- (drop_msa) residual add and the output store ----
    tma::wait(resB + c, T & 1);
    float dm[16][2];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int cc = nt * 8 + (lane & 3) * 2;
      if (DMASK != nullptr) {
        const float2 m0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(DMASK + (long)(i0 + r_acc) * D + cc));
        const float2 m1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(DMASK + (long)(i0 + r_acc + 8) * D + cc));
        dm[nt * 2][0] = m0.x * dscale; dm[nt * 2][1] = m0.y * dscale; dm[nt * 2 + 1][0] = m1.x * dscale; dm[nt * 2 + 1][1] = m1.y * dscale;
      } else {
        dm[nt * 2][0] = dm[nt * 2][1] = dm[nt * 2 + 1][0] = dm[nt * 2 + 1][1] = 1.f;
      }
    }
#pragma unroll
    for (int si = 0; si < BS; ++si) {
      float* out = si == 0 ? out0 : out1;
      __nv_bfloat16* so = sOUT + si * TILE;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int cc = nt * 8 + (lane & 3) * 2;
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so + tma::sw128(r_acc, cc)));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(so + tma::sw128(r_acc + 8, cc)));
        // the stock module rounds the update to bf16 before the (dropout and the) residual add
        float u0 = __bfloat162float(__float2bfloat16(out[nt * 4 + 0])), u1 = __bfloat162float(__float2bfloat16(out[nt * 4 + 1]));
        float u2 = __bfloat162float(__float2bfloat16(out[nt * 4 + 2])), u3 = __bfloat162float(__float2bfloat16(out[nt * 4 + 3]));
        if (DMASK != nullptr) {
          u0 = __bfloat162float(__float2bfloat16(u0 * dm[nt * 2][0])); u1 = __bfloat162float(__float2bfloat16(u1 * dm[nt * 2][1]));
          u2 = __bfloat162float(__float2bfloat16(u2 * dm[nt * 2 + 1][0])); u3 = __bfloat162float(__float2bfloat16(u3 * dm[nt * 2 + 1][1]));
        }
        *reinterpret_cast<uint32_t*>(so + tma::sw128(r_acc, cc)) = pack2(lo.x + u0, lo.y + u1);
        *reinterpret_cast<uint32_t*>(so + tma::sw128(r_acc + 8, cc)) = pack2(hi.x + u2, hi.y + u3);
      }
    }
    wg::proxy_fence();
    wg_sync();
    if (wtid == 0) {
      for (int si = 0; si < BS; ++si) tma::store_2d(&outmap, tma::sa(sOUT + si * TILE), 0, (s0 + si) * N + i0);
      tma::commit();
    }
  }
  if (wtid == 0) tma::wait_all();
  }
}

PFN_cuTensorMapEncodeTiled_v12000 tma_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    // CUDA 13 has no unversioned PFN name and deprecates cudaGetDriverEntryPoint: ask for the 12.0 ABI by version
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
#endif
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
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
CUtensorMap map_nat(void* base, long row_stride, int N, int S, int bi) {   // natural [S][N][row_stride] bf16, box (32 c, bi tok, 2 s), 64B swizzle
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
void launch_gateout(int N, int S, int ntile, int grid, const __nv_bfloat16* ymap, const __nv_bfloat16* dmask, float dscale, const CUtensorMap& otmap, const CUtensorMap& rmap,
                    const CUtensorMap& gmap, const CUtensorMap& omap_w, const CUtensorMap& outmap) {
  constexpr int BYTES = SMG<NST>::BYTES;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_gateout_kernel<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
    attr = true;
  }
  pwa_gateout_kernel<NST><<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(N, S, ntile, ymap, dmask, dscale, otmap, rmap, gmap, omap_w, outmap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor pwa_gateout(torch::Tensor o_hm, torch::Tensor y, torch::Tensor wgw, torch::Tensor wow, torch::Tensor msa, int64_t blocks_per_sm, int64_t nst,
                          c10::optional<torch::Tensor> dmask, double dscale) {
  TORCH_CHECK(o_hm.is_cuda() && o_hm.scalar_type() == torch::kBFloat16 && o_hm.is_contiguous() && o_hm.dim() == 3 && o_hm.size(0) == H, "o: [H, N, S*C] bf16");
  const int N = (int)o_hm.size(1);
  TORCH_CHECK(o_hm.size(2) % C == 0, "o: [H, N, S*C]");
  const int S = (int)(o_hm.size(2) / C);
  TORCH_CHECK(N % 128 == 0 && S % BS == 0, "N must be a multiple of 128 and S even");
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}), "y: [S, N, D]");
  TORCH_CHECK(msa.is_contiguous() && msa.sizes() == torch::IntArrayRef({S, N, D}), "msa: [S, N, D]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}), "wg: [HC, D]");
  TORCH_CHECK(wow.is_contiguous() && wow.sizes() == torch::IntArrayRef({D, HC}), "wo: [D, HC]");
  auto out = torch::empty({S, N, D}, y.options());
  CUtensorMap otmap = map2d(o_hm.data_ptr(), (long)H * N, (long)S * C, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "o");
  const __nv_bfloat16* ymap = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  CUtensorMap rmap = map2d(msa.data_ptr(), (long)S * N, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "msa");
  CUtensorMap gmap = map2d(wgw.data_ptr(), HC, D, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap omap_w = map2d(wow.data_ptr(), D, HC, 32, 64, CU_TENSOR_MAP_SWIZZLE_64B, "wo");
  CUtensorMap outmap = map2d(out.data_ptr(), (long)S * N, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "out");
  static int sms = 0;
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, y.device().index());
  const int ntile = (S / BS) * (N / 128);
  const int grid = (int)std::min<long>(ntile, (long)sms * blocks_per_sm);
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->sizes() == torch::IntArrayRef({N, D}), "dmask: [N, D] bf16 keep-mask");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  const float dsc = (float)dscale;
  switch (nst) {
    case 3: launch_gateout<3>(N, S, ntile, grid, ymap, dmp, dsc, otmap, rmap, gmap, omap_w, outmap); break;
    case 4: launch_gateout<4>(N, S, ntile, grid, ymap, dmp, dsc, otmap, rmap, gmap, omap_w, outmap); break;
    case 6: launch_gateout<6>(N, S, ntile, grid, ymap, dmp, dsc, otmap, rmap, gmap, omap_w, outmap); break;
    case 8: launch_gateout<8>(N, S, ntile, grid, ymap, dmp, dsc, otmap, rmap, gmap, omap_w, outmap); break;
    default: TORCH_CHECK(false, "nst must be 3, 4, 6 or 8");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("pwa_gateout", &pwa_gateout, "PWA gate / out pass over a head-major o (the cuBLAS contraction)", py::arg("o"), py::arg("y"), py::arg("wg"), py::arg("wo"), py::arg("msa"),
          py::arg("blocks_per_sm") = 1, py::arg("nst") = 4, py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
}
