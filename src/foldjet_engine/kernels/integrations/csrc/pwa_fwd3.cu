// PWA forward, warp-specialized (sm_90a):  out[s,i,:] = msa[s,i,:] + sum_h (sigmoid(y[s,i,:] Wg_h^T) .* o_h[s,i,:]) Wo_h,
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
template <int NST> struct SMX {
  static constexpr int SRING = 0;                                   // [NST][JPS][3][TILE]: per j-chunk W chunk c0, W chunk c1, v chunk
  static constexpr int SWG = SRING + NST * STG;                     // [2 head parity][32*64] Wg_h   (y lives in registers as RS fragments)
  static constexpr int SWO = SWG + 2 * 32 * 64;                     // [2 head parity][64*32] Wo_h (64B swizzle)
  static constexpr int SOUT = SWO + 2 * 64 * 32;                    // [2 consumers][2 s][TILE] output staging (residual in, sum out)
  static constexpr int SO = SOUT + 4 * TILE;                        // [2 consumers][TILE] o staging (natural layout tile, 64B swizzle)
  static constexpr int SEND = SO + 2 * TILE;
  static constexpr int NBAR = 3 * NST + 2 + 2 + 2;                  // full, empty (local v), emptyW (cluster-wide, rank 0), fullW[2], wfree[2], resB[2]
  static constexpr int BYTES = SEND * 2 + NBAR * 8 + 1024;
};

// CS CTAs of a cluster share one i-pair and take CS consecutive s-pairs: the W chunk of a stage is TMA-multicast by rank 0
// into every CTA's slot (one L2 read instead of CS), the v chunk is fetched locally.  Slot release: each consumer warp
// arrives on its local empty (guards the local v load) and remotely on rank 0's emptyW (guards the multicast).
template <int NST, int CS>
__global__ void __launch_bounds__(THREADS, 1) pwa_fwd2_kernel(int N, int S, int ntile, int save_o,
    const __grid_constant__ CUtensorMap wmap,    // w16 [H*N][N]        box (64 j, 64 i)
    const __grid_constant__ CUtensorMap vmap,    // vhm [H*N][S*C]      box (64 (s,c), 64 j)
    const __nv_bfloat16* __restrict__ Yg,        // y   [S*N][D]        (read as RS fragments)
    const __nv_bfloat16* __restrict__ DMASK,     // drop_msa keep-mask [N][D] (0/1, shared over s) or nullptr
    float dscale,                                // 1 / (1 - p)
    const __grid_constant__ CUtensorMap rmap,    // msa [S*N][D]        box (64, 64)
    const __grid_constant__ CUtensorMap gmap,    // wg  [HC][D]         box (64, 32)
    const __grid_constant__ CUtensorMap omap_w,  // wo  [D][HC]         box (32, 64), 64B swizzle
    const __grid_constant__ CUtensorMap outmap,  // out [S*N][D]        box (64, 64)
    const __grid_constant__ CUtensorMap savemap) { // o   natural [S][N][HC] 3-D box (32 c, 64 tok, 2 s), 64B swizzle
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  unsigned char* smb = smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  __nv_bfloat16* sm = reinterpret_cast<__nv_bfloat16*>(smb);
  using L = SMX<NST>;
  __nv_bfloat16* sRing = sm + L::SRING;
  __nv_bfloat16* sWGb = sm + L::SWG;
  __nv_bfloat16* sWOb = sm + L::SWO;
  __nv_bfloat16* sOUTb = sm + L::SOUT;
  __nv_bfloat16* sOb = sm + L::SO;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smb + L::SEND * 2);
  uint64_t* full = bars;                 // [NST]
  uint64_t* empty = full + NST;          // [NST], count 8 (local consumer warps)
  uint64_t* emptyW = empty + NST;        // [NST], count 8 * CS (every consumer warp of the cluster), used on rank 0 only
  uint64_t* fullW = emptyW + NST;        // [2] head parity
  uint64_t* wfree = fullW + 2;           // [2], count 2
  uint64_t* resB = wfree + 2;            // [2] per consumer: residual tiles landed
  const int tid = threadIdx.x, lane = tid & 31;
  const int NI2 = N / 128;                                       // i-pairs
  const int NSPH = N / (64 * JPS);                               // stages per head
  const uint32_t rank = CS > 1 ? tma::cluster_rank() : 0;
  const int ncl = gridDim.x / CS, cid = blockIdx.x / CS;         // cluster count / id: the cluster walks tiles cid, cid + ncl, ...
  // cluster tile ct -> (i-pair, s-pair group); this CTA's s-pair = group * CS + rank
  auto tile_sp = [&](int ct) { return (ct / NI2) * CS + (int)rank; };
  auto tile_ip = [&](int ct) { return ct % NI2; };

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { tma::bar_init(full + i, 1); tma::bar_init(empty + i, 8); tma::bar_init(emptyW + i, 2 * CS); }
    for (int i = 0; i < 2; ++i) { tma::bar_init(fullW + i, 1); tma::bar_init(wfree + i, 2); tma::bar_init(resB + i, 1); }
    tma::bar_init_fence();
    wg::proxy_fence();
  }
  __syncthreads();
  if (CS > 1) tma::cluster_sync();                                // every CTA's barriers are initialised before any remote arrive

  if (tid >= 256) {
    // ===================== producer: one issuing thread per ring SLOT (lane 0 of producer warp p owns slot p) =====================
    // One thread issuing 8 KB TMA ops tops out near 27 GB/s per SM (L2-resident ring test); several issuers scale.  The uses of
    // a slot must be issued in order by ONE thread: a parity wait cannot tell phase k from phase k-2, so two threads waiting on
    // the same empty barrier one use apart would let the later one refill a slot still in use.
    static_assert(NST <= 4, "one producer warp per ring slot");
    wg::reg_dealloc();
    const int pw = (tid - 256) >> 5;
    if (lane == 0 && pw < NST) {
      int gs = 0;                                                    // global stage counter
      for (int T = 0;; ++T) {
        const int t = cid + T * ncl;
        if (t >= ntile) break;
        const int sp = tile_sp(t), ip = tile_ip(t), i0 = ip * 128;
        for (int h = 0; h < H; ++h) {
          const int gh = T * H + h, hb = gh & 1;
          if (pw == 0) {
            if (gh >= 2) tma::wait(wfree + hb, ((gh >> 1) - 1) & 1);
            tma::expect_tx(fullW + hb, (32 * 64 + 64 * 32) * 2);
            tma::load_2d(&gmap, tma::sa(sWGb + hb * 32 * 64), fullW + hb, 0, h * C);
            tma::load_2d(&omap_w, tma::sa(sWOb + hb * 64 * 32), fullW + hb, h * C, 0);
          }
          for (int js = 0; js < NSPH; ++js, ++gs) {                 // one stage = JPS j-chunks
            const int st = gs % NST;
            if (st != pw) continue;
            __nv_bfloat16* stg = sRing + st * STG;
            if (gs >= NST) tma::wait(empty + st, ((gs / NST) - 1) & 1);
            tma::expect_tx(full + st, STG * 2);                      // W (multicast from rank 0) + v (local); tx may go transiently negative
            if (CS > 1 && rank == 0 && gs >= NST) tma::wait_cluster(emptyW + st, ((gs / NST) - 1) & 1);   // every CTA released this slot
            for (int q = 0; q < JPS; ++q) {
              const int jc = js * JPS + q;
              __nv_bfloat16* chunk = stg + q * 3 * TILE;
              if (CS == 1) tma::load_2d(&wmap, tma::sa(chunk), full + st, jc * BJ, h * N + i0);            // one box of 128 i rows: both consumers' W chunks
              else if (rank == 0) tma::load_2d_mc(&wmap, tma::sa(chunk), full + st, jc * BJ, h * N + i0, (uint16_t)((1u << CS) - 1));
              tma::load_2d(&vmap, tma::sa(chunk + 2 * TILE), full + st, sp * NB, h * N + jc * BJ);
            }
          }
        }
      }
    }
  } else {

  // ===================== consumer c =====================
  wg::reg_alloc();
  const int c = tid >> 7, wtid = tid & 127, wl = wtid >> 5;
  auto wg_sync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(1 + c) : "memory"); };
  const int r_acc = wl * 16 + (lane >> 2);
  auto release = [&](int g) {                                    // this warp is done with stage g's slot; the wgmma group is warpgroup-wide,
    if (lane == 0) tma::arrive(empty + (g % NST));               // so one remote arrive per warpgroup covers the multicast slot
    if (CS > 1 && wtid == 0) tma::arrive_remote(emptyW + (g % NST), 0);
  };
  __nv_bfloat16* sOUT = sOUTb + c * 2 * TILE;
  __nv_bfloat16* sO = sOb + c * TILE;
  float oacc[32], gacc0[16], gacc1[16], out0[32], out1[32];
  uint32_t yf[2][16];                                            // y tiles of this tile as RS A-fragments: [s][k-step*4 + {r, r+8, r k+8, r+8 k+8}]
  const int cq = (lane & 3) * 2;
  auto load_y = [&](int T) {                                     // plain global loads straight into the fragment layout
    const int t = cid + T * ncl;
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
    const int t = cid + T * ncl;
    if (t >= ntile) break;
    const int sp = tile_sp(t), ip = tile_ip(t), i0 = ip * 128 + c * 64, s0 = sp * BS;
    // (the residual tiles are fetched into the output staging mid-tile, once the previous tile's store has surely retired)
#pragma unroll
    for (int q = 0; q < 32; ++q) { out0[q] = 0.f; out1[q] = 0.f; }
    for (int h = 0; h < H; ++h) {
      const int gh = T * H + h, hb = gh & 1;
      if (h == H / 2 && wtid == 0) {                               // residual tiles into the output staging: all but the newest bulk group
        tma::wait_read<1>();                                       // (this head's o store) have retired, so the previous tile's out store has
        tma::expect_tx(resB + c, 2 * TILE * 2);
        for (int si = 0; si < BS; ++si) tma::load_2d(&rmap, tma::sa(sOUT + si * TILE), resB + c, 0, (s0 + si) * N + i0);
      }
      // ---- the gate GEMMs of this head (both s): independent of the contraction, issued ahead of it (h > 0) so they overlap;
      //      at h == 0 the y tiles of this tile may still be landing, so they run after chunk 2 instead ----
      tma::wait(fullW + hb, (gh >> 1) & 1);
      const __nv_bfloat16* wgt = sWGb + hb * 32 * 64;
      const __nv_bfloat16* wot = sWOb + hb * 64 * 32;
      // gate GEMMs of this head (both s), A = the y fragments in registers: issued ahead of the contraction, they overlap it
      wg::fence();
#pragma unroll
      for (int si = 0; si < BS; ++si)
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          wg::mma_m64n32k16_rs(yf[si][ks * 4 + 0], yf[si][ks * 4 + 1], yf[si][ks * 4 + 2], yf[si][ks * 4 + 3], wg::desc(wgt + wg::off(0, ks * 2)), si == 0 ? gacc0 : gacc1, ks == 0 ? 0 : 1);
      wg::commit();
      // ---- contraction: NSPH stages of JPS j-chunks ----
      for (int js = 0; js < NSPH; ++js, ++gs) {
        const int st = gs % NST;
        tma::wait(full + st, (gs / NST) & 1);
        wg::fence();
#pragma unroll
        for (int q = 0; q < JPS; ++q) {
          const __nv_bfloat16* a = sRing + st * STG + q * 3 * TILE + c * TILE;
          const __nv_bfloat16* bv = sRing + st * STG + q * 3 * TILE + 2 * TILE;
#pragma unroll
          for (int ks = 0; ks < BJ / 16; ++ks)
            wg::mma_m64n64k16_kmn(wg::desc(a + wg::off(0, ks * 2)), wg::desc_mn(bv + ks * 16 * 64), oacc, (js == 0 && q == 0 && ks == 0) ? 0 : 1);
        }
        wg::commit();
        if (js > 0) { wg::wait<1>(); release(gs - 1); }
        if (js == 0 && h > 0) {                                      // the previous head's out-projection group is retired (older than this stage's
          wg::wait<1>();                                             // group): every warp is past that head's Wo_h / Wg_h tiles
          wg_sync();
          if (wtid == 0) tma::arrive(wfree + ((gh - 1) & 1));
        }
      }
      wg::wait<0>();
      release(gs - 1);
      // ---- o for the backward: natural-layout tile via 3-D TMA from dedicated staging ----
      if (save_o) {
        if (wtid == 0) tma::wait_read<0>();                          // the previous head's o store (and the tile's out store)
        wg_sync();
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int si = nt >> 2, cc = (nt & 3) * 8 + (lane & 3) * 2;
          *reinterpret_cast<uint32_t*>(sO + tma::sw64nat<64>(r_acc, si, cc)) = pack2(oacc[nt * 4 + 0], oacc[nt * 4 + 1]);
          *reinterpret_cast<uint32_t*>(sO + tma::sw64nat<64>(r_acc + 8, si, cc)) = pack2(oacc[nt * 4 + 2], oacc[nt * 4 + 3]);
        }
        wg::proxy_fence();
        wg_sync();
        if (wtid == 0) { tma::store_3d(&savemap, tma::sa(sO), h * C, i0, s0); tma::commit(); }
      }
      // ---- multiply by the gate and out-project, one s at a time (the gate accumulators are complete: wait<0> above) ----
      wg::fence();
#pragma unroll
      for (int si = 0; si < BS; ++si) {
        uint32_t af[8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float* o = oacc + (si * 4 + nt) * 4;
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
      wg::commit();                                                  // NOT waited: it retires under the next head's contraction
      if (h == H - 1) {
        wg::wait<0>();
        wg_sync();                                                   // every warp is past this head's weight tiles
        if (wtid == 0) tma::arrive(wfree + hb);
        load_y(T + 1);                                               // the y fragments are free: prefetch the next tile's into them
      }
    }
    // ---- (drop_msa) residual add and the output store ----
    tma::wait(resB + c, T & 1);
    // drop_msa: one keep-mask per (token, channel) shared over s: u = bf16(bf16(update) * (mask / (1 - p))) -- the stock bf16(x / (1 - p))
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
  if (CS > 1) tma::cluster_sync();                                // no CTA leaves while a peer may still arrive on its barriers
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

template <int NST, int CS>
void launch_fwd2(int N, int S, int ntile, int save_o, int grid, const __nv_bfloat16* dmask, float dscale, const CUtensorMap& wmap, const CUtensorMap& vmap, const __nv_bfloat16* ymap, const CUtensorMap& rmap,
                 const CUtensorMap& gmap, const CUtensorMap& omap_w, const CUtensorMap& outmap, const CUtensorMap& savemap) {
  constexpr int BYTES = SMX<NST>::BYTES;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_fwd2_kernel<NST, CS>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
    int nb = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pwa_fwd2_kernel<NST, CS>, THREADS, BYTES));
    TORCH_WARN("pwa_fwd3_kernel<", NST, ",", CS, ">: ", BYTES, " B smem -> ", nb, " blocks/SM");
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = BYTES; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, pwa_fwd2_kernel<NST, CS>, N, S, ntile, save_o, wmap, vmap, ymap, dmask, dscale, rmap, gmap, omap_w, outmap, savemap));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> pwa_fwd3(torch::Tensor w16, torch::Tensor vhm, torch::Tensor y, torch::Tensor wgw, torch::Tensor wow, torch::Tensor msa, bool save_o, int64_t blocks_per_sm, int64_t nst, int64_t cs, int64_t max_clusters,
                                    c10::optional<torch::Tensor> dmask, double dscale) {
  TORCH_CHECK(w16.is_cuda() && w16.scalar_type() == torch::kBFloat16 && w16.is_contiguous() && w16.dim() == 3 && w16.size(0) == H, "w16: [H, N, N] bf16");
  const int N = (int)w16.size(1);
  TORCH_CHECK(vhm.is_contiguous() && vhm.dim() == 3 && vhm.size(0) == H && vhm.size(1) == N && vhm.size(2) % C == 0, "vhm: [H, N, S*C]");
  const int S = (int)(vhm.size(2) / C);
  TORCH_CHECK(N % 128 == 0 && S % BS == 0, "N must be a multiple of 128 (i-pairs, two j-chunks per stage) and S even");
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}), "y: [S, N, D]");
  TORCH_CHECK(msa.is_contiguous() && msa.sizes() == torch::IntArrayRef({S, N, D}), "msa: [S, N, D]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}), "wg: [HC, D]");
  TORCH_CHECK(wow.is_contiguous() && wow.sizes() == torch::IntArrayRef({D, HC}), "wo: [D, HC]");
  auto out = torch::empty({S, N, D}, y.options());
  auto o = save_o ? torch::empty({S, N, HC}, y.options()) : torch::empty({0}, y.options());
  CUtensorMap wmap = map2d(w16.data_ptr(), (long)H * N, N, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "w");
  CUtensorMap vmap = map2d(vhm.data_ptr(), (long)H * N, (long)S * C, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "v");
  const __nv_bfloat16* ymap = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  CUtensorMap rmap = map2d(msa.data_ptr(), (long)S * N, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "msa");
  CUtensorMap gmap = map2d(wgw.data_ptr(), HC, D, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap omap_w = map2d(wow.data_ptr(), D, HC, 32, 64, CU_TENSOR_MAP_SWIZZLE_64B, "wo");
  CUtensorMap outmap = map2d(out.data_ptr(), (long)S * N, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "out");
  CUtensorMap savemap = save_o ? map_nat(o.data_ptr(), HC, N, S, 64) : outmap;
  static int sms = 0;
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, y.device().index());
  TORCH_CHECK((S / BS) % cs == 0 && (cs == 1 || cs == 2 || cs == 4), "cluster size must be 1, 2 or 4 and divide S/2");
  const int ntile = (S / BS / (int)cs) * (N / 128);                 // cluster tiles
  int nclus = (int)std::min<long>(ntile, (long)(sms / cs) * blocks_per_sm);
  if (max_clusters > 0 && max_clusters < nclus) nclus = (int)max_clusters;
  const int grid = nclus * (int)cs;
  const int so = save_o ? 1 : 0;
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->sizes() == torch::IntArrayRef({N, D}), "dmask: [N, D] bf16 keep-mask");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  const float dsc = (float)dscale;
  switch (nst * 10 + cs) {
    case 31: launch_fwd2<3, 1>(N, S, ntile, so, grid, dmp, dsc, wmap, vmap, ymap, rmap, gmap, omap_w, outmap, savemap); break;
    case 32: launch_fwd2<3, 2>(N, S, ntile, so, grid, dmp, dsc, wmap, vmap, ymap, rmap, gmap, omap_w, outmap, savemap); break;
    case 21: launch_fwd2<2, 1>(N, S, ntile, so, grid, dmp, dsc, wmap, vmap, ymap, rmap, gmap, omap_w, outmap, savemap); break;
    default: TORCH_CHECK(false, "(nst, cs) must be one of (3,1) (3,2) (2,1)");
  }
  return {out, o};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("pwa_fwd3", &pwa_fwd3, "PWA forward, warp-specialized TMA version with cluster multicast of W", py::arg("w16"), py::arg("vhm"), py::arg("y"), py::arg("wg"), py::arg("wo"), py::arg("msa"), py::arg("save_o") = false, py::arg("blocks_per_sm") = 1, py::arg("nst") = 3, py::arg("cs") = 1, py::arg("max_clusters") = 0, py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
}
