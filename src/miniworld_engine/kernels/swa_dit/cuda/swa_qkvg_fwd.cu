#ifndef MW_SWA_QKVG_SP
#define MW_SWA_QKVG_SP 8
#endif
#ifndef MW_SWA_FFN_SP
#define MW_SWA_FFN_SP 8
#endif
#ifndef MW_SWA_FWD_STAGES
#define MW_SWA_FWD_STAGES 3
#endif
// ESMFold2 SWA atom block, first stage forward (CUDA sm_90a, C = 128, H = 4, D = 32, bf16 operands, fp32 accumulate):
//   x = rn(RMS(q) * (1 + scale_a) + shift_a);  p_{q,k,v,g} = x W_{q,k,v,g}^T;
//   Q = rn(rope(rn(headRMS(rn(p_q)))))  (same for K);  V = rn(p_v);  G = rn(p_g)       (rounding points of the Triton _qkvg_fwd)
//   Q / K / V are written head-major [N, H, S, D]; G row-major [M, C].  Optionally saves x and the pre-norm rn(p_q), rn(p_k).
// Structure as swa_ffn_fwd.cu: two consumer warpgroups ping-pong on alternate tiles of SP=8 augments x AT=8 atoms, x is built in
// registers as the bf16 A operand; per projection (m64n128, 4 per tile) the epilogue works in the fragment layout (a head = 4 n8
// blocks of one thread quad; RoPE pairs d, d+16 = blocks i, i+2 of the same thread) and leaves through smem staging + TMA stores.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdlib>

namespace {
constexpr int C = 128, H = 4, D = 32, BR = 64, SP = MW_SWA_QKVG_SP, AT = BR / SP;
constexpr int NST = MW_SWA_FWD_STAGES, SLOT = 32768, SPT = 4;             // weight ring: Wq | Wk | Wv | Wg, [128 out][128 in] each
constexpr int THREADS = 384;
constexpr int TB = BR * 128;                              // bf16 [64 rows][64] k-block (8 KB)
constexpr int SQ = 0, SMOD = 2 * TB, SSTG = SMOD + 8 * AT * 128, SET = SSTG + 2 * 2 * TB;   // q|x 16K, mod 8K, 2 x (out 16K) staging
constexpr int SW = 2 * SET, SEND = SW + NST * SLOT;
constexpr int NBAR = 4 + 2 * NST;
constexpr int BYTES = SEND + NBAR * 8;
#ifndef CONS_REGS
#define CONS_REGS "232"
#define PROD_REGS "40"
#endif

namespace wg {
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(64) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
#include "gen/wgmma_bf16.inc"
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
__device__ __forceinline__ void load_4d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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
}  // namespace tma

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float2 unpack(uint32_t u) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&u);
  return make_float2(__low2float(v), __high2float(v));
}
// bf16 [64][128] tile as 2 k-blocks of [64 rows][128 B], 128 B swizzle
__device__ __forceinline__ int offb(int r, int c) { return (c >> 6) * TB + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2; }
// head-major staging: head h = [64 rows][32] bf16 (64 B rows, 64 B swizzle) at h * 4 KB
__device__ __forceinline__ int offh(int r, int c) { return (c >> 5) * 4096 + r * 64 + (((((c & 31) >> 3) ^ (r >> 1)) & 3) << 4) + (c & 7) * 2; }
// fp32 modulation tile [8 col-blocks (shift_a, scale_a) x 4][AT atoms][32], 128 B swizzle
__device__ __forceinline__ const float2* modp(const unsigned char* sm, int k, int al, int c) {
  const int row = (k * 4 + (c >> 5)) * AT + al;
  return reinterpret_cast<const float2*>(sm + row * 128 + ((((c & 31) >> 2) ^ (row & 7)) << 4) + (c & 3) * 4);
}

struct Args {
  const float *cs, *sn;                                   // [B*S, D/2]
  int S, A, B, nab, nag, ntile;
  float eps, qk_eps;
  int save;
};

__global__ void __launch_bounds__(THREADS, 1) qkvg_fwd_kernel(Args args, const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mmod,
                                                             const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mQ,
                                                             const __grid_constant__ CUtensorMap mK, const __grid_constant__ CUtensorMap mV,
                                                             const __grid_constant__ CUtensorMap mG, const __grid_constant__ CUtensorMap mX,
                                                             const __grid_constant__ CUtensorMap mPQ, const __grid_constant__ CUtensorMap mPK) {
  extern __shared__ __align__(1024) unsigned char smem[];
  if (tma::sa(smem) & 1023u) __trap();
  unsigned char* sW = smem + SW;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + SEND);
  uint64_t *xfull = bars, *xempty = bars + 2, *wfull = bars + 4, *wempty = bars + 4 + NST;
  const int tid = threadIdx.x, lane = tid & 31, wgi = tid >> 7, warp = (tid >> 5) & 3, tq = tid & 127;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { tma::bar_init(xfull + i, 1); tma::bar_init(xempty + i, 1); }
    for (int i = 0; i < NST; ++i) { tma::bar_init(wfull + i, 1); tma::bar_init(wempty + i, 8); }
    tma::bar_init_fence(); wg::proxy_fence();
  }
  __syncthreads();
  const int ntT = (int)blockIdx.x < args.ntile ? (args.ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int S = args.S;
  auto coords = [&](int t, int& b, int& a0, int& s0) {
    const int ab = t % args.nab, r = t / args.nab, ag = r % args.nag;
    b = r / args.nag; a0 = ag * SP; s0 = ab * AT;
  };
  if (wgi == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 " PROD_REGS ";\n" ::: "memory");
    if (warp == 0 && lane == 0) {
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
        const int st = T & 1;
        if (T >= 2) tma::wait(xempty + st, ((T >> 1) - 1) & 1);
        unsigned char* sx = smem + st * SET;
        tma::expect_tx(xfull + st, 2 * TB + 8 * AT * 128);
        for (int kb = 0; kb < 2; ++kb) tma::load_4d(&mq, tma::sa(sx + SQ + kb * TB), xfull + st, kb * 64, s0, b, a0);
        tma::load_3d(&mmod, tma::sa(sx + SMOD), xfull + st, 0, b * S + s0, 0);    // col-blocks 0..7 = shift_a, scale_a
      }
    } else if (warp == 1 && lane == 0) {
      const int npair = (ntT + 1) / 2;
      int g = 0;
      for (int P = 0; P < npair; ++P)
        for (int s = 0; s < SPT; ++s, ++g) {
          const int st = g % NST;
          if (g >= NST) tma::wait(wempty + st, ((g / NST) - 1) & 1);
          unsigned char* d = sW + st * SLOT;
          tma::expect_tx(wfull + st, 32768);
          tma::load_2d(&mw, tma::sa(d), wfull + st, 0, s * 128);
          tma::load_2d(&mw, tma::sa(d + 16384), wfull + st, 64, s * 128);
        }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 " CONS_REGS ";\n" ::: "memory");
  const int cs = wgi;
  unsigned char* sx = smem + cs * SET;
  unsigned char *sQ = sx + SQ, *sM = sx + SMOD, *sS = sx + SSTG;
  const int r0 = warp * 16 + (lane >> 2), cq = (lane & 3) * 2, al = r0 % AT;
  const int bar_id = 1 + cs;
  const bool save = args.save != 0;
  auto bsync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory"); };
  float acc[64];
  uint32_t fa[32];
  int P = 0;
  for (int T = cs; T < ntT; T += 2, ++P) {
    int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
    const int g0 = P * SPT;
    const int mrow = b * S + (s0 + al < S ? s0 + al : S - 1);
    // RoPE tables of this thread's atom: pair indices d = 8i' + cq, +1 (i' = 0, 1)
    float2 cv[2][2], sv[2][2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ar = (r0 + 8 * rr) % AT;
      const int mr = b * S + (s0 + ar < S ? s0 + ar : S - 1);
      cv[rr][i] = __ldg(reinterpret_cast<const float2*>(args.cs + (long)mr * (D / 2) + 8 * i + cq));
      sv[rr][i] = __ldg(reinterpret_cast<const float2*>(args.sn + (long)mr * (D / 2) + 8 * i + cq));
    }
    tma::wait(xfull + cs, (T >> 1) & 1);
    // ---- x = rn(q * rstd * (1 + scale) + shift) -> A fragments (and saved into Q in place) ----
    float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const float2 v = unpack(*reinterpret_cast<const uint32_t*>(sQ + offb(r0 + 8 * rr, i * 8 + cq)));
        acc[4 * i + 2 * rr] = v.x; acc[4 * i + 2 * rr + 1] = v.y;
        if (rr == 0) ss0 += v.x * v.x + v.y * v.y; else ss1 += v.x * v.x + v.y * v.y;
      }
    ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1); ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
    ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1); ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
    const float rs0 = rsqrtf(ss0 * (1.f / C) + args.eps), rs1 = rsqrtf(ss1 * (1.f / C) + args.eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = i * 8 + cq;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int ar = (r0 + 8 * rr) % AT;
        const float2 sh = *modp(sM, 0, ar, c), sc = *modp(sM, 1, ar, c);
        const float rs = rr ? rs1 : rs0;
        const uint32_t xv = pack(acc[4 * i + 2 * rr] * rs * (1.f + sc.x) + sh.x, acc[4 * i + 2 * rr + 1] * rs * (1.f + sc.y) + sh.y);
        fa[4 * (i >> 1) + 2 * (i & 1) + rr] = xv;
        if (save) *reinterpret_cast<uint32_t*>(sQ + offb(r0 + 8 * rr, c)) = xv;
      }
    }
    if (save) {
      wg::proxy_fence(); bsync();
      if (tq == 0) { for (int kb = 0; kb < 2; ++kb) tma::store_4d(&mX, sQ + kb * TB, kb * 64, s0, b, a0); tma::store_commit(); }
    }
    // ---- projections: chunk p = q, k, v, g ----
#pragma unroll 1
    for (int p = 0; p < 4; ++p) {
      const int gs = g0 + p, st = gs % NST;
      tma::wait(wfull + st, (gs / NST) & 1);
      const unsigned char* bw = sW + st * SLOT;
      wg::fence();
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const uint64_t d = wg::desc(bw + (s >> 2) * 16384 + (s & 3) * 32);
        if (s == 0) wg::mma_rs_n128_first(fa[0], fa[1], fa[2], fa[3], d, acc);
        else wg::mma_rs_n128(fa[4 * s], fa[4 * s + 1], fa[4 * s + 2], fa[4 * s + 3], d, acc, 1);
      }
      wg::commit(); wg::wait<0>();
      if (lane == 0) tma::arrive(wempty + st);
      unsigned char* stg = sS + (p & 1) * 2 * TB;          // this chunk's staging slot (16 KB)
      if (tq == 0) tma::store_wait_read<1>();              // the store that last read this slot (two chunks ago) is done
      bsync();
#pragma unroll
      for (int e = 0; e < 64; ++e) acc[e] = rnb(acc[e]);
      if (p == 3) {                                        // G: row-major
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr)
            *reinterpret_cast<uint32_t*>(stg + offb(r0 + 8 * rr, i * 8 + cq)) = pack(acc[4 * i + 2 * rr], acc[4 * i + 2 * rr + 1]);
      } else if (p == 2) {                                 // V: head-major
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr)
            *reinterpret_cast<uint32_t*>(stg + offh(r0 + 8 * rr, i * 8 + cq)) = pack(acc[4 * i + 2 * rr], acc[4 * i + 2 * rr + 1]);
      } else {                                             // Q / K: (save pre-norm) -> head RMSNorm -> rn -> RoPE -> rn, head-major
        if (save) {                                        // pre-norm rn(p) -> the x region (its store has been waited for below)
          if (tq == 0) tma::store_wait_read<0>();
          bsync();
#pragma unroll
          for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
              *reinterpret_cast<uint32_t*>(sQ + offb(r0 + 8 * rr, i * 8 + cq)) = pack(acc[4 * i + 2 * rr], acc[4 * i + 2 * rr + 1]);
        }
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            float ss = 0.f;
#pragma unroll
            for (int ib = 0; ib < 4; ++ib) { const float u = acc[4 * (4 * h + ib) + 2 * rr], w = acc[4 * (4 * h + ib) + 2 * rr + 1]; ss += u * u + w * w; }
            ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            const float rq = rsqrtf(ss * (1.f / D) + args.qk_eps);
#pragma unroll
            for (int ib = 0; ib < 2; ++ib)                // first-half block ib pairs with block ib + 2
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int i1 = 4 * (4 * h + ib) + 2 * rr + e, i2 = 4 * (4 * h + ib + 2) + 2 * rr + e;
                const float x1 = rnb(acc[i1] * rq), x2 = rnb(acc[i2] * rq);
                const float c_ = e ? cv[rr][ib].y : cv[rr][ib].x, s_ = e ? sv[rr][ib].y : sv[rr][ib].x;
                acc[i1] = rnb(x1 * c_ - x2 * s_); acc[i2] = rnb(x2 * c_ + x1 * s_);
              }
          }
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr)
            *reinterpret_cast<uint32_t*>(stg + offh(r0 + 8 * rr, i * 8 + cq)) = pack(acc[4 * i + 2 * rr], acc[4 * i + 2 * rr + 1]);
      }
      wg::proxy_fence();
      bsync();
      if (tq == 0) {
        if (p == 3) { for (int kb = 0; kb < 2; ++kb) tma::store_4d(&mG, stg + kb * TB, kb * 64, s0, b, a0); }
        else {
          const CUtensorMap* m = p == 0 ? &mQ : p == 1 ? &mK : &mV;
          for (int h = 0; h < H; ++h) tma::store_5d(m, stg + h * 4096, 0, s0, h, b, a0);
        }
        if (save && p < 2) { const CUtensorMap* m = p == 0 ? &mPQ : &mPK; for (int kb = 0; kb < 2; ++kb) tma::store_4d(m, sQ + kb * TB, kb * 64, s0, b, a0); }
        tma::store_commit();
      }
    }
    if (tq == 0) { tma::store_wait_read<0>(); tma::arrive(xempty + cs); }
  }
  const int npair = (ntT + 1) / 2;
  for (; P < npair; ++P)
    if (lane == 0) for (int s = 0; s < SPT; ++s) { const int g = P * SPT + s; tma::wait(wfull + g % NST, (g / NST) & 1); tma::arrive(wempty + g % NST); }
}

PFN_cuTensorMapEncodeTiled_v12000 tma_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr; cudaDriverEntryPointQueryResult qr;
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
CUtensorMap encode(CUtensorMapDataType dt, int rank, void* ptr, const uint64_t* gdim, const uint64_t* gstride, const uint32_t* bdim, CUtensorMapSwizzle sw) {
  alignas(64) CUtensorMap m{};
  uint32_t estride[5] = {1, 1, 1, 1, 1};
  CUresult r = tma_encode()(&m, dt, rank, ptr, gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
CUtensorMap map4d(const torch::Tensor& t, long A, long B, long S) {   // [A][B][S][128] bf16 -> box (64 ch, AT atoms, 1, SP augments)
  uint64_t gdim[4] = {(uint64_t)C, (uint64_t)S, (uint64_t)B, (uint64_t)A};
  uint64_t gstride[3] = {(uint64_t)C * 2, (uint64_t)S * C * 2, (uint64_t)B * S * C * 2};
  uint32_t bdim[4] = {64, AT, 1, SP};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_128B);
}
CUtensorMap maphm(const torch::Tensor& t, long A, long B, long S) {   // [A][B][H][S][32] bf16 -> box (32, AT atoms, 1 head, 1, SP augments)
  uint64_t gdim[5] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B, (uint64_t)A};
  uint64_t gstride[4] = {(uint64_t)D * 2, (uint64_t)S * D * 2, (uint64_t)H * S * D * 2, (uint64_t)B * H * S * D * 2};
  uint32_t bdim[5] = {D, AT, 1, 1, SP};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_64B);
}
CUtensorMap mapmod(const torch::Tensor& t, long rows) {   // [rows][24 blocks][32] fp32 viewed as (32 ch, rows, 24 blocks) -> box (32, AT, 8)
  uint64_t gdim[3] = {32, (uint64_t)rows, 24}; uint64_t gstride[2] = {6 * C * 4, 128};
  uint32_t bdim[3] = {32, AT, 8};
  return encode(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_128B);
}
CUtensorMap map2d(const torch::Tensor& t, uint64_t rows, uint64_t cols, uint32_t brows) {
  uint64_t gdim[2] = {cols, rows}; uint64_t gstride[1] = {cols * 2};
  uint32_t bdim[2] = {64, brows};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_128B);
}
int num_sms(int dev) { static int s = 0; if (s == 0) cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }
}  // namespace

// q [M, 128] bf16 (rows ((a*B + b)*S + s)); mod [B*S, 768] fp32; cos / sin [B*S, 16] fp32; w [512, 128] = [Wq; Wk; Wv; Wg] bf16.
// Returns [Q, K, V (head-major [N, H, S, D]), G [M, 128], x, pq, pk ([M, 128], empty unless save)].
std::vector<torch::Tensor> qkvg_fwd(torch::Tensor q, torch::Tensor mod, torch::Tensor cs, torch::Tensor sn, torch::Tensor w, int64_t A, int64_t B,
                                    int64_t S, double eps, double qk_eps, bool save) {
  const long M = A * B * S, Nn = A * B;
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.is_contiguous() && q.numel() == M * C, "q: [M, 128] bf16");
  TORCH_CHECK(mod.scalar_type() == torch::kFloat && mod.is_contiguous() && mod.numel() == B * S * 6 * C, "mod [B*S, 768] fp32");
  for (const auto* t : {&cs, &sn}) TORCH_CHECK(t->scalar_type() == torch::kFloat && t->is_contiguous() && t->numel() >= B * S * (D / 2), "cos / sin [B*S, 16] fp32");
  TORCH_CHECK(w.is_contiguous() && w.scalar_type() == torch::kBFloat16 && w.sizes() == torch::IntArrayRef({4 * C, C}), "w [512, 128] bf16");
  auto opt = q.options();
  auto Q = torch::empty({Nn, H, S, D}, opt), K = torch::empty_like(Q), V = torch::empty_like(Q), G = torch::empty({M, C}, opt);
  const long Ms = save ? M : 1;
  auto X = torch::empty({Ms, C}, opt), PQ = torch::empty({Ms, C}, opt), PK = torch::empty({Ms, C}, opt);
  CUtensorMap mq = map4d(q, A, B, S), mG = map4d(G, A, B, S);
  CUtensorMap mQ = maphm(Q, A, B, S), mK = maphm(K, A, B, S), mV = maphm(V, A, B, S);
  CUtensorMap mX = save ? map4d(X, A, B, S) : mG, mPQ = save ? map4d(PQ, A, B, S) : mG, mPK = save ? map4d(PK, A, B, S) : mG;
  CUtensorMap mmod = mapmod(mod, B * S), mw = map2d(w, 4 * C, C, 128);
  Args args; args.cs = cs.data_ptr<float>(); args.sn = sn.data_ptr<float>();
  args.S = (int)S; args.A = (int)A; args.B = (int)B; args.eps = (float)eps; args.qk_eps = (float)qk_eps; args.save = save ? 1 : 0;
  args.nab = (int)((S + AT - 1) / AT); args.nag = (int)((A + SP - 1) / SP); args.ntile = args.nab * args.nag * (int)B;
  cudaKernel_t kernel; C10_CUDA_CHECK(cudaGetKernel(&kernel,qkvg_fwd_kernel));
  C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,BYTES));
  const int grid = std::min(args.ntile,num_sms(q.device().index()));
  void* launchargs[]={&args,&mq,&mmod,&mw,&mQ,&mK,&mV,&mG,&mX,&mPQ,&mPK};
  C10_CUDA_CHECK(cudaLaunchKernel((const void*)kernel,dim3(grid),dim3(THREADS),launchargs,BYTES,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {Q, K, V, G, X, PQ, PK};
}

// Deterministic native auxiliary stages.  Modulation reductions own one
// (batch, atom) row and visit augments in fixed order; no floating atomics.
namespace swa_native {
struct Rows { const void* x[12]; void* y[8]; bool xb[12],yb[8],bf; int A,B,S,stage; float eps; };
__device__ float get(const Rows& p,int k,long i) { return p.xb[k]?__bfloat162float(((const __nv_bfloat16*)p.x[k])[i]):((const float*)p.x[k])[i]; }
__device__ void put(const Rows& p,int k,long i,float v) { if(p.yb[k]) ((__nv_bfloat16*)p.y[k])[i]=__float2bfloat16_rn(v); else ((float*)p.y[k])[i]=v; }
__device__ float rnd(float v,bool bf) { return bf?__bfloat162float(__float2bfloat16_rn(v)):v; }
__device__ float sum(float v) { for(int d=16;d;d>>=1)v+=__shfl_xor_sync(0xffffffff,v,d); return v; }
__device__ float sig(float v) { return 1.f/(1.f+expf(-v)); }
template<typename K,typename...T> void launch(K fn,int grid,T...args) {
 cudaKernel_t k; C10_CUDA_CHECK(cudaGetKernel(&k,fn)); void* a[]={static_cast<void*>(&args)...};
 C10_CUDA_CHECK(cudaLaunchKernel((const void*)k,dim3(grid),dim3(128),a,0,at::cuda::getCurrentCUDAStream())); C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void rows(Rows p) {
 long r=(long)blockIdx.x*4+threadIdx.x/32; int lane=threadIdx.x%32; long M=(long)p.A*p.B*p.S;
 bool reduce=p.stage==8||p.stage==9||p.stage==12||p.stage==6;
 if(r>=(reduce?(long)p.B*p.S:M))return;
 long mr=r%(p.B*p.S);
 if(p.stage==0) {
  float z[4],ss=0; for(int h=0;h<4;++h){z[h]=get(p,0,r*128+h*32+lane);ss+=z[h]*z[h];}float rs=rsqrtf(sum(ss)/128+p.eps);
  for(int h=0;h<4;++h){int c=h*32+lane;put(p,0,r*128+c,z[h]*rs*(1+get(p,1,mr*768+128+c))+get(p,1,mr*768+c));}
 } else if(p.stage==1) {
  for(int h=0;h<4;++h){int c=h*32+lane;long hm=((r/p.S*4+h)*p.S+r%p.S)*32+lane;
   float cs=get(p,1,mr*16+lane%16),sn=get(p,2,mr*16+lane%16);
   for(int k=0;k<2;++k){float z=get(p,0,r*512+k*128+c);put(p,4+k,r*128+c,z);float v=rnd(z*rsqrtf(sum(z*z)/32+p.eps),p.bf);float pair=__shfl_xor_sync(0xffffffff,v,16);put(p,k,hm,v*cs+(lane<16?-pair:pair)*sn);}
   put(p,2,hm,get(p,0,r*512+256+c));put(p,3,r*128+c,get(p,0,r*512+384+c));
  }
 } else if(p.stage==2) {
  for(int h=0;h<4;++h){long i=r*128+h*32+lane;put(p,0,i,get(p,0,i)*sig(get(p,1,i)));}
 } else if(p.stage==3) {
  float z[4],ss=0;for(int h=0;h<4;++h){int c=h*32+lane;long i=r*128+c;z[h]=rnd(get(p,0,i)+rnd(rnd(get(p,2,mr*768+256+c),p.bf)*get(p,1,i),p.bf),p.bf);ss+=z[h]*z[h];put(p,0,i,z[h]);}
  float rs=rsqrtf(sum(ss)/128+p.eps);for(int h=0;h<4;++h){int c=h*32+lane;put(p,1,r*128+c,z[h]*rs*(1+get(p,2,mr*768+512+c))+get(p,2,mr*768+384+c));}
 } else if(p.stage==4||p.stage==7) {
  for(int h=0;h<8;++h){int c=h*32+lane;float a=get(p,0,r*512+c),b=get(p,0,r*512+256+c),s=sig(a);if(p.stage==4)put(p,0,r*256+c,a*s*b);else{float dh=get(p,1,r*256+c);put(p,0,r*512+c,dh*b*s*(1+a*(1-s)));put(p,0,r*512+256+c,dh*a*s);put(p,1,r*256+c,a*s*b);}}
 } else if(p.stage==5) {
  for(int h=0;h<4;++h){int c=h*32+lane;long i=r*128+c;put(p,0,i,get(p,0,i)+rnd(rnd(get(p,2,mr*768+640+c),p.bf)*get(p,1,i),p.bf));}
 } else if(p.stage==6||p.stage==9) {
  int col=p.stage==6?640:256;float dm[4]={};
  for(int a=0;a<p.A;++a){long rr=r+(long)a*p.B*p.S;for(int h=0;h<4;++h){int c=h*32+lane;long i=rr*128+c;float d=get(p,0,i);dm[h]+=d*get(p,2,i);put(p,0,i,d*rnd(get(p,1,mr*768+col+c),p.bf));if(p.stage==9)put(p,1,i,get(p,3,i)*sig(get(p,4,i)));}}
  for(int h=0;h<4;++h)put(p,p.stage==6?1:2,mr*768+col+h*32+lane,dm[h]);
 } else if(p.stage==8||p.stage==12) {
  int col=p.stage==8?384:0;float ds[4]={},dt[4]={};
  for(int a=0;a<p.A;++a){long rr=r+(long)a*p.B*p.S;float q[4],g[4],ss=0,dot=0;for(int h=0;h<4;++h){int c=h*32+lane;q[h]=get(p,2,rr*128+c);ss+=q[h]*q[h];}
   float rs=rsqrtf(sum(ss)/128+p.eps);for(int h=0;h<4;++h){int c=h*32+lane;float d=get(p,1,rr*128+c);q[h]*=rs;ds[h]+=d;dt[h]+=d*q[h];g[h]=d*(1+get(p,3,mr*768+col+128+c));dot+=g[h]*q[h];}dot=sum(dot)/128;
   for(int h=0;h<4;++h){int c=h*32+lane;put(p,0,rr*128+c,get(p,0,rr*128+c)+rs*(g[h]-q[h]*dot));}}
  for(int h=0;h<4;++h){int c=h*32+lane;put(p,1,mr*768+col+c,ds[h]);put(p,1,mr*768+col+128+c,dt[h]);}
 } else if(p.stage==10) {
  for(int h=0;h<4;++h){long i=r*128+h*32+lane;float d=get(p,0,i),o=get(p,1,i),s=sig(get(p,2,i));float dO=rnd(d*s,true);put(p,0,i,dO);put(p,1,i,d*o*s*(1-s));float delta=sum(dO*o);if(lane==0)put(p,2,(r/p.S*4+h)*p.S+r%p.S,delta);}
 } else if(p.stage==11) {
  for(int h=0;h<4;++h){int c=h*32+lane;long hm=((r/p.S*4+h)*p.S+r%p.S)*32+lane;float cs=get(p,6,mr*16+lane%16),sn=get(p,7,mr*16+lane%16);
   for(int k=0;k<2;++k){float g=get(p,2+k,hm),pair=__shfl_xor_sync(0xffffffff,g,16);g=rnd(g*cs+(lane<16?pair:-pair)*sn,p.bf);float z=get(p,k,r*128+c),rs=rsqrtf(sum(z*z)/32+p.eps),xh=z*rs;float dz=rs*(g-xh*sum(g*xh)/32);put(p,0,r*512+k*128+c,dz);}
   put(p,0,r*512+256+c,get(p,4,hm));put(p,0,r*512+384+c,get(p,5,r*128+c));}
 }
}
struct Att {const __nv_bfloat16 *q,*k,*v,*d;const float *lse,*delta;const int* used;__nv_bfloat16 *o,*dq,*dk,*dv;float* olse;int N,S,stage;};
__global__ void attention(Att p) {
 int id=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(id>=p.N*4*p.S)return;int i=id%p.S,nh=id/p.S,n=nh/4,h=nh%4,su=p.used[n];long hb=(long)nh*p.S*32,oi=((long)n*p.S+i)*128+h*32+lane;
 const float scale=.1767766952966369f;float acc=0,av=0;
 if(p.stage==0){float q=i<su?__bfloat162float(p.q[hb+i*32+lane]):0,maxv=-INFINITY,den=0;
  for(int j=max(0,i-64);j<min(su,i+65)&&i<su;++j){float sc=sum(q*__bfloat162float(p.k[hb+j*32+lane]))*scale;maxv=fmaxf(maxv,sc);}
  for(int j=max(0,i-64);j<min(su,i+65)&&i<su;++j){float pr=expf(sum(q*__bfloat162float(p.k[hb+j*32+lane]))*scale-maxv);den+=pr;acc+=rnd(pr,true)*__bfloat162float(p.v[hb+j*32+lane]);}
  p.o[oi]=__float2bfloat16_rn(den>0?acc/den:0);if(lane==0)p.olse[id]=den>0?maxv+logf(den):0;
 }else if(p.stage==1){float q=i<su?__bfloat162float(p.q[hb+i*32+lane]):0,d=i<su?__bfloat162float(p.d[oi]):0;
  for(int j=max(0,i-64);j<min(su,i+65)&&i<su;++j){float k=__bfloat162float(p.k[hb+j*32+lane]),v=__bfloat162float(p.v[hb+j*32+lane]);float pr=expf(sum(q*k)*scale-p.lse[id]);float ds=rnd(pr*(sum(d*v)-p.delta[id]),true);acc+=ds*k;}p.dq[hb+i*32+lane]=__float2bfloat16_rn(acc*scale);
 }else{float k=i<su?__bfloat162float(p.k[hb+i*32+lane]):0,v=i<su?__bfloat162float(p.v[hb+i*32+lane]):0;
  for(int j=max(0,i-64);j<min(su,i+65)&&i<su;++j){float q=__bfloat162float(p.q[hb+j*32+lane]),d=__bfloat162float(p.d[((long)n*p.S+j)*128+h*32+lane]);float pr=expf(sum(q*k)*scale-p.lse[nh*p.S+j]);float ds=rnd(pr*(sum(d*v)-p.delta[nh*p.S+j]),true);acc+=ds*q;av+=rnd(pr,true)*d;}p.dk[hb+i*32+lane]=__float2bfloat16_rn(acc*scale);p.dv[hb+i*32+lane]=__float2bfloat16_rn(av);}
}
struct Copy {const void* x[8];void* y[8];bool xb[8],yb[8];long sizes[8];int count,stage;};
__global__ void copies(Copy p){long j=(long)blockIdx.x*128+threadIdx.x;for(int k=0;k<p.count;++k){if(j>=p.sizes[k])continue;float z=p.xb[k]?__bfloat162float(((const __nv_bfloat16*)p.x[k])[j]):((const float*)p.x[k])[j];
 if(p.stage==1)z=rnd(z*sig(z),p.xb[k]);long dest=j;if(p.stage==2){long row=j/128,c=j%128;dest=((row%256)/64*128+(row/256)*64+row%64)*128+c;}
 if(p.stage==3){if(k==1)dest=(j%256)*128+j/256;else{long row=j/128,c=j%128,pr=(row%256)/32*64+(row/256)*32+row%32;dest=k==0?pr*128+c:c*512+pr;}}
 if(p.yb[k])((__nv_bfloat16*)p.y[k])[dest]=__float2bfloat16_rn(z);else((float*)p.y[k])[dest]=z;}}
__global__ void mod_bwd(Rows p,int count){int i=blockIdx.x*128+threadIdx.x;if(i>=count)return;float c=get(p,0,i),s=sig(c),d=rnd(get(p,1,i),p.bf);put(p,0,i,d*s*(1+c*(1-s)));}
}
void native_rows(std::vector<torch::Tensor> x,std::vector<torch::Tensor> y,int stage,int A,int B,int S,bool bf,double eps){
 TORCH_CHECK(x.size()<=12&&y.size()<=8,"native row arity");swa_native::Rows p{};p.A=A;p.B=B;p.S=S;p.stage=stage;p.bf=bf;p.eps=eps;
 for(int k=0;k<x.size();++k){TORCH_CHECK(x[k].is_cuda()&&x[k].is_contiguous(),"native row inputs must be contiguous");p.x[k]=x[k].data_ptr();p.xb[k]=x[k].scalar_type()==torch::kBFloat16;}
 for(int k=0;k<y.size();++k){TORCH_CHECK(y[k].is_cuda()&&y[k].is_contiguous(),"native row outputs must be contiguous");p.y[k]=y[k].data_ptr();p.yb[k]=y[k].scalar_type()==torch::kBFloat16;}
 if(stage==14)swa_native::launch(swa_native::mod_bwd,(x[0].numel()+127)/128,p,(int)x[0].numel());
 else {int nr=(stage==6||stage==8||stage==9||stage==12)?B*S:A*B*S;swa_native::launch(swa_native::rows,(nr+3)/4,p);}
}
void native_copy(std::vector<torch::Tensor>x,std::vector<torch::Tensor>y,int stage){TORCH_CHECK(x.size()==y.size()&&x.size()<=8,"native copy arity");swa_native::Copy p{};p.count=x.size();p.stage=stage;long mx=0;
 for(int k=0;k<x.size();++k){TORCH_CHECK(x[k].is_cuda()&&y[k].is_cuda()&&x[k].is_contiguous()&&y[k].is_contiguous()&&x[k].numel()==y[k].numel(),"native copy shape/layout");p.x[k]=x[k].data_ptr();p.y[k]=y[k].data_ptr();p.xb[k]=x[k].scalar_type()==torch::kBFloat16;p.yb[k]=y[k].scalar_type()==torch::kBFloat16;p.sizes[k]=x[k].numel();mx=std::max(mx,p.sizes[k]);}swa_native::launch(swa_native::copies,(mx+127)/128,p);}
void native_attention(std::vector<torch::Tensor>x,std::vector<torch::Tensor>y,int stage){swa_native::Att p{};p.q=(const __nv_bfloat16*)x[0].data_ptr();p.k=(const __nv_bfloat16*)x[1].data_ptr();p.v=(const __nv_bfloat16*)x[2].data_ptr();p.used=x[3].data_ptr<int>();p.N=x[0].size(0);p.S=x[0].size(2);p.stage=stage;
 if(stage==0){p.o=(__nv_bfloat16*)y[0].data_ptr();p.olse=y[1].data_ptr<float>();}else{p.d=(const __nv_bfloat16*)x[4].data_ptr();p.lse=x[5].data_ptr<float>();p.delta=x[6].data_ptr<float>();p.dq=(__nv_bfloat16*)y[0].data_ptr();p.dk=(__nv_bfloat16*)y[1].data_ptr();p.dv=(__nv_bfloat16*)y[2].data_ptr();}swa_native::launch(swa_native::attention,(p.N*4*p.S+3)/4,p);}


// Hopper translation of the B200 window-attention lifetime: all three score
// tiles remain in registers until the window-wide maximum is known. Each
// persistent CTA owns 64 output rows; two TMA slots overlap adjacent items.
// Backward owns one dQ or one (dK,dV) tile, avoiding atomic gradient reductions.
namespace swa_tiled {
constexpr int Q = 64, K = 192, DH = 32, HN = 4, PITCH = 128;
constexpr float SCALE = .1767766952966369f, LOG2E = 1.4426950408889634f, LN2 = .6931471805599453f;
__device__ __forceinline__ float qsum(float x) {
 x += __shfl_xor_sync(0xffffffffu,x,1); return x+__shfl_xor_sync(0xffffffffu,x,2);
}
__device__ __forceinline__ float qmax(float x) {
 x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,1));return fmaxf(x,__shfl_xor_sync(0xffffffffu,x,2));
}
__device__ __forceinline__ float ex2(float x) {
 float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;
}
__device__ __forceinline__ void pv(float* d,const uint32_t* a,uint64_t b) {
 asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 1;"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15])
 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b):"memory");
}
template<int MODE, int SLOTS=2> __global__ void __launch_bounds__(160,1)
attention(swa_native::Att p,const __grid_constant__ CUtensorMap mq,
          const __grid_constant__ CUtensorMap mk,const __grid_constant__ CUtensorMap mv,
          const __grid_constant__ CUtensorMap md) {
 constexpr int QR=MODE==2?192:64, KR=MODE==2?64:192;
 constexpr int QB=QR*PITCH,KB=KR*PITCH,DB=MODE==0?0:QR*PITCH;
 constexpr int ST=QB+2*KB+DB;
 extern __shared__ __align__(1024) unsigned char sm[];
 auto* full=reinterpret_cast<uint64_t*>(sm+SLOTS*ST);auto* empty=full+SLOTS;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const int mt=p.S/64,items=p.N*HN*mt;
 const int nit=(items>(int)blockIdx.x)?(items-(int)blockIdx.x+(int)gridDim.x-1)/(int)gridDim.x:0;
 if(tid==0){for(int j=0;j<SLOTS;++j){tma::bar_init(full+j,1);tma::bar_init(empty+j,4);}tma::bar_init_fence();}
 __syncthreads();
 if(tid>=128) {
  if(tid==128)for(int it=0;it<nit;++it){
   const int item=blockIdx.x+it*gridDim.x,nh=item/mt,i0=(item%mt)*64;
   const int slot=it%SLOTS;if(it>=SLOTS)tma::wait(empty+slot,((it/SLOTS)-1)&1);
   unsigned char* st=sm+slot*ST;const int q0=i0-(MODE==2?64:0),k0=i0-(MODE==2?0:64);
   tma::expect_tx(full+slot,ST);
   tma::load_2d(&mq,tma::sa(st),full+slot,0,nh*p.S+q0);
   tma::load_2d(&mk,tma::sa(st+QB),full+slot,0,nh*p.S+k0);
   tma::load_2d(&mv,tma::sa(st+QB+KB),full+slot,0,nh*p.S+k0);
   if constexpr(MODE!=0)tma::load_2d(&md,tma::sa(st+QB+2*KB),full+slot,(nh%HN)*DH,(nh/HN)*p.S+q0);
  }
  return;
 }
 const int r0=warp*16+(lane>>2),c0=(lane&3)*2;
 for(int it=0;it<nit;++it){
  const int item=blockIdx.x+it*gridDim.x,nh=item/mt,i0=(item%mt)*64,n=nh/HN,h=nh%HN;
  const int used=p.used[n],slot=it%SLOTS;
  unsigned char* st=sm+slot*ST;unsigned char* qs=st;unsigned char* ks=st+QB;unsigned char* vs=st+QB+KB;unsigned char* ds=st+QB+2*KB;
  tma::wait(full+slot,(it/SLOTS)&1);
  if constexpr(MODE==0){
   float sc[3][32],acc[16]={},mx[2]={-INFINITY,-INFINITY},den[2]={};
   wg::fence();
#pragma unroll
   for(int tile=0;tile<3;++tile){
    wg::mma_ss_n64_first(wg::desc(qs),wg::desc(ks+tile*64*PITCH),sc[tile]);
    wg::mma_ss_n64(wg::desc(qs)+2,wg::desc(ks+tile*64*PITCH)+2,sc[tile],1);
   }
   wg::commit();wg::wait<0>();
#pragma unroll
   for(int rr=0;rr<2;++rr){
    const int qi=i0+r0+8*rr;
#pragma unroll
    for(int tile=0;tile<3;++tile)
#pragma unroll
     for(int j=0;j<8;++j)
#pragma unroll
      for(int v=0;v<2;++v){
       const int ki=i0-64+64*tile+j*8+c0+v,ix=4*j+2*rr+v;
       if(qi>=used||ki<0||ki>=used||abs(qi-ki)>64)sc[tile][ix]=-INFINITY;
       mx[rr]=fmaxf(mx[rr],sc[tile][ix]);
      }
    mx[rr]=qmax(mx[rr]);if(mx[rr]==-INFINITY)mx[rr]=0;
   }
#pragma unroll
   for(int tile=0;tile<3;++tile){
    uint32_t pa[4][4];
#pragma unroll
    for(int rr=0;rr<2;++rr)
#pragma unroll
     for(int j=0;j<8;++j){
      float p0=ex2(fmaf(sc[tile][4*j+2*rr],SCALE*LOG2E,-mx[rr]*SCALE*LOG2E));
      float p1=ex2(fmaf(sc[tile][4*j+2*rr+1],SCALE*LOG2E,-mx[rr]*SCALE*LOG2E));
      den[rr]+=p0+p1;pa[j>>1][2*(j&1)+rr]=pack(p0,p1);
     }
    wg::fence();
#pragma unroll
    for(int j=0;j<4;++j)pv(acc,pa[j],wg::desc(vs+tile*64*PITCH+j*16*PITCH));
    wg::commit();wg::wait<0>();
   }
#pragma unroll
   for(int rr=0;rr<2;++rr){
    const int qi=i0+r0+8*rr;const float dd=qsum(den[rr]),inv=dd>0?1.f/dd:0.f;
#pragma unroll
    for(int j=0;j<4;++j)*reinterpret_cast<__nv_bfloat162*>(p.o+((long)n*p.S+qi)*128+h*32+j*8+c0)=__floats2bfloat162_rn(acc[4*j+2*rr]*inv,acc[4*j+2*rr+1]*inv);
    if((lane&3)==0)p.olse[(long)nh*p.S+qi]=dd>0?(mx[rr]*SCALE*LOG2E+__log2f(dd))*LN2:0.f;
   }
  } else {
   float grad[16]={},gv[16]={};
#pragma unroll
   for(int tile=0;tile<3;++tile){
    float sc[32],dp[32];unsigned char* ma=MODE==1?qs:ks;unsigned char* mb=MODE==1?ks+tile*64*PITCH:qs+tile*64*PITCH;
    unsigned char* da=MODE==1?ds:vs;unsigned char* db=MODE==1?vs+tile*64*PITCH:ds+tile*64*PITCH;
    wg::fence();wg::mma_ss_n64_first(wg::desc(ma),wg::desc(mb),sc);
    wg::mma_ss_n64(wg::desc(ma)+2,wg::desc(mb)+2,sc,1);
    wg::mma_ss_n64_first(wg::desc(da),wg::desc(db),dp);
    wg::mma_ss_n64(wg::desc(da)+2,wg::desc(db)+2,dp,1);
    wg::commit();wg::wait<0>();
    uint32_t pd[4][4],pp[4][4];
#pragma unroll
    for(int rr=0;rr<2;++rr)
#pragma unroll
     for(int j=0;j<8;++j){
      float pr[2],dg[2];
#pragma unroll
      for(int v=0;v<2;++v){
       const int row=i0+r0+8*rr,col=i0-64+tile*64+j*8+c0+v;
       const int qi=MODE==1?row:col,ki=MODE==1?col:row,ix=4*j+2*rr+v;
       const bool valid=qi>=0&&qi<used&&ki>=0&&ki<used&&abs(qi-ki)<=64;
       const float ls=valid?p.lse[(long)nh*p.S+qi]:0.f,dd=valid?p.delta[(long)nh*p.S+qi]:0.f;
       pr[v]=valid?ex2(fmaf(sc[ix],SCALE*LOG2E,-ls*LOG2E)):0.f;
       dg[v]=pr[v]*(dp[ix]-dd);
      }
      pd[j>>1][2*(j&1)+rr]=pack(dg[0],dg[1]);if constexpr(MODE==2)pp[j>>1][2*(j&1)+rr]=pack(pr[0],pr[1]);
     }
    unsigned char* out_b=MODE==1?ks+tile*64*PITCH:qs+tile*64*PITCH;
    wg::fence();
#pragma unroll
    for(int j=0;j<4;++j){
     pv(grad,pd[j],wg::desc(out_b+j*16*PITCH));
     if constexpr(MODE==2)pv(gv,pp[j],wg::desc(ds+tile*64*PITCH+j*16*PITCH));
    }
    wg::commit();wg::wait<0>();
   }
#pragma unroll
   for(int rr=0;rr<2;++rr)
#pragma unroll
    for(int j=0;j<4;++j){
     const long pos=((long)nh*p.S+i0+r0+8*rr)*32+j*8+c0;
     auto* dst=MODE==1?p.dq:p.dk;
     *reinterpret_cast<__nv_bfloat162*>(dst+pos)=__floats2bfloat162_rn(grad[4*j+2*rr]*SCALE,grad[4*j+2*rr+1]*SCALE);
     if constexpr(MODE==2)*reinterpret_cast<__nv_bfloat162*>(p.dv+pos)=__floats2bfloat162_rn(gv[4*j+2*rr],gv[4*j+2*rr+1]);
    }
  }
  // All consumer warps finish shared reads before the producer can recycle it.
  asm volatile("bar.sync 1,128;":::"memory");wg::proxy_fence();
  if(lane==0)tma::arrive(empty+slot);
 }
}
CUtensorMap head_map(const torch::Tensor& t,int rows) {
 uint64_t gd[2]={32,(uint64_t)(t.numel()/32)},gs[1]={64};uint32_t box[2]={64,(uint32_t)rows};
 return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,t.data_ptr(),gd,gs,box,CU_TENSOR_MAP_SWIZZLE_128B);
}
CUtensorMap do_map(const torch::Tensor& t,int rows) {
 uint64_t gd[2]={128,(uint64_t)(t.numel()/128)},gs[1]={256};uint32_t box[2]={64,(uint32_t)rows};
 return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,t.data_ptr(),gd,gs,box,CU_TENSOR_MAP_SWIZZLE_128B);
}
template<int MODE, int SLOTS=2> void launch(swa_native::Att p,CUtensorMap mq,CUtensorMap mk,CUtensorMap mv,CUtensorMap md,int dev) {
 constexpr int smem=SLOTS*(MODE==0?57344:65536)+256;
 cudaKernel_t k;C10_CUDA_CHECK(cudaGetKernel(&k,attention<MODE,SLOTS>));
 C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)k,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 int waves=MODE==0?2:(SLOTS==1?2:1);
 int grid=std::min(p.N*4*(p.S/64),num_sms(dev)*waves);
 void* args[]={&p,&mq,&mk,&mv,&md};
 C10_CUDA_CHECK(cudaLaunchKernel((const void*)k,dim3(grid),dim3(160),args,smem,at::cuda::getCurrentCUDAStream()));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace swa_tiled

void tiled_attention(std::vector<torch::Tensor>x,std::vector<torch::Tensor>y,int stage) {
 TORCH_CHECK(stage>=0&&stage<=2&&x.size()==(stage?7:4)&&y.size()==(stage?3:2),"tiled attention arity");
 swa_native::Att p{};p.N=x[0].size(0);p.S=x[0].size(2);p.stage=stage;
 TORCH_CHECK(x[0].dim()==4&&x[0].size(1)==4&&x[0].size(3)==32&&p.S%64==0,"tiled SWA requires [N,4,S,32], S multiple64");
 for(int i=0;i<3;++i)TORCH_CHECK(x[i].is_cuda()&&x[i].is_contiguous()&&x[i].scalar_type()==torch::kBFloat16&&x[i].sizes()==x[0].sizes(),"tiled SWA QKV contract");
 TORCH_CHECK(x[3].scalar_type()==torch::kInt&&x[3].is_contiguous()&&x[3].numel()==p.N,"tiled SWA seqused contract");
 p.q=(const __nv_bfloat16*)x[0].data_ptr();p.k=(const __nv_bfloat16*)x[1].data_ptr();p.v=(const __nv_bfloat16*)x[2].data_ptr();p.used=x[3].data_ptr<int>();
 const int qr=stage==2?192:64,kr=stage==2?64:192;
 auto mq=swa_tiled::head_map(x[0],qr),mk=swa_tiled::head_map(x[1],kr),mv=swa_tiled::head_map(x[2],kr),md=mq;
 if(stage==0){p.o=(__nv_bfloat16*)y[0].data_ptr();p.olse=y[1].data_ptr<float>();swa_tiled::launch<0>(p,mq,mk,mv,md,x[0].get_device());}
 else{
  TORCH_CHECK(x[4].is_contiguous()&&x[4].scalar_type()==torch::kBFloat16&&x[4].numel()==(long)p.N*p.S*128,"tiled SWA dO contract");
  p.d=(const __nv_bfloat16*)x[4].data_ptr();p.lse=x[5].data_ptr<float>();p.delta=x[6].data_ptr<float>();
  p.dq=(__nv_bfloat16*)y[0].data_ptr();p.dk=(__nv_bfloat16*)y[1].data_ptr();p.dv=(__nv_bfloat16*)y[2].data_ptr();md=swa_tiled::do_map(x[4],qr);
  // One attention smem slot (the two-slot launch<1> / launch<2> instantiations stay in the template, unselected).
  if(stage==1)swa_tiled::launch<1,1>(p,mq,mk,mv,md,x[0].get_device());else swa_tiled::launch<2,1>(p,mq,mk,mv,md,x[0].get_device());
 }
}

std::vector<torch::Tensor> native_ffn_pack(torch::Tensor wu, torch::Tensor wd, torch::Tensor dmod, bool clear) {
 TORCH_CHECK(wu.scalar_type()==torch::kBFloat16&&wd.scalar_type()==torch::kBFloat16&&wu.sizes()==torch::IntArrayRef({512,128})&&wd.sizes()==torch::IntArrayRef({128,256}),"native FFN pack requires BF16 Wu[512,128], Wd[128,256]");
 TORCH_CHECK(dmod.is_cuda()&&dmod.is_contiguous()&&dmod.scalar_type()==torch::kFloat,"native FFN dmod must be contiguous FP32");
 auto wab=torch::empty_like(wu),wdt=torch::empty({256,128},wd.options()),wabt=torch::empty({128,512},wu.options());
 native_copy({wu,wd,wu},{wab,wdt,wabt},3);
 if(clear) C10_CUDA_CHECK(cudaMemsetAsync(dmod.data_ptr(),0,dmod.numel()*sizeof(float),at::cuda::getCurrentCUDAStream()));
 return {wab,wdt,wabt};
}

// B200 T7 lifetime on Hopper: Wo is resident in shared memory, datt stays in
// registers through WGMMA, and the gate/attention epilogue consumes dgated
// without materializing it. Required protocol conditioning has one augment.
namespace swa_fused_back {
struct OProj {
 const __nv_bfloat16 *dq1,*att,*o,*g;
 const float* mod;
 __nv_bfloat16 *datt,*gated,*dO,*dG;
 float *delta,*dmod;
 int M,S;
};
__device__ __forceinline__ int offw(int r,int c) {
 return (c>>6)*16384+r*128+((((c&63)>>3)^(r&7))<<4)+(c&7)*2;
}
__global__ void __launch_bounds__(160,1)
oproj(OProj p,const __grid_constant__ CUtensorMap mw) {
 extern __shared__ __align__(1024) unsigned char sm[];
 auto* bar=reinterpret_cast<uint64_t*>(sm+65536);
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 if(tid==0){tma::bar_init(bar,1);tma::bar_init_fence();}
 __syncthreads();
 if(tid==128) {
  tma::expect_tx(bar,32768);
  tma::load_2d(&mw,tma::sa(sm),bar,0,0);
  tma::load_2d(&mw,tma::sa(sm+16384),bar,64,0);
 }
 tma::wait(bar,0);
 if(tid<128)for(int i=tid;i<16384;i+=128){
  const int r=i>>7,c=i&127;
  *reinterpret_cast<__nv_bfloat16*>(sm+32768+offw(c,r))=
   *reinterpret_cast<__nv_bfloat16*>(sm+offw(r,c));
 }
 __syncthreads();wg::proxy_fence();
 if(tid>=128)return;
 const int r0=warp*16+(lane>>2),c0=(lane&3)*2;
 for(int tile=blockIdx.x;tile<(p.M+63)/64;tile+=gridDim.x) {
  const int base=tile*64;
  uint32_t fa[8][4];
#pragma unroll
  for(int rr=0;rr<2;++rr)
#pragma unroll
   for(int j=0;j<16;++j){
    const int r=base+r0+rr*8,c=j*8+c0;const long idx=(long)r*128+c;
    float d0=0,d1=0;
    if(r<p.M){
     const float2 d=unpack(*reinterpret_cast<const uint32_t*>(p.dq1+idx));
     const float2 att=unpack(*reinterpret_cast<const uint32_t*>(p.att+idx));
     const float2 mod=*reinterpret_cast<const float2*>(p.mod+(long)r*768+256+c);
     d0=d.x*rnb(mod.x);d1=d.y*rnb(mod.y);
     *reinterpret_cast<float2*>(p.dmod+(long)r*768+256+c)=make_float2(d.x*att.x,d.y*att.y);
     *reinterpret_cast<uint32_t*>(p.datt+idx)=pack(d0,d1);
    }
    fa[j>>1][2*(j&1)+rr]=pack(d0,d1);
   }
  float acc[64];wg::fence();
#pragma unroll
  for(int k=0;k<8;++k){
   const uint64_t desc=wg::desc(sm+32768+(k>>2)*16384)+(k&3)*2;
   if(k==0)wg::mma_rs_n128_first(fa[k][0],fa[k][1],fa[k][2],fa[k][3],desc,acc);
   else wg::mma_rs_n128(fa[k][0],fa[k][1],fa[k][2],fa[k][3],desc,acc,1);
  }
  wg::commit();wg::wait<0>();
#pragma unroll
  for(int rr=0;rr<2;++rr){
   const int r=base+r0+rr*8;
#pragma unroll
   for(int h=0;h<4;++h){
    float dd=0;
#pragma unroll
    for(int j=0;j<4;++j){
     const int nc=h*4+j,c=nc*8+c0,ai=4*nc+2*rr;const long idx=(long)r*128+c;
     float2 o={0,0},g={0,0};if(r<p.M){o=unpack(*reinterpret_cast<const uint32_t*>(p.o+idx));g=unpack(*reinterpret_cast<const uint32_t*>(p.g+idx));}
     const float s0=swa_native::sig(g.x),s1=swa_native::sig(g.y);
     const float do0=rnb(acc[ai]*s0),do1=rnb(acc[ai+1]*s1);
     dd+=do0*o.x+do1*o.y;
     if(r<p.M){
      *reinterpret_cast<uint32_t*>(p.gated+idx)=pack(o.x*s0,o.y*s1);
      *reinterpret_cast<uint32_t*>(p.dO+idx)=pack(do0,do1);
      *reinterpret_cast<uint32_t*>(p.dG+idx)=pack(acc[ai]*o.x*s0*(1-s0),acc[ai+1]*o.y*s1*(1-s1));
     }
    }
    dd=swa_tiled::qsum(dd);
    if(r<p.M&&(lane&3)==0)p.delta[((long)(r/p.S)*4+h)*p.S+r%p.S]=dd;
   }
  }
 }
}
} // namespace swa_fused_back

void fused_oproj_bwd(std::vector<torch::Tensor>x,std::vector<torch::Tensor>y,int S) {
 TORCH_CHECK(x.size()==6&&y.size()==6&&S>0,"fused SWA oproj arity");
 const int M=x[0].numel()/128;
 for(int i:{0,2,3,4,5})TORCH_CHECK(x[i].is_cuda()&&x[i].is_contiguous()&&x[i].scalar_type()==torch::kBFloat16,"fused oproj BF16 inputs");
 TORCH_CHECK(x[5].numel()==16384&&x[1].scalar_type()==torch::kFloat&&x[1].numel()==(long)M*768,"fused oproj unshared modulation");
 swa_fused_back::OProj p{};
 p.dq1=(const __nv_bfloat16*)x[0].data_ptr();p.mod=x[1].data_ptr<float>();p.att=(const __nv_bfloat16*)x[2].data_ptr();
 p.o=(const __nv_bfloat16*)x[3].data_ptr();p.g=(const __nv_bfloat16*)x[4].data_ptr();p.M=M;p.S=S;
 p.datt=(__nv_bfloat16*)y[0].data_ptr();p.gated=(__nv_bfloat16*)y[1].data_ptr();p.dO=(__nv_bfloat16*)y[2].data_ptr();p.dG=(__nv_bfloat16*)y[3].data_ptr();p.delta=y[4].data_ptr<float>();p.dmod=y[5].data_ptr<float>();
 uint64_t gd[2]={128,128},gs[1]={256};uint32_t box[2]={64,128};
 auto mw=encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,x[5].data_ptr(),gd,gs,box,CU_TENSOR_MAP_SWIZZLE_128B);
 cudaKernel_t k;C10_CUDA_CHECK(cudaGetKernel(&k,swa_fused_back::oproj));
 constexpr int smem=65536+256;C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)k,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 int grid=std::min((M+63)/64,num_sms(x[0].get_device())*2);
 void* args[]={&p,&mw};C10_CUDA_CHECK(cudaLaunchKernel((const void*)k,dim3(grid),dim3(160),args,smem,at::cuda::getCurrentCUDAStream()));C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// B200 T9 boundary: RoPE/head-RMS gradients feed WGMMA directly; dP is
// retained only for the weight GEMM, and the final input adaLN consumes dX
// before it reaches HBM. Each modulation row has exactly one writer (A=1).
namespace swa_fused_back {
struct Qkvg {
 const __nv_bfloat16 *q,*pq,*pk,*dq,*dk,*dv,*dg,*dq1;
 const float *mod,*cs,*sn;
 __nv_bfloat16 *dx,*dp;
 float* dmod;
 int M,S;float eps;
};
__global__ void __launch_bounds__(160,1)
qkvg(Qkvg p,const __grid_constant__ CUtensorMap mw) {
 extern __shared__ __align__(1024) unsigned char sm[];
 auto* bar=reinterpret_cast<uint64_t*>(sm+147456);
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 if(tid==0){tma::bar_init(bar,1);tma::bar_init_fence();}
 __syncthreads();
 // A 16-KiB TMA scratch tile transposes into the 128-KiB resident W^T.
 for(int kb=0;kb<8;++kb){
  if(tid==128){
   tma::expect_tx(bar,16384);
   tma::load_2d(&mw,tma::sa(sm),bar,0,kb*64);
   tma::load_2d(&mw,tma::sa(sm+8192),bar,64,kb*64);
  }
  tma::wait(bar,kb&1);
  if(tid<128)for(int i=tid;i<8192;i+=128){
   const int r=i>>7,c=i&127;
   *reinterpret_cast<__nv_bfloat16*>(sm+16384+offw(c,kb*64+r))=
    *reinterpret_cast<__nv_bfloat16*>(sm+offb(r,c));
  }
  __syncthreads();
 }
 wg::proxy_fence();if(tid>=128)return;
 const int r0=warp*16+(lane>>2),c0=(lane&3)*2;
 for(int tile=blockIdx.x;tile<(p.M+63)/64;tile+=gridDim.x){
  const int base=tile*64;float dx[64];
#pragma unroll
  for(int part=0;part<4;++part){
   uint32_t fa[8][4];
#pragma unroll
   for(int rr=0;rr<2;++rr){
    const int r=base+r0+8*rr;
#pragma unroll
    for(int h=0;h<4;++h){
     float2 z[4],g[4];float zz=0;
#pragma unroll
     for(int j=0;j<4;++j){
      const int c=h*32+j*8+c0;const long ix=(long)r*128+c;
      const long hm=((long)(r/p.S)*4+h)*p.S*32+(long)(r%p.S)*32+j*8+c0;
      z[j]=make_float2(0,0);g[j]=make_float2(0,0);
      if(r<p.M){
       if(part<2){
        z[j]=unpack(*reinterpret_cast<const uint32_t*>((part==0?p.pq:p.pk)+ix));
        g[j]=unpack(*reinterpret_cast<const uint32_t*>((part==0?p.dq:p.dk)+hm));
        zz+=z[j].x*z[j].x+z[j].y*z[j].y;
       }else if(part==2)g[j]=unpack(*reinterpret_cast<const uint32_t*>(p.dv+hm));
       else g[j]=unpack(*reinterpret_cast<const uint32_t*>(p.dg+ix));
      }
     }
     if(part<2){
      const float rs=rsqrtf(swa_tiled::qsum(zz)/32+p.eps);
      float2 un[4];float dot=0;
#pragma unroll
      for(int j=0;j<4;++j){
       float2 cs=make_float2(1,1),sn=make_float2(0,0);
       if(r<p.M){cs=*reinterpret_cast<const float2*>(p.cs+(long)r*16+(j&1)*8+c0);sn=*reinterpret_cast<const float2*>(p.sn+(long)r*16+(j&1)*8+c0);}
       const float sign=j<2?1.f:-1.f;
       un[j]=make_float2(rnb(g[j].x*cs.x+sign*g[j^2].x*sn.x),rnb(g[j].y*cs.y+sign*g[j^2].y*sn.y));
       z[j].x*=rs;z[j].y*=rs;dot+=un[j].x*z[j].x+un[j].y*z[j].y;
      }
      dot=swa_tiled::qsum(dot)/32;
#pragma unroll
      for(int j=0;j<4;++j)g[j]=make_float2(rs*(un[j].x-z[j].x*dot),rs*(un[j].y-z[j].y*dot));
     }
#pragma unroll
     for(int j=0;j<4;++j){
      const int nc=h*4+j;const uint32_t v=pack(g[j].x,g[j].y);
      fa[nc>>1][2*(nc&1)+rr]=v;
      if(r<p.M)*reinterpret_cast<uint32_t*>(p.dp+(long)r*512+part*128+nc*8+c0)=v;
     }
    }
   }
   wg::fence();
#pragma unroll
   for(int k=0;k<8;++k){
    const uint64_t desc=wg::desc(sm+16384+part*32768+(k>>2)*16384)+(k&3)*2;
    if(part==0&&k==0)wg::mma_rs_n128_first(fa[k][0],fa[k][1],fa[k][2],fa[k][3],desc,dx);
    else wg::mma_rs_n128(fa[k][0],fa[k][1],fa[k][2],fa[k][3],desc,dx,1);
   }
   wg::commit();wg::wait<0>();
  }
#pragma unroll
  for(int rr=0;rr<2;++rr){
   const int r=base+r0+8*rr;float2 q[16],dxh[16];float ss=0,dot=0;
#pragma unroll
   for(int j=0;j<16;++j){
    q[j]=make_float2(0,0);if(r<p.M)q[j]=unpack(*reinterpret_cast<const uint32_t*>(p.q+(long)r*128+j*8+c0));
    ss+=q[j].x*q[j].x+q[j].y*q[j].y;
   }
   const float rs=rsqrtf(swa_tiled::qsum(ss)/128+p.eps);
#pragma unroll
   for(int j=0;j<16;++j){
    const int c=j*8+c0,ai=4*j+2*rr;q[j].x*=rs;q[j].y*=rs;
    float2 sc=make_float2(0,0);if(r<p.M)sc=*reinterpret_cast<const float2*>(p.mod+(long)r*768+128+c);
    dxh[j]=make_float2(dx[ai]*(1+sc.x),dx[ai+1]*(1+sc.y));dot+=dxh[j].x*q[j].x+dxh[j].y*q[j].y;
    if(r<p.M){
     *reinterpret_cast<float2*>(p.dmod+(long)r*768+c)=make_float2(dx[ai],dx[ai+1]);
     *reinterpret_cast<float2*>(p.dmod+(long)r*768+128+c)=make_float2(dx[ai]*q[j].x,dx[ai+1]*q[j].y);
    }
   }
   dot=swa_tiled::qsum(dot)/128;
#pragma unroll
   for(int j=0;j<16;++j)if(r<p.M){
    const long ix=(long)r*128+j*8+c0;const float2 skip=unpack(*reinterpret_cast<const uint32_t*>(p.dq1+ix));
    *reinterpret_cast<uint32_t*>(p.dx+ix)=pack(skip.x+rs*(dxh[j].x-q[j].x*dot),skip.y+rs*(dxh[j].y-q[j].y*dot));
   }
  }
 }
}
} // namespace swa_fused_back
void fused_qkvg_bwd(std::vector<torch::Tensor>x,std::vector<torch::Tensor>y,int S,double eps){
 TORCH_CHECK(x.size()==12&&y.size()==3&&S>0,"fused SWA qkvg arity");
 const int M=x[0].numel()/128;
 for(int i:{0,4,5,6,7,8,9,10,11})TORCH_CHECK(x[i].is_cuda()&&x[i].is_contiguous()&&x[i].scalar_type()==torch::kBFloat16,"fused qkvg BF16 inputs");
 TORCH_CHECK(x[11].numel()==65536&&x[1].scalar_type()==torch::kFloat&&x[1].numel()==(long)M*768,"fused qkvg unshared modulation");
 swa_fused_back::Qkvg p{};p.M=M;p.S=S;p.eps=eps;
 p.q=(const __nv_bfloat16*)x[0].data_ptr();p.mod=x[1].data_ptr<float>();p.cs=x[2].data_ptr<float>();p.sn=x[3].data_ptr<float>();
 p.pq=(const __nv_bfloat16*)x[4].data_ptr();p.pk=(const __nv_bfloat16*)x[5].data_ptr();
 p.dq=(const __nv_bfloat16*)x[6].data_ptr();p.dk=(const __nv_bfloat16*)x[7].data_ptr();p.dv=(const __nv_bfloat16*)x[8].data_ptr();p.dg=(const __nv_bfloat16*)x[9].data_ptr();p.dq1=(const __nv_bfloat16*)x[10].data_ptr();
 p.dx=(__nv_bfloat16*)y[0].data_ptr();p.dp=(__nv_bfloat16*)y[1].data_ptr();p.dmod=y[2].data_ptr<float>();
 uint64_t gd[2]={128,512},gs[1]={256};uint32_t box[2]={64,64};
 auto mw=encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,x[11].data_ptr(),gd,gs,box,CU_TENSOR_MAP_SWIZZLE_128B);
 cudaKernel_t k;C10_CUDA_CHECK(cudaGetKernel(&k,swa_fused_back::qkvg));
 constexpr int smem=147456+256;C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)k,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 int grid=std::min((M+63)/64,num_sms(x[0].get_device()));
 void* args[]={&p,&mw};C10_CUDA_CHECK(cudaLaunchKernel((const void*)k,dim3(grid),dim3(160),args,smem,at::cuda::getCurrentCUDAStream()));C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
struct SwaHandle { cublasHandle_t value; SwaHandle(){TORCH_CHECK(cublasCreate(&value)==CUBLAS_STATUS_SUCCESS,"SWA cublasCreate");} ~SwaHandle(){cublasDestroy(value);} };
torch::Tensor native_owned_mm(torch::Tensor a,torch::Tensor b,bool tf32) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()&&a.dim()==2&&b.dim()==2&&a.size(1)==b.size(0)&&a.scalar_type()==b.scalar_type()&&(a.scalar_type()==at::kFloat||a.scalar_type()==at::kBFloat16),"SWA owned GEMM operands");
 c10::cuda::CUDAGuard guard(a.device());
 static thread_local std::map<int,std::unique_ptr<SwaHandle>> handles;
 auto& owned=handles[a.get_device()];if(!owned)owned=std::make_unique<SwaHandle>();
 auto ck=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"SWA private cuBLAS status ",int(s));};
 auto handle=owned->value;auto stream=at::cuda::getCurrentCUDAStream();
 ck(cublasSetStream(handle,stream));ck(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));ck(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
 const bool fp32=a.scalar_type()==at::kFloat;
 ck(cublasSetMathMode(handle,fp32&&tf32?CUBLAS_TF32_TENSOR_OP_MATH:CUBLAS_DEFAULT_MATH));
 const size_t bytes=at::cuda::getChosenWorkspaceSize();auto workspace=torch::empty({int64_t(bytes)},a.options().dtype(at::kByte));
 ck(cublasSetWorkspace(handle,workspace.data_ptr(),bytes));
 auto out=torch::empty({a.size(0),b.size(1)},a.options().dtype(at::kFloat));
 auto leading=[](torch::Tensor t){bool row=t.stride(1)==1;TORCH_CHECK(row?t.stride(0)>=t.size(1):(t.stride(0)==1&&t.stride(1)>=t.size(0)),"SWA dense matrix view");return int(row?t.stride(0):t.stride(1));};
 int lda=leading(a),ldb=leading(b),m=a.size(0),n=b.size(1),k=a.size(1);float alpha=1,beta=0;
 const auto dtype=fp32?CUDA_R_32F:CUDA_R_16BF;const auto compute=fp32&&tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
 ck(cublasGemmEx(handle,b.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,a.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,n,m,k,&alpha,b.data_ptr(),dtype,ldb,a.data_ptr(),dtype,lda,&beta,out.data_ptr(),CUDA_R_32F,n,compute,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("native_owned_mm",&native_owned_mm);
  m.def("native_rows",&native_rows);m.def("native_copy",&native_copy);m.def("native_attention",&native_attention);
  m.def("fused_oproj_bwd",&fused_oproj_bwd);
  m.def("fused_qkvg_bwd",&fused_qkvg_bwd);
  m.def("native_ffn_pack",&native_ffn_pack,pybind11::arg("wu"),pybind11::arg("wd"),pybind11::arg("dmod"),pybind11::arg("clear")=true);
  m.def("tiled_attention",&tiled_attention);
  m.def("qkvg_fwd", &qkvg_fwd, "ESMFold2 SWA atom block RMSNorm/adaLN + QKVG projection + qk-norm + RoPE (bf16 wgmma)");
}
