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
constexpr int C = 128, H = 4, D = 32, BR = 64, SP = 8, AT = BR / SP;
constexpr int NST = 3, SLOT = 32768, SPT = 4;             // weight ring: Wq | Wk | Wv | Wg, [128 out][128 in] each
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
    float2 cv[2], sv[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cv[i] = __ldg(reinterpret_cast<const float2*>(args.cs + (long)mrow * (D / 2) + 8 * i + cq));
      sv[i] = __ldg(reinterpret_cast<const float2*>(args.sn + (long)mrow * (D / 2) + 8 * i + cq));
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
      const float2 sh = *modp(sM, 0, al, c), sc = *modp(sM, 1, al, c);
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
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
                const float c_ = e ? cv[ib].y : cv[ib].x, s_ = e ? sv[ib].y : sv[ib].x;
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
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(qkvg_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES)); attr = true; }
  const int grid = std::min(args.ntile, num_sms(q.device().index()));
  qkvg_fwd_kernel<<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(args, mq, mmod, mw, mQ, mK, mV, mG, mX, mPQ, mPK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {Q, K, V, G, X, PQ, PK};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkvg_fwd", &qkvg_fwd, "ESMFold2 SWA atom block RMSNorm/adaLN + QKVG projection + qk-norm + RoPE (bf16 wgmma)");
}
