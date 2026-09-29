// ESMFold2 SWA atom block, second half forward (CUDA sm_90a, C = 128, SwiGLU hidden N = 256, bf16 operands, fp32 accumulate):
//   gated = rn(sigmoid(g) * o);  att = rn(gated Wo^T);  q1 = rn(q + rn(gate_a * att));
//   y = rn(RMS(q1) * (1 + scale_f) + shift_f);  a|b = y Wu^T;  h = rn(silu(a) * b);  ffn = rn(h Wd^T);  out = rn(q1 + rn(gate_f * ffn))
// (the rounding points follow the Triton kernel _oproj_ffn_fwd).  Optionally saves q1, att, y, ffn for the backward.
// Structure: persistent CTAs, two consumer warpgroups ping-pong on alternate tiles (each does its whole tile in the wgmma fragment
// layout: every activation feeding a GEMM is built in registers as the bf16 A operand, so no smem round trips), one producer warp
// (TMA: q / g / o tiles and the tile's adaLN rows per consumer set, and a weight ring shared by both consumers).
// Tiles are SP=8 augments x AT=8 atoms of one batch element (rows ((a*B + b)*S + s)), so the tile's modulation rows are 8 atoms.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <type_traits>

namespace {
constexpr int C = 128, N = 256, HC = 64, NCH = N / HC, BR = 64, SP = 8, AT = BR / SP;
constexpr int NST = 3, SLOT = 32768;                     // weight ring: Wo | per chunk [Wa_j ; Wb_j] (32 KB), Wd_j (16 KB)
constexpr int SPT = 1 + 2 * NCH;                          // stages per tile
constexpr int THREADS = 384;
constexpr int TB = BR * 128;                              // bf16 [64 rows][64] k-block (8 KB); a [64][128] tile = 2 k-blocks
constexpr int SQ = 0, SG = 2 * TB, SO = 4 * TB, SMOD = 6 * TB, SET = SMOD + 16 * AT * 128;   // per consumer: q | g | o | mod (16 KB)
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
__device__ __forceinline__ void store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void store_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(K) : "memory"); }
}  // namespace tma

__device__ __forceinline__ float rcp_approx(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float sigmoid(float x) { return rcp_approx(1.f + __expf(-x)); }
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float2 unpack(uint32_t u) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&u);
  return make_float2(__low2float(v), __high2float(v));
}
// byte offset of element c (even) of row r in a bf16 [64][128] tile stored as 2 k-blocks of [64 rows][128 B], 128 B swizzle
__device__ __forceinline__ int offb(int r, int c) { return (c >> 6) * TB + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2; }
// fp32 modulation tile [16 col-blocks (kinds 2..5 x 4)][AT atoms][32], 128 B swizzle: float2 at (kind k, atom al, channel c)
__device__ __forceinline__ const float2* modp(const unsigned char* sm, int k, int al, int c) {
  const int row = (k * 4 + (c >> 5)) * AT + al;
  return reinterpret_cast<const float2*>(sm + row * 128 + ((((c & 31) >> 2) ^ (row & 7)) << 4) + (c & 3) * 4);
}

struct Args {
  int S, A, B, nab, nag, ntile;
  float eps;
  int save;
};

__global__ void __launch_bounds__(THREADS, 1) ffn_fwd_kernel(Args args, const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mg,
                                                            const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mmod,
                                                            const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab,
                                                            const __grid_constant__ CUtensorMap mwd, const __grid_constant__ CUtensorMap mout,
                                                            const __grid_constant__ CUtensorMap mq1, const __grid_constant__ CUtensorMap matt,
                                                            const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mffn) {
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
    if (warp == 0 && lane == 0) {                          // activation tiles of consumer set T & 1
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
        const int st = T & 1;
        if (T >= 2) tma::wait(xempty + st, ((T >> 1) - 1) & 1);
        unsigned char* sx = smem + st * SET;
        tma::expect_tx(xfull + st, 6 * TB + 16 * AT * 128);
        for (int kb = 0; kb < 2; ++kb) {
          tma::load_4d(&mq, tma::sa(sx + SQ + kb * TB), xfull + st, kb * 64, s0, b, a0);
          tma::load_4d(&mg, tma::sa(sx + SG + kb * TB), xfull + st, kb * 64, s0, b, a0);
          tma::load_4d(&mo, tma::sa(sx + SO + kb * TB), xfull + st, kb * 64, s0, b, a0);
        }
        tma::load_3d(&mmod, tma::sa(sx + SMOD), xfull + st, 0, b * S + s0, 8);   // col-blocks 8..23 = gate_a, shift_f, scale_f, gate_f
      }
    } else if (warp == 1 && lane == 0) {                   // weight ring: per PAIR of tiles (both consumers read every stage)
      const int npair = (ntT + 1) / 2;
      int g = 0;
      for (int P = 0; P < npair; ++P)
        for (int s = 0; s < SPT; ++s, ++g) {
          const int st = g % NST;
          if (g >= NST) tma::wait(wempty + st, ((g / NST) - 1) & 1);
          unsigned char* d = sW + st * SLOT;
          if (s == 0) {                                    // Wo [128 out][128 in]: 2 k-blocks [128][64]
            tma::expect_tx(wfull + st, 32768);
            tma::load_2d(&mwo, tma::sa(d), wfull + st, 0, 0);
            tma::load_2d(&mwo, tma::sa(d + 16384), wfull + st, 64, 0);
          } else if ((s & 1) == 1) {                       // [Wa_j ; Wb_j]: 128 rows x 128 ch
            const int j = (s - 1) >> 1;
            tma::expect_tx(wfull + st, 32768);
            tma::load_2d(&mwab, tma::sa(d), wfull + st, 0, j * 128);
            tma::load_2d(&mwab, tma::sa(d + 16384), wfull + st, 64, j * 128);
          } else {                                         // Wd_j: [128 out][64 hidden]
            const int j = (s - 2) >> 1;
            tma::expect_tx(wfull + st, 16384);
            tma::load_2d(&mwd, tma::sa(d), wfull + st, j * HC, 0);
          }
        }
    }
    return;
  }
  // ================= consumers: WG c takes tiles T = c, c + 2, ... =================
  asm volatile("setmaxnreg.inc.sync.aligned.u32 " CONS_REGS ";\n" ::: "memory");
  const int cs = wgi;
  unsigned char* sx = smem + cs * SET;
  unsigned char *sQ = sx + SQ, *sG = sx + SG, *sO = sx + SO, *sM = sx + SMOD;
  const int r0 = warp * 16 + (lane >> 2), cq = (lane & 3) * 2, al = r0 % AT;   // rows r0 and r0 + 8 share the atom (AT = 8)
  const int bar_id = 1 + cs;
  const bool save = args.save != 0;
  auto bsync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory"); };
  float acc[64], acc2[64];
  uint32_t fa[32];                                          // bf16 A fragments over K = 128: k-step s -> fa[4s .. 4s+3]
  uint32_t ha[16];
  int P = 0;
  for (int T = cs; T < ntT; T += 2, ++P) {
    int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
    const int g0 = P * SPT;
    tma::wait(xfull + cs, (T >> 1) & 1);
    // ---- gated = rn(sigmoid(g) * o) as the A operand of att = gated Wo^T ----
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int h = 0; h < 2; ++h)                         // k half: cols 16s + 8h + cq
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {                  // rows r0, r0 + 8
          const int c = 16 * s + 8 * h + cq, r = r0 + 8 * rr;
          const float2 gv = unpack(*reinterpret_cast<const uint32_t*>(sG + offb(r, c))), ov = unpack(*reinterpret_cast<const uint32_t*>(sO + offb(r, c)));
          fa[4 * s + 2 * h + rr] = pack(sigmoid(gv.x) * ov.x, sigmoid(gv.y) * ov.y);
        }
    {
      const int st = g0 % NST;
      tma::wait(wfull + st, (g0 / NST) & 1);
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
    }
    // ---- q1 = rn(q + rn(gate_a * rn(att))); rstd; y = rn(q1 * rstd * (1 + scale_f) + shift_f) -> A fragments; saves ----
    float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = i * 8 + cq;
      const float2 ga = *modp(sM, 0, al, c);
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = r0 + 8 * rr;
        const float t0 = rnb(acc[4 * i + 2 * rr]), t1 = rnb(acc[4 * i + 2 * rr + 1]);
        uint32_t* pq = reinterpret_cast<uint32_t*>(sQ + offb(r, c));
        const float2 qv = unpack(*pq);
        const float q0 = rnb(qv.x + rnb(ga.x) * t0), q1 = rnb(qv.y + rnb(ga.y) * t1);
        acc[4 * i + 2 * rr] = q0; acc[4 * i + 2 * rr + 1] = q1;                     // acc now holds q1
        if (save) *reinterpret_cast<uint32_t*>(sG + offb(r, c)) = pack(t0, t1);    // att -> G (gated is in registers)
        *reinterpret_cast<uint32_t*>(sO + offb(r, c)) = pack(q0, q1);             // q1 -> O (kept for the output)
        if (rr == 0) ss0 += q0 * q0 + q1 * q1; else ss1 += q0 * q0 + q1 * q1;
      }
    }
    ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1); ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
    ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1); ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
    const float rs0 = rsqrtf(ss0 * (1.f / C) + args.eps), rs1 = rsqrtf(ss1 * (1.f / C) + args.eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = i * 8 + cq;
      const float2 sh = *modp(sM, 1, al, c), sc = *modp(sM, 2, al, c);
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const float rs = rr ? rs1 : rs0;
        const uint32_t yv = pack(acc[4 * i + 2 * rr] * rs * (1.f + sc.x) + sh.x, acc[4 * i + 2 * rr + 1] * rs * (1.f + sc.y) + sh.y);
        fa[4 * (i >> 1) + 2 * (i & 1) + rr] = yv;
        if (save) *reinterpret_cast<uint32_t*>(sQ + offb(r0 + 8 * rr, c)) = yv;    // y -> Q (q is dead)
      }
    }
    wg::proxy_fence();
    bsync();
    if (save && tq == 0) {
      for (int kb = 0; kb < 2; ++kb) {
        tma::store_4d(&matt, sG + kb * TB, kb * 64, s0, b, a0);
        tma::store_4d(&mq1, sO + kb * TB, kb * 64, s0, b, a0);
        tma::store_4d(&my, sQ + kb * TB, kb * 64, s0, b, a0);
      }
      tma::store_commit();
    }
    // ---- FFN: per 64-wide hidden chunk a|b = y [Wa_j ; Wb_j]^T (m64n128), h = rn(silu(a) b), ffn += h Wd_j^T ----
#pragma unroll 1
    for (int j = 0; j < NCH; ++j) {
      const int ga_ = g0 + 1 + 2 * j, gd_ = ga_ + 1;
      {
        const int st = ga_ % NST;
        tma::wait(wfull + st, (ga_ / NST) & 1);
        const unsigned char* bw = sW + st * SLOT;
        wg::fence();
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const uint64_t d = wg::desc(bw + (s >> 2) * 16384 + (s & 3) * 32);
          if (s == 0) wg::mma_rs_n128_first(fa[0], fa[1], fa[2], fa[3], d, acc);
          else wg::mma_rs_n128(fa[4 * s], fa[4 * s + 1], fa[4 * s + 2], fa[4 * s + 3], d, acc, 1);
        }
        wg::commit(); wg::wait<0>();                     // also retires the previous chunk's ffn GEMM
        if (lane == 0) { tma::arrive(wempty + st); if (j > 0) tma::arrive(wempty + (gd_ - 2) % NST); }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {                        // hidden cols 8i + cq: a = acc[4i..], b = acc[32 + 4i..]
        const float a0v = acc[4 * i], a1v = acc[4 * i + 1], a2v = acc[4 * i + 2], a3v = acc[4 * i + 3];
        const float h0 = a0v * sigmoid(a0v) * acc[32 + 4 * i], h1 = a1v * sigmoid(a1v) * acc[32 + 4 * i + 1];
        const float h2 = a2v * sigmoid(a2v) * acc[32 + 4 * i + 2], h3 = a3v * sigmoid(a3v) * acc[32 + 4 * i + 3];
        ha[4 * (i >> 1) + 2 * (i & 1)] = pack(h0, h1); ha[4 * (i >> 1) + 2 * (i & 1) + 1] = pack(h2, h3);
      }
      {
        const int st = gd_ % NST;
        tma::wait(wfull + st, (gd_ / NST) & 1);
        const unsigned char* bw = sW + st * SLOT;
        wg::fence();
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint64_t d = wg::desc(bw + s * 32);
          if (j == 0 && s == 0) wg::mma_rs_n128_first(ha[0], ha[1], ha[2], ha[3], d, acc2);
          else wg::mma_rs_n128(ha[4 * s], ha[4 * s + 1], ha[4 * s + 2], ha[4 * s + 3], d, acc2, 1);
        }
        wg::commit();
      }
    }
    wg::wait<0>();
    if (lane == 0) tma::arrive(wempty + (g0 + SPT - 1) % NST);
    // ---- out = rn(q1 + rn(gate_f * rn(ffn))); ffn saved ----
    if (tq == 0) tma::store_wait_read<0>();                // att / q1 / y stores have read G / O / Q
    bsync();
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = i * 8 + cq;
      const float2 gf = *modp(sM, 3, al, c);
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = r0 + 8 * rr;
        const float f0 = rnb(acc2[4 * i + 2 * rr]), f1 = rnb(acc2[4 * i + 2 * rr + 1]);
        uint32_t* po = reinterpret_cast<uint32_t*>(sO + offb(r, c));
        const float2 qv = unpack(*po);
        *po = pack(qv.x + rnb(gf.x) * f0, qv.y + rnb(gf.y) * f1);
        if (save) *reinterpret_cast<uint32_t*>(sG + offb(r, c)) = pack(f0, f1);
      }
    }
    wg::proxy_fence();
    bsync();
    if (tq == 0) {
      for (int kb = 0; kb < 2; ++kb) {
        tma::store_4d(&mout, sO + kb * TB, kb * 64, s0, b, a0);
        if (save) tma::store_4d(&mffn, sG + kb * TB, kb * 64, s0, b, a0);
      }
      tma::store_commit();
      tma::store_wait_read<0>();
      tma::arrive(xempty + cs);
    }
  }
  // a consumer with fewer tiles than its partner still has to release the ring stages of the pair it skips
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
CUtensorMap map2d(const torch::Tensor& t, uint64_t rows, uint64_t cols, uint32_t brows) {   // [rows][cols] bf16 -> box (64 cols, brows rows)
  alignas(64) CUtensorMap m{};
  uint64_t gdim[2] = {cols, rows}; uint64_t gstride[1] = {cols * 2};
  uint32_t bdim[2] = {64, brows}; uint32_t estride[2] = {1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(2d) failed: ", (int)r);
  return m;
}
CUtensorMap map4d(const torch::Tensor& t, long A, long B, long S) {   // [A][B][S][128] bf16 -> box (64 ch, AT atoms, 1, SP augments)
  alignas(64) CUtensorMap m{};
  uint64_t gdim[4] = {(uint64_t)C, (uint64_t)S, (uint64_t)B, (uint64_t)A};
  uint64_t gstride[3] = {(uint64_t)C * 2, (uint64_t)S * C * 2, (uint64_t)B * S * C * 2};
  uint32_t bdim[4] = {64, AT, 1, SP}; uint32_t estride[4] = {1, 1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(4d) failed: ", (int)r);
  return m;
}
CUtensorMap mapmod(const torch::Tensor& t, long rows) {   // [rows][24 blocks][32] fp32 viewed as (32 ch, rows, 24 blocks) -> box (32, AT, 16)
  alignas(64) CUtensorMap m{};
  uint64_t gdim[3] = {32, (uint64_t)rows, 24}; uint64_t gstride[2] = {6 * C * 4, 128};
  uint32_t bdim[3] = {32, AT, 16}; uint32_t estride[3] = {1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, t.data_ptr(), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(mod) failed: ", (int)r);
  return m;
}
int num_sms(int dev) { static int s = 0; if (s == 0) cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }
}  // namespace

// q, g, o: [M, 128] bf16 with rows ((a*B + b)*S + s); mod [B*S, 768] fp32; wo [128, 128]; wab [512, 128] = per 64-wide chunk j rows
// [Wu[64j..64j+63] ; Wu[256+64j..]]; wd [128, 256].  Returns [out, q1, att, y, ffn] (the last four empty unless save).
std::vector<torch::Tensor> ffn_fwd(torch::Tensor q, torch::Tensor g, torch::Tensor o, torch::Tensor mod, torch::Tensor wo, torch::Tensor wab,
                                   torch::Tensor wd, int64_t A, int64_t B, int64_t S, double eps, bool save) {
  const long M = A * B * S;
  for (const auto* t : {&q, &g, &o}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == M * C, "rows: [M, 128] bf16");
  TORCH_CHECK(mod.scalar_type() == torch::kFloat && mod.is_contiguous() && mod.numel() == B * S * 6 * C, "mod [B*S, 768] fp32");
  TORCH_CHECK(wo.is_contiguous() && wo.scalar_type() == torch::kBFloat16 && wo.sizes() == torch::IntArrayRef({C, C}), "wo [128, 128] bf16");
  TORCH_CHECK(wab.is_contiguous() && wab.scalar_type() == torch::kBFloat16 && wab.sizes() == torch::IntArrayRef({2 * N, C}), "wab [512, 128] bf16");
  TORCH_CHECK(wd.is_contiguous() && wd.scalar_type() == torch::kBFloat16 && wd.sizes() == torch::IntArrayRef({C, N}), "wd [128, 256] bf16");
  auto opt = q.options();
  auto out = torch::empty({M, C}, opt);
  const long Ms = save ? M : 1;
  auto q1 = torch::empty({Ms, C}, opt), att = torch::empty({Ms, C}, opt), y = torch::empty({Ms, C}, opt), ffn = torch::empty({Ms, C}, opt);
  CUtensorMap mq = map4d(q, A, B, S), mg = map4d(g, A, B, S), mo = map4d(o, A, B, S), mout = map4d(out, A, B, S);
  CUtensorMap mq1 = save ? map4d(q1, A, B, S) : mout, matt = save ? map4d(att, A, B, S) : mout, my = save ? map4d(y, A, B, S) : mout,
              mffn = save ? map4d(ffn, A, B, S) : mout;
  CUtensorMap mmod = mapmod(mod, B * S), mwo = map2d(wo, C, C, 128), mwab = map2d(wab, 2 * N, C, 128), mwd = map2d(wd, C, N, 128);
  Args args; args.S = (int)S; args.A = (int)A; args.B = (int)B; args.eps = (float)eps; args.save = save ? 1 : 0;
  args.nab = (int)((S + AT - 1) / AT); args.nag = (int)((A + SP - 1) / SP); args.ntile = args.nab * args.nag * (int)B;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(ffn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES)); attr = true; }
  const int grid = std::min(args.ntile, num_sms(q.device().index()));
  ffn_fwd_kernel<<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(args, mq, mg, mo, mmod, mwo, mwab, mwd, mout, mq1, matt, my, mffn);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, q1, att, y, ffn};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ffn_fwd", &ffn_fwd, "ESMFold2 SWA atom block out-proj + FFN forward (bf16 wgmma)");
}
