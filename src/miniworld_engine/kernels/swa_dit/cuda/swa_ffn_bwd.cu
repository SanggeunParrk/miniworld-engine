// ESMFold2 SWA atom block, FFN half backward (CUDA sm_90a, C = 128, SwiGLU hidden N = 256, bf16 operands, fp32 accumulate):
//   forward:  y = rn(RMS(q1) * (1 + scale_f) + shift_f);  a|b = y Wu^T;  h = silu(a) * b;  ffn = rn(h Wd^T);  q2 = q1 + gate_f * ffn
//   backward: dffn = rn(dq2 * gate_f);  dh = dffn Wd;  da = rn(dh*b*sa*(1 + a*(1-sa))), db = rn(dh*a*sa);  dy = da Wa + db Wb;
//             dq1 = dq2 + rstd*(dxh - xh*mean(dxh*xh)), dxh = dy*(1 + scale_f), xh = q1*rstd;
//             d scale_f = sum_aug dy*xh, d shift_f = sum_aug dy, d gate_f = sum_aug dq2*ffn  (atomics into dmod [B*S][6C]).
// Also writes the bf16 operands of the weight gradients (dffn [M][C], h [M][N], [da|db] [M][2N]); dWu = [da|db]^T y, dWd = dffn^T h
// run as cuBLAS GEMMs on the host.  a|b are recomputed from the saved y (the forward does not store them).
// Structure (port of the AF3 transition bwd pass 1): persistent CTAs; WG0 = GEMMs of every tile (hidden chunks of 32: a|b m64n64 SS,
// dh m64n32 SS, then dy += [da|db] Wab_j as RS m64n128 with the accumulator fragment used directly as the bf16 A operand);
// WG1 / WG2 = row work (pass A: dffn; final pass: dq1 + per-atom sums) for tiles of buffer set 0 / 1; WG3 = TMA producer
// (y / dq2 tiles as 4-D boxes of SP augments x AT atoms of one batch element, and a ring of weight stages).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <type_traits>

namespace {
constexpr int C = 128, N = 256, HC = 32, NCH = N / HC, BR = 64;
constexpr int NST = 6, SPC = 3;                           // weight ring (16 KB slots): per chunk [Wa_j ; Wb_j] | Wd^T_j | [Wa_j ; Wb_j]^T
constexpr int THREADS = 512;
#ifndef GEMM_REGS
#define GEMM_REGS "184"
#define ROW_REGS "136"
#define PROD_REGS "56"
#endif
#ifndef FIN_ILP
#define FIN_ILP 4
#endif
constexpr int TB = BR * 128;                              // one bf16 k-block of a tile: 64 rows x 128 B
constexpr int SET = 65536;                                // per buffer set: X (y) 16 KB | D (dq2 -> dffn) 16 KB | P (fp32 partials) 32 KB; X|D -> fp32 dy
constexpr int SSET = 0, SW = 2 * SET, SEND = SW + NST * 16384;
constexpr int NBAR = 10 + 2 * NST;
constexpr int BYTES = SEND + NBAR * 8;
constexpr int XTF = BR * 32;                              // fp32 [64][32] block (floats)

namespace wg {
__device__ __forceinline__ int off32(int r, int c4) { return r * 32 + ((c4 ^ (r & 7)) << 2); }
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
// byte offset of bf16 element c (< 32) of row r in a [64 rows][64 B] staging tile, 64 B swizzle (conflict-free fragment writes)
__device__ __forceinline__ int off64(int r, int c) { return r * 64 + ((((c >> 3) ^ (r >> 1)) & 3) << 4) + (c & 7) * 2; }

__device__ __forceinline__ float rcp_approx(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float sigmoid(float x) { return rcp_approx(1.f + __expf(-x)); }
__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {       // 4 bf16 (8 B) -> float4
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x), b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
  return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}
__device__ __forceinline__ float4 ldg4(const __nv_bfloat16* p) {
  const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x), b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
  return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}
__device__ __forceinline__ uint2 pk4(float4 v) { return make_uint2(pack(v.x, v.y), pack(v.z, v.w)); }
// byte offset of element c (multiple of 4) of row r in a bf16 [64][128] tile stored as 2 k-blocks of [64 rows][128 B], 128 B swizzle
__device__ __forceinline__ int offb(int r, int c) { return (c >> 6) * TB + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2; }

struct Args {
  const __nv_bfloat16 *dq2, *q1, *ffn;
  const float* mod;                                        // [B*S][6C]
  __nv_bfloat16 *dq1, *dffn, *h, *dab;
  float* dmod;
  int S, A, B, nab, nag, ntile;
  float eps;
  int ablate;
  unsigned long long* prof;                                             // experiments: 1 = no dab/h stores, 2 = no row final pass, 4 = no pass A, 8 = no GEMMs
};

template <int SP>
__global__ void __launch_bounds__(THREADS, 1) ffn_bwd_kernel(Args args, const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mdq,
                                                            const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwdt,
                                                            const __grid_constant__ CUtensorMap mwabt, const __grid_constant__ CUtensorMap mdab,
                                                            const __grid_constant__ CUtensorMap mh) {
  constexpr int AT = BR / SP;                             // tile row r = sl * AT + al  (augment a0 + sl, atom s0 + al)
  extern __shared__ __align__(1024) unsigned char smem[];
  if (tma::sa(smem) & 1023u) __trap();
  unsigned char* sW = smem + SW;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + SEND);
  uint64_t *xfull = bars, *xempty = bars + 2, *afull = bars + 4, *cfull = bars + 6, *pfree = bars + 8, *wfull = bars + 10, *wempty = bars + 10 + NST;
  const int tid = threadIdx.x, lane = tid & 31, wgi = tid >> 7, warp = (tid >> 5) & 3, tq = tid & 127;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { tma::bar_init(xfull + i, 1); tma::bar_init(xempty + i, 1); tma::bar_init(afull + i, 1); tma::bar_init(cfull + i, 1); tma::bar_init(pfree + i, 1); }
    for (int i = 0; i < NST; ++i) { tma::bar_init(wfull + i, 1); tma::bar_init(wempty + i, 4); }
    tma::bar_init_fence(); wg::proxy_fence();
  }
  __syncthreads();
  const int ntT = (int)blockIdx.x < args.ntile ? (args.ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const long S = args.S, A = args.A, B = args.B;
  long long pt_ = clock64();
#define PR(k, who) do { if (args.prof != nullptr && (who)) { const long long c_ = clock64(); atomicAdd(args.prof + (k), (unsigned long long)(c_ - pt_)); pt_ = c_; } } while (0)
  auto coords = [&](int t, int& b, int& a0, int& s0) {
    const int ab = t % args.nab, r = t / args.nab, ag = r % args.nag;
    b = r / args.nag; a0 = ag * SP; s0 = ab * AT;
  };
  if (wgi == 3) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 " PROD_REGS ";\n" ::: "memory");
    if (warp >= 2) {                                       // pass A (warps 2, 3: 32 rows each): dffn = rn(dq2 * rn(gate_f)) -> D (in place) and global
      const int c0 = lane * 4;
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
        const int st = T & 1;
        unsigned char* sD = smem + SSET + st * SET + 2 * TB;
        PR(9, warp == 2 && lane == 0);
        tma::wait(xfull + st, (T >> 1) & 1);
        PR(10, warp == 2 && lane == 0);
#pragma unroll 1
        for (int i0 = 0; i0 < 32; i0 += 2) {
          float4 gv[2]; long Rr[2]; bool okr[2];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int rl = (warp - 2) * 32 + i0 + j;
            const long a = a0 + rl / AT, s_ = s0 + rl % AT;
            okr[j] = a < A && s_ < S;
            Rr[j] = (a * B + b) * S + s_;
            gv[j] = __ldg(reinterpret_cast<const float4*>(args.mod + ((long)b * S + (s_ < S ? s_ : 0)) * (6 * C) + 5 * C + c0));
          }
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int rl = (warp - 2) * 32 + i0 + j;
            __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(sD + offb(rl, c0));
            const float4 d = ld4(p);
            const uint2 o = pk4(make_float4(d.x * rnb(gv[j].x), d.y * rnb(gv[j].y), d.z * rnb(gv[j].z), d.w * rnb(gv[j].w)));
            *reinterpret_cast<uint2*>(p) = o;
            if (okr[j]) *reinterpret_cast<uint2*>(args.dffn + Rr[j] * C + c0) = o;
          }
        }
        wg::proxy_fence();
        asm volatile("bar.sync 5, 64;\n" ::: "memory");
        PR(11, warp == 2 && lane == 0);
        if (warp == 2 && lane == 0) tma::arrive(afull + st);
      }
      return;
    }
    if (warp == 0 && lane == 0) {                          // y and dq2 tiles
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
        const int st = T & 1;
        if (T >= 2) tma::wait(xempty + st, ((T >> 1) - 1) & 1);
        unsigned char* sX = smem + SSET + st * SET;
        tma::expect_tx(xfull + st, 4 * TB);
        for (int kb = 0; kb < 2; ++kb) {
          tma::load_4d(&my, tma::sa(sX + kb * TB), xfull + st, kb * 64, s0, b, a0);
          tma::load_4d(&mdq, tma::sa(sX + 2 * TB + kb * TB), xfull + st, kb * 64, s0, b, a0);
        }
      }
    } else if (warp == 1 && lane == 0) {                   // weight stages
      int g = 0;
      for (int T = 0; T < ntT; ++T)
        for (int j = 0; j < NCH; ++j)
          for (int s = 0; s < SPC; ++s, ++g) {
            const int st = g % NST;
            if (g >= NST) tma::wait(wempty + st, ((g / NST) - 1) & 1);
            unsigned char* d = sW + st * 16384;
            if (s == 0) {                                  // [Wa_j ; Wb_j]: 64 rows x 128 ch = 2 k-blocks [64][64]
              tma::expect_tx(wfull + st, 16384);
              tma::load_2d(&mwab, tma::sa(d), wfull + st, 0, j * 64);
              tma::load_2d(&mwab, tma::sa(d + 8192), wfull + st, 64, j * 64);
            } else if (s == 1) {                           // Wd^T_j: 32 hidden rows x 128 ch = 2 k-blocks [32][64]
              tma::expect_tx(wfull + st, 8192);
              tma::load_2d(&mwdt, tma::sa(d), wfull + st, 0, j * HC);
              tma::load_2d(&mwdt, tma::sa(d + 4096), wfull + st, 64, j * HC);
            } else {                                       // [Wa_j ; Wb_j]^T: 128 ch rows x 64 k
              tma::expect_tx(wfull + st, 16384);
              tma::load_2d(&mwabt, tma::sa(d), wfull + st, j * 64, 0);
            }
          }
    }
    return;
  }
  const int r0 = warp * 16 + (lane >> 2), cq = (lane & 3) * 2;
  if (wgi == 0) {
    // ================= GEMM warpgroup =================
    asm volatile("setmaxnreg.inc.sync.aligned.u32 " GEMM_REGS ";\n" ::: "memory");
    float acc_ab[32], acc_dh[16], acc_dx[64];
    uint32_t pk[16], hp[8];
    int g = 0;
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
      const int st = T & 1;
      unsigned char* sX = smem + SSET + st * SET;
      unsigned char* sD = sX + 2 * TB;
      unsigned char* sStg = sX + 4 * TB;                   // the set's partials region is free while its tile is in the GEMMs: [da|db|h] staging x 2
      PR(0, tq == 0);
      tma::wait(afull + st, (T >> 1) & 1);
      PR(1, tq == 0);
      auto grow = [&](int rl, bool& ok) {
        const long a = a0 + rl / AT, s = s0 + rl % AT;
        ok = a < A && s < S;
        return ok ? (a * B + b) * S + s : 0;
      };
      bool ok0, ok1;
      const long R0 = grow(r0, ok0), R1 = grow(r0 + 8, ok1);
      auto chunk = [&](auto first_tag, int j) {
        constexpr bool FIRST = decltype(first_tag)::value;
        const int gs0 = g + SPC * j;
        if (args.ablate & 8) { if (lane == 0) { tma::arrive(wempty + gs0 % NST); tma::arrive(wempty + (gs0 + 1) % NST); if (j > 0) tma::arrive(wempty + (gs0 - 1) % NST); } return; }
        wg::fence();
        {
          const int s_ = gs0 % NST;
          tma::wait(wfull + s_, (gs0 / NST) & 1);
          const unsigned char* bw = sW + s_ * 16384;
#pragma unroll
          for (int kb = 0; kb < 2; ++kb)
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              if (kb == 0 && ks == 0) wg::mma_ss_n64_first(wg::desc(sX), wg::desc(bw), acc_ab);
              else wg::mma_ss_n64(wg::desc(sX + kb * TB + ks * 32), wg::desc(bw + kb * 8192 + ks * 32), acc_ab, 1);
            }
        }
        {
          const int s_ = (gs0 + 1) % NST;
          tma::wait(wfull + s_, ((gs0 + 1) / NST) & 1);
          const unsigned char* bw = sW + s_ * 16384;
#pragma unroll
          for (int kb = 0; kb < 2; ++kb)
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              if (kb == 0 && ks == 0) wg::mma_ss_n32_first(wg::desc(sD), wg::desc(bw), acc_dh);
              else wg::mma_ss_n32(wg::desc(sD + kb * TB + ks * 32), wg::desc(bw + kb * 4096 + ks * 32), acc_dh, 1);
            }
        }
        wg::commit(); wg::wait<0>();
        if (lane == 0) {
          tma::arrive(wempty + gs0 % NST); tma::arrive(wempty + (gs0 + 1) % NST);
          if (j > 0) tma::arrive(wempty + (gs0 - 1) % NST);                   // previous chunk's [Wa;Wb]^T (its dy GEMM has completed)
        }
        // da | db (bf16, packed straight into the A fragments of the dy GEMM) and h
#pragma unroll
        for (int i = 0; i < 4; ++i) {                      // n8 block i of the 32-wide chunk: rows r0 (e 0,1) / r0 + 8 (e 2,3)
          float da[4], db[4], hv[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float a = acc_ab[4 * i + e], bb = acc_ab[16 + 4 * i + e], s_ = sigmoid(a), d = acc_dh[4 * i + e];
            da[e] = d * bb * s_ * (1.f + a * (1.f - s_));
            db[e] = d * a * s_;
            hv[e] = a * s_ * bb;
          }
          const int ks = i >> 1, hi = (i & 1) * 2;          // k-step (16 cols) and register pair within it
          pk[ks * 4 + hi] = pack(da[0], da[1]); pk[ks * 4 + hi + 1] = pack(da[2], da[3]);
          pk[8 + ks * 4 + hi] = pack(db[0], db[1]); pk[8 + ks * 4 + hi + 1] = pack(db[2], db[3]);
          hp[2 * i] = pack(hv[0], hv[1]); hp[2 * i + 1] = pack(hv[2], hv[3]);
        }
        wg::fence();
        {
          const int s_ = (gs0 + 2) % NST;
          tma::wait(wfull + s_, ((gs0 + 2) / NST) & 1);
          const unsigned char* bw = sW + s_ * 16384;
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {                 // k = [da_j (ks 0, 1) | db_j (ks 2, 3)]
            const uint32_t* p = pk + (ks >> 1) * 8 + (ks & 1) * 4;
            // A fragment: (r0, k cq..) (r0+8, k cq..) (r0, k 8+cq..) (r0+8, k 8+cq..) = blocks 2m (regs 0, 1) and 2m + 1 (regs 2, 3)
            if (FIRST && ks == 0) wg::mma_rs_n128_first(p[0], p[1], p[2], p[3], wg::desc(bw), acc_dx);
            else wg::mma_rs_n128(p[0], p[1], p[2], p[3], wg::desc(bw + ks * 32), acc_dx, 1);
          }
        }
        wg::commit();
        if (!(args.ablate & 1)) {                          // [da_j | db_j | h_j] -> staging slot j & 1 -> TMA stores (clipped at the tensor edges)
          unsigned char* stg = sStg + (j & 1) * 12288;
          if (tq == 0 && j >= 2) tma::store_wait_read<1>();
          PR(2, tq == 0);
          if (j == 0 && T >= 2) tma::wait(pfree + st, ((T >> 1) - 1) & 1);   // the row warpgroup is done with the previous tile's partials
          PR(3, tq == 0);
          asm volatile("bar.sync 3, 128;\n" ::: "memory");
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int ks = i >> 1, hi = (i & 1) * 2, o0 = off64(r0, i * 8 + cq), o1 = off64(r0 + 8, i * 8 + cq);
            *reinterpret_cast<uint32_t*>(stg + o0) = pk[ks * 4 + hi];         *reinterpret_cast<uint32_t*>(stg + o1) = pk[ks * 4 + hi + 1];
            *reinterpret_cast<uint32_t*>(stg + 4096 + o0) = pk[8 + ks * 4 + hi]; *reinterpret_cast<uint32_t*>(stg + 4096 + o1) = pk[8 + ks * 4 + hi + 1];
            *reinterpret_cast<uint32_t*>(stg + 8192 + o0) = hp[2 * i];        *reinterpret_cast<uint32_t*>(stg + 8192 + o1) = hp[2 * i + 1];
          }
          wg::proxy_fence();
          asm volatile("bar.sync 3, 128;\n" ::: "memory");
          if (tq == 0) {
            tma::store_4d(&mdab, stg, j * HC, s0, b, a0);
            tma::store_4d(&mdab, stg + 4096, N + j * HC, s0, b, a0);
            tma::store_4d(&mh, stg + 8192, j * HC, s0, b, a0);
            tma::store_commit();
          }
        }
      };
      chunk(std::true_type{}, 0);
#pragma unroll 1
      for (int j = 1; j < NCH; ++j) chunk(std::false_type{}, j);
      wg::wait<0>();
      if (lane == 0) tma::arrive(wempty + (g + SPC * NCH - 1) % NST);
      g += SPC * NCH;
      PR(4, tq == 0);
      if (tq == 0) tma::store_wait_read<0>();
      asm volatile("bar.sync 3, 128;\n" ::: "memory");
      if (tq == 0) tma::arrive(xempty + st);               // y / dffn are dead: the next tile of this set can land while the row pass runs
      // dy (fp32) -> P as [4 blocks][64 rows][32], hand the tile to its row warpgroup
      float* sF = reinterpret_cast<float*>(sStg);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int col = i * 8 + cq, kb = col >> 5, c4 = (col & 31) >> 2, oo = col & 3;
        *reinterpret_cast<float2*>(sF + kb * XTF + wg::off32(r0, c4) + oo) = make_float2(acc_dx[i * 4 + 0], acc_dx[i * 4 + 1]);
        *reinterpret_cast<float2*>(sF + kb * XTF + wg::off32(r0 + 8, c4) + oo) = make_float2(acc_dx[i * 4 + 2], acc_dx[i * 4 + 3]);
      }
      asm volatile("bar.sync 3, 128;\n" ::: "memory");
      PR(5, tq == 0);
      if (tq == 0) tma::arrive(cfull + st);
    }
    return;
  }
  // ================= row warpgroups: WG1 -> buffer set 0 (tiles T = 0, 2, ..), WG2 -> set 1 =================
  // Warp w owns tile rows 16w..16w+15, lane owns channels 4*lane..; the row data the final pass needs (dq2, q1 as packed bf16, rstd)
  // is kept in registers from pass A, so the final pass after the GEMMs only reads dy from smem and scale_f.
  asm volatile("setmaxnreg.inc.sync.aligned.u32 " ROW_REGS ";\n" ::: "memory");
  const int rs = wgi - 1, bar_id = 1 + rs;
  unsigned char* sX = smem + SSET + rs * SET;
  unsigned char* sD = sX + 2 * TB;
  float* sF = reinterpret_cast<float*>(sX);                // dy (fp32) after the GEMMs
  float* sP = reinterpret_cast<float*>(sX + 4 * TB);       // partials
  const int kb_l = lane >> 3, c4_l = lane & 7, c0 = lane * 4;
  const long MODW = 6 * C;
  const int c4r = tq & 31, agr = tq >> 5;
  for (int T = rs; T < ntT; T += 2) {
    int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
    auto row = [&](int rl, bool& ok, long& mrow) {
      const long a = a0 + rl / AT, s = s0 + rl % AT;
      ok = a < A && s < S;
      mrow = (long)b * S + (s < S ? s : 0);
      return ok ? (a * B + b) * S + s : 0;
    };
    auto reduce = [&](const float* buf, int col0) {        // per-atom sums over the tile's SP augments, one red.add per (atom, 4 channels)
#pragma unroll 1
      for (int al = agr; al < AT; al += 4) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int sl = 0; sl < SP; ++sl) {
          const float4 v = *reinterpret_cast<const float4*>(buf + (c4r >> 3) * XTF + wg::off32(sl * AT + al, c4r & 7));
          acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        if (s0 + al < S) red_add_v4(args.dmod + ((long)b * S + s0 + al) * MODW + col0 + c4r * 4, acc);
      }
    };
    PR(6, tq == 0);
    tma::wait(cfull + rs, (T >> 1) & 1);                   // dy (fp32) in P
    PR(7, tq == 0);
    if (args.ablate & 2) { asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory"); if (tq == 0) tma::arrive(pfree + rs); continue; }
    reduce(sP, 3 * C);                                     // d shift_f = sum dy (padding rows are exact zeros)
    asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory");
    PR(12, tq == 0);
    // ---- final pass: dq1; P <- d scale_f partials ----
#pragma unroll
    for (int i0 = 0; i0 < 16; i0 += FIN_ILP) {
      float4 scv[FIN_ILP]; uint2 q1r[FIN_ILP], dqr[FIN_ILP];
#pragma unroll
      for (int j = 0; j < FIN_ILP; ++j) {
        bool ok; long mrow; const long R = row(warp * 16 + i0 + j, ok, mrow);
        scv[j] = (args.ablate & 32) ? make_float4(0.1f, 0.1f, 0.1f, 0.1f) : __ldg(reinterpret_cast<const float4*>(args.mod + mrow * MODW + 4 * C + c0));
        q1r[j] = (ok && !(args.ablate & 64)) ? __ldg(reinterpret_cast<const uint2*>(args.q1 + R * C + c0)) : make_uint2(0x3f803f80u, 0x3f803f80u);
        dqr[j] = ok ? __ldg(reinterpret_cast<const uint2*>(args.dq2 + R * C + c0)) : make_uint2(0u, 0u);
      }
#pragma unroll
      for (int j = 0; j < FIN_ILP; ++j) {
        const int rl = warp * 16 + i0 + j;
        bool ok; long mrow; const long R = row(rl, ok, mrow);
        const int so = kb_l * XTF + wg::off32(rl, c4_l);
        const float4 dn = *reinterpret_cast<const float4*>(sP + so);
        const float4 q = ld4(reinterpret_cast<const __nv_bfloat16*>(&q1r[j]));
        float ss = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
#pragma unroll
        if (!(args.ablate & 16))
          for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
        const float r_ = rsqrtf(ss * (1.f / C) + args.eps);
        const float4 xh = make_float4(q.x * r_, q.y * r_, q.z * r_, q.w * r_);
        const float4 dxh = make_float4(dn.x * (1.f + scv[j].x), dn.y * (1.f + scv[j].y), dn.z * (1.f + scv[j].z), dn.w * (1.f + scv[j].w));
        float s2 = dxh.x * xh.x + dxh.y * xh.y + dxh.z * xh.z + dxh.w * xh.w;
#pragma unroll
        if (!(args.ablate & 16))
          for (int o2 = 16; o2 > 0; o2 >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o2);
        s2 *= (1.f / C);
        if (ok) {
          const float4 d = ld4(reinterpret_cast<const __nv_bfloat16*>(&dqr[j]));
          *reinterpret_cast<uint2*>(args.dq1 + R * C + c0) =
              pk4(make_float4(d.x + r_ * (dxh.x - xh.x * s2), d.y + r_ * (dxh.y - xh.y * s2), d.z + r_ * (dxh.z - xh.z * s2), d.w + r_ * (dxh.w - xh.w * s2)));
        }
        *reinterpret_cast<float4*>(sP + so) = make_float4(dn.x * xh.x, dn.y * xh.y, dn.z * xh.z, dn.w * xh.w);   // xh = 0 on padding rows
      }
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory");
    PR(13, tq == 0);
    reduce(sP, 4 * C);
    asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory");
    PR(14, tq == 0);
#pragma unroll
    for (int i0 = 0; i0 < 16; i0 += 4) {                  // d gate_f partial = dq2 * ffn (L2-hot re-reads) -> P
      uint2 dv[4], fv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        bool ok; long mrow; const long R = row(warp * 16 + i0 + j, ok, mrow);
        dv[j] = ok ? __ldg(reinterpret_cast<const uint2*>(args.dq2 + R * C + c0)) : make_uint2(0u, 0u);
        fv[j] = ok ? __ldg(reinterpret_cast<const uint2*>(args.ffn + R * C + c0)) : make_uint2(0u, 0u);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 d = ld4(reinterpret_cast<const __nv_bfloat16*>(&dv[j])), f = ld4(reinterpret_cast<const __nv_bfloat16*>(&fv[j]));
        *reinterpret_cast<float4*>(sP + kb_l * XTF + wg::off32(warp * 16 + i0 + j, c4_l)) = make_float4(d.x * f.x, d.y * f.y, d.z * f.z, d.w * f.w);
      }
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory");
    PR(15, tq == 0);
    reduce(sP, 5 * C);
    asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory");
    PR(8, tq == 0);
    if (tq == 0) tma::arrive(pfree + rs);                  // P -> the GEMM warpgroup (staging of this set's next tile)
  }
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
CUtensorMap map4d(const torch::Tensor& t, long A, long B, long S, uint32_t at, uint32_t sp, long cols = C, uint32_t bcols = 64,
                  CUtensorMapSwizzle sw = CU_TENSOR_MAP_SWIZZLE_128B) {   // [A][B][S][cols] bf16 -> box (bcols, at atoms, 1, sp augments)
  alignas(64) CUtensorMap m{};
  uint64_t gdim[4] = {(uint64_t)cols, (uint64_t)S, (uint64_t)B, (uint64_t)A};
  uint64_t gstride[3] = {(uint64_t)cols * 2, (uint64_t)S * cols * 2, (uint64_t)B * S * cols * 2};
  uint32_t bdim[4] = {bcols, at, 1, sp}; uint32_t estride[4] = {1, 1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(4d) failed: ", (int)r);
  return m;
}
torch::Tensor g_prof;
int num_sms(int dev) { static int s = 0; if (s == 0) cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }

template <int SP>
void launch(Args args, const torch::Tensor& y, const torch::Tensor& dq2, const CUtensorMap& mwab, const CUtensorMap& mwdt, const CUtensorMap& mwabt,
            const torch::Tensor& dab, const torch::Tensor& h) {
  constexpr int AT = BR / SP;
  CUtensorMap my = map4d(y, args.A, args.B, args.S, AT, SP), mdq = map4d(dq2, args.A, args.B, args.S, AT, SP);
  CUtensorMap mdab = map4d(dab, args.A, args.B, args.S, AT, SP, 2 * N, HC, CU_TENSOR_MAP_SWIZZLE_64B);
  CUtensorMap mh = map4d(h, args.A, args.B, args.S, AT, SP, N, HC, CU_TENSOR_MAP_SWIZZLE_64B);
  args.nab = (args.S + AT - 1) / AT; args.nag = (args.A + SP - 1) / SP;
  args.ntile = args.nab * args.nag * args.B;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(ffn_bwd_kernel<SP>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES)); attr = true; }
  const int grid = std::min(args.ntile, num_sms(y.device().index()));
  ffn_bwd_kernel<SP><<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(args, my, mdq, mwab, mwdt, mwabt, mdab, mh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// dq2, q1, y, ffn: [M, 128] bf16 with rows ((a*B + b)*S + s); mod [B*S, 768] fp32; dmod [B*S, 768] fp32 (accumulated: shift_f / scale_f /
// gate_f columns).  wab [512, 128] = per chunk j rows [Wu[32j..32j+31] ; Wu[256+32j..]]; wdt = Wd^T [256, 128]; wabt = wab^T [128, 512].
// Returns [dq1, dffn [M, 128], h [M, 256], dab [M, 512]] (bf16; dab = [da | db] in Wu's row order).
std::vector<torch::Tensor> ffn_bwd(torch::Tensor dq2, torch::Tensor q1, torch::Tensor y, torch::Tensor ffn, torch::Tensor mod, torch::Tensor dmod,
                                   torch::Tensor wab, torch::Tensor wdt, torch::Tensor wabt, int64_t A, int64_t B, int64_t S, double eps, int64_t sp) {
  const long M = A * B * S;
  for (const auto* t : {&dq2, &q1, &y, &ffn}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == M * C, "rows: [M, 128] bf16");
  TORCH_CHECK(mod.scalar_type() == torch::kFloat && mod.is_contiguous() && mod.numel() == B * S * 6 * C, "mod [B*S, 768] fp32");
  TORCH_CHECK(dmod.scalar_type() == torch::kFloat && dmod.is_contiguous() && dmod.numel() == B * S * 6 * C, "dmod [B*S, 768] fp32");
  TORCH_CHECK(wab.is_contiguous() && wab.scalar_type() == torch::kBFloat16 && wab.sizes() == torch::IntArrayRef({2 * N, C}), "wab [512, 128] bf16");
  TORCH_CHECK(wdt.is_contiguous() && wdt.scalar_type() == torch::kBFloat16 && wdt.sizes() == torch::IntArrayRef({N, C}), "wdt [256, 128] bf16");
  TORCH_CHECK(wabt.is_contiguous() && wabt.scalar_type() == torch::kBFloat16 && wabt.sizes() == torch::IntArrayRef({C, 2 * N}), "wabt [128, 512] bf16");
  auto opt = dq2.options();
  auto dq1 = torch::empty({M, C}, opt), dffn = torch::empty({M, C}, opt), h = torch::empty({M, N}, opt), dab = torch::empty({M, 2 * N}, opt);
  CUtensorMap mwab = map2d(wab, 2 * N, C, 64), mwdt = map2d(wdt, N, C, HC), mwabt = map2d(wabt, C, 2 * N, C);
  Args args;
  args.dq2 = reinterpret_cast<const __nv_bfloat16*>(dq2.data_ptr()); args.q1 = reinterpret_cast<const __nv_bfloat16*>(q1.data_ptr());
  args.ffn = reinterpret_cast<const __nv_bfloat16*>(ffn.data_ptr()); args.mod = mod.data_ptr<float>();
  args.dq1 = reinterpret_cast<__nv_bfloat16*>(dq1.data_ptr()); args.dffn = reinterpret_cast<__nv_bfloat16*>(dffn.data_ptr());
  args.h = reinterpret_cast<__nv_bfloat16*>(h.data_ptr()); args.dab = reinterpret_cast<__nv_bfloat16*>(dab.data_ptr());
  args.dmod = dmod.data_ptr<float>();
  args.S = (int)S; args.A = (int)A; args.B = (int)B; args.eps = (float)eps;
  { const char* ab = std::getenv("SWA_FFN_ABL"); args.ablate = ab != nullptr ? std::atoi(ab) : 0; }
  args.prof = nullptr;
  { const char* pe = std::getenv("SWA_FFN_PROF"); if (pe != nullptr && pe[0] == '1') { g_prof = torch::zeros({16}, dq2.options().dtype(torch::kLong)); args.prof = reinterpret_cast<unsigned long long*>(g_prof.data_ptr<int64_t>()); } }
  if (sp == 0) sp = A % 4 == 0 ? 4 : A % 2 == 0 ? 2 : 1;
  if (sp == 8) launch<8>(args, y, dq2, mwab, mwdt, mwabt, dab, h);
  else if (sp == 4) launch<4>(args, y, dq2, mwab, mwdt, mwabt, dab, h);
  else if (sp == 2) launch<2>(args, y, dq2, mwab, mwdt, mwabt, dab, h);
  else launch<1>(args, y, dq2, mwab, mwdt, mwabt, dab, h);
  return {dq1, dffn, h, dab};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ffn_bwd", &ffn_bwd, "ESMFold2 SWA atom block FFN backward (bf16 wgmma)");
  m.def("prof", []() { return g_prof; });
}
