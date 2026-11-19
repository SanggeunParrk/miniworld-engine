// k1h.cu -- the TriMul front (k1w.cu) on M = 64 tcgen05 tiles, for small L inference (D64 / D128, [P, M] planes):
//   xn       = bf16(fma((x - mean) * rstd, g, b))      plane[c] = bf16(sigmoid(xn Wg_c^T) * (xn Wp_c^T))      (masked pairs: 0)
// Why (2026-11-17, round 3 sixth pass): at L128 / L256 k1w runs one 128-token tile per CTA and its time is that tile's dependency
// chain (load -> LayerNorm on one thread per row -> MMA -> epilogue on 32 rows per warp -> store). Here a tile is 64 tokens
// (tcgen05.mma cta_group::1 M = 64): twice the CTAs, two per SM (<= 113 KB shared memory, 256 TMEM columns, 384 threads each),
// the LayerNorm of a row on two threads (half the channels each, statistics combined by one shuffle) and the epilogue on 16 rows per
// warp, so every stage of the chain is about half as long.
//
// Layouts. x tile [64 tok][D] as D / 64 SW128 K-blocks of [64][64] (8 KB); the transform normalises it IN PLACE (xn is the SS A
// operand: no TMEM A at M = 64). Weights: k1w's [128 rows][64 k] slots (64 gate rows | 64 projection rows of a 64-channel chunk).
// Accumulator of a chunk: M = 64 data-path layout (row 16 w + i in TMEM lane 32 w + i, i < 16), columns [0, 64) gate, [64, 128)
// projection; read with tcgen05.ld.16x256b (the k1w epilogue's fragment layout, one 16-lane half). Output: per warp [64 ch][16 tok]
// boxes (SW32), TMA store.
//
// 384 threads: warp 0 TMA x (+ every weight block at entry), warp 1 MMA, warp 2 TMEM, warp 3 TMA weights (streamed);
// warps 4-7 transform (thread pair = token row), then (one tile per CTA) the epilogue of the odd chunks; warps 8-11 epilogue.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;
#ifdef TM_TL
static __device__ unsigned long long g_tlh[320][16];
#define TLH(slot) { g_tlh[blockIdx.x][slot] = clock64(); }
#else
#define TLH(slot) {}
#endif
#ifdef PH_TL   // per-CTA globaltimer phase marks (ns): [entry, griddepcontrol.wait returned, end]; read with ph_k1h()
static __device__ unsigned long long g_ph[320][3];
#define PHM(slot) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_ph[blockIdx.x][slot] = t_; }
#else
#define PHM(slot) {}
#endif

namespace k1h {

constexpr int TOK = 64, XB = TOK * 64 * 2, WB = 128 * 64 * 2;   // x K-block [64][64] (8 KB); weight K-block [128][64] (16 KB)
constexpr int OSTW = 64 * 16 * 2;                                // per-warp output box [64 ch][16 tok] = 2 KB
constexpr int NTHREADS = 384;
constexpr uint32_t IDESC = idesc_bf16(64, 128);

template <int D>
struct Cfg {
  static constexpr int KB = D / 64;
  static constexpr int NXB = D == 64 ? 2 : 1;   // x tiles (each D / 64 K-blocks)
  static constexpr int NWS = 4;                 // weight ring
  static constexpr int O_W = 0, O_X = O_W + NWS * WB, O_OUT = O_X + NXB * KB * XB, O_GB = O_OUT + 8 * OSTW, O_BAR = O_GB + 2 * D * 4;
  static constexpr int SMEM = O_BAR + 256 + 1024;
  static_assert(SMEM + 1024 <= 233472 / 2, "two CTAs per SM");
};

struct Bars {
  uint64_t w_full[4], w_empty[4], x_full[2], x_empty[2], xn_full[2], acc_full[2], acc_empty[2];
  uint32_t tmem;
};

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float2 sigmoid2(float2 g) {   // k1w's 0.5 + 0.5 tanh.approx(g / 2)
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  return __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
}
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV void stmatrix_x4_trans(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

template <int D, int P>
__global__ void __launch_bounds__(NTHREADS, 2)
    k1h_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw0, const __grid_constant__ CUtensorMap mw1,
               const __grid_constant__ CUtensorMap mw2, const __grid_constant__ CUtensorMap mw3, const __grid_constant__ CUtensorMap mplane,
               const uint8_t* __restrict__ tokmask, int L, const float* __restrict__ gamma, const float* __restrict__ beta, int tiles,
               float eps, float rf3) {
  using G_ = Cfg<D>;
  constexpr int KB = G_::KB, NXB = G_::NXB, NWS = G_::NWS, NCH = P / 64;
  constexpr bool RES = NCH * KB <= NWS;   // every weight block resident: loaded once at entry
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;
  const bool lat = tiles <= G;   // one tile per CTA: warps 4-7 take the odd chunks' epilogue after the transform
  float rf3s = 1.f;   // RoseTTAFold3 length factor (rf3 >= 0): the constant, or bf16(1 / max(n, 1)) from the token mask counted here
  if (rf3 >= 0.f) {
    if (rf3 > 0.f || tokmask == nullptr) rf3s = rf3;
    else {
      int n = 0;
      for (int i0 = 0; i0 < L; i0 += NTHREADS) n += __syncthreads_count(i0 + tid < L && tokmask[i0 + tid] != 0);
      rf3s = __bfloat162float(__float2bfloat16_rn(1.0f / (float)max(n, 1)));
    }
  }
  const CUtensorMap* mw[4] = {&mw0, &mw1, &mw2, &mw3};
  auto wload = [&](int slot, int q) {   // weight block q = chunk c, K-block kb -> slot
    const int row = (q / KB) * 64, kb = q % KB, side = row < P / 2 ? 0 : 1, r = side ? row - P / 2 : row;
    mbar_expect_tx(&B.w_full[slot], WB);
    tma_load_2d(sm + G_::O_W + slot * WB, mw[2 * side + 1], &B.w_full[slot], kb * 64, r, EVICT_LAST);            // gate rows
    tma_load_2d(sm + G_::O_W + slot * WB + WB / 2, mw[2 * side], &B.w_full[slot], kb * 64, r, EVICT_LAST);       // projection rows
  };
  auto xload = [&](int b, int i) {
    mbar_expect_tx(&B.x_full[b], KB * XB);
    for (int kb = 0; kb < KB; ++kb) tma_load_2d(sm + G_::O_X + (b * KB + kb) * XB, &mx, &B.x_full[b], kb * 64, (cta + i * G) * TOK, EVICT_NORMAL);
  };
  const int nx0 = min(n_local, NXB), nw0 = RES ? (n_local > 0 ? NCH * KB : 0) : min(n_local * NCH * KB, NWS);
  pdl_trigger();   // the contraction (a programmatic dependent) may schedule its CTAs as these retire
  if (tid == 0) TLH(0)
  if (tid == 0) PHM(0)
  if (tid == 0) {
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], 1); }
    for (int b = 0; b < NXB; ++b) { mbar_init(&B.x_full[b], 1); mbar_init(&B.x_empty[b], 1); mbar_init(&B.xn_full[b], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    fence_mbar_init();
    prefetch_tmap(&mx); prefetch_tmap(&mplane);
    pdl_wait();   // nothing global is read before this
    PHM(1)
    for (int b = 0; b < nx0; ++b) xload(b, b);
    for (int q = 0; q < nw0; ++q) wload(q, q);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 256); tmem_relinquish(); }
  if (tid != 0) pdl_wait();
  for (int k = tid; k < D; k += NTHREADS) {
    reinterpret_cast<float*>(sm + G_::O_GB)[k] = gamma[k];
    reinterpret_cast<float*>(sm + G_::O_GB)[D + k] = beta[k];
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TLH(1)

  if (warp < 4) {
    if (warp == 0 && lane == 0) {
      for (int i = nx0; i < n_local; ++i) {   // x tiles past the first fills: slot free once the tile's products are done
        const int b = i % NXB;
        mbar_wait(&B.x_empty[b], ((i / NXB) - 1) & 1);
        xload(b, i);
      }
    } else if (warp == 3 && lane == 0 && !RES) {
      int s = nw0 % NWS, q = nw0;
      uint32_t ph = (nw0 / NWS) & 1;
      for (; q < n_local * NCH * KB; ++q) {
        mbar_wait(&B.w_empty[s], ph ^ 1);
        wload(s, q % (NCH * KB));
        if (++s == NWS) { s = 0; ph ^= 1; }
      }
    } else if (warp == 1) {
      const bool ldr = elect_one();
      int s = 0, a = 0;
      uint32_t ph = 0, pa = 0;
      for (int i = 0; i < n_local; ++i) {
        const int b = i % NXB;
        mbar_wait(&B.xn_full[b], (i / NXB) & 1);
        tc_fence_after();
        const uint32_t xa = su + G_::O_X + b * KB * XB;
#pragma unroll 1
        for (int c = 0; c < NCH; ++c) {
          mbar_wait(&B.acc_empty[a], pa ^ 1);
          tc_fence_after();
          const uint32_t d = tmem + a * 128;
#pragma unroll 1
          for (int kb = 0; kb < KB; ++kb) {
            const int slot = RES ? c * KB + kb : s;
            if (!RES || i == 0) mbar_wait(&B.w_full[slot], RES ? 0 : ph);
            tc_fence_after();
            if (ldr) {
              const uint32_t wa = su + G_::O_W + slot * WB, xk = xa + kb * XB;
#pragma unroll
              for (int k = 0; k < 4; ++k) umma_ss(d, desc_k_sw128(xk + k * 32), desc_k_sw128(wa + k * 32), IDESC, (kb | k) != 0);
              if (!RES) umma_commit(&B.w_empty[s]);
            }
            __syncwarp();
            if (!RES && ++s == NWS) { s = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[a]);
          __syncwarp();
          if (++a == 2) { a = 0; pa ^= 1; }
        }
        if (ldr) umma_commit(&B.x_empty[b]);
        __syncwarp();
        if (ldr && i == 0) TLH(5)
      }
    }
  } else {
    auto epi = [&](int grp, int ngrp) {
      // ---------------------------------------------------------------- epilogue: warp w % 4 = rows 16 (w % 4) .. + 15 of the tile
      const int q = warp & 3;
      const uint32_t stg = su + G_::O_OUT + (warp - 4) * OSTW;
      const int k = lane >> 3, jr = lane & 7;
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + i * G) * TOK;
        for (int c = grp; c < NCH; c += ngrp) {
          const int ci = i * NCH + c, acc = ci & 1;
          mbar_wait(&B.acc_full[acc], (ci >> 1) & 1);
          tc_fence_after();
          if (lane == 0 && i == 0 && c == grp && q == 0) TLH(6 + grp)
          if (lane == 0) bulk_wait_read<0>();   // this warp's previous box has left the staging buffer
          __syncwarp();
          const uint32_t ta = tmem + acc * 128 + ((uint32_t)(q * 32) << 16);
          uint32_t gv[16], pv[16];
          auto work = [&](int cq, const uint32_t (&g)[16], const uint32_t (&p)[16]) {
            uint32_t m[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float2 s0 = sigmoid2(make_float2(__uint_as_float(g[4 * j]), __uint_as_float(g[4 * j + 1])));
              const float2 s1 = sigmoid2(make_float2(__uint_as_float(g[4 * j + 2]), __uint_as_float(g[4 * j + 3])));
              float2 pq0 = make_float2(__uint_as_float(p[4 * j]), __uint_as_float(p[4 * j + 1])), pq1 = make_float2(__uint_as_float(p[4 * j + 2]), __uint_as_float(p[4 * j + 3]));
              if (rf3 >= 0.f && c < NCH / 2) { pq0 = __fmul2_rn(pq0, make_float2(rf3s, rf3s)); pq1 = __fmul2_rn(pq1, make_float2(rf3s, rf3s)); }   // RoseTTAFold3
              const float2 a0 = __fmul2_rn(s0, pq0);
              const float2 a1 = __fmul2_rn(s1, pq1);
              m[2 * j] = pack_bf16(a0.x, a0.y);        // token tr, channels 8 j + 2 (t % 4) + {0, 1}
              m[2 * j + 1] = pack_bf16(a1.x, a1.y);    // token tr + 8
            }
#pragma unroll
            for (int x = 0; x < 2; ++x) {
              const int chl = cq * 32 + (2 * x + (k >> 1)) * 8 + jr, qc = k & 1;   // [64 ch][16 tok] rows of 32 B, SW32
              stmatrix_x4_trans(stg + chl * 32 + ((qc ^ ((chl >> 2) & 1)) << 4), m[4 * x], m[4 * x + 1], m[4 * x + 2], m[4 * x + 3]);
            }
          };
          tmem_ld16x256_x4(ta, gv);
          tmem_ld16x256_x4(ta + 64, pv);
          tmem_wait_ld();
          work(0, gv, pv);
          tmem_ld16x256_x4(ta + 32, gv);
          tmem_ld16x256_x4(ta + 96, pv);
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.acc_empty[acc]);
          work(1, gv, pv);
          fence_async_smem();
          __syncwarp();
          if (lane == 0) {
            tma_store_2d(&mplane, sm + G_::O_OUT + (warp - 4) * OSTW, row0 + q * 16, c * 64);
            bulk_commit();
          }
        }
      }
      if (lane == 0) bulk_wait<0>();
      if (lane == 0 && q == 0) TLH(8 + grp)
    };
    if (warp < 8) {
      // ---------------------------------------------------------------- transform: thread pair = token row r, half h of the channels
      const int t = tid - 128, r = t >> 1, h = t & 1;
      auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
      constexpr int NQ = D / 16;   // 16-byte chunks per half row
      for (int i = 0; i < n_local; ++i) {
        const int b = i % NXB, row = (cta + i * G) * TOK + r;
        const int ll = L * L, sb = row / ll, rin = row - sb * ll;
        const uint32_t kmask = (tokmask == nullptr || (tokmask[sb * L + rin / L] && tokmask[sb * L + rin % L])) ? 0xffffffffu : 0u;
        mbar_wait(&B.x_full[b], (i / NXB) & 1);
        if (t == 0 && i == 0) TLH(2)
        const uint32_t xt = su + G_::O_X + b * KB * XB;
        auto adr = [&](int k) { const int qq = h * NQ + k; return xt + (qq >> 3) * XB + sw128(r, qq & 7); };
        uint4 v[NQ];
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
          v[k] = lds128(adr(k));
          a0 = __fadd2_rn(a0, up(v[k].x)); a1 = __fadd2_rn(a1, up(v[k].y)); a2 = __fadd2_rn(a2, up(v[k].z)); a3 = __fadd2_rn(a3, up(v[k].w));
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        float s = __fadd_rn(a0.x, a0.y);
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
        const float mu = __fmul_rn(s, 1.f / D);
        const float2 nm = make_float2(-mu, -mu);
        a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
          float2 dd;
          dd = __fadd2_rn(up(v[k].x), nm); a0 = __ffma2_rn(dd, dd, a0);
          dd = __fadd2_rn(up(v[k].y), nm); a1 = __ffma2_rn(dd, dd, a1);
          dd = __fadd2_rn(up(v[k].z), nm); a2 = __ffma2_rn(dd, dd, a2);
          dd = __fadd2_rn(up(v[k].w), nm); a3 = __ffma2_rn(dd, dd, a3);
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        float ss = __fadd_rn(a0.x, a0.y);
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
        const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(ss, 1.f / D), eps));
        const float2 rr = make_float2(rs, rs);
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
          const int col = (h * NQ + k) * 8;
          const float4 ga = lds128f(su + G_::O_GB + col * 4), gb4 = lds128f(su + G_::O_GB + col * 4 + 16);
          const float4 ba = lds128f(su + G_::O_GB + D * 4 + col * 4), bb = lds128f(su + G_::O_GB + D * 4 + col * 4 + 16);
          auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
            const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
            return pack_bf16(y.x, y.y) & kmask;
          };
          sts128(adr(k), aff2(v[k].x, ga.x, ga.y, ba.x, ba.y), aff2(v[k].y, ga.z, ga.w, ba.z, ba.w), aff2(v[k].z, gb4.x, gb4.y, bb.x, bb.y),
                 aff2(v[k].w, gb4.z, gb4.w, bb.z, bb.w));
        }
        fence_async_smem();   // xn (generic stores) -> the MMA's operand reads (async proxy)
        named_bar_sync(1, 128);
        if (t == 0) { mbar_arrive(&B.xn_full[b]); if (i == 0) TLH(3) }
      }
      if (lat) epi(1, 2);
    } else {
      epi(0, lat ? 2 : 1);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TLH(12)
  if (tid == 0) PHM(2)
  if (warp == 2) tmem_dealloc(tmem, 256);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k1h

// x [M, D] bf16 (M = B L L, tokens b-major, M % 64 == 0); wl, wlg, wr, wrg [P/2, D] bf16 row-major; tokmask [B L] bool / uint8 or
// None; gamma / beta [D] fp32; planes [P, (B,) L, L] bf16. grid 0: min(tiles, 2 SMs). Launched as a programmatic dependent.
static void k1h_forward_impl(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                 c10::optional<torch::Tensor> tokmask, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps, double rf3) {
  using namespace k1h;
  const int64_t M = x.size(0), D = x.size(1), P = planes.size(0);
  TORCH_CHECK(M % TOK == 0 && planes.numel() == P * M && x.is_contiguous() && planes.is_contiguous() && planes.dim() >= 3);
  const int L = (int)planes.size(planes.dim() - 1);
  const int64_t nb = M / ((int64_t)L * L);
  TORCH_CHECK(nb * L * L == M && planes.size(planes.dim() - 2) == L);
  TORCH_CHECK(!tokmask.has_value() || (tokmask->numel() == nb * L && tokmask->element_size() == 1 && tokmask->is_contiguous()));
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap mw[4];
  const torch::Tensor* ws[4] = {&wl, &wlg, &wr, &wrg};
  for (int k = 0; k < 4; ++k) {
    const torch::Tensor& w = *ws[k];
    TORCH_CHECK(w.size(0) == P / 2 && w.size(1) == D && w.stride(1) == 1 && (w.stride(0) * 2) % 16 == 0,
                "k1h_forward: front weights must be [P/2, D] with unit column stride");
    mw[k] = tmap::make(w.data_ptr(), bf, {(uint64_t)D, (uint64_t)(P / 2)}, {(uint64_t)w.stride(0) * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  auto mp = tmap::make(planes.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {16u, 64}, CU_TENSOR_MAP_SWIZZLE_32B);
  const int tiles = (int)(M / TOK);
  int g = grid > 0 ? (int)grid : 2 * num_sms();
  g = std::min(g, tiles);
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* mk = tokmask.has_value() ? reinterpret_cast<const uint8_t*>(tokmask->data_ptr()) : nullptr;
  const float *ga = gamma.data_ptr<float>(), *be = beta.data_ptr<float>();
#define K1H_CASE(DD, PP)                                                                                                         \
  if (D == DD && P == PP) {                                                                                                      \
    static bool attr = false;                                                                                                    \
    if (!attr) { cudaFuncSetAttribute(k1h_kernel<DD, PP>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<DD>::SMEM); attr = true; } \
    tmap::launch_pdl(k1h_kernel<DD, PP>, dim3(g), dim3(NTHREADS), Cfg<DD>::SMEM, st, true, mx, mw[0], mw[1], mw[2], mw[3], mp, mk, L, ga, be, \
                     tiles, (float)eps, (float)rf3);                                                                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                              \
    return;                                                                                                                      \
  }
  K1H_CASE(64, 128) K1H_CASE(64, 256) K1H_CASE(128, 256) K1H_CASE(128, 512)
#undef K1H_CASE
  TORCH_CHECK(false, "k1h_forward: unsupported (D, P) = (", D, ", ", P, ")");
}

torch::Tensor ph_k1h() {
#ifdef PH_TL
  torch::Tensor h = torch::empty({320, 3}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_ph, sizeof(g_ph));
  return h;
#else
  return torch::empty({0});
#endif
}

torch::Tensor k1h_tl() {
#ifdef TM_TL
  torch::Tensor h = torch::empty({320, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tlh, sizeof(unsigned long long) * 320 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}

// rf3 < 0: off; > 0 RoseTTAFold3's constant length factor on the left projection; == 0 f = bf16(1 / max(n, 1)), n the kept tokens of tokmask [L]
void k1h_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, c10::optional<torch::Tensor> tokmask,
                 torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps) {
  k1h_forward_impl(x, wl, wlg, wr, wrg, tokmask, gamma, beta, planes, grid, eps, -1.0);
}
void k1h_forward_rf3(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, c10::optional<torch::Tensor> tokmask,
                     torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps, double rf3) {
  k1h_forward_impl(x, wl, wlg, wr, wrg, tokmask, gamma, beta, planes, grid, eps, rf3);
}
