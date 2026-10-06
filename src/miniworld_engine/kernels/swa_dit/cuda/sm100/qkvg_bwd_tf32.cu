// qkvg_bwd_tf32.cu — the fp32 SWA atom block's first-stage backward on sm_100a, TF32 tensor cores (the fp32 counterpart of qkvg_bwd.cu;
// the outputs of the Triton _swa_qkvg_bwd_fp32_kernel, every one fp32, from fp32 dQ / dK / dV):
//   dpq = headRMS_bwd(pq, unrope(dQ));  dpk likewise;  dP = [dpq | dpk | dV | dG]  ([M, 512], the cuBLAS dWqkv | dWg operand)
//   dx = dP [Wq; Wk; Wv; Wg];  xh = q rstd;  dxh = dx (1 + scale_a);  dq = dq1 + rstd (dxh - xh mean(dxh xh))
//   d shift_a += sum_aug dx,  d scale_a += sum_aug dx xh
// dpq / dpk leave rounded to TF32 (cvt.rna; dV and dG arrive rounded from attn_dkv_tf32.cu / oproj_bwd_tf32.cu): dP is this kernel's MMA
// operand and the TF32 weight gradient's, which round it the same way. dq is plain fp32.
// Transposed (lane = input channel): dx^T [128 in][128 rows] = W^T dP^T, M = N = 128, K = 512 in 16 slices of 32, kind::tf32:
//   A = the W^T slice [128 in][32 out] (W^T = [Wqkv; Wg]^T rounded on the host, 256 KB: streamed from L2 through a weight ring),
//   B = the dP slice [128 rows][32] as it sits in the activation ring (K-major, 128-B swizzled): dpq_h / dpk_h written in place over the
//       pq / pk head slices by warpgroup 0, dV_h (head-major, 5-D map) and dG slices exactly as loaded. Every dP slice also leaves from
//       its slot by a TMA store (no copy).
// Warp roles: 0 activation producer (per tile, in order: {pq_h, dQ_h, pk_h, dK_h} h = 0..3, dV_0..3, dG_0..3), 3 weight producer (the 16
// W^T slices in MMA order), 1 MMA issuer (slices dpq_0, dpk_0, ..., dpq_3, dpk_3, dV_0..3, dG_0..3), 2 dP stores + TMEM allocation,
// 4-7 warpgroup 0: one row per thread, the eight (q / k, head) jobs: RoPE transpose + head-RMS backward on 32 values (as qkvg_bwd.cu's
// q / k phase); 8-11 warpgroup 1: the RMS-adaLN backward (one channel per thread, coalesced q / dq1 row loads; ffn_bwd_tf32.cu's rms_bwd),
// one tile behind.
// Shared memory: activation ring 8 x 16 KB | weight ring 4 x 16 KB | exchange 640 B | barriers  (193 KB).  TMEM: dx^T[2] at 0 / 128.
// Registers: <= 128 / thread (launch bound 512, launched with 384 threads).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include <type_traits>
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (as augmented_attention/cuda/sm100/sm100.cuh, measured on B200)
namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI uint32_t rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
}  // namespace tf

constexpr int C = 128, D = 32, MODW = 6 * C, NS = 8, NW = 4, UPT = 24, WPT = 16;
constexpr int SL = 128 * 128;                                              // a [128][32] fp32 slice (16 KB)
constexpr int O_A = 0, O_W = NS * SL, O_EX = O_W + NW * SL, O_BAR = O_EX + 640, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t I_DX = tf::idesc(128, 128);
enum { U_PQ = 0, U_DQH = 1, U_PK = 2, U_DKH = 3, U_DV = 16, U_DG = 20 };   // activation uses per tile: 4 h + {PQ, DQH, PK, DKH}, then dV, dG

struct Bars {
  uint64_t afull[NS], aempty[NS], ardy[NS], wfull[NW], wempty[NW], dxfull[2], dxfree[2];
  uint32_t tmem;
};
DEVI void mbar_arrive_cnt(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory"); }
DEVI void tma_load_5d(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}
DEVI void red1(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
DEVI float pick8(const float (&a)[8], int k) {                           // a[k] for a runtime k without indexing registers
  float v = a[0];
#pragma unroll
  for (int e = 1; e < 8; ++e) v = k == e ? a[e] : v;
  return v;
}
DEVI float ldf(const float* p) { return __ldg(p); }
DEVI float rscatter32(float (&v)[32], int lane) {                          // v[j] in every lane -> lane l: sum over the warp of v[l]
#pragma unroll
  for (int i = 0; i < 16; ++i) { const bool up = lane & 16; const float k = up ? v[i + 16] : v[i], s = up ? v[i] : v[i + 16]; v[i] = k + __shfl_xor_sync(~0u, s, 16); }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const bool up = lane & 8; const float k = up ? v[i + 8] : v[i], s = up ? v[i] : v[i + 8]; v[i] = k + __shfl_xor_sync(~0u, s, 8); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const bool up = lane & 4; const float k = up ? v[i + 4] : v[i], s = up ? v[i] : v[i + 4]; v[i] = k + __shfl_xor_sync(~0u, s, 4); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { const bool up = lane & 2; const float k = up ? v[i + 2] : v[i], s = up ? v[i] : v[i + 2]; v[i] = k + __shfl_xor_sync(~0u, s, 2); }
  { const bool up = lane & 1; const float k = up ? v[1] : v[0], s = up ? v[0] : v[1]; v[0] = k + __shfl_xor_sync(~0u, s, 1); }
  return v[0];
}

// ------------------------------------------------------------------ the RMS-adaLN backward of one tile (as in ffn_bwd_tf32.cu)
struct RmsArgs {
  const float* XIN; const float* DRES; float* OUT; const float* MOD; float* DMOD;
  int scol, dcol, b, a0, s0, S, A, Bn, SP, AT; float eps;
};
template <bool AT8>
DEVI void rms_bwd(uint32_t tacc, const RmsArgs& p, float* xch, float2* rs, int bar, int qw, int lane) {
  const int c = qw * 32 + lane, R = p.SP * p.AT;
  constexpr int NA = AT8 ? 8 : 1;
  constexpr int NG = AT8 ? 1 : 16;
  float sc8[NA], ash[NA], asc[NA];
#pragma unroll
  for (int k = 0; k < NA; ++k) {
    sc8[k] = (AT8 && p.s0 + k < p.S) ? 1.f + ldf(p.MOD + ((size_t)p.b * p.S + p.s0 + k) * MODW + p.scol + c) : 1.f;
    ash[k] = 0.f; asc[k] = 0.f;
  }
#pragma unroll 1
  for (int r0 = 0; r0 < R; r0 += 16) {
    uint32_t dv[16];
    tmem_ld16(tacc + (uint32_t)r0, dv);
    float xv[16], svg[NG];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      const int j = r0 + jj, sp = AT8 ? j >> 3 : j / p.AT, at = AT8 ? j & 7 : j - sp * p.AT;
      const bool ok = j < R && p.a0 + sp < p.A && p.s0 + at < p.S;
      xv[jj] = ok ? ldf(p.XIN + ((((size_t)(p.a0 + sp) * p.Bn + p.b) * p.S + p.s0 + at) * C + c)) : 0.f;
      if constexpr (!AT8) svg[jj] = ok ? 1.f + ldf(p.MOD + ((size_t)p.b * p.S + p.s0 + at) * MODW + p.scol + c) : 1.f;
    }
    auto scale = [&](int jj) -> float {
      if constexpr (AT8) return sc8[jj & 7]; else return svg[jj];
    };
    tmem_wait_ld();
    {
      float v[32];
#pragma unroll
      for (int jj = 0; jj < 16; ++jj) { v[2 * jj] = xv[jj] * xv[jj]; v[2 * jj + 1] = xv[jj] * scale(jj) * __uint_as_float(dv[jj]); }
      xch[qw * 32 + lane] = rscatter32(v, lane);
    }
    named_bar_sync(bar, 128);
    if (qw == 0) {
      const float tot = (xch[lane] + xch[32 + lane]) + (xch[64 + lane] + xch[96 + lane]);
      const float oth = __shfl_xor_sync(~0u, tot, 1);
      if (!(lane & 1)) {
        const float rstd = 1.f / sqrtf(tot * (1.f / C) + p.eps);
        rs[lane >> 1] = make_float2(rstd, rstd * oth * (1.f / C));
      }
    }
    named_bar_sync(bar, 128);
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      const int j = r0 + jj, sp = AT8 ? j >> 3 : j / p.AT, at = AT8 ? j & 7 : j - sp * p.AT;
      const bool ok = j < R && p.a0 + sp < p.A && p.s0 + at < p.S;
      const float2 rr = rs[jj];
      const float dx = __uint_as_float(dv[jj]), xh = xv[jj] * rr.x, dxh = dx * scale(jj);
      if (ok) {
        const size_t gr = (((size_t)(p.a0 + sp) * p.Bn + p.b) * p.S + p.s0 + at) * C + c;
        p.OUT[gr] = ldf(p.DRES + gr) + rr.x * (dxh - xh * rr.y);
      }
      if constexpr (AT8) {
        ash[jj & 7] += ok ? dx : 0.f; asc[jj & 7] += ok ? dx * xh : 0.f;
      } else if (ok) {
        float* dm = p.DMOD + ((size_t)p.b * p.S + p.s0 + at) * MODW + p.dcol + c;
        red1(dm, dx); red1(dm + C, dx * xh);
      }
    }
  }
  if constexpr (AT8) {
    const int rot = (int)blockIdx.x & 7;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int k = (kk + rot) & 7;
      if (p.s0 + k < p.S) {
        float* dm = p.DMOD + ((size_t)p.b * p.S + p.s0 + k) * MODW + p.dcol + c;
        red1(dm, pick8(ash, k)); red1(dm + C, pick8(asc, k));
      }
    }
  }
}

// MMA slice i (0..15) -> its activation use within the tile and its column in dP / W^T
DEVI int slice_use(int i) { return i < 8 ? 4 * (i >> 1) + 2 * (i & 1) : i < 12 ? U_DV + (i - 8) : U_DG + (i - 12); }
DEVI int slice_col(int i) { return i < 8 ? (i & 1) * C + (i >> 1) * D : i < 12 ? 2 * C + (i - 8) * D : 3 * C + (i - 12) * D; }

extern "C" __global__ void __launch_bounds__(512, 1)
swa_qkvg_bwd_tf32_sm100(const __grid_constant__ CUtensorMap mpq, const __grid_constant__ CUtensorMap mpk,
                        const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdqh,
                        const __grid_constant__ CUtensorMap mdkh, const __grid_constant__ CUtensorMap mdvh,
                        const __grid_constant__ CUtensorMap mwt, const __grid_constant__ CUtensorMap mdp,
                        const float* __restrict__ Q, const float* __restrict__ DQ1, const float* __restrict__ MOD,
                        const float* __restrict__ COS, const float* __restrict__ SIN, float* __restrict__ DQ, float* __restrict__ DMOD,
                        int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps, float qk_eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int R = SP * AT;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };
  auto aslot = [&](int T, int u, int& s, uint32_t& ph) { const int g = UPT * T + u; s = g % NS; ph = (uint32_t)(g / NS) & 1u; };

  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(&B.afull[i], 1); mbar_init(&B.aempty[i], 2); mbar_init(&B.ardy[i], 1); }
    for (int i = 0; i < NW; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    for (int i = 0; i < 2; ++i) { mbar_init(&B.dxfull[i], 1); mbar_init(&B.dxfree[i], 4); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ activation producer
    if (lane == 0) {
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int u = 0; u < UPT; ++u) {
          const int g = UPT * T + u, s = g % NS;
          if (g >= NS) mbar_wait(&B.aempty[s], ((g / NS) - 1) & 1);
          const uint32_t dst = su + O_A + s * SL;
          mbar_expect_tx(&B.afull[s], (uint32_t)(R * 128));
          if (u < U_DV) {
            const int h = u >> 2, kind = u & 3;
            if (kind == 0) tma_load_4d(dst, &mpq, &B.afull[s], D * h, s0, b, a0);
            else if (kind == 2) tma_load_4d(dst, &mpk, &B.afull[s], D * h, s0, b, a0);
            else tma_load_5d(dst, kind == 1 ? &mdqh : &mdkh, &B.afull[s], 0, s0, h, b, a0);
          } else if (u < U_DG) {
            tma_load_5d(dst, &mdvh, &B.afull[s], 0, s0, u - U_DV, b, a0);
          } else {
            tma_load_4d(dst, &mdg, &B.afull[s], D * (u - U_DG), s0, b, a0);
          }
          if ((u & 1) || u >= U_DV) mbar_arrive(&B.ardy[s]);               // not thread-written: keeps the rdy phases in step with the ring
        }
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ weight producer: W^T slices in MMA order
    if (lane == 0) {
      for (int T = 0; T < ntT; ++T)
        for (int i = 0; i < WPT; ++i) {
          const int g = WPT * T + i, s = g % NW;
          if (g >= NW) mbar_wait(&B.wempty[s], ((g / NW) - 1) & 1);
          mbar_expect_tx(&B.wfull[s], SL);
          tma_load_2d(su + O_W + s * SL, &mwt, &B.wfull[s], slice_col(i), 0);
        }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer: dx^T = W^T dP^T
    for (int T = 0; T < ntT; ++T) {
      const int x = T & 1;
      if (T >= 2) mbar_wait(&B.dxfree[x], ((T >> 1) - 1) & 1);             // warpgroup 1 has read dx^T of tile T - 2
      for (int i = 0; i < WPT; ++i) {
        int s; uint32_t ph; aslot(T, slice_use(i), s, ph);
        if (i < 8) mbar_wait(&B.ardy[s], ph); else mbar_wait(&B.afull[s], ph);
        const int gw = WPT * T + i, ws = gw % NW;
        mbar_wait(&B.wfull[ws], (gw / NW) & 1);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            tf::mma_ss(tmem + x * 128, desc_k128(su + O_W + ws * SL) + (uint64_t)(k * 2), desc_k128(su + O_A + s * SL) + (uint64_t)(k * 2),
                       I_DX, (i > 0 || k > 0) ? 1u : 0u);
          tc_commit(&B.wempty[ws]);
          tc_commit(&B.aempty[s]);                                         // one of the slot's two releases (the dP store is the other)
          if (i == WPT - 1) tc_commit(&B.dxfull[x]);
        }
        __syncwarp();
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ dP stores, straight from the slots
    if (lane == 0) {
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int i = 0; i < WPT; ++i) {
          int s; uint32_t ph; aslot(T, slice_use(i), s, ph);
          if (i < 8) mbar_wait(&B.ardy[s], ph); else mbar_wait(&B.afull[s], ph);
          tma_store_4d(&mdp, su + O_A + s * SL, slice_col(i), s0, b, a0);
          tma_store_commit();
          tma_store_wait_read0();
          mbar_arrive(&B.aempty[s]);
        }
      }
      tma_store_wait0();
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------------------ warpgroup 0: dpq / dpk, row r per thread
    const int r = (warp & 3) * 32 + lane;
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int sp = r / AT, at = r - sp * AT;
      const bool rok = r < R && s0 + at < S;
      const float4* cs4 = reinterpret_cast<const float4*>(COS + ((size_t)b * S + (rok ? s0 + at : 0)) * (D / 2));
      const float4* sn4 = reinterpret_cast<const float4*>(SIN + ((size_t)b * S + (rok ? s0 + at : 0)) * (D / 2));
#pragma unroll 1
      for (int job = 0; job < 8; ++job) {                                  // (head h, q / k)
        const int h = job >> 1, uP = 4 * h + 2 * (job & 1);
        int sP, sD; uint32_t pP, pD;
        aslot(T, uP, sP, pP); aslot(T, uP + 1, sD, pD);
        mbar_wait(&B.afull[sP], pP);
        mbar_wait(&B.afull[sD], pD);
        const uint32_t pb = su + O_A + sP * SL, db = su + O_A + sD * SL;
        float p[32], gv[32];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 u = lds128(pb + sw128((uint32_t)r, (uint32_t)q)), v = lds128(db + sw128((uint32_t)r, (uint32_t)q));
          p[4 * q] = __uint_as_float(u.x); p[4 * q + 1] = __uint_as_float(u.y); p[4 * q + 2] = __uint_as_float(u.z); p[4 * q + 3] = __uint_as_float(u.w);
          gv[4 * q] = __uint_as_float(v.x); gv[4 * q + 1] = __uint_as_float(v.y); gv[4 * q + 2] = __uint_as_float(v.z); gv[4 * q + 3] = __uint_as_float(v.w);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {                                      // transpose of RoPE: dx1 = dy1 c + dy2 s; dx2 = dy2 c - dy1 s
          const float4 c4 = rok ? __ldg(cs4 + k) : make_float4(1.f, 1.f, 1.f, 1.f), s4 = rok ? __ldg(sn4 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
          const float cc[4] = {c4.x, c4.y, c4.z, c4.w}, ss[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int d = 4 * k + e;
            const float y1 = gv[d], y2 = gv[d + 16];
            gv[d] = y1 * cc[e] + y2 * ss[e];
            gv[d + 16] = y2 * cc[e] - y1 * ss[e];
          }
        }
        float s4[4] = {0.f, 0.f, 0.f, 0.f};                                // head RMS backward: dp = rr (g - yh mean(g yh)), yh = p rr
#pragma unroll
        for (int d = 0; d < 32; ++d) s4[d & 3] = fmaf(p[d], p[d], s4[d & 3]);
        const float rr = 1.f / sqrtf(((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / D) + qk_eps);
#pragma unroll
        for (int k = 0; k < 4; ++k) s4[k] = 0.f;
#pragma unroll
        for (int d = 0; d < 32; ++d) { p[d] *= rr; s4[d & 3] = fmaf(gv[d], p[d], s4[d & 3]); }
        const float m = ((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / D);
#pragma unroll
        for (int q = 0; q < 8; ++q)                                        // dpq in place over pq (the bytes this thread read)
          sts128(pb + sw128((uint32_t)r, (uint32_t)q),
                 make_uint4(tf::rna(rr * (gv[4 * q] - p[4 * q] * m)), tf::rna(rr * (gv[4 * q + 1] - p[4 * q + 1] * m)),
                            tf::rna(rr * (gv[4 * q + 2] - p[4 * q + 2] * m)), tf::rna(rr * (gv[4 * q + 3] - p[4 * q + 3] * m))));
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (r == 0) { mbar_arrive(&B.ardy[sP]); mbar_arrive_cnt(&B.aempty[sD], 2); }
      }
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------------ warpgroup 1: RMS-adaLN backward (channel c)
    const int qw = warp & 3;
    float* xch = reinterpret_cast<float*>(sm + O_EX);
    float2* rs = reinterpret_cast<float2*>(sm + O_EX + 512);
    for (int T = 0; T < ntT; ++T) {
      const int x = T & 1;
      RmsArgs p{Q, DQ1, DQ, MOD, DMOD, C, 0, 0, 0, 0, S, A, Bn, SP, AT, eps};
      coords(T, p.b, p.a0, p.s0);
      mbar_wait(&B.dxfull[x], (T >> 1) & 1);
      tc_fence_after();
      const uint32_t tacc = tmem + ((uint32_t)(qw * 32) << 16) + (uint32_t)(x * 128);
      if (AT == 8) rms_bwd<true>(tacc, p, xch, rs, 2, qw, lane); else rms_bwd<false>(tacc, p, xch, rs, 2, qw, lane);
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxfree[x]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
