// k1wb.cu -- backward of the B200 wide TriMul front (D = 256 / 384 / 512): recomputes k1w's pre-activations
//   g_c = xn Wg_c^T, p_c = xn Wp_c^T (same transform, same TMEM products, masked token rows zero)
// and turns the plane gradient da [P, L*L] (channel-major, from the contraction backward) into the pre-activation gradients
//   dg_c = da_c p_c sigmoid(g_c) (1 - sigmoid(g_c)),   dp_c = da_c sigmoid(g_c)          (0 for a masked pair)
// written token-major into dpre [M, 2P] with k1w's packed-row order (chunk c = columns 128c.. : 64 gate, 64 projection), so
// cuBLAS takes dW1 = dpre^T xn and dxn = dpre W1 directly. The epilogue loads the da tile with ldmatrix.trans from the same
// [64 ch][32 tok] SW64 box that k1w's epilogue stores with stmatrix.trans, which puts it in the accumulator fragment layout.
// Warp roles as k1w; per epilogue warp one 4 KB da box and one 8 KB dpre staging tile ([32 tok][2 x 64] SW128).
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace k1wb {

constexpr int TOK = 128, KBB = TOK * 64 * 2;   // one [128][64] bf16 K-block, SW128 = 16 KB
constexpr int DAB = 64 * 32 * 2;               // per-warp da box: [64 ch][32 tok] bf16, 64-B swizzle = 4 KB
constexpr int DPB = 32 * 64 * 2;               // per-warp dpre box: [32 tok][64 col] bf16, 128-B swizzle = 4 KB (x2: gate, proj)
constexpr int OSTW = DAB + 2 * DPB;
constexpr int NTHREADS = 512;
constexpr float LOG2E = 1.4426950408889634f;
constexpr uint32_t IDESC = idesc_bf16(128, 128);

template <int D>
struct Cfg {
  static constexpr int KB = D / 64;                       // K-blocks per token row
  static constexpr int XC = D / 2;                        // TMEM columns of one xn tile (bf16 pairs)
  static constexpr int NXN = (D <= 256) ? 2 : 1;          // xn tiles resident in TMEM
  static constexpr int NACC = (D <= 128) ? 3 : 2;         // 128-column accumulators
  static constexpr uint32_t T_ACC = 0, T_XN = NACC * 128;
  static_assert(T_XN + NXN * XC <= 512, "tmem");
  static constexpr int NWS = 4;                           // weight K-block ring
  static constexpr int NXS = 3;                           // x K-block ring
  static constexpr int O_W = 0, O_X = O_W + NWS * KBB, O_OUT = O_X + NXS * KBB, O_GB = O_OUT + 8 * OSTW;
  static constexpr int O_BAR = O_GB + 2 * D * 4;
  static constexpr int SMEM = O_BAR + 512 + 1024;
  static_assert(SMEM <= 232448, "smem");
};

struct WMaps { CUtensorMap m[4]; };   // W_l, W_lg, W_r, W_rg

struct Bars {
  uint64_t w_full[8], w_empty[8], x_full[4], x_empty[4], xn_full[2], xn_empty[2], acc_full[3], acc_empty[3], da_full[8];
  uint32_t tmem;
};

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// k1.cu K1_SIG == 2: 0.5 + 0.5 tanh.approx(g / 2)
DEV float2 sigmoid2(float2 g) {
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
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV void ldmatrix_x4_trans(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
DEV void stmatrix_x4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
DEV float2 __fsub2_rn(float2 a, float2 b) { return __fadd2_rn(a, make_float2(-b.x, -b.y)); }
DEV void stmatrix_x4_trans(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

template <int D, int P>
__global__ void __launch_bounds__(NTHREADS, 1)
    k1wb_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ WMaps mw, const __grid_constant__ CUtensorMap mda,
                const __grid_constant__ CUtensorMap mdpre,
               const uint8_t* __restrict__ tokmask, int L, float* __restrict__ mean, float* __restrict__ rstd,
               const float* __restrict__ gamma, const float* __restrict__ beta, int tiles, float eps, int stats_out) {
  using G_ = Cfg<D>;
  constexpr int KB = G_::KB, XC = G_::XC, NXN = G_::NXN, NACC = G_::NACC, NWS = G_::NWS, NXS = G_::NXS;
  constexpr int NCH = P / 64;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;

  if (tid == 0) {
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], 1); }
    for (int s = 0; s < NXS; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 4); }
    for (int b = 0; b < NACC; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int b = 0; b < NXN; ++b) { mbar_init(&B.xn_full[b], 1); mbar_init(&B.xn_empty[b], 1); }
    fence_mbar_init();
    for (int w = 0; w < 8; ++w) mbar_init(&B.da_full[w], 1);
    prefetch_tmap(&mx); prefetch_tmap(&mda); prefetch_tmap(&mdpre);
    for (int k = 0; k < 4; ++k) prefetch_tmap(&mw.m[k]);
  }
  for (int k = tid; k < D; k += NTHREADS) {
    reinterpret_cast<float*>(sm + G_::O_GB)[k] = gamma[k];
    reinterpret_cast<float*>(sm + G_::O_GB)[D + k] = beta[k];
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) {
    if (warp == 0 && lane == 0) {
      // ---------------------------------------------------------------- x K-blocks -> ring (slot free once the transform read it)
      int s = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = (cta + i * G) * TOK;
        for (int kb = 0; kb < KB; ++kb) {
          mbar_wait(&B.x_empty[s], ph ^ 1);
          mbar_expect_tx(&B.x_full[s], KBB);
          tma_load_2d(sm + G_::O_X + s * KBB, &mx, &B.x_full[s], kb * 64, row, EVICT_FIRST);
          if (++s == NXS) { s = 0; ph ^= 1; }
        }
      }
    } else if (warp == 3 && lane == 0) {
      // ---------------------------------------------------------------- weight K-blocks: tile, chunk, K-block -> ring
      int s = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i)
        for (int c = 0; c < NCH; ++c)
          for (int kb = 0; kb < KB; ++kb) {
            mbar_wait(&B.w_empty[s], ph ^ 1);
            mbar_expect_tx(&B.w_full[s], KBB);
            const int row = c * 64, side = row < P / 2 ? 0 : 1, r = side ? row - P / 2 : row;
            tma_load_2d(sm + G_::O_W + s * KBB, &mw.m[2 * side + 1], &B.w_full[s], kb * 64, r, EVICT_LAST);            // gate rows
            tma_load_2d(sm + G_::O_W + s * KBB + KBB / 2, &mw.m[2 * side], &B.w_full[s], kb * 64, r, EVICT_LAST);      // projection
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
    } else if (warp == 1) {
      // ---------------------------------------------------------------- MMA: acc[tok][0:64 gate | 64:128 proj] = xn (TMEM) . W_c^T
      const bool ldr = elect_one();
      const uint64_t w0 = desc_k_sw128(su + G_::O_W);
      constexpr uint64_t WSD = KBB >> 4;
      int s = 0, a = 0;
      uint32_t ph = 0, pa = 0;
      for (int i = 0; i < n_local; ++i) {
        const int b = (NXN == 1) ? 0 : (i & 1);
        mbar_wait(&B.xn_full[b], (i / NXN) & 1);
        tc_fence_after();
        const uint32_t xa = tmem + G_::T_XN + b * XC;
#pragma unroll 1
        for (int c = 0; c < NCH; ++c) {
          mbar_wait(&B.acc_empty[a], pa ^ 1);
          tc_fence_after();
          const uint32_t d = tmem + G_::T_ACC + a * 128;
#pragma unroll 1
          for (int kb = 0; kb < KB; ++kb) {
            mbar_wait(&B.w_full[s], ph);
            tc_fence_after();
            if (ldr) {
              const uint64_t wd = w0 + (uint64_t)s * WSD;
              const uint32_t xk = xa + kb * 32;
              umma_ts(d, xk, wd, IDESC, kb > 0);
              umma_ts(d, xk + 8, wd + 2, IDESC, 1);
              umma_ts(d, xk + 16, wd + 4, IDESC, 1);
              umma_ts(d, xk + 24, wd + 6, IDESC, 1);
              umma_commit(&B.w_empty[s]);
            }
            __syncwarp();
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[a]);
          __syncwarp();
          if (++a == NACC) { a = 0; pa ^= 1; }
        }
        if (ldr) umma_commit(&B.xn_empty[b]);
        __syncwarp();
      }
    }
  } else if (warp < 8) {
    // ------------------------------------------------------------------ transform: thread = token row r, K-block by K-block
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    int s = 0; uint32_t ph = 0;
    for (int i = 0; i < n_local; ++i) {
      const int row = (cta + i * G) * TOK + r, b = (NXN == 1) ? 0 : (i & 1);
      const uint32_t kmask = (tokmask == nullptr || (tokmask[row / L] && tokmask[row % L])) ? 0xffffffffu : 0u;
      float mu, rs;
      if constexpr (D <= 128) {
        // whole row resident (KB <= 2 ring slots): two-pass statistics here, as k1.cu (packed f32x2 sums)
        int ss = s; uint32_t pp = ph;
        for (int kb = 0; kb < KB; ++kb) { mbar_wait(&B.x_full[ss], pp); if (++ss == NXS) { ss = 0; pp ^= 1; } }
        auto ldx = [&](int q) { const int kb = q >> 3; int sl = s + kb; if (sl >= NXS) sl -= NXS;
                                return lds128(su + G_::O_X + sl * KBB + sw128(r, q & 7)); };
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
        for (int q = 0; q < D / 8; ++q) {
          const uint4 t = ldx(q);
          a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D);
        const float2 nm0 = make_float2(-mu, -mu);
        a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
        for (int q = 0; q < D / 8; ++q) {
          const uint4 t = ldx(q);
          float2 d;
          d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
          d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
          d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
          d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D), eps));
        if (stats_out) { mean[row] = mu; rstd[row] = rs; }
      } else {
        mu = mean[row]; rs = rstd[row];
      }
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
      if (i >= NXN) mbar_wait(&B.xn_empty[b], ((i / NXN) - 1) & 1);
      tc_fence_after();
      for (int kb = 0; kb < KB; ++kb) {
        if constexpr (D > 128) mbar_wait(&B.x_full[s], ph);
        const uint32_t xr = su + G_::O_X + s * KBB;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {          // 16 channels (8 packed words) per TMEM store
          uint32_t o[8];
#pragma unroll
          for (int qq = 0; qq < 2; ++qq) {
            const int q = q4 * 2 + qq;
            const uint4 t = lds128(xr + sw128(r, q));
            const int col = kb * 64 + q * 8;
            const float4 ga = lds128f(su + G_::O_GB + col * 4);
            const float4 gb4 = lds128f(su + G_::O_GB + col * 4 + 16);
            const float4 ba = lds128f(su + G_::O_GB + D * 4 + col * 4);
            const float4 bb = lds128f(su + G_::O_GB + D * 4 + col * 4 + 16);
            auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
              const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
              return pack_bf16(y.x, y.y) & kmask;
            };
            o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
            o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
            o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
            o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
          }
          tmem_st8(trow + G_::T_XN + b * XC + kb * 32 + q4 * 8, o);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.x_empty[s]);
        if (++s == NXS) { s = 0; ph ^= 1; }
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (warp == 4 && lane == 0) mbar_arrive(&B.xn_full[b]);
    }
  } else {
    // ------------------------------------------------------------------ backward epilogue: warp quarter q = tokens 32q..32q+31
    const int grp = (warp - 8) >> 2, q = warp & 3, ew = warp - 8;
    const uint32_t sda = su + G_::O_OUT + ew * OSTW, sdp = sda + DAB;     // da box, then dpre gate / proj boxes
    uint32_t dph = 0;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + i * G) * TOK;
      for (int c = grp; c < NCH; c += 2) {
        const int ci = i * NCH + c, acc = ci % NACC;
        if (lane == 0) {
          bulk_wait_read<0>();                               // this warp's previous dpre store has left the staging tile
          mbar_expect_tx(&B.da_full[ew], DAB);
          tma_load_2d(sm + G_::O_OUT + ew * OSTW, &mda, &B.da_full[ew], row0 + q * 32, c * 64, EVICT_FIRST);
        }
        __syncwarp();
        mbar_wait(&B.acc_full[acc], (ci / NACC) & 1);
        tc_fence_after();
        mbar_wait(&B.da_full[ew], dph);
        dph ^= 1;
        // token mask of this thread's two fragment rows (t / 4 and t / 4 + 8 of each 16-token group)
        uint32_t km[2][2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int tok = row0 + q * 32 + hh * 16 + e * 8 + (lane >> 2);
            km[hh][e] = (tokmask == nullptr || (tokmask[tok / L] && tokmask[tok % L])) ? 1u : 0u;
          }
        const uint32_t tq = tmem + G_::T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16);
        uint32_t ga[16], pa[16], gb[16], pb[16];
        auto ldg = [&](int grp4, uint32_t (&g)[16], uint32_t (&p)[16]) {
          const uint32_t ta = tq + ((uint32_t)((grp4 >> 1) * 16) << 16) + (grp4 & 1) * 32;
          tmem_ld16x256_x4(ta, g);
          tmem_ld16x256_x4(ta + 64, p);
        };
        auto work = [&](int grp4, const uint32_t (&g)[16], const uint32_t (&p)[16]) {
          const int hh = grp4 >> 1, cq = grp4 & 1;
          const int k = lane >> 3, jr = lane & 7;
          uint32_t da[8];
#pragma unroll
          for (int x = 0; x < 2; ++x) {                    // the inverse of k1w's stmatrix.trans: da in the fragment layout
            const int chl = cq * 32 + (2 * x + (k >> 1)) * 8 + jr;
            const int qc = hh * 2 + (k & 1);
            ldmatrix_x4_trans(sda + chl * 64 + ((qc ^ ((chl >> 1) & 3)) << 4), da[4 * x], da[4 * x + 1], da[4 * x + 2], da[4 * x + 3]);
          }
          uint32_t mg[8], mp[8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {                  // e: token t/4 (+0) or t/4 + 8
              const float2 gg = make_float2(__uint_as_float(g[4 * j + 2 * e]), __uint_as_float(g[4 * j + 2 * e + 1]));
              const float2 pp = make_float2(__uint_as_float(p[4 * j + 2 * e]), __uint_as_float(p[4 * j + 2 * e + 1]));
              const uint32_t dw = da[2 * j + e];
              const float2 dd = km[hh][e] ? make_float2(bf16lo(dw), bf16hi(dw)) : make_float2(0.f, 0.f);
              const float2 sg = sigmoid2(gg);
              const float2 dp = __fmul2_rn(dd, sg);
              const float2 dg = __fmul2_rn(__fmul2_rn(dp, pp), __fsub2_rn(make_float2(1.f, 1.f), sg));
              mg[2 * j + e] = pack_bf16(dg.x, dg.y);
              mp[2 * j + e] = pack_bf16(dp.x, dp.y);
            }
          }
#pragma unroll
          for (int x = 0; x < 2; ++x) {                    // token-major [32 tok][64 col] SW128 boxes (gate, projection)
            const int tr = hh * 16 + (k & 1) * 8 + jr, cb = cq * 4 + 2 * x + (k >> 1);
            const uint32_t off = tr * 128 + ((cb ^ (tr & 7)) << 4);
            stmatrix_x4(sdp + off, mg[4 * x], mg[4 * x + 1], mg[4 * x + 2], mg[4 * x + 3]);
            stmatrix_x4(sdp + DPB + off, mp[4 * x], mp[4 * x + 1], mp[4 * x + 2], mp[4 * x + 3]);
          }
        };
        ldg(0, ga, pa);
        tmem_wait_ld();
        ldg(1, gb, pb);
        work(0, ga, pa);
        tmem_wait_ld();
        ldg(2, ga, pa);
        work(1, gb, pb);
        tmem_wait_ld();
        ldg(3, gb, pb);
        work(2, ga, pa);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[acc]);
        work(3, gb, pb);
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mdpre, sm + G_::O_OUT + ew * OSTW + DAB, c * 128, row0 + q * 32);
          tma_store_2d(&mdpre, sm + G_::O_OUT + ew * OSTW + DAB + DPB, c * 128 + 64, row0 + q * 32);
          bulk_commit();
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}


int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k1wb

// x [M, D]; wl, wlg, wr, wrg [P/2, D] (unit column stride); tokmask [L] or none; mean / rstd [M] (k1w_stats); gamma / beta [D];
// da [P, L, L] bf16 (plane gradient); dpre [M, ldc] bf16 view, columns 0 .. 2P - 1 written.
void k1wb_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                  c10::optional<torch::Tensor> tokmask, torch::Tensor mean, torch::Tensor rstd, torch::Tensor gamma, torch::Tensor beta,
                  torch::Tensor da, torch::Tensor dpre, int64_t grid) {
  using namespace k1wb;
  const int64_t M = x.size(0), D = x.size(1), P = da.size(0);
  TORCH_CHECK(M % TOK == 0 && da.numel() == P * M && x.is_contiguous() && da.is_contiguous() && da.dim() == 3);
  TORCH_CHECK(dpre.size(0) == M && dpre.size(1) >= 2 * P && dpre.stride(1) == 1 && (dpre.stride(0) * 2) % 16 == 0);
  const int L = (int)da.size(1);
  TORCH_CHECK(!tokmask.has_value() || (tokmask->numel() == L && tokmask->element_size() == 1 && tokmask->is_contiguous()));
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  WMaps mw;
  const torch::Tensor* ws[4] = {&wl, &wlg, &wr, &wrg};
  for (int k = 0; k < 4; ++k) {
    const torch::Tensor& w = *ws[k];
    TORCH_CHECK(w.size(0) == P / 2 && w.size(1) == D && w.stride(1) == 1 && (w.stride(0) * 2) % 16 == 0);
    mw.m[k] = tmap::make(w.data_ptr(), bf, {(uint64_t)D, (uint64_t)(P / 2)}, {(uint64_t)w.stride(0) * 2}, {64, 64},
                         CU_TENSOR_MAP_SWIZZLE_128B);
  }
  auto mda = tmap::make(da.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {32u, 64}, CU_TENSOR_MAP_SWIZZLE_64B);
  auto mdp = tmap::make(dpre.data_ptr(), bf, {(uint64_t)(2 * P), (uint64_t)M}, {(uint64_t)dpre.stride(0) * 2}, {64, 32},
                        CU_TENSOR_MAP_SWIZZLE_128B);
  const int tiles = (int)(M / TOK);
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, tiles);
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* mk = tokmask.has_value() ? reinterpret_cast<const uint8_t*>(tokmask->data_ptr()) : nullptr;
#define K1WB_CASE(DD, PP) \
  if (D == DD && P == PP) { \
    cudaFuncSetAttribute(k1wb_kernel<DD, PP>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<DD>::SMEM); \
    k1wb_kernel<DD, PP><<<g, NTHREADS, Cfg<DD>::SMEM, st>>>(mx, mw, mda, mdp, mk, L, mean.data_ptr<float>(), rstd.data_ptr<float>(), \
        gamma.data_ptr<float>(), beta.data_ptr<float>(), tiles, 1e-5f, 0); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K1WB_CASE(256, 1024) K1WB_CASE(256, 512) K1WB_CASE(384, 1536) K1WB_CASE(384, 768) K1WB_CASE(512, 2048) K1WB_CASE(512, 1024)
#undef K1WB_CASE
  TORCH_CHECK(false, "k1wb_forward: unsupported (D, P) = (", D, ", ", P, ")");
}
