// transition_bwd_f32dx.cu -- the fp32 Transition backward's dgrad at D = 128 (hidden n D) as ONE fused sm_90a kernel, native TF32 tensor cores:
//
//   dh = dy Ws      a = xn Wa^T, b = xn Wb^T      sig = sigmoid(a)      h = a sig b      dA = dh b sig (1 + a (1 - sig))      dB = dh a sig
//   d_xn = dA Wa + dB Wb      dx = rstd (gamma d_xn - xhat mean(gamma d_xn xhat) - mean(gamma d_xn)) + dy,   xhat = x rstd - c1
//
// The DX role of the bf16 D = 128 backward (transition_fused_bwd_sm90a_kernel.cu) in TF32: a persistent CTA per 128-row tile, a producer
// warpgroup (TMA) and two consumer warpgroups of 64 rows; the hidden chunks (HS = 32 units) stream through a two-slot weight ring, and per
// chunk each warpgroup computes dh (RS: dy in A-fragment registers, loaded once per tile and rounded to nearest TF32), [a | b] (SS over the
// resident xn tile), the SwiGLU backward in the accumulator registers, and accumulates d_xn with an RS m64n128 whose A fragments ARE the dA /
// dB accumulators (rounded to TF32): d_xn never leaves registers; the LayerNorm backward, the residual and the dgamma / dbeta partials follow
// in the same fragment layout. h and [dA | dB] are stored for the weight gradients (cuBLAS TF32 GEMMs, the bf16 wide builds' wgrad).
//
// K order inside a k-step: a TF32 A fragment holds columns (q, q + 4) of each 8-wide k-step, the C fragment columns (2q, 2q + 1). Taking the
// C fragment as the A fragment reads the k-step's columns in the order (0, 2, 4, 6, 1, 3, 5, 7); the B operands of the two RS products (Ws^T
// over d, [Wa; Wb]^T over the hidden units) are stored with their K columns permuted the same way (fused_wide_sm90a._f32_bwd_forms), and dy
// is read into the A fragments as (2q, 2q + 1) pairs.
//
// dgamma / dbeta: each warp accumulates its own [2][D] fp32 row of `part` ([grid][8 warps][2][D]) across its tiles -- one owner per address,
// a fixed order -- and `f32_part_reduce` sums the rows in a fixed order: replays are bit-identical.
#pragma once
#include "transition_fwd_f32io.cu"
#include "../../generic_sm90a/wgmma_ops.cuh"

TMN_DEVI void tf32_rs_n32(float (&d)[16], const uint32_t (&a)[4], uint64_t desc, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %21, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(accumulate));
}
TMN_DEVI void tf32_rs_n128(float (&d)[64], const uint32_t (&a)[4], uint64_t desc, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(accumulate));
}
TMN_DEVI void tf32_rs_n64(float (&d)[32], const uint32_t (&a)[4], uint64_t desc, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(accumulate));
}
// L2 prefetches: a TMA box of a tensor map, and a contiguous byte range (16-byte multiple)
TMN_DEVI void l2_prefetch_box(const CUtensorMap* m, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(m), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void l2_prefetch_bytes(const void* p, uint32_t bytes) { asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory"); }
TMN_DEVI float lds_f(uint32_t addr) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr)); return v; }
TMN_DEVI uint32_t tf32u(float v) { uint32_t u; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(v)); return u; }

template <int D, int N>
struct F32DxCfg {
  static constexpr int H = N * D, HS = 32, NCH = H / HS, ROWS = 128, KX = D / 32;   // KX: 32-column (128-byte) K blocks of an fp32 row
  // slot: Ws^T_j (perm) [HS][D] (KX blocks of [HS rows][128 B]) | [Wa_j; Wb_j] [2 HS][D] (KX blocks of [Wa_j HS rows; Wb_j HS rows][128 B],
  // one n64 B tile per k-step) | [Wa_j; Wb_j]^T (perm) [D][2 HS] (2 blocks of [D rows][128 B])
  static constexpr int W_BLK = HS * 128, F_WS = 0, F_WAB = KX * W_BLK, F_WT = 3 * KX * W_BLK;
  static constexpr int F_SLOT = F_WT + 2 * D * 128;
  static constexpr int F_XN = 2 * F_SLOT;              // xn tile: [2 warpgroups][KX blocks][64 rows][128 B]
  static constexpr int F_BAR = F_XN + 2 * KX * 8192;
  static constexpr int SMEM = F_BAR + 128;
  static_assert(D == 128, "the register budget is the D = 128 one");
  static_assert(SMEM <= 231424, "shared memory budget");
};

// STORE: also write h and [dA | dB] (for weight gradients outside the kernel); off when transition_bwd_f32dw.cu computes them
template <int D, int N, bool STORE = true>
__global__ void __launch_bounds__(384, 1)
transition_bwd_f32dx(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwst, const __grid_constant__ CUtensorMap mwa,
                     const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mwabt, const float* __restrict__ dy,
                     const float* __restrict__ x, const float* __restrict__ rstd, const float* __restrict__ c1, const float* __restrict__ gamma,
                     float* __restrict__ dx, float* __restrict__ hout, float* __restrict__ dab, float* __restrict__ part, int M, int tiles) {
  using C = F32DxCfg<D, N>;
  constexpr int H = C::H, HS = C::HS, NCH = C::NCH, KX = C::KX;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + 2; uint64_t* x_full = bars + 4; uint64_t* x_free = bars + 5;
  const int cta = blockIdx.x, ncta = gridDim.x;
  const int n_local = (tiles > cta) ? (tiles - cta + ncta - 1) / ncta : 0;
  if (tid == 0) {
    mbar_init(w_full, 1); mbar_init(w_full + 1, 1); mbar_init(w_free, 2); mbar_init(w_free + 1, 2);
    mbar_init(x_full, 1); mbar_init(x_free, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mxn); tma_prefetch_desc(&mwst); tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb); tma_prefetch_desc(&mwabt);
    uint32_t seq = 0;
    for (int i = 0; i < n_local; ++i) {
      const int row = (cta + i * ncta) * C::ROWS;
      if (i > 0) mbar_wait(x_free, (i - 1) & 1u);          // both warpgroups' last [a | b] of tile i - 1 retired
      mbar_arrive_expect_tx(x_full, 2 * KX * 8192);
      for (int h = 0; h < 2; ++h)
        for (int kb = 0; kb < KX; ++kb) tma_load_2d(sm + C::F_XN + (h * KX + kb) * 8192, &mxn, x_full, kb * 32, row + 64 * h);
      for (int j = 0; j < NCH; ++j, ++seq) {
        const int s = (int)(seq & 1);
        if (seq >= 2) mbar_wait(w_free + s, ((seq >> 1) - 1) & 1u);
        uint8_t* b = sm + s * C::F_SLOT;
        mbar_arrive_expect_tx(w_full + s, C::F_SLOT);
        for (int kb = 0; kb < KX; ++kb) {
          tma_load_2d(b + C::F_WS + kb * C::W_BLK, &mwst, w_full + s, kb * 32, j * HS);
          tma_load_2d(b + C::F_WAB + kb * 2 * C::W_BLK, &mwa, w_full + s, kb * 32, j * HS);
          tma_load_2d(b + C::F_WAB + kb * 2 * C::W_BLK + C::W_BLK, &mwb, w_full + s, kb * 32, j * HS);
        }
        tma_load_2d(b + C::F_WT, &mwabt, w_full + s, j * HS, 0);                 // [Wa_j]^T (perm): columns j HS .. of [D][2 H]
        tma_load_2d(b + C::F_WT + D * 128, &mwabt, w_full + s, H + j * HS, 0);   // [Wb_j]^T (perm)
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  const int q = lane & 3, rf = 16 * warp + (lane >> 2);   // fragment rows rf, rf + 8 of this warpgroup's 64
  const uint32_t xs = su + C::F_XN + cw * KX * 8192;
  float* prow = part + ((size_t)blockIdx.x * 8 + cw * 4 + warp) * 2 * D;   // this warp's [2][D] partial row
  if (lane < 4) {
    for (int c = 2 * q; c < D; c += 8) { prow[c] = 0.f; prow[c + 1] = 0.f; prow[D + c] = 0.f; prow[D + c + 1] = 0.f; }
  }
  uint32_t seq = 0;
  for (int i = 0; i < n_local; ++i) {
    const int r0 = (cta + i * ncta) * C::ROWS + 64 * cw + rf;   // global rows r0, r0 + 8
    const bool ok0 = r0 < M, ok1 = r0 + 8 < M;
    // ---------------------------------------------------------------- dy -> TF32 A fragments (k-step ks: columns 8 ks + 2q, + 1 of rows r0, r0 + 8)
    uint32_t fy[D / 8][4];
#pragma unroll
    for (int ks = 0; ks < D / 8; ++ks) {
      const float2 v0 = ok0 ? ldg64f(dy + (size_t)r0 * D + 8 * ks + 2 * q) : make_float2(0.f, 0.f);
      const float2 v1 = ok1 ? ldg64f(dy + (size_t)(r0 + 8) * D + 8 * ks + 2 * q) : make_float2(0.f, 0.f);
      fy[ks][0] = tf32u(v0.x); fy[ks][1] = tf32u(v1.x); fy[ks][2] = tf32u(v0.y); fy[ks][3] = tf32u(v1.y);
    }
    float acc[64];                                        // d_xn [64 rows][D = 128]
#pragma unroll
    for (int e = 0; e < 64; ++e) acc[e] = 0.f;
    mbar_wait(x_full, i & 1u);
#pragma unroll 1
    for (int j = 0; j < NCH; ++j, ++seq) {
      const int s = (int)(seq & 1);
      const uint32_t slot = su + s * C::F_SLOT;
      mbar_wait(w_full + s, (seq >> 1) & 1u);
      float dh[16], ab[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) dh[e] = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) ab[e] = 0.f;
      fence_regs(dh); fence_regs(ab); fence_regs(acc);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < D / 8; ++ks) {
        const uint32_t ko = (uint32_t)((ks >> 2) * C::W_BLK + (ks & 3) * 32);
        tf32_rs_n32(dh, fy[ks], smem_desc(slot + C::F_WS + ko, 16, 1024, 1), 1);
        tr90::mma<float, 64>(ab, smem_desc(xs + (uint32_t)((ks >> 2) * 8192 + (ks & 3) * 32), 16, 1024, 1),
                             smem_desc(slot + C::F_WAB + (uint32_t)((ks >> 2) * 2 * C::W_BLK + (ks & 3) * 32), 16, 1024, 1));
      }
      wgmma_commit();
      wgmma_wait<1>();                                    // the previous chunk's d_xn update has retired
      fence_regs(acc);
      if (j > 0 && wtid == 0) mbar_arrive(w_free + ((seq - 1) & 1));
      wgmma_wait<0>(); fence_regs(dh); fence_regs(ab);
      if (j == NCH - 1 && wtid == 0) mbar_arrive(x_free);  // every [a | b] of this tile has read xn
      // SwiGLU backward in the C layout; ab[0 .. 15] = a (columns 0 .. HS - 1 of the n64 product: Wa_j), ab[16 .. 31] = b
      uint32_t fa[HS / 8][4], fb[HS / 8][4];
#pragma unroll
      for (int g = 0; g < HS / 8; ++g) {
        float hv[4], dav[4], dbv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float a = ab[4 * g + e], b = ab[16 + 4 * g + e], d = dh[4 * g + e];
          const float sg = f32io_sigmoid(a), sw = a * sg;
          hv[e] = sw * b;
          dav[e] = d * b * sg * (1.f + a * (1.f - sg));
          dbv[e] = d * sw;
        }
        // C regs (r, 2q), (r, 2q + 1), (r + 8, 2q), (r + 8, 2q + 1) -> A fragment (k = q: col 2q, k = q + 4: col 2q + 1)
        fa[g][0] = tf32u(dav[0]); fa[g][1] = tf32u(dav[2]); fa[g][2] = tf32u(dav[1]); fa[g][3] = tf32u(dav[3]);
        fb[g][0] = tf32u(dbv[0]); fb[g][1] = tf32u(dbv[2]); fb[g][2] = tf32u(dbv[1]); fb[g][3] = tf32u(dbv[3]);
        const int c = j * HS + 8 * g + 2 * q;
        if (STORE && ok0) {
          stg64f(hout + (size_t)r0 * H + c, hv[0], hv[1]);
          stg64f(dab + (size_t)r0 * 2 * H + c, __uint_as_float(fa[g][0]), __uint_as_float(fa[g][2]));
          stg64f(dab + (size_t)r0 * 2 * H + H + c, __uint_as_float(fb[g][0]), __uint_as_float(fb[g][2]));
        }
        if (STORE && ok1) {
          stg64f(hout + (size_t)(r0 + 8) * H + c, hv[2], hv[3]);
          stg64f(dab + (size_t)(r0 + 8) * 2 * H + c, __uint_as_float(fa[g][1]), __uint_as_float(fa[g][3]));
          stg64f(dab + (size_t)(r0 + 8) * 2 * H + H + c, __uint_as_float(fb[g][1]), __uint_as_float(fb[g][3]));
        }
      }
      fence_regs(acc);
      wgmma_fence();
#pragma unroll
      for (int g = 0; g < HS / 8; ++g)                     // K block 0 of [Wa_j; Wb_j]^T: the dA k-steps
        tf32_rs_n128(acc, fa[g], smem_desc(slot + C::F_WT + g * 32, 16, 1024, 1), 1);
#pragma unroll
      for (int g = 0; g < HS / 8; ++g)                     // K block 1: the dB k-steps
        tf32_rs_n128(acc, fb[g], smem_desc(slot + C::F_WT + D * 128 + g * 32, 16, 1024, 1), 1);
      wgmma_commit();
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(w_free + ((seq - 1) & 1));
    // ---------------------------------------------------------------- LayerNorm backward + residual in the fragment layout
    float rs[2], cc[2], sa[2] = {0.f, 0.f}, sb[2] = {0.f, 0.f};
    rs[0] = ok0 ? rstd[r0] : 0.f; cc[0] = ok0 ? c1[r0] : 0.f;
    rs[1] = ok1 ? rstd[r0 + 8] : 0.f; cc[1] = ok1 ? c1[r0 + 8] : 0.f;
    float xh[64];
#pragma unroll
    for (int g = 0; g < D / 8; ++g) {
      const float2 gm = __ldg(reinterpret_cast<const float2*>(gamma + 8 * g + 2 * q));
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        const bool ok = R ? ok1 : ok0;
        const float2 xv = ok ? ldg64f(x + (size_t)(r0 + 8 * R) * D + 8 * g + 2 * q) : make_float2(0.f, 0.f);
        xh[4 * g + 2 * R] = xv.x * rs[R] - cc[R]; xh[4 * g + 2 * R + 1] = xv.y * rs[R] - cc[R];
        const float w0 = gm.x * acc[4 * g + 2 * R], w1 = gm.y * acc[4 * g + 2 * R + 1];
        sa[R] += w0 * xh[4 * g + 2 * R] + w1 * xh[4 * g + 2 * R + 1];
        sb[R] += w0 + w1;
      }
    }
#pragma unroll
    for (int R = 0; R < 2; ++R) {
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) { sa[R] += __shfl_xor_sync(0xffffffffu, sa[R], o); sb[R] += __shfl_xor_sync(0xffffffffu, sb[R], o); }
      sa[R] *= 1.f / D; sb[R] *= 1.f / D;
    }
#pragma unroll
    for (int g = 0; g < D / 8; ++g) {
      const int c = 8 * g + 2 * q;
      const float2 gm = __ldg(reinterpret_cast<const float2*>(gamma + c));
      float pg[2] = {0.f, 0.f}, pb[2] = {0.f, 0.f};
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        const bool ok = R ? ok1 : ok0;
        if (ok) {
          const int r = r0 + 8 * R;
          const float2 dv = ldg64f(dy + (size_t)r * D + c);
          const float d0 = acc[4 * g + 2 * R], d1 = acc[4 * g + 2 * R + 1], x0 = xh[4 * g + 2 * R], x1 = xh[4 * g + 2 * R + 1];
          stg64f(dx + (size_t)r * D + c, (gm.x * d0 - x0 * sa[R] - sb[R]) * rs[R] + dv.x, (gm.y * d1 - x1 * sa[R] - sb[R]) * rs[R] + dv.y);
          pg[0] += d0 * x0; pg[1] += d1 * x1; pb[0] += d0; pb[1] += d1;
        }
      }
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1)
#pragma unroll
        for (int e = 0; e < 2; ++e) { pg[e] += __shfl_xor_sync(0xffffffffu, pg[e], o); pb[e] += __shfl_xor_sync(0xffffffffu, pb[e], o); }
      if (lane < 4) { prow[c] += pg[0]; prow[c + 1] += pg[1]; prow[D + c] += pb[0]; prow[D + c + 1] += pb[1]; }
    }
  }
}

// dgamma / dbeta from the [rows][2][D] partials, summed in a fixed order: a block of 1024 threads per 32 output columns, 32 row groups (rows g,
// g + 32, ...) then the 32 group sums in order
__global__ void __launch_bounds__(1024) f32_part_reduce(const float* __restrict__ part, float* __restrict__ dg, float* __restrict__ db, int rows, int D) {
  __shared__ float red[32][33];
  const int c = blockIdx.x * 32 + (threadIdx.x & 31), g = threadIdx.x >> 5;
  float s = 0.f;
  if (c < 2 * D)
    for (int r = g; r < rows; r += 32) s += part[(size_t)r * 2 * D + c];
  red[g][threadIdx.x & 31] = s;
  __syncthreads();
  if (g == 0 && c < 2 * D) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < 32; ++w) t += red[w][threadIdx.x & 31];
    (c < D ? dg : db)[c < D ? c : c - D] = t;
  }
}
