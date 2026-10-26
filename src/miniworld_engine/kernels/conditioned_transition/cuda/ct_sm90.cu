// ct_sm90.cu -- the two GEMMs of the ConditionedTransition tail on sm_90a (bf16 on wgmma k16, fp32 on the TF32 tensor cores).
//
//   expand_swiglu:  h = silu(x Wa^T) * (x Wb^T)               one GEMM, [a | b] never reaches HBM (``gemm_swiglu_sm90.cu``'s structure)
//   squeeze_gate:   y = sigmoid(cond Wsc^T + b) * (h Ws^T)    (+ residual)   two accumulators over two K ranges, the gate and the residual
//                                                              in the epilogue: h Ws^T and cond Wsc^T share one pass over the tile
//
// Both: a producer warpgroup (setmaxnreg 40) streams 128-B-swizzled A [128 rows] / B slabs by TMA through an NST ring; two consumer warpgroups
// (setmaxnreg 232) each own 64 rows (wgmma m64nN); the epilogue stages the tile in shared memory and TMA-stores it. Persistent grid.
// Programmatic dependent launch: the weight slabs (B) of the first NST stages are requested before ``griddepcontrol.wait``, so they are in
// flight while the previous kernel (the AdaLN GEMM) is still finishing; the activations are read after it.
#include "sm90_common.cuh"

using namespace ctk;

constexpr int TBM = 128;

// ======================================================================================================================== expand + SwiGLU
// B is Wab packed per tile: tile t = [Wa rows HT t .. HT t + HT - 1 | Wb rows HT t .. HT t + HT - 1] (2 HT rows), so one 128 x 2 HT output tile
// holds the a AND b of HT hidden units and its epilogue writes h[:, HT t : HT t + HT].
template <bool F32, int HT> struct ECfg {
  static constexpr int TBK = F32 ? 32 : 64;
  static constexpr int TBN = 2 * HT;
  static constexpr int F_SA = TBM * 128, F_SB = TBN * 128, F_ST = F_SA + F_SB;
  static constexpr int STGW = 64 * HT * (F32 ? 4 : 2);
  static constexpr int NST = (225 * 1024 - 2 * STGW - 1024) / F_ST > 6 ? 6 : (225 * 1024 - 2 * STGW - 1024) / F_ST;
  static constexpr int STG = NST * F_ST, BAR = STG + 2 * STGW, SMEM = BAR + 256;
  static_assert(SMEM + 1024 <= 232448, "shared memory budget");
  static_assert(HT == 128 || HT == 64, "HT");
};

template <bool F32, int HT> TMN_DEVI void mma_e(float (&acc)[HT], uint64_t a, uint64_t b) {
  if constexpr (F32) { if constexpr (HT == 128) awg_tf32_n256(acc, a, b); else awg_tf32_n128(acc, a, b); }
  else { if constexpr (HT == 128) awg_bf16_n256(acc, a, b); else awg_bf16_n128(acc, a, b); }
}

template <bool F32, int HT, bool RND>
__global__ void __launch_bounds__(384, 1)
expand_swiglu_k(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB,
                const __grid_constant__ CUtensorMap mH, int M, int K, int H) {
  using C = ECfg<F32, HT>;
  constexpr int NSTAGE = C::NST, TBK = C::TBK, TBN = C::TBN, F_SA = C::F_SA, F_ST = C::F_ST;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::BAR);
  uint64_t* empty = full + NSTAGE;
  const int MT = (M + TBM - 1) / TBM, NT = H / HT, KB = K / TBK, ntiles = MT * NT;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  pdl_trigger();
  if (wg == 0) {                                                          // ---------------------------------- producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mA); tma_prefetch_desc(&mB);
    uint32_t it = 0;
    bool first = true;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      const int m0 = (t % MT) * TBM, n0 = (t / MT) * TBN;
      int kb0 = 0;
      if (first) {                                                        // weights first, then the activations once the predecessor is done
        first = false;
        const int P = KB < NSTAGE ? KB : NSTAGE;
        for (int kb = 0; kb < P; ++kb) {
          mbar_arrive_expect_tx(full + kb, F_ST);
          tma_load_2d(sm + kb * F_ST + F_SA, &mB, full + kb, kb * TBK, n0);
        }
        pdl_wait();
        for (int kb = 0; kb < P; ++kb) tma_load_2d(sm + kb * F_ST, &mA, full + kb, kb * TBK, m0);
        it = P; kb0 = P;
      }
      for (int kb = kb0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        mbar_arrive_expect_tx(full + s, F_ST);
        tma_load_2d(sm + s * F_ST, &mA, full + s, kb * TBK, m0);
        tma_load_2d(sm + s * F_ST + F_SA, &mB, full + s, kb * TBK, n0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                                  // --------------------------------- consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + C::STG + cw * C::STGW;
  uint32_t it = 0;
  bool firstt = true;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const int m0 = (t % MT) * TBM, nt = t / MT;
    float acc[HT];
#pragma unroll
    for (int e = 0; e < HT; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_ST + cw * 8192, b0 = su + s * F_ST + F_SA;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_e<F32, HT>(acc, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
    if (!firstt && wtid == 0) tma_store_wait_read<0>();                   // the previous tile's store has left the staging tile
    firstt = false;
    named_bar_sync(1 + cw, 128);
    if constexpr (F32) {                                                  // HT / 32 [64][32] fp32 boxes, 128-B rows
      const int r0 = 16 * warp + (lane >> 2), c2 = 2 * (lane & 3);
#pragma unroll
      for (int g = 0; g < HT / 8; ++g)
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int ia = 4 * g + 2 * rb, ib = 4 * (g + HT / 8) + 2 * rb, c = 8 * g + c2;
          const float x0 = acc[ia], x1 = acc[ia + 1];
          float h0 = x0 / (1.f + __expf(-x0)) * acc[ib], h1 = x1 / (1.f + __expf(-x1)) * acc[ib + 1];
          if (RND) { h0 = tf32_rna(h0); h1 = tf32_rna(h1); }
          *reinterpret_cast<float2*>(stg + (c >> 5) * 8192 + swz128((uint32_t)(r0 + 8 * rb), (uint32_t)((c & 31) * 4))) = make_float2(h0, h1);
        }
    } else {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < HT / 16; ++gp) {                              // 16 h columns a chunk
        uint32_t hp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + HT / 8) + 2 * rb;
          const float x0 = acc[ia], x1 = acc[ia + 1];
          hp[q] = pack_bf16(x0 / (1.f + __expf(-x0)) * acc[ib], x1 / (1.f + __expf(-x1)) * acc[ib + 1]);
        }
        const int col = 8 * (2 * (gp & 3) + (mi >> 1));                  // within the 64-column half gp / 4
        stsm_x4(smem_u32(stg + (gp >> 2) * 8192) + swz128((uint32_t)mrow, (uint32_t)(col * 2)), hp[0], hp[1], hp[2], hp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      constexpr int NBOX = F32 ? HT / 32 : HT / 64, BOXC = F32 ? 32 : 64;
#pragma unroll
      for (int b = 0; b < NBOX; ++b) tma_store_2d(&mH, stg + b * 8192, nt * HT + b * BOXC, m0 + 64 * cw);
      tma_store_commit();
    }
  }
  if (wtid == 0) tma_store_wait_all();
}

template <bool F32, int HT, bool RND>
void run_expand(const at::Tensor& x, const at::Tensor& wab, const at::Tensor& h, int sms, bool pdl) {
  using C = ECfg<F32, HT>;
  const int64_t M = x.size(0), K = x.size(1), H = h.size(1);
  TORCH_CHECK(K % C::TBK == 0 && H % HT == 0, "shapes: K % (64 bf16 / 32 fp32), H % HT");
  const auto mA = map2d(x.data_ptr(), F32, M, K, x.stride(0), C::TBK, TBM), mB = map2d(wab.data_ptr(), F32, 2 * H, K, K, C::TBK, C::TBN),
             mH = map2d(h.data_ptr(), F32, M, H, H, F32 ? 32 : 64, 64);
  const int tiles = (int)(((M + TBM - 1) / TBM) * (H / HT));
  auto k = expand_swiglu_k<F32, HT, RND>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM + 1024);
  launch(k, (unsigned)std::min(tiles, sms), 384u, (size_t)C::SMEM + 1024, pdl, mA, mB, mH, (int)M, (int)K, (int)H);
}

// ======================================================================================================================== squeeze + gate (+ residual)
template <bool F32, int BN> struct SCfg {
  static constexpr int TBK = F32 ? 32 : 64;
  static constexpr int F_SA = TBM * 128, F_SB = BN * 128, F_ST = F_SA + F_SB;
  static constexpr int STGW = 64 * BN * (F32 ? 4 : 2);
  static constexpr int NST = (225 * 1024 - 2 * STGW - 1024) / F_ST > 6 ? 6 : (225 * 1024 - 2 * STGW - 1024) / F_ST;
  static constexpr int STG = NST * F_ST, BAR = STG + 2 * STGW, SMEM = BAR + 256;
  static_assert(SMEM + 1024 <= 232448, "shared memory budget");
};
template <bool F32, int BN> TMN_DEVI void mma_s(float (&acc)[BN / 2], uint64_t a, uint64_t b) {
  if constexpr (F32) { if constexpr (BN == 128) awg_tf32_n128(acc, a, b); else if constexpr (BN == 64) awg_tf32_n64(acc, a, b); else awg_tf32_n256(acc, a, b); }
  else { if constexpr (BN == 128) awg_bf16_n128(acc, a, b); else if constexpr (BN == 64) awg_bf16_n64(acc, a, b); else awg_bf16_n256(acc, a, b); }
}
TMN_DEVI float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }

template <bool F32, int BN, bool RES>
__global__ void __launch_bounds__(384, 1)
squeeze_gate_k(const __grid_constant__ CUtensorMap mH, const __grid_constant__ CUtensorMap mWs, const __grid_constant__ CUtensorMap mC,
               const __grid_constant__ CUtensorMap mWsc, const __grid_constant__ CUtensorMap mR, const __grid_constant__ CUtensorMap mO,
               const float* __restrict__ bsc, int M, int ND, int DC, int D) {
  using C = SCfg<F32, BN>;
  constexpr int NSTAGE = C::NST, TBK = C::TBK, F_SA = C::F_SA, F_ST = C::F_ST, NQ = BN / (F32 ? 32 : 64);
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::BAR);
  uint64_t* empty = full + NSTAGE;
  uint64_t* xfull = empty + NSTAGE;                        // [2]: the residual tile of each consumer warpgroup has landed
  uint64_t* xfree = xfull + 2;                             // [2]: that warpgroup's staging tile may be refilled
  const int MT = (M + TBM - 1) / TBM, NT = D / BN, KB1 = ND / TBK, KB2 = DC / TBK, KB = KB1 + KB2, ntiles = MT * NT;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    for (int c = 0; c < 2; ++c) { mbar_init(xfull + c, 1); mbar_init(xfree + c, 1); }
    fence_barrier_init();
  }
  __syncthreads();
  pdl_trigger();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mH); tma_prefetch_desc(&mWs); tma_prefetch_desc(&mC); tma_prefetch_desc(&mWsc);
    uint32_t it = 0, tl = 0;
    bool first = true;
    auto load_a = [&](int s, int kb, int m0) {
      if (kb < KB1) tma_load_2d(sm + s * F_ST, &mH, full + s, kb * TBK, m0);
      else tma_load_2d(sm + s * F_ST, &mC, full + s, (kb - KB1) * TBK, m0);
    };
    auto load_b = [&](int s, int kb, int n0) {
      if (kb < KB1) tma_load_2d(sm + s * F_ST + F_SA, &mWs, full + s, kb * TBK, n0);
      else tma_load_2d(sm + s * F_ST + F_SA, &mWsc, full + s, (kb - KB1) * TBK, n0);
    };
    auto load_x = [&](int m0, int n0, uint32_t tl_) {
      if constexpr (RES) {
        for (int c = 0; c < 2; ++c) {                       // residual tiles: into each warpgroup's staging once it is free
          if (tl_ > 0) mbar_wait(xfree + c, (tl_ - 1) & 1u);
          uint8_t* xs = sm + C::STG + c * C::STGW;
          mbar_arrive_expect_tx(xfull + c, C::STGW);
          for (int q = 0; q < NQ; ++q) tma_load_2d(xs + q * 8192, &mR, xfull + c, n0 + q * (F32 ? 32 : 64), m0 + 64 * c);
        }
      }
    };
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++tl) {
      const int m0 = (t % MT) * TBM, n0 = (t / MT) * BN;    // m fastest: the CTAs of a wave share the weight tile
      int kb0 = 0;
      if (first) {
        first = false;
        const int P = KB < NSTAGE ? KB : NSTAGE;
        for (int kb = 0; kb < P; ++kb) { mbar_arrive_expect_tx(full + kb, F_ST); load_b(kb, kb, n0); }
        pdl_wait();
        load_x(m0, n0, tl);
        for (int kb = 0; kb < P; ++kb) load_a(kb, kb, m0);
        it = P; kb0 = P;
      } else {
        load_x(m0, n0, tl);
      }
      for (int kb = kb0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        mbar_arrive_expect_tx(full + s, F_ST);
        load_a(s, kb, m0);
        load_b(s, kb, n0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + C::STG + cw * C::STGW;
  uint32_t it = 0, tl = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++tl) {
    const int m0 = (t % MT) * TBM, n0 = (t / MT) * BN;
    float2 bv[BN / 8];   // the gate bias in registers before the main loop (a global load inside the epilogue costs a full memory latency per use)
#pragma unroll
    for (int g = 0; g < BN / 8; ++g) bv[g] = *reinterpret_cast<const float2*>(bsc + n0 + 8 * g + 2 * (lane & 3));
    float acc1[BN / 2], acc2[BN / 2];
#pragma unroll
    for (int e = 0; e < BN / 2; ++e) { acc1[e] = 0.f; acc2[e] = 0.f; }
    int prev = -1;
    for (int kb = 0; kb < KB1; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_ST + cw * 8192, b0 = su + s * F_ST + F_SA;
      fence_regs(acc1); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_s<F32, BN>(acc1, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc1); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    for (int kb = 0; kb < KB2; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_ST + cw * 8192, b0 = su + s * F_ST + F_SA;
      fence_regs(acc2); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_s<F32, BN>(acc2, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc1); fence_regs(acc2); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc1); fence_regs(acc2);
    if (wtid == 0) mbar_arrive(empty + prev);
    if constexpr (RES) mbar_wait(xfull + cw, tl & 1u);
    else {
      if (tl > 0 && wtid == 0) tma_store_wait_read<0>();    // the previous tile's store has left the staging tile
      named_bar_sync(1 + cw, 128);
    }
    // ---- y = sigmoid(cond Wsc^T + b) * (h Ws^T) (+ the residual, added after the gated update is rounded to the stream dtype, as the unfused ops do)
    if constexpr (F32) {
      const int r0 = 16 * warp + (lane >> 2), c2 = 2 * (lane & 3);
#pragma unroll
      for (int g = 0; g < BN / 8; ++g) {
        const float2 bb = bv[g];
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int ia = 4 * g + 2 * rb, c = 8 * g + c2;
          float* p = reinterpret_cast<float*>(stg + (c >> 5) * 8192 + swz128((uint32_t)(r0 + 8 * rb), (uint32_t)((c & 31) * 4)));
          float2 v = RES ? *reinterpret_cast<float2*>(p) : make_float2(0.f, 0.f);
          v.x += sigmoid_exact(acc2[ia] + bb.x) * acc1[ia];
          v.y += sigmoid_exact(acc2[ia + 1] + bb.y) * acc1[ia + 1];
          *reinterpret_cast<float2*>(p) = v;
        }
      }
    } else {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < BN / 16; ++gp) {
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t ad = smem_u32(stg) + (col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        uint32_t xr[4] = {0u, 0u, 0u, 0u};
        if constexpr (RES) ldsm_x4(xr, ad);
        const int g0 = 2 * gp, g1 = 2 * gp + 1;
        const float2 b0 = bv[2 * gp], b1 = bv[2 * gp + 1];
        float y[8];
        y[0] = sigmoid_exact(acc2[4 * g0 + 0] + b0.x) * acc1[4 * g0 + 0];
        y[1] = sigmoid_exact(acc2[4 * g0 + 1] + b0.y) * acc1[4 * g0 + 1];
        y[2] = sigmoid_exact(acc2[4 * g0 + 2] + b0.x) * acc1[4 * g0 + 2];
        y[3] = sigmoid_exact(acc2[4 * g0 + 3] + b0.y) * acc1[4 * g0 + 3];
        y[4] = sigmoid_exact(acc2[4 * g1 + 0] + b1.x) * acc1[4 * g1 + 0];
        y[5] = sigmoid_exact(acc2[4 * g1 + 1] + b1.y) * acc1[4 * g1 + 1];
        y[6] = sigmoid_exact(acc2[4 * g1 + 2] + b1.x) * acc1[4 * g1 + 2];
        y[7] = sigmoid_exact(acc2[4 * g1 + 3] + b1.y) * acc1[4 * g1 + 3];
        if constexpr (RES) {
          y[0] = bf16r(y[0]) + bf16lo(xr[0]); y[1] = bf16r(y[1]) + bf16hi(xr[0]);
          y[2] = bf16r(y[2]) + bf16lo(xr[1]); y[3] = bf16r(y[3]) + bf16hi(xr[1]);
          y[4] = bf16r(y[4]) + bf16lo(xr[2]); y[5] = bf16r(y[5]) + bf16hi(xr[2]);
          y[6] = bf16r(y[6]) + bf16lo(xr[3]); y[7] = bf16r(y[7]) + bf16hi(xr[3]);
        }
        stsm_x4(ad, pack_bf16(y[0], y[1]), pack_bf16(y[2], y[3]), pack_bf16(y[4], y[5]), pack_bf16(y[6], y[7]));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      for (int q = 0; q < NQ; ++q) tma_store_2d(&mO, stg + q * 8192, n0 + q * (F32 ? 32 : 64), m0 + 64 * cw);
      tma_store_commit();
      if constexpr (RES) {
        tma_store_wait_read<0>();                           // the store has read the staging tile: the producer may refill it
        mbar_arrive(xfree + cw);
      }
    }
  }
  if (wtid == 0) tma_store_wait_all();
}

template <bool F32, int BN, bool RES>
void run_squeeze(const at::Tensor& h, const at::Tensor& ws, const at::Tensor& cond, const at::Tensor& wsc, const at::Tensor& bsc,
                 const at::Tensor* res, const at::Tensor& out, int sms, bool pdl) {
  using C = SCfg<F32, BN>;
  const int64_t M = h.size(0), ND = h.size(1), D = ws.size(0), DC = cond.size(1);
  TORCH_CHECK(ND % C::TBK == 0 && DC % C::TBK == 0 && D % BN == 0, "shapes: ND, DC % (64 bf16 / 32 fp32), D % BN");
  const int bc = F32 ? 32 : 64;
  const auto mH = map2d(h.data_ptr(), F32, M, ND, h.stride(0), C::TBK, TBM), mWs = map2d(ws.data_ptr(), F32, D, ND, ND, C::TBK, BN),
             mC = map2d(cond.data_ptr(), F32, M, DC, cond.stride(0), C::TBK, TBM), mWsc = map2d(wsc.data_ptr(), F32, D, DC, DC, C::TBK, BN),
             mR = map2d(RES ? res->data_ptr() : out.data_ptr(), F32, M, D, RES ? res->stride(0) : D, bc, 64),
             mO = map2d(out.data_ptr(), F32, M, D, D, bc, 64);
  const int tiles = (int)(((M + TBM - 1) / TBM) * (D / BN));
  auto k = squeeze_gate_k<F32, BN, RES>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM + 1024);
  launch(k, (unsigned)std::min(tiles, sms), 384u, (size_t)C::SMEM + 1024, pdl, mH, mWs, mC, mWsc, mR, mO, (const float*)bsc.data_ptr(), (int)M,
         (int)ND, (int)DC, (int)D);
}

// ======================================================================================================================== host entry points
// x [M, K] (row stride may exceed K), wab [2H, K] packed per 2 HT rows, h [M, H]: all bf16, or all fp32 (TF32 tensor cores; ``rnd``: h rounded to
// the TF32 grid to nearest, which the next GEMM then reads exactly).
void expand_swiglu(torch::Tensor x, torch::Tensor wab, torch::Tensor h, int64_t sms, int64_t ht, bool rnd, bool pdl) {
  const auto dt = x.scalar_type();
  TORCH_CHECK((dt == torch::kBFloat16 || dt == torch::kFloat32) && wab.scalar_type() == dt && h.scalar_type() == dt, "bf16 or fp32 operands, all alike");
  TORCH_CHECK(x.stride(1) == 1 && wab.is_contiguous() && h.is_contiguous() && wab.size(0) == 2 * h.size(1) && wab.size(1) == x.size(1) &&
              h.size(0) == x.size(0), "shapes");
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(ht == 128 || ht == 64, "ht: 128 or 64");
  const bool f32 = dt == torch::kFloat32;
  if (f32) {
    if (ht == 64) { if (rnd) run_expand<true, 64, true>(x, wab, h, sms, pdl); else run_expand<true, 64, false>(x, wab, h, sms, pdl); }
    else { if (rnd) run_expand<true, 128, true>(x, wab, h, sms, pdl); else run_expand<true, 128, false>(x, wab, h, sms, pdl); }
  } else {
    if (ht == 64) run_expand<false, 64, false>(x, wab, h, sms, pdl); else run_expand<false, 128, false>(x, wab, h, sms, pdl);
  }
}

// h [M, ND], ws [D, ND], cond [M, DC], wsc [D, DC], bsc [D] fp32, res [M, D] or undefined, out [M, D]
void squeeze_gate(torch::Tensor h, torch::Tensor ws, torch::Tensor cond, torch::Tensor wsc, torch::Tensor bsc, c10::optional<torch::Tensor> res,
                  torch::Tensor out, int64_t sms, int64_t bn, bool pdl) {
  const auto dt = h.scalar_type();
  TORCH_CHECK((dt == torch::kBFloat16 || dt == torch::kFloat32) && ws.scalar_type() == dt && cond.scalar_type() == dt && wsc.scalar_type() == dt &&
              out.scalar_type() == dt && bsc.scalar_type() == torch::kFloat32, "bf16 or fp32 operands, all alike; bias fp32");
  TORCH_CHECK(h.stride(1) == 1 && cond.stride(1) == 1 && ws.is_contiguous() && wsc.is_contiguous() && out.is_contiguous() && bsc.is_contiguous(), "layouts");
  if (res) TORCH_CHECK(res->scalar_type() == dt && res->stride(1) == 1 && res->sizes() == out.sizes(), "residual");
  const at::cuda::CUDAGuard guard(h.device());
  TORCH_CHECK(bn == 128 || bn == 64, "bn: 128 or 64");
  const bool f32 = dt == torch::kFloat32, r = res.has_value();
  const at::Tensor* rp = r ? &*res : nullptr;
#define SQ(F, B, R) run_squeeze<F, B, R>(h, ws, cond, wsc, bsc, rp, out, sms, pdl)
  if (f32) { if (bn == 64) { if (r) SQ(true, 64, true); else SQ(true, 64, false); } else { if (r) SQ(true, 128, true); else SQ(true, 128, false); } }
  else { if (bn == 64) { if (r) SQ(false, 64, true); else SQ(false, 64, false); } else { if (r) SQ(false, 128, true); else SQ(false, 128, false); } }
#undef SQ
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expand_swiglu", &expand_swiglu);
  m.def("squeeze_gate", &squeeze_gate);
#ifdef CT_TRACE
  m.def("trace", &trace_read);
#endif
}
