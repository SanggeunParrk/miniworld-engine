// adaln_sm90.cu -- AdaptiveLayerNorm on sm_90a:   y = sigmoid(LN(c) Wscale^T + b) * LN(x) + LN(c) Wbias^T,   LN(c) = (c - mean) rstd * lnw.
//
//   ln_rows:     one warp per row: aff = LN(cond) * lnw (bf16, or fp32 on the TF32 grid) and the statistics of x (mean, rstd); the training
//                forward also keeps the statistics of cond. One launch, whatever the number of rows.
//   adaln_gemm:  [scale | bias] = aff [Wscale | Wbias]^T as ONE GEMM (the tile's B rows are HT of Wscale then the same HT of Wbias), the
//                gate and the LayerNorm of x in its epilogue: x is TMA-staged by the producer during the main loop, updated in place in the
//                accumulator layout and TMA-stored; the training forward also stores gate = sigmoid(scale + b).
// Structure: ct_sm90.cu (producer warpgroup + 2 consumer warpgroups, mbarrier ring, persistent grid, PDL: the weight slabs are requested
// before ``griddepcontrol.wait``, the activations and the statistics after it).
#include "sm90_common.cuh"

using namespace ctk;

constexpr int TBM = 128;

// ======================================================================================================================== LayerNorm rows
template <bool F32> constexpr int vec_of() { return F32 ? 4 : 8; }

// load a row chunk-strided (chunk = VEC contiguous elements, lane owns chunks lane, lane + 32, ...), as floats; unowned chunks are 0
template <bool F32, int MAXCH>
TMN_DEVI void load_row(const T<F32>* row, int N, int lane, float (&v)[MAXCH][vec_of<F32>()]) {
  constexpr int VEC = vec_of<F32>();
#pragma unroll
  for (int i = 0; i < MAXCH; ++i) {
    const int c = lane + 32 * i;
    if (c * VEC < N) {
      if constexpr (F32) {
        const float4 q = *reinterpret_cast<const float4*>(row + c * VEC);
        v[i][0] = q.x; v[i][1] = q.y; v[i][2] = q.z; v[i][3] = q.w;
      } else {
        const uint4 q = *reinterpret_cast<const uint4*>(row + c * VEC);
        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) { v[i][2 * j] = bf16lo(w[j]); v[i][2 * j + 1] = bf16hi(w[j]); }
      }
    } else {
#pragma unroll
      for (int j = 0; j < VEC; ++j) v[i][j] = 0.f;
    }
  }
}

TMN_DEVI float warp_sum(float s) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

// mean and rstd of a row held in ``v`` (two-pass, centred, fp32)
template <bool F32, int MAXCH>
TMN_DEVI void row_stats(const float (&v)[MAXCH][vec_of<F32>()], int N, int lane, float eps, float& mean, float& rstd) {
  constexpr int VEC = vec_of<F32>();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < MAXCH; ++i)
#pragma unroll
    for (int j = 0; j < VEC; ++j) s += v[i][j];
  mean = warp_sum(s) / (float)N;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < MAXCH; ++i) {
    const bool own = (lane + 32 * i) * VEC < N;
#pragma unroll
    for (int j = 0; j < VEC; ++j) { const float d = v[i][j] - mean; q += own ? d * d : 0.f; }
  }
  rstd = rsqrtf(warp_sum(q) / (float)N + eps);
}

template <bool F32, bool SAVE_C>
__global__ void __launch_bounds__(256)
ln_rows_k(const T<F32>* __restrict__ cond, const float* __restrict__ lnw, const T<F32>* __restrict__ x, T<F32>* __restrict__ aff,
          float2* __restrict__ sx, float2* __restrict__ sc, int M, int NC, int NX, float eps_c,
          float eps_x) {
  constexpr int VEC = vec_of<F32>(), MAXCH = F32 ? 8 : 4;
  pdl_trigger();
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float vx[MAXCH][VEC], v[MAXCH][VEC];
  load_row<F32, MAXCH>(x + (int64_t)row * NX, NX, lane, vx);       // both rows are requested before either is reduced
  load_row<F32, MAXCH>(cond + (int64_t)row * NC, NC, lane, v);
  {
    float mean, rstd;
    row_stats<F32, MAXCH>(vx, NX, lane, eps_x, mean, rstd);
    if (lane == 0) sx[row] = make_float2(mean, rstd);
  }
  float mean, rstd;
  row_stats<F32, MAXCH>(v, NC, lane, eps_c, mean, rstd);
  if (SAVE_C && lane == 0) sc[row] = make_float2(mean, rstd);
#pragma unroll
  for (int i = 0; i < MAXCH; ++i) {
    const int c = lane + 32 * i;
    if (c * VEC < NC) {
      float o[VEC];
#pragma unroll
      for (int j4 = 0; j4 < VEC / 4; ++j4) {
        const float4 w = *reinterpret_cast<const float4*>(lnw + c * VEC + 4 * j4);
        o[4 * j4 + 0] = (v[i][4 * j4 + 0] - mean) * rstd * w.x;
        o[4 * j4 + 1] = (v[i][4 * j4 + 1] - mean) * rstd * w.y;
        o[4 * j4 + 2] = (v[i][4 * j4 + 2] - mean) * rstd * w.z;
        o[4 * j4 + 3] = (v[i][4 * j4 + 3] - mean) * rstd * w.w;
      }
      T<F32>* dst = aff + (int64_t)row * NC + c * VEC;
      if constexpr (F32) {
        *reinterpret_cast<float4*>(dst) = make_float4(tf32_rna(o[0]), tf32_rna(o[1]), tf32_rna(o[2]), tf32_rna(o[3]));
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]), pack_bf16(o[4], o[5]), pack_bf16(o[6], o[7]));
      }
    }
  }
}

template <bool F32, bool SAVE_C>
void run_ln(const at::Tensor& cond, const at::Tensor& lnw, const at::Tensor& x, const at::Tensor& aff, const at::Tensor& sx, const at::Tensor* sc, double eps_c, double eps_x) {
  const int M = (int)cond.size(0), NC = (int)cond.size(1), NX = (int)x.size(1);
  launch(ln_rows_k<F32, SAVE_C>, (unsigned)((M + 7) / 8), 256u, (size_t)0, false, (const T<F32>*)cond.data_ptr(), (const float*)lnw.data_ptr(),
         (const T<F32>*)x.data_ptr(), (T<F32>*)aff.data_ptr(), (float2*)sx.data_ptr(),
         SAVE_C ? (float2*)sc->data_ptr() : (float2*)nullptr, M, NC, NX, (float)eps_c, (float)eps_x);
}

// ======================================================================================================================== GEMM + gate
template <bool F32, int HT, bool SAVE_GATE> struct ACfg {
  static constexpr int TBK = F32 ? 32 : 64;
  static constexpr int TBN = 2 * HT;
  static constexpr int F_SA = TBM * 128, F_SB = TBN * 128, F_ST = F_SA + F_SB;
  static constexpr int STGW = 64 * HT * (F32 ? 4 : 2);
  static constexpr int NSTG = SAVE_GATE ? 2 : 1;                           // staging tiles per consumer warpgroup: y (in place over x), gate
  static constexpr int NST0 = (225 * 1024 - 2 * NSTG * STGW - 1024) / F_ST;
  static constexpr int NST = NST0 > 6 ? 6 : NST0;
  static constexpr int STG = NST * F_ST, BAR = STG + 2 * NSTG * STGW, SMEM = BAR + 256;
  static_assert(SMEM + 1024 <= 232448, "shared memory budget");
};
template <bool F32, int HT> TMN_DEVI void mma_a(float (&acc)[HT], uint64_t a, uint64_t b) {
  if constexpr (F32) { if constexpr (HT == 128) awg_tf32_n256(acc, a, b); else awg_tf32_n128(acc, a, b); }
  else { if constexpr (HT == 128) awg_bf16_n256(acc, a, b); else awg_bf16_n128(acc, a, b); }
}

template <bool F32, int HT, bool SAVE_GATE, bool RNA>
__global__ void __launch_bounds__(384, 1)
adaln_gemm_k(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB, const __grid_constant__ CUtensorMap mX,
             const __grid_constant__ CUtensorMap mY, const __grid_constant__ CUtensorMap mG, const float* __restrict__ bs,
             const float2* __restrict__ st, int M, int NC, int NX) {
  using C = ACfg<F32, HT, SAVE_GATE>;
  constexpr int NSTAGE = C::NST, TBK = C::TBK, TBN = C::TBN, F_SA = C::F_SA, F_ST = C::F_ST, NQ = HT / (F32 ? 32 : 64);
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::BAR);
  uint64_t* empty = full + NSTAGE;
  uint64_t* xfull = empty + NSTAGE;                        // [2]: the x tile of each consumer warpgroup has landed
  uint64_t* xfree = xfull + 2;                             // [2]: that warpgroup's staging tiles may be refilled
  const int MT = (M + TBM - 1) / TBM, NT = NX / HT, KB = NC / TBK, ntiles = MT * NT;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    for (int c = 0; c < 2; ++c) { mbar_init(xfull + c, 1); mbar_init(xfree + c, 1); }
    fence_barrier_init();
  }
  __syncthreads();
  pdl_trigger();
  if (tid == 0) { TRACE(0); TRACE_T(14); }
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mA); tma_prefetch_desc(&mB);
    uint32_t it = 0, tl = 0;
    bool first = true;
    auto load_x = [&](int m0, int n0, uint32_t tl_) {
      for (int c = 0; c < 2; ++c) {
        if (tl_ > 0) mbar_wait(xfree + c, (tl_ - 1) & 1u);
        uint8_t* xs = sm + C::STG + c * C::NSTG * C::STGW;
        mbar_arrive_expect_tx(xfull + c, C::STGW);
        for (int q = 0; q < NQ; ++q) tma_load_2d(xs + q * 8192, &mX, xfull + c, n0 + q * (F32 ? 32 : 64), m0 + 64 * c);
      }
    };
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++tl) {
      const int m0 = (t % MT) * TBM, nt = t / MT, n0 = nt * HT;
      int kb0 = 0;
      if (first) {
        first = false;
        const int P = KB < NSTAGE ? KB : NSTAGE;
        for (int kb = 0; kb < P; ++kb) {
          mbar_arrive_expect_tx(full + kb, F_ST);
          tma_load_2d(sm + kb * F_ST + F_SA, &mB, full + kb, kb * TBK, n0);
          tma_load_2d(sm + kb * F_ST + F_SA + HT * 128, &mB, full + kb, kb * TBK, NX + n0);
        }
        pdl_wait();
        TRACE(1);
        load_x(m0, n0, tl);
        for (int kb = 0; kb < P; ++kb) tma_load_2d(sm + kb * F_ST, &mA, full + kb, kb * TBK, m0);
        it = P; kb0 = P;
      } else {
        load_x(m0, n0, tl);
      }
      for (int kb = kb0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        mbar_arrive_expect_tx(full + s, F_ST);
        tma_load_2d(sm + s * F_ST, &mA, full + s, kb * TBK, m0);
        tma_load_2d(sm + s * F_ST + F_SA, &mB, full + s, kb * TBK, n0);
        tma_load_2d(sm + s * F_ST + F_SA + HT * 128, &mB, full + s, kb * TBK, NX + n0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + C::STG + cw * C::NSTG * C::STGW;    // y (over x)
  uint8_t* gst = stg + C::STGW;                           // gate (SAVE_GATE)
  uint32_t it = 0, tl = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++tl) {
    const int m0 = (t % MT) * TBM, nt = t / MT, n0 = nt * HT;
    // the statistics are the predecessor's (ln_rows) output: wait for it, then keep them and the scale bias in registers before the main loop
    // (a global load inside the epilogue, behind its shared-memory asm statements, is a full memory latency per use)
    pdl_wait();
    const int ra = m0 + 64 * cw + 16 * warp + (lane >> 2), rb = ra + 8;
    const float2 stA = ra < M ? st[ra] : make_float2(0.f, 0.f), stB = rb < M ? st[rb] : make_float2(0.f, 0.f);
    const float meanA = stA.x, rstdA = stA.y, meanB = stB.x, rstdB = stB.y;
    float2 bv[HT / 8];
#pragma unroll
    for (int g = 0; g < HT / 8; ++g) bv[g] = *reinterpret_cast<const float2*>(bs + n0 + 8 * g + 2 * (lane & 3));
    float acc[HT];
#pragma unroll
    for (int e = 0; e < HT; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      if (tid == 128 && it == 0) TRACE(2);
      const uint32_t a0 = su + s * F_ST + cw * 8192, b0 = su + s * F_ST + F_SA;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_a<F32, HT>(acc, dsw(a0 + ks * 32), dsw(b0 + ks * 32));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (tid == 128 && tl == 0) TRACE(3);
    if (wtid == 0) mbar_arrive(empty + prev);
    mbar_wait(xfull + cw, tl & 1u);
    if (tid == 128 && tl == 0) TRACE(4);
    if constexpr (SAVE_GATE) {                                // the gate staging: free once the previous tile's store has read it
      if (tl > 0 && wtid == 0) tma_store_wait_read<0>();
      named_bar_sync(1 + cw, 128);
    }
    if constexpr (F32) {
      const int r0 = 16 * warp + (lane >> 2), c2 = 2 * (lane & 3);
#pragma unroll
      for (int g = 0; g < HT / 8; ++g) {
        const float2 bb = bv[g];
#pragma unroll
        for (int rbk = 0; rbk < 2; ++rbk) {
          const int ia = 4 * g + 2 * rbk, ib = 4 * (g + HT / 8) + 2 * rbk, c = 8 * g + c2;
          const uint32_t off = (c >> 5) * 8192 + swz128((uint32_t)(r0 + 8 * rbk), (uint32_t)((c & 31) * 4));
          float2* p = reinterpret_cast<float2*>(stg + off);
          const float2 xv = *p;
          const float mn = rbk ? meanB : meanA, rs = rbk ? rstdB : rstdA;
          const float g0 = sigmoid_exact(acc[ia] + bb.x), g1 = sigmoid_exact(acc[ia + 1] + bb.y);
          float y0 = g0 * ((xv.x - mn) * rs) + acc[ib], y1 = g1 * ((xv.y - mn) * rs) + acc[ib + 1];
          if (RNA) { y0 = tf32_rna(y0); y1 = tf32_rna(y1); }
          *p = make_float2(y0, y1);
          if constexpr (SAVE_GATE) *reinterpret_cast<float2*>(gst + off) = make_float2(acc[ia] + bb.x, acc[ia + 1] + bb.y);
        }
      }
    } else {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < HT / 16; ++gp) {
        const int col = 8 * (2 * (gp & 3) + (mi >> 1));
        const uint32_t off = (gp >> 2) * 8192 + swz128((uint32_t)mrow, (uint32_t)(col * 2));
        uint32_t xr[4];
        ldsm_x4(xr, smem_u32(stg) + off);
        const float2 b0 = bv[2 * gp], b1 = bv[2 * gp + 1];
        const int g0 = 2 * gp, g1 = 2 * gp + 1;
        float gt[8], pre[8], y[8];
        const float xs[8] = {bf16lo(xr[0]), bf16hi(xr[0]), bf16lo(xr[1]), bf16hi(xr[1]), bf16lo(xr[2]), bf16hi(xr[2]), bf16lo(xr[3]), bf16hi(xr[3])};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int g = q < 4 ? g0 : g1, e = q & 3;                                  // e: 0,1 row A cols; 2,3 row B cols
          const float bcol = (q < 4 ? (e & 1 ? b0.y : b0.x) : (e & 1 ? b1.y : b1.x));
          const int ia = 4 * g + e, ib = 4 * (g + HT / 8) + e;
          const float mn = (e >> 1) ? meanB : meanA, rs = (e >> 1) ? rstdB : rstdA;
          pre[q] = acc[ia] + bcol; gt[q] = sigmoid_exact(pre[q]);
          y[q] = gt[q] * ((xs[q] - mn) * rs) + acc[ib];
        }
        stsm_x4(smem_u32(stg) + off, pack_bf16(y[0], y[1]), pack_bf16(y[2], y[3]), pack_bf16(y[4], y[5]), pack_bf16(y[6], y[7]));
        if constexpr (SAVE_GATE)
          stsm_x4(smem_u32(gst) + off, pack_bf16(pre[0], pre[1]), pack_bf16(pre[2], pre[3]), pack_bf16(pre[4], pre[5]), pack_bf16(pre[6], pre[7]));
      }
    }
    if (tid == 128 && tl == 0) TRACE(5);
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      for (int q = 0; q < NQ; ++q) tma_store_2d(&mY, stg + q * 8192, n0 + q * (F32 ? 32 : 64), m0 + 64 * cw);
      if constexpr (SAVE_GATE)
        for (int q = 0; q < NQ; ++q) tma_store_2d(&mG, gst + q * 8192, n0 + q * (F32 ? 32 : 64), m0 + 64 * cw);
      tma_store_commit();
      if (tid == 128) TRACE(6);
      tma_store_wait_read<0>();                             // the stores have read the staging tiles: the producer may refill the x tile
      if (tid == 128) TRACE(7);
      mbar_arrive(xfree + cw);
    }
  }
  if (wtid == 0) tma_store_wait_all();
  if (tid == 128) { TRACE(8); TRACE_T(15); }
}

template <bool F32, int HT, bool SG, bool RNA>
void run_gemm(const at::Tensor& aff, const at::Tensor& wp, const at::Tensor& bs, const at::Tensor& x, const at::Tensor& st,
              const at::Tensor& y, const at::Tensor* gate, int sms, bool pdl) {
  using C = ACfg<F32, HT, SG>;
  const int64_t M = x.size(0), NX = x.size(1), NC = aff.size(1);
  TORCH_CHECK(NC % C::TBK == 0 && NX % HT == 0, "shapes: NC % (64 bf16 / 32 fp32), NX % HT");
  const int bc = F32 ? 32 : 64;
  const auto mA = map2d(aff.data_ptr(), F32, M, NC, aff.stride(0), C::TBK, TBM), mB = map2d(wp.data_ptr(), F32, 2 * NX, NC, NC, C::TBK, HT),
             mX = map2d(x.data_ptr(), F32, M, NX, x.stride(0), bc, 64), mY = map2d(y.data_ptr(), F32, M, NX, NX, bc, 64),
             mG = map2d(SG ? gate->data_ptr() : y.data_ptr(), F32, M, NX, 2 * NX, bc, 64);
  const int tiles = (int)(((M + TBM - 1) / TBM) * (NX / HT));
  auto k = adaln_gemm_k<F32, HT, SG, RNA>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM + 1024);
  launch(k, (unsigned)std::min(tiles, sms), 384u, (size_t)C::SMEM + 1024, pdl, mA, mB, mX, mY, mG, (const float*)bs.data_ptr(), (const float2*)st.data_ptr(), (int)M, (int)NC, (int)NX);
}

// ======================================================================================================================== host entry points
// cond [M, NC], lnw [NC] fp32, x [M, NX]; writes aff [M, NC], mean / rstd of x [M]; save_c: also mean / rstd of cond
void ln_rows(torch::Tensor cond, torch::Tensor lnw, torch::Tensor x, torch::Tensor aff, torch::Tensor sx, c10::optional<torch::Tensor> sc_, double eps_c, double eps_x) {
  const auto dt = cond.scalar_type();
  TORCH_CHECK((dt == torch::kBFloat16 || dt == torch::kFloat32) && x.scalar_type() == dt && aff.scalar_type() == dt && lnw.scalar_type() == torch::kFloat32,
              "bf16 or fp32 activations, fp32 weight");
  TORCH_CHECK(cond.is_contiguous() && x.is_contiguous() && aff.is_contiguous() && lnw.is_contiguous(), "contiguous");
  const int vec = dt == torch::kFloat32 ? 4 : 8;
  TORCH_CHECK(cond.size(1) % vec == 0 && x.size(1) % vec == 0 && cond.size(1) <= 1024 && x.size(1) <= 1024, "widths: multiple of 8 (4 fp32), <= 1024");
  const at::cuda::CUDAGuard guard(cond.device());
  const bool f32 = dt == torch::kFloat32, sc = sc_.has_value();
  const at::Tensor* pm = sc ? &*sc_ : nullptr;
  if (f32) { if (sc) run_ln<true, true>(cond, lnw, x, aff, sx, pm, eps_c, eps_x); else run_ln<true, false>(cond, lnw, x, aff, sx, pm, eps_c, eps_x); }
  else { if (sc) run_ln<false, true>(cond, lnw, x, aff, sx, pm, eps_c, eps_x); else run_ln<false, false>(cond, lnw, x, aff, sx, pm, eps_c, eps_x); }
}

// aff [M, NC], wp [2 NX, NC] packed per 2 HT rows (HT rows of Wscale, then the same HT rows of Wbias), bs [NX] fp32 (scale bias), x [M, NX],
// mean / rstd [M] fp32, y [M, NX], gate [M, NX] or undefined (training); rna: y rounded to the TF32 grid (fp32 only: the next GEMM reads it)
void adaln_gemm(torch::Tensor aff, torch::Tensor wp, torch::Tensor bs, torch::Tensor x, torch::Tensor st, torch::Tensor y,
                c10::optional<torch::Tensor> gate, int64_t sms, int64_t ht, bool rna, bool pdl) {
  const auto dt = x.scalar_type();
  TORCH_CHECK((dt == torch::kBFloat16 || dt == torch::kFloat32) && aff.scalar_type() == dt && wp.scalar_type() == dt && y.scalar_type() == dt &&
              bs.scalar_type() == torch::kFloat32 && st.scalar_type() == torch::kFloat32 && st.size(-1) == 2, "dtypes");
  TORCH_CHECK(x.stride(1) == 1 && aff.stride(1) == 1 && wp.is_contiguous() && y.is_contiguous() && bs.is_contiguous() && wp.size(0) == 2 * x.size(1) &&
              wp.size(1) == aff.size(1), "layouts");
  if (gate) TORCH_CHECK(gate->scalar_type() == dt && gate->is_contiguous() && gate->size(0) == y.size(0) && gate->size(1) == 2 * y.size(1), "pre-activation buffer [M, 2 NX]");
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(ht == 128 || ht == 64, "ht: 128 or 64");
  const bool f32 = dt == torch::kFloat32, sg = gate.has_value();
  const at::Tensor* gp = sg ? &*gate : nullptr;
#define AG(F, H, S, R) run_gemm<F, H, S, R>(aff, wp, bs, x, st, y, gp, sms, pdl)
  if (f32) {
    if (ht == 64) { if (sg) { if (rna) AG(true, 64, true, true); else AG(true, 64, true, false); } else { if (rna) AG(true, 64, false, true); else AG(true, 64, false, false); } }
    else { if (sg) { if (rna) AG(true, 128, true, true); else AG(true, 128, true, false); } else { if (rna) AG(true, 128, false, true); else AG(true, 128, false, false); } }
  } else {
    if (ht == 64) { if (sg) AG(false, 64, true, false); else AG(false, 64, false, false); }
    else { if (sg) AG(false, 128, true, false); else AG(false, 128, false, false); }
  }
#undef AG
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_rows", &ln_rows);
  m.def("adaln_gemm", &adaln_gemm);
#ifdef CT_TRACE
  m.def("trace", &trace_read);
#endif
}
