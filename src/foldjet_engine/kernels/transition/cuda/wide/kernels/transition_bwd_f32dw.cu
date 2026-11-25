// transition_bwd_f32dw.cu -- the fp32 Transition backward's weight gradients at D = 128 (hidden n D) in ONE sm_90a kernel, native TF32:
//
//   dWa = dA^T xn,  dWb = dB^T xn,  dWs = dy^T h          (dh, a, b, h, dA, dB recomputed from dy and xn, as transition_bwd_f32dx.cu)
//
// The DW role of the bf16 D = 128 backward (transition_fused_bwd_sm90a_kernel.cu) in TF32. CTA b owns hidden slice s = b % NSL (HS = 32
// units) and the 128-row tiles r, r + REPL, ... (r = b / NSL); the slice's Ws^T_s / [Wa_s; Wb_s] (48 KB) stay resident. Per tile the producer
// TMA-loads xn and dy; each consumer warpgroup recomputes dh (RS, dy fragments from shared memory) and [a | b] (SS) for its 64 rows, forms h,
// dA, dB (rounded to nearest TF32) and writes them TRANSPOSED to shared memory ([hidden][rows], K-major over the rows: TF32 wgmma reads
// only K-major operands, so the contraction over rows needs rows-contiguous B tiles; the A operands xn^T / dy^T are read from the
// row-major tiles straight into A-fragment registers). Then warpgroup cw accumulates, for the d half [64 cw, 64 cw + 64), over all 128 rows:
//   dW_ab^T [64 d][2 HS] += xn^T [d][rows] [dA | dB] [rows][2 HS]       RS m64n64
//   dW_s    [64 d][HS]   += dy^T [d][rows] h [rows][HS]                RS m64n32
// The accumulators live across the CTA's tiles and are written once as the CTA's partial ([D][3 HS] per CTA); `f32_dw_reduce` sums the
// REPL replicas of each slice in a fixed order and writes dWa, dWb [H][D] and dWs [D][H]: no atomics, replays are bit-identical.
#pragma once
#include "transition_bwd_f32dx.cu"

template <int D, int N>
struct F32DwCfg {
  static constexpr int H = N * D, HS = 32, NSL = H / HS, ROWS = 128, KX = D / 32;
  static constexpr int F_WS = 0;                       // Ws^T_s [HS][D]: KX blocks of [HS rows][128 B]
  static constexpr int F_WAB = KX * HS * 128;          // [Wa_s; Wb_s] [2 HS][D]: KX blocks of [2 HS rows][128 B]
  static constexpr int F_XN = F_WAB + KX * 2 * HS * 128;   // xn tile [2 halves][KX][64 rows][128 B]
  static constexpr int F_DY = F_XN + 2 * KX * 8192;    // dy tile, same layout
  static constexpr int F_HT = F_DY + 2 * KX * 8192;    // h^T [ROWS / 32 K blocks][HS][128 B]
  static constexpr int F_ABT = F_HT + (ROWS / 32) * HS * 128;   // [dA | dB]^T [ROWS / 32][2 HS][128 B]
  static constexpr int F_BAR = F_ABT + (ROWS / 32) * 2 * HS * 128;
  static constexpr int SMEM = F_BAR + 128;
  static_assert(D == 128, "the shared-memory plan is the D = 128 one");
  static_assert(SMEM <= 231424, "shared memory budget");
};

template <int D, int N>
__global__ void __launch_bounds__(384, 1)
transition_bwd_f32dw(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mwst,
                     const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb, float* __restrict__ part, int M, int repl) {
  using C = F32DwCfg<D, N>;
  constexpr int HS = C::HS, KX = C::KX, NSL = C::NSL;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* w_full = bars; uint64_t* t_full = bars + 1; uint64_t* t_free = bars + 2;
  const int s = blockIdx.x % NSL, r = blockIdx.x / NSL;
  const int tiles = (M + C::ROWS - 1) / C::ROWS;
  const int n_local = (tiles > r) ? (tiles - r + repl - 1) / repl : 0;
  if (tid == 0) {
    mbar_init(w_full, 1); mbar_init(t_full, 1); mbar_init(t_free, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    tma_prefetch_desc(&mxn); tma_prefetch_desc(&mdy); tma_prefetch_desc(&mwst); tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb);
    mbar_arrive_expect_tx(w_full, C::F_XN);
    for (int kb = 0; kb < KX; ++kb) {
      tma_load_2d(sm + C::F_WS + kb * HS * 128, &mwst, w_full, kb * 32, s * HS);
      tma_load_2d(sm + C::F_WAB + kb * 2 * HS * 128, &mwa, w_full, kb * 32, s * HS);
      tma_load_2d(sm + C::F_WAB + kb * 2 * HS * 128 + HS * 128, &mwb, w_full, kb * 32, s * HS);
    }
    for (int i = 0; i < n_local; ++i) {
      const int row = (r + i * repl) * C::ROWS;
      if (i > 0) mbar_wait(t_free, (i - 1) & 1u);
      mbar_arrive_expect_tx(t_full, 4 * KX * 8192);
      for (int h = 0; h < 2; ++h)
        for (int kb = 0; kb < KX; ++kb) {
          tma_load_2d(sm + C::F_XN + (h * KX + kb) * 8192, &mxn, t_full, kb * 32, row + 64 * h);
          tma_load_2d(sm + C::F_DY + (h * KX + kb) * 8192, &mdy, t_full, kb * 32, row + 64 * h);
        }
      if (i + 1 < n_local)                                // the next tile -> L2 while this one is consumed (its TMA load waits for it)
        for (int h = 0; h < 2; ++h)
          for (int kb = 0; kb < KX; ++kb) {
            l2_prefetch_box(&mxn, kb * 32, row + repl * C::ROWS + 64 * h);
            l2_prefetch_box(&mdy, kb * 32, row + repl * C::ROWS + 64 * h);
          }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  const int q = lane & 3, rq = lane >> 2, rf = 16 * warp + rq;
  // fp32 element (row, col) of a TMA-loaded [128 rows][D] tile: half row / 64, K block col / 32, 128-byte swizzle
  auto tile_addr = [&](uint32_t base, int row, int col) -> uint32_t {
    return base + (uint32_t)(((row >> 6) * KX + (col >> 5)) * 8192) + swz128((uint32_t)(row & 63), (uint32_t)((col & 31) * 4));
  };
  const uint32_t xs = su + C::F_XN, ys = su + C::F_DY, hts = su + C::F_HT, abts = su + C::F_ABT;
  float dwab[32], dws[16];                                 // dW_ab^T [64 d][2 HS], dW_s [64 d][HS] of this warpgroup's d half
#pragma unroll
  for (int e = 0; e < 32; ++e) dwab[e] = 0.f;
#pragma unroll
  for (int e = 0; e < 16; ++e) dws[e] = 0.f;
  mbar_wait(w_full, 0);
  for (int i = 0; i < n_local; ++i) {
    mbar_wait(t_full, i & 1u);
    // ---------------------------------------------------------------- gate recompute for rows 64 cw .. of the tile
    {
      float dh[16], ab[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) dh[e] = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) ab[e] = 0.f;
      fence_regs(dh); fence_regs(ab);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < D / 8; ++ks) {
        uint32_t fy[4];                                    // dy A fragment: (row rf / rf + 8, col 8 ks + q / + 4), rounded to TF32
        const int rr = 64 * cw + rf, cc = 8 * ks + q;
        fy[0] = tf32u(lds_f(tile_addr(ys, rr, cc))); fy[1] = tf32u(lds_f(tile_addr(ys, rr + 8, cc)));
        fy[2] = tf32u(lds_f(tile_addr(ys, rr, cc + 4))); fy[3] = tf32u(lds_f(tile_addr(ys, rr + 8, cc + 4)));
        tf32_rs_n32(dh, fy, smem_desc(su + C::F_WS + (uint32_t)((ks >> 2) * HS * 128 + (ks & 3) * 32), 16, 1024, 1), 1);
        tr90::mma<float, 64>(ab, smem_desc(xs + (uint32_t)((cw * KX + (ks >> 2)) * 8192 + (ks & 3) * 32), 16, 1024, 1),
                             smem_desc(su + C::F_WAB + (uint32_t)((ks >> 2) * 2 * HS * 128 + (ks & 3) * 32), 16, 1024, 1));
      }
      wgmma_commit();
      wgmma_wait<0>(); fence_regs(dh); fence_regs(ab);
      // h, dA, dB -> transposed tiles: element (hidden k, tile row t) at K block t / 32, row k, byte (t % 32) * 4
#pragma unroll
      for (int g = 0; g < HS / 8; ++g)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float a = ab[4 * g + e], b = ab[16 + 4 * g + e], d = dh[4 * g + e];
          const float sg = f32io_sigmoid(a), sw = a * sg;
          const int k = 8 * g + 2 * q + (e & 1), t = 64 * cw + rf + 8 * (e >> 1);
          const uint32_t tb = (uint32_t)(t >> 5), to = (uint32_t)((t & 31) * 4);
          sts32(hts + tb * HS * 128 + swz128((uint32_t)k, to), tf32u(sw * b));
          sts32(abts + tb * 2 * HS * 128 + swz128((uint32_t)k, to), tf32u(d * b * sg * (1.f + a * (1.f - sg))));
          sts32(abts + tb * 2 * HS * 128 + swz128((uint32_t)(HS + k), to), tf32u(d * sw));
        }
    }
    fence_proxy_async();                                  // the transposed tiles (generic stores) -> the wgmma operand reads
    named_bar_sync(1, 256);                               // both warpgroups' rows are in
    // ---------------------------------------------------------------- dW over the tile's 128 rows, this warpgroup's d half
    fence_regs(dwab); fence_regs(dws);
    wgmma_fence();
#pragma unroll
    for (int kk = 0; kk < C::ROWS / 8; ++kk) {
      // A fragments of xn^T / dy^T: (m = d 64 cw + rf / + 8, k = tile row 8 kk + q / + 4)
      const int dm = 64 * cw + rf, tk = 8 * kk + q;
      uint32_t fx[4], fd[4];
      fx[0] = __float_as_uint(lds_f(tile_addr(xs, tk, dm)));     fx[1] = __float_as_uint(lds_f(tile_addr(xs, tk, dm + 8)));
      fx[2] = __float_as_uint(lds_f(tile_addr(xs, tk + 4, dm))); fx[3] = __float_as_uint(lds_f(tile_addr(xs, tk + 4, dm + 8)));
      fd[0] = tf32u(lds_f(tile_addr(ys, tk, dm)));     fd[1] = tf32u(lds_f(tile_addr(ys, tk, dm + 8)));
      fd[2] = tf32u(lds_f(tile_addr(ys, tk + 4, dm))); fd[3] = tf32u(lds_f(tile_addr(ys, tk + 4, dm + 8)));
      const uint32_t ko = (uint32_t)((kk & 3) * 32);
      tf32_rs_n64(dwab, fx, smem_desc(abts + (uint32_t)((kk >> 2) * 2 * HS * 128) + ko, 16, 1024, 1), 1);
      tf32_rs_n32(dws, fd, smem_desc(hts + (uint32_t)((kk >> 2) * HS * 128) + ko, 16, 1024, 1), 1);
    }
    wgmma_commit();
    wgmma_wait<0>(); fence_regs(dwab); fence_regs(dws);
    if (wtid == 0) mbar_arrive(t_free);                   // this warpgroup is done with the xn / dy tiles
    named_bar_sync(1, 256);                               // ... and with the transposed tiles: the next tile may overwrite them
  }
  // ---------------------------------------------------------------- the CTA's partial: [D][3 HS] (dWa^T | dWb^T | dWs) of slice s
  float* p = part + (size_t)blockIdx.x * D * 3 * HS;
#pragma unroll
  for (int R = 0; R < 2; ++R) {
    const int d = 64 * cw + rf + 8 * R;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int c = 8 * g + 2 * q;                         // columns of the n64 accumulator: dA (0 .. 31) | dB (32 .. 63)
      *reinterpret_cast<float2*>(p + (size_t)d * 3 * HS + c) = make_float2(dwab[4 * g + 2 * R], dwab[4 * g + 2 * R + 1]);
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int c = 8 * g + 2 * q;
      *reinterpret_cast<float2*>(p + (size_t)d * 3 * HS + 2 * HS + c) = make_float2(dws[4 * g + 2 * R], dws[4 * g + 2 * R + 1]);
    }
  }
}

// dWa, dWb [H][D] and dWs [D][H] from the partials [NSL * repl][D][3 HS] (CTA b = slice b % NSL, replica b / NSL): replicas summed in order
__global__ void __launch_bounds__(256) f32_dw_reduce(const float* __restrict__ part, float* __restrict__ dwa, float* __restrict__ dwb,
                                                    float* __restrict__ dws, int D, int H, int repl) {
  constexpr int HS = 32;
  const int nsl = H / HS;
  const long long n = (long long)D * 3 * H;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const int c = (int)(i % (3 * HS)), d = (int)((i / (3 * HS)) % D), sl = (int)(i / (3 * HS * D));
    float acc = 0.f;
    for (int r = 0; r < repl; ++r) acc += part[((size_t)(r * nsl + sl) * D + d) * 3 * HS + c];
    if (c < HS) dwa[(size_t)(sl * HS + c) * D + d] = acc;
    else if (c < 2 * HS) dwb[(size_t)(sl * HS + c - HS) * D + d] = acc;
    else dws[(size_t)d * H + sl * HS + c - 2 * HS] = acc;
  }
}
