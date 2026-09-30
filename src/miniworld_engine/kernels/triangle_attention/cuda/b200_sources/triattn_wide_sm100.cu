// TriangleAttention on B200 for the widths other than d_pair 128: the small row kernels around one cuBLAS projection GEMM, the
// sm_100a attention core (triattn_sm100.cu, head dim 32) and one cuBLAS out-projection (addmm with the residual):
//
//   y = LN(x)                                   tri_ln_rows          x [R, C] bf16 -> y [R, C] bf16   (C = 64 .. 512, a multiple of 64)
//   P = y . [Wq; Wk; Wv; Wg; Wb]^T              cuBLAS               P [R, 4 HDp + 16] bf16
//   bias[b, h, j, k] = P[(b, j, k), 4 HDp + h]  tri_bias_heads       masked keys = finfo(bf16).min, head-major for the core
//   o = attention(q, k, v, bias)                triattn_fwd          q / k / v = column slices of P (rows at stride 4 HDp + 16)
//   u = sigmoid(g) o o                          tri_gate_mul         g = a column slice of P
//   out = x + u . Wo^T                          cuBLAS addmm_ in place on a copy of x the LN kernel wrote while reading x
//
// Training adds the dropout scale in the tail and the backward (tri_scale_rows -> tri_wgbwd -> the core's backward ->
// tri_db_rows -> tri_whbwd -> cuBLAS weight-gradient GEMMs, see b200_triattn.WideTrain).
//
// HDp = heads x 32: a head dim of 16 is zero-padded to 32 in the packed weights (the padded channels of q / k / v / g are 0, the
// padded columns of Wo are 0, and the softmax scale stays 1/sqrt(16)), so every width runs the one head-dim-32 core.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

namespace {

__device__ __forceinline__ float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
__device__ __forceinline__ uint32_t pk2(float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// ---- LayerNorm over C channels, one warp a row: 16-byte chunks j = lane, lane + 32, ... (C / 8 of them), two-pass statistics
template <int C>
__global__ void __launch_bounds__(256) tri_ln_rows(const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
                                                   const float* __restrict__ Bv, __nv_bfloat16* __restrict__ Y, long R, float eps,
                                                   __nv_bfloat16* __restrict__ XC) {   // optional copy of x (the out-projection's residual)
  constexpr int NCH = C / 8, PER = (NCH + 31) / 32;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= R) return;
  const uint4* xr = reinterpret_cast<const uint4*>(X + row * C);
  float v[PER][8];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int j = lane + 32 * i;
    if (j < NCH) {
      const uint4 u = xr[j];
      if (XC != nullptr) reinterpret_cast<uint4*>(XC + row * C)[j] = u;
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = bf2(w[e]); v[i][2 * e] = f.x; v[i][2 * e + 1] = f.y; s += f.x + f.y; }
    }
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s * (1.f / C);
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i)
    if (lane + 32 * i < NCH)
#pragma unroll
      for (int e = 0; e < 8; ++e) { const float d = v[i][e] - mean; q = fmaf(d, d, q); }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rstd = rsqrtf(q * (1.f / C) + eps);
  uint4* yr = reinterpret_cast<uint4*>(Y + row * C);
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int j = lane + 32 * i;
    if (j < NCH) {
      const float4 w0 = reinterpret_cast<const float4*>(W)[2 * j], w1 = reinterpret_cast<const float4*>(W)[2 * j + 1];
      const float4 b0 = reinterpret_cast<const float4*>(Bv)[2 * j], b1 = reinterpret_cast<const float4*>(Bv)[2 * j + 1];
      const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}, bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = pk2((v[i][2 * e] - mean) * rstd * ww[2 * e] + bb[2 * e], (v[i][2 * e + 1] - mean) * rstd * ww[2 * e + 1] + bb[2 * e + 1]);
      yr[j] = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}

// ---- bias[b, h, j, k] = P[(b, j, k), cb + h] (masked keys -> finfo(bf16).min): a thread a row (consecutive rows = consecutive k,
// so every head plane is written coalesced)
__global__ void __launch_bounds__(256) tri_bias_heads(const __nv_bfloat16* __restrict__ P, long ld, int cb, int NH,
                                                      const bool* __restrict__ MASK, __nv_bfloat16* __restrict__ BIAS, int L, long R) {
  const long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  const int kx = (int)(row % L);
  const long bj = row / L;
  const int j = (int)(bj % L), b = (int)(bj / L);
  const bool keep = MASK == nullptr || MASK[(long)b * L + kx];
  const __nv_bfloat16* src = P + row * ld + cb;
  const __nv_bfloat16 lo = __float2bfloat16_rn(-3.3895313892515355e38f);
  for (int h = 0; h < NH; ++h) BIAS[(((long)b * NH + h) * L + j) * L + kx] = keep ? src[h] : lo;
}

// ---- u = sigmoid(g) o o, 8 channels a thread; g rows at stride ldg, o and u dense [R, HD]
__global__ void __launch_bounds__(256) tri_gate_mul(const __nv_bfloat16* __restrict__ G, long ldg, const __nv_bfloat16* __restrict__ O,
                                                    __nv_bfloat16* __restrict__ U, int HD, long R) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;       // 8-channel chunk
  const int per = HD / 8;
  if (i >= R * per) return;
  const long row = i / per;
  const int c = (int)(i % per) * 8;
  const uint4 g = *reinterpret_cast<const uint4*>(G + row * ldg + c), o = *reinterpret_cast<const uint4*>(O + row * HD + c);
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, ow[4] = {o.x, o.y, o.z, o.w};
  uint32_t w[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 gv = bf2(gw[e]), ov = bf2(ow[e]);
    w[e] = pk2(ov.x / (1.f + __expf(-gv.x)), ov.y / (1.f + __expf(-gv.y)));
  }
  *reinterpret_cast<uint4*>(U + row * HD + c) = make_uint4(w[0], w[1], w[2], w[3]);
}


// =============================================================================================================================
// Fused kernels for the wide path (tcgen05 / TMEM / TMA; the weights stream through shared memory in 64-column K chunks).
constexpr int WBM = 128;                                   // row tile (TMEM lanes)
constexpr int WDRAIN_BUF = 3;                              // per-drain-warp staging ring of [32 rows][32] bf16 (2 KiB each)
constexpr int WSMEM_CAP = 222 * 1024;                      // usable dynamic shared memory for these kernels (after 1 KiB align slack)
__device__ __forceinline__ float wsig(float x) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x)); return fmaf(0.5f, t, 0.5f); }

// ---- tail:  out = x + (sigmoid(g) o o) . Wo^T  in one pass.  C = output width (template), HDp = K (runtime, 64 k).
// Work item = (128-row tile, CO-column block of out); C > 256 runs two blocks per tile (CO = C / 2) so both TMEM accumulators
// fit and alternate: the drain of one item runs under the MMAs of the next (the second block re-reads g | o, from L2).
// warp 0: TMA (per K chunk: g | o [128][64] and the Wo rows of the block [CO][64]); warp 1: tcgen05; warps 2-9: u = sigmoid(g)
// o o over g; warps 10-13: drain (TMEM lane = row) -> per-warp [32][32] staging -> TMA stores.  Without dropout (fold) the
// residual goes through the MMA: the item's x chunks [128][64] come first, packed into one ring stage, and D[:, 64 j ..] =
// x_j . I64 (a resident 64 x 64 identity) starts each accumulator block at x exactly; with dropout the drain adds x (global
// loads) after the scale, out = x + drop o (u . Wo^T).
template <int C> struct WT {
  static constexpr int CO = (C > 256 && (C / 2) % 64 == 0) ? C / 2 : C;       // output columns per item
  static constexpr int NH = C / CO;                                            // items per row tile
  static constexpr int NN = CO > 256 ? 2 : 1, NNR = CO / NN;                   // MMA N split
  static constexpr int NACC = 2 * CO <= 512 ? 2 : 1;
  static constexpr int XC = WBM * 64 * 2;                                      // one [128][64] bf16 chunk
  static constexpr int SB = 2 * XC + CO * 64 * 2;                              // stage bytes: g | o | W
  static constexpr int CPS = SB / XC;                                          // residual x chunks a stage holds
  static constexpr int KXS = (CO / 64 + CPS - 1) / CPS;                        // residual stages per item
  static constexpr int DR = 4 * WDRAIN_BUF * 32 * 32 * 2;
  static constexpr int IB = 64 * 64 * 2;                                       // the identity block
  static constexpr int NSTG0 = (232448 - 1024 - 256 - DR - IB) / SB;            // 227 KiB opt-in, 1 KiB alignment slack
  static constexpr int NSTG = NSTG0 > 4 ? 4 : NSTG0;
  static constexpr int SMEM = 1024 + NSTG * SB + IB + DR + 256;
  static_assert(NSTG >= 2, "tail stages");
  static_assert(CO % 64 == 0 || NACC == 1, "residual chunks");
};

template <int C>
__global__ void __launch_bounds__(448, 1) tri_wtail_sm100(int ntiles, int KC, int fold,
    const __grid_constant__ CUtensorMap gmap, const __grid_constant__ CUtensorMap omap,   // [R][HDp], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,                                               // Wo [C][HDp], box (64, NNR), 128B swizzle
    const __grid_constant__ CUtensorMap xmap,                                               // x [R][C], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,                                               // out [R][C], box (32, 32), 64B swizzle
    const __nv_bfloat16* __restrict__ X,                                                    // residual [R][C] (dropout path)
    const __nv_bfloat16* __restrict__ DROP, int L) {        // dropout scale [B][L (the row's last index)][C] or nullptr
  using T = WT<C>;
  constexpr int CO = T::CO, KX = CO / 64;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* stg = smb;                                     // [NSTG][g | o | W]
  __nv_bfloat16* sI = reinterpret_cast<__nv_bfloat16*>(smb + T::NSTG * T::SB);   // I64 [64][64], 128B swizzle
  __nv_bfloat16* sO = sI + 64 * 64;                                                // drain staging [4][WDRAIN_BUF][32][32]
  uint64_t* bars = reinterpret_cast<uint64_t*>(smb + T::NSTG * T::SB + T::IB + T::DR);
  uint64_t* sf = bars;                 // [NSTG] chunk landed
  uint64_t* uf = sf + T::NSTG;         // [NSTG] count 8: u written
  uint64_t* se = uf + T::NSTG;         // [NSTG] its GEMMs retired
  uint64_t* af = se + T::NSTG;         // [NACC]
  uint64_t* ae = af + T::NACC;         // [NACC] count 4
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ae + T::NACC);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nitems = ntiles * T::NH, KXS = fold ? T::KXS : 0;
  if (tid == 0) {
    for (int i = 0; i < T::NSTG; ++i) { bar_init(sf + i, 1); bar_init(uf + i, 8); bar_init(se + i, 1); }
    for (int i = 0; i < T::NACC; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init_fence();
  }
  if (fold) {
    for (int i = tid; i < 64 * 64; i += blockDim.x) sI[sw128(i >> 6, i & 63)] = __float2bfloat16_rn((i >> 6) == (i & 63) ? 1.f : 0.f);
    fence_proxy_async();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto G = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB); };
  auto O = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB + T::XC); };
  auto Wc = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB + 2 * T::XC); };
  auto Xc = [&](int s, int i) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB + i * T::XC); };

  if (warp == 0) {
    if (lane == 0) {
      int y = 0;
      for (int it = blockIdx.x; it < nitems; it += gridDim.x) {
        const int t = it / T::NH, c0 = (it % T::NH) * CO;
        for (int kc = 0; kc < KXS + KC; ++kc, ++y) {
          const int s = y % T::NSTG;
          if (y >= T::NSTG) wait(se + s, ((y / T::NSTG) - 1) & 1);
          if (kc < KXS) {                                             // residual stage: up to CPS x chunks of the block
            const int j0 = kc * T::CPS, j1 = min(KX, j0 + T::CPS);
            expect_tx(sf + s, (j1 - j0) * T::XC);
            for (int j = j0; j < j1; ++j) load_2d(&xmap, Xc(s, j - j0), sf + s, c0 + j * 64, t * WBM);
            continue;
          }
          const int ku = kc - KXS;
          expect_tx(sf + s, T::SB);
          load_2d(&gmap, G(s), sf + s, ku * 64, t * WBM);
          load_2d(&omap, O(s), sf + s, ku * 64, t * WBM);
          for (int nh = 0; nh < T::NN; ++nh) load_2d(&wmap, Wc(s) + nh * T::NNR * 64, sf + s, ku * 64, c0 + nh * T::NNR);
        }
      }
    }
  } else if (warp == 1) {
    // converged loop, one elected lane issues; descriptors / barrier addresses precomputed (as in the front)
    const uint32_t ldr = elect_one() ? 1u : 0u;
    constexpr uint32_t ID = idesc_bf16(128, T::NNR, 0, 0), IDX = idesc_bf16(128, 64, 0, 0);
    const uint64_t dS0 = desc_k128(stg), dI = desc_k128(sI);
    const uint32_t buf_ = sa(uf), bse = sa(se), baf = sa(af), bae = sa(ae);
    int y = 0, lt = 0;
    for (int it = blockIdx.x; it < nitems; it += gridDim.x, ++lt) {
      const int acc = lt % T::NACC;
      if (lt >= T::NACC) wait(bae + 8 * acc, ((lt / T::NACC) - 1) & 1);
      for (int kc = 0; kc < KXS + KC; ++kc, ++y) {                   // with the residual every u chunk has kc >= 1: accumulates
        const int s = y % T::NSTG;
        wait(buf_ + 8 * s, (y / T::NSTG) & 1);
        tc_fence_after();
        const uint64_t dS = dS0 + (uint64_t)((s * T::SB) >> 4);
        if (kc < KXS) {
          const int j0 = kc * T::CPS, j1 = min(KX, j0 + T::CPS);
          for (int j = j0; j < j1; ++j)
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              mma_ss_if(ldr, tmem + acc * CO + j * 64, dS + (uint64_t)(((j - j0) * T::XC) >> 4) + 2 * ks, dI + 2 * ks, IDX, ks ? 1u : 0u);
          mma_commit_if(ldr, bse + 8 * s);
          __syncwarp();
          continue;
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
          for (int nh = 0; nh < T::NN; ++nh)
            mma_ss_if(ldr, tmem + acc * CO + nh * T::NNR, dS + 2 * ks, dS + (uint64_t)((2 * T::XC + nh * T::NNR * 128) >> 4) + 2 * ks, ID,
                      (kc | ks) ? 1u : 0u);
        mma_commit_if(ldr, bse + 8 * s);
        __syncwarp();
      }
      mma_commit_if(ldr, baf + 8 * acc);
      __syncwarp();
    }
  } else if (warp < 10) {
    // u = sigmoid(g) o o over g: row r, 16-byte chunks part*4 .. part*4+3 of the chunk's 8
    const int r = (warp - 2) * 16 + (lane & 15), part = lane >> 4;
    int y = 0;
    for (int it = blockIdx.x; it < nitems; it += gridDim.x)
      for (int kc = 0; kc < KXS + KC; ++kc, ++y) {
        const int s = y % T::NSTG;
        wait(sf + s, (y / T::NSTG) & 1);
        if (kc < KXS) { __syncwarp(); if (lane == 0) arrive(uf + s); continue; }   // residual stage: nothing to compute
        __nv_bfloat16* gr = G(s) + r * 64;
        const __nv_bfloat16* orow = O(s) + r * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int off = ((part * 4 + i) ^ (r & 7)) << 3;
          const uint4 a = *reinterpret_cast<const uint4*>(gr + off), o = *reinterpret_cast<const uint4*>(orow + off);
          const uint32_t aa[4] = {a.x, a.y, a.z, a.w}, oo[4] = {o.x, o.y, o.z, o.w};
          uint32_t w[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) { const float2 gv = bf2(aa[e]), ov = bf2(oo[e]); w[e] = pk2(wsig(gv.x) * ov.x, wsig(gv.y) * ov.y); }
          *reinterpret_cast<uint4*>(gr + off) = make_uint4(w[0], w[1], w[2], w[3]);
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) arrive(uf + s);
      }
  } else {
    const int qq = warp & 3;
    __nv_bfloat16* ring = sO + qq * WDRAIN_BUF * 32 * 32;
    int nst = 0, lt = 0;
    for (int it = blockIdx.x; it < nitems; it += gridDim.x, ++lt) {
      const int t = it / T::NH, c0 = (it % T::NH) * CO;
      const int acc = lt % T::NACC;
      const long row = (long)t * WBM + qq * 32 + lane;
      const uint4* xr = reinterpret_cast<const uint4*>(X + row * C + c0);
      uint4 xv[2][4] = {};
      if (!fold) {
#pragma unroll
        for (int c = 0; c < 4; ++c) xv[0][c] = __ldg(xr + c);
      }
      wait(af + acc, (lt / T::NACC) & 1);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < CO / 32; ++k) {
        float v[32];
        tmem_ld32(tmem_at(tmem + acc * CO + k * 32, qq * 32, 0), v);
        if (!fold && k + 1 < CO / 32) {
#pragma unroll
          for (int c = 0; c < 4; ++c) xv[(k + 1) & 1][c] = __ldg(xr + (k + 1) * 4 + c);
        }
        tmem_wait_ld();
        if (k == CO / 32 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + acc); }
        if (DROP != nullptr) {
          const uint4* sp = reinterpret_cast<const uint4*>(DROP + ((row / ((long)L * L)) * L + row % L) * C + c0 + k * 32);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint4 u = __ldg(sp + c);
            const uint32_t sw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) { const float2 sv = bf2(sw[e]); v[c * 8 + 2 * e] *= sv.x; v[c * 8 + 2 * e + 1] *= sv.y; }
          }
        }
        uint32_t w[16];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t xw[4] = {xv[k & 1][c].x, xv[k & 1][c].y, xv[k & 1][c].z, xv[k & 1][c].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { const float2 xx = bf2(xw[e]); w[c * 4 + e] = pk2(v[c * 8 + 2 * e] + xx.x, v[c * 8 + 2 * e + 1] + xx.y); }
        }
        __nv_bfloat16* buf = ring + (nst % WDRAIN_BUF) * 32 * 32;
        if (lane == 0 && nst >= WDRAIN_BUF) bulk_wait_read<WDRAIN_BUF - 1>();
        __syncwarp();
        unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + lane * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c)
          *reinterpret_cast<uint4*>(rowp + ((c ^ ((lane >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { store_2d(&ymap, buf, c0 + k * 32, t * WBM + qq * 32); bulk_commit(); }
        ++nst;
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- front:  y = LN(x) (never stored);  [q | k | v | g | bias] = y . W^T  in one pass.  C = input width (template), HDp and the
// head count runtime.  W = [Wq; Wk; Wv; Wg; Wb (16 rows)] [4 HDp + 16][C].  Output blocks: each of q / k / v / g in NBR <= 256-row
// pieces, then the 16-row bias block; two TMEM accumulators of 256 columns alternate between the GEMMs and the drain.
// 2-CTA clusters: the pair works on two 128-row tiles (2 pt + rank) in lockstep and the leader (rank 0) issues M = 256 products
// (cta_group::2) whose B, the block's W rows, is split by N across the pair: each CTA stages NBR / 2 rows of every W chunk, so each
// SM streams half of the weights (the per-SM TMA inflow, ~65 B/clk, paced the one-CTA kernel).  Each CTA keeps its own x / y tile
// and its own accumulator rows.  Completions the leader waits for are counted on its barriers (W transactions of both CTAs, LN
// and drain arrivals by remote arrive); the leader's commits arrive on both CTAs' barriers (multicast).
// warp 0: TMA (this CTA's half of the W chunks); warp 1: tcgen05 (leader); warps 2-9: LayerNorm in place (two threads a row) and
// the next tile's x load (warp 2 lane 0) -- with one x buffer (C >= 384) the statistics of the next tile come from global
// memory ahead of time and the tile is loaded and normalised in 64-column chunks, each signalled to the MMA on its own barrier,
// so the first block's GEMMs start on chunk 0; warps 10 .. 9 + ND: drain (ND / 4 warps a TMEM lane quarter taking alternate 64-column
// rounds) -> per-warp staging -> TMA stores, the bias block -> head-major bias.  ND = 8 where the drain paces the tile (C <= 256,
// blocks of >= 128 columns), else 4 (the wider C keep the shared memory for W stages).
struct Maps4 { CUtensorMap m[4]; };                                            // q | k | v | g output maps

__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
// no ordering on the arrive: at setup the barriers were published by fence.mbarrier_init.release.cluster; at teardown the pair's
// TMEM / shared traffic is ordered by the commits and the tcgen05 fences
__device__ __forceinline__ void cl_sync_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
}
// arrive on the barrier at the same offset in the leader (rank 0): release at CTA scope for shared operands this CTA's threads
// wrote (after fence.proxy.async; a cluster-scope release costs ~0.5 us), relaxed for TMEM-only handoffs (ordered by the tcgen05 fence)
__device__ __forceinline__ void arrive_lead_cta(uint32_t bar) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
__device__ __forceinline__ void arrive_lead_relaxed(uint32_t bar) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(bar));
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
__device__ __forceinline__ void wait_cl(uint32_t b, uint32_t parity) {       // acquire at cluster scope (arrivals from the peer)
  asm volatile("{\n.reg .pred p;\nWAITC_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra WAITC_%=;\n}\n"
               :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void tmem_alloc2(uint32_t* slot, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(sa(slot)), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n" :: "r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void mma2_ss_if(uint32_t leader, uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.b32 p, %4, 0;\nsetp.ne.b32 q, %5, 0;\n"
               "@q tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate), "r"(leader) : "memory");
}
// arrive on the barrier at this offset in both CTAs once the leader's prior tcgen05 ops are complete
__device__ __forceinline__ void commit2_mc_if(uint32_t leader, uint32_t bar) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %1, 0;\n"
               "@q tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %2;\n}\n"
               :: "r"(bar), "r"(leader), "h"((uint16_t)3) : "memory");
}
// TMA into this CTA's shared memory, the transaction completing on the leader's barrier at the same offset (peer bit cleared)
__device__ __forceinline__ void load_2d_2sm(const void* map, uint32_t dst, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory");
}

template <int C, int ND> struct WF {
  static constexpr int KC = C / 64;
  static constexpr int XB = WBM * C * 2;                                       // x / y tile bytes
  static constexpr int MAXN = 256;                                             // block rows = the pair's MMA N
  static constexpr int WSB = MAXN / 2 * 64 * 2;                                // W stage bytes: this CTA's half of a chunk
  static constexpr int FB = (ND == 8 ? C <= 128 : C < 384) ? 4 : 2;           // drain ring [32][32] buffers a warp: 4 = a round in flight
  static constexpr int DR = ND * FB * 32 * 32 * 2;
  static constexpr int PB = 2 * C * 4;                                         // gamma | beta fp32
  static constexpr int CAP = 232448 - 512;                                     // 227 KiB opt-in less the barriers; no alignment slack
  static constexpr int NX = (2 * XB + 2 * WSB + DR + PB) <= CAP ? 2 : 1;
  static constexpr int SSB = NX == 1 ? WBM * 8 : 0;                            // (mean, rstd) of the rows (one x buffer)
  static constexpr int NW0 = (CAP - NX * XB - DR - PB - SSB) / WSB;
  static constexpr int NW = NW0 > 6 ? 6 : NW0;
  static constexpr int SMEM = NX * XB + NW * WSB + DR + PB + SSB + 512;
  static_assert(NW >= 2, "front W stages");
};

template <int C, int ND>
__global__ void __launch_bounds__((10 + ND) * 32, 1) tri_wfront_sm100(int ntiles, int L, int HDp, int NH, int NBR, float eps,
    const __grid_constant__ CUtensorMap xmap,                                   // x [R][C], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,                                   // W [4 HDp + 16][C], box (64, NBR / 2), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,                                  // the same W, box (64, 8)
    const __grid_constant__ Maps4 omaps,                                        // q | k | v | g [R][HDp], box (32, 32), 64B swizzle
    const __nv_bfloat16* __restrict__ X,                                        // x [R][C] (row statistics, one x buffer)
    const float* __restrict__ LNW, const float* __restrict__ LNB, const bool* __restrict__ MASK,
    __nv_bfloat16* __restrict__ BIAS,                                           // [B][NH][L][L]
    float2* __restrict__ STATS) {                                               // (mean, rstd) [R] for the backward, or nullptr
  using T = WF<C, ND>;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((sa(raw) & 1023u) != 0u) __trap();                                              // SMEM carries no alignment slack
  unsigned char* smb = raw;
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);                          // [NX][KC][128][64]
  unsigned char* sW = smb + T::NX * T::XB;                                            // [NW][<=128][64]
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sW + T::NW * T::WSB);          // drain staging
  float* sP = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(sO) + T::DR); // gamma | beta
  float2* sS = reinterpret_cast<float2*>(sP + 2 * C);                                  // (mean, rstd) [128] (one x buffer)
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(sS) + T::SSB);
  constexpr int NXF = T::NX == 1 ? T::KC : T::NX;                                      // x barriers: a chunk each with one buffer
  uint64_t* xf = bars;               // [NXF] x landed (local)
  uint64_t* yf = xf + NXF;           // [NXF] leader: count 16, the LN warps of both CTAs
  uint64_t* xe = yf + NXF;           // [NX] the tile's GEMMs retired (multicast)
  uint64_t* wf = xe + T::NX;         // [NW] leader: the W transactions of both CTAs
  uint64_t* we = wf + T::NW;         // [NW] (multicast)
  uint64_t* af = we + T::NW;         // [2] (multicast)
  uint64_t* ae = af + 2;             // [2] leader: count 2 ND, the drain warps of both CTAs
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ae + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cl_rank();
  const bool lead = rank == 0;
  const int NBT = HDp / NBR, NBLK = 4 * NBT + 1;                                // blocks per tensor, blocks a tile (NBR rows each)
  const int np = ntiles >> 1, pair = blockIdx.x >> 1, npair = gridDim.x >> 1;   // pair tiles: this CTA's tile is 2 pt + rank
  if (tid == 0) {
    for (int i = 0; i < NXF; ++i) { bar_init(xf + i, 1); bar_init(yf + i, 16); }
    for (int i = 0; i < T::NX; ++i) bar_init(xe + i, 1);
    for (int i = 0; i < T::NW; ++i) { bar_init(wf + i, 1); bar_init(we + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 2 * ND); }
    bar_init_fence();
  }
  for (int i = tid; i < 2 * C; i += blockDim.x) sP[i] = i < C ? LNW[i] : LNB[i - C];
  if (warp == 1) tmem_alloc2(tslot, 512);
  tc_fence_before();
  __syncthreads();
  cl_sync_relaxed();                                   // both CTAs' barriers exist before any remote arrive or pair transaction
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto blk_rows = [&](int b) { return b < 4 * NBT ? NBR : 16; };
  auto blk_row0 = [&](int b) { return b < 4 * NBT ? (b / NBT) * HDp + (b % NBT) * NBR : 4 * HDp; };
  auto load_x = [&](int t, int xb) {
    expect_tx(xf + xb, T::XB);
    for (int kc = 0; kc < T::KC; ++kc) load_2d(&xmap, sX + (xb * T::KC + kc) * WBM * 64, xf + xb, kc * 64, t * WBM);
  };

  if (warp == 0) {
    if (lane == 0) {
      const uint32_t bwf = sa(wf), bw0 = sa(sW);
      int y = 0;
      for (int pt = pair; pt < np; pt += npair)
        for (int b = 0; b < NBLK; ++b)
          for (int kc = 0; kc < T::KC; ++kc, ++y) {
            const int s = y % T::NW, half = blk_rows(b) >> 1;
            if (y >= T::NW) wait(we + s, ((y / T::NW) - 1) & 1);
            if (lead) expect_tx(wf + s, 2 * half * 128);                 // the pair's bytes, counted on the leader's barrier
            load_2d_2sm(half == 8 ? &wbmap : &wmap, bw0 + s * T::WSB, bwf + 8 * s, kc * 64, blk_row0(b) + rank * half);
          }
    }
  } else if (warp == 1) {
    if (lead) {
      // converged loop: every lane waits, one elected lane issues; descriptors and barrier addresses precomputed
      const uint32_t ldr = elect_one() ? 1u : 0u;
      const uint64_t dY0 = desc_k128(sX), dW0 = desc_k128(sW);
      const uint32_t bwf = sa(wf), bwe = sa(we), baf = sa(af), bae = sa(ae), byf = sa(yf), bxe = sa(xe);
      int y = 0, gb = 0, lt = 0;
      for (int pt = pair; pt < np; pt += npair, ++lt) {
        const int xb = lt % T::NX;
        if (T::NX == 2) wait_cl(byf + 8 * xb, (lt / T::NX) & 1);      // one buffer: per chunk, in the first block
        const uint64_t dY = dY0 + (uint64_t)((xb * T::XB) >> 4);
        for (int b = 0; b < NBLK; ++b, ++gb) {
          const int acc = gb & 1;
          const uint32_t id = idesc_bf16(256, blk_rows(b), 0, 0);
          if (gb >= 2) wait_cl(bae + 8 * acc, ((gb >> 1) - 1) & 1);
          for (int kc = 0; kc < T::KC; ++kc, ++y) {
            const int s = y % T::NW;
            if (T::NX == 1 && b == 0) wait_cl(byf + 8 * kc, lt & 1);
            wait(bwf + 8 * s, (y / T::NW) & 1);
            tc_fence_after();
            const uint64_t a = dY + (uint64_t)((kc * WBM * 64 * 2) >> 4), bw = dW0 + (uint64_t)((s * T::WSB) >> 4);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) mma2_ss_if(ldr, tmem + acc * 256, a + 2 * ks, bw + 2 * ks, id, (kc | ks) ? 1u : 0u);
            commit2_mc_if(ldr, bwe + 8 * s);
            __syncwarp();
          }
          commit2_mc_if(ldr, baf + 8 * acc);
          __syncwarp();
        }
        commit2_mc_if(ldr, bxe + 8 * xb);
        __syncwarp();
      }
    }
  } else if (warp < 10) {
    // LayerNorm in place: row r, the 16-byte chunks j of the row with j % 2 == part (the row's C / 8 chunks)
    if constexpr (T::NX == 1) {
      // statistics of the next tile from global (this warp's 16 rows, four a step, a row's 16-byte vectors across the lanes); the
      // tile in 64-column chunks: normalised as they land, each chunk released to the MMA on its own barrier
      const int r0 = (warp - 2) * 16, r = r0 + (lane & 15), part = lane >> 4;
      const bool leader = warp == 2 && lane == 0;
      const uint32_t byf = sa(yf);
      constexpr int NV = C / 8, PER = (NV + 31) / 32;
      const uint4* X4 = reinterpret_cast<const uint4*>(X);
      auto load_chunks = [&](int tt) {
        for (int kc = 0; kc < T::KC; ++kc) {
          expect_tx(xf + kc, WBM * 64 * 2);
          load_2d(&xmap, sX + kc * WBM * 64, xf + kc, kc * 64, tt * WBM);
        }
      };
      auto stats = [&](int tt) {
#pragma unroll 1
        for (int rr = 0; rr < 16; rr += 4) {
          uint4 u[4][PER];
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int p = 0; p < PER; ++p) {
              const int j = lane + 32 * p;
              u[i][p] = j < NV ? __ldg(X4 + ((long)tt * WBM + r0 + rr + i) * NV + j) : make_uint4(0u, 0u, 0u, 0u);
            }
          float s1[4], s2[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            s1[i] = 0.f; s2[i] = 0.f;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
              const uint32_t w[4] = {u[i][p].x, u[i][p].y, u[i][p].z, u[i][p].w};
#pragma unroll
              for (int e = 0; e < 4; ++e) { const float2 f = bf2(w[e]); s1[i] += f.x + f.y; s2[i] = fmaf(f.x, f.x, fmaf(f.y, f.y, s2[i])); }
            }
          }
#pragma unroll
          for (int o = 16; o >= 1; o >>= 1)
#pragma unroll
            for (int i = 0; i < 4; ++i) { s1[i] += __shfl_xor_sync(0xffffffffu, s1[i], o); s2[i] += __shfl_xor_sync(0xffffffffu, s2[i], o); }
          if (lane < 4) {
            float a = s1[0], q = s2[0];
#pragma unroll
            for (int i = 1; i < 4; ++i) if (lane == i) { a = s1[i]; q = s2[i]; }
            const float mean = a * (1.f / C);
            sS[r0 + rr + lane] = make_float2(mean, rsqrtf(fmaxf(q * (1.f / C) - mean * mean, 0.f) + eps));
          }
        }
        __syncwarp();
      };
      if (pair < np) {
        if (leader) load_chunks(2 * pair + rank);
        stats(2 * pair + rank);
      }
      int lt = 0;
      for (int pt = pair; pt < np; pt += npair, ++lt) {
        const int tn = 2 * (pt + npair) + rank;
        const float2 st = sS[r];
        if (STATS != nullptr && part == 0) STATS[(long)(2 * pt + rank) * WBM + r] = st;
        for (int kc = 0; kc < T::KC; ++kc) {
          wait(xf + kc, lt & 1);
          __nv_bfloat16* cp = sX + kc * WBM * 64 + r * 64;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int q8 = part * 4 + i;                                   // 16-byte piece of the chunk row
            uint4* p = reinterpret_cast<uint4*>(cp + ((q8 ^ (r & 7)) << 3));
            const uint4 u = *p;
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
            uint32_t o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int c = kc * 64 + q8 * 8 + 2 * e;
              const float2 f = bf2(w[e]);
              o[e] = pk2((f.x - st.x) * st.y * sP[c] + sP[C + c], (f.y - st.x) * st.y * sP[c + 1] + sP[C + c + 1]);
            }
            *p = make_uint4(o[0], o[1], o[2], o[3]);
          }
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) arrive_lead_cta(byf + 8 * kc);
        }
        if (tn < ntiles) {
          stats(tn);                                                   // this warp's rows: read above, before the overwrite
          if (leader) { wait(xe, lt & 1); load_chunks(tn); }           // the buffer is free once this tile's GEMMs retired
        }
      }
    } else {
    const int r = (warp - 2) * 16 + (lane & 15), part = lane >> 4;
    const bool leader = warp == 2 && lane == 0;
    const uint32_t byf = sa(yf);
    constexpr int NCH = C / 8;
    if (leader && pair < np) load_x(2 * pair + rank, 0);
    int lt = 0;
    for (int pt = pair; pt < np; pt += npair, ++lt) {
      const int xb = lt % T::NX, tn = 2 * (pt + npair) + rank;
      wait(xf + xb, (lt / T::NX) & 1);
      __nv_bfloat16* Xt = sX + xb * T::KC * WBM * 64;
      auto chunk = [&](int j) { return Xt + (j >> 3) * WBM * 64 + r * 64 + (((j & 7) ^ (r & 7)) << 3); };
      float s1 = 0.f, s2 = 0.f;
#pragma unroll 4
      for (int j = part; j < NCH; j += 2) {
        const uint4 u = *reinterpret_cast<const uint4*>(chunk(j));
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 f = bf2(w[e]); s1 += f.x + f.y; s2 = fmaf(f.x, f.x, fmaf(f.y, f.y, s2)); }
      }
      s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
      s2 += __shfl_xor_sync(0xffffffffu, s2, 16);
      const float mean = s1 * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
      if (STATS != nullptr && part == 0) STATS[(long)(2 * pt + rank) * WBM + r] = make_float2(mean, rstd);
#pragma unroll 4
      for (int j = part; j < NCH; j += 2) {
        uint4* p = reinterpret_cast<uint4*>(chunk(j));
        const uint4 u = *p;
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = j * 8 + 2 * e;
          const float2 f = bf2(w[e]);
          o[e] = pk2((f.x - mean) * rstd * sP[c] + sP[C + c], (f.y - mean) * rstd * sP[c + 1] + sP[C + c + 1]);
        }
        *p = make_uint4(o[0], o[1], o[2], o[3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive_lead_cta(byf + 8 * xb);
      if (T::NX == 2 && leader && tn < ntiles) {                       // the next tile into the other buffer, once the previous
        if (lt >= 1) wait(xe + (xb ^ 1), ((lt - 1) / 2) & 1);         // tile's GEMMs are done (after this LN: the LN of a tile
        load_x(tn, xb ^ 1);                                            // runs under the GEMMs of the one before)
      }
      if (T::NX == 1 && leader && tn < ntiles) {                       // one buffer: the next x waits for this tile's GEMMs
        wait(xe, lt & 1);
        load_x(tn, 0);
      }
    }
    }
  } else {
    const int qq = warp & 3, hv = (warp - 10) >> 2;     // TMEM lane quarter; the quarter's ND / 4 warps take alternate rounds
    const uint32_t bae = sa(ae);
    __nv_bfloat16* ring = sO + (warp - 10) * T::FB * 32 * 32;
    const __nv_bfloat16 lo = __float2bfloat16_rn(-3.3895313892515355e38f);
    const int NSL = NBR / 32, NRD = (NSL + 1) / 2;      // 32-column slices, 64-column rounds of a block
    int nst = 0, gb = 0;
    for (int pt = pair; pt < np; pt += npair) {
      const int t = 2 * pt + rank;
      const long row = (long)t * WBM + qq * 32 + lane;
      for (int b = 0; b < NBLK; ++b, ++gb) {
        const int acc = gb & 1;
        wait(af + acc, (gb >> 1) & 1);
        tc_fence_after();
        if (b == 4 * NBT) {                        // bias block: 16 columns, heads 0 .. NH-1 -> [b][h][j][k], masked keys
          float v[16];
          if (hv == 0) { tmem_ld16(tmem_at(tmem + acc * 256, qq * 32, 0), v); tmem_wait_ld(); }
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive_lead_relaxed(bae + 8 * acc);
          if (hv == 0) {
            const int kx = (int)(row % L);
            const long bj = row / L;
            const int j = (int)(bj % L), bb = (int)(bj / L);
            const bool keep = MASK == nullptr || MASK[(long)bb * L + kx];
            for (int h = 0; h < NH; ++h) BIAS[(((long)bb * NH + h) * L + j) * L + kx] = keep ? __float2bfloat16_rn(v[h]) : lo;
          }
          continue;
        }
        if (hv >= NRD) {                           // no round of this block for this warp
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive_lead_relaxed(bae + 8 * acc);
          continue;
        }
        const int ten = b / NBT, c0 = (b % NBT) * NBR;
        for (int rd = hv; rd < NRD; rd += ND / 4) {            // two 32-column slices a round: one TMEM wait, one fence, one commit
          const int ch = 2 * rd;
          const bool two = ch + 1 < NSL;
          float v[2][32];
          tmem_ld32(tmem_at(tmem + acc * 256 + ch * 32, qq * 32, 0), v[0]);
          if (two) tmem_ld32(tmem_at(tmem + acc * 256 + ch * 32 + 32, qq * 32, 0), v[1]);
          tmem_wait_ld();
          if (rd + ND / 4 >= NRD) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive_lead_relaxed(bae + 8 * acc); }
          if (lane == 0) bulk_wait_read<T::FB == 4 ? 1 : 0>();  // (4 buffers) all but the last round read: this round's two are free
          __syncwarp();
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            if (p == 1 && !two) break;
            unsigned char* rowp = reinterpret_cast<unsigned char*>(ring + ((nst + p) % T::FB) * 32 * 32) + lane * 64;
#pragma unroll
            for (int c = 0; c < 4; ++c)
              *reinterpret_cast<uint4*>(rowp + ((c ^ ((lane >> 1) & 3)) << 4)) =
                  make_uint4(pk2(v[p][c * 8], v[p][c * 8 + 1]), pk2(v[p][c * 8 + 2], v[p][c * 8 + 3]), pk2(v[p][c * 8 + 4], v[p][c * 8 + 5]),
                             pk2(v[p][c * 8 + 6], v[p][c * 8 + 7]));
          }
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) {
            store_2d(&omaps.m[ten], ring + (nst % T::FB) * 32 * 32, c0 + ch * 32, t * WBM + qq * 32);
            if (two) store_2d(&omaps.m[ten], ring + ((nst + 1) % T::FB) * 32 * 32, c0 + ch * 32 + 32, t * WBM + qq * 32);
            bulk_commit();
          }
          nst += two ? 2 : 1;
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  cl_sync_relaxed();                                   // the leader's last MMAs / commits into the peer are done
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}

// =============================================================================================================================
// Backward of the wide path.

// ---- dy_s = dy o drop[b, j, :] (the module's broadcast dropout scale), 8 channels a thread
__global__ void __launch_bounds__(256) tri_scale_rows(const __nv_bfloat16* __restrict__ DY, const __nv_bfloat16* __restrict__ DROP,
                                                      __nv_bfloat16* __restrict__ OUT, int C, int L, long R) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int per = C / 8;
  if (i >= R * per) return;
  const long row = i / per;
  const int c = (int)(i % per) * 8;
  const long srow = (row / ((long)L * L)) * L + row % L;
  const uint4 d = *reinterpret_cast<const uint4*>(DY + row * C + c), m = __ldg(reinterpret_cast<const uint4*>(DROP + srow * C + c));
  const uint32_t dw[4] = {d.x, d.y, d.z, d.w}, mw[4] = {m.x, m.y, m.z, m.w};
  uint32_t o[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) { const float2 a = bf2(dw[e]), b = bf2(mw[e]); o[e] = pk2(a.x * b.x, a.y * b.y); }
  *reinterpret_cast<uint4*>(OUT + row * C + c) = make_uint4(o[0], o[1], o[2], o[3]);
}

// ---- dbias fp32 [B][H][L][L] -> rows [R][64] bf16 (row (b, j, k): its H heads, then zeros), the head backward's A block
__global__ void __launch_bounds__(256) tri_db_rows(const float* __restrict__ DB, __nv_bfloat16* __restrict__ OUT, long ldo, int H, int L,
                                                   long R) {
  const long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  const int kx = (int)(row % L);
  const long bj = row / L;
  const int j = (int)(bj % L), b = (int)(bj / L);
  uint32_t w[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int h0 = 2 * i, h1 = 2 * i + 1;
    const float a = h0 < H ? DB[(((long)b * H + h0) * L + j) * L + kx] : 0.f;
    const float c = h1 < H ? DB[(((long)b * H + h1) * L + j) * L + kx] : 0.f;
    w[i] = pk2(a, c);
  }
  uint4* dst = reinterpret_cast<uint4*>(OUT + row * ldo);
#pragma unroll
  for (int q = 0; q < 8; ++q) dst[q] = make_uint4(w[4 * q], w[4 * q + 1], w[4 * q + 2], w[4 * q + 3]);
}

// ---- gate backward:  du = dy_s . Wo  (dy_s = dy o drop);  s = sigmoid(g);  do = du o s;  dg = du o o o s (1 - s);  u = s o o;
// delta[b, i, h, j] = sum over head h's 32 channels of do o o.  Per output block of NBR <= 128 columns the dy_s tile (A) and the
// WoT rows (B) stream together in 64-column K chunks (dy_s is re-read per block, from L2) and du accumulates in two alternating
// TMEM accumulators.  The drain computes the gate math per 32-column slice (one head): eight warps, two a TMEM lane quarter taking
// alternate slices, each prefetching its next g | o slice by TMA into a three-slot ring; do | dg | u overwrite the slot in place
// and leave by TMA.  warp 0: TMA (A | W stages); warp 1: tcgen05; warps 2-9: drain.
constexpr int GB_SLOT = 3 * 32 * 32 * 2;                  // a drain slot: g | o in (TMA), then do | dg | u out
template <int C> struct WG {
  static constexpr int KC = C / 64;
  static constexpr int MAXN = 128;                        // block columns (MMA N)
  static constexpr int SB = WBM * 64 * 2 + MAXN * 64 * 2;  // stage: dy_s chunk [128][64] | WoT chunk [<=128][64]
  static constexpr int NSL = 3;                           // drain slots a warp
  static constexpr int DR = 8 * NSL * GB_SLOT;
  static constexpr int NW0 = (232448 - 1024 - 512 - DR) / SB;
  static constexpr int NW = NW0 > 4 ? 4 : NW0;
  static constexpr int SMEM = 1024 + NW * SB + DR + 512;
  static_assert(NW >= 2, "gate backward stages");
};

template <int C>
__global__ void __launch_bounds__(320, 1) tri_wgbwd_sm100(int ntiles, int L, int HDp, int NBR,
    const __grid_constant__ CUtensorMap amap,                                   // dy_s [R][C], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,                                   // WoT [HDp][C], box (64, NBR), 128B swizzle
    const __grid_constant__ CUtensorMap domap, const __grid_constant__ CUtensorMap dgmap,
    const __grid_constant__ CUtensorMap umap,                                   // [R][HDp], box (32, 32), 64B swizzle
    const __grid_constant__ CUtensorMap gmap, const __grid_constant__ CUtensorMap omap,   // g / o [R][HDp], box (32, 32), 64B swizzle
    float* __restrict__ DELTA) {                                                // [B][L][H][L]
  using T = WG<C>;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* stg = smb;                                                           // [NW][A | W]
  unsigned char* sOut = stg + T::NW * T::SB;                                          // [8 warps][NSL][g | o -> do | dg | u]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sOut + T::DR);
  uint64_t* sf = bars;               // [NW]
  uint64_t* se = sf + T::NW;         // [NW]
  uint64_t* af = se + T::NW;         // [2]
  uint64_t* ae = af + 2;             // [2] count 8
  uint64_t* gl = ae + 2;             // [8][NSL] a slot's g | o landed
  uint32_t* tslot = reinterpret_cast<uint32_t*>(gl + 8 * T::NSL);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int NBT = HDp / NBR, H = HDp / 32, NSB = NBR / 32;                     // blocks, heads, slices a block
  if (tid == 0) {
    for (int i = 0; i < T::NW; ++i) { bar_init(sf + i, 1); bar_init(se + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 8); }
    for (int i = 0; i < 8 * T::NSL; ++i) bar_init(gl + i, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      int y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int b = 0; b < NBT; ++b)
          for (int kc = 0; kc < T::KC; ++kc, ++y) {
            const int s = y % T::NW;
            if (y >= T::NW) wait(se + s, ((y / T::NW) - 1) & 1);
            expect_tx(sf + s, (WBM + NBR) * 128);
            load_2d(&amap, stg + s * T::SB, sf + s, kc * 64, t * WBM);
            load_2d(&wmap, stg + s * T::SB + WBM * 128, sf + s, kc * 64, b * NBR);
          }
    }
  } else if (warp == 1) {
    const uint32_t ldr = elect_one() ? 1u : 0u;
    const uint32_t id = idesc_bf16(128, NBR, 0, 0);
    const uint64_t dS0 = desc_k128(stg);
    const uint32_t bsf = sa(sf), bse = sa(se), baf = sa(af), bae = sa(ae);
    int y = 0, gb = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
      for (int b = 0; b < NBT; ++b, ++gb) {
        const int acc = gb & 1;
        if (gb >= 2) wait(bae + 8 * acc, ((gb >> 1) - 1) & 1);
        for (int kc = 0; kc < T::KC; ++kc, ++y) {
          const int s = y % T::NW;
          wait(bsf + 8 * s, (y / T::NW) & 1);
          tc_fence_after();
          const uint64_t a = dS0 + (uint64_t)((s * T::SB) >> 4), bw = a + (uint64_t)((WBM * 128) >> 4);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) mma_ss_if(ldr, tmem + acc * 256, a + 2 * ks, bw + 2 * ks, id, (kc | ks) ? 1u : 0u);
          mma_commit_if(ldr, bse + 8 * s);
          __syncwarp();
        }
        mma_commit_if(ldr, baf + 8 * acc);
        __syncwarp();
      }
  } else if (warp < 10) {
    const int dw = warp - 2, qq = warp & 3, hv = dw >> 2;                    // TMEM lane quarter; slices ch = hv, hv + 2, ...
    const int PER = (NSB - hv + 1) / 2;                                        // this warp's slices a block
    unsigned char* ring = sOut + dw * T::NSL * GB_SLOT;
    uint64_t* glw = gl + dw * T::NSL;
    // load cursor (one slice ahead of the processing cursor)
    int lt_ = blockIdx.x, lb = 0, lj = 0, nl = 0;
    auto load_one = [&]() {                                                     // lane 0: slice (lt_, lb, lj) into slot nl % NSL
      const int sl = nl % T::NSL, col = lb * NBR + (hv + 2 * lj) * 32, row = lt_ * WBM + qq * 32;
      unsigned char* p = ring + sl * GB_SLOT;
      expect_tx(glw + sl, 2 * 2048);
      load_2d(&gmap, p, glw + sl, col, row);
      load_2d(&omap, p + 2048, glw + sl, col, row);
      ++nl;
      if (++lj == PER) { lj = 0; if (++lb == NBT) { lb = 0; lt_ += gridDim.x; } }
    };
    if (lane == 0 && PER > 0 && lt_ < ntiles) load_one();
    int n = 0, gb = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      const long row = (long)t * WBM + qq * 32 + lane;
      const int jx = (int)(row % L);
      const long bi = row / L;
      const int ix = (int)(bi % L), bb = (int)(bi / L);
      for (int b = 0; b < NBT; ++b, ++gb) {
        const int acc = gb & 1;
        wait(af + acc, (gb >> 1) & 1);
        tc_fence_after();
        if (PER == 0) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + acc); continue; }
        for (int j = 0; j < PER; ++j, ++n) {
          const int ch = hv + 2 * j, col0 = b * NBR + ch * 32, h = col0 / 32, sl = n % T::NSL;
          if (lane == 0 && lt_ < ntiles) { bulk_wait_read<1>(); load_one(); }  // the slot of slice n - 2: its stores have been read
          float du[32];
          tmem_ld32(tmem_at(tmem + acc * 256 + ch * 32, qq * 32, 0), du);
          tmem_wait_ld();
          if (j == PER - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + acc); }
          wait(glw + sl, (n / T::NSL) & 1);
          unsigned char* slot = ring + sl * GB_SLOT;
          unsigned char* rowp = slot + lane * 64;
          uint4 gv[4], ov[4];
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int off = (c ^ ((lane >> 1) & 3)) << 4;
            gv[c] = *reinterpret_cast<const uint4*>(rowp + off);
            ov[c] = *reinterpret_cast<const uint4*>(rowp + 2048 + off);
          }
          uint32_t wdo[16], wdg[16], wu[16];
          float dsum = 0.f;
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint32_t gw[4] = {gv[c].x, gv[c].y, gv[c].z, gv[c].w}, ow[4] = {ov[c].x, ov[c].y, ov[c].z, ov[c].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 g2 = bf2(gw[e]), o2 = bf2(ow[e]);
              const float s0 = wsig(g2.x), s1 = wsig(g2.y);
              const float u0 = du[c * 8 + 2 * e], u1 = du[c * 8 + 2 * e + 1];
              const float d0 = u0 * s0, d1 = u1 * s1;
              dsum = fmaf(d0, o2.x, fmaf(d1, o2.y, dsum));
              wdo[c * 4 + e] = pk2(d0, d1);
              wdg[c * 4 + e] = pk2(u0 * o2.x * s0 * (1.f - s0), u1 * o2.y * s1 * (1.f - s1));
              wu[c * 4 + e] = pk2(s0 * o2.x, s1 * o2.y);
            }
          }
          DELTA[(((long)bb * L + ix) * H + h) * L + jx] = dsum;
          __syncwarp();                                                         // every lane has read its g | o: overwrite in place
          const uint32_t* srcs[3] = {wdo, wdg, wu};
#pragma unroll
          for (int q = 0; q < 3; ++q) {
            unsigned char* rp = slot + q * 2048 + lane * 64;
#pragma unroll
            for (int c = 0; c < 4; ++c)
              *reinterpret_cast<uint4*>(rp + ((c ^ ((lane >> 1) & 3)) << 4)) =
                  make_uint4(srcs[q][c * 4], srcs[q][c * 4 + 1], srcs[q][c * 4 + 2], srcs[q][c * 4 + 3]);
          }
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) {
            store_2d(&domap, slot, col0, t * WBM + qq * 32);
            store_2d(&dgmap, slot + 2048, col0, t * WBM + qq * 32);
            store_2d(&umap, slot + 4096, col0, t * WBM + qq * 32);
            bulk_commit();
          }
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- head backward:  g_y = [dq | dk | dv | dg | db_rows] . [Wq; Wk; Wv; Wg; Wb64]  (the gradient at the LN output y), then the
// LayerNorm backward and the residual:  dx = dy + rstd (g_y o gamma - mean(g_y o gamma) - xhat mean(g_y o gamma o xhat)); also
// writes xhat | 1 | 0.. [R][C + 64] for the weight-gradient GEMMs.  A = the five gradient tensors in 64-column K chunks, B =
// W^T [C][4 HDp + 64] chunks (both streamed), the [128][C] accumulator in TMEM.  Each row's (mean, rstd) comes from the forward;
// the drain reads x (pass 1: the two row sums) and x | dy (pass 2: dx, xhat) as [32][32] slices a TMA ahead in a three-slot ring
// per warp, dx | xhat overwriting the slot in place.  warp 0: TMA; warp 1: tcgen05; warps 2 .. 1 + ND: drain.  C > 256 (one
// accumulator: the drain and the next tile's MMAs run in series) takes ND = 8 drain warps, two a TMEM lane quarter on alternate
// 32-column slices with the pass-1 row sums combined through shared memory, and stages of one W half ([256][64], the D chunk
// loaded with each) so the smem covers the larger drain; C <= 256 takes ND = 4 and whole-W stages.
constexpr int HB_SLOT = 2 * 32 * 32 * 2;                  // a drain slot: x | dy in, then dx | xhat out
template <int C> struct WH {
  static constexpr int NN = C > 256 ? 2 : 1, NNR = C / NN;                     // W halves: one a stage
  static constexpr int ND = C > 256 ? 8 : 4;                                   // drain warps
  static constexpr int SB = WBM * 64 * 2 + NNR * 64 * 2;                       // A chunk | W chunk (half)
  static constexpr int NSL = C > 384 ? 2 : 3;                                 // drain ring slots a warp (the widest keep W stages)
  static constexpr int DR = ND * NSL * HB_SLOT;
  static constexpr int KB = 2 * 32 * 32 * 2 + C * 4 + 2 * 2 * WBM * 8;         // constant pad tiles, gamma, row-sum exchange
  static constexpr int NSTG0 = (232448 - 1024 - 512 - DR - KB) / SB;
  static constexpr int NSTG = NSTG0 > 4 ? 4 : NSTG0;
  static constexpr int NACC = C <= 256 ? 2 : 1;
  static constexpr int SMEM = 1024 + NSTG * SB + DR + KB + 512;
  static_assert(NSTG >= 2, "head backward stages");
};

template <int C>
__global__ void __launch_bounds__((2 + WH<C>::ND) * 32, 1) tri_whbwd_sm100(int ntiles, int HDp,
    const __grid_constant__ CUtensorMap amap,                                            // D = [dq | dk | dv | dg | db] [R][4 HDp + 64], box (64, 128), 128B
    const __grid_constant__ CUtensorMap wmap,                                            // W^T [C][4 HDp + 64], box (64, NNR), 128B
    const __grid_constant__ CUtensorMap dxmap,                                           // dx [R][C], box (32, 32), 64B
    const __grid_constant__ CUtensorMap xhmap,                                           // xhat | 1 [R][C + 64], box (32, 32), 64B
    const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap dymap,  // x / dy [R][C], box (32, 32), 64B
    const float2* __restrict__ STATS, const float* __restrict__ LNW) {
  using T = WH<C>;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* stg = smb;                                                            // [NSTG][A | W]
  unsigned char* sOut = smb + T::NSTG * T::SB;                                        // [4][NSL][x | dy -> dx | xhat][32][32]
  unsigned char* sPad = sOut + T::DR;                                                 // [2][32][32] bf16: (1, 0, ..) and zeros
  float* sG = reinterpret_cast<float*>(sPad + 2 * 32 * 32 * 2);                       // gamma [C]
  float2* sXc = reinterpret_cast<float2*>(sG + C);                                    // [2 tiles][2 halves][128] partial row sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sXc + 2 * 2 * WBM);
  uint64_t* sf = bars;                 // [NSTG]
  uint64_t* se = sf + T::NSTG;         // [NSTG]
  uint64_t* af = se + T::NSTG;         // [NACC]
  uint64_t* ae = af + T::NACC;         // [NACC] count ND
  uint64_t* gl = ae + T::NACC;         // [ND][NSL] a slot's slices landed
  uint32_t* tslot = reinterpret_cast<uint32_t*>(gl + T::ND * T::NSL);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int HD64 = HDp / 64, KA = 4 * HD64 + 1;
  if (tid == 0) {
    for (int i = 0; i < T::NSTG; ++i) { bar_init(sf + i, 1); bar_init(se + i, 1); }
    for (int i = 0; i < T::NACC; ++i) { bar_init(af + i, 1); bar_init(ae + i, T::ND); }
    for (int i = 0; i < T::ND * T::NSL; ++i) bar_init(gl + i, 1);
    bar_init_fence();
  }
  for (int i = tid; i < C; i += blockDim.x) sG[i] = LNW[i];
  {                                                                                   // pad tiles: column C = 1, the rest 0
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(sPad);
    for (int i = tid; i < 2 * 32 * 32; i += blockDim.x) p[i] = __float2bfloat16_rn(0.f);
  }
  __syncthreads();
  if (tid < 32) {                                                                     // row tid, column 0 of the first tile (64B swizzle)
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(sPad + tid * 64 + ((0 ^ ((tid >> 1) & 3)) << 4));
    p[0] = __float2bfloat16_rn(1.f);
  }
  fence_proxy_async();
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto Ast = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB); };
  auto Wst = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB + WBM * 64 * 2); };

  if (warp == 0) {
    if (lane == 0) {
      int y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int kc = 0; kc < KA; ++kc)
          for (int nh = 0; nh < T::NN; ++nh, ++y) {                  // a stage: the D chunk and one W half
            const int s = y % T::NSTG;
            if (y >= T::NSTG) wait(se + s, ((y / T::NSTG) - 1) & 1);
            expect_tx(sf + s, T::SB);
            load_2d(&amap, Ast(s), sf + s, kc * 64, t * WBM);
            load_2d(&wmap, Wst(s), sf + s, kc * 64, nh * T::NNR);
          }
    }
  } else if (warp == 1) {
    const uint32_t ldr = elect_one() ? 1u : 0u;
    constexpr uint32_t ID = idesc_bf16(128, T::NNR, 0, 0);
    const uint64_t dS0 = desc_k128(stg);
    const uint32_t bsf = sa(sf), bse = sa(se), baf = sa(af), bae = sa(ae);
    int y = 0, lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int acc = lt % T::NACC;
      if (lt >= T::NACC) wait(bae + 8 * acc, ((lt / T::NACC) - 1) & 1);
      for (int kc = 0; kc < KA; ++kc)
        for (int nh = 0; nh < T::NN; ++nh, ++y) {
          const int s = y % T::NSTG;
          wait(bsf + 8 * s, (y / T::NSTG) & 1);
          tc_fence_after();
          const uint64_t a = dS0 + (uint64_t)((s * T::SB) >> 4), bw = a + (uint64_t)((WBM * 128) >> 4);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) mma_ss_if(ldr, tmem + acc * C + nh * T::NNR, a + 2 * ks, bw + 2 * ks, ID, (kc | ks) ? 1u : 0u);
          mma_commit_if(ldr, bse + 8 * s);
          __syncwarp();
        }
      mma_commit_if(ldr, baf + 8 * acc);
      __syncwarp();
    }
  } else {
    const int dw = warp - 2, qq = warp & 3, hv = dw >> 2;                    // TMEM lane quarter; slices k = hv, hv + NH2, ...
    constexpr int NH2 = T::ND / 4, NK = C / 32, NKW = NK / NH2, NJ = 2 * NKW;  // jobs a tile: pass 1 (x), pass 2 (x | dy)
    unsigned char* ring = sOut + dw * T::NSL * HB_SLOT;
    uint64_t* glw = gl + dw * T::NSL;
    int lt_ = blockIdx.x, lj = 0, nl = 0;                                     // load cursor, one job ahead
    auto load_one = [&]() {
      const int sl = nl % T::NSL, row = lt_ * WBM + qq * 32;
      unsigned char* p = ring + sl * HB_SLOT;
      if (lj < NKW) {
        expect_tx(glw + sl, 2048);
        load_2d(&xmap, p, glw + sl, (hv + NH2 * lj) * 32, row);
      } else {
        expect_tx(glw + sl, 4096);
        load_2d(&xmap, p, glw + sl, (hv + NH2 * (lj - NKW)) * 32, row);
        load_2d(&dymap, p + 2048, glw + sl, (hv + NH2 * (lj - NKW)) * 32, row);
      }
      ++nl;
      if (++lj == NJ) { lj = 0; lt_ += gridDim.x; }
    };
    if (lane == 0 && lt_ < ntiles) load_one();
    int n = 0, lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int acc = lt % T::NACC;
      const long row = (long)t * WBM + qq * 32 + lane;
      const float2 st = STATS[row];
      const float mean = st.x, rstd = st.y;
      wait(af + acc, (lt / T::NACC) & 1);
      tc_fence_after();
      float a = 0.f, bsum = 0.f;                     // sum of g_y gamma, sum of g_y gamma xhat
#pragma unroll 1
      for (int kk = 0; kk < NKW; ++kk, ++n) {
        const int k = hv + NH2 * kk, sl = n % T::NSL;
        if (lane == 0 && lt_ < ntiles) { bulk_wait_read<T::NSL - 2>(); load_one(); }   // the slot of job n - 2: its stores have been read
        float v[32];
        tmem_ld32(tmem_at(tmem + acc * C + k * 32, qq * 32, 0), v);
        tmem_wait_ld();
        wait(glw + sl, (n / T::NSL) & 1);
        const unsigned char* rowp = ring + sl * HB_SLOT + lane * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint4 xv = *reinterpret_cast<const uint4*>(rowp + ((c ^ ((lane >> 1) & 3)) << 4));
          const uint32_t w[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 f = bf2(w[e]);
            const int i = c * 8 + 2 * e, col = k * 32 + i;
            const float g0 = v[i] * sG[col], g1 = v[i + 1] * sG[col + 1];
            a += g0 + g1;
            bsum = fmaf(g0, (f.x - mean) * rstd, fmaf(g1, (f.y - mean) * rstd, bsum));
          }
        }
        __syncwarp();
        if (lane == 0) bulk_commit();                // an empty group: every job commits one, so the ring's read-wait counts jobs
      }
      if (NH2 == 2) {                                // the quarter's two warps: combine the partial sums of their slices
        float2* xc = sXc + (lt & 1) * 2 * WBM;
        xc[hv * WBM + qq * 32 + lane] = make_float2(a, bsum);
        named_sync(1 + qq, 64);
        const float2 o2 = xc[(hv ^ 1) * WBM + qq * 32 + lane];
        a += o2.x; bsum += o2.y;
      }
      const float ma = a * (1.f / C), mb = bsum * (1.f / C);
#pragma unroll 1
      for (int kk = 0; kk < NKW; ++kk, ++n) {
        const int k = hv + NH2 * kk, sl = n % T::NSL;
        if (lane == 0 && lt_ < ntiles) { bulk_wait_read<T::NSL - 2>(); load_one(); }
        float v[32];
        tmem_ld32(tmem_at(tmem + acc * C + k * 32, qq * 32, 0), v);
        tmem_wait_ld();
        if (kk == NKW - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + acc); }
        wait(glw + sl, (n / T::NSL) & 1);
        unsigned char* slot = ring + sl * HB_SLOT;
        unsigned char* rowp = slot + lane * 64;
        uint4 xv[4], dv[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int off = (c ^ ((lane >> 1) & 3)) << 4;
          xv[c] = *reinterpret_cast<const uint4*>(rowp + off);
          dv[c] = *reinterpret_cast<const uint4*>(rowp + 2048 + off);
        }
        uint32_t wdx[16], wxh[16];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t xw[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w}, dww[4] = {dv[c].x, dv[c].y, dv[c].z, dv[c].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 f = bf2(xw[e]), d = bf2(dww[e]);
            const int i = c * 8 + 2 * e, col = k * 32 + i;
            const float h0 = (f.x - mean) * rstd, h1 = (f.y - mean) * rstd;
            const float x0 = rstd * (v[i] * sG[col] - ma - h0 * mb), x1 = rstd * (v[i + 1] * sG[col + 1] - ma - h1 * mb);
            wdx[c * 4 + e] = pk2(d.x + x0, d.y + x1);
            wxh[c * 4 + e] = pk2(h0, h1);
          }
        }
        __syncwarp();                                // every lane has read its x | dy: overwrite in place
        const uint32_t* srcs[2] = {wdx, wxh};
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          unsigned char* rp = slot + q * 2048 + lane * 64;
#pragma unroll
          for (int c = 0; c < 4; ++c)
            *reinterpret_cast<uint4*>(rp + ((c ^ ((lane >> 1) & 3)) << 4)) =
                make_uint4(srcs[q][c * 4], srcs[q][c * 4 + 1], srcs[q][c * 4 + 2], srcs[q][c * 4 + 3]);
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          store_2d(&dxmap, slot, k * 32, t * WBM + qq * 32);
          store_2d(&xhmap, slot + 2048, k * 32, t * WBM + qq * 32);
          if (k == NK - 1) {                            // xhat's extra columns: 1 at C, zeros after
            store_2d(&xhmap, sPad, C, t * WBM + qq * 32);
            store_2d(&xhmap, sPad + 2048, C + 32, t * WBM + qq * 32);
          }
          bulk_commit();
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---- parameter gradients from G = D^T [xhat | 1] (fp32 [4 HDp + 64][C + 64]):  for a projection row o of W (padded layout,
// the rows of wcat): dW[o][c] = G[o][c] gamma[c] + G[o][C] beta[c];  dgamma[c] = sum_o W[o][c] G[o][c];  dbeta[c] = sum_o W[o][c] G[o][C];
// rows of zero-padded channels are skipped in the outputs (their W rows are 0).  Grid (C/32 + dWo blocks, row chunks of FR rows):
// threads (8 rows x 32 columns), coalesced along c; each block leaves its dgamma / dbeta partial in PART [chunk][2][C], which
// tri_wfinish_sum adds in chunk order.  Blocks x >= C/32 copy dWo [C][HDp] -> [C][HD] without the padded channels.
constexpr int FR = 32;

__device__ __forceinline__ void st_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <typename OutT>
__global__ void __launch_bounds__(256) tri_wfinish(const float* __restrict__ G, const __nv_bfloat16* __restrict__ W, const float* __restrict__ GAM,
    const float* __restrict__ BET, int C, int HDp, int NH, int hd, OutT* __restrict__ DW, OutT* __restrict__ DWB, float* __restrict__ PART,
    const float* __restrict__ DWO, OutT* __restrict__ DWOU) {
  const int ncb = C / 32, ldg = C + 64, HD = NH * hd, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
  if ((int)blockIdx.x >= ncb) {
    const long i = ((long)(blockIdx.x - ncb) * gridDim.y + blockIdx.y) * 256 + threadIdx.x;   // an element of dWo [C][HD]
    if (i >= (long)C * HD) return;
    const int c = (int)(i / HD), o = (int)(i % HD), op = (o / hd) * 32 + o % hd;
    st_out(DWOU + i, DWO[(long)c * HDp + op]);
    return;
  }
  const int c = blockIdx.x * 32 + tx, r0 = blockIdx.y * FR, r1 = min(r0 + FR, 4 * HDp + NH);
  const float g = GAM[c], b = BET[c];
  float dg = 0.f, db = 0.f;
  for (int r = r0 + ty; r < r1; r += 8) {                              // wcat and G share the row index (bias block / db columns at 4 HDp)
    const float w = __bfloat162float(W[(long)r * C + c]);
    const float gv = G[(long)r * ldg + c], gs = G[(long)r * ldg + C];
    dg = fmaf(w, gv, dg);
    db = fmaf(w, gs, db);
    const float v = gv * g + gs * b;
    if (r < 4 * HDp) {
      const int t = r / HDp, rr = r % HDp, head = rr / 32, ch = rr % 32;
      if (ch < hd) st_out(DW + ((long)t * HD + head * hd + ch) * C + c, v);
    } else {
      st_out(DWB + (long)(r - 4 * HDp) * C + c, v);
    }
  }
  __shared__ float red[2][8][32];
  red[0][ty][tx] = dg;
  red[1][ty][tx] = db;
  __syncthreads();
  if (ty == 0) {
    float sg = 0.f, sb = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { sg += red[0][i][tx]; sb += red[1][i][tx]; }
    PART[((long)blockIdx.y * 2) * C + c] = sg;
    PART[((long)blockIdx.y * 2 + 1) * C + c] = sb;
  }
}

// PART [NCH][2][C] -> dgamma | dbeta [2][C] (fixed order)
__global__ void __launch_bounds__(256) tri_wfinish_sum(const float* __restrict__ PART, int NCH, int C, float* __restrict__ OUT) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * C) return;
  float s = 0.f;
  for (int k = 0; k < NCH; ++k) s += PART[(long)k * 2 * C + i];
  OUT[i] = s;
}

}  // namespace

// x [R, C] bf16 contiguous, w / b [C] (any float dtype) -> y [R, C] bf16
torch::Tensor tri_ln_rows_h(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, torch::Tensor xcopy) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: [R, C] bf16 contiguous");
  const long R = x.size(0), C = x.size(1);
  auto wf = w.to(torch::kFloat32).contiguous(), bf = b.to(torch::kFloat32).contiguous();
  TORCH_CHECK(wf.numel() == C && bf.numel() == C, "LN params [C]");
  auto y = torch::empty_like(x);
  __nv_bfloat16* xc = nullptr;
  if (xcopy.defined() && xcopy.numel() > 0) {
    TORCH_CHECK(xcopy.sizes() == x.sizes() && xcopy.is_contiguous() && xcopy.scalar_type() == torch::kBFloat16, "xcopy: like x");
    xc = reinterpret_cast<__nv_bfloat16*>(xcopy.data_ptr<at::BFloat16>());
  }
  const dim3 grid((unsigned)((R + 7) / 8));
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  switch (C) {
    case 64: tri_ln_rows<64><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 128: tri_ln_rows<128><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 192: tri_ln_rows<192><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 256: tri_ln_rows<256><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 320: tri_ln_rows<320><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 384: tri_ln_rows<384><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 448: tri_ln_rows<448><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    case 512: tri_ln_rows<512><<<grid, 256, 0, st>>>(xp, wf.data_ptr<float>(), bf.data_ptr<float>(), yp, R, (float)eps, xc); break;
    default: TORCH_CHECK(false, "tri_ln_rows: C must be a multiple of 64 in 64 .. 512, got ", C);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// P [R, ld] bf16 (rows (b, j, k)), the NH bias columns at cb; mask [B, L] bool or empty -> bias [B, NH, L, L] bf16
torch::Tensor tri_bias_heads_h(torch::Tensor P, int64_t cb, int64_t NH, torch::Tensor mask, int64_t B, int64_t L) {
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16 && P.dim() == 2 && P.stride(1) == 1, "P: [R, ld] bf16, dense rows");
  const long R = P.size(0);
  TORCH_CHECK(R == B * L * L && cb + NH <= P.size(1), "P rows = B*L*L, bias columns inside P");
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  auto bias = torch::empty({B, NH, L, L}, P.options());
  tri_bias_heads<<<(unsigned)((R + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(P.data_ptr<at::BFloat16>()), P.stride(0), (int)cb, (int)NH, mp,
      reinterpret_cast<__nv_bfloat16*>(bias.data_ptr<at::BFloat16>()), (int)L, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bias;
}

// g [R, HD] bf16 with dense rows at any 16-byte-multiple stride, o [R, HD] bf16 contiguous -> u = sigmoid(g) o o [R, HD]
torch::Tensor tri_gate_mul_h(torch::Tensor g, torch::Tensor o) {
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == torch::kBFloat16 && g.dim() == 2 && g.stride(1) == 1 && g.stride(0) % 8 == 0, "g: [R, HD] bf16");
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.sizes() == g.sizes(), "o: [R, HD] bf16 contiguous");
  const long R = g.size(0);
  const int HD = (int)g.size(1);
  TORCH_CHECK(HD % 8 == 0, "HD % 8");
  auto u = torch::empty_like(o);
  const long n = R * (HD / 8);
  tri_gate_mul<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()), g.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(o.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(u.data_ptr<at::BFloat16>()), HD, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return u;
}


// ---- fused wide tail: g [R, HDp] (dense rows at a 16-byte-multiple stride), o [R, HDp], x [R, C], wo [C, HDp] -> x + (sigmoid(g) o o) . wo^T
template <int C>
static void launch_wtail(const torch::Tensor& g, const torch::Tensor& o, const torch::Tensor& x, const torch::Tensor& wo, torch::Tensor& out,
                         const __nv_bfloat16* drop, long L) {
  using T = WT<C>;
  const long R = x.size(0), HDp = o.size(1);
  auto m2 = [&](const torch::Tensor& t, uint64_t cols, uint64_t rows, uint64_t ld, uint32_t b0, uint32_t b1, CUtensorMapSwizzle sw, const char* w) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {ld}, {b0, b1}, sw, w); };
  CUtensorMap gm = m2(g, HDp, R, g.stride(0), 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "g");
  CUtensorMap om = m2(o, HDp, R, HDp, 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "o");
  CUtensorMap wm = m2(wo, HDp, C, HDp, 64, T::NNR, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  CUtensorMap ym = m2(out, C, R, C, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "out");
  CUtensorMap xm = m2(x, C, R, C, 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wtail_sm100<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM)); attr = true; }
  const int ntiles = (int)(R / WBM);
  tri_wtail_sm100<C><<<std::min(ntiles * T::NH, num_sms(x.device().index())), 448, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      ntiles, (int)(HDp / 64), drop == nullptr ? 1 : 0, gm, om, wm, xm, ym, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      drop, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor tri_wtail_h(torch::Tensor g, torch::Tensor o, torch::Tensor x, torch::Tensor wo, torch::Tensor drop, int64_t L) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: [R, C] bf16");
  const long R = x.size(0), C = x.size(1);
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.dim() == 2 && o.size(0) == R && o.size(1) % 64 == 0, "o: [R, HDp]");
  TORCH_CHECK(g.scalar_type() == torch::kBFloat16 && g.sizes() == o.sizes() && g.stride(1) == 1 && g.stride(0) % 8 == 0, "g: [R, HDp]");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.size(0) == C && wo.size(1) == o.size(1), "wo: [C, HDp]");
  TORCH_CHECK(R % WBM == 0, "rows % 128");
  const __nv_bfloat16* dp = nullptr;
  if (drop.defined() && drop.numel() > 0) {
    TORCH_CHECK(drop.scalar_type() == torch::kBFloat16 && drop.is_contiguous() && drop.numel() * L == R * C && R % (L * L) == 0,
                "drop: [B, L, C] bf16 contiguous");
    dp = reinterpret_cast<const __nv_bfloat16*>(drop.data_ptr<at::BFloat16>());
  }
  auto out = torch::empty_like(x);
  switch (C) {
    case 64: launch_wtail<64>(g, o, x, wo, out, dp, L); break;
    case 128: launch_wtail<128>(g, o, x, wo, out, dp, L); break;
    case 192: launch_wtail<192>(g, o, x, wo, out, dp, L); break;
    case 256: launch_wtail<256>(g, o, x, wo, out, dp, L); break;
    case 320: launch_wtail<320>(g, o, x, wo, out, dp, L); break;
    case 384: launch_wtail<384>(g, o, x, wo, out, dp, L); break;
    case 448: launch_wtail<448>(g, o, x, wo, out, dp, L); break;
    case 512: launch_wtail<512>(g, o, x, wo, out, dp, L); break;
    default: TORCH_CHECK(false, "tri_wtail: C must be a multiple of 64 in 64 .. 512");
  }
  return out;
}

// ---- fused wide front: x [R, C], LN w / b [C], W [4 HDp + 16, C] bf16 -> (q, k, v, g [R, HDp] bf16, bias [B, NH, L, L] bf16)
template <int C, int ND>
static void launch_wfront_nd(const torch::Tensor& x, const torch::Tensor& lw, const torch::Tensor& lb, double eps, const torch::Tensor& W,
                             long HDp, long NH, const bool* mp, long L, long NBR, std::vector<torch::Tensor>& outs, float2* stp) {
  using T = WF<C, ND>;
  const long R = x.size(0);
  auto m2 = [&](const torch::Tensor& t, uint64_t cols, uint64_t rows, uint64_t ld, uint32_t b0, uint32_t b1, CUtensorMapSwizzle sw, const char* w) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {ld}, {b0, b1}, sw, w); };
  CUtensorMap xm = m2(x, C, R, C, 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap wm = m2(W, C, W.size(0), C, 64, (uint32_t)(NBR / 2), CU_TENSOR_MAP_SWIZZLE_128B, "W");
  CUtensorMap wb = m2(W, C, W.size(0), C, 64, 8, CU_TENSOR_MAP_SWIZZLE_128B, "Wb");
  CUtensorMap om[4];
  for (int i = 0; i < 4; ++i) om[i] = m2(outs[i], HDp, R, HDp, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "qkvg");
  const int ntiles = (int)(R / WBM);
  TORCH_CHECK(ntiles % 2 == 0, "tri_wfront: an even number of 128-row tiles");
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.blockDim = dim3((10 + ND) * 32); cfg.dynamicSmemBytes = T::SMEM; cfg.stream = at::cuda::getCurrentCUDAStream(); cfg.attrs = at; cfg.numAttrs = 1;
  static int ncl[16] = {0};                               // co-resident 2-CTA clusters (the persistent grid must be one wave)
  const int dev = x.device().index();
  if (ncl[dev] == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wfront_sm100<C, ND>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    cfg.gridDim = dim3(num_sms(dev) & ~1);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&ncl[dev], reinterpret_cast<const void*>(&tri_wfront_sm100<C, ND>), &cfg));
    TORCH_CHECK(ncl[dev] >= 1, "tri_wfront: no 2-CTA cluster fits");
  }
  cfg.gridDim = dim3(std::min(2 * ncl[dev], ntiles));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tri_wfront_sm100<C, ND>, ntiles, (int)L, (int)HDp, (int)NH, (int)NBR, (float)eps, xm, wm, wb,
                                    Maps4{{om[0], om[1], om[2], om[3]}}, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                                    (const float*)lw.data_ptr<float>(), (const float*)lb.data_ptr<float>(), mp,
                                    reinterpret_cast<__nv_bfloat16*>(outs[4].data_ptr<at::BFloat16>()), stp));
}

template <int C>
static void launch_wfront(const torch::Tensor& x, const torch::Tensor& lw, const torch::Tensor& lb, double eps, const torch::Tensor& W,
                          long HDp, long NH, const bool* mp, long L, std::vector<torch::Tensor>& outs, float2* stp) {
  long NBR = 0;                                           // rows per output block: the largest divisor of HDp <= 256, a multiple of 32
  for (long d = 256; d >= 32 && !NBR; d -= 32) if (HDp % d == 0) NBR = d;
  if (C <= 256 && NBR >= 128) launch_wfront_nd<C, 8>(x, lw, lb, eps, W, HDp, NH, mp, L, NBR, outs, stp);
  else launch_wfront_nd<C, 4>(x, lw, lb, eps, W, HDp, NH, mp, L, NBR, outs, stp);
}

std::vector<torch::Tensor> tri_wfront_h(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, torch::Tensor W, int64_t NH,
                                        torch::Tensor mask, int64_t B, int64_t L, torch::Tensor stats) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: [R, C] bf16");
  const long R = x.size(0), C = x.size(1);
  TORCH_CHECK(R == B * L * L && L % WBM == 0, "rows = B*L*L, L % 128");
  TORCH_CHECK(W.scalar_type() == torch::kBFloat16 && W.is_contiguous() && W.size(1) == C && (W.size(0) - 16) % 256 == 0, "W: [4 HDp + 16, C]");
  const long HDp = (W.size(0) - 16) / 4;
  TORCH_CHECK(HDp % 64 == 0 && HDp <= 512 && NH >= 1 && NH <= 16, "HDp 64 .. 512 (a multiple of 64), heads <= 16");
  auto lw = w.to(torch::kFloat32).contiguous(), lb = b.to(torch::kFloat32).contiguous();
  float2* stp = nullptr;                                  // stats: fp32 [R, 2] (mean, rstd) for the backward, or empty
  if (stats.numel() > 0) {
    TORCH_CHECK(stats.scalar_type() == torch::kFloat32 && stats.is_contiguous() && stats.numel() == 2 * R, "stats: fp32 [R, 2]");
    stp = reinterpret_cast<float2*>(stats.data_ptr<float>());
  }
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  auto opt = x.options();
  std::vector<torch::Tensor> outs = {torch::empty({R, HDp}, opt), torch::empty({R, HDp}, opt), torch::empty({R, HDp}, opt),
                                     torch::empty({R, HDp}, opt), torch::empty({B, NH, L, L}, opt)};
  switch (C) {
    case 64: launch_wfront<64>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 128: launch_wfront<128>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 192: launch_wfront<192>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 256: launch_wfront<256>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 320: launch_wfront<320>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 384: launch_wfront<384>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 448: launch_wfront<448>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    case 512: launch_wfront<512>(x, lw, lb, eps, W, HDp, NH, mp, L, outs, stp); break;
    default: TORCH_CHECK(false, "tri_wfront: C must be a multiple of 64 in 64 .. 512");
  }
  return outs;
}


// ---- backward host wrappers
torch::Tensor tri_scale_rows_h(torch::Tensor dy, torch::Tensor drop, int64_t L) {
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == torch::kBFloat16 && dy.is_contiguous() && dy.dim() == 2, "dy: [R, C] bf16");
  const long R = dy.size(0), C = dy.size(1);
  TORCH_CHECK(drop.scalar_type() == torch::kBFloat16 && drop.is_contiguous() && drop.numel() * L == R * C && C % 8 == 0, "drop: [B, L, C]");
  auto out = torch::empty_like(dy);
  const long n = R * (C / 8);
  tri_scale_rows<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(drop.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), (int)C, (int)L, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tri_db_rows_h(torch::Tensor db, torch::Tensor out, int64_t B, int64_t L) {
  TORCH_CHECK(db.is_cuda() && db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.dim() == 4 && db.size(0) == B, "db: fp32 [B, H, L, L]");
  const int H = (int)db.size(1);
  TORCH_CHECK(H <= 64, "heads <= 64");
  const long R = B * L * L;
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(0) == R && out.size(1) == 64 && out.stride(1) == 1 &&
              out.stride(0) % 8 == 0, "out: [R, 64] bf16 view, dense rows at a 16-byte-multiple stride");
  tri_db_rows<<<(unsigned)((R + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(db.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), out.stride(0), H, (int)L, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

static long pick_rows(long HDp, long maxn) {             // the largest divisor of HDp <= maxn that is a multiple of 32
  for (long d = maxn; d >= 32; d -= 32) if (HDp % d == 0) return d;
  return 32;
}

template <int C>
static void launch_wgbwd(const torch::Tensor& dys, const torch::Tensor& g, const torch::Tensor& o, const torch::Tensor& wot, long L,
                         std::vector<torch::Tensor>& outs) {                      // outs[1] (dg) may be a strided view
  using T = WG<C>;
  const long R = dys.size(0), HDp = g.size(1), NBR = pick_rows(HDp, T::MAXN);
  auto m2 = [&](const torch::Tensor& t, uint64_t cols, uint64_t rows, uint64_t ld, uint32_t b0, uint32_t b1, CUtensorMapSwizzle sw, const char* w) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {ld}, {b0, b1}, sw, w); };
  CUtensorMap am = m2(dys, C, R, C, 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "dy");
  CUtensorMap wm = m2(wot, C, HDp, C, 64, (uint32_t)NBR, CU_TENSOR_MAP_SWIZZLE_128B, "wo^T");
  CUtensorMap dom = m2(outs[0], HDp, R, HDp, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "do");
  CUtensorMap dgm = m2(outs[1], HDp, R, outs[1].stride(0), 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "dg");
  CUtensorMap um = m2(outs[2], HDp, R, HDp, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "u");
  CUtensorMap gm = m2(g, HDp, R, HDp, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "g");
  CUtensorMap omm = m2(o, HDp, R, HDp, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "o");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wgbwd_sm100<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM)); attr = true; }
  const int ntiles = (int)(R / WBM);
  tri_wgbwd_sm100<C><<<std::min(ntiles, num_sms(dys.device().index())), 320, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      ntiles, (int)L, (int)HDp, (int)NBR, am, wm, dom, dgm, um, gm, omm, outs[3].data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dy_s [R, C], g / o [R, HDp], wo^T [HDp, C] -> (do, dg, u [R, HDp] bf16, delta fp32 [B, L, HDp/32, L])
std::vector<torch::Tensor> tri_wgbwd_h(torch::Tensor dys, torch::Tensor g, torch::Tensor o, torch::Tensor wot, torch::Tensor dg_out,
                                       int64_t B, int64_t L) {
  TORCH_CHECK(dys.is_cuda() && dys.scalar_type() == torch::kBFloat16 && dys.is_contiguous() && dys.dim() == 2, "dy_s: [R, C] bf16");
  const long R = dys.size(0), C = dys.size(1);
  TORCH_CHECK(R == B * L * L && R % WBM == 0, "rows = B*L*L, a multiple of 128");
  for (const auto* t : {&g, &o}) TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->dim() == 2 && t->size(0) == R, "g / o: [R, HDp]");
  const long HDp = g.size(1);
  TORCH_CHECK(o.size(1) == HDp && HDp % 64 == 0 && HDp <= 512, "HDp 64 .. 512");
  TORCH_CHECK(wot.scalar_type() == torch::kBFloat16 && wot.is_contiguous() && wot.size(0) == HDp && wot.size(1) == C, "wo^T: [HDp, C]");
  auto opt = g.options();
  TORCH_CHECK(dg_out.scalar_type() == torch::kBFloat16 && dg_out.dim() == 2 && dg_out.size(0) == R && dg_out.size(1) == HDp &&
              dg_out.stride(1) == 1 && dg_out.stride(0) % 8 == 0, "dg_out: [R, HDp] bf16 view");
  std::vector<torch::Tensor> outs = {torch::empty({R, HDp}, opt), dg_out, torch::empty({R, HDp}, opt),
                                     torch::empty({B, L, HDp / 32, L}, opt.dtype(torch::kFloat32))};
  switch (C) {
    case 64: launch_wgbwd<64>(dys, g, o, wot, L, outs); break;
    case 128: launch_wgbwd<128>(dys, g, o, wot, L, outs); break;
    case 192: launch_wgbwd<192>(dys, g, o, wot, L, outs); break;
    case 256: launch_wgbwd<256>(dys, g, o, wot, L, outs); break;
    case 320: launch_wgbwd<320>(dys, g, o, wot, L, outs); break;
    case 384: launch_wgbwd<384>(dys, g, o, wot, L, outs); break;
    case 448: launch_wgbwd<448>(dys, g, o, wot, L, outs); break;
    case 512: launch_wgbwd<512>(dys, g, o, wot, L, outs); break;
    default: TORCH_CHECK(false, "tri_wgbwd: C must be a multiple of 64 in 64 .. 512");
  }
  return outs;
}

template <int C>
static void launch_whbwd(const torch::Tensor& Dall, const torch::Tensor& x, const torch::Tensor& dy, const torch::Tensor& lw,
                         const torch::Tensor& stats, const torch::Tensor& wt, torch::Tensor& dx, torch::Tensor& xh) {
  using T = WH<C>;
  const long R = x.size(0), HDp = (Dall.size(1) - 64) / 4;
  auto m2 = [&](const torch::Tensor& t, uint64_t cols, uint64_t rows, uint64_t ld, uint32_t b0, uint32_t b1, CUtensorMapSwizzle sw, const char* w) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {ld}, {b0, b1}, sw, w); };
  CUtensorMap am = m2(Dall, Dall.size(1), R, Dall.size(1), 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "D");
  CUtensorMap wm = m2(wt, wt.size(1), C, wt.size(1), 64, T::NNR, CU_TENSOR_MAP_SWIZZLE_128B, "W^T");
  CUtensorMap dxm = m2(dx, C, R, C, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "dx");
  CUtensorMap xhm = m2(xh, C + 64, R, C + 64, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "xhat");
  CUtensorMap xm = m2(x, C, R, C, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "x");
  CUtensorMap dym = m2(dy, C, R, C, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "dy");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_whbwd_sm100<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM)); attr = true; }
  const int ntiles = (int)(R / WBM);
  tri_whbwd_sm100<C><<<std::min(ntiles, num_sms(x.device().index())), (2 + T::ND) * 32, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      ntiles, (int)HDp, am, wm, dxm, xhm, xm, dym, reinterpret_cast<const float2*>(stats.data_ptr<float>()), lw.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// D = [dq | dk | dv | dg | db rows] [R, 4 HDp + 64] bf16; x, dy [R, C]; LN gamma [C]; W^T = [Wq; Wk; Wv; Wg; Wb64]^T [C, 4 HDp + 64]
// -> (dx = dy + LN-backward(D . W) [R, C] bf16, xhat | 1 | 0.. [R, C + 64] bf16)
std::vector<torch::Tensor> tri_whbwd_h(torch::Tensor Dall, torch::Tensor x, torch::Tensor dy, torch::Tensor lnw, torch::Tensor stats,
                                       torch::Tensor wt) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: [R, C] bf16");
  const long R = x.size(0), C = x.size(1);
  TORCH_CHECK(R % WBM == 0, "rows % 128");
  TORCH_CHECK(dy.sizes() == x.sizes() && dy.is_contiguous() && dy.scalar_type() == torch::kBFloat16, "dy: like x");
  TORCH_CHECK(Dall.scalar_type() == torch::kBFloat16 && Dall.is_contiguous() && Dall.dim() == 2 && Dall.size(0) == R && (Dall.size(1) - 64) % 256 == 0,
              "D: [R, 4 HDp + 64] bf16");
  const long HDp = (Dall.size(1) - 64) / 4;
  TORCH_CHECK(HDp >= 64 && HDp <= 512, "HDp 64 .. 512");
  TORCH_CHECK(wt.scalar_type() == torch::kBFloat16 && wt.is_contiguous() && wt.size(0) == C && wt.size(1) == Dall.size(1), "W^T: [C, 4 HDp + 64]");
  auto lw = lnw.to(torch::kFloat32).contiguous();
  TORCH_CHECK(lw.numel() == C, "LN gamma [C]");
  TORCH_CHECK(stats.scalar_type() == torch::kFloat32 && stats.is_contiguous() && stats.numel() == 2 * R, "stats: fp32 [R, 2] (the forward's)");
  auto dx = torch::empty_like(x), xh = torch::empty({R, C + 64}, x.options());
  switch (C) {
    case 64: launch_whbwd<64>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 128: launch_whbwd<128>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 192: launch_whbwd<192>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 256: launch_whbwd<256>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 320: launch_whbwd<320>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 384: launch_whbwd<384>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 448: launch_whbwd<448>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    case 512: launch_whbwd<512>(Dall, x, dy, lw, stats, wt, dx, xh); break;
    default: TORCH_CHECK(false, "tri_whbwd: C must be a multiple of 64 in 64 .. 512");
  }
  return {dx, xh};
}

// G = D^T [xhat | 1] fp32 [4 HDp + 64, C + 64]; wcat [4 HDp + 16, C] bf16; LN gamma / beta [C]; dWo_p fp32 [C, HDp]
// -> (dW_q|k|v|g [4, HD, C], dWb [NH, C], dWo [C, HD] in the parameters' dtype (fp32 or bf16), dgamma | dbeta fp32 [2, C])
std::vector<torch::Tensor> tri_wfinish_h(torch::Tensor G, torch::Tensor wcat, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor dwo,
                                         int64_t NH, int64_t hd, bool fp32_out) {
  TORCH_CHECK(G.is_cuda() && G.scalar_type() == torch::kFloat32 && G.is_contiguous() && G.dim() == 2, "G: fp32 [4 HDp + 64, C + 64]");
  const long C = G.size(1) - 64, HDp = (G.size(0) - 64) / 4, HD = NH * hd;
  TORCH_CHECK(C % 32 == 0 && wcat.scalar_type() == torch::kBFloat16 && wcat.is_contiguous() && wcat.size(0) == 4 * HDp + 16 && wcat.size(1) == C,
              "wcat: [4 HDp + 16, C] bf16");
  TORCH_CHECK(dwo.scalar_type() == torch::kFloat32 && dwo.is_contiguous() && dwo.size(0) == C && dwo.size(1) == HDp, "dWo: fp32 [C, HDp]");
  TORCH_CHECK(NH * 32 == HDp && (hd == 16 || hd == 32), "heads x 32 = HDp, head dim 16 / 32");
  auto gam = lnw.to(torch::kFloat32).contiguous(), bet = lnb.to(torch::kFloat32).contiguous();
  auto opt = G.options().dtype(fp32_out ? torch::kFloat32 : torch::kBFloat16);
  auto dw = torch::empty({4, HD, C}, opt), dwb = torch::empty({NH, C}, opt), dwou = torch::empty({C, HD}, opt);
  const int nch = (int)((4 * HDp + NH + FR - 1) / FR);
  auto part = torch::empty({nch, 2, C}, G.options()), dgb = torch::empty({2, C}, G.options());
  const long ncopy = ((C * HD + 255) / 256 + nch - 1) / nch;          // dWo copy blocks along x (each covers nch x 256 elements)
  const dim3 grid((unsigned)(C / 32 + ncopy), (unsigned)nch);
  auto st = at::cuda::getCurrentCUDAStream();
  if (fp32_out)
    tri_wfinish<float><<<grid, 256, 0, st>>>(G.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(wcat.data_ptr<at::BFloat16>()),
        gam.data_ptr<float>(), bet.data_ptr<float>(), (int)C, (int)HDp, (int)NH, (int)hd, dw.data_ptr<float>(), dwb.data_ptr<float>(),
        part.data_ptr<float>(), dwo.data_ptr<float>(), dwou.data_ptr<float>());
  else
    tri_wfinish<__nv_bfloat16><<<grid, 256, 0, st>>>(G.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(wcat.data_ptr<at::BFloat16>()),
        gam.data_ptr<float>(), bet.data_ptr<float>(), (int)C, (int)HDp, (int)NH, (int)hd,
        reinterpret_cast<__nv_bfloat16*>(dw.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dwb.data_ptr<at::BFloat16>()),
        part.data_ptr<float>(), dwo.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dwou.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tri_wfinish_sum<<<(unsigned)((2 * C + 255) / 256), 256, 0, st>>>(part.data_ptr<float>(), nch, (int)C, dgb.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dw, dwb, dwou, dgb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_scale_rows", &tri_scale_rows_h, "dy o drop[b, j, :]", py::arg("dy"), py::arg("drop"), py::arg("L"));
  m.def("tri_db_rows", &tri_db_rows_h, "dbias [B, H, L, L] fp32 -> rows [R, 64] bf16 (into a view)", py::arg("db"), py::arg("out"),
        py::arg("B"), py::arg("L"));
  m.def("tri_wgbwd", &tri_wgbwd_h, "wide gate backward: (do, dg, u, delta); dg into a view", py::arg("dys"), py::arg("g"), py::arg("o"),
        py::arg("wot"), py::arg("dg_out"), py::arg("B"), py::arg("L"));
  m.def("tri_whbwd", &tri_whbwd_h, "wide head backward: (dx, xhat | 1)", py::arg("D"), py::arg("x"), py::arg("dy"), py::arg("lnw"),
        py::arg("stats"), py::arg("wt"));
  m.def("tri_wfinish", &tri_wfinish_h, "wide parameter gradients from G = D^T [xhat | 1]", py::arg("G"), py::arg("wcat"), py::arg("lnw"),
        py::arg("lnb"), py::arg("dwo"), py::arg("NH"), py::arg("hd"), py::arg("fp32_out"));
  m.def("tri_wfront", &tri_wfront_h, "fused LN + [q | k | v | g | bias] projections for any width",
        py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("W"), py::arg("NH"), py::arg("mask"), py::arg("B"), py::arg("L"),
        py::arg("stats"));
  m.def("tri_wtail", &tri_wtail_h, "fused x + (sigmoid(g) o o) . Wo^T for any width", py::arg("g"), py::arg("o"), py::arg("x"), py::arg("wo"),
        py::arg("drop"), py::arg("L"));
  m.def("tri_ln_rows", &tri_ln_rows_h, "LayerNorm over C (64 .. 512) rows: bf16 -> bf16", py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("xcopy"));
  m.def("tri_bias_heads", &tri_bias_heads_h, "bias columns of the projection -> head-major masked bias [B, NH, L, L]",
        py::arg("P"), py::arg("cb"), py::arg("NH"), py::arg("mask"), py::arg("B"), py::arg("L"));
  m.def("tri_gate_mul", &tri_gate_mul_h, "u = sigmoid(g) o o", py::arg("g"), py::arg("o"));
}
