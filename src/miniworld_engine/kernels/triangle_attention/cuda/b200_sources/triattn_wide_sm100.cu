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
// warp 0: TMA (per K chunk: g | o [128][64] and the Wo chunk [C][64]); warp 1: tcgen05; warps 2-9: u = sigmoid(g) o o over g;
// warps 10-13: drain (TMEM lane = row) + residual x (global loads) -> per-warp [32][32] staging -> TMA stores.
template <int C> struct WT {
  static constexpr int NN = C > 256 ? 2 : 1, NNR = C / NN;                     // MMA N split
  static constexpr int SB = 2 * WBM * 64 * 2 + C * 64 * 2;                     // stage bytes: g | o | W
  static constexpr int DR = 4 * WDRAIN_BUF * 32 * 32 * 2;
  static constexpr int NSTG = (WSMEM_CAP - DR) / SB > 4 ? 4 : (WSMEM_CAP - DR) / SB;
  static constexpr int NACC = C <= 256 ? 2 : 1;
  static constexpr int SMEM = 1024 + NSTG * SB + DR + 256;
  static_assert(NSTG >= 2, "tail stages");
};

template <int C>
__global__ void __launch_bounds__(448, 1) tri_wtail_sm100(int ntiles, int KC,
    const __grid_constant__ CUtensorMap gmap, const __grid_constant__ CUtensorMap omap,   // [R][HDp], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,                                               // Wo [C][HDp], box (64, NNR), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,                                               // out [R][C], box (32, 32), 64B swizzle
    const __nv_bfloat16* __restrict__ X) {                                                  // residual [R][C]
  using T = WT<C>;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* stg = smb;                                     // [NSTG][g | o | W]
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(smb + T::NSTG * T::SB);   // drain staging [4][WDRAIN_BUF][32][32]
  uint64_t* bars = reinterpret_cast<uint64_t*>(smb + T::NSTG * T::SB + T::DR);
  uint64_t* sf = bars;                 // [NSTG] chunk landed
  uint64_t* uf = sf + T::NSTG;         // [NSTG] count 8: u written
  uint64_t* se = uf + T::NSTG;         // [NSTG] its GEMMs retired
  uint64_t* af = se + T::NSTG;         // [NACC]
  uint64_t* ae = af + T::NACC;         // [NACC] count 4
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ae + T::NACC);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < T::NSTG; ++i) { bar_init(sf + i, 1); bar_init(uf + i, 8); bar_init(se + i, 1); }
    for (int i = 0; i < T::NACC; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto G = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB); };
  auto O = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB + WBM * 64 * 2); };
  auto Wc = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(stg + s * T::SB + 2 * WBM * 64 * 2); };

  if (warp == 0) {
    if (lane == 0) {
      int y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int kc = 0; kc < KC; ++kc, ++y) {
          const int s = y % T::NSTG;
          if (y >= T::NSTG) wait(se + s, ((y / T::NSTG) - 1) & 1);
          expect_tx(sf + s, T::SB);
          load_2d(&gmap, G(s), sf + s, kc * 64, t * WBM);
          load_2d(&omap, O(s), sf + s, kc * 64, t * WBM);
          for (int nh = 0; nh < T::NN; ++nh) load_2d(&wmap, Wc(s) + nh * T::NNR * 64, sf + s, kc * 64, nh * T::NNR);
        }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      constexpr uint32_t ID = idesc_bf16(128, T::NNR, 0, 0);
      int y = 0, lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int acc = lt % T::NACC;
        if (lt >= T::NACC) wait(ae + acc, ((lt / T::NACC) - 1) & 1);
        for (int kc = 0; kc < KC; ++kc, ++y) {
          const int s = y % T::NSTG;
          wait(uf + s, (y / T::NSTG) & 1);
          tc_fence_after();
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
#pragma unroll
            for (int nh = 0; nh < T::NN; ++nh)
              mma_ss(tmem + acc * C + nh * T::NNR, desc_k128(G(s) + ks * 16), desc_k128(Wc(s) + nh * T::NNR * 64 + ks * 16), ID,
                     (kc | ks) ? 1u : 0u);
          mma_commit(se + s);
        }
        mma_commit(af + acc);
      }
    }
  } else if (warp < 10) {
    // u = sigmoid(g) o o over g: row r, 16-byte chunks part*4 .. part*4+3 of the chunk's 8
    const int r = (warp - 2) * 16 + (lane & 15), part = lane >> 4;
    int y = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
      for (int kc = 0; kc < KC; ++kc, ++y) {
        const int s = y % T::NSTG;
        wait(sf + s, (y / T::NSTG) & 1);
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
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int acc = lt % T::NACC;
      const long row = (long)t * WBM + qq * 32 + lane;
      const uint4* xr = reinterpret_cast<const uint4*>(X + row * C);
      uint4 xv[2][4];
#pragma unroll
      for (int c = 0; c < 4; ++c) xv[0][c] = __ldg(xr + c);
      wait(af + acc, (lt / T::NACC) & 1);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < C / 32; ++k) {
        float v[32];
        tmem_ld32(tmem_at(tmem + acc * C + k * 32, qq * 32, 0), v);
        if (k + 1 < C / 32) {
#pragma unroll
          for (int c = 0; c < 4; ++c) xv[(k + 1) & 1][c] = __ldg(xr + (k + 1) * 4 + c);
        }
        tmem_wait_ld();
        if (k == C / 32 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + acc); }
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
        if (lane == 0) { store_2d(&ymap, buf, k * 32, t * WBM + qq * 32); bulk_commit(); }
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
// head count runtime.  W = [Wq; Wk; Wv; Wg; Wb (16 rows)] [4 HDp + 16][C].  Output blocks: each of q / k / v / g in NB <= 256-row
// pieces, then the 16-row bias block; two TMEM accumulators of 256 columns alternate between the GEMMs and the drain.
// warp 0: TMA (the W chunks [rows][64]); warp 1: tcgen05; warps 2-9: LayerNorm in place (two threads a row) and the next tile's
// x load (warp 2 lane 0); warps 10-13: drain -> per-warp staging -> TMA stores, the bias block -> head-major masked bias.
template <int C> struct WF {
  static constexpr int KC = C / 64;
  static constexpr int XB = WBM * C * 2;                                       // x / y tile bytes
  static constexpr int MAXN = C >= 384 ? 128 : 256;                           // rows per output block (a big y tile leaves room for
  static constexpr int WSB = MAXN * 64 * 2;                                    //  more, smaller W stages); W stage bytes
  static constexpr int DR = 4 * WDRAIN_BUF * 32 * 32 * 2;
  static constexpr int PB = 2 * C * 4;                                         // gamma | beta fp32
  static constexpr int NX = (2 * XB + 2 * WSB + DR + PB) <= WSMEM_CAP ? 2 : 1;
  static constexpr int NW0 = (WSMEM_CAP - NX * XB - DR - PB) / WSB;
  static constexpr int NW = NW0 > 4 ? 4 : NW0;
  static constexpr int SMEM = 1024 + NX * XB + NW * WSB + DR + PB + 512;
  static_assert(NW >= 2, "front W stages");
};

template <int C>
__global__ void __launch_bounds__(448, 1) tri_wfront_sm100(int ntiles, int L, int HDp, int NH, int NBR, float eps,
    const __grid_constant__ CUtensorMap xmap,                                   // x [R][C], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,                                   // W [4 HDp + 16][C], box (64, NBR), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,                                  // the same W, box (64, 16)
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, const __grid_constant__ CUtensorMap gmap,   // [R][HDp], box (32, 32), 64B swizzle
    const float* __restrict__ LNW, const float* __restrict__ LNB, const bool* __restrict__ MASK,
    __nv_bfloat16* __restrict__ BIAS) {                                         // [B][NH][L][L]
  using T = WF<C>;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);                          // [NX][KC][128][64]
  unsigned char* sW = smb + T::NX * T::XB;                                            // [NW][<=256][64]
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sW + T::NW * T::WSB);          // drain staging
  float* sP = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(sO) + T::DR); // gamma | beta
  uint64_t* bars = reinterpret_cast<uint64_t*>(sP + 2 * C);
  uint64_t* xf = bars;               // [NX] x landed
  uint64_t* yf = xf + T::NX;         // [NX] count 8: y written
  uint64_t* xe = yf + T::NX;         // [NX] the tile's GEMMs retired
  uint64_t* wf = xe + T::NX;         // [NW]
  uint64_t* we = wf + T::NW;         // [NW]
  uint64_t* af = we + T::NW;         // [2]
  uint64_t* ae = af + 2;             // [2] count 4
  uint32_t* tslot = reinterpret_cast<uint32_t*>(ae + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int NBT = HDp / NBR, NBLK = 4 * NBT + 1;                                // blocks per tensor, blocks a tile (NBR rows each)
  if (tid == 0) {
    for (int i = 0; i < T::NX; ++i) { bar_init(xf + i, 1); bar_init(yf + i, 8); bar_init(xe + i, 1); }
    for (int i = 0; i < T::NW; ++i) { bar_init(wf + i, 1); bar_init(we + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init_fence();
  }
  for (int i = tid; i < 2 * C; i += blockDim.x) sP[i] = i < C ? LNW[i] : LNB[i - C];
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto Wst = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(sW + s * T::WSB); };
  auto blk_rows = [&](int b) { return b < 4 * NBT ? NBR : 16; };
  auto blk_row0 = [&](int b) { return b < 4 * NBT ? (b / NBT) * HDp + (b % NBT) * NBR : 4 * HDp; };
  auto load_x = [&](int t, int xb) {
    expect_tx(xf + xb, T::XB);
    for (int kc = 0; kc < T::KC; ++kc) load_2d(&xmap, sX + (xb * T::KC + kc) * WBM * 64, xf + xb, kc * 64, t * WBM);
  };

  if (warp == 0) {
    if (lane == 0) {
      int y = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int b = 0; b < NBLK; ++b)
          for (int kc = 0; kc < T::KC; ++kc, ++y) {
            const int s = y % T::NW, rows = blk_rows(b);
            if (y >= T::NW) wait(we + s, ((y / T::NW) - 1) & 1);
            expect_tx(wf + s, rows * 128);
            load_2d(rows == 16 ? &wbmap : &wmap, Wst(s), wf + s, kc * 64, blk_row0(b));
          }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int y = 0, gb = 0, lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int xb = lt % T::NX;
        wait(yf + xb, (lt / T::NX) & 1);
        const __nv_bfloat16* Y = sX + xb * T::KC * WBM * 64;
        for (int b = 0; b < NBLK; ++b, ++gb) {
          const int acc = gb & 1, rows = blk_rows(b);
          const uint32_t id = idesc_bf16(128, rows, 0, 0);
          if (gb >= 2) wait(ae + acc, ((gb >> 1) - 1) & 1);
          for (int kc = 0; kc < T::KC; ++kc, ++y) {
            const int s = y % T::NW;
            wait(wf + s, (y / T::NW) & 1);
            tc_fence_after();
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              mma_ss(tmem + acc * 256, desc_k128(Y + kc * WBM * 64 + ks * 16), desc_k128(Wst(s) + ks * 16), id, (kc | ks) ? 1u : 0u);
            mma_commit(we + s);
          }
          mma_commit(af + acc);
        }
        mma_commit(xe + xb);
      }
    }
  } else if (warp < 10) {
    // LayerNorm in place: row r, the 16-byte chunks j of the row with j % 2 == part (the row's C / 8 chunks)
    const int r = (warp - 2) * 16 + (lane & 15), part = lane >> 4;
    const bool leader = warp == 2 && lane == 0;
    constexpr int NCH = C / 8;
    if (leader && (int)blockIdx.x < ntiles) load_x(blockIdx.x, 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int xb = lt % T::NX;
      if (T::NX == 2 && leader && t + (int)gridDim.x < ntiles) {     // prefetch the next tile into the other buffer
        if (lt >= 1) wait(xe + (xb ^ 1), ((lt - 1) / 2) & 1);
        load_x(t + gridDim.x, xb ^ 1);
      }
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
      if (lane == 0) arrive(yf + xb);
      if (T::NX == 1 && leader && t + (int)gridDim.x < ntiles) {     // one buffer: the next x waits for this tile's GEMMs
        wait(xe, lt & 1);
        load_x(t + gridDim.x, 0);
      }
    }
  } else {
    const int qq = warp & 3;
    const CUtensorMap* maps[4] = {&qmap, &kmap, &vmap, &gmap};
    __nv_bfloat16* ring = sO + qq * WDRAIN_BUF * 32 * 32;
    const __nv_bfloat16 lo = __float2bfloat16_rn(-3.3895313892515355e38f);
    int nst = 0, gb = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      const long row = (long)t * WBM + qq * 32 + lane;
      for (int b = 0; b < NBLK; ++b, ++gb) {
        const int acc = gb & 1;
        wait(af + acc, (gb >> 1) & 1);
        tc_fence_after();
        if (b == 4 * NBT) {                        // bias block: 16 columns, heads 0 .. NH-1 -> [b][h][j][k], masked keys
          float v[16];
          tmem_ld16(tmem_at(tmem + acc * 256, qq * 32, 0), v);
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) arrive(ae + acc);
          const int kx = (int)(row % L);
          const long bj = row / L;
          const int j = (int)(bj % L), bb = (int)(bj / L);
          const bool keep = MASK == nullptr || MASK[(long)bb * L + kx];
          for (int h = 0; h < NH; ++h) BIAS[(((long)bb * NH + h) * L + j) * L + kx] = keep ? __float2bfloat16_rn(v[h]) : lo;
          continue;
        }
        const int ten = b / NBT, c0 = (b % NBT) * NBR;
        for (int ch = 0; ch < NBR / 32; ++ch) {
          float v[32];
          tmem_ld32(tmem_at(tmem + acc * 256 + ch * 32, qq * 32, 0), v);
          tmem_wait_ld();
          if (ch == NBR / 32 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + acc); }
          uint32_t w[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) w[i] = pk2(v[2 * i], v[2 * i + 1]);
          __nv_bfloat16* buf = ring + (nst % WDRAIN_BUF) * 32 * 32;
          if (lane == 0 && nst >= WDRAIN_BUF) bulk_wait_read<WDRAIN_BUF - 1>();
          __syncwarp();
          unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + lane * 64;
#pragma unroll
          for (int c = 0; c < 4; ++c)
            *reinterpret_cast<uint4*>(rowp + ((c ^ ((lane >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) { store_2d(maps[ten], buf, c0 + ch * 32, t * WBM + qq * 32); bulk_commit(); }
          ++nst;
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
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
static void launch_wtail(const torch::Tensor& g, const torch::Tensor& o, const torch::Tensor& x, const torch::Tensor& wo, torch::Tensor& out) {
  using T = WT<C>;
  const long R = x.size(0), HDp = o.size(1);
  auto m2 = [&](const torch::Tensor& t, uint64_t cols, uint64_t rows, uint64_t ld, uint32_t b0, uint32_t b1, CUtensorMapSwizzle sw, const char* w) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {ld}, {b0, b1}, sw, w); };
  CUtensorMap gm = m2(g, HDp, R, g.stride(0), 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "g");
  CUtensorMap om = m2(o, HDp, R, HDp, 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "o");
  CUtensorMap wm = m2(wo, HDp, C, HDp, 64, T::NNR, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  CUtensorMap ym = m2(out, C, R, C, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "out");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wtail_sm100<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM)); attr = true; }
  const int ntiles = (int)(R / WBM);
  tri_wtail_sm100<C><<<std::min(ntiles, num_sms(x.device().index())), 448, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      ntiles, (int)(HDp / 64), gm, om, wm, ym, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor tri_wtail_h(torch::Tensor g, torch::Tensor o, torch::Tensor x, torch::Tensor wo) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: [R, C] bf16");
  const long R = x.size(0), C = x.size(1);
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.dim() == 2 && o.size(0) == R && o.size(1) % 64 == 0, "o: [R, HDp]");
  TORCH_CHECK(g.scalar_type() == torch::kBFloat16 && g.sizes() == o.sizes() && g.stride(1) == 1 && g.stride(0) % 8 == 0, "g: [R, HDp]");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.size(0) == C && wo.size(1) == o.size(1), "wo: [C, HDp]");
  TORCH_CHECK(R % WBM == 0, "rows % 128");
  auto out = torch::empty_like(x);
  switch (C) {
    case 64: launch_wtail<64>(g, o, x, wo, out); break;
    case 128: launch_wtail<128>(g, o, x, wo, out); break;
    case 192: launch_wtail<192>(g, o, x, wo, out); break;
    case 256: launch_wtail<256>(g, o, x, wo, out); break;
    case 320: launch_wtail<320>(g, o, x, wo, out); break;
    case 384: launch_wtail<384>(g, o, x, wo, out); break;
    case 448: launch_wtail<448>(g, o, x, wo, out); break;
    case 512: launch_wtail<512>(g, o, x, wo, out); break;
    default: TORCH_CHECK(false, "tri_wtail: C must be a multiple of 64 in 64 .. 512");
  }
  return out;
}

// ---- fused wide front: x [R, C], LN w / b [C], W [4 HDp + 16, C] bf16 -> (q, k, v, g [R, HDp] bf16, bias [B, NH, L, L] bf16)
template <int C>
static void launch_wfront(const torch::Tensor& x, const torch::Tensor& lw, const torch::Tensor& lb, double eps, const torch::Tensor& W,
                          long HDp, long NH, const bool* mp, long L, std::vector<torch::Tensor>& outs) {
  using T = WF<C>;
  const long R = x.size(0);
  long NBR = 0;                                           // rows per output block: the largest divisor of HDp <= MAXN, a multiple of 32
  for (long d = T::MAXN; d >= 32 && !NBR; d -= 32) if (HDp % d == 0) NBR = d;
  auto m2 = [&](const torch::Tensor& t, uint64_t cols, uint64_t rows, uint64_t ld, uint32_t b0, uint32_t b1, CUtensorMapSwizzle sw, const char* w) {
    return make_map<2>(t.data_ptr(), {cols, rows}, {ld}, {b0, b1}, sw, w); };
  CUtensorMap xm = m2(x, C, R, C, 64, WBM, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap wm = m2(W, C, W.size(0), C, 64, (uint32_t)NBR, CU_TENSOR_MAP_SWIZZLE_128B, "W");
  CUtensorMap wb = m2(W, C, W.size(0), C, 64, 16, CU_TENSOR_MAP_SWIZZLE_128B, "Wb");
  CUtensorMap om[4];
  for (int i = 0; i < 4; ++i) om[i] = m2(outs[i], HDp, R, HDp, 32, 32, CU_TENSOR_MAP_SWIZZLE_64B, "qkvg");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wfront_sm100<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM)); attr = true; }
  const int ntiles = (int)(R / WBM);
  tri_wfront_sm100<C><<<std::min(ntiles, num_sms(x.device().index())), 448, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      ntiles, (int)L, (int)HDp, (int)NH, (int)NBR, (float)eps, xm, wm, wb, om[0], om[1], om[2], om[3], lw.data_ptr<float>(), lb.data_ptr<float>(), mp,
      reinterpret_cast<__nv_bfloat16*>(outs[4].data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> tri_wfront_h(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, torch::Tensor W, int64_t NH,
                                        torch::Tensor mask, int64_t B, int64_t L) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: [R, C] bf16");
  const long R = x.size(0), C = x.size(1);
  TORCH_CHECK(R == B * L * L && L % WBM == 0, "rows = B*L*L, L % 128");
  TORCH_CHECK(W.scalar_type() == torch::kBFloat16 && W.is_contiguous() && W.size(1) == C && (W.size(0) - 16) % 256 == 0, "W: [4 HDp + 16, C]");
  const long HDp = (W.size(0) - 16) / 4;
  TORCH_CHECK(HDp % 64 == 0 && HDp <= 512 && NH >= 1 && NH <= 16, "HDp 64 .. 512 (a multiple of 64), heads <= 16");
  auto lw = w.to(torch::kFloat32).contiguous(), lb = b.to(torch::kFloat32).contiguous();
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  auto opt = x.options();
  std::vector<torch::Tensor> outs = {torch::empty({R, HDp}, opt), torch::empty({R, HDp}, opt), torch::empty({R, HDp}, opt),
                                     torch::empty({R, HDp}, opt), torch::empty({B, NH, L, L}, opt)};
  switch (C) {
    case 64: launch_wfront<64>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 128: launch_wfront<128>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 192: launch_wfront<192>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 256: launch_wfront<256>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 320: launch_wfront<320>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 384: launch_wfront<384>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 448: launch_wfront<448>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    case 512: launch_wfront<512>(x, lw, lb, eps, W, HDp, NH, mp, L, outs); break;
    default: TORCH_CHECK(false, "tri_wfront: C must be a multiple of 64 in 64 .. 512");
  }
  return outs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_wfront", &tri_wfront_h, "fused LN + [q | k | v | g | bias] projections for any width",
        py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("W"), py::arg("NH"), py::arg("mask"), py::arg("B"), py::arg("L"));
  m.def("tri_wtail", &tri_wtail_h, "fused x + (sigmoid(g) o o) . Wo^T for any width", py::arg("g"), py::arg("o"), py::arg("x"), py::arg("wo"));
  m.def("tri_ln_rows", &tri_ln_rows_h, "LayerNorm over C (64 .. 512) rows: bf16 -> bf16", py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("xcopy"));
  m.def("tri_bias_heads", &tri_bias_heads_h, "bias columns of the projection -> head-major masked bias [B, NH, L, L]",
        py::arg("P"), py::arg("cb"), py::arg("NH"), py::arg("mask"), py::arg("B"), py::arg("L"));
  m.def("tri_gate_mul", &tri_gate_mul_h, "u = sigmoid(g) o o", py::arg("g"), py::arg("o"));
}
