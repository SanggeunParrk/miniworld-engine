// ESMFold2 SWA atom block, sliding-window attention forward and backward (CUDA sm_90a, 4 heads x D = 32, half window HW = 64,
// bf16 operands, fp32 accumulate).  The math and the rounding points are the Triton kernels' (_swa_attn_fwd_kernel,
// _swa_attn_bwd_dq_kernel, _swa_attn_bwd_dkv_kernel):
//   forward   s_ij = (q_i . k_j) / sqrt(D) over |i - j| <= HW, 0 <= j < seqused[n];  p = exp(s - max);  l = sum p (fp32);
//             o_i = (rn(p) v) / l,  o_i = 0 for i >= seqused[n];  lse_i = max + log(l)  (0 where nothing is allowed)
//   backward  p = exp(s - lse_i) on the allowed pairs (padding queries and keys masked);  ds = rn(p * (dO_i . v_j - D_i));
//             dq = rn(scale * ds k);  dk = rn(scale * ds^T q);  dv = rn(rn(p)^T dO)
// The window makes every 64-query tile reach exactly the 192 keys [i0 - 64, i0 + 128), and one warp's 16 queries reach 144 of them, so
// the forward is a single pass (no online softmax): each warp holds its 16 x 144 scores in registers.
// mma.sync m16n8k16 with ldmatrix fragments from 64 B-row tiles (XOR-swizzled 16 B chunks, conflict-free); at head dim 32 the work is a
// few hundred MMAs per warp and the kernels are bound by latency and memory, which is what the structure optimises:
//   forward:  CTA = (64-query tile, n*H + h), 4 warps x 16 queries; Q 64 rows, K / V 192 rows via cp.async (zero-filled outside
//             [0, seqused)); O leaves through a smem staging tile as 16 B stores into O [N*S, C] (head h at columns 32h..).
//   backward: ONE kernel for dq and dk/dv (Triton runs two).  CTA = (64-row tile t, n*H + h), 8 warps: warps 0-3 own the tile's 64
//             queries (dq over their 144 keys), warps 4-7 own the tile's 64 keys (dk, dv over their 144 queries, computed transposed so
//             p^T / ds^T are A fragments in registers).  Q, K, V, dO, lse, D of rows [i0 - 64, i0 + 128) are loaded once for both.
//             dq / dk / dv go out either head-major [N, H, S, D] (what the Triton qkvg backward reads) or row-major into columns
//             0..C / C..2C / 2C..3C of the [N*S, 3C] projection-gradient buffer the CUDA qkvg backward works in (``rowmajor``).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include "pdl.cuh"

namespace {
constexpr int H = 4, D = 32, C = 128, HW = 64, BM = 64, KR = BM + 2 * HW;   // KR = 192 rows a tile's window reaches
constexpr int TILE = KR * 64;                                                 // bytes of a [192][32] bf16 tile

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
// byte offset of 16 B chunk c (0..3) of row r in a [rows][32] bf16 tile (64 B rows); XOR swizzle makes 8-row ldmatrix reads conflict-free
__device__ __forceinline__ int sw(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
constexpr float LOG2E = 1.4426950408889634f;

// A fragments (16 rows x 32 = 2 k-steps) of rows r0..r0+15 of a swizzled [rows][32] tile
__device__ __forceinline__ void load_a(const unsigned char* t, int r0, int lane, uint32_t (&a)[2][4]) {
  const int r = r0 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
  for (int s = 0; s < 2; ++s) ldsm4(sa(t + sw(r, 2 * s + (lane >> 4))), a[s][0], a[s][1], a[s][2], a[s][3]);
}
// acc[2][4] (+)= A (16 x 32) . T[n0..n0+15]^T : B = rows n0..n0+15 of the tile, k = the 32 columns (non-transposed ldmatrix)
__device__ __forceinline__ void mma_abt(float (&acc)[2][4], const uint32_t (&a)[2][4], const unsigned char* t, int n0, int lane) {
  const int r = n0 + (lane & 7) + (lane >> 4) * 8;
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    uint32_t b0, b1, b2, b3;
    ldsm4(sa(t + sw(r, 2 * s + ((lane >> 3) & 1))), b0, b1, b2, b3);
    mma(acc[0], a[s], b0, b1);
    mma(acc[1], a[s], b2, b3);
  }
}
// acc[4][4] (+)= A (16 x 16, k = rows k0..k0+15 of the tile) . T[k0..k0+15][0..32) : B row-major in n (transposed ldmatrix)
__device__ __forceinline__ void mma_ab(float (&acc)[4][4], const uint32_t (&a)[4], const unsigned char* t, int k0, int lane) {
  const int r = k0 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    uint32_t b0, b1, b2, b3;
    ldsm4t(sa(t + sw(r, 2 * p + (lane >> 4))), b0, b1, b2, b3);
    mma(acc[2 * p], a, b0, b1);
    mma(acc[2 * p + 1], a, b2, b3);
  }
}

// ---------------------------------------------------------------- forward ----------------------------------------------------------------
__global__ void __launch_bounds__(128) attn_fwd_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                                       const __nv_bfloat16* __restrict__ V, const int* __restrict__ seqused,
                                                       __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S, float scale) {
  __shared__ __align__(128) unsigned char sQ[BM * 64];
  __shared__ __align__(128) unsigned char sK[TILE];
  __shared__ __align__(128) unsigned char sV[TILE];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
  pdl_gate();
  const int t = blockIdx.x, nh = blockIdx.y, n = nh / H, h = nh - n * H;
  const int su = seqused[n];
  const int i0 = t * BM, kb = i0 - HW;
  const long hb = (long)nh * S * D;
  if (i0 >= su) {                                          // an all-padding tile: O = 0, lse = 0
    for (int e = tid; e < BM * 4; e += 128) {
      const int r = e >> 2, c = e & 3, i = i0 + r;
      if (i < S) *reinterpret_cast<uint4*>(O + ((long)n * S + i) * C + h * D + c * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int r = tid; r < BM; r += 128) if (i0 + r < S) LSE[(long)nh * S + i0 + r] = 0.f;
    return;
  }
  for (int e = tid; e < BM * 4; e += 128) {
    const int r = e >> 2, c = e & 3, i = i0 + r;
    const bool ok = i < su;
    cp16(sa(sQ + sw(r, c)), Q + hb + (long)(ok ? i : 0) * D + c * 8, ok);
  }
  for (int e = tid; e < KR * 4; e += 128) {
    const int r = e >> 2, c = e & 3, j = kb + r;
    const bool ok = j >= 0 && j < su;
    const long off = hb + (long)(ok ? j : 0) * D + c * 8;
    cp16(sa(sK + sw(r, c)), K + off, ok);
    cp16(sa(sV + sw(r, c)), V + off, ok);
  }
  cp_commit(); cp_wait_all();
  __syncthreads();
  // warp w: queries i0 + 16w + (g, g + 8); keys = tile rows 16w .. 16w + 143 (global kb + row)
  uint32_t qa[2][4];
  load_a(sQ, 16 * w, lane, qa);
  float sc[9][2][4];
#pragma unroll
  for (int m = 0; m < 9; ++m) {
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) sc[m][b][e] = 0.f;
    mma_abt(sc[m], qa, sK, 16 * w + 16 * m, lane);
  }
  const int qi0 = i0 + 16 * w + g;                          // rows qi0 (e 0, 1) and qi0 + 8 (e 2, 3)
  const float sl2 = scale * LOG2E;
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int m = 0; m < 9; ++m)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = kb + 16 * w + 16 * m + 8 * b + 2 * tq + (e & 1), i = qi0 + (e >> 1) * 8, d = i - j;
        const bool ok = d <= HW && d >= -HW && j >= 0 && j < su;
        const float v = ok ? sc[m][b][e] * sl2 : -INFINITY;
        sc[m][b][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
  float l[2] = {0.f, 0.f};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
  }
  const float ms[2] = {mx[0] == -INFINITY ? 0.f : mx[0], mx[1] == -INFINITY ? 0.f : mx[1]};
  float o[4][4];
#pragma unroll
  for (int b = 0; b < 4; ++b)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[b][e] = 0.f;
#pragma unroll
  for (int m = 0; m < 9; ++m) {
    float p[2][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) { p[b][e] = ex2(sc[m][b][e] - ms[e >> 1]); l[e >> 1] += p[b][e]; }
    const uint32_t pa[4] = {pack(p[0][0], p[0][1]), pack(p[0][2], p[0][3]), pack(p[1][0], p[1][1]), pack(p[1][2], p[1][3])};
    mma_ab(o, pa, sV, 16 * w + 16 * m, lane);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
  }
  // O (0 on padding queries) -> this warp's rows of the Q tile (only this warp read them) -> 16 B stores
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = qi0 + 8 * r, row = 16 * w + g + 8 * r;
    const float inv = (i < su && l[r] > 0.f) ? 1.f / l[r] : 0.f;
#pragma unroll
    for (int b = 0; b < 4; ++b)
      *reinterpret_cast<uint32_t*>(sQ + sw(row, b) + tq * 4) = pack(o[b][2 * r] * inv, o[b][2 * r + 1] * inv);
    if (tq == 0 && i < S) LSE[(long)nh * S + i] = (i < su && l[r] > 0.f) ? mx[r] / LOG2E + logf(l[r]) : 0.f;
  }
  __syncwarp();
#pragma unroll
  for (int e = lane; e < 64; e += 32) {
    const int row = 16 * w + (e >> 2), c = e & 3, i = i0 + row;
    if (i < S) *reinterpret_cast<uint4*>(O + ((long)n * S + i) * C + h * D + c * 8) = *reinterpret_cast<const uint4*>(sQ + sw(row, c));
  }
}

// ---------------------------------------------------------------- backward ---------------------------------------------------------------
struct BwdOut {
  __nv_bfloat16 *dq, *dk, *dv;
  int rowmajor;                                            // 0: head-major [N, H, S, D] each; 1: rows of [N*S, ld] at column h*D (+ offsets)
  int ld;
};
constexpr int BWD_SMEM = 4 * TILE + 2 * KR * 4 + 3 * BM * 64;   // Q, K, V, dO tiles | lse, D | dq, dk, dv staging

__global__ void __launch_bounds__(256) attn_bwd_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                                       const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                                                       const float* __restrict__ LSE, const float* __restrict__ Dv,
                                                       const int* __restrict__ seqused, BwdOut out, int S, float scale) {
  extern __shared__ __align__(128) unsigned char smem[];
  pdl_gate();
  unsigned char *sQ = smem, *sK = smem + TILE, *sV = smem + 2 * TILE, *sO = smem + 3 * TILE;
  float* sL = reinterpret_cast<float*>(smem + 4 * TILE);
  float* sD = sL + KR;
  unsigned char* sStg = smem + 4 * TILE + 2 * KR * 4;      // 3 x [64][32] bf16: dq | dk | dv
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
  const int t = blockIdx.x, nh = blockIdx.y, n = nh / H, h = nh - n * H;
  const int su = seqused[n];
  const int i0 = t * BM, kb = i0 - HW;
  const long hb = (long)nh * S * D;
  auto dst = [&](int which, int i) -> __nv_bfloat16* {     // row i of dq (0) / dk (1) / dv (2)
    __nv_bfloat16* base = which == 0 ? out.dq : which == 1 ? out.dk : out.dv;
    return out.rowmajor ? base + ((long)n * S + i) * out.ld + h * D : base + hb + (long)i * D;
  };
  if (i0 >= su) {                                          // no valid query or key in the tile: dq = dk = dv = 0
    for (int e = tid; e < 3 * BM * 4; e += 256) {
      const int which = e / (BM * 4), r = (e >> 2) % BM, c = e & 3, i = i0 + r;
      if (i < S) *reinterpret_cast<uint4*>(dst(which, i) + c * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
    return;
  }
  for (int e = tid; e < KR * 4; e += 256) {
    const int r = e >> 2, c = e & 3, j = kb + r;
    const bool ok = j >= 0 && j < su;
    const long off = hb + (long)(ok ? j : 0) * D + c * 8;
    cp16(sa(sQ + sw(r, c)), Q + off, ok);
    cp16(sa(sK + sw(r, c)), K + off, ok);
    cp16(sa(sV + sw(r, c)), V + off, ok);
    cp16(sa(sO + sw(r, c)), dO + ((long)n * S + (ok ? j : 0)) * C + h * D + c * 8, ok);
  }
  cp_commit();
  for (int r = tid; r < KR; r += 256) {
    const int j = kb + r;
    const bool ok = j >= 0 && j < su;
    sL[r] = ok ? LSE[(long)nh * S + j] : 0.f;
    sD[r] = ok ? Dv[(long)nh * S + j] : 0.f;
  }
  cp_wait_all();
  __syncthreads();
  const float sl2 = scale * LOG2E;
  if (w < 4) {
    // ---- dq: queries i0 + 16w + (g, g + 8) = tile rows 64 + 16w + ..; keys = tile rows 16w .. 16w + 143 ----
    const int qr = HW + 16 * w;
    uint32_t qa[2][4], da[2][4];
    load_a(sQ, qr, lane, qa);
    load_a(sO, qr, lane, da);
    const int qi0 = i0 + 16 * w + g;
    const float lse2[2] = {sL[qr + g] * LOG2E, sL[qr + g + 8] * LOG2E}, dd[2] = {sD[qr + g], sD[qr + g + 8]};
    const bool qok[2] = {qi0 < su, qi0 + 8 < su};
    float dq[4][4];
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[b][e] = 0.f;
#pragma unroll 1
    for (int m = 0; m < 9; ++m) {
      const int k0 = 16 * w + 16 * m;
      float s[2][4] = {}, dp[2][4] = {};
      mma_abt(s, qa, sK, k0, lane);
      mma_abt(dp, da, sV, k0, lane);
      float ds[2][4];
#pragma unroll
      for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = kb + k0 + 8 * b + 2 * tq + (e & 1), d = qi0 + (e >> 1) * 8 - j;
          const bool ok = d <= HW && d >= -HW && j >= 0 && j < su && qok[e >> 1];
          const float p = ok ? ex2(s[b][e] * sl2 - lse2[e >> 1]) : 0.f;
          ds[b][e] = p * (dp[b][e] - dd[e >> 1]);
        }
      const uint32_t a[4] = {pack(ds[0][0], ds[0][1]), pack(ds[0][2], ds[0][3]), pack(ds[1][0], ds[1][1]), pack(ds[1][2], ds[1][3])};
      mma_ab(dq, a, sK, k0, lane);
    }
    unsigned char* st = sStg;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        *reinterpret_cast<uint32_t*>(st + sw(16 * w + g + 8 * r, b) + tq * 4) = pack(dq[b][2 * r] * scale, dq[b][2 * r + 1] * scale);
    __syncwarp();
#pragma unroll
    for (int e = lane; e < 64; e += 32) {
      const int row = 16 * w + (e >> 2), c = e & 3, i = i0 + row;
      if (i < S) *reinterpret_cast<uint4*>(dst(0, i) + c * 8) = *reinterpret_cast<const uint4*>(st + sw(row, c));
    }
  } else {
    // ---- dk, dv (transposed): keys i0 + 16w' + (g, g + 8) = tile rows 64 + 16w' + ..; queries = tile rows 16w' .. 16w' + 143 ----
    const int ww = w - 4, kr = HW + 16 * ww;
    uint32_t ka[2][4], va[2][4];
    load_a(sK, kr, lane, ka);
    load_a(sV, kr, lane, va);
    const int kj0 = i0 + 16 * ww + g;
    const bool kok[2] = {kj0 < su, kj0 + 8 < su};
    float dk[4][4], dv[4][4];
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) { dk[b][e] = 0.f; dv[b][e] = 0.f; }
#pragma unroll 1
    for (int m = 0; m < 9; ++m) {
      const int q0 = 16 * ww + 16 * m;
      float s[2][4] = {}, dp[2][4] = {};
      mma_abt(s, ka, sQ, q0, lane);                        // s^T [keys][queries]
      mma_abt(dp, va, sO, q0, lane);                       // dp^T = v dO^T
      float p[2][4], ds[2][4];
#pragma unroll
      for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int qrow = q0 + 8 * b + 2 * tq + e, i = kb + qrow;
          const float l2 = sL[qrow] * LOG2E, dd = sD[qrow];
          const bool qv = i >= 0 && i < su;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int d = i - (kj0 + 8 * r);
            const bool ok = qv && kok[r] && d <= HW && d >= -HW;
            const float pv = ok ? ex2(s[b][2 * r + e] * sl2 - l2) : 0.f;
            p[b][2 * r + e] = pv;
            ds[b][2 * r + e] = pv * (dp[b][2 * r + e] - dd);
          }
        }
      const uint32_t pa[4] = {pack(p[0][0], p[0][1]), pack(p[0][2], p[0][3]), pack(p[1][0], p[1][1]), pack(p[1][2], p[1][3])};
      const uint32_t sa4[4] = {pack(ds[0][0], ds[0][1]), pack(ds[0][2], ds[0][3]), pack(ds[1][0], ds[1][1]), pack(ds[1][2], ds[1][3])};
      mma_ab(dv, pa, sO, q0, lane);
      mma_ab(dk, sa4, sQ, q0, lane);
    }
    unsigned char* stk = sStg + BM * 64;
    unsigned char* stv = sStg + 2 * BM * 64;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int o = sw(16 * ww + g + 8 * r, b) + tq * 4;
        *reinterpret_cast<uint32_t*>(stk + o) = pack(dk[b][2 * r] * scale, dk[b][2 * r + 1] * scale);
        *reinterpret_cast<uint32_t*>(stv + o) = pack(dv[b][2 * r], dv[b][2 * r + 1]);
      }
    __syncwarp();
#pragma unroll
    for (int e = lane; e < 128; e += 32) {
      const int which = 1 + (e >> 6), row = 16 * ww + ((e >> 2) & 15), c = e & 3, i = i0 + row;
      const unsigned char* st = which == 1 ? stk : stv;
      if (i < S) *reinterpret_cast<uint4*>(dst(which, i) + c * 8) = *reinterpret_cast<const uint4*>(st + sw(row, c));
    }
  }
}

void check_heads(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.dim() == 4 && t.size(1) == H && t.size(3) == D,
              name, ": head-major [N, 4, S, 32] bf16");
}
}  // namespace

// Qh / Kh / Vh head-major [N, 4, S, 32] bf16; seqused [N] int32.  Returns [O [N*S, 128] bf16, lse [N, 4, S] fp32].
std::vector<torch::Tensor> attn_fwd(torch::Tensor Qh, torch::Tensor Kh, torch::Tensor Vh, torch::Tensor seqused, double scale) {
  check_heads(Qh, "Qh"); check_heads(Kh, "Kh"); check_heads(Vh, "Vh");
  const long N = Qh.size(0), S = Qh.size(2);
  TORCH_CHECK(Kh.sizes() == Qh.sizes() && Vh.sizes() == Qh.sizes(), "Q / K / V shapes differ");
  TORCH_CHECK(seqused.is_cuda() && seqused.scalar_type() == torch::kInt && seqused.numel() == N, "seqused [N] int32");
  auto O = torch::empty({N * S, C}, Qh.options());
  auto lse = torch::empty({N, H, S}, Qh.options().dtype(torch::kFloat));
  if (N * S == 0) return {O, lse};
  dim3 grid((unsigned)((S + BM - 1) / BM), (unsigned)(N * H));
  pdl_launch(attn_fwd_kernel, grid, 128, 0, at::cuda::getCurrentCUDAStream(), true,
      reinterpret_cast<const __nv_bfloat16*>(Qh.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(Kh.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(Vh.data_ptr()), seqused.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
      lse.data_ptr<float>(), (int)S, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {O, lse};
}

// dO [N*S, 128] bf16 (row-major, head h at columns 32h..); lse, Dv [N, 4, S] fp32.
// dP undefined -> returns head-major [dQh, dKh, dVh]; dP [N*S, >= 384] bf16 given -> writes dq | dk | dv into its columns 0 | 128 | 256
// (row-major) and returns [dP].
std::vector<torch::Tensor> attn_bwd(torch::Tensor Qh, torch::Tensor Kh, torch::Tensor Vh, torch::Tensor dO, torch::Tensor lse,
                                    torch::Tensor Dv, torch::Tensor seqused, c10::optional<torch::Tensor> dP, double scale) {
  check_heads(Qh, "Qh"); check_heads(Kh, "Kh"); check_heads(Vh, "Vh");
  const long N = Qh.size(0), S = Qh.size(2);
  TORCH_CHECK(dO.is_cuda() && dO.scalar_type() == torch::kBFloat16 && dO.is_contiguous() && dO.numel() == N * S * C, "dO [N*S, 128] bf16");
  for (const auto* t : {&lse, &Dv}) TORCH_CHECK(t->scalar_type() == torch::kFloat && t->is_contiguous() && t->numel() == N * H * S, "lse / D [N, 4, S] fp32");
  TORCH_CHECK(seqused.is_cuda() && seqused.scalar_type() == torch::kInt && seqused.numel() == N, "seqused [N] int32");
  BwdOut out;
  std::vector<torch::Tensor> ret;
  if (dP.has_value() && dP->defined()) {
    TORCH_CHECK(dP->scalar_type() == torch::kBFloat16 && dP->dim() == 2 && dP->size(0) == N * S && dP->size(1) >= 3 * C && dP->stride(1) == 1 &&
                dP->stride(0) % 8 == 0, "dP [N*S, >= 384] bf16, unit column stride");
    auto* base = reinterpret_cast<__nv_bfloat16*>(dP->data_ptr());
    out.dq = base; out.dk = base + C; out.dv = base + 2 * C; out.rowmajor = 1; out.ld = (int)dP->stride(0);
    ret = {*dP};
  } else {
    auto dQh = torch::empty_like(Qh), dKh = torch::empty_like(Qh), dVh = torch::empty_like(Qh);
    out.dq = reinterpret_cast<__nv_bfloat16*>(dQh.data_ptr()); out.dk = reinterpret_cast<__nv_bfloat16*>(dKh.data_ptr());
    out.dv = reinterpret_cast<__nv_bfloat16*>(dVh.data_ptr()); out.rowmajor = 0; out.ld = 0;
    ret = {dQh, dKh, dVh};
  }
  if (N * S == 0) return ret;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BWD_SMEM)); attr = true; }
  dim3 grid((unsigned)((S + BM - 1) / BM), (unsigned)(N * H));
  pdl_launch(attn_bwd_kernel, grid, 256, BWD_SMEM, at::cuda::getCurrentCUDAStream(), true,
      reinterpret_cast<const __nv_bfloat16*>(Qh.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(Kh.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(Vh.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr()), lse.data_ptr<float>(),
      Dv.data_ptr<float>(), seqused.data_ptr<int>(), out, (int)S, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_fwd", &attn_fwd, "ESMFold2 SWA atom block sliding-window attention forward (bf16 mma.sync)");
  m.def("attn_bwd", &attn_bwd, "ESMFold2 SWA atom block sliding-window attention backward, dq + dk + dv in one kernel (bf16 mma.sync)");
}
