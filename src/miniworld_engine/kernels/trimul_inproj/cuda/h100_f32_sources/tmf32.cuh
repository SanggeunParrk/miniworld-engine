// tmf32.cuh -- bidirectional TriMul on H100 in fp32 activations / TF32 tensor cores (D = 128, hidden = 128, 2 x 128 channels per side).
// SPDX-License-Identifier: Apache-2.0
//
// The module's maths (see modules/triangle_multiplication/bidirectional.py), with every matmul operand rounded to nearest TF32 (what cuBLAS
// does under allow_tf32) and fp32 accumulation:
//
//   x_n    = LN_in(x)                                            ln_in_kernel          x -> x_n (T x 128), mean, rstd
//   g, p   = x_n W_{lg,l,rg,r}^T ;  plane = sigmoid(g) p pm      front_kernel          wgmma TF32 -> planes [512][T] (left 0..255, right 256..511)
//   tri    = outgoing / incoming contractions of the planes      cuBLAS (the caller)
//   out_n  = LN_out(tri)                                         ln_out_t_kernel       c-major planes -> token-major rows (T x 256)
//   y      = x + ds * sigmoid(x_n Wg^T) * (out_n Wo^T)           back_kernel           wgmma TF32 (two accumulators), residual in the epilogue
//
// T = L * L tokens of one sample (b-major batches are run sample by sample by the caller).  Every GEMM runs on K-major shared-memory
// operands (TF32 wgmma requires them): activations [rows][K] row-major, weights in the nn.Linear layout [out][in].
#pragma once
#include "tmn_ptx.cuh"
#include "wgmma_tf32.cuh"

namespace tmf {
using namespace tmn;

constexpr int BM = 128;                       // token rows per tile (two consumer warpgroups x 64)
constexpr int BK = 32;                        // floats per K step = one 128-byte swizzle row

__device__ __forceinline__ float rna(float x) { uint32_t u; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x)); return __uint_as_float(u); }
__device__ __forceinline__ float sigm(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }
__device__ __forceinline__ uint64_t dsc(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }

// ------------------------------------------------------------------------------------------------------------------------------
// weights: round to TF32 and pack.  jobs: 0 = [Wlg; Wl; Wrg; Wr] (4 x [256][128]) -> wpk [1024][128]; 1 = Wo [128][256]; 2 = Wg [128][128].
struct PackArgs {
  const float* src[6];       // Wlg, Wl, Wrg, Wr, Wo, Wg
  float* wpk;                // [1024][128]
  float* wo;                 // [128][256]
  float* wg;                 // [128][128]
};
__global__ void pack_kernel(PackArgs a) {
  const int job = blockIdx.y;
  const float* s; float* d; int n;
  if (job < 4) { s = a.src[job]; d = a.wpk + (size_t)job * 256 * 128; n = 256 * 128; }
  else if (job == 4) { s = a.src[4]; d = a.wo; n = 128 * 256; }
  else { s = a.src[5]; d = a.wg; n = 128 * 128; }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = rna(s[i]);
}

// ------------------------------------------------------------------------------------------------------------------------------
// LN_in: one warp per row, 4 floats per lane.  xn is stored rounded to TF32 (it is the A operand of two TF32 GEMMs).
__global__ void __launch_bounds__(256) ln_in_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                    float* __restrict__ xn, float* __restrict__ mean_o, float* __restrict__ rstd_o,
                                                    int64_t T, float eps) {
  const int64_t row = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= T) return;
  const float4 v = reinterpret_cast<const float4*>(x + row * 128)[lane];
  float s = v.x + v.y + v.z + v.w;
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  const float mean = s * (1.f / 128.f);
  const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
  float q = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
  const float rstd = rsqrtf(q * (1.f / 128.f) + eps);
  const float4 ww = reinterpret_cast<const float4*>(w)[lane], bb = reinterpret_cast<const float4*>(b)[lane];
  float4 o4;
  o4.x = rna(d0 * rstd * ww.x + bb.x);
  o4.y = rna(d1 * rstd * ww.y + bb.y);
  o4.z = rna(d2 * rstd * ww.z + bb.z);
  o4.w = rna(d3 * rstd * ww.w + bb.w);
  reinterpret_cast<float4*>(xn + row * 128)[lane] = o4;
  if (lane == 0 && mean_o) { mean_o[row] = mean; rstd_o[row] = rstd; }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Front: planes[side][ch][t] = sigmoid(xn Wg_side^T) * (xn Wp_side^T) * pm(t),  side 0 = left, 1 = right, 256 channels per side.
// Persistent CTAs over (128-token m-tile, nt) work units, nt = 4 per m-tile: side = nt >> 1, half = nt & 1 (128 channels).
// One work unit = A (128 tokens x K 128) x [128 gate rows ; 128 proj rows] (BN = 256).  Producer warpgroup: TMA into a 4-stage ring;
// two consumer warpgroups of 64 token rows each.  Epilogue: gate, mask, direct stores into the channel-major planes (a quad of lanes writes
// one 32-byte sector of 8 consecutive tokens of one channel).  SAVE also stores the pre-activations g, p (training): pre[side*2 + {g, p}][ch][t].
constexpr int F_BN = 256, F_NT = 4, F_KB = 4, F_NSTAGE = 4;
constexpr int F_A = BM * 128, F_B = F_BN * 128, F_STAGE = F_A + F_B;
constexpr int F_SMEM = F_NSTAGE * F_STAGE + (2 * F_NSTAGE) * 8 + 1024;

template <bool SAVE>
__global__ void __launch_bounds__(384, 1)
front_kernel(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB, int64_t T, int L,
             const uint8_t* __restrict__ tmask, float* __restrict__ planes, float* __restrict__ pre) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_NSTAGE * F_STAGE);
  uint64_t* empty = full + F_NSTAGE;
  const int64_t mtiles = (T + BM - 1) / BM, tiles = mtiles * F_NT;
  if (tid == 0) {
    for (int s = 0; s < F_NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0;
    for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int m0 = (int)(t / F_NT) * BM, nt = (int)(t % F_NT);
      const int grow = (nt >> 1) * 512 + (nt & 1) * 128;
      for (int kb = 0; kb < F_KB; ++kb, ++it) {
        const int s = (int)(it % F_NSTAGE);
        if (it >= F_NSTAGE) mbar_wait(empty + s, ((it / F_NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * F_STAGE;
        mbar_arrive_expect_tx(full + s, F_STAGE);
        tma_load_2d(st, &mA, full + s, kb * BK, m0);
        tma_load_2d(st + F_A, &mB, full + s, kb * BK, grow);                       // gate rows
        tma_load_2d(st + F_A + 128 * 128, &mB, full + s, kb * BK, grow + 256);     // proj rows
      }
    }
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  uint32_t it = 0;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int m0 = (int)(t / F_NT) * BM, nt = (int)(t % F_NT);
    const int side = nt >> 1, half = nt & 1;
    float acc[F_BN / 2];
#pragma unroll
    for (int e = 0; e < F_BN / 2; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < F_KB; ++kb, ++it) {
      const int s = (int)(it % F_NSTAGE);
      mbar_wait(full + s, (it / F_NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_STAGE + cw * 8192, b0 = su + s * F_STAGE + F_A;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_tf32<F_BN>(acc, dsc(a0 + ks * 32), dsc(b0 + ks * 32));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
    // ---------------------------------------------------------------------------------------------------- epilogue
    const int rl = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3);
    float* pl = planes + (size_t)(side * 256 + half * 128) * T;
    float* pg = nullptr; float* pp = nullptr;
    if (SAVE) { pg = pre + (size_t)((side * 2) * 256 + half * 128) * T; pp = pre + (size_t)((side * 2 + 1) * 256 + half * 128) * T; }
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int64_t r = (int64_t)m0 + 64 * cw + rl + 8 * hf;
      const bool valid = r < T;
      float pm = 1.f;
      if (tmask != nullptr && valid) {
        const int i = (int)(r / L), j = (int)(r - (int64_t)i * L);
        pm = (tmask[i] != 0 && tmask[j] != 0) ? 1.f : 0.f;
      }
      if (valid) {
#pragma unroll
        for (int g = 0; g < F_BN / 16; ++g) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const float a = acc[4 * g + 2 * hf + i], b = acc[4 * (g + F_BN / 16) + 2 * hf + i];
            const int c = 8 * g + c0 + i;
            pl[(size_t)c * T + r] = sigm(a) * b * pm;
            if (SAVE) { pg[(size_t)c * T + r] = a; pp[(size_t)c * T + r] = b; }
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
// LN_out + transpose: tri [256][T] (channel-major) -> on [T][256] token-major, normalised over the 256 channels, TF32-rounded.  64 tokens per
// block, the tile staged in shared memory ([256][65]: conflict-free both ways).
__global__ void __launch_bounds__(256) ln_out_t_kernel(const float* __restrict__ tri, const float* __restrict__ w, const float* __restrict__ b,
                                                       float* __restrict__ on, float* __restrict__ mean_o, float* __restrict__ rstd_o,
                                                       int64_t T, float eps) {
  extern __shared__ float s[];                       // [256][65]
  __shared__ float red[4][64], ms[64], rs[64];
  const int64_t t0 = (int64_t)blockIdx.x * 64;
  const int tx = threadIdx.x & 63, ty = threadIdx.x >> 6;
  const int64_t tt = t0 + tx;
  for (int c = ty; c < 256; c += 4) s[c * 65 + tx] = (tt < T) ? tri[(size_t)c * T + tt] : 0.f;
  __syncthreads();
  float p = 0.f;
  for (int c = ty; c < 256; c += 4) p += s[c * 65 + tx];
  red[ty][tx] = p;
  __syncthreads();
  const float mean = (red[0][tx] + red[1][tx] + red[2][tx] + red[3][tx]) * (1.f / 256.f);
  __syncthreads();
  p = 0.f;
  for (int c = ty; c < 256; c += 4) { const float d = s[c * 65 + tx] - mean; p += d * d; }
  red[ty][tx] = p;
  __syncthreads();
  const float var = (red[0][tx] + red[1][tx] + red[2][tx] + red[3][tx]) * (1.f / 256.f);
  const float rstd = rsqrtf(var + eps);
  if (ty == 0) { ms[tx] = mean; rs[tx] = rstd; if (mean_o && tt < T) { mean_o[tt] = mean; rstd_o[tt] = rstd; } }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int tk = warp; tk < 64; tk += 8) {
    const int64_t t = t0 + tk;
    if (t >= T) break;
    const float mu = ms[tk], rr = rs[tk];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int c = lane + 32 * q;
      on[(size_t)t * 256 + c] = rna((s[c * 65 + tk] - mu) * rr * w[c] + b[c]);
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Back: y = x + ds * sigmoid(xn Wg^T) * (on Wo^T).  A1 = on [T][256], B1 = Wo [128][256]; A2 = xn [T][128], B2 = Wg [128][128]; K steps of 32:
// 8 for the projection accumulator, 4 for the gate accumulator.  ds: [L][128] row-drop scale (row j = t mod L) or null.  SAVE stores the gate
// logit and the projection ([T][128] each) for the backward.
constexpr int B_KB1 = 8, B_KB2 = 4, B_KB = B_KB1 + B_KB2, B_NSTAGE = 6;
constexpr int B_STAGE = BM * 128 + 128 * 128;
constexpr int B_SMEM = B_NSTAGE * B_STAGE + (2 * B_NSTAGE) * 8 + 1024;

template <bool SAVE>
__global__ void __launch_bounds__(384, 1)
back_kernel(const __grid_constant__ CUtensorMap mA1, const __grid_constant__ CUtensorMap mB1, const __grid_constant__ CUtensorMap mA2,
            const __grid_constant__ CUtensorMap mB2, int64_t T, int L, const float* __restrict__ x, const float* __restrict__ ds,
            float* __restrict__ y, float* __restrict__ gsave, float* __restrict__ psave) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + B_NSTAGE * B_STAGE);
  uint64_t* empty = full + B_NSTAGE;
  const int64_t mtiles = (T + BM - 1) / BM;
  if (tid == 0) {
    for (int s = 0; s < B_NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0;
    for (int64_t t = blockIdx.x; t < mtiles; t += gridDim.x) {
      const int m0 = (int)t * BM;
      for (int kb = 0; kb < B_KB; ++kb, ++it) {
        const int s = (int)(it % B_NSTAGE);
        if (it >= B_NSTAGE) mbar_wait(empty + s, ((it / B_NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * B_STAGE;
        mbar_arrive_expect_tx(full + s, B_STAGE);
        if (kb < B_KB1) { tma_load_2d(st, &mA1, full + s, kb * BK, m0); tma_load_2d(st + BM * 128, &mB1, full + s, kb * BK, 0); }
        else { tma_load_2d(st, &mA2, full + s, (kb - B_KB1) * BK, m0); tma_load_2d(st + BM * 128, &mB2, full + s, (kb - B_KB1) * BK, 0); }
      }
    }
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  uint32_t it = 0;
  for (int64_t t = blockIdx.x; t < mtiles; t += gridDim.x) {
    const int m0 = (int)t * BM;
    float accP[64], accG[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) { accP[e] = 0.f; accG[e] = 0.f; }
    int prev = -1;
    for (int kb = 0; kb < B_KB; ++kb, ++it) {
      const int s = (int)(it % B_NSTAGE);
      mbar_wait(full + s, (it / B_NSTAGE) & 1u);
      const uint32_t a0 = su + s * B_STAGE + cw * 8192, b0 = su + s * B_STAGE + BM * 128;
      fence_regs(accP); fence_regs(accG); wgmma_fence();
      if (kb < B_KB1) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma_tf32<128>(accP, dsc(a0 + ks * 32), dsc(b0 + ks * 32));
      } else {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma_tf32<128>(accG, dsc(a0 + ks * 32), dsc(b0 + ks * 32));
      }
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(accP); fence_regs(accG); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(accP); fence_regs(accG);
    if (wtid == 0) mbar_arrive(empty + prev);
    const int rl = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int64_t r = (int64_t)m0 + 64 * cw + rl + 8 * hf;
      if (r >= T) continue;
      const int j = (int)(r % L);
#pragma unroll
      for (int g = 0; g < 16; ++g) {
        const int c = 8 * g + c0;
        const float2 xr = *reinterpret_cast<const float2*>(x + (size_t)r * 128 + c);
        float2 sc = make_float2(1.f, 1.f);
        if (ds != nullptr) sc = *reinterpret_cast<const float2*>(ds + (size_t)j * 128 + c);
        const float p0 = accP[4 * g + 2 * hf], p1 = accP[4 * g + 2 * hf + 1];
        const float g0 = accG[4 * g + 2 * hf], g1 = accG[4 * g + 2 * hf + 1];
        float2 o;
        o.x = xr.x + sc.x * (sigm(g0) * p0);
        o.y = xr.y + sc.y * (sigm(g1) * p1);
        *reinterpret_cast<float2*>(y + (size_t)r * 128 + c) = o;
        if (SAVE) {
          *reinterpret_cast<float2*>(gsave + (size_t)r * 128 + c) = make_float2(g0, g1);
          *reinterpret_cast<float2*>(psave + (size_t)r * 128 + c) = make_float2(p0, p1);
        }
      }
    }
  }
}

}  // namespace tmf
