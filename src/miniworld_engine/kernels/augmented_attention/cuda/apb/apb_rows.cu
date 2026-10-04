// AttentionPairBias on B200 (CUDA): the pair -> bias pass, its backward, and the row passes of the single track
// (integrations/attention_pair_bias_b200.py). The attention core is sm_100a (kernels/augmented_attention/cuda/sm100), the
// projections cuBLAS.
//
// Heads x head dim: 8 x 48, 12 x 32, 24 x 16, 16 x 24 (d_single 384) and 16 x 32 (d_single 512). The pair kernels run the heads in mma groups of 8 (12 heads as 16:
// the 4 pad heads have zero Wf / dbias and are neither stored nor accumulated). For 16 x 24 every per-head row segment is 32 wide in memory (24 channels, 8 zeros):
// q | k | v | g, O, og, dO and their gradients are W = heads x 32 = 512 wide; `prep` writes the padded weight packs, `finalize`
// gathers the real rows / columns back. The others have no padding (W = d_single).
//
//   pair_bias_fwd   bias[h, i, j] = Wf[h] . LN(pair[i, j])   (d_pair 128; LN without affine: its weight is folded into Wf, its
//                   bias adds a per-head constant that the softmax cancels), -big on masked keys, bf16 head-major. One read of
//                   the pair, nothing else of its size.
//   pair_bias_bwd   dpair = LN_bwd(Wf^T dbias) and dWf = dbias^T LN(pair) (+ the per-head dbias sums): one more read of the
//                   pair, one write of dpair.
//   ln_rows         xa = LN(x) w + b over d_single columns (bf16), (mean, rstd) for the backward, and a copy of x as the output's
//                   residual seed (to_out adds onto it in place: no separate residual pass)
//   gate_rows       og = sigmoid(g) o        gate_bwd  dO, D = rowsum_head(dO o), dg       (W columns)
//   qkv_bwd         dq | dk | dv (fp32, from the core) -> the bf16 dqkvg columns, dbq            (W columns)
//   ln_bwd          dx = dy + LN_bwd(dxa w), dw, db
//   prep            the step's packed weights: W q | k | v | g (padded rows), its bias, Wo (padded columns), Wf
//   finalize        the parameter gradients out of the shared buffers, each into its own tensor (one launch)
// Row passes: a warp per row, lane l on CPL = W / 32 contiguous columns (12 or 16; a head is 4 or 2 lanes: its sums are lane
// shuffles). Column sums (dw, db, dbq, dWf) go block-reduced into one zeroed fp32 accumulator with atomics.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>

#include "../../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

constexpr int DP = 128, RW = 4, NT = RW * 32;           // d_pair; row kernels: RW warps, a row each
// accumulator (fp32, zeroed per step): dlnw [d] | dlnb [d] | dbq [W] | dWf [NH, DP] | per-head dbias sums [NH]
__host__ __device__ constexpr int acc_n(int d, int nh, int w) { return 2 * d + w + nh * DP + nh; }

__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }
__device__ __forceinline__ float4 sig4(float4 a) { return make_float4(sigm(a.x), sigm(a.y), sigm(a.z), sigm(a.w)); }
__device__ __forceinline__ float4 bfround(float4 a) {
  return make_float4(__bfloat162float(__float2bfloat16_rn(a.x)), __bfloat162float(__float2bfloat16_rn(a.y)),
                     __bfloat162float(__float2bfloat16_rn(a.z)), __bfloat162float(__float2bfloat16_rn(a.w)));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// Row statistics of a 16-row group on the tensor cores. With the A fragment of rows g, g + 8 (a0..a3 of m16n8k16), the same
// registers are the B fragment of X^T for rows g (a0, a2) or g + 8 (a1, a3): mma(A, that B) accumulates the Gram block
// X X^T[0..15][0..7] (or [0..15][8..15]) whose diagonal is sum_c x^2; a B of ones gives the row sums. Lane (g, q) then reads
// its rows' diagonal entries from lane (g, g / 2).
__device__ __forceinline__ void gram_step(float (&g0)[4], float (&g1)[4], float (&cs)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
  constexpr uint32_t ONES = 0x3F803F80u;                                    // bf16 (1, 1)
  mma16816(g0, a0, a1, a2, a3, a0, a2);
  mma16816(g1, a0, a1, a2, a3, a1, a3);
  mma16816(cs, a0, a1, a2, a3, ONES, ONES);
}
// (mean, rstd) of rows g and g + 8 from gram_step's accumulators
__device__ __forceinline__ void gram_stats(const float (&g0)[4], const float (&g1)[4], const float (&cs)[4], float eps, float& mu0,
                                           float& rs0, float& mu1, float& rs1) {
  const int lane = threadIdx.x & 31, g = lane >> 2, src = 4 * g + (g >> 1);
  const float s0 = __shfl_sync(0xffffffffu, (g & 1) ? g0[1] : g0[0], src), s1 = __shfl_sync(0xffffffffu, (g & 1) ? g1[3] : g1[2], src);
  mu0 = cs[0] * (1.f / DP); mu1 = cs[2] * (1.f / DP);
  rs0 = rsqrtf(fmaxf(s0 * (1.f / DP) - mu0 * mu0, 0.f) + eps);
  rs1 = rsqrtf(fmaxf(s1 * (1.f / DP) - mu1 * mu1, 0.f) + eps);
}

// ------------------------------------------------------------------------------------------------ bulk copies (sm_90+)
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(b)), "r"(n)); }
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra.uni W%=;\n }"
               ::"r"(su32(b)), "r"(parity) : "memory");
}

// ------------------------------------------------------------------------------------------------ pair -> bias
// Persistent, two blocks per SM: a block walks tiles of 128 consecutive keys j0.. of one query row i (32 KB of pair,
// contiguous; one bulk copy each, NSF stages deep). A warp takes one group of 16 rows of the tile (8 warps; 4 warps x 2 groups
// and 3 stages measured the same or slower: the step is latency-bound per block, so two blocks per SM overlap each other). In a
// group, lane (g = lane / 4, q = lane % 4) holds rows g and g + 8, columns 32 t + 8 q + e (t = 0..3): the dot product runs over K
// in any order, so the logical k of the A fragments is mapped onto those columns and Wf's B fragments use the same map (k step
// s = 2 t + u, slot 2q + v <-> column 32 t + 8 q + 4 u + v, slot 2q + 8 + v <-> + 2). N = 8 heads per mma (NH / 8 of them). LN
// folded into the projection: Wf . LN(x) = rstd (Wf . x - mean sum_c Wf): the mma runs on the raw bf16 words; the row statistics
// come from the tensor cores too (gram_step): no per-element float work.
#ifndef FWD_STAGES
#define FWD_STAGES 2
#endif
constexpr int NSF = FWD_STAGES, FT = 128 * DP * 2, FW = 8, FNT = 32 * FW;   // stages, tile bytes, warps, threads
template <int NH> constexpr int nhp() { return (NH + 7) / 8 * 8; }    // heads rounded up to the mma's 8
template <int NH> constexpr int fwd_smem() { return NSF * FT + 2 * nhp<NH>() * (128 + 4) * 4 + NSF * 8; }
// Wf's B fragments (lane (g, q): head g + 8 nt; zero past NH) and sum_c Wf of heads 2q, 2q + 1 (+ 8 nt) for this lane's C columns
template <int NT8, int NH>
__device__ __forceinline__ void wf_frags(const bf* W, uint32_t (&b)[NT8][8][2], float (&sw0)[NT8], float (&sw1)[NT8]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    float sw = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const bool live = g + 8 * nt < NH;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + (live ? g + 8 * nt : 0) * DP + 32 * (s >> 1) + 8 * q + 4 * (s & 1));
      b[nt][s][0] = live ? wp[0] : 0u; b[nt][s][1] = live ? wp[1] : 0u;
      const float2 u = unpack2(b[nt][s][0]), v = unpack2(b[nt][s][1]);
      sw += u.x + u.y + v.x + v.y;
    }
    sw = quad_sum(sw);                                                      // sum_c Wf[g + 8 nt, c]
    sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
  }
}

template <int NH>
__global__ void __launch_bounds__(FNT) pair_bias_fwd_k(const bf* __restrict__ Z, const bf* __restrict__ W, const bool* __restrict__ MASK,
    bf* __restrict__ OUT, int L, float eps, float neg) {
  constexpr int NT8 = nhp<NH>() / 8, NHP = nhp<NH>();
  extern __shared__ __align__(128) uint8_t smem[];
  float (*out_s)[NHP][128 + 4] = reinterpret_cast<float (*)[NHP][128 + 4]>(smem + NSF * FT);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NSF * FT + 2 * NHP * (128 + 4) * 4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + 127) / 128;
  const long ntiles = (long)L * ntj;
  auto issue = [&](long n) {                                                // thread 0: this block's n-th tile into its stage
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0);
    uint64_t* fb = &full[n % NSF];
    mbar_expect_tx(fb, (uint32_t)rows * DP * 2);
    bulk_g2s(smem + (n % NSF) * FT, Z + ((long)i * L + j0) * DP, (uint32_t)rows * DP * 2, fb);
  };
  if (threadIdx.x == 0) {
    for (int k = 0; k < NSF; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int k = 0; k < NSF; ++k) issue(k);
  uint32_t b[NT8][8][2];
  float sw0[NT8], sw1[NT8];
  wf_frags<NT8, NH>(W, b, sw0, sw1);
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0), st = (int)(n % NSF);
    mbar_wait(&full[st], (uint32_t)((n / NSF) & 1));
    const uint8_t* tile = smem + st * FT;
    float (*os)[128 + 4] = out_s[n & 1];
    const int rb = warp * 16;
    if (rb < rows) {
      uint4 r0[4], r1[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        r0[u] = *reinterpret_cast<const uint4*>(tile + (rb + g) * DP * 2 + 64 * u + 16 * q);
        r1[u] = *reinterpret_cast<const uint4*>(tile + (rb + g + 8) * DP * 2 + 64 * u + 16 * q);
      }
      float c[NT8][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint32_t u0[4] = {r0[u].x, r0[u].y, r0[u].z, r0[u].w}, u1[4] = {r1[u].x, r1[u].y, r1[u].z, r1[u].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {                                       // k step s = 2 u + h: words 2h, 2h + 1
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt)
            mma16816(c[nt], u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], b[nt][2 * u + h][0], b[nt][2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float m0, rs0, m1, rs1;
      gram_stats(g0, g1, cs, eps, m0, rs0, m1, rs1);
      const int jl = rb + g;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const int h0 = 2 * q + 8 * nt;
        os[h0][jl] = rs0 * (c[nt][0] - m0 * sw0[nt]); os[h0 + 1][jl] = rs0 * (c[nt][1] - m0 * sw1[nt]);
        os[h0][jl + 8] = rs1 * (c[nt][2] - m1 * sw0[nt]); os[h0 + 1][jl + 8] = rs1 * (c[nt][3] - m1 * sw1[nt]);
      }
    }
    __syncthreads();                                                        // the stage is read; out_s[n & 1] written
    if (threadIdx.x == 0) issue(n + NSF);
    for (int task = threadIdx.x; task < NH * 16; task += FNT) {             // (head, 8 keys) per task: 16-B stores
      const int h = task >> 4, ch = task & 15, j = j0 + ch * 8;
      if (ch * 8 >= rows) continue;
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float a = os[h][ch * 8 + 2 * e], bb = os[h][ch * 8 + 2 * e + 1];
        if (MASK) { if (!MASK[j + 2 * e]) a = neg; if (!MASK[j + 2 * e + 1]) bb = neg; }
        o[e] = pack2(a, bb);
      }
      *reinterpret_cast<uint4*>(OUT + ((long)h * L + i) * L + j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}

// Backward, persistent (two blocks per SM, 8 warps), over the forward's tiles (i, j0..j0+127). Per tile, TMA brings the pair
// rows as two 128-B-swizzled boxes of 64 columns (the swizzle keeps both the row reads and the ldmatrix.trans below
// conflict-free) and dbias[0..NH-1][i][j0..] (fp32), NSB stages deep. A warp takes 16 rows; lane (g, q) holds rows g, g + 8 at
// columns 32 u + 8 q + e as in the forward. With LN folded (x^ = (x - mean) rstd, s_h = sum_c Wf[h, c], a = rstd dbias):
//   dWf[h, c]  = sum_r a[r, h] x[r, c] - sum_r a[r, h] mean_r        mma: M = 16 columns (x^T by ldmatrix.trans), N = 8 heads,
//                                                                     K = the 16 rows; the second term is one scalar per head
//   dx^[c]     = sum_h Wf[h, c] dbias[h]                             mma with K = the heads (m16n8k8 for 8, m16n8k16 for 16), C
//                                                                     on the lane's columns
//   dpair[c]   = rstd (dx^[c] - m1 - x^[c] m2),  m1 = sum_h dbias_h s_h / 128,  m2 = rstd (S2 - mean sum_h dbias_h s_h) / 128
// with S2 = sum_c dx^[c] x[c] = sum_h dbias_h (Wf . x)_h: 8 heads take it from the forward projection y = Wf . x (one mma pass on
// the raw words); 16 heads, whose dWf accumulators take 64 registers, from a first dx^ pass instead (no Wf B fragments held).
// The only per-element float work is the two FMAs of dpair (the row statistics are gram_step's).
#ifndef BWD_STAGES
#define BWD_STAGES 2
#endif
constexpr int BT_PAIR = 128 * DP * 2;
// 8 heads: two blocks per SM (128 registers), BWD_STAGES deep; 16 / 24 heads (64 / 96 dWf accumulators): one block per SM, 3 stages
template <int NH> constexpr int bwd_bps() { return NH == 8 ? 2 : 1; }   // (12 heads run as 16)
template <int NH> constexpr int bwd_stages() { return NH == 8 ? BWD_STAGES : 3; }
template <int NH> constexpr int bwd_tile() { return BT_PAIR + NH * 128 * 4; }                  // pair, then dbias
template <int NH> constexpr int bwd_smem() {
  constexpr int red = 8 * (NH * DP + NH) * 4, ring = bwd_stages<NH>() * bwd_tile<NH>();        // the end reduce reuses the ring
  return (ring > red ? ring : red) + 1024 + 64 + 32 * 16 * 4 * (nhp<NH>() / 8);
}
__device__ __forceinline__ uint32_t swz(int r, int c) {                     // byte offset of (row, column) in a swizzled pair tile
  return (uint32_t)((c >> 6) * (BT_PAIR / 2) + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2);
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(c2), "r"(su32(b)) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {                     // not merged with an earlier read of the same address
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void mma1688(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
template <int NH>
__global__ void __launch_bounds__(256, bwd_bps<NH>()) pair_bias_bwd_k(const __grid_constant__ CUtensorMap tmz, const __grid_constant__ CUtensorMap tmd,
    const bf* __restrict__ W, bf* __restrict__ DZ, float* __restrict__ AWF, float* __restrict__ AHS, int L, float eps) {
  constexpr int NT8 = nhp<NH>() / 8, BW = NT8, BT = bwd_tile<NH>(), NSB = bwd_stages<NH>();   // BW: dx^ B words per tile (one per group)
  constexpr bool YP = NH == 8;                                              // S2 from the projection y (8) or a first dx^ pass (16)
  constexpr int RING = NSB * BT > 8 * (NH * DP + NH) * 4 ? NSB * BT : 8 * (NH * DP + NH) * 4;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + RING);
  uint32_t* bwt = reinterpret_cast<uint32_t*>(smem + RING + 64);          // [u][lane][4 tiles x BW words]: Wf[heads][f(t, g)]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + 127) / 128;
  const long ntiles = (long)L * ntj;
  auto issue = [&](long n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128;
    uint64_t* fb = &full[n % NSB];
    const uint32_t st = su32(smem + (n % NSB) * BT);
    mbar_expect_tx(fb, BT);
    const int row = i * L + j0;
    tma2d(st, &tmz, 0, row, fb);
    tma2d(st + BT_PAIR / 2, &tmz, 64, row, fb);
    tma3d(st + BT_PAIR, &tmd, j0, i, 0, fb);
  };
  if (threadIdx.x == 0) {
    for (int k = 0; k < NSB; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  if (warp == 0) {                                                          // dx^ tile t's B: heads 2q, 2q+1 (| 2q+8, 2q+9) at f(t, g)
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int col = 32 * (t >> 2) + 8 * (g >> 1) + 2 * (t & 3) + (g & 1);
#pragma unroll
      for (int k = 0; k < BW; ++k)
        bwt[((t >> 2) * 32 + lane) * 4 * BW + (t & 3) * BW + k] =
            2 * q + 8 * k < NH ? pack2(__bfloat162float(W[(2 * q + 8 * k) * DP + col]), __bfloat162float(W[(2 * q + 1 + 8 * k) * DP + col])) : 0u;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int k = 0; k < NSB; ++k) issue(k);
  uint32_t bf_[1][8][2];                                                    // Wf as the forward's B (8 heads only)
  float sw0[NT8], sw1[NT8];
  if constexpr (YP) {
    wf_frags<1, 8>(W, bf_, sw0, sw1);
  } else {
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {                                      // sum_c Wf of heads 2q, 2q + 1 (+ 8 nt) alone
      float sw = 0.f;
      for (int c = 8 * q; c < DP; c += 32)
#pragma unroll
        for (int e = 0; e < 8; ++e) sw += g + 8 * nt < NH ? __bfloat162float(W[(g + 8 * nt) * DP + c + e]) : 0.f;
      sw = quad_sum(sw);
      sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
    }
  }
  float acc[8][NT8][4] = {};                                                // dWf^T tile mt: cols 16 mt + g (+8), heads 2q, 2q+1 (+8 nt)
  float hsum[NT8] = {}, corr[NT8] = {};                                     // head g + 8 nt: sum dbias, sum a mean
  const uint32_t lrow = (uint32_t)((lane & 7) + ((lane >> 4) << 3)), lcol = (uint32_t)(((lane >> 3) & 1) * 8);
  const int rb = warp * 16;
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0), st = (int)(n % NSB);
    mbar_wait(&full[st], (uint32_t)((n / NSB) & 1));
    const uint8_t* tile = smem + st * BT;
    const uint32_t tb = su32(tile);
    const float* dbs = reinterpret_cast<const float*>(tile + BT_PAIR);       // [NH][128]
    if (rb < rows) {
      float y[1][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint4 r0 = lds128(tb + swz(rb + g, 32 * u + 8 * q)), r1 = lds128(tb + swz(rb + g + 8, 32 * u + 8 * q));
        const uint32_t u0[4] = {r0.x, r0.y, r0.z, r0.w}, u1[4] = {r1.x, r1.y, r1.z, r1.w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (YP) mma16816(y[0], u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], bf_[0][2 * u + h][0], bf_[0][2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float mu0, rs0, mu1, rs1;
      gram_stats(g0, g1, cs, eps, mu0, rs0, mu1, rs1);
      // dbias of rows g, g + 8 at heads 2q + j + 8 nt (A of dx^; the row sums)
      float d0[NT8][2], d1[NT8][2], s1a = 0.f, s1b = 0.f;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const bool live = 2 * q + j + 8 * nt < NH;
          d0[nt][j] = live ? dbs[(2 * q + j + 8 * nt) * 128 + rb + g] : 0.f;
          d1[nt][j] = live ? dbs[(2 * q + j + 8 * nt) * 128 + rb + g + 8] : 0.f;
        }
        s1a += d0[nt][0] * sw0[nt] + d0[nt][1] * sw1[nt]; s1b += d1[nt][0] * sw0[nt] + d1[nt][1] * sw1[nt];
      }
      uint32_t ad0[NT8], ad1[NT8];                                          // A of dx^ per 8-head group: rows g, g + 8
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) { ad0[nt] = pack2(d0[nt][0], d0[nt][1]); ad1[nt] = pack2(d1[nt][0], d1[nt][1]); }
      // dx^ of chunk u (tiles 4u .. 4u + 3) into c[k][4]; x words of rows g, g + 8 at the same columns
      auto dxhat = [&](int u, float (&c)[4][4], uint32_t (&u0)[4], uint32_t (&u1)[4]) {
        const uint4 x0v = lds128(tb + swz(rb + g, 32 * u + 8 * q)), x1v = lds128(tb + swz(rb + g + 8, 32 * u + 8 * q));
        u0[0] = x0v.x; u0[1] = x0v.y; u0[2] = x0v.z; u0[3] = x0v.w; u1[0] = x1v.x; u1[1] = x1v.y; u1[2] = x1v.z; u1[3] = x1v.w;
        uint32_t bw4[4 * BW];
#pragma unroll
        for (int k = 0; k < BW; ++k) {
          const uint4 bv = lds128(su32(bwt + (u * 32 + lane) * 4 * BW + 4 * k));
          bw4[4 * k] = bv.x; bw4[4 * k + 1] = bv.y; bw4[4 * k + 2] = bv.z; bw4[4 * k + 3] = bv.w;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {                                       // K = the heads: m16n8k16 per two groups, m16n8k8 for an odd one
          c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0.f;
#pragma unroll
          for (int p = 0; p + 1 < NT8; p += 2)
            mma16816(c[k], ad0[p], ad1[p], ad0[p + 1], ad1[p + 1], bw4[BW * k + p], bw4[BW * k + p + 1]);
          if constexpr (NT8 & 1) mma1688(c[k], ad0[NT8 - 1], ad1[NT8 - 1], bw4[BW * k + NT8 - 1]);
        }
      };
      float s2a = 0.f, s2b = 0.f;
      if constexpr (YP) {
        s2a = d0[0][0] * y[0][0] + d0[0][1] * y[0][1]; s2b = d1[0][0] * y[0][2] + d1[0][1] * y[0][3];
      } else {
#pragma unroll
        for (int u = 0; u < 4; ++u) {                                       // first pass: S2 = sum_c dx^ x
          float c[4][4]; uint32_t u0[4], u1[4];
          dxhat(u, c, u0, u1);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 x0 = unpack2(u0[k]), x1 = unpack2(u1[k]);
            s2a = fmaf(c[k][0], x0.x, fmaf(c[k][1], x0.y, s2a)); s2b = fmaf(c[k][2], x1.x, fmaf(c[k][3], x1.y, s2b));
          }
        }
      }
      const float S1a = quad_sum(s1a), S1b = quad_sum(s1b), S2a = quad_sum(s2a), S2b = quad_sum(s2b);
      const float m1a = S1a * (1.f / DP), m1b = S1b * (1.f / DP);
      const float m2a = rs0 * (S2a - mu0 * S1a) * (1.f / DP), m2b = rs1 * (S2b - mu1 * S1b) * (1.f / DP);
      const float ka = -rs0 * rs0 * m2a, kb = -rs1 * rs1 * m2b;              // dpair = rstd dx^ + k x + c0
      const float ca = rs0 * (rs0 * m2a * mu0 - m1a), cb = rs1 * (rs1 * m2b * mu1 - m1b);
      const long R = (long)i * L + j0 + rb;
#pragma unroll
      for (int u = 0; u < 4; ++u) {                                         // tiles 4u .. 4u + 3 fill chunk u: one 16-B store
        float c[4][4]; uint32_t u0[4], u1[4];
        dxhat(u, c, u0, u1);
        uint32_t o0[4], o1[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 x0 = unpack2(u0[k]), x1 = unpack2(u1[k]);
          o0[k] = pack2(fmaf(rs0, c[k][0], fmaf(ka, x0.x, ca)), fmaf(rs0, c[k][1], fmaf(ka, x0.y, ca)));
          o1[k] = pack2(fmaf(rs1, c[k][2], fmaf(kb, x1.x, cb)), fmaf(rs1, c[k][3], fmaf(kb, x1.y, cb)));
        }
        *reinterpret_cast<uint4*>(DZ + (R + g) * DP + 32 * u + 8 * q) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
        *reinterpret_cast<uint4*>(DZ + (R + g + 8) * DP + 32 * u + 8 * q) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
      }
      // dWf: B = a = rstd dbias[head g + 8 nt][rows 2q, 2q+1 | 2q+8, 2q+9]; rstd / mean of those rows from their lanes
      const float ra = __shfl_sync(0xffffffffu, rs0, 8 * q), rb_ = __shfl_sync(0xffffffffu, rs0, 8 * q + 4);
      const float rc = __shfl_sync(0xffffffffu, rs1, 8 * q), rd = __shfl_sync(0xffffffffu, rs1, 8 * q + 4);
      const float ma = __shfl_sync(0xffffffffu, mu0, 8 * q), mb = __shfl_sync(0xffffffffu, mu0, 8 * q + 4);
      const float mc = __shfl_sync(0xffffffffu, mu1, 8 * q), md = __shfl_sync(0xffffffffu, mu1, 8 * q + 4);
      uint32_t db0[NT8], db1[NT8];
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const bool live = g + 8 * nt < NH;
        const float2 e01 = live ? *reinterpret_cast<const float2*>(dbs + (g + 8 * nt) * 128 + rb + 2 * q) : make_float2(0.f, 0.f);
        const float2 e89 = live ? *reinterpret_cast<const float2*>(dbs + (g + 8 * nt) * 128 + rb + 2 * q + 8) : make_float2(0.f, 0.f);
        const float aa = ra * e01.x, ab = rb_ * e01.y, ac = rc * e89.x, ad = rd * e89.y;
        hsum[nt] += e01.x + e01.y + e89.x + e89.y;
        corr[nt] += aa * ma + ab * mb + ac * mc + ad * md;
        db0[nt] = pack2(aa, ab); db1[nt] = pack2(ac, ad);
      }
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        uint32_t a[4];                                                      // A = x^T: [16 columns][16 rows]
        ldsm_x4_trans(a, tb + swz(rb + (int)lrow, 16 * mt + (int)lcol));
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) mma16816(acc[mt][nt], a[0], a[1], a[2], a[3], db0[nt], db1[nt]);
      }
    }
    __syncthreads();                                                        // every warp is done with the stage
    if (threadIdx.x == 0) issue(n + NSB);
  }
  // dWf -= the per-head mean term, then block-reduce the warps' tiles and head sums; one atomic add per entry
  constexpr int RS = NH * DP + NH;
  float* red = reinterpret_cast<float*>(smem);                              // the ring is idle now
  float* pw = red + warp * RS;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    const float cq = quad_sum(corr[nt]), hq = quad_sum(hsum[nt]);           // head g + 8 nt
    const float c0 = __shfl_sync(0xffffffffu, cq, 8 * q), c1 = __shfl_sync(0xffffffffu, cq, 8 * q + 4);   // heads 2q, 2q + 1 (+ 8 nt)
    const int h0 = 2 * q + 8 * nt;
    if (h0 < NH) {                                                          // (NH is even: h0 + 1 < NH too)
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        pw[h0 * DP + 16 * mt + g] = acc[mt][nt][0] - c0; pw[(h0 + 1) * DP + 16 * mt + g] = acc[mt][nt][1] - c1;
        pw[h0 * DP + 16 * mt + g + 8] = acc[mt][nt][2] - c0; pw[(h0 + 1) * DP + 16 * mt + g + 8] = acc[mt][nt][3] - c1;
      }
    }
    if (q == 0 && g + 8 * nt < NH) pw[NH * DP + g + 8 * nt] = hq;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < RS; c += 256) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w * RS + c];
    atomicAdd(c < NH * DP ? AWF + c : AHS + (c - NH * DP), v);
  }
}

// ------------------------------------------------------------------------------------------------ single track
// Lane l's CPL contiguous columns of a row as CPL / 4 four-wide vectors.
template <int V, typename T> __device__ __forceinline__ void ldv(const T* row, float4 (&v)[V]) {
  const int c = 4 * V * (threadIdx.x & 31);
#pragma unroll
  for (int k = 0; k < V; ++k) v[k] = V4<T>::load(row + c + 4 * k);
}
template <int V, typename T> __device__ __forceinline__ void stv(T* row, const float4 (&v)[V]) {
  const int c = 4 * V * (threadIdx.x & 31);
#pragma unroll
  for (int k = 0; k < V; ++k) V4<T>::store(row + c + 4 * k, v[k]);
}
template <int V> __device__ __forceinline__ float sumv(const float4 (&v)[V]) {
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < V; ++k) t += sum4(v[k]);
  return t;
}
template <int V> __device__ __forceinline__ float dotv(const float4 (&a)[V], const float4 (&b)[V]) {
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < V; ++k) t += sum4(mul4(a[k], b[k]));
  return t;
}
// Column sums of the block's rows (v: this lane's columns, zero for a row past M) added into out[0 .. 128 V - 1].
template <int V> __device__ __forceinline__ void colsum_add(const float4 (&v)[V], float* red, float* out) {
  constexpr int WD = 128 * V;
  const int warp = threadIdx.x >> 5, c = 4 * V * (threadIdx.x & 31);
#pragma unroll
  for (int k = 0; k < V; ++k) *reinterpret_cast<float4*>(red + warp * WD + c + 4 * k) = v[k];
  __syncthreads();
  for (int j = threadIdx.x; j < WD; j += NT) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < RW; ++w) t += red[w * WD + j];
    atomicAdd(out + j, t);
  }
}

template <typename XT, int V>
__global__ void __launch_bounds__(NT) ln_rows_k(const XT* __restrict__ X, const float* __restrict__ LW, const float* __restrict__ LB,
    bf* __restrict__ XA, float2* __restrict__ ST, XT* __restrict__ Y, int M, float eps) {
  constexpr int D = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  float4 x[V], w[V], b[V];
  ldv<V>(X + r * D, x); ldv<V>(LW, w); ldv<V>(LB, b);
  if (Y) stv<V>(Y + r * D, x);              // the residual seed: to_out accumulates onto it in place (cuBLAS beta = 1)
  const float mean = warp_sum(sumv<V>(x)) * (1.f / D);
#pragma unroll
  for (int k = 0; k < V; ++k) x[k] = make_float4(x[k].x - mean, x[k].y - mean, x[k].z - mean, x[k].w - mean);
  const float rstd = rsqrtf(warp_sum(dotv<V>(x, x)) * (1.f / D) + eps);
#pragma unroll
  for (int k = 0; k < V; ++k)
    x[k] = make_float4(x[k].x * rstd * w[k].x + b[k].x, x[k].y * rstd * w[k].y + b[k].y, x[k].z * rstd * w[k].z + b[k].z,
                       x[k].w * rstd * w[k].w + b[k].w);
  stv<V>(XA + r * D, x);
  if (ST && (threadIdx.x & 31) == 0) ST[r] = make_float2(mean, rstd);
}

// og = sigmoid(g) o over W = 128 V columns (g: the qkvg row's last quarter)
template <int V>
__global__ void __launch_bounds__(NT) gate_rows_k(const float* __restrict__ O, const bf* __restrict__ QKVG, bf* __restrict__ OG, int M) {
  constexpr int WD = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  float4 o[V], g[V];
  ldv<V>(O + r * WD, o); ldv<V>(QKVG + r * 4 * WD + 3 * WD, g);
#pragma unroll
  for (int k = 0; k < V; ++k) o[k] = mul4(sig4(g[k]), o[k]);
  stv<V>(OG + r * WD, o);
}

// og = sigmoid(g) o: dO = dog s (bf16, the core's input), D = rowsum_head(dO o) as [NH, M], dg = dog o s (1 - s). A head is
// HL lanes (W / NH / (4 V)).
template <int V, int HL>
__global__ void __launch_bounds__(NT) gate_bwd_k(const bf* __restrict__ DOG, const float* __restrict__ O, const bf* __restrict__ QKVG,
    bf* __restrict__ DOB, float* __restrict__ DD, bf* __restrict__ DQKVG, int M) {
  constexpr int WD = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  const int lane = threadIdx.x & 31;
  float4 dog[V], o[V], s[V], d_o[V], dg[V];
  ldv<V>(DOG + r * WD, dog); ldv<V>(O + r * WD, o); ldv<V>(QKVG + r * 4 * WD + 3 * WD, s);
#pragma unroll
  for (int k = 0; k < V; ++k) {
    s[k] = sig4(s[k]);
    d_o[k] = mul4(dog[k], s[k]);
    const float4 t = mul4(dog[k], o[k]);
    dg[k] = make_float4(t.x * s[k].x * (1.f - s[k].x), t.y * s[k].y * (1.f - s[k].y), t.z * s[k].z * (1.f - s[k].z),
                        t.w * s[k].w * (1.f - s[k].w));
  }
  stv<V>(DOB + r * WD, d_o);
  stv<V>(DQKVG + r * 4 * WD + 3 * WD, dg);
  float dsum = dotv<V>(d_o, o);
#pragma unroll
  for (int o_ = 1; o_ < HL; o_ <<= 1) dsum += __shfl_xor_sync(0xffffffffu, dsum, o_);
  if (lane % HL == 0) DD[(long)(lane / HL) * M + r] = dsum;
}

// gate_bwd with a head per lane (24 x 16, 12 x 32; W = 384): lane h < NH owns head h's HC columns, the rest idle
template <int NH, int HC>
__global__ void __launch_bounds__(NT) gate_bwd_hl_k(const bf* __restrict__ DOG, const float* __restrict__ O, const bf* __restrict__ QKVG,
    bf* __restrict__ DOB, float* __restrict__ DD, bf* __restrict__ DQKVG, int M) {
  constexpr int WD = NH * HC, V = HC / 4;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (r >= M || lane >= NH) return;
  const int c = HC * lane;
  float4 dog[V], o[V], s[V], d_o[V], dg[V];
#pragma unroll
  for (int k = 0; k < V; ++k) {
    dog[k] = V4<bf>::load(DOG + r * WD + c + 4 * k); o[k] = V4<float>::load(O + r * WD + c + 4 * k);
    s[k] = sig4(V4<bf>::load(QKVG + r * 4 * WD + 3 * WD + c + 4 * k));
    d_o[k] = mul4(dog[k], s[k]);
    const float4 t = mul4(dog[k], o[k]);
    dg[k] = make_float4(t.x * s[k].x * (1.f - s[k].x), t.y * s[k].y * (1.f - s[k].y), t.z * s[k].z * (1.f - s[k].z),
                        t.w * s[k].w * (1.f - s[k].w));
    V4<bf>::store(DOB + r * WD + c + 4 * k, d_o[k]);
    V4<bf>::store(DQKVG + r * 4 * WD + 3 * WD + c + 4 * k, dg[k]);
  }
  DD[(long)lane * M + r] = dotv<V>(d_o, o);
}

// dq (NCH key-chunk partials [NCH, M, W], added in chunk order: deterministic) | dk | dv fp32 [M, W] -> dqkvg[:, 0:3W] bf16;
// ABQ[c] += sum_r dq (bf16-rounded, what the GEMMs see)
template <int V>
__global__ void __launch_bounds__(NT) qkv_bwd_k(const float* __restrict__ DQ, const float* __restrict__ DK, const float* __restrict__ DV,
    bf* __restrict__ DQKVG, float* __restrict__ ABQ, int M, int NCH) {
  constexpr int WD = 128 * V;
  __shared__ __align__(16) float red[RW * WD];
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  float4 v[V] = {};
  if (r < M) {
    ldv<V>(DQ + r * WD, v);
    for (int c = 1; c < NCH; ++c) {
      float4 p[V];
      ldv<V>(DQ + ((long)c * M + r) * WD, p);
#pragma unroll
      for (int k = 0; k < V; ++k) v[k] = add4(v[k], p[k]);
    }
#pragma unroll
    for (int k = 0; k < V; ++k) v[k] = bfround(v[k]);
    stv<V>(DQKVG + r * 4 * WD, v);
    float4 t[V];
    ldv<V>(DK + r * WD, t); stv<V>(DQKVG + r * 4 * WD + WD, t);
    ldv<V>(DV + r * WD, t); stv<V>(DQKVG + r * 4 * WD + 2 * WD, t);
  }
  colsum_add<V>(v, red, ABQ);
}

// x^ = LN(x); dx = dy + rstd (dxa w - mean(dxa w) - x^ mean(dxa w x^)); ALNW += dxa x^, ALNB += dxa
template <typename XT, typename OT, int V>
__global__ void __launch_bounds__(NT) ln_bwd_k(const bf* __restrict__ DXA, const XT* __restrict__ X, const float2* __restrict__ ST,
    const float* __restrict__ LW, const OT* __restrict__ DY, OT* __restrict__ DX, float* __restrict__ ALNW, float* __restrict__ ALNB, int M) {
  constexpr int D = 128 * V;
  __shared__ __align__(16) float red[RW * D];
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  float4 aw[V] = {}, ab[V] = {};
  if (r < M) {
    float4 dxa[V], xh[V], w[V], g[V], dy[V];
    ldv<V>(DXA + r * D, dxa); ldv<V>(X + r * D, xh); ldv<V>(LW, w); ldv<V>(DY + r * D, dy);
    const float2 st = ST[r];
#pragma unroll
    for (int k = 0; k < V; ++k) {
      xh[k] = make_float4((xh[k].x - st.x) * st.y, (xh[k].y - st.x) * st.y, (xh[k].z - st.x) * st.y, (xh[k].w - st.x) * st.y);
      g[k] = mul4(dxa[k], w[k]);
      aw[k] = mul4(dxa[k], xh[k]);
      ab[k] = dxa[k];
    }
    const float m1 = warp_sum(sumv<V>(g)) * (1.f / D), m2 = warp_sum(dotv<V>(g, xh)) * (1.f / D);
#pragma unroll
    for (int k = 0; k < V; ++k)
      dy[k] = make_float4(dy[k].x + st.y * (g[k].x - m1 - xh[k].x * m2), dy[k].y + st.y * (g[k].y - m1 - xh[k].y * m2),
                          dy[k].z + st.y * (g[k].z - m1 - xh[k].z * m2), dy[k].w + st.y * (g[k].w - m1 - xh[k].w * m2));
    stv<V>(DX + r * D, dy);
  }
  colsum_add<V>(aw, red, ALNW);
  __syncthreads();
  colsum_add<V>(ab, red, ALNB);
}

// Head geometry: nh heads of dh real channels, dhp wide in memory (dhp = dh: no padding). A padded index p (within a W = nh dhp
// segment) is real channel (p / dhp) dh + p % dhp when p % dhp < dh, else a pad (zero weight / ignored gradient).
struct Geo { int nh, dh, dhp, d; };                                         // d = nh dh = d_single
__device__ __forceinline__ int real_of(const Geo& G, int p) { const int c = p % G.dhp; return c < G.dh ? (p / G.dhp) * G.dh + c : -1; }

// W q | k | v | g [4 W, D] (padded rows; q rows x qs), its bias [4 W] (bq x qs, then 0), Wo [D, W] (padded columns) or none,
// Wf = Wb * ln_pair.weight * ws [NH, 128]; bf16. 8 weights per thread.
__global__ void __launch_bounds__(256) prep_k(const bf* __restrict__ WQ, const bf* __restrict__ BQ, const bf* __restrict__ WK,
    const bf* __restrict__ WV, const bf* __restrict__ WG, const bf* __restrict__ WO, const bf* __restrict__ WB, const float* __restrict__ LPW,
    bf* __restrict__ WP, bf* __restrict__ BV, bf* __restrict__ WOP, bf* __restrict__ WF, Geo G, float qs, float ws) {
  const int W = G.nh * G.dhp, D = G.d;
  const long n1 = 4L * W * D / 8, n2 = WOP ? (long)D * W / 8 : 0, i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  if (i < n1) {                                                             // packed row p (of segment blk), 8 input columns
    const long e = i * 8, prow = e / D, col = e % D;
    const int blk = (int)(prow / W), rr = real_of(G, (int)(prow % W));
    uint4 u = zero;
    if (rr >= 0) {
      const bf* src = (blk == 0 ? WQ : blk == 1 ? WK : blk == 2 ? WV : WG) + (long)rr * D + col;
      u = *reinterpret_cast<const uint4*>(src);
      if (blk == 0) {
        uint32_t* w = reinterpret_cast<uint32_t*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = unpack2(w[k]); w[k] = pack2(f.x * qs, f.y * qs); }
      }
    }
    *reinterpret_cast<uint4*>(WP + e) = u;
    return;
  }
  if (i < n1 + n2) {                                                        // Wo row r, 8 padded columns (dh % 8 == 0: one head)
    const long e = (i - n1) * 8, r = e / W, pc = e % W;
    const int rc = real_of(G, (int)pc);
    *reinterpret_cast<uint4*>(WOP + e) = rc >= 0 ? *reinterpret_cast<const uint4*>(WO + r * D + rc) : zero;
    return;
  }
  const long j = i - n1 - n2;
  if (j < 4 * W) {
    const int rr = j < W ? real_of(G, (int)j) : -1;
    BV[j] = __float2bfloat16_rn(rr >= 0 ? __bfloat162float(BQ[rr]) * qs : 0.f);
  } else if (j < 4 * W + G.nh * DP) {
    const int k = (int)(j - 4 * W);
    WF[k] = __float2bfloat16_rn(__bfloat162float(WB[k]) * LPW[k % DP] * ws);
  }
}

// Parameter gradients, each into its own fp32 tensor (the parameters may be an fp32 master; the integration casts to theirs):
// dWq | dWk | dWv | dWg (the real rows of dWp [4 W, D]), dWo (the real columns of dWo [D, W]), and from the accumulator dlnw, dlnb,
// dbq, dWb = dWf ln_pair.weight, dln_pair.weight = sum_h dWf Wb, dln_pair.bias = 0 (the softmax cancels a per-head constant; see the
// integration).
__global__ void __launch_bounds__(256) finalize_k(const float* __restrict__ DWP, const float* __restrict__ DWO, const float* __restrict__ ACC,
    const bf* __restrict__ WB, const float* __restrict__ LPW, float* __restrict__ OQ, float* __restrict__ OKe, float* __restrict__ OV,
    float* __restrict__ OG, float* __restrict__ OO, float* __restrict__ OLW, float* __restrict__ OLB, float* __restrict__ OBQ,
    float* __restrict__ OWB, float* __restrict__ OLPW, float* __restrict__ OLPB, Geo G) {
  const int W = G.nh * G.dhp, NHD = G.nh * DP, D = G.d;
  const float *ALNW = ACC, *ALNB = ACC + D, *ABQ = ACC + 2 * D, *AWF = ACC + 2 * D + W;
  const long nv = 5L * D * D / 4, i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nv) {                                                             // 4 gradients per thread
    const long e = i * 4, blk = e / (D * D), off = e % (D * D), r = off / D, c = off % D;
    if (blk < 4) {                                                          // real row r of segment blk <- padded row
      const long prow = (long)blk * W + (r / G.dh) * G.dhp + r % G.dh;
      float* dst = (blk == 0 ? OQ : blk == 1 ? OKe : blk == 2 ? OV : OG) + off;
      V4<float>::store(dst, V4<float>::load(DWP + prow * D + c));
    } else {                                                                // dWo row r, real columns c .. c + 3 (within one head)
      V4<float>::store(OO + off, V4<float>::load(DWO + r * W + (c / G.dh) * G.dhp + c % G.dh));
    }
    return;
  }
  const long j = i - nv;
  if (j < D) OLW[j] = ALNW[j];
  else if (j < 2 * D) OLB[j - D] = ALNB[j - D];
  else if (j < 3 * D) { const int c = (int)(j - 2 * D); OBQ[c] = ABQ[(c / G.dh) * G.dhp + c % G.dh]; }
  else if (j < 3 * D + NHD) { const int k = (int)(j - 3 * D); OWB[k] = AWF[k] * LPW[k % DP]; }
  else if (j < 3 * D + NHD + DP) {
    const int c = (int)(j - 3 * D - NHD);
    float t = 0.f;
    for (int h = 0; h < G.nh; ++h) t += AWF[h * DP + c] * __bfloat162float(WB[h * DP + c]);
    OLPW[c] = t;
  } else if (j < 3 * D + NHD + 2 * DP) OLPB[j - 3 * D - NHD - DP] = 0.f;
}

// ------------------------------------------------------------------------------------------------ host
#define APB_DISPATCH(T, NAME, ...)                                                                   \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    TORCH_CHECK((T) == at::kBFloat16, "fp32 or bf16");                                                 \
    using NAME = bf; return __VA_ARGS__();                                                           \
  }()
template <typename C> C* P(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }
template <typename C> const C* CP(const at::Tensor& t) { return reinterpret_cast<const C*>(t.data_ptr()); }
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
unsigned blocks(int64_t M) { return (unsigned)((M + RW - 1) / RW); }
void chk(const at::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), n, ": contiguous CUDA tensor of the expected dtype");
}
// (heads, d_single) -> geometry: 8 x 48, 12 x 32, 16 x 24 (32 wide), 24 x 16 at d 384; 16 x 32 at d 512
Geo geo(int64_t nh, int64_t d) {
  const Geo G = d == 384 ? (nh == 8 ? Geo{8, 48, 48, 384} : nh == 12 ? Geo{12, 32, 32, 384} : nh == 16 ? Geo{16, 24, 32, 384}
                                     : Geo{24, 16, 16, 384})
                         : Geo{16, 32, 32, 512};
  TORCH_CHECK((d == 384 && (nh == 8 || nh == 12 || nh == 16 || nh == 24)) || (d == 512 && nh == 16),
              "heads x d_single: 8 / 12 / 16 / 24 x 384 or 16 x 512, got ", nh, " x ", d);
  return G;
}
int64_t width(const Geo& G) { return (int64_t)G.nh * G.dhp; }
void chk_acc(const at::Tensor& acc, const Geo& G) {
  chk(acc, at::kFloat, "acc");
  const int n = acc_n(G.d, G.nh, (int)width(G));
  TORCH_CHECK(acc.numel() == n, "acc: ", n, " floats");
}
int nsm_of(const at::Tensor& t) {
  int nsm = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, t.get_device()));
  return nsm;
}

template <int NH>
void pair_bias_fwd_t(const at::Tensor& z, const at::Tensor& w, const c10::optional<at::Tensor>& mask, at::Tensor& out, int64_t L, double eps,
                     double neg) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_fwd_k<NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, fwd_smem<NH>()));
    attr = true;
  }
  const long ntiles = L * ((L + 127) / 128);
  pair_bias_fwd_k<NH><<<(unsigned)std::min<long>(2L * nsm_of(z), ntiles), FNT, fwd_smem<NH>(), S()>>>(
      CP<bf>(z), CP<bf>(w), mask ? CP<bool>(*mask) : nullptr, P<bf>(out), (int)L, (float)eps, (float)neg);
}
void pair_bias_fwd(at::Tensor z, at::Tensor w, c10::optional<at::Tensor> mask, at::Tensor out, int64_t L, double eps, double neg) {
  chk(z, at::kBFloat16, "pair"); chk(w, at::kBFloat16, "wf"); chk(out, at::kBFloat16, "bias");
  const int64_t nh = w.size(0);
  TORCH_CHECK(L % 16 == 0 && z.numel() == L * L * DP && w.numel() == nh * DP && out.numel() == nh * L * L, "pair_bias_fwd shapes");
  TORCH_CHECK(nh == 8 || nh == 12 || nh == 16 || nh == 24, "pair_bias: 8, 12, 16 or 24 heads");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  const at::cuda::CUDAGuard gd(z.device());
  if (nh == 8) pair_bias_fwd_t<8>(z, w, mask, out, L, eps, neg);
  else if (nh == 12) pair_bias_fwd_t<12>(z, w, mask, out, L, eps, neg);
  else if (nh == 16) pair_bias_fwd_t<16>(z, w, mask, out, L, eps, neg);
  else pair_bias_fwd_t<24>(z, w, mask, out, L, eps, neg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

CUtensorMap tensor_map(void* ptr, CUtensorMapDataType dt, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                       const cuuint32_t* box, CUtensorMapSwizzle sw) {
  static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
  if (!enc) {
    cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), cudaEnableDefault, &qr));
    TORCH_CHECK(qr == cudaDriverEntryPointSuccess && enc, "cuTensorMapEncodeTiled unavailable");
  }
  CUtensorMap m;
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult r = enc(&m, dt, rank, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

template <int NH>
void pair_bias_bwd_t(const at::Tensor& z, const at::Tensor& dbias, const at::Tensor& w, at::Tensor& dz, at::Tensor& acc, int64_t L, double eps,
                     const Geo& G) {
  const cuuint64_t zd[2] = {(cuuint64_t)DP, (cuuint64_t)(L * L)}, zs[1] = {DP * 2};
  const cuuint32_t zb[2] = {64, 128};
  const CUtensorMap mz = tensor_map(z.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, zd, zs, zb, CU_TENSOR_MAP_SWIZZLE_128B);
  const cuuint64_t dd[3] = {(cuuint64_t)L, (cuuint64_t)L, NH}, ds[2] = {(cuuint64_t)L * 4, (cuuint64_t)(L * L * 4)};
  const cuuint32_t db[3] = {128, 1, NH};
  const CUtensorMap md = tensor_map(dbias.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, dd, ds, db, CU_TENSOR_MAP_SWIZZLE_NONE);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_bwd_k<NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, bwd_smem<NH>()));
    attr = true;
  }
  const long ntiles = L * ((L + 127) / 128);
  float* awf = P<float>(acc) + 2 * G.d + width(G);
  pair_bias_bwd_k<NH><<<(unsigned)std::min<long>((long)bwd_bps<NH>() * nsm_of(z), ntiles), 256, bwd_smem<NH>(), S()>>>(
      mz, md, CP<bf>(w), P<bf>(dz), awf, awf + NH * DP, (int)L, (float)eps);
}
void pair_bias_bwd(at::Tensor z, at::Tensor dbias, at::Tensor w, at::Tensor dz, at::Tensor acc, int64_t L, double eps, int64_t d) {
  chk(z, at::kBFloat16, "pair"); chk(dbias, at::kFloat, "dbias"); chk(w, at::kBFloat16, "wf"); chk(dz, at::kBFloat16, "dpair");
  const int64_t nh = w.size(0);
  const Geo G = geo(nh, d);
  chk_acc(acc, G);
  TORCH_CHECK(L % 16 == 0 && z.numel() == L * L * DP && dbias.numel() == nh * L * L, "pair_bias_bwd shapes");
  const at::cuda::CUDAGuard gd(z.device());
  if (nh == 8) pair_bias_bwd_t<8>(z, dbias, w, dz, acc, L, eps, G);
  else if (nh == 12) pair_bias_bwd_t<12>(z, dbias, w, dz, acc, L, eps, G);
  else if (nh == 16) pair_bias_bwd_t<16>(z, dbias, w, dz, acc, L, eps, G);
  else pair_bias_bwd_t<24>(z, dbias, w, dz, acc, L, eps, G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_rows(at::Tensor x, at::Tensor lw, at::Tensor lb, at::Tensor xa, c10::optional<at::Tensor> st, double eps,
             c10::optional<at::Tensor> y) {
  const int64_t D = lw.numel(), M = x.numel() / D;
  TORCH_CHECK(D == 384 || D == 512, "ln_rows: d_single 384 or 512");
  TORCH_CHECK(x.is_contiguous(), "ln_rows: contiguous x"); chk(lw, at::kFloat, "lw"); chk(lb, at::kFloat, "lb"); chk(xa, at::kBFloat16, "xa");
  if (y) chk(*y, x.scalar_type(), "y (residual seed, x's dtype)");
  const at::cuda::CUDAGuard gd(x.device());
  APB_DISPATCH(x.scalar_type(), XT, [&] {
    auto k = D == 384 ? ln_rows_k<XT, 3> : ln_rows_k<XT, 4>;
    k<<<blocks(M), NT, 0, S()>>>(CP<XT>(x), CP<float>(lw), CP<float>(lb), P<bf>(xa), st ? P<float2>(*st) : nullptr,
                                 y ? P<XT>(*y) : nullptr, (int)M, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// width W = o.size(1): 384 (8 x 48: 3 vectors per lane, a head 4 lanes) or 512 (16 x 32: 4 vectors, a head 2 lanes)
void gate_rows(at::Tensor o, at::Tensor qkvg, at::Tensor og) {
  const int64_t M = o.size(0), W = o.size(1);
  chk(o, at::kFloat, "o"); chk(qkvg, at::kBFloat16, "qkvg"); chk(og, at::kBFloat16, "og");
  TORCH_CHECK((W == 384 || W == 512) && qkvg.size(1) == 4 * W && og.size(1) == W, "gate_rows widths");
  const at::cuda::CUDAGuard gd(o.device());
  if (W == 384) gate_rows_k<3><<<blocks(M), NT, 0, S()>>>(CP<float>(o), CP<bf>(qkvg), P<bf>(og), (int)M);
  else gate_rows_k<4><<<blocks(M), NT, 0, S()>>>(CP<float>(o), CP<bf>(qkvg), P<bf>(og), (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_bwd(at::Tensor dog, at::Tensor o, at::Tensor qkvg, at::Tensor dob, at::Tensor dd, at::Tensor dqkvg) {
  const int64_t M = o.size(0), W = o.size(1);
  chk(dog, at::kBFloat16, "dog"); chk(o, at::kFloat, "o"); chk(dob, at::kBFloat16, "dob"); chk(dd, at::kFloat, "dd");
  chk(dqkvg, at::kBFloat16, "dqkvg");
  const int64_t nh = dd.size(0);
  TORCH_CHECK((W == 384 && (nh == 8 || nh == 12 || nh == 24)) || (W == 512 && nh == 16), "gate_bwd widths");
  const at::cuda::CUDAGuard gd(o.device());
  if (nh == 24)
    gate_bwd_hl_k<24, 16><<<blocks(M), NT, 0, S()>>>(CP<bf>(dog), CP<float>(o), CP<bf>(qkvg), P<bf>(dob), P<float>(dd), P<bf>(dqkvg), (int)M);
  else if (nh == 12)
    gate_bwd_hl_k<12, 32><<<blocks(M), NT, 0, S()>>>(CP<bf>(dog), CP<float>(o), CP<bf>(qkvg), P<bf>(dob), P<float>(dd), P<bf>(dqkvg), (int)M);
  else if (W == 384) gate_bwd_k<3, 4><<<blocks(M), NT, 0, S()>>>(CP<bf>(dog), CP<float>(o), CP<bf>(qkvg), P<bf>(dob), P<float>(dd), P<bf>(dqkvg), (int)M);
  else gate_bwd_k<4, 2><<<blocks(M), NT, 0, S()>>>(CP<bf>(dog), CP<float>(o), CP<bf>(qkvg), P<bf>(dob), P<float>(dd), P<bf>(dqkvg), (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dq: [M, W] or key-chunk partials [NCH, M, W]; acc: the step's accumulator (d = d_single locates its dbq slice)
void qkv_bwd(at::Tensor dq, at::Tensor dk, at::Tensor dv, at::Tensor dqkvg, at::Tensor acc, int64_t d) {
  const int64_t NCH = dq.dim() == 3 ? dq.size(0) : 1, M = dk.size(0), W = dk.size(1);
  TORCH_CHECK(dq.numel() == NCH * M * W, "qkv_bwd: dq [NCH, M, W]");
  chk(dq, at::kFloat, "dq"); chk(dk, at::kFloat, "dk"); chk(dv, at::kFloat, "dv"); chk(dqkvg, at::kBFloat16, "dqkvg");
  chk(acc, at::kFloat, "acc");
  TORCH_CHECK(W == 384 || W == 512, "qkv_bwd width");
  const at::cuda::CUDAGuard gd(dq.device());
  float* abq = P<float>(acc) + 2 * d;
  if (W == 384) qkv_bwd_k<3><<<blocks(M), NT, 0, S()>>>(CP<float>(dq), CP<float>(dk), CP<float>(dv), P<bf>(dqkvg), abq, (int)M, (int)NCH);
  else qkv_bwd_k<4><<<blocks(M), NT, 0, S()>>>(CP<float>(dq), CP<float>(dk), CP<float>(dv), P<bf>(dqkvg), abq, (int)M, (int)NCH);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_bwd(at::Tensor dxa, at::Tensor x, at::Tensor st, at::Tensor lw, at::Tensor dy, at::Tensor dx, at::Tensor acc) {
  const int64_t D = lw.numel(), M = x.numel() / D;
  TORCH_CHECK(D == 384 || D == 512, "ln_bwd: d_single 384 or 512");
  chk(dxa, at::kBFloat16, "dxa"); chk(lw, at::kFloat, "lw"); chk(st, at::kFloat, "stats"); chk(acc, at::kFloat, "acc");
  TORCH_CHECK(x.is_contiguous() && dy.is_contiguous() && dx.is_contiguous() && dy.scalar_type() == dx.scalar_type(), "ln_bwd");
  const at::cuda::CUDAGuard gd(x.device());
  APB_DISPATCH(x.scalar_type(), XT, [&] {
    APB_DISPATCH(dx.scalar_type(), OT, [&] {
      auto k = D == 384 ? ln_bwd_k<XT, OT, 3> : ln_bwd_k<XT, OT, 4>;
      k<<<blocks(M), NT, 0, S()>>>(CP<bf>(dxa), CP<XT>(x), CP<float2>(st), CP<float>(lw), CP<OT>(dy), P<OT>(dx), P<float>(acc),
                                   P<float>(acc) + D, (int)M);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prep(at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor wv, at::Tensor wg, c10::optional<at::Tensor> wo, at::Tensor wb,
          at::Tensor lpw, at::Tensor wp, at::Tensor bv, c10::optional<at::Tensor> wop, at::Tensor wf, double qs, double ws) {
  for (auto* t : {&wq, &bq, &wk, &wv, &wg, &wb, &wp, &bv, &wf}) chk(*t, at::kBFloat16, "prep: bf16");
  chk(lpw, at::kFloat, "ln_pair.weight");
  const Geo G = geo(wb.size(0), wq.size(1));
  const int64_t W = width(G), D = G.d;
  TORCH_CHECK(wq.numel() == D * D && wp.numel() == 4 * W * D && bv.numel() == 4 * W && wf.numel() == G.nh * DP, "prep shapes");
  TORCH_CHECK(wo.has_value() == wop.has_value(), "prep: wo and wop together");
  if (wop) { chk(*wo, at::kBFloat16, "wo"); chk(*wop, at::kBFloat16, "wop"); TORCH_CHECK(wop->numel() == D * W, "wop [D, W]"); }
  const at::cuda::CUDAGuard gd(wq.device());
  const long n = 4L * W * D / 8 + (wop ? (long)D * W / 8 : 0) + 4 * W + G.nh * DP;
  prep_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<bf>(wq), CP<bf>(bq), CP<bf>(wk), CP<bf>(wv), CP<bf>(wg), wo ? CP<bf>(*wo) : nullptr,
      CP<bf>(wb), CP<float>(lpw), P<bf>(wp), P<bf>(bv), wop ? P<bf>(*wop) : nullptr, P<bf>(wf), G, (float)qs, (float)ws);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize(at::Tensor dwp, at::Tensor dwo, at::Tensor acc, at::Tensor wb, at::Tensor lpw, std::vector<at::Tensor> out) {
  chk(dwp, at::kFloat, "dwp"); chk(dwo, at::kFloat, "dwo"); chk(wb, at::kBFloat16, "wb"); chk(lpw, at::kFloat, "lpw");
  const Geo G = geo(wb.size(0), dwp.size(1));
  const int64_t W = width(G), D = G.d;
  chk_acc(acc, G);
  TORCH_CHECK(dwp.numel() == 4 * W * D && dwo.numel() == D * W, "finalize: dwp [4 W, D], dwo [D, W]");
  TORCH_CHECK(out.size() == 11, "finalize: 11 outputs");
  for (int i = 0; i < 11; ++i) chk(out[i], at::kFloat, "finalize output");
  const at::cuda::CUDAGuard gd(dwp.device());
  const long n = 5L * D * D / 4 + 3 * D + G.nh * DP + 2 * DP;
  // out: lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, lnpb, wb (the leaves' order)
  finalize_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<float>(dwp), CP<float>(dwo), CP<float>(acc), CP<bf>(wb), CP<float>(lpw),
      P<float>(out[2]), P<float>(out[4]), P<float>(out[5]), P<float>(out[6]), P<float>(out[7]), P<float>(out[0]), P<float>(out[1]),
      P<float>(out[3]), P<float>(out[10]), P<float>(out[8]), P<float>(out[9]), G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_bias_fwd", &pair_bias_fwd);
  m.def("pair_bias_bwd", &pair_bias_bwd);
  m.def("ln_rows", &ln_rows);
  m.def("gate_rows", &gate_rows);
  m.def("gate_bwd", &gate_bwd);
  m.def("qkv_bwd", &qkv_bwd);
  m.def("ln_bwd", &ln_bwd);
  m.def("prep", &prep);
  m.def("finalize", &finalize);
}
