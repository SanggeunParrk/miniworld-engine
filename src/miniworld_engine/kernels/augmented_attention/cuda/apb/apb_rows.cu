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
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstdlib>
#include <map>
#include <memory>
#ifdef APB_H100_FP32_SHADOW
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#endif

#include "../../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

constexpr int DP = 128, RW = 4, NT = RW * 32;           // d_pair; row kernels: RW warps, a row each
// accumulator (fp32, zeroed per step): dlnw [d] | dlnb [d] | dbq [W] | dWf [NH, DP] | per-head dbias sums [NH]
__host__ __device__ constexpr int acc_n(int d, int nh, int w) { return 2 * d + w + nh * DP + nh; }

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
template <int DPW = DP>
__device__ __forceinline__ void gram_stats(const float (&g0)[4], const float (&g1)[4], const float (&cs)[4], float eps, float& mu0,
                                           float& rs0, float& mu1, float& rs1) {
  const int lane = threadIdx.x & 31, g = lane >> 2, src = 4 * g + (g >> 1);
  const float s0 = __shfl_sync(0xffffffffu, (g & 1) ? g0[1] : g0[0], src), s1 = __shfl_sync(0xffffffffu, (g & 1) ? g1[3] : g1[2], src);
  mu0 = cs[0] * (1.f / DPW); mu1 = cs[2] * (1.f / DPW);
  rs0 = rsqrtf(fmaxf(s0 * (1.f / DPW) - mu0 * mu0, 0.f) + eps);
  rs1 = rsqrtf(fmaxf(s1 * (1.f / DPW) - mu1 * mu1, 0.f) + eps);
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
constexpr int FW = 8, FNT = 32 * FW;   // warps, threads
// pair width DPW (128; 256 for 8 heads): tile bytes and ring depth (the wider tile takes one block per SM and three stages)
template <int DPW = DP> constexpr int fwd_stages() { return DPW > 128 ? 3 : FWD_STAGES; }
template <int DPW = DP> constexpr int fwd_tile() { return 128 * DPW * 2; }
template <int NH> constexpr int nhp() { return (NH + 7) / 8 * 8; }    // heads rounded up to the mma's 8
template <int NH, int DPW = DP> constexpr int fwd_smem() { return fwd_stages<DPW>() * fwd_tile<DPW>() + 2 * nhp<NH>() * (128 + 4) * 4 + fwd_stages<DPW>() * 8; }
// Wf's B fragments (lane (g, q): head g + 8 nt; zero past NH) and sum_c Wf of heads 2q, 2q + 1 (+ 8 nt) for this lane's C columns
template <int NT8, int NH, int DPW = DP>
__device__ __forceinline__ void wf_frags(const bf* W, uint32_t (&b)[NT8][DPW / 16][2], float (&sw0)[NT8], float (&sw1)[NT8]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    float sw = 0.f;
#pragma unroll
    for (int s = 0; s < DPW / 16; ++s) {
      const bool live = g + 8 * nt < NH;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + (live ? g + 8 * nt : 0) * DPW + 32 * (s >> 1) + 8 * q + 4 * (s & 1));
      b[nt][s][0] = live ? wp[0] : 0u; b[nt][s][1] = live ? wp[1] : 0u;
      const float2 u = unpack2(b[nt][s][0]), v = unpack2(b[nt][s][1]);
      sw += u.x + u.y + v.x + v.y;
    }
    sw = quad_sum(sw);                                                      // sum_c Wf[g + 8 nt, c]
    sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
  }
}

// ST (optional): the (mean, rstd) of every pair row, for a backward that does not recompute them
template <int NH, int DPW = DP>
__global__ void __launch_bounds__(FNT) pair_bias_fwd_k(const bf* __restrict__ Z, const bf* __restrict__ W, const bool* __restrict__ MASK,
    bf* __restrict__ OUT, float2* __restrict__ ST, int L, float eps, float neg) {
  constexpr int NT8 = nhp<NH>() / 8, NHP = nhp<NH>(), NSF = fwd_stages<DPW>(), FT = fwd_tile<DPW>(), UU = DPW / 32;
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
    mbar_expect_tx(fb, (uint32_t)rows * DPW * 2);
    bulk_g2s(smem + (n % NSF) * FT, Z + ((long)i * L + j0) * DPW, (uint32_t)rows * DPW * 2, fb);
  };
  if (threadIdx.x == 0) {
    for (int k = 0; k < NSF; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int k = 0; k < NSF; ++k) issue(k);
  uint32_t b[NT8][DPW / 16][2];
  float sw0[NT8], sw1[NT8];
  wf_frags<NT8, NH, DPW>(W, b, sw0, sw1);
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0), st = (int)(n % NSF);
    mbar_wait(&full[st], (uint32_t)((n / NSF) & 1));
    const uint8_t* tile = smem + st * FT;
    float (*os)[128 + 4] = out_s[n & 1];
    const int rb = warp * 16;
    if (rb < rows) {
      uint4 r0[UU], r1[UU];
#pragma unroll
      for (int u = 0; u < UU; ++u) {
        r0[u] = *reinterpret_cast<const uint4*>(tile + (rb + g) * DPW * 2 + 64 * u + 16 * q);
        r1[u] = *reinterpret_cast<const uint4*>(tile + (rb + g + 8) * DPW * 2 + 64 * u + 16 * q);
      }
      float c[NT8][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
#pragma unroll
      for (int u = 0; u < UU; ++u) {
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
      gram_stats<DPW>(g0, g1, cs, eps, m0, rs0, m1, rs1);
      const int jl = rb + g;
      if (ST != nullptr && q == 0) {
        ST[(long)i * L + j0 + jl] = make_float2(m0, rs0);
        ST[(long)i * L + j0 + jl + 8] = make_float2(m1, rs1);
      }
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
    const bf* __restrict__ W, bf* __restrict__ DZ, float* __restrict__ AWF, float* __restrict__ AHS, int L, float eps, int acc_stride) {
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
    float* dst = c < NH * DP ? AWF + c : AHS + (c - NH * DP);
    if (acc_stride) dst[(long)blockIdx.x * acc_stride] = v;
    else atomicAdd(dst, v);
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
template <int V> __device__ __forceinline__ void colsum_add(const float4 (&v)[V], float* red, float* out, int acc_stride) {
  constexpr int WD = 128 * V;
  const int warp = threadIdx.x >> 5, c = 4 * V * (threadIdx.x & 31);
#pragma unroll
  for (int k = 0; k < V; ++k) *reinterpret_cast<float4*>(red + warp * WD + c + 4 * k) = v[k];
  __syncthreads();
  for (int j = threadIdx.x; j < WD; j += NT) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < RW; ++w) t += red[w * WD + j];
    if (acc_stride) out[(long)blockIdx.x * acc_stride + j] = t;
    else atomicAdd(out + j, t);
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
    bf* __restrict__ DQKVG, float* __restrict__ ABQ, int M, int NCH, int acc_stride) {
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
    for (int k = 0; k < V; ++k) v[k] = rnd<bf>(v[k]);
    stv<V>(DQKVG + r * 4 * WD, v);
    float4 t[V];
    ldv<V>(DK + r * WD, t); stv<V>(DQKVG + r * 4 * WD + WD, t);
    ldv<V>(DV + r * WD, t); stv<V>(DQKVG + r * 4 * WD + 2 * WD, t);
  }
  colsum_add<V>(v, red, ABQ, acc_stride);
}

// x^ = LN(x); dx = dy + rstd (dxa w - mean(dxa w) - x^ mean(dxa w x^)); ALNW += dxa x^, ALNB += dxa
template <typename XT, typename OT, int V>
__global__ void __launch_bounds__(NT) ln_bwd_k(const bf* __restrict__ DXA, const XT* __restrict__ X, const float2* __restrict__ ST,
    const float* __restrict__ LW, const OT* __restrict__ DY, OT* __restrict__ DX, float* __restrict__ ALNW, float* __restrict__ ALNB, int M,
    int acc_stride) {
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
  colsum_add<V>(aw, red, ALNW, acc_stride);
  __syncthreads();
  colsum_add<V>(ab, red, ALNB, acc_stride);
}

// Head geometry: nh heads of dh real channels, dhp wide in memory (dhp = dh: no padding). A padded index p (within a W = nh dhp
// segment) is real channel (p / dhp) dh + p % dhp when p % dhp < dh, else a pad (zero weight / ignored gradient).
struct Geo { int nh, dh, dhp, d; };                                         // d = nh dh = d_single
__device__ __forceinline__ int real_of(const Geo& G, int p) { const int c = p % G.dhp; return c < G.dh ? (p / G.dhp) * G.dh + c : -1; }

// W q | k | v | g [4 W, D] (padded rows; q rows x qs), its bias [4 W] (bq x qs, then 0), Wo [D, W] (padded columns) or none,
// Wf = Wb * ln_pair.weight * ws [NH, 128], and Wb as bf16 [NH, 128] (the backward's dlnpw); bf16 out. 8 weights per thread.
// The weights are read in their own dtype T (bf16, or an fp32 master: rounded to bf16 first, exactly the cast the parameters would
// have been given, then scaled): the live master-weight cast is part of this pass, not a launch of its own.
template <typename T> __device__ __forceinline__ void load8(const T* p, float (&f)[8]);
template <> __device__ __forceinline__ void load8<bf>(const bf* p, float (&f)[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) { const float2 t = unpack2(w[k]); f[2 * k] = t.x; f[2 * k + 1] = t.y; }
}
template <> __device__ __forceinline__ void load8<float>(const float* p, float (&f)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint4 pack8(const float (&f)[8], float scale) {
  uint4 u;
  u.x = pack2(f[0] * scale, f[1] * scale); u.y = pack2(f[2] * scale, f[3] * scale);
  u.z = pack2(f[4] * scale, f[5] * scale); u.w = pack2(f[6] * scale, f[7] * scale);
  return u;
}
template <typename T>
__global__ void __launch_bounds__(256) prep_k(const T* __restrict__ WQ, const T* __restrict__ BQ, const T* __restrict__ WK,
    const T* __restrict__ WV, const T* __restrict__ WG, const T* __restrict__ WO, const T* __restrict__ WB, const float* __restrict__ LPW,
    bf* __restrict__ WP, bf* __restrict__ BV, bf* __restrict__ WOP, bf* __restrict__ WF, bf* __restrict__ WBO, Geo G, float qs, float ws) {
  const int W = G.nh * G.dhp, D = G.d;
  const long n1 = 4L * W * D / 8, n2 = WOP ? (long)D * W / 8 : 0, i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  if (i < n1) {                                                             // packed row p (of segment blk), 8 input columns
    const long e = i * 8, prow = e / D, col = e % D;
    const int blk = (int)(prow / W), rr = real_of(G, (int)(prow % W));
    uint4 u = zero;
    if (rr >= 0) {
      float f[8];
      load8<T>((blk == 0 ? WQ : blk == 1 ? WK : blk == 2 ? WV : WG) + (long)rr * D + col, f);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = rbf(f[k]);                         // the bf16 parameter, then its scale
      u = pack8(f, blk == 0 ? qs : 1.f);
    }
    *reinterpret_cast<uint4*>(WP + e) = u;
    return;
  }
  if (i < n1 + n2) {                                                        // Wo row r, 8 padded columns (dh % 8 == 0: one head)
    const long e = (i - n1) * 8, r = e / W, pc = e % W;
    const int rc = real_of(G, (int)pc);
    uint4 u = zero;
    if (rc >= 0) { float f[8]; load8<T>(WO + r * D + rc, f); u = pack8(f, 1.f); }
    *reinterpret_cast<uint4*>(WOP + e) = u;
    return;
  }
  const long j = i - n1 - n2;
  if (j < 4 * W) {
    const int rr = j < W ? real_of(G, (int)j) : -1;
    float v = 0.f;
    if (rr >= 0) { const T b = BQ[rr]; v = rbf(static_cast<float>(b)) * qs; }
    BV[j] = __float2bfloat16_rn(v);
  } else if (j < 4 * W + G.nh * DP) {
    const int k = (int)(j - 4 * W);
    const float wb = rbf(static_cast<float>(WB[k]));
    WF[k] = __float2bfloat16_rn(wb * LPW[k % DP] * ws);
    if (WBO) WBO[k] = __float2bfloat16_rn(wb);
  }
}

// Parameter gradients, each into its own fp32 tensor (the parameters may be an fp32 master; the integration casts to theirs):
// dWq | dWk | dWv | dWg (the real rows of dWp [4 W, D]), dWo (the real columns of dWo [D, W]), and from the accumulator dlnw, dlnb,
// dbq, dWb = dWf ln_pair.weight, dln_pair.weight = sum_h dWf Wb.
__global__ void __launch_bounds__(256) finalize_k(const float* __restrict__ DWP, const float* __restrict__ DWO, const float* __restrict__ ACC,
    const bf* __restrict__ WB, const float* __restrict__ LPW, float* __restrict__ OQ, float* __restrict__ OKe, float* __restrict__ OV,
    float* __restrict__ OG, float* __restrict__ OO, float* __restrict__ OLW, float* __restrict__ OLB, float* __restrict__ OBQ,
    float* __restrict__ OWB, float* __restrict__ OLPW, Geo G) {
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
  }
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
  TORCH_CHECK((acc.dim() == 1 && acc.numel() == n) || (acc.dim() == 2 && acc.size(1) == n), "acc: ", n, " floats per partial");
}
int nsm_of(const at::Tensor& t) {
  int nsm = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, t.get_device()));
  return nsm;
}

template <int NH, int DPW>
void pair_bias_fwd_t(const at::Tensor& z, const at::Tensor& w, const c10::optional<at::Tensor>& mask, at::Tensor& out, float2* st, int64_t L, double eps,
                     double neg) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_fwd_k<NH, DPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, fwd_smem<NH, DPW>()));
    attr = true;
  }
  const long ntiles = L * ((L + 127) / 128);
  const long per_sm = DPW > 128 ? 1 : 2;  // the 256-wide ring fills the SM's shared memory
  pair_bias_fwd_k<NH, DPW><<<(unsigned)std::min<long>(per_sm * nsm_of(z), ntiles), FNT, fwd_smem<NH, DPW>(), S()>>>(
      CP<bf>(z), CP<bf>(w), mask ? CP<bool>(*mask) : nullptr, P<bf>(out), st, (int)L, (float)eps, (float)neg);
}
// d_pair 128 (8 / 12 / 16 / 24 heads) or 256 (8 heads); ``stats`` (optional [L L, 2] fp32) receives every row's (mean, rstd)
void pair_bias_fwd_impl(at::Tensor z, at::Tensor w, c10::optional<at::Tensor> mask, at::Tensor out, c10::optional<at::Tensor> stats, int64_t L, double eps,
                        double neg) {
  chk(z, at::kBFloat16, "pair"); chk(w, at::kBFloat16, "wf"); chk(out, at::kBFloat16, "bias");
  const int64_t nh = w.size(0), dp = w.size(1);
  TORCH_CHECK(dp == DP || (dp == 2 * DP && (nh == 8 || nh == 16)), "pair_bias: d_pair 128, or 256 with 8 or 16 heads");
  TORCH_CHECK(L % 16 == 0 && z.numel() == L * L * dp && w.numel() == nh * dp && out.numel() == nh * L * L, "pair_bias_fwd shapes");
  TORCH_CHECK(nh == 8 || nh == 12 || nh == 16 || nh == 24, "pair_bias: 8, 12, 16 or 24 heads");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  float2* st = nullptr;
  if (stats) { chk(*stats, at::kFloat, "stats"); TORCH_CHECK(stats->numel() == 2 * L * L, "stats [L L, 2]"); st = reinterpret_cast<float2*>(stats->data_ptr()); }
  const at::cuda::CUDAGuard gd(z.device());
  if (dp == 2 * DP) { if (nh == 16) pair_bias_fwd_t<16, 2 * DP>(z, w, mask, out, st, L, eps, neg); else pair_bias_fwd_t<8, 2 * DP>(z, w, mask, out, st, L, eps, neg); }
  else if (nh == 8) pair_bias_fwd_t<8, DP>(z, w, mask, out, st, L, eps, neg);
  else if (nh == 12) pair_bias_fwd_t<12, DP>(z, w, mask, out, st, L, eps, neg);
  else if (nh == 16) pair_bias_fwd_t<16, DP>(z, w, mask, out, st, L, eps, neg);
  else pair_bias_fwd_t<24, DP>(z, w, mask, out, st, L, eps, neg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pair_bias_fwd(at::Tensor z, at::Tensor w, c10::optional<at::Tensor> mask, at::Tensor out, int64_t L, double eps, double neg) {
  pair_bias_fwd_impl(z, w, mask, out, c10::nullopt, L, eps, neg);
}
void pair_bias_fwd_stats(at::Tensor z, at::Tensor w, c10::optional<at::Tensor> mask, at::Tensor out, at::Tensor stats, int64_t L, double eps, double neg) {
  pair_bias_fwd_impl(z, w, mask, out, stats, L, eps, neg);
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
      mz, md, CP<bf>(w), P<bf>(dz), awf, awf + NH * DP, (int)L, (float)eps, acc.dim() == 2 ? (int)acc.size(1) : 0);
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

// Blocks of pair_bias_bwd's grid at L (the partial rows it writes): the Hopper accumulator's row count for its columns.
int64_t pair_bwd_blocks(at::Tensor like, int64_t nh, int64_t L) {
  const int bps = nh == 8 ? 2 : 1;
  return std::min<int64_t>((int64_t)bps * nsm_of(like), L * ((L + 127) / 128));
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
  const int stride = acc.dim() == 2 ? (int)acc.size(1) : 0;
  if (W == 384) qkv_bwd_k<3><<<blocks(M), NT, 0, S()>>>(CP<float>(dq), CP<float>(dk), CP<float>(dv), P<bf>(dqkvg), abq, (int)M, (int)NCH, stride);
  else qkv_bwd_k<4><<<blocks(M), NT, 0, S()>>>(CP<float>(dq), CP<float>(dk), CP<float>(dv), P<bf>(dqkvg), abq, (int)M, (int)NCH, stride);
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
                                   P<float>(acc) + D, (int)M, acc.dim() == 2 ? (int)acc.size(1) : 0);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Weights bf16 or fp32 (one dtype for the seven), outputs bf16. ``wbo`` (optional): Wb as bf16 for the backward.
void prep(at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor wv, at::Tensor wg, c10::optional<at::Tensor> wo, at::Tensor wb,
          at::Tensor lpw, at::Tensor wp, at::Tensor bv, c10::optional<at::Tensor> wop, at::Tensor wf, double qs, double ws,
          c10::optional<at::Tensor> wbo) {
  const auto dt = wq.scalar_type();
  TORCH_CHECK(dt == at::kBFloat16 || dt == at::kFloat, "prep: bf16 or fp32 weights");
  for (auto* t : {&bq, &wk, &wv, &wg, &wb}) chk(*t, dt, "prep: one weight dtype");
  for (auto* t : {&wp, &bv, &wf}) chk(*t, at::kBFloat16, "prep: bf16 outputs");
  chk(lpw, at::kFloat, "ln_pair.weight");
  const Geo G = geo(wb.size(0), wq.size(1));
  const int64_t W = width(G), D = G.d;
  TORCH_CHECK(wq.numel() == D * D && wp.numel() == 4 * W * D && bv.numel() == 4 * W && wf.numel() == G.nh * DP, "prep shapes");
  TORCH_CHECK(wo.has_value() == wop.has_value(), "prep: wo and wop together");
  if (wop) { chk(*wo, dt, "wo"); chk(*wop, at::kBFloat16, "wop"); TORCH_CHECK(wop->numel() == D * W, "wop [D, W]"); }
  if (wbo) { chk(*wbo, at::kBFloat16, "wbo"); TORCH_CHECK(wbo->numel() == G.nh * DP, "wbo [NH, 128]"); }
  const at::cuda::CUDAGuard gd(wq.device());
  const long n = 4L * W * D / 8 + (wop ? (long)D * W / 8 : 0) + 4 * W + G.nh * DP;
  const unsigned grid = (unsigned)((n + 255) / 256);
  if (dt == at::kBFloat16)
    prep_k<bf><<<grid, 256, 0, S()>>>(CP<bf>(wq), CP<bf>(bq), CP<bf>(wk), CP<bf>(wv), CP<bf>(wg), wo ? CP<bf>(*wo) : nullptr, CP<bf>(wb),
        CP<float>(lpw), P<bf>(wp), P<bf>(bv), wop ? P<bf>(*wop) : nullptr, P<bf>(wf), wbo ? P<bf>(*wbo) : nullptr, G, (float)qs, (float)ws);
  else
    prep_k<float><<<grid, 256, 0, S()>>>(CP<float>(wq), CP<float>(bq), CP<float>(wk), CP<float>(wv), CP<float>(wg),
        wo ? CP<float>(*wo) : nullptr, CP<float>(wb), CP<float>(lpw), P<bf>(wp), P<bf>(bv), wop ? P<bf>(*wop) : nullptr, P<bf>(wf),
        wbo ? P<bf>(*wbo) : nullptr, G, (float)qs, (float)ws);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fixed-order reduction of the block partials: column c sums rows 0 .. count(c) - 1, count = rows_a for the columns below split
// (dlnw | dlnb | dbq, written by the row kernels) and rows_b past it (dWf | head sums, written by pair_bias_bwd), so partial rows
// a producer grid never wrote are never read (no zero fill). A block is 32 columns x 16 warps; warp w adds rows w, w + 16, ...
// in increasing order and warp 0 adds the sixteen sums in warp order: deterministic, and 17 loads deep instead of 264.
constexpr int RED_WARPS = 16;
__global__ void __launch_bounds__(32 * RED_WARPS) acc_reduce_k(const float* parts, float* out, int width, int split, int rows_a, int rows_b) {
  __shared__ float sums[RED_WARPS][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = blockIdx.x * 32 + lane;
  const int count = c < split ? rows_a : rows_b;
  float value = 0.f;
  if (c < width) {
#pragma unroll 4
    for (int row = warp; row < count; row += RED_WARPS) value += parts[(long)row * width + c];
  }
  sums[warp][lane] = value;
  __syncthreads();
  if (warp == 0 && c < width) {
    value = sums[0][lane];
#pragma unroll
    for (int w = 1; w < RED_WARPS; ++w) value += sums[w][lane];
    out[c] = value;
  }
}
__global__ void zero_acc_k(float* out, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 0.f;
}
void zero_acc(at::Tensor acc) {
  chk(acc, at::kFloat, "acc");
  const at::cuda::CUDAGuard gd(acc.device());
  zero_acc_k<<<(unsigned)((acc.numel() + 255) / 256), 256, 0, S()>>>(P<float>(acc), acc.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// ``rows_a`` / ``rows_b`` (Hopper block partials only): the partial rows the row kernels (ln_bwd, qkv_bwd) and pair_bias_bwd wrote.
void finalize(at::Tensor dwp, at::Tensor dwo, at::Tensor acc, at::Tensor wb, at::Tensor lpw, std::vector<at::Tensor> out, int64_t rows_a,
              int64_t rows_b) {
  chk(dwp, at::kFloat, "dwp"); chk(dwo, at::kFloat, "dwo"); chk(wb, at::kBFloat16, "wb"); chk(lpw, at::kFloat, "lpw");
  const Geo G = geo(wb.size(0), dwp.size(1));
  const int64_t W = width(G), D = G.d;
  chk_acc(acc, G);
  TORCH_CHECK(dwp.numel() == 4 * W * D && dwo.numel() == D * W, "finalize: dwp [4 W, D], dwo [D, W]");
  TORCH_CHECK(out.size() == 10, "finalize: 10 outputs");
  for (int i = 0; i < 10; ++i) chk(out[i], at::kFloat, "finalize output");
  const at::cuda::CUDAGuard gd(dwp.device());
  if (acc.dim() == 2) {
    // Hopper: every producer wrote a block partial row; reduce them in a fixed order.
    TORCH_CHECK(rows_a >= 1 && rows_b >= 1 && rows_a <= acc.size(0) && rows_b <= acc.size(0), "finalize: partial row counts");
    auto total = at::empty({acc.size(1)}, acc.options());
    acc_reduce_k<<<(unsigned)((acc.size(1) + 31) / 32), 32 * RED_WARPS, 0, S()>>>(CP<float>(acc), P<float>(total), (int)acc.size(1),
        (int)(2 * D + W), (int)rows_a, (int)rows_b);
    acc = total;
  }
  const long n = 5L * D * D / 4 + 3 * D + G.nh * DP + DP;
  // out: lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, wb (the leaves' order)
  finalize_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<float>(dwp), CP<float>(dwo), CP<float>(acc), CP<bf>(wb), CP<float>(lpw),
      P<float>(out[2]), P<float>(out[4]), P<float>(out[5]), P<float>(out[6]), P<float>(out[7]), P<float>(out[0]), P<float>(out[1]),
      P<float>(out[3]), P<float>(out[9]), P<float>(out[8]), G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Accuracy-first FP32 shadow of Hopper APB's exact pair-weight (dWb) gradient (ln_pair has no beta).
// Fixed arithmetic/reduction order; no ATen compute or reference calls. The
// materialized FP32 buffers intentionally trade speed for exact oracle rounding.
// Arithmetic order audited against PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb.
namespace apb_exact {
// cuBLAS capture nodes keep a pointer to scratch. Give the APB shadow graph
// its own allocation so destroying another graph cannot invalidate that pointer.
struct OwnedSgemmHandle {
 cublasHandle_t value;
 OwnedSgemmHandle(){TORCH_CHECK(cublasCreate(&value)==CUBLAS_STATUS_SUCCESS,"APB owned SGEMM handle");}
 ~OwnedSgemmHandle(){cublasDestroy(value);}
};
torch::Tensor exact_mm_impl(torch::Tensor a,torch::Tensor b,c10::optional<torch::Tensor> bias,bool graph_owned) {
 c10::cuda::CUDAGuard device(a.device());
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()&&a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat,"exact cuBLAS FP32 operands");
 TORCH_CHECK(a.dim()==b.dim()&&a.dim()>=2&&a.dim()<=4&&a.size(-1)==b.size(-2),"exact cuBLAS matrix geometry");
 int M=a.size(-2),K=a.size(-1),N=b.size(-1);
 TORCH_CHECK((a.stride(-1)==1||a.stride(-2)==1)&&(b.stride(-1)==1||b.stride(-2)==1),"exact cuBLAS dense matrix views");
 if(a.dim()==4)TORCH_CHECK(a.size(0)==1&&b.size(0)==1,"exact cuBLAS leading batch must be one");
 int batches=a.dim()>2?a.size(-3):1;
 if(a.dim()>2)TORCH_CHECK(b.size(-3)==batches,"exact cuBLAS batch mismatch");
 auto shape=a.sizes().vec();shape.back()=N;auto y=torch::empty(shape,a.options());
 auto op_a=a.stride(-1)==1?CUBLAS_OP_N:CUBLAS_OP_T;
 auto op_b=b.stride(-1)==1?CUBLAS_OP_N:CUBLAS_OP_T;
 int lda=a.stride(-1)==1?a.stride(-2):a.stride(-1);
 int ldb=b.stride(-1)==1?b.stride(-2):b.stride(-1);
 float one=1.f,zero=0.f;
 auto checked=[](cublasStatus_t status){TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"exact cuBLAS failure: ",int(status));};
 if(!bias){
  // Thread-local guard configures this thread's cuBLAS handle, without changing
  // the caller's global TF32 policy or another module's precision.
  at::NoTF32Guard ieee;
  auto handle=at::cuda::getCurrentCUDABlasHandle();
  torch::Tensor owned_workspace;
  if(graph_owned){
   // Copy the incumbent handle's precision and atomics policy under NoTF32Guard;
   // retain the exact SGEMM APIs, argument order, leading dimensions and strides.
   cublasMath_t math_mode;cublasAtomicsMode_t atomics_mode;
   checked(cublasGetMathMode(handle,&math_mode));
   checked(cublasGetAtomicsMode(handle,&atomics_mode));
   static thread_local std::map<int,std::unique_ptr<OwnedSgemmHandle>> handles;
   auto& owned=handles[a.get_device()];if(!owned)owned=std::make_unique<OwnedSgemmHandle>();
   handle=owned->value;
   checked(cublasSetStream(handle,at::cuda::getCurrentCUDAStream()));
   checked(cublasSetMathMode(handle,math_mode));
   checked(cublasSetAtomicsMode(handle,atomics_mode));
   checked(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
   size_t bytes=at::cuda::getChosenWorkspaceSize();
   owned_workspace=torch::empty({int64_t(bytes)},a.options().dtype(at::kByte));
   checked(cublasSetWorkspace(handle,owned_workspace.data_ptr(),bytes));
  }
  if(a.dim()==2)checked(cublasSgemm(handle,op_b,op_a,N,M,K,&one,b.data_ptr<float>(),ldb,a.data_ptr<float>(),lda,&zero,y.data_ptr<float>(),N));
  else checked(cublasSgemmStridedBatched(handle,op_b,op_a,N,M,K,&one,b.data_ptr<float>(),ldb,b.stride(-3),a.data_ptr<float>(),lda,a.stride(-3),&zero,y.data_ptr<float>(),N,(long long)M*N,batches));
  return y;
 }
 TORCH_CHECK(a.dim()==2&&bias->is_cuda()&&bias->device()==a.device()&&bias->is_contiguous()&&bias->scalar_type()==at::kFloat&&bias->numel()==N,"exact cuBLAS bias");
 // cuBLASLt has an explicit compute type; its bias epilogue does not honor
 // NoTF32Guard. Match the FP32 reference's fused bias and heuristic policy.
 struct Descriptors{
  cublasLtMatmulDesc_t operation=nullptr;cublasLtMatrixLayout_t lhs=nullptr,rhs=nullptr,out=nullptr;cublasLtMatmulPreference_t preference=nullptr;
  ~Descriptors(){if(preference)cublasLtMatmulPreferenceDestroy(preference);if(out)cublasLtMatrixLayoutDestroy(out);if(rhs)cublasLtMatrixLayoutDestroy(rhs);if(lhs)cublasLtMatrixLayoutDestroy(lhs);if(operation)cublasLtMatmulDescDestroy(operation);}
 } d;
 checked(cublasLtMatmulDescCreate(&d.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
 checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSA,&op_b,sizeof(op_b)));
 checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSB,&op_a,sizeof(op_a)));
 cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;const float* bias_ptr=bias->data_ptr<float>();
 checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
 checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias_ptr,sizeof(bias_ptr)));
 checked(cublasLtMatrixLayoutCreate(&d.lhs,CUDA_R_32F,op_b==CUBLAS_OP_N?N:K,op_b==CUBLAS_OP_N?K:N,ldb));
 checked(cublasLtMatrixLayoutCreate(&d.rhs,CUDA_R_32F,op_a==CUBLAS_OP_N?K:M,op_a==CUBLAS_OP_N?M:K,lda));
 checked(cublasLtMatrixLayoutCreate(&d.out,CUDA_R_32F,N,M,N));
 checked(cublasLtMatmulPreferenceCreate(&d.preference));
 size_t workspace_bytes=at::cuda::getCUDABlasLtWorkspaceSize();void* workspace=at::cuda::getCUDABlasLtWorkspace();
 checked(cublasLtMatmulPreferenceSetAttribute(d.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
 const void* ptrs[]={b.data_ptr(),a.data_ptr(),y.data_ptr(),bias_ptr};
 cublasLtMatmulPreferenceAttributes_t attrs[]={CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
 for(int i=0;i<4;i++){uint32_t alignment=256;while(reinterpret_cast<uintptr_t>(ptrs[i])%alignment)alignment>>=1;checked(cublasLtMatmulPreferenceSetAttribute(d.preference,attrs[i],&alignment,sizeof(alignment)));}
 cublasLtMatmulHeuristicResult_t algo{};int found=0;auto handle=at::cuda::getCurrentCUDABlasLtHandle();
 checked(cublasLtMatmulAlgoGetHeuristic(handle,d.operation,d.lhs,d.rhs,d.out,d.out,d.preference,1,&algo,&found));
 TORCH_CHECK(found==1,"exact FP32 cuBLASLt bias geometry unavailable");
 checked(cublasLtMatmul(handle,d.operation,&one,b.data_ptr(),d.lhs,a.data_ptr(),d.rhs,&zero,y.data_ptr(),d.out,y.data_ptr(),d.out,&algo.algo,workspace,workspace_bytes,at::cuda::getCurrentCUDAStream()));
 return y;
}
torch::Tensor exact_mm(torch::Tensor a,torch::Tensor b,c10::optional<torch::Tensor> bias) {
 return exact_mm_impl(a,b,bias,false);
}
torch::Tensor exact_mm_owned(torch::Tensor a,torch::Tensor b,c10::optional<torch::Tensor> bias) {
 return exact_mm_impl(a,b,bias,true);
}
struct Stats { float mean, m2, count; };
__device__ Stats merge(Stats a,Stats b) {
 float n=a.count+b.count;
 if(n==0.f)return {0.f,0.f,0.f};
 float inv=1.f/n, delta=a.mean-b.mean, wa=a.count*inv, wb=b.count*inv;
 return {wb*b.mean+wa*a.mean,b.m2+a.m2+delta*delta*b.count*wa,n};
}
__global__ void ln_k(const float* x,const float* g,const float* b,float* y,int N,float eps) {
 __shared__ float s[12];int t=threadIdx.x,l=t&31,w=t>>5;
 Stats a{0.f,0.f,0.f};
 for(int base=4*t;base<N;base+=512) {
  #pragma unroll
  for(int j=0;j<4;j++){
   float v=x[(long)blockIdx.x*N+base+j], d=v-a.mean, n=a.count+1.f;
   float m=a.mean+d*(1.f/n);a={m,a.m2+d*(v-m),n};
  }
 }
 for(int o=16;o;o>>=1){
  Stats b{__shfl_down_sync(0xffffffff,a.mean,o),__shfl_down_sync(0xffffffff,a.m2,o),__shfl_down_sync(0xffffffff,a.count,o)};
  a=merge(a,b);
 }
 for(int o=2;o;o>>=1){
  if(l==0&&w>=o&&w<2*o){s[2*(w-o)]=a.mean;s[2*(w-o)+1]=a.m2;s[4+w-o]=a.count;}
  __syncthreads();
  if(l==0&&w<o)a=merge(a,{s[2*w],s[2*w+1],s[4+w]});
  __syncthreads();
 }
 if(t==0){s[0]=a.mean;s[1]=rsqrtf(a.m2/(float)N+eps);}
 __syncthreads();
 for(int c=t;c<N;c+=128)y[(long)blockIdx.x*N+c]=g[c]*(s[1]*(x[(long)blockIdx.x*N+c]-s[0]))+b[c];
}
torch::Tensor ln(torch::Tensor x,torch::Tensor g,torch::Tensor b,double eps){
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kFloat&&g.is_contiguous()&&b.is_contiguous()&&g.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat,"APB exact LN layout/dtype");
 int N=x.size(-1);TORCH_CHECK((N==64||N==128||N==384||N==512)&&g.numel()==N&&b.numel()==N,"APB exact LN width");auto y=torch::empty_like(x);int M=x.numel()/N;
 ln_k<<<M,128,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),g.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>(),N,eps);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
// BF16 load fusion for the accuracy shadow. FP32 output and operation order
// remain unchanged; this removes the FP32 input temporary, but deliberately
// keeps normalized pair/projection buffers (not yet the B200 fused lifetime).
__global__ void ln_bf16_k(const __nv_bfloat16* x,const float* g,float* y,int N,float eps) {
 __shared__ float s[12];int t=threadIdx.x,l=t&31,w=t>>5;
 Stats a{0.f,0.f,0.f};
 for(int base=4*t;base<N;base+=512) {
  #pragma unroll
  for(int j=0;j<4;j++){
   float v=__bfloat162float(x[(long)blockIdx.x*N+base+j]), d=v-a.mean, n=a.count+1.f;
   float m=a.mean+d*(1.f/n);a={m,a.m2+d*(v-m),n};
  }
 }
 for(int o=16;o;o>>=1){
  Stats b{__shfl_down_sync(0xffffffff,a.mean,o),__shfl_down_sync(0xffffffff,a.m2,o),__shfl_down_sync(0xffffffff,a.count,o)};
  a=merge(a,b);
 }
 for(int o=2;o;o>>=1){
  if(l==0&&w>=o&&w<2*o){s[2*(w-o)]=a.mean;s[2*(w-o)+1]=a.m2;s[4+w-o]=a.count;}
  __syncthreads();
  if(l==0&&w<o)a=merge(a,{s[2*w],s[2*w+1],s[4+w]});
  __syncthreads();
 }
 if(t==0){s[0]=a.mean;s[1]=rsqrtf(a.m2/(float)N+eps);}
 __syncthreads();
 for(int c=t;c<N;c+=128)y[(long)blockIdx.x*N+c]=g[c]*(s[1]*(__bfloat162float(x[(long)blockIdx.x*N+c])-s[0]));
}

__global__ void ln_bf16_warp_k(const __nv_bfloat16* x,const float* g,float* y,int N,int M,float eps) {
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
 if(row>=M)return;
 Stats a{0.f,0.f,0.f};
 // Width64/128 uses only warp0 in ln_k. Keep its four-value Welford
 // sequence, shuffled merges and the two subsequent empty-warp merges.
 for(int base=4*lane;base<N;base+=512) {
  #pragma unroll
  for(int j=0;j<4;j++){
   float v=__bfloat162float(x[(long)row*N+base+j]),d=v-a.mean,n=a.count+1.f;
   float m=a.mean+d*(1.f/n);a={m,a.m2+d*(v-m),n};
  }
 }
 for(int o=16;o;o>>=1){
  Stats z{__shfl_down_sync(0xffffffff,a.mean,o),__shfl_down_sync(0xffffffff,a.m2,o),__shfl_down_sync(0xffffffff,a.count,o)};
  a=merge(a,z);
 }
 if(lane==0){a=merge(a,{0.f,0.f,0.f});a=merge(a,{0.f,0.f,0.f});}
 float mean=__shfl_sync(0xffffffff,a.mean,0);
 float rstd=lane==0?rsqrtf(a.m2/(float)N+eps):0.f;
 rstd=__shfl_sync(0xffffffff,rstd,0);
 for(int c=lane;c<N;c+=32)y[(long)row*N+c]=g[c]*(rstd*(__bfloat162float(x[(long)row*N+c])-mean));
}
torch::Tensor ln_bf16(torch::Tensor x,torch::Tensor g,double eps,bool warp_rows) {
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_cuda()&&g.device()==x.device()&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16&&g.is_contiguous()&&g.scalar_type()==at::kFloat,"APB exact BF16 LN layout/dtype");
 int N=x.size(-1);TORCH_CHECK((N==64||N==128)&&g.numel()==N,"APB exact BF16 LN width");
 auto y=torch::empty(x.sizes(),x.options().dtype(at::kFloat));int M=x.numel()/N;
 if(warp_rows)ln_bf16_warp_k<<<(M+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),g.data_ptr<float>(),y.data_ptr<float>(),N,M,eps);
 else ln_bf16_k<<<M,128,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),g.data_ptr<float>(),y.data_ptr<float>(),N,eps);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// Fuse the exact pair LayerNorm, FP32 projection and optional mask/layout
// producer. Preserve Welford merges, affine order and sequential K128 FMA;
// normalized pair and (for masked attention) row-major bias never reach HBM.
template<int H,int ROWS,int THREADS,bool HEAD_MAJOR>
__global__ void ln_pair_project_k(const __nv_bfloat16* x,const float* g,const float* w,float* y,const bool* mask,int M,int L,float eps){
 __shared__ float norm[ROWS][129],weight[H][129],projected[H][ROWS+1];
 int lane=threadIdx.x&31,warp=threadIdx.x/32;
 for(int i=threadIdx.x;i<H*128;i+=THREADS)weight[i/128][i%128]=w[i];
 for(int rr=warp;rr<ROWS;rr+=THREADS/32){
  int row=blockIdx.x*ROWS+rr;Stats a{0.f,0.f,0.f};
  #pragma unroll
  for(int j=0;j<4;j++){
   float v=__bfloat162float(x[(long)row*128+4*lane+j]),d=v-a.mean,n=a.count+1.f;
   float m=a.mean+d*(1.f/n);a={m,a.m2+d*(v-m),n};
  }
  for(int o=16;o;o>>=1){
   Stats z{__shfl_down_sync(0xffffffff,a.mean,o),__shfl_down_sync(0xffffffff,a.m2,o),__shfl_down_sync(0xffffffff,a.count,o)};a=merge(a,z);
  }
  if(lane==0){a=merge(a,{0.f,0.f,0.f});a=merge(a,{0.f,0.f,0.f});}
  float mean=__shfl_sync(0xffffffff,a.mean,0);
  float rstd=lane==0?rsqrtf(a.m2/128.f+eps):0.f;rstd=__shfl_sync(0xffffffff,rstd,0);
  #pragma unroll
  for(int c=lane;c<128;c+=32)norm[rr][c]=g[c]*(rstd*(__bfloat162float(x[(long)row*128+c])-mean));
 }
 __syncthreads();
 constexpr int PER=ROWS*H/THREADS;float acc[PER]={};
 const int head=threadIdx.x%H,base=threadIdx.x/H,step=THREADS/H;
 #pragma unroll
 for(int k=0;k<128;k++){
  float wk=weight[head][k];
  #pragma unroll
  for(int p=0;p<PER;p++)acc[p]=__fmaf_rn(norm[base+p*step][k],wk,acc[p]);
 }
 #pragma unroll
 for(int p=0;p<PER;p++){
  if constexpr(HEAD_MAJOR)projected[head][base+p*step]=acc[p];
  else y[((long)blockIdx.x*ROWS+base+p*step)*H+head]=acc[p];
 }
 if constexpr(HEAD_MAJOR){
  __syncthreads();
  for(int i=threadIdx.x;i<H*ROWS;i+=THREADS){int h=i/ROWS,r=i%ROWS,row=blockIdx.x*ROWS+r;
   y[(long)h*M+row]=mask[row%L]?projected[h][r]:-3.4028234663852886e38f;
  }
 }
}
torch::Tensor ln_pair_project(torch::Tensor x,torch::Tensor g,torch::Tensor w,c10::optional<torch::Tensor> mask,double eps){
 TORCH_CHECK(x.is_cuda()&&g.device()==x.device()&&w.device()==x.device()&&w.dim()==2,"APB exact LN pair projection CUDA/device/matrix layout");
 c10::cuda::CUDAGuard guard(x.device());int M=x.numel()/128,H=w.size(0),L=mask?mask->numel():0;
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16&&x.size(-1)==128&&(M==384*384||M==768*768)
 &&g.is_contiguous()&&w.is_contiguous()&&g.scalar_type()==at::kFloat&&w.scalar_type()==at::kFloat
 &&g.numel()==128&&w.size(1)==128,"LN pair projection layout");
 TORCH_CHECK(!mask||(mask->is_cuda()&&mask->device()==x.device()&&mask->scalar_type()==at::kBool&&mask->is_contiguous()&&L*L==M),"mask");
 auto y=torch::empty(mask?std::vector<int64_t>{H,L,L}:std::vector<int64_t>{M,H},x.options().dtype(at::kFloat));auto st=at::cuda::getCurrentCUDAStream();
 #define LC(HH,R,T) case HH:if(mask)ln_pair_project_k<HH,R,T,true><<<M/R,T,0,st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),g.data_ptr<float>(),w.data_ptr<float>(),y.data_ptr<float>(),mask->data_ptr<bool>(),M,L,eps);else ln_pair_project_k<HH,R,T,false><<<M/R,T,0,st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),g.data_ptr<float>(),w.data_ptr<float>(),y.data_ptr<float>(),nullptr,M,0,eps);break;
 switch(H){LC(8,32,128) LC(12,32,192) LC(16,16,128) default:TORCH_CHECK(false,"H");}
 #undef LC
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

template<int LOG,bool BACK> __global__ void soft_k(const float* x,const float* p,float* y,int L,int M) {
 constexpr int I=(1<<LOG)/32;int r=blockIdx.x*4+threadIdx.x/32,l=threadIdx.x&31;
 if(r>=M)return;
 float a[I],q[I],s=0.f,v=-INFINITY;
 #pragma unroll
 for(int j=0;j<I;j++){int c=l+32*j;a[j]=c<L?x[(long)r*L+c]:(BACK?0.f:-INFINITY);
 if constexpr(BACK){q[j]=c<L?p[(long)r*L+c]:0.f;a[j]=a[j]*q[j];}else{v=v>a[j]?v:a[j];}}
 if constexpr(!BACK)for(int o=16;o;o>>=1){float z=__shfl_xor_sync(0xffffffff,v,o);v=v>z?v:z;}
 #pragma unroll
 for(int j=0;j<I;j++){if constexpr(!BACK)a[j]=expf(a[j]-v);s+=a[j];}
 for(int o=16;o;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
 #pragma unroll
 for(int j=0;j<I;j++)if(l+32*j<L){if constexpr(BACK)y[(long)r*L+l+32*j]=a[j]-q[j]*s;else y[(long)r*L+l+32*j]=a[j]/s;}
}

template<int LOG> __global__ void soft_bwd_k(const float* x,const float* p,float* y,int L,int M){
 constexpr int N=(1<<LOG)/32;
 const int lane=threadIdx.x%32;
 const int row=blockIdx.x*4+threadIdx.x/32;
 if(row>=M)return;
 float vals[N],prob[N],sum=0.f;
 #pragma unroll
 for(int j=0;j<N;j++){
  int col=lane+j*32;
  prob[j]=col<L?p[(long)row*L+col]:0.f;
  vals[j]=col<L?__fmul_rn(x[(long)row*L+col],prob[j]):0.f;
  sum=__fadd_rn(sum,vals[j]);
 }
 #pragma unroll
 for(int delta=16;delta>0;delta/=2){
  sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,delta));
 }
 #pragma unroll
 for(int j=0;j<N;j++){
  int col=lane+j*32;
  if(col<L)y[(long)row*L+col]=fmaf(-prob[j],sum,vals[j]);
 }
}

torch::Tensor soft(torch::Tensor x,torch::Tensor p,bool back){
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_contiguous()&&p.is_contiguous()&&x.scalar_type()==at::kFloat&&p.scalar_type()==at::kFloat&&x.sizes()==p.sizes(),"APB exact softmax layout/dtype");
 auto y=torch::empty_like(x);int L=x.size(-1),M=x.numel()/L;
 TORCH_CHECK(L>=128&&L<=768&&L%128==0,"APB exact softmax bucket");
 auto stream=at::cuda::getCurrentCUDAStream();
 #define LAUNCH(LG,B) soft_k<LG,B><<<(M+3)/4,128,0,stream>>>(x.data_ptr<float>(),p.data_ptr<float>(),y.data_ptr<float>(),L,M)
 #define BWD(LG) soft_bwd_k<LG><<<(M+3)/4,128,0,stream>>>(x.data_ptr<float>(),p.data_ptr<float>(),y.data_ptr<float>(),L,M)
 if(back){if(L<=128){BWD(7);}else if(L<=256){BWD(8);}else if(L<=512){BWD(9);}else{BWD(10);}}
 #undef BWD
 else{if(L<=128){LAUNCH(7,false);}else if(L<=256){LAUNCH(8,false);}else if(L<=512){LAUNCH(9,false);}else{LAUNCH(10,false);}}
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void beta_small(const float* x,float* y,int M){
 __shared__ float s[32][33];int c=blockIdx.x*32+threadIdx.x,l=threadIdx.x,w=threadIdx.y;float v=0.f;
 for(int r=w*8;r<M;r+=256){
  #pragma unroll
  for(int j=0;j<8;j++)v+=x[(long)(r+j)*128+c];
 }
 s[w][l]=v;__syncthreads();v=s[l][w];
 for(int o=16;o;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o);
 if(l==0)y[blockIdx.x*32+w]=v;
}
__global__ void beta_parts(const float* x,float* y,int M,int P){
 int c=blockIdx.x*32+threadIdx.x;float v=0.f;
 for(int r=blockIdx.y*32;r<M;r+=32*P){
  #pragma unroll
  for(int j=0;j<32;j++)v+=x[(long)(r+j)*128+c];
 }
 y[(long)blockIdx.y*128+c]=v;
}
__global__ void beta_reduce_parts(const float* x,float* y,int P,int C){
 __shared__ float s[4][128];int lane=threadIdx.x,w=threadIdx.y;
 #pragma unroll
 for(int j=0;j<4;j++){
  int c=lane*4+j;float v[4]={0.f,0.f,0.f,0.f};int step=4*C;
  for(int r=w+blockIdx.x*4;r<P;r+=step*4){
   #pragma unroll
   for(int i=0;i<4;i++)if(r+i*step<P)v[i]+=x[(long)(r+i*step)*128+c];
  }
  float z=((v[0]+v[1])+v[2])+v[3];s[w][c]=z;
 }
 __syncthreads();
 for(int o=2;o;o>>=1){if(w<o){
  #pragma unroll
  for(int j=0;j<4;j++){int c=lane*4+j;s[w][c]+=s[w+o][c];}
 }__syncthreads();}
 if(w==0){
  #pragma unroll
  for(int j=0;j<4;j++){int c=lane*4+j;y[blockIdx.x*128+c]=s[0][c];}
 }
}
__global__ void beta_final(const float* x,float* y,int C){
 __shared__ float s[4][128];int l=threadIdx.x,w=threadIdx.y;
 #pragma unroll
 for(int j=0;j<4;j++){int c=l*4+j;float z=0.f;for(int r=w;r<C;r+=4)z+=x[r*128+c];s[w][c]=z;}
 __syncthreads();
 for(int o=2;o;o>>=1){if(w<o){
  #pragma unroll
  for(int j=0;j<4;j++){int c=l*4+j;s[w][c]+=s[w+o][c];}
 }__syncthreads();}
 if(w==0){
  #pragma unroll
  for(int j=0;j<4;j++){int c=l*4+j;y[c]=s[0][c];}
 }
}
torch::Tensor beta(torch::Tensor x){
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kFloat&&x.size(-1)==128&&x.numel()%16384==0,"APB exact beta layout");
 int M=x.numel()/128;auto y=torch::empty({128},x.options());auto stream=at::cuda::getCurrentCUDAStream();
 if(M<=65536)beta_small<<<4,dim3(32,32),0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),M);
 else{int P=std::min(8192,M/32),C=(P+63)/64;auto p=torch::empty({P,128},x.options()),q=torch::empty({C,128},x.options());
 beta_parts<<<dim3(4,P),32,0,stream>>>(x.data_ptr<float>(),p.data_ptr<float>(),M,P);
 beta_reduce_parts<<<C,dim3(32,4),0,stream>>>(p.data_ptr<float>(),q.data_ptr<float>(),P,C);
 beta_final<<<1,dim3(32,4),0,stream>>>(q.data_ptr<float>(),y.data_ptr<float>(),C);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// Fuse the final exact FP32 projection with beta_parts without changing either
// dot-product FMA order or the ordered row reduction. Four independent row dots
// share warp-broadcast input loads; the L*L*128 FP32 intermediate is never stored.
template<int H> __global__ void beta_mm_parts(const float* a,const float* b,float* y,int M,int P){
 int c=threadIdx.x,lane=threadIdx.x&31;float sum=0.f,w[H];
 #pragma unroll
 for(int k=0;k<H;++k)w[k]=b[k*128+c];
 for(int r=blockIdx.y*32;r<M;r+=32*P){
  #pragma unroll 1
  for(int j=0;j<32;j+=4){
   float v[4]={0.f,0.f,0.f,0.f},ax[4];
   #pragma unroll
   for(int n=0;n<4;++n)ax[n]=lane<H?a[(long)(r+j+n)*H+lane]:0.f;
   #pragma unroll
   for(int k=0;k<H;++k){
    #pragma unroll
    for(int n=0;n<4;++n)v[n]=__fmaf_rn(__shfl_sync(0xffffffffu,ax[n],k),w[k],v[n]);
   }
   #pragma unroll
   for(int n=0;n<4;++n)sum=__fadd_rn(sum,v[n]);
  }
 }
 y[(long)blockIdx.y*128+c]=sum;
}

torch::Tensor beta_mm(torch::Tensor a,torch::Tensor b){
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()
  &&a.dim()==2&&b.dim()==2&&a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat
  &&a.is_contiguous()&&b.is_contiguous(),"APB exact beta mm CUDA FP32 matrix layout");
 c10::cuda::CUDAGuard guard(a.device());
 int64_t M64=a.size(0),H64=a.size(1);
 TORCH_CHECK(M64>65536&&M64%128==0&&M64<=INT_MAX&&b.size(0)==H64&&b.size(1)==128
  &&(H64==8||H64==12||H64==16||H64==24),"APB exact beta mm required large-row/head layout");
 int M=M64,H=H64,P=std::min(8192,M/32),C=(P+63)/64;
 auto p=torch::empty({P,128},a.options()),q=torch::empty({C,128},a.options()),out=torch::empty({128},a.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 #define APB_BETA_MM_CASE(HH) case HH:beta_mm_parts<HH><<<dim3(1,P),128,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),p.data_ptr<float>(),M,P);break;
 switch(H){APB_BETA_MM_CASE(8) APB_BETA_MM_CASE(12) APB_BETA_MM_CASE(16) APB_BETA_MM_CASE(24)}
 #undef APB_BETA_MM_CASE
 beta_reduce_parts<<<C,dim3(32,4),0,stream>>>(p.data_ptr<float>(),q.data_ptr<float>(),P,C);
 beta_final<<<1,dim3(32,4),0,stream>>>(q.data_ptr<float>(),out.data_ptr<float>(),C);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void fp32_copy_k(const __nv_bfloat16* x,float* y,long n){
 long i=(long)blockIdx.x*256+threadIdx.x;if(i<n)y[i]=__bfloat162float(x[i]);
}
std::vector<torch::Tensor> fp32_inputs(std::vector<torch::Tensor> inputs){
 TORCH_CHECK(inputs.size()>0,"APB exact inputs empty");
 c10::cuda::CUDAGuard guard(inputs[0].device());std::vector<torch::Tensor> out;
 for(const auto& t:inputs){
  TORCH_CHECK(t.is_cuda()&&t.device()==inputs[0].device()&&t.is_contiguous(),"APB exact inputs must be contiguous on one CUDA device");
  TORCH_CHECK(t.scalar_type()==at::kFloat||t.scalar_type()==at::kBFloat16,"APB exact input dtype");
  if(t.scalar_type()==at::kFloat)out.push_back(t);
  else{
   auto y=torch::empty(t.sizes(),t.options().dtype(at::kFloat));
   fp32_copy_k<<<(t.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()),y.data_ptr<float>(),t.numel());out.push_back(y);
  }
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

struct FP32InputCasts {
 const __nv_bfloat16* src[16]; float* dst[16];
 long sizes[16]; int ends[16]; int count;
};
__global__ void fp32_inputs_grouped_k(FP32InputCasts args) {
 int begin=0;
 for(int j=0;j<args.count;++j) {
  if(blockIdx.x<args.ends[j]) {
   long i=(long)(blockIdx.x-begin)*256+threadIdx.x;
   if(i<args.sizes[j])args.dst[j][i]=__bfloat162float(args.src[j][i]);
   return;
  }
  begin=args.ends[j];
 }
}
std::vector<torch::Tensor> fp32_inputs_grouped(std::vector<torch::Tensor> inputs){
 TORCH_CHECK(!inputs.empty()&&inputs.size()<=16,"APB exact grouped inputs count");
 c10::cuda::CUDAGuard guard(inputs[0].device());std::vector<torch::Tensor> out;
 FP32InputCasts args{};int blocks=0;
 for(const auto& t:inputs) {
  TORCH_CHECK(t.is_cuda()&&t.device()==inputs[0].device()&&t.is_contiguous(),"APB exact grouped inputs layout");
  TORCH_CHECK(t.scalar_type()==at::kFloat||t.scalar_type()==at::kBFloat16,"APB exact grouped inputs dtype");
  if(t.scalar_type()==at::kFloat)out.push_back(t);
  else {
   auto y=torch::empty(t.sizes(),t.options().dtype(at::kFloat));
   int j=args.count++;args.src[j]=reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());args.dst[j]=y.data_ptr<float>();args.sizes[j]=t.numel();
   blocks+=(t.numel()+255)/256;args.ends[j]=blocks;out.push_back(y);
  }
 }
 if(blocks)fp32_inputs_grouped_k<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(args);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

template<int H,bool BACK>
__global__ void exact_transpose_tiled_k(const float* src,const bool* mask,float* dst,int L,int M) {
 __shared__ float tile[32][H+1];int t=threadIdx.x,base=blockIdx.x*32;
 if(BACK) {
  for(int h=t/32;h<H;h+=4) {int r=base+t%32;if(r<M)tile[t%32][h]=src[(long)h*M+r];}
 } else {
  for(int i=t;i<32*H;i+=128){int r=base+i/H;if(r<M)tile[i/H][i%H]=src[(long)r*H+i%H];}
 }
 __syncthreads();
 if(BACK) {
  for(int i=t;i<32*H;i+=128){int r=base+i/H;if(r<M)dst[(long)r*H+i%H]=(mask&&!mask[r%L])?0.f:tile[i/H][i%H];}
 } else {
  int r=base+t%32;bool masked=mask&&r<M&&!mask[r%L];
  for(int h=t/32;h<H;h+=4)if(r<M)dst[(long)h*M+r]=masked?-3.4028234663852886e38f:tile[t%32][h];
 }
}
torch::Tensor exact_transpose_tiled(torch::Tensor x,c10::optional<torch::Tensor> mask,int64_t L,bool back) {
 c10::cuda::CUDAGuard guard(x.device());int M=L*L,H=x.numel()/M;
 TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kFloat,"exact tiled float layout");
 if(mask)TORCH_CHECK(mask->is_contiguous()&&mask->is_cuda()&&mask->device()==x.device()&&mask->scalar_type()==at::kBool&&mask->numel()==L,"exact tiled mask");
 auto y=back?torch::empty({M,H},x.options()):torch::empty({H,L,L},x.options());
 const bool* mp=mask?mask->data_ptr<bool>():nullptr;auto stream=at::cuda::getCurrentCUDAStream();
 #define EXACT_TILE_CASE(N) case N: if(back)exact_transpose_tiled_k<N,true><<<(M+31)/32,128,0,stream>>>(x.data_ptr<float>(),mp,y.data_ptr<float>(),L,M);else exact_transpose_tiled_k<N,false><<<(M+31)/32,128,0,stream>>>(x.data_ptr<float>(),mp,y.data_ptr<float>(),L,M);break
 switch(H){EXACT_TILE_CASE(8);EXACT_TILE_CASE(12);EXACT_TILE_CASE(16);EXACT_TILE_CASE(24);default:TORCH_CHECK(false,"exact tiled heads");}
 #undef EXACT_TILE_CASE
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// PWA's exact sigmoid product feeds only the head-major contraction operand.
// Preserve the precise point_k arithmetic while eliminating its FP32 temporary.
__global__ void point_pack_k(const float* x,const float* g,float* y,long n,int S,int L){
 long i=(long)blockIdx.x*256+threadIdx.x;
 if(i<n){
  int c=i%32,s=(i/32)%S,q=(i/(32L*S))%L,h=i/(32L*S*L);
  long r=((long)s*L+q)*256+h*32+c;
  y[i]=x[r]*(1.f/(1.f+expf(-g[r])));
 }
}
torch::Tensor point_pack(torch::Tensor x,torch::Tensor g,int64_t S,int64_t L){
 TORCH_CHECK(x.is_cuda()&&g.is_cuda()&&x.device()==g.device()&&x.is_contiguous()&&g.is_contiguous()
  &&x.scalar_type()==at::kFloat&&g.scalar_type()==at::kFloat&&x.numel()==g.numel()&&x.numel()==S*L*256,"FP32 PWA point pack");
 c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty({8,L,S*32},x.options());
 point_pack_k<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),g.data_ptr<float>(),y.data_ptr<float>(),x.numel(),S,L);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

__global__ void point_k(const float* x,const float* z,float* y,long n,float scale,int op){
 long i=(long)blockIdx.x*256+threadIdx.x;
 if(i<n)y[i]=op==0?x[i]*scale:x[i]*(1.f/(1.f+expf(-z[i])));
}
torch::Tensor point(torch::Tensor x,torch::Tensor z,double scale,int op){
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kFloat,"APB exact point input");
 TORCH_CHECK(op==0||(z.is_contiguous()&&z.scalar_type()==at::kFloat&&z.numel()==x.numel()),"APB exact gate");
 auto y=torch::empty_like(x);
 point_k<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),z.data_ptr<float>(),y.data_ptr<float>(),x.numel(),scale,op);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void bias_add_k(float* s,const float* bias,const bool* mask,int L,int H){
 long i=(long)blockIdx.x*256+threadIdx.x,n=(long)H*L*L;
 if(i<n){int h=i/(L*L),r=i%(L*L);float b=mask&&!mask[r%L]?-3.4028234663852886e38f:bias[(long)r*H+h];s[i]+=b;}
}
void bias_add(torch::Tensor scores,torch::Tensor bias,c10::optional<torch::Tensor> mask){
 c10::cuda::CUDAGuard guard(scores.device());int L=scores.size(-1),H=scores.numel()/L/L;
 TORCH_CHECK(scores.is_contiguous()&&bias.is_contiguous()&&scores.scalar_type()==at::kFloat&&bias.scalar_type()==at::kFloat,"APB exact bias layout");
 if(mask)TORCH_CHECK(mask->is_contiguous()&&mask->scalar_type()==at::kBool&&mask->numel()==L,"APB exact mask");
 bias_add_k<<<(scores.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(scores.data_ptr<float>(),bias.data_ptr<float>(),mask?mask->data_ptr<bool>():nullptr,L,H);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void db_transpose_k(const float* x,float* y,const bool* mask,int L,int H){
 long i=(long)blockIdx.x*256+threadIdx.x,n=(long)L*L*H;
 if(i<n){int h=i%H,r=i/H;y[i]=mask&&!mask[r%L]?0.f:x[(long)h*L*L+r];}
}
torch::Tensor db_transpose(torch::Tensor x,c10::optional<torch::Tensor> mask){
 c10::cuda::CUDAGuard guard(x.device());int L=x.size(-1),H=x.numel()/L/L;auto y=torch::empty({L*L,H},x.options());
 TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kFloat,"APB exact dBias layout");
 if(mask)TORCH_CHECK(mask->is_contiguous()&&mask->scalar_type()==at::kBool&&mask->numel()==L,"APB exact dBias mask");
 db_transpose_k<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),mask?mask->data_ptr<bool>():nullptr,L,H);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

}

#ifdef APB_H100_FP32_SHADOW
// Diagnostic project-owned host wrappers instantiate installed BSD-licensed PyTorch
// memory-efficient CUTLASS templates (torch git cf30153c), using CUTLASS 4.2 headers.
// No ATen attention/softmax operator is called.
namespace apb_shadow_cutlass {
using F=PyTorchMemEffAttention::AttentionKernel<float,cutlass::arch::Sm80,true,64,64,64,true,true>;
template<int K> using B=PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80,float,true,false,false,64,64,K>;
template<typename K>
__global__ __launch_bounds__(K::kNumThreads,K::kMinBlocksPerSm)
void forward_entry(typename K::Params p) {
 if(!p.advance_to_block())return;
 // The installed template normalizes output using s_prime, then its LSE
 // epilogue mutates s_prime without a CTA barrier. Keep those statistics
 // read-only until every output warp has finished (racecheck 22545).
 auto* lse=p.logsumexp_ptr;p.logsumexp_ptr=nullptr;
 K::attention_kernel(p);
 __syncthreads();
 extern __shared__ char smem_buffer[];
 auto& shared=*reinterpret_cast<typename K::SharedStorage*>(smem_buffer);
 int t=threadIdx.x+blockDim.x*threadIdx.y;
 if(lse&&t<K::kQueriesPerBlock&&t<p.num_queries){
  float s=shared.s_prime[t],m=shared.mi[t];
  s=s==0.f?1.f:s;
  m=m==-cutlass::platform::numeric_limits<float>::infinity()?0.f:m;
  constexpr float log2e=1.4426950408889634074f;
  lse[t]=float(m/log2e)+cutlass::fast_log(float(s));
 }
}
template<typename K>
__global__ __launch_bounds__(K::kNumThreads,K::kMinBlocksPerSm)
void backward_entry(typename K::Params p) { if(p.advance_to_block())K::attention_kernel(p); }
// Exact FP32 dBias producer. Reuse the installed template's SIMT MMA and
// epilogue types, including their accumulation order, but never form discarded
// dQ/dK/dV or their cross-CTA workspaces. Each CTA owns one query/key tile.
template<int MaxK>
__global__ __launch_bounds__(B<MaxK>::kNumThreads,B<MaxK>::kMinBlocksPerSm)
void backward_bias_only_entry(typename B<MaxK>::Params p) {
 using Kernel=B<MaxK>;
 using QK=typename Kernel::MatmulQK;
 using DOV=typename Kernel::MatmulDOIVJ;
 extern __shared__ char storage[];
 auto& sm=*reinterpret_cast<typename Kernel::SharedStorage*>(storage);
 const int t=threadIdx.x,warp=t/32,lane=t%32;
 const int qi=blockIdx.x*64,kj=blockIdx.y*64,h=blockIdx.z;
 p.query_ptr+=h*p.q_strideH;p.key_ptr+=h*p.k_strideH;p.value_ptr+=h*p.v_strideH;
 p.grad_output_ptr+=h*p.gO_strideH;p.bias_ptr+=h*p.bias_strideH;
 p.logsumexp_ptr+=h*p.lse_strideH;p.delta_ptr+=h*p.delta_strideH;
 p.grad_bias_ptr+=h*p.gB_strideH;
 if(t<64)sm.di()[t]=p.delta_ptr[qi+t];
 __syncthreads();
 const cutlass::MatrixCoord origin{0,0};
 {
  using Mma=typename QK::Mma;
  typename Mma::IteratorA a({int32_t(p.k_strideM)},const_cast<float*>(p.key_ptr+kj*p.k_strideM),{64,p.head_dim},t,origin);
  typename Mma::IteratorB b({int32_t(p.q_strideM)},const_cast<float*>(p.query_ptr+qi*p.q_strideM),{p.head_dim,64},t,origin);
  Mma mma(sm.mm_qk_k(),sm.mm_qk_q(),t,warp,lane);
  typename Mma::FragmentC acc;acc.clear();
  mma.set_prologue_done(false);mma.set_zero_outside_bounds(true);
  mma((p.head_dim+Mma::Shape::kK-1)/Mma::Shape::kK,acc,a,b,acc);
  // The reference backward rounds QK*scale before adding bias.  The smaller
  // dBias-only producer otherwise lets NVCC contract these into an FMA;
  // that changes the tiny, cancellation-dominated pair-LN beta gradient.
  CUTLASS_PRAGMA_UNROLL
  for(int i=0;i<Mma::FragmentC::kElements;++i)acc[i]=__fmul_rn(p.scale,acc[i]);
  const int w=warp%(Mma::Base::WarpCount::kM*Mma::Base::WarpCount::kN);
  const cutlass::MatrixCoord tile{w%Mma::Base::WarpCount::kM,w/Mma::Base::WarpCount::kM};
  __syncthreads();
  typename QK::BiasLoader::GmemTileIterator gi({cutlass::layout::RowMajor(p.bias_strideM)},const_cast<float*>(p.bias_ptr+qi*p.bias_strideM+kj),{64,64},t);
  cutlass::TensorRef<float,cutlass::layout::RowMajor> br(sm.bias().data(),cutlass::layout::RowMajor(QK::ThreadblockShape::kM));
  typename QK::BiasLoader::SmemTileIterator si(br,t);
  QK::BiasLoader::load(gi,si);
  __syncthreads();
  auto offset=QK::AccumLambdaIterator::get_lane_offset(lane,warp,tile);
  QK::AccumLambdaIterator::iterateRows(offset,[&](int){},[&](int m,int n,int i){acc[i]+=br.at({n,m});},[&](int){});
  __syncthreads();
  QK::B2bGemm::accumApplyLSEToSmem(sm.attn_shared_storage(),acc,p.logsumexp_ptr+qi,64,t,warp,lane,tile);
 }
 __syncthreads();
 {
  using Mma=typename DOV::Mma;
  typename Mma::IteratorA a({int32_t(p.gO_strideM)},const_cast<float*>(p.grad_output_ptr+qi*p.gO_strideM),{64,p.head_dim_value},t,origin);
  typename Mma::IteratorB b({int32_t(p.v_strideM)},const_cast<float*>(p.value_ptr+kj*p.v_strideM),{p.head_dim_value,64},t,origin);
  Mma mma(sm.mm_doivj(),t,warp,lane);
  mma.set_prologue_done(false);mma.set_zero_outside_bounds(true);
  typename Mma::FragmentC acc;acc.clear();
  mma((p.head_dim_value+Mma::Shape::kK-1)/Mma::Shape::kK,acc,a,b,acc);
  __syncthreads();
  const int w=warp%(Mma::Base::WarpCount::kM*Mma::Base::WarpCount::kN);
  const cutlass::MatrixCoord tile{w%Mma::Base::WarpCount::kM,w/Mma::Base::WarpCount::kM};
  using It=typename DOV::AccumLambdaIterator;
  auto offset=It::get_lane_offset(lane,warp,tile);
  auto attn=sm.attn_shared_storage().accum_ref();float delta;
  It::iterateRows(offset,[&](int m){delta=sm.di()[m];},[&](int m,int n,int i){acc[i]=(acc[i]-delta)*attn.at({n,m});},[&](int){});
  __syncthreads();
  typename DOV::BiasGradEpilogue::OutputTileIterator out(typename DOV::BiasGradEpilogue::OutputTileIterator::Params{p.gB_strideM},p.grad_bias_ptr+qi*p.gB_strideM+kj,{64,64},t);
  typename DOV::BiasGradEpilogue::OutputOp op({1,1});
  typename DOV::BiasGradEpilogue ep(sm.gradB_epilogue(),t,warp,lane);
  ep(op,out,acc,out);
 }
}
template<int MaxK>
torch::Tensor bwd_bias_only_t(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor lse,torch::Tensor go,torch::Tensor delta) {
 using Kernel=B<MaxK>;typename Kernel::Params p;
 int L=q.size(0),H=q.size(1),D=q.size(2),W=H*D;
 auto gb=torch::empty_like(b);
 p.query_ptr=q.data_ptr<float>();p.key_ptr=k.data_ptr<float>();p.value_ptr=v.data_ptr<float>();p.bias_ptr=b.data_ptr<float>();
 p.logsumexp_ptr=lse.data_ptr<float>();p.grad_output_ptr=go.data_ptr<float>();p.delta_ptr=delta.data_ptr<float>();p.grad_bias_ptr=gb.data_ptr<float>();
 p.head_dim=D;p.head_dim_value=D;p.num_queries=L;p.num_keys=L;p.num_heads=H;p.num_batches=1;p.scale=static_cast<float>(1.0/std::sqrt(double(D)));
 p.q_strideM=p.k_strideM=p.v_strideM=p.gO_strideM=W;
 p.q_strideH=p.k_strideH=p.v_strideH=p.gO_strideH=D;
 p.bias_strideM=p.gB_strideM=L;p.bias_strideH=p.gB_strideH=L*L;p.lse_strideH=p.delta_strideH=L;
 constexpr int smem=sizeof(typename Kernel::SharedStorage);
 auto fn=backward_bias_only_entry<MaxK>;
 C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 void* args[]={&p};C10_CUDA_CHECK(cudaLaunchKernel((const void*)fn,dim3(L/64,L/64,H),dim3(Kernel::kNumThreads),args,smem,at::cuda::getCurrentCUDAStream()));
 return gb;
}
torch::Tensor bwd_bias_only(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor lse,torch::Tensor go,torch::Tensor delta) {
 c10::cuda::CUDAGuard guard(q.device());
 TORCH_CHECK(q.dim()==3&&q.size(0)%64==0,"APB exact dBias-only geometry");
 int L=q.size(0),H=q.size(1),D=q.size(2);
 TORCH_CHECK((D==16||D==24||D==32||D==48)&&k.sizes()==q.sizes()&&v.sizes()==q.sizes()&&go.sizes()==q.sizes()
   &&b.sizes()==torch::IntArrayRef({H,L,L})&&lse.sizes()==torch::IntArrayRef({H,L})&&delta.sizes()==lse.sizes(),"APB exact dBias-only shapes");
 for(const auto& t:{q,k,v,b,lse,go,delta})TORCH_CHECK(t.is_cuda()&&t.device()==q.device()&&t.is_contiguous()&&t.scalar_type()==at::kFloat,"APB exact dBias-only FP32 contiguous");
 return D<=32?bwd_bias_only_t<32>(q,k,v,b,lse,go,delta):bwd_bias_only_t<64>(q,k,v,b,lse,go,delta);
}
std::vector<torch::Tensor> fwd(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b) {
 c10::cuda::CUDAGuard guard(q.device());
 int L=q.size(0),H=q.size(1),D=q.size(2),W=H*D;
 TORCH_CHECK(q.dim()==3 && k.sizes()==q.sizes() && v.sizes()==q.sizes()
   && b.sizes()==torch::IntArrayRef({H,L,L}),"APB shadow shapes");
 for(const auto& t:{q,k,v,b})TORCH_CHECK(t.is_cuda()&&t.device()==q.device()&&t.is_contiguous()&&t.scalar_type()==at::kFloat,"APB shadow CUDA FP32 contiguous");
 TORCH_CHECK(q.scalar_type()==at::kFloat && L%64==0 && D<=64,"shadow FP32 geometry");
 auto o=torch::empty_like(q);auto lse=torch::empty({H,L},q.options());
 F::Params p;
 p.query_ptr=q.data_ptr<float>();p.key_ptr=k.data_ptr<float>();p.value_ptr=v.data_ptr<float>();
 p.attn_bias_ptr=b.data_ptr<float>();p.output_ptr=o.data_ptr<float>();p.logsumexp_ptr=lse.data_ptr<float>();
 p.head_dim=D;p.head_dim_value=D;p.num_queries=L;p.num_keys=L;p.num_keys_absolute=L;
 p.scale=static_cast<float>(1.0/std::sqrt(double(D)));p.num_heads=H;p.num_batches=1;
 p.q_strideM=p.k_strideM=p.v_strideM=p.o_strideM=W;
 p.q_strideH=p.k_strideH=p.v_strideH=D;
 p.q_strideB=p.k_strideB=p.v_strideB=L*W;
 p.bias_strideM=L;p.bias_strideH=L*L;p.bias_strideB=H*L*L;
 F::check_supported(p);
 constexpr int smem=sizeof(F::SharedStorage);
 auto fn=forward_entry<F>;
 C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 void* args[]={&p};
 C10_CUDA_CHECK(cudaLaunchKernel((const void*)fn,p.getBlocksGrid(),p.getThreadsGrid(),args,smem,at::cuda::getCurrentCUDAStream()));
 return {o,lse};
}
template<int K> torch::Tensor bwd_t(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor o,torch::Tensor lse,torch::Tensor go,torch::Tensor delta) {
 using Kernel=B<K>;typename Kernel::Params p;
 int L=q.size(0),H=q.size(1),D=q.size(2),W=H*D;
 auto gq=torch::empty_like(q),gk=torch::empty_like(k),gv=torch::empty_like(v),gb=torch::empty_like(b);
 p.query_ptr=q.data_ptr<float>();p.key_ptr=k.data_ptr<float>();p.value_ptr=v.data_ptr<float>();p.bias_ptr=b.data_ptr<float>();
 p.output_ptr=o.data_ptr<float>();p.logsumexp_ptr=lse.data_ptr<float>();p.grad_output_ptr=go.data_ptr<float>();p.delta_ptr=delta.data_ptr<float>();
 p.grad_query_ptr=gq.data_ptr<float>();p.grad_key_ptr=gk.data_ptr<float>();p.grad_value_ptr=gv.data_ptr<float>();p.grad_bias_ptr=gb.data_ptr<float>();
 p.head_dim=D;p.head_dim_value=D;p.num_queries=L;p.num_keys=L;p.num_heads=H;p.num_batches=1;p.scale=static_cast<float>(1.0/std::sqrt(double(D)));
 p.q_strideM=p.k_strideM=p.v_strideM=p.gO_strideM=W;
 p.q_strideH=p.k_strideH=p.v_strideH=p.o_strideH=p.gO_strideH=p.gQ_strideH=p.gK_strideH=p.gV_strideH=D;
 p.q_strideB=p.k_strideB=p.v_strideB=p.o_strideB=p.gO_strideB=p.gQ_strideB=p.gK_strideB=p.gV_strideB=L*W;
 p.bias_strideM=p.gB_strideM=L;p.bias_strideH=p.gB_strideH=L*L;p.bias_strideB=p.gB_strideB=H*L*L;
 p.lse_strideH=p.delta_strideH=L;p.lse_strideB=p.delta_strideB=H*L;
 // Each key tile writes disjoint dBias entries. Match the reference's key
 // parallelism; only its discarded dQ workspace uses cross-CTA accumulation.
 // The returned dBias and fixed-order beta reduction remain deterministic.
 p.num_splits_key=std::min((L+63)/64,std::max(1,200/H));
 auto ws=torch::empty({p.workspace_size()},q.options().dtype(at::kByte));
 if(ws.numel())p.workspace=reinterpret_cast<float*>(ws.data_ptr());
 if(p.should_zero_workspace())C10_CUDA_CHECK(cudaMemsetAsync(ws.data_ptr(),0,ws.numel(),at::cuda::getCurrentCUDAStream()));
 Kernel::check_supported(p);constexpr int smem=sizeof(typename Kernel::SharedStorage);
 auto fn=backward_entry<Kernel>;
 C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 void* args[]={&p};C10_CUDA_CHECK(cudaLaunchKernel((const void*)fn,p.getBlocksGrid(),p.getThreadsGrid(),args,smem,at::cuda::getCurrentCUDAStream()));
 return gb;
}
torch::Tensor bwd(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor o,torch::Tensor lse,torch::Tensor go,torch::Tensor delta) {
 c10::cuda::CUDAGuard guard(q.device());
 TORCH_CHECK(q.dim()==3&&q.size(0)%64==0,"APB shadow backward geometry");
 int L=q.size(0),H=q.size(1),D=q.size(2);
 TORCH_CHECK((D==16||D==24||D==32||D==48)&&k.sizes()==q.sizes()&&v.sizes()==q.sizes()&&o.sizes()==q.sizes()&&go.sizes()==q.sizes()
   &&b.sizes()==torch::IntArrayRef({H,L,L})&&lse.sizes()==torch::IntArrayRef({H,L})&&delta.sizes()==lse.sizes(),"APB shadow backward shapes");
 for(const auto& t:{q,k,v,b,o,lse,go,delta})TORCH_CHECK(t.is_cuda()&&t.device()==q.device()&&t.is_contiguous()&&t.scalar_type()==at::kFloat,"APB shadow backward FP32 contiguous");
 return q.size(2)<=32?bwd_t<32>(q,k,v,b,o,lse,go,delta):bwd_t<64>(q,k,v,b,o,lse,go,delta);
}

__global__ void delta_k(const float* go,const float* o,float* dst,int L,int H,int D) {
 int row=blockIdx.x*4+threadIdx.y,lane=threadIdx.x;
 if(row>=L*H)return;
 int bw=D<32?16:32;
 float a=0.f,b=0.f;
 if(lane<bw) {
  a=__fmul_rn(go[(long)row*D+lane],o[(long)row*D+lane]);
  if(lane+bw<D)b=__fmul_rn(go[(long)row*D+lane+bw],o[(long)row*D+lane+bw]);
 }
 float val=__fadd_rn(a,b);
 for(int off=bw/2;off;off/=2)val=__fadd_rn(val,__shfl_down_sync(0xffffffff,val,off));
 if(lane==0)dst[(row%H)*L+row/H]=val;
}
torch::Tensor delta(torch::Tensor go,torch::Tensor o) {
 c10::cuda::CUDAGuard guard(go.device());int L=go.size(0),H=go.size(1),D=go.size(2);
 TORCH_CHECK(D==16||D==24||D==32||D==48,"APB shadow delta geometry");
 auto y=torch::empty({H,L},go.options());
 delta_k<<<(L*H+3)/4,dim3(32,4),0,at::cuda::getCurrentCUDAStream()>>>(go.data_ptr<float>(),o.data_ptr<float>(),y.data_ptr<float>(),L,H,D);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

__global__ void mask_bias_k(const float* src,const bool* mask,float* dst,int L,int H) {
 long i=(long)blockIdx.x*256+threadIdx.x,n=(long)H*L*L;
 if(i<n){int h=i/(L*L),ij=i%(L*L);dst[i]=mask&&!mask[ij%L]?-3.4028234663852886e38f:src[(long)ij*H+h];}
}
torch::Tensor mask_bias(torch::Tensor bias,c10::optional<torch::Tensor> mask,int64_t L) {
 c10::cuda::CUDAGuard guard(bias.device());int H=bias.size(1);
 TORCH_CHECK(bias.is_contiguous()&&bias.scalar_type()==at::kFloat&&bias.size(0)==L*L,"APB shadow pair projection");
 if(mask)TORCH_CHECK(mask->is_contiguous()&&mask->scalar_type()==at::kBool&&mask->numel()==L,"APB shadow mask");
 auto y=torch::empty({H,L,L},bias.options());
 mask_bias_k<<<(y.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(bias.data_ptr<float>(),mask?mask->data_ptr<bool>():nullptr,y.data_ptr<float>(),L,H);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef APB_H100_FP32_SHADOW
  m.def("exact_attention_fwd", &apb_shadow_cutlass::fwd);
  m.def("exact_attention_db", &apb_shadow_cutlass::bwd);
  m.def("exact_attention_db_only", &apb_shadow_cutlass::bwd_bias_only);
  m.def("exact_delta", &apb_shadow_cutlass::delta);
  m.def("exact_mask_bias", &apb_shadow_cutlass::mask_bias);
#endif
  m.def("exact_ln", &apb_exact::ln);
  m.def("exact_ln_pair_project", &apb_exact::ln_pair_project);
  m.def("exact_ln_bf16", &apb_exact::ln_bf16, py::arg("x"), py::arg("g"), py::arg("eps"), py::arg("warp_rows") = false);
  m.def("exact_mm", &apb_exact::exact_mm, py::arg("a"), py::arg("b"), py::arg("bias") = py::none());
  m.def("exact_mm_owned", &apb_exact::exact_mm_owned, py::arg("a"), py::arg("b"), py::arg("bias") = py::none());
  m.def("exact_soft", &apb_exact::soft);
  m.def("exact_beta", &apb_exact::beta);
  m.def("exact_beta_mm", &apb_exact::beta_mm);
  m.def("exact_fp32_inputs", &apb_exact::fp32_inputs);
  m.def("exact_fp32_inputs_grouped", &apb_exact::fp32_inputs_grouped);
  m.def("exact_transpose_tiled", &apb_exact::exact_transpose_tiled);
  m.def("exact_point", &apb_exact::point);
  m.def("exact_point_pack", &apb_exact::point_pack);
  m.def("exact_bias_add", &apb_exact::bias_add);
  m.def("exact_db_transpose", &apb_exact::db_transpose);
  m.def("zero_acc", &zero_acc);
  m.def("pair_bwd_blocks", &pair_bwd_blocks);
  m.def("pair_bias_fwd", &pair_bias_fwd);
  m.def("pair_bias_fwd_stats", &pair_bias_fwd_stats);
  m.def("pair_bias_bwd", &pair_bias_bwd);
  m.def("ln_rows", &ln_rows);
  m.def("gate_rows", &gate_rows);
  m.def("gate_bwd", &gate_bwd);
  m.def("qkv_bwd", &qkv_bwd);
  m.def("ln_bwd", &ln_bwd);
  m.def("prep", &prep);
  m.def("finalize", &finalize);
}
