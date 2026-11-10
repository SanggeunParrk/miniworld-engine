// transition_lnsg_f32.cu -- the first half of the fp32 Transition forward at D >= 384 (the bf16 build's ln_swiglu_gemm.cu with fp32
// activations): h = silu(LN(x) Wa^T) * (LN(x) Wb^T) with the LayerNorm INSIDE the expand GEMM.
//
// A CTA owns a 128-row block (two warpgroups of 64 rows): the LayerNorm reads x straight from global memory (a warp per row, fp32
// statistics) and leaves xn resident in shared memory as fp16 (K-major, 128-byte swizzled), then every 64-hidden-unit N tile of the block
// runs against it: [a | b] = xn [Wa_j; Wb_j]^T as one m64n128 chain per warpgroup over K = D, the weight K slabs ([Wa_j 64 rows; Wb_j 64
// rows] x 64 columns, 16 KB) streaming through an NSTAGE TMA ring refilled by thread 0; SwiGLU on the fp32 accumulators; h16 = h / s_h
// staged per warpgroup and TMA-stored. The training build also writes xn (TF32 value of the fp16 operand times s_x), rstd and c1.
// The squeeze (y = x + s_h s_ws (h16 Ws16^T)) is the generic build's squeeze_res_h16 (generic_sm90a), as in the large-row fp32 path.
//
// Numerics as transition_fwd_f32io.cu (fp16 operands at TF32's rounding, the device scales ``sc``).
#pragma once
#include "transition_fwd_f32io.cu"
#include "../../generic_sm90a/wgmma_ops.cuh"

template <int D, int NSTAGE>
struct LnsgF32Cfg {
  static constexpr int ROWS = 128, KB = D / 64, CPL = D / 32;
  static constexpr int F_XN = 0;                       // xn: [2 warpgroups][KB][64 rows][128 B] fp16
  static constexpr int F_SLAB = 2 * KB * 8192;         // ring: NSTAGE x [128 weight rows][128 B]
  static constexpr int F_STG = F_SLAB + NSTAGE * 16384;  // h16 staging: [2 warpgroups][64 rows][128 B]
  static constexpr int F_BAR = F_STG + 2 * 8192;
  static constexpr int SMEM = F_BAR + 256;
  static_assert(D % 64 == 0 && CPL % 8 == 0, "tile shape (a lane owns whole 16-byte fp16 chunks of xn)");
  static_assert(SMEM + 1024 <= 231424, "shared memory budget");
};

template <int D, int NSTAGE, bool SAVE>
__global__ void __launch_bounds__(256, 1)
transition_lnsg_f32(const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mh,
                    const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ sc,
                    float* __restrict__ xn_out, float* __restrict__ rstd_out, float* __restrict__ c1_out, int M, int H, float eps) {
  using C = LnsgF32Cfg<D, NSTAGE>;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, ncta = gridDim.x;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* empty = full + NSTAGE;
  const int blocks = (M + C::ROWS - 1) / C::ROWS, NT = H / 64;
  const int n_local = (blocks > cta) ? (blocks - cta + ncta - 1) / ncta : 0;
  const uint32_t smax = (uint32_t)n_local * NT * C::KB;  // slabs this CTA consumes
  auto issue = [&](uint32_t seq) {                      // slab seq: N tile (seq / KB) % NT, K block seq % KB
    const int s = (int)(seq % NSTAGE), kb = (int)(seq % C::KB), j = (int)((seq / C::KB) % NT);
    uint8_t* b = sm + C::F_SLAB + s * 16384;
    mbar_arrive_expect_tx(full + s, 16384);
    tma_load_2d(b, &mwa, full + s, kb * 64, j * 64);
    tma_load_2d(b + 8192, &mwb, full + s, kb * 64, j * 64);
  };
  if (tid == 0) {
    tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb); tma_prefetch_desc(&mh);
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) for (uint32_t s = 0; s < (uint32_t)NSTAGE && s < smax; ++s) issue(s);
  const float s_inx = __ldg(sc), s_ab = __ldg(sc + 1), s_inh = __ldg(sc + 2);
  const float s_x = 1.f / s_inx;
  const int c0 = C::CPL * lane;
  float g_[C::CPL], b_[C::CPL];
#pragma unroll
  for (int e = 0; e < C::CPL; e += 4) {
    const float4 g4 = __ldg(reinterpret_cast<const float4*>(gamma + c0 + e)), b4 = __ldg(reinterpret_cast<const float4*>(beta + c0 + e));
    g_[e] = g4.x; g_[e + 1] = g4.y; g_[e + 2] = g4.z; g_[e + 3] = g4.w;
    b_[e] = b4.x; b_[e + 1] = b4.y; b_[e + 2] = b4.z; b_[e + 3] = b4.w;
  }
  const uint32_t xs = su + C::F_XN + wg * C::KB * 8192;
  const uint32_t stg = su + C::F_STG + wg * 8192;
  const int q = lane & 3, rf = 16 * warp + (lane >> 2);
  uint32_t seq = 0;
  for (int i = 0; i < n_local; ++i) {
    const int trow = (cta + i * ncta) * C::ROWS + wg * 64;
    // ---------------------------------------------------------------- LayerNorm (fp32) -> resident fp16 xn, four rows of the warp at a time
#pragma unroll 1
    for (int rb = 0; rb < 16; rb += 4) {
      float v[4][C::CPL];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = trow + 16 * warp + rb + u;
#pragma unroll
        for (int e = 0; e < C::CPL; e += 4) {
          float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
          if (r < M) t = __ldg(reinterpret_cast<const float4*>(x + (size_t)r * D + c0 + e));
          v[u][e] = t.x; v[u][e + 1] = t.y; v[u][e + 2] = t.z; v[u][e + 3] = t.w;
        }
      }
      float s_[4], mean[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < C::CPL; ++e) a += v[u][e];
        s_[u] = a;
      }
#pragma unroll
      for (int k = 16; k; k >>= 1)
#pragma unroll
        for (int u = 0; u < 4; ++u) s_[u] += __shfl_xor_sync(0xffffffffu, s_[u], k);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        mean[u] = s_[u] * (1.f / D);
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < C::CPL; ++e) { const float d_ = v[u][e] - mean[u]; a += d_ * d_; }
        s_[u] = a;
      }
#pragma unroll
      for (int k = 16; k; k >>= 1)
#pragma unroll
        for (int u = 0; u < 4; ++u) s_[u] += __shfl_xor_sync(0xffffffffu, s_[u], k);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int rl = 16 * warp + rb + u, r = trow + rl;
        const float rs = rsqrtf(s_[u] * (1.f / D) + eps);
        uint32_t p[C::CPL / 2];
#pragma unroll
        for (int e = 0; e < C::CPL; e += 2) {
          p[e / 2] = pack_f16(((v[u][e] - mean[u]) * rs * g_[e] + b_[e]) * s_inx, ((v[u][e + 1] - mean[u]) * rs * g_[e + 1] + b_[e + 1]) * s_inx);
        }
        // lane's CPL columns: K block c0 / 64, bytes (c0 % 64) * 2 .. + 2 CPL of the 128-byte row
#pragma unroll
        for (int e = 0; e < C::CPL / 2; e += 4) {
          const uint32_t a16 = (uint32_t)((c0 & 63) * 2 + e * 4);   // the 16-byte chunk of this group inside the row
          const uint32_t adr = xs + (uint32_t)(((c0 + 2 * e) >> 6) * 8192) + swz128((uint32_t)rl, a16 & 127u);
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(adr), "r"(p[e]), "r"(p[e + 1]), "r"(p[e + 2]), "r"(p[e + 3]) : "memory");
        }
        if constexpr (SAVE) {
          if (r < M) {
#pragma unroll
            for (int e = 0; e < C::CPL; e += 4) {
              float o[4];
#pragma unroll
              for (int t = 0; t < 4; t += 2) {
                const __half2 hv = *reinterpret_cast<const __half2*>(&p[(e + t) / 2]);
                o[t] = f32io_tf32(__low2float(hv) * s_x); o[t + 1] = f32io_tf32(__high2float(hv) * s_x);
              }
              *reinterpret_cast<float4*>(xn_out + (size_t)r * D + c0 + e) = make_float4(o[0], o[1], o[2], o[3]);
            }
            if (lane == 0) { rstd_out[r] = rs; c1_out[r] = mean[u] * rs; }
          }
        }
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    // ---------------------------------------------------------------- the N tiles: 64 hidden units of a and b each
#pragma unroll 1
    for (int j = 0; j < NT; ++j) {
      float acc[64];
#pragma unroll
      for (int e = 0; e < 64; ++e) acc[e] = 0.f;
      int prev = -1;
#pragma unroll 1
      for (int kb = 0; kb < C::KB; ++kb, ++seq) {
        const int s = (int)(seq % NSTAGE);
        mbar_wait(full + s, (seq / NSTAGE) & 1u);
        const uint32_t bs = su + C::F_SLAB + s * 16384;
        fence_regs(acc); wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          tr90::mma<__half, 128>(acc, smem_desc(xs + (uint32_t)(kb * 8192 + ks * 32), 16, 1024, 1), smem_desc(bs + (uint32_t)(ks * 32), 16, 1024, 1));
        wgmma_commit();
        if (prev >= 0) {
          wgmma_wait<1>(); fence_regs(acc);
          if (wtid == 0) mbar_arrive(empty + prev);
          if (tid == 0 && seq - 1 + NSTAGE < smax) { mbar_wait(empty + prev, ((seq - 1) / NSTAGE) & 1u); issue(seq - 1 + NSTAGE); }
        }
        prev = s;
      }
      wgmma_wait<0>(); fence_regs(acc);
      if (wtid == 0) mbar_arrive(empty + prev);
      if (tid == 0 && seq - 1 + NSTAGE < smax) { mbar_wait(empty + prev, ((seq - 1) / NSTAGE) & 1u); issue(seq - 1 + NSTAGE); }
      // SwiGLU -> h16 staging (the previous tile's store must have read it)
      if (wtid == 0) tma_store_wait_read<0>();
      named_bar_sync(1 + wg, 128);
#pragma unroll
      for (int g = 0; g < 8; ++g)
#pragma unroll
        for (int R = 0; R < 2; ++R) {
          const float a0 = acc[4 * g + 2 * R] * s_ab, a1 = acc[4 * g + 2 * R + 1] * s_ab;
          const float b0 = acc[4 * (g + 8) + 2 * R] * s_ab, b1 = acc[4 * (g + 8) + 2 * R + 1] * s_ab;
          sts32(stg + swz128((uint32_t)(rf + 8 * R), (uint32_t)((8 * g + 2 * q) * 2)),
                pack_f16(a0 * f32io_sigmoid(a0) * b0 * s_inh, a1 * f32io_sigmoid(a1) * b1 * s_inh));
        }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (wtid == 0) { f32io_store_2d(&mh, sm + C::F_STG + wg * 8192, j * 64, trow); tma_store_commit(); }
    }
    named_bar_sync(1 + wg, 128);                          // the last N tile's reads of xn retired before the next LayerNorm
  }
  if (wtid == 0) tma_store_wait_all();
}
