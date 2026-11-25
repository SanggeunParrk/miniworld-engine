// transition_fwd_f32xs.cu -- the fp32 Transition forward at D = 256 (hidden n D) as ONE fused sm_90a kernel, inference and the
// training forward that saves what the backward reads (xn, rstd, c1):
//
//   y = x + Ws (silu(Wa LN(x)) * (Wb LN(x)))        x, y fp32 [M][D]; LayerNorm statistics and normalisation in fp32
//
// The D = 256 structure of the bf16 build (transition_fwd_d256.cu) with fp32 activations: a persistent CTA per 128-row tile, two
// warpgroups of 64 rows, the output accumulator of a warpgroup (64 x 256 fp32) held across every hidden chunk. The fp32 x tile (128 KB) does
// not fit next to the weight ring, so the LayerNorm reads x straight from global memory (a warp per row, coalesced) and leaves xn in
// shared memory as fp16 (the expand GEMM's K-major A operand); the residual re-reads x in the fragment layout at the epilogue (32-byte
// sectors per quad) and y is stored the same way. Per hidden chunk j of HS units:
//   [a|b] = xn [Wa_j; Wb_j]^T     SS m64n(2 HS)k16 over K = D
//   h_j   = fp16(silu(a) b)       straight into the A fragments of the squeeze
//   acc  += h_j Ws^T_j            RS m64n64k16 per 64-column atom
// The weight chunks stream through an NSLOT TMA ring refilled by thread 0 once both warpgroups retired a chunk (as transition_fwd_f32io.cu).
//
// Numerics are those of transition_fwd_f32io.cu: fp16 tensor cores with fp32 accumulation at TF32's rounding, the four device scales
// ``sc`` keeping every fp16 operand in range. The saved xn is the fp16 operand times s_x, rounded to nearest TF32 (exact when s_x = 1):
// the backward's TF32 products read the value the forward multiplied.
#include "transition_fwd_f32io.cu"
#include "../../generic_sm90a/wgmma_ops.cuh"      // tr90::mma<__half, N>: SS wgmma, both operands K-major

template <int D, int N, int HS, int NSLOT>
struct F32XsCfg {
  static constexpr int H = N * D, NCH = H / HS, ROWS = 128, KB = D / 64, NA = D / 64, CPL = D / 32;
  static constexpr int W1 = 2 * HS * 128;              // one K block of the packed [Wa_j; Wb_j] tile: 2 HS rows of 128 B
  static constexpr int F_WST = KB * W1;                // then Ws^T_j: NA atoms of [HS hidden][64 d] (MN-major B of the squeeze)
  static constexpr int F_SLOT = F_WST + NA * HS * 128;
  static constexpr int F_XN = NSLOT * F_SLOT;          // xn: [2 warpgroups][KB][64 rows][128 B] fp16
  static constexpr int F_BAR = F_XN + 2 * KB * 8192;
  static constexpr int SMEM = F_BAR + 256;
  static_assert(D % 64 == 0 && H % HS == 0 && (HS == 32 || HS == 64) && CPL % 4 == 0, "tile shape");
  static_assert(SMEM + 1024 <= 231424, "shared memory budget");
};

template <int D, int N, int HS, int NSLOT, bool SAVE>
__global__ void __launch_bounds__(256, 1)
transition_fwd_f32xs(const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mwst,
                     const float* __restrict__ x, float* __restrict__ out, const float* __restrict__ gamma, const float* __restrict__ beta,
                     const float* __restrict__ sc, float* __restrict__ xn_out, float* __restrict__ rstd_out, float* __restrict__ c1_out,
                     int M, int tiles, float eps) {
  using C = F32XsCfg<D, N, HS, NSLOT>;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, ncta = gridDim.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + NSLOT;
  const int n_local = (tiles > cta) ? (tiles - cta + ncta - 1) / ncta : 0;
  const uint32_t wmax = (uint32_t)n_local * C::NCH;
  auto issue_w = [&](uint32_t seq) {                    // chunk seq % NCH into slot seq % NSLOT
    const int s = (int)(seq % NSLOT), j = (int)(seq % C::NCH); uint8_t* b = sm + s * C::F_SLOT;
    mbar_arrive_expect_tx(w_full + s, C::F_SLOT);
#pragma unroll
    for (int kb = 0; kb < C::KB; ++kb) {
      tma_load_2d(b + kb * C::W1, &mwa, w_full + s, kb * 64, j * HS);
      tma_load_2d(b + kb * C::W1 + HS * 128, &mwb, w_full + s, kb * 64, j * HS);
    }
#pragma unroll
    for (int na = 0; na < C::NA; ++na) tma_load_2d(b + C::F_WST + na * HS * 128, &mwst, w_full + s, na * 64, j * HS);
  };
  if (tid == 0) {
    tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb); tma_prefetch_desc(&mwst);
    for (int s = 0; s < NSLOT; ++s) { mbar_init(w_full + s, 1); mbar_init(w_free + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) for (uint32_t s = 0; s < (uint32_t)NSLOT && s < wmax; ++s) issue_w(s);
  const float s_inx = __ldg(sc), s_ab = __ldg(sc + 1), s_inh = __ldg(sc + 2), s_out = __ldg(sc + 3);
  const float s_x = 1.f / s_inx;
  // LayerNorm: lane owns CPL consecutive columns c0 .. c0 + CPL - 1 of a row
  const int c0 = C::CPL * lane;
  float g_[C::CPL], b_[C::CPL];
#pragma unroll
  for (int e = 0; e < C::CPL; e += 4) {
    const float4 g4 = __ldg(reinterpret_cast<const float4*>(gamma + c0 + e)), b4 = __ldg(reinterpret_cast<const float4*>(beta + c0 + e));
    g_[e] = g4.x; g_[e + 1] = g4.y; g_[e + 2] = g4.z; g_[e + 3] = g4.w;
    b_[e] = b4.x; b_[e + 1] = b4.y; b_[e + 2] = b4.z; b_[e + 3] = b4.w;
  }
  const uint32_t xs = su + C::F_XN + wg * C::KB * 8192;  // this warpgroup's xn tile
  const int q = lane & 3, rf = 16 * warp + (lane >> 2);  // fragment rows rf, rf + 8 of the warpgroup's 64
  uint32_t wseq = 0;

  for (int i = 0; i < n_local; ++i) {
    const int trow = (cta + i * ncta) * C::ROWS + wg * 64;   // first row of this warpgroup
    // ---------------------------------------------------------------- LayerNorm (fp32), four rows of the warp at a time -> fp16 xn
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
      float s_[4];
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
      float mean[4];
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
        float xo[C::CPL];
#pragma unroll
        for (int e = 0; e < C::CPL; e += 2) {
          const float n0 = (v[u][e] - mean[u]) * rs * g_[e] + b_[e], n1 = (v[u][e + 1] - mean[u]) * rs * g_[e + 1] + b_[e + 1];
          p[e / 2] = pack_f16(n0 * s_inx, n1 * s_inx);
          if constexpr (SAVE) {
            const __half2 hv = *reinterpret_cast<const __half2*>(&p[e / 2]);
            xo[e] = f32io_tf32(__low2float(hv) * s_x); xo[e + 1] = f32io_tf32(__high2float(hv) * s_x);
          }
        }
        // xn16 (row rl, columns c0 ..): K block c0 / 64, byte (c0 % 64) * 2 of the 128-byte row; CPL * 2 bytes (8 or 16)
        const uint32_t ad = xs + (uint32_t)((c0 >> 6) * 8192) + swz128((uint32_t)rl, (uint32_t)((c0 & 63) * 2));
#pragma unroll
        for (int e = 0; e < C::CPL / 2; e += 4)
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ad + e * 4), "r"(p[e]), "r"(p[e + 1]), "r"(p[e + 2]), "r"(p[e + 3]) : "memory");
        if constexpr (SAVE) {
          if (r < M) {
#pragma unroll
            for (int e = 0; e < C::CPL; e += 4)
              *reinterpret_cast<float4*>(xn_out + (size_t)r * D + c0 + e) = make_float4(xo[e], xo[e + 1], xo[e + 2], xo[e + 3]);
            if (lane == 0) { rstd_out[r] = rs; c1_out[r] = mean[u] * rs; }
          }
        }
      }
    }
    fence_proxy_async();                                  // xn (generic stores) -> the wgmma operand reads
    named_bar_sync(1 + wg, 128);
    // ---------------------------------------------------------------- the hidden chunks
    float acc[C::NA][32];
#pragma unroll
    for (int na = 0; na < C::NA; ++na)
#pragma unroll
      for (int e = 0; e < 32; ++e) acc[na][e] = 0.f;
#pragma unroll 1
    for (int j = 0; j < C::NCH; ++j, ++wseq) {
      const int s = (int)(wseq % NSLOT);
      const uint32_t slot = su + s * C::F_SLOT;
      mbar_wait(w_full + s, (uint32_t)((wseq / NSLOT) & 1));
      float AB[HS];
#pragma unroll
      for (int e = 0; e < HS; ++e) AB[e] = 0.f;
      fence_regs(AB);
#pragma unroll
      for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        const uint64_t da = smem_desc(xs + (uint32_t)((ks >> 2) * 8192 + (ks & 3) * 32), 16, 1024, 1);
        const uint64_t db = smem_desc(slot + (uint32_t)((ks >> 2) * C::W1 + (ks & 3) * 32), 16, 1024, 1);
        if constexpr (HS == 64) tr90::mma<__half, 128>(AB, da, db);
        else tr90::mma<__half, 64>(AB, da, db);
      }
      wgmma_commit();
      wgmma_wait<1>();                                  // also retires the previous chunk's squeeze update
#pragma unroll
      for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
      if (wseq > 0) {
        const uint32_t ps = (wseq - 1) % NSLOT;
        if (wtid == 0) mbar_arrive(w_free + ps);
        if (tid == 0 && wseq - 1 + NSLOT < wmax) { mbar_wait(w_free + ps, ((wseq - 1) / NSLOT) & 1u); issue_w(wseq - 1 + NSLOT); }
      }
      wgmma_wait<0>(); fence_regs(AB);
      uint32_t fh[HS / 16][4];
#pragma unroll
      for (int g = 0; g < HS / 8; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r] * s_ab, a1 = AB[e0 + 2 * r + 1] * s_ab;
          const float b0 = AB[e0 + HS / 2 + 2 * r] * s_ab, b1 = AB[e0 + HS / 2 + 2 * r + 1] * s_ab;
          fh[ks][o + r] = pack_f16(a0 * f32io_sigmoid(a0) * b0 * s_inh, a1 * f32io_sigmoid(a1) * b1 * s_inh);
        }
      }
#pragma unroll
      for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < HS / 16; ++ks)
#pragma unroll
        for (int na = 0; na < C::NA; ++na)
          f16_rs_n64<1>(acc[na], fh[ks], smem_desc(slot + (uint32_t)(C::F_WST + na * HS * 128 + ks * 2048), HS * 128, 1024, 1), 1);
      wgmma_commit();
    }
    wgmma_wait<0>();
#pragma unroll
    for (int na = 0; na < C::NA; ++na) fence_regs(acc[na]);
    named_bar_sync(1 + wg, 128);                          // every warp's expand reads of xn retired before the next LayerNorm overwrites it
    // ---------------------------------------------------------------- y = x + acc s_out (fp32), fragment layout, 32-byte sectors per quad
#pragma unroll
    for (int R = 0; R < 2; ++R) {
      const int r = trow + rf + 8 * R;
      if (r < M) {
        const float* xr = x + (size_t)r * D;
        float* yr = out + (size_t)r * D;
#pragma unroll
        for (int na = 0; na < C::NA; ++na)
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            const int c = 64 * na + 8 * g + 2 * q;
            const float2 xv = ldg64f(xr + c);
            stg64f(yr + c, xv.x + acc[na][4 * g + 2 * R] * s_out, xv.y + acc[na][4 * g + 2 * R + 1] * s_out);
          }
      }
    }
  }
  // the last chunk's slot: arrive for symmetry is not needed (no refill follows)
}
