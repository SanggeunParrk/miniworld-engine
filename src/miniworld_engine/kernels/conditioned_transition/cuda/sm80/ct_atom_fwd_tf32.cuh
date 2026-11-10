// ct_atom_fwd_tf32.cuh -- the ConditionedTransition tail at the atom width (d_hidden = d_cond = 128, expansion 2: hidden 256), fp32 rows, TF32 tensor cores, A100 / sm_80: the fp32 twin of ``ct_atom_fwd.cuh``
// (inference), in TWO kernels because fp32 rows double every register the bf16 kernel keeps (the gate's cond rows and the residual rows do not fit next to the A fragments and the accumulators):
//
//   ct_tail_tf32_kernel   [a | b] = xa [Wa; Wb]^T      xa = the AdaLN's output (fp32); a, b in fp32 accumulators, never stored
//                         h = silu(a) b                 on the accumulators, rounded to TF32 as the operand of the squeeze
//                         z = h Ws^T                    fp32 accumulation -> z [M, 128] fp32
//   ct_gate_tf32_kernel   g = cond Wg^T + bg            a fourth product on the raw conditioning (cond row r % P)
//                         y = x + sigmoid(g) z          the residual (x = nullptr: the update alone)
//
// The pre-activation and the hidden h never leave the registers (the f1 order of adaln_tf32.cuh: the accumulators of one product are the A fragments of the next, see the m16n8k8 mapping below).  The weights
// stream through a three-stage ring of 32-hidden-unit chunks (Wa | Wb | Ws rows, 48 KB a stage, prefetched with ``cp.async`` two chunks ahead, one barrier per chunk) and are pre-packed on the host
// (``kernels/conditioned_transition/cuda/sm80.py``: TF32-rounded, the squeeze weight permuted).
//
// h as the A operand of the squeeze: the expand's accumulators of n tile T (8 hidden units 8 T + n, n = 2 q4 + e0) hold, for row g8, units 8 T + 2 q4 + {0, 1} in (c0, c1) and for row g8 + 8 in (c2, c3).
// k step (s, p) of the squeeze (s = 0, 1 within a chunk of 4 n tiles, p = 0, 1) takes a0 = h(g8; T0 = 2 s; c_p), a1 = h(g8 + 8; T0; c_{2 + p}), a2 = h(g8; T1 = 2 s + 1; c_p), a3 = h(g8 + 8; T1; c_{2 + p}):
// "k = q4" is hidden unit 8 T0 + 2 q4 + p and "k = q4 + 4" is unit 8 T1 + 2 q4 + p, so the squeeze weight's B fragment (b0 = Ws[o][8 T0 + 2 q4 + p], b1 = Ws[o][8 T1 + 2 q4 + p]) needs, per chunk and output row, the 8
// units {2 q4 + p, 8 + 2 q4 + p, 16 + 2 q4 + p, 24 + 2 q4 + p}: the host packs a chunk's 32 units of a row as [q4][idx], idx = 4 s + 2 sel + p (sel = 0: T0, 1: T1), two LDS.128 a row and chunk.
#pragma once
#include "adaln_tf32.cuh"

namespace adl {

struct CtTailTf32Params {
  const float* xa;               // [M][128]
  const float* wab;              // [512][128]  [Wa; Wb] rows, TF32-rounded
  const float* wsp;              // [128][256]  the squeeze weight packed for the kernel (rows in the f1 order, chunk blocks of [q4][idx]), TF32-rounded
  float* z;                      // [M][128]
  long M;
};

template <int NW_>
struct CtTailTf32Cfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = 1;
  static constexpr int HC = 32, NCH = 256 / HC, NST = 3;                // hidden units a chunk, chunks a tile, ring depth
  static constexpr int STAGE = 3 * HC * 512;                            // Wa | Wb | Ws rows of a chunk (HC rows of 512 B, and 128 rows of 128 B)
  static constexpr int SMEM = NST * STAGE;
};

// byte offset of 16-byte granule `g` (0 .. 7) of packed squeeze row `row` (rows of 128 B; the granule is XOR-ed with 1 on odd rows: the two rows a quarter warp reads fall on disjoint bank groups)
ADL_DEVI uint32_t tf_wsoff(uint32_t row, uint32_t g) { return row * 128u + ((g ^ (row & 1u)) << 4); }

template <class G>
ADL_DEVI void tail_tf32_load_chunk(const CtTailTf32Params& p, uint32_t stage, int c, int tid) {
  for (int i = tid; i < G::HC * 32; i += G::NTHR) {                      // Wa | Wb: HC rows of 32 granules
    const int r = i >> 5, ch = i & 31;
    cp_async16(stage + tf_woff(r, ch), p.wab + (size_t)(G::HC * c + r) * 128 + 4 * ch);
    cp_async16(stage + G::HC * 512 + tf_woff(r, ch), p.wab + (size_t)(256 + G::HC * c + r) * 128 + 4 * ch);
  }
  for (int i = tid; i < 128 * 8; i += G::NTHR) {                         // Ws: 128 rows of 8 granules (32 units)
    const int r = i >> 3, g = i & 7;
    cp_async16(stage + 2 * G::HC * 512 + tf_wsoff(r, g), p.wsp + (size_t)r * 256 + G::HC * c + 4 * g);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) ct_tail_tf32_kernel(const CtTailTf32Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  const long ntile = (p.M + 15) / 16;
  for (long base = (long)blockIdx.x * G::NW; base < ntile; base += (long)G::NW * gridDim.x) {
    const long tile = base + warp;
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = tile < ntile && rr[hh] < p.M; }
    float4 ux[2][4][2];                                                  // xa rows: the A fragments of the expand
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) ux[hh][gq][h] = rok[hh] ? ldg_f4(p.xa + rr[hh] * 128 + 32 * gq + 8 * q4 + 4 * h) : zero4();
    __syncthreads();                                                     // the previous tile group's reads of the ring are done
    tail_tf32_load_chunk<G>(p, sring, 0, tid);
    cp_async_commit();
    tail_tf32_load_chunk<G>(p, sring + G::STAGE, 1, tid);
    cp_async_commit();

    uint32_t a[16][4];                                                   // k step 4 gk + s: a0 / a1 = element s of rows g8 / g8 + 8, a2 / a3 = element s + 4
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float v[8] = {ux[hh][gq][0].x, ux[hh][gq][0].y, ux[hh][gq][0].z, ux[hh][gq][0].w, ux[hh][gq][1].x, ux[hh][gq][1].y, ux[hh][gq][1].z, ux[hh][gq][1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) { a[4 * gq + s][hh] = to_tf32(v[s]); a[4 * gq + s][2 + hh] = to_tf32(v[s + 4]); }
      }

    float dn[4][4][4];                                                   // z = h Ws^T: group og of 4 n tiles = channels 32 og + 8 q4 + 0 .. 7 of rows g8 / g8 + 8
#pragma unroll
    for (int og = 0; og < 4; ++og)
#pragma unroll
      for (int j = 0; j < 4; ++j) { dn[og][j][0] = dn[og][j][1] = dn[og][j][2] = dn[og][j][3] = 0.f; }

#pragma unroll 1
    for (int c = 0; c < G::NCH; ++c) {
      if (c + 1 < G::NCH) cp_async_wait<1>(); else cp_async_wait<0>();
      __syncthreads();                                                   // chunk c is in shared memory for every warp, and stage (c + 2) % 3 is free
      if (c + 2 < G::NCH) { tail_tf32_load_chunk<G>(p, sring + ((c + 2) % G::NST) * G::STAGE, c + 2, tid); cp_async_commit(); }
      const uint32_t stg = sring + (c % G::NST) * G::STAGE, wa_s = stg, wb_s = stg + G::HC * 512, ws_s = stg + 2 * G::HC * 512;

      float ac[4][4], bc[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) { ac[t][0] = ac[t][1] = ac[t][2] = ac[t][3] = 0.f; bc[t][0] = bc[t][1] = bc[t][2] = bc[t][3] = 0.f; }
#pragma unroll
      for (int gk = 0; gk < 4; ++gk)
#pragma unroll
        for (int sp = 0; sp < 2; ++sp) {                                 // k steps 2 sp, 2 sp + 1 of group gk: one LDS.128 a tile and matrix holds (b0, b1) of both (adaln_tf32.cuh)
          float4 ba[4], bb[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint32_t row = 8 * t + g8, cg = 8 * gk + 2 * q4 + sp;
            ba[t] = lds_f4(wa_s + tf_woff(row, cg));
            bb[t] = lds_f4(wb_s + tf_woff(row, cg));
          }
#pragma unroll
          for (int ss = 0; ss < 2; ++ss)                                 // the 8 accumulators are independent: a tile's next MMA is 8 MMAs away
#pragma unroll
            for (int t = 0; t < 4; ++t) {
              mma1688(ac[t], a[4 * gk + 2 * sp + ss], __float_as_uint(ss ? ba[t].z : ba[t].x), __float_as_uint(ss ? ba[t].w : ba[t].y));
              mma1688(bc[t], a[4 * gk + 2 * sp + ss], __float_as_uint(ss ? bb[t].z : bb[t].x), __float_as_uint(ss ? bb[t].w : bb[t].y));
            }
        }

      uint32_t hf[4][4];                                                 // k step kk = 2 s + p of the squeeze
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
          const int T0 = 2 * s, T1 = 2 * s + 1;
          float hv[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {                                  // e = 2 sel + hh: (T0 | T1) x (row g8 | g8 + 8)
            const int T = T0 + (e >> 1), hh = e & 1;
            const float av = ac[T][2 * hh + pp], bv = bc[T][2 * hh + pp];
            hv[e] = av * sigmoidf(av) * bv;
          }
          hf[2 * s + pp][0] = to_tf32(hv[0]);                            // (T0, row g8)
          hf[2 * s + pp][1] = to_tf32(hv[1]);                            // (T0, row g8 + 8)
          hf[2 * s + pp][2] = to_tf32(hv[2]);                            // (T1, row g8)
          hf[2 * s + pp][3] = to_tf32(hv[3]);                            // (T1, row g8 + 8)
        }

#pragma unroll
      for (int og = 0; og < 4; ++og) {
        float4 w0[4], w1[4];                                             // the 4 n tiles of output group og: steps (s = 0: p = 0, 1) in w0, (s = 1) in w1
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t row = (4 * og + j) * 8 + g8;
          w0[j] = lds_f4(ws_s + tf_wsoff(row, 2 * q4));
          w1[j] = lds_f4(ws_s + tf_wsoff(row, 2 * q4 + 1));
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)                                   // 4 independent accumulators a step
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float4 wv = (kk >> 1) ? w1[j] : w0[j];
            mma1688(dn[og][j], hf[kk], __float_as_uint((kk & 1) ? wv.y : wv.x), __float_as_uint((kk & 1) ? wv.w : wv.z));   // idx 2 p + 0 (T0) and 2 p + 2 (T1)
          }
      }
    }

#pragma unroll
    for (int og = 0; og < 4; ++og)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        if (rok[hh]) {
          stg_f4(p.z + rr[hh] * 128 + 32 * og + 8 * q4, make_float4(dn[og][0][2 * hh], dn[og][0][2 * hh + 1], dn[og][1][2 * hh], dn[og][1][2 * hh + 1]));
          stg_f4(p.z + rr[hh] * 128 + 32 * og + 8 * q4 + 4, make_float4(dn[og][2][2 * hh], dn[og][2][2 * hh + 1], dn[og][3][2 * hh], dn[og][3][2 * hh + 1]));
        }
      }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------- gate + residual
struct CtGateTf32Params {
  const float* z;                // [M][128]
  const float* xin;              // [M][128] or nullptr
  const float* cond;             // [P][128]
  const float* wsc;              // [128][128]  gate weight [out][in]
  const float* bsc;              // [128]
  float* y;                      // [M][128]
  long M, P;
};

template <int NW_, int MINB_>
struct CtGateTf32Cfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = MINB_;
  static constexpr int SMEM = 128 * 512;                                 // Wg, fp32
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) ct_gate_tf32_kernel(const CtGateTf32Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t swg = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  load_weight128_tf32(swg, p.wsc, tid, G::NTHR);
  __syncthreads();

  const long ntile = (p.M + 15) / 16;
  for (long tile = (long)blockIdx.x * G::NW + warp; tile < ntile; tile += (long)gridDim.x * G::NW) {
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = rr[hh] < p.M; }
    float4 uc[2][4][2], uz[2][4][2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long crow = rok[hh] ? rr[hh] % p.P : 0;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) uc[hh][gq][h] = rok[hh] ? ldg_f4(p.cond + crow * 128 + 32 * gq + 8 * q4 + 4 * h) : zero4();
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int h = 0; h < 2; ++h) uz[hh][gq][h] = rok[hh] ? ldg_f4(p.z + rr[hh] * 128 + 32 * gq + 8 * q4 + 4 * h) : zero4();

    uint32_t ca[16][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float v[8] = {uc[hh][gq][0].x, uc[hh][gq][0].y, uc[hh][gq][0].z, uc[hh][gq][0].w, uc[hh][gq][1].x, uc[hh][gq][1].y, uc[hh][gq][1].z, uc[hh][gq][1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) { ca[4 * gq + s][hh] = to_tf32(v[s]); ca[4 * gq + s][2 + hh] = to_tf32(v[s + 4]); }
      }

#pragma unroll
    for (int og = 0; og < 4; ++og) {
      float gacc[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) { gacc[t][0] = gacc[t][1] = gacc[t][2] = gacc[t][3] = 0.f; }
#pragma unroll
      for (int gk = 0; gk < 4; ++gk)
#pragma unroll
        for (int sp = 0; sp < 2; ++sp) {                                 // k steps 2 sp, 2 sp + 1 of group gk (b0, b1 of both in one LDS.128), the 4 n tiles interleaved
          float4 bg[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) bg[t] = lds_f4(swg + tf_woff((4 * og + t) * 8 + g8, 8 * gk + 2 * q4 + sp));
#pragma unroll
          for (int ss = 0; ss < 2; ++ss)
#pragma unroll
            for (int t = 0; t < 4; ++t)
              mma1688(gacc[t], ca[4 * gk + 2 * sp + ss], __float_as_uint(ss ? bg[t].z : bg[t].x), __float_as_uint(ss ? bg[t].w : bg[t].y));
        }
      const float4 bs0 = ldg_f4(p.bsc + 32 * og + 8 * q4), bs1 = ldg_f4(p.bsc + 32 * og + 8 * q4 + 4);
      const float bs[8] = {bs0.x, bs0.y, bs0.z, bs0.w, bs1.x, bs1.y, bs1.z, bs1.w};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float zv[8] = {uz[hh][og][0].x, uz[hh][og][0].y, uz[hh][og][0].z, uz[hh][og][0].w, uz[hh][og][1].x, uz[hh][og][1].y, uz[hh][og][1].z, uz[hh][og][1].w};
        float xv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, o[8];
        if (p.xin != nullptr && rok[hh]) {
          const float4 x0 = ldg_f4(p.xin + rr[hh] * 128 + 32 * og + 8 * q4), x1 = ldg_f4(p.xin + rr[hh] * 128 + 32 * og + 8 * q4 + 4);
          xv[0] = x0.x; xv[1] = x0.y; xv[2] = x0.z; xv[3] = x0.w; xv[4] = x1.x; xv[5] = x1.y; xv[6] = x1.z; xv[7] = x1.w;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = fmaf(sigmoidf(gacc[i >> 1][2 * hh + (i & 1)] + bs[i]), zv[i], xv[i]);
        if (rok[hh]) {
          stg_f4(p.y + rr[hh] * 128 + 32 * og + 8 * q4, make_float4(o[0], o[1], o[2], o[3]));
          stg_f4(p.y + rr[hh] * 128 + 32 * og + 8 * q4 + 4, make_float4(o[4], o[5], o[6], o[7]));
        }
      }
    }
  }
}

}  // namespace adl
