// ct_atom_fwd.cuh -- the tail of the ConditionedTransition at the atom width (d_hidden = d_cond = 128, expansion 2: hidden 256), bf16 rows, A100 / sm_80, ONE kernel:
//
//   [a | b] = xa [Wa; Wb]^T          xa = the AdaLN's output (``kernels/adaln/cuda/sm80``), fp32 accumulation
//   h = rn(silu(a) b)                the SwiGLU on the accumulators; h is the A operand of the squeeze
//   z = h Ws^T                       fp32 accumulation, NOT rounded for the output (the training forward also stores rn(z))
//   g = cond Wsc^T + bsc             a fourth product on the raw conditioning, fp32
//   y = rn(x + sigmoid(g) z)         the residual (x = nullptr: the update alone)
//
// A warp owns 16 consecutive rows from the first load to the last store (adaln_mma.cuh: the thread's 32 channels of a row are four 16-byte vectors that are its A fragments and,
// through the f1 weight-row order, its accumulators), so the [M, 512] pre-activation and the hidden h never leave the registers.  One CTA of NW warps per SM; the gate weight stays
// in shared memory, the FFN weights stream through a two-stage ring of 64-hidden-unit chunks (Wa | Wb | Ws rows, 16 KB each, prefetched under the previous chunk's products and shared
// by the warps).  The kernel is the FFN half of ``kernels/swa_dit/cuda/sm80/ffn_fwd_sm80.cuh`` with this block's prologue and epilogue.
#pragma once
#include "adaln_mma.cuh"

namespace adl {

struct CtTailFwdParams {
  const bf* xa;                  // [M][128]
  const bf* xin;                 // [M][128] or nullptr
  const bf* cond;                // [P][128]
  const bf* wab;                 // [512][128]  Wa (rows 0 .. 255) | Wb: expand weights [hidden][in]
  const bf* wd;                  // [128][256]  squeeze weight [out][hidden]
  const bf* wsc;                 // [128][128]  gate weight [out][in]
  const bf* bsc;                 // [128]
  bf* y;                         // [M][128]
  bf* zs;                        // [M][128] or nullptr: rn(z)
  long M, P;
};

template <int NW_>
struct CtTailFwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = 1;
  static constexpr int WSC_BYTES = QF_ROWS * 256, CH_BYTES = 3 * 16384, SMEM = WSC_BYTES + 2 * CH_BYTES;
};

// chunk j = hidden units 64 j .. 64 j + 63: Wa_j and Wb_j (64 rows of 256 B each, natural order, the qf_woff swizzle), Ws_j (128 output rows in the f1 order x 64 hidden units: 128 B rows, sw128)
template <class G>
ADL_DEVI void tail_load_chunk(const CtTailFwdParams& p, uint32_t stage, int j, int tid) {
  for (int i = tid; i < 64 * 16; i += G::NTHR) {
    const int r = i >> 4, c = i & 15;
    cp_async16(stage + qf_woff(r, c), p.wab + (size_t)(64 * j + r) * 128 + c * 8);
    cp_async16(stage + 16384 + qf_woff(r, c), p.wab + (size_t)(256 + 64 * j + r) * 128 + c * 8);
  }
  for (int i = tid; i < 128 * 8; i += G::NTHR) {
    const int s = i >> 3, c = i & 7;
    cp_async16(stage + 32768 + sw128(s, c), p.wd + (size_t)qf_channel(s) * 256 + 64 * j + c * 8);
  }
}

ADL_DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) ct_atom_fwd_kernel(const CtTailFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw), swsc = sbase, sring = sbase + G::WSC_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  load_weight128(swsc, p.wsc, tid, G::NTHR);                                  // the gate weight, once per CTA, in the f1 row order
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wbq[4];                                                            // B-fragment address of channel chunk kb in row g8 of a group of 8 rows
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  const long ntile = (p.M + 15) / 16;
  for (long base = (long)blockIdx.x * G::NW; base < ntile; base += (long)G::NW * gridDim.x) {
    const long tile = base + warp;
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = tile < ntile && rr[hh] < p.M; }
    uint4 ux[2][4];                                                           // xa: the A fragments of the first product
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) ux[hh][gq] = rok[hh] ? ldg128(p.xa + rr[hh] * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();                                                          // the previous iteration's reads of the ring are done
    tail_load_chunk<G>(p, sring, 0, tid);
    cp_async_commit();
    tail_load_chunk<G>(p, sring + G::CH_BYTES, 1, tid);
    cp_async_commit();

    uint32_t a[8][4];
    a_from_vec(a, ux);

    float dn[4][4][4];                                                        // z = h Ws^T: group gq of 4 n tiles = channels 32 gq + 8 q4 + 0 .. 7 of rows g8 / g8 + 8
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { dn[gq][j][0] = dn[gq][j][1] = dn[gq][j][2] = dn[gq][j][3] = 0.f; }
    uint4 uc[2][4], uxin[2][4];
#pragma unroll 1
    for (int j = 0; j < 4; ++j) {
      if (j < 3) cp_async_wait<1>(); else cp_async_wait<0>();
      __syncthreads();                                                        // chunk j is in shared memory for every warp
      const uint32_t stg = sring + (j & 1) * G::CH_BYTES, w1 = stg, w2 = stg + 16384, wdb = stg + 32768;
      float ac[8][4], bc[8][4];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        ac[t][0] = ac[t][1] = ac[t][2] = ac[t][3] = 0.f;
        bc[t][0] = bc[t][1] = bc[t][2] = bc[t][3] = 0.f;
      }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const uint4 b1 = lds128_ro(w1 + t * 8 * 256 + wbq[kb]);
          const uint4 b2 = lds128_ro(w2 + t * 8 * 256 + wbq[kb]);
          mma16816(ac[t], a[2 * kb], b1.x, b1.y);
          mma16816(ac[t], a[2 * kb + 1], b1.z, b1.w);
          mma16816(bc[t], a[2 * kb], b2.x, b2.y);
          mma16816(bc[t], a[2 * kb + 1], b2.z, b2.w);
        }
      uint32_t hf[4][4];                                                      // k step ks of the squeeze = hidden units 16 ks .. 16 ks + 15 of the chunk = n tiles 2 ks, 2 ks + 1
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        float hv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float av = ac[2 * ks + (e >> 2)][e & 3], bv = bc[2 * ks + (e >> 2)][e & 3];
          hv[e] = av * sigmoidf(av) * bv;
        }
        hf[ks][0] = pack_bf16(hv[0], hv[1]); hf[ks][1] = pack_bf16(hv[2], hv[3]);
        hf[ks][2] = pack_bf16(hv[4], hv[5]); hf[ks][3] = pack_bf16(hv[6], hv[7]);
      }
      if (j == 3) {                                                           // the epilogue's operands: requested now, consumed after the squeeze's products
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const long crow = rok[hh] ? rr[hh] % p.P : 0;
#pragma unroll
          for (int gq = 0; gq < 4; ++gq) {
            uc[hh][gq] = uxin[hh][gq] = make_uint4(0u, 0u, 0u, 0u);
            if (rok[hh]) {
              uc[hh][gq] = ldg128(p.cond + crow * 128 + 32 * gq + 8 * q4);
              if (p.xin != nullptr) uxin[hh][gq] = ldg128(p.xin + rr[hh] * 128 + 32 * gq + 8 * q4);
            }
          }
        }
      }
      // the squeeze: out channel n tile nt (packed Ws rows 8 nt ..), k = 64 hidden units = 2 ldmatrix.x4 of 32 units
#pragma unroll
      for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          uint32_t r[4];
          const uint32_t row = 8 * nt + (lane & 7), gr = 4 * u + (lane >> 3);
          ldsm_x4(r, wdb + sw128(row, gr));
          mma16816(dn[nt >> 2][nt & 3], hf[2 * u], r[0], r[1]);
          mma16816(dn[nt >> 2][nt & 3], hf[2 * u + 1], r[2], r[3]);
        }
      __syncthreads();                                                        // every warp is done with this stage
      if (j + 2 < 4) { tail_load_chunk<G>(p, stg, j + 2, tid); cp_async_commit(); }
    }

    // ---- g = cond Wsc^T + bsc on the raw conditioning, then y = x + sigmoid(g) z, per group of 32 output channels
    uint32_t ca[8][4];
    a_from_vec(ca, uc);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      float gacc[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) { gacc[t][0] = gacc[t][1] = gacc[t][2] = gacc[t][3] = 0.f; }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const uint4 b = lds128_ro(swsc + (4 * gq + t) * 8 * 256 + wbq[kb]);
          mma16816(gacc[t], ca[2 * kb], b.x, b.y);
          mma16816(gacc[t], ca[2 * kb + 1], b.z, b.w);
        }
      float bs[8];
      unpack8(ldg128(p.bsc + 32 * gq + 8 * q4), bs);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float xv[8], o[8], zr[8];
        unpack8(uxin[hh][gq], xv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float z = dn[gq][i >> 1][2 * hh + (i & 1)];
          const float gate = sigmoidf(gacc[i >> 1][2 * hh + (i & 1)] + bs[i]);
          o[i] = fmaf(gate, z, p.xin != nullptr ? xv[i] : 0.f);
          zr[i] = z;
        }
        if (rok[hh]) {
          stg128(p.y + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(o));
          if (p.zs != nullptr) stg128(p.zs + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(zr));
        }
      }
    }
  }
}

}  // namespace adl
