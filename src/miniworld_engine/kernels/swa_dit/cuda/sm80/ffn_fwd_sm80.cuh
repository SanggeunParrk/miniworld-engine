// ffn_fwd_sm80.cuh -- the last stage of the SWA atom DiT block, A100 / sm_80: the gated output projection with its residual, the adaLN-modulated RMSNorm and the SwiGLU FFN with
// its residual.  The twin of the Triton ``_swa_oproj_ffn_fwd_kernel`` with its rounding points (rn = round to bf16):
//
//   gated = rn(sigmoid(G) O)             att = rn(gated Wo^T)             q1 = rn(Qin + rn(rn(gate_a) att))
//   y     = rn(rn(q1 rstd) (1 + scale_f) + shift_f)       rstd = 1 / sqrt(mean(q1^2) + eps)
//   h_u   = rn(a_u sigmoid(a_u) b_u),  a = y Wu1^T, b = y Wu2^T   (fp32)        ffn = rn(sum_u h_u Wd[:, u])        out = rn(q1 + rn(rn(gate_f) ffn))
//
// gate_a | shift_f | scale_f | gate_f are mod[mrow, 256:384] | [384:512] | [512:640] | [640:768].  With ``save`` the kernel also writes q1, att, y and ffn (the backward's operands).
//
// One CTA = NW warps x 16 rows per iteration (tiles are dealt round-robin over the CTAs, so a small problem still uses every SM).  The output projection's weights stay in shared
// memory (rows in the f1 order: a thread's accumulator pair over the 4 n tiles of a group is 8 consecutive channels, so a row's channels are the same 8 + 8 + 8 + 8 in the
// accumulators of the projection, in the A fragments of the next product and in the FFN's output); the FFN's weights stream through a two-stage ring of 64-unit chunks
// (Wu1 | Wu2 | Wd, 48 KB each), prefetched under the previous chunk's products; the hidden activations go from the accumulators straight into the A fragments of the down projection.
#pragma once
#include "qkvg_fwd_sm80.cuh"     // qf_channel / qf_woff, rn

namespace sw80 {

struct FfnFwdParams {
  const __nv_bfloat16* qi;                           // [M][128] the residual stream (the stage's input)
  const __nv_bfloat16* g;                            // [M][128]
  const __nv_bfloat16* o;                            // [M][128] attention output
  const float* mod;                                  // [B S][768]
  const __nv_bfloat16* wo;                           // [128][128]
  const __nv_bfloat16* wu;                           // [512][128]: Wu1 (rows 0..255) | Wu2
  const __nv_bfloat16* wd;                           // [128][256]
  __nv_bfloat16* out;                                // [M][128]
  __nv_bfloat16 *q1s, *atts, *ys, *ffs;              // [M][128] or nullptr (save)
  int M, S, B;
  float eps;
  unsigned long long* prof;                          // SWA_PROF builds: cycles per phase of the warp 0 of every CTA (nullptr otherwise)
};

#ifdef SWA_PROF
#define PT(i) do { if (warp == 0 && lane == 0 && p.prof != nullptr) { const unsigned long long now = clock64(); atomicAdd(&p.prof[i], now - tprev); tprev = now; } } while (0)
#else
#define PT(i) do { } while (0)
#endif

template <int NW_>
struct FfnCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32;
  static constexpr int WO_BYTES = 128 * 256, CH_BYTES = 3 * 16384, SMEM = WO_BYTES + 2 * CH_BYTES;
};


template <class G>
DEVI void ffn_load_chunk(const FfnFwdParams& p, uint32_t sb_stage, int j, int tid) {
  // Wu1_j and Wu2_j: 64 rows each of 256 B (natural order, the chunk swizzle of qf_woff); Wd_j: 128 permuted rows of 64 hidden units (128 B, swizzled)
  for (int i = tid; i < 64 * 16; i += G::NTHR) {
    const int r = i >> 4, c = i & 15;
    cp_async16(sb_stage + qf_woff(r, c), p.wu + (size_t)(64 * j + r) * 128 + c * 8);
    cp_async16(sb_stage + 16384 + qf_woff(r, c), p.wu + (size_t)(256 + 64 * j + r) * 128 + c * 8);
  }
  for (int i = tid; i < 128 * 8; i += G::NTHR) {
    const int s = i >> 3, c = i & 7;
    cp_async16(sb_stage + 32768 + sw128(s, c), p.wd + (size_t)qf_channel(s) * 256 + 64 * j + c * 8);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) ffn_fwd_kernel(const FfnFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const uint32_t so = sb, sring = sb + G::WO_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- Wo into shared memory in the f1 row order, once per CTA
  for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(so + qf_woff(i >> 4, i & 15), p.wo + (size_t)qf_channel(i >> 4) * 128 + (i & 15) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wbq[4];                                               // B-fragment address of channel chunk kb in row g8 (a row of Wo or of a Wu chunk)
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  // Tiles are 16 consecutive atoms of one sample, numbered with the SAMPLE FASTEST: the 8 warps of a CTA take consecutive tiles, i.e. (mostly) the same atoms of 8 samples, which
  // read the same rows of the hoisted modulation (L1 hits) and whose loads are issued together.  A warp past the last tile runs on masked rows (the barriers stay uniform).
  const int nsamp = p.M / p.S, ntile = ((p.S + 15) / 16) * nsamp;
  for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
#ifdef SWA_PROF
    unsigned long long tprev = clock64();
#endif
    const int tile = base + warp, tsi = tile / nsamp, nsm = tile - tsi * nsamp;
    int rr[2], mrow[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int s = 16 * tsi + g8 + 8 * hh;
      rok[hh] = tile < ntile && s < p.S;
      rr[hh] = nsm * p.S + s;
      mrow[hh] = (nsm % p.B) * p.S + s;
    }
    // every input of the tile is requested before anything waits for it
    uint4 ug[2][4], uo[2][4], uq[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        ug[hh][kb] = uo[hh][kb] = uq[hh][kb] = make_uint4(0u, 0u, 0u, 0u);
        if (rok[hh]) {
          const size_t off = (size_t)rr[hh] * 128 + 32 * kb + 8 * q4;
          ug[hh][kb] = ldg128(p.g + off);
          uo[hh][kb] = ldg128(p.o + off);
          uq[hh][kb] = ldg128(p.qi + off);
        }
      }
    __syncthreads();                                             // the previous iteration's reads of the ring are done
    ffn_load_chunk<G>(p, sring, 0, tid);
    cp_async_commit();
    ffn_load_chunk<G>(p, sring + G::CH_BYTES, 1, tid);
    cp_async_commit();
    PT(0);

    // ---- gated = rn(sigmoid(G) O): the A fragments of the output projection (channel chunk kb = 32 kb + 8 q4 .. + 7 of row g8 + 8 hh)
    uint32_t a[8][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const uint4 ugk = ug[hh][kb], uok = uo[hh][kb];
        const float gv[8] = {bf16lo(ugk.x), bf16hi(ugk.x), bf16lo(ugk.y), bf16hi(ugk.y), bf16lo(ugk.z), bf16hi(ugk.z), bf16lo(ugk.w), bf16hi(ugk.w)};
        const float ov[8] = {bf16lo(uok.x), bf16hi(uok.x), bf16lo(uok.y), bf16hi(uok.y), bf16lo(uok.z), bf16hi(uok.z), bf16lo(uok.w), bf16hi(uok.w)};
        float t[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = sigmoidf(gv[i]) * ov[i];
        a[2 * kb][hh] = pack_bf16(t[0], t[1]);      a[2 * kb][2 + hh] = pack_bf16(t[2], t[3]);
        a[2 * kb + 1][hh] = pack_bf16(t[4], t[5]);  a[2 * kb + 1][2 + hh] = pack_bf16(t[6], t[7]);
      }

    PT(1);
    // ---- att = rn(gated Wo^T): 4 groups x 4 n tiles (the f1 order: a group's 4 tiles are 8 consecutive channels of the thread)
    float acc[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 b = lds128_ro(so + nt * 8 * 256 + wbq[kb]);
        mma16816(acc[nt >> 2][nt & 3], a[2 * kb], b.x, b.y);
        mma16816(acc[nt >> 2][nt & 3], a[2 * kb + 1], b.z, b.w);
      }

    PT(2);
    // ---- q1 = rn(Qin + rn(rn(gate_a) att)), then the row's sum of squares; q1 stays in registers (bf16 pairs q1p[hh][group][j] = channels 32 gq + 8 q4 + 2 j + {0, 1})
    uint32_t q1p[2][4][4];
    float ss[2] = {0.f, 0.f};
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float* mp = p.mod + (size_t)mrow[hh] * 768;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 uqk = uq[hh][gq];
        const float4 g0 = *reinterpret_cast<const float4*>(mp + 256 + 32 * gq + 8 * q4), g1 = *reinterpret_cast<const float4*>(mp + 256 + 32 * gq + 8 * q4 + 4);
        const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        const uint32_t qp[4] = {uqk.x, uqk.y, uqk.z, uqk.w};
        uint32_t atp[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          atp[j] = pack_bf16(acc[gq][j][2 * hh], acc[gq][j][2 * hh + 1]);                                  // att = rn(gated Wo^T)
          q1p[hh][gq][j] = fma_bf16x2(pack_bf16(gv[2 * j], gv[2 * j + 1]), atp[j], qp[j]);                  // q1 = Qin + gate_a att, one rounding
          const float e0 = bf16lo(q1p[hh][gq][j]), e1 = bf16hi(q1p[hh][gq][j]);
          ss[hh] = fmaf(e0, e0, fmaf(e1, e1, ss[hh]));
        }
        if (p.q1s != nullptr && rok[hh]) {
          stg128(p.q1s + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(q1p[hh][gq][0], q1p[hh][gq][1], q1p[hh][gq][2], q1p[hh][gq][3]));
          stg128(p.atts + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(atp[0], atp[1], atp[2], atp[3]));
        }
      }
    }
    PT(3);
    // ---- y = rn(rn(q1 rstd) (1 + scale_f) + shift_f): the A fragments of the FFN (the channel chunks gq are the A fragments' kb)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float rstd = 1.f / sqrtf(quad_sum(ss[hh]) * (1.f / 128.f) + p.eps);
      const float* mp = p.mod + (size_t)mrow[hh] * 768;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const float4 s0 = *reinterpret_cast<const float4*>(mp + 384 + 32 * gq + 8 * q4), s1 = *reinterpret_cast<const float4*>(mp + 384 + 32 * gq + 8 * q4 + 4);
        const float4 c0 = *reinterpret_cast<const float4*>(mp + 512 + 32 * gq + 8 * q4), c1 = *reinterpret_cast<const float4*>(mp + 512 + 32 * gq + 8 * q4 + 4);
        const float sh[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        const float sc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float y[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          y[2 * j] = fmaf(bf16lo(q1p[hh][gq][j]) * rstd, 1.f + sc[2 * j], sh[2 * j]);
          y[2 * j + 1] = fmaf(bf16hi(q1p[hh][gq][j]) * rstd, 1.f + sc[2 * j + 1], sh[2 * j + 1]);
        }
        const uint32_t n0 = pack_bf16(y[0], y[1]), n1 = pack_bf16(y[2], y[3]), n2 = pack_bf16(y[4], y[5]), n3 = pack_bf16(y[6], y[7]);
        a[2 * gq][hh] = n0;      a[2 * gq][2 + hh] = n1;
        a[2 * gq + 1][hh] = n2;  a[2 * gq + 1][2 + hh] = n3;
        if (p.ys != nullptr && rok[hh]) stg128(p.ys + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(n0, n1, n2, n3));
      }
    }

    PT(4);
    // ---- the FFN: 4 chunks of 64 hidden units; a, b = y Wu1^T, y Wu2^T; h = rn(a sigmoid(a) b) -> the A fragments of the down projection
    float dn[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { dn[gq][j][0] = dn[gq][j][1] = dn[gq][j][2] = dn[gq][j][3] = 0.f; }
#pragma unroll 1
    for (int j = 0; j < 4; ++j) {
      if (j < 3) cp_async_wait<1>(); else cp_async_wait<0>();
      __syncthreads();                                           // chunk j is in shared memory for every warp
      PT(5);
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
      PT(6);
      uint32_t hf[4][4];                                         // k step ks of the down projection = hidden units 16 ks .. 16 ks + 15 of the chunk = n tiles 2 ks, 2 ks + 1
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
      PT(7);
      // down projection: out channel n tile nt (packed Wd rows 8 nt ..), k = 64 hidden units = 2 ldmatrix.x4 of 32 units
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
      PT(8);
      __syncthreads();                                           // every warp is done with this stage
      if (j + 2 < 4) { ffn_load_chunk<G>(p, stg, j + 2, tid); cp_async_commit(); }
      PT(9);
    }

    // ---- ffn = rn(dn); out = rn(q1 + rn(rn(gate_f) ffn))
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float* mp = p.mod + (size_t)mrow[hh] * 768;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const float4 f0 = *reinterpret_cast<const float4*>(mp + 640 + 32 * gq + 8 * q4), f1 = *reinterpret_cast<const float4*>(mp + 640 + 32 * gq + 8 * q4 + 4);
        const float gv[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
        uint32_t ffp[4], op[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ffp[j] = pack_bf16(dn[gq][j][2 * hh], dn[gq][j][2 * hh + 1]);                                    // ffn = rn(down projection)
          op[j] = fma_bf16x2(pack_bf16(gv[2 * j], gv[2 * j + 1]), ffp[j], q1p[hh][gq][j]);                  // out = q1 + gate_f ffn, one rounding
        }
        if (rok[hh]) {
          stg128(p.out + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(op[0], op[1], op[2], op[3]));
          if (p.ffs != nullptr) stg128(p.ffs + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(ffp[0], ffp[1], ffp[2], ffp[3]));
        }
      }
    }
    PT(10);
  }
}

}  // namespace sw80
