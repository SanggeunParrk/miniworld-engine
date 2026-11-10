// post_fwd.cuh -- the attention output and the conditioned transition of the AF3 local atom DiT block on A100 / sm_80 (per row of M = A N; the bf16 module's rounding points, rn = round to bf16;
// s2 | bi2 | so | st = mod[r, 512:1024] from cond_fwd, the scales and gates already as rn(sigmoid)):
//
//   gated = rn(SG O);   u = rn(gated Wo^T);   a2 = rn(x + rn(so u))
//   x2 = rn(rn(s2 rn(LayerNorm(a2))) + bi2);   h = rn(rn(silu(rn(x2 Wa^T))) rn(x2 Wb^T));   t = rn(h Ws^T);   out = rn(a2 + rn(st t))
// With ``save`` it also writes u, a2, x2 and t (the backward's operands).
//
// The SWA ffn stage's structure (ffn_fwd_sm80.cuh): one persistent CTA per SM of NW warps x 16 rows per iteration; the output projection's weights stay in shared memory in the f1 row order;
// the transition's weights stream through a two-stage ring of 64-unit chunks (Wa | Wb | Ws, 48 KB each), prefetched under the previous chunk's products; the hidden activations go from the
// accumulators straight into the A fragments of the down projection.
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

struct PostFwdParams {
  const __nv_bfloat16* x;                            // [M][128] the residual stream (the stage's input)
  const __nv_bfloat16* o;                            // [M][128] attention output
  const __nv_bfloat16* sg;                           // [M][128] rn(sigmoid(G))
  const __nv_bfloat16* mod;                          // [M][1024]
  const __nv_bfloat16* wo;                           // [128][128]
  const __nv_bfloat16* wu;                           // [512][128]: Wa (rows 0..255) | Wb
  const __nv_bfloat16* wd;                           // [128][256]
  __nv_bfloat16* out;                                // [M][128]
  __nv_bfloat16 *us, *a2s, *x2s, *ts;                // [M][128] or nullptr (save)
  int M;
  float eps;
  int prefetch;                                      // 1: the next tile's rows are requested into L2 while the tile before runs
};

// NW_ warps a CTA, HC_ hidden units a chunk of the transition's weight ring (64: 48 KB a stage, 32: 24 KB), MINB_ CTAs a SM (two 4-warp CTAs with their own rings are two groups
// that drift apart: one's elementwise phase overlaps the other's products)
template <int NW_, int HC_ = 64, int MINB_ = 1>
struct PostCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, HC = HC_, MINB = MINB_, NCH = 256 / HC_, NT8 = HC_ / 8, KS = HC_ / 16, DU = HC_ / 32;
  static constexpr int WO_BYTES = 128 * 256, W1 = HC_ * 256, CH_BYTES = 3 * HC_ * 256, SMEM = WO_BYTES + 2 * CH_BYTES;
};

// byte offset of 16-byte chunk c of row s of a [128][HC units] tile of bf16: rows of 128 B (HC = 64, 8 chunks, XOR with s & 7) or of 64 B (HC = 32, 4 chunks, XOR with (s >> 1) & 3)
template <int HC>
DEVI uint32_t swh(uint32_t s, uint32_t c) {
  return HC == 64 ? sw128(s, c) : s * 64u + ((c ^ ((s >> 1) & 3u)) << 4);
}

template <class G>
DEVI void pof_load_chunk(const PostFwdParams& p, uint32_t sb_stage, int j, int tid) {
  // Wa_j and Wb_j: HC rows each of 256 B (natural order, the chunk swizzle of qf_woff); Ws_j: 128 permuted rows of HC hidden units, swizzled
  for (int i = tid; i < G::HC * 16; i += G::NTHR) {
    const int r = i >> 4, c = i & 15;
    cp_async16(sb_stage + qf_woff(r, c), p.wu + (size_t)(G::HC * j + r) * 128 + c * 8);
    cp_async16(sb_stage + G::W1 + qf_woff(r, c), p.wu + (size_t)(256 + G::HC * j + r) * 128 + c * 8);
  }
  constexpr int GR = G::HC / 8;                                    // 16-byte chunks of a row of the Ws tile
  for (int i = tid; i < 128 * GR; i += G::NTHR) {
    const int s = i / GR, c = i % GR;
    cp_async16(sb_stage + 2 * G::W1 + swh<G::HC>(s, c), p.wd + (size_t)qf_channel(s) * 256 + G::HC * j + c * 8);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) post_fwd_kernel(const PostFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const uint32_t so = sb, sring = sb + G::WO_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- Wo into shared memory in the f1 row order, once per CTA
  for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(so + qf_woff(i >> 4, i & 15), p.wo + (size_t)qf_channel(i >> 4) * 128 + (i & 15) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wbq[4];                                               // B-fragment address of channel chunk kb in row g8 (a row of Wo or of a Wa / Wb chunk)
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  const int ntile = (p.M + 15) / 16;
  for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
    const int tile = base + warp;
    int rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      rr[hh] = 16 * tile + g8 + 8 * hh;
      rok[hh] = tile < ntile && rr[hh] < p.M;
    }
    if (p.prefetch) {                                            // the next tile's rows into L2 (its loads then hit): x, o, sg and the mod columns 512 .. 1023
      const int nt = tile + G::NW * (int)gridDim.x;
      if (nt < ntile) {
        pf_rows(p.x, 256, 0, 2, 16 * nt, 16, p.M, lane);
        pf_rows(p.o, 256, 0, 2, 16 * nt, 16, p.M, lane);
        pf_rows(p.sg, 256, 0, 2, 16 * nt, 16, p.M, lane);
        pf_rows(p.mod, 2048, 1024, 8, 16 * nt, 16, p.M, lane);
      }
    }
    // every input of the tile is requested before anything waits for it
    uint4 ug[2][4], uo[2][4], ux[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        ug[hh][kb] = uo[hh][kb] = ux[hh][kb] = make_uint4(0u, 0u, 0u, 0u);
        if (rok[hh]) {
          const size_t off = (size_t)rr[hh] * 128 + 32 * kb + 8 * q4;
          ug[hh][kb] = ldg128(p.sg + off);
          uo[hh][kb] = ldg128(p.o + off);
          ux[hh][kb] = ldg128(p.x + off);
        }
      }
    __syncthreads();                                             // the previous iteration's reads of the ring are done
    pof_load_chunk<G>(p, sring, 0, tid);
    cp_async_commit();
    pof_load_chunk<G>(p, sring + G::CH_BYTES, 1, tid);
    cp_async_commit();

    // ---- gated = rn(SG O): the A fragments of the output projection (packed bf16 arithmetic, one rounding per operation)
    uint32_t a[8][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
        put_a(a, kb, hh, make_uint4(pmul(ug[hh][kb].x, uo[hh][kb].x), pmul(ug[hh][kb].y, uo[hh][kb].y), pmul(ug[hh][kb].z, uo[hh][kb].z), pmul(ug[hh][kb].w, uo[hh][kb].w)));

    // ---- u = rn(gated Wo^T): 4 groups x 4 n tiles (a group's 4 tiles are 8 consecutive channels of the thread)
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

    // ---- a2 = rn(x + rn(so u)); stays in registers as bf16 pairs a2p[hh][group][j] = channels 32 gq + 8 q4 + 2 j + {0, 1}; the row's mean and rstd
    uint32_t a2p[2][4][4];
    float mrs[2], rstd[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float sum = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 sov = rok[hh] ? ldg128(p.mod + (size_t)rr[hh] * 1024 + 768 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t uw[4] = {pk2(acc[gq][0][2 * hh], acc[gq][0][2 * hh + 1]), pk2(acc[gq][1][2 * hh], acc[gq][1][2 * hh + 1]),
                                pk2(acc[gq][2][2 * hh], acc[gq][2][2 * hh + 1]), pk2(acc[gq][3][2 * hh], acc[gq][3][2 * hh + 1])};
        const uint4 a2u = make_uint4(padd(ux[hh][gq].x, pmul(sov.x, uw[0])), padd(ux[hh][gq].y, pmul(sov.y, uw[1])),
                                     padd(ux[hh][gq].z, pmul(sov.z, uw[2])), padd(ux[hh][gq].w, pmul(sov.w, uw[3])));
        a2p[hh][gq][0] = a2u.x; a2p[hh][gq][1] = a2u.y; a2p[hh][gq][2] = a2u.z; a2p[hh][gq][3] = a2u.w;
#pragma unroll
        for (int e = 0; e < 4; ++e) sum += bf16lo(a2p[hh][gq][e]) + bf16hi(a2p[hh][gq][e]);
        if (p.a2s != nullptr && rok[hh]) {
          stg128(p.a2s + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, a2u);
          stg128(p.us + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(uw[0], uw[1], uw[2], uw[3]));
        }
      }
      const float mean = quad_sum(sum) * (1.f / 128.f);
      float vs = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float d0 = bf16lo(a2p[hh][gq][e]) - mean, d1 = bf16hi(a2p[hh][gq][e]) - mean;
          vs = fmaf(d0, d0, fmaf(d1, d1, vs));
        }
      rstd[hh] = 1.f / sqrtf(quad_sum(vs) * (1.f / 128.f) + p.eps);
      mrs[hh] = mean * rstd[hh];
    }

    // ---- x2 = rn(rn(s2 rn(LayerNorm(a2))) + bi2): the A fragments of the transition (the channel groups gq are the A fragments' kb)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const __nv_bfloat16* mp = p.mod + (size_t)rr[hh] * 1024 + 512 + 32 * gq + 8 * q4;
        const uint4 s2v = rok[hh] ? ldg128(mp) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 b2v = rok[hh] ? ldg128(mp + 128) : make_uint4(0u, 0u, 0u, 0u);
        uint32_t xn[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) xn[e] = pk2(fmaf(bf16lo(a2p[hh][gq][e]), rstd[hh], -mrs[hh]), fmaf(bf16hi(a2p[hh][gq][e]), rstd[hh], -mrs[hh]));
        const uint4 n = make_uint4(padd(pmul(s2v.x, xn[0]), b2v.x), padd(pmul(s2v.y, xn[1]), b2v.y), padd(pmul(s2v.z, xn[2]), b2v.z), padd(pmul(s2v.w, xn[3]), b2v.w));
        put_a(a, gq, hh, n);
        if (p.x2s != nullptr && rok[hh]) stg128(p.x2s + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, n);
      }

    // ---- the transition: NCH chunks of HC hidden units; a, b = x2 Wa^T, x2 Wb^T; h = rn(rn(silu(rn(a))) rn(b)) -> the A fragments of the down projection
    float dn[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { dn[gq][j][0] = dn[gq][j][1] = dn[gq][j][2] = dn[gq][j][3] = 0.f; }
#pragma unroll 1
    for (int j = 0; j < G::NCH; ++j) {
      if (j < G::NCH - 1) cp_async_wait<1>(); else cp_async_wait<0>();
      __syncthreads();                                           // chunk j is in shared memory for every warp
      const uint32_t stg = sring + (j & 1) * G::CH_BYTES, w1 = stg, w2 = stg + G::W1, wdb = stg + 2 * G::W1;
      float ac[G::NT8][4], bc[G::NT8][4];
#pragma unroll
      for (int t = 0; t < G::NT8; ++t) {
        ac[t][0] = ac[t][1] = ac[t][2] = ac[t][3] = 0.f;
        bc[t][0] = bc[t][1] = bc[t][2] = bc[t][3] = 0.f;
      }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int t = 0; t < G::NT8; ++t) {
          const uint4 b1 = lds128_ro(w1 + t * 8 * 256 + wbq[kb]);
          const uint4 b2 = lds128_ro(w2 + t * 8 * 256 + wbq[kb]);
          mma16816(ac[t], a[2 * kb], b1.x, b1.y);
          mma16816(bc[t], a[2 * kb], b2.x, b2.y);
          mma16816(ac[t], a[2 * kb + 1], b1.z, b1.w);
          mma16816(bc[t], a[2 * kb + 1], b2.z, b2.w);
        }
      uint32_t hf[G::KS][4];                                     // k step ks of the down projection = hidden units 16 ks .. 16 ks + 15 of the chunk = n tiles 2 ks, 2 ks + 1
#pragma unroll
      for (int ks = 0; ks < G::KS; ++ks) {
        uint32_t hp[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {                            // elements 2 e, 2 e + 1 of the 8: tile 2 ks + (e >> 1), accumulator pair (e & 1)
          const float* aa = &ac[2 * ks + (e >> 1)][2 * (e & 1)];
          const float* bb = &bc[2 * ks + (e >> 1)][2 * (e & 1)];
          const uint32_t ar = pk2(aa[0], aa[1]), br = pk2(bb[0], bb[1]);                       // rn(a), rn(b)
          hp[e] = pmul(pk2(silu_f(bf16lo(ar)), silu_f(bf16hi(ar))), br);                       // rn(rn(silu(a)) b)
        }
        hf[ks][0] = hp[0]; hf[ks][1] = hp[1]; hf[ks][2] = hp[2]; hf[ks][3] = hp[3];
      }
      // down projection: out channel n tile nt (packed Ws rows 8 nt ..), k = HC hidden units = DU ldmatrix.x4 of 32 units
#pragma unroll
      for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int u = 0; u < G::DU; ++u) {
          uint32_t r[4];
          const uint32_t row = 8 * nt + (lane & 7), gr = 4 * u + (lane >> 3);
          ldsm_x4(r, wdb + swh<G::HC>(row, gr));
          mma16816(dn[nt >> 2][nt & 3], hf[2 * u], r[0], r[1]);
          mma16816(dn[nt >> 2][nt & 3], hf[2 * u + 1], r[2], r[3]);
        }
      __syncthreads();                                           // every warp is done with this stage
      if (j + 2 < G::NCH) { pof_load_chunk<G>(p, stg, j + 2, tid); cp_async_commit(); }
    }

    // ---- t = rn(down projection); out = rn(a2 + rn(st t))
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 stv = rok[hh] ? ldg128(p.mod + (size_t)rr[hh] * 1024 + 896 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t tw[4] = {pk2(dn[gq][0][2 * hh], dn[gq][0][2 * hh + 1]), pk2(dn[gq][1][2 * hh], dn[gq][1][2 * hh + 1]),
                                pk2(dn[gq][2][2 * hh], dn[gq][2][2 * hh + 1]), pk2(dn[gq][3][2 * hh], dn[gq][3][2 * hh + 1])};
        if (rok[hh]) {
          stg128(p.out + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4,
                 make_uint4(padd(a2p[hh][gq][0], pmul(stv.x, tw[0])), padd(a2p[hh][gq][1], pmul(stv.y, tw[1])), padd(a2p[hh][gq][2], pmul(stv.z, tw[2])), padd(a2p[hh][gq][3], pmul(stv.w, tw[3]))));
          if (p.ts != nullptr) stg128(p.ts + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(tw[0], tw[1], tw[2], tw[3]));
        }
      }
  }
}

}  // namespace lr80
