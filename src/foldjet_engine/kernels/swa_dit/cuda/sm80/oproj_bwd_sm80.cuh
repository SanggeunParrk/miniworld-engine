// oproj_bwd_sm80.cuh -- the backward of the gated output projection of the SWA atom DiT block, A100 / sm_80: the twin of the Triton ``_swa_oproj_bwd_kernel`` with its rounding points.
// Forward:  gated = rn(sigmoid(G) O),  att = rn(gated Wo^T),  q1 = Qin + gate_a att.  For a row (rn = round to bf16, all else fp32):
//
//   gated = rn(sigmoid(G) O)                    -> ``gated``  (the dWo operand)
//   datt  = rn(dq1 rn(gate_a))                  -> ``datt``   (the other dWo operand)
//   dgated = datt Wo                            (fp32 accumulate)
//   dO = rn(dgated sigmoid(G)),  dG = rn(dgated O sigmoid(G) (1 - sigmoid(G)))
//   dv[head] = sum over the head's 32 channels of rn(dO) O                (the attention backward's D = rowsum(dO o), fp32 [N][4][S])
//   d gate_a = dq1 att                          (fp32; per row, see ``bwd_rows_sm80.cuh`` for the two modes)
//
// Structure of the forward ``ffn_fwd_kernel``: one persistent CTA per SM of NW warps, tiles of 16 rows dealt round-robin, Wo^T (the packed [n][k] weight) resident in shared memory in
// the f1 row order (a thread's accumulator pair over the 4 n tiles of a group is 8 consecutive channels, which are also the channels of its A fragments and of its 16-byte loads and stores).
#pragma once
#include "bwd_rows_sm80.cuh"
#include "qkvg_fwd_sm80.cuh"     // qf_channel / qf_woff, rn

namespace sw80 {

struct OprojBwdParams {
  const __nv_bfloat16* dq1;                          // [M][128] the gradient of q1
  const __nv_bfloat16* o;                            // [M][128] the attention output
  const __nv_bfloat16* g;                            // [M][128] the gate projection
  const __nv_bfloat16* att;                          // [M][128] att = rn(gated Wo^T), saved by the forward
  const float* mod;                                  // [B S][768]
  const __nv_bfloat16* wot;                          // [128 n][128 k] = Wo^T
  __nv_bfloat16 *dO, *dG, *datt, *gated;             // [M][128]
  float* dv;                                         // [N][4][S]
  float* dmod;                                       // MODE_SINGLE: [B S][768]; MODE_HOIST: [nblk][B S][768].  Columns 256 .. 383 are d gate_a
  int M, S, B;
};

template <int NW_>
struct OprojBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, SMEM = 128 * 256;
};

DEVI float4 ldg_f4m(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, 1) oproj_bwd_kernel(const OprojBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t so = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(so + qf_woff(i >> 4, i & 15), p.wot + (size_t)qf_channel(i >> 4) * 128 + (i & 15) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wbq[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  // the dq1 / att operands of a tile's first phase are requested during the previous tile (they are consumed before the products, which leave the registers free)
  uint4 cdq[2][4], cat[2][4];
  auto load_first = [&](const RowMap& r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        cdq[hh][kb] = cat[hh][kb] = make_uint4(0u, 0u, 0u, 0u);
        if (r.rok[hh]) {
          const size_t off = (size_t)r.rr[hh] * 128 + 32 * kb + 8 * q4;
          cdq[hh][kb] = ldg128(p.dq1 + off);
          cat[hh][kb] = ldg128(p.att + off);
        }
      }
  };
  const int ntile = bwd_ntile<MODE>(p.M, p.S, p.B);
  RowMap rm = row_map<MODE>(blockIdx.x * G::NW + warp, p.M, p.S, p.B, g8);
  load_first(rm);
  for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
    const int tile = base + warp;
    const RowMap nxt = row_map<MODE>(tile + G::NW * gridDim.x, p.M, p.S, p.B, g8);      // the next tile's rows

    // ---- datt = rn(dq1 rn(gate_a)) as the A fragments; d gate_a = dq1 att leaves at once (MODE_HOIST: summed over the tile's 16 rows first)
    uint32_t a[8][4];
    float dgr[4] = {0.f, 0.f, 0.f, 0.f};                           // MODE_HOIST: the two lanes' share of d gate_a of block kb, summed over the row halves
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float* mp = p.mod + (size_t)rm.mrow[hh] * 768 + 256;
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const uint4 ud = cdq[hh][kb], ua = cat[hh][kb];
        const float4 g0 = ldg_f4m(mp + 32 * kb + 8 * q4), g1 = ldg_f4m(mp + 32 * kb + 8 * q4 + 4);
        const float gav[8] = {rn(g0.x), rn(g0.y), rn(g0.z), rn(g0.w), rn(g1.x), rn(g1.y), rn(g1.z), rn(g1.w)};
        const float dv8[8] = {bf16lo(ud.x), bf16hi(ud.x), bf16lo(ud.y), bf16hi(ud.y), bf16lo(ud.z), bf16hi(ud.z), bf16lo(ud.w), bf16hi(ud.w)};
        const float at8[8] = {bf16lo(ua.x), bf16hi(ua.x), bf16lo(ua.y), bf16hi(ua.y), bf16lo(ua.z), bf16hi(ua.z), bf16lo(ua.w), bf16hi(ua.w)};
        float t[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = dv8[i] * gav[i];
        const uint32_t n0 = pack_bf16(t[0], t[1]), n1 = pack_bf16(t[2], t[3]), n2 = pack_bf16(t[4], t[5]), n3 = pack_bf16(t[6], t[7]);
        a[2 * kb][hh] = n0;      a[2 * kb][2 + hh] = n1;
        a[2 * kb + 1][hh] = n2;  a[2 * kb + 1][2 + hh] = n3;
        if (rm.rok[hh]) stg128(p.datt + (size_t)rm.rr[hh] * 128 + 32 * kb + 8 * q4, make_uint4(n0, n1, n2, n3));
        if (MODE == MODE_SINGLE) {
          if (rm.rok[hh]) {
            float* dm = p.dmod + (size_t)rm.mrow[hh] * 768 + 256 + 32 * kb + 8 * q4;
            *reinterpret_cast<float4*>(dm) = make_float4(dv8[0] * at8[0], dv8[1] * at8[1], dv8[2] * at8[2], dv8[3] * at8[3]);
            *reinterpret_cast<float4*>(dm + 4) = make_float4(dv8[4] * at8[4], dv8[5] * at8[5], dv8[6] * at8[6], dv8[7] * at8[7]);
          }
        } else {
          float v[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) v[i] = dv8[i] * at8[i];
          reduce_scatter_g8<8>(v, lane);
          dgr[kb] += v[0];
        }
      }
    }
    if (MODE == MODE_HOIST && rm.rok[0]) {
      const int i0 = g8_base(lane, 8);
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) p.dmod[((size_t)rm.blk * p.B * p.S + rm.mrow[0]) * 768 + 256 + 32 * kb + 8 * q4 + i0] = dgr[kb];
    }

    load_first(nxt);                                               // the next tile's first-phase operands (the registers of this tile's are free)

    // ---- G and O of the epilogue are requested before the products
    uint4 ug[2][4], uo[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        ug[hh][gq] = uo[hh][gq] = make_uint4(0u, 0u, 0u, 0u);
        if (rm.rok[hh]) {
          const size_t off = (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4;
          ug[hh][gq] = ldg128(p.g + off);
          uo[hh][gq] = ldg128(p.o + off);
        }
      }

    // ---- dgated = datt Wo: 4 groups x 4 n tiles (the f1 order)
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

    // ---- gated, dO, dG and the per-head dv
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 ugk = ug[hh][gq], uok = uo[hh][gq];
        const float gv[8] = {bf16lo(ugk.x), bf16hi(ugk.x), bf16lo(ugk.y), bf16hi(ugk.y), bf16lo(ugk.z), bf16hi(ugk.z), bf16lo(ugk.w), bf16hi(ugk.w)};
        const float ov[8] = {bf16lo(uok.x), bf16hi(uok.x), bf16lo(uok.y), bf16hi(uok.y), bf16lo(uok.z), bf16hi(uok.z), bf16lo(uok.w), bf16hi(uok.w)};
        float gt[8], dov[8], dgv[8], dvs = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float sg = sigmoidf(gv[i]);
          const float dgt = acc[gq][i >> 1][2 * hh + (i & 1)];             // dgated of channel 32 gq + 8 q4 + i: n tile j = i / 2, column 2 q4 + (i & 1) of the quad's pair
          gt[i] = sg * ov[i];
          dov[i] = rn(dgt * sg);
          dgv[i] = ((dgt * ov[i]) * sg) * (1.f - sg);
          dvs = fmaf(dov[i], ov[i], dvs);
        }
        if (rm.rok[hh]) {
          const size_t off = (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4;
          stg128(p.gated + off, make_uint4(pack_bf16(gt[0], gt[1]), pack_bf16(gt[2], gt[3]), pack_bf16(gt[4], gt[5]), pack_bf16(gt[6], gt[7])));
          stg128(p.dO + off, make_uint4(pack_bf16(dov[0], dov[1]), pack_bf16(dov[2], dov[3]), pack_bf16(dov[4], dov[5]), pack_bf16(dov[6], dov[7])));
          stg128(p.dG + off, make_uint4(pack_bf16(dgv[0], dgv[1]), pack_bf16(dgv[2], dgv[3]), pack_bf16(dgv[4], dgv[5]), pack_bf16(dgv[6], dgv[7])));
        }
        dvs = quad_sum(dvs);
        if (q4 == 0 && rm.rok[hh]) p.dv[((size_t)rm.nrow[hh] * 4 + gq) * p.S + rm.sat[hh]] = dvs;
      }
    rm = nxt;
  }
}

}  // namespace sw80
