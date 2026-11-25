// adaln_atom_fwd.cuh -- AdaLN forward at the atom width (d_hidden = d_cond = 128), bf16 rows, A100 / sm_80, ONE kernel:
//
//   aff = rn(LN(cond) w)                                  fp32 statistics; the GEMM operand is rounded to bf16 as the framework's does
//   S | B = aff [Ws | Wb]^T                               two 128 x 128 products on mma.sync m16n8k16, fp32 accumulation, NOT rounded
//   y = rn(sigmoid(S + sb) (x - mean) rstd + B)           LN(x) without affine, fp32 statistics; one rounding
//
// The three row-sized streams (x, cond in, y out: 768 bytes a row) are all the kernel moves: the [M, 256] S | B of the cuBLAS composition never exists.  One persistent CTA per SM
// (or two), NW warps; the weights stay in shared memory in the f1 row order (adaln_mma.cuh); every warp loops over tiles of 16 consecutive rows (tile = 16 rows of the flattened
// [M, 128]), loading the tile's x and cond rows as the vectors of its A fragments before anything waits for them.  Row r reads cond row r % P (P < M: one conditioning shared by
// M / P samples).  With ``xst`` / ``cst`` the kernel also writes the (mean, rstd) of every x row and of every cond row (P == M) for the backward.
#pragma once
#include "adaln_mma.cuh"

namespace adl {

struct AdalnAtomFwdParams {
  const bf* x;                   // [M][128]
  const bf* cond;                // [P][128]
  const float* lnw;              // [128]
  const bf* ws;                  // [128][128]  to_scale.weight [out][in]
  const bf* wb;                  // [128][128]  to_bias.weight
  const bf* sb;                  // [128]       to_scale.bias
  bf* y;                         // [M][128]
  float2* xst;                   // [M] (mean, rstd) or nullptr
  float2* cst;                   // [P] (P == M) or nullptr
  long M, P;
  float eps_x, eps_c;
};

template <int NW_, int MINB_, bool PF_ = false>
struct AdalnAtomFwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = MINB_;
  static constexpr bool PF = PF_;                                        // cp.async prefetch of the next tile into a per-warp staging buffer (x rows | cond rows: 8 KB)
  static constexpr int STAGE = 8192, SMEM = 2 * QF_ROWS * 256 + (PF_ ? NW_ * 8192 : 0);   // Ws | Wb (| the staging buffers)
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) adaln_atom_fwd_kernel(const AdalnAtomFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw), sws = sbase, swb = sbase + QF_ROWS * 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  load_weight128(sws, p.ws, tid, G::NTHR);
  load_weight128(swb, p.wb, tid, G::NTHR);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wbq[4];                                                       // B-fragment address of channel chunk kb in row g8 of a group of 8 packed rows
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  const long ntile = (p.M + 15) / 16;
  const long first = (long)blockIdx.x * G::NW + warp, stride = (long)gridDim.x * G::NW;
  const uint32_t sstage = sbase + 2 * QF_ROWS * 256 + warp * G::STAGE;
  // the staging buffer receives the tile's x rows (4 KB) then its cond rows (4 KB), 256 B rows in the qf_woff order: a warp copies 2 rows per instruction (coalesced 512 B)
  auto prefetch = [&](long t) {
    for (int i = lane; i < 256; i += 32) {
      const int r = i >> 4, c = i & 15;
      const long row = t * 16 + r;
      const bool ok = row < p.M;
      cp_async16(sstage + qf_woff(r, c), p.x + (ok ? row : 0) * 128 + c * 8, ok ? 16u : 0u);
      cp_async16(sstage + 4096 + qf_woff(r, c), p.cond + (ok ? row % p.P : 0) * 128 + c * 8, ok ? 16u : 0u);
    }
    cp_async_commit();
  };
  if (G::PF && first < ntile) prefetch(first);
  for (long tile = first; tile < ntile; tile += stride) {
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = rr[hh] < p.M; }
    // every input of the tile is requested before anything waits for it (PF: it was requested during the previous tile and is read from the staging buffer)
    uint4 ux[2][4], uc[2][4];
    if (G::PF) {
      cp_async_wait<0>();
      __syncwarp();
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int gq = 0; gq < 4; ++gq) {
          ux[hh][gq] = lds128(sstage + qf_woff(g8 + 8 * hh, 4 * gq + q4));
          uc[hh][gq] = lds128(sstage + 4096 + qf_woff(g8 + 8 * hh, 4 * gq + q4));
        }
      __syncwarp();
      if (tile + stride < ntile) prefetch(tile + stride);
    } else {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const long crow = rok[hh] ? rr[hh] % p.P : 0;
#pragma unroll
        for (int gq = 0; gq < 4; ++gq) {
          ux[hh][gq] = uc[hh][gq] = make_uint4(0u, 0u, 0u, 0u);
          if (rok[hh]) {
            ux[hh][gq] = ldg128(p.x + rr[hh] * 128 + 32 * gq + 8 * q4);
            uc[hh][gq] = ldg128(p.cond + crow * 128 + 32 * gq + 8 * q4);
          }
        }
      }
    }

    // ---- aff = rn(LN(cond) w): the A fragments of the two products
    float meanc[2], rstdc[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) { float v[8]; unpack8(uc[hh][gq], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) s += v[i]; }
      meanc[hh] = quad_sum(s) * (1.f / 128.f);
      float q = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) { float v[8]; unpack8(uc[hh][gq], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) { const float d = v[i] - meanc[hh]; q = fmaf(d, d, q); } }
      rstdc[hh] = rsqrtf(quad_sum(q) * (1.f / 128.f) + p.eps_c);
      if (p.cst != nullptr && q4 == 0 && rok[hh]) p.cst[rr[hh]] = make_float2(meanc[hh], rstdc[hh]);
    }
    uint32_t a[8][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const float4 w0 = ldg_f4(p.lnw + 32 * gq + 8 * q4), w1 = ldg_f4(p.lnw + 32 * gq + 8 * q4 + 4);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float v[8], t[8];
        unpack8(uc[hh][gq], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = (v[i] - meanc[hh]) * rstdc[hh] * wv[i];
        a[2 * gq][hh] = pack_bf16(t[0], t[1]);      a[2 * gq][2 + hh] = pack_bf16(t[2], t[3]);
        a[2 * gq + 1][hh] = pack_bf16(t[4], t[5]);  a[2 * gq + 1][2 + hh] = pack_bf16(t[6], t[7]);
      }
    }

    // ---- the statistics of the x rows
    float meanx[2], rstdx[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) { float v[8]; unpack8(ux[hh][gq], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) s += v[i]; }
      meanx[hh] = quad_sum(s) * (1.f / 128.f);
      float q = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) { float v[8]; unpack8(ux[hh][gq], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) { const float d = v[i] - meanx[hh]; q = fmaf(d, d, q); } }
      rstdx[hh] = rsqrtf(quad_sum(q) * (1.f / 128.f) + p.eps_x);
      if (p.xst != nullptr && q4 == 0 && rok[hh]) p.xst[rr[hh]] = make_float2(meanx[hh], rstdx[hh]);
    }

    // ---- per group of 32 output channels: S | B = aff [Ws | Wb]^T, then the gate and the sum
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      float as[4][4], ab[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) { as[t][0] = as[t][1] = as[t][2] = as[t][3] = 0.f; ab[t][0] = ab[t][1] = ab[t][2] = ab[t][3] = 0.f; }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const uint32_t off = (4 * gq + t) * 8 * 256 + wbq[kb];
          const uint4 b1 = lds128_ro(sws + off), b2 = lds128_ro(swb + off);
          mma16816(as[t], a[2 * kb], b1.x, b1.y);
          mma16816(as[t], a[2 * kb + 1], b1.z, b1.w);
          mma16816(ab[t], a[2 * kb], b2.x, b2.y);
          mma16816(ab[t], a[2 * kb + 1], b2.z, b2.w);
        }
      float sbv[8];
      unpack8(ldg128(p.sb + 32 * gq + 8 * q4), sbv);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float xv[8], o[8];
        unpack8(ux[hh][gq], xv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float g = sigmoidf(as[i >> 1][2 * hh + (i & 1)] + sbv[i]);
          o[i] = fmaf(g, (xv[i] - meanx[hh]) * rstdx[hh], ab[i >> 1][2 * hh + (i & 1)]);
        }
        if (rok[hh]) stg128(p.y + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(o));
      }
    }
  }
}

}  // namespace adl
