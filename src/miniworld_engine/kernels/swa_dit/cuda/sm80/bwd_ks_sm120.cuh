// bwd_ks_sm120.cuh -- the FFN backward's dy kernel and the qkvg backward kernel of the SWA atom DiT block for the RTX 5090 (sm_120a; 99 KB of shared memory per block), the K-split form.
// A 16-row tile of 5120 rows is one of only 320: a warp per tile leaves 2 warps on an SM and the mma of one warp alone (512 products of m16n8k16 per tile) ~ 9 us.  Here a tile is FOUR warps ("parts"),
// each summing the K blocks kb = 4 kg + part (kg = 0 .. 3) of the 512 K units, and the CTA holds NW tiles (one group per CTA, grid = the number of groups).  The weight W^T [128 channels][512] streams
// through a 3-stage cp.async ring of 32 KB (stage kg = the four K blocks 4 kg .. 4 kg + 3, 8 KB each, shared by all tiles of the CTA).  The partial sums meet in shared memory (the ring, free after the
// last stage): part p receives the others' partial sums of ITS group of 32 channels (a reduce-scatter in the fixed order of the parts), and the adaLN backward then runs per channel group on every
// part (the row statistic of the dot product is exchanged through shared memory).  The sm_80 headers are not changed.
#pragma once
#include "bwd_stream_sm120.cuh"

namespace sw80 {

template <int NW_, int DBG_ = 0>
struct KsCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 128, KPS = 4, NST = 3, MINB = 1, DBG = DBG_;
  static constexpr int STAGE = 32768, SMEM = NST * STAGE + 2048;
};

// one K-split tile reduce-scatter: part r writes its partial sums of the groups of the other parts (acc[gq] of 4 x 4 per lane) into xb [gq][3][16][32 lanes]
DEVI void ks_scatter(const float (&acc)[4][4][4], float* xb, int part, int lane) {
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
    if (gq != part) {
      const int rp = part < gq ? part : part - 1;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) xb[((gq * 3 + rp) * 16 + j * 4 + e) * 32 + lane] = acc[gq][j][e];
    }
}

// mine[j][e] = sum over the parts r = 0 .. 3 (in this order) of the partial sums of this part's group
DEVI void ks_gather(float (&mine)[4][4], const float (&acc)[4][4][4], const float* xb, int part, int lane) {
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float own = acc[0][j][e];
#pragma unroll
      for (int gq = 1; gq < 4; ++gq) own = gq == part ? acc[gq][j][e] : own;
      float t = 0.f;
#pragma unroll
      for (int r = 0; r < 4; ++r) t += r == part ? own : xb[((part * 3 + (r < part ? r : r - 1)) * 16 + j * 4 + e) * 32 + lane];
      mine[j][e] = t;
    }
}

DEVI uint4 pick4(const uint4 (&v)[4], int i) {
  uint4 r = v[0];
#pragma unroll
  for (int k = 1; k < 4; ++k) r = k == i ? v[k] : r;
  return r;
}

// the adaLN backward on this part's group of 32 channels of a tile (mine[j][2 hh + e]: channel 32 part + 8 q4 + 2 j + e of row g8 + 8 hh); uq = q of all 4 groups, udr = the residual gradient of
// the group; xd [part][16 rows] of the tile in shared memory carries the partial dot products.  ks_pre: what does not depend on the product (the row statistic, the scale), computed while the ring streams.
struct KsPre {
  float rstd[2], sc[2][8], ev[2][8];
};

DEVI void ks_pre(KsPre& k, const RowMap& rm, const uint4 (&uq)[2][4], int part, int q4, const float* mod, int scale_col, float eps) {
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const float* mp = mod + (size_t)rm.mrow[hh] * 768 + scale_col + 32 * part + 8 * q4;
    const float4 s0 = ldg_f4b(mp), s1 = ldg_f4b(mp + 4);
    k.sc[hh][0] = s0.x; k.sc[hh][1] = s0.y; k.sc[hh][2] = s0.z; k.sc[hh][3] = s0.w; k.sc[hh][4] = s1.x; k.sc[hh][5] = s1.y; k.sc[hh][6] = s1.z; k.sc[hh][7] = s1.w;
    float ss = 0.f;
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const uint4 u = uq[hh][gq];
      const float v[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
#pragma unroll
      for (int i = 0; i < 8; ++i) ss = fmaf(v[i], v[i], ss);
    }
    k.rstd[hh] = 1.f / sqrtf(quad_sum(ss) * (1.f / 128.f) + eps);
    const uint4 o = pick4(uq[hh], part);
    k.ev[hh][0] = bf16lo(o.x); k.ev[hh][1] = bf16hi(o.x); k.ev[hh][2] = bf16lo(o.y); k.ev[hh][3] = bf16hi(o.y);
    k.ev[hh][4] = bf16lo(o.z); k.ev[hh][5] = bf16hi(o.z); k.ev[hh][6] = bf16lo(o.w); k.ev[hh][7] = bf16hi(o.w);
  }
}

DEVI void ks_epilogue(const float (&mine)[4][4], const RowMap& rm, const KsPre& k, const uint4 (&udr)[2], int part, int q4, int g8, __nv_bfloat16* dq_out, float* dmod, int shift_col, float* xd) {
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) dot = fmaf(mine[i >> 1][2 * hh + (i & 1)] * (1.f + k.sc[hh][i]), k.ev[hh][i] * k.rstd[hh], dot);
    dot = quad_sum(dot);
    if (q4 == 0) xd[part * 16 + g8 + 8 * hh] = dot;
  }
  __syncthreads();
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int row = g8 + 8 * hh;
    const float mean = (((xd[row] + xd[16 + row]) + xd[32 + row]) + xd[48 + row]) * (1.f / 128.f);
    const uint4 ur = udr[hh];
    const float dr[8] = {bf16lo(ur.x), bf16hi(ur.x), bf16lo(ur.y), bf16hi(ur.y), bf16lo(ur.z), bf16hi(ur.z), bf16lo(ur.w), bf16hi(ur.w)};
    float o[8], w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float xh = k.ev[hh][i] * k.rstd[hh], dx = mine[i >> 1][2 * hh + (i & 1)];
      o[i] = dr[i] + k.rstd[hh] * (dx * (1.f + k.sc[hh][i]) - xh * mean);
      w[i] = dx;
      w[8 + i] = dx * xh;
    }
    if (rm.rok[hh]) {
      stg128(dq_out + (size_t)rm.rr[hh] * 128 + 32 * part + 8 * q4, make_uint4(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]), pack_bf16(o[4], o[5]), pack_bf16(o[6], o[7])));
      float* dm = dmod + (size_t)rm.mrow[hh] * 768 + shift_col + 32 * part + 8 * q4;
      *reinterpret_cast<float4*>(dm) = make_float4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<float4*>(dm + 4) = make_float4(w[4], w[5], w[6], w[7]);
      *reinterpret_cast<float4*>(dm + 128) = make_float4(w[8], w[9], w[10], w[11]);
      *reinterpret_cast<float4*>(dm + 132) = make_float4(w[12], w[13], w[14], w[15]);
    }
  }
}

// the three-stage ring: wait for stage kg, make it visible, and (kg == 1) request stage 3 into the buffer of stage 0
template <class G>
DEVI void ks_ring_wait(int kg, const __nv_bfloat16* wt, uint32_t sring, int tid) {
  if (kg == 0) cp_async_wait<2>();
  else if (kg == 1) cp_async_wait<1>();
  else if (kg == 2) cp_async_wait<1>();
  else cp_async_wait<0>();
  __syncthreads();
  if (kg == 1) {
    bs_load_stage<G>(wt, sring, 3, tid);
    cp_async_commit();
  }
}

template <class G>
DEVI void ks_prologue(const __nv_bfloat16* wt, uint32_t sring, int tid) {
#pragma unroll
  for (int s = 0; s < 3; ++s) {
    bs_load_stage<G>(wt, sring + s * G::STAGE, s, tid);
    cp_async_commit();
  }
}

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, 1) ffn_bwd_dy_ks_kernel(const FfnBwdDyParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int slot = warp >> 2, part = warp & 3;
  const uint32_t wlane = part * 8192u + g8 * 64u + q4 * 16u;
  ks_prologue<G>(p.wut, sring, tid);
  const RowMap rm = row_map<MODE>((int)blockIdx.x * G::NW + slot, p.M, p.S, p.B, g8);
  uint4 ua[4][2], uq[2][4], udr[2];
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
    for (int kg = 0; kg < 4; ++kg) ua[kg][hh] = rm.rok[hh] ? ldg128(p.dab + (size_t)rm.rr[hh] * 512 + 32 * (4 * kg + part) + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) uq[hh][gq] = rm.rok[hh] ? ldg128(p.q1 + (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
    udr[hh] = rm.rok[hh] ? ldg128(p.dy + (size_t)rm.rr[hh] * 128 + 32 * part + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
  }
  float acc[4][4][4];
  KsPre kp;
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
#pragma unroll
  for (int kg = 0; kg < 4; ++kg) {
    ks_ring_wait<G>(kg, p.wut, sring, tid);
    uint32_t a[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { a[0][hh] = ua[kg][hh].x; a[0][2 + hh] = ua[kg][hh].y; a[1][hh] = ua[kg][hh].z; a[1][2 + hh] = ua[kg][hh].w; }
    const uint32_t stg = sring + (kg % 3) * G::STAGE;
#pragma unroll
    for (int nt = 0; nt < (G::DBG == 2 ? 0 : 16); ++nt) {
      const uint4 b = lds128(stg + nt * 512 + wlane);   // volatile: must not sink past the barrier that lets the ring buffer be refilled
      mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
      mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
    }
    if (kg == 0) ks_pre(kp, rm, uq, part, q4, p.mod, 512, p.eps);
  }
  if (G::DBG == 1) { float sm = 0.f; for (int g = 0; g < 4; ++g) for (int j = 0; j < 4; ++j) for (int e = 0; e < 4; ++e) sm += acc[g][j][e]; if (sm == 12345.f) p.dq1[0] = __float2bfloat16(sm); return; }
  __syncthreads();                                              // the ring is free: the reduce-scatter buffer
  float* xb = reinterpret_cast<float*>(smem_raw) + slot * 6144;
  ks_scatter(acc, xb, part, lane);
  __syncthreads();
  float mine[4][4];
  ks_gather(mine, acc, xb, part, lane);
  if (G::DBG == 3) { float sm = 0.f; for (int j = 0; j < 4; ++j) for (int e = 0; e < 4; ++e) sm += mine[j][e]; if (sm == 12345.f) p.dq1[0] = __float2bfloat16(sm); return; }
  ks_epilogue(mine, rm, kp, udr, part, q4, g8, p.dq1, p.dmod, 384, reinterpret_cast<float*>(smem_raw + 3 * G::STAGE) + slot * 64);
}

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, 1) qkvg_bwd_ks_kernel(const QkvgBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int slot = warp >> 2, part = warp & 3;                   // part = the head
  const uint32_t wlane = part * 8192u + g8 * 64u + q4 * 16u;
  ks_prologue<G>(p.wt, sring, tid);
  const RowMap rm = row_map<MODE>((int)blockIdx.x * G::NW + slot, p.M, p.S, p.B, g8);
  float cs[2][8], sn[2][8];
  uint4 cd[4][2], cpv[2][2], uq[2][4], udr[2];
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const float* cp = p.cos + (size_t)rm.mrow[hh] * 16 + 8 * (q4 & 1);
    const float* sp = p.sin + (size_t)rm.mrow[hh] * 16 + 8 * (q4 & 1);
    const float4 c0 = rm.rok[hh] ? ldg_f4b(cp) : make_float4(1.f, 1.f, 1.f, 1.f), c1 = rm.rok[hh] ? ldg_f4b(cp + 4) : make_float4(1.f, 1.f, 1.f, 1.f);
    const float4 s0 = rm.rok[hh] ? ldg_f4b(sp) : make_float4(0.f, 0.f, 0.f, 0.f), s1 = rm.rok[hh] ? ldg_f4b(sp + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    cs[hh][0] = c0.x; cs[hh][1] = c0.y; cs[hh][2] = c0.z; cs[hh][3] = c0.w; cs[hh][4] = c1.x; cs[hh][5] = c1.y; cs[hh][6] = c1.z; cs[hh][7] = c1.w;
    sn[hh][0] = s0.x; sn[hh][1] = s0.y; sn[hh][2] = s0.z; sn[hh][3] = s0.w; sn[hh][4] = s1.x; sn[hh][5] = s1.y; sn[hh][6] = s1.z; sn[hh][7] = s1.w;
    const bool ok = rm.rok[hh];
    const size_t hm = (((size_t)rm.nrow[hh] * 4 + part) * p.S + rm.sat[hh]) * 32 + 8 * q4;
    const size_t rg = (size_t)rm.rr[hh] * 128 + 32 * part + 8 * q4;
    cd[0][hh] = ok ? ldg128(p.dqh + hm) : make_uint4(0u, 0u, 0u, 0u);
    cd[1][hh] = ok ? ldg128(p.dkh + hm) : make_uint4(0u, 0u, 0u, 0u);
    cd[2][hh] = ok ? ldg128(p.dvh + hm) : make_uint4(0u, 0u, 0u, 0u);
    cd[3][hh] = ok ? ldg128(p.dg + rg) : make_uint4(0u, 0u, 0u, 0u);
    cpv[0][hh] = ok ? ldg128(p.pq + rg) : make_uint4(0u, 0u, 0u, 0u);
    cpv[1][hh] = ok ? ldg128(p.pk + rg) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) uq[hh][gq] = ok ? ldg128(p.qi + (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
    udr[hh] = ok ? ldg128(p.dq1 + rg) : make_uint4(0u, 0u, 0u, 0u);
  }
  float acc[4][4][4];
  KsPre kp;
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
#pragma unroll
  for (int kg = 0; kg < 4; ++kg) {                               // kg: 0 q, 1 k, 2 v, 3 g of head `part`
    ks_ring_wait<G>(kg, p.wt, sring, tid);
    uint32_t a[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      uint4 u = cd[kg][hh];
      if (kg < 2) {
        const uint4 up = cpv[kg][hh];
        float f[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
        float t[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float o = __shfl_xor_sync(0xffffffffu, f[i], 2);
          t[i] = q4 < 2 ? fmaf(o, sn[hh][i], f[i] * cs[hh][i]) : fmaf(-o, sn[hh][i], f[i] * cs[hh][i]);
        }
        const float pv[8] = {bf16lo(up.x), bf16hi(up.x), bf16lo(up.y), bf16hi(up.y), bf16lo(up.z), bf16hi(up.z), bf16lo(up.w), bf16hi(up.w)};
        float ssq = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ssq = fmaf(pv[i], pv[i], ssq);
        const float r = 1.f / sqrtf(quad_sum(ssq) * (1.f / 32.f) + p.qk_eps);
        float dd = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) dd = fmaf(t[i], pv[i] * r, dd);
        dd = quad_sum(dd) * (1.f / 32.f);
        float d[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) d[i] = r * (t[i] - (pv[i] * r) * dd);
        u = make_uint4(pack_bf16(d[0], d[1]), pack_bf16(d[2], d[3]), pack_bf16(d[4], d[5]), pack_bf16(d[6], d[7]));
      }
      a[0][hh] = u.x; a[0][2 + hh] = u.y; a[1][hh] = u.z; a[1][2 + hh] = u.w;
      if (rm.rok[hh]) stg128(p.dp + (size_t)rm.rr[hh] * 512 + 128 * kg + 32 * part + 8 * q4, u);
    }
    const uint32_t stg = sring + (kg % 3) * G::STAGE;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const uint4 b = lds128(stg + nt * 512 + wlane);   // volatile: must not sink past the barrier that lets the ring buffer be refilled
      mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
      mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
    }
    if (kg == 0) ks_pre(kp, rm, uq, part, q4, p.mod, 128, p.eps);
  }
  __syncthreads();
  float* xb = reinterpret_cast<float*>(smem_raw) + slot * 6144;
  ks_scatter(acc, xb, part, lane);
  __syncthreads();
  float mine[4][4];
  ks_gather(mine, acc, xb, part, lane);
  ks_epilogue(mine, rm, kp, udr, part, q4, g8, p.dq, p.dmod, 0, reinterpret_cast<float*>(smem_raw + 3 * G::STAGE) + slot * 64);
}

}  // namespace sw80
