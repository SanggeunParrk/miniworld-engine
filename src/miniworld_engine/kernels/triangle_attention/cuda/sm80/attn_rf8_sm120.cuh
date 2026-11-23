// attn_rf8_sm120.cuh -- the e4m3 P.V variant of the register-resident triangle-attention core (attn_rf_fwd_sm80.cuh), RTX 5090 (sm_120a), OPT-IN
// (kernels/triangle_attention/cuda/sm120_fp8.py; never reached by default).  Heads of 32 channels, bf16 q / k / bias, the gated epilogue.
//
//   out[z, i, j, h, :] = sigmoid(g) . ( softmax_k( scale q.k + bias[z, h, j, k] ) @ v[z, i, k, h, :] )
//
// What differs from attn_rf_fwd_sm80.cuh:
//   * V arrives as e4m3 bytes (vt_quant_kernel: a standalone pass, V x vinv[h] rounded with cvt.rn.satfinite): per (pair row, head, 32-key block) a tile
//     of [32 channels][32 keys] bytes whose key order inside the block is permuted so that one 8-byte shared load per lane IS the B fragment (b0, b1) of
//     mma.m16n8k32 (e4m3): lane (g = lane >> 2, t = lane & 3) reads channel 8 dn + g, bytes 8 t .. 8 t + 7 = the keys {2t, 2t+1, 8+2t, 9+2t} (b0) and the
//     same + 16 (b1).  The A fragment (P) of the same instruction is the four 8-key columns of the 32-key slot the score tiles already hold: register
//     0 / 1 = slice 0 rows g / g + 8, register 2 / 3 = slice 1, so no data moves between the score and the P V stage;
//   * P = 2^(S c1 - m + 8) with m the running row max (exact, per 32-key slot, the quad agreeing by two shuffles): the largest weight is 2^8 (< 448),
//     e4m3 bytes by cvt.rn.satfinite.e4m3x2; the row sums are an e4m3 ones-column mma over the SAME bytes (numerator and denominator share the rounding);
//     O and the sums are rescaled in fp32 when the max rises (as attn_rf_body<SAFE = true>): no fixed-shift fast path, hence no recomputation kernel;
//   * the output is O x vsc[h] / l (vsc = 1 / vinv).
// Same arithmetic order of the QK^T / bias products as the default core.  One geometry (R = 2 pair rows, 4 warps x 32 queries).
#pragma once
#include "attn_rf_fwd_sm80.cuh"

#include <cuda_fp8.h>

namespace a100 {

#ifndef RF8_ST
#define RF8_ST 2
#endif

DEVI void mma_e4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// four e4m3 bytes, x0 in the lowest
DEVI uint32_t pack_e4m3x4(float x0, float x1, float x2, float x3) {
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

struct Attn8Params {
  AttnParams p;                 // q, k, gate, aout, strides, L, H, scl, key_tiles (v / bias as in AttnParams; v unused)
  const uint8_t* v8;            // [z L rows][H][L / 32][32 channels][32 permuted keys]
  const float* vsc;             // [H]: 1 / vinv, the V scale
};

// ---- V -> e4m3 tiles.  One CTA = one (pair row, 32-key block): reads 32 tokens x (H 32) channels, writes H tiles of 1 KiB.  peak: int32 [2]:
// [0] the running max of |v vinv| (float bits, positive floats order as ints), [1] 1 when a value exceeded 448 (saturated by the cvt).
// Position of key j (0..31) inside a tile: 8 (j >> 1 & 3) + 4 (j >> 4) + 2 (j >> 3 & 1) + (j & 1).
DEVI int rf8_keypos(int j) { return 8 * ((j >> 1) & 3) + 4 * (j >> 4) + 2 * ((j >> 3) & 1) + (j & 1); }

template <int H>
__global__ void __launch_bounds__(256) vt_quant_kernel(const __nv_bfloat16* __restrict__ v, long long ldv, const float* __restrict__ vinv, uint8_t* __restrict__ out, int L, int* peak) {
  __shared__ __align__(16) uint8_t tile[H * 1024];
  const int u = blockIdx.x, row = blockIdx.y;     // key block, pair row (z L + i)
  const int tid = threadIdx.x;
  float pk = 0.f;
  constexpr int CPT = H * 4;                      // 16-byte chunks (8 channels) per token
  for (int c = tid; c < 32 * CPT; c += 256) {
    const int j = c / CPT, cc = c % CPT, h = cc >> 2;
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(v + ((long long)row * (long long)(gridDim.x * 32) + u * 32 + j) * ldv + cc * 8));
    const float inv = __ldg(vinv + h);
    const int pos = rf8_keypos(j);
    const uint32_t* w = &raw.x;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float a = bf16lo(w[e]) * inv, b = bf16hi(w[e]) * inv;
      pk = fmaxf(pk, fmaxf(fabsf(a), fabsf(b)));
      const __nv_fp8_storage_t qa = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3), qb = __nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3);
      const int ch = (cc & 3) * 8 + 2 * e;
      tile[h * 1024 + ch * 32 + pos] = qa;
      tile[h * 1024 + (ch + 1) * 32 + pos] = qb;
    }
  }
  __syncthreads();
  const long long nb = gridDim.x;
  for (int c = tid; c < H * 64; c += 256) {
    const int h = c >> 6, q = c & 63;
    const uint4 val = *reinterpret_cast<const uint4*>(tile + h * 1024 + q * 16);
    *reinterpret_cast<uint4*>(out + (((long long)row * H + h) * nb + u) * 1024 + q * 16) = val;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) pk = fmaxf(pk, __shfl_xor_sync(0xffffffffu, pk, o));
  if ((tid & 31) == 0 && pk > 0.f) {
    atomicMax(peak, __float_as_int(pk));
    if (pk > 448.f) peak[1] = 1;
  }
}

template <class G>
struct F8Lay {   // G = AttnRCfg<1, 2, 4, ST, GATE, 32>: the slot with a 1 KiB e4m3 V tile per pair row
  static constexpr int V8ROW = 1024;
  static constexpr int KOFF = G::BIAS_BYTES, VOFF = KOFF + G::R * G::KV_ROW;
  static constexpr int SLOT_RAW = VOFF + G::R * V8ROW;
  static constexpr int SLOT = SLOT_RAW > G::Q_BYTES ? SLOT_RAW : G::Q_BYTES;   // Q rides in through one slot; the epilogue stages O and g in the ring
  static constexpr int SMEM = G::ST * SLOT;
  static constexpr int V_IT = G::R * 64 / G::NTHR;
  static_assert(G::HD == 32 && G::R * 64 % G::NTHR == 0, "e4m3 core: 32-channel heads, whole 16-byte V chunk steps");
  static_assert(2 * G::Q_BYTES <= G::ST * SLOT, "the epilogue stages O and g in the ring");
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) attn_rf8_kernel(const Attn8Params a8, float sinv_hi, float sinv_lo) {
  using F = F8Lay<G>;
  constexpr int RW = G::RW, ST = G::ST, HD = G::HD, GR = G::GR, KK = G::KK, DN = G::DN;
  static_assert(RW == 2 && DN == 4 && KK == 2, "32-channel heads, two rows per warp");
  const AttnParams& p = a8.p;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wr = warp / G::QG, qg = warp - wr * G::QG;
  const int L = p.L, qt = blockIdx.x, row0 = blockIdx.y * G::R;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  const long long tok0 = (long long)z * L * L;
  const int NU = p.key_tiles ? min(L / G::KEYS, max(1, p.key_tiles[z])) : L / G::KEYS;

  const int ck = (tid / GR) % 32, cg = tid % GR, cr = tid / G::CPR;
  auto kv_row = [](int i) { return (i / G::KV_M) * G::KV_RS; };
  auto kv_key = [](int i) { return (i % G::KV_M) * (G::NTHR / GR); };
  const __nv_bfloat16* ksrc = p.k + (tok0 + (long long)(row0 + cr) * L + ck) * p.ldk + h * HD + cg * 8;
  const __nv_bfloat16* bsrc = p.bias + ((long long)(z * p.H + h) * L + qt * G::BM + (tid >> 2)) * L + (tid & 3) * 8;
  const uint32_t kdst = F::KOFF + cr * G::KV_ROW + rf_swz<HD>(ck, cg), bdst = swz64(tid >> 2, tid & 3);
  const long long kadv = (long long)G::KEYS * p.ldk;
  const long long nkb = L / 32;
  // e4m3 V tile chunks: thread c = tid + NTHR i -> pair row c / 64, 16-byte chunk c % 64 of that row's 1 KiB tile
  const uint8_t* vsrc = a8.v8 + ((((long long)z * L + row0) * p.H + h) * nkb) * 1024;
  auto issue = [&](int u) {
    if (u < NU) {
      const uint32_t s = sbase + (u % ST) * F::SLOT;
#pragma unroll
      for (int i = 0; i < G::KV_IT; ++i) {
        if (G::KV_PART && tid >= G::R * G::CPR) break;
        const uint32_t d = kv_row(i) * G::KV_ROW + kv_key(i) * G::RB;
        cp_async16(s + kdst + d, ksrc + ((long long)kv_row(i) * L + kv_key(i)) * p.ldk);
      }
#pragma unroll
      for (int i = 0; i < F::V_IT; ++i) {
        const int c = tid + G::NTHR * i, r = c >> 6, q = c & 63;
        cp_async16(s + F::VOFF + r * F::V8ROW + q * 16, vsrc + ((long long)r * p.H * nkb + u) * 1024 + q * 16);
      }
#pragma unroll
      for (int i = 0; i < G::B_IT; ++i) cp_async16(s + bdst + i * (G::NTHR / 4) * G::ROWB, bsrc + (long long)i * (G::NTHR / 4) * L);
      ksrc += kadv; bsrc += G::KEYS;
    }
    cp_async_commit();
  };

  {
    const uint32_t qs = sbase + (ST - 1) * F::SLOT;
#pragma unroll
    for (int i = 0; i < G::Q_IT; ++i) {
      const int c = tid + G::NTHR * i, r = c / (G::BM * GR), q = (c / GR) % G::BM, g = c % GR;
      cp_async16(qs + r * G::BM * G::RB + rf_swz<HD>(q, g), p.q + (tok0 + (long long)(row0 + r) * L + qt * G::BM + q) * p.ldq + h * HD + g * 8);
    }
    cp_async_commit();
  }
#pragma unroll
  for (int u = 0; u < ST - 1; ++u) issue(u);

  const int qw = qg * 32;
  const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
  uint32_t qa[RW][2][KK][4];
  cp_async_wait<ST - 1>();
  __syncthreads();
  {
    const uint32_t qs = sbase + (ST - 1) * F::SLOT;
#pragma unroll
    for (int w = 0; w < RW; ++w)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int kk = 0; kk < KK; ++kk)
          ldsm_x4(qa[w][mt][kk], qs + (wr * RW + w) * G::BM * G::RB + rf_swz<HD>(qw + 16 * mt + lrow, 2 * kk + (lane >> 4)));
  }

  const int gq = lane >> 2, tq = lane & 3;
  const uint32_t bs0 = pack_bf16(2 * tq == gq ? sinv_hi : 0.f, 2 * tq + 1 == gq ? sinv_hi : 0.f);
  const uint32_t bs1 = pack_bf16(2 * tq == gq ? sinv_lo : 0.f, 2 * tq + 1 == gq ? sinv_lo : 0.f);
  constexpr uint32_t ONE8 = 0x38383838u;          // e4m3 1.0 x 4
  const float c1 = p.scl;

  float o[RW][2][DN][4], l4[RW][2][4], m[RW][2][2];
#pragma unroll
  for (int w = 0; w < RW; ++w)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        l4[w][mt][e] = 0.f;
#pragma unroll
        for (int dn = 0; dn < DN; ++dn) o[w][mt][dn][e] = 0.f;
      }
      m[w][mt][0] = m[w][mt][1] = -INFINITY;
    }

  const uint32_t a_off = swz64(qw + lrow, 0);
  const uint32_t k_off = rf_swz<HD>(lane & 7, (lane >> 3) & (GR - 1));
  const uint32_t vb_off = (uint32_t)(gq * 32 + 8 * tq);   // + 256 dn + 1024 row: this lane's 8 bytes (b0, b1) of channel 8 dn + g

  auto s_column = [&](uint32_t sb, int jj, float (&sc)[RW][2][4][4]) {
    float cb[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      uint32_t a[4];
      ldsm_x4(a, sb + mt * 16 * G::ROWB + (a_off ^ (jj << 4)));
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      mma_c(cb[mt], a, bs0, bs1, zero);
    }
#pragma unroll
    for (int w = 0; w < RW; ++w) {
      uint32_t kb[4];
      ldsm_x4(kb, sb + F::KOFF + (wr * RW + w) * G::KV_ROW + jj * 8 * G::RB + k_off);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mma_c(sc[w][mt][jj], qa[w][mt][0], kb[0], kb[1], cb[mt]);
        mma_acc(sc[w][mt][jj], qa[w][mt][1], kb[2], kb[3]);
      }
    }
  };

  // P bytes (A fragments of the k32 step) of row w of the slot, after raising the running max; the row sums by the e4m3 ones mma
  auto weights = [&](float (&sc)[RW][2][4][4], int w, uint32_t (&pa)[2][4]) {
    float off[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float mx = fmaxf(fmaxf(fmaxf(sc[w][mt][0][2 * hh], sc[w][mt][0][2 * hh + 1]), fmaxf(sc[w][mt][1][2 * hh], sc[w][mt][1][2 * hh + 1])),
                         fmaxf(fmaxf(sc[w][mt][2][2 * hh], sc[w][mt][2][2 * hh + 1]), fmaxf(sc[w][mt][3][2 * hh], sc[w][mt][3][2 * hh + 1])));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mnew = fmaxf(m[w][mt][hh], mx * c1);
        const float mu = mnew == -INFINITY ? 0.f : mnew;
        const float alpha = ex2f(m[w][mt][hh] - mu);
        m[w][mt][hh] = mnew;
        off[mt][hh] = 8.f - mu;
        l4[w][mt][2 * hh] *= alpha; l4[w][mt][2 * hh + 1] *= alpha;
#pragma unroll
        for (int dn = 0; dn < DN; ++dn) { o[w][mt][dn][2 * hh] *= alpha; o[w][mt][dn][2 * hh + 1] *= alpha; }
      }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float pw[4][4];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) pw[jj][e] = ex2f(fmaf(sc[w][mt][jj][e], c1, off[mt][e >> 1]));
      pa[mt][0] = pack_e4m3x4(pw[0][0], pw[0][1], pw[1][0], pw[1][1]);
      pa[mt][1] = pack_e4m3x4(pw[0][2], pw[0][3], pw[1][2], pw[1][3]);
      pa[mt][2] = pack_e4m3x4(pw[2][0], pw[2][1], pw[3][0], pw[3][1]);
      pa[mt][3] = pack_e4m3x4(pw[2][2], pw[2][3], pw[3][2], pw[3][3]);
      mma_e4(l4[w][mt], pa[mt], ONE8, ONE8);
    }
  };
  auto pv = [&](uint32_t sb, int w, const uint32_t (&pa)[2][4]) {
#pragma unroll
    for (int dn = 0; dn < DN; ++dn) {
      const uint2 b = lds64(sb + F::VOFF + (wr * RW + w) * F::V8ROW + dn * 256 + vb_off);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) mma_e4(o[w][mt][dn], pa[mt], b.x, b.y);
    }
  };

  for (int u = 0; u < NU; ++u) {
    cp_async_wait<ST - 2>();
    __syncthreads();
    issue(u + ST - 1);
    const uint32_t sb = sbase + (u % ST) * F::SLOT;
    float sc[RW][2][4][4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) s_column(sb, jj, sc);
#pragma unroll
    for (int w = 0; w < RW; ++w) {
      uint32_t pa[2][4];
      weights(sc, w, pa);
      pv(sb, w, pa);
    }
  }

  cp_async_wait<0>();
  __syncthreads();
  const uint32_t so = sbase, sg = sbase + G::Q_BYTES;
  auto tile_chunk = [&](int i, int& r, int& q, int& g) { const int c = tid + G::NTHR * i; r = c / (G::BM * GR); q = (c / GR) % G::BM; g = c % GR; };
  if constexpr (G::GATE) {
#pragma unroll
    for (int i = 0; i < G::Q_IT; ++i) {
      int r, q, g;
      tile_chunk(i, r, q, g);
      cp_async16(sg + r * G::BM * G::RB + rf_swz<HD>(q, g), p.gate + (tok0 + (long long)(row0 + r) * L + qt * G::BM + q) * p.ldgate + h * HD + g * 8);
    }
    cp_async_commit();
  }
  const float vsc = __ldg(a8.vsc + h);
#pragma unroll
  for (int w = 0; w < RW; ++w)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float lt = l4[w][mt][2 * hh];
        const float inv = lt > 0.f ? vsc / lt : 0.f;
        const uint32_t row = so + (wr * RW + w) * G::BM * G::RB, qr = qw + 16 * mt + gq + 8 * hh;
#pragma unroll
        for (int dn = 0; dn < DN; ++dn) sts32(row + rf_swz<HD>(qr, dn) + 4 * tq, pack_bf16(o[w][mt][dn][2 * hh] * inv, o[w][mt][dn][2 * hh + 1] * inv));
      }
  if constexpr (G::GATE) cp_async_wait<0>();
  __syncthreads();
#pragma unroll
  for (int i = 0; i < G::Q_IT; ++i) {
    int r, q, g;
    tile_chunk(i, r, q, g);
    const uint32_t a = r * G::BM * G::RB + rf_swz<HD>(q, g);
    const uint4 ov = lds128(so + a);
    const long long tok = tok0 + (long long)(row0 + r) * L + qt * G::BM + q;
    if (!G::GATE || p.out != nullptr) stg128(p.out + tok * p.ldo + h * HD + g * 8, ov);
    if constexpr (G::GATE) {
      const uint4 gv = lds128(sg + a);
      const uint32_t* ow = &ov.x; const uint32_t* gw = &gv.x;
      uint32_t aw[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) aw[e] = pack_bf16(bf16lo(ow[e]) * sigmoid(bf16lo(gw[e])), bf16hi(ow[e]) * sigmoid(bf16hi(gw[e])));
      stg128(p.aout + tok * p.ldao + h * HD + g * 8, make_uint4(aw[0], aw[1], aw[2], aw[3]));
    }
  }
}

namespace {
template <class G>
inline cudaError_t attn_rf8_launch(const Attn8Params& a8, int z, double scl, cudaStream_t stream) {
  using F = F8Lay<G>;
  static bool set[64] = {};
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return e;
  if (!set[dev & 63]) {
    if ((e = cudaFuncSetAttribute(attn_rf8_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, F::SMEM)) != cudaSuccess) return e;
    set[dev & 63] = true;
  }
  const double s = 1.4426950408889634 / scl;
  const float hi = __bfloat162float(__float2bfloat16_rn((float)s)), lo = __bfloat162float(__float2bfloat16_rn((float)(s - hi)));
  const AttnParams& p = a8.p;
  const dim3 grid(p.L / G::BM, p.L / G::R, z * p.H);
  attn_rf8_kernel<G><<<grid, G::NTHR, F::SMEM, stream>>>(a8, hi, lo);
  return cudaGetLastError();
}
}  // namespace

}  // namespace a100
