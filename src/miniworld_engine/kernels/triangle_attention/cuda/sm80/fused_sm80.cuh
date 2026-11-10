// fused_sm80.cuh -- the fused front / back kernels of the A100 (sm_80) triangle attention for d_pair 64 and 128 and 2 or 4 heads of 32 channels (hidden 64 / 128): the d_pair-128
// kernels of f1 / f3 / b3 / b1_sm80.cuh with the widths as template parameters (the <128, 128> instantiation is the same arithmetic in the same order).
//
//   DIN  = d_pair (64 | 128), DH = the hidden width the core sees (64 | 128: H = DH / 32 heads of 32; a 16-channel head is zero-padded to 32 by the packed weights),
//   NOUT = 4 DH (q | k | v | g channels), ROWS = NOUT + 8 packed rows (the bias heads fill one n tile of 8), KBI = DIN / 32, KBH = DH / 32 (groups of 32 channels).
//
// Every kernel is the layout of the d_pair-128 front (f1_sm80.cuh): a persistent CTA of NW warps per SM, the weights resident in shared memory, each warp walks tiles of tokens on
// its own, the tokens go straight from global memory into the mma A fragments (the k order permuted so a thread holds 8 consecutive channels: 16-byte loads), the packed weight
// rows follow ``f1_channel`` so the accumulators hold 8 consecutive output channels per thread (16-byte stores).  Weight rows are DIN * 2 bytes (the front, the front's backward's
// Wp^T rows are DH * 2 ...): ``fwoff`` is f1_woff with the row pitch as a parameter (the 16-byte chunk index is XOR-ed with 4 on odd rows: conflict-free fragment loads for rows of
// 128 B and more).
#pragma once
#include "f3_sm80.cuh"
#include "b3_sm80.cuh"
#include "b1_sm80.cuh"
#include "wgrad_sm80.cuh"

namespace a100 {

// HD = the head dim of the layout the kernels write and read: 32 (a head is 32 channels; a 16-channel head is zero-padded to it by the packed weights) or 16 (native: the heads of
// 16 channels are adjacent, DH = H 16 -- the attention core reads them with head dim 16).  H = DH / HD is the number of heads either way.
template <int DIN_, int DH_, int NW_, int MINB_, int HD_ = 32>
struct FusedCfg {
  static constexpr int DIN = DIN_, DH = DH_, HD = HD_, H = DH_ / HD_, NW = NW_, NTHR = NW_ * 32, MINB = MINB_;
  static_assert(DIN % 64 == 0 && DH % 32 == 0 && H >= 2 && H % 2 == 0 && (HD == 16 || HD == 32), "widths");
  static constexpr int KBI = DIN / 32, KBH = DH / 32;           // groups of 32 channels of the input / hidden width
  static constexpr int NOUT = 4 * DH, NBLK = NOUT / 64;         // q | k | v | g channels and their blocks of 64
  static constexpr int ROWS = NOUT + 8;                         // packed rows of the front's weights
  static constexpr int NTO = DIN / 8, NTH = DH / 8;             // n tiles of the input / hidden width
  static constexpr int F1_SMEM = ROWS * DIN * 2 + ROWS * 4;     // W' rows of DIN * 2 bytes + the shift
  static constexpr int F3_SMEM = DIN * DH * 2;                  // Wo' : DIN packed rows of DH * 2 bytes
  static constexpr int B3_SMEM = DH * DIN * 2;                  // Wo^T packed: DH rows of DIN * 2 bytes
  static constexpr int B1_PB = DIN * DH * 2;                    // one Wp^T block: DIN rows of DH * 2 bytes
  static constexpr int B1_SMEM = 4 * B1_PB + DIN * H * 2 + DIN * 4;   // 4 x Wp^T + Wb^T (H bf16 per input channel) + gamma
};

// byte offset of 16-byte chunk `chunk` of packed row `row` (rows of `rowb` bytes)
template <int ROWB>
DEVI uint32_t fwoff(uint32_t row, uint32_t chunk) { return row * ROWB + ((chunk ^ ((row & 1u) << 2)) << 4); }

// ------------------------------------------------------------------------------------------------------------------------------------------------------------ pack
// W'[s] = bf16(W[c(s)] diag gamma), b[c] = W[c] . beta for the 4 DH channels of Wq | Wk | Wv | Wg (head h, channel c < hd real, the rest of a head zero rows) and the bias heads.
struct FusedPackParams {
  const __nv_bfloat16* w[5];    // wq, wk, wv, wg [H hd][DIN]; wb [H][DIN]
  const float* gamma;
  const float* beta;
  __nv_bfloat16* wp;            // [ROWS][DIN]
  float* bvec;                  // [ROWS]
  int DIN, DH, H, hd, slot;    // slot = channels per head slot of the layout: 32 (a head padded to 32) or hd (native heads)
};

__global__ void __launch_bounds__(256) fused_pack_kernel(const FusedPackParams p) {
  const int NOUT = 4 * p.DH, ROWS = NOUT + 8;
  const int s = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (s >= ROWS) return;
  const int c = s < NOUT ? f1_channel(s) : (s < NOUT + p.H ? s : -1);       // channel (NOUT + head for the bias rows), -1: padding
  const __nv_bfloat16* src = nullptr;
  if (c >= 0 && c < NOUT) {
    const int pj = c / p.DH, r = c - pj * p.DH, h = r / p.slot, ch = r % p.slot;
    if (h < p.H && ch < p.hd) src = p.w[pj] + (size_t)(h * p.hd + ch) * p.DIN;
  } else if (c >= NOUT) {
    src = p.w[4] + (size_t)(c - NOUT) * p.DIN;
  }
  float dotb = 0.f;
  for (int c2 = lane; c2 < p.DIN / 2; c2 += 32) {
    uint32_t u = 0u;
    if (src != nullptr) {
      const uint32_t raw = reinterpret_cast<const uint32_t*>(src)[c2];
      const float w0 = bf16lo(raw), w1 = bf16hi(raw);
      const float2 g = reinterpret_cast<const float2*>(p.gamma)[c2], b = reinterpret_cast<const float2*>(p.beta)[c2];
      u = pack_bf16(w0 * g.x, w1 * g.y);
      dotb = fmaf(w1, b.y, fmaf(w0, b.x, dotb));
    }
    reinterpret_cast<uint32_t*>(p.wp + (size_t)s * p.DIN)[c2] = u;
  }
  dotb = warp_sum(dotb);
  if (lane == 0) p.bvec[c >= 0 ? c : s] = src != nullptr ? dotb : 0.f;
}

// The backward's weights in its kernels' layouts: wt[p][c][o'] = W_p[o][c] (o' = padded hidden channel, zero where padded), wbt[c][h] = Wb[h][c], wot[o'][c] = Wo[c][o], gamma32.
struct FusedBwdPackParams {
  const __nv_bfloat16* w[6];    // wq, wk, wv, wg [H hd][DIN]; wb [H][DIN]; wo [DIN][H hd]
  const void* gamma;
  int gamma_bf16;
  __nv_bfloat16* wt;            // [4][DIN][DH]
  __nv_bfloat16* wbt;           // [DIN][H]
  __nv_bfloat16* wot;           // [DH][DIN]
  float* gamma32;               // [DIN]
  int DIN, DH, H, hd, slot;    // slot = channels per head slot of the layout: 32 (a head padded to 32) or hd (native heads)
};

__global__ void __launch_bounds__(256) fused_bwd_pack_kernel(const FusedBwdPackParams p) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t n4 = (size_t)4 * p.DIN * p.DH, nw = (size_t)p.DH * p.DIN, nb = (size_t)p.DIN * p.H;
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.f);
  if (i < n4) {
    const int pj = (int)(i / ((size_t)p.DIN * p.DH)), rem = (int)(i - (size_t)pj * p.DIN * p.DH), c = rem / p.DH, o = rem - c * p.DH;       // write wt[pj][c][o]
    const int h = o / p.slot, ch = o % p.slot;
    p.wt[i] = (h < p.H && ch < p.hd) ? p.w[pj][(size_t)(h * p.hd + ch) * p.DIN + c] : zero;
  } else if (i < n4 + nw) {
    const size_t j = i - n4;
    const int o = (int)(j / p.DIN), c = (int)(j - (size_t)o * p.DIN);                                                                       // wot[o][c] = Wo[c][o_real]
    const int h = o / p.slot, ch = o % p.slot;
    p.wot[j] = (h < p.H && ch < p.hd) ? p.w[5][(size_t)c * (p.H * p.hd) + h * p.hd + ch] : zero;
  } else if (i < n4 + nw + nb) {
    const size_t j = i - n4 - nw;
    const int c = (int)(j / p.H), h = (int)(j - (size_t)c * p.H);
    p.wbt[j] = p.w[4][(size_t)h * p.DIN + c];
  } else if (i < n4 + nw + nb + p.DIN) {
    const int c = (int)(i - n4 - nw - nb);
    p.gamma32[c] = p.gamma_bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p.gamma)[c]) : reinterpret_cast<const float*>(p.gamma)[c];
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------ the front
template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) fused_f1_kernel(const F1Params p) {
  constexpr int DIN = G::DIN, KBI = G::KBI, NOUT = G::NOUT, H = G::H, ROWB = DIN * 2, CHR = DIN / 8;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* bv = reinterpret_cast<float*>(smem_raw + G::ROWS * ROWB);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < G::ROWS * CHR; i += G::NTHR) cp_async16(sb + fwoff<ROWB>(i / CHR, i % CHR), p.w + (size_t)(i / CHR) * DIN + (i % CHR) * 8);
  cp_async_commit();
  for (int i = tid; i < G::ROWS; i += G::NTHR) bv[i] = p.bvec[i];
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[KBI];
#pragma unroll
  for (int kb = 0; kb < KBI; ++kb) wb[kb] = sb + fwoff<ROWB>(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    const unsigned t0 = tile * 32, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;

    uint4 xr[2][2][KBI];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const unsigned r = mt * 16 + g8 + 8 * hh;
        const size_t src = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
#pragma unroll
        for (int kb = 0; kb < KBI; ++kb) xr[mt][hh][kb] = ldg128(p.x + src * DIN + 32 * kb + 8 * q4);
      }
    bool keep[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) keep[mt][hh] = p.mask == nullptr || p.mask[z * L + b0 + mt * 16 + g8 + 8 * hh] != 0;

    uint32_t a[2][2 * KBI][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float s = 0.f;
#pragma unroll
        for (int kb = 0; kb < KBI; ++kb) {
          const uint4 u = xr[mt][hh][kb];
          s += (bf16lo(u.x) + bf16hi(u.x)) + (bf16lo(u.y) + bf16hi(u.y)) + (bf16lo(u.z) + bf16hi(u.z)) + (bf16lo(u.w) + bf16hi(u.w));
        }
        const float mean = quad_sum(s) * (1.f / DIN);
        float v = 0.f;
#pragma unroll
        for (int kb = 0; kb < KBI; ++kb) {
          const uint4 u = xr[mt][hh][kb];
          float d;
          d = bf16lo(u.x) - mean; v = fmaf(d, d, v);  d = bf16hi(u.x) - mean; v = fmaf(d, d, v);
          d = bf16lo(u.y) - mean; v = fmaf(d, d, v);  d = bf16hi(u.y) - mean; v = fmaf(d, d, v);
          d = bf16lo(u.z) - mean; v = fmaf(d, d, v);  d = bf16hi(u.z) - mean; v = fmaf(d, d, v);
          d = bf16lo(u.w) - mean; v = fmaf(d, d, v);  d = bf16hi(u.w) - mean; v = fmaf(d, d, v);
        }
        const float rstd = rsqrtf(quad_sum(v) * (1.f / DIN) + p.eps);
        if (p.lnst != nullptr && q4 == 0) {
          const size_t tok = (size_t)t0 + mt * 16 + g8 + 8 * hh;
          *reinterpret_cast<float2*>(p.lnst + 2 * tok) = make_float2(mean, rstd);
        }
        if (p.xh != nullptr && q4 == 0) stg128(p.xh + ((size_t)t0 + mt * 16 + g8 + 8 * hh) * (DIN + 8) + DIN, make_uint4(0x3f80u, 0u, 0u, 0u));
#pragma unroll
        for (int kb = 0; kb < KBI; ++kb) {
          const uint4 u = xr[mt][hh][kb];
          const uint32_t n0 = pack_bf16((bf16lo(u.x) - mean) * rstd, (bf16hi(u.x) - mean) * rstd);
          const uint32_t n1 = pack_bf16((bf16lo(u.y) - mean) * rstd, (bf16hi(u.y) - mean) * rstd);
          const uint32_t n2 = pack_bf16((bf16lo(u.z) - mean) * rstd, (bf16hi(u.z) - mean) * rstd);
          const uint32_t n3 = pack_bf16((bf16lo(u.w) - mean) * rstd, (bf16hi(u.w) - mean) * rstd);
          a[mt][2 * kb][hh] = n0;      a[mt][2 * kb][2 + hh] = n1;
          a[mt][2 * kb + 1][hh] = n2;  a[mt][2 * kb + 1][2 + hh] = n3;
          if (p.xh != nullptr) stg128(p.xh + ((size_t)t0 + mt * 16 + g8 + 8 * hh) * (DIN + 8) + 32 * kb + 8 * q4, make_uint4(n0, n1, n2, n3));
        }
      }

#pragma unroll 1
    for (int blk = 0; blk < G::NBLK; ++blk) {
      float acc[2][8][4];
#pragma unroll
      for (int og = 0; og < 2; ++og) {
        const float4 b0v = *reinterpret_cast<const float4*>(bv + 64 * blk + 32 * og + 8 * q4);
        const float4 b1v = *reinterpret_cast<const float4*>(bv + 64 * blk + 32 * og + 8 * q4 + 4);
        const float bb[8] = {b0v.x, b0v.y, b0v.z, b0v.w, b1v.x, b1v.y, b1v.z, b1v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            acc[mt][4 * og + j][0] = acc[mt][4 * og + j][2] = bb[2 * j];
            acc[mt][4 * og + j][1] = acc[mt][4 * og + j][3] = bb[2 * j + 1];
          }
      }
      const uint32_t wblk = 64u * blk * ROWB;
      uint4 bn = lds128_ro(wb[0] + wblk);
#pragma unroll
      for (int kb = 0; kb < KBI; ++kb)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint4 bc = bn;
          constexpr int last = KBI * 8 - 1;
          if (kb * 8 + nt < last) bn = lds128_ro(wb[(kb * 8 + nt + 1) >> 3] + wblk + ((kb * 8 + nt + 1) & 7) * 8 * ROWB);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc[mt][nt], a[mt][2 * kb], bc.x, bc.y);
            mma16816(acc[mt][nt], a[mt][2 * kb + 1], bc.z, bc.w);
          }
        }
#pragma unroll
      for (int og = 0; og < 2; ++og)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const size_t tok = (size_t)t0 + mt * 16 + g8 + 8 * hh;
            uint4 v;
            v.x = pack_bf16(acc[mt][4 * og][2 * hh], acc[mt][4 * og][2 * hh + 1]);
            v.y = pack_bf16(acc[mt][4 * og + 1][2 * hh], acc[mt][4 * og + 1][2 * hh + 1]);
            v.z = pack_bf16(acc[mt][4 * og + 2][2 * hh], acc[mt][4 * og + 2][2 * hh + 1]);
            v.w = pack_bf16(acc[mt][4 * og + 3][2 * hh], acc[mt][4 * og + 3][2 * hh + 1]);
            stg128(p.out + tok * NOUT + 64 * blk + 32 * og + 8 * q4, v);
          }
    }

    {                                                            // the bias heads: rows NOUT .. NOUT + 7 are one n tile (columns 2 q4, 2 q4 + 1 = heads; only H are real)
      float acc[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        acc[mt][0] = acc[mt][2] = bv[NOUT + 2 * q4];
        acc[mt][1] = acc[mt][3] = bv[NOUT + 2 * q4 + 1];
      }
#pragma unroll
      for (int kb = 0; kb < KBI; ++kb) {
        const uint4 b = lds128_ro(wb[kb] + (uint32_t)NOUT * ROWB);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc[mt], a[mt][2 * kb], b.x, b.y);
          mma16816(acc[mt], a[mt][2 * kb + 1], b.z, b.w);
        }
      }
      if (2 * q4 < H) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const float v = acc[mt][2 * hh + e];
              p.bias[(size_t)(z * H + 2 * q4 + e) * LL + rem0 + mt * 16 + g8 + 8 * hh] =
                  keep[mt][hh] ? __float2bfloat16_rn(v) : __float2bfloat16_rn(-3.3895313892515355e38f);
            }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------ the back
template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) fused_f3_kernel(const F3Params p) {
  constexpr int DIN = G::DIN, DH = G::DH, KBH = G::KBH, NOG = G::KBI, NTO = G::NTO, ROWB = DH * 2, CHR = DH / 8;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < DIN * CHR; i += G::NTHR) cp_async16(sb + fwoff<ROWB>(i / CHR, i % CHR), p.wo + (size_t)f1_channel(i / CHR) * DH + (i % CHR) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[KBH];
#pragma unroll
  for (int kb = 0; kb < KBH; ++kb) wb[kb] = sb + fwoff<ROWB>(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    const unsigned t0 = tile * 16, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;
    size_t drow[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const unsigned r = g8 + 8 * hh;
      drow[hh] = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
    }

    uint4 ov[2][KBH], gv[2][KBH], rv[2][NOG];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < KBH; ++kb) {
        const size_t tok = (size_t)t0 + g8 + 8 * hh;
        ov[hh][kb] = ldg128(p.o + tok * DH + 32 * kb + 8 * q4);
        gv[hh][kb] = ldg128(p.g + tok * p.ldg + 32 * kb + 8 * q4);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int og = 0; og < NOG; ++og) rv[hh][og] = ldg128(p.res + drow[hh] * DIN + 32 * og + 8 * q4);

    uint32_t a[2 * KBH][4];
#pragma unroll
    for (int kb = 0; kb < KBH; ++kb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4 o = ov[hh][kb], g = gv[hh][kb];
        a[2 * kb][hh] = gate_pair(o.x, g.x);      a[2 * kb][2 + hh] = gate_pair(o.y, g.y);
        a[2 * kb + 1][hh] = gate_pair(o.z, g.z);  a[2 * kb + 1][2 + hh] = gate_pair(o.w, g.w);
      }

    float acc[NTO][4];
#pragma unroll
    for (int nt = 0; nt < NTO; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    uint4 bn = lds128_ro(wb[0]);
#pragma unroll
    for (int kb = 0; kb < KBH; ++kb)
#pragma unroll
      for (int nt = 0; nt < NTO; ++nt) {
        const uint4 bc = bn;
        if (kb * NTO + nt < KBH * NTO - 1) bn = lds128_ro(wb[(kb * NTO + nt + 1) / NTO] + ((kb * NTO + nt + 1) % NTO) * 8 * ROWB);
        mma16816(acc[nt], a[2 * kb], bc.x, bc.y);
        mma16816(acc[nt], a[2 * kb + 1], bc.z, bc.w);
      }

#pragma unroll
    for (int og = 0; og < NOG; ++og)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4 r = rv[hh][og];
        uint32_t y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) y[j] = pack_bf16(acc[4 * og + j][2 * hh], acc[4 * og + j][2 * hh + 1]);
        if (p.ds != nullptr) {
          const uint4 d = ldg128(p.ds + ((size_t)z * L + b0 + g8 + 8 * hh) * DIN + 32 * og + 8 * q4);
          y[0] = mul_bf16x2(y[0], d.x); y[1] = mul_bf16x2(y[1], d.y); y[2] = mul_bf16x2(y[2], d.z); y[3] = mul_bf16x2(y[3], d.w);
        }
        uint4 v;
        v.x = add_bf16x2(r.x, y[0]);
        v.y = add_bf16x2(r.y, y[1]);
        v.z = add_bf16x2(r.z, y[2]);
        v.w = add_bf16x2(r.w, y[3]);
        stg128(p.out + drow[hh] * DIN + 32 * og + 8 * q4, v);
      }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------ the back's backward
template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) fused_b3_kernel(const B3Params p) {
  constexpr int DIN = G::DIN, DH = G::DH, H = G::H, KBI = G::KBI, KBH = G::KBH, NTH = G::NTH, ROWB = DIN * 2, CHR = DIN / 8;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < DH * CHR; i += G::NTHR) cp_async16(sb + fwoff<ROWB>(i / CHR, i % CHR), p.wot + (size_t)f1_channel(i / CHR) * DIN + (i % CHR) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[KBI];
#pragma unroll
  for (int kb = 0; kb < KBI; ++kb) wb[kb] = sb + fwoff<ROWB>(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    const unsigned t0 = tile * 16, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;
    size_t drow[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const unsigned r = g8 + 8 * hh;
      drow[hh] = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
    }

    uint4 dv[2][KBI], ov[2][KBH], gv[2][KBH];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
      for (int kb = 0; kb < KBI; ++kb) dv[hh][kb] = ldg128(p.dout + drow[hh] * DIN + 32 * kb + 8 * q4);
#pragma unroll
      for (int kb = 0; kb < KBH; ++kb) {
        const size_t tok = (size_t)t0 + g8 + 8 * hh;
        ov[hh][kb] = ldg128(p.o + tok * DH + 32 * kb + 8 * q4);
        gv[hh][kb] = ldg128(p.g + tok * p.ldg + 32 * kb + 8 * q4);
      }
    }
    if (p.ds != nullptr) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int kb = 0; kb < KBI; ++kb) {
          const uint4 d = ldg128(p.ds + ((size_t)z * L + b0 + g8 + 8 * hh) * DIN + 32 * kb + 8 * q4);
          uint4& u = dv[hh][kb];
          u.x = mul_bf16x2(u.x, d.x); u.y = mul_bf16x2(u.y, d.y); u.z = mul_bf16x2(u.z, d.z); u.w = mul_bf16x2(u.w, d.w);
        }
    }

    uint32_t a[2 * KBI][4];
#pragma unroll
    for (int kb = 0; kb < KBI; ++kb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4 u = dv[hh][kb];
        a[2 * kb][hh] = u.x;      a[2 * kb][2 + hh] = u.y;
        a[2 * kb + 1][hh] = u.z;  a[2 * kb + 1][2 + hh] = u.w;
      }

    float acc[NTH][4];
#pragma unroll
    for (int nt = 0; nt < NTH; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    uint4 bn = lds128_ro(wb[0]);
#pragma unroll
    for (int kb = 0; kb < KBI; ++kb)
#pragma unroll
      for (int nt = 0; nt < NTH; ++nt) {
        const uint4 bc = bn;
        if (kb * NTH + nt < KBI * NTH - 1) bn = lds128_ro(wb[(kb * NTH + nt + 1) / NTH] + ((kb * NTH + nt + 1) % NTH) * 8 * ROWB);
        mma16816(acc[nt], a[2 * kb], bc.x, bc.y);
        mma16816(acc[nt], a[2 * kb + 1], bc.z, bc.w);
      }

#pragma unroll
    for (int og = 0; og < KBH; ++og)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const size_t tok = (size_t)t0 + g8 + 8 * hh;
        const uint4 ou = ov[hh][og], gu = gv[hh][og];
        const uint32_t ow[4] = {ou.x, ou.y, ou.z, ou.w}, gw[4] = {gu.x, gu.y, gu.z, gu.w};
        uint32_t d_o[4], d_g[4], av[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float da0 = round_bf16f(acc[4 * og + j][2 * hh]), da1 = round_bf16f(acc[4 * og + j][2 * hh + 1]);
          const float o0 = bf16lo(ow[j]), o1 = bf16hi(ow[j]);
          const float s0 = sigmoid(bf16lo(gw[j])), s1 = sigmoid(bf16hi(gw[j]));
          d_o[j] = pack_bf16(da0 * s0, da1 * s1);
          d_g[j] = pack_bf16(da0 * o0 * (s0 * (1.f - s0)), da1 * o1 * (s1 * (1.f - s1)));
          av[j] = pack_bf16(o0 * s0, o1 * s1);
        }
        const size_t col = 32 * og + 8 * q4;
        stg128(p.dov + tok * DH + col, make_uint4(d_o[0], d_o[1], d_o[2], d_o[3]));
        stg128(p.dg + tok * p.lddg + col, make_uint4(d_g[0], d_g[1], d_g[2], d_g[3]));
        stg128(p.a + tok * DH + col, make_uint4(av[0], av[1], av[2], av[3]));
        if (p.delta != nullptr) {
          float dl = 0.f;
#pragma unroll
          for (int j = 0; j < 4; ++j) dl = fmaf(bf16lo(ow[j]), bf16lo(d_o[j]), fmaf(bf16hi(ow[j]), bf16hi(d_o[j]), dl));
          if constexpr (G::HD == 32) {                               // a head = the 32 channels of the 4 lanes of a quad
            dl = quad_sum(dl);
            if (q4 == 0) p.delta[(size_t)(z * H + og) * LL + rem0 + g8 + 8 * hh] = dl;
          } else {                                                   // heads of 16 channels: a pair of lanes each, head 2 og + (q4 >> 1)
            dl += __shfl_xor_sync(0xffffffffu, dl, 1);
            if ((q4 & 1) == 0) p.delta[(size_t)(z * H + 2 * og + (q4 >> 1)) * LL + rem0 + g8 + 8 * hh] = dl;
          }
        }
      }
#pragma unroll
    for (int kb = 0; kb < KBI; ++kb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
        stg128(p.dy + ((size_t)t0 + g8 + 8 * hh) * DIN + 32 * kb + 8 * q4, make_uint4(a[2 * kb][hh], a[2 * kb][2 + hh], a[2 * kb + 1][hh], a[2 * kb + 1][2 + hh]));
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------ the front's backward
template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) fused_b1_kernel(const B1Params p) {
  constexpr int DIN = G::DIN, DH = G::DH, H = G::H, KBI = G::KBI, KBH = G::KBH, NTO = G::NTO, ROWB = DH * 2, CHR = DH / 8, PB = G::B1_PB, NOG = G::KBI;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  unsigned char* wbt_s = smem_raw + 4 * PB;
  float* gam = reinterpret_cast<float*>(smem_raw + 4 * PB + DIN * H * 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 4 * DIN * CHR; i += G::NTHR) {
    const int blk = i / (DIN * CHR), row = (i / CHR) % DIN, ch = i % CHR;
    cp_async16(sb + blk * PB + fwoff<ROWB>(row, ch), p.wt + ((size_t)blk * DIN + f1_channel(row)) * DH + ch * 8);
  }
  cp_async_commit();
  for (int i = tid; i < DIN * H; i += G::NTHR) reinterpret_cast<__nv_bfloat16*>(wbt_s)[i] = p.wbt[(size_t)f1_channel(i / H) * H + (i % H)];
  for (int i = tid; i < DIN; i += G::NTHR) gam[i] = p.gamma[i];
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[KBH];
#pragma unroll
  for (int kb = 0; kb < KBH; ++kb) wb[kb] = sb + fwoff<ROWB>(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    const unsigned t0 = tile * 16, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;
    size_t drow[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const unsigned r = g8 + 8 * hh;
      drow[hh] = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
    }

    float acc[NTO][4];
#pragma unroll
    for (int nt = 0; nt < NTO; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

    uint4 av[2][KBH];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < KBH; ++kb) av[hh][kb] = ldg128(p.d + ((size_t)t0 + g8 + 8 * hh) * p.ldd + 32 * kb + 8 * q4);
    uint4 xv[2][KBI], ov[2][KBI];
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
      uint32_t a[2 * KBH][4];
#pragma unroll
      for (int kb = 0; kb < KBH; ++kb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const uint4 u = av[hh][kb];
          a[2 * kb][hh] = u.x;      a[2 * kb][2 + hh] = u.y;
          a[2 * kb + 1][hh] = u.z;  a[2 * kb + 1][2 + hh] = u.w;
        }
      if (blk < 3) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int kb = 0; kb < KBH; ++kb) av[hh][kb] = ldg128(p.d + ((size_t)t0 + g8 + 8 * hh) * p.ldd + DH * (blk + 1) + 32 * kb + 8 * q4);
      } else {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int kb = 0; kb < KBI; ++kb) {
            xv[hh][kb] = ldg128(p.x + drow[hh] * DIN + 32 * kb + 8 * q4);
            ov[hh][kb] = ldg128(p.dout + drow[hh] * DIN + 32 * kb + 8 * q4);
          }
      }
      uint4 bn = lds128_ro(wb[0] + blk * PB);
#pragma unroll
      for (int kb = 0; kb < KBH; ++kb)
#pragma unroll
        for (int nt = 0; nt < NTO; ++nt) {
          const uint4 bc = bn;
          if (kb * NTO + nt < KBH * NTO - 1) bn = lds128_ro(wb[(kb * NTO + nt + 1) / NTO] + blk * PB + ((kb * NTO + nt + 1) % NTO) * 8 * ROWB);
          mma16816(acc[nt], a[2 * kb], bc.x, bc.y);
          mma16816(acc[nt], a[2 * kb + 1], bc.z, bc.w);
        }
    }
    {                                                            // the bias heads: one k step (k = head, H of 16 real: lanes q4 < H / 2 hold heads 2 q4, 2 q4 + 1)
      uint32_t adb[4] = {0u, 0u, 0u, 0u};
      if (2 * q4 < H) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const size_t at = (size_t)(z * H + 2 * q4) * LL + rem0 + g8 + 8 * hh;
          const uint32_t lo = __bfloat16_as_ushort(p.db[at]), hi = __bfloat16_as_ushort(p.db[at + LL]);
          adb[hh] = lo | (hi << 16);
        }
      }
#pragma unroll
      for (int nt = 0; nt < NTO; ++nt) {
        const uint32_t b0v = 2 * q4 < H ? lds32(sb + 4 * PB + (nt * 8 + g8) * (H * 2) + 4 * q4) : 0u;
        mma16816(acc[nt], adb, b0v, 0u);
      }
    }

#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t tok = (size_t)t0 + g8 + 8 * hh;
      const float2 st = *reinterpret_cast<const float2*>(p.lnst + 2 * tok);
      const float mean = st.x, rstd = st.y;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int og = 0; og < NOG; ++og) {
        const uint4 xu = xv[hh][og];
        const uint32_t xw[4] = {xu.x, xu.y, xu.z, xu.w};
        const float4 g0 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4), g1 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4 + 4);
        const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float x0 = bf16lo(xw[j]), x1 = bf16hi(xw[j]);
          const float d0 = round_bf16f(acc[4 * og + j][2 * hh]) * gg[2 * j], d1 = round_bf16f(acc[4 * og + j][2 * hh + 1]) * gg[2 * j + 1];
          s1 += d0 + d1;
          s2 = fmaf(d0, (x0 - mean) * rstd, fmaf(d1, (x1 - mean) * rstd, s2));
        }
      }
      const float m1 = quad_sum(s1) * (1.f / DIN), m2 = quad_sum(s2) * (1.f / DIN);
#pragma unroll
      for (int og = 0; og < NOG; ++og) {
        const uint4 xu = xv[hh][og], ou = ov[hh][og];
        const uint32_t xw[4] = {xu.x, xu.y, xu.z, xu.w}, ow[4] = {ou.x, ou.y, ou.z, ou.w};
        const float4 g0 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4), g1 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4 + 4);
        const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        uint32_t r[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float xh0 = (bf16lo(xw[j]) - mean) * rstd, xh1 = (bf16hi(xw[j]) - mean) * rstd;
          const float d0 = round_bf16f(acc[4 * og + j][2 * hh]) * gg[2 * j], d1 = round_bf16f(acc[4 * og + j][2 * hh + 1]) * gg[2 * j + 1];
          r[j] = add_bf16x2(ow[j], pack_bf16(rstd * (d0 - m1 - xh0 * m2), rstd * (d1 - m1 - xh1 * m2)));
        }
        stg128(p.dpair + drow[hh] * DIN + 32 * og + 8 * q4, make_uint4(r[0], r[1], r[2], r[3]));
      }
    }
  }
}

}  // namespace a100
