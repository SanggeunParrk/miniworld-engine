// f1_sm80.cuh -- the front of the triangle attention, A100 / sm_80: input LayerNorm + the q | k | v | g projections (4 x 128 outputs) + the pair-bias
// projection (4 outputs) in one pass over the pair tensor.
//
//   xh = (x - mean) rstd                       per token, fp32 statistics, rounded to bf16 as the operand of the products
//   [q | k | v | g] = xh . W'^T + b            W' = bf16(W diag gamma), b = W beta   (the LayerNorm affine folded into the weights)
//   bias[h, t] = xh . Wb'[h]^T + bb[h]         head planes, masked keys = bf16 min (the module's masked_fill)
//
// Token t = (a, b) of the "starting" problem; ``transposed`` reads it from row (b L + a) of x (the ending node, without a transposing copy).
//
// One persistent CTA per SM of NW warps.  The packed weights W' (130 KiB) stay in shared memory for the CTA's life and every warp runs its own
// loop over tiles of 32 tokens (no CTA barrier after the start): the tokens go straight from global memory into the A fragments of mma.sync (two
// m16 tiles), LayerNorm is done on them in registers, then 9 blocks of weights: the B fragments are 16-byte shared loads, the accumulators start
// from the shift b, and the epilogue stores 16 bytes per thread (STG.128) straight from the accumulators.
//
// Two permutations make the 16-byte accesses possible, and cost nothing: the k order of the products (a thread's two k steps of 16 hold 8 consecutive
// channels of a row: A and B take their halves of the same 8) and the n order (the packed weight rows of a block of 64 channels are laid out so that
// the accumulator pair of thread q4 in the 4 n tiles of a group is 8 consecutive output channels; ``f1_channel``).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

constexpr int F1_ROWS = 520;   // packed weight rows: 512 (q | k | v | g) and the 4 bias rows, padded to an n tile

struct F1Params {
  const __nv_bfloat16* x;       // [T][128]
  const __nv_bfloat16* w;       // [520][128] W' in the kernel's row order (f1_channel), rows 516-519 zero
  const float* bvec;            // [520] b = W beta by channel (512 + head: the bias rows; 516-519 zero)
  const uint8_t* mask;          // [Z][L] key mask (non-zero = real key) or nullptr
  __nv_bfloat16* out;           // [T][512] q | k | v | g
  __nv_bfloat16* bias;          // [Z][4][L][L]
  float* lnst;                  // [T][2] (mean, rstd) or nullptr
  __nv_bfloat16* xh;            // [T][136] the normalised input bf16(xh) | 1 | 0 ... (training: the operand of the weight-gradient GEMM, whose ones column gives the column sums) or nullptr
  unsigned ntile;               // T / 32
  int L, transposed;            // T = Z L^2 tokens, L a multiple of 128
  float eps;
};

// Packed row s < 512 -> the output channel it holds.  In a block of 64 rows the 8 n tiles of the mma are (og, j) = (nt >> 2, nt & 3) and column n of
// a tile holds channel 32 og + 8 (n >> 1) + 2 j + (n & 1): the accumulator pair (2 q4, 2 q4 + 1) of the 4 tiles of a group j = 0..3 is channels
// 8 q4 .. 8 q4 + 7 of that group.
DEVI int f1_channel(int s) { return (s & ~63) + 32 * ((s >> 5) & 1) + 8 * ((s >> 1) & 3) + 2 * ((s >> 3) & 3) + (s & 1); }

// The weights of the front in the kernel's layout, one launch (one warp per packed row): W'[s] = bf16(W[c(s)] diag gamma) for the 516 rows of
// Wq | Wk | Wv | Wg | Wb (rows 516-519 zero), b[c] = W[c] . beta in fp32 (the butterfly order is fixed: replays are bit-identical).
struct F1PackParams {
  const __nv_bfloat16* w[5];    // Wq, Wk, Wv, Wg: [128][128]; Wb: [4][128]
  const float* gamma;           // [128]
  const float* beta;            // [128]
  __nv_bfloat16* wp;            // [520][128]
  float* bvec;                  // [520]
};

__global__ void __launch_bounds__(256) f1_pack_kernel(const F1PackParams p) {
  const int s = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (s >= F1_ROWS) return;
  const int c = s < 512 ? f1_channel(s) : (s < 516 ? s : -1);          // channel (512 + head for the bias rows), -1: padding
  uint2 packed = make_uint2(0u, 0u);
  float dotb = 0.f;
  if (c >= 0) {
    const __nv_bfloat16* src = c < 512 ? p.w[c >> 7] + (size_t)(c & 127) * 128 : p.w[4] + (size_t)(c - 512) * 128;
    const uint2 raw = *reinterpret_cast<const uint2*>(src + lane * 4);
    const float w0 = bf16lo(raw.x), w1 = bf16hi(raw.x), w2 = bf16lo(raw.y), w3 = bf16hi(raw.y);
    const float4 g = *reinterpret_cast<const float4*>(p.gamma + lane * 4), b = *reinterpret_cast<const float4*>(p.beta + lane * 4);
    packed = make_uint2(pack_bf16(w0 * g.x, w1 * g.y), pack_bf16(w2 * g.z, w3 * g.w));
    dotb = fmaf(w3, b.w, fmaf(w2, b.z, fmaf(w1, b.y, w0 * b.x)));
  }
  *reinterpret_cast<uint2*>(p.wp + (size_t)s * 128 + lane * 4) = packed;
  dotb = warp_sum(dotb);
  if (lane == 0) p.bvec[c >= 0 ? c : s] = dotb;
}

template <int NW_>
struct F1Cfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32;
  static constexpr int SMEM = F1_ROWS * 256 + F1_ROWS * 4;      // W' (rows of 256 B) + b
};

// byte offset of 16-byte chunk `chunk` of packed row `row`: the chunk index is XOR-ed with 4 on odd rows, so the 2 rows x 4 chunks a quarter warp
// reads per B-fragment load fall in 8 different bank groups
DEVI uint32_t f1_woff(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ ((row & 1u) << 2)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) f1_kernel(const F1Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* bv = reinterpret_cast<float*>(smem_raw + F1_ROWS * 256);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- W' and b into shared memory, once per CTA
  for (int i = tid; i < F1_ROWS * 16; i += G::NTHR) cp_async16(sb + f1_woff(i >> 4, i & 15), p.w + (size_t)(i >> 4) * 128 + (i & 15) * 8);
  cp_async_commit();
  for (int i = tid; i < F1_ROWS; i += G::NTHR) bv[i] = p.bvec[i];
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[4];                                                // this thread's B-fragment address of channel block kb in row g8 of block 0
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = sb + f1_woff(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    // 32 tokens of one pair row (L is a multiple of 128): problem z, row arow, columns b0 .. b0 + 31
    const unsigned t0 = tile * 32, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;

    // ---- tokens: 2 m tiles x rows (g8, g8 + 8) x 4 blocks of 8 channels, straight from global memory
    uint4 xr[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const unsigned r = mt * 16 + g8 + 8 * hh;
        const size_t src = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);   // the ending node reads element (b, a)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) xr[mt][hh][kb] = ldg128(p.x + src * 128 + 32 * kb + 8 * q4);
      }
    bool keep[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) keep[mt][hh] = p.mask == nullptr || p.mask[z * L + b0 + mt * 16 + g8 + 8 * hh] != 0;

    // ---- LayerNorm on the registers: a thread holds 32 of the 128 channels of its row, a quad the whole row
    uint32_t a[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float s = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[mt][hh][kb];
          s += (bf16lo(u.x) + bf16hi(u.x)) + (bf16lo(u.y) + bf16hi(u.y)) + (bf16lo(u.z) + bf16hi(u.z)) + (bf16lo(u.w) + bf16hi(u.w));
        }
        const float mean = quad_sum(s) * (1.f / 128.f);
        float v = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[mt][hh][kb];
          float d;
          d = bf16lo(u.x) - mean; v = fmaf(d, d, v);  d = bf16hi(u.x) - mean; v = fmaf(d, d, v);
          d = bf16lo(u.y) - mean; v = fmaf(d, d, v);  d = bf16hi(u.y) - mean; v = fmaf(d, d, v);
          d = bf16lo(u.z) - mean; v = fmaf(d, d, v);  d = bf16hi(u.z) - mean; v = fmaf(d, d, v);
          d = bf16lo(u.w) - mean; v = fmaf(d, d, v);  d = bf16hi(u.w) - mean; v = fmaf(d, d, v);
        }
        const float rstd = rsqrtf(quad_sum(v) * (1.f / 128.f) + p.eps);
        if (p.lnst != nullptr && q4 == 0) {
          const size_t tok = (size_t)t0 + mt * 16 + g8 + 8 * hh;
          *reinterpret_cast<float2*>(p.lnst + 2 * tok) = make_float2(mean, rstd);
        }
        if (p.xh != nullptr && q4 == 0)                          // the augmentation: a column of ones (bf16 1.0), then zeros, in columns 128 .. 135
          stg128(p.xh + ((size_t)t0 + mt * 16 + g8 + 8 * hh) * 136 + 128, make_uint4(0x3f80u, 0u, 0u, 0u));
        // normalised, bf16: the A fragments (a0 / a2 of step 2 kb: u.x / u.y of row g8, a1 / a3: of row g8 + 8; step 2 kb + 1: u.z / u.w)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[mt][hh][kb];
          const uint32_t n0 = pack_bf16((bf16lo(u.x) - mean) * rstd, (bf16hi(u.x) - mean) * rstd);
          const uint32_t n1 = pack_bf16((bf16lo(u.y) - mean) * rstd, (bf16hi(u.y) - mean) * rstd);
          const uint32_t n2 = pack_bf16((bf16lo(u.z) - mean) * rstd, (bf16hi(u.z) - mean) * rstd);
          const uint32_t n3 = pack_bf16((bf16lo(u.w) - mean) * rstd, (bf16hi(u.w) - mean) * rstd);
          a[mt][2 * kb][hh] = n0;      a[mt][2 * kb][2 + hh] = n1;
          a[mt][2 * kb + 1][hh] = n2;  a[mt][2 * kb + 1][2 + hh] = n3;
          if (p.xh != nullptr)                                  // training: the normalised row (bf16), 16 bytes per thread: exactly the A fragments' values
            stg128(p.xh + ((size_t)t0 + mt * 16 + g8 + 8 * hh) * 136 + 32 * kb + 8 * q4, make_uint4(n0, n1, n2, n3));
        }
      }

    // ---- 8 blocks of 64 output channels: q | k | v | g
#pragma unroll 1
    for (int blk = 0; blk < 8; ++blk) {
      float acc[2][8][4];
#pragma unroll
      for (int og = 0; og < 2; ++og) {                           // the accumulators start from the shift b of the 8 channels the thread owns
        const float4 b0 = *reinterpret_cast<const float4*>(bv + 64 * blk + 32 * og + 8 * q4);
        const float4 b1 = *reinterpret_cast<const float4*>(bv + 64 * blk + 32 * og + 8 * q4 + 4);
        const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            acc[mt][4 * og + j][0] = acc[mt][4 * og + j][2] = bb[2 * j];
            acc[mt][4 * og + j][1] = acc[mt][4 * og + j][3] = bb[2 * j + 1];
          }
      }
      const uint32_t wblk = 64u * blk * 256u;
      uint4 bn = lds128_ro(wb[0] + wblk);                        // the B fragments, one n tile ahead of the mma that uses them
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint4 bc = bn;
          constexpr int last = 31;
          if (kb * 8 + nt < last) bn = lds128_ro(wb[(kb * 8 + nt + 1) >> 3] + wblk + ((kb * 8 + nt + 1) & 7) * 8 * 256);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc[mt][nt], a[mt][2 * kb], bc.x, bc.y);
            mma16816(acc[mt][nt], a[mt][2 * kb + 1], bc.z, bc.w);
          }
        }
      // ---- epilogue: bf16, 16 bytes (8 channels) per thread and row of the q | k | v | g output
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
            stg128(p.out + tok * 512 + 64 * blk + 32 * og + 8 * q4, v);
          }
    }

    // ---- the 4 bias heads: rows 512 .. 519 of the packed weights are one n tile (columns 2 q4, 2 q4 + 1 = heads: only 4 of the 8 are real)
    {
      float acc[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        acc[mt][0] = acc[mt][2] = bv[512 + 2 * q4];
        acc[mt][1] = acc[mt][3] = bv[512 + 2 * q4 + 1];
      }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const uint4 b = lds128_ro(wb[kb] + 512u * 256u);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc[mt], a[mt][2 * kb], b.x, b.y);
          mma16816(acc[mt], a[mt][2 * kb + 1], b.z, b.w);
        }
      }
      if (q4 < 2) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const float v = acc[mt][2 * hh + e];
              p.bias[(size_t)(z * 4 + 2 * q4 + e) * LL + rem0 + mt * 16 + g8 + 8 * hh] =
                  keep[mt][hh] ? __float2bfloat16_rn(v) : __float2bfloat16_rn(-3.3895313892515355e38f);
            }
      }
    }
  }
}

}  // namespace a100
