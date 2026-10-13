#pragma once
#include "fused_sm80.cuh"
namespace triattn_affine_sm80 {
using namespace a100;
struct Params:a100::F1Params{const float* gamma;const float* beta;bool save_affine;};
template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) native_f1_kernel(const Params p) {
  constexpr int DIN = G::DIN, KBI = G::KBI, NOUT = G::NOUT, H = G::H, ROWB = DIN * 2, CHR = DIN / 8;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* bv = reinterpret_cast<float*>(smem_raw + G::ROWS * ROWB);
  float* sg=bv+G::ROWS;float* sbeta=sg+DIN;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < G::ROWS * CHR; i += G::NTHR) cp_async16(sb + fwoff<ROWB>(i / CHR, i % CHR), p.w + (size_t)(i / CHR) * DIN + (i % CHR) * 8);
  cp_async_commit();
  for (int i = tid; i < G::ROWS; i += G::NTHR) bv[i] = p.bvec[i];
  for(int i=tid;i<DIN;i+=G::NTHR){sg[i]=p.gamma[i];sbeta[i]=p.beta[i];}
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
          const float h0a=(bf16lo(u.x)-mean)*rstd,h0b=(bf16hi(u.x)-mean)*rstd;
          const uint32_t n0=pack_bf16(h0a,h0b);
          const int c0=32*kb+8*q4+0;
          const uint32_t af0=pack_bf16(fmaf(h0a,sg[c0],sbeta[c0]),fmaf(h0b,sg[c0+1],sbeta[c0+1]));
          const float h1a=(bf16lo(u.y)-mean)*rstd,h1b=(bf16hi(u.y)-mean)*rstd;
          const uint32_t n1=pack_bf16(h1a,h1b);
          const int c1=32*kb+8*q4+2;
          const uint32_t af1=pack_bf16(fmaf(h1a,sg[c1],sbeta[c1]),fmaf(h1b,sg[c1+1],sbeta[c1+1]));
          const float h2a=(bf16lo(u.z)-mean)*rstd,h2b=(bf16hi(u.z)-mean)*rstd;
          const uint32_t n2=pack_bf16(h2a,h2b);
          const int c2=32*kb+8*q4+4;
          const uint32_t af2=pack_bf16(fmaf(h2a,sg[c2],sbeta[c2]),fmaf(h2b,sg[c2+1],sbeta[c2+1]));
          const float h3a=(bf16lo(u.w)-mean)*rstd,h3b=(bf16hi(u.w)-mean)*rstd;
          const uint32_t n3=pack_bf16(h3a,h3b);
          const int c3=32*kb+8*q4+6;
          const uint32_t af3=pack_bf16(fmaf(h3a,sg[c3],sbeta[c3]),fmaf(h3b,sg[c3+1],sbeta[c3+1]));
          a[mt][2 * kb][hh] = af0;      a[mt][2 * kb][2 + hh] = af1;
          a[mt][2 * kb + 1][hh] = af2;  a[mt][2 * kb + 1][2 + hh] = af3;
          if (p.xh != nullptr) stg128(p.xh + ((size_t)t0 + mt * 16 + g8 + 8 * hh) * (DIN + 8) + 32 * kb + 8 * q4, (p.save_affine?make_uint4(af0,af1,af2,af3):make_uint4(n0,n1,n2,n3)));
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


}
