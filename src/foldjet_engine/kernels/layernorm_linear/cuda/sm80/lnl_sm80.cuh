// lnl_sm80.cuh -- fused LayerNorm + projection to a few outputs on sm_80 (A100): out[r, h] = sum_c ((x[r, c] - mean_r) rstd_r gamma_c) W[h, c], h < nh <= 16, over rows of D channels
// (D = 16 or a multiple of 64 up to 512), bf16 in / bf16 out, fp32 statistics and accumulation.  The LayerNorm has no beta and the Linear no bias (ops.layer_norm_linear).
//
//   forward   one warp per tile of 16 rows: the rows go from global memory straight into the A fragments of mma.sync (a lane of a quad owns CW consecutive channels of each
//             32-channel group, so every load is 16 bytes), the statistics are quad shuffles on the registers, n = bf16((x - mean) rstd gamma) is the mma operand, W^T is the B
//             operand from shared memory.  Training also writes (mean, rstd) and u = the fp32 result before rounding (the backward's row sums).
//   backward  one warp per (slice of 64 channels, tile of 16 rows), everything slice-local given the saved statistics:
//                 dxn = dout W         (mma: A = dout [16 rows x 16 heads], B = W [16 heads x channels], the B fragments stay in registers)
//                 dx  = rstd (dxn gamma - s1 / D - xhat s2 / D)    s1 = dout . (W gamma), s2 = dout . u  (the row sums over the channels are row sums over the heads)
//                 Gn += dout^T xhat    (mma over the rows: the xhat words are transposed with movmatrix)    dW = Gn gamma,  dgamma = sum_h W Gn
//             The Gn accumulators of a warp are written once as partials; ``lnl_finalize`` sums them in a fixed order.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace lnl {

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI uint4 ldg128(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
DEVI uint2 ldg64(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
DEVI void stg128(void* p, uint4 v) { asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
DEVI void stg64(void* p, uint2 v) { asm volatile("st.global.v2.u32 [%0], {%1,%2};\n" ::"l"(p), "r"(v.x), "r"(v.y) : "memory"); }
DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }
DEVI uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
DEVI uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
DEVI void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float bf16f(unsigned short v) { return __uint_as_float((uint32_t)v << 16); }

DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// the transpose of the 8x8 bf16 matrix held in the usual fragment layout (lane = 4 row + col pair): the lane (g, q) receives rows 2q, 2q + 1 of column g of the input
DEVI uint32_t movmatrix_t(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
DEVI long long mn64(long long a, long long b) { return a < b ? a : b; }
DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// byte offset of 16-byte chunk `chunk` of row `row` of a table with rows of `rowb` bytes (a multiple of 128): the chunk index is XOR-ed with 4 on odd rows, so the two rows x four
// chunks a quarter warp reads per fragment load fall in 8 different bank groups
template <int ROWB>
DEVI uint32_t woff(uint32_t row, uint32_t chunk) { return row * ROWB + ((chunk ^ ((row & 1u) << 2)) << 4); }

// how a row of D channels is spread over the 4 lanes of a quad: CW consecutive channels per lane and 32-channel (CW = 8) or 16-channel (CW = 4) group
template <int D>
struct Row {
  static_assert(D == 16 || (D % 64 == 0 && D <= 512), "D");
  static constexpr int CW = D == 16 ? 4 : 8;                  // channels per lane per group
  static constexpr int GW = 4 * CW;                           // channels per group (the quad's)
  static constexpr int KB = D / GW;                           // groups per row
  static constexpr int STEPS = CW / 4;                        // 16-wide k steps per group: the mma consumes 4 channels of a lane per step
  static constexpr int ROWB = D * 2;
};

// load the CW channels (8 or 16 bytes) at the lane's position of group kb; rows of D channels
template <int CW>
struct Chunk;
template <>
struct Chunk<8> {
  using T = uint4;
  static DEVI T ld(const __nv_bfloat16* p) { return ldg128(p); }
  static DEVI void st(__nv_bfloat16* p, const T& v) { stg128(p, v); }
  static DEVI T zero() { return make_uint4(0u, 0u, 0u, 0u); }
  static DEVI uint32_t word(const T& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
  static DEVI T make(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return make_uint4(a, b, c, d); }
};
template <>
struct Chunk<4> {
  using T = uint2;
  static DEVI T ld(const __nv_bfloat16* p) { return ldg64(p); }
  static DEVI void st(__nv_bfloat16* p, const T& v) { stg64(p, v); }
  static DEVI T zero() { return make_uint2(0u, 0u); }
  static DEVI uint32_t word(const T& v, int i) { return i == 0 ? v.x : v.y; }
  static DEVI T make(uint32_t a, uint32_t b, uint32_t, uint32_t) { return make_uint2(a, b); }
};

// ------------------------------------------------------------------------------------------------------------------------------------------------------------- forward
struct FwdParams {
  const __nv_bfloat16* x;       // [M][D]
  const __nv_bfloat16* w;       // [nh][D]
  const float* gamma;           // [D] fp32 (the LayerNorm scale)
  __nv_bfloat16* out;           // [M][nh]
  float* stats;                 // [M][2] (mean, rstd) or nullptr
  float* u;                     // [M][nh] fp32 or nullptr
  long long M;
  int nh;
  float eps;
};

template <int D, int NT, int NTHR_>
struct FwdCfg {
  using R = Row<D>;
  static constexpr int NTHR = NTHR_, NW = NTHR_ / 32;
  static constexpr int SMEM = NT * 8 * R::ROWB + D * 4;      // W rows (zero rows past nh) + gamma
};

template <class G, int D, int NT>
__global__ void __launch_bounds__(G::NTHR, 1) lnl_fwd_kernel(const FwdParams p) {
  using R = Row<D>;
  constexpr int CW = R::CW, KB = R::KB, STEPS = R::STEPS, ROWB = R::ROWB;
  using C = Chunk<CW>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* gam = reinterpret_cast<float*>(smem_raw + NT * 8 * ROWB);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // W (rows >= nh zero) and gamma into shared memory, once per CTA; rows of 16-byte chunks, the chunk index swizzled for D >= 64
  for (int i = tid; i < NT * 8 * (D / 8); i += G::NTHR) {
    const int row = i / (D / 8), ch = i % (D / 8);
    const uint4 v = row < p.nh ? ldg128(p.w + (size_t)row * D + ch * 8) : make_uint4(0u, 0u, 0u, 0u);
    uint32_t dst = D == 16 ? (uint32_t)(row * ROWB + ch * 16) : woff<ROWB>(row, ch);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(sb + dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  }
  for (int i = tid; i < D; i += G::NTHR) gam[i] = p.gamma[i];
  __syncthreads();

  // this lane's B-fragment address of group kb in row g8 of n tile 0 (n tile nt: + 8 nt rows)
  uint32_t wb[KB];
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) {
    if (D == 16) wb[kb] = sb + (uint32_t)(g8 * ROWB + q4 * CW * 2);
    else wb[kb] = sb + woff<ROWB>(g8, 4 * kb + q4);
  }

  const long long ntile = (p.M + 15) / 16;
  for (long long tile = (long long)blockIdx.x * G::NW + warp; tile < ntile; tile += (long long)gridDim.x * G::NW) {
    const long long r0 = tile * 16;
    typename C::T xr[2][KB];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long long row = r0 + g8 + 8 * hh;
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) xr[hh][kb] = row < p.M ? C::ld(p.x + (size_t)row * D + kb * R::GW + q4 * CW) : C::zero();
    }
    uint32_t a[KB * STEPS][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s = 0.f;
#pragma unroll
      for (int kb = 0; kb < KB; ++kb)
#pragma unroll
        for (int i = 0; i < CW / 2; ++i) { const uint32_t w = C::word(xr[hh][kb], i); s += bf16lo(w) + bf16hi(w); }
      const float mean = quad_sum(s) * (1.f / D);
      float v = 0.f;
#pragma unroll
      for (int kb = 0; kb < KB; ++kb)
#pragma unroll
        for (int i = 0; i < CW / 2; ++i) {
          const uint32_t w = C::word(xr[hh][kb], i);
          float d = bf16lo(w) - mean; v = fmaf(d, d, v);
          d = bf16hi(w) - mean; v = fmaf(d, d, v);
        }
      const float rstd = rsqrtf(quad_sum(v) * (1.f / D) + p.eps);
      const long long row = r0 + g8 + 8 * hh;
      if (p.stats != nullptr && q4 == 0 && row < p.M) *reinterpret_cast<float2*>(p.stats + 2 * row) = make_float2(mean, rstd);
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) {
        float gm[CW];
#pragma unroll
        for (int i = 0; i < CW; ++i) gm[i] = gam[kb * R::GW + q4 * CW + i];
        uint32_t nw[CW / 2];
#pragma unroll
        for (int i = 0; i < CW / 2; ++i) {
          const uint32_t w = C::word(xr[hh][kb], i);
          nw[i] = pack_bf16((bf16lo(w) - mean) * rstd * gm[2 * i], (bf16hi(w) - mean) * rstd * gm[2 * i + 1]);
        }
#pragma unroll
        for (int s2 = 0; s2 < STEPS; ++s2) { a[kb * STEPS + s2][hh] = nw[2 * s2]; a[kb * STEPS + s2][2 + hh] = nw[2 * s2 + 1]; }
      }
    }

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
    for (int kb = 0; kb < KB; ++kb)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint32_t b[4];
        if (CW == 8) { const uint4 bv = lds128(wb[kb] + nt * 8 * ROWB); b[0] = bv.x; b[1] = bv.y; b[2] = bv.z; b[3] = bv.w; }
        else { const uint2 bv = lds64(wb[kb] + nt * 8 * ROWB); b[0] = bv.x; b[1] = bv.y; b[2] = b[3] = 0u; }
#pragma unroll
        for (int s2 = 0; s2 < STEPS; ++s2) mma16816(acc[nt], a[kb * STEPS + s2], b[2 * s2], b[2 * s2 + 1]);
      }

    // out[row][8 nt + 2 q4 + e] (rows g8: acc 0 1, g8 + 8: acc 2 3)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long long row = r0 + g8 + 8 * hh;
      if (row >= p.M) continue;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int col = 8 * nt + 2 * q4;
        if (col + 1 < p.nh && (p.nh & 1) == 0) {
          stg32(p.out + (size_t)row * p.nh + col, pack_bf16(acc[nt][2 * hh], acc[nt][2 * hh + 1]));
        } else {
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (col + e < p.nh) p.out[(size_t)row * p.nh + col + e] = __float2bfloat16_rn(acc[nt][2 * hh + e]);
        }
        if (p.u != nullptr) {
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (col + e < p.nh) p.u[(size_t)row * p.nh + col + e] = acc[nt][2 * hh + e];
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------ backward
struct BwdParams {
  const __nv_bfloat16* x;       // [M][D]
  const __nv_bfloat16* w;       // [nh][D]
  const float* gamma;           // [D] fp32
  const __nv_bfloat16* dout;    // [M][nh]
  const float* stats;           // [M][2]
  const float* u;               // [M][nh] fp32 (the forward's result before rounding)
  __nv_bfloat16* dx;            // [M][D]
  float* part;                  // [P][nh][D] fp32 partial sums of Gn = dout^T xhat (P = gridDim.x * RP)
  long long M;
  int nh;
};

// slices of GR groups (64 channels for D >= 64, 16 for D = 16); a CTA is SL slices x RP row partitions of warps
template <int D>
struct BwdShape {
  using R = Row<D>;
  static constexpr int GR = D == 16 ? 1 : 2;
  static constexpr int SWC = GR * R::GW;                       // channels per slice
  static constexpr int SL = D / SWC;                           // slices per row
  static constexpr int RP = SL >= 8 ? 1 : (SL >= 5 ? 2 : 8 / SL);      // row partitions: 6 .. 14 warps per CTA
  static constexpr int NW = SL * RP;
  static constexpr int NTHR = NW * 32;
  static constexpr int NJ = R::CW / 2;                         // n tiles per group
  // per-warp staging (two buffers): dout tile [16][16] bf16 (zero-padded rows past nh), u tile [16][16] fp32, stats [16][2]
  static constexpr int DOUT_B = 16 * 16 * 2, U_B = 16 * 16 * 4, ST_B = 16 * 8, BUF = DOUT_B + U_B + ST_B;
  static constexpr int WG_B = 16 * 4;                          // W gamma row sums (fp32, 16 entries)
  static constexpr int STAGE_B = NW * 2 * BUF + WG_B;
  static constexpr int RED_B = RP * 16 * D * 4;                // the CTA's partial sums [RP][nh][D] (nh <= 16), in the staging memory after the loop
  static constexpr int SMEM = STAGE_B > RED_B ? STAGE_B : RED_B;
};

template <int D>
__global__ void __launch_bounds__(BwdShape<D>::NTHR, 1) lnl_bwd_kernel(const BwdParams p) {
  using S = BwdShape<D>;
  using R = Row<D>;
  constexpr int CW = R::CW, GR = S::GR, NJ = S::NJ, SL = S::SL, RP = S::RP;
  using C = Chunk<CW>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* wgs = reinterpret_cast<float*>(smem_raw + S::NW * 2 * S::BUF);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int slice = warp % SL, rp = warp / SL;
  const int nh = p.nh;

  // W gamma row sums: wgs[h] = sum_c W[h][c] gamma[c] (a fixed order: replays are bit-identical)
  for (int h = warp; h < 16; h += S::NW) {
    float s = 0.f;
    if (h < nh) for (int c = lane; c < D; c += 32) s = fmaf(__bfloat162float(p.w[(size_t)h * D + c]), p.gamma[c], s);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) wgs[h] = s;
  }
  __syncthreads();

  // the lane's channels and the constants that stay in registers: gamma, the B fragments of W (n = this lane's channel slot)
  float gm[GR][CW];
  uint32_t bw[GR][NJ][2];
#pragma unroll
  for (int gr = 0; gr < GR; ++gr) {
    const int base = slice * S::SWC + gr * R::GW;
#pragma unroll
    for (int i = 0; i < CW; ++i) gm[gr][i] = p.gamma[base + q4 * CW + i];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = base + CW * (g8 >> 1) + 2 * j + (g8 & 1);       // n = g8 of n tile j
      const int h0 = 2 * q4, h1 = 2 * q4 + 1, h2 = 2 * q4 + 8, h3 = 2 * q4 + 9;
      const unsigned short w0 = h0 < nh ? __bfloat16_as_ushort(p.w[(size_t)h0 * D + c]) : (unsigned short)0, w1 = h1 < nh ? __bfloat16_as_ushort(p.w[(size_t)h1 * D + c]) : (unsigned short)0;
      const unsigned short w2 = h2 < nh ? __bfloat16_as_ushort(p.w[(size_t)h2 * D + c]) : (unsigned short)0, w3 = h3 < nh ? __bfloat16_as_ushort(p.w[(size_t)h3 * D + c]) : (unsigned short)0;
      bw[gr][j][0] = (uint32_t)w0 | ((uint32_t)w1 << 16);
      bw[gr][j][1] = (uint32_t)w2 | ((uint32_t)w3 << 16);
    }
  }

  float accw[GR][NJ][4];
#pragma unroll
  for (int gr = 0; gr < GR; ++gr)
#pragma unroll
    for (int j = 0; j < NJ; ++j) accw[gr][j][0] = accw[gr][j][1] = accw[gr][j][2] = accw[gr][j][3] = 0.f;

  const long long ntile = (p.M + 15) / 16;
  const long long first = (long long)blockIdx.x * RP + rp, stride = (long long)gridDim.x * RP;
  const uint32_t mybuf = sb + warp * 2 * S::BUF;

  // cp.async staging of a tile's dout / u / statistics (contiguous in memory: 16 rows) into buffer `which` of this warp
  auto stage = [&](long long tile, int which) {
    const uint32_t buf = mybuf + which * S::BUF;
    const long long r0 = tile * 16;
    const bool live = tile < ntile;
    // dout: 16 rows x nh bf16 = 2 nh chunks of 16 bytes, rows r0 ..; the zero-padded tile is [16][16] (row stride 32 bytes): copy row-wise 16-byte pieces is not aligned for odd nh, so
    // the tile is gathered by the compute loop from the contiguous copy at offset 0 (row stride nh * 2 bytes)
    for (int c = lane; c < 2 * nh; c += 32) {
      const long long e0 = r0 * nh + (long long)c * 8;                 // first element of the chunk
      const uint32_t bytes = (live && e0 < p.M * nh) ? (uint32_t)mn64(16, (p.M * nh - e0) * 2) : 0u;
      cp_async16(buf + c * 16, p.dout + (live ? e0 : 0), bytes);
    }
    for (int c = lane; c < 4 * nh; c += 32) {
      const long long e0 = r0 * nh + (long long)c * 4;
      const uint32_t bytes = (live && e0 < p.M * nh) ? (uint32_t)mn64(16, (p.M * nh - e0) * 4) : 0u;
      cp_async16(buf + S::DOUT_B + c * 16, p.u + (live ? e0 : 0), bytes);
    }
    if (lane < 8) {
      const long long e0 = r0 * 2 + (long long)lane * 4;
      const uint32_t bytes = (live && e0 < p.M * 2) ? (uint32_t)mn64(16, (p.M * 2 - e0) * 4) : 0u;
      cp_async16(buf + S::DOUT_B + S::U_B + lane * 16, p.stats + (live ? e0 : 0), bytes);
    }
    cp_async_commit();
  };

  stage(first, 0);
  int cur = 0;
  typename C::T xr[GR][2];
  auto load_x = [&](long long tile) {
#pragma unroll
    for (int gr = 0; gr < GR; ++gr)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const long long row = tile * 16 + g8 + 8 * hh;
        xr[gr][hh] = (tile < ntile && row < p.M) ? C::ld(p.x + (size_t)row * D + slice * S::SWC + gr * R::GW + q4 * CW) : C::zero();
      }
  };
  load_x(first);

  for (long long tile = first; tile < ntile; tile += stride) {
    stage(tile + stride, cur ^ 1);
    cp_async_wait<1>();
    __syncwarp();
    const uint32_t buf = mybuf + cur * S::BUF;
    const unsigned char* bptr = smem_raw + (buf - sb);
    const __nv_bfloat16* dsm = reinterpret_cast<const __nv_bfloat16*>(bptr);                       // [16][nh] contiguous
    const float* usm = reinterpret_cast<const float*>(bptr + S::DOUT_B);                           // [16][nh]
    const float2* stsm = reinterpret_cast<const float2*>(bptr + S::DOUT_B + S::U_B);               // [16] (mean, rstd)
    const long long r0 = tile * 16;

    // the dout fragments: A[m = row][k = head] for dxn (rows g8 / g8 + 8, heads 2 q4, 2 q4 + 1 | + 8) and A'[m = head][k = row] for dW (heads g8 / g8 + 8, rows 2 q4, 2 q4 + 1 | + 8)
    auto dv = [&](int row, int h) -> unsigned short { return (row < 16 && h < nh) ? __bfloat16_as_ushort(dsm[row * nh + h]) : (unsigned short)0; };
    uint32_t ad[4], at[4];
    ad[0] = (uint32_t)dv(g8, 2 * q4) | ((uint32_t)dv(g8, 2 * q4 + 1) << 16);
    ad[1] = (uint32_t)dv(g8 + 8, 2 * q4) | ((uint32_t)dv(g8 + 8, 2 * q4 + 1) << 16);
    ad[2] = (uint32_t)dv(g8, 2 * q4 + 8) | ((uint32_t)dv(g8, 2 * q4 + 9) << 16);
    ad[3] = (uint32_t)dv(g8 + 8, 2 * q4 + 8) | ((uint32_t)dv(g8 + 8, 2 * q4 + 9) << 16);
    at[0] = (uint32_t)dv(2 * q4, g8) | ((uint32_t)dv(2 * q4 + 1, g8) << 16);
    at[1] = (uint32_t)dv(2 * q4, g8 + 8) | ((uint32_t)dv(2 * q4 + 1, g8 + 8) << 16);
    at[2] = (uint32_t)dv(2 * q4 + 8, g8) | ((uint32_t)dv(2 * q4 + 9, g8) << 16);
    at[3] = (uint32_t)dv(2 * q4 + 8, g8 + 8) | ((uint32_t)dv(2 * q4 + 9, g8 + 8) << 16);

    // s1 / s2 of rows g8 and g8 + 8: sums over the heads, the 4 lanes of a quad taking h = q4, q4 + 4, ..
    float s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f};
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
      for (int h = q4; h < nh; h += 4) {
        const float d = __uint_as_float((uint32_t)__bfloat16_as_ushort(dsm[(g8 + 8 * hh) * nh + h]) << 16);
        s1[hh] = fmaf(d, wgs[h], s1[hh]);
        s2[hh] = fmaf(d, usm[(g8 + 8 * hh) * nh + h], s2[hh]);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { s1[hh] = quad_sum(s1[hh]) * (1.f / D); s2[hh] = quad_sum(s2[hh]) * (1.f / D); }

    // the next tile's x while this one is computed
    typename C::T xc[GR][2];
#pragma unroll
    for (int gr = 0; gr < GR; ++gr)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) xc[gr][hh] = xr[gr][hh];
    load_x(tile + stride);

#pragma unroll
    for (int gr = 0; gr < GR; ++gr) {
      float accx[NJ][4];
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        accx[j][0] = accx[j][1] = accx[j][2] = accx[j][3] = 0.f;
        mma16816(accx[j], ad, bw[gr][j][0], bw[gr][j][1]);
      }
      uint32_t xw[2][CW / 2];                                                  // the lane's x as bf16 pairs, then xhat (bf16) for the dW mma
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int i = 0; i < CW / 2; ++i) xw[hh][i] = C::word(xc[gr][hh], i);
      uint32_t out[2][CW / 2], xh[2][CW / 2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float2 st = stsm[g8 + 8 * hh];
        const float mean = st.x, rstd = st.y;
#pragma unroll
        for (int i = 0; i < CW / 2; ++i) {                                     // word i = channels 2 i, 2 i + 1 of the lane's CW: n tile j = i, columns e = 0, 1
          const float x0 = bf16lo(xw[hh][i]), x1 = bf16hi(xw[hh][i]);
          const float h0 = (x0 - mean) * rstd, h1 = (x1 - mean) * rstd;
          xh[hh][i] = pack_bf16(h0, h1);
          const float d0 = accx[i][2 * hh] * gm[gr][2 * i], d1 = accx[i][2 * hh + 1] * gm[gr][2 * i + 1];
          out[hh][i] = pack_bf16(rstd * (d0 - s1[hh] - h0 * s2[hh]), rstd * (d1 - s1[hh] - h1 * s2[hh]));
        }
        const long long row = r0 + g8 + 8 * hh;
        if (row < p.M) C::st(p.dx + (size_t)row * D + slice * S::SWC + gr * R::GW + q4 * CW, C::make(out[hh][0], out[hh][1], CW == 8 ? out[hh][CW / 2 - 2] : 0u, CW == 8 ? out[hh][CW / 2 - 1] : 0u));
      }
      // Gn[h][c] += sum_rows dout[row][h] xhat[row][c]: B'[k = rows][n = channel slot] = movmatrix of the lane's word i (rows g8 | g8 + 8 -> k 0..7 | 8..15)
#pragma unroll
      for (int j = 0; j < NJ; ++j) mma16816(accw[gr][j], at, movmatrix_t(xh[0][j]), movmatrix_t(xh[1][j]));
    }
    __syncwarp();
    cur ^= 1;
  }
  cp_async_wait<0>();
  __syncthreads();                                                 // every warp is done with its staging buffers: the shared memory now holds the CTA's partial sums

  // the partial Gn of every warp, [rp][h][channel] (lane: h = g8, g8 + 8; channels base + CW q4 + 2 j + e), then one partial row per CTA: the sum over the row partitions in order
  float* red = reinterpret_cast<float*>(smem_raw);
#pragma unroll
  for (int gr = 0; gr < GR; ++gr)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = g8 + 8 * (e >> 1), col = slice * S::SWC + gr * R::GW + CW * q4 + 2 * j + (e & 1);
        if (h < nh) red[((size_t)rp * nh + h) * D + col] = accw[gr][j][e];
      }
  __syncthreads();
  float* part = p.part + (size_t)blockIdx.x * nh * D;
  for (int i = tid; i < nh * D; i += S::NTHR) {
    float s = red[i];
    for (int r2 = 1; r2 < RP; ++r2) s += red[(size_t)r2 * nh * D + i];
    part[i] = s;
  }
}

// Gn[h][c] = the sum of the P partials in order, dW[h][c] = gamma_c Gn[h][c] (bf16); then dgamma_c = sum_h W[h][c] Gn[h][c] (the LayerNorm scale's dtype).  Two small launches:
// one thread per (h, c) for the first (the partials are read coalesced along c), one thread per channel for the second.
struct FinParams {
  const float* part;            // [P][nh][D]
  float* gn;                    // [nh][D]
  const __nv_bfloat16* w;       // [nh][D]
  const float* gamma;           // [D]
  __nv_bfloat16* dw;            // [nh][D]
  void* dgamma;                 // [D] fp32 or bf16
  int P, nh, D, ln_bf16;
};

__global__ void __launch_bounds__(256) lnl_reduce_kernel(const FinParams p) {
  const int i = blockIdx.x * 256 + threadIdx.x;                    // h D + c
  if (i >= p.nh * p.D) return;
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  const size_t stride = (size_t)p.nh * p.D;
  int k = 0;
  for (; k + 4 <= p.P; k += 4) {                                    // four independent chains, combined in a fixed order below
    s0 += p.part[(size_t)k * stride + i];
    s1 += p.part[(size_t)(k + 1) * stride + i];
    s2 += p.part[(size_t)(k + 2) * stride + i];
    s3 += p.part[(size_t)(k + 3) * stride + i];
  }
  for (; k < p.P; ++k) s0 += p.part[(size_t)k * stride + i];
  const float s = (s0 + s1) + (s2 + s3);
  p.gn[i] = s;
  p.dw[i] = __float2bfloat16_rn(s * p.gamma[i % p.D]);
}

__global__ void __launch_bounds__(128) lnl_dgamma_kernel(const FinParams p) {
  const int c = blockIdx.x * 128 + threadIdx.x;
  if (c >= p.D) return;
  float dg = 0.f;
  for (int h = 0; h < p.nh; ++h) dg = fmaf(__bfloat162float(p.w[(size_t)h * p.D + c]), p.gn[(size_t)h * p.D + c], dg);
  if (p.ln_bf16) reinterpret_cast<__nv_bfloat16*>(p.dgamma)[c] = __float2bfloat16_rn(dg); else reinterpret_cast<float*>(p.dgamma)[c] = dg;
}

}  // namespace lnl
