// SPDX-License-Identifier: Apache-2.0
// MiniWorld single-direction wide inference K3: output projection with LN_out folded, gate from a
// precomputed bf16 tensor (one cuBLAS GEMM on the normalised input rows), residual epilogue:
//   y[t, o] = bf16( x[t, o] + bf16(P[t, o]) * sigmoid(g[t, o]) ),   g = bf16(xn . Wg^T)
//   P[t, o] = rs_t * (sum_c tri[c, t] Wp'[o, c] - mu_t u[o]) + v[o]
//   Wp' = bf16(gamma_out o Wp), u = rowsum(Wp'), v = Wp . beta_out   (host-prepared, per call)
// The channel-major triangle tile is the MN-major A operand exactly as TMA wrote it (no transpose, no
// normalised copy); producer warps 2/3 compute its per-token (mu, rs) while the consumers multiply.
// CTA tile = 128 tokens: warpgroup 0 = TMA producer + statistics, warpgroups 1/2 = consumers of 64 tokens
// each, sharing every streamed [128 n][64 k] weight slot. Gate and residual vectors of a column block are
// prefetched (16-byte loads) before its MMAs; the output is a 16-byte store after a quad shuffle.
#include "tmn_kernels.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf = __nv_bfloat16;
#ifndef PREGS
#define PREGS 40
#endif
#ifndef WAITN
#define WAITN 0
#endif
constexpr int D = WIDTH, H = HIDDEN, NB = 128, NSL = MW_NSLOT;
constexpr int CREGS_ = (168 * 384 - 128 * PREGS) / 256 / 8 * 8, CREGS = CREGS_ > 240 ? 240 : CREGS_;
constexpr int KP = H / 64, NCB = D / NB;
constexpr int THALF = H * 128;            // 64 tokens x H channels (bf16), channel-major chunks [64 ch][64 tok]
constexpr int SLOT = NB * 128;            // weight chunk [128 n][64 k]
constexpr int SMEM_T = 2 * THALF;
constexpr int SMEM_STATS = 128 * 8;       // [128 tokens] (mu, rs)
static_assert(D % NB == 0 && H % 64 == 0, "uni wide K3 v4 tiling");
static_assert(SMEM_T + NSL * SLOT + SMEM_STATS + (2 * NSL + 6) * 8 <= 232448, "shared memory");
struct Params { CUtensorMap tri, wp; const bf* x; const bf* g; bf* y; const float *u, *v; int tiles, pad; };

TMN_DEVI void mma_mk(float (&d)[64], uint64_t a, uint64_t b, int acc) {   // A MN-major (triangle), B K-major (weights)
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
               ", %64, %65, p, 1, 1, 1, 0; }"
               : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
               : "l"(a), "l"(b), "r"(acc));
}
TMN_DEVI uint32_t mw_lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
TMN_DEVI uint32_t pick4(const uint32_t (&a)[4], int i) { return i == 0 ? a[0] : i == 1 ? a[1] : i == 2 ? a[2] : a[3]; }

extern "C" __global__ __launch_bounds__(384, 1) void mw_uni_wide_output(__grid_constant__ const Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* sT = sm;
  uint8_t* ring = sm + SMEM_T;
  float* stats = reinterpret_cast<float*>(sm + SMEM_T + NSL * SLOT);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + SMEM_T + NSL * SLOT + SMEM_STATS);
  uint64_t *full = bars, *empty = bars + NSL;
  uint64_t *tfull = bars + 2 * NSL, *tempty = bars + 2 * NSL + 2, *sfull = bars + 2 * NSL + 4;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (tid == 0) {
    for (int s = 0; s < NSL; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 8); }
    for (int w = 0; w < 2; ++w) { mbar_init(tfull + w, 1); mbar_init(tempty + w, 2); mbar_init(sfull + w, 32); }
    fence_barrier_init();
  }
  __syncthreads();
  const int ntile = (p.tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;

  if (warp < 4) {
    setmaxnreg_dec<PREGS>();
    if (warp == 0) {
      if (lane == 0) {                        // weight ring, in consumption order
        int it = 0;
        for (int t = 0; t < ntile; ++t)
          for (int cb = 0; cb < NCB; ++cb)
            for (int k = 0; k < KP; ++k, ++it) {
              const int s = it % NSL, u = it / NSL;
              if (u > 0) mbar_wait(empty + s, (u - 1) & 1);
              mbar_arrive_expect_tx(full + s, SLOT);
              tma_load_2d(ring + s * SLOT, &p.wp, full + s, 64 * k, cb * NB);
            }
      }
    } else if (warp == 1) {
      if (lane == 0) {                        // triangle halves (released by consumer + statistics warp)
        for (int t = 0; t < ntile; ++t) {
          const int row0 = ((int)blockIdx.x + t * (int)gridDim.x) * 128;
          for (int w = 0; w < 2; ++w) {
            if (t > 0) mbar_wait(tempty + w, (t - 1) & 1);
            mbar_arrive_expect_tx(tfull + w, THALF);
            for (int kc = 0; kc < KP; ++kc) tma_load_2d(sT + w * THALF + kc * 8192, &p.tri, tfull + w, row0 + 64 * w, 64 * kc);
          }
        }
      }
    } else {                                  // warps 2/3: LN_out statistics of triangle half w
      const int w = warp - 2;
      const uint32_t base = smem_u32(sT + w * THALF);
      for (int t = 0; t < ntile; ++t) {
        mbar_wait(tfull + w, t & 1);
        // lane owns tokens 2 lane, 2 lane + 1 (one 32-bit word of every channel row); shifted sums
        const uint32_t k0 = mw_lds32(base + swz128(0, lane * 4));
        const float klo = bf16lo(k0), khi = bf16hi(k0);
        float s1l = 0.f, s2l = 0.f, s1h = 0.f, s2h = 0.f;
#pragma unroll 1
        for (int c0 = 0; c0 < H; c0 += 8) {
          uint32_t v[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) { const int c = c0 + j; v[j] = mw_lds32(base + (c / 64) * 8192 + swz128(c % 64, lane * 4)); }
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float a = bf16lo(v[j]) - klo, b = bf16hi(v[j]) - khi;
            s1l += a; s2l = fmaf(a, a, s2l); s1h += b; s2h = fmaf(b, b, s2h);
          }
        }
        __syncwarp();
        fence_proxy_async();
        if (lane == 0) mbar_arrive(tempty + w);
        const float ml = s1l / H, mh = s1h / H;
        float4 o;
        o.x = klo + ml; o.y = rsqrtf(fmaxf(s2l / H - ml * ml, 0.f) + 1e-5f);
        o.z = khi + mh; o.w = rsqrtf(fmaxf(s2h / H - mh * mh, 0.f) + 1e-5f);
        *reinterpret_cast<float4*>(stats + (64 * w + 2 * lane) * 2) = o;
        mbar_arrive(sfull + w);
      }
    }
    return;
  }

  setmaxnreg_inc<CREGS>();
  const int w = warp / 4 - 1, wi = warp % 4, qp = lane % 4;
  const uint32_t tbase = smem_u32(sT + w * THALF), rbase = smem_u32(ring);
  const int rl = 16 * wi + lane / 4;          // this thread's rows rl, rl + 8 (tile-half relative)
  int it = 0;
#pragma unroll 1
  for (int t = 0; t < ntile; ++t) {
    const int row0 = ((int)blockIdx.x + t * (int)gridDim.x) * 128 + 64 * w;
    float mu[2], rs[2];
#pragma unroll 1
    for (int cb = 0; cb < NCB; ++cb) {
      // gate and residual vectors (quad-transposed layout: row rl + 8h, cols cb NB + 8 (4m + qp) .. + 7)
      uint4 xv[2][4], gv[2][4];
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const size_t o = (size_t)(row0 + rl + 8 * h) * D + cb * NB + 8 * (4 * m + qp);
          gv[h][m] = ldg128(p.g + o);
          xv[h][m] = ldg128(p.x + o);
        }
      if (cb == 0) mbar_wait(tfull + w, t & 1);
      float pa[64];
      int prev = 0;
#pragma unroll 1
      for (int k = 0; k < KP; ++k) {
        const int s = it % NSL;
        mbar_wait(full + s, (it / NSL) & 1);
        const uint32_t a = tbase + k * 8192, b = rbase + s * SLOT;
        fence_regs(pa);
        wgmma_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) mma_mk(pa, smem_desc(a + 2048 * q, 16, 1024, 1), smem_desc(b + 32 * q, 16, 1024, 1), k > 0 || q > 0);
        wgmma_commit();
#if WAITN == 0
        wgmma_wait<0>();
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + s);
#else
        if (k > 0) { wgmma_wait<1>(); __syncwarp(); if (lane == 0) mbar_arrive(empty + prev); }
        prev = s;
#endif
        ++it;
      }
#if WAITN != 0
      wgmma_wait<0>();
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + prev);
#endif
      (void)prev;
      fence_regs(pa);
      if (cb == NCB - 1) {                    // this consumer is done with its triangle half
        named_bar_sync(1 + w, 128);
        if (wi == 0 && lane == 0) mbar_arrive(tempty + w);
      }
      if (cb == 0) {
        mbar_wait(sfull + w, t & 1);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 st = *reinterpret_cast<const float2*>(stats + (64 * w + rl + 8 * h) * 2);
          mu[h] = st.x; rs[h] = st.y;
        }
      }
      // LN_out fold, rounded once to bf16 (pairs): pr[2 i + h] = cols 8 i + 2 qp, +1 of row rl + 8 h
      uint32_t pr[32];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int c = cb * NB + 8 * i + 2 * qp;
        const float2 uu = __ldg(reinterpret_cast<const float2*>(p.u + c)), vv = __ldg(reinterpret_cast<const float2*>(p.v + c));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = 4 * i + 2 * h;
          pr[2 * i + h] = pack_bf16(fmaf(rs[h], pa[j] - mu[h] * uu.x, vv.x), fmaf(rs[h], pa[j + 1] - mu[h] * uu.y, vv.y));
        }
      }
#pragma unroll
      for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          // 4x4 quad transpose of words: lane qp receives i = 4m + qp from every quad lane src (cols 2 src, 2 src + 1)
          uint32_t in[4], o[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) in[jj] = pr[2 * (4 * m + jj) + h];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int send = (qp - k) & 3, src = (qp + k) & 3;
            uint32_t a = pick4(in, send);
            if (k) a = __shfl_sync(0xffffffffu, a, (lane & ~3) | src);
#pragma unroll
            for (int s = 0; s < 4; ++s) if (s == src) o[s] = a;
          }
          const uint4 x4 = xv[h][m], g4 = gv[h][m];
          uint4 y4;
          y4.x = pack_bf16(bf16lo(x4.x) + bf16lo(o[0]) * math::sigmoid(bf16lo(g4.x)), bf16hi(x4.x) + bf16hi(o[0]) * math::sigmoid(bf16hi(g4.x)));
          y4.y = pack_bf16(bf16lo(x4.y) + bf16lo(o[1]) * math::sigmoid(bf16lo(g4.y)), bf16hi(x4.y) + bf16hi(o[1]) * math::sigmoid(bf16hi(g4.y)));
          y4.z = pack_bf16(bf16lo(x4.z) + bf16lo(o[2]) * math::sigmoid(bf16lo(g4.z)), bf16hi(x4.z) + bf16hi(o[2]) * math::sigmoid(bf16hi(g4.z)));
          y4.w = pack_bf16(bf16lo(x4.w) + bf16lo(o[3]) * math::sigmoid(bf16lo(g4.w)), bf16hi(x4.w) + bf16hi(o[3]) * math::sigmoid(bf16hi(g4.w)));
          stg128(p.y + (size_t)(row0 + rl + 8 * h) * D + cb * NB + 8 * (4 * m + qp), y4);
        }
    }
  }
}
