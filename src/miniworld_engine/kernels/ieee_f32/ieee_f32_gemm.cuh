// ieee_f32_gemm.cuh -- the IEEE fp32 GEMM tile kernel of ieee_f32.cu (no torch / ATen: a plain CUDA header, so a standalone benchmark compiles the same code).
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <cstdlib>

#ifndef MWE_BK
#define MWE_BK 16
#endif
#ifndef MWE_MINB
#define MWE_MINB 2  // CTAs a SM for a B operand stored [N, K] (the weight of a forward); a [K, N] B (backward GEMMs) runs 1: it keeps its 151 registers (measured: 46 -> 61 TFLOPs with an MK A, 38 -> 50 with a KM A)
#endif
constexpr int BN = 128, BK = MWE_BK, NTHR = 256, PAD = 4;  // BK: the k tile (and the granularity of a split-K range)
enum { LA_MK = 0, LA_KM = 1 };
enum { LB_NK = 0, LB_KN = 1 };
enum { EPI_STORE = 0, EPI_RES = 1, EPI_SWIGLU = 2, EPI_DAB = 3 };

struct Args {
  const float* a; long lda;
  const float* b0; const float* b1; long ldb; int bsplit;
  float* c; long ldc; long slab;
  const float* r; long ldr;
  float* aux; long ldaux;
  int M, N, K, kper;  // N: columns of the product (EPI_SWIGLU: H)
  int H;              // EPI_SWIGLU / EPI_DAB: the hidden width
};

__device__ __forceinline__ float4 f4zero() { return make_float4(0.f, 0.f, 0.f, 0.f); }

template <int LA, int LB, int EPI, int BMT>
__global__ void __launch_bounds__(NTHR, (LB == LB_KN ? 1 : MWE_MINB)) gemm_k(const Args p) {
  constexpr int KB = BK;
  constexpr int KQ = KB / 4;                  // float4 of one k row of a tile
  constexpr int NLA = (BMT * KQ + NTHR - 1) / NTHR, NLB = (BN * KQ) / NTHR;  // float4 of the A / B tile a thread stages
  constexpr int TM = BMT / 16;            // rows a thread owns: halves of 4 at ty * 4 and 64 + ty * 4
  __shared__ __align__(16) float As[2][KB][BMT + PAD];
  __shared__ __align__(16) float Bs[2][KB][BN + PAD];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * BMT;
  const int n0 = (EPI == EPI_SWIGLU) ? blockIdx.x * 64 : blockIdx.x * BN;
  const int kbeg = blockIdx.z * p.kper, kend = min(p.K, kbeg + p.kper);
  float4 ra[NLA], rb[NLB];

  auto gload = [&](int k0) {
#pragma unroll
    for (int e = 0; e < NLA; ++e) {
      const int idx = tid + NTHR * e;
      if (BMT * KQ < NTHR * NLA && idx >= BMT * KQ) { ra[e] = f4zero(); continue; }
      if (LA == LA_MK) {
        const int m = idx / KQ, k = k0 + (idx % KQ) * 4;
        ra[e] = (m0 + m < p.M && k < kend) ? *reinterpret_cast<const float4*>(p.a + (long)(m0 + m) * p.lda + k) : f4zero();
      } else {
        const int kk = idx / (BMT / 4), m4 = (idx % (BMT / 4)) * 4, k = k0 + kk;
        ra[e] = (k < kend && m0 + m4 < p.M) ? *reinterpret_cast<const float4*>(p.a + (long)k * p.lda + m0 + m4) : f4zero();
      }
    }
#pragma unroll
    for (int e = 0; e < NLB; ++e) {
      const int idx = tid + NTHR * e;
      if (LB == LB_NK) {
        const int j = idx / KQ, k = k0 + (idx % KQ) * 4;
        const float* src = nullptr;
        if (EPI == EPI_SWIGLU) {
          const int h = n0 + (j & 63);
          if (h < p.N) src = (j < 64 ? p.b0 : p.b1) + (long)h * p.ldb;
        } else {
          if (n0 + j < p.N) src = p.b0 + (long)(n0 + j) * p.ldb;
        }
        rb[e] = (src != nullptr && k < kend) ? *reinterpret_cast<const float4*>(src + k) : f4zero();
      } else {
        const int kk = idx >> 5, n4 = (idx & 31) << 2, k = k0 + kk;
        const float* src = nullptr;
        if (k < kend && n0 + n4 < p.N) src = (k < p.bsplit ? p.b0 + (long)k * p.ldb : p.b1 + (long)(k - p.bsplit) * p.ldb) + n0 + n4;
        rb[e] = src != nullptr ? *reinterpret_cast<const float4*>(src) : f4zero();
      }
    }
  };
  auto sstore = [&](int buf) {
#pragma unroll
    for (int e = 0; e < NLA; ++e) {
      const int idx = tid + NTHR * e;
      if (BMT * KQ < NTHR * NLA && idx >= BMT * KQ) continue;
      if (LA == LA_MK) {
        const int m = idx / KQ, k = (idx % KQ) * 4;
        As[buf][k][m] = ra[e].x; As[buf][k + 1][m] = ra[e].y; As[buf][k + 2][m] = ra[e].z; As[buf][k + 3][m] = ra[e].w;
      } else {
        *reinterpret_cast<float4*>(&As[buf][idx / (BMT / 4)][(idx % (BMT / 4)) * 4]) = ra[e];
      }
    }
#pragma unroll
    for (int e = 0; e < NLB; ++e) {
      const int idx = tid + NTHR * e;
      if (LB == LB_NK) {
        const int j = idx / KQ, k = (idx % KQ) * 4;
        Bs[buf][k][j] = rb[e].x; Bs[buf][k + 1][j] = rb[e].y; Bs[buf][k + 2][j] = rb[e].z; Bs[buf][k + 3][j] = rb[e].w;
      } else {
        *reinterpret_cast<float4*>(&Bs[buf][idx >> 5][(idx & 31) << 2]) = rb[e];
      }
    }
  };

  float acc[TM][8];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

  const int nk = (kend - kbeg + KB - 1) / KB;
  if (nk > 0) {
    gload(kbeg);
    sstore(0);
    for (int it = 0; it < nk; ++it) {
      __syncthreads();
      const int cur = it & 1;
      if (it + 1 < nk) gload(kbeg + (it + 1) * KB);
#pragma unroll
      for (int kk = 0; kk < KB; ++kk) {
        float av[TM];
#pragma unroll
        for (int hh = 0; hh < BMT / 64; ++hh) {
          const float4 a4 = *reinterpret_cast<const float4*>(&As[cur][kk][hh * 64 + ty * 4]);
          av[hh * 4] = a4.x; av[hh * 4 + 1] = a4.y; av[hh * 4 + 2] = a4.z; av[hh * 4 + 3] = a4.w;
        }
        const float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][kk][tx * 4]);
        const float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][kk][64 + tx * 4]);
        const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
      }
      if (it + 1 < nk) sstore(cur ^ 1);
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + (i >> 2) * 64 + ty * 4 + (i & 3);
    if (row >= p.M) continue;
    if (EPI == EPI_SWIGLU) {
      const int h = n0 + tx * 4;
      if (h >= p.N) continue;
      float hv[4], av[4], bv[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        av[q] = acc[i][q];
        bv[q] = acc[i][4 + q];
        hv[q] = av[q] / (1.f + expf(-av[q])) * bv[q];
      }
      *reinterpret_cast<float4*>(p.c + (long)row * p.ldc + h) = make_float4(hv[0], hv[1], hv[2], hv[3]);
      if (p.aux != nullptr) {
        *reinterpret_cast<float4*>(p.aux + (long)row * p.ldaux + h) = make_float4(av[0], av[1], av[2], av[3]);
        *reinterpret_cast<float4*>(p.aux + (long)row * p.ldaux + p.H + h) = make_float4(bv[0], bv[1], bv[2], bv[3]);
      }
    } else {
#pragma unroll
      for (int jh = 0; jh < 2; ++jh) {
        const int col = n0 + jh * 64 + tx * 4;
        if (col >= p.N) continue;
        float4 v = make_float4(acc[i][jh * 4], acc[i][jh * 4 + 1], acc[i][jh * 4 + 2], acc[i][jh * 4 + 3]);
        if (EPI == EPI_STORE) {
          *reinterpret_cast<float4*>(p.c + (long)blockIdx.z * p.slab + (long)row * p.ldc + col) = v;
        } else if (EPI == EPI_RES) {
          const float4 rr = *reinterpret_cast<const float4*>(p.r + (long)row * p.ldr + col);
          v.x += rr.x; v.y += rr.y; v.z += rr.z; v.w += rr.w;
          *reinterpret_cast<float4*>(p.c + (long)row * p.ldc + col) = v;
        } else {  // EPI_DAB
          const float4 ar = *reinterpret_cast<const float4*>(p.r + (long)row * p.ldr + col);
          const float4 br = *reinterpret_cast<const float4*>(p.r + (long)row * p.ldr + p.H + col);
          const float dh[4] = {v.x, v.y, v.z, v.w}, a[4] = {ar.x, ar.y, ar.z, ar.w}, b[4] = {br.x, br.y, br.z, br.w};
          float da[4], db[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float s = 1.f / (1.f + expf(-a[q]));
            db[q] = dh[q] * (a[q] * s);
            da[q] = dh[q] * b[q] * (s * (1.f + a[q] * (1.f - s)));
          }
          *reinterpret_cast<float4*>(p.aux + (long)row * p.ldaux + col) = make_float4(da[0], da[1], da[2], da[3]);
          *reinterpret_cast<float4*>(p.aux + (long)row * p.ldaux + p.H + col) = make_float4(db[0], db[1], db[2], db[3]);
        }
      }
    }
  }
}

