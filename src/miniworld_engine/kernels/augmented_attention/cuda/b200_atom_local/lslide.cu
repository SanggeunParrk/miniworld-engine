// lslide.cu — the last step of AlphaFold 3's slid key window (``LocalDiTBlock(key_window="slide")``) in the backward.
//
// The attention kernels run the windows the slide moves in edge CTAs of their own launches (slide.cuh): the forward writes their O / LSE,
// dQ their dQ rows, dK / dV their dbias windows -- and, since the clipped-window CTAs own the dK / dV rows of every key, the edge CTAs'
// dK / dV of the slid keys go to an fp32 partial spart [2 groups][A][4 heads][dK 32 | dV 32 channels][128 keys] (key-contiguous: the
// edge CTAs' lanes are keys). This kernel adds them: dk[key] = rn(dk[key] + (group 0's partial + group 1's partial)), in a fixed order
// (deterministic; group 0 = keys [0, 128), group 1 = [max(n_real - 128, 0), ..) when it has a window, the two overlapping when
// n_real < 256). A block takes (sample, 32 slots of a group): the partial in through shared memory (coalesced over the keys), then a warp
// per key row read-modify-writes its dK | dV columns (coalesced over the channels). n_real (the key mask's population count) is the one
// the dK / dV kernel's first edge CTA took, stored after the partial.
//   spart [2 A 4 64 128 + 4] fp32 (the last 16 B: n_real as an int); dkv [A N, *] rows ldd elements apart, dK at columns kcol .., dV at vcol .. (bf16, or fp32 on the fp32 path).
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>
#include "slide.cuh"

constexpr int KR = 128, NC = 4 * 64, NT = 256;                       // keys per group, partial channels per key (4 heads x dK | dV)

__device__ __forceinline__ void unpack(uint4 u, float (&x)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) { x[2 * j] = __uint_as_float(w[j] << 16); x[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u); }
}
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
  const __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&r);
}

template <typename T>
__device__ __forceinline__ void slide_add(const float* __restrict__ spart, T* __restrict__ dkv, int ldd,
                                          int kcol, int vcol, int N, int A) {
  __shared__ float tile[32][NC + 1];                                  // [key of the tile][channel]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nr = __ldg(reinterpret_cast<const int*>(spart) + 2 * A * NC * KR);   // n_real, as the dK / dV kernel's edge CTAs took it
  int wf1, kb1;
  const bool g1 = slide_group(1, nr, wf1, kb1);
  if (nr <= 0) return;
  for (int u = blockIdx.x; u < A * 8; u += gridDim.x) {               // (sample, group, 32-slot tile)
    const int a = u >> 3, g = (u >> 2) & 1, key0 = (g == 0 ? 0 : kb1) + 32 * (u & 3);
    if (g == 1 && !g1) continue;
    {                                                                 // the tile's partials (both groups' where they overlap), in order
      constexpr int PER = 32 * NC / NT;                               // 32 per thread, all loads issued before any store
      const int kk = tid & 31, key = key0 + kk;
      const bool ok = !(g == 1 && key < KR) && key < N;               // (a group-1 slot below 128: group 0's tile adds it)
      const bool in0 = ok && key < KR, in1 = ok && g1 && key >= kb1 && key < kb1 + KR;
      const float* p0 = spart + (size_t)(0 * A + a) * NC * KR + (in0 ? key : 0);
      const float* p1 = spart + (size_t)(1 * A + a) * NC * KR + (in1 ? key - kb1 : 0);
      float x[PER];
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int hc = (tid >> 5) + (NT / 32) * i;
        x[i] = in0 ? __ldg(p0 + hc * KR) : 0.f;
      }
      if (in1) {
#pragma unroll
        for (int i = 0; i < PER; ++i) x[i] += __ldg(p1 + ((tid >> 5) + (NT / 32) * i) * KR);
      }
#pragma unroll
      for (int i = 0; i < PER; ++i) tile[kk][(tid >> 5) + (NT / 32) * i] = x[i];
    }
    __syncthreads();
    constexpr int E = 16 / (int)sizeof(T), VPL = NC / E / 32;         // elements per 16-B vector, vectors per lane and key
    constexpr int KPW = 32 / (NT / 32);                               // keys per warp
    uint4 cur[KPW][VPL];
    T* dst[KPW][VPL];
#pragma unroll
    for (int j = 0; j < KPW; ++j) {                                   // every read of the warp's rows first
      const int key = key0 + warp + (NT / 32) * j;
      const bool ok = !((g == 1 && key < KR) || key >= N);
#pragma unroll
      for (int i = 0; i < VPL; ++i) {
        const int cc = E * (lane + 32 * i), h = cc >> 6, c = cc & 63;
        dst[j][i] = ok ? dkv + ((size_t)a * N + key) * ldd + (c < 32 ? kcol + 32 * h + c : vcol + 32 * h + c - 32) : nullptr;
        cur[j][i] = ok ? *reinterpret_cast<const uint4*>(dst[j][i]) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int j = 0; j < KPW; ++j) {
      const int kk = warp + (NT / 32) * j;
#pragma unroll
      for (int i = 0; i < VPL; ++i) {
        if (dst[j][i] == nullptr) continue;
        const int cc = E * (lane + 32 * i);
        const uint4 u4 = cur[j][i];
        float t[E];                                                   // tile[kk][cc ..]: element (r + lane / 4) % E first (no bank conflicts)
#pragma unroll
        for (int r = 0; r < E; ++r) {
          const int e = (r + (lane >> 2)) & (E - 1);
          const float y = tile[kk][cc + e];
#pragma unroll
          for (int f = 0; f < E; ++f)
            if (r == 0) t[f] = f == e ? y : 0.f; else t[f] = f == e ? y : t[f];
        }
        if constexpr (sizeof(T) == 2) {
          float x[8];
          unpack(u4, x);
          *reinterpret_cast<uint4*>(dst[j][i]) =
              make_uint4(bf2(x[0] + t[0], x[1] + t[1]), bf2(x[2] + t[2], x[3] + t[3]), bf2(x[4] + t[4], x[5] + t[5]), bf2(x[6] + t[6], x[7] + t[7]));
        } else {
          *reinterpret_cast<uint4*>(dst[j][i]) =
              make_uint4(__float_as_uint(__uint_as_float(u4.x) + t[0]), __float_as_uint(__uint_as_float(u4.y) + t[1]),
                         __float_as_uint(__uint_as_float(u4.z) + t[2]), __float_as_uint(__uint_as_float(u4.w) + t[3]));
        }
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ void __launch_bounds__(NT)
local_slide_add(const float* spart, __nv_bfloat16* dkv, int ldd, int kcol, int vcol, int N, int A) {
  slide_add(spart, dkv, ldd, kcol, vcol, N, A);
}
extern "C" __global__ void __launch_bounds__(NT)
local_slide_add_f32(const float* spart, float* dkv, int ldd, int kcol, int vcol, int N, int A) {
  slide_add(spart, dkv, ldd, kcol, vcol, N, A);
}
