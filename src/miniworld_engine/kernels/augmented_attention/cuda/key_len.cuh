// key_len.cuh -- the live key count of a padded key mask, read by the attention cores themselves.
//
// FoldForge buckets its token counts: a key mask is True for the first n tokens and False after. The keys past a row's last attended
// key are masked in every query row; once a row has seen a live key its running max is a live logit, so their exponentials are exact
// zeros and contribute nothing: a core that stops its key loop at the tile holding the last attended key writes the same bits.
//
// Each warp that needs the bound reads the mask row itself (n bytes, 16 per lane-step when aligned, a warp max): no host-side tensor, no
// extra launch, and the bound follows the mask as it is at replay time (a CUDA graph refreshed with a new fold's mask needs nothing else).
// A row with no attended key keeps every key (the references fill masked logits with a finite minimum: such a row is a uniform average).
#pragma once
#include <stdint.h>

// 1 + the last nonzero byte of m[0, n), or 0 when every byte is zero. Any n and alignment (16-byte loads when m is aligned). Called by all
// 32 lanes of a warp (converged); every lane gets the value.
__device__ __forceinline__ int warp_key_len(const uint8_t* __restrict__ m, int n) {
  const int lane = threadIdx.x & 31;
  int last = -1, tail = 0;
  if ((reinterpret_cast<uintptr_t>(m) & 15) == 0) {
    for (int c = lane; c < n / 16; c += 32) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(m) + c);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int e = 0; e < 4; ++e)
        if (w[e]) last = max(last, 16 * c + 4 * e + ((31 - __clz(w[e])) >> 3));
    }
    tail = n / 16 * 16;
  }
  for (int i = tail + lane; i < n; i += 32)
    if (__ldg(m + i)) last = max(last, i);
  return __reduce_max_sync(0xffffffffu, last) + 1;
}

// The BN-key tiles a core of L keys (L >= n: the pad keys past the mask are masked too) visits for a key mask row m[0, n): up to the tile
// holding the last attended key, every tile when none is attended. Warp-collective as warp_key_len.
__device__ __forceinline__ int warp_key_tiles(const uint8_t* __restrict__ m, int n, int L, int BN) {
  const int kl = warp_key_len(m, n);
  return kl == 0 ? L / BN : max(1, min(L / BN, (kl + BN - 1) / BN));
}

// warp_key_len for one thread (a producer that issues alone): read from the end, eight 16-byte loads in flight per step when m is 16-byte
// aligned and n a multiple of 16, so a padded tail costs one load latency per 128 masked keys.
__device__ __forceinline__ int thread_key_len(const uint8_t* __restrict__ m, int n) {
  if ((reinterpret_cast<uintptr_t>(m) & 15) == 0 && n % 16 == 0) {
    for (int c = n / 16; c > 0; c -= 8) {
      uint4 v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = c - 1 - i >= 0 ? __ldg(reinterpret_cast<const uint4*>(m) + (c - 1 - i)) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
        for (int e = 3; e >= 0; --e)
          if (w[e]) return 16 * (c - 1 - i) + 4 * e + ((31 - __clz(w[e])) >> 3) + 1;
      }
    }
    return 0;
  }
  for (int i = n - 1; i >= 0; --i)
    if (__ldg(m + i)) return i + 1;
  return 0;
}

// warp_key_tiles for one thread
__device__ __forceinline__ int thread_key_tiles(const uint8_t* __restrict__ m, int n, int L, int BN) {
  const int kl = thread_key_len(m, n);
  return kl == 0 ? L / BN : max(1, min(L / BN, (kl + BN - 1) / BN));
}
