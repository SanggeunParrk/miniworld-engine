// conf_mask.cu — Chai-1's conformer mask on the windowed pair bias, in place, one launch (B200 variant path, 2026-11-17; sm_100a).
//
//   bias [4][nwin][32][128] fp32 (natural units, the tcgen05 attention's operand): slot (w, i, j) -> -inf unless the query 32 w + i and the
//   key in that slot share a conformer id, or the query is a padding atom (kmask[q] == 0: it keeps every valid key).
//   The key of slot j: 32 w - 48 + j (the clipped window), or, ``slide``, the slid window's first key in the windows the tcgen05 attention
//   moves (w <= (R - 1) / 32: start = min(max(32 w - 48, 0), max(R - 128, 0)), R = the real atom count = popcount(kmask) or n), clamped
//   into [0, n) (the slots outside the atoms are -inf already or masked by kmask in the kernel).
// One CTA per query window (256 threads); every CTA counts R itself (n bytes from L2) so there is no second launch and no host read.
// uid int64 [n] (padded with -1), kmask uint8 [n] or null.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

extern "C" __global__ void __launch_bounds__(256) atom_conf_mask(float* __restrict__ bias, const long long* __restrict__ uid,
                                                                  const unsigned char* __restrict__ kmask, int n, int slide) {
  const int nwin = n >> 5, w = blockIdx.x, tid = threadIdx.x;
  __shared__ int s_r[8];
  int start = 32 * w - 48;
  if (slide) {
    int c = 0;
    if (kmask) {
      for (int i = tid * 16; i < n; i += 256 * 16) {
        const uint4 v = *reinterpret_cast<const uint4*>(kmask + i);       // n is a multiple of 128
        c += __popc(v.x & 0x01010101u) + __popc(v.y & 0x01010101u) + __popc(v.z & 0x01010101u) + __popc(v.w & 0x01010101u);
      }
    } else if (tid == 0) {
      c = n;
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((tid & 31) == 0) s_r[tid >> 5] = c;
    __syncthreads();
    int r = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) r += s_r[i];
    if (w <= (r - 1) / 32) start = min(max(start, 0), max(r - 128, 0));
  }
  for (int e = tid; e < 32 * 128; e += 256) {
    const int qi = e >> 7, j = e & 127, q = 32 * w + qi;
    const int key = min(max(start + j, 0), n - 1);
    const bool keep = uid[q] == uid[key] || (kmask != nullptr && kmask[q] == 0);
    if (!keep) {
#pragma unroll
      for (int h = 0; h < 4; ++h) bias[((size_t)h * nwin + w) * 4096 + e] = -__int_as_float(0x7f800000);
    }
  }
}
