// slide.cuh — AlphaFold 3's slid key window (``LocalDiTBlock(key_window="slide")``) inside the clipped-window attention kernels
// (lattn_{fwd,dq,dkv}.cu and their _tf32 twins).
//
// A slid window starts at S_w = min(max(32 w - 48, 0), max(n_real - 128, 0)) instead of 32 w - 48 (n_real: the real atoms, which come
// first; the trunked pair is laid out from the window's first key either way, so the per-window bias tensor is the same). The slide moves
// windows 0 and 1 (keys [0, 128)) and the real windows that end past n_real, 32 w + 80 > n_real (at most the last three, keys
// [max(n_real - 128, 0), n_real)); every other window keeps its clipped keys. The moved windows of each of those two groups share their
// keys, so one "edge" CTA per (group, head, sample range) runs the kernel's own per-CTA program over them: 128 query rows from the group's
// first window and the group's 128 keys at the start of the CTA's key range, every window of the CTA at band offset 0 (its slot j = key
// column j). The edge CTAs come last in the grid (the last 2 * 4 * SPE): they fill the SMs the clipped-window CTAs leave free, or else run
// as a short tail (SPE splits their samples finer than the main split). The clipped-window CTAs skip the moved windows (forward / dQ:
// their rows are not stored; dK / dV: their cells are masked and their dbias windows not written), the edge CTAs write only theirs.
// n_real is the key mask's population count, taken on the device by every CTA (no host read: CUDA-graph capturable, a replay with another
// mask is right), or N without a mask.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// n_real: the population count of km [N] (N a multiple of 16, km 16-B aligned), or N when km is null. Every thread of the block calls it;
// red: >= 32 ints of shared memory; one __syncthreads. qb (IntelliFold2's ``key_window="slide_qblock"``): the same slide against n_real
// rounded up to a whole query window of 32 (the last window may then reach into padding keys, which the key mask switches off).
__device__ __forceinline__ int slide_nreal(const uint8_t* __restrict__ km, int N, int* red, int qb = 0) {
  if (km == nullptr) return N;
  const int tid = threadIdx.x, nw = ((int)blockDim.x + 31) >> 5;
  int c = 0;
  for (int i = 16 * tid; i < N; i += 16 * (int)blockDim.x) {
    const uint4 b = __ldg(reinterpret_cast<const uint4*>(km + i));
    const uint32_t w4[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {                                     // bit 8 k of t = OR of byte k's bits
      uint32_t t = w4[j] | (w4[j] >> 4);
      t |= t >> 2;
      t |= t >> 1;
      c += __popc(t & 0x01010101u);
    }
  }
  c = __reduce_add_sync(0xffffffffu, c);
  if ((tid & 31) == 0) red[tid >> 5] = c;
  __syncthreads();
  int nr = 0;
  for (int i = 0; i < nw; ++i) nr += red[i];
  return qb ? min((nr + 31) & ~31, N) : nr;
}

// window w is one the slide moves: windows 0 and 1, and every window from the first whose clipped keys end past n_real on (the last
// real windows AND every window of padding queries past them, 2026-11-17: they all read [max(n_real - 128, 0), n_real) like the reference)
__device__ __forceinline__ bool slide_moved(int w, int nr) {
  return nr > 0 && (w < 2 || 32 * w + 80 > nr);
}
// The edge groups (2026-11-23: the TAIL on the tensor cores). Group 0 holds windows 0 and 1 (keys [0, 128)); group 1 the moved windows
// wf1 .. wf1 + 3 (wf1 = max(2, last - 2), last = the last window holding a real atom: the moved real windows and the first padding ones);
// the TAIL (every window of padding queries past them) reads group 1's slid keys [kb, kb + 128) too, so it is more groups of the same
// shape: group g >= 1 holds the windows wf1 + 4 (g - 1) .. + 3, and every group runs the kernel's own tcgen05 edge program (before: the
// tail ran on CUDA cores in the one group-1 CTA per head, ~114 ms of dQ + dK / dV at N = 4096, 2,048 real atoms, 48 samples).
// The launch carries slide_ngroups(nwin) groups (the most any n_real needs: wf1 >= 2); a group without a window returns at once (n_real is
// a device value, so the grid cannot follow it: CUDA-graph capturable, a replay with another mask is right).
__host__ __device__ __forceinline__ constexpr int slide_ngroups(int nwin) { return 1 + (nwin + 1) / 4; }
// group g's first window wf and key start kb; false when the group has no window (group 1 always has one when n_real > 0)
__device__ __forceinline__ bool slide_group_x(int g, int nr, int nwin, int& wf, int& kb) {
  wf = 0;
  kb = 0;
  if (nr <= 0) return false;
  if (g == 0) return true;
  wf = max(2, ((nr - 1) >> 5) - 2) + 4 * (g - 1);
  kb = max(nr - 128, 0);
  return g == 1 || wf < nwin;
}
// window w belongs to group g's edge CTA (every moved window has exactly one group)
__device__ __forceinline__ bool slide_mine_x(int g, int w, int nr) {
  if (!slide_moved(w, nr)) return false;
  if (g == 0) return w < 2;
  const int wf = max(2, ((nr - 1) >> 5) - 2) + 4 * (g - 1);
  return w >= 2 && w >= wf && w < wf + 4;
}
