// REJECTED (S90 trimul-train): gate_bwd2_kernel, gate_bwd with the next iteration's loads issued first (software pipeline), same arithmetic.
// Micro (L2-flushed graph replays, random data): D256 L512 170 -> 184 us, D512 L384 189 -> 204 us (slower: the kernel is not load-issue bound).
template <int D>
__device__ __forceinline__ void gb2_load(GbLd& r, size_t e, size_t n, const __nv_bfloat16* __restrict__ dy, const __nv_bfloat16* __restrict__ p,
                                         const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ ds,
                                         const float* __restrict__ mu_o, const float* __restrict__ rs_o, int M, int L) {
  constexpr int G8 = D / 8;
  if (e < n) {
    const int row = (int)(e / G8), c8 = (int)(e % G8) * 8;
    r.y = reinterpret_cast<const uint4*>(dy)[e];
    r.p = reinterpret_cast<const uint4*>(p)[e];
    r.g = reinterpret_cast<const uint4*>(g)[e];
    if (ds) {
      const int dr = M > L * L ? row % L + row / (L * L) * L : row % L;
      r.d = *reinterpret_cast<const uint4*>(ds + (size_t)dr * D + c8);
    }
    r.ro = rs_o[row]; r.mu = mu_o[row];
  }
}
template <int D>
__global__ void __launch_bounds__(256) gate_bwd2_kernel(const __nv_bfloat16* __restrict__ dy, const __nv_bfloat16* __restrict__ p,
                                                         const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ ds,
                                                         const float* __restrict__ vec, const float* __restrict__ mu_o,
                                                         const float* __restrict__ rs_o, __nv_bfloat16* __restrict__ dpr_out,
                                                         __nv_bfloat16* __restrict__ dg_out, int ldg, float* __restrict__ S1,
                                                         float* __restrict__ S2, float* __restrict__ part, int M, int L) {
  constexpr int G8 = D / 8;
  const size_t T = (size_t)gridDim.x * blockDim.x, n = (size_t)M * G8;
  const size_t e0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int q = (int)(e0 % G8), c8 = q * 8;
  float sp[8], ep[8], c0[8], c1[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) { sp[j] = vec[c8 + j]; ep[j] = vec[D + c8 + j]; c0[j] = 0.f; c1[j] = 0.f; }
  GbLd cur, nxt;
  gb2_load<D>(cur, e0, n, dy, p, g, ds, mu_o, rs_o, M, L);
  for (size_t base = 0; base < n; base += T) {
    const size_t e = base + e0;
    const bool ok = e < n;
    const int row = ok ? (int)(e / G8) : -1;
    if (GB2_PF) gb2_load<D>(nxt, e + T, n, dy, p, g, ds, mu_o, rs_o, M, L);
    else gb2_load<D>(cur, e, n, dy, p, g, ds, mu_o, rs_o, M, L);
    float s1 = 0.f, s2 = 0.f;
    if (ok) {
      float fy[8], fp[8], fg[8], fd[8];
      unpack8(cur.y, fy); unpack8(cur.p, fp); unpack8(cur.g, fg);
      if (ds) unpack8(cur.d, fd);
      else {
#pragma unroll
        for (int j = 0; j < 8; ++j) fd[j] = 1.f;
      }
      const float ro = cur.ro, rm = ro * cur.mu;
      float odpr[8], odg[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float sg = sigmoidf_fast(fg[j]);
        const float dpv = fy[j] * fd[j] * sg;
        odpr[j] = dpv * ro;
        odg[j] = dpv * fp[j] * (1.f - sg);
        s1 = fmaf(dpv, sp[j], s1);
        s2 = fmaf(dpv, fp[j] - ep[j], s2);
        c0[j] += dpv;
        c1[j] = fmaf(dpv, rm, c1[j]);
      }
      reinterpret_cast<uint4*>(dpr_out)[e] = pack8(odpr);
      *reinterpret_cast<uint4*>(dg_out + (size_t)row * ldg + c8) = pack8(odg);
    }
    seg_sum_atomic(s1, row, S1);
    seg_sum_atomic(s2, row, S2);
    if (GB2_PF) cur = nxt;
  }
  __shared__ float tp[256][17];
#pragma unroll
  for (int j = 0; j < 8; ++j) { tp[threadIdx.x][j] = c0[j]; tp[threadIdx.x][8 + j] = c1[j]; }
  __syncthreads();
  const int q0 = (int)(((size_t)blockIdx.x * blockDim.x) % G8);
  for (int c = threadIdx.x; c < 2 * D; c += blockDim.x) {
    const int col = c % D, slot = (c / D) * 8 + col % 8;
    int t = col / 8 - q0;
    if (t < 0) t += G8;
    float s = 0.f;
    for (; t < (int)blockDim.x; t += G8) s += tp[t][slot];
    part[(size_t)blockIdx.x * 2 * D + c] = s;
  }
}

