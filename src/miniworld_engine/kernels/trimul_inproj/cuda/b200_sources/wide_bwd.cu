// wide_bwd.cu -- memory-bound pieces of the B200 wide TriMul backward (D = 256 / 384 / 512), between the cuBLAS GEMMs, the
// contraction backward and k1wb.
//   gate_bwd  from dy, the saved p / g and ds:  dp = dy ds sigmoid(g),  dg = dp p (1 - sigmoid(g))   ->  dpr = dp rs_o, dg
//             per token S1 = dp . sp, S2 = dp . (p - ep)   (the LN_out backward's two row sums, from the fold identities
//             sum_c g_o[c] do[m, c] = dp . sp  and  sum_c g_o[c] do[m, c] xhat[m, c] = dp . (p - ep), with do = dp Wp)
//             per column r0 = sum_m dp, r1 = sum_m dp rs_o mu_o   (dWp = g_o (dpr^T t^T - r1) + b_o r0)
//   lnout_bwd dor = dpr Wp = rs_o do [M, H] token-major, t [H, M] channel-major ->
//             dt [H, M] = g_o dor - rs_o (S1 + xhat S2) / H;  dg_o += dor (t - mu_o),  db_o += dor / rs_o
//   lnin_bwd  dxn [M, D] -> dx = dy + rs_i (g_i dxn - mean(g_i dxn) - xhat mean(g_i dxn xhat)); dg_i, db_i
//   ln_apply  xn = LN_in(x) (unmasked), the A operand of the weight-gradient GEMMs
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace wbwd {

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) { f[2 * j] = bf_lo(w[j]); f[2 * j + 1] = bf_hi(w[j]); }
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(pack(f[0], f[1]), pack(f[2], f[3]), pack(f[4], f[5]), pack(f[6], f[7]));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

constexpr int ROWS = 64;          // rows per block (8 warps x 8 rows) for the row kernels

// ---------------------------------------------------------------------------------------------------------------- gate_bwd
// Element-parallel: thread = 8 contiguous channels of one row (16-byte loads), grid-strided with a total thread count that is
// a multiple of D / 8, so a thread always owns the same 8 columns and keeps their r0 / r1 partials in registers. Row sums
// S1 / S2: segmented warp reduction over the lanes of the same row, one atomic per segment (S1 / S2 zeroed first).
__device__ __forceinline__ float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float sigmoidf_fast(float g) { return rcp_ftz(1.f + ex2_ftz(-1.4426950408889634f * g)); }
__device__ __forceinline__ void seg_sum_atomic(float v, int row, float* dst) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float w = __shfl_down_sync(0xffffffffu, v, o);
    const int r = __shfl_down_sync(0xffffffffu, row, o);
    if (lane + o < 32 && r == row) v += w;
  }
  const int prev = __shfl_up_sync(0xffffffffu, row, 1);
  if (row >= 0 && (lane == 0 || prev != row)) atomicAdd(dst + row, v);   // row -1: lanes past the end (last pass)
}
constexpr int GB_BLOCKS = 148 * 15;          // x 256 threads = a multiple of 192 (= lcm of D / 8 for D = 256 / 384 / 512)
template <int D>
__global__ void __launch_bounds__(256) gate_bwd_kernel(const __nv_bfloat16* __restrict__ dy, const __nv_bfloat16* __restrict__ p,
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
  for (size_t base = 0; base < n; base += T) {            // uniform trip count per warp (shuffles need every lane)
    const size_t e = base + e0;
    const bool ok = e < n;
    const int row = ok ? (int)(e / G8) : -1;
    float s1 = 0.f, s2 = 0.f;
    if (ok) {
      float fy[8], fp[8], fg[8], fd[8];
      unpack8(reinterpret_cast<const uint4*>(dy)[e], fy);
      unpack8(reinterpret_cast<const uint4*>(p)[e], fp);
      unpack8(reinterpret_cast<const uint4*>(g)[e], fg);
      if (ds) unpack8(*reinterpret_cast<const uint4*>(ds + (size_t)(row % L) * D + c8), fd);
      else {
#pragma unroll
        for (int j = 0; j < 8; ++j) fd[j] = 1.f;
      }
      const float ro = rs_o[row], rm = ro * mu_o[row];
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
  }
  __shared__ float cs[2][D];
  for (int k = threadIdx.x; k < 2 * D; k += blockDim.x) (&cs[0][0])[k] = 0.f;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) { atomicAdd(&cs[0][c8 + j], c0[j]); atomicAdd(&cs[1][c8 + j], c1[j]); }
  __syncthreads();
  for (int c = threadIdx.x; c < 2 * D; c += blockDim.x) atomicAdd(part + c, (&cs[0][0])[c]);
}

// ---------------------------------------------------------------------------------------------------------------- lnout_bwd
// dor arrives channel-major (cuBLAS computes dor^T = Wp^T dpr^T), so t, dor and dt all share the [H, M] layout and nothing is
// transposed: a thread owns 8 consecutive tokens of a 256-token group (16-byte accesses; a warp covers 512 contiguous bytes of
// a channel row), keeps their per-token terms in registers and walks the channel rows of its warp; a block strides over the
// token groups and keeps its channels' dg_o / db_o in registers, writing one row of per-block partials (colsum_kernel).
constexpr int LOB_CH = 64;                   // channel rows per block (8 warps x 8)
__global__ void __launch_bounds__(256) lnout_bwd_kernel(const __nv_bfloat16* __restrict__ dor, const __nv_bfloat16* __restrict__ t,
                                                         const float* __restrict__ mu_o, const float* __restrict__ rs_o,
                                                         const float* __restrict__ S1, const float* __restrict__ S2,
                                                         const float* __restrict__ go, __nv_bfloat16* __restrict__ dt,
                                                         float* __restrict__ dgb, int M, int H) {   // dgb: partials [gridDim.x][2H]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c0 = blockIdx.y * LOB_CH + warp * (LOB_CH / 8);
  float sgs[LOB_CH / 8], sbs[LOB_CH / 8];
#pragma unroll
  for (int k = 0; k < LOB_CH / 8; ++k) { sgs[k] = 0.f; sbs[k] = 0.f; }
  for (int grp = blockIdx.x; grp < M / 256; grp += gridDim.x) {
  const int tok = grp * 256 + lane * 8;
  float mu[8], a1[8], a2r[8], irs[8];
  {
    const float4 m0 = *reinterpret_cast<const float4*>(mu_o + tok), m1 = *reinterpret_cast<const float4*>(mu_o + tok + 4);
    const float4 r0 = *reinterpret_cast<const float4*>(rs_o + tok), r1 = *reinterpret_cast<const float4*>(rs_o + tok + 4);
    const float4 p0 = *reinterpret_cast<const float4*>(S1 + tok), p1 = *reinterpret_cast<const float4*>(S1 + tok + 4);
    const float4 q0 = *reinterpret_cast<const float4*>(S2 + tok), q1 = *reinterpret_cast<const float4*>(S2 + tok + 4);
    const float mm[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w}, rr[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
    const float ss1[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w}, ss2[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const float ih = 1.f / H;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      mu[j] = mm[j]; irs[j] = rcp_ftz(rr[j]);
      a1[j] = rr[j] * ss1[j] * ih;
      a2r[j] = rr[j] * rr[j] * ss2[j] * ih;          // xhat S2 rs / H = (t - mu) rs^2 S2 / H
    }
  }
#pragma unroll 2
  for (int k = 0; k < LOB_CH / 8; ++k) {
    const int c = c0 + k;
    const size_t off = (size_t)c * M + tok;
    float dv[8], tv[8], o[8];
    unpack8(*reinterpret_cast<const uint4*>(dor + off), dv);
    unpack8(*reinterpret_cast<const uint4*>(t + off), tv);
    const float gc = go[c];
    float sg = 0.f, sb = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float tc = tv[j] - mu[j];
      o[j] = gc * dv[j] - a1[j] - tc * a2r[j];
      sg = fmaf(dv[j], tc, sg);
      sb = fmaf(dv[j], irs[j], sb);
    }
    *reinterpret_cast<uint4*>(dt + off) = pack8(o);
    sgs[k] += sg; sbs[k] += sb;
  }
  }
#pragma unroll
  for (int k = 0; k < LOB_CH / 8; ++k) {       // one warp sum per channel and block -> this block's row of the partials
    const float sg = warp_sum(sgs[k]), sb = warp_sum(sbs[k]);
    if (lane == 0) { dgb[(size_t)blockIdx.x * 2 * H + c0 + k] = sg; dgb[(size_t)blockIdx.x * 2 * H + H + c0 + k] = sb; }
  }
}

// out[c] = sum_r part[r][c]: block = 32 columns (lanes, coalesced) x 8 warps striding the rows, combined in shared memory.
// Replaces same-address atomics from every token block (serialised in L2: 576-way at L384) and torch's column sum.
__global__ void __launch_bounds__(256) colsum_kernel(const float* __restrict__ part, float* __restrict__ out, int R, int C) {
  __shared__ float red[8][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c = blockIdx.x * 32 + lane;
  float a = 0.f;
  if (c < C)
    for (int r = warp; r < R; r += 8) a += part[(size_t)r * C + c];
  red[warp][lane] = a;
  __syncthreads();
  if (warp == 0 && c < C) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][lane];
    out[c] = s;
  }
}

// ---------------------------------------------------------------------------------------------------------------- lnin_bwd
// Row per warp, RR rows at once: every load of the RR rows (dxn, x, dy) is issued before the first row reduction, so a warp
// keeps RR x 3 x NI 16-byte loads in flight (one row at a time was latency-bound at about half the HBM bandwidth).
template <int D>
__global__ void __launch_bounds__(256) lnin_bwd_kernel(const __nv_bfloat16* __restrict__ dxn, const __nv_bfloat16* __restrict__ x,
                                                        const __nv_bfloat16* __restrict__ dy, const float* __restrict__ mu_i,
                                                        const float* __restrict__ rs_i, const float* __restrict__ gi,
                                                        __nv_bfloat16* __restrict__ dx, float* __restrict__ part, int M) {
  constexpr int G8 = D / 8, NI = (G8 + 31) / 32, RR = 2;
  __shared__ float cs[2][8][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float gg[NI][8], c0[NI][8], c1[NI][8];
#pragma unroll
  for (int it = 0; it < NI; ++it)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int q = lane + 32 * it;
      gg[it][j] = q < G8 ? gi[q * 8 + j] : 0.f;
      c0[it][j] = 0.f; c1[it][j] = 0.f;
    }
  for (int blk = blockIdx.x; blk * ROWS < M; blk += gridDim.x)   // fixed grid: dg_i / db_i partials in a fixed order
  for (int rr = 0; rr < ROWS / 8; rr += RR) {
    const int row0 = blk * ROWS + warp * (ROWS / 8) + rr;
    uint4 vd[RR][NI], vx[RR][NI], vy[RR][NI];
#pragma unroll
    for (int r = 0; r < RR; ++r)
#pragma unroll
      for (int it = 0; it < NI; ++it) {
        const int q = lane + 32 * it, row = row0 + r;
        if (q < G8 && row < M) {
          const size_t e = (size_t)row * G8 + q;
          vd[r][it] = reinterpret_cast<const uint4*>(dxn)[e];
          vx[r][it] = reinterpret_cast<const uint4*>(x)[e];
          vy[r][it] = reinterpret_cast<const uint4*>(dy)[e];
        }
      }
#pragma unroll
    for (int r = 0; r < RR; ++r) {
      const int row = row0 + r;
      if (row >= M) break;
      const float mu = mu_i[row], rs = rs_i[row];
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int it = 0; it < NI; ++it) {
        if (lane + 32 * it >= G8) break;
        float fd[8], fx[8];
        unpack8(vd[r][it], fd); unpack8(vx[r][it], fx);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float xh = (fx[j] - mu) * rs, gd = gg[it][j] * fd[j];
          a += gd; b = fmaf(gd, xh, b);
          c0[it][j] = fmaf(fd[j], xh, c0[it][j]);
          c1[it][j] += fd[j];
        }
      }
      a = warp_sum(a) * (1.f / D); b = warp_sum(b) * (1.f / D);
#pragma unroll
      for (int it = 0; it < NI; ++it) {
        const int q = lane + 32 * it;
        if (q >= G8) break;
        float fd[8], fx[8], fy[8], o[8];
        unpack8(vd[r][it], fd); unpack8(vx[r][it], fx); unpack8(vy[r][it], fy);
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = fy[j] + rs * (gg[it][j] * fd[j] - a - (fx[j] - mu) * rs * b);
        reinterpret_cast<uint4*>(dx)[(size_t)row * G8 + q] = pack8(o);
      }
    }
  }
#pragma unroll
  for (int it = 0; it < NI; ++it) {
    const int q = lane + 32 * it;
    if (q >= G8) break;
#pragma unroll
    for (int j = 0; j < 8; ++j) { cs[0][warp][q * 8 + j] = c0[it][j]; cs[1][warp][q * 8 + j] = c1[it][j]; }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 256) {
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) { a += cs[0][w][c]; b += cs[1][w][c]; }
    part[(size_t)blockIdx.x * 2 * D + c] = a;           // this block's row; colsum_kernel adds the rows in a fixed order
    part[(size_t)blockIdx.x * 2 * D + D + c] = b;
  }
}

// ---------------------------------------------------------------------------------------------------------------- ln_apply
template <int D>
__global__ void __launch_bounds__(256) ln_apply_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ mean,
                                                        const float* __restrict__ rstd, const float* __restrict__ g,
                                                        const float* __restrict__ b, __nv_bfloat16* __restrict__ xn, int M) {
  constexpr int G8 = D / 8;
  __shared__ __align__(16) float gs[2 * D];
  for (int k = threadIdx.x; k < D; k += 256) { gs[k] = g[k]; gs[D + k] = b[k]; }
  __syncthreads();
  const size_t n = (size_t)M * G8;
  for (size_t e = (size_t)blockIdx.x * 256 + threadIdx.x; e < n; e += (size_t)gridDim.x * 256) {
    const int row = (int)(e / G8), c8 = (int)(e % G8) * 8;
    float f[8];
    unpack8(reinterpret_cast<const uint4*>(x)[e], f);
    const float mu = mean[row], rs = rstd[row];
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = fmaf((f[j] - mu) * rs, gs[c8 + j], gs[D + c8 + j]);
    reinterpret_cast<uint4*>(xn)[e] = pack8(f);
  }
}

}  // namespace wbwd

#define WB_WIDTHS(X) X(256) X(384) X(512)
#define BF(t) reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define CBF(t) reinterpret_cast<const __nv_bfloat16*>((t).data_ptr())

// dy, p, g [M, D]; ds [L, D] or none; vec [4, D] (k3w's sp, ep, ...); mu_o / rs_o [M] -> dpr [M, D]; dg -> dg_out[:, :D]
// (row stride ldg); S1 / S2 [M]; r01 [2D] = (r0 ; r1).
void wide_gate_bwd(torch::Tensor dy, torch::Tensor p, torch::Tensor g, c10::optional<torch::Tensor> ds, torch::Tensor vec,
                   torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor dpr, torch::Tensor dg_out, torch::Tensor S1,
                   torch::Tensor S2, torch::Tensor r01, int64_t L) {
  using namespace wbwd;
  const int M = (int)dy.size(0), D = (int)dy.size(1);
  TORCH_CHECK(dg_out.stride(1) == 1 && dg_out.stride(0) % 8 == 0);
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* dsp = ds.has_value() ? CBF(*ds) : nullptr;
  S1.zero_(); S2.zero_(); r01.zero_();
#define GB(DD) if (D == DD) { gate_bwd_kernel<DD><<<GB_BLOCKS, 256, 0, st>>>(CBF(dy), CBF(p), CBF(g), dsp, \
    vec.data_ptr<float>(), mu_o.data_ptr<float>(), rs_o.data_ptr<float>(), BF(dpr), BF(dg_out), (int)dg_out.stride(0), \
    S1.data_ptr<float>(), S2.data_ptr<float>(), r01.data_ptr<float>(), M, (int)L); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  WB_WIDTHS(GB)
#undef GB
  TORCH_CHECK(false, "wide_gate_bwd: unsupported D ", D);
}

// dor = (dpr Wp)^T [H, M] channel-major, t [H, M]; -> dt [H, M]; dgb [2H] = (dg_o ; db_o)
void wide_lnout_bwd(torch::Tensor dout, torch::Tensor t, torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor S1, torch::Tensor S2,
                    torch::Tensor go, torch::Tensor dt, torch::Tensor dgb) {
  using namespace wbwd;
  const int H = (int)t.size(0), M = (int)(t.numel() / H);
  TORCH_CHECK(M % 256 == 0 && H % 64 == 0 && dout.is_contiguous() && t.is_contiguous() && dt.is_contiguous());
  const int gx = M / 256;
  auto part = torch::empty({(int64_t)gx * 2 * H}, mu_o.options());
lnout_bwd_kernel<<<dim3(gx, H / LOB_CH), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      CBF(dout), CBF(t), mu_o.data_ptr<float>(), rs_o.data_ptr<float>(), S1.data_ptr<float>(), S2.data_ptr<float>(),
      go.data_ptr<float>(), BF(dt), part.data_ptr<float>(), M, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  colsum_kernel<<<(2 * H + 31) / 32, 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), dgb.data_ptr<float>(), gx, 2 * H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dxn, x, dy [M, D]; -> dx [M, D]; dgb [2D] = (dg_i ; db_i)
void wide_lnin_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor dy, torch::Tensor mu_i, torch::Tensor rs_i, torch::Tensor gi,
                   torch::Tensor dx, torch::Tensor dgb) {
  using namespace wbwd;
  const int M = (int)x.size(0), D = (int)x.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  // a fixed grid (not one block per 64 rows) so the dg_i / db_i reduction order, and the result, do not depend on scheduling:
  // they are fp32 LayerNorm parameters, where run-to-run atomic-order noise (~1e-4 relative) is visible
  const int nb = std::min((M + ROWS - 1) / ROWS, 148 * 4);
  auto part = torch::empty({(int64_t)nb * 2 * D}, mu_i.options());
#define LB(DD) if (D == DD) { lnin_bwd_kernel<DD><<<nb, 256, 0, st>>>(CBF(dxn), CBF(x), CBF(dy), \
    mu_i.data_ptr<float>(), rs_i.data_ptr<float>(), gi.data_ptr<float>(), BF(dx), part.data_ptr<float>(), M); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    colsum_kernel<<<(2 * D + 31) / 32, 256, 0, st>>>(part.data_ptr<float>(), dgb.data_ptr<float>(), nb, 2 * D); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  WB_WIDTHS(LB)
#undef LB
  TORCH_CHECK(false, "wide_lnin_bwd: unsupported D ", D);
}

void wide_ln_apply(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, torch::Tensor g, torch::Tensor b, torch::Tensor xn) {
  using namespace wbwd;
  const int M = (int)x.size(0), D = (int)x.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
#define LA(DD) if (D == DD) { ln_apply_kernel<DD><<<148 * 8, 256, 0, st>>>(CBF(x), mean.data_ptr<float>(), rstd.data_ptr<float>(), \
    g.data_ptr<float>(), b.data_ptr<float>(), BF(xn), M); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  WB_WIDTHS(LA)
#undef LA
  TORCH_CHECK(false, "wide_ln_apply: unsupported D ", D);
}
