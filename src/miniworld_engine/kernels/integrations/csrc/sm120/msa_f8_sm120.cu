// msa_f8_sm120.cu -- RTX 5090 (sm_120a) helpers of the opt-in e4m3 OuterProductMean / MSA pair-weighted-averaging routes (docs/fp8.md "RTX 5090").
//
// The 5090's cuBLASLt takes e4m3 operands in the TN form only (both K-major), so the s-major operands of the default kernels are turned around once:
//
//   transpose_quant   X [B, R, N] bf16 -> Y [B, N, Rp] e4m3:  Y[b, n, r] = e4m3_satfinite(X[b, r, n] * q)  (r < R; 0 for R <= r < Rp, Rp % 16 == 0),
//                     the max |X q| over the call kept in amax (atomicMax on the fp32 bits: the e4m3 saturation guard)
//                     OPM: A, B [S, 32 L] -> A2q, BTq [32 L, S]  (O = A^T B over K = S);  PWA BIG: v_hm [8, L, S 32] -> vT [8, S 32, L]  (o = w v over K = L)
//   w_quant           the softmax weights w (probabilities) -> e4m3 of 256 w (no saturation: 256 < 448)
//   ln_vt8            LN(m) . Wv^T for 64 tokens of one MSA row, written as e4m3 vT [8, S C, N] (j contiguous) with the static v scale (the 8 x 8 / 16 / 32 widths
//                     of pwa_sm80: d_msa 64 / 128); vT tile = Wv . y^T on mma.sync bf16 (M = 8 C channels, N = 64 tokens, K = d_msa)
//
// Nothing here is reached unless the opt-in route is on (opm_fp8_sm120 / pwa_fp8_sm120).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "../../../_sm_count.cuh"

#define DEVI __device__ __forceinline__

namespace {

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI uint16_t e4m3x2(float lo, float hi) {   // low byte = lo
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
DEVI void amax_store(float* amax, float v) {   // v >= 0 (NaN: kept as the largest bits, the guard trips)
  atomicMax(reinterpret_cast<int*>(amax), __float_as_int(v));
}

// ------------------------------------------------------------------------------------------------------------ transpose_quant
// one CTA: 64 rows r x 256 columns n of one batch; thread t owns column n0 + t: reads its 64 values down the staged tile, writes 64 bytes of Y's row.
constexpr int TQ_R = 64, TQ_N = 256, TQ_ROW = TQ_N * 2 + 16;   // staged row: 512 B + 16 B pad (the column reads of a warp hit 16 consecutive words)

__global__ void __launch_bounds__(256) transpose_quant_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ y, const float* __restrict__ qp,
                                                               float* __restrict__ amax, int R, int N, int Rp) {
  __shared__ __align__(16) unsigned char tile[TQ_R * TQ_ROW];
  const int tid = threadIdx.x, n0 = blockIdx.x * TQ_N, r0 = blockIdx.y * TQ_R, b = blockIdx.z;
  const __nv_bfloat16* xb = x + (long)b * R * N;
  // load: 64 rows x 32 chunks of 16 B
#pragma unroll
  for (int it = 0; it < TQ_R * (TQ_N / 8) / 256; ++it) {
    const int u = it * 256 + tid, row = u / (TQ_N / 8), ch = u % (TQ_N / 8);
    const int r = r0 + row, n = n0 + ch * 8;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < R && n < N) v = __ldg(reinterpret_cast<const uint4*>(xb + (long)r * N + n));
    *reinterpret_cast<uint4*>(tile + row * TQ_ROW + ch * 16) = v;
  }
  __syncthreads();
  const int n = n0 + tid;
  const float q = *qp;
  float mx = 0.f;
  if (n < N) {
    uint8_t* yr = y + ((long)b * N + n) * Rp + r0;
    const unsigned char* col = tile + (tid >> 1) * 4;
    const int half = tid & 1;
#pragma unroll
    for (int c = 0; c < TQ_R / 16; ++c) {
      if (r0 + 16 * c >= Rp) break;
      uint32_t w[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float f[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t word = *reinterpret_cast<const uint32_t*>(col + (16 * c + 4 * k + e) * TQ_ROW);
          f[e] = (half ? bf16hi(word) : bf16lo(word)) * q;
          mx = fmaxf(mx, fabsf(f[e]));
        }
        w[k] = (uint32_t)e4m3x2(f[0], f[1]) | ((uint32_t)e4m3x2(f[2], f[3]) << 16);
      }
      *reinterpret_cast<uint4*>(yr + 16 * c) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
  mx = warp_max(mx);
  if ((tid & 31) == 0 && mx > 0.f) amax_store(amax, mx);
}

void transpose_quant(torch::Tensor x, torch::Tensor y, torch::Tensor q, torch::Tensor amax) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3, "transpose_quant: x bf16 [B, R, N] contiguous");
  TORCH_CHECK(y.is_cuda() && y.element_size() == 1 && y.is_contiguous() && y.dim() == 3, "transpose_quant: y 1-byte [B, N, Rp] contiguous");
  const int B = (int)x.size(0), R = (int)x.size(1), N = (int)x.size(2), Rp = (int)y.size(2);
  TORCH_CHECK(y.size(0) == B && y.size(1) == N && Rp >= R && Rp % 16 == 0 && N % 8 == 0, "transpose_quant: y [B, N, Rp >= R], Rp % 16, N % 8");
  TORCH_CHECK(q.scalar_type() == at::kFloat && amax.scalar_type() == at::kFloat && q.numel() >= 1 && amax.numel() >= 1, "transpose_quant: q / amax fp32");
  const c10::cuda::CUDAGuard guard(x.device());
  dim3 grid((N + TQ_N - 1) / TQ_N, (Rp + TQ_R - 1) / TQ_R, B);
  TORCH_CHECK(grid.y <= 65535 && grid.z <= 65535, "transpose_quant: grid");
  transpose_quant_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint8_t*>(y.data_ptr()), q.data_ptr<float>(), amax.data_ptr<float>(), R, N, Rp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------------------ w_quant
__global__ void w_quant_kernel(const uint4* __restrict__ w, uint2* __restrict__ out, long n8, float scale) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const uint4 v = __ldg(w + i);
    uint2 o;
    o.x = (uint32_t)e4m3x2(bf16lo(v.x) * scale, bf16hi(v.x) * scale) | ((uint32_t)e4m3x2(bf16lo(v.y) * scale, bf16hi(v.y) * scale) << 16);
    o.y = (uint32_t)e4m3x2(bf16lo(v.z) * scale, bf16hi(v.z) * scale) | ((uint32_t)e4m3x2(bf16lo(v.w) * scale, bf16hi(v.w) * scale) << 16);
    out[i] = o;
  }
}

void w_quant(torch::Tensor w, torch::Tensor out, double scale) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() % 8 == 0, "w_quant: bf16 contiguous, numel % 8");
  TORCH_CHECK(out.is_cuda() && out.element_size() == 1 && out.is_contiguous() && out.numel() == w.numel(), "w_quant: out 1-byte, same numel");
  const c10::cuda::CUDAGuard guard(w.device());
  const long n8 = w.numel() / 8;
  const int grid = (int)std::min<long>((n8 + 255) / 256, (long)mw::sm_count() * 8);
  w_quant_kernel<<<std::max(grid, 1), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<uint2*>(out.data_ptr()),
                                                                               n8, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------------------ ln_vt8
// vT[h][s C + c][j] = e4m3( (LN(m[s, j, :]) . Wv[h C + c, :]) * vinv ),  m [S, N, D] bf16, Wv [8 C, D] bf16, vT [8, S C, N] e4m3 (N % 16 == 0).
// One tile = MSA row s x NT consecutive tokens (a contiguous block of m); persistent CTAs, the next tile by cp.async into the other buffer. The LayerNorm runs in
// place on the staged rows (y bf16 = the default ln_v's rounding), then vT tile = Wv . y^T on mma.sync m16n8k16 (A = Wv rows, B = y rows: both K-major in
// shared memory), fp32 accumulators times vinv -> e4m3 pairs (two consecutive tokens) staged as [8 C][NT bytes] and stored as 16-byte runs along j.
DEVI void cp_async16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <int NCH> DEVI uint32_t swzn(uint32_t row, uint32_t chunk) { return row * (16u * NCH) + ((chunk ^ (row & 7u)) << 4); }

struct Vt8Params {
  const __nv_bfloat16* m;   // [S][N][D]
  const float* lnw;         // [D]
  const float* lnb;         // [D]
  const __nv_bfloat16* wv;  // [8 C][D]
  uint8_t* vt;              // [8][S C][N]
  const float* vinv;        // [1]
  float* amax;              // [1]
  int S, N, ntj, ntile;
  float eps;
};

template <int D, int C, int NT> struct Vt8Cfg {
  static constexpr int HC = 8 * C, NCH = D / 8, MB = NT * D * 2, WB = HC * D * 2, OB = HC * NT;
  static constexpr int SMEM = 2 * MB + WB + OB + 2 * D * 4;
  static constexpr int WM = HC / 16, WN = NT / 8, WMW = WM < 8 ? WM : 8, WNW = 8 / WMW, MT = WM / WMW, NTT = WN / WNW;
  static_assert((D == 64 || D == 128) && (C == 8 || C == 16 || C == 32) && (NT == 32 || NT == 64), "width");
  static_assert(NTT % 2 == 0 && NTT >= 2, "B fragments come in pairs of n8 tiles");
  static_assert(SMEM <= 101376, "sm_120 shared memory");
};

// staged output row: NT bytes (NT / 16 chunks), the chunk index XORed with bits of the row so the 8 rows of an accumulator store spread over the banks
template <int NT> DEVI uint32_t swzo(uint32_t row, uint32_t chunk) {
  constexpr uint32_t NC = NT / 16;
  return row * NT + ((chunk ^ ((row >> (NC == 4 ? 1 : 2)) & (NC - 1))) << 4);
}

template <int D, int C, int NT>
DEVI void vt8_load(const Vt8Params& p, uint32_t dst, int tl, int tid) {
  using G = Vt8Cfg<D, C, NT>;
  const int s = tl / p.ntj, j0 = (tl - s * p.ntj) * NT;
#pragma unroll
  for (int it = 0; it < NT * G::NCH / 256; ++it) {
    const int u = it * 256 + tid, t = u / G::NCH, ch = u % G::NCH;
    const bool ok = j0 + t < p.N;
    cp_async16(dst + swzn<G::NCH>(t, ch), p.m + (ok ? ((long)s * p.N + j0 + t) * D + ch * 8 : 0), ok ? 16u : 0u);
  }
}

template <int D, int C, int NT, int MINB>
__global__ void __launch_bounds__(256, MINB) ln_vt8_kernel(const Vt8Params p) {
  using G = Vt8Cfg<D, C, NT>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sm0 = smem_u32(smem), sw = sm0 + 2 * G::MB, so = sw + G::WB;
  float* sgb = reinterpret_cast<float*>(smem + 2 * G::MB + G::WB + G::OB);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  for (int u = tid; u < G::HC * G::NCH; u += 256) {
    const int row = u / G::NCH, ch = u % G::NCH;
    cp_async16(sw + swzn<G::NCH>(row, ch), p.wv + (long)row * D + ch * 8, 16u);
  }
  for (int k = tid; k < D; k += 256) { sgb[k] = p.lnw[k]; sgb[D + k] = p.lnb[k]; }
  int tl = blockIdx.x, buf = 0;
  if (tl < p.ntile) vt8_load<D, C, NT>(p, sm0, tl, tid);
  cp_async_commit();
  const float vinv = *p.vinv;
  float mx = 0.f;
  const int wm = warp % G::WMW, wn = warp / G::WMW;   // this warp: m16 tiles wm MT .. , n8 tiles wn NTT ..

#pragma unroll 1
  for (; tl < p.ntile; tl += gridDim.x, buf ^= 1) {
    const int s = tl / p.ntj, j0 = (tl - s * p.ntj) * NT;
    const int nx = tl + gridDim.x;
    if (nx < p.ntile) vt8_load<D, C, NT>(p, sm0 + (buf ^ 1) * G::MB, nx, tid);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint32_t sm = sm0 + buf * G::MB;

    // ---- LayerNorm in place: warp w normalises tokens w, w + 8, ...; lane holds D / 32 consecutive values
#pragma unroll
    for (int tt = 0; tt < NT / 8; ++tt) {
      const int t = warp + 8 * tt;
      constexpr int PER = D / 32;   // 2 or 4 bf16
      const uint32_t addr = sm + swzn<G::NCH>(t, (lane * PER) / 8) + ((lane * PER) % 8) * 2;
      float x[PER];
      if constexpr (PER == 2) {
        uint32_t w;
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(w) : "r"(addr));
        x[0] = bf16lo(w); x[1] = bf16hi(w);
      } else {
        uint32_t w0, w1;
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(w0), "=r"(w1) : "r"(addr));
        x[0] = bf16lo(w0); x[1] = bf16hi(w0); x[2] = bf16lo(w1); x[3] = bf16hi(w1);
      }
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < PER; ++e) sum += x[e];
      const float mean = warp_sum(sum) * (1.f / D);
      float var = 0.f;
#pragma unroll
      for (int e = 0; e < PER; ++e) { const float d = x[e] - mean; var = fmaf(d, d, var); }
      const float rs = rsqrtf(warp_sum(var) * (1.f / D) + p.eps);
      const int k0 = lane * PER;
      uint32_t y[PER / 2];
#pragma unroll
      for (int e = 0; e < PER / 2; ++e)
        y[e] = pack_bf16(fmaf((x[2 * e] - mean) * rs, sgb[k0 + 2 * e], sgb[D + k0 + 2 * e]),
                         fmaf((x[2 * e + 1] - mean) * rs, sgb[k0 + 2 * e + 1], sgb[D + k0 + 2 * e + 1]));
      if constexpr (PER == 2) {
        asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(addr), "r"(y[0]) : "memory");
      } else {
        asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" ::"r"(addr), "r"(y[0]), "r"(y[1]) : "memory");
      }
    }
    __syncthreads();

    // ---- vT tile = Wv . y^T
    float acc[G::MT][G::NTT][4];
#pragma unroll
    for (int a = 0; a < G::MT; ++a)
#pragma unroll
      for (int b = 0; b < G::NTT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      uint32_t af[G::MT][4];
#pragma unroll
      for (int a = 0; a < G::MT; ++a) ldsm_x4(af[a], sw + swzn<G::NCH>((wm * G::MT + a) * 16 + (lane & 15), 2 * ks + (lane >> 4)));
#pragma unroll
      for (int bp = 0; bp < G::NTT / 2; ++bp) {
        uint32_t bf[4];
        ldsm_x4(bf, sm + swzn<G::NCH>((wn * G::NTT + 2 * bp) * 8 + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
#pragma unroll
        for (int a = 0; a < G::MT; ++a) {
          mma16816(acc[a][2 * bp], af[a], bf[0], bf[1]);
          mma16816(acc[a][2 * bp + 1], af[a], bf[2], bf[3]);
        }
      }
    }
    // ---- scale, amax, e4m3 pairs into the staged rows
#pragma unroll
    for (int a = 0; a < G::MT; ++a) {
      const int r0 = (wm * G::MT + a) * 16 + gq, r1 = r0 + 8;
#pragma unroll
      for (int b = 0; b < G::NTT; ++b) {
        const int tok = (wn * G::NTT + b) * 8 + 2 * q;   // first of the two tokens
        float v0 = acc[a][b][0] * vinv, v1 = acc[a][b][1] * vinv, v2 = acc[a][b][2] * vinv, v3 = acc[a][b][3] * vinv;
        if (j0 + tok < p.N) mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));   // N % 16: both tokens valid or none
        const uint16_t e01 = e4m3x2(v0, v1), e23 = e4m3x2(v2, v3);
        const uint32_t ch = tok >> 4, byte = tok & 15;
        asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(so + swzo<NT>(r0, ch) + byte), "h"(e01) : "memory");
        asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(so + swzo<NT>(r1, ch) + byte), "h"(e23) : "memory");
      }
    }
    __syncthreads();
    // ---- stores: row hc = h C + c of the tile -> vT[h][s C + c][j0 ..]
#pragma unroll
    for (int it = 0; it < (G::HC * (NT / 16) + 255) / 256; ++it) {
      const int u = it * 256 + tid;
      if (u < G::HC * (NT / 16)) {
        const int row = u / (NT / 16), ch = u % (NT / 16), j = j0 + 16 * ch;
        if (j < p.N) {
          const int h = row / C, c = row % C;
          uint4 v;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(so + swzo<NT>(row, ch)));
          *reinterpret_cast<uint4*>(p.vt + (((long)h * p.S + s) * C + c) * p.N + j) = v;
        }
      }
    }
  }
  cp_async_wait<0>();
  mx = warp_max(mx);
  if (lane == 0 && mx > 0.f) amax_store(p.amax, mx);
}

template <int D, int C, int NT>
void launch_vt8(const Vt8Params& p, int blocks_per_sm, cudaStream_t st) {
  using G = Vt8Cfg<D, C, NT>;
  auto k = blocks_per_sm >= 2 ? ln_vt8_kernel<D, C, NT, 2> : ln_vt8_kernel<D, C, NT, 1>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
  int occ = 1;   // persistent CTAs: never more than are resident at once (a second wave would start only after a whole first-wave loop)
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, 256, G::SMEM));
  const int grid = std::min(p.ntile, mw::sm_count() * std::max(1, std::min(occ, blocks_per_sm)));
  k<<<std::max(grid, 1), 256, G::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_vt8(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, torch::Tensor vinv, torch::Tensor amax, torch::Tensor vt,
            int64_t nt, int64_t blocks_per_sm) {
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == at::kBFloat16 && m.is_contiguous() && m.dim() == 3, "ln_vt8: m bf16 [S, N, D]");
  const int S = (int)m.size(0), N = (int)m.size(1), D = (int)m.size(2);
  TORCH_CHECK(wv.scalar_type() == at::kBFloat16 && wv.is_contiguous() && wv.dim() == 2 && wv.size(1) == D && wv.size(0) % 8 == 0, "ln_vt8: wv bf16 [8 C, D]");
  const int C = (int)wv.size(0) / 8;
  TORCH_CHECK(lnw.scalar_type() == at::kFloat && lnb.scalar_type() == at::kFloat && lnw.numel() == D && lnb.numel() == D, "ln_vt8: LayerNorm fp32 [D]");
  TORCH_CHECK(vt.element_size() == 1 && vt.is_contiguous() && vt.numel() == 8LL * S * C * N && N % 16 == 0, "ln_vt8: vt 1-byte [8, S C, N], N % 16");
  TORCH_CHECK(vinv.scalar_type() == at::kFloat && amax.scalar_type() == at::kFloat, "ln_vt8: vinv / amax fp32");
  const c10::cuda::CUDAGuard guard(m.device());
  Vt8Params p{};
  p.m = reinterpret_cast<const __nv_bfloat16*>(m.data_ptr());
  p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>();
  p.wv = reinterpret_cast<const __nv_bfloat16*>(wv.data_ptr());
  p.vt = reinterpret_cast<uint8_t*>(vt.data_ptr());
  p.vinv = vinv.data_ptr<float>(); p.amax = amax.data_ptr<float>();
  p.S = S; p.N = N; p.eps = (float)eps;
  p.ntj = (N + (int)nt - 1) / (int)nt;
  TORCH_CHECK((long)S * p.ntj < (1L << 31), "ln_vt8: tiles");
  p.ntile = S * p.ntj;
  auto st = at::cuda::getCurrentCUDAStream();
  const int bps = (int)blocks_per_sm;
#define VT8(D_, C_, NT_) if (D == D_ && C == C_ && nt == NT_) { launch_vt8<D_, C_, NT_>(p, bps, st); return; }
  VT8(64, 8, 32) VT8(64, 8, 64) VT8(64, 16, 32) VT8(64, 16, 64) VT8(64, 32, 32) VT8(64, 32, 64)
  VT8(128, 8, 32) VT8(128, 8, 64) VT8(128, 16, 32) VT8(128, 16, 64)
#undef VT8
  TORCH_CHECK(false, "ln_vt8: unsupported (d_msa, C, NT)");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("transpose_quant", &transpose_quant);
  mod.def("w_quant", &w_quant);
  mod.def("ln_vt8", &ln_vt8);
}
