// ESMFold2 SWA atom block, the hoisted adaLN modulation (CUDA sm_90a, mma.sync bf16 tensor cores, fp32 accumulate), bf16 c / Wmod:
//   forward   mod[R, 768] fp32 = silu(c)[R, 128] . Wmod[768, 128]^T                       (``mod_fwd``; ``save`` also keeps silu(c))
//   backward  dc[R, 128]  = (dmod[R, 768] . Wmod) * silu'(c)                              (``mod_bwd``, dc role)
//             dWmod[768, 128] = dmod^T . silu(c)                                          (``mod_bwd``, dW role + dw_reduce)
// Numerics: silu is rounded to bf16 (as F.silu on a bf16 tensor), so every product with Wmod is exact in fp32 and the forward is the
// fp32 GEMM up to summation order. dmod (fp32) is split into three bf16 planes (24 bits: exact), so dc and dWmod are the fp32 GEMMs'
// too -- two planes (2^-17) would already sit below the bf16 rounding of the results, but the rounding boundaries then move with
// run-to-run differences of dmod's atomic accumulation order and the gradient stops repeating to 1e-6.
// R (atoms of the batch, B*S) is arbitrary; rows past R are clamped on load and masked on store.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include "swa_ptx.cuh"
#include "pdl.cuh"

namespace {
constexpr int C = 128, MW = 768;
constexpr int LDS = 136;                 // padded smem row stride in bf16 elements (272 B: ldmatrix is conflict-free)
using bf16 = __nv_bfloat16;

// 16-byte async copy global -> shared; ``ok`` false zero-fills the destination (the source is not read)
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok = true) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
using swa_mma::ldsm4;  // swa_ptx.cuh (cp16 above has a default argument: its own text)
using swa_mma::ldsm4t;
using swa_mma::mma;
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x * sigmoid_f(x); }
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

// split a pair of fp32 values into three bf16 planes (v = p0 + p1 + p2 EXACTLY), each packed as a bf16x2 word. Truncating masks
// instead of round-to-nearest: the remainder after each 8-bit plane is exact in fp32, so the three planes carry all 24 bits and the
// split is five integer / FP ops per element and three permutes per pair.
__device__ __forceinline__ void split3(float v0, float v1, uint32_t& p0, uint32_t& p1, uint32_t& p2) {
  const uint32_t h0 = __float_as_uint(v0) & 0xffff0000u, h1 = __float_as_uint(v1) & 0xffff0000u;
  const float r0 = v0 - __uint_as_float(h0), r1 = v1 - __uint_as_float(h1);
  const uint32_t m0 = __float_as_uint(r0) & 0xffff0000u, m1 = __float_as_uint(r1) & 0xffff0000u;
  const float s0 = r0 - __uint_as_float(m0), s1 = r1 - __uint_as_float(m1);
  p0 = __byte_perm(h0, h1, 0x7632);
  p1 = __byte_perm(m0, m1, 0x7632);
  p2 = __byte_perm(__float_as_uint(s0), __float_as_uint(s1), 0x7632);      // s has <= 8 significant bits: its top half is exact
}

// ------------------------------------------------------------------------------------------------------------------------------
// forward: persistent over 64-row tiles. CTA = (row group blockIdx.x, channel tile blockIdx.y of 128 modulation channels); the Wmod
// chunk [128 n][128 k] stays in shared memory and the CTA walks the tiles blockIdx.x, += gridDim.x. 4 warps x 16 rows; silu(c) is built
// in registers from global loads issued one tile ahead, and (channel tile 0, ``a_out``) written out as bf16 for the backward.
__global__ void __launch_bounds__(128) mod_fwd_kernel(const bf16* __restrict__ c, const bf16* __restrict__ w, float* __restrict__ out,
                                                      bf16* __restrict__ a_out, int R) {
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* wh = reinterpret_cast<bf16*>(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int ct = blockIdx.y;
  {
    const bf16* W = w + (size_t)ct * 128 * C;
    for (int i = tid; i < 128 * 16; i += 128) cp16(sa(wh + (i >> 4) * LDS + (i & 15) * 8), W + (i >> 4) * C + (i & 15) * 8);
    cp_commit();
  }
  uint32_t ub[8][4];                                         // raw c pairs: a0..a3 of k step ks
  auto fetch = [&](int tile) {
    const int ra = min(tile * 64 + warp * 16 + g, R - 1), rb = min(tile * 64 + warp * 16 + g + 8, R - 1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k = ks * 16 + 2 * t;
      ub[ks][0] = __ldg(reinterpret_cast<const unsigned*>(c + (size_t)ra * C + k));
      ub[ks][1] = __ldg(reinterpret_cast<const unsigned*>(c + (size_t)rb * C + k));
      ub[ks][2] = __ldg(reinterpret_cast<const unsigned*>(c + (size_t)ra * C + k + 8));
      ub[ks][3] = __ldg(reinterpret_cast<const unsigned*>(c + (size_t)rb * C + k + 8));
    }
  };
  const int ntiles = (R + 63) / 64;
  int tile = blockIdx.x;
  if (tile < ntiles) fetch(tile);
  cp_wait<0>();
  __syncthreads();
  for (; tile < ntiles; tile += gridDim.x) {
    uint32_t ah[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int j = 0; j < 4; ++j) ah[ks][j] = pack(silu_f(bf_lo(ub[ks][j])), silu_f(bf_hi(ub[ks][j])));
    if (tile + (int)gridDim.x < ntiles) fetch(tile + gridDim.x);
    if (a_out != nullptr && ct == 0) {
      const int ra = tile * 64 + warp * 16 + g, rb = ra + 8;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int k = ks * 16 + 2 * t;
        if (ra < R) {
          *reinterpret_cast<uint32_t*>(a_out + (size_t)ra * C + k) = ah[ks][0];
          *reinterpret_cast<uint32_t*>(a_out + (size_t)ra * C + k + 8) = ah[ks][2];
        }
        if (rb < R) {
          *reinterpret_cast<uint32_t*>(a_out + (size_t)rb * C + k) = ah[ks][1];
          *reinterpret_cast<uint32_t*>(a_out + (size_t)rb * C + k + 8) = ah[ks][3];
        }
      }
    }
    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {                       // two n8 tiles per ldmatrix.x4
        uint32_t b[4];
        const int n = np * 16 + (lane & 7) + (lane >> 4) * 8, k = ks * 16 + ((lane >> 3) & 1) * 8;
        ldsm4(sa(wh + n * LDS + k), b[0], b[1], b[2], b[3]);
        mma(acc[2 * np], ah[ks], b[0], b[1]);
        mma(acc[2 * np + 1], ah[ks], b[2], b[3]);
      }
    }
    const int oa = tile * 64 + warp * 16 + g, ob = oa + 8;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int col = ct * 128 + nt * 8 + 2 * t;
      if (oa < R) __stcs(reinterpret_cast<float2*>(out + (size_t)oa * MW + col), make_float2(acc[nt][0], acc[nt][1]));
      if (ob < R) __stcs(reinterpret_cast<float2*>(out + (size_t)ob * MW + col), make_float2(acc[nt][2], acc[nt][3]));
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
// backward: ONE launch, two roles by blockIdx.x (256 threads, 8 warps each; both read dmod once, through shared-memory bf16 planes):
//  dc  (blockIdx.x < n_dc): 32 rows x 128 channels, K = 768 in six chunks of 128. The chunk's dmod [32][128] (coalesced float4, one
//      chunk ahead in registers) is split into three bf16 planes in shared memory; the Wmod chunk [128 k][128 n] arrives by cp.async
//      (double buffered); silu'(c) in the epilogue.
//  dW  (else): (channel tile of 128 modulation channels, row range): acc[128 ch][128 n] += planes^T . silu(c) over tiles of KC rows.
//      dmod (fp32) and the saved silu(c) arrive by cp.async through a 3-stage ring, one barrier before the planes are built and one
//      before the MMAs; partial sums go to ws[range][768][128] and are reduced in order by mod_dw_reduce_kernel.
constexpr int KC = 32;                                       // rows per dW tile
constexpr int NRING = 3;
constexpr int RAWB = KC * C * 4;                             // bytes of one fp32 raw tile [KC][128]
constexpr int ABYT = KC * LDS * 2;                           // bytes of one silu stage [KC][LDS] bf16
constexpr int PBYT = 3 * KC * LDS * 2;                       // the three dmod planes
constexpr int SMEM_DW = NRING * RAWB + NRING * ABYT + PBYT;
constexpr int SMEM_DC = 2 * 128 * LDS * 2 + 2 * 3 * 32 * LDS * 2;
constexpr int SMEM_BWD = SMEM_DW > SMEM_DC ? SMEM_DW : SMEM_DC;

__global__ void __launch_bounds__(256) mod_bwd_kernel(const float* __restrict__ gm, const bf16* __restrict__ c, const bf16* __restrict__ a_,
                                                      const bf16* __restrict__ w, bf16* __restrict__ dc, float* __restrict__ ws, int R,
                                                      int n_dc, int rpc) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int mi = lane >> 3;
  if (blockIdx.x < n_dc) {
    // -------------------------------------------------------------- dc
    constexpr int WST = 128 * LDS;                           // one W stage [128 k][LDS]
    bf16* wb = reinterpret_cast<bf16*>(smem);                // 2 stages
    bf16* pl = wb + 2 * WST;                                 // 2 buffers x 3 planes x [32][LDS]
    const int r0 = blockIdx.x * 32, wr = (warp & 1) * 16, wc = (warp >> 1) * 32;
    auto load_w = [&](int kc, int buf) {
      bf16* dst = wb + buf * WST;
      const bf16* W = w + (size_t)kc * 128 * C;
      for (int i = tid; i < 128 * 16; i += 256) cp16(sa(dst + (i >> 4) * LDS + (i & 15) * 8), W + (i >> 4) * C + (i & 15) * 8);
      cp_commit();
    };
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    float4 gq[4];                                            // 32 rows x 32 float4: thread -> (row = j * 8 + tid / 32, float4 = tid % 32)
    auto fetch = [&](int kc) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int row = min(r0 + j * 8 + (tid >> 5), R - 1);
        gq[j] = __ldg(reinterpret_cast<const float4*>(gm + (size_t)row * MW + kc * 128 + (tid & 31) * 4));
      }
    };
    load_w(0, 0);
    pdl_gate();
    fetch(0);
#pragma unroll 1
    for (int kc = 0; kc < 6; ++kc) {
      bf16* pb = pl + (kc & 1) * 3 * 32 * LDS;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int row = j * 8 + (tid >> 5), q = tid & 31;
        uint32_t p00, p01, p02, p10, p11, p12;
        split3(gq[j].x, gq[j].y, p00, p01, p02);
        split3(gq[j].z, gq[j].w, p10, p11, p12);
        *reinterpret_cast<uint2*>(pb + row * LDS + q * 4) = make_uint2(p00, p10);
        *reinterpret_cast<uint2*>(pb + (32 + row) * LDS + q * 4) = make_uint2(p01, p11);
        *reinterpret_cast<uint2*>(pb + (64 + row) * LDS + q * 4) = make_uint2(p02, p12);
      }
      if (kc + 1 < 6) { load_w(kc + 1, (kc + 1) & 1); fetch(kc + 1); cp_wait<1>(); } else cp_wait<0>();
      __syncthreads();
      const bf16* sh = wb + (kc & 1) * WST;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t a0[4], a1[4], a2[4];
        const int arow = wr + (lane & 7) + ((lane >> 3) & 1) * 8, acol = ks * 16 + (lane >> 4) * 8;
        ldsm4(sa(pb + arow * LDS + acol), a0[0], a0[1], a0[2], a0[3]);
        ldsm4(sa(pb + (32 + arow) * LDS + acol), a1[0], a1[1], a1[2], a1[3]);
        ldsm4(sa(pb + (64 + arow) * LDS + acol), a2[0], a2[1], a2[2], a2[3]);
#pragma unroll
        for (int np = 0; np < 2; ++np) {                     // two n8 tiles per x4.trans
          const int kk = ks * 16 + (mi & 1) * 8 + (lane & 7), nn = wc + np * 16 + (mi >> 1) * 8;
          uint32_t b[4];
          ldsm4t(sa(sh + kk * LDS + nn), b[0], b[1], b[2], b[3]);
          mma(acc[2 * np], a2, b[0], b[1]);                  // small planes first: the larger the term, the later it is added
          mma(acc[2 * np], a1, b[0], b[1]);
          mma(acc[2 * np], a0, b[0], b[1]);
          mma(acc[2 * np + 1], a2, b[2], b[3]);
          mma(acc[2 * np + 1], a1, b[2], b[3]);
          mma(acc[2 * np + 1], a0, b[2], b[3]);
        }
      }
    }
    const int oa = r0 + wr + g, ob = oa + 8;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = wc + nt * 8 + 2 * t;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = h ? ob : oa;
        if (row >= R) continue;
        const float d0 = acc[nt][2 * h], d1 = acc[nt][2 * h + 1];
        const unsigned u = *reinterpret_cast<const unsigned*>(c + (size_t)row * C + col);
        const float x0 = bf_lo(u), x1 = bf_hi(u), s0 = sigmoid_f(x0), s1 = sigmoid_f(x1);
        *reinterpret_cast<uint32_t*>(dc + (size_t)row * C + col) = pack(d0 * s0 * (1.f + x0 * (1.f - s0)), d1 * s1 * (1.f + x1 * (1.f - s1)));
      }
    }
    return;
  }
  // ---------------------------------------------------------------- dW
  pdl_gate();
  float* raw = reinterpret_cast<float*>(smem);                                       // NRING x [KC][128] fp32
  bf16* ast = reinterpret_cast<bf16*>(smem + NRING * RAWB);                          // NRING x [KC][LDS]
  bf16* gp = reinterpret_cast<bf16*>(smem + NRING * RAWB + NRING * ABYT);            // 3 planes x [KC][LDS]
  const int job = blockIdx.x - n_dc, kcn = job % 6, rg = job / 6;
  const int s0 = rg * rpc, s1 = min(R, s0 + rpc);
  const int ntile = (s1 - s0 + KC - 1) / KC;
  const int wch = (warp & 3) * 32, wn = (warp >> 2) * 64;
  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  auto load = [&](int tl) {                                  // tile tl -> ring slot tl % NRING (zero-filled past the range)
    const int r = s0 + tl * KC, slot = tl % NRING;
    float* rw = raw + slot * KC * C;
    bf16* as = ast + slot * KC * LDS;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                            // 32 rows x 32 float4
      const int i = tid + j * 256, row = i >> 5, q = i & 31;
      const bool ok = r + row < s1;
      cp16(sa(rw + row * C + q * 4), gm + (size_t)(ok ? r + row : 0) * MW + kcn * 128 + q * 4, ok);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {                            // 32 rows x 16 vectors of silu(c)
      const int i = tid + j * 256, row = i >> 4, v = i & 15;
      const bool ok = r + row < s1;
      cp16(sa(as + row * LDS + v * 8), a_ + (size_t)(ok ? r + row : 0) * C + v * 8, ok);
    }
  };
  for (int tl = 0; tl < NRING - 1; ++tl) {
    if (tl < ntile) load(tl);
    cp_commit();
  }
  for (int tl = 0; tl < ntile; ++tl) {
    cp_wait<NRING - 2>();                                    // tile tl has landed (this thread's copies)
    __syncthreads();                                         // ... for every thread, and every warp is past the MMAs of tile tl - 1
    if (tl + NRING - 1 < ntile) load(tl + NRING - 1);
    cp_commit();
    const float* rw = raw + (tl % NRING) * KC * C;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                            // dmod -> three bf16 planes
      const int i = tid + j * 256, row = i >> 5, q = i & 31;
      const float4 v = *reinterpret_cast<const float4*>(rw + row * C + q * 4);
      uint32_t p00, p01, p02, p10, p11, p12;
      split3(v.x, v.y, p00, p01, p02);
      split3(v.z, v.w, p10, p11, p12);
      *reinterpret_cast<uint2*>(gp + row * LDS + q * 4) = make_uint2(p00, p10);
      *reinterpret_cast<uint2*>(gp + (KC + row) * LDS + q * 4) = make_uint2(p01, p11);
      *reinterpret_cast<uint2*>(gp + (2 * KC + row) * LDS + q * 4) = make_uint2(p02, p12);
    }
    __syncthreads();
    const bf16* ah = ast + (tl % NRING) * KC * LDS;
#pragma unroll
    for (int ks = 0; ks < KC / 16; ++ks) {
      uint32_t af[3][2][4];
#pragma unroll
      for (int pl = 0; pl < 3; ++pl)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int kk = ks * 16 + (mi >> 1) * 8 + (lane & 7), mm = wch + mt * 16 + (mi & 1) * 8;
          ldsm4t(sa(gp + (pl * KC + kk) * LDS + mm), af[pl][mt][0], af[pl][mt][1], af[pl][mt][2], af[pl][mt][3]);
        }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int kk = ks * 16 + (mi & 1) * 8 + (lane & 7), nn = wn + np * 16 + (mi >> 1) * 8;
        uint32_t b[4];
        ldsm4t(sa(ah + kk * LDS + nn), b[0], b[1], b[2], b[3]);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int pl = 2; pl >= 0; --pl) {                  // small planes first
            mma(acc[mt][2 * np], af[pl][mt], b[0], b[1]);
            mma(acc[mt][2 * np + 1], af[pl][mt], b[2], b[3]);
          }
      }
    }
  }
  float* o = ws + (size_t)rg * MW * C;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int ch = kcn * 128 + wch + mt * 16 + g, col = wn + nt * 8 + 2 * t;
      *reinterpret_cast<float2*>(o + (size_t)ch * C + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<float2*>(o + (size_t)(ch + 8) * C + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
}

// dW = sum over the row ranges in order (deterministic), bf16
__global__ void __launch_bounds__(256) mod_dw_reduce_kernel(const float* __restrict__ ws, bf16* __restrict__ dw, int nsplit) {
  pdl_gate();
  const int i = (blockIdx.x * 256 + threadIdx.x) * 4;       // 768 * 128 / 4 = 24576 vectors
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int p = 0; p < nsplit; ++p) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(ws + (size_t)p * MW * C + i));
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  *reinterpret_cast<uint2*>(dw + i) = make_uint2(pack(s.x, s.y), pack(s.z, s.w));
}

void check(const torch::Tensor& t, const char* name, int64_t cols, torch::ScalarType dt) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.size(1) == cols && t.scalar_type() == dt,
              name, " must be a contiguous ", c10::toString(dt), " CUDA tensor [R, ", cols, "]");
}
}  // namespace

// returns [mod [R, 768] fp32, silu(c) [R, 128] bf16 (an empty tensor unless ``save``)]; c [R, 128], Wmod [768, 128] bf16
std::vector<torch::Tensor> mod_fwd(torch::Tensor c, torch::Tensor w, bool save) {
  check(c, "c", C, torch::kBFloat16);
  check(w, "wmod", C, torch::kBFloat16);
  TORCH_CHECK(w.size(0) == MW, "wmod must be [768, 128]");
  const int R = static_cast<int>(c.size(0));
  auto out = torch::empty({R, MW}, c.options().dtype(torch::kFloat32));
  auto aout = save ? torch::empty({R, C}, c.options()) : torch::empty({0}, c.options());
  if (R == 0) return {out, aout};
  const size_t smem = 128 * LDS * sizeof(bf16);
  static const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int tiles = (R + 63) / 64;
  const dim3 grid(std::max(1, std::min(tiles, (sms * 3 + MW / 128 - 1) / (MW / 128))), MW / 128);   // ~3 CTAs per SM
  cudaFuncSetAttribute(mod_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  mod_fwd_kernel<<<grid, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(c.data_ptr()), reinterpret_cast<const bf16*>(w.data_ptr()), out.data_ptr<float>(),
      save ? reinterpret_cast<bf16*>(aout.data_ptr()) : nullptr, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, aout};
}

// returns [dc [R, 128], dWmod [768, 128]] (bf16); an output not asked for is an empty tensor. ``a`` is silu(c) from mod_fwd(save).
std::vector<torch::Tensor> mod_bwd(torch::Tensor g, torch::Tensor c, torch::Tensor a, torch::Tensor w, bool need_dc, bool need_dw) {
  check(g, "dmod", MW, torch::kFloat32);
  check(c, "c", C, torch::kBFloat16);
  check(a, "a", C, torch::kBFloat16);
  check(w, "wmod", C, torch::kBFloat16);
  const int R = static_cast<int>(c.size(0));
  TORCH_CHECK(g.size(0) == R && a.size(0) == R && w.size(0) == MW, "mod_bwd: shape mismatch");
  auto stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor dc = need_dc ? torch::empty({R, C}, c.options()) : torch::empty({0}, c.options());
  torch::Tensor dw = need_dw ? torch::empty({MW, C}, w.options()) : torch::empty({0}, w.options());
  if (R == 0) {
    if (need_dw) dw.zero_();
    return {dc, dw};
  }
  // dc: 32-row tiles; dW: 6 channel tiles x ~16 row ranges of a multiple of KC rows
  const int n_dc = need_dc ? (R + 31) / 32 : 0;
  const int rpc = std::max(KC, ((R + 15) / 16 + KC - 1) / KC * KC);
  const int nrg = need_dw ? (R + rpc - 1) / rpc : 0;
  torch::Tensor ws = torch::empty({nrg, MW, C}, c.options().dtype(torch::kFloat32));
  cudaFuncSetAttribute(mod_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BWD);
  pdl_launch(mod_bwd_kernel, n_dc + 6 * nrg, 256, SMEM_BWD, stream, true,
      g.data_ptr<float>(), reinterpret_cast<const bf16*>(c.data_ptr()), reinterpret_cast<const bf16*>(a.data_ptr()),
      reinterpret_cast<const bf16*>(w.data_ptr()), need_dc ? reinterpret_cast<bf16*>(dc.data_ptr()) : nullptr, ws.data_ptr<float>(), R,
      n_dc, rpc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (need_dw) {
    pdl_launch(mod_dw_reduce_kernel, MW * C / 4 / 256, 256, 0, stream, true, ws.data_ptr<float>(), reinterpret_cast<bf16*>(dw.data_ptr()), nrg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dc, dw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mod_fwd", &mod_fwd, "adaLN modulation silu(c) Wmod^T -> fp32 [R, 768] (and silu(c) bf16 when save)");
  m.def("mod_bwd", &mod_bwd, "adaLN modulation backward: dc and dWmod from dmod [R, 768] fp32 and the saved silu(c)");
}
