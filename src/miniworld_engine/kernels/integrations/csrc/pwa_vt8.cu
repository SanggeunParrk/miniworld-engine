// EXPERIMENTAL, opt-in (pwa_fp8.py): the PWA forward's value producer with v stored as e4m3, K-major for the e4m3 contraction (sm_90a).
//   y[s,n,:]            = bf16(LN(m[s,n,:]))                         [S][N][64]   (the same LayerNorm code / bytes as ln_vg)
//   vT[h][s*C + c][n]   = e4m3(vinv * (y[s,n,:] . Wv[h*C+c, :]))     [H][S*C][N]  (j = n contiguous: Hopper's e4m3 MMA wants K-major operands)
//   amax                = max |vinv * v| over the call (atomic, the saturation guard: > 448 means the cast clipped)
// One 128-thread block (one warpgroup) per item (s, 64 consecutive tokens), persistent over items with a NST-deep TMA load ring and Wv
// staged once.  vT tile = Wv (A, 64 hc rows per wgmma) x y^T (B = the y tile itself: [64 n][64 d] is K-major); the accumulator rows are hc,
// its columns n, so a thread's two consecutive-n values pack into one e4m3x2 halfword; the [8 h][32 c][64 n] tile (16 KiB) leaves by one TMA store.
// Also: w_quant (w [H,N,N] bf16 -> e4m3 of w * 256).
// Widths: VT_D (d_msa) / VT_C (channels per head), built per width: 64 / 32 (pwa_train's MiniWorld widths, the default build) and the
// pwa_wide widths 64 / 8 (AF3 family), 128 / 8, 128 / 16.  d_msa 128 runs as two 64-column (128-byte-swizzled) blocks per tile and Wv;
// HC = 8 C rows of the accumulator as HC / 64 m64 blocks.  The wide route keeps no y (store_y = false: its gate pass normalises m again),
// and then N need only be a multiple of 16: the last item's tokens past N are computed from the next row's (or zero-filled) rows and
// clipped by the vT store, and left out of amax.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include "sm90_ptx.cuh"

#ifndef VT_D
#define VT_D 64
#endif
#ifndef VT_C
#define VT_C 32
#endif

namespace {
constexpr int H = 8, C = VT_C, D = VT_D, HC = H * C;
constexpr int BN = 64;                                   // tokens per item
constexpr int TILE = BN * D;                             // elements of one x / y tile (8 KiB in bf16 at d_msa 64)
constexpr int KB = D / 64, MB = HC / 64;                 // 64-column blocks of d; m64 blocks of hc
constexpr int TPR = D / 32, RP = 128 / TPR, PASSES = BN / RP;   // LayerNorm: threads per row (32 columns each), rows per pass
static_assert(D % 64 == 0 && HC % 64 == 0 && HC <= 256 && C <= 256, "widths: d_msa a multiple of 64, 8 x C a multiple of 64 up to 256");

namespace wg {
using csrc_sm90::wg::commit;
using csrc_sm90::wg::fence;
using csrc_sm90::wg::proxy_fence;
using csrc_sm90::wg::wait;
__device__ __forceinline__ int off(int r, int kc) { return r * 64 + ((kc ^ (r & 7)) << 3); }        // 128B swizzle, 64-wide rows
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(64) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
// both K-major, m64 n64 k16
__device__ __forceinline__ void mma_m64n64k16_kk(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31},\n"
      " %33, %34, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
}  // namespace wg

namespace tma {
using csrc_sm90::tma::bar_init;
using csrc_sm90::tma::bar_init_fence;
using csrc_sm90::tma::commit;
using csrc_sm90::tma::expect_tx;
using csrc_sm90::tma::load_2d;
using csrc_sm90::tma::sa;
using csrc_sm90::tma::store_2d;
using csrc_sm90::tma::store_3d;
using csrc_sm90::tma::wait;
using csrc_sm90::tma::wait_all;
using csrc_sm90::tma::wait_read;
}  // namespace tma

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint16_t e4m3x2(float lo, float hi) {   // lo -> low byte (lower address)
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// smem: [NST][KB][BN][64] bf16 x/y stages | [KB][HC][64] bf16 Wv (128B swizzle) | [HC][BN] e4m3 vT staging | barriers
#ifndef VT_NAT
#define VT_NAT 0
#endif
// VT_NAT (the FP8 TRAINING route, pwa_fp8_train.py): also v as e4m3 in the natural head-major layout [H][N][S*C] (the dw GEMM's K-major
// operand), staged [H][BN][C] behind the vT staging and stored by a second TMA box
// VT_BF (round 2, backward-only FP8: pwa_fp8_train.py "dv,dw"): with VT_NAT, ALSO v as bf16 natural [H][N][S*C] (the forward contraction on
// cuBLAS stays bf16) and no vT (the forward is not FP8): staged [H][BN][C] bf16 behind the e4m3 natural tile, one more TMA box
#ifndef VT_BF
#define VT_BF 0
#endif
template <int NST> struct SMQ {
  static constexpr int SX = 0, SW = SX + NST * TILE * 2, SV = SW + HC * D * 2, SEND = SV + HC * BN * (1 + VT_NAT) + VT_BF * HC * BN * 2;
  static constexpr int BYTES = SEND + 1024 + 64;
};

// SY: store y (pwa_train's route: its gate pass reads y; then N % 64 == 0); without, N % 16 == 0 and the token tail is clipped
template <int NST, bool SY>
__global__ void __launch_bounds__(128) ln_vt8_kernel(const float* __restrict__ lnw, const float* __restrict__ lnb, float eps, int N, int nitem,
                                                     const float* __restrict__ vinv_p, float* __restrict__ amax_p,
                                                     const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap ymap,
                                                     const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap vmap
#if VT_NAT
                                                     , const __grid_constant__ CUtensorMap nmap
#endif
                                                     ) {   // (VT_BF: vmap is the bf16 natural v's map, there is no vT)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  unsigned char* sm = smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  using L = SMQ<NST>;
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(sm + L::SX);
  __nv_bfloat16* sWv = reinterpret_cast<__nv_bfloat16*>(sm + L::SW);
  uint16_t* sV = reinterpret_cast<uint16_t*>(sm + L::SV);       // [HC][BN] bytes, as halfwords
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + L::SEND);   // [NST] full + [1] weight
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int nblk = (N + BN - 1) / BN;
  const float vinv = *vinv_p;

  auto item_of = [&](int it) { return (int)blockIdx.x + it * (int)gridDim.x; };
  auto issue = [&](int it) {
    const int t = item_of(it);
    if (t >= nitem) return;
    const int st = it % NST;
    const int n0 = (t % nblk) * BN, s = t / nblk;
    tma::expect_tx(bars + st, TILE * 2);
    #pragma unroll
    for (int kb = 0; kb < KB; ++kb) tma::load_2d(&xmap, tma::sa(sX + st * TILE + kb * BN * 64), bars + st, kb * 64, s * N + n0);
  };
  if (tid == 0) {
    for (int i = 0; i <= NST; ++i) tma::bar_init(bars + i, 1);
    tma::bar_init_fence();
    wg::proxy_fence();
    tma::expect_tx(bars + NST, HC * D * 2);
    #pragma unroll
    for (int kb = 0; kb < KB; ++kb) tma::load_2d(&wmap, tma::sa(sWv + kb * HC * 64), bars + NST, kb * 64, 0);
    for (int i = 0; i < NST - 1; ++i) issue(i);
  }
  __syncthreads();
  const int tq = tid % TPR;                                     // this thread's 32 columns: tq * 32 ..
  float g[32], b[32];
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int c0 = tq * 32 + q * 8;
    const float4 g0 = *reinterpret_cast<const float4*>(lnw + c0), g1 = *reinterpret_cast<const float4*>(lnw + c0 + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(lnb + c0), b1 = *reinterpret_cast<const float4*>(lnb + c0 + 4);
    g[q * 8 + 0] = g0.x; g[q * 8 + 1] = g0.y; g[q * 8 + 2] = g0.z; g[q * 8 + 3] = g0.w; g[q * 8 + 4] = g1.x; g[q * 8 + 5] = g1.y; g[q * 8 + 6] = g1.z; g[q * 8 + 7] = g1.w;
    b[q * 8 + 0] = b0.x; b[q * 8 + 1] = b0.y; b[q * 8 + 2] = b0.z; b[q * 8 + 3] = b0.w; b[q * 8 + 4] = b1.x; b[q * 8 + 5] = b1.y; b[q * 8 + 6] = b1.z; b[q * 8 + 7] = b1.w;
  }
  tma::wait(bars + NST, 0);
  float amax = 0.f;
  float acc[MB][32];
  for (int it = 0;; ++it) {
    const int t = item_of(it);
    if (t >= nitem) break;
    const int st = it % NST;
    const int n0 = (t % nblk) * BN, s = t / nblk;
    // SY: groups of thread 0, in order: y(it-1), v(it-1). <=1 pending -> y(it-1) has read its stage: refill it.  Without y the stage
    // of item it-1 was last read by its wgmma (waited on before the barrier that precedes this point)
    if (tid == 0) { if (SY && it >= 1) tma::wait_read<1>(); issue(it + NST - 1); }
    tma::wait(bars + st, (it / NST) & 1);
    __nv_bfloat16* sx = sX + st * TILE;
    // ---- LayerNorm in place (ln_vg's code: bf16 in, fp32 stats, bf16 out); TPR threads per row, PASSES passes ----
    #pragma unroll
    for (int pass = 0; pass < PASSES; ++pass) {
      const int r = pass * RP + tid / TPR;
      float x[32];
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ch = tq * 4 + q;
        const uint4 u = *reinterpret_cast<const uint4*>(sx + (ch >> 3) * BN * 64 + wg::off(r, ch & 7));
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&u);
        #pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(p2[k]); x[q * 8 + k * 2] = f.x; x[q * 8 + k * 2 + 1] = f.y; }
      }
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < 32; ++k) sum += x[k];
      #pragma unroll
      for (int o = 1; o < TPR; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      const float mean = sum * (1.f / D);
      float ss = 0.f;
      #pragma unroll
      for (int k = 0; k < 32; ++k) { x[k] -= mean; ss += x[k] * x[k]; }
      #pragma unroll
      for (int o = 1; o < TPR; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rstd = rsqrtf(ss * (1.f / D) + eps);
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ch = tq * 4 + q;
        uint4 u;
        u.x = pack2(g[q * 8 + 0] * (rstd * x[q * 8 + 0]) + b[q * 8 + 0], g[q * 8 + 1] * (rstd * x[q * 8 + 1]) + b[q * 8 + 1]);
        u.y = pack2(g[q * 8 + 2] * (rstd * x[q * 8 + 2]) + b[q * 8 + 2], g[q * 8 + 3] * (rstd * x[q * 8 + 3]) + b[q * 8 + 3]);
        u.z = pack2(g[q * 8 + 4] * (rstd * x[q * 8 + 4]) + b[q * 8 + 4], g[q * 8 + 5] * (rstd * x[q * 8 + 5]) + b[q * 8 + 5]);
        u.w = pack2(g[q * 8 + 6] * (rstd * x[q * 8 + 6]) + b[q * 8 + 6], g[q * 8 + 7] * (rstd * x[q * 8 + 7]) + b[q * 8 + 7]);
        *reinterpret_cast<uint4*>(sx + (ch >> 3) * BN * 64 + wg::off(r, ch & 7)) = u;
      }
    }
    wg::proxy_fence();
    __syncthreads();
    if (SY && tid == 0) {
      #pragma unroll
      for (int kb = 0; kb < KB; ++kb) tma::store_2d(&ymap, tma::sa(sx + kb * BN * 64), kb * 64, s * N + n0);
      tma::commit();
    }
    // ---- vT = Wv . y^T : [HC hc][64 n], MB m64 blocks, D / 16 k-steps over KB 64-column blocks ----
    wg::fence();
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
      #pragma unroll
      for (int ks = 0; ks < D / 16; ++ks)
        wg::mma_m64n64k16_kk(wg::desc(sWv + (ks >> 2) * HC * 64 + mb * 64 * 64 + wg::off(0, (ks & 3) * 2)),
                             wg::desc(sx + (ks >> 2) * BN * 64 + wg::off(0, (ks & 3) * 2)), acc[mb], ks == 0 ? 0 : 1);
    }
    wg::commit();
    wg::wait();
    // the previous item's v store must have read the staging (pending now: v(it-1) [, y(it)])
    if (tid == 0) { if (SY) tma::wait_read<1>(); else tma::wait_read<0>(); }
    __syncthreads();
    const int rr = warp * 16 + (lane >> 2), cc = (lane & 3) * 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float a0 = acc[mb][i * 4 + 0] * vinv, a1 = acc[mb][i * 4 + 1] * vinv, a2 = acc[mb][i * 4 + 2] * vinv, a3 = acc[mb][i * 4 + 3] * vinv;
        const int row = mb * 64 + rr, col = i * 8 + cc;
        if (SY || n0 + col < N) amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3))));
#if !VT_BF
        sV[(row * BN + col) >> 1] = e4m3x2(a0, a1);
        sV[((row + 8) * BN + col) >> 1] = e4m3x2(a2, a3);
#endif
#if VT_BF
        {   // bf16 natural v (unscaled accumulator, rounded like ln_vg's): [H][BN][C] behind the e4m3 tile
          __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(sV) + HC * BN * 2);
          const int h0 = row / C, c0 = row % C, h1 = (row + 8) / C, c1 = (row + 8) % C;
          sB[(h0 * BN + col) * C + c0] = __float2bfloat16_rn(acc[mb][i * 4 + 0]);
          sB[(h0 * BN + col + 1) * C + c0] = __float2bfloat16_rn(acc[mb][i * 4 + 1]);
          sB[(h1 * BN + col) * C + c1] = __float2bfloat16_rn(acc[mb][i * 4 + 2]);
          sB[(h1 * BN + col + 1) * C + c1] = __float2bfloat16_rn(acc[mb][i * 4 + 3]);
        }
#endif
#if VT_NAT
        {
          unsigned char* sN = reinterpret_cast<unsigned char*>(sV) + HC * BN;   // [H][BN][C]
          const uint16_t p0 = e4m3x2(a0, a1), p1 = e4m3x2(a2, a3);
          const int h0 = row / C, c0 = row % C, h1 = (row + 8) / C, c1 = (row + 8) % C;
          sN[(h0 * BN + col) * C + c0] = (unsigned char)(p0 & 0xff);
          sN[(h0 * BN + col + 1) * C + c0] = (unsigned char)(p0 >> 8);
          sN[(h1 * BN + col) * C + c1] = (unsigned char)(p1 & 0xff);
          sN[(h1 * BN + col + 1) * C + c1] = (unsigned char)(p1 >> 8);
        }
#endif
      }
    }
    wg::proxy_fence();
    __syncthreads();
#if VT_NAT
    if (tid == 0) {
#if !VT_BF
      tma::store_3d(&vmap, tma::sa(sV), n0, s * C, 0);
#else
      tma::store_3d(&vmap, tma::sa(reinterpret_cast<unsigned char*>(sV) + HC * BN * 2), s * C, n0, 0);
#endif
      tma::store_3d(&nmap, tma::sa(reinterpret_cast<unsigned char*>(sV) + HC * BN), s * C, n0, 0);
      tma::commit();
    }
#else
    if (tid == 0) { tma::store_3d(&vmap, tma::sa(sV), n0, s * C, 0); tma::commit(); }
#endif
  }
  if (tid == 0) tma::wait_all();
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (lane == 0) atomicMax(reinterpret_cast<int*>(amax_p), __float_as_int(amax));   // non-negative floats order as ints
}

// w [H*N*N] bf16 (probabilities) -> e4m3 of w * 256 (exact scaling: a power of two), 8 per thread
__global__ void w_quant_kernel(const uint4* __restrict__ w, uint2* __restrict__ q, long n8) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const uint4 u = w[i];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
    uint16_t o[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(p[k]); o[k] = e4m3x2(f.x * 256.f, f.y * 256.f); }
    uint2 r;
    r.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
    r.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
    q[i] = r;
  }
}

PFN_cuTensorMapEncodeTiled_v12000 tma_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
#endif
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
CUtensorMap enc(CUtensorMapDataType dt, int rank, void* base, const uint64_t* gdim, const uint64_t* gstride, const uint32_t* bdim, CUtensorMapSwizzle sw,
                const char* what) {
  alignas(64) CUtensorMap m{};
  uint32_t estride[3] = {1, 1, 1};
  CUresult r = tma_encode()(&m, dt, rank, base, gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", what, ") failed: ", (int)r);
  return m;
}

#if !VT_NAT
template <int NST, bool SY>
void launch(const torch::Tensor& lnw, const torch::Tensor& lnb, double eps, int N, int nitem, int64_t blocks_per_sm, const torch::Tensor& vinv,
            const torch::Tensor& amax, const CUtensorMap& xmap, const CUtensorMap& ymap, const CUtensorMap& wmap, const CUtensorMap& vmap) {
  constexpr int BYTES = SMQ<NST>::BYTES;
  static bool attr = false;
  static int sms = 0;
  if (!attr) {
    cudaFuncSetAttribute(ln_vt8_kernel<NST, SY>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, lnw.device().index());
    attr = true;
  }
  const int grid = (int)std::min<long>(nitem, (long)sms * blocks_per_sm);
  ln_vt8_kernel<NST, SY><<<grid, 128, BYTES, at::cuda::getCurrentCUDAStream()>>>(lnw.data_ptr<float>(), lnb.data_ptr<float>(), (float)eps, N, nitem,
                                                                                  vinv.data_ptr<float>(), amax.data_ptr<float>(), xmap, ymap, wmap, vmap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool SY>
void launch_nst(int64_t nst, const torch::Tensor& lnw, const torch::Tensor& lnb, double eps, int N, int nitem, int64_t blocks_per_sm,
                const torch::Tensor& vinv, const torch::Tensor& amax, const CUtensorMap& xmap, const CUtensorMap& ymap, const CUtensorMap& wmap,
                const CUtensorMap& vmap) {
  switch (nst) {
    case 2: launch<2, SY>(lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap); break;
    case 3: launch<3, SY>(lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap); break;
    case 4: launch<4, SY>(lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap); break;
    default: TORCH_CHECK(false, "nst must be 2, 3 or 4");
  }
}
#endif
}  // namespace

#if !VT_NAT

// returns (vT e4m3 [H, S*C, N], y [S, N, D] bf16, or an empty tensor when store_y is false); amax (fp32 [1], NOT zeroed here) takes
// max |vinv * v| of this call.  store_y: N a multiple of 64; without: of 16
std::vector<torch::Tensor> ln_vt8(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, torch::Tensor vinv, torch::Tensor amax,
                                  int64_t blocks_per_sm, int64_t nst, bool store_y) {
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == D, "m: [S, N, d_msa] bf16 contiguous");
  TORCH_CHECK(wv.scalar_type() == torch::kBFloat16 && wv.is_contiguous() && wv.dim() == 2 && wv.size(0) == HC && wv.size(1) == D, "wv: [8 C, d_msa] bf16");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat && lnw.is_contiguous() && lnw.numel() == D && lnb.scalar_type() == torch::kFloat && lnb.is_contiguous() &&
                  lnb.numel() == D, "LN params: fp32[d_msa]");
  TORCH_CHECK(vinv.scalar_type() == torch::kFloat && vinv.numel() == 1 && amax.scalar_type() == torch::kFloat && amax.numel() == 1, "vinv / amax: fp32[1]");
  const int S = (int)m.size(0), N = (int)m.size(1);
  TORCH_CHECK(store_y ? N % BN == 0 : N % 16 == 0, store_y ? "N must be a multiple of 64" : "N must be a multiple of 16");
  auto y = store_y ? torch::empty_like(m) : torch::empty({0}, m.options());
  auto vt = torch::empty({H, (long)S * C, N}, m.options().dtype(torch::kFloat8_e4m3fn));
  uint64_t gx[2] = {(uint64_t)D, (uint64_t)S * N}, sx[1] = {(uint64_t)D * 2};
  uint32_t bx[2] = {64, BN};
  CUtensorMap xmap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, m.data_ptr(), gx, sx, bx, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap ymap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, store_y ? y.data_ptr() : m.data_ptr(), gx, sx, bx, CU_TENSOR_MAP_SWIZZLE_128B, "y");
  uint64_t gw[2] = {(uint64_t)D, (uint64_t)HC}, sw[1] = {(uint64_t)D * 2};
  uint32_t bw[2] = {64, HC};
  CUtensorMap wmap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wv.data_ptr(), gw, sw, bw, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  uint64_t gv[3] = {(uint64_t)N, (uint64_t)S * C, (uint64_t)H}, sv[2] = {(uint64_t)N, (uint64_t)S * C * N};
  uint32_t bv[3] = {BN, C, H};
  CUtensorMap vmap = enc(CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, vt.data_ptr(), gv, sv, bv, CU_TENSOR_MAP_SWIZZLE_NONE, "vT");
  const int nitem = ((N + BN - 1) / BN) * S;
  if (store_y)
    launch_nst<true>(nst, lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap);
  else
    launch_nst<false>(nst, lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap);
  return {vt, y};
}

torch::Tensor w_quant(torch::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous() && w.numel() % 8 == 0, "w: bf16 contiguous, numel % 8");
  auto q = torch::empty(w.sizes(), w.options().dtype(torch::kFloat8_e4m3fn));
  const long n8 = w.numel() / 8;
  const int grid = (int)std::min<long>((n8 + 255) / 256, 132L * 8);
  w_quant_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<uint2*>(q.data_ptr()), n8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("ln_vt8", &ln_vt8, "LayerNorm + value projection, v as e4m3 K-major: (vT [H, S*C, N] e4m3, y [S, N, d_msa] bf16 or empty)", py::arg("m"), py::arg("lnw"),
          py::arg("lnb"), py::arg("wv"), py::arg("eps"), py::arg("vinv"), py::arg("amax"), py::arg("blocks_per_sm") = 2, py::arg("nst") = 3,
          py::arg("store_y") = true);
  mod.def("w_quant", &w_quant, "w [H, N, N] bf16 -> e4m3 of 256 w");
}
#else   // VT_NAT: the FP8 training route's producer (pwa_fp8_train.py)
namespace {
template <int NST>
void launch_nat(const torch::Tensor& lnw, const torch::Tensor& lnb, double eps, int N, int nitem, int64_t blocks_per_sm, const torch::Tensor& vinv,
                const torch::Tensor& amax, const CUtensorMap& xmap, const CUtensorMap& ymap, const CUtensorMap& wmap, const CUtensorMap& vmap,
                const CUtensorMap& nmap) {
  constexpr int BYTES = SMQ<NST>::BYTES;
  static bool attr = false;
  static int sms = 0;
  if (!attr) {
    cudaFuncSetAttribute(ln_vt8_kernel<NST, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, lnw.device().index());
    attr = true;
  }
  const int grid = (int)std::min<long>(nitem, (long)sms * blocks_per_sm);
  ln_vt8_kernel<NST, true><<<grid, 128, BYTES, at::cuda::getCurrentCUDAStream()>>>(lnw.data_ptr<float>(), lnb.data_ptr<float>(), (float)eps, N, nitem,
                                                                                    vinv.data_ptr<float>(), amax.data_ptr<float>(), xmap, ymap, wmap, vmap, nmap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// returns (vT e4m3 [H, S*C, N], v e4m3 natural [H, N, S*C], y [S, N, D] bf16); amax (fp32 [1], NOT zeroed) takes max |vinv * v|.  N % 64 == 0.
std::vector<torch::Tensor> ln_vt8n(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, torch::Tensor vinv, torch::Tensor amax,
                                   int64_t blocks_per_sm, int64_t nst) {
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == D, "m: [S, N, d_msa] bf16 contiguous");
  TORCH_CHECK(wv.scalar_type() == torch::kBFloat16 && wv.is_contiguous() && wv.dim() == 2 && wv.size(0) == HC && wv.size(1) == D, "wv: [8 C, d_msa] bf16");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat && lnw.is_contiguous() && lnw.numel() == D && lnb.scalar_type() == torch::kFloat && lnb.is_contiguous() &&
                  lnb.numel() == D, "LN params: fp32[d_msa]");
  TORCH_CHECK(vinv.scalar_type() == torch::kFloat && vinv.numel() == 1 && amax.scalar_type() == torch::kFloat && amax.numel() == 1, "vinv / amax: fp32[1]");
  const int S = (int)m.size(0), N = (int)m.size(1);
  TORCH_CHECK(N % BN == 0, "N must be a multiple of 64");
  auto y = torch::empty_like(m);
#if VT_BF   // round 2: slot "vT" = v bf16 natural [H, N, S*C]
  auto vt = torch::empty({H, N, (long)S * C}, m.options());
#else
  auto vt = torch::empty({H, (long)S * C, N}, m.options().dtype(torch::kFloat8_e4m3fn));
#endif
  auto vn = torch::empty({H, N, (long)S * C}, m.options().dtype(torch::kFloat8_e4m3fn));
  uint64_t gx[2] = {(uint64_t)D, (uint64_t)S * N}, sx[1] = {(uint64_t)D * 2};
  uint32_t bx[2] = {64, BN};
  CUtensorMap xmap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, m.data_ptr(), gx, sx, bx, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap ymap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, y.data_ptr(), gx, sx, bx, CU_TENSOR_MAP_SWIZZLE_128B, "y");
  uint64_t gw[2] = {(uint64_t)D, (uint64_t)HC}, sw[1] = {(uint64_t)D * 2};
  uint32_t bw[2] = {64, HC};
  CUtensorMap wmap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wv.data_ptr(), gw, sw, bw, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  uint64_t gv[3] = {(uint64_t)N, (uint64_t)S * C, (uint64_t)H}, sv[2] = {(uint64_t)N, (uint64_t)S * C * N};
  uint32_t bv[3] = {BN, C, H};
  uint64_t gn[3] = {(uint64_t)S * C, (uint64_t)N, (uint64_t)H}, sn[2] = {(uint64_t)S * C, (uint64_t)N * S * C};
  uint32_t bn[3] = {C, BN, H};
#if VT_BF
  uint64_t snb[2] = {(uint64_t)S * C * 2, (uint64_t)N * S * C * 2};
  CUtensorMap vmap = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, vt.data_ptr(), gn, snb, bn, CU_TENSOR_MAP_SWIZZLE_NONE, "v bf16 natural");
#else
  CUtensorMap vmap = enc(CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, vt.data_ptr(), gv, sv, bv, CU_TENSOR_MAP_SWIZZLE_NONE, "vT");
#endif
  CUtensorMap nmap = enc(CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, vn.data_ptr(), gn, sn, bn, CU_TENSOR_MAP_SWIZZLE_NONE, "v natural");
  const int nitem = (N / BN) * S;
  switch (nst) {
    case 2: launch_nat<2>(lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap, nmap); break;
    case 3: launch_nat<3>(lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap, nmap); break;
    case 4: launch_nat<4>(lnw, lnb, eps, N, nitem, blocks_per_sm, vinv, amax, xmap, ymap, wmap, vmap, nmap); break;
    default: TORCH_CHECK(false, "nst must be 2, 3 or 4");
  }
  return {vt, vn, y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("ln_vt8n", &ln_vt8n, "LayerNorm + value projection, v as e4m3 twice: (vT [H, S*C, N], v natural [H, N, S*C], y [S, N, d_msa] bf16)", py::arg("m"),
          py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps"), py::arg("vinv"), py::arg("amax"), py::arg("blocks_per_sm") = 2, py::arg("nst") = 3);
}
#endif
