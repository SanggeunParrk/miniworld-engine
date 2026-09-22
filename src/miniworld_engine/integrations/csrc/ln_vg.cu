// Fused LayerNorm + value projection for the PWA forward (sm_90a):
//   y[s,n,:] = bf16(LN(m[s,n,:]))                      [S][N][64]  (natural)
//   v[h][n][s*C + c] = bf16(y[s,n,:] . Wv[h*C+c, :])    head-major, the layout the contraction kernel reads
// One 128-thread block owns a tile of 64 consecutive s for ONE token n (token order): the eight per-head
// output pieces of the tile are then 64 x 64 B = 4 KiB contiguous each, and the x rows / y rows are 128 B
// lines 49 KiB apart -- every global access is a full line, all through TMA.  Persistent grid: a block
// walks tiles blockIdx.x, +gridDim.x, ... with a 3-stage TMA-load pipeline (2 ahead), the weight staged once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace {
constexpr int H = 8, C = 32, D = 64, HC = H * C;
constexpr int BSR = 64;                                  // s rows per tile
constexpr int TILE = BSR * D;                            // 8 KiB in bf16

namespace wg {
__device__ __forceinline__ int off(int r, int kc) { return r * 64 + ((kc ^ (r & 7)) << 3); }        // 128B swizzle, 64-wide rows
__device__ __forceinline__ int off64(int r, int kc) { return r * 32 + ((kc ^ ((r >> 1) & 3)) << 3); } // 64B swizzle, 32-wide rows
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(64) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
// both K-major, n = 128: half of the value projection
__device__ __forceinline__ void mma_m64n128k16_kk(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %64, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31,\n"
      " %32, %33, %34, %35, %36, %37, %38, %39,\n"
      " %40, %41, %42, %43, %44, %45, %46, %47,\n"
      " %48, %49, %50, %51, %52, %53, %54, %55,\n"
      " %56, %57, %58, %59, %60, %61, %62, %63},\n"
      " %65, %66, p, 1, 1, 0, 0;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31]),
        "+f"(d[32]),
        "+f"(d[33]),
        "+f"(d[34]),
        "+f"(d[35]),
        "+f"(d[36]),
        "+f"(d[37]),
        "+f"(d[38]),
        "+f"(d[39]),
        "+f"(d[40]),
        "+f"(d[41]),
        "+f"(d[42]),
        "+f"(d[43]),
        "+f"(d[44]),
        "+f"(d[45]),
        "+f"(d[46]),
        "+f"(d[47]),
        "+f"(d[48]),
        "+f"(d[49]),
        "+f"(d[50]),
        "+f"(d[51]),
        "+f"(d[52]),
        "+f"(d[53]),
        "+f"(d[54]),
        "+f"(d[55]),
        "+f"(d[56]),
        "+f"(d[57]),
        "+f"(d[58]),
        "+f"(d[59]),
        "+f"(d[60]),
        "+f"(d[61]),
        "+f"(d[62]),
        "+f"(d[63])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
}  // namespace wg

namespace tma {
__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra WAIT_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_3d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_3d(const void* map, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
}  // namespace tma

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}

// smem (bf16 elements from a 1 KiB-aligned base): [NST][TILE] x/y stages, [HC][64] Wv (K-major, 128B swizzle), [H][64][32] v staging (64B swizzle)
template <int NST, int NTOK> struct SMX { static constexpr int SX = 0, SW = SX + NST * NTOK * TILE, SV = SW + HC * D, SEND = SV + NTOK * H * BSR * C;
  static constexpr int BYTES = SEND * 2 + 1024 /* alignment slack */ + 64 /* barriers */; };

// NTOK tokens per tile, one warpgroup each: the x / y lines of a tile are NTOK * 128 B contiguous
template <int NST, int NTOK>
__global__ void __launch_bounds__(128 * NTOK) ln_vg_kernel(const float* __restrict__ lnw, const float* __restrict__ lnb, float eps, int N, int S, int ntile,
                                                        const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap ymap,
                                                        const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap vmap) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  unsigned char* sm = smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  using L = SMX<NST, NTOK>;
  constexpr int STAGE = NTOK * TILE;
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(sm) + L::SX;
  __nv_bfloat16* sWv = reinterpret_cast<__nv_bfloat16*>(sm) + L::SW;
  __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(sm) + L::SV;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + L::SEND * 2);       // [NST] full barriers + [1] weight barrier
  const int tid = threadIdx.x;
  const int wgi = tid >> 7, wtid = tid & 127;                       // warpgroup = token within the tile
  const int lane = wtid & 31, warp = wtid >> 5;
  const int nblk = N / NTOK;

  auto tile_of = [&](int it) { return (int)blockIdx.x + it * (int)gridDim.x; };
  auto issue = [&](int it) {   // thread 0: TMA-load tile `it`'s x block into stage it % NST
    const int t = tile_of(it);
    if (t >= ntile) return;
    const int st = it % NST;
    const int n = (t % nblk) * NTOK, s0 = (t / nblk) * BSR;
    tma::expect_tx(bars + st, STAGE * 2);
    tma::load_3d(&xmap, tma::sa(sX + st * STAGE), bars + st, 0, s0, n);
  };
  if (tid == 0) {
    for (int i = 0; i <= NST; ++i) tma::bar_init(bars + i, 1);
    tma::bar_init_fence();
    wg::proxy_fence();
    tma::expect_tx(bars + NST, HC * D * 2);
    tma::load_2d(&wmap, tma::sa(sWv), bars + NST, 0, 0);
    for (int i = 0; i < NST - 1; ++i) issue(i);
  }
  __syncthreads();
  // each thread owns half a row: 4 chunks of 8 columns
  float g[32], b[32];
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int c0 = (wtid & 1) * 32 + q * 8;
    const float4 g0 = *reinterpret_cast<const float4*>(lnw + c0), g1 = *reinterpret_cast<const float4*>(lnw + c0 + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(lnb + c0), b1 = *reinterpret_cast<const float4*>(lnb + c0 + 4);
    g[q * 8 + 0] = g0.x; g[q * 8 + 1] = g0.y; g[q * 8 + 2] = g0.z; g[q * 8 + 3] = g0.w; g[q * 8 + 4] = g1.x; g[q * 8 + 5] = g1.y; g[q * 8 + 6] = g1.z; g[q * 8 + 7] = g1.w;
    b[q * 8 + 0] = b0.x; b[q * 8 + 1] = b0.y; b[q * 8 + 2] = b0.z; b[q * 8 + 3] = b0.w; b[q * 8 + 4] = b1.x; b[q * 8 + 5] = b1.y; b[q * 8 + 6] = b1.z; b[q * 8 + 7] = b1.w;
  }
  tma::wait(bars + NST, 0);

  float acc[2][64];
  for (int it = 0;; ++it) {
    const int t = tile_of(it);
    if (t >= ntile) break;
    const int st = it % NST;
    const int n = (t % nblk) * NTOK, s0 = (t / nblk) * BSR;
    // refill the stage tile it-1 used: its y store must have finished reading (only the newer v store may still be in flight)
    if (tid == 0) { if (it >= 1) tma::wait_read<1>(); issue(it + NST - 1); }
    tma::wait(bars + st, (it / NST) & 1);
    __nv_bfloat16* sx = sX + st * STAGE + wgi * TILE;
    // ---- LayerNorm in place (bf16 in, fp32 stats, bf16 out) ----
    const int r = wtid >> 1, hf = wtid & 1;
    float x[32];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 u = *reinterpret_cast<const uint4*>(sx + wg::off(r, hf * 4 + q));
      const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&u);
      #pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(p2[k]); x[q * 8 + k * 2] = f.x; x[q * 8 + k * 2 + 1] = f.y; }
    }
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) sum += x[k];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    const float mean = sum * (1.f / D);
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) { x[k] -= mean; ss += x[k] * x[k]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    const float rstd = rsqrtf(ss * (1.f / D) + eps);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint4 u;
      u.x = pack2(g[q * 8 + 0] * (rstd * x[q * 8 + 0]) + b[q * 8 + 0], g[q * 8 + 1] * (rstd * x[q * 8 + 1]) + b[q * 8 + 1]);
      u.y = pack2(g[q * 8 + 2] * (rstd * x[q * 8 + 2]) + b[q * 8 + 2], g[q * 8 + 3] * (rstd * x[q * 8 + 3]) + b[q * 8 + 3]);
      u.z = pack2(g[q * 8 + 4] * (rstd * x[q * 8 + 4]) + b[q * 8 + 4], g[q * 8 + 5] * (rstd * x[q * 8 + 5]) + b[q * 8 + 5]);
      u.w = pack2(g[q * 8 + 6] * (rstd * x[q * 8 + 6]) + b[q * 8 + 6], g[q * 8 + 7] * (rstd * x[q * 8 + 7]) + b[q * 8 + 7]);
      *reinterpret_cast<uint4*>(sx + wg::off(r, hf * 4 + q)) = u;
    }
    wg::proxy_fence();
    __syncthreads();
    if (tid == 0) { tma::store_3d(&ymap, tma::sa(sX + st * STAGE), 0, s0, n); tma::commit(); }
    // ---- v = y . Wv^T : [64 s][256] = [64][64] x [256][64]^T ----
    wg::fence();
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint64_t da = wg::desc(sx + wg::off(0, ks * 2));
      wg::mma_m64n128k16_kk(da, wg::desc(sWv + wg::off(0, ks * 2)), acc[0], ks == 0 ? 0 : 1);
      wg::mma_m64n128k16_kk(da, wg::desc(sWv + 128 * 64 + wg::off(0, ks * 2)), acc[1], ks == 0 ? 0 : 1);
    }
    wg::commit();
    wg::wait();
    // ---- pack to the eight [64 s][32 c] head tiles (the previous tile's v stores must have read the staging) ----
    // bulk groups are per thread: warpgroup 0's leader also owns the y store (the most recent group), the others only v stores
    if (wtid == 0) { if (wgi == 0) tma::wait_read<1>(); else tma::wait_read<0>(); }
    __syncthreads();
    __nv_bfloat16* sVt = sV + wgi * (H * BSR * C);
    const int r0 = warp * 16 + (lane >> 2), cp = (lane & 3) * 2;
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int col = nb * 128 + i * 8 + cp, h = col >> 5, c = col & 31;
        __nv_bfloat16* sv = sVt + h * (BSR * C);
        *reinterpret_cast<uint32_t*>(sv + wg::off64(r0, c >> 3) + (c & 7)) = pack2(acc[nb][i * 4 + 0], acc[nb][i * 4 + 1]);
        *reinterpret_cast<uint32_t*>(sv + wg::off64(r0 + 8, c >> 3) + (c & 7)) = pack2(acc[nb][i * 4 + 2], acc[nb][i * 4 + 3]);
      }
    }
    wg::proxy_fence();
    __syncthreads();
    if (wtid == 0) {
      #pragma unroll
      for (int h = 0; h < H; ++h) tma::store_3d(&vmap, tma::sa(sVt + h * (BSR * C)), 0, s0, h * N + n + wgi);
      tma::commit();
    }
  }
  if (wtid == 0) tma::wait_all();
}

PFN_cuTensorMapEncodeTiled tma_encode() {
  static PFN_cuTensorMapEncodeTiled fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(p);
  }
  return fn;
}
CUtensorMap enc(int rank, void* base, const uint64_t* gdim, const uint64_t* gstride, const uint32_t* bdim, CUtensorMapSwizzle sw, const char* what) {
  alignas(64) CUtensorMap m{};
  uint32_t estride[3] = {1, 1, 1};
  CUresult r = tma_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, base, gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", what, ") failed: ", (int)r);
  return m;
}
}  // namespace

template <int NST, int NTOK>
void launch_nst(const torch::Tensor& lnw, const torch::Tensor& lnb, double eps, int N, int S, int ntile, int64_t blocks_per_sm,
                const CUtensorMap& xmap, const CUtensorMap& ymap, const CUtensorMap& wmap, const CUtensorMap& vmap) {
  constexpr int BYTES = SMX<NST, NTOK>::BYTES;
  constexpr int THREADS = 128 * NTOK;
  static bool attr = false;
  static int sms = 0;
  if (!attr) {
    cudaFuncSetAttribute(ln_vg_kernel<NST, NTOK>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ln_vg_kernel<NST, NTOK>, THREADS, BYTES);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, lnw.device().index());
    TORCH_WARN("ln_vg_kernel<", NST, ",", NTOK, ">: ", BYTES, " B smem -> ", nb, " blocks/SM (", sms, " SMs)");
    attr = true;
  }
  const int grid = (int)std::min<long>(ntile, (long)sms * blocks_per_sm);
  ln_vg_kernel<NST, NTOK><<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(lnw.data_ptr<float>(), lnb.data_ptr<float>(), (float)eps, N, S, ntile, xmap, ymap, wmap, vmap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> ln_vg(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, int64_t blocks_per_sm, int64_t nst, int64_t ntok) {
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == D, "m: [S, N, 64] bf16 contiguous");
  TORCH_CHECK(wv.scalar_type() == torch::kBFloat16 && wv.is_contiguous() && wv.dim() == 2 && wv.size(0) == HC && wv.size(1) == D, "wv: [256, 64] bf16");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat && lnw.is_contiguous() && lnw.numel() == D && lnb.scalar_type() == torch::kFloat && lnb.is_contiguous() && lnb.numel() == D, "LN params: fp32[64]");
  const int S = (int)m.size(0), N = (int)m.size(1);
  auto y = torch::empty_like(m);
  auto v = torch::empty({H, N, (long)S * C}, m.options());
  TORCH_CHECK(N % ntok == 0 && (ntok == 1 || ntok == 2), "N must be a multiple of ntok (1 or 2)");
  uint64_t gx[3] = {(uint64_t)D, (uint64_t)S, (uint64_t)N}, sx[2] = {(uint64_t)N * D * 2, (uint64_t)D * 2};   // (d, s, n) view: strides need not be monotonic
  uint32_t bx[3] = {D, BSR, (uint32_t)ntok};
  CUtensorMap xmap = enc(3, m.data_ptr(), gx, sx, bx, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap ymap = enc(3, y.data_ptr(), gx, sx, bx, CU_TENSOR_MAP_SWIZZLE_128B, "y");
  uint64_t gw[2] = {(uint64_t)D, (uint64_t)HC}, sw[1] = {(uint64_t)D * 2};
  uint32_t bw[2] = {D, HC};
  CUtensorMap wmap = enc(2, wv.data_ptr(), gw, sw, bw, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  uint64_t gv[3] = {(uint64_t)C, (uint64_t)S, (uint64_t)H * N}, sv[2] = {(uint64_t)C * 2, (uint64_t)S * C * 2};
  uint32_t bv[3] = {C, BSR, 1};
  CUtensorMap vmap = enc(3, v.data_ptr(), gv, sv, bv, CU_TENSOR_MAP_SWIZZLE_64B, "v");
  const int ntile = (N / (int)ntok) * ((S + BSR - 1) / BSR);
  switch (nst * 10 + ntok) {
    case 21: launch_nst<2, 1>(lnw, lnb, eps, N, S, ntile, blocks_per_sm, xmap, ymap, wmap, vmap); break;
    case 31: launch_nst<3, 1>(lnw, lnb, eps, N, S, ntile, blocks_per_sm, xmap, ymap, wmap, vmap); break;
    case 41: launch_nst<4, 1>(lnw, lnb, eps, N, S, ntile, blocks_per_sm, xmap, ymap, wmap, vmap); break;
    case 22: launch_nst<2, 2>(lnw, lnb, eps, N, S, ntile, blocks_per_sm, xmap, ymap, wmap, vmap); break;
    case 32: launch_nst<3, 2>(lnw, lnb, eps, N, S, ntile, blocks_per_sm, xmap, ymap, wmap, vmap); break;
    case 42: launch_nst<4, 2>(lnw, lnb, eps, N, S, ntile, blocks_per_sm, xmap, ymap, wmap, vmap); break;
    default: TORCH_CHECK(false, "nst must be 2, 3 or 4 and ntok 1 or 2");
  }
  return {v, y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("ln_vg", &ln_vg, "LayerNorm + value projection: (v head-major [H,N,S*C], y [S,N,64])",
          py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps") = 1e-5, py::arg("blocks_per_sm") = 2, py::arg("nst") = 3, py::arg("ntok") = 1);
}
